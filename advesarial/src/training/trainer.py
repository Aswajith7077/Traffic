import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from agents import ActorCritic
from config import config
from environment import Environment
from schema import TraciConfig, TransformerEncoderConfig
from services import TraciService
from torch.distributions import Categorical
from utils import RegionalStateBuffer

from models import GATLayer, LocalEncoder, SubGoalGenerator, TransformerEncoder

from .adversarial import compute_meta_loss, compute_sub_loss
from .gae import compute_gae
from .rollout import RolloutBuffer

# Table 5 hyperparameters, plus a couple of PPO-mechanics defaults the paper's
# tables don't specify (ppo_epochs, temporal_window/meta_update_interval come
# from the originally-pasted spec's Table-5-adjacent config, not this table).
OBS_DIM = 10  # this repo's per-intersection observation size (see report: paper's 66 not reconstructed)
D_REG = 4  # regional-state dim, Appendix A
NUM_ACTIONS = 8
TEMPORAL_WINDOW = 20
LSTM_HIDDEN = 256
LSTM_LAYERS = 4
SUBGOAL_DIM = 16
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 1.0
MAX_GRAD_NORM = 10.0
LEARNING_RATE = 3e-4
ROLLOUT_LENGTH = 240
META_UPDATE_INTERVAL = 10
ETA1 = 0.1
ETA2 = 0.1
PPO_EPOCHS = 4  # not paper-specified; a standard PPO default


class Trainer:
    def __init__(self, scenario, sumocfg_path, use_gui=False, delay=0.0):
        self.scenario = scenario

        traci_config = TraciConfig(config_path=sumocfg_path, use_gui=use_gui, delay=delay)
        self.traci_service = TraciService(traci_config)
        self.traci_service.start_simulation()

        self.environment = Environment(traci_service=self.traci_service)
        self.adjacency_list = self.traci_service.get_adjacency_list()
        self.intersections = self.traci_service.get_all_intersections()

        raw_clusters = config.clusters
        tls_set = set(self.intersections)
        self.clusters = {cid: [n for n in nodes if n in tls_set] for cid, nodes in raw_clusters.items()}
        self.clusters = {cid: nodes for cid, nodes in self.clusters.items() if nodes}
        self.M = len(self.clusters)

        self.F = OBS_DIM  # shared-MLP output dim == raw obs dim, per the paper's own worked example
        self.fused_dim = 5 * self.F + D_REG

        # Sub-Policy
        self.local_encoder = LocalEncoder(in_dim=OBS_DIM, hidden_dim=self.F)
        self.gat = GATLayer(feature_dim=self.F)
        self.actor_critic = ActorCritic(state_dimension=self.fused_dim, action_dimension=NUM_ACTIONS)

        # Meta-Policy
        self.transformer_config = TransformerEncoderConfig(d_reg=D_REG)
        self.transformer_encoder = TransformerEncoder(self.transformer_config)
        self.subgoal_generator = SubGoalGenerator(
            d_model=self.transformer_config.d_model,
            d_hidden=LSTM_HIDDEN,
            M=self.M,
            d_g=SUBGOAL_DIM,
            num_layers=LSTM_LAYERS,
        )
        self.regional_buffer = RegionalStateBuffer(window_size=TEMPORAL_WINDOW, num_regions=self.M, d_reg=D_REG)

        self.sub_optimizer = torch.optim.Adam(
            list(self.local_encoder.parameters()) + list(self.gat.parameters()) + list(self.actor_critic.parameters()),
            lr=LEARNING_RATE,
        )
        self.meta_optimizer = torch.optim.Adam(
            list(self.transformer_encoder.parameters()) + list(self.subgoal_generator.parameters()),
            lr=LEARNING_RATE,
        )

        self.rollout_buffer = RolloutBuffer()

        self.running_mean = torch.zeros(OBS_DIM)
        self.running_var = torch.ones(OBS_DIM)
        self.obs_alpha = 0.01

        self.step_count = 0
        self.meta_losses = []
        self.sub_loss_history = []
        self.rewards_history = []

    def _normalize_obs(self, obs):
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)

        self.running_mean = (1 - self.obs_alpha) * self.running_mean + self.obs_alpha * batch_mean
        self.running_var = (1 - self.obs_alpha) * self.running_var + self.obs_alpha * batch_var

        normalized = (obs - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return torch.clamp(normalized, -5, 5)

    def _encode_step(self, obs_t, f_g_t):
        """local_encoder -> GAC -> fuse with F_g. Kept differentiable so the
        PPO update can recompute this fresh and train local_encoder/GAT."""

        h = self.local_encoder(obs_t)  # (N, F)
        z = self.gat(h, self.adjacency_list)  # (N, 5F)
        f_g_b = f_g_t.unsqueeze(0).repeat(z.size(0), 1)  # (N, d_reg)
        return torch.cat([z, f_g_b], dim=1)  # (N, fused_dim)

    def _meta_forward(self):
        regional_state = self.traci_service.get_regional_state(self.clusters)
        self.regional_buffer.push(regional_state)
        window = self.regional_buffer.get_window()  # (T, M, d_reg)

        global_embedding, subregion_embeddings = self.transformer_encoder(window)
        f_g = global_embedding[-1]  # current-timestep global feature, Section 4.1.1
        _, goal = self.subgoal_generator(subregion_embeddings)  # (1, 2) -> (G_w, G_q)

        return f_g, goal

    def _update_meta(self, goal, w_global, q_global, r_g):
        loss = compute_meta_loss(goal, w_global, q_global, r_g, ETA1)

        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.transformer_encoder.parameters()) + list(self.subgoal_generator.parameters()),
            max_norm=MAX_GRAD_NORM,
        )
        self.meta_optimizer.step()

        self.meta_losses.append(loss.item())

    def _update_sub(self):
        if len(self.rollout_buffer) == 0:
            return

        batch = self.rollout_buffer.get()
        obs, f_g = batch["obs"], batch["f_g"]  # (T, N, obs_dim), (T, d_reg)
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        old_values = batch["values"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        goals = batch["goals"]
        w_globals = batch["w_globals"]
        q_globals = batch["q_globals"]

        T = obs.size(0)

        # Bootstrap with the last collected value estimate rather than a true
        # V(s_{T+1}) (not cheaply available mid-loop) — a standard, documented
        # approximation when the next observation isn't already on hand.
        bootstrap_value = torch.zeros_like(old_values[-1]) if dones[-1].any() else old_values[-1]

        advantages, returns = compute_gae(rewards, old_values, dones, bootstrap_value, GAMMA, GAE_LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        components = None
        for _ in range(PPO_EPOCHS):
            new_log_probs, values, entropies = [], [], []
            for t in range(T):
                final_state = self._encode_step(obs[t], f_g[t])
                action_probs, value_t, _ = self.actor_critic(final_state)
                dist = Categorical(action_probs)
                new_log_probs.append(dist.log_prob(actions[t]))
                values.append(value_t.squeeze(-1))
                entropies.append(dist.entropy())

            loss, components = compute_sub_loss(
                torch.cat(new_log_probs),
                torch.cat(entropies),
                torch.cat(values),
                old_log_probs.reshape(-1),
                advantages.reshape(-1),
                returns.reshape(-1),
                goals,
                w_globals,
                q_globals,
                self.environment.beta_q,
                self.environment.beta_w,
                ETA2,
                CLIP_EPS,
                ENTROPY_COEF,
                VALUE_COEF,
            )

            self.sub_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.local_encoder.parameters())
                + list(self.gat.parameters())
                + list(self.actor_critic.parameters()),
                max_norm=MAX_GRAD_NORM,
            )
            self.sub_optimizer.step()

        self.sub_loss_history.append(components)
        self.rollout_buffer.clear()

    def _step(self):
        f_g, goal = self._meta_forward()

        obs = self.traci_service.get_observations()
        obs = self._normalize_obs(obs)

        with torch.no_grad():
            final_state = self._encode_step(obs, f_g)
            action_probs, values, _ = self.actor_critic(final_state)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        goal_detached = goal.detach()
        goal_pair = (goal_detached[0, 0], goal_detached[0, 1])

        _, reward, done = self.environment.step(action, goal=goal_pair)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)

        w_global_t, q_global_t = self.traci_service.compute_global_state_now()
        w_global_sum = w_global_t.sum()
        q_global_sum = q_global_t.sum()
        r_g = self.environment.compute_goal_reward(goal_pair)

        self.rollout_buffer.add(
            obs=obs.detach(),
            f_g=f_g.detach(),
            action=action,
            log_prob=log_prob,
            value=values.squeeze(-1),
            reward=reward.detach(),
            done=torch.full((len(self.intersections),), float(done)),
            goal=goal_detached.squeeze(0),
            w_global=w_global_sum.detach(),
            q_global=q_global_sum.detach(),
        )

        self.step_count += 1
        if self.step_count % META_UPDATE_INTERVAL == 0:
            self._update_meta(goal, w_global_sum, q_global_sum, r_g)

        if len(self.rollout_buffer) >= ROLLOUT_LENGTH or done:
            self._update_sub()

        self.rewards_history.append(reward.mean().item())

        return done

    def reset_episode(self):
        self.traci_service.reset_simulation()
        self.environment.reset()
        self.regional_buffer.reset()
        self.rollout_buffer.clear()

    def run_episode(self, episode_steps):
        self.reset_episode()
        for _ in range(episode_steps):
            done = self._step()
            if done:
                break

        # Flush whatever's left in the rollout buffer even if the episode
        # ended before reaching a full ROLLOUT_LENGTH window, so short
        # episodes (or a short final one) still train the Sub-Policy.
        self._update_sub()

    def create_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random

        run_dir = f"../models/{self.scenario}/run_{timestamp}_{random.randint(1000, 9999)}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_checkpoint(self, run_dir, episode):
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = f"{run_dir}/checkpoint_ep{episode:03d}.pth"

        torch.save(
            {
                "transformer_encoder": self.transformer_encoder.state_dict(),
                "subgoal_generator": self.subgoal_generator.state_dict(),
                "local_encoder": self.local_encoder.state_dict(),
                "GAT": self.gat.state_dict(),
                "actor_critic": self.actor_critic.state_dict(),
                "meta_optimizer": self.meta_optimizer.state_dict(),
                "sub_optimizer": self.sub_optimizer.state_dict(),
                "episode": episode,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "beta_q": self.environment.beta_q,
                "beta_w": self.environment.beta_w,
                "M": self.M,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

        plots_dir = f"{run_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)

        plt.figure()
        plt.plot(self.meta_losses, label="Meta Loss")
        plt.xlabel("Meta update")
        plt.ylabel("Loss")
        plt.title("Meta Policy Loss Curve")
        plt.legend()
        plt.savefig(f"{plots_dir}/meta_loss_ep{episode:03d}.png")
        plt.close()

        if self.sub_loss_history:
            plt.figure()
            plt.plot([c["l_ac"] for c in self.sub_loss_history], label="Actor-Critic (PPO) Loss")
            plt.plot([c["alignment_loss"] for c in self.sub_loss_history], label="Goal Alignment Loss")
            plt.xlabel("PPO update")
            plt.ylabel("Loss")
            plt.title("Sub-Policy Loss Curves")
            plt.legend()
            plt.savefig(f"{plots_dir}/subpolicy_loss_ep{episode:03d}.png")
            plt.close()

        plt.figure()
        plt.plot(self.rewards_history, label="Mean Reward", color="green")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.legend()
        plt.savefig(f"{plots_dir}/rewards_ep{episode:03d}.png")
        plt.close()

        with open("../metrics.txt", "a") as f:
            f.write(f"--- Training Snapshot: {self.scenario} run_{episode} ---\n")
            f.write(f"Cluster Method: {os.environ.get('CLUSTER_METHOD', 'dbscan')}\n")
            f.write(f"Episode: {episode}\n")
            f.write(f"Steps taken: {len(self.rewards_history)}\n")
            if self.meta_losses:
                f.write(f"Latest Meta Loss: {self.meta_losses[-1]:.4f}\n")
            if self.sub_loss_history:
                f.write(f"Latest Sub L_AC: {self.sub_loss_history[-1]['l_ac']:.4f}\n")
                f.write(f"Latest Alignment Loss: {self.sub_loss_history[-1]['alignment_loss']:.4f}\n")
            if self.rewards_history:
                f.write(f"Latest Reward: {self.rewards_history[-1]:.4f}\n")
            f.write("\n")

    def latest_checkpoint(self, model_dir):
        checkpoints = sorted(glob.glob(f"{model_dir}/checkpoint_ep*.pth"))
        return checkpoints[-1] if checkpoints else None

    def load_checkpoint(self, model_dir, episode=None):
        if episode is not None:
            checkpoint_path = f"{model_dir}/checkpoint_ep{episode:03d}.pth"
        else:
            checkpoint_path = self.latest_checkpoint(model_dir)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.transformer_encoder.load_state_dict(checkpoint["transformer_encoder"])
        self.subgoal_generator.load_state_dict(checkpoint["subgoal_generator"])
        self.local_encoder.load_state_dict(checkpoint["local_encoder"])
        self.gat.load_state_dict(checkpoint["GAT"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])

        if "meta_optimizer" in checkpoint:
            self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])
        if "sub_optimizer" in checkpoint:
            self.sub_optimizer.load_state_dict(checkpoint["sub_optimizer"])

        self.environment.beta_q = checkpoint.get("beta_q", self.environment.beta_q)
        self.environment.beta_w = checkpoint.get("beta_w", self.environment.beta_w)

        print(f"Models loaded successfully from {checkpoint_path}")
        return checkpoint

    def train(self, total_episodes, episode_steps, save_every, run_dir):
        current_ep = 0
        try:
            for ep in range(total_episodes):
                current_ep = ep + 1
                print(f"Episode {current_ep}/{total_episodes}")
                self.run_episode(episode_steps)

                if current_ep % save_every == 0 or current_ep == total_episodes:
                    self.save_checkpoint(run_dir, current_ep)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving models...")
            self.save_checkpoint(run_dir, current_ep)
        except Exception as e:
            print(f"\nTraining error: {e}. Saving models...")
            self.save_checkpoint(run_dir, current_ep)
            raise
        finally:
            print("Training completed.")

    def close(self):
        self.traci_service.close_simulation()
