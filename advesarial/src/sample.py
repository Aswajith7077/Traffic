import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from agents import ActorCritic
from config import config
from environment import Environment
from schema import TraciConfig, TransformerEncoderConfig
from services import TraciService

from models import GATLayer, LocalEncoder, SubGoalGenerator, TransformerEncoder

# Sub Policy
local_encoder = LocalEncoder()
GAT = GATLayer(feature_dim=64)
# State = concat(z_i, mean local features, subgoal) -> 64 + 64 + 2*len(tls_set)
actor_critic = ActorCritic(state_dimension=144, action_dimension=7)

traci_config = TraciConfig(config_path="../scenarios/cologne8/cologne8.sumocfg")
traci_service = TraciService(traci_config)
traci_service.start_simulation()

environment = Environment(traci_service=traci_service)

adjacency_list = traci_service.get_adjacency_list()

running_mean = torch.zeros(10)
running_var = torch.ones(10)
count = 1e-4
alpha = 0.01

gamma = 0.99

eta1 = 0.1
eta2 = 0.1

# HiLight-style PPO hyperparameters
clip_epsilon = 0.2
entropy_coeff = 0.01
value_coeff = 1.0
gae_lambda = 0.95
ppo_epochs = 10
minibatch_timesteps = 60
control_interval = 10
steps_per_episode = 3600 // control_interval
eval_interval = 25

meta_losses = []
ac_losses = []
sub_losses = []
rewards_history = []

raw_clusters = config.clusters
tls_set = set(traci_service.get_all_intersections())

global_w_mean = torch.zeros(len(tls_set))
global_w_var = torch.ones(len(tls_set))
global_q_mean = torch.zeros(len(tls_set))
global_q_var = torch.ones(len(tls_set))

clusters = {}

for cid, nodes in raw_clusters.items():
    filtered = [n for n in nodes if n in tls_set]
    if len(filtered) > 0:
        clusters[cid] = filtered

# Meta Policy
m = len(clusters)
transformer_encoder_config = TransformerEncoderConfig(
    d_model=128, nhead=8, num_layers=6
)
transformer_encoder = TransformerEncoder(transformer_encoder_config)

subgoal_generator = SubGoalGenerator(d_reg=128, d_hidden=128, M=m, d_g=2 * len(tls_set))


def __normalize_states(states):
    global running_mean, running_var, count, alpha

    batch_mean = states.mean(dim=0)
    batch_var = states.var(dim=0, unbiased=False)

    running_mean = (1 - alpha) * running_mean + alpha * batch_mean
    running_var = (1 - alpha) * running_var + alpha * batch_var

    states = (states - running_mean) / (torch.sqrt(running_var) + 1e-8)

    return torch.clamp(states, -5, 5)


def _encode_sub_policy_state(observations, sub_goal_vector):
    """
    Encode local observations and fuse the meta-policy subgoal into the
    sub-policy state, matching HiLight's pi_L(a_i | o_i, a_H) coupling.
    Returns (num_intersections, 144) = (z_i, mean local, subgoal).
    """

    hidden_state = local_encoder(observations)
    local_features = GAT(hidden_state, adjacency_list)

    mean_feat = torch.mean(local_features, dim=0)
    final_state = []
    for z_i in local_features:
        final_state.append(
            torch.cat([z_i, mean_feat, sub_goal_vector[0]], dim=-1)
        )

    return torch.stack(final_state, dim=0)


def _find_global_observation_from(cluster_states):
    """
    Meta-policy forward pass producing the global subgoal vector (1, d_g).
    """

    global_encoding, local_encoding = transformer_encoder(cluster_states.unsqueeze(0))
    subgoal_vector = subgoal_generator(local_encoding, global_encoding)

    return subgoal_vector


def collect_episode():
    """
    Roll out one full 3600s episode (360 decisions at a 10s control
    interval), storing detached inputs needed for PPO updates.
    """

    obs_hist = []
    cluster_hist = []
    subgoals = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    targets = []
    meta_rewards = []

    global global_w_mean, global_w_var, global_q_mean, global_q_var

    for step in range(steps_per_episode):
        cluster_states = traci_service.get_cluster_states(clusters)
        sub_goal_vector = _find_global_observation_from(cluster_states).detach()

        observations = traci_service.get_observations()
        observations = __normalize_states(observations)

        final_state = _encode_sub_policy_state(observations, sub_goal_vector)

        action_prob, state_values = actor_critic(final_state)
        dist = torch.distributions.Categorical(action_prob)
        action = dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)

        w_global, q_global = traci_service.compute_global_state_now()
        rg = environment.compute_global_reward()

        # Normalize raw W/Q targets with running statistics so the meta MSE
        # loss stays O(1) instead of exploding on raw counts/thousands.
        w_norm = (w_global - global_w_mean) / (torch.sqrt(global_w_var) + 1e-6)
        q_norm = (q_global - global_q_mean) / (torch.sqrt(global_q_var) + 1e-6)

        global_w_mean = (1 - alpha) * global_w_mean + alpha * w_global
        global_w_var = (1 - alpha) * global_w_var + alpha * (w_global - global_w_mean) ** 2
        global_q_mean = (1 - alpha) * global_q_mean + alpha * q_global
        global_q_var = (1 - alpha) * global_q_var + alpha * (q_global - global_q_mean) ** 2

        _, reward, done = environment.step(action, needs_obs=False)

        obs_hist.append(observations.detach())
        cluster_hist.append(cluster_states.detach())
        subgoals.append(sub_goal_vector.detach())
        actions.append(action.detach())
        log_probs.append(log_prob.detach())
        values.append(state_values.detach())
        rewards.append(reward)
        dones.append(done)
        targets.append(torch.cat([w_norm, q_norm], dim=-1).detach())
        meta_rewards.append(rg.item() if isinstance(rg, torch.Tensor) else rg)

        if done:
            break

    return (
        obs_hist, cluster_hist, subgoals, actions, log_probs,
        values, rewards, dones, targets, meta_rewards,
    )


def compute_gae(rewards, values, dones):
    """
    Generalized Advantage Estimation over the episode. Rewards are shared
    globally; per-intersection values give per-agent advantages.
    """

    T = len(rewards)
    advantages = []
    gae = None

    for t in reversed(range(T)):
        v = values[t]
        next_v = values[t + 1] if t + 1 < T else torch.zeros_like(v)
        mask = 1.0 if dones[t] else 0.0

        delta = rewards[t] + gamma * next_v * (1 - mask) - v
        gae = delta if gae is None else delta + gamma * gae_lambda * (1 - mask) * gae
        advantages.append(gae)

    advantages.reverse()
    returns = [g + v for g, v in zip(advantages, values)]

    return advantages, returns


def ppo_update(obs_hist, subgoals, actions, old_log_probs, advantages, returns):
    """
    HiLight-style PPO update for the sub-policy (local encoder -> GAT ->
    actor-critic): clipped surrogate objective, entropy bonus and value
    loss. Minibatched: each minibatch covers a disjoint set of timesteps,
    and those timesteps' states are re-encoded fresh so gradients reach the
    encoder/GAT without reusing stale graphs (a timestep appears in exactly
    one minibatch per epoch).
    """

    n_tls = obs_hist[0].shape[0]
    T = len(obs_hist)

    a_flat = torch.cat(actions, dim=0).squeeze(-1)
    old_lp_flat = torch.cat(old_log_probs, dim=0).squeeze(-1)
    adv_flat = torch.cat(advantages, dim=0).squeeze(-1)
    ret_flat = torch.cat(returns, dim=0).squeeze(-1)

    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    def build_states_for(t_idxs):
        states = []
        for t in t_idxs:
            hidden_state = local_encoder(obs_hist[t])
            local_features = GAT(hidden_state, adjacency_list)
            mean_feat = torch.mean(local_features, dim=0).detach()
            sg = subgoals[t]
            states.append(
                torch.cat(
                    [local_features, mean_feat.expand(n_tls, -1), sg.expand(n_tls, -1)],
                    dim=-1,
                )
            )
        return torch.cat(states, dim=0)

    def sample_indices(t_idxs):
        return torch.tensor(
            [t * n_tls + i for t in t_idxs for i in range(n_tls)], dtype=torch.long
        )

    policy_losses = []
    value_losses = []
    ratio_devs = []
    entropies = []
    actor_grad_norms = []

    for _ in range(ppo_epochs):
        perm = torch.randperm(T)
        for start in range(0, T, minibatch_timesteps):
            t_idxs = perm[start : start + minibatch_timesteps].tolist()
            idx = sample_indices(t_idxs)
            states = build_states_for(t_idxs)

            action_probs, state_values = actor_critic(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs_now = dist.log_prob(a_flat[idx])
            ratio = (log_probs_now - old_lp_flat[idx]).exp()
            ratio_devs.append((ratio - 1.0).abs().mean().item())

            surr1 = ratio * adv_flat[idx]
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_flat[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -dist.entropy().mean()
            entropies.append(dist.entropy().mean().item())
            value_loss = F.mse_loss(state_values.squeeze(-1), ret_flat[idx])

            total_loss = (
                policy_loss
                + entropy_coeff * entropy_loss
                + value_coeff * value_loss
            )

            actor_optimizer.zero_grad()
            local_optimizer.zero_grad()
            gat_optimizer.zero_grad()

            total_loss.backward()
            actor_grad_norms.append(
                torch.nn.utils.clip_grad_norm_(
                    actor_critic.parameters(), max_norm=10.0
                ).item()
            )
            torch.nn.utils.clip_grad_norm_(local_encoder.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(GAT.parameters(), max_norm=10.0)

            actor_optimizer.step()
            local_optimizer.step()
            gat_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

    print(
        f"PPO diag: mean|ratio-1| {sum(ratio_devs) / len(ratio_devs):.4f}, "
        f"entropy {sum(entropies) / len(entropies):.4f}, "
        f"actor grad {sum(actor_grad_norms) / len(actor_grad_norms):.4f}"
    )

    return policy_losses, value_losses


def meta_update(cluster_hist, targets, meta_rewards):
    """
    Adversarial meta-policy update (core idea kept): the subgoal is trained
    to predict the actual global traffic state (normalized W/Q) while an
    eta1 * r_g term pushes toward more ambitious goals. Plus the goal
    alignment term eta2 * alignment_loss. Vectorized over the episode.
    """

    cluster_states = torch.stack(cluster_hist, dim=0)  # (T, M, 10)
    target_tensor = torch.stack(targets, dim=0)        # (T, 16)
    rg = torch.tensor(meta_rewards, dtype=torch.float32)

    global_encoding, local_encoding = transformer_encoder(cluster_states)
    subgoals = subgoal_generator(local_encoding, global_encoding)  # (T, 16)

    G_w, G_q = torch.chunk(subgoals, 2, dim=-1)
    W_target, Q_target = torch.chunk(target_tensor, 2, dim=-1)

    meta_loss = F.mse_loss(subgoals, target_tensor) + eta1 * rg.mean()
    align_loss = (
        environment.beta1 * (W_target - G_w).pow(2).mean()
        + environment.beta2 * (Q_target - G_q).pow(2).mean()
    )
    total_loss = meta_loss + eta2 * align_loss

    transformer_optimizer.zero_grad()
    subgoal_optimizer.zero_grad()

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), max_norm=10.0)
    torch.nn.utils.clip_grad_norm_(subgoal_generator.parameters(), max_norm=10.0)

    transformer_optimizer.step()
    subgoal_optimizer.step()

    return total_loss.item(), meta_loss.item(), align_loss.item()


transformer_optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=3e-4)
subgoal_optimizer = torch.optim.Adam(subgoal_generator.parameters(), lr=3e-4)
local_optimizer = torch.optim.Adam(local_encoder.parameters(), lr=3e-4)
gat_optimizer = torch.optim.Adam(GAT.parameters(), lr=3e-4)
actor_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)


def save_models():
    """Save all trained models with timestamp"""
    # Create models directory if it doesn't exist
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)

    # Generate timestamp for model directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{models_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # Save all model components
    torch.save(transformer_encoder.state_dict(), f"{run_dir}/transformer_encoder.pth")
    torch.save(subgoal_generator.state_dict(), f"{run_dir}/subgoal_generator.pth")
    torch.save(local_encoder.state_dict(), f"{run_dir}/local_encoder.pth")
    torch.save(GAT.state_dict(), f"{run_dir}/gat.pth")
    torch.save(actor_critic.state_dict(), f"{run_dir}/actor_critic.pth")

    # Save optimizers and env configs
    torch.save(
        {
            "transformer_optimizer": transformer_optimizer.state_dict(),
            "subgoal_optimizer": subgoal_optimizer.state_dict(),
            "local_optimizer": local_optimizer.state_dict(),
            "gat_optimizer": gat_optimizer.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "timestamp": timestamp,
            "beta1": environment.beta1,
            "beta2": environment.beta2,
        },
        f"{run_dir}/training_state.pth",
    )

    print(f"Models saved successfully to {run_dir}/")

    return run_dir

    # Generate visual training plots
    viz_dir = "../visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    plt.figure()
    plt.plot(meta_losses, label="Meta Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Meta Policy Loss Curve")
    plt.legend()
    plt.savefig(f"{viz_dir}/meta_loss_{timestamp}.png")
    plt.close()

    plt.figure()
    plt.plot(ac_losses, label="Actor-Critic Loss")
    plt.plot(sub_losses, label="Subgoal Alignment Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Sub-Policy Loss Curves")
    plt.legend()
    plt.savefig(f"{viz_dir}/subpolicy_loss_{timestamp}.png")
    plt.close()

    plt.figure()
    plt.plot(rewards_history, label="Global Rewards", color="green")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("Training Global Reward Curve")
    plt.legend()
    plt.savefig(f"{viz_dir}/rewards_{timestamp}.png")
    plt.close()

    print(f"Training charts saved to {viz_dir}/")

    with open("../metrics.txt", "a") as f:
        f.write(f"--- Training Snapshot: {timestamp} ---\n")
        f.write(f"Steps taken: {len(meta_losses)}\n")
        if meta_losses:
            f.write(f"Latest Meta Loss: {meta_losses[-1]:.4f}\n")
            f.write(f"Latest AC Loss: {ac_losses[-1]:.4f}\n")
            f.write(f"Latest Sub Loss: {sub_losses[-1]:.4f}\n")
            f.write(f"Latest Reward: {rewards_history[-1]:.4f}\n")
        f.write("\n")


def load_models(model_path):
    """Load trained models from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Load model states
    transformer_encoder.load_state_dict(
        torch.load(
            f"{model_path}/transformer_encoder.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    subgoal_generator.load_state_dict(
        torch.load(
            f"{model_path}/subgoal_generator.pth", map_location="cpu", weights_only=True
        )
    )
    local_encoder.load_state_dict(
        torch.load(
            f"{model_path}/local_encoder.pth", map_location="cpu", weights_only=True
        )
    )
    GAT.load_state_dict(
        torch.load(f"{model_path}/gat.pth", map_location="cpu", weights_only=True)
    )
    actor_critic.load_state_dict(
        torch.load(
            f"{model_path}/actor_critic.pth", map_location="cpu", weights_only=True
        )
    )

    training_state = torch.load(
        f"{model_path}/training_state.pth", map_location="cpu", weights_only=False
    )

    # Load optimizer states
    transformer_optimizer.load_state_dict(training_state["transformer_optimizer"])
    subgoal_optimizer.load_state_dict(training_state["subgoal_optimizer"])
    local_optimizer.load_state_dict(training_state["local_optimizer"])
    gat_optimizer.load_state_dict(training_state["gat_optimizer"])
    if "actor_optimizer" in training_state:
        actor_optimizer.load_state_dict(training_state["actor_optimizer"])

    # Load hyperparameters
    environment.beta1 = training_state["beta1"]
    environment.beta2 = training_state["beta2"]

    print(f"Models loaded successfully from {model_path}/")
    print(f"Model timestamp: {training_state.get('timestamp', 'unknown')}")

    return training_state


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Traffic Control Model (HiLight-style PPO)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Number of full-hour (3600s) training episodes",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Model directory to resume training from (loads weights + optimizers)",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy bonus weight (lower = faster specialization)",
    )
    parser.add_argument(
        "--min-green",
        type=int,
        default=1,
        help="Minimum green duration in control calls (1 call = 10s sim)",
    )
    parser.add_argument(
        "--raw-reward",
        action="store_true",
        help="Use unnormalized rewards (preserves per-step traffic signal on sparse demand)",
    )
    parser.add_argument(
        "--reward-alpha",
        type=float,
        default=0.01,
        help="Reward normalization time constant (lower = slower-adapting stats, keeps ramp signal)",
    )
    args = parser.parse_args()

    global entropy_coeff
    entropy_coeff = args.entropy_coeff

    if args.min_green != traci_config.min_green_time:
        traci_config.min_green_time = args.min_green
        traci_service.min_green_time = args.min_green

    environment.normalize_rewards = not args.raw_reward
    environment.reward_alpha = args.reward_alpha

    if args.resume:
        load_models(args.resume)

    try:
        for episode in range(args.episodes):
            print(f"\n=== Episode {episode + 1}/{args.episodes} (3600s) ===")

            rollout = collect_episode()
            (
                obs_hist, cluster_hist, subgoals, actions, log_probs,
                values, rewards, dones, targets, meta_rewards,
            ) = rollout

            advantages, returns = compute_gae(rewards, values, dones)
            policy_losses, value_losses = ppo_update(
                obs_hist, subgoals, actions, log_probs, advantages, returns
            )
            total_meta, meta_loss, align_loss = meta_update(
                cluster_hist, targets, meta_rewards
            )

            episode_reward = sum(rewards) / len(rewards)
            meta_losses.append(meta_loss)
            ac_losses.append(sum(policy_losses) / len(policy_losses))
            sub_losses.append(align_loss)
            rewards_history.append(episode_reward)

            print(
                f"Episode {episode + 1} done: mean reward {episode_reward:.4f}, "
                f"meta {meta_loss:.4f}, policy {ac_losses[-1]:.4f}, "
                f"value {sum(value_losses) / len(value_losses):.4f}"
            )
            saved_run_dir = save_models()

            if (episode + 1) % eval_interval == 0:
                print(f"\n=== Auto-eval at episode {episode + 1} ===")
                traci_service.close_simulation()
                from evaluate import evaluate_models

                evaluate_models(saved_run_dir, steps=360, min_green=traci_service.min_green_time)
                traci_service.start_simulation()

            if episode + 1 < args.episodes:
                environment.restart()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models...")
        save_models()
    except Exception as e:
        print(f"\nTraining error: {e}. Saving models...")
        save_models()
        raise
    finally:
        # Final save
        save_models()
        print("Training completed. Final models saved.")


if __name__ == "__main__":
    main()
