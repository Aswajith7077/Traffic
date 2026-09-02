import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from agents import ActorCritic
from config import SCENARIO, config
from environment import Environment
from memory import ReplayBuffer
from schema import TraciConfig, TransformerEncoderConfig
from services import TraciService
from utils import compute_ac_loss, compute_goal_alignment_loss, compute_meta_loss

from models import GATLayer, LocalEncoder, SubGoalGenerator, TransformerEncoder

# Sub Policy
local_encoder = LocalEncoder()
GAT = GATLayer(feature_dim=64)
actor_critic = ActorCritic(state_dimension=128, action_dimension=7)

traci_config = TraciConfig(config_path=f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg")
traci_service = TraciService(traci_config)
traci_service.start_simulation()

environment = Environment(traci_service=traci_service)

adjacency_list = traci_service.get_adjacency_list()
buffer = ReplayBuffer()

running_mean = torch.zeros(10)
running_var = torch.ones(10)
count = 1e-4
alpha = 0.01

batch_size = 16

gamma = 0.99

eta1 = 0.1
eta2 = 0.1

EPISODE_STEPS = int(os.environ.get("TRAFFIC_EPISODE_STEPS", "1000"))
TOTAL_EPISODES = int(os.environ.get("TRAFFIC_EPISODES", "10"))
SAVE_EVERY = int(os.environ.get("TRAFFIC_SAVE_EVERY", "10"))

meta_losses = []
ac_losses = []
sub_losses = []
rewards_history = []

raw_clusters = config.clusters
tls_set = set(traci_service.get_all_intersections())

buffer = ReplayBuffer()

clusters = {}

for cid, nodes in raw_clusters.items():
    filtered = [n for n in nodes if n in tls_set]
    if len(filtered) > 0:
        clusters[cid] = filtered

# Meta Policy
m = len(clusters)
transformer_encoder_config = TransformerEncoderConfig(d_model=128, nhead=8, num_layers=6)
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


def _find_local_observations():
    """
    Find local observation for each intersection
    """

    observations = traci_service.get_observations()
    observations = __normalize_states(observations)

    hidden_state = local_encoder(observations)
    local_features = GAT(hidden_state, adjacency_list)

    final_state = []
    for i in range(len(local_features)):
        z_i = local_features[i]
        final_state.append(torch.cat([z_i, torch.mean(local_features, dim=0)], dim=-1))

    return final_state


def _find_global_observation():
    """
    Find global observation for the entire network
    """

    cluster_states = traci_service.get_cluster_states(clusters)
    global_encoding, local_encoding = transformer_encoder(cluster_states.unsqueeze(0))

    subgoal_vector = subgoal_generator(local_encoding, global_encoding)

    return subgoal_vector


def execute():
    subgoal_vector = _find_global_observation()
    final_state = _find_local_observations()
    final_state = torch.stack(final_state, dim=0)

    action_prob, state_values = actor_critic(final_state)
    action = torch.multinomial(action_prob, 1)

    _, reward, done = environment.step(action)

    next_final_state = _find_local_observations()
    next_final_state = torch.stack(next_final_state, dim=0)

    # Ensure tensors are detached correctly for storage to prevent graph leakage
    buffer.add(final_state.detach(), action.detach(), reward, next_final_state.detach(), done)


def sample():

    if len(buffer) < batch_size:
        return

    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = batch
    sub_goal_vector = _find_global_observation()

    w_global, q_global = traci_service.compute_global_state_now()
    rg = environment.compute_global_reward()

    meta_loss = compute_meta_loss(sub_goal_vector, w_global, q_global, rg, eta1)

    transformer_optimizer.zero_grad()
    subgoal_optimizer.zero_grad()

    meta_loss.backward()

    torch.nn.utils.clip_grad_norm_(transformer_encoder.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(subgoal_generator.parameters(), max_norm=0.5)

    transformer_optimizer.step()
    subgoal_optimizer.step()

    ac_loss, _ = compute_ac_loss(actor_critic, states, actions, rewards, next_states, dones, gamma)
    sub_loss = compute_goal_alignment_loss(
        w_global,
        q_global,
        sub_goal_vector.detach(),
        environment.beta1,
        environment.beta2,
    )
    total_loss = ac_loss + eta2 * sub_loss

    local_optimizer.zero_grad()
    gat_optimizer.zero_grad()

    total_loss.backward()
    # Gradient clipping on Sub-Policy parameters
    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(local_encoder.parameters(), max_norm=0.5)
    torch.nn.utils.clip_grad_norm_(GAT.parameters(), max_norm=0.5)

    local_optimizer.step()
    gat_optimizer.step()

    meta_losses.append(meta_loss.item())
    ac_losses.append(ac_loss.item())
    sub_losses.append(sub_loss.item())
    rewards_history.append(rg.item() if isinstance(rg, torch.Tensor) else rg)


# def train_step():


#     if len(buffer) < batch_size:
#         return


#     # -------- Sample Batch --------
#     states, actions, rewards, next_states, dones = buffer.sample(batch_size)


#     # -------- META POLICY --------
#     G_t = meta_policy(_find_global_observation())

#     W_global, Q_global = get_global_state_after_k_steps(k=2)

#     r_g = compute_meta_reward(G_t, W_global, Q_global)

#     meta_loss = compute_meta_loss(G_t, W_global, Q_global, r_g, eta1)

#     meta_optimizer.zero_grad()
#     meta_loss.backward()
#     meta_optimizer.step()

#     # -------- SUB POLICY --------
#     sub_loss = compute_sub_loss(
#         actor_critic,
#         batch,
#         G_t.detach(),   # important: stop gradient from meta
#         W_global,
#         Q_global,
#         gamma,
#         eta2,
#         beta_q,
#         beta_w
#     )

#     optimizer.zero_grad()
#     sub_loss.backward()
#     optimizer.step()


# action = torch.argmax(action_prob, dim=-1)

# buffer_item = ReplayBufferItem(
#     state=final_state,
#     action=action,
#     reward=0,
#     next_state=None,
#     done=False
# )
# buffer.push(buffer_item)


# print(traci_service.get_all_intersections())

transformer_optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=5e-5)

subgoal_optimizer = torch.optim.Adam(
    subgoal_generator.parameters(),
    lr=5e-4,  # usually higher for policy learning
)

local_optimizer = torch.optim.Adam(local_encoder.parameters(), lr=5e-5)
gat_optimizer = torch.optim.Adam(GAT.parameters(), lr=5e-5)


def create_run_dir():
    """Create ONE session folder for this training run (no per-checkpoint timestamps).

    Microsecond + random suffix prevents collisions when several training runs
    (e.g. different CLUSTER_METHODs) are launched in the same second.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import random

    run_dir = f"../models/{SCENARIO}/run_{timestamp}_{random.randint(1000, 9999)}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_models(run_dir, episode):
    """Save all trained models as a session checkpoint inside the run folder."""
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = f"{run_dir}/checkpoint_ep{episode:03d}.pth"

    torch.save(
        {
            "transformer_encoder": transformer_encoder.state_dict(),
            "subgoal_generator": subgoal_generator.state_dict(),
            "local_encoder": local_encoder.state_dict(),
            "GAT": GAT.state_dict(),
            "actor_critic": actor_critic.state_dict(),
            "transformer_optimizer": transformer_optimizer.state_dict(),
            "subgoal_optimizer": subgoal_optimizer.state_dict(),
            "local_optimizer": local_optimizer.state_dict(),
            "gat_optimizer": gat_optimizer.state_dict(),
            "episode": episode,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "beta1": environment.beta1,
            "beta2": environment.beta2,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved to {checkpoint_path}")

    # Generate visual training plots per checkpoint
    plots_dir = f"{run_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure()
    plt.plot(meta_losses, label="Meta Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Meta Policy Loss Curve")
    plt.legend()
    plt.savefig(f"{plots_dir}/meta_loss_ep{episode:03d}.png")
    plt.close()

    plt.figure()
    plt.plot(ac_losses, label="Actor-Critic Loss")
    plt.plot(sub_losses, label="Subgoal Alignment Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Sub-Policy Loss Curves")
    plt.legend()
    plt.savefig(f"{plots_dir}/subpolicy_loss_ep{episode:03d}.png")
    plt.close()

    plt.figure()
    plt.plot(rewards_history, label="Global Rewards", color="green")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("Training Global Reward Curve")
    plt.legend()
    plt.savefig(f"{plots_dir}/rewards_ep{episode:03d}.png")
    plt.close()

    print(f"Training charts saved to {plots_dir}/")

    with open("../metrics.txt", "a") as f:
        f.write(f"--- Training Snapshot: {SCENARIO} run_{episode} ---\n")
        f.write(f"Cluster Method: {os.environ.get('CLUSTER_METHOD', 'dbscan')}\n")
        f.write(f"Episode: {episode}\n")
        f.write(f"Steps taken: {len(meta_losses)}\n")
        if meta_losses:
            f.write(f"Latest Meta Loss: {meta_losses[-1]:.4f}\n")
            f.write(f"Latest AC Loss: {ac_losses[-1]:.4f}\n")
            f.write(f"Latest Sub Loss: {sub_losses[-1]:.4f}\n")
            f.write(f"Latest Reward: {rewards_history[-1]:.4f}\n")
        f.write("\n")


def latest_checkpoint(model_dir):
    """Return the path of the highest-indexed checkpoint in a run folder."""
    checkpoints = sorted(glob.glob(f"{model_dir}/checkpoint_ep*.pth"))
    return checkpoints[-1] if checkpoints else None


def load_models(model_path, episode=None):
    """Load trained models, optionally at a specific episode checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    if episode is not None:
        checkpoint_path = f"{model_path}/checkpoint_ep{episode:03d}.pth"
    else:
        checkpoint_path = latest_checkpoint(model_path)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found in {model_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    transformer_encoder.load_state_dict(checkpoint["transformer_encoder"])
    subgoal_generator.load_state_dict(checkpoint["subgoal_generator"])
    local_encoder.load_state_dict(checkpoint["local_encoder"])
    GAT.load_state_dict(checkpoint["GAT"])
    actor_critic.load_state_dict(checkpoint["actor_critic"])

    transformer_optimizer.load_state_dict(checkpoint["transformer_optimizer"])
    subgoal_optimizer.load_state_dict(checkpoint["subgoal_optimizer"])
    local_optimizer.load_state_dict(checkpoint["local_optimizer"])
    gat_optimizer.load_state_dict(checkpoint["gat_optimizer"])

    environment.beta1 = checkpoint["beta1"]
    environment.beta2 = checkpoint["beta2"]

    print(f"Models loaded successfully from {checkpoint_path}")
    print(f"Checkpoint episode: {checkpoint.get('episode', 'unknown')}")

    return checkpoint


def reset_episode():
    """Start a fresh 1000s episode: restart SUMO, reset env state, clear buffer."""
    traci_service.reset_simulation()
    environment.reset()
    buffer.clear()


def main():
    run_dir = create_run_dir()
    current_ep = 0
    try:
        for ep in range(TOTAL_EPISODES):
            current_ep = ep + 1
            reset_episode()
            print(f"Episode {current_ep}/{TOTAL_EPISODES}")

            for t in range(EPISODE_STEPS):
                execute()
                sample()

            if current_ep % SAVE_EVERY == 0 or current_ep == TOTAL_EPISODES:
                save_models(run_dir, current_ep)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models...")
        save_models(run_dir, current_ep)
    except Exception as e:
        print(f"\nTraining error: {e}. Saving models...")
        save_models(run_dir, current_ep)
        raise
    finally:
        save_models(run_dir, current_ep)
        print("Training completed. Final models saved.")


if __name__ == "__main__":
    main()
