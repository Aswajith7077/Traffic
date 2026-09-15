import torch
import torch.nn.functional as F


def compute_meta_loss(goal, w_global, q_global, r_g, eta1):
    """L_Meta = ||G^t - (W_global, Q_global)||^2 + eta1 * r_g^t, Section 4.3.2.

    goal: (1, 2) -> (G_w, G_q). w_global, q_global, r_g: scalars for the
    current timestep.
    """

    target = torch.stack([w_global, q_global]).reshape(1, 2).detach()
    mse = F.mse_loss(goal, target)

    if not isinstance(r_g, torch.Tensor):
        r_g = torch.tensor(r_g, dtype=torch.float32)

    return mse + eta1 * r_g


def compute_sub_loss(
    new_log_probs,
    entropy,
    values,
    old_log_probs,
    advantages,
    returns,
    goals,
    w_globals,
    q_globals,
    beta_q,
    beta_w,
    eta2,
    clip_eps=0.2,
    entropy_coef=0.01,
    value_coef=1.0,
):
    """L_Sub = L_AC (PPO-clipped actor-critic loss) + eta2 * goal-alignment, Section 4.3.2.

    new_log_probs, values, old_log_probs, advantages, returns: (B,), flattened
    over (rollout_length * N_agents).
    goals: (T, 2) -> (G_w, G_q) per rollout timestep (T, not B — a
    network-level quantity, not per-agent, detached from the Meta-Policy so
    this loss never backprops into it).
    w_globals, q_globals: (T,) actual global state per rollout timestep.
    """

    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)

    l_ac = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

    g_w = goals[:, 0]
    g_q = goals[:, 1]
    alignment_loss = (beta_q * (q_globals - g_q) + beta_w * (w_globals - g_w)).mean()

    total_loss = l_ac + eta2 * alignment_loss

    components = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
        "alignment_loss": alignment_loss.item(),
        "l_ac": l_ac.item(),
    }

    return total_loss, components
