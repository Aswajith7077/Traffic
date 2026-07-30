import torch
import torch.nn.functional as F


def compute_meta_loss(G_t, W_global, Q_global, r_g, eta1):
    """
    G_t: predicted goal (batch, goal_dim)
    W_global, Q_global: actual global state
    r_g: meta reward
    """

    target = torch.cat([W_global, Q_global], dim=-1)
    if target.dim() == 1 and G_t.dim() == 2:
        target = target.unsqueeze(0).expand(G_t.size(0), -1)

    if not isinstance(r_g, torch.Tensor):
        r_g = torch.tensor(r_g, device=G_t.device, dtype=torch.float32)

    mse_loss = F.mse_loss(G_t, target)
    loss = mse_loss + eta1 * r_g.mean()

    return loss


def compute_ac_loss(actor_critic, states, actions, rewards, next_states, dones, gamma):

    action_probs, state_values = actor_critic(states)
    _, next_state_values = actor_critic(next_states)

    state_values = state_values.squeeze(-1)
    next_state_values = next_state_values.squeeze(-1)

    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
    if dones.dim() == 1:
        dones = dones.unsqueeze(1)

    targets = rewards + gamma * next_state_values * (1 - dones)

    critic_loss = F.mse_loss(state_values, targets.detach())

    if actions.dim() == 2:
        actions = actions.unsqueeze(-1)

    log_probs = torch.log(action_probs.gather(-1, actions).squeeze(-1))
    advantages = (targets - state_values).detach()
    actor_loss = -(log_probs * advantages).mean()

    return actor_loss + critic_loss, advantages


def compute_goal_alignment_loss(W_global, Q_global, G_t, beta_q, beta_w):

    # Split goal
    G_w, G_q = torch.chunk(G_t, 2, dim=-1)

    if W_global.dim() == 1 and G_w.dim() == 2:
        W_global = W_global.unsqueeze(0).expand_as(G_w)
    if Q_global.dim() == 1 and G_q.dim() == 2:
        Q_global = Q_global.unsqueeze(0).expand_as(G_q)

    loss_q = beta_q * (Q_global - G_q).pow(2).mean()
    loss_w = beta_w * (W_global - G_w).pow(2).mean()

    return loss_q + loss_w


def compute_sub_loss(actor_critic, batch, G_t, W_global, Q_global, gamma, eta2, beta_q, beta_w):

    states, actions, rewards, next_states, dones = batch

    # Actor-Critic loss
    ac_loss, _ = compute_ac_loss(actor_critic, states, actions, rewards, next_states, dones, gamma)

    # Alignment loss
    align_loss = compute_goal_alignment_loss(W_global, Q_global, G_t, beta_q, beta_w)

    total_loss = ac_loss + eta2 * align_loss

    return total_loss
