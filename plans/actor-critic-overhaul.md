# Actor-Critic Model Overhaul Plan

## Overview

Detailed plan to fix and optimize the Actor-Critic (AC) implementation used for
adaptive traffic signal control in the `advesarial` module. Training objective:
REINFORCE + entropy bonus + GAE (no PPO). Scope: full overhaul (correctness,
algorithm, engineering, validation).

---

## Part A — Current Implementation Anatomy

### Model definition (`advesarial/src/agents/actor.py`)

A 2-layer MLP: `Linear(state_dim -> 128) -> ReLU` followed by two heads:

- **Actor**: `Linear(128 -> 7) + softmax` -> categorical per-intersection phase distribution
- **Critic**: `Linear(128 -> 1)` -> scalar state-value per intersection

Instantiated at `sample.py:20` as `ActorCritic(128, 7)`. A single AC is shared
across all N intersections (applied node-wise).

### Input feature — how `state_dim=128` is built (`sample.py:86-102`)

```
obs_i (10-dim) -- LocalEncoder(10->64->64, ReLU) --> h_i (64)
h_i -- GATLayer(64, residual zi = hi + sum(alpha*hj)) --> z_i (64)
final_i = [ z_i ; mean_j z_j ]  --> (128,)
```

The AC sees each intersection's local GAT embedding plus the network-wide mean.
The 10-dim observation (`traci.py:308-362`): car_num, queue_length, occupancy,
flow, stop_car_num, waiting_time, avg_speed, pressure, congestion_ratio, delay.

### Action semantics

- Training (`sample.py:124`): `action = torch.multinomial(action_prob, 1)` — stochastic.
- Eval (`evaluate.py:183`): `torch.argmax` — greedy.
- `Environment.step` -> `TraciService.set_phase(tls, act)` maps
  `action % num_valid_phases` to a green phase index (`traci.py:169-171`).

### Training algorithm (`sample.py`, `utils/loss.py`)

REINFORCE-style TD Actor-Critic with 4 optimizers:

| Optimizer | LR | Trains | Loss |
|---|---|---|---|
| `transformer_optimizer` | 5e-5 | Meta policy | `compute_meta_loss` |
| `subgoal_optimizer` | 5e-4 | Subgoal generator | same meta loss |
| `local_optimizer` | 5e-5 | LocalEncoder | `ac_loss + eta2*sub_loss` |
| `gat_optimizer` | 5e-5 | GAT | same sub-policy loss |

`compute_ac_loss` (`loss.py:25-49`):

```
targets   = r + gamma*V(s')*(1-done)
critic    = MSE(V(s), targets.detach())
advantages= (targets - V(s)).detach()
actor     = -mean( log pi(a|s) * advantage )
```

`ReplayBuffer` holds `(state, action, reward, next_state, done)` (cleared each
episode), sampled 16 at a time; `sample()` runs every env step. Meta loss =
`MSE(G_t, [W_global; Q_global]) + eta1*r_g`; subgoal alignment =
`eta2*MSE(G_t, [W_global; Q_global])` folded into the sub-policy loss.

---

## Part B — Critical Defects

### C1 (BLOCKER) — Actor-Critic is never trained
`sample.py:149-180`: gradients are computed and clipped for
`actor_critic.parameters()`, but **no optimizer includes them**. The AC stays
at random init forever; its gradients are never zeroed and **accumulate
unboundedly** across every `backward()` — very likely the "AC loss explodes to
~129k" divergence documented in `AGENTS.md`.

### C2 (HIGH) — Fixed `action_dimension=7` vs. per-intersection valid phases
`set_phase` maps `action % num_phases`. For manhattan many TLS have
`valid_phases=[0]` (all 7 logits -> same phase -> zero learning signal); osm
has 2-3 valid greens. The distribution is over generic slots, not the
intersection's real phase set.

### C3 (HIGH) — Subgoal never reaches the actor
`execute()` computes `subgoal_vector` but `actor_critic(final_state)` never
sees it. The hierarchy is coupled only through the alignment loss, not through
action conditioning — the "meta policy conditions sub-policy" claim is false
in practice.

### C4 (HIGH) — Regression target scale mismatch
`W_global`/`Q_global` are **raw** waiting-times/queue-lengths
(`traci.py:245-275`, thousands of units) used as MSE targets for the 944-dim
subgoal (`d_g = 2*len(tls)`, `sample.py:69`). Unnormalized large-magnitude
regression -> unstable gradients.

### C5 (MED) — Normalization issues
- Obs normalization (`sample.py:72-83`) starts at `running_mean=0, var=1` with
  `alpha=0.01` -> early pull-to-zero; heterogeneous feature scales
  (occupancy 0-1 vs. waiting_time ~1e3).
- Cluster states fed to the Transformer are **not normalized** at all.
- Duplicated running-stats logic (env vs. sample.py).

### C6 (MED) — Off-policy / misaligned sampling
Buffer keeps up to 1000-step-old transitions, but `sub_goal_vector`, `W/Q`,
and `rg` are recomputed at *current* time for alignment (`sample.py:142-145`)
-> targets attached to stale transitions.

### C7 (MED) — `done` is always False
`Environment.max_t=3600 > EPISODE_STEPS=1000` -> TD targets never see a real
terminal; value bootstrap only.

### C8 (MED) — No exploration control / no GAE
No entropy bonus (premature collapse), single-step TD (high bias), no
N-step/GAE, no ratio clipping.

### C9 (LOW) — Misc
Global scalar reward shared by all N intersections (no per-node credit);
`state_dim=128` hardcoded to encoder width; hyperparams scattered as module
globals; config inconsistency (schema defaults `nhead=4, layers=3` vs. actual
`nhead=8, layers=6`); LSTM "subgoal generator" processes M subregions per
timestep, not time (no temporal memory).

---

## Part C — Implementation Plan

### Phase 0 — Correctness (make the AC actually train)

**0.1 Actor optimizer + gradient hygiene** — `sample.py`
- Add `actor_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=<cfg>)`.
- In `sample()`: `actor_optimizer.zero_grad()` alongside local/GAT, and
  `actor_optimizer.step()`.
- Fixes the unbounded gradient accumulation (root of the divergence).

**0.2 Per-intersection action heads (kill `% num_phases`)** —
`agents/actor.py`, `sample.py`
- `ActorCritic` outputs `max_valid_phases` logits
  (e.g. `n_head = max(len(v) for v in valid_phases)`).
- Pass a per-node `(N, n_head)` **mask** into `forward`; apply
  `logits.masked_fill(~mask, -inf)` before softmax. Unused slots are ignored;
  each intersection samples only from its real green phases.
- `set_phase` keeps its mapping but now receives an already-valid index.
- Eval: `argmax` over the masked distribution.

**0.3 Subgoal conditioning (restore the hierarchy)** — `sample.py`, `models/lstm.py`
- Project the 2N-dim subgoal to a small bottleneck (e.g. `Linear(2N -> 64)`)
  and concat broadcast to each node's `final_state` before the AC.
- `ActorCritic` `state_dim` derived from `local_enc_dim*2 + subgoal_dim`, not
  hardcoded `128`.

**0.4 Normalized, aligned regression targets** — `sample.py`, `services/traci.py`
- Run `W/Q` (and cluster states) through the same running-mean/var
  normalization used for obs; store the *normalized* `W/Q` **at collection
  time** into the replay buffer (fixes C6 — no more current-state attached to
  stale transitions).

### Phase 1 — Algorithm (REINFORCE + entropy + GAE)

**1.1 Episodic trajectory learning** — `memory/replay_buffer.py`, `sample.py`
- Switch from per-step random 16-sample TD to **per-episode trajectory**
  storage: one `(states[N,T], actions, rewards, dones, masks, W, Q)` per episode.
- At episode end compute returns/advantages once over the full rollout (no
  mid-episode mixing).

**1.2 GAE + entropy in `compute_ac_loss`** — `utils/loss.py`
- `advantages = GAE(values, next_values, rewards, dones, gamma=0.99, lam=0.95)`.
- `actor = -mean(log pi(a)*A)` (REINFORCE), `critic = MSE(V, returns)`,
  `entropy = mean(H[pi])`.
- `loss = actor + c_v*critic - c_e*entropy` (new coeffs `c_v`, `c_e` in config).

**1.3 Real terminal handling** — `environment/environment.py`, `sample.py`
- Make `max_t == EPISODE_STEPS` (or `done=True` on the final step) so
  bootstrapping is correct.

**1.4 Decision cadence (optional, worth testing)** — `environment.py`
- Align one RL decision per 5s green (now that greens are fixed 5s): step SUMO
  5x per decision, or set `step_length=5`. Reduces redundant actions and
  stabilizes REINFORCE variance.

### Phase 2 — Engineering

**2.1 Vectorized GAT** — `models/gat.py`
- Replace the Python `for i / for j` loops with adjacency-matrix propagation
  (one matmul pass over `S x S` mask), keeping the shared attention param (or
  add per-node key/query projections). Single-head stays.

**2.2 Config-driven hyperparameters** — `schema/train_config.py` (new), `sample.py`
- Move `batch/gamma/lambda/c_v/c_e/eta1/eta2/lrs/EPISODE_STEPS/SAVE_EVERY` out
  of module globals into one `TrainConfig`; fix the transformer config
  mismatch (one source of truth for `d_model/nhead/num_layers`).

**2.3 Unified normalization** — `sample.py`, `environment/environment.py`
- Single `RunningNormalizer` utility shared by obs, cluster states, and W/Q
  (remove the duplicated/initialized-to-0 running stats).

**2.4 Training-time evaluation + logging** — `sample.py`
- Every N episodes: greedy eval on a held-out route (reuse `evaluate.py`
  logic), log AC/actor/critic/entropy/reward to `metrics.txt` + optional
  TensorBoard.

### Phase 3 — Validation

- Short smoke run on `manhattan` (default) and `osm`: assert AC loss **no
  longer diverges** (was ~129k), entropy stays in a healthy band, and
  reward/eval metrics (avg wait, avg queue from `evaluate.py`) improve vs. the
  previous fixed-phase behavior.
- `ruff check . && ruff format .`; live TraCI smoke test on unchanged nets.

---

## Files Touched

- `advesarial/src/agents/actor.py` — dynamic masked action heads, configurable dims
- `advesarial/src/utils/loss.py` — GAE, entropy term, coefficient-weighted loss
- `advesarial/src/memory/replay_buffer.py` — episodic trajectory storage
- `advesarial/src/models/gat.py` — vectorized graph attention
- `advesarial/src/models/lstm.py` — subgoal bottleneck projection
- `advesarial/src/schema/train_config.py` (new) — hyperparameter config
- `advesarial/src/sample.py` — actor optimizer, conditioning, episodic loop, logging
- `advesarial/src/environment/environment.py` — terminal handling, decision cadence
- `advesarial/src/evaluate.py` — masked greedy argmax
