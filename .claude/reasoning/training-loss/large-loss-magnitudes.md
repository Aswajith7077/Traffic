# Loss values as large, scenario-dependent integers

## Root cause: the Meta-Policy's target was a network-wide sum, not a mean

`TraciService.compute_global_state_now()` (`advesarial/src/services/traci.py`)
returns a per-intersection vector of raw waiting time (seconds, summed over
every vehicle on every controlled lane) and queue length (vehicle count).
That's intentional and correct — the comment there ("No normalization to
preserve information... use raw values for better state representation")
is about keeping per-intersection fidelity for the state representation.

The bug was one level up, in how the two callers turned that per-intersection
vector into a single global scalar:

- `Environment.compute_goal_reward()` — `W_global = W_global.sum()`
- `Trainer._step()` — `w_global_sum = w_global_t.sum()`

Summing (instead of averaging) across intersections means the target the
Meta-Policy's goal output has to match scales linearly with **how many
intersections the scenario has** — and that varies a lot across the five
scenarios in the comparison table:

| Scenario | Signalized intersections |
|---|---|
| cologne8 | 8 |
| grid4x4 / arterial4x4 | 16 |
| manhattan | ~72 regions / many more raw TLS |
| ingolstadt21 | 21 |

`compute_meta_loss` is `MSE(goal, [W_global, Q_global])` where `goal` is a
small, roughly unit-scale network output (`SubGoalGenerator`'s LSTM head).
Regressing a ~O(1) prediction against a target that is a sum over up to
dozens of intersections' accumulated waiting seconds puts the squared error
at `O(target^2)` — for cologne8 that might be hundreds to low thousands;
for a 21+ intersection network, the same congestion level produces a target
2-3x larger, so the *squared* error (and thus the loss) is 4-9x larger for
no modeling reason, purely from network size. The same sum feeds
`alignment_loss` in `compute_sub_loss` (`training/adversarial.py`) via
`w_globals`/`q_globals`, so the sub-policy's logged loss inherits the same
scenario-size-dependent scale.

This is also consistent with the documented instability in AGENTS.md ("AC
loss can explode from ~258 to >129k within 385 steps"): once the raw sum
starts climbing as an episode's congestion builds, the MSE target grows
roughly quadratically in that sum, and nothing in the training loop rescales
it back down — gradient clipping (`MAX_GRAD_NORM=10.0`) bounds how fast the
*parameters* move per update, but it does not bound the *reported loss
value* itself, which can keep climbing as the target keeps climbing.

## Fix applied

Changed both call sites to `.mean()` instead of `.sum()`:

```python
# environment.py: compute_goal_reward()
W_global = W_global.mean()
Q_global = Q_global.mean()

# trainer.py: _step()
w_global_mean = w_global_t.mean()
q_global_mean = q_global_t.mean()
```

This makes `W_global`/`Q_global` an **average per-intersection** congestion
level instead of a network-wide total, so the Meta-Policy's goal target is
on a comparable scale across all five scenarios regardless of how many
intersections each one has. It does not touch `compute_global_state_now()`
itself, so the per-intersection fairness/Jain's-index calculations (which do
need the raw, un-aggregated per-node values) are unaffected.

## Is there further algorithmic headroom here?

Two secondary, lower-priority items worth knowing about but **not** changed
in this pass (kept out of scope to avoid touching more of the PPO update
than the measured problem required):

1. **No value-function clipping.** `compute_sub_loss` uses a plain
   `F.mse_loss(values, returns)` for the critic, rather than the
   clipped-value-loss variant from the original PPO implementation
   (`max((V - returns)^2, (clip(V, V_old-eps, V_old+eps) - returns)^2)`).
   With the two fixes above removing the main source of outlier targets,
   this is lower priority; it would still raise the batch's robustness to a
   single occasional large reward/return outlier if spikes reappear at
   longer training horizons.
2. **`value_coef = 1.0`** gives the value loss equal weight to the policy
   loss. If spikes persist after the scale fix above at longer training
   horizons, lowering `value_coef` (e.g. to 0.5) is the standard first knob
   to try before adding value clipping.

See also [[sharp-spikes]] for the separate (but related) reward-normalizer
variance bug, and [[reward-component-dominance]] for the dominant bug found
afterward: R_eff swamping R_fair/R_emergency by ~3,500x regardless of
w1/w2/w3.
