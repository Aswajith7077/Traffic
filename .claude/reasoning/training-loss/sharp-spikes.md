# Sharp spikes in training loss and reward curves

## Root cause: the reward normalizer tracks the wrong variance

`Environment._normalize_reward()` (`advesarial/src/environment/environment.py`)
is a running EMA mean/std normalizer applied to the per-agent reward vector
before it's used for GAE/PPO. Before this fix, the variance update was:

```python
batch_mean = reward.mean().item()
self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
self.reward_var  = (1 - alpha) * self.reward_var  + alpha * (batch_mean - self.reward_mean) ** 2
```

This tracks the variance **of the step-to-step batch mean**, not the
variance of the individual per-agent rewards being normalized. Those are
different quantities whenever the N intersections' rewards agree with each
other within a step (small spread *across agents*) while the reward's
absolute scale still drifts *across steps* as congestion builds or clears.
In that regime — which is the normal case for a traffic network, where
neighboring intersections' efficiency/fairness rewards move together — the
old formula systematically **underestimates** the true standard deviation.

Reproduced with a synthetic stream of true std 3.0 (10 agents/step, same
alpha=0.01→0.1 EMA, 2000 steps):

| Formula | Tracked std | True std | Error |
|---|---|---|---|
| Old (variance of batch mean) | 0.886 | 3.0 | **3.4x underestimate** |
| Fixed (combined/parallel variance) | 2.978 | 3.0 | ~1% |

Dividing by a 3-4x-too-small denominator inflates `normalized_reward` by the
same factor. Because `alpha = 0.01` is a slow-moving EMA and
`Environment.reset()` resets `reward_mean`/`reward_var` to `0.0`/`1.0` at the
start of every episode, the normalizer is re-learning this (wrong) estimate
from scratch each episode — so the distortion is largest right after a
reset and right after any sudden shift in traffic pattern (an EMV event, a
surge of arrivals), which is exactly where the spikes were observed.

This reward feeds GAE (`compute_gae`) directly, so an amplified reward
produces amplified advantages and returns, which produces amplified
`value_loss` in `compute_sub_loss` — this is the direct mechanical link from
a reward-normalization bug to a loss spike.

## Fix applied

Replaced the variance update with the standard combined/parallel-variance
decomposition — the within-batch variance (how much the N agents disagree
*this step*) plus the squared drift of the batch mean from the running mean
(how much the average is moving *across steps*):

```python
batch_var = reward.var(unbiased=False).item()
diff = batch_mean - self.reward_mean
self.reward_mean = self.reward_mean + alpha * diff
self.reward_var  = (1 - alpha) * self.reward_var + alpha * (batch_var + diff**2)
```

This mirrors the pattern `Trainer._normalize_obs()` already uses correctly
for observations (`obs.var(dim=0, unbiased=False)` — a real batch variance),
so the reward path is now consistent with the rest of the codebase instead
of being the one place with ad hoc, incorrect statistics.

## What this does *not* explain, and isn't a bug

Average waiting time/queue length at an intersection genuinely does surge
when congestion builds (e.g. an incident, a platoon arrival, an EMV
preemption cascade) — SUMO's `getWaitingTime` is a real, bounded-per-vehicle
quantity, not a numerical artifact. Some real variance in the reward and
loss curves is expected and correct; the fix above removes the *artificial*
amplification from mis-tracked normalization statistics, not the underlying
traffic dynamics. Residual spikes that correlate with a visible congestion
event in the simulation log are signal, not a flaw to chase further.

See also [[large-loss-magnitudes]] for the separate (but related) bug in
how the global state feeding the Meta-Policy's loss was scaled, and
[[reward-component-dominance]] for the dominant bug found afterward: R_eff
swamping R_fair/R_emergency by ~3,500x regardless of w1/w2/w3.
