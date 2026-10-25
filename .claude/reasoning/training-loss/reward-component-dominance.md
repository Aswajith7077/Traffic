# R_eff silently swamps R_fair/R_emergency despite w1/w2/w3

## This is the dominant bug, not the scale/variance ones

[[large-loss-magnitudes]] and [[sharp-spikes]] fixed two real but secondary
issues. Measured directly from a completed Cologne8 training run's logged
per-step reward components (`paper_metrics/raw/cologne8_without_emv_seed0.json`,
pre-fix), before any component-level fix:

| Component | min | max | mean | stdev |
|---|---|---|---|---|
| `r_eff` | -8975.4 | 3.3 | -2486.9 | 1508.8 |
| `r_fair` | -2.29 | 0.00 | -0.72 | 0.37 |
| `r_emv` | -2.50 | 0.00 | -0.01 | 0.09 |

`Environment._compute_local_reward()` combines these as
`w1*r_eff + w2*r_fair + w3*r_emv` with `w1=0.6, w2=0.3, w3=0.1`. Applying the
weights to the numbers above: `0.6*(-2487) = -1492` vs `0.3*(-0.72) = -0.22`
vs `0.1*(-0.01) = -0.001`. **R_fair and R_emergency contribute roughly
0.015% and 0.00007% of the combined signal respectively.** The weights
`w1/w2/w3` are supposed to be "the user-facing knob" for balancing these
objectives (the paper's own Section 6.3 discussion already names this risk:
"an unweighted efficiency term can swamp the fairness/EMV terms unless
deliberately scaled") — but with a ~3,500x native-scale gap between R_eff
and R_fair, no setting of `w2` in `[0, 1]` can restore fairness's intended
influence. This is the direct, mechanical explanation for the Cologne8
evaluation result: Algorithm came out *worse* than both baselines on Jain's
index too, not just raw travel time — the training signal was never
meaningfully optimizing for fairness in the first place, regardless of the
`w2=0.3` setting in `RewardConfig`.

## Root cause #1: an unbounded term inside R_eff itself

`TraciService.get_intersection_reward()` (`advesarial/src/services/traci.py`)
computes a `delay_time` term per controlled lane as
`length / max(mean_speed, 0.1) - length / max_speed`. As a lane's mean speed
approaches the 0.1 m/s floor, `length / mean_speed` blows up unboundedly --
e.g. a single 300m lane at near-zero speed contributes ~3000 "seconds" of
delay from that lane alone, vs. `queue_length`/`waiting_time`/`pressure`
each O(1)-O(10). This single term, present in every near-stalled lane at a
congested intersection, is the direct source of R_eff's -8975 minimum and
its 1508 stdev.

**Fix applied:** capped each lane's delay contribution at 5x its free-flow
time (`min(max(actual_time - ideal_time, 0.0), 5.0 * ideal_time)`) -- still
the harshest per-lane penalty for a genuinely stalled lane, but bounded
instead of diverging as speed -> 0. This alone cut `r_eff`'s range roughly
10x in a smoke test (min -8975 -> -740, stdev 1509 -> 167 over a comparable
rollout), but R_eff still dwarfed R_fair/R_emergency by ~600x -- not enough
on its own.

## Root cause #2: no per-component scale normalization before weighting

Even with the above cap, R_eff's *native units* (summed queue length +
waiting time + pressure across several lanes, in the tens to hundreds) are
inherently much larger than R_fair's (a `[0,1]`-bounded Jain's-index term
plus two small penalty terms, O(1)) and R_emergency's (seconds of a single
emergency vehicle's wait, usually 0 since EMVs are rare events). No constant
cap fixes a fundamental unit mismatch between three differently-defined
quantities -- `w1/w2/w3` can only do their stated job if each raw component
is on a comparable scale *before* the weights are applied.

**Fix applied:** added a per-component running mean/var EMA normalizer
(`Environment._normalize_component()`, same corrected formula as
[[sharp-spikes]]'s reward normalizer, one running `(mean, var)` pair each for
`eff`, `fair`, `emv`), applied to each component individually before
`w1*r_eff_n + w2*r_fair_n + w3*r_emv_n`. `last_reward_components` (used by
`reward_components.png`, Section 6.2's Figure~\ref{fig:reward}) still logs
the *raw*, unnormalized components -- that figure's entire point is showing
the native-scale disparity, so normalizing what gets plotted there would
hide the exact finding this document describes.

## Expected effect

This does not guarantee Algorithm beats the baselines -- that's an empirical
question for the retrained Cologne8 run, not something a reward-shaping fix
can promise (see also the "no edits guarantee optimality" discussion). What
it does fix is a specific, measured bug: fairness and EMV-responsiveness
were, in practice, not being optimized at all despite `w2`/`w3` being set to
nonzero values. Whether that specifically explains the fairness-metric gap
observed in the first Cologne8 evaluation (Algorithm's Jain's index below
both Fixed Time and Max Pressure) is exactly what the retest will show.
