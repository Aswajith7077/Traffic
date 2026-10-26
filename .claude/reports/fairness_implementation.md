# Fairness Reality Check

Does the hierarchical algorithm's slower travel time buy better fairness, the way the paper's framing hopes? Measured on 4 scenarios (Manhattan still running at time of writing), seed 0, Jain's Fairness Index over vehicle waiting times.

Interactive version with charts: https://claude.ai/artifact/2MTuPugBiYJ8gXJbfyJd1f

## Verdict

**No — the opposite.** Algorithm has the *worst* fairness of all three methods in 3 of 4 scenarios, not the best. Max Pressure, a zero-training reactive heuristic, is the most fair controller in every single scenario measured.

There is no scenario where Algorithm trades speed for fairness favorably. On Arterial4x4 it's faster than Fixed Time but less fair. On Grid4x4 and Ingolstadt21 it is both slower *and* less fair than Fixed Time. Only Cologne8 shows Algorithm beating Fixed Time on both axes.

## Headline numbers

- **3 / 4** scenarios where Algorithm has the lowest Jain's Index of all 3 methods
- **4 / 4** scenarios where Max Pressure has the highest Jain's Index
- **−0.20** correlation between travel time and fairness across all 12 runs — weak, and the wrong sign for a trade-off story (faster methods trend slightly *more* fair in this data, not less)
- **1 / 4** scenarios where Algorithm beats Fixed Time on fairness (Cologne8 only)

## Full data

Fairness rank within each scenario (1 = most fair of the 3 methods, 3 = least fair).

| Scenario | Method | ATT [s] | Jain's Index | Starvation (% waiting ≥ W_max) | Fairness rank |
|---|---|---|---|---|---|
| Cologne8 | Fixed Time | 155.90 | 0.7326 | 28% | #3 |
| Cologne8 | Max Pressure | 96.54 | 0.9396 | 0% | #1 |
| Cologne8 | Algorithm | 133.51 | 0.7997 | 20% | #2 |
| Arterial4x4 | Fixed Time | 519.49 | 0.9116 | 100% | #1 |
| Arterial4x4 | Max Pressure | 266.39 | 0.7548 | 48% | #2 |
| Arterial4x4 | Algorithm | 341.62 | 0.6898 | 81% | #3 |
| Grid4x4 | Fixed Time | 232.15 | 0.6237 | 56% | #2 |
| Grid4x4 | Max Pressure | 147.64 | 0.9423 | 0% | #1 |
| Grid4x4 | Algorithm | 266.98 | 0.5229 | 69% | #3 |
| Ingolstadt21 | Fixed Time | 293.30 | 0.6968 | 45% | #2 |
| Ingolstadt21 | Max Pressure | 207.00 | 0.8998 | 2% | #1 |
| Ingolstadt21 | Algorithm | 319.21 | 0.5929 | 54% | #3 |

## Source

`paper_metrics/eval_results.csv`, labels `{scenario}_fixed` / `{scenario}_max_pressure` / `{scenario}_without_emv`, seed 0, 1200-step evaluation after the 5-episode + behavior-cloning training pass described in `.claude/reasoning/training-loss/`. Manhattan's campaign was still running at the time of this report and is not included. Jain's Index here is the network-level index over completed vehicles' waiting times.

See [[fairness-improvement-attempt]] for the follow-up: changes made to the reward weighting to try to close this gap, and whether retraining actually improved it.
