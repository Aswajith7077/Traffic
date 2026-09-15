# HiLight Spec Alignment Report

**Scope:** Compares the pasted HiLight implementation spec (arXiv:2506.14391v1) against the current state of this repository on branch `clustering-approaches`.
**Bottom line:** the repo implements a *different but related* traffic-signal-control system inspired by the same paper. The prescribed `hilight/` package layout, config system, and PPO/adversarial training algorithm do not exist. Some pieces cover the same ground under different names/shapes; a few pieces exceed the spec; several core algorithmic requirements are not implemented at all.

---

## 0. Update (2026-09-09): paper-fidelity remediation pass

Following this report, a follow-up pass re-read the HiLight paper directly (Section 4,
Table 4/5/6, Appendix A/E/F/G) and closed most of the *paper/algorithm* fidelity gaps below,
by explicit, confirmed scope: **paper fidelity only**, inside `advesarial/src/` — no
`hilight/` rename, no `main.py`/CLI rewrite, no `outputs/`/`saved_models/` contract rebuild
(§1, §2, §6 below are therefore still accurate and unchanged). Verified via unit-level shape
checks plus short smoke-test runs on `cologne8` (train + eval), not full 380k-step training.

Fixed, with detail in §3–§5, §8:
- **GAC is now genuinely concatenation-based** (`models/gat.py`), zero-padded to 4 neighbors, output `5F`.
- **`F_g` is now actually wired into the Sub-Policy's observation** — previously the code
  computed the Meta-Policy's global feature but fed the Sub-Policy `mean(local_features)`
  instead, meaning the hierarchy never influenced action selection. Also discovered and fixed:
  the Transformer was only ever given a single instant (T=1) instead of the T=20 temporal
  window, because no regional-state buffer existed — added `utils/replay_buffer.py`'s
  `RegionalStateBuffer` and rewired the LSTM to run over the time axis per subregion, per
  Section 4.1.2 (previously it ran over the region axis instead).
- **Regional state now matches Appendix A exactly**: `[stop_car_num, waiting_time, centroid_x,
  centroid_y]` (`services/traci.py::get_regional_state`), replacing an unrelated 10-feature
  edge-occupancy blend.
- **Training algorithm replaced**: PPO-clip + GAE (`training/gae.py`, `training/adversarial.py`,
  `training/trainer.py`) with the paper's `L_Meta`/`L_Sub` adversarial losses, replacing
  REINFORCE. Also fixed a related bug found during the rewrite: the old code stored
  already-fused, detached states in its buffer, meaning gradients could never reach
  `local_encoder`/`GAT` even though they had their own optimizers — the new rollout buffer
  stores raw observations so the PPO update can recompute the encoder/GAC fresh each epoch.
- **Reward switched to the paper's exact formula** (local `-(ql+wt+dt+ps-ss)`, now genuinely
  per-agent rather than a team-average, plus the global goal-reward vs. Meta-Policy targets),
  replacing the fairness/emergency/emission composite (per the user's confirmed decision).
- **Hyperparameters aligned to Table 5/6** where unambiguous: transformer layers=3/heads=2/
  ff=165/d_model=4, LSTM layers=4/hidden=256, learning rate 3e-4 (was an ad hoc 5e-5/5e-4
  split), one Adam optimizer per policy level (Meta, Sub) instead of four.
- Bug fixes: `PhaseTracker.get_entropy()` `NameError`, `train.py`'s broken imports, missing
  `models/encoder/__init__.py`, a non-buffer positional-encoding tensor that would have broken
  on GPU.

Explicitly **not** attempted, and why:
- **`observation_dim=66`** — Appendix A gives feature meanings, not a dimension breakdown, and
  the true 66 depends on an undisclosed fixed lane-count assumption. Kept the existing 10-scalar
  per-intersection observation (already covers all 9 Table-4 features in aggregated form), but
  made every downstream dim (`F`, GAC output, actor/critic sizes) derive from the actual obs dim
  rather than hardcoding — so it would scale correctly if obs were later expanded.
- **Critic's "latent plan" branch** (Table 6) is implemented for architectural fidelity but has
  no described loss in the paper text, so it stays unused/undocumented-purpose.
- **PPO epoch count / minibatch size** aren't in the paper's tables — used a standard default
  (4 epochs, full-batch) and said so in code, rather than presenting it as paper-specified.
- `grid5x5` (no data available) and Manhattan's scaffold-only status were out of scope per the
  confirmed decision to focus on algorithm fidelity, not scenario coverage.
- §1, §2, §6 (directory/CLI/config/checkpoint scaffold) — confirmed out of scope for this pass.

---

## 1. Directory layout

| Spec (`hilight/...`) | Actual repo | Verdict |
|---|---|---|
| `main.py` CLI (`--scenarios --pipeline --run --seed --device`) | `pipeline.py` (`--scenario` singular, subcommand `all\|baseline\|cluster\|copy\|train\|eval`, `--episodes --episode-steps --save-every --steps`) | **Renamed & narrower** — no multi-scenario sequential run, no `--pipeline` union, no `--run` resume validation, no `--seed`/`--device` |
| `config/{base,model}.yaml`, `config/scenarios/*.yaml` | none — `advesarial/src/config.py` only resolves cluster-JSON env vars; Pydantic schemas in `advesarial/src/schema/*.py` hold some settings | **Missing.** No reproducible `config_snapshot.yaml`, no single merged-config source of truth |
| `envs/{sumo_env,observation,reward,subregion}.py` | `advesarial/src/environment/{environment,traffic_env}.py` + `region-splitting/` (separate module) | **Present, split across two modules**, structured differently |
| `models/{meta_policy,sub_policy,transformer,graph_attention,networks}.py` | `advesarial/src/models/{lstm,gat,sub_policy}.py`, `advesarial/src/agents/{agent,actor}.py` | **Present, renamed/merged**, but internals diverge (see §3) |
| `training/{trainer,rollout,adversarial}.py` | `advesarial/src/{train,sample}.py` | **Partial** — `train.py` is broken (see §5); `sample.py` is the actual working training loop |
| `evaluation/evaluator.py` | `advesarial/src/evaluate.py` | **Present**, narrower output contract |
| `utils/{logger,config,replay_buffer,checkpoint}.py` | `advesarial/src/memory/{replay_buffer,phase_tracker}.py`, `advesarial/src/utils/{compute_phase_history,loss}.py` | **Partial** — no dedicated config/checkpoint/logger utils |
| `outputs/<scenario>/<run_id>/...`, `saved_models/.../epoch_<N>/` | `models/<scenario>/run_<timestamp>/checkpoint_ep<N>.pth`; metrics in root `metrics.txt`/`collect_metrics.py`; `visualizations/` | **Partial**, different contract (see §6) |

There is no `hilight/` package at all — this is a from-scratch structure, not a renamed version of the spec's tree.

---

## 2. Scenarios

- Spec wants: `cologne8, grid4x4, arterial4x4, grid5x5, ingolstadt21` (+ `manhattan` scaffolded, training skipped by default).
- Repo has: `cologne8, manhattan, ingolstadt21, arterial4x4, grid4x4`.
  - **`grid5x5` does not exist anywhere** in the repo.
  - **`manhattan` is a fully trained primary scenario** — the opposite of the spec's "scaffold only" instruction.
  - **`grid4x4` is explicitly excluded** from training/GUI runs (`instructions.md`: "no traffic lights in this network").

---

## 3. Model architecture — key divergences

**Updated 2026-09-09 — see §0.** Status column reflects the current state.

| Component | Spec | Actual (as of 2026-09-09) | Verdict |
|---|---|---|---|
| Graph attention | **Concatenation** across self + up to 4 neighbors → `z_i ∈ R^{5F}` (explicitly *not* summation — spec calls this out by name, "GAC") | `models/gat.py`: rewritten to concatenate `hi ‖ α·h_j1 ‖ ... ‖ α·h_j4`, zero-padded, output `5F` | ✅ **Fixed** — genuine GAC now |
| Transformer encoder | layers=3, heads=2, hidden=4, ff=165, no layer norm, sinusoidal PE + class token | `schema/encoder_config.py`: `d_model=4, nhead=2, num_layers=3, dim_feedforward=165` (matches); final extra `nn.LayerNorm` removed, but PyTorch's `TransformerEncoderLayer` still keeps its two built-in per-layer norms (no vanilla way to fully remove without a custom block) | ✅ **Fixed** except the still-present per-layer norms (documented, minor) |
| LSTM sub-goal generator | 4 layers, hidden=256 | `models/lstm.py`: `num_layers=4, hidden=256` (Trainer default); now runs over the **T (time) axis** per subregion (was the M/region axis), per Section 4.1.2 | ✅ **Fixed** |
| Sub-Policy shared trunk | `334 → 256 → 128 → 114` | `agents/actor.py`: `fused_dim(54) → 128 → 64 → 32` — scaled down proportionally since this repo's obs dim is 10, not 66 (so `5F+4=54`, not 334) | ✅ **Fixed in structure** (shared trunk + dropout, no final activation); absolute widths intentionally scaled down, documented in code |
| Actor/Critic | Actor `114→57→8`; dual-branch critic (value branch + separate 56-dim latent-plan branch, with partial-feature fusion) | `agents/actor.py`: actor `32→16→8`; critic branch 1 `32→16→concat(16, first 16 of shared)→1`; branch 2 (latent plan) `32→16→16`, unused by any loss (matches paper — the latent-plan branch has no described loss either) | ✅ **Fixed** |
| `F_g` wiring into Sub-Policy | Meta-Policy's current-timestep global Transformer embedding, fed to every agent's fused observation | Previously **not wired at all** — `sample.py` fed `mean(local_features)` instead of `F_g`. Now `training/trainer.py::_encode_step` fuses the real `F_g = global_embedding[-1]` | ✅ **Fixed** — was the single highest-value bug found |
| Regional state → Transformer | `T=20` window of `(M, d_reg=4)` snapshots | Previously the Transformer only ever saw `T=1` (no temporal buffer existed). Added `utils/replay_buffer.py::RegionalStateBuffer` (20-step circular buffer, zero-padded at episode start) | ✅ **Fixed** |
| `agents/agent.py` | (spec has no direct equivalent — assumed to be part of sub_policy orchestration) | still a stub (`pass`) — unused, out of scope (not part of any call path) | Unchanged, harmless dead code |
| `services/lstm.py` / `services/encoders/*` | n/a | still empty/dead-duplicate — unused, out of scope (not on any call path; `models/encoder/*` is the real, now-fixed implementation) | Unchanged, harmless dead code |

---

## 4. Training algorithm — the biggest gap

**Fixed 2026-09-09 — see §0.**

- `training/gae.py` implements standard GAE (`γ=0.99, λ=0.95`).
- `training/adversarial.py::compute_sub_loss` implements the PPO-clipped surrogate objective
  (`clip ε=0.2`) + value loss (coef 1.0) + entropy bonus (coef 0.01), replacing REINFORCE's
  `-(log_prob * advantage).mean()`. `training/trainer.py` clips gradients at norm 10.0
  (was 0.5) and uses one Adam optimizer per policy level (Meta, Sub) at lr 3e-4 (was an
  ad hoc 5e-5/5e-4 split across 4 optimizers), per Table 5.
- `training/adversarial.py::compute_meta_loss` implements `L_Meta = MSE(G,(W,Q)) + η1·r_g`;
  `compute_sub_loss`'s alignment term implements `L_Sub`'s `η2·(β_q(W-G_q)+β_w(W-G_w))` addition
  — the joint Meta/Sub adversarial loss the paper is named for.
- Found and fixed a related bug in the old code while rewriting this: it stored fully
  detached, already-fused states in its buffer before computing losses, so gradients could
  never reach `local_encoder`/`GAT` even though they had their own optimizers stepping every
  update (those optimizer steps were always no-ops). The new `training/rollout.py::RolloutBuffer`
  stores raw observations + `F_g` instead, so the PPO update recomputes
  `local_encoder → GAC → fuse → actor_critic` fresh each epoch with gradients intact.
- `sample.py`/`train.py` are now thin CLI wrappers around `training/trainer.py::Trainer`;
  same env vars and `pipeline.py` call sites as before.
- Not paper-specified and therefore a documented engineering default: PPO epoch count (4)
  and minibatch strategy (full-batch per rollout) — the paper's tables don't give these.

---

## 5. Reward function — now matches the paper

**Fixed 2026-09-09 — see §0**, per the user's confirmed decision to switch to the paper's formula.

`environment/environment.py::_compute_reward` now implements the paper's exact formula:
per-agent local reward `r_i^t = -(ql_i+wt_i+dt_i+ps_i-ss_i)` (via
`services/traci.py::get_intersection_reward`, which also gained the previously-missing
`delay_time` term — a lane-level ideal-vs-actual travel-time proxy, since the paper defines
delay_time per-vehicle-at-completion but gives no per-step formula), plus the global
goal-reward `r_g^t = -(β_q(W_global-G_q)+β_w(Q_global-G_w))` against the Meta-Policy's latest
`(G_w, G_q)`. Also fixed along the way: the local reward is now genuinely **per-agent** (an
`(N,)` tensor using each intersection's own metrics) rather than a team-wide mean broadcast
identically to every agent.

The previous fairness/emergency/emission composite (Jain's-index fairness, envy penalty,
pedestrian-conflict and emergency-vehicle terms) is no longer wired into the trajectory
reward, but its `traci.py` helper methods (`compute_jain_index`,
`get_pedestrian_waiting_times`, `get_emergency_waiting_time`, etc.) were deliberately left in
place, unused, in case that direction is wanted again later.

---

## 6. Checkpointing & output contract

- Checkpoints save every `--save-every` **episodes** (not every epoch as spec requires) as one combined `.pth` file with model + 3 optimizer states (spec wants 2 models + 2 optimizers, saved separately) — `sample.py`.
- No `latest/`/`best/` symlinks, no `keep_last_n` pruning, no `epoch_meta.json`.
- No `config_snapshot.yaml`, `metrics.csv`, or `metrics_summary.json` per spec's per-run contract. Metrics currently land in a root-level `metrics.txt` / via `collect_metrics.py`, not per-run directories.
- `evaluate.py` **does** correctly implement the ATT/ADT formulas from Appendix G (`travel_time = exit − entry`, `delay = travel_time − free_flow_time`, averaged over completed vehicles), but only for a single deterministic run — no multi-episode mean/std, no `att_curve.png`/`adt_curve.png`/`component_rewards.png` plot contract.

---

## 7. Where the repo exceeds the spec

- **Subregion partitioning**: the spec explicitly defers community-detection-based partitioning to future work and only asks for grid-based partitioning now. This repo already ships a full `region-splitting/` module with **Leiden, Louvain, DBSCAN, and four hybrid combinations** (`dbscan_leiden`, `dbscan_louvian`, `louvian_dbscan`, `leiden_dbscan`), with precomputed cluster JSONs per scenario under `advesarial/clusters/`. This is real, working, spec-exceeding effort — it's the most mature part of the repo relative to the paper's ambitions.

---

## 8. Known self-documented issues (from `AGENTS.md`)

- ~~`advesarial/src/train.py` is broken (import crash)~~ — ✅ **Fixed 2026-09-09**: rewritten as
  a thin wrapper around `training/trainer.py::Trainer`, same as `sample.py`.
- ~~`PhaseTracker.get_entropy()` raises `NameError`~~ — ✅ **Fixed 2026-09-09**: added the
  missing `compute_phase_entropy` import.
- `ReplayBufferItem` Pydantic schema is defined but unused/inconsistent with the actual buffer
  implementation. **Unchanged** — out of scope (it was already unused before this pass and
  remains so; the new `RolloutBuffer`/`RegionalStateBuffer` don't use Pydantic schemas either).
- ~~Missing `__init__.py` in a `models/encoder/` path~~ — ✅ **Fixed 2026-09-09**: added
  (Python 3.14 namespace packages meant this wasn't a hard crash, but it's now explicit).
- Early training runs saw actor-critic loss diverge (258 → 129k) before being stabilized via
  normalization fixes. **Historical**, predates this pass; the new PPO loss was smoke-tested
  and produced finite, non-frozen values over multiple rollout/update cycles on `cologne8`
  (see §0), but has not been run long enough to say whether it's stable over a full 380k-step
  training run.
- Newly found during this pass, also fixed: a non-buffer positional-encoding tensor
  (`models/encoder/positional.py`) that would silently break on GPU (`.to(device)` wouldn't
  move it) — now `register_buffer`'d. Not currently exercised since nothing in this codebase
  moves models to GPU yet.

---

## 9. Summary verdict

| Area | Status (as of 2026-09-09) |
|---|---|
| Directory/CLI/config scaffolding | Still missing — out of scope for the 2026-09-09 pass (confirmed: paper fidelity only) |
| Meta-Policy (Transformer+LSTM) | ✅ Fixed — hyperparameters match Table 6; now genuinely processes the T=20 window; `F_g` now reaches the Sub-Policy |
| Sub-Policy graph attention (GAC) | ✅ Fixed — genuine concatenation, `5F` output |
| Actor-Critic architecture | ✅ Fixed — shared trunk + dual-branch critic per Table 6, widths scaled to this repo's smaller obs dim |
| PPO + GAE + adversarial loss | ✅ Fixed — implemented in `training/`, replacing REINFORCE |
| Observation/reward | ✅ Reward fixed to the paper's exact formula, now per-agent. Observation dim intentionally kept at 10 (see §0) rather than reconstructing an under-specified 66 |
| Subregion partitioning | **Exceeds spec** — community detection already done (unchanged) |
| Checkpointing/output contract | Still partial — out of scope for this pass; checkpoint *content* was updated (2 optimizers instead of 4, `M` recorded) but the file/symlink layout is unchanged |
| Evaluation (ATT/ADT) | Formula matches paper (unchanged); output contract still doesn't match spec (out of scope); `evaluate.py` was rewired to the new `Trainer`/PPO architecture and smoke-tested successfully |
| Scenario coverage | Unchanged — out of scope for this pass |

**Net assessment (updated)**: the model architecture and training algorithm now track the
paper closely — GAC, the Transformer+LSTM Meta-Policy with a real temporal window, `F_g`
actually reaching the Sub-Policy, PPO+GAE, the adversarial `L_Meta`/`L_Sub` losses, and the
paper's reward formula are all implemented and smoke-tested (cologne8, train + eval,
multiple PPO/meta update cycles, finite non-frozen losses). What's still true from the
original assessment: this remains a different codebase from the spec's `hilight/` package —
no shared directory/CLI/config/checkpoint layout — and that gap was explicitly left
unaddressed by scope decision, not oversight. The clustering/subregion work continues to be
the one area that exceeds the paper's own ambitions for these five scenarios.
