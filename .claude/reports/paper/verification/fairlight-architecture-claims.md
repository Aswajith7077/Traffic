# Verification: FairLight §3.2–3.5 claims vs. `advesarial`/`region-splitting` code

**Source of the claims:** two page images the user supplied, containing draft text for
"§3 Proposed FairLight Framework" → §3.2 Traffic-Aware Regional Decomposition, §3.3
Hierarchical Meta-Policy, §3.4 Intersection-Level Sub-Policy, §3.5 Safe Signal Switching.

**Important caveat found during verification:** this text does not exist in the repo's
tracked paper source (`papers/paper-draft/main.tex` / `main.pdf`). That file currently only
has Introduction → Related Work → "Multi-Objective Reward Function Design" → Future Works,
and its reward section (`R_total = 4/9 R_eff + 2/9 R_fair + 3/9 R_emergency`) describes a
reward formula the code no longer uses at all (see `HILIGHT_SPEC_ALIGNMENT_REPORT.md` §5 —
it was deliberately replaced by the paper's `r_i = -(ql+wt+dt+ps-ss) + r_g` formula). So the
images are from a different/newer draft not yet committed here. The check below treats the
image text as the claim under test regardless, per the user's request, verified directly
against `advesarial/src/` and `region-splitting/` as they stand on `clustering-approaches`.

Also directly relevant: `HILIGHT_SPEC_ALIGNMENT_REPORT.md` (repo root) already performed a
related, independent audit against the *original* HiLight paper (arXiv:2506.14391) on
2026-09-09 and corroborates several findings below from the opposite direction.

---

## Summary verdict

| # | Claim (image) | Implemented as claimed? |
|---|---|---|
| 1 | Regional state `s_k` is 10-dim: occupancy, mean wait, vehicle count, mean speed, in/out/net-flow, pressure, congestion ratio, phase | ❌ No — actual regional state is 4-dim |
| 2 | Transformer: 6 layers, 8 heads, `d_model=128`, pre-layer-norm, sinusoidal PE | ⚠️ Partial — sinusoidal PE yes; layers/heads/d_model/norm all wrong |
| 3 | LSTM hidden size 128 | ❌ No — hidden size is 256 |
| 4 | Sub-goal `G_t = [G_t^W, G_t^Q] ∈ R^{2N}` | ❌ No — it's 2 scalars total, network-wide, not per-region/intersection |
| 5 | Intersection obs `o_i` 10-dim: vehicles, queue, occupancy, flow, stops, waiting, speed, pressure, congestion, delay | ✅ Yes — matches field-for-field |
| 6 | Observations standardized via EMA, clipped to [-5,5] | ✅ Yes |
| 7 | Local MLP maps obs to 64-dim | ❌ No — maps to 10-dim |
| 8 | GAT: `z_i' = z_i + Σ α_ij z_j` (summation), LeakyReLU(0.2), softmax | ⚠️ Partial — LeakyReLU(0.2)/softmax match; aggregation is concatenation (`5F`-dim), not summation, and explicitly documented as such in code |
| 9 | `s_i = [z_i' ‖ mean_j z_j ‖ G_t] ∈ R^144` | ❌ No — fused state is `[GAT-concat(5F) ‖ F_g(4)] = 54`-dim; uses the Meta-Policy's *global Transformer embedding* `F_g`, not the sub-goal `G_t`; no separate network-mean term |
| 10 | Actor selects 1 of 7 action indices | ⚠️ Close — head has 8 action slots (masked per-intersection to its real phase count) |
| 11 | Safe switching: yellow transition + min-green constraint | ✅ Yes, mechanism exists |
| 12 | "10-s control cadence"; yellow time and min-green both equal one decision step = 10s | ❌ No — there is no 10s decision cadence (every sim step is a decision, `step_length=1.0s`); defaults are yellow=3s, green/min-green=5s |
| 13 | `w_e = αn_e + βw_e + γκ_e n_e`, `(α,β,γ)=(1.0,0.3,2.0)` | ✅ Yes — exact formula and constants |
| 14 | Traffic stats averaged over 1000 simulation steps | ✅ Yes — `compute_weights(max_iter=1000)` default |
| 15 | Leiden partition on cologne8: modularity 0.624 before filtering; after filtering, each of the 8 signalized intersections is its own region | ❌ No — current output is 2 merged clusters (sizes 6 and 2), modularity ≈0.0123; a `merge_singletons` post-process step actively undoes the singleton-per-intersection outcome the claim describes |

---

## Detail

### §3.2 Traffic-Aware Regional Decomposition

**Edge weight — confirmed.** `region-splitting/services/traci.py::get_edge_weight`:

```python
alpha, beta, gamma = 1.0, 0.3, 2.0
weight = alpha * vehicles + beta * waiting + gamma * congestion * vehicles
```

matches `w_e = αn_e + βω_e + γκ_e n_e` with `(α,β,γ)=(1.0,0.3,2.0)` exactly.

**1000-step averaging — confirmed.** `LeidenService.compute_weights(self, max_iter=1000)`
steps the simulation `max_iter` times and divides accumulated edge weights by `max_iter`;
default is 1000, matching the claim.

**Leiden/cologne8 result — contradicted.** Reading
`region-splitting/clusters/leiden/cologne8_clusters.json` (the file this exact pipeline
produces for cologne8, all 8 TLS nodes present) gives:

```
num clusters: 2
sizes: [6, 2]
modularity: 0.012345679012345845
```

not modularity 0.624 and not "each of the eight signalized intersections occupies its own
region." The mechanical reason: `LeidenService.get_clusters()` calls
`self.merge_singletons(cluster)` right after `leidenalg.find_partition`, which explicitly
re-merges any singleton community (degree ≤1 in the TLS-projected graph) into a neighboring
community. Whatever singleton-per-signal partition Leiden may originally find is actively
undone before the metrics (including modularity) are even computed — the 0.0123 modularity
is measured on the *merged* 2-cluster result, not the claimed pre-filter partition. There is
also no separate "before filtering"/"after filtering" step in this code path at all: Leiden
already runs only on the 8-node TLS-projected graph (`build_tls_projected_graph`), not on the
full raw network graph that would later need filtering down to signals.

### §3.3 Hierarchical Meta-Policy

**Regional state dimensionality — contradicted.** Claim: `s_k` is a 10-dim vector
`[occupancy, mean wait, vehicle count, mean speed, in-flow, out-flow, net flow, pressure,
congestion ratio, phase]`. Actual (`advesarial/src/services/traci.py::get_regional_state`,
also `advesarial/src/training/trainer.py: D_REG = 4`):

```python
region_states.append(torch.tensor([stop_car_num, waiting_time, centroid_x, centroid_y]))
```

a 4-dim vector of stopped-vehicle count, total waiting time, and normalized region centroid
(x, y) — none of flow/pressure/congestion-ratio/phase is present at the regional level
(pressure/congestion/phase only exist in the *per-intersection* observation, §3.4). This is
independently corroborated by `HILIGHT_SPEC_ALIGNMENT_REPORT.md` §0/§3, which documents this
4-feature regional state as an intentional 2026-09-09 fix to match the (original HiLight)
paper's Appendix A.

**Transformer encoder — contradicted on every listed hyperparameter except PE type.**
Claim: 6 pre-layer-normalized layers, 8 heads, `d_model=128`, sinusoidal PE. Actual
(`advesarial/src/schema/encoder_config.py`, instantiated unmodified in
`training/trainer.py`):

```python
d_reg: int = 4
d_model: int = 4
nhead: int = 2
num_layers: int = 3
dim_feedforward: int = 165
```

and `models/encoder/transformer.py` builds `nn.TransformerEncoderLayer(..., norm_first=False)`
— explicitly *not* pre-norm; the code comment even says the (original HiLight) spec calls for
"no layer norm" at all, and PyTorch's layer can't fully drop its two built-in per-layer norms,
so this is post-norm-ish by construction, the opposite of "pre-layer-normalized." Sinusoidal
positional encoding is correctly implemented (`models/encoder/positional.py`, `sin`/`cos` with
a 10000 scaling factor) — that part of the claim holds.

**LSTM hidden size — contradicted.** Claim: hidden size 128. Actual
(`training/trainer.py: LSTM_HIDDEN = 256`, `LSTM_LAYERS = 4`, passed straight into
`SubGoalGenerator`): hidden size 256 (and 4 layers, which the claim doesn't mention at all).

**Sub-goal shape — contradicted.** Claim: `G_t = [G_t^W, G_t^Q] ∈ R^{2N}` — i.e., a
waiting/queue pair *per region or intersection* (`N` of them). Actual
(`models/lstm.py::SubGoalGenerator`): the LSTM's per-subregion hidden states are concatenated
and projected through `phi` to one `d_g=16`-dim vector `G` for the *entire network*, then a
`goal_head: Linear(16, 2)` collapses that to exactly **2 scalars** `(G_w, G_q)`, used only as
MSE regression targets against the network-wide totals `(W_global, Q_global)`
(`training/adversarial.py::compute_meta_loss`). There is no per-region or per-intersection
sub-goal anywhere — not `R^{2N}`, just `R^2`. The richer 16-dim `G` is computed but never
consumed by anything downstream.

### §3.4 Intersection-Level Sub-Policy

**Local observation `o_i` — confirmed, field order included.** Claim:
`[vehicles, queue, occupancy, flow, stops, waiting, speed, pressure, congestion, delay]`.
Actual (`advesarial/src/services/traci.py::get_observations`):

```python
[car_num, queue_length, occupancy, flow, stop_car_num, waiting_time,
 average_speed, pressure, congestion_ratio, delay]
```

Matches 1:1 in both count and semantics (minor note: `flow` and `car_num`/`vehicles` are
computed with the identical line `traci.lane.getLastStepVehicleNumber(lane)`, i.e. they are
numerically the same quantity counted twice under two different feature names — cosmetic,
doesn't affect the claim's truth).

**EMA standardization clipped to [-5,5] — confirmed.** `training/trainer.py::_normalize_obs`
keeps a running EMA mean/var (`obs_alpha=0.01`) and does
`torch.clamp(normalized, -5, 5)`, exactly as claimed.

**Local MLP → 64-dim — contradicted.** Claim: local MLP output is 64-dim. Actual
(`training/trainer.py: self.F = OBS_DIM` → `LocalEncoder(in_dim=10, hidden_dim=10)`): the
local encoder maps 10→10, not 10→64. (The code comment attributes this to "the shared-MLP
output dim == raw obs dim, per the paper's own worked example" — i.e. a different numeric
assumption than the 64 claimed here.)

**Graph attention — aggregation rule contradicted, scoring mechanics confirmed.** Claim:
`e_ij = LeakyReLU_0.2(a^T[z_i‖z_j])`, softmax normalization, and summation update
`z_i' = z_i + Σ_j α_ij z_j`. Actual (`models/gat.py::GATLayer`): the LeakyReLU(0.2) attention
score and softmax normalization are implemented as claimed, but the update is
**concatenation**, not summation: `z_i' = h_i ‖ α_1 h_{j1} ‖ ... ‖ α_4 h_{j4}` (zero-padded to
4 neighbors), giving a `5F`-dim output per node instead of an `F`-dim residual-sum output. The
module's own docstring calls this out explicitly ("This is concatenation, not the summation
aggregation of a standard GAT").

**Fused state `s_i` — contradicted in both composition and size.** Claim:
`s_i = [z_i' ‖ (1/N)Σ_j z_j ‖ G_t] ∈ R^144`. Actual (`training/trainer.py::_encode_step`,
`models/sub_policy.py::SubPolicy.fuse_global`): `final = [z (5F=50) ‖ F_g (D_REG=4)] → 54`-dim.
Two structural deviations beyond the raw size mismatch (54 vs. 144): (a) there is no separate
network-mean term — the "neighbor averaging" is already folded into the GAT's own
attention-weighted concatenation, it isn't computed again as a plain mean; (b) the vector
concatenated in is `F_g`, the Meta-Policy's current-timestep **Transformer global embedding**
(labeled `F_g` throughout the code, dim 4), not `G_t`, the **LSTM-produced sub-goal**. This is
a deliberate, documented design choice — `HILIGHT_SPEC_ALIGNMENT_REPORT.md` §3/§0 calls wiring
`F_g` into the Sub-Policy "the single highest-value bug found" and fixed it that way on
purpose — but it means the sub-goal `G_t` the paper's equation names never actually reaches
any intersection's state vector.

**Action count — off by one.** Claim: "one of seven action indices." Actual
(`training/trainer.py: NUM_ACTIONS = 8`): the shared actor head has 8 output slots; real,
per-intersection valid-phase counts are masked in via `get_action_mask`, so a given
intersection may well only ever use 7 (or fewer) of them, but the architecture itself is
built around 8, not 7.

### §3.5 Safe Signal Switching

**Mechanism — confirmed.** `advesarial/src/services/traci.py::set_phase` implements exactly
the claimed scheme: a phase change request is first routed through a yellow phase
(`in_transition`/`pending_phase`/`yellow_timer`), and a `time_since_last_switch` counter blocks
any new phase change before `min_green_time` steps have elapsed. This structurally matches
"yellow transition and minimum-green constraints."

**"10-s control cadence," yellow/min-green both = one 10s decision step — contradicted.**
Defaults (`advesarial/src/schema/traci_config.py`): `step_length=1.0`, `yellow_duration=3.0`,
`green_duration=5.0`, `min_green_steps=5` (→ 5 simulated seconds of minimum green, since each
"step" is `step_length` seconds). There is no 10-second decision cadence anywhere: the RL
agent acts and `set_phase` is called once per `traci.simulationStep()` — i.e. a decision every
1 simulated second, not every 10. Neither the yellow time (3s) nor the min-green time (5s)
equals a 10-second step, and no code path aggregates 10 simulation steps into one decision.

---

## Net assessment

Of the 15 individually checkable numeric/structural claims in the two images, **4 are fully
confirmed** (per-intersection obs composition, EMA+clip standardization, the regional edge
weight formula and its 1000-step averaging), **3 are partially confirmed** (sinusoidal PE
inside an otherwise-wrong Transformer config; LeakyReLU/softmax inside an otherwise-wrong GAT
aggregation; the action count is close but off by one), and **8 are not implemented as
claimed**, most of them not by a small margin: the regional state is 4-dim where 10 is
claimed, the Transformer and LSTM hyperparameters are smaller/different across the board, the
sub-goal is a 2-scalar network-wide quantity rather than an `R^{2N}` per-region vector that
never even reaches the Sub-Policy's state, the local encoder outputs 10-dim rather than
64-dim, the fused Sub-Policy state is 54-dim rather than 144-dim and uses a different signal
(`F_g` instead of `G_t`) than the one named in the equation, there's no 10-second decision
cadence, and the headline Leiden/cologne8 result (modularity 0.624, 8 singleton regions) is
contradicted by the cluster file this exact pipeline currently produces (modularity ≈0.012,
2 merged clusters) — because of an active `merge_singletons` step that works directly against
the claimed outcome.

In short: the broader architectural *shape* (Transformer+LSTM meta-policy → GAT-based
sub-policy → safe phase switching, graph-weighted regional decomposition) is real and does
run end-to-end, but most of the specific numbers and a few structural details in this draft
text describe an earlier or aspirational version of the design rather than the code currently
on `clustering-approaches`. The regional-state dimensionality, the sub-goal's shape/role, the
local-encoder/fused-state widths, the 10-second cadence, and the cologne8 Leiden result are
the claims most worth either fixing in code or rewriting in the paper before this section is
finalized.
