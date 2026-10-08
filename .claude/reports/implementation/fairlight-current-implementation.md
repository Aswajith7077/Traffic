# FairLight — Current Implementation Report

**Scope:** `region-splitting/` + `advesarial/` on branch `clustering-approaches`, as the code
actually runs today (not the paper draft's claims — see
`.claude/reports/paper/verification/fairlight-architecture-claims.md` for that comparison).
Cross-referenced with `metrics.txt` (empirical run logs), `AGENTS.md`, and
`HILIGHT_SPEC_ALIGNMENT_REPORT.md`.

---

## 1. Proposed Work

### 1.1 Overall idea

City-scale adaptive traffic signal control must reconcile two failure modes: a fully
centralized controller does not scale past a handful of intersections, while a fully
decentralized one (one independent agent per intersection) has no way to coordinate
network-wide objectives like keeping a corridor moving or balancing load across districts.

The implemented system resolves this with a **two-level hierarchy sitting on top of an
automatically-constructed regional partition**:

1. **Region construction** (`region-splitting/`) — the signalized road network is modeled as
   a weighted graph and automatically partitioned into subregions using graph community
   detection (Louvain, Leiden) and/or density clustering (DBSCAN), plus four two-stage hybrid
   combinations of the two families. This step runs once, offline, before training, and its
   output (`clusters/<method>/<scenario>_clusters.json`) is consumed by the RL side.
2. **Meta-Policy** (`advesarial/src/models/encoder/transformer.py`,
   `advesarial/src/models/lstm.py`) — a Transformer + LSTM that watches a short temporal
   window of aggregate per-region traffic statistics and produces a single, network-wide
   target pair (a desired global waiting time, a desired global queue length).
3. **Sub-Policy** (`advesarial/src/models/{encoder/local,gat,sub_policy}.py`,
   `advesarial/src/agents/actor.py`) — one shared actor-critic head, applied independently at
   every signalized intersection, that consumes its own 10-dimensional local observation, an
   attention-weighted summary of its immediate neighbors (graph attention over the
   intersection-adjacency graph, not the region graph), and the Meta-Policy's current global
   embedding, and outputs a signal-phase decision.
4. **Safe switching** (`advesarial/src/services/traci.py::set_phase`) — every phase change is
   mechanically routed through a yellow transition and a minimum-green hold, independent of
   what the learned policy wants, so the network cannot encounter an aphysical signal change
   even if the policy has not yet learned to avoid one.

The two levels are trained jointly (`advesarial/src/training/trainer.py`) with an adversarial
pair of losses: the Sub-Policy is pushed, in part, to make the network's *actual* queue/wait
totals track the Meta-Policy's issued targets, while the Meta-Policy is pushed to keep issuing
targets that are themselves achievable (regressed against the actual totals it observes).
Region construction, in turn, is explicitly treated as part of the research problem rather
than a fixed preprocessing step — the pipeline can regenerate every method × scenario
combination (`region-splitting/generate_clusters.py`) so partitioning choice itself can be
compared as an experimental variable.

### 1.2 Mathematical model of architecture

**Region graph and edge weight** (`region-splitting/services/traci.py::get_edge_weight`,
averaged over `max_iter=1000` simulation steps in `compute_weights`):

```
w_e = α·n_e + β·ω_e + γ·κ_e·n_e        (α, β, γ) = (1.0, 0.3, 2.0)
```

where `n_e` is the edge's last-step vehicle count, `ω_e` its waiting time, and
`κ_e = clip(1 − speed/max_speed, 0, 1)` its congestion ratio. Non-signalized junctions between
two traffic lights are contracted away (`services/topology.py::build_tls_projected_graph`), so
every partitioning method only ever sees the signalized-intersection graph.

**Louvain modularity** (`services/louvian.py`, via `community_louvain.best_partition`):

```
Q = (1/2m) · Σ_{i,j} [A_ij − k_i·k_j / 2m] · δ(c_i, c_j)
```

maximized greedily; resulting communities smaller than `min_cluster_size=5` are merged into
the neighboring community reached by the single heaviest edge. Leiden (`services/leiden.py`)
maximizes the same quantity via `leidenalg.ModularityVertexPartition` and merges any resulting
singleton (degree ≤ 1) into its sole neighbor. DBSCAN (`services/dbscan.py`) instead clusters
raw junction (x, y) coordinates by density, with `eps`/`min_samples` auto-tuned
(`auto_tune_config`) to keep mean cluster size in a target band and noise fraction under 20%.
The four hybrid methods (`services/hybrid.py`) chain these two families in both orders over an
intermediate supernode graph.

**Regional state** fed to the Meta-Policy, per region `k`
(`advesarial/src/services/traci.py::get_regional_state`):

```
s_k = [stop_car_num_k, waiting_time_k, centroid_x_k, centroid_y_k]   ∈ ℝ⁴
```

aggregated by summing over the region's intersections, with the centroid min-max-normalized
to the network's bounding box.

**Meta-Policy** (`models/encoder/transformer.py`, `models/lstm.py`,
`schema/encoder_config.py`): a linear input projection `ℝ⁴ → ℝ⁴` (`d_model=4`), a learnable
global token prepended to the `M` regional tokens, sinusoidal positional encoding, and a
3-layer `TransformerEncoder` (`nhead=2`, `dim_feedforward=165`) run over a `T=20`-step sliding
window (`utils/replay_buffer.py::RegionalStateBuffer`). The global token's output at the
current timestep is the **global embedding** `F_g ∈ ℝ⁴`; the per-region token outputs across
the whole window feed an LSTM (`hidden=256`, `4` layers) run independently per region over the
time axis. The LSTM's final hidden states across all `M` regions are concatenated
(`M·256`-dim) and projected by a small MLP to a 16-dim vector `G`, then linearly collapsed to
two scalars:

```
goal = (G_w, G_q) = Linear_{16→2}(G)
```

**Meta loss** (`training/adversarial.py::compute_meta_loss`):

```
L_Meta = MSE((G_w, G_q), (W_global, Q_global)) + η1 · r_g       η1 = 0.1
```

**Per-intersection observation** (`services/traci.py::get_observations`):

```
o_i = [vehicles, queue, occupancy, flow, stops, waiting, speed, pressure, congestion, delay]  ∈ ℝ¹⁰
```

EMA-normalized (`α=0.01`) against a running mean/variance and clipped to `[-5, 5]`
(`training/trainer.py::_normalize_obs`).

**Sub-Policy** (`models/encoder/local.py`, `models/gat.py`, `models/sub_policy.py`,
`agents/actor.py`): a local MLP `10 → 10 → 10` (ReLU); a **Graph Attention Concat (GAC)** layer
over the intersection-adjacency graph (up to 4 neighbors, zero-padded):

```
e_ij = LeakyReLU_0.2(aᵀ[h_i ‖ h_j]),   α_ij = softmax_j(e_ij)
z_i  = h_i ‖ α_{i,1}h_{j1} ‖ α_{i,2}h_{j2} ‖ α_{i,3}h_{j3} ‖ α_{i,4}h_{j4}   ∈ ℝ^{5F} = ℝ⁵⁰
```

— concatenation, deliberately not the summation update of a standard GAT. This is fused with
the Meta-Policy's global embedding:

```
state_i = [z_i ‖ F_g]   ∈ ℝ⁵⁴
```

and passed through a shared trunk `54 → 128 → 64 → 32` (ReLU + dropout 0.1), an actor head
`32 → 16 → 8` (softmax, masked per-intersection to its real valid-phase count), and a
dual-branch critic: `32 → 16 → concat(16, first 16 of shared) → 1` (state value), plus an
unused second "latent plan" branch kept for structural completeness.

**PPO-clip + GAE** (`training/gae.py`, `training/adversarial.py::compute_sub_loss`):

```
δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
A_t = δ_t + γλ·(1−done_t)·A_{t+1}                    γ = 0.99, λ = 0.95

ratio = exp(logπ_new − logπ_old)
L_policy = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)    ε = 0.2
L_value  = MSE(V, returns)
L_AC     = L_policy + 1.0·L_value − 0.01·H[π]
```

**Sub loss**, with the alignment term pulling the Sub-Policy toward the Meta-Policy's issued
targets (gradient blocked from flowing back into the Meta-Policy):

```
L_Sub = L_AC + η2 · [β_q·(Q_global − G_q) + β_w·(W_global − G_w)]     η2 = 0.1, β_q = β_w = 0.5
```

**Reward** (`services/traci.py::get_intersection_reward`,
`environment/environment.py::_compute_reward`/`compute_goal_reward`):

```
r_i^t = −(queue_length_i + waiting_time_i + delay_time_i + pressure_i − speed_score_i)
r_g^t = −[β_q·(W_global − G_q) + β_w·(Q_global − G_w)]
r^t   = r_i^t + r_g^t             (per-agent (N,) vector, r_g broadcast identically)
```

normalized by a scalar EMA (`α=0.01`) before being stored in the rollout buffer.
`delay_time_i` is a per-lane proxy: `Σ_lanes max(length/max(speed,0.1) − length/max_speed, 0)`.
`pressure_i = Σ_{links} (queue_in − queue_out)`.

**Safe switching** (`services/traci.py::set_phase`) is a per-intersection state machine rather
than an equation: a requested phase change is held until `time_since_last_switch ≥
min_green_time` (5 simulated seconds by default), then routed through a yellow phase held for
`yellow_time` (3 seconds) before the target green phase is actually committed.

**Evaluation** (`evaluate.py`), Average Travel Time / Average Delay Time over completed
vehicles only:

```
ATT = exit_time − entry_time
ADT = ATT − free_flow_time             free_flow_time = Σ_{edges in route} length / max_speed
```

### 1.3 Flow of architecture

Per RL decision (one simulated second, `step_length=1.0`):

1. `TraciService.get_regional_state(clusters)` → 4-dim state per region, pushed into a 20-step
   circular buffer.
2. **Meta-Policy forward**: Transformer encodes the buffered window → current global embedding
   `F_g` + per-region embeddings across the window → LSTM over the time axis per region →
   `(G_w, G_q)`.
3. `TraciService.get_observations()` → 10-dim observation per intersection → EMA-normalize +
   clip.
4. **Sub-Policy forward**: local MLP → GAC (self + up to 4 TLS-adjacency neighbors) → fuse with
   `F_g` → shared trunk → masked softmax over 8 action slots → sample (training) / argmax
   (evaluation) a phase index per intersection.
5. `TraciService.set_phase()` enforces the yellow/min-green state machine before the requested
   phase actually takes effect on the SUMO traffic light.
6. `Environment.step()` advances SUMO one second, computes the per-agent local reward and the
   shared goal-reward, and EMA-normalizes the total.
7. The raw (undetached-graph-safe) step is appended to the `RolloutBuffer`. Every
   `META_UPDATE_INTERVAL=10` steps, one Meta-Policy gradient step runs (`L_Meta`, grad-clipped
   at norm 10, Adam lr `3e-4`). Once the buffer reaches `ROLLOUT_LENGTH=240` steps (or the
   episode ends), `PPO_EPOCHS=4` Sub-Policy update epochs run — each epoch **recomputes**
   `local_encoder → GAC → fuse → actor_critic` from the stored raw observations (not cached
   fused states), so gradients reach the encoder/GAT weights, not just the actor-critic head.
8. Repeat for `episode_steps` simulated seconds per episode (default 1000); checkpoint every
   `save_every` episodes (default 10) to `models/<scenario>/run_<timestamp>_<rand>/`.

Region construction runs once, ahead of training, as its own pipeline:

1. Contract the SUMO net down to a traffic-light-only graph
   (`build_tls_projected_graph`).
2. Step the simulation 1000 times, accumulating the per-edge weight above, then average.
3. Run the selected partitioner (Louvain / Leiden / DBSCAN / one of the four hybrids) over the
   resulting weighted graph (or a supernode graph, for hybrids).
4. Apply the method's own small/singleton-cluster merge rule so every signalized intersection
   ends up in some region.
5. Write `{"clusters": ..., "metrics": ...}` JSON and copy it into
   `advesarial/clusters/<method>/<scenario>_clusters.json`.

### 1.4 Steps (operational)

```bash
# 1. Partition the network (per scenario/method, or all at once)
cd region-splitting && python generate_clusters.py                       # all methods × 5 scenarios
cd region-splitting && python main.py --scenario cologne8 --method dbscan_leiden

# 2. Cluster JSON is auto-copied into advesarial/clusters/<method>/ by generate_clusters.py;
#    otherwise copy it manually (the pipeline's `copy` step is a safety net).

# 3. Train
cd advesarial && CLUSTER_METHOD=dbscan_leiden TRAFFIC_SCENARIO=cologne8 python src/sample.py

# 4. Evaluate
cd advesarial && python src/evaluate.py --model-dir ../models/cologne8/run_<timestamp> --steps 500

# One-shot cluster → copy → train → eval for a single scenario:
python train_eval_cologne8.py --cluster-method louvian_dbscan --episodes 10 --episode-steps 1000 --steps 500
```

### 1.5 Fairness

The framework's stated motivation (echoed in the paper draft's abstract/introduction) is that
optimizing only network-wide averages — throughput, mean delay — can hide a policy that
systematically starves a minority of intersections or vehicles in order to keep the dominant
corridors moving. Fairness is framed there as a first-class control objective, to be measured
via waiting-time variance, a tail/starvation penalty, Jain's Fairness Index, and an "envy" term
comparing each intersection against its best-performing neighbor.

**Benefits of treating fairness as a first-class objective:**

- Prevents a small number of chronically deprioritized approaches from absorbing all of the
  network's efficiency gains — the failure mode a pure average-delay objective cannot see.
- Keeps the Louvain/Leiden/DBSCAN regional decomposition honest: a poorly balanced subregion
  (one dominant intersection plus several starved ones) is exactly the situation a
  fairness-blind Meta-Policy would never notice, since it only ever sees regional *sums*.
- Improves the real-world acceptability of an automated controller — a system that is
  efficient on average but visibly unfair to specific neighborhoods is a harder deployment
  sell than one with a bounded worst case.
- Bounds how aggressively the Meta-Policy's goal-alignment term can be satisfied by sacrificing
  a subset of intersections to hit a favorable global `(W_global, Q_global)` total — without a
  fairness term, two very different per-intersection distributions can produce the identical
  global sum the Meta-Policy actually regresses against.

**Honest status in the current codebase:** the explicit fairness composite described above is
**not currently wired into the reward that drives training**. Today's trajectory reward is
only the efficiency term `r_i = −(ql+wt+dt+ps−ss)` plus the Meta-Policy's goal-alignment term
`r_g` (Section 1.2) — no variance, starvation, Jain's-index, or envy term enters the loss
PPO actually optimizes. The fairness-relevant *measurements* do already exist in
`advesarial/src/services/traci.py` — `compute_jain_index`, `get_pedestrian_waiting_times`,
`get_pedestrian_conflict_count`, `get_non_emv_waiting_times`, `get_emergency_waiting_time` are
all implemented — but are presently dead code, deliberately kept in place
(`HILIGHT_SPEC_ALIGNMENT_REPORT.md` §5) for a future reward revision rather than deleted.

One fairness-adjacent property **is** active today, though: the local reward is computed and
applied **per-agent** — an `(N,)` tensor using each intersection's own lane metrics — rather
than a single team-wide mean broadcast identically to every intersection
(`environment/environment.py::_compute_local_reward`). This matters because a shared/team
reward lets the policy hide a persistently bad intersection behind a well-performing neighbor's
average; the per-agent reward at least ensures each intersection is scored, and therefore
trained, on its own outcomes.

---

## 2. Results and Discussion

### 2.1 Cluster-method comparison (cologne8, same training/eval batch, `metrics.txt`)

| Cluster method | Eval steps | Completed vehicles | Avg queue | Avg wait (s) | ATT (s) | ADT (s) |
|---|---|---|---|---|---|---|
| `dbscan` | 1000 | 132 | 512.44 | 17.13 | 164.26 | 117.67 |
| `dbscan_louvian` | 1000 | 263 | 317.79 | 6.74 | 109.46 | 60.20 |
| `dbscan_leiden` | 1000 | 265 | 315.60 | 7.66 | 111.40 | 61.95 |
| `louvian_dbscan` | 1000 | 359 | 370.69 | 28.73 | 191.90 | 144.30 |
| `leiden_dbscan` | 1000 | 263 | 317.79 | 6.74 | 109.46 | 60.20 |

In this batch, the "clustering-first" hybrids (`dbscan_louvian`, `dbscan_leiden`) roughly
**halve ATT/ADT and double completed throughput** relative to plain `dbscan`, consistent with
the design intuition recorded in `AGENTS.md`: DBSCAN alone gives coarse, spatially compact
regions the Meta-Policy was designed for, while a community-detection refinement on top adds
traffic-flow-aware boundaries DBSCAN's pure geometry misses.

**Caveat — this ranking does not hold up across runs.** A separate batch of longer (3600-step)
evaluations on cologne8, trained at different times, shows a different ordering: `dbscan`
(ATT 329.70/ADT 261.98), `dbscan_leiden` (ATT 453.06/ADT 384.90 — now the *worst*), `louvian`
(ATT 306.79/ADT 237.70 — now the *best*), and the static-grid `manual_clustering` baseline
(ATT 324.83/ADT 264.64) all within the same rough band. `metrics.txt` also records a cologne8
training run whose **Meta Loss spiked to 4.7 × 10⁹** mid-training (run_50) before falling back
to the thousands a few checkpoints later — direct evidence of the instability `AGENTS.md`
already documents ("AC loss can explode from ~258 to >129k"). Given 10-episode/1000-step
training budgets, a single seed per method, and this degree of loss instability, the
cluster-method comparison above should be read as **suggestive of hybrid methods being
competitive, not as a settled ranking** — it would need multiple seeds and a stabilized
training run per method before drawing a firm conclusion.

### 2.2 Why specific elements matter

- **Reward weight selection (`β_q = β_w = 0.5`)**: `W_global` (summed waiting time, can run into
  the thousands of seconds) and `Q_global` (summed queue length, a few hundred vehicles) live on
  very different natural scales. An equal 0.5/0.5 split in `r_g` and in the Meta loss's MSE
  target therefore does *not* give the two quantities equal influence in practice — whichever
  has the larger raw magnitude (typically `W_global`) dominates the gradient. This is the direct
  mechanical explanation for the Meta-Loss spikes in §2.1: `MSE((G_w,G_q),(W_global,Q_global))`
  is regressing against an unnormalized target, so a stretch of unusually high network-wide
  waiting time produces a squared-error term in the billions almost by construction.
- **`η1 = η2 = 0.1`** (meta-loss goal-reward weight / sub-loss alignment weight): kept small and
  symmetric so the Meta/Sub adversarial coupling acts as a soft nudge rather than overwhelming
  either level's primary objective (the MSE regression for the Meta-Policy, the PPO-clip
  objective for the Sub-Policy) — a larger value would let a swing in one level's loss drown
  out the other's own learning signal.
- **PPO mechanics (`clip ε=0.2`, `entropy_coef=0.01`, `value_coef=1.0`)**: standard defaults,
  but they carry extra weight here because the actor head is *shared* across intersections with
  different real phase counts via action masking (§1.2). Too tight an entropy bonus or too loose
  a clip could let the policy collapse onto a single aliased action index before it has learned
  to respect the per-intersection mask.
- **`max_grad_norm = 10.0`, Adam `lr = 3e-4`**: the main defense against the divergence in
  §2.1/`AGENTS.md`. This repo previously used a tighter clip (0.5) before being loosened to 10.0
  for closer fidelity to the reference paper's Table 5 — a direct, visible trade-off between
  paper fidelity and empirical training stability, with the metrics-log loss spikes as the cost
  of that choice.
- **GAE (`γ=0.99`, `λ=0.95`)**: a long effective horizon with a moderate bias/variance
  trade-off; reasonable given the per-step reward is itself an EMA-normalized z-score rather
  than raw units, which keeps advantages roughly comparable across rollouts of differing raw
  reward scale.
- **Safe-switching timers (`min_green_time=5` steps, `yellow_time=3` steps, green cap `5s`)**:
  these bound how fast the learned policy can flip phases. Too short invites exactly the
  rapid-flipping pathology the fixed-duration net normalization (`AGENTS.md`) was meant to
  prevent; too long reintroduces the original long-wait problem that normalization pass fixed
  (manhattan greens up to 82s pre-normalization). This is also a quiet fairness lever: an
  overly long min-green on one phase is time a competing approach is guaranteed to wait,
  regardless of how empty or full it is.
- **Clustering edge weight `(α, β, γ) = (1.0, 0.3, 2.0)`**: weighting congestion (`κ_e·n_e`) at
  2× vehicle count means the partitioning graph intentionally treats "is this edge currently a
  bottleneck" as more important than "is this edge currently busy" — this is what lets the
  resulting regions track actual traffic coherence rather than pure static road topology. The
  1000-step averaging window exists specifically to smooth over single-step simulation noise so
  the partition reflects a representative demand pattern, not one transient snapshot.
- **Region count `M`**: not a tuning knob directly, but a consequence of the chosen
  partitioning method that reshapes the Meta-Policy's actual capacity — `M` multiplies the
  LSTM's `phi` input size (`M · d_hidden`) and sets the Transformer's token count. This is the
  concrete mechanism behind `AGENTS.md`'s warning that Leiden's ~1974 singleton regions on
  manhattan are unsuitable for this Meta-Policy: `d_g=16` is a fixed-size sub-goal regardless of
  `M`, so with `M` in the thousands the per-region signal is diluted far below what a 16-dim
  summary can usefully carry, whereas DBSCAN's 72 coarse manhattan regions keep that signal
  dense enough to be informative.

---

## 3. Conclusion and Future Work

The current implementation delivers a working, end-to-end hierarchical pipeline — automatic,
pluggable regional decomposition (Louvain/Leiden/DBSCAN and four hybrids) feeding a
Transformer+LSTM Meta-Policy and a GAT-based, mask-aware Sub-Policy, trained jointly with
PPO+GAE and an adversarial goal-alignment loss, behind a hard safe-switching layer. It runs on
five real scenarios end-to-end (cluster → copy → train → evaluate) and produces genuine
ATT/ADT measurements. Fairness is currently present as a strong design motivation and as
largely-unused instrumentation (Jain's index, pedestrian/EMV waiting-time helpers) rather than
as an active term in the loss being optimized; the one fairness-relevant property already
active is the per-agent (not team-averaged) local reward. Training stability (meta-loss spikes
into the billions) and the single-seed, short-budget cluster-method comparisons are the two
weakest links in the current empirical picture.

**Future work:** wire the existing fairness helpers (Jain's index, tail-waiting, envy) into the
trajectory reward rather than leaving them unused; normalize the Meta-Policy's `(W_global,
Q_global)` regression targets to stop the loss-magnitude spikes; and re-run the cluster-method
comparison across multiple seeds and a longer, stabilized training budget before treating any
partitioning method as conclusively best.
