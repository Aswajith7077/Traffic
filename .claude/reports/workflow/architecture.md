# Traffic HRL — High-Level Architecture Flow

Derived directly from the current code (not from `AGENTS.md` or `summary.md`, which
have drifted — see §6). Covers both modules end to end: `region-splitting/` →
cluster JSON → `advesarial/` → trained policy → evaluation metrics.

---

## 1. Repo-level orchestration

Two CLI entry points sit above both modules and drive them as subprocesses.
Neither is mentioned in `AGENTS.md`, but `pipeline.py` is the only place the
full 5-stage flow (baseline → cluster → copy → train → eval) is wired together.

```mermaid
flowchart TD
    CLI1["pipeline.py<br/>(all / baseline / cluster / copy / train / eval / run_gui)<br/>any scenario, any --cluster-method"]
    CLI2["train_eval_cologne8.py<br/>cologne8 only, no baseline step"]

    CLI1 --> ST1
    CLI2 -. "imports step_cluster, step_copy,<br/>step_train, step_eval" .-> ST2

    subgraph STEPS ["pipeline.py step_* functions"]
        direction LR
        ST1["step_baseline<br/>plain SUMO, actuated TLS"] --> ST2["step_cluster"]
        ST2 --> ST3["step_copy"]
        ST3 --> ST4["step_train"]
        ST4 --> ST5["step_eval"]
    end

    ST1 -. "subprocess: sumo -c scenario.sumocfg" .-> SUMO1[("tripinfos.xml / stats.xml<br/>→ metrics.txt")]
    ST2 -. "subprocess: python main.py --method M" .-> RS["region-splitting/main.py"]
    ST3 -. "shutil.copy2 (safety net)" .-> COPYFS[("region-splitting/clusters/M/*.json<br/>→ advesarial/clusters/M/*.json")]
    ST4 -. "subprocess: python src/sample.py" .-> SAMPLE["advesarial/src/sample.py"]
    ST5 -. "subprocess: python src/evaluate.py" .-> EVAL["advesarial/src/evaluate.py"]

    style CLI1 fill:#2b6cb0,color:#fff
    style CLI2 fill:#2b6cb0,color:#fff
```

**Why drawn this way:** `pipeline.py` passes scenario/method/episode config to
its children purely through environment variables (`TRAFFIC_SCENARIO`,
`CLUSTER_METHOD`, `TRAFFIC_EPISODES`, …) and `subprocess.run(cwd=...)` — there
is no shared Python import between `region-splitting` and `advesarial` at
runtime, which is why `AGENTS.md` correctly calls them "two independent
modules." `pipeline.py` is the seam that makes them act like one pipeline.

---

## 2. Module 1 — `region-splitting/`: network → cluster JSON

```mermaid
flowchart TD
    MAIN["main.py: run_partition(scenario, method)"] --> REG["services/registry.py: create_service(method)"]

    REG -->|"leiden / louvian"| CD["LeidenService / LouvianService<br/>(needs a live TraciService —<br/>edge weights from simulation)"]
    REG -->|"dbscan"| DBS["DBSCANService<br/>(static geometry only)"]
    REG -->|"manual_clustering"| GRID["GridService"]
    REG -->|"dbscan_louvian / dbscan_leiden /<br/>louvian_dbscan / leiden_dbscan"| HYB["HybridClusteringService<br/>(2-stage: DBSCAN + community detection,<br/>either order)"]

    CD --> TOPO
    DBS --> TOPO
    HYB --> TOPO
    GRID -.-> TOPO

    TOPO["topology.build_tls_projected_graph<br/>contracts every chain of non-signalized<br/>junctions into one weighted edge —<br/>graph vertices = traffic lights only"]

    TOPO --> BG["service.build_graph()"]
    BG --> GC["service.get_clusters()"]

    GC -->|"dbscan only, eps unset"| AUTO["dbscan.auto_tune_config<br/>sweep log-spaced eps,<br/>pick cluster count in [mean_min,mean_max]<br/>with lowest noise"]
    AUTO --> GC

    GC --> VIZ["generate_visualization()<br/>→ visualizations/&lt;method&gt;/&lt;scenario&gt;.png"]
    GC --> OUT[("clusters/&lt;method&gt;/&lt;scenario&gt;_clusters.json<br/>{ clusters: {region_id: [tls_ids]},<br/>metrics: {...} }")]

    OUT --> GEN["generate_clusters.py / generate_dbscan_clusters.py<br/>auto-copy"]
    GEN --> ADV[("advesarial/clusters/&lt;method&gt;/&lt;scenario&gt;_clusters.json")]
```

**Why drawn this way:** every service shares the same three-method interface
(`BaseClusteringService`), so the registry is a pure strategy-pattern switch —
the pipeline never branches on method name outside `registry.py`. The hybrid
services reuse the *same* DBSCAN primitives (`run_dbscan`, `auto_tune_config`)
as the standalone `DBSCANService` rather than re-implementing them, so a bug
fix in `dbscan.py` propagates to all four hybrid combinations automatically.
`topology.py` runs before every method (including DBSCAN, which otherwise
needs no graph) because DBSCAN still clusters node *coordinates* collected
from that same TLS-only graph — the graph isn't optional infrastructure, it's
the shared node universe every method partitions.

---

## 3. Module 2 — `advesarial/`: Trainer composition

`Trainer` (`advesarial/src/training/trainer.py`) is the de facto architecture
root — `sample.py`, `train.py` (now a thin compatibility wrapper, not broken —
see §6) and `evaluate.py` all just construct one and call its methods.

```mermaid
flowchart TD
    subgraph INIT["Trainer.__init__"]
        direction TB
        TC["TraciConfig"] --> TS["TraciService<br/>(TraCI/SUMO wrapper:<br/>phases, observations, rewards)"]
        TS --> ENVC["Environment<br/>(applies actions, computes reward)"]
        TS --> AUX["adjacency_list · intersections (N) · action_mask (N×8)"]

        CFGMOD["config.py singleton<br/>CLUSTER_METHOD env var →<br/>clusters/&lt;method&gt;/&lt;scenario&gt;_clusters.json<br/>(falls back dbscan → leiden)"] --> CL["self.clusters, M = len(clusters)"]

        subgraph SUB["Sub-Policy (per-intersection actions)"]
            LE["LocalEncoder<br/>MLP 10 → F"]
            GAT["GATLayer<br/>GAC concat: N × 5F"]
            AC["ActorCritic<br/>shared trunk → actor (8 actions)<br/>+ dual-branch critic"]
        end

        subgraph META["Meta-Policy (regional sub-goal)"]
            RSB["RegionalStateBuffer<br/>sliding T=20-step window"]
            TE["TransformerEncoder<br/>+ learned global token<br/>+ PositionalEncoder"]
            SG["SubGoalGenerator<br/>per-region LSTM → φ → goal_head<br/>→ (G_w, G_q)"]
        end

        CL --> RSB
        SubOpt["sub_optimizer (Adam)<br/>params: LE + GAT + AC"]
        MetaOpt["meta_optimizer (Adam)<br/>params: TE + SG"]
        ROLL["RolloutBuffer<br/>on-policy, cleared every update<br/>ROLLOUT_LENGTH = 240 steps"]
    end

    LE -.params.-> SubOpt
    GAT -.params.-> SubOpt
    AC -.params.-> SubOpt
    TE -.params.-> MetaOpt
    SG -.params.-> MetaOpt

    style SUB fill:#2f855a,color:#fff
    style META fill:#975a16,color:#fff
```

**Why drawn this way:** the two policy stacks are optimized completely
independently (two `Adam` instances, two loss functions) but share two
runtime objects — `TraciService` (the only source of simulation state for
both) and the per-step `goal`/`f_g` values that cross from Meta→Sub. That
cross-policy dependency is exactly what the sequence diagram in §4 traces.
Note `models/sub_policy.py` (`SubPolicy` class, exported from
`models/__init__.py`) implements the *same* encode→GAT→fuse logic that
`Trainer._encode_step` hand-rolls inline — `Trainer` never imports
`SubPolicy`. It's dead, duplicate code, not a second path through the system.

---

## 4. One training step — control & data flow

This is what `Trainer._step()` actually executes, called once per sim-second
inside `run_episode()`.

```mermaid
sequenceDiagram
    participant Tr as Trainer._step
    participant RSB as RegionalStateBuffer
    participant TE as TransformerEncoder
    participant SG as SubGoalGenerator
    participant TS as TraciService
    participant LE as LocalEncoder
    participant GAT as GATLayer
    participant AC as ActorCritic
    participant Env as Environment
    participant Roll as RolloutBuffer

    Note over Tr,SG: Meta-Policy forward (_meta_forward)
    Tr->>TS: get_regional_state(clusters)
    TS-->>Tr: regional_state (M, 4): [stop_cars, wait_time, centroid_x, centroid_y]
    Tr->>RSB: push(regional_state)
    RSB-->>Tr: window (T=20, M, 4)
    Tr->>TE: forward(window)
    TE-->>Tr: global_embedding (T, d_model), subregion_embeddings (T, M, d_model)
    Tr->>SG: forward(subregion_embeddings)
    SG-->>Tr: G, goal = (G_w, G_q)

    Note over Tr,AC: Sub-Policy forward (no_grad at collection time)
    Tr->>TS: get_observations()
    TS-->>Tr: obs (N, 10) per-intersection features
    Tr->>Tr: _normalize_obs (running mean/var EMA, clamp [-5,5])
    Tr->>LE: forward(obs)
    LE-->>Tr: h (N, F)
    Tr->>GAT: forward(h, adjacency_list)
    GAT-->>Tr: z (N, 5F)
    Tr->>Tr: concat(z, f_g broadcast) → fused_state (N, 5F+4)
    Tr->>AC: forward(fused_state, action_mask)
    AC-->>Tr: action_probs, value, (unused latent_plan)
    Tr->>Tr: sample action ~ Categorical(action_probs)

    Note over Tr,Env: Environment step
    Tr->>Env: step(action, goal)
    Env->>TS: set_phase(tls_id, action) for each intersection<br/>(min-green + yellow transition state machine)
    Env->>TS: simulationStep()
    Env->>TS: get_intersection_reward(i) per agent
    Env-->>Tr: reward (N,) normalized, done
    Tr->>TS: compute_global_state_now() → W_global, Q_global
    Tr->>Env: compute_goal_reward(goal) → r_g

    Tr->>Roll: add(obs, f_g, action, log_prob, value, reward, done, goal, w_global_sum, q_global_sum)

    alt step_count % META_UPDATE_INTERVAL(10) == 0
        Tr->>Tr: _update_meta → L_Meta = MSE(goal, (W,Q)) + eta1·r_g<br/>backward through TE+SG only
    end
    alt len(RolloutBuffer) >= 240 or done
        Tr->>Roll: get() batch (T, N, ...)
        Tr->>Tr: compute_gae → advantages, returns
        loop PPO_EPOCHS = 4
            Tr->>LE: re-encode obs[t] (gradients now enabled)
            Tr->>GAT: re-encode
            Tr->>AC: re-evaluate action_probs, value
            Tr->>Tr: L_Sub = PPO-clip(ratio,advantage) + value_coef·MSE − entropy_coef·H<br/>+ eta2·goal_alignment(beta_q,beta_w)<br/>backward through LE+GAT+AC only
        end
        Tr->>Roll: clear()
    end
```

**Why drawn this way:** the two update branches are gated on *different*
clocks (`META_UPDATE_INTERVAL` vs `ROLLOUT_LENGTH`/episode-end) and backprop
through disjoint parameter sets — this is what actually enforces the
hierarchical/independent-optimizer design, not anything explicit elsewhere.
The PPO loop re-runs `LocalEncoder → GAT → ActorCritic` from raw `obs`/`f_g`
on every epoch (see `RolloutBuffer`'s docstring) specifically so gradients
reach `LocalEncoder`/`GAT` — storing the already-fused tensor instead would
silently freeze those two modules.

---

## 5. Cluster-JSON data contract (the only coupling between modules)

```mermaid
flowchart LR
    RSOUT[("region-splitting/clusters/&lt;method&gt;/&lt;scenario&gt;_clusters.json")] -->|"shutil.copy2<br/>(manual, or automatic via<br/>generate_clusters.py / pipeline step_copy)"| ADVIN[("advesarial/clusters/&lt;method&gt;/&lt;scenario&gt;_clusters.json")]

    ADVIN --> RESOLVE["config.py: _resolve_cluster_path()<br/>try CLUSTER_METHOD verbatim<br/>→ fall back to dbscan<br/>→ fall back to leiden"]
    RESOLVE --> SINGLETON["config = Config(path)<br/>module-level singleton,<br/>evaluated at import time"]
    SINGLETON --> USE["Trainer.__init__: raw_clusters = config.clusters<br/>filtered to nodes ∈ TraciService intersections<br/>→ self.clusters, M regions"]
```

**Why drawn this way:** `config.py` runs `config = Config(_resolve_cluster_path())`
at **module import time**, not inside a function — so the cluster file is
locked in the moment anything does `from config import config/SCENARIO`,
before `argparse` in `sample.py`/`evaluate.py` even runs. `CLUSTER_METHOD`
and `TRAFFIC_SCENARIO` must therefore already be correct environment
variables *before* the interpreter starts, which is exactly why
`pipeline.py` sets them via `env=` on a fresh `subprocess.run` rather than
through in-process state.

---

## 6. Divergences found vs. the documented model (`AGENTS.md`, `summary.md`)

These were checked against the current code, not assumed from the docs:

| Documented claim | Current reality |
|---|---|
| "REINFORCE-style with 4 Adam optimizers" | **PPO-clip + GAE with 2 optimizers** (`sub_optimizer`, `meta_optimizer`); `training/adversarial.py`, `gae.py`, `rollout.py` implement this. No REINFORCE path exists. |
| "`train.py` is broken — `from agents import Actor` crashes" | `train.py` no longer imports `agents` at all; it's now a working thin wrapper around `training.Trainer`, identical in shape to `sample.py`. This bug appears **fixed**. |
| "`PhaseTracker.get_entropy()` calls `compute_phase_entropy()` without importing it — `NameError`" | `memory/phase_tracker.py` now imports it correctly from `utils`. Still **dead code** (never called in the training loop), but no longer a crash bug. |
| "`ReplayBufferItem` schema is unused" | Still true — `memory/replay_buffer.py::ReplayBuffer` (deque-based) is also unused; the actual training buffer is the unrelated `training/rollout.py::RolloutBuffer`. |
| Not mentioned | Root-level `pipeline.py` (375 lines) is the actual orchestrator tying both modules together end to end, plus a baseline-SUMO step neither module's own docs describe. |
| Not mentioned | `models/sub_policy.py::SubPolicy` and `agents/agent.py::Agent` / `environment/traffic_env.py::TrafficEnvironment` are unused stub/duplicate classes — `Trainer` reimplements `SubPolicy`'s logic inline rather than using it. |

**Implication:** `AGENTS.md`'s architecture section (optimizer count, training
algorithm) is stale relative to `clustering-approaches` branch HEAD. The
PPO/GAE rewrite is a substantial behavioral change (clipped policy updates,
advantage normalization, multi-epoch reuse of rollouts) that isn't reflected
in the checked-in instructions — worth a follow-up doc update if this branch
merges.
