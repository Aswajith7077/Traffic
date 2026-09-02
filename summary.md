# Traffic Signal Control with Hierarchical Reinforcement Learning

## Project Summary

A research implementation of **hierarchical reinforcement learning (HRL)** for adaptive traffic signal control, built on the SUMO traffic simulator. The project partitions a real-world road network into clusters using community detection algorithms, then trains a two-level RL agent (meta-policy + sub-policy) to optimize traffic flow, fairness, emergency vehicle priority, and emission reduction.

**Reference Paper:** *"HiLight: A Hierarchical Reinforcement Learning Framework"* (PDF in `papers/`)

**Author:** Aswajith S  
**Development Period:** March 28 - May 17, 2026

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Prerequisites](#2-prerequisites)
3. [Module 1: Region Splitting](#3-module-1-region-splitting)
4. [Module 2: Adversarial RL Training](#4-module-2-adversarial-rl-training)
5. [Architecture Overview](#5-architecture-overview)
6. [Algorithms Implemented](#6-algorithms-implemented)
7. [Training Pipeline](#7-training-pipeline)
8. [Reward Function](#8-reward-function)
9. [Evaluation Results](#9-evaluation-results)
10. [Known Bugs & Issues](#10-known-bugs--issues)
11. [Dead Code & Stubs](#11-dead-code--stubs)
12. [What Needs to Be Done](#12-what-needs-to-be-done)
13. [Development Timeline](#13-development-timeline)

---

## 1. Directory Structure

```
Traffic/
├── AGENTS.md                          # AI agent instructions
├── .gitignore                         # Ignores .env, .venv/, __pycache__, /models/, visualizations/
├── metrics.txt                        # Training/evaluation metrics log (307 lines)
│
├── papers/                            # Reference research papers
│   ├── HiLight_A_Hierarchical_Reinforcement_Learning_Fram (1).pdf
│   └── docs/
│       └── Springer_Nature_LaTeX_Template*.pdf  # Paper writing template (4 copies)
│
├── region-splitting/                  # MODULE 1: Network clustering
│   ├── main.py                        # Entry point
│   ├── get_edge_list.py               # Utility: SUMO edges → NetworkX graph
│   ├── requirements.txt               # networkx, python-louvain, matplotlib, traci, pydantic, sumolib, leidenalg, python-igraph
│   ├── tls_config.json                # Traffic light → lane mappings (178 lines)
│   ├── edges.json                     # 472 edge IDs from OSM network
│   ├── schema/
│   │   ├── __init__.py
│   │   └── traci_config.py            # Pydantic TraciConfig
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseClusteringService ABC
│   │   ├── traci.py                   # SUMO wrapper for clustering
│   │   ├── louvian.py                 # Louvain community detection
│   │   ├── leiden.py                  # Leiden community detection
│   │   └── dbscan.py                  # DBSCAN density clustering + metrics/diagnostics
│   ├── tests/
│   │   └── test_dbscan.py             # DBSCAN clustering tests
│   ├── clusters/                      # Generated cluster JSON outputs
│   │   ├── louvian/                   # Louvain clusters (osm + simple)
│   │   ├── leiden/                    # Leiden clusters (osm + simple)
│   │   └── dbscan/                    # DBSCAN clusters
│   ├── sumo/                          # SUMO simulation files
│   │   ├── osm/                       # Real-world OSM network (~1200+ intersections)
│   │   └── simple/                    # Synthetic small network
│   └── visualizations/                # Generated clustering PNGs
│       ├── louvian/
│       ├── leiden/
│       └── dbscan/
│
├── advesarial/                        # MODULE 2: RL training (intentional misspelling)
│   ├── requirements.txt               # torch, numpy, traci, pydantic, ruff, matplotlib
│   ├── clusters/                      # Cluster JSON consumed by RL module
│   │   ├── louvian/
│   │   ├── leiden/
│   │   ├── osm_clusters.json          # Default clusters
│   │   └── simple_clusters.json
│   ├── src/
│   │   ├── config.py                  # Cluster JSON loader
│   │   ├── sample.py                  # *** MAIN TRAINING ENTRY POINT ***
│   │   ├── train.py                   # Legacy/draft training script (broken)
│   │   ├── evaluate.py                # Model evaluation with CLI
│   │   ├── agents/
│   │   │   ├── __init__.py            # Exports ActorCritic
│   │   │   ├── actor.py               # ActorCritic nn.Module
│   │   │   └── agent.py               # Empty Agent stub
│   │   ├── environment/
│   │   │   ├── __init__.py            # Exports Environment
│   │   │   ├── environment.py         # Core: reward computation
│   │   │   └── traffic_env.py         # Empty TrafficEnvironment stub
│   │   ├── memory/
│   │   │   ├── __init__.py            # Exports ReplayBuffer, PhaseTracker
│   │   │   ├── replay_buffer.py       # Deque-based experience replay
│   │   │   └── phase_tracker.py       # Phase history tracker (broken import)
│   │   ├── models/
│   │   │   ├── __init__.py            # Exports all model classes
│   │   │   ├── gat.py                 # Graph Attention Network layer
│   │   │   ├── lstm.py                # SubGoalGenerator: LSTM + FFN
│   │   │   ├── sub_policy.py          # SubPolicy (unused in training)
│   │   │   └── encoder/
│   │   │       ├── local.py           # LocalEncoder: 2-layer MLP (10→64→64)
│   │   │       ├── positional.py      # Sinusoidal PositionalEncoder
│   │   │       └── transformer.py     # TransformerEncoder with global token
│   │   ├── schema/
│   │   │   ├── __init__.py
│   │   │   ├── encoder_config.py      # TransformerEncoderConfig
│   │   │   ├── environment_config.py  # EnvironmentConfig (unused)
│   │   │   ├── replay_buffer.py       # ReplayBufferItem (unused)
│   │   │   └── traci_config.py        # TraciConfig
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── traci.py               # *** CORE: TraCI wrapper (615 lines) ***
│   │   │   ├── lstm.py                # Empty file
│   │   │   └── encoders/
│   │   │       ├── __init__.py
│   │   │       ├── positional.py      # Duplicate PositionalEncoding
│   │   │       └── transformer.py     # Duplicate TransformerEncoding
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── compute_phase_history.py  # Shannon entropy of phases
│   │       └── loss.py               # Meta loss, AC loss, goal alignment loss
│   ├── sumo/                          # SUMO simulation files (OSM network)
│   └── models/                        # Saved model checkpoints (gitignored)
│
├── graphify-out/                      # Code graph analysis (external tool output)
│   ├── GRAPH_REPORT.md                # 713-line analysis report
│   ├── graph.html, graph.json
│   └── ...
│
└── visualizations/                    # Runtime output (gitignored)
    └── metrics/
        └── metrics.txt                # SUMO simulation metrics
```

---

## 2. Prerequisites

- **SUMO** installed with `SUMO_HOME` environment variable set (both modules `sys.exit()` if missing)
- Python 3.x with per-module `requirements.txt`
- Install: `pip install -r region-splitting/requirements.txt` and `pip install -r advesarial/requirements.txt`

---

## 3. Module 1: Region Splitting

**Purpose:** Partition a SUMO traffic network into clusters using graph-based community detection. Run first; produces cluster JSON files consumed by the RL module.

### Entry Point

```bash
cd region-splitting
python main.py  # defaults to "osm" network
```

### Algorithms

#### 3.1 Louvain Community Detection (`services/louvian.py`)

| Aspect | Details |
|--------|---------|
| **Library** | `python-louvain` (NetworkX-based) |
| **Algorithm** | Louvain modularity optimization — greedily partitions graph to maximize modularity Q |
| **Graph Construction** | Reads SUMO network via `sumolib`, builds NetworkX DiGraph, converts to weighted undirected |
| **Edge Weights** | Currently uses static lane count only (dynamic weight computation exists but is never called) |
| **Post-processing** | Merges small clusters (< `min_cluster_size=5`) into best neighboring cluster |
| **Metrics** | Modularity score, per-cluster size/edge counts |
| **Visualization** | Matplotlib + NetworkX, color-coded by community |
| **Status** | Implemented, but `compute_weights()` is dead code; weight key mismatch bug (`lane_weight` vs `static_weight`) |

#### 3.2 Leiden Community Detection (`services/leiden.py`)

| Aspect | Details |
|--------|---------|
| **Library** | `leidenalg` + `python-igraph` |
| **Algorithm** | Leiden algorithm — improved modularity optimization with guaranteed connected communities |
| **Graph Construction** | Builds igraph from SUMO network, computes dynamic edge weights from 1000 simulation steps |
| **Edge Weights** | Dynamic: weighted sum of vehicle count (α=1.0), waiting time (β=0.3), congestion ratio (γ=2.0) |
| **Post-processing** | Merges singleton clusters into best neighboring cluster |
| **Metrics** | Modularity score, per-cluster quality metrics |
| **Visualization** | Matplotlib + igraph, color-coded by community |
| **Status** | Fully implemented and active (Louvain is commented out in main.py) |

#### 3.3 DBSCAN Density Clustering (`services/dbscan.py`)

| Aspect | Details |
|--------|---------|
| **Library** | `scikit-learn` (`sklearn.cluster.DBSCAN`) |
| **Algorithm** | Density-based spatial clustering — clusters points connected by dense neighborhoods, marks isolated points as noise |
| **Input** | Road-network node coordinates (networkx graph of nodes with x/y positions) |
| **Key Parameters** | `eps` (neighborhood radius, must be tuned per network), `min_samples` (minimum neighborhood size for a core point) |
| **Noise Handling** | Noise points (label `-1`) stored under the `"-1"` cluster key; explicitly excluded from all internal metric calculations |
| **Metrics** | Silhouette, Davies-Bouldin, Calinski-Harabasz, noise fraction, cluster-size statistics (incl. Gini), density metrics; external metrics (ARI/NMI/AMI/FM/purity/V-measure) when ground truth is available |
| **Diagnostics** | `k_distance_plot_data`, `suggest_eps` (kneedle elbow), `eps_sensitivity_analysis`, `min_samples_sensitivity_analysis`, `cluster_stability_score`, `dbscan_grid_search` |
| **CLI** | `python main.py --method dbscan --eps <radius> --min-samples <n>` |
| **Visualization** | Matplotlib + NetworkX, color-coded by cluster, noise in the default color |
| **Status** | Fully implemented, tested (see `tests/test_dbscan.py`) |

#### 3.4 SUMO Wrapper for Clustering (`services/traci.py`)

- Starts/steps/resets SUMO simulation
- Computes dynamic edge weights from live traffic data
- Extracts TLS (traffic light system) IDs and edge lists
- Writes `tls_config.json` as a side effect

### Cluster Output Format

```json
{
  "clusters": {
    "0": ["node_1", "node_2", ...],
    "1": ["node_3", "node_4", ...],
    ...
  },
  "metrics": {
    "modularity": 0.85,
    "cluster_quality": {...}
  }
}
```

- **OSM network:** 487 intersections → partitions into ~20-50 clusters depending on algorithm
- **Simple network:** Small synthetic network for testing

### Known Issues (Region Splitting)

1. **Louvain weight bug:** `to_weighted_undirected()` reads `data.get("lane_weight", 1.0)` but `build_graph()` stores the attribute as `static_weight` — dynamic weights are never used
2. **Dead code:** `compute_weights()` method in `LouvianService` exists but is never called
3. **Louvain commented out:** Only Leiden is active in `main.py`

---

## 4. Module 2: Adversarial RL Training

**Purpose:** Train a hierarchical reinforcement learning agent to control traffic signals across the partitioned network.

### Entry Points

```bash
cd advesarial/src
python sample.py                    # Training (main entry point)
python evaluate.py --steps 500      # Evaluation (auto-selects latest model)
python evaluate.py --model-dir ../models/run_YYYYMMDD_HHMMSS --steps 500
```

### Key Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| Batch size | 16 | `sample.py` |
| Discount factor (γ) | 0.99 | `sample.py` |
| Meta loss weight (η₁) | 0.1 | `sample.py` |
| Sub loss weight (η₂) | 0.1 | `sample.py` |
| Transformer d_model | 128 | `sample.py` |
| Transformer nhead | 8 | `sample.py` |
| Transformer layers | 6 | `sample.py` |
| Transformer LR | 5e-5 | `sample.py` |
| SubGoal LR | 5e-4 | `sample.py` |
| Gradient clip | 0.5 | `sample.py` |
| Replay buffer capacity | 10,000 | `replay_buffer.py` |
| Phase tracker window | 20 | `phase_tracker.py` |
| Max environment steps | 3,600 | `environment.py` |
| Save interval | 100 episodes | `sample.py` |
| Total episodes | 1,000 | `sample.py` |

---

## 5. Architecture Overview

### Hierarchical RL Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    META-POLICY (Global)                      │
│                                                              │
│  Cluster States ──► TransformerEncoder ──► Global Token      │
│  (10-dim × M clusters)    │                    │             │
│                           │                    ▼             │
│                           └──────────► SubGoalGenerator      │
│                                       (LSTM + FFN)          │
│                                              │               │
│                                              ▼               │
│                                        Goal Vector G_t       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    SUB-POLICY (Local)                        │
│                                                              │
│  Intersection Obs ──► LocalEncoder ──► GATLayer ──► ActorCritic│
│  (10-dim × N nodes)    (MLP)      (Graph Attn)    (Actor+Critic)│
│                                              │               │
│                                              ▼               │
│                                    Action π(a|s) + V(s)      │
└─────────────────────────────────────────────────────────────┘
```

### Observation Space (10-dimensional per intersection)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Vehicle count | Number of vehicles in intersection area |
| 1 | Queue length | Number of halting vehicles |
| 2 | Occupancy | Lane occupancy ratio |
| 3 | Flow | Vehicle throughput |
| 4 | Stops | Number of stops |
| 5 | Waiting time | Cumulative waiting time |
| 6 | Speed | Average vehicle speed |
| 7 | Pressure | Incoming halts minus outgoing halts |
| 8 | Congestion ratio | Congested lanes / total lanes |
| 9 | Delay | Travel time delay |

### Action Space

- Discrete: select one of the valid traffic signal phases for each intersection
- Safe switching enforced: yellow-light transitions with minimum green time (5 steps)

---

## 6. Algorithms Implemented

| # | Algorithm | Location | Purpose | Status |
|---|-----------|----------|---------|--------|
| 1 | **Louvain Community Detection** | `region-splitting/services/louvian.py` | Graph partitioning into clusters | Implemented (has bugs, commented out) |
| 2 | **Leiden Community Detection** | `region-splitting/services/leiden.py` | Improved graph partitioning | Fully implemented, active |
| 3 | **DBSCAN Density Clustering** | `region-splitting/services/dbscan.py` | Density-based spatial clustering (node coordinates) | Fully implemented, tested |
| 4 | **Transformer Encoder** | `advesarial/src/models/encoder/transformer.py` | Cluster-level state encoding with learnable global token | Fully implemented |
| 5 | **LSTM Subgoal Generation** | `advesarial/src/models/lstm.py` | Produce goal vector from cluster encodings | Fully implemented |
| 6 | **Graph Attention Network (GAT)** | `advesarial/src/models/gat.py` | Per-intersection neighbor aggregation | Implemented (single-head, single-layer) |
| 7 | **Actor-Critic (REINFORCE)** | `advesarial/src/agents/actor.py` | Policy gradient with value baseline | Fully implemented |
| 8 | **Multi-Objective Reward** | `advesarial/src/environment/environment.py` | Weighted efficiency + fairness + emergency + emission | Fully implemented |
| 9 | **Safe Phase Switching** | `advesarial/src/services/traci.py` | Yellow-light transitions, min green time | Fully implemented |
| 10 | **Shannon Entropy** | `advesarial/src/utils/compute_phase_history.py` | Phase switching pattern characterization | Fully implemented |
| 11 | **Running Mean/Variance Normalization** | `sample.py`, `environment.py` | State and reward normalization | Fully implemented |
| 12 | **Jain's Fairness Index** | `advesarial/src/environment/environment.py` | Intersection-level fairness metric | Fully implemented |
| 13 | **Dynamic Edge Weighting** | `region-splitting/services/traci.py` | Traffic-responsive graph weights | Fully implemented |

---

## 7. Training Pipeline

### Flow

1. **Cluster Loading:** `config.py` loads cluster JSON → maps intersections to clusters
2. **Environment Reset:** SUMO simulation restarts, returns initial observations
3. **Per-Step Loop (`execute()`):**
   - LocalEncoder encodes each intersection's 10-dim observation → 64-dim features
   - GATLayer aggregates features over intersection adjacency graph
   - TransformerEncoder encodes cluster-level states → global token + subregion embeddings
   - SubGoalGenerator (LSTM) produces goal vector G_t from cluster embeddings
   - ActorCritic selects phase action (softmax) and estimates state value
   - TraciService applies action to SUMO (with safe phase switching)
   - Environment computes multi-component reward
   - Transition stored in ReplayBuffer
4. **Training Step (`sample()`):**
   - Sample batch from ReplayBuffer
   - Compute meta loss: MSE(G_t, [W, Q]) + η₁ × meta_reward
   - Compute AC loss: TD(0) critic loss + policy gradient actor loss
   - Compute goal alignment loss: MSE between goal halves and actual W, Q
   - Backpropagate with gradient clipping (0.5)
   - Update 4 separate Adam optimizers
5. **Checkpoints:** Models saved every 100 episodes to `../models/run_<timestamp>/`

### Saved Model Components

- `transformer.pth` — TransformerEncoder weights
- `subgoal_generator.pth` — SubGoalGenerator (LSTM) weights
- `local_encoder.pth` — LocalEncoder (MLP) weights
- `gat_layer.pth` — GATLayer weights
- `actor_critic.pth` — ActorCritic weights
- `training_state.pth` — Optimizers, running stats, beta1/beta2, episode count

---

## 8. Reward Function

The reward is a weighted multi-objective function:

```
R = 0.4 × R_efficiency + 0.2 × R_fairness + 0.3 × R_emergency + 0.1 × R_emission
```

### Components

| Component | Formula | Description |
|-----------|---------|-------------|
| **R_efficiency** | Δ(queue) + Δ(wait) + local_reward | Queue/wait reduction + per-intersection throughput |
| **R_fairness** | variance_penalty + max_wait_penalty + Jain_index + envy + ped_wait + ped_conflict | Intersection-level fairness using Jain's index |
| **R_emergency** | Emergency vehicle waiting time penalty | Priority for ambulances/fire trucks |
| **R_emission** | Stop count proxy | Fewer stops = lower emissions |

### Normalization

- Exponential moving average for reward normalization
- Running mean/variance for state normalization (separate from Environment's own normalization)

---

## 9. Evaluation Results

### Evaluation Run 1 (March 30, 2026)

```
Model: run_20260330_224316
Steps: 500
Average Queue Length: 115.65 vehicles
Average Wait Time:    7,829.17 seconds
Peak Queue Length:    231.00 vehicles
Peak Wait Time:      23,802.00 seconds
```

### Evaluation Run 2 (April 6, 2026)

```
Model: run_20260406_083214
Steps: 500
Average Queue Length:  97.49 vehicles  (↓15.7%)
Average Wait Time:     6,665.06 seconds (↓14.9%)
Peak Queue Length:     179.00 vehicles  (↓22.5%)
Peak Wait Time:       19,215.00 seconds (↓19.3%)
```

### SUMO Simulation Stats (Evaluation 2)

```
Duration: 186.57s (real time factor: 2.68)
Vehicles inserted: 403 (loaded: 488)
Running: 197, Waiting: 1
Teleports: 10 (Jam: 1, Yield: 9)
Emergency Stops: 2, Emergency Braking: 3
Average Speed: 14.01, Average Duration: 161.59s
```

### Training Loss Trends (from metrics.txt)

| Session | Steps | Meta Loss | AC Loss | Reward |
|---------|-------|-----------|---------|--------|
| Apr 6, Session 1 | 85 | 406.83 | 391.97 | -34.50 |
| | 385 | 1,067.44 | 3,043.43 | -116.50 |
| | 785 | 65,465.41 | 90,977.10 | -501.30 |
| | 985 | 5,763.03 | 83,120.46 | -134.30 |
| Apr 15, Session 1 | 85 | 322.82 | 258.01 | -13.70 |
| | 385 | 3,399.93 | 129,556.17 | -24.40 |
| | 985 | 302.43 | 76,734.68 | -124.10 |
| Apr 15, Session 2 | 85 | 981.87 | 6.99 | -32.80 |
| | 985 | 24,736.13 | 0.89 | -188.50 |

**Observation:** Training is unstable. AC loss diverges dramatically in early sessions. The third session (Apr 15) shows better AC loss stability but rewards remain negative, indicating the agent has not yet learned effective traffic control policies.

---

## 10. Known Bugs & Issues

### Critical (will crash)

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| 1 | `environment/environment.py` | `reset()` | Calls `self.get_observations()` — method does not exist on `Environment`. Should be `self.traci_service.get_observations()` |
| 2 | `memory/phase_tracker.py` | `get_entropy()` | Calls `compute_phase_entropy()` without importing it — `NameError` at runtime |
| 3 | `services/encoders/transformer.py` | `__init__` | References `config.max_len` but schema field is `max_regions` — `AttributeError` if instantiated |
| 4 | `train.py` | imports | Imports `Actor` from `agents` but only `ActorCritic` is exported — `ImportError`. Also calls `Environment(config=..., traci_config=...)` with incompatible signature |

### Medium (logic errors)

| # | File | Issue |
|---|------|-------|
| 5 | `services/louvian.py` | `compute_weights()` is never called; `to_weighted_undirected()` reads key `lane_weight` but data stores `static_weight` — falls back to 1.0 |
| 6 | `models/encoder/` | Missing `__init__.py` — works via parent relative imports but is fragile |
| 7 | `sample.py` | `ReplayBuffer` instantiated twice (line 34 and 56), first one is silently overwritten |
| 8 | `environment/environment.py` | `my_utility` set inside loop but `current_intersection` is commented out — ends up being utility of last intersection in loop |
| 9 | `sample.py` | Optimizers defined after functions that reference them as globals — fragile ordering dependency |

### Low (code quality)

| # | File | Issue |
|---|------|-------|
| 10 | `services/traci.py` | Commented-out duplicate `get_adjacency_list` (lines 583-614) |
| 11 | `sample.py` | No `close_simulation()` call on exit; `finally` block may double-save |
| 12 | Multiple | Duplicate positional encoder implementations in `models/encoder/` and `services/encoders/` |
| 13 | `evaluate.py` | `_find_global_observation()` called but result discarded |
| 14 | `environment.py` + `sample.py` | Running mean/variance normalization duplicated in both files |

---

## 11. Dead Code & Stubs

| File | Status | Notes |
|------|--------|-------|
| `agents/agent.py` | Empty stub | `Agent` class with no methods |
| `environment/traffic_env.py` | Empty stub | `TrafficEnvironment` class, all methods `pass` |
| `services/lstm.py` | Empty file | 0 lines, not imported |
| `train.py` | Legacy draft | Top-level code runs on import; incompatible with current API |
| `models/sub_policy.py` | Unused | Implemented but training uses `LocalEncoder` + `GATLayer` separately |
| `schema/environment_config.py` | Unused | `EnvironmentConfig` defined but never imported |
| `schema/replay_buffer.py` | Unused | `ReplayBufferItem` only used by broken `train.py` |
| `services/encoders/` | Unused | Alternative encoder implementations, not used by training loop |
| `region-splitting/get_edge_list.py` | Scratch file | Standalone debug script, not imported |

---

## 12. What Needs to Be Done

### Immediate Fixes (Before Training Can Succeed)

1. **Fix `Environment.reset()`** — Change `self.get_observations()` to `self.traci_service.get_observations()`
2. **Fix Louvain weight bug** — Either call `compute_weights()` or fix the key name from `lane_weight` to `static_weight`
3. **Add missing `__init__.py`** to `advesarial/src/models/encoder/`
4. **Clean up dead code** — Remove or archive `train.py`, empty stubs, duplicate encoder files
5. **Fix `PhaseTracker`** — Add missing import or remove the class if unused

### Training Stability (Critical)

6. **Investigate AC loss divergence** — Loss explodes from ~258 to >129,000 within 385 steps. Options:
   - Reduce learning rate for AC optimizer
   - Add entropy regularization to prevent premature convergence
   - Implement PPO or A2C instead of vanilla REINFORCE
   - Add value function clipping
7. **Reward is always negative** — The agent is not learning to reduce congestion. Consider:
   - Reward shaping with better baselines
   - Curriculum learning (start with simple scenarios)
   - Normalize rewards per-batch instead of running average
8. **Add missing `close_simulation()`** call in training cleanup

### Feature Improvements

9. **Extend GAT** — Currently single-head, single-layer. Multi-head GAT would improve neighborhood aggregation
10. **Implement the `SubPolicy` class properly** — Currently defined but unused; could replace the manual pipeline in `sample.py`
11. **Add proper evaluation metrics** — Compare against fixed-time baseline and actuated control
12. **Multi-seed training** — Run multiple seeds for statistical significance
13. **Add TensorBoard/WandB logging** — Current metrics.txt logging is primitive
14. **Implement proper test suite** — No tests exist currently

### Research Completeness

15. **Write the paper** — Springer Nature template is in `papers/docs/`; the HiLight paper is the reference
16. **Ablation studies** — Test each reward component independently
17. **Scalability analysis** — Test on larger networks
18. **Baselines** — Compare with fixed-time, Webster's, and actuated control

---

## 13. Development Timeline

| Date | Commits | Milestone |
|------|---------|-----------|
| **Mar 28** | 3 | First commit, Louvain clustering, Leiden clustering with benchmarks |
| **Mar 29** | 5 | Positional encoding, LSTM encoding, Actor-Critic implementation, features, formatting |
| **Mar 30** | 4 | Step function, training + backpropagation, metrics logging, formatting |
| **Apr 6** | 2 | Emergency vehicle (EMV) rewards, training metrics |
| **Apr 15** | 1 | Fairness features (Jain index, envy, pedestrian conflicts) |
| **May 17** | 1 | Paper references added |

**16 total commits, single author: Aswajith S**

---

## Summary

This project implements a **hierarchical reinforcement learning system for adaptive traffic signal control**, following the HiLight framework. The region-splitting module successfully partitions the network into clusters using Leiden community detection. The adversarial RL module implements a complete two-level architecture (Transformer meta-policy + GAT sub-policy) with a sophisticated multi-objective reward function covering efficiency, fairness, emergency priority, and emissions.

**Current status:** The architecture is fully implemented but training is unstable — AC loss diverges and rewards remain negative. Critical bugs in `Environment.reset()` and `PhaseTracker` need fixing before training can proceed reliably. The project has significant dead code from iterative development that should be cleaned up. The next major milestone is achieving stable training that outperforms fixed-time signal control baselines.
