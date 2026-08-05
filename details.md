# Traffic Signal Control using Hierarchical Reinforcement Learning (HRL)

> **Resume-ready project brief** — A complete, self-contained research project: adaptive
> traffic signal control on real-world SUMO road networks using a two-level hierarchical
> reinforcement learning agent, with a full automation pipeline and a live GUI playback tool.

---

## 1. One-Line Summary

Built a **hierarchical reinforcement learning (HRL)** system that automatically partitions a
real-world road network into regions (community detection), trains a **meta-policy + sub-policy**
agent to control every traffic signal in the network, and runs a **live SUMO GUI simulation** so
the controller's decisions can be watched in real time.

---

## 2. Problem Statement

Conventional traffic signal control (fixed-time or simple actuated) cannot adapt to dynamic,
non-uniform congestion across a large network. Centralizing control of every intersection is
computationally infeasible as networks scale. The goal was to:

1. Decompose a large road network into **semantically meaningful regions**.
2. Train a **scalable, hierarchical RL agent** — one global meta-policy that decides regional
   objectives, and a local sub-policy that controls individual intersections.
3. Optimize not just throughput but **fairness, emergency-vehicle priority, and emissions**.
4. Provide a **human-viewable live simulation** of the trained policy making decisions.

---

## 3. Tech Stack

| Layer | Technology |
|-------|-----------|
| **Simulation** | SUMO (Eclipse), TraCI Python API, `sumolib` |
| **ML framework** | PyTorch (`nn.Module`, custom training loop) |
| **Graph learning** | Community detection: `leidenalg` (Leiden), `python-louvain`, `networkx`, `igraph` |
| **Graph attention** | Custom single-layer Graph Attention Network (GAT) implementation |
| **Data/config** | Pydantic v2 (schema validation), JSON cluster files |
| **Dependency/venv** | `uv` (Astral), Python ≥ 3.14 |
| **Tooling** | `ruff` linting/formatting, git branches (`main`, `EMV`, `manhattan-dataset`) |
| **Visualization** | Matplotlib (clustering + training-loss curves), `sumo-gui` (live playback) |

---

## 4. High-Level Architecture

The project is split into **two independent modules**, glued together by an end-to-end pipeline.

```
region-splitting/                advesarial/
┌─────────────────────┐         ┌──────────────────────────────────────┐
│ SUMO Network        │         │ META-POLICY (global)                 │
│   ↓                 │         │  TransformerEncoder (cluster states) │
│ Leiden/Louvain      │──JSON──→│    ↓                                │
│   ↓                 │         │  SubGoalGenerator (LSTM) → G_t       │
│ Cluster JSON        │         │    ↓                                │
└─────────────────────┘         │ SUB-POLICY (local)                   │
                                │  LocalEncoder (MLP)                  │
                                │    ↓                                │
                                │  GATLayer (graph attention)          │
                                │    ↓                                │
                                │  ActorCritic → phase + value V(s)    │
                                │    ↓                                │
                                │  SUMO Environment (TraCI)            │
                                └──────────────────────────────────────┘
```

**Two-level decision making (HiLight framework):**

- **Meta-policy (global):** reads a 10-dim state per *region*, encodes it with a
  TransformerEncoder (d_model=128, 8 heads, 6 layers, learnable global token), and an LSTM-based
  SubGoalGenerator emits a **goal vector** `G_t` per region capturing regional traffic objectives.
- **Sub-policy (local):** per *intersection*, a 2-layer MLP (`10 → 64 → 64`) local encoder, a
  **graph attention layer** that aggregates neighbor states over the intersection adjacency graph,
  and an Actor-Critic head that picks a discrete signal phase for each traffic light.

---

## 5. Observation & Action Spaces

**Observation (10 features per intersection):** vehicle count, queue length, lane occupancy, flow,
stop count, waiting time, average speed, intersection pressure, congestion ratio, and delay —
read live from SUMO via TraCI.

**Action:** discrete selection of one valid traffic-light phase per intersection, enforced through
a **safe phase-switching protocol** (mandatory yellow-light transitions with a minimum green time)
to keep the policy physically realistic.

---

## 6. Multi-Objective Reward Design

The reward is a weighted sum of four objectives:

```
R = 0.4 · R_efficiency + 0.2 · R_fairness + 0.3 · R_emergency + 0.1 · R_emission
```

| Component | What it captures |
|-----------|------------------|
| **Efficiency** | Reduction in queue length and waiting time (Δ queue, Δ wait), local throughput |
| **Fairness** | Intersection-level fairness via **Jain's index**, variance penalty, max-wait penalty, *envy-free* constraint, pedestrian wait/conflict penalties |
| **Emergency** | Total waiting time of emergency vehicles (ambulance/fire priority) |
| **Emission** | Stop-count proxy — fewer stops imply lower emissions |

Both states and rewards are normalized with **running mean/variance** statistics.

---

## 7. Datasets / Scenarios

Five SUMO network datasets are supported, four of which are used in the RL pipeline:

| Scenario | Description | Status |
|----------|-------------|--------|
| `manhattan` | Real OSM-derived Manhattan grid, large TLS count | Trained + evaluated |
| `cologne8` | Real Cologne TAPAS Cologne network | Trained + evaluated |
| `ingolstadt21` | Ingolstadt sub-network | Supported |
| `arterial4x4` | Synthetic arterial network, **1400 demand route files** | Supported |
| `grid4x4` | Synthetic grid — **no traffic lights**, excluded from RL | Documented |

Each map has: a `{map}.sumocfg`, a Leiden cluster file, and a per-map model folder
`models/{map}/run_<timestamp>/checkpoint_ep*.pth`.

---

## 8. End-to-End Pipeline (`pipeline.py`)

A single CLI automates the full workflow for any scenario:

```
python pipeline.py --scenario cologne8                 # full run
python pipeline.py --scenario arterial4x4 --route 42   # pick a demand route
python pipeline.py baseline | cluster | copy | train | eval   # individual steps
```

5 steps: **(1)** baseline actuated SUMO run → **2)** Leiden region splitting → **(3)** copy cluster
JSON → **(4)** HRL training (`sample.py`) → **(5)** evaluation. Uses `uv`-managed environment,
passes `TRAFFIC_SCENARIO`/`TRAFFIC_ROUTE`/episode config via environment variables.

---

## 9. Live GUI Playback (`run_gui.py`)

A dedicated runner loads a trained model and plays it back in `sumo-gui`:

- **Interactive map selection** — auto-discovers maps that have a sumocfg, a cluster file, and a
  trained model, then presents a numbered menu (or `--scenario`).
- **Per-map model loading** — picks the latest run under `models/{map}/run_*`, auto-matching the
  checkpoint's architecture (M clusters, subgoal dim) to avoid silent shape mismatches.
- **Human-viewable realtime playback** — configurable GUI delay (`--delay`, default 500 ms,
  1000 ms ≈ realtime), with an optional `--verbose` mode printing the phase chosen for every
  intersection at each step.

```bash
uv run python advesarial/run_gui.py --scenario cologne8 --steps 3000 --delay 1000
```

---

## 10. Key Metrics & Results (from `metrics.txt`)

**Evaluation on cologne8** (greedy policy, 500 steps):

| Run | Completed Vehicles | Avg Queue | Avg Waiting | Avg Travel Time |
|-----|-------------------|-----------|-------------|-----------------|
| `run_20260804_181646` | 113 | 252.75 | 24.50 s | 147.27 s |
| `run_20260804_191558` (latest) | 115 | **193.51** | **13.96 s** | **85.57 s** |

→ Second training run improved avg travel time **~42%** and queue length **~23%** versus the first.

**Earlier manhattan-era runs** also showed improvement across sessions (queue length −15.7%,
waiting time −14.9%, peak queue −22.5% between consecutive runs).

**Training (REINFORCE):** 4 independent Adam optimizers, gradient clipping 0.5, batch size 16,
γ = 0.99, checkpoints saved every N episodes with loss/reward plots per run.

---

## 11. Algorithms Implemented

| # | Algorithm | Purpose |
|---|-----------|---------|
| 1 | **Leiden community detection** (active) | Partition network into regions; dynamic traffic-weighted edges |
| 2 | Louvain community detection | Alternative partitioning (implemented) |
| 3 | **TransformerEncoder** + learnable global token | Cluster-level state encoding |
| 4 | **LSTM SubGoalGenerator** | Emit per-region goal vector `G_t` |
| 5 | **Graph Attention Network (GAT)** | Neighborhood aggregation over intersection graph |
| 6 | **Actor-Critic (REINFORCE)** | Phase selection policy + value baseline |
| 7 | **Multi-objective reward** | Efficiency + fairness (Jain) + emergency + emissions |
| 8 | **Safe phase switching** | Yellow-light transitions with min-green enforcement |
| 9 | **Shannon phase entropy** | Characterize signal-switching behavior |
| 10 | **Dynamic edge weighting** | Traffic-responsive graph weights for clustering |

---

## 12. Project Structure (highlights)

```
Traffic/
├── pipeline.py              # End-to-end automation for all scenarios (uv)
├── scenarios/               # {map}/{map}.sumocfg + demand routes + stats
├── models/                  # {map}/run_<timestamp>/checkpoint_ep*.pth
├── region-splitting/        # Leiden/Louvain clustering module
│   ├── main.py
│   └── services/ (traci, louvian, leiden)
├── advesarial/              # RL module
│   ├── run_gui.py           # Live SUMO GUI playback (map selection)
│   ├── src/
│   │   ├── sample.py        # Training entry point
│   │   ├── evaluate.py      # Evaluation
│   │   ├── agents/          # ActorCritic
│   │   ├── models/          # Transformer, GAT, LSTM, LocalEncoder
│   │   ├── environment/     # SUMO env wrapper + reward
│   │   ├── services/        # TraCI wrapper
│   │   └── memory/          # ReplayBuffer, PhaseTracker
├── papers/                  # HiLight reference paper, Springer template
├── instructions.md          # Runbook (pipeline + GUI) for all scenarios
└── summary.md               # Deep codebase analysis & bug tracker
```

---

## 13. Engineering Highlights & Challenges

- **Scalability via hierarchy** — instead of one centralized policy over hundreds of signals, the
  meta-policy reasons over a small number of regions while the sub-policy parallelizes control per
  intersection.
- **Checkpoint–architecture reconciliation** — the evaluation runner introspects each checkpoint's
  tensor shapes (M clusters, subgoal dim) before constructing models, eliminating opaque
  size-mismatch crashes when cluster files change between runs.
- **Path-robust tooling** — `run_gui.py` anchors itself to its own directory so it runs from the
  repo root or the module root, keeping the fragile relative-path convention consistent.
- **Safe control physics** — the agent never snaps between conflicting phases; every switch routes
  through a yellow phase with minimum green dwell time, so the learned policy is deployable, not
  just a toy.
- **Known research frontier** — vanilla REINFORCE training can diverge (documented in `metrics.txt`
  / `summary.md`); subsequent sessions added normalization and tighter clipping, stabilizing the
  actor-critic loss (AC loss ≈ 1.0 in later runs vs >100k earlier).

---

## 14. How to Reproduce

```bash
uv sync && source .venv/bin/activate

# Full pipeline for one scenario
uv run python pipeline.py --scenario cologne8 --episodes 10 --episode-steps 3000

# Live GUI playback of the trained policy
uv run python advesarial/run_gui.py --scenario cologne8 --steps 3000 --delay 1000
```

---

## 15. Skills Demonstrated

- **Reinforcement Learning:** policy gradients, actor-critic, hierarchical RL, reward shaping,
  state normalization, stability debugging.
- **Deep Learning (PyTorch):** Transformers, LSTMs, GATs, MLPs, custom training loops,
  checkpointing.
- **Graph algorithms:** community detection (Leiden/Louvain), modularity, dynamic edge weighting.
- **Simulation & tooling:** SUMO, TraCI, Python automation pipelines, CLI design, `uv`.
- **Research engineering:** paper-to-code implementation, multi-objective design, documentation,
  reproducible experiments.

---

*Reference paper: **"HiLight: A Hierarchical Reinforcement Learning Framework"** (in `papers/`).*
