# Traffic Signal Control with Hierarchical Reinforcement Learning

A research implementation of **hierarchical reinforcement learning (HRL)** for adaptive traffic signal control, built on the [SUMO](https://eclipse.dev/sumo/) traffic simulator. The system partitions a real-world road network into clusters using community detection, then trains a two-level RL agent to optimize traffic flow, fairness, emergency vehicle priority, and emissions.

Based on the paper: **"HiLight: A Hierarchical Reinforcement Learning Framework"** (`papers/`).

## Prerequisites

- **Python >= 3.14**
- **SUMO** installed with `SUMO_HOME` environment variable set
- **[uv](https://docus.astral.sh/uv/)** (recommended) or pip

## Installation

```bash
git clone <repo-url> && cd Traffic

# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
pip install torch numpy traci pydantic ruff matplotlib networkx python-louvain sumolib leidenalg python-igraph
```

## Usage

### 1. Cluster the Network (run first)

```bash
cd region-splitting && python main.py
```

Partitions the SUMO network into communities using the Leiden algorithm. Produces cluster JSON files in `clusters/leiden/`.

### 2. Copy Clusters to RL Module

```bash
cp region-splitting/clusters/leiden/osm_clusters.json advesarial/clusters/leiden/
```

### 3. Train

```bash
cd advesarial && python src/sample.py
```

> **Note:** Run from `advesarial/`, not `advesarial/src/`. File paths are relative to the module root.

Saves model checkpoints to `advesarial/models/run_<timestamp>/`.

### 4. Evaluate

```bash
cd advesarial && python src/evaluate.py --model-dir ../models/run_YYYYMMDD_HHMMSS --steps 500
```

Omit `--model-dir` to auto-select the latest run.

### Lint

```bash
ruff check . && ruff format .
```

## Architecture

```
region-splitting/                advesarial/
┌─────────────────────┐         ┌──────────────────────────────────────┐
│ SUMO Network        │         │ Meta Policy                          │
│   ↓                 │         │   TransformerEncoder (clusters)      │
│ Leiden/Louvain      │──JSON──→│     ↓                               │
│   ↓                 │         │   SubGoalGenerator (LSTM) → G_t     │
│ Cluster JSON        │         │     ↓                               │
└─────────────────────┘         │ Sub Policy                          │
                                │   LocalEncoder (MLP)                │
                                │     ↓                               │
                                │   GATLayer (graph attention)        │
                                │     ↓                               │
                                │   ActorCritic → phase action + V(s) │
                                │     ↓                               │
                                │ SUMO Environment (TraCI)            │
                                │   Reward = 0.4·Eff + 0.2·Fair      │
                                │        + 0.3·Emerg + 0.1·Emiss     │
                                └──────────────────────────────────────┘
```

- **Observation:** 10-dim per intersection (vehicle count, queue, occupancy, flow, stops, waiting time, speed, pressure, congestion, delay)
- **Action:** Discrete traffic signal phase selection with safe yellow-light transitions
- **Training:** REINFORCE with 4 Adam optimizers, gradient clipping at 0.5, batch size 16

## Project Structure

```
Traffic/
├── region-splitting/       # Network clustering module
│   ├── main.py             # Entry point
│   ├── services/           # TraciService, LouvianService, LeidenService
│   ├── clusters/           # Generated cluster JSON outputs
│   └── sumo/               # SUMO network files (osm, simple)
│
├── advesarial/             # RL training module (intentional misspelling)
│   ├── src/
│   │   ├── sample.py       # Training entry point
│   │   ├── evaluate.py     # Evaluation script
│   │   ├── config.py       # Cluster JSON loader
│   │   ├── agents/         # ActorCritic
│   │   ├── models/         # TransformerEncoder, GATLayer, SubGoalGenerator, LocalEncoder
│   │   ├── environment/    # SUMO environment wrapper + reward computation
│   │   ├── services/       # TraCI wrapper (615 lines)
│   │   ├── memory/         # ReplayBuffer, PhaseTracker
│   │   ├── schema/         # Pydantic configs
│   │   └── utils/          # Loss functions, entropy computation
│   ├── clusters/           # Cluster JSON (copied from region-splitting)
│   ├── models/             # Saved checkpoints (gitignored)
│   └── sumo/               # SUMO network files
│
├── papers/                 # Reference papers (HiLight, Springer template)
├── summary.md              # Detailed codebase analysis and bug tracker
└── pyproject.toml          # uv/pip dependencies and ruff config
```

## Key Details

- **Two SUMO networks:** `"osm"` (real-world, ~1200 intersections) and `"simple"` (synthetic, testing only)
- **Reward function** is multi-objective: efficiency (queue/wait reduction), fairness (Jain's index), emergency vehicle priority, and emission proxy (stop count)
- **Models saved** every 100 episodes as individual `.pth` files + combined `training_state.pth`
- `visualizations/` and `models/` are gitignored but created at runtime

## Known Issues

See `summary.md` for a full bug list. Key items:

- `train.py` is broken — use `sample.py` for training
- Training loss can diverge quickly (AC loss: 258 → 129k+ within 385 steps)
- `PhaseTracker.get_entropy()` has a missing import (not used in training loop)

## License

Research project — see papers for citation requirements.
