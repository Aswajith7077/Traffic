# AGENTS.md

## Project Overview

Hierarchical reinforcement learning for adaptive traffic signal control using SUMO. Two independent modules — no shared package manager or monorepo tooling:

- **`region-splitting/`** — Partitions a traffic network into clusters via Louvain/Leiden community detection. Run first; produces cluster JSON files.
- **`advesarial/`** — RL training and evaluation (intentional misspelling). Consumes cluster output from `region-splitting`.

## Setup

- **SUMO** installed with `SUMO_HOME` env var set. Both modules `sys.exit()` if missing.
- **uv** manages dependencies at the repo root. Activate the venv before running anything:
  ```
  source .venv/bin/activate
  ```
- All dependencies (both modules) are in `pyproject.toml`. Do NOT use the per-module `requirements.txt` files.

## Running

### Region Splitting (run first)

```bash
cd region-splitting && python main.py
```

Defaults to `"osm"` network. Edit `main()` for `"simple"`. Produces `clusters/leiden/osm_clusters.json` (Louvain is commented out).

### Training (Adversarial)

```bash
cd advesarial && python src/sample.py
```

**Do NOT `cd advesarial/src`.** The scripts must run from `advesarial/` so file paths resolve (e.g. `sumo/osm.sumocfg` → `advesarial/sumo/osm.sumocfg`). Python adds `src/` to `sys.path` automatically since the script lives there, making bare imports (`from memory import ReplayBuffer`) work.

Saves models to `advesarial/models/run_<timestamp>/`.

### Evaluation

```bash
cd advesarial && python src/evaluate.py --model-dir ../models/run_YYYYMMDD_HHMMSS --steps 500
```

Omit `--model-dir` to auto-select the latest run.

### Lint

```bash
ruff check . && ruff format .
```

Config in `pyproject.toml`: line-length=120, rules E/F/I/W.

## Critical Working Directory Convention

All scripts use **relative paths from the module root**:
- `sumo/osm.sumocfg` — relative to module root (`advesarial/` or `region-splitting/`)
- `clusters/louvian/osm_clusters.json` — relative to module root
- `../models/` — relative to `advesarial/src/` (resolves to `advesarial/models/`)

**Cluster JSON transfer is manual.** After running region-splitting, copy cluster files from `region-splitting/clusters/` to `advesarial/clusters/`.

## Architecture (Adversarial Module)

Entry point: `advesarial/src/sample.py` (has `main()`)

- **Meta Policy**: TransformerEncoder → SubGoalGenerator (LSTM). Produces a subgoal vector from cluster-level traffic states.
- **Sub Policy**: LocalEncoder (MLP) → GATLayer (graph attention) → ActorCritic (MLP). Produces per-intersection signal actions.
- **Environment**: Wraps SUMO via TraCI. Observations are 10-dim per intersection.
- **Training**: REINFORCE-style with 4 Adam optimizers (transformer, subgoal, local encoder, GAT). Gradient clipping at 0.5. Models saved every 100 episodes.
- **`config.py` singleton**: Module-level `config = Config("clusters/louvian/osm_clusters.json")` runs at import time. Changing cluster source requires editing this line.

## Two SUMO Network Variants

- `"osm"` — real-world OSM-derived network (both modules)
- `"simple"` — synthetic small network (region-splitting only)

## Known Issues

- **`train.py` is broken** — `from agents import Actor` crashes (only `ActorCritic` is exported). Top-level code runs on import. Use `sample.py` instead.
- **`PhaseTracker.get_entropy()`** calls `compute_phase_entropy()` without importing it — `NameError` if used. Not called in training loop.
- **`ReplayBufferItem` schema** (Pydantic) is unused. `ReplayBuffer.add()` takes separate args, not a single item.
- **`advesarial/src/models/encoder/`** has no `__init__.py` — works via parent relative imports but is fragile.
- **Training loss diverges** — AC loss can explode from ~258 to >129k within 385 steps. See `metrics.txt`.
- **`visualizations/`** is gitignored but created at runtime by both modules.
- **`summary.md`** at repo root has detailed bug list and architecture analysis if you need deeper context.
