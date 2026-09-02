# AGENTS.md

## Project Overview

Hierarchical reinforcement learning for adaptive traffic signal control using SUMO. Two independent modules — no shared package manager or monorepo tooling:

- **`region-splitting/`** — Partitions a traffic network into regions via community detection (Louvain/Leiden), density clustering (DBSCAN), or their combinations. Run first; produces cluster JSON files.
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
cd region-splitting && python main.py --scenario manhattan --method dbscan
```

**Partitioning is pluggable.** `--method` (via `services/registry.py`) accepts
`leiden`, `louvian`, `dbscan`, and the four hybrid community-detection +
clustering combinations `dbscan_louvian`, `dbscan_leiden`, `louvian_dbscan`,
`leiden_dbscan`. `dbscan` auto-tunes `eps`/`min_samples` when not supplied.

To generate every method × scenario in one shot (also copies into
`advesarial/clusters/<method>/`):

```bash
cd region-splitting && python generate_clusters.py            # all methods × 5 scenarios
cd region-splitting && python generate_clusters.py --method dbscan_leiden --scenario cologne8
```

The production DBSCAN partitions can be regenerated with
`python generate_dbscan_clusters.py` (same auto-tuned pipeline).

### Training (Adversarial)

```bash
cd advesarial && python src/sample.py
```

**Do NOT `cd advesarial/src`.** The scripts must run from `advesarial/` so file paths resolve (e.g. `../scenarios/manhattan/manhattan.sumocfg` → `advesarial/../scenarios/manhattan/manhattan.sumocfg`). Python adds `src/` to `sys.path` automatically since the script lives there, making bare imports (`from memory import ReplayBuffer`) work.

The scenario is selected by the `TRAFFIC_SCENARIO` env var (`src/config.py`, default `manhattan`; others: `cologne8`, `ingolstadt21`, `arterial4x4`, `grid4x4`). Training/eval nets live under `../scenarios/<scenario>/`; the `advesarial/sumo/osm.sumocfg` files are **not** used by `sample.py`.

Saves models to `advesarial/models/run_<timestamp>/`.

### Evaluation

```bash
cd advesarial && python src/evaluate.py --model-dir ../models/run_YYYYMMDD_HHMMSS --steps 500
```

Omit `--model-dir` to auto-select the latest run.

### Single-scenario pipeline (cologne8)

```bash
python train_eval_cologne8.py --cluster-method louvian_dbscan --episodes 10 --episode-steps 1000 --steps 500
```

Runs cluster → copy → train → evaluate for cologne8 only, then reports average
travel time and average delay time in `advesarial/metrics.txt`.

### Lint

```bash
ruff check . && ruff format .
```

Config in `pyproject.toml`: line-length=120, rules E/F/I/W.

## Critical Working Directory Convention

All scripts use **relative paths from the module root**:
- `../scenarios/<TRAFFIC_SCENARIO>/<TRAFFIC_SCENARIO>.sumocfg` — the net loaded by `advesarial/src/sample.py` (default `manhattan`)
- `advesarial/sumo/osm.sumocfg` and `region-splitting/sumo/osm/osm.sumocfg` — relative to module root; osm is used by region-splitting, not by `sample.py`
- `clusters/louvian/osm_clusters.json` — relative to module root
- `../models/` — relative to `advesarial/src/` (resolves to `advesarial/models/`)

**Cluster JSON transfer is manual.** After running region-splitting, copy cluster files from `region-splitting/clusters/` to `advesarial/clusters/`. `generate_clusters.py` / `generate_dbscan_clusters.py` do this automatically; the pipeline's `copy` step is a safety net.

## Architecture (Adversarial Module)

Entry point: `advesarial/src/sample.py` (has `main()`)

- **Meta Policy**: TransformerEncoder → SubGoalGenerator (LSTM). Produces a subgoal vector from cluster-level traffic states.
- **Sub Policy**: LocalEncoder (MLP) → GATLayer (graph attention) → ActorCritic (MLP). Produces per-intersection signal actions.
- **Environment**: Wraps SUMO via TraCI. Observations are 10-dim per intersection.
- **Training**: REINFORCE-style with 4 Adam optimizers (transformer, subgoal, local encoder, GAT). Gradient clipping at 0.5. Models saved every 100 episodes.
- **`config.py` singleton**: Module-level `config = Config(_resolve_cluster_path())` runs at import time. The cluster method is selected by `CLUSTER_METHOD` env var (default `dbscan`). `_resolve_cluster_path()` loads `clusters/<method>/{SCENARIO}_clusters.json`, honoring the explicit method verbatim (so hybrid partitions like `leiden_dbscan` can be forced) and falling back to `dbscan` → `leiden` when the file is missing. DBSCAN is the production partition — it produces coarse, spatially compact regions (manhattan: 72 regions, M≈66) that the meta-policy was designed for, unlike Leiden's ~1974 singletons.

## Two SUMO Network Variants

- `"osm"` — real-world OSM-derived network (both modules)
- `"simple"` — synthetic small network (region-splitting only)

## Traffic Light Timings

All net files have been normalized to fixed phase durations (green ≤ 5s, yellow ≤ 3s; `minDur`/`maxDur` removed, osm tlLogics converted from `actuated` to `static`). This removes the previous `maxDur="50"` cap (and up to 82s manhattan greens) that forced ~50s waits.

`TraciService.__normalize_phase_durations()` re-applies the same clamp at startup as defense-in-depth, so an unpatched net cannot reintroduce long greens. Timing is configurable via `TraciConfig` (`green_duration`, `yellow_duration`, `min_green_steps`).

## Known Issues

- **`train.py` is broken** — `from agents import Actor` crashes (only `ActorCritic` is exported). Top-level code runs on import. Use `sample.py` instead.
- **`PhaseTracker.get_entropy()`** calls `compute_phase_entropy()` without importing it — `NameError` if used. Not called in training loop.
- **`ReplayBufferItem` schema** (Pydantic) is unused. `ReplayBuffer.add()` takes separate args, not a single item.
- **`advesarial/src/models/encoder/`** has no `__init__.py` — works via parent relative imports but is fragile.
- **Training loss diverges** — AC loss can explode from ~258 to >129k within 385 steps. See `metrics.txt`.
- **`visualizations/`** is gitignored but created at runtime by both modules.
- **`summary.md`** at repo root has detailed bug list and architecture analysis if you need deeper context.
