# Instructions: Running the Traffic Signal Pipeline

End-to-end pipeline (baseline → region splitting → cluster copy → RL training → evaluation) for the included SUMO scenarios.

## Prerequisites

- SUMO installed with `SUMO_HOME` set
- uv-managed venv at the repo root:

```bash
source .venv/bin/activate
```

## Run the full pipeline for each map

```bash
# cologne8
python pipeline.py --scenario cologne8

# manhattan (default scenario)
python pipeline.py --scenario manhattan

# ingolstadt21
python pipeline.py --scenario ingolstadt21

# arterial4x4 (has 1400 demand route files; pick a route)
python pipeline.py --scenario arterial4x4 --route 42
```

Each full run executes: baseline SUMO simulation → Leiden region splitting → copy cluster JSON to `advesarial/clusters/` → train → evaluate.

## Step-by-step pipeline (per map)

```bash
# === cologne8 ===
python pipeline.py --scenario cologne8 baseline
python pipeline.py --scenario cologne8 cluster
python pipeline.py --scenario cologne8 copy
python pipeline.py --scenario cologne8 train --episodes 10 --episode-steps 3000
python pipeline.py --scenario cologne8 eval

# === manhattan ===
python pipeline.py --scenario manhattan baseline
python pipeline.py --scenario manhattan cluster
python pipeline.py --scenario manhattan copy
python pipeline.py --scenario manhattan train --episodes 10 --episode-steps 1000
python pipeline.py --scenario manhattan eval

# === ingolstadt21 ===
python pipeline.py --scenario ingolstadt21 baseline
python pipeline.py --scenario ingolstadt21 cluster
python pipeline.py --scenario ingolstadt21 copy
python pipeline.py --scenario ingolstadt21 train --episodes 10 --episode-steps 1000
python pipeline.py --scenario ingolstadt21 eval

# === arterial4x4 (with demand route 42) ===
python pipeline.py --scenario arterial4x4 --route 42 baseline
python pipeline.py --scenario arterial4x4 --route 42 cluster
python pipeline.py --scenario arterial4x4 --route 42 copy
python pipeline.py --scenario arterial4x4 --route 42 train --episodes 10 --episode-steps 1000
python pipeline.py --scenario arterial4x4 --route 42 eval
```

## Training-only / eval-only

```bash
python pipeline.py --scenario cologne8 train --episodes 20 --save-every 10
python pipeline.py --scenario cologne8 eval
```

## Where models are stored

Models are saved per map to:

```
advesarial/models/<map>/run_<timestamp>/
```

e.g. `advesarial/models/cologne8/run_20260804_120000/`.

Evaluation auto-selects the latest run folder for the scenario. To evaluate a specific run:

```bash
python pipeline.py --scenario cologne8 eval   # latest
# or directly:
cd advesarial && python src/evaluate.py --model-dir ../models/cologne8/run_20260804_120000 --steps 500
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | `manhattan` | `manhattan`, `arterial4x4`, `cologne8`, `ingolstadt21` |
| `--route` | none | Demand route file index (required for `arterial4x4`) |
| `--episodes` | `10` | Number of training episodes |
| `--episode-steps` | `1000` | Sim-seconds per episode |
| `--save-every` | `10` | Save a checkpoint every N episodes |

## Notes

- Run everything from the repo root. Do NOT `cd advesarial/src` — relative paths break.
- `grid4x4` is excluded (no traffic lights in the network).
- Use `python pipeline.py train` for training — `advesarial/src/train.py` is broken.
