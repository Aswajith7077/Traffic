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

## Running a trained model in the SUMO GUI (live simulation)

`advesarial/run_gui.py` loads a trained model and plays it back live in `sumo-gui` so the
signal decisions are visible in real time.

Run it with `uv` from the `advesarial/` directory (the script uses paths relative to it — do NOT `cd advesarial/src`):

```bash
cd advesarial
uv run python run_gui.py --scenario cologne8 --steps 500 --delay 1000 --verbose
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--scenario` | interactive menu | Map to run (`cologne8`, `manhattan`, `ingolstadt21`, `arterial4x4`, `grid4x4`). Omitted → you pick from a numbered menu. |
| `--steps` | `500` | Max simulation steps to run (each step = 1 sim-second) |
| `--delay` | `500` | ms per step in the GUI. `500` ≈ 2x realtime; `1000` ≈ realtime; higher = slower / easier to watch |
| `--model-dir` | latest run | Specific run under `models/<map>/` to use, e.g. `run_20260804_191558` |
| `--verbose` | off | Print the phase chosen for every intersection each step |

### All 5 scenarios, with different max time steps

```bash
cd advesarial

# cologne8 — 3000-step run
uv run python run_gui.py --scenario cologne8 --steps 3000 --delay 1000

# manhattan — 2000-step run
uv run python run_gui.py --scenario manhattan --steps 2000 --delay 800

# ingolstadt21 — 1000-step run
uv run python run_gui.py --scenario ingolstadt21 --steps 1000 --delay 1000

# arterial4x4 — 500-step run, pick the latest trained model explicitly
uv run python run_gui.py --scenario arterial4x4 --steps 500 --delay 1000

# grid4x4 — note: no traffic lights in this network, so it cannot be run here
uv run python run_gui.py --scenario grid4x4 --steps 500 --delay 1000
```

Notes:

- The map must have a **trained model** under `models/<map>/run_*/`. Currently only `cologne8` and `manhattan` have trained runs, so `ingolstadt21`, `arterial4x4`, and `grid4x4` will report "no trained model" until you train them with `pipeline.py`.
- `grid4x4` was excluded from the pipeline because it has no traffic lights, so it will never run in the GUI.
- No `--scenario` argument → interactive map-selection menu; just pick the number.

## Notes

- Run everything from the repo root. Do NOT `cd advesarial/src` — relative paths break.
- `grid4x4` is excluded (no traffic lights in the network).
- Use `python pipeline.py train` for training — `advesarial/src/train.py` is broken.



# Full pipeline (cluster + copy + train + eval)
uv run python pipeline.py --scenario cologne8 train --episodes 20 --episode-steps 3600 --save-every 10

