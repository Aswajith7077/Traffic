# Plan: Episode-Based Training (1000s per episode) + New Model Folder

## Context / Problem

Current training in `advesarial/src/sample.py` is **not episodic**:

- `main()` (sample.py:375) loops `for t in range(1000)` over a **single continuous SUMO
  simulation** started once at import (sample.py:21-25) and never reset between iterations.
- That is 1000 wall-clock sim-seconds of one traffic day, not 1000 episodes.
- `evaluate.py` keeps the 500-step protocol as a side/reference metric (no changes to the
  eval step count).

Goal: train each **episode = 1000 sim-seconds of fresh demand**, restarting SUMO per episode,
and store all checkpoints of a training session in **one dedicated folder**.

## Findings That Shape the Plan

| # | Finding | Location |
|---|---------|----------|
| 1 | One continuous SUMO run, never reset | `sample.py:21-25`, main loop `sample.py:375` |
| 2 | `Environment.reset()` is broken — returns `self.get_observations()` which does not exist; only zeroes `t`; does not reset reward stats | `environment.py:182-184` |
| 3 | `traci_service.reset_simulation()` exists and is correct (close + restart SUMO, re-inits phase state) | `services/traci.py:83-85` |
| 4 | `ReplayBuffer` samples across the whole accumulated history (off-policy for REINFORCE); no `clear()` | `memory/replay_buffer.py` |
| 5 | `save_models()` stamps a new `run_<timestamp>` folder on **every** save → a session fragments into one folder per checkpoint (plus per-save plots) and seconds-collisions are possible; `evaluate.py` auto-select only scans flat `models/run_*` (non-recursive) | `sample.py:262-264`, `evaluate.py:212-215` |
| 6 | All model components are stateless across calls (Transformer + positional, GAT, LocalEncoder; LSTM gets fresh hidden each `_find_global_observation`) | `models/`, `services/encoders/` |

## Changes

### 1. Episode loop in `advesarial/src/sample.py`

Replace `main()`:

```python
EPISODE_STEPS = int(os.environ.get("TRAFFIC_EPISODE_STEPS", "1000"))
TOTAL_EPISODES = int(os.environ.get("TRAFFIC_EPISODES", "10"))
SAVE_EVERY = int(os.environ.get("TRAFFIC_SAVE_EVERY", "10"))

def main():
    run_dir = create_run_dir()              # ONE session folder, created once
    try:
        for ep in range(TOTAL_EPISODES):
            reset_episode()
            for t in range(EPISODE_STEPS):
                execute()
                sample()
            if (ep + 1) % SAVE_EVERY == 0 or ep == TOTAL_EPISODES - 1:
                save_models(run_dir, ep + 1)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving models...")
        save_models(run_dir, current_ep)
    except Exception as e:
        print(f"\nTraining error: {e}. Saving models...")
        save_models(run_dir, current_ep)
        raise
    finally:
        save_models(run_dir, TOTAL_EPISODES)
        print("Training completed. Final models saved.")
```

`reset_episode()`:

```python
def reset_episode():
    traci_service.reset_simulation()   # restart SUMO: fresh 1000s of demand
    environment.reset()                # zero timers + reward running stats
    buffer.clear()                     # REINFORCE stays on-policy within the episode
```

Notes:
- `execute()/sample()` already ignore `done`, so no reliance on `env.max_t` (still 3600).
- `traci_service.reset_simulation()` reloads phase state; keep `adjacency_list`,
  `tls_set`, `clusters` (same network each episode).
- Keep the status-only logging convention already in place (no per-step prints).

### 2. Fix `Environment.reset()` (`environment.py`)

- Return `self.traci_service.get_observations()` (the missing method is on `TraciService`).
- Reset `self.t = 0`, `self.reward_mean/var`, `self.reward_count`, `self.prev_queue`,
  `self.prev_wait`.

### 3. `ReplayBuffer.clear()` (`memory/replay_buffer.py`)

```python
def clear(self):
    self.buffer.clear()
```

So each 1000s episode starts with an empty on-policy buffer.

### 4. Models → one folder per training session (`sample.py` + `evaluate.py`)

**Problem with per-save timestamps:** the current `save_models()` stamps a *new*
`run_<YYYYMMDD_HHMMSS>` folder on **every** call (sample.py:262-264). A single training
session therefore fragments into one `run_*` folder *per checkpoint*, plus per-save plot
timestamps, and seconds-collisions are possible. `evaluate.py` auto-select (newest folder)
then only ever sees the final checkpoint.

**Revised scheme — session folder created ONCE, episode-indexed checkpoints inside:**

```
models/<scenario>/run_<sessionts>/        # created once at training start (config.SCENARIO)
  checkpoint_ep010.pth                     # ALL model states + optimizer states + beta1/beta2
  checkpoint_ep020.pth
  ...
  checkpoint_ep100.pth                    # final
  plots/
    meta_loss_ep010.png
    subpolicy_loss_ep010.png
    rewards_ep010.png
  metrics.txt                           # appended snapshots (unchanged)
```

- `save_models(run_dir, ep)` → writes a single `checkpoint_ep<ep:03d>.pth` containing every model
  state dict **and** the optimizer states + `beta1`/`beta2`, inside the **existing** session
  folder (no timestamp regeneration).
- `create_run_dir()` → `../models/<scenario>/run_<YYYYMMDD_HHMMSS>/`, created exactly once in
  `main()`. Scenario from `config.SCENARIO` (defaults `manhattan`).
- `load_models(model_dir, ep=None)` → loads the given checkpoint, or the highest episode index
  in that run folder by default (for eval/resume).
- `evaluate.py` auto-select: when no `--model-dir`, scan `../models/<scenario>/run_*/` (the
  newest), then load its latest `checkpoint_ep*.pth`.

### 5. Evaluation stays at 500s (side reference)

- No change to eval step count. `pipeline.py eval` keeps passing `--steps 500`.
- The 500s run is a quick side-reference; the checkpoint chosen is the latest one in the
  newest session folder.

### 6. Pipeline wiring (`pipeline.py`)

- Add flags `--episodes` (default 10) and `--episode-steps` (default 1000) on the
  `train`/`all` command.
- Forward them via `pipeline_env()` as `TRAFFIC_EPISODES` / `TRAFFIC_EPISODE_STEPS` /
  `TRAFFIC_SAVE_EVERY`.
- `eval` step unchanged (500s).

## Defaults (open for confirmation)

- `TOTAL_EPISODES = 10` (each 1000s episode ≈ 2-3 min wall on arterial4x4 → ~30 min/run).
- One session folder `models/<scenario>/run_<sessionts>/`, no per-checkpoint timestamps.
- `SAVE_EVERY = 10` → one checkpoint per 10 episodes + final save per run folder.

## Out of scope

- Full-horizon (3600s) evaluation / paper-comparison protocol — deliberately deferred;
  500s eval remains the reference.
- Training-loss divergence / reward shaping issues listed in `summary.md`.
- `grid4x4` scenario (network has no traffic lights, RL cannot run).