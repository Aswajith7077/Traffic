"""Orchestrates the reduced-scale (but real) training campaign for the paper's
Table 1/2 and figures: with-EMV, without-EMV, and the HiLight-equivalent
ablation (fairness/EMV weights zeroed), each over a few seeds. Run from the
repo root: `python paper_metrics/run_experiments.py`.
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO = "cologne8"
EPISODES = 5  # 25 PPO updates/run (ROLLOUT_LENGTH=240) instead of 5 in the first pass
EPISODE_STEPS = 1200

# name -> (w1, w2, w3, use_emv_routing, seeds)
CONFIGS = {
    "with_emv": (0.6, 0.3, 0.1, True, [0, 1, 2]),
    "without_emv": (0.6, 0.3, 0.1, False, [0, 1, 2]),
    "hilight_ablation": (1.0, 0.0, 0.0, False, [0]),
}


def run_one(config_name, seed, w1, w2, w3, use_emv):
    env = os.environ.copy()
    env.update(
        {
            "TRAFFIC_SCENARIO": SCENARIO,
            "REWARD_W1": str(w1),
            "REWARD_W2": str(w2),
            "REWARD_W3": str(w3),
            "USE_EMV_ROUTING": "1" if use_emv else "0",
            "USE_ACCIDENTS": "0",
        }
    )
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "paper_metrics", "train_one.py"),
        "--config-name",
        config_name,
        "--seed",
        str(seed),
        "--episodes",
        str(EPISODES),
        "--episode-steps",
        str(EPISODE_STEPS),
    ]
    print(f">>> training {config_name} seed={seed} (w1={w1}, w2={w2}, w3={w3}, emv={use_emv})", flush=True)
    result = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "advesarial"), env=env, capture_output=True, text=True)
    print(result.stdout[-1500:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"Training failed: {config_name} seed={seed}")

    run_dir = None
    for line in result.stdout.splitlines():
        if line.startswith("RUN_DIR "):
            run_dir = line.split(" ", 1)[1].strip()
    return run_dir


def main():
    manifest = {}
    for config_name, (w1, w2, w3, use_emv, seeds) in CONFIGS.items():
        manifest[config_name] = []
        for seed in seeds:
            run_dir = run_one(config_name, seed, w1, w2, w3, use_emv)
            manifest[config_name].append({"seed": seed, "run_dir": run_dir, "use_emv_routing": use_emv})

    manifest_path = os.path.join(REPO_ROOT, "paper_metrics", "train_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"DONE -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
