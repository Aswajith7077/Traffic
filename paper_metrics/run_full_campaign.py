"""Generalizes run_experiments.py + run_eval.py across all 5 scenarios in the
scenarios table (cologne8, arterial4x4, grid4x4, ingolstadt21, manhattan),
after the global-state scale/variance fixes in
.claude/reasoning/training-loss/. For each scenario, trains with_emv and
without_emv (3 seeds each, same 5-episode/1200-step reduced-scale protocol as
the original cologne8-only campaign), then evaluates algorithm (seed 0,
both configs) + fixed + max_pressure, all for 1200 steps. Appends to
paper_metrics/eval_results.csv with scenario-prefixed labels
("<scenario>_with_emv", "<scenario>_fixed", ...) so the pre-fix cologne8 rows
(unprefixed labels from the original campaign) are left untouched for
historical comparison. Run from the repo root:
`python paper_metrics/run_full_campaign.py`.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_SCENARIOS = ["cologne8", "arterial4x4", "grid4x4", "ingolstadt21", "manhattan"]
EVAL_CSV_PATH = os.path.join(REPO_ROOT, "paper_metrics", "eval_results.csv")


def _has_emv(scenario):
    """cologne8 ships its own *_emv.rou.xml; the other 4 scenarios get one
    via a separate, additively-generated scenarios/<scenario>/emv/ subfolder
    (train_one.py/evaluate.py pick it up automatically when
    USE_EMV_ROUTING=1) instead of editing their original, working files."""
    direct = os.path.join(REPO_ROOT, "scenarios", scenario, f"{scenario}_emv.rou.xml")
    separate = os.path.join(REPO_ROOT, "scenarios", scenario, "emv", f"{scenario}_emv.rou.xml")
    return os.path.exists(direct) or os.path.exists(separate)


def _duplicate_eval_row(scenario, from_label, to_label):
    """For a scenario with no EMVs, reuse without_emv's real eval row as
    with_emv's row instead of re-running an identical training+eval cycle."""
    with open(EVAL_CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    fieldnames = rows[0].keys() if rows else None
    source = None
    for row in reversed(rows):
        if row["label"] == f"{scenario}_{from_label}":
            source = row
            break
    if source is None:
        raise RuntimeError(f"No eval row found for {scenario}_{from_label} to duplicate")

    new_row = dict(source)
    new_row["label"] = f"{scenario}_{to_label}"
    new_row["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_row["use_emv_routing"] = "False"  # no EMVs exist in this scenario regardless of the flag
    with open(EVAL_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(new_row)
    print(f">>> [{scenario}] reused {from_label} eval row for {to_label} (no EMVs in this scenario)", flush=True)
EPISODES = 5
EPISODE_STEPS = 1200
EVAL_STEPS = 1200

# name -> (w1, w2, w3, use_emv_routing, seeds)
CONFIGS = {
    "with_emv": (0.5, 0.4, 0.1, True, [0, 1, 2]),
    "without_emv": (0.5, 0.4, 0.1, False, [0, 1, 2]),
}


def run_train(scenario, config_name, seed, w1, w2, w3, use_emv):
    env = os.environ.copy()
    env.update(
        {
            "TRAFFIC_SCENARIO": scenario,
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
        f"{scenario}_{config_name}",
        "--seed",
        str(seed),
        "--episodes",
        str(EPISODES),
        "--episode-steps",
        str(EPISODE_STEPS),
    ]
    print(f">>> [{scenario}] training {config_name} seed={seed} (emv={use_emv})", flush=True)
    result = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "advesarial"), env=env, capture_output=True, text=True)
    print(result.stdout[-1500:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"Training failed: {scenario}/{config_name} seed={seed}")

    run_dir = None
    for line in result.stdout.splitlines():
        if line.startswith("RUN_DIR "):
            run_dir = line.split(" ", 1)[1].strip()
    return run_dir


def run_eval(scenario, label, controller, model_dir, use_emv):
    env = os.environ.copy()
    env.update(
        {
            "TRAFFIC_SCENARIO": scenario,
            "USE_EMV_ROUTING": "1" if use_emv else "0",
            "USE_ACCIDENTS": "0",
        }
    )
    cmd = [
        sys.executable,
        "src/evaluate.py",
        "--controller",
        controller,
        "--steps",
        str(EVAL_STEPS),
        "--label",
        f"{scenario}_{label}",
        "--csv",
        "../paper_metrics/eval_results.csv",
    ]
    if model_dir:
        cmd += ["--model-dir", model_dir]

    print(f">>> [{scenario}] evaluating {label} ({controller})", flush=True)
    result = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "advesarial"), env=env, capture_output=True, text=True)
    print(result.stdout[-2500:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"Evaluation failed: {scenario}/{label}")


def main():
    global EPISODES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=ALL_SCENARIOS,
        choices=ALL_SCENARIOS,
        help="Subset of scenarios to run (default: all 5). Useful for a cheap single-scenario retest.",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES, help="PPO fine-tuning episodes per seed (after BC warm-start).")
    parser.add_argument(
        "--only-with-emv",
        action="store_true",
        help="Train/eval only the with_emv config (skip without_emv + fixed + max_pressure), for adding a "
        "with_emv leg to scenarios that already have without_emv/fixed/max_pressure data from a prior run.",
    )
    args = parser.parse_args()
    scenarios = args.scenarios
    EPISODES = args.episodes

    manifest_path = os.path.join(REPO_ROOT, "paper_metrics", "train_manifest_full.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    for scenario in scenarios:
        has_emv = _has_emv(scenario)

        if args.only_with_emv:
            if not has_emv:
                print(f">>> [{scenario}] --only-with-emv but no emv route file found, skipping", flush=True)
                continue
            configs_to_train = {"with_emv": CONFIGS["with_emv"]}
        else:
            configs_to_train = CONFIGS if has_emv else {"without_emv": CONFIGS["without_emv"]}
            if not has_emv:
                print(f">>> [{scenario}] no {scenario}_emv.rou.xml -- skipping redundant with_emv training", flush=True)
            manifest[scenario] = {}  # fresh manifest entry only for a full (non-additive) run

        manifest.setdefault(scenario, {})
        run_dirs = {}
        for config_name, (w1, w2, w3, use_emv, seeds) in configs_to_train.items():
            manifest[scenario][config_name] = []
            for seed in seeds:
                run_dir = run_train(scenario, config_name, seed, w1, w2, w3, use_emv)
                manifest[scenario][config_name].append({"seed": seed, "run_dir": run_dir, "use_emv_routing": use_emv})
                if seed == 0:
                    run_dirs[config_name] = run_dir

            manifest_path = os.path.join(REPO_ROOT, "paper_metrics", "train_manifest_full.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        if args.only_with_emv:
            run_eval(scenario, "with_emv", "algorithm", run_dirs["with_emv"], use_emv=True)
            print(f"DONE scenario={scenario} (with_emv only)", flush=True)
            continue

        run_eval(scenario, "without_emv", "algorithm", run_dirs["without_emv"], use_emv=False)
        if has_emv:
            run_eval(scenario, "with_emv", "algorithm", run_dirs["with_emv"], use_emv=True)
        else:
            _duplicate_eval_row(scenario, "without_emv", "with_emv")
        run_eval(scenario, "fixed", "fixed", None, use_emv=False)
        run_eval(scenario, "max_pressure", "max_pressure", None, use_emv=False)
        print(f"DONE scenario={scenario}", flush=True)

    print("CAMPAIGN DONE", flush=True)


if __name__ == "__main__":
    main()
