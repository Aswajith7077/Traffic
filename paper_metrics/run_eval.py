"""Evaluates every trained config (seed 0) plus the Fixed-Time and
Max-Pressure baselines, filling paper_metrics/eval_results.csv (Tables 1-2).
Run after run_experiments.py has written train_manifest.json.
"""

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO = "cologne8"
EVAL_STEPS = 1200


def run(label, controller, model_dir, use_emv):
    env = os.environ.copy()
    env.update(
        {
            "TRAFFIC_SCENARIO": SCENARIO,
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
        label,
        "--csv",
        "../paper_metrics/eval_results.csv",
    ]
    if model_dir:
        cmd += ["--model-dir", model_dir]

    print(f">>> evaluating {label} ({controller})", flush=True)
    result = subprocess.run(cmd, cwd=os.path.join(REPO_ROOT, "advesarial"), env=env, capture_output=True, text=True)
    print(result.stdout[-2500:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], flush=True)
        raise RuntimeError(f"Evaluation failed: {label}")


def main():
    manifest_path = os.path.join(REPO_ROOT, "paper_metrics", "train_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    with_emv_dir = manifest["with_emv"][0]["run_dir"]
    without_emv_dir = manifest["without_emv"][0]["run_dir"]
    hilight_dir = manifest["hilight_ablation"][0]["run_dir"]

    run("with_emv", "algorithm", with_emv_dir, use_emv=True)
    run("without_emv", "algorithm", without_emv_dir, use_emv=False)
    run("hilight_ablation", "algorithm", hilight_dir, use_emv=False)
    run("fixed", "fixed", None, use_emv=False)
    run("max_pressure", "max_pressure", None, use_emv=False)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
