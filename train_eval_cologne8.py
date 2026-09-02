#!/usr/bin/env python3
"""Train and evaluate the RL agent for the cologne8 scenario only.

This is the single-scenario pipeline: ensure the cologne8 partition exists,
train the model, then evaluate it in SUMO and report average travel time and
average delay time.

Usage:
    python train_eval_cologne8.py                              # dbscan partition
    python train_eval_cologne8.py --cluster-method louvian_dbscan
    python train_eval_cologne8.py --episodes 20 --episode-steps 3600 --save-every 10
    python train_eval_cologne8.py --steps 1000                # eval steps
"""

import argparse
import sys

from pipeline import (
    SCENARIOS,
    step_cluster,
    step_copy,
    step_eval,
    step_train,
)

SCENARIO = "cologne8"


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate the RL agent for cologne8 only")
    parser.add_argument(
        "--cluster-method",
        default="dbscan",
        choices=[
            "dbscan",
            "leiden",
            "louvian",
            "dbscan_louvian",
            "dbscan_leiden",
            "louvian_dbscan",
            "leiden_dbscan",
        ],
        help="Partitioning method to train/evaluate on (default: dbscan)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Training episodes (default: 10)")
    parser.add_argument("--episode-steps", type=int, default=1000, help="Sim-seconds per episode")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval")
    parser.add_argument("--steps", type=int, default=500, help="Evaluation steps (default: 500)")
    args = parser.parse_args()

    sdir = SCENARIOS / SCENARIO
    if not sdir.exists():
        print(f"  ERROR: scenario directory not found: {sdir}")
        sys.exit(1)

    # 1) Ensure the partition exists for the requested method.
    if not step_cluster(SCENARIO, None, args.cluster_method):
        print("  Pipeline stopped: clustering failed")
        sys.exit(1)
    if not step_copy(SCENARIO, None, args.cluster_method):
        print("  Pipeline stopped: cluster copy failed")
        sys.exit(1)

    # 2) Train.
    if not step_train(
        SCENARIO,
        None,
        args.episodes,
        args.episode_steps,
        args.save_every,
        args.cluster_method,
    ):
        print("  Pipeline stopped: training failed")
        sys.exit(1)

    # 3) Evaluate in SUMO.
    if not step_eval(SCENARIO, None, args.cluster_method, args.steps):
        print("  Pipeline stopped: evaluation failed")
        sys.exit(1)

    print(f"\n  cologne8 pipeline completed (cluster-method={args.cluster_method}).")
    print("  See advesarial/metrics.txt for Average Travel Time / Average Delay Time.")


if __name__ == "__main__":
    main()
