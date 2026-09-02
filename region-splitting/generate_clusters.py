"""Generate partitions for every method (single + hybrid) and every scenario.

Writes cluster JSON (including clustering / community-detection metrics) to
``region-splitting/clusters/<method>/`` and copies it into
``advesarial/clusters/<method>/`` so the RL pipeline can consume any method.

Usage:
    python generate_clusters.py                          # all methods x all scenarios
    python generate_clusters.py --method dbscan_leiden   # one method
    python generate_clusters.py --scenario cologne8      # one scenario
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

from main import run_partition
from services import ALL_METHODS

SCENARIOS = ["manhattan", "cologne8", "ingolstadt21", "arterial4x4", "grid4x4"]


def copy_to_advesarial(method: str, scenario: str):
    src = f"clusters/{method}/{scenario}_clusters.json"
    dst_dir = f"../advesarial/clusters/{method}"
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, f"{dst_dir}/{scenario}_clusters.json")


def main():
    parser = argparse.ArgumentParser(description="Generate partitions for all methods and scenarios")
    parser.add_argument("--method", default=None, choices=ALL_METHODS, help="Only this method (default: all)")
    parser.add_argument("--scenario", default=None, help="Only this scenario (default: all)")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(ALL_METHODS)
    scenarios = [args.scenario] if args.scenario else SCENARIOS

    summary = {}
    for method in methods:
        for scenario in scenarios:
            try:
                clusters = run_partition(scenario, method=method)
                copy_to_advesarial(method, scenario)
                metrics = clusters.get("metrics", {})
                summary.setdefault(method, {})[scenario] = {
                    "n_clusters": metrics.get("n_clusters"),
                    "modularity": metrics.get("modularity") or (metrics.get("cd_stage") or {}).get("modularity"),
                    "silhouette": metrics.get("silhouette"),
                }
            except Exception as exc:
                print(f"  ERROR {method}/{scenario}: {exc}")

    print("\n=== Summary ===")
    print(f"{'method':16s} {'scenario':14s} {'n_clusters':>10s} {'modularity':>10s} {'silhouette':>10s}")
    for method in methods:
        for scenario, row in summary.get(method, {}).items():
            print(
                f"{method:16s} {scenario:14s} {str(row['n_clusters']):>10s} "
                f"{str(row['modularity']):>10s} {str(row['silhouette']):>10s}"
            )


if __name__ == "__main__":
    main()
