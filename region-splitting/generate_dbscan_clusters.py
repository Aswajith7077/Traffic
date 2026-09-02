"""Generate production DBSCAN partitions for the RL pipeline.

For each scenario: auto-tune ``eps`` (targeting a moderate region count with
low noise), run DBSCAN with noise reassignment for full coverage, and write
the partition to both ``region-splitting/clusters/dbscan/`` and
``advesarial/clusters/dbscan/``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from services.dbscan import DBSCANConfig, DBSCANService, auto_tune_config

SCENARIOS = ["manhattan", "cologne8", "ingolstadt21", "arterial4x4", "grid4x4"]


def net_path(scenario):
    return f"../scenarios/{scenario}/{scenario}.net.xml"


def generate_for_scenario(scenario: str, out_dirs=("clusters/dbscan", "../advesarial/clusters/dbscan")):
    path = net_path(scenario)
    if not os.path.exists(path):
        print(f"  {scenario:14s} net missing, skipping")
        return False

    svc = DBSCANService(path, None, DBSCANConfig())
    svc.build_graph()
    coords = np.asarray(svc.coords, dtype=float)
    svc.config = auto_tune_config(coords)
    result = svc.get_clusters(reassign_noise=True)
    total = sum(len(v) for v in result["clusters"].values())

    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/{scenario}_clusters.json", "w") as f:
            json.dump(result, f, indent=4)

    m = result["metrics"]
    print(
        f"  {scenario:14s} eps={m['eps']:<9.3g} ms={m['min_samples']:<3} "
        f"n_clusters={m['n_clusters']:<4} coverage={total} nodes"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate production DBSCAN partitions")
    parser.add_argument("--scenario", default=None, help="Only generate for this scenario (default: all)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    for scenario in scenarios:
        generate_for_scenario(scenario)


if __name__ == "__main__":
    main()
