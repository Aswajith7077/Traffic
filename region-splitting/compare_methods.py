#!/usr/bin/env python3
"""Compare all partition methods on the clustering + community-detection metrics.

Reads the generated cluster JSON files (from ``region-splitting/clusters/`` or
``advesarial/clusters/``) and prints, per method × scenario:

  M       final region count (what the RL meta-policy sees)
  cov     node coverage of the partition
  mod     modularity of the community-detection stage
  sil/db/ch  silhouette, Davies-Bouldin, Calinski-Harabasz of the final partition
  n_comm  number of stage-1 communities / supernode communities

Usage:
    python compare_methods.py            # advesarial/clusters (default)
    python compare_methods.py --root region-splitting/clusters
"""

import argparse
import json
import os

SCENARIOS = ["manhattan", "cologne8", "ingolstadt21", "arterial4x4", "grid4x4"]
METHODS = ["dbscan", "leiden", "louvian", "dbscan_louvian", "dbscan_leiden", "louvian_dbscan", "leiden_dbscan"]


def fmt(value, width=7):
    if value is None:
        return "-".rjust(width)
    if isinstance(value, bool):
        return str(value).rjust(width)
    if isinstance(value, int):
        return str(value).rjust(width)
    try:
        f = float(value)
        return f"{f:.3f}".rjust(width)
    except TypeError, ValueError:
        return str(value).rjust(width)


def main():
    parser = argparse.ArgumentParser(description="Compare partition methods by clustering + CD metrics")
    parser.add_argument("--root", default="advesarial/clusters", help="Cluster root directory")
    args = parser.parse_args()

    header = f"{'method':<15s} {'scenario':<14s} {'M':>4s} {'cov':>6s} {'mod':>7s}"
    header += f" {'sil':>7s} {'db':>7s} {'ch':>9s} {'n_comm':>7s}"
    print(header)
    print("-" * 88)
    for method in METHODS:
        for scenario in SCENARIOS:
            path = f"{args.root}/{method}/{scenario}_clusters.json"
            if not os.path.exists(path):
                print(f"{method:<15s} {scenario:<14s} MISSING")
                continue
            with open(path) as f:
                data = json.load(f)
            met = data["metrics"]
            cd = met.get("cd_stage") or {}
            n_clusters = met.get("n_clusters")
            coverage = met.get("coverage")
            modularity = met.get("modularity", cd.get("modularity"))
            n_comm = cd.get("n_communities")
            print(
                f"{method:<15s} {scenario:<14s} {fmt(n_clusters, 4)} {fmt(coverage, 6)} "
                f"{fmt(modularity, 7)} {fmt(met.get('silhouette'), 7)} "
                f"{fmt(met.get('davies_bouldin'), 7)} {fmt(met.get('calinski_harabasz'), 9)} "
                f"{fmt(n_comm, 7)}"
            )


if __name__ == "__main__":
    main()
