import argparse
import json
import os

import numpy as np
from schema import TraciConfig
from services import create_service, is_hybrid
from services.dbscan import auto_tune_config
from services.traci import TraciService


def run_partition(module_name: str, method: str = "leiden", eps: float | None = None, min_samples: int | None = None):
    """Run one partitioner and write its JSON + visualization."""
    net_path = f"../scenarios/{module_name}/{module_name}.net.xml"

    # Leiden/Louvain weight the graph from a live SUMO simulation; DBSCAN and
    # the hybrid combinations use static/geometric features and need no SUMO.
    traci_service = None
    if method in ("leiden", "louvian"):
        config = TraciConfig(config_path=f"../scenarios/{module_name}/{module_name}.sumocfg")
        traci_service = TraciService(config)
        traci_service.start_simulation()

    service = create_service(method, net_path, traci_service, eps=eps, min_samples=min_samples)
    service.build_graph()

    if method == "dbscan" and eps is None:
        coords = np.asarray(service.coords, dtype=float)
        service.config = auto_tune_config(coords, min_samples=min_samples)
        clusters = service.get_clusters(reassign_noise=True)
    else:
        clusters = service.get_clusters()

    os.makedirs(f"visualizations/{method}", exist_ok=True)
    os.makedirs(f"clusters/{method}", exist_ok=True)
    service.generate_visualization(clusters["clusters"], f"visualizations/{method}/{module_name}.png")

    metrics = clusters.setdefault("metrics", {})
    metrics.setdefault("n_clusters", len(clusters["clusters"]))
    total_nodes = sum(len(v) for v in clusters["clusters"].values())
    total_net_nodes = len(service.net.getNodes())
    metrics.setdefault("coverage", total_nodes / max(1, total_net_nodes))

    with open(f"clusters/{method}/{module_name}_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)

    print(f"  {module_name:14s} method={method:16s} n_clusters={metrics.get('n_clusters')}")

    if traci_service is not None:
        traci_service.close_simulation()

    return clusters


def main(module_name: str, method: str = "leiden", eps: float | None = None, min_samples: int | None = None):
    if is_hybrid(method):
        return run_partition(module_name, method, eps, min_samples)
    return run_partition(module_name, method, eps, min_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Region splitting (Leiden/Louvain/DBSCAN and hybrid combinations)")
    parser.add_argument(
        "--scenario",
        default=os.environ.get("TRAFFIC_SCENARIO", "manhattan"),
        help="Scenario dataset to cluster (default: manhattan)",
    )
    parser.add_argument(
        "--method",
        choices=["leiden", "louvian", "dbscan", "dbscan_louvian", "dbscan_leiden", "louvian_dbscan", "leiden_dbscan"],
        default="leiden",
        help="Partitioning algorithm to run (default: leiden)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="DBSCAN neighborhood radius (default: auto-tuned)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="DBSCAN minimum samples per neighborhood (default: auto-scaled)",
    )
    args = parser.parse_args()
    main(args.scenario, method=args.method, eps=args.eps, min_samples=args.min_samples)
