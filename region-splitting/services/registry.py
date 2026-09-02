"""Pluggable partitioner registry.

Maps a method name to a partitioner service. This is the single switch that
lets the pipeline swap between the standalone community-detection methods
(``leiden``, ``louvian``), the standalone density-clustering method
(``dbscan``), and the four two-stage combinations of community detection +
clustering:

* ``dbscan_louvian``  — DBSCAN first, Louvain on the cluster supernode graph
* ``dbscan_leiden``   — DBSCAN first, Leiden on the cluster supernode graph
* ``louvian_dbscan``  — Louvain first, DBSCAN over community centroids
* ``leiden_dbscan``   — Leiden first, DBSCAN over community centroids

All methods share the :class:`BaseClusteringService` interface
(``build_graph`` / ``get_clusters`` / ``generate_visualization``), so the
pipeline treats them identically.
"""

from __future__ import annotations

from .base import BaseClusteringService
from .dbscan import DBSCANConfig, DBSCANService
from .hybrid import HybridClusteringService
from .leiden import LeidenService
from .louvian import LouvianService
from .traci import TraciService

SINGLE_METHODS = ("leiden", "louvian", "dbscan")
HYBRID_METHODS = ("dbscan_louvian", "dbscan_leiden", "louvian_dbscan", "leiden_dbscan")
ALL_METHODS = SINGLE_METHODS + HYBRID_METHODS

# Default dbscan parameters used by the standalone dbscan method.
DEFAULT_DBSCAN_EPS = 0.5
DEFAULT_DBSCAN_MIN_SAMPLES = 5


def is_hybrid(method: str) -> bool:
    return method in HYBRID_METHODS


def create_service(
    method: str,
    net_config_path: str,
    traci_service: TraciService | None = None,
    eps: float | None = None,
    min_samples: int | None = None,
) -> BaseClusteringService:
    """Instantiate the partitioner registered for ``method``.

    Args:
        method: One of :data:`ALL_METHODS`.
        net_config_path: Path to the SUMO ``.net.xml`` file.
        traci_service: Optional TraciService for traffic-weighted methods.
        eps: DBSCAN ``eps`` (ignored unless the method uses DBSCAN).
        min_samples: DBSCAN ``min_samples`` (ignored unless the method uses DBSCAN).

    Returns:
        A configured clustering service.
    """
    if method == "leiden":
        return LeidenService(net_config_path, traci_service)
    if method == "louvian":
        return LouvianService(net_config_path, traci_service)
    if method == "dbscan":
        return DBSCANService(
            net_config_path,
            traci_service,
            DBSCANConfig(eps=eps, min_samples=min_samples),
        )
    if method in HYBRID_METHODS:
        first, second = method.split("_", 1)
        return HybridClusteringService(
            net_config_path,
            traci_service,
            first,
            second,
            eps=eps,
            min_samples=min_samples,
        )
    raise ValueError(f"Unknown clustering method '{method}'. Choose from {sorted(ALL_METHODS)}.")
