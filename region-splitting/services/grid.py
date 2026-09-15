"""Static grid-based (manual) partitioning for the region-splitting module.

Mirrors the HiLight paper's own preprocessing step for its largest scenario
(Manhattan2668): rather than running community detection or density
clustering, the network's traffic-light nodes are sliced by their raw x/y
coordinates into a fixed N x M grid of rectangular cells (see Appendix E,
"When dividing the network into subregions using a grid-based approach...").
No traffic data or graph structure is used — the partition is a pure,
deterministic function of node geometry, which is what makes it "manual" /
"static" relative to the learned or graph-based methods (Leiden, Louvain,
DBSCAN, and their hybrids) already in this module.

Grid shape selection follows the paper's stated target of keeping ~10^1
intersections per region: the number of cells is chosen so the mean cluster
size falls in ``[mean_min, mean_max]`` (reusing
:func:`services.dbscan.cluster_bounds`), then split into rows/cols matching
the network's bounding-box aspect ratio -- the same reasoning the paper gives
for using an asymmetric 4x15 grid on Manhattan's long, narrow layout.
"""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .base import BaseClusteringService
from .dbscan import cluster_bounds
from .topology import build_tls_projected_graph, get_tls_node_ids
from .traci import TraciService


def compute_grid_shape(coords: np.ndarray, mean_min: int = 10, mean_max: int = 50) -> tuple[int, int]:
    """Pick a (rows, cols) grid so mean cells-per-node falls in [mean_min, mean_max].

    The cell count targets the midpoint of the mean-size band, clamped to the
    range :func:`cluster_bounds` reports as valid for the given node count,
    then factored into rows/cols to match the bounding box's aspect ratio.
    """
    n_nodes = len(coords)
    low, high = cluster_bounds(n_nodes, mean_min=mean_min, mean_max=mean_max)
    target = max(1, round(n_nodes / ((mean_min + mean_max) / 2)))
    target = min(max(target, low), high)

    xs, ys = coords[:, 0], coords[:, 1]
    width = float(xs.max() - xs.min()) or 1.0
    height = float(ys.max() - ys.min()) or 1.0
    aspect = width / height

    rows = max(1, round(math.sqrt(target / aspect)))
    cols = max(1, round(target / rows))
    return rows, cols


def assign_grid_cells(coords: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Map each (x, y) point to a flat cell index ``row * cols + col``."""
    xs, ys = coords[:, 0], coords[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = (x_max - x_min) or 1.0
    height = (y_max - y_min) or 1.0

    col_idx = np.floor((xs - x_min) / width * cols).astype(int)
    row_idx = np.floor((ys - y_min) / height * rows).astype(int)
    col_idx = np.clip(col_idx, 0, cols - 1)
    row_idx = np.clip(row_idx, 0, rows - 1)
    return row_idx * cols + col_idx


class GridService(BaseClusteringService):
    """Static geometric grid partition over road-network node coordinates.

    Mirrors the :class:`DBSCANService` / :class:`LeidenService` interface:
    ``build_graph`` collects traffic-light nodes and their coordinates (same
    TLS-only projected graph the other methods use, for comparability),
    ``get_clusters`` slices the bounding box into a grid and groups nodes by
    cell, and ``generate_visualization`` renders the assignment.
    """

    def __init__(
        self,
        net_config_path: str,
        traci_service: TraciService | None,
        rows: int | None = None,
        cols: int | None = None,
        mean_min: int = 10,
        mean_max: int = 50,
    ):
        super().__init__(net_config_path, traci_service)
        self.rows = rows
        self.cols = cols
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.node_ids: list[str] = []
        self.coords: list[tuple[float, float]] = []
        self.graph = nx.Graph()

    def build_graph(self):
        print("  Constructing graph from traffic-light junction coordinates...")
        tls_ids = get_tls_node_ids(self.net)
        self.graph = build_tls_projected_graph(self.net, tls_ids)

        self.node_ids = list(self.graph.nodes())
        self.coords = [(data["x"], data["y"]) for _, data in self.graph.nodes(data=True)]

        print(f"  Graph built: {len(self.graph.nodes)} traffic-light nodes, {len(self.graph.edges)} edges")
        return self.graph

    def get_clusters(self):
        if not self.coords:
            raise RuntimeError("No coordinates collected — call build_graph() first.")

        coords = np.asarray(self.coords, dtype=float)
        rows, cols = (
            (self.rows, self.cols)
            if self.rows and self.cols
            else compute_grid_shape(coords, self.mean_min, self.mean_max)
        )
        cell_ids = assign_grid_cells(coords, rows, cols)

        clusters: dict[str, list[str]] = {}
        for node_id, cell_id in zip(self.node_ids, cell_ids):
            clusters.setdefault(str(int(cell_id)), []).append(node_id)

        sizes = [len(v) for v in clusters.values()]
        metrics = {
            "method": "manual_clustering",
            "grid_shape": [rows, cols],
            "n_clusters": len(clusters),
            "cluster_size_statistics": {
                "min": min(sizes),
                "max": max(sizes),
                "mean": float(np.mean(sizes)),
            },
        }
        return {"clusters": clusters, "metrics": metrics}

    def generate_visualization(self, clusters: dict, output_path: str):
        pos = {node: (data["x"], data["y"]) for node, data in self.graph.nodes(data=True)}

        partition_map = defaultdict(lambda: -1)
        for cid, nodes in clusters.items():
            for node in nodes:
                partition_map[node] = int(cid)

        colors = [partition_map[n] for n in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw(
            self.graph,
            pos,
            node_size=8,
            node_color=colors,
            with_labels=False,
            cmap=plt.cm.tab20,
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
