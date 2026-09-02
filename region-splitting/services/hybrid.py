"""Composed community-detection + clustering partitioner.

Combines Louvain/Leiden (graph community detection) with DBSCAN (density
clustering) in both orders, producing a single node -> region partition that
always covers every node:

* ``dbscan -> <cd>`` (clustering first): DBSCAN coarse spatial clusters are
  contracted into a *supernode* graph (one node per cluster, edges weighted
  by inter-cluster road links) and Louvain/Leiden merges related clusters
  into regions.
* ``<cd> -> dbscan`` (community detection first): Louvain/Leiden communities
  are reduced to centroids and DBSCAN groups neighbouring communities into
  regions.

The emitted JSON mirrors the single-method services (``{"clusters",
"metrics"}``) where ``metrics`` carries the intermediate stage metrics
(DBSCAN quality, community-detection modularity) plus final region-quality
metrics, so each combination can be scored before it is pushed through the
RL pipeline.
"""

from __future__ import annotations

from collections import defaultdict

import community as community_louvain
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .base import BaseClusteringService
from .dbscan import (
    DBSCANConfig,
    DBSCANResult,
    auto_tune_config,
    compute_internal_metrics,
    min_samples_for,
    run_dbscan,
)
from .traci import TraciService

CD_METHODS = ("louvian", "leiden")
ORDER_CD_FIRST = ("dbscan", "leiden", "louvian")


def _reassign_noise_labels(coords: np.ndarray, labels: np.ndarray, noise_label: int = -1) -> np.ndarray:
    """Assign every noise point to the nearest cluster centroid (full coverage)."""
    labels = np.asarray(labels).copy()
    noise_mask = labels == noise_label
    if not noise_mask.any():
        return labels
    cluster_ids = sorted({int(label) for label in labels if label != noise_label})
    if not cluster_ids:
        labels[:] = 0
        return labels
    centroids = {cid: coords[labels == cid].mean(axis=0) for cid in cluster_ids}
    for idx in np.nonzero(noise_mask)[0]:
        nearest = min(cluster_ids, key=lambda cid: float(np.linalg.norm(coords[idx] - centroids[cid])))
        labels[idx] = nearest
    return labels


def _labels_to_clusters(node_ids, labels) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for node, label in zip(node_ids, labels):
        clusters[str(int(label))].append(node)
    return dict(clusters)


class HybridClusteringService(BaseClusteringService):
    """Two-stage ``(stage_1, stage_2)`` partitioner over a road network.

    Args:
        net_config_path: Path to the SUMO ``.net.xml`` file.
        traci_service: Optional TraciService (unused — weights are static).
        first: First stage: "dbscan", "louvian" or "leiden".
        second: Second stage: "dbscan", "louvian" or "leiden".
        eps: Fixed DBSCAN ``eps``. When None, auto-tuned per point cloud.
        min_samples: Fixed DBSCAN ``min_samples``. When None, scaled to size.
    """

    def __init__(
        self,
        net_config_path: str,
        traci_service: TraciService | None,
        first: str,
        second: str,
        eps: float | None = None,
        min_samples: int | None = None,
    ):
        super().__init__(net_config_path, traci_service)
        self.first = first
        self.second = second
        self.eps = eps
        self.min_samples = min_samples
        self.node_ids: list[str] = []
        self.coords: list[tuple[float, float]] = []
        self.graph = nx.Graph()

    # ------------------------------------------------------------------ #
    # graph construction
    # ------------------------------------------------------------------ #
    def build_graph(self):
        print("  Constructing graph from network nodes (coordinates)...")
        for node in self.net.getNodes():
            nid = node.getID()
            x, y = node.getCoord()
            self.node_ids.append(nid)
            self.coords.append((float(x), float(y)))
            self.graph.add_node(nid, x=float(x), y=float(y))

        for edge in self.net.getEdges():
            if edge.isSpecial():
                continue
            self.graph.add_edge(
                edge.getFromNode().getID(),
                edge.getToNode().getID(),
                id=edge.getID(),
                static_weight=edge.getLaneNumber(),
            )

        print(f"  Graph built: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        return self.graph

    # ------------------------------------------------------------------ #
    # dbscan stage
    # ------------------------------------------------------------------ #
    def _dbscan(self, coords: np.ndarray, fine: bool = False) -> tuple[np.ndarray, DBSCANConfig, dict]:
        mean_min, mean_max = (4, 10) if fine else (8, 40)
        if self.eps is None:
            config = auto_tune_config(coords, min_samples=self.min_samples, mean_min=mean_min, mean_max=mean_max)
        else:
            config = DBSCANConfig(
                eps=self.eps,
                min_samples=self.min_samples or min_samples_for(len(coords)),
            )
        result = run_dbscan(coords, config)
        labels = _reassign_noise_labels(coords, result.labels)
        report = compute_internal_metrics(coords, result, config)
        stage_metrics = {
            "n_clusters": result.n_clusters,
            "n_noise_before_reassign": result.n_noise,
            "noise_fraction_before_reassign": result.noise_fraction,
            "eps": config.eps,
            "min_samples": config.min_samples,
            "target_mean_size": f"{mean_min}-{mean_max}",
            "silhouette": report.get("silhouette"),
            "davies_bouldin": report.get("davies_bouldin"),
            "calinski_harabasz": report.get("calinski_harabasz"),
        }
        return labels, config, stage_metrics

    # ------------------------------------------------------------------ #
    # community-detection stage
    # ------------------------------------------------------------------ #
    def _run_louvian(self, graph: nx.Graph, weight: str = "weight"):
        partition = community_louvain.best_partition(graph, weight=weight)
        modularity = community_louvain.modularity(partition, graph, weight=weight)
        communities: dict[str, list[int]] = defaultdict(list)
        for node, comm in partition.items():
            communities[str(int(comm))].append(node)
        return communities, modularity

    def _run_leiden(self, graph: nx.Graph, weight: str = "weight"):
        vertices = list(graph.nodes())
        index = {v: i for i, v in enumerate(vertices)}
        edges = []
        weights = []
        for u, v, data in graph.edges(data=True):
            edges.append((index[u], index[v]))
            weights.append(data.get(weight, 1.0))
        g = ig.Graph(n=len(vertices), edges=edges, directed=False)
        g.es["weight"] = weights
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight")
        communities: dict[str, list[int]] = defaultdict(list)
        for cid, members in enumerate(partition):
            for vid in members:
                communities[str(cid)].append(vertices[vid])
        return communities, partition.modularity

    def _community_detection(self, graph: nx.Graph, method: str):
        if method == "louvian":
            communities, modularity = self._run_louvian(graph)
        else:
            communities, modularity = self._run_leiden(graph)
        return communities, modularity

    # ------------------------------------------------------------------ #
    # merge helpers
    # ------------------------------------------------------------------ #
    def _merge_supernode_communities(self, communities, clusters):
        """Expand supernode (cluster-level) communities back into road nodes."""
        regions: dict[str, list[str]] = defaultdict(list)
        for comm_id, cluster_ids in communities.items():
            for cid in cluster_ids:
                regions[str(comm_id)].extend(clusters.get(str(cid), []))
        return dict(regions)

    # ------------------------------------------------------------------ #
    # stage-2 helpers
    # ------------------------------------------------------------------ #
    def _supernode_graph(self, clusters: dict[str, list[str]]) -> nx.Graph:
        """Contract DBSCAN clusters into a weighted undirected supernode graph."""
        node_to_cluster: dict[str, str] = {}
        for cid, nodes in clusters.items():
            for node in nodes:
                node_to_cluster[node] = cid

        adjacency: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for u, v, _ in self.graph.edges(data=True):
            cu = node_to_cluster.get(u)
            cv = node_to_cluster.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            adjacency[cu][cv] += 1
            adjacency[cv][cu] += 1

        super_graph = nx.Graph()
        for cu, neighbors in adjacency.items():
            super_graph.add_node(cu)
            for cv, count in neighbors.items():
                if super_graph.has_edge(cu, cv):
                    super_graph[cu][cv]["weight"] += count
                else:
                    super_graph.add_edge(cu, cv, weight=count)
        return super_graph

    def _road_graph_undirected(self) -> nx.Graph:
        undirected = nx.Graph()
        for u, v, data in self.graph.edges(data=True):
            w = data.get("static_weight", 1.0)
            if undirected.has_edge(u, v):
                undirected[u][v]["weight"] += w
            else:
                undirected.add_edge(u, v, weight=w)
        return undirected

    # ------------------------------------------------------------------ #
    # get_clusters dispatch
    # ------------------------------------------------------------------ #
    def get_clusters(self):
        if self.first == "dbscan":
            return self._clustering_then_community_detection()
        return self._community_detection_then_clustering()

    def _clustering_then_community_detection(self):
        coords = np.asarray(self.coords, dtype=float)
        # Fine spatial cells first so the supernode graph is rich enough for
        # community detection to merge meaningfully.
        labels, config, dbscan_metrics = self._dbscan(coords, fine=True)
        clusters = _labels_to_clusters(self.node_ids, labels)

        super_graph = self._supernode_graph(clusters)
        if super_graph.number_of_nodes() < 2:
            regions = clusters
            cd_metrics = {"modularity": None, "n_communities": len(clusters), "cd_method": self.second}
        else:
            communities, modularity = self._community_detection(super_graph, self.second)
            regions = self._merge_supernode_communities(communities, clusters)
            cd_metrics = {
                "modularity": modularity,
                "n_communities": len(communities),
                "cd_method": self.second,
                "cluster_quality": self._cluster_quality(super_graph, communities),
            }

        metrics = {
            "order": f"{self.first}_then_{self.second}",
            "first_stage": "dbscan",
            "second_stage": self.second,
            "dbscan_stage": dbscan_metrics,
            "cd_stage": cd_metrics,
        }
        metrics.update(self._final_metrics(regions, coords, config))
        return {"clusters": regions, "metrics": metrics}

    def _community_detection_then_clustering(self):
        coords = np.asarray(self.coords, dtype=float)
        road_graph = self._road_graph_undirected()
        communities, modularity = self._community_detection(road_graph, self.first)

        # Within each community, subdivide spatially large communities into
        # compact sub-regions with DBSCAN (hierarchical refinement).
        regions: dict[str, list[str]] = defaultdict(list)
        dbscan_stages = {}
        region_counter = 0
        for comm_id, nodes in communities.items():
            indices = [self.node_ids.index(node) for node in nodes]
            comm_coords = coords[indices]
            if len(comm_coords) >= 2:
                labels, config, stage_metrics = self._dbscan(comm_coords, fine=True)
            else:
                labels = np.asarray([0])
                config = DBSCANConfig(eps=self.eps or 1.0, min_samples=1)
                stage_metrics = {"n_clusters": 1, "n_noise_before_reassign": 0, "noise_fraction_before_reassign": 0.0}
            dbscan_stages[comm_id] = stage_metrics
            sub_regions = _labels_to_clusters(nodes, labels)
            for sub_id, members in sub_regions.items():
                regions[str(region_counter)] = members
                region_counter += 1

        cd_metrics = {
            "modularity": modularity,
            "n_communities": len(communities),
            "cd_method": self.first,
            "cluster_quality": self._cluster_quality(road_graph, communities),
        }
        metrics = {
            "order": f"{self.first}_then_{self.second}",
            "first_stage": self.first,
            "second_stage": "dbscan",
            "cd_stage": cd_metrics,
            "dbscan_stage": {
                "per_community": dbscan_stages,
                "n_communities": len(communities),
            },
        }
        metrics.update(self._final_metrics(dict(regions), coords, config))
        return {"clusters": dict(regions), "metrics": metrics}

    # ------------------------------------------------------------------ #
    # final metrics
    # ------------------------------------------------------------------ #
    def _cluster_quality(self, graph: nx.Graph, communities: dict[str, list]):
        quality = {}
        node_to_comm: dict[str, str] = {}
        for cid, members in communities.items():
            for node in members:
                node_to_comm[node] = cid

        internal: dict[str, float] = defaultdict(float)
        external: dict[str, float] = defaultdict(float)
        for u, v, data in graph.edges(data=True):
            w = data.get("weight", 1.0)
            cu = node_to_comm.get(u)
            cv = node_to_comm.get(v)
            if cu is None or cv is None:
                continue
            if cu == cv:
                internal[cu] += w
            else:
                external[cu] += w
                external[cv] += w

        for cid in communities:
            quality[cid] = {
                "internal": round(internal.get(cid, 0.0), 2),
                "external": round(external.get(cid, 0.0), 2),
                "ratio": round(internal.get(cid, 0.0) / max(external.get(cid, 1e-6), 1e-6), 2),
            }
        return quality

    def _final_metrics(self, regions: dict[str, list[str]], coords: np.ndarray, config: DBSCANConfig) -> dict:
        labels = np.full(len(self.node_ids), -1, dtype=int)
        for cid, nodes in regions.items():
            for node in nodes:
                labels[self.node_ids.index(node)] = int(cid)

        sizes = sorted(len(v) for v in regions.values())
        report = compute_internal_metrics(
            coords,
            DBSCANResult(
                labels=labels,
                n_clusters=len(regions),
                n_noise=0,
                noise_fraction=0.0,
                core_sample_indices=np.asarray([], dtype=int),
                config=config,
                runtime_seconds=0.0,
            ),
            config,
        )

        return {
            "n_clusters": len(regions),
            "coverage": sum(sizes) / len(self.node_ids) if self.node_ids else 0.0,
            "cluster_size_statistics": {
                "min": min(sizes) if sizes else None,
                "max": max(sizes) if sizes else None,
                "mean": float(np.mean(sizes)) if sizes else None,
                "median": float(np.median(sizes)) if sizes else None,
                "gini": report["cluster_size_statistics"]["gini"],
            },
            "silhouette": report["silhouette"],
            "davies_bouldin": report["davies_bouldin"],
            "calinski_harabasz": report["calinski_harabasz"],
        }

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def generate_visualization(self, clusters: dict, output_path: str):
        pos = {node: (data["x"], data["y"]) for node, data in self.graph.nodes(data=True)}
        partition_map = defaultdict(lambda: -1)
        for cid, nodes in clusters.items():
            for node in nodes:
                partition_map[node] = int(cid)
        colors = [partition_map[n] for n in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_size=8, node_color=colors, with_labels=False, cmap=plt.cm.tab20)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
