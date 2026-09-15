from collections import defaultdict

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx

from .base import BaseClusteringService
from .topology import build_tls_projected_graph, get_tls_node_ids
from .traci import TraciService


class LouvianService(BaseClusteringService):
    def __init__(
        self,
        net_config_path: str,
        traci_service: TraciService,
        min_cluster_size: int = 5,
    ):
        super().__init__(net_config_path, traci_service)
        self.min_cluster_size = min_cluster_size

        self.graph = nx.Graph()
        self.edge_weights = {}

    def build_graph(self):
        print("  Constructing traffic-light-projected graph from network edges...")
        tls_ids = get_tls_node_ids(self.net)
        self.graph = build_tls_projected_graph(self.net, tls_ids, weight_fn=self._edge_weight)

        print(f"  Graph built: {len(self.graph.nodes)} traffic-light nodes, {len(self.graph.edges)} edges")
        return self.graph

    def _edge_weight(self, edge) -> float:
        """Live simulation weight when available, falling back to lane count."""
        w = self.edge_weights.get(edge.getID(), 0.0)
        if w > 0:
            return w
        return float(edge.getLaneNumber())

    def compute_weights(self, max_iter=1000):

        edge_weights = {}
        for _ in range(max_iter):
            self.traci_service.step()

            for edge in self.net.getEdges():
                if edge.isSpecial():
                    continue

                weight = self.traci_service.get_edge_weight(edge)
                edge_weights[edge.getID()] = edge_weights.get(edge.getID(), 0) + weight

        for key in edge_weights:
            edge_weights[key] /= max_iter

        self.edge_weights = edge_weights

    def get_clusters(self):
        # Step 1: run Louvain directly on the (already undirected, weighted)
        # traffic-light-projected graph
        partition = community_louvain.best_partition(self.graph, weight="weight")
        modularity = community_louvain.modularity(partition, self.graph, weight="weight")

        # Step 2: group nodes by community
        clusters = {}
        for node, comm_id in partition.items():
            clusters.setdefault(comm_id, []).append(node)

        clusters = self.merge_small_clusters(clusters)

        metrics = {}
        metrics["modularity"] = modularity
        metrics["cluster_quality"] = self.cluster_quality(clusters)

        result = {"clusters": clusters, "metrics": metrics}

        return result

    def cluster_quality(self, clusters):
        results = {}

        for cid, nodes in clusters.items():
            internal = 0
            external = 0

            for u in nodes:
                for v in self.graph.neighbors(u):
                    if v in nodes:
                        internal += 1
                    else:
                        external += 1

            results[cid] = {
                "internal": internal,
                "external": external,
                "ratio": internal / (external + 1),
            }

        return results

    def merge_small_clusters(self, clusters):
        large = {}
        small = {}

        for cid, nodes in clusters.items():
            if len(nodes) < self.min_cluster_size:
                small[cid] = nodes
            else:
                large[cid] = nodes

        for cid, nodes in small.items():
            best_target = None
            best_weight = -1

            for node in nodes:
                for nbr in self.graph.neighbors(node):
                    for target_id, target_nodes in large.items():
                        if nbr in target_nodes:
                            w = self.graph[node][nbr]["weight"]
                            if w > best_weight:
                                best_weight = w
                                best_target = target_id

            if best_target is not None:
                large[best_target].extend(nodes)
            else:
                # No large neighbor to merge into (e.g. an isolated
                # traffic-light component). Keep it as its own region rather
                # than silently dropping these traffic lights from the
                # partition — every TLS node must end up in some region.
                large[cid] = nodes

        return large

    def generate_visualization(self, clusters, output_path: str):

        pos = {node: self.net.getNode(node).getCoord() for node in self.graph.nodes()}

        partition_map = defaultdict(lambda: -1)
        for cid, nodes in clusters.items():
            for n in nodes:
                partition_map[n] = int(cid)

        colors = [partition_map[n] for n in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw(
            self.graph.to_undirected(),
            pos,
            node_size=8,
            node_color=colors,
            with_labels=False,
            cmap=plt.cm.tab20,
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
