import networkx as nx
import community as community_louvain
from collections import defaultdict

from .traci import TraciService
from .base import BaseClusteringService
import matplotlib.pyplot as plt


class LouvianService(BaseClusteringService):
    def __init__(
        self,
        net_config_path: str,
        traci_service: TraciService,
        min_cluster_size: int = 5,
    ):
        super().__init__(net_config_path, traci_service)
        self.min_cluster_size = min_cluster_size

        self.graph = nx.DiGraph()
        self.edge_weights = {}

    def build_graph(self):

        for edge in self.net.getEdges():
            if edge.isSpecial():  # skip internal edges
                continue

            source = edge.getFromNode().getID()
            destination = edge.getToNode().getID()

            self.graph.add_edge(
                source, destination, id=edge.getID(), static_weight=edge.getLaneNumber()
            )

        return self.graph

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

    def normalize_edge_id(self, edge_id: str):
        if edge_id.startswith("-"):
            return edge_id[1:]
        return edge_id

    def to_weighted_undirected(self):
        undirected_graph = nx.Graph()

        for u, v, data in self.graph.edges(data=True):
            edge_id = self.normalize_edge_id(data["id"])
            w = self.edge_weights.get(edge_id, 0)

            if w <= 0:
                w = data.get("lane_weight", 1.0)

            if undirected_graph.has_edge(u, v):
                undirected_graph[u][v]["weight"] += w
            else:
                undirected_graph.add_edge(u, v, weight=w)

        return undirected_graph

    def get_clusters(self):
        # Step 1: convert to undirected graph
        undirected_graph = self.to_weighted_undirected()

        # Step 2: run Louvain
        partition = community_louvain.best_partition(undirected_graph, weight="weight")
        modularity = community_louvain.modularity(
            partition, undirected_graph, weight="weight"
        )

        # Step 3: group nodes by community
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
        G = self.to_weighted_undirected()

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
                for nbr in G.neighbors(node):
                    for target_id, target_nodes in large.items():
                        if nbr in target_nodes:
                            w = G[node][nbr]["weight"]
                            if w > best_weight:
                                best_weight = w
                                best_target = target_id

            if best_target is not None:
                large[best_target].extend(nodes)

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
