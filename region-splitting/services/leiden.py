from collections import defaultdict

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx

from .base import BaseClusteringService
from .traci import TraciService


class LeidenService(BaseClusteringService):
    def __init__(self, net_config_path: str, traci_service: TraciService):
        super().__init__(net_config_path, traci_service)
        self.graph = ig.Graph(directed=True)

    def build_graph(self):
        print("  Building edge weight profile (may take a while on large networks)...")
        self.compute_weights()

        print("  Constructing graph from network edges...")
        vertices = set()
        edges_to_add = []

        for edge in self.net.getEdges():
            if edge.isSpecial():
                continue

            edge_id = edge.getID()
            source = edge.getFromNode().getID()
            destination = edge.getToNode().getID()

            vertices.add(source)
            vertices.add(destination)

            weight = self.edge_weights.get(edge_id, 0)
            edges_to_add.append((source, destination, weight))
        # Add unique vertices once
        self.graph.add_vertices(list(vertices))

        # Add edges
        for source, destination, weight in edges_to_add:
            self.graph.add_edge(source, destination, weight=weight)

        print(f"  Graph built: {len(self.graph.vs)} nodes, {len(self.graph.es)} edges")

        # for v in self.graph.vs:
        #     print(v["name"], self.graph.degree(v.index))

        # print(self.graph.vs["name"])

        return self.graph

    def compute_weights(self, max_iter=1000):

        edge_weights = {}
        log_interval = max(1, max_iter // 10)
        for i in range(max_iter):
            self.traci_service.step()

            for edge in self.net.getEdges():
                if edge.isSpecial():
                    continue

                weight = self.traci_service.get_edge_weight(edge)
                edge_weights[edge.getID()] = edge_weights.get(edge.getID(), 0) + weight

            if (i + 1) % log_interval == 0:
                print(f"  compute_weights: {i + 1}/{max_iter} steps completed")

        for key in edge_weights:
            edge_weights[key] /= max_iter

        self.edge_weights = edge_weights
        print(f"  compute_weights: finished ({max_iter} steps averaged)")

    def get_clusters(self):
        partition = leidenalg.find_partition(
            self.graph,
            leidenalg.ModularityVertexPartition,
            weights="weight",  # specify the edge weight attribute
        )

        cluster = {}

        for cid, community in enumerate(partition):
            cluster[str(cid)] = [self.graph.vs[v]["name"] for v in community]

        cluster = self.merge_singletons(cluster)

        metrics = {}
        metrics["modularity"] = partition.modularity
        metrics["cluster_quality"] = self.compute_cluster_quality(cluster)

        return {"clusters": cluster, "metrics": metrics}

    def merge_singletons(self, clusters):
        node_to_cluster = {}

        for cid, nodes in clusters.items():
            for node in nodes:
                node_to_cluster[node] = cid

        merged = {k: list(v) for k, v in clusters.items()}

        to_delete = []

        for cid, nodes in list(clusters.items()):
            if len(nodes) != 1:
                continue

            node = nodes[0]
            vid = self.graph.vs.find(name=node).index

            if self.graph.degree(vid, mode="all") <= 1:
                neighbors = self.graph.neighbors(vid, mode="all")

                if neighbors:
                    neighbor_name = self.graph.vs[neighbors[0]]["name"]
                    target_cluster = node_to_cluster.get(neighbor_name)

                    if target_cluster and target_cluster != cid:
                        merged[target_cluster].append(node)
                        to_delete.append(cid)

        for cid in to_delete:
            merged.pop(cid, None)

        return merged

    def compute_cluster_quality(self, clusters):
        quality = {}

        for cid, nodes in clusters.items():
            cluster_set = set(nodes)

            internal = 0.0
            external = 0.0

            for edge in self.graph.es:
                u = self.graph.vs[edge.source]["name"]
                v = self.graph.vs[edge.target]["name"]
                w = edge["weight"] if "weight" in edge.attributes() else 1.0

                u_in = u in cluster_set
                v_in = v in cluster_set

                if u_in and v_in:
                    internal += w
                elif u_in or v_in:
                    external += w

            ratio = round(internal / max(external, 1e-6), 2)

            quality[cid] = {"internal": internal, "external": external, "ratio": ratio}

        return quality

    def generate_visualization(self, clusters: dict, output_path: str):
        # Convert igraph to networkx
        G = nx.Graph()

        # Add nodes
        for v in self.graph.vs:
            G.add_node(v["name"])

        # Add edges
        for e in self.graph.es:
            source = self.graph.vs[e.source]["name"]
            target = self.graph.vs[e.target]["name"]
            G.add_edge(source, target)

        # Positions from SUMO net
        pos = {node: self.net.getNode(node).getCoord() for node in G.nodes()}

        # Cluster → color mapping
        partition_map = defaultdict(lambda: -1)

        for cid, nodes in clusters.items():
            for node in nodes:
                partition_map[node] = int(cid)

        colors = [partition_map[node] for node in G.nodes()]

        plt.figure(figsize=(12, 12))

        nx.draw(G, pos=pos, node_size=50, node_color=colors, cmap=plt.cm.tab20)

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
