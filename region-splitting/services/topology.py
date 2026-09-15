"""Shared graph utilities for restricting clustering/community detection to
traffic-signal-controlled junctions only.

The raw SUMO road network is far denser than the set of traffic lights the RL
system actually controls (see ``plans/tls-only-clustering-scope.md``):
dead-ends, unsignalized priority junctions, and OSM geometry nodes vastly
outnumber real traffic-light intersections on real-world networks (e.g. only
~17% of ``manhattan``'s junctions and ~10% of ``cologne8``'s are traffic
lights). Feeding the full node set into Louvain/Leiden/DBSCAN distorts
community boundaries and density estimates away from the population the RL
Meta-Policy actually needs to partition.

``build_tls_projected_graph`` fixes this the standard way: contract every
maximal chain of non-signalized junctions between two traffic lights into a
single edge, so the resulting graph's vertices are exactly the traffic-light
ids (matching ``traci.trafficlight.getIDList()``, the same source
``advesarial/src/services/traci.py::get_all_intersections()`` already
trusts), while preserving connectivity and an aggregated notion of "how much
road" ties two signals together.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import networkx as nx
import sumolib


def get_tls_node_ids(net: sumolib.net.Net) -> set[str]:
    """Junction-node ids that are traffic-signal controlled.

    Uses ``node.getTLSID()`` (non-empty iff the junction has a controlling
    ``tlLogic`` program) rather than ``node.getType() == "traffic_light"``
    (unreliable — e.g. SUMO uses ``traffic_light_right_on_red`` for some
    networks) or ``net.getTrafficLights()`` (returns *program* ids, which
    for a "joined"/clustered signal can differ from every node id it
    controls — e.g. ``ingolstadt21`` has junction node
    ``cluster_1041665625_cluster_1387938793_...`` controlled by program
    ``gneJ143``; ``getTrafficLights()`` would return ``gneJ143``, which
    doesn't exist as a node and breaks graph construction).

    Road-network edges reference junction *node* ids
    (``edge.getFromNode().getID()``), so node ids are the correct id space
    for graph construction here. Note this can diverge from
    ``traci.trafficlight.getIDList()`` (program ids) for joined signals —
    see ``plans/tls-only-clustering-scope.md`` for how that's reconciled
    downstream.
    """
    return {node.getID() for node in net.getNodes() if node.getTLSID()}


def _undirected_adjacency(net: sumolib.net.Net, weight_fn: Callable) -> dict[str, dict[str, float]]:
    """node_id -> {neighbor_id: summed_edge_weight}, ignoring internal edges.

    Parallel edges between the same pair of nodes (e.g. both directions of a
    two-way street, each a separate SUMO edge) are summed into one entry.
    """
    adjacency: dict[str, dict[str, float]] = {}
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        u = edge.getFromNode().getID()
        v = edge.getToNode().getID()
        if u == v:
            continue
        w = float(weight_fn(edge))
        adjacency.setdefault(u, {})
        adjacency.setdefault(v, {})
        adjacency[u][v] = adjacency[u].get(v, 0.0) + w
        adjacency[v][u] = adjacency[v].get(u, 0.0) + w
    return adjacency


def build_tls_projected_graph(
    net: sumolib.net.Net,
    tls_ids: set[str],
    weight_fn: Callable = lambda edge: 1.0,
) -> nx.Graph:
    """Undirected graph whose vertices are exactly ``tls_ids``.

    An edge ``(a, b)`` exists iff the road network has a path between
    traffic lights ``a`` and ``b`` that passes through no other traffic
    light; its weight is the sum of ``weight_fn(edge)`` over every edge on
    every such path. This is a graph *contraction* over runs of
    non-signalized junctions, not a filter — dropping non-TLS nodes
    outright would disconnect signals that are only linked via
    unsignalized streets. When two traffic lights are connected by more
    than one qualifying path, their weights are summed (parallel routes
    make for a stronger regional tie).

    Isolated traffic lights (no path to any other TLS node) are still added
    as vertices with no edges, so every traffic light is guaranteed to
    appear in the result — callers must not silently drop them.
    """
    adjacency = _undirected_adjacency(net, weight_fn)
    graph = nx.Graph()

    for tls_id in sorted(tls_ids):
        node = net.getNode(tls_id)
        x, y = node.getCoord()
        graph.add_node(tls_id, x=float(x), y=float(y))

    for start in sorted(tls_ids):
        if start not in adjacency:
            continue

        # BFS outward from `start`, walking through non-TLS nodes only;
        # whenever another TLS node is reached, record the accumulated
        # weight as one edge and do not continue past it.
        visited = {start}
        queue = deque((neighbor, weight) for neighbor, weight in adjacency[start].items())

        while queue:
            node_id, acc_weight = queue.popleft()

            if node_id in tls_ids:
                if node_id == start:
                    continue
                if graph.has_edge(start, node_id):
                    graph[start][node_id]["weight"] += acc_weight
                else:
                    graph.add_edge(start, node_id, weight=acc_weight)
                continue

            if node_id in visited:
                continue
            visited.add(node_id)

            for neighbor, w in adjacency.get(node_id, {}).items():
                queue.append((neighbor, acc_weight + w))

    return graph
