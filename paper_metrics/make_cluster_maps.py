"""before_cluster.png / after_cluster.png for Cologne8 (Section 3.1/5).

"Before" = all 8 signalized intersections, one color (no regional
abstraction yet). "After" = colored by cluster label.

Cologne8 only has 8 traffic lights, and the paper's production DBSCAN
partition (AGENTS.md) auto-tunes to a single region on a network this small
(confirmed: advesarial/clusters/dbscan/cologne8_clusters.json has n_clusters=1,
a real, reportable consequence of DBSCAN's density assumption on a small,
dense network). Leiden gives a 2-way split instead, which is the one that
actually demonstrates "same color = same cluster" visually, so "after" uses
Leiden; the caption says which method was used and notes DBSCAN's result.

No SUMO process needed -- sumolib.net.readNet parses the net file directly.
"""

import json
import os

import matplotlib.pyplot as plt
import sumolib.net

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_PATH = os.path.join(REPO_ROOT, "scenarios", "cologne8", "cologne8.net.xml")
CLUSTER_PATH = os.path.join(REPO_ROOT, "advesarial", "clusters", "leiden", "cologne8_clusters.json")
OUT_DIR = os.path.join(REPO_ROOT, "paper_metrics")

CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _tls_coords(net):
    coords = {}
    for tls_id in sorted({tl.getID() for tl in net.getTrafficLights()}):
        node = net.getNode(tls_id)
        coords[tls_id] = node.getCoord()
    return coords


def _draw_background(ax, net):
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        shape = edge.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, color="#cccccc", linewidth=0.6, zorder=1)


def main():
    net = sumolib.net.readNet(NET_PATH)
    coords = _tls_coords(net)

    with open(CLUSTER_PATH) as f:
        clusters = json.load(f)["clusters"]
    node_to_cluster = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            node_to_cluster[node] = cluster_id

    # before_cluster.png: every intersection, single color
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_background(ax, net)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ax.scatter(xs, ys, s=160, color="#444444", edgecolor="black", zorder=3)
    for tls_id, (x, y) in coords.items():
        ax.annotate(tls_id, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_title("Cologne8: signalized intersections before clustering")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "before_cluster.png"), dpi=150)
    plt.close(fig)

    # after_cluster.png: colored by cluster label
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_background(ax, net)
    cluster_ids = sorted(clusters.keys())
    color_map = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}
    for cid in cluster_ids:
        pts = [coords[n] for n in clusters[cid] if n in coords]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=160, color=color_map[cid], edgecolor="black", zorder=3)
    for tls_id, (x, y) in coords.items():
        ax.annotate(tls_id, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_title("Cologne8: intersections after Leiden clustering\n(same color = same cluster)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "after_cluster.png"), dpi=150)
    plt.close(fig)

    assert len(coords) == 8, f"expected 8 Cologne8 TLS, found {len(coords)}"
    print(f"Wrote before_cluster.png / after_cluster.png ({len(cluster_ids)} clusters, {len(coords)} intersections)")


if __name__ == "__main__":
    main()
