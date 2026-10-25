"""before_cluster_<scenario>.png / after_cluster_<scenario>.png for the 4
scenarios other than Cologne8 (which already has its own before/after pair
under Section 3.1 via make_cluster_maps.py), for Section 6's clustering
figures.

Per scenario, "after" uses the production DBSCAN partition
(advesarial/clusters/dbscan/) when it actually splits the network into more
than one region, falling back to Leiden (advesarial/clusters/leiden/) when
DBSCAN collapses to a single region -- the same convention
make_cluster_maps.py already uses for Cologne8, and for the same reason
(a single-region "after" plot is visually identical to "before" and doesn't
demonstrate the clustering).

No SUMO process needed -- sumolib.net.readNet parses the net file directly.
"""

import json
import os

import matplotlib.pyplot as plt
import sumolib.net

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "paper_metrics")

CLUSTER_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f",
]

# scenario -> method to use for "after" (chosen per-scenario: dbscan when it
# actually splits the network into >1 region, leiden otherwise).
SCENARIOS = {
    "arterial4x4": "leiden",
    "grid4x4": "leiden",
    "ingolstadt21": "dbscan",
    "manhattan": "dbscan",
}


def _tls_coords(net):
    # node.getTLSID(), not net.getTrafficLights() (program ids, which for a
    # "joined" signal can diverge from every node id it controls -- e.g.
    # ingolstadt21, see region-splitting/services/topology.py:get_tls_node_ids).
    return {node.getID(): node.getCoord() for node in net.getNodes() if node.getTLSID()}


def _draw_background(ax, net):
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        shape = edge.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, color="#cccccc", linewidth=0.5, zorder=1)


def make_pair(scenario, method):
    net_path = os.path.join(REPO_ROOT, "scenarios", scenario, f"{scenario}.net.xml")
    cluster_path = os.path.join(REPO_ROOT, "advesarial", "clusters", method, f"{scenario}_clusters.json")

    net = sumolib.net.readNet(net_path)
    coords = _tls_coords(net)

    with open(cluster_path) as f:
        clusters = json.load(f)["clusters"]
    clusters = {cid: [n for n in nodes if n in coords] for cid, nodes in clusters.items()}
    clusters = {cid: nodes for cid, nodes in clusters.items() if nodes}

    point_size = 160 if len(coords) <= 30 else 40

    # No per-node text labels: some scenarios (e.g. ingolstadt21) have long
    # joined-cluster TLS ids that overflow tight_layout() and break the
    # figure; the color-by-cluster coding doesn't need per-node ids anyway.

    # before: every intersection, single color, no regional abstraction
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_background(ax, net)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    ax.scatter(xs, ys, s=point_size, color="#444444", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_title(f"{scenario}: signalized intersections before clustering")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"before_cluster_{scenario}.png"), dpi=150)
    plt.close(fig)

    # after: colored by cluster label
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_background(ax, net)
    cluster_ids = sorted(clusters.keys())
    color_map = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}
    for cid in cluster_ids:
        pts = [coords[n] for n in clusters[cid]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(
            xs, ys, s=point_size, color=color_map[cid], edgecolor="black", linewidth=0.3, zorder=3,
            label=f"Cluster {cid} (n={len(pts)})",
        )
    ax.set_title(f"{scenario}: intersections after {method.upper()} clustering\n(same color = same cluster)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    if len(cluster_ids) <= 12:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"after_cluster_{scenario}.png"), dpi=150)
    plt.close(fig)

    print(f"{scenario}: {len(coords)} TLS, {len(cluster_ids)} clusters ({method}) -> "
          f"before_cluster_{scenario}.png / after_cluster_{scenario}.png")


def main():
    for scenario, method in SCENARIOS.items():
        make_pair(scenario, method)


if __name__ == "__main__":
    main()
