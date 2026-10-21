"""Diagrammatic architecture figure for the paper's Section 3.10 pipeline (Eq 22):
G -> R -> Meta-Policy -> (F_g, (G_w, G_q)) -> Sub-Policies -> Signal Actions -> Traffic Environment.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_diagram.png")

STAGES = [
    ("Road Network\nG = (V, E)", "#f0f0f0"),
    ("Regional Partition\nR = {R_1..R_M}\n(DBSCAN / Louvain / Leiden)", "#cfe8f3"),
    ("Meta-Policy\nTransformer + LSTM\n-> F_g, (G_w, G_q)", "#f9d9a0"),
    ("Sub-Policies\nLocalEncoder + GAT\n+ ActorCritic\n(per intersection)", "#c9e4b5"),
    ("Signal Actions\n(safety-constrained\nmin-green / yellow / EMV)", "#f4b6b6"),
    ("Traffic Environment\n(SUMO via TraCI)", "#d9c6ec"),
]


def main():
    fig, ax = plt.subplots(figsize=(16, 3.2))
    n = len(STAGES)
    box_w, box_h, gap = 2.4, 1.4, 0.45
    x = 0.2

    centers = []
    for label, color in STAGES:
        box = mpatches.FancyBboxPatch(
            (x, 0.5),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, 0.5 + box_h / 2, label, ha="center", va="center", fontsize=8.5, wrap=True)
        centers.append(x + box_w)
        x += box_w + gap

    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(centers[i] + gap, 0.5 + box_h / 2),
            xytext=(centers[i], 0.5 + box_h / 2),
            arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.4},
        )

    ax.set_xlim(0, x)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    ax.set_title("FairLight end-to-end architecture (Section 3.10, Eq 22)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
