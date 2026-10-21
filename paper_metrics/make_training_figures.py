"""training_loss_ac.png, reward_components.png, with_emv.png, without_emv.png
from the raw per-seed histories dumped by train_one.py. Mean line + shaded
std band across seeds, matplotlib/seaborn styled.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(config_name):
    runs = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, f"{config_name}_seed*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def _stack_series(runs, extractor):
    series = [extractor(r) for r in runs]
    series = [s for s in series if len(s) > 0]
    if not series:
        return None
    min_len = min(len(s) for s in series)
    return np.array([s[:min_len] for s in series])  # (seeds, min_len)


def _plot_band(ax, arr, label, color=None):
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    x = np.arange(len(mean))
    ax.plot(x, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


def make_ac_loss_figure(config_name, out_name, title):
    runs = _load(config_name)
    if not runs:
        print(f"skip {out_name}: no raw data for {config_name}")
        return
    arr = _stack_series(runs, lambda r: [c["l_ac"] for c in r["sub_loss_history"]])
    if arr is None:
        print(f"skip {out_name}: empty l_ac history")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_band(ax, arr, label=f"Actor-Critic loss (n={arr.shape[0]} seeds)", color="#1f77b4")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} ({arr.shape[0]} seeds, {arr.shape[1]} updates)")


def make_reward_components_figure():
    """R_eff/R_fair/R_emergency are Eq 25/29/30's raw, unweighted components
    (w1/w2/w3 only combine them into R_total, Eq 23) and differ in magnitude
    by orders of size -- R_fair and R_emergency being small next to R_eff is
    exactly the risk Section 4.2 itself flags for its lambda coefficients, not
    a plotting bug. A symlog y-axis keeps all three legible on one axis; the
    normalized overall reward R_total (z-scored by Environment._normalize_reward,
    an unrelated scale) gets its own right-hand axis.
    """
    runs = _load("with_emv")
    if not runs:
        print("skip reward_components.png: no raw data for with_emv")
        return

    overall = _stack_series(runs, lambda r: r["rewards_history"])
    r_eff = _stack_series(runs, lambda r: [c["r_eff"] for c in r["reward_components_history"]])
    r_fair = _stack_series(runs, lambda r: [c["r_fair"] for c in r["reward_components_history"]])
    r_emv = _stack_series(runs, lambda r: [c["r_emv"] for c in r["reward_components_history"]])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"overall": "#1f77b4", "r_eff": "#ff7f0e", "r_fair": "#2ca02c", "r_emv": "#d62728"}
    n_seeds = 0
    if r_eff is not None:
        _plot_band(ax, r_eff, "R_eff (efficiency, raw)", colors["r_eff"])
        n_seeds = r_eff.shape[0]
    if r_fair is not None:
        _plot_band(ax, r_fair, "R_fair (fairness, raw)", colors["r_fair"])
    if r_emv is not None:
        _plot_band(ax, r_emv, "R_emergency (EMV, raw)", colors["r_emv"])
    ax.set_yscale("symlog")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Raw reward component (symlog)")

    if overall is not None:
        ax2 = ax.twinx()
        _plot_band(ax2, overall, "Overall normalized reward R_total (right axis)", colors["overall"])
        ax2.set_ylabel("Normalized R_total")
        n_seeds = overall.shape[0]
        lines = ax.get_lines() + ax2.get_lines()
    else:
        lines = ax.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=8, framealpha=0.9)

    ax.set_title(f"Reward components during training (Cologne8, EMV routing on, n={n_seeds} seeds)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reward_components.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    make_ac_loss_figure(
        "with_emv",
        "training_loss_ac.png",
        "Actor-Critic training loss (Cologne8)",
    )
    make_ac_loss_figure(
        "with_emv",
        "with_emv.png",
        "Actor-Critic training loss — city-wide EMV routing ON",
    )
    make_ac_loss_figure(
        "without_emv",
        "without_emv.png",
        "Actor-Critic training loss — city-wide EMV routing OFF (local preemption only)",
    )
    make_reward_components_figure()


if __name__ == "__main__":
    main()
