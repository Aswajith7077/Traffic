"""Generalizes make_training_figures.py (which only covered Cologne8) across
all 5 scenarios: for each scenario, an Actor-Critic loss figure and a reward-
components figure for both with_emv and without_emv, from the raw per-seed
histories train_one.py/run_full_campaign.py already dump to paper_metrics/raw/.

Cologne8 keeps its original, paper-referenced filenames
(training_loss_ac.png, with_emv.png, without_emv.png, reward_components.png,
produced by make_training_figures.py) untouched. This script only adds the
scenario-suffixed figures for the other 4 scenarios plus a with/without-emv
reward-components pair for every scenario including Cologne8 (the original
script only ever plotted with_emv's reward components).

Run after run_full_campaign.py has produced paper_metrics/raw/<scenario>_*
for the scenarios you want figures for:
`python paper_metrics/make_training_figures_all.py`.
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
SCENARIOS = ["cologne8", "arterial4x4", "grid4x4", "ingolstadt21", "manhattan"]
SCENARIO_DISPLAY = {
    "cologne8": "Cologne8",
    "arterial4x4": "Arterial4x4",
    "grid4x4": "Grid4x4",
    "ingolstadt21": "Ingolstadt21",
    "manhattan": "Manhattan",
}


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
    return np.array([s[:min_len] for s in series])


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
        return False
    arr = _stack_series(runs, lambda r: [c["l_ac"] for c in r["sub_loss_history"]])
    if arr is None:
        print(f"skip {out_name}: empty l_ac history")
        return False

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
    return True


def make_reward_components_figure(config_name, out_name, title):
    runs = _load(config_name)
    if not runs:
        print(f"skip {out_name}: no raw data for {config_name}")
        return False

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

    ax.set_title(title)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path} (n={n_seeds} seeds)")
    return True


def main():
    made, skipped = [], []
    for scenario in SCENARIOS:
        disp = SCENARIO_DISPLAY[scenario]
        for emv_flag, emv_label in [("with_emv", "ON"), ("without_emv", "OFF (local preemption only)")]:
            config_name = f"{scenario}_{emv_flag}"

            if scenario == "cologne8":
                ac_out = "with_emv.png" if emv_flag == "with_emv" else "without_emv.png"
            else:
                ac_out = f"training_loss_ac_{scenario}_{emv_flag}.png"
            ok = make_ac_loss_figure(config_name, ac_out, f"Actor-Critic training loss -- {disp}, EMV routing {emv_label}")
            (made if ok else skipped).append(ac_out)

            rc_out = "reward_components.png" if config_name == "cologne8_with_emv" else f"reward_components_{scenario}_{emv_flag}.png"
            ok = make_reward_components_figure(
                config_name, rc_out, f"Reward components during training -- {disp}, EMV routing {emv_label}"
            )
            (made if ok else skipped).append(rc_out)

    print(f"\n{len(made)} figures written, {len(skipped)} skipped (missing raw data):")
    for s in skipped:
        print(" ", s)


if __name__ == "__main__":
    main()
