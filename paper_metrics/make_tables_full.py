"""Reads paper_metrics/eval_results.csv (after run_full_campaign.py +
per-scenario --only-with-emv follow-up passes) and prints two ready-to-paste
LaTeX tables for Section 6: a 5-scenario x {with_emv, without_emv} EMV-routing
comparison, and a 5-scenario x {Fixed, Max Pressure, Algorithm} baseline
comparison. Columns: Total Avg. Trip Time [s], Total Waiting Time [s], Total
Avg. Waiting Time [s] -- matching the original single-scenario tables'
column convention. Lowest (best) value per column within each scenario's row
group is bolded, HiLight-table style.

Run after the full campaign (including each scenario's --only-with-emv
follow-up) has finished: `python paper_metrics/make_tables_full.py`.
"""

import csv
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
SCENARIOS = ["cologne8", "arterial4x4", "grid4x4", "ingolstadt21", "manhattan"]
SCENARIO_DISPLAY = {
    "cologne8": "Cologne8",
    "arterial4x4": "Arterial4x4",
    "grid4x4": "Grid4x4",
    "ingolstadt21": "Ingolstadt21",
    "manhattan": "Manhattan",
}


def load_rows():
    rows = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            rows[row["label"]] = row  # last write wins -- most recent run for that label
    return rows


def metrics_for(row):
    att = float(row["avg_travel_time"])
    total_wait = att and float(row["avg_waiting_time"]) * int(row["completed_vehicles"])
    avg_wait = float(row["avg_waiting_time"])
    return att, total_wait, avg_wait


def fmt_row(scenario_label, row_label, values, bold_mask, is_first):
    cells = []
    for v, bold in zip(values, bold_mask):
        s = f"{v:.2f}" if v is not None else "--"
        cells.append(f"\\textbf{{{s}}}" if bold else s)
    prefix = scenario_label if is_first else ""
    return f"{prefix} & {row_label} & " + " & ".join(cells) + " \\\\"


def emit_group(rows, scenario, members, missing_warnings):
    """members: list of (label_suffix, display_name)."""
    data = []
    for suffix, disp in members:
        key = f"{scenario}_{suffix}"
        if key not in rows:
            missing_warnings.append(key)
            data.append((disp, None))
            continue
        data.append((disp, metrics_for(rows[key])))

    # bold the min per column among members that have data
    n_cols = 3
    best = []
    for c in range(n_cols):
        vals = [d[1][c] for d in data if d[1] is not None]
        best.append(min(vals) if vals else None)

    lines = []
    n = f"\\multirow{{{len(members)}}}{{*}}{{{SCENARIO_DISPLAY[scenario]}}}"
    for i, (disp, vals) in enumerate(data):
        if vals is None:
            bold_mask = [False] * n_cols
            values = [None] * n_cols
        else:
            values = vals
            bold_mask = [best[c] is not None and abs(values[c] - best[c]) < 1e-9 for c in range(n_cols)]
        lines.append(fmt_row(n, disp, values, bold_mask, i == 0))
    return lines


def main():
    rows = load_rows()
    missing = []

    print("=" * 70)
    print("TABLE: EMV routing comparison, all 5 scenarios")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Base model with vs.\ without city-wide EMV routing, all 5 scenarios "
          r"(5 training episodes + behavior-cloning warm-start, seed 0). Lowest value per "
          r"scenario in bold.}")
    print(r"\label{tab:emv-routing}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Scenario & Configuration & Total Avg.\ Trip Time [s] & Total Waiting Time [s] & "
          r"Total Avg.\ Waiting Time [s] \\")
    print(r"\midrule")
    for i, scenario in enumerate(SCENARIOS):
        members = [("without_emv", "Without city-wide EMV routing"), ("with_emv", "With city-wide EMV routing")]
        for line in emit_group(rows, scenario, members, missing):
            print(line)
        print(r"\midrule" if i < len(SCENARIOS) - 1 else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print()
    print("=" * 70)
    print("TABLE: Fixed / Max Pressure / Algorithm, all 5 scenarios")
    print("=" * 70)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Comparison of traffic signal control methods, all 5 scenarios "
          r"(5 training episodes + behavior-cloning warm-start, seed 0, where applicable). "
          r"Lowest value per scenario in bold.}")
    print(r"\label{tab:baseline-comparison}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Scenario & Method & Total Avg.\ Trip Time [s] & Total Waiting Time [s] & "
          r"Total Avg.\ Waiting Time [s] \\")
    print(r"\midrule")
    for i, scenario in enumerate(SCENARIOS):
        members = [("fixed", "Fixed"), ("max_pressure", "Max Pressure"), ("without_emv", "Algorithm")]
        for line in emit_group(rows, scenario, members, missing):
            print(line)
        print(r"\midrule" if i < len(SCENARIOS) - 1 else r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    if missing:
        print("\n\nWARNING: missing rows (campaign incomplete?):")
        for m in missing:
            print(" ", m)


if __name__ == "__main__":
    main()
