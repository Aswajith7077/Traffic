"""Reads paper_metrics/eval_results.csv (after run_full_campaign.py) and
prints the two LaTeX 3D tables (scenario x method/config, cols = metrics)
for Section 6: the Fixed/Max-Pressure/Algorithm comparison across all 5
scenarios, and the with/without city-wide EMV routing comparison across all
5 scenarios. Paste the printed \\multirow blocks into main.tex's
tab:baseline-comparison / tab:emv-routing tables.

Run after run_full_campaign.py finishes: `python paper_metrics/make_tables_full.py`.
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


def f(row, key, fmt="{:.2f}"):
    try:
        return fmt.format(float(row[key]))
    except (KeyError, ValueError, TypeError):
        return "--"


def main():
    rows = load_rows()
    missing = [f"{s}_{c}" for s in SCENARIOS for c in ("fixed", "max_pressure", "without_emv", "with_emv")
               if f"{s}_{c}" not in rows]
    if missing:
        print(f"WARNING: missing rows (campaign incomplete?): {missing}\n")

    print("=" * 70)
    print("Table: Fixed / Max Pressure / Algorithm, all 5 scenarios")
    print("Columns: Avg Travel Time [s], Avg Waiting Time [s], Avg Delay Time [s], Jain's Index")
    print("=" * 70)
    for s in SCENARIOS:
        disp = SCENARIO_DISPLAY[s]
        print(f"\\multirow{{3}}{{*}}{{{disp}}}")
        for label, method in [("fixed", "Fixed"), ("max_pressure", "Max Pressure"), ("without_emv", "Algorithm")]:
            key = f"{s}_{label}"
            if key not in rows:
                print(f"  & {method} & -- & -- & -- & -- \\\\")
                continue
            r = rows[key]
            print(
                f"  & {method} & {f(r, 'avg_travel_time')} & {f(r, 'avg_waiting_time')} & "
                f"{f(r, 'avg_delay_time')} & {f(r, 'jain_network', '{:.4f}')} \\\\"
            )
        print("\\midrule")

    print("\n" + "=" * 70)
    print("Table: With vs. without city-wide EMV routing, all 5 scenarios")
    print("Columns: Avg Travel Time [s], Avg Waiting Time [s], EMV mean resp. [s], EMV P95 resp. [s]")
    print("=" * 70)
    for s in SCENARIOS:
        disp = SCENARIO_DISPLAY[s]
        print(f"\\multirow{{2}}{{*}}{{{disp}}}")
        for label, config in [("without_emv", "Without city-wide EMV routing"), ("with_emv", "With city-wide EMV routing")]:
            key = f"{s}_{label}"
            if key not in rows:
                print(f"  & {config} & -- & -- & -- & -- \\\\")
                continue
            r = rows[key]
            print(
                f"  & {config} & {f(r, 'avg_travel_time')} & {f(r, 'avg_waiting_time')} & "
                f"{f(r, 'emv_avg_response_time')} & {f(r, 'emv_p95_response_time')} \\\\"
            )
        print("\\midrule")

    print("\n=== Full rows (for sanity-checking / further columns) ===")
    for s in SCENARIOS:
        for c in ("fixed", "max_pressure", "without_emv", "with_emv"):
            key = f"{s}_{c}"
            if key in rows:
                print(key, dict(rows[key]))


if __name__ == "__main__":
    main()
