"""Reads paper_metrics/eval_results.csv and prints the two LaTeX tables
(Table 1: with/without city-wide EMV routing; Table 2: Fixed/Max-Pressure/
Algorithm baselines) plus a plain-text summary, for pasting into main.tex.
"""

import csv
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")


def load_rows():
    with open(CSV_PATH) as f:
        return {row["label"]: row for row in csv.DictReader(f)}


def main():
    rows = load_rows()

    print("=== Table 1: with vs. without city-wide EMV routing ===")
    table1_rows = [
        ("without_emv", "Without city-wide EMV routing"),
        ("with_emv", "With city-wide EMV routing"),
    ]
    for label, row_label in table1_rows:
        r = rows[label]
        print(
            f"{row_label} & {r['avg_travel_time']} & {float(r['avg_waiting_time']) * int(r['completed_vehicles']):.2f}"
            f" & {r['avg_waiting_time']} \\\\"
        )
        print(
            f"  (EMV n={r['emv_count']}, mean resp.={r['emv_avg_response_time']}s, "
            f"P95={r['emv_p95_response_time']}s; JFI incident={r['jain_intersection_incident']} "
            f"normal={r['jain_intersection_normal']})"
        )

    print("\n=== Table 2: Fixed / Max-Pressure / Algorithm (= without_emv) ===")
    for label, row_label in [("fixed", "Fixed"), ("max_pressure", "Max Pressure"), ("without_emv", "Algorithm")]:
        r = rows[label]
        print(
            f"{row_label} & {r['avg_travel_time']} & {float(r['avg_waiting_time']) * int(r['completed_vehicles']):.2f}"
            f" & {r['avg_waiting_time']} \\\\"
        )

    print("\n=== Full rows ===")
    for label, r in rows.items():
        print(label, dict(r))


if __name__ == "__main__":
    main()
