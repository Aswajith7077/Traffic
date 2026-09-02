#!/usr/bin/env python3
"""Collate training + evaluation metrics from advesarial/metrics.txt.

Usage:
    python collect_metrics.py            # latest training + eval snapshot per scenario/method
    python collect_metrics.py --all      # every snapshot
"""

import argparse
import re

METRICS = "metrics.txt"

EVAL_KEYS = [
    "Completed Vehicles",
    "Average Queue Length",
    "Average Waiting Time",
    "Average Travel Time",
    "Average Delay Time",
    "Peak Total Queue Length",
]
TRAIN_KEYS = ["Episode", "Steps taken", "Latest Meta Loss", "Latest AC Loss", "Latest Sub Loss", "Latest Reward"]


def parse():
    sections = []
    current = None
    with open(METRICS) as f:
        for line in f:
            line = line.rstrip("\n")
            header = re.match(r"^--- (\w+) Snapshot: (.+?) ---$", line)
            if header:
                current = {"kind": header.group(1), "id": header.group(2), "fields": {}}
                sections.append(current)
                continue
            if current is None:
                continue
            if line == "":
                continue
            m = re.match(r"^([A-Za-z ]+): (.*)$", line)
            if m:
                current["fields"][m.group(1).strip()] = m.group(2).strip()
    return sections


def fmt(value):
    import re

    if isinstance(value, str):
        m = re.match(r"^[\d.]+", value.strip())
        value = m.group(0) if m else value
    try:
        f = float(value)
        return f"{f:.2f}"
    except (TypeError, ValueError):
        return str(value)


def render(sections, all_snapshots):
    evals = [s for s in sections if s["kind"] == "Evaluation"]
    trains = [s for s in sections if s["kind"] == "Training"]

    if all_snapshots:
        latest_evals, latest_trains = evals, trains
    else:
        latest_evals, latest_trains = {}, {}
        for s in evals:
            key = (s["fields"].get("Scenario"), s["fields"].get("Cluster Method"))
            latest_evals[key] = s
        for s in trains:
            key = (s["id"].split()[0], s["fields"].get("Cluster Method"))
            latest_trains[key] = s
        latest_evals = list(latest_evals.values())
        latest_trains = list(latest_trains.values())

    print("=" * 100)
    print("EVALUATION METRICS (average travel time / average delay time)")
    print("=" * 100)
    header = f"{'scenario':<14s} {'method':<15s} {'vehicles':>9s} {'queue':>8s}"
    header += f" {'wait':>8s} {'travel(s)':>9s} {'delay(s)':>9s}"
    print(header)
    print("-" * 100)
    for s in latest_evals:
        f = s["fields"]
        scenario = f.get("Scenario", "-")
        method = f.get("Cluster Method", "-")
        print(
            f"{scenario:<14s} {method:<15s} {f.get('Completed Vehicles', '-'):>9s} "
            f"{fmt(f.get('Average Queue Length', '-')):>8s} {fmt(f.get('Average Waiting Time', '-')):>8s} "
            f"{fmt(f.get('Average Travel Time', '-')):>9s} {fmt(f.get('Average Delay Time', '-')):>9s}"
        )

    print()
    print("=" * 100)
    print("TRAINING METRICS (latest snapshot per scenario/method)")
    print("=" * 100)
    header = f"{'scenario':<14s} {'method':<15s} {'steps':>6s} {'meta_loss':>10s}"
    header += f" {'ac_loss':>10s} {'sub_loss':>10s} {'reward':>8s}"
    print(header)
    print("-" * 100)
    for s in latest_trains:
        f = s["fields"]
        scenario = s["id"].split()[0]
        method = f.get("Cluster Method", "-")
        print(
            f"{scenario:<14s} {method:<15s} {f.get('Steps taken', '-'):>6s} "
            f"{fmt(f.get('Latest Meta Loss', '-')):>10s} {fmt(f.get('Latest AC Loss', '-')):>10s} "
            f"{fmt(f.get('Latest Sub Loss', '-')):>10s} {fmt(f.get('Latest Reward', '-')):>8s}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate training + evaluation metrics")
    parser.add_argument("--all", action="store_true", help="Show every snapshot, not just the latest per key")
    args = parser.parse_args()
    render(parse(), args.all)
