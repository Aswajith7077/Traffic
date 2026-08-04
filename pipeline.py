#!/usr/bin/env python3
"""
End-to-end pipeline for traffic signal control on the included SUMO scenarios.

Datasets live in scenarios/<scenario>/ and are used directly (no copies).
Run region-splitting, training, and evaluation for any of them.

Usage:
    python pipeline.py                       # run full pipeline (manhattan)
    python pipeline.py --scenario arterial4x4 --route 42
    python pipeline.py baseline              # step 1 only
    python pipeline.py cluster               # step 2 only
    python pipeline.py copy                  # step 3 only
    python pipeline.py train                 # step 4 only
    python pipeline.py eval                  # step 5 only

Scenarios: manhattan, arterial4x4, cologne8, ingolstadt21.
(grid4x4 is excluded: its network has no traffic lights, so RL signal
control cannot run on it.)

arterial4x4 ships 1400 demand route files (arterial4x4_N.rou.xml); pick
one with --route N (default: the sumocfg's default, _1).
"""

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python3"
SUMO_BIN = shutil.which("sumo")

REGION_SPLITTING = ROOT / "region-splitting"
ADVESARIAL = ROOT / "advesarial"
SCENARIOS = ROOT / "scenarios"

SCENARIO_CHOICES = ["manhattan", "arterial4x4", "cologne8", "ingolstadt21"]


def banner(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def run(cmd, cwd, env):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
    return result.returncode == 0


def pipeline_env(scenario, route=None, episodes=None, episode_steps=None, save_every=None):
    env = dict(os.environ)
    env["TRAFFIC_SCENARIO"] = scenario
    if route is not None:
        env["TRAFFIC_ROUTE"] = str(route)
    if episodes is not None:
        env["TRAFFIC_EPISODES"] = str(episodes)
    if episode_steps is not None:
        env["TRAFFIC_EPISODE_STEPS"] = str(episode_steps)
    if save_every is not None:
        env["TRAFFIC_SAVE_EVERY"] = str(save_every)
    return env


def scenario_dir(scenario):
    return SCENARIOS / scenario


def step_baseline(scenario, route):
    banner(f"Step 1/5: Baseline SUMO simulation (actuated control) — {scenario}")

    sdir = scenario_dir(scenario)
    if not sdir.exists():
        print(f"  ERROR: scenario directory not found: {sdir}")
        return False

    for f in ["tripinfos.xml", "stats.xml"]:
        (sdir / f).unlink(missing_ok=True)

    if SUMO_BIN is None:
        print("  ERROR: 'sumo' not found on PATH. Is SUMO_HOME set?")
        return False

    cmd = [
        SUMO_BIN,
        "-c",
        f"{scenario}.sumocfg",
        "--no-step-log",
        "--tripinfo-output",
        str(sdir / "tripinfos.xml"),
        "--statistic-output",
        str(sdir / "stats.xml"),
    ]
    if route is not None:
        route_file = sdir / f"{scenario}_{route}.rou.xml"
        if route_file.exists():
            cmd += ["--route-files", str(route_file)]
        else:
            print(f"  ERROR: route file not found: {route_file}")
            return False

    ok = run(cmd, cwd=sdir, env=pipeline_env(scenario, route))
    if not ok:
        return False

    tripinfo = sdir / "tripinfos.xml"
    if not tripinfo.exists():
        print("  tripinfos.xml not found (no output generated)")
        return True

    tree = ET.parse(str(tripinfo))
    trips = tree.getroot().findall("tripinfo")

    if not trips:
        print("  No vehicles completed during the simulation")
        return True

    durations = [float(t.get("duration", 0)) for t in trips]
    waits = [float(t.get("waitingTime", 0)) for t in trips]
    lengths = [float(t.get("routeLength", 0)) for t in trips]
    depart_delays = [float(t.get("departDelay", 0)) for t in trips]

    n = len(trips)
    print("\n  BASELINE METRICS  (actuated control)")
    print(f"  {'Completed vehicles:':<28} {n}")
    print(f"  {'Avg travel time:':<28} {sum(durations) / n:.2f} s")
    print(f"  {'Avg waiting time:':<28} {sum(waits) / n:.2f} s")
    print(f"  {'Avg route length:':<28} {sum(lengths) / n:.2f} m")
    print(f"  {'Avg depart delay:':<28} {sum(depart_delays) / n:.2f} s")

    metrics_file = ADVESARIAL / "metrics.txt"
    with open(str(metrics_file), "a") as f:
        f.write(f"--- Baseline (actuated) [{scenario}] ---\n")
        f.write(f"Vehicles: {n}\n")
        f.write(f"Avg travel time: {sum(durations) / n:.2f}s\n")
        f.write(f"Avg waiting time: {sum(waits) / n:.2f}s\n\n")

    return True


def step_cluster(scenario, route):
    banner(f"Step 2/5: Region splitting (Leiden clustering) — {scenario}")

    return run(
        [VENV_PYTHON, "main.py", "--scenario", scenario],
        cwd=REGION_SPLITTING,
        env=pipeline_env(scenario, route),
    )


def step_copy(scenario, route):
    banner(f"Step 3/5: Copying cluster files to advesarial — {scenario}")

    src = REGION_SPLITTING / "clusters" / "leiden" / f"{scenario}_clusters.json"
    dst_dir = ADVESARIAL / "clusters" / "leiden"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{scenario}_clusters.json"

    if not src.exists():
        print(f"  Source not found: {src}")
        print("  Run 'python pipeline.py cluster' first")
        return False

    shutil.copy2(str(src), str(dst))
    print(f"  Copied: {src}")
    print(f"  To:     {dst}")
    return True


def step_train(scenario, route, episodes, episode_steps, save_every):
    banner(f"Step 4/5: Training RL agent ({episodes} episodes of {episode_steps}s) — {scenario}")

    return run(
        [VENV_PYTHON, "src/sample.py"],
        cwd=ADVESARIAL,
        env=pipeline_env(scenario, route, episodes, episode_steps, save_every),
    )


def step_eval(scenario, route):
    banner(f"Step 5/5: Evaluating trained model ({scenario}) — latest checkpoint in newest run folder")

    return run(
        [VENV_PYTHON, "src/evaluate.py", "--steps", "500"],
        cwd=ADVESARIAL,
        env=pipeline_env(scenario, route),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for traffic signal control on the included SUMO scenarios"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "baseline", "cluster", "copy", "train", "eval"],
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--scenario",
        default="manhattan",
        choices=SCENARIO_CHOICES,
        help="Scenario dataset to use (default: manhattan). "
        "grid4x4 is excluded (no traffic lights in the network).",
    )
    parser.add_argument(
        "--route",
        type=int,
        default=None,
        help="Demand route file index for scenarios with multiple route files "
        "e.g. arterial4x4_N.rou.xml (default: the sumocfg's default route)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to train (default: 10)",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=1000,
        help="Sim-seconds per training episode (default: 1000)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save a checkpoint every N episodes (default: 10)",
    )
    args = parser.parse_args()

    scenario = args.scenario
    route = args.route
    episodes = args.episodes
    episode_steps = args.episode_steps
    save_every = args.save_every

    if route is not None and not (scenario_dir(scenario) / f"{scenario}_{route}.rou.xml").exists():
        print(f"  ERROR: route file not found: {scenario_dir(scenario) / f'{scenario}_{route}.rou.xml'}")
        sys.exit(1)

    steps = {
        "all": [
            lambda: step_baseline(scenario, route),
            lambda: step_cluster(scenario, route),
            lambda: step_copy(scenario, route),
            lambda: step_train(scenario, route, episodes, episode_steps, save_every),
            lambda: step_eval(scenario, route),
        ],
        "baseline": [lambda: step_baseline(scenario, route)],
        "cluster": [lambda: step_cluster(scenario, route)],
        "copy": [lambda: step_copy(scenario, route)],
        "train": [lambda: step_train(scenario, route, episodes, episode_steps, save_every)],
        "eval": [lambda: step_eval(scenario, route)],
    }

    for fn in steps[args.command]:
        if not fn():
            print(f"\n  Pipeline stopped (scenario: {scenario})")
            sys.exit(1)

    print(f"\n  Pipeline '{args.command}' completed successfully ({scenario})")


if __name__ == "__main__":
    main()
