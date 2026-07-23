import argparse
import json
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

import traci

sys.path.insert(0, "src")
from benchmark import evaluate_episode


def _run_benchmark(config_path, max_steps, use_gui, route_lookup_path):
    if not config_path.endswith(".sumocfg"):
        sys.exit(f"Config path must end with .sumocfg: {config_path}")
    if not os.path.exists(config_path):
        sys.exit(f"Config file not found: {config_path}")

    route_length_lookup = None
    if route_lookup_path:
        if not os.path.exists(route_lookup_path):
            sys.exit(f"Route lookup file not found: {route_lookup_path}")
        with open(route_lookup_path) as f:
            route_length_lookup = json.load(f)
        print(f"Loaded route lookup: {len(route_length_lookup)} routes")

    binary = "sumo-gui" if use_gui else "sumo"
    cmd = [
        binary,
        "-c",
        os.path.abspath(config_path),
        "--step-length",
        "1.0",
        "--start",
        "--no-step-log",
        "--verbose",
        "false",
    ]
    if use_gui:
        cmd += ["--delay", "50"]

    print(f"Starting SUMO with config: {config_path}")
    print(f"Max steps: {max_steps}")
    print()

    traci.start(cmd)

    try:
        metrics = evaluate_episode(traci, route_length_lookup, max_steps)
    finally:
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for SUMO traffic signal control")
    parser.add_argument(
        "--config",
        type=str,
        default="sumo/osm.sumocfg",
        help="Path to SUMO config file (relative to advesarial/)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO GUI",
    )
    parser.add_argument(
        "--route-lookup",
        type=str,
        default=None,
        help="Path to JSON file with precomputed route ideal times",
    )

    args = parser.parse_args()

    metrics = _run_benchmark(
        config_path=args.config,
        max_steps=args.steps,
        use_gui=args.gui,
        route_lookup_path=args.route_lookup,
    )

    print()
    print("=" * 44)
    print("          BENCHMARK RESULTS")
    print("=" * 44)
    print(f"  Config                   : {args.config}")
    print(f"  Steps evaluated          : {args.steps}")
    print(f"  Completed vehicles       : {metrics['num_completed_vehicles']}")
    print()
    print(f"  Average Trip Time (ATT)  : {metrics['average_trip_time']:>8.2f} s")
    print(f"  Average Delay Time (ADT) : {metrics['average_delay_time']:>8.2f} s")
    print()
    print(f"  Median Trip Time         : {metrics['median_trip_time']:>8.2f} s")
    print(f"  Std Trip Time            : {metrics['std_trip_time']:>8.2f} s")
    print(f"  95th %ile Trip Time      : {metrics['p95_trip_time']:>8.2f} s")
    print(f"  Median Delay Time        : {metrics['median_delay_time']:>8.2f} s")
    print(f"  Std Delay Time           : {metrics['std_delay_time']:>8.2f} s")
    print(f"  95th %ile Delay Time     : {metrics['p95_delay_time']:>8.2f} s")
    print("=" * 44)


if __name__ == "__main__":
    main()
