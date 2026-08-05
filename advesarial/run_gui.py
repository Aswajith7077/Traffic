import argparse
import glob
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

# Anchor all relative paths to this script's directory (advesarial/) so the
# script works no matter where it is invoked from (repo root or advesarial/).
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

SCENARIOS_DIR = "../scenarios"
MODELS_DIR = "../models"
CLUSTERS_DIR = "clusters/leiden"


def discover_maps():
    """Return maps that have a sumocfg, a cluster file, and at least one trained run."""
    maps = []
    if not os.path.isdir(SCENARIOS_DIR):
        return maps

    for name in sorted(os.listdir(SCENARIOS_DIR)):
        sumo_cfg = os.path.join(SCENARIOS_DIR, name, f"{name}.sumocfg")
        cluster_file = os.path.join(CLUSTERS_DIR, f"{name}_clusters.json")
        runs = sorted(glob.glob(os.path.join(MODELS_DIR, name, "run_*")))
        has_checkpoint = any(
            glob.glob(os.path.join(run_dir, "checkpoint_ep*.pth")) or glob.glob(os.path.join(run_dir, "*.pth"))
            for run_dir in runs
        )
        if os.path.exists(sumo_cfg) and os.path.exists(cluster_file) and runs and has_checkpoint:
            maps.append(name)
    return maps


def select_map(args):
    maps = discover_maps()
    if not maps:
        print("Error: No maps with trained models found.")
        print("Train models first (python src/sample.py), then re-run this script.")
        sys.exit(1)

    if args.scenario:
        if args.scenario not in maps:
            print(f"Error: '{args.scenario}' has no trained model (or missing sumocfg/cluster file).")
            print(f"Available maps: {', '.join(maps)}")
            sys.exit(1)
        return args.scenario

    print("=" * 60)
    print("Select the map to run in the SUMO GUI:")
    for i, name in enumerate(maps, 1):
        runs = sorted(glob.glob(os.path.join(MODELS_DIR, name, "run_*")))
        latest = max(runs, key=os.path.getmtime) if runs else ""
        checkpoints = sorted(glob.glob(os.path.join(latest, "checkpoint_ep*.pth")))
        ep = os.path.basename(checkpoints[-1]) if checkpoints else "no checkpoint"
        print(f"  [{i}] {name:<14} model={os.path.basename(latest)} ({ep})")
    print("=" * 60)

    while True:
        choice = input(f"Select map (1-{len(maps)}) [default 1]: ").strip()
        if not choice:
            choice = "1"
        if choice.isdigit() and 1 <= int(choice) <= len(maps):
            return maps[int(choice) - 1]
        print(f"Invalid selection. Choose a number between 1 and {len(maps)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained model in the SUMO GUI")
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Map to run (e.g. manhattan, cologne8). Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Specific run directory under models/{map}/ to use (defaults to the latest run).",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run")
    parser.add_argument(
        "--delay",
        type=float,
        default=500,
        help="SUMO GUI delay in ms between steps (higher = slower, human-viewable). 1000ms ≈ realtime.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the phase decision for each intersection every step.",
    )
    args = parser.parse_args()

    scenario = select_map(args)
    os.environ["TRAFFIC_SCENARIO"] = scenario

    # Import after the scenario is set so the config singleton picks the right cluster file.
    from config import SCENARIO
    from evaluate import evaluate_models, find_latest_model_dir

    target_dir = args.model_dir
    if target_dir and not os.path.isabs(target_dir):
        target_dir = os.path.join(MODELS_DIR, SCENARIO, target_dir)
    target_dir = find_latest_model_dir(target_dir)

    if not target_dir or not os.path.exists(target_dir):
        print("Error: Could not find any saved models and no valid --model-dir was provided.")
        print("Train models first (python src/sample.py), then re-run this script.")
        sys.exit(1)

    config_path = os.path.join(SCENARIOS_DIR, SCENARIO, f"{SCENARIO}.sumocfg")
    print(f"Running trained model {target_dir} in SUMO GUI (scenario={SCENARIO})...")
    print(f"Sumo config: {config_path} | GUI delay: {args.delay} ms")

    evaluate_models(
        target_dir,
        steps=args.steps,
        use_gui=True,
        delay=args.delay,
        config_path=config_path,
        verbose=args.verbose,
    )
