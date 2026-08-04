import argparse
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

sys.path.insert(0, "src")
from config import SCENARIO
from evaluate import evaluate_models, find_latest_model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained model in the SUMO GUI")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing the saved model parts e.g. ../models/run_XX",
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--delay", type=float, default=1, help="GUI simulation delay in ms")
    args = parser.parse_args()

    target_dir = find_latest_model_dir(args.model_dir)
    if not target_dir or not os.path.exists(target_dir):
        print("Error: Could not find any saved models in ../models/ and no valid --model-dir was provided.")
        print("Train models first (python src/sample.py), then re-run this script.")
        sys.exit(1)

    print(f"Running trained model {target_dir} in SUMO GUI...")
    evaluate_models(
        target_dir,
        steps=args.steps,
        use_gui=True,
        delay=args.delay,
        config_path=f"sumo/{SCENARIO}.sumocfg",
    )
