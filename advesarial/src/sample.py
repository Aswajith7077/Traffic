import argparse
import os

from config import SCENARIO
from training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train the RL agent (canonical entrypoint used by pipeline.py)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("TRAFFIC_EPISODES", "10")),
        help="Number of training episodes (default: 10; falls back to TRAFFIC_EPISODES env var)",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=int(os.environ.get("TRAFFIC_EPISODE_STEPS", "1000")),
        help="Sim-seconds per training episode (default: 1000; falls back to TRAFFIC_EPISODE_STEPS env var)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=int(os.environ.get("TRAFFIC_SAVE_EVERY", "10")),
        help="Save a checkpoint every N episodes (default: 10; falls back to TRAFFIC_SAVE_EVERY env var)",
    )
    args = parser.parse_args()

    trainer = Trainer(scenario=SCENARIO, sumocfg_path=f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg")
    run_dir = trainer.create_run_dir()

    try:
        trainer.train(args.episodes, args.episode_steps, args.save_every, run_dir)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
