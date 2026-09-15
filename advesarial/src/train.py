"""Thin wrapper kept for compatibility with anything calling `train.train(...)`
directly; `sample.py` is the canonical CLI entrypoint used by pipeline.py."""

from config import SCENARIO
from training import Trainer


def train(episodes: int = 10, episode_steps: int = 1000, save_every: int = 10):
    trainer = Trainer(scenario=SCENARIO, sumocfg_path=f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg")
    run_dir = trainer.create_run_dir()
    try:
        trainer.train(episodes, episode_steps, save_every, run_dir)
    finally:
        trainer.close()


if __name__ == "__main__":
    train()
