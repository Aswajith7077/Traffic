"""Train one (config, seed) combination and dump loss/reward histories to JSON.

Run with cwd=advesarial (same convention as sample.py/evaluate.py, see AGENTS.md)
so the relative scenario/model paths resolve. Invoked by run_experiments.py.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=1200)
    parser.add_argument(
        "--bc-steps",
        type=int,
        default=int(os.environ.get("BC_STEPS", "1200")),
        help="Behavior-cloning warm-start steps (imitate Max Pressure) before PPO fine-tuning. 0 disables it.",
    )
    parser.add_argument("--out-dir", default="../paper_metrics/raw")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from schema import RewardConfig  # noqa: E402
    from training import Trainer  # noqa: E402

    reward_config = RewardConfig(
        w1=float(os.environ.get("REWARD_W1", "0.6")),
        w2=float(os.environ.get("REWARD_W2", "0.3")),
        w3=float(os.environ.get("REWARD_W3", "0.1")),
    )

    scenario = os.environ.get("TRAFFIC_SCENARIO", "cologne8")
    sumocfg_path = f"../scenarios/{scenario}/{scenario}.sumocfg"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Traffic
    emv_sumocfg_abs = os.path.join(repo_root, "scenarios", scenario, "emv", f"{scenario}_emv.sumocfg")
    # Only cologne8's own sumocfg ships EMVs; the other 4 scenarios get them
    # from this separate, additively-generated sumocfg instead of editing
    # their original, working files (see scenarios/*/emv/).
    if os.environ.get("USE_EMV_ROUTING") == "1" and os.path.exists(emv_sumocfg_abs):
        sumocfg_path = f"../scenarios/{scenario}/emv/{scenario}_emv.sumocfg"

    trainer = Trainer(
        scenario=scenario,
        sumocfg_path=sumocfg_path,
        reward_config=reward_config,
    )
    run_dir = trainer.create_run_dir()
    try:
        bc_losses = []
        if args.bc_steps > 0:
            bc_losses = trainer.pretrain_behavior_cloning(args.bc_steps)
            print(f"BC pretrain: {args.bc_steps} steps, final loss={bc_losses[-1]:.4f}", flush=True)
            trainer.reset_episode()

        for _ in range(args.episodes):
            trainer.run_episode(args.episode_steps)
        trainer.save_checkpoint(run_dir, args.episodes)
    finally:
        trainer.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "config_name": args.config_name,
        "seed": args.seed,
        "run_dir": run_dir,
        "bc_losses": bc_losses,
        "meta_losses": trainer.meta_losses,
        "sub_loss_history": trainer.sub_loss_history,
        "rewards_history": trainer.rewards_history,
        "reward_components_history": trainer.reward_components_history,
        "use_emv_routing": trainer.traci_service.config.use_emv_routing,
        "w1": reward_config.w1,
        "w2": reward_config.w2,
        "w3": reward_config.w3,
    }
    out_path = os.path.join(args.out_dir, f"{args.config_name}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"WROTE {out_path}")
    print(f"RUN_DIR {run_dir}")


if __name__ == "__main__":
    main()
