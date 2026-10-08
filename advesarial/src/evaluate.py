import glob
import os
from datetime import datetime

import torch
import traci
from config import SCENARIO
from training import Trainer


def find_latest_model_dir(model_dir=None):
    """Resolve the model directory from --model-dir, or auto-select the latest run."""
    target_dir = model_dir
    if not target_dir:
        models_base = "../models"
        if os.path.exists(models_base):
            runs = sorted(glob.glob(os.path.join(models_base, SCENARIO, "run_*")))
            if not runs:
                runs = sorted(glob.glob(os.path.join(models_base, "run_*")))
            if runs:
                target_dir = max(runs, key=os.path.getmtime)
                print(f"Auto-selected latest model directory: {target_dir}")
    return target_dir


def get_free_flow_travel_time(route):
    """Calculate the route time at each edge's maximum lane speed."""
    free_flow_time = 0.0
    for edge in route:
        lane_id = f"{edge}_0"
        length = traci.lane.getLength(lane_id)
        max_speed = traci.lane.getMaxSpeed(lane_id)
        if max_speed > 0:
            free_flow_time += length / max_speed
    return free_flow_time


def run_gui_simulation(model_dir, steps=3600, delay=50.0, config_path=None, verbose=False):
    """Watch a trained model drive the scenario in sumo-gui. No metrics, no logging to disk."""
    print("Initializing GUI environment...")
    if config_path is None:
        config_path = f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg"

    trainer = Trainer(scenario=SCENARIO, sumocfg_path=config_path, use_gui=True, delay=delay)

    checkpoint_path = trainer.latest_checkpoint(model_dir)
    if not checkpoint_path:
        print(f"Error: no checkpoint found in {model_dir}")
        trainer.close()
        return

    trainer.load_checkpoint(model_dir)
    trainer.transformer_encoder.eval()
    trainer.subgoal_generator.eval()
    trainer.local_encoder.eval()
    trainer.gat.eval()
    trainer.actor_critic.eval()

    print(f"Launching sumo-gui for scenario '{SCENARIO}' with model: {model_dir}")
    try:
        with torch.no_grad():
            for t in range(steps):
                f_g, goal = trainer._meta_forward()

                obs = trainer.traci_service.get_observations()
                obs = trainer._normalize_obs(obs)
                final_state = trainer._encode_step(obs, f_g)

                action_prob, _, _ = trainer.actor_critic(final_state, action_mask=trainer.action_mask)
                action = torch.argmax(action_prob, dim=-1)

                if verbose:
                    phases = {
                        tl: int(p)
                        for tl, p in zip(trainer.traci_service.get_all_intersections(), action.tolist())
                    }
                    print(f"[t={traci.simulation.getTime():.0f}] chosen phases: {phases}")

                goal_pair = (goal[0, 0], goal[0, 1])
                _, _, done = trainer.environment.step(action, goal=goal_pair)

                if done:
                    print(f"Simulation finished at step {t}")
                    break
    except (KeyboardInterrupt, traci.exceptions.FatalTraCIError):
        print("\nGUI simulation closed.")
    finally:
        trainer.close()


def evaluate_models(model_dir, steps=500, use_gui=False, delay=0.0, config_path=None, verbose=False):
    print("Initializing environment...")
    if config_path is None:
        config_path = f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg"

    trainer = Trainer(scenario=SCENARIO, sumocfg_path=config_path, use_gui=use_gui, delay=delay)

    checkpoint_path = trainer.latest_checkpoint(model_dir)
    if not checkpoint_path:
        print(f"Error: no checkpoint found in {model_dir}")
        trainer.close()
        return

    checkpoint_M = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("M", trainer.M)
    print(f"Architecture: detected M={trainer.M} clusters; checkpoint trained with M={checkpoint_M}.")
    if checkpoint_M != trainer.M:
        print(
            f"WARNING: checkpoint was trained with M={checkpoint_M} clusters but the current run "
            f"detected M={trainer.M}. Re-run region-splitting and copy the cluster file, or the "
            "meta-policy forward pass will not match training."
        )

    trainer.load_checkpoint(model_dir)
    trainer.transformer_encoder.eval()
    trainer.subgoal_generator.eval()
    trainer.local_encoder.eval()
    trainer.gat.eval()
    trainer.actor_critic.eval()

    vehicle_metrics = {}
    vehicle_travel_times = []
    vehicle_delay_times = []
    vehicle_waiting_times = []
    total_queue_length = []

    def update_active_vehicle_metrics():
        """Cache data while vehicles remain queryable through TraCI."""
        current_time = traci.simulation.getTime()
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in vehicle_metrics:
                vehicle_metrics[vehicle_id] = {
                    "entry_time": current_time,
                    "free_flow_time": get_free_flow_travel_time(traci.vehicle.getRoute(vehicle_id)),
                    "waiting_time": 0.0,
                }
            vehicle_metrics[vehicle_id]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

    print(f"Starting evaluation for {steps} steps...")

    try:
        with torch.no_grad():
            for t in range(steps):
                f_g, goal = trainer._meta_forward()

                obs = trainer.traci_service.get_observations()
                obs = trainer._normalize_obs(obs)
                final_state = trainer._encode_step(obs, f_g)

                action_prob, _, _ = trainer.actor_critic(final_state, action_mask=trainer.action_mask)
                action = torch.argmax(action_prob, dim=-1)

                if verbose:
                    phases = {
                        tl: int(p)
                        for tl, p in zip(trainer.traci_service.get_all_intersections(), action.tolist())
                    }
                    print(f"[t={traci.simulation.getTime():.0f}] chosen phases: {phases}")

                update_active_vehicle_metrics()
                goal_pair = (goal[0, 0], goal[0, 1])
                _, _, done = trainer.environment.step(action, goal=goal_pair)

                current_time = traci.simulation.getTime()
                for vehicle_id in traci.simulation.getArrivedIDList():
                    metrics = vehicle_metrics.pop(vehicle_id, None)
                    if metrics is None:
                        continue
                    travel_time = current_time - metrics["entry_time"]
                    vehicle_travel_times.append(travel_time)
                    vehicle_delay_times.append(travel_time - metrics["free_flow_time"])
                    vehicle_waiting_times.append(metrics["waiting_time"])

                raw_queue = sum(
                    trainer.traci_service.total_queue_length(i) for i in trainer.traci_service.get_all_intersections()
                )
                total_queue_length.append(raw_queue)

                if (t + 1) % 50 == 0:
                    print(
                        f"Eval Step {t + 1}/{steps} - Current Queue Total: {raw_queue:.2f}, "
                        f"Completed Vehicles: {len(vehicle_travel_times)}"
                    )

                if done:
                    print(f"Environment finished early at step {t}")
                    break
    except (KeyboardInterrupt, traci.exceptions.FatalTraCIError):
        print("\nSimulation interrupted (GUI closed). Exiting.")
        trainer.close()
        return

    avg_queue = sum(total_queue_length) / len(total_queue_length)
    peak_queue = max(total_queue_length)
    completed_vehicles = len(vehicle_travel_times)
    avg_wait = sum(vehicle_waiting_times) / completed_vehicles if completed_vehicles else 0.0
    avg_travel_time = sum(vehicle_travel_times) / completed_vehicles if completed_vehicles else 0.0
    avg_delay_time = sum(vehicle_delay_times) / completed_vehicles if completed_vehicles else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Steps Evaluated      : {len(total_queue_length)}")
    print(f"Completed Vehicles         : {completed_vehicles}")
    print(f"Average Queue Length       : {avg_queue:.2f} vehicles")
    print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
    print(f"Average Travel Time (ATT)  : {avg_travel_time:.2f} seconds")
    print(f"Average Delay Time (ADT)   : {avg_delay_time:.2f} seconds")
    print(f"Peak Total Queue Length    : {peak_queue:.2f} vehicles")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("../metrics.txt", "a") as f:
        f.write(f"--- Evaluation Snapshot: {timestamp} ---\n")
        f.write(f"Scenario: {SCENARIO}\n")
        f.write(f"Cluster Method: {os.environ.get('CLUSTER_METHOD', 'dbscan')}\n")
        f.write(f"Evaluating Model: {model_dir}\n")
        f.write(f"Total Steps Evaluated: {len(total_queue_length)}\n")
        f.write(f"Completed Vehicles: {completed_vehicles}\n")
        f.write(f"Average Queue Length: {avg_queue:.2f} vehicles\n")
        f.write(f"Average Waiting Time: {avg_wait:.2f} seconds\n")
        f.write(f"Average Travel Time (ATT): {avg_travel_time:.2f} seconds\n")
        f.write(f"Average Delay Time (ADT): {avg_delay_time:.2f} seconds\n")
        f.write(f"Peak Total Queue Length: {peak_queue:.2f} vehicles\n")
        f.write("\n")

    trainer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Traffic Control Model")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing the saved model parts e.g. ../models/run_XX",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to evaluate (default: 500)")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch sumo-gui and watch the model drive the scenario. Skips metrics computation entirely.",
    )
    parser.add_argument("--delay", type=float, default=50.0, help="sumo-gui playback delay in ms per step (--gui only)")
    args = parser.parse_args()

    target_dir = find_latest_model_dir(args.model_dir)

    if not target_dir or not os.path.exists(target_dir):
        print("Error: Could not find any saved models in ../models/ and no valid --model-dir was provided.")
        print("Make sure you have trained and saved models before evaluating.")
        exit(1)

    if args.gui:
        run_gui_simulation(target_dir, steps=args.steps or 3600, delay=args.delay)
    else:
        evaluate_models(target_dir, args.steps or 500)
