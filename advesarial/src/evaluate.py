import os
import torch
from datetime import datetime

from models import GATLayer
from models import LocalEncoder
from models import TransformerEncoder
from models import SubGoalGenerator
from agents import ActorCritic
from services import TraciService
from schema import TraciConfig
from schema import TransformerEncoderConfig
from config import config
from environment import Environment

import traci


def evaluate_models(model_dir, steps=500):
    print("Initializing environment...")
    traci_config = TraciConfig(config_path="sumo/osm.sumocfg")

    traci_service = TraciService(traci_config)
    traci_service.start_simulation()

    environment = Environment(traci_service=traci_service)
    adjacency_list = traci_service.get_adjacency_list()

    raw_clusters = config.clusters
    tls_set = set(traci_service.get_all_intersections())
    clusters = {}
    for cid, nodes in raw_clusters.items():
        filtered = [n for n in nodes if n in tls_set]
        if len(filtered) > 0:
            clusters[cid] = filtered

    m = len(clusters)

    # Initialize models
    print("Initializing architecture...")
    local_encoder = LocalEncoder()
    GAT = GATLayer(feature_dim=64)
    actor_critic = ActorCritic(state_dimension=128, action_dimension=7)

    transformer_encoder_config = TransformerEncoderConfig(
        d_model=128, nhead=8, num_layers=6
    )
    transformer_encoder = TransformerEncoder(transformer_encoder_config)
    subgoal_generator = SubGoalGenerator(
        d_reg=128, d_hidden=128, M=m, d_g=2 * len(tls_set)
    )

    # Load models
    print(f"Loading weights from {model_dir}...")
    transformer_encoder.load_state_dict(
        torch.load(
            f"{model_dir}/transformer_encoder.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    subgoal_generator.load_state_dict(
        torch.load(
            f"{model_dir}/subgoal_generator.pth", map_location="cpu", weights_only=True
        )
    )
    local_encoder.load_state_dict(
        torch.load(
            f"{model_dir}/local_encoder.pth", map_location="cpu", weights_only=True
        )
    )
    GAT.load_state_dict(
        torch.load(f"{model_dir}/gat.pth", map_location="cpu", weights_only=True)
    )
    actor_critic.load_state_dict(
        torch.load(
            f"{model_dir}/actor_critic.pth", map_location="cpu", weights_only=True
        )
    )

    transformer_encoder.eval()
    subgoal_generator.eval()
    local_encoder.eval()
    GAT.eval()
    actor_critic.eval()

    def __normalize_states(states):
        batch_mean = states.mean(dim=0)
        batch_var = states.var(dim=0, unbiased=False)
        return torch.clamp(
            (states - batch_mean) / (torch.sqrt(batch_var) + 1e-8), -5, 5
        )

    def _find_local_observations():
        observations = traci_service.get_observations()
        observations = __normalize_states(observations)
        hidden_state = local_encoder(observations)
        local_features = GAT(hidden_state, adjacency_list)
        final_state = []
        for i in range(len(local_features)):
            z_i = local_features[i]
            final_state.append(
                torch.cat([z_i, torch.mean(local_features, dim=0)], dim=-1)
            )
        return torch.stack(final_state, dim=0)

    def _find_global_observation():
        cluster_states = traci_service.get_cluster_states(clusters)
        global_encoding, local_encoding = transformer_encoder(
            cluster_states.unsqueeze(0)
        )
        return subgoal_generator(local_encoding, global_encoding)

    total_queue_length = []
    vehicle_metrics = {}
    vehicle_travel_times = []
    vehicle_delay_times = []
    vehicle_waiting_times = []

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

    def update_active_vehicle_metrics():
        """Cache data while vehicles remain queryable through TraCI."""
        current_time = traci.simulation.getTime()
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in vehicle_metrics:
                vehicle_metrics[vehicle_id] = {
                    "entry_time": current_time,
                    "free_flow_time": get_free_flow_travel_time(
                        traci.vehicle.getRoute(vehicle_id)
                    ),
                    "waiting_time": 0.0,
                }
            vehicle_metrics[vehicle_id]["waiting_time"] = (
                traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            )

    print(f"Starting evaluation for {steps} steps...")

    with torch.no_grad():
        for t in range(steps):
            # Unused explicitly in greedy greedy execution loop but compute global token to keep RNN states moving if any
            _ = _find_global_observation()
            final_state = _find_local_observations()

            action_prob, state_values = actor_critic(final_state)
            action = torch.argmax(action_prob, dim=-1).unsqueeze(-1)

            # Store vehicle data before stepping because arrived vehicles can no
            # longer be queried from TraCI after the simulation advances.
            update_active_vehicle_metrics()
            _, reward, done = environment.step(action)

            current_time = traci.simulation.getTime()
            for vehicle_id in traci.simulation.getArrivedIDList():
                metrics = vehicle_metrics.pop(vehicle_id, None)
                if metrics is None:
                    continue
                travel_time = current_time - metrics["entry_time"]
                vehicle_travel_times.append(travel_time)
                vehicle_delay_times.append(travel_time - metrics["free_flow_time"])
                vehicle_waiting_times.append(metrics["waiting_time"])

            # Fetch unnormalized raw values across the intersections
            intersections = traci_service.get_all_intersections()
            raw_queue = sum(traci_service.total_queue_length(i) for i in intersections)

            total_queue_length.append(raw_queue)

            if (t + 1) % 50 == 0:
                print(
                    f"Eval Step {t + 1}/{steps} - Current Queue Total: {raw_queue:.2f}, "
                    f"Completed Vehicles: {len(vehicle_travel_times)}"
                )

            if done:
                print(f"Environment finished early at step {t}")
                break

    # Summarize results
    avg_queue = sum(total_queue_length) / len(total_queue_length)
    peak_queue = max(total_queue_length)
    completed_vehicles = len(vehicle_travel_times)
    avg_wait = (
        sum(vehicle_waiting_times) / completed_vehicles if completed_vehicles else 0.0
    )
    avg_travel_time = (
        sum(vehicle_travel_times) / completed_vehicles if completed_vehicles else 0.0
    )
    avg_delay_time = (
        sum(vehicle_delay_times) / completed_vehicles if completed_vehicles else 0.0
    )

    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)
    print(f"Total Steps Evaluated      : {len(total_queue_length)}")
    print(f"Completed Vehicles         : {completed_vehicles}")
    print(f"Average Queue Length       : {avg_queue:.2f} vehicles")
    print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
    print(f"Average Travel Time        : {avg_travel_time:.2f} seconds")
    print(f"Average Delay Time         : {avg_delay_time:.2f} seconds")
    print(f"Peak Total Queue Length    : {peak_queue:.2f} vehicles")
    print("=" * 50)

    # Save Evaluation Run Stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("../metrics.txt", "a") as f:
        f.write(f"--- Evaluation Snapshot: {timestamp} ---\n")
        f.write(f"Evaluating Model: {model_dir}\n")
        f.write(f"Total Steps Evaluated: {len(total_queue_length)}\n")
        f.write(f"Completed Vehicles: {completed_vehicles}\n")
        f.write(f"Average Queue Length: {avg_queue:.2f} vehicles\n")
        f.write(f"Average Waiting Time: {avg_wait:.2f} seconds\n")
        f.write(f"Average Travel Time: {avg_travel_time:.2f} seconds\n")
        f.write(f"Average Delay Time: {avg_delay_time:.2f} seconds\n")
        f.write(f"Peak Total Queue Length: {peak_queue:.2f} vehicles\n")
        f.write("\n")

    traci_service.close_simulation()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Traffic Control Model")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing the saved model parts e.g. ../models/run_XX",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Number of steps to evaluate"
    )
    args = parser.parse_args()

    target_dir = args.model_dir
    if not target_dir:
        models_base = "../models"
        if os.path.exists(models_base):
            runs = [
                os.path.join(models_base, d)
                for d in os.listdir(models_base)
                if d.startswith("run_")
            ]
            if runs:
                target_dir = max(runs, key=os.path.getmtime)
                print(f"Auto-selected latest model directory: {target_dir}")

    if not target_dir or not os.path.exists(target_dir):
        print(
            "Error: Could not find any saved models in ../models/ and no valid --model-dir was provided."
        )
        print("Make sure you have trained and saved models before evaluating.")
        exit(1)

    evaluate_models(target_dir, args.steps)
