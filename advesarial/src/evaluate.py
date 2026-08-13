import os
from datetime import datetime

import torch
import traci
from agents import ActorCritic
from config import config
from environment import Environment
from schema import TraciConfig, TransformerEncoderConfig
from services import TraciService
from utils.tripinfo_metrics import parse_tripinfo, print_tripinfo_summary

from models import GATLayer, LocalEncoder, SubGoalGenerator, TransformerEncoder


def evaluate_models(
    model_dir,
    steps=360,
    scenario_cfg="../scenarios/cologne8/cologne8.sumocfg",
    min_green=1,
    drain_budget=3600,
):
    print("Initializing environment...")
    traci_config = TraciConfig(
        config_path=scenario_cfg,
        tripinfo_output="visualizations/tripinfo_eval.xml",
        min_green_time=min_green,
    )

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
    actor_critic = ActorCritic(state_dimension=144, action_dimension=7)

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

    running_mean = torch.zeros(10)
    running_var = torch.ones(10)
    norm_alpha = 0.01

    def __normalize_states(states):
        nonlocal running_mean, running_var

        batch_mean = states.mean(dim=0)
        batch_var = states.var(dim=0, unbiased=False)

        running_mean = (1 - norm_alpha) * running_mean + norm_alpha * batch_mean
        running_var = (1 - norm_alpha) * running_var + norm_alpha * batch_var

        states = (states - running_mean) / (torch.sqrt(running_var) + 1e-8)

        return torch.clamp(states, -5, 5)

    def _find_global_observation():
        cluster_states = traci_service.get_cluster_states(clusters)
        global_encoding, local_encoding = transformer_encoder(
            cluster_states.unsqueeze(0)
        )
        subgoal_vector = subgoal_generator(local_encoding, global_encoding)
        return subgoal_vector

    def _find_local_observations(sub_goal_vector):
        observations = traci_service.get_observations()
        observations = __normalize_states(observations)
        hidden_state = local_encoder(observations)
        local_features = GAT(hidden_state, adjacency_list)
        mean_feat = torch.mean(local_features, dim=0)
        final_state = []
        for z_i in local_features:
            final_state.append(
                torch.cat([z_i, mean_feat, sub_goal_vector[0]], dim=-1)
            )
        return torch.stack(final_state, dim=0)

    total_queue_length = []
    total_vehicle_count = []
    vehicle_metrics = {}
    vehicle_travel_times = []
    vehicle_delay_times = []
    vehicle_waiting_times = []

    def get_free_flow_travel_time(route):
        """Calculate the route time at each edge's maximum lane speed."""
        free_flow_time = 0.0
        for edge in route:
            lane_id = f"{edge}_0"
            try:
                length = traci.lane.getLength(lane_id)
                max_speed = traci.lane.getMaxSpeed(lane_id)
            except traci.exceptions.TraCIException:
                continue
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

    sim_begin = traci.simulation.getTime()

    def total_halted_queue(i):
        return sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in traci.trafficlight.getControlledLanes(i)
        )

    print(f"Starting evaluation for {steps} steps (sim time {sim_begin:.0f}s)...")

    with torch.no_grad():
        intersections = traci_service.get_all_intersections()
        for t in range(steps):
            sub_goal_vector = _find_global_observation()
            final_state = _find_local_observations(sub_goal_vector)

            action_prob, state_values = actor_critic(final_state)
            action = torch.argmax(action_prob, dim=-1).unsqueeze(-1)

            # Store vehicle data before stepping because arrived vehicles can no
            # longer be queried from TraCI after the simulation advances.
            update_active_vehicle_metrics()
            _, reward, done = environment.step(action, needs_obs=False)

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
            raw_queue = sum(total_halted_queue(i) for i in intersections)
            raw_vehicles = sum(traci_service.total_queue_length(i) for i in intersections)

            total_queue_length.append(raw_queue)
            total_vehicle_count.append(raw_vehicles)

            if (t + 1) % 30 == 0:
                print(
                    f"Eval Step {t + 1}/{steps} (sim {current_time:.0f}s) - "
                    f"Queue (halted): {raw_queue:.2f}, Vehicles on TL lanes: {raw_vehicles:.2f}, "
                    f"Completed Vehicles: {len(vehicle_travel_times)}"
                )

            if done:
                print(f"Environment finished early at step {t}")
                break

    # Drain phase: keep stepping (no new control decisions) until every vehicle
    # has arrived, so tripinfo covers all trips (paper protocol). Capped by a
    # budget in case a vehicle deadlocks.
    control_interval = 10
    drain_steps = 0
    remaining = traci.simulation.getMinExpectedNumber()
    while remaining > 0 and drain_steps * control_interval < drain_budget:
        drain_steps += 1
        update_active_vehicle_metrics()
        traci_service.step(control_interval)
        current_time = traci.simulation.getTime()
        for vehicle_id in traci.simulation.getArrivedIDList():
            metrics = vehicle_metrics.pop(vehicle_id, None)
            if metrics is None:
                continue
            travel_time = current_time - metrics["entry_time"]
            vehicle_travel_times.append(travel_time)
            vehicle_delay_times.append(travel_time - metrics["free_flow_time"])
            vehicle_waiting_times.append(metrics["waiting_time"])
        remaining = traci.simulation.getMinExpectedNumber()
        if drain_steps % 60 == 0:
            print(f"Drain: sim {current_time:.0f}s - {remaining} vehicles remaining")

    if remaining > 0:
        print(f"WARNING: drain budget ({drain_budget}s) exhausted, {remaining} vehicles still running")

    # Summarize results
    avg_queue = sum(total_queue_length) / len(total_queue_length)
    peak_queue = max(total_queue_length)
    avg_vehicles = (
        sum(total_vehicle_count) / len(total_vehicle_count)
        if total_vehicle_count
        else 0.0
    )
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
    print(f"Simulation Window         : {sim_begin:.0f}s - {current_time:.0f}s")
    print(f"Total Steps Evaluated      : {len(total_queue_length)}")
    print(f"Completed Vehicles         : {completed_vehicles}")
    print(f"Average Queue Length       : {avg_queue:.2f} vehicles (halted)")
    print(f"Avg Vehicles on TL lanes   : {avg_vehicles:.2f} vehicles")
    print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
    print(f"Average Travel Time (ATT)  : {avg_travel_time:.2f} seconds")
    print(f"Average Delay Time (ADT)   : {avg_delay_time:.2f} seconds")
    print(f"Peak Total Queue Length    : {peak_queue:.2f} vehicles")
    print("=" * 50)

    # Save Evaluation Run Stats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("../metrics.txt", "a") as f:
        f.write(f"--- Evaluation Snapshot: {timestamp} ---\n")
        f.write(f"Evaluating Model: {model_dir}\n")
        f.write(f"Simulation Window: {sim_begin:.0f}s - {current_time:.0f}s\n")
        f.write(f"Total Steps Evaluated: {len(total_queue_length)}\n")
        f.write(f"Completed Vehicles: {completed_vehicles}\n")
        f.write(f"Average Queue Length: {avg_queue:.2f} vehicles (halted)\n")
        f.write(f"Average Waiting Time: {avg_wait:.2f} seconds\n")
        f.write(f"Average Travel Time (ATT): {avg_travel_time:.2f} seconds\n")
        f.write(f"Average Delay Time (ADT): {avg_delay_time:.2f} seconds\n")
        f.write(f"Peak Total Queue Length: {peak_queue:.2f} vehicles\n")
        f.write("\n")

    traci_service.close_simulation()

    # Paper-style metrics from tripinfo output
    tripinfo_path = "visualizations/tripinfo_eval.xml"
    if os.path.exists(tripinfo_path):
        trip_metrics = parse_tripinfo(tripinfo_path)
        print_tripinfo_summary(trip_metrics)
        f = open("../metrics.txt", "a")
        f.write(f"--- Tripinfo Metrics: {datetime.now().strftime('%Y%m%d_%H%M%S')} ---\n")
        f.write(f"Evaluating Model: {model_dir}\n")
        f.write(f"ATT: {trip_metrics['att']:.2f} seconds\n")
        f.write(f"ADT: {trip_metrics['adt']:.2f} seconds\n")
        f.write(f"Completed: {trip_metrics['vehicles_ended']}\n")
        f.write(f"Teleports: {trip_metrics['teleports']}\n")
        f.write("\n")
        f.close()
    else:
        print(f"WARNING: tripinfo output not found at {tripinfo_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Traffic Control Model")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing the saved model parts e.g. ../models/run_XX",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=360,
        help=(
            "Number of control steps to evaluate (default 360 = full 3600s "
            "episode at a 10s control interval, matching HiLight protocol)"
        ),
    )
    parser.add_argument(
        "--min-green",
        type=int,
        default=1,
        help="Minimum green duration in control calls (1 call = 10s sim)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="../scenarios/cologne8/cologne8.sumocfg",
        help="SUMO config for the scenario to evaluate on",
    )
    parser.add_argument(
        "--drain-budget",
        type=int,
        default=3600,
        help="Max additional sim-seconds to drain vehicles after the control steps",
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

    evaluate_models(
        target_dir,
        args.steps,
        scenario_cfg=args.scenario,
        min_green=args.min_green,
        drain_budget=args.drain_budget,
    )
