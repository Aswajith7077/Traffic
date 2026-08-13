"""Diagnostic: evaluate an OLD (pre-subgoal, 128-dim) checkpoint at the new
10s control cadence so results are comparable to the PPO run."""

import torch
import traci
from agents import ActorCritic
from config import config
from environment import Environment
from schema import TraciConfig, TransformerEncoderConfig
from services import TraciService

from models import GATLayer, LocalEncoder, SubGoalGenerator, TransformerEncoder

MODEL_DIR = "../models/run_20260802_140914"

traci_config = TraciConfig(config_path="../scenarios/cologne8/cologne8.sumocfg")
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

local_encoder = LocalEncoder()
GAT = GATLayer(feature_dim=64)
actor_critic = ActorCritic(state_dimension=128, action_dimension=7)

transformer_encoder_config = TransformerEncoderConfig(
    d_model=128, nhead=8, num_layers=6
)
transformer_encoder = TransformerEncoder(transformer_encoder_config)
subgoal_generator = SubGoalGenerator(d_reg=128, d_hidden=128, M=m, d_g=2 * len(tls_set))

actor_critic.load_state_dict(
    torch.load(f"{MODEL_DIR}/actor_critic.pth", map_location="cpu", weights_only=True)
)
transformer_encoder.load_state_dict(
    torch.load(f"{MODEL_DIR}/transformer_encoder.pth", map_location="cpu", weights_only=True)
)
subgoal_generator.load_state_dict(
    torch.load(f"{MODEL_DIR}/subgoal_generator.pth", map_location="cpu", weights_only=True)
)
local_encoder.load_state_dict(
    torch.load(f"{MODEL_DIR}/local_encoder.pth", map_location="cpu", weights_only=True)
)
GAT.load_state_dict(
    torch.load(f"{MODEL_DIR}/gat.pth", map_location="cpu", weights_only=True)
)

for mod in (transformer_encoder, subgoal_generator, local_encoder, GAT, actor_critic):
    mod.eval()

running_mean = torch.zeros(10)
running_var = torch.ones(10)
norm_alpha = 0.01


def __normalize_states(states):
    global running_mean, running_var
    batch_mean = states.mean(dim=0)
    batch_var = states.var(dim=0, unbiased=False)
    running_mean = (1 - norm_alpha) * running_mean + norm_alpha * batch_mean
    running_var = (1 - norm_alpha) * running_var + norm_alpha * batch_var
    states = (states - running_mean) / (torch.sqrt(running_var) + 1e-8)
    return torch.clamp(states, -5, 5)


def _find_local_observations():
    observations = traci_service.get_observations()
    observations = __normalize_states(observations)
    hidden_state = local_encoder(observations)
    local_features = GAT(hidden_state, adjacency_list)
    final_state = []
    for z_i in local_features:
        final_state.append(torch.cat([z_i, torch.mean(local_features, dim=0)], dim=-1))
    return torch.stack(final_state, dim=0)


vehicle_metrics = {}
vehicle_travel_times = []
vehicle_delay_times = []
vehicle_waiting_times = []
total_queue_length = []


def get_free_flow_travel_time(route):
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
intersections = traci_service.get_all_intersections()


def total_halted_queue(i):
    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in traci.trafficlight.getControlledLanes(i)
    )


with torch.no_grad():
    for t in range(360):
        final_state = _find_local_observations()
        action_prob, _ = actor_critic(final_state)
        action = torch.argmax(action_prob, dim=-1).unsqueeze(-1)

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

        total_queue_length.append(sum(total_halted_queue(i) for i in intersections))
        if (t + 1) % 90 == 0:
            print(
                f"Eval Step {t + 1}/360 (sim {current_time:.0f}s) - "
                f"Queue (halted): {total_queue_length[-1]:.2f}, "
                f"Completed Vehicles: {len(vehicle_travel_times)}"
            )

        if done:
            print(f"Environment finished early at step {t}")
            break

completed = len(vehicle_travel_times)
avg_att = sum(vehicle_travel_times) / completed if completed else 0.0
avg_adt = sum(vehicle_delay_times) / completed if completed else 0.0
avg_wait = sum(vehicle_waiting_times) / completed if completed else 0.0
avg_queue = sum(total_queue_length) / len(total_queue_length)

print("=" * 50)
print(f"OLD MODEL (run_20260802_140914) @10s cadence")
print("=" * 50)
print(f"Completed Vehicles         : {completed}")
print(f"Average Travel Time (ATT)  : {avg_att:.2f} seconds")
print(f"Average Delay Time (ADT)   : {avg_adt:.2f} seconds")
print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
print(f"Average Queue Length       : {avg_queue:.2f} vehicles (halted)")
print(f"Peak Total Queue Length    : {max(total_queue_length):.2f} vehicles")
print("=" * 50)

traci_service.close_simulation()
