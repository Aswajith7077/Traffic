"""Diagnostic: verify the eval measurement at 10s control cadence against
SUMO tripinfo ground truth (fixed-time program, no TraCI interference)."""

import traci
from schema import TraciConfig
from services import TraciService

traci_config = TraciConfig(config_path="../scenarios/cologne8/cologne8.sumocfg")
traci_service = TraciService(traci_config)
traci_service.start_simulation()

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


for t in range(360):
    update_active_vehicle_metrics()
    traci_service.step(10)
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

completed = len(vehicle_travel_times)
avg_att = sum(vehicle_travel_times) / completed if completed else 0.0
avg_adt = sum(vehicle_delay_times) / completed if completed else 0.0
avg_wait = sum(vehicle_waiting_times) / completed if completed else 0.0
avg_queue = sum(total_queue_length) / len(total_queue_length)

print("=" * 50)
print("HARNESS @10s CADENCE (fixed-time program)")
print("=" * 50)
print(f"Simulation Window         : {sim_begin:.0f}s - {current_time:.0f}s")
print(f"Completed Vehicles         : {completed}")
print(f"Average Travel Time (ATT)  : {avg_att:.2f} seconds")
print(f"Average Delay Time (ADT)   : {avg_adt:.2f} seconds")
print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
print(f"Average Queue Length       : {avg_queue:.2f} vehicles (halted)")
print("=" * 50)
print("SUMO tripinfo (exact)      : ATT 114.39, ADT 49.26 (timeLoss), waiting 29.27")
print("=" * 50)

traci_service.close_simulation()
