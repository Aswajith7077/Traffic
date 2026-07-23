import numpy as np


def _build_edge_cache(traci):
    cache = {}
    for edge_id in traci.edge.getIDList():
        try:
            lane_count = traci.edge.getLaneNumber(edge_id)
            if lane_count > 0:
                lane_id = f"{edge_id}_0"
                cache[edge_id] = (
                    traci.lane.getLength(lane_id),
                    traci.lane.getSpeed(lane_id),
                )
            else:
                cache[edge_id] = (5.0, 50.0 / 3.6)
        except Exception:
            cache[edge_id] = (5.0, 50.0 / 3.6)
    return cache


def _compute_route_ideal_time(route_edges, edge_cache):
    total = 0.0
    for edge_id in route_edges:
        length, speed = edge_cache.get(edge_id, (5.0, 50.0 / 3.6))
        if speed > 0:
            total += length / speed
    return total


def evaluate_episode(traci, route_length_lookup=None, max_steps=100000):
    """
    Evaluate one simulation episode and compute benchmark metrics.

    Tracks every vehicle from entry to exit. Computes:
        - Average Trip Time (ATT)
        - Average Delay Time (ADT)

    Args:
        traci: An active TraCI connection (the traci module after
            traci.start() has been called).
        route_length_lookup: Optional precomputed dict {route_id: ideal_time}.
            If None, ideal times are computed from edge lengths and speed limits.
        max_steps: Maximum number of simulation steps before stopping.

    Returns:
        dict with keys:
            num_completed_vehicles (int)
            average_trip_time (float)
            average_delay_time (float)
            trip_times (list[float])
            delay_times (list[float])
            median_trip_time (float)
            median_delay_time (float)
            std_trip_time (float)
            std_delay_time (float)
            p95_trip_time (float)
            p95_delay_time (float)
    """
    edge_cache = _build_edge_cache(traci)

    active_vehicles = {}

    trip_times = []
    delay_times = []

    for _ in range(max_steps):
        traci.simulationStep()

        for veh_id in traci.simulation.getArrivedIDList():
            record = active_vehicles.pop(veh_id, None)
            if record is None:
                continue

            entry_time = record["entry_time"]
            route_id = record["route_id"]
            route_edges = record["route_edges"]
            exit_time = traci.simulation.getTime()

            actual_time = exit_time - entry_time

            if route_length_lookup is not None and route_id in route_length_lookup:
                ideal_time = route_length_lookup[route_id]
            else:
                ideal_time = _compute_route_ideal_time(route_edges, edge_cache)

            delay = actual_time - ideal_time

            trip_times.append(actual_time)
            delay_times.append(delay)

        for veh_id in traci.simulation.getDepartedIDList():
            if veh_id not in active_vehicles:
                route_edges = traci.vehicle.getRoute(veh_id)
                route_id = traci.vehicle.getRouteID(veh_id)
                active_vehicles[veh_id] = {
                    "entry_time": traci.simulation.getTime(),
                    "route_id": route_id,
                    "route_edges": route_edges,
                }

        if not active_vehicles and traci.simulation.getMinExpectedNumber() <= 0:
            break

    num_completed = len(trip_times)

    if num_completed == 0:
        return {
            "num_completed_vehicles": 0,
            "average_trip_time": 0.0,
            "average_delay_time": 0.0,
            "trip_times": [],
            "delay_times": [],
            "median_trip_time": 0.0,
            "median_delay_time": 0.0,
            "std_trip_time": 0.0,
            "std_delay_time": 0.0,
            "p95_trip_time": 0.0,
            "p95_delay_time": 0.0,
        }

    return {
        "num_completed_vehicles": num_completed,
        "average_trip_time": float(np.mean(trip_times)),
        "average_delay_time": float(np.mean(delay_times)),
        "trip_times": trip_times,
        "delay_times": delay_times,
        "median_trip_time": float(np.median(trip_times)),
        "median_delay_time": float(np.median(delay_times)),
        "std_trip_time": float(np.std(trip_times)),
        "std_delay_time": float(np.std(delay_times)),
        "p95_trip_time": float(np.percentile(trip_times, 95)),
        "p95_delay_time": float(np.percentile(delay_times, 95)),
    }
