import csv
import glob
import os
from datetime import datetime

import torch
import traci
from baselines import FixedTimeController, MaxPressureController
from config import SCENARIO
from training import Trainer

CSV_FIELDS = [
    "timestamp",
    "label",
    "controller",
    "scenario",
    "model_dir",
    "use_emv_routing",
    "use_accidents",
    "steps",
    "completed_vehicles",
    "avg_queue",
    "peak_queue",
    "avg_waiting_time",
    "avg_travel_time",
    "avg_delay_time",
    "max_wait",
    "p95_wait",
    "frac_over_wmax",
    "jain_network",
    "jain_mean_intersection",
    "jain_intersection_incident",
    "jain_intersection_normal",
    "throughput_veh_per_step",
    "emv_count",
    "emv_avg_response_time",
    "emv_p95_response_time",
    "starvation_overrides",
]


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

                obs = trainer._normalize_obs(trainer._get_observations())
                final_state = trainer._encode_step(obs, f_g)

                action_prob, _, _ = trainer.actor_critic(final_state, action_mask=trainer.action_mask)
                action = torch.argmax(action_prob, dim=-1)

                if verbose:
                    phases = {
                        tl: int(p) for tl, p in zip(trainer.traci_service.get_all_intersections(), action.tolist())
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


def _write_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Metrics row appended to {csv_path}")


def evaluate(
    controller="algorithm",
    model_dir=None,
    steps=500,
    use_gui=False,
    delay=0.0,
    config_path=None,
    verbose=False,
    csv_path="../paper_metrics/eval_results.csv",
    label=None,
):
    """Run `steps` simulation steps under one of three controllers (Section 5.2):
    the trained hierarchical policy ("algorithm"), or the non-learning
    "fixed"/"max_pressure" baselines. Reports the full metric set of Sections
    4.4/5.4 (mean/max/P95 wait, queue, throughput, ATT/ADT, network + per-
    intersection JFI split incident-vs-incident-free, and EMV response time)
    and appends one row to `csv_path`.
    """
    print("Initializing environment...")
    if config_path is None:
        config_path = f"../scenarios/{SCENARIO}/{SCENARIO}.sumocfg"

    trainer = Trainer(scenario=SCENARIO, sumocfg_path=config_path, use_gui=use_gui, delay=delay)

    if controller == "algorithm":
        checkpoint_path = trainer.latest_checkpoint(model_dir)
        if not checkpoint_path:
            print(f"Error: no checkpoint found in {model_dir}")
            trainer.close()
            return None

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

        def choose_and_apply():
            f_g, goal = trainer._meta_forward()
            obs = trainer._normalize_obs(trainer._get_observations())
            final_state = trainer._encode_step(obs, f_g)
            with torch.no_grad():
                action_prob, _, _ = trainer.actor_critic(final_state, action_mask=trainer.action_mask)
            action = torch.argmax(action_prob, dim=-1)
            if verbose:
                phases = {
                    tl: int(p) for tl, p in zip(trainer.traci_service.get_all_intersections(), action.tolist())
                }
                print(f"[t={traci.simulation.getTime():.0f}] chosen phases: {phases}")
            trainer.environment._apply_action(action)

    elif controller == "fixed":
        choose_and_apply = FixedTimeController(trainer.traci_service).step
    elif controller == "max_pressure":
        choose_and_apply = MaxPressureController(trainer.traci_service).step
    else:
        trainer.close()
        raise ValueError(f"Unknown controller: {controller!r} (expected algorithm, fixed, or max_pressure)")

    vehicle_metrics = {}
    vehicle_travel_times = []
    vehicle_delay_times = []
    vehicle_waiting_times = []
    emv_response_times = []
    total_queue_length = []
    intersection_jain = []
    jain_incident = []
    jain_normal = []
    max_timer_seen = 0.0

    def update_active_vehicle_metrics():
        """Cache data while vehicles remain queryable through TraCI."""
        current_time = traci.simulation.getTime()
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in vehicle_metrics:
                vehicle_metrics[vehicle_id] = {
                    "entry_time": current_time,
                    "free_flow_time": get_free_flow_travel_time(traci.vehicle.getRoute(vehicle_id)),
                    "waiting_time": 0.0,
                    "is_emv": traci.vehicle.getTypeID(vehicle_id) == "emergency",
                }
            vehicle_metrics[vehicle_id]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)

    print(f"Starting evaluation for {steps} steps (controller={controller})...")

    try:
        for t in range(steps):
            choose_and_apply()
            trainer.traci_service.step()
            trainer.traci_service.update_wait_timers()

            update_active_vehicle_metrics()

            current_time = traci.simulation.getTime()
            for vehicle_id in traci.simulation.getArrivedIDList():
                metrics = vehicle_metrics.pop(vehicle_id, None)
                if metrics is None:
                    continue
                travel_time = current_time - metrics["entry_time"]
                vehicle_travel_times.append(travel_time)
                vehicle_delay_times.append(travel_time - metrics["free_flow_time"])
                vehicle_waiting_times.append(metrics["waiting_time"])
                if metrics["is_emv"]:
                    emv_response_times.append(travel_time)

            raw_queue = sum(
                trainer.traci_service.total_queue_length(i) for i in trainer.traci_service.get_all_intersections()
            )
            total_queue_length.append(raw_queue)

            in_incident = trainer.traci_service.in_incident
            for tls in trainer.traci_service.get_all_intersections():
                waits = trainer.traci_service.get_vehicle_waits(tls)
                if waits:
                    jain = trainer.traci_service.get_intersection_fairness(tls)[0]
                    intersection_jain.append(jain)
                    (jain_incident if in_incident else jain_normal).append(jain)
                    max_timer_seen = max(max_timer_seen, max(waits))

            if (t + 1) % 50 == 0:
                print(
                    f"Eval Step {t + 1}/{steps} - Current Queue Total: {raw_queue:.2f}, "
                    f"Completed Vehicles: {len(vehicle_travel_times)}"
                )
    except (KeyboardInterrupt, traci.exceptions.FatalTraCIError):
        print("\nSimulation interrupted (GUI closed). Exiting.")
        trainer.close()
        return None

    avg_queue = sum(total_queue_length) / len(total_queue_length) if total_queue_length else 0.0
    peak_queue = max(total_queue_length) if total_queue_length else 0.0
    completed_vehicles = len(vehicle_travel_times)
    avg_wait = sum(vehicle_waiting_times) / completed_vehicles if completed_vehicles else 0.0
    avg_travel_time = sum(vehicle_travel_times) / completed_vehicles if completed_vehicles else 0.0
    avg_delay_time = sum(vehicle_delay_times) / completed_vehicles if completed_vehicles else 0.0

    w_max = trainer.traci_service.max_wait
    sorted_waits = sorted(vehicle_waiting_times)
    max_wait_veh = sorted_waits[-1] if sorted_waits else 0.0
    p95_wait = sorted_waits[min(int(0.95 * len(sorted_waits)), len(sorted_waits) - 1)] if sorted_waits else 0.0
    frac_over = sum(1 for w in sorted_waits if w >= w_max) / len(sorted_waits) if sorted_waits else 0.0
    network_jain = trainer.traci_service.compute_jain_index(
        [trainer.traci_service.service_score(w) for w in sorted_waits]
    )
    mean_intersection_jain = sum(intersection_jain) / len(intersection_jain) if intersection_jain else 1.0
    jain_incident_mean = sum(jain_incident) / len(jain_incident) if jain_incident else 1.0
    jain_normal_mean = sum(jain_normal) / len(jain_normal) if jain_normal else 1.0
    overrides = trainer.traci_service.override_count
    throughput = completed_vehicles / len(total_queue_length) if total_queue_length else 0.0

    emv_count = len(emv_response_times)
    sorted_emv = sorted(emv_response_times)
    emv_avg = sum(sorted_emv) / emv_count if emv_count else 0.0
    emv_p95 = sorted_emv[min(int(0.95 * len(sorted_emv)), len(sorted_emv) - 1)] if sorted_emv else 0.0

    print("\n" + "=" * 50)
    print(f"EVALUATION METRICS SUMMARY ({controller})")
    print("=" * 50)
    print(f"Total Steps Evaluated      : {len(total_queue_length)}")
    print(f"Completed Vehicles         : {completed_vehicles}")
    print(f"Throughput                 : {throughput:.4f} vehicles/step")
    print(f"Average Queue Length       : {avg_queue:.2f} vehicles")
    print(f"Average Waiting Time       : {avg_wait:.2f} seconds")
    print(f"Average Travel Time (ATT)  : {avg_travel_time:.2f} seconds")
    print(f"Average Delay Time (ADT)   : {avg_delay_time:.2f} seconds")
    print(f"Peak Total Queue Length    : {peak_queue:.2f} vehicles")
    print(f"Jain Index (vehicle waits) : {network_jain:.4f}")
    print(f"Mean Intersection Jain     : {mean_intersection_jain:.4f}")
    print(f"  ...during incidents      : {jain_incident_mean:.4f} (n={len(jain_incident)})")
    print(f"  ...incident-free         : {jain_normal_mean:.4f} (n={len(jain_normal)})")
    print(f"Max / P95 Vehicle Wait     : {max_wait_veh:.2f} / {p95_wait:.2f} seconds")
    print(f"Vehicles with wait >= {w_max:.0f}s : {frac_over * 100:.2f}%")
    print(f"Max Intersection Timer     : {max_timer_seen:.2f} seconds")
    print(f"Starvation/EMV Overrides   : {overrides}")
    print(f"EMV Response Time (mean/P95): {emv_avg:.2f} / {emv_p95:.2f} seconds (n={emv_count})")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "timestamp": timestamp,
        "label": label or controller,
        "controller": controller,
        "scenario": SCENARIO,
        "model_dir": model_dir or "",
        "use_emv_routing": trainer.traci_service.config.use_emv_routing,
        "use_accidents": trainer.traci_service.config.use_accidents,
        "steps": len(total_queue_length),
        "completed_vehicles": completed_vehicles,
        "avg_queue": round(avg_queue, 4),
        "peak_queue": round(peak_queue, 4),
        "avg_waiting_time": round(avg_wait, 4),
        "avg_travel_time": round(avg_travel_time, 4),
        "avg_delay_time": round(avg_delay_time, 4),
        "max_wait": round(max_wait_veh, 4),
        "p95_wait": round(p95_wait, 4),
        "frac_over_wmax": round(frac_over, 4),
        "jain_network": round(network_jain, 4),
        "jain_mean_intersection": round(mean_intersection_jain, 4),
        "jain_intersection_incident": round(jain_incident_mean, 4),
        "jain_intersection_normal": round(jain_normal_mean, 4),
        "throughput_veh_per_step": round(throughput, 4),
        "emv_count": emv_count,
        "emv_avg_response_time": round(emv_avg, 4),
        "emv_p95_response_time": round(emv_p95, 4),
        "starvation_overrides": overrides,
    }
    _write_csv_row(csv_path, row)

    with open("../metrics.txt", "a") as f:
        f.write(f"--- Evaluation Snapshot: {timestamp} ({controller}) ---\n")
        for key, value in row.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

    trainer.close()
    return row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Traffic Control Model")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing the saved model parts e.g. ../models/run_XX (required for --controller algorithm)",
    )
    parser.add_argument(
        "--controller",
        choices=["algorithm", "fixed", "max_pressure"],
        default="algorithm",
        help="Which controller to evaluate (Section 5.2): the trained policy, or a non-learning baseline.",
    )
    parser.add_argument("--label", type=str, default=None, help="Row label for the CSV output (default: controller)")
    parser.add_argument("--csv", type=str, default="../paper_metrics/eval_results.csv", help="CSV output path")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to evaluate (default: 500)")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch sumo-gui and watch the model drive the scenario. Skips metrics computation entirely.",
    )
    parser.add_argument("--delay", type=float, default=50.0, help="sumo-gui playback delay in ms per step (--gui only)")
    args = parser.parse_args()

    target_dir = None
    if args.controller == "algorithm":
        target_dir = find_latest_model_dir(args.model_dir)
        if not target_dir or not os.path.exists(target_dir):
            print("Error: Could not find any saved models in ../models/ and no valid --model-dir was provided.")
            print("Make sure you have trained and saved models before evaluating.")
            exit(1)

    if args.gui:
        run_gui_simulation(target_dir, steps=args.steps or 3600, delay=args.delay)
    else:
        evaluate(args.controller, target_dir, args.steps or 500, label=args.label, csv_path=args.csv)
