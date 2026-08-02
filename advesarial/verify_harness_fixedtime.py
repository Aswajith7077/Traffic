"""TEMP verification: run the evaluation's measurement logic with a FIXED-TIME
controller (no phase changes) and cross-check ATT/ADT against SUMO's own
tripinfo output for the same run.

Run from advesarial/ with SUMO_HOME set.
"""

import os
import sys
import json
from statistics import mean

sys.path.insert(0, os.path.abspath("src"))

os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo"
import traci
from sumolib import checkBinary

SCENARIO = "D:\\College\\Semester 9\\RSL\\Traffic\\scenarios\\cologne8\\cologne8.sumocfg"
TRIPINFO_OUT = "D:\\College\\Semester 9\\RSL\\Traffic\\scenarios\\cologne8\\tripinfos_harness.xml"
STEPS = 3600

cmd = [
    checkBinary("sumo"),
    "-c",
    SCENARIO,
    "--step-length",
    "1.0",
    "--start",
    "--no-step-log",
    "--tripinfo-output",
    TRIPINFO_OUT,
]
traci.start(cmd)

vehicle_metrics = {}
travel_times, delay_times, waiting_times = [], [], []
queue_samples = []


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


def total_halted_queue(tls):
    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in traci.trafficlight.getControlledLanes(tls)
    )


intersections = sorted(set(traci.trafficlight.getIDList()))
sim_begin = traci.simulation.getTime()

for t in range(STEPS):
    current_time = traci.simulation.getTime()
    for veh in traci.vehicle.getIDList():
        if veh not in vehicle_metrics:
            vehicle_metrics[veh] = {
                "entry_time": current_time,
                "free_flow_time": get_free_flow_travel_time(traci.vehicle.getRoute(veh)),
                "waiting_time": 0.0,
            }
        vehicle_metrics[veh]["waiting_time"] = traci.vehicle.getAccumulatedWaitingTime(veh)

    traci.simulationStep()

    current_time = traci.simulation.getTime()
    for veh in traci.simulation.getArrivedIDList():
        m = vehicle_metrics.pop(veh, None)
        if m is None:
            continue
        tt = current_time - m["entry_time"]
        travel_times.append(tt)
        delay_times.append(tt - m["free_flow_time"])
        waiting_times.append(m["waiting_time"])

    queue_samples.append(sum(total_halted_queue(i) for i in intersections))

    if (t + 1) % 600 == 0:
        print(f"step {t + 1}/{STEPS} completed={len(travel_times)}")

traci.close()

n = len(travel_times)
print("\n" + "=" * 50)
print("HARNESS FIXED-TIME SUMMARY")
print("=" * 50)
print(f"Sim window: {sim_begin:.0f} - {current_time:.0f}")
print(f"Completed vehicles: {n}")
print(f"Avg queue (halted): {mean(queue_samples):.2f}")
print(f"Avg waiting: {mean(waiting_times):.2f}")
print(f"ATT (harness): {mean(travel_times):.2f}")
print(f"ADT (harness): {mean(delay_times):.2f}")
print("=" * 50)

with open(TRIPINFO_OUT, "r") as f:
    raw = f.read()
import re

infos = re.findall(
    r'<tripinfo id="([^"]+)" [^>]*duration="([\d.]+)"[^>]*waitingTime="([\d.]+)"[^>]*timeLoss="([\d.]+)"',
    raw,
)
durs = [float(x[1]) for x in infos]
waits = [float(x[2]) for x in infos]
losses = [float(x[3]) for x in infos]
print("SUMO tripinfo cross-check:")
print(f"  tripinfos: {len(infos)}")
print(f"  SUMO ATT (mean duration): {mean(durs):.2f}   vs harness {mean(travel_times):.2f}")
print(f"  SUMO avg waiting: {mean(waits):.2f}           vs harness {mean(waiting_times):.2f}")
print(f"  SUMO ADT (mean timeLoss): {mean(losses):.2f}  vs harness {mean(delay_times):.2f}")

by_id = {x[0]: x for x in infos}
ids = sorted(set(by_id) & {k for k in travel_times}) if False else sorted(by_id.keys())
harness_by_id = {}
# re-derive per-vehicle from the same tracking (approximate: recompute map)
print("\nPer-vehicle comparison (first 10):")
for vid in sorted(by_id)[:10]:
    d, w, l = float(by_id[vid][1]), float(by_id[vid][2]), float(by_id[vid][3])
    print(f"  {vid}: sumo dur={d:.1f} wait={w:.1f} loss={l:.1f}")
