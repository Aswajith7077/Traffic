"""Stationary baselines: Fixed-Time Control (FTC) and MaxPressure (MP).

Both run the same 3600s episode, same control cadence (10s) and report
identical metrics as the RL evaluation: per-step halted queues plus
paper-style ATT/ADT parsed from SUMO tripinfo output.

Usage (from advesarial/):
    python src/baselines.py                          # paper demand
    python src/baselines.py --scenario ../scenarios/cologne8/cologne8.sumocfg
"""

import argparse
import os

import traci
from environment import Environment
from schema import TraciConfig
from services import TraciService
from utils.tripinfo_metrics import parse_tripinfo, print_tripinfo_summary

CONTROL_INTERVAL = 10
STEPS = 360
DEFAULT_SCENARIO = "../scenarios/cologne8/cologne8.sumocfg"
DRAIN_BUDGET = 3600


def total_halted_queue(intersections):
    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for i in intersections
        for lane in traci.trafficlight.getControlledLanes(i)
    )


class BaselineRunner:
    def __init__(self, name, scenario_cfg):
        os.makedirs("visualizations", exist_ok=True)
        self.name = name
        self.traci_config = TraciConfig(
            config_path=scenario_cfg,
            tripinfo_output=f"visualizations/tripinfo_{name}.xml",
        )
        self.service = TraciService(self.traci_config)
        self.service.start_simulation()
        self.environment = Environment(traci_service=self.service)
        self.intersections = self.service.get_all_intersections()

    def close(self):
        self.service.close_simulation()

    def run(self, control_fn):
        queue_hist = []
        for t in range(STEPS):
            control_fn()
            self.service.step(CONTROL_INTERVAL)
            queue_hist.append(total_halted_queue(self.intersections))
            if (t + 1) % 90 == 0:
                print(
                    f"[{self.name}] step {t + 1}/{STEPS} "
                    f"(sim {traci.simulation.getTime():.0f}s) "
                    f"queue {queue_hist[-1]:.0f}"
                )

        # Drain phase: keep stepping until every vehicle has arrived so the
        # tripinfo metrics cover all trips (paper protocol), capped by a budget.
        remaining = traci.simulation.getMinExpectedNumber()
        drain_steps = 0
        while remaining > 0 and drain_steps * CONTROL_INTERVAL < DRAIN_BUDGET:
            drain_steps += 1
            self.service.step(CONTROL_INTERVAL)
            remaining = traci.simulation.getMinExpectedNumber()
            if drain_steps % 60 == 0:
                print(
                    f"[{self.name}] drain: sim {traci.simulation.getTime():.0f}s - "
                    f"{remaining} vehicles remaining"
                )
        if remaining > 0:
            print(
                f"[{self.name}] WARNING: drain budget ({DRAIN_BUDGET}s) exhausted, "
                f"{remaining} vehicles still running"
            )

        avg_queue = sum(queue_hist) / len(queue_hist)
        peak_queue = max(queue_hist)
        print(f"[{self.name}] avg halted queue: {avg_queue:.2f}, peak: {peak_queue:.0f}")
        return avg_queue, peak_queue


def ftc_control(service):
    """Fixed-Time Control: leave the static signal program from the net.xml
    running untouched (all-red/yellow/green durations from the network file)."""
    pass


def mp_control(service, logics):
    """MaxPressure: at every control step switch each TL to the valid phase
    with the highest pressure (in-lane vehicles minus out-lane vehicles)."""

    def pressure_for(tls, phase_idx):
        state = logics[tls].phases[phase_idx].state
        links = traci.trafficlight.getControlledLinks(tls)
        p = 0.0
        for i, group in enumerate(links):
            if i >= len(state) or state[i] not in "gG":
                continue
            for link in group:
                in_lane, out_lane = link[0], link[1]
                p += traci.lane.getLastStepVehicleNumber(in_lane) - traci.lane.getLastStepVehicleNumber(
                    out_lane
                )
        return p

    for tls in service.get_all_intersections():
        valid = service.valid_phases[tls]
        if not valid:
            continue
        pressures = [pressure_for(tls, pid) for pid in valid]
        best_idx = max(range(len(valid)), key=lambda i: pressures[i])
        service.set_phase(tls, best_idx)


def main():
    global STEPS
    parser = argparse.ArgumentParser(description="Run FTC and MaxPressure baselines")
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--steps", type=int, default=STEPS,
        help="Number of 10s control steps (default 360 = full episode)",
    )
    args = parser.parse_args()
    STEPS = args.steps

    os.makedirs("../visualizations", exist_ok=True)

    # --- FTC -----------------------------------------------------------------
    print("\n=== FTC (Fixed-Time Control) ===")
    ftc = BaselineRunner("ftc", args.scenario)
    ftc.run(lambda: ftc_control(ftc.service))
    ftc.close()
    trip = parse_tripinfo("visualizations/tripinfo_ftc.xml")
    print_tripinfo_summary(trip)

    # --- MaxPressure --------------------------------------------------------
    print("\n=== MaxPressure Control ===")
    mp = BaselineRunner("mp", args.scenario)
    logics = {
        tls: traci.trafficlight.getAllProgramLogics(tls)[0]
        for tls in mp.service.get_all_intersections()
    }
    mp.run(lambda: mp_control(mp.service, logics))
    mp.close()
    trip = parse_tripinfo("visualizations/tripinfo_mp.xml")
    print_tripinfo_summary(trip)


if __name__ == "__main__":
    main()
