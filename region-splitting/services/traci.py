import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

import traci
import numpy as np
from sumolib import checkBinary
from schema import TraciConfig
import json


class TraciService:
    def __init__(self, config: TraciConfig):
        self.config = config
        self.__set_config_path(config.config_path)
        self.tls_ids = []

    def __set_config_path(self, config_path):
        if not config_path.endswith(".sumocfg"):
            raise ValueError("Config path must end with .sumocfg")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        self.config_path = config_path

    def _build_sumo_command(self) -> list:
        binary = checkBinary("sumo-gui" if self.config.use_gui else "sumo")
        cmd = [
            binary,
            "-c",
            self.config_path,
            "--step-length",
            str(self.config.step_length),
            "--start",
            "--no-step-log",
            "--verbose",
            "true",
        ]
        if self.config.use_gui:
            cmd += ["--delay", str(self.config.delay)]
        return cmd

    def start_simulation(self):
        cmd = self._build_sumo_command()
        traci.start(cmd)
        self.tls_ids = sorted(traci.trafficlight.getIDList())

    def close_simulation(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

    def reset_simulation(self):
        self.close_simulation()
        self.start_simulation()

    def step(self):
        traci.simulationStep()

    def get_tls_ids(self):
        tls_ids = sorted(traci.trafficlight.getIDList())

        tls_config = {}

        for tls_id in tls_ids:
            tls_config[tls_id] = [
                lane for lane in traci.trafficlight.getControlledLanes(tls_id)
            ]

        with open("tls_config.json", "w") as f:
            json.dump(tls_config, f, indent=4)
        return tls_ids

    def get_edge_list(self):
        return traci.edge.getIDList()

    def get_edge_weight(self, edge):

        edge_id = edge.getID()
        vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
        waiting = traci.edge.getWaitingTime(edge_id)

        speed = traci.edge.getLastStepMeanSpeed(edge_id)
        max_speed = edge.getSpeed()

        if max_speed > 0:
            congestion = 1 - (speed / max_speed)
            congestion = max(0, min(1, congestion))
        else:
            congestion = 0

        alpha, beta, gamma = 1.0, 0.3, 2.0

        weight = alpha * vehicles + beta * waiting + gamma * congestion * vehicles

        return weight
