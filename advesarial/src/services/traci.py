import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

from collections import defaultdict, deque

import sumolib.net
import torch
import traci
from schema import TraciConfig
from sumolib import checkBinary
from utils.compute_phase_history import compute_phase_entropy


class TraciService:
    def __init__(self, config: TraciConfig):
        self.config = config
        self.__set_config_path(config.config_path)
        self.tls_ids = []
        self.phase_history = self.phase_histories = defaultdict(lambda: deque(maxlen=20))
        self.edge_ids = set()
        self.tl_ids = set()
        self.node_to_edges = defaultdict(list)
        self._net = None
        self._net_bounds = None

        self.valid_phases = {}
        self.min_green_time = config.min_green_steps
        self.green_duration = config.green_duration
        self.yellow_time = config.yellow_duration

        self.time_since_last_switch = {}
        self.in_transition = {}
        self.pending_phase = {}
        self.yellow_timer = {}
        self.yellow_phases = {}

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
        cmd += self.__route_file_override()
        return cmd

    def __route_file_override(self) -> list:
        """Prefer a specific demand route file (e.g. arterial4x4_42.rou.xml) over the sumocfg default."""
        route = os.environ.get("TRAFFIC_ROUTE")
        scenario = os.environ.get("TRAFFIC_SCENARIO", "manhattan")
        if route is None:
            return []
        route_file = os.path.abspath(f"../scenarios/{scenario}/{scenario}_{route}.rou.xml")
        if not os.path.exists(route_file):
            raise FileNotFoundError(f"Route file not found: {route_file}")
        return ["--route-files", route_file]

    def start_simulation(self):
        cmd = self._build_sumo_command()
        traci.start(cmd)
        self.edge_ids = set(traci.edge.getIDList())
        self.tl_ids = set(traci.trafficlight.getIDList())
        self.tls_ids = sorted(list(self.tl_ids))
        self.__initialize_traci_state()
        self.__normalize_phase_durations()
        self._build_node_to_edges()

    def _build_node_to_edges(self):
        """Map each network junction to its incident (non-internal) edge IDs.

        Loaded once from the net file so cluster states can aggregate real
        edge-level traffic features (queue, wait, count, speed) per region
        instead of falling back to pressure-only tokens.
        """
        net_path = os.path.abspath(self.config_path.replace(".sumocfg", ".net.xml"))
        if not os.path.exists(net_path):
            return

        try:
            net = sumolib.net.readNet(net_path)
        except Exception:
            return

        self._net = net
        self._net_bounds = net.getBoundary()  # (min_x, min_y, max_x, max_y)

        for edge in net.getEdges():
            if edge.isSpecial():
                continue
            self.node_to_edges[edge.getFromNode().getID()].append(edge.getID())
            self.node_to_edges[edge.getToNode().getID()].append(edge.getID())

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

    def compute_intersection_pressure(self, tl_id):
        pressure = 0.0

        controlled_links = traci.trafficlight.getControlledLinks(tl_id)

        for signal_group in controlled_links:
            for link in signal_group:
                in_lane = link[0]
                out_lane = link[1]

                q_in = traci.lane.getLastStepHaltingNumber(in_lane)
                q_out = traci.lane.getLastStepHaltingNumber(out_lane)

                pressure += q_in - q_out

        return pressure

    def get_all_intersections(self):
        return sorted(list(set(traci.trafficlight.getIDList())))

    def __get_valid_phases(self, tls_id):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = logic.phases

        valid = []

        for i, phase in enumerate(phases):
            state = phase.state
            if ("G" in state or "g" in state) and ("y" not in state):
                valid.append(i)

        return valid

    def __build_valid_phases(self):
        self.valid_phases = {}

        for tls_id in self.get_all_intersections():
            v = self.__get_valid_phases(tls_id)
            if not len(v):
                v = [0]

            self.valid_phases[tls_id] = v

    def __initialize_traci_state(self):
        self.__build_valid_phases()
        intersections = self.tls_ids
        self.time_since_last_switch = {tls_id: 0 for tls_id in intersections}
        self.in_transition = {tls: False for tls in intersections}
        self.pending_phase = {tls: None for tls in intersections}
        self.yellow_timer = {tls: 0 for tls in intersections}
        self.yellow_phases = {tls: self.__get_yellow_phases(tls) for tls in intersections}

    def __get_yellow_phases(self, tls_id):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = logic.phases

        yellow = []
        for i, phase in enumerate(phases):
            if "y" in phase.state:
                yellow.append(i)

        return yellow

    def __normalize_phase_durations(self):
        from sumolib.net import Phase

        for tls_id in self.tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = []
            for phase in logic.phases:
                dur = phase.duration
                state = phase.state
                if "y" in state:
                    dur = min(dur, self.yellow_time)
                elif "G" in state or "g" in state:
                    dur = min(dur, self.green_duration)
                phases.append(Phase(dur, state, minDur=dur, maxDur=dur))

            new_logic = traci.trafficlight.Logic(
                logic.programID,
                logic.type,
                0,
                phases,
                logic.subParameter,
            )
            traci.trafficlight.setProgramLogic(tls_id, new_logic)

        self.__build_valid_phases()
        self.yellow_phases = {tls: self.__get_yellow_phases(tls) for tls in self.tls_ids}

    def set_phase(self, tls_id, action: int) -> int:
        """
        Returns:
            safe_phase: int
        """
        valid_phases = self.valid_phases[tls_id]
        num_phases = len(valid_phases)
        safe_phase = valid_phases[action % num_phases]

        current_phase = traci.trafficlight.getPhase(tls_id)

        if self.in_transition[tls_id]:
            self.yellow_timer[tls_id] += 1

            if self.yellow_timer[tls_id] >= self.yellow_time:
                # move to target green phase
                traci.trafficlight.setPhase(tls_id, self.pending_phase[tls_id])

                next_phase = self.pending_phase[tls_id]
                self.in_transition[tls_id] = False
                self.pending_phase[tls_id] = None
                self.time_since_last_switch[tls_id] = 0

                return next_phase

            return current_phase

        if current_phase == safe_phase:
            self.time_since_last_switch[tls_id] = min(self.time_since_last_switch[tls_id] + 1, self.min_green_time)
            return current_phase

        if self.time_since_last_switch[tls_id] < self.min_green_time:
            self.time_since_last_switch[tls_id] += 1
            return current_phase

        yellow_list = self.yellow_phases[tls_id]
        if len(yellow_list) > 0:
            yellow_phase = yellow_list[0]  # simple strategy

            traci.trafficlight.setPhase(tls_id, yellow_phase)

            self.in_transition[tls_id] = True
            self.pending_phase[tls_id] = safe_phase
            self.yellow_timer[tls_id] = 0

            return yellow_phase

        traci.trafficlight.setPhase(tls_id, safe_phase)
        self.time_since_last_switch[tls_id] = 0

        return safe_phase

    def compute_global_state_now(self):
        W, Q = [], []

        for tl_id in self.get_all_intersections():
            lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))

            waiting_time = 0
            queue_length = 0
            # vehicle_count = 0

            for lane in lanes:
                veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                # vehicle_count += len(veh_ids)

                for v in veh_ids:
                    waiting_time += traci.vehicle.getWaitingTime(v)

                queue_length += sum(1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.3)

            W.append(waiting_time)
            Q.append(queue_length)
            # V.append(vehicle_count)

        W = torch.tensor(W, dtype=torch.float32)
        Q = torch.tensor(Q, dtype=torch.float32)
        # V = torch.tensor(V, dtype=torch.float32)

        # No normalization to preserve information and avoid sparsity
        # Use raw values for better state representation

        return W, Q

    def get_adjacency_list(self):
        """
        Returns:
            adj_list: Dict[int, List[int]]
        """

        intersections = self.get_all_intersections()
        id_to_idx = {node_id: idx for idx, node_id in enumerate(intersections)}

        adj_list = {idx: set() for idx in range(len(intersections))}

        for edge in traci.edge.getIDList():
            try:
                source = traci.edge.getFromNode(edge)
                target = traci.edge.getToNode(edge)

                if source in id_to_idx and target in id_to_idx:
                    i = id_to_idx[source]
                    j = id_to_idx[target]

                    adj_list[i].add(j)
                    adj_list[j].add(i)

            except Exception:
                continue

        # convert to list
        adj_list = {i: list(neighbors) for i, neighbors in adj_list.items()}

        return adj_list

    def get_observations(self):
        """
        Returns:
        tensor of shape (num_clusters, 10)
        """

        states = []

        for tl_id in self.tls_ids:
            car_num = 0
            queue_length = 0
            occupancy = 0
            flow = 0
            stop_car_num = 0
            waiting_time = 0
            average_speed = 0

            lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))

            for lane in lanes:
                car_num += traci.lane.getLastStepVehicleNumber(lane)
                queue_length += traci.lane.getLastStepHaltingNumber(lane)
                occupancy += traci.lane.getLastStepOccupancy(lane)
                flow += traci.lane.getLastStepVehicleNumber(lane)
                stop_car_num += traci.lane.getLastStepHaltingNumber(lane)
                waiting_time += traci.lane.getWaitingTime(lane)
                average_speed += traci.lane.getLastStepMeanSpeed(lane)

            num_lanes = len(lanes)
            occupancy /= num_lanes if num_lanes > 0 else 1
            average_speed = average_speed / num_lanes if num_lanes > 0 else 0

            pressure = self.compute_intersection_pressure(tl_id)
            congestion_ratio = queue_length / car_num if car_num > 0 else 0.0
            delay = waiting_time / car_num if car_num > 0 else 0.0

            state = torch.tensor(
                [
                    car_num,
                    queue_length,
                    occupancy,
                    flow,
                    stop_car_num,
                    waiting_time,
                    average_speed,
                    pressure,
                    congestion_ratio,
                    delay,
                ],
                dtype=torch.float32,
            )

            states.append(state)

        return torch.stack(states)

    def get_intersection_reward(self, tls):
        """Local reward r_i = -(ql + wt + dt + ps - ss), Section 4.3.1.

        delay_time is approximated per-step as the sum, over incoming lanes, of
        (actual travel time - ideal free-flow travel time) implied by the
        lane's current mean speed vs. its speed limit (the paper defines
        delay_time as "ideal travel time - actual travel time" per vehicle but
        gives no per-step formula; this lane-level speed proxy is the natural
        per-step analogue used elsewhere in this codebase for the same idea).
        """

        queue_length = 0
        waiting_time = 0
        pressure = self.compute_intersection_pressure(tls)
        speed_score = 0
        delay_time = 0.0

        lanes = traci.trafficlight.getControlledLanes(tls)
        for lane in lanes:
            waiting_time += traci.lane.getWaitingTime(lane)
            queue_length += traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            speed_score += mean_speed

            length = traci.lane.getLength(lane)
            max_speed = traci.lane.getMaxSpeed(lane)
            if max_speed > 0:
                ideal_time = length / max_speed
                actual_time = length / max(mean_speed, 0.1)
                delay_time += max(actual_time - ideal_time, 0.0)

        speed_score /= len(lanes) if len(lanes) > 0 else 1

        final = -(queue_length + waiting_time + delay_time + pressure - speed_score)
        return final

    def total_waiting_time(self, tls):
        return sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls))

    def total_queue_length(self, tls):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls))

    def _get_incoming_lanes(self, tls_id):
        incoming_lanes = set()
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for link_group in controlled_links:
            for in_lane, _, _ in link_group:
                incoming_lanes.add(in_lane)

        return list(incoming_lanes)

    def get_stop_count(self, intersection_id, speed_threshold=0.1):
        """
        Counts number of stopped vehicles near an intersection.
        """

        stop_count = 0

        # Get all incoming lanes of this intersection
        incoming_lanes = self._get_incoming_lanes(intersection_id)

        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

            for veh_id in vehicle_ids:
                speed = traci.vehicle.getSpeed(veh_id)

                if speed < speed_threshold:
                    stop_count += 1

        return stop_count

    def get_emergency_waiting_time(self, intersection_id):
        """
        Computes total waiting time of emergency vehicles at an intersection.
        """

        total_waiting_time = 0.0

        incoming_lanes = self._get_incoming_lanes(intersection_id)

        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

            for veh_id in vehicle_ids:
                veh_type = traci.vehicle.getTypeID(veh_id)

                if veh_type == "emergency":
                    waiting_time = traci.vehicle.getWaitingTime(veh_id)
                    total_waiting_time += waiting_time

        return total_waiting_time

    def get_non_emv_waiting_times(self, tls_id):
        """
        Returns waiting times of non-emergency vehicles at an intersection.
        """
        waiting_times = []

        incoming_lanes = self._get_incoming_lanes(tls_id)

        for lane in incoming_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

            for veh_id in vehicle_ids:
                veh_type = traci.vehicle.getTypeID(veh_id)

                if veh_type != "emergency":
                    waiting_time = traci.vehicle.getWaitingTime(veh_id)
                    waiting_times.append(waiting_time)

        return waiting_times

    def get_pedestrian_waiting_times(self, tls_id):
        """
        Returns waiting times of pedestrians at an intersection.
        """
        waiting_times = []

        incoming_lanes = self._get_incoming_lanes(tls_id)

        for lane in incoming_lanes:
            # Get pedestrians on this lane
            edge_id = traci.lane.getEdgeID(lane)
            person_ids = traci.edge.getLastStepPersonIDs(edge_id)

            for person_id in person_ids:
                # Waiting time for pedestrians
                waiting_time = traci.person.getWaitingTime(person_id)
                waiting_times.append(waiting_time)

        return waiting_times

    def get_pedestrian_conflict_count(self, tls_id):
        """
        Returns number of pedestrian-vehicle conflicts at an intersection.
        """

        conflict_count = 0

        # Get lanes controlled by this traffic light
        incoming_lanes = set(self._get_incoming_lanes(tls_id))

        # Get all collisions in current step
        collisions = traci.simulation.getCollisions()

        for col in collisions:
            # SUMO >= 1.20 uses "collider"/"victim"; older versions used "colliderID"/"victimID"
            collider = getattr(col, "collider", None) or getattr(col, "colliderID", None)
            victim = getattr(col, "victim", None) or getattr(col, "victimID", None)

            if collider is None or victim is None:
                continue

            # Determine types
            collider_is_vehicle = collider in traci.vehicle.getIDList()
            victim_is_vehicle = victim in traci.vehicle.getIDList()

            collider_is_person = collider in traci.person.getIDList()
            victim_is_person = victim in traci.person.getIDList()

            # We only care about vehicle <-> pedestrian collisions
            if (collider_is_vehicle and victim_is_person) or (collider_is_person and victim_is_vehicle):
                # Get lane of the vehicle (person may not have lane always)
                if collider_is_vehicle:
                    lane_id = traci.vehicle.getLaneID(collider)
                else:
                    lane_id = traci.vehicle.getLaneID(victim)

                # Check if collision happened in this intersection
                if lane_id in incoming_lanes:
                    conflict_count += 1

                    print(f"[Conflict @ {tls_id}] time={traci.simulation.getTime()} | {collider} hit {victim}")

        return conflict_count

    def get_regional_state(self, clusters):
        """Per-subregion state fed to the Meta-Policy's Transformer (Appendix A):

        [stop_car_num, waiting_time, centroid_x, centroid_y] aggregated over the
        subregion's traffic lights, with (x, y) normalized to [0, 1] by the
        network's bounding box.

        Returns:
            tensor of shape (num_clusters, 4)
        """

        min_x, min_y, max_x, max_y = self._net_bounds or (0.0, 0.0, 1.0, 1.0)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)

        region_states = []

        for _, cluster_nodes in clusters.items():
            stop_car_num = 0.0
            waiting_time = 0.0
            centroid_x = 0.0
            centroid_y = 0.0
            valid_nodes = 0

            for node in cluster_nodes:
                if node in self.tl_ids:
                    stop_car_num += self.get_stop_count(node)
                    waiting_time += self.total_waiting_time(node)

                if self._net is not None:
                    try:
                        x, y = self._net.getNode(node).getCoord()
                        centroid_x += x
                        centroid_y += y
                        valid_nodes += 1
                    except Exception:
                        pass

            if valid_nodes == 0:
                valid_nodes = 1

            centroid_x = (centroid_x / valid_nodes - min_x) / width
            centroid_y = (centroid_y / valid_nodes - min_y) / height

            region_states.append(
                torch.tensor([stop_car_num, waiting_time, centroid_x, centroid_y], dtype=torch.float32)
            )

        return torch.stack(region_states)

    def cluster_entropy(self, cluster_nodes, phase_histories):
        total = 0.0

        for node in cluster_nodes:
            history = phase_histories[node]
            total += compute_phase_entropy(list(history))

        return total / len(cluster_nodes)

    # def get_adjacency_list(self, num_clusters):
    #     from collections import defaultdict

    #     adj_set = defaultdict(set)

    #     for edge_id in traci.edge.getIDList():
    #         try:
    #             from_node = traci.edge.getFromNode(edge_id)
    #             to_node   = traci.edge.getToNode(edge_id)

    #             if from_node in node_to_cluster and to_node in node_to_cluster:
    #                 c1 = node_to_cluster[from_node]
    #                 c2 = node_to_cluster[to_node]

    #                 if c1 != c2:
    #                     adj_set[c1].add(c2)
    #                     adj_set[c2].add(c1)

    #         except:
    #             continue

    #     adj_list = {i: list(adj_set[i]) for i in range(num_clusters)}

    #     return adj_list

    #                     adj_set[c2].add(c1)

    #         except:
    #             continue

    #     adj_list = {i: list(adj_set[i]) for i in range(num_clusters)}

    #     return adj_list
