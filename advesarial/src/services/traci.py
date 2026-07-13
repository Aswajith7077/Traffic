import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

import torch
import traci
from sumolib import checkBinary
from schema import TraciConfig
from collections import defaultdict
from collections import deque
from utils.compute_phase_history import compute_phase_entropy


class TraciService:
    def __init__(self, config: TraciConfig):
        self.config = config
        self.__set_config_path(config.config_path)
        self.tls_ids = []
        self.phase_history = self.phase_histories = defaultdict(
            lambda: deque(maxlen=20)
        )
        self.edge_ids = set()
        self.tl_ids = set()

        self.valid_phases = {}
        self.min_green_time = 5
        self.yellow_time = 3

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

        print("config_path: ", config_path)
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
        self.edge_ids = set(traci.edge.getIDList())
        self.tl_ids = set(traci.trafficlight.getIDList())
        self.tls_ids = sorted(list(self.tl_ids))
        self.__initialize_traci_state()

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
        self.yellow_phases = {
            tls: self.__get_yellow_phases(tls) for tls in intersections
        }

    def __get_yellow_phases(self, tls_id):
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phases = logic.phases

        yellow = []
        for i, phase in enumerate(phases):
            if "y" in phase.state:
                yellow.append(i)

        return yellow

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
            self.time_since_last_switch[tls_id] = min(
                self.time_since_last_switch[tls_id] + 1, self.min_green_time
            )
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

                queue_length += sum(
                    1 for v in veh_ids if traci.vehicle.getSpeed(v) < 0.3
                )

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

        queue_length = 0
        waiting_time = 0
        pressure = self.compute_intersection_pressure(tls)
        speed_score = 0

        lanes = traci.trafficlight.getControlledLanes(tls)
        for lane in lanes:
            waiting_time += traci.lane.getWaitingTime(lane)
            queue_length += traci.lane.getLastStepVehicleNumber(lane)
            speed_score += traci.lane.getLastStepMeanSpeed(lane)

        speed_score /= len(lanes) if len(lanes) > 0 else 1

        final = -(queue_length + waiting_time + pressure - speed_score)
        return final

    def total_waiting_time(self, tls):
        return sum(
            traci.lane.getWaitingTime(lane)
            for lane in traci.trafficlight.getControlledLanes(tls)
        )

    def total_queue_length(self, tls):
        return sum(
            traci.lane.getLastStepVehicleNumber(lane)
            for lane in traci.trafficlight.getControlledLanes(tls)
        )

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
            collider = col.colliderID
            victim = col.victimID

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

                    print(
                        f"[Conflict @ {tls_id}] time={traci.simulation.getTime()} | "
                        f"{collider} hit {victim}"
                    )

        return conflict_count

    def get_cluster_states(self, clusters):
        """
        Returns:
        tensor of shape (num_clusters, 10)
        """

        cluster_states = []

        for _, cluster_nodes in clusters.items():
            total_queue_length = 0.0
            total_waiting_time = 0.0
            total_vehicle_count = 0.0
            total_speed = 0.0

            incoming_flow = 0.0
            outgoing_flow = 0.0

            internal_cluster_flow = 0.0
            pressure = 0.0

            valid_nodes = 0
            for node in cluster_nodes:
                if node in self.edge_ids:
                    try:
                        occupancy = traci.edge.getLastStepOccupancy(node)
                        waiting_time = traci.edge.getWaitingTime(node)
                        vehicle_count = traci.edge.getLastStepVehicleNumber(node)
                        mean_speed = traci.edge.getLastStepMeanSpeed(node)

                        total_queue_length += occupancy
                        total_waiting_time += waiting_time
                        total_vehicle_count += vehicle_count
                        total_speed += mean_speed

                        incoming_flow += vehicle_count
                        outgoing_flow += vehicle_count

                        valid_nodes += 1
                    except Exception:
                        pass

                if node in self.tl_ids:
                    pressure += self.compute_intersection_pressure(node)

            if valid_nodes == 0:
                valid_nodes = 1

            avg_waiting_time = total_waiting_time / valid_nodes
            avg_speed = total_speed / valid_nodes

            internal_cluster_flow = incoming_flow - outgoing_flow

            congestion_ratio = total_queue_length / (total_vehicle_count + 1e-6)

            signal_phase_entropy = compute_phase_entropy(self.phase_history)

            cluster_state = torch.tensor(
                [
                    total_queue_length,
                    avg_waiting_time,
                    total_vehicle_count,
                    avg_speed,
                    incoming_flow,
                    outgoing_flow,
                    internal_cluster_flow,
                    pressure,
                    congestion_ratio,
                    signal_phase_entropy,
                ],
                dtype=torch.float32,
            )

            cluster_states.append(cluster_state)

        return torch.stack(cluster_states)

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
