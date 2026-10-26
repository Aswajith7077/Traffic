import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

import math
import random
from collections import defaultdict, deque

import sumolib.net
import torch
import traci
from schema import TraciConfig
from sumolib import checkBinary
from utils.compute_phase_history import compute_phase_entropy

EMV_TYPE_ID = "emergency"
EMV_BIAS_DECAY = 0.5  # time-decay rate per intersection hop, Section 3.2
ACCIDENT_PROB_PER_STEP = 0.002  # ~1 incident every 500s on average, Section 3.3
ACCIDENT_MIN_DURATION = 30
ACCIDENT_MAX_DURATION = 120


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

        # Fairness: per-vehicle waiting timers and phase -> served-lane maps.
        self.max_wait = config.max_wait
        self.wait_ref = config.wait_ref
        self.wait_timers = {}
        self.timer_lanes = {}
        self.phase_served_lanes = {}
        self._incoming_lanes_cache = {}
        self.override_count = 0

        # Accident/incident injection (Section 3.3): at most one active
        # incident at a time, a random lane forced to ~0 speed for a window.
        self._active_accident = None

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
            # Keep accumulated waiting time over the whole trip (SUMO default window is 100 s),
            # so fairness metrics can see waits beyond W_max.
            "--waiting-time-memory",
            "100000",
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
        self.__build_fairness_maps()
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
        self.maybe_inject_accident()
        traci.simulationStep()

    @property
    def in_incident(self):
        return self._active_accident is not None

    def maybe_inject_accident(self):
        """Random lane-closure incidents (Section 3.3), gated by config.use_accidents.
        At most one active incident at a time; it self-resolves after a random
        duration, restoring the lane's original speed limit."""
        if not self.config.use_accidents:
            return

        if self._active_accident is not None:
            self._active_accident["remaining"] -= 1
            if self._active_accident["remaining"] <= 0:
                traci.lane.setMaxSpeed(self._active_accident["lane"], self._active_accident["original_speed"])
                self._active_accident = None
            return

        if random.random() < ACCIDENT_PROB_PER_STEP and self.edge_ids:
            edge = random.choice(list(self.edge_ids))
            lane_id = f"{edge}_0"
            try:
                original_speed = traci.lane.getMaxSpeed(lane_id)
            except traci.exceptions.TraCIException:
                return
            traci.lane.setMaxSpeed(lane_id, 0.1)
            self._active_accident = {
                "lane": lane_id,
                "original_speed": original_speed,
                "remaining": random.randint(ACCIDENT_MIN_DURATION, ACCIDENT_MAX_DURATION),
            }

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

    def get_action_mask(self, num_actions):
        """(len(tls_ids), num_actions) bool mask: True at raw indices [0, num_phases)
        for that intersection. A single shared actor head (fixed num_actions) is reused
        across intersections with different real phase counts; without masking,
        `action % num_phases` aliases several raw indices onto the same real phase,
        and a policy that settles on an aliased index can hold one phase forever."""

        mask = torch.zeros((len(self.tls_ids), num_actions), dtype=torch.bool)
        for row, tls_id in enumerate(self.tls_ids):
            n = min(len(self.valid_phases[tls_id]), num_actions)
            mask[row, :n] = True

        return mask

    def __build_fairness_maps(self):
        """For every TLS and phase, the set of incoming lanes that phase gives a green to."""
        self.wait_timers = {tls_id: {} for tls_id in self.tls_ids}
        self.timer_lanes = {tls_id: {} for tls_id in self.tls_ids}
        self.override_count = 0
        self.phase_served_lanes = {}
        self._incoming_lanes_cache = {}

        for tls_id in self.tls_ids:
            links = traci.trafficlight.getControlledLinks(tls_id)
            phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
            served = {}
            for idx, phase in enumerate(phases):
                lanes = set()
                for sig, link_group in enumerate(links):
                    if sig < len(phase.state) and phase.state[sig] in "Gg":
                        lanes.update(link[0] for link in link_group)
                served[idx] = lanes
            self.phase_served_lanes[tls_id] = served

    def update_wait_timers(self):
        """Per-vehicle waiting timers w_{v,j} (Section 4.1).

        A vehicle's timer grows by one step while it is stopped (speed < 0.1 m/s) on an
        incoming lane of intersection j. It is kept while the vehicle creeps forward in the
        queue and discarded once the vehicle is served and leaves the incoming lanes.
        """
        dt = self.config.step_length
        for tls_id in self.tls_ids:
            timers = self.wait_timers.setdefault(tls_id, {})
            present = {}
            for lane in self._get_incoming_lanes(tls_id):
                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    waited = timers.get(veh_id, 0.0)
                    if traci.vehicle.getSpeed(veh_id) < 0.1:
                        waited += dt
                    present[veh_id] = (waited, lane)
            self.wait_timers[tls_id] = {v: w for v, (w, _) in present.items()}
            self.timer_lanes[tls_id] = {v: lane for v, (_, lane) in present.items()}

    def get_vehicle_waits(self, tls_id):
        return list(self.wait_timers.get(tls_id, {}).values())

    @staticmethod
    def compute_jain_index(values):
        """Jain's fairness index (sum x)^2 / (n * sum x^2); 1.0 when there is nothing to compare."""
        n = len(values)
        if n == 0:
            return 1.0
        denom = n * sum(x * x for x in values)
        if denom == 0:
            return 1.0
        return sum(values) ** 2 / denom

    def service_score(self, wait):
        """Bounded service score s = 1 / (1 + w / W_ref) used as the Jain-index input."""
        return 1.0 / (1.0 + wait / self.wait_ref)

    def compute_network_jain_now(self):
        """Network-wide Jain's index over every vehicle currently in the
        simulation, using accumulated waiting time -> service_score -- the
        same quantities evaluate.py's reported jain_network metric uses,
        just computed live instead of only at the end of an eval run.

        get_intersection_fairness() below is a *local*, per-intersection,
        instantaneous signal; a policy can score well on it everywhere while
        still producing large accumulated-wait disparities network-wide
        (e.g. some vehicles repeatedly re-queueing at several intersections
        in sequence). This method gives every agent the same network-level
        fairness signal each step, the way compute_goal_reward() already
        broadcasts global W/Q state, so the thing actually being reported
        and compared against baselines (.claude/reports/fairness_implementation.md)
        has direct training signal behind it, not just a local proxy.
        """
        waits = [traci.vehicle.getAccumulatedWaitingTime(v) for v in traci.vehicle.getIDList()]
        if not waits:
            return 1.0
        scores = [self.service_score(w) for w in waits]
        return self.compute_jain_index(scores)

    def get_intersection_fairness(self, tls_id):
        """Local fairness statistics at one intersection (Section 4.1).

        Returns (J, var, max_wait): Jain index over service scores, variance of the
        reference-normalized waiting times, and the maximum waiting timer. With no
        waiting vehicles: (1.0, 0.0, 0.0).
        """
        waits = self.get_vehicle_waits(tls_id)
        if not waits:
            return 1.0, 0.0, 0.0
        normalized = [w / self.wait_ref for w in waits]
        mean = sum(normalized) / len(normalized)
        var = sum((x - mean) ** 2 for x in normalized) / len(normalized)
        jain = self.compute_jain_index([self.service_score(w) for w in waits])
        return jain, var, max(waits)

    def _active_emv_ids(self):
        return [v for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == EMV_TYPE_ID]

    def _emv_phase(self, tls_id, valid_phases, current_phase):
        """Local EMV preemption (Section 5.2's baseline: EMVs pre-empted locally
        at each intersection, with no network-wide route coordination). Active
        regardless of `use_emv_routing` — the city-wide extension only adds the
        anticipatory bias in `compute_emv_bias`/observations on top of this."""
        incoming = set(self._get_incoming_lanes(tls_id))
        for veh_id in self._active_emv_ids():
            try:
                lane = traci.vehicle.getLaneID(veh_id)
            except traci.exceptions.TraCIException:
                continue
            if lane not in incoming:
                continue
            candidates = ([current_phase] if current_phase in valid_phases else []) + [
                p for p in valid_phases if p != current_phase
            ]
            for phase in candidates:
                if lane in self.phase_served_lanes[tls_id].get(phase, ()):
                    return phase
        return None

    def compute_emv_bias(self):
        """Time-decaying, per-intersection anticipatory bias along each active
        EMV's route to its destination (Section 3.2, city-wide multi-EMV
        routing). Arbitration across concurrent EMVs takes the max bias per
        intersection rather than summing, so one intersection can't be pulled
        toward two different EMVs' corridors at once.

        Returns:
            dict[tls_id -> float in (0, 1]], intersections with no upcoming EMV omitted.
        """
        bias = defaultdict(float)
        if self._net is None:
            return bias

        for veh_id in self._active_emv_ids():
            try:
                route = traci.vehicle.getRoute(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)
            except traci.exceptions.TraCIException:
                continue
            if route_index < 0:
                continue

            for hop, edge_id in enumerate(route[route_index:]):
                try:
                    node_id = self._net.getEdge(edge_id).getToNode().getID()
                except (KeyError, traci.exceptions.TraCIException):
                    continue
                if node_id not in self.tl_ids:
                    continue
                decay = math.exp(-EMV_BIAS_DECAY * hop)
                if decay > bias[node_id]:
                    bias[node_id] = decay

        return bias

    def _starved_phase(self, tls_id, valid_phases, current_phase):
        """Valid phase serving the longest-waiting vehicle at/over W_max, else None.

        Checks both the local (this-intersection) wait timer and the
        vehicle's accumulated trip-wide waiting time -- a vehicle that
        crossed W_max via several shorter waits at earlier intersections
        (e.g. 40s at one, 60s at the next) never trips a purely local check,
        since neither single intersection ever sees the full 90s, even
        though the vehicle is genuinely starved overall
        (.claude/reports/fairness_implementation.md). The current phase is
        preferred when it already serves the starved vehicle.
        """
        timers = self.wait_timers.get(tls_id, {})
        lanes = self.timer_lanes.get(tls_id, {})
        starved = sorted(
            ((max(w, traci.vehicle.getAccumulatedWaitingTime(v)), v) for v, w in timers.items()),
            reverse=True,
        )
        starved = [(w, v) for w, v in starved if w >= self.max_wait]
        for _, veh_id in starved:
            lane = lanes.get(veh_id)
            candidates = ([current_phase] if current_phase in valid_phases else []) + [
                p for p in valid_phases if p != current_phase
            ]
            for phase in candidates:
                if lane in self.phase_served_lanes[tls_id].get(phase, ()):
                    return phase
        return None

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
        # `action` comes from a masked softmax (see get_action_mask) so it should
        # already be < num_phases; clamp instead of wrapping (`% num_phases`) as
        # defense-in-depth so an out-of-range action can't alias onto an unintended
        # real phase.
        safe_phase = valid_phases[min(action, num_phases - 1)]

        current_phase = traci.trafficlight.getPhase(tls_id)

        # EMV local preemption (Section 3.2/5.2) takes priority over the fairness
        # starvation override below — "emergency vehicles require substantially
        # higher priority than regular traffic" (Section 4.3). Fairness safeguard
        # (Section 4.1): a vehicle that has reached W_max overrides the policy's
        # choice with a phase that serves it. Either override only changes the
        # *requested* phase; min-green and yellow clearance below still apply.
        if not self.in_transition[tls_id]:
            emv_phase = self._emv_phase(tls_id, valid_phases, current_phase)
            if emv_phase is not None and emv_phase != safe_phase:
                safe_phase = emv_phase
                self.override_count += 1
            else:
                starved_phase = self._starved_phase(tls_id, valid_phases, current_phase)
                if starved_phase is not None and starved_phase != safe_phase:
                    safe_phase = starved_phase
                    self.override_count += 1

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

    def get_observations(self, emv_bias=None):
        """
        emv_bias: optional dict[tls_id -> float], from `compute_emv_bias()`
        (Section 3.2, city-wide EMV routing). Appended as an 11th per-
        intersection feature, 0.0 for every intersection when omitted/off
        (`use_emv_routing=False`), so OBS_DIM stays fixed either way.

        Returns:
        tensor of shape (num_clusters, 11)
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
                    (emv_bias or {}).get(tl_id, 0.0),
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
                # length / max(mean_speed, 0.1) blows up as mean_speed -> the
                # 0.1 floor (a near-stopped lane can contribute thousands of
                # "seconds" from one lane alone), swamping queue_length/
                # waiting_time/pressure (O(1)-O(10)) and effectively zeroing
                # out R_fair/R_emergency's influence despite w1/w2/w3 --
                # see .claude/reasoning/training-loss/reward-component-dominance.md.
                # Capped at 5x free-flow time: still the harshest term for a
                # genuinely stalled lane, but bounded to the same order of
                # magnitude as the other components instead of unbounded.
                actual_time = length / max(mean_speed, 0.1)
                delay_time += min(max(actual_time - ideal_time, 0.0), 5.0 * ideal_time)

        speed_score /= len(lanes) if len(lanes) > 0 else 1

        final = -(queue_length + waiting_time + delay_time + pressure - speed_score)
        return final

    def total_waiting_time(self, tls):
        return sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls))

    def total_queue_length(self, tls):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls))

    def _get_incoming_lanes(self, tls_id):
        if tls_id in self._incoming_lanes_cache:
            return self._incoming_lanes_cache[tls_id]

        incoming_lanes = set()
        controlled_links = traci.trafficlight.getControlledLinks(tls_id)

        for link_group in controlled_links:
            for in_lane, _, _ in link_group:
                incoming_lanes.add(in_lane)

        self._incoming_lanes_cache[tls_id] = sorted(incoming_lanes)
        return self._incoming_lanes_cache[tls_id]

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
