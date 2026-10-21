"""Non-learning baseline controllers for Table 2 (Section 5.2): Fixed-Time and
Max-Pressure. Both drive `TraciService.set_phase` directly, the same call the
trained policy uses, so they inherit the same min-green/yellow/EMV/starvation
safety layer (Section 3.9) as the learned controller.
"""

import traci


class FixedTimeController:
    """Cyclic, predefined signal-phase schedule with a fixed green/yellow split
    and no responsiveness to real-time traffic conditions (Section 5.2)."""

    def __init__(self, traci_service, phase_duration=20):
        self.traci_service = traci_service
        self.phase_duration = phase_duration
        self._cycle_pos = {tls: 0 for tls in traci_service.get_all_intersections()}
        self._phase_idx = {tls: 0 for tls in traci_service.get_all_intersections()}

    def step(self):
        for tls in self.traci_service.get_all_intersections():
            valid = self.traci_service.valid_phases[tls]
            self._cycle_pos[tls] += 1
            if self._cycle_pos[tls] >= self.phase_duration:
                self._cycle_pos[tls] = 0
                self._phase_idx[tls] = (self._phase_idx[tls] + 1) % len(valid)
            self.traci_service.set_phase(tls, self._phase_idx[tls])


class MaxPressureController:
    """Reactive controller that greedily selects the phase whose served
    incoming lanes carry the most queued vehicles (Varaiya 2013, cited in the
    paper as [5]). `phase_served_lanes` only tracks each phase's incoming
    lanes (not their downstream lanes), so this is a documented max-queue-first
    approximation of true incoming-minus-outgoing pressure, not the exact
    formulation — consistent with the other approximations already noted
    elsewhere in this codebase (e.g. `get_intersection_reward`'s delay_time)."""

    def __init__(self, traci_service):
        self.traci_service = traci_service

    def step(self):
        for tls in self.traci_service.get_all_intersections():
            valid = self.traci_service.valid_phases[tls]
            self.traci_service.set_phase(tls, self._best_phase(tls, valid))

    def _best_phase(self, tls_id, valid_phases):
        best_idx, best_pressure = 0, float("-inf")
        for i, phase in enumerate(valid_phases):
            lanes = self.traci_service.phase_served_lanes[tls_id].get(phase, ())
            pressure = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            if pressure > best_pressure:
                best_pressure, best_idx = pressure, i
        return best_idx
