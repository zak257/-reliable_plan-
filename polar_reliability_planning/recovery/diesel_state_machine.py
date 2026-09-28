"""Boundary states. Hidden repair completion is never passed to policy code."""
from dataclasses import dataclass


@dataclass
class DieselState:
    healthy: bool = True
    running: bool = False
    pending: bool = False
    onsite: bool = False
    prep: float = 0.0
    min_up: int = 0
    min_down: int = 0
    repair_until: int = -1
    runtime: int = 0
    demands: int = 0

    def observable(self):
        return (self.healthy, self.running, self.pending, self.onsite,
                self.prep, self.min_up, self.min_down)

    def label(self, temperature, ready, prep_hours):
        if not self.healthy:
            return "FAILED"
        if self.running:
            return "RUNNING"
        if not self.pending:
            return "STANDBY"
        if not self.onsite:
            return "WAIT_ACCESS"
        if self.prep < prep_hours:
            return "PREPARING"
        if temperature < ready - 1e-9:
            return "WAIT_HEAT"
        return "READY"
