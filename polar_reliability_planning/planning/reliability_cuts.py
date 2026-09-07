"""Exclude a failed point's complete lower orthant on the five-dimensional grid."""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from ..data import COMPONENTS


@dataclass(frozen=True)
class ReliabilityCut:
    bad_units: tuple[int, ...]
    sample_fingerprint: str
    eens_kwh: float
    cvar_kwh: float
    metrics_are_lower_bounds: bool = False
    evaluated_scenarios: int | None = None
    losses_are_relaxation_bounds: bool = False

    @property
    def point(self) -> dict[str, int]:
        return dict(zip(COMPONENTS, self.bad_units))

    def excludes(self, units: dict[str, int]) -> bool:
        return all(units[k] <= self.point[k] for k in COMPONENTS)

    def as_dict(self) -> dict:
        return {"bad_units": self.point, "sample_fingerprint": self.sample_fingerprint,
                "eens_kwh": self.eens_kwh, "cvar_kwh": self.cvar_kwh,
                "metrics_are_lower_bounds": self.metrics_are_lower_bounds,
                "evaluated_scenarios": self.evaluated_scenarios,
                "losses_are_relaxation_bounds": self.losses_are_relaxation_bounds}


def add_reliability_cut(model: gp.Model, unit_vars: dict[str, gp.Var], bounds: dict,
                        cut: ReliabilityCut, index: int):
    choices = []
    for key, bad in cut.point.items():
        if bad < bounds[key][0] or bad > bounds[key][1]:
            raise ValueError(f"Cut point outside bounds: {key}={bad}")
        if bad == bounds[key][1]:
            continue
        increase = model.addVar(vtype=GRB.BINARY, name=f"cut_{index}_{key}_increase")
        # A one-way indicator is sufficient for an exact existential disjunction.
        model.addGenConstrIndicator(increase, True, unit_vars[key] >= bad + 1,
                                    name=f"cut_{index}_{key}_threshold")
        choices.append(increase)
    # An empty sum >= 1 correctly proves infeasibility at the all-max corner.
    model.addConstr(gp.quicksum(choices) >= 1, name=f"reliability_cut_{index}")
