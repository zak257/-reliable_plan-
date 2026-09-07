"""Nominal chronological MILP, incrementally augmented by reliability cuts."""
from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from ..config import SolverOptions
from ..data import CaseData, COMPONENTS
from ..reliability.operation_model import add_dispatch, configure_model, dispatch_values, OptimizationError
from .reliability_cuts import ReliabilityCut, add_reliability_cut


class MasterInfeasible(OptimizationError):
    pass


@dataclass
class MasterSolution:
    units: dict[str, int]
    capacities: dict[str, float]
    objective_yuan: float
    investment_yuan: float
    fuel_yuan: float
    startup_yuan: float
    lower_bound_yuan: float
    gap: float
    status: int
    runtime_seconds: float
    dispatch: dict[str, np.ndarray]

    def summary(self) -> dict:
        return {k: v for k, v in vars(self).items() if k != "dispatch"}


class MasterMILP:
    def __init__(self, data: CaseData, options: SolverOptions | None = None,
                 cuts: list[ReliabilityCut] | None = None, env: gp.Env | None = None):
        self.data = data
        self.options = options or SolverOptions()
        self.model = gp.Model("nominal_capacity_master", env=env)
        configure_model(self.model, self.options)
        self.n = self.model.addMVar(len(COMPONENTS), lb=[data.unit_bounds[k][0] for k in COMPONENTS],
                                   ub=[data.unit_bounds[k][1] for k in COMPONENTS], vtype=GRB.INTEGER, name="modules")
        self.unit_vars = {k: self.n[i].item() for i, k in enumerate(COMPONENTS)}
        capacity = {k: self.n[i] * data.module_sizes[k] for i, k in enumerate(COMPONENTS)}
        self.block = add_dispatch(self.model, data, data.wind_pu * capacity["wind"], data.pv_pu * capacity["pv"],
                                  capacity["diesel"], capacity["pcs"],
                                  (data.soc_max - data.soc_min) * capacity["battery_energy"], data.load_kw, False,
                                  self.n[COMPONENTS.index("diesel")])
        objective = np.array([data.period_cost_per_unit[k] for k in COMPONENTS]) @ self.n + data.dt_hours * data.fuel_cost_per_kwh * self.block.variables["diesel_kw"].sum()
        if self.block.commitment is not None:
            objective += data.unit_commitment.startup_cost_yuan * self.block.commitment.startup.sum()
        self.model.setObjective(objective, GRB.MINIMIZE)
        self.cuts: list[ReliabilityCut] = []
        for cut in cuts or []:
            self.add_cut(cut)
        self.model.update()

    def add_cut(self, cut: ReliabilityCut):
        if self.cuts and cut.sample_fingerprint != self.cuts[0].sample_fingerprint:
            raise ValueError("Cannot mix cuts from different scenario sets")
        add_reliability_cut(self.model, self.unit_vars, self.data.unit_bounds, cut, len(self.cuts))
        self.cuts.append(cut)

    def fix_units(self, units: dict[str, int]):
        units = self.data.validate_units(units)
        for key, value in units.items():
            self.unit_vars[key].LB = value
            self.unit_vars[key].UB = value

    def solve(self) -> MasterSolution:
        self.model.optimize()
        if self.model.Status == GRB.INF_OR_UNBD:
            self.model.Params.DualReductions = 0
            self.model.optimize()
        if self.model.Status == GRB.INFEASIBLE:
            raise MasterInfeasible("Nominal master plus reliability cuts is infeasible on the declared capacity grid")
        if self.model.SolCount < 1 or self.model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.INTERRUPTED):
            raise OptimizationError(f"Master has no usable incumbent; Gurobi status={self.model.Status}")
        units = self.data.validate_units(dict(zip(COMPONENTS, self.n.X)))
        capacities = self.data.capacities(units)
        investment = sum(self.data.period_cost_per_unit[k] * units[k] for k in COMPONENTS)
        dispatch = dispatch_values(self.block, self.data, capacities["battery_energy"])
        fuel = float(dispatch["diesel_kw"].sum() * self.data.dt_hours * self.data.fuel_cost_per_kwh)
        startup = self.data.unit_commitment.startup_cost_yuan * float(dispatch.get("diesel_startup_units", np.zeros(1)).sum())
        return MasterSolution(units, capacities, float(self.model.ObjVal), investment, fuel, startup,
                               float(self.model.ObjBound), float(self.model.MIPGap), int(self.model.Status),
                               float(self.model.Runtime), dispatch)

    def close(self):
        self.model.dispose()
