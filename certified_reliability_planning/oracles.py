"""Bound-valued dispatch solves; stopping early never creates an exact loss.

The physical formulation and independent audits are imported unchanged from the
legacy package. Solver certificates are numerical (Gurobi feasibility tolerance
1e-8 and the legacy audit tolerance 1e-5), not rational-arithmetic proofs. Both
ends are rounded outwards by ``numerical_margin``. That guard is deliberately
retained even after Gurobi reports OPTIMAL; a positive loss is never zeroed by
comparison with a tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.config import SolverOptions
from polar_reliability_planning.data import CaseData
from polar_reliability_planning.reliability.dispatch_audit import audit_physical_dispatch
from polar_reliability_planning.reliability.operation_model import (
    add_dispatch, configure_model, dispatch_values, dispatch_without_storage,
)
from polar_reliability_planning.reliability.unit_commitment import (
    audit_commitment, available_diesel_modules, standby_commitment,
)
from polar_reliability_planning.scenario_generation import Scenario


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    work: float = 0.0
    runtime_seconds: float = 0.0
    status: str = "unresolved"
    dispatch: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("Invalid optimization interval")
        if self.lower < 0 or not math.isfinite(self.work) or self.work < 0:
            raise ValueError("Bounds and work must be nonnegative")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("Runtime must be finite and nonnegative")

    @property
    def width(self) -> float:
        return 0.0 if self.lower == self.upper else self.upper - self.lower

    @property
    def gap(self) -> float:
        return self.width


def numerical_margin(value: float, accumulation_scale: float) -> float:
    """Conservative floating-point guard, separate from requested MIP gap.

    This is a declared numerical convention, not a verified error bound for an
    arbitrary ill-conditioned MILP. Audit failures discard an incumbent.
    """
    return 1e-7 * max(1.0, abs(value), accumulation_scale)


_STATUS = {
    GRB.OPTIMAL: "optimal_within_solver_tolerance",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "infeasible_or_unbounded",
    GRB.UNBOUNDED: "unbounded",
    GRB.TIME_LIMIT: "time_limit",
    GRB.NODE_LIMIT: "node_limit",
    GRB.ITERATION_LIMIT: "iteration_limit",
    GRB.SOLUTION_LIMIT: "solution_limit",
    GRB.INTERRUPTED: "interrupted",
    GRB.SUBOPTIMAL: "suboptimal",
    GRB.NUMERIC: "numerical_failure",
}


class MicrogridOracle:
    """Independent refinable UC and nominal economic optimization intervals.

    Each request builds a fresh model, so one limited solve cannot accidentally
    leave fixed commitments or bounds in the next request. The caller intersects
    repeated intervals and accounts for the incremental ``work`` returned here.
    The economic objective includes period investment, fuel, and startup costs.
    """

    def __init__(self, data: CaseData, options: SolverOptions | None = None,
                 env: gp.Env | None = None):
        self.data = data
        self.options = options or SolverOptions()
        self.env = env
        self.solve_count = 0
        self.last_audit: dict | None = None

    def close(self):
        """Models are disposed after each solve; the caller owns its environment."""

    def operation(self, units: dict[str, int], scenario: Scenario,
                  budget_seconds: float, absolute_gap: float) -> Interval:
        return self._solve(units, scenario, budget_seconds, absolute_gap, economic=False)

    def economic(self, units: dict[str, int], budget_seconds: float,
                 absolute_gap: float) -> Interval:
        units = self.data.validate_units(units)
        return self._solve(units, Scenario.nominal(self.data, units),
                           budget_seconds, absolute_gap, economic=True)

    def _audit(self, units, scenario, dispatch, economic: bool) -> bool:
        try:
            physical = audit_physical_dispatch(self.data, units, scenario, dispatch)
            commitment = audit_commitment(self.data, units, scenario, dispatch)
            # No-shedding is an additional economic constraint: the physical
            # audit also serves reliability solves and therefore permits shed.
            no_shedding = not economic or bool(np.all(dispatch["shed_kw"] == 0.0))
            self.last_audit = {"physical": physical, "commitment": commitment,
                               "no_shedding": no_shedding}
            return physical["passed"] and commitment["passed"] and no_shedding
        except (KeyError, ValueError, TypeError, FloatingPointError) as error:
            self.last_audit = {"passed": False, "error": str(error)}
            return False

    def _all_shed(self, units, scenario):
        data = self.data
        dispatch = {name: np.zeros(data.hours) for name in (
            "wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw", "battery_charge_mode")}
        dispatch["shed_kw"] = np.array(scenario.load_kw, copy=True)
        dispatch["usable_energy_kwh"] = np.zeros(data.hours + 1)
        dispatch["stored_energy_kwh"] = np.full(data.hours + 1,
            data.soc_min * data.capacities(units)["battery_energy"])
        if data.unit_commitment.enabled:
            shape = (data.unit_bounds["diesel"][1], data.hours)
            for name in ("online", "startup", "shutdown"):
                dispatch[f"diesel_unit_{name}"] = np.zeros(shape)
                dispatch[f"diesel_{name}_units"] = np.zeros(data.hours)
        return dispatch

    def _dispatch_cost(self, units, dispatch):
        data = self.data
        capital = sum(data.period_cost_per_unit[k] * n for k, n in units.items())
        fuel = data.dt_hours * data.fuel_cost_per_kwh * float(dispatch["diesel_kw"].sum())
        startups = float(dispatch.get("diesel_startup_units", np.zeros(1)).sum())
        return capital + fuel + data.unit_commitment.startup_cost_yuan * startups

    def _solve(self, units, scenario, budget_seconds, absolute_gap, economic):
        start = time.perf_counter()
        if not math.isfinite(budget_seconds) or budget_seconds < 0:
            raise ValueError("budget_seconds must be finite and nonnegative")
        if not math.isfinite(absolute_gap) or absolute_gap < 0:
            raise ValueError("absolute_gap must be finite and nonnegative")
        data = self.data
        units = data.validate_units(units)
        capacity = data.capacities(units)
        for name in ("wind_available_kw", "pv_available_kw", "diesel_available_kw",
                     "pcs_available_kw", "load_kw"):
            values = np.asarray(getattr(scenario, name))
            if values.shape != (data.hours,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid scenario series: {name}")
        available = available_diesel_modules(data, units, scenario) if data.unit_commitment.enabled else None
        capital = sum(data.period_cost_per_unit[k] * n for k, n in units.items()) if economic else 0.0
        lower = capital
        upper = math.inf if economic else float(np.sum(scenario.load_kw) * data.dt_hours)
        best = None if economic else self._all_shed(units, scenario)
        if best is not None and not self._audit(units, scenario, best, economic=False):
            raise ValueError("All-shed dispatch failed its physical/commitment audit")
        # This constructive witness never uses a small numerical objective as
        # proof of zero: the helper sets shedding and storage identically zero.
        policy = None if available is None else standby_commitment(
            available, max(1, math.ceil(data.unit_commitment.min_down_hours / data.dt_hours)),
            max(1, math.ceil(float(np.max(scenario.load_kw)) / data.module_sizes["diesel"])))
        direct = dispatch_without_storage(data, units, scenario, policy)
        if direct is not None and self._audit(units, scenario, direct, economic):
            if not economic:
                return Interval(0.0, 0.0, runtime_seconds=time.perf_counter() - start,
                                status="constructed_zero_loss", dispatch=direct)
            upper = self._dispatch_cost(units, direct)
            upper = float(np.nextafter(upper + numerical_margin(upper, data.hours), math.inf))
            best = direct
        if budget_seconds == 0 or time.perf_counter() - start >= budget_seconds:
            return Interval(lower, upper, runtime_seconds=time.perf_counter() - start,
                            status="no_budget", dispatch=best)

        with gp.Model("certified_nominal_cost" if economic else "certified_uc_loss", env=self.env) as model:
            configure_model(model, self.options, oracle=True)
            model.Params.MIPGap = self.options.mip_gap if economic else 0.0
            model.Params.MIPGapAbs = absolute_gap
            # Every objective is nonnegative and all-shed operation is feasible;
            # DualReductions=0 disambiguates nominal infeasibility in one solve.
            model.Params.DualReductions = 0
            block = add_dispatch(model, data, scenario.wind_available_kw, scenario.pv_available_kw,
                scenario.diesel_available_kw, scenario.pcs_available_kw,
                (data.soc_max - data.soc_min) * capacity["battery_energy"],
                scenario.load_kw, not economic)
            if block.commitment is not None:
                block.commitment.update_availability(available)
            if economic:
                objective = capital + data.dt_hours * data.fuel_cost_per_kwh * block.variables["diesel_kw"].sum()
                if block.commitment is not None:
                    objective += data.unit_commitment.startup_cost_yuan * block.commitment.startup.sum()
            else:
                objective = data.dt_hours * block.variables["shed_kw"].sum()
            model.setObjective(objective, GRB.MINIMIZE)
            if best is not None:
                for name, variable in block.variables.items():
                    if name in best:
                        variable.Start = best[name]
                if block.commitment is not None:
                    for name in ("online", "startup", "shutdown"):
                        getattr(block.commitment, name).Start = best[f"diesel_unit_{name}"]
            remaining = budget_seconds - (time.perf_counter() - start)
            if remaining <= 0:
                return Interval(lower, upper, runtime_seconds=time.perf_counter() - start,
                                status="model_build_budget_exhausted", dispatch=best)
            model.Params.TimeLimit = remaining
            model.optimize()
            self.solve_count += 1
            work = float(model.Work)
            status = _STATUS.get(model.Status, f"gurobi_status_{model.Status}")
            if model.Status == GRB.INFEASIBLE and economic:
                return Interval(math.inf, math.inf, work, time.perf_counter() - start, "nominal_infeasible")
            if model.Status in (GRB.NUMERIC, GRB.INF_OR_UNBD, GRB.UNBOUNDED, GRB.INFEASIBLE):
                return Interval(lower, upper, work, time.perf_counter() - start, status, best)
            try:
                bound = float(model.ObjBound) if model.IsMIP else (
                    float(model.ObjVal) if model.Status == GRB.OPTIMAL else -math.inf)
            except (gp.GurobiError, AttributeError):
                bound = -math.inf
            scale = data.hours * data.dt_hours * (max(1.0, data.fuel_cost_per_kwh) if economic else 1.0)
            if math.isfinite(bound):
                lower = max(lower, float(np.nextafter(bound - numerical_margin(bound, scale), -math.inf)))
            if model.SolCount:
                dispatch = dispatch_values(block, data, capacity["battery_energy"])
                if self._audit(units, scenario, dispatch, economic):
                    value = self._dispatch_cost(units, dispatch) if economic else float(
                        data.dt_hours * np.maximum(dispatch["shed_kw"], 0.0).sum())
                    candidate = float(np.nextafter(value + numerical_margin(value, scale), math.inf))
                    if candidate < upper:
                        upper, best = candidate, dispatch
                else:
                    status += ":incumbent_audit_failed"
            # Inconsistent numerical evidence is unusable, so retain only the
            # universal lower bound and an independently audited feasible upper.
            if lower > upper:
                lower = capital
                status += ":inconsistent_bound_discarded"
            return Interval(lower, upper, work, time.perf_counter() - start, status, best)
