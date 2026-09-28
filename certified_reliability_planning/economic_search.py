"""Nominal MILP bound over the capacity region remaining after risk failures.

The legacy master formulation is reused without modifying it.  Its only added
constraints exclude downsets with population-risk failure certificates supplied
by the planner.  Reliability-feasible points remain in this model, so its dual
bound is a valid economic lower bound for every remaining candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.config import SolverOptions
from polar_reliability_planning.data import CaseData, COMPONENTS
from polar_reliability_planning.planning.master_milp import MasterMILP
from polar_reliability_planning.planning.reliability_cuts import ReliabilityCut
from polar_reliability_planning.reliability.dispatch_audit import audit_physical_dispatch
from polar_reliability_planning.reliability.operation_model import dispatch_values
from polar_reliability_planning.reliability.unit_commitment import audit_commitment
from polar_reliability_planning.scenario_generation import Scenario

from .oracles import numerical_margin, _STATUS


@dataclass(frozen=True)
class RegionResult:
    lower: float
    point: tuple[int, ...] | None = None
    upper: float | None = None
    dispatch: dict[str, np.ndarray] | None = None
    work: float = 0.0
    runtime_seconds: float = 0.0
    status: str = "unresolved"
    infeasible: bool = False

    def __post_init__(self):
        if math.isnan(self.lower) or self.lower < 0:
            raise ValueError("Region lower bound must be nonnegative")
        if self.upper is not None and (not math.isfinite(self.upper) or self.upper < self.lower):
            raise ValueError("Region incumbent must have a finite consistent upper bound")
        if (self.point is None) != (self.upper is None):
            raise ValueError("Region point and incumbent upper bound must be provided together")
        if not math.isfinite(self.work) or self.work < 0:
            raise ValueError("Region work must be finite and nonnegative")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("Region runtime must be finite and nonnegative")
        if self.infeasible and (self.lower != math.inf or self.point is not None):
            raise ValueError("An infeasible region has infinite lower bound and no incumbent")


class EconomicSearch:
    """Refinable nominal bound over the complete grid minus failed downsets.

    ``solve(budget_seconds, absolute_gap)`` returns a ``RegionResult`` even if a
    limited solve has no incumbent.  The model is built lazily so construction
    is included in the first call's runtime. Every call sets the configured
    relative MIP gap and the supplied absolute gap; the planner controls its refinement
    schedule.  The caller owns an optional Gurobi environment.
    """

    def __init__(self, data: CaseData, options: SolverOptions | None = None,
                 env: gp.Env | None = None):
        self.data = data
        self.options = options or SolverOptions()
        self.env = env
        self._master: MasterMILP | None = None
        self._cuts: list[ReliabilityCut] = []
        self._lower = sum(data.period_cost_per_unit[k] * data.unit_bounds[k][0] for k in COMPONENTS)
        self._point: tuple[int, ...] | None = None
        self._upper: float | None = None
        self._dispatch: dict[str, np.ndarray] | None = None
        self._infeasible = False
        self._closed = False
        self.solve_count = 0
        self.last_audit: dict | None = None

    def add_failure(self, point: Sequence[int] | Mapping[str, int]) -> bool:
        """Exclude a certified risk failure's lower orthant; never a timeout.

        The caller must supply only points with a simultaneous population-risk
        lower-bound failure certificate.  Metadata in the reused legacy cut is
        deliberately not used as a risk estimate.  Returns False for a downset
        already contained in an existing cut.
        """
        if self._closed:
            raise RuntimeError("EconomicSearch is closed")
        if isinstance(point, Mapping):
            units = self.data.validate_units(point)
        else:
            values = tuple(point)
            if len(values) != len(COMPONENTS):
                raise ValueError("A risk failure point must contain every component")
            units = self.data.validate_units(dict(zip(COMPONENTS, values)))
        key = tuple(units[k] for k in COMPONENTS)
        if any(cut.excludes(units) for cut in self._cuts):
            return False
        cut = ReliabilityCut(key, "population_certified", 0.0, 0.0,
                             metrics_are_lower_bounds=True)
        self._cuts.append(cut)
        if self._master is not None:
            self._master.add_cut(cut)
        if self._point is not None and all(a <= b for a, b in zip(self._point, key)):
            self._point, self._upper, self._dispatch = None, None, None
        return True

    def _result(self, start, status, work=0.0):
        return RegionResult(self._lower, self._point, self._upper, self._dispatch,
                            work, time.perf_counter() - start, status, self._infeasible)

    def _audit(self, units, dispatch):
        try:
            scenario = Scenario.nominal(self.data, units)
            physical = audit_physical_dispatch(self.data, units, scenario, dispatch)
            commitment = audit_commitment(self.data, units, scenario, dispatch)
            no_shedding = bool(np.all(dispatch["shed_kw"] == 0.0))
            self.last_audit = {"physical": physical, "commitment": commitment,
                               "no_shedding": no_shedding}
            return physical["passed"] and commitment["passed"] and no_shedding
        except (KeyError, ValueError, TypeError, FloatingPointError) as error:
            self.last_audit = {"passed": False, "error": str(error)}
            return False

    def solve(self, budget_seconds: float, absolute_gap: float) -> RegionResult:
        if self._closed:
            raise RuntimeError("EconomicSearch is closed")
        if not math.isfinite(budget_seconds) or budget_seconds < 0:
            raise ValueError("budget_seconds must be finite and nonnegative")
        if not math.isfinite(absolute_gap) or absolute_gap < 0:
            raise ValueError("absolute_gap must be finite and nonnegative")
        start = time.perf_counter()
        if self._infeasible:
            return self._result(start, "nominal_region_infeasible")
        if budget_seconds == 0:
            return self._result(start, "no_budget")
        if self._master is None:
            self._master = MasterMILP(self.data, self.options, cuts=self._cuts, env=self.env)
        master = self._master
        model = master.model
        model.Params.MIPGap = self.options.mip_gap
        model.Params.MIPGapAbs = absolute_gap
        model.Params.DualReductions = 0
        remaining = budget_seconds - (time.perf_counter() - start)
        if remaining <= 0:
            return self._result(start, "model_build_budget_exhausted")
        model.Params.TimeLimit = remaining
        # MasterMILP.solve() raises if there is no incumbent, which would throw
        # away useful dual evidence from a time-limited region search.
        model.optimize()
        self.solve_count += 1
        work = float(model.Work)
        status = _STATUS.get(model.Status, f"gurobi_status_{model.Status}")
        if model.Status == GRB.INFEASIBLE:
            if self._point is not None:
                return self._result(start, "inconsistent_infeasibility_discarded", work)
            self._lower, self._infeasible = math.inf, True
            return self._result(start, "nominal_region_infeasible", work)
        if model.Status in (GRB.NUMERIC, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
            return self._result(start, status, work)

        prior_lower = self._lower
        scale = self.data.hours * self.data.dt_hours * max(1.0, self.data.fuel_cost_per_kwh)
        try:
            dual = float(model.ObjBound)
        except (gp.GurobiError, AttributeError):
            dual = -math.inf
        if math.isfinite(dual):
            guarded = float(np.nextafter(dual - numerical_margin(dual, scale), -math.inf))
            self._lower = max(self._lower, guarded)

        if model.SolCount:
            try:
                units = self.data.validate_units(dict(zip(COMPONENTS, master.n.X)))
                key = tuple(units[k] for k in COMPONENTS)
                dispatch = dispatch_values(master.block, self.data, self.data.capacities(units)["battery_energy"])
                allowed = not any(cut.excludes(units) for cut in self._cuts)
                if allowed and self._audit(units, dispatch):
                    investment = sum(self.data.period_cost_per_unit[k] * units[k] for k in COMPONENTS)
                    fuel = float(dispatch["diesel_kw"].sum() * self.data.dt_hours * self.data.fuel_cost_per_kwh)
                    startups = float(dispatch.get("diesel_startup_units", np.zeros(1)).sum())
                    cost = investment + fuel + self.data.unit_commitment.startup_cost_yuan * startups
                    upper = float(np.nextafter(cost + numerical_margin(cost, scale), math.inf))
                    if self._upper is None or upper < self._upper:
                        self._point, self._upper, self._dispatch = key, upper, dispatch
                else:
                    status += ":incumbent_audit_failed"
            except (gp.GurobiError, ValueError, AttributeError):
                status += ":incumbent_extraction_failed"
        if self._upper is not None and self._lower > self._upper:
            # A dual bound contradicted by an audited feasible dispatch cannot
            # be used to certify an economic gap.
            if prior_lower > self._upper:
                raise ValueError("Persistent region lower bound contradicts an audited incumbent")
            self._lower = prior_lower
            status += ":inconsistent_bound_discarded"
        return self._result(start, status, work)

    __call__ = solve

    def close(self):
        if self._master is not None:
            self._master.close()
            self._master = None
        self._closed = True
