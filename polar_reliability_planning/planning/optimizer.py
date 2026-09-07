"""Solve master -> evaluate fixed Omega -> add a monotonic cut -> replan."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import gurobipy as gp

from ..config import SolverOptions
from ..data import CaseData, COMPONENTS
from ..reliability.monte_carlo import ReliabilityOracle, ReliabilityResult
from .master_milp import MasterMILP, MasterSolution, MasterInfeasible
from .reliability_cuts import ReliabilityCut


@dataclass
class PlanningResult:
    status: str
    solution: MasterSolution | None
    reliability: ReliabilityResult | None
    history: list[dict]
    cuts: list[ReliabilityCut]
    lower_bound_yuan: float | None
    elapsed_seconds: float


def lift_failed_point(units: dict[str, int], data: CaseData, oracle: ReliabilityOracle) -> tuple[dict, ReliabilityResult]:
    """Optional safe strengthening: every accepted lifted point is evaluated."""
    point = dict(units)
    result = oracle.evaluate(point, certify_infeasible_early=True)
    if result.feasible:
        raise ValueError("Cannot lift a feasible point into a reliability cut")
    for key in COMPONENTS:
        lo, hi = point[key], data.unit_bounds[key][1]
        while lo < hi:
            middle = (lo + hi + 1) // 2
            probe = {**point, key: middle}
            evaluated = oracle.certify_failure(probe)
            if evaluated is None:
                hi = middle - 1
            else:
                lo = middle
                point, result = probe, evaluated
    return point, result


def optimize_reliability(data: CaseData, oracle: ReliabilityOracle, options: SolverOptions | None = None,
                         max_iterations: int = 100, lift_cuts: bool = False,
                         on_iteration: Callable[[dict], None] | None = None,
                         env: gp.Env | None = None) -> PlanningResult:
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    start = time.perf_counter()
    master = MasterMILP(data, options, env=env)
    history, lower_bound = [], None
    try:
        for iteration in range(1, max_iterations + 1):
            try:
                candidate = master.solve()
            except MasterInfeasible:
                return PlanningResult("sample_infeasible", None, None, history, list(master.cuts), lower_bound,
                                       time.perf_counter() - start)
            lower_bound = max(lower_bound if lower_bound is not None else candidate.lower_bound_yuan, candidate.lower_bound_yuan)
            reliability = oracle.evaluate(candidate.units, certify_infeasible_early=True)
            row = {"iteration": iteration, "units": candidate.units, "objective_yuan": candidate.objective_yuan,
                   "lower_bound_yuan": lower_bound, "master_gap": candidate.gap,
                   "master_seconds": candidate.runtime_seconds, **reliability.summary()}
            history.append(row)
            if reliability.feasible:
                if on_iteration:
                    on_iteration(row)
                gap = max(0.0, candidate.objective_yuan - lower_bound) / max(abs(candidate.objective_yuan), 1e-10)
                status = "sample_optimal_within_gap" if gap <= master.options.mip_gap + 1e-9 else "sample_feasible"
                return PlanningResult(status, candidate, reliability, history, list(master.cuts), lower_bound,
                                       time.perf_counter() - start)
            bad, evaluated = (lift_failed_point(candidate.units, data, oracle) if lift_cuts
                               else (candidate.units, reliability))
            cut = ReliabilityCut(tuple(bad[k] for k in COMPONENTS), oracle.pool.fingerprint,
                                  evaluated.eens_kwh, evaluated.cvar_kwh, evaluated.metrics_are_lower_bounds,
                                  sum(q is not None for q in evaluated.losses_kwh), evaluated.losses_are_relaxation_bounds)
            row["cut"] = cut.as_dict()
            master.add_cut(cut)
            if on_iteration:
                on_iteration(row)
        return PlanningResult("iteration_limit", None, None, history, list(master.cuts), lower_bound,
                               time.perf_counter() - start)
    finally:
        master.close()
