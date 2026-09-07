"""A reproducible 48-point system with explicit, shared contingency trajectories."""
from __future__ import annotations

from itertools import product
import time

import gurobipy as gp
import numpy as np

from ..config import ReliabilityLimits, SolverOptions, UnitCommitmentOptions
from ..data import CaseData, COMPONENTS
from ..data.cap_plan_loader import FAILABLE_COMPONENTS
from ..planning.master_milp import MasterMILP, MasterInfeasible
from ..planning.optimizer import optimize_reliability
from ..reliability import ReliabilityOracle
from ..scenario_generation import ScenarioPool


def small_system(unit_commitment: bool = False) -> tuple[CaseData, ScenarioPool, ReliabilityLimits]:
    sizes = dict(zip(COMPONENTS, (2.0, 3.0, 4.0, 4.0, 2.0)))
    bounds = {k: (0, 2 if k == "diesel" else 1) for k in COMPONENTS}
    period_cost = dict(zip(COMPONENTS, (1.5, 1.2, 0.8, 0.5, 0.3)))
    data = CaseData("small_48_point_grid", np.array([3, 4, 3, 4]), np.array([0.5, 1, 0, 0.5]),
                    np.array([0, 0, 1, 0]), sizes, bounds,
                    {k: v * 8760 / 4 for k, v in period_cost.items()}, 1.0, 0.9, 0.0, 1.0,
                    unit_commitment=UnitCommitmentOptions(enabled=unit_commitment))
    availability = {k: np.ones((3, bounds[k][1], data.hours), dtype=np.uint8) for k in FAILABLE_COMPONENTS}
    availability["diesel"][1, 0, 1:3] = 0
    availability["diesel"][2, :, 2] = 0
    one = np.ones((3, data.hours))
    pool = ScenarioPool(availability, np.zeros((3, data.hours), dtype=np.uint8),
                        np.array([0.6, 0.3, 0.1]), one, one, one,
                        {"description": "exact three-contingency test distribution", "dt_hours": 1.0})
    return data, pool, ReliabilityLimits(eens_kwh=0.05, cvar_kwh=0.5)


def enumerate_grid(data: CaseData, oracle: ReliabilityOracle, options: SolverOptions,
                   env: gp.Env | None = None, max_points: int = 10000) -> dict:
    size = int(np.prod([hi - lo + 1 for lo, hi in data.unit_bounds.values()]))
    if size > max_points:
        raise ValueError(f"Grid has {size} points, exceeding enumeration limit {max_points}")
    start, rows, best = time.perf_counter(), [], None
    master = MasterMILP(data, options, env=env)
    try:
        for values in product(*(range(data.unit_bounds[k][0], data.unit_bounds[k][1] + 1) for k in COMPONENTS)):
            units = dict(zip(COMPONENTS, values))
            master.fix_units(units)
            try:
                solution = master.solve()
            except MasterInfeasible:
                rows.append({"units": units, "nominal_feasible": False, "reliability_feasible": None})
                continue
            reliability = oracle.evaluate(units)
            rows.append({"units": units, "nominal_feasible": True, "reliability_feasible": reliability.feasible,
                         "objective_yuan": solution.objective_yuan, "eens_kwh": reliability.eens_kwh,
                         "cvar_kwh": reliability.cvar_kwh})
            if reliability.feasible and (best is None or solution.objective_yuan < best.objective_yuan - 1e-7):
                best = solution
    finally:
        master.close()
    return {"grid_points": size, "best": best.summary() if best else None, "rows": rows,
            "oracle_evaluations": len(oracle.cache), "elapsed_seconds": time.perf_counter() - start}


def validate_small_system(env: gp.Env | None = None, on_iteration=None, unit_commitment: bool = True) -> dict:
    data, pool, limits = small_system(unit_commitment)
    options = SolverOptions(mip_gap=0, time_limit=30)
    search_oracle = ReliabilityOracle(data, pool, limits, options, env)
    grid_oracle = ReliabilityOracle(data, pool, limits, options, env)
    try:
        search = optimize_reliability(data, search_oracle, options, max_iterations=48,
                                      on_iteration=on_iteration, env=env)
        grid = enumerate_grid(data, grid_oracle, options, env)
        if search.solution is None or grid["best"] is None:
            raise AssertionError("The small test system must have a feasible solution")
        difference = abs(search.solution.objective_yuan - grid["best"]["objective_yuan"])
        if difference > 1e-6:
            raise AssertionError(f"Boundary search and exhaustive optimum differ by {difference}")
        feasible_points = [r["units"] for r in grid["rows"] if r["reliability_feasible"]]
        invalid = [cut.as_dict() for cut in search.cuts if any(cut.excludes(point) for point in feasible_points)]
        if invalid:
            raise AssertionError(f"Cuts excluded a feasible grid point: {invalid}")
        return {"passed": True, "unit_commitment": unit_commitment, "eens_limit_kwh": limits.eens_kwh,
                "cvar_limit_kwh": limits.cvar_kwh, "objective_difference_yuan": difference, "grid": grid,
                "boundary": {"status": search.status, "solution": search.solution.summary(),
                             "reliability": search.reliability.summary(), "iterations": len(search.history),
                             "cuts": [c.as_dict() for c in search.cuts], "history": search.history,
                             "oracle_evaluations": len(search_oracle.cache), "elapsed_seconds": search.elapsed_seconds},
                "feasible_points_wrongly_excluded": 0, "scope": "exact finite scenario distribution and 48-point capacity grid"}
    finally:
        search_oracle.close()
        grid_oracle.close()
