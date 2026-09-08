"""Holdout scenarios, never used to construct training cuts."""
from __future__ import annotations

import gurobipy as gp

from ..config import ReliabilityLimits, SolverOptions
from ..data import CaseData
from ..reliability import ReliabilityOracle
from ..reliability.operation_model import OperationModel, OptimizationError
from ..reliability.unit_commitment import audit_commitment
from ..reliability.dispatch_audit import audit_physical_dispatch
from ..scenario_generation import Scenario, ScenarioPool


def validate_capacity(data: CaseData, units: dict[str, int], pool: ScenarioPool,
                       limits: ReliabilityLimits, options: SolverOptions, env: gp.Env | None = None,
                       on_progress=None, on_scenario=None) -> dict:
    oracle = ReliabilityOracle(data, pool, limits, options, env, on_progress, on_scenario)
    try:
        result = oracle.evaluate(units)
        return {**result.summary(), "losses_kwh": list(result.losses_kwh), "sample_role": "independent_holdout",
                "solver_calls": oracle.operation.solver_calls,
                "full_mip_solves": oracle.operation.full_mip_solves,
                "direct_zero_loss_certificates": oracle.operation.direct_zero_loss_certificates,
                "storage_zero_loss_certificates": oracle.operation.storage_zero_loss_certificates,
                "interpretation": "empirical check, not a confidence-certified population reliability claim"}
    finally:
        oracle.close()


def audit_nominal_solution(data: CaseData, solution, options: SolverOptions,
                            env: gp.Env | None = None) -> dict:
    operation = OperationModel(data, options, env)
    try:
        nominal = Scenario.nominal(data, solution.units)
        commitment = audit_commitment(data, solution.units, nominal, solution.dispatch)
        physical = audit_physical_dispatch(data, solution.units, nominal, solution.dispatch)
        if not commitment["passed"] or not physical["passed"]:
            raise OptimizationError(f"Nominal dispatch failed audit: UC={commitment}, physical={physical}")
        eens, _ = operation.solve(solution.units, nominal)
        fuel, _ = operation.solve(solution.units, nominal, objective="nominal_cost", warm_start=solution.dispatch)
        economic_solve = operation.last_solve_summary
        total = fuel + solution.investment_yuan
        difference = abs(total - solution.objective_yuan)
        # A limited master incumbent can have a suboptimal dispatch: report that fact.
        return {"no_failure_eens_kwh": eens,
                "fixed_capacity_optimal_cost_yuan": total if economic_solve["optimal"] else None,
                "fixed_capacity_feasible_cost_yuan": total,
                "fixed_capacity_cost_lower_bound_yuan": (economic_solve["lower_bound"] + solution.investment_yuan
                    if economic_solve["lower_bound"] is not None else None),
                "economic_solve": economic_solve,
                "master_cost_difference_yuan": difference,
                "no_shedding_passed": eens <= 1e-5, "unit_commitment": commitment,
                "physical_dispatch": physical,
                "dispatch_cost_consistent": difference <= max(1e-5, abs(total) * 1e-7)}
    finally:
        operation.close()
