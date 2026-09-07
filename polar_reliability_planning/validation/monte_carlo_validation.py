"""Holdout scenarios, never used to construct training cuts."""
from __future__ import annotations

import gurobipy as gp

from ..config import ReliabilityLimits, SolverOptions
from ..data import CaseData
from ..reliability import ReliabilityOracle
from ..reliability.operation_model import OperationModel
from ..reliability.unit_commitment import audit_commitment
from ..scenario_generation import Scenario, ScenarioPool


def validate_capacity(data: CaseData, units: dict[str, int], pool: ScenarioPool,
                       limits: ReliabilityLimits, options: SolverOptions, env: gp.Env | None = None) -> dict:
    oracle = ReliabilityOracle(data, pool, limits, options, env)
    try:
        result = oracle.evaluate(units)
        return {**result.summary(), "losses_kwh": list(result.losses_kwh), "sample_role": "independent_holdout",
                "interpretation": "empirical check, not a confidence-certified population reliability claim"}
    finally:
        oracle.close()


def audit_nominal_solution(data: CaseData, solution, options: SolverOptions,
                            env: gp.Env | None = None) -> dict:
    operation = OperationModel(data, options, env)
    try:
        nominal = Scenario.nominal(data, solution.units)
        commitment = audit_commitment(data, solution.units, nominal, solution.dispatch)
        eens, _ = operation.solve(solution.units, nominal)
        fuel, _ = operation.solve(solution.units, nominal, objective="nominal_cost")
        total = fuel + solution.investment_yuan
        difference = abs(total - solution.objective_yuan)
        # A limited master incumbent can have a suboptimal dispatch: report that fact.
        return {"no_failure_eens_kwh": eens, "fixed_capacity_optimal_cost_yuan": total,
                "master_cost_difference_yuan": difference,
                "no_shedding_passed": eens <= 1e-5, "unit_commitment": commitment,
                "dispatch_cost_consistent": difference <= max(1e-5, abs(total) * 1e-7)}
    finally:
        operation.close()
