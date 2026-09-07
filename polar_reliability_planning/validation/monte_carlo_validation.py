"""Holdout scenarios, never used to construct training cuts."""
from __future__ import annotations

import gurobipy as gp
import numpy as np

from ..config import ReliabilityLimits, SolverOptions
from ..data import CaseData
from ..reliability import ReliabilityOracle
from ..reliability.operation_model import OperationModel, OptimizationError
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


def audit_physical_dispatch(data: CaseData, units: dict[str, int], scenario: Scenario, dispatch: dict) -> dict:
    """Recompute balance, storage dynamics and bounds from the exported values."""
    power_names = ("wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw", "shed_kw")
    if any(np.shape(dispatch[k]) != (data.hours,) or not np.isfinite(dispatch[k]).all() for k in power_names):
        return {"passed": False, "error": "Invalid power array"}
    energy = dispatch["usable_energy_kwh"]
    if np.shape(energy) != (data.hours + 1,) or not np.isfinite(energy).all():
        return {"passed": False, "error": "Invalid energy array"}
    wind, pv, diesel, charge, discharge, shed = (dispatch[k] for k in power_names)
    capacity = data.capacities(units)
    balance = wind + pv + diesel + discharge + shed - scenario.load_kw - charge
    transition = energy[1:] - energy[:-1] - data.dt_hours * (data.efficiency * charge - discharge / data.efficiency)
    bounds = [0.0, float(-energy.min()), float(energy.max() -
        (data.soc_max - data.soc_min) * capacity["battery_energy"])]
    bounds.extend(float(-dispatch[k].min()) for k in power_names)
    bounds.extend(float(np.max(value)) for value in (wind - scenario.wind_available_kw,
        pv - scenario.pv_available_kw, diesel - scenario.diesel_available_kw,
        charge + discharge - scenario.pcs_available_kw, shed - scenario.load_kw))
    result = {"power_balance_max_abs_kw": float(np.max(np.abs(balance))),
              "energy_transition_max_abs_kwh": float(np.max(np.abs(transition))),
              "cyclic_energy_abs_kwh": float(abs(energy[-1] - energy[0])),
              "bound_max_violation": max(bounds)}
    return {"passed": max(result.values()) <= 1e-5, **result}


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
