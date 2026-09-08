"""Chronological dispatch with the same physical UC constraints in master and oracle."""
from __future__ import annotations

from dataclasses import dataclass
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from ..config import SolverOptions
from ..data import CaseData
from ..scenario_generation import Scenario
from .dispatch_audit import audit_physical_dispatch
from .unit_commitment import (CommitmentBlock, add_unit_commitment, available_diesel_modules,
                              standby_commitment, audit_commitment)


class OptimizationError(RuntimeError):
    pass


def configure_model(model: gp.Model, options: SolverOptions, oracle: bool = False):
    model.Params.OutputFlag = int(options.output_flag and not oracle)
    model.Params.TimeLimit = options.oracle_time_limit if oracle else options.time_limit
    model.Params.Threads = 1 if oracle else options.threads
    model.Params.Seed = options.seed
    model.Params.FeasibilityTol = 1e-8
    model.Params.OptimalityTol = 1e-8
    if oracle:
        model.Params.Method = 1  # Reuse the dual-simplex basis for RHS changes.
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 0
        model.Params.IntFeasTol = 1e-8
    else:
        model.Params.Method = -1
        model.Params.MIPGap = options.mip_gap
        model.Params.IntFeasTol = 1e-8


@dataclass
class DispatchBlock:
    variables: dict[str, gp.MVar]
    constraints: dict[str, gp.MConstr]
    commitment: CommitmentBlock | None = None


def add_dispatch(model: gp.Model, data: CaseData, wind_limit, pv_limit, diesel_limit,
                 pcs_limit, usable_energy_limit, load_kw, allow_shedding: bool,
                 installed_diesel_count=None) -> DispatchBlock:
    h = data.hours
    v = {k: model.addMVar(h, lb=0, name=k) for k in
         ("wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw")}
    v["shed_kw"] = model.addMVar(h, lb=0, ub=load_kw if allow_shedding else 0, name="shed_kw")
    # Usable energy above SOC_min. Increasing capacity expands feasible storage states.
    v["usable_energy_kwh"] = model.addMVar(h + 1, lb=0, name="usable_energy_kwh")
    c = {
        "wind": model.addConstr(v["wind_kw"] <= wind_limit, name="wind_available"),
        "pv": model.addConstr(v["pv_kw"] <= pv_limit, name="pv_available"),
        "diesel": model.addConstr(v["diesel_kw"] <= diesel_limit, name="diesel_available"),
        "pcs": model.addConstr(v["charge_kw"] + v["discharge_kw"] <= pcs_limit, name="pcs_available"),
        "energy": model.addConstr(v["usable_energy_kwh"] <= usable_energy_limit, name="usable_energy_limit"),
        "balance": model.addConstr(v["wind_kw"] + v["pv_kw"] + v["diesel_kw"] + v["discharge_kw"] + v["shed_kw"]
                                    == load_kw + v["charge_kw"], name="power_balance"),
    }
    energy = v["usable_energy_kwh"]
    model.addConstr(energy[1:] == energy[:-1] + data.dt_hours *
                    (data.efficiency * v["charge_kw"] - v["discharge_kw"] / data.efficiency), name="soc_transition")
    model.addConstr(energy[-1] == energy[0], name="cyclic_soc")
    commitment = add_unit_commitment(model, data, v["diesel_kw"], installed_diesel_count)
    if data.unit_commitment.enabled:
        # Minimum diesel output makes dissipative simultaneous cycling material.
        mode = model.addMVar(h, vtype=GRB.BINARY, name="battery_charge_mode")
        power_max = data.module_sizes["pcs"] * data.unit_bounds["pcs"][1]
        model.addConstr(v["charge_kw"] <= power_max * mode, name="charge_mode")
        model.addConstr(v["discharge_kw"] <= power_max * (1 - mode), name="discharge_mode")
        v["battery_charge_mode"] = mode
    return DispatchBlock(v, c, commitment)


def dispatch_values(block: DispatchBlock, data: CaseData, battery_kwh: float) -> dict[str, np.ndarray]:
    result = {k: np.asarray(v.X).copy() for k, v in block.variables.items()}
    result["stored_energy_kwh"] = result["usable_energy_kwh"] + data.soc_min * battery_kwh
    if block.commitment is not None:
        for name in ("online", "startup", "shutdown"):
            values = np.asarray(getattr(block.commitment, name).X).copy()
            result[f"diesel_unit_{name}"] = values
            result[f"diesel_{name}_units"] = values.sum(axis=0)
    return result


def dispatch_without_storage(data: CaseData, units: dict[str, int], scenario: Scenario,
                             online: np.ndarray | None) -> dict | None:
    """Construct a feasible dispatch when hourly supply suffices without storage."""
    if online is not None:
        online = np.asarray(online, dtype=float)
    maximum = data.module_sizes["diesel"] * online.sum(axis=0) if online is not None else scenario.diesel_available_kw
    minimum = data.unit_commitment.min_output_fraction * maximum if online is not None else np.zeros(data.hours)
    load = np.asarray(scenario.load_kw)
    if np.any(load < minimum) or np.any(load > maximum + scenario.wind_available_kw + scenario.pv_available_kw):
        return None
    diesel = np.minimum(load, maximum)
    wind = np.minimum(scenario.wind_available_kw, load - diesel)
    pv = np.maximum(0.0, load - diesel - wind)
    dispatch = {name: np.zeros(data.hours) for name in
                ("charge_kw", "discharge_kw", "shed_kw", "battery_charge_mode")}
    dispatch.update(wind_kw=wind, pv_kw=pv, diesel_kw=diesel,
                    usable_energy_kwh=np.zeros(data.hours + 1),
                    stored_energy_kwh=np.full(data.hours + 1, data.soc_min * data.capacities(units)["battery_energy"]))
    if online is not None:
        previous = np.pad(online[:, :-1], ((0, 0), (1, 0)))
        for name, values in (("online", online), ("startup", np.maximum(online - previous, 0)),
                             ("shutdown", np.maximum(previous - online, 0))):
            dispatch[f"diesel_unit_{name}"] = values
            dispatch[f"diesel_{name}_units"] = values.sum(axis=0)
    return dispatch


class OperationModel:
    def __init__(self, data: CaseData, options: SolverOptions, env: gp.Env | None = None):
        self.data = data
        self.options = options
        self.model = gp.Model("minimum_unserved_energy", env=env)
        configure_model(self.model, options, oracle=True)
        zeros = np.zeros(data.hours)
        self.block = add_dispatch(self.model, data, zeros, zeros, zeros, zeros, 0.0, data.load_kw, True)
        self.model.setObjective(data.dt_hours * self.block.variables["shed_kw"].sum(), GRB.MINIMIZE)
        self.model.update()
        self.solve_count = 0
        self.solver_calls = 0
        self.full_mip_solves = 0
        self.zero_loss_certificates = 0
        self.direct_zero_loss_certificates = 0
        self.storage_zero_loss_certificates = 0
        self.last_audit = None
        self.last_solve_summary = None

    def solve(self, units: dict[str, int], scenario: Scenario, return_dispatch: bool = False,
              objective: str = "eens", warm_start: dict | None = None
              ) -> tuple[float, dict[str, np.ndarray] | None]:
        self.last_solve_summary = None
        if objective not in ("eens", "nominal_cost"):
            raise ValueError(f"Unknown operation objective {objective}")
        capacity = self.data.capacities(units)
        commitment = self.block.commitment
        series = {"wind": scenario.wind_available_kw, "pv": scenario.pv_available_kw,
                  "diesel": scenario.diesel_available_kw, "pcs": scenario.pcs_available_kw,
                  "balance": scenario.load_kw}
        for name, values in series.items():
            values = np.asarray(values)
            if values.shape != (self.data.hours,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid scenario {name} series")
        available = None
        policy = None
        if self.data.unit_commitment.enabled:
            available = available_diesel_modules(self.data, units, scenario)
        if objective == "eens":
            if commitment is not None:
                policy = standby_commitment(available, max(1, math.ceil(
                    self.data.unit_commitment.min_down_hours / self.data.dt_hours)),
                    max(1, math.ceil(float(np.max(scenario.load_kw)) / self.data.module_sizes["diesel"])))
            dispatch = dispatch_without_storage(self.data, units, scenario, policy)
            if dispatch is not None:
                self._accept_zero_dispatch(units, scenario, dispatch, "constructed_without_storage")
                self.direct_zero_loss_certificates += 1
                return 0.0, dispatch if return_dispatch else None
        self.model.NumStart = 0
        configure_model(self.model, self.options, oracle=objective == "eens")
        if commitment is not None:
            commitment.online.LB = 0
            commitment.update_availability(available)
        if "battery_charge_mode" in self.block.variables:
            self.block.variables["battery_charge_mode"].LB = 0
            self.block.variables["battery_charge_mode"].UB = 1
        for name, values in series.items():
            self.block.constraints[name].RHS = values
        self.block.constraints["energy"].RHS = np.full(self.data.hours + 1,
            (self.data.soc_max - self.data.soc_min) * capacity["battery_energy"])
        if objective == "eens":
            self.block.variables["shed_kw"].UB = scenario.load_kw
            self.model.setObjective(self.data.dt_hours * self.block.variables["shed_kw"].sum(), GRB.MINIMIZE)
        elif objective == "nominal_cost":
            self.block.variables["shed_kw"].UB = 0
            cost = self.data.dt_hours * self.data.fuel_cost_per_kwh * self.block.variables["diesel_kw"].sum()
            if commitment is not None:
                cost += self.data.unit_commitment.startup_cost_yuan * commitment.startup.sum()
            self.model.setObjective(cost, GRB.MINIMIZE)
        else:
            raise ValueError(f"Unknown operation objective {objective}")
        if warm_start is not None:
            for name, variable in self.block.variables.items():
                if name in warm_start:
                    variable.Start = warm_start[name]
            if commitment is not None:
                for name in ("online", "startup", "shutdown"):
                    getattr(commitment, name).Start = warm_start[f"diesel_unit_{name}"]
        # A feasible zero-loss UC dispatch meets the universal lower bound Q>=0.
        # This is an exact certificate, never an LP relaxation passed off as UC.
        if objective == "eens" and commitment is not None:
            commitment.online.LB = policy
            commitment.online.UB = policy
            mode = self.block.variables["battery_charge_mode"]
            potential = scenario.wind_available_kw + scenario.pv_available_kw + self.data.module_sizes["diesel"] * policy.sum(axis=0)
            charge_mode = (potential >= scenario.load_kw).astype(float)
            mode.LB = charge_mode
            mode.UB = charge_mode
            self.solver_calls += 1
            self.model.optimize()
            if self.model.SolCount and abs(self.model.ObjVal) <= 1e-8:
                dispatch = dispatch_values(self.block, self.data, capacity["battery_energy"])
                self._accept_zero_dispatch(units, scenario, dispatch, "fixed_commitment_with_storage", int(self.model.Status))
                self.storage_zero_loss_certificates += 1
                return 0.0, dispatch if return_dispatch else None
            # The policy was only a feasible probe; release it completely before optimization.
            commitment.online.LB = 0
            commitment.online.UB = available
            mode.LB = 0
            mode.UB = 1
        self.solver_calls += 1
        self.model.optimize()
        self.solve_count += 1
        self.full_mip_solves += int(self.data.unit_commitment.enabled)
        optimal = self.model.Status == GRB.OPTIMAL
        # Economic re-optimization is an audit, not a loss oracle. A limited
        # feasible cost is an upper bound and must never be labelled optimal.
        limited_cost = objective == "nominal_cost" and self.model.SolCount > 0 and self.model.Status in (
            GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, GRB.SOLUTION_LIMIT, GRB.INTERRUPTED)
        if not optimal and not limited_cost:
            requirement = "optimal losses are required before cutting" if objective == "eens" else "no usable economic incumbent"
            raise OptimizationError(f"Operation {objective} status={self.model.Status}; {requirement}")
        value = max(0.0, float(self.model.ObjVal))
        dispatch = dispatch_values(self.block, self.data, capacity["battery_energy"])
        self._audit_dispatch(units, scenario, dispatch)
        bound = float(self.model.ObjBound) if self.model.IsMIP else (value if optimal else None)
        if bound is not None and not math.isfinite(bound):
            bound = None
        gap = max(0.0, value - bound) / max(abs(value), 1e-10) if bound is not None else None
        self.last_solve_summary = {"status": int(self.model.Status),
            "optimal": optimal and gap is not None and gap <= 1e-9,
            "optimal_within_gap": optimal, "requested_mip_gap": float(self.model.Params.MIPGap),
            "objective_value": value, "lower_bound": bound,
            "gap": gap,
            "runtime_seconds": float(self.model.Runtime)}
        return value, dispatch if return_dispatch else None

    def _audit_dispatch(self, units, scenario, dispatch):
        self.last_audit = audit_commitment(self.data, units, scenario, dispatch)
        physical = audit_physical_dispatch(self.data, units, scenario, dispatch)
        if not self.last_audit["passed"] or not physical["passed"]:
            raise OptimizationError(f"Operation dispatch failed audit: UC={self.last_audit}, physical={physical}")

    def _accept_zero_dispatch(self, units, scenario, dispatch, certificate, status=None):
        self._audit_dispatch(units, scenario, dispatch)
        self.solve_count += 1
        self.zero_loss_certificates += 1
        self.last_solve_summary = {"status": status, "optimal": True, "objective_value": 0.0,
            "lower_bound": 0.0, "gap": 0.0, "certificate": certificate}

    def close(self):
        self.model.dispose()


def evaluate_scenario(data: CaseData, units: dict[str, int], scenario: Scenario,
                      options: SolverOptions | None = None) -> float:
    operation = OperationModel(data, options or SolverOptions())
    try:
        return operation.solve(units, scenario)[0]
    finally:
        operation.close()
