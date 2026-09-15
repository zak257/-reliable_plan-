"""Joint finite-scenario UC with decisions measurable in observed history.

This is an empirical scenario-tree policy, not an independent pathwise loss
oracle and not an out-of-sample controller. No per-scenario perfect-foresight
certificate is used for acceptance. Policies on unseen histories are undefined.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
import tempfile
from pathlib import Path
import shutil

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..config import SolverOptions
from ..data import COMPONENTS
from ..data.cap_plan_loader import FAILABLE_COMPONENTS
from .operation_model import add_dispatch, configure_model, dispatch_values, OptimizationError
from .unit_commitment import available_diesel_modules, audit_commitment
from .dispatch_audit import audit_physical_dispatch
from .monte_carlo import ReliabilityResult
from .eens_calculation import calculate_EENS
from .cvar_calculation import calculate_CVaR


@dataclass(frozen=True)
class NonanticipativeOptions:
    # Known before the first observation, common to every scenario.
    initial_soc: float = 0.8
    # Guard against accidentally building the historical 20000 x 8760 model.
    max_scenario_hours: int = 200000

    def __post_init__(self):
        if not math.isfinite(self.initial_soc) or not 0 <= self.initial_soc <= 1:
            raise ValueError("initial_soc must be between zero and one")
        if isinstance(self.max_scenario_hours, bool) or not isinstance(self.max_scenario_hours, int) or self.max_scenario_hours < 1:
            raise ValueError("max_scenario_hours must be a positive integer")


def observation_nodes(pool, units):
    """Node IDs after this hour's observation; branches never merge again.

    Observe weather, current factors and installed modules only. Neither scenario
    index, future observations nor uninstalled module states enter a history.
    Parent IDs encode the whole past, avoiding quadratic prefix comparisons.
    """
    nodes = np.empty((pool.samples, pool.hours), dtype=np.int64)
    parents = np.zeros(pool.samples, dtype=np.int64)
    for t in range(pool.hours):
        groups = {}
        for s in range(pool.samples):
            observation = tuple(pool.availability[k][s, :units[k], t].tobytes()
                                for k in FAILABLE_COMPONENTS)
            key = (int(parents[s]), int(pool.weather[s, t]),
                   float(pool.wind_factor[s, t]), float(pool.pv_factor[s, t]),
                   float(pool.load_factor[s, t]), *observation)
            nodes[s, t] = groups.setdefault(key, len(groups))
        parents = nodes[:, t]
    return nodes


def audit_nonanticipativity(dispatches, nodes, initial_energy):
    error = 0.0
    for d in dispatches:
        error = max(error, abs(float(d["usable_energy_kwh"][0]) - initial_energy))
    for t in range(nodes.shape[1]):
        representatives = {}
        for s in range(nodes.shape[0]):
            r = representatives.setdefault(int(nodes[s, t]), s)
            if s == r:
                continue
            for name in ("wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw", "shed_kw", "battery_charge_mode"):
                if name in dispatches[s]:
                    error = max(error, abs(float(dispatches[s][name][t] - dispatches[r][name][t])))
            for name in ("diesel_unit_online", "diesel_unit_startup", "diesel_unit_shutdown"):
                if name in dispatches[s]:
                    error = max(error, float(np.max(np.abs(dispatches[s][name][:, t] - dispatches[r][name][:, t]), initial=0)))
            for k in (t, t + 1):
                error = max(error, abs(float(dispatches[s]["usable_energy_kwh"][k] - dispatches[r]["usable_energy_kwh"][k])))
    return {"passed": error <= 1e-5, "max_violation": error,
            "information_nodes": sum(len(np.unique(nodes[:, t])) for t in range(nodes.shape[1]))}


class NonanticipativeOracle:
    def __init__(self, data, pool, limits, options=None, env=None, on_progress=None,
                 on_scenario=None, dispatch_options=None):
        pool.check_data(data)
        self.data, self.pool, self.limits = data, pool, limits
        self.options = options or SolverOptions()
        self.dispatch_options = dispatch_options or NonanticipativeOptions()
        if not data.soc_min <= self.dispatch_options.initial_soc <= data.soc_max:
            raise ValueError("initial_soc must be within the case SOC bounds")
        self.check_size(pool.samples, data.hours, self.dispatch_options)
        self.env, self.on_progress, self.on_scenario = env, on_progress, on_scenario
        self.cache = {}
        self.cache_hits = 0
        self.joint_model_solves = 0
        self.last_units = None
        self.last_dispatches = None
        self.last_nodes = None
        self.last_audit = None
        self._policy_directory = tempfile.TemporaryDirectory(prefix="reliable_nonanticipative_")
        self._policy_files = {}

    @staticmethod
    def check_size(samples, hours, settings):
        if samples * hours > settings.max_scenario_hours:
            raise ValueError(f"Nonanticipative joint model requests {samples} x {hours} = {samples * hours} "
                             f"scenario-hours, exceeding max_scenario_hours={settings.max_scenario_hours}. "
                             "Use an explicitly smaller tree/horizon or raise accident_dispatch.max_scenario_hours "
                             "after assessing memory; no perfect-foresight fallback is performed.")

    def certify_failure(self, units):
        result = self.evaluate(units)
        return None if result.feasible else result

    def evaluate(self, units, certify_infeasible_early=False):
        units = self.data.validate_units(units)
        key = tuple(units[k] for k in COMPONENTS)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        start = time.perf_counter()
        data, pool, limits = self.data, self.pool, self.limits
        capacity = data.capacities(units)
        initial = (self.dispatch_options.initial_soc - data.soc_min) * capacity["battery_energy"]
        nodes = observation_nodes(pool, units)
        blocks, scenarios = [], []
        model = gp.Model("joint_nonanticipative_accident_dispatch", env=self.env)
        try:
            configure_model(model, self.options, oracle=True)
            for s in range(pool.samples):
                scenario = pool.scenario(data, units, s)
                scenarios.append(scenario)
                block = add_dispatch(model, data, scenario.wind_available_kw, scenario.pv_available_kw,
                                     scenario.diesel_available_kw, scenario.pcs_available_kw,
                                     (data.soc_max - data.soc_min) * capacity["battery_energy"],
                                     scenario.load_kw, True)
                model.addConstr(block.variables["usable_energy_kwh"][0] == initial,
                                name=f"common_initial_inventory_{s}")
                if block.commitment is not None:
                    block.commitment.update_availability(available_diesel_modules(data, units, scenario))
                blocks.append(block)
            # Link full control vectors at every information set, including UC
            # and storage mode, rather than only linking generation or shed.
            for t in range(data.hours):
                representatives = {}
                for s in range(pool.samples):
                    r = representatives.setdefault(int(nodes[s, t]), s)
                    if r == s:
                        continue
                    for name, variable in blocks[s].variables.items():
                        model.addConstr(variable[t] == blocks[r].variables[name][t],
                                        name=f"na_{name}_{s}_{t}")
                    if blocks[s].commitment is not None:
                        for name in ("online", "startup", "shutdown"):
                            model.addConstr(getattr(blocks[s].commitment, name)[:, t] ==
                                            getattr(blocks[r].commitment, name)[:, t],
                                            name=f"na_{name}_{s}_{t}")
            loss = model.addMVar(pool.samples, lb=0, name="annual_loss")
            for s, block in enumerate(blocks):
                model.addConstr(loss[s] == data.dt_hours * block.variables["shed_kw"].sum())
            # Feasibility of both risks is optimized jointly. Minimizing EENS
            # alone can miss a different policy satisfying CVaR simultaneously.
            violation = model.addVar(lb=0, name="joint_risk_violation_kwh")
            model.addConstr(pool.probabilities @ loss <= limits.eens_kwh + violation)
            if limits.cvar_kwh is not None:
                eta = model.addVar(lb=0, name="cvar_var")
                excess = model.addMVar(pool.samples, lb=0, name="cvar_excess")
                model.addConstr(excess >= loss - eta)
                model.addConstr(eta + (pool.probabilities @ excess) / (1 - limits.alpha)
                                <= limits.cvar_kwh + violation)
            model.setObjective(violation, GRB.MINIMIZE)
            model.optimize()
            self.joint_model_solves += 1
            if model.SolCount < 1:
                raise OptimizationError(f"Joint nonanticipative model has no usable policy: status={model.Status}")
            bound = float(model.ObjBound)
            # Once zero joint violation is established, improve mean loss within
            # the dual-risk feasible set. This avoids reporting arbitrary shed
            # from a degenerate zero-violation feasible policy.
            if float(violation.X) <= limits.tolerance_kwh:
                allowed = max(0.0, float(violation.X))
                warm = {v: v.X for v in model.getVars()}
                model.addConstr(violation <= allowed, name="preserve_joint_risk_feasibility")
                model.setObjective(pool.probabilities @ loss, GRB.MINIMIZE)
                for variable, value in warm.items():
                    variable.Start = value
                model.optimize()
                if model.SolCount < 1:
                    raise OptimizationError("No usable policy after joint mean-loss refinement")
            dispatches = [dispatch_values(b, data, capacity["battery_energy"]) for b in blocks]
            for s, dispatch in enumerate(dispatches):
                physical = audit_physical_dispatch(data, units, scenarios[s], dispatch)
                commitment = audit_commitment(data, units, scenarios[s], dispatch)
                if not physical["passed"] or not commitment["passed"]:
                    raise OptimizationError(f"Scenario {s} failed dispatch audit: {physical}, {commitment}")
            audit = audit_nonanticipativity(dispatches, nodes, initial)
            if not audit["passed"]:
                raise OptimizationError(f"Nonanticipativity audit failed: {audit}")
            losses = tuple(max(0.0, float(d["shed_kw"].sum() * data.dt_hours)) for d in dispatches)
            eens = calculate_EENS(losses, pool.probabilities)
            cvar = calculate_CVaR(losses, limits.alpha, pool.probabilities)
            e_ok = eens <= limits.eens_kwh + limits.tolerance_kwh
            c_ok = limits.cvar_kwh is None or cvar <= limits.cvar_kwh + limits.tolerance_kwh
            feasible = e_ok and c_ok
            # An incumbent with excessive losses never proves infeasibility.
            # A positive global bound on minimum joint violation does.
            if not feasible and (not math.isfinite(bound) or bound <= limits.tolerance_kwh):
                raise OptimizationError("Nonanticipative risk remains unresolved: incumbent fails, "
                                        "but joint violation lower bound does not certify failure; no cut added")
            info = {"information_structure": "observed_history_scenario_tree",
                    "loss_basis": "one_joint_risk_policy_not_pathwise_minima",
                    "policy_scope": "fixed_empirical_tree_only; unseen_histories_undefined",
                    "initial_soc": self.dispatch_options.initial_soc,
                    "joint_violation_kwh": float(violation.X), "joint_violation_bound_kwh": bound,
                    "joint_solver_status": int(model.Status), "nonanticipativity_audit": audit,
                    "failure_proof": None if feasible else "positive_joint_violation_lower_bound",
                    "standard_error_interpretation": "not_reported_for_sample_optimized_policy"}
            result = ReliabilityResult(eens, cvar, feasible, losses, None, time.perf_counter() - start,
                                       pool.fingerprint, e_ok, c_ok, information=info)
            self.cache[key] = result
            self.last_units, self.last_dispatches, self.last_nodes, self.last_audit = dict(units), dispatches, nodes, audit
            policy_file = Path(self._policy_directory.name) / f"policy_{len(self._policy_files)}.npz"
            np.savez_compressed(policy_file, nodes=nodes, units=np.array(key),
                                **{k: np.stack([d[k] for d in dispatches]) for k in dispatches[0]})
            self._policy_files[key] = policy_file
            if self.on_progress:
                self.on_progress(pool.samples, pool.samples, eens, cvar)
            if self.on_scenario:
                for s, q in enumerate(losses):
                    self.on_scenario(units, s, float(pool.probabilities[s]), q,
                                     {"certificate": "joint_nonanticipative_policy", "pathwise_optimal": False,
                                      "joint_violation_bound_kwh": bound})
            return result
        finally:
            model.dispose()

    def save_policy(self, path, units=None):
        if units is None:
            units = self.last_units
        if units is None:
            raise ValueError("No joint policy has been solved")
        key = tuple(self.data.validate_units(units)[k] for k in COMPONENTS)
        shutil.copyfile(self._policy_files[key], path)

    def close(self):
        self._policy_directory.cleanup()
