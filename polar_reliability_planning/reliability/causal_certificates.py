"""Scalable nonanticipative feasibility certificates and causal holdout.

A single fixed causal controller supplies an upper witness on joint tree risk.
The perfect-foresight LP supplies only lower bounds: it cannot accept a capacity.
An inconclusive sandwich is reported as unresolved, never cut as infeasible.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from ..config import SolverOptions
from ..data import COMPONENTS
from .monte_carlo import ReliabilityOracle, ReliabilityResult
from .nonanticipative import NonanticipativeOptions, NonanticipativeOracle
from .operation_model import OptimizationError
from .unit_commitment import available_diesel_modules, standby_commitment, audit_commitment
from .dispatch_audit import audit_physical_dispatch
from .eens_calculation import calculate_EENS
from .cvar_calculation import calculate_CVaR


class ColdReserveController:
    """Always maintain up to a fixed target; no voluntary trips or storage use.

    The target uses public nominal forecasts and declared weather factors, not
    realized future load. Observed faults trip a unit; cold reserve starts at the
    current step subject to minimum downtime. Once started a unit stays on until
    its next fault. The original standby routine is an event-accelerated causal
    recursion; it does not make pre-fault decisions using future events.
    """
    name = "causal_cold_reserve_idle_storage_v1"

    def __init__(self, data, units, initial_soc=0.8, weather=None):
        self.data, self.units = data, data.validate_units(units)
        weather = dict(weather or {})
        factors = [1.0]
        if weather.get("enabled", False):
            factors.append(float(weather.get("extreme_load_factor", 1.0)))
        if not all(math.isfinite(x) and x >= 0 for x in factors):
            raise ValueError("Invalid declared load factor")
        self.load_floor = float(data.load_kw.min()) * min(factors)
        self.load_ceiling = float(data.load_kw.max()) * max(factors)
        self.target = max(1, math.ceil(self.load_ceiling / data.module_sizes["diesel"]))
        if not data.soc_min <= initial_soc <= data.soc_max:
            raise ValueError("Controller initial SOC is outside physical bounds")
        self.initial_soc = initial_soc
        self.initial_energy = (initial_soc - data.soc_min) * data.capacities(units)["battery_energy"]
        minimum = min(self.target, units["diesel"]) * data.module_sizes["diesel"] * data.unit_commitment.min_output_fraction
        if data.unit_commitment.enabled and minimum > self.load_floor + 1e-8:
            raise ValueError("Cold-reserve controller cannot guarantee absorption of minimum diesel output "
                             "for this declared load range; select a different causal controller")
        self.spec = {"controller": self.name, "units": self.units, "target_online": self.target,
                     "initial_soc": self.initial_soc, "storage_rule": "constant_inventory_no_charge_or_discharge",
                     "information_timing": "current_module_availability_then_dispatch",
                     "terminal_energy": "equals_fixed_initial_energy_on_every_path",
                     "unit_commitment": asdict(data.unit_commitment), "module_sizes": data.module_sizes,
                     "dt_hours": data.dt_hours, "hours": data.hours,
                     "declared_load_range_kw": [self.load_floor, self.load_ceiling],
                     "nominal_load_sha256": hashlib.sha256(data.load_kw.tobytes()).hexdigest()}
        self.fingerprint = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    def dispatch(self, scenario):
        data = self.data
        load = np.asarray(scenario.load_kw)
        if (np.any(load < self.load_floor - 1e-8) or np.any(load > self.load_ceiling + 1e-8)
                or not np.isfinite(load).all()):
            raise ValueError("Observed load outside the controller's declared range")
        available = available_diesel_modules(data, self.units, scenario)
        down = max(1, math.ceil(data.unit_commitment.min_down_hours / data.dt_hours)) if data.unit_commitment.enabled else 1
        online = standby_commitment(available, down, self.target)
        diesel = np.minimum(load, data.module_sizes["diesel"] * online.sum(axis=0))
        remaining = np.maximum(0., load - diesel)
        wind = np.minimum(scenario.wind_available_kw, remaining)
        pv = np.minimum(scenario.pv_available_kw, np.maximum(0., remaining - wind))
        shed = np.maximum(0., remaining - wind - pv)
        result = {"wind_kw": wind, "pv_kw": pv, "diesel_kw": diesel, "shed_kw": shed,
                  "charge_kw": np.zeros(data.hours), "discharge_kw": np.zeros(data.hours),
                  "battery_charge_mode": np.zeros(data.hours),
                  "usable_energy_kwh": np.full(data.hours + 1, self.initial_energy),
                  "stored_energy_kwh": np.full(data.hours + 1, self.initial_soc * data.capacities(self.units)["battery_energy"])}
        previous = np.pad(online[:, :-1], ((0, 0), (1, 0)))
        for name, value in (("online", online), ("startup", np.maximum(0, online - previous)),
                            ("shutdown", np.maximum(0, previous - online))):
            result[f"diesel_unit_{name}"] = value
            result[f"diesel_{name}_units"] = value.sum(axis=0)
        return result


def evaluate_controller(data, pool, limits, units, settings=None, on_progress=None, on_scenario=None,
                        sample_role="training"):
    settings = settings or NonanticipativeOptions()
    controller = ColdReserveController(data, units, settings.initial_soc, pool.metadata.get("weather"))
    start = time.perf_counter()
    losses = []
    largest_violation = 0.0
    diesel_energy = 0.0
    for s in range(pool.samples):
        scenario = pool.scenario(data, units, s)
        dispatch = controller.dispatch(scenario)
        physical = audit_physical_dispatch(data, units, scenario, dispatch)
        uc = audit_commitment(data, units, scenario, dispatch)
        if not physical["passed"] or not uc["passed"]:
            raise OptimizationError(f"Causal controller audit failed for scenario {s}: {physical}, {uc}")
        violation = max(uc.get("max_violation", 0.), *(v for k, v in physical.items() if k != "passed"))
        largest_violation = max(largest_violation, violation)
        q = float(dispatch["shed_kw"].sum() * data.dt_hours)
        diesel_energy += float(pool.probabilities[s] * dispatch["diesel_kw"].sum() * data.dt_hours)
        losses.append(q)
        if on_scenario:
            on_scenario(units, s, float(pool.probabilities[s]), q,
                        {"certificate": "audited_causal_policy", "controller_sha256": controller.fingerprint,
                         "pathwise_optimal": False, "physical_and_UC_audit_passed": True,
                         "max_violation": violation, "initial_usable_energy_kwh": controller.initial_energy,
                         "terminal_usable_energy_kwh": float(dispatch["usable_energy_kwh"][-1])})
        if on_progress and (s == 0 or (s + 1) % 256 == 0 or s + 1 == pool.samples):
            # Fixed full-pool weights; incomplete sums are lower bounds on the
            # performance of THIS fixed controller, never optimal-policy bounds.
            unknown_zero = np.zeros(pool.samples)
            unknown_zero[:s + 1] = losses
            on_progress(s + 1, pool.samples, calculate_EENS(unknown_zero, pool.probabilities),
                        calculate_CVaR(unknown_zero, limits.alpha, pool.probabilities))
    eens, cvar = calculate_EENS(losses, pool.probabilities), calculate_CVaR(losses, limits.alpha, pool.probabilities)
    e_ok = eens <= limits.eens_kwh + limits.tolerance_kwh
    c_ok = limits.cvar_kwh is None or cvar <= limits.cvar_kwh + limits.tolerance_kwh
    stderr = (float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
              if sample_role == "independent_holdout" and len(losses) > 1
              and np.allclose(pool.probabilities, 1.0 / len(losses)) else None)
    info = {"information_structure": "causal_policy_valid_on_unseen_histories",
            "loss_basis": "fixed_causal_policy_losses_not_pathwise_minima",
            "controller": controller.spec, "controller_sha256": controller.fingerprint,
            "sample_role": sample_role,
            "physical_and_UC_audit": {"passed": True, "scenarios": pool.samples, "max_violation": largest_violation},
            "mean_accident_diesel_energy_kwh": diesel_energy,
            "failure_proof": None,
            "joint_feasibility_proof": "one_audited_causal_policy_satisfies_both_risks" if e_ok and c_ok else None,
            "interpretation": "finite_sample_policy_performance; not population confidence certification"}
    return ReliabilityResult(eens, cvar, e_ok and c_ok, tuple(losses), stderr,
                             time.perf_counter() - start, pool.fingerprint, e_ok, c_ok, information=info)


class NonanticipativeCertificateOracle:
    """Sandwich joint feasibility without constructing the full scenario MILP."""
    def __init__(self, data, pool, limits, options=None, env=None, on_progress=None,
                 on_scenario=None, dispatch_options=None):
        pool.check_data(data)
        self.data, self.pool, self.limits = data, pool, limits
        self.options, self.env = options or SolverOptions(), env
        self.settings = dispatch_options or NonanticipativeOptions()
        self.on_progress, self.on_scenario = on_progress, on_scenario
        relaxed = replace(data, unit_commitment=replace(data.unit_commitment, enabled=False))
        self.lower = ReliabilityOracle(relaxed, pool, limits, self.options, env)
        self.cache, self.witnesses = {}, {}
        self.joint = None
        self.joint_model_solves = 0
        self.cache_hits = 0

    def certify_failure(self, units):
        units = self.data.validate_units(units)
        key = tuple(units[k] for k in COMPONENTS)
        if key in self.cache:
            return None if self.cache[key].feasible else self.cache[key]
        bound = self.lower.evaluate(units, certify_infeasible_early=True)
        if bound.feasible:
            return None  # Lower bound not excessive: no failure evidence.
        result = replace(bound, metrics_are_lower_bounds=True, losses_are_relaxation_bounds=True,
                         standard_error_kwh=None,
                         eens_feasible=False if bound.eens_feasible is False else None,
                         cvar_feasible=False if bound.cvar_feasible is False else None,
                         information={"information_structure": "nonanticipative_feasibility_lower_bound",
                                      "failure_proof": "perfect_foresight_LP_risk_lower_bound_exceeds_limit",
                                      "scope": "lower_bound_only_never_acceptance"})
        self.cache[key] = result
        return result

    def evaluate(self, units, certify_infeasible_early=False):
        units = self.data.validate_units(units)
        key = tuple(units[k] for k in COMPONENTS)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        witness = evaluate_controller(self.data, self.pool, self.limits, units, self.settings,
                                      self.on_progress, self.on_scenario)
        self.witnesses[key] = witness
        if witness.feasible:
            self.cache[key] = witness
            return witness
        # A failing feasible controller is not a proof of joint infeasibility.
        failed = self.certify_failure(units)
        if failed is not None:
            return failed
        if self.pool.samples * self.data.hours <= self.settings.max_scenario_hours:
            if self.joint is None:
                self.joint = NonanticipativeOracle(self.data, self.pool, self.limits, self.options, self.env,
                                                   dispatch_options=self.settings)
            result = self.joint.evaluate(units)
            self.joint_model_solves = self.joint.joint_model_solves
            self.cache[key] = result
            return result
        raise OptimizationError("Nonanticipative feasibility unresolved: causal witness exceeds a limit, "
                                "but perfect-foresight LP cannot prove failure. No invalid cut or "
                                "perfect-foresight acceptance is performed. A richer causal policy or "
                                "scalable joint solve is required.")

    def save_policy(self, path, units):
        key = tuple(self.data.validate_units(units)[k] for k in COMPONENTS)
        info = self.cache[key].information
        if "controller" not in info:
            raise OptimizationError("Accepted tree-only policy cannot be exported as an unseen-history controller")
        Path(path).write_text(json.dumps({"controller": info["controller"],
                                         "controller_sha256": info["controller_sha256"]}, indent=2) + "\n")

    def close(self):
        self.lower.close()
        if self.joint is not None:
            self.joint.close()
