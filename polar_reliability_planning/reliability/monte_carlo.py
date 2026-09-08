"""Fixed-sample EENS/CVaR oracle using certified chronological UC dispatch."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import gurobipy as gp
import numpy as np

from ..config import ReliabilityLimits, SolverOptions
from ..data import CaseData, COMPONENTS
from ..scenario_generation import ScenarioPool
from .cvar_calculation import calculate_CVaR
from .eens_calculation import calculate_EENS
from .operation_model import OperationModel


@dataclass(frozen=True)
class ReliabilityResult:
    eens_kwh: float
    cvar_kwh: float
    feasible: bool
    losses_kwh: tuple[float | None, ...]
    standard_error_kwh: float | None
    elapsed_seconds: float
    sample_fingerprint: str
    eens_feasible: bool | None = True
    cvar_feasible: bool | None = True
    metrics_are_lower_bounds: bool = False
    losses_are_relaxation_bounds: bool = False

    def summary(self) -> dict:
        return {"eens_kwh": self.eens_kwh, "cvar_kwh": self.cvar_kwh, "feasible": self.feasible,
                "standard_error_kwh": self.standard_error_kwh, "samples": len(self.losses_kwh),
                "elapsed_seconds": self.elapsed_seconds, "sample_fingerprint": self.sample_fingerprint,
                "eens_feasible": self.eens_feasible, "cvar_feasible": self.cvar_feasible,
                "metrics_are_lower_bounds": self.metrics_are_lower_bounds,
                "losses_are_relaxation_bounds": self.losses_are_relaxation_bounds,
                "evaluated_scenarios": sum(q is not None for q in self.losses_kwh),
                "violated_constraints": [name for name, ok in (("EENS", self.eens_feasible), ("CVaR", self.cvar_feasible)) if ok is False]}


class ReliabilityOracle:
    def __init__(self, data: CaseData, pool: ScenarioPool, limits: ReliabilityLimits,
                 options: SolverOptions | None = None, env: gp.Env | None = None, on_progress=None, on_scenario=None):
        pool.check_data(data)
        self.data, self.pool, self.limits = data, pool, limits
        self.operation = OperationModel(data, options or SolverOptions(), env)
        self.cache: dict[tuple[int, ...], ReliabilityResult] = {}
        self.cache_hits = 0
        self.on_progress = on_progress
        self.on_scenario = on_scenario
        self._options, self._env = options, env
        self.relaxation_oracle = None

    def certify_failure(self, units: dict[str, int]) -> ReliabilityResult | None:
        """LP-relaxed losses can prove UC failure, but never UC feasibility."""
        if not self.data.unit_commitment.enabled:
            result = self.evaluate(units, certify_infeasible_early=True)
            return None if result.feasible else result
        if self.relaxation_oracle is None:
            relaxed_data = replace(self.data, unit_commitment=replace(self.data.unit_commitment, enabled=False))
            self.relaxation_oracle = ReliabilityOracle(relaxed_data, self.pool, self.limits, self._options,
                                                       self._env, self.on_progress)
        result = self.relaxation_oracle.evaluate(units, certify_infeasible_early=True)
        if result.feasible:
            return None
        certificate = replace(result, metrics_are_lower_bounds=True, losses_are_relaxation_bounds=True,
                               standard_error_kwh=None,
                               eens_feasible=False if result.eens_feasible is False else None,
                               cvar_feasible=False if result.cvar_feasible is False else None)
        self.cache[tuple(units[k] for k in COMPONENTS)] = certificate
        return certificate

    def evaluate(self, units: dict[str, int], certify_infeasible_early: bool = False) -> ReliabilityResult:
        units = self.data.validate_units(units)
        key = tuple(units[k] for k in COMPONENTS)
        if key in self.cache and (certify_infeasible_early or not self.cache[key].metrics_are_lower_bounds):
            self.cache_hits += 1
            return self.cache[key]
        if certify_infeasible_early and self.data.unit_commitment.enabled:
            failed = self.certify_failure(units)
            if failed is not None:
                return failed
        start = time.perf_counter()
        losses = (list(self.cache[key].losses_kwh) if key in self.cache and not self.cache[key].losses_are_relaxation_bounds
                  else [None] * self.pool.samples)
        for i in range(self.pool.samples):
            if losses[i] is None:
                losses[i] = self.operation.solve(units, self.pool.scenario(self.data, units, i))[0]
                if self.on_scenario:
                    self.on_scenario(units, i, float(self.pool.probabilities[i]), losses[i], self.operation.last_solve_summary)
            bound_losses = [q if q is not None else 0.0 for q in losses]
            eens = calculate_EENS(bound_losses, self.pool.probabilities)
            cvar = calculate_CVaR(bound_losses, self.limits.alpha, self.pool.probabilities)
            interval = max(16, min(256, self.pool.samples // 20))
            if self.on_progress and (i == 0 or (i + 1) % interval == 0 or i + 1 == self.pool.samples):
                self.on_progress(i + 1, self.pool.samples, eens, cvar)
            if certify_infeasible_early and (eens > self.limits.eens_kwh + self.limits.tolerance_kwh or
                    (self.limits.cvar_kwh is not None and cvar > self.limits.cvar_kwh + self.limits.tolerance_kwh)):
                break
        incomplete = any(q is None for q in losses)
        eens_feasible = eens <= self.limits.eens_kwh + self.limits.tolerance_kwh
        cvar_feasible = self.limits.cvar_kwh is None or cvar <= self.limits.cvar_kwh + self.limits.tolerance_kwh
        feasible = not incomplete and eens_feasible and cvar_feasible
        stderr = (float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
                  if not incomplete and len(losses) > 1 and np.allclose(self.pool.probabilities, 1 / len(losses)) else None)
        result = ReliabilityResult(eens, cvar, feasible, tuple(losses), stderr,
                                   time.perf_counter() - start, self.pool.fingerprint,
                                   None if incomplete and eens_feasible else eens_feasible,
                                   None if incomplete and cvar_feasible else cvar_feasible, incomplete)
        self.cache[key] = result
        return result

    def close(self):
        self.operation.close()
        if self.relaxation_oracle is not None:
            self.relaxation_oracle.close()
