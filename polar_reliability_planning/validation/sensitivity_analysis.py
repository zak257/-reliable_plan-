"""EENS-limit sensitivity on the same fixed sample set."""
from __future__ import annotations

from dataclasses import replace

from ..planning import optimize_reliability
from ..reliability import ReliabilityOracle


def run_sensitivity(data, pool, limits, eens_limits, options, max_iterations=100,
                     lift_cuts=False, env=None, on_iteration=None):
    rows = []
    for limit in eens_limits:
        current_limits = replace(limits, eens_kwh=float(limit))
        oracle = ReliabilityOracle(data, pool, current_limits, options, env)
        try:
            def callback(row):
                if on_iteration:
                    on_iteration({"sensitivity_eens_limit_kwh": float(limit), **row})

            result = optimize_reliability(data, oracle, options, max_iterations, lift_cuts, callback, env)
            rows.append({"eens_limit_kwh": float(limit), "status": result.status,
                         "iterations": len(result.history), "cuts": len(result.cuts),
                         "solution": result.solution.summary() if result.solution else None,
                         "reliability": result.reliability.summary() if result.reliability else None,
                         "lower_bound_yuan": result.lower_bound_yuan})
        finally:
            oracle.close()
    return rows
