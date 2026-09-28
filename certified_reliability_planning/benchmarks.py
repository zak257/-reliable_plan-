"""Analytic examples with population truth separate from iid certification data.

The default example has only nine capacity vectors, yet its total cost falls
and then rises as capacity grows.  This catches the invalid practice of
discarding all capacities above the first feasible vector.  No optimization
license or existing planning implementation is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
import time


@dataclass(frozen=True)
class AnalyticInterval:
    """Solver-independent implementation of the optimizer's interval protocol."""

    lower: float
    upper: float
    work: float = 1.0
    runtime_seconds: float = 0.0
    status: str = "analytic_optimal"


class TinyCapacityOracle:
    """Two iid demand states and two substitutable capacity technologies.

    A fresh independent uniform draw chooses demand 0.8 or 1.0, each with
    probability 1/2.  All capacity vectors see that same draw.  Installed
    capacities ``(x, y)`` remove ``0.35*x + 0.25*y`` units of unserved demand.

    ``interval_floor`` deliberately leaves a rigorously valid, nonzero loss
    interval even after a call requesting exactness.  This simulates a limited
    solve and permits budget-exhaustion tests without sleeping or a MIP solver.
    It is zero in normal use.  ``refinable=True`` initially returns a wider
    bracket and halves its width on subsequent calls for the same sample.
    """

    loss_bound = 1.0
    dimension_names = ("firm_modules", "renewable_modules")

    def __init__(self, seed: int = 7, *, interval_floor: float = 0.0,
                 economic_floor: float = 0.0, refinable: bool = False):
        for name, value in (("interval_floor", interval_floor),
                            ("economic_floor", economic_floor)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.seed = seed
        self.interval_floor = interval_floor
        self.economic_floor = economic_floor
        self.refinable = refinable
        self.candidates = list(product(range(3), repeat=2))
        self.capital_lower_bounds = {
            point: float(2 * point[0] + 3 * point[1]) for point in self.candidates
        }
        self.cost_lower_bounds = [self.capital_lower_bounds[point] for point in self.candidates]
        self.costs = {
            point: self.capital_lower_bounds[point]
            + 40.0 * max(0.0, 1.0 - 0.45 * point[0] - 0.30 * point[1])
            for point in self.candidates
        }
        self._rng = random.Random(seed)
        self._draws: list[float] = []
        self._operation_visits: dict[tuple[tuple[int, ...], int], int] = {}
        self._economic_visits: dict[tuple[int, ...], int] = {}
        self.operation_calls = 0
        self.economic_calls = 0

    @property
    def samples(self) -> int:
        return len(self._draws)

    @property
    def draws(self) -> tuple[float, ...]:
        """Immutable diagnostic copy of the common iid sample prefix."""
        return tuple(self._draws)

    def extend(self, m: int) -> None:
        """Extend to a *total* of m samples, retaining every old draw exactly."""
        if isinstance(m, bool) or not isinstance(m, int) or m < self.samples:
            raise ValueError("sample total must be an integer at least samples")
        self._draws.extend(self._rng.random() for _ in range(m - self.samples))

    def _point(self, point) -> tuple[int, int]:
        result = tuple(point)
        if result not in self.capital_lower_bounds:
            raise ValueError(f"capacity outside the finite domain: {point}")
        return result

    def true_loss(self, point) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Exact law (loss values, probabilities), never used for certification."""
        x, y = self._point(point)
        capacity = 0.35 * x + 0.25 * y
        return (max(0.0, 0.8 - capacity), max(0.0, 1.0 - capacity)), (0.5, 0.5)

    def true_metrics(self, point, alpha: float = 0.5) -> dict[str, float]:
        if not math.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha must belong to [0, 1)")
        values, probabilities = self.true_loss(point)
        mean = sum(value * probability for value, probability in zip(values, probabilities))
        # The finite-law RU objective is minimized at a support value.
        cvar = min(t + sum(p * max(0.0, q - t) for q, p in zip(values, probabilities))
                   / (1.0 - alpha) for t in values)
        return {"eens": mean, "cvar": cvar, "cost": self.costs[self._point(point)]}

    @staticmethod
    def _validate_request(budget_seconds: float, absolute_gap: float) -> None:
        if math.isnan(budget_seconds) or budget_seconds < 0:
            raise ValueError("budget_seconds must be nonnegative")
        if not math.isfinite(absolute_gap) or absolute_gap < 0:
            raise ValueError("absolute_gap must be finite and nonnegative")

    def operation(self, point, index: int, budget_seconds: float,
                  absolute_gap: float) -> AnalyticInterval:
        self._validate_request(budget_seconds, absolute_gap)
        started = time.perf_counter()
        point = self._point(point)
        if not 0 <= index < self.samples:
            raise IndexError(index)
        if budget_seconds == 0:
            return AnalyticInterval(0.0, self.loss_bound, 0.0, 0.0, "time_limit")
        self.operation_calls += 1
        key = (point, index)
        visits = self._operation_visits.get(key, 0) + 1
        self._operation_visits[key] = visits
        values, _ = self.true_loss(point)
        value = values[int(self._draws[index] >= 0.5)]
        width = max(self.interval_floor, absolute_gap)
        if self.refinable:
            width = max(width, 0.5 ** visits)
        lower = max(0.0, value - width / 2.0)
        upper = min(self.loss_bound, value + width / 2.0)
        return AnalyticInterval(lower, upper, 1.0, time.perf_counter() - started,
                                "analytic_optimal" if lower == upper else "gap_limit")

    def economic(self, point, budget_seconds: float,
                 absolute_gap: float) -> AnalyticInterval:
        self._validate_request(budget_seconds, absolute_gap)
        started = time.perf_counter()
        point = self._point(point)
        if budget_seconds == 0:
            return AnalyticInterval(self.capital_lower_bounds[point], math.inf,
                                    0.0, 0.0, "time_limit")
        self.economic_calls += 1
        visits = self._economic_visits.get(point, 0) + 1
        self._economic_visits[point] = visits
        width = max(self.economic_floor, absolute_gap)
        if self.refinable:
            width = max(width, 16.0 * 0.5 ** visits)
        value = self.costs[point]
        lower = max(self.capital_lower_bounds[point], value - width / 2.0)
        upper = value + width / 2.0
        return AnalyticInterval(lower, upper, 1.0, time.perf_counter() - started,
                                "analytic_optimal" if lower == upper else "gap_limit")

    def truth_table(self, eens_limit: float = 0.3, cvar_limit: float = 0.5,
                    alpha: float = 0.5) -> list[dict]:
        """Exhaustive population reference, for checking reported certificates."""
        rows = []
        for point in self.candidates:
            row = {"point": list(point), **self.true_metrics(point, alpha)}
            # Account only for floating-point arithmetic in this reference
            # table; population certification uses the stated limits directly.
            row["feasible"] = (row["eens"] <= eens_limit + 1e-12
                               and row["cvar"] <= cvar_limit + 1e-12)
            rows.append(row)
        return rows


def tiny_benchmark(seed: int = 7, **kwargs) -> TinyCapacityOracle:
    """Construct the default analytic instance, whose unique optimum is (2, 1)."""
    return TinyCapacityOracle(seed, **kwargs)
