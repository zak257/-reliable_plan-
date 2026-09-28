"""Simultaneous distribution-free bounds for bounded scenario losses.

Each scenario contributes an interval containing its exact recourse loss.  An
unresolved scenario must remain in the arrays as ``[0, bound]``: the denominator
is the number of sampled scenarios, never the number of solved scenarios.

For a finite family of ``cardinality`` capacities, the allocation
``delta_nm = 6 * delta / (pi**2 * cardinality * m**2)`` makes the DKW event
simultaneous over every capacity and every positive sample prefix.  This
permits adaptive selection of capacities, oracle refinements, and stopping
times, provided the scenario samples for each capacity are i.i.d. from the
specified distribution and every reported oracle interval is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RiskBounds:
    """Certified risk bounds and their empirical/oracle decomposition."""

    eens_lower: float
    eens_upper: float
    cvar_lower: float
    cvar_upper: float
    radius: float
    mean_oracle_width: float
    sample_count: int
    empirical_eens_lower: float
    empirical_eens_upper: float
    empirical_cvar_lower: float
    empirical_cvar_upper: float

    @property
    def eens_width(self) -> float:
        return max(0.0, self.eens_upper - self.eens_lower)

    @property
    def cvar_width(self) -> float:
        return max(0.0, self.cvar_upper - self.cvar_lower)

    @property
    def eens_sampling_width(self) -> float:
        """Width contributed by the confidence band, beyond oracle intervals."""
        return max(0.0, self.eens_width - self.mean_oracle_width)

    @property
    def cvar_oracle_width(self) -> float:
        return max(0.0, self.empirical_cvar_upper - self.empirical_cvar_lower)

    @property
    def cvar_sampling_width(self) -> float:
        return max(0.0, self.cvar_width - self.cvar_oracle_width)


def _scalar(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array") from exc
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _alpha(value: Real) -> float:
    result = _scalar(value, "alpha")
    if not 0.0 <= result < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    return result


def empirical_cvar(
    values: Sequence[float],
    alpha: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Return upper-tail CVaR, including a fractional quantile-boundary atom.

    This evaluates ``min_eta eta + E[(X - eta)+] / (1 - alpha)`` over
    continuous eta exactly for a finite distribution; no eta grid is used.
    Optional nonnegative weights are normalized to sum to one.  At alpha=0,
    CVaR equals the weighted mean.  Loss values may be any finite real numbers.
    """
    losses = _vector(values, "values")
    confidence = _alpha(alpha)
    if weights is None:
        probability = np.full(losses.size, 1.0 / losses.size)
    else:
        probability = _vector(weights, "weights")
        if probability.shape != losses.shape or np.any(probability < 0.0):
            raise ValueError("weights must match values and be nonnegative")
        # Scaling first avoids overflow when finite weights have a large sum.
        largest = float(np.max(probability))
        if largest <= 0.0:
            raise ValueError("weights must have a positive sum")
        probability = probability / largest
        probability = probability / np.sum(probability)

    order = np.argsort(losses)[::-1]
    descending = losses[order]
    probability = probability[order]
    tail = 1.0 - confidence
    previous_mass = np.concatenate(([0.0], np.cumsum(probability[:-1])))
    tail_weights = np.minimum(probability, np.maximum(0.0, tail - previous_mass))
    result = float(np.dot(descending, tail_weights / tail))
    # Summation roundoff must not put a constant distribution outside its range.
    positive = probability > 0.0
    return float(np.clip(result, np.min(descending[positive]), np.max(descending[positive])))


def _band_distributions(
    lower: np.ndarray, upper: np.ndarray, bound: float, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Discrete distributions with CDF min(F_l+r,1) and max(F_u-r,0).

    The former receives extra mass at zero and loses mass from its largest
    atoms.  The latter loses mass from its smallest atoms and receives extra
    mass at ``bound``.  Both CDFs equal one at ``bound``; their endpoint atoms
    are essential when all observed losses are zero or equal to the bound.
    """
    m = lower.size
    shift = min(radius, 1.0)
    # Cell overlaps with quantile intervals are exact even when m*r is not an
    # integer, retaining the appropriate fraction of the boundary observation.
    left = np.arange(m, dtype=float) / m
    right = np.arange(1, m + 1, dtype=float) / m
    low_weights = np.maximum(0.0, np.minimum(right, 1.0 - shift) - left)
    high_weights = np.maximum(0.0, right - np.maximum(left, shift))
    return (
        np.concatenate(([0.0], np.sort(lower))),
        np.concatenate(([shift], low_weights)),
        np.concatenate((np.sort(upper), [bound])),
        np.concatenate((high_weights, [shift])),
    )


def risk_bounds(
    lower: Sequence[float],
    upper: Sequence[float],
    bound: float,
    alpha: float,
    delta: float,
    cardinality: int,
) -> RiskBounds:
    """Compute exact EENS/CVaR extrema under simultaneous DKW CDF bands.

    ``lower[s] <= Q[s] <= upper[s]`` must hold for *all* sampled scenarios,
    including unresolved ones, and ``0 <= Q <= bound`` must be a deterministic
    valid support bound.  ``cardinality`` is the size of the complete finite
    capacity search space, not merely the number evaluated so far.  ``delta``
    is the total failure probability across all capacities and sample counts.

    For 0 <= x < bound, the enclosing CDF bands are
    ``max(F_upper(x) - radius, 0)`` and
    ``min(F_lower(x) + radius, 1)``.  Their induced discrete distributions are
    integrated directly, with exact continuous-eta CVaR and endpoint masses.
    """
    lows = _vector(lower, "lower")
    highs = _vector(upper, "upper")
    support = _scalar(bound, "bound")
    confidence = _alpha(alpha)
    failure = _scalar(delta, "delta")
    if support < 0.0:
        raise ValueError("bound must be nonnegative")
    if not 0.0 < failure < 1.0:
        raise ValueError("delta must satisfy 0 < delta < 1")
    if isinstance(cardinality, (bool, np.bool_)) or not isinstance(cardinality, Integral) or cardinality < 1:
        raise ValueError("cardinality must be a positive integer")
    if lows.shape != highs.shape:
        raise ValueError("lower and upper must have the same shape")
    if np.any(lows < 0.0) or np.any(lows > highs) or np.any(highs > support):
        raise ValueError("oracle intervals must satisfy 0 <= lower <= upper <= bound")

    m = int(lows.size)
    # Work in logs so large finite search spaces and tiny delta cannot underflow
    # the allocation before its logarithm is taken.
    log_allocation = (
        math.log(6.0) + math.log(failure) - 2.0 * math.log(math.pi)
        - math.log(int(cardinality)) - 2.0 * math.log(m)
    )
    radius = math.sqrt((math.log(2.0) - log_allocation) / (2.0 * m))
    low_values, low_probability, high_values, high_probability = _band_distributions(
        lows, highs, support, radius
    )
    return RiskBounds(
        eens_lower=float(np.clip(np.dot(low_values, low_probability), 0.0, support)),
        eens_upper=float(np.clip(np.dot(high_values, high_probability), 0.0, support)),
        cvar_lower=empirical_cvar(low_values, confidence, low_probability),
        cvar_upper=empirical_cvar(high_values, confidence, high_probability),
        radius=radius,
        mean_oracle_width=float(np.mean(highs - lows)),
        sample_count=m,
        empirical_eens_lower=float(np.mean(lows)),
        empirical_eens_upper=float(np.mean(highs)),
        empirical_cvar_lower=empirical_cvar(lows, confidence),
        empirical_cvar_upper=empirical_cvar(highs, confidence),
    )
