"""Capacity-independent, repairable module availability trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FailureParameters:
    normal_rate_per_hour: float = 0.0
    extreme_rate_per_hour: float = 0.0
    mean_repair_hours: float = 24.0
    extreme_repair_multiplier: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(x) and x >= 0 for x in (self.normal_rate_per_hour, self.extreme_rate_per_hour)):
            raise ValueError("Failure rates must be finite and nonnegative")
        if not all(math.isfinite(x) and x > 0 for x in (self.mean_repair_hours, self.extreme_repair_multiplier)):
            raise ValueError("Repair hours and multiplier must be finite and positive")


def generate_failure(weather: np.ndarray, parameters: FailureParameters,
                     rng: np.random.Generator, dt_hours: float = 1.0) -> np.ndarray:
    failure_prob = [-math.expm1(-parameters.normal_rate_per_hour * dt_hours),
                    -math.expm1(-parameters.extreme_rate_per_hour * dt_hours)]
    repair_prob = [-math.expm1(-dt_hours / parameters.mean_repair_hours),
                   -math.expm1(-dt_hours / (parameters.mean_repair_hours * parameters.extreme_repair_multiplier))]
    # Stationary distribution of the discrete hourly chain, conditional on w[0].
    w0 = int(weather[0])
    failed_probability = failure_prob[w0] / (failure_prob[w0] + repair_prob[w0])
    available = bool(rng.random() >= failed_probability)
    draws = rng.random(len(weather) - 1)
    states = np.empty(len(weather), dtype=np.uint8)
    states[0] = available
    for t in range(1, len(weather)):
        w = int(weather[t])
        if draws[t - 1] < (failure_prob[w] if available else repair_prob[w]):
            available = not available
        states[t] = available
    return states
