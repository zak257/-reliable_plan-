"""Optional common weather chain. Disabled for the baseline."""
import math
import numpy as np


def generate_weather(hours: int, dt_hours: float, rng: np.random.Generator, config: dict) -> np.ndarray:
    if not config.get("enabled", False):
        return np.zeros(hours, dtype=np.uint8)
    rates = [float(config.get("normal_to_extreme_rate_per_hour", 0.002)),
             float(config.get("extreme_to_normal_rate_per_hour", 1 / 24))]
    if not all(math.isfinite(x) and x >= 0 for x in rates):
        raise ValueError("Weather transition rates must be finite and nonnegative")
    p01, p10 = [-math.expm1(-x * dt_hours) for x in rates]
    probability_extreme = p01 / (p01 + p10) if p01 + p10 else 0.0
    state = int(rng.random() < probability_extreme)
    draws = rng.random(max(0, hours - 1))
    result = np.empty(hours, dtype=np.uint8)
    result[0] = state
    for t in range(1, hours):
        if draws[t - 1] < (p10 if state else p01):
            state = 1 - state
        result[t] = state
    return result
