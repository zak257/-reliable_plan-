"""Exact discrete CVaR, including partial probability mass at the VaR atom."""
import math
import numpy as np

from .eens_calculation import validate_losses


def calculate_CVaR(losses, alpha: float = 0.95, probabilities=None) -> float:
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    q, p = validate_losses(losses, probabilities)
    order = np.argsort(q)[::-1]
    tail_mass = 1 - alpha
    remaining, total = tail_mass, 0.0
    for i in order:
        take = min(float(p[i]), remaining)
        total += take * float(q[i])
        remaining -= take
        if remaining <= 0:
            break
    return total / tail_mass
