"""Risk of whole-path loss, retaining the original probability mass."""
import numpy as np
from .cvar_calculation import calculate_CVaR
from .eens_calculation import validate_losses


def risk_summary(losses, alpha=0.95, probabilities=None):
    q, p = validate_losses(losses, probabilities)
    order = np.argsort(q)
    quantile = min(len(q) - 1, int(np.searchsorted(np.cumsum(p[order]), alpha, side="left")))
    var = float(q[order[quantile]])
    return dict(eens_kwh=float(q @ p), cvar_kwh=calculate_CVaR(q, alpha, p), var_kwh=var,
                positive_loss_fraction=float(p[q > 1e-7].sum()),
                tail_effective_scenarios=(1 - alpha) / float(p @ p),
                zero_var_cvar_identity=bool(var == 0), samples=len(q))


def passes(risk, reliability):
    return (risk["eens_kwh"] <= reliability["eens_limit_kwh"] + 1e-7 and
            risk["cvar_kwh"] <= reliability["cvar_limit_kwh"] + 1e-7)


def partial_bounds(completed_losses, total_samples, demand_bound, alpha=0.95):
    remaining = total_samples - len(completed_losses)
    if remaining < 0 or demand_bound < 0:
        raise ValueError("Invalid partial risk inputs")
    return (risk_summary(list(completed_losses) + [0.0] * remaining, alpha),
            risk_summary(list(completed_losses) + [demand_bound] * remaining, alpha))
