import numpy as np


def validate_losses(losses, probabilities=None) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(losses, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all() or np.any(q < -1e-7):
        raise ValueError("Losses must be a nonempty finite nonnegative vector")
    q = np.maximum(q, 0)
    p = np.full(len(q), 1 / len(q)) if probabilities is None else np.asarray(probabilities, dtype=float)
    if p.shape != q.shape or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Probabilities must be nonnegative, match losses, and sum to 1")
    return q, p / p.sum()


def calculate_EENS(losses, probabilities=None) -> float:
    q, p = validate_losses(losses, probabilities)
    return float(p @ q)
