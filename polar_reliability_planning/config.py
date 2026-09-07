"""Small, explicit configuration layer; TOML needs no extra venv dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import tomllib

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class UnitCommitmentOptions:
    enabled: bool = True
    min_output_fraction: float = 0.2
    min_up_hours: float = 3.0
    min_down_hours: float = 3.0
    # cap_plan does not charge starts; a calibrated startup cost can be added here.
    startup_cost_yuan: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.min_output_fraction) or not 0 <= self.min_output_fraction <= 1:
            raise ValueError("min_output_fraction must be in [0, 1]")
        for value in (self.min_up_hours, self.min_down_hours, self.startup_cost_yuan):
            if not math.isfinite(value) or value < 0:
                raise ValueError("UC duration and startup cost must be finite and nonnegative")


@dataclass(frozen=True)
class SolverOptions:
    time_limit: float = 120.0
    oracle_time_limit: float = 60.0
    mip_gap: float = 1e-4
    threads: int = 2
    seed: int = 0
    output_flag: bool = False

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.time_limit, self.oracle_time_limit)):
            raise ValueError("Solver time limits must be finite and positive")
        if not math.isfinite(self.mip_gap) or not 0 <= self.mip_gap < 1:
            raise ValueError("mip_gap must be in [0, 1)")
        if self.threads < 0 or self.seed < 0:
            raise ValueError("threads and seed must be nonnegative")


@dataclass(frozen=True)
class ReliabilityLimits:
    eens_kwh: float
    cvar_kwh: float | None = None
    alpha: float = 0.95
    tolerance_kwh: float = 1e-5

    def __post_init__(self):
        for value in (self.eens_kwh, self.tolerance_kwh):
            if not math.isfinite(value) or value < 0:
                raise ValueError("EENS limit and comparison tolerance must be finite and nonnegative")
        if self.cvar_kwh is not None and (not math.isfinite(self.cvar_kwh) or self.cvar_kwh < 0):
            raise ValueError("CVaR limit must be finite and nonnegative")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha < 1:
            raise ValueError("CVaR alpha must be in [0, 1)")


def read_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)
