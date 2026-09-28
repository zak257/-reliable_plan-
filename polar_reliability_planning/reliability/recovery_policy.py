"""Frozen finite causal library; observation has no noise or future weather."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Observation:
    hour: int
    load_kw: float
    renewable_kw: float
    soc: float
    access_safe: bool
    states: tuple
    temperatures: tuple


@dataclass(frozen=True)
class RecoveryPolicy:
    name: str
    online_floor: int
    reserve_units: int
    recharge_soc: float
    recharge_kw: float
    heat_priority: bool
    keep_standby_warm: bool = True

    def __post_init__(self):
        if not self.name or self.online_floor < 0 or self.reserve_units < 0:
            raise ValueError("Invalid policy name or online target")
        if int(self.online_floor) != self.online_floor or int(self.reserve_units) != self.reserve_units:
            raise ValueError("Online targets must be integers")
        if not 0 <= self.recharge_soc <= 1 or not math.isfinite(self.recharge_kw) or self.recharge_kw < 0:
            raise ValueError("Invalid recharge rule")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def target(self, obs: Observation, module_kw: float, installed: int):
        charge = self.recharge_kw if obs.soc < self.recharge_soc else 0.0
        needed = max(0.0, obs.load_kw - obs.renewable_kw + charge)
        return min(installed, max(self.online_floor,
                                  math.ceil(needed / module_kw - 1e-10) + self.reserve_units))
