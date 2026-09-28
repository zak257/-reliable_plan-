"""Exact 1R1C update. All heat inputs are actual supplied kW."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThermalParameters:
    ua_kw_per_k: float
    c_kwh_per_k: float
    heater_max_kw: float
    ready_temperature_c: float
    initial_temperature_c: float
    efficiency: float = 1.0
    retained_generator_heat_kw: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Thermal parameters must be finite")
        if self.ua_kw_per_k < 0 or self.c_kwh_per_k <= 0 or self.heater_max_kw < 0:
            raise ValueError("Invalid UA, thermal capacity or heater rating")
        if not 0 < self.efficiency <= 1 or self.retained_generator_heat_kw < 0:
            raise ValueError("Invalid heater efficiency or retained heat")

    def step(self, temperature, ambient, supplied_heat_kw, running_count=0):
        heat = self.efficiency * supplied_heat_kw + running_count * self.retained_generator_heat_kw
        if self.ua_kw_per_k == 0:
            return temperature + heat / self.c_kwh_per_k
        decay = math.exp(-self.ua_kw_per_k / self.c_kwh_per_k)
        return (decay * temperature + (1 - decay) * ambient
                + (1 - decay) / self.ua_kw_per_k * heat)

    def command(self, temperature, ambient, running_count=0):
        """Minimum hourly average input reaching/holding the ready threshold."""
        free = self.step(temperature, ambient, 0, running_count)
        gain = (self.efficiency / self.c_kwh_per_k if self.ua_kw_per_k == 0 else
                -math.expm1(-self.ua_kw_per_k / self.c_kwh_per_k)
                / self.ua_kw_per_k * self.efficiency)
        return min(self.heater_max_kw, max(0.0, (self.ready_temperature_c - free) / gain))
