"""Append-only IID full-path samples with legacy-compatible random streams."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from polar_reliability_planning.data import CaseData
from polar_reliability_planning.data.cap_plan_loader import FAILABLE_COMPONENTS
from polar_reliability_planning.scenario_generation import Scenario, ScenarioPool
from polar_reliability_planning.scenario_generation.climate_generator import generate_weather
from polar_reliability_planning.scenario_generation.failure_generator import FailureParameters, generate_failure


class SharedScenarioStream:
    """Expose one common sample prefix to every capacity candidate.

    Sample ``s`` and module ``m`` always use SeedSequence([seed, component, s,
    m]); weather uses [seed, 0, s]. Extension generates *only* previously unseen
    samples. Paths are correlated within a sample through weather and chronology;
    complete paths are independent across sample indices.
    """

    def __init__(self, data: CaseData, seed: int, failures: dict,
                 weather: dict | None = None):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        unknown = set(failures) - set(FAILABLE_COMPONENTS)
        if unknown:
            raise ValueError(f"Unsupported failure components: {sorted(unknown)}")
        self.data, self.seed = data, int(seed)
        self.failures = deepcopy(failures)
        self.weather_config = deepcopy(weather or {})
        self._parameters = {k: FailureParameters(**self.failures.get(k, {})) for k in FAILABLE_COMPONENTS}
        self._extreme_factors = {k: float(self.weather_config.get(f"extreme_{k}_factor", 1.0))
                                 for k in ("wind", "pv", "load")}
        if not all(math.isfinite(v) and v >= 0 for v in self._extreme_factors.values()):
            raise ValueError("Weather factors must be finite and nonnegative")
        # Validate transition rates now, without advancing any sampled stream.
        if self.weather_config.get("enabled", False):
            for key, default in (("normal_to_extreme_rate_per_hour", 0.002),
                                 ("extreme_to_normal_rate_per_hour", 1 / 24)):
                value = float(self.weather_config.get(key, default))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Weather transition rates must be finite and nonnegative")
        maximum_load_factor = max(1.0, self._extreme_factors["load"])
        # Multiplication and summation do not commute in floating point:
        # sum(load * factor) can exceed sum(load) * factor by a few ulps.
        # Cover the actual pointwise envelope plus accumulation rounding, so a
        # valid all-shed Oracle incumbent never exceeds the declared support.
        envelope_energy = math.fsum(float(x) for x in data.load_kw * maximum_load_factor) * data.dt_hours
        nominal_energy = data.demand_kwh * maximum_load_factor
        base_bound = max(envelope_energy, nominal_energy)
        accumulation_guard = 4 * np.finfo(float).eps * (data.hours + 2)
        self.loss_bound = (float(np.nextafter(base_bound * (1 + accumulation_guard), math.inf))
                           if base_bound else 0.0)
        if not math.isfinite(self.loss_bound):
            raise ValueError("The deterministic loss bound must be finite")
        self._weather: list[np.ndarray] = []
        self._availability: dict[str, list[np.ndarray]] = {k: [] for k in FAILABLE_COMPONENTS}
        self._factors: dict[str, list[np.ndarray]] = {k: [] for k in ("wind", "pv", "load")}
        self._pool: ScenarioPool | None = None

    @property
    def samples(self) -> int:
        return len(self._weather)

    @property
    def metadata(self) -> dict:
        return {"seed": self.seed, "dt_hours": self.data.dt_hours,
                "failures": deepcopy(self.failures), "weather": deepcopy(self.weather_config),
                "initial_state": "hourly_stationary_conditional_on_initial_weather",
                "battery_cells": "ideal", "schema_version": 1,
                "stream_schema": "append_only_legacy_seedsequence_v1",
                "samples": self.samples, "loss_bound_kwh": self.loss_bound}

    @property
    def fingerprint(self) -> str:
        if self.samples:
            return self.to_pool().fingerprint
        # Empty streams cannot be represented by the legacy ScenarioPool.
        return hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def extend(self, total_count: int) -> "SharedScenarioStream":
        if isinstance(total_count, bool) or not isinstance(total_count, (int, np.integer)) or total_count < 0:
            raise ValueError("total_count must be a nonnegative integer")
        if total_count <= self.samples:
            return self
        data = self.data
        for s in range(self.samples, int(total_count)):
            weather = generate_weather(data.hours, data.dt_hours,
                np.random.default_rng(np.random.SeedSequence([self.seed, 0, s])), self.weather_config)
            paths = {}
            for component, key in enumerate(FAILABLE_COMPONENTS, start=1):
                count = data.unit_bounds[key][1]
                values = np.empty((count, data.hours), dtype=np.uint8)
                for module in range(count):
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, component, s, module]))
                    values[module] = generate_failure(weather, self._parameters[key], rng, data.dt_hours)
                values.flags.writeable = False
                paths[key] = values
            weather.flags.writeable = False
            self._weather.append(weather)
            for key, values in paths.items():
                self._availability[key].append(values)
            for key in self._factors:
                values = np.where(weather == 1, self._extreme_factors[key], 1.0)
                values.flags.writeable = False
                self._factors[key].append(values)
        self._pool = None
        return self

    def scenario(self, units: dict[str, int], index: int) -> Scenario:
        units = self.data.validate_units(units)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.samples:
            raise IndexError(index)
        active = {k: self._availability[k][index][:units[k]].sum(axis=0) * self.data.module_sizes[k]
                  for k in FAILABLE_COMPONENTS}
        return Scenario(int(index), active["wind"] * self.data.wind_pu * self._factors["wind"][index],
                        active["pv"] * self.data.pv_pu * self._factors["pv"][index], active["diesel"],
                        active["pcs"], self.data.load_kw * self._factors["load"][index],
                        self._availability["diesel"][index][:units["diesel"]])

    def to_pool(self) -> ScenarioPool:
        if not self.samples:
            raise ValueError("Extend the stream before exporting a scenario pool")
        if self._pool is None:
            self._pool = ScenarioPool({k: np.stack(v) for k, v in self._availability.items()},
                np.stack(self._weather), np.full(self.samples, 1.0 / self.samples),
                np.stack(self._factors["wind"]), np.stack(self._factors["pv"]),
                np.stack(self._factors["load"]), self.metadata)
        return self._pool

    def save(self, path: str | Path):
        self.to_pool().save(path)
