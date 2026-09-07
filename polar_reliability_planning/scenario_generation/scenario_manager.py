"""Persist one sample set and select nested prefixes of module trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.cap_plan_loader import CaseData, FAILABLE_COMPONENTS
from .climate_generator import generate_weather
from .failure_generator import FailureParameters, generate_failure


@dataclass(frozen=True)
class Scenario:
    index: int
    wind_available_kw: np.ndarray
    pv_available_kw: np.ndarray
    diesel_available_kw: np.ndarray
    pcs_available_kw: np.ndarray
    load_kw: np.ndarray
    diesel_module_availability: np.ndarray | None = None

    @classmethod
    def nominal(cls, data: CaseData, units: dict[str, int]) -> "Scenario":
        capacity = data.capacities(units)
        return cls(-1, capacity["wind"] * data.wind_pu, capacity["pv"] * data.pv_pu,
                   np.full(data.hours, capacity["diesel"]), np.full(data.hours, capacity["pcs"]), data.load_kw,
                   np.ones((units["diesel"], data.hours), dtype=np.uint8))


class ScenarioPool:
    def __init__(self, availability: dict[str, np.ndarray], weather: np.ndarray,
                 probabilities: np.ndarray, wind_factor: np.ndarray, pv_factor: np.ndarray,
                 load_factor: np.ndarray, metadata: dict | None = None):
        raw_weather = np.asarray(weather)
        if raw_weather.ndim != 2 or min(raw_weather.shape) < 1 or not np.isin(raw_weather, (0, 1)).all():
            raise ValueError("Weather must have shape (samples, hours) with states 0/1")
        self.weather = raw_weather.astype(np.uint8, copy=True)
        self.samples, self.hours = self.weather.shape
        if set(availability) != set(FAILABLE_COMPONENTS):
            raise ValueError(f"Availability must contain exactly {FAILABLE_COMPONENTS}")
        self.availability = {}
        for key, a in availability.items():
            a = np.asarray(a)
            if a.ndim != 3 or a.shape[0] != self.samples or a.shape[2] != self.hours or not np.isin(a, (0, 1)).all():
                raise ValueError(f"Invalid 0/1 module trajectory array for {key}")
            self.availability[key] = a.astype(np.uint8, copy=True)
        p = np.array(probabilities, dtype=float, copy=True)
        if p.shape != (self.samples,) or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("Scenario probabilities must be positive and sum to 1")
        self.probabilities = p / p.sum()
        for name, value in (("wind_factor", wind_factor), ("pv_factor", pv_factor), ("load_factor", load_factor)):
            value = np.array(value, dtype=float, copy=True)
            if value.shape != self.weather.shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f"Invalid {name} array")
            setattr(self, name, value)
        self.metadata = dict(metadata or {})
        for a in (self.weather, self.probabilities, self.wind_factor, self.pv_factor, self.load_factor,
                  *self.availability.values()):
            a.flags.writeable = False
        self.fingerprint = self._fingerprint()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, a in sorted(self._arrays().items()):
            digest.update(name.encode())
            digest.update(str(a.shape).encode())
            digest.update(a.dtype.str.encode())
            digest.update(a.tobytes())
        return digest.hexdigest()

    def _arrays(self) -> dict[str, np.ndarray]:
        return {**{f"available_{k}": v for k, v in self.availability.items()},
                "weather": self.weather, "probabilities": self.probabilities,
                "wind_factor": self.wind_factor, "pv_factor": self.pv_factor, "load_factor": self.load_factor}

    def check_data(self, data: CaseData):
        if data.hours != self.hours:
            raise ValueError("Scenario and case horizons differ")
        if "dt_hours" in self.metadata and self.metadata["dt_hours"] != data.dt_hours:
            raise ValueError("Scenario and case time steps differ")
        for key in FAILABLE_COMPONENTS:
            if self.availability[key].shape[1] < data.unit_bounds[key][1]:
                raise ValueError(f"Scenario pool has too few {key} module trajectories")

    def scenario(self, data: CaseData, units: dict[str, int], index: int) -> Scenario:
        units = data.validate_units(units)
        if not 0 <= index < self.samples:
            raise IndexError(index)
        # The first n installed modules have the SAME histories for every candidate.
        active = {k: self.availability[k][index, :units[k], :].sum(axis=0) * data.module_sizes[k]
                  for k in FAILABLE_COMPONENTS}
        return Scenario(index, active["wind"] * data.wind_pu * self.wind_factor[index],
                        active["pv"] * data.pv_pu * self.pv_factor[index], active["diesel"],
                        active["pcs"], data.load_kw * self.load_factor[index],
                        self.availability["diesel"][index, :units["diesel"], :])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **self._arrays(), metadata=np.array(json.dumps(self.metadata, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioPool":
        with np.load(path, allow_pickle=False) as archive:
            return cls({k: archive[f"available_{k}"] for k in FAILABLE_COMPONENTS},
                       archive["weather"], archive["probabilities"], archive["wind_factor"],
                       archive["pv_factor"], archive["load_factor"], json.loads(str(archive["metadata"].item())))


def generate_pool(data: CaseData, samples: int, seed: int, failures: dict,
                  weather_config: dict | None = None) -> ScenarioPool:
    if samples < 1 or int(samples) != samples or seed < 0 or int(seed) != seed:
        raise ValueError("samples must be positive and seed nonnegative integers")
    unknown = set(failures) - set(FAILABLE_COMPONENTS)
    if unknown:
        raise ValueError(f"Unsupported failure components: {sorted(unknown)}; battery cells are ideal in this baseline")
    weather_config = dict(weather_config or {})
    weather = np.stack([generate_weather(data.hours, data.dt_hours,
                         np.random.default_rng(np.random.SeedSequence([seed, 0, s])), weather_config)
                        for s in range(samples)])
    availability = {}
    for component_index, key in enumerate(FAILABLE_COMPONENTS, start=1):
        parameters = FailureParameters(**failures.get(key, {}))
        count = data.unit_bounds[key][1]
        paths = np.empty((samples, count, data.hours), dtype=np.uint8)
        for s in range(samples):
            for module in range(count):
                # Stream identifiers preserve paths when samples/module bounds grow.
                rng = np.random.default_rng(np.random.SeedSequence([seed, component_index, s, module]))
                paths[s, module] = generate_failure(weather[s], parameters, rng, data.dt_hours)
        availability[key] = paths
    factors = {key: np.where(weather == 1, float(weather_config.get(f"extreme_{key}_factor", 1.0)), 1.0)
               for key in ("wind", "pv", "load")}
    return ScenarioPool(availability, weather, np.full(samples, 1 / samples), factors["wind"], factors["pv"],
                        factors["load"], {"seed": seed, "dt_hours": data.dt_hours, "failures": failures,
                                         "weather": weather_config, "initial_state": "hourly_stationary_conditional_on_initial_weather",
                                         "battery_cells": "ideal", "schema_version": 1})
