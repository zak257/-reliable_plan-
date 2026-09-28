"""A compact synthetic resilience-planning model with load classes and UPS sizing.

This module is deliberately separate from Recovery-v1.  It supplies a runnable
research example for the new requirements: core/rigid/flexible demand, an
optimised UPS size, grid and renewable-bus faults, and grid-forming PCS and
diesel resources.  The data and failure parameters are synthetic and are
labelled as such in every output.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from collections import deque

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticYear:
    timestamps: tuple[str, ...]
    core_kw: np.ndarray
    rigid_kw: np.ndarray
    flex_interruptible_kw: np.ndarray
    flex_shiftable_kw: np.ndarray
    ambient_c: np.ndarray
    wind_clean_pu: np.ndarray
    pv_pu: np.ndarray
    extreme_weather: np.ndarray
    weather_risk: np.ndarray

    @property
    def hours(self) -> int:
        return len(self.timestamps)

    @property
    def total_kw(self) -> np.ndarray:
        return self.core_kw + self.rigid_kw + self.flex_interruptible_kw + self.flex_shiftable_kw

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "core_kw": self.core_kw,
            "rigid_kw": self.rigid_kw,
            "flex_interruptible_kw": self.flex_interruptible_kw,
            "flex_shiftable_kw": self.flex_shiftable_kw,
            "load_total_kw": self.total_kw,
            "ambient_c": self.ambient_c,
            "wind_clean_pu": self.wind_clean_pu,
            "pv_pu": self.pv_pu,
            "extreme_weather": self.extreme_weather.astype(int),
            "weather_risk": self.weather_risk.astype(int),
        })


@dataclass(frozen=True)
class Capacity:
    wind_kw: float
    pv_kw: float
    diesel_units: int
    battery_kwh: float
    pcs_kw: float
    ups_kwh: float
    ups_kw: float

    @property
    def diesel_kw(self) -> float:
        return 100.0 * self.diesel_units

    @property
    def investment_yuan(self) -> float:
        return (
            1500.0 * self.wind_kw
            + 1000.0 * self.pv_kw
            + 1800.0 * self.diesel_kw
            + 250.0 * self.battery_kwh
            + 400.0 * self.pcs_kw
            + 450.0 * self.ups_kwh
            + 250.0 * self.ups_kw
        )


@dataclass(frozen=True)
class ResilienceConfig:
    hours: int = 8760
    seed: int = 20260920
    training_scenarios: int = 16
    validation_scenarios: int = 32
    dt_hours: float = 1.0
    diesel_unit_kw: float = 100.0
    diesel_start_delay_h: int = 2
    diesel_mttf_h: float = 1662.0
    diesel_start_failure_probability: float = 0.0013
    diesel_repair_mean_h: float = 37.0
    pcs_mttf_h: float = 4000.0
    pcs_repair_mean_h: float = 24.0
    ups_mttf_h: float = 5000.0
    ups_repair_mean_h: float = 24.0
    grid_fault_rate_per_h: float = 0.00065
    grid_fault_mean_h: float = 10.0
    renewable_bus_fault_rate_per_h: float = 0.00085
    renewable_bus_fault_mean_h: float = 8.0
    battery_initial_soc: float = 0.60
    battery_min_soc: float = 0.10
    ups_initial_soc: float = 0.50
    ups_min_soc: float = 0.05
    ups_low_reserve_hours: float = 2.0
    ups_high_reserve_hours: float = 8.0
    # Deterministic core-load bridge requirement.  It represents the declared
    # maximum diesel recovery window used to size a UPS before stochastic
    # scenarios are evaluated.
    core_recovery_hours: float = 8.0
    core_power_margin: float = 1.50
    high_risk_reserve_soc: float = 0.90
    low_risk_reserve_soc: float = 0.60
    battery_efficiency: float = 0.95
    # A storage PCS is not installed without a battery.  The one-hour
    # minimum duration is the least restrictive physical coupling used by
    # this synthetic model; longer durations can be supplied for a site study.
    battery_pcs_min_duration_h: float = 1.0
    pcs_gfm_min_power_kw: float = 1.0
    loss_of_load_cost_yuan_per_kwh: float = 1000.0
    ups_efficiency: float = 0.98
    rigid_eens_limit_kwh: float = 100.0
    rigid_cvar_limit_kwh: float = 1000.0
    alpha: float = 0.95
    flex_adjustment_cost_yuan_per_kwh: float = 0.12
    flex_shift_window_h: int = 24
    grid_energy_cost_yuan_per_kwh: float = 0.65
    diesel_cost_yuan_per_kwh: float = 0.95
    fuel_kwh_per_kg: float = 4.5
    random_failures_enabled: bool = True
    all_pcs_grid_forming: bool = True
    all_diesel_grid_forming: bool = True


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    seed: int
    risk_class: str
    compound_extreme: bool
    weather_enabled: bool
    random_enabled: bool
    grid_fault: np.ndarray
    renewable_bus_fault: np.ndarray
    diesel_run_draws: np.ndarray
    diesel_start_draws: np.ndarray
    pcs_draws: np.ndarray
    ups_draws: np.ndarray
    repair_draws: np.ndarray


@dataclass
class SimulationResult:
    valid: bool
    core_unserved_kwh: float
    rigid_unserved_kwh: float
    cvar_loss_kwh: float
    flex_adjusted_kwh: float
    operating_cost_yuan: float
    metrics: dict
    trace: list[dict] = field(default_factory=list)


@dataclass
class PlanResult:
    status: str
    selected: dict | None
    evaluations: list[dict]
    validation: dict | None
    metadata: dict


CAPACITY_STEPS = {
    "wind_kw": 100.0, "pv_kw": 100.0, "diesel_units": 1,
    "battery_kwh": 50.0, "pcs_kw": 50.0, "ups_kwh": 50.0, "ups_kw": 50.0,
}


def assert_discrete_capacity(cap: Capacity) -> None:
    for name, step in CAPACITY_STEPS.items():
        value = getattr(cap, name)
        if not math.isfinite(value) or value < 0 or abs(value / step - round(value / step)) > 1e-9:
            raise ValueError(f"{name}={value} is not a nonnegative multiple of {step}")


def day_ahead_weather_risk(extreme: np.ndarray) -> np.ndarray:
    """Synthetic perfect next-calendar-day Boolean, released at 00:00 daily.

    No event hour, duration, later-day weather or equipment fault is exposed.
    This is an assumed forecast, not an observed operational forecast.
    """
    hours = len(extreme)
    risk = np.zeros(hours, dtype=bool)
    for start in range(0, hours, 24):
        risk[start:min(hours, start + 24)] = bool(np.any(extreme[start + 24:start + 48]))
    return risk


def storage_pcs_coupling_pass(cap: Capacity, cfg: ResilienceConfig) -> bool:
    """Check that a storage PCS has a connected battery with energy backing."""
    if cap.pcs_kw <= 1e-9:
        return True
    return cap.battery_kwh + 1e-9 >= cfg.battery_pcs_min_duration_h * cap.pcs_kw


def required_ups_bridge_kwh(year: SyntheticYear, cfg: ResilienceConfig) -> float:
    """Return the nominal UPS energy needed for the declared core bridge.

    ``core_recovery_hours`` is an energy requirement at the core bus.  The
    installed nameplate must be larger because an unpredictable fault can
    occur while the UPS is held at the low-risk reserve SOC, and only the
    energy between that reserve and the minimum SOC is usable.
    """
    peak_core = float(np.max(year.core_kw))
    usable_fraction = max(
        cfg.low_risk_reserve_soc - cfg.ups_min_soc, 1e-9
    ) * cfg.ups_efficiency
    return peak_core * cfg.core_recovery_hours / usable_fraction


def required_ups_bridge_kw(year: SyntheticYear, cfg: ResilienceConfig) -> float:
    """Return the UPS power nameplate required by the core-load margin."""
    return float(np.max(year.core_kw)) * cfg.core_power_margin


def generate_synthetic_year(hours: int = 8760, seed: int = 20260920) -> SyntheticYear:
    """Generate a plausible hourly load/weather year for the new model.

    The output is intentionally synthetic.  It has an explicit 4-class demand
    structure and predictable weather-risk windows so a forecast-aware UPS
    reserve rule can be tested without claiming a field measurement.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Etc/GMT-8")
    h = np.arange(hours, dtype=float)
    hour = h % 24.0
    day = h / 24.0
    seasonal = np.cos(2 * np.pi * (day - 15.0) / 365.0)
    workday = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 7.0) / 24.0)
    workday = np.clip(workday, 0.0, 1.0)

    core = 32.0 + 4.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 3.0) / 24.0))
    core += 1.5 * seasonal + rng.normal(0.0, 0.7, hours)
    rigid = 128.0 + 20.0 * workday + 10.0 * seasonal + rng.normal(0.0, 3.0, hours)
    interruptible = 28.0 + 10.0 * workday + rng.normal(0.0, 1.5, hours)
    shiftable = 22.0 + 8.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (hour - 10.0) / 24.0))
    core = np.maximum(core, 20.0)
    rigid = np.maximum(rigid, 80.0)
    interruptible = np.maximum(interruptible, 8.0)
    shiftable = np.maximum(shiftable, 5.0)

    ambient = 4.0 + 10.0 * seasonal + 2.0 * np.sin(2 * np.pi * (hour - 14.0) / 24.0)
    ambient += rng.normal(0.0, 1.2, hours)
    wind_speed = 7.5 + 2.0 * np.sin(2 * np.pi * (day - 40.0) / 365.0)
    wind_speed += rng.normal(0.0, 1.4, hours)
    wind_speed = np.maximum(wind_speed, 0.5)

    extreme = np.zeros(hours, dtype=bool)
    # Four predictable winter storm windows.  Their exact dates are part of
    # the synthetic scenario contract, not an observation claim.
    for start_day, duration in ((35, 30), (105, 22), (235, 26), (315, 34)):
        start = int(start_day * 24)
        stop = min(hours, start + duration)
        extreme[start:stop] = True
    wind_speed[extreme] = np.maximum(wind_speed[extreme], 16.0)
    ambient[extreme] = np.minimum(ambient[extreme], -1.0)

    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    pv = daylight * (0.70 + 0.18 * seasonal) * np.clip(1.0 + rng.normal(0.0, 0.08, hours), 0.35, 1.05)
    pv = np.clip(pv, 0.0, 1.0)
    wind_pu = np.clip((wind_speed - 3.0) / 10.0, 0.0, 1.0)
    wind_pu *= np.clip(0.95 + rng.normal(0.0, 0.04, hours), 0.55, 1.05)
    wind_pu = np.clip(wind_pu, 0.0, 1.0)

    # Day-ahead forecast contract.  The signal released during calendar day d
    # is one Boolean: whether calendar day d+1 is extreme.  It does not
    # reveal the exact hour of tomorrow's event, and it never looks beyond
    # tomorrow.  A controller can still observe the current hour's weather.
    weather_risk = day_ahead_weather_risk(extreme)

    return SyntheticYear(
        timestamps=tuple(index.astype(str)),
        core_kw=core,
        rigid_kw=rigid,
        flex_interruptible_kw=interruptible,
        flex_shiftable_kw=shiftable,
        ambient_c=ambient,
        wind_clean_pu=wind_pu,
        pv_pu=pv,
        extreme_weather=extreme,
        weather_risk=weather_risk,
    )


def _event_process(rng: np.random.Generator, hours: int, rate: float, mean_h: float) -> np.ndarray:
    active = np.zeros(hours, dtype=bool)
    t = 0
    while t < hours:
        if rng.random() < rate:
            duration = max(1, int(rng.geometric(1.0 / max(1.0, mean_h))))
            active[t:min(hours, t + duration)] = True
            t += duration
        else:
            t += 1
    return active


def make_scenarios(year: SyntheticYear, cfg: ResilienceConfig, count: int, seed: int,
                   risk_class: str, scenario_id_start: int = 0) -> list[Scenario]:
    """Create common random paths without imposing a planning unit limit.

    One random column is materialised in each scenario.  When a capacity has
    more units, the simulator creates stable deterministic values for the
    additional units on demand, so scenario generation cannot exclude a
    capacity from the planning search.
    """
    scenarios: list[Scenario] = []
    draw_units = 1
    for scenario_id in range(count):
        scenario_seed = int(seed + scenario_id * 1009)
        rng = np.random.default_rng(scenario_seed)
        compound_extreme = risk_class in ("compound_extreme", "extreme_compound")
        random_enabled = risk_class in ("random", "joint") and cfg.random_failures_enabled
        weather_enabled = risk_class in ("weather", "joint", "compound_extreme", "extreme_compound")
        grid_fault = (
            year.extreme_weather.copy()
            if compound_extreme
            else (_event_process(rng, year.hours, cfg.grid_fault_rate_per_h, cfg.grid_fault_mean_h) if random_enabled else np.zeros(year.hours, bool))
        )
        renewable_bus_fault = (
            year.extreme_weather.copy()
            if compound_extreme
            else (_event_process(rng, year.hours, cfg.renewable_bus_fault_rate_per_h, cfg.renewable_bus_fault_mean_h) if random_enabled else np.zeros(year.hours, bool))
        )
        shape = (year.hours, draw_units)
        scenarios.append(Scenario(
            scenario_id=scenario_id_start + scenario_id,
            seed=scenario_seed,
            risk_class=risk_class,
            compound_extreme=compound_extreme,
            weather_enabled=weather_enabled,
            random_enabled=random_enabled,
            grid_fault=grid_fault,
            renewable_bus_fault=renewable_bus_fault,
            diesel_run_draws=rng.random(shape),
            diesel_start_draws=rng.random(shape),
            pcs_draws=rng.random(year.hours),
            ups_draws=rng.random(year.hours),
            repair_draws=rng.random((year.hours, draw_units)),
        ))
    return scenarios


def _unit_draw(draws: np.ndarray, t: int, unit: int, scenario_seed: int, stream: int) -> float:
    """Return a reproducible per-unit random value with no unit-count bound."""
    if unit < draws.shape[1]:
        return float(draws[t, unit])
    # SplitMix64-style integer mixing avoids allocating a random matrix for an
    # arbitrary number of diesel units while preserving common paths for the
    # materialised columns.
    value = (
        int(scenario_seed)
        + 0x9E3779B97F4A7C15 * (int(t) + 1)
        + 0xBF58476D1CE4E5B * (int(unit) + 1)
        + 0x94D049BB133111EB * int(stream)
    ) & ((1 << 64) - 1)
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    value ^= value >> 31
    return float(value >> 11) / float(1 << 53)


def _repair_duration(u: float, mean_h: float) -> int:
    return max(1, int(math.floor(math.log1p(-min(float(u), 1 - 1e-12)) / math.log1p(-1.0 / max(1.0, mean_h)))) + 1)


def _cvar(losses: Iterable[float], alpha: float) -> float:
    values = np.sort(np.asarray(list(losses), dtype=float))[::-1]
    if len(values) == 0:
        return 0.0
    tail = (1.0 - alpha) * len(values)
    whole = int(math.floor(tail))
    remainder = tail - whole
    total = float(values[:whole].sum())
    if remainder > 1e-12 and whole < len(values):
        total += remainder * float(values[whole])
    return total / max(tail, 1e-12)


def _risk_summary(losses: list[float], alpha: float) -> dict:
    arr = np.asarray(losses, dtype=float)
    return {
        "eens_kwh": float(arr.mean()) if len(arr) else 0.0,
        "cvar_kwh": _cvar(arr, alpha),
        "max_kwh": float(arr.max()) if len(arr) else 0.0,
        "positive_fraction": float(np.mean(arr > 1e-9)) if len(arr) else 0.0,
        "samples": int(len(arr)),
    }


def simulate(year: SyntheticYear, cap: Capacity, scenario: Scenario, cfg: ResilienceConfig,
             trace: bool = False) -> SimulationResult:
    """Run one joint weather/fault path with a forecast-aware UPS reserve rule."""
    hcount = year.hours
    if cfg.dt_hours != 1.0:
        raise ValueError("this hourly simulator requires dt_hours=1")
    if cfg.diesel_unit_kw != 100.0:
        raise ValueError("diesel research module must remain 100 kW")
    diesel_healthy = np.ones(cap.diesel_units, dtype=bool)
    diesel_online = np.zeros(cap.diesel_units, dtype=bool)
    diesel_start_at = np.full(cap.diesel_units, -1, dtype=int)
    diesel_repair_until = np.full(cap.diesel_units, -1, dtype=int)
    pcs_repair_until = -1
    ups_repair_until = -1
    battery = cap.battery_kwh * cfg.battery_initial_soc
    ups_energy = cap.ups_kwh * cfg.ups_initial_soc
    battery_min = cap.battery_kwh * cfg.battery_min_soc
    ups_min = cap.ups_kwh * cfg.ups_min_soc
    core_loss = rigid_loss = flex_adjusted = fuel_cost = grid_cost = 0.0
    # UPS activation is a subset diagnostic.  Its components are charged in
    # the ordinary loss-cost expression below exactly once.
    ups_activation_rigid_loss = 0.0
    ups_activation_flex_loss = 0.0
    flex_forced_loss = flex_shifted = flex_repaid = 0.0
    shift_queue: deque = deque()
    # Synthetic shiftable-load equipment rating: twice its observed peak.
    shift_power_limit = 2.0 * float(np.max(year.flex_shiftable_kw))
    valid = True
    max_balance_residual = 0.0
    starts = start_failures = diesel_failures = 0
    bus_fault_hours = renewable_fault_hours = 0
    ups_discharge = ups_charge = battery_discharge = battery_charge = 0.0
    trace_rows: list[dict] = []
    run_probability = 1.0 - math.exp(-cfg.dt_hours / cfg.diesel_mttf_h)
    pcs_probability = 1.0 - math.exp(-cfg.dt_hours / cfg.pcs_mttf_h)
    ups_probability = 1.0 - math.exp(-cfg.dt_hours / cfg.ups_mttf_h)

    for t in range(hcount):
        if t >= pcs_repair_until:
            pcs_repair_until = -1
        if t >= ups_repair_until:
            ups_repair_until = -1
        for i in range(cap.diesel_units):
            if not diesel_healthy[i] and t >= diesel_repair_until[i]:
                diesel_healthy[i] = True
                diesel_repair_until[i] = -1
        random_enabled = scenario.random_enabled
        grid_fault = bool(scenario.grid_fault[t])
        renewable_fault = bool(scenario.renewable_bus_fault[t])
        weather_stress = bool(scenario.weather_enabled and year.extreme_weather[t])
        weather_risk = bool(
            scenario.weather_enabled and (year.weather_risk[t] or weather_stress)
        )
        bus_fault_hours += int(grid_fault)
        renewable_fault_hours += int(renewable_fault)

        if random_enabled and pcs_repair_until < 0 and scenario.pcs_draws[t] < pcs_probability:
            pcs_repair_until = t + _repair_duration(scenario.repair_draws[t, 0], cfg.pcs_repair_mean_h)
        if random_enabled and ups_repair_until < 0 and scenario.ups_draws[t] < ups_probability:
            ups_repair_slot = 1 % max(1, scenario.repair_draws.shape[1])
            ups_repair_until = t + _repair_duration(
                _unit_draw(scenario.repair_draws, t, ups_repair_slot, scenario.seed, 41),
                cfg.ups_repair_mean_h,
            )
        for i in range(cap.diesel_units):
            if random_enabled and diesel_online[i] and diesel_healthy[i] and _unit_draw(
                scenario.diesel_run_draws, t, i, scenario.seed, 11
            ) < run_probability:
                diesel_online[i] = False
                diesel_healthy[i] = False
                diesel_repair_until[i] = t + _repair_duration(
                    _unit_draw(scenario.repair_draws, t, i, scenario.seed, 37),
                    cfg.diesel_repair_mean_h,
                )
                diesel_failures += 1

        pcs_available = cap.pcs_kw if pcs_repair_until < 0 and storage_pcs_coupling_pass(cap, cfg) else 0.0
        ups_available = ups_repair_until < 0
        renewable_factor = 0.35 if weather_stress else 1.0
        renewable_available = 0.0 if renewable_fault else (
            cap.wind_kw * year.wind_clean_pu[t] * renewable_factor
            + cap.pv_kw * year.pv_pu[t] * (0.65 if weather_stress else 1.0)
        )

        # Weather is forecastable: pre-start one grid-forming diesel before a
        # high-risk window. A random bus fault cannot be anticipated.
        # Only today's observations and the released next-day Boolean enter
        # this rule.  No later event date or future fault is consulted.
        healthy_standby = [i for i in range(cap.diesel_units) if diesel_healthy[i] and not diesel_online[i] and diesel_start_at[i] < 0]
        pending = int(np.sum(diesel_start_at >= 0))
        if grid_fault:
            target_diesel = min(cap.diesel_units, max(1, int(math.ceil((year.core_kw[t] + year.rigid_kw[t]) / cfg.diesel_unit_kw))))
        elif weather_risk:
            target_diesel = min(cap.diesel_units, int(math.ceil(
                (year.core_kw[t] + year.rigid_kw[t]) / cfg.diesel_unit_kw
            )))
        else:
            target_diesel = 0
        online_ids = np.flatnonzero(diesel_online)
        for i in online_ids[target_diesel:]:
            diesel_online[i] = False
        if target_diesel == 0:
            diesel_start_at[:] = -1
            pending = 0
        needed = max(0, target_diesel - int(np.sum(diesel_online)) - pending)
        for i in healthy_standby[:needed]:
            diesel_start_at[i] = t + cfg.diesel_start_delay_h + (1 if weather_stress else 0)

        for i in range(cap.diesel_units):
            if diesel_start_at[i] == t:
                diesel_start_at[i] = -1
                starts += 1
                if random_enabled and _unit_draw(
                    scenario.diesel_start_draws, t, i, scenario.seed, 23
                ) < cfg.diesel_start_failure_probability:
                    diesel_healthy[i] = False
                    diesel_repair_until[i] = t + _repair_duration(
                        _unit_draw(scenario.repair_draws, t, i, scenario.seed, 37),
                        cfg.diesel_repair_mean_h,
                    )
                    start_failures += 1
                else:
                    diesel_online[i] = True

        # Deferred tasks are explicit energy obligations, not free shedding.
        expired_shift = 0.0
        while shift_queue and shift_queue[0][0] <= t:
            expired_shift += shift_queue.popleft()[1]
        pending_shift = sum(item[1] for item in shift_queue)
        core = float(year.core_kw[t])
        rigid = float(year.rigid_kw[t])
        interruptible = float(year.flex_interruptible_kw[t])
        shift_new = float(year.flex_shiftable_kw[t])
        shift_request = min(shift_power_limit, shift_new + pending_shift)
        flex = interruptible + shift_request
        total_load = core + rigid + flex
        islanded = grid_fault
        diesel_power = min(float(np.sum(diesel_online)) * cfg.diesel_unit_kw, total_load)
        battery_power_available = max(0.0, (battery - battery_min) * cfg.battery_efficiency)
        battery_dis = 0.0
        if islanded and pcs_available > 1e-9:
            deficit = total_load - renewable_available - diesel_power
            if deficit > 0:
                battery_dis = min(pcs_available, battery_power_available, deficit)
            elif diesel_power <= 1e-9:
                # Inject real power and curtail renewable generation to balance.
                battery_dis = min(pcs_available, battery_power_available,
                                  cfg.pcs_gfm_min_power_kw, total_load)
        gfm_power = ((battery_dis if cfg.all_pcs_grid_forming else 0.0)
                     + (diesel_power if cfg.all_diesel_grid_forming else 0.0))
        main_bus_live = (not islanded) or gfm_power > 1e-9
        if not main_bus_live:
            battery_dis = diesel_power = 0.0
        renewable_on_bus = renewable_available if main_bus_live else 0.0
        base_supply = renewable_on_bus + diesel_power + battery_dis
        grid_import = max(0.0, total_load - base_supply) if not islanded else 0.0
        supply = base_supply + grid_import
        core_from_main = min(core, supply)
        remaining = max(0.0, supply - core_from_main)
        ups_dis = 0.0
        if ups_available and cap.ups_kw > 0:
            ups_dis = min(core - core_from_main, cap.ups_kw,
                          max(0.0, (ups_energy - ups_min) * cfg.ups_efficiency))
        core_shed = max(0.0, core - core_from_main - ups_dis)
        rigid_served = min(rigid, remaining)
        remaining -= rigid_served
        rigid_shed = max(0.0, rigid - rigid_served)
        # Repay oldest shiftable tasks first, then today's shiftable work;
        # interruptible demand has the lowest supply priority.
        shift_served = min(shift_request, remaining)
        remaining -= shift_served
        interrupt_served = min(interruptible, remaining)
        work_left = shift_served
        repaid_now = 0.0
        while shift_queue and work_left > 1e-9:
            due, energy = shift_queue[0]
            done = min(energy, work_left)
            work_left -= done
            repaid_now += done
            if energy - done <= 1e-9:
                shift_queue.popleft()
            else:
                shift_queue[0] = (due, energy - done)
        shift_new_missing = max(0.0, shift_new - work_left)
        interrupt_missing = max(0.0, interruptible - interrupt_served)
        # A dead bus / emergency UPS transfer is an involuntary outage.  It
        # cannot be relabelled as a cheap, contracted flexibility service.
        emergency = not main_bus_live or ups_dis > 1e-9
        flex_forced_now = expired_shift
        flex_adjust = 0.0
        deferred_now = 0.0
        if emergency:
            flex_forced_now += interrupt_missing + shift_new_missing
        else:
            flex_adjust = interrupt_missing
            if shift_new_missing > 1e-9:
                shift_queue.append((t + cfg.flex_shift_window_h, shift_new_missing))
                deferred_now = shift_new_missing
        if ups_dis > 1e-9:
            ups_activation_rigid_loss += rigid_shed
            ups_activation_flex_loss += flex_forced_now
        flex_forced_loss += flex_forced_now
        flex_shifted += deferred_now
        flex_repaid += repaid_now
        battery -= battery_dis / cfg.battery_efficiency
        ups_energy -= ups_dis / cfg.ups_efficiency
        main_served = core_from_main + rigid_served + shift_served + interrupt_served
        surplus = max(0.0, base_supply - main_served)

        risk_reserve_soc = cfg.high_risk_reserve_soc if weather_risk else cfg.low_risk_reserve_soc
        reserve_hours = cfg.ups_high_reserve_hours if weather_risk else cfg.ups_low_reserve_hours
        reserve_target = min(cap.ups_kwh, max(cap.ups_kwh * risk_reserve_soc, core * reserve_hours))
        # Charge the battery before filling UPS beyond its reserve.  Charge
        # and discharge are mutually exclusive and each converter has one
        # shared hourly power limit, including any grid charging.
        ups_ch = battery_ch = 0.0
        if ups_available and ups_dis <= 1e-9:
            reserve_need = max(0.0, (reserve_target - ups_energy) / cfg.ups_efficiency)
            ups_ch = min(surplus, cap.ups_kw, reserve_need)
            ups_energy += ups_ch * cfg.ups_efficiency
            surplus -= ups_ch
        if pcs_available > 0 and battery_dis <= 1e-9:
            battery_ch = min(surplus, pcs_available,
                             max(0.0, (cap.battery_kwh - battery) / cfg.battery_efficiency))
            battery += battery_ch * cfg.battery_efficiency
            surplus -= battery_ch
        if ups_available and ups_dis <= 1e-9:
            extra = min(surplus, max(0.0, cap.ups_kw - ups_ch),
                        max(0.0, (cap.ups_kwh - ups_energy) / cfg.ups_efficiency))
            ups_energy += extra * cfg.ups_efficiency
            ups_ch += extra
            surplus -= extra
        renewable_charge = ups_ch + battery_ch
        if not islanded:
            if ups_available and ups_dis <= 1e-9:
                extra = min(max(0.0, cap.ups_kw - ups_ch),
                            max(0.0, (reserve_target - ups_energy) / cfg.ups_efficiency))
                ups_ch += extra
                ups_energy += extra * cfg.ups_efficiency
                grid_import += extra
            # Restore ordinary storage after a fault; do not reset its SOC.
            if pcs_available > 0 and battery_dis <= 1e-9:
                extra = min(max(0.0, pcs_available - battery_ch), max(0.0,
                    (cap.battery_kwh * cfg.battery_initial_soc - battery) / cfg.battery_efficiency))
                battery_ch += extra
                battery += extra * cfg.battery_efficiency
                grid_import += extra
        renewable_used = min(renewable_on_bus, max(0.0,
            main_served + renewable_charge - diesel_power - battery_dis
            - (max(0.0, total_load - base_supply) if not islanded else 0.0)))
        residual = (renewable_used + diesel_power + battery_dis + grid_import
                    - main_served - battery_ch - ups_ch)
        max_balance_residual = max(max_balance_residual, abs(residual))
        valid = valid and (abs(residual) <= 1e-6
            and battery_min - 1e-6 <= battery <= cap.battery_kwh + 1e-6
            and ups_min - 1e-6 <= ups_energy <= cap.ups_kwh + 1e-6)
        grid_cost += grid_import * cfg.grid_energy_cost_yuan_per_kwh
        fuel_cost += diesel_power * cfg.diesel_cost_yuan_per_kwh
        battery_discharge += battery_dis
        battery_charge += battery_ch
        ups_discharge += ups_dis
        ups_charge += ups_ch
        core_loss += core_shed
        rigid_loss += rigid_shed
        flex_adjusted += flex_adjust
        if trace:
            trace_rows.append({
                "hour": t,
                "timestamp": year.timestamps[t],
                "grid_fault": int(grid_fault),
                "renewable_bus_fault": int(renewable_fault),
                "compound_extreme": int(scenario.compound_extreme),
                "weather_risk": int(weather_risk),
                "weather_stress": int(weather_stress),
                "core_kw": core,
                "rigid_kw": rigid,
                "flex_kw": flex,
                "renewable_kw": renewable_available,
                "renewable_used_kw": renewable_used,
                "grid_import_kw": grid_import,
                "battery_charge_kw": battery_ch,
                "power_balance_residual_kw": residual,
                "main_bus_live": int(main_bus_live),
                "flex_forced_shed_kw": flex_forced_now,
                "flex_shift_deferred_kw": deferred_now,
                "flex_shift_repaid_kw": repaid_now,
                "flex_shift_pending_kwh": sum(item[1] for item in shift_queue),
                "diesel_kw": diesel_power,
                "gfm_power_kw": gfm_power,
                "battery_discharge_kw": battery_dis,
                "ups_discharge_kw": ups_dis,
                "ups_charge_kw": ups_ch,
                "battery_kwh": battery,
                "ups_kwh": ups_energy,
                "core_shed_kw": core_shed,
                "rigid_shed_kw": rigid_shed,
                "flex_adjusted_kw": flex_adjust,
                "diesel_online": int(np.sum(diesel_online)),
                "pcs_available_kw": pcs_available,
                "ups_available": int(ups_available),
                "ups_reserve_target_kwh": reserve_target,
            })

    # Unfinished shifted work at the horizon is a real loss; it cannot be
    # pushed outside the accounting period for free.
    terminal_shift_loss = sum(item[1] for item in shift_queue)
    flex_forced_loss += terminal_shift_loss
    metrics = {
        "terminal_shift_loss_kwh": terminal_shift_loss,
        "flex_forced_unserved_kwh": flex_forced_loss,
        "regular_unserved_kwh": rigid_loss + flex_forced_loss,
        "flex_shifted_kwh": flex_shifted,
        "flex_shift_repaid_kwh": flex_repaid,
        "fuel_cost_yuan": fuel_cost,
        "grid_cost_yuan": grid_cost,
        "flex_adjustment_cost_yuan": (flex_adjusted + flex_shifted) * cfg.flex_adjustment_cost_yuan_per_kwh,
        "outage_loss_cost_yuan": (core_loss + rigid_loss + flex_forced_loss) * cfg.loss_of_load_cost_yuan_per_kwh,
        "max_power_balance_residual_kw": max_balance_residual,
        "core_unserved_kwh": core_loss,
        "rigid_unserved_kwh": rigid_loss,
        "flex_adjusted_kwh": flex_adjusted,
        "starts": starts,
        "start_failures": start_failures,
        "diesel_run_failures": diesel_failures,
        "bus_fault_hours": bus_fault_hours,
        "renewable_bus_fault_hours": renewable_fault_hours,
        "battery_discharge_kwh": battery_discharge,
        "battery_charge_kwh": battery_charge,
        "ups_discharge_kwh": ups_discharge,
        "ups_charge_kwh": ups_charge,
        "ups_activation_rigid_loss_kwh": ups_activation_rigid_loss,
        "ups_activation_flex_loss_kwh": ups_activation_flex_loss,
        "ups_activation_load_loss_kwh": ups_activation_rigid_loss + ups_activation_flex_loss,
        "ups_activation_loss_cost_yuan": (
            ups_activation_rigid_loss * cfg.loss_of_load_cost_yuan_per_kwh
            + ups_activation_flex_loss * cfg.loss_of_load_cost_yuan_per_kwh
        ),
        # Core and rigid interruption use the configured value-of-lost-load.
        # Allowed flexible-load adjustment remains outside EENS/CVaR and uses
        # its separate adjustment price.  Each kWh is charged once.
        "load_shed_cost_yuan": (
            (core_loss + rigid_loss + flex_forced_loss) * cfg.loss_of_load_cost_yuan_per_kwh
            + (flex_adjusted + flex_shifted) * cfg.flex_adjustment_cost_yuan_per_kwh
        ),
        "final_battery_kwh": battery,
        "final_ups_kwh": ups_energy,
        "gfm_all_resources": bool(cfg.all_pcs_grid_forming and cfg.all_diesel_grid_forming),
        "storage_pcs_coupling_pass": storage_pcs_coupling_pass(cap, cfg),
    }
    valid = valid and bool(np.isfinite(core_loss + rigid_loss + flex_adjusted) and battery >= battery_min - 1e-6 and ups_energy >= ups_min - 1e-6)
    loss_cost = (
        (core_loss + rigid_loss + flex_forced_loss) * cfg.loss_of_load_cost_yuan_per_kwh
        + (flex_adjusted + flex_shifted) * cfg.flex_adjustment_cost_yuan_per_kwh
    )
    return SimulationResult(valid, core_loss, rigid_loss, 0.0, flex_adjusted,
                            fuel_cost + grid_cost + loss_cost, metrics, trace_rows)


def evaluate_capacity(year: SyntheticYear, cap: Capacity, scenarios: list[Scenario], cfg: ResilienceConfig) -> dict:
    if not scenarios:
        raise ValueError("at least one scenario is required")
    assert_discrete_capacity(cap)
    results = [simulate(year, cap, scenario, cfg) for scenario in scenarios]
    core = [r.core_unserved_kwh for r in results]
    rigid = [r.rigid_unserved_kwh for r in results]
    flex = [r.flex_adjusted_kwh for r in results]
    rigid_risk = _risk_summary(rigid, cfg.alpha)
    regular_risk = _risk_summary([r.metrics["regular_unserved_kwh"] for r in results], cfg.alpha)
    core_risk = _risk_summary(core, cfg.alpha)
    required_ups_kwh = required_ups_bridge_kwh(year, cfg)
    required_ups_kw = required_ups_bridge_kw(year, cfg)
    core_bridge_pass = cap.ups_kwh + 1e-9 >= required_ups_kwh and cap.ups_kw + 1e-9 >= required_ups_kw
    pcs_battery_pass = storage_pcs_coupling_pass(cap, cfg)
    feasible = (
        all(r.valid for r in results)
        and pcs_battery_pass
        and
        core_bridge_pass
        and
        max(core, default=0.0) <= 1e-7
        and regular_risk["eens_kwh"] <= cfg.rigid_eens_limit_kwh
        and regular_risk["cvar_kwh"] <= cfg.rigid_cvar_limit_kwh
    )
    mean_operation = float(np.mean([r.operating_cost_yuan for r in results])) if results else 0.0
    return {
        "capacity": asdict(cap),
        "investment_yuan": cap.investment_yuan,
        "mean_operation_yuan": mean_operation,
        "objective_yuan": cap.investment_yuan + mean_operation,
        "feasible": feasible,
        "physical_audit_pass": all(r.valid for r in results),
        "core_bridge_pass": core_bridge_pass,
        "pcs_battery_coupling_pass": pcs_battery_pass,
        "required_ups_kwh": required_ups_kwh,
        "required_ups_kw": required_ups_kw,
        "core_risk": core_risk,
        "rigid_risk": rigid_risk,
        "regular_risk": regular_risk,
        "mean_cost_components_yuan": {
            key: float(np.mean([r.metrics[key] for r in results]))
            for key in ("fuel_cost_yuan", "grid_cost_yuan", "outage_loss_cost_yuan",
                        "flex_adjustment_cost_yuan")
        },
        "flex_adjusted_mean_kwh": float(np.mean(flex)) if flex else 0.0,
        "scenario_metrics": [r.metrics for r in results],
    }


def prepare_outage_bounds(year: SyntheticYear, scenarios: list[Scenario]):
    """Disjoint grid-outage energy windows for an optimistic risk lower bound.

    It grants full healthy diesel immediately, all renewable energy, a full
    battery at every event, and no core/flex energy demand.  Ignoring start,
    faults, GFM and inter-event SOC constraints can only reduce rigid loss.
    """
    windows = []
    for scenario in scenarios:
        path = []
        edges = np.diff(np.r_[False, scenario.grid_fault, False].astype(int))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            weather = year.extreme_weather[a:b] & scenario.weather_enabled
            connected = ~scenario.renewable_bus_fault[a:b]
            wind = year.wind_clean_pu[a:b] * np.where(weather, .35, 1.) * connected
            pv = year.pv_pu[a:b] * np.where(weather, .65, 1.) * connected
            path.append((int(b - a), float(year.rigid_kw[a:b].sum()),
                         float(wind.sum()), float(pv.sum())))
        windows.append(path)
    return windows


def outage_risk_lower_bound(cap: Capacity, cfg: ResilienceConfig, windows) -> dict:
    losses = []
    for path in windows:
        loss = 0.0
        for duration, demand, wind, pv in path:
            battery = min(cap.pcs_kw * duration,
                          cap.battery_kwh * (1 - cfg.battery_min_soc) * cfg.battery_efficiency)
            loss += max(0.0, demand - cap.diesel_kw * duration
                        - cap.wind_kw * wind - cap.pv_kw * pv - battery)
        losses.append(loss)
    return _risk_summary(losses, cfg.alpha)


def plan_capacity(year: SyntheticYear, cfg: ResilienceConfig,
                  capacities: Iterable[Capacity], scenarios: list[Scenario],
                  time_limit_seconds: float | None = 10000.0,
                  initial_incumbent: Capacity | None = None,
                  declared_candidate_count: int | None = None,
                  investment_ordered: bool = False,
                  progress_callback: Callable | None = None,
                  use_outage_bound: bool = True) -> PlanResult:
    """Complete integer search for the fixed causal simulation policy.

    A lazy iterator must explicitly promise investment order before the
    ordered stopping proof may be used. Arbitrary iterators are fully scanned.
    Timeouts always leave unresolved candidates and never imply optimality.
    """
    started = time.monotonic()
    if any(value < 0 for value in (cfg.loss_of_load_cost_yuan_per_kwh,
            cfg.flex_adjustment_cost_yuan_per_kwh, cfg.grid_energy_cost_yuan_per_kwh,
            cfg.diesel_cost_yuan_per_kwh)):
        raise ValueError("nonnegative operating costs required for investment bound")
    if isinstance(capacities, (list, tuple)):
        candidates = iter(sorted(capacities, key=lambda cap: cap.investment_yuan))
        investment_ordered = True
    else:
        candidates = iter(capacities)
    evaluations = []
    incumbent = None
    if initial_incumbent is not None:
        incumbent = evaluate_capacity(year, initial_incumbent, scenarios, cfg)
        if not incumbent["feasible"]:
            raise ValueError("initial incumbent must pass all training constraints")
    reference_evaluated = incumbent is not None
    skipped_by_cost = skipped_by_core_bridge = skipped_by_coupling = 0
    skipped_by_risk_bound = candidate_count = 0
    time_limit_reached = False
    last_cost = 0.0
    next_lower_bound = 0.0
    required_energy = required_ups_bridge_kwh(year, cfg)
    required_power = required_ups_bridge_kw(year, cfg)
    windows = prepare_outage_bounds(year, scenarios) if use_outage_bound else None
    last_progress = -float("inf")

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if progress_callback and (force or now - last_progress >= 10):
            progress_callback({
                "phase": "capacity_search", "elapsed_seconds": now - started,
                "visited": candidate_count, "evaluated": len(evaluations),
                "risk_bound_rejections": skipped_by_risk_bound,
                "selected": incumbent, "planning_complete": False,
                "last_investment_lower_bound_yuan": last_cost,
            })
            last_progress = now

    progress(True)
    while True:
        if time_limit_seconds is not None and time.monotonic() - started >= time_limit_seconds:
            time_limit_reached = True
            next_lower_bound = last_cost if investment_ordered else 0.0
            break
        try:
            cap = next(candidates)
        except StopIteration:
            break
        assert_discrete_capacity(cap)
        if investment_ordered and cap.investment_yuan < last_cost - 1e-9:
            raise ValueError("candidate iterator violated investment ordering")
        last_cost = cap.investment_yuan
        next_lower_bound = last_cost if investment_ordered else 0.0
        candidate_count += 1
        if incumbent and cap.investment_yuan >= incumbent["objective_yuan"]:
            skipped_by_cost += 1
            if investment_ordered:
                break
            continue
        if not storage_pcs_coupling_pass(cap, cfg):
            skipped_by_coupling += 1
            continue
        if cap.ups_kwh + 1e-9 < required_energy or cap.ups_kw + 1e-9 < required_power:
            skipped_by_core_bridge += 1
            continue
        if use_outage_bound:
            lower = outage_risk_lower_bound(cap, cfg, windows)
            if (lower["eens_kwh"] > cfg.rigid_eens_limit_kwh + 1e-9
                    or lower["cvar_kwh"] > cfg.rigid_cvar_limit_kwh + 1e-9):
                skipped_by_risk_bound += 1
                progress()
                continue
        result = evaluate_capacity(year, cap, scenarios, cfg)
        evaluations.append(result)
        improved = result["feasible"] and (incumbent is None or result["objective_yuan"] < incumbent["objective_yuan"])
        if improved:
            incumbent = result
        progress(improved)
    complete = not time_limit_reached
    if incumbent and complete:
        status = "global_optimal_within_discrete_domain_and_fixed_policy"
    elif incumbent:
        status = "time_limit_feasible_incumbent_not_global"
    else:
        status = "no_feasible_design_in_declared_domain" if complete else "time_limit_without_feasible_incumbent"
    lower_bound = (incumbent["objective_yuan"] if incumbent and complete else
                   min(next_lower_bound, incumbent["objective_yuan"]) if incumbent else next_lower_bound)
    progress(True)
    return PlanResult(status, incumbent, evaluations, None, {
        "model": "resilience-v2-synthetic-fine-modules-day-ahead-v3",
        "synthetic": True, "hours": year.hours,
        "candidate_capacity_count": declared_candidate_count,
        "candidate_count_basis": "rectangular economic domain before budget, UPS and coupling filters",
        "candidate_capacity_visited_count": candidate_count,
        "evaluated_capacity_count": len(evaluations),
        "skipped_by_investment_bound": skipped_by_cost,
        "skipped_by_core_bridge": skipped_by_core_bridge,
        "skipped_by_pcs_battery_coupling": skipped_by_coupling,
        "skipped_by_optimistic_risk_bound": skipped_by_risk_bound,
        "reference_incumbent_evaluated": reference_evaluated,
        "planning_complete": complete, "time_limit_seconds": time_limit_seconds,
        "planning_elapsed_seconds": time.monotonic() - started,
        "objective_lower_bound_yuan": lower_bound,
        "relative_gap": ((incumbent["objective_yuan"] - lower_bound) / incumbent["objective_yuan"]) if incumbent else None,
        "optimality_scope": "discrete capacities, fixed training samples and fixed causal dispatch policy",
        "all_pcs_grid_forming": cfg.all_pcs_grid_forming,
        "all_diesel_grid_forming": cfg.all_diesel_grid_forming,
        "resilience_metric_load": "rigid_plus_involuntary_flexible_loss",
        "core_load_rule": "zero_unserved_in_all_evaluated_scenarios",
        "flexible_load_rule": "allowed interruption and repaid shifting excluded; forced interruption and unfinished shifting included",
        "scenario_weather": "synthetic perfect next-calendar-day Boolean forecast",
        "scenario_faults": "unpredictable_grid_bus_renewable_bus_and_equipment_faults",
        "compound_extreme_scenario": "grid_and_renewable_bus_disconnected_during_extreme_weather",
        "storage_pcs_rule": "battery backing and actual nonzero current active power required for GFM",
    })


def validate_capacity(year: SyntheticYear, cap: Capacity, cfg: ResilienceConfig,
                      scenarios: list[Scenario]) -> dict:
    result = evaluate_capacity(year, cap, scenarios, cfg)
    result["role"] = "validation"
    return result


def write_plan_outputs(output: Path, year: SyntheticYear, plan: PlanResult,
                       validation: dict | None = None, selected_trace: list[dict] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    year.frame().to_csv(output / "synthetic_resilience_year.csv", index=False)
    summary = {
        "status": plan.status,
        "selected": plan.selected,
        "validation": validation,
        "metadata": plan.metadata,
        "evaluations": plan.evaluations,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "capacity_evaluations.csv").open("w", newline="") as stream:
        if plan.evaluations:
            keys = ["objective_yuan", "investment_yuan", "mean_operation_yuan", "feasible", "core_bridge_pass", "required_ups_kwh", "required_ups_kw", "flex_adjusted_mean_kwh"]
            writer = csv.DictWriter(stream, fieldnames=["wind_kw", "pv_kw", "diesel_units", "battery_kwh", "pcs_kw", "ups_kwh", "ups_kw"] + keys)
            writer.writeheader()
            for row in plan.evaluations:
                writer.writerow({**row["capacity"], **{key: row[key] for key in keys}})
    if validation is not None:
        (output / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if selected_trace is not None:
        with (output / "selected_trace.csv").open("w", newline="") as stream:
            if selected_trace:
                writer = csv.DictWriter(stream, fieldnames=list(selected_trace[0]))
                writer.writeheader()
                writer.writerows(selected_trace)
