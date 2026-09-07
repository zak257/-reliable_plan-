"""Read cap_plan CSVs without importing or changing its historical models."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import UnitCommitmentOptions

COMPONENTS = ("wind", "pv", "diesel", "battery_energy", "pcs")
FAILABLE_COMPONENTS = ("wind", "pv", "diesel", "pcs")
DEFAULT_MODULES = dict(zip(COMPONENTS, (100.0, 100.0, 300.0, 50.0, 50.0)))


@dataclass
class CaseData:
    name: str
    load_kw: np.ndarray
    wind_pu: np.ndarray
    pv_pu: np.ndarray
    module_sizes: dict[str, float]
    unit_bounds: dict[str, tuple[int, int]]
    annual_cost_per_unit: dict[str, float]
    fuel_cost_per_kwh: float
    efficiency: float
    soc_min: float
    soc_max: float
    dt_hours: float = 1.0
    timestamps: list[str] = field(default_factory=list)
    manifest: list[dict] = field(default_factory=list)
    input_metadata: dict = field(default_factory=dict)
    unit_commitment: UnitCommitmentOptions = field(default_factory=UnitCommitmentOptions)

    def __post_init__(self):
        for key in ("load_kw", "wind_pu", "pv_pu"):
            values = np.array(getattr(self, key), dtype=float, copy=True)
            if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{key} must be a finite nonnegative 1-D series")
            values.flags.writeable = False
            setattr(self, key, values)
        if not len(self.load_kw) == len(self.wind_pu) == len(self.pv_pu):
            raise ValueError("Time series lengths differ")
        if not 0 < self.efficiency <= 1 or not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("Invalid battery efficiency or SOC bounds")
        if not math.isfinite(self.dt_hours) or self.dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        if not math.isfinite(self.fuel_cost_per_kwh) or self.fuel_cost_per_kwh < 0:
            raise ValueError("Fuel cost must be finite and nonnegative")
        for key in COMPONENTS:
            lo, hi = self.unit_bounds[key]
            if lo != int(lo) or hi != int(hi) or lo < 0 or hi < lo:
                raise ValueError(f"Invalid module bounds for {key}: {(lo, hi)}")
            if not math.isfinite(self.module_sizes[key]) or self.module_sizes[key] <= 0:
                raise ValueError(f"Invalid module size for {key}")
            if not math.isfinite(self.annual_cost_per_unit[key]) or self.annual_cost_per_unit[key] < 0:
                raise ValueError(f"Invalid annual cost for {key}")

    @property
    def hours(self) -> int:
        return len(self.load_kw)

    @property
    def demand_kwh(self) -> float:
        return float(self.load_kw.sum() * self.dt_hours)

    @property
    def period_cost_per_unit(self) -> dict[str, float]:
        scale = self.hours * self.dt_hours / 8760.0
        return {k: scale * v for k, v in self.annual_cost_per_unit.items()}

    def validate_units(self, units: Mapping[str, int]) -> dict[str, int]:
        if set(units) != set(COMPONENTS):
            raise ValueError(f"Capacity must contain exactly {COMPONENTS}")
        result = {}
        for key in COMPONENTS:
            value = float(units[key])
            lo, hi = self.unit_bounds[key]
            if not math.isfinite(value) or abs(value - round(value)) > 1e-6 or not lo <= round(value) <= hi:
                raise ValueError(f"{key}={value} is outside the integer grid [{lo}, {hi}]")
            result[key] = int(round(value))
        return result

    def capacities(self, units: Mapping[str, int]) -> dict[str, float]:
        return {k: self.module_sizes[k] * n for k, n in self.validate_units(units).items()}


def _value(frame: pd.DataFrame, *aliases: str, default: float | None = None) -> float:
    for name in aliases:
        if name in frame and len(frame) and pd.notna(frame[name].iloc[0]):
            value = float(frame[name].iloc[0])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite CSV parameter: {name}")
            return value
    if default is not None:
        return default
    raise ValueError(f"Required CSV parameter missing: {aliases}")


def _fill_like_cap_plan(values: np.ndarray) -> tuple[np.ndarray, int]:
    # Preserve the reference CaseData.fill_missing convention, recording repairs.
    result = np.asarray(values, dtype=float).copy()
    count = int((~np.isfinite(result)).sum())
    for i in range(len(result)):
        if np.isfinite(result[i]):
            continue
        if i == 0:
            following = min(1, len(result) - 1)
            result[i] = result[following] if np.isfinite(result[following]) else 0.0
        elif i == len(result) - 1 or not np.isfinite(result[i + 1]):
            result[i] = result[i - 1]
        else:
            result[i] = 0.5 * (result[i - 1] + result[i + 1])
    return result, count


def load_case(data_root: str | Path, case: str, start_hour: int = 0, hours: int = 8760,
              modules: Mapping[str, float] | None = None, load_scale: float = 1.0,
              unit_commitment: UnitCommitmentOptions | None = None) -> CaseData:
    root = Path(data_root).expanduser().resolve()
    directory = (root / case / "input").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Case must be inside data_root")
    names = ("curve", "es_para", "pv_para", "ts_para", "wd_para")
    frames, manifest = {}, []
    for name in names:
        path = directory / f"{name}.csv"
        frames[name] = pd.read_csv(path, encoding="utf-8-sig")
        manifest.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "rows": len(frames[name])})
    curve, es, pv, dg, wd = (frames[k] for k in names)
    if start_hour < 0 or hours < 1 or start_hour + hours > len(curve):
        raise ValueError(f"Requested [{start_hour}, {start_hour + hours}) but CSV has {len(curve)} hours")
    if not math.isfinite(load_scale) or load_scale <= 0:
        raise ValueError("load_scale must be finite and positive")
    module_sizes = dict(DEFAULT_MODULES if modules is None else modules)
    if set(module_sizes) != set(COMPONENTS):
        raise ValueError(f"modules must contain exactly {COMPONENTS}")
    repaired = {}
    series = {}
    for key, column in (("wind", "风速(m/s)"), ("sun", "单位面积太阳辐射(W/m^2)"), ("load", "用电功率(kW)")):
        a, repaired[column] = _fill_like_cap_plan(pd.to_numeric(curve[column], errors="raise").to_numpy())
        series[key] = a[start_hour:start_hour + hours]
    cut_in = _value(wd, "风机切入风速(m/s)", "切入风速(m/s)")
    rated = _value(wd, "风机额定风速(m/s)", "额定风速(m/s)")
    cut_out = _value(wd, "风机切出风速(m/s)", "切出风速(m/s)")
    if not 0 <= cut_in < rated < cut_out:
        raise ValueError("Wind speeds must satisfy 0 <= cut-in < rated < cut-out")
    wind_pu = np.where((series["wind"] >= cut_in) & (series["wind"] < cut_out),
                       np.minimum(1.0, (series["wind"] - cut_in) / (rated - cut_in)), 0.0)
    pv_pu = np.maximum(0.0, series["sun"] / 1000 * _value(pv, "光伏单位辐照出力系数", "单位容量发电能力(kW/单位辐射)"))
    bounds_kw = {
        "wind": (_value(wd, "风机设计容量下限(kW)", "最小设计容量(kW)"), _value(wd, "风机设计容量上限(kW)", "最大设计容量(kW)")),
        "pv": (_value(pv, "光伏设计容量下限(kW)", "最小设计容量(kW)"), _value(pv, "光伏设计容量上限(kW)", "最大设计容量(kW)")),
        "diesel": (_value(dg, "柴油机设计容量下限(kW)", "柴油发电设计容量下限(kW)"), _value(dg, "柴油机额定容量(kW)", "柴油发电容量(kW)")),
        "battery_energy": (_value(es, "电池设计容量下限(kWh)", "最小设计容量(kWh)"), _value(es, "电池设计容量上限(kWh)", "最大设计容量(kWh)")),
        "pcs": (_value(es, "PCS设计容量下限(kW)", default=0.0), _value(es, "PCS设计容量上限(kW)")),
    }
    battery_life = _value(es, "电池设计使用年限(年)", "设计使用年限(年)")
    prices_lives = {
        "wind": (_value(wd, "风机单位造价(元/kW)", "单位容量造价(元/kW)"), _value(wd, "风机设计使用年限(年)", "设计使用年限(年)")),
        "pv": (_value(pv, "光伏单位造价(元/kW)", "单位容量造价(元/kW)"), _value(pv, "光伏设计使用年限(年)", "设计使用年限(年)")),
        "diesel": (_value(dg, "柴油机单位容量造价(元/kW)", "柴油发电单位容量造价(元/kW)"), _value(dg, "柴油机设计使用年限(年)", "设计使用年限(年)")),
        "battery_energy": (_value(es, "电池单位容量造价(元/kWh)", "电储单位容量造价(元/kWh)"), battery_life),
        "pcs": (_value(es, "PCS单位造价(元/kW)", "PCS单位成本(元/kW)"), battery_life),
    }
    unit_bounds, annual = {}, {}
    for key in COMPONENTS:
        size = module_sizes[key]
        lo, hi = bounds_kw[key]
        price, life = prices_lives[key]
        if not math.isfinite(size) or size <= 0 or lo < 0 or hi < lo or price < 0 or life <= 0:
            raise ValueError(f"Invalid source parameters for {key}")
        unit_bounds[key] = (math.ceil(lo / size - 1e-10), math.floor(hi / size + 1e-10))
        annual[key] = size * price / life
    fuel_eff = _value(dg, "柴油机燃油效率(kWh/kg)", "柴油发电效率(kWh/kg)")
    if fuel_eff <= 0:
        raise ValueError("Fuel efficiency must be positive")
    selected = curve.iloc[start_hour:start_hour + hours]
    timestamps = (selected["日期"].astype(str) + " " + selected["时刻"].astype(str)).tolist()
    return CaseData(case, series["load"] * load_scale, wind_pu, pv_pu, module_sizes, unit_bounds, annual,
                    _value(dg, "柴油燃料价格(元/kg)", "柴油发电成本(元/kg)") / fuel_eff,
                    _value(es, "储能充放电效率", "充放电效率"),
                    _value(es, "储能SOC下限", "运行中SOC最小值"),
                    _value(es, "储能SOC上限", "运行中SOC最大值"), timestamps=timestamps, manifest=manifest,
                    unit_commitment=unit_commitment or UnitCommitmentOptions(),
                    input_metadata={"start_hour": start_hour, "hours": hours, "available_rows": len(curve),
                                    "imputed_values_in_full_csv": repaired, "load_scale": load_scale,
                                    "continuous_cyclic_soc": True, "battery_failure_model": "ideal_cells_with_failable_pcs",
                                    "diesel_dispatch": "module_unit_commitment" if (unit_commitment or UnitCommitmentOptions()).enabled else "continuous_0_to_available_capacity",
                                    "capacity_bounds_source": bounds_kw})
