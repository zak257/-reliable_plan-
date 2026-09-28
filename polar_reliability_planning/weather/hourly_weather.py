"""Same hourly weather drives access, icing and wind hazard. No extra chain."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class WeatherYear:
    timestamps: tuple[str, ...]
    load_kw: np.ndarray
    ambient_c: np.ndarray
    wind_speed_ms: np.ndarray
    wind_clean_pu: np.ndarray
    pv_pu: np.ndarray
    access_safe: np.ndarray
    icing_active: np.ndarray
    ice_power_factor: np.ndarray
    wind_protective_stop: np.ndarray
    extreme_hazard: np.ndarray
    hard_stop: np.ndarray
    metadata: dict

    def __post_init__(self):
        n = len(self.timestamps)
        times = pd.DatetimeIndex(self.timestamps)
        if not n or times.tz is None or times.hasnans or times.has_duplicates or (n > 1 and not np.all(np.diff(times.as_unit("ns").asi8) == 3_600_000_000_000)):
            raise ValueError("Weather timestamps must be timezone-aware, unique, hourly and consecutive")
        bool_fields = {"access_safe", "icing_active", "wind_protective_stop", "extreme_hazard", "hard_stop"}
        for key in vars(self).keys() - {"metadata", "timestamps"}:
            a = np.asarray(getattr(self, key))
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                raise ValueError(f"Invalid weather field {key}")
            if key in bool_fields and not np.all((a == 0) | (a == 1)):
                raise ValueError(f"{key} must be binary")
            if key not in bool_fields and key != "ambient_c" and np.any(a < 0):
                raise ValueError(f"{key} must be nonnegative")
            if key in ("wind_clean_pu", "ice_power_factor") and np.any(a > 1):
                raise ValueError(f"{key} must be <= 1")
            a = np.array(a, dtype=bool if key in bool_fields else float, copy=True)
            a.flags.writeable = False
            setattr(self, key, a)
        if np.any((~self.icing_active) & (self.ice_power_factor != 1)):
            raise ValueError("Ice derating without an active icing label")

    @property
    def hours(self):
        return len(self.timestamps)


def load_weather(data, config):
    path = Path(config["data_root"]) / config["case"] / "input" / "curve.csv"
    source = pd.read_csv(path)
    start, count = config.get("start_hour", 0), data.hours
    rows = source.iloc[start:start + count]
    # Unlike radiation, no temperature imputation is permitted here.
    ambient = pd.to_numeric(rows["温度(摄氏度)"], errors="raise").to_numpy()
    speed = pd.to_numeric(rows["风速(m/s)"], errors="raise").to_numpy()
    times = pd.to_datetime(rows["日期"] + " " + rows["时刻"], format="%m/%d/%Y %H:%M:%S")
    times = pd.DatetimeIndex(times).tz_localize(config["data_contract"]["timezone"])
    w = config["weather"]
    if w["wind_curve_basis"] not in ("clean_power_curve", "observed_net_output"):
        raise ValueError("Unknown wind_curve_basis")
    if w["wind_curve_basis"] != "clean_power_curve":
        raise ValueError("cap_plan adapter computes a clean curve; observed net input needs its own adapter")
    columns = ("access_safe", "icing_active", "ice_power_factor", "wind_protective_stop", "extreme_hazard")
    metadata = {"ambient_source": str(path), "weather_label_status": "observed",
                "timezone_status": config["data_contract"]["timezone_status"],
                "available_source_hours": len(source), "evaluated_hours": count,
                "period_start": times[0].isoformat(), "period_end": times[-1].isoformat(),
                "climate_scope": "conditional_on_this_weather_sequence"}
    if w["label_mode"] == "observed_csv":
        label_path = Path(w["labels_path"])
        labels = pd.read_csv(label_path)
        label_times = pd.DatetimeIndex(pd.to_datetime(labels["timestamp"]))
        if label_times.tz is None or not label_times.equals(times):
            raise ValueError("Weather labels must align exactly with the selected timezone-aware timestamps")
        arrays = [labels[k].to_numpy() for k in columns]
        metadata.update(labels_path=str(label_path), labels_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest())
    elif w["label_mode"] == "assumed_stress_case" and config["synthetic"]:
        access = speed < w["access_wind_threshold_ms"]
        extreme = (~access) & (ambient <= w["icing_temperature_ceiling_c"])
        active = np.zeros(count, dtype=bool)
        remaining = 0
        for h in range(count):
            remaining = w["residual_icing_hours"] + 1 if extreme[h] else max(0, remaining - 1)
            active[h] = extreme[h] or remaining > 0
        factor = np.where(active, w["ice_power_factor"], 1.0)
        # Residual ice can keep protection active after wind/access recover.
        protect = active.copy()
        arrays = [access, active, factor, protect, extreme]
        metadata.update(weather_label_status="assumed_stress_case", label_rules=w,
                        warning="Wind/temperature thresholds are engineering scenarios, not observed blizzard or icing classification")
    else:
        raise ValueError("Missing observed labels or explicit synthetic stress-case flag")
    metadata.update(access_blocked_hours=int(np.sum(np.asarray(arrays[0]) == 0)),
                    icing_hours=int(np.sum(arrays[1])), protective_stop_hours=int(np.sum(arrays[3])))
    # cap_plan already applies cut-in/rated/cut-out. A hard stop cannot be bypassed.
    wind_parameters = pd.read_csv(path.parent / "wd_para.csv")
    from ..data.cap_plan_loader import _value
    cut_out = _value(wind_parameters, "风机切出风速(m/s)", "切出风速(m/s)")
    return WeatherYear(tuple(t.isoformat() for t in times), data.load_kw, ambient, speed,
                       data.wind_pu, data.pv_pu, *arrays, speed >= cut_out, metadata)
