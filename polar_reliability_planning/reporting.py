from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, default=_json_default) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def write_dispatch(path: Path, data, solution):
    d = solution.dispatch
    rows = []
    for t in range(data.hours):
        row = {"hour": t, "timestamp": data.timestamps[t] if data.timestamps else str(t), "load_kw": data.load_kw[t]}
        row.update({k: d[k][t] for k in ("wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw", "shed_kw")})
        for key in ("diesel_online_units", "diesel_startup_units", "diesel_shutdown_units", "battery_charge_mode"):
            if key in d:
                row[key] = d[key][t]
        for key in ("diesel_unit_online", "diesel_unit_startup", "diesel_unit_shutdown"):
            if key in d:
                row.update({f"{key}_{i+1}": values[t] for i, values in enumerate(d[key])})
        row.update({"energy_start_kwh": d["stored_energy_kwh"][t], "energy_end_kwh": d["stored_energy_kwh"][t + 1],
                    "wind_curtailment_kw": max(0.0, data.wind_pu[t] * solution.capacities["wind"] - d["wind_kw"][t]),
                    "pv_curtailment_kw": max(0.0, data.pv_pu[t] * solution.capacities["pv"] - d["pv_kw"][t])})
        rows.append(row)
    write_csv(path, rows)
