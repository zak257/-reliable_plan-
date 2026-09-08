"""Independent physical checks for solver and constructed dispatches."""
from __future__ import annotations

import numpy as np

from ..data import CaseData
from ..scenario_generation import Scenario


def audit_physical_dispatch(data: CaseData, units: dict[str, int], scenario: Scenario, dispatch: dict) -> dict:
    """Recompute balance, storage dynamics and bounds from the exported values."""
    power_names = ("wind_kw", "pv_kw", "diesel_kw", "charge_kw", "discharge_kw", "shed_kw")
    if any(np.shape(dispatch[k]) != (data.hours,) or not np.isfinite(dispatch[k]).all() for k in power_names):
        return {"passed": False, "error": "Invalid power array"}
    energy = dispatch["usable_energy_kwh"]
    if np.shape(energy) != (data.hours + 1,) or not np.isfinite(energy).all():
        return {"passed": False, "error": "Invalid energy array"}
    wind, pv, diesel, charge, discharge, shed = (dispatch[k] for k in power_names)
    capacity = data.capacities(units)
    balance = wind + pv + diesel + discharge + shed - scenario.load_kw - charge
    transition = energy[1:] - energy[:-1] - data.dt_hours * (data.efficiency * charge - discharge / data.efficiency)
    bounds = [0.0, float(-energy.min()), float(energy.max() -
        (data.soc_max - data.soc_min) * capacity["battery_energy"])]
    bounds.extend(float(-dispatch[k].min()) for k in power_names)
    bounds.extend(float(np.max(value)) for value in (wind - scenario.wind_available_kw,
        pv - scenario.pv_available_kw, diesel - scenario.diesel_available_kw,
        charge + discharge - scenario.pcs_available_kw, shed - scenario.load_kw))
    result = {"power_balance_max_abs_kw": float(np.max(np.abs(balance))),
              "energy_transition_max_abs_kwh": float(np.max(np.abs(transition))),
              "cyclic_energy_abs_kwh": float(abs(energy[-1] - energy[0])),
              "bound_max_violation": max(bounds)}
    return {"passed": max(result.values()) <= 1e-5, **result}
