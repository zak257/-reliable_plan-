"""Individual diesel commitment, with forced outages overriding minimum uptime."""
from __future__ import annotations

from dataclasses import dataclass
import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy import sparse

from ..data import CaseData


@dataclass
class CommitmentBlock:
    online: gp.MVar
    startup: gp.MVar
    shutdown: gp.MVar
    min_up_constraints: dict[int, gp.MConstr]

    def update_availability(self, available: np.ndarray):
        self.online.UB = available
        # Each startup is enforced only until its next forced outage. Applying
        # an aggregate 'recent trips' slack would incorrectly relax later starts.
        for lag, constraint in self.min_up_constraints.items():
            uninterrupted = np.ones((available.shape[0], available.shape[1] - lag))
            for offset in range(1, lag + 1):
                uninterrupted *= available[:, offset:available.shape[1] - lag + offset]
            constraint.RHS = 1 - uninterrupted


def add_unit_commitment(model: gp.Model, data: CaseData, diesel_power: gp.MVar,
                        installed_count=None) -> CommitmentBlock | None:
    count, hours = data.unit_bounds["diesel"][1], data.hours
    if not data.unit_commitment.enabled or not count:
        return None
    settings = data.unit_commitment
    online = model.addMVar((count, hours), vtype=GRB.BINARY, name="diesel_unit_online")
    # Given binary online states, these transition variables are automatically 0/1.
    startup = model.addMVar((count, hours), lb=0, ub=1, name="diesel_unit_startup")
    shutdown = model.addMVar((count, hours), lb=0, ub=1, name="diesel_unit_shutdown")
    if installed_count is not None:
        installed = model.addMVar(count, vtype=GRB.BINARY, name="diesel_installed_prefix")
        model.addConstr(installed.sum() == installed_count, name="installed_diesel_count")
        if count > 1:
            model.addConstr(installed[1:] <= installed[:-1], name="installed_prefix_order")
        model.addConstr(online <= installed[:, None], name="online_only_if_installed")
    model.addConstr(startup[:, 0] == online[:, 0], name="initial_diesel_start")
    model.addConstr(shutdown[:, 0] == 0, name="initial_diesel_stop")
    model.addConstr(startup <= online, name="start_requires_online")
    model.addConstr(shutdown <= 1 - online, name="stop_requires_offline")
    if hours > 1:
        model.addConstr(online[:, 1:] - online[:, :-1] == startup[:, 1:] - shutdown[:, 1:], name="commitment_transition")
        model.addConstr(startup[:, 1:] <= 1 - online[:, :-1], name="start_requires_previously_off")
        model.addConstr(shutdown[:, 1:] <= online[:, :-1], name="stop_requires_previously_on")
    # Initial units have already satisfied downtime. Minimum times crossing the
    # right horizon boundary are truncated; no artificial terminal shutdown.
    up_steps = min(hours, max(1, math.ceil(settings.min_up_hours / data.dt_hours)))
    down_steps = min(hours, max(1, math.ceil(settings.min_down_hours / data.dt_hours)))
    up_constraints = {lag: model.addConstr(startup[:, :hours - lag] - online[:, lag:] <= 0,
                                           name=f"minimum_uptime_lag_{lag}")
                      for lag in range(1, up_steps)}
    rolling = sparse.diags([np.ones(hours - k) for k in range(down_steps)],
                            offsets=range(down_steps), shape=(hours, hours), format="csr")
    model.addConstr(shutdown @ rolling <= 1 - online, name="minimum_downtime_including_forced_trips")
    online_capacity = data.module_sizes["diesel"] * online.sum(axis=0)
    model.addConstr(diesel_power <= online_capacity, name="diesel_online_max_output")
    model.addConstr(diesel_power >= settings.min_output_fraction * online_capacity, name="diesel_online_min_output")
    return CommitmentBlock(online, startup, shutdown, up_constraints)


def available_diesel_modules(data: CaseData, units: dict[str, int], scenario) -> np.ndarray:
    count = data.unit_bounds["diesel"][1]
    result = np.zeros((count, data.hours))
    source = scenario.diesel_module_availability
    if source is None:
        raise ValueError("Unit commitment requires individual diesel availability trajectories")
    source = np.asarray(source)
    if source.shape != (units["diesel"], data.hours) or not np.isin(source, (0, 1)).all():
        raise ValueError("Diesel module trajectories must match installed units and horizon")
    if not np.allclose(source.sum(axis=0) * data.module_sizes["diesel"], scenario.diesel_available_kw, atol=1e-8, rtol=0):
        raise ValueError("Aggregate diesel availability disagrees with its module trajectories")
    result[:units["diesel"]] = source
    return result


def standby_commitment(availability: np.ndarray, min_down_steps: int, target_online: int) -> np.ndarray:
    """Keep the required units on, and start cold standby units after trips."""
    count, hours = availability.shape
    online = np.zeros_like(availability)
    active = np.zeros(count, dtype=bool)
    may_restart_at = np.zeros(count, dtype=int)
    for t in range(hours):
        trips = active & (availability[:, t] == 0)
        may_restart_at[trips] = t + min_down_steps
        active[trips] = False
        for i in range(count):
            if active.sum() >= target_online:
                break
            if not active[i] and availability[i, t] and t >= may_restart_at[i]:
                active[i] = True
        online[:, t] = active
    return online


def audit_commitment(data: CaseData, units: dict[str, int], scenario, dispatch: dict) -> dict:
    if not data.unit_commitment.enabled:
        return {"enabled": False, "passed": True}
    available = available_diesel_modules(data, units, scenario)
    online = dispatch.get("diesel_unit_online", np.zeros_like(available))
    starts = dispatch.get("diesel_unit_startup", np.zeros_like(available))
    stops = dispatch.get("diesel_unit_shutdown", np.zeros_like(available))
    previous = np.pad(online[:, :-1], ((0, 0), (1, 0)))
    errors = [0.0]
    if online.size:
        errors.extend([float(np.abs(online - np.rint(online)).max()),
                       float(np.abs(online - previous - starts + stops).max()),
                       float((online - available).max()),
                       float(np.abs(starts - np.maximum(online - previous, 0)).max()),
                       float(np.abs(stops - np.maximum(previous - online, 0)).max())])
        up = max(1, math.ceil(data.unit_commitment.min_up_hours / data.dt_hours))
        down = max(1, math.ceil(data.unit_commitment.min_down_hours / data.dt_hours))
        for i in range(len(online)):
            for t in np.flatnonzero(starts[i] > 0.5):
                for k in range(t, min(t + up, data.hours)):
                    if not available[i, k]:
                        break
                    errors.append(1 - online[i, k])
            for t in np.flatnonzero(stops[i] > 0.5):
                errors.append(float(online[i, t:min(t + down, data.hours)].max()))
    installed_online_kw = data.module_sizes["diesel"] * online.sum(axis=0)
    errors.extend([float((dispatch["diesel_kw"] - installed_online_kw).max()),
                   float((data.unit_commitment.min_output_fraction * installed_online_kw - dispatch["diesel_kw"]).max()),
                   float(np.minimum(dispatch["charge_kw"], dispatch["discharge_kw"]).max())])
    maximum = max(errors)
    return {"enabled": True, "passed": maximum <= 1e-5, "max_violation": float(maximum),
            "startups": int(round(float(starts.sum()))), "shutdowns": int(round(float(stops.sum()))),
            "forced_shutdowns": int(round(float((previous * (1 - available)).sum())))}
