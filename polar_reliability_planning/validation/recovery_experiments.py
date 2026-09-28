"""Export 72-hour physics, exhaustive reference, and same-evaluator ablations.

Run before the annual CLI. Synthetic fixtures and Zhongshan results are stored
in separate directories, all with explicit claim scope.
"""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..data.cap_plan_loader import CaseData, COMPONENTS
from ..planning.recovery_optimizer import PathArchive, evaluate_policy, plan_grid
from ..recovery.config import read_recovery_config, settings_from_config
from ..recovery.container_thermal import ThermalParameters
from ..recovery_cli import ROOT, save_trace, write_csv, write_json
from ..reliability.causal_simulator import simulate
from ..reliability.recovery_policy import RecoveryPolicy
from ..validation.recovery_audit import audit_trace
from ..weather.hourly_weather import WeatherYear


class ScriptedNoise:
    warmup = 0

    def uniform(self, mechanism, hour, device):
        if (mechanism, hour, device) in {("diesel_run", 24, 0), ("diesel_run", 48, 1),
                                       ("diesel_start", 13, 1), ("wind_run", 40, 0)}:
            return 0.0
        return 0.9


def synthetic_case():
    config = read_recovery_config(ROOT / "config/zhongshan_recovery_stress.toml")
    hours = 72
    load = np.full(hours, 120.0)
    pv = np.maximum(0, np.sin((np.arange(hours) % 24 - 6) * np.pi / 12))
    data = CaseData("synthetic_recovery_72h", load, np.full(hours, .5), pv,
                    dict(zip(COMPONENTS, (100., 100., 100., 50., 50.))),
                    {k: (0, 40 if k == "battery_energy" else 10) for k in COMPONENTS},
                    dict(zip(COMPONENTS, (100000., 45000., 50000., 12500., 10416.66666667))),
                    3.33333333333333, .88, .25, .85)
    access = np.ones(hours, dtype=bool)
    access[:6] = False
    access[23:32] = False
    access[47:55] = False
    ice = ~access
    ice[32:36] = True
    ice[55:60] = True
    timestamps = tuple(t.isoformat() for t in pd.date_range("2020-01-01", periods=hours, freq="h", tz="UTC"))
    weather = WeatherYear(timestamps, load, np.full(hours, -10.), np.where(access, 5., 18.),
                          data.wind_pu, data.pv_pu, access, ice, np.where(ice, .5, 1.),
                          ice, ~access, np.zeros(hours), {"synthetic": True})
    settings = replace(settings_from_config(config), warmup_hours=0,
                       thermal=ThermalParameters(.1, 1., 5., 5., -10.),
                       diesel_repair_mean_hours=6, repair_distribution="fixed",
                       remote_heat=True)
    config["policies"] = [dict(name="load_priority", online_floor=2, reserve_units=0,
                               recharge_soc=.8, recharge_kw=50., heat_priority=False, keep_standby_warm=True),
                          dict(name="heat_priority", online_floor=2, reserve_units=1,
                               recharge_soc=.8, recharge_kw=50., heat_priority=True, keep_standby_warm=True)]
    config["grid"] = dict(wind=[0, 1], pv=[1], diesel=[2, 3], battery_energy=[12, 24, 40], pcs=[3])
    config["reliability"].update(samples=4, validation_samples=8, eens_limit_kwh=100., cvar_limit_kwh=300.)
    return data, weather, config, settings


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    data, weather, config, settings = synthetic_case()
    p1 = output / "physics_72h"
    p1.mkdir()
    units = dict(wind=1, pv=1, diesel=3, battery_energy=40, pcs=3)
    policy = RecoveryPolicy(**config["policies"][1])
    result = simulate(data, weather, units, policy, settings, ScriptedNoise(), trace=True)
    audit = audit_trace(result, data, units, settings)
    if not audit["passed"]:
        raise AssertionError(audit)
    save_trace(p1, result, audit)
    write_json(p1 / "summary.json", dict(synthetic=True, loss_kwh=result.loss_kwh,
               operating_cost_yuan=result.operating_cost_yuan, metrics=result.metrics,
               script="access blocks 0:6,23:32,47:55; run failures at boundaries 25 and 49; residual ice retained"))

    comparisons = []
    plans = {}
    for mode in ("R0", "R1", "R2", "R3"):
        path = output / f"exhaustive_{mode}"
        (path / "checkpoints").mkdir(parents=True)
        local = copy.deepcopy(config)
        local["planning"] = dict(early_risk_screen=False, investment_bound_pruning=False)
        plan = plan_grid(data, weather, local, replace(settings, physical_mode=mode), path)
        plans[mode] = plan
        write_json(path / "plan.json", plan)
        incumbent = plan["incumbent"]
        if incumbent:
            # All four chosen designs/policies evaluated in identical FULL R3.
            checked = evaluate_policy(data, weather, incumbent["units"], RecoveryPolicy(**incumbent["policy"]),
                                      settings, config["reliability"], PathArchive(path / "full_evaluator.jsonl"),
                                      early_screen=False)
            comparisons.append(dict(mode=mode, units=incumbent["units"], policy=incumbent["policy"]["name"],
                                    own_model_cost_yuan=incumbent["objective_yuan"], own_model_risk=incumbent["risk"],
                                    full_model_cost_yuan=checked.get("objective_yuan"),
                                    full_model_risk=checked.get("risk"), full_model_feasible=checked["feasible"]))
        else:
            comparisons.append(dict(mode=mode, status=plan["status"]))

    accelerated_path = output / "accelerated_R3"
    (accelerated_path / "checkpoints").mkdir(parents=True)
    config["planning"] = dict(early_risk_screen=True, investment_bound_pruning=True)
    accelerated = plan_grid(data, weather, config, settings, accelerated_path)
    if accelerated["incumbent"] != plans["R3"]["incumbent"]:
        raise AssertionError("Accelerated and exhaustive optima differ")
    before = time.perf_counter()
    resumed = plan_grid(data, weather, config, settings, accelerated_path)
    resume_seconds = time.perf_counter() - before
    if resumed["incumbent"] != accelerated["incumbent"]:
        raise AssertionError("Resume changed optimum")
    write_json(accelerated_path / "plan.json", accelerated)
    write_csv(output / "physical_ablation.csv", comparisons)
    report = dict(synthetic=True, physical_audit_passed=audit["passed"],
                  exhaustive_accelerated_same_optimum=True, cached_resume_same_optimum=True,
                  exhaustive_seconds=plans["R3"]["training_seconds"],
                  accelerated_seconds=accelerated["training_seconds"], cached_resume_seconds=resume_seconds,
                  grid_designs=plans["R3"]["total_designs"], policies=len(config["policies"]),
                  scenarios=config["reliability"]["samples"], comparisons=comparisons,
                  cuts_generated=0, advanced_lp_cuts="not implemented; no claim of coupled-cut speedup")
    write_json(output / "summary.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
