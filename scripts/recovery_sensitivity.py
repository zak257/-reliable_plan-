"""Fixed-design diagnostics on training primitives; never reselects on holdout."""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from polar_reliability_planning.recovery_cli import load_inputs, write_csv, write_json
from polar_reliability_planning.reliability.causal_simulator import simulate
from polar_reliability_planning.reliability.recovery_policy import RecoveryPolicy
from polar_reliability_planning.reliability.risk_bounds import risk_summary
from polar_reliability_planning.scenario_generation.primitive_noise import PrimitiveNoise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads((args.run / "resolved_config.json").read_text())["config"]
    frozen = json.loads((args.run / "frozen_design.json").read_text())
    units, policy = frozen["units"], RecoveryPolicy(**frozen["policy"])
    changes = [("baseline", {})]
    changes += [(f"warmup_{h}h", {"recovery": {"warmup_hours": h}}) for h in (0, 168, 672)]
    changes += [(f"heater_{p}kW", {"thermal": {"heater_max_kw": p}}) for p in (3., 6., 24.)]
    changes += [(f"mttf_{m}h", {"diesel": {"mttf_hours": m}}) for m in (1180., 2410.)]
    changes += [(f"start_failure_{p}", {"diesel": {"start_failure_probability": p}}) for p in (.0010, .0017)]
    changes += [("fixed_37h_repair", {"diesel": {"repair_distribution": "fixed"}}),
                ("two_startup_crews", {"recovery": {"crews": 2}}),
                ("all_work_safe", {"recovery": {"personnel_mode": "all_work_safe"}})]
    changes += [(f"residual_ice_{h}h", {"weather": {"residual_icing_hours": h}}) for h in (0, 24)]
    changes += [(f"physical_{mode}", {"recovery": {"physical_mode": mode}}) for mode in ("R0", "R1", "R2")]
    rows = []
    for name, change in changes:
        config = copy.deepcopy(base)
        for section, update in change.items():
            config[section].update(update)
        data, weather, settings = load_inputs(config)
        started = time.perf_counter()
        results = []
        for scenario in range(base["reliability"]["samples"]):
            noise = PrimitiveNoise(base["reliability"]["seed"], scenario, data.hours,
                                   {k: b[1] for k, b in data.unit_bounds.items()}, settings.warmup_hours)
            results.append(simulate(data, weather, units, policy, settings, noise))
        valid = all(r.valid for r in results)
        row = dict(case=name, change=change, valid=valid, seconds=time.perf_counter() - started)
        if valid:
            row.update(risk_summary([r.loss_kwh for r in results], base["reliability"]["alpha"]))
            row.update({k: sum(r.metrics[k] for r in results) / len(results)
                        for k in ("heater_kwh", "diesel_fuel_kg", "battery_discharge_kwh",
                                  "wait_access_unit_hours", "wait_prep_unit_hours", "wait_heat_unit_hours")})
            row["expected_operating_cost_yuan"] = sum(r.operating_cost_yuan for r in results) / len(results)
        rows.append(row)
        print(json.dumps(row), flush=True)
        write_csv(args.output / "fixed_design_sensitivity.csv", rows)
    write_json(args.output / "summary.json", dict(synthetic=True, frozen_design=frozen,
               sample_role="training_primitives_descriptive_only", scenarios=base["reliability"]["samples"],
               reselection=False, results=rows, source_run=str(args.run.resolve())))


if __name__ == "__main__":
    main()
