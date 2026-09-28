"""Reproducible recovery planning CLI. Legacy main/CLI remains unchanged."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .config import UnitCommitmentOptions
from .data.cap_plan_loader import load_case
from .planning.recovery_optimizer import PathArchive, evaluate_policy, plan_grid
from .recovery.config import read_recovery_config, settings_from_config
from .reliability.causal_simulator import simulate
from .reliability.recovery_policy import RecoveryPolicy
from .scenario_generation.primitive_noise import NOISE_VERSION, PrimitiveNoise
from .validation.recovery_audit import audit_trace
from .weather.hourly_weather import load_weather

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        if not records:
            return
        keys = list(dict.fromkeys(k for row in records for k in row))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def load_inputs(config):
    settings = settings_from_config(config)
    data = load_case(config["data_root"], config["case"], config.get("start_hour", 0), config["hours"],
                     config["modules"], config.get("load_scale", 1.0),
                     UnitCommitmentOptions(True, settings.min_output_fraction,
                                           settings.min_up_hours, settings.min_down_hours, settings.startup_cost_yuan),
                     config.get("max_units"))
    weather = load_weather(data, config)
    # Override legacy metadata only on this new in-memory object.
    data.input_metadata.update(continuous_cyclic_soc=False, diesel_dispatch="causal_recovery_v1",
                               initial_state="cold_then_declared_cyclic_weather_warmup",
                               terminal_state="free_continuation_no_free_energy_or_repair")
    return data, weather, settings


def fingerprint(config, data, weather):
    files = sorted((ROOT / "polar_reliability_planning").rglob("*.py"))
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    contents = dict(config=config, input_manifest=data.manifest, weather=weather.metadata,
                    source_sha256=sources, noise_version=NOISE_VERSION,
                    numpy_version=np.__version__, python_version=platform.python_version())
    digest = hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()
    return digest, contents


def migration_text(data, weather, config):
    return f"""# Recovery-v1 迁移核查

- 本地旧提交：`{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}`。
- 旧配置 `config/zhongshan_nonanticipative_2000.toml` 保留；柴油模块 300 → 100 kW，模块年资本成本和容量上限通过原加载器重算。
- 复用 `data/cap_plan_loader.py` 的成本、风光曲线与补值逻辑，复用精确离散 CVaR；不复用旧日历柴油可用率、单调割或容量缓存。
- 新实现：实际在线暴露、有效启动需求、日历维修、人员排队、并行准备/热状态、实际电加热、连续 SOC 与因果策略库。
- 原始数据 SHA256 见 run_manifest.json；原数据文件没有改写。
- 数据含气温；记录数 {weather.metadata['available_source_hours']}，本次连续选取 {data.hours} 小时，{weather.metadata['period_start']} 至 {weather.metadata['period_end']}。2020 闰年完整数据是 8784 小时，8760 小时实验不称完整日历年。
- CSV 全数据补值数：{json.dumps(data.input_metadata['imputed_values_in_full_csv'], ensure_ascii=False)}。气温不做虚构补值。
- 时间戳原文件未注明时区，当前声明：{config['data_contract']['timezone']}，状态 {config['data_contract']['timezone_status']}。
- 天气标签：{weather.metadata['weather_label_status']}。并未根据低温声称识别真实暴风雪或覆冰。
- synthetic={config['synthetic']}。本配置热参数、箱体映射、人数和辅助负荷口径仍为研究假设，正式配置必须提供来源。
- 柴油 100 kW 是研究额定值，最低出力沿用 20% 研究假设；4.5 kWh/kg 是旧 CSV 的线性成本参数，并非已确认本模块的 CAT 油耗曲线。
- 37 h 几何日历修复及启动失败后的相同恢复规律为参考假设；修复完成保留温度，准备交接可配置，启动班组仍需合法到场。
- 热身 {config['recovery']['warmup_hours']} 小时采用年度末段循环拼接，不宣称已平稳；费用与风险只统计所选主时域，库存差单列。
- 维修班组与启动班组分开处理，未假定 37 h 是共享班组的人工工时；未生成无频次依据的计划维护。
- 求解范围：预声明网格 × 策略库 × 固定样本。经济下界仅用投资成本，不调用 Gurobi；不声称已实现高级恢复 LP 耦合割。
- Python {platform.python_version()}，NumPy {np.__version__}。完整源文件指纹、随机机制版本与策略指纹见 run_manifest.json。
"""


def save_trace(output, result, audit):
    write_csv(output / "hourly_dispatch.csv", result.hourly)
    write_csv(output / "event_log.csv", result.events)
    write_csv(output / "delay_decomposition.csv", result.delays)
    write_json(output / "audit_report.json", audit)


def run(args, output):
    started = time.perf_counter()
    config = read_recovery_config(args.config)
    for key in ("hours",):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("samples", "validation_samples"):
        if getattr(args, key) is not None:
            value = getattr(args, key)
            if value < (1 if key == "samples" else 0):
                raise ValueError(f"Invalid {key}")
            config["reliability"][key] = value
    if args.wind_mode:
        config["weather"]["wind_mode"] = args.wind_mode
    if args.no_screen:
        config["planning"]["early_risk_screen"] = False
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("max-designs must be positive")
    data, weather, settings = load_inputs(config)
    digest, inputs = fingerprint(config, data, weather)
    manifest = output / "run_manifest.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text())["run_sha256"] != digest:
            raise ValueError("Resume fingerprint mismatch: config/data/code/random mechanism changed")
    else:
        write_json(manifest, dict(run_sha256=digest, created_utc=datetime.now(timezone.utc).isoformat(), **inputs))
    write_json(output / "resolved_config.json", dict(config=config, settings=asdict(settings),
               unit_bounds=data.unit_bounds, annual_cost_per_unit=data.annual_cost_per_unit,
               input_metadata=data.input_metadata, weather_metadata=weather.metadata,
               policies=[dict(**p, sha256=RecoveryPolicy(**p).fingerprint) for p in config["policies"]]))
    (output / "migration_audit.md").write_text(migration_text(data, weather, config))
    weather_fields = ("load_kw", "ambient_c", "wind_speed_ms", "wind_clean_pu", "pv_pu",
                      "access_safe", "icing_active", "ice_power_factor", "wind_protective_stop",
                      "extreme_hazard", "hard_stop")
    write_csv(output / "hourly_weather.csv", [dict(timestamp=weather.timestamps[h],
              **{k: getattr(weather, k)[h].item() for k in weather_fields}) for h in range(data.hours)])
    if args.command == "preflight":
        print(json.dumps(dict(status="preflight_passed", synthetic=config["synthetic"],
                             weather=weather.metadata, unit_bounds=data.unit_bounds), ensure_ascii=False))
        return 0
    last_print = [0.0]
    def progress(value):
        write_json(output / "progress.json", dict(elapsed_seconds=time.perf_counter() - started, **value))
        now = time.perf_counter()
        if now - last_print[0] >= 5:
            print(json.dumps(value, ensure_ascii=False), flush=True)
            last_print[0] = now
    if args.command == "evaluate":
        if not args.units:
            raise ValueError("evaluate requires --units JSON module counts")
        units = data.validate_units(json.loads(args.units))
        policy = next((RecoveryPolicy(**p) for p in config["policies"] if p["name"] == args.policy), None)
        if policy is None:
            raise ValueError("Choose --policy from frozen configured library")
        archive = PathArchive(output / "checkpoints" / "paths.jsonl")
        checked = evaluate_policy(data, weather, units, policy, settings, config["reliability"], archive, early_screen=False)
        plan = dict(status="fixed_capacity_evaluation", incumbent=dict(units=units, capacities=data.capacities(units),
                   policy=asdict(policy), policy_sha256=policy.fingerprint,
                   **{k: v for k, v in checked.items() if k != "records"}),
                   population_certified=False, relative_gap=None)
    else:
        plan = plan_grid(data, weather, config, settings, output, args.max_designs, progress)
        write_csv(output / "policy_evaluations.csv", plan.pop("evaluations"))
        with (output / "visited_designs.jsonl").open("w") as stream:
            for item in plan.pop("visited"):
                stream.write(json.dumps(item) + "\n")
        # Intentionally empty: library/visited exclusions are not mathematical cuts.
        (output / "cuts.jsonl").write_text("")
    plan.update(schema_version="recovery-v1", synthetic=config["synthetic"],
                result_kind="synthetic_test_only" if config["synthetic"] else "conditional_site_experiment",
                objective_basis=config["objective_basis"], weather_scope=weather.metadata,
                validation_status="not_requested", terminal_inventory="continuous_state_no_free_reset")
    incumbent = plan["incumbent"]
    if incumbent:
        units, policy = incumbent["units"], RecoveryPolicy(**incumbent["policy"])
        archive = PathArchive(output / "checkpoints" / "paths.jsonl")
        # Freeze capacity and policy before any holdout evaluation.
        write_json(output / "frozen_design.json", incumbent)
        if config["reliability"]["validation_samples"]:
            progress(dict(stage="holdout", capacity=units, policy=policy.name))
            holdout = evaluate_policy(data, weather, units, policy, settings, config["reliability"], archive,
                                      role="holdout", early_screen=False)
            plan["validation"] = {k: v for k, v in holdout.items() if k != "records"}
            plan["validation_status"] = "holdout_passed_empirically" if holdout["feasible"] else "holdout_failed"
        selected = [r for r in archive.records.values() if r["units"] == units and r["policy"] == policy.name]
        write_csv(output / "scenario_losses.csv", [dict(**r, probability=1 / config["reliability"]["samples" if r["role"] == "training" else "validation_samples"])
                                                   for r in selected])
        worst = max(selected, key=lambda r: r["loss_kwh"])
        noise = PrimitiveNoise(worst["seed"], worst["scenario"], data.hours,
                               {k: b[1] for k, b in data.unit_bounds.items()}, settings.warmup_hours)
        replay = simulate(data, weather, units, policy, settings, noise, trace=True)
        audit = audit_trace(replay, data, units, settings)
        audit.update(replay_seed=worst["seed"], replay_scenario=worst["scenario"], replay_role=worst["role"],
                     loss_matches_checkpoint=abs(replay.loss_kwh - worst["loss_kwh"]) < 1e-7,
                     max_hour_boundary_event_localization_error_hours=1,
                     nonanticipation_contract="Observation excludes hidden noise, repair clocks and future realized weather")
        if not audit["loss_matches_checkpoint"]:
            audit["passed"] = False
        save_trace(output, replay, audit)
        plan["audit_passed"] = audit["passed"]
        plan["selected_trace_metrics"] = replay.metrics
        roles = ("training", "holdout")
        diagnostics = {}
        for role in roles:
            records = [r for r in selected if r["role"] == role and r["valid"]]
            if records:
                metrics = records[0]["metrics"]
                diagnostics[role] = {key: sum(r["metrics"][key] for r in records) / len(records)
                                     for key, value in metrics.items() if isinstance(value, (int, float))}
                diagnostics[role]["initial_energy_kwh"] = [r["metrics"]["initial_state"]["energy_kwh"] for r in records]
                diagnostics[role]["final_energy_kwh"] = [r["metrics"]["final_state"]["energy_kwh"] for r in records]
        plan["mean_diagnostics"] = diagnostics
    plan["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "capacity_result.json", plan)
    write_json(output / "summary.json", plan)
    progress(dict(stage="complete", status=plan["status"], validation_status=plan["validation_status"]))
    print(json.dumps(dict(status=plan["status"], result_kind=plan["result_kind"],
                          incumbent=incumbent, validation=plan.get("validation"),
                          elapsed_seconds=plan["elapsed_seconds"]), ensure_ascii=False), flush=True)
    if not incumbent:
        return 2
    if not plan.get("audit_passed"):
        return 4
    if plan["validation_status"] == "holdout_failed":
        return 3
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "plan", "evaluate"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/zhongshan_recovery_stress.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--max-designs", type=int)
    parser.add_argument("--no-screen", action="store_true")
    parser.add_argument("--wind-mode", choices=("protective_shutdown", "operate_with_hazard_multiplier"))
    parser.add_argument("--units")
    parser.add_argument("--policy")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        print("ERROR: output is nonempty; use a new directory or --resume with identical inputs", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    try:
        return run(args, output)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        write_json(output / "error.json", dict(type=type(exc).__name__, error=str(exc)))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
