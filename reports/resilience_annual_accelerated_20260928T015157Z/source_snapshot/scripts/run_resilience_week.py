#!/usr/bin/env python3
"""Plan a 168-hour slice with a linked 72-hour stress window and 12-hour repairs."""
import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.resilience_v2.model import SyntheticYear
from polar_reliability_planning.resilience_milp.model import MILPConfig, FaultRecoveryConfig
from polar_reliability_planning.resilience_milp.annual import annual_scenarios, AnnualResilienceMILP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'reports/resilience_joint_comparison_15000s_20260924T013515Z/shared_year.npz'


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_week(source, start, hours=168):
    with np.load(source, allow_pickle=False) as saved:
        length = len(saved['timestamps'])
        if start < 0 or start % 24 or start + hours > length:
            raise ValueError('week must start at midnight and fit the saved input')
        data = {f.name: saved[f.name][start:start+hours].copy() for f in fields(SyntheticYear)}
    data['timestamps'] = tuple(map(str, data['timestamps']))
    return SyntheticYear(**data)


def write_report(output, result, recovery):
    def val(value):
        if value is None:
            return '未获得'
        return f'{value:,.4f}' if isinstance(value, float) else str(value)
    lines = ['# 168小时规划与12小时设备恢复：运行结果', '',
        '主时域168小时，嵌入一个72小时、16分支韧性窗口。全部设备按整数模块自由规划。',
        f'柴发运行故障、启动失败、PCS、UPS和新能源母线恢复均设为{recovery.diesel_run_hours}小时；主母线强制失电{recovery.main_bus_hours}小时。',
        f'UPS最低桥接时长为{result.get("resolved_config", {}).get("ups_bridge_hours", "见运行定义")}小时。目标是设备投资加168小时期望运行和失供费用，不能据此当作全年经济性结论。',
        'EENS限额100 kWh、95% CVaR上界限额1000 kWh沿用现值，本次指标均仅对应168小时时域。', '',
        '| 指标 | 结果 |', '|---|---:|']
    for name, key in [('状态', 'status'), ('建模时间/秒', 'build_seconds'), ('累计求解时间/秒', 'runtime_seconds'),
                      ('计算阶段耗时/秒', 'total_elapsed_seconds'), ('含导出总耗时/秒', 'process_wall_seconds'),
                      ('目标费用/元', 'objective_yuan'), ('目标下界/元', 'objective_bound_yuan'),
                      ('相对最优性差距', 'mip_gap'), ('投资/元', 'investment_yuan'),
                      ('期望运行及失供费用/元', 'expected_operation_yuan'),
                      ('168小时EENS/kWh', 'eens_kwh'), ('168小时CVaR上界/kWh', 'cvar_upper_bound_kwh')]:
        lines.append(f'| {name} | {val(result.get(key))} |')
    if result.get('selected'):
        lines += ['', '## 设备配置', '', '| 设备 | 容量 | 整数模块数 |', '|---|---:|---:|']
        for key, name in [('wind_kw', '风电/kW'), ('pv_kw', '光伏/kW'), ('diesel_units', '柴发/台，每台100kW'),
                          ('battery_kwh', '普通电池/kWh'), ('pcs_kw', '储能PCS/kW'),
                          ('ups_kwh', 'UPS电量/kWh'), ('ups_kw', 'UPS功率/kW')]:
            lines.append(f'| {name} | {val(result["selected"][key])} | {result["modules"][key]} |')
        lines += ['', '## 韧性分支', '', '| 分支 | 权重 | 常规失供/kWh | 核心失供/kWh | UPS供能/kWh |', '|---|---:|---:|---:|---:|']
        for row in result['scenario_metrics']:
            lines.append(f'| {row["name"]} | {row["conditional_weight"]:.3f} | {row["regular_loss_kwh"]:.4f} | {row["core_loss_kwh"]:.4f} | {row["ups_emergency_kwh"]:.4f} |')
        lines += ['', f'物理与信息约束审计：{result["audit"]["passed"]}；最大残差：{result["audit"]["max_violation"]:.6g}。']
    else:
        lines += ['', '当前没有完整整数可行方案，不能给出容量结论；未找到解与已证明不可行须按状态区分。']
    lines += ['', '原始记录：[汇总](summary.json)、[求解日志](gurobi.log)、[运行定义](run_definition.json)。', '']
    (output / 'RESULT_REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--start-hour', type=int, default=792)
    parser.add_argument('--stress-start-hour', type=int, default=24)
    parser.add_argument('--recovery-hours', type=int, default=12)
    parser.add_argument('--main-bus-hours', type=int, default=1)
    parser.add_argument('--ups-bridge-hours', type=float, default=12.)
    parser.add_argument('--time-limit-seconds', type=float, default=15000.)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any((output / x).exists() for x in ('run_status.json', 'summary.json')):
        parser.error('output already contains a run; use a new directory')
    cfg = replace(MILPConfig(), time_limit_seconds=args.time_limit_seconds,
                  threads=args.threads, ups_bridge_hours=args.ups_bridge_hours)
    cfg.check()
    recovery = FaultRecoveryConfig(
        renewable_bus_normal_hours=args.recovery_hours, renewable_bus_storm_hours=args.recovery_hours,
        pcs_hours=args.recovery_hours, ups_hours=args.recovery_hours,
        diesel_run_hours=args.recovery_hours, diesel_start_hours=args.recovery_hours,
        main_bus_hours=args.main_bus_hours)
    recovery.check()
    year = load_week(args.source, args.start_hour)
    base, windows = annual_scenarios(year, starts=(args.stress_start_hour,), recovery=recovery)
    output.mkdir(parents=True, exist_ok=True)
    base.year.frame().to_csv(output / 'input_168h.csv', index=False)
    np.savez_compressed(output / 'input_168h.npz', **{f.name: np.asarray(getattr(base.year, f.name)) for f in fields(SyntheticYear)})
    sources = [Path(__file__).resolve(), ROOT / 'polar_reliability_planning/resilience_milp/model.py',
               ROOT / 'polar_reliability_planning/resilience_milp/annual.py', ROOT / 'polar_reliability_planning/resilience_v2/model.py']
    for source in sources:
        target = output / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    definition = dict(hours=168, window_hours=72, window_start=args.stress_start_hour,
        conditional_branch_count=16, source=str(args.source.resolve()), source_start_hour=args.start_hour,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        timestamps_first=year.timestamps[0], timestamps_last=year.timestamps[-1],
        all_capacities_optimized=True, synthetic_data=True, config=asdict(cfg), recovery=asdict(recovery),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cost_basis='investment plus 168 hours expected operation; no annual extrapolation',
        risk_basis='168-hour EENS and CVaR upper bound; thresholds unchanged, not annual metrics')
    write_json(output / 'run_definition.json', definition)
    write_json(output / 'resolved_config.json', asdict(cfg))
    # Expose source events and exact durations even before a solution exists.
    scenario_input = []
    for p in windows[0]['paths']:
        scenario_input.append(dict(name=p.name, weight=p.weight, run_repair_hours=p.run_repair_hours,
            start_repair_hours=p.start_repair_hours,
            renewable_bus_outage_hours=np.flatnonzero(p.renewable_bus_fault).tolist(),
            main_bus_outage_hours=np.flatnonzero(~p.main_bus_available).tolist(),
            pcs_outage_hours=np.flatnonzero(~p.pcs_available).tolist(),
            ups_outage_hours=np.flatnonzero(~p.ups_available).tolist()))
    write_json(output / 'scenario_inputs.json', scenario_input)
    started = time.monotonic()
    status = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat())
    import os
    status['pid'] = os.getpid()
    write_json(output / 'run_status.json', status)
    first_feasible = None
    last_progress = -60.
    def callback(model, where):
        nonlocal first_feasible, last_progress
        elapsed = time.monotonic() - started
        if where == GRB.Callback.MIPSOL:
            if first_feasible is None:
                first_feasible = elapsed
            progress = dict(stage='integer_incumbent', elapsed_seconds=elapsed,
                            objective_yuan=model.cbGet(GRB.Callback.MIPSOL_OBJ))
        elif where == GRB.Callback.MIP and elapsed-last_progress >= 60:
            last_progress = elapsed
            hi = model.cbGet(GRB.Callback.MIP_OBJBST)
            lo = model.cbGet(GRB.Callback.MIP_OBJBND)
            progress = dict(stage='optimizing', elapsed_seconds=elapsed,
                objective_yuan=float(hi) if abs(hi) < GRB.INFINITY else None,
                bound_yuan=float(lo) if abs(lo) < GRB.INFINITY else None)
        else:
            return
        write_json(output / 'progress.json', progress)
        with (output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(progress) + '\n')
    try:
        budget = cfg.initial_economic_budget_yuan
        rounds = []
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            while True:
                planner = AnnualResilienceMILP(base, windows, cfg, budget, env)
                try:
                    m = planner.model
                    m.Params.OutputFlag = 1
                    m.Params.LogToConsole = 0
                    m.Params.LogFile = str(output / 'gurobi.log')
                    m.Params.Method = 3
                    remaining = cfg.time_limit_seconds - (time.monotonic()-started)
                    if cfg.time_limit_seconds and remaining <= 0:
                        result = dict(status='time_limit_during_model_build', selected=None,
                                      economic_domain_certified=False, build_seconds=planner.build_seconds)
                    else:
                        m.Params.TimeLimit = remaining if cfg.time_limit_seconds else GRB.INFINITY
                        m.optimize(callback)
                        result = planner.result()
                    rounds.append(dict(economic_budget_yuan=budget, build_seconds=planner.build_seconds,
                                       solver_runtime_seconds=float(m.Runtime), status=result['status']))
                    elapsed = time.monotonic()-started
                    done = (result.get('economic_domain_certified') or
                            (cfg.time_limit_seconds and elapsed >= cfg.time_limit_seconds) or
                            m.Status in (GRB.TIME_LIMIT, GRB.INTERRUPTED))
                    if done:
                        result.update(total_elapsed_seconds=elapsed, economic_bound_rounds=rounds,
                            runtime_seconds=sum(r['solver_runtime_seconds'] for r in rounds),
                            build_seconds=sum(r['build_seconds'] for r in rounds),
                            recovery=asdict(recovery), first_solver_feasible_seconds=first_feasible,
                            hours=168, all_capacities_optimized=True)
                        planner.save(output, result)
                        break
                    budget *= 2
                finally:
                    planner.close()
        result['process_wall_seconds'] = time.monotonic()-started
        write_json(output / 'summary.json', result)
        write_report(output, result, recovery)
        status.update(status='finished', result_status=result['status'], has_feasible_solution=bool(result.get('selected')),
                      selected=result.get('selected'), objective_yuan=result.get('objective_yuan'),
                      mip_gap=result.get('mip_gap'), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      process_wall_seconds=time.monotonic()-started)
        write_json(output / 'run_status.json', status)
        print(json.dumps({k: result.get(k) for k in ('status', 'selected', 'objective_yuan', 'mip_gap', 'eens_kwh', 'cvar_upper_bound_kwh', 'total_elapsed_seconds')}, ensure_ascii=False, indent=2))
    except BaseException:
        status.update(status='error', error=traceback.format_exc(), process_wall_seconds=time.monotonic()-started)
        write_json(output / 'run_status.json', status)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
