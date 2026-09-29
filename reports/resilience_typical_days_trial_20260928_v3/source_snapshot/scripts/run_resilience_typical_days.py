#!/usr/bin/env python3
"""Monthly medoid capacity search followed by original-8760-hour validation."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.resilience_milp.model import MILPConfig, FaultRecoveryConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP, annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import (
    merge_duplicate_paths, seed_modules, fix_seed_policy, apply_start, read_solution,
)
from polar_reliability_planning.resilience_milp.typical_days import (
    cluster_days, TypicalDayResilienceMILP, remove_exact_duplicate_constraints,
)
from scripts.run_resilience_week import ROOT, DEFAULT_INPUT, load_week, write_json


def transfer_calendar_start(source, target):
    """Expand representative controls into every original calendar occurrence."""
    source.model.update(); target.model.update()
    values = source.model.getAttr('X')
    by_name = dict(zip(source.model.getAttr('VarName'), values))
    pattern = re.compile(r's(\d+)_(\w+)\[([0-9,]+)\]$')
    starts = []; missing = []
    for var in target.model.getVars():
        match = pattern.fullmatch(var.VarName)
        value = by_name.get(var.VarName)
        if match:
            s, name, indices = match.groups()
            key = tuple(map(int, indices.split(',')))
            if len(key) == 1: key = key[0]
            try: value = values[source.blocks[int(s)][name][key].index]
            except KeyError: pass
        if value is None:
            if var.LB == var.UB: value = var.LB
            else: missing.append(var.VarName); value = GRB.UNDEFINED
        starts.append(value)
    target.model.setAttr('Start', target.model.getVars(), starts)
    target.model.update()
    return dict(missing_count=len(missing), missing_examples=missing[:20])


def solve_and_audit(planner, seconds, log, callback=None):
    """Retry numerical audit failure without presolve; never accept its result."""
    m = planner.model
    m.Params.OutputFlag = 1; m.Params.LogToConsole = 0; m.Params.LogFile = str(log)
    m.Params.Method = 1; m.Params.NumericFocus = 2
    # Bound preprocessing work on the millions of repeated chronological rows.
    # An unrestricted presolve consumed the entire first 300-second trial.
    m.Params.Presolve = 1; m.Params.PrePasses = 1
    m.Params.TimeLimit = max(.01, seconds)
    m.optimize(callback)
    try:
        return planner.result()
    except RuntimeError as error:
        if 'audit failed' not in str(error): raise
        write_json(log.with_suffix('.failed_audit.json'), dict(error=str(error), accepted=False))
        m.reset(); m.Params.Presolve = 0; m.Params.TimeLimit = max(.01, seconds)
        m.optimize(callback)
        result = planner.result()
        result['numerical_retry_without_presolve'] = True
        return result


def report(output, summary):
    lines = ['# 典型日规划与原始全年复核', '',
        '聚类使用当前模拟全年输入；各月选择原始日作为代表，保留极端连续时段和四个72小时事故窗口。',
        '普通日共用运行决策，电池/UPS电量、柴油温度及所有跨日约束仍按365天连续计算。',
        '聚类同时近似了输入并限制了日内策略，其最优性下界不能用于原始全年问题。', '',
        '| 典型日 | 保留日 | 压缩规划费用/元 | 原始全年费用/元 | 全年EENS/kWh | 全年CVaR上界/kWh | 原始全年审计 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for run in summary.get('runs', []):
        plan, reduced, full = run['aggregation'], run['planning'], run.get('validation', {})
        def value(data, key):
            v = data.get(key)
            return '未获得' if v is None else f'{v:,.4f}'
        lines.append(f"| {plan['ordinary_typical_days']} | {plan['protected_days']} | {value(reduced,'objective_yuan')} | "
                     f"{value(full,'objective_yuan')} | {value(full,'eens_kwh')} | {value(full,'cvar_upper_bound_kwh')} | "
                     f"{full.get('audit',{}).get('passed',False)} |")
        if reduced.get('modules'):
            lines += ['', f"## {plan['ordinary_typical_days']}个普通典型日", '',
                f"容量模块：`{json.dumps(reduced['modules'],ensure_ascii=False)}`。",
                f"规划状态：{reduced['status']}；压缩模型gap：{reduced.get('mip_gap')}。",
                f"原始全年复核状态：{full.get('status','未执行')}；固定容量调度gap：{full.get('mip_gap')}。",
                f"压缩模型变量：{reduced.get('variables')}；二进制变量：{reduced.get('binary_variables')}；"
                f"线性约束：{reduced.get('linear_constraints')}。",
                f"原始全年固定容量变量：{full.get('variables')}；二进制变量：{full.get('binary_variables')}；"
                f"线性约束：{full.get('linear_constraints')}。两者柴油槽位数不同，不作为严格等条件规模对比。",
                f"详情：[压缩规划](k{plan['ordinary_typical_days']}/planning/summary.json)、"
                f"[全年复核](k{plan['ordinary_typical_days']}/validation/summary.json)。"]
    lines += ['', '全年复核通过仅证明候选容量在原始全年有限场景模型内可行。',
        '固定容量复核的gap只衡量该容量下的调度费用；不证明全年容量规划全局最优。',
        '计算时长为本次实现、求解参数与预算下的实测值；没有同条件全年自由容量对照时不报告求解加速倍数。', '']
    (output/'RESULT_REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--per-month', type=int, nargs='+', default=[2, 3], choices=[1, 2, 3])
    parser.add_argument('--planning-seconds', type=float, default=600.)
    parser.add_argument('--validation-seconds', type=float, default=180.)
    parser.add_argument('--seed-seconds', type=float, default=120.)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--mip-gap', type=float, default=.01)
    parser.add_argument('--extra-days', type=int, nargs='*', default=[])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): parser.error('use a new output directory')
    if min(args.planning_seconds, args.validation_seconds, args.seed_seconds) <= 0: parser.error('positive budgets required')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cfg = replace(MILPConfig(), ups_bridge_hours=12., time_limit_seconds=args.planning_seconds,
                  threads=args.threads, mip_gap=args.mip_gap)
    cfg.check(); recovery = FaultRecoveryConfig()
    summary = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), runs=[],
        config=asdict(cfg), recovery=asdict(recovery), source=str(args.source.resolve()),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        budget_scope='per solve; model construction, audit/export and numerical retry are additional',
        original_year_capacity_optimality_certified=False)
    def event(stage, **data):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **data)
        with (output/'events.jsonl').open('a') as f: f.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(output/'progress.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    def persist():
        summary['elapsed_seconds'] = time.monotonic()-started
        write_json(output/'summary.json', summary); report(output, summary)
    last = {}
    def callback(stage):
        def cb(m, where):
            elapsed = time.monotonic()-started
            if where == GRB.Callback.MIPSOL:
                event('incumbent', solve=stage, objective=float(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
            elif where == GRB.Callback.MIP and elapsed-last.get(stage, -60) >= 45:
                last[stage] = elapsed
                best = m.cbGet(GRB.Callback.MIP_OBJBST); bound = m.cbGet(GRB.Callback.MIP_OBJBND)
                event('solver_progress', solve=stage, objective=None if abs(best) >= GRB.INFINITY else float(best),
                      bound=None if abs(bound) >= GRB.INFINITY else float(bound))
        return cb
    sources = [Path(__file__).resolve(), ROOT/'scripts/run_resilience_week.py',
        *[ROOT/'polar_reliability_planning/resilience_milp'/name for name in
          ('model.py', 'annual.py', 'acceleration.py', 'typical_days.py')]]
    for source in sources:
        target = output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    write_json(output/'run_definition.json', summary)
    year = load_week(args.source, 0, 8760)
    original_base, original_windows = annual_scenarios(year, recovery=recovery)
    original_windows, aliases = merge_duplicate_paths(original_windows)
    write_json(output/'scenario_aliases.json', aliases)
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0); env.start()
            for per_month in args.per_month:
                aggregation = cluster_days(year, per_month=per_month, extra_days=args.extra_days)
                plan = aggregation.report(year)
                directory = output/f'k{plan["ordinary_typical_days"]}'; directory.mkdir()
                write_json(directory/'aggregation.json', plan)
                base, windows = annual_scenarios(aggregation.reconstruct(year), recovery=recovery)
                windows, _ = merge_duplicate_paths(windows)
                event('clustered', typical_days=plan['ordinary_typical_days'], protected_days=plan['protected_days'],
                      dispatch_days=plan['distinct_dispatch_days'])
                point = seed_modules(base, cfg)
                event('seed_building', modules=point)
                seed = TypicalDayResilienceMILP(base, windows, cfg, aggregation, year,
                    env=env, fixed_modules=point, compact_fixed=True)
                try:
                    fix_seed_policy(seed)
                    event('seed_solving', variables=seed.model.NumVars)
                    seed_result = solve_and_audit(seed, args.seed_seconds, directory/'seed.log')
                    seed.save(directory/'seed', seed_result, write_model=False)
                    if not seed_result.get('selected'): raise RuntimeError('restricted seed unresolved; no validated warm start')
                    event('seed_audited', objective=seed_result['objective_yuan'], audit=seed_result['audit'])
                finally: seed.close()
                budget = max(cfg.initial_economic_budget_yuan, seed_result['objective_yuan']+1.)
                event('planning_building', economic_budget=budget)
                planner = TypicalDayResilienceMILP(base, windows, cfg, aggregation, year,
                    env=env, economic_budget=budget)
                try:
                    before = dict(variables=planner.model.NumVars, binaries=planner.model.NumBinVars,
                                  constraints=planner.model.NumConstrs, diesel_slots=planner.dmax)
                    dedup = remove_exact_duplicate_constraints(planner.model)
                    transfer = apply_start(planner, read_solution(directory/'seed/solution.sol'), point['diesel_units'])
                    if transfer['missing_count']: raise RuntimeError(f'incomplete representative warm start: {transfer}')
                    event('planning_solving', **before, **dedup)
                    result = solve_and_audit(planner, args.planning_seconds, directory/'planning.log', callback(f'k{per_month*12}'))
                    result.update(pre_dedup_statistics=before, deduplication=dedup, warm_start=transfer)
                    planner.save(directory/'planning', result, write_model=False)
                    run = dict(aggregation=plan, planning=result)
                    summary['runs'].append(run); persist()
                    if not result.get('selected'):
                        event('planning_unresolved', status=result['status']); continue
                    event('planning_audited', modules=result['modules'], objective=result['objective_yuan'], gap=result['mip_gap'])
                    event('validation_building', original_hours=8760, modules=result['modules'])
                    validation = AnnualResilienceMILP(original_base, original_windows, cfg,
                        env=env, economic_budget=budget, fixed_modules=result['modules'], compact_fixed=True, compact=True)
                    try:
                        transferred = transfer_calendar_start(planner, validation)
                        if transferred['missing_count']: raise RuntimeError(f'incomplete calendar expansion: {transferred}')
                        variables = validation.model.getVars()
                        fixed = [(v, v.LB, v.UB) for v in variables if v.VType != GRB.CONTINUOUS]
                        for v, _, _ in fixed:
                            value = round(v.Start); v.LB = value; v.UB = value
                        validation.model.update()
                        event('validation_policy_solving', variables=validation.model.NumVars)
                        restricted = solve_and_audit(validation, args.seed_seconds, directory/'validation_policy.log')
                        restricted['scope'] = 'original year with candidate integer controls fixed; feasibility witness only'
                        restricted['status'] = ('original_year_restricted_policy_feasible' if restricted.get('selected')
                                                else 'restricted_policy_unresolved_not_capacity_infeasible')
                        restricted.pop('objective_bound_yuan', None); restricted.pop('mip_gap', None)
                        write_json(directory/'validation_policy.json', restricted)
                        if restricted.get('selected'):
                            validation.model.setAttr('Start', variables, validation.model.getAttr('X', variables))
                        for v, lb, ub in fixed: v.LB = lb; v.UB = ub
                        validation.model.update()
                        # All capacity-fixing equalities remain, but operating
                        # decisions are now free on the original hourly inputs.
                        event('validation_free_dispatch_solving', policy_feasible=bool(restricted.get('selected')))
                        full = solve_and_audit(validation, args.validation_seconds, directory/'validation.log', callback(f'validation{per_month*12}'))
                        if full.get('selected'):
                            full['full_year_original_data_validated'] = True
                            full['original_year_capacity_optimality_certified'] = False
                            full['cost_change_from_reduced_fraction'] = full['objective_yuan']/result['objective_yuan']-1
                        elif restricted.get('selected'):
                            # Keep the separately audited original-year witness
                            # even if a later free-dispatch solve is unresolved.
                            full['prior_original_year_feasible_witness'] = restricted
                        validation.save(directory/'validation', full, write_model=False)
                        run['validation'] = full
                        event('validation_finished', status=full['status'], audit=full.get('audit'),
                              objective=full.get('objective_yuan'), eens=full.get('eens_kwh'),
                              cvar=full.get('cvar_upper_bound_kwh'))
                        persist()
                    finally: validation.close()
                finally: planner.close()
        summary['status'] = ('completed_with_full_year_validated_candidates'
            if any(r.get('validation', {}).get('full_year_original_data_validated') for r in summary['runs'])
            else 'completed_without_full_year_validated_candidate')
        persist(); event('completed', status=summary['status'])
    except Exception:
        summary.update(status='error', error=traceback.format_exc()); persist()
        event('error', error=summary['error']); raise


if __name__ == '__main__': main()
