#!/usr/bin/env python3
"""Continue the unchanged annual MILP from saved incumbents, with a new budget."""
import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import gurobipy as gp
from gurobipy import GRB

from polar_reliability_planning.resilience_milp.model import MILPConfig, FaultRecoveryConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP, annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import merge_duplicate_paths, read_solution, apply_start
from polar_reliability_planning.resilience_milp.continuation import (
    fix_incumbent_commitment, temporarily_fix_battery_modes, restore_bounds,
)
from polar_reliability_planning.resilience_milp.capacity_certificate import safe_lower
from scripts.run_resilience_week import ROOT, load_week, write_json


def verify_previous_model(previous):
    """Only result serialization may differ when reusing a previous lower bound."""
    names = ('model.py', 'annual.py', 'acceleration.py')
    checks = {}
    for name in names:
        relative = Path('polar_reliability_planning/resilience_milp') / name
        before = ast.parse((previous / 'source_snapshot' / relative).read_text())
        after = ast.parse((ROOT / relative).read_text())
        if name == 'annual.py':
            for tree in (before, after):
                for node in tree.body:
                    if isinstance(node, ast.ClassDef) and node.name == 'AnnualResilienceMILP':
                        node.body = [part for part in node.body if not (
                            isinstance(part, ast.FunctionDef) and part.name in ('result', 'save'))]
        same = ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
        if not same:
            raise ValueError(f'Physical/model semantics changed: {relative}; old bound cannot be reused automatically')
        checks[str(relative)] = True
    relative = Path('polar_reliability_planning/resilience_v2/model.py')
    if (previous / 'source_snapshot' / relative).read_bytes() != (ROOT / relative).read_bytes():
        raise ValueError('Input/scenario module changed')
    checks[str(relative)] = True
    return checks


def write_report(out, result):
    selected = result.get('selected') or {}
    lines = ['# 全年8760小时延长求解结果', '',
        '沿用上一轮模型、输入、成本及风险约束；未加入柴油用量或新能源占比限制。',
        '保存的可行解用于热启动，上一轮搜索树和单纯形迭代状态没有恢复。', '',
        '| 指标 | 结果 |', '|---|---:|']
    for title, key in [('状态','status'), ('总费用/元','objective_yuan'),
        ('有效全局下界/元','objective_bound_yuan'), ('最优性差距','mip_gap'),
        ('本轮预算/秒','time_limit_seconds'), ('本轮全流程耗时/秒','process_wall_seconds'),
        ('两轮全流程累计耗时/秒','cumulative_process_wall_seconds'),
        ('全年EENS/kWh','eens_kwh'), ('全年CVaR保守上界/kWh','cvar_upper_bound_kwh')]:
        lines.append(f'| {title} | {result.get(key)} |')
    lines += ['', '| 设备 | 容量 | 整数模块数 |','|---|---:|---:|']
    for key, value in selected.items():
        lines.append(f'| {key} | {value} | {result["modules"][key]} |')
    lines += ['', '只有通过完整年度约束审计的方案才作为最终可行解保存。初始解改进阶段的下界不作为全局下界。',
        '是否达到1%目标须看最终差距；延长时限不保证一定达到目标。', '',
        '[汇总](summary.json) · [正式日志](gurobi.log) · [阶段记录](events.jsonl) · [运行定义](run_definition.json)', '']
    (out / 'RESULT_REPORT.md').write_text('\n'.join(lines))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit-seconds', type=float, default=60000.)
    parser.add_argument('--capacity-polish-seconds', type=float, default=300.)
    parser.add_argument('--storage-polish-seconds', type=float, default=600.)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--method', type=int, default=3, choices=(1,2,3))
    parser.add_argument('--mip-gap', type=float, default=.01)
    args = parser.parse_args()
    if min(args.time_limit_seconds,args.capacity_polish_seconds,args.storage_polish_seconds) <= 0:
        parser.error('Time budgets must be positive')
    previous, out = args.previous.resolve(), args.output.resolve()
    if out.exists() and any(path.name not in ('console.log','launch.json','RUN_NOTE.md') for path in out.iterdir()):
        parser.error('Output directory must be new or empty')
    prior = json.loads((previous / 'summary.json').read_text())
    definition = json.loads((previous / 'run_definition.json').read_text())
    if prior['hours'] != 8760 or not prior.get('selected') or not prior['audit']['passed']:
        parser.error('Previous run must contain a fully audited annual incumbent')
    semantic_checks = verify_previous_model(previous)
    source = Path(definition['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != definition['source_sha256']:
        raise ValueError('Previous input file changed')
    cfg = replace(MILPConfig(**prior['resolved_config']),time_limit_seconds=args.time_limit_seconds,
                  threads=args.threads,mip_gap=args.mip_gap)
    cfg.check()
    recovery = FaultRecoveryConfig(**definition['recovery'])
    prior_lower = float(prior['objective_bound_yuan'])
    best, best_dir = prior, previous
    stages = []
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir()
    state = dict(status='running',stage='preparing',pid=os.getpid(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),previous_run=str(previous),
        time_limit_seconds=args.time_limit_seconds,search_tree_restored=False)
    last = {}
    def remaining():
        return max(0.,args.time_limit_seconds-(time.monotonic()-started))
    def event(action, **details):
        row = dict(action=action,elapsed_seconds=time.monotonic()-started,**details)
        with (out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(out / 'progress.json',row)
        state.update(stage=action,elapsed_seconds=row['elapsed_seconds'])
        write_json(out / 'run_status.json',state)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def callback(stage, planner):
        def cb(model, where):
            if remaining() <= 0:
                model.terminate()
            elapsed = time.monotonic()-started
            if where == GRB.Callback.MIPSOL:
                event('solver_incumbent',stage=stage,
                    objective_yuan=float(model.cbGet(GRB.Callback.MIPSOL_OBJ)),
                    modules={k:float(model.cbGetSolution(v)) for k,v in planner.n.items()},
                    audit_status='pending_full_audit')
            elif where in (GRB.Callback.MIP,GRB.Callback.SIMPLEX,GRB.Callback.BARRIER,GRB.Callback.PRESOLVE) and elapsed-last.get(stage,-60.)>=45.:
                last[stage]=elapsed
                details=dict(stage=stage,callback_location=where)
                if where == GRB.Callback.MIP:
                    upper=float(model.cbGet(GRB.Callback.MIP_OBJBST))
                    lower=float(model.cbGet(GRB.Callback.MIP_OBJBND))
                    details['objective_yuan']=upper if abs(upper)<GRB.INFINITY else None
                    if stage=='joint':
                        lower=max(prior_lower,lower-max(1e-4,abs(lower)*1e-9)) if abs(lower)<GRB.INFINITY else prior_lower
                        details.update(objective_bound_yuan=lower,
                            mip_gap=max(0.,upper-lower)/abs(upper) if 0<abs(upper)<GRB.INFINITY else None)
                    else:
                        details['restricted_policy_bound_yuan']=lower if abs(lower)<GRB.INFINITY else None
                    details['nodes']=float(model.cbGet(GRB.Callback.MIP_NODCNT))
                event('solver_progress',**details)
        return cb
    def configure(model, stage, seconds):
        model.Params.OutputFlag=1
        model.Params.LogToConsole=0
        model.Params.LogFile=str(out / ('gurobi.log' if stage=='joint' else stage+'.log'))
        model.Params.TimeLimit=max(.01,min(seconds,remaining()))
        model.Params.Method=args.method if stage=='joint' else 1
        model.Params.MIPFocus=1
        model.Params.Heuristics=.2
        model.Params.SolFiles=str(out / 'checkpoints' / stage)
    def retain(planner, stage):
        nonlocal best,best_dir
        candidate=planner.result()
        stages.append(dict(stage=stage,solver_seconds=planner.model.Runtime,status=candidate['status'],
                           objective_yuan=candidate.get('objective_yuan')))
        if candidate.get('selected') and candidate['objective_yuan'] < best['objective_yuan']-1e-4:
            candidate.update(status='restricted_commitment_feasible_not_global',
                global_bound_valid=False,restricted_policy=True)
            planner.save(out / stage,candidate,write_model=False)
            best,best_dir=candidate,out / stage
            event('improved_start_audited',stage=stage,objective_yuan=best['objective_yuan'],
                improvement_from_previous_yuan=prior['objective_yuan']-best['objective_yuan'],
                selected=best['selected'],audit=best['audit'])
        else:
            write_json(out / (stage+'_result.json'),candidate)
            event('restricted_stage_finished',stage=stage,retained_objective_yuan=best['objective_yuan'])
    try:
        year=load_week(source,0,8760)
        base,original=annual_scenarios(year,starts=tuple(definition['window_start_hours']),recovery=recovery)
        windows,aliases=merge_duplicate_paths(original)
        write_json(out / 'scenario_aliases.json',aliases)
        shutil.copyfile(previous / 'input_8760.npz',out / 'input_8760.npz')
        sources=[Path(__file__).resolve(),ROOT/'scripts/run_resilience_week.py',
            *[ROOT/'polar_reliability_planning/resilience_milp'/name for name in
              ('model.py','annual.py','acceleration.py','continuation.py','capacity_certificate.py')],
            ROOT/'polar_reliability_planning/resilience_v2/model.py']
        for path in sources:
            target=out/'source_snapshot'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
        write_json(out/'run_definition.json',dict(previous_run=str(previous),hours=8760,
            source=str(source),source_sha256=definition['source_sha256'],
            window_start_hours=definition['window_start_hours'],window_hours=72,
            config=asdict(cfg),recovery=asdict(recovery),root_method=args.method,
            capacity_polish_seconds=args.capacity_polish_seconds,storage_polish_seconds=args.storage_polish_seconds,
            original_model_semantic_checks=semantic_checks,previous_valid_lower_bound_yuan=prior_lower,
            final_all_capacities_free=True,post_solution_rounding=False,search_tree_restored=False,
            diesel_fuel_limit=None,renewable_share_limit=None,
            source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}))
        event('previous_incumbent_loaded',objective_yuan=prior['objective_yuan'],lower_bound_yuan=prior_lower)
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0)
            env.start()
            event('restricted_building')
            polish=AnnualResilienceMILP(base,windows,cfg,math.ceil(prior['objective_yuan']+1e-5),env,
                fixed_modules={'diesel_units':prior['modules']['diesel_units']},compact_fixed=True,compact=True)
            try:
                values=read_solution(previous/'solution.sol')
                transfer=apply_start(polish,values,prior['modules']['diesel_units'])
                if transfer['missing_count']:
                    raise ValueError(f'Incomplete incumbent transfer: {transfer}')
                policy=fix_incumbent_commitment(polish,values)
                bounds=temporarily_fix_battery_modes(polish,values)
                event('capacity_polish_started',transfer=transfer,**policy)
                configure(polish.model,'capacity_polish',args.capacity_polish_seconds)
                polish.model.Params.Aggregate=0
                polish.model.Params.NumericFocus=3
                polish.model.optimize(callback('capacity_polish',polish))
                retain(polish,'capacity_polish')
                restore_bounds(polish.model,bounds)
                # The second search allows battery modes at every annual hour
                # and all flexible-load service variables. No capacity grid.
                transfer=apply_start(polish,read_solution(best_dir/'solution.sol'),best['modules']['diesel_units'])
                event('storage_polish_started',transfer=transfer)
                configure(polish.model,'storage_polish',args.storage_polish_seconds)
                polish.model.optimize(callback('storage_polish',polish))
                retain(polish,'storage_polish')
                del values
            finally:
                polish.close()
            budget=math.ceil(best['objective_yuan']+1e-5)
            write_json(out/'economic_bound.json',dict(budget_yuan=budget,source=str(best_dir),
                source_objective_yuan=best['objective_yuan'],global_coverage_certified=True))
            event('joint_building',economic_budget_yuan=budget,all_capacities_free=True)
            planner=AnnualResilienceMILP(base,windows,cfg,budget,env,compact=True)
            try:
                m=planner.model
                transfer=apply_start(planner,read_solution(best_dir/'solution.sol'),best['modules']['diesel_units'])
                if transfer['missing_count']:
                    raise ValueError(f'Incomplete full-model start: {transfer}')
                write_json(out/'warm_start_transfer.json',dict(source=str(best_dir),**transfer))
                write_json(out/'model_statistics.json',dict(variables=m.NumVars,binary_variables=m.NumBinVars,
                    linear_constraints=m.NumConstrs,indicators=m.NumGenConstrs,diesel_slots=planner.dmax,
                    upper_module_bounds=planner.upper,all_capacity_types={k:v.VType for k,v in planner.n.items()}))
                event('joint_started',start_objective_yuan=best['objective_yuan'],transfer=transfer,
                    remaining_seconds=remaining(),root_method=args.method)
                configure(m,'joint',remaining())
                m.write(str(out/'solver_parameters.prm'))
                m.optimize(callback('joint',planner))
                candidate=planner.result()
                lower=max(prior_lower,min(budget,safe_lower(m)))
                stages.append(dict(stage='joint',solver_seconds=m.Runtime,status=candidate['status'],
                    objective_yuan=candidate.get('objective_yuan')))
                if candidate.get('selected') and candidate['objective_yuan']<=best['objective_yuan']+1e-5:
                    result=candidate
                    result['solution_source']='joint_free_capacity_model'
                    planner.save(out,result,write_model=False)
                else:
                    result={**best,'status':'feasible_audited_start_not_improved',
                        'fixed_modules':None,'solution_source':str(best_dir),
                        'restricted_policy':False,'global_bound_valid':True,'solver_status':int(m.Status)}
                    for name in ('solution.sol','dispatch.npz','hourly_dispatch.csv','scenario_manifest.json','resolved_config.json'):
                        shutil.copyfile(best_dir/name,out/name)
                if lower>result['objective_yuan']+1e-4:
                    raise RuntimeError('Retained lower bound exceeds the audited upper bound')
                result.update(objective_bound_yuan=lower,
                    mip_gap=max(0.,result['objective_yuan']-lower)/abs(result['objective_yuan']),
                    stages=stages,previous_run=str(previous),search_tree_restored=False,
                    prior_objective_yuan=prior['objective_yuan'],
                    improvement_from_previous_yuan=prior['objective_yuan']-result['objective_yuan'],
                    all_capacities_optimized=True,global_bound_valid=True,economic_domain_certified=True,
                    time_limit_seconds=args.time_limit_seconds,resolved_config=asdict(cfg),recovery=asdict(recovery),
                    total_elapsed_seconds=time.monotonic()-started,
                    runtime_seconds=sum(stage['solver_seconds'] for stage in stages))
            finally:
                planner.close()
        result['process_wall_seconds']=time.monotonic()-started
        result['cumulative_process_wall_seconds']=prior['process_wall_seconds']+result['process_wall_seconds']
        write_json(out/'summary.json',result)
        write_report(out,result)
        state.update(status='finished',stage='finished',finished_at_utc=datetime.now(timezone.utc).isoformat(),
            objective_yuan=result['objective_yuan'],mip_gap=result['mip_gap'],selected=result['selected'],
            process_wall_seconds=result['process_wall_seconds'])
        write_json(out/'run_status.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException:
        state.update(status='error',error=traceback.format_exc(),process_wall_seconds=time.monotonic()-started)
        write_json(out/'run_status.json',state)
        raise


if __name__=='__main__':
    main()
