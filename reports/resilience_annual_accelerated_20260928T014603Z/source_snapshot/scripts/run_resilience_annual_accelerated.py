#!/usr/bin/env python3
"""8760-hour discrete planning: exact compression, audited seed, free joint MILP."""
import argparse
from dataclasses import asdict,fields,replace
from datetime import datetime,timezone
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
import numpy as np

from polar_reliability_planning.resilience_milp.model import MILPConfig,FaultRecoveryConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP,annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import (
    merge_duplicate_paths,seed_modules,fix_seed_policy,read_solution,apply_start,
)
from polar_reliability_planning.resilience_milp.capacity_certificate import safe_lower
from scripts.run_resilience_week import ROOT,DEFAULT_INPUT,load_week,write_json


def write_report(out,r):
    lines=['# 全年8760小时加速规划运行结果','',
        '保留8760个小时、四个72小时韧性窗口和七类整数设备容量。设备恢复12小时，主母线失电1小时，UPS桥接12小时。',
        '原64条分支记录经严格等价合并后保留56条；原始标签及权重对应关系见scenario_aliases.json。',
        '总时间预算15000秒；财务口径为一次性投资加全年期望运行及失供费用。', '',
        '| 指标 | 结果 |','|---|---:|']
    for label,key in [('状态','status'),('目标费用/元','objective_yuan'),('有效全局下界/元','objective_bound_yuan'),
                      ('投资/元','investment_yuan'),('全年期望运行及损失/元','expected_operation_yuan'),
                      ('全年EENS/kWh','eens_kwh'),('全年CVaR上界/kWh','cvar_upper_bound_kwh'),
                      ('首次通过完整审计的可行解时间/秒','first_audited_feasible_seconds'),
                      ('计算阶段总时间/秒','total_elapsed_seconds')]:
        v=r.get(key);s='未获得' if v is None else (f'{v:,.6f}' if isinstance(v,float) else str(v))
        lines.append(f'| {label} | {s} |')
    gap=r.get('mip_gap')
    lines.append(f'| 相对最优性差距 | {100*gap:.6f}% |' if gap is not None else '| 相对最优性差距 | 未获得 |')
    if r.get('selected'):
        lines+=['','| 设备 | 容量 | 模块数 |','|---|---:|---:|']
        for k,v in r['selected'].items(): lines.append(f'| {k} | {v} | {r["modules"][k]} |')
        lines+=['',f'完整约束审计：{r["audit"]["passed"]}；最大残差：{r["audit"]["max_violation"]:.6g}。',
                '可行解不自动意味着达到最优性精度，请结合状态、下界和差距读取。']
    else: lines+=['','本次未取得完整可行解，不能将下界或初始候选视为规划结果。']
    lines+=['','初始方案的容量和运行策略限制只用于构造热启动；正式规划中七类容量全部放开。',
            '记录：[汇总](summary.json)、[阶段记录](events.jsonl)、[初始方案](seed/summary.json)、[正式求解日志](gurobi.log)。','']
    (out/'RESULT_REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--time-limit-seconds',type=float,default=15000.)
    parser.add_argument('--seed-seconds',type=float,default=1200.)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--method',type=int,default=1,choices=[1,2,3])
    parser.add_argument('--mip-gap',type=float,default=.01)
    args=parser.parse_args();out=args.output.resolve()
    if args.time_limit_seconds<=0 or args.seed_seconds<=0: parser.error('Time budgets must be positive')
    if (out/'run_status.json').exists() or (out/'summary.json').exists(): parser.error('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    cfg=replace(MILPConfig(),time_limit_seconds=args.time_limit_seconds,threads=args.threads,
                mip_gap=args.mip_gap,ups_bridge_hours=12.)
    cfg.check();recovery=FaultRecoveryConfig()
    state=dict(status='running',stage='preparing',pid=os.getpid(),started_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_status.json',state)
    first_feasible=None;first_audited=None;last={};rounds=[];seed_result=None
    def remaining(): return max(0.,cfg.time_limit_seconds-(time.monotonic()-started))
    def event(action,**kwargs):
        row=dict(action=action,elapsed_seconds=time.monotonic()-started,**kwargs)
        with (out/'events.jsonl').open('a') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(out/'progress.json',row)
        state.update(stage=action,elapsed_seconds=row['elapsed_seconds'])
        write_json(out/'run_status.json',state)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def callback(stage):
        def cb(m,where):
            nonlocal first_feasible
            if remaining()<=0: m.terminate()
            elapsed=time.monotonic()-started
            if where==GRB.Callback.MIPSOL:
                if first_feasible is None: first_feasible=elapsed
                event('solver_incumbent',stage=stage,objective_yuan=float(m.cbGet(GRB.Callback.MIPSOL_OBJ)))
            elif where in (GRB.Callback.MIP,GRB.Callback.SIMPLEX,GRB.Callback.BARRIER,GRB.Callback.PRESOLVE) and elapsed-last.get(stage,-60)>=45:
                last[stage]=elapsed
                data=dict(stage=stage,callback_location=where)
                if where==GRB.Callback.MIP:
                    hi=m.cbGet(GRB.Callback.MIP_OBJBST);lo=m.cbGet(GRB.Callback.MIP_OBJBND)
                    data.update(objective_yuan=float(hi) if abs(hi)<GRB.INFINITY else None,
                                bound_yuan=float(lo) if abs(lo)<GRB.INFINITY else None,
                                nodes=float(m.cbGet(GRB.Callback.MIP_NODCNT)))
                event('solver_progress',**data)
        return cb
    def configure(m,log,seconds):
        m.Params.OutputFlag=1;m.Params.LogToConsole=0;m.Params.LogFile=str(log)
        m.Params.Method=args.method;m.Params.MIPFocus=1;m.Params.Heuristics=.2
        m.Params.TimeLimit=max(.01,min(seconds,remaining()))
    try:
        year=load_week(args.source,0,8760)
        base,original=annual_scenarios(year,recovery=recovery)
        windows,aliases=merge_duplicate_paths(original)
        write_json(out/'scenario_aliases.json',aliases)
        base.year.frame().to_csv(out/'annual_input.csv',index=False)
        np.savez_compressed(out/'input_8760.npz',**{f.name:np.asarray(getattr(base.year,f.name)) for f in fields(base.year)})
        reference_file=ROOT/'reports/resilience_week_168h_repair12_20260924T072856Z/summary.json'
        reference=json.loads(reference_file.read_text())['modules']
        point=seed_modules(base,cfg,reference)
        sources=[Path(__file__).resolve(),ROOT/'scripts/run_resilience_week.py',
                 ROOT/'polar_reliability_planning/resilience_milp/model.py',ROOT/'polar_reliability_planning/resilience_milp/annual.py',
                 ROOT/'polar_reliability_planning/resilience_milp/acceleration.py',ROOT/'polar_reliability_planning/resilience_milp/capacity_certificate.py',
                 ROOT/'polar_reliability_planning/resilience_v2/model.py']
        for source in sources:
            target=out/'source_snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        definition=dict(hours=8760,window_start_hours=[w['start'] for w in windows],window_hours=72,
            original_branch_count=64,merged_branch_count=sum(len(w['paths']) for w in windows),
            source=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            config=asdict(cfg),recovery=asdict(recovery),seed_modules=point,seed_policy='causal available-units-on plus startup battery; original full audit required',
            seed_reference=str(reference_file),final_all_capacities_free=True,post_solution_rounding=False,
            compact=True,root_method=args.method,threads=args.threads,
            time_budget_scope='seed build/solve/audit/export plus free-model build/solve; final export recorded separately',
            source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        write_json(out/'run_definition.json',definition)
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            event('seed_building',modules=point)
            seed_budget=max(cfg.initial_economic_budget_yuan,sum(cfg.module_costs[k]*v for k,v in point.items())+1.)
            seed=AnnualResilienceMILP(base,windows,cfg,seed_budget,env,point,compact_fixed=True,compact=True)
            try:
                policy=fix_seed_policy(seed)
                event('seed_built',variables=seed.model.NumVars,binaries=seed.model.NumBinVars,
                      constraints=seed.model.NumConstrs,compression=seed.compression,**policy)
                configure(seed.model,out/'seed.log',args.seed_seconds)
                seed.model.optimize(callback('seed'))
                candidate=seed.result()
                rounds.append(dict(stage='seed',build_seconds=seed.build_seconds,runtime_seconds=seed.model.Runtime,status=candidate['status']))
                if candidate.get('selected'):
                    first_audited=time.monotonic()-started
                    seed_result=candidate
                    seed.save(out/'seed',candidate,write_model=False)
                    event('seed_audited',objective_yuan=candidate['objective_yuan'],eens_kwh=candidate['eens_kwh'],audit=candidate['audit'])
                else:
                    write_json(out/'seed_failure.json',candidate)
                    event('seed_unresolved',status=candidate['status'],original_model_infeasible=False)
            finally: seed.close()
            budget=math.ceil(seed_result['objective_yuan']+1e-5) if seed_result else cfg.initial_economic_budget_yuan
            write_json(out/'economic_bound.json',dict(budget_yuan=budget,source='audited full-year feasible total cost' if seed_result else 'expandable initial computational region',
                original_global_coverage_certified=bool(seed_result),all_costs_nonnegative=True))
            while True:
                event('joint_building',economic_bound_yuan=budget,all_capacities_free=True)
                planner=AnnualResilienceMILP(base,windows,cfg,budget,env,compact=True)
                try:
                    m=planner.model
                    model_info=dict(variables=m.NumVars,binary_variables=m.NumBinVars,linear_constraints=m.NumConstrs,
                                    indicators=m.NumGenConstrs,diesel_slots=planner.dmax,compression=planner.compression,
                                    all_capacity_types={k:v.VType for k,v in planner.n.items()},upper_module_bounds=planner.upper)
                    write_json(out/'model_statistics.json',model_info)
                    event('joint_built',**model_info)
                    if seed_result:
                        start=apply_start(planner,read_solution(out/'seed/solution.sol'),seed_result['modules']['diesel_units'])
                        write_json(out/'warm_start_transfer.json',start);event('warm_start_loaded',**start)
                    if remaining()>0:
                        configure(m,out/'gurobi.log',remaining())
                        m.optimize(callback('joint'))
                        result=planner.result()
                        result['objective_bound_yuan']=min(budget,safe_lower(m))
                    else:
                        result=dict(status='time_limit_during_build',selected=None,economic_domain_certified=bool(seed_result),objective_bound_yuan=0.)
                    compute_elapsed=time.monotonic()-started
                    rounds.append(dict(stage='joint',build_seconds=planner.build_seconds,runtime_seconds=m.Runtime,
                                       economic_bound_yuan=budget,status=result['status']))
                    if result.get('selected'):
                        if first_audited is None: first_audited=time.monotonic()-started
                        result['solution_source']='joint_free_capacity_model'
                        result['mip_gap']=max(0.,result['objective_yuan']-result['objective_bound_yuan'])/abs(result['objective_yuan'])
                        planner.save(out,result,write_model=False)
                    elif seed_result:
                        bound=result['objective_bound_yuan'];joint_status=result['status']
                        result={**seed_result,'status':'feasible_audited_seed_not_improved','joint_status':joint_status,
                                'solver_status':int(m.Status),'seed_solver_status':seed_result['solver_status'],
                                'joint_model_solution_count':int(m.SolCount),
                                'solution_source':'audited_seed','fixed_modules':None,'economic_domain_certified':True,
                                'objective_bound_yuan':bound,'mip_gap':max(0.,seed_result['objective_yuan']-bound)/abs(seed_result['objective_yuan'])}
                        for name in ('solution.sol','hourly_dispatch.csv','dispatch.npz','scenario_manifest.json'):
                            shutil.copyfile(out/'seed'/name,out/name)
                    elif m.Status==GRB.INFEASIBLE and remaining()>0:
                        budget*=2;event('economic_region_expanded',budget_yuan=budget);continue
                    result.update(total_elapsed_seconds=compute_elapsed,
                        runtime_seconds=sum(r['runtime_seconds'] for r in rounds),build_seconds=sum(r['build_seconds'] for r in rounds),
                        stages=rounds,first_solver_feasible_seconds=first_feasible,first_audited_feasible_seconds=first_audited,
                        original_branch_count=64,merged_branch_count=56,all_capacities_optimized=True,recovery=asdict(recovery))
                    break
                finally: planner.close()
        result['process_wall_seconds']=time.monotonic()-started
        write_json(out/'summary.json',result);write_report(out,result)
        state.update(status='finished',stage='finished',result_status=result['status'],has_feasible_solution=bool(result.get('selected')),
            selected=result.get('selected'),objective_yuan=result.get('objective_yuan'),mip_gap=result.get('mip_gap'),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),process_wall_seconds=time.monotonic()-started)
        write_json(out/'run_status.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException:
        state.update(status='error',error=traceback.format_exc(),process_wall_seconds=time.monotonic()-started)
        write_json(out/'run_status.json',state);raise


if __name__=='__main__': main()
