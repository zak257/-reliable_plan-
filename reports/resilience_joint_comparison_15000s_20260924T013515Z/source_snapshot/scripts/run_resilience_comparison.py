#!/usr/bin/env python3
"""Parallel, equal-budget comparison on one frozen annual planning instance."""
from dataclasses import asdict,fields,replace
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.resilience_v2.model import SyntheticYear,generate_synthetic_year
from polar_reliability_planning.resilience_milp.model import MILPConfig
from polar_reliability_planning.resilience_milp.annual import annual_scenarios,AnnualResilienceMILP
from polar_reliability_planning.resilience_milp.capacity_certificate import run_capacity_search,safe_lower


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def fingerprint(base,windows,cfg):
    h=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode())
    for w in windows:
        h.update(json.dumps([w['start'],w['stop']]).encode())
        for p in w['paths']:
            h.update(json.dumps([p.name,p.weight,sorted(p.run_shocks),sorted(p.start_shocks),p.run_repair_hours,p.start_repair_hours]).encode())
            for f in fields(p.year):
                v=getattr(p.year,f.name)
                h.update(json.dumps(v).encode() if isinstance(v,tuple) else v.tobytes())
            for name in ('grid_fault','renewable_bus_fault','pcs_available','ups_available','main_bus_available'):
                h.update(getattr(p,name).tobytes())
    return h.hexdigest()


def run_worker(root,method):
    started=time.monotonic();out=root/method;out.mkdir(exist_ok=True)
    manifest=json.loads((root/'comparison_manifest.json').read_text())
    affinity=manifest['cpu_affinity'][method]
    os.sched_setaffinity(0,affinity)
    state={'status':'running','method':method,'pid':os.getpid(),'cpu_affinity':affinity,
           'started_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(out/'run_status.json',state)
    log=(out/'events.jsonl').open('a',buffering=1)
    first=[None];last={};total_runtime=0.
    def event(action,**kw):
        row={'action':action,'elapsed_seconds':time.monotonic()-started,**kw}
        log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(out/'progress.json',row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def configure(m,path):
        m.Params.OutputFlag=1;m.Params.LogToConsole=0;m.Params.LogFile=str(path)
        m.Params.Method=3  # Both arms avoid the earlier deterministic concurrent spin.
        m.Params.Threads=manifest['config']['threads']
        m.Params.Seed=manifest['config']['seed']
    def progress(stage):
        def callback(m,where):
            if where not in (GRB.Callback.MIP,GRB.Callback.MIPSOL): return
            now=time.monotonic()-started
            if where==GRB.Callback.MIPSOL:
                hi=m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if stage!='master' and first[0] is None: first[0]=now
                event('solver_incumbent',stage=stage,objective_yuan=float(hi))
                return
            if now-last.get(stage,-60)<60: return
            last[stage]=now
            hi=m.cbGet(GRB.Callback.MIP_OBJBST);lo=m.cbGet(GRB.Callback.MIP_OBJBND)
            event('solver_progress',stage=stage,
                  solver_upper_yuan=float(hi) if abs(hi)<GRB.INFINITY else None,
                  solver_lower_yuan=float(lo) if abs(lo)<GRB.INFINITY else None,
                  nodes=float(m.cbGet(GRB.Callback.MIP_NODCNT)))
        return callback
    try:
        cfg=MILPConfig(**manifest['config'])
        with np.load(root/'shared_year.npz',allow_pickle=False) as values:
            data={f.name:values[f.name] for f in fields(SyntheticYear)}
        data['timestamps']=tuple(map(str,data['timestamps']))
        year=SyntheticYear(**data)
        base,windows=annual_scenarios(year,tuple(manifest['window_start_hours']))
        actual=fingerprint(base,windows,cfg)
        if actual!=manifest['instance_sha256']: raise RuntimeError('Shared annual instance fingerprint mismatch')
        write_json(out/'instance_verification.json',{'instance_sha256':actual,'passed':True,'hours':year.hours,
            'window_count':len(windows),'conditional_branch_count':sum(len(w['paths']) for w in windows)})
        event('instance_verified',instance_sha256=actual,hours=year.hours)
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            if method=='proposed':
                result=run_capacity_search(base,windows,cfg,out,env,started,event,configure,progress)
            else:
                budget=cfg.initial_economic_budget_yuan;rounds=[]
                while True:
                    planner=AnnualResilienceMILP(base,windows,cfg,budget,env)
                    try:
                        configure(planner.model,out/'gurobi.log')
                        remaining=max(0.,cfg.time_limit_seconds-(time.monotonic()-started))
                        if remaining>0:
                            planner.model.Params.TimeLimit=remaining
                            planner.model.optimize(progress('joint'))
                            total_runtime+=planner.model.Runtime
                            result=planner.result()
                            result['objective_bound_yuan']=min(budget,safe_lower(planner.model))
                            if result.get('objective_yuan') is not None:
                                result['mip_gap']=(result['objective_yuan']-result['objective_bound_yuan'])/abs(result['objective_yuan'])
                        else:
                            result={'status':'time_limit_during_build','selected':None,'economic_domain_certified':False}
                        rounds.append({'economic_budget_yuan':budget,'build_seconds':planner.build_seconds,
                                       'solver_runtime_seconds':float(planner.model.Runtime),'status':result['status']})
                        elapsed=time.monotonic()-started
                        if result.get('economic_domain_certified') or elapsed>=cfg.time_limit_seconds or planner.model.Status in (GRB.TIME_LIMIT,GRB.INTERRUPTED):
                            result.update(method='direct_gurobi_joint_milp',runtime_seconds=total_runtime,
                                          economic_bound_rounds=rounds,total_elapsed_seconds=elapsed)
                            planner.save(out,result)
                            break
                        budget*=2;event('economic_region_expanded',budget_yuan=budget)
                    finally: planner.close()
        result.update(instance_sha256=actual,time_limit_seconds=cfg.time_limit_seconds,
                      threads=cfg.threads,gurobi_method=3,hours=year.hours,
                      first_solver_feasible_seconds=first[0],synthetic=True)
        write_json(out/'summary.json',result)
        state.update(status='finished',result_status=result['status'],selected=result.get('selected'),
                     has_audited_incumbent=bool(result.get('selected')),
                     objective_yuan=result.get('objective_yuan'),mip_gap=result.get('mip_gap'))
    except BaseException:
        state.update(status='error',error=traceback.format_exc())
        print(state['error'],flush=True)
    finally:
        state['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        state['process_wall_seconds']=time.monotonic()-started
        write_json(out/'run_status.json',state);log.close()
    return 0 if state['status']=='finished' else 1


def aggregate(root):
    manifest=json.loads((root/'comparison_manifest.json').read_text())
    results={};states={}
    for name in ('gurobi','proposed'):
        p=root/name
        states[name]=json.loads((p/'run_status.json').read_text())
        results[name]=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else {}
    rows=[]
    for name,r in results.items():
        rows.append({'method':name,'status':r.get('status',states[name]['status']),
                     'selected':r.get('selected'),'objective_yuan':r.get('objective_yuan'),
                     'objective_bound_yuan':r.get('objective_bound_yuan'),'mip_gap':r.get('mip_gap'),
                     'first_solver_feasible_seconds':r.get('first_solver_feasible_seconds'),
                     'total_elapsed_seconds':r.get('total_elapsed_seconds'),
                     'process_wall_seconds':states[name]['process_wall_seconds'],
                     'eens_kwh':r.get('eens_kwh'),'cvar_upper_bound_kwh':r.get('cvar_upper_bound_kwh'),
                     'audit_passed':r.get('audit',{}).get('passed'),
                     'economic_domain_certified':r.get('economic_domain_certified')})
    write_json(root/'comparison.json',{'instance_sha256':manifest['instance_sha256'],
        'time_limit_seconds_per_method':manifest['config']['time_limit_seconds'],'results':rows})
    def val(x): return '尚无结果' if x is None else (f'{x:,.4f}' if isinstance(x,float) else str(x))
    lines=['# 同一年度联合MILP的并行对照结果','',
           f'两组使用相同{manifest["hours"]}小时数据、{len(manifest["window_start_hours"])}组72小时窗口、相同条件分支、整数模块、风险约束及目标函数。',
           f'每组独立预算{manifest["config"]["time_limit_seconds"]:g}秒、{manifest["config"]["threads"]}线程；预算包括建模、搜索和子问题求解。最终导出另计进程耗时。','',
           '自提方法为原单调容量搜索与上下界框架的有限场景联合模型适配版。Gurobi同样用于其主问题和固定容量校验；不使用原模型的统计总体证书。','',
           '| 指标 | 直接Gurobi | 改进自提方法 |','|---|---:|---:|']
    for label,key in [('状态','status'),('目标上界/元','objective_yuan'),('全局下界/元','objective_bound_yuan'),
                      ('相对差距','mip_gap'),('首次整数可行解/秒','first_solver_feasible_seconds'),
                      ('计算计时/秒','total_elapsed_seconds'),('含导出的进程时间/秒','process_wall_seconds'),
                      ('EENS/kWh','eens_kwh'),('CVaR年度上界/kWh','cvar_upper_bound_kwh'),('物理与信息审计','audit_passed')]:
        lines.append(f'| {label} | {val(rows[0].get(key))} | {val(rows[1].get(key))} |')
    lines+=['','## 容量','', '| 容量 | 直接Gurobi | 改进自提方法 |','|---|---:|---:|']
    for key in ('wind_kw','pv_kw','diesel_units','battery_kwh','pcs_kw','ups_kwh','ups_kw'):
        lines.append(f'| {key} | {val((rows[0].get("selected") or {}).get(key))} | {val((rows[1].get("selected") or {}).get(key))} |')
    lines+=['','未找到解或未获得差距证书时，不能写成不可行或全局最优。单次并行实验只能描述本次实例表现，不能单独证明普遍算法优势。',
            '费用口径为设备投资加全年期望运行与失供损失费用，不是资本年化成本。数据为授权生成的模拟数据。','']
    (root/'COMPARISON_REPORT.md').write_text('\n'.join(lines))


def supervise(root):
    processes={}
    for method in ('gurobi','proposed'):
        out=root/method;out.mkdir(exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'console.log').open('ab',buffering=0) as log:
            p=subprocess.Popen([sys.executable,'-u','-m','scripts.run_resilience_comparison','--worker',method,'--output',str(root)],
                cwd=Path(__file__).resolve().parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes[method]=p
    write_json(root/'processes.json',{'supervisor_pid':os.getpid(),'workers':{k:p.pid for k,p in processes.items()}})
    codes={k:p.wait() for k,p in processes.items()}
    write_json(root/'completion.json',{'exit_codes':codes,'finished_at_utc':datetime.now(timezone.utc).isoformat()})
    aggregate(root)


def launch(root,limit,threads):
    cfg=replace(MILPConfig(),time_limit_seconds=limit,threads=threads)
    cfg.check()
    if limit<=0: raise ValueError('comparison requires a positive equal wall-time budget')
    root.mkdir(parents=True,exist_ok=False)
    year=generate_synthetic_year(8760,cfg.seed)
    np.savez_compressed(root/'shared_year.npz',**{f.name:np.asarray(getattr(year,f.name)) for f in fields(year)})
    year.frame().to_csv(root/'shared_year.csv',index=False)
    starts=(816,2496,5616,7536);base,windows=annual_scenarios(year,starts)
    cores=[];seen=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity=((topology/'physical_package_id').read_text().strip(),(topology/'core_id').read_text().strip())
        if identity not in seen: cores.append(cpu);seen.add(identity)
    if len(cores)<2*threads: raise ValueError('not enough distinct CPU cores for equal-resource parallel comparison')
    project=Path(__file__).resolve().parents[1]
    files=[Path(__file__).resolve(),project/'polar_reliability_planning/resilience_milp/model.py',
           project/'polar_reliability_planning/resilience_milp/annual.py',
           project/'polar_reliability_planning/resilience_milp/capacity_certificate.py',
           project/'polar_reliability_planning/resilience_v2/model.py']
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':asdict(cfg),'hours':year.hours,
              'window_start_hours':starts,'instance_sha256':fingerprint(base,windows,cfg),
              'cpu_affinity':{'gurobi':cores[:threads],'proposed':cores[threads:2*threads]},
              'gurobi_method':3,'synthetic_data':True,'shared_input_sha256':hashlib.sha256((root/'shared_year.npz').read_bytes()).hexdigest(),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'proposed_method':'finite-tree joint-model adaptation of monotone capacity search with upper/lower certificates',
              'monotonicity':'same diesel count only; no unproved diesel-count propagation',
              'no_results_or_bounds_reused_from_prior_runs':True}
    write_json(root/'comparison_manifest.json',manifest)
    for p in files:
        target=root/'source_snapshot'/p.relative_to(project);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    with (root/'supervisor.log').open('ab',buffering=0) as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','scripts.run_resilience_comparison','--supervise','--output',str(root)],
            cwd=project,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(json.dumps({'output':str(root),'supervisor_pid':proc.pid,'instance_sha256':manifest['instance_sha256'],
                      'per_method_budget_seconds':limit,'cpu_affinity':manifest['cpu_affinity']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--time-limit-seconds',type=float,default=15000.)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--worker',choices=('gurobi','proposed'))
    p.add_argument('--supervise',action='store_true')
    args=p.parse_args();root=args.output.resolve()
    if args.worker: return run_worker(root,args.worker)
    if args.supervise: supervise(root);return 0
    launch(root,args.time_limit_seconds,args.threads);return 0


if __name__=='__main__': raise SystemExit(main())
