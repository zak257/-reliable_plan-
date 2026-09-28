#!/usr/bin/env python3
"""Validate compression on the frozen 168-hour witness; benchmark root methods."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import re
import time

import gurobipy as gp

from scripts.run_resilience_week import ROOT,DEFAULT_INPUT,load_week,write_json
from polar_reliability_planning.resilience_milp.model import MILPConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP,annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import (
    merge_duplicate_paths,read_solution,seed_modules,fix_seed_policy,
)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    old=ROOT/'reports/resilience_week_168h_repair12_20260924T072856Z'
    summary=json.loads((old/'summary.json').read_text());values=read_solution(old/'solution.sol')
    cfg=replace(MILPConfig(**summary['resolved_config']),threads=8)
    base,original=annual_scenarios(load_week(DEFAULT_INPUT,792),starts=(24,))
    windows,aliases=merge_duplicate_paths(original)
    old_index={p.name:i for i,p in enumerate(original[0]['paths'])}
    new_to_old={i:old_index[p.name] for i,p in enumerate(windows[0]['paths'])}
    path_var=re.compile(r's(\d+)_(.*)')
    na_var=re.compile(r'(na_(?:pre_delta|start_delta|pre_history|post_history))_(\d+)_(\d+)_(.*)')
    risk_var=re.compile(r'window_0_branch_(\d+)_(.*)')
    records=[]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        for method in (1,2):
            p=AnnualResilienceMILP(base,windows,cfg,env=env,compact=True)
            try:
                missing=[]
                for var in p.model.getVars():
                    name=var.VarName
                    match=path_var.fullmatch(name)
                    if match: name=f's{new_to_old[int(match[1])]}_{match[2]}'
                    match=na_var.fullmatch(name)
                    if match: name=f'{match[1]}_{new_to_old[int(match[2])]}_{new_to_old[int(match[3])]}_{match[4]}'
                    match=risk_var.fullmatch(name)
                    if match: name=f'window_0_branch_{new_to_old[int(match[1])]}_{match[2]}'
                    if name in values: var.Start=values[name]
                    elif var.LB==var.UB: var.Start=var.LB
                    else: missing.append(name)
                if missing: raise AssertionError(f'Missing frozen witness variables: {missing[:10]}')
                p.model.Params.OutputFlag=1;p.model.Params.LogToConsole=0
                p.model.Params.LogFile=str(out/f'week_root_method{method}.log')
                p.model.Params.TimeLimit=45;p.model.Params.NodeLimit=0
                p.model.Params.Method=method;p.model.Params.MIPFocus=1;p.model.Params.Heuristics=.2
                began=time.monotonic();p.model.optimize()
                if not p.model.SolCount: raise AssertionError('Frozen weekly feasible solution was not retained')
                result=p.result()
                if result['objective_yuan']>summary['objective_yuan']+1e-4:
                    raise AssertionError('Compression lost the supplied feasible witness')
                record=dict(method=method,seconds=time.monotonic()-began,status=p.model.Status,
                    variables=p.model.NumVars,binaries=p.model.NumBinVars,constraints=p.model.NumConstrs,
                    indicators=p.model.NumGenConstrs,objective_yuan=result['objective_yuan'],
                    bound_yuan=p.model.ObjBound,audit=result['audit'],compression=p.compression)
                records.append(record);write_json(out/'weekly_root_checks.json',records)
                print(json.dumps(record),flush=True)
            finally:p.close()
        # Test the initialization policy on the real winter slice, including
        # the long repairs and all 16 original branch meanings.
        point=seed_modules(base,cfg,summary['modules'])
        p=AnnualResilienceMILP(base,windows,cfg,env=env,fixed_modules=point,compact_fixed=True,compact=True)
        try:
            policy=fix_seed_policy(p)
            p.model.Params.Method=1;p.model.Params.TimeLimit=120
            p.model.optimize()
            if not p.model.SolCount: raise AssertionError('Winter seed policy has no feasible solution')
            result=p.result();write_json(out/'winter_seed_check.json',dict(modules=point,policy=policy,result=result))
            print(json.dumps(dict(winter_seed_passed=True,modules=point,objective_yuan=result['objective_yuan'],audit=result['audit'])),flush=True)
        finally:p.close()
    # Root lower-bound progress is indicative only: this short benchmark is
    # not a claim about full-year timing or superiority across all instances.
    # Prefer dual simplex on a bound tie, because the historical full-year run
    # spent most root time in barrier crossover. Subsecond timing noise is not
    # evidence that either method is faster.
    selected=max(records,key=lambda r:(r['bound_yuan'],-r['method']))['method']
    write_json(out/'verification.json',dict(passed=True,frozen_week_objective_yuan=summary['objective_yuan'],
        original_variables=summary['variables'],original_binaries=summary['binary_variables'],
        original_constraints=summary['linear_constraints'],weekly_root_tests=records,
        root_method_recommendation=selected,winter_seed_passed=True))


if __name__=='__main__':main()
