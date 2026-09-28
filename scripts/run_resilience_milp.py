#!/usr/bin/env python3
"""Direct Gurobi capacity and operation MILP with emergency-only UPS."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import json
import hashlib
import shutil

from polar_reliability_planning.resilience_v2.model import generate_synthetic_year
from polar_reliability_planning.resilience_milp import MILPConfig,make_demo_paths,solve_planning
from polar_reliability_planning.resilience_milp.annual import annual_scenarios,solve_annual_planning


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/resilience_milp_emergency_ups'))
    parser.add_argument('--mode',choices=('annual','demo'),default='annual',help='annual=8760h plus linked stress windows; demo=standalone short validation')
    parser.add_argument('--hours',type=int,default=None)
    parser.add_argument('--stress-start-days',type=int,nargs='+',default=[34,104,234,314],help='zero-based start days of the four 72h windows')
    parser.add_argument('--time-limit-seconds',type=float,default=10000.)
    parser.add_argument('--mip-gap',type=float,default=.001)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--loss-cost',type=float,default=1000.)
    parser.add_argument('--ups-energy-cost',type=float,default=750.)
    parser.add_argument('--ups-power-cost',type=float,default=1200.)
    parser.add_argument('--economic-budget',type=float,default=3_000_000.)
    args=parser.parse_args()
    hours=args.hours if args.hours is not None else (8760 if args.mode=='annual' else 72)
    if args.mode=='annual' and hours!=8760:
        parser.error('annual planning requires --hours 8760; use --mode demo for a standalone short test')
    cfg=replace(MILPConfig(),time_limit_seconds=args.time_limit_seconds,mip_gap=args.mip_gap,
                threads=args.threads,loss_yuan_per_kwh=args.loss_cost,
                ups_yuan_per_kwh=args.ups_energy_cost,ups_yuan_per_kw=args.ups_power_cost,
                initial_economic_budget_yuan=args.economic_budget)
    cfg.check()
    if (args.output/'summary.json').exists():
        parser.error('output already contains a completed run; use a new directory')
    args.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__).resolve(),root/'polar_reliability_planning/resilience_milp/model.py',
             root/'polar_reliability_planning/resilience_milp/annual.py',root/'polar_reliability_planning/resilience_v2/model.py']
    (args.output/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    (args.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    snapshots=args.output/'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for source in sources:
        target=snapshots/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    (args.output/'run_definition.json').write_text(json.dumps({'mode':args.mode,'hours':hours,
        'stress_window_hours':72 if args.mode=='annual' else None,
        'stress_start_days':args.stress_start_days if args.mode=='annual' else [],
        'all_capacities_optimized':True,'synthetic_data':True},indent=2)+'\n')
    year=generate_synthetic_year(hours,cfg.seed)
    if args.mode=='annual':
        base,windows=annual_scenarios(year,starts=tuple(24*d for d in args.stress_start_days))
        year.frame().to_csv(args.output/'generated_year.csv',index=False)
        result=solve_annual_planning(base,windows,cfg,args.output)
    else:
        paths=make_demo_paths(year)
        result=solve_planning(paths,cfg,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resolved_config','scenario_metrics')},ensure_ascii=False,indent=2))
    return 0 if result.get('selected') else 2


if __name__=='__main__': raise SystemExit(main())
