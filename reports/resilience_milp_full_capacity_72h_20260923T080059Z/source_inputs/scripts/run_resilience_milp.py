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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('reports/resilience_milp_emergency_ups'))
    parser.add_argument('--hours',type=int,default=72)
    parser.add_argument('--time-limit-seconds',type=float,default=10000.)
    parser.add_argument('--mip-gap',type=float,default=.001)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--loss-cost',type=float,default=1000.)
    parser.add_argument('--ups-energy-cost',type=float,default=750.)
    parser.add_argument('--ups-power-cost',type=float,default=1200.)
    parser.add_argument('--economic-budget',type=float,default=3_000_000.)
    args=parser.parse_args()
    cfg=replace(MILPConfig(),time_limit_seconds=args.time_limit_seconds,mip_gap=args.mip_gap,
                threads=args.threads,loss_yuan_per_kwh=args.loss_cost,
                ups_yuan_per_kwh=args.ups_energy_cost,ups_yuan_per_kw=args.ups_power_cost,
                initial_economic_budget_yuan=args.economic_budget)
    cfg.check()
    if (args.output/'summary.json').exists():
        parser.error('output already contains a completed run; use a new directory')
    args.output.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'polar_reliability_planning/resilience_milp/model.py']
    (args.output/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    (args.output/'resolved_config.json').write_text(json.dumps(asdict(cfg),indent=2)+'\n')
    snapshots=args.output/'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for source in sources:
        shutil.copyfile(source,snapshots/source.name)
    year=generate_synthetic_year(args.hours,cfg.seed)
    paths=make_demo_paths(year)
    result=solve_planning(paths,cfg,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resolved_config','scenario_metrics')},ensure_ascii=False,indent=2))
    return 0 if result.get('selected') else 2


if __name__=='__main__': raise SystemExit(main())
