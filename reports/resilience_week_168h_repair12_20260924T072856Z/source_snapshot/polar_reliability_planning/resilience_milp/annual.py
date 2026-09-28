"""Annual chronological MILP with state-linked, recombining stress windows.

One normal annual trajectory and alternative 72-hour windows share capacity.
Only decisions inside a window are duplicated. States and outstanding flexible
jobs return to the annual trajectory; no free initial or terminal inventories.
Annual CVaR is conservatively bounded by the sum of window CVaRs.
"""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from .model import (MILPConfig, MILPPath, FaultRecoveryConfig, ResilienceMILP, weighted_cvar,
                    exogenous_information_nodes)
from ..resilience_v2.model import SyntheticYear, CAPACITY_STEPS, day_ahead_weather_risk


def healthy_path(year, name='annual_normal', weight=1.):
    h=year.hours
    return MILPPath(name,weight,year,np.zeros(h,bool),np.zeros(h,bool),
                    np.ones(h,bool),np.ones(h,bool),main_bus_available=np.ones(h,bool))


def annual_scenarios(year: SyntheticYear, starts=(816,2496,5616,7536), window_hours=72,
                     recovery: FaultRecoveryConfig | None = None):
    """Four declared seasonal risk opportunities, not calibrated annual rates.

    Each window has the same 2 x 8 conditional branches as the short demo.
    The healthy branch reuses the annual variables and has probability .48.
    Several windows can experience events in the same year.
    """
    recovery=recovery or FaultRecoveryConfig()
    recovery.check()
    if window_hours!=72:
        raise ValueError('the declared stress recipe uses 72-hour windows')
    starts=tuple(sorted(starts))
    if not starts or any(a<0 or a%24 or a+window_hours>year.hours for a in starts):
        raise ValueError('windows must fit the annual clock and start at midnight')
    if any(a+window_hours>b for a,b in zip(starts,starts[1:])):
        raise ValueError('stress windows must not overlap')
    normal=replace(year,extreme_weather=np.zeros(year.hours,bool),weather_risk=np.zeros(year.hours,bool))
    base=replace(healthy_path(normal),run_repair_hours=recovery.diesel_run_hours,
                 start_repair_hours=recovery.diesel_start_hours)
    windows=[]
    kinds=[('none',.60),('renewable_bus',.10),('main_bus',.10),('compound',.08),
           ('pcs',.03),('ups',.03),('diesel_run',.04),('diesel_start',.02)]
    for j,start in enumerate(starts):
        branches=[]
        for storm,pweather in ((False,.8),(True,.2)):
            extreme=np.zeros(year.hours,bool)
            if storm: extreme[start+24:start+54]=True
            ambient=normal.ambient_c.copy()
            ambient[extreme]=-12.
            weather=replace(normal,extreme_weather=extreme,
                            weather_risk=day_ahead_weather_risk(extreme),ambient_c=ambient)
            for kind,pfault in kinds:
                weight=pweather*pfault
                if not storm and kind=='none':
                    branches.append(replace(base,weight=weight))
                    continue
                name=f'w{j}_{"storm" if storm else "normal"}_{kind}'
                p=replace(healthy_path(weather,name,weight),
                          run_repair_hours=recovery.diesel_run_hours,
                          start_repair_hours=recovery.diesel_start_hours)
                event=start+36
                duration=recovery.renewable_bus_storm_hours if storm else recovery.renewable_bus_normal_hours
                if event+max(duration,recovery.main_bus_hours,recovery.pcs_hours,recovery.ups_hours)>start+window_hours:
                    raise ValueError('fault recovery must fit the stress window')
                end=event+duration
                if kind in ('renewable_bus','compound'): p.renewable_bus_fault[event:end]=True
                if kind=='main_bus': p.main_bus_available[event:event+recovery.main_bus_hours]=False
                if kind=='pcs': p.pcs_available[event:event+recovery.pcs_hours]=False
                if kind=='ups': p.ups_available[event:event+recovery.ups_hours]=False
                # A run shock is observable only if this engine was online.
                if kind=='diesel_run': p=replace(p,run_shocks=frozenset({(0,event-1)}))
                if kind=='diesel_start': p=replace(p,start_shocks=frozenset((0,t) for t in range(event,end)))
                branches.append(p)
        windows.append({'start':start,'stop':start+window_hours,'paths':branches})
    return base,windows


class AnnualResilienceMILP(ResilienceMILP):
    def __init__(self,base,windows,cfg,economic_budget=None,env=None,fixed_modules=None,compact_fixed=False):
        started=time.monotonic()
        if not windows or any(a['stop']>b['start'] for a,b in zip(windows,windows[1:])):
            raise ValueError('annual stress windows must be nonempty, ordered and disjoint')
        self.windows=windows
        print(f'Building annual trajectory: {base.year.hours} hourly steps',flush=True)
        super().__init__([replace(base,weight=1.)],cfg,economic_budget,env,fixed_modules,compact_fixed)
        m=self.model
        # Replace the standalone risk limits with annual weighted limits.
        # Ordinary load shedding is allowed (and charged) also outside events.
        m.remove(m.getConstrByName('eens_limit'))
        m.remove(m.getConstrByName('cvar_limit'))
        self.groups=[]
        for j,window in enumerate(windows):
            start,stop=window['start'],window['stop']
            if start%24 or stop%24 or not 0<=start<stop<=self.h:
                raise ValueError('window must align with complete annual days')
            if abs(sum(p.weight for p in window['paths'])-1)>1e-8:
                raise ValueError('conditional branch weights must sum to one within each window')
            group=[]
            print(f'Building stress window {j+1}/{len(windows)}: hours {start}..{stop-1}',flush=True)
            for p in window['paths']:
                p.check()
                if p.year.timestamps!=base.year.timestamps:
                    raise ValueError('stress and annual clocks must align')
                if p.name==base.name:
                    index=0
                else:
                    index=len(self.paths)
                    self.paths.append(p)
                    self.blocks.append(self._add_path(index,p,parent=self.blocks[0],start=start,stop=stop))
                group.append((index,p.weight))
            self.groups.append(group)
        self.nodes=exogenous_information_nodes(self.paths)
        for window,group in zip(windows,self.groups):
            self._add_nonanticipativity([s for s,_ in group],window['start'],window['stop'])

        # Disjoint windows replace the corresponding nominal operating costs.
        # Express outside cost directly (no signed extrapolation of scenarios).
        components={k:gp.LinExpr() for k in self.blocks[0]['cost_components']}
        outside_loss=gp.LinExpr()
        cursor=0
        for window in windows:
            for k,v in self._cost_between(self.blocks[0],cursor,window['start']).items(): components[k]+=v
            outside_loss+=self._loss_between(self.blocks[0],cursor,window['start'])
            cursor=window['stop']
        for k,v in self._cost_between(self.blocks[0],cursor,self.h).items(): components[k]+=v
        outside_loss+=self._loss_between(self.blocks[0],cursor,self.h)
        self.outside_loss=outside_loss
        self.window_losses=[];self.window_cvars=[];self.window_eens=[]
        for j,(window,group) in enumerate(zip(windows,self.groups)):
            losses=[]
            eta=m.addVar(lb=0,name=f'window_{j}_cvar_eta')
            tail=gp.LinExpr();expected=gp.LinExpr()
            for k,(s,weight) in enumerate(group):
                cost=self._cost_between(self.blocks[s],window['start'],window['stop'])
                for name,expr in cost.items(): components[name]+=weight*expr
                q=m.addVar(lb=0,name=f'window_{j}_branch_{k}_loss_kwh')
                m.addConstr(q==self._loss_between(self.blocks[s],window['start'],window['stop']))
                z=m.addVar(lb=0,name=f'window_{j}_branch_{k}_cvar_excess')
                m.addConstr(z>=q-eta)
                expected+=weight*q;tail+=weight*z
                losses.append(q)
            self.window_losses.append(losses)
            self.window_eens.append(expected)
            self.window_cvars.append(eta+tail/(1-cfg.alpha))
        self.annual_components=components
        self.operation=gp.quicksum(components.values())
        self.eens=outside_loss+gp.quicksum(self.window_eens)
        self.cvar=outside_loss+gp.quicksum(self.window_cvars)
        m.addConstr(self.eens<=cfg.eens_limit_kwh,name='annual_eens_limit')
        m.addConstr(self.cvar<=cfg.cvar_limit_kwh,name='annual_cvar_conservative_limit')
        m.setObjective(self.investment+self.operation,GRB.MINIMIZE)
        m.update()
        if m.NumQConstrs or m.NumQNZs: raise AssertionError('annual model must remain MILP')
        self.build_seconds=time.monotonic()-started
        print(f'Annual MILP built in {self.build_seconds:.1f}s: {m.NumVars} variables, {m.NumConstrs} linear constraints',flush=True)

    def audit_annual(self):
        """Check active trajectories and solver feasibility without materializing
        sixty copies of the unchanged annual dispatch.
        """
        c=self.cfg;m=self.model
        errors={'solver_constraints':float(m.ConstrVio),'solver_bounds':float(m.BoundVio),
                'solver_integrality':float(m.IntVio),
                'integer_modules':max(abs(v.X-round(v.X)) for v in self.n.values()),
                'power_balance':0.,'core_loss':0.,'battery_transition':0.,'ups_transition':0.,
                'thermal_transition':0.,'gfm_without_power':0.,'annual_window_link':0.,
                'shift_energy':0.,'nonanticipativity':0.,'annual_cost_accounting':0.}
        def put(name,v): errors[name]=max(errors[name],abs(float(v)))
        decay=math.exp(-c.thermal_ua_kw_per_k/c.thermal_c_kwh_per_k)
        gain=(1-decay)/c.thermal_ua_kw_per_k if c.thermal_ua_kw_per_k else 1/c.thermal_c_kwh_per_k
        for s,(p,b) in enumerate(zip(self.paths,self.blocks)):
            start,stop=b['active_start'],b['active_stop']
            for t in range(start,stop):
                shifted=sum(b['shift_service'][a,t].X for a in range(max(0,t-c.shift_window_hours+1),t+1))
                diesel=sum(b['diesel_power'][i,t].X for i in range(self.dmax))
                heaters=sum(b['heater'][i,t].X for i in range(self.dmax))
                supply=b['wind'][t].X+b['pv'][t].X+diesel+b['grid_main'][t].X+b['battery_discharge'][t].X
                demand=b['core_main'][t].X+p.year.rigid_kw[t]-b['rigid_shed'][t].X+b['interrupt_served'][t].X+shifted+b['battery_charge'][t].X+heaters+b['ups_diesel_charge'][t].X
                put('power_balance',supply-demand)
                put('core_loss',p.year.core_kw[t]-b['core_main'][t].X-b['ups_discharge'][t].X)
                put('battery_transition',b['battery_energy'][t+1].X-b['battery_energy'][t].X-c.battery_efficiency*b['battery_charge'][t].X+b['battery_discharge'][t].X/c.battery_efficiency)
                put('ups_transition',b['ups_energy'][t+1].X-b['ups_energy'][t].X-c.ups_efficiency*b['ups_charge'][t].X+b['ups_discharge'][t].X/c.ups_efficiency)
                real=max([b['battery_charge'][t].X,b['battery_discharge'][t].X]+[b['diesel_power'][i,t].X for i in range(self.dmax)])
                if (not c.external_grid_enabled or p.grid_fault[t]) and b['bus_live'][t].X>.5:
                    put('gfm_without_power',max(0,c.gfm_min_power_kw-real))
                for i in range(self.dmax):
                    if self.installed[i].X>.5:
                        put('thermal_transition',b['temperature'][i,t+1].X-decay*b['temperature'][i,t].X-(1-decay)*p.year.ambient_c[t]-gain*b['heater'][i,t].X)
            for a in range(max(0,start-c.shift_window_hours+1),stop):
                served=sum(b['shift_service'][a,t].X for t in range(a,min(self.h,a+c.shift_window_hours)))
                put('shift_energy',served+b['shift_emergency_shed'][a].X+b['shift_late_shed'][a].X-p.year.flex_shiftable_kw[a])
            if s:
                for t in (start,stop):
                    for name in ('battery_energy','ups_energy'):
                        put('annual_window_link',b[name][t].X-self.blocks[0][name][t].X)
                    for i in range(self.dmax):
                        put('annual_window_link',b['temperature'][i,t].X-self.blocks[0]['temperature'][i,t].X)
        for window,group in zip(self.windows,self.groups):
            parents={s:0 for s,_ in group}
            for t in range(window['start'],window['stop']):
                pre_groups={};post_groups={};next_parents={}
                for s,_ in group:
                    b=self.blocks[s]
                    pre=(parents[s],int(self.nodes[s,t]),tuple(round(b['failed_pre'][i,t].X) for i in range(self.dmax)))
                    r=pre_groups.setdefault(pre,s)
                    for a,v in zip(b['pre_controls'][t],self.blocks[r]['pre_controls'][t]): put('nonanticipativity',a.X-v.X)
                    post=(pre,tuple(round(b['start_fail'][i,t].X) for i in range(self.dmax)))
                    r=post_groups.setdefault(post,s)
                    for name in ('post_controls','post_states'):
                        for a,v in zip(b[name][t],self.blocks[r][name][t]): put('nonanticipativity',a.X-v.X)
                    next_parents[s]=r
                parents=next_parents
        put('annual_cost_accounting',m.ObjVal-self.investment.getValue()-sum(v.getValue() for v in self.annual_components.values()))
        return {'passed':max(errors.values())<=1e-5,'max_violation':max(errors.values()),
                'violations':errors,'scope':'annual and window algebra, state links, finite-tree information; no transient stability proof'}

    def result(self):
        m=self.model
        annual=self.h==8760
        result={'solver':'Gurobi','solver_version':list(gp.gurobi.version()),'solver_status':int(m.Status),
                'model_scope':'annual_with_linked_resilience_windows' if annual else 'chronological_with_linked_resilience_windows','hours':self.h,
                'window_hours':[w['stop']-w['start'] for w in self.windows],
                'window_count':len(self.windows),'scenario_count':sum(map(len,self.groups)),
                'unique_dispatch_blocks':len(self.blocks),'synthetic':True,
                'cost_basis':f'capital investment plus {"one full year" if annual else str(self.h)+" hours"} expected operation; capital is not annualized',
                'risk_basis':f'{"annual" if annual else str(self.h)+"-hour horizon"} EENS=outside loss+sum of window EENS; CVaR upper bound=outside loss+sum of window CVaRs',
                'policy_scope':'finite seasonal scenario tree with common recovered states between windows',
                'fixed_modules':self.fixed_modules,'build_seconds':self.build_seconds,'runtime_seconds':float(m.Runtime),
                'variables':int(m.NumVars),'binary_variables':int(m.NumBinVars),'integer_variables':int(m.NumIntVars),
                'linear_constraints':int(m.NumConstrs),'general_linear_constraints':int(m.NumGenConstrs),
                'quadratic_constraints':int(m.NumQConstrs),'solution_count':int(m.SolCount),
                'economic_budget_yuan':self.budget,'time_limit_seconds':self.cfg.time_limit_seconds,
                'resolved_config':asdict(self.cfg),'required_ups_kwh':self.required_ups_kwh,'required_ups_kw':self.required_ups_kw}
        if not m.SolCount:
            result.update(status='infeasible_within_economic_bound' if m.Status==GRB.INFEASIBLE else 'unresolved_without_incumbent',
                          selected=None,economic_domain_certified=False)
            return result
        audit=self.audit_annual()
        if not audit['passed']: raise RuntimeError(f'Annual MILP audit failed: {audit}')
        modules={k:int(round(v.X)) for k,v in self.n.items()}
        selected={k:CAPACITY_STEPS[k]*v for k,v in modules.items()}
        metrics=[];windows=[]
        for j,(window,group,losses) in enumerate(zip(self.windows,self.groups,self.window_losses)):
            q=[float(v.X) for v in losses];weights=[p for _,p in group]
            windows.append({'window':j,'start_hour':window['start'],'stop_hour':window['stop'],
                            'eens_kwh':sum(a*b for a,b in zip(q,weights)),
                            'cvar_kwh':weighted_cvar(q,weights,self.cfg.alpha)})
            for (s,p),loss in zip(group,q):
                b=self.blocks[s];ts=range(window['start'],window['stop'])
                metrics.append({'window':j,'name':self.paths[s].name,'conditional_weight':p,
                    'regular_loss_kwh':loss,'core_loss_kwh':0.,
                    'ups_emergency_kwh':sum(b['ups_discharge'][t].X for t in ts),
                    'diesel_run_failures':sum(b['run_fail'][i,t].X for i in range(self.dmax) for t in ts),
                    'diesel_start_failures':sum(b['start_fail'][i,t].X for i in range(self.dmax) for t in ts),
                    'battery_initial_kwh':b['battery_energy'][window['start']].X,
                    'battery_terminal_kwh':b['battery_energy'][window['stop']].X})
        certified=m.ObjVal<=self.budget+1e-6
        status=('optimal_within_solver_gap_on_annual_scenario_tree' if annual else 'optimal_within_solver_gap_on_horizon_scenario_tree') if m.Status==GRB.OPTIMAL else 'feasible_incumbent_not_proven_optimal'
        if not certified: status='economic_bound_requires_expansion'
        if self.fixed_modules is not None: status='fixed_capacity_annual_dispatch_'+('optimal_within_gap' if m.Status==GRB.OPTIMAL else 'feasible_incumbent')
        result.update(status=status,selected=selected,modules=modules,economic_domain_certified=certified,
                      objective_yuan=float(m.ObjVal),objective_bound_yuan=float(m.ObjBound),mip_gap=float(m.MIPGap),
                      investment_yuan=float(self.investment.getValue()),expected_operation_yuan=float(self.operation.getValue()),
                      mean_cost_components_yuan={k:float(v.getValue()) for k,v in self.annual_components.items()},
                      outside_window_loss_kwh=float(self.outside_loss.getValue()),
                      eens_kwh=float(self.outside_loss.getValue())+sum(w['eens_kwh'] for w in windows),
                      cvar_upper_bound_kwh=float(self.outside_loss.getValue())+sum(w['cvar_kwh'] for w in windows),
                      window_metrics=windows,scenario_metrics=metrics,audit=audit,post_solution_rounding=False)
        return result

    def save(self,output,summary,write_model=True):
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (output/'resolved_config.json').write_text(json.dumps(asdict(self.cfg),indent=2)+'\n')
        self.paths[0].year.frame().to_csv(output/'annual_input.csv',index=False)
        manifest=[]
        for j,(window,group) in enumerate(zip(self.windows,self.groups)):
            manifest.append({'window':j,'start_hour':window['start'],'stop_hour':window['stop'],
                'branches':[{'block':s,'name':self.paths[s].name,'conditional_weight':weight,
                    'run_repair_hours':self.paths[s].run_repair_hours,
                    'start_repair_hours':self.paths[s].start_repair_hours,
                    'renewable_bus_outage_hours':int(self.paths[s].renewable_bus_fault[window['start']:window['stop']].sum()),
                    'main_bus_outage_hours':0 if self.paths[s].main_bus_available is None else int((~self.paths[s].main_bus_available[window['start']:window['stop']]).sum()),
                    'pcs_outage_hours':int((~self.paths[s].pcs_available[window['start']:window['stop']]).sum()),
                    'ups_outage_hours':int((~self.paths[s].ups_available[window['start']:window['stop']]).sum())}
                    for s,weight in group]})
        (output/'scenario_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if self.model.SolCount:
            arrays={}
            hourly=('wind','pv','battery_charge','battery_discharge','ups_charge','ups_discharge','core_main',
                    'rigid_shed','interrupt_shed','interrupt_adjust','bus_live','ups_on')
            unit_names=('online','request','attempt','pending','diesel_connected','diesel_power','heater',
                        'heater_enabled','failed_pre','failed_post','run_fail','start_fail')
            with (output/'hourly_dispatch.csv').open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['block','scenario','annual_hour','timestamp',*hourly,'diesel_kw','heater_kw',
                                 'shift_served_kw','battery_energy_kwh','ups_energy_kwh'])
                for s,(p,b) in enumerate(zip(self.paths,self.blocks)):
                    a,z=b['active_start'],b['active_stop'];times=range(a,z)
                    for k in hourly: arrays[f's{s}_{k}']=np.array([b[k][t].X for t in times])
                    for k in ('battery_energy','ups_energy'):
                        arrays[f's{s}_{k}']=np.array([b[k][t].X for t in range(a,z+1)])
                    for k in unit_names+('temperature',):
                        arrays[f's{s}_{k}']=np.array([[b[k][i,t].X for t in range(a,z+int(k=='temperature'))] for i in range(self.dmax)])
                    arrays[f's{s}_annual_hours']=np.arange(a,z)
                    for k in ('grid_fault','renewable_bus_fault','pcs_available','ups_available','main_bus_available'):
                        arrays[f's{s}_{k}']=getattr(p,k)[a:z]
                    for k in ('core_kw','rigid_kw','flex_interruptible_kw','flex_shiftable_kw','ambient_c',
                              'wind_clean_pu','pv_pu','extreme_weather','weather_risk'):
                        arrays[f's{s}_{k}']=getattr(p.year,k)[a:z]
                    for t in times:
                        shifted=sum(b['shift_service'][k,t].X for k in range(max(0,t-self.cfg.shift_window_hours+1),t+1))
                        writer.writerow([s,p.name,t,p.year.timestamps[t],*[b[k][t].X for k in hourly],
                            sum(b['diesel_power'][i,t].X for i in range(self.dmax)),sum(b['heater'][i,t].X for i in range(self.dmax)),
                            shifted,b['battery_energy'][t].X,b['ups_energy'][t].X])
            np.savez_compressed(output/'dispatch.npz',**arrays)
            self.model.write(str(output/'solution.sol'))
        if write_model: self.model.write(str(output/'model.lp'))


def solve_annual_planning(base,windows,cfg,output=None,fixed_modules=None):
    started=time.monotonic();budget=cfg.initial_economic_budget_yuan;rounds=[]
    if output:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if (output/'summary.json').exists(): raise ValueError('output already contains a completed run')
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        while True:
            planner=AnnualResilienceMILP(base,windows,cfg,budget,env,fixed_modules)
            try:
                remaining=max(0.,cfg.time_limit_seconds-(time.monotonic()-started)) if cfg.time_limit_seconds else None
                if remaining is not None and remaining<=0:
                    result={'status':'time_limit_during_model_build','selected':None,'economic_domain_certified':False,
                            'build_seconds':planner.build_seconds,'hours':base.year.hours,'synthetic':True}
                else:
                    result=planner.optimize(remaining or 0,output/'gurobi.log' if output else None)
                rounds.append({'budget_yuan':budget,'status':result['status'],
                    'build_seconds':planner.build_seconds,'runtime_seconds':result.get('runtime_seconds',0.)})
                elapsed=time.monotonic()-started
                done=(result.get('economic_domain_certified',False) or fixed_modules is not None
                      or (cfg.time_limit_seconds and elapsed>=cfg.time_limit_seconds)
                      or planner.model.Status in (GRB.TIME_LIMIT,GRB.INTERRUPTED))
                if done:
                    result['total_elapsed_seconds']=elapsed;result['economic_bound_rounds']=rounds
                    if output: planner.save(output,result)
                    return result
            finally: planner.close()
            budget*=2
