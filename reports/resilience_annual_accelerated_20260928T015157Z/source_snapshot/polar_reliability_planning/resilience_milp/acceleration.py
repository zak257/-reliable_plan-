"""Exact scenario compression and audited, restricted-policy warm starts.

Restrictions in ``fix_seed_policy`` are used ONLY to construct an incumbent.
The final capacity model has none of those operating/capacity restrictions.
"""
from dataclasses import fields, replace
import hashlib
import json
import math
import re

from gurobipy import GRB
import numpy as np


def path_fingerprint(path):
    digest=hashlib.sha256()
    for f in fields(path):
        if f.name in ('name','weight'): continue
        value=getattr(path,f.name)
        if f.name=='year':
            for yf in fields(value):
                item=getattr(value,yf.name)
                digest.update(yf.name.encode())
                if isinstance(item,np.ndarray):
                    digest.update(str(item.dtype).encode());digest.update(item.tobytes())
                else: digest.update(json.dumps(item).encode())
        elif isinstance(value,np.ndarray):
            digest.update(str(value.dtype).encode());digest.update(value.tobytes())
        else:
            digest.update(json.dumps(sorted(value) if isinstance(value,frozenset) else value).encode())
    return digest.hexdigest()


def merge_duplicate_paths(windows):
    """Merge only equal physics, primitive shocks AND observation histories."""
    merged=[];aliases=[]
    for j,window in enumerate(windows):
        groups={}
        for p in window['paths']:
            groups.setdefault(path_fingerprint(p),[]).append(p)
        paths=[]
        for key,group in groups.items():
            representative=replace(group[0],weight=sum(p.weight for p in group))
            paths.append(representative)
            aliases.append(dict(window=j,representative=representative.name,
                members=[dict(name=p.name,weight=p.weight) for p in group],
                merged_weight=representative.weight,physics_information_sha256=key))
        merged.append({**window,'paths':paths})
    return merged,aliases


def diesel_seed_schedule(path,count,cfg):
    """Causal all-available-units-on policy, including preparation and exposure."""
    h=path.year.hours
    names=('request','attempt','pending','online','diesel_connected','failed_pre',
           'failed_post','run_fail','start_fail','heater_enabled')
    states={k:np.zeros((count,h),dtype=np.int8) for k in names}
    requested=np.full(count,-100000,dtype=int)
    for t in range(h):
        for i in range(count):
            prev=states['online'][i,t-1] if t else 0
            pending=states['pending'][i,t-1] if t else 0
            states['run_fail'][i,t]=int((i,t-1) in path.run_shocks)*prev if t else 0
            failed=(states['run_fail'][i,max(0,t-path.run_repair_hours+1):t+1].sum()+
                    states['start_fail'][i,max(0,t-path.start_repair_hours+1):t].sum())
            states['failed_pre'][i,t]=failed
            attempt=int(pending and not failed and t-requested[i]>=cfg.preparation_hours)
            states['attempt'][i,t]=attempt
            start_fail=int((i,t) in path.start_shocks)*attempt
            states['start_fail'][i,t]=start_fail
            states['failed_post'][i,t]=failed+start_fail
            states['pending'][i,t]=pending-attempt
            states['online'][i,t]=int(not(failed+start_fail) and (prev or attempt))
        # Requests depend on the pre-start information only. Whether a start
        # succeeded does not affect same-hour requests for other idle units.
        for i in range(count):
            prev=states['online'][i,t-1] if t else 0
            was_pending=states['pending'][i,t-1] if t else 0
            request=int(not prev and not was_pending and not states['failed_pre'][i,t]
                        and not path.year.extreme_weather[t]
                        and states['pending'][:,t].sum()<cfg.crews)
            if request:
                requested[i]=t;states['pending'][i,t]=1
            states['request'][i,t]=request
            states['heater_enabled'][i,t]=int(request or (states['heater_enabled'][i,t-1] if t else 0))
            live=path.main_bus_available is None or path.main_bus_available[t]
            states['diesel_connected'][i,t]=int(live and states['online'][i,t])
    return states


def seed_modules(base,cfg,reference=None):
    """Use the checked weekly wind/PV point; size a conservative startup bridge.

    These module counts are a heuristic seed, never final optimization bounds.
    """
    y=base.year;load=y.core_kw+y.rigid_kw+y.flex_interruptible_kw+y.flex_shiftable_kw
    count=math.ceil(float(load.max())/100)+1
    schedule=diesel_seed_schedule(base,count,cfg)
    wind=int((reference or {}).get('wind_kw',2));pv=int((reference or {}).get('pv_kw',0))
    startup=min(y.hours,(count+1)*cfg.preparation_hours)
    renewable=100*wind*y.wind_clean_pu[:startup]+100*pv*y.pv_pu[:startup]
    deficit=np.maximum(0,load[:startup]+cfg.heater_kw-renewable-100*schedule['online'][:,:startup].sum(axis=0))
    bridge=1.25*float(deficit.sum())/(cfg.battery_efficiency*(cfg.battery_initial_soc-cfg.battery_min_soc))
    pcs=math.ceil((float(load.max())+cfg.heater_kw)/50)
    battery=max(int((reference or {}).get('battery_kwh',0)),pcs,math.ceil(bridge/50))
    peak=float(y.core_kw.max())
    return dict(wind_kw=wind,pv_kw=pv,diesel_units=count,battery_kwh=battery,pcs_kw=pcs,
        ups_kwh=math.ceil(peak*cfg.ups_bridge_hours/((cfg.ups_standby_soc-cfg.ups_min_soc)*cfg.ups_efficiency)/50),
        ups_kw=math.ceil(peak*cfg.ups_power_margin/50))


def fix_seed_policy(planner):
    """Fix an explicitly restricted feasible-policy search, not the final MILP.

    Full physics, information, risk, boundary-state and thermal constraints are
    retained. A solution is accepted only after the original annual audit.
    """
    m=planner.model;c=planner.cfg
    if planner.fixed_modules is None: raise ValueError('seed requires fixed integer capacities')
    assigned={}
    def fix(var,value):
        key=var.index;value=float(value)
        if key in assigned and abs(assigned[key]-value)>1e-10:
            raise ValueError(f'Conflicting seed assignments for {var.VarName}')
        if value<var.LB-1e-9 or value>var.UB+1e-9:
            raise ValueError(f'Seed contradicts known state {var.VarName}: {value}')
        assigned[key]=value;var.LB=value;var.UB=value
    for k,var in planner.n.items(): fix(var,planner.fixed_modules[k])
    count=planner.fixed_modules['diesel_units']
    for i,var in planner.installed.items(): fix(var,int(i<count))
    schedules=[]
    for p,b in zip(planner.paths,planner.blocks):
        states=diesel_seed_schedule(p,planner.dmax,c)
        schedules.append(states)
        y=p.year;load=y.core_kw+y.rigid_kw+y.flex_interruptible_kw+y.flex_shiftable_kw
        for t in range(b['active_start'],b['active_stop']):
            for name,array in states.items():
                for i in range(planner.dmax): fix(b[name][i,t],array[i,t])
            live=int(p.main_bus_available is None or p.main_bus_available[t])
            ups=int(not live)
            fix(b['bus_live'][t],live);fix(b['ups_on'][t],ups)
            diesel=100*states['online'][:,t].sum()
            # The first hours need battery support before the crew can start
            # engines. Replenish during the remainder of the first two days.
            dis=int(live and p.pcs_available[t] and t<count*c.preparation_hours
                    and diesel<load[t]+c.heater_kw)
            charge=int(live and p.pcs_available[t] and not dis and t<48
                       and diesel>=load[t]+c.heater_kw)
            fix(b['battery_dis_on'][t],dis);fix(b['battery_ch_on'][t],charge)
            if live:
                fix(b['rigid_shed'][t],0);fix(b['interrupt_shed'][t],0)
        # No deferred tasks in this seed; zero late loss. Emergency cancellation
        # during a main-bus trip remains active and is charged by the full model.
        for (a,t),var in b['shift_service'].items():
            if a!=t and b['active_start']<=t<b['active_stop']: fix(var,0)
        for a,var in b['shift_late_shed'].items():
            if b['active_start']<=min(a+c.shift_window_hours-1,planner.h-1)<b['active_stop']: fix(var,0)
    m.update()
    # All endogenous information distinctions are determined by the fixed,
    # causal physical schedule. Fix these auxiliary binaries as well.
    delta=re.compile(r'na_(pre_delta|start_delta)_(\d+)_(\d+)_(\d+)_(\d+)')
    history=re.compile(r'na_(pre_history|post_history)_(\d+)_(\d+)_(\d+)')
    past={};pre={}
    for var in m.getVars():
        match=delta.fullmatch(var.VarName)
        if match:
            kind,s,r,i,t=match.groups();s,r,i,t=map(int,(s,r,i,t))
            key='failed_pre' if kind=='pre_delta' else 'start_fail'
            fix(var,int(schedules[s][key][i,t]!=schedules[r][key][i,t]))
        match=history.fullmatch(var.VarName)
        if match:
            kind,s,r,t=match.groups();s,r,t=map(int,(s,r,t))
            if kind=='pre_history':
                value=int(past.get((s,r,t-1),0) or np.any(schedules[s]['failed_pre'][:,t]!=schedules[r]['failed_pre'][:,t]))
                pre[s,r,t]=value
            else:
                value=int(pre[s,r,t] or np.any(schedules[s]['start_fail'][:,t]!=schedules[r]['start_fail'][:,t]))
                past[s,r,t]=value
            fix(var,value)
    m.update()
    # Resolve fixed indicators exactly, before Gurobi introduces SOS/big-M
    # auxiliaries around the unbounded thermal states. In the seed all
    # controllers must be fixed; the free model retains its original logic.
    indicators=m.getGenConstrs();active=0
    for gc in indicators:
        if gc.GenConstrType!=GRB.GENCONSTR_INDICATOR: raise ValueError('Unexpected seed general constraint')
        control,value,expr,sense,rhs=m.getGenConstrIndicator(gc)
        if control.LB!=control.UB: raise ValueError(f'Unfixed seed indicator {control.VarName}')
        if int(round(control.LB))==value:
            m.addLConstr(expr,sense,rhs);active+=1
    m.remove(indicators);m.update()
    return dict(fixed_variables=len(assigned),resolved_indicators=len(indicators),active_linearized_indicators=active,
                scope='warm-start policy only; no restrictions transferred to final planning')


def read_solution(path):
    values={}
    for line in path.read_text().splitlines():
        if line and not line.startswith('#'):
            name,value=line.split();values[name]=float(value)
    return values


def apply_start(planner,values,seed_diesel_count):
    """Transfer a fully audited dispatch; fill newly created uninstalled units."""
    c=planner.cfg;y=planner.paths[0].year
    decay=math.exp(-c.thermal_ua_kw_per_k/c.thermal_c_kwh_per_k)
    passive=np.empty(planner.h+1);passive[0]=c.initial_temperature_c
    for t in range(planner.h): passive[t+1]=decay*passive[t]+(1-decay)*y.ambient_c[t]
    unit=re.compile(r's\d+_(\w+)\[(\d+),(\d+)\]$')
    installed=re.compile(r'diesel_installed\[(\d+)\]$')
    matched=0;filled=0;missing=[]
    for var in planner.model.getVars():
        name=var.VarName
        if name in values:
            var.Start=values[name];matched+=1;continue
        match=unit.fullmatch(name)
        if match and int(match[2])>=seed_diesel_count and match[1]!='shift_service':
            var.Start=float(passive[int(match[3])]) if match[1]=='temperature' else 0.
            filled+=1;continue
        match=installed.fullmatch(name)
        if match and int(match[1])>=seed_diesel_count:
            var.Start=0.;filled+=1;continue
        if var.LB==var.UB:
            var.Start=var.LB;filled+=1;continue
        missing.append(name)
    planner.model.update()
    return dict(matched=matched,additional_uninstalled_values=filled,missing_count=len(missing),missing_examples=missing[:20])
