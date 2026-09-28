"""Joint integer capacity / causal dispatch MILP; no external capacity enumeration.

Costs, weather and fault weights in the demonstration are synthetic assumptions.
Nonanticipativity includes endogenous revelation of diesel run/start failures.
The resulting policy is defined on the supplied finite scenario tree only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from itertools import combinations
from collections import ChainMap
import json
import math
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from ..resilience_v2.model import SyntheticYear, CAPACITY_STEPS, day_ahead_weather_risk


@dataclass(frozen=True)
class MILPConfig:
    wind_yuan_per_kw: float = 1500.
    pv_yuan_per_kw: float = 1000.
    diesel_yuan_per_kw: float = 1800.
    battery_yuan_per_kwh: float = 250.
    pcs_yuan_per_kw: float = 400.
    ups_yuan_per_kwh: float = 750.
    ups_yuan_per_kw: float = 1200.
    external_grid_enabled: bool = False
    ups_recharge_deadline_hours: int = 24
    grid_yuan_per_kwh: float = .65
    diesel_yuan_per_kwh: float = .95
    loss_yuan_per_kwh: float = 1000.
    flex_adjustment_yuan_per_kwh: float = .12
    battery_cycle_yuan_per_kwh: float = .01
    battery_efficiency: float = .95
    battery_initial_soc: float = .60
    battery_min_soc: float = .10
    ups_efficiency: float = .98
    ups_standby_soc: float = .95
    ups_min_soc: float = .05
    ups_bridge_hours: float = 8.
    ups_power_margin: float = 1.5
    storage_min_duration_h: float = 1.
    gfm_min_power_kw: float = 1.
    diesel_min_output_kw: float = 1.
    diesel_idle_equivalent_kw: float = 5.
    preparation_hours: int = 3
    minimum_up_hours: int = 3
    crews: int = 1
    thermal_ua_kw_per_k: float = .15
    thermal_c_kwh_per_k: float = 1.5
    heater_kw: float = 12.
    ready_temperature_c: float = 5.
    initial_temperature_c: float = -10.
    interruptible_max_fraction: float = .5
    interruptible_daily_energy_fraction: float = .2
    shift_window_hours: int = 24
    shift_power_multiplier: float = 2.
    eens_limit_kwh: float = 100.
    cvar_limit_kwh: float = 1000.
    alpha: float = .95
    # An economic computational bound, verified against the attained objective.
    # If no incumbent beats it, it is expanded, never claimed to be global.
    initial_economic_budget_yuan: float = 3_000_000.
    time_limit_seconds: float = 10000.
    mip_gap: float = .001
    threads: int = 4
    seed: int = 20260923

    @property
    def module_costs(self):
        return {'wind_kw':100*self.wind_yuan_per_kw, 'pv_kw':100*self.pv_yuan_per_kw,
                'diesel_units':100*self.diesel_yuan_per_kw,
                'battery_kwh':50*self.battery_yuan_per_kwh, 'pcs_kw':50*self.pcs_yuan_per_kw,
                'ups_kwh':50*self.ups_yuan_per_kwh, 'ups_kw':50*self.ups_yuan_per_kw}

    def check(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('all configuration values must be finite')
        if self.ups_yuan_per_kwh <= self.battery_yuan_per_kwh or self.ups_yuan_per_kw <= self.pcs_yuan_per_kw:
            raise ValueError('UPS energy/power costs must exceed battery/PCS costs respectively')
        if min(self.module_costs.values()) <= 0:
            raise ValueError('strictly positive module investment costs required')
        if min(self.grid_yuan_per_kwh,self.diesel_yuan_per_kwh,self.loss_yuan_per_kwh,
               self.flex_adjustment_yuan_per_kwh,self.battery_cycle_yuan_per_kwh) < 0:
            raise ValueError('operating and loss costs must be nonnegative')
        if not 0 <= self.battery_min_soc < self.battery_initial_soc <= 1:
            raise ValueError('invalid battery SOC')
        if not 0 <= self.ups_min_soc < self.ups_standby_soc <= 1:
            raise ValueError('invalid UPS SOC')
        if not 0 < self.battery_efficiency <= 1 or not 0 < self.ups_efficiency <= 1:
            raise ValueError('invalid efficiency')
        if not 0 < self.alpha < 1 or min(self.eens_limit_kwh,self.cvar_limit_kwh) < 0:
            raise ValueError('invalid risk limits')
        for name in ('preparation_hours','minimum_up_hours','crews','shift_window_hours','threads','ups_recharge_deadline_hours'):
            if getattr(self,name) < 1 or int(getattr(self,name)) != getattr(self,name):
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < self.gfm_min_power_kw <= 50 or not self.gfm_min_power_kw <= self.diesel_min_output_kw <= 100:
            raise ValueError('invalid active GFM power thresholds')
        if not 0 <= self.interruptible_max_fraction <= 1 or not 0 <= self.interruptible_daily_energy_fraction <= 1:
            raise ValueError('invalid allowed flexibility')
        if min(self.thermal_c_kwh_per_k,self.heater_kw,self.storage_min_duration_h,self.initial_economic_budget_yuan) <= 0:
            raise ValueError('positive thermal/coupling/economic parameters required')
        if min(self.thermal_ua_kw_per_k,self.ups_bridge_hours,self.time_limit_seconds,self.mip_gap) < 0:
            raise ValueError('negative physical/solver parameter')


@dataclass(frozen=True)
class MILPPath:
    name: str
    weight: float
    year: SyntheticYear
    grid_fault: np.ndarray
    renewable_bus_fault: np.ndarray
    pcs_available: np.ndarray
    ups_available: np.ndarray
    # Sparse primitive shocks indexed (unit, hour); hidden until exposure.
    # Run shock (i,t) acts during [t,t+1), revealed at t+1 iff i is online.
    run_shocks: frozenset[tuple[int,int]] = frozenset()
    start_shocks: frozenset[tuple[int,int]] = frozenset()
    run_repair_hours: int = 6
    start_repair_hours: int = 6
    main_bus_available: np.ndarray | None = None

    def check(self):
        h = self.year.hours
        if self.weight <= 0 or not math.isfinite(self.weight):
            raise ValueError('positive finite scenario weights required')
        for values in (self.grid_fault,self.renewable_bus_fault,self.pcs_available,self.ups_available):
            if values.shape != (h,) or not np.isin(values,[0,1]).all():
                raise ValueError('invalid scenario availability path')
        if self.main_bus_available is not None and (self.main_bus_available.shape!=(h,) or not np.isin(self.main_bus_available,[0,1]).all()):
            raise ValueError('invalid main bus availability')
        expected = day_ahead_weather_risk(self.year.extreme_weather)
        if not np.array_equal(self.year.weather_risk,expected):
            raise ValueError('weather risk must be the declared next-calendar-day Boolean')
        for values in (self.year.core_kw,self.year.rigid_kw,self.year.flex_interruptible_kw,
                       self.year.flex_shiftable_kw,self.year.wind_clean_pu,self.year.pv_pu):
            if values.shape != (h,) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('invalid demand/renewable data')
        if min(self.run_repair_hours,self.start_repair_hours) < 1:
            raise ValueError('repair durations must be positive')


def make_demo_paths(base: SyntheticYear) -> list[MILPPath]:
    """16 explicit, weighted stress branches sharing pre-observation history.

    Two weather alternatives and eight fault alternatives. These are assumed
    stress weights, not calibrated annual event probabilities. No year-long
    deterministic blizzard schedule is fed to a dispatch controller.
    """
    h = base.hours
    if h < 72:
        raise ValueError('the demonstration needs at least 72 hours')
    event = min(h-24, max(36,(h//2//24)*24+12))
    event_day = event//24
    event_types = [('none',.60),('renewable_bus',.10),('grid',.10),('compound',.08),
                   ('grid_pcs',.03),('grid_ups',.03),('diesel_run',.04),('diesel_start',.02)]
    paths=[]
    for weather,weather_weight in ((False,.8),(True,.2)):
        extreme=np.zeros(h,bool)
        if weather:
            extreme[event_day*24:min(h,event_day*24+30)]=True
        year=replace(base,extreme_weather=extreme,weather_risk=day_ahead_weather_risk(extreme),
                     ambient_c=np.where(extreme,-12.,base.ambient_c))
        for kind,probability in event_types:
            grid=np.zeros(h,bool); bus=np.zeros(h,bool); pcs=np.ones(h,bool); ups=np.ones(h,bool); main=np.ones(h,bool)
            stop=min(h,event+(12 if weather else 8))
            if kind in ('grid','compound','grid_pcs','grid_ups','diesel_run','diesel_start'):
                grid[event:stop]=True
            if kind in ('renewable_bus','compound'):
                bus[event:stop]=True
            if kind=='grid': main[event]=False  # one-hour whole-main-bus trip
            if kind=='grid_pcs': pcs[event: min(h,event+4)]=False
            if kind=='grid_ups': ups[event: min(h,event+4)]=False
            run=frozenset({(0,event-1)}) if kind=='diesel_run' else frozenset()
            start=frozenset((0,t) for t in range(event,stop)) if kind=='diesel_start' else frozenset()
            paths.append(MILPPath(f'{"storm" if weather else "normal"}_{kind}',
                                 weather_weight*probability,year,grid,bus,pcs,ups,run,start,main_bus_available=main))
    return paths


def exogenous_information_nodes(paths: list[MILPPath]) -> np.ndarray:
    """Current observations plus past history, excluding all primitive shocks."""
    nodes=np.empty((len(paths),paths[0].year.hours),int)
    previous=np.zeros(len(paths),int)
    for t in range(nodes.shape[1]):
        groups={}
        for s,p in enumerate(paths):
            y=p.year
            key=(int(previous[s]),bool(y.weather_risk[t]),bool(y.extreme_weather[t]),
                 float(y.ambient_c[t]),float(y.wind_clean_pu[t]),float(y.pv_pu[t]),
                 float(y.core_kw[t]),float(y.rigid_kw[t]),float(y.flex_interruptible_kw[t]),
                 float(y.flex_shiftable_kw[t]),bool(p.grid_fault[t]),bool(p.renewable_bus_fault[t]),
                 bool(p.pcs_available[t]),bool(p.ups_available[t]),
                 True if p.main_bus_available is None else bool(p.main_bus_available[t]))
            nodes[s,t]=groups.setdefault(key,len(groups))
        previous=nodes[:,t]
    return nodes


class ResilienceMILP:
    def __init__(self, paths: list[MILPPath], cfg: MILPConfig, economic_budget=None,
                 env: gp.Env | None=None, fixed_modules: dict | None=None, compact_fixed=False):
        started=time.monotonic()
        cfg.check()
        if not paths: raise ValueError('at least one scenario is required')
        for p in paths: p.check()
        if len({p.year.timestamps for p in paths})!=1: raise ValueError('scenario clocks must align')
        if abs(sum(p.weight for p in paths)-1)>1e-8: raise ValueError('scenario weights must sum to one')
        self.cfg,self.paths=cfg,paths
        self.fixed_modules=fixed_modules
        self.h=paths[0].year.hours
        self.budget=cfg.initial_economic_budget_yuan if economic_budget is None else economic_budget
        self.upper={k:int(math.floor(self.budget/cost)) for k,cost in cfg.module_costs.items()}
        self.dmax=self.upper['diesel_units']
        if compact_fixed and fixed_modules is not None and 'diesel_units' in fixed_modules:
            count=fixed_modules['diesel_units']
            if count<0 or int(count)!=count: raise ValueError('invalid fixed diesel count')
            self.dmax=min(self.dmax,int(count))
        self.model=gp.Model('resilience_capacity_and_causal_dispatch',env=env)
        m=self.model
        m.Params.OutputFlag=0
        m.Params.MIPGap=cfg.mip_gap
        m.Params.Threads=cfg.threads
        m.Params.Seed=cfg.seed
        m.Params.FeasibilityTol=1e-7
        m.Params.IntFeasTol=1e-8
        # Preserve an unambiguous INFEASIBLE status for failed stress cases.
        m.Params.DualReductions=0
        self.n={k:m.addVar(lb=0,ub=self.upper[k],vtype=GRB.INTEGER,name='modules_'+k) for k in CAPACITY_STEPS}
        self.cap={k:step*self.n[k] for k,step in CAPACITY_STEPS.items()}
        self.investment=gp.quicksum(cfg.module_costs[k]*self.n[k] for k in self.n)
        m.addConstr(self.investment<=self.budget,name='economic_investment_bound')
        if fixed_modules is not None:
            for k,v in fixed_modules.items():
                if k not in self.n or int(v)!=v or v<0: raise ValueError('invalid fixed module count')
                m.addConstr(self.n[k]==v,name='test_fixed_'+k)
        m.addConstr(self.cap['battery_kwh']>=cfg.storage_min_duration_h*self.cap['pcs_kw'],name='battery_pcs_backing')
        peak=max(float(p.year.core_kw.max()) for p in paths)
        self.required_ups_kwh=peak*cfg.ups_bridge_hours/((cfg.ups_standby_soc-cfg.ups_min_soc)*cfg.ups_efficiency)
        self.required_ups_kw=peak*cfg.ups_power_margin
        m.addConstr(self.cap['ups_kwh']>=self.required_ups_kwh,name='ups_recovery_bridge')
        m.addConstr(self.cap['ups_kw']>=self.required_ups_kw,name='ups_core_power_margin')
        self.installed=m.addVars(self.dmax,vtype=GRB.BINARY,name='diesel_installed')
        m.addConstr(gp.quicksum(self.installed.values())==self.n['diesel_units'])
        for i in range(1,self.dmax): m.addConstr(self.installed[i]<=self.installed[i-1])
        self.blocks=[self._add_path(s,p) for s,p in enumerate(paths)]
        self.nodes=exogenous_information_nodes(paths)
        self.na_pairs=0
        self._add_nonanticipativity()
        self.loss=m.addVars(len(paths),lb=0,name='regular_loss_kwh')
        for s,b in enumerate(self.blocks):
            m.addConstr(self.loss[s]==b['regular_loss_expr'])
        self.eens=gp.quicksum(p.weight*self.loss[s] for s,p in enumerate(paths))
        self.eta=m.addVar(lb=0,name='cvar_eta')
        self.excess=m.addVars(len(paths),lb=0,name='cvar_excess')
        for s in range(len(paths)): m.addConstr(self.excess[s]>=self.loss[s]-self.eta)
        self.cvar=self.eta+gp.quicksum(p.weight*self.excess[s] for s,p in enumerate(paths))/(1-cfg.alpha)
        m.addConstr(self.eens<=cfg.eens_limit_kwh,name='eens_limit')
        m.addConstr(self.cvar<=cfg.cvar_limit_kwh,name='cvar_limit')
        self.operation=gp.quicksum(p.weight*b['operation_expr'] for p,b in zip(paths,self.blocks))
        m.setObjective(self.investment+self.operation,GRB.MINIMIZE)
        m.update()
        if m.NumQConstrs or m.NumQNZs: raise AssertionError('model must remain MILP')
        self.build_seconds=time.monotonic()-started

    def _add_path(self,s,p,parent=None,start=0,stop=None):
        m,c,h,d=self.model,self.cfg,self.h,self.dmax
        stop=h if stop is None else stop
        active=range(start,stop)
        arrivals=range(max(0,start-c.shift_window_hours+1),stop)
        y=p.year; cap=self.cap
        prefix=f's{s}_'
        def hourly(name,ub=GRB.INFINITY,kind=GRB.CONTINUOUS,length=None,lb=0):
            if parent is None:
                return m.addVars(h if length is None else length,lb=lb,ub=ub,vtype=kind,name=prefix+name)
            keys=range(start+1,stop) if length==h+1 else active
            if name=='shift_late_shed':
                keys=[a for a in arrivals if start<=min(h-1,a+c.shift_window_hours-1)<stop]
            return ChainMap(m.addVars(keys,lb=lb,ub=ub,vtype=kind,name=prefix+name),parent[name])
        def unit_vars(name,ub=GRB.INFINITY,kind=GRB.CONTINUOUS,state=False,lb=0):
            times=range(h+int(state)) if parent is None else (range(start+1,stop) if state else active)
            local=m.addVars(d,times,lb=lb,ub=ub,vtype=kind,name=prefix+name)
            return local if parent is None else ChainMap(local,parent[name])
        b={k:hourly(k) for k in ('wind','pv','grid_main','battery_charge','battery_discharge',
             'ups_charge','ups_grid_charge','ups_diesel_charge','ups_discharge','core_main','rigid_shed','interrupt_served','interrupt_adjust',
             'interrupt_shed','shift_emergency_shed','shift_late_shed')}
        b.update({k:hourly(k,1,GRB.BINARY) for k in ('bus_live','battery_ch_on','battery_dis_on','ups_on')})
        b['battery_energy']=hourly('battery_energy',length=h+1)
        b['ups_energy']=hourly('ups_energy',length=h+1)
        for k in ('request','attempt','pending','online','diesel_connected','failed_pre','failed_post','run_fail','start_fail','heater_enabled'):
            b[k]=unit_vars(k,ub=1,kind=GRB.BINARY)
        b['diesel_power']=unit_vars('diesel_power',ub=100)
        b['heater']=unit_vars('heater',ub=c.heater_kw)
        b['temperature']=unit_vars('temperature',state=True,lb=-GRB.INFINITY)
        service_keys=[(a,t) for a in arrivals for t in range(max(a,start),min(stop,a+c.shift_window_hours))]
        service=m.addVars(service_keys,lb=0,name=prefix+'shift_service')
        b['shift_service']=service if parent is None else ChainMap(service,parent['shift_service'])
        for name in ('pre_controls','post_controls','post_states'):
            b[name]=[[] for _ in range(h)] if parent is None else list(parent[name])
            if parent is not None:
                for t in active: b[name][t]=[]
        for t in range(start,stop+1):
            m.addConstr(b['battery_energy'][t]>=c.battery_min_soc*cap['battery_kwh'])
            m.addConstr(b['battery_energy'][t]<=cap['battery_kwh'])
            m.addConstr(b['ups_energy'][t]>=c.ups_min_soc*cap['ups_kwh'])
            m.addConstr(b['ups_energy'][t]<=c.ups_standby_soc*cap['ups_kwh'])
        if parent is None:
            m.addConstr(b['battery_energy'][0]==c.battery_initial_soc*cap['battery_kwh'])
            m.addConstr(b['ups_energy'][0]==c.ups_standby_soc*cap['ups_kwh'])
            m.addConstr(b['battery_energy'][h]>=b['battery_energy'][0],name=prefix+'battery_terminal_inventory')
            m.addConstr(b['ups_energy'][h]>=b['ups_energy'][0],name=prefix+'ups_terminal_inventory')
        for a in arrivals:
            m.addConstr(gp.quicksum(b['shift_service'][a,t] for t in range(a,min(h,a+c.shift_window_hours)))
                        +b['shift_emergency_shed'][a]+b['shift_late_shed'][a]==float(y.flex_shiftable_kw[a]))
            # Late loss is a decision at the deadline, never at task arrival.
            deadline=min(h-1,a+c.shift_window_hours-1)
            if start<=deadline<stop: b['post_controls'][deadline].append(b['shift_late_shed'][a])
        for day in range(start//24*24,stop,24):
            m.addConstr(gp.quicksum(b['interrupt_adjust'][t] for t in range(day,min(h,day+24)))
                <=c.interruptible_daily_energy_fraction*float(y.flex_interruptible_kw[day:day+24].sum()))
        decay=math.exp(-c.thermal_ua_kw_per_k/c.thermal_c_kwh_per_k)
        heat_gain=(1-decay)/c.thermal_ua_kw_per_k if c.thermal_ua_kw_per_k else 1/c.thermal_c_kwh_per_k
        for i in range(d):
            if parent is None: m.addConstr(b['temperature'][i,0]==c.initial_temperature_c)
            for t in active:
                previous=b['online'][i,t-1] if t else 0
                pending=b['pending'][i,t-1] if t else 0
                enabled=b['heater_enabled'][i,t-1] if t else 0
                run_shock=int((i,t-1) in p.run_shocks) if t else 0
                m.addConstr(b['run_fail'][i,t]==run_shock*previous)
                m.addConstr(b['start_fail'][i,t]==int((i,t) in p.start_shocks)*b['attempt'][i,t])
                previous_fail=gp.quicksum(b['run_fail'][i,k] for k in range(max(0,t-p.run_repair_hours+1),t+1))
                previous_fail+=gp.quicksum(b['start_fail'][i,k] for k in range(max(0,t-p.start_repair_hours+1),t))
                m.addConstr(b['failed_pre'][i,t]==previous_fail)
                m.addConstr(b['failed_post'][i,t]==b['failed_pre'][i,t]+b['start_fail'][i,t])
                m.addConstr(b['request'][i,t]<=self.installed[i])
                m.addConstr(b['request'][i,t]<=1-pending)
                m.addConstr(b['request'][i,t]<=1-previous)
                m.addConstr(b['request'][i,t]<=1-b['failed_pre'][i,t])
                m.addConstr(b['request'][i,t]<=int(not y.extreme_weather[t]))
                m.addConstr(b['pending'][i,t]==pending+b['request'][i,t]-b['attempt'][i,t])
                m.addConstr(b['attempt'][i,t]<=pending)
                m.addConstr(b['attempt'][i,t]<=1-b['failed_pre'][i,t])
                # Pending minus recent requests is the mature preparation queue.
                # This recurrence avoids quadratic-size annual history sums.
                eligible=pending-gp.quicksum(b['request'][i,k] for k in range(max(0,t-c.preparation_hours+1),t))
                m.addConstr(b['attempt'][i,t]<=eligible)
                m.addGenConstrIndicator(b['attempt'][i,t],True,b['temperature'][i,t]>=c.ready_temperature_c)
                m.addConstr(b['online'][i,t]<=self.installed[i])
                m.addConstr(b['online'][i,t]<=1-b['failed_post'][i,t])
                success=b['attempt'][i,t]-b['start_fail'][i,t]
                m.addConstr(b['online'][i,t]<=previous+success)
                m.addConstr(b['online'][i,t]>=success)
                for future in range(t,min(h,t+c.minimum_up_hours)):
                    m.addConstr(b['online'][i,future]>=success-gp.quicksum(b['run_fail'][i,k] for k in range(t+1,future+1)))
                m.addConstr(b['diesel_power'][i,t]<=100*b['online'][i,t])
                m.addConstr(b['diesel_power'][i,t]<=100*b['diesel_connected'][i,t])
                m.addConstr(b['diesel_connected'][i,t]<=b['online'][i,t])
                m.addConstr(b['diesel_connected'][i,t]<=b['bus_live'][t])
                m.addConstr(b['diesel_power'][i,t]>=c.diesel_min_output_kw*b['diesel_connected'][i,t])
                m.addConstr(b['heater_enabled'][i,t]>=enabled)
                m.addConstr(b['heater_enabled'][i,t]>=b['request'][i,t])
                m.addConstr(b['heater_enabled'][i,t]<=enabled+b['request'][i,t])
                m.addConstr(b['heater'][i,t]<=c.heater_kw*b['heater_enabled'][i,t])
                m.addConstr(b['heater'][i,t]<=c.heater_kw*b['bus_live'][t])
                thermal=(b['temperature'][i,t+1]==decay*b['temperature'][i,t]+(1-decay)*float(y.ambient_c[t])+heat_gain*b['heater'][i,t])
                if parent is None: m.addConstr(thermal)
                else: m.addGenConstrIndicator(self.installed[i],True,thermal)
                b['pre_controls'][t].extend((b['request'][i,t],b['attempt'][i,t]))
                b['post_controls'][t].extend((b['online'][i,t],b['diesel_connected'][i,t],b['diesel_power'][i,t],b['heater'][i,t]))
                b['post_states'][t].extend((b['temperature'][i,t+1],b['pending'][i,t],b['heater_enabled'][i,t]))
        if parent is not None:
            # Rejoin the annual trajectory with identical discrete memories.
            # Storage and temperature endpoints already share annual variables.
            memory=max(c.preparation_hours,c.minimum_up_hours,p.run_repair_hours,p.start_repair_hours)
            for i in range(d):
                for t in range(max(start,stop-memory),stop):
                    for name in ('request','attempt','online','pending','heater_enabled','run_fail','start_fail'):
                        m.addConstr(b[name][i,t]==parent[name][i,t],name=prefix+f'rejoin_{name}_{i}_{t}')
                for t in range(max(0,start-c.minimum_up_hours+1),start):
                    success=b['attempt'][i,t]-b['start_fail'][i,t]
                    for future in range(start,min(stop,t+c.minimum_up_hours)):
                        m.addConstr(b['online'][i,future]>=success-gp.quicksum(b['run_fail'][i,k] for k in range(t+1,future+1)))
        for t in active:
            m.addConstr(gp.quicksum(b['pending'][i,t] for i in range(d))<=c.crews,name=prefix+f'crew_{t}')
            if p.main_bus_available is not None and not p.main_bus_available[t]:
                m.addConstr(b['bus_live'][t]==0)
            live=b['bus_live'][t]; bd=b['battery_discharge'][t]; bc=b['battery_charge'][t]
            # Each converter mode is backed by actual nonzero current power.
            m.addConstr(bd<=cap['pcs_kw']*int(p.pcs_available[t]))
            m.addConstr(bc<=cap['pcs_kw']*int(p.pcs_available[t]))
            m.addGenConstrIndicator(b['battery_dis_on'][t],False,bd==0)
            m.addGenConstrIndicator(b['battery_ch_on'][t],False,bc==0)
            m.addConstr(bd>=c.gfm_min_power_kw*b['battery_dis_on'][t])
            m.addConstr(bc>=c.gfm_min_power_kw*b['battery_ch_on'][t])
            m.addConstr(b['battery_ch_on'][t]+b['battery_dis_on'][t]<=1)
            m.addConstr(b['battery_ch_on'][t]<=live)
            m.addConstr(b['battery_dis_on'][t]<=live)
            if not c.external_grid_enabled or p.grid_fault[t]:
                m.addConstr(b['grid_main'][t]==0)
                m.addConstr(live<=b['battery_ch_on'][t]+b['battery_dis_on'][t]+gp.quicksum(b['diesel_connected'][i,t] for i in range(d)))
                m.addGenConstrIndicator(b['battery_ch_on'][t],True,
                    b['battery_energy'][t]>=c.battery_min_soc*cap['battery_kwh']+c.gfm_min_power_kw/c.battery_efficiency)
            else:
                if p.main_bus_available is None or p.main_bus_available[t]:
                    m.addConstr(live==1)
                else:
                    m.addConstr(b['grid_main'][t]==0)
            factor=.35 if y.extreme_weather[t] else 1.
            m.addConstr(b['wind'][t]<=cap['wind_kw']*float(y.wind_clean_pu[t])*factor*int(not p.renewable_bus_fault[t]))
            m.addConstr(b['pv'][t]<=cap['pv_kw']*float(y.pv_pu[t])*(.65 if y.extreme_weather[t] else 1.)*int(not p.renewable_bus_fault[t]))
            m.addGenConstrIndicator(live,False,b['wind'][t]==0)
            m.addGenConstrIndicator(live,False,b['pv'][t]==0)
            m.addConstr(b['core_main'][t]+b['ups_discharge'][t]==float(y.core_kw[t]),name=prefix+f'zero_core_loss_{t}')
            m.addConstr(b['core_main'][t]<=float(y.core_kw[t])*live)
            m.addConstr(b['ups_discharge'][t]<=cap['ups_kw']*int(p.ups_available[t]))
            m.addConstr(b['ups_discharge'][t]<=float(y.core_kw[t])*b['ups_on'][t])
            m.addConstr(b['ups_discharge'][t]>=min(c.gfm_min_power_kw,float(y.core_kw[t]))*b['ups_on'][t])
            # Emergency permission must come from an actual observed event.
            # The optimizer may not create permission by choosing to de-energize
            # a healthy bus or by scheduling too little ordinary generation.
            emergency_observed=int(p.grid_fault[t] or p.renewable_bus_fault[t]
                or not p.pcs_available[t] or y.extreme_weather[t]
                or (p.main_bus_available is not None and not p.main_bus_available[t]))
            m.addConstr(b['ups_on'][t]<=emergency_observed
                        +gp.quicksum(b['failed_post'][i,t] for i in range(d)))
            m.addConstr(b['ups_on'][t]<=int(p.ups_available[t]))
            if (c.external_grid_enabled and not p.grid_fault[t]
                    and (p.main_bus_available is None or p.main_bus_available[t])):
                m.addConstr(b['ups_on'][t]==0)
            # Strict core priority: UPS transfer signifies ordinary-load outage.
            m.addConstr(b['rigid_shed'][t]<=float(y.rigid_kw[t]))
            m.addConstr(b['rigid_shed'][t]>=float(y.rigid_kw[t])*(1-live))
            m.addGenConstrIndicator(b['ups_on'][t],True,b['rigid_shed'][t]==float(y.rigid_kw[t]))
            m.addConstr(b['interrupt_served'][t]+b['interrupt_adjust'][t]+b['interrupt_shed'][t]==float(y.flex_interruptible_kw[t]))
            m.addConstr(b['interrupt_served'][t]<=float(y.flex_interruptible_kw[t])*live)
            m.addConstr(b['interrupt_adjust'][t]<=c.interruptible_max_fraction*float(y.flex_interruptible_kw[t])*live)
            m.addGenConstrIndicator(b['ups_on'][t],True,b['interrupt_shed'][t]==float(y.flex_interruptible_kw[t]))
            m.addConstr(b['shift_emergency_shed'][t]==float(y.flex_shiftable_kw[t])*b['ups_on'][t])
            serving=[b['shift_service'][a,t] for a in range(max(0,t-c.shift_window_hours+1),t+1)]
            shift_total=gp.quicksum(serving)
            m.addConstr(shift_total<=c.shift_power_multiplier*float(y.flex_shiftable_kw.max())*live)
            m.addGenConstrIndicator(b['ups_on'][t],True,shift_total==0)
            # Dedicated backup replenishment: only diesel or the optional
            # external source may supply the UPS charger, never wind/PV or
            # the ordinary battery. There is no UPS economic discharge.
            m.addConstr(b['ups_charge'][t]==b['ups_grid_charge'][t]+b['ups_diesel_charge'][t])
            m.addConstr(b['ups_charge'][t]<=cap['ups_kw']*int(p.ups_available[t]))
            m.addGenConstrIndicator(b['ups_on'][t],True,b['ups_charge'][t]==0)
            m.addConstr(c.ups_efficiency*b['ups_charge'][t]
                        <=c.ups_standby_soc*cap['ups_kwh']-b['ups_energy'][t])
            if not c.external_grid_enabled or p.grid_fault[t]:
                m.addConstr(b['ups_grid_charge'][t]==0)
            m.addGenConstrIndicator(live,False,b['ups_diesel_charge'][t]==0)
            m.addConstr(b['ups_diesel_charge'][t]<=gp.quicksum(b['diesel_power'][i,t] for i in range(d)))
            # Inventory may only remain depleted by discharges in the recent
            # restoration window; older use must have been replenished.
            recent=gp.quicksum(b['ups_discharge'][k] for k in range(max(0,t-c.ups_recharge_deadline_hours),t))
            m.addConstr(b['ups_energy'][t]>=c.ups_standby_soc*cap['ups_kwh']-recent/c.ups_efficiency)
            m.addConstr(b['battery_energy'][t+1]==b['battery_energy'][t]+c.battery_efficiency*bc-bd/c.battery_efficiency)
            m.addConstr(b['ups_energy'][t+1]==b['ups_energy'][t]+c.ups_efficiency*b['ups_charge'][t]-b['ups_discharge'][t]/c.ups_efficiency)
            diesel=gp.quicksum(b['diesel_power'][i,t] for i in range(d))
            heaters=gp.quicksum(b['heater'][i,t] for i in range(d))
            m.addConstr(b['wind'][t]+b['pv'][t]+diesel+b['grid_main'][t]+bd
                ==b['core_main'][t]+float(y.rigid_kw[t])-b['rigid_shed'][t]
                  +b['interrupt_served'][t]+shift_total+bc+heaters+b['ups_diesel_charge'][t],name=prefix+f'main_balance_{t}')
            for name in ('wind','pv','grid_main','battery_charge','battery_discharge','ups_charge','ups_grid_charge','ups_diesel_charge','ups_discharge',
                         'core_main','rigid_shed','interrupt_served','interrupt_adjust','interrupt_shed',
                         'shift_emergency_shed','bus_live','battery_ch_on','battery_dis_on','ups_on'):
                b['post_controls'][t].append(b[name][t])
            b['post_controls'][t].extend(serving)
            b['post_states'][t].extend((b['battery_energy'][t+1],b['ups_energy'][t+1]))
        b['active_start'],b['active_stop']=start,stop
        b['regular_loss_expr']=self._loss_between(b,start,stop)
        b['cost_components']=self._cost_between(b,start,stop)
        b['operation_expr']=gp.quicksum(b['cost_components'].values())
        return b

    def _loss_between(self,b,start,stop):
        c,h=self.cfg,self.h
        late=[a for a in range(max(0,start-c.shift_window_hours+1),stop)
              if start<=min(h-1,a+c.shift_window_hours-1)<stop]
        return (gp.quicksum(b[k][t] for k in ('rigid_shed','interrupt_shed','shift_emergency_shed') for t in range(start,stop))
                +gp.quicksum(b['shift_late_shed'][a] for a in late))

    def _cost_between(self,b,start,stop):
        c,d=self.cfg,self.dmax
        active=range(start,stop)
        return {
            'grid':c.grid_yuan_per_kwh*gp.quicksum(b['grid_main'][t]+b['ups_grid_charge'][t] for t in active),
            'diesel':c.diesel_yuan_per_kwh*gp.quicksum(b['diesel_power'][i,t]+c.diesel_idle_equivalent_kw*b['online'][i,t] for i in range(d) for t in active),
            'load_loss':c.loss_yuan_per_kwh*self._loss_between(b,start,stop),
            'flex_adjustment':c.flex_adjustment_yuan_per_kwh*(gp.quicksum(b['interrupt_adjust'][t] for t in active)
                +gp.quicksum(b['shift_service'][a,t] for t in active for a in range(max(0,t-c.shift_window_hours+1),t))),
            'battery_throughput':c.battery_cycle_yuan_per_kwh*gp.quicksum(b['battery_charge'][t]+b['battery_discharge'][t] for t in active),
        }

    def _add_nonanticipativity(self,indices=None,start=0,stop=None):
        m=self.model
        h,d=self.h,self.dmax
        def xor(a,b,name):
            z=m.addVar(vtype=GRB.BINARY,name=name)
            m.addConstr(z>=a-b); m.addConstr(z>=b-a)
            m.addConstr(z<=a+b); m.addConstr(z<=2-a-b)
            return z
        def logical_or(items,name):
            z=m.addVar(vtype=GRB.BINARY,name=name)
            for item in items: m.addConstr(z>=item)
            m.addConstr(z<=gp.quicksum(items))
            return z
        def tie(left,right,distinguished):
            if len(left)!=len(right): raise AssertionError('inconsistent control structure')
            for a,b in zip(left,right):
                if a.sameAs(b): continue
                if isinstance(distinguished,int): m.addConstr(a==b)
                else: m.addGenConstrIndicator(distinguished,False,a==b)
        for s,r in combinations(range(len(self.paths)) if indices is None else indices,2):
            p,q=self.paths[s],self.paths[r]
            bs,br=self.blocks[s],self.blocks[r]
            previous_distinction=0
            dynamic=False
            for t in range(start,h if stop is None else stop):
                if self.nodes[s,t]!=self.nodes[r,t]: break  # observed histories never merge
                self.na_pairs+=1
                # Hidden primitive shocks are NOT observations. They only
                # determine whether endogenous observation branching is needed.
                different_run={i for i,k in p.run_shocks.symmetric_difference(q.run_shocks) if k<t}
                different_start={i for i,k in p.start_shocks.symmetric_difference(q.start_shocks) if k<=t}
                if p.run_repair_hours!=q.run_repair_hours:
                    different_run|={i for i,k in p.run_shocks.union(q.run_shocks) if k<t}
                if p.start_repair_hours!=q.start_repair_hours:
                    different_start|={i for i,k in p.start_shocks.union(q.start_shocks) if k<=t}
                affected=sorted(i for i in different_run|different_start if i<d)
                dynamic=dynamic or bool(affected)
                if not dynamic:
                    tie(bs['pre_controls'][t],br['pre_controls'][t],0)
                    tie(bs['post_controls'][t]+bs['post_states'][t],br['post_controls'][t]+br['post_states'][t],0)
                    continue
                differences=[xor(bs['failed_pre'][i,t],br['failed_pre'][i,t],f'na_pre_delta_{s}_{r}_{i}_{t}') for i in affected]
                pre=logical_or([previous_distinction]+differences,f'na_pre_history_{s}_{r}_{t}')
                tie(bs['pre_controls'][t],br['pre_controls'][t],pre)
                differences=[xor(bs['start_fail'][i,t],br['start_fail'][i,t],f'na_start_delta_{s}_{r}_{i}_{t}') for i in affected]
                post=logical_or([pre]+differences,f'na_post_history_{s}_{r}_{t}')
                tie(bs['post_controls'][t]+bs['post_states'][t],br['post_controls'][t]+br['post_states'][t],post)
                previous_distinction=post

    def optimize(self,time_limit=None,log_file:Path|None=None):
        m=self.model
        limit=self.cfg.time_limit_seconds if time_limit is None else time_limit
        m.Params.TimeLimit=limit if limit>0 else GRB.INFINITY
        if log_file:
            m.Params.OutputFlag=1
            m.Params.LogToConsole=0
            m.Params.LogFile=str(log_file)
        started=time.monotonic()
        m.optimize()
        self.solve_wall_seconds=time.monotonic()-started
        return self.result()

    def values(self):
        result=[]
        hourly=('wind','pv','grid_main','battery_charge','battery_discharge','ups_charge','ups_grid_charge','ups_diesel_charge','ups_discharge',
                'core_main','rigid_shed','interrupt_served','interrupt_adjust','interrupt_shed',
                'shift_emergency_shed','shift_late_shed','bus_live','battery_ch_on','battery_dis_on','ups_on',
                'battery_energy','ups_energy')
        units=('request','attempt','pending','online','diesel_connected','failed_pre','failed_post','run_fail','start_fail',
               'heater_enabled','diesel_power','heater','temperature')
        for b in self.blocks:
            v={k:np.array([x.X for x in b[k].values()]) for k in hourly}
            v.update({k:np.array([[b[k][i,t].X for t in range(self.h+(k=='temperature'))]
                                  for i in range(self.dmax)]).reshape(self.dmax,self.h+(k=='temperature')) for k in units})
            v['shift_service']={(a,t):x.X for (a,t),x in b['shift_service'].items()}
            v['pre_controls']=[np.array([x.X for x in row]) for row in b['pre_controls']]
            v['post_controls']=[np.array([x.X for x in row]) for row in b['post_controls']]
            v['post_states']=[np.array([x.X for x in row]) for row in b['post_states']]
            result.append(v)
        return result

    def audit(self,values):
        """Independent numerical audit of balances, information and actual power."""
        c,h=self.cfg,self.h
        capacity={k:CAPACITY_STEPS[k]*self.n[k].X for k in self.n}
        errors={k:0. for k in ('power_balance','battery_transition','ups_transition','core_loss',
                              'energy_bounds','pcs_coupling','converter_limits','normal_ups_discharge',
                              'ups_charging_rule','gfm_without_power','startup_timing','weather_access',
                              'diesel_failure_exposure','thermal_transition','start_temperature','crew_limit',
                              'shift_energy','nonanticipativity','ups_terminal_reserve','ups_recharge_deadline',
                              'dead_bus_supply','diesel_current_power','flexibility_budget')}
        def put(key,value): errors[key]=max(errors[key],float(abs(value)))
        decay=math.exp(-c.thermal_ua_kw_per_k/c.thermal_c_kwh_per_k)
        gain=(1-decay)/c.thermal_ua_kw_per_k if c.thermal_ua_kw_per_k else 1/c.thermal_c_kwh_per_k
        for p,v in zip(self.paths,values):
            y=p.year
            for t in range(h):
                shifted=sum(value for (a,k),value in v['shift_service'].items() if k==t)
                supply=v['wind'][t]+v['pv'][t]+v['grid_main'][t]+v['diesel_power'][:,t].sum()+v['battery_discharge'][t]
                demand=v['core_main'][t]+y.rigid_kw[t]-v['rigid_shed'][t]+v['interrupt_served'][t]+shifted+v['battery_charge'][t]+v['heater'][:,t].sum()+v['ups_diesel_charge'][t]
                put('power_balance',supply-demand)
                put('core_loss',y.core_kw[t]-v['core_main'][t]-v['ups_discharge'][t])
                put('battery_transition',v['battery_energy'][t+1]-v['battery_energy'][t]-c.battery_efficiency*v['battery_charge'][t]+v['battery_discharge'][t]/c.battery_efficiency)
                put('ups_transition',v['ups_energy'][t+1]-v['ups_energy'][t]-c.ups_efficiency*v['ups_charge'][t]+v['ups_discharge'][t]/c.ups_efficiency)
                emergency_observed=bool(p.grid_fault[t] or p.renewable_bus_fault[t] or not p.pcs_available[t]
                    or y.extreme_weather[t] or v['failed_post'][:,t].sum()>.5
                    or (p.main_bus_available is not None and not p.main_bus_available[t]))
                if not emergency_observed: put('normal_ups_discharge',v['ups_discharge'][t])
                put('ups_charging_rule',v['ups_charge'][t]-v['ups_grid_charge'][t]-v['ups_diesel_charge'][t])
                put('ups_charging_rule',max(0,v['ups_diesel_charge'][t]-v['diesel_power'][:,t].sum()))
                if not c.external_grid_enabled or p.grid_fault[t]:
                    put('ups_charging_rule',v['ups_grid_charge'][t])
                put('ups_charging_rule',max(0,c.ups_efficiency*v['ups_charge'][t]
                    -(c.ups_standby_soc*capacity['ups_kwh']-v['ups_energy'][t])))
                recent=float(v['ups_discharge'][max(0,t-c.ups_recharge_deadline_hours):t].sum())
                put('ups_recharge_deadline',max(0,c.ups_standby_soc*capacity['ups_kwh']
                    -recent/c.ups_efficiency-v['ups_energy'][t]))
                if p.main_bus_available is not None and not p.main_bus_available[t]:
                    put('dead_bus_supply',v['bus_live'][t])
                    put('dead_bus_supply',supply)
                put('flexibility_budget',max(0,v['interrupt_adjust'][t]
                    -c.interruptible_max_fraction*y.flex_interruptible_kw[t]))
                real_power=max(v['battery_charge'][t],v['battery_discharge'][t],float(v['diesel_power'][:,t].max(initial=0)))
                if (not c.external_grid_enabled or p.grid_fault[t]) and v['bus_live'][t]>.5:
                    put('gfm_without_power',max(0,c.gfm_min_power_kw-real_power))
                for name,limit in (('battery_charge',capacity['pcs_kw']*p.pcs_available[t]),
                                   ('battery_discharge',capacity['pcs_kw']*p.pcs_available[t]),
                                   ('ups_charge',capacity['ups_kw']),('ups_discharge',capacity['ups_kw']*p.ups_available[t])):
                    put('converter_limits',max(0,v[name][t]-limit))
                put('converter_limits',min(v['battery_charge'][t],v['battery_discharge'][t]))
                put('converter_limits',min(v['ups_charge'][t],v['ups_discharge'][t]))
                put('crew_limit',max(0,float(v['pending'][:,t].sum())-c.crews))
                for i in range(self.dmax):
                    if v['diesel_connected'][i,t]>.5:
                        put('diesel_current_power',max(0,c.diesel_min_output_kw-v['diesel_power'][i,t]))
                    else:
                        put('diesel_current_power',v['diesel_power'][i,t])
                    if y.extreme_weather[t]: put('weather_access',v['request'][i,t])
                    if v['attempt'][i,t]>.5:
                        eligible=v['request'][i,:max(0,t-c.preparation_hours+1)].sum()-v['attempt'][i,:t].sum()
                        put('startup_timing',max(0,1-eligible))
                        put('start_temperature',max(0,c.ready_temperature_c-v['temperature'][i,t]))
                    expected_run=float((i,t-1) in p.run_shocks)*v['online'][i,t-1] if t else 0
                    put('diesel_failure_exposure',v['run_fail'][i,t]-expected_run)
                    put('diesel_failure_exposure',v['start_fail'][i,t]-float((i,t) in p.start_shocks)*v['attempt'][i,t])
                    put('thermal_transition',v['temperature'][i,t+1]-decay*v['temperature'][i,t]-(1-decay)*y.ambient_c[t]-gain*v['heater'][i,t])
            for a in range(h):
                served=sum(value for (k,t),value in v['shift_service'].items() if k==a)
                put('shift_energy',served+v['shift_emergency_shed'][a]+v['shift_late_shed'][a]-y.flex_shiftable_kw[a])
            for day in range(0,h,24):
                put('flexibility_budget',max(0,float(v['interrupt_adjust'][day:day+24].sum())
                    -c.interruptible_daily_energy_fraction*float(y.flex_interruptible_kw[day:day+24].sum())))
            put('ups_terminal_reserve',v['ups_energy'][h]-v['ups_energy'][0])
            for name,lower,upper in (('battery_energy',c.battery_min_soc*capacity['battery_kwh'],capacity['battery_kwh']),
                                     ('ups_energy',c.ups_min_soc*capacity['ups_kwh'],c.ups_standby_soc*capacity['ups_kwh'])):
                put('energy_bounds',max(0,lower-float(v[name].min()),float(v[name].max())-upper))
        put('pcs_coupling',max(0,c.storage_min_duration_h*capacity['pcs_kw']-capacity['battery_kwh']))
        # Reconstruct actually revealed histories from solution states. Shocks
        # on off/uninstalled diesel units do not distinguish observations.
        parents=[0]*len(self.paths)
        for t in range(h):
            pre_groups={}; post_groups={}; next_parents=[]
            for s,v in enumerate(values):
                pre_key=(parents[s],int(self.nodes[s,t]),tuple(np.rint(v['failed_pre'][:,t]).astype(int)))
                r=pre_groups.setdefault(pre_key,s)
                if s!=r: put('nonanticipativity',np.max(np.abs(v['pre_controls'][t]-values[r]['pre_controls'][t]),initial=0))
                post_key=(pre_key,tuple(np.rint(v['start_fail'][:,t]).astype(int)))
                r=post_groups.setdefault(post_key,s)
                if s!=r:
                    for name in ('post_controls','post_states'):
                        put('nonanticipativity',np.max(np.abs(v[name][t]-values[r][name][t]),initial=0))
                next_parents.append(r)
            parents=next_parents
        return {'passed':max(errors.values(),default=0)<=1e-5,'max_violation':max(errors.values(),default=0),
                'violations':errors,'scope':'hourly algebra and finite-tree information; no frequency/transient proof'}

    def result(self):
        m=self.model
        summary={'solver':'Gurobi','solver_version':list(gp.gurobi.version()),'solver_status':int(m.Status),
                 'synthetic':True,'build_seconds':self.build_seconds,'runtime_seconds':float(m.Runtime),
                 'variables':int(m.NumVars),'binary_variables':int(m.NumBinVars),
                 'integer_variables':int(m.NumIntVars),'linear_constraints':int(m.NumConstrs),
                 'general_linear_constraints':int(m.NumGenConstrs),'quadratic_constraints':int(m.NumQConstrs),
                 'scenario_count':len(self.paths),'hours':self.h,'economic_budget_yuan':self.budget,
                 'time_limit_seconds':self.cfg.time_limit_seconds,'solution_count':int(m.SolCount),
                 'information_contract':'current observations, next-day Boolean, and exposed diesel failures only',
                 'policy_scope':'fixed finite weighted scenario tree; no unseen-history or population certificate',
                 'ups_rule':'core emergency supply only; diesel or optional external-grid backup replenishment; no renewable charging',
                 'cost_basis':'capital investment plus modeled-horizon expected operation; not annualized',
                 'required_ups_kwh':self.required_ups_kwh,'required_ups_kw':self.required_ups_kw,
                 'resolved_config':asdict(self.cfg),'fixed_modules':self.fixed_modules}
        if not m.SolCount:
            summary.update(status='infeasible_within_economic_bound' if m.Status==GRB.INFEASIBLE else 'unresolved_without_incumbent',
                           economic_domain_certified=False,selected=None)
            return summary
        raw={k:float(v.X) for k,v in self.n.items()}
        if max(abs(value-round(value)) for value in raw.values())>1e-5:
            raise RuntimeError('solver returned a nonintegral module value')
        # Integer conversion here only serializes solver INTEGER variables;
        # capacities were never continuous decision variables rounded to fit.
        modules={k:int(round(value)) for k,value in raw.items()}
        selected={k:CAPACITY_STEPS[k]*modules[k] for k in modules}
        values=self.values(); audit=self.audit(values)
        if not audit['passed']: raise RuntimeError(f'MILP physical/information audit failed: {audit}')
        certified=float(m.ObjVal)<=self.budget+1e-6
        scenario_metrics=[]
        for s,(p,b,v) in enumerate(zip(self.paths,self.blocks,values)):
            q={k:float(v[k].sum()) for k in ('rigid_shed','interrupt_shed','shift_emergency_shed','shift_late_shed','interrupt_adjust')}
            q.update(name=p.name,weight=p.weight,regular_loss_kwh=float(self.loss[s].X),core_loss_kwh=0.,
                     ups_emergency_kwh=float(v['ups_discharge'].sum()),
                     ups_companion_loss_kwh=float(sum(v['rigid_shed'][t]+v['interrupt_shed'][t]+v['shift_emergency_shed'][t] for t in range(self.h) if v['ups_on'][t]>.5)),
                     ups_recharge_kwh=float(v['ups_charge'].sum()),
                     ups_diesel_recharge_kwh=float(v['ups_diesel_charge'].sum()),
                     ups_grid_recharge_kwh=float(v['ups_grid_charge'].sum()),
                     diesel_starts=float(v['attempt'].sum()),diesel_run_failures=float(v['run_fail'].sum()),
                     diesel_start_failures=float(v['start_fail'].sum()),
                     grid_fault_hours=int(p.grid_fault.sum()),renewable_bus_fault_hours=int(p.renewable_bus_fault.sum()),
                     cost_components_yuan={k:float(expr.getValue()) for k,expr in b['cost_components'].items()})
            scenario_metrics.append(q)
        losses=[x['regular_loss_kwh'] for x in scenario_metrics]
        cvar=weighted_cvar(losses,[p.weight for p in self.paths],self.cfg.alpha)
        status=('optimal_within_solver_gap_on_scenario_tree' if m.Status==GRB.OPTIMAL else 'feasible_incumbent_not_proven_optimal') if certified else 'economic_bound_requires_expansion'
        if self.fixed_modules is not None:
            status=('fixed_capacity_dispatch_optimal_within_gap' if m.Status==GRB.OPTIMAL
                    else 'fixed_capacity_dispatch_feasible_incumbent')
        summary.update(status=status,economic_domain_certified=certified,selected=selected,modules=modules,
                       objective_yuan=float(m.ObjVal),objective_bound_yuan=float(m.ObjBound),mip_gap=float(m.MIPGap),
                       investment_yuan=float(self.investment.getValue()),expected_operation_yuan=float(self.operation.getValue()),
                       eens_kwh=float(self.eens.getValue()),cvar_kwh=cvar,scenario_metrics=scenario_metrics,audit=audit,
                       mean_cost_components_yuan={k:sum(p.weight*float(b['cost_components'][k].getValue()) for p,b in zip(self.paths,self.blocks))
                                                 for k in self.blocks[0]['cost_components']},
                       post_solution_rounding=False)
        return summary

    def save(self,output:Path,summary):
        output.mkdir(parents=True,exist_ok=True)
        (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (output/'resolved_config.json').write_text(json.dumps(asdict(self.cfg),ensure_ascii=False,indent=2)+'\n')
        (output/'scenario_manifest.json').write_text(json.dumps([
            {'name':p.name,'weight':p.weight,'run_shocks':sorted(p.run_shocks),'start_shocks':sorted(p.start_shocks),
             'run_repair_hours':p.run_repair_hours,'start_repair_hours':p.start_repair_hours}
            for p in self.paths],ensure_ascii=False,indent=2)+'\n')
        if self.model.SolCount:
            arrays={}
            for s,(p,v) in enumerate(zip(self.paths,self.values())):
                for k,value in v.items():
                    if isinstance(value,np.ndarray): arrays[f's{s}_{k}']=value
                arrays[f's{s}_shift_service']=np.array([(a,t,value) for (a,t),value in v['shift_service'].items()])
                for k in ('grid_fault','renewable_bus_fault','pcs_available','ups_available'):
                    arrays[f's{s}_{k}']=getattr(p,k)
                arrays[f's{s}_main_bus_available']=p.main_bus_available if p.main_bus_available is not None else np.ones(self.h,bool)
                for k in ('core_kw','rigid_kw','flex_interruptible_kw','flex_shiftable_kw','ambient_c',
                          'wind_clean_pu','pv_pu','extreme_weather','weather_risk'):
                    arrays[f's{s}_{k}']=getattr(p.year,k)
            np.savez_compressed(output/'dispatch.npz',**arrays)
            self.model.write(str(output/'solution.sol'))
        self.model.write(str(output/'model.lp'))

    def close(self): self.model.dispose()


def weighted_cvar(losses,weights,alpha):
    remaining=1-alpha; tail=0.
    for loss,weight in sorted(zip(losses,weights),reverse=True):
        used=min(remaining,weight);tail+=used*loss;remaining-=used
        if remaining<1e-12:break
    return tail/(1-alpha)


def solve_planning(paths,cfg: MILPConfig,output:Path|None=None,fixed_modules=None):
    """Use Gurobi for the whole MILP, expanding only an uncertified cost bound.

    Positive cost makes any point excluded by investment>B worse than an
    incumbent with total objective<=B. No arbitrary device-number cap remains
    once that inequality is verified. Timeout returns unresolved if it is not.
    """
    started=time.monotonic(); budget=cfg.initial_economic_budget_yuan; rounds=[]
    if output:
        output.mkdir(parents=True,exist_ok=True)
        if (output/'summary.json').exists(): raise ValueError('output already contains results')
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        while True:
            planner=ResilienceMILP(paths,cfg,budget,env,fixed_modules)
            try:
                remaining=max(0.,cfg.time_limit_seconds-(time.monotonic()-started)) if cfg.time_limit_seconds else None
                if remaining is not None and remaining<=0:
                    result={'status':'time_limit_during_model_build','selected':None,'economic_domain_certified':False,
                            'build_seconds':planner.build_seconds,'synthetic':True}
                else:
                    result=planner.optimize(remaining or 0,output/'gurobi.log' if output else None)
                rounds.append({'budget_yuan':budget,'status':result['status'],
                               'build_seconds':planner.build_seconds,'runtime_seconds':result.get('runtime_seconds',0)})
                elapsed=time.monotonic()-started
                done=(result.get('economic_domain_certified',False) or fixed_modules is not None
                      or (cfg.time_limit_seconds and elapsed>=cfg.time_limit_seconds)
                      or planner.model.Status in (GRB.TIME_LIMIT,GRB.INTERRUPTED))
                if done:
                    result['total_elapsed_seconds']=elapsed;result['economic_bound_rounds']=rounds
                    if output: planner.save(output,result)
                    return result
            finally:
                planner.close()
            budget*=2
