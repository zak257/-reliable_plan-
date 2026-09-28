"""Finite-tree adaptation of capacity search and upper/lower certificates.

This is not the old population/DKW problem. Bounds and incumbents refer to the
exact AnnualResilienceMILP. Feasibility monotonicity is used only within a fixed
diesel-count slice; battery/PCS coupling remains enforced in the master.
"""
from dataclasses import replace
import math
import time

import gurobipy as gp
from gurobipy import GRB

from .model import ResilienceMILP, weighted_cvar
from .annual import AnnualResilienceMILP
from ..resilience_v2.model import CAPACITY_STEPS


def safe_lower(model):
    if model.Status in (GRB.NUMERIC,GRB.INF_OR_UNBD,GRB.UNBOUNDED): return 0.
    try: value=float(model.ObjBound)
    except (gp.GurobiError,AttributeError): return 0.
    if not math.isfinite(value) or abs(value)>=GRB.INFINITY: return 0.
    return max(0.,value-max(1e-4,abs(value)*1e-9))


class CapacityMaster(ResilienceMILP):
    """Necessary annual constraints and a rigorous discounted cost lower bound.

    The healthy annual path is present in every original window. Its weighted
    operating cost, plus mandatory main-bus losses on other branches, is a
    lower bound. No non-normal branch is accepted on this relaxation alone.
    """
    def __init__(self,base,windows,cfg,budget,env=None):
        started=time.monotonic()
        super().__init__([replace(base,weight=1.)],cfg,budget,env)
        m=self.model;b=self.blocks[0]
        m.remove(m.getConstrByName('eens_limit'));m.remove(m.getConstrByName('cvar_limit'))
        operation=gp.LinExpr();eens=gp.LinExpr();cvar=gp.LinExpr();cursor=0
        recharge_requires_diesel=False
        for j,w in enumerate(windows):
            a,z=w['start'],w['stop']
            cost=self._cost_between(b,cursor,a)
            operation+=gp.quicksum(cost.values())
            outside=self._loss_between(b,cursor,a);eens+=outside;cvar+=outside
            nominal=next(p for p in w['paths'] if p.name==base.name)
            loss=self._loss_between(b,a,z)
            operation+=nominal.weight*gp.quicksum(self._cost_between(b,a,z).values())
            mandatory=[];weights=[]
            for p in w['paths']:
                q=0.
                for t in range(a,z):
                    if p.main_bus_available is not None and not p.main_bus_available[t] and p.year.core_kw[t]>0:
                        q+=p.year.rigid_kw[t]+p.year.flex_interruptible_kw[t]+p.year.flex_shiftable_kw[t]
                        recharge_requires_diesel=True
                mandatory.append(float(q));weights.append(p.weight)
            constant=sum(q*p for q,p in zip(mandatory,weights))
            operation+=cfg.loss_yuan_per_kwh*constant
            eens+=nominal.weight*loss+constant
            tail=m.addVar(lb=weighted_cvar(mandatory,weights,cfg.alpha),name=f'master_tail_{j}')
            m.addConstr(tail>=min(1.,nominal.weight/(1-cfg.alpha))*loss)
            cvar+=tail;cursor=z
        operation+=gp.quicksum(self._cost_between(b,cursor,self.h).values())
        outside=self._loss_between(b,cursor,self.h);eens+=outside;cvar+=outside
        m.addConstr(eens<=cfg.eens_limit_kwh,name='necessary_annual_eens')
        m.addConstr(cvar<=cfg.cvar_limit_kwh,name='necessary_annual_cvar')
        if recharge_requires_diesel and not cfg.external_grid_enabled:
            # A forced UPS transfer consumes positive energy; restoration is
            # compulsory and diesel is its only permitted source in the island.
            m.addConstr(self.n['diesel_units']>=1,name='necessary_diesel_for_ups_replenishment')
        self.theta=m.addVar(lb=0,name='full_objective_lower_epigraph')
        m.addConstr(self.theta>=self.investment+operation)
        m.setObjective(self.theta,GRB.MINIMIZE)
        m.update();self.build_seconds=time.monotonic()-started
        self.cut_count=0

    def _difference(self,key,value,direction,prefix):
        if direction<0 and value<=0: return None
        if direction>0 and value>=self.upper[key]: return None
        z=self.model.addVar(vtype=GRB.BINARY,name=prefix)
        constraint=self.n[key]<=value-1 if direction<0 else self.n[key]>=value+1
        self.model.addGenConstrIndicator(z,True,constraint)
        return z

    def add_certified_failure(self,point):
        """Exclude dominated capacities ONLY at this same diesel count.

        More wind/PV may be curtailed. Additional battery inventory can be
        translated by initial_soc * delta_capacity; UPS by standby_soc * delta.
        Existing schedules remain feasible whenever the PCS coupling holds.
        Diesel count is excluded from this proof due to thermal recovery rules.
        """
        terms=[];prefix=f'failure_{self.cut_count}'
        for key,value in point.items():
            directions=(-1,1) if key=='diesel_units' else (1,)
            for direction in directions:
                z=self._difference(key,value,direction,f'{prefix}_{key}_{direction}')
                if z is not None: terms.append(z)
        self.model.addConstr(gp.quicksum(terms)>=1,name=prefix)
        self.cut_count+=1

    def add_point_lower(self,point,lower):
        """A fixed-capacity oracle's bound applies only at that exact point."""
        terms=[];prefix=f'point_bound_{self.cut_count}'
        for key,value in point.items():
            for direction in (-1,1):
                z=self._difference(key,value,direction,f'{prefix}_{key}_{direction}')
                if z is not None: terms.append(z)
        same=self.model.addVar(vtype=GRB.BINARY,name=prefix+'_same')
        self.model.addConstr(same+gp.quicksum(terms)>=1)
        self.model.addGenConstrIndicator(same,True,self.theta>=lower)
        self.cut_count+=1


def reference_candidate(base,cfg,budget):
    """A data-derived starting candidate, never an accepted/rounded LP answer."""
    peak=float(base.year.total_kw.max())
    d=math.ceil(peak/100)+1
    p=50*math.ceil((peak+cfg.heater_kw*d)/50)
    t=min(base.year.hours,cfg.preparation_hours*d+6)
    bridge=sum(max(0.,float(base.year.total_kw[k])+cfg.heater_kw*d
                   -100*min(d,k//cfg.preparation_hours)) for k in range(t))
    e=50*math.ceil(1.15*bridge/(cfg.battery_initial_soc-cfg.battery_min_soc)/50)
    core=float(base.year.core_kw.max())
    ue=50*math.ceil(core*cfg.ups_bridge_hours/((cfg.ups_standby_soc-cfg.ups_min_soc)*cfg.ups_efficiency)/50)
    up=50*math.ceil(core*cfg.ups_power_margin/50)
    point={'wind_kw':0,'pv_kw':0,'diesel_units':d,'battery_kwh':math.ceil(max(e,p*cfg.storage_min_duration_h)/50),
           'pcs_kw':int(p/50),'ups_kwh':int(ue/50),'ups_kw':int(up/50)}
    investment=sum(cfg.module_costs[k]*v for k,v in point.items())
    return point if investment<=budget else None


def run_capacity_search(base,windows,cfg,output,env,started,event,configure,progress):
    """Global lower bounds + exact fixed-capacity joint MILP upper witnesses.

    Timeouts never generate a failure cut. Repeated unresolved points receive
    progressively larger oracle budgets; all time, including builds, is charged.
    """
    def remaining(): return max(0.,cfg.time_limit_seconds-(time.monotonic()-started))
    budget=cfg.initial_economic_budget_yuan
    master=None;incumbent=None;upper=math.inf;lower=0.;total_runtime=0.;build_time=0.
    calls=0;iterations=0;seen={};history=[];seed=reference_candidate(base,cfg,budget)
    first_feasible=None;stop='time_limit';certified=False
    try:
        while remaining()>1:
            if seed is not None:
                point=seed;seed=None;origin='reference_candidate'
            else:
                if master is None:
                    master=CapacityMaster(base,windows,cfg,budget,env)
                    build_time+=master.build_seconds
                    configure(master.model,output/'master.log')
                    event('master_built',variables=master.model.NumVars,budget_yuan=budget)
                if remaining()<=1: break
                iterations+=1
                master.model.Params.TimeLimit=min(1200.*(1+(iterations-1)//3),remaining())
                master.model.optimize(progress('master'))
                total_runtime+=master.model.Runtime
                lower=max(lower,min(budget,safe_lower(master.model)))
                if master.model.Status==GRB.INFEASIBLE:
                    # Only the bounded region was exhausted; outside points
                    # have objective >= budget by nonnegative operating cost.
                    lower=max(lower,budget)
                    if upper<=budget:
                        raise RuntimeError('Master infeasibility contradicts audited incumbent')
                    budget*=2;master.close();master=None
                    event('economic_region_expanded',budget_yuan=budget)
                    continue
                if upper<=budget and upper-lower<=cfg.mip_gap*abs(upper):
                    certified=True;stop='gap_reached';break
                if not master.model.SolCount:
                    event('master_unresolved',lower_yuan=lower);continue
                raw={k:v.X for k,v in master.n.items()}
                if any(abs(v-round(v))>1e-5 for v in raw.values()): raise RuntimeError('noninteger master candidate')
                point={k:int(round(v)) for k,v in raw.items()};origin='master'
            key=tuple(point[k] for k in CAPACITY_STEPS)
            visit=seen.get(key,0);seen[key]=visit+1
            calls+=1
            event('candidate',call=calls,origin=origin,modules=point,visit=visit+1,lower_yuan=lower)
            oracle=AnnualResilienceMILP(base,windows,cfg,budget,env,point,compact_fixed=True)
            build_time+=oracle.build_seconds
            try:
                if remaining()<=1: break
                configure(oracle.model,output/f'oracle_{calls:04d}.log')
                oracle.model.Params.TimeLimit=min(900.*2**min(visit,5),remaining())
                oracle.model.optimize(progress('oracle'))
                total_runtime+=oracle.model.Runtime
                lo=safe_lower(oracle.model)
                row={'call':calls,'modules':point,'solver_status':int(oracle.model.Status),
                     'runtime_seconds':oracle.model.Runtime,'lower_yuan':lo,'solution_count':oracle.model.SolCount}
                if oracle.model.SolCount:
                    result=oracle.result()  # Full original physics/information audit.
                    row['upper_yuan']=result['objective_yuan']
                    if result['objective_yuan']<upper:
                        upper=result['objective_yuan'];incumbent=result
                        if first_feasible is None: first_feasible=time.monotonic()-started
                        oracle.save(output/'incumbent',result,write_model=False)
                        event('audited_incumbent',objective_yuan=upper,modules=point,first_feasible_seconds=first_feasible)
                history.append(row)
                if master is None and remaining()>1:
                    master=CapacityMaster(base,windows,cfg,budget,env)
                    build_time+=master.build_seconds
                    configure(master.model,output/'master.log')
                if master is not None:
                    if oracle.model.Status==GRB.INFEASIBLE:
                        master.add_certified_failure(point)
                        event('same_diesel_monotone_failure_cut',modules=point)
                    elif lo>0:
                        master.add_point_lower(point,lo)
                        event('point_cost_lower_bound',modules=point,lower_yuan=lo)
                # A bound for a single candidate is NEVER used as a global LB.
                if upper<=budget and upper-lower<=cfg.mip_gap*abs(upper):
                    certified=True;stop='gap_reached';break
            finally: oracle.close()
    finally:
        if master is not None: master.close()
    result=dict(incumbent) if incumbent is not None else {'selected':None,'modules':None}
    result.update(method='capacity_search_bounds_joint_milp_adaptation',
        status='optimal_within_gap' if certified else ('time_limit_with_audited_incumbent' if incumbent else 'time_limit_without_incumbent'),
        stop_reason=stop,fixed_modules=None,objective_yuan=upper if math.isfinite(upper) else None,
        objective_bound_yuan=lower,mip_gap=(upper-lower)/abs(upper) if math.isfinite(upper) and upper else None,
        economic_domain_certified=upper<=budget,economic_budget_yuan=budget,
        total_elapsed_seconds=time.monotonic()-started,runtime_seconds=total_runtime,build_seconds=build_time,
        first_feasible_seconds=first_feasible,oracle_calls=calls,master_calls=iterations,candidate_history=history,
        monotonicity_scope='six non-diesel capacity coordinates within the same diesel-count slice and valid PCS/battery coupling',
        certificate_scope='fixed weighted annual scenario tree; no population/DKW confidence claim')
    return result
