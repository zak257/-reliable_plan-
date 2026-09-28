"""Exact-model equivalence and safe cut/bound tests for the comparison method."""
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

import gurobipy as gp
from gurobipy import GRB

from resilience_tests.test_resilience_milp import year_case,path_case
from polar_reliability_planning.resilience_milp.model import MILPConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP,annual_scenarios
from polar_reliability_planning.resilience_milp.capacity_certificate import CapacityMaster,run_capacity_search
from polar_reliability_planning.resilience_v2.model import CAPACITY_STEPS


class CapacityCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=gp.Env(empty=True);cls.env.setParam('OutputFlag',0);cls.env.start()

    @classmethod
    def tearDownClass(cls): cls.env.dispose()

    def instance(self):
        y=year_case(120,core=10,rigid=20,inter=0,shift=0)
        base=path_case(y,'annual_normal')
        fault=path_case(y,'main_bus',.2,main_trip=(60,61),grid_fault=(60,61))
        windows=[{'start':24,'stop':96,'paths':[replace(base,weight=.8),fault]}]
        cfg=replace(MILPConfig(),external_grid_enabled=True,initial_economic_budget_yuan=300000.,
                    time_limit_seconds=30.,mip_gap=.001,threads=1)
        return base,windows,cfg

    def build(self,base,windows,cfg,**kw):
        obj=AnnualResilienceMILP(base,windows,cfg,env=self.env,**kw)
        self.addCleanup(obj.close);return obj

    def test_compact_fixed_oracle_matches_original_model(self):
        b,w,c=self.instance();c=replace(c,mip_gap=0.)
        point={'wind_kw':0,'pv_kw':0,'diesel_units':0,'battery_kwh':0,'pcs_kw':0,'ups_kwh':2,'ups_kw':1}
        full=self.build(b,w,c,fixed_modules=point)
        compact=self.build(b,w,c,fixed_modules=point,compact_fixed=True)
        a=full.optimize();z=compact.optimize()
        self.assertTrue(a['audit']['passed']);self.assertTrue(z['audit']['passed'])
        self.assertAlmostEqual(a['objective_yuan'],z['objective_yuan'],places=5)
        self.assertAlmostEqual(a['eens_kwh'],z['eens_kwh'],places=5)
        self.assertAlmostEqual(a['cvar_upper_bound_kwh'],z['cvar_upper_bound_kwh'],places=5)
        self.assertLess(compact.model.NumVars,full.model.NumVars)
        self.assertEqual(compact.dmax,0)

    def test_master_is_lower_bound_on_same_full_joint_problem(self):
        b,w,c=self.instance();c=replace(c,mip_gap=0.)
        full=self.build(b,w,c);r=full.optimize()
        master=CapacityMaster(b,w,c,c.initial_economic_budget_yuan,self.env)
        self.addCleanup(master.close);master.model.optimize()
        self.assertEqual(master.model.Status,GRB.OPTIMAL)
        self.assertLessEqual(master.model.ObjVal,r['objective_yuan']+1e-6)
        self.assertEqual(full.model.Status,GRB.OPTIMAL)

    def bare_master(self):
        obj=object.__new__(CapacityMaster)
        obj.model=gp.Model(env=self.env);obj.model.Params.OutputFlag=0
        obj.upper={k:3 for k in CAPACITY_STEPS}
        obj.n={k:obj.model.addVar(vtype=GRB.INTEGER,lb=0,ub=3,name=k) for k in CAPACITY_STEPS}
        obj.theta=obj.model.addVar(lb=0);obj.model.setObjective(obj.theta)
        obj.cut_count=0;self.addCleanup(obj.close)
        return obj

    def test_failure_cut_propagates_only_within_same_diesel_count(self):
        bad={k:1 for k in CAPACITY_STEPS}
        cases=[(0,0,True),(1,0,False),(1,2,True),(2,0,True)]
        for diesel,wind,feasible in cases:
            m=self.bare_master();m.add_certified_failure(bad)
            point={k:0 for k in CAPACITY_STEPS};point.update(diesel_units=diesel,wind_kw=wind)
            for k,v in point.items():m.model.addConstr(m.n[k]==v)
            m.model.optimize()
            self.assertEqual(m.model.Status,GRB.OPTIMAL if feasible else GRB.INFEASIBLE)

    def test_point_cost_bound_does_not_cut_other_capacities(self):
        point={k:1 for k in CAPACITY_STEPS}
        for change,expected in [(0,123.),(1,0.)]:
            m=self.bare_master();m.add_point_lower(point,123.)
            q=dict(point);q['wind_kw']+=change
            for k,v in q.items():m.model.addConstr(m.n[k]==v)
            m.model.optimize()
            self.assertEqual(m.model.Status,GRB.OPTIMAL)
            self.assertAlmostEqual(m.model.ObjVal,expected)

    def test_capacity_search_matches_direct_joint_optimum(self):
        b,w,c=self.instance()
        full=self.build(b,w,replace(c,mip_gap=0.));reference=full.optimize()
        with tempfile.TemporaryDirectory() as folder:
            events=[]
            def event(action,**kw): events.append({'action':action,**kw})
            result=run_capacity_search(b,w,c,Path(folder),self.env,time.monotonic(),event,
                                      lambda m,p:None,lambda stage:None)
        self.assertTrue(result['selected'])
        self.assertEqual(result['status'],'optimal_within_gap')
        self.assertTrue(result['audit']['passed'])
        self.assertLessEqual(result['objective_bound_yuan'],reference['objective_yuan']+1e-5)
        self.assertGreaterEqual(result['objective_yuan'],reference['objective_yuan']-1e-5)
        self.assertLessEqual(result['objective_yuan']-reference['objective_yuan'],c.mip_gap*result['objective_yuan'])
        self.assertTrue(any(e['action']=='point_cost_lower_bound' for e in events))

    def test_six_coordinate_monotone_schedule_translation(self):
        b,w,c=self.instance();c=replace(c,initial_economic_budget_yuan=800000.)
        small={'wind_kw':0,'pv_kw':0,'diesel_units':0,'battery_kwh':2,'pcs_kw':1,'ups_kwh':2,'ups_kw':1}
        large={'wind_kw':1,'pv_kw':1,'diesel_units':0,'battery_kwh':4,'pcs_kw':2,'ups_kwh':3,'ups_kw':2}
        a=self.build(b,w,c,fixed_modules=small,compact_fixed=True)
        a.model.addConstr(a.blocks[0]['battery_charge'][20]==10.)
        a.optimize()
        z=self.build(b,w,c,fixed_modules=large,compact_fixed=True)
        old={v.VarName:v.X for v in a.model.getVars()}
        for v in z.model.getVars():
            if v.VarName.startswith('modules_'):continue
            value=old[v.VarName]
            if '_battery_energy[' in v.VarName: value+=c.battery_initial_soc*100
            if '_ups_energy[' in v.VarName: value+=c.ups_standby_soc*50
            z.model.addConstr(v==value)
        r=z.optimize()
        self.assertEqual(z.model.Status,GRB.OPTIMAL)
        self.assertTrue(r['audit']['passed'])

    def test_compact_16_branch_witness_is_feasible_in_uncompressed_model(self):
        y=year_case(120,core=10,rigid=20,inter=5,shift=5,wind=1.)
        b,w=annual_scenarios(y,starts=(24,))
        c=replace(MILPConfig(),initial_economic_budget_yuan=900000.,time_limit_seconds=30.,initial_temperature_c=5.)
        point={'wind_kw':1,'pv_kw':0,'diesel_units':1,'battery_kwh':10,'pcs_kw':1,'ups_kwh':2,'ups_kw':1}
        compact=self.build(b,w,c,fixed_modules=point,compact_fixed=True)
        first=compact.optimize()
        self.assertTrue(first['audit']['passed'])
        full=self.build(b,w,c,fixed_modules=point)
        source={v.VarName:v.X for v in compact.model.getVars()}
        for v in full.model.getVars():
            if v.VarName in source: full.model.addConstr(v==source[v.VarName])
        second=full.optimize()
        self.assertTrue(second['audit']['passed'])
        self.assertAlmostEqual(first['objective_yuan'],second['objective_yuan'],places=5)


if __name__=='__main__': unittest.main()
