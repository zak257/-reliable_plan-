"""Saved annual policies can improve without fixing the final planning model."""
from dataclasses import replace
import unittest

import gurobipy as gp
from gurobipy import GRB

from polar_reliability_planning.resilience_milp.model import MILPConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP, annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import (
    seed_modules, fix_seed_policy, merge_duplicate_paths, apply_start,
)
from polar_reliability_planning.resilience_milp.continuation import (
    fix_incumbent_commitment, temporarily_fix_battery_modes, restore_bounds, saved_binary,
)
from resilience_tests.test_resilience_milp import year_case


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=gp.Env(empty=True)
        cls.env.setParam('OutputFlag',0)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def build(self, base, windows, cfg, **kwargs):
        planner=AnnualResilienceMILP(base,windows,cfg,env=self.env,compact=True,**kwargs)
        self.addCleanup(planner.close)
        planner.model.Params.TimeLimit=30
        planner.model.Params.Aggregate=0
        planner.model.Params.NumericFocus=3
        return planner

    def test_improvement_then_full_start_preserves_audit_and_free_capacities(self):
        base,windows=annual_scenarios(year_case(168,wind=.5),starts=(24,))
        windows,_=merge_duplicate_paths(windows)
        cfg=replace(MILPConfig(),ups_bridge_hours=12,threads=2)
        point=seed_modules(base,cfg)
        initial=self.build(base,windows,cfg,fixed_modules=point,compact_fixed=True)
        fix_seed_policy(initial)
        initial.model.optimize()
        first=initial.result()
        self.assertTrue(first.get('audit',{}).get('passed'),first)
        values={v.VarName:v.X for v in initial.model.getVars()}
        polish=self.build(base,windows,cfg,fixed_modules={'diesel_units':point['diesel_units']},compact_fixed=True)
        self.assertEqual(apply_start(polish,values,point['diesel_units'])['missing_count'],0)
        fix_incumbent_commitment(polish,values)
        for key,var in polish.n.items():
            self.assertEqual(var.VType,GRB.INTEGER)
            self.assertLess(var.LB,var.UB)
            if key!='diesel_units':
                self.assertIsNone(polish.model.getConstrByName('test_fixed_'+key))
        bounds=temporarily_fix_battery_modes(polish,values)
        polish.model.optimize()
        second=polish.result()
        self.assertTrue(second['audit']['passed'])
        self.assertLessEqual(second['objective_yuan'],first['objective_yuan']+1e-5)
        self.assertTrue(bounds)
        restore_bounds(polish.model,bounds)
        self.assertTrue(all(v.LB==lo and v.UB==hi for v,lo,hi in bounds))
        # An ordinary hour after the old startup period must now be free.
        self.assertEqual(polish.blocks[0]['battery_ch_on'][80].UB,1)
        self.assertLess(polish.blocks[0]['battery_ch_on'][80].LB,1)
        polish.model.Params.Method=3
        polish.model.optimize()
        self.assertTrue(polish.result()['audit']['passed'])
        improved={v.VarName:v.X for v in polish.model.getVars()}
        joint=self.build(base,windows,cfg)
        self.assertEqual(apply_start(joint,improved,point['diesel_units'])['missing_count'],0)
        self.assertIsNone(joint.fixed_modules)
        self.assertTrue(all(v.VType==GRB.INTEGER and v.LB<v.UB for v in joint.n.values()))
        joint.model.Params.NodeLimit=0
        joint.model.optimize()
        self.assertTrue(joint.result()['audit']['passed'])

    def test_saved_binary_rejects_missing_and_fractional_values(self):
        model=gp.Model(env=self.env)
        self.addCleanup(model.dispose)
        var=model.addVar(vtype=GRB.BINARY,name='policy')
        model.update()
        with self.assertRaises(ValueError): saved_binary(var,{})
        with self.assertRaises(ValueError): saved_binary(var,{'policy':.2})


if __name__=='__main__':
    unittest.main()
