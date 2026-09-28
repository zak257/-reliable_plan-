"""Preserve original physics/information adversarial tests under compression."""
from dataclasses import replace
from functools import partial
from unittest.mock import patch
import unittest

import numpy as np
from gurobipy import GRB

from polar_reliability_planning.resilience_milp.model import ResilienceMILP, MILPConfig
from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP, annual_scenarios
from polar_reliability_planning.resilience_milp.acceleration import (
    merge_duplicate_paths, path_fingerprint, seed_modules, fix_seed_policy, apply_start,
)
from resilience_tests import test_resilience_milp as physical
from resilience_tests import test_resilience_annual as annual


class CompactPhysicsTests(physical.ResilienceMILPTests):
    def build(self,paths,cfg=None,modules=None):
        model=ResilienceMILP(paths,cfg or self.config(),env=self.env,fixed_modules=modules,compact=True)
        self.addCleanup(model.close)
        return model


class CompactAnnualTests(annual.AnnualMILPTests):
    def build(self,*args,**kwargs):
        with patch.object(annual,'AnnualResilienceMILP',partial(AnnualResilienceMILP,compact=True)):
            return super().build(*args,**kwargs)

    def test_merge_preserves_weights_and_hidden_shocks(self):
        base,windows=annual_scenarios(physical.year_case(168),starts=(24,))
        merged,aliases=merge_duplicate_paths(windows)
        self.assertEqual(len(merged[0]['paths']),14)
        self.assertAlmostEqual(sum(p.weight for p in merged[0]['paths']),1.)
        storm=next(p for p in merged[0]['paths'] if p.name.endswith('storm_renewable_bus'))
        self.assertAlmostEqual(storm.weight,.036)
        self.assertEqual(sum(len(a['members']) for a in aliases),16)
        self.assertNotEqual(path_fingerprint(base),path_fingerprint(replace(base,start_shocks=frozenset({(0,60)}))))
        self.assertNotEqual(path_fingerprint(base),path_fingerprint(replace(base,run_repair_hours=13)))

    def test_seed_is_audited_with_all_information_and_boundary_constraints(self):
        base,windows=annual_scenarios(physical.year_case(168,wind=.5),starts=(24,))
        cfg=replace(MILPConfig(),ups_bridge_hours=12,time_limit_seconds=60)
        windows,aliases=merge_duplicate_paths(windows)
        point=seed_modules(base,cfg)
        m=AnnualResilienceMILP(base,windows,cfg,env=self.env,fixed_modules=point,compact_fixed=True,compact=True)
        self.addCleanup(m.close)
        fix_seed_policy(m)
        m.model.Params.TimeLimit=60;m.model.optimize()
        self.assertGreater(m.model.SolCount,0)
        result=m.result()
        self.assertTrue(result['audit']['passed'])
        self.assertAlmostEqual(result['eens_kwh'],.1*30,places=5)
        self.assertEqual(result['post_solution_rounding'],False)
        joint=AnnualResilienceMILP(base,windows,cfg,env=self.env,compact=True)
        self.addCleanup(joint.close)
        transfer=apply_start(joint,{v.VarName:v.X for v in m.model.getVars()},point['diesel_units'])
        self.assertEqual(transfer['missing_count'],0)
        self.assertIsNone(joint.fixed_modules)
        self.assertTrue(all(v.LB==0 and v.UB>point[k] for k,v in joint.n.items()))
        joint.model.Params.NodeLimit=0;joint.model.Params.TimeLimit=30;joint.model.optimize()
        self.assertGreater(joint.model.SolCount,0)
        self.assertTrue(joint.result()['audit']['passed'])

    def test_compact_and_original_have_equal_optima(self):
        y=physical.year_case(96,inter=1,shift=2,wind=.2)
        base,windows=annual_scenarios(y,starts=(0,))
        cfg=replace(MILPConfig(),external_grid_enabled=True,ups_bridge_hours=0,
                    initial_economic_budget_yuan=250000.,initial_temperature_c=5.,mip_gap=0.)
        point=dict(wind_kw=0,pv_kw=0,diesel_units=0,battery_kwh=2,pcs_kw=1,ups_kwh=2,ups_kw=1)
        objectives=[];sizes=[]
        for compact in (False,True):
            w=merge_duplicate_paths(windows)[0] if compact else windows
            m=AnnualResilienceMILP(base,w,cfg,env=self.env,fixed_modules=point,compact=compact)
            self.addCleanup(m.close)
            m.model.Params.TimeLimit=60;m.model.optimize()
            self.assertEqual(m.model.Status,GRB.OPTIMAL)
            self.assertTrue(m.result()['audit']['passed'])
            objectives.append(m.model.ObjVal);sizes.append(m.model.NumConstrs)
        self.assertAlmostEqual(*objectives,places=5)
        self.assertLess(sizes[1],sizes[0])


if __name__=='__main__': unittest.main()
