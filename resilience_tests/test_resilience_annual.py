"""Annual/window accounting, inherited inventories, queues and information."""
from dataclasses import replace
import unittest
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP, annual_scenarios
from polar_reliability_planning.resilience_milp.model import MILPConfig, FaultRecoveryConfig, exogenous_information_nodes
from resilience_tests.test_resilience_milp import year_case,path_case


class AnnualMILPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=gp.Env(empty=True);cls.env.setParam('OutputFlag',0);cls.env.start()

    @classmethod
    def tearDownClass(cls): cls.env.dispose()

    def build(self,h=144,starts=(24,),shift=0.,battery=False):
        y=year_case(h,core=10,rigid=20,inter=0,shift=shift)
        base=path_case(y,name='annual_normal')
        windows=[]
        for j,start in enumerate(starts):
            event=start+36
            fault=path_case(y,name=f'fault_{j}',weight=.2,main_trip=(event,event+1),grid_fault=(event,event+1))
            windows.append({'start':start,'stop':start+72,'paths':[replace(base,weight=.8),fault]})
        cfg=replace(MILPConfig(),external_grid_enabled=True,ups_bridge_hours=0.,
                    initial_economic_budget_yuan=250000.,initial_temperature_c=5.,
                    mip_gap=0.,time_limit_seconds=30.)
        modules={'wind_kw':0,'pv_kw':0,'diesel_units':0,'battery_kwh':2 if battery else 0,
                 'pcs_kw':1 if battery else 0,'ups_kwh':2,'ups_kw':1}
        planner=AnnualResilienceMILP(base,windows,cfg,env=self.env,fixed_modules=modules)
        self.addCleanup(planner.close)
        return planner

    def solve(self,planner):
        result=planner.optimize()
        self.assertEqual(planner.model.Status,GRB.OPTIMAL)
        self.assertTrue(result['audit']['passed'])
        return result

    def test_full_year_recipe_and_day_ahead_release(self):
        year=year_case(8760)
        base,windows=annual_scenarios(year)
        self.assertEqual(base.year.hours,8760)
        self.assertEqual(len(windows),4)
        self.assertEqual(sum(len(w['paths']) for w in windows),64)
        for w in windows:
            self.assertEqual(w['stop']-w['start'],72)
            self.assertAlmostEqual(sum(p.weight for p in w['paths']),1)
            p=next(p for p in w['paths'] if p.name.endswith('storm_compound'))
            a=w['start']
            self.assertEqual(p.renewable_bus_fault.sum(),12)
            self.assertTrue(p.year.weather_risk[a])
            self.assertFalse(p.year.weather_risk[a-1])
            self.assertTrue(p.year.extreme_weather[a+24])
            self.assertFalse(p.grid_fault.any())

    def test_week_repairs_last_twelve_hours_and_main_bus_one_hour(self):
        base,windows=annual_scenarios(year_case(168),starts=(24,))
        paths=windows[0]['paths'];event=60
        self.assertEqual(len(paths),16)
        for p in paths:
            self.assertEqual(p.run_repair_hours,12)
            self.assertEqual(p.start_repair_hours,12)
            for suffix,flag in [('renewable_bus',p.renewable_bus_fault),('compound',p.renewable_bus_fault),
                                ('pcs',~p.pcs_available),('ups',~p.ups_available)]:
                if p.name.endswith('_'+suffix):
                    np.testing.assert_array_equal(np.flatnonzero(flag),np.arange(event,event+12))
            if p.name.endswith('_main_bus'):
                np.testing.assert_array_equal(np.flatnonzero(~p.main_bus_available),[event])
            if p.name.endswith('_diesel_start'):
                self.assertEqual(p.start_shocks,frozenset((0,t) for t in range(event,event+12)))
        storm=next(p for p in paths if p.name.endswith('storm_compound'))
        no_fault=next(p for p in paths if p.name.endswith('storm_none'))
        nodes=exogenous_information_nodes([storm,no_fault])
        np.testing.assert_array_equal(nodes[0,:event],nodes[1,:event])
        self.assertNotEqual(nodes[0,event],nodes[1,event])
        self.assertFalse(storm.year.weather_risk[23])
        self.assertTrue(storm.year.weather_risk[24])

    def test_recovery_configuration_is_explicit_and_not_silently_truncated(self):
        recovery=FaultRecoveryConfig(renewable_bus_normal_hours=8,renewable_bus_storm_hours=12,
            main_bus_hours=2,pcs_hours=5,ups_hours=7,diesel_run_hours=6,diesel_start_hours=9)
        base,windows=annual_scenarios(year_case(168),starts=(24,),recovery=recovery)
        self.assertEqual(base.run_repair_hours,6)
        self.assertEqual(base.start_repair_hours,9)
        for p in windows[0]['paths']:
            if p.name.endswith('normal_renewable_bus'): self.assertEqual(p.renewable_bus_fault.sum(),8)
            if p.name.endswith('_pcs'): self.assertEqual((~p.pcs_available).sum(),5)
            if p.name.endswith('_ups'): self.assertEqual((~p.ups_available).sum(),7)
            if p.name.endswith('_main_bus'): self.assertEqual((~p.main_bus_available).sum(),2)
        with self.assertRaises(ValueError):
            annual_scenarios(year_case(168),starts=(24,),recovery=replace(recovery,pcs_hours=37))
        with self.assertRaises(ValueError):
            replace(recovery,diesel_run_hours=1.5).check()

    def test_annual_costs_replace_window_instead_of_double_counting(self):
        m=self.build();r=self.solve(m)
        expected_grid=.65*(144*30+.2*(-30+10/.98**2))
        self.assertAlmostEqual(r['mean_cost_components_yuan']['grid'],expected_grid,places=5)
        self.assertAlmostEqual(r['mean_cost_components_yuan']['load_loss'],.2*20*1000,places=5)
        self.assertAlmostEqual(r['eens_kwh'],4.)
        self.assertAlmostEqual(r['cvar_upper_bound_kwh'],20.)
        self.assertAlmostEqual(r['objective_yuan'],r['investment_yuan']+r['expected_operation_yuan'])
        self.assertEqual(r['hours'],144)
        self.assertIn('144 hours',r['cost_basis'])
        self.assertEqual(r['model_scope'],'chronological_with_linked_resilience_windows')

    def test_windows_inherit_actual_annual_inventory_and_restore_it(self):
        m=self.build(battery=True);a,z=24,96
        base,branch=m.blocks
        m.model.addConstr(base['battery_energy'][a]==20.)
        r=self.solve(m)
        self.assertAlmostEqual(branch['battery_energy'][a].X,20.)
        self.assertTrue(branch['battery_energy'][a].sameAs(base['battery_energy'][a]))
        self.assertTrue(branch['battery_energy'][z].sameAs(base['battery_energy'][z]))
        self.assertTrue(branch['online'][0,a-1].sameAs(base['online'][0,a-1]))
        self.assertTrue(branch['temperature'][0,a].sameAs(base['temperature'][0,a]))
        m.model.addConstr(branch['battery_energy'][a]>=80.)
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_unfinished_shift_jobs_cannot_disappear_at_window_start(self):
        m=self.build(shift=5.);b=m.blocks[1]
        # Job arriving just before the window must still get its 5 kWh.
        m.model.addConstr(m.blocks[0]['shift_service'][23,23]==0.)
        self.solve(m)
        served=sum(b['shift_service'][23,t].X for t in range(24,47))
        self.assertAlmostEqual(served,5.)
        m.model.addConstr(gp.quicksum(b['shift_service'][23,t] for t in range(24,47))==0.)
        m.model.addConstr(b['shift_late_shed'][23]==0.)
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_shift_jobs_crossing_window_end_share_the_annual_future(self):
        m=self.build(shift=5.)
        base,branch=m.blocks
        self.assertTrue(branch['shift_service'][95,96].sameAs(base['shift_service'][95,96]))
        self.assertTrue(branch['shift_late_shed'][95].sameAs(base['shift_late_shed'][95]))
        self.assertFalse(branch['shift_service'][95,95].sameAs(base['shift_service'][95,95]))
        self.solve(m)

    def test_fault_does_not_allow_different_pre_fault_controls(self):
        m=self.build(battery=True)
        m.model.addConstr(m.blocks[0]['battery_discharge'][59]==0.)
        m.model.addConstr(m.blocks[1]['battery_discharge'][59]==1.)
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_annual_risk_sums_risk_opportunities(self):
        m=self.build(h=192,starts=(24,120));r=self.solve(m)
        self.assertAlmostEqual(r['eens_kwh'],8.)
        self.assertAlmostEqual(r['cvar_upper_bound_kwh'],40.)
        self.assertEqual(r['window_count'],2)

    def test_regular_loss_outside_stress_windows_is_allowed_and_counted(self):
        m=self.build()
        m.model.addConstr(m.blocks[0]['rigid_shed'][0]==1.)
        r=self.solve(m)
        self.assertAlmostEqual(r['outside_window_loss_kwh'],1.)
        self.assertAlmostEqual(r['eens_kwh'],5.)
        self.assertAlmostEqual(r['cvar_upper_bound_kwh'],21.)
        self.assertAlmostEqual(r['mean_cost_components_yuan']['load_loss'],5000.)


if __name__=='__main__': unittest.main()
