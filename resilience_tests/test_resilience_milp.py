"""MILP physics, costs and information tests using the actual Gurobi solver."""
from dataclasses import replace
from itertools import product
import unittest
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from polar_reliability_planning.resilience_v2.model import generate_synthetic_year,day_ahead_weather_risk
from polar_reliability_planning.resilience_milp.model import (
    MILPConfig,MILPPath,ResilienceMILP,exogenous_information_nodes,make_demo_paths,weighted_cvar,
)


def year_case(h=12,core=10.,rigid=20.,inter=5.,shift=5.,wind=0.,storm=None):
    y=generate_synthetic_year(h,15)
    extreme=np.zeros(h,bool)
    if storm: extreme[storm[0]:storm[1]]=True
    return replace(y,core_kw=np.full(h,core),rigid_kw=np.full(h,rigid),
                   flex_interruptible_kw=np.full(h,inter),flex_shiftable_kw=np.full(h,shift),
                   wind_clean_pu=np.full(h,wind),pv_pu=np.zeros(h),ambient_c=np.full(h,10.),
                   extreme_weather=extreme,weather_risk=day_ahead_weather_risk(extreme))


def path_case(y,name='normal',weight=1.,grid_fault=None,main_trip=None,pcs_fault=None,ups_fault=None,**kw):
    h=y.hours; grid=np.zeros(h,bool); bus=np.zeros(h,bool)
    pcs=np.ones(h,bool);ups=np.ones(h,bool);main=np.ones(h,bool)
    if grid_fault: grid[grid_fault[0]:grid_fault[1]]=True
    if main_trip: main[main_trip[0]:main_trip[1]]=False
    if pcs_fault: pcs[pcs_fault[0]:pcs_fault[1]]=False
    if ups_fault: ups[ups_fault[0]:ups_fault[1]]=False
    return MILPPath(name,weight,y,grid,bus,pcs,ups,main_bus_available=main,**kw)


class ResilienceMILPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=gp.Env(empty=True);cls.env.setParam('OutputFlag',0);cls.env.start()

    @classmethod
    def tearDownClass(cls): cls.env.dispose()

    def config(self,**kwargs):
        return replace(MILPConfig(),external_grid_enabled=True,ups_bridge_hours=0.,
                       initial_economic_budget_yuan=900000.,time_limit_seconds=30.,
                       initial_temperature_c=5.,mip_gap=0.,**kwargs)

    def build(self,paths,cfg=None,modules=None):
        model=ResilienceMILP(paths,cfg or self.config(),env=self.env,fixed_modules=modules)
        self.addCleanup(model.close)
        return model

    def modules(self,**kwargs):
        result={'wind_kw':0,'pv_kw':0,'diesel_units':0,'battery_kwh':0,'pcs_kw':0,'ups_kwh':2,'ups_kw':1}
        result.update(kwargs);return result

    def solve(self,m):
        result=m.optimize()
        self.assertEqual(m.model.Status,GRB.OPTIMAL)
        self.assertTrue(result['audit']['passed'])
        return result,m.values()

    def test_integer_modules_and_higher_ups_costs(self):
        c=self.config();m=self.build([path_case(year_case())],c)
        self.assertEqual(m.model.NumQConstrs,0)
        self.assertEqual(m.model.NumQNZs,0)
        self.assertTrue(all(v.VType==GRB.INTEGER for v in m.n.values()))
        self.assertEqual(c.ups_yuan_per_kwh/c.battery_yuan_per_kwh,3)
        self.assertEqual(c.ups_yuan_per_kw/c.pcs_yuan_per_kw,3)
        with self.assertRaises(ValueError): replace(c,ups_yuan_per_kw=300).check()
        with self.assertRaises(ValueError): replace(c,ups_yuan_per_kwh=200).check()
        r,_=self.solve(m)
        for name,step in [('wind_kw',100),('pv_kw',100),('battery_kwh',50),('pcs_kw',50),('ups_kwh',50),('ups_kw',50)]:
            self.assertAlmostEqual(r['selected'][name]/step,round(r['selected'][name]/step))
        self.assertTrue(r['economic_domain_certified'])

    def test_ups_does_not_absorb_surplus_renewables_or_supply_normal_load(self):
        m=self.build([path_case(year_case(wind=1.))],modules=self.modules(wind_kw=1,battery_kwh=2,pcs_kw=1))
        _,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['ups_discharge'].sum(),0)
        self.assertAlmostEqual(v['ups_charge'].sum(),0)
        np.testing.assert_allclose(v['ups_energy'],95.)
        self.assertGreater(v['wind'].sum(),0)

    def test_optimizer_cannot_create_emergency_by_turning_off_healthy_bus(self):
        y=year_case(h=8,wind=1.)
        cfg=replace(self.config(),external_grid_enabled=False)
        m=self.build([path_case(y)],cfg,self.modules(wind_kw=1,battery_kwh=10,pcs_kw=1))
        m.model.addConstr(m.blocks[0]['bus_live'][0]==0)
        m.model.addConstr(m.blocks[0]['ups_discharge'][0]==10)
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_ups_actual_outage_loss_charged_once_and_inventory_restored(self):
        y=year_case()
        m=self.build([path_case(y,grid_fault=(5,7),main_trip=(5,7))],modules=self.modules())
        r,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['ups_discharge'].sum(),20.)
        self.assertAlmostEqual(v['ups_energy'][7],95-20/.98)
        self.assertAlmostEqual(v['ups_charge'].sum(),20/.98**2)
        self.assertAlmostEqual(v['ups_energy'][-1],95.)
        loss=2*(20+5+5)
        self.assertAlmostEqual(r['eens_kwh'],loss)
        self.assertAlmostEqual(r['scenario_metrics'][0]['ups_companion_loss_kwh'],loss)
        self.assertAlmostEqual(r['mean_cost_components_yuan']['load_loss'],loss*1000.)
        self.assertAlmostEqual(r['objective_yuan'],r['investment_yuan']+sum(r['mean_cost_components_yuan'].values()))

    def test_ups_cannot_power_rigid_load(self):
        y=year_case(inter=0,shift=0)
        m=self.build([path_case(y,grid_fault=(5,6),main_trip=(5,6))],modules=self.modules())
        _,v=self.solve(m)
        self.assertAlmostEqual(v[0]['ups_discharge'][5],10.)
        self.assertAlmostEqual(v[0]['rigid_shed'][5],20.)

    def test_empty_battery_cannot_have_storage_pcs(self):
        m=self.build([path_case(year_case())],modules=self.modules(pcs_kw=1,battery_kwh=0))
        m.model.optimize();self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_island_renewables_require_actual_gfm_power(self):
        y=year_case(h=8,wind=1.)
        cfg=replace(self.config(),external_grid_enabled=False)
        m=self.build([path_case(y)],cfg,self.modules(wind_kw=1,battery_kwh=10,pcs_kw=1))
        r,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['grid_main'].sum(),0.)
        self.assertTrue(np.all(np.maximum(v['battery_charge'],v['battery_discharge'])>=1-1e-6))
        self.assertAlmostEqual(v['ups_charge'].sum()+v['ups_discharge'].sum(),0.)
        self.assertTrue(r['audit']['passed'])
        # Disable all actual PCS powers: nameplate capacity alone cannot form.
        bad=self.build([path_case(y)],cfg,self.modules(wind_kw=1,battery_kwh=10,pcs_kw=1))
        for t in range(y.hours):
            bad.model.addConstr(bad.blocks[0]['battery_charge'][t]==0)
            bad.model.addConstr(bad.blocks[0]['battery_discharge'][t]==0)
        bad.model.optimize();self.assertEqual(bad.model.Status,GRB.INFEASIBLE)

    def test_island_ups_replenishment_is_diesel_funded_not_renewable(self):
        y=year_case(h=16,wind=1.)
        cfg=replace(self.config(),external_grid_enabled=False)
        m=self.build([path_case(y,main_trip=(7,8))],cfg,
                     self.modules(wind_kw=1,diesel_units=1,battery_kwh=10,pcs_kw=1))
        r,v=self.solve(m);v=v[0]
        self.assertGreater(v['ups_charge'].sum(),0.)
        self.assertAlmostEqual(v['ups_grid_charge'].sum(),0.)
        self.assertAlmostEqual(v['ups_diesel_charge'].sum(),v['ups_charge'].sum())
        self.assertTrue(np.all(v['ups_diesel_charge']<=v['diesel_power'].sum(axis=0)+1e-6))
        self.assertAlmostEqual(r['mean_cost_components_yuan']['grid'],0.)

    def test_preparation_access_temperature_and_heater_power(self):
        y=year_case(storm=(0,3))
        cfg=self.config()
        m=self.build([path_case(y)],cfg,self.modules(diesel_units=1))
        b=m.blocks[0]
        # Earliest request at 3, earliest successful preparation at 6.
        m.model.addConstr(b['attempt'][0,6]==1)
        r,v=self.solve(m);v=v[0]
        self.assertLess(v['request'][0,:3].sum(),1e-6)
        self.assertAlmostEqual(v['request'][0,3],1.)
        self.assertGreaterEqual(v['temperature'][0,6],cfg.ready_temperature_c-1e-6)
        early=self.build([path_case(y)],cfg,self.modules(diesel_units=1))
        early.model.addConstr(early.blocks[0]['attempt'][0,5]==1)
        early.model.optimize();self.assertEqual(early.model.Status,GRB.INFEASIBLE)
        cold=replace(y,ambient_c=np.full(y.hours,-20.))
        weak=replace(cfg,initial_temperature_c=-10.,heater_kw=1.)
        thermal=self.build([path_case(cold)],weak,self.modules(diesel_units=1))
        thermal.model.addConstr(thermal.blocks[0]['attempt'][0,6]==1)
        thermal.model.optimize();self.assertEqual(thermal.model.Status,GRB.INFEASIBLE)

    def test_arrived_crew_continues_preparation_during_storm(self):
        y=year_case(storm=(1,6))
        m=self.build([path_case(y)],modules=self.modules(diesel_units=1))
        m.model.addConstr(m.blocks[0]['request'][0,0]==1)
        m.model.addConstr(m.blocks[0]['attempt'][0,3]==1)
        _,v=self.solve(m)
        self.assertAlmostEqual(v[0]['attempt'][0,3],1.)

    def test_online_zero_output_engine_does_not_count_as_gfm(self):
        y=year_case()
        m=self.build([path_case(y,grid_fault=(7,8),main_trip=(7,8))],modules=self.modules(diesel_units=1))
        b=m.blocks[0]
        m.model.addConstr(b['attempt'][0,3]==1)
        m.model.addConstr(b['online'][0,7]==1)
        _,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['online'][0,7],1.)
        self.assertAlmostEqual(v['diesel_power'][0,7],0.)
        self.assertAlmostEqual(v['diesel_connected'][0,7],0.)
        self.assertAlmostEqual(v['bus_live'][7],0.)
        self.assertAlmostEqual(v['ups_discharge'][7],10.)

    def test_exposed_run_failure_and_calendar_repair(self):
        y=year_case()
        p=path_case(y,run_shocks=frozenset({(0,5)}),run_repair_hours=2)
        m=self.build([p],modules=self.modules(diesel_units=1))
        b=m.blocks[0]
        m.model.addConstr(b['attempt'][0,3]==1)
        _,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['online'][0,5],1.)
        self.assertAlmostEqual(v['run_fail'][0,6],1.)
        np.testing.assert_allclose(v['failed_pre'][0,6:9],[1,1,0])
        off=self.build([p],modules=self.modules())
        _,v=self.solve(off)
        self.assertAlmostEqual(v[0]['run_fail'].sum(),0.)

    def test_start_failure_is_per_actual_attempt(self):
        y=year_case()
        p=path_case(y,start_shocks=frozenset({(0,3)}),start_repair_hours=2)
        m=self.build([p],modules=self.modules(diesel_units=1))
        m.model.addConstr(m.blocks[0]['attempt'][0,3]==1)
        _,v=self.solve(m);v=v[0]
        self.assertAlmostEqual(v['start_fail'][0,3],1.)
        self.assertAlmostEqual(v['online'][0,3],0.)
        np.testing.assert_allclose(v['failed_post'][0,3:6],[1,1,0])

    def test_day_ahead_forecast_does_not_reveal_later_days(self):
        a,b=year_case(72),year_case(72,storm=(60,65))
        ps=[path_case(a,'a',.5),path_case(b,'b',.5)]
        nodes=exogenous_information_nodes(ps)
        np.testing.assert_array_equal(nodes[0,:24],nodes[1,:24])
        self.assertNotEqual(nodes[0,24],nodes[1,24])
        # A model with identical current information cannot request different starts.
        cfg=replace(self.config(),initial_economic_budget_yuan=250000.)
        m=self.build(ps,cfg)
        m.model.addConstr(m.blocks[0]['request'][0,0]==1)
        m.model.addConstr(m.blocks[1]['request'][0,0]==0)
        m.model.optimize();self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_future_fault_is_not_observed_before_it_occurs(self):
        y=year_case()
        ps=[path_case(y,'a',.5),path_case(y,'b',.5,grid_fault=(7,9))]
        nodes=exogenous_information_nodes(ps)
        np.testing.assert_array_equal(nodes[0,:7],nodes[1,:7])
        self.assertNotEqual(nodes[0,7],nodes[1,7])
        m=self.build(ps,modules=self.modules(diesel_units=1))
        m.model.addConstr(m.blocks[0]['request'][0,2]==1)
        m.model.addConstr(m.blocks[1]['request'][0,2]==0)
        m.model.optimize();self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_hidden_shock_on_off_unit_cannot_unlock_future_decisions(self):
        y=year_case()
        ps=[path_case(y,'a',.5),path_case(y,'b',.5,run_shocks=frozenset({(0,1)}))]
        m=self.build(ps,modules=self.modules(diesel_units=1))
        # No unit can be online at hour 1 because preparation takes three hours.
        # The hidden shock therefore reveals nothing at hour 2.
        m.model.addConstr(m.blocks[0]['request'][0,2]==1)
        m.model.addConstr(m.blocks[1]['request'][0,2]==0)
        m.model.optimize();self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_actual_failure_allows_recourse_after_revelation(self):
        y=year_case()
        ps=[path_case(y,'a',.5),path_case(y,'b',.5,run_shocks=frozenset({(0,5)}))]
        m=self.build(ps,modules=self.modules(diesel_units=1))
        for b in m.blocks: m.model.addConstr(b['attempt'][0,3]==1)
        m.model.addConstr(m.blocks[0]['online'][0,6]==1)
        m.model.addConstr(m.blocks[1]['online'][0,6]==0)
        r,v=self.solve(m)
        self.assertLess(r['audit']['violations']['nonanticipativity'],1e-5)
        np.testing.assert_allclose(v[0]['request'][:,:6],v[1]['request'][:,:6],atol=1e-6)

    def test_start_failure_revealed_after_common_start_decision(self):
        y=year_case()
        ps=[path_case(y,'a',.5),path_case(y,'b',.5,start_shocks=frozenset({(0,3)}))]
        m=self.build(ps,modules=self.modules(diesel_units=1))
        for b in m.blocks: m.model.addConstr(b['attempt'][0,3]==1)
        r,v=self.solve(m)
        self.assertAlmostEqual(v[0]['online'][0,3],1.)
        self.assertAlmostEqual(v[1]['online'][0,3],0.)
        np.testing.assert_allclose(v[0]['pre_controls'][3],v[1]['pre_controls'][3],atol=1e-6)
        self.assertLess(r['audit']['violations']['nonanticipativity'],1e-5)

    def test_ups_cannot_recharge_from_wind_without_diesel_or_external_input(self):
        y=year_case(h=16,wind=1.)
        cfg=replace(self.config(),external_grid_enabled=False)
        m=self.build([path_case(y,main_trip=(7,8))],cfg,
                     self.modules(wind_kw=1,battery_kwh=10,pcs_kw=1))
        # The bus trip forces UPS use; the terminal backup must be restored.
        # Abundant renewable power cannot be routed into its dedicated charger.
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_future_repair_duration_not_revealed_at_failure_time(self):
        y=year_case()
        ps=[path_case(y,'a',.5,run_shocks=frozenset({(0,5)}),run_repair_hours=2),
            path_case(y,'b',.5,run_shocks=frozenset({(0,5)}),run_repair_hours=4)]
        m=self.build(ps,modules=self.modules(diesel_units=1))
        for b in m.blocks: m.model.addConstr(b['attempt'][0,3]==1)
        # Both units are failed at hour 7. Future repair dates are hidden, so
        # different current grid imports are illegal even though the dates differ.
        m.model.addConstr(m.blocks[0]['grid_main'][7]==40)
        m.model.addConstr(m.blocks[1]['grid_main'][7]==30)
        m.model.optimize()
        self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_tomorrow_forecast_does_not_disclose_its_exact_event_hour(self):
        a,b=year_case(72,storm=(25,26)),year_case(72,storm=(45,46))
        nodes=exogenous_information_nodes([path_case(a,'a',.5),path_case(b,'b',.5)])
        np.testing.assert_array_equal(nodes[0,:24],nodes[1,:24])

    def test_flexibility_excluded_only_if_legally_adjusted_or_repaid(self):
        y=year_case(h=6,core=10,rigid=20,inter=10,shift=10)
        m=self.build([path_case(y)],modules=self.modules())
        b=m.blocks[0]
        m.model.addConstr(b['shift_service'][0,0]==0)
        m.model.addConstr(b['shift_service'][0,1]==10)
        m.model.addConstr(b['interrupt_adjust'][0]==5)
        r,v=self.solve(m)
        self.assertAlmostEqual(r['eens_kwh'],0.)
        self.assertGreater(r['mean_cost_components_yuan']['flex_adjustment'],0.)
        lost=self.build([path_case(y)],modules=self.modules())
        lost.model.addConstr(lost.blocks[0]['shift_service'][5,5]==0)
        r,v=self.solve(lost)
        self.assertAlmostEqual(r['eens_kwh'],10.)
        self.assertAlmostEqual(r['mean_cost_components_yuan']['load_loss'],10000.)

    def test_loss_price_changes_objective_by_actual_loss_not_ups_energy(self):
        y=year_case()
        p=path_case(y,grid_fault=(5,6),main_trip=(5,6))
        lo=self.build([p],replace(self.config(),loss_yuan_per_kwh=1000),self.modules())
        hi=self.build([p],replace(self.config(),loss_yuan_per_kwh=2000),self.modules())
        a,_=self.solve(lo);b,_=self.solve(hi)
        self.assertAlmostEqual(a['eens_kwh'],30.)
        self.assertAlmostEqual(b['objective_yuan']-a['objective_yuan'],30000.,places=4)

    def test_weighted_cvar_and_hard_risk_limit(self):
        self.assertAlmostEqual(weighted_cvar([0,100,200],[.9,.08,.02],.95),140.)
        y=year_case()
        cfg=replace(self.config(),eens_limit_kwh=0.,cvar_limit_kwh=0.)
        m=self.build([path_case(y,grid_fault=(5,6),main_trip=(5,6))],cfg,self.modules())
        m.model.optimize();self.assertEqual(m.model.Status,GRB.INFEASIBLE)

    def test_compound_scenario_has_real_storm_and_renewable_bus_failure(self):
        paths=make_demo_paths(year_case(72))
        self.assertAlmostEqual(sum(p.weight for p in paths),1.)
        p=next(p for p in paths if p.name=='storm_compound')
        self.assertGreater((p.year.extreme_weather & p.renewable_bus_fault).sum(),0)
        before=np.flatnonzero(p.renewable_bus_fault)[0]-1
        alternative=next(q for q in paths if q.name=='storm_none')
        nodes=exogenous_information_nodes([p,alternative])
        np.testing.assert_array_equal(nodes[0,:before+1],nodes[1,:before+1])

    def test_timeout_status_does_not_claim_optimality(self):
        m=self.build([path_case(year_case())])
        m.model.Params.TimeLimit=0
        m.model.optimize()
        r=m.result()
        self.assertEqual(m.model.Status,GRB.TIME_LIMIT)
        self.assertEqual(r['status'],'unresolved_without_incumbent')
        self.assertFalse(r['economic_domain_certified'])


if __name__=='__main__': unittest.main()
