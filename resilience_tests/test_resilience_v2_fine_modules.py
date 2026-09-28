"""Regression evidence for the requested fine module and information contract."""
from dataclasses import replace
from itertools import product
import unittest

import numpy as np

from polar_reliability_planning.resilience_v2.model import (
    Capacity, ResilienceConfig, assert_discrete_capacity, day_ahead_weather_risk,
    evaluate_capacity, generate_synthetic_year, make_scenarios, simulate,
    prepare_outage_bounds, outage_risk_lower_bound, plan_capacity,
)
from scripts.run_resilience_v2 import capacity_grid_levels, iter_capacity_grid


def year_case(hours=72, core=10., rigid=20., interruptible=5., shiftable=5., storm=None):
    y = generate_synthetic_year(hours, 120)
    extreme = np.zeros(hours, bool)
    if storm is not None:
        extreme[storm[0]:storm[1]] = True
    return replace(y, core_kw=np.full(hours, core), rigid_kw=np.full(hours, rigid),
                   flex_interruptible_kw=np.full(hours, interruptible),
                   flex_shiftable_kw=np.full(hours, shiftable), extreme_weather=extreme,
                   weather_risk=day_ahead_weather_risk(extreme),
                   wind_clean_pu=np.ones(hours), pv_pu=np.zeros(hours))


def path_case(y, cfg, fault=None, bus=False):
    s = make_scenarios(y, cfg, 1, 55, 'weather')[0]
    grid = np.zeros(y.hours, bool)
    if fault is not None:
        grid[fault[0]:fault[1]] = True
    return replace(s, grid_fault=grid, renewable_bus_fault=grid.copy() if bus else np.zeros(y.hours, bool))


class FineModuleTests(unittest.TestCase):
    def test_all_module_levels_and_economic_bounds(self):
        cfg = ResilienceConfig()
        levels = capacity_grid_levels(year_case(), cfg, 2_000_000.)
        steps = (100, 100, 1, 50, 50, 50, 50)
        costs = (1500, 1000, 180000, 250, 400, 450, 250)
        for (name, values), step, cost in zip(levels.items(), steps, costs):
            self.assertEqual(values[0], 0)
            self.assertTrue(all(b-a == step for a, b in zip(values, values[1:])))
            self.assertLessEqual(values[-1]*cost, 2_000_000)
            self.assertGreater((values[-1]+step)*cost, 2_000_000)
        self.assertGreater(levels['wind_kw'][-1], 2 * year_case().total_kw.max())

    def test_heap_matches_exhaustive_cartesian_oracle(self):
        levels = {'wind_kw': (0.,100.,200.), 'pv_kw': (0.,100.), 'diesel_units': (0,1),
                  'battery_kwh': (0.,50.,100.), 'pcs_kw': (0.,50.),
                  'ups_kwh': (0.,50.,100.), 'ups_kw': (0.,50.,100.)}
        cfg = ResilienceConfig()
        expected = [Capacity(*x) for x in product(*levels.values())]
        expected = [x for x in expected if x.investment_yuan <= 420000
                    and x.ups_kwh >= 50 and x.ups_kw >= 50 and x.battery_kwh >= x.pcs_kw]
        actual = list(iter_capacity_grid(levels, cfg, 420000, 49., 49.))
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual([x.investment_yuan for x in actual], sorted(x.investment_yuan for x in actual))

    def test_reject_fractional_or_wrong_sized_modules(self):
        good = Capacity(100,100,1,50,50,100,50)
        assert_discrete_capacity(good)
        for key, value in [('wind_kw',150),('pv_kw',50),('battery_kwh',75),
                           ('pcs_kw',25),('ups_kwh',125),('ups_kw',75),('diesel_units',1.5)]:
            with self.assertRaises(ValueError):
                assert_discrete_capacity(replace(good, **{key:value}))

    def test_only_next_day_boolean_is_released(self):
        y = year_case(storm=(60,65))
        self.assertFalse(y.weather_risk[:24].any())
        self.assertTrue(y.weather_risk[24:48].all())
        self.assertFalse(y.weather_risk[48:].any())
        other = year_case(storm=(49,50))
        np.testing.assert_array_equal(y.weather_risk[:48], other.weather_risk[:48])

    def test_future_weather_and_faults_do_not_change_today_actions(self):
        cfg = ResilienceConfig(random_failures_enabled=False)
        a = year_case()
        b = year_case(storm=(60,65))
        sa = path_case(a, cfg)
        sb = path_case(b, cfg, fault=(50,65), bus=True)
        cap = Capacity(100,0,2,200,100,200,50)
        ra = simulate(a,cap,sa,cfg,trace=True)
        rb = simulate(b,cap,sb,cfg,trace=True)
        self.assertEqual(ra.trace[:24], rb.trace[:24])
        # Tomorrow risk becomes available at hour 24, and can change decisions.
        self.assertNotEqual(ra.trace[24]['ups_reserve_target_kwh'], rb.trace[24]['ups_reserve_target_kwh'])

    def test_tomorrow_event_hour_is_not_visible_today(self):
        cfg = ResilienceConfig(random_failures_enabled=False)
        a, b = year_case(storm=(25,26)), year_case(storm=(45,46))
        cap = Capacity(0,0,1,100,50,200,50)
        ra = simulate(a,cap,path_case(a,cfg),cfg,True)
        rb = simulate(b,cap,path_case(b,cfg),cfg,True)
        self.assertEqual(ra.trace[:24], rb.trace[:24])

    def test_ups_energy_exhaustion_and_real_outage_cost(self):
        cfg = ResilienceConfig(ups_initial_soc=.6, random_failures_enabled=False)
        y = year_case(4, core=20.)
        s = path_case(y,cfg,(0,4),True)
        cap = Capacity(0,0,0,0,0,100,50)
        r = simulate(y,cap,s,cfg,True)
        self.assertTrue(r.valid)
        self.assertAlmostEqual(r.metrics['ups_discharge_kwh'], (60-5)*.98)
        self.assertAlmostEqual(r.metrics['final_ups_kwh'], 5.)
        self.assertAlmostEqual(r.core_unserved_kwh, 80-(60-5)*.98)
        self.assertAlmostEqual(r.metrics['regular_unserved_kwh'], 4*(20+5+5))
        self.assertEqual(r.flex_adjusted_kwh, 0.)
        zero = simulate(y,cap,s,replace(cfg,loss_of_load_cost_yuan_per_kwh=0))
        expected = (r.core_unserved_kwh+r.metrics['regular_unserved_kwh'])*1000
        self.assertAlmostEqual(r.operating_cost_yuan-zero.operating_cost_yuan, expected)
        self.assertAlmostEqual(r.operating_cost_yuan, r.metrics['load_shed_cost_yuan'])
        self.assertGreater(r.metrics['ups_activation_loss_cost_yuan'],0)
        self.assertLessEqual(r.metrics['ups_activation_loss_cost_yuan'],r.operating_cost_yuan)

    def test_authorized_interruptible_adjustment_is_not_outage(self):
        cfg = ResilienceConfig(diesel_start_delay_h=0, random_failures_enabled=False)
        y = year_case(2,core=10.,rigid=90.,interruptible=20.,shiftable=0.)
        r = simulate(y,Capacity(0,0,1,0,0,200,50),path_case(y,cfg,(0,2)),cfg,True)
        self.assertTrue(r.valid)
        self.assertEqual(r.flex_adjusted_kwh,40.)
        self.assertEqual(r.metrics['regular_unserved_kwh'],0.)
        self.assertAlmostEqual(r.metrics['flex_adjustment_cost_yuan'],4.8)

    def test_shifted_work_must_be_repaid(self):
        cfg = ResilienceConfig(diesel_start_delay_h=0, random_failures_enabled=False)
        y = year_case(2,core=20.,rigid=90.,interruptible=0.,shiftable=20.)
        cap = Capacity(0,0,1,0,0,200,50)
        r = simulate(y,cap,path_case(y,cfg,(0,1)),cfg,True)
        self.assertTrue(r.valid)
        self.assertEqual(r.metrics['flex_shifted_kwh'],20.)
        self.assertEqual(r.metrics['flex_shift_repaid_kwh'],20.)
        self.assertEqual(r.metrics['terminal_shift_loss_kwh'],0.)
        self.assertEqual(r.metrics['flex_forced_unserved_kwh'],0.)
        short = year_case(1,core=20.,rigid=90.,interruptible=0.,shiftable=20.)
        unfinished = simulate(short,cap,path_case(short,cfg,(0,1)),cfg)
        self.assertEqual(unfinished.metrics['terminal_shift_loss_kwh'],20.)
        self.assertEqual(unfinished.metrics['regular_unserved_kwh'],30.)

    def test_surplus_renewables_need_real_pcs_power_and_energy(self):
        cfg = ResilienceConfig(random_failures_enabled=False)
        y = year_case(4)
        cap = Capacity(100,0,0,100,50,200,50)
        r = simulate(y,cap,path_case(y,cfg,(0,4)),cfg,True)
        self.assertTrue(r.valid)
        for row in r.trace:
            self.assertEqual(row['gfm_power_kw'],row['battery_discharge_kw'])
            self.assertGreater(row['battery_discharge_kw'],0)
            self.assertEqual(row['battery_charge_kw'],0.)
            self.assertAlmostEqual(row['power_balance_residual_kw'],0.)
        self.assertAlmostEqual(r.metrics['final_battery_kwh'],60-4/.95)
        empty = simulate(y,replace(cap,battery_kwh=0),path_case(y,cfg,(0,4)),cfg,True)
        self.assertEqual(empty.trace[0]['gfm_power_kw'],0.)
        self.assertEqual(empty.trace[0]['main_bus_live'],0)

    def test_ups_charging_uses_one_shared_converter_limit(self):
        cfg = ResilienceConfig(ups_initial_soc=.05, random_failures_enabled=False)
        y = year_case(2)
        y = replace(y,wind_clean_pu=np.full(2,.65))
        r = simulate(y,Capacity(100,0,0,0,0,1000,50),path_case(y,cfg),cfg,True)
        self.assertTrue(r.valid)
        self.assertTrue(all(row['ups_charge_kw'] <= 50. for row in r.trace))
        self.assertAlmostEqual(r.metrics['final_ups_kwh'],50+100*.98)

    def test_diesel_fuel_is_paid_when_grid_connected_and_stops_after_risk(self):
        cfg = ResilienceConfig(diesel_start_delay_h=0, random_failures_enabled=False)
        y = year_case(72,storm=(24,25))
        r = simulate(y,Capacity(0,0,1,0,0,200,50),path_case(y,cfg),cfg,True)
        self.assertGreater(r.metrics['fuel_cost_yuan'],0.)
        self.assertAlmostEqual(r.metrics['fuel_cost_yuan'],sum(row['diesel_kw'] for row in r.trace)*.95)
        self.assertEqual(r.trace[26]['diesel_kw'],0.)

    def test_compound_stress_is_nonempty_and_balance_holds(self):
        cfg = ResilienceConfig(random_failures_enabled=False)
        y = year_case(72,storm=(24,54))
        s = make_scenarios(y,cfg,1,55,'compound_extreme')[0]
        self.assertEqual(s.grid_fault.sum(),30)
        self.assertTrue(np.array_equal(s.grid_fault,s.renewable_bus_fault))
        r = simulate(y,Capacity(100,100,1,200,100,200,50),s,cfg,True)
        self.assertTrue(r.valid)
        self.assertLess(r.metrics['max_power_balance_residual_kw'],1e-6)
        self.assertTrue(all(row['renewable_kw']==0 for row in r.trace if row['weather_stress']))

    def test_risk_bound_is_below_simulated_loss_for_every_tiny_capacity(self):
        cfg = ResilienceConfig(random_failures_enabled=False)
        y = year_case(48,storm=(24,40))
        ss = [path_case(y,cfg,(0,5),False),path_case(y,cfg,(24,40),True)]
        windows = prepare_outage_bounds(y,ss)
        for wind,diesel,energy in product((0,100),(0,1),(0,100)):
            cap = Capacity(wind,0,diesel,energy,50 if energy else 0,200,50)
            lower = outage_risk_lower_bound(cap,cfg,windows)
            e = evaluate_capacity(y,cap,ss,cfg)
            self.assertLessEqual(lower['eens_kwh'],e['regular_risk']['eens_kwh']+1e-8)
            self.assertLessEqual(lower['cvar_kwh'],e['regular_risk']['cvar_kwh']+1e-8)

    def test_bound_search_matches_full_simulation_oracle(self):
        cfg = ResilienceConfig(random_failures_enabled=False,core_recovery_hours=0,
                               rigid_eens_limit_kwh=30,rigid_cvar_limit_kwh=100)
        y = year_case(48,storm=(24,30))
        ss = [path_case(y,cfg,(24,30),True)]
        caps = [Capacity(w,0,d,b,50 if b else 0,50,50)
                for w,d,b in product((0,100),(0,1),(0,100))]
        oracle = plan_capacity(y,cfg,caps,ss,None,use_outage_bound=False)
        accelerated = plan_capacity(y,cfg,iter(sorted(caps,key=lambda c:c.investment_yuan)),ss,None,
                                    investment_ordered=True)
        self.assertEqual(oracle.selected,accelerated.selected)
        self.assertTrue(accelerated.metadata['planning_complete'])
        self.assertGreater(accelerated.metadata['skipped_by_optimistic_risk_bound'],0)

    def test_timeout_never_claims_global_and_unordered_iterator_does_not_skip(self):
        cfg = ResilienceConfig(random_failures_enabled=False,core_recovery_hours=0)
        y = year_case(2)
        ss = [path_case(y,cfg)]
        cheap = Capacity(0,0,0,0,0,50,50)
        expensive = replace(cheap,wind_kw=100)
        timed = plan_capacity(y,cfg,[cheap,expensive],ss,0,cheap)
        self.assertFalse(timed.metadata['planning_complete'])
        self.assertEqual(timed.status,'time_limit_feasible_incumbent_not_global')
        result = plan_capacity(y,cfg,iter([expensive,cheap]),ss,None,expensive)
        self.assertEqual(result.selected['capacity']['wind_kw'],0)


if __name__ == '__main__':
    unittest.main()
