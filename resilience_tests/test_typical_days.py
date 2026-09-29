"""Chronology, weighting and independent full-input semantics of aggregation."""
from dataclasses import replace
import json
import unittest

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from polar_reliability_planning.resilience_milp.annual import AnnualResilienceMILP
from polar_reliability_planning.resilience_milp.model import MILPConfig
from polar_reliability_planning.resilience_milp.typical_days import (
    DayAggregation, TypicalDayResilienceMILP, cluster_days, remove_exact_duplicate_constraints,
)
from resilience_tests.test_resilience_milp import year_case, path_case


class TypicalDayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(empty=True); cls.env.setParam('OutputFlag', 0); cls.env.start()

    @classmethod
    def tearDownClass(cls): cls.env.dispose()

    def fixture(self, aggregated=True):
        year = year_case(240, core=10, rigid=20, inter=1, shift=2)
        mapping = np.array([0, 1, 2, 3, 4, 5, 6, 2, 2, 9])
        plan = DayAggregation(mapping, {d: ['retained'] for d in (0, 1, 3, 4, 5, 6, 9)}, (2,), np.ones(7))
        base = path_case(year, name='annual_normal')
        fault = path_case(year, name='fault', weight=.2, main_trip=(108, 109))
        windows = [dict(start=72, stop=144, paths=[replace(base, weight=.8), fault])]
        cfg = replace(MILPConfig(), external_grid_enabled=True, ups_bridge_hours=0,
                      initial_economic_budget_yuan=250000, mip_gap=0., time_limit_seconds=30.)
        point = dict(wind_kw=0, pv_kw=0, diesel_units=0, battery_kwh=2, pcs_kw=1, ups_kwh=2, ups_kw=1)
        args = dict(env=self.env, fixed_modules=point, compact_fixed=True, compact=True)
        model = (TypicalDayResilienceMILP(base, windows, cfg, plan, year, **args) if aggregated
                 else AnnualResilienceMILP(base, windows, cfg, **args))
        self.addCleanup(model.close)
        return model

    def test_clustering_preserves_extremes_windows_and_annual_weights(self):
        year = year_case(8760)
        core = year.core_kw.copy(); core[5000] = 100
        year = replace(year, core_kw=core)
        plan = cluster_days(year)
        report = plan.report(year)
        self.assertEqual(report['represented_days'], 365)
        self.assertEqual(report['ordinary_typical_days'], 24)
        self.assertEqual(plan.mapping[5000//24], 5000//24)
        for start in (816, 2496, 5616, 7536):
            np.testing.assert_array_equal(plan.mapping[start//24:start//24+3], np.arange(start//24, start//24+3))
        self.assertEqual(plan.reconstruct(year).core_kw.max(), 100)
        self.assertEqual(plan.reconstruct(year).timestamps, year.timestamps)

    def test_shared_controls_keep_distinct_continuous_inventories(self):
        model = self.fixture(); block = model.blocks[0]
        self.assertTrue(block['wind'][48].sameAs(block['wind'][168]))
        self.assertFalse(block['battery_energy'][48].sameAs(block['battery_energy'][168]))
        self.assertFalse(block['ups_energy'][48].sameAs(block['ups_energy'][168]))
        self.assertEqual(len(block['battery_energy']), 241)
        # A shared arrival profile still has a real service variable on the next
        # calendar day, and all occurrences must satisfy the original queues.
        self.assertTrue(block['shift_service'][71, 72].sameAs(block['shift_service'][191, 192]))
        model.model.addConstr(block['shift_service'][191, 192] == 2.)
        result = model.optimize()
        self.assertEqual(model.model.Status, GRB.OPTIMAL)
        self.assertTrue(result['audit']['passed'])
        self.assertFalse(result['full_year_original_data_validated'])
        self.assertFalse(result['economic_domain_certified'])

    def test_repeated_days_costs_risks_equal_full_calendar_and_dedup_preserves_optimum(self):
        original = self.fixture(False); original_result = original.optimize()
        reduced = self.fixture(); before = reduced.optimize()
        self.assertEqual(original.model.Status, GRB.OPTIMAL)
        self.assertEqual(reduced.model.Status, GRB.OPTIMAL)
        stats = remove_exact_duplicate_constraints(reduced.model)
        self.assertGreater(stats['removed_linear_constraints'], 0)
        after = reduced.optimize()
        self.assertTrue(after['audit']['passed'])
        for key in ('objective_yuan', 'eens_kwh', 'cvar_upper_bound_kwh'):
            self.assertAlmostEqual(original_result[key], before[key], places=5)
            self.assertAlmostEqual(before[key], after[key], places=5)
        self.assertLess(reduced.model.NumVars, original.model.NumVars)

    def test_thermal_bounds_include_physical_invariant_range(self):
        # Positive and zero heat-loss coefficients must both bound a trajectory
        # starting below ambient, without imposing an artificial ready state.
        from polar_reliability_planning.resilience_milp.model import ResilienceMILP
        for ua in (.15, 0.):
            cfg = replace(MILPConfig(), thermal_ua_kw_per_k=ua, external_grid_enabled=True,
                          ups_bridge_hours=0., initial_economic_budget_yuan=500000.)
            model = ResilienceMILP([path_case(year_case(24))], cfg, env=self.env, compact=True)
            self.addCleanup(model.close)
            temp = model.blocks[0]['temperature'][0, 0]
            self.assertEqual(temp.LB, cfg.initial_temperature_c)
            self.assertGreaterEqual(temp.UB, 10+cfg.heater_kw/(ua or cfg.thermal_c_kwh_per_k))

    def test_full_calendar_warm_start_expands_aliases(self):
        from scripts.run_resilience_typical_days import transfer_calendar_start
        reduced = self.fixture(); reduced.optimize()
        original = self.fixture(False)
        transfer = transfer_calendar_start(reduced, original)
        self.assertEqual(transfer['missing_count'], 0)
        self.assertAlmostEqual(original.blocks[0]['wind'][168].Start, reduced.blocks[0]['wind'][48].X)
        self.assertAlmostEqual(original.blocks[0]['battery_energy'][168].Start,
                               reduced.blocks[0]['battery_energy'][168].X)
        result = original.optimize()
        self.assertTrue(result['audit']['passed'])

    def test_incumbent_without_finite_dual_bound_can_be_exported(self):
        original = self.fixture(False)
        original.optimize()
        class PresolveTimeoutView:
            def __init__(self, model): self.model = model
            def __getattr__(self, name):
                if name == 'ObjBound': return -float('inf')
                if name == 'MIPGap': return float('inf')
                if name == 'Status': return GRB.TIME_LIMIT
                return getattr(self.model, name)
        original.model = PresolveTimeoutView(original.model)
        result = original.result()
        self.assertTrue(result['audit']['passed'])
        self.assertIsNone(result['objective_bound_yuan'])
        self.assertIsNone(result['mip_gap'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__': unittest.main()
