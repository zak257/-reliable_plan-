import unittest

import numpy as np

from polar_reliability_planning.resilience_v2.model import (
    Capacity,
    ResilienceConfig,
    evaluate_capacity,
    generate_synthetic_year,
    make_scenarios,
    simulate,
)
from scripts.run_resilience_v2 import (
    assert_discrete_capacity_grid, capacity_grid_levels, global_capacity_grid,
    global_capacity_grid_iterator,
)


class ResilienceV2Tests(unittest.TestCase):
    def test_synthetic_year_has_three_load_classes(self):
        year = generate_synthetic_year(72, 1)
        self.assertEqual(year.hours, 72)
        self.assertTrue((year.core_kw > 0).all())
        self.assertTrue((year.rigid_kw > 0).all())
        self.assertTrue((year.flex_interruptible_kw > 0).all())
        self.assertTrue((year.flex_shiftable_kw > 0).all())
        self.assertTrue(np.allclose(year.total_kw - year.core_kw - year.rigid_kw
                                    - year.flex_interruptible_kw - year.flex_shiftable_kw, 0.0))

    def test_ups_bridge_requirement_is_capacity_constraint(self):
        cfg = ResilienceConfig(hours=48, core_recovery_hours=8.0, core_power_margin=1.5)
        year = generate_synthetic_year(48, 2)
        scenarios = make_scenarios(year, cfg, 2, 3, "joint")
        too_small = Capacity(500, 300, 0, 1000, 200, 100, 50)
        result = evaluate_capacity(year, too_small, scenarios, cfg)
        self.assertFalse(result["core_bridge_pass"])
        self.assertFalse(result["feasible"])
        self.assertGreater(result["required_ups_kwh"], too_small.ups_kwh)

    def test_all_pcs_and_diesel_are_grid_forming(self):
        cfg = ResilienceConfig(hours=48)
        year = generate_synthetic_year(48, 4)
        scenario = make_scenarios(year, cfg, 1, 5, "joint")[0]
        result = simulate(year, Capacity(500, 300, 0, 1000, 200, 400, 100), scenario, cfg)
        self.assertTrue(result.metrics["gfm_all_resources"])
        self.assertEqual(len(result.trace), 0)
        traced = simulate(year, Capacity(500, 300, 0, 1000, 200, 400, 100), scenario, cfg, trace=True)
        self.assertEqual(len(traced.trace), 48)
        self.assertIn("gfm_power_kw", traced.trace[0])

    def test_flexible_adjustment_is_reported_separately(self):
        cfg = ResilienceConfig(hours=48)
        year = generate_synthetic_year(48, 6)
        scenario = make_scenarios(year, cfg, 1, 7, "random")[0]
        result = simulate(year, Capacity(0, 0, 0, 0, 0, 400, 100), scenario, cfg)
        self.assertGreaterEqual(result.flex_adjusted_kwh, 0.0)
        self.assertEqual(result.rigid_unserved_kwh, float(result.rigid_unserved_kwh))

    def test_global_grid_starts_diesel_at_zero_and_uses_cost_budget(self):
        cfg = ResilienceConfig(hours=48)
        year = generate_synthetic_year(48, 8)
        scenarios = make_scenarios(year, cfg, 2, 9, "joint")
        reference = evaluate_capacity(
            year, Capacity(500, 300, 0, 1000, 200, 1000, 100), scenarios, cfg
        )
        self.assertTrue(reference["feasible"])
        levels, iterator = global_capacity_grid_iterator(year, cfg, reference["objective_yuan"])
        self.assertEqual(levels["wind_kw"][:3], (0.0, 100.0, 200.0))
        self.assertEqual(levels["battery_kwh"][:3], (0.0, 50.0, 100.0))
        self.assertEqual(levels["pcs_kw"][:3], (0.0, 50.0, 100.0))
        self.assertEqual(levels["ups_kwh"][:3], (0.0, 50.0, 100.0))
        self.assertEqual(levels["ups_kw"][:3], (0.0, 50.0, 100.0))
        sample = [next(iterator) for _ in range(20)]
        assert_discrete_capacity_grid(sample)
        self.assertEqual(sample[0].diesel_units, 0)
        self.assertTrue(all(cap.ups_kwh >= 0 and cap.ups_kw >= 0 for cap in sample))
        self.assertTrue(all(cap.battery_kwh >= cap.pcs_kw for cap in sample))

    def test_scenario_draws_do_not_limit_diesel_count(self):
        cfg = ResilienceConfig(hours=24)
        year = generate_synthetic_year(24, 10)
        scenario = make_scenarios(year, cfg, 1, 11, "random")[0]
        result = simulate(year, Capacity(0, 0, 13, 0, 200, 400, 100), scenario, cfg)
        self.assertTrue(np.isfinite(result.rigid_unserved_kwh))

    def test_ups_is_not_used_for_normal_grid_energy_deficit(self):
        cfg = ResilienceConfig(hours=48)
        year = generate_synthetic_year(48, 12)
        scenario = make_scenarios(year, cfg, 1, 13, "random")[0]
        result = simulate(year, Capacity(0, 0, 0, 0, 0, 600, 100), scenario, cfg, trace=True)
        normal_rows = [row for row in result.trace if not row["grid_fault"]]
        self.assertEqual(sum(row["ups_discharge_kw"] for row in normal_rows), 0.0)

    def test_storage_pcs_requires_battery_and_nonzero_gfm_power(self):
        cfg = ResilienceConfig(hours=48)
        year = generate_synthetic_year(48, 14)
        scenario = make_scenarios(year, cfg, 1, 15, "random")[0]
        invalid = Capacity(0, 0, 1, 0, 200, 600, 100)
        invalid_eval = evaluate_capacity(year, invalid, [scenario], cfg)
        self.assertFalse(invalid_eval["pcs_battery_coupling_pass"])
        self.assertFalse(invalid_eval["feasible"])
        valid = Capacity(0, 0, 1, 1500, 200, 600, 100)
        valid_eval = evaluate_capacity(year, valid, [scenario], cfg)
        self.assertTrue(valid_eval["pcs_battery_coupling_pass"])

    def test_compound_extreme_scenario_overlaps_blizzard_and_renewable_bus_fault(self):
        cfg = ResilienceConfig(hours=72)
        year = generate_synthetic_year(72, 16)
        scenario = make_scenarios(year, cfg, 1, 17, "compound_extreme")[0]
        self.assertTrue(scenario.compound_extreme)
        self.assertTrue(np.array_equal(scenario.grid_fault, year.extreme_weather))
        self.assertTrue(np.array_equal(scenario.renewable_bus_fault, year.extreme_weather))
        self.assertTrue(scenario.weather_enabled)


if __name__ == "__main__":
    unittest.main()
