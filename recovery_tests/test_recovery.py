import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from polar_reliability_planning.data.cap_plan_loader import CaseData, COMPONENTS
from polar_reliability_planning.planning.recovery_optimizer import PathArchive, plan_grid
from polar_reliability_planning.recovery.config import read_recovery_config, settings_from_config
from polar_reliability_planning.recovery.container_thermal import ThermalParameters
from polar_reliability_planning.recovery_cli import ROOT, load_inputs
from polar_reliability_planning.reliability.causal_simulator import simulate
from polar_reliability_planning.reliability.recovery_policy import Observation, RecoveryPolicy
from polar_reliability_planning.reliability.risk_bounds import partial_bounds, risk_summary
from polar_reliability_planning.scenario_generation.primitive_noise import PrimitiveNoise, repair_duration
from polar_reliability_planning.validation.recovery_audit import audit_trace
from polar_reliability_planning.weather.hourly_weather import WeatherYear


class FixedNoise:
    warmup = 0

    def __init__(self, forced=None, warmup=0):
        self.forced = forced or {}
        self.warmup = warmup

    def uniform(self, mechanism, hour, device):
        return self.forced.get((mechanism, hour, device), 0.9)


def fixture(hours=24, load=30.0, pv=1.0, access=None, ambient=-10.0):
    config = read_recovery_config(ROOT / "config/zhongshan_recovery_stress.toml")
    settings = replace(settings_from_config(config), warmup_hours=0,
                       thermal=ThermalParameters(0, 1, 1, 5, 5),
                       min_up_hours=0, min_down_hours=0, initial_soc=0.8,
                       failures={k: dict(rate_per_hour=0.0, repair_mean_hours=2) for k in ("wind", "pv", "pcs")},
                       diesel_repair_mean_hours=2, repair_distribution="fixed")
    data = CaseData("synthetic", np.full(hours, load), np.ones(hours), np.full(hours, pv),
                    dict(zip(COMPONENTS, (100.0, 100.0, 100.0, 50.0, 50.0))),
                    {k: (0, 10) for k in COMPONENTS}, {k: 100.0 for k in COMPONENTS},
                    2.0, 0.9, 0.0, 0.95)
    times = tuple(t.isoformat() for t in pd.date_range("2020-01-01", periods=hours, freq="h", tz="UTC"))
    weather = WeatherYear(times, data.load_kw, np.full(hours, ambient), np.full(hours, 5.0),
                          data.wind_pu, data.pv_pu, np.ones(hours) if access is None else np.array(access),
                          np.zeros(hours), np.ones(hours), np.zeros(hours), np.zeros(hours), np.zeros(hours),
                          {"synthetic": True})
    units = dict(wind=0, pv=1, diesel=1, battery_energy=4, pcs=2)
    policy = RecoveryPolicy("test", 1, 0, 0.8, 0.0, False)
    return data, weather, units, policy, settings


def starts(result):
    return [e["hour"] for e in result.events if e["kind"] == "diesel_start"]


class PhysicsTests(unittest.TestCase):
    def run_case(self, f, noise=None):
        result = simulate(*f, noise or FixedNoise(), trace=True)
        self.assertTrue(result.valid, result.error)
        audit = audit_trace(result, f[0], f[2], f[4])
        self.assertTrue(audit["passed"], audit)
        return result

    def test_three_complete_hours_and_hour_boundary(self):
        r = self.run_case(fixture())
        self.assertEqual(starts(r)[0], 3)
        self.assertEqual(r.hourly[2]["diesel_kw"], 0)

    def test_access_wait_then_three_hours(self):
        r = self.run_case(fixture(access=[0] * 6 + [1] * 18))
        self.assertEqual(starts(r)[0], 9)
        self.assertEqual([e["hour"] for e in r.events if e["kind"] == "crew_arrival"][0], 6)

    def test_arrival_only_work_continues_indoors(self):
        r = self.run_case(fixture(access=[1, 0, 0] + [1] * 21))
        self.assertEqual(starts(r)[0], 3)

    def test_all_work_safe_pauses_without_reset(self):
        f = list(fixture(access=[1, 0, 0] + [1] * 21))
        f[4] = replace(f[4], personnel_mode="all_work_safe")
        r = self.run_case(f)
        self.assertEqual(starts(r)[0], 5)
        self.assertEqual(r.hourly[3]["diesel_states"][0]["prep"], 1)

    def test_preparation_and_four_hour_heat_parallel(self):
        f = list(fixture())
        f[4] = replace(f[4], thermal=ThermalParameters(0, 1, 1, 5, 1))
        r = self.run_case(f)
        self.assertEqual(starts(r)[0], 4)
        self.assertEqual(r.hourly[3]["diesel_kw"], 0)
        delay = r.delays[0]
        self.assertEqual(delay["prep_complete_hour"], 3)
        self.assertEqual(delay["temperature_ready_hour"], 4)
        self.assertEqual(sum(v for k, v in delay.items() if k.startswith("wait_")), delay["elapsed_hours"])

    def test_no_source_no_heating_no_self_start(self):
        f = list(fixture(pv=0))
        f[2] = dict(f[2], battery_energy=0)
        f[4] = replace(f[4], thermal=ThermalParameters(0.2, 1, 12, 5, -10))
        r = self.run_case(f)
        self.assertFalse(starts(r))
        self.assertTrue(all(h["heater_kw"] == 0 for h in r.hourly))
        self.assertEqual(r.loss_kwh, 24 * 30)

    def test_shared_container_heated_once(self):
        f = list(fixture(load=50))
        f[2] = dict(f[2], diesel=2)
        f[3] = replace(f[3], online_floor=2)
        f[4] = replace(f[4], container_ids=(0, 0), crews=2,
                       thermal=ThermalParameters(0, 1, 1, 5, 1))
        r = self.run_case(f)
        self.assertEqual(r.metrics["heater_kwh"], 4)
        self.assertEqual(starts(r)[:2], [4, 4])

    def test_single_crew_queue(self):
        f = list(fixture(load=50))
        f[2] = dict(f[2], diesel=2)
        f[3] = replace(f[3], online_floor=2)
        r = self.run_case(f)
        self.assertEqual(starts(r)[:2], [3, 7])
        self.assertGreater(r.metrics["wait_crew_unit_hours"], 0)

    def test_battery_supplies_actual_load_and_exhausts(self):
        f = list(fixture(pv=0))
        f[2] = dict(f[2], diesel=0, battery_energy=1)
        f[3] = replace(f[3], online_floor=0)
        r = self.run_case(f)
        self.assertAlmostEqual(r.metrics["battery_discharge_kwh"], 50 * 0.8 * 0.9)
        self.assertAlmostEqual(r.loss_kwh, 24 * 30 - 36)

    def test_actual_pcs_failure_restricts_battery(self):
        f = list(fixture(pv=0))
        f[2] = dict(f[2], diesel=0, pcs=1)
        f[3] = replace(f[3], online_floor=0)
        failures = copy.deepcopy(f[4].failures)
        failures["pcs"]["rate_per_hour"] = 0.1
        f[4] = replace(f[4], failures=failures)
        r = self.run_case(f, FixedNoise({("pcs_run", 0, 0): 0}))
        self.assertEqual(r.hourly[1]["pcs_available_kw"], 0)
        self.assertEqual(r.hourly[1]["discharge_kw"], 0)

    def test_wind_protection_not_mechanical_damage(self):
        f = list(fixture())
        f[1] = replace(f[1], icing_active=np.ones(24), wind_protective_stop=np.ones(24))
        f[2] = dict(f[2], wind=1)
        r = self.run_case(f)
        self.assertEqual(r.metrics["wind_failures"], 0)
        self.assertTrue(all(row["wind_available_kw"] == 0 for row in r.hourly))

    def test_hard_protection_cannot_be_bypassed(self):
        f = list(fixture())
        f[1] = replace(f[1], hard_stop=np.ones(24))
        f[2] = dict(f[2], wind=1)
        f[4] = replace(f[4], wind_mode="operate_with_hazard_multiplier")
        r = self.run_case(f)
        self.assertTrue(all(row["wind_available_kw"] == 0 for row in r.hourly))

    def test_extreme_hazard_multiplier_only_when_not_protected(self):
        f = list(fixture())
        f[1] = replace(f[1], extreme_hazard=np.ones(24), wind_protective_stop=np.ones(24))
        f[2] = dict(f[2], wind=1)
        failures = copy.deepcopy(f[4].failures)
        failures["wind"]["rate_per_hour"] = .1
        f[4] = replace(f[4], failures=failures)
        shock = FixedNoise({("wind_run", 0, 0): .15})
        protected = self.run_case(f, shock)
        f[4] = replace(f[4], wind_mode="operate_with_hazard_multiplier")
        exposed = self.run_case(f, shock)
        self.assertEqual(protected.metrics["wind_failures"], 0)
        self.assertEqual(exposed.metrics["wind_failures"], 1)

    def test_start_outcome_cannot_change_pre_demand_dispatch(self):
        base = self.run_case(fixture())
        changed = self.run_case(fixture(), FixedNoise({("diesel_start", 3, 0): 0}))
        self.assertEqual(base.hourly[3]["actions"], changed.hourly[3]["actions"])

    def test_r0_ignores_personnel_and_r1_has_three_hour_delay(self):
        f = list(fixture(load=50, access=[0] * 24))
        f[2] = dict(f[2], diesel=2)
        f[3] = replace(f[3], online_floor=2)
        f[4] = replace(f[4], physical_mode="R0")
        r0 = self.run_case(f)
        self.assertEqual(starts(r0)[:2], [0, 0])
        f[4] = replace(f[4], physical_mode="R1")
        r1 = self.run_case(f)
        self.assertEqual(starts(r1)[:2], [3, 3])

    def test_storm_does_not_stop_online_diesel(self):
        r = self.run_case(fixture(access=[1] * 4 + [0] * 20))
        self.assertTrue(all(row["diesel_states"][0]["running"] for row in r.hourly[4:]))

    def test_standby_has_no_running_failure_exposure(self):
        f = list(fixture(hours=100))
        f[3] = replace(f[3], online_floor=0)
        forced = {("diesel_run", h, 0): 0.0 for h in range(100)}
        r = self.run_case(f, FixedNoise(forced))
        self.assertEqual(r.metrics["diesel_run_failures"], 0)
        self.assertEqual(r.metrics["diesel_online_hours"], 0)

    def test_start_failure_locks_until_calendar_repair(self):
        r = self.run_case(fixture(), FixedNoise({("diesel_start", 3, 0): 0}))
        self.assertEqual(starts(r)[0], 5)
        self.assertEqual(r.metrics["start_failures"], 1)
        self.assertFalse(r.hourly[4]["actions"]["start_requested"])

    def test_running_failure_hour_end_repair_exact_boundaries(self):
        r = self.run_case(fixture(), FixedNoise({("diesel_run", 3, 0): 0}))
        self.assertGreater(r.hourly[3]["diesel_kw"], 0)
        self.assertEqual(r.hourly[4]["diesel_kw"], 0)
        self.assertEqual(r.hourly[5]["diesel_kw"], 0)
        self.assertEqual(starts(r)[:2], [3, 6])

    def test_repair_weather_does_not_pause_calendar_clock(self):
        r = self.run_case(fixture(access=[1] * 4 + [0] * 5 + [1] * 15),
                          FixedNoise({("diesel_run", 3, 0): 0}))
        self.assertEqual([e["hour"] for e in r.events if e["kind"] == "diesel_repair_complete"], [6])
        self.assertEqual(starts(r)[:2], [3, 9])

    def test_repeated_faults_do_not_reset_soc(self):
        f = fixture(pv=0)
        r = self.run_case(f, FixedNoise({("diesel_run", 5, 0): 0, ("diesel_run", 12, 0): 0}))
        for previous, current in zip(r.hourly, r.hourly[1:]):
            self.assertAlmostEqual(previous["energy_end_kwh"], current["energy_start_kwh"])
        self.assertNotEqual(r.hourly[13]["energy_start_kwh"], r.hourly[0]["energy_start_kwh"])

    def test_future_noise_does_not_change_decision_prefix(self):
        base = self.run_case(fixture())
        changed = self.run_case(fixture(), FixedNoise({("diesel_run", 10, 0): 0}))
        self.assertEqual(base.hourly[:11], changed.hourly[:11])

    def test_hidden_repair_duration_not_in_observation(self):
        self.assertNotIn("repair_until", Observation.__dataclass_fields__)
        f = list(fixture())
        a = self.run_case(f, FixedNoise({("diesel_run", 4, 0): 0}))
        f[4] = replace(f[4], diesel_repair_mean_hours=10)
        b = self.run_case(f, FixedNoise({("diesel_run", 4, 0): 0}))
        self.assertEqual(a.hourly[5]["actions"], b.hourly[5]["actions"])

    def test_min_down_separate_from_preparation(self):
        f = list(fixture())
        f[4] = replace(f[4], min_down_hours=5)
        r = self.run_case(f, FixedNoise({("diesel_start", 3, 0): 0}))
        self.assertEqual(starts(r)[0], 8)

    def test_minimum_output_without_dump_can_fail_policy(self):
        f = list(fixture(load=0, pv=0))
        f[2] = dict(f[2], battery_energy=0)
        result = simulate(*f, FixedNoise(), trace=True)
        self.assertFalse(result.valid)
        self.assertIn("unmodelled dump", result.error)

    def test_negative_warmup_does_not_mark_all_devices_failed(self):
        f = list(fixture())
        f[4] = replace(f[4], warmup_hours=12)
        r = self.run_case(f, FixedNoise(warmup=12))
        self.assertEqual(r.metrics["initial_state"]["online"], 1)
        self.assertGreater(r.hourly[0]["pv_available_kw"], 0)

    def test_thermal_exact_decay_and_zero_ua(self):
        p = ThermalParameters(0.5, 2, 10, 5, 0)
        self.assertAlmostEqual(p.step(10, -10, 0), -10 + 20 * np.exp(-0.25))
        self.assertAlmostEqual(ThermalParameters(0, 2, 10, 5, 0).step(0, -10, 4), 2)

    def test_corrupted_trace_rejected(self):
        f = fixture()
        r = self.run_case(f)
        r.hourly[10]["heater_kw"] += 1
        self.assertFalse(audit_trace(r, f[0], f[2], f[4])["passed"])


class ProbabilityAndPlanningTests(unittest.TestCase):
    def test_module_prefix_and_horizon_noise_invariance(self):
        a = PrimitiveNoise(17, 0, 10, {k: 1 for k in COMPONENTS}, 3)
        b = PrimitiveNoise(17, 0, 20, {k: 3 for k in COMPONENTS}, 6)
        for mechanism in a.arrays:
            for h in range(-3, 10):
                self.assertEqual(a.uniform(mechanism, h, 0), b.uniform(mechanism, h, 0))
        self.assertNotEqual(a.uniform("diesel_run", 0, 0), a.uniform("diesel_start", 0, 0))

    def test_geometric_mean_and_support(self):
        u = (np.arange(100000) + 0.5) / 100000
        r = [repair_duration(float(x), 37) for x in u]
        self.assertAlmostEqual(np.mean(r), 37, delta=0.003)
        self.assertGreaterEqual(min(r), 1)

    def test_partial_atom_cvar(self):
        risk = risk_summary([0, 10, 100], 0.5, [0.6, 0.3, 0.1])
        self.assertAlmostEqual(risk["eens_kwh"], 13)
        self.assertAlmostEqual(risk["cvar_kwh"], 26)
        self.assertEqual(risk["var_kwh"], 0)
        risk = risk_summary([0, 10, 100], 0.75, [0.6, 0.3, 0.1])
        self.assertAlmostEqual(risk["cvar_kwh"], 46)
        self.assertEqual(risk["var_kwh"], 10)

    def test_zero_var_identity_and_original_weights(self):
        losses = [0] * 99 + [100]
        risk = risk_summary(losses, 0.95)
        self.assertAlmostEqual(risk["cvar_kwh"], 20 * risk["eens_kwh"])
        lower, upper = partial_bounds([100], 100, 200, 0.95)
        self.assertAlmostEqual(lower["eens_kwh"], 1)
        self.assertAlmostEqual(upper["eens_kwh"], 199)

    def test_grid_accelerated_exhaustive_and_resume_match(self):
        data, weather, units, policy, settings = fixture(hours=24)
        config = dict(grid={k: [units[k]] for k in COMPONENTS},
                      policies=[vars(policy), vars(replace(policy, name="zero_floor", online_floor=0))],
                      reliability=dict(samples=3, seed=11, validation_samples=0, validation_seed=22,
                                       alpha=.95, eens_limit_kwh=30, cvar_limit_kwh=100),
                      planning=dict(early_risk_screen=False, investment_bound_pruning=False))
        config["grid"].update(pv=[0, 1], diesel=[0, 1], battery_energy=[0, 4])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            baseline = plan_grid(data, weather, config, settings, root)
            config["planning"] = dict(early_risk_screen=True, investment_bound_pruning=True)
            accelerated = plan_grid(data, weather, config, settings, root)
            self.assertEqual(baseline["incumbent"], accelerated["incumbent"])
            self.assertEqual(accelerated["relative_gap"], 0)
            self.assertTrue(all(not x["physical_infeasibility_proven"] for x in accelerated["visited"]))
            for r in baseline["evaluations"]:
                if r.get("complete"):
                    self.assertLessEqual(r["investment_cost_yuan"], r["objective_yuan"])
            partial = plan_grid(data, weather, config, settings, root, max_designs=1)
            self.assertTrue(partial["unresolved_designs"])
            self.assertNotEqual(partial["status"], "sample_optimal_within_policy_library_gap")

    def test_checkpoint_truncated_last_record_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.jsonl"
            path.write_text('{"key":"a","loss":2}\n{"key":')
            archive = PathArchive(path)
            archive.save(dict(key="b", loss=3))
            self.assertEqual(set(PathArchive(path).records), {"a", "b"})

    def test_formal_config_fails_without_confirmed_sources(self):
        raw = (ROOT / "config/zhongshan_recovery_stress.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.toml"
            path.write_text(raw.replace("synthetic = true", "synthetic = false"))
            with self.assertRaisesRegex(ValueError, "Formal run"):
                read_recovery_config(path)

    def test_missing_thermal_parameters_fail(self):
        raw = (ROOT / "config/zhongshan_recovery_stress.toml").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.toml"
            path.write_text(raw.replace("c_kwh_per_k = 1.5\n", ""))
            with self.assertRaises(TypeError):
                read_recovery_config(path)

    def test_weather_requires_timezone_and_exact_hourly_alignment(self):
        _, weather, *_ = fixture()
        with self.assertRaises(ValueError):
            replace(weather, timestamps=tuple(t[:-6] for t in weather.timestamps))
        with self.assertRaises(ValueError):
            replace(weather, access_safe=np.full(24, .5))

    def test_zhongshan_loader_costs_and_actual_temperature(self):
        config = read_recovery_config(ROOT / "config/zhongshan_recovery_stress.toml")
        if not Path(config["data_root"]).exists():
            self.skipTest("External station CSV unavailable")
        data, weather, _ = load_inputs(config)
        self.assertEqual(data.module_sizes["diesel"], 100)
        self.assertEqual(data.unit_bounds["diesel"], (0, 15))
        self.assertEqual(data.annual_cost_per_unit["diesel"], 50000)
        self.assertAlmostEqual(weather.ambient_c[0], -2.9)
        self.assertEqual(weather.metadata["available_source_hours"], 8784)
        self.assertEqual(weather.metadata["weather_label_status"], "assumed_stress_case")
        self.assertFalse(data.input_metadata["continuous_cyclic_soc"])
        # The declared residual duration counts hours AFTER the final extreme hour.
        extreme = weather.extreme_hazard
        for h in range(1, data.hours - 13):
            if extreme[h - 1] and not extreme[h:h + 13].any():
                self.assertTrue(weather.icing_active[h:h + 12].all())
                self.assertFalse(weather.icing_active[h + 12])
                break
        else:
            self.fail("Expected a complete residual-icing interval in the station stress case")


if __name__ == "__main__":
    unittest.main()
