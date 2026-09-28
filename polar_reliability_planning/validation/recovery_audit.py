"""Independent recomputation from recorded boundary states and power traces."""
import numpy as np


def audit_trace(result, data, units, settings):
    violations = []
    residuals = {"power_kw": 0.0, "soc_kwh": 0.0, "thermal_c": 0.0, "state_continuity_kwh": 0.0}
    mapping = settings.container_ids[:units["diesel"]]
    box_ids = sorted(set(mapping))
    mapping = [box_ids.index(i) for i in mapping]
    energy_cap = units["battery_energy"] * data.module_sizes["battery_energy"]
    last_energy = None
    last_temp = None
    prep_hours = 0 if settings.physical_mode == "R0" else settings.prep_hours
    loss, operating = 0.0, 0.0
    for row in result.hourly:
        h = row["hour"]
        def check(condition, name):
            if not condition:
                violations.append({"hour": h, "constraint": name})
        balance = (row["renewable_used_kw"] + row["diesel_kw"] + row["discharge_kw"] + row["shed_kw"]
                   - row["load_kw"] - row["charge_kw"] - row["heater_kw"])
        residuals["power_kw"] = max(residuals["power_kw"], abs(balance))
        predicted = row["energy_start_kwh"] + data.efficiency * row["charge_kw"] - row["discharge_kw"] / data.efficiency
        residuals["soc_kwh"] = max(residuals["soc_kwh"], abs(predicted - row["energy_end_kwh"]))
        if last_energy is not None:
            residuals["state_continuity_kwh"] = max(residuals["state_continuity_kwh"], abs(last_energy - row["energy_start_kwh"]))
            check(np.allclose(last_temp, row["temperatures_start"], atol=1e-8, rtol=0), "temperature_continuity")
        last_energy, last_temp = row["energy_end_kwh"], row["temperatures_end"]
        check(data.soc_min * energy_cap - 1e-6 <= last_energy <= data.soc_max * energy_cap + 1e-6, "soc_bounds")
        check(min(row["charge_kw"], row["discharge_kw"]) < 1e-7, "charge_discharge_mutex")
        check(max(row["charge_kw"], row["discharge_kw"]) <= row["pcs_available_kw"] + 1e-7, "pcs_rating")
        check(0 <= row["shed_kw"] <= row["load_kw"] + 1e-7, "user_load_shedding_bounds")
        check(0 <= row["renewable_used_kw"] <= row["wind_available_kw"] + row["pv_available_kw"] + 1e-7, "renewable_availability")
        crew_limit = units["diesel"] if settings.physical_mode in ("R0", "R1") else settings.crews
        check(row["crews_onsite"] <= crew_limit, "crew_capacity")
        check(not row["actions"]["dispatched"] or row["access_safe"], "dispatch_access")
        check(abs(sum(row["heaters"]) - row["heater_kw"]) < 1e-7, "shared_box_heater_accounting")
        online = sum(s["running"] for s in row["diesel_states"])
        check(online * data.module_sizes["diesel"] * settings.min_output_fraction - 1e-6 <= row["diesel_kw"] <= online * data.module_sizes["diesel"] + 1e-6, "diesel_output")
        for i in row["actions"]["start_requested"]:
            before = row["observation_states"][i]
            check(before[0] and before[4] >= prep_hours and before[6] == 0, "start_health_preparation_min_down")
            check(before[3] or i in row["actions"]["dispatched"], "start_crew_present")
            check(settings.physical_mode != "R3" or row["temperatures_start"][mapping[i]] >= settings.thermal.ready_temperature_c - 1e-8, "start_boundary_temperature")
        for i in row["actions"]["stopped"]:
            check(row["observation_states"][i][5] == 0, "minimum_up_time")
        for c, temperature in enumerate(row["temperatures_start"]):
            check(0 <= row["heaters"][c] <= settings.thermal.heater_max_kw + 1e-7, "heater_rating")
            running = sum(s["running"] for i, s in enumerate(row["diesel_states"]) if mapping[i] == c)
            p = settings.thermal
            # Independent closed form, not ThermalParameters.step().
            heat = p.efficiency * row["heaters"][c] + running * p.retained_generator_heat_kw
            if p.ua_kw_per_k:
                decay = np.exp(-p.ua_kw_per_k / p.c_kwh_per_k)
                expected = decay * temperature + (1 - decay) * (row["ambient_c"] + heat / p.ua_kw_per_k)
            else:
                expected = temperature + heat / p.c_kwh_per_k
            residuals["thermal_c"] = max(residuals["thermal_c"], abs(expected - row["temperatures_end"][c]))
        if row["measured"]:
            loss += row["shed_kw"]
            operating += (row["diesel_kw"] * data.fuel_cost_per_kwh
                          + len(row["actions"]["start_requested"]) * settings.startup_cost_yuan
                          + (row["charge_kw"] + row["discharge_kw"]) * settings.storage_cost_yuan_per_kwh)
    if not result.hourly:
        violations.append({"constraint": "missing_trace"})
    if abs(loss - result.loss_kwh) > 1e-6 or abs(operating - result.operating_cost_yuan) > 1e-5:
        violations.append({"constraint": "cost_or_loss_reconstruction"})
    for name, value in residuals.items():
        if value > 1e-6:
            violations.append({"constraint": name, "residual": value})
    return dict(passed=result.valid and not violations, violations=violations, residuals=residuals,
                checked_hours=len(result.hourly), reconstructed_loss_kwh=loss,
                reconstructed_operating_cost_yuan=operating,
                coverage="selected replay trace; all simulated paths also check power/SOC feasibility")
