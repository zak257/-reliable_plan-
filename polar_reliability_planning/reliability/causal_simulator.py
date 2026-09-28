"""Continuous hourly execution of a frozen recovery policy.

Power is an average over [h,h+1); preparation, SOC and temperatures are states
at h. Run failures apply at h+1; startup failures apply at h. Physical dispatch
failure is reported as failure of this policy, never as capacity infeasibility.
"""
from dataclasses import dataclass
import math

from ..data.cap_plan_loader import COMPONENTS
from ..recovery.diesel_state_machine import DieselState
from ..recovery.personnel import advance_preparation
from ..scenario_generation.primitive_noise import repair_duration
from .recovery_policy import Observation


@dataclass
class SimulationResult:
    valid: bool
    loss_kwh: float
    operating_cost_yuan: float
    metrics: dict
    hourly: list
    events: list
    delays: list
    error: str | None = None


def simulate(data, weather, units, policy, settings, noise, trace=False):
    units = data.validate_units(units)
    if data.dt_hours != 1 or weather.hours != data.hours:
        raise ValueError("Recovery simulation requires aligned one-hour inputs")
    if not data.soc_min <= settings.initial_soc <= data.soc_max:
        raise ValueError("Initial SOC outside battery bounds")
    if len(settings.container_ids) < units["diesel"]:
        raise ValueError("Missing diesel-to-container mapping")
    if noise.warmup != settings.warmup_hours:
        raise ValueError("Noise and simulator warmup differ")
    nd = units["diesel"]
    mapping = settings.container_ids[:nd]
    containers = sorted(set(mapping))
    box_index = {c: k for k, c in enumerate(containers)}
    mapping = [box_index[c] for c in mapping]
    nc = len(containers)
    states = [DieselState() for _ in range(nd)]
    temperatures = [settings.thermal.initial_temperature_c] * nc
    enabled = [settings.remote_heat] * nc
    other_repair = {k: [-10**15] * units[k] for k in ("wind", "pv", "pcs")}
    energy_cap = units["battery_energy"] * data.module_sizes["battery_energy"]
    energy_min, energy_max = data.soc_min * energy_cap, data.soc_max * energy_cap
    energy = settings.initial_soc * energy_cap
    diesel_kw = data.module_sizes["diesel"]
    eff = data.efficiency
    thermal = settings.thermal
    rows, events, delays = [], [], []
    open_delays = {}
    m = {k: 0.0 for k in ("shed_hours", "longest_shed_hours", "heater_kwh", "diesel_kwh",
         "diesel_fuel_kg", "battery_discharge_kwh", "battery_charge_kwh", "starts", "start_failures",
         "diesel_run_failures", "diesel_online_hours", "wind_failures", "pv_failures", "pcs_failures",
         "wind_protective_stop_hours", "wait_access_unit_hours", "wait_crew_unit_hours",
         "wait_prep_unit_hours", "wait_heat_unit_hours", "unserved_heater_command_kwh",
         "max_balance_residual_kw", "max_soc_residual_kwh")}
    loss, cost, consecutive = 0.0, 0.0, 0
    prep_required = 0 if settings.physical_mode == "R0" else settings.prep_hours
    heat_required = settings.physical_mode == "R3"
    crew_limit = nd if settings.physical_mode in ("R0", "R1") else settings.crews
    run_probability = -math.expm1(-1 / settings.diesel_mttf_hours)
    previous_protect = False

    def event(hour, kind, device=-1, **extra):
        if trace:
            events.append(dict(hour=hour, measured=hour >= 0, kind=kind, device=device, **extra))

    def close_delay(i, h, outcome):
        if i in open_delays:
            d = open_delays.pop(i)
            d.update(end_hour=h, outcome=outcome, elapsed_hours=h - d["request_hour"])
            if trace:
                delays.append(d)

    initial = None
    for h in range(-settings.warmup_hours, data.hours):
        wi = h % data.hours  # explicitly declared cyclic weather splice for warmup
        measured = h >= 0
        load = float(weather.load_kw[wi])
        ambient = float(weather.ambient_c[wi])
        access = bool(weather.access_safe[wi]) or settings.physical_mode in ("R0", "R1")
        pre_temp = temperatures.copy()
        pre_energy = energy
        # Boundary h: reveal completed repairs, retaining physical temperature.
        for i, st in enumerate(states):
            if not st.healthy and h >= st.repair_until:
                st.healthy = True
                st.prep = prep_required if settings.repair_handover_prepared else 0.0
                event(h, "diesel_repair_complete", i, temperature_c=temperatures[mapping[i]])
        for component, repairs in other_repair.items():
            for i, until in enumerate(repairs):
                if until == h:
                    event(h, component + "_repair_complete", i)
        if h == 0:
            initial = dict(energy_kwh=energy, temperatures_c=temperatures.copy(),
                           online=sum(s.running for s in states),
                           diesel_states=[vars(s).copy() for s in states])
        protect = bool(weather.hard_stop[wi] or
                       (settings.wind_mode == "protective_shutdown" and weather.wind_protective_stop[wi]))
        if protect != previous_protect:
            event(h, "wind_protection_enter" if protect else "wind_protection_exit")
            previous_protect = protect
        healthy_counts = {c: sum(until <= h for until in repair) for c, repair in other_repair.items()}
        wind_available = (healthy_counts["wind"] * data.module_sizes["wind"] * float(weather.wind_clean_pu[wi])
                          * float(weather.ice_power_factor[wi]) * (not protect))
        pv_available = healthy_counts["pv"] * data.module_sizes["pv"] * float(weather.pv_pu[wi])
        renewable = wind_available + pv_available
        pcs_kw = healthy_counts["pcs"] * data.module_sizes["pcs"]
        observation = Observation(h, load, renewable, energy / energy_cap if energy_cap else 0.0,
                                  access, tuple(s.observable() for s in states), tuple(temperatures))
        target = policy.target(observation, diesel_kw, nd)
        actions = {"requested": [], "dispatched": [], "start_requested": [], "stopped": []}
        # Stops respect minimum up time. Manual preparation resets after a stop.
        online = sum(s.running for s in states)
        for i in reversed(range(nd)):
            st = states[i]
            if online > target and st.running and st.min_up == 0:
                st.running, st.prep, st.onsite = False, 0.0, False
                st.min_down = settings.min_down_hours
                online -= 1
                actions["stopped"].append(i)
                event(h, "diesel_stop", i)
        pending_count = sum(s.pending for s in states)
        for i in reversed(range(nd)):
            st = states[i]
            if online + pending_count > target and st.pending:
                st.pending, st.onsite = False, False
                pending_count -= 1
                close_delay(i, h, "cancelled_by_policy")
        for i, st in enumerate(states):
            if online + pending_count < target and st.healthy and not st.running and not st.pending:
                st.pending = True
                pending_count += 1
                actions["requested"].append(i)
                open_delays[i] = dict(device=i, request_hour=h, arrival_hour=None,
                    prep_complete_hour=h if st.prep >= prep_required else None,
                    temperature_ready_hour=h if (not heat_required or temperatures[mapping[i]] >= thermal.ready_temperature_c - 1e-9) else None,
                    wait_access_hours=0, wait_crew_hours=0, wait_prep_hours=0, wait_heat_hours=0, wait_min_down_hours=0)
                event(h, "standby_request", i)
        # All dispatch requests are chosen before any startup outcome is revealed.
        crews_used = sum(s.onsite for s in states)
        for i, st in enumerate(states):
            if st.pending and st.healthy and not st.onsite and access and crews_used < crew_limit:
                st.onsite = True
                enabled[mapping[i]] = True
                crews_used += 1
                actions["dispatched"].append(i)
                open_delays[i]["arrival_hour"] = h
                event(h, "crew_arrival", i)
        for i, st in enumerate(states):
            if st.pending and st.healthy and st.onsite and st.prep >= prep_required and st.min_down == 0:
                if not heat_required or temperatures[mapping[i]] >= thermal.ready_temperature_c - 1e-9:
                    actions["start_requested"].append(i)
        # Demand noise is used once for each legal demand, never for blocked starts.
        for i in actions["start_requested"]:
            st = states[i]
            st.demands += 1
            st.pending, st.onsite = False, False
            if noise.uniform("diesel_start", h, i) < settings.start_failure_probability:
                duration = repair_duration(noise.uniform("diesel_repair", h, i), settings.diesel_repair_mean_hours, settings.repair_distribution)
                st.healthy, st.running, st.prep = False, False, 0.0
                st.repair_until = h + duration
                st.min_down = settings.min_down_hours
                event(h, "diesel_start_failure", i, repair_complete_hour=st.repair_until)
                close_delay(i, h, "start_failure")
                if measured:
                    m["start_failures"] += 1
            else:
                st.running = True
                st.min_up = settings.min_up_hours
                event(h, "diesel_start", i)
                close_delay(i, h, "started")
                if measured:
                    m["starts"] += 1
            if measured:
                cost += settings.startup_cost_yuan

        running = [i for i, st in enumerate(states) if st.running]
        online = len(running)
        running_in_box = [0] * nc
        active_box = [False] * nc
        for i, st in enumerate(states):
            running_in_box[mapping[i]] += st.running
            active_box[mapping[i]] |= st.running or st.pending
        heat_commands = [thermal.command(temperatures[c], ambient, running_in_box[c])
                         if heat_required and enabled[c] and (active_box[c] or policy.keep_standby_warm) else 0.0
                         for c in range(nc)]
        requested_heat = sum(heat_commands)
        discharge_limit = min(pcs_kw, max(0.0, (energy - energy_min) * eff))
        charge_limit = min(pcs_kw, max(0.0, (energy_max - energy) / eff))
        dg_max, dg_min = online * diesel_kw, online * diesel_kw * settings.min_output_fraction
        total_supply = renewable + dg_max + discharge_limit
        if policy.heat_priority:
            actual_heat = min(requested_heat, total_supply)
            served_load = min(load, max(0.0, total_supply - actual_heat))
        else:
            served_load = min(load, total_supply)
            actual_heat = min(requested_heat, max(0.0, total_supply - served_load))
        demand = served_load + actual_heat
        charge_target = min(charge_limit, policy.recharge_kw,
                            max(0.0, (policy.recharge_soc * energy_cap - energy) / eff))
        # Excess renewables charge freely; diesel charges only to policy target.
        dg = min(dg_max, max(dg_min, demand + charge_target - renewable))
        if dg > demand + charge_limit + 1e-7:
            return SimulationResult(False, loss, cost, m, rows, events, delays,
                                    f"hour {h}: minimum diesel output cannot be absorbed without an unmodelled dump load")
        ren_used = min(renewable, max(0.0, demand + charge_limit - dg))
        surplus = dg + ren_used - demand
        charge = max(0.0, surplus)
        discharge = max(0.0, -surplus)
        shed = max(0.0, load - served_load)
        if charge > charge_limit + 1e-7 or discharge > discharge_limit + 1e-7:
            raise AssertionError("Dispatch exceeds actual storage power/energy")
        energy = energy + eff * charge - discharge / eff
        heaters = []
        remaining_heat = actual_heat
        # Fixed box ordering is part of the frozen controller, not hindsight.
        for command in heat_commands:
            delivered = min(command, remaining_heat)
            heaters.append(delivered)
            remaining_heat -= delivered
        balance = ren_used + dg + discharge + shed - load - charge - actual_heat
        soc_residual = energy - pre_energy - eff * charge + discharge / eff
        if abs(balance) > 1e-6 or energy < energy_min - 1e-6 or energy > energy_max + 1e-6:
            raise AssertionError("Power/energy conservation failure")
        # Trace boundary state BEFORE preparation, heat and end-hour run failures.
        if trace:
            row = dict(hour=h, measured=measured, timestamp=weather.timestamps[wi], load_kw=load,
                ambient_c=ambient, access_safe=access, wind_protective_stop=protect,
                wind_available_kw=wind_available, pv_available_kw=pv_available,
                renewable_used_kw=ren_used, diesel_kw=dg, charge_kw=charge, discharge_kw=discharge,
                heater_kw=actual_heat, heater_command_kw=requested_heat, shed_kw=shed,
                energy_start_kwh=pre_energy, energy_end_kwh=energy, pcs_available_kw=pcs_kw,
                target_online=target, crews_onsite=sum(s.onsite for s in states), actions=actions,
                temperatures_start=pre_temp, heaters=heaters,
                diesel_states=[vars(s).copy() for s in states],
                observation_states=observation.states)
            rows.append(row)
        for i, st in enumerate(states):
            if st.pending:
                d = open_delays[i]
                # Exclusive blocking partition: overlapping physical processes
                # are recorded as milestone times, never added a second time.
                if not st.onsite:
                    block = "wait_access_hours" if not access else "wait_crew_hours"
                elif st.prep < prep_required:
                    block = "wait_prep_hours"
                elif heat_required and temperatures[mapping[i]] < thermal.ready_temperature_c - 1e-9:
                    block = "wait_heat_hours"
                else:
                    block = "wait_min_down_hours"
                d[block] += 1
                metric = block.replace("_hours", "_unit_hours")
                if measured and metric in m:
                    m[metric] += 1
            st.prep = advance_preparation(st.prep, prep_required, st.onsite, access, settings.personnel_mode)
            if st.pending and st.prep >= prep_required and open_delays[i]["prep_complete_hour"] is None:
                open_delays[i]["prep_complete_hour"] = h + 1
                event(h + 1, "preparation_complete", i)
        temperatures = [thermal.step(temperatures[c], ambient, heaters[c], running_in_box[c]) for c in range(nc)]
        for i, st in enumerate(states):
            if st.pending and (not heat_required or temperatures[mapping[i]] >= thermal.ready_temperature_c - 1e-9):
                if open_delays[i]["temperature_ready_hour"] is None:
                    open_delays[i]["temperature_ready_hour"] = h + 1
                    event(h + 1, "temperature_ready", i)
            st.min_up = max(0, st.min_up - 1)
            st.min_down = max(0, st.min_down - 1)
        if trace:
            row["temperatures_end"] = temperatures.copy()
        # Run failure is hidden until boundary h+1. Standby has zero exposure.
        for i in running:
            st = states[i]
            st.runtime += 1
            if noise.uniform("diesel_run", h, i) < run_probability:
                duration = repair_duration(noise.uniform("diesel_repair", h, i), settings.diesel_repair_mean_hours, settings.repair_distribution)
                st.healthy, st.running, st.prep, st.onsite = False, False, 0.0, False
                st.min_up = 0
                st.min_down = settings.min_down_hours
                st.repair_until = h + 1 + duration
                event(h + 1, "diesel_run_failure", i, repair_complete_hour=st.repair_until)
                if measured:
                    m["diesel_run_failures"] += 1
        for component, repairs in other_repair.items():
            law = settings.failures[component]
            multiplier = (settings.hazard_multiplier if component == "wind" and weather.extreme_hazard[wi] and not protect else 1.0)
            probability = -math.expm1(-law["rate_per_hour"] * multiplier)
            for i, until in enumerate(repairs):
                if until <= h and noise.uniform(component + "_run", h, i) < probability:
                    duration = repair_duration(noise.uniform(component + "_repair", h, i), law["repair_mean_hours"])
                    repairs[i] = h + 1 + duration
                    event(h + 1, component + "_failure", i, repair_complete_hour=repairs[i])
                    if measured:
                        m[component + "_failures"] += 1
        if measured:
            loss += shed
            cost += dg * data.fuel_cost_per_kwh + (charge + discharge) * settings.storage_cost_yuan_per_kwh
            consecutive = consecutive + 1 if shed > 1e-7 else 0
            m["shed_hours"] += shed > 1e-7
            m["longest_shed_hours"] = max(m["longest_shed_hours"], consecutive)
            m["heater_kwh"] += actual_heat
            m["unserved_heater_command_kwh"] += requested_heat - actual_heat
            m["diesel_kwh"] += dg
            m["diesel_fuel_kg"] += dg / settings.fuel_kwh_per_kg
            m["battery_discharge_kwh"] += discharge
            m["battery_charge_kwh"] += charge
            m["diesel_online_hours"] += online
            m["wind_protective_stop_hours"] += protect
            m["max_balance_residual_kw"] = max(m["max_balance_residual_kw"], abs(balance))
            m["max_soc_residual_kwh"] = max(m["max_soc_residual_kwh"], abs(soc_residual))
    for i in list(open_delays):
        close_delay(i, data.hours, "unfinished_at_horizon")
    m.update(initial_state=initial, final_state=dict(energy_kwh=energy, temperatures_c=temperatures,
             online=sum(s.running for s in states), diesel_states=[vars(s).copy() for s in states],
             other_repair_until=other_repair),
             energy_inventory_change_kwh=energy - initial["energy_kwh"],
             forecast_basis="current_measurements_only", dump_load_kw=0,
             repair_clock="calendar_reference", operating_cost_basis="same_executed_policy")
    return SimulationResult(True, loss, cost, m, rows, events, delays)
