"""Screen single renewable-module additions to this saved dispatch; no solve.

Only hours outside the startup and stress windows are changed. These arithmetic
checks are not a replacement for the original full-model feasibility audit.
"""
import json
from pathlib import Path

import numpy as np


run = Path(__file__).resolve().parent
summary = json.loads((run / 'summary.json').read_text())
cfg = summary['resolved_config']
manifest = json.loads((run / 'scenario_manifest.json').read_text())
data = np.load(run / 'dispatch.npz', allow_pickle=False)
hours = data['s0_annual_hours']
eligible = hours >= 48
for window in manifest:
    eligible &= ~((hours >= window['start_hour']) & (hours < window['stop_hour']))
assert not data['s0_extreme_weather'][eligible].any()

power = data['s0_diesel_power']
floor = cfg['diesel_min_output_kw'] * data['s0_diesel_connected']
unit_headroom = np.maximum(0., power - floor)
headroom = np.minimum(unit_headroom.sum(axis=0),
                      np.maximum(0., power.sum(axis=0) - data['s0_ups_charge']))
cases = []
for kind, price_key in [('wind', 'wind_yuan_per_kw'), ('pv', 'pv_yuan_per_kw')]:
    profile = data['s0_wind_clean_pu' if kind == 'wind' else 's0_pv_pu']
    available = profile * data['s0_bus_live'] * (~data['s0_renewable_bus_fault'])
    replacement = np.where(eligible, np.minimum(100. * available, headroom), 0.)
    # Allocate the reduction over connected engines without switching them off.
    fraction = np.divide(replacement, unit_headroom.sum(axis=0),
                         out=np.zeros_like(replacement), where=unit_headroom.sum(axis=0) > 0.)
    adjusted_power = power - unit_headroom * fraction
    renewable = data['s0_' + kind] + replacement
    new_capacity = summary['selected'][kind + '_kw'] + 100.
    assert np.min(adjusted_power - floor) > -1e-6
    assert np.max(adjusted_power - power) < 1e-6
    assert np.min(adjusted_power.sum(axis=0) - data['s0_ups_charge']) > -1e-6
    assert np.max(renewable - new_capacity * available) < 1e-6
    assert np.max(np.abs(adjusted_power.sum(axis=0) + replacement - power.sum(axis=0))) < 1e-6
    assert np.max(np.abs(replacement[~eligible]), initial=0.) == 0.
    investment = 100. * cfg[price_key]
    savings = float(replacement.sum()) * cfg['diesel_yuan_per_kwh']
    cases.append(dict(device=kind,added_integer_modules=1,added_capacity_kw=100,
        avoided_diesel_kwh=float(replacement.sum()),added_capex_yuan=investment,
        annual_fuel_savings_yuan=savings,net_objective_reduction_yuan=savings-investment))

result = dict(scope='Independent single-module marginal screens; no optimization or full-model audit. Do not add the two savings together.',
    unchanged='All startup/stress-window decisions, engine commitment, storage, loads, UPS and temperatures.',
    screened_hours=int(eligible.sum()),cases=cases,
    clean_profile_equivalent_hours={k:float(data['s0_'+k].sum()) for k in ('wind_clean_pu','pv_pu')},
    local_power_checks_passed=True)
(run / 'renewable_marginal_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
