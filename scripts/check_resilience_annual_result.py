"""Reconcile a saved annual result and its audited warm start, without solving."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--reference',default='seed',help='Saved starting dispatch directory, relative to the run')
    args=parser.parse_args()
    run=args.run.resolve()
    reference=run/args.reference
    s=json.loads((run/'summary.json').read_text());c=s['resolved_config'];cap=s['selected']
    seed=json.loads((reference/'summary.json').read_text())
    definition=json.loads((run/'run_definition.json').read_text())
    manifest=json.loads((run/'scenario_manifest.json').read_text())
    frame=pd.read_csv(run/'hourly_dispatch.csv')
    data=np.load(run/'dispatch.npz',allow_pickle=False)
    initial=np.load(reference/'dispatch.npz',allow_pickle=False)
    values={}
    for line in (run/'solution.sol').read_text().splitlines():
        if line and not line.startswith('#'):
            name,value=line.split();values[name]=float(value)
    for name,digest in definition['source_hashes'].items():
        assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    assert cap and s['audit']['passed'] and s['hours']==8760
    assert not s['post_solution_rounding'] and s['fixed_modules'] is None
    steps=dict(wind_kw=100,pv_kw=100,diesel_units=1,battery_kwh=50,pcs_kw=50,ups_kwh=50,ups_kw=50)
    for k,step in steps.items():
        assert cap[k]==step*s['modules'][k]
        assert values['modules_'+k]==int(values['modules_'+k])==s['modules'][k]
    prices=dict(wind_kw=c['wind_yuan_per_kw'],pv_kw=c['pv_yuan_per_kw'],diesel_units=100*c['diesel_yuan_per_kw'],
        battery_kwh=c['battery_yuan_per_kwh'],pcs_kw=c['pcs_yuan_per_kw'],ups_kwh=c['ups_yuan_per_kwh'],ups_kw=c['ups_yuan_per_kw'])
    capex={k:cap[k]*v for k,v in prices.items()}
    errors={}
    def check(name,value): errors[name]=max(errors.get(name,0.),float(np.max(np.abs(value),initial=0.)))
    check('investment',sum(capex.values())-s['investment_yuan'])
    check('total_cost',s['objective_yuan']-s['investment_yuan']-s['expected_operation_yuan'])
    blocks={b['block']:b for w in manifest for b in w['branches']}
    baseline_weight=np.ones(s['hours']);outside=np.ones(s['hours'],dtype=bool);metadata={}
    for w in manifest:
        a,z=w['start_hour'],w['stop_hour']
        outside[a:z]=False
        assert abs(sum(b['conditional_weight'] for b in w['branches'])-1)<1e-10
        for b in w['branches']:
            if b['block']==0:baseline_weight[a:z]=b['conditional_weight']
            else:metadata[b['block']]=w
    totals={k:0. for k in ('wind','pv','diesel','online_unit_hours','battery_charge','battery_discharge',
                           'ups_charge','ups_discharge','interrupt_adjust','shift_delayed','loss')}
    block_losses={};events=[];seed_difference=0.
    decay=math.exp(-c['thermal_ua_kw_per_k']/c['thermal_c_kwh_per_k'])
    gain=(1-decay)/c['thermal_ua_kw_per_k']
    for i,b in blocks.items():
        keys=('annual_hours','wind','pv','core_kw','rigid_kw','flex_interruptible_kw','flex_shiftable_kw','core_main',
              'rigid_shed','interrupt_shed','interrupt_adjust','ups_on','battery_charge','battery_discharge','battery_energy',
              'ups_charge','ups_discharge','ups_energy','pcs_available','ups_available','renewable_bus_fault','bus_live',
              'diesel_power','online','run_fail','start_fail','temperature','heater','ambient_c')
        a={k:data[f's{i}_{k}'] for k in keys}
        hours=a['annual_hours'];rows=frame[frame.block==i].sort_values('annual_hour')
        assert np.array_equal(rows.annual_hour.to_numpy(),hours)
        weights=baseline_weight if i==0 else np.full(len(hours),b['conditional_weight'])
        delayed=rows.shift_served_kw.to_numpy()-np.array([values.get(f's{i}_shift_service[{t},{t}]',values[f's0_shift_service[{t},{t}]']) for t in hours])
        loss=a['rigid_shed']+a['interrupt_shed']+a['flex_shiftable_kw']*a['ups_on']
        for j,t in enumerate(hours):
            for arrival in range(max(0,t-c['shift_window_hours']+1),t+1):
                if min(arrival+c['shift_window_hours']-1,s['hours']-1)==t:
                    loss[j]+=values.get(f's{i}_shift_late_shed[{arrival}]',values[f's0_shift_late_shed[{arrival}]'])
        block_losses[i]=loss
        diesel=a['diesel_power'].sum(axis=0)
        service=a['rigid_kw']-a['rigid_shed']+a['flex_interruptible_kw']-a['interrupt_adjust']-a['interrupt_shed']+rows.shift_served_kw.to_numpy()
        check('power_balance',a['wind']+a['pv']+diesel+a['battery_discharge']-a['core_main']-service-a['battery_charge']-a['heater'].sum(axis=0)-a['ups_charge'])
        check('core_balance',a['core_main']+a['ups_discharge']-a['core_kw'])
        check('battery_recurrence',np.diff(a['battery_energy'])-c['battery_efficiency']*a['battery_charge']+a['battery_discharge']/c['battery_efficiency'])
        check('ups_recurrence',np.diff(a['ups_energy'])-c['ups_efficiency']*a['ups_charge']+a['ups_discharge']/c['ups_efficiency'])
        check('temperature_recurrence',a['temperature'][:cap['diesel_units'],1:]-decay*a['temperature'][:cap['diesel_units'],:-1]-(1-decay)*a['ambient_c']-gain*a['heater'][:cap['diesel_units']])
        check('battery_soc_bounds',np.maximum(0,c['battery_min_soc']*cap['battery_kwh']-a['battery_energy'])+np.maximum(0,a['battery_energy']-cap['battery_kwh']))
        check('ups_soc_bounds',np.maximum(0,c['ups_min_soc']*cap['ups_kwh']-a['ups_energy'])+np.maximum(0,a['ups_energy']-c['ups_standby_soc']*cap['ups_kwh']))
        check('pcs_outage',(a['battery_charge']+a['battery_discharge'])*(~a['pcs_available']))
        check('ups_outage',(a['ups_charge']+a['ups_discharge'])*(~a['ups_available']))
        check('renewable_outage',(a['wind']+a['pv'])*a['renewable_bus_fault'])
        check('gfm_active_power',np.maximum(0,c['gfm_min_power_kw']-np.maximum.reduce([a['battery_charge'],a['battery_discharge'],a['diesel_power'].max(axis=0)]))*a['bus_live'])
        if i:
            w=metadata[i]
            for name in ('battery_energy','ups_energy'):
                check('window_state_links',a[name][[0,-1]]-data[f's0_{name}'][[w['start_hour'],w['stop_hour']]])
        energies={k:a[k] for k in ('wind','pv','battery_charge','battery_discharge','ups_charge','ups_discharge','interrupt_adjust')}
        energies.update(diesel=diesel,online_unit_hours=a['online'].sum(axis=0),shift_delayed=delayed,loss=loss)
        for k,v in energies.items():totals[k]+=float(weights@v)
        for name in ('wind','pv','battery_energy','ups_energy','diesel_power','online','run_fail','start_fail'):
            v=a[name][:cap['diesel_units']] if a[name].ndim==2 else a[name]
            ref=initial[f's{i}_{name}']
            seed_difference=max(seed_difference,float(np.max(np.abs(v-ref),initial=0.)))
        if i:
            w=metadata[i];fault=(hours>=w['start_hour']+36)&(hours<w['start_hour']+48)
            events.append(dict(window=w['window'],name=b['name'],weight=b['conditional_weight'],
                regular_loss_kwh=float(loss.sum()),ups_kwh=float(a['ups_discharge'].sum()),
                diesel_event_kwh=float(diesel[fault].sum()),battery_event_kwh=float(a['battery_discharge'][fault].sum()),
                interrupt_event_kwh=float(a['interrupt_adjust'][fault].sum()),
                actual_run_failures=float(a['run_fail'].sum()),actual_start_failures=float(a['start_fail'].sum())))
    costs=dict(grid=0.,diesel=c['diesel_yuan_per_kwh']*(totals['diesel']+c['diesel_idle_equivalent_kw']*totals['online_unit_hours']),
               load_loss=c['loss_yuan_per_kwh']*totals['loss'],flex_adjustment=c['flex_adjustment_yuan_per_kwh']*(totals['interrupt_adjust']+totals['shift_delayed']),
               battery_throughput=c['battery_cycle_yuan_per_kwh']*(totals['battery_charge']+totals['battery_discharge']))
    for k,v in costs.items():check('cost_'+k,v-s['mean_cost_components_yuan'][k])
    check('eens',totals['loss']-s['eens_kwh'])
    outside_loss=float(block_losses[0][outside].sum())
    check('outside_window_loss',outside_loss-s['outside_window_loss_kwh'])
    # Deterministic loss outside the windows shifts every annual outcome,
    # hence it is included once in both EENS and the annual CVaR upper bound.
    cvar=outside_loss
    for w in manifest:
        q=[];weights=[]
        for b in w['branches']:
            values_in_window=block_losses[b['block']]
            if b['block']==0:values_in_window=values_in_window[w['start_hour']:w['stop_hour']]
            q.append(float(values_in_window.sum()));weights.append(b['conditional_weight'])
        q=np.array(q);weights=np.array(weights)
        cvar+=min(eta+float(weights@np.maximum(0,q-eta))/(1-c['alpha']) for eta in np.r_[0,q])
    check('cvar_upper_bound',cvar-s['cvar_upper_bound_kwh'])
    after_initial=data['s0_annual_hours']>=48
    check('battery_terminal',max(0,data['s0_battery_energy'][0]-data['s0_battery_energy'][-1]))
    check('ups_terminal',data['s0_ups_energy'][0]-data['s0_ups_energy'][-1])
    assert max(errors.values())<1e-5,errors
    result=dict(passed=True,max_residual=max(errors.values()),checks=errors,investment_components_yuan=capex,
        weighted_annual_energy_kwh=totals,costs_yuan=costs,seed_dispatch_max_difference=seed_difference,
        objective_improvement_from_seed_yuan=seed['objective_yuan']-s['objective_yuan'],
        reference_run=str(reference),outside_window_loss_kwh=outside_loss,
        outside_loss_hours=[dict(annual_hour=int(t),loss_kwh=float(block_losses[0][t]))
            for t in np.flatnonzero(outside & (block_losses[0]>1e-6))],
        battery_charge_after_hour48_kwh=float(data['s0_battery_charge'][after_initial].sum()),
        battery_discharge_after_hour48_kwh=float(data['s0_battery_discharge'][after_initial].sum()),
        battery_min_soc=float(data['s0_battery_energy'].min()/cap['battery_kwh']),
        gap_percent=100*s['mip_gap'],absolute_bound_difference_yuan=s['objective_yuan']-s['objective_bound_yuan'],
        scenario_metrics=events,scope='saved annual dispatch reconciliation; no new optimization')
    (run/'checked_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(events).to_csv(run/'checked_scenario_metrics.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k!='scenario_metrics'},ensure_ascii=False,indent=2))
    print(json.dumps([e for e in events if e['name'].endswith('storm_renewable_bus')],indent=2))


if __name__=='__main__':main()
