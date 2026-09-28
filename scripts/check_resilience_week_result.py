"""Independently reconcile saved weekly dispatch, costs and risk; make review plots."""
from pathlib import Path
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


def main():
    run = Path(sys.argv[1]).resolve()
    summary = json.loads((run / 'summary.json').read_text())
    definition = json.loads((run / 'run_definition.json').read_text())
    manifest = json.loads((run / 'scenario_manifest.json').read_text())[0]
    cfg = summary['resolved_config']; cap = summary['selected']
    if not cap:
        raise ValueError('No feasible solution to inspect')
    frame = pd.read_csv(run / 'hourly_dispatch.csv')
    inputs = pd.read_csv(run / 'input_168h.csv')
    arrays = np.load(run / 'dispatch.npz', allow_pickle=False)
    values = {}
    for line in (run / 'solution.sol').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        key, value = line.split()
        values[key] = float(value)
    for source, digest in definition['source_hashes'].items():
        assert hashlib.sha256((run / 'source_snapshot' / source).read_bytes()).hexdigest() == digest
    assert summary['solver_status'] == 2 and summary['economic_domain_certified']
    assert summary['post_solution_rounding'] is False
    for key, count in summary['modules'].items():
        assert values['modules_' + key] == count and count == int(count)
    assert abs(summary['objective_yuan'] - summary['investment_yuan'] - summary['expected_operation_yuan']) < 1e-5
    steps = {'wind_kw':100,'pv_kw':100,'diesel_units':1,'battery_kwh':50,'pcs_kw':50,'ups_kwh':50,'ups_kw':50}
    for key in cap:
        assert abs(cap[key] - steps[key]*summary['modules'][key]) < 1e-8
    investment = (cap['wind_kw']*cfg['wind_yuan_per_kw'] + cap['pv_kw']*cfg['pv_yuan_per_kw'] +
                  100*cap['diesel_units']*cfg['diesel_yuan_per_kw'] + cap['battery_kwh']*cfg['battery_yuan_per_kwh'] +
                  cap['pcs_kw']*cfg['pcs_yuan_per_kw'] + cap['ups_kwh']*cfg['ups_yuan_per_kwh'] + cap['ups_kw']*cfg['ups_yuan_per_kw'])
    assert abs(investment-summary['investment_yuan']) < 1e-8
    start,stop=manifest['start_hour'],manifest['stop_hour']
    errors = {}
    def record(name, value):
        errors[name] = max(errors.get(name,0.), float(np.max(np.abs(value), initial=0.)))
    def variable(block, name, t):
        return values.get(f's{block}_{name}[{t}]', values[f's0_{name}[{t}]'])
    weights = {b['block']:b['conditional_weight'] for b in manifest['branches']}
    totals = {k:0. for k in ['wind_kwh','diesel_kwh','online_unit_hours','battery_charge_kwh','battery_discharge_kwh',
                             'ups_discharge_kwh','interrupt_adjust_kwh','shift_service_kwh','shift_delayed_service_kwh','loss_kwh']}
    branches=[]
    for b in manifest['branches']:
        i=b['block']; get=lambda key:arrays[f's{i}_{key}']
        rows=frame[frame.block==i].sort_values('annual_hour').copy()
        times=rows.annual_hour.to_numpy(dtype=int)
        assert np.array_equal(times,get('annual_hours'))
        rigid=get('rigid_kw'); inter=get('flex_interruptible_kw')
        inter_served=inter-get('interrupt_adjust')-get('interrupt_shed')
        regular_served=rigid-get('rigid_shed')+inter_served+rows.shift_served_kw.to_numpy()
        loss=np.array([rows.iloc[k].rigid_shed+rows.iloc[k].interrupt_shed+
                       variable(i,'shift_emergency_shed',t)+
                       sum(variable(i,'shift_late_shed',a) for a in range(max(0,t-23),t+1) if min(a+23,167)==t)
                       for k,t in enumerate(times)])
        record('main_power_balance',get('wind')+get('pv')+rows.diesel_kw.to_numpy()+get('battery_discharge')-
               get('core_main')-regular_served-get('battery_charge')-rows.heater_kw.to_numpy()-get('ups_charge'))
        record('core_power_balance',get('core_main')+get('ups_discharge')-get('core_kw'))
        record('battery_transition',np.diff(get('battery_energy'))-.95*get('battery_charge')+get('battery_discharge')/.95)
        record('ups_transition',np.diff(get('ups_energy'))-.98*get('ups_charge')+get('ups_discharge')/.98)
        record('pcs_fault_no_power',(get('battery_charge')+get('battery_discharge'))*(~get('pcs_available')))
        record('ups_fault_no_power',(get('ups_charge')+get('ups_discharge'))*(~get('ups_available')))
        record('renewable_bus_fault_no_power',(get('wind')+get('pv'))*get('renewable_bus_fault'))
        record('battery_min_soc',np.maximum(0,.1*cap['battery_kwh']-get('battery_energy')))
        record('battery_max_soc',np.maximum(0,get('battery_energy')-cap['battery_kwh']))
        record('ups_min_soc',np.maximum(0,.05*cap['ups_kwh']-get('ups_energy')))
        record('ups_max_soc',np.maximum(0,get('ups_energy')-.95*cap['ups_kwh']))
        real_power=np.maximum.reduce([get('battery_charge'),get('battery_discharge'),get('diesel_power').max(axis=0)])
        record('gfm_without_active_power',np.maximum(0,1-real_power)*get('bus_live'))
        attempts=get('attempt')[:cap['diesel_units']]
        record('start_temperature',np.maximum(0,5-get('temperature')[:cap['diesel_units'],:-1])*attempts)
        record('crew_limit',np.maximum(0,get('pending').sum(axis=0)-1))
        if i:
            for name in ['battery_energy','ups_energy']:
                record('window_state_link',get(name)[[0,-1]]-arrays[f's0_{name}'][[start,stop]])
        w=np.full(len(times),weights[i])
        if i==0:w[(times<start)|(times>=stop)]=1.
        items={'wind_kwh':get('wind'),'diesel_kwh':rows.diesel_kw.to_numpy(),
               'online_unit_hours':get('online').sum(axis=0),'battery_charge_kwh':get('battery_charge'),
               'battery_discharge_kwh':get('battery_discharge'),'ups_discharge_kwh':get('ups_discharge'),
               'interrupt_adjust_kwh':get('interrupt_adjust'),'shift_service_kwh':rows.shift_served_kw.to_numpy(),'loss_kwh':loss}
        same_hour=np.array([values.get(f's{i}_shift_service[{t},{t}]',values[f's0_shift_service[{t},{t}]']) for t in times])
        items['shift_delayed_service_kwh']=rows.shift_served_kw.to_numpy()-same_hour
        for key, v in items.items():totals[key]+=float(w@v)
        window=(times>=start)&(times<stop)
        q=float(loss[window].sum())
        original=next(x for x in summary['scenario_metrics'] if x['name']==b['name'])
        record('branch_loss_matches_summary',q-original['regular_loss_kwh'])
        fault=(times>=60)&(times<72)
        branches.append(dict(name=b['name'],weight=weights[i],regular_loss_kwh=q,
            core_loss_kwh=float(np.maximum(0,get('core_kw')-get('core_main')-get('ups_discharge')).sum()),
            ups_supply_kwh=float(get('ups_discharge')[window].sum()),
            battery_min_kwh=float(get('battery_energy').min()),
            run_failure_hours=times[get('run_fail').sum(axis=0)>.5].tolist(),
            start_failure_hours=times[get('start_fail').sum(axis=0)>.5].tolist(),
            event_diesel_kwh=float(rows.diesel_kw.to_numpy()[fault].sum()),
            event_battery_supply_kwh=float(get('battery_discharge')[fault].sum()),
            event_interrupt_adjust_kwh=float(get('interrupt_adjust')[fault].sum())))
        frame.loc[rows.index,'regular_loss_kw']=loss
        frame.loc[rows.index,'regular_served_kw']=regular_served
        frame.loc[rows.index,'core_input_kw']=get('core_kw')
    costs={'grid':0.,'diesel':.95*(totals['diesel_kwh']+5*totals['online_unit_hours']),
           'load_loss':1000*totals['loss_kwh'],
           'flex_adjustment':.12*(totals['interrupt_adjust_kwh']+totals['shift_delayed_service_kwh']),
           'battery_throughput':.01*(totals['battery_charge_kwh']+totals['battery_discharge_kwh'])}
    for key,v in costs.items():record('cost_'+key,v-summary['mean_cost_components_yuan'][key])
    q=np.array([b['regular_loss_kwh'] for b in branches]);w=np.array([b['weight'] for b in branches])
    cvar=min(eta+sum(w*np.maximum(0,q-eta))/.05 for eta in np.r_[0,q])
    record('eens',totals['loss_kwh']-summary['eens_kwh'])
    record('cvar',cvar-summary['cvar_upper_bound_kwh'])
    record('battery_terminal',max(0,arrays['s0_battery_energy'][0]-arrays['s0_battery_energy'][-1]))
    record('ups_terminal',arrays['s0_ups_energy'][0]-arrays['s0_ups_energy'][-1])
    assert max(errors.values())<1e-5,errors
    main_loss=float(inputs.loc[60,['rigid_kw','flex_interruptible_kw','flex_shiftable_kw']].sum())
    assert abs(summary['eens_kwh']-.1*main_loss)<1e-7
    result={'passed':True,'max_residual':max(errors.values()),'checks':errors,'weighted_horizon_energy':totals,
            'costs_yuan':costs,'scenario_metrics':branches,'main_bus_loss_floor_kwh':main_loss,
            'eens_floor_kwh':.1*main_loss,'optimality_gap_percent':100*summary['mip_gap'],
            'objective_bound_difference_yuan':summary['objective_yuan']-summary['objective_bound_yuan'],
            'scope':'saved dispatch reconciliation, not a new optimization or transient stability test'}
    (run/'checked_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(branches).to_csv(run/'checked_scenario_metrics.csv',index=False)

    figures=run/'figures';figures.mkdir(exist_ok=True)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    compound=frame[(frame.scenario=='w0_storm_compound')&frame.annual_hour.between(48,83)]
    x=compound.annual_hour.to_numpy();edges=np.r_[x,x[-1]+1]
    extend=lambda v:np.r_[np.asarray(v),np.asarray(v)[-1]]
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    axes[0].stackplot(edges,extend(compound.wind),extend(compound.diesel_kw),extend(compound.battery_discharge),
                      step='post',labels=['风电','柴油机','普通电池放电'],colors=['#529b89','#617ca4','#dfa54a'])
    demand=(compound.core_main+compound.regular_served_kw+compound.heater_kw+compound.ups_charge+compound.battery_charge)
    axes[0].step(edges,extend(demand),where='post',color='#202b35',lw=1.2,label='主母线用电（含充电/加热）')
    axes[0].set(ylabel='功率 / kW');axes[0].legend(ncols=2,frameon=False,loc='upper right',fontsize=9)
    axes[1].step(edges,extend(compound.battery_energy_kwh/750*100),where='post',label='普通电池SOC',color='#c48b26',lw=2)
    axes[1].step(edges,extend(compound.ups_energy_kwh/550*100),where='post',label='UPS SOC',color='#b65c65',ls='--')
    axes[1].axhline(10,color='#888',ls=':',label='电池最低SOC 10%')
    axes[1].set(ylabel='SOC / %',ylim=(0,108));axes[1].legend(ncols=3,frameon=False,loc='lower left',fontsize=9)
    axes[2].step(edges,extend(compound.interrupt_adjust),where='post',label='合法中断调节',color='#529b89',lw=2)
    axes[2].step(edges,extend(compound.regular_loss_kw),where='post',label='实际常规失供',color='#b65c65',ls='--',lw=2)
    axes[2].set(ylabel='功率 / kW',xlabel='周内小时（第60—71小时新能源母线断开）',ylim=(-1,25))
    axes[2].legend(ncols=2,frameon=False)
    for a in axes:a.axvspan(60,72,color='#e6b9b7',alpha=.25);a.grid(alpha=.15);a.set(xlim=(48,84))
    fig.suptitle('暴风雪 + 新能源母线断开12小时：实际运行结果',fontsize=16)
    fig.savefig(figures/'compound_dispatch.png',dpi=180,bbox_inches='tight')
    fig.savefig(figures/'compound_dispatch.svg',bbox_inches='tight');plt.close(fig)
    main=frame[(frame.scenario=='w0_storm_main_bus')&frame.annual_hour.between(56,69)]
    x=main.annual_hour.to_numpy();edges=np.r_[x,x[-1]+1]
    fig,axes=plt.subplots(3,1,figsize=(11,8.5),sharex=True,layout='constrained')
    axes[0].stackplot(edges,extend(main.core_main),extend(main.ups_discharge),step='post',
                      labels=['主母线供应核心负荷','UPS供应核心负荷'],colors=['#617ca4','#b65c65'])
    axes[0].step(edges,extend(main.core_input_kw),where='post',label='核心需求',color='#263441',ls='--')
    axes[0].set(ylabel='核心供电 / kW',ylim=(0,47));axes[0].legend(ncols=3,frameon=False,fontsize=9,loc='upper center')
    axes[1].bar(x,main.regular_loss_kw,width=1,align='edge',color='#bd6c68')
    axes[1].set(ylabel='常规失供 / kW',ylim=(0,270))
    axes[1].text(61.2,223.5,'1小时失供223.516 kWh',fontsize=11)
    axes[2].step(edges,extend(main.ups_energy_kwh/550*100),where='post',color='#b65c65',lw=2)
    axes[2].set(ylabel='UPS SOC / %',xlabel='周内小时（第60小时主母线强制失电）',ylim=(85,97))
    for a in axes:a.axvspan(60,61,color='#e6b9b7',alpha=.25);a.grid(alpha=.15);a.set(xlim=(56,70))
    fig.suptitle('主母线失电1小时：UPS接管核心负荷并恢复备用',fontsize=16)
    fig.savefig(figures/'main_bus_ups.png',dpi=180,bbox_inches='tight')
    fig.savefig(figures/'main_bus_ups.svg',bbox_inches='tight');plt.close(fig)
    print(json.dumps({'passed':True,'max_residual':max(errors.values()),'energy':totals,'costs':costs},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
