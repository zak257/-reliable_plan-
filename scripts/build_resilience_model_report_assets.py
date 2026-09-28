"""Build input-data figures for the group-meeting model report; no optimization."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/resilience_joint_comparison_15000s_20260924T013515Z"
OUT = ROOT / "docs/assets/resilience_model_group_meeting"
COLS = ["core_kw", "rigid_kw", "flex_interruptible_kw", "flex_shiftable_kw"]
LABELS = ["核心负荷", "常规刚性负荷", "可中断负荷", "可平移负荷"]
COLORS = ["#c94e50", "#4978a6", "#4b9e90", "#e5b44f"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_manifest = json.loads((RUN / "comparison_manifest.json").read_text())
    cfg = cfg_manifest["config"]
    actual_hash = hashlib.sha256((RUN / "shared_year.npz").read_bytes()).hexdigest()
    assert actual_hash == cfg_manifest["shared_input_sha256"]
    with np.load(RUN / "shared_year.npz", allow_pickle=False) as z:
        f = pd.DataFrame({k: z[k] for k in COLS + ["ambient_c", "wind_clean_pu", "pv_pu"]})
        f["timestamp"] = z["timestamps"].astype(str)
    f["load_total_kw"] = f[COLS].sum(axis=1)
    assert len(f) == 8760
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = FontProperties(fname=font_path)
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    total = float(f.load_total_kw.sum())
    summary = []
    for name, label in zip(COLS + ["load_total_kw"], LABELS + ["合计"]):
        v = f[name]
        summary.append(dict(key=name, label=label, energy_kwh=float(v.sum()),
                            share_pct=float(v.sum() / total * 100), mean_kw=float(v.mean()),
                            min_kw=float(v.min()), max_kw=float(v.max())))
    pd.DataFrame(summary).to_csv(OUT / "load_statistics.csv", index=False)
    windows = []
    for j, a in enumerate(cfg_manifest["window_start_hours"]):
        event = a + 36
        regular = float(f.loc[event, COLS[1:]].sum())
        windows.append(dict(window=j+1, start_hour=a, end_hour=a+72,
                            start_time=f.timestamp[a], end_time=f.timestamp[a+72],
                            event_time=f.timestamp[event], core_kw=float(f.core_kw[event]),
                            forced_regular_loss_kwh=regular, weighted_loss_kwh=.1*regular))
    floor = sum(w["weighted_loss_kwh"] for w in windows)
    cvar_floor = sum(w["forced_regular_loss_kwh"] for w in windows)
    required_e = float(f.core_kw.max())*cfg["ups_bridge_hours"] / (
        (cfg["ups_standby_soc"]-cfg["ups_min_soc"])*cfg["ups_efficiency"])
    required_p = float(f.core_kw.max())*cfg["ups_power_margin"]
    peak_hour = int(f.load_total_kw.idxmax())
    stats = dict(input_file=str(RUN / "shared_year.npz"), input_sha256=actual_hash,
                 synthetic=True, hours=len(f), dt_hours=1, seed=cfg["seed"], loads=summary,
                 annual_peak_hour=peak_hour, annual_peak=f.loc[peak_hour].to_dict(),
                 core_hourly_share_min_pct=float((f.core_kw/f.load_total_kw).min()*100),
                 core_hourly_share_max_pct=float((f.core_kw/f.load_total_kw).max()*100),
                 regular_energy_kwh=total-summary[0]["energy_kwh"],
                 flexible_energy_share_pct=summary[2]["share_pct"]+summary[3]["share_pct"],
                 required_ups_kwh=required_e, required_ups_kw=required_p,
                 minimum_ups_energy_modules=math.ceil(required_e/50),
                 minimum_ups_power_modules=math.ceil(required_p/50),
                 main_bus_eens_floor_kwh=floor, sum_window_cvar_floor_kwh=cvar_floor,
                 eens_remaining_kwh=cfg["eens_limit_kwh"]-floor,
                 cvar_bound_remaining_kwh=cfg["cvar_limit_kwh"]-cvar_floor,
                 windows=windows,
                 baseline_resource_stats={k:dict(mean=float(f[k].mean()), min=float(f[k].min()),
                                                 max=float(f[k].max()))
                                          for k in ["wind_clean_pu", "pv_pu", "ambient_c"]})
    (OUT / "model_input_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2)+"\n")
    pd.DataFrame(windows).to_csv(OUT / "window_statistics.csv", index=False)

    # Figure 1: same color/class order for annual and hourly views.
    daily = f[COLS].groupby(np.arange(len(f))//24).mean()
    daily.insert(0, "day_from_1", np.arange(1,366))
    daily.to_csv(OUT / "daily_mean_load.csv", index=False)
    week_start = cfg_manifest["window_start_hours"][0]
    week = f.iloc[week_start:week_start+168]
    week[["timestamp", *COLS, "load_total_kw"]].to_csv(OUT / "example_week_load.csv", index=False)
    fig, ax = plt.subplots(2, 1, figsize=(13,8.0), gridspec_kw={"height_ratios":[1,1.05]}, layout="constrained")
    ax[0].stackplot(daily.day_from_1, *[daily[k] for k in COLS], colors=COLORS, labels=LABELS)
    ax[0].set(xlim=(1,365), ylim=(0,290), ylabel="日平均功率 / kW", xlabel="年内第几天",
              title="(a) 全年负荷分级：8760 小时输入的日平均值")
    ax[0].legend(ncols=4, loc="upper center", frameon=False)
    ax[1].stackplot(np.arange(168), *[week[k] for k in COLS], colors=COLORS)
    ax[1].set(xlim=(0,167), ylim=(0,290), ylabel="逐时功率 / kW", xlabel="周内小时（0 = 2 月 4 日 00:00）",
              title="(b) 代表周：2 月 4—10 日，保留逐时波动")
    for a in ax: a.grid(axis="y", alpha=.2); a.set_axisbelow(True)
    fig.suptitle("负荷分级堆叠图｜模拟基准需求，不是调度后的实际供电", fontsize=16)
    save(fig, "01_load_stack")

    fig, ax = plt.subplots(figsize=(11,4.8), layout="constrained")
    bars=ax.barh(LABELS[::-1], [v["energy_kwh"]/1000 for v in summary[:4]][::-1], color=COLORS[::-1], height=.58)
    for b,v in zip(bars,summary[:4][::-1]):
        ax.text(b.get_width()+15, b.get_y()+b.get_height()/2,
                f'{v["energy_kwh"]/1000:,.3f} MWh  |  {v["share_pct"]:.3f}%', va="center")
    ax.set(xlim=(0,1570), xlabel="年基准用电量 / MWh",
           title=f"各类负荷年电量占比｜全年合计 {total/1000:,.3f} MWh")
    ax.grid(axis="x",alpha=.2); ax.set_axisbelow(True)
    save(fig, "02_load_energy")

    # Figure 3: explicitly separate the main bus and UPS critical-load branch.
    fig, ax=plt.subplots(figsize=(13.5,7.5))
    ax.set(xlim=(0,14),ylim=(0,8));ax.axis("off")
    def box(x,y,w,h,text,color="#edf2f6"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                                   fc=color,ec="#587083",lw=1.25))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11)
    def arrow(a,b,label=None,both=False,color="#496174"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="<->" if both else "-|>",mutation_scale=15,color=color,lw=1.8))
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.14,label,ha="center",va="bottom",fontsize=9,color=color)
    box(.25,6.0,2.2,1.05,"风电 + 光伏\n100 kW / 模块")
    box(3.15,6.0,2.3,1.05,"新能源母线\n可整体断开", "#fff0d7")
    box(.25,3.85,2.2,1.1,"柴油发电机\n100 kW / 台")
    box(.25,1.85,2.2,1.1,"普通电池\n50 kWh / 模块", "#e2f0ea")
    box(3.15,1.85,2.3,1.1,"储能 PCS\n50 kW / 模块", "#e2f0ea")
    ax.plot([6.45,6.45],[1.5,7.0],color="#354e68",lw=6)
    ax.text(6.45,7.24,"主母线",ha="center",fontsize=13,fontweight="bold")
    box(8.0,5.3,2.7,1.2,"常规负荷\n刚性 / 可中断 / 可平移")
    box(10.95,2.9,2.65,1.25,"核心负荷支路\n要求逐时零失供", "#f9e2e1")
    box(8.0,.65,2.7,1.3,"专用应急 UPS\n50 kWh / 50 kW\n能量与功率分别规划", "#f9e2e1")
    arrow((2.45,6.52),(3.1,6.52));arrow((5.45,6.52),(6.4,6.52))
    arrow((2.45,4.4),(6.4,4.4),"实际发电 + 构网")
    arrow((2.45,2.4),(3.1,2.4),both=True);arrow((5.45,2.4),(6.4,2.4),both=True)
    arrow((6.5,5.9),(7.95,5.9));arrow((6.5,3.65),(10.9,3.65),"正常主供电")
    arrow((6.5,1.5),(7.95,1.5),"柴发专门补电")
    arrow((10.7,1.3),(12.25,2.86),"应急供电",color="#b94a4b")
    ax.text(3.6,.88,"主母线带电时：至少一台柴发或储能 PCS\n具有当前有效有功功率（≥ 1 kW）",ha="center",fontsize=10)
    ax.text(7,.05,"拓扑示意：独立微电网；UPS 不向常规负荷或主母线反送电。",ha="center",fontsize=10,color="#465361")
    fig.suptitle("多级供电结构与设备职责",fontsize=17)
    save(fig,"03_system_topology")

    fig, axes=plt.subplots(2,1,figsize=(13,7.4),layout="constrained",gridspec_kw={"height_ratios":[1,2]})
    a=axes[0];a.set(xlim=(0,8760),ylim=(-.8,1.0),yticks=[],xlabel="全年小时（从 0 开始）",
                    title="(a) 一条连续的 8760 小时运行轨迹，嵌入四个恢复窗口")
    a.plot([0,8760],[0,0],lw=8,color="#c9d4de",solid_capstyle="round")
    for j,w in enumerate(windows):
        x=w["start_hour"];a.plot([x,x+72],[0,0],lw=15,color="#b65e63",solid_capstyle="butt")
        a.annotate(f'窗口 {j+1}\n{w["start_time"][5:10]} / h={x}',(x+36,0),xytext=(x+36,.45),ha="center",fontsize=10,
                   arrowprops=dict(arrowstyle="-",color="#777"))
    for spine in ["left","bottom"]:a.spines[spine].set_visible(False)
    a=axes[1]
    a.set(xlim=(-1,73),ylim=(-.8,4.0),yticks=[0,1,2,3],
          yticklabels=["新能源母线断开（风雪分支）","暴风雪及出力降额","允许日前准备","状态与年度轨迹衔接"],
          xlabel="窗口内小时",title="(b) 一个 72 小时窗口的输入时序；实际充放电和启动由模型决定")
    a.broken_barh([(0,72)],(2.85,.3),facecolors="#ccd8e0")
    a.broken_barh([(0,24)],(1.8,.4),facecolors="#81b09c")
    a.broken_barh([(24,30)],(.8,.4),facecolors="#7c93b0")
    a.broken_barh([(36,12)],(-.2,.4),facecolors="#c96b68")
    for x in [0,24,36,48,54,72]:a.axvline(x,ls=":",lw=.9,color="#aab4bd")
    a.text(0,3.38,"继承年度状态\n发布次日天气信号",ha="left",fontsize=10)
    a.text(72,3.38,"恢复并接回\n年度公共状态",ha="right",fontsize=10)
    a.text(36,2.25,"设备故障发生后才揭示",ha="left",color="#ac494b",fontsize=10)
    a.set_xticks([0,12,24,36,48,54,60,72])
    fig.suptitle("全年规划与 72 小时韧性场景的关系",fontsize=16)
    save(fig,"04_annual_windows")

    sl=slice(816,888);x=np.arange(72);storm=(x>=24)&(x<54);fault=(x>=36)&(x<48)
    wb=f.wind_clean_pu.iloc[sl].to_numpy();pb=f.pv_pu.iloc[sl].to_numpy()
    temp=f.ambient_c.iloc[sl].to_numpy();ws=wb*np.where(storm,.35,1);ps=pb*np.where(storm,.65,1)
    compound=pd.DataFrame(dict(window_hour=x,wind_baseline_pu=wb,wind_storm_pu=ws,
                              wind_compound_pu=ws*(~fault),pv_baseline_pu=pb,pv_storm_pu=ps,
                              pv_compound_pu=ps*(~fault),ambient_baseline_c=temp,
                              ambient_storm_c=np.where(storm,-12,temp)))
    compound.to_csv(OUT / "compound_scenario_inputs.csv",index=False)
    fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True,layout="constrained")
    edge_x=np.arange(73)
    extend=lambda values:np.r_[values,values[-1]]
    for a,base,weather,name in [(axs[0],wb,ws,"风电可送出上限 / 装机容量"),(axs[1],pb,ps,"光伏可送出上限 / 装机容量")]:
        a.step(edge_x,extend(base),where="post",color="#88939d",ls="--",label="正常天气基准")
        a.step(edge_x,extend(weather),where="post",color="#477faa",label="仅暴风雪")
        a.step(edge_x,extend(weather*(~fault)),where="post",color="#be5153",lw=2,label="暴风雪 + 新能源母线断开")
        a.set(ylabel=name,ylim=(-.04,1.09));a.legend(ncols=3,loc="upper right",fontsize=9,frameon=False)
    axs[2].step(edge_x,extend(temp),where="post",ls="--",color="#88939d",label="基准环境温度")
    axs[2].step(edge_x,extend(np.where(storm,-12,temp)),where="post",color="#477faa",lw=2,label="暴风雪分支温度")
    axs[2].set(ylabel="环境温度 / ℃",xlabel="窗口内小时（0 = 2 月 4 日 00:00）")
    axs[2].legend(ncols=2,frameon=False,loc="upper right",fontsize=9)
    for a in axs:
        a.axvspan(24,54,color="#dbe7f1",alpha=.35,zorder=-5)
        a.axvspan(36,48,color="#eed0ce",alpha=.5,zorder=-4)
        a.grid(alpha=.15);a.set(xlim=(0,72));a.set_xticks([0,12,24,36,48,54,60,72])
    fig.suptitle("复合压力场景的输入变化｜蓝底：风雪 30 h；红底：新能源母线断开 12 h",fontsize=15)
    save(fig,"05_compound_inputs")

    fig,axs=plt.subplots(2,1,figsize=(11,5),layout="constrained")
    for a,value,limit,title in [(axs[0],floor,100,"年度 EENS 限额"),
                              (axs[1],cvar_floor,1000,"各窗口 CVaR 之和的限额（年度风险上界）")]:
        a.barh([0],[value],color="#c76b68",height=.46)
        a.barh([0],[limit-value],left=[value],color="#b8d1c6",height=.46)
        a.text(value/2,0,f"强制主母线故障已占用 ≥ {value:.3f} kWh",va="center",ha="center",color="white",fontsize=11)
        a.text(value+(limit-value)/2,0,f"剩余至多\n{limit-value:.3f}",ha="center",va="center",fontsize=10)
        a.set(xlim=(0,limit),ylim=(-.55,.55),yticks=[],xlabel="kWh",title=title)
        a.spines["left"].set_visible(False)
    fig.suptitle("风险限额中的不可避免部分｜由场景和供电规则推导，非优化结果",fontsize=15)
    save(fig,"06_risk_floor")
    print(json.dumps({"output":str(OUT),"figures":6,"core_energy_share_pct":summary[0]["share_pct"],
                      "required_ups_kwh":required_e,"main_bus_eens_floor_kwh":floor},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
