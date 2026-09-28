# Recovery-v1 规划模型代码报告

## 1. 模型定位

本文说明仓库中新加入的 `recovery-v1` 小时级可靠性容量规划程序。模型针对这样的恢复过程：同一段暴风雪或极端天气可能同时降低风电出力、阻止启动人员前往备用柴油机，并使备用机组所在集装箱需要额外加热。储能需要覆盖天气等待、人员准备、温度达标和启动结果不确定形成的恢复窗口。

程序入口是 [run_recovery.sh](/home/yzk/reliable_plan/run_recovery.sh)，主 CLI 是 [recovery_cli.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery_cli.py)，默认配置是 [zhongshan_recovery_stress.toml](/home/yzk/reliable_plan/config/zhongshan_recovery_stress.toml)，最终 128/256 场景实验配置是 [zhongshan_recovery_128_256.toml](/home/yzk/reliable_plan/config/zhongshan_recovery_128_256.toml)。Recovery-v1 与旧版 `main.py`、旧配置和历史结果并行存在，不覆盖原有模型。

当前实现包含：

- 一小时离散时间步；
- 人员可达性、准备进度、集装箱温度、柴油状态和储能库存的跨小时递推；
- 预先声明的有限因果策略库；
- 固定容量网格上的完整搜索、风险筛选和成本比较；
- 独立随机种子的 holdout 评价；
- 逐时运行、事件、延迟分解和物理约束审计。

本版本尚未实现分钟级启动、全站电热综合规划、人员路径优化、恢复 LP 耦合割和总体置信认证。因此，报告中的“最优”始终限定为固定容量网格、固定场景样本、固定策略库和给定天气序列内的结果。

## 2. 规划对象与数据流

容量向量为

$$
x=(n_W,n_{PV},n_D,n_E,n_P),
$$

其中：

| 变量 | 含义 | 模块额定值 |
|---|---|---:|
| $n_W$ | 风机模块数 | 100 kW/台 |
| $n_{PV}$ | 光伏模块数 | 100 kW/台 |
| $n_D$ | 柴油模块数 | 100 kW/台 |
| $n_E$ | 电池能量模块数 | 50 kWh/模块 |
| $n_P$ | PCS 模块数 | 50 kW/模块 |

柴油模块的 100 kW 是项目研究额定值。配置加载器会拒绝其他柴油模块值，防止只修改额定功率而遗漏单机成本、台数边界、随机设备编号和箱体映射的同步变化。

代码数据流如下：

```text
cap_plan CSV
    ├─ 负荷、风速、太阳辐射、容量边界、资本成本、燃料参数
    ↓
load_case()
    ↓
WeatherYear
    ├─ 气温、风速、clean 风电、光伏系数
    ├─ 可达性、覆冰、覆冰降额、保护停机、极端危险标签
    ↓
RecoverySettings + PrimitiveNoise + RecoveryPolicy
    ↓
simulate()
    ├─ 逐小时功率/SOC/温度/柴油状态
    ├─ 失供电量、燃料、加热、事件、延迟
    ↓
evaluate_policy() → plan_grid()
    ↓
容量、策略、EENS、CVaR、费用和审计文件
```

旧的 [cap_plan_loader.py](/home/yzk/reliable_plan/polar_reliability_planning/data/cap_plan_loader.py) 负责读取站点曲线和成本；新的 [hourly_weather.py](/home/yzk/reliable_plan/polar_reliability_planning/weather/hourly_weather.py) 补上气温与联合天气字段；新的 [causal_simulator.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/causal_simulator.py) 执行完整小时级物理轨迹。

## 3. 数据合同和中山输入

[recovery/config.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/config.py) 在模拟前执行 fail-fast 检查：

- `schema_version` 必须为 `recovery-v1`；
- 柴油模块必须为 100 kW；
- 目标口径必须为 `expected_causal_operation`；
- 负荷必须声明为母线侧电力需求，不能把未转换的电功率和热功率直接相加；
- 加热器必须声明为新增辅助需求假设，或者给出已从历史负荷中剔除的证据；
- 正式运行 `synthetic=false` 时必须提供热参数、箱体映射、人员配置、柴油额定解释、天气标签和负荷来源；
- 正式天气标签必须带时区，并与选定时序逐点对齐。

中山压力测试配置为 `synthetic=true`。它使用中山原始 CSV 的负荷、风速、太阳辐射和气温，但以下内容仍是工程假设：

- 由风速阈值和气温条件构造人员可达性与覆冰标签；
- 每台柴油模块对应一个独立集装箱；
- 一组启动准备班组；
- `UA=0.15 kW/K`、`C=1.5 kWh/K`、最大加热功率 12 kW、达标温度 5 ℃；
- 柴油 100 kW 研究额定值、20% 最低出力和线性燃油换算；
- 新增加热暂按辅助需求假设计入，历史负荷是否已包含该项尚未核实。

原始中山 CSV 有 8784 条连续小时记录。本实验取前 8760 条，因此结束于 12 月 30 日 23:00，不能称为完整 2020 日历年。气温不做虚构补值；太阳辐射缺值沿用旧加载器的补值规则并记录到元数据。原始 CSV 未标注时区，配置声明 UTC+08 为假设。

## 4. 联合天气和风机模型

天气对象 `WeatherYear` 保存每小时的

$$
\omega_h=(T_h^{amb},v_h,\phi_h^{clean},\phi_h^{PV},
 a_h^{access},i_h^{ice},\gamma_h^{ice},z_h^{protect},e_h^{extreme},z_h^{hard}).
$$

这些字段分别表示气温、风速、clean 风电系数、光伏系数、人员可达性、覆冰、覆冰剩余出力比例、保护停机、极端危险和硬保护。所有字段来自同一小时天气记录或同一明确工程情景，不再独立抽取互不关联的天气过程。

中山压力规则是：

```text
access_safe      = wind_speed_ms < 15
extreme_hazard   = not access_safe and ambient_c <= 0
icing_active     = extreme_hazard or residual_icing_is_still_counting_down
ice_power_factor = 0.5 when icing_active, otherwise 1.0
protective_stop  = icing_active
```

这是研究压力情景，不是现场暴风雪或覆冰识别算法。正式运行应将这些标签替换为观测或工程标注。

clean 风功率经过切入、额定和切出风速计算后，Recovery-v1 使用

$$
\bar P^W_h=n_W P_W^{unit}\phi_h^{clean}\gamma_h^{ice}
(1-z_h^{protect})a_h^{hardware}.
$$

当前有两套固定风机规则：

1. `protective_shutdown`：覆冰保护时停止发电，不自动抽机械维修时间；
2. `operate_with_hazard_multiplier`：在硬保护允许的范围内继续运行，极端小时将风机危险率乘以 2。

如果输入是已经包含覆冰损失的 observed net output，当前适配器会拒绝运行，避免重复施加覆冰降额。保护停机与机械损坏分开记录。

## 5. 柴油、人员和恢复状态

每台柴油机由 [diesel_state_machine.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/diesel_state_machine.py) 的 `DieselState` 保存：

```text
healthy                  本体是否健康
running                  是否在线运行
pending                  是否已有启动请求
onsite                   启动班组是否到场
prep                     已完成准备小时数
min_up / min_down        最小开停机剩余时间
repair_until             隐藏维修完成边界
runtime                  实际累计在线小时
demands                  有效启动需求次数
```

可显示为：

```text
STANDBY → WAIT_ACCESS → PREPARING → WAIT_HEAT → READY → RUNNING
                         └──────────────→ FAILED → repair complete
```

`WAIT_ACCESS`、`PREPARING` 和 `WAIT_HEAT` 是健康但暂时不能供电的状态，不计为机械故障。`FAILED` 表示硬件故障、启动失败后的排故或运行故障。

默认人员模式是 `arrival_only`：坏天气禁止新派遣，已经进入箱体的人员可以继续室内工作。`all_work_safe` 是敏感性模式，坏天气暂停准备但保留已完成进度。启动班组数量显式配置，不能默认一组人员同时完成无限多个机组任务。

准备时间与最小停机时间是两个独立条件。准备进度只有在人员到场并且人员模式允许工作时增加。启动资格必须同时满足

$$
healthy_{i,h}=1,
\quad onsite_{i,h}=1,
\quad prep_{i,h}\ge3,
\quad \theta_{c(i),h}\ge\theta^{ready},
\quad min\_down_{i,h}=0.
$$

因此，本小时末才完成准备或温度达标的机组，最早下一小时才可发电。已经在线的柴油机不会因为人员无法外出而自动停机。

## 6. 集装箱热模型

每个箱体使用单节点 1R1C 模型：

$$
C_c\frac{d\theta_c}{dt}
=\eta_H P^H_c+Q_c^{DG}-UA_c(\theta_c-T^{amb}).
$$

当 $UA_c>0$ 时，令

$$
a_c=\exp(-UA_c\Delta t/C_c),
$$

使用精确小时更新：

$$
\theta_{c,h+1}=a_c\theta_{c,h}+(1-a_c)T_h^{amb}
+\frac{1-a_c}{UA_c}
(\eta_HP^H_{c,h}+Q^{DG}_{c,h}).
$$

当 $UA_c=0$ 时使用极限式

$$
\theta_{c,h+1}=\theta_{c,h}
+\frac{\eta_HP^H_{c,h}+Q^{DG}_{c,h}}{C_c}\Delta t.
$$

实现位于 [container_thermal.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/container_thermal.py)。加热器先计算达到阈值所需的功率，然后受真实电力平衡限制。温度更新使用实际供给功率，而不是可能无法执行的额定指令。

同一个 `container_id` 只计一次箱体加热。没有电源或储能时，电加热不能继续工作，也不能通过“先启动柴油、再给自己加热”构造循环可行性。

## 7. 小时边界和信息结构

第 $h$ 个时段为 $[h,h+1)$，负荷和功率为该小时平均量，SOC、温度、准备进度和设备状态为小时起点状态。`simulate()` 的顺序是：

1. 接收上一小时末发生的故障并完成到期日历维修；
2. 读取当前负荷、天气、SOC、温度、人员和设备状态；
3. 因果策略决定在线目标、派遣、准备、启动申请、加热和储能；
4. 对合法完整启动需求抽取一次启动成败；
5. 执行风光、在线柴油、储能、用户负荷和实际加热的小时平衡；
6. 推进准备、温度、SOC 和最小开停机时间；
7. 只有实际在线柴油机抽取运行故障，故障在下一小时边界生效；
8. 保存逐时状态和事件审计。

[recovery_policy.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/recovery_policy.py) 中的 `Observation` 只包含当前可观测信息：当前小时、负荷、可再生出力、SOC、可达性、可观测设备状态和箱体温度。它不包含未来故障、随机数、真实维修完成时刻或未来天气。

## 8. 功率平衡和储能

小时功率平衡为

$$
P^W_h+P^{PV}_h+\sum_iP^D_{i,h}+P^{dis}_h+L^{shed}_h
=L_h+P^{ch}_h+\sum_cP^H_{c,h}.
$$

其中 $L_h$ 是原始母线负荷，$P^H_{c,h}$ 是实际箱体加热功率，$L^{shed}_h$ 是用户负荷失供，满足

$$
0\le L^{shed}_h\le L_h.
$$

储能更新为

$$
E_{h+1}=E_h+\eta_cP^{ch}_h\Delta t-
\frac{P^{dis}_h\Delta t}{\eta_d},
$$

并满足

$$
SOC_{min}E_B\le E_h\le SOC_{max}E_B,
\quad
0\le P^{ch}_h,P^{dis}_h\le P^{PCS}_{available,h},
\quad P^{ch}_hP^{dis}_h=0.
$$

程序会检查充放电互斥、SOC 上下界、PCS 可用功率、电力平衡和温度更新。故障后不会自动把电池恢复为初始 SOC，也不会在年末免费补能。当前没有无依据的无限消纳负荷；如果柴油最低出力无法吸收，策略会被标记为执行失败。

## 9. 故障和维修

### 9.1 柴油运行故障

柴油运行 MTTF 使用研究参数 1662 h，在线小时故障概率为

$$
p_D^{run}=1-\exp(-\Delta t/1662).
$$

停机、等待人员、准备中、温度不足和维修中的机组不累计此暴露；低出力在线仍属于运行状态。

### 9.2 启动失败

只有健康、到场、准备完成、温度达标且最小停机完成的机组才产生有效启动需求。一次完整需求只抽取一次

$$
p_D^{start}=0.0013.
$$

启动失败后进入故障/排故流程，不能下一小时无成本无限重试。启动失败后的修复使用显式建模假设，不能表述成原始数据对专属启动失败修复时间的证明。

### 9.3 日历修复

基准柴油修复均值为 37 h，默认使用均值为 37 h 的离散几何分布：

$$
R\sim Geometric(1/37),\quad R\in\{1,2,\ldots\}.
$$

修复按日历小时推进，不因暴风雪自动暂停，也不把 37 h 解释成 37 h 的现场人工工时。维修完成后硬件恢复，但温度、人员任务和 SOC 沿用当前状态。

风、光和 PCS 分别使用配置中的故障率和修复时间。风机极端危险率乘数只在允许继续运行且没有保护停机时生效；保护动作本身不会自动变成机械维修事件。

## 10. 因果策略库

本版不对每个场景事后选择最有利的调度，而是先冻结有限策略库，再把每一套策略完整地用于所有场景。策略字段为：

```text
online_floor       最低在线柴油台数
reserve_units      净负荷之外的在线备用台数
recharge_soc       储能补能目标 SOC
recharge_kw        储能补能功率目标
heat_priority      加热相对用户负荷的固定优先级
keep_standby_warm  是否给等待任务保持加热
```

默认中山配置包含：

```text
load_first_three_online
    online_floor=3, heat_priority=false

heat_first_four_online
    online_floor=4, heat_priority=true
```

策略根据当前观测设定在线目标：

$$
n^{target}_h=\min\left(n_D,
\max\left(n^{floor},
\left\lceil\frac{\max(0,L_h-P^{RE}_h+P^{recharge}_h)}{P_D^{unit}}\right\rceil
+n^{reserve}\right)\right).
$$

策略执行失败只排除该容量/策略组合，不能直接宣布该容量对所有控制器不可行。代码将这类记录标为 `scope="this_capacity_and_policy_only"`。

## 11. 目标函数与风险约束

场景 $s$ 的全年失供电量为

$$
Q_s(x,\pi)=\sum_{h=0}^{H-1}L^{shed}_{h,s}\Delta t.
$$

运行费用由实际轨迹计算：

$$
C_s(x,\pi)=C^{fuel}_s+C^{startup}_s+C^{storage}_s.
$$

规划目标为

$$
\min_{x\in X_{grid},\pi\in\Pi}
C^{inv}(x)+\sum_s w_sC_s(x,\pi),
$$

并要求同一套因果策略同时满足

$$
\operatorname{EENS}=\sum_sw_sQ_s(x,\pi)\le\bar E,
$$

$$
\operatorname{CVaR}_{\alpha}(Q)\le\bar C.
$$

代码使用 [risk_bounds.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/risk_bounds.py) 和精确离散 CVaR 计算器，按分位点处的部分概率质量计算 CVaR，不用“取最坏整数个场景平均”的近似。

中山最终实验使用 $\alpha=0.95$、EENS≤100 kWh、CVaR≤1000 kWh。若至少 95% 场景没有失供，VaR 为 0，CVaR 等于 EENS/0.05；因此程序同时输出 VaR 和非零失供比例，避免把两项风险指标误解为完全独立。

## 12. 容量网格搜索

规划器位于 [recovery_optimizer.py](/home/yzk/reliable_plan/polar_reliability_planning/planning/recovery_optimizer.py)，执行：

```text
读取配置并校验
    ↓
加载中山负荷、风光、气温和联合天气
    ↓
冻结容量网格和有限策略库
    ↓
按投资成本排序容量组合
    ↓
逐个容量 × 策略运行训练场景
    ↓
计算完整 EENS/CVaR 和实际平均费用
    ↓
保留通过双约束的最低费用组合
    ↓
冻结容量与策略
    ↓
用独立随机种子执行 holdout
    ↓
回放最坏路径并独立审计
```

### 风险早停

完成部分场景后，程序用未完成场景零损失构造乐观下界：

$$
Q^{LB}=(Q_1,\ldots,Q_k,0,\ldots,0).
$$

若该下界已经超过 EENS 或 CVaR 限值，剩余场景可停止计算。原始场景权重不被改变，早停只是计算筛选。

### 成本下界与证据范围

当前经济下界只使用非负投资费用，不把无故障标称调度成本当作真实因果运行费用下界。容量访问记录、投资成本剪枝和物理执行失败分别保存。当前没有恢复 LP 耦合割，因此不声称对所有控制策略得到全局不可行证明。

### 场景存档和共同随机数

场景记录按以下键保存：

```text
role : capacity tuple : policy fingerprint : scenario id
```

随机数按 `seed/scenario/mechanism/device/hour` 生成，不依赖访问顺序或线程编号。`run_manifest.json` 保存配置、源代码 SHA256、输入数据 SHA256、天气标签指纹、随机机制版本和软件版本。`--resume` 时指纹不一致会拒绝续算。

## 13. R0–R3 物理消融模式

| 模式 | 人员可达性 | 准备 | 箱体温度 | 作用 |
|---|---|---|---|---|
| R0 | 忽略 | 忽略 | 忽略 | 即时备用基线 |
| R1 | 忽略 | 使用准备延迟 | 忽略 | 固定延迟对照 |
| R2 | 使用 | 使用 | 始终满足 | 分离天气/人员影响 |
| R3 | 使用 | 使用 | 使用实际热状态 | 完整 Recovery-v1 |

所有对照方案最终都可以放回完整 R3 评价器，避免用简化模型自身的风险结果证明简化模型安全。72 小时合成网格中，R0/R1 选择的 600 kWh 电池方案在 R3 评价器中出现明显失供，而完整 R3 选择了更高电池能量。这是机制验证，不是站点容量结论。

## 14. 代码模块映射

| 模块 | 责任 | 关键对象/函数 |
|---|---|---|
| [recovery/config.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/config.py) | 参数合同与校验 | `RecoverySettings`, `read_recovery_config` |
| [weather/hourly_weather.py](/home/yzk/reliable_plan/polar_reliability_planning/weather/hourly_weather.py) | 气温、风速和联合天气标签 | `WeatherYear`, `load_weather` |
| [recovery/container_thermal.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/container_thermal.py) | 精确 1R1C 更新 | `ThermalParameters.step` |
| [recovery/personnel.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/personnel.py) | 派遣和准备进度 | `may_dispatch`, `advance_preparation` |
| [recovery/diesel_state_machine.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery/diesel_state_machine.py) | 柴油逐机状态 | `DieselState` |
| [scenario_generation/primitive_noise.py](/home/yzk/reliable_plan/polar_reliability_planning/scenario_generation/primitive_noise.py) | 键控随机数、修复时长 | `PrimitiveNoise`, `repair_duration` |
| [reliability/recovery_policy.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/recovery_policy.py) | 观察量和策略 | `Observation`, `RecoveryPolicy` |
| [reliability/causal_simulator.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/causal_simulator.py) | 完整小时物理执行 | `simulate`, `SimulationResult` |
| [reliability/risk_bounds.py](/home/yzk/reliable_plan/polar_reliability_planning/reliability/risk_bounds.py) | EENS/CVaR和下界 | `risk_summary`, `partial_bounds` |
| [planning/recovery_optimizer.py](/home/yzk/reliable_plan/polar_reliability_planning/planning/recovery_optimizer.py) | 容量网格、策略搜索 | `evaluate_policy`, `plan_grid` |
| [validation/recovery_audit.py](/home/yzk/reliable_plan/polar_reliability_planning/validation/recovery_audit.py) | 独立物理和成本回算 | `audit_trace` |
| [recovery_cli.py](/home/yzk/reliable_plan/polar_reliability_planning/recovery_cli.py) | CLI、输出、holdout | `run`, `main` |

## 15. 中山 128/256 算例

最终结果位于 [full_grid/summary.json](/home/yzk/reliable_plan/reports/zhongshan_recovery_8760_protective_128_256_v1/full_grid/summary.json)。容量网格有 48 个组合，策略库有 2 套策略。冻结方案是：

```text
wind             = 5 modules = 500 kW
pv               = 3 modules = 300 kW
diesel           = 5 modules = 500 kW
battery_energy   = 20 modules = 1000 kWh
pcs              = 4 modules = 200 kW
policy           = heat_first_four_online
```

| 指标 | 128 个训练场景 | 256 个独立验证场景 | 上限 |
|---|---:|---:|---:|
| EENS | 0.6081 kWh | 5.5371 kWh | 100 kWh |
| CVaR₀.₉₅ | 12.1620 kWh | 110.7413 kWh | 1000 kWh |
| VaR₀.₉₅ | 0 | 0 | — |
| 非零失供比例 | 1/128 | 4/256 | — |
| 年化投资 | 1,210,000 元 | 1,210,000 元 | — |
| 训练目标总费用 | 5,519,867.49 元 | — | — |

结果状态是 `sample_optimal_within_policy_library_gap` 和 `holdout_passed_empirically`，`population_certified=false`。最终审计回放最坏验证场景覆盖含热身在内的 9096 小时，功率、SOC 和热更新残差均在数值容差内，没有审计违规。

这组结果只能解释为：在当前天气序列、压力标签、热参数、容量网格、两套策略和有限样本内，找到了一套通过训练与 holdout 的可执行方案。它不是总体可靠性认证，也不是现场采购建议。

## 16. 运行方法

预检：

```bash
cd /home/yzk/reliable_plan
./run_recovery.sh preflight \
  --config config/zhongshan_recovery_128_256.toml \
  --output outputs/recovery_preflight
```

规划：

```bash
./run_recovery.sh plan \
  --config config/zhongshan_recovery_128_256.toml \
  --output outputs/zhongshan_recovery_128_256
```

切换风机规则：

```bash
./run_recovery.sh plan \
  --config config/zhongshan_recovery_128_256.toml \
  --wind-mode operate_with_hazard_multiplier \
  --output outputs/zhongshan_recovery_operating
```

固定容量评价：

```bash
./run_recovery.sh evaluate \
  --config config/zhongshan_recovery_128_256.toml \
  --units '{"wind":5,"pv":3,"diesel":5,"battery_energy":20,"pcs":4}' \
  --policy heat_first_four_online \
  --output outputs/recovery_fixed_capacity
```

续算必须使用同一配置、代码、数据和随机机制：

```bash
./run_recovery.sh plan \
  --config config/zhongshan_recovery_128_256.toml \
  --output outputs/zhongshan_recovery_128_256 \
  --resume
```

测试：

```bash
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s recovery_tests -v
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s tests -v
```

## 17. 输出文件

| 文件 | 内容 |
|---|---|
| `run_manifest.json` | 配置、代码、数据和随机机制指纹 |
| `migration_audit.md` | 旧模型到 Recovery-v1 的迁移核查 |
| `resolved_config.json` | 实际参数、模块边界、天气元数据、策略指纹 |
| `hourly_weather.csv` | 逐小时联合天气与标签 |
| `capacity_result.json` | 容量、策略、风险、费用、gap、验证状态 |
| `scenario_losses.csv` | 场景概率、失供电量和实际运行费用 |
| `hourly_dispatch.csv` | 最坏回放轨迹的功率、SOC、温度和状态 |
| `event_log.csv` | 派遣、启动、失败、维修和保护事件 |
| `delay_decomposition.csv` | 到场、班组、准备、温度等延迟里程碑 |
| `audit_report.json` | 功率、SOC、热状态、成本和信息结构审计 |
| `policy_evaluations.csv` | 容量 × 策略评价记录 |
| `visited_designs.jsonl` | 已访问容量，作用域限定为策略库 |
| `cuts.jsonl` | 当前为空，本版未实现恢复 LP 耦合割 |
| `checkpoints/paths.jsonl` | 可续算的场景记录 |

## 18. 当前限制和后续工作

下一阶段优先补充现场证据，而不是继续堆叠未经校准的模型复杂度：

1. 提供同一时区下的现场可达性、能见度、降雪和覆冰标签；
2. 核实聚合负荷是否包含集装箱加热或其他辅助负荷；
3. 取得集装箱 `UA`、等效热容、加热器额定功率和温度阈值；
4. 核实柴油模块 Prime/Standby 身份和适用的 100 kW 油耗曲线；
5. 明确启动班组与维修班组是否共享，以及修复后是否重新准备；
6. 提供与现场设备匹配的风机、光伏、PCS 故障率、修复时间和共因故障数据；
7. 用更多天气年替代给定单一年份天气条件下的设备随机可靠性；
8. 在 R3 小网格逐点验证后，再实现恢复 LP 风险下界割；
9. 扩展经过审计的因果策略库或合法预报驱动的 MPC；
10. 若需要总体可靠性声明，单独设计尾部样本量、置信区间和自适应搜索证书。

Recovery-v1 的准确定位是：**一个可执行、可审计、具有明确因果信息结构的小时级恢复过程容量规划基线**。它可以回答人员和热状态是否影响容量、储能是否真实接续、风机保护是否与机械故障分开，以及同一策略是否同时通过 EENS/CVaR；它尚未替代经过现场数据校准和统计认证的工程设计流程。
