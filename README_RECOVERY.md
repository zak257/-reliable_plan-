# 人员可达性、集装箱温度与覆冰：Recovery-v1

这是新增的小时级因果容量规划入口。它读取 `cap_plan` 的中山原始数据，保留旧入口、旧配置、历史结果与工作区已有的 certified 程序。实现依据是本次附件中的物理、信息与概率合同；高级恢复 LP 耦合割尚未实现，本版使用可完整枚举、可审计的有限策略库基线。

代码结构、状态递推、数学公式、风险约束、输出字段和中山算例结果见 [Recovery-v1 规划模型代码报告](docs/recovery_planning_model_code_report.md)。

```bash
cd /home/yzk/reliable_plan

# 数据预检，输出迁移核查、数据/代码 SHA256、逐小时联合天气表
./run_recovery.sh preflight --output outputs/recovery_preflight

# 新模型验收测试（包含并行/串行最优解一致性）
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s recovery_tests -v

# 72 h 手工事故闭环、12 点穷举、筛选/缓存对照、R0–R3 同模型复核
/home/yzk/cap_plan/venv/bin/python -m polar_reliability_planning.validation.recovery_experiments \
  --output outputs/recovery_synthetic_72h

# 8760 h；48 个容量点 × 2 策略；16 个训练样本，128 个独立验证样本
./run_recovery.sh plan --output outputs/zhongshan_recovery_protective

# 固定另一套风机保护规则，单独规划并验证
./run_recovery.sh plan --wind-mode operate_with_hazard_multiplier \
  --output outputs/zhongshan_recovery_operating

# 扩大训练到 128 场景，使用新的独立种子验证 256 场景；4 个容量分区并行
# 全部分区使用同一物理模型，合并时校验配置/数据/源码/噪声指纹，最后复核整个网格
/home/yzk/cap_plan/venv/bin/python -m scripts.recovery_parallel_grid \
  --samples 128 --validation-samples 256 --validation-seed 2026091703 \
  --workers 4 --output outputs/zhongshan_recovery_expanded

# 原样续算；所有配置、源码、数据和随机机制指纹必须相同
./run_recovery.sh plan --output outputs/zhongshan_recovery_protective --resume

# 固定容量评价，输入是模块数，电池每模块 50 kWh，PCS 每模块 50 kW
./run_recovery.sh evaluate \
  --units '{"wind":7,"pv":3,"diesel":5,"battery_energy":40,"pcs":6}' \
  --policy heat_first_four_online --output outputs/recovery_fixed_capacity
```

`--config` 指定新格式 TOML；`--hours`、`--samples`、`--validation-samples` 覆盖规模。`--max-designs` 控制本次最多访问多少容量点，未访问点保留在全局下界中，提前退出不会误报完整穷举最优。`--no-screen` 关闭按原权重计算的风险下界筛选。非空输出目录默认拒绝覆盖。

16/128 场景测试的两套风机规则都出现了“训练通过、独立验证失败”，原始失败证据完整保留。因此小样本默认配置用于快速测试，不能把其最优点直接作为已验证的建设建议。扩大样本后仍只声明有限样本结果，详见 `docs/zhongshan_recovery_results.md`。并行脚本分区运行串行入口，合并精确场景存档后再由原完整网格规划器计算全局库内最优；它是计算实现上的加速，不是新增数学割。

默认测试配置是 `config/zhongshan_recovery_stress.toml`。**该配置 `synthetic=true`，使用中山原始负荷、风光曲线和气温，叠加显式工程假设；结果不得称为现场标定的最终建设方案。** 不同输出目录分别保存合成物理测试和中山曲线压力测试。

| 输入 | 本地核查及处理 |
|---|---|
| 中山时序 | 原始 CSV 为 2020 闰年 8784 小时，时间连续；默认取前 8760 小时，即 1 月 1 日至 12 月 30 日，不称完整 2020 日历年 |
| 气温 | 读取同一 CSV 的 `温度(摄氏度)`，不造温度，不自动补气温缺值 |
| 风光与聚合用电 | 复用既有加载器；374 个太阳辐射缺值沿用已有补值规则，记录在 manifest |
| 时区 | 源 CSV 未标注，配置显式声明假设 UTC+08，正式数据需确认 |
| 母线负荷 | CSV 列名为用电功率；计量边界尚未核实，显式 `assumed_bus_electricity` |
| 人员/覆冰标签 | 风速阈值与气温条件构造同一组工程压力标签，保留 12 小时残冰；不声称识别了实测暴风雪 |
| 热参数 | UA=0.15 kW/K，C=1.5 kWh/K，加热器 12 kW，准备阈值 5℃，均为研究假设 |
| 机组/箱体/人员 | 每台 100 kW 柴油研究模块一个箱体，一组启动人员，人工准备 3 小时；修理过程用独立日历恢复参考 |
| 加热计量 | 新增辅助需求假设；历史负荷是否含同项未获确认，不能宣称无重复计量 |
| 柴油故障 | 在线 MTTF=1662 h；每个合法完整启动需求失败率 0.0013；几何日历修复均值 37 h |
| 柴油成本 | 复用 CSV 的资本成本和 4.5 kWh/kg 线性燃料参数；不称为这台 100 kW CAT 的真实油耗曲线 |

正式运行需设置 `synthetic=false` 并补齐来源声明、已确认的母线负荷/加热计量、热参数、箱体映射、人数、柴油额定解释和 `observed_csv` 联合天气标签。标签 CSV 必须具有带时区 `timestamp`，与所选时序逐点相同，含 `access_safe, icing_active, ice_power_factor, wind_protective_stop, extreme_hazard`。缺少关键参数立即报错。当前 cap_plan 适配器只接受 clean 风功率曲线；已含覆冰的净实测功率需要专门适配，不能重复乘覆冰损失。

模型按以下时序执行：小时起点接收故障/维修完成状态，控制器根据当前可见信息申请派遣与启动，合法启动需求才揭示成败，然后供电、推进准备与温度、更新储能，在线机组的运行故障于下一小时起点生效。当前小时末刚达到温度条件的备用机组最早下一小时发电。

准备与升温并行；`arrival_only` 允许已经到场的人员继续室内工作，`all_work_safe` 暂停并保留进度。班组在一次启动需求完成前占用一台机组；同一小时不在获知启动结果后追溯派遣到第二台。运行中机组不会因无法外出自动停机。修复后交接可以保留已完成的准备，但温度不重置，仍需启动班组合法到场。

1R1C 热更新使用精确离散式与 UA=0 极限式，按箱体计一次实际加热功率。既有电源和储能共同供电；没有电能就不能继续按额定功率升温。储能实际充放电、计效率、受可用 PCS 和 SOC 约束；没有虚构消纳负荷。最低出力无法吸收时，报告该控制策略执行失败。

故障随机数按种子、场景、物理模块、机制和小时索引生成 Philox 流。改变容量或运行顺序不改变共享模块的基本随机数；真实柴油故障仍依赖各方案的在线行为。控制器的 `Observation` 不含未来故障、随机数、真实维修完成时刻或未来实测天气。本版只用当前测量，无完美预报。

默认用显式循环拼接的 336 小时天气热身，然后连续计量 8760 小时。不宣称热身保证平稳；事故之间不重置 SOC、温度或人员，年末不免费补能或修完机组。输出保存每个场景初末库存、温度和残余维修。热身长度需要另做敏感性，温度模型也不等同于发动机内部热状态或通风安全模型。

目标是年化投资加同一控制策略的样本平均实际运行费用。燃料、已设置的启动和储能吞吐成本来自实际轨迹，电加热通过电平衡进入燃料费用，不重复按外购电价收费。缺乏依据的启停/磨损价格显式为零并记录实物量。箱体、加热器和保温选型在当前实验中固定，缺乏独立价格时不制造新增资本报价；已有柴油模块单价是否已包含这些设备仍需核实，当前经济结果不是设备采购预算。

EENS 与 CVaR 对完整时域的场景失供量计算，CVaR 精确分摊分位点处的概率质量。必须同一策略同时通过双约束；不会把不同场景事后最优策略拼接。默认 EENS≤100 kWh、CVaR₀.₉₅≤1000 kWh 为继承研究限值。16 个训练样本的尾部有效样本仅 0.8，128 个验证样本对应 6.4，因此这些规模用于功能和初步数值验证，不构成罕见尾部充分收敛证据。

规划只在预声明的 48 个容量点和 2 套策略内比较。旧容量上限已由 100 kW 模块重新计算，但当前网格是其中的子集。样本内 gap=0 只证明此网格、样本、策略库内的最优性。所有运行成本非负，使用投资成本作为诚实但较弱的经济下界；当前没有调用 Gurobi、没有容量单调割、没有恢复 LP 耦合割。失败策略的排除与已访问容量管理都不会冒充对全部控制策略的物理不可行证明。

中断续算以完整年度场景为 checkpoint 粒度；中断中的单个场景由相同键控随机数从头重放。运行 manifest 核查源码、配置、原始数据、随机机制与 NumPy 版本；不会把旧 300 kW 缓存用于新模型。最终容量与策略在独立验证前冻结，验证失败如实输出 `holdout_failed`，不利用验证集再选策略。

每次输出包括 `run_manifest.json`、`migration_audit.md`、`resolved_config.json`、`hourly_weather.csv`、`capacity_result.json`、`scenario_losses.csv`、最坏样本 `hourly_dispatch.csv/event_log.csv/delay_decomposition.csv`、`audit_report.json`、`policy_evaluations.csv`、`visited_designs.jsonl`、`cuts.jsonl` 和 `checkpoints/paths.jsonl`。采用 CSV 而非 Parquet，嵌套逐机/箱体状态为 JSON 单元格，不增加 pyarrow 依赖。

审计独立重算选定完整轨迹的功率、SOC、热更新、启动资格、人员数量及成本/风险；全部仿真路径同时检查电平衡和储能约束。排故结束时间可以保存在事后日志中，但不会进入控制器观察。延迟原因按互斥区间计数，另存准备完成、温度达标等并行里程碑，不把并行过程重复相加。

退出码：0 完成，1 输入/配置错误，2 没有找到库内可行容量或未决，3 独立验证失败，4 轨迹审计失败。`population_certified` 始终为 false。
