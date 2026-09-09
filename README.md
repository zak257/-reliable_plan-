# 可靠性边界识别容量规划：UC + EENS/CVaR

当前版本联合规划风电、光伏、柴油、电池和 PCS 容量，规划和故障场景运行均考虑柴油机启停、最低稳定出力、最小开停机时间，以及储能充放电互斥。可靠性同时要求 EENS 与 CVaR 通过。

公开仓库：https://github.com/zak257/-reliable_plan-

计算流程：**容量 MILP → 固定故障场景运行 → EENS/CVaR → 单调可靠性 cut → 重新规划**。完整方程与边界条件见 [模型说明](docs/model.md)。

按“数学模型公式、输入参数表、256/2000 场景结果对比”整理的中文说明见 [模型结构与中山算例对比](docs/model_structure_and_zhongshan_comparison.md)。

最新完成了 [故障模型及文献参数核查](docs/failure_parameter_review.md)，并按 **8760 小时、2000 个规划场景、10000 个独立验证场景**运行 [文献参照敏感性版本](docs/zhongshan_literature_results.md)。新方案为风电 700 kW、光伏 300 kW、柴油 900 kW、电池 1200 kWh、PCS 300 kW；年成本 **3,508,931.70 元**，gap **0.909327%**。规划 EENS/CVaR 为 23.007381 / 460.147629 kWh，独立验证为 **12.661689 / 253.233778 kWh**，双约束通过，完整运行 993.564 秒。47 项测试通过，逐机统计证实任意已安装机组均可故障，也可多机同时故障。新参数注明了运行小时与日历小时的换算限制，属于连续暴露的文献参照压力情景，尚未取得中山站实测故障校准。

原研究假设的 **中山站 8760 小时、2000 个规划场景与 10000 个独立验证场景**见 [原大样本结果报告](docs/zhongshan_large_validation_results.md)。方案为风电 700 kW、光伏 300 kW、柴油 600 kW、电池 1450 kWh、PCS 300 kW；成本 3,362,791.24 元/年，实际 gap **0.988921%**，达到设定的 1%。规划 EENS/CVaR 为 9.869889 / 197.397773 kWh，独立验证为 **12.508809 / 250.176188 kWh**，双约束均通过。每次求解时限设为 3600 秒，完整流程实际耗时 783.051 秒。报告包含全部 12000 个年度场景的损失、规划与独立验证的风险曲线，以及逐轮规划风险记录。

此前的全年运行见 [UC + 双风险验证报告](docs/uc_validation_results.md)。该次方案为风电 500 kW、光伏 300 kW、柴油 600 kW、电池 800 kWh、PCS 150 kW；128 个规划样本 EENS/CVaR 均为零，256 个独立样本分别为 26.022918 / 520.458354 kWh，低于 100 / 1000 kWh 的限值。主规划成本 1,756,226.80 元/年、有效成本下界 1,737,400.89 元/年，gap 1.071952%，状态为样本可行，尚未证明最优。

中山站也已完成 [8760 小时、256 个规划场景运行](docs/zhongshan_validation_results.md)：风电 700 kW、光伏 300 kW、柴油 600 kW、电池 1000 kWh、PCS 400 kW，主规划成本 3,376,864.54 元/年，gap 1.409050%。规划 EENS/CVaR 为 5.138991 / 102.779818 kWh，另 256 个独立验证场景为 43.161469 / 863.229374 kWh，双约束均通过。报告含月度发电图和完整场景损失表。

## 环境与运行

本地 `run.sh` 默认使用 `/home/yzk/cap_plan/venv/bin/python`，求解器为 Gurobi 13.0.1。数据直接读取 `/home/yzk/cap_plan/data`，原项目未被修改。

其他机器可以安装 `requirements.txt` 中的依赖、配置 Gurobi 许可，然后使用 `python main.py`。原站点 CSV 不包含在仓库中，通过 `--data-root` 指定数据目录；`validate-small` 使用内置合成数据，可以单独运行。

```bash
cd /home/yzk/reliable_plan

# 含 UC 和双风险约束的 48 点穷举对照
./run.sh validate-small

# 含 UC 的全年经济规划
./run.sh baseline --case changcheng

# 全年双指标可靠性规划
./run.sh plan --case changcheng

# 中山站：8760 小时、256 个规划场景及 256 个独立验证场景
./run.sh plan --case zhongshan --hours 8760 --samples 256 --validation-samples 256

# 中山站大样本版本：2000 个规划场景、10000 个独立验证场景
# 每次求解时限 3600 秒；经济主问题及成本复算 gap 1%，正失供评价 gap 0
./run.sh plan --config config/zhongshan_2000.toml --solver-config config/solver_3600_1pct.toml

# 文献参照故障参数：同样为 2000 / 10000 场景、每次 3600 秒、经济 gap 1%
# 注意：柴油运行 MTTF 按连续日历暴露使用；不是中山站实测参数
./run.sh plan --config config/zhongshan_literature_2000.toml --solver-config config/solver_3600_1pct.toml

# 自定义限值、尾部水平与样本数
./run.sh plan --eens-limit 100 --cvar-limit 1000 --alpha 0.95 --samples 128 --validation-samples 256

# 气候条件化故障
./run.sh plan --climate

# 短时域调试；限值均为当前时域 kWh，不自动年化
./run.sh plan --hours 168 --samples 8 --validation-samples 16

# 给定容量评价；电池为 kWh，其余为 kW
./run.sh evaluate --capacity '{"wind":400,"pv":400,"diesel":600,"battery_energy":700,"pcs":100}'

# 固定 CVaR 上限，改变 EENS 上限
./run.sh sensitivity --eens-limits 50,100,500

# 测试
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s tests -v
```

`--output 新目录` 指定结果目录，非空目录会被拒绝。`--scenarios training_scenarios.npz` 复用完整样本；`--validation-samples 0` 关闭独立验证；`--no-unit-commitment` 用于连续调度对照。`--time-limit` 和 `--oracle-time-limit` 分别控制主问题和场景子问题的时限。

## 默认参数

见 [系统配置](config/system_config.toml) 和 [求解器配置](config/solver_config.toml)。

| 内容 | 默认值 |
|---|---|
| 风 / 光 / 柴油模块 | 100 / 100 / 300 kW |
| 电池 / PCS 模块 | 50 kWh / 50 kW |
| 单台柴油最低出力 | 额定功率 20%，300 kW 单机对应 60 kW |
| 最小开机 / 停机时间 | 各 3 小时 |
| 初始机组状态 | 全部停机，已满足先前最小停机时间 |
| 时域末端 | 持续时间截断到末端，不强制年末停机 |
| 启动成本 | 0 元/次，与 cap_plan 相同；可配置校准值 |
| EENS 上限 | 100 kWh/当前时域 |
| CVaR 上限 / α | 1000 kWh/当前时域 / 0.95 |
| 规划 / 独立验证样本 | 128 / 256 |
| 规划 / 验证种子 | 20260906 / 20260907 |

年均资本、燃料成本、容量边界和风光曲线来自 cap_plan。UC 参数沿用其原生 Gurobi 模型中的常数。原 CSV 不提供故障统计，因此故障率、维修时间和天气参数仍是明确的研究假设。

## 可靠性解释

每个场景是一整年（或所选时域）的设备可用性和天气轨迹。给定容量后，以相同的机组启停及储能约束求最小失供电量 Q_s。EENS 限制平均失供电量，CVaR 限制最严重 5% 场景的平均失供电量；这里的尾部是**年度场景损失**，不是最差的 5% 小时。

两个指标必须同时满足。CVaR 精确计算边界场景的部分概率质量；128 个等权场景、α=0.95 时对应最坏的 6.4 个场景。

任一指标失败就排除当前容量及所有逐分量更小的容量。固定样本的模块前缀始终相同；新增柴油机可以保持停机，新增风光可以弃电，因此启停不会破坏可靠性随容量增加而改善的单调性。

规划可用已求解场景构成风险下界，安全提前排除失败候选。下界用 `metrics_are_lower_bounds=true` 明确标记，不能当作完整风险估计。所有接受方案和独立验证都评价全部场景。零失供子问题可以通过已审计的合法 UC 调度及目标非负下界证明最优；非最优的高失供可行解不会被用于失败 cut。

## 代码与输出

- `data/cap_plan_loader.py`：CSV、模块、成本和输入指纹。
- `scenario_generation/`：固定天气、故障和维修轨迹及 NPZ 存档。
- `reliability/unit_commitment.py`：逐机启停、故障例外、最小持续时间和运行审计。
- `reliability/operation_model.py`：标称与故障运行模型。
- `reliability/monte_carlo.py`：EENS/CVaR Oracle 和容量缓存。
- `planning/`：经济主问题、单调 cut 和迭代。
- `validation/`：完整穷举、独立验证及敏感性。

每次输出包括实际配置、输入 SHA256、固定场景、逐轮历史、最终场景损失和 `summary.json`。`dispatch.csv` 包含小时出力、电量、充放电状态及每台柴油机的在线/启动/停机状态，成本单列资本、燃油和启动项。

大样本运行另提供 `progress.json` 阶段进度，以及 `training_checkpoint.jsonl`、`validation_checkpoint.jsonl` 逐场景精确损失存档。存档可用于检查已完成的场景；当前不自动恢复中断的规划。正常可供电场景可直接构造合法调度，经功率、储能和启停审计后证明零失供；需要储能时先尝试固定启停及充放电模式的可行调度，其余场景仍求完整 MILP。故障生成和备用策略按状态变化事件加速，原有随机轨迹保持一致。

经济成本求解的 `optimal_within_gap` 表示达到所设 gap；只有上下界闭合时才填写固定容量最优成本。3600 秒是每次 Gurobi 优化调用的时限，整个场景实验可能更长。

`sample_optimal_within_gap` 只表示当前固定样本和容量网格上的成本 gap；`sample_feasible` 表示风险通过但成本 gap 尚未满足。独立验证另报 `holdout_passed` / `holdout_failed`。退出码为 0 完成、1 配置/求解错误、2 无可行规划结果、3 独立验证失败。

标称运行审计独立复核功率平衡、储能动态与边界、机组启停，并用主问题调度热启动固定容量成本复算。经济复算触及时限时保留可行成本、下界和 gap，`fixed_capacity_optimal_cost_yuan` 留空，继续独立可靠性验证；正失供量的可靠性子问题仍要求严格最优。中途写出的摘要以 `validation_status=pending` 表明验证尚未完成。

当前采用完全预见的场景调度，电芯理想、PCS 可故障；没有跨场景非预见性、频率安全或自适应统计置信停止。无 UC 的旧结果保存在 [历史验证记录](docs/validation_results.md)，不能当作当前版本数值。
