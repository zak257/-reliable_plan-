# 历史验证记录：无 UC 的第一版

**当前代码默认已加入机组启停、充放电互斥和 EENS/CVaR 双约束。下文仅为旧版实验记录，输出链接指向原本地工作区；运行生成文件不随源代码发布，旧数值不能作为新版结果。**

以下结果来自本工作区的实际 Gurobi 求解。Python 为 `/home/yzk/cap_plan/venv/bin/python`（3.12.3），Gurobi 为 13.0.1，源数据目录为 `/home/yzk/cap_plan/data`。原项目代码和数据未被修改。

## 自动化测试

执行：

```bash
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s tests -v
```

**23 项测试全部通过。** 覆盖：

- 全年 `changcheng`、`zhongshan` 的负荷、风光出力、模块边界、投资及燃料成本与 `cap_plan_dual_bound_gurobi.CaseData` 一致。
- 无故障主问题与固定容量经济运行 LP 的成本一致、无失供。
- 无储能时最小失供与逐小时解析短缺公式一致。
- 储能时域闭合不能创造初始免费电量；充放电效率和 PCS 故障限制正确。
- 48 点小系统中，各场景失供量对五个容量分量逐一单调不增。
- 单调 cut 的 indicator 表达与整张网格的逻辑判定一致，上界点 cut 正确导致不可行。
- 固定随机流在增加样本数、设备模块上界时保留已有轨迹，NPZ 保存/加载指纹一致。
- 故障设备能维修恢复，气候状态改变故障概率，不支持的电芯故障配置明确报错。
- EENS 与不等权/部分尾部概率的 CVaR 正确，CVaR 限值可单独改变可行性。
- 普通 cut 和可选坐标提升都保留小系统精确最优值。
- 迭代上限不输出未通过的容量；敏感性分析使用同一场景集并记录对应限值。

## 小系统精确穷举

命令：`./run.sh validate-small --output outputs/small_grid_validation`

4 小时、风/光/柴油/电池/PCS 五个容量维度，共 48 个组合；采用概率为 0.6、0.3、0.1 的三个明确列出的故障场景，EENS 限值为 0.05 kWh。

| 指标 | 边界 cut 搜索 | 完整穷举 |
|---|---:|---:|
| 最优时域成本（元） | 11.3 | 11.3 |
| 最优模块向量（风、光、柴油、电池、PCS） | (1,1,1,1,1) | (1,1,1,1,1) |
| 最终 EENS（kWh） | 0 | 0 |
| Oracle 容量评价次数 | 4 | 32（其余 16 点标称不可行） |
| 主循环轮数 | 4 | 48 个容量点 |
| 可靠性 cut | 3 | — |
| 被错误排除的可行容量 | 0 | — |

原始结果：[summary.json](../outputs/small_grid_validation/summary.json)、[grid.csv](../outputs/small_grid_validation/grid.csv)。这个结果验证有限算例上的正确性，不能据此断言相对文献算法的一般速度优势。

## 长城站全年规划

8760 小时标称用电量为 **1,163,937.6 kWh**。下表两组可靠性规划使用完全相同的 128 个规划场景和 256 个独立验证场景，便于比较。

| 指标 | 默认 EENS 限值（0.1% 用电量） | 严格 EENS 限值 |
|---|---:|---:|
| EENS 上限（kWh/年） | 1,163.9376 | 100 |
| 风电（kW） | 400 | 400 |
| 光伏（kW） | 400 | 400 |
| 柴油（kW） | 300 | 600 |
| 电池（kWh） | 700 | 700 |
| PCS（kW） | 100 | 100 |
| 年均资本成本（元/年） | 817,500 | 832,500 |
| 标称燃油成本（元/年） | 904,054.004065 | 904,054.004065 |
| 总成本（元/年） | **1,721,554.004065** | **1,736,554.004065** |
| 规划样本 EENS（kWh/年） | 1,145.740114 | 0 |
| 规划样本 CVaR₀.₉₅（kWh/年） | 10,693.527124 | 0 |
| 独立样本 EENS（kWh/年） | **1,087.252779** | **27.125093** |
| 独立样本 CVaR₀.₉₅（kWh/年） | 8,478.619421 | 542.501859 |
| 规划轮数 / cut 数 | 1 / 0 | **27 / 26** |
| 主问题最终 gap | 0 | 0 |
| 独立 EENS 验证 | 通过 | 通过 |

上述两组未施加 CVaR 上限，CVaR 只作为附加风险指标报告。严格限值增加一台 300 kW 柴油机，成本增加 **15,000 元/年，约 0.87%**。严格算例共求解 3,456 个规划场景运行 LP；主循环约 334 秒，另有场景生成和独立验证耗时。不同机器、并行负载和 Gurobi 运行状态会改变耗时。

无故障经济基线与默认限值方案容量及成本一致，全年无故障失供为 0。两个最终方案的固定容量运行复算成本与主问题成本相差约 $9.3\times10^{-10}$ 元。

原始输出：

- [无故障基线 summary.json](../outputs/changcheng_8760_baseline/summary.json)
- [默认限值 summary.json](../outputs/changcheng_8760_default_128/summary.json)
- [严格限值 summary.json](../outputs/changcheng_8760_eens100/summary.json)
- [严格算例逐轮历史](../outputs/changcheng_8760_eens100/history.csv)
- [严格算例最终 8760 小时运行](../outputs/changcheng_8760_eens100/dispatch.csv)
- [严格算例独立样本失供量](../outputs/changcheng_8760_eens100/validation_losses.csv)

复现命令（使用新的输出目录）：

```bash
./run.sh baseline --case changcheng --output outputs/reproduce_baseline
./run.sh plan --case changcheng --eens-limit 100 --output outputs/reproduce_eens100
./run.sh plan --case changcheng \
  --scenarios outputs/reproduce_eens100/training_scenarios.npz \
  --output outputs/reproduce_default
```

当前配置中的规划种子为 `20260906`，独立验证种子为 `20260907`。固定规划数组指纹为 `6165749e127afe7944122939d15ea64c09d98c17244d556c749c167e42a822e2`；独立验证数组指纹为 `c90a7a995b4a5c0c0134d7f0f8af40d411492e40c80f8fa5a88e2d8614b52b25`。逐文件输入 SHA256 见各次运行的 `input_manifest.json`。

## 保留的样本不足与入口检查结果

首次用 16 个规划样本 / 32 个独立样本测试默认限值时，规划 EENS 为 147.550555 kWh，而独立 EENS 为 1,501.627921 kWh，超过 1,163.9376 kWh。程序返回退出码 3，结果保留在 [首次少样本结果](../outputs/changcheng_8760_reliability/summary.json)。随后增大为 128/256，并保持种子和模块轨迹的前缀不变。

气候 + CVaR 的 168 小时入口也实际运行过：8 个规划样本得到 EENS=0，8 个独立样本得到 EENS=95.432587 kWh，超过该短时域 21.0096 kWh 的 EENS 限值；CVaR=763.460699 kWh 低于所设 1000 kWh 上限。该运行同样明确返回 `holdout_failed`，见 [气候/CVaR 入口结果](../outputs/climate_cvar_168_smoke/summary.json)。这验证的是配置和评价链路，不是已收敛的气候论文算例。

另外已运行 24 小时 EENS 限值敏感性入口，以及全零容量评价。全零容量的 EENS、CVaR 都等于该时域全部负荷 **3,038.4 kWh**，与物理预期一致。

所有故障率、维修时间和天气转移参数仍为显式研究假设。上述独立验证只报告有限样本经验指标，尚未计算统计置信上界；规划样本 EENS=0 也不代表总体失供概率为零。
