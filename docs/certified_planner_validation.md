# 独立认证规划程序：实现与验证记录

新程序根据 [寻找微电网可靠性创新算法](codex://threads/01a08931-6119-7901-989c-82dc8be8546b) 中最后确定的 formulation 实现。使用说明见 [README_CERTIFIED.md](../README_CERTIFIED.md)。本记录区分解析总体认证、实际 UC 接口检查与旧程序完整性，实际站点检查没有得到新容量的总体风险证书。

## 解析总体认证

命令：

```bash
./run_certified.sh validate-small --output outputs/certified/analytic_validation_final
```

该算例有 9 个容量点、两个等概率需求状态，真实损失与最优成本可以独立枚举。规划器只能访问共享独立样本及有界 Oracle 响应，真实总体表仅用于结果核验。

| 指标 | 结果 |
|---|---:|
| 返回状态 | `certified_optimal` |
| 容量点 | `(2, 1)` |
| 枚举总体最优成本 | 7 |
| 规划成本下界 | 7 |
| 规划成本上界 | 7.000078125 |
| 绝对成本差距 | 0.000078125 |
| 设定 ε / δ | 0.01 / 0.05 |
| 已知真实 EENS | 0.025 |
| 已知真实 CVaR₀.₅ | 0.05 |
| EENS / CVaR 限值 | 0.3 / 0.5 |
| 最终共享样本数 | 256 |
| 完成阶段数 | 5 |

总体最优解与规划结果一致，所有统计标签及其传播区域均逐点对照真实总体值核验。基准总成本刻意不单调：`(1,1)` 成本约 15，增容到 `(2,1)` 后成本为 7，再增容到 `(2,2)` 后为 10。因而测试能够发现把可靠性上闭区域错误当作成本剪枝区域的问题。

结果文件：`outputs/certified/analytic_validation_final/summary.json`；真实总体参考：同目录 `population_truth.json`；样本、区间及动作日志同时存档。

## 中山站实际模型

读取原 `config/zhongshan_literature_2000.toml` 的系统和故障参数，数据来自 `/home/yzk/cap_plan/data`，新随机种子为 20260910。未改变原容量网格，共 568,512 个候选。

| 运行 | 24 小时 | 8760 小时 |
|---|---:|---:|
| 阶段 / 样本预算 | 2 阶段，最多 8 样本 | 2 阶段，最多 8 样本 |
| 时间预算 | 45 秒 | 70 秒 |
| 实际规划耗时 | 0.226 秒 | 70.096 秒 |
| 生成完整轨迹数 | 8 | 8 |
| 运行 Oracle 调用 | 6 | 2 |
| 经济 Oracle 调用（含区域主问题） | 8 | 6 |
| 成本下界（元/所选时域） | 8,934.613837 | 3,157,447.256017 |
| 返回状态 | `budget_exhausted` | `budget_exhausted` |
| 总体风险证书 | 未获得 | 未获得 |

这些运行检验了真实 CSV、全年数据长度、经济容量主问题、逐机故障共享轨迹、合法零失供构造、有限求解预算与证据输出链路。它们**不表示**得到新中山站最优容量，也不构成全年风险认证成功或算法加速证明。正失供、强制多机故障、有限求解区间和实际 UC 最优值的关系另由自动测试验证。

24 小时运行命令：

```bash
./run_certified.sh plan --hours 24 --initial-samples 4 --max-samples 8 \
  --max-stages 2 --max-oracle-calls 32 --max-economic-calls 10 \
  --base-call-seconds 3 --max-call-seconds 6 --max-seconds 45 \
  --output outputs/certified/zhongshan_24h_smoke_v1
```

全年运行命令：

```bash
./run_certified.sh plan --initial-samples 4 --max-samples 8 \
  --max-stages 2 --max-oracle-calls 32 --max-economic-calls 6 \
  --base-call-seconds 20 --max-call-seconds 30 --max-seconds 70 \
  --output outputs/certified/zhongshan_8760h_smoke_v1
```

两次退出码均为 4，表示预算用尽仍未决。用以上路径重跑时会因目录已有结果而被拒绝，应指定新的输出目录。

全年确定性损失上界约为 **2,038,009.397 kWh**，原风险限值为 EENS 100 kWh、CVaR₀.₉₅ 1000 kWh。以该自然界构造的分布无关 DKW 带很宽；即使已求解场景全部零失供，也不能把总体风险上界直接写成零。紧尾部界仍是决定年度认证实用性的后续研究重点。

## 自动测试与完整性

新测试覆盖同时置信事件、分数尾部、0/B 端点、未解样本完整分母、样本前缀稳定性、逐机身份、UC 正损失区间、实际有限求解响应、调度审计失败、成本不可行与时限的区别、单调传播、经济非单调、全局经济界、公平抽样推进、错误证据拒绝，以及输出防覆盖。

最终新程序 53 项测试、原有程序 48 项测试，共 101 项全部通过，无失败、错误或跳过。原有 90 个跟踪文件 SHA-256 全部保持一致，`git diff` 对已有文件为空；旧入口和旧结果未被替换。

完整最终测试计数、日志位置与新增源文件指纹记录于 [验证机器记录](certified_planner_verification.json)。数值求解器、完全预见模型与有限预算的适用边界见新程序 README。
