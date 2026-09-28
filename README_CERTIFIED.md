# 独立规划程序：单调容量搜索与总体可靠性证书

这是根据任务 [寻找微电网可靠性创新算法](codex://threads/01a08931-6119-7901-989c-82dc8be8546b) **最后确定的 formulation 和收敛条件**建立的独立实现。对应的主线是固定概率模型、共享独立年度样本、容量单调性、可精化 UC 区间，以及经济最优性证书搜索。

新入口为 `./run_certified.sh`，代码在 `certified_reliability_planning/`。旧 `main.py`、`run.sh`、`polar_reliability_planning/`、原配置和结果均未修改。新程序通过只读导入复用原始物理模型、数据读取和调度审计，使用独立的采样管理、求解接口、规划器和输出目录。构建前的 90 个已有跟踪文件 SHA-256 记录在 `certified_reliability_planning/legacy_sha256.json`。

这是一个可运行、可检验的保守认证基线，包含中山站实际数据和 Gurobi UC 接口。当前没有紧尾部概率界、重要抽样、非预见策略或已证明的加速率。大年度损失界使 DKW 认证非常保守，默认有限预算可能返回未决；这属于明确的运行结果。

## 运行

```bash
cd /home/yzk/reliable_plan

# 无需 Gurobi 许可的解析算例：真实最优解已知，检查全部传播标签
./run_certified.sh validate-small

# 同一解析规划示范，保留完整动作日志、样本和区间证据
./run_certified.sh demo

# 中山站：读取独立新配置，再只读引用原文献参照系统参数
./run_certified.sh plan --config config/certified_zhongshan.toml

# 中山站：经济求解和最终证书采用相对 gap 1%
./run_certified.sh plan --config config/certified_zhongshan_1pct.toml --mip-gap 0.01

# 取消总规划时间上限；单次求解、样本、阶段和调用次数仍遵循配置
./run_certified.sh plan --config config/certified_zhongshan_1pct.toml --no-time-limit

# 24 小时端到端检查；小样本下预期返回 budget_exhausted，退出码 4
./run_certified.sh plan --hours 24 \
  --initial-samples 4 --max-samples 8 --max-stages 2 \
  --max-oracle-calls 32 --max-economic-calls 10 \
  --base-call-seconds 3 --max-call-seconds 6 --max-seconds 45

# 指定统计置信参数、绝对经济容差和计算预算
./run_certified.sh plan --delta 0.05 --epsilon-cost 1000 \
  --initial-samples 16 --max-samples 1024 --max-stages 10 \
  --max-oracle-calls 10000 --max-economic-calls 200 --max-seconds 3600

# 验证旧文件未发生改变
./run_certified.sh verify-isolation

# 新程序与旧程序分别测试
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s new_tests -v
/home/yzk/cap_plan/venv/bin/python -m unittest discover -s tests -v
```

`--output 新目录` 指定输出位置；非空目录或已有文件会被拒绝。默认目录为 `outputs/certified/时间戳/`，不复用任何旧运行目录。每次运行是新实验，暂不自动恢复检查点。

启动器默认沿用 `/home/yzk/cap_plan/venv/bin/python`。其他环境安装原 `requirements.txt` 中的依赖后，可执行 `python -m certified_reliability_planning demo` 或 `plan`。也可用环境变量 `CERTIFIED_PLAN_PYTHON` 指定解释器。解析示范只需要 NumPy；实际微电网规划需要 Gurobi 许可和外部 CSV。

## 状态含义

| 状态 | 退出码 | 可得结论 |
|---|---:|---|
| `certified_optimal` | 0 | 在指定模型、有效 Oracle 和同时统计事件条件下，返回满足原始双风险限值且成本差距不超过 ε 的容量 |
| `certified_infeasible` | 2 | 全部候选已被风险证据或名义运行不可行证明排除 |
| `budget_exhausted` | 4 | 样本、阶段、调用次数或时间预算不足，未得到完整证书；不能据此断言无解 |
| `error` | 1 | 输入错误或证据不一致，停止返回证书 |

若预算用尽时已有可靠性认证容量，`incumbent` 会保留该容量及可实施成本上界，同时 `certificate=null` 表明尚未完成经济最优性证书。若只有经济主问题找到容量，而统计风险仍未决，则该容量不会成为 `incumbent`；它及成本证据仍保留在动作日志和状态档案中。

默认 ε 是**元/所选时域的绝对成本容差**。新增 `--mip-gap 0.01`（同义参数 `--relative-gap 0.01`）或配置 `relative_gap = 0.01` 后，停止条件改为 **`(UB−LB)/abs(UB) ≤ 1%`**，原绝对 ε 不再作为可替代的停止条件；UB=LB=0 时相对 gap 定义为 0。经济主问题及固定容量经济复算设置相同相对 MIP gap，并关闭绝对 gap 提前停止。可靠性 UC 仍返回可精化的损失区间，不把 1% 当成风险容差。报告中的 `optimality_criterion` 写明本次采用的标准。

风险限值是**kWh/所选时域**；缩短到 24 或 168 小时时不会自动年化或缩放。δ 覆盖整个单次搜索的全部容量、全部样本前缀和所有有效精化状态，而非每次测试各自允许 δ。重复开展多个实验不自动共享一个整体错误概率预算。

## 已实现的规划步骤

1. 保留完整允许容量网格。中山站当前输入包含 568,512 个候选，初始逐点成本下界采用投资成本；总成本可以随容量增加而下降。
2. 经济区域 MILP 直接寻找仍允许区域内的便宜容量并给出全局下界。只有获得统计失败证据后，才加入该容量的下闭区域 cut；可靠性可行容量继续留在经济搜索中。
3. 每个新增年度索引生成完整天气、逐机故障和维修轨迹。所有容量复用相同索引与设备前缀，扩展样本时旧轨迹保持不变。
4. 对每个容量—场景对保存 `[l,u]`。未求解项为 `[0,B]`；下界来自有效 MILP 对偶界，上界来自经审计的可行调度。提前结束求解仍返回区间，后续求解与旧区间取交。
5. 将逐路径容量单调性用于区间传播：较大容量的损失下界可传播给较小容量，较小容量的损失上界可传播给较大容量。只假设容量单调性，不假设故障场景之间的排序。
6. 计算同时有效的 EENS/CVaR 上下界，执行双指标通过或任一指标失败的分类，再传播可靠性标签。
7. 根据仍有经济竞争力的未决点，比较平均 UC 区间宽度与统计误差，选择保留样本继续精化或增加样本。较宽的路径区间优先求解；阶段精度目标逐步下降。
8. 经济提议之外保留循环访问机制。没有区域 Oracle 的解析基线按有限网格逐阶段遍历；实际规划每阶段进行有限次提议，并安排循环访问，防止永久忽略经济竞争点。
9. 维护可靠 incumbent、全部未排除设计的经济下界和成本差距。按本次绝对或相对停止标准，仅在证书条件成立时宣布最优或无解，资源不足则明确返回未决。

比较抽样与求解误差时不会让已不具经济竞争力的旧点阻止新样本增长。经济上跳过的区域仍保留其成本下界，不能从全局最优性证明中悄悄删除。

## 统计界与证书

设容量空间基数上界为 K，年度损失有已知确定性上界 B。所有容量共享 `m` 条独立年度轨迹。对每个容量和样本前缀分配：

\[
\delta_{n,m}=\frac{6\delta}{\pi^2Km^2},\qquad
r_m=\sqrt{\frac{\log(2/\delta_{n,m})}{2m}}.
\]

对 `0 ≤ x < B` 构造：

\[
\underline F(x)=\operatorname{clip}_{[0,1]}
\left(\frac1m\sum_s\mathbf1\{u_s\le x\}-r_m\right),\qquad
\overline F(x)=\operatorname{clip}_{[0,1]}
\left(\frac1m\sum_s\mathbf1\{l_s\le x\}+r_m\right).
\]

支撑区间外的 CDF 为 0/1；数值实现显式保留 0 和 B 处的概率质量。分母始终为全部 `m`，自适应选择的一部分已解场景不能独立当成统计样本集合。

在概率至少 `1−δ` 的共同覆盖事件上，上述 CDF 包住所有容量的真实损失分布及所有样本前缀。跨容量不要求独立，独立性要求落在年度样本索引上。

通过对阶梯 CDF 精确积分得到 EENS；CVaR 按离散概率分布的分数尾部质量精确计算，不使用 η 网格近似。这对应原对话中的连续 η 定义。

\[
U_E\le\overline E,\quad U_C\le\overline C
\ \Longrightarrow\ \text{可行认证};\qquad
L_E>\overline E\ \text{或}\ L_C>\overline C
\ \Longrightarrow\ \text{不可行认证}.
\]

最终还需要可行容量的可实施成本上界 `UB` 与全局成本下界 `LB` 满足默认的 `UB−LB≤ε`，或相对模式下的 `(UB−LB)/abs(UB)≤relative_gap`。若同时使用逐点下界和区域 MILP 下界，取两者中较强的有效值。区域 MILP 在加入失败 cut 后重新求解，即使时限内没有 incumbent，也保留有效对偶界。

日志中同时给出经验损失界和总体置信界；二者含义不同。`g=mean(u−l)` 反映 UC 未知量，保守宽度分解为：

\[
U_E-L_E\le g+2Br_m,\qquad
U_C-L_C\le\frac{g+2Br_m}{1-\alpha}.
\]

B 从全时域负荷和最坏天气负荷系数推导，并向外处理浮点求和误差。**观察到的最大失供量不能替代 B。** 代码中的样本概率模型来自指定参数，不包含故障参数估计误差或站点迁移误差。

## 收敛范围与数值限制

与原对话一致，需要区分证书有效性和有限终止。在有限容量集、真实风险严格离开阈值、年度样本独立同分布且损失有界、有效同时置信界、关键容量公平获得计算、平均 UC 区间宽度趋零、经济上下界按需收敛，以及正 ε 条件下，可建立高概率有限终止结论。

实际程序设置了有限的 `max_samples`、`max_stages`、调用次数与时间预算，并且 `max_call_seconds` 限制单次求解，因此不会承诺对任意输入总能完成认证。移除有限预算之后，仍需 Oracle 可以在有限工作内达到所需正精度；公平规则本身不能替代求解器的精度收敛。风险恰在阈值上时可能长期未决。

Gurobi 采用数值求解，调度采用原有独立物理与 UC 审计。对求解器端点增加了公开的向外数值余量；该余量不是对任意病态 MILP 的严格有理数误差证明。故实际 UC 证书条件包括求解器上下界与审计数值语义的有效性。这个固定数值余量也构成精度下限：经济 ε 或风险余量过小时，不能声称误差自动趋零。相对模式设置固定的经济求解 gap，不能直接沿用经济 Oracle 区间宽度趋零的充分条件；最终证书仍须逐项核对实际上下界差距。单次求解达到 1% 不自动代表整个可靠性规划取得 1% 最优性证书。

完全预见 UC、可自由选择但首尾循环的初始可用储能、理想电芯与可故障 PCS 均沿用现有物理模型。统计认证不产生可因果执行的非预见运行策略。这些假设会写入每次 `resolved_config.json` 和结果摘要。

`max_seconds > 0` 约束规划循环；配置 `max_seconds = 0`、CLI `--max-seconds 0` 或 `--no-time-limit` 取消总时间上限。`--max-seconds` 与 `--no-time-limit` 不能同时指定。无总时限模式保留有限的单次 `max_call_seconds`，以便调度器继续精化区间和公平访问其他容量；样本、阶段和调用次数上限也保持有效，仍可能返回 `budget_exhausted`。负数、NaN 和无穷大的时间配置会被拒绝。

Oracle 返回的实际时间包含构模与审计。正在构造模型、生成样本或写盘的操作不能被硬实时中断，最终端到端耗时可能略超过设置的有限总预算。每次 operation 重新建立模型并与旧区间取交；当前还没有跨调用保留分支树或自动恢复能力。

## 文件与输出

| 文件 | 作用 |
|---|---|
| `planner.py` | 经济目标导向选择、公平访问、区间传播、风险标签与停止规则 |
| `risk.py` | DKW 同时 CDF 带、分数尾部 CVaR、两类误差宽度 |
| `oracles.py` | 有限预算 UC、经济区间、可行调度审计 |
| `economic_search.py` | 完整剩余区域的经济 MILP、统计失败 cut、全局对偶界 |
| `sampling.py` | 可追加共享年度流、确定性损失界、原随机轨迹兼容 |
| `benchmarks.py` | 真实总体风险和最优成本可解析计算的独立基准 |
| `cli.py` | 独立入口、配置解析、证据存档和错误状态 |

每次运行输出：

- `resolved_config.json`：实际参数、输入 CSV 指纹、容量网格、成本与信息结构；实际站点运行另记录系统配置 SHA-256。
- `events.jsonl`：新增阶段、每次经济/UC 区间响应，以及风险分类证据。
- `progress.json`：当前阶段或最终状态。
- `certificate_state.npz`：全部候选、成本区间、可靠性标签和已查询容量的路径区间。
- `summary.json`：结果状态、证书或未决原因、已认证 incumbent、经济差距与复杂度计数。
- `shared_scenarios.npz`：实际站点的共享天气与逐机故障轨迹；解析算例对应 `iid_uniform_samples.npy`。
- `nominal_dispatch.npz`：已有认证 incumbent 时导出的可实施名义调度。
- `candidate_dispatch.npz`：当前最便宜名义可行候选的调度，即使统计风险仍未决也可查看；摘要 `best_nominal_candidate.reliability_certified` 明确其认证状态，不能当成已认证规划结果。
- `population_truth.json`：仅解析验证生成，保存独立的精确总体枚举参考。

复杂度分开记录：年度样本数 M、实际查询容量—场景对数 J、运行 Oracle 次数 R、累计 Gurobi Work、UC/经济/样本生成时间和整体规划耗时。解析基准的工作量为单位查询费用，与 Gurobi Work 不是相同计费单位，不能直接比较数值大小。

具体运行结果见 [新程序验证记录](docs/certified_planner_validation.md)。
