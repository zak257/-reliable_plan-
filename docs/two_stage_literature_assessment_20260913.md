# 二阶段可靠性规划与单调逻辑切：文献核验和发表定位

检索与核验日期：2026-09-13（当地日期）。本报告针对当前仓库中的容量规划、全年故障运行评价与可靠性反馈方法。只新增本报告，未修改模型、代码或既有实验。

## 结论

1. **“没有人用投资主问题—可靠性子问题—切反馈做规划”的前提不成立。** Bloom（1983，Operations Research）和 da Costa 等（2021，IEEE Transactions on Power Systems）有明确先例。后者公开作者稿的式(1)、式(11)及 Sec. IV 与当前经济—可靠性反馈结构直接接近。
2. **不只是框架已有，当前基础切的数学形式也有直接先例。** Liu 等（2024，European Journal of Operational Research）的公开作者稿 Sec. 4.2.1 式(8)，就是禁止所有分量均不超过失败设计的整数向量：\(\neg(x\le x^j)=\bigvee_i(x_i\ge x_i^j+1)\)。该文研究能源系统设计与混合整数运行子问题。
3. **这类方法可以作为高水平规划论文的计算框架，已有发表事实支持；但“二阶段+单调性+基础析取切”不足以单独主张新的算法贡献。** 当前实现的全年故障 UC、EENS/CVaR、共享模块路径、保守失败证明和总体统计认证，需要分别与已有方法比较。未核实这些细节全部相同的论文，不等于证明无人做过，更不等于组合本身足以发表。
4. **不能把完全预见运行当作顶刊发表的否决条件。** 文献确有这种假设，也有针对其局限的改进。应限定可靠性结论对应的信息、初末状态与运行目标，并检查容量方案在实际可实施调度下的表现。

## 一、先明确比较对象

当前固定样本版本可概括为：

\[
\min_{n\in\mathcal N}\ C_{\rm cap}(n)+C_{\rm nominal}(n),
\quad \widehat{\mathrm{EENS}}(n)\le\bar E,
\quad \widehat{\mathrm{CVaR}}_\alpha(n)\le\bar C,
\]

其中

\[
Q_s(n)=\min_{y_s\in\mathcal Y_s(n)}\sum_t\mathrm{shed}_{s,t}\Delta t
\]

是固定容量、固定完整年度天气—故障路径下的最小失供。场景运行含逐机启停、最小开停机时间、最低出力、储能动态与充放电互斥，使用全年完全预见。当前储能初始状态允许随场景优化，并要求年末恢复初始状态。

经济主问题包含标称经济运行；故障场景的燃油和启机费用没有按其发生概率进入目标。故障运行以最小失供为目标。这个成本口径可以被明确定义，但与“真实运行策略下的期望总成本最小”不同。

失败容量 \(b\) 被可靠证明后，加入

\[
\bigvee_{j:b_j<\bar n_j}(n_j\ge b_j+1),
\]

排除 \(\{n:n\le b\}\)。容量上界处不能再增加的分量不构成有效分支。全部分量都在上界且仍失败时，可以证明当前容量网格无解。

较新版本把经验风险检查替换为指定概率模型下的总体 EENS/CVaR 认证，并使用可精化 UC 上下界、共享样本、自适应采样、同时置信界与经济上下界。不能把新版直接称为普通固定样本 SAA，也不能把其全部认证机制归结为上述基础切。

这里有三个容易混淆的概念：

- 随机规划的“阶段”表示决策时掌握的信息；先投资、后获知整条路径并运行，是一种两阶段信息假设。
- 主问题与子问题表示求解分解；出现两个计算模块，本身不证明随机规划的新信息结构。
- 用户的切持续反馈，允许所有容量重新选择，因此属于一体化问题的迭代分解。它不等同于“先固定经济方案，再只允许补装设备”的单向过程。

对固定样本、相互独立的场景运行决策与逐分量单调的风险指标，可把所有场景运行变量及 EENS/CVaR 约束直接放入一个扩展 MILP。外部评价和加切是在避免一次性建立或求解这个巨大模型；其存在不自动构成一种新的规划模型。

## 二、三个最直接的全文证据

### 2.1 da Costa 等：经济—可靠性分解架构已经用于 TPWRS 扩容规划

Luiz Carlos da Costa, Fernanda Souza Thome, Joaquim Dias Garcia, Mario V. F. Pereira. **Reliability-Constrained Power System Expansion Planning: A Stochastic Risk-Averse Optimization Approach**. IEEE TPWRS 36(1):97–106, 2021（online 2020）。

- [正式 DOI](https://doi.org/10.1109/TPWRS.2020.3007974)
- [公开作者稿全文](https://arxiv.org/pdf/1910.12972)

以下公式编号来自公开作者稿，并非声称正式排版页码完全相同。

式(1)定义 \(\min_x I(x)+O(x)\)，约束 \(R(x)\le\bar R\)。Sec. IV 把问题分成投资主问题、运行成本子问题和可靠性子问题。式(11)包含成本最优性切与可靠性可行性切；每轮把同一个投资方案送至两个子问题并反馈，直到获得满足可靠性要求的最低成本方案。

其可靠性模型以随机可用容量与负荷缺口计算 EPNS 或 CVaR；可靠性子问题没有当前代码的全年储能及逐机 UC。其实际经济运行模块使用多阶段 SDDP。因此，这篇证明大框架有直接先例，不能被描述为用户整套年度故障模型的完全复刻。

Sec. IV-C 明确指出：非凸子问题不能自动保证常规切不误删可行解；其标准分解不直接处理引入整数的 LOLP/VaR 形式。作者也讨论了逐点删除不可行整数方案的替代办法，原文为“remove infeasible integer solutions one by one”。这说明非凸运行需要不同的切，并非说明这种分解不存在。

**文献纠错：该 DOI 的作者是 da Costa 等，不能沿用此前答复中误写的“Leite 等”。**

### 2.2 Liu 等：能源设计论文已经写出相同基础切

Bingqian Liu, Côme Bissuel, François Courtot, Céline Gicquel, Dominique Quadri. **A generalized Benders decomposition approach for the optimal design of a local multi-energy system**. EJOR 318(1):43–54, 2024。

- [正式 DOI](https://doi.org/10.1016/j.ejor.2024.05.013)
- [公开作者稿全文](https://hal.science/hal-04802899/document)

本文先做能源设备设计，再求含整数变量的运行子问题。其 Proposition 4.1 利用可行域包含证明：资源向量逐分量增加，子问题最优值不增加。式(6d)给出 \(\mathbf1_{\{x\le x^j\}}=0\)；Sec. 4.2.1 式(8)明确给出

\[
\neg(x\le x^j)=\bigvee_i(x_i\ge x_i^j+1).
\]

因此，相同的是**排除区域及析取公式本身**，不只是笼统的 Benders 名称。不同的是该文研究多能源系统运行可行性与费用，不能据此说它已经研究了当前全年设备故障、EENS/CVaR 或总体统计认证。

严谨的对应需要注意：用户单个“最小 ENS”问题通常始终可行，风险失败不等于这个 UC 问题不可行。可将检查定义为“寻找所有场景运行，并同时满足 EENS 和 CVaR 门槛”的联合可行性问题；在当前独立场景运行模型中，各场景最小失供可以同时达到，因此风险失败等价于这个加入风险门槛的联合问题不可行。对其使用同型切成立。

本文 Sec. 4.3 还直接解释了基础方法为何可能慢：低容量候选产生的排除区域很小，初期接近逐个设计枚举；生成切需要求解多个 MILP，还会增加主问题的辅助二进制变量。作者因此加入 LP 松弛和经典 Benders 切加速。当前主问题已有标称运行，不能把该文“首轮零容量”的具体表现直接套用到用户；但基础切的弱点与反复求解成本确实相关。

### 2.3 Forbes 等：仿真反馈与单调切加强也已有系统研究

M. Forbes, M. Harris, H. Jansen, F. van der Schoot, T. Taimre. **Combining optimisation and simulation using logic-based Benders decomposition**. EJOR 312(3):840–854, 2024。

- [正式 DOI](https://doi.org/10.1016/j.ejor.2023.07.032)
- [公开作者稿全文](https://arxiv.org/pdf/2107.08390)

该文研究整数资源配置、单调仿真性能、固定样本 SAA 和逻辑 Benders。作者稿式(9)利用单调性，把当前值函数下界传播至逐分量更小的资源向量。式(10)把部分资源提高到最大值、检查性能是否仍可证明，以去除不必要的切分量。后续还推导更强的切并做消融比较。

它不是电力可靠性规划论文，也没有证明已实施用户的双风险总体认证。但它说明，“优化—仿真—反馈切”以及“额外评价以扩大切的有效区域”已有成熟研究。特别是不能把“把其他设备提高到上界后仍失败，因此只对剩余设备加约束”直接包装成从未出现过的算法原则。本文结尾提到 CVaR 作为可研究扩展，不能说该文已完成 CVaR 实验。

## 三、24 项相关文献及证据层级

“全文”包含明确标注的公开作者稿；“摘要”表示已核验摘要及出版信息，但未取得全文。未从摘要缺少某项描述推断论文一定不包含该项内容。

| 编号 | 文献及链接 | 访问层级 | 与本问题的关系及边界 |
|---|---|---|---|
| 1 | Bloom, 1983, Operations Research, [Solving an Electricity Generating Capacity Expansion Planning Problem by Generalized Benders’ Decomposition](https://doi.org/10.1287/opre.31.1.84) | 摘要 | 投资主问题、逐年运行子问题、概率可靠性与模拟反馈。证明框架有很早的先例；不推断其全年 UC 细节。 |
| 2 | da Costa et al., 2021, TPWRS, [Reliability-Constrained Power System Expansion Planning: A Stochastic Risk-Averse Optimization Approach](https://doi.org/10.1109/TPWRS.2020.3007974) | 作者稿全文 | 投资—经济运行—MC 可靠性分解，EPNS/CVaR 可行性切。当前最直接的电力规划架构先例；静态可靠性缺口不同于年度整数运行。 |
| 3 | Wei et al., 2022, IET Energy Systems Integration, [Multi-objective optimal configuration of stand-alone microgrids based on Benders decomposition considering power supply reliability](https://doi.org/10.1049/esi2.12060) | 摘要 | 风光柴储、发电与储能故障、经济/环境配置主问题与可靠性检查子问题互动。未确认具体切同式。 |
| 4 | Jirutitijaroen & Singh, 2008, TPWRS, [Reliability Constrained Multi-Area Adequacy Planning Using Stochastic Programming With Sample-Average Approximations](https://doi.org/10.1109/TPWRS.2008.919422) | 摘要 | 两阶段混合整数扩容、SAA、MC/Latin hypercube、上下界置信区间。不据题名声称同时包含当前两项硬风险限制。 |
| 5 | Peker, Kocaman & Kara, 2018, IJEPES, [A two-stage stochastic programming approach for reliability constrained power system expansion planning](https://doi.org/10.1016/j.ijepes.2018.06.013) | 全文 | 投资与事故补救、N−1、事故筛选/聚合，目标含事故期望运行费。并非当前年度 EENS/CVaR 外部切方法。 |
| 6 | Cao et al., 2020, TSG, [A Risk-Averse Conic Model for Networked Microgrids Planning With Reconfiguration and Reorganizations](https://doi.org/10.1109/TSG.2019.2927833) | 作者稿全文 | 微电网规划、孤岛运行与 CVaR；证明适用条件下 SOCP 强对偶并定制 Benders。CVaR 用在损失费用目标，非当前双硬约束。 |
| 7 | R. Wu & Sansavini, 2020, Applied Energy, [Integrating reliability and resilience to support the transition from passive distribution grids to islanding microgrids](https://doi.org/10.1016/j.apenergy.2020.115254) | 作者稿全文 | 设计主问题与故障事件评价、定制 C&CG、事件筛选、上下界；不是同一概率风险公式。 |
| 8 | X. Wu et al., 2021, TSG, [An MILP-Based Planning Model of a Photovoltaic/Diesel/Battery Stand-Alone Microgrid Considering the Reliability](https://doi.org/10.1109/TSG.2021.3084935) | 摘要 | 滚动顺序 MC、离线可靠性近似、在线 MILP；摘要明确反复可靠性模拟的计算负担。属于替代路线。 |
| 9 | Xie et al., 2024, IJEPES, [A reliability-constrained planning model for antarctic electricity and heat integrated energy system](https://doi.org/10.1016/j.ijepes.2024.110346) | 摘要 | 南极综合能源、风光柴储氢、滚动 MC、可靠性函数近似。说明南极场景标签和可靠性规划组合已有先例。 |
| 10 | Xiong, Shen & Sun, 2025, TSG, [Two-Stage Robust Planning for Park-Level Integrated Energy System Considering Uncertain Equipment Contingency](https://doi.org/10.1109/TSG.2024.3524879) | 摘要 | 日内故障与维修、两阶段鲁棒规划、整数追索及改进嵌套 C&CG。是顶刊接受相关结构的近年证据，风险口径不同。 |
| 11 | Liu et al., 2024, EJOR, [A generalized Benders decomposition approach for the optimal design of a local multi-energy system](https://doi.org/10.1016/j.ejor.2024.05.013) | 作者稿全文 | 能源设计、整数运行、单调性、**相同基础析取切**、有限收敛、LP 加强。应列为核心算法对照。 |
| 12 | Forbes et al., 2024, EJOR, [Combining optimisation and simulation using logic-based Benders decomposition](https://doi.org/10.1016/j.ejor.2023.07.032) | 作者稿全文 | 单调仿真性能、SAA、单调和局部加强切。不是电力模型，但算法关系直接。 |
| 13 | Liu & Gicquel, 2026, [Indicator Cuts for Benders Decomposition with Mixed-Integer Subproblems](https://optimization-online.org/2026/08/indicator-cuts-for-benders-decomposition-with-mixed-integer-subproblems/) | 预印本全文 | 2026-08-14 发布，统一单调值函数 indicator cuts；式(46)含当前排除形式为特例。**仅按预印本处理，不充当已发表顶刊证据。** |
| 14 | Hooker & Ottosson, 2003, Mathematical Programming, [Logic-based Benders decomposition](https://doi.org/10.1007/s10107-003-0375-9) | 摘要 | 用逻辑推理产生切的基础框架；不声称本文已含当前具体年度风险切。 |
| 15 | Laporte & Louveaux, 1993, Operations Research Letters, [The integer L-shaped method for stochastic integer programs with complete recourse](https://doi.org/10.1016/0167-6377(93)90002-X) | 摘要 | 整数追索的经典算法来源；其 complete-recourse 最优性切不应与当前失败切混称。 |
| 16 | Tuy, Minoux & Hoai-Phuong, 2006, SIAM Journal on Optimization, [Discrete Monotonic Optimization with Application to a Discrete Location Problem](https://doi.org/10.1137/04060932X) | 摘要 | 摘要明确研究 monotonicity cuts；未以未取得的全文宣称与当前切同式。 |
| 17 | Dratsas, Psarros & Papathanassiou, 2024, TPWRS, [A Real-Time Redispatch Method to Evaluate the Contribution of Storage to Capacity Adequacy](https://doi.org/10.1109/TPWRS.2023.3243669) | 摘要 | 明确比较预知故障/纯可靠性调度与经济日程加实时重调度；用于解释信息假设边界。正式年份 2024。 |
| 18 | Misconel et al., 2022, RSER, [Systematic comparison of high-resolution electricity system modeling approaches focusing on investment, dispatch and generation adequacy](https://doi.org/10.1016/j.rser.2021.111785) | 摘要 | 统一输入下比较模型；24/36 小时滚动与全年完全预见影响结果。正式卷期年份 2022。 |
| 19 | Dowling et al., 2020, Joule, [Role of Long-Duration Energy Storage in Variable Renewable Electricity Systems](https://doi.org/10.1016/j.joule.2020.07.007) | 主文摘要及补充全文 | 补充 Sec. 2.1 明确采用 perfect foresight。证明这种假设可以用于高水平规划分析，不证明设备故障可靠性与用户同构。 |
| 20 | Pecci & Jenkins, 2025, TPWRS, [Regularized Benders Decomposition for High Performance Capacity Expansion Models](https://doi.org/10.1109/TPWRS.2025.3526413) | 作者稿全文 | 跨时段储能连接、正则化 Benders、大规模验证。**运行 UC 连续松弛，投资整数**；不能说已精确求解当前整数运行追索。 |
| 21 | Zou, Ahmed & Sun, 2019, TPWRS, [Multistage Stochastic Unit Commitment Using Stochastic Dual Dynamic Integer Programming](https://doi.org/10.1109/TPWRS.2018.2880996) | 摘要 | 整数 UC 与信息阶段导致的计算困难及多阶段算法；其第一阶段承诺与当前第一阶段容量不同。 |
| 22 | Cho, Li & Grossmann, 2022, Computers & Chemical Engineering, [Recent advances and challenges in optimization models for expansion planning of power systems and reliability optimization](https://doi.org/10.1016/j.compchemeng.2022.107924) | 作者稿全文 | 扩容、可靠性、聚合和分解综述；为时间精度、运行整数性和计算负担的取舍提供背景。 |
| 23 | Jooshaki et al., 2022, TPWRS, [An Enhanced MILP Model for Multistage Reliability-Constrained Distribution Network Expansion Planning](https://doi.org/10.1109/TPWRS.2021.3098065) | 摘要 | 显式可靠性 MILP 替代传统仿真优化；不能把其对传统启发式的批评套到所有有效切算法。 |
| 24 | Li et al., 2021, TPWRS, [A Reliability-Constrained Expansion Planning Model for Mesh Distribution Networks](https://doi.org/10.1109/TPWRS.2020.3015061) | 摘要 | 故障后恢复与显式可靠性约束的 MILP 路线；不同系统、不同风险结构。 |

主要公开全文补充入口：[Peker](https://repository.bilkent.edu.tr/bitstreams/290c805c-91de-4f5a-a90e-ac018247f734/download)、[Cao](https://www.osti.gov/servlets/purl/1870060)、[Wu & Sansavini](https://arxiv.org/pdf/2004.00877)、[Pecci & Jenkins](https://arxiv.org/pdf/2403.02559)、[Cho 综述](https://www.osti.gov/servlets/purl/1981586)、[Joule 补充材料](https://authors.library.caltech.edu/records/9djjb-yxd48/files/1-s2.0-S2542435120303251-mmc1.pdf)、[2026 indicator-cut 预印本](https://optimization-online.org/wp-content/uploads/2026/08/Inverse_Cone_Benders_Cut.pdf)。

## 四、为何有人选择其他计算路线

以下区分作者明确讨论的动机和基于数学结构的解释，不代替所有学者猜测真实动机。

### 4.1 年度时序、故障样本与整数运行共同抬高代价

一个候选容量可能要评价成千上万条 8760 小时轨迹，每条含整数 UC。外层容量迭代会继续放大这项代价。Wu（2021）的摘要明确用离线近似减少反复可靠性计算；Peker（2018）通过事故筛选和聚合降低场景负担；Pecci（2025）使用连续运行 UC，使标准对偶分解可用。这些是不同取舍，并非说明外部可靠性检查在数学上不可行。

### 4.2 基础逻辑切安全，却未必能快速描述边界

Liu（2024）Sec. 4.3 已直接讨论这一问题。举一个示意性例子：若真实可靠容量条件恰好为 \(n_1+n_2\ge K\)，一个线性不等式可以表示边界；逐个失败点的矩形下闭区域切，可能需要很多条才能描述同一集合。这个例子只说明切的表达效率，不声称用户的真实风险函数线性。

把失败点向更大容量提升能够加强切，但需额外评价，且未必得到斜向替代关系。强切节约的主问题搜索与生成强切所花的 UC/仿真时间需要共同计量。仅报告“切更强”或“排除了多少点”不足以证明总时间改善。

### 4.3 整数运行使经典对偶切不能直接沿用

将 LP 的对偶乘子直接当作原整数风险函数的全局支撑切，通常没有保证。用户利用单调性而非整数问题的普通 LP 对偶来证明排除区域，是合理的处理方法，但 Liu（2024）、Forbes（2024）及更早逻辑分解已有相关方法。

LP 松弛仍然有用：其最小失供是整数最小失供的下界；若下界已超过风险门槛，可以证明失败。相反，一个超时 UC 可行解给的是最小失供的上界；上界超过门槛不能证明真实最优值超过门槛。

### 4.4 容量单调性需要模型条件，不能只凭物理直觉

当前共享潜在模块路径与安装前缀，使较大容量保留较小容量的已有模块；新增柴油机可关闭，风光可弃电，储能可沿用原轨迹。这为可行域嵌入与逐路径单调性提供依据，不要求总经济成本也单调。

若更换容量后重新独立抽样，则一次样本结果不再自动有逐路径大小关系。若加入改变原设备故障分布的共同原因故障、维修资源竞争、强制最小利用、设计改变初始状态等机制，应重新证明单调性。采用一套固定的滚动或启发式控制器时，“设备增多后的实际策略损失”也不自动单调；新增资源改变经济决策和 SOC，可能破坏直接传播规则。最优因果策略在策略集合嵌套时仍可具有单调性，但不能由单个控制器的实现效果替代该证明。

### 4.5 完全预见回答的是特定能力问题

当前每条路径的全年优化预知后续故障和修复，且可选择场景特定的初始 SOC。它可以在未来故障前留电或启机。因果滚动运行则需继承当时实际 SOC、机组开停状态及剩余最小开停机义务，不知道未来故障的实现值。

Joule（2020）的补充材料证明完全预见本身不妨碍高水平发表；Dratsas（2024）与 Misconel（2022）说明这项假设值得单独检验。因此正确的论文定位可以是“给定信息假设下的最优补救容量规划”，但不能未经验证直接声称真实运行政策已经满足同一可靠性指标。

在相同初末条件且完全预见可行集包含因果策略轨迹时，有最优完全预见损失不大于因果策略损失。当前年末 SOC 等于初始 SOC，而某些滚动评价不施加相同终端条件；不能忽略这一差别就直接宣称现有两个结果之间必然满足该下界关系。

### 4.6 样本优化与总体认证的难度不同

固定 SAA 样本风险通过不等于总体风险已获有限样本保证。小 EENS 和尾部 CVaR 约束尤其可能受稀有事故支配。搜索容量、追加样本、反复检查后停止，还需要处理自适应选择造成的统计问题。Jirutitijaroen 与 Singh（2008）已在可靠性扩容中研究 SAA 和统计界；不能仅因加入统计置信度便认定总体算法首创。

新版同时置信区间解决的是另一层保证，可能形成进一步研究内容。但界有效和界足够实用是不同要求。如果损失上界很大、目标风险很小，基于全损失范围的分布无关界可能需要过多样本。它不证明系统不可靠，只表示现有数据与这套置信构造还不能证明可靠。

## 五、数学正确性与算法创新分别评价

对有限固定容量网格和固定样本，以下条件足以支持精确的约束生成解释：

1. 每轮经济主问题全局求解，或保留有效的全局成本下界。
2. 失败只用有效风险下界证明；通过只用有效可行运行或风险上界证明。
3. 所有切均不排除目标模型的可靠容量，每次失败切至少排除当前候选。
4. 未决候选最终能够精化至正确判定；阈值等号和数值容差按明确定义处理。

精确情况下，每轮主问题仍是原规划问题的松弛。若其全局最优候选通过可靠性，则该成本同时是原问题下界和可行解上界，所以就是该固定样本模型的全局最优值。有限容量域保证在这些条件下有限终止；并不保证多项式时间，也不保证任何实际预算内完成。

主问题有 MIP gap 时，应报告可靠方案成本与有效经济下界之间的 gap。固定样本证书不能直接改称总体概率证书；改变样本集后旧经验失败切也需要重新确认。总体认证版有其另外的同时覆盖和终止条件。

| 拟主张的贡献 | 文献核验后的判断 |
|---|---|
| 把投资和故障后运行分成两阶段 | 已有广泛先例，不能单独作为新算法 |
| 经济主问题与可靠性子问题迭代反馈 | da Costa、Bloom、Wei 等已有直接先例 |
| 依据容量单调性排除失败点以下全部容量 | Liu（2024）给出同式，不能声称基础切首创 |
| 利用额外评价扩大排除区域、删去无关设备分量 | Forbes 等有直接相关加强思想；特定提升规则仍需逐项比较 |
| 全年故障轨迹、整数 UC 与双风险约束 | 是更具体、更困难的问题设定；不足以自动成为算法创新 |
| 在可靠性问题上推导新的有效加强界/切并证明效率收益 | 可能形成算法贡献，需区别于已有单调切、LP 加强和相关分解 |
| 可精化整数评价与自适应总体风险认证及全局经济界的联合机制 | 尚未核实完全同构先例，有待专项比较和理论/实验支持；不能仅凭模块组合认定创新 |
| 极地数据和极昼极夜 | 有应用价值，但南极可靠性规划已有论文；单换数据不是算法贡献 |

## 六、当前项目证据能支持到哪里

旧版 [中山站实验报告](/home/yzk/reliable_plan/docs/zhongshan_literature_results.md) 记录 8760 小时、2000 规划样本和 10000 独立验证样本，两轮规划和一条加强切。该切将其他设备提升到上界后，证明在该样本和容量边界下柴油机至少需要 3 台。报告的经济 gap 为 0.909327%。

这支持“该实例可以求解，失败证据能够导出有意义的设备需求”，但两轮一条切的实例不足以证明复杂容量替代边界上的算法可扩展性，也不能证明比已发表的单调逻辑 Benders 更高效。独立验证样本表现和原 SAA 最优性应分别报告。

新版 [2026-09-11 阶段性报告](/home/yzk/reliable_plan/docs/certified_zhongshan_extended_budget_progress_20260911T005252Z.md) 是历史快照，不代表当前后台实时状态。其记录：即使 16,384 条年度样本全部精确零失供，当时 DKW 同时置信构造的 EENS 上界最低仍约 69,562.27 kWh/年，CVaR 上界最低约 1,391,245.48 kWh/年，目标门槛分别为 100 与 1000。该样本上限与置信构造不能签发通过证书；增加单次 UC 时间不能消除这一统计界限制。

因此现有资料尚不足以支持“已经得到计算高效、总体可靠性认证完成的顶刊算法”。这不是二阶段分解无效，也不是系统真实风险被证明超标；它明确指向该版置信界的实用性障碍。

## 七、面向算法论文的最低比较要求

若保留当前研究路线，贡献应对准已经核实的先例，而非把框架重新命名：

- 在相同固定样本、相同 UC、相同初末状态、相同风险和成本口径下，小算例与完整扩展 MILP 对齐最优值；它检验分解正确性。
- 与逐点 no-good、Liu 类基础单调切及 LP 加强、当前提升规则进行同模型比较。采用 Forbes 或 2026 更一般的切时，应说明可迁移部分及新增适用证明。
- 报告多组规模、风险紧度、储能/柴油替代关系下的总时间、UC 调用、主问题时间、未决比例与经济 gap；不能只选一条设备下界即可解决的实例。
- 新版需专门评价有效置信保证达到实际风险门槛的样本数和成本，区分求解误差、统计误差与概率模型误差。
- 用继承状态的滚动或事件触发运行验证规划方案；若保留完全预见做设计，量化其对最终容量和可靠性结论的影响。这个验证不需要把所有算法对照都改成不同问题。

可以使用已有求解框架发表重要的规划研究，贡献也可以来自新的物理机制和规划发现；但如果论文明确以“算法创新”为主线，必须给出超出已有框架、基础切和常规模块组合的可验证增量。当前最稳妥的定位是继续使用这套分解，并把研究重点放到整数可靠性评价、切强度和实用的误差认证上。是否足以发表，取决于这些增量的理论与实验结果，不能由方法名称或应用地域替代。

## 检索范围与限制

本次沿电力扩容、微电网可靠性、仿真优化、随机整数规划、单调优化与逻辑 Benders 等检索线，使用 OpenAlex/Crossref 学术索引、出版信息、作者仓储、arXiv、OSTI、Bilkent、HAL、Caltech 与 Optimization Online，结合关键词和参考文献追溯。关键词包括 reliability-constrained expansion planning、reliability Benders Monte Carlo、microgrid planning reliability checking、stochastic integer recourse、monotonic Benders cuts、simulation optimisation logic-based Benders、perfect foresight storage adequacy 等。

普通网页搜索中不可访问或返回挑战/无关结果的页面未用作证据。以上是跨领域的针对性检索，不能声称穷尽全网或所有非公开全文。直接肯定结论由已核实公式和摘要支持；关于未发现完全相同组合的结论，保持为检索范围内的未确认，不能写成全球首次。
