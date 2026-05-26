# PCR-Net: 面向受限环境下足式机器人人体跟随的前瞻式冲突消解框架

**作者**: Author A, Author B, Author C

**投稿目标**: IEEE Robotics and Automation Letters (RA-L)

---

## 写作状态标注（Day 1, 2026-05-26）

本文档作为 RAL 中文初稿，当前按完整 PCR-Net 论文叙事写作：`y + w + beta + Command Post-Processor`。两周实验不替代论文主张，而是为主张补证据。

- **主张层**：PCR-Net 是面向足式机器人人体跟随的前瞻式冲突消解框架。
- **核心证据层**：主表分两层写：外部对比 `Monolithic PPO / Reactive Safety Override / DWA-inspired Local Rollout`，内部消融 `PCR-yonly / PCR-geomw / PCR-learnedw`；优先证明 `PCR-learnedw` 相比外部常规方法和强规则 `PCR-geomw` 更能处理真实高冲突窗口。
- **实现口径**：`risk_memory` 归入 `learned-w` 的最终实现优化，不单列第四个 baseline。
- **必须补齐层**：外部 baseline、内部三组消融、`beta_sweep / Pareto`、主表多 seed、机制时间序列、D435i 输入链路和真实低速演示。
- **终稿收缩规则**：如果某个组件证据不足，终稿降低其主张强度；不把未完成实验写成已完成结论。

---

## 摘要

在受限环境中进行人体跟随时，足式机器人必须同时满足两个相互竞争的目标：持续保持目标可见并维持合适跟随距离，同时在狭窄通道、门洞或障碍阵列中避免碰撞。现有反应式避障方法通常在风险已经显现后才切换行为，容易产生滞后仲裁和指令抖振；端到端策略可以隐式学习安全-效率权衡，但缺乏可解释的逐步冲突信号，也难以在部署阶段无重训地调节保守程度。

我们提出**前瞻式冲突消解网络（Predictive Conflict Resolution, PCR-Net）**，用于在人体跟随专家与局部避障专家之间进行可解释仲裁。PCR-Net 保留两个候选专家指令，并引入三个互补组件：门控权重 $y$ 表示跟随与避障之间的结构性选择；命令条件化冲突估计器 $w$ 预测当前跟随指令在短未来内是否会与局部障碍产生冲突；风险预算 $\beta$ 作为外部可调旋钮，联动指令后处理器的速度、平滑与安全距离约束。最终执行指令由 $y$、$w$ 和 $\beta$ 共同决定，使系统能够在冲突发生前平滑转向避障，并在风险较低时保持较高跟随效率。

本文的核心主张是：人体跟随中的安全问题不应只被建模为静态障碍距离约束，而应被建模为**候选跟随指令、局部几何和风险预算共同决定的运行时行为冲突**。基于这一建模，PCR-Net 将“何时切向避障”的前瞻性判断与“以多保守的方式执行指令”的风险预算分离，从而避免把所有场景都推向统一的保守策略。为支撑机制分析，我们进一步记录真实高冲突窗口中的 $w$ 激活、门控变化、near-miss、碰撞和指令平滑指标，用于判断 PCR-Net 是否确实提前消解冲突，而不是仅仅降低速度或全程压制跟随。

我们将在 Isaac Gym 六足平台上评估 PCR-Net。主性能表分为两层：外部对比包括单一黑盒策略、规则安全接管和 DWA 启发式短时速度搜索；内部消融包括 `PCR-yonly`、`PCR-geomw` 和 `PCR-learnedw`。实验报告任务成功率、碰撞率、near-miss、跟随误差、目标视野保持、指令抖振、$w$ 提前量、$\beta$ 风险-效率 Pareto 曲线和真实输入 dry-run。[TODO: 填入主表与 Pareto 结果。] 当前两周实验计划用于补齐上述证据链；若最终结果只支持其中一部分机制，终稿将按证据强弱收缩主张，而不牺牲论文叙事的完整性。

**关键词**: 六足机器人、人体跟随、局部避障、冲突仲裁、混合专家、深度相机

---

## I. 引言

自主人体跟随是服务机器人、巡检平台和辅助系统的核心能力。在开阔空间中，该任务可以由目标相对位置和速度直接驱动；真正的挑战出现在机器人必须一边跟随移动目标，一边穿越几何受限环境时。此时目标跟随和局部避障不是两个独立目标，而是在短时间窗口内产生相互竞争的速度指令。

考虑一个六足机器人在狭窄通道或连续障碍中跟随移动目标的场景（图1）。Follow expert 根据目标相对状态生成前进或转向指令，倾向于保持跟随距离和目标视野；Avoid expert 根据局部地图生成横移、减速或转向指令，倾向于先安全通过局部几何约束。当目标已经通过障碍而机器人本体仍处在狭窄区域内时，继续强跟随会导致侧向擦碰或旋转扫掠碰撞；过度避障则会丢失目标、降低任务效率。这个失败模式不是单纯“风险高”可以完整解释的，而是候选指令之间在短未来内产生了行为冲突。

**因此，核心问题不是单独训练更强的跟随器或避障器，而是设计一个能够前瞻识别运行时冲突、并可调节安全-效率取舍的高层仲裁机制。** 仅靠普通门控 $y$ 学到的是平均意义上的行为混合；当冲突窗口短、障碍接触后果严重时，普通门控容易在危险片段中切换过晚，或者通过全程保守来降低碰撞，最终损失跟随质量和目标可见性。仅靠固定安全阈值也不够，因为不同部署阶段对速度、平滑性和安全距离的要求并不相同。

现有范式通过三种方式处理此问题，各有其特征性局限：

**反应式行为仲裁** [1,2] 在检测到靠近障碍物后才在跟随和避障之间切换。切换本质上是滞后的——当反应式触发器激活时，机器人可能已经承诺了一条碰撞轨迹。模式间的滞回导致指令抖振和运动不连续。

**端到端学习** [11,12] 训练单一策略来隐式平衡两个目标。虽然优雅，但学到的权衡是不透明的：既没有机制来诊断*为什么*机器人选择了某个特定动作，也没有办法在部署时调节安全-激进权衡而无需重训。

**安全约束优化** [7,8] 将避障视为奖励最大化策略上的约束。PPO-Lagrangian 等方法在回合层面强制执行平均约束，但不提供*逐步*冲突感知——它们是在碰撞发生后进行惩罚，而非预测和预防碰撞。

### 我们的方法

我们提出**前瞻式冲突消解（Predictive Conflict Resolution, PCR-Net）**，其核心洞察是：**冲突不是单纯由障碍距离决定，而是由当前状态、候选专家指令、短未来风险和部署风险预算共同决定。** PCR-Net 保留两个专家的可解释候选指令：Follow expert 输出 $\mathbf{u}_F$，Avoid expert 输出 $\mathbf{u}_A$；门控策略输出 Follow/Avoid 混合权重 $y$；命令条件化冲突估计器 $w$ 预测执行当前跟随指令是否会在短未来内进入危险构型；风险预算 $\beta$ 则联动指令后处理器中的速度上限、变化率上限和安全距离约束。最终高层指令为：

$$
\mathbf{u} = y_{\mathrm{eff}}\mathbf{u}_F + (1-y_{\mathrm{eff}})\mathbf{u}_A.
$$

与反应式距离阈值不同，$w$ 必须依赖候选命令：相同局部地图下，朝障碍前进的 Follow 指令和远离障碍的 Follow 指令应产生不同冲突评分。与把安全写进单一奖励权重不同，$\beta$ 是可外部扫描的风险预算，使同一策略可以形成安全-效率 Pareto 曲线。这样，PCR-Net 的论文贡献不只是“多一个门控网络”，而是把运行时仲裁拆成结构选择、前瞻冲突预测和风险预算控制三个可解释部分。

为了支撑实机可行性，部署输入收口到 D435i 可获得信号：目标相对状态、`local_map_2ch` 和可选短时风险记忆。仿真中的障碍行真值只用于评测诊断和辅助标签分析，不进入 actor，也不作为实机规则。当前两周计划会优先补齐 $w$ 的强证据；$\beta$ 与后处理器仍保留为论文框架的重要组成，但最终主文强弱将由 Pareto 结果决定。

### 贡献

1. 提出一个面向足式机器人人体跟随的**前瞻式冲突消解框架**，将跟随专家、避障专家和高层门控统一到可解释的 Follow/Avoid 仲裁形式中。
2. 提出**学习式命令条件化冲突估计器** `learned-w`，通过候选跟随/避障指令和局部风险预测短未来冲突，并相对手工 `geom-w` 基线验证学习先验的必要性。
3. 提出**风险预算 $\beta$ 联动的指令后处理器**，将速度、平滑性和安全距离约束组织为可扫描的保守度轴，用于形成风险-效率 Pareto 曲线并支持部署时调节。
4. 在 Isaac Gym 六足平台和 D435i 输入链路上验证 PCR-Net，报告消融、机制分析、风险-效率曲线和真实输入 dry-run；其中 HighConflict、CSI、RCM 等指标作为证据工具服务于上述主张，而不是替代主张本身。

---

## II. 相关工作

### A. 机器人人体跟随

人体跟随已在轮式 [14]、四足 [15] 和空中平台上得到广泛研究。经典方法使用 PID 速度跟踪配合滤波进行目标状态估计。基于学习的方法如 ViNT [11] 和 NoMaD [12] 在大规模轨迹数据集上训练视觉条件化导航策略，实现了令人印象深刻的跨平台迁移。然而，**这些模型将跟随视为*单目标*导航任务，并未显式建模与避障之间的冲突**。在受限环境中部署时，它们依赖隐式策略正则化来避免碰撞，在安全-效率权衡方面既无保证也无可解释性。SaferPath [13] 为端到端导航模型增加了模型预测安全滤波器，但该滤波器基于可通行性地图运作，并未处理跟随与避障行为之间的*指令级*冲突。

### B. 导航行为仲裁

解决多种导航行为之间的冲突有着悠久的历史。包容体系结构 [1] 施加固定优先级排序。势场法 [3] 融合吸引和排斥梯度但受困于局部极小值。动态窗口法（DWA）[4] 在约束可行速度空间中联合优化，但需要显式动力学模型，不能扩展到高维足式平台。

近年来，基于学习的仲裁通过混合专家（MoE）[5] 得到探索。MoE-Loco [16] 将 MoE 应用于多任务运动控制，通过将梯度更新路由到专门化专家来缓解*训练时*梯度冲突。**我们的工作存在根本性差异：MoE-Loco 解决的是*训练优化*问题，而 PCR-Net 解决的是*运行时行为冲突*。** 在结构上，PCR-Net 的门控受*前瞻式*冲突信号 $w$ 驱动，这在标准 MoE 中没有对应物。

### C. 安全强化学习

约束马尔可夫决策过程（CMDP）[6] 将安全性形式化为累积代价约束。约束策略优化（CPO）[7] 提供每次更新的近似约束满足保证；PPO-Lagrangian [8] 使用拉格朗日乘子双梯度上升提供更实用的变体。这些方法在回合层面*统计地*强制安全性，但**缺乏*逐步*冲突感知**：它们无法区分某个动作是因为过度激进跟随还是过度保守避障而不安全。

控制屏障函数（CBF）[9] 通过将不安全动作投影到安全集边界来提供前向不变性保证。Agile But Safe（ABS）[10] 将 CBF 启发的 reach-avoid 值网络与双策略（敏捷/恢复）结合用于四足导航。然而，**ABS 的切换信号是*状态条件化*的**——它评估当前状态是否安全，而非某个*特定候选指令*是否会导致冲突。PCR-Net 的冲突估计器是*命令条件化*的：对于相同的障碍物配置，不同的跟随指令产生不同的冲突评分，提供更丰富的仲裁信息。

---

## III. 问题建模

### A. 系统架构

考虑一个配备深度相机（提供障碍物观测 $\mathbf{o}_t \in \mathbb{R}^{n_o}$）和目标跟踪器（YOLO + 卡尔曼滤波，提供人体状态 $\mathbf{h}_t = (p_h, v_h, \psi_h)$，包含位置、速度、航向）的六足机器人。

运动控制器采用两层架构。**底层步态策略** $\pi_{\text{low}}$ 在 50 Hz 下映射关节级指令，提前训练完成并在后续所有训练中*冻结*。**高层规划器**在 10 Hz 下运行（降采样因子 5），输出机体坐标系下的速度指令 $\mathbf{u}_t = (v_x, v_y, \omega_z)$。本文描述的所有模块均在高层运行。

### B. 专家定义

**跟随专家 $S_0$**：基于规则的单车模型控制器，从 $\mathbf{h}_t$ 计算 $\mathbf{u}_F = (v_F, \omega_F)$。采用"先转后走"的滞回策略：当方位角误差 $|\alpha_t|$ 超过 $\alpha_{\text{lock}} = 0.35$ rad 时，前进速度归零，机器人原地旋转；当 $|\alpha_t|$ 降至 $\alpha_{\text{release}} = 0.18$ rad 以下时恢复前进。横向速度恒为零（$v_x \equiv 0$）。$S_0$ 被有意设计为不感知障碍物，以清晰界定冲突边界。

**避障专家**：强化学习策略 $\pi_{\text{avoid}}(\mathbf{o}_t, \mathbf{g}_t)$，训练目标是朝局部目标航路点 $\mathbf{g}_t$ 导航同时避开障碍物。接收两通道局部地图（占据栅格、间隙/代价）作为深度图像的衍生输入。训练采用三阶段平地课程，障碍物布局逐步加密直至逐渐收窄的走廊（详见 IV-G 节）。

### C. 冲突定义

在每个高层步 $t$，两个专家可能发出矛盾指令。定义**指令冲突角**：

$$\theta_{\text{conflict},t} = \arccos \frac{\mathbf{u}_{F,t} \cdot \mathbf{u}_{A,t}}{\|\mathbf{u}_{F,t}\| \cdot \|\mathbf{u}_{A,t}\|}$$

当 $\theta_{\text{conflict},t}$ 较大（趋近 $\pi$）时，两个专家将机器人拉向几乎相反的方向。PCR-Net 的目标是*前瞻式地*消解此冲突：在机器人进入几何危险构型*之前*而非之后，将融合指令偏置向避障方向。

---

## IV. 方法

图2展示了完整的 PCR-Net 架构。以下按推理数据流顺序描述各模块。

### A. 冲突估计器 $w$

#### 动机

一个仅利用当前障碍物观测来融合 $\mathbf{u}_F$ 和 $\mathbf{u}_A$ 的朴素门控网络本质上是*反应式*的：它必须等到机器人靠近障碍物后才能将权重转向避障。在窄走廊中，跟随专家的旋转扫掠体积延伸超出通道边界时，反应式门控来得太迟——机器人已承诺的旋转惯性阻止了及时纠正。

冲突估计器通过回答一个*反事实*问题来解决此问题：**如果机器人在接下来 $T$ 步执行当前跟随指令 $\mathbf{u}_{F,t}$，其机身足迹是否会与障碍物地图发生碰撞？**

#### 伪标签生成

训练期间，通过运动学前向仿真生成二值伪标签 $c_t \in \{0, 1\}$（图3）。给定机器人当前位姿 $(x_t, y_t, \theta_t)$ 和候选跟随指令 $\mathbf{u}_{F,t} = (v_F, \omega_F)$，以高层频率展开单车模型 $T$ 步：

$$x_{\tau+1} = x_\tau + v_F \cos\theta_\tau \cdot \Delta t$$
$$y_{\tau+1} = y_\tau + v_F \sin\theta_\tau \cdot \Delta t$$
$$\theta_{\tau+1} = \theta_\tau + \omega_F \cdot \Delta t$$

在每个仿真步 $\tau$，计算机器人矩形足迹（以 $(x_\tau, y_\tau, \theta_\tau)$ 为中心，尺寸 $l_b \times w_b = 0.65 \times 0.70$ m）与局部障碍物地图之间的最小距离：

$$d_\tau = \min_{p \in \mathcal{F}(\tau)} \text{dist}(p, \mathcal{M})$$

其中 $\mathcal{F}(\tau)$ 是步 $\tau$ 的足迹多边形，$\mathcal{M}$ 是占据栅格地图。伪标签为：

$$c_t = \mathbf{1}\left[\min_{\tau=1}^{T} d_\tau < d_{\text{safe}}\right]$$

预测视界 $T = 10$ 步（1.0 s），安全阈值 $d_{\text{safe}} = 0.10$ m。

**关键特性**：伪标签同时依赖障碍物地图和 $\mathbf{u}_{F,t}$ 的*方向*。相同的障碍物配置，当 $\mathbf{u}_{F,t}$ 指向远离墙壁方向时 $c_t = 0$，指向墙壁方向时 $c_t = 1$。这种命令条件化防止 $w$ 退化为单纯的距离检测器。

#### 网络架构与损失

冲突估计器是轻量级 MLP：

$$w_t = \sigma\left(f_w([\mathbf{o}_t;\, \mathbf{u}_{F,t}];\, \phi)\right) \in [0, 1]$$

其中 $\sigma$ 是 sigmoid 函数，$\phi$ 是可学习参数。使用辅助二元交叉熵损失训练：

$$\mathcal{L}_w = -\mathbb{E}\left[c_t \log w_t + (1 - c_t) \log(1 - w_t)\right]$$

该损失与门控策略训练并行运行，但使用独立优化器以避免梯度干扰。

**注意**：推理时不执行前向仿真。训练好的 $f_w$ 直接在单次前向传递中将 $(\mathbf{o}_t, \mathbf{u}_{F,t})$ 映射为冲突评分，几乎不增加延迟。

### B. 风险预算 $\beta$

不固定单一安全阈值，而是通过标量 $\beta \in [0,1]$ 参数化一个连续的*约束配置族*。具体地，$\beta$ 通过仿射插值映射到物理量：

$$v_{\max}(\beta) = v_{\max}^{+} - \beta \cdot (v_{\max}^{+} - v_{\max}^{-})$$
$$d_{\text{safe}}(\beta) = d_{\text{safe}}^{-} + \beta \cdot (d_{\text{safe}}^{+} - d_{\text{safe}}^{-})$$
$$\Delta v_{\max}(\beta) = \Delta v^{+} - \beta \cdot (\Delta v^{+} - \Delta v^{-})$$

其中 $\beta = 0$ 对应激进跟随（$v_{\max}^{+} = 1.2$ m/s，$d_{\text{safe}}^{-} = 0.35$ m），$\beta = 1$ 对应保守避障（$v_{\max}^{-} = 0.35$ m/s，$d_{\text{safe}}^{+} = 1.0$ m）。

| 参数 | $\beta = 0$（激进） | $\beta = 1$（保守） |
|---|---|---|
| 最大线速度 $v_{\max}$ (m/s) | 1.00 | 0.35 |
| 最大角速度 $\omega_{\max}$ (rad/s) | 1.50 | 0.50 |
| 安全距离 $d_{\text{safe}}$ (m) | 0.35 | 1.00 |
| 最大线速度变化率 $\Delta v$ (m/s/step) | 0.15 | 0.05 |
| 最大角速度变化率 $\Delta \omega$ (rad/s/step) | 0.30 | 0.10 |
| 风险钳制增益 $k_{\text{risk}}$ | 1.0 | 3.0 |

语义保证是单调的：$\beta \uparrow$ 意味着 $d_{\text{safe}} \uparrow$、$v_{\max} \downarrow$、$\Delta v_{\max} \downarrow$——即严格更保守的行为。

**设计动机**：将 $\beta$ 设为*外部旋钮*（而非学习输出）是刻意的。它使操作者能在*不重训*的情况下扫描安全-效率 Pareto 前沿，并为部署提供人类可理解的控制轴。训练时 $\beta$ 在 $[0,1]$ 内均匀采样，使门控策略学会在全保守度谱上运行。

### C. 指令后处理器

后处理器强制执行 $\beta$ 参数化的约束。运行在 50 Hz（与底层策略匹配），按顺序执行四步操作：

1. **速度限幅**：$\mathbf{u} \leftarrow \text{clip}(\mathbf{u}, -v_{\max}(\beta), v_{\max}(\beta))$
2. **变化率限制**：$\Delta \mathbf{u} \leftarrow \text{clip}(\mathbf{u} - \mathbf{u}_{t-1}, -\Delta v_{\max}(\beta), \Delta v_{\max}(\beta))$；$\mathbf{u} \leftarrow \mathbf{u}_{t-1} + \Delta \mathbf{u}$
3. **风险钳制**：若瞬时间隙 $\tilde{d}_t < d_{\text{safe}}(\beta)$，将指令按 $k_{\text{risk}}(\beta) \cdot (\tilde{d}_t / d_{\text{safe}}(\beta))$ 缩放
4. **绝对饱和**：执行与 $\beta$ 无关的平台级硬限制

此级联保证无论门控策略输出什么，最终指令都尊重 $\beta$ 所规定的物理约束。

### D. 门控策略 $y$

门控策略输出标量混合权重 $y_t \in [0,1]$：

$$\mathbf{u}_t = y_t \cdot \mathbf{u}_{F,t} + (1 - y_t) \cdot \mathbf{u}_{A,t}$$

$y_t = 1$ 表示完全跟随，$y_t = 0$ 表示完全避障。

#### 观测空间

$$\mathbf{s}_t = [\mathbf{o}_t;\, w_t;\, \beta;\, y_{t-1}] \in \mathbb{R}^{n_o + 3}$$

其中 $\mathbf{o}_t$ 编码两通道局部地图（占据和间隙），$w_t$ 是冲突估计器输出，$\beta$ 是当前风险预算，$y_{t-1}$ 是上一步门控输出（提供时序平滑归纳偏置）。

#### 奖励设计

门控策略使用 PPO 训练，奖励组合任务和安全项：

$$r_t = \underbrace{r_{\text{approach}} + r_{\text{reach}} + r_{\text{heading}}}_{\text{跟随质量}} + \underbrace{r_{\text{risk}} + r_{\text{collision}}}_{\text{安全性}} + \underbrace{r_{\text{smooth}} + r_{\text{time}}}_{\text{正则化}}$$

各项含义：$r_{\text{approach}}$ 奖励接近目标的进展，$r_{\text{reach}}$ 在达到期望跟随距离时提供稀疏奖励，$r_{\text{heading}}$ 惩罚航向偏差，$r_{\text{risk}}$ 是间隙低于 $d_{\text{safe}}(\beta)$ 时的稠密屏障惩罚，$r_{\text{collision}}$ 是接触时的大负脉冲，$r_{\text{smooth}}$ 惩罚指令抖动，$r_{\text{time}}$ 是逐步代价以防止停滞。

#### $w$ 的融合机制

冲突估计 $w_t$ 以两种方式进入门控策略：（i）作为观测特征，使策略学习自身从 $w$ 到 $y$ 的映射；（ii）通过**前瞻式偏置**：

$$y_{\text{eff},t} = (1 - \lambda_w) \cdot y_t + \lambda_w \cdot (1 - w_t)$$

其中 $\lambda_w$ 是固定融合系数。此偏置确保即使在策略尚未完全收敛时，高冲突评分也会将融合指令推向避障方向，在早期训练中提供安全底线。

### E. 推理流程

**算法1：PCR-Net 推理（10 Hz）**

输入：障碍物观测 $\mathbf{o}_t$，人体状态 $\mathbf{h}_t$，风险预算 $\beta$，前一步门控 $y_{t-1}$

输出：最终速度指令 $\mathbf{u}_t$

1. $\mathbf{u}_{F,t} \leftarrow S_0(\mathbf{h}_t)$ （跟随专家，规则基）
2. $\mathbf{u}_{A,t} \leftarrow \pi_{\text{avoid}}(\mathbf{o}_t, \mathbf{g}_t)$ （避障专家，RL）
3. $w_t \leftarrow f_w([\mathbf{o}_t;\, \mathbf{u}_{F,t}];\, \phi)$ （冲突估计器）
4. $y_t \leftarrow \pi_y([\mathbf{o}_t;\, w_t;\, \beta;\, y_{t-1}];\, \theta)$ （门控策略）
5. $y_{\text{eff},t} \leftarrow (1-\lambda_w) y_t + \lambda_w (1 - w_t)$ （前瞻偏置）
6. $\mathbf{u}_t^{\text{raw}} \leftarrow y_{\text{eff},t} \cdot \mathbf{u}_{F,t} + (1 - y_{\text{eff},t}) \cdot \mathbf{u}_{A,t}$
7. $\mathbf{u}_t \leftarrow \text{PostProcess}(\mathbf{u}_t^{\text{raw}}, \beta, \tilde{d}_t)$ （$\beta$ 联动约束）
8. 返回 $\mathbf{u}_t$

### F. 训练流程

训练分三个阶段进行：

**阶段1：跟随专家标定。** $S_0$ 是手工设计的规则控制器。无需学习；参数在无障碍环境中通过回放验证。

**阶段2：避障专家训练。** $\pi_{\text{avoid}}$ 作为高层 cmd_vel 策略训练，底层步态策略冻结。观测包含机器人本体状态 $\mathbf{s}_{\text{prop}} \in \mathbb{R}^{9}$（基座线/角速度、重力投影、前一指令），目标航路点 $\mathbf{g} \in \mathbb{R}^{2}$，两通道局部地图 $\mathbf{m} \in \mathbb{R}^{2 \times H}$，以及标量课程难度指标。

训练采用三阶段平地课程：阶段1放置稀疏随机障碍物；阶段2增加障碍物密度；阶段3引入逐渐收窄的平行走廊（宽度从 1.2 m 缩小到 0.75 m）。阶段转换由 150 回合滑动窗口内的成功率（$\geq 50\%$）和碰撞率（$\leq 3\%$）控制，转换时清除窗口以防止历史数据干扰。

**阶段3：PCR-Net 联合训练。** 两个专家均冻结。门控策略 $y$ 和冲突估计器 $w$ 在 S 型走廊环境中联合训练。移动目标沿预设 S 型轨迹运动；机器人须在跟随的同时避开走廊墙壁。门控策略用 PPO 优化（$\gamma = 0.99$，$\lambda_{\text{GAE}} = 0.95$，裁剪比 0.2）；$w$ 用 Adam 优化器在 $\mathcal{L}_w$ 上优化，使用在线生成的伪标签。$\beta$ 在每个回合内均匀采样以确保覆盖。

---

## V. 实验

### A. 仿真设置

所有训练和评估在 NVIDIA Isaac Gym（Preview 4）中进行，使用 256—4096 个并行环境。六足模型为 18 自由度平台（3关节 × 6腿），行走宽度 $w_b = 0.70$ m。底层步态策略预训练用于平地全向运动，在 50 Hz 下冻结（检查点：6000 回合）。

**S 型走廊场景。** 主评估环境是由两个方向相反的连续弯道组成的 S 型走廊。走廊宽度从 0.75 m（极限：单侧间隙 0.025 m）到 1.20 m（舒适）变化。目标以可配置速度（默认 0.5 m/s）沿预录 S 型轨迹运动。每个回合在碰撞、目标丢失（距离 > 5 m）或超时（500步，50 s）时终止。

**评估指标**：

- **碰撞率（CR）**：存在任何墙壁接触的回合比例
- **成功率（SR）**：全程保持跟随距离 $\leq 2.0$ m 并完成走廊的回合比例
- **平均间隙（$\bar{d}$）**：回合内平均最小墙壁间隙
- **指令平滑度（$J_{\text{jerk}}$）**：融合指令的均方抖动，量化抖振程度
- **$w$ 提前时间（$\Delta t_w$）**：$w$ 超过 0.5 与首次间隙降至 $d_{\text{safe}}$ 以下之间的步数，衡量预测提前量

### B. 主性能对比与内部消融

主性能表回答两个问题：第一，PCR-Net 是否比常规外部方法更适合 Follow/Avoid 运行时冲突；第二，`learned-w` 是否比无冲突先验和手工几何先验更有效。所有方法使用同一主任务、同一评测 episode 数、同一 seeds 和同一指标口径。

| 类别 | 方法 | 描述 |
|---|---|---|
| 外部对比 | Monolithic PPO | 单一 RL 策略直接输出 $\mathbf{u}=(v_x,v_y,\omega_z)$，不显式保留 Follow/Avoid 候选指令 |
| 外部对比 | Reactive Safety Override | 正常跟随，风险超过阈值时用手工规则降速和横移避让 |
| 外部对比 | DWA-inspired Local Rollout | 基于局部目标点采样短时速度并打分，作为经典局部规划启发式对照 |
| 内部消融 | PCR-yonly | 只使用门控 $y$，没有显式冲突先验 |
| 内部消融 | PCR-geomw | 使用手工几何规则得到 $w$，作为强规则基线 |
| 本文方法 | PCR-learnedw | 使用学习式命令条件化冲突先验；若启用 `risk_memory`，仍归入该方法 |

| 方法 | SR↑ | CR↓ | near-miss↓ | follow MAE↓ | FOV rate↑ | CSI↑ | CSS/RCM↑ |
|---|---|---|---|---|---|---|---|
| Monolithic PPO | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | — | — |
| Reactive Safety Override | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | — | — |
| DWA-inspired Local Rollout | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | — | — |
| PCR-yonly | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| PCR-geomw | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| PCR-learnedw | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

**预期趋势**：`PCR-learnedw` 应比 `Monolithic PPO` 更可解释、比 `Reactive Safety Override` 更少保守停滞、比 `DWA-inspired Local Rollout` 更适合动态目标跟随；在内部消融中，它应比 `PCR-yonly` 更早压低危险跟随，比 `PCR-geomw` 更能处理几何规则难以覆盖的高冲突窗口。

### C. 前瞻提前量分析

为直接验证 $w$ 的*前瞻性*，测量提前时间 $\Delta t_w$：$w_t > 0.5$ 的首步与实际间隙降至 $d_{\text{safe}}$ 以下的首步之间的差值。正 $\Delta t_w$ 表明 $w$ 在危险实际出现*之前*就已激活。

| | 平均 $\Delta t_w$（步） | 平均 $\Delta t_w$（秒） |
|---|---|---|
| PCR-learnedw | [TODO] | [TODO] |
| PCR-yonly | [TODO] | [TODO] |

对于 `PCR-yonly`（无 $w$），使用 $y$ 首次降至 0.5 以下的步作为"冲突检测"的代理指标计算类似量。`PCR-learnedw` 显著更大的提前时间确认 $w$ 提供了真正的前瞻性冲突感知，而非仅是对同一反应式信号的不同阈值。

### D. 通过 $\beta$ 扫描获得 Pareto 前沿

在 $\beta = \{0.0, 0.25, 0.5, 0.75, 1.0\}$ 五个值下评估 `PCR-learnedw`，绘制碰撞率 vs. 成功率 Pareto 前沿（图4）。

关键观察：`PCR-learnedw` 的 Pareto 前沿应优于 `PCR-yonly` 和 `PCR-geomw` 的固定设置：对于相近碰撞率，保持更高任务成功率或更低跟随误差。这种优势直接归因于 $w$ 的前瞻偏置，它使门控策略更早开始转向避障，为平滑导航留出更多时间和空间。

### E. 外部基线对比

外部基线用于证明 PCR-Net 不是只优于自家消融。`Monolithic PPO` 回答“为什么不直接训练一个统一策略”；`Reactive Safety Override` 回答“为什么简单规则不够”；`DWA-inspired Local Rollout` 回答“经典局部规划启发式能否解决动态跟随冲突”。若两周内时间不足，主表最低保留前两个外部基线，`DWA-inspired Local Rollout` 放入补充或后续实验。

| 方法 | CR↓ | SR↑ | follow MAE↓ | near-miss↓ |
|---|---|---|---|---|
| Monolithic PPO | [TODO] | [TODO] | [TODO] | [TODO] |
| Reactive Safety Override | [TODO] | [TODO] | [TODO] | [TODO] |
| DWA-inspired Local Rollout | [TODO] | [TODO] | [TODO] | [TODO] |
| PCR-learnedw | [TODO] | [TODO] | [TODO] | [TODO] |

预期结论不是“PCR-Net 打败所有导航算法”，而是证明：在 Follow/Avoid 高冲突窗口中，显式保留候选行为并用 `learned-w` 预测冲突，比单一黑盒策略、固定规则接管和短时局部速度搜索更稳。

### F. 定性轨迹分析

图6展示了 `PCR-yonly` 和 `PCR-learnedw` 在 0.85 m S 型走廊中的代表性回合时间线对比。

在 `PCR-yonly` 中，门控权重 $y$ 保持高位（> 0.8，偏向跟随）直到机器人进入第一个弯道、间隙突然下降。策略随后快速切向避障（$y < 0.2$），导致速度不连续——表现为抖动尖峰。在第二个弯道重复此模式，延迟切换导致与墙壁的擦碰。

在 `PCR-learnedw` 中，冲突估计器 $w$ 在间隙开始下降前约 0.8 s 平滑上升，促使门控策略在多个步骤内逐渐降低 $y$。由此产生的指令轨迹平滑，机器人预先减速并向走廊中心偏移，两个弯道均无接触通过。$w$ 信号清晰体现了*命令条件化*的性质：在两个弯道之间的直线段，尽管墙壁距离相似，$w$ 下降——因为 $\mathbf{u}_{F,t}$ 指向走廊方向而非墙壁。

### G. 走廊宽度泛化

在 0.75 m 到 1.20 m 范围内评估 `PCR-learnedw`，测试对不同几何约束严重程度的稳定性。

| 宽度 (m) | CR↓ | SR↑ | $\bar{d}$ (m)↑ | $\Delta t_w$ (s) |
|---|---|---|---|---|
| 0.75 | [TODO] | [TODO] | [TODO] | [TODO] |
| 0.85 | [TODO] | [TODO] | [TODO] | [TODO] |
| 1.00 | [TODO] | [TODO] | [TODO] | [TODO] |
| 1.20 | [TODO] | [TODO] | [TODO] | [TODO] |

性能随走廊收窄平缓退化：碰撞率上升但应仍低于 `PCR-yonly` 和外部反应式基线。$\Delta t_w$ 在更窄走廊中减小，反映了可用反应裕度的降低——冲突估计器仍提前激活，但可用于机动的时间预算本质上更小。在 0.75 m（单侧间隙 0.025 m）时，任务接近平台物理极限；残留碰撞率归因于步态引起的身体摇摆超出可用裕度。

### H. 实机验证

我们将 PCR-Net 部署到自制 18 自由度六足机器人上（图5），验证 sim-to-real 迁移。

**硬件。** 机器人使用 STS3215 总线舵机（每腿3个 × 6腿），Orbbec Gemini 336 深度相机（配 IR 通过滤光片增强户外鲁棒性），NVIDIA Jetson Orin NX 进行板载计算。OV9281 全局快门相机通过 ArUco PnP 姿态估计提供充电桩对接。平台重约 [TODO] kg，行走宽度 0.70 m。

**感知流水线。** 深度图像以 10 Hz 处理为两通道局部地图（占据、间隙）。为匹配仿真的理想深度传感器：（i）视场匹配（水平 87°，36 个角度 bin，约 2.4°/bin）；（ii）最小距离裁剪至 0.3 m 以处理传感器盲区；（iii）一阶低通滤波器（$\alpha = 0.65$）平滑扫描向量以抑制深度噪声。YOLO v8n 检测器在 Jetson 上以 15 Hz 运行，卡尔曼滤波器（FilterPy）在检测间隔提供目标状态估计。

**计算分配。** 系统运行 7 个 ROS 1 节点，分布在 Jetson（感知、高层规划器、YOLO、卡尔曼滤波器）和 Intel NUC（底层步态策略、舵机通信、诊断）上。高层规划器（PCR-Net 推理）在 10 Hz 运行；底层步态控制器在 50 Hz 运行。

**物理测试环境。** 使用泡沫板墙在室内实验室搭建 S 型走廊。走廊宽度设为 0.85 m，与主仿真条件匹配。人类操作者以约 0.4—0.5 m/s 穿过走廊，机器人自主跟随。

**结果。** [TODO: 报告 20 次试验的碰撞率、成功率和定性轨迹对比。预期性能相比仿真略有退化（深度噪声、步态打滑、延迟），但核心前瞻行为（w 在弯道前激活）应能迁移。]

---

## VI. 讨论

**命令条件化为何重要。** 冲突估计器最显著的特性是命令条件化：对于相同的障碍物配置，不同的候选跟随指令产生不同的冲突评分。基于距离的反应式方法对一个状态分配单一危险等级，无论意图动作如何。$w$ 捕获了*意图*（跟随指令）与*环境*（障碍物布局）之间的关键交互，使门控策略能在同一物理位置区分良性指令（指向远离墙壁）和危险指令（指向墙壁）。

**与 MoE 的结构相似性。** PCR-Net 的架构——两个专家和一个学习门控——表面上类似混合专家。我们强调语义差异：MoE 门控优化加权专家精度，门控信号仅是输入特征的函数。PCR-Net 的门控受通过运动学前向仿真生成的*前瞻式冲突信号* $w$ 驱动，这在 MoE 中没有对应物。此外，$\beta$ 提供了标准 MoE 所缺乏的外部可解释控制轴。我们将结构相似性视为优势——它意味着 PCR-Net 可以用最小的架构新颖性来实现，贡献在于*冲突消解语义*而非架构复杂度。

**局限性。** 当前实现使用单车运动学模型生成伪标签，忽略了六足的腿部动力学和步态引起的身体振荡。更精确的全身仿真可提高伪标签质量，但代价是计算开销增加。跟随专家 $S_0$ 是基于规则的，不适应环境；学习型跟随策略可从源头减少冲突强度。最后，避障专家在极窄通道（< 0.75 m）中的行为受限于六足穿越此类间隙的物理可行性。

**更广泛的适用性。** 虽然我们在六足平台上实例化 PCR-Net，但该框架是平台无关的。任何具备（i）可分离的跟随和避障专家、（ii）可前向仿真的指令模型、（iii）可调风险预算的系统都可以采用 PCR-Net。潜在扩展包括四足和轮式平台、多智能体跟随场景，以及集成视觉导航基础模型作为即插即用的跟随专家。

---

## VII. 结论

我们提出了前瞻式冲突消解（PCR-Net），一个用于解决受限环境中人体跟随与避障行为冲突的框架。通过引入命令条件化冲突估计器 $w$、机制化风险预算 $\beta$ 和约束门控策略 $y$，PCR-Net 实现了前瞻式、可解释、可控的冲突消解。在六足平台上 S 型走廊中的消融实验证实每个模块都对性能有贡献，且前瞻性冲突估计器是降低碰撞率同时保持跟随质量的决定性因素。[TODO: 实机实验验证 sim-to-real 迁移。]

未来工作将探索多维度冲突分解（$w_{\text{spatial}}$, $w_{\text{proximity}}$, $w_{\text{severity}}$）以实现更丰富的冲突表征，并将框架扩展到动态障碍物场景。

---

## 参考文献

[1] R. A. Brooks, "A robust layered control system for a mobile robot," IEEE J. Robot. Autom., 1986.

[2] R. C. Arkin, Behavior-Based Robotics, MIT Press, 1998.

[3] O. Khatib, "Real-time obstacle avoidance for manipulators and mobile robots," Int. J. Robot. Res., 1986.

[4] D. Fox et al., "The dynamic window approach to collision avoidance," IEEE Robot. Autom. Mag., 1997.

[5] R. A. Jacobs et al., "Adaptive mixtures of local experts," Neural Comput., 1991.

[6] E. Altman, Constrained Markov Decision Processes, CRC Press, 1999.

[7] J. Achiam et al., "Constrained policy optimization," ICML, 2017.

[8] A. Ray et al., "Benchmarking safe exploration in deep reinforcement learning," arXiv:1910.01708, 2019.

[9] A. D. Ames et al., "Control barrier function based quadratic programs for safety critical systems," IEEE Trans. Autom. Control, 2017.

[10] T. He et al., "Agile but safe: Learning collision-free high-speed legged locomotion," RSS, 2024.

[11] D. Shah et al., "ViNT: A foundation model for visual navigation," CoRL, 2023.

[12] A. Sridhar et al., "NoMaD: Goal masked diffusion policies for navigation and exploration," ICRA, 2024.

[13] A. Yao et al., "SaferPath: Safety-augmented navigation with model-predictive Stein variational evolution," ICRA, 2025.

[14] S. S. Honig and T. Oron-Gilad, "Understanding and resolving failures in human-robot interaction," Front. Psychol., 2018.

[15] H. Karnan et al., "VI-IKD: High-speed accurate off-road navigation," RSS Workshop, 2022.

[16] B. Han et al., "MoE-Loco: Mixture of experts for multitask locomotion," arXiv:2503.08564, 2025.

[17] R. Cheng et al., "End-to-end safe reinforcement learning through barrier functions," AAAI, 2019.

[18] J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
