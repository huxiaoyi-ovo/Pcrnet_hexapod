# RA-L 审稿回复总账（内部工作稿，非提交版）

> 最后更新：2026-09-08
> 范围：AE、R1、R2、R10 的全部可执行意见，以及本轮讨论中出现的证据、回应方向和实验决策。本轮未上传、发送或推送这些材料。
> **证据边界**：`已讨论`不等于`已验证`，`候选实验`不等于`已批准执行`，`暂缓`不等于`已解决`。没有完成并复核的实验，不能在正式回复或修订稿中写成结果。

## 2026-09-08 输入依赖与历史硬件证据补核（诊断工具已自测，非实验结果）

- 已完成诊断工具自测：`tools/audit_actor_xy.py` 将在冻结旧 checkpoint 的真实 eval rollout 中旁路采集 Avoid/Gate actor 输入，并只在 no-grad 副本中替换 state 的 `x/y`；live action、map、goal、difficulty、risk/memory 与环境推进不改。已有 critic state 保持不变，未传时副本显式固定为原 actor state。输出独立 npz/json，记录脚本/checkpoint SHA 与 eval-source Git SHA；非 self-test 保证 Isaac Gym 先于 Torch。当前只通过本地语法/Torch self-test，尚未加载 checkpoint 或产生 rollout 结果。
- 历史硬件定量证据仍不足：现有 Fig. 7 可追溯到一个 bag 的机制曲线，而非 40 次逐 trial 成功率证据；原始 bag、逐 trial 人工标注和实际启动命令尚未找到。因此正式回复不得写成已审计的“40 trials”事实，除非补回原始记录。
- 当前 `pcr_realplay.py` 的 ROS 模板没有传 `state_topic`；缺 state 时补足维度的零 state，`risk_memory_velocity_source=body` 将读 state[4]=0，memory 不随前进衰减。file bridge 同样无条件构造零 state。它不是自动回退 `cmd_F`，也不能反推历史实机使用了哪种启动参数；需在修订稿中把部署记录与历史结果分开。
- 历史 Table I `.60 m/s` Learned-w 的三 seed collision 计数为 `3/3/9`，合计 `15/384=0.0390625`。历史代码线索显示 strict collision 组合 hull-clearance（margin `.01`）和 `s_avoid_episode_collision`，但 metrics 不含准确源码 SHA；回复中不能把旧数值拆为 contact/envelope，也不能以当前实现反推历史逐 episode 原因。
- 需在方法表中显式区分 Avoid 的 goal：仿真独立训练和 PCR 复用使用 cross-line `goal_raw`，而实机 `pcr_realplay.py:991-997` 将 target-relative goal 输入 Avoid。该差异目前是待冻结的输入 contract，不能声称三处完全同一语义。Gate 开训须等待已冻结的训练输入、奖励/终止、核心闭环与修正服务器 Isaac smoke；Avoid、Mono、Gate 依赖分开放行，不能以本审计改变算法。

## 2026-09-08 当前确定性修正状态（非实验结果）

- 已修正 Follow body→world 逆变换与 scene raster 完全越界 bbox 的边界压缩；本地 CPU 源码回归通过，旧源码分别在 Follow 往返与 front-outside raster fixture 失败。代码提交 `781e830` 与记录 `eefa8cf` 已推送 GitHub；通过一次性 SSH 反向代理成功拉取至服务器 detached root `/home/dell/RL_hexapod_gym_revision_geometry_20260908`，批准文件 hash 与本地一致，未作永久网络配置改动。因此 Isaac 闭环、服务器原生 CPU、重训、重评测和实机验证仍均未执行。
- 服务器旧 Avoid 运行及其产物保留，但不作为正式返修 checkpoint；本批未修改或冻结 sim-real policy contract。

## 2026-09-08 最新执行状态（Avoid 首轮已过；非结果）

- 作者批准最小独立 Avoid 开训：独立训练的第 14 维保留 forced-forward speed 仅作专家促学习；融合调用 Avoid 时该第 14 维继续补 `0`，前进与 yaw 仍由 Follow 提供。
- Avoid14D 反事实诊断不再阻断开训；草稿未运行、未测试、未提交。上述决定不构成新实验、性能或论文结果。
- Avoid 已启动并完成首轮 PPO 日志：value/policy/entropy=`0.3073/0.0285/1.0838`，nonfinite skip/sanitize 与 action 均为 `0/0/0`，stage=1。completed episodes=0，success mean 的 NaN 是空集合，不能称数值故障；训练成功或权重保存均未确认。Mono 设置尚未讨论，用户明确要求不启动、不自动排队。

## 2026-09-08 作者冻结约束（本轮决定，非历史核验结论）

- 后续重训的目标速度上限固定为 `0.50 m/s`；`0.60 m/s` 只作为速度外推评测。
- 训练布局须保持生成旧 checkpoint 时的真实布局。Git 日期、训后布局修改与命名变更仍待只读历史核对；本文件不据此推断历史映射。
- 新的速度外推测试布局须先与作者共同商量后才能搭建；当前未批准布局设计、代码修改、训练或评测。
- 除必要修正外，训练设置保持与旧训练一致。memory 消融、奖励修改、网络修改及其他此前建议均未获本轮批准，也不因此新增实验。

本节记录作者本轮决策，不替代 checkpoint、服务器或历史代码证据；主线程的历史核对结果返回前，不得将其写成已验证事实。

## 使用方式与状态定义

每条记录以稳定 ID 对应一条审稿主意见；同一条中的多问项在“待补/核对”中逐项保留。后续讨论、代码核对、实验或论文修改发生时，只更新相关 ID、证据和本文件末尾的变更记录；不在后台自动修改。

| 状态 | 含义 |
|---|---|
| `方向已讨论` | 已有可用的回应原则，但仍须由作者核实措辞、原稿位置或证据。 |
| `待证据核对` | 已定位线索，尚不能写成正式事实或论文结论。 |
| `候选补充，未批准` | 审稿人合理要求或讨论过的建议，但尚未决定运行。 |
| `暂缓` | 已明确后置；不应在回复中说已修复。 |
| `拟议` | 本轮尚未讨论到具体策略，保留为后续最小回应草案。 |
| `已具备材料，待人工复核` | 本地已有表、图或记录；需确认与投稿版本一致后才能引用。 |

## 总览：优先级、主线与不应混淆的事项

| 优先级 | 主题 | 对应意见 | 当前结论 |
|---|---|---|---|
| P0 | Mono-PPO 公平性、训练/测试分离 | AE-01, R1-01/02, R2-26/29 | 需要先把训练预算、场景生成和 held-out 口径核实清楚；不把现有同类场景结果称为泛化证明。 |
| P0 | 风险查询几何、机器人尺寸与安全边界 | R2-08--18, R2-25 | 必须诚实收缩为命令方向的 clearance proxy / risk mapping；不是全足 swept-volume 安全证明。 |
| P0 | 历史轴序审计（内部） | INT-01 | 已完成有限 checkpoint/元数据范围核对；下一步仅做最小轴序修正和生产条件几何验证，随后按冻结名单重训/重评测；均未启动、未授权。 |
| P0 | Avoid 专家训练语义与横移范围 | R1-16, R2-19--22, R10-01 | 3-DoF 输出来自独立 Avoid 专家训练接口的复用，不是为增加系统复杂度；具体有效训练维度仍待 checkpoint/config 核对。横移仅覆盖结构化错列通道，不能解决墙、死端或凹障碍。 |
| P1 | 贡献、相关工作、命名和论文叙事 | AE-01, R1-03--05/07/13--14, R2-31--32, R10-02 | 以“在同一高层命令接口下学习冲突仲裁”的受限贡献为中心，不能泛化为完整导航或端到端 RL 的普遍结论。 |
| P1 | 可复现性和表述 | R1-06/08--12/17, R2-01--07/20--24, R10-03 | 优先补清楚方法、训练阶段、网络和符号；所有架构/参数须从论文 checkpoint/config 复核。 |
| P1 | 实机、速度外推、失败边界 | R2-10/27--29 | 实机 trial 数量有口径冲突，先对原稿/人工标注核对；`0.60 m/s` 是 CONTEXT 的现有工作口径所称的超训练速度评测，终审事实仍待核对。 |

### 已讨论的关键边界

- `d_safe=0.27 m`、`d_free=0.57 m` 是把命令方向 clearance 映射到风险的阈值（代码中 `0.57 = 0.27 + 0.30`）；它们本身不等同于六足 leg-inclusive envelope，也不自动给出全机/转向安全保证。
- 查询半角 `25°`（总角 `50°`）是早期硬编码的轻量 command-conditioned clearance proxy。**不是**由 D435i 的 `87°` 水平 FOV 推导，也没有证据表明它是最优角度。扩大 cone 也不会创造深度相机视野外的侧向信息。
- 不将 `0.27 m` 描述为“机身半宽加裕度”；该推导没有证据。先前提出的 `±5 cm` 阈值扫描已撤回；`15/25/45°` 扫描及 swept-footprint 查询只是候选想法，均非已批准必做实验。
- 已发现并记录训练侧地图轴序不一致（见 INT-01）。这是需要与审稿风险/复现问题隔离处理的内部问题：历史论文 checkpoint 和结果不可与未来修正后的结果混写。

## 证据索引（引用前均需人工复核）

| ID | 本地来源 | 可支持的范围 / 限制 |
|---|---|---|
| E01 | `/home/artrc/文档/RA-L_26-3764_reviews_combined_UTF8.md` | 审稿原文；本表所有 ID 的唯一意见来源。 |
| E01b | `/home/artrc/文档/26-3764_01_MS (1).pdf` | 投稿论文 PDF；本轮未逐页复核，后续用于落实原稿位置和文字修改。 |
| E02 | `CONTEXT.md` | 当前论文叙事、图表绑定、速度与实机 trial 的现行工作口径；与审稿文中“60 trials”不一致，不能单独决定事实。 |
| E03 | `TODO_LOG.md:13-22` | INT-01 的 CPU 复现记录、暂缓决策和最小后续边界。 |
| E04 | `legged_gym/scripts/train_highlevel.py:2249-2257, 2393-2396, 3169-3221, 3808-3819` | 现有代码中 clearance、风险阈值和 `25°` proxy 的定义；不是历史 checkpoint 的充分证明。 |
| E05 | `agents/final_paper_outputs_v3/tableA1_ppo_hyperparams.md`, `agents/final_paper_outputs_v3/tableA4_network_structure.md`, `agents/final_paper_outputs_v3/table3_mono_ppo_stage_probe_notes.md` | 现存 PPO/网络/Mono 产物线索；须和投稿 checkpoint、配置、原稿对应。 |
| E06 | `agents/final_paper_outputs_v3/tableA_heldout_irregular_rows.md` | 已存在 held-out 压力测试材料；其结果和协议不自动构成多环境泛化结论。 |
| E07 | `agents/final_paper_outputs_v3/fig7_real_robot_arbitration_15s_notes.md` | 一段实机仲裁轨迹的机制证据；不能单独支持 trial 成功率、碰撞率或 sim-to-real 优越性。 |

## 跨评论共用的最小工作项

| 工作项 | 覆盖 ID | 内容 | 状态 |
|---|---|---|---|
| X-01 原稿/产物口径核对 | R1-01/02/17, R2-18--24/26/28--29, R10-03 | 固定投稿版本、每个 checkpoint、训练命令/配置、评测布局和 trial 人工标注；特别核对“40 trials / 60 trials”和速度范围。 | `待证据核对` |
| X-02 方法与复现表 | R1-11/12/16/17, R2-08--10/14--18/20--24, R10-03 | 以历史实际配置补训练阶段、观测/动作、地图/encoder、风险与 memory 输入、终止和 simulator/asset；不可按当前代码倒推出历史事实。 | `待证据核对` |
| X-03 审稿版定位与限制段 | AE-01, R1-03--05/07--10/13--14, R2-08--13/16/19/25/27/30--32, R10-01/02 | 收紧贡献、命名、相关工作、横移范围、静态障碍、安全边界；先写最小且可证实的文字。 | `方向已讨论` |
| X-04 追加实验决策门 | R1-01--03, R2-13/26/29 | Mono 公平训练、独立 held-out、多 embodiment、动态障碍、速度表等均先冻结协议与成本，再由作者批准是否运行。 | `候选补充，未批准` |

## AE

| ID | 审稿意见（忠实概述） | 当前回应方向 | 证据 / 待补 | 状态 |
|---|---|---|---|---|
| AE-01 | 更清楚突出主贡献和相对既有、尤其紧密相关工作的 novelty。 | 将贡献限定为：在固定低层行走与明确专家命令接口上，学习风险条件化的连续 command-space conflict arbitration；解释为什么它不同于固定优先级、直接相加、单体策略和 target-aware velocity-space search。不要声称已解决通用导航。 | X-03；R1-04/05/13，R2-31/32。需逐段核对原稿和引文。 | `方向已讨论` |

## R1（5 major + 10 minor + 2 questions）

| ID | 审稿意见（忠实概述） | 当前回应方向 | 证据 / 待补 | 状态 |
|---|---|---|---|---|
| R1-01 | Mono-PPO “same training budget”不清，比较未证明公平；若要主张训练成本，须让 Mono 充分训练并把成本另列。 | 不将现有 Mono 结果外推为“所有端到端 PPO 都不行”。正文只可说本 benchmark、该训练预算和相同接口下，专家分解提供有用归纳偏置。先核对实际 budget、停止准则、种子、接口和候选 checkpoint；是否补训由 X-04 决定。 | E02 的 Mono 定位；E05 的现有 probe。 | `待证据核对` |
| R1-02 | 训练与评测同环境可能过拟合，建议多个未见环境。 | 明确主表训练/validation/eval 的关系；现有 irregular-row 仅可作为有限 stress/OOD 材料，不能包装成充分的多环境泛化。若补充，先预注册独立布局和不参与调参的最终评测。 | E06；TODO 的 held-out 记录；X-01/X-04。 | `候选补充，未批准` |
| R1-03 | 标题中的六足 embodiment 动机不足，建议更简单 embodiment 的仿真消融。 | 先在论文中说明六足平台是已部署的目标平台，而非已证明的 embodiment-general 方法；不凭空声称简化平台不需要该架构。跨 embodiment 消融是候选重大实验。 | X-03/X-04。 | `候选补充，未批准` |
| R1-04 | 相关工作只对比，未充分说明为何本法应更优。 | 用问题假设和接口差异，而非泛化 superiority，说明：固定/加性融合缺少 state-dependent conflict arbitration；本法的有限收益须由同一基线实验支持。 | AE-01、R1-13、R2-32。 | `方向已讨论` |
| R1-05 | 系统看似过度复杂，消融后仍未说明必要性。 | 以最小结构叙事回应：解析 Follow、单独训练的 Avoid 与仅学习 gate 各自承担可识别责任；保留复杂度/局限，避免说每一部件都“必要”。补充 ablation 的真实覆盖和未覆盖范围。 | E02；R1-01/03/16。 | `方向已讨论` |
| R1-06 | 多处措辞过度 elaborate，影响清晰。 | 逐句改为可定义、可验证的短句；删除没有量化支撑的修辞。 | X-03；需逐页标注原稿位置。 | `拟议` |
| R1-07 | PCR-Net 与 Sarode 等已有 PcrNet 名称重叠，且 PCR-Net / gate / Learned-w 指代不一致。 | 决定正式名称/缩写前，不对外承诺。修订稿中应统一“系统、gate、具体变体”的名称，并显式说明它不是点云配准 PcrNet；若保留名称，给出明确区分和引用。 | R1 原文；X-03。 | `拟议` |
| R1-08 | “conflict”未定义；本法也未保证跟踪不会在避障中丢失。 | 明确定义为 Follow 与 Avoid 候选命令在受限命令空间中的目标进度/局部 clearance 张力；把“preserve tracking”改为经验性 trade-off 目标或观测结果，不能写成保证。 | E02 的 learned-w 叙事；R2-27。 | `方向已讨论` |
| R1-09 | 把 occupancy map 称为 affordance map 容易混淆。 | 在首次出现处分开 occupancy、clearance/cost proxy、local map 三者；若输入确实是双通道 occupancy + proxy，不将其泛称为语义 affordance。 | E04: `AVOID_LOCAL_MAP_CHANNELS`。历史版本仍需核实。 | `待证据核对` |
| R1-10 | 正交子空间的两目标未必不能同时最大化；原句数学上可疑。 | 删除绝对数学断言，改成实际受速度/执行器/环境约束下，候选命令可能呈经验性权衡；不给出未证明的不可兼容性主张。 | X-03。 | `方向已讨论` |
| R1-11 | 符号/变量不完整、`m_t`未使用，`y`同时作坐标和 gate scalar。 | 建立符号表并逐式清理：删未用量、补定义、区分位置坐标与 gate（如 `y_g` 或其他不与既有 `w` 冲突的符号，以原稿最终符号为准）。 | X-02。 | `拟议` |
| R1-12 | gate 参数、risk+memory、memory 输入方式、slew-rate 与 clearance scaling 的必要性不清。 | 分别说明每项的定义、是否进入 actor、为何存在及其适用边界；仅用已完成消融支持“有益”。对 memory，当前代码线索为风险随前向位移衰减的保持机制，不能据此证明投稿 checkpoint 的输入。 | E04: 941-1021；TODO risk-memory 记录；X-02。 | `待证据核对` |
| R1-13 | 与 Scheidemann 的 command-space arbitration 区别不清；[3]/[4] 同引但只比较其一，须说明。 | 逐篇按任务、观测、命令层级、训练/部署条件说明差别；对未作直接实验的文献给范围和可比性理由，不贬低。 | X-03；需核对参考文献和原稿。 | `拟议` |
| R1-14 | 关于既有 stack 耦合不同 sensors/planners/controllers 的说法缺引用。 | 加直接支持该句的引用，或删/收缩为本工作接口描述。 | X-03。 | `拟议` |
| R1-15 | Fig. 6 底排太小、文字不可读。 | 复核投稿 PDF 中 Fig. 6 的实际绑定和字号；再拟定最小图/排版修改方案。 | E01b；E02 Fig. 6 绑定。 | `待证据核对` |
| R1-16 | 为何训练 3-DoF Avoid，却在之后丢弃其中两维，而不直接训练 1-D？ | 诚实说明 3-DoF head 源于 Avoid 专家单独训练阶段的复用需求，不是为了刻意增加 PCR 运行模块复杂度；部署只采用与命令分解一致的横移分量。必须核对历史 checkpoint 的实际命令、缩放和有效训练维度；不声称 3-DoF 优于 1-D，也不虚构 1-D 消融。 | E05 `tableA4_network_structure.md` 仅显示 Avoid output=3；X-01/X-02。 | `方向已讨论` |
| R1-17 | PPO 的地图是否未修改输入？若是，可能偏离标准并损害性能。 | 精确披露 map 的通道、归一化/encoder、是否 raw input，以及 Avoid、Gate、Mono 各自输入；不以“标准实践”替代实际配置。 | E04 的当前 map 生成线索；E05；X-02。 | `待证据核对` |

## R2（3 组织 + 3 展示 + 1 强项建议 + 6 感知 + 5 风险 + 4 分解 + 2 训练 + 1 安全 + 4 结果 + 3 文献）

| ID | 审稿意见（忠实概述） | 当前回应方向 | 证据 / 待补 | 状态 |
|---|---|---|---|---|
| R2-01 | 引言 em dash 过多、部分句子复杂。 | 全文语言压缩，与 R1-06 合并处理。 | X-03。 | `拟议` |
| R2-02 | 基础概念到实现细节的层次不一致。 | 先问题/命令分解/风险定义，再训练和实现细节；不借重排改变方法主张。 | X-03。 | `拟议` |
| R2-03 | 大小写不一致。 | 建立术语表并全文统一。 | X-03。 | `拟议` |
| R2-04 | 补充视频需标明加速倍数。 | 核对原视频时间基准后加速度指示；不能猜测倍数。 | X-01。 | `待证据核对` |
| R2-05 | Fig. 2 分辨率低，建议矢量图。 | 若源图可重导出，使用矢量版本；同时保持图中方法名与正文一致。 | X-03；R2-06。 | `拟议` |
| R2-06 | Fig. 2 未标出 PCR/Gate/conflict-aware fusion，文字和图不一致。 | 统一图内 block 名称、caption 与 Sec. IV 用语；不增加未实际存在的功能块。 | X-03。 | `拟议` |
| R2-07 | 强调仅学习 gate 带来的可解释性/确定性。 | 可写“解析专家和明确融合公式使命令来源可追踪”；不能把 deterministic inference 或 convex hull 误写成安全保证。 | R2-25；E02。 | `方向已讨论` |
| R2-08 | 地图覆盖、侧向可见性、instantaneous/built、robot-centric 构建方式不清。 | 明确相机观测、投影/可见性、局部坐标和地图是否累计；在感知语义上，视野外是未知而非自由。同时如实披露当前 query 对无可见 blocked cell 会返回最大距离/风险零的实现，不能暗示代码已采用 unknown-aware 风险处理。 | E04；TODO 的相机/局部图记录；X-02。 | `待证据核对` |
| R2-09 | 无侧向 affordance 时如何 side-step。 | 说明横移命令的风险判断只基于可见局部地图；它不创造侧向观测，不能保证 camera FOV 外 clearance。必要时把这一点列为感知限制。 | `25°` 边界说明；R2-12。 | `方向已讨论` |
| R2-10 | YOLO 是否只检测目标？目标深度 mask 与 stereo depth 的职责应明确。 | 逐项披露：目标检测、目标相关深度剔除/掩码、障碍来自何种深度/占用过程；动态物体未处理需单独限界。 | X-02；R2-11。 | `待证据核对` |
| R2-11 | 风险查询未见 footprint/body-size，约 `0.65 m` 全腿包络和原地转向如何处理。 | 策略第二通道已有按固定 clearance 做的障碍膨胀，故不能把它写成完全没有尺寸裕度；但这不是全腿 swept-volume、完整轮廓碰撞或原地转向验证。命令 cone clearance 仍只是命令方向 proxy；DWA-style rollout 的 footprint checking 不能反向证明 PCR runtime query 有同样保证。 | E04；E02 DWA 说明；INT-01。 | `方向已讨论` |
| R2-12 | 凹障碍、后退指向/凹腔时会怎样。 | 将 lateral-only formulation 限定为错列开口中的横向通过；不声称能恢复于凹腔、死端或连续横墙。 | R10-01；X-03。 | `方向已讨论` |
| R2-13 | 动态障碍未处理，至少讨论限制，最好实验。 | 明确静态障碍假设，移动目标不等于动态障碍处理。动态障碍实验为候选，当前未批准。 | E02 的 out-of-scope 记录；X-04。 | `候选补充，未批准` |
| R2-14 | `0.09375 m` 栅格精度表述不合适。 | 以“3 m extent、32×32 resolution（约 9.4 cm/cell）”表达；在原稿/历史配置核对后再写。 | E03 的 32×32/3m CPU 条件；E04。 | `待证据核对` |
| R2-15 | clearance proxy channel 如何计算。 | 给出 occupancy 与 clearance/cost proxy 的计算、可见性处理、范围和 actor 输入；第二通道含固定 clearance 膨胀的 safety/cost 语义，但不等同完整轮廓碰撞验证。区分该 Avoid local-map channel 与 command-cone minimum clearance。 | E04: 587-636, 3169-3221；X-02。 | `待证据核对` |
| R2-16 | 为什么仅 `25°` 半角；近障时是否要为全机 swept motion 看更侧方。 | 解释为早期、固定、命令对齐的最小 proxy，而非 FOV 对齐、full-footprint sweep 或最优安全角；已有第二通道固定裕度也不改变这一限制。扩大角度若没有观测不会新增信息。仅在作者批准后再做敏感性/替代查询。 | E04: 3173；R2-08/09/11。 | `方向已讨论` |
| R2-17 | `d_safe=.27`、`d_free=.57` 无依据，与结果 clearance margin 关系和敏感性不清。 | 将其标为风险映射超参数，报告来源/实际设定与结果 metric 是否不同；不宣称几何推导或普适安全阈值。敏感性是候选，先核实是否有历史 validation。 | E04: 2249-2257, 2393-2396, 3808-3819；X-02/X-04。 | `待证据核对` |
| R2-18 | speed-scaled risk 在方法段提出又丢弃，报告结果到底用哪个版本。 | 以投稿 checkpoint 和评测配置为准，明确最终报告变体；把其他变体移至消融/附录或删除。 | X-01/X-02。 | `待证据核对` |
| R2-19 | 横移 Avoid 缺动机，后退是否在死端/凹几何必要。 | 指向任务范围：训练/评测强调错列行间 passability mismatch，横移是该受限任务的专家职责；明确不覆盖墙、死端、凹障碍或 target 在障碍前骤停。 | R1-16，R10-01。 | `方向已讨论` |
| R2-20 | 未清楚说明 Avoid 的 reward、observation、termination。 | 用实际训练配置列完整表，含外生前进/动作有效维度（若适用）、奖励、终止及 curriculum；不能从当前源码猜历史数值。 | X-01/X-02。 | `待证据核对` |
| R2-21 | Avoid 训练时 Follow 是否在环？该阶段有何 tuning？ | 说明独立 Avoid 训练与后续 PCR gate 训练的先后、是否注入 Follow，以及各阶段可调项；需要实际日志/config 佐证。 | R1-16；X-01/X-02。 | `待证据核对` |
| R2-22 | 使用何种 simulator setup？ | 对三阶段分别列 simulator、机器人 asset、terrain/layout、低层策略与随机化。 | E05；X-02。 | `待证据核对` |
| R2-23 | Sec. IV-F 需对三个训练阶段分别说明 simulator 和 asset setup。 | 形成按阶段的 simulator、机器人 asset、terrain/layout、低层策略与随机化表；用历史实际配置，而不是当前默认值。 | E05；X-01/X-02。 | `待证据核对` |
| R2-24 | 每个 learned module 的精确 obs/action、gate 的“robot state”是什么？ | 形成 module 表，完整到 map channels、标量维度、动作语义和网络输入拼接。 | E05；R1-17，R10-03，X-02。 | `待证据核对` |
| R2-25 | convex-hull/bounded-residual 不意味着专家混合安全，尤其高冲突。 | 保留为动作幅值与命令来源受限的性质，不称 safety guarantee；承认 expert 合理性和小扰动安全性是假设，冲突高时仍可能失败，需要经验评测。 | E02；R1-08，R2-07。 | `方向已讨论` |
| R2-26 | 最难 curriculum stage 是 seen 还是 held-out，须明确 train/test。 | 逐表、逐图标注 seen/validation/held-out；不要用固定 hardest stage 替代独立测试。 | R1-02；E06；X-01。 | `待证据核对` |
| R2-27 | 应讨论 failure modes、触发限制和是否能 recovery。 | 增加受控的 failure taxonomy：超出侧向可通过假设、不可见侧障碍、凹/连续障碍、目标/深度问题等；只报告已有观察或明确为限制，不编造 recovery。 | R2-08--13/19/25；E07 限制。 | `方向已讨论` |
| R2-28 | 只有 gate 随速度变化？为什么超过训练速度时会坏，不能只给 gate 的弱速度梯度。 | 分清 target trajectory、专家、gate、低层/地图采样及训练分布；`0.60 m/s` 仅为超训练速度评测，不能把性能归因于单一 gate 或宣称强鲁棒。 | E02；X-01。 | `待证据核对` |
| R2-29 | Mono 只在 `.35 m/s` 难度曲线评测，补 `.50/.60` 会更完整（nice-to-have）。 | 记录为非关键候选补充；先完成 R1-01 的预算公平性核对，再决定是否运行。 | R1-01；X-04。 | `候选补充，未批准` |
| R2-30 | 缺 human-following/HRI 文献。 | 补与行人目标跟随直接相关的工作，并明确本工作不是社会导航/人机交互行为建模。 | X-03。 | `拟议` |
| R2-31 | 希望有 reward shaping vs constrained RL 的文献依据。 | 补适用的强化学习安全/约束文献；不将当前 reward 设计说成约束 RL。 | X-03。 | `拟议` |
| R2-32 | MoE/learned gating 只引 1991，需更新近代工作、应用和动机。 | 补近代 learned gating / expert composition 文献，并说明仅借用 gating 思想，不夸大为通用 MoE 结论。 | X-03；AE-01。 | `拟议` |

## R10（保持审稿编号，不改写为 R3）

| ID | 审稿意见（忠实概述） | 当前回应方向 | 证据 / 待补 | 状态 |
|---|---|---|---|---|
| R10-01 | 严格横移避障假设不能处理连续横墙、死端或目标在障碍前骤停。 | 明确本方法针对可通过的错列开口中横向避障，不是 complete local planner；这些情况需要上层停止、后退或重规划，当前方法未验证。 | R2-12/19；X-03。 | `方向已讨论` |
| R10-02 | 基准和硬件多为周期错列行；须讨论随机杂乱/非凸障碍范围。 | 说明规则行用于隔离 lateral passability mismatch；对不规则/非凸环境只作受限 stress evidence（如有），不称广泛泛化。 | E06；R1-02，R2-12/26。 | `方向已讨论` |
| R10-03 | 缺 map encoder 的 CNN kernel/stride/channel、投影维度和与 31 scalar 的拼接细节。 | 从实际模型定义/保存配置补 exact architecture；不要只写 `256-256 ELU + Beta heads`。 | E05 `tableA4_network_structure.md` 是线索；X-01/X-02。 | `待证据核对` |

## INT-01：内部发现的训练地图轴序不一致（不是审稿人已知结论）

**2026-09-08 已批准的确定性修正（CPU 已验证，Isaac 待验证）。** Follow expert 内部的 world→body 与转向符号保持不变；仅将 PCR/real runner 中 body-relative `goal_buf` 还原到世界坐标的公式改为 `R(+heading)`，使往返恢复原目标。另对 scene raster 增加“完全无正面积地图交集即跳过、部分交集先裁剪”的规则，防止地图外障碍被 index clamp 压到边界格。两项均不改变 cone、FOV、safe/free、奖励、课程、网络或 sim-real 输入口径；不写作已重训、已重评测或实机性能结论。

**已验证的有限事实。** 当前训练代码的 occupied-cell / angle 路径使用 `ij` 语义，而 distance-map 路径使用 `xy` 语义；在独立 CPU AST 抽取测试中，32×32、3 m 地图、局部障碍点 `(0.046875, 0.515625)` 对应格 `[16,5]`：同格几何距离为 `0.517751 m`，distance map 读数为 `1.833526 m`。一个合成小障碍覆盖四格时，cone 查询得到 `1.784947 m`，而选中格的正确几何最近值为 `0.517751 m`；全局 yaw 为 `0°/90°` 均复现，坐标旋转未抵消该差异。

**不能由此推出的事。** 该 CPU 检查不是 Isaac 闭环训练/评测，未证明论文 checkpoint 的定量性能变化，也不证明实机全部路径错误。它已证实的是同格 distance 错配，**不必然**证明 occupancy 与 angle 的选区整体互换。部署侧两个 map 建图路径均为 `ij` 的线索，仍需逐模块核实。原有障碍位置和 PhysX 碰撞并未因这项检查被改写。

**P0 下一步：最小修正前的生产条件核对。** 已完成有限历史审计：服务器 Git 显示 `xy` distance-map 路径早于 2026-01-04 存在，`ij` occupied/geometry 路径于 2026-03-12 加入，之后未见移除；三份归档 checkpoint 均在其后。仍须用实际生产分辨率、extent 和原点，在全可见条件下做四组方向-障碍 probe：前方障碍 × 前进/右移命令，以及右侧障碍 × 右移/前进命令，并加入正确距离数值断言。全可见只去除 FOV 筛选，不扩展原有地图支持域；若边界点不在有效格内，选最近的域内代表点并记录。`16×16` 可视化可辅助理解，但不能替代上述生产条件的断言。

### 本轮最小执行结论（仅建议，未启动）

该问题属于**栅格同索引空间含义不一致**：同一 `[i,j]` 同时被解释为 `ij` 几何格与 `xy` 距离格。本轮只读核对的 checkpoint 哈希均与 `agents/final_paper_outputs_v3/MANIFEST.md` 一致；下表不把旧结果的成功率或碰撞率自动判为错误。

| 对象 | 本轮可证范围 | 修正后的最小动作 | 证据强度与唯一局限 |
|---|---|---|---|
| Avoid | `agents/avoid_best.pt`（`fd23…`）作为 Gate 的已绑定依赖；其 checkpoint `experiment_meta.disable_risk_scale=false`，历史代码期内 PP 风险缩放可用。 | 先从头重训 Avoid；再供后续 Gate 使用。 | 强：`resolved_protocol.json` 的 `aux_checkpoints.avoid_ckpt.experiment_meta` 直接来自 checkpoint 元数据。缺原训练命令/源码快照，不能复原每一轮训练。 |
| Learned-w | `agents/moe_teacher_best_learnedw.pt`（`0626…`）的已绑定元数据为 `learnedw2`、`risk_memory=true`、`disable_risk_scale=false`。 | 使用修正后的 Avoid 从头重训。 | 强：`agents/eval_data_learnedw_diag/s_0.6/moe_teacher_s_pcr_line_avoid_basic_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_seed1_20260602_141839/resolved_protocol.json` 的 `primary_checkpoint.experiment_meta`；缺训练源码快照。 |
| Risk-only | `agents/moe_teacher_best_risk_only.pt`（`d0a4…`）与原 run 中 `best_online_reward.pt` 已 SHA-256 相同；元数据为 `risk_only`、`risk_memory=false`、`disable_risk_scale=false`。 | 使用修正后的 Avoid 从头重训。 | 强：`agents/risk_only_seed1_targetview/moe_teacher_risk_only_gam0.15_20260609_230028/run_meta.json`，及 `agents/eval_data_risk_only/s_0.6/moe_teacher_s_pcr_line_avoid_basic_risk_only_gam0.15_seed1_20260610_153258/resolved_protocol.json` 的 `primary_checkpoint.experiment_meta`。 |
| Mono-PPO | 找到 5 份 `agents/mono_ppo_seed1*/**/run_meta.json`，但未将投稿/主表选择 checkpoint 以哈希唯一绑定。 | 公平性补训时统一使用修正版本；不因此阻塞上述名单，也不在本轮启动。 | 中：原 run 记录存在，但轴序影响范围和主表 checkpoint 绑定未证实。 |
| DWA、Rule-Override、Additive 等无学习基线 | 依赖同一距离图的 clearance/near-miss 指标须与修正版本统一重评测；依赖 Avoid 的方法换用新 Avoid。 | 重评测并重算受影响指标，不默认重训。 | 中：本轮代码核对已定位指标路径；各方法启用配置仍须随重评测留痕。 |
| Analytic Follow、冻结低层 | 未发现其自身需要学习该距离图。 | 不默认重训；仅作为新组合中的冻结依赖复核。 | 有限：不把“未见直接训练输入”扩大为性能不受影响的证明。 |

**修正、训练与实机边界。** 最小轴序代码修正与 CPU 几何验证已在独立分支完成；重训、重评测、部署和实机动作均未启动。固定代码 + 旧 checkpoint 的运行只可作为明确标注的干预诊断，不能当作重训后公平主结果；short-train/fine-tune 也只是未批准选择，不能替代 clean retrain。即便实机 map 路径为 `ij`，旧实机 trial 也不自动验证新策略；是否补实机必须绑定部署 checkpoint、配置和改动另行决定。任何未来修正结果必须与投稿历史结果分开呈现。

**当前状态：`最小轴序代码修正和生产条件 CPU 几何验证已完成`。** 32×32、3 m、全可见、base-center 的 AST CPU 测试覆盖 distance、scene raster 和 heightfield 的 `[x_right, y_forward]` 索引、旧源码的 `(0.046875, 0.515625)` 反例、yaw 0°/90°、方向负查询及零命令回退；本地既有 torch 1.8.1 CPU 环境中，旧 HEAD 在反例 `1.833526 m != 0.517751 m` 失败，修正工作树通过。测试夹具只为旧 torch 保留源码 `xy/ij` 语义，不改变生产实现。服务器未部署，训练、评测和实机动作均未启动。仿真 query 显式应用 visible mask，近零命令以 `1e-3` 回退 global clearance；实机 shim 依赖输入 map、无独立 visible 筛选，近零 `1e-6` 返回 extent，且 `pcr_realplay.py` 优先用带固定裕度膨胀的 `risk_blocked_map`、再取前向风险上界并过滤。这些既有差异未在本轮统一，不是安全或论文性能结论。仍按原冻结名单决定后续 clean 重训/重评测，正式回复不得写“已重训”或“排序不变”。

相关记录：E03；当前代码的 `train_highlevel.py` 地图生成/查询区域。关联审稿问题：R2-08--17、R1-17、R10-03（仅说明需要更严谨复核，不能在正式回复中把内部发现当成已完成修复）。

## 正式答辩前的收口顺序

1. **最小轴序修正与生产条件 CPU 几何验证已完成**：独立分支的四组方向-障碍 probe 与正确距离断言通过，未顺带改 cone、阈值或 memory。下一步若获批准，按冻结名单执行 `Avoid → Learned-w/Risk-only` 从头重训，并对无学习基线统一重评测；均尚未启动。
2. **再做 X-01/X-02/X-03**：逐项核对投稿 PDF/LaTex、实际训练/评测/实机口径，消除 `40` 与审稿概述中 `60` 的冲突，并据此补方法表、网络细节、符号、图和范围限制，形成逐条英文答辩草案。
3. **最后过 X-04 决策门**：只对仍会改变接受判断的缺口批准最小补实验；目前没有训练、重训、动态障碍、multi-embodiment 或参数扫描正在执行。

## 简要变更记录

| 日期 | 内容 |
|---|---|
| 2026-09-08 | 代码 `c4353cc` 已同步至审稿分支，并在隔离服务器 root 完成原生 torch 1.12 CPU 语法、轴序、速度课程和 Avoid 14D 检查；用户随后要求先由 GPT 审阅代码，Avoid 启动命令未发送、无本轮训练会话/进程/输出。 |
| 2026-09-08 | 新建总账：覆盖 AE 1 条、R1 17 条、R2 32 条、R10 3 条，共 **53** 条可执行主意见；建立证据边界、跨评论工作项与 INT-01 历史审计优先、修正/训练未启动的记录。 |
| 2026-09-08 | INT-01 完成只读历史范围核对：三份归档 checkpoint 的 SHA-256 与 MANIFEST 一致；据 checkpoint 训练元数据和历史代码，Avoid → Learned-w/Risk-only 纳入修正后从头重训建议，Mono 公平性补训与无学习基线重评测分开处理；未启动修正、训练或评测。 |
| 2026-09-08 | INT-01 最小轴序修正：训练侧 heightfield 输出与 distance map 改为 `ij` `[x_right, y_forward]` 语义；AST CPU 几何测试覆盖 32×32、3 m、前/右障碍与前/右命令正负组合、yaw 0°/90°、已知反例和 heightfield/scene 同轴。本地既有 torch 1.8.1 中旧 HEAD red、修正工作树 green；未部署服务器、未启动训练或评测；这不构成论文结果或完整 footprint 安全验证。 |
