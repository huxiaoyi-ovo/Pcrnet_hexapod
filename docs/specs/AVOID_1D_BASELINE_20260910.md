# Avoid 1-D 正式工程基线（2026-09-10）

> 状态：**C1/D2 的纯 CPU 代码与 contract 检查、以及真实 `s_avoid_clutter` 的静态 Isaac 场景核对已完成；首个冻结 Stage-1 Sanity 正在独立远端目录运行，尚无训练、评测、部署或能力结论。** 最新 CPU-PhysX 四阶段原生 viewer 预览通过，live actor 与 runtime metadata 完全一致，证据在 `outputs/avoid_clutter_constructive_acceptance_20260910/preview/`；红线不是策略轨迹。用户明确授权后，独立 fresh server bundle 的 4-env/24-step CPU-PhysX smoke 通过：fresh import、低层权重 SHA、`[4,14]` state、`[4,1]` action、`[4,3]` command 和全程 finite 均通过，`done_total=0`。原始日志与 provenance 在 `outputs/avoid_clutter_constructive_acceptance_20260910/smoke/`。历史本地 Torch 1.8 仍在 post-processor `Tensor.clamp` 处阻塞 0 个高层步；这是独立环境事实。上述 smoke 不构成训练、策略性能、评测、部署或安全验收。后续 fresh snapshot 的 GPU2 2-env/2-step smoke 尚未运行：用户已有训练授权，但自动审核两次阻塞了该具体源码/resources/low-level payload 到指定新隔离目录（见 `outputs/avoid_sanity_20260910/deployment_blocked.json`）。控制器未 arm，formal 未启动；Sanity 的原 201-iteration 预算和 watchdog 不变，完成 `model_200.pt` 后仍可验收。real builder、runtime 与部署网络已接入 revised contract，但真实 checkpoint、ROS state provider、硬件与原训练环境仍待单独验证。本文不是投稿稿、实验报告或实机运行证明。

> **2026-09-10 用户长期冻结（仅本项目 revise 阶段）**：一次选择性能、正确性、sim-real 一致性和返修说服力综合最优的可实施方案；只专门验证影响核心架构、训练成立或论文结论的高风险不确定性。低概率低影响猜测采用合理工程默认，不以小测试或横向扩审计/消融/场景/参数拖延开训；必要实现正确性检查仍须完成，且不得编造结果。地图已冻结为 sim/PCR/real 同语义即时 canonical local map，关闭 real-only obstacle-map memory，保留 PCR learned-w risk memory；不新增 dropout 小测试或再分支讨论。相机实测 origin/offset 等未知不伪造，也不新增测试要求。C1/D2 已完成代码与 CPU contract 检查，不构成 Isaac、训练或部署验收。

## 1. 冻结的行为与权限边界

Avoid 是 robot-centric 的局部横移专家，不是全局规划器。它的最终 action 只拥有 lateral authority：

```text
a_A = u_A,x
Standalone request: u_req = [u_A,x, v_drive_nom, 0]
PCR: u_A = [u_A,x, 0, 0]
     u_F = [0, u_F,y, u_F,omega]
     u = y_eff u_F + (1 - y_eff) u_A
```

公共 post-processor 位于请求和低层 locomotion 之间；若安全层降低 forward，后续步骤不得把它强行抬回。Avoid 不拥有 forward/yaw authority，却可见名义 forward request 与实际运动状态。

因此 PPO 的 action distribution、log-prob、entropy、rollout buffer 与 `evaluate_actions` 都必须为 1-D；不再采用“独立训练 3-D，再在 PCR 丢弃两维”的路径。PCR/Gate reward、旧 `s_avoid_basic` 和 staggered-row 几何本轮不变。

## 2. Observation contract

Avoid actor 的 14 维 state 为：

```text
[0, 0, 0, vx, vy, yaw_rate, 0, roll, pitch, 0,
 last_exec_x, last_exec_y, last_exec_omega, v_drive_nom]
```

按 0-based index，永久屏蔽的是五槽 `[0,1,2,6,9]`，即 absolute `x/y/yaw`、height、`prev_gate_y`；它们保留状态布局兼容性但不提供 Avoid actor 信息。`vx/vy/yaw_rate/roll/pitch` 是真实 body-frame motion/IMU；`last_exec_*` 是上一拍真正下发的完整命令，不是物理测得速度。`v_drive_nom` 的唯一语义如下：

| 场合 | `v_drive_nom` |
| --- | --- |
| Standalone Avoid | 当前外部 forced-forward request |
| PCR 融合 | Follow 的 forward proposal `u_F,y` |
| 实机 | 实时 Follow 的 forward proposal `u_F,y` |

它不是实际速度或最终前进命令，因此 `vy` 和 `last_exec_y` 保留；PCR 中该 proposal 仍可被 `y_eff` 和 post-processor 改变。历史 `forced_forward_speed` 不再是新基线名称或语义。关键状态缺失、非有限或过期时必须进入明确 hold/safe handling，不能静默补零继续执行。

Avoid actor 另接收：canonical 2-channel local map、`[side_preference, 0]`、canonical map-derived `difficulty`。Gate/Mono actor 继续只屏蔽 `x/y`，critic 保留真实 `x/y`；Follow、几何计算、reward 与 termination 一律使用 raw state。Avoid critic 可以使用真实 `x/y/yaw/height`、GT/full local map、progress/cross-line 等明确的训练 privileged information；这些不得进入 Avoid actor/部署路径。

`side_preference` 统一为 `s=target_x/sqrt(target_x^2+target_y^2)`，坐标 `+X=right,+Y=forward`，actor goal 为 `[s,0]`。目标无效、过期、后方或距离异常时 `s=0`。Standalone 只生成虚拟目标方位并调用与 PCR/实机相同的 converter；bearing 初始范围约 `[-40°,40°]`，需左右对称、时间平滑、含 `s=0`，且独立于障碍几何与 forward speed。

## 3. 场景、课程与物理候选范围

新建 `s_avoid_clutter`；旧 `s_avoid_basic` 与旧 PCR 布局保持不变。批 A 的单一障碍带 generator 已完成固定种子 CPU 几何检查；批 B 的 actor-pool、reset、实际带指标与终止/课程接入已在真实 Isaac 静态场景中实例化。候选使用 flat plane、半径 `0.15 m`、高 `0.50 m` 圆柱、少量 `y`-jitter 与每带 1–2 个真实 gap；不增加箱体、墙、动态障碍或地形随机化。`Free/Central` 保留为同一分布中的样本，不是额外模板；非零 lateral shift 范围只用于确实需要横移的样本，`Free/Central` 可为 0。此前“静态截图不构成训练批准”是当时事实；用户随后已明确授权隔离快照的过夜 Sanity、独立资格评测及仅在正式 gate 通过后的训练。该授权仍不把静态截图或 smoke 变成性能、部署或安全结论。

每个时刻维护机器人中心的 reachable interval **并集**，而非最近端点距离或单一凸包。障碍 envelope 先给出 safe intervals；再用侧向能力、响应延迟、前进速度与首次可见后的反应时间，把当前每个可达分支分别传播。两个合法 gap 在不可见的未来障碍带揭示前都必须仍可恢复，不能因中间不可达区被凸包填平而虚构切换。实际二维障碍/envelope 校验还必须确认 gap 以外没有意外斜向通道。

每带先记录 `d_min=dist(R,G)` 和既有时间预算所用的 `d_max=max_{x∈R}dist(x,G)`：`mandatory` 当且仅当 `d_min>=.10 m`。`choice` 不是“不同分支各有一个出口”，而是存在同一前序点 `x∈R`，能在预算内到达两个不同 gap 的内缩区间；候选 `x` 来自 `R` 与两条 gap 的预算膨胀区间交集。两条从该同源点进入、穿出带的二维路线必须通过当前全部已生成圆柱，且后续新增圆柱也须再次检查全部历史见证路线。`effective=mandatory OR choice`，同一带同时满足两者只计一次；这是存在性候选，不声称每个实际策略状态都有两个出口。

`10 m` half-width 是圆柱 row 的资源上限：每带实际 gap 必须位于该 row coverage 内，但该 cap 不是远端越界终止、居中惩罚或形式化 anti-bypass guarantee。外围绕行只降为训练中的行为 QA：观察是否出现长期单向横漂、绕过障碍带端点后仍成功的稳定捷径；未出现则不再改动，出现后才针对实际行为作最小修正，因此当前不阻塞训练。若以后确实加入远端 guard，`out_of_bounds` 必须作为独立 termination reason 记录，不得并入 `collision` 或污染论文 safety metric。名义 forward request 不等于实际速度；若执行中减速或停滞，不能无条件宣称有限带宽必然不可绕。出生缓冲与首次可见后的反应余量保留。

课程统一为 `Sanity → Core → Choice → Full`，而非八套场景模板。参数是能力约束筛选前的工程候选，不是已标定值、训练结果或物理保证；`Full` 只是扩展范围，不同步取每项最难值。

| 阶段 | `v_drive` (m/s) | `m_clear` (m) | `side offset` (m) | `κ_time` | 候选带数 | 有效决策筛选 |
| --- | --- | --- | --- | --- | --- |
| Sanity | .20–.30 | .20–.30 | .15–.30 | 2.5–3 | 1 | decision layout 至少 1 个 mandatory |
| Core | .15–.35 | .15–.30 | .15–.40 | 1.8–3 | 1–2 | decision layout 至少 1 个 mandatory |
| Choice | .20–.40 | .10–.25 | .15–.50 | 1.4–2.5 | 2–3 | decision layout 至少 1 个同源 true choice |
| Full | .15–.50 | .075–.30 | 0–.60 | 1.2–3 | 2–5 | decision layout 至少 2 个 effective |

`side offset` 是 mandatory gap 相对当前 reachable union 的实际偏移：Sanity/Core 从阶段范围与当前 speed/κ 的相机可行域交集抽取；Full 固定 `.10 m`，它位于 Full 的 `[0,.60]` 范围内并保证两条 mandatory 带的联合可行性。safe center width 固定为 `2m_clear`，physical gap width 为 `2m_clear+.56 m`。每阶段采样为 10% `Free`、10% `Central`、80% `Decision`；`Free/Central` 不计有效决策。Choice 的最后一带使用同源双 gap；Full 不要求 true choice，前两带为 mandatory。非开口 chain 保留两端精确位置，内部 `x` 在相邻二维间距下限与上限的剩余量内独立随机偏移；短段仅一个 interval 时固定，既不移动端点也不产生额外通道。旧截图来自旧间距，不重拍且不视为本次最新证据。

构造式生成器已通过固定 1024 布局 CPU 综合检查（seed 9173、env 3，每阶段 256）：所有样本保持真实 reachable-union 传播、二维 recovery proof、gap-in-coverage、相机/45 s/192-cylinder 边界。Decision 的候选带数实际落在 Sanity `1`、Core `1–2`、Choice `2–3`、Full `2–5`；Choice 都有至少一个 true choice，Full 都有至少两个 effective。首带中心固定至少距 spawn `2 m`，但 `.03–.12 m` ε 只服务响应时间：`free=max(speed*κ*(delay+d_max/v_lat_eff)+ε,1.53)`。带数上界以每带最多 `2.78 m` 的相机 free distance 加完整 envelope 保守计入 45 s，不依赖低估首带或第二条 mandatory 的距离。没有重抽 speed、截断 margin、放宽物理或相机参数，也未训练或重拍截图。当前统计见 `outputs/avoid_clutter_effective_decisions/summary.json`。

课程统计仅当前阶段的 decision episodes；阶段切换时清空统计窗口，只报告 traversal success 与包含 fall 的 safety failure（不只接触）。经 `hex_ground` clutter 分支核对，升阶要求完整 200 条 history、success `>=.85`、failure `<=.10`。阶段 1–3 的 nominal speed 固定；Full 每 2–4 s 重采目标 `[.15, 当前 episode nominal]`，并以 `.75 s` 过渡。本表是代码已核对的课程说明，不推测未核对分支。

日志语义固定如下：`Avoid/StageWindowCollisionRate` 是课程用的 safety failure，包含 physical、strict-envelope 与 fall；`Diag/EpisodeCollisionRate` 只记录 physical 或 strict-envelope，不包含 fall。clutter 的当前阶段 decision history 同步发布 completed、exposure、progress、success 与 row-success；trainer 不再读取初始化残留值。首个 Sanity 已从修正前的隔离源码快照启动，故只把它作为有限训练运行记录，eligibility 一律使用后续固定至少 200 decision episodes 的独立评测，不借旧 TensorBoard 字段判断。

## 4. Avoid reward

### 2026-09-11 Sanity clarification and Central counterfactual

下一次 Sanity 相对最初 Sanity 有两项已批准变化：relative forward-clearance reward 与仅在**步骤起始时**为 Stage-1 环境关闭 preference tie-break reward。后者不改变 `goal` observation、虚拟 preference 采样、动作与全局 preference coefficient `.0005`，Stage-2 及以后继续保留该项。该 gate 读取低层推进/reset 前的 stage snapshot，不能使用 reset 后的新回合 stage；因此本次不能表述为严格单因素对照。

独立资格的 45 条 Central layout 将只进行一次 `u_x=0` 反事实：以原 layout、原 reset、nominal forward request、post-processor 和低层权重运行，记录 policy raw、forced/raw/executed `u_x`，以及首次 terminal 前 root/dof/contact-force/link position 证据。原始资格行未保存完整 root/dof/RNG/低层内部状态，因此该运行不宣称逐位复现历史状态；它不改 PPO、entropy、安全缩放、几何或课程，也不新增 policy 组。

以下为已批准的 reward 结构。C1 已以 pure-Torch helper 实现默认 terminal-first 算式并完成 CPU 检查；静态 Isaac 场景不执行 reward，动态 wrapper smoke 尚未完成，且未在 PPO 训练、评测或部署中运行。每步定义为：

```text
r_t = -R_fail,                                                     safety failure
    = +R_succ,                                                     success
    = k_p Δp + r_clear - c_lat - c_smooth + r_pref,               otherwise
```

failure 优先。failure 与 success 都是终止步 outcome，覆盖该步 dense 项：不再叠加 progress、clearance、lateral、smoothness 或 preference；success bonus 只给一次。总共只保留以下六组，不新增其他项：

1. terminal outcome：`-R_fail` 用于碰撞或已冻结安全包络违规；`+R_succ` 用于 success；
2. progress：`k_p Δp`，前进为正、后退为负；
3. clearance：`r_clear=.03*(dt/.1)*rho_0*clamp(rho_0-rho_u,-1,1)`，其中 `rho(c)=clip((.57-c)/.57,0,1)`；`rho_0` 来自步骤前同一 visible map 上的 nominal forward request，`rho_u` 来自同一 map 上的 raw request。直行安全时该项严格为零；直行危险时，横移改善为正、等风险为零、恶化为负。不使用 post-processor preview/executed direction 或 GT gap；
4. lateral magnitude：`c_lat = k_l |raw u_A,x| >= 0`；
5. executed smoothness：`c_smooth = k_Δ |exec u_x(t)-exec u_x(t-1)| >= 0`；
6. preference tie-break：`r_pref` 仍仅在确需绕障、两侧安全且相近、preference 有效且有正向进度时，以 post-processor 后的 `u_exec,x` 计。

timeout 与 escape 是独立事件，不等同 collision 或 success；本轮不为 escape 增加额外罚分，也不把它们写作已实现的 `terminated/truncated`。后续 bootstrap 必须按任务终止与时间截断分别冻结，不能用单一 `done` 处理替代。

当前计划默认值为高层 `dt=.1`：failure `-20`、success `+2`、progress `Δp`、relative clear `.03*rho_0*clip(rho_0-rho_u,-1,1)`、lateral `.005|u_x|`、smooth `.01|Δu_exec,x|`、preference `.0005*g*s*u_exec,x/.6`；除 smooth 外按 `dt/.1` 缩放。以 `gamma=.99,T=500`，折扣和约 `99.343`，最大前进 dense 项约 `4.967`，clear/lateral/preference 的量级上界分别约 `2.980/.298/.050`。这是设计核算，不是经验回报；失败项同样折扣，有限 `-20` 不能声称保证安全。

`cross_line_distance` 只可留在环境内部，不进 actor。旧 `row_lat/row_gap/row_cmdx/early_next_gap/...` 退出新 Avoid 总 reward；PCR/Gate reward 不随本基线改动。

## 5. Canonical map：当前事实、批准方向与待定项

批准方向是“优先保留当前 sim actor map 的形状、xy raster 与 soft passability 公式”，并以**同一公共函数**由可观测 occupancy 建立 actor map 和 difficulty。它不等于已经完成 sim-real 一致性，也不要求把深度点云伪称为 GT bbox raster。验收应使用同一 occupancy/visibility fixture，使两路径输出 map/D 相等；不声称真实深度渲染逐像素等同 GT。

| 项目 | 当前 sim 事实 | 当前实机事实 | 新基线状态 |
| --- | --- | --- | --- |
| 尺寸、域与轴 | 32×32、3 m、`ij` 格点中心、`[x_right,y_forward]` | 默认 32×32、3 m，`ix=x_right,iy=y_forward` | 保留，已批准方向 |
| channel 0 | scene bbox 生成的 occupancy | 深度点/高度带生成的二值 occupancy | 统一为观测 occupancy 输入；采样来源不同 |
| channel 1 | `c=max(avgpool3x3(p),p)*(1-occ)`；仅膨胀边缘为软值，不是米制欧氏距离 | immediate raw occupancy 经共享函数生成 soft passability，FOV 外 `[0,0]` | shared canonical helper 已接入 builder/runtime |
| visibility / unknown | actor map 有 visible mask；FOV 外 `[0,0]` 对旧 D 有代价 | dense intrinsics 投影 mask；passable 乘 mask | FOV/unknown 语义必须统一；mask 先后位置待定 |
| inflation | `.27 m`、`.09375 m` cell、`ceil=3` 的 7×7 方形 max-pool；full-GT 先膨胀再 FOV mask，可能影响可见边缘 | `.27 m` 默认、`.09375 m` cell、`ceil=3` 的 7×7方核 | 从 observed occupancy 生成 actor图；膨胀顺序待定 |
| 相机/原点 | `camera_mount=[0,.22,.08]` | map offset 可配，默认 `.30 m`；实际入口默认 `.23 m` height、0° pitch | 数值不等同历史实机值，最终 origin/offset 待定 |
| FOV | 当前 config 对齐 D435 设定，但 visibility 是几何可见掩码 | runtime intrinsics + image bounds + depth range；未硬编码固定 FOV | 不把官方 FOV 写为已实装；待统一 |
| actor obstacle memory | actor map 是即时 map（默认 frame stack 1） | 默认关闭；memory/inflation 只作 debug，不进入 policy map/D | **已实现为关闭 real-only memory** |
| risk memory | 独立 risk/memory 语义 | pcr_realplay risk_memory 只供 learned-w row feature | 保持独立，已批准方向 |
| difficulty | `D=.5 mean_R(occ)+.5 mean_R(1-c)`，`R` 默认 2 m，分母不乘 visible mask；FOV 外 `[0,0]` 贡献 `.5` | builder 与 revised runtime 调用 shared canonical D；front/near-field 仅 debug | shared function 已接入；未做真实传感/Isaac 验收 |

sim 来源固定为当前 HEAD `6c491b0`：camera/extent 配置在 `legged_gym/envs/hex_v4/hex_ground_config.py:23-44,81,171`；Avoid map 建图在 `legged_gym/scripts/train_highlevel.py:583-635`，camera/map geometry 在 `:2230-2331`，scene bbox raster 在 `:2868-2930`，膨胀在 `:3020-3030`，visible mask 在 `:3093-3133`，difficulty 在 `:3147-3180`，actor/critic map 选择与 difficulty 传入在 `:4093-4113,5764-5778`，teacher Avoid map 选择在 `:6854-6880`。`:3147-3180` 的旧 `fullGT` docstring 不能当作 actor 事实：实际 actor 用 visible local map。上述 sim D 是当前事实，不是新公共函数已经落地的声明。

当前实机 builder 为 `src_real/interface/scripts/pcr_real/real_pcr_input_check.py:369-639`，实际 ROS launch 入口见 `src_real/interface/launch/pcr_real_compat.launch:49-123`。两份 `pcr_realplay.py` 内容相同；两个 input-check 副本只在默认 pitch/height 不同，实际 launch 用 `src_real` 副本，故不得由默认值反推历史 trial 的启动参数。

实机 builder 的可定位事实：camera optical `(+X right,+Y down,+Z forward)` 转 robot `(+X right,+Y forward)` 在 `real_pcr_input_check.py:344-366`；point raster 在 `:418-461`；FOV mask 为实时 intrinsics、image bounds、`min/max_depth` 的投影，见 `:741-768`；clearance override/几何 fallback 为 `:771-783`；默认参数见 `:1638-1677`。`pcr_realplay.py:515-527` 接收 `(2,32,32)` map，`:768-782` 优先接收 builder 传来的 difficulty，否则使用不一致 fallback，因此当前尚不满足“同一函数”。

## 6. Difficulty 与实机记忆 Findings

1. **soft/map处理**：实机 builder 已用 shared helper 从 raw observed occupancy 与 fresh visibility 建立 actor soft map；debug 的 inflated/memory map 不再进入 actor。
2. **可见性与记忆**：real-only obstacle memory 默认关闭；即使显式调试开启也不影响 canonical policy map/D。`pcr_realplay.py` 的 learned-w risk memory 仍独立，revised path 的 risk query 使用 canonical raw occupancy。
3. **D 来源**：builder 与 revised runtime 调用 shared canonical D；front/near-field 仍输出诊断，但不再与 actor D 取 max。CPU fixture 覆盖 map/D；尚未验证真实传感器、真实 checkpoint 或 Isaac。

### Patterns

- **同 shape、语义不同**：sim/real 都可为 `(2,32,32)`，但当前 channel-1、unknown、膨胀/visibility 时序、D 与 memory 并不相同；尺寸一致不能证明部署一致。

### AGENTS Update Proposal

- 历史状态：此前为“不新增”。该旧批状态已被本次用户显式长期记录取代；`AGENTS.md` 已新增仅适用于本项目 revise 阶段的决策规则，不把未实现方向写为既有事实。

### TODO Suggestion

- 按已冻结方向实现同语义即时 canonical map，关闭 real-only obstacle-map memory，并保留 PCR risk memory；不为低风险猜测新增 dropout 小测试或横向扩展。相机实测 offset 等未知不伪造，也不新增测试要求。

## 7. Release gates 与下一批最小范围

1. 已冻结：实现同语义即时 canonical map/D，关闭 real-only actor obstacle-map memory，并保留 PCR learned-w risk memory；不新增 dropout 小测试或再次分支讨论。相机 origin/offset 实测值未知，不伪造，也不因此新增测试要求。
2. 批 A 已以单一障碍带的 reachable-interval 并集、二维 envelope 与局部反应时间筛选候选；批 B 另有真实 Isaac 静态实例化和 actor-layout trace。仍须短闭环标定 `W_eff/v_lat_eff/τ_response`。当前 `.15 m/s/.30 s` 仅为保守工程筛选配置，不能作能力或不可绕保证。
3. 然后在同一 generator 的 Sanity 基本情形中核对 `Left-only→left`、`Right-only→right`、`Central/Free→不横移` 与反 preference；这不是独立模板或额外训练实验。诊断预算、窗口和阈值仍待冻结，“训练尚未启动”不是失败或通过证据。
4. 已在 Isaac Gym 实际实例化不同阶段的新 generator 并生成最新四张原生截图；live actor 与 runtime metadata 完全一致。截图的红线是几何 nominal reference、不是策略轨迹，约 5 mm 的中心路线余隙也不代表全腿包络安全。正式训练仍须用户明确批准这些截图，不能以概念示意图或旧 preview 替代。
5. 历史本地 4-env CPU-PhysX wrapper smoke 已构造环境、加载冻结低层权重、完成 reset 并进入首个高层调用，但因本地 Torch 1.8 的 Tensor `clamp` 兼容错误完成 0 个高层步。用户明确授权后，fresh server bundle 的 4-env/24-step CPU-PhysX smoke 已通过 finite/shape/权重 SHA 检查，且不改服务器原仓库；它只解除该有限动态闭环可运行性的阻塞。正式 1-D PPO 的 `Core/Choice/Full` 仍须取得前述用户训练批准，本节设计冻结不代表性能或安全通过。

下一步是只读地在原训练环境核对该有限动态 smoke；在用户明确批准前，不启动 PPO、SSH、commit 或 push。本轮静态场景、CPU 检查与本地 blocked smoke 均不构成训练放行。
