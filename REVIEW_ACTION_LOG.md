# 审稿行动日志（内部工作稿）

> 本日志记录作者决策与后续核对边界；不记录为已完成实验或历史事实。

## 2026-09-10 Avoid overnight continuation blocked before GPU2 smoke

- Sanity continues under its original server watchdog with the unchanged 201-iteration budget. The reviewed continuation controller is unarmed; no new formal process was spawned.
- The fresh `21,525,518`-byte snapshot (SHA `5155f082…02f45`, source manifest `b1144ccb…cd303`) for `/home/dell/Pcrnet_hexapod_runs/avoid_overnight_20260910_gpu2_smoke` was rejected twice by automatic transfer review. GPU2 2-env/2-step smoke, qualification and formal training therefore did not run. Exact evidence is `outputs/avoid_sanity_20260910/deployment_blocked.json`.
- The user has authorized the training workflow. The current stop is only the automatic review of this specific external payload/destination; do not retry it unless the user explicitly names and approves both.

## 2026-09-10 Avoid clutter 日志语义修正（CPU contract 通过；当前 Sanity 快照不变）

- `hex_ground` 的 clutter decision-only 分支此前只写入 collision/success history，并遗漏 trainer/TensorBoard 消费的当前阶段 completed、exposure、progress 与 row-success 字段；修正后只对 `decision && stage==current_stage` 写入五项已有 history，并发布同一当前阶段的 extras。课程升阶仍只看完整窗口的 success/failure，reward、终止与课程条件不变。
- dedicated clutter high-level info 现显式返回 `s_avoid_episode_collision=terminal_physical|terminal_envelope`。因此 `Diag/EpisodeCollisionRate` 表示 physical/envelope，`Avoid/StageWindowCollisionRate` 表示 curriculum safety failure（physical/envelope/fall）；两者不能互相替代。后续独立评测将分别保存 physical、envelope 与 fall，不从两条旧标量反推失败类型。
- `test_clutter_env_contract.py` 覆盖 current-stage decision filter、五项 history、阶段切换后的清窗/计数；`test_avoid_clutter_training.py` 覆盖 physical/envelope alias 与仅 fall failure 的区分。二者在既有 `grasp_hexapod` Python 环境通过。正在运行的 Sanity 使用修正前的隔离快照，故其旧日志不用于 eligibility gate；该问题不影响已发生的梯度、reward 或终止。

## 2026-09-10 Avoid clutter server 动态 smoke 通过；未训练

- 用户明确授权后，将 SHA `e9f1ccb6…` 的独立 bundle 传至 `pcrnet-server:/tmp/pcr_constructive_smoke_20260910_9s0AUs`，仅在该新临时目录运行既有、未修改的 4-env/24-step CPU-PhysX smoke。fresh import 路径均指向该临时 bundle；远端实际低层 checkpoint SHA `75dfd7aa…` 和 smoke script SHA `3c61ff27…` 与本机一致。
- 结果为 `state=[4,14]`、`action=[4,1]`、`command=[4,3]`、24 high-level steps 全程 finite、`done_total=0`。actor 是随机初始化的 `CmdVelExpert`，因此这只证明有限动态 wrapper 闭环可运行，不是策略性能、训练、评测、sim-real 或安全结果。没有写服务器原仓库、安装包或启动训练。
- 原始远端 log/result、fresh import 路径和完整 provenance 在 `outputs/avoid_clutter_constructive_acceptance_20260910/smoke/`。此前自动审批拒绝、尚未传输的记录保留在同目录的 `remote_transfer_blocked.json` 作为历史；随后用户明确授权的传输已实际完成，不能继续标作“未启动”。

## 2026-09-10 Avoid clutter 最新静态预览通过；远端 smoke 未启动

- 本机 `grasp_hexapod` 环境以 CPU PhysX、seed 9173 和四环境原生 viewer 重建了 Sanity/Core/Choice/Full 的静态预览；每个 active cylinder 的 live actor position 与 runtime metadata 完全相等（最大误差 `0.0 m`）。四张截图、metadata、启动口径和源码 SHA 位于 `outputs/avoid_clutter_constructive_acceptance_20260910/preview/`；红线仅为 nominal geometry/dynamic reference，不是策略轨迹或性能证据。
- 远端 4-env、24 high-level-step CPU-PhysX smoke **未启动**。自动审批拒绝将准备好的 `/tmp/pcr_constructive_smoke_20260910_9s0AUs/bundle.tar.gz`（`21,646,544` bytes，SHA `e9f1ccb6…`）传给 `pcrnet-server:/tmp/pcr_constructive_smoke_20260910_9s0AUs`，因为 bundle 包含项目源码、resources 和 low-level checkpoint。没有传输、远端目录写入、安装、训练或远端运行；明确内容、hash 与预定命令见 `outputs/avoid_clutter_constructive_acceptance_20260910/smoke/remote_transfer_blocked.json`。须由用户明确授权该具体传输后才可继续。
- 历史本地 Torch 1.8 wrapper smoke 的 `Tensor.clamp` 阻塞（0 high-level steps）仍是独立事实，不能与这次“未启动”的 server smoke 混写。

## 2026-09-10 Avoid clutter constructive（CPU 几何完成；不训练）

- 每带保持 `R` 的可达区间并集，记录 `d_min=dist(R,G)` 与既有局部时间预算使用的 `d_max`；`mandatory` 仅由 `d_min>=.10 m` 判定，`effective=mandatory OR choice`，同带双条件只计一次，不把候选带数当作决策数。
- `choice` 仅接受同一前序点可到两个不同内缩 gap 的预算交集见证；两条二维进/出带路线加入 proof，并在每次新增圆柱后对所有历史 proof 复检。不同前序分支各通一个出口不是 choice；该候选不代表每个实际策略状态均有双出口。
- Sanity/Core 的 decision layout 至少含一条 mandatory，Choice 的最后一带至少含一条 true choice，Full 前两带为 `.10 m` mandatory 以提供至少两条 effective；Free/Central 保持各 10% 且不计决策。候选带数在 speed/κ/45 s 可行集合内抽取，保留阶段范围而不重抽 speed 或截断 margin。
- 固定 seed 9173、env 3 的 1024 layout CPU 几何检查通过：检查实际 reachable propagation、每个 gap 的 coverage、相机、45 s、192 cylinders、二维历史 recovery proof、真实 stage band-count 与 effective 门槛。`10 m` 是 resource cap，不是 anti-bypass 证明，也未增加远端终止、reward 或居中项；未运行 Isaac、训练、SSH、提交或推送。

## 2026-09-10 Avoid clutter 非开口间距随机化（CPU 几何完成；未训练）

- 仅 `s_avoid_clutter` generator 的非开口 chain 内部点加入有界独立 `x` 偏移；chain 两端、gap 边界、piece 数 `ceil(length/.60)`、既有 `y`-jitter、二维 gap/线段圆校验不变。偏移同时受真实圆柱不重叠和 expanded-circle 上界约束，因此不新增 gap 外通道；短段只有一个 interval 时保持固定。
- geometry CPU 回归覆盖固定 seed 复现、不同 seed 的多段 `x` 间距变化、端点精确不动及二维上下界；既有 1024 固定样本检查仍执行。未修改课程范围、奖励、模型、训练参数或 `hex_ground`，未训练、SSH、截图、提交或推送。
- 课程表同步写明 `side offset` 仅是单开口生成偏移而非实际 `Δx` 保证、safe/physical gap 宽度、采样比例和双 gap 条件候选；升阶与 Full 速度上下文只记录已核对的 `hex_ground` clutter 分支事实。旧截图出自旧间距，未重拍且不作为本次最新证据。

## 2026-09-10 PCR revise 长期冻结（C1/D2 代码完成；未训练）

- 本项目 revise 阶段固定一次选择性能、正确性、sim-real 一致性和返修说服力综合最优的可实施方案；仅验证影响核心架构、训练成立或论文结论的高风险不确定性。低概率低影响猜测用合理工程默认，不新增小测试或横向扩审计/消融/场景/参数拖延开训；不跳过必要实现正确性检查，也不编造结果。
- 地图最终冻结为 sim/PCR/real 同语义即时 canonical local map；real-only obstacle-map memory 关闭，PCR learned-w risk memory 保留。D2 已将 shared map/D 接入 real builder/runtime，CPU fixture 通过；无 dropout 小测试或再分支讨论。相机实测 offset 等未知不伪造、也不新增测试要求。真实 Isaac 静态场景已核对；未训练、未运行 ROS/硬件。

## 2026-09-10 最新优先声明：Avoid 1-D 基线（C1/D2 代码完成，未训练）

- 正式 Avoid 已采用 1-D lateral actor 与 `v_drive_nom` 的统一语义；C1/D2 完成代码、pure-CPU contract 和 real dry-run fixture，不训练、不 SSH、不提交或推送。
- 该方向回应 R1-16 的“训练 3-DoF 后丢两维”问题：将来 PPO 的 action distribution、log-prob、entropy、rollout buffer 与 `evaluate_actions` 均须为 1-D，而不是训练 3-D 后在 PCR 中丢弃 forward/yaw。尚无新训练、消融或性能结果。
- canonical map 的批准目标是优先保留当前仿真 32×32、3 m、`[x_right,y_forward]` 与 soft passability 形状，并由同一函数从观测 occupancy 构造 actor 图和 difficulty。real-only actor obstacle-map memory 已冻结为关闭，PCR risk memory 独立保留；相机 origin/offset 实测值仍未知，不得伪造或要求新增测试；不得写作现已对齐。
- 本轮实机与仿真 map 来源取证、静态 Isaac 场景核对已完成；动态 wrapper smoke 受本地 Torch 1.8 阻塞，均不构成训练或实机结果。

详见 [Avoid 1-D baseline](docs/specs/AVOID_1D_BASELINE_20260910.md)。历史 14-D/3-DoF 记录仅为历史，不能覆盖本节。

## 2026-09-10 Avoid 1-D reward 结构冻结（C1 实现完成，未训练）

- failure 优先的 terminal outcome 与六组 reward 已在 C1 接入 dedicated Avoid loop 并经 CPU fixture 检查；旧 row-specific 项退出新 Avoid；PCR/Gate reward 不改。未训练、未作 Isaac 验收。
- timeout/escape 不并入 collision/success，且本轮不新增 escape 罚分或声称 `terminated/truncated` 与 bootstrap 已实现。系数仍待按 `dt/T/gamma` 的折扣累计核算；示例步数、系数和 gamma 不是最终 config，不新增 sweep、训练或测试。

## 2026-09-10 Avoid 单一障碍带与课程冻结（实现与静态核对完成；未训练）

- `s_avoid_clutter` 采用单一圆柱障碍带 generator、少量 `y`-jitter 与每带 1–2 个真实 gap；机器人中心可达区按 safe intervals 的并集传播，保留未来不可见前的两个合法 gap 分支，并以二维 envelope 排除 gap 外斜向通道。局部反应时间、出生缓冲与执行速度假设是筛选边界，不构成已验证的不可绕保证。
- 课程固定为 `Sanity/Core/Choice/Full`；`Free/Central` 是分布样本，Sanity 的四种基本情形仍来自同一 generator。只统计 decision episodes，并在切换阶段清窗口、报告 traversal success 与 safety failure；范围、窗口与旧 `85%/10%` 均为待标定工程候选。generator、actor pool 与静态 Isaac 实例化已完成；未训练或动态能力验证，旧 PCR 布局及 §4 reward 不改。
- 已用真实 `s_avoid_clutter` 生成四张不同阶段的原生 Isaac 静态截图，并逐布局核对 actor 位置与 metadata。红线是 nominal 几何参考而非策略轨迹；约 5 mm 的中心路线余隙不代表全腿包络安全。“用户截图批准仍是训练门槛”是当时记录；用户随后明确授权隔离过夜 Sanity、资格评测和 gated continuation，不能继续作为 SSH/PPO 的当前禁止项。

## 2026-09-10 Avoid 1-D 批 A（未接训练环境）

- `legged_gym/pcr_observation.py` 新增 pure-Torch canonical observed map/difficulty、14-D Avoid actor state 与 side-preference helper；channel 0 保持 raw observed occupancy，soft channel 单独按 pooled free 公式处理，unknown 保持 `[0,0]`。CmdVelExpert 新增显式 `action_dim=1` 与 actor-only mask，GatePolicy 新增显式 actor `xy` mask；新 mask 模式默认调用也保留 raw critic，旧 3-D 默认与参数 key 不变。
- 本批未修改 CNN 结构、PPO、默认权重 scale、训练环境或实机调用；仅将既有 CNN 的 `meshgrid` 写法换为等价默认 `ij` 以兼容当前 CPU Torch。后续调用必须显式选择新 1-D/mask 配置。CPU 契约测试已覆盖并通过 map、difficulty、actor/critic mask 与 1-D 分布 shape；未训练、SSH 或提交。

## 2026-09-10 Avoid 1-D 批 B（场景与课程；未训练）

- 批 A 的 deterministic NumPy generator 已完成；其 1024 个固定 CPU 样本及独立几何检查通过。批 B 的真圆柱 URDF actor-pool、reset、实际 layout 指标、终止/课程已在真实 Isaac 静态场景中实例化；这不是动态训练验收。
- 旧 `s_avoid_basic` 预设、资产与路径未改。候选 generator 使用 `r=.15 m,h=.50 m` 圆柱、safe/reachable interval 并集、二维线段-圆 envelope、局部反应时间与 50 s 名义可完成性；`.15 m/s/.30 s` 仅为筛选配置，不构成物理能力或不可绕证明。训练、评测与部署未执行。

## 2026-09-10 Avoid 1-D C1（CPU 实现；未训练）

- C1 为 standalone `s_avoid_clutter` 增加 1-D lateral actor 的训练入口、14-D actor mask/原始 critic state、canonical observed map/difficulty，以及 pure-Torch terminal-first six-group reward helper。训练请求由 1-D lateral action 展开为 `[x, v_drive_nom, 0]`，不改旧任务路径。
- 指定 Conda 下的语法和 CPU reward/state/action 夹具通过，真实 Isaac 静态截图已完成。4-env wrapper 的动态 smoke 完成 reset、但在首个高层 step 的本地 Torch 1.8 Tensor `clamp` 兼容错误处停止（完成 0 个高层步）；未运行 PPO、SSH、评测或部署。实际闭环和开训仍受该运行环境与用户截图门槛阻断。

## 2026-09-10 静态预览出生缓冲核对

- `s_avoid_basic` 的真实出生函数返回 `-1.6 m`，reset 也使用同值；本次未见实际不安全实例，故不改训练源码。`s_pcr_line_avoid_basic` 继承该几何，仅增加移动目标；Mono、Rule、Additive、DWA 是同一 task 的动作分支。本结论不外推为所有历史场景已完成安全核验。
- 独立 Isaac 设计预览已将旧障碍/路线前移 `1.6 m`、保留机器人 `y=.45 m` 并加直行入口，产物单独写入 `outputs/avoid_clutter_design_preview/spawn_fixed/`。它仍不是训练 generator、四类目的场景集合、策略输出或性能/安全证据。

## 2026-09-08 Actor x/y 输入诊断（旧 checkpoint rollout 已完成；非正式实验）

- 新增 `tools/audit_actor_xy.py`，仅在已有 eval 的 `CmdVelExpert.get_action` 与 `GatePolicy.get_action` 处旁路记录输入；live action 先原样执行并返回，所有额外输出均为 no-grad deterministic 副本，不送入环境、风险更新或下一步控制。
- 每十次 action batch 最多保存 Avoid/Gate 各 512 帧；只干预 actor state 的 `x/y`（`Δx=±0.25 m`、`Δy=±0.50 m`、absolute `y=0/2/4/6 m`），保存 map/state/goal/difficulty/动作和脚本/Gate/Avoid/low-level checkpoint/eval-source SHA，并写出采样 state x/y 的实际 min/max。已有 critic state 保持不变；未传 critic state 时副本显式固定为原 actor state，输入变异或非有限值会 fail-fast。非 self-test 保证 Isaac Gym 先于 Torch 导入。
- 本地 `py_compile` 与 Torch self-test 已通过。随后在旧 eval `29ee797` 上完成一次 `s_pcr_line_avoid_basic`、`skill=moe` 的冻结 checkpoint rollout（seed 1、8 env/8 episodes、difficulty 0、stage-4 freeze、0.35、GPU2）；Avoid/Gate 各 200 帧、251 calls，state `x∈[-0.492,0.438]`、`y∈[-1.610,6.751]`。live action 未被诊断副本替换，未写旧实验目录、未启动训练。
- 四个主干扰结果：Avoid `|Δu_x|` mean/P95 为 `x− .08753/.15720`、`x+ .08684/.16024`、`y− .00838/.02176`、`y+ .00782/.02040`；有效横移符号 flip 为 `5.67%/7.73%/0/0`（194 帧分母）。Gate 的 `Δy`/`Δw` 最大 P95 为 `.00837/.00491`。absolute y=0/2/4/6 的 Avoid mean/P95=`.04210/.13841,.03615/.10814,.03649/.12402,.04865/.15365`，有效 flip=`3.09%,1.03%,1.03%,3.09%`；Gate `Δy`=`.01370/.05655,.01144/.04287,.01183/.03763,.01878/.05963`，`Δw`=`.01576/.03895,.01322/.02984,.01545/.02755,.02560/.04698`。绝对 y 扫描只用来发现可能的位置相关性，不得称为背地图证明。
- 这些是后处理之前的 actor action 坐标（Avoid 为 `tanh(cmd_raw)*cmd_scale`，Gate 为 Beta 输出），不标作 m/s；仅支持此单速度/布局 rollout 中 Gate 对小 x/y 扰动较弱、Avoid 对 x 明显且对 `±0.5 y` 较弱，不能支持删 x/y、背地图、性能或安全结论。原始 npz/json 位于 `/home/dell/RL_hexapod_gym_revision_geometry_20260908/outputs/actor_xy_old29ee_293c518_retry4/`（eval SHA `29ee797…`、script SHA `4d49fa36…`）。
- 启动曾因日志未重定向、旧类延迟加载、未激活既有 Conda 的 PATH，以及误传 `skill` 先后失败，均未产生有效 rollout。改用已有 `isaac_gym` 环境并修正 metadata 参数后 `EXIT_CODE=0`；未安装依赖，未修改旧 root 或 GPU1 的 `505b937` Avoid。
- 同轮实机历史核对的修正：`pcr_realplay.py` 缺 ROS state 时补足维度的全零 state；`risk_memory_velocity_source=body` 因维度仍至少五而读零前向速度，非自动回退 `cmd_F`。file bridge 也无条件使用全零 state。历史 40 trial 的原始逐次标签/启动命令仍未找到，只能记为待证据核对。
- Table I 的历史 `.60 m/s` Learned-w 三 seed collision 原始计数为 `3/3/9`，即 `15/384=0.0390625`。历史 `a8429b4` 前已有 strict hull-clearance（margin `.01`）与 `s_avoid_episode_collision` 的 OR；但当前 metrics 未保存精确源码 SHA，不能将其拆成 physical/envelope，也不回填或改旧表名。仿真独立/PCR Avoid 使用 cross-line `goal_raw`，而实机 Avoid 使用 target-relative goal；该输入差异待冻结。
- 后续 Gate 正式开训只等待已冻结的训练输入、奖励/终止、核心闭环与修正服务器 Isaac smoke；Avoid、Mono、Gate 依赖分别放行，本轮不改算法。

## 2026-09-08 当前确定性修正状态（优先于下方历史运行记录）

- 已完成且仅完成两项代码修正：PCR/real Follow 的 body→world 逆变换，以及 scene affordance 对完全越界 bbox 的边界压缩；部分相交按地图物理边界裁剪，原有在界内量化不变。
- 本地 CPU 语法、真实 helper→真实 Follow world→body 往返、static 与 `s_avoid` box 的 camera-mount 栅格回归均通过；旧源码 `/tmp/pcrnet_train_highlevel_before_geometry_fix.py` 分别在 Follow 往返和 front-outside static bbox 检查失败。代码提交 `781e830` 与记录 `eefa8cf` 已推送 GitHub；通过一次性 SSH 反向代理成功拉取，服务器 detached root `/home/dell/RL_hexapod_gym_revision_geometry_20260908` 位于 `eefa8cf`，八个批准文件 hash 与本地一致；未修改永久代理或既有 worktree，服务器原生 CPU、Isaac、训练、评测和实机均未执行。
- 服务器 `505b937` Avoid 的既有运行保留，不停止、不覆盖，也不认定为正式返修 checkpoint。sim-real policy contract 本批未修改、未冻结。

## 2026-09-08 最新执行状态（Avoid 首轮已过；运行中）

- 作者明确批准最小独立 Avoid 训练：训练保留第 14 维 forced-forward speed 用于促学习；融合调用 Avoid 时该第 14 维继续补 `0`，前进与 yaw 仍由 Follow 提供。
- Avoid14D 反事实诊断撤出开训前置；其本地草稿未运行、未测试、未提交。
- 已执行参数：`task=s_avoid_basic --mode teacher --skill avoid --seed 42 --num_envs 512 --num_steps 24 --num_epochs 2 --mini_batch_size 4096 --lr 1e-5 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.05 --value_loss_coef 0.5 --entropy_coef 0.04 --max_grad_norm 0.5 --cmd_slew_lin 0.2 --cmd_slew_ang 0.4 --aff_stack 1 --decimation 5 --num_iterations 1000 --save_interval 50`；`--low_level_ckpt /home/dell/RL_hexapod_gym/logs/hex_ground/Dec31_16-52-59_/model_6000.pt`，不带 `--resume`、`--finetune_from`、`--force_cmd_y` 或 `--generalize`。仅 `1000` 是本轮预算、非历史已确认值；其余由旧 `run_meta` 复原。
- Avoid 已于 `20:29:34` 在服务器 tmux `pcr_revision_avoid_505b937` 启动：root `/home/dell/RL_hexapod_gym_revision_20260908`，代码 `505b937b7687ba87afd34d5631163b84a139ebdd`，PID `2818920`，输出 `outputs/revision_avoid_505b937/train.log`，GPU1 的 `CUDA_VISIBLE_DEVICES=1` 对应进程可见 `cuda:0`。已进入训练循环并完成首轮 PPO 日志：value/policy/entropy=`0.3073/0.0285/1.0838`，nonfinite skip/sanitize=`0/0`、action=`0/0/0`、stage=1。completed episodes=0，success mean 为 NaN 的空集合，不能称数值故障；checkpoint 保存仍未确认。
- Mono 的训练设置尚未讨论；用户明确要求不启动、不自动排队。

## 2026-09-08 第一批确定性几何修正（已完成；CPU 验证通过）

- 范围仅为两项已证实的实现修正：Follow 调用链的 body→world 逆旋转，以及 scene affordance 对完全位于 local-map 外 bbox 的边界压缩。
- Follow expert 内部的 world→body、`atan2(x_right, y_forward)`、`+Y forward`、cone、FOV、风险阈值、奖励、课程、网络与 sim-real 输入口径均不在本批范围。
- scene raster 保留既有边界内格点量化；只先判 bbox 与 `[-extent/2, extent/2]×[0, extent]` 是否有正面积交集，完全无交集跳过，部分相交按物理边界裁剪。
- 不启动、停止或改写现有 Avoid 训练；本批完成后由主线程验收，再决定 GitHub 同步、Isaac 验证和重训范围。

## 2026-09-08 代码同步与原生 CPU 核对（完成；未训练）

- 本地代码提交为 `c4353cc`（`Fix axis semantics and restore Avoid state`），分支 `codex/revision-axis-fix-20260908`，仅含获批的五个代码路径；`.vscode`、`TODO_LOG.md`、`REVIEW_ACTION_LOG.md` 与 `REVIEWER_RESPONSE.md` 未暂存、不会随 code-only 同步。
- 本地语法与 CPU 测试已完成：速度课程、Avoid 14D 与轴序测试均通过；轴序本地 torch 1.8 使用兼容层。未训练、未评测、未加载 checkpoint。
- 用户明确批准后，`c4353cc` 已推送到该 GitHub 分支；`git ls-remote` 远端 SHA 为同一提交。服务器以单次无 proxy fetch 得到同一 `FETCH_HEAD`，并建立 detached root `/home/dell/RL_hexapod_gym_revision_20260908`。
- 新 root 的 `/home/dell/miniconda3/envs/isaac_gym/bin/python`（torch `1.12.1+cu113`）已通过 `py_compile`、轴序、速度课程与 Avoid 14D 三项 CPU 测试；轴序输出 `meshgrid=native`。`import isaacgym` 在 `torch` 前执行，且 `legged_gym.__file__`、`rsl_rl.__file__` 均指向新 root。未加载 checkpoint、未启动 PhysX、训练或评测。
- 原 `/home/dell/RL_hexapod_gym` 保持干净 `29ee797`；新 root 保持干净 `c4353cc`。训练尚未由本批启动。
- 作者已批准保留 Avoid stage-4 实际末行 `6.85` 的终点判定；本批仅同步和 CPU 核对，未据此启动训练。
- 用户随后要求先由 GPT 审阅代码，主动暂停 Avoid 开训；启动命令从未发送。启动前只读检查与主线程复核均显示没有 `pcr_revision_avoid_20260908` tmux、没有 `train_highlevel` 进程，且没有本轮训练输出或 checkpoint。

## 2026-09-08 已批准的本地最小修正（已完成；未训练）

- 范围仅为：保留现有 `train_highlevel.py` 两处 `ij` 修正；将 `s_pcr_new` 课程 L1/L2/L3 目标速度上限收至 `0.50 m/s`；补 camera-mount 与课程采样 CPU 测试；同步本日志与 TODO。
- 不改 `s_avoid_basic` 布局、goal、课程 seed/权重/stage 分配、机器人 forced-forward、generalize、25° 阈值、网络或奖励；不启动训练、评测、提交或推送。
- 已核实 server Git `49235d2` 的 depth camera 位置为 `[0.00,0.22,0.08]`。这支持本轮 camera-mount 测试参数，但不替代旧训练完整源码快照。
- 已将 `_sample_pcr_new_curriculum` 的 L1/L2/L3 目标速度上限由 `0.55/0.55/0.65` 收至 `0.50 m/s`；下限、课程 progress、权重、stage 与 environment/episode seed 公式均未改。`s_avoid_basic` 及其 forced-forward/generalize 路径未改。
- 已补充 camera-mount `[0.00,0.22,0.08]` 的 yaw `0`/`pi/2` 真实参考位姿轴序测试，并新增 AST 课程采样测试：同 seed 下原 level/stage 映射不变、覆盖 level 3、所有采样目标速度均不高于 `0.50 m/s`。Tier 0 `py_compile` 与 Tier 1 两项 CPU 测试均通过一次；未启动训练、评测、提交或推送。

## 2026-09-08 Avoid 状态输入契约（第二批已修正；未训练）

### 已执行

- 对当前 `train_highlevel.py` 只读确认：`_get_high_level_obs` 构造 9 维机器人状态，追加 `prev_gate_y` 与 `post_processor.last_cmd`（3 维），当前训练在创建 `CmdVelExpert` 前直接由该观测推导 `state_dim`。其中没有 `forced_forward_speed` 的状态拼接，也未发现其他补回该维度的路径。
- 当前 `match_state_dim` 仅在运行状态小于 checkpoint 目标维度时以零填充，适用于 checkpoint 兼容；它不改变从当前观测新建专家时的 `state_dim`。因此不能把该兼容行为视为新 Avoid 训练保持旧 14 维输入的实现。
- 历史源码 `49235d2` 的 forced-forward 采样上限为 `0.5/0.4`，与当前函数 AST 相同；陈旧 metadata 的 `0.8/0.6` 不构成回改依据。低层旧路径与 `agents/low_level_best.pt` SHA-256 同为 `75dfd7aae52b20f08ca9f654c37b2e20ba1e33f8b953737437ad774d122d1481`。
- 第二批已在 `_get_high_level_obs` 的 `last_cmd` 后恢复 `forced_forward_speed.detach().clone().unsqueeze(1)`，但严格限于 `args.skill == 'avoid' and env.s_avoid_enabled`。Avoid 由 9+1+3 恢复为 14 维；MoE/Gate、Mono（`skill=moe`）和 Follow 保持 13 维。未改 fallback、policy head、奖励或 forced-forward 采样。
- 新增 AST 前缀测试只执行 `_get_high_level_obs` 到 goal 前的状态构建段；fixture 明确只覆盖 9D `robot_state_buf`、`prev_gate_y`、`last_cmd` 和 forced-forward speed。它验证 Avoid 的第 14 列为实际速度且前 13 列不变，以及 Gate/Mono/非 Avoid/Follow 无扩列。Tier 0 `py_compile` 与该 CPU 测试通过一次。
- 轴序测试现先检测运行时是否支持 `torch.meshgrid(..., indexing=...)`：支持时直接使用原生 torch，不支持时才用旧兼容层。本地 torch 1.8 走兼容层并通过一次；服务器 torch 1.12 的原生分支尚未在本批执行。

### 待核实／未批准

- 主线程的服务器 AST 对照显示旧 Avoid 状态为 14 维、此前当前源为 13 维，差异候选为历史末尾的 `forced_forward_speed`；本轮已按该最小范围恢复。旧训练完整源码快照缺失，故仍不能把本次兼容修正外推为全部历史设置均已还原。
- 旧训练仍无完整源码快照；`camera_mount=[0.00,0.22,0.08]` 是已核实的历史 Git 配置，不能证明训练时无未提交改动。

## 2026-09-08 Avoid 终点／成功语义漂移（开训阻断；待作者决定）

### 已执行

- 服务器 Git 的 `-S _get_s_avoid_fixed_stage_last_row_y` 显示 helper 于 `38e5a67`（2026-05-09）加入。`49235d2` 的 `_get_s_avoid_cross_line_terms` 直接读取 `avoid_stageN_last_row_y`；当前实现改为 helper 计算实际 scaled row 的末行。
- `HexAvoidBasicCfg` 的 row spacing 旧/当前均为 `1.5`。stage 4 基础行坐标为 `(0.60, 1.85, 3.10, 4.35, 5.60)`，中心为 `3.10`；按当前实际行位置末行为 `6.85`，旧 cross-line 直接使用基础值 `5.60`。

### 待作者决定／未批准

- 该差异会改变 Avoid 训练的 success、goal/reward 终点相关语义，不能视为场景更名。用户要求其他设置与旧训练一致，因此不能在未冻结下一批前擅自保留当前行为或恢复旧的疑似缺陷。
- 仅可继续本地获批代码同步与原生环境检查；不得启动 full 或 smoke training，直到作者选择并冻结此项处理方式。

## 2026-09-08 重训前冻结

### 已执行

- 仅完成本轮文档记录；未修改训练源码、布局、奖励或网络，未启动训练或评测。

### 待核实

- 只读核对旧 checkpoint 生成时的真实训练布局、对应 Git 日期，以及训练后布局修改或命名变更。
- 核对历史训练的实际目标速度范围；在此之前，`0.50 m/s` 上限是后续重训决定，不是历史事实认定。

### 未批准

- 不擅自设计或搭建新的速度外推测试布局；该布局须与作者共同商量后再决定。
- 不新增 memory 消融、奖励修改、网络修改或其他训练设置变更。
- 不启动训练、评测或服务器侧修改。

### 已冻结的后续约束

- 后续重训目标速度上限为 `0.50 m/s`；`0.60 m/s` 只作速度外推评测。
- 训练布局保持旧 checkpoint 生成时的真实布局；除必要修正外，训练设置与旧训练一致。

## 2026-09-08 历史训练与布局只读核对（主线程实证）

### 已执行

- 本地仓库为 shallow，只可见 `97112c7`（2026-09-04）；服务器 `/home/dell/RL_hexapod_gym` 为非 shallow、清洁 HEAD `29ee797`（2026-06-26）。本轮仅读取 Git、JSON 与 TensorBoard 线索；未启动训练、评测或服务器修改。
- 服务器提交 `49235d2`（2026-04-03 10:08:22）新增 `s_pcr_line_avoid_basic`；其文档字符串明确保留 `s_avoid_basic` 的几何与课程、仅增加脚本移动目标，亦非单纯场景更名。
- 服务器提交 `a1671fa`（2026-05-22 17:04:43+08）新增 `s_pcr_new`，其继承 `s_pcr_line_avoid_basic`，不是场景更名；新增的是速度与障碍行数的二维课程。源码四档速度分别为 `0.25--0.40`、`0.35--0.55`、`0.30--0.55`、`0.35--0.65 m/s`。
- Learned-w 评测 JSON 的 primary checkpoint metadata 指向训练任务 `s_pcr_new` 与原始运行目录 `outputs/planner/moe_teacher_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_20260522_203529`；对应原始 `run_meta` 存在。
- Risk-only 评测 JSON 指向训练任务 `s_pcr_line_avoid_basic` 与 `agents/risk_only_seed1_targetview/moe_teacher_risk_only_gam0.15_20260609_230028`；Avoid auxiliary metadata 指向 `s_avoid_basic` 与 `outputs/planner/avoid_teacher_20260403_100941`。
- 服务器从 `6102102` 到 HEAD 的 `hex_scenes_config.py`、`scene_manager.py` 无差异。`hex_ground.py` 有 6 月后改动；其中新行距、row count 与 obstacle generation 的相关改动受 `eval_layout == 'heldout_irregular_rows'` 显式分支限定，但该文件另有 replay 等变化，不能将全文件表述为无改动。
- 提交 `7e3c3d4`（2026-06-18 13:45:21）修改 `heldout_irregular_rows` 的行距与障碍形状，属于温和 OOD 评测布局实改，不得混写为旧训练布局更名。
- TensorBoard 读取已正常结束（exit 0）：原始 Learned-w `run_meta` 为 `s_pcr_new`，共 1000 条记录；`PCRNew/CurriculumProgress` 最大/末尾 `0.6168500`，`PCRNew/Level3Ratio` 最大/末尾 `0.3125/0.28125`，`PCRNew/TargetSpeedMean` 最大/末尾 `0.4447942/0.4347386`，`Stats/TargetSpeed` 最大 `0.4435739`。这确认课程启用且出现 level 3；均值不证明逐环境最大速度。
- 为避免 pickle 执行风险，未用普通 `torch.load` 读取 `.pt`；只读取 JSON 与 TensorBoard 线索。

### 待核实

- Learned-w 原始运行目录前最近提交为 `6102102`（2026-05-22 20:08:16），但 metadata、目录时间或最近提交均不等同精确训练版本；不存在训练源码快照，不能据此钉死训练 commit 或排除未提交改动。
- 历史“训练最高 `0.50 m/s`”口径与 `s_pcr_new` 源码最高 `0.65 m/s` 冲突。TensorBoard 仅给出速度均值，不能由均值低于 `0.50 m/s` 推出逐环境最大速度不超过 `0.50 m/s`；旧模型是否见过 `0.60 m/s` 仍待逐环境速度或训练源码快照证据。
- Learned-w、Risk-only 与 Avoid metadata 均记录 `affordance_origin_mode=camera_mount`；后续生产验证不能把 base-center CPU fixture 视为已覆盖历史训练 origin。
- `hex_ground.py` 的其余 6 月后改动仍待逐项核对，不能声称所有旧训练设置已经完全还原。

### 已冻结边界不变

- 后续重训仍以 `0.50 m/s` 为最高训练速度，`0.60 m/s` 只作外推评测；该后续决定不因历史速度待核实而改变。
- 新速度外推测试布局仍须与作者共同商量后才可搭建；本轮未新增任何实验。
## 2026-09-10 Avoid 1-D C2 reader contract（CPU 完成；未运行 Isaac）

- 新增纯 Torch checkpoint contract：仅 `avoid_1d_canonical_v1` 的实际 1-D command head、14-D state encoder、2-D goal encoder 与五槽 mask 可作为 revised Avoid 读取；未知 1-D 一律拒绝，旧 3-D 权重保持旧输入。
- PCR 复用时 revised Avoid 固定接收 Follow proposal 的 `y` 作为第 14 槽和 Follow goal 的 side preference，只输出 `[x_right,0,0]`；旧 3-D 调用仍保持原 state/goal。
- C1 standalone、C2 helper、训练/eval/play 源码语法与 pure-Torch 夹具已通过；未加载真实 checkpoint，未启动 Isaac、训练、评测或实机。
