# 审稿行动日志（内部工作稿）

> 本日志记录作者决策与后续核对边界；不记录为已完成实验或历史事实。

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
