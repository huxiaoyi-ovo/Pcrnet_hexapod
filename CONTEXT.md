# Context Handoff Summary

## 1. 项目目标（Goal）

- 当前项目主线是六足机器人高层 PCR（Perception-Commanded / Planner Command Routing）跟随-避障融合，用 `follow expert + avoid expert + gate y + learned/signed w` 解决目标跟随与局部避障的冲突仲裁。
- 论文目标不是扩成大系统，而是尽快证明：在行障碍/连续障碍穿越中，command-conditioned conflict prior `w` 能让策略在高冲突窗口更合理地压低 Follow、提高 Avoid，同时不显著牺牲跟随质量。
- 实机目标是把仿真高层输入对齐到 D435i 可获得信号：`target state + local_map_2ch + optional deployable risk memory`，先 dry-run，再上小速度闭环。

## 2. 背景与范围（Context & Scope）

- In scope:
  - `s_pcr_line_avoid_basic` 主线训练、评测、play、论文图表证据链。
  - `learnedw2 / signed-w` 语义统一、`w_aux` 高冲突辅助监督、`risk_memory` 可部署短时风险记忆。
  - 新评测尺子：真正的 Follow-Avoid 高冲突窗口，而不是只按 `risk_F` 分桶。
  - `s_pcr_new` 二维课程场景与 `--generalize` 泛化评测。
  - D435i 实机输入检查、`local_map_2ch` 与仿真口径对齐、`pcr_realplay.py` dry-run。
- Out of scope:
  - 暂时不做复杂 learned beta、大范围奖励重写、端到端视觉策略。
  - 暂时不把目标 FOV 直接塞进 `w` 主创新；FOV 是可观测性约束，不是 `w` 的核心贡献。
  - 暂时不追求漂亮的软件形态；优先保证训练/评测口径正确、能出论文证据、能安全 dry-run。

## 3. 关键约束（Constraints）

- 技术/环境：
  - 仓库：`/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym`。
  - shell：`bash`。
  - Python 命令默认用 `python3`。
  - Isaac Gym 如果系统 Python 导入失败，切到 `/home/hxy/anaconda3/envs/hexapod_rl_env/bin/python3`。
  - 坐标固定：`goal_buf=(x_right,y_forward)`，相机前方是机器人 body `+Y`，`cmd=[x_lateral, y_forward, yaw]`。
- 实验优先级：
  - 先证明主线贡献是否成立。
  - 再提升关键指标。
  - 再做实机演示。
  - 最后补更完整的随机化、泛化和额外设定。
- 安全/部署：
  - `row_not_released`、障碍行编号、精确障碍中心只能用于仿真诊断或辅助标签，不能进入 actor 或实机控制。
  - `pcr_realplay.py` 默认 dry-run；没有 `--publish_cmd` 不发机器人运动。
  - 目标丢失、深度异常、目标太近必须触发安全输出，不能继续正常融合驱动。
- 并行训练资源：
  - 默认 `PCR -> GPU 0`，`avoid expert -> GPU 1`，除非用户指定别的卡。

## 4. 已确认的关键决策（Decisions）

- D1: 论文主公式采用 signed-w 语义，代码采用 signed-w-compatible 的 learnedw2 简化公式。
  - 不再同时保留 `learned / learnedw2 / signed` 三套叙事。
  - 统一公式：
    ```text
    w_s = 2w - 1
    w_s' = 0 if |w_s| < margin else w_s
    y_eff = clip(y + lambda * w_s' + gamma * (risk_A - risk_F), 0, 1)
    cmd = y_eff * cmd_F + (1 - y_eff) * cmd_A
    ```
  - `w` 网络仍输出 sigmoid `w in [0,1]`；高冲突避障标签 `w_hat=0`，等价于 `w_s=-1`。

- D2: 当前推荐 signed-w 训练配置是小幅、可解释、够用的版本。
  - 推荐：
    ```bash
    --w_mode learned
    --w_blend_mode signed
    --signed_w_lambda 0.30
    --signed_w_gamma_risk 0.15
    --signed_w_margin 0.05
    --pcr_w_aux_enable
    --pcr_w_aux_coef 0.05
    --pcr_w_aux_risk_f_threshold 0.25
    --pcr_w_aux_risk_margin 0.05
    --pcr_w_aux_cmd_cos_threshold 0.5
    --w_disable_gate_safe_clamp
    ```
  - `gamma_risk` 只作为小安全偏置，不作为主项。
  - `signed_w_margin` 必须保留，用于过滤 `w_s` 在 0 附近抖动。

- D3: `w_aux` 的 row 真值只能作为仿真训练辅助监督，不能进入部署 actor。
  - 当前强条件可能偏苛刻：
    ```text
    row_active & follow_risky & avoid_safer & command_conflict
    ```
  - 如果 valid rate 过低，推荐扩展成：
    ```python
    valid_row = (
        row_active
        & (risk_f > risk_thr)
        & ((risk_f - risk_a) > margin)
        & (cmd_cos < cos_thr)
    )

    valid_global = (
        (risk_f > max(risk_thr + 0.20, 0.45))
        & ((risk_f - risk_a) > max(margin + 0.10, 0.15))
        & (cmd_cos < min(cos_thr, 0.3))
    )

    valid = valid_row | valid_global
    ```
  - 目标 valid rate 约 `2%~8%`；低于 `0.5%` 太苛刻，高于 `20%` 可能太松。

- D4: `risk_memory` 是可选独立模块，必须可部署。
  - 不能用 row phase、行编号、精确障碍中心。
  - 使用深度图得到的 Follow 方向风险与机器人实际前进位移：
    ```text
    m_t = max(r_F(t), m_{t-1} * exp(-max(v_body_forward,0) * dt / L_clear))
    L_clear = L_body / 2 + L_margin ~= 0.40m
    ```
  - Actor 只看最终标量 `m_t`，不要把 `v_body / L_body / delta_s` 都塞给 actor。

- D5: 当前 eval 的主矛盾不是策略一定没用，而是旧尺子没有筛出真正的 Follow-Avoid 仲裁冲突。
  - 新高冲突定义：
    ```text
    HighConflict = ObstacleInteractionWindow & FollowPressure & AvoidPressure
    FollowPressure = cmd_F_forward > 0.20
    AvoidPressure = |cmd_A_lateral| > 0.10
    ObstacleInteractionWindow =
        row_current_valid
        & robot_front_y > row_front_y - 1.2
        & robot_rear_y < row_back_y + 0.4
    ```
  - 核心指标：
    ```text
    CSI = E[y_raw - y_eff | HighConflict]
    RCM = E[y_eff - y_raw | HighConflict] - E[y_eff - y_raw | not HighConflict]
    ```
  - 期望：高冲突窗口 `signed_w` 更低、`y_eff-y_raw` 更低；低冲突窗口不要全程压 Follow。

- D6: 目标可见性是可观测性约束，不是普通 reward。
  - `w` 解决 Follow/Avoid 行为冲突。
  - FOV constraint 保证 Follow 信息链不断。
  - 如果目标连续 `K=3~5` 个高层步丢失，应进入 safe stop/search，而不是继续正常 PCR 融合。

- D7: `s_pcr_new` 是新任务，只改二维课程，不动其他 `s_pcr_line` 配置。
  - Level 定义：
    ```text
    L0: speed [0.25,0.40], rows=2
    L1: speed [0.35,0.55], rows=2
    L2: speed [0.30,0.55], rows=3
    L3: speed [0.35,0.65], rows=4 or 5
    ```
  - 混合采样：
    ```text
    0-25%:   L0 70%, L1 30%, L2 0%,  L3 0%
    25-50%:  L0 30%, L1 40%, L2 30%, L3 0%
    50-75%:  L0 15%, L1 30%, L2 35%, L3 20%
    75-100%: L0 10%, L1 20%, L2 30%, L3 40%
    ```
  - `--generalize` 用于高难泛化评测，不替代主表格：固定 5 行、高速 `[0.55,0.75]`、压缩 row spacing。

- D8: D435i 实机输入必须对齐仿真 `local_map_2ch` 口径。
  - 默认主线：`D435i depth -> local_map_2ch -> avoid expert / PCR`。
  - 人由 YOLO bbox 内目标深度薄层 mask 排除；其他非人、非地板深度点作为障碍。
  - 地板必须排除；当前用简化高度阈值，后续需要标定相机外参。

- D9: PCR 主实验 baseline 收口为五策略对比。
  - 最终主表方法：
    ```text
    Y-only
    Geom-w
    Learned-w
    Mono-PPO
    Rule-Override
    ```
  - `Y-only / Geom-w / Learned-w` 回答内部 `w` 消融问题。
  - `Mono-PPO` 回答“为什么不直接训练统一策略”。
  - `Rule-Override` 回答“为什么不用简单安全规则覆盖”。
  - 不再优先扩展内部 `E2E gate` 消融，也不优先接 DWA / TEB / MPC。

- D10: Rule-Override 先于 Mono-PPO 实现。
  - Rule-Override 是无需训练的强规则 baseline，应先补进 0.35 / 0.50 / 0.60 × 3 seeds 评测。
  - 规则必须保留 slow-forward 和 yaw-preserve，避免变成弱 baseline：
    ```text
    risk_gap = risk_F - risk_A
    s = sigmoid(k * (risk_gap - margin))
    if risk_F > hard_thr:
        s = max(s, s_min)

    cmd_x   = s * cmd_A_x
    cmd_y   = (1 - s + slow_ratio * s) * cmd_F_y
    cmd_yaw = (1 - yaw_keep_loss * s) * cmd_F_yaw
    ```
  - 默认参数：
    ```text
    k=8, margin=0.10, hard_thr=0.60, s_min=0.70,
    slow_ratio=0.20, yaw_keep_loss=0.50
    ```
  - Mono-PPO 后做，必须公平训练：同场景、同 curriculum、同 seeds、同训练步数、同底层策略、同 action limit、同 eval 协议，且只吃 deployable observation，不吃 `cmd_F/cmd_A/risk_F/risk_A`。

## 5. 当前代码与结构（Code Map）

- `legged_gym/scripts/train_highlevel.py`
  - 高层训练入口。
  - checkpoint contract 检查在这里影响 train/play 启动。
  - 之前 low_level_ckpt 路径不一致已从强制停止降为 warning；真正观测/动作契约不一致仍应严格。

- `legged_gym/scripts/play_highlevel.py`
  - 用于快速观察 PCR 行为。
  - 需要重点看：`gate(raw/eff/w)`、`conflict`、`rowNR`、`cmd_F/cmd_A/pred/exec`、目标是否丢视野。

- `legged_gym/scripts/eval_highlevel.py`
  - 当前论文指标与机制统计入口。
  - 已围绕高冲突窗口、FOV 指标、timeseries dump 做过多轮修正。
  - 后续判断 `w` 是否有效，应看新 `HighConflict` 尺子下的 `CSI/RCM/signed_w/y_eff-y_raw`，不要只看 `risk_F bin`。

- `legged_gym/envs/hex_v4/hex_ground_config.py`
  - 当前 PCR 训练默认 `navigation.affordance_grid_size=32`。
  - D435i 仿真相机参数参考：
    ```text
    depth resolution: 1280x720
    depth FOV: 87 x 58 deg
    near/far: 0.28 / 3.0 m
    fps: 30
    camera position: [0.00, 0.22, 0.08]
    pitch_deg=0, roll_deg=20, yaw_deg=90
    capture_interval=2
    output_size=128
    ```
  - 实机默认 `local_map_2ch` 当前按 `32x32` 对齐训练。

- `legged_gym/scripts/real_pcr_input_check.py`
  - 当前实机相机输入检查脚本，已改为可执行。
  - 默认 `--map_size 32`。
  - 输出：`goal_buf`、`target_valid/lost/too_close`、`target_vel`、`depth_invalid`、`local_map_2ch`。
  - `local_map_2ch[0]=occupancy`，`1` 表示障碍。
  - `local_map_2ch[1]=passable/safety`，`1` 表示经过机器人半径膨胀后仍可通行。
  - 当前还没有 ROS1 发布真实输入。

- `legged_gym/scripts/pcr_realplay.py`
  - 实机策略 dry-run / 后续发布入口。
  - 默认 `--map_size 32`。
  - 默认不发布运动；只有显式 `--publish_cmd` 才发布。
  - 下一步需要接收 `real_pcr_input_check.py` 发布的真实 `target_state + local_map_2ch`。

- `docs/specs/PCR_THEORY_DEFINITIONS_CN.md`
  - 已新增论文理论定义文档。
  - 用于沉淀高冲突定义、signed-w 公式、risk memory、FOV constraint 等后续论文可直接复用内容。

- `TODO_LOG.md`
  - 重大方向、评测协议、任务变更写这里。
  - 小修小补不需要记录。

## 6. 当前进度（Status）

- 已完成：
  - learnedw2 与 signed-w 叙事基本收口：论文用 signed-w 语义，代码用兼容公式。
  - `w_aux` 高冲突辅助监督口径已明确，阈值建议已确定。
  - `risk_memory` 设计为可部署独立模块，可选开启。
  - 新 eval 高冲突窗口定义已经形成，目标是修正“证据尺子”。
  - `s_pcr_new` 二维课程任务已作为独立任务设计/实现过，原则是只改速度范围与障碍行数。
  - `--generalize` 高难评测方案已设计/实现过，用于泛化表或补充表。
  - FOV / target observability 已上升为论文级约束。
  - D435i 输入检查脚本已能显示 RGB、目标 mask、障碍/安全图，并能输出真实输入字段。
  - `real_pcr_input_check.py` 与 `pcr_realplay.py` 默认 `map_size` 已从 16 改为 32，对齐当前训练口径。
  - `0.35 / 0.50 / 0.60 × Y-only / Geom-w / Learned-w × 3 seeds` 主表已拼好，结果支持 learned-w 在中高速下显著降低碰撞并保持成功率。
  - 速度曲线图已收口：主图只保留 `Command Conflict Rate` 与 `Success Rate / Collision Rate / Tracking MAE`；`Forward Risk` 仅保存在画图数据中备用，不进正文主图。
  - baseline 设计已确定：先做 `Rule-Override`，再做 `Mono-PPO`，形成五策略主实验。
  - 语法检查曾通过：
    ```bash
    python3 -m py_compile legged_gym/scripts/real_pcr_input_check.py legged_gym/scripts/pcr_realplay.py
    git diff --check -- legged_gym/scripts/real_pcr_input_check.py legged_gym/scripts/pcr_realplay.py
    ```

- 进行中：
  - 正在准备接入 `Rule-Override` eval 分支：在 `cmd_F/cmd_A/risk_F/risk_A` 已经算完后替代原 PCR `y_eff` 融合，不进入 learned-w 逻辑。
  - 正在把 D435i 生成的 `local_map_2ch` 与仿真 `occupancy/passable` 口径完全对齐。
  - 正在准备把相机脚本输出接入 `pcr_realplay.py` dry-run。

- 未开始：
  - `Rule-Override` 的三速度三 seed 评测与五策略图表更新。
  - `Mono-PPO` 训练分支、checkpoint meta、训练命令和公平性审查。
  - `real_pcr_input_check.py --ros1_publish` 发布 `/pcr/target_state` 与 `/pcr/local_map_2ch`。
  - 真实相机输入驱动 `pcr_realplay.py` dry-run。
  - 小速度真实机器人运动测试。
  - 五策略最终主表与机制表拆分：主性能表五策略都报，机制表中 Mono-PPO 的 `CSI@C_avoid` 记为 `N/A`。

## 7. 已知问题与风险（Known Issues & Risks）

- P0: 旧 eval 如果只按 `risk_F` 或 overall success 判断，无法证明 `w` 是否真的在冲突窗口起作用。
- P0: 真实输入中目标丢失时不能给策略发布 NaN；应发布 `valid=0`、目标状态置 0，并触发 safe stop/search。
- P0: `real_pcr_input_check.py` 还没有 ROS1 发布，`pcr_realplay.py` 还不能直接吃真实相机输出。
- P1: 实机地板排除仍依赖简化外参与高度阈值；如果相机高度/俯仰不准，地板或柜子底部可能误判。
- P1: `risk_memory` 如果部署时没有可靠 body velocity，只能先用 fused cmd 近似，记忆释放可能偏乐观。
- P1: 当前 learnedw2/memory 策略可能转向不足，目标离开 RGB FOV 后 Follow 信息链会断；这需要 eval/play 指标和实机 hard guard 支撑。
- P1: `s_pcr_new --generalize` 适合做泛化表，不应替代主任务表，否则审稿人可能认为训练/评测任务漂移。
- P2: 可视化颜色曾多次误导判断；最终以数值字段和策略输入数组语义为准。
- P2: `clearance_mean` 旧命名容易误导；当前第二通道应理解为 `passable/safety`。

## 8. 重要命令备忘（Commands）

- learnedw2 / signed-w 主训练参考：
  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 legged_gym/scripts/train_highlevel.py \
    --task s_pcr_line_avoid_basic \
    --mode teacher \
    --skill moe \
    --wlearned2 \
    --w_blend_mode signed \
    --signed_w_lambda 0.30 \
    --signed_w_gamma_risk 0.15 \
    --signed_w_margin 0.05 \
    --pcr_w_aux_enable \
    --pcr_w_aux_coef 0.05 \
    --pcr_w_aux_risk_f_threshold 0.25 \
    --pcr_w_aux_risk_margin 0.05 \
    --pcr_w_aux_cmd_cos_threshold 0.5 \
    --w_disable_gate_safe_clamp
  ```

- learnedw2 eval 示例：
  ```bash
  CUDA_VISIBLE_DEVICES=2 python3 legged_gym/scripts/eval_highlevel.py \
    --task s_pcr_line_avoid_basic \
    --mode teacher \
    --skill moe \
    --pcr_ckpt outputs/planner/moe_teacher_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_20260521_173601/best_online_reward.pt \
    --avoid_ckpt agents/avoid_best.pt \
    --lowlevel_ckpt agents/low_level_best.pt \
    --num_envs 64 \
    --episodes 128 \
    --seed 1 \
    --output_dir outputs/eval/wlearned2_seed1 \
    --avoid_stage_override 4 \
    --freeze_avoid_stage \
    --dump_timeseries \
    --timeseries_episodes 64
  ```

- yonly eval 对照应使用同一 task、seed、episodes、stage override、avoid/lowlevel ckpt，只替换 PCR ckpt 与输出目录。

- D435i 输入检查：
  ```bash
  cd /home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym

  ./legged_gym/scripts/real_pcr_input_check.py \
    --show \
    --width 640 \
    --height 480 \
    --fps 30 \
    --display_scale 1.2 \
    --debug_map_px 260 \
    --ground_remove_height_m 0.04 \
    --robot_clearance_m 0.27
  ```

- realplay dry-run 模板：
  ```bash
  python3 legged_gym/scripts/pcr_realplay.py \
    --pcr_ckpt <PCR_CKPT.pt> \
    --avoid_ckpt agents/avoid_best.pt \
    --lowlevel_ckpt agents/low_level_best.pt \
    --allow_missing_state
  ```
  不要加 `--publish_cmd`，直到 dry-run 日志确认正常。

## 9. 待办清单（Next Actions / TODO）

- [P0] 用新高冲突尺子重新评测 yonly、learnedw2、learnedw2+risk_memory，并输出 `CSI/RCM/signed_w/y_eff-y_raw` 对比。
- [P0] 先实现 `Rule-Override` eval 分支并跑 0.35 / 0.50 / 0.60 × 3 seeds，补出第四个方法的主表和速度曲线。
- [P0] 扩展汇总表和画图脚本到五策略顺序：`Y-only / Geom-w / Learned-w / Mono-PPO / Rule-Override`。
- [P0] 设计并实现 `Mono-PPO` 训练分支，保证同场景、同预算、同 seeds、同 eval 协议，且不输入专家命令或专家风险。
- [P0] 给 `real_pcr_input_check.py` 增加 `--ros1_publish`，发布 `/pcr/target_state` 与 `/pcr/local_map_2ch`。
- [P0] 确保目标丢失时 ROS 输出为 `valid=0`、`goal/vel=0`、`target_lost=true`，禁止 NaN 进入策略。
- [P0] 用真实相机输入跑 `pcr_realplay.py` dry-run，确认能打印 `cmd_F / cmd_A / y / w / y_eff / cmd_safe`。
- [P0] 做三组静态输入验收：空地、正前方箱子、人旁边箱子。
- [P1] 标定相机外参：`camera_height_m / camera_pitch_down_deg / camera_forward_offset_m / ground_remove_height_m`。
- [P1] 给 realplay 加 target lost hard guard：连续丢失 `K=3~5` 步后 safe stop/search。
- [P1] 用 `s_pcr_new --generalize` 做泛化评测，但先不要替代主任务表。
- [P1] 修正 `pcr_realplay.py fake_input`，让假障碍图也按 `robot_clearance_m` 膨胀。
- [P2] 整理 `docs/specs/PCR_THEORY_DEFINITIONS_CN.md`，持续沉淀论文公式、定义和规范表述。

## 10. 需要新会话首先回答的三个问题（Top Questions）

1. Rule-Override 应接在 `eval_highlevel.py` 哪个最小分支，才能复用现有 `cmd_F/cmd_A/risk_F/risk_A` 且不污染 learned-w 路径？
2. Mono-PPO 训练分支应复用现有哪个 actor 观测构造，才能做到 deployable observation 公平对比？
3. 现在是先跑 Rule-Override 三速度三 seed，还是先接 ROS1 dry-run 做实机输入闭环？
