# Context Handoff Summary

## 1. 项目目标（Goal）

- 当前主线是训练 `s_avoid_basic` 的高层避障专家。
- 目标不是做全局规划，而是学出“局部短时安全通过”：持续朝局部子目标前进，主要靠横移和小幅姿态变化通过障碍，不撞、不贴边硬挤、不过久卡住。
- 当前优先级是先把固定模板场景上的连续逐行横移通过学稳定，再看后续融合与实机叙事。

## 2. 背景与范围（Context & Scope）

- In scope:
  - `avoid expert` 的奖励、课程升级、固定模板几何、回合时长、调试与训练输出口径。
  - 关键文件：`legged_gym/scripts/train_highlevel.py`、`legged_gym/envs/hex_v4/hex_ground.py`、`legged_gym/envs/hex_v4/hex_scenes_config.py`、`legged_gym/scripts/play_highlevel.py`。
- Out of scope:
  - PCR / follow 融合主线。
  - 底层 locomotion。
  - 视觉 student 蒸馏与实机深度处理实现细节。

## 3. 关键约束（Constraints）

- 技术/环境：
  - 默认任务为 `s_avoid_basic`，高层观测固定为 `state + goal_buf + local_map_2ch`。
  - `goal_buf` 解释为局部子目标；当前固定模板里，横向归位等价于回到目标横向中线。
  - `band` 语义固定为左右横移外墙，只罚 `x` 超界。
- 性能/可靠性：
  - 避障训练默认可用 `GPU 1`。
  - 必须保持奖励、课程、日志三条链路口径一致，避免静默污染训练结论。
- 安全/合规：
  - 不允许靠撞过去、贴边硬挤、或奖励黑客拿到 success。
- 不可改动（Do-not-touch）：
  - 当前 `avoid expert` 的功能定位：局部安全通过专家，不是全局规划器。
  - `band` 作为外墙的语义不要再放松。

## 4. 已确认的关键决策（Decisions）

- D1: 固定模板继续保留，但拉长纵深并增宽宽侧。
  - 选择了“每行间距统一加长 30cm，宽侧外扩 10cm，窄侧不动”。
  - 原因：先用更干净、可控的教学场把连续逐行换边学出来。
  - 否决了继续随机化或重新设计复杂场景，因为那会拖慢主线验证。

- D2: 奖励主线改成“远处横向归位 + 近处横移避障”。
  - 选择了门控双模式：远障碍时奖励真实横向归位，近障碍时奖励真实 clearance 改善。
  - 原因：旧的单一 `clearance_improve` 只会教会第一排前的局部躲避。
  - 否决了继续只靠碰撞/外墙惩罚，因为那只能学到硬挤坏解。

- D3: 课程升级改成逐行比例口径。
  - 选择了“progress = 已通过行数/总行数，success = 无碰撞通过行数/总行数”。
  - 原因：旧的 final-cross-line 口径无法代表连续逐行通过能力。
  - 否决了继续用“最后一行是否过线”做主升级条件，因为会放行坏策略。

## 5. 当前代码与结构（Code Map）

- 目录/模块概览：
  - `legged_gym/scripts/train_highlevel.py`: 高层训练、avoid 奖励、日志输出。
  - `legged_gym/envs/hex_v4/hex_ground.py`: 场景生成、episode 统计、课程升级。
  - `legged_gym/envs/hex_v4/hex_scenes_config.py`: avoid 任务最终配置口径。
  - `legged_gym/scripts/play_highlevel.py`: debug 可视化与 band 线框显示。
  - `TODO_LOG.md`: 重大方向调整记录。

- 关键文件：
  - `legged_gym/scripts/train_highlevel.py`: 
    - `s_avoid` 奖励分支已收口为有效主项输出。
    - 近障碍 `clearance_improve` 使用真实 `nearest_obs_dist` 改善。
    - 远障碍 `align_center` 使用真实 `|goal_x|` 改善，不再奖励命令方向。
    - `band` 激活已与配置统一；当前 `avoid_band_activate_progress=0.0` 等价于出生即激活。
  - `legged_gym/envs/hex_v4/hex_ground.py`:
    - 已新增逐行统计缓冲：`rows_passed_best`、`rows_success_best`。
    - 课程升级现在吃逐行 progress/success 浮点比例。
  - `legged_gym/envs/hex_v4/hex_scenes_config.py`:
    - 已落地新固定模板几何、最终奖励系数、课程升级阈值、回合时长。
  - `legged_gym/scripts/play_highlevel.py`:
    - `band` 的 `x` 边界用红线，`y` 边界用绿线。
    - 会打印 `robot_xy / band_x_min / band_x_max / dx_out`。

## 6. 当前最终训练口径（Status）

- 已完成：
  - 固定模板几何已更新：
    - 宽侧外扩 10cm：
      - `avoid_fixed_row_x_open_right = (-0.85, -0.25, 1.05)`
      - `avoid_fixed_row_x_open_left  = (-1.05, 0.25, 0.85)`
      - `avoid_fixed_row_x_open_right_even = (-0.85, -0.35)`
      - `avoid_fixed_row_x_open_left_even  = (0.35, 0.85)`
    - 行间距统一拉长 30cm：
      - `stage1_row_y = (0.80, 2.45)`
      - `stage2_row_y = (0.70, 2.15, 3.60)`
      - `stage3_row_y = (0.65, 2.00, 3.35, 4.70)`
      - `stage4_row_y = (0.60, 1.85, 3.10, 4.35, 5.60)`
    - `episode_length_s = 50`
  - 当前最终生效奖励系数：
    - `approach = 4.0`
    - `target_visible = 0.05`
    - `clearance_improve = 6.0`
    - `align_center = 4.0`
    - `avoid_band_penalty = 3.0`
    - `collision_penalty = -20.0`
    - `stability = 0.01`
    - `terminal_fail_penalty = -10.0`
    - `time_penalty = -0.01`
  - 当前已关闭奖励：
    - `heading = 0.0`
    - `target_center = 0.0`
    - `velocity = 0.0`
    - `backward = 0.0`
    - `body_backward = 0.0`
    - `turn_penalty = 0.0`
    - `yaw_rate_penalty = 0.0`
    - `passable_align = 0.0`
    - `crossable_align = 0.0`
    - `risk_barrier = 0.0`
    - `reach = 0.0`
    - `target_lost = 0`
  - 课程升级口径已改对：
    - `progress = 已通过行数 / 总行数`
    - `success = 无碰撞通过行数 / 总行数`
    - `stage1 -> 2`: `window=200`, `min_eps=300`, `progress>=0.80`, `success>=0.50`, `collision<0.05`
    - `stage2 -> 3`: `window=200`, `min_eps=400`, `progress>=0.85`, `success>=0.55`, `collision<0.05`
    - `stage3 -> 4`: `window=200`, `min_eps=500`, `progress>=0.88`, `success>=0.60`, `collision<0.04`
  - 调试口径已统一：
    - `band` 红线是左右外墙，绿线是前后边框。
    - `band` 罚的是机器人根刚体原点的 world `x`。
    - 训练控制台只打印当前真正生效的主奖励项。

- 进行中：
  - 下一轮短训要验证这版“远处归位 + 近处避障 + 逐行升级”是否能真正学出第二排、第三排连续换边。

- 未开始：
  - 基于新口径的长训结果分析。
  - 若仍只会过第一排，再继续收 `align_center` 定义，而不是先乱动主惩罚。

## 7. 已知问题与风险（Known Issues & Risks）

- 当前没有必须在继续训练前先修的主链实现错误。
- 仍需注意的残留风险：
  - `align_center` 仍是横向归位代理信号，不是“下一排行真正可通过位置”的真值奖励。
  - `target_visible` 虽已降到 `0.05`，但在没有主动转向命令时，仍可能轻微压制大横移。
  - 配置里还残留一些旧课程字段，如 `progress_delta / success_distance / exposure_threshold`，当前已不再用于升级判断，容易误导读配置。
  - `stage4_shrink_*` 在 `avoid_use_fixed_presets=True` 主线下基本不生效，也是残留字段。

## 8. 待办清单（Next Actions / TODO）

- [P0] 用当前最终口径跑新的短训，重点看第二排、第三排是否继续主动横移。
- [P0] 对照回放与控制台：`Reward(main appr/vis/clear/align)`、`Reward(cost band/col/stab/term)`、`Avoid stage prog/succ`。
- [P1] 若仍只会过第一排，优先继续收 `align_center` 的定义，不先动 `band/collision` 主惩罚。
- [P1] 后续有空时清理配置里的死字段，避免继续误读课程升级条件。

## 9. 当前推荐训练命令（Command）

- 长训：
  - `CUDA_VISIBLE_DEVICES=1 bash -lc "source /home/hxy/anaconda3/etc/profile.d/conda.sh && conda activate hexapod_rl_env && python legged_gym/scripts/train_highlevel.py --task s_avoid_basic --mode teacher --skill avoid --low_level_ckpt logs/hex_ground/Dec31_16-52-59_/model_6000.pt --num_envs 2048 --num_iterations 2000 --seed 42 --headless --save_interval 200"`
- 快速短训：
  - `CUDA_VISIBLE_DEVICES=1 bash -lc "source /home/hxy/anaconda3/etc/profile.d/conda.sh && conda activate hexapod_rl_env && python legged_gym/scripts/train_highlevel.py --task s_avoid_basic --mode teacher --skill avoid --low_level_ckpt logs/hex_ground/Dec31_16-52-59_/model_6000.pt --num_envs 2048 --num_iterations 300 --seed 42 --headless --save_interval 100"`

## 10. 需要新会话首先回答的三个问题（Top Questions）

1. 这版训练里，第二排和第三排是否开始连续主动横移，而不是只会过第一排？
2. 当前 `align_center + target_visible` 的组合，是否仍然在轻微压制大横移？
3. 如果连续逐行通过仍不够，下一刀该继续收 `align_center`，还是该改局部通路奖励定义？
