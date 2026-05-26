# PCR RAL Day 1：论文主张与评测协议

日期：2026-05-26

本文档是两周 RAL 冲刺 Day 1 的落地产物。作用是把论文主张、方法边界、主表字段、机制图字段和 Day 2 评测命令固定下来，避免后续实验在方法名、指标口径和对照设置上漂移。

## 1. 一句话研究问题

六足机器人在移动目标跟随时遇到连续局部障碍，Follow expert 想继续追目标，Avoid expert 想横移或减速避障，两者在短时间窗口内产生冲突；本文研究如何用 PCR-Net 的 `y + w + beta + Command Post-Processor` 做前瞻式、可解释、可调节的 Follow/Avoid 仲裁，减少危险跟随和碰撞，同时尽量保持目标跟随质量。

英文初稿：

```text
We study how a hexapod robot can arbitrate between target following and local obstacle avoidance when the follow and avoid experts produce competing commands, and propose PCR-Net, a predictive conflict resolution framework with a gating policy, a command-conditioned conflict prior, and a risk-budgeted command post-processor.
```

## 2. 当前论文主张

主张写法采用“先完整写，终稿按证据收缩”：

```text
PCR-Net uses y + w + beta + Command Post-Processor to achieve predictive, interpretable, and tunable runtime conflict resolution for hexapod target following in cluttered local environments.
```

中文展开：

```text
在六足机器人移动目标跟随与局部避障冲突场景中，PCR-Net 将 Follow/Avoid 仲裁拆成三部分：门控权重 y 负责结构性选择，learned-w 负责从候选命令与局部风险中学习短未来冲突先验，风险预算 beta 联动指令后处理器负责安全-效率取舍。当前两周实验的主线是证明 learned-w 相比 yonly 和 geom-w 能在真实高冲突窗口中自适应调制仲裁：既避免 yonly 的危险硬跟随，也避免规则避障的过度保守，从而保持任务推进、降低碰撞并改善目标跟随质量。
```

这次 RAL 版本不主张：

- 不主张 `Student` 系统已经完成。
- 不主张 learned beta 是核心贡献。
- 不主张端到端视觉部署已经解决。
- 不主张泛化到所有障碍和所有人类运动模式。

## 3. 方法命名

当前写作使用以下命名：

Comparative baselines：

- `Monolithic PPO`：同样输入观测，单一 RL 策略直接输出 `[vx, vy, wz]`，回答“为什么不直接训练一个黑盒策略？”
- `Reactive Safety Override`：跟随优先，风险超过阈值时用手工 safety override 降速/横移，回答“为什么简单规则不够？”
- `DWA-inspired Local Rollout`：用 moving local waypoint 和短时速度采样 rollout 打分，回答“经典局部规划启发式能否解决？”

Internal PCR ablations：

- `PCR-yonly`：只用 gate 输出原始 Follow 权重 `y_raw`，无显式冲突先验。
- `PCR-geomw`：使用手工几何规则产生命令条件化冲突先验，是 learned-w 必须打败的强规则基线。
- `PCR-learnedw`：学习式命令条件化冲突先验，是本文主贡献和主方法。

脚本/日志兼容命名：

- 代码参数中当前仍可能出现 `learnedw2 / --wlearned2`，论文和计划文档统一写作 `learned-w`。
- `risk_memory` 归入 learned-w 的最终实现优化，用于补偿短时可见风险消失后的过早恢复 Follow；它不单列为第四个 baseline。

禁止写法：

- 不把 `learnedw2` 写成论文显示名；正文统一使用 `learned-w`。
- 不把 `w` 写成只看 clearance 的普通距离规则。
- 不把评测用 privileged high-conflict mask 写成 actor 输入。
- 不把 `risk_memory` 写成独立主贡献或主表第四行；若启用，它属于 learned-w。

## 4. 贡献列表

### C1：learned-w 命令条件化冲突先验

贡献表述：

```text
We introduce learned-w, a learned command-conditioned conflict prior that adaptively modulates Follow/Avoid arbitration from candidate expert commands and local risk observations, preserving task progress while maintaining safety under geometric conflict.
```

这是本文主贡献。必须用 `yonly / geom-w / learned-w` 三组主表、机制表和时间序列支撑。

验收：

- `PCR-learnedw` 相比 `PCR-yonly` 明显降低碰撞或 near-miss。
- `PCR-learnedw` 相比 `PCR-geomw` 在至少一个主指标或机制指标上有优势，且跟随误差和目标视野不明显恶化。
- `learned-w` 的优势能在 HighConflict 时间序列里解释，不只是表格偶然变好；解释不预设一定压低 Follow，允许表现为 progress-preserving Follow-support。

### C2：PCR-Net 前瞻式冲突消解框架

贡献表述：

```text
We formulate legged target following in clutter as a runtime conflict-resolution problem and instantiate PCR-Net with gating, learned conflict prediction, and risk-budgeted command post-processing.
```

必须用整体架构图、主表和消融表支撑。

验收：

- `PCR-yonly -> PCR-geomw -> PCR-learnedw` 的递进关系清楚。
- 主表能说明 PCR-Net 相比外部 baseline 和 `PCR-yonly` 都更适合 Follow/Avoid 冲突，且不靠完全牺牲跟随质量换指标。
- 论文中明确 `s_pcr_line_avoid_basic` 是当前产数主场景，不把它包装成所有受限环境的全集。

### C3：风险预算 `beta` 与指令后处理器

贡献表述：

```text
We use a risk budget beta to parameterize the command post-processor, enabling a measurable risk-efficiency Pareto curve without retraining the policy.
```

必须用 `beta_sweep / Pareto` 支撑。

验收：

- 至少 5 个 beta 点：`0.0 / 0.25 / 0.5 / 0.75 / 1.0`。
- 指标至少包含碰撞或 near-miss、进度或成功率、跟随误差、cmd jerk。
- 曲线必须能解释安全-效率取舍；若不单调，必须写入失败分析，不强行包装。

### C4：实机可得输入链路与低速演示

贡献表述：

```text
We align the policy inputs with a deployable D435i depth-based local map and validate the real-input chain toward a low-speed robot demonstration.
```

这不是主理论贡献，但对 RAL 机器人论文很关键。

验收：

- 三组 D435i 静态输入检查通过。
- `pcr_realplay.py` dry-run 能打印 `cmd_F / cmd_A / y / w / y_eff / cmd_safe`。
- 目标丢失、深度异常、目标太近能触发 safe output。
- 两周目标冲真实低速跟随避障演示；若实机条件不满足，最低保留架空或静态下发 cmd 录像作为补救证据。

## 4.5 论文主张-证据对应表

| 论文主张 | 当前证据任务 | 必须产物 | 通过标准 |
|---|---|---|---|
| PCR-Net 优于黑盒单策略 | `PCR-learnedw` vs `Monolithic PPO` | Table 1 | 安全或进度更好，且跟随质量不明显崩掉 |
| PCR-Net 优于简单规则仲裁 | `PCR-learnedw` vs `Reactive Safety Override` | Table 1 | 高冲突窗口中更少 near-miss/collision，且不全程保守 |
| PCR-Net 优于经典局部启发式 | `PCR-learnedw` vs `DWA-inspired Local Rollout` | Table 1 | 在动态目标跟随冲突中更稳 |
| `learned-w` 优于无冲突先验 | `PCR-learnedw` vs `PCR-yonly` | Table 1 | 碰撞或 near-miss 降低，跟随误差和目标视野不明显崩掉 |
| `learned-w` 优于手工几何先验 | `PCR-learnedw` vs `PCR-geomw` | Table 1 + Table 2 | 至少一个主指标或机制指标优于 `PCR-geomw`，且能被 HighConflict 时间序列解释 |
| `beta` 是可调风险预算，不是奖励调参口号 | `beta_sweep` | Pareto Figure | 5 个 beta 点能显示安全-效率变化趋势 |
| 当前场景能稳定产生论文问题 | `s_pcr_line_avoid_basic` seed 1 smoke eval | metrics + timeseries | `priv_high_conflict_step_rate` 不接近 0，窗口不过宽 |
| 实机链路具备部署可信度 | D435i 静态检查 + dry-run + 低速演示 | 图/视频/日志 | dry-run 安全输出通过，低速演示作为两周冲刺目标 |

## 5. 主任务与补充任务

主任务：

```text
s_pcr_line_avoid_basic
```

补充任务：

```text
s_pcr_new --generalize
```

主表只用 `s_pcr_line_avoid_basic`。`s_pcr_new --generalize` 只做补充泛化，不替代主任务表。

## 6. 固定评测设置

主表默认设置：

```text
task = s_pcr_line_avoid_basic
mode = teacher
skill = moe
avoid_stage_override = 4
freeze_avoid_stage = true
num_envs = 64
episodes = 128
seeds = 1,2,3
avoid_ckpt = agents/avoid_best.pt
lowlevel_ckpt = agents/low_level_best.pt
dump_timeseries = true
timeseries_episodes = 64
```

如果 Day 2 速度太慢：

```text
episodes = 64
timeseries_episodes = 32
```

该压缩只用于趋势判断；最终主表优先补回 128 episodes。

## 7. 主表字段

Table 1：主性能对比。

Comparative baselines：

- `Monolithic PPO`
- `Reactive Safety Override`
- `DWA-inspired Local Rollout`

Internal PCR ablations：

- `PCR-yonly`
- `PCR-geomw`
- `PCR-learnedw`

最低主表：

- `Monolithic PPO`
- `Reactive Safety Override`
- `PCR-yonly`
- `PCR-geomw`
- `PCR-learnedw`

列：

- `row_progress_success_mean`
- `full_task_success_rate`
- `episode_collision_rate`
- `near_miss_rate_mean`
- `follow_mae_m_mean`
- `follow_rmse_m_mean`
- `target_in_rgb_fov_rate`
- `target_in_fov_rate_in_priv_conflict`
- `priv_high_conflict_step_rate`

最低判断标准：

- `PCR-learnedw` 的 `episode_collision_rate` 或 `near_miss_rate_mean` 低于 `PCR-yonly`。
- `PCR-learnedw` 相比 `PCR-geomw` 至少在一个安全或机制指标上有可解释优势。
- `PCR-learnedw` 相比 `Monolithic PPO` 和 `Reactive Safety Override` 至少要有清楚优势。
- `PCR-learnedw` 的 `row_progress_success_mean` 不明显低于 `PCR-geomw`。
- `PCR-learnedw` 的 `follow_mae_m_mean / follow_rmse_m_mean` 不明显恶化。
- `target_in_rgb_fov_rate` 不能为了避障收益大幅崩掉。

## 8. 机制表字段

Table 2：PCR 内部 w 机制消融。

注意：机制证据分两层。

- `priv_conflict_*` 表示 row-command conflict：机器人在障碍行窗口内，Follow 有前进压力，Avoid 有横移压力。它证明“发生了 Follow/Avoid 行为冲突”，但不等价于 Follow 路径已经危险。
- `unsafe_conflict_*` 表示 unsafe command conflict：对 `cmd_F / cmd_A / cmd_S` 三个外生候选命令做短时几何风险对比，要求 `risk_F` 高、`risk_F - min(risk_A, risk_S)` 高、命令方向分歧明显、目标仍可恢复。它用于定位真正危险的 Follow 候选，不直接规定 learned-w 必须压低 Follow。
- `avoid_conflict_*` 表示 `C_avoid`：`C_unsafe` 中 `Avoid` 的任务效用高于 `Stop/Slow`，效用同时考虑风险、前向推进和目标距离拉开代价，用于判断哪些危险窗口确实该由 Follow/Avoid 仲裁处理。
- `stop_conflict_*` 表示 `C_stop`：`C_unsafe` 中 Stop/Slow 的任务效用不低于 Avoid，这部分不强行归因给 w，应交给 beta 或安全后处理解释。

行：

- `PCR-yonly`
- `PCR-geomw`
- `PCR-learnedw`

列：

- `priv_conflict_w_mean`
- `priv_conflict_signed_w_mean`（只用于 `PCR-learnedw / learnedw2`，`PCR-yonly` 与 `PCR-geomw` 不用它解释）
- `priv_conflict_delta_y_mean`
- `conflict_suppression_index`
- `conflict_selective_suppression`
- `relative_conflict_modulation`
- `priv_conflict_phase_approach_w_mean`
- `priv_conflict_phase_inside_w_mean`
- `priv_conflict_phase_release_w_mean`
- `priv_conflict_phase_approach_signed_w_mean`（只用于 `PCR-learnedw / learnedw2`）
- `priv_conflict_phase_inside_signed_w_mean`（只用于 `PCR-learnedw / learnedw2`）
- `priv_conflict_phase_release_signed_w_mean`（只用于 `PCR-learnedw / learnedw2`）
- `priv_conflict_phase_approach_delta_y_mean`
- `priv_conflict_phase_inside_delta_y_mean`
- `priv_conflict_phase_release_delta_y_mean`
- `unsafe_conflict_step_rate`
- `unsafe_conflict_signed_w_mean`
- `unsafe_conflict_delta_y_mean`
- `unsafe_conflict_suppression_index`
- `unsafe_conflict_selective_suppression`
- `unsafe_relative_conflict_modulation`
- `unsafe_conflict_phase_approach_delta_y_mean`
- `unsafe_conflict_phase_inside_delta_y_mean`
- `unsafe_conflict_phase_release_delta_y_mean`
- `avoid_conflict_step_rate`
- `avoid_conflict_signed_w_mean`
- `avoid_conflict_delta_y_mean`
- `avoid_conflict_suppression_index`
- `stop_conflict_step_rate`
- `stop_conflict_delta_y_mean`
- `risk_rollout_f_mean / risk_rollout_a_mean / risk_rollout_s_mean`
- `risk_rollout_gap_f_min_as_mean`

关键判断：

- `conflict_suppression_index > 0` 表示高冲突中 Follow 权重被压低；`conflict_suppression_index < 0` 表示 learned-w 在该窗口增强 Follow。
- `conflict_selective_suppression > 0` 表示压低 Follow 主要集中在高冲突窗口；若为负，则解释为选择性 Follow-support，不得写成避障抑制。
- `relative_conflict_modulation` 与 `conflict_selective_suppression` 同号同义；正值表示高冲突窗口相对非冲突窗口更压 Follow，负值表示更支持 Follow。
- `priv_conflict_signed_w_mean` 只对 `PCR-learnedw / learnedw2` 解释；对 `PCR-geomw` 不作为机制结论。
- `PCR-geomw` 机制图看 `w` 与 `y_raw-y_eff`；`PCR-yonly` 机制图只看 `y_raw-y_eff`。
- 论文当前主张不写“learned-w 一定在危险冲突中抑制 Follow”；若 `CSI < 0` 且成功率/碰撞/跟随误差更好，应写成 learned-w 学到 progress-preserving Follow-support。
- 论文里“w 对 Follow/Avoid 仲裁真正有贡献”的核心证据优先看 `PCR-learnedw` 在 `task_success_rate / episode_collision_rate / follow_mae_m_mean` 上优于 `yonly / geom-w`，并用 `signed_w / delta_y / CSI` 解释它到底是在 avoid-support 还是 follow-support。

## 8.5 beta Pareto 字段

Table 3：风险预算扫描。

默认先用 learned-w 扫 `beta`。若 Day 2 learned-w checkpoint 尚未可用，就先用 `geom-w` 做 smoke Pareto；最终主文必须换成 learned-w 或最终主方法补齐。

beta 点：

- `0.0`
- `0.25`
- `0.5`
- `0.75`
- `1.0`

列：

- `beta`
- `row_progress_success_mean`
- `episode_collision_rate`
- `near_miss_rate_mean`
- `follow_mae_m_mean`
- `target_in_rgb_fov_rate`
- `cmd_jerk_mean` 或已有等价平滑字段

关键判断：

- `beta` 变大时，风险指标应下降或至少出现可解释趋势。
- 若 `beta=1.0` 让机器人基本停住，说明映射太保守，不能直接进主文。
- 若风险和效率都不随 `beta` 变化，说明后处理器证据不足，当天要先查 `--beta` 是否真正进入评测。

## 9. 图表清单

### Figure 1：任务与冲突示意

显示：

- 移动目标轨迹。
- 障碍行。
- 机器人穿越路径。
- `cmd_F` 与 `cmd_A` 的冲突。
- `HighConflict` 窗口。

### Figure 2：机制时间序列

字段：

- `priv_conflict_score`
- `risk_F`
- `risk_A`
- `signed_w` 或 `w`
- `y_raw`
- `y_eff`
- `y_eff-y_raw`
- `follow_dist`
- `clearance / near_miss`
- `target_in_rgb_fov`

验收：

- 高冲突窗口中 `w` 与 `y_eff` 有对应变化。
- 离开冲突区后 Follow 能恢复。
- 至少保留一个失败或边界案例用于限制讨论。

### Figure 3：安全-跟随权衡

横轴：

- `follow_mae_m_mean` 或 `follow_rmse_m_mean`

纵轴：

- `episode_collision_rate` 或 `near_miss_rate_mean`

验收：

- `learned-w` 应比 `yonly` 更靠近低碰撞、低跟随误差区域。
- `learned-w` 相比 `geom-w` 至少要有一个可解释的优势，否则主贡献证据不足。

### Figure 4：实机输入 dry-run

显示：

- RGB 目标框或目标 mask。
- occupancy map。
- safety/passable map。
- `cmd_F / cmd_A / y / w / y_eff / cmd_safe`。

## 10. Day 2 命令模板

下面命令用于 seed 1 先跑通全链路。`<PCR_CKPT>` 需要替换成对应策略。

### MoE-y

```bash
CUDA_VISIBLE_DEVICES=0 python3 legged_gym/scripts/eval_highlevel.py \
  --task s_pcr_line_avoid_basic \
  --mode teacher \
  --skill moe \
  --pcr_ckpt agents/moe_teacher_best_yonly.pt \
  --avoid_ckpt agents/avoid_best.pt \
  --lowlevel_ckpt agents/low_level_best.pt \
  --num_envs 64 \
  --episodes 128 \
  --seed 1 \
  --output_dir outputs/eval/ral_day2/yonly_seed1 \
  --avoid_stage_override 4 \
  --freeze_avoid_stage \
  --yonly \
  --dump_timeseries \
  --timeseries_episodes 64
```

### geom-w

```bash
CUDA_VISIBLE_DEVICES=0 python3 legged_gym/scripts/eval_highlevel.py \
  --task s_pcr_line_avoid_basic \
  --mode teacher \
  --skill moe \
  --pcr_ckpt agents/moe_teacher_best_yonly.pt \
  --avoid_ckpt agents/avoid_best.pt \
  --lowlevel_ckpt agents/low_level_best.pt \
  --num_envs 64 \
  --episodes 128 \
  --seed 1 \
  --output_dir outputs/eval/ral_day2/geomw_seed1 \
  --avoid_stage_override 4 \
  --freeze_avoid_stage \
  --wgeom \
  --w_tau 0.15 \
  --w_blend_mode multiply \
  --w_disable_gate_safe_clamp \
  --dump_timeseries \
  --timeseries_episodes 64
```

### learned-w

```bash
CUDA_VISIBLE_DEVICES=0 python3 legged_gym/scripts/eval_highlevel.py \
  --task s_pcr_line_avoid_basic \
  --mode teacher \
  --skill moe \
  --pcr_ckpt <LEARNED_W_CKPT> \
  --avoid_ckpt agents/avoid_best.pt \
  --lowlevel_ckpt agents/low_level_best.pt \
  --num_envs 64 \
  --episodes 128 \
  --seed 1 \
  --output_dir outputs/eval/ral_day2/learnedw_seed1 \
  --avoid_stage_override 4 \
  --freeze_avoid_stage \
  --wlearned2 \
  --w_blend_mode signed \
  --signed_w_lambda 0.30 \
  --signed_w_gamma_risk 0.15 \
  --signed_w_margin 0.05 \
  --w_disable_gate_safe_clamp \
  --dump_timeseries \
  --timeseries_episodes 64
```

`risk_memory` 不单列 Day 2 主表。若当前最强 learned-w 实现需要 `risk_memory`，表格行名仍写 `learned-w`，并在 `resolved_protocol.json` 和实验记录中标注 `risk_memory=true`。

### beta Pareto smoke

Day 2 若 `<LEARNED_W_CKPT>` 可用，优先用 learned-w 扫 5 个固定 beta 点；若不可用，先用 `geom-w` 做 smoke。这里没有一键 `--beta_sweep` 参数，先手动替换 `<BETA>`、`<PCR_CKPT>` 和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python3 legged_gym/scripts/eval_highlevel.py \
  --task s_pcr_line_avoid_basic \
  --mode teacher \
  --skill moe \
  --pcr_ckpt <PCR_CKPT> \
  --avoid_ckpt agents/avoid_best.pt \
  --lowlevel_ckpt agents/low_level_best.pt \
  --num_envs 64 \
  --episodes 64 \
  --seed 1 \
  --output_dir outputs/eval/ral_day2/beta_<METHOD>_<BETA>_seed1 \
  --avoid_stage_override 4 \
  --freeze_avoid_stage \
  <METHOD_FLAGS> \
  --beta <BETA> \
  --dump_timeseries \
  --timeseries_episodes 32
```

取值：

```text
<BETA> = 0.0, 0.25, 0.5, 0.75, 1.0
```

## 11. Day 2 结果检查

每组评测结束后必须检查：

```text
metrics.json
metrics.csv
timeseries.csv
resolved_protocol.json
```

控制台必须能看到：

```text
Row-progress success
Episode diagnostics full/strict/event/event+collision/collision/zero-progress-timeout
Priv-conflict step/score/CSI/CSS/RCM
Priv-conflict signed_w/delta_y in/out
Target FOV in/all/lost/maxLost/bearing p95[deg]
```

若出现以下情况，当天停止扩展实验，只修协议：

- `priv_high_conflict_step_rate` 接近 0，说明评测没有采到冲突窗口。
- `priv_obstacle_window_rate >= 0.95`，说明窗口太宽，机制图不可用。
- `target_in_rgb_fov_rate` 极低，说明跟随信息链断掉。
- 任一组缺 `metrics.json` 或 `resolved_protocol.json`。
- 任一组 checkpoint 被自动识别成错误 `w_mode`。

## 12. Day 1 完成标准

Day 1 完成后，后续两周默认遵守：

- 论文初稿先按完整 PCR-Net 框架写，终稿按证据强弱收缩。
- 主表采用两层对比：外部 comparative baselines + PCR internal ablations。
- 外部 baseline 至少包括 `Monolithic PPO` 和 `Reactive Safety Override`；时间允许时补 `DWA-inspired Local Rollout`。
- 内部消融固定为 `PCR-yonly / PCR-geomw / PCR-learnedw`。
- `learned-w` 是论文主贡献；`geom-w` 是强规则基线；`yonly` 是无冲突先验基线。
- `risk_memory` 归入 learned-w 的最终实现优化，不单列第四个 baseline。
- `beta` 必须做 `beta_sweep / Pareto`，否则风险预算贡献只能降级表述。
- 主表先做公平对照，不先做花哨扩展。
- 机制证据优先看 `HighConflict`，不只看 `risk_F bin`。
- 实机先过 D435i 静态检查和 dry-run，再冲真实低速跟随避障演示。
- 如果 learned-w 不明显优于 geom-w，就先不要硬写主贡献成立，必须回到训练或机制证据补强。
