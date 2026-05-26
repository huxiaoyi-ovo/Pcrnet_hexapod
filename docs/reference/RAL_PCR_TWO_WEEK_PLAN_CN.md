# RAL PCR 两周冲刺计划

## 1. 目标

在两周内把 PCR 主线推进到“基本具备 RAL 初稿与内部送审条件”的状态。这里的“基本做完”不是指所有 V7 扩展都完成，而是指论文主张、主表、机制图、实机输入链路和关键风险说明都能闭环。

当前论文初稿口径固定为：

```text
PCR-Net 用 y + w + beta + Command Post-Processor 在足式机器人人体跟随与局部避障冲突中实现前瞻式、可解释、可调节的运行时冲突消解。
```

两周内的证据重点优先放在 `learned-w`：证明学习式命令条件化冲突先验相比 `yonly` 和 `geom-w` 更能在真实高冲突窗口中自适应调制仲裁。当前 0.6 m/s 压力评测的工作假设是：learned-w 可能表现为 progress-preserving Follow-support，而不是简单的危险时压低 Follow。`risk_memory` 视为 learned-w 最终实现里的短时风险记忆优化，不单列为第四个 baseline。主表分两层：外部 comparative baselines 用于回答“为什么不是常规方法”，PCR internal ablations 用于回答“为什么 learned-w 必要”。`beta` 不做 learned beta 训练，但保留为风险预算与后处理器联动的论文组成，并尽量通过 `beta_sweep / Pareto` 给出支持证据。Student、完整视觉退化矩阵和大规模 OOD 仍作为后续扩展或补充讨论。

## 2. 判定标准

本计划以后续实验和实机的默认对照标准为：

- 参考文档：`docs/reference/论文写作自查清单.pdf`
- 论文级证据尺子：`HighConflict / CSI / RCM / signed_w / y_eff-y_raw / beta_sweep Pareto`
- 主任务：`s_pcr_line_avoid_basic`
- 补充泛化任务：`s_pcr_new --generalize`
- 主表对照：`Monolithic PPO / Reactive Safety Override / DWA-inspired Local Rollout / PCR-yonly / PCR-geomw / PCR-learnedw`
- 实机链路：`D435i depth -> local_map_2ch -> pcr_realplay.py dry-run -> 低速跟随避障演示`

投稿前最低通过线：

- 主表有同一设置下的公平对照。
- 每个主方法至少有多 seed 统计，优先 3 seeds。
- 机制图能显示 `HighConflict` 中 `w` 改变、`y_eff` 相对 `y_raw` 改变，并对应安全或进度收益。
- `beta_sweep` 至少覆盖 5 个点，形成可解释的风险-效率 Pareto 曲线；若不单调，必须写入失败分析并当天修正 beta 映射。
- 实机至少完成三组静态输入验收、真实输入 dry-run，并以真实低速跟随避障演示作为两周冲刺目标。
- 论文不夸大：如果 `learned-w` 不赢 `geom-w`，不要硬写主贡献成立，必须回到训练或机制证据补强。

## 3. 不做清单

两周内默认不做：

- learned beta 训练。
- Student 蒸馏。
- 完整 S1/S2/S6 大矩阵。
- TEB。
- 完整 ROS DWA 接入。
- MPC。
- SAC / HER。
- 端到端 RGB policy。
- 大范围奖励重写。
- 1 m/s 高速目标。
- 为了完整性加入新大结构。

这些内容不会直接提高当前 RAL 主结论说服力，除非主表结果已经完成且仍有明确空余时间。

## 4. 核心实验表

### Table 1：主性能对比

固定条件：

- task：`s_pcr_line_avoid_basic`
- avoid stage：固定最终难度，默认 `--avoid_stage_override 4 --freeze_avoid_stage`
- episodes：优先每 seed 128；时间不够时先 64 判断趋势
- seeds：优先 `1,2,3`
- low-level ckpt：`agents/low_level_best.pt`
- avoid ckpt：`agents/avoid_best.pt`

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

说明：

- 代码里如果使用 `learnedw2 + risk_memory`，论文和主表仍记为 `PCR-learnedw`。
- `risk_memory=true` 写入实验记录和 `resolved_protocol.json`，不新增主表行。

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

### Table 2：w 机制消融

只看 PCR 内部三条：

- `PCR-yonly`
- `PCR-geomw`
- `PCR-learnedw`

机制证据分两层：

- `priv_conflict_*`：row-command conflict，只说明障碍行窗口里 Follow 和 Avoid 候选动作发生分歧。
- `unsafe_conflict_*`：unsafe command conflict，用 `cmd_F / cmd_A / cmd_S` 三候选短时几何风险对比定义，同时要求 `risk_F` 高、`risk_F - min(risk_A, risk_S)` 高、命令方向分歧明显、目标仍可恢复；它定位危险 Follow 候选，但不预设 learned-w 必须压低 Follow。
- `avoid_conflict_*`：`C_avoid`，表示危险冲突里 Avoid 的任务效用高于 Stop/Slow；效用同时考虑风险、带上限的横移打开通路收益、Stop/Slow 前向保距收益和目标距离拉开代价，避免 Stop 因为风险最低天然吞掉所有 unsafe。
- `stop_conflict_*`：`C_stop`，表示 Stop/Slow 的任务效用不低于 Avoid，不能强行归因给 w，应在 beta 或安全后处理里解释。

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

解释口径：

- `priv_conflict_delta_y_mean = y_eff - y_raw`；负值表示偏向 Avoid，正值表示偏向 Follow。
- `conflict_suppression_index = -priv_conflict_delta_y_mean`；正值表示高冲突中压低 Follow，负值表示 learned-w 在该窗口支持 Follow。
- `conflict_selective_suppression = priv_non_conflict_delta_y_mean - priv_conflict_delta_y_mean`。
- `relative_conflict_modulation` 与 `conflict_selective_suppression` 同号同义；正值表示抑制主要发生在高冲突窗口，不能再按旧口径解释为“越负越好”。
- 图表中 `signed_w` 只用于 `learnedw2`；`geom-w` 图看 `w` 和 `y_raw-y_eff`，`yonly` 图只看 `y_raw-y_eff`。
- 当前主张不写“learned-w 一定抑制危险 Follow”；若 `CSI < 0` 且任务成功率、碰撞率、跟随误差更好，应解释为 learned-w 学到 progress-preserving Follow-support。
- “w 对 Follow/Avoid 仲裁本体有贡献”优先看 `PCR-learnedw` 在 `task_success_rate / episode_collision_rate / follow_mae_m_mean` 上优于 `yonly / geom-w`，再用 `signed_w / delta_y / CSI` 解释调制方向。

### Table 3：补充泛化

仅作为补充，不替代主表。

固定条件：

- task：`s_pcr_new`
- flag：`--generalize`
- rows：5 行
- speed：`[0.55,0.75]`
- row spacing：主场景 `0.85` 倍

最小行：

- `PCR-yonly`
- 最终主方法

## 5. 核心图

### Figure 1：场景与问题图

必须显示：

- 移动目标路径。
- 六足机器人穿越障碍行。
- Follow 命令和 Avoid 命令的冲突区域。
- `HighConflict` 的定义直觉。

### Figure 2：机制时间序列

选择一段代表 episode，绘制：

- `priv_conflict_score`
- `risk_F / risk_A`
- `signed_w` 或 `w`
- `y_raw`
- `y_eff`
- `y_eff-y_raw`
- `follow_dist`
- `clearance / near_miss`
- `target_in_rgb_fov`

验收标准：

- 高冲突窗口中 `y_eff` 相比 `y_raw` 发生明确改变。
- 离开高冲突后策略不长期压 Follow。
- 不能只展示成功片段，也要保留一个失败片段用于限制讨论。

### Figure 3：安全-跟随权衡图

横轴：

- `follow_mae_m_mean` 或 `follow_rmse_m_mean`

纵轴：

- `episode_collision_rate` 或 `near_miss_rate_mean`

理想结果：

- `PCR-learnedw` 相对外部 baselines 和 PCR 内部 ablations 更靠近低碰撞、低跟随误差区域。

### Figure 4：实机输入与 dry-run 图

显示：

- RGB 目标框或目标 mask。
- occupancy map。
- safety/passable map。
- `cmd_F / cmd_A / y / w / y_eff / cmd_safe` 日志片段。

## 6. 两周日程

### Day 1：锁定论文主张与评测协议

目标：

- 固定论文一句话问题、方法名、贡献范围。
- 固定主表和机制图字段。
- 确认所有评测命令都写到同一份实验记录中。

产出：

- 主张草稿。
- 主表字段清单。
- 评测命令模板。

验收：

- 论文初稿继续按 `y + w + beta + Command Post-Processor` 叙事。
- 两周实验证据优先证明 `learned-w` 相比 `yonly / geom-w` 的优势，同时保留 `beta_sweep / Pareto` 作为风险预算证据。
- 主表采用两层对比：外部 comparative baselines + PCR internal ablations。
- Student、完整 OOD 不进入两周主表。
- 每个主张都有对应实验。

止损条件：

- 如果方法名还在 `learned / learnedw2 / signed / PCR-Net++` 之间摇摆，当天必须统一：论文写 `learned-w`，代码兼容名可继续使用 `learnedw2 / --wlearned2`。

### Day 2：单 seed 评测全通

目标：

- 先跑通 `PCR-yonly / PCR-geomw / PCR-learnedw` 的 seed 1。
- 确认 `metrics.json`、`metrics.csv`、`timeseries.csv` 都能生成。

产出：

- 三组 seed 1 结果目录。
- 一份临时汇总表。

验收：

- 能看到 `CSI / RCM / signed_w / y_eff-y_raw / FOV`。
- 没有 NaN 或明显协议不一致。

止损条件：

- 如果某一组 checkpoint 或参数不匹配，只修加载/协议问题，不改奖励。

### Day 3：单 seed 结果判读

目标：

- 判断 learned-w 是否相对 `geom-w` 有继续投入价值。
- 找出最需要补跑的对照。
- 确定外部 baseline 的最小实现顺序。

产出：

- seed 1 判读记录。
- 一张临时机制图。

验收：

- 能回答：HighConflict 中 `signed_w / delta_y / CSI` 的真实调制方向是什么。
- 能回答：`y_eff` 相对 `y_raw` 是压 Follow 还是支持 Follow。
- 能回答：安全收益是否来自真实冲突窗口的自适应调制，而不是全程固定偏向某一专家。

止损条件：

- 如果 learned-w 完全不工作，不继续长训；当天回到训练或机制证据补强，不把 `geom-w` 直接改写成论文主贡献。

### Day 4-5：多 seed 主表

目标：

- 对最终候选组补齐 seeds `1,2,3`。
- 优先完成 PCR internal ablations 三组。
- 同步实现并 smoke `Reactive Safety Override`。

产出：

- 每组至少 3 seed 的 `metrics.json/csv`。
- mean/std 汇总。

验收：

- 主表能填数。
- 异常 seed 有日志解释。
- 所有组使用同一 task、episodes、stage、avoid ckpt、low-level ckpt。

止损条件：

- 如果 128 episodes 太慢，先用 64 episodes 得趋势；只对最终候选补 128 episodes。

### Day 5-6：外部 baseline 补齐

目标：

- `Monolithic PPO`：确认 checkpoint 或启动训练计划，优先用同观测直接输出 `[vx, vy, wz]`。
- `Reactive Safety Override`：实现最小规则版本并跑 seed 1。
- `DWA-inspired Local Rollout`：若前两项完成，再实现短时 rollout baseline；否则后置。

产出：

- 至少 `Monolithic PPO + Reactive Safety Override` 两个外部 baseline 的 seed 1 结果。
- 若时间允许，补 `DWA-inspired Local Rollout` seed 1。

验收：

- 外部 baseline 和 PCR 方法使用同一主任务、同一 seeds、同一 metrics。
- 若某外部 baseline 接入成本超过 1 天且不能明显提高论文说服力，先后置。

### Day 6：机制图定稿

目标：

- 生成 Figure 2 和 Figure 3 的论文候选图。

产出：

- `mechanism_priv_conflict_bins` 图。
- 代表 episode 时间序列图。
- 安全-跟随权衡图。

验收：

- 图能直接回答 `w` 是否在高冲突窗口起作用。
- 图中单位、图例、方法名统一。

止损条件：

- 如果机制图解释不清，不调方法；优先换代表 episode 或补充 phase 图。

### Day 7：主方法决策

目标：

- 判断 `learned-w` 是否已经能支撑论文主贡献。
- 判断外部 baseline 是否足够支撑主表。

决策规则：

- 若 `learned-w` 在 collision / near-miss / CSI / RCM 上优于 `geom-w`，且 Follow 不明显恶化，则主贡献成立。
- 若 `learned-w` 不稳定或只接近 `geom-w`，则不硬吹主贡献，优先补训练、补机制证据或收窄主张。
- 若外部 baseline 只有两个可用，最低主表按五组写；`DWA-inspired Local Rollout` 放补充或后续。

产出：

- 方法决策记录。
- 最终主表行名。

验收：

- 后续写作不再改主方法。

### Day 8：D435i 静态输入验收

目标：

- 用真实相机完成三组输入检查。

三组场景：

- 空地。
- 正前方箱子。
- 人旁边箱子。

产出：

- 每组截图。
- 每组 `occ_mean / safety_mean / observed_mean / inflated_blocked_mean`。
- 目标状态字段：`target_valid / target_lost / target_too_close / depth_invalid_ratio`。

验收：

- 空地不应被大面积判为障碍。
- 箱子应进入 occupancy，且膨胀后 safety 变低。
- 人不应被整框误删导致旁边箱子消失。
- 目标丢失时输出有效 safe state，不出现 NaN。

止损条件：

- 如果地板误判严重，先调相机外参和地面阈值，不上机器人运动。

### Day 9：ROS1 dry-run 接入

目标：

- 让 `real_pcr_input_check.py` 的真实输入进入 `pcr_realplay.py`。

产出：

- `/pcr/target_state`。
- `/pcr/local_map_2ch`。
- `pcr_realplay.py` dry-run 日志。

验收：

- dry-run 打印 `cmd_F / cmd_A / y / w / y_eff / cmd_safe`。
- 无 `--publish_cmd` 时绝不发运动命令。
- target lost、depth invalid、target too close 都能触发 safe output。

止损条件：

- 如果 ROS 字段不稳定，先保存离线输入再喂给 realplay，不急着开运动。

### Day 10：小速度实机演示

目标：

- 只做低速、安全、可停止的演示片段。

产出：

- 一段跟随-避障 dry-run 或低速实机视频。
- 同步日志。
- 安全停止记录。

验收：

- 实机行为与论文结论对应：冲突区域降低 Follow，避障优先。
- 目标丢失或深度异常时安全停止。

止损条件：

- 若输入质量没过 Day 8-9 验收，不做真实运动。

### Day 11：论文初稿骨架

目标：

- 写出 RAL 初稿结构。

产出：

- Abstract。
- Introduction。
- Method。
- Experiments 表格占位。
- Limitations。

验收：

- 摘要能用三句话讲清：问题、方法、提升点。
- 贡献列表每条都能指向一个表或图。
- 不出现超过当前证据的声称。

### Day 12：结果写入与失败案例

目标：

- 把主表、机制图、实机图写进论文叙事。

产出：

- Results 初稿。
- Failure cases 小节。
- Limitations 小节。

验收：

- 失败案例不削弱主结论，而是说明适用边界。
- 不能只报成功率，必须同时报 collision、near-miss、follow error、FOV。

### Day 13：补充材料与复现包

目标：

- 准备后续投稿需要的补充信息。

产出：

- 评测命令清单。
- checkpoint 与结果目录索引。
- seeds / episodes / task 设置。
- 实机输入验收截图索引。

验收：

- 新开终端可以按文档复跑主表。
- 表格数字能追溯到 `metrics.json`。

### Day 14：严格自查与投稿准备判断

目标：

- 按 `论文写作自查清单.pdf` 逐项打分。

产出：

- 自查结果。
- 剩余 P0/P1/P2。
- 是否进入 RAL 正式排版的判断。

验收：

- 若实验充分性仍不达标，不进入排版，继续补主表。
- 若只剩图表格式和文字问题，可以进入 RAL 模板整理。

## 7. 每日固定检查

每天结束前检查：

- 今天是否产出了可保存的结果目录或文档？
- 是否有新结果能支持论文主张？
- 是否发现会污染主表的协议不一致？
- 是否记录了 seed、episodes、ckpt、task、stage？
- 是否有任何结果依赖部署不可得输入？

## 8. 风险与处理

### R1：learned-w 不明显优于 geom-w

处理：

- 不硬吹 learned-w。
- 回到训练或机制证据补强，优先找 learned-w 为什么没超过强规则基线。
- 如果时间不够，主文只能收缩为诚实结果，不能把 `geom-w` 改写成本文主贡献。

### R2：安全提升来自全程压 Follow

处理：

- 必须看 `RCM` 和非冲突窗口 `y_eff-y_raw`。
- 如果低冲突也长期压 Follow，当前方法不能按“冲突先验”声称。

### R3：目标 FOV 断链

处理：

- 报告 `target_in_rgb_fov_rate` 和高冲突窗口 FOV。
- 如果 FOV 明显变差，实机必须加 target lost hard guard。

### R4：实机输入误判地板或目标

处理：

- 不上运动。
- 先调相机外参、地面阈值和目标 mask。

### R5：多 seed 方差大

处理：

- 报告均值和标准差。
- 找异常 seed 的失败阶段。
- 不选择只在单 seed 好看的方法作为主方法。

## 9. 最终提交前自评口径

按自查表，当前两周冲刺后的最低目标是：

- 选题与问题定义：达标。
- 方法与创新点：达标或接近达标。
- 实验与验证：至少主表和机制图达标。
- 写作与逻辑结构：完成初稿。
- 图表规范：论文候选图达标。
- 选刊与投稿准备：具备 RAL 内部送审条件。

如果第 14 天仍缺主表多 seed 或实机 dry-run，不建议进入正式投稿排版。
