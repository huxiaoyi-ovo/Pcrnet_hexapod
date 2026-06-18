# TODO Log

完成后请改为 `[x] ~~条目内容~~`。
后续“计划好的代码改动方案”请先在此处写入精简版，便于新开终端直接接手。

规则：

- 仅记录“重大改变/思路调整”：架构变更、训练主线调整、评测协议变化、跨模块关键改动。
- 小修小补默认不记录：局部 bugfix、轻微参数微调、文案/注释修改、格式整理。
- 短期 TODO（动态滚动）只保留最近 20 天；超过 20 天（以日期标题 `## YYYY-MM-DD` 为准）的段落直接删除。
- 中长期 TODO 只能在你明确同意后才能新增条目。

## 中长期 TODO（置顶：V7 训练与论文主线）

> 这部分保持长期不变：只追加，不随短期进展频繁挪动；完成项同样用 `[x] ~~...~~` 标记。

### Stage 0.5：训练架构定稿（先于所有训练）

- [ ] [P0] 固化语义契约：`y/y_eff` 为 Follow 权重；`w` 为冲突强度；`y_eff_raw=(1-λ)*y+λ*(1-w)`；`cmd_base=y_eff*cmd_F+(1-y_eff)*cmd_A`
- [ ] [P0] 固化 gate 链路 ABI：gate 训练/推理统一产出并记录 `cmd_F/cmd_A` 候选与 `risk_F/risk_A`（命令条件化 w 的输入来源）
- [ ] [P0] 固化后处理 ABI：`CommandPostProcessor(cmd_raw, beta, clearance)`；统一 `post_info` 透出字段用于日志与论文图
- [ ] [P0] 固化评测协议：固定 seeds/episodes/β sweep 点位/记录 jerk/near-miss/switch-rate（不再只看 success）

### Stage S：训练场景设计优化（为证据链服务）

- [ ] [P0] S0 平地移动目标跟随（预训练底座）：1m 跟随 + 视野中心窗口约束 + 丢失 K 步 reset
- [ ] [P0] S1 门洞走廊强化为“窄缝主演示”：门洞宽度 curriculum、门洞位置/出生/目标段互斥契约、关键冲突片段可复现
- [ ] [P0] S1-moving 强制冲突版：目标必须穿门洞（gate-by-gate 阶段机），确保稳定产生 Follow vs Avoid 冲突段
- [ ] [P0] S2 柱阵森林分布与 curriculum：Poisson/聚类/通道型 2–3 模式 + 密度课程 + 可复现统计
- [ ] [P0] 结构化 hold-out：在 S1/S2 内留出“组合 OOD”（参数组合训练不出现，评测专测）
- [ ] [P0] 新增路线/时机类指标口径（服务 w/β 贡献）：门洞前“提前决策时机”、事件对齐曲线等

### Stage A'（过渡）：β 仅作为评测/部署旋钮（先产出 Pareto）

- [ ] [P0] β→约束族参数映射落地（safe_dist/free_dist/max_cmd/max_delta/risk_gain），支持 `--beta_sweep` 固定权重扫 Pareto
- [ ] [P0] Pareto 单调性与“β=1 不停住”验收

### Stage A：专家（Follow/Avoid）训练（冻结底层）

- [ ] [P0] Follow expert：目标效率/跟随稳定；Avoid expert：安全/风险；两者共享统一奖励骨架（仅权重不同）
- [ ] [P0] 专家验收：S1/S2 固定种子评测 + β sweep（在 A’ 的后处理联动下）
- [ ] [P1] （可选）专家内风格条件：将 β 作为 expert 观测输入，验证“同权重随 β 连续变化”优于仅靠后处理联动

### Stage B：PCR-Net++ gate（y + w + β + y_eff + 条件融合）

- [ ] [P0] gate 基线：y-only（复现 late switch / chattering）
- [ ] [P0] y+β(link) 与 β(no-link) 对照（证明“机制化接口”）
- [ ] [P0] 加 w（命令条件化）：输入包含 `cmd_F/cmd_A` 与 `risk_F/risk_A`；做 w 退化检测（w≠clearance）
- [ ] [P0] w 预测性训练：实现 `simulate_command_trajectory`/RiskAlong-lite（向量化、GPU 友好、horizon=0.8s）生成伪标签并加入 w_aux loss
- [ ] [P0] 条件融合 + 平滑/滞回（推理启用为主）：稳定性指标必须改善（jerk/near-miss/switch）
- [ ] [P0] 主证据评测：IID(S1/S2) + OOD(S6) + β sweep Pareto + 事件对齐图（y/w/β）
- [ ] [P1] （可选）多维 w：spatial/proximity/severity + fusion（单帧输入），提供可解释性图与消融

### Stage C：Student 蒸馏（视觉部署）

- [ ] [P0] Teacher/Student 边界与蒸馏目标定稿（cmd + 可选 y_eff/β），并做视觉退化矩阵 + OOD
- [ ] [P1] 消融：只蒸馏 cmd vs cmd+β（证明 β 对部署一致性与可控性的贡献）

---

## 短期 TODO（动态滚动：下面按日期维护）

## 2026-06-15 RA-L 投稿前补救实验代码

- [x] ~~[P0] 完成 Additive-Fusion eval-only 对照：直接相加 `u_F` 与 lateral-projected `u_A`，共用 post-processing 和诊断汇总；必须记录 command 被 clamp 的比例，用于回应“正交命令为何不用直接相加”的质疑；若时间允许，额外支持 `normalized_additive_fusion` 作为 appendix 备选，但主表先只放 `additive_fusion`。~~
- [ ] [P0] 正式补跑 Additive-Fusion 三速三 seed：smoke 显示 0.60 下 `Task Success=0.50 / Collision=0.50 / Row-progress=0.85 / Follow MAE=0.326`，说明直接相加能推进和跟随但安全不稳；下一步按主表口径跑 `stage4 × speeds 0.35/0.50/0.60 × seeds 1/2/3 × 128 episodes`，用于决定它进入主表还是作为 Direct command fusion 消融单独呈现。
- [x] ~~[P0] 升级 Target-aware Velocity-Space Search 对照：保持不接完整 ROS DWA、不宣称完整 DWA；从单步启发式命令枚举升级为 DWA-style 速度空间局部规划器，必须加入 1.0-1.5s 多步 rollout、body-frame 位姿积分、footprint collision 直接 infeasible、安全 fallback、`compute_time_ms / feasible_count / infeasible_count / fallback_rate / min_predicted_clearance / best_cost` 诊断，并继续保留 validation tuning protocol 与 `cmd_search_raw / cmd_search_after_filter / cmd_safe` 记录。~~
- [ ] [P0] 继续收敛 Velocity-Search 外部对照定位：当前已确认它存在两端失败模式，`scale/safe` 口径偏安全但不推进，`soft/aggressive` 口径能推进但 0.35 也高碰撞；下一步停止无限调参，只做 `safe / balanced / tracking` 三个 preset 在 `validation_layout` 上的有限验证（speed=0.35/0.60，seed=1，episodes=32，num_envs=32），用于判断是否存在可进主表的 balanced 点；若三者都无法同时满足低碰撞与有效 row-progress，则将 Velocity-Search 定位为 diagnostic external local-planning baseline，不进入主表。
- [ ] [P0] 完成 Learned-w 推理期因果消融：同一 checkpoint 下支持禁用 `Delta y_w`、`Delta y_r`、`risk_memory`，并保存 raw/used delta，生成 inference ablation 表。
- [ ] [P0] 新增 layout 分层评测：保留主表场景不变，明确区分 `validation_layout` 和 `heldout_test`；validation 只用于 Velocity-Search 调参，heldout_test 只用于最终评估，并输出独立 held-out 表。
- [ ] [P0] 完成实机 trial 统计分析：按 method/layout/trial 保存 session 与 metrics，新增 bag 分析脚本输出 trial-level 和 summary CSV；必须支持 `manual_collision / manual_target_lost / manual_final_row_clearance / manual_intervention / manual_success / failure_reason`，最终 real-robot success 以人工标注为准，自动指标只作为辅助。
- [ ] [P0] 集成新增对照到 summary/final table 输出：统一 method tag、metadata 和最终表格，保证新增结果自动汇总且不污染已有五方法主表。

## 2026-06-15 实机双侧真实反馈就绪与陈旧反馈停机

- [x] ~~[P0] 让 CAN 状态帧提供逐电机真实回包计数与时间；B 初始化前左右两侧必须各自拿齐 9 个电机新回包并完成双侧握手，运行中任一电机反馈陈旧或 Pos_vel 期间出现共享故障时立即中断并锁存停机，禁止全零/旧状态冒充有效反馈。~~

## 2026-06-11 实机手柄持续控制串口负载修复

- [x] ~~[P0] 移除手柄路径 `/sita_des` 的 0/8/8 ms 三连发，改为与 PCR 相同的单次均匀发布；CAN 侧只保留最新命令，并在首次串口 I/O 故障后停止继续写入，避免连续推杆时突发负载击穿 USB 串口后形成错误风暴。~~
- [x] ~~[P0] 将任一侧串口 I/O 故障同步到左右 CAN 节点与 `run_agent2`：健康侧只执行一次 `Disable`，故障会话内拒绝后续手柄/PCR 运动命令，必须重启控制节点并重新 B→Y 才能恢复。~~

## 2026-06-11 PCR Fig.6 轨迹图数据闭环

- [x] ~~[P0] 为最终论文 Fig.6 增加 stage4 三速度五策略轨迹图所需数据链：eval timeseries 必须保存 robot/target 逐帧位置、真实障碍圆柱位置与 episode 终止原因；最终出图脚本必须先检查字段完整性，缺字段时停止并提示补跑 timeseries-only eval。~~
- [x] ~~[P0] 将最终论文 Fig.6 收口为 0.60 m/s 单张代表性轨迹图：从主表原始 eval timeseries 中按同一障碍布局选轨迹，Learned-w 必须选成功轨迹，baseline 尽量选 2-3 条 collision 和至少 1 条 lost/timeout，并在 source CSV 中记录每条轨迹的真实 run、episode 与终止原因。~~
- [x] ~~[P0] 增加 Fig.6 轨迹池审计脚本：扫描 `agents/` 下所有原始 timeseries，统计 0.60 m/s 下各方法 success/collision/lost/timeout 与布局签名，允许非同一布局但必须避免明显镜像拼图，为最终轨迹图选择提供证据。~~
- [x] ~~[P0] 让最终 Fig.6 出图脚本可直接读取轨迹池审计候选 CSV，按指定 rank 精确加载五条真实轨迹，避免重新自动选择导致图与审计结果不一致。~~

## 2026-06-10 PCR 论文 v3 图表收口

- [x] ~~[P0] 将最终论文图表生成入口升级到 `PCR-Net_figure_table_plan_v3.md` 口径：Table I/II/III 输出 markdown + LaTeX booktabs，接入重训 Risk-only，新增 Fig.3 主性能图、Fig.4 learned-w 机制图和 A1-A5 附录表；字段只能来自现有代码、配置和 eval 输出，不填猜测值。~~

## 2026-06-10 PCR 实机短时障碍空间记忆

- [x] ~~[P0] 在实机 `local_map_2ch` 膨胀前加入按真实时间衰减的短时占据记忆，保留遮挡期间的障碍横向位置；该层只稳定 PCR 观测，不发布控制命令，首轮暂不加入避障方向锁定。~~
- [ ] [P0] 录制带 `/pcr/raw_occ_map`、`/pcr/memory_occ_map` 和 `/pcr_realplay/debug` 的实机 bag，对比高风险阶段 `cmd_x` 反转次数、障碍保留时间与通过后的释放时间，再决定是否加入 `avoid_latch`。
- [x] ~~[P0] 增加手柄触发的实机实验录制：同一会话由相机进程异步保存 `draw_debug()` 原始 viewer、帧时间戳和运行元数据，ROS bag 仅保留关键数值与地图，并从 `/pcr_realplay/debug` 导出 `risk_F / risk_A / y_eff / w / cmd` 论文曲线。~~

## 2026-06-09 PCR Risk-only 机制统计口径补齐

- [ ] [P0] 补齐 eval/summarize/final-table 中 `C_unsafe` 窗口下的 `delta_y_w / delta_y_r / delta_y_total` 拆分统计，用旧 Risk-only checkpoint 重新 eval 判断显式风险差项是否在冲突窗口真实生效，避免 Table II 只看 `C_avoid` 窄窗口导致误读。
- [ ] [P0] 将 `Risk-only` 覆盖为从头训练 baseline：训练时 actor 只输出 `y`，控制只使用 `Delta y_r=gamma(risk_A-risk_F)`，删除旧 learnedw2 eval-only 切除身份，Table II 的 `Risk-only` 统一解释为 trained-from-scratch 对照。

## 2026-06-05 PCR 实机接管安全修正

- [x] ~~[P0] 修正 PCR 授权后的零速度接管问题：`run_agent2.py` 只缓存有效 PCR 速度命令，PCR 全零命令不再触发 50Hz `Traj_follow`；同时 `joy_ctrl.cpp` 不再让未被底层使用的 `z_vec` 轴触发手柄运动发布，避免按 14 后或 z 轴噪声导致低层进入零速步态。~~

## 2026-06-04 PCR 论文最终图表产出

- [x] ~~[P0] 新增最终论文图表生成入口：读取已完成的 internal / Risk-only / Rule-Override / Mono-PPO stage probe 评测结果，统一输出 Table I/II/III 与 Fig.4 所需 CSV、Markdown 和曲线图，避免继续手工拼表污染论文结果。~~

## 2026-06-02 PCR 实机命令选择安全层

- [x] ~~[P0] 新增 `/usr_command_mux`：键盘/手柄手动输入优先，PCR 授权且消息新鲜时才转发，PCR 丢目标/超时/零运动时输出 `set_init=True` 站立命令，避免多个节点抢 `/usr/command` 或无目标时继续触发底层 `Traj_follow`。~~
- [x] ~~[P0] 将 PCR 实机控制权选择下沉到 `run_agent2.py`：手柄/键盘发布 `/usr/command_manual`，PCR 发布 `/usr/command_pcr`，由底层根据 `set_init / disable_torque / change_mode / timeout` 选择当前命令，避免外部 mux 反复制造 `Pos_vel / Traj_follow` 模式切换。~~

## 2026-06-02 Mono-PPO 诊断指标补强

- [x] ~~[P1] 扩展高层 eval 诊断输出：新增 L1/L2/L3 成功率分解、目标 FOV/bearing 汇总落表、goal-command 相关性与横向命令偏置/熵指标，用于量化 Mono-PPO 是否学到目标条件化跟随，而不是只学到固定横向避障捷径。~~

## 2026-06-01 PCR 外部 baseline 收口

- [ ] [P0] 完成 `Risk-only` eval-only 因果切除消融：加载同一个 `learnedw2` checkpoint，保留 policy 输出与 `signed_w` 诊断，但控制时强制 `delta_y_w_used=0`，只保留 `delta_y_r=gamma(risk_A-risk_F)`，验证 learned-w 通道是否提供手写风险差之外的增益。
- [ ] [P1] 保留 `Mono-PPO` 外部 baseline 的公平训练曲线：当前修复后已学会前进跟随，但完整任务仍易碰撞；优先用 Row-progress / step collision / Follow MAE 解释部分能力，不新增 Mono 专属专家辅助或专属任务奖励。

## 2026-05-29 PCR src_real 实机 ROS 部署收口

- [x] ~~[P0] 补齐 `src_real/interface` 的 PCR ROS 运行外层：完善 `joy_command` 消息编译依赖、提供 PCR 专用 launch、保持 `run_agent2.py / joy_command.msg / CAN` 已验证实机控制链路不改，默认 dry-run，只有显式开关才发布 `/usr/command`。~~

## 2026-05-28 PCR 主表多 seed 评测批量入口

- [x] ~~[P0] 补齐 PCR 主表评测批量入口：固定 `0.35/0.50/0.60m/s × yonly/geomw/learned-w × 多 seed`，保留每组原始 `metrics.json/csv/timeseries`，同时输出论文主表单 seed 行与 mean/std 汇总，避免继续手工拼表污染评测口径。~~

## 2026-05-26 RAL 投稿自查标准与两周冲刺计划

- [x] ~~[P0] 将《论文写作自查清单》作为后续 PCR 实验、实机与论文准备的统一参考标准，并在 `docs/reference/` 中保存原始 PDF 与两周 RAL 冲刺计划；后续默认以 HighConflict 机制证据、主表公平对照、多 seed 统计、D435i 输入 dry-run 和实机安全验收判断 RAL 准备度。~~
- [x] ~~[P0] 完成 Day 1 论文主张与评测协议收口：新增 `docs/specs/PCR_RAL_DAY1_CLAIM_EVAL_PROTOCOL_CN.md`，固定当前 RAL 主张为 command-conditioned conflict prior `w` 改善真实 HighConflict 窗口仲裁，并写清主表、机制表、图表与 Day 2 评测命令模板。~~
- [x] ~~[P0] 曾将 `docs/reference/pcr_net_ral_cn.md` 按 Day 1 实验口径误收窄为 `w` 单点主张；该口径已撤回，只保留为两周证据补强标准，不再替代 PCR-Net 初稿的完整论文叙事。~~
- [x] ~~[P0] 恢复 `docs/reference/pcr_net_ral_cn.md` 的论文型口径：中文初稿继续以 PCR-Net 的 `y + w + beta + Command Post-Processor` 为主张，两周实验计划只作为证据补强与终稿收缩标准，不替代论文叙事本身。~~
- [x] ~~[P0] Day 1 口径最终确认：论文主张先完整写、终稿按证据收缩；主场景用 `s_pcr_line_avoid_basic` 产数；贡献排序为 PCR-Net 框架第一、`w` 第二、`beta` 第三；两周内必须产出 `beta` Pareto；实机目标冲真实低速跟随避障演示；Day 1 产物同时修协议文档和论文初稿标注。~~
- [x] ~~[P0] Day 1 论文对照口径进一步收口：主表三条为 `yonly / geom-w / learned-w`，其中 `learned-w` 是论文主贡献，`geom-w` 是强规则基线，`yonly` 是无冲突先验基线；`risk_memory` 归入 learned-w 的最终实现优化，不单列第四个 baseline。~~
- [x] ~~[P0] Day 1 主表口径升级为两层对比：外部 comparative baselines 至少包括 `Monolithic PPO`、`Reactive Safety Override`、`DWA-inspired Local Rollout`，内部 PCR ablation variants 为 `PCR-yonly / PCR-geomw / PCR-learnedw`；若时间不足，最低主表保留 `Monolithic PPO + Reactive Safety Override + PCR-yonly + PCR-geomw + PCR-learnedw` 五组。~~
- [x] ~~[P0] 修正 PCR eval 机制指标解释口径：`relative_conflict_modulation` 改为正向指标，并新增 `conflict_selective_suppression`；后续论文表格统一按 `CSI > 0` 表示高冲突压 Follow、`CSS/RCM > 0` 表示抑制主要集中在高冲突窗口，避免把正确数据解释成相反结论。~~
- [x] ~~[P0] 修正 PCR 机制图字段口径：`signed_w` 只在 `learnedw2` 图和论文解释中使用；`geom-w` 图改画原始 `w` 与 `y_raw-y_eff`，`yonly` 图只画 `y_raw-y_eff`，避免在非 signed-w 方法上出现全零 `signed_w` 造成误读。~~
- [x] ~~[P0] 优化 PCR 机制图显示比例：Conflict Modulation 子图改为自适应 y 轴以显示小量级调制差异，Bin Support 改用 steps/episodes 双轴显示，避免真实差异被固定 0-1 或单 count 轴压扁。~~
- [x] ~~[P0] 修正 PCR eval 冲突证据分层：保留 `priv_conflict_*` 作为 row-command conflict，只证明障碍行窗口内 Follow/Avoid 候选动作分歧；新增 `unsafe_conflict_*`，要求 `risk_F` 高、`risk_F-risk_A` 高且命令分歧明显，后续“危险冲突中抑制 Follow”的论文结论必须优先由 unsafe 指标支撑。~~
- [x] ~~[P0] 升级 PCR eval 危险冲突证据链：`unsafe_conflict_*` 改为 `cmd_F / cmd_A / cmd_S` 三候选短时风险对比，并新增 `avoid_conflict_* / stop_conflict_*`；后续证明 w 贡献优先看 `C_avoid` 上的 `signed_w < 0`、`delta_y < 0`、`CSI > 0`，`C_stop` 不强行归因给 w。~~
- [x] ~~[P0] 根据 0.6 m/s 压力评测修正 learned-w 论文口径：当前主张改为 adaptive / progress-preserving arbitration，不再预设 learned-w 必须在高危冲突中压低 Follow；若 `signed_w > 0、CSI < 0` 但任务成功率、碰撞率、跟随误差更好，解释为 learned-w 学到 Follow-support 调制。~~
- [x] ~~[P0] 修正 `C_avoid/C_stop` 评测定义：由 risk-only 改为 utility-based，效用同时考虑候选命令风险、前向推进和目标距离拉开代价，避免 Stop 因风险最低天然吞掉所有 unsafe conflict。~~
- [x] ~~[P0] 修正 `C_avoid/C_stop` utility 二次审查问题：`cmd_A` 是横移避障，不能用前向分量衡量 Avoid 任务收益；正式评测改为风险 + 带上限的 Avoid 横移打开通路收益 + Stop/Slow 前向保距收益 + 目标距离拉开代价，并标注旧 `unsafe_conflict_avoid_stop_margin` 为 legacy unused。~~
- [x] ~~[P0] 修正 D435i 实机输入 difficulty 口径：`local_map_2ch[1]` 是 passable/safety，不再按平均 safety/occupancy 粗略估难度；`real_pcr_input_check.py` 与 `pcr_realplay.py` 同步改为最近 blocked 距离主导 + 距离加权 blocked 密度，保证同一障碍越靠近机器人时 `nearest_blocked_m` 降低、`near_risk/actor_difficulty` 升高。~~
- [x] ~~[P0] 修正 D435i 实机 safety/passable 几何口径：默认按训练侧 `fixed_layout_robot_clearance=0.27m` 做障碍膨胀，并保留 `body_width=0.25m、body_length=0.40m、swing_abduction=0.15m` 几何参数作为可选覆盖；`real_pcr_input_check.py` 输出 cell、clearance 与 inflation cells，避免稀疏深度覆盖或过大膨胀把 safemap 弄成全黑。~~
- [x] ~~[P0] 修正 D435i 实机可见区与目标链路诊断：`real_pcr_input_check.py` 不再用稀疏深度点覆盖定义 policy visible，而是按当前 RealSense 对齐后相机内参生成稠密可见区；unknown 只以软成本进入 difficulty，并新增 bbox/depth/self-mask/膨胀前后障碍数量诊断，定位 target_lost 与全阻塞地图来源。~~

## 2026-05-22 PCR w 真实冲突证据尺子

- [x] ~~[P0] 将 PCR eval 的 w 机制统计从仅依赖 `risk_F / conflict_score` 扩展到 privileged obstacle-window 冲突口径：用障碍交互窗口、Follow 前进压力与 Avoid 横移压力生成 `priv_conflict_score / mask / phase`，先修正机制图和评测证据，不改当前策略训练主线。~~
- [x] ~~[P0] 收紧 PCR eval 的 privileged conflict 尺子：避免多行当前 row 交互窗口覆盖整段 episode，补 phase 内 `w / y_eff-y_raw` 统计与过宽窗口警告；该 row 真值尺子只用于评测诊断和训练期监督讨论，不进入 actor 输入。~~

## 2026-05-21 PCR learnedw2 终版 w 公式验证

- [ ] [P0] 新增 `learnedw2` 作为独立 signed-w 公式实验线：保留旧 `learnedw` 单向 suppression 语义，`learnedw2` 使用 `w_s=2w-1` 的 signed conflict prior 与 `risk_A-risk_F` 安全修正；同步 `w_aux` 的 row/global 高置信样本，先做短训一致性验收再决定是否开长训。

## 2026-05-20 PCR realplay 实机部署入口

- [x] ~~[P0] 新增 `pcr_realplay.py` 作为 learned-w 策略实机部署主入口：支持 fake 输入、ROS1 dry-run、策略加载、输入口径检查、命令限幅与 ROS cmd 输出，默认不发运动命令。~~

## 2026-05-20 PCR sim2real 输入对齐

- [x] ~~[P0] 新增 RealSense/YOLO 到 PCR 策略输入的检查脚本：固定相机正前方对应机器人 `+Y`，按 D435i 下俯角生成 `goal/follow_goal=(x_right,y_forward)`、`local_map_2ch` 与 `actor_difficulty`，为后续 ROS1 数据传递接入做口径验收。~~

## 2026-05-20 PCR learned-w 实机可得输入收口

- [x] ~~[P0] 从 learned-w actor 输入中移除真实 `row_not_released`，保留固定 0 占位维度；真实 `row_not_released` 只允许用于 w_aux 监督、日志与诊断，避免仿真特权信息污染实机策略。~~
- [x] ~~[P0] 将 MoE 解析 follow expert 收口到 `state + goal_buf` 计算，不再优先读取仿真 `target_world`，避免 `cmd_F` 特征携带实机不可得的目标真值。~~
- [x] ~~[P0] 审清 follow / avoid / PCR 各 actor 输入边界：部署可得量才能进 actor，训练阶段专用量只允许留在 critic、监督标签或日志。~~

## 2026-05-19 PCR learned-w 行内释放安全收口

- [x] ~~[P0] 将近期 PCR w 主线切到 `learned-w + row_not_released + row-aware w_aux`：保持当前 `w` 单向压 follow 语义和 `y_eff = y_raw * (1 - w)` 不变，先解决当前障碍行未释放前过早追目标导致的侧向碰撞；暂缓双向 follow-support 公式、learned beta 与大范围奖励改动。~~

## 2026-04-02 场景命名体系收口

- [x] ~~[P0] 将项目主线场景体系收口为两类命名：`s_` 仅表示训练场景，`e_` 仅表示实验场景；除这两类外的旧场景名默认废弃，不再作为主线讨论、命令示例与结果汇报依据~~

## 2026-04-02 PCR 第一场景：固定避障几何 + 直线移动目标

- [x] ~~[P0] 新增 `s_pcr_line_avoid_basic`：场景几何、固定模板、课程与 `s_avoid_basic` 完全一致，只增加一条固定 `x_line` 的直线移动目标，用于第一版 PCR 跟随-避障协调训练~~

## 2026-04-07 PCR 第一场景奖励重构：从双任务相加改为联合结果评价

- [x] ~~[P0] 在 `s_pcr_line_avoid_basic` 中新增独立 PCR 奖励分支：以 `R_core(前向进度×follow_quality)` 为主信号，加入解析 `conflict` 的 Gate 辅助项与逐行 `gap_success` 里程碑，同时显式关闭旧的 `approach/follow_outside` 双任务叠加口径，并将 row 系列降级为小整形项~~
- [x] ~~[P0] 将 `s_pcr_line_avoid_basic` 的 PCR 奖励配置收口为单点全覆盖：所有参与训练、终止或显式关闭的奖励系数都在该场景配置里显式声明，并在训练入口对关键配置缺失直接报错，避免静默继承父类或默认值~~
- [x] ~~[P0] 将 PCR 奖励定稿为最终版分层结构：`R_core + R_gate_aux + R_gap_success + R_shape + R_hard`，并强制 PCR 分支只绑定 `s_pcr_line_avoid_basic` 场景名，同时在 wrapper `_reset_idx` 中显式清零 `row_success_flags`~~
- [x] ~~[P0] 修复 PCR 指标口径：`conflict/y_high/y_low/corr` 只统计当前 row 有效步骤；`TargetFinishRate/FollowLostRate` 改为按 episode 结束事件统计，避免短训验收被 step 平均静默污染~~
- [x] ~~[P0] 将训练输出与 best checkpoint 选择从 `avoid` 口径中分流：`avoid` 保持原输出；`pcr` 切到独立 `PCR/*` 输出与 PCR 专属 online selection metric，避免 TensorBoard 与 best ckpt 被 `avoid` 指标误导~~

## 2026-04-08 PCR 取消角度恢复机制

- [x] ~~[P0] 在 `s_pcr_line_avoid_basic` 的训练与 `play` 共用高层逻辑中禁用 `rotate_only` 角度恢复接管，避免额外硬规则替 Gate 做仲裁；保留只读状态字段用于确认 PCR 下该机制恒不触发~~

## 2026-04-08 PCR gap_success 记账改为上一行释放事件

- [x] ~~[P0] 将 `s_pcr_line_avoid_basic` 的 `R_gap_success` 从“直接使用当前动态 `current_row_idx` 记账”改为“基于上一行释放/切换事件记账”，继续沿用 avoid 的行几何定义，但避免当前行过早切换导致逐行里程碑长期为零~~
- [x] ~~[P0] 将 PCR 的逐行成功拒绝条件从“释放瞬间无违规”收口为“整行生命周期内无 collision/band fail”，并新增 PCR 真正 success 指标用于 best checkpoint 选择，避免 `target_finish` 结束事件继续冒充任务成功~~
- [x] ~~[P0] 将 PCR 的 train 终端输出收口为与 avoid 相同的版式与频率，只替换成 PCR 语义参数，避免后续短训读取体验和节奏与 avoid 分叉~~

## 2026-04-15 PCR 融合口径收口：avoid 只提供横移

- [x] ~~[P0] 在 PCR 的 `resolve_moe_gate_pcr()` 中将 `cmd_A` 收口为横移-only：保留 lateral，清零 forward 与 yaw，使 follow 重新主导目标距离与前向节奏，同时训练与 play 共用同一融合口径~~

## 2026-04-22 episode 级 eval 指标统一收口

- [x] ~~[P0] 在 `train_highlevel.py` 中新增 `EpisodeMetricsTracker` 与 `eval/` 日志统一汇总口径：基于现有 rollout 字段按 episode 统计跟随稳定性、碰撞安全、任务成功、软融合质量与 PCR 核心贡献验证指标，用于后续 baseline、论文结果表与实机前诊断统一对齐~~
- [x] ~~[P0] 将 `EpisodeMetricsTracker` 的接线从“每步逐 env Python 循环 + .item()”收口为“张量累计 + done 时按 env 汇总”，避免在 `2048/4096 env` 下静默拖慢训练吞吐，并同步去掉 `locals()` 字段检测与 `scipy` 额外依赖~~

## 2026-04-24 PCR eval 统计口径最终收口

- [x] ~~[P0] 将 `eval/` episode 统计收口为训练成功语义优先：`success_rate` 优先读取环境 `success_mask`，跟随距离达标仅作为独立诊断项，同时避免 `follow_dist` 缺失继续写入假低误差或污染 tradeoff 指标~~

## 2026-04-24 PCR 成功与终止口径改为机器人主导

- [x] ~~[P0] 将 `s_pcr_line_avoid_basic` 的回合成功从”目标到终点触发结束后检查机器人进度”改为”机器人越过最后一行 + 无碰撞 + 跟随距离在 band 内”才成功；移动目标继续沿直线前进，不再因目标到线提前切断回合~~

## 2026-04-27 PCR 目标速度与近障 yaw 抑制

- [x] ~~[P0] 将 `s_pcr_line_avoid_basic` 的移动目标速度提高到 `0.35m/s`，并在 PCR 冲突/近障阶段加入轻量 `pcr_yaw_suppress` 奖励：`obstacle_risk` 优先使用 moe 路径传入的 `gate_diag["risk_F"]`（跟随专家命令方向的风险），fallback 依次为 `forward_clearance`、`front_half_nearest_obs`、`pcr_conflict`，只惩罚近障时过大的 yaw，减少为了朝向目标而侧身撞障碍~~

## 2026-05-09 PCR 最后一行判定线口径修正

- [x] ~~[P0] 将 `s_pcr_line_avoid_basic` / `s_avoid_basic` 的 final cross line 与 PCR target end diagnostic 统一到实际 scaled row_y 的最后一行，避免 `avoid_fixed_row_y_spacing_scale=1.5` 后仍使用 raw `avoid_stage*_last_row_y` 导致提前判定越过最后一行~~
- [x] ~~[P0] 新增 PCR 成功回合平均跟随距离指标，区分全 rollout `FollowDistMean` 与成功 episode 内部的平均距离，避免失败样本拉高后误判演示距离~~

## 2026-05-18 PCR learned-w command-conflict 主线

- [x] ~~[P0] 新增 `w_mode=learned` 作为第三个 PCR 对照：GatePolicy 输出 `y` 与 `w` 两维，`w` 的输入显式包含 follow/avoid 命令、`risk_F/risk_A`、命令夹角与 `conflict_score`，并与 `yonly / geom-w` 共用同一套 PCR 融合公式和 eval/play 口径~~
- [x] ~~[P0] 将 PCR 训练、eval 与机制图输出命名收口到策略特征标签：训练目录、eval 目录和 figures 子目录必须包含 `yonly / geomw_w* / learnedw_rowrel_aux*` 等关键信息，并为 PCR 常用训练参数提供默认值，减少命令行重复项和误填风险~~
- [x] ~~[P0] 将 PCR w 机制图脚本收口为 `yonly / geom-w / learned-w` 任意两组或三组同图对照，避免新增 learned-w 后只能固定画单一组合。~~

## 2026-03-30 avoid 连续避障收口阶段

- [x] ~~[P0] 在 `s_avoid_basic` 中加入“过缝后收横移 + 小范围回正”的最小奖励：gap 内抑制 `cmd_x` 高频切换，开放小 `omega`，并按 world `+Y` 增加轻量 heading keep，优先解决连续多排行间穿出~~

## 2026-03-30 avoid 机身长度感知的行释放条件

- [x] ~~[P0] 将 `s_avoid_basic` 的当前行释放 / cross_line / success 判定从“机身中心过线”收口为“机身后缘过线”，避免头部刚过第一行就提前为下一行横移，导致后半身蹭回当前行障碍~~

## 2026-03-30 avoid rotate-only 回正门控

- [x] ~~[P0] 将 `s_avoid_basic` 的回正策略从“常态 `omega` 混合控制”收口为“偏航超阈值时的 rotate-only 硬门控”：只在已基本进 gap 或当前行已释放时触发，触发后强制 `cmd_x/cmd_y=0`，仅保留限幅 `cmd_omega`~~

## 2026-03-26 CONTEXT 文档刷新

- [x] ~~[P1] 用当前最终生效的 avoid 专家奖励、课程升级、固定模板几何、调试口径与训练命令，覆盖刷新 `CONTEXT.md`，避免后续新会话继续沿用旧的显存排查主线与过期奖励配置~~

## 2026-03-28 avoid 近障惩罚语义收口

- [x] ~~[P0] 将 `row_near_penalty` 从“二维表面距离惩罚”收口为“未进入当前最近一行 effective gap 时的前向接近惩罚”，避免继续静默错罚安全贴边通过~~
- [x] ~~[P0] 在当前“gap 判定正确但横向纠偏仍压不过前进主项”的阶段，同时上调 `rowLat`、下调 `rowNear`，先强推进入 effective gap、再避免 near 负项继续压主线~~
- [x] ~~[P0] 放弃 `rowNear` 主线，改成 `rowGap + rowPush`：`rowGap` 只在当前最近一行接近时放大横向对准，`rowPush` 只罚“近了但还没进入 effective gap”的状态~~
- [x] ~~[P0] 将 `rowPush` 从二值 `out_gap` 收口为连续 `push_err`，让“偏离一点”和“偏离很远”在 gap 外阶段受到不同强度的状态惩罚~~
- [x] ~~[P0] 在 row-gap 主线下补一个有方向的 `rowCmdX` 动作奖励，直接塑形朝当前行 effective gap 方向的横移指令，避免继续只靠状态差分拉动 `cmd_x`~~
- [x] ~~[P1] 将 `play` 诊断收口到“只显示当前最近一行”的 row-gap 主线：显式画出当前行 `raw/effective gap + gap center`，并打印 `x_dir_to_gap / cmd_x_toward_gap / row_cmdx_reward` 直接核对动作奖励方向~~
- [x] ~~[P0] 将 `rowCmdX` 从“弱正向朝向奖励”改成“目标型原始 `cmd_pred_x` 约束”：gap 外按 `row_err` 生成有方向、有最小幅值的 `x_target`，直接惩罚 `|cmd_pred_x - x_target|`，并同步输出 `cmd_pred_x_mean / x_target_mean / pred_err_x_mean`~~
- [ ] [P0] 在 `rowCmdX` 已确认落地但量级仍远小于 `rowPush` 的阶段，先将 `rowCmdX` 提到可见量级、同时压低 `rowPush`，优先验证横移指令是否能被直接拉起
- [ ] [P0] 在 `rowCmdX` 已放大但仍未拉起 `cmd_x` 的阶段，先做短程探索对照：把 `entropy_coef` 提到 `0.08`，优先验证是不是早期探索陷阱在压横移动作

## 2026-03-26 训练一致性审计规则补强

- [x] ~~[P1] 将“奖励生命周期 / 门控参考量 / 死信号 / done-reset 污染”补入 `AGENTS.md` 长期审计规则，后续默认先查这类静默污染问题~~

## 2026-03-26 avoid 奖励参考量收口与死信号清理

- [x] ~~Must: 将 `clearance_improve` 的增量参考量收口到前向阻塞 `forward_clearance`，避免与 near/far 门控语义分叉~~
- [x] ~~Must: 将 `s_avoid` 下的 `target_visible` 死信号明确关闭，避免日志继续把它当成有效奖励~~

## 2026-03-26 avoid progress 统计口径收口

- [x] ~~Must: 将 `eval` 里的 `progress` 从“任意非零即算到达”收口为与训练一致的逐行比例均值，并单独保留 `progress_any_rate`~~
- [x] ~~Must: 将 `s_avoid` 下已关闭的 `target_visible` 从 TensorBoard 奖励项输出里去掉，避免继续把零项当成有效奖励~~

## 2026-03-27 avoid 横移主奖励按第一性原理重写

- [x] ~~Must: 保留 `approach / collision / band / terminal / time` 主线不动，只替换当前 `clearance_improve + align_center` 的横移塑形逻辑~~
- [x] ~~Must: 将横移奖励改成“前方通畅时惩罚横移、前方受阻时奖励正确选边、前方受阻时奖励前向通路改善”~~
- [x] ~~Must: 下调 `passable_gate` 阈值，让当前固定模板里选边信号真正能进场~~
- [x] ~~Must: 将 `lat_choice` 改成有符号方向奖励：对边为正、错边为负，避免长期死零~~
- [x] ~~Must: 将 `block/free` 的基础量从固定前向 `forward_clearance` 改为 `nearest_obs_dist`，让侧前方障碍也能及时进入避障模式~~
- [x] ~~Must: 将 `block/free` 进一步收口为“前半平面最近障碍”，避免身后障碍继续误触发避障模式~~
- [x] ~~Must: 将 `lat_choice` 的方向信号改成更直接的左右通行差，并去掉过强的 `passable_gate` 次级门控，强化“该往哪边避”的老师信号~~
- [x] ~~Must: 将 `lat_choice` 进一步改成“前方左右近障风险差”老师，右侧近障更危险就奖励向左，左侧更危险就奖励向右~~
- [x] ~~Must: 将 `avoid_lat_choice_scale` 从 `2.0` 提到 `6.0`，让方向老师真正进入主梯度通道~~
- [x] ~~Must: 将 `lat_choice` 改成“方向辅助 + 幅值目标主导 + 过大约束”，避免继续奖励微小但同号的横移偏置~~
- [x] ~~Must: 将训练日志和 play 调试输出对齐到当前真实方向老师，并拆分 `lat_dir / lat_mag / lat_over` 三个子项，避免继续边猜边调~~
- [x] ~~Must: 去掉当前 teacher 图主线奖励，直接用 `s_avoid_active / s_avoid_pos_world` 对“最近一行最宽通道中心”的横移进展给奖励，避免继续奖励小幅同号命令而不奖励真正进通道~~
- [x] ~~Must: 将 `row_lat` 的训练/回放/快照诊断统一切到“最近一行最宽通道”口径，并修掉跨行切换时的监督静音空窗~~
- [x] ~~Must: 补一个只看“前方最近一行”的近障软惩罚 `row_near_penalty`，抑制“先直走撞近了再横移”的晚躲坏解~~
- [x] ~~Must: 将最近一行奖励从“追 gap 中心点”改成“进入当前最近一行的有效 gap 区间”，并让 `row_near_penalty` 只在尚未进入该行有效 gap 时生效，避免安全贴边通过继续被罚~~
- [x] ~~Must: 修补 effective gap 主线的一致性缺口：补齐 `_compute_nearest_row_gap_target()` 的返回值分支，给 `row_near_penalty` 增加 `gap_eff_valid` 门控，并将 `play` 主输出收口到 row-gap 主线~~
- [x] ~~Must: 将 row-gap 主线彻底从 `nearest_obs` 外层门控中解开，并给 `row_near_penalty` 增加 effective-gap 边界过渡带，同时进一步清掉 `play`/snapshot 中残留的旧 `block/passable` 主解释~~
- [x] ~~Must: 在当前“gap 判定已基本正确、但策略无视横向纠偏”的阶段，优先放大 `rowLat`（不先减 `approach`），先验证是否能把 `cmd_x` 明显抬起来~~

## 2026-03-26 avoid 去掉朝中/朝向约束并收紧有效奖励输出

- [x] ~~[P0] 将 `s_avoid` 的 `heading` 与 `target_center` 从有效奖励里去掉，只保留 `target_visible` 作为最小视野约束，减少朝前/居中信号对横移避障的干扰~~
- [x] ~~[P0] 将训练控制台输出收口为“只打印真正还在生效的奖励项”，并补齐 `target_visible/stability/align_center` 等当前实际在起作用的项，避免继续被零项和隐藏项误导~~

## 2026-03-26 avoid 横向归位语义修正 - align_center 改看 robot_x

- [x] ~~Must: 将 `align_center` 从读取 `goal_buf.x` 改为读取机器人相对 `band` 中线的横向偏差改善量，修复当前恒为零的死奖励~~
- [ ] Must: 保持 `collision / band / terminal` 口径不动，只做单因素短训验证 `align / band_out / progress / success`

## 2026-03-26 avoid 近障碍横移接入 passable_x 软门控

- [x] ~~Must: 将 `passable_dir.x` 接入近障碍 `clearance_improve`，只在“朝可通行侧横移且确实更安全”时放大奖励~~
- [x] ~~Must: 放宽 `passable_dir` 计算条件，让 `s_avoid` 即使不开 `passable_align` 也能拿到方向与门控统计~~

## 2026-03-26 avoid passable_x 软门控口径修正

- [x] ~~Must: 将近障碍 `clearance_improve` 的方向门控接回 `passable_gate`，避免前方遮挡不足时过早注入选边信号~~
- [x] ~~Must: 将方向系数改成基于执行方向横向分量的对齐度，减少对原始横移幅值的敏感性~~

## 2026-03-26 avoid 失败语义与 near-far 切换口径修正

- [x] ~~Must: 保留 done 步防污染清零逻辑，但让碰撞终止的最终惩罚不再比普通碰撞更轻~~
- [x] ~~Must: 将 near/far 切换从“全局最近障碍”改成“朝穿越线前方的阻塞 clearance”，让通过后归位更及时~~

## 2026-03-26 avoid 横移奖励改为近障碍避障 / 远障碍归位门控

- [x] ~~[P0] 将 `s_avoid` 的单一 `clearance_improve` 收口为双模式门控：近障碍继续奖励横移后更安全，远障碍改为奖励朝目标横向位置归位；当前固定模板下等价于朝中线准备下一排行~~
- [x] ~~[P0] 保留 `approach/heading/target_center/target_visible/avoid_band/collision` 主线不动，只替换横移辅助信号，避免继续只学会第一排前的局部横移~~

## 2026-03-25 PCR 第一版主线落地（y-only基线 + w_geom + eval口径）

- [x] ~~[P0] 以现有 `CmdVelExpert + GatePolicy + CommandPostProcessor` 为唯一 PCR 主入口，先补 `y-only` 基线日志与评测字段，再在 gate rollout 中加入最小 `w_geom` 与 `y_eff` 融合，固定 `switch_rate / near-miss / cmd_jerk / y_raw_vs_y_eff` 证据链~~

## 2026-03-25 PCR 评测/回放口径修复

- [x] ~~[P0] 修复 `eval/play` 的 PCR 口径漂移：`w_geom` 必须基于当前单帧 `gate_aff` 计算，`cmd_F/cmd_A` 必须继续使用与 gate 训练一致的 expert state（`prev_gate_y=0`）~~

## 2026-03-25 PCR raw/eff 时间语义与 CLI 显式参数优先

- [x] ~~[P0] 收口 PCR 主实验语义：`raw` 负责 history 和 `gate_smooth`，`eff` 只负责执行；同时让 `eval/play` 中仅 CLI 显式传入的 `beta/w_*` 运行期消融参数覆盖 ckpt meta，并打印 resolved config~~

## 2026-03-25 PCR gap 日志语义拆分

- [x] ~~[P1] 将 PCR 训练日志里的 `GateYGap` 从混合量拆成 `gap_clamp / gap_w / gap_total`，分别对应 `y_raw->y_safe`、`y_safe->y_eff`、`y_raw->y_eff`，避免继续把 safe clamp 与 `w` 贡献混在一起~~

## 2026-03-25 PCR moe 跟随侧改为解析式 expert

- [x] ~~[P0] 将 `train_highlevel.py` 的 `moe` 主线从“加载 follow policy ckpt”改成“直接调用 `expert_s0_follow.py` 的解析式 follow expert”，只保留 `--avoid_ckpt` 为必需输入，先打通 gate/PCR 的最小训练链路~~

## 2026-03-25 avoid 固定模板纵向拉长与时长同步

- [x] ~~[P0] 将 `s_avoid` 固定模板各 stage 的相邻行 `y` 间距统一增加 `0.15m`，并同步更新 `last_row_y`，保持穿越线/固定 goal/success 判据一致~~
- [x] ~~[P0] 将 `s_avoid_basic` 的回合总时长从 `15s` 先增加到 `20s`，随后再按最慢速度与最远穿越线重算到 `30s`，避免新几何下被旧时长过早截断~~

## 2026-03-25 avoid 降速与课程放宽

- [x] ~~[P0] 将 `s_avoid` 外生前进速度上限收紧为 `stage1=0.5, stage2=0.4`，并把 `stage/train warmup` 分别放慢到 `100/200 iter`，避免当前固定模板下前进过猛导致高碰撞~~
- [x] ~~[P0] 同步放宽 `stage1->2->3->4` 的 `progress/success/collision` 升级门槛，减少当前强教学几何下课程长期卡死~~
- [x] ~~[P1] 复查后补齐实际代码：修正 `train_highlevel.py` 中仍残留的旧速度表（`0.8/0.6`）与旧 warmup（`50/100 iter`），确保与上面训练口径一致~~

## 2026-03-25 avoid 横移安全正信号补齐

- [x] ~~[P0] 在 `s_avoid` 高层奖励中加入 `clearance_improve`：只在进入障碍区后奖励 `nearest_obs_dist` 的正向改善，给“提前横移后更安全”一个明确正信号，其他几何/band/成功口径保持不动~~
- [x] ~~[P0] 复查后将 `clearance_improve` 收口为“只在 band 内生效”的辅助正信号，并把系数从 `6.0` 调回 `2.0`，避免继续奖励向外墙逃逸~~
- [x] ~~[P0] 按“外生前进 + 主要学墙内横移避障”的训练意图，去掉 `velocity/body_backward/yaw_rate_penalty` 三项旧前冲辅助信号，并将 `clearance_improve` 提到 `6.0`，让横移安全正反馈成为主要新增引导~~
- [x] ~~[P0] 将 `clearance_improve` 的激活时机前置为出生即生效，不再等进入障碍区后才开启；继续保留“仅在 band 内给奖励”约束~~
- [x] ~~[P0] 将 `avoid_band` 的外墙约束也前置为出生即生效，不再等前进 `0.5m` 后才开始处罚 `x` 向出墙~~

## 2026-03-25 并行训练显卡口径固定

- [x] ~~[P1] 固定并行窗口资源约定：`PCR` 训练默认使用 `GPU 0`，`avoid expert` 训练默认使用 `GPU 1`，并在长期协作规则中补充日志/ckpt 默认避免重名覆盖~~

## 2026-03-23 avoid 专家改为外生前进+穿越判成功

- [x] ~~[P0] 将 `s_avoid` 固定模板从整排平移改成逐行显式左右开口的之字形布局：只改 `x` 向结构、不改任何 `y` 向参数，逼策略逐行根据 `local_map` 判断左右方向~~
- [x] ~~[P0] 尝试将固定模板进一步收口到“band 作为外墙、通路只在障碍群内部”的口径；后续按用户要求回退这次额外改动的障碍物分布，仅保留 `band_margin_x=0.05` 作为外墙收紧~~
- [x] ~~[P1] 收口 `s_avoid` 的在线监控口径：默认最好模型按 `success/progress/reward` 选点，训练日志里的旧 `goal_dist` 字段改名为穿越线目标距离，避免继续误导 eval/play 选模与曲线判断~~
- [x] ~~[P0] 收口 `forced_forward` 后的训练/评测一致性：真实执行命令回写到命令历史观测，`success` 直接终止高层 episode，`eval/play` 与导出结果统一切到穿越成功与 `cross_line` 口径，并补实验元信息留痕~~
- [x] ~~[P0] 将 `s_avoid_basic` 收口为“每个 episode 固定强制前进 + 固定正前方过障 goal + 穿越最后一行且无碰撞判成功”，让 avoid expert 专注学局部横移避障~~
- [x] ~~[P0] 将 `s_avoid_basic` 的 `approach/heading` 奖励改为服务“朝穿越线持续前进 + 朝世界 +Y 对齐”，不再奖励朝固定 goal 点中心收缩~~
- [x] ~~[P0] 将外生前进速度改成按 `stage` 与已完成 episode 数逐步放开上限：`stage1` 允许最高 `0.8m/s`，后续阶段随难度递减，避免简单场景前进探索过早被保守上限限制~~
- [x] ~~[P0] 将外生前进速度进一步收成“双重慢启动”：每个 stage 内按 `stage_iter/50` 放开，同时整条训练前 100 个 iteration 再加一层全局慢启动，两者取最小值；每次升到新 stage 重新从慢速开始~~
- [x] ~~[P0] 在外生强制前进口径下关闭 `risk_barrier_scale`，避免“前向减速”旧语义继续给策略施加无法通过动作消除的负信号~~
- [x] ~~[P0] 将 `goal_buf` 收口为“只表示到穿越线的前向距离”，去掉固定中心点 `x=0` 带来的横向偏置，同时让 `eval_highlevel` 的 success 判定切到穿越成功语义~~

## 2026-03-20 avoid 障碍体改回高阻尼动态体

- [x] ~~[P0] 将 `s_avoid` pooled obstacle 从 `fix_base_link=True` 静态体改回高质量高阻尼动态体，恢复碰撞几何随 root state 同步更新，同时保持障碍近似固定~~

## 2026-03-20 avoid 固定模板奇偶行交错重排

- [x] ~~[P0] 按“奇数行 3 障碍 / 偶数行 2 障碍”的规则重建 `stage1~4` 固定模板，放宽行间距、前推起始行，并将目标采样收口到最后一行后方 `0.35m+`~~

## 2026-03-20 avoid 障碍接触裕量收口

- [x] ~~[P0] 对 `s_avoid_basic` 的 robot/obstacle 显式提高 shape `contact_offset` 并小幅上调去穿出速度上限，减少 `play` 中薄障碍晚触发接触造成的穿障现象~~

## 2026-03-20 avoid 四阶段横向收紧与穿障止损

- [x] ~~[P0] 收紧 `s_avoid` 四阶段固定模板的离散障碍横向间距，同时补基于障碍真实几何的严格穿入判定，避免策略继续利用穿障样本污染训练结论~~

## 2026-03-20 avoid 固定模板左右偏置与纵深拉开

- [x] ~~[P0] 为 `s_avoid` 固定模板加入左右不对称偏置 4 变体，并将固定行 `y` 间距按更稀疏口径拉开，同时让 goal 后方采样跟随真实障碍最前沿而不是旧 `last_row_y`~~

## 2026-03-20 avoid 固定模板横向间距再放宽

- [x] ~~[P0] 继续放宽 `s_avoid` 固定模板各行障碍的 `x` 方向间距；若新布局撞到横向边界，再同步放大障碍区横向范围与相关采样边界~~

## 2026-03-20 avoid jitter 后横向宽度硬约束

- [x] ~~[P0] 将 `s_avoid` 固定模板的最小横向宽度约束收紧为“加入 `±0.06m jitter` 后仍必须满足”，并让抖动采样只接受满足该宽度的样本~~

## 2026-03-20 avoid 固定模板在线诊断改成最终样本口径

- [x] ~~[P1] 将 `avoid_preset_*` 在线日志从“基础 preset 统计”改为“preset 选中并完成 jitter 后的最终样本统计”，避免训练过程被假零值/假坏值误导~~

## 2026-03-20 avoid 固定模板 passage 验收放宽

- [x] ~~[P0] 将 `avoid_stage12/23/34_passage_width_min` 与 `avoid_preset_passage_width_min` 统一下调到 `0.45`，减少当前固定模板在 `jitter` 后被过严横向通路阈值全部判死的情况~~

## 2026-03-20 avoid 碰撞力阈值下调

- [x] ~~[P0] 将 `collision_force_threshold` 从 `0.5` 下调到 `0.05`，让当前 actor 碰撞力日志能更早进入碰撞检测与惩罚链，先恢复训练期碰撞信号~~

## 2026-03-20 avoid 碰撞力阈值回退到 1.0

- [x] ~~[P0] 将 `collision_force_threshold` 回退到 `1.0`，收回本轮临时低阈值设定，避免继续把碰撞力门槛本身作为主实验变量~~

## 2026-03-20 avoid 几何碰撞诊断索引修复

- [x] ~~[P0] 将 `obs-hit cand/strict/min` 的 `rb_states` 索引从 rigid-body handle 口径改成真正的 rigid-body tensor index，修复当前 `285` 量级异常值并恢复 strict geometric check 的可解释性~~

## 2026-03-20 avoid 回收轻量后退惩罚

- [x] ~~[P1] 将 `body_backward_scale` 从 `0.0` 回收到 `0.5`，在不恢复旧强惩罚的前提下抑制明显后退拖时间行为~~

## 2026-03-17 actor 障碍碰撞口径纠正

- [x] ~~[P0] 按 Isaac Gym 官方 group/filter 语义修正 actor 障碍碰撞设置：同 env 内 robot/obstacle 保持同组，scene filter 改回 0，先恢复真实物理碰撞~~

## 2026-03-17 Avoid obstacle band 软约束

- [x] ~~[P1] 为 `s_avoid_basic` 增加基于障碍簇几何的 obstacle band 软约束：reset 时缓存 per-env band，训练时按 `y_progress` 激活并对越界距离给软惩罚（先不加 reset）~~

## 2026-03-17 s_avoid 场景清洁化与课程几何重整

- [x] ~~[P0] 将 `s_avoid` 未激活障碍从“各 env 本地场地外停车带”改为“全局超远停车区”，避免圆柱/方块/墙出现在别的 env 旁边~~
- [x] ~~[P0] 将 `stage1/2/3` 的预设几何带分开定义，不再复用同一套 `stage12_band/core` 范围~~
- [x] ~~[P1] 为 `stage1/2/3` 预设库补最小可通过筛选，先剔除明显堵死的布局~~
- [x] ~~[P1] 清理 4 段课程的旧 `stage3 shrink` 口径残留，统一日志与阶段解释~~

## 2026-03-19 avoid 目标切片中心与 y 上限口径收口

- [x] ~~[P0] 将 `s_avoid` 目标横向中心改为基于真实 `goal_y` 切片求通路中心，继续压低侧绕目标分布~~
- [x] ~~[P1] 将 `goal_range_y` 上限显式改到 `4.0`，统一配置与真实后方目标采样口径~~

## 2026-03-19 avoid 碰撞螺旋止损与朝向轻增强

- [x] ~~[P0] 将 `collision_penalty` 与 `terminal_fail_penalty` 拆开，避免碰撞与提前终止共用同一大负值拖垮 value 学习~~
- [x] ~~[P1] 小幅回收 `velocity_scale`、提高 `s_avoid_basic` 默认 `entropy_coef`，先稳住探索与 critic~~
- [x] ~~[P1] 轻微增强朝向/目标居中约束，促进横移穿缝而不是原地旋转~~

## 2026-03-19 avoid 训练口径收口

- [x] ~~[P0] 统一 `entropy_coef` 生效口径：命令行显式覆盖优先，否则回落到任务配置默认值~~
- [x] ~~[P1] 明确 `terminal_fail_penalty` 走原始 `reward_cfg` 字典单独读取，不再隐式混入 `NavigationRewardConfig`~~
- [x] ~~[P1] 让 `goal_allow_fallback` 在 `s_avoid` 目标采样主线真正生效，避免配置成为死参数~~

## 2026-03-19 avoid 场景纵深可通过约束

- [x] ~~[P0] 为 `s_avoid` 预设生成补单独的 `y` 向最小间隔约束，避免同 lane 前后障碍过近直接惩罚横移~~
- [x] ~~[P0] 在预设验收中增加“连续纵深可通过”检查，筛掉横向有缝但纵向卡横移的坏场景~~

## 2026-03-19 avoid 场景纵深验收收口与生成诊断

- [x] ~~[P0] 将纵深验收窗口从 `band` 任意位置收口到障碍簇实际 `y` 包络附近，避免空白区假通道过检~~
- [x] ~~[P0] 为预设构建补最小诊断量：重试次数、采样失败次数、验收失败次数、真实最小 `y_gap`、真实纵深~~

## 2026-03-19 avoid stage3 lane 判断按真实横向占用收口

- [x] ~~[P0] 将 `same-lane` 判断从中心点 `dx` 改成按 capsule/box 真实横向占用区间比较，优先减少 stage3 横移撞前后 box 的坏样本~~

## 2026-03-20 s_avoid 固定模板改成四阶段离散障碍教学场

- [x] ~~[P0] 去掉 `stage4` 墙走廊口径，四个课程统一改成难度递增的离散障碍布局~~
- [x] ~~[P0] 每次 reset 给固定模板障碍加入小幅 `xy` 扰动，避免策略记固定路线而不读 `local_map`~~

## 2026-03-20 s_avoid 打破单侧右绕刻板化

- [x] ~~[P0] 为固定模板补左右条件覆盖：在对称模板外增加左右偏置布局，并保留镜像与 reset 扰动，逼策略学条件决策而不是记固定右绕路径~~
- [x] ~~[P1] 将 `HexAvoidBasicCfgPPO` 的 `entropy_coef` 从 `0.015` 提到 `0.03`，延缓探索过早收缩~~

## 2026-03-20 s_avoid 障碍统一收口为真实碰撞体

- [x] ~~[P0] 将 `s_avoid` 的 capsule/box/wall 全部收成固定基座障碍，禁止被机器人推着走，同时保持接触继续进入现有碰撞惩罚链~~

## 2026-03-20 s_avoid 固定模板 jitter 后最小复检

- [x] ~~[P0] 对固定模板 reset 抖动后的障碍实例补最小通过性复检；若抖动样本明显堵死则重采样，仍失败时回退到基础模板，避免坏 jitter 直接污染训练~~

## 2026-03-20 s_avoid stage1 条件决策模板补齐

- [x] ~~[P0] 为 `stage1` 增加左右条件差异模板（左堵右通 / 右堵左通），让策略从最早课程就学习按局部障碍选边，而不是固化默认右移起手式~~

## 2026-03-20 avoid 专家收口为纯平移避障

- [x] ~~[P0] 将 avoid 高层角速度命令上限设为 `0.0`，同时增强 `heading` 并恢复轻量 `yaw_rate_penalty`，把策略收成“前进+横移通过”的纯平移避障专家~~

## 2026-03-17 s_avoid 障碍碰撞根因排查收口

- [x] ~~[P0] 增加 `s_avoid_basic` 的“单障碍直接创建”调试对照，验证穿透是否来自 pooled actor 地下创建后再搬运这条用法~~

## 2026-03-17 checkpoint 保存频率补充

- [x] ~~[P1] 在 `train_highlevel.py` 增加固定 checkpoint 留存：除原有 `save_interval` 外，每 100 轮额外保存一次策略 `pt`~~

## 2026-03-17 Python 命令口径固定

- [x] ~~[P1] 在 `AGENTS.md` 增加长期规则：命令行运行 Python 与语法检查默认使用 `python3`，避免系统 `python` 落到 `python2.7`~~

## 2026-03-17 训练输出频率补充

- [x] ~~[P1] 在 `train_highlevel.py` 保持原有日志格式不变的前提下，额外增加“每 30 轮输出一次训练总结”的控制台打印~~

## 2026-03-17 协议留痕闭环与 Student 视觉契约校验

- [x] ~~[P0] 补齐 `eval/play` 的最终生效协议落盘，并为 `student/moe` checkpoint 增加口径一致性硬检查，避免评测/回放继续混用不兼容模型~~

## 2026-03-17 训练样本语义与 sim2real 留痕收口

- [x] ~~[P0] 修复 `invalid sample` 对 GAE 的跨步污染，补齐非有限观测的 fail-fast / recovery 语义，并让 reward / eval / play / checkpoint 对齐当前观测与 D435i 口径~~

## 2026-03-17 D435i sim2real 训练基准固化

- [x] ~~[P0] 联网核对 `Intel RealSense D435i` 官方参数，并将“仿真默认对齐 D435i 实机口径”写入 `AGENTS.md`，作为后续视觉相关训练的长期硬约束~~

## 2026-03-15 训练协议与结果留痕收口

- [x] ~~[P1] 将“训练一致性审计”规则固化到 `AGENTS.md`：统一审计双层输出、AGENTS 写入边界、高危静默污染检查项与固定收尾格式~~
- [x] ~~[P0] 禁止 `s_ood_holdout` 被误当训练场景，补齐训练/评测命令与脚本的硬约束~~
- [x] ~~[P0] 禁止 `student + moe` 伪蒸馏训练，并把 `best_model` 在线监控口径改名，避免误当论文最优 checkpoint~~
- [x] ~~[P0] 让无效动作样本彻底退出 value loss，并修复 gate 安全钳制、`prev_robot_pos`、`done_during` 的训练口径偏差~~
- [x] ~~[P0] 补齐 run/checkpoint/eval 的关键信息留痕与 resume 一致性检查，避免 teacher/vision 来源被静默替换~~
- [x] ~~[P0] 把 avoid 课程切换证据保存在 TB，避免只在 console 短暂打印后丢失~~

## 2026-03-15 训练动力学与实验对照口径补齐

- [x] ~~[P0] 区分 `timeout` 与真实终止，补齐 GAE timeout bootstrap，避免长回合 value/advantage 被系统性截断~~
- [x] ~~[P0] 让 expert 接管样本彻底退出 advantage 标准化，避免不同接管比例下 PPO 更新强度不可比~~
- [x] ~~[P0] 强制 `student` 模式必须提供 `teacher_ckpt`，并让 student 路径的 reward 回到 GT 几何口径~~
- [x] ~~[P0] 将 `risk_barrier` 改成基于 GT 几何的沿命令方向风险，并修复 low-level 提前 done 时的 reward 分项污染~~
- [x] ~~[P0] 显式暴露并记录 EGPO 学习率策略，清理 `s_ood_holdout` 的训练命令冲突，保证实验协议可解释~~

## 2026-03-15 高层训练口径与动作后处理一致性修复

- [x] ~~[P0] 将 `s_avoid_basic` 课程统计改为按 episode 实际所属 stage 记账，避免旧 stage 尾部样本污染新 stage 升级窗口~~
- [x] ~~[P0] 去掉 avoid actor/student 的特权 `difficulty` 泄漏，并补齐 `moe` 下 follow/avoid/gate 各自正确的输入口径~~
- [x] ~~[P0] 修正 `goal occlusion` bootstrap、expert/teacher 有限值防护，以及 expert 接管时的 PPO 更新口径~~
- [x] ~~[P0] 让 `CommandPostProcessor` 记住真实执行命令，改用沿命令方向的 clearance，并把必要历史量补进高层观测~~
- [x] ~~[P0] 统一课程/诊断命名与指标：stage23 独立 progress/success 定义、命令诊断口径一致、TB/console 正确展示 progress/success 与 stage 窗口~~

## 2026-03-12 Avoid 难度标量解耦与展示/评测口径统一

- [x] ~~[P0] 将 Avoid 的 `actor_difficulty` 从视锥裁剪后的 `local_map_2ch` 脱钩，改为基于全 GT 局部图计算客观拥挤度标量~~
- [x] ~~[P0] 统一 `play_highlevel` 的后处理输入，让 Teacher/Student 都只基于当前策略输入图计算 `clearance_override`~~
- [x] ~~[P0] 将 `eval_highlevel` 对齐到当前 Avoid 主线：`local_map_2ch` 输入、旧高层 ckpt 兼容、后处理口径一致~~
- [x] ~~[P0] 修复 Student 蒸馏 loss 不回传当前策略梯度的问题，避免蒸馏训练挂空挡~~
- [x] ~~[P0] 将高层局部地图默认提升到 `32x32`，并让高层编码器与 student 预测图共同兼容 `16/32` 输入~~

## 2026-03-12 高层 Avoid Teacher 非对称 PPO 口径

- [x] ~~[P0] 将高层 Avoid Teacher 改为非对称 PPO：Actor 只吃 D435i 视锥约束后的 `local_map_2ch`，Critic 吃全 GT 地图~~
- [x] ~~[P0] 在高层观测中同时产出 Actor 可视图与 Critic 全图，避免训练阶段再用 `gt_affordance=local_map_2ch` 直接覆盖~~
- [x] ~~[P0] 将 rollout/buffer/bootstrap/update 全部接上 actor/critic 双输入，并让 reward shaping 与 post-processor 分别走全图/可视图~~
- [x] ~~[P0] 兼容旧高层 checkpoint：缺失 `critic_*` 权重时自动从 actor 主干镜像初始化，保证 train/play 都能继续加载旧模型~~

## 2026-03-12 Avoid 专家 B 组弱视野约束对照

- [x] ~~[P0] 在 Avoid 高层奖励中加入弱 `target_center(goal_buf)` 与小 `yaw_rate_penalty`，验证是否能减少甩头并为后续 `w` 融合保留更连续的视觉上下文~~
- [x] ~~[P0] 在 Avoid 高层奖励中加入机体系倒退弱惩罚，并补充前向/倒退速度日志，压制“全程倒着走”而不明显削弱全向绕障~~

## 2026-03-12 s_avoid_basic 课程升级口径修正

- [x] ~~[P0] 将 `s_avoid_basic` 的 `stage 1->2 / 2->3` 从“低碰撞 + 障碍暴露覆盖”改为“低碰撞 + 障碍暴露覆盖 + 有效推进”，避免靠保守退让也提前升阶段~~
- [x] ~~[P0] 在 `s_avoid_basic` 课程升级中加入“接近局部子目标才算成功”的 `success100` 约束，并将最早升级回合数从 `800` 提到 `1600`~~

## 2026-03-12 s_avoid_basic 完美图验证与 actor-only GT 地图

- [x] ~~[P0] 锁定 `s_avoid_basic` 的 GT affordance 只走 `plane + actor` 几何栅格化，禁止回退到 heightfield 混合口径~~
- [x] ~~[P0] 在高层观测中显式产出 `local_map_2ch(occupancy + clearance/cost)`，并与 Avoid 专家输入保持一致~~
- [x] ~~[P0] 在 `play_highlevel` 增加 `front/left/right` 静态导出，保存 `raw gt_affordance`、`local_map_2ch` 与坐标核对结果~~
- [x] ~~[P0] 将 `s_avoid_basic` 的 GT 图参考点从机身中心切到深度相机安装点，并在静态图里增加“机器人朝向 vs 障碍连线”夹角~~
- [x] ~~[P0] 为 `s_avoid_basic` 静态调试增加“强制出生朝向”输入，按机体 `+y` 与世界 `+Y` 的夹角设定，并输出目标/实际/误差~~
- [x] ~~[P0] 将 GT 图可见范围收口到 D435i depth 参数（87x58, 0.3~3.0m），并增加障碍物距离真值对照导出~~
- [x] ~~[P0] 将 GT 视场从“二维水平扇形”进一步收口到“深度相机视锥与地面的交域”，并避免视场外 clearance 被平滑成已知自由区~~
- [x] ~~[P0] 修正 `visible_mask` 的 `x/y` 轴顺序，使 `local_map_2ch` 的视锥裁剪与障碍栅格写入使用同一套 `x_right/y_forward` 口径~~
- [x] ~~[P0] 将 GT 视锥中的相机高度改为“机身初始高度 + 相机安装 z 偏移”，避免把深度相机误当成离地 10cm~~

## 2026-03-06 避障专家输入与上机主线口径更新

- [x] ~~[P0] 避障专家默认观测更新为 `state + goal_buf + local_map_2ch(occupancy + clearance/cost)`，不再将 `low_obstacle/跨越信息` 作为默认必需输入~~
- [x] ~~[P0] `goal_buf` 统一解释为“局部子目标”，来源可为目标跟踪、路径点或外部定位，不要求必须是人的真值位置~~
- [x] ~~[P0] 避障 sim2real 主线更新为 `D435i depth -> local_map_2ch -> avoid expert`；teacher 真值图主要用于仿真参考，不作为默认部署输入~~
- [x] ~~[P0] student 地位更新为“可选项”：若传统深度处理已能稳定生成局部地图，则不强制经过 student~~
- [x] ~~[P0] 文档优先级补充：V7 作为阶段性方案保留，后续若与 `AGENTS.md` 最新实验口径不一致，则默认以 `AGENTS.md` 为准~~

## 2026-03-05 s_avoid_basic 三阶段避障课程（plane + actor）

- [x] ~~[P0] `s_avoid_basic` 固化为三阶段课程并统一 `mesh_type='plane'`（禁止高度场口径）~~
- [x] ~~[P0] 引入指标切换：最近100轮平均碰撞率 + 至少200轮后才允许阶段切换~~
- [x] ~~[P0] 阶段3走廊宽度线性收紧：最近50轮碰撞率<8% 时每次 -0.05m，最小 0.85m~~
- [x] ~~[P0] 障碍生成统一为 PhysX 基元 actor（capsule/box），目标保持无碰撞参与~~
- [x] ~~[P0] 场景叙事与命令清单同步更新（`SCENE_NAME_MAP.txt` / `traincmd*.txt`）~~

## 2026-03-05 论文场景序列扩展（e_*）

- [x] ~~[P0] 新增 `e_` 场景命名序列入口，并落地第一个场景 `e_L_confilct`~~
- [x] ~~[P0] 实现 `e_L_confilct` 场景：目标先直行后 90°右转，转角内侧圆柱障碍~~
- [x] ~~[P0] 打通高层回放入口与场景清单文档，支持按新名称直接查看~~
- [x] ~~[P0] `e_L_conflict` 几何口径修正：单圆柱直径 0.30m，位置与 L 形轨迹两条边严格相切~~
- [x] ~~[P0] `e_L_conflict` 圆柱改为按 legged_gym 场景障碍规范创建（场景障碍描述 + 非地形压印实现）~~

## 2026-03-05 Follow 专家直管输出（play+train 统一）

- [x] ~~[P0] 新增 `--use_follow_expert` 统一开关：`skill=follow` 时由 `expert_s0_follow` 直接输出高层指令~~
- [x] ~~[P0] `--use_follow_expert` 生效时放宽 `--teacher_ckpt` 依赖（可不提供）~~
- [x] ~~[P0] 统一日志口径：显式打印 `cmd_source=follow_expert` 并确保 `cmd_pred == cmd_expert`（接管路径）~~

## 2026-03-05 e_L_conflict 几何与轨迹修正（可视化对齐）

- [x] ~~[P0] 修正 e_L_conflict 障碍物摆放与高度：单圆柱贴地、位置约束在地面有效范围内~~
- [x] ~~[P0] 将目标轨迹从“硬直角折线”改为“直行+圆角右转+直行”，确保绕障通过且不穿障~~
- [x] ~~[P0] 增加 e_L_conflict 关键几何日志（障碍物中心/轨迹关键点），用于快速验收~~

## 2026-03-05 e_L_conflict 初始契约修正（朝向/间距/直线段）

- [x] ~~[P0] e_L_conflict 出生姿态修正：机器人初始朝向目标点~~
- [x] ~~[P0] e_L_conflict 初始间距修正：机器人与目标固定 0.5m~~
- [x] ~~[P0] e_L_conflict 直线段增长：转弯前直线段在当前基础上增加 1.5m~~

## 2026-03-06 论文场景扩展：e_S_corridor

- [x] ~~[P0] 新增 `e_S_corridor`：平地 + S 型窄走廊，重点评测连续弯道中的跟随表现~~
- [x] ~~[P0] 用 `box actor` 生成 S 型走廊双侧墙体，目标沿走廊中心线脚本运动~~
- [x] ~~[P0] 打通注册、场景命名清单与快速查看入口，保证 `play/quick_scene_view` 可直接使用~~

## 2026-03-06 e_S_corridor 场景表达切换（论文单场景验证）

- [x] ~~[P0] 将 `e_S_corridor` 从分段 `box actor` 墙体切换为连续 `trimesh` 曲线墙~~
- [x] ~~[P0] 保留目标中心线脚本与碰撞，确保单场景验证时可见连续墙面~~
- [x] ~~[P0] 将 `e_S_corridor` 从“高度图转 mesh”改为“直接解析 mesh”，统一 +Y 轴向并消除锯齿~~

## 2026-03-06 e_S_corridor 论文评测口径补齐

- [ ] [P0] 修正 e_S_corridor 内侧墙面局部突块，进一步提升连续弯道几何平滑度
- [x] ~~[P0] 增加 e_S_corridor 可调参数覆盖：支持命令行修改走廊宽度与弯道曲率~~
- [x] ~~[P0] 在 play 侧增加论文评测输出：碰撞率、跟随率、成功完成率与自动绘图~~
- [x] ~~[P0] 统一 e_S_corridor 目标轨迹与墙体中轴线来源，避免目标偏墙~~
- [x] ~~[P0] 修正 e_S_corridor play 碰撞统计口径：改为每步真实碰撞，不再依赖 done/reset~~
- [x] ~~[P0] 将 e_S_corridor 碰撞统计拆为“碰撞事件数 + 碰撞步数占比”，统一论文口径~~
- [x] ~~[P0] 收口 e_S_corridor 论文统计字段：删除碰撞事件数，只保留成功率/跟随率/回合碰撞率/碰撞步数占比~~

## 2026-03-05 场景命名口径升级（主入口改为 s_*）

- [x] ~~[P0] 统一场景任务主入口命名为 `s_*`（如 `s_cylinder`），不再以 `hex_s*` 作为主入口~~
- [x] ~~[P0] 保留旧任务名 `hex_*` 兼容别名，避免历史脚本失效~~
- [x] ~~[P0] 同步高层训练/回放脚本的任务白名单、默认任务、提示文案到 `s_*` 口径~~
- [x] ~~[P0] 同步命令模板与使用文档中的任务名到 `s_*` 口径~~
- [x] ~~[P0] 下线 s1 相关三场景入口（`s_open_field/s_follow_static/s_follow_moving`）并新增 `s_avoid_basic` 空场占位~~

## 2026-03-05 入口参数收口（取消默认场景）

- [x] ~~[P0] 取消 train/play/eval 的默认 `--task`，统一改为必须显式传入~~
- [x] ~~[P0] 清理运行期提示中的旧场景文案（统一到 `s_avoid_basic` 口径）~~
- [x] ~~[P0] 收口剩余命名残留（如 `s_open_field_large`）到 `s_avoid_basic` 命名族~~

## 2026-03-05 场景清单落盘（命名↔内容）

- [x] ~~[P0] 新增 `SCENE_NAME_MAP.txt`，统一维护“场景命名与内容描述”对照~~
- [x] ~~[P0] 同步 `train_cmd.txt` 与 `traincmd.txt` 到同一套 `s_*` 场景叙事~~

## 2026-03-05 场景快速检查脚本

- [x] ~~[P0] 新增 `quick_scene_view.py`：支持按 `--task` 快速打开场景环境查看~~
- [x] ~~[P0] 支持 `--list_scenes` / `--all`，用于场景名枚举与批量短时巡检~~
- [x] ~~[P0] 同步更新 `traincmd.txt` 的使用示例，降低手工检查成本~~

## 2026-03-05 避障主线启动（全场景配置 + 首轮开训）

- [x] ~~[P0] 统一避障主线场景口径：S1~S5 作为训练场景，S6 作为结构化 OOD 验证场景~~
- [x] ~~[P0] 统一训练入口口径：`train_highlevel.py` 对避障主线任务给出明确推荐列表与告警策略~~
- [x] ~~[P0] 统一场景配置口径：避障场景导航朝向/奖励关键参数与当前 +Y 前进约定对齐~~
- [ ] [P0] 完成 S1~S5（训练）+ S6（验证）最小可运行冒烟检查并记录结果（用户要求暂不跑）
- [ ] [P0] 启动首轮避障训练 run（保存日志目录、关键参数、首批在线指标）（用户要求由人工启动）

## 2026-03-03 S0 目标轨迹回切（脚本圆轨迹 -> 训练随机轨迹）

- [x] ~~[P0] 将 `hex_s0_follow` 的 `moving_target_mode` 从 `s0_circle_right` 回切到训练默认随机模式（含左右转与变速）~~
- [ ] [P1] play 验收：目标点不再沿固定圆弧，出现随机转向与速度波动

## 2026-03-03 S0 随机轨迹课程重调（缓转 + 极限变速）

- [x] ~~[P0] 下调随机转向强度并拉长命令周期，避免出现突发 180 度转向~~
- [x] ~~[P0] 提升速度上下限与波动幅度，增强快慢切换与加减速极限~~
- [x] ~~[P0] 提高边界预转向缓冲并降低中心偏置，减少边界反弹导致的瞬时反向~~
- [ ] [P1] play 验收：转向为长弧缓变（左右都有），速度变化明显增大，且不再突发掉头

## 2026-03-03 S0 方向口径固化入 AGENTS（长期规则）

- [x] ~~[P0] 固化 S0 坐标与符号口径：`heading=0=>+Y forward`、`goal_buf=(x_right,y_forward)`、`bearing=atan2(x_right,y_forward)`~~
- [x] ~~[P0] 固化世界→机身投影公式：统一使用 `R(-heading)`（写入 AGENTS）~~
- [x] ~~[P0] 固化转向符号：`+omega=左转(CCW)`，并固定 `omega=-k_yaw*bearing` 判据~~
- [x] ~~[P0] 固化最小验收标准与排查顺序：`alpha/omega/step_yaw` 三联判据 + force_cmd→goal_buf→expert→轨迹~~

## 2026-03-03 S0 转向镜像排查（世界→机身坐标口径统一）

- [x] ~~[P0] 统一修正 S0 路径中的世界→机身坐标投影符号（expert / goal_buf / play 诊断一致）~~
- [x] ~~[P0] 复验起步转弯阶段：`pre_alpha` 与 `cmd_omega` 必须满足“朝目标收敛”方向关系~~
- [x] ~~[P1] play 验收：仅看 step0~80，目标开始转弯时不再出现对称反向发散~~

## 2026-03-03 S0 转向发散专项修复（方向优先 + 目标大半径）

- [x] ~~[P0] S0 expert 恢复“偏角大先转向”门控：`|bearing|>lock` 时仅转向，`|bearing|<release` 后恢复前进~~
- [x] ~~[P0] S0 expert 调参为“转向优先”：提高 yaw 权重/上限，降低前进上限，抑制目标转弯阶段外扩~~
- [x] ~~[P0] S0 移动目标右转圆轨迹半径扩大到 3 倍（含圆心同步放大），降低曲率避免急转~~
- [x] ~~[P0] 按实测口径统一转向符号：`+omega=左转(CCW)`，expert 改为 `omega=-k_yaw*bearing`，并同步修正 play/验证脚本判据~~
- [ ] [P1] play 验收：`bear_y` 在转弯段回落、`dist` 不再持续上升、轨迹不再朝反方向发散

## 2026-03-03 S0 Expert 约束切换（仅前后+转向）

- [x] ~~[P0] expert 输出改为 `cmd_x=0`，仅保留 `cmd_y/omega`，移除侧向平移控制~~
- [x] ~~[P0] 距离控制改为欧氏距离误差，并加入“先转后走”门控（大偏角优先转向）~~
- [x] ~~[P0] 补齐滞回锁（lock/release）与 `omega_deadzone`，抑制阈值附近抖动~~
- [ ] [P1] play 验收：`cmd_expert[:,0]` 恒为 0，重置后不再出现侧移驱动的绕圈

## 2026-03-03 S0 expert 动作调试（重置可见化 + 前向控制修正）

- [x] ~~[P0] play 增加 reset 原因打印（success / done_during / timeout）并在 done 后清空轨迹统计缓存，避免重置跳变误判~~
- [x] ~~[P0] S0 expert 纵向控制改为机体系前向误差主导（替代纯距离误差），抑制侧后方场景下 `cmd_y` 打满追圈~~
- [ ] [P1] play 验收：重置后首 20 步不出现无因打满转向，`cmd_y` 不再在大侧偏时持续顶满

## 2026-03-03 S0 Expert 连续控制律切换（cos(alpha)）

- [ ] [P0] expert 从“阈值锁定先转后走”切换为连续控制：`v = k_v * (d-d_des) * cos(alpha)`，`omega = k_omega * alpha`
- [ ] [P0] 保持约束不变：仅输出 `cmd=[0, v_forward, omega]`（禁止侧移）
- [ ] [P1] play 验收：目标转弯时 `bear_y` 不再快速发散，`dist` 增长斜率下降

## 2026-02-28 S0 目标轨迹改为“慢速右转整圆”并关闭超时终止

- [x] ~~[P0] S0 moving target 新增脚本模式 `s0_circle_right`：固定半径、顺时针、整圈回到起点~~
- [x] ~~[P0] S0 play/train 关闭 episode timeout 终止（保留其他终止条件）~~
- [ ] [P1] play 验收：目标轨迹完成整圈且机器人可持续跟随，不因时长被 reset

## 2026-02-28 S0 Expert 重写（去影子点，三路解耦控制）

- [x] ~~[P0] 重写 `expert_s0_follow.py`：完全移除 shadow point 逻辑，仅用 `robot_pos/robot_heading/target_pos`~~
- [x] ~~[P0] expert 改为三路控制：纵向距离控制 + 横向偏差控制 + 偏航对准控制（输出 `cmd[x_right, y_forward, yaw]`）~~
- [ ] [P1] play 下 expert 接管验证：目标前方时 `cmd_y>0`、侧偏时 `cmd_x`修正方向正确、整体能稳定回到目标后方

## 2026-02-28 S0 朝向奖励全程生效（关闭 heading gate）

- [x] ~~[P0] S0 显式设置 `heading_gate_use=False`，避免低速/低进度时朝向奖励被清零~~
- [x] ~~[P0] S0 显式设置 `heading_use_difficulty_gate=False`，避免高难度阶段朝向奖励被缩弱~~
- [ ] [P1] 复跑到 iter100 验收：heading>0.1、backward>-0.05、Goal dist<=1.8

## 2026-02-28 S0 Expert/Reward 朝向口径对齐（BC 标签一致化）

- [x] ~~[P0] 传入 expert 的 `robot_heading` 增加 `heading_offset_rad`，与奖励函数朝向口径一致~~
- [ ] [P1] 复跑到 iter100 验收：heading>0.1、backward>-0.05、Goal dist<=1.8

## 2026-02-28 S0 朝向基准修正（+Y 前向口径）

- [x] ~~[P0] S0 `heading_offset_rad` 从 `0.0` 调整到 `1.5708`，对齐机体 +Y 前向约定~~
- [ ] [P1] 复跑到 iter100 验收：heading>0.1、backward>-0.05、Goal dist<=1.8

## 2026-02-28 S0 诊断口径一致性修复（EGPO heading debug 对齐 expert）

- [x] ~~[P0] 修复训练 debug 中 `dir_world` 计算：与 expert 保持一致（`target_vel/heading` 缺失时走几何方向）~~
- [ ] [P1] 复跑观察：`EGPO heading` 诊断与 play 行为是否一致

## 2026-02-28 Heading 侧身惩罚口径调整（cos 偏移）

- [x] ~~[P0] `heading_reward` 从 `cos(err)` 改为 `cos(err)-0.15`，让侧身状态变为负分~~
- [ ] [P1] 复跑对比：heading reward 均值、backward、Goal dist 与 play 绕圈比例

## 2026-02-28 S0 Expert 方向回退修复（避免 `target_vel=None` 退化到固定 +Y）

- [x] ~~[P0] 修复 expert 方向生成：缺失 `target_vel/target_heading` 时改为几何方向回退（`target - robot`）~~
- [ ] [P1] 复跑观察：alpha=0 后 backward 是否回落、Goal dist 是否不再飙升

## 2026-02-28 S0 奖励再平衡（target_lost 降权 + sigma 回调）

- [x] ~~[P0] `target_lost` 持续惩罚从 `-2.0` 下调到 `-0.3`（避免总奖励长期为负）~~
- [x] ~~[P0] `follow_distance_sigma` 从 `0.15` 回调到 `0.20`（先恢复可学区间）~~
- [ ] [P1] 复跑到 iter150 复核：`Rollout step reward` 转正且 `Goal dist` 开始回落

## 2026-02-28 S0 奖励修正（坐标系一致 + 丢失持续惩罚 + 转向惩罚降权）

- [x] ~~[P0] `vel_towards_goal` 对齐到世界系：奖励输入速度改为 `root_states[:,7:10]`~~
- [x] ~~[P0] `target_lost` 改为持续惩罚，不再触发 reset~~
- [x] ~~[P0] `turn_penalty_scale` 从 `0.50` 下调到 `0.20`~~
- [ ] [P1] 复跑并对比：反向走比例、Goal dist、target_lost 触发率、heading 误差

## 2026-02-28 S0 奖励塑形强化（朝向/视野/后退/无效转向）

- [x] ~~[P0] S0 奖励权重调整：`heading_scale=2.5`、`follow_distance_sigma=0.15`~~
- [x] ~~[P0] S0 目标丢失约束收紧：`target_lost_k=5`~~
- [x] ~~[P0] 新增 S0 奖励项：后退惩罚（`backward_scale`）与小角误差下无效转向惩罚（`turn_penalty_scale`）~~
- [ ] [P1] 复跑训练并对比：Goal dist、heading error、target_lost 触发率与 play 反向走比例

## 2026-02-28 S0 Expert 标签观测对齐（修正学反方向主线）

- [x] ~~[P0] 训练阶段 expert 标签改为仅使用策略可观测信息（去除 `target_vel_world/target_heading` 依赖）~~
- [ ] [P1] 复跑训练并复核：`alpha=0` 后 Goal dist 是否收敛、play 中是否仍出现反向远离

## 2026-02-28 S0 接管期动态学习率（alpha=0 后放大学习步长）

- [x] ~~[P0] 接管期动态 `lr`：`expert_alpha_update<=0` 时 `1.5e-4`，否则 `1.5e-5`~~
- [x] ~~[P0] 回退动态 clip 试验，避免与动态学习率实验互相干扰~~
- [ ] [P1] 1000 轮跑到 iter250 验证：alpha=0 后 KL/clip_frac 是否非零且 Goal dist 改善

## 2026-02-28 S0 接管期动态 clip 调整（alpha=0 后放宽 PPO 裁剪）

- [x] ~~[P0] 接管期动态 `clip_range`：`expert_alpha_update<=0` 时用 0.20，否则保持 0.05~~
- [x] ~~[P0] 固定评测口径：回归 1000 轮训练预算，观察 iter250 接管段 KL/clip_frac 非零性~~
- [ ] [P1] 验证目标：alpha=0 后 `KL` 与 `clip_frac` 从近零变为可学习区间（非零）并改善 Goal dist

## 2026-02-25 S0 回归原始 EGPO 衰减机制（移除 hold）

- [x] ~~[P0] 删除 S0 特有 hold 平台：`alpha` 从迭代 0 开始连续衰减（与原始 EGPO 一致）~~
- [x] ~~[P0] 删除 hold 相关 loss 屏蔽链路，恢复完整 EGPO loss 更新路径~~
- [x] ~~[P0] `EXPERT_BC_COEF` 恢复至 2.0，默认学习率保持 1.5e-5~~
- [x] ~~[P1] 日志频率改为自动 `num_iterations/20`（至少每 1 轮）~~
- [ ] [P1] 跑到 iter230 验证接管稳定性（重点看 Goal dist / KL / rollout reward）

## 2026-02-25 EGPO 接管机制修正（hold阶段保留BC预热）

- [x] ~~[P0] 修正 hold 阶段损失组合：屏蔽 PPO policy/entropy，仅保留 value + BC~~
- [x] ~~[P0] 删除 hold 阶段“仅 value_head 保留梯度”的手动清零逻辑，避免误伤 actor 的 BC 梯度~~
- [x] ~~[P0] 调整 `EXPERT_BC_COEF` 到 0.5，确保 hold 阶段 actor 预热强度足够~~
- [ ] [P1] 从头训练复核接管期稳定性（iter110/140/230）

## 2026-02-25 实验优先协作口径升级（提示词规则落盘）

- [x] ~~[P0] 将“实验结果优先”的协作身份与输出风格写入 `AGENTS.md`（保留原意：先看指标提升）~~
- [x] ~~[P0] 语言口径固定：解释中文；代码与注释英文；不使用软件工程术语~~
- [x] ~~[P0] 固定实验分析首句模板：`主矛盾是___，建议改___，预期效果是___`~~
- [x] ~~[P1] 同步 `CLAUDE.md` 与 `PROJECT_OVERVIEW_CN.md`，避免多文档口径冲突~~

## 2026-02-13 项目架构优化（文档与结构，不改核心代码）

- [x] ~~[P0] 冻结架构边界：`legged_gym/`、`rsl_rl/`、`resources/`、`tools/` 不改~~
- [x] ~~[P0] 建立 `docs/` 骨架与文档职责索引（specs/operations/reference/archive）~~
- [x] ~~[P0] 更新顶层文档（`README.md`、`AGENTS.md`）以统一“单一事实源”口径~~
- [x] ~~[P1] 全局引用自检：修正失效文档路径，避免后续维护歧义~~

## 2026-02-13 项目架构优化（第二阶段：文档归位）

- [x] ~~[P0] 文档迁移到 `docs/`：`PHASE_SWITCHING_GUIDE.md`、`训练指令.txt`、`参数一览表.md`、`思路设计.md`~~
- [x] ~~[P0] 为迁移文档补充状态标记（当前规范/当前参考/历史参考）~~
- [x] ~~[P1] 更新 `README.md` 与 `docs/README.md` 索引路径并完成引用自检~~

## 2026-02-13 项目架构优化（第三阶段：导航与职责收口）

- [x] ~~[P0] 根目录收口：`ROBOT_SPECS.md` 迁移到 `docs/reference/ROBOT_SPECS.md`~~
- [x] ~~[P0] 新增 `docs/NAVIGATION.md`，统一“当前规范/当前参考/历史参考”导航~~
- [x] ~~[P0] 对齐 `AGENTS.md`、`CLAUDE.md`、`PROJECT_OVERVIEW_CN.md` 的文档职责口径~~
- [x] ~~[P1] 活跃文档路径自检：确认迁移后入口文档无残留旧路径~~

## 2026-02-13 TODO_LOG 记录机制优化（仅重大变更记录）

- [x] ~~[P0] 新增规则：TODO_LOG 仅记录重大改变/思路调整（架构、主线、评测协议、跨模块关键改动）~~
- [x] ~~[P0] 新增规则：小修小补默认不记录（局部 bugfix、轻微调参、文案/注释、格式整理）~~
- [x] ~~[P1] 与 AGENTS.md / CHANGELOG_CN 机制对齐并完成自检~~

## 2026-02-13 角色机制改造（自动路由 + 项目主理人）

- [x] ~~[P0] 将“新对话必问角色”改为“默认自动路由 + 用户可随时强制切换”~~
- [x] ~~[P0] 重写角色(1)为“项目主理人（RAL共同作者）”提示词（高效、行动导向）~~
- [x] ~~[P0] 保留并对齐角色(2)“编程主力”表述，确保与新切换机制兼容~~
- [x] ~~[P0] 参考方案基准从 V4 切换为两个 V7 文档（`hexapod_RAL_complete_technical_spec_v7.md` + `hexapod_RAL_integrated_final_v7.md`）~~
- [x] ~~[P1] 完成文档自检：语义一致、无互相冲突规则~~

## 2026-02-13 S0 训练链路 P0 稳定性修复（训练主线）

- [x] ~~[P0] `train_highlevel.py` 引入配置隔离（cfg 深拷贝）与 S0 任务-地形类型强校验，避免跨任务污染与错误场景开训~~
- [x] ~~[P0] 统一 `scene_difficulty_override` 归一化入口（wrapper/env 双端）并消除重复实现~~
- [x] ~~[P0] S0 跟随成功判定改为单一事实源：使用 `follow_distance_desired/min/max` 与高层步长推导窗口~~
- [x] ~~[P0] 冻结窗口语义修复：目标冻结期间不累计 S0 stable-follow 成功步数~~
- [x] ~~[P0] EGPO 非有限值策略改造：默认 fail-fast（可配置），避免 silent sanitize 掩盖训练崩坏~~

## 2026-02-13 论文级评测管线（训练监控与独立评测解耦）

- [x] ~~[P0] 新建独立 eval 入口（frozen policy + 固定评测集），禁止直接使用训练在线统计写论文结果~~
- [x] ~~[P0] 落地论文必须指标：Follow MAE/RMSE、Success Rate、Time-to-Success(mean/median/p95 + fail ratio)~~
- [x] ~~[P1] 落地论文应补指标：CoT、Inference Latency(p50/p95)、Params(total/trainable)~~
- [x] ~~[P0] 输出标准结果文件：`metrics.json` + `metrics.csv`，支持多难度分组统计~~
- [x] ~~[P1] 训练侧标量与论文口径分离：保留在线监控用途，不作为论文主表直接数据源~~

## 2026-02-13 会话级 TODO 规则

- [x] ~~[P0] 在 `AGENTS.md` 新增“每次先输出会话级 TODO（3-7条动作要点），用户回复‘执行’后再动手”的强制规则~~
- [x] ~~[P1] 规则口径与“只记录动作要点，不写过程细节”保持一致~~

## 2026-02-13 Updated Plan 清单规则

- [x] ~~[P0] 将“会话级 TODO”升级为“Updated Plan（带勾选框清单）”规则~~
- [x] ~~[P0] 明确“未收到用户回复‘执行’前，不动手修改文件或运行命令”~~
- [x] ~~[P1] 明确执行中持续更新勾选状态（□/✓）~~

## 2026-02-11 S0 课程生效修复（仅S0）

- [x] ~~[P0] S0 课程调度改为按本次训练迭代归一化（不再依赖固定 1000）~~
- [x] ~~[P0] S0 难度每轮即时下发到全体 env，消除 reset 才生效的延迟~~
- [x] ~~[P0] 按用户要求保持最小改动：不新增 S0 课程日志项~~
- [x] ~~[P1] 语法检查：`train_highlevel.py` 编译通过~~

## 2026-02-11 清理无关diff + 专家朝向/质心跟随增强

- [x] ~~[P0] 清理无关 diff：回退 `__pycache__/*.pyc` 与训练产物未跟踪目录~~
- [x] ~~[P0] 专家控制增强：保持朝向优先，同时提升转弯段质心贴合~~
- [x] ~~[P1] 语法检查：`expert_s0_follow.py` 编译通过~~
- [ ] [P1] 短跑验证：观察转弯段机头一致性与质心贴合

## 2026-02-11 Expert 改为朝向优先控制（Heading-first）

- [x] ~~[P0] 专家控制增加朝向优先门控：大角误差时强抑制平移（先转向再推进）~~
- [x] ~~[P0] 同步增强角速度权重：角误差越大，`omega` 放大越明显~~
- [x] ~~[P1] 语法检查：`expert_s0_follow.py` 编译通过~~
- [ ] [P1] 短跑验证：观察机头朝向一致性与转弯段质心贴合

## 2026-02-11 EGPO 参与率固定平台期（前10%）

- [ ] [P0] `alpha` 调度加入固定平台期：前 `10%` 总迭代 `alpha=1.0`
- [ ] [P0] 平台期后在接口窗口内按既有 schedule 衰减，不新增参数
- [ ] [P1] 日志打印补充平台期长度，便于核对
- [ ] [P1] 语法检查并等待短跑验证

## 2026-02-11 Expert 精度跟随增强（转弯偏差修复）

- [x] ~~[P0] 专家控制从纯位置反馈升级为“前馈+反馈”：引入目标速度在机体系的 along/perp 前馈~~
- [x] ~~[P0] 提升跟随带宽：`kff=1.0`、`v_along_max=1.0`、`v_perp_max=0.12`、`k_yaw=1.2`~~
- [x] ~~[P0] 增加转弯状态补偿：大转角时降前向、增横向与角速度，降低弯道切角误差~~
- [x] ~~[P1] 语法检查完成~~
- [ ] [P1] 短跑复测：检查转弯段质心贴合误差与机头朝向一致性

## 2026-02-11 朝向一致性二轮修复（诊断口径 + 去侧移）

- [x] ~~[P0] 修正 `EGPO yaw response match` 统计口径：使用 `post_info.cmd_slew[:,2]`（实际下发角速度），并提高有效阈值以抑制噪声~~
- [x] ~~[P0] 专家去侧移最小调参：`kp_perp 0.6->0.35`，`v_perp_max 0.15->0.08`（保持 `omega` 正号方案）~~
- [x] ~~[P1] 语法检查完成~~
- [ ] [P1] 短跑复测：对比朝向一致性与 `EGPO yaw response match`

## 2026-02-11 专家朝向稳定化回退（原地绕圈回归）

- [x] ~~[P0] 回退 `expert_s0_follow.py` 的“目标点朝向融合 + 状态权重”改动，恢复到单一 `dir_body_angle` 角速度控制~~
- [ ] [P1] 根因排查：区分“专家角速度映射问题”与“低层执行/观测口径问题”

## 2026-02-11 专家朝向稳定化（符号约定 + 距离带降权）

- [x] ~~[P0] 专家 `omega` 恢复到已采用的转向符号约定（避免左右反向回归）~~
- [x] ~~[P0] 目标点朝向项改为状态相关权重：跟随距离带内降权，低速时降权，降低近目标抖动~~
- [x] ~~[P1] 修正 S0 注释与当前 `heading_scale=0.08` 一致~~
- [ ] [P1] 短跑验证：确认机头反向减轻且无原地绕圈回归

## 2026-02-11 专家朝向融合（目标运动方向 + 目标点）

- [x] ~~[P0] 仅修改 `expert_s0_follow.py` 的 `omega` 计算：加入目标点朝向角并与运动方向角固定加权融合~~
- [x] ~~[P0] 不新增参数，不修改任何 offset，不改线速度控制链~~
- [x] ~~[P1] 语法检查：`expert_s0_follow.py` 编译通过~~
- [ ] [P1] 短跑验证：观察“机头反向”是否减轻且无原地绕圈回归

## 2026-02-11 硬性指标（S0 offset 冻结）

- [x] ~~[P0] `hex_s0_follow` 的 `navigation.heading_offset_rad` 必须恒等于 `0.0`~~
- [x] ~~[P0] `hex_s0_follow` 的 `reward_cfg.heading_offset_rad` 必须恒等于 `0.0`~~
- [x] ~~[P0] 未经用户明确批准，禁止修改上述两个 offset 字段（以本条为后续执行基线）~~

## 2026-02-11 S0 朝向修正（offset固定0，仅调scale）

- [x] ~~[P0] 固定约束：S0 `navigation.heading_offset_rad=0.0` 与 `reward_cfg.heading_offset_rad=0.0`，后续不再修改~~
- [x] ~~[P0] 仅调整 S0 `reward_cfg["heading_scale"]`：`0.0 -> 0.04`~~
- [x] ~~[P1] 语法检查：`hex_scenes_config.py` 编译通过~~
- [ ] [P1] 训练短跑验证：观察机头朝向是否改善且无原地绕圈

## 2026-02-11 S0 朝向对齐修复（机头朝向与目标运动方向一致）

- [x] ~~[P0] 统一 S0 heading 参考：`navigation.heading_offset_rad` 恢复到 `+pi/2`（与项目默认 +Y 前向契约一致）~~
- [x] ~~[P0] 同步 S0 奖励朝向参考：`reward_cfg.heading_offset_rad` 恢复到 `+pi/2`~~
- [x] ~~[P1] 启用小权重朝向约束：S0 `reward_cfg.heading_scale` 从 `0.0` 调整到保守正值（先用 `0.08`）~~
- [ ] [P1] 完成后进行短跑验证：重点检查“机头朝向 vs 目标运动方向”是否仍反向

## 2026-02-11 S0 风险修复（课程覆盖/冻结语义/成功时长/EGPO 专家）

- [x] ~~[P0] 修复 `scene_difficulty_override` 注入链路：高层 wrapper 无条件写入 env，环境侧提供显式属性，保证 S0 难度课程实际生效~~
- [x] ~~[P0] 修复 moving target 冻结语义：部分 env reset 时冻结窗口内目标保持静止（不漂移）~~
- [x] ~~[P0] 修复 S0 success 时长口径：成功步数按高层 `high_level_dt` 计算，避免 decimation 改动导致判定秒数偏移~~
- [x] ~~[P1] 调整 S0 EGPO 专家沿轨控制：允许固定小幅负向速度，支持超前时回退纠偏（不新增可选参数）~~
- [x] ~~[P1] debug 可视化：reset 后清空目标/机器人轨迹绘制，避免跨 episode 轨迹残留~~

## 2026-02-11 EGPO 朝向口径修复（S0 follow 专家）

- [x] ~~[P0] 修复 EGPO 专家 `robot_heading` 来源：不再使用 `state[:,2]`，改为由 `env.env.root_states` 四元数现算（与环境坐标契约一致）~~
- [x] ~~[P1] 增加 debug 方向诊断：统计“机器人前向 vs 目标运动方向”对齐度（mean cos / p95角度）用于排查反向跟随~~

## 2026-02-11 EGPO 转向符号修复（S0 follow 专家）

- [x] ~~[P0] 修复 S0 expert `omega` 符号：目标右转时机器人同向转（消除左右转反向）~~
- [x] ~~[P1] 增加 debug 转向一致性诊断：统计 expert `omega` 与目标方向夹角符号的一致率~~

## 2026-02-11 EGPO 转向回归修正（基于实机观测）

- [x] ~~[P0] 回滚 S0 expert `omega` 符号到原实现（修复“开局原地绕圈”回归问题）~~
- [x] ~~[P1] 将诊断改为“命令角速度 vs 实际 yaw 响应”符号一致率（替代自洽但不可靠的几何符号诊断）~~

## 2026-02-06 新会话交接摘要

- [x] ~~生成 `CONTEXT_HANDOFF_SUMMARY.md`（可直接用于新对话接手继续推进）~~

## 2026-02-06 代码执行缓存区（规则与模板）

- [ ] 新增 `代码执行缓存区.md`：作为“单次执行缓存”，执行完成后自动清空回模板（无需用户确认）

## 2026-02-06 S0 高层训练 reset 链路修复（语义统一 + 观测一致）

- [x] ~~[P0] 高层 done 语义统一：汇总 reach/lost/timeout 的 reset 触发，避免分支漏 reset~~
- [x] ~~[P0] 手动 reset 后观测刷新：保证 next_obs 与 reset 后 state 对齐（robot_state/goal/depth）~~
- [x] ~~[P1] 回归检查：低层 auto-reset（done_during）与高层 manual reset 无重复/冲突~~

## 2026-02-06 轨迹清空与 reset 时序对齐

- [x] ~~[P0] 关闭默认周期清空轨迹线，仅保留 reset_idx 内清空（清空时机与 reset 对齐）~~
- [x] ~~[P1] 保留可选周期清空开关（默认关闭），便于长时调试按需开启~~
- [x] ~~[P1] 最小回归：确认语法通过且 reset 后仍会清轨迹~~

## 2026-02-04 S0 可学性修复（出生朝向锁定 + 目标配套 + 距离奖励增强）

- [x] ~~[P0] S0 reset 原子性：先锁定 robot yaw≈0（±2deg，小抖动）再重置 moving target（目标始终在视野中心附近并沿 +Y 运动）~~
- [x] ~~[P0] 方向契约（以 S1 为准）校验：修正 goal_buf 计算为 (x_right, y_forward)=dot(delta, right/fwd)，排查“出生后全体后退”~~
- [x] ~~[P0] Debug：训练脚本增加 `--force_cmd_y`（全程强制 cmd=[0,+v,0]）验证命令方向是否反了~~
- [x] ~~[P0] S0 reset 强制同步 robot root_state：位置/速度/朝向写回 sim，避免“机器人沿用上回合末位置继续走”~~
- [x] ~~[P1] reset 可视化清屏去重：同一 reset 内只 clear_lines 一次，避免干扰观察~~
- [x] ~~[P1] S0 follow 奖励增强：加大“保持 1m 距离”的奖励（含 band reward 的权重），避免早期被惩罚项淹没~~

## 2026-02-04 S0 Reward Cleanup（通用骨架继承 + S0 置零无关项）

- [x] ~~[P0] S0: 显著降低 target_center_scale（避免前期绕圈/原地转）~~
- [x] ~~[P0] S0: reward_cfg 覆盖并置 0：passable_align/crossable_align/gate_smooth/risk_barrier/time/collision~~

## 2026-02-04 S0 Crash/Explode Hotfix（viewer项裁剪 + plane分离）

- [x] ~~[P0] S0: 将 mesh_type 改为 plane（避免 env_origins 重叠带来的潜在 PhysX GPU 崩溃/异常）~~
- [x] ~~[P0] S0: 显著降低 target_visible_scale（避免视野罚项主导训练）~~
- [x] ~~[P0] train_highlevel: 对 target_center/visible 的 margin/excess 做 clamp（避免平方爆炸导致 KL 发散）~~
- [x] ~~[P0] S0: 允许 plane 启动：HexS0FollowCfg.terrain.debug_allow_plane=True（通过 HexGround mesh_type 检查）~~

## 2026-02-03 Hotfix（review 修复：S2 layout/公开接口/小优化）

- [x] ~~[P0] 修复 S2 layout_modes/layout_mode_probs 选择逻辑：difficulty>0.5 时必须使用 hard 的概率分布~~
- [x] ~~[P1] 去除对 CommandPostProcessor 私有字段 `_beta_safe_dist` 的访问：新增公开 getter 并在 gate_safe_clamp 使用~~
- [x] ~~[P2] 清理 target_lost_steps 重复清零（仅保留 done_any 统一清零）~~

## 2026-02-03 Stage S（S1-moving 强制冲突：目标过门洞脚本）

- [x] ~~[P0] 新增任务 `hex_s1_follow_moving`：启用 moving_target_mode=`s1_gate_script`，follow 距离 1m + 视野窗口 + 丢失K步reset~~
- [x] ~~[P0] HexGround 实现 s1_gate_script：gate-by-gate 阶段机（Approach/Align/Pass/Post），门洞处小偏置，出门洞后小横移~~
- [x] ~~[P0] 性能：目标更新频率对齐 scene_high_dt=0.1（10Hz），避免 4096 并行吞吐掉下去~~

## 2026-02-03 Hotfix（review：beta插值去重 + S1门宽边界 + 注释）

- [x] ~~[P0] CommandPostProcessor：抽取 beta 插值为内部方法，消除 get_effective_params/process 重复代码~~
- [x] ~~[P1] S1 门宽约束：默认仅考虑当前 gate_idx 的门；异常边界回退到“取最窄门宽”的保守策略，并加注释说明~~
- [x] ~~[P2] RNG 注释：说明 RandomState(seed) 独立于全局 numpy state（不改 seed 乘子以保持复现）~~
- [x] ~~[P3] scripted mode 参数注释：moving_target_turn_rate_max 等在 s1_gate_script 中不使用（保留无害）~~

## 2026-02-03 Debug（viewer 可视化：目标/轨迹）

- [x] ~~[P1] debug_viz：画出所有 env 的移动目标点+目标轨迹（绿色）与机器人轨迹（红色），10Hz 增量绘制并定期清屏~~

## 2026-02-03 S0 训练可学性（目标课程 + 速度上限 + 方向契约）

- [x] ~~[P0] S0 目标运动课程：difficulty=0 时 v_typical=0.2；降低早期转弯/加速度/切换频率；difficulty 提升时更快更复杂~~
- [x] ~~[P0] S0 高层速度上限匹配目标：max_lin_vel_command=1.2（post_processor/cmd_scale 同步）~~
- [x] ~~[P0] 方向契约统一（以 S1 为准）：goal_buf 统一为 (x_right, y_forward)，避免 view-centering/reward/policy 口径冲突~~
- [x] ~~[P1] debug_viz：任意 reset 后清空所有轨迹线（避免越画越乱）~~

## 2026-02-03 S0 Early-Stage Learnability（目标对齐 + 起步冻结 + 最低速度 + 回合长度）

- [x] ~~[P0] S0 reset：目标点强制落在机器人正前方（视野中心附近），横向偏置=0~~
- [x] ~~[P0] S0 reset：目标静止 freeze=1.5s 后再开始运动（避免出生即跑飞）~~
- [x] ~~[P0] S0 最低课程：v_min=v_typical=0.05m/s，difficulty=0 只前进（不横移/不后退/不对角）~~
- [x] ~~[P0] S0 增加回合长度：episode_length_s=45~~

## 2026-02-02 Stage 0.5 训练架构收口（ABI/口径先行）

- [x] ~~P0: Gate 链路 ABI 收口：训练时区分 y(raw) 与 y_eff（先令 y_eff=y），env.step 传 y_eff；记录 cmd_F/cmd_A 与 risk_F/risk_A（proxy）~~
- [x] ~~P0: 增加“沿候选命令方向的最小 clearance”近似（cone-min over occupancy），作为 risk_F/risk_A 的统一 proxy~~
- [x] ~~P0: post_info 补齐：cmd_F/cmd_A、y_raw、y_eff、clearance_F/A、risk_F/A（仅记录，不影响训练）~~

## 2026-02-02 Stage S 训练场景设计优化（S1/S2）

- [x] ~~P0: S1 门洞宽度分布对齐 V7：difficulty 插值范围改为 [1.0, 0.85]，删除不现实的 0.65~~
- [x] ~~P0: S2 增加可复现分布模式：poisson / cluster / lane（三选一或按概率混合），并写入 terrain.meta~~
- [x] ~~P0: S2 配置补齐 layout_modes/layout_probs 与 lane/cluster 参数（S2 与 S2Large 同口径）~~

## 2026-02-02 Stage A' β评测旋钮（后处理约束族联动）

- [x] ~~P0: CommandPostProcessor.process 支持 beta 输入并实现 safe/max_cmd/max_delta/risk_gain 插值（beta=None 时保持旧行为）~~
- [x] ~~P0: train_highlevel/play_highlevel 增加 --beta，并在 env.step 内将 beta 传入 CommandPostProcessor.process（beta=None 不改变旧行为）~~
- [x] ~~P0: post_info 统一透出 beta 生效后的 safe/free/max_cmd/max_delta/risk_gain（日志/论文图口径一致）~~

## 2026-02-02 Hotfix（review 修复：pyc/dtype/beta一致性/S2小坑）

- [x] ~~P0: 从 Git 停止跟踪 `legged_gym/utils/__pycache__/terrain.cpython-38.pyc`（不改 .gitignore）~~
- [x] ~~P0: CommandPostProcessor 的 `max_cmd/max_delta` dtype 对齐为 float32，并在 process() 强制与 cmd.dtype 对齐（避免隐式 float64）~~
- [x] ~~P1: gate_safe_clamp 阈值与 --beta 生效后的 safe_distance 对齐（clamp/后处理同口径）~~
- [x] ~~P1: S2 cluster center：cx≈0 时随机左右偏移，避免 sign(0) 落入 clear band 导致重采样~~

## 2026-02-02 Stage S0（平地移动目标跟随：先练稳定跟随+视野居中）

- [x] ~~P0: 新增 `hex_s0_follow` 任务：flat heightfield（terrain_type=`s0_follow_plane`）+ 关闭 spawn_edge~~
- [x] ~~P0: HexGround 增加移动目标生成器（difficulty 越高：切换更频繁、速度/方向更复杂；v_max=1.2）~~
- [x] ~~P0: 目标输入与奖励口径：follow_distance_desired=1.0m；reach 逻辑禁用（不以“接触目标点”为结束）~~
- [x] ~~P0: 视野居中口径：soft=0.35*fov、hard=0.70*fov；S0 连续 K=5 超出 hard 判丢失并 reset~~
- [x] ~~P0: S0 训练稳定性：暂时关闭 target_lost 硬重置（target_lost_k=0），仅保留 target_center/visible shaping（S1 再启用硬重置）~~
- [ ] P0: S0 完成后：S1 resume（Follow expert 继承稳定跟随能力，再引入门洞冲突）

## 2026-01-30 机器人参数盘点（URDF + expert/ground 约束）

- [x] ~~P0: 统一文档口径：y/y_eff 为 Follow 权重，w 为冲突强度（需取 1-w 融入 y_eff），修复 v6/v7 公式自相矛盾~~
- [x] ~~P1: 基于训练用 URDF 与 legged_gym/envs/hex_v4/expert.py 汇总机器人物理参数（质量/惯量/几何包络/关节限位/最大速度角速度等）~~
- [x] ~~P1: 精简 ROBOT_SPECS.md：仅保留 URDF/expert/训练配置的硬指标，删除推算与口径混用项~~

## 2026-01-28 Stage 0 协议收口（seed / goal_th=0.1 / 指标口径）

- [x] ~~P0: train_highlevel + play_highlevel 增加 --seed，并确保传入 env 创建流程~~
- [x] ~~P0: goal threshold 兜底统一为 0.1（navigation_env / hex_terrain_config / hex_ground / hex_terrain + train_highlevel 同步常量）~~
- [x] ~~P0: 训练链路新增 post_info 透出 + CmdJerk / NearMiss / GateSwitchRate 指标（TensorBoard）~~
- [x] ~~P0: 修复 jerk 统计跨 episode 差分 + near-miss 指标命名为 excess~~
- [x] ~~P0: seed 覆盖写死在 train_highlevel/play_highlevel（get_cfgs 后、make_env 前，env_cfg.seed=argseed）~~
- [x] ~~P0: GateSwitchRate/GateYChange 统计屏蔽跨 episode Δy~~

## 2026-01-27 S3–S6 classic heightfield 入口打通（无动态 actor）

- [x] ~~Must: S3–S6（含 Large）配置补齐 terrain_type~~
- [x] ~~Must: terrain.py 追加 S3–S6 classic heightfield 生成分支（走契约视图）~~
- [x] ~~Must: Terrain.make_terrain 支持 s3/s4/s5/s6 terrain_type 别名映射~~
- [ ] Must: hex_s3/hex_s4/hex_s5/hex_s6 最小启动验证（num_envs 小）

## 2026-01-27 S1 出生段/门洞段/目标段契约（避免重叠）

- [x] ~~P0: S1 明确走廊局部坐标契约（+Y 走廊轴，y∈[-L/2,+L/2]，spawn 在起点段，gate/goal 在中后段）~~
- [x] ~~P0: s1_corridor_gate_terrain 加入 gate 排除区间与硬报错（含 spawn/goal 段保护）~~
- [x] ~~P0: corridor_gates 写入 meta 前按 y0 升序排序~~
- [ ] P0: 最小验证（hex_s1 固定 seed）：spawn 段、gate 段、goal 段互斥且无 reset-loop

## 2026-01-27 S2 出生点安全通道修复（clear_band）

- [x] ~~P0: _apply_scene_spawn 对 s2_forest 强制出生在 clear_band 安全通道内，避免出生即碰撞~~
- [ ] P0: 最小验证（hex_s2 固定 seed）：出生 x 均落在 clear_band 且不再崩溃

## 2026-01-27 导航训练链路修复（S3-6 解锁 / num_envs / depth / goal 阈值）

- [x] ~~P0: train_highlevel 放开 hex_s3/4/5/6/hex_mix_gate 的硬拦截（改为 warning）~~
- [x] ~~P0: train_highlevel 的 --num_envs 默认改为 None，仅显式传参才覆盖 cfg~~
- [x] ~~P0: Teacher 不创建 depth buffer（仅 Student/camera_enable 才创建）~~
- [x] ~~P0: goal_reached_threshold 与 reward_cfg.goal_reach_threshold 统一为 0.1（含一致性断言）~~

## 2026-01-27 相机关闭时 step_separate 修复 + .pyc 清理

- [x] ~~P0: step_separate 相机关闭不触碰 depth buffer；相机开启时惰性创建~~
- [x] ~~P0: .gitignore 确认/添加 __pycache__/ 与 *.pyc，清理已跟踪 .pyc~~

## 2026-01-27 separated 观测噪声维度修复（headless 检查）

- [x] ~~P0: compute_observations_separated 噪声 slice 按实际 buffer 维度切片，并加长度校验~~

## 2026-01-27 S1 spawn 放置优化（避免 reset-loop）

- [x] ~~P0: S1 spawn x 采样扣除 clearance，y 采样避开 gate 段，失败走 deterministic fallback（一次性 warn）~~

## 2026-01-27 调试经验更新（S1 2048 并行稳定）

- [ ] P0: 将 S1 2048 并行稳定经验写入 AGENTS.md“调试总结”

## 2026-01-27 调试总结拆分为独立文件

- [x] ~~P0: 将 AGENTS.md 的“调试总结”剪切到 DEBUG_SUMMARY_CN.md，并在 AGENTS.md 留引用~~

## 2026-01-27 调试总结追加（S1 2048 并行经验）

- [x] ~~P0: 向 DEBUG_SUMMARY_CN.md 追加 S1 2048 并行稳定经验（详细版）~~

## 2026-01-27 调试总结追加（简版 checklist）

- [x] ~~P0: 向 DEBUG_SUMMARY_CN.md 追加简版 checklist~~

## 2026-01-26 训练链路收口 - debug/plane + 可复现 + grid 透明

- [x] ~~Must: 新增 hex_debug_plane 任务配置，工具脚本统一改用~~
- [x] ~~Must: hex_ground debug_allow_plane 放行 plane；训练主线仍严格 terrain_v2~~
- [x] ~~Must: train_highlevel 默认任务改为 hex_s1~~
- [x] ~~Must: terrain_v2 shuffle 可复现（显式 seed）~~
- [x] ~~Must: terrain_v2 auto grid 支持部分指定 + 启动日志打印关键参数~~
- [x] ~~Must: SceneSpec helper 生成 rect_hf（debug/s2 接入）~~
- [x] ~~Must: terrain_v2 SubTerrain 轴序显式映射 + 日志/可审计~~
- [ ] Must: A/B 测 GPU vs CPU pipeline（同一地形/num_envs）
- [x] ~~Must: _create_heightfield 恢复 swap 口径（与 hex 语义一致）~~
- [x] ~~Must: heightfield 传入 PhysX 前强制 C-contiguous~~
- [x] ~~Must: CPU 模式下 actuator_net 允许 map_location=cpu（用于 A/B 证伪）~~
- [x] ~~Must: terrain_v2_max_rows 仅限 hex_calib + auto_grid 优先级日志~~
- [x] ~~Must: terrain_v2_max_tot_rows/cols（仅 hex_calib）+ auto_grid 像素预算约束（deterministic）~~

## 2026-01-26 经典 legged_gym 回退（hexpod 口径）

- [x] ~~Must: 删除 terrain_v2 全链路与配置字段（仅保留 classic）~~
- [x] ~~Must: Terrain 经典 TileGrid + 固定 5x10 网格，num_envs 复用 tile~~
- [x] ~~Must: _create_heightfield / _get_env_origins 复刻 hexpod 口径~~
- [x] ~~Must: S1/S2 classic builder（seed 可复现 + tile_meta）~~
- [x] ~~Must: hex_s1/hex_s2 配置收口（terrain_type + 固定网格）~~
- [x] ~~Must: tools/scene_audit 改为 classic 口径~~

## 2026-01-26 classic 主线稳定性修复（spawn/工具/S2审计）

- [x] ~~P0: _apply_scene_spawn 不再依赖 scene_generator（S1 出生必须在走廊内）~~
- [x] ~~P0: 删除 legged_gym/envs/hex_v4/terrain_v2/ 并迁移 tools/debug_s2_scene.py 到 classic builder~~
- [x] ~~P1: S2 meta 记录实际放置数量（避免审计误导）~~
- [x] ~~P1: S1 wall_thickness_m 实现有限厚度（不再无限填满）~~

## 2026-01-26 classic 拼图轴序适配（SubTerrain shape）

- [x] ~~P0: add_terrain_to_map 形状判定后必要时转置；env_origin_z 同口径采样；首次触发仅日志一次~~

## 2026-01-26 S1 出生与朝向收口（classic）

- [x] ~~P0: S1 创建时禁用 jitter（避免初始 pose 偏移出走廊）~~
- [x] ~~P0: _apply_scene_spawn 强制 margin clamp + 非法范围硬报错~~
- [x] ~~P0: S1 出生朝向强制沿 +Y（保留 yaw 抖动）~~
- [x] ~~P0: 修正 S1 出生朝向公式（heading_offset 采用加法，避免垂直墙）~~
- [x] ~~P0: S1 spawn 依赖 scene_spec_cache；缺失时硬报错并提示 mesh_type/terrain_type/tile_meta~~

## 2026-01-26 轴向契约与校准验收（classic）

- [x] ~~P0: 写死轴向契约（World/Tile/Heightfield），作为单一真源~~
- [x] ~~P0: 统一唯一映射层到 add_terrain_to_map（含 env_origin_z 同口径）~~
- [x] ~~P0: debug_axis 自动验收（+Y 单调、+X 恒定，失败即报错）~~
- [x] ~~P0: 启动日志一次性打印 tile/subterrain 轴映射（none/transpose）~~

## 2026-01-26 S1 朝向/重生收口（classic）

- [x] ~~P0: hex_s1 系列将 heading_offset 置 0（机体 +Y forward，避免再偏转）~~
- [x] ~~P0: hex_s1 reward_cfg 同步 heading_offset 置 0（避免奖励方向矛盾）~~

## 2026-01-26 文档补充 - 调试总结

- [x] ~~P0: AGENTS.md 新增“调试总结”章节（轴向契约 + 唯一映射层 + 校准验收 + heading_offset 经验）~~

## 2026-01-26 工具链收口 - debug 入口与误用提示

- [x] ~~Must: hex_ground.py __main__ 改为 hex_debug_plane~~
- [x] ~~Must: hex_ground 误用提示升级（明确容器任务 + 示例命令）~~
- [x] ~~Must: 新增 hex_debug_heightfield 任务并接入 RGB/Depth 联合测试~~

## 2026-01-26 文档更新 - 训练指令与场景说明

- [x] ~~Must: 训练指令.txt 按现有代码更新命令与场景说明~~
- [x] ~~Must: 训练指令.txt 补充 gate_width/door_width 映射与权威字段说明~~

## 2026-01-23 18:07:54 - terrain_v2 重构（阶段 A）

- [x] ~~Must: 新增 terrain_v2 + debug_axis_calib + S1/S2 + backend/contracts/audit~~
- [x] ~~Must: terrain.py 接入 terrain_v2（hex 显式开关）+ unique tile + auto expand + 禁止 legacy fallback~~
- [x] ~~Must: 更新 hex_ground/hex_s1/hex_s2 配置与训练入口日志~~
- [x] ~~Must: 删除旧 scene_gen_v2/scene_manager/terrain_builder 与旧引用~~

## 2026-01-23 20:35:10 - hex 场景配置合并

- [x] ~~Must: 合并 hex_calib/hex_s1..s6/hex_mix_gate 配置到单文件 hex_scenes_config.py~~
- [x] ~~Must: 更新所有 import 引用并删除旧配置文件~~
- [x] ~~Must: 验证无残留引用（rg）并更新 CHANGELOG_CN.md~~

## 2026-01-23 21:05:30 - terrain 变更复核与旧文件恢复

- [x] ~~Must: 恢复被删除的 terrain 相关 legacy 文件（仅保留、不使用）~~
- [x] ~~Must: 复核 terrain.py 与 terrain 入口改动，确保默认路径不受影响~~
- [x] ~~Must: 重新梳理引用，保证 legacy 入口不再被调用~~

## 2026-01-23 21:25:10 - hex_terrain 硬报错 + 删除旧配置文件

- [x] ~~Must: 所有入口对 hex_terrain 硬报错并给出提示~~
- [x] ~~Must: 删除旧 hex_s1..s6/hex_calib/hex_mix_gate 配置文件~~
- [x] ~~Must: rg 全仓确认无残留引用并更新 CHANGELOG_CN.md~~

## 2026-01-23 21:40:20 - 高层训练链路修复

- [ ] Must: scene_type 概率全零兜底（terrain_v2）
- [ ] Must: highlevel 入口拦截未实现任务并给出清晰提示
- [ ] Must: train_highlevel task 提示优化（hex_s1/hex_s2 不误报）

## 2026-01-20 18:12:10 - V5 场景定义对齐与 TODO 收尾

- [x] ~~Must: 技术方案中 S6 定义改为“结构化 OOD hold-out”，同步训练/验证/测试划分口径~~
- [x] ~~Must: 核对 S1–S6 代码变更已落地，更新 TODO_LOG 17:25 条目为完成~~

## 2026-01-23 14:09:13 - 场景生成系统重构（阶段1：S1/S2 闭环）

- [x] ~~Must: 新建 SceneSpec/ObstacleSpec + HeightfieldBackend，统一单一真源入口~~
- [x] ~~Must: 实现 S1/S2 generator + contract，并提供 tools/scene_audit 体检入口~~
- [x] ~~Must: 训练入口接入新链路（hex_s1/hex_s2），旧 SceneManager/scene_cfg 直连下线~~

## 2026-01-23 14:34:48 - scene_gen_v2 强化（参考 parkour）

- [x] ~~Must: 统一入口打印 “Using scene_gen_v2”，并保持 legacy guard~~
- [x] ~~Must: 增加 quantizer（米→格）并在 backend 强制量化~~
- [x] ~~Must: 增加 guards（spawn/goal 清空 + edge pad）~~
- [x] ~~Must: 强化 S1/S2 contract 与 scene_audit（pass率/指标）~~

## 2026-01-23 14:49:37 - 训练指令更新（scene_gen_v2）

- [x] ~~Must: 更新 `训练指令.txt`，补充 scene_gen_v2 提示与 scene_audit 命令~~

## 2026-01-23 14:51:45 - scene_audit golden seeds

- [x] ~~Must: 写入 golden seeds 文件（tools/scene_golden.json）~~
- [x] ~~Must: 更新 `训练指令.txt` 记录 golden seeds 与 S1 现状~~

## 2026-01-23 15:22:00 - S1 outside_escape 修复与 golden seeds 重建

- [x] ~~Must: 修复 S1 corridor 外绕连通泄露（outside_escape）~~
- [x] ~~Must: 重新运行 scene_audit 生成 golden seeds 并更新训练指令~~

## 2026-01-23 15:40:00 - scene_gen_v2 tile 轴与坐标系修复

- [x] ~~Must: scene_gen_v2 输出轴语义与 tile 约定对齐（length/width）~~
- [x] ~~Must: tile 局部坐标系改为中心原点（y∈[-L/2,+L/2]）并修正 guards/contract~~
- [x] ~~Must: 复测 scene_audit 并更新 golden seeds/训练指令~~

## 2026-01-23 16:30:00 - P0/P1 虚空修复与外围墙下线

- [x] ~~Must: 对齐 HeightFieldParams 轴语义（nbRows/nbColumns 按 parkour），并以 env_origins 全落入 world 覆盖范围为通过标准（num_envs=128）~~
- [x] ~~Must: 继续保证 tile 轴顺序一致（axis0=length, axis1=width）~~
- [x] ~~Must: 关闭外围墙（edge_pad 默认 0），S1 通过“走廊墙覆盖 + 绕行外侧连通性检查”保障不可绕行~~
- [x] ~~Must: 复测 scene_audit 并更新 golden seeds/训练指令~~

## 2026-01-23 17:50:00 - _create_heightfield 轴语义与 view 对齐（parkour）

- [x] ~~Must: 对齐 nbRows/nbColumns 与 height_samples 视图语义，保持与 parkour 口径一致~~
- [x] ~~Must: height_samples 统一一维 flatten/order 与长度校验（base/hex_climb）~~

## 2026-01-23 18:05:00 - S1 连续墙 + 门洞内凸（去离散块）

- [x] ~~Must: S1 门洞改为贴合走廊墙的内凸块，移除浮空离散块效果~~

## 2026-01-23 18:25:00 - S1 连续墙强化 + 训练网格扩列

- [x] ~~Must: num_envs 超过 terrain grid 时自动扩列，避免 env 重叠导致 GPU 崩溃/虚空~~
- [x] ~~Must: S1 门洞内凸与外墙重叠，确保视觉连续~~

## 2026-01-23 19:00:00 - Heightfield 轴语义校准场景（calib_axis）

- [x] ~~Must: 新增 calib_axis 场景与 task 注册，用于 30s 轴语义校准~~

## 2026-01-23 19:20:00 - calib_axis 长宽区分（长方形）

- [x] ~~Must: calib_axis 使用明显长宽比（length!=width）便于肉眼判轴~~

## 2026-01-23 19:35:00 - scene_gen_v2 tile 维度对齐 SubTerrain

- [x] ~~Must: scene_gen_v2 SubTerrain(width/length) 与 backend tile shape 对齐~~

## 2026-01-23 13:41:24 - 高层训练崩溃与 heightfield 形状修复

- [x] ~~Must: hex_ground._reset_scene 兼容 heightfield meta(dict)，避免 build_meta 访问 static_obstacles 报错~~
- [x] ~~Must: 修复 heightfield height_samples 形状，满足 Isaac Gym nbRows*nbColumns 要求~~

## 2026-01-23 14:30:00 - Agent 角色模式开关（新对话必问）

- [x] ~~将“新对话先询问角色：1 决策辅助 / 2 编程主力”的规则写入 `AGENTS.md`~~

## 2026-01-23 15:00:00 - Agent 写入项目核心诉求与约束

- [x] ~~将“高效率训练（3090/2048/4096）+ 高仿真质量 + 泛化/Sim2Real + RAL 证据链要求”的背景与约束写入 `AGENTS.md`~~

## 2026-01-23 09:10:00 - 地形生成精简（保留必要动态障碍）

- [x] ~~Must: 移除静态 actor 生成与同步逻辑，仅保留 heightfield 静态障碍~~
- [x] ~~Must: 仅保留 S4 动态障碍 actor 路径，其它场景禁用动态 actor~~
- [x] ~~Must: 精简 SceneManager/HexGround 中 actor 相关分支与配置字段~~
- [x] ~~Must: 对齐 S5/S6 scene_type 分支以保证 heightfield 生成~~

## 2026-01-20 19:20:04 - 训练指令清单更新（V5 全链路）

- [x] ~~Must: 更新 `训练指令.txt`，补齐 Follow/Avoid/Gate 训练与微调、Test-ID/Test-OOD/Hold-out、play/eval 相关命令~~

## 2026-01-20 19:34:20 - 修复机器人 root_states 索引（多 actor 场景）

- [x] ~~Must: 记录 robot actor indices，root_states 只指向机器人，障碍写回 all_root_states~~

## 2026-01-20 19:42:30 - 修复场景 affordance NaN（GT rasterize）

- [x] ~~Must: _compute_gt_affordance_from_scene 对 NaN/无效 cell 做防护，避免 rasterize 崩溃~~

## 2026-01-20 20:08:20 - 修复 actor 创建顺序（env 内一次性创建）

- [x] ~~Must: 基类加入创建钩子，HexGround 在每个 env 内创建 robot+障碍 actor，消除 creation order warning 风险~~

## 2026-01-20 20:18:30 - 修复 DOF reset 使用 actor indices

- [x] ~~Must: LeggedRobot._reset_dofs 使用 robot_actor_indices，避免多 actor 场景非法访问~~

## 2026-01-21 16:00:56 - 多 actor 收口修复与走廊墙体保障

- [x] ~~Must: hex_ground._reset_dofs 改用 robot_actor_indices_int32（禁止 env_ids 直接传给 DOF indexed）~~
- [x] ~~Must: 引入 scene_static_wall_max / scene_static_block_max，S1 wall 不截断~~
- [x] ~~Must: 启动时断言 robot actor 顺序（env 内第一个）~~
- [x] ~~Should: affordance 坐标系校验开关（用于 +Y forward 验证）~~

## 2026-01-21 16:12:10 - S1 走廊障碍补齐与出生修复

- [x] ~~Must: corridor 场景补充随机障碍（不改变墙体/门洞逻辑）~~
- [x] ~~Must: corridor 场景出生点限制在通道内，避免出生卡墙~~

## 2026-01-21 18:30:00 - 按最新描述重做 S1–S6 场景语义

- [x] ~~Must: S1 走廊+门洞实现（门洞收缩段、位置/数量随机化、宽度随难度收缩），并保证 wall_truncate_rate=0~~
- [x] ~~Must: S1 走廊内障碍与出生/目标采样约束在通道内（避免出生卡墙）~~
- [x] ~~Must: S2 Forest 障碍密度/形状比例/尺寸随机化按难度生效，区间量化并保留覆盖~~
- [x] ~~Must: S3 Doorway 房间-门洞拓扑生成（门洞位置/数量/宽度随机化，房间内少量障碍）~~
- [x] ~~Must: S4 Crossing 动态横穿轨迹由反应窗口步数反推速度，记录可复现轨迹参数~~
- [x] ~~Must: S5 Sparse→Dense 分段密度与分界随机化（分界位置/密度差随难度）~~
- [x] ~~Must: S6 OOD 结构化模板（cluster/nonconvex/maze）混合采样，并支持模板固定开关~~

## 2026-01-21 19:10:00 - 同步文档与训练指令（S1–S6 语义）

- [x] ~~Must: 更新 `训练指令.txt` 的场景映射与任务说明（S1=Corridor, S2=Forest, S3=Doorway, S6=OOD mix）~~
- [x] ~~Must: 更新 `技术方案/技术方案V4_完整统一版.md` 的 S1–S6 描述与 Train/Val/Test 口径~~
- [x] ~~Should: 更新 `TRAINING_PIPELINE_V5_CHECKLIST.txt` 补充 S1–S6 映射说明与 S6 模板开关~~

## 2026-01-21 19:24:00 - 训练指令更新与场景说明补充

- [x] ~~Must: 更新 `训练指令.txt` 的最新训练指令（S2 Follow/Avoid 主训、S1 Avoid 微调、S4 Gate、Mix Gate）~~
- [x] ~~Must: 在 `训练指令.txt` 中补充 S1–S6 场景细节说明与 S6 模板开关提示~~

## 2026-01-21 19:46:00 - S2 场景排查脚本

- [x] ~~Must: 新增 debug 脚本输出 hex_s2 的场景配置与障碍数量摘要（不触发 isaacgym）~~

## 2026-01-21 20:05:00 - 训练输出精简与 debug 开关

- [x] ~~Must: train_highlevel 非必要输出改为 --debug 控制，默认安静~~
- [x] ~~Should: play_highlevel 同步 debug 开关控制诊断输出~~

## 2026-01-22 09:10:00 - debug 模式打印障碍落地位置

- [x] ~~Must: train_highlevel --debug 下打印前 N 个静态障碍位置/尺寸与 z 统计，便于排查“看不到障碍”~~

## 2026-01-22 09:40:00 - reset 时全量写回障碍 root_states（仅重采样）

- [x] ~~Must: _reset_scene 末尾全量 set_actor_root_state_tensor（只用于 reset/重采样）~~
- [x] ~~Must: debug 抽查 1-3 个静态障碍 root_state 与 spec/world 偏差（wall 尺度为主，block/pole 用自身尺度）~~

## 2026-01-22 10:05:00 - S1 spawn 采样 rng 初始化修复

- [x] ~~Must: _apply_scene_spawn 在 S1 分支前初始化 rng（防止 UnboundLocalError）~~

## 2026-01-22 10:15:00 - 障碍 collision filter 修复（防穿墙）

- [x] ~~Must: 场景障碍 actor 使用 scene_collision_filter=0xFFFFFFFF 并写回 shape filter~~

## 2026-01-22 10:25:00 - scene_collision_filter 兼容有符号 int

- [x] ~~Must: 将 0xFFFFFFFF 归一为 -1，避免 create_actor 参数类型报错~~

## 2026-01-22 10:40:00 - S1-S6 actor 场景改为 plane + env_spacing

- [x] ~~Must: hex_s1..hex_s6 全部设 mesh_type="plane"，并设置 env.env_spacing=12.0~~

## 2026-01-22 10:50:00 - mix gate 改为 plane + env_spacing

- [x] ~~Must: hex_mix_gate_config 设 mesh_type="plane"，env.env_spacing=12.0~~
- [x] ~~Should: train_highlevel --debug 打印 num_envs/env_spacing/env_origins[:3]~~

## 2026-01-22 11:00:00 - plane 模式下 scene_manager 初始化修复

- [x] ~~Must: HexGround._pre_create_envs 不依赖 self.terrain，必要时用 cfg.terrain 创建 SceneManager~~

## 2026-01-22 11:05:00 - plane 模式下 scene_specs 访问修复

- [x] ~~Must: HexGround.__init__ 访问 scene_specs 时不依赖 self.terrain~~

## 2026-01-22 11:20:00 - 隐藏问题收口修复

- [x] ~~Must: 障碍创建时 collision_filter=0，reset 同步时按 active 开/关碰撞（避免 hidden pool 崩溃）~~
- [x] ~~Should: S3/S6 wall 截断报警（非 S1 硬断言）~~
- [x] ~~Should: S1 goal 采样极端参数 fallback（避免 y_max<=y_min 卡死）~~

## 2026-01-22 12:10:00 - 隐藏问题收口修复（续）

- [x] ~~Must: 修复 S1 goal 采样 fallback（y_max<=y_min 时转入通用采样）~~
- [x] ~~Must: 确认 hidden pool 碰撞过滤策略（创建 filter=0，reset 时按 active/hidden 切换）~~
- [x] ~~Should: 校验 S3/S6 wall 截断仅告警，S1 仍硬断言~~

## 2026-01-22 12:30:00 - 墙体碰撞修复（机器人穿墙）

- [x] ~~Must: 机器人 actor shape filter 设为 scene_collision_filter（避免与障碍不碰撞）~~
- [x] ~~Should: debug 下打印一次 robot/wall 的 shape filter 值，便于确认~~

## 2026-01-22 12:40:00 - plane 模式 debug_viz 守护

- [x] ~~Must: HexGround 覆盖 _draw_debug_vis，缺少 terrain 时直接 return~~

## 2026-01-22 13:00:00 - 2048 env train_large 配置与 actor budget

- [x] ~~Must: 新增 train_large 配置（S1-S6 + mix），按配额限制 wall/block/dyn 上限~~
- [x] ~~Must: 启动时打印 actors/env 与 total_actors，并按 budget 断言~~

## 2026-01-22 13:20:00 - train_large 量化误差与 mix 统计

- [x] ~~Must: S1 门洞量化误差写入 meta 并按阈值告警（含 n_tiles==0 兜底）~~
- [x] ~~Must: mix 场景占比与 spawned 统计写入 extras，debug 下打印~~

## 2026-01-22 13:35:00 - S3 Doorway train_large 截断修复

- [x] ~~Must: Doorway 门框改走 wall pool（避免 block_max 截断）~~
- [x] ~~Should: train_large room_width 拉大（减少 outer wall tiles）~~

## 2026-01-22 13:45:00 - S6 OOD train_large 截断修复

- [x] ~~Must: OOD 迷宫/非凸墙体改走 wall pool~~
- [x] ~~Should: 确认 S6 wall_truncate_rate 仍为 warn（非 S1 不 raise）~~

## 2026-01-22 14:00:00 - S6_large 模板固定与 mix 滑动统计

- [ ] Must: S6_large 固定 ood_template=cluster（避免迷宫墙体截断）
- [ ] Must: mix 场景统计增加最近 200 reset 滑动窗口

## 2026-01-22 15:10:00 - 训练指令更新与潜在问题审计

- [x] ~~Must: 更新 `训练指令.txt` 为最新任务/large 指令，并补充场景说明~~
- [x] ~~Must: 审计潜在问题并输出清单（不改代码）~~

## 2026-01-22 15:25:00 - S6 模板口径与 mix 统计触发修复

- [x] ~~Must: 基础版 S6 使用 mix，large 固定 cluster~~
- [x] ~~Must: mix 统计窗口按 reset 计数触发（每 200 次）~~

## 2026-01-22 15:40:00 - mix 统计阈值与 large 结构截断硬断言

- [x] ~~Must: mix 统计改为按 env-reset 大阈值触发（训练 50k，debug 200）~~
- [x] ~~Must: train_large 对 S1/S3/S6 wall 截断升级为 hard assert~~

## 2026-01-22 16:10:00 - 碰撞 group=0 诊断确认

- [x] ~~Must: debug 模式打印 robot/wall 的 collision group/filter，确认 env0 group=0 问题~~

## 2026-01-22 16:20:00 - 碰撞 group=0 修复

- [x] ~~Must: robot/obstacle create_actor 使用 env_id+1 作为 collision group，避免 group=0 失效~~

## 2026-01-22 16:35:00 - group_id 未定义修复与潜在错误复检

- [x] ~~Must: _create_env_actors 内补充 group_id 定义，排除 NameError~~
- [x] ~~Must: 复检近期改动中的潜在未定义变量/口径问题~~

## 2026-01-22 17:00:00 - actor 场景 collision_filter 统一

- [x] ~~Must: robot create_actor filter 在 actor 场景下改用 scene_collision_filter~~
- [x] ~~Must: obstacle create_actor filter 改为 scene_collision_filter，保留 inactive filter=0~~

## 2026-01-22 17:55:00 - 走廊穿墙修复（group bitmask + filter 统一）

- [x] ~~Must: actor 场景下 group_id 改为固定 bitmask=1（机器人+障碍统一）~~
- [x] ~~Must: 新增统一 helper 写入 shape filter（robot/obstacle 都走同一逻辑）~~
- [x] ~~Must: debug 下 reset 后打印一次 robot/wall 的 shape filter 与 group（确认生效）~~

## 2026-01-22 22:05:00 - TerrainBuilder 重构（heightfield 主线）

- [x] ~~Must: 新增 TerrainBuilder（S1/S2/S3/S5/S6 heightfield）~~
- [x] ~~Must: Terrain 入口接入 scene_type -> TerrainBuilder（legacy 保留）~~
- [x] ~~Must: hex_s1..hex_s6 configs 切换 mesh_type=heightfield + scene_type/scene_cfg~~
- [x] ~~Must: TRAINING_COMMANDS.md 写最小验证命令~~

## 2026-01-20 11:39:46 - MoE cmd-space 重构与 V5 方案更新

- [x] ~~Must: 训练顺序调整为 Avoid → Follow → Gate（Gate 前 Avoid 碰撞率需很低）~~
- [x] ~~Must: CommandPostProcessor 放到可复用位置（不只放在 train_highlevel.py）~~
- [x] ~~Must: `rsl_rl/algorithms/high_level_planner.py` 新增 CmdVelExpert/GatePolicy，y 仅门控~~
- [x] ~~Must: `legged_gym/scripts/train_highlevel.py` 支持 follow/avoid/moe，cmd-space 融合~~
- [x] ~~Must: `legged_gym/envs/hex_v4/navigation_env.py` 增加连续风险成本（risk barrier）~~
- [x] ~~Must: 技术方案升级到 V5，写明 y 语义、MoE 结构、S1-S6/T1-T2 划分、OOD hold-out~~
- [x] ~~Must: 收尾 `legged_gym/scripts/train_highlevel.py`（清理残留字段、日志与 buffer 对齐）~~
- [x] ~~Must: 对齐 `compute_reward` 调用签名与 reward 统计键~~
- [x] ~~Must: 完成后更新 `CHANGELOG_CN.md` 记录大改动~~
- [x] ~~Should: Gate 加 y-rate penalty，训练早期可启用安全 clamp（d_min < d_safe）~~
- [x] ~~Should: CmdVelExpert 使用 tanh + 物理尺度映射，避免极值~~

## 2026-01-20 16:12:58 - S1-S6 场景落地（每场景一个 task config）

- [x] ~~Must: 新增 `legged_gym/envs/hex_v4/scene_manager.py`，提供 SceneSpec/SceneManager 与重采样机制（create/reset/level_change）~~
- [x] ~~Must: 新增 `hex_s1_config.py`…`hex_s6_config.py`（继承 HexGroundCfg），绑定 scene_type 与课程 easy/hard 区间~~
- [x] ~~Must: 接入 SceneManager 到 hex_ground/terrain 生成流程，替代 fixed_layout 的场景生成入口~~
- [x] ~~Must: S4 动态障碍 kinematic 轨迹可复现（SceneSpec 显式 path/速度），碰撞强惩罚/终止~~
- [x] ~~Must: 反应窗口步数 → 横穿速度映射（10Hz 下训练 8–20 步，OOD 4–10 步）~~
- [x] ~~Must: 任务注册 hex_s1…hex_s6 到 `legged_gym/envs/__init__.py` 并输出 SceneSpec meta 日志~~
- [x] ~~Should: 保持不新建目录，仅新增文件与接入逻辑~~

## 2026-01-20 17:03:38 - S1-S6 场景问题修复

- [x] ~~Must: S1–S6 禁用 slalom 出生逻辑（避免 terrain_proportions=[1.0] 触发）~~
- [x] ~~Must: 场景静态重采样改为“列切换”方式（scene_resample_on_reset + num_cols>1）~~

## 2026-01-20 17:25:08 - 场景与高层训练最终方案（V5）

- [x] ~~Must: SceneManager 补齐 meta（layout_id/hash/密度代理/动态体参数），并支持 mix gate 的 scene_types 采样~~
- [x] ~~Must: hex_ground 支持 on_reset/on_level_change 重采样与 actor pool 复用，动态障碍轨迹可复现~~
- [x] ~~Must: 新增 `hex_mix_gate_config.py` 并注册任务，保持 S1-S6 配置一致性~~
- [x] ~~Must: train_highlevel 接入 PPO runner，支持 --resume/--finetune_from 语义~~
- [x] ~~Must: task_registry 修复 hex_s* runner 选择（仅 PPO，不引入 EGPO）~~
- [ ] Must: 验证 scene meta、layout 变更、resume/finetune 行为，并记录最小可运行指令

## 2026-01-20 18:18:40 - 训练指令补充（V5 最小可运行）

- [x] ~~Must: 训练指令.txt 追加 S1/S2/S4/mix gate 与 resume/finetune 示例~~

## 2026-01-20 19:05:40 - V5 场景修复补全

## 2026-01-22 23:10:00 - TerrainBuilder 场景重写（heightfield 科研规范）

- [x] ~~Must: terrain_builder.py 按科研规范重写 S1/S2/S3/S5/S6（+Y 坐标、clear_rect、zig-zag 门洞、分段密度）~~
- [x] ~~Must: build_heightfield 接收 horizontal/vertical_scale，返回 (hf, meta) 并在 terrain.py 传参对齐~~
- [x] ~~Must: hex_s1..hex_s6 scene_cfg 字段名对齐新规范（clearance/length_mul/half_width_mul 等）~~
- [x] ~~Must: V5 奖励口径下沉到 hex_* config，train_highlevel 不再按 task 特判~~
- [x] ~~Must: actor 场景目标采样改为 scene_spec 线段阻挡判定，并增加可控开关~~
- [x] ~~Must: S3/S5/S6 障碍尺寸参数生效（radius/height/cluster_radius）~~
- [x] ~~Should: play_highlevel help 文案更新支持 hex_s1…hex_s6/mix gate~~

## 2026-01-20 19:34:02 - Actor pool 动态分配与 Follow 一致性

- [x] ~~Must: 静态障碍 actor pool 改为总池动态分配，记录 truncate_rate~~
- [x] ~~Must: Follow config 关闭 blocking-line；play_highlevel 保底同步~~

## 2026-01-20 12:08:32 - 修复 V5 语法与 play_highlevel 适配

- [x] ~~Must: 修复 NavigationRewardFunction.compute_reward 参数顺序（默认参数在末尾）~~
- [x] ~~Must: play_highlevel 适配 CmdVelExpert 输出与 env.step(cmd_vel)~~
- [x] ~~Must: 清理 play_highlevel 的 intensity/subgoal 旧日志字段~~
- [ ] Should: T1 弱遮挡训练仅作用于 policy 输入（不改 env 目标/奖励）

## 2026-01-20 12:18:35 - Pylance 报错清理

- [x] ~~Must: train_highlevel 增加延迟导入占位，消除未定义/可调用检查~~
- [x] ~~Must: play_highlevel 对 vision_model/obs 增加显式保护~~

## 2026-01-20 12:28:03 - T1 弱遮挡训练与 play_highlevel MoE

- [x] ~~Must: train_highlevel 增加 T1 弱遮挡（仅影响 policy 输入）~~
- [x] ~~Must: play_highlevel 支持 follow/avoid/moe 与 gate 混合演示~~
- [x] ~~Must: play_highlevel 增加 gate_y 日志~~

## 2026-01-20 12:34:41 - CommandPostProcessor 训练/演示一致性

- [x] ~~Must: play_highlevel 补齐 cmd post-processor 参数并对齐训练默认值~~
- [x] ~~Must: 训练与演示统一使用 env.post_processor 结果~~

## 2026-01-20 12:39:12 - Gate 训练梯度隔离

- [x] ~~Must: Gate 训练 expert 前向必须 no_grad + requires_grad=False~~
- [x] ~~Must: Gate 训练仅记录 gate 的 logprob/value/adv~~

## 2026-01-20 12:44:10 - Reward 语义断言（可检查）

- [ ] Must: gate_smooth 仅依赖 y_t - y_{t-1}，不得读取 intensity/subgoal（检查 `NavigationRewardFunction.compute_reward`）
- [ ] Must: risk_barrier 连续可导（至少 piecewise 连续）；collision 仍为强惩罚/终止（检查 `NavigationRewardFunction.compute_reward` 与 env 终止条件）
- [ ] Must: intensity 仅作兼容 fallback，主线训练/演示不再依赖它（日志不输出 intensity）

## 2026-01-20 12:52:20 - V5 风险点 Must-check

- [x] ~~Must: 默认 task 与论文一致（V5 默认 hex_ground；hex_terrain 标注 legacy/非主线）~~
- [x] ~~Must: Gate 输入特权信息审计（difficulty 是否进入 gate；如进入需说明来源，或移除/固定）~~
- [x] ~~Must: 风险后处理消融开关（可单独关闭 risk_scale/钳制做归因）~~
- [x] ~~Must: moe 模式 affordance 来源固定（teacher/student 在 train/test 一致并记录）~~
- [x] ~~Must: rollout horizon 合理性验证（gate 至少对比 num_steps=48）~~

## 2026-01-20 13:04:52 - 默认 aff_stack 改为 1

- [x] ~~Must: train_highlevel 默认 aff_stack=1~~
- [x] ~~Must: play_highlevel 默认 aff_stack=1~~
- [x] ~~Must: 更新 V5 训练清单默认值~~

## 2026-01-19 10:19:12 - play_highlevel 手动课程修复

- [x] ~~play_highlevel: 禁用自动课程，仅响应 A/D 手动升降级~~
- [x] ~~play_highlevel: 启动时固定 terrain_levels=0 并同步 env_origins~~
- [x] ~~play_highlevel: 修正 max_level 计算与降级逻辑，避免直接跳最高级~~

## 2026-01-19 10:33:37 - play_highlevel 门缝诊断输出

- [x] ~~play_highlevel: 输出可通行/低障方向与门控、扇区可见率等排查指标~~
- [x] ~~play_highlevel: 输出门缝偏离目标方向的夹角与判定~~

## 2026-01-19 10:43:27 - play_highlevel 诊断输出格式化修复

- [x] ~~play_highlevel: 诊断输出中 Tensor 格式化为 float，避免 format 报错~~

## 2026-01-19 10:56:05 - play_highlevel heading_offset 覆盖验证

- [x] ~~play_highlevel: 增加 heading_offset override/flip 以验证朝向口径~~

## 2026-01-19 10:58:18 - play_highlevel goal 旋转诊断输出

- [x] ~~play_highlevel: 输出 goal_raw/goal_rot 与多口径 bearing 以定位旋转符号~~

## 2026-01-19 17:30:00 - 高层 goal 旋转修正（+Y 前进）

- [x] ~~train_highlevel: 修正 goal 旋转公式并移除 guidance_goal_fix 兼容逻辑~~
- [x] ~~play_highlevel: 清理 guidance_goal_fix 相关选项与诊断输出~~

## 2026-01-19 16:13:57 - play_highlevel 引导目标修正（仅诊断）

- [x] ~~train_highlevel: passable 引导使用 +Y 前进目标（可开关）~~
- [x] ~~play_highlevel: 默认开启 guidance_goal_fix 并输出 fix 相关诊断~~

## 2026-01-26 训练链路收口 - terrain_v2 规则与语义

- [x] Must: hex_ground 强制 terrain_v2 + heightfield + 明确 scene_type(s) 提示
- [x] Must: terrain_v2 自动 grid 可复现（rows/cols 规则固定）
- [x] Must: door_width 语义钉死（生成/使用/审计一致）
- [x] Must: spawn/goal rect_hf 强制存在且清空只用量化 rect_hf

## 2026-01-27 方案文档 - PCR-Net++ (w+beta 联动后处理)

- [ ] 在 docs/archive/思路设计.md 增补第二章：PCR-Net++（替代 Delta 残差为主线）
- [ ] 明确 (y, w, beta) 分工：结构选择 / 预测式 prior / 风险预算(联动 Post-Processor)
- [ ] 写清训练策略：专家预训练 -> 冻结专家训练 PCR-Net++ -> 可选端到端微调
- [ ] 写清指标/消融：y-only vs y+w vs y+beta vs y+w+beta；beta 联动/不联动后处理

## 2026-01-27 技术方案 - V6（实验总指导大纲落盘）

- [ ] 新建 `技术方案/技术方案V6.md`：整理“最新实验设计总指导大纲”（y/w/beta 语义、训练/评测/消融矩阵、风险与失败判据）
- [ ] 扩写 `技术方案/技术方案V6.md`：补齐可执行实验指导（语义验收 checklist、beta->后处理映射规范、场景分布参数表、训练三阶段预算建议、指标口径与消融矩阵模板）

## 2026-03-06 20:15:00 - Avoid 课程继续收紧（出生/遇障/切阶段）

- [x] Must: `s_avoid_basic` 阶段1/2 障碍采样改为围绕前向通路组织，避免“有的空场、有的一出生卡死”
- [x] Must: 提高出生安全距离，并按不同障碍尺寸计算 spawn/goal 避让半径
- [x] Must: 课程切换不再只看碰撞率，增加“最近窗口确实遇到障碍”的约束
- [x] Must: 增加障碍物接触调试输出，明确区分“撞障碍”和“正常地面接触”

## 2026-03-06 20:45:00 - Avoid 完美图验证与 actor 障碍接入

- [x] Must: `gt_affordance` 在平地 actor 障碍场景下优先走 scene/actor 栅格化，不再被 heightfield/空测量支路吞掉
- [x] Must: `s_avoid_basic` 的 capsule/box/wall actor 进入完美图，先验证“正前/左前/右前能进图”
- [x] Must: 导出 raw `gt_affordance` 与 `local_map_2ch` 的 sanity 图，避免继续盲训

## 2026-03-13 22:49:31 - Avoid 课程升级判据改为分 stage 能力证明

- [x] Must: `s_avoid_basic` 课程统计改为按 stage 独立记账，切换 stage 时清空目标 stage 的滑动窗口，杜绝历史窗口继承导致的连跳
- [x] Must: `stage1->2` 与 `stage2->3` 拆成两套独立门槛与窗口，不再共用同一套 exposure/progress/success/collision 条件
- [x] Must: `stage3` corridor 收窄改为独立窗口 + 更长冷却，避免刚进 stage3 就快速连续收窄

## 2026-03-17 训练语义审计 - Avoid 碰撞统计口径收口

- [x] Must: 将 `terminal fail penalty` 从 `reward_terms["collision"]` 中拆出，避免控制台把终止失败误读成真实碰撞项
- [x] Must: 为 `avoid` 增加最小 obstacle 命中审计量，至少区分“靠近/命中 active obstacle actor”与一般身体接触

## 2026-03-17 s_avoid 物理修复 - 清理地下 actor 池残留

- [x] Must: 不再让 `s_avoid` 的 pooled 障碍共用 `(0, 0, -5)` 地下待机点，避免下一回合出现障碍从地底冒出
- [x] Must: 保留已恢复的 actor 碰撞，同时把 inactive pooled 障碍移到场景外独立停车位，保证 avoid 训练场景干净

## 2026-03-17 s_avoid 课程主线调整 - 预设场景 + 过渡 stage

- [x] Must: `s_avoid_basic` 从连续随机采样切到“预生成预设库 + 回合间切预设索引”，先保证碰撞和场景语义稳定
- [x] Must: 课程改成 `stage1 -> stage1.5 -> stage2 -> stage3`，降低 `stage1` 到混合障碍阶段的难度跳变

## 2026-03-17 融合训练顺序结论 - 先 avoid，后统一 follow 输入契约

- [ ] Must: 当前先继续推进 `avoid expert`，不要因为 follow 输入契约统一而打断避障主线
- [ ] Must: 在进入 follow/avoid 融合训练前，统一所有专家与仲裁层的目标输入口径，禁止继续依赖部署不可得的真值目标信息

## 2026-03-17 avoid 训练链路最后收口 - 固定墙 + 硬筛选 + 课程命名统一

- [x] Must: `stage4` 的墙单独处理，避免继续沿用前 3 段 pooled 可动障碍语义
- [x] Must: 预设筛选改成硬保证，禁止连续失败后静默回退到未验收预设
- [x] Must: `avoid_stage3_*` 的旧 shrink 配置与日志口径统一收成 `stage4`

## 2026-03-18 avoid 分布再收口 - 正前方目标 + 渐进式通道压迫

- [x] Must: 收紧 `s_avoid_basic` 的 `goal_range_x/y`，减少“侧边目标绕行解”，让目标更多落在障碍区正前方
- [x] Must: `stage1/2/3` 只逐步增加一个难度维度，避免早期同时增加障碍数量和通道压迫
- [x] Must: 为目标初始采样补 `retry/fallback` 诊断，并接到 TensorBoard，避免采样空间过窄却看不出来

## 2026-03-18 avoid 通路下限与目标后方约束统一

- [x] ~~Must: 将 `stage1/2/3/4` 的最低通路统一按 `0.75m` 作为硬下限，禁止任一阶段掉到该宽度以下~~
- [x] ~~Must: 将 `s_avoid` 的目标采样收口到障碍簇后方，避免目标大量落在障碍区侧方继续诱导侧绕~~
- [x] ~~Must: 补目标分布诊断（至少 `goal_behind_rate / goal_side_rate / retry / fallback`），让短训后能直接判断场景分布是否真的收口~~

## 2026-03-19 avoid 目标采样收口修复

- [x] ~~Must: 修复“目标在障碍后方”约束失效时静默退回旧分布的问题，保证 fallback 也不再落到障碍前/中部~~
- [x] ~~Must: 将目标横向中心从障碍质心改成更接近可通过缝中心的定义，减少朝障碍堆中心推进~~
- [x] ~~Must: 将 `goal_retry/fallback/behind/side` 诊断改成 iteration 级累计统计，避免只记录最后一次 reset~~

## 2026-03-19 avoid 通路中心与分阶段目标统计修复

- [ ] Must: 将目标横向中心从“整带最大空隙中心”改成“目标 y 切片上的通路中心”，避免边界大空隙继续诱导侧绕
- [ ] Must: 将 `GoalBehindRate / GoalSideRate / GoalFallbackRate` 至少拆成 `stage1-3` 与 `stage4` 两套统计，避免走廊阶段把前 3 段问题冲掉
- [ ] Must: 将 `goal_range_y` 主口径与当前更远的后方目标采样行为对齐，避免配置和真实分布继续漂移

## 2026-03-19 avoid 局部解修正 - 提前拉开微调

- [x] Must: 在保持当前前进与朝向能力的前提下，优先增强 `passable_align`，让策略更愿意沿可通行方向通过
- [x] Must: 略微上调 `risk_barrier_safe/free`，让障碍信号更早生效，减少贴边硬挤
- [x] Must: 使用当前 best checkpoint 做 `--finetune_from` 微调，不从头重训

## 2026-03-19 avoid 后续优化顺序 - 先安全再效率

- [x] Must: 先用当前 best checkpoint 做“提前拉开 + 轻微边界约束”微调，不同时引入碰撞代价和新效率奖励
- [x] Must: 只有在阶段A后仍明显蹭障碍时，才单独提高 `collision_penalty`
- [x] Must: 只有在安全问题基本压住后，才新增轻量 `path_efficiency` 去减少大弧绕行

## 2026-03-19 avoid 动作型态修正 - 一次性合并微调

- [x] Must: 同时加强朝向保持、边界约束、碰撞代价，避免继续学成“绕边+转身接近目标”
- [x] Must: 新增轻量且低风险门控的 `path_efficiency`，让低风险时更偏向短路径推进
- [x] Must: 继续使用当前 best checkpoint 做 `--finetune_from`，保留已经学到的前进与基本避障能力

## 2026-03-19 avoid 阶段A参数落地 - 封外圈 + 正前方找缝

- [x] Must: 将 `avoid_band_penalty_scale` 提到 3.0，并收窄 `avoid_band_margin_x` 到 0.30，显著提高外圈路径代价
- [x] Must: 将 `passable_sector_deg` 固定到 60°，只保留正前方可通过缝的吸引
- [x] Must: 将 `path_efficiency` 改成“每步位移在目标方向上的前向投影奖励”，直接压低大弧绕行收益

## 2026-03-19 avoid 奖励收口 - 去掉重复进度项

- [x] Must: 删除与 `approach` 语义重叠的 `path_efficiency`，回到单一主进度项
- [x] Must: 保留 `band + passable + heading/turn + collision` 作为动作型态约束主线
- [x] Must: 继续把场景纵向间隔问题留给后续单独场景对照，不再和奖励修改混在一起

## 2026-03-19 avoid 场景几何最终收口 - 核心段纵深 + y间距硬化

- [x] Must: 将纵深验收从“障碍簇包络附近存在通道”收紧到“必须覆盖障碍簇核心段”，避免空白区假通道继续过检
- [x] Must: 将 `min_y_spacing` 的放松下限收成按配置比例约束，避免名义大间隔在生成时被悄悄放到过弱
- [x] Must: 为核心段纵深补单独诊断并接到日志，后续训练里直接判断场景是不是按“横移可通过”在生成

## 2026-03-19 avoid 训练主线切换 - 固定模板替代随机预设

- [x] Must: 停用 `stage1/2/3` 的随机摆障碍与验收重试链，避免继续在不同 stage 上反复卡生成
- [x] Must: 改成固定交错模板，并保证 `x/y` 两个方向都真实可通过
- [x] Must: 最多只保留左右镜像两个固定模板，先把“前进 + 横移穿中间”的动作型态稳定学出来

## 2026-03-19 协作口径收口 - 论文导向与轻重缓急优先

- [x] Must: 默认按“最快拿到可发表结果”的标准判断优先级，不为次要完整性消耗主线时间
- [x] Must: 若固定模板、简化设定或学术默认做法已经足够支撑论文结论，优先采用，不先追求更复杂方案
- [x] Must: 只优先修会影响训练口径、关键指标、实机行为和论文叙事的问题；其余问题可明确后置

## 2026-03-19 协作口径升级 - 完整论文导向搭档提示词落盘

- [x] Must: 将“像人类研究员一样判断轻重缓急、先抓论文主线、够用就往下走”的完整协作口径写入 `AGENTS.md`
- [x] Must: 明确固定模板、简化设定、学术默认做法在当前论文主线中的优先级，避免再次在次要完整性上空转
- [x] Must: 明确只有影响论文主结论、关键实验指标、实机演示与叙事的问题才作为优先修复项

## 2026-03-19 AGENTS 文档治理 - 精简重复与过时表述

- [x] Must: 合并重复的论文导向协作规则，保留一套长期有效的完整口径
- [x] Must: 删除已被新规则覆盖的重复表述，减少后续阅读负担
- [x] Must: 保留当前仍有效的硬约束与实验口径，不误删正在使用的主线规则

## 2026-03-19 avoid 固定模板主线再精简 - 砍重引导与课程门槛

- [x] Must: 在固定模板主线下停用 `passable/crossable` 相关引导，避免继续为找缝做全图扫描
- [x] Must: 将 `s_avoid_basic` 的奖励收成主进度、碰撞、终止、边界、朝向、风险这几条主线
- [x] Must: 将课程升级条件收成 `success + collision` 两项，并把阶段窗口统一收成 `100`

## 2026-03-19 avoid 固定模板课程回调 - success/collision/progress 三项

- [x] Must: 固定模板主线下恢复 `progress` 作为课程升级兜底，避免低碰撞但未真正通过的解过早升 stage
- [x] Must: 将阶段窗口从 `100` 回调到 `150`，在保持较快升级的同时降低短时波动影响
- [x] Must: 将 `min_episodes` 回调到 `400`，避免第一轮有效学习前就切到更难 stage

## 2026-03-19 avoid 固定模板奖励回调 - 打掉后退避险坏解

- [x] Must: 恢复轻量前进驱动，避免固定模板下出现“离障碍越来越远反而 reward 更好”的局部解
- [x] Must: 恢复机体系后退惩罚，直接压制 `Body back speed` 持续升高
- [x] Must: 恢复轻量时间惩罚，避免策略通过原地拖延或慢退来拿到更低风险

## 2026-03-19 avoid 功能定位收口 - 局部短时安全通过专家

- [x] Must: 明确 `avoid expert` 的定位不是全局规划器，而是基于 `local_map_2ch + goal_buf` 的局部短时安全通过专家
- [x] Must: 明确它负责输出一小段安全、可执行、持续朝局部子目标推进的动作倾向，不负责决定整条路线从哪边绕
- [x] Must: 明确训练与奖励设计应优先服务“不撞、不过分贴边、不过久卡住、持续朝局部子目标推进”，不要把“追几何中心/追模板中缝”固化成长期主目标

## 2026-03-20 avoid 回放协议补充 - 直接查看 stage2/3

- [x] Must: 为 `play_highlevel.py` 增加 `s_avoid` 的阶段覆盖开关，允许直接回放 `stage1/2/3/4`
- [x] Must: 回放时在禁用课程的同时保留指定阶段，不再默认只能看到 `stage1`
- [x] Must: 该开关只服务当前固定模板避障诊断，不改变训练主线与课程逻辑

## 2026-03-19 avoid 固定模板再放宽 - 教学场优先

- [x] Must: 将 `stage1/2/3` 的固定障碍模板在 `x/y` 两个方向都明显放宽，先把“向前推进 + 横移通过”动作型态学出来
- [x] Must: 优先增大横向主缝宽和纵向连续纵深，不再让当前模板接近最终评测难度
- [x] Must: 保持固定模板、镜像、课程与奖励不变，只单独调整障碍坐标，保证这轮变化可解释

## 2026-03-20 s_avoid 场景几何再放大 - 间距翻倍与区域扩张

- [x] Must: 将 `stage1/2/3/4` 的固定障碍模板在 `x/y` 两个方向的相对间距整体拉大，优先解决“根本过不去”的几何问题
- [x] Must: 将 `band/core/goal` 的范围同步放大，避免新模板仍然被旧的小范围采样与诊断口径截断
- [x] Must: 先排除“模板过紧 + reset 初始重叠”导致 viewer 看起来像穿过障碍的基础问题，再继续判断策略本身

## 2026-03-20 avoid 命令数值止损 - 零缩放维 NaN 修复

- [x] Must: 修复 `CmdVelExpert.evaluate_actions` 在禁用动作维（如 `omega scale=0`）上的反变换 NaN
- [x] Must: 禁用动作维不再参与 `log_prob/entropy`，避免继续污染 `avoid` 训练统计
- [x] Must: 重新跑最小 `s_avoid_basic` smoke，确认训练不再在 `cmd_raw[...,2]` 处中断

## 2026-03-20 avoid 固定模板改成三列交错布局

- [x] Must: 将固定模板中原本每行 2 个主障碍的布局收成每行 3 个主障碍，保证行间横向错位
- [x] Must: 在保持 `x/y` 双方向可通过的前提下重排 `stage1~4` 坐标，不去改奖励与课程逻辑
- [x] Must: 回跑最小 `s_avoid_basic` smoke，确认新三列模板仍能通过当前几何验收并正常训练

## 2026-03-25 avoid 固定模板重做 - stage1 3/3 与 stage234 3/2/3 之字形

- [x] Must: 将 `stage1` 改成两行都放 3 个障碍物，宽侧逐行交错，不再沿用旧的 `3/2` 行结构
- [x] Must: 将 `stage2/3/4` 保持为 `3/2/3...` 行结构，但连 2 障碍物行也做成单侧宽通路，形成真正逐行之字形
- [x] Must: 只改 `x` 向布局，不改任何 `row_y/last_row_y/y_spacing` 参数，并用最小 smoke 验证 `retry/pfail` 与 passage 诊断

## 2026-03-26 avoid 固定模板纵向拉长 - 行距+30cm、宽侧+10cm、行级成功进度

- [x] Must: 在严格保持各 stage 分布形态不变的前提下，将相邻障碍行间距统一再拉长 `0.30m`
- [x] Must: 每行内部仅增宽宽侧通路 `0.10m`，窄侧障碍位置保持不变
- [x] Must: 将成功/进度口径改成“每无碰撞通过一行就累计一次”，并按最慢速度与最远场景同步增加回合时间
- [x] Must: 将 `align_center` 从“奖励侧向命令方向”改成“奖励真实横向归位结果”，避免假动作吃奖励
- [x] Must: 收口 `band` 激活口径，让配置、训练与调试都统一为出生即激活
- [x] Must: 将课程升级阈值改成逐行比例口径，避免只会过前一两行就过早升级
- [x] Must: 下调 `target_visible`、上调 `align_center`，减少视野约束对横移的抑制并增强远处归位信号

## 2026-03-26 avoid 外逃止损 - band 软惩罚升级为硬失败

- [x] Must: 将 `s_avoid` 的 `band` 外逃从连续软惩罚升级为高层 episode 硬终止，堵住“出带绕障”退化解
- [x] Must: 为 `band` 硬终止保留小余量，避免边界附近的正常摆动被误杀
- [ ] Must: 先保持其余奖励系数不动，只做单因素短训验证 `band_out / episode_len / progress / success`

## 2026-03-27 avoid 方向老师改成左右候选通道 clearance 差

- [x] Must: 不再用“前半平面左右最近障碍距离差”做主方向老师，改成比较左右候选通道的前向 clearance
- [x] Must: 保持 `block/free`、`lat_penalty`、`lat_clear` 主线不变，只替换“该往哪边避”的判断依据
- [x] Must: 同步更新 `play` 调试输出，让 `side_risk` 与当前真实训练老师一致，避免继续用旧 `risk_lr` 误判

## 2026-03-27 play 导出真实老师通道图

- [x] Must: 在 `play_highlevel.py` 中按时间间隔导出老师真实使用的通道图，避免只靠终端数字猜老师是否看到了通道
- [x] Must: 图上只使用当前训练真实老师的量：左右候选通道 clearance、`side_risk`、`block`、实际 `cmd_exec`，不再用旧 `passable_side` 充数

## 2026-03-28 avoid rowCmdX 从目标型惩罚切回正向动作奖励

- [x] Must: 停止使用 `|cmd_pred_x - x_target|` 目标型惩罚，改回“朝 gap 正确方向的原始 `cmd_pred_x`”正奖励，先把横移动作型态拉出来
- [x] Must: 将 `avoid_row_cmdx_scale` 提到能与碰撞项竞争的量级，先验证 `|CmdX pred|` 能否脱离 `0.01` 档

## 2026-03-28 avoid 碰撞即高层重置

- [x] Must: 将 `s_avoid_basic` 的障碍碰撞从“扣分后继续滚样本”改成高层回合直接终止，先把撞前 through 动作学干净
- [x] Must: 保持其余奖励项不动，只改碰撞后的回合生命周期，方便后续用同一 checkpoint 放开碰撞重置继续 finetune 连续避障

## 2026-03-28 avoid rowCmdX 从“只奖对方向”改成“方向错也罚”

- [x] Must: 将 `rowCmdX` 从 `clamp(cmd_pred_x * x_dir_to_gap, min=0)` 的单边正奖励，改成有符号方向奖励，让背离 gap 的横移也产生负梯度
- [x] Must: 同步更新训练统计与 `play` 调试口径，避免继续用旧的 `toward` 正半轴解释方向学习

## 2026-03-28 avoid 放大 rowGap 方向结果信号

- [x] Must: 将 `avoid_row_gap_scale` 从 `8.0` 提到 `32.0`，优先让“往正确通路靠近”的位置结果在早期就能进入可见量级
- [x] Must: 保持其余奖励、碰撞重置与课程设置不动，只做单因素验证 `rowGap -> signed_pos/neg -> succ` 这条链

## 2026-03-28 play 三路图输入因果诊断

- [x] Must: 在同一时刻同一状态下，同时比较原始图、左右翻转图、全零图的 deterministic `cmd_pred_x`，直接验证高层是否真的在用图里的左右信息
- [x] Must: 只做 `play` 调试输出，不改真实执行命令，不影响训练与评测口径

## 2026-03-28 高层 CoordConv 轴顺序对齐

- [x] Must: 修正高层 affordance encoder 的 CoordConv 坐标轴解释，使其与上游地图真实空间轴顺序 `[x_right, y_forward]` 一致
- [x] Must: 保持上游 GT 图构造、reward 几何与 `play` 左右翻图维度不变，只修正 encoder 侧坐标通道，避免继续把左右信息读成错位语义

## 2026-03-29 play encoder 特征因果诊断

- [x] Must: 在原图/翻图/清零图三路 deterministic 对照上，同时打印 `affordance encoder` 和 actor hidden 的差异，直接判断图信息是没编码出来，还是编码出来后被后续状态/目标淹没
- [x] Must: 保持真实执行命令不变，只补 `play` 调试输出

## 2026-03-30 avoid 几何口径与执行层可见性一致性修复

- [x] Must: 将 `s_avoid` 的逐行 `progress` 也改成按机身后缘计算，和 `success/cross_line` 口径统一
- [x] Must: 将高层命令日志拆成 `cmd_post` 与最终执行命令，避免 `rotate_only` 之后继续用混合口径解释训练
- [x] Must: 在 `play/eval` 里补出 `rotate_only_active` 与最终命令可见性，方便直接判断门控是否在抢控制权

## 2026-03-30 avoid 课程 success 口径修复

- [x] Must: 将课程与训练面板里的 `stage success` 从 `row_success_ratio` 改成真正的 `episode_success_flag`，让升级判断与 `play/eval` 的最终 through 成功定义一致
- [x] Must: 保留 `row_success_ratio` 作为单独统计项，避免继续把“按行无撞比例”误读成“最终 through 成功率”

## 2026-03-30 avoid 终局统计口径与行切换阈值修复

- [x] Must: 将“通过当前行”的判据收成机身后缘过 `row_y`，但将下一行 gap 切换阈值单独收成机身后缘过 `row_y + 0.15`
- [x] Must: 为 `success / collision / progress / cross_line_dist_end` 增加 reset 前终局快照，避免 `play/eval/info` 读到新回合状态
- [x] Must: 拆清 step 级与 episode 级碰撞统计命名，并在 `play` 里直接打印 `rear_y / cross_line_y / cross_line_dist / success_mask`

## 2026-03-30 avoid stage1->2 自动升级门槛放宽

- [x] Must: 仅放宽 `stage1 -> stage2` 的自动升级门槛，让当前已能在 `play` 中通过人工验收的策略尽快进入下一阶段训练
- [x] Must: 保持 `stage2/3/4` 升级阈值、奖励项和统计语义不变，避免一次改多因子

## 2026-03-30 avoid 全流程训练一致性审查与修复

- [x] Must: 沿 `env -> wrapper -> train -> eval -> play -> logging` 做一次全链路一致性审查，优先检查成功/进度/碰撞/终局快照/课程升级语义
- [x] Must: 对会污染训练结论、评测口径或日志解释的问题做最小修复，不做无关改动

## 2026-03-30 train/play 输出语义清理

- [x] Must: 拆清 `train` 输出里的 `step级 / episode级 / stage窗口级` 指标，避免同一行混不同层级
- [x] Must: 拆清 `play` 调试输出里的 `goal_dist_delta / cross_line_dist / episode_progress / success / collision`，避免继续用模糊名字误判行为
- [x] Must: 将 `play` 里的 `side(map/gt/cmd)` 改成同一当前 row 语义下可比较的三列，并修正 `row_y` 世界坐标直接索引局部图的混用

## 2026-03-31 avoid gap内横移反向切换惩罚

- [x] Must: 将 `avoid_smooth` 从泛化的 `cmd_x` 执行增量惩罚改成“仅在当前 gap 内的横移反向切换惩罚”，直接打到无意义左右翻向
- [x] Must: 补出 `in-gap` 翻向事件统计，方便直接判断这次改动有没有把 gap 内来回横移压下去

## 2026-03-31 train/play 开关率与地图调试口径拆清

- [x] Must: 将 `play` 里混在一行的 `GT raw affordance` 与 `local_map_2ch` 调试量拆开，避免继续把两路来源误当成同一老师信号
- [x] Must: 将训练输出里的 `switchRate` 显式区分 `Exec` 与 `PredInGap`，避免横向比较时误读 smooth 的实际作用层

## 2026-03-31 avoid gap内横移切换惩罚加大

- [x] Must: 在不改其他奖励项的前提下，仅上调 `gap` 内横移方向切换惩罚强度，优先压掉已进通道后的无意义左右翻向

## 2026-03-31 朝向修正退出角度下限提高

- [x] Must: 将 `rotate_only` 的退出角度下限提高到 `5°`，减少小角度误差下反复进出朝向修正

## 2026-03-31 avoid 过线与切行语义调整

- [x] Must: 将 `progress` 与最终 `success` 的过线判定统一改成“机身中心过线”，并让最后一行中心过线计入 `progress=1.0`
- [x] Must: 保持下一行 `gap` 切换更保守，仅当机身后缘超过当前行 `+0.2m` 后才切到下一行

## 2026-03-31 avoid stage2/3 升级碰撞门槛放宽

- [x] Must: 仅放宽 `stage2 -> stage3` 与 `stage3 -> stage4` 的 `collision_rate` 门槛，避免当前已具备通过能力的策略继续被过严碰撞阈值卡住

## 2026-03-31 avoid progress 终拍补齐

- [x] Must: 将 `progress / rowSuccessRatio` 的统计改成“历史最好值和当前实时过行数取最大”，避免最后一行刚过线的成功回合仍被记成 `2/3`

## 2026-03-31 avoid gap外朝向轻惩罚 + gap内切换中惩罚

- [x] Must: 将朝向惩罚限定为 `gap` 外生效，专门压住前往通道过程里因动作过猛导致的偏航
- [x] Must: 将 `gap` 内 `x` 方向切换惩罚进一步加大到中等强度，专门减少已经对准后的无意义微调

## 2026-03-31 avoid 朝向修复稀疏事件惩罚

- [x] Must: 取消连续朝向平方惩罚，改成“触发一次朝向修复就罚一次”的稀疏事件惩罚
- [x] Must: 单次朝向修复事件惩罚设为 `-0.2`，让该信号真正进入策略取舍

## 2026-04-01 avoid 后两级课程移除 progress 硬门槛

- [x] Must: 将 `stage2 -> stage3` 与 `stage3 -> stage4` 的升级条件从 `success + collision + progress` 收成 `success + collision`
- [x] Must: 保留 `progress` 统计用于观察，但不再作为后两级课程的硬门槛

## 2026-04-01 avoid x切换惩罚与朝向修复事件惩罚继续加大

- [x] Must: 将 `gap` 内 `x` 方向切换惩罚继续加大，优先直接压低 `CmdXPredSignSwitchRateInGap`
- [x] Must: 将触发 `rotate_only` 的单次事件惩罚继续加大，让策略更明确回避需要角度恢复的状态

## 2026-04-02 avoid through阶段提前切下一行横移惩罚

- [x] Must: 将当前行 through 阶段单独收口，只要 `gap` 还没切换就不鼓励提前朝下一行做明显横移准备
- [x] Must: 对“当前行未 release 且明显朝下一行 gap 横移”的命令加入专门惩罚，优先减少 stage3/4 里擦到当前行侧面的碰撞

## 2026-04-02 avoid gap切换release改为障碍后缘+0.05

- [x] Must: 将当前行 release 判据从“后缘过行中心+固定偏移”改成“后缘过当前行障碍后缘+0.05”，避免尚未完全 through 就切到下一行
- [x] Must: 同步修改最近行选择与前向距离 helper，确保 train/play 使用同一 release 语义

## 2026-04-02 avoid 课程升级条件统一

- [x] Must: 将三段课程升级统一成只看 `success_rate + collision_rate`，不再保留任何 `progress` 硬门槛
- [x] Must: 将三段 `min_episodes` 统一为 `400`，`window` 统一为 `200`
- [x] Must: 升级阈值统一为 `1->2: success 0.70 / collision 0.20`，`2->3: success 0.90 / collision 0.10`，`3->4: success 0.85 / collision 0.15`

## 2026-04-03 avoid through惩罚改为打逆当前gap方向大横移

- [x] Must: 将 through 阶段惩罚从“朝下一行 gap 提前准备”改成“当前行未 release 时逆当前 gap 方向的大横移”
- [x] Must: 触发门槛收成 `cmd_x_toward_gap < -0.20`，优先直接打中 stage3/4 里的侧擦碰坏样本

## 2026-05-09 e_L_conflict 直角墙面实验场景

- [x] Must: 将 `e_L_conflict` 从单圆柱拐角改成受几何约束的 L 型墙面走廊，优先服务 MoE 策略直角转弯泛化演示
- [x] Must: 同步物理墙体、目标轨迹和 `gt_affordance` 障碍图，避免仿真里有墙但策略输入里看不到墙

## 2026-05-14 PCR eval 口径对齐

- [x] Must: 将 `eval_highlevel.py` 的 PCR 运行口径收回到 `s_pcr_line_avoid_basic` 训练末期与 `play_highlevel.py` 可视化口径，保留 difficulty sweep，同时避免论文指标混入冷启动状态量或不一致的 viewer/debug 规则

## 2026-05-14 PCR w 仲裁证据与机制图

- [x] Must: 在 `eval_highlevel.py` 中补出高风险区间统计和 `risk_f` 分桶数据，用于直接证明 `geom-w` 在冲突区域降低 follow 权重并改善成功/碰撞结果
- [x] Must: 新增论文级 `MoE-y` vs `MoE-y+geom-w` 机制图脚本，等两组 eval 输出完成后统一生成同图对比曲线与任务收益柱状图
- [x] Must: 修复 `eval_highlevel.py` 的 PCR success 统计口径：episode 内累计成功事件，论文主成功必须同时满足成功事件与无碰撞，并显式输出互斥 outcome 诊断项
- [x] Must: 将 `eval_highlevel.py` 收回为 `play_highlevel.py` 的统计版：复用 play 的环境创建、配置覆盖、affordance 输入和 checkpoint 加载口径，只保留 difficulty sweep、批量 episode 与 metrics 输出为 eval 特有能力

## 2026-05-18 PCR eval 逐行进度主指标

- [x] Must: 将 `eval_highlevel.py` 的论文主成功指标改为“无碰撞逐行通过进度得分”：通过 1 行记 `1/总行数`，通过多行累加；严格整回合成功仅保留为诊断字段
- [x] Must: 同步修改 PCR w 机制图脚本，图 C 不再画互斥 success/collision 堆叠，而是画逐行进度得分与碰撞率，避免误导论文结论

## 2026-05-21 PCR ckpt 参数统一

- [x] Must: 将 train/play/eval 的 PCR 主策略 checkpoint 入口统一收口为 `--pcr_ckpt`，`teacher_ckpt/ckpt` 仅保留为历史兼容入口
- [x] Must: play/eval 默认复用训练侧 `agents/low_level_best.pt` 与 `agents/avoid_best.pt`，并根据 `pcr_ckpt` 的 metadata / actor 输出维度自动识别 `w_mode`

## 2026-05-21 PCR 实机输入体检

- [x] Must: 将 `real_pcr_input_check.py` 从整框删除目标人改为 bbox 内按目标深度薄层 mask，避免把被跟随目标当作普通障碍，同时保留目标后方真实障碍
- [x] Must: 输出 `target_lost / target_too_close / depth_invalid_ratio` 等安全状态，作为 ROS1 接入前的实机输入验收字段

## 2026-05-21 learnedw2 坐标口径修复

- [x] Must: 将 `goal_buf=(x_right,y_forward)` 到世界坐标的反投影统一为项目固定 S0 口径，避免 `cmd_F / risk_F / conflict_score` 在非零 heading 时静默反向
- [x] Must: 同步修复训练与实机 `pcr_realplay.py`，并补最小坐标自检，保证 learnedw2 开训前输入语义干净

## 2026-05-21 PCR 可部署风险记忆

- [x] Must: 新增可选 `--risk_memory`，用距离衰减的 `risk_F` 记忆替换 learned-w 旧 row slot，解决当前帧看不到侧边障碍时过早恢复跟随的问题

## 2026-05-22 PCR 二维课程新场景

- [x] Must: 新增 `s_pcr_new`，完全继承 `s_pcr_line_avoid_basic` 的奖励、gap、成功判据与 PCR 口径，只把训练分布改成目标速度范围 × 障碍行数的二维课程，用于先在 play/eval 中验证旧模型泛化，再决定是否重训主表对照。
- [x] Must: 修正 `s_pcr_new` 在 play/eval 首回合的二维课程口径，确保旧模型泛化观察与 stage override 不被初始 reset 的早期课程分布污染。
- [x] Must: 补齐 `s_pcr_new` 的训练分布日志与评测记录，直接输出课程进度、level 比例、目标速度和实际行数，避免长训与论文候选评测只剩旧 stage 读数。
- [x] Must: 给 `s_pcr_new` 增加 `--generalize` 高难评测口径，固定 5 行障碍、上移目标速度到 `[0.55,0.75]`，并将纵向行距压到主场景的 `0.85` 倍，用于论文泛化结果。
- [x] Must: 保持 `row_not_released` 只用于 `w_aux` 监督与诊断，不进入 actor，确保训练输入可实机复现

## 2026-05-22 PCR w 真实冲突评测收口

- [x] Must: 将 `priv_conflict_bins` 收成障碍交互窗口内的真实冲突强度分桶，避免非冲突自由段混入论文机制图最低桶
- [x] Must: 将障碍窗口阶段占比与高冲突阶段占比分开记录，并让旧 eval 输出在新冲突图入口处直接提示重跑

## 2026-05-24 PCR 目标可观测性评测

- [x] Must: 在 PCR eval/play 中补充目标 bearing 与 RGB-FOV 可见性诊断，先判断 memory / learnedw2 是否通过牺牲目标视野换取通过率；本轮不修改训练奖励、w 公式或 actor 输入

## 2026-05-28 PCR 实机兼容启动口径

- [x] Must: 实机 PCR 闭环不再沿用默认 `manage.launch` 启动手柄节点；PCR 模式只允许一个 `/usr/command` 来源，避免 `joy_ctrl` 与 PCR 同时抢控制权。
- [x] Must: 保留 `src_real` 已验证的 `run_agent2.py -> /sita_des -> 电机` 链路，只在 PCR 侧增加相机观测发布与 `/usr/command` 兼容输出。
- [x] Must: 将 D435i 实机风险拆成 `risk_blocked_map / front_distance_risk / risk_F / risk_A`，让 PCR 接收命令条件风险，而不是直接把 `actor_difficulty` 当成 learned-w 的风险输入。
- [x] Must: `file_bridge` 模式允许继续发布 `/usr/command`，用于笔记本验证 D435i 观测和 PCR 输出；只有真实发布指令或启动 `run_agent2.py` 时才强制 ROS 环境。
- [x] Must: `src_real/interface/scripts/pcr_real` 与本地 `legged_gym/scripts` 的 PCR 实机脚本保持同步；本地启动脚本自动回到 `src_real` 完整代码目录，避免辅助文件缺失。

## 2026-05-29 PCR 五策略 baseline 主实验收口

- [x] Must: 将主实验方法收口为 `Y-only / Geom-w / Learned-w / Mono-PPO / Rule-Override`，其中前三者回答内部 `w` 消融，后两者回答外部审稿质疑。
- [x] Must: baseline 优先级固定为先做 `Rule-Override`，再做 `Mono-PPO`；不再优先扩展内部 `E2E gate` 消融，也不优先接 DWA / TEB / MPC。
- [x] Must: `Rule-Override` 采用强规则版本，风险高时增强横向避障，但保留 slow-forward 和 yaw-preserve，避免变成弱 baseline。
- [ ] Must: 先实现 `Rule-Override` eval 分支并跑 0.35 / 0.50 / 0.60 × 3 seeds，补进五策略主表和速度曲线。
- [x] Must: 实现 `Mono-PPO` 最小训练与评测分支；它只允许使用部署观测并直接输出 `[x_right, y_forward, yaw]`，不得调用 Follow/Avoid expert、`resolve_moe_gate_pcr` 或读取 `cmd_F/cmd_A/risk_F/risk_A/y/w`。
- [x] Must: 修正 `Mono-PPO` 评测主表口径：主性能表保留 `Unsafe Rate / C_avoid Rate`，仅机制项 `CSI@C_avoid` 记为 N/A；冲突诊断只用于 eval 统计，不进入 Mono-PPO 策略输入。

## 2026-05-29 PCR 实机 ROS 风险输入闭环

- [x] ~~Must: 将 `real_pcr_input_check.py --publish_ros` 发布的实时输入扩展到 `risk_blocked_map / policy_visible_map / front_distance_risk`，让 ROS 链路与已验证的 `obs_file` 口径一致。~~
- [x] ~~Must: 将 `pcr_realplay.py` 订阅并使用上述实时风险输入；没有风险图时才退回 `local_map_2ch` fallback。~~
- [x] ~~Must: 保持本地 `legged_gym/scripts` 与 `src_real/interface/scripts/pcr_real` 副本同步，并完成语法检查与同步检查。~~

## 2026-06-03 PCR 实机低层频率对齐

- [x] Must: 将 PCR 接入改为 `pcr_realplay` 10Hz 缓存速度、`run_agent2.py` 50Hz 均匀执行低层策略；PCR 路径先禁用 `/sita_des` 三连发，后续实机验证确认手柄三连发会形成突发串口负载，因此手柄与 PCR 当前均为单次均匀发布，抢占语义不变。

## 2026-06-09 PCR 实机目标/障碍坐标语义对齐

- [x] Must: 将实机目标前向 offset 与障碍地图前向 offset 拆开，目标状态对齐 robot base，局部障碍图对齐 camera mount，避免障碍距离被错误前移导致避障触发偏晚。

## 2026-06-12 Fig.6 分层失败轨迹选样

- [x] Must: Fig.6 候选按重置前真实终止位置标注障碍行与进度，默认优先选择“第 3 排碰撞 + 第 4 排碰撞 + 早期掉队 + 晚期掉队 + Learned-w 成功”的五策略组合，避免失败轨迹集中在同一位置。

## 2026-06-12 Fig.3(b) 候选图对比

- [x] Must: 基于 Table I 的 0.60 m/s 五方法三 seed 聚合数据，采用 Collision-Follow MAE 点大小编码散点图作为最终 Fig.3(b)；点面积随 Task Success 单调增大，并补偿星形 marker 的视觉面积，不覆盖现有 Fig.3。

## 2026-06-12 论文附录与最终产物口径修正

- [x] Must: A1/A2/A4 改为读取最终 checkpoint 的实际训练配置与网络维度，禁止用脚本默认值冒充论文复现参数。
- [x] Must: A5 的 Learned-w 必须完整报告 All/C_unsafe/C_avoid 下的 Delta y_w、Delta y_r、Delta y_total；旧数据缺字段时明确要求补跑机制诊断，不允许从总量倒推。
- [x] Must: 修正 Table II 小量级显示、Table III TTC/Mono 容量说明、A3 传感设置和 Fig.3(b) 点面积措辞。
- [x] Must: Fig.4/Fig.6 同时生成 PNG 与 PDF，并让 MANIFEST 逐项核验实际文件存在性；本地缺 Fig.6 原始 timeseries 时保留已审定 PNG 并显式标记 raster-wrapped PDF，服务器端有源数据时重绘矢量 PDF。

## 2026-06-14 论文神经网络架构图

- [x] Must: 按正式 checkpoint 与当前代码生成独立 SVG 网络图，覆盖 Learned-w Gate、基础 Gate 变体、Avoid Expert、Mono-PPO、固定底层 locomotion 和可选 Affordance Estimator；解析式模块不画成神经网络。
- [x] Must: 将初版模块流程框图改为论文常用的神经元层级与 CNN 特征图示意图，仅保留真实网络层、分支和训练 critic。
- [x] Must: 额外生成用于总流程图的紧凑型透明 SVG 图标，分别概括 Affordance Map、Learned-w Gate、Avoid Expert 和固定 Locomotion 的输入、主要中间层与输出。
- [x] Must: 总流程图小图标改用 TikZ 标准神经元样式，仅显示带真实参数的输入层与输出层；隐藏内部层，保证缩小时仍可辨认。

## 2026-06-15 PCR-Net 整体训练与部署架构图

- [x] Must: 按当前训练和实机代码绘制左右分栏的论文级整体架构图，准确展示专家预训练、PCR Gate 训练、D435i 实机感知、PCR-Net 仲裁、固定底层策略与电机反馈闭环；不纳入手柄控制或可选 Affordance Estimator。

## 2026-06-15 论文视频仿真可视化

- [x] Must: 为 `play_highlevel.py` 增加独立论文视频模式，同步展示虚拟深度相机、actor 实际 `local_map_2ch`、3D 相机视锥、perception 红点与机身坐标系执行速度箭头；只改变 viewer 显示，不改变训练、评测、观测或控制口径。
- [x] Must: 为 Fig.6 五种方法增加 Stage 4 场景复现入口，按已审定 episode 恢复确定性障碍布局，保留高层策略与底层六足的真实在线行为，用于论文视频录制。
- [x] Must: 修正 Fig.6 回放中全局 `episode_id` 与环境内部场景序号混用的问题，改为从最终 source 清单和原始 timeseries 精确注入障碍物与目标初始状态，并在运行前校验 LayoutID。

## 2026-06-15 PCR 实机安全默认值与反馈保护

- [x] Must: 将 `src_real` 实机 PCR 默认相机安装参数收口为 `camera_height_m=0.23`、`camera_pitch_down_deg=0`，避免每次实机命令都依赖额外参数覆盖。
- [x] Must: 恢复 CAN 侧 `sita_des` 进入电机前的关节范围检查，按 `LB/LF/LM` 与 `RB/RF/RM` 的 URDF 真实关节范围限幅；畸形或非有限命令触发共享故障锁存。
- [x] Must: `run_agent2.py` 在左右电机各返回至少一帧有效反馈前禁止进入运动策略；不使用时间超时门控，避免事件触发反馈与命令准入互相等待。

## 2026-06-17 Table II 机制诊断口径收紧

- [x] Must: Table II 从性能/容量混合口径收口为机制诊断口径，保留 `C_avoid Rate` 与 `CSI@C_avoid`，并将 `Delta y_total@C_unsafe` / `Params` 替换为 `Delta y_w@C_avoid` 与 `Delta y_r@C_avoid`，直接支撑 learned-w channel 在冲突窗口内的贡献解释。

## 2026-06-17 DWA-style 外部诊断表

- [x] Must: 将 Target-aware Velocity-Space Search 只作为外部诊断替代路线，不进入主表；最终输出读取 bounded validation 的 safe / balanced / tracking preset，报告 safety-following trade-off，而不是包装成强主表 baseline。

## 2026-06-17 Velocity-Search dynamic-window 诊断增强

- [x] Must: 为 Target-aware Velocity-Space Search 增加可选 dynamic-window candidate filter，只约束 velocity-search 候选速度可达性，不影响 Learned-w / Additive-Fusion / Rule-Override；记录过滤前后候选数量和 rejected count，用于判断候选可达性是否导致 feasible_count 塌陷。

## 2026-06-17 DWA-style 最终诊断表切换到 dynamic-window 版本

- [x] Must: 最终 `tableA_dwa_velocity_search_diagnostic` 不再读取旧 Velocity-Search preset 结果，而是读取启用 dynamic-window candidate filter 的 Safe / Balanced / Tracking × 0.35 / 0.60 结果；若任一行不是 dynamic-window eval，生成表格时直接报错，避免旧结果混入论文最终表。

## 2026-06-17 PCR 实机 CAN 控制层回到原始逻辑

- [x] Must: 按 `/home/hxy/src` 原始正常手柄控制程序恢复 `can_control/scripts/node.py` 的事件触发 CAN 执行逻辑，撤销 peer ready / stale feedback / shared serial fault 等新增锁死条件；PCR 与手柄上层话题拆分保留在 `run_agent2.py`，避免继续干扰原本能跑的电机通信节奏。

## 2026-06-17 PCR 实机手柄发布节奏对齐原版

- [x] Must: `run_agent2.py` 中手柄/manual 与 legacy 命令恢复原始 `/sita_des` 三连发节奏，保留 PCR 50Hz 路径单次发布，避免 PCR 插入改变原手柄连续推杆时的低层命令节奏。

## 2026-06-17 PCR 实机默认速度口径对齐

- [x] Must: 将 PCR 实机默认命令上限调到与 `run_agent2.py` 的速度解释一致，使默认最终低层命令可达约 `x=0.4`、`y=0.8`，同时保留命令行覆盖能力。

## 2026-06-17 Held-out irregular-row eval-only 补救实验

- [x] Must: 增加 `heldout_irregular_rows` eval-only 布局，固定 Stage 4、0.60 m/s、非镜像不规则障碍行，允许有限尺寸的混合障碍形状；只用于最终评测，不进入训练、validation tuning 或主表调参。
- [x] Must: 输出 held-out 表格与真实仿真布局导出文件，保证表格指标、障碍位置和可视化均来自同一次 eval 环境状态。
- [x] Must: 将 `heldout_irregular_rows` 从高难 stress test 收口为温和 OOD：保持或略放松 Stage 4 行距，主要改变障碍排布与形状类型，并以 box 为主，避免第一行过窄导致泛化结论被难度混淆。
