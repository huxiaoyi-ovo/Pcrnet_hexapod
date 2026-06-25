# Context Handoff Summary

## 1. 项目目标（Goal）

- 当前主线是 PCR-Net 论文收尾：证明六足机器人在杂乱行障碍中进行目标跟随时，command-space conflict arbitration 能比固定规则、加法融合、单体策略和 DWA-style 速度搜索更好地平衡跟随与避障。
- 投稿目标是 IEEE RA-L / ICRA 格式的 8 页正文，当前论文工作目录为 `/home/hxy/文档/pcr论文`，主文件是 `root.tex`，当前编译产物是 `root.pdf`。
- 当前论文题目已收口为：`PCR-Net: Command-Space Conflict Arbitration for Hexapod Target Following in Clutter`。

## 2. 背景与范围（Context & Scope）

- In scope:
  - 论文正文、图表、caption、表格说明、claim 收缩和版面排布。
  - PCR-Net 主表、机制表、Mono-PPO 表、DWA-style 表、实机 40 次 trial 表。
  - `learned-w` 机制叙事：不是一味压 Follow，而是在冲突中保留可恢复的 Follow/yaw 支持，同时通过 lateral Avoid 注入避障动作。
  - 仿真主结果：训练目标速度到 `0.50 m/s`，`0.60 m/s` 是速度外推 eval，不要写成训练到 `0.65 m/s`。
  - 实机证据：40 次人工标注 trial，正文只做谨慎硬件证据，不夸大为完整真实场景泛化。
- Out of scope:
  - 不再扩展大系统、新训练主线、动态障碍实验、大规模 sensor-noise matrix。
  - DWA-style baseline 不再继续大量调参；它作为 interface-matched、validation-selected 的 planner-style external alternative 呈现。
  - Held-out irregular-row 当前结果更像 stress test，不作为主文强泛化核心证据。

## 3. 关键约束（Constraints）

- 技术/环境：
  - 代码仓库：`/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym`。
  - 论文目录：`/home/hxy/文档/pcr论文`。
  - LaTeX 主文件：`root.tex`；编译命令用 `pdflatex` 两遍。
  - 当前 PDF：`root.pdf`，最近检查为 8 页、letter 页面、无 LaTeX error / overfull / undefined 引用。
- 论文排版：
  - 不要擅自替换用户已定的 PDF 图。
  - 所有论文配图以 PDF 文件为准，不看 JPG/MP4。
  - Fig.5 必须用裁剪后的 `figs/fig5_sim_overview_crop.pdf`，单栏宽度 `\columnwidth`。
  - Fig.6 是三联实机仲裁曲线，必须保持 `figs/fig7_arbitration.pdf`，不要换成 horizontal 版本。
  - Fig.7 是实机帧图，由 `figs/fig4a_real_follow.pdf` 和 `figs/fig4b_real_onboard.pdf` 组成双栏图。
- 论文口径：
  - 不主动暴露“baseline 没调好 / diagnostic not fully tuned”这类软肋。
  - 说法应是：外部基线按相同高层命令接口、相同局部观测和 validation-selected 配置做了公平适配。
  - limitation 写得克制，承认范围边界但不把缺点展开成审稿人攻击清单。

## 4. 已确认的关键决策（Decisions）

- D1: PCR 融合公式与符号口径已锁定。
  - `u = y_eff u_F + (1-y_eff) u_A`。
  - `y_eff = clip(y + Δy_r^dir + Δy_w, 0, 1)`。
  - `Δy_r^dir = γ(ρ_A^dir - ρ_F^dir)`，当前 `γ=0.15`。
  - `Δy_w = λ deadband(2w-1)`，当前 `λ=0.30`，deadband margin `0.05`。
  - `w>0.5` 表示支持 Follow，`w<0.5` 表示支持 Avoid；aux-BCE 是弱先验，不主导最终 PPO 行为。

- D2: Table II 的核心矛盾已经通过符号系统修复。
  - `C_avoid^eval` 不是旧的 directional-risk 代数窗口。
  - 新定义是 eval-only rollout/utility 诊断窗口：
    ```text
    C_avoid^eval = { C_unsafe^roll and U_A^roll > U_S^roll + epsilon }
    ```
  - `ρ_F^dir / ρ_A^dir / Δy_r^dir` 是 gate 内部 directional risk correction。
  - 因此 `Δy_r^dir@C_avoid^eval ≈ 0` 不再矛盾；它说明 directional risk difference 在 rollout-identified conflict window 中不主导，learned-w 提供主要调节。
  - Caption 中必须保留意思：`C_avoid^eval` comes from short-horizon rollout/utility diagnostics and is not the algebraic directional-risk condition used by the gate。

- D3: Learned-w 叙事已从“单纯避障压 Follow”改成“冲突中保留进度”。
  - 仿真 Table II 和 real Fig.6/7 的共同叙事是：learned-w 不只是让机器人更保守，而是在风险响应下恢复一部分 Follow/yaw 支持，使 lateral Avoid 与 forward progress 能同时存在。
  - 不要写 learned-w “proves planners cannot solve the task” 或 “purely suppresses Follow”。

- D4: DWA-style baseline 口径已收口。
  - 论文中称为 `DWA-style target-aware velocity-space search` 或 `Target-aware Velocity-Space Search`。
  - 不宣称完整 ROS DWA。
  - 正文需要说明它：使用同样 local map 和 relative target state，采样高层速度命令，dynamic-window 过滤，短时 rollout，footprint collision checking，validation-selected cost weights。
  - 不要用“diagnostic weak baseline”这类词；表述为 bounded validation reveals a fixed-cost safety-following trade-off。

- D5: Mono-PPO 的位置固定。
  - Mono-PPO 不放进 Fig.3 主仲裁图。
  - Mono-PPO 是外部架构 baseline / Table III，用来说明去掉专家分解和 gate 后，同样观测/动作接口下单体策略不稳定。
  - 不要声称“端到端 RL 一般不行”，只说在本 benchmark、训练预算和接口下结构化分解提供了有用归纳偏置。

- D6: 当前论文图表编号和文件绑定必须保持。
  - Fig.1：`figs/fig1_real_scene.pdf`，实机场景 teaser。
  - Fig.2：`figs/fig1_system.pdf`，系统结构图。
  - Fig.3：`figs/fig2_main_singlecol_compact.pdf`，主性能图，单栏。
  - Fig.4：`figs/fig6_traj.pdf`，0.60 m/s 五策略轨迹图，当前是横向单栏版本。
  - Fig.5：`figs/fig5_sim_overview_crop.pdf`，Simulation view of hardest stage，单栏满宽。
  - Fig.6：`figs/fig7_arbitration.pdf`，三联实机机制曲线，单栏。
  - Fig.7：`figs/fig4a_real_follow.pdf` + `figs/fig4b_real_onboard.pdf`，实机帧图，双栏。

- D7: 实机结果口径固定为 40 trials。
  - Table V 当前为 two scenes：`Staggered rows` 和 `Sharp target turn`。
  - 总体成功率写成 36/40 或 90%，失败包括 obstacle contact / target lost，人工标注为准。
  - 正文可以说 provides quantitative hardware evidence for deployed perception-arbitration-locomotion chain，不要夸成 full real-world robustness。

## 5. 当前代码与结构（Code Map）

- 论文目录：
  - `/home/hxy/文档/pcr论文/root.tex`：当前 IEEE 正文主文件。
  - `/home/hxy/文档/pcr论文/root.pdf`：当前编译后 PDF。
  - `/home/hxy/文档/pcr论文/figs/`：论文所有 PDF 图。
  - `/home/hxy/文档/pcr论文/make_fig3_compact.py`：Fig.3 单栏紧凑版生成脚本。
- 关键 repo 文件：
  - `agents/final_paper_outputs_v3/`：最终图表和 CSV 来源。
  - `legged_gym/scripts/build_final_paper_outputs.py`：论文图表生成主入口。
  - `legged_gym/scripts/eval_highlevel.py`：高层评测、timeseries、DWA/Additive/机制统计相关。
  - `legged_gym/scripts/play_highlevel.py`：可视化 play、debug camera、DWA/各 baseline 观察入口。
  - `src_real/interface/src/joy_ctrl.cpp`：实机手柄速度比例已按 PCR 接管后的低速控制方向改过，编译后必须重启相关 ROS 节点才生效。

## 6. 当前进度（Status）

- 已完成：
  - `root.tex` 已适配 IEEE `ieeeconf` 根模板。
  - 当前 `root.pdf` 编译为 8 页。
  - Fig.5 白边问题已处理：生成并引用 `fig5_sim_overview_crop.pdf`。
  - Fig.6 已恢复为原三联机制曲线 `fig7_arbitration.pdf`，不要再误换。
  - Fig.7 实机帧图已作为双栏图放入正文后半部分。
  - Table II 的 `C_avoid^eval / Δy_r^dir` 逻辑矛盾已修正。
  - DWA-style、Mono-PPO、Additive-Fusion、Rule-Override 的论文定位已基本统一。
  - 实机 40-trial 表已经进入正文。
- 进行中：
  - 继续细调版面：Fig.3(b) 标签位置、表格表头简洁度、caption 压缩。
  - 最后审查正文是否还有过度 claim 或主动暴露短板的表述。
- 未开始 / 暂缓：
  - 不再优先做新的 DWA 大调参。
  - 不再优先做动态障碍和大规模 sim2real 噪声矩阵。
  - 不把 heldout irregular-row 作为主文强泛化证据。

## 7. 已知问题与风险（Known Issues & Risks）

- Fig.5 原始 PDF 自带白边，必须继续使用 crop 版本；如果换回 `fig5_sim_overview.pdf`，看起来会再次“不撑满单栏”。
- `fig7_arbitration_horizontal.pdf` 是之前临时生成的横版曲线，不应替换当前 Fig.6，除非用户明确要求。
- DWA-style 如果写成“未充分调优 / diagnostic weak baseline”，会给审稿人递刀；应写成已按接口公平适配并在 validation 上选择配置。
- Table II 不能再写旧定义 `C_avoid = ρ_F>0.45 and ρ_F-ρ_A>0.15`，否则 `Δy_r≈0` 会被代数打穿。
- 论文中 `0.60 m/s` 必须写成 speed-extrapolation eval beyond training range；训练速度不要写到 `0.65 m/s`。
- 实机图和机制图支持“发生了有效仲裁”，但不要声称完全解决真实环境中目标-障碍遮挡、深度抖动、动态行人等问题。

## 8. 待办清单（Next Actions / TODO）

- [P0] 打开 `/home/hxy/文档/pcr论文/root.pdf`，逐页检查图号、caption、表格位置和跨栏图顺序。
- [P0] 检查正文所有 `C_avoid / ρ_F / ρ_A / Δy_r` 是否都按最新 `dir/eval` 符号写清。
- [P0] 检查 DWA-style 段落，删除 `diagnostic / not fully tuned / weak baseline` 等容易被抓的词。
- [P0] 检查 Fig.5、Fig.6、Fig.7 的文件引用，禁止再次混淆。
- [P1] 微调 Fig.3(b) 点标签，让 Risk-only / Geom-w 文字更贴近各自图标且不重叠。
- [P1] 压缩各表格表头，把解释性文字放到正文，不放在表格区域。
- [P1] 最后跑两遍 `pdflatex`，确认 8 页、无 error、无 overfull、无 undefined refs。

## 9. 需要我在新会话首先回答的三个问题（Top Questions）

1. 当前 `root.pdf` 逐页看下来，Fig.5/6/7 的顺序和大小是否已经满意？
2. 是否需要我继续直接修改 `root.tex`，把 DWA/Mono/limitation 的投稿语言再收紧一轮？
3. 是否要在最终提交前生成一份“审稿人可能攻击点—正文防守句”清单？
