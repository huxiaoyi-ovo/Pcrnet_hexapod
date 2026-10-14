# 实验与部署口径

训练、评测、场景、传感、观测和部署输入相关任务按需读取。本指南保存当前冻结口径及其历史适用范围，不代表这些设置都已在当前代码或硬件上验收。执行前查看当前源码、配置、manifest、checkpoint 与有日期记录；V7 是阶段性背景，后续适用范围明确的实验结论优先。S1–S6/T1/T2、S6 结构化 OOD hold-out 和鲁棒性矩阵是 V7 长期目标，不等于当前返修必须新增整套实验。

## 可复现与场景

固定 seeds、评测设置、episode 数和训练预算，保存关键配置与关键结果；公平比较方法和预算。将方案要求落实为可检查的场景和任务条件，确认任务真实发生后再扩大训练。避免依赖仿真特有捷径；吞吐、物理、传感质量按实际证据评价。

项目当前场景命名为 `s_`（训练场景）和 `e_`（实验场景）。其他旧名仅用于追溯其与当前场景的关系；没有证据时不创建映射，也不把旧名作为新实验命令或结论依据。

PCR 与 avoid expert 并行训练时默认 `PCR -> GPU 0`、`avoid expert -> GPU 1`，用户另行指定时按其要求。并行任务使用不同日志目录、checkpoint 名和实验名，不写入同一路径。

## PCR revise 冻结决策与版本识别

2026-09-10 为本项目 revise 阶段冻结的决策：一次性综合选择性能、正确性、仿真—实机一致性和说服力；仅对影响核心架构、训练成立或论文结论的高风险不确定性专项验证。低风险默认不新增小测试或横向扩展；必要正确性检查和证据真实性约束仍有效。未知的相机 origin/offset 等参数不得伪造，也不因此新增测试要求。

同日冻结的地图语义为：sim、PCR、real 使用同语义的即时 canonical local map；关闭 real-only obstacle-map memory；保留 PCR learned-w risk memory；不为 memory 新增 dropout 分支验证。后续 TODO 记载 C1/D2 已完成代码接入，但这不自动证明 checkpoint、Isaac、ROS state provider 或硬件已验收。实现、训练和验收状态须查当前代码、关键结果与有日期里程碑；不把 CPU 结果写成实机验证。

2026-05-19 `learned-w` 方向属于当时阶段记录：单向压 follow、`row_not_released` 输入与 row-aware `w_aux`；该阶段不默认切换到双向 follow-support、learned beta 或大范围奖励改动。它是有日期的阶段限制，不自动成为后续版本的现行指令。后续候选可能改变公式，旧版本标识为 `learnedw`，后续候选公式标识为 `learnedw2`；不得合并二者，或仅凭版本名称推断 checkpoint 采用的公式与实际权重。开始具体训练或评测前，按冻结协议、公式名和 checkpoint 确认版本。具体计划入口：[PCR W 主线计划](../specs/PCR_W_MAINLINE_PLAN_CN.md)。本指南不裁决当前训练方向。

## Follow / avoid 输入与职责

- 优先稳定 avoid expert，再做 follow/avoid 融合前的输入统一。解析式 follow expert 可单独使用仿真真值目标信息验证；融合训练或 sim2real 对齐前，专家与仲裁层都必须只依赖部署可得信息。
- avoid 的 `goal_buf` 是局部子目标/waypoint，不要求是人的真值位置；融合前 follow 侧改用部署可得的目标相对状态包。
- avoid 默认观测为 `state + goal_buf + local_map_2ch`，地图通道为 `occupancy` 与 `clearance/cost`；默认不要求跨越信息，`low_obstacle` 不是必需输入。
- avoid 是局部短时安全通过专家，不是全局规划器。目标是在安全前提下持续朝局部子目标前进，主要用横移和小幅姿态调整通过，避免后退、久停、碰撞、贴边硬挤和卡住。不要把固定中缝、少数模板路线或长期“往中间走”训练成目标，也不要替代全局规划或最终仲裁。
- 仿真 `teacher` 真值图是训练参考，不默认作为部署输入；`student` 可选。避障 sim2real 默认链路为 `D435i depth -> local_map_2ch -> avoid expert`。

## S0 方向约定与最小排查

坐标约定必须保持如下原义：

- `heading=0` 时，机身前向对齐 world `+Y`，机身右向对齐 world `+X`。
- `goal_buf=(x_right, y_forward)`，`bearing=atan2(x_right, y_forward)`。
- 世界到机身坐标使用 `R(-heading)`：`x_right = cos(h)*dx + sin(h)*dy`；`y_forward = -sin(h)*dx + cos(h)*dy`。
- `+omega = 左转(CCW)`；按以上 bearing 定义，朝目标转向须满足 `omega = -k_yaw * bearing`。
- S0 expert 只允许 `cmd=[0, cmd_y, omega]`，禁止侧向平移。

最小验收先看方向一致性，不先看总 reward：`pre_alpha * pre_omega < 0`；`step_yaw` 与 `cmd_omega` 同号；`pre_goal_err` 约等于 0。固定观察起步 `step 0~80`。反向问题依次：1. 用 `--force_cmd` 核实底层 `+omega` 方向；2. 对比 `goal_buf` 与实时几何投影的 `pre_goal_err`；3. 核对 expert 的 `pre_alpha` 与 `pre_omega` 收敛关系；4. 再看轨迹与 `dist`。

## D435i 视觉设定

以下数值是既定项目参数口径的迁移记录，本次未作外部复核；具体任务使用前仍核对 Intel/RealSense 官方资料与设备支持模式。

- 默认深度相机：Intel RealSense D435i。depth FOV `87°(H) x 58°(V)`，对角约 `95°`；RGB FOV `69.4°(H) x 42.5°(V)`，对角约 `77°`。
- 常用 depth 模式 `1280x720@30`；高速模式 `848x480@90`；RGB 常用模式 `1920x1080@30`。
- 默认深度工作范围 `0.3 ~ 3.0 m`；HD 深度最小距离参考约 `0.28 m`。
- 不混用设备不支持的分辨率/帧率组合，例如不默认写 `1280x720@90`。没有明确高速对照时用 `1280x720@30` 或常见起步值 `848x480@30`；用户明确要求高帧率对照时才考虑 `848x480@90`。
- 仿真中的 FOV、near/far、分辨率、采集率、可见范围与 local-map 裁剪默认向设备口径收敛；没有传感器对照需求时，不扩 FOV、工作距离或可见范围来换训练收益。
- 使用 teacher GT 时明确其为仿真参考，不把超出 D435i 可见范围的几何当作部署可得信息。视觉相关训练/评测记录须列明 FOV、near/far、输入分辨率、`local_map_2ch` 尺寸及是否使用 GT 参考图。未知的相机 origin/offset 不得伪造，也不因此新增测试要求。

## 项目长期实验目标

项目目标是在单卡 3090 与目标 2048/4096 并行约束下，获得可复现且能支持 RAL 结论的高质量策略结果。场景任务、物理接触和传感口径须可量化、可复核，避免穿模、碰撞几何错误、绕行漏洞和奖励投机。底层 locomotion 不使用深度相机，按方案用特权高度图/采样；高层使用 affordance/目标状态输出 `cmd_vel` 与门控 `y`，训练与部署保持相同命令后处理语义。优先选择吞吐友好的场景表达，昂贵仿真要素需服务明确论文结论。

长期证据目标包括公平预算与范式级基线、关键结果的均值/方差或置信区间、S6 结构化 OOD hold-out，以及传感退化和动力学扰动下的鲁棒性曲线。它们是项目整体目标，不自动构成本轮返修或每次局部实验的新增要求；由当前论文问题和已冻结实验协议确定必要工作量。
