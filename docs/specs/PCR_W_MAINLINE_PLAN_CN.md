# PCR w 主线近期任务计划

## 1. 当前结论

当前 PCR 主线的核心问题不是继续扩展系统，而是把 `w` 这个贡献点打穿：

```text
MoE-y 只学习原始 follow 权重 y；
MoE-y+w 在 y 的基础上加入 command-conditioned conflict prior w；
w 在 follow 命令方向危险时降低实际 follow 权重 y_eff；
最终目标是减少危险跟随，同时尽量保持通过率和成功跟随距离。
```

当前代码中的 `w` 是 `geom w`，不是学习出来的网络头：

```text
w_geom = exp(-clearance_F / w_tau)
y_eff = y_raw * (1 - w_geom)
cmd = y_eff * cmd_F + (1 - y_eff) * cmd_A
```

其中 `clearance_F` 是在 affordance map 上沿 `cmd_F` 方向、以约 `25°` cone 统计得到的最近障碍距离。

## 2. 当前 geom w 版本

### 2.1 实现位置

主函数：

```text
legged_gym/scripts/train_highlevel.py
resolve_moe_gate_pcr()
```

训练、评测、演示共用同一逻辑：

```text
train_highlevel.py
eval_highlevel.py
play_highlevel.py
```

### 2.2 关键参数

```text
w_mode = geom
w_tau = 0.15
w_blend_mode = multiply
w_disable_gate_safe_clamp = True
```

### 2.3 当前语义

```text
y_raw：gate 网络输出的原始 follow 权重
w_geom：follow 命令方向风险先验
y_eff：实际执行 follow 权重
cmd_F：follow expert 候选命令
cmd_A：avoid expert 候选命令，PCR 中主要保留横移
```

`w_geom` 的作用路径固定为：

```text
clearance_F 变小
-> w_geom 变大
-> y_eff 变小
-> follow 权重下降
-> avoid 权重上升
-> dangerous follow 减少
```

### 2.4 论文表述边界

当前不能写成：

```text
the network learns w
```

应写成：

```text
we augment a learned MoE gate with a command-conditioned geometric conflict prior w
```

或者：

```text
MoE-y learns the nominal arbitration weight, while w provides a lightweight command-conditioned conflict prior computed from candidate expert commands.
```

## 3. geom w 的论文价值与风险

### 3.1 价值

`geom w` 虽然公式简单，但不是普通最近障碍距离。它的价值在于：

```text
它沿候选 follow 命令方向计算风险；
它直接修正 learned gate；
它面向 follow-vs-avoid 冲突，而不是普通避障；
它能提供清晰机制图：risk_F 上升 -> w 上升 -> y_eff 下降 -> near-miss / collision 下降。
```

### 3.2 风险

`geom w` 容易过保守：

```text
只看 risk_F，不看 risk_A；
follow 方向有风险时总是压 follow；
如果目标已经快跟丢，它仍可能继续压 follow；
如果 avoid 也危险，它仍可能错误地偏向 avoid。
```

因此主实验必须同时报告：

```text
CollisionRate
NearMissRate
FollowLostRate
FollowDistMean
SuccessfulFollowDistMean
PCR Success
RobotCrossedRate
```

不能只报告 collision 下降。

## 4. learned w 最终版本

### 4.1 目标

最终版本的 `w` 应从解析几何 prior 升级为网络预测的 conflict prior：

```text
w_pred = f_theta(state, affordance_map, cmd_F, cmd_A, risk_F, risk_A, follow_dist, y_raw)
```

理想语义：

```text
w_pred 表示 follow proposal 相对 avoid proposal 是否更危险。
```

它不应退化成普通 clearance，也不应只是保守开关。

### 4.2 最小网络改法

当前 gate 输出 1 维：

```text
y_raw
```

learned w 版本改为输出 2 维：

```text
y_raw, w_pred
```

建议保持执行逻辑不变：

```text
y_eff = y_raw * (1 - w_pred)
cmd = y_eff * cmd_F + (1 - y_eff) * cmd_A
```

这样 geom w 与 learned w 的比较最干净。

### 4.3 risk supervision

`risk supervision` 不是当前已有参数，而是 learned w 的辅助监督。

核心想法：

```text
网络预测 w_pred；
代码根据候选命令方向风险构造 w_label；
训练时加入 loss_w，让 w_pred 学习 command-conditioned conflict。
```

推荐标签：

```text
w_label = clamp(risk_F - risk_A, 0, 1)
```

更平滑版本：

```text
w_label = sigmoid(k * (risk_F - risk_A - margin))
```

含义：

```text
只有 follow 比 avoid 更危险时，w 才应该明显变大。
```

可能新增参数：

```text
learn_w = True
w_aux_loss_coef = 0.1
w_label_mode = risk_delta
w_label_margin = 0.1
w_label_temperature = 5.0
```

这些参数目前还未实现。

## 5. 实验阶段安排

### Stage 1：geom w 主结果收口

先使用当前已修正 final line 的代码，重训并对比：

```text
MoE-y
MoE-y+geom-w
```

固定参数：

```text
w_mode = geom
w_tau = 0.15
w_blend_mode = multiply
w_disable_gate_safe_clamp = True
target_speed = 0.35 m/s
follow_distance_desired = 1.5 m
follow_distance_min/max = 1.0 / 2.2 m
```

目标：

```text
确认 geom w 是否能在真实 final line 口径下显著降低 CollisionRate / NearMissRate；
确认 PCR Success / RobotCrossedRate 是否接近或优于 MoE-y；
用 SuccessfulFollowDistMean 判断成功回合是否真的保持合理距离。
```

如果 geom w 已经满足主结论，就先写 geom w 版本，不急着实现 learned w。

### Stage 2：跟随权重再平衡

仅当 geom w 过保守时执行。

触发条件：

```text
CollisionRate / NearMissRate 明显优于 MoE-y；
但 FollowLostRate 明显偏高；
SuccessfulFollowDistMean 也偏大；
PCR Success / RobotCrossedRate 被跟随拖住。
```

优先动作：

```text
调大已有 follow band / follow quality 相关奖励；
不新增 recovery reward；
不继续扫 w_tau。
```

目标：

```text
保留 geom w 的安全收益；
补回低风险区域的跟随推进能力。
```

### Stage 3：learned w 升级

仅当下面任一条件满足时执行：

```text
geom w 结果只能证明“更保守更安全”，无法支撑主贡献；
论文叙事需要更强的学习型创新；
geom w 机制图清楚，但 FollowLostRate 始终压不下来。
```

最小实现目标：

```text
GatePolicy 输出 y_raw 和 w_pred；
用 risk_delta 构造 w_label；
加入 w_aux_loss；
训练 MoE-y+learned-w；
与 MoE-y、MoE-y+geom-w 对比。
```

## 6. 论文图表规划

### Table 1：主性能对比

行：

```text
Follow-only
Avoid-only
MoE-y
MoE-y+geom-w
MoE-y+learned-w（若实现）
```

列：

```text
PCR Success
RobotCrossedRate
CollisionRate
NearMissRate
FollowLostRate
FollowDistMean
SuccessfulFollowDistMean
PCRGapSuccessPerEpisode
```

### Table 2：w 消融

最小版本：

```text
MoE-y
MoE-y+geom-w
MoE-y+learned-w
```

可选附录：

```text
geom-w tau=0.25
geom-w tau=0.15
geom-w tau=0.10
```

但主文不建议把扫 `w_tau` 作为核心结果。

### Figure 1：PCR 场景示意

必须标明：

```text
moving target line x = -0.60 m
target speed = 0.35 m/s
follow band = 1.0~2.2 m
actual scaled final row
障碍行交错开口
Follow vs Avoid conflict region
```

### Figure 2：w 机制时间序列

画一段代表性 episode：

```text
risk_F
risk_A
w 或 w_geom / w_pred
y_raw
y_eff
follow_dist
clearance / near-miss event
```

要证明：

```text
冲突区域 risk_F 上升；
w 上升；
y_eff 相比 y_raw 下降；
危险贴障减少；
离开冲突区后跟随恢复。
```

### Figure 3：安全-跟随权衡图

横轴：

```text
FollowLostRate 或 SuccessfulFollowDistMean
```

纵轴：

```text
CollisionRate 或 NearMissRate
```

点：

```text
MoE-y
MoE-y+geom-w
MoE-y+learned-w
Rule-Mix（若有）
```

理想结论：

```text
MoE-y+w 在安全-跟随权衡上更靠近 Pareto 优势区域。
```

### Figure 4：轨迹对比

对比：

```text
MoE-y：冲突区危险跟随 / late switch / near-miss
MoE-y+geom-w：提前降低 y_eff，绕开障碍
MoE-y+learned-w：若实现，应显示更少过保守、更快恢复跟随
```

## 7. 当前近期执行顺序

当前最小主线：

```text
1. 使用 final line 修正后的代码重训 MoE-y；
2. 使用相同代码重训 MoE-y+geom-w；
3. 用 eval 输出 Table 1 所需指标；
4. 导出 Figure 2 所需机制时间序列；
5. 判断 geom w 是否足够支撑论文；
6. 若 geom w 过保守，再做 follow 奖励再平衡；
7. 若仍不够，再实现 learned w。
```

当前不优先：

```text
不把 beta 放进主训练贡献；
不继续扫 w_tau；
不先做大规模随机化；
不先追 1 m/s 目标速度；
不把 learned w 直接插到未收干净的基线前面。
```

## 8. 成功标准

geom w 版本够用的最低标准：

```text
CollisionRate 明显低于 MoE-y；
NearMissRate 明显低于 MoE-y；
PCR Success 不明显低于 MoE-y，最好接近或更高；
SuccessfulFollowDistMean 落在合理跟随范围附近；
Figure 2 能清楚显示 risk_F -> w -> y_eff 的机制链。
```

learned w 版本值得成为最终方法的标准：

```text
安全指标接近 geom w；
FollowLostRate / SuccessfulFollowDistMean 接近或优于 MoE-y；
PCR Success 高于 MoE-y 和 geom w；
w_pred 不退化成 clearance，而是和 risk_F-risk_A / 冲突事件更一致。
```

