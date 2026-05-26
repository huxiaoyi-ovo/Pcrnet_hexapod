# PCR Theory Definitions

本文档用于沉淀 PCR 主线中可直接进入论文的方法定义、术语、公式和评测口径。后续凡是已经收口、可规范化表达的理论定义，都统一追加到本文档，避免论文写作时反复改口径。

## 1. High-Conflict State

### 1.1 Definition

高冲突状态定义为：机器人处于障碍行交互窗口内，同时跟随专家与避障专家产生显著且相互竞争的控制需求。

英文论文表述建议：

```text
A high-conflict state is defined as a state in which the robot is interacting with an obstacle row, while the follow expert and the avoid expert simultaneously produce strong and competing control demands.
```

该定义不是单纯的前向风险定义。`risk_F` 只能描述当前 Follow 命令方向上的可见风险，不能完整表示 Follow/Avoid 是否正在发生仲裁冲突。因此论文主机制分析优先使用高冲突状态，而不是仅按 `risk_F` 分桶。

### 1.2 Command Convention

高层命令统一写为：

```text
u = [v_x, v_y, omega]
```

其中：

```text
v_x     : body-frame lateral velocity, positive to robot right
v_y     : body-frame forward velocity
omega   : yaw rate
```

跟随专家命令记为：

```text
u_F = [v_{F,x}, v_{F,y}, omega_F]
```

避障专家命令记为：

```text
u_A = [v_{A,x}, v_{A,y}, omega_A]
```

当前 PCR 主线中，跟随压力主要由 `v_{F,y}` 表示，避障压力主要由 `|v_{A,x}|` 表示。

### 1.3 Binary High-Conflict Mask

高冲突二值判据写为：

```text
H_conflict = O_row AND P_follow AND P_avoid
```

其中，障碍交互窗口为：

```text
O_row =
    row_valid
    AND robot_front_y > row_front_y - d_pre
    AND robot_rear_y  < row_back_y  + d_post
```

跟随压力为：

```text
P_follow = 1[v_{F,y} > v_F_thr]
```

避障压力为：

```text
P_avoid = 1[|v_{A,x}| > v_A_thr]
```

当前 eval 默认参数为：

```text
d_pre   = 0.6 m
d_post  = 0.3 m
v_F_thr = 0.20
v_A_thr = 0.10
```

这些参数的作用不是定义训练任务，而是定义评测中“真实发生 Follow/Avoid 仲裁冲突”的状态集合。

### 1.4 Continuous Conflict Score

为了绘制机制图和做分桶统计，可以定义连续高冲突强度：

```text
C_conflict =
    O_row
    * clip((v_{F,y} - v_F_low) / (v_F_high - v_F_low), 0, 1)
    * clip((|v_{A,x}| - v_A_low) / (v_A_high - v_A_low), 0, 1)
```

推荐参数：

```text
v_F_low  = 0.15
v_F_high = 0.45
v_A_low  = 0.05
v_A_high = 0.30
```

二值高冲突状态可由连续分数得到：

```text
H_conflict = 1[C_conflict > tau_conflict]
```

当前默认：

```text
tau_conflict = 0.25
```

论文机制图中，可以按 `C_conflict` 分桶观察：

```text
signed_w
y_eff - y_raw
y_raw and y_eff
row-progress score
collision rate
```

核心证据链为：

```text
C_conflict increases
=> learned conflict prior changes
=> y_eff - y_raw changes
=> task safety / progress changes
```

### 1.5 Phase Labels

障碍交互窗口内可进一步分为三个阶段，但阶段标签只用于分析，不作为主高冲突判据。

```text
approach:
    robot_front_y < row_front_y

inside:
    robot_front_y >= row_front_y
    AND robot_rear_y <= row_back_y

release:
    robot_front_y > row_back_y
    AND robot_rear_y < row_back_y + d_post
```

论文中推荐写法：

```text
We use a unified high-conflict mask for the main analysis and categorize the selected states into approach, inside, and release phases only for diagnostic visualization.
```

### 1.6 Usage Boundary

该高冲突定义依赖障碍行几何真值，因此只能用于：

```text
1. evaluation diagnostics
2. mechanism plots
3. auxiliary-label analysis during simulation training
```

禁止作为：

```text
1. actor input
2. real-world deployment input
3. direct execution-time rule
```

论文中必须明确：

```text
The privileged high-conflict mask is used only for evaluation diagnostics and auxiliary-label analysis. It is not provided to the actor policy and is not required during real-world deployment.
```

中文口径：

```text
该高冲突定义是评测与分析用的特权诊断尺子，不作为策略输入，也不作为实机部署输入。
```

### 1.7 Relation to Risk-F Binning

`risk_F` 分桶仍可保留为辅助图，但不能作为证明 PCR 仲裁机制的唯一证据。

原因是：

```text
risk_F high:
    Follow direction is risky, but Avoid may not be active.

risk_F low:
    Follow direction appears safe, but the robot may still be in a release phase
    where side obstacles remain dangerous or target observability is at risk.
```

因此论文主机制图应优先使用：

```text
privileged high-conflict score / mask
```

而不是只使用：

```text
risk_F bin
```

