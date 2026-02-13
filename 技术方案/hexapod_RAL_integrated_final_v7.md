# 六足机器人移动目标跟随与自主避障：面向IEEE RAL的最终论文架构

## 整合说明

本文档整合了以下内容：

1. **用户原有方案**：技术方案V6（PCR-Net++：y + w + β，β联动后处理）
2. **Claude意见**：预测性Subgoal + 门控专家策略 + 全向视野保持
3. **Gemini意见**：窄缝场景亮点 + 理论深度 + 条件性融合防抖
4. **GPT意见**：可达性判别可复现 + β机制化联动 + Pareto曲线

---

# 第一部分：核心定位与贡献重组

## 1.1 你原方案与三方意见的对比分析

| 维度       | 你的V6方案                    | 三方讨论方案                   | 整合决策                      |
| -------- | ------------------------- | ------------------------ | ------------------------- |
| **核心架构** | MoE（y）+ Prior（w）+ 风险预算（β） | 预测Subgoal + 门控MoE + 安全滤波 | **保留你的y+w+β架构，更成熟**       |
| **预测机制** | w：命令条件化预测冲突prior          | Subgoal：轨迹预测+可达性判别       | **w的定义需要强化为"命令条件化"，避免退化** |
| **安全机制** | β联动Post-Processor约束族      | 安全滤波器（β参数化）              | **完全一致，你的方案更完整**          |
| **核心场景** | S1门洞走廊 + S2柱阵森林           | 窄缝场景（人可通行/机不可通行）         | **将S1门洞场景强化为"窄缝场景"的核心演示** |
| **叙事亮点** | β的Pareto曲线                | 窄缝场景的视觉演示                | **两者结合：窄缝场景+Pareto曲线**    |
| **六足特性** | 未显式强调                     | 全向运动+视野保持                | **需要补充：六足全向特性作为使能技术**     |

## 1.2 整合后的核心判断

**你的V6方案已经非常完整**，与三方讨论的方向高度一致。需要强化的是：

1. **将"窄缝场景"（人可通行/机不可通行）显式化为论文的核心演示场景**
   
   - 对应你的S1门洞场景，但需要强调"门洞宽度设计"使得人能过但机器人紧凑通过

2. **将六足全向运动特性显式纳入贡献**
   
   - 你的底层locomotion已经支持全向运动，但高层需要显式利用这一特性实现"视野保持"

3. **强化w的"命令条件化"属性，避免审稿人质疑**
   
   - GPT明确指出：如果w只是学习clearance，贡献会降级
   - 你的V6已经有3.2节的设计，需要在实验中严格验证

4. **Pareto曲线的实验设计需要更严格**
   
   - GPT指出：需要证明β变化导致行为变化的单调性和可解释性

## 1.3 最终论文贡献定位（三点，与你的V6对齐但叙事强化）

| 贡献     | V6中的对应            | RAL强化版叙事                                      | 审稿人关注点                    |
| ------ | ----------------- | --------------------------------------------- | ------------------------- |
| **C1** | w：命令条件化预测冲突prior  | **前瞻性冲突消解**：通过命令条件化预测prior实现从"被动反应"到"前瞻切换"的跨越 | w必须是命令条件化的，不能退化为clearance |
| **C2** | y+w融合 + 平滑/滞回机制   | **稳定门控融合**：带冲突先验的软门控专家融合，抑制切换抖动               | 条件性融合防止线性融合陷阱             |
| **C3** | β联动Post-Processor | **风险预算机制化**：β调节约束族参数，形成可复现的风险-效率Pareto曲线      | β必须明确控制哪些参数，Pareto必须单调可解释 |

**补充贡献（作为系统使能技术，非主贡献）**：

- 六足全向运动下的视野保持约束

---

# 第二部分：统一叙事主线

## 2.1 论文标题建议

**英文**：
Risk-Aware Gated Expert Policy with Predictive Conflict Prior for Hexapod Robot Human Following

**中文**：
基于预测性冲突先验的风险感知门控专家策略：面向六足机器人人类跟随

> 说明：标题中包含了三个核心贡献的关键词：Risk-Aware（β）、Gated Expert（y）、Predictive Conflict Prior（w）

## 2.2 统一问题定义

**核心问题**（与你的V6一致）：
在"移动目标跟随（Follow）"与"复杂避障（Avoid）"冲突场景下，实现**稳定、可控、丝滑**的行为选择。

**核心挑战**（V6的P1痛点）：
y-only MoE的两大失败模式：

1. **滞后切换（late switch）**：进入高风险区域后才偏向Avoid，导致near-miss/碰撞
2. **边界抖动（chattering）**：风险边界附近y来回切换，导致cmd抖动、速度振荡

**核心场景**（强化版）：

- **S1门洞走廊**：走廊内设置门洞，门洞宽度设计为**人可轻松通过（>0.8m）但机器人需紧凑通过（接近机器人宽度）**
- 这个场景天然形成"人先通过门洞、机器人需要决策是紧跟还是减速排队"的冲突

**我们的方案**：
PCR-Net++（y + w + β，β联动后处理）

- y：MoE仲裁权重（Follow vs Avoid的结构性选择）
- w：命令条件化预测冲突prior（提前性）
- β：风险预算（可控性 + 部署旋钮），联动Post-Processor约束族

## 2.3 论文主张（与V6的0.3节一致）

**P1（y-only的系统性问题）**：
在Follow/Avoid冲突片段（门洞、贴边绕行、柱阵死胡同）中，y-only MoE存在滞后切换和边界抖动问题。

**P2（β的核心贡献）**：
引入风险预算β并联动Command Post-Processor的"约束族参数"，把安全/平滑从"奖励调参问题"转为"机制可控问题"，从而：

- 显著降低near-miss、switch rate与cmd jerk
- 形成可复现的**风险-效率Pareto曲线**

**P3（w的核心贡献）**：
引入w（命令条件化预测冲突prior），把"提前性"从瞬时门控中解耦，进一步降低滞后与切换震荡。

---

# 第三部分：技术方案详细设计（基于V6，整合三方意见强化）

## 3.1 系统架构（与V6一致，补充六足特性）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              感知层 (5-10Hz)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────────┐          │
│  │ 深度相机     │  │ 目标检测     │  │ 本体感知 (IMU + 关节编码器) │          │
│  │ (仅高层用)   │→│ + 跟踪      │  │                            │          │
│  └─────────────┘  └─────────────┘  └────────────────────────────┘          │
└────────────────────────────┬────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Affordance Estimator                                 │
│                                                                             │
│   Depth (128×128) → {Occupancy 16×16, PassableGap 16×16, Difficulty d}     │
│                                                                             │
│   Teacher: GT affordance | Student: Estimated affordance (蒸馏)             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PCR-Net++：y + w + β 仲裁网络                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        输入特征                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │   │
│  │  │ 状态特征        │  │ 候选命令特征    │  │ 短历史统计特征          │ │   │
│  │  │ - robot_state  │  │ - cmd_follow   │  │ - recent clearance    │ │   │
│  │  │ - target_state │  │ - cmd_avoid    │  │ - recent Δcmd         │ │   │
│  │  │ - affordance   │  │ - RiskAlong(F) │  │ - recent progress     │ │   │
│  │  │ - difficulty   │  │ - RiskAlong(A) │  │ - collision_flag      │ │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PCR-Net++ 网络结构                                │   │
│  │                                                                     │   │
│  │   输入 ──→ MLP(256,128) ──→ ┬──→ y_head ──→ y ∈ [0,1]             │   │
│  │                             ├──→ w_head ──→ w ∈ [0,1] (prior)     │   │
│  │                             └──→ β_head ──→ β ∈ [0,1] (risk)      │   │
│  │                                                                     │   │
│  │   关键：w_head的输入必须包含cmd_follow/cmd_avoid或其投影特征          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    y_eff 融合 + 平滑机制                             │   │
│  │                                                                     │   │
│  │   # 约定：y/y_eff 为 Follow 权重；w 为冲突强度（w↑ => 更应偏 Avoid）          │   │
│  │   y_eff_raw = clamp((1-λ_w)·y + λ_w·(1-w), 0, 1)                   │   │
│  │                                                                     │   │
│  │   平滑：y_eff_smooth = γ·y_eff_prev + (1-γ)·y_eff_raw              │   │
│  │                                                                     │   │
│  │   滞回：if |y_eff_smooth - y_eff_prev| < δ_hyst: y_eff = y_eff_prev│   │
│  │          else: y_eff = y_eff_smooth                                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         专家策略 + 命令融合                                  │
│                                                                             │
│   ┌────────────┐                                  ┌────────────┐           │
│   │ Follow专家  │                                  │ Avoid专家   │           │
│   │ π_follow   │                                  │ π_avoid    │           │
│   │            │                                  │            │           │
│   │ 目标：效率  │                                  │ 目标：安全  │           │
│   │ 输出：cmd_F │                                  │ 输出：cmd_A │           │
│   └─────┬──────┘                                  └──────┬─────┘           │
│         │                                                │                 │
│         └──────────────────┬─────────────────────────────┘                 │
│                            ↓                                               │
│              cmd_base = y_eff·cmd_F + (1 - y_eff)·cmd_A                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              条件性融合（解决线性融合陷阱）                            │  │
│   │                                                                     │  │
│   │   consistency = cos_sim(cmd_F, cmd_A)                              │  │
│   │                                                                     │  │
│   │   if consistency > τ_consist:     # 方向大致一致                    │  │
│   │       cmd_raw = cmd_base          # 标准线性融合                    │  │
│   │   elif P_collision > τ_danger:    # 高碰撞风险 + 方向矛盾           │  │
│   │       cmd_raw = cmd_A             # Winner-Take-All，安全优先       │  │
│   │   else:                           # 中等冲突                        │  │
│   │       weights = softmax([y_eff, 1-y_eff] / τ_temp)                 │  │
│   │       cmd_raw = weights[0]·cmd_F + weights[1]·cmd_A  # 非线性软切换 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                            ↓ cmd_raw                                       │
└────────────────────────────┬────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    β联动 Command Post-Processor                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    约束族参数映射（β → 参数）                        │  │
│   │                                                                     │  │
│   │   safe_dist(β)    = safe_min + β·(safe_max - safe_min)             │  │
│   │   max_cmd_lin(β)  = v_fast - β·(v_fast - v_safe)                   │  │
│   │   max_cmd_ang(β)  = ω_fast - β·(ω_fast - ω_safe)                   │  │
│   │   max_delta_lin(β)= slew_fast - β·(slew_fast - slew_slow)          │  │
│   │   max_delta_ang(β)= slew_fast_ang - β·(slew_fast_ang - slew_slow_ang)│  │
│   │   risk_clamp_gain(β) = gain_min + β·(gain_max - gain_min)          │  │
│   │                                                                     │  │
│   │   语义保证：β↑（更保守）=> safe_dist↑, max_cmd↓, max_delta↓        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Post-Processor 处理流程                          │  │
│   │                                                                     │  │
│   │   1. 速度限幅：cmd_clamp = clip(cmd_raw, -max_cmd(β), max_cmd(β))  │  │
│   │                                                                     │  │
│   │   2. 变化率限制：                                                   │  │
│   │      Δcmd = cmd_clamp - cmd_prev                                   │  │
│   │      Δcmd_clamp = clip(Δcmd, -max_delta(β), max_delta(β))          │  │
│   │      cmd_smooth = cmd_prev + Δcmd_clamp                            │  │
│   │                                                                     │  │
│   │   3. 风险钳制（基于clearance）：                                    │  │
│   │      if clearance < safe_dist(β):                                  │  │
│   │          scale = clearance / safe_dist(β)                          │  │
│   │          cmd_safe = cmd_smooth · scale^risk_clamp_gain(β)          │  │
│   │      else:                                                          │  │
│   │          cmd_safe = cmd_smooth                                     │  │
│   │                                                                     │  │
│   │   4. 视野保持约束（六足全向特性）：                                  │  │
│   │      θ_desired = atan2(target_y, target_x)  # 期望朝向目标          │  │
│   │      ω_correction = K_p · (θ_desired - θ_current)                  │  │
│   │      cmd_final = (cmd_safe.vx, cmd_safe.vy, cmd_safe.ω + ω_correction)│  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                            ↓ cmd_final                                     │
└────────────────────────────┬────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    底层 Locomotion (50Hz, 冻结)                              │
│                                                                             │
│   EGPO Policy: proprio + privileged + height_map + cmd_final → 18-d actions│
│                                                                             │
│   注意：底层不使用深度相机，只用特权高度图                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 贡献1详细设计：w（命令条件化预测冲突prior）

### 3.2.1 w的定义（论文一句话说清）

> w表达"短未来内Follow候选命令 vs Avoid候选命令"的**可行性差异/冲突趋势**，用于提前性仲裁（降低滞后与切换震荡）。

### 3.2.2 w必须命令条件化（GPT强调的关键点）

**w的输入必须包含至少一种"命令条件"信息**：

```python
# w的输入特征（最小集合）
w_input = {
    # 必选（至少一类）
    'cmd_follow': cmd_F,                    # 候选命令本身
    'cmd_avoid': cmd_A,                     
    'risk_along_follow': RiskAlong(cmd_F),  # 沿候选命令方向的风险估计
    'risk_along_avoid': RiskAlong(cmd_A),   

    # 建议增加
    'affordance_stack': recent_affordance,   # 短历史affordance
    'recent_cmd': recent_cmd_history,        # 最近命令历史
    'target_velocity': target_vel,           # 目标运动信息
}
```

**RiskAlong计算**：

```python
def RiskAlong(cmd, occupancy_map, lookahead=1.0):
    """
    沿候选命令方向采样未来路径的风险

    Args:
        cmd: (vx, vy, ω) 候选命令
        occupancy_map: 16×16 占据栅格
        lookahead: 前瞻距离 (m)

    Returns:
        risk: [0, 1] 风险值
    """
    # 计算命令方向
    direction = normalize(cmd[:2])  # (vx, vy) 归一化

    # 沿方向采样点
    sample_points = []
    for t in np.linspace(0, lookahead, 10):
        point = current_pos + direction * t
        sample_points.append(point)

    # 查询占据栅格
    occupancy_values = query_occupancy(occupancy_map, sample_points)

    # 风险 = 最大占据值或加权平均
    risk = max(occupancy_values)  # 或 np.mean(occupancy_values)

    return risk
```

### 3.2.3 w的训练信号（避免退化）

**弱监督标签**：

```python
# 基于命令条件风险差异构造软标签
risk_F = RiskAlong(cmd_follow)
risk_A = RiskAlong(cmd_avoid)

# w* 表示"应该更信Avoid的程度"
w_target = sigmoid(k * (risk_F - risk_A))  # k=5.0

# 辅助损失
L_w = BCE(w_pred, w_target)
```

**退化检测**（V6的3.5节）：

- 相关性退化：`corr(w, clearance) > 0.9` 且对near-miss/jitter无改善 → 判退化
- 迟滞退化：w只有在极近距离才变化 → 仍是滞后式，不满足"预测prior"定位

### 3.2.4 w与Subgoal的关系

三方讨论中提到的"预测性Subgoal"可以作为w的一种**高级实现**：

| 方案                         | 实现复杂度 | 效果  | 推荐        |
| -------------------------- | ----- | --- | --------- |
| **方案A：RiskAlong特征**        | 低     | 中   | **MVP首选** |
| **方案B：轨迹预测+可达性→Subgoal→w** | 高     | 高   | 后续增强      |

对于RAL投稿，**建议先用方案A**验证收益，如果效果不足再考虑方案B。

## 3.3 贡献2详细设计：y_eff融合 + 平滑机制

### 3.3.1 y与w的融合

```python
# y_eff计算
lambda_w = 0.5  # 可调超参，或由网络输出

y_eff_raw = clamp((1 - lambda_w) * y + lambda_w * (1 - w), 0, 1)
```

### 3.3.2 平滑与滞回机制（解决抖动）

```python
# 时序平滑
gamma = 0.8  # 平滑系数
y_eff_smooth = gamma * y_eff_prev + (1 - gamma) * y_eff_raw

# 滞回机制
delta_hyst = 0.1  # 滞回阈值
if abs(y_eff_smooth - y_eff_prev) < delta_hyst:
    y_eff = y_eff_prev  # 不更新
else:
    y_eff = y_eff_smooth
```

### 3.3.3 条件性融合（解决线性融合陷阱，Gemini建议）

```python
def conditional_fusion(cmd_F, cmd_A, y_eff, conflict_level):
    """
    条件性融合，避免"两个专家输出矛盾时产生垃圾指令"
    """
    # 计算一致性
    consistency = cosine_similarity(cmd_F[:2], cmd_A[:2])

    if consistency > 0.5:  # 方向大致一致
        # 标准线性融合
        cmd = y_eff * cmd_F + (1 - y_eff) * cmd_A

    elif conflict_level > 0.8:  # 高冲突 + 方向矛盾
        # Winner-Take-All，安全优先
        cmd = cmd_A

    else:  # 中等冲突
        # 非线性软切换
        temperature = 0.5
        weights = softmax([y_eff, 1 - y_eff] / temperature)
        cmd = weights[0] * cmd_F + weights[1] * cmd_A

    return cmd
```

### 3.3.4 训练时的切换惩罚

```python
# 奖励函数中添加
r_switch = -w_switch * abs(y_eff_t - y_eff_t_minus_1)
r_jerk = -w_jerk * norm(cmd_t - cmd_t_minus_1)
```

## 3.4 贡献3详细设计：β联动Post-Processor

### 3.4.1 β的语义（V6的1.2节）

- **β大 = 更保守 / 更安全（更强约束）**
- β必须与Post-Processor参数映射方向一致

### 3.4.2 约束族参数映射表（需要你提供具体数值）

| 参数     | 符号                    | β=0（激进）  | β=1（保守）  | 映射公式 |
| ------ | --------------------- | -------- | -------- | ---- |
| 安全距离阈值 | $d_{\text{safe}}$     | ? m      | ? m      | 线性插值 |
| 最大线速度  | $v_{\max}$            | ? m/s    | ? m/s    | 线性插值 |
| 最大角速度  | $\omega_{\max}$       | ? rad/s  | ? rad/s  | 线性插值 |
| 线速度变化率 | $\dot{v}_{\max}$      | ? m/s²   | ? m/s²   | 线性插值 |
| 角速度变化率 | $\dot{\omega}_{\max}$ | ? rad/s² | ? rad/s² | 线性插值 |
| 风险钳制增益 | $k_{\text{risk}}$     | ?        | ?        | 线性插值 |

### 3.4.3 β退化防护（V6的4.3节）

**β=1不能等价于"几乎无法前进"**

```python
# 约束映射范围
v_safe_min = 0.3  # 即使β=1，也要保证最低速度
v_fast_max = 1.0

# 映射函数
def v_max(beta):
    return v_fast_max - beta * (v_fast_max - v_safe_min)
    # β=0 → 1.0 m/s
    # β=1 → 0.3 m/s（不是0！）
```

**进度保底机制**：

```python
# 奖励函数中添加
r_time = -w_time * dt  # 时间惩罚，防止"停住刷分"
r_progress = w_progress * progress_to_goal  # 进度奖励
```

### 3.4.4 视野保持约束（六足特性，补充贡献）

```python
def gaze_constraint(cmd, target_rel_pos, K_p=1.0):
    """
    利用六足全向运动能力，保持目标在视野中心

    六足优势：可以边走边转，不需要停下来调整朝向
    """
    # 期望朝向（指向目标）
    theta_desired = atan2(target_rel_pos[1], target_rel_pos[0])

    # 当前朝向
    theta_current = robot_yaw

    # 朝向误差
    theta_error = normalize_angle(theta_desired - theta_current)

    # 角速度修正
    omega_correction = K_p * theta_error

    # 修正后的命令（保持线速度，修正角速度）
    cmd_corrected = (cmd[0], cmd[1], cmd[2] + omega_correction)

    return cmd_corrected
```

---

# 第四部分：场景设计（与V6对齐，强化"窄缝场景"）

## 4.1 场景定义

### S1：门洞走廊（核心演示场景）

**几何设计**（强化为"窄缝场景"）：

- 走廊宽度：2.0m
- 门洞宽度：**0.8m - 1.0m**（关键：人可轻松通过，机器人紧凑通过）
- 门洞数量：1-3个
- 机器人宽度：假设0.6m（需要你确认）

**场景意图**：

- 人类目标以正常速度穿越门洞
- 机器人需要决策：紧跟穿越 vs 减速排队 vs 提前调整位置

**验证的贡献**：

- w的提前性：在门洞前w是否提前上升
- β的保守度：不同β下穿越门洞的行为差异
- 视野保持：穿越门洞时是否保持目标在视野中

### S2：柱阵森林

**几何设计**：

- 柱子半径：0.1m - 0.3m
- 柱子密度：稀疏（2/m²）→ 密集（5/m²）
- 分布模式：随机/Poisson/聚类

**场景意图**：

- 开放域避障 + 跟随速度约束
- 存在多条可行路径，需要权衡效率和安全

### S3-S5：扩展场景（训练多样性）

与V6的5.2节一致。

### S6：结构化OOD（核心证据）

与V6的5.3节一致：训练绝不出现，作为论文主证据与回归测试基准。

---

# 第五部分：实验设计（整合三方意见强化）

## 5.1 实验目标

| 实验类别         | 目标                             | 对应贡献       |
| ------------ | ------------------------------ | ---------- |
| **基准对比**     | 证明PCR-Net++整体优于y-only MoE和其他基线 | 全部         |
| **消融实验**     | 验证y、w、β、条件性融合、平滑机制的各自贡献        | C1, C2, C3 |
| **Pareto实验** | 展示β风险预算的机制化效果（GPT强调）           | C3         |
| **w退化检测**    | 证明w是命令条件化的，不是clearance分类器      | C1         |
| **泛化性实验**    | S6 OOD + 视觉退化 + 动力学随机化         | 全部         |

## 5.2 基准对比（与V6的9.1节一致，补充）

| 基线                       | 描述              |
| ------------------------ | --------------- |
| **Single Policy**        | 单一端到端策略（无MoE）   |
| **y-only MoE**           | 标准MoE，仅门控y      |
| **Rule-based**           | 基于距离/间隙阈值的规则仲裁  |
| **Fixed Post-Processor** | MoE但β不可学，固定为0.5 |
| **APF**                  | 人工势场法（传统方法基线）   |
| **DWA**                  | 动态窗口法（传统方法基线）   |

## 5.3 消融矩阵（与V6的9.2-9.4节一致）

| 版本               | y   | w   | β   | β联动 | 条件融合 | 平滑机制 |
| ---------------- | --- | --- | --- | --- | ---- | ---- |
| y-only           | ✓   | ✗   | ✗   | ✗   | ✗    | ✗    |
| y + w            | ✓   | ✓   | ✗   | ✗   | ✗    | ✗    |
| y + β (no-link)  | ✓   | ✗   | ✓   | ✗   | ✗    | ✗    |
| y + β (link)     | ✓   | ✗   | ✓   | ✓   | ✗    | ✗    |
| y + w + β (link) | ✓   | ✓   | ✓   | ✓   | ✗    | ✗    |
| **Full (Ours)**  | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    |

**关键消融（GPT强调）**：

- **β联动 vs 不联动**：证明贡献是"机制化接口"，而不是"多一个网络头"

## 5.4 评估指标（与V6的8.3节一致）

### 必报指标

| 类别     | 指标          | 符号                   | 定义              |
| ------ | ----------- | -------------------- | --------------- |
| **任务** | 成功率         | $r_{\text{success}}$ | 到达目标比例          |
|        | 到达时间        | $T_{\text{goal}}$    | 平均完成时间          |
| **安全** | 碰撞率         | $r_{\text{col}}$     | 碰撞episode数/总数   |
|        | Near-miss   | $f_{\text{nm}}$      | 低clearance时间占比  |
| **丝滑** | cmd jerk    | $J_{\text{cmd}}$     | 命令变化率RMS        |
|        | switch rate | $r_{\text{switch}}$  | y_eff显著变化事件频率   |
| **可控** | Pareto曲线    | -                    | β sweep的风险-效率曲线 |

### 指标口径（V6的8.3.1节）

```python
# near-miss（推荐主指标）
c_thr = 0.3  # 阈值，需根据机器人尺寸调整
near_miss = sum_t(max(0, c_thr - clearance_t) * dt)

# switch rate
dy_thr = 0.2  # 显著变化阈值
switch_count = sum_t(1 if abs(y_eff_t - y_eff_t_minus_1) > dy_thr else 0)
switch_rate = switch_count / T  # 每秒事件数

# cmd jerk
jerk_lin = std(diff(cmd_lin))
jerk_ang = std(diff(cmd_ang))
```

## 5.5 Pareto实验（GPT强调的核心验证）

### 5.5.1 实验协议

```python
# β sweep设置
beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # 最少5点

# 固定条件
model_weights = trained_model  # 同一模型权重
seeds = [0, 1, 2, 3, 4]  # 固定随机种子
episodes_per_beta = 100  # 每个β点的episode数
scene = "S1"  # 门洞走廊
```

### 5.5.2 预期结果

**表：不同β值下的性能指标**

| β    | $T_{\text{goal}}$ (s)↓ | $\bar{v}$ (m/s)↑ | $d_{\min}^{\text{obs}}$ (m)↑ | $f_{\text{nm}}$ (%)↓ | $J_{\text{cmd}}$↓ |
| ---- | ---------------------- | ---------------- | ---------------------------- | -------------------- | ----------------- |
| 0.0  | 最短                     | 最快               | 最小                           | 最高                   | 最高                |
| 0.25 | -                      | -                | -                            | -                    | -                 |
| 0.5  | 中等                     | 中等               | 中等                           | 中等                   | 中等                |
| 0.75 | -                      | -                | -                            | -                    | -                 |
| 1.0  | 最长                     | 最慢               | 最大                           | 最低                   | 最低                |

**预期特性**（GPT强调必须满足）：

1. **单调性**：效率指标随β单调下降，安全指标随β单调上升
2. **可解释性**：β=0.5是效率-安全的平衡点
3. **无"停住"**：β=1时仍有足够的前进速度

### 5.5.3 Pareto曲线绘制

```
安全性指标 (d_min_obs)
    ↑
    │                              ★ β=1.0
    │                         ★ β=0.75
    │                    ★ β=0.5
    │               ★ β=0.25
    │          ★ β=0.0
    └──────────────────────────────────→ 效率指标 (1/T_goal)
```

## 5.6 w退化检测实验（GPT强调）

### 5.6.1 相关性检测

```python
# 计算w与clearance的相关性
w_values = collect_w_over_episodes()
clearance_values = collect_clearance_over_episodes()

correlation = pearsonr(w_values, clearance_values)

# 判定
if correlation > 0.9 and near_miss_improvement < 5%:
    print("警告：w退化为clearance分类器！")
```

### 5.6.2 命令条件化验证

```python
# 验证w确实依赖于候选命令
# 实验：固定状态，改变cmd_follow/cmd_avoid，观察w变化

for state in test_states:
    w_values = []
    for cmd_F, cmd_A in different_commands:
        w = model.predict_w(state, cmd_F, cmd_A)
        w_values.append(w)

    # 如果w随命令变化而变化，说明是命令条件化的
    w_variance = var(w_values)
    assert w_variance > threshold, "w不是命令条件化的！"
```

## 5.7 窄缝场景专项实验

### 5.7.1 场景设计

```python
# 门洞宽度设置
gate_widths = [0.7, 0.8, 0.9, 1.0, 1.2]  # m
robot_width = 0.6  # m（需确认）
human_width = 0.5  # m

# 关键对比
# gate_width = 0.8m：人可轻松通过，机器人紧凑通过
# gate_width = 0.7m：人可通过，机器人可能卡住
```

### 5.7.2 指标

| 指标        | 定义              |
| --------- | --------------- |
| 穿越成功率     | 成功通过门洞的比例       |
| 提前决策时机    | w开始上升的时刻距离门洞的距离 |
| 穿越时的y_eff | 穿越门洞时的平均y_eff值  |
| 视野保持率     | 穿越时目标在视野中的时间占比  |

---

# 第六部分：论文结构与写作建议

## 6.1 论文结构（RAL 6+1页）

| 章节                       | 页数  | 内容                                                                  |
| ------------------------ | --- | ------------------------------------------------------------------- |
| I. Introduction          | 0.7 | 问题定义、y-only痛点、三点贡献                                                  |
| II. Related Work         | 0.5 | MoE控制、风险感知导航、人类跟随                                                   |
| III. Problem Formulation | 0.4 | 状态/动作空间、奖励函数                                                        |
| IV. Method               | 2.0 | A. w命令条件化prior (0.6) B. y_eff融合+平滑 (0.6) C. β联动Post-Processor (0.8) |
| V. Experiments           | 1.9 | A. 设置 B. 基准对比 C. 消融 D. Pareto E. 窄缝场景 F. OOD                        |
| VI. Conclusion           | 0.3 | 总结、局限、未来工作                                                          |
| References               | 0.2 | ~25篇                                                                |

## 6.2 关键图表规划

| 图/表       | 内容                   | 位置           |
| --------- | -------------------- | ------------ |
| Fig.1     | 系统架构（三层）             | Introduction |
| Fig.2     | y-only失败模式可视化（滞后+抖动） | Introduction |
| Fig.3     | PCR-Net++网络结构        | Method       |
| Fig.4     | β→约束族参数映射            | Method       |
| Fig.5     | 基准对比轨迹               | Experiments  |
| Fig.6     | **Pareto曲线（核心图）**    | Experiments  |
| Fig.7     | y/w/β时序曲线 + 事件对齐     | Experiments  |
| Fig.8     | 窄缝场景演示               | Experiments  |
| Table I   | 基准对比指标               | Experiments  |
| Table II  | 消融实验指标               | Experiments  |
| Table III | β参数映射表               | Method       |

---

# 第七部分：待确认参数

为了完成最终方案，需要你提供以下参数：

## 7.1 机器人物理参数

| 参数          | 需要的值    |
| ----------- | ------- |
| 机体宽度（行走时最大） | ? m     |
| 最大前进速度      | ? m/s   |
| 最大侧向速度      | ? m/s   |
| 最大旋转速度      | ? rad/s |

## 7.2 β约束族参数范围

| 参数            | β=0（激进）  | β=1（保守）  |
| ------------- | -------- | -------- |
| safe_dist     | ? m      | ? m      |
| max_cmd_lin   | ? m/s    | ? m/s    |
| max_cmd_ang   | ? rad/s  | ? rad/s  |
| max_delta_lin | ? m/s²   | ? m/s²   |
| max_delta_ang | ? rad/s² | ? rad/s² |

## 7.3 场景参数

| 参数               | 需要的值       |
| ---------------- | ---------- |
| S1门洞宽度范围         | ? m - ? m  |
| S2柱子密度范围         | ? - ? 个/m² |
| near-miss阈值c_thr | ? m        |

---

# 第八部分：总结

## 8.1 你的V6方案与三方讨论的关系

**你的V6方案已经非常完整**，核心架构（y+w+β，β联动Post-Processor）与三方讨论高度一致。

需要强化的点：

1. **窄缝场景显式化**：将S1门洞场景强化为"人可通行/机不可通行"的核心演示
2. **六足全向特性**：在Post-Processor中添加视野保持约束
3. **w的命令条件化验证**：添加退化检测实验
4. **Pareto曲线严格性**：确保单调性和可解释性

## 8.2 最小可行版本（MVP）

如果时间紧张，可以先实现：

1. **y + β (link)**：验证β联动Post-Processor的收益
2. **添加w**：验证提前性的收益
3. **窄缝场景**：录制演示视频

后续再补充：

- 条件性融合
- 完整消融矩阵
- OOD泛化实验

## 8.3 风险与应对

| 风险            | 应对                 |
| ------------- | ------------------ |
| w退化为clearance | 严格命令条件化输入 + 退化检测实验 |
| β=1停住         | 收紧映射范围 + 进度保底      |
| Pareto不单调     | 检查映射函数 + 调整参数范围    |
| 门控抖动          | 平滑机制 + 滞回 + 切换惩罚   |
