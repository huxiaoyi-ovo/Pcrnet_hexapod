# 六足机器人移动目标跟随与自主避障：完整技术方案

## PCR-Net++: Predictive Conflict Resolution Network for Hexapod Robot Human Following

**版本**: V7.0 Final  
**日期**: 2026-01-30  
**目标**: IEEE RAL投稿

---

# 目录

1. [核心参数定义](#第一部分核心参数定义)
2. [数学公式体系](#第二部分数学公式体系)
3. [网络架构与代码实现](#第三部分网络架构与代码实现)
4. [Post-Processor实现](#第四部分post-processor实现)
5. [奖励函数设计](#第五部分奖励函数设计)
6. [仿真场景设计](#第六部分仿真场景设计)
7. [完整训练流程](#第七部分完整训练流程)
8. [评估与实验设计](#第八部分评估与实验设计)
9. [行动指南与检查清单](#第九部分行动指南与检查清单)

---

# 第一部分：核心参数定义

## 1.1 机器人物理参数（来自URDF）

```python
ROBOT_PARAMS = {
    # 机体尺寸
    'body_size': (0.2, 0.44, 0.05),      # (x, y, z) in meters
    'body_mass': 5.5207,                   # kg
    'total_mass': 12.4534,                 # kg

    # 行走尺寸
    'walking_width': 0.70,                 # m（固定，不可变）
    'walking_length': 0.50,                # m（估计）

    # 足端参数
    'toe_radius': 0.009,                   # m
    'leg_reach_range': (0.14, 0.24),       # m（足端径向范围）

    # 关节限制
    'joint_effort_limit': 27.0,            # N·m
    'joint_velocity_limit': 5.0,           # rad/s
}
```

## 1.2 运动学参数（来自训练配置）

```python
KINEMATICS_PARAMS = {
    # 速度限制（训练上限）
    'max_lin_vel_x': 1.0,                  # m/s
    'max_lin_vel_y': 1.5,                  # m/s（六足优势：侧向更快）
    'max_ang_vel_yaw': 2.0,                # rad/s

    # 控制频率
    'sim_dt': 0.005,                       # s
    'control_decimation': 4,               # steps
    'low_level_freq': 50,                  # Hz
    'high_level_freq': 10,                 # Hz
    'high_level_dt': 0.1,                  # s
}
```

## 1.3 任务参数

```python
TASK_PARAMS = {
    # 目标参数
    'target_max_speed': 1.2,               # m/s（上限：覆盖人类快走/小跑与课程扰动）
    'target_typical_speed': 0.6,           # m/s（人类典型速度）

    # 跟随距离
    'follow_distance_desired': 1.0,        # m（统一口径：论文/训练/评测/部署）
    'follow_distance_min': 0.7,            # m
    'follow_distance_max': 1.4,            # m

    # 视野保持（中心窗口约束，不使用“全FOV视野内即可”的宽松标准）
    'target_fov_soft_scale': 0.35,         # soft_half = 0.5*fov*scale
    'target_fov_hard_scale': 0.70,         # hard_half = 0.5*fov*scale
    'target_lost_k': 5,                    # 连续K个高层步超出hard窗口 -> 判丢失并reset

    # 门洞参数
    'gate_width_range': (0.85, 1.0),       # m
    'gate_width_curriculum': [1.0, 0.95, 0.90, 0.85],  # 课程学习
}
```

## 1.4 β约束族参数

```python
BETA_PARAMS = {
    # β=0（激进）和β=1（保守）的端点值
    'safe_dist': (0.35, 1.00),             # m
    'max_lin_vel': (1.00, 0.35),           # m/s
    'max_ang_vel': (1.50, 0.50),           # rad/s
    'max_delta_lin': (0.15, 0.05),         # m/s per high-level step
    'max_delta_ang': (0.30, 0.10),         # rad/s per high-level step
    'risk_clamp_gain': (1.0, 3.0),         # dimensionless
}

def beta_interpolate(beta, param_name):
    """线性插值β参数"""
    low, high = BETA_PARAMS[param_name]
    # 注意：对于max_vel类参数，β越大值越小，所以是high + β*(low-high)反向
    if param_name in ['max_lin_vel', 'max_ang_vel', 'max_delta_lin', 'max_delta_ang']:
        return high + (1 - beta) * (low - high)  # β↑ → 值↓
    else:
        return low + beta * (high - low)          # β↑ → 值↑
```

---

# 第二部分：数学公式体系

## 2.1 状态空间定义

### 2.1.1 底层观测（proprioception，67维）

$$o_{proprio} = [q_{base}, \omega_{base}, a_{base}, \theta_{joint} - \theta_{default}, \dot{\theta}_{joint}, \tau_{joint}, cmd]$$

| 分量                                  | 维度  | 说明                            |
| ----------------------------------- | --- | ----------------------------- |
| $q_{base}$                          | 4   | 基座四元数 [x,y,z,w]               |
| $\omega_{base}$                     | 3   | 基座角速度                         |
| $a_{base}$                          | 3   | 基座线加速度                        |
| $\theta_{joint} - \theta_{default}$ | 18  | 关节位置偏差                        |
| $\dot{\theta}_{joint}$              | 18  | 关节速度                          |
| $\tau_{joint}$                      | 18  | 关节力矩                          |
| $cmd$                               | 3   | 当前速度指令 $(v_x, v_y, \omega_z)$ |

### 2.1.2 高层观测

**机器人状态（robot_state，9维）**：
$$s_{robot} = [x, y, \psi, v_x, v_y, \omega_z, h, \phi, \theta]$$

| 分量             | 说明     | 坐标系    |
| -------------- | ------ | ------ |
| $x, y$         | 位置     | 环境局部坐标 |
| $\psi$         | 偏航角    | 世界系    |
| $v_x, v_y$     | 线速度    | 机体系    |
| $\omega_z$     | 角速度    | 机体系    |
| $h$            | 相对地面高度 |        |
| $\phi, \theta$ | 横滚、俯仰角 |        |

**目标状态（target_state，6维）**：
$$s_{target} = [p_x^{rel}, p_y^{rel}, d, \dot{d}, v_x^{target}, v_y^{target}]$$

| 分量                           | 说明    | 坐标系 |
| ---------------------------- | ----- | --- |
| $p_x^{rel}, p_y^{rel}$       | 相对位置  | 机体系 |
| $d$                          | 相对距离  |     |
| $\dot{d}$                    | 距离变化率 |     |
| $v_x^{target}, v_y^{target}$ | 目标速度  | 世界系 |

**Affordance（来自GT或估计，35维）**：
$$a = [O_{16\times16}^{flat}, G_{16\times16}^{flat}, L_{16\times16}^{flat}, d_{difficulty}]$$

| 分量  | 维度     | 说明               |
| --- | ------ | ---------------- |
| $O$ | 256→压缩 | 占据栅格（16×16展平后降维） |
| $G$ | 256→压缩 | 可通行间隙            |
| $L$ | 256→压缩 | 低矮障碍             |
| $d$ | 1      | 地形难度             |

> 实际实现中，16×16的affordance map通过卷积或池化压缩到合理维度（如32维）

### 2.1.3 候选命令特征（命令条件化输入，关键！）

$$f_{cmd} = [cmd_F, cmd_A, \Delta cmd, risk_F, risk_A, \Delta risk]$$

| 分量            | 维度  | 说明                                      |
| ------------- | --- | --------------------------------------- |
| $cmd_F$       | 3   | Follow专家输出 $(v_x^F, v_y^F, \omega_z^F)$ |
| $cmd_A$       | 3   | Avoid专家输出 $(v_x^A, v_y^A, \omega_z^A)$  |
| $\Delta cmd$  | 3   | 两专家命令差 $cmd_F - cmd_A$                  |
| $risk_F$      | 1   | 沿Follow方向的风险                            |
| $risk_A$      | 1   | 沿Avoid方向的风险                             |
| $\Delta risk$ | 1   | 风险差 $risk_F - risk_A$                   |

**RiskAlong计算公式**：

$$risk_F = \text{RiskAlong}(cmd_F) = \max_{i \in [1,K]} O(p_{robot} + i \cdot \Delta t \cdot cmd_F[:2])$$

其中：

- $K = 10$：前瞻步数
- $\Delta t = 0.1s$：时间步长
- $O(\cdot)$：占据栅格查询函数

## 2.2 PCR-Net++输出定义

### 2.2.1 y（门控权重）

$$y \in [0, 1]$$

- $y = 1$：完全Follow
- $y = 0$：完全Avoid

**语义**（全系统统一）：
$$cmd_{base} = y \cdot cmd_F + (1 - y) \cdot cmd_A$$

### 2.2.2 w（命令条件化预测冲突prior）

$$w \in [0, 1]$$

- $w \to 1$：预测未来冲突高，应提前偏向Avoid
- $w \to 0$：预测未来冲突低，可继续Follow

**w必须依赖于候选命令**（避免退化为clearance分类器）：
$$w = f_w(s_{robot}, s_{target}, a, cmd_F, cmd_A, risk_F, risk_A)$$

### 2.2.3 β（风险预算）

$$\beta \in [0, 1]$$

- $\beta = 0$：激进模式（效率优先）
- $\beta = 1$：保守模式（安全优先）

**β的语义**（全系统统一）：
$$\beta \uparrow \Rightarrow safe\_dist \uparrow, max\_vel \downarrow, max\_delta \downarrow$$

## 2.3 融合与后处理公式

### 2.3.1 有效门控y_eff

由于 $w$ 表示“冲突强度”（$w \uparrow \Rightarrow$ 更应偏向 Avoid），因此在将其融入 Follow 权重时需要取反：

$$y_{eff,raw} = \text{clamp}((1 - \lambda_w) \cdot y + \lambda_w \cdot (1 - w), 0, 1)$$

其中$\lambda_w = 0.4$（可调超参，w的影响权重）

### 2.3.2 时序平滑

$$y_{eff,smooth}(t) = \gamma \cdot y_{eff,smooth}(t-1) + (1-\gamma) \cdot y_{eff,raw}(t)$$

其中$\gamma = 0.7$（平滑系数）

### 2.3.3 滞回机制

$$y_{eff}(t) = \begin{cases} 
y_{eff,smooth}(t) & \text{if } |y_{eff,smooth}(t) - y_{eff}(t-1)| > \delta_{hyst} \\
y_{eff}(t-1) & \text{otherwise}
\end{cases}$$

其中$\delta_{hyst} = 0.08$（滞回阈值）

### 2.3.4 基础命令融合

$$cmd_{base} = y_{eff} \cdot cmd_F + (1 - y_{eff}) \cdot cmd_A$$

### 2.3.5 条件性融合（解决线性融合陷阱）

定义一致性：
$\text{consistency} = \frac{cmd_F[:2] \cdot cmd_A[:2]}{\|cmd_F[:2]\| \cdot \|cmd_A[:2]\| + \epsilon}$

融合规则：
$cmd_{raw} = \begin{cases}
cmd_{base} & \text{if consistency} > \tau_{consist} \\
cmd_A & \text{if consistency} \leq \tau_{consist} \text{ and } P_{col} > \tau_{danger} \\
\text{softmax\_blend}(cmd_F, cmd_A, y_{eff}) & \text{otherwise}
\end{cases}$

其中：

- $\tau_{consist} = 0.3$（一致性阈值）
- $\tau_{danger} = 0.7$（危险阈值）

Softmax融合：
$\text{softmax\_blend} = \frac{e^{y_{eff}/\tau}}{e^{y_{eff}/\tau} + e^{(1-y_{eff})/\tau}} \cdot cmd_F + \frac{e^{(1-y_{eff})/\tau}}{e^{y_{eff}/\tau} + e^{(1-y_{eff})/\tau}} \cdot cmd_A$

其中$\tau = 0.3$（温度参数）

## 2.4 Post-Processor公式

### 2.4.1 β参数映射

对于参数$\theta$，映射函数为：
$\theta(\beta) = \theta_{min} + \beta \cdot (\theta_{max} - \theta_{min})$

具体映射（注意方向）：

| 参数                          | 公式                 | β=0            | β=1            |
| --------------------------- | ------------------ | -------------- | -------------- |
| $d_{safe}(\beta)$           | $0.35 + 0.65\beta$ | 0.35m          | 1.00m          |
| $v_{max}(\beta)$            | $1.00 - 0.65\beta$ | 1.00m/s        | 0.35m/s        |
| $\omega_{max}(\beta)$       | $1.50 - 1.00\beta$ | 1.50rad/s      | 0.50rad/s      |
| $\Delta v_{max}(\beta)$     | $0.15 - 0.10\beta$ | 0.15m/s/step   | 0.05m/s/step   |
| $\Delta\omega_{max}(\beta)$ | $0.30 - 0.20\beta$ | 0.30rad/s/step | 0.10rad/s/step |
| $k_{risk}(\beta)$           | $1.0 + 2.0\beta$   | 1.0            | 3.0            |

### 2.4.2 速度限幅

$cmd_{clamp} = \text{clip}(cmd_{raw}, -[v_{max}, v_{max}, \omega_{max}], [v_{max}, v_{max}, \omega_{max}])$

### 2.4.3 变化率限制（Slew Rate Limiting）

$\Delta cmd = cmd_{clamp} - cmd_{prev}$

$\Delta cmd_{limited} = \text{clip}(\Delta cmd, -[\Delta v_{max}, \Delta v_{max}, \Delta\omega_{max}], [\Delta v_{max}, \Delta v_{max}, \Delta\omega_{max}])$

$cmd_{smooth} = cmd_{prev} + \Delta cmd_{limited}$

### 2.4.4 风险钳制

当$clearance < d_{safe}(\beta)$时：

$scale = \left(\frac{clearance}{d_{safe}(\beta)}\right)^{k_{risk}(\beta)}$

$cmd_{safe} = cmd_{smooth} \cdot scale$

### 2.4.5 视野保持约束（六足全向特性）

$\theta_{error} = \text{atan2}(p_y^{rel}, p_x^{rel}) - 0$

$\omega_{correction} = K_p \cdot \theta_{error}$

$cmd_{final} = (cmd_{safe}[0], cmd_{safe}[1], cmd_{safe}[2] + \omega_{correction})$

其中$K_p = 0.5$（视野保持增益）

---

# 第三部分：网络架构与代码实现

## 3.1 Follow专家网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FollowExpert(nn.Module):
    """
    Follow专家：优化跟随效率

    输入：robot_state(9) + target_state(6) + affordance(32) + beta(1) = 48
    输出：cmd_vel(3) = (vx, vy, omega_z)
    """

    def __init__(self, 
                 robot_state_dim=9,
                 target_state_dim=6,
                 affordance_dim=32,
                 hidden_dim=128):
        super().__init__()

        input_dim = robot_state_dim + target_state_dim + affordance_dim + 1  # +1 for beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.cmd_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
            nn.Tanh(),  # 输出范围[-1, 1]，后续乘以max_cmd
        )

        # 输出缩放
        self.cmd_scale = nn.Parameter(torch.tensor([1.0, 1.5, 2.0]), requires_grad=False)

    def forward(self, robot_state, target_state, affordance, beta):
        """
        Args:
            robot_state: (B, 9)
            target_state: (B, 6)
            affordance: (B, 32)
            beta: (B, 1)
        Returns:
            cmd: (B, 3) - (vx, vy, omega_z)
        """
        x = torch.cat([robot_state, target_state, affordance, beta], dim=-1)
        features = self.encoder(x)
        cmd_normalized = self.cmd_head(features)
        cmd = cmd_normalized * self.cmd_scale
        return cmd
```

## 3.2 Avoid专家网络

```python
class AvoidExpert(nn.Module):
    """
    Avoid专家：优化避障安全

    输入：robot_state(9) + affordance(32) + waypoint(2) + beta(1) = 44
    输出：cmd_vel(3) = (vx, vy, omega_z)
    """

    def __init__(self,
                 robot_state_dim=9,
                 affordance_dim=32,
                 waypoint_dim=2,
                 hidden_dim=128):
        super().__init__()

        input_dim = robot_state_dim + affordance_dim + waypoint_dim + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.cmd_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

        self.cmd_scale = nn.Parameter(torch.tensor([1.0, 1.5, 2.0]), requires_grad=False)

    def forward(self, robot_state, affordance, waypoint, beta):
        """
        Args:
            robot_state: (B, 9)
            affordance: (B, 32)
            waypoint: (B, 2) - 局部导航点方向
            beta: (B, 1)
        Returns:
            cmd: (B, 3)
        """
        x = torch.cat([robot_state, affordance, waypoint, beta], dim=-1)
        features = self.encoder(x)
        cmd_normalized = self.cmd_head(features)
        cmd = cmd_normalized * self.cmd_scale
        return cmd
```

## 3.3 PCR-Net++仲裁网络

```python
class PCRNetPlusPlus(nn.Module):
    """
    PCR-Net++: 预测式冲突消解网络

    输出：y(1) + w(1) + beta(1) = 3

    关键设计：
    1. w必须是命令条件化的（输入包含cmd_F, cmd_A, risk_F, risk_A）
    2. beta联动Post-Processor
    3. 平滑机制在forward中实现
    """

    def __init__(self,
                 robot_state_dim=9,
                 target_state_dim=6,
                 affordance_dim=32,
                 cmd_dim=3,
                 history_dim=16,
                 hidden_dim=256):
        super().__init__()

        # 状态编码器（不含命令信息）
        state_input_dim = robot_state_dim + target_state_dim + affordance_dim + history_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 128),
            nn.ELU(),
        )

        # 命令条件化编码器（关键！用于w）
        cmd_input_dim = cmd_dim * 2 + 3 + 2  # cmd_F(3) + cmd_A(3) + delta_cmd(3) + risk_F(1) + risk_A(1)
        self.cmd_encoder = nn.Sequential(
            nn.Linear(cmd_input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # y头（基于状态）
        self.y_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # w头（基于状态+命令，命令条件化）
        self.w_head = nn.Sequential(
            nn.Linear(128 + 32, 64),  # 状态特征 + 命令特征
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # beta头（基于状态）
        self.beta_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # 平滑相关的buffer（用于推理时的时序平滑）
        self.register_buffer('y_eff_prev', torch.zeros(1))
        self.register_buffer('cmd_prev', torch.zeros(3))

        # 超参数
        self.lambda_w = 0.4      # w的影响权重
        self.gamma = 0.7         # 时序平滑系数
        self.delta_hyst = 0.08   # 滞回阈值

    def forward(self, robot_state, target_state, affordance, history_features,
                cmd_F, cmd_A, risk_F, risk_A, apply_smoothing=True):
        """
        Args:
            robot_state: (B, 9)
            target_state: (B, 6)
            affordance: (B, 32)
            history_features: (B, 16) - 短历史统计特征
            cmd_F: (B, 3) - Follow专家输出
            cmd_A: (B, 3) - Avoid专家输出
            risk_F: (B, 1) - 沿Follow方向的风险
            risk_A: (B, 1) - 沿Avoid方向的风险
            apply_smoothing: bool - 是否应用时序平滑（训练时False，推理时True）

        Returns:
            y: (B, 1) - 门控权重
            w: (B, 1) - 预测冲突prior
            beta: (B, 1) - 风险预算
            y_eff: (B, 1) - 有效门控（融合y和w，可选平滑）
        """
        B = robot_state.shape[0]

        # 状态编码
        state_input = torch.cat([robot_state, target_state, affordance, history_features], dim=-1)
        state_features = self.state_encoder(state_input)

        # 命令条件化编码
        delta_cmd = cmd_F - cmd_A
        cmd_input = torch.cat([cmd_F, cmd_A, delta_cmd, risk_F, risk_A], dim=-1)
        cmd_features = self.cmd_encoder(cmd_input)

        # y输出（纯状态）
        y = self.y_head(state_features)

        # w输出（状态+命令条件化，关键！）
        w_input = torch.cat([state_features, cmd_features], dim=-1)
        w = self.w_head(w_input)

        # beta输出（纯状态）
        beta = self.beta_head(state_features)

        # 计算y_eff
        # w 表示冲突强度：w↑ => 更应偏 Avoid，因此融入 Follow 权重时取 (1-w)
        y_eff_raw = torch.clamp((1 - self.lambda_w) * y + self.lambda_w * (1.0 - w), 0, 1)

        if apply_smoothing and not self.training:
            # 时序平滑（仅推理时）
            y_eff_smooth = self.gamma * self.y_eff_prev + (1 - self.gamma) * y_eff_raw

            # 滞回
            change = torch.abs(y_eff_smooth - self.y_eff_prev)
            y_eff = torch.where(change > self.delta_hyst, y_eff_smooth, self.y_eff_prev.expand_as(y_eff_smooth))

            # 更新buffer
            self.y_eff_prev = y_eff.mean().detach()  # 简化处理，实际应保持batch维度
        else:
            y_eff = y_eff_raw

        return y, w, beta, y_eff

    def compute_cmd_base(self, cmd_F, cmd_A, y_eff):
        """计算基础融合命令"""
        return y_eff * cmd_F + (1 - y_eff) * cmd_A

    def compute_conditional_fusion(self, cmd_F, cmd_A, y_eff, P_collision,
                                   tau_consist=0.3, tau_danger=0.7, tau_temp=0.3):
        """
        条件性融合（解决线性融合陷阱）
        """
        # 计算一致性
        cmd_F_2d = cmd_F[:, :2]
        cmd_A_2d = cmd_A[:, :2]

        dot_product = (cmd_F_2d * cmd_A_2d).sum(dim=-1, keepdim=True)
        norm_F = torch.norm(cmd_F_2d, dim=-1, keepdim=True) + 1e-6
        norm_A = torch.norm(cmd_A_2d, dim=-1, keepdim=True) + 1e-6
        consistency = dot_product / (norm_F * norm_A)

        # 基础线性融合
        cmd_base = self.compute_cmd_base(cmd_F, cmd_A, y_eff)

        # Softmax融合
        # 约定：y_eff 是 Follow 权重，因此 logits[0]=Follow, logits[1]=Avoid
        logits = torch.cat([y_eff, 1 - y_eff], dim=-1) / tau_temp
        weights = F.softmax(logits, dim=-1)
        cmd_softmax = weights[:, 0:1] * cmd_F + weights[:, 1:2] * cmd_A

        # 条件选择
        # Case 1: 一致性高 → 线性融合
        # Case 2: 一致性低 + 高危险 → 纯Avoid
        # Case 3: 其他 → Softmax融合

        high_consistency = consistency > tau_consist
        high_danger = P_collision > tau_danger

        cmd_raw = torch.where(
            high_consistency,
            cmd_base,
            torch.where(
                high_danger,
                cmd_A,
                cmd_softmax
            )
        )

        return cmd_raw
```

## 3.4 历史特征提取器

```python
class HistoryFeatureExtractor(nn.Module):
    """
    提取短历史统计特征，用于PCR-Net++的输入

    不使用LSTM，而是简单的统计量（均值、方差、趋势）
    """

    def __init__(self, history_length=10, output_dim=16):
        super().__init__()

        self.history_length = history_length
        self.output_dim = output_dim

        # 历史buffer
        self.register_buffer('clearance_history', torch.zeros(history_length))
        self.register_buffer('cmd_history', torch.zeros(history_length, 3))
        self.register_buffer('progress_history', torch.zeros(history_length))
        self.register_buffer('y_eff_history', torch.zeros(history_length))
        self.register_buffer('ptr', torch.tensor(0))

        # 特征压缩
        self.feature_proj = nn.Linear(20, output_dim)  # 20个原始统计量

    def update(self, clearance, cmd, progress, y_eff):
        """更新历史buffer"""
        idx = self.ptr.item() % self.history_length
        self.clearance_history[idx] = clearance
        self.cmd_history[idx] = cmd
        self.progress_history[idx] = progress
        self.y_eff_history[idx] = y_eff
        self.ptr += 1

    def extract_features(self):
        """提取统计特征"""
        features = []

        # Clearance统计
        features.append(self.clearance_history.mean())
        features.append(self.clearance_history.std())
        features.append(self.clearance_history[-1] - self.clearance_history[0])  # 趋势
        features.append(self.clearance_history.min())

        # Cmd统计
        for i in range(3):  # vx, vy, omega
            features.append(self.cmd_history[:, i].mean())
            features.append(self.cmd_history[:, i].std())

        # Progress统计
        features.append(self.progress_history.mean())
        features.append(self.progress_history.sum())  # 累计进度
        features.append(self.progress_history[-1] - self.progress_history[0])  # 趋势

        # y_eff统计
        features.append(self.y_eff_history.mean())
        features.append(self.y_eff_history.std())
        features.append((torch.abs(self.y_eff_history[1:] - self.y_eff_history[:-1]) > 0.1).float().sum())  # 切换次数

        raw_features = torch.stack(features)
        return self.feature_proj(raw_features.unsqueeze(0))
```

## 3.5 RiskAlong计算

```python
def compute_risk_along(cmd, occupancy_grid, robot_pos, grid_resolution=0.1, 
                       lookahead_steps=10, dt=0.1):
    """
    计算沿命令方向的风险

    Args:
        cmd: (3,) - (vx, vy, omega_z)
        occupancy_grid: (16, 16) - 占据栅格，以机器人为中心
        robot_pos: (2,) - 机器人位置（栅格中心）
        grid_resolution: float - 栅格分辨率 (m/cell)
        lookahead_steps: int - 前瞻步数
        dt: float - 时间步长

    Returns:
        risk: float in [0, 1]
    """
    # 速度方向（2D）
    vel_2d = cmd[:2]
    vel_norm = torch.norm(vel_2d)

    if vel_norm < 0.01:
        # 速度几乎为0，返回当前位置的风险
        center_idx = occupancy_grid.shape[0] // 2
        return occupancy_grid[center_idx, center_idx]

    # 归一化方向
    direction = vel_2d / vel_norm

    # 沿方向采样
    max_risk = 0.0
    for i in range(1, lookahead_steps + 1):
        # 预测位置（相对于机器人当前位置）
        delta_pos = direction * vel_norm * dt * i

        # 转换为栅格坐标
        grid_x = int(robot_pos[0] + delta_pos[0] / grid_resolution)
        grid_y = int(robot_pos[1] + delta_pos[1] / grid_resolution)

        # 边界检查
        if 0 <= grid_x < occupancy_grid.shape[0] and 0 <= grid_y < occupancy_grid.shape[1]:
            risk = occupancy_grid[grid_x, grid_y]
            max_risk = max(max_risk, risk.item())
        else:
            # 超出栅格范围，假设未知区域有一定风险
            max_risk = max(max_risk, 0.5)

    return max_risk
```

> 实现注意：上面的 for-loop 代码仅用于说明 RiskAlong 的语义；实际训练实现必须向量化/GPU 友好（4096 并行不允许 per-env Python 循环）。

---

## 3.6 多维冲突先验 w（正文：可解释增强）

> 目的：把“冲突”拆成互补的三个侧面，避免 w 退化为单一 clearance 分类器，并提供论文级可解释性图（w 分量与融合权重随场景变化）。

### 3.6.1 三个维度（几何分解，不是三种方式算同一个 risk）

- **w_spatial（空间拥挤/通行性）**：这个方向“挤不挤、通不通？”
  - 主要依赖：沿候选命令方向的占用/可通行性（affordance/occupancy），单帧即可。
- **w_proximity（接近紧迫度）**：按当前速度，“多快会到危险区域？”
  - 主要依赖：当前距离与当前速度的 TTC proxy（不做动态障碍预测），单帧即可。
- **w_severity（后果严重性）**：如果发生冲突，“撞上有多危险？”
  - 主要依赖：速度、距离、相对角度/正面程度等，单帧即可。

> 命名说明：这里使用 **proximity**，而不是 temporal，避免被误解为“必须做时序/轨迹预测”。

### 3.6.2 单帧输入优势

- 输入只需当前帧：affordance/occupancy + proprioception（含速度）+ target_state + cmd_F/cmd_A（候选命令条件化）。
- 不需要：历史帧、光流、LSTM/RNN（推理更快、训练更稳、部署更可靠）。

### 3.6.3 融合为标量 w（保持接口）

设融合权重 `λ = softmax(g(features))`，则：

`w = λ_s*w_spatial + λ_p*w_proximity + λ_v*w_severity`

- **baseline**：先用简单平均（λ 固定为 1/3），确保稳定增益。
- **enhanced**：再启用学习融合（λ 随场景自适应），用于可解释性展示与消融。

---

## 3.7 w 的预测性辅助监督（推荐：让 w 变“事前预测”）

> 关键：w 评估的是“如果执行候选命令（尤其是 cmd_F）会怎样”，而不是“实际执行轨迹怎样”。实际执行轨迹的风险惩罚由 r_risk 负责（见 5.4）。

### 3.7.1 伪标签生成：两种方案

- **方案A（推荐）：轨迹推演 / RiskAlong-lite（命令条件化 + horizon）**
  - 输入：当前 affordance/occupancy + 候选命令 `cmd_F`
  - 输出：在 horizon `H=0.8s` 内沿命令方向的最小 clearance（或等价风险）
  - 标签：`conflict_F = 1{ min_clearance_F(H) < safe_dist(beta) }`
  - 实现要求：必须 **向量化/GPU 友好**（禁止 per-env Python 循环）。
- **方案B（备选）：未来K步真实事件**
  - 标签：未来 K 步内是否 near-miss/collision
  - 缺点：会混入“当时到底执行了 Follow 还是 Avoid”的污染，不如方案A干净。

### 3.7.2 辅助损失

`L_w_aux = BCE(w, conflict_F)`，总损失：

`L_total = L_PPO + λ_w_aux * L_w_aux + ...`

并记录诊断：`conflict_rate`、`w` 提前激活时序图（w vs near-miss/collision 事件对齐）。

# 第四部分：Post-Processor实现

## 4.1 完整Post-Processor类

```python
class CommandPostProcessor:
    """
    命令后处理器

    功能：
    1. β参数映射
    2. 速度限幅
    3. 变化率限制（Slew Rate）
    4. 风险钳制
    5. 视野保持约束

    运行频率：50Hz（与底层同步）
    """

    def __init__(self, high_level_dt=0.1, low_level_dt=0.02):
        self.high_level_dt = high_level_dt
        self.low_level_dt = low_level_dt
        self.steps_per_high_level = int(high_level_dt / low_level_dt)  # 5

        # β参数范围
        self.beta_params = {
            'safe_dist': (0.35, 1.00),
            'max_lin_vel': (1.00, 0.35),
            'max_ang_vel': (1.50, 0.50),
            'max_delta_lin': (0.15, 0.05),  # 每高层step的变化
            'max_delta_ang': (0.30, 0.10),
            'risk_clamp_gain': (1.0, 3.0),
        }

        # 状态
        self.cmd_prev = torch.zeros(3)
        self.current_step = 0
        self.current_cmd_target = torch.zeros(3)  # 高层目标命令

        # 视野保持参数
        self.gaze_Kp = 0.5

    def beta_map(self, beta, param_name):
        """β参数映射"""
        low, high = self.beta_params[param_name]

        if param_name in ['max_lin_vel', 'max_ang_vel', 'max_delta_lin', 'max_delta_ang']:
            # β↑ → 值↓
            return low - beta * (low - high)
        else:
            # β↑ → 值↑
            return low + beta * (high - low)

    def process(self, cmd_raw, beta, clearance, target_rel_pos, 
                is_new_high_level_cmd=False):
        """
        处理命令

        Args:
            cmd_raw: (3,) - 原始命令 (vx, vy, omega_z)
            beta: float - 风险预算
            clearance: float - 当前最小障碍距离
            target_rel_pos: (2,) - 目标相对位置 (用于视野保持)
            is_new_high_level_cmd: bool - 是否是新的高层命令

        Returns:
            cmd_final: (3,) - 处理后的命令
        """
        beta = float(beta)

        # 获取β映射后的参数
        safe_dist = self.beta_map(beta, 'safe_dist')
        max_lin_vel = self.beta_map(beta, 'max_lin_vel')
        max_ang_vel = self.beta_map(beta, 'max_ang_vel')
        max_delta_lin_high = self.beta_map(beta, 'max_delta_lin')
        max_delta_ang_high = self.beta_map(beta, 'max_delta_ang')
        risk_clamp_gain = self.beta_map(beta, 'risk_clamp_gain')

        # 转换为低层step的变化率限制
        max_delta_lin = max_delta_lin_high / self.steps_per_high_level
        max_delta_ang = max_delta_ang_high / self.steps_per_high_level

        # 如果是新的高层命令，更新目标
        if is_new_high_level_cmd:
            self.current_cmd_target = cmd_raw.clone()
            self.current_step = 0

        # Step 1: 速度限幅
        cmd_clamp = torch.zeros(3)
        cmd_clamp[0] = torch.clamp(cmd_raw[0], -max_lin_vel, max_lin_vel)
        cmd_clamp[1] = torch.clamp(cmd_raw[1], -max_lin_vel, max_lin_vel)
        cmd_clamp[2] = torch.clamp(cmd_raw[2], -max_ang_vel, max_ang_vel)

        # Step 2: 变化率限制（Slew Rate Limiting）
        delta_cmd = cmd_clamp - self.cmd_prev
        delta_cmd_limited = torch.zeros(3)
        delta_cmd_limited[0] = torch.clamp(delta_cmd[0], -max_delta_lin, max_delta_lin)
        delta_cmd_limited[1] = torch.clamp(delta_cmd[1], -max_delta_lin, max_delta_lin)
        delta_cmd_limited[2] = torch.clamp(delta_cmd[2], -max_delta_ang, max_delta_ang)

        cmd_smooth = self.cmd_prev + delta_cmd_limited

        # Step 3: 风险钳制
        if clearance < safe_dist:
            scale = (clearance / safe_dist) ** risk_clamp_gain
            cmd_safe = cmd_smooth * scale
        else:
            cmd_safe = cmd_smooth

        # Step 4: 视野保持约束
        if torch.norm(target_rel_pos) > 0.1:  # 目标有效
            theta_error = torch.atan2(target_rel_pos[1], target_rel_pos[0])
            omega_correction = self.gaze_Kp * theta_error
            cmd_final = torch.tensor([
                cmd_safe[0],
                cmd_safe[1],
                cmd_safe[2] + omega_correction
            ])
            # 再次限幅角速度
            cmd_final[2] = torch.clamp(cmd_final[2], -max_ang_vel, max_ang_vel)
        else:
            cmd_final = cmd_safe

        # 更新状态
        self.cmd_prev = cmd_final.clone()
        self.current_step += 1

        return cmd_final

    def reset(self):
        """重置状态"""
        self.cmd_prev = torch.zeros(3)
        self.current_step = 0
        self.current_cmd_target = torch.zeros(3)
```

## 4.2 批量处理版本（用于训练）

```python
class BatchCommandPostProcessor:
    """
    批量命令后处理器（用于Isaac Gym并行训练）
    """

    def __init__(self, num_envs, device, high_level_dt=0.1, low_level_dt=0.02):
        self.num_envs = num_envs
        self.device = device
        self.high_level_dt = high_level_dt
        self.low_level_dt = low_level_dt
        self.steps_per_high_level = int(high_level_dt / low_level_dt)

        # β参数范围
        self.safe_dist_range = torch.tensor([0.35, 1.00], device=device)
        self.max_lin_vel_range = torch.tensor([1.00, 0.35], device=device)
        self.max_ang_vel_range = torch.tensor([1.50, 0.50], device=device)
        self.max_delta_lin_range = torch.tensor([0.15, 0.05], device=device)
        self.max_delta_ang_range = torch.tensor([0.30, 0.10], device=device)
        self.risk_clamp_gain_range = torch.tensor([1.0, 3.0], device=device)

        # 状态buffer
        self.cmd_prev = torch.zeros(num_envs, 3, device=device)

        # 视野保持
        self.gaze_Kp = 0.5

    def beta_map_batch(self, beta):
        """
        批量β参数映射

        Args:
            beta: (N, 1)

        Returns:
            dict of (N, 1) tensors
        """
        beta = beta.squeeze(-1)  # (N,)

        params = {}
        # β↑ → 值↓
        params['max_lin_vel'] = self.max_lin_vel_range[0] - beta * (self.max_lin_vel_range[0] - self.max_lin_vel_range[1])
        params['max_ang_vel'] = self.max_ang_vel_range[0] - beta * (self.max_ang_vel_range[0] - self.max_ang_vel_range[1])
        params['max_delta_lin'] = (self.max_delta_lin_range[0] - beta * (self.max_delta_lin_range[0] - self.max_delta_lin_range[1])) / self.steps_per_high_level
        params['max_delta_ang'] = (self.max_delta_ang_range[0] - beta * (self.max_delta_ang_range[0] - self.max_delta_ang_range[1])) / self.steps_per_high_level

        # β↑ → 值↑
        params['safe_dist'] = self.safe_dist_range[0] + beta * (self.safe_dist_range[1] - self.safe_dist_range[0])
        params['risk_clamp_gain'] = self.risk_clamp_gain_range[0] + beta * (self.risk_clamp_gain_range[1] - self.risk_clamp_gain_range[0])

        return params

    def process_batch(self, cmd_raw, beta, clearance, target_rel_pos):
        """
        批量处理命令

        Args:
            cmd_raw: (N, 3)
            beta: (N, 1)
            clearance: (N, 1)
            target_rel_pos: (N, 2)

        Returns:
            cmd_final: (N, 3)
        """
        N = cmd_raw.shape[0]
        params = self.beta_map_batch(beta)

        # Step 1: 速度限幅
        max_lin = params['max_lin_vel'].unsqueeze(-1)  # (N, 1)
        max_ang = params['max_ang_vel'].unsqueeze(-1)

        cmd_clamp = torch.zeros_like(cmd_raw)
        cmd_clamp[:, 0] = torch.clamp(cmd_raw[:, 0], -max_lin.squeeze(), max_lin.squeeze())
        cmd_clamp[:, 1] = torch.clamp(cmd_raw[:, 1], -max_lin.squeeze(), max_lin.squeeze())
        cmd_clamp[:, 2] = torch.clamp(cmd_raw[:, 2], -max_ang.squeeze(), max_ang.squeeze())

        # Step 2: 变化率限制
        delta_cmd = cmd_clamp - self.cmd_prev
        max_d_lin = params['max_delta_lin'].unsqueeze(-1)
        max_d_ang = params['max_delta_ang'].unsqueeze(-1)

        delta_cmd_limited = torch.zeros_like(delta_cmd)
        delta_cmd_limited[:, 0] = torch.clamp(delta_cmd[:, 0], -max_d_lin.squeeze(), max_d_lin.squeeze())
        delta_cmd_limited[:, 1] = torch.clamp(delta_cmd[:, 1], -max_d_lin.squeeze(), max_d_lin.squeeze())
        delta_cmd_limited[:, 2] = torch.clamp(delta_cmd[:, 2], -max_d_ang.squeeze(), max_d_ang.squeeze())

        cmd_smooth = self.cmd_prev + delta_cmd_limited

        # Step 3: 风险钳制
        safe_dist = params['safe_dist'].unsqueeze(-1)  # (N, 1)
        risk_gain = params['risk_clamp_gain'].unsqueeze(-1)

        scale = torch.where(
            clearance < safe_dist,
            (clearance / safe_dist) ** risk_gain,
            torch.ones_like(clearance)
        )
        cmd_safe = cmd_smooth * scale

        # Step 4: 视野保持
        target_norm = torch.norm(target_rel_pos, dim=-1, keepdim=True)
        theta_error = torch.atan2(target_rel_pos[:, 1], target_rel_pos[:, 0])
        omega_correction = self.gaze_Kp * theta_error

        cmd_final = cmd_safe.clone()
        valid_target = (target_norm.squeeze() > 0.1)
        cmd_final[valid_target, 2] = cmd_safe[valid_target, 2] + omega_correction[valid_target]

        # 再次限幅
        cmd_final[:, 2] = torch.clamp(cmd_final[:, 2], -max_ang.squeeze(), max_ang.squeeze())

        # 更新状态
        self.cmd_prev = cmd_final.clone()

        return cmd_final

    def reset(self, env_ids=None):
        """重置指定环境的状态"""
        if env_ids is None:
            self.cmd_prev.zero_()
        else:
            self.cmd_prev[env_ids] = 0
```

---

# 第五部分：奖励函数设计

## 5.1 奖励函数总体架构

```python
class RewardManager:
    """
    奖励函数管理器

    原则：
    1. 底层学"稳"，高层学"何时跟随、何时避障"
    2. 后处理负责硬约束，网络只输出可解释控制量
    3. 避免reward hacking
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # 奖励权重（按阶段不同）
        self.weights = {}

    def set_stage(self, stage):
        """设置训练阶段，调整权重"""
        if stage == 'follow_expert':
            self.weights = {
                'progress': 2.0,
                'distance_keeping': 1.5,
                'heading_alignment': 0.5,
                'time_penalty': -0.01,
                'collision': -10.0,
                'stability': 0.3,
                'cmd_smoothness': 0.2,
            }
        elif stage == 'avoid_expert':
            self.weights = {
                'progress': 1.0,
                'collision': -20.0,
                'near_miss': -2.0,
                'risk_barrier': -1.0,
                'clearance_bonus': 0.5,
                'stability': 0.3,
                'cmd_smoothness': 0.3,
            }
        elif stage == 'pcr_net':
            self.weights = {
                'progress': 1.5,
                'distance_keeping': 1.0,
                'collision': -15.0,
                'near_miss': -1.5,
                'risk_barrier': -0.8,
                'switch_penalty': -0.3,
                'y_eff_smoothness': -0.2,
                'beta_smoothness': -0.1,
                'cmd_smoothness': 0.3,
                'time_penalty': -0.005,
            }
```

## 5.2 Follow专家奖励函数

```python
class FollowExpertReward:
    """Follow专家的奖励函数"""

    def __init__(self, cfg):
        self.desired_distance = cfg.get('desired_distance', 1.5)
        self.distance_tolerance = cfg.get('distance_tolerance', 0.3)
        self.max_speed = cfg.get('max_speed', 1.0)

    def compute(self, env_state):
        """
        计算Follow专家奖励

        Args:
            env_state: dict containing:
                - target_rel_pos: (N, 2)
                - target_rel_vel: (N, 2)
                - robot_vel: (N, 3)
                - collision: (N,)
                - cmd: (N, 3)
                - cmd_prev: (N, 3)
                - dt: float
        """
        rewards = {}

        # 1. 进度奖励（朝向目标移动）
        target_rel_pos = env_state['target_rel_pos']
        target_distance = torch.norm(target_rel_pos, dim=-1)
        target_direction = target_rel_pos / (target_distance.unsqueeze(-1) + 1e-6)

        robot_vel_2d = env_state['robot_vel'][:, :2]
        progress = (robot_vel_2d * target_direction).sum(dim=-1)
        rewards['progress'] = torch.clamp(progress, -self.max_speed, self.max_speed)

        # 2. 距离保持奖励（保持在期望距离）
        distance_error = torch.abs(target_distance - self.desired_distance)
        rewards['distance_keeping'] = torch.exp(-distance_error / self.distance_tolerance)

        # 3. 朝向对齐奖励（面向目标）
        heading_to_target = torch.atan2(target_rel_pos[:, 1], target_rel_pos[:, 0])
        rewards['heading_alignment'] = torch.cos(heading_to_target)  # 正对目标时为1

        # 4. 碰撞惩罚
        rewards['collision'] = -env_state['collision'].float()

        # 5. 时间惩罚（鼓励高效）
        rewards['time_penalty'] = -torch.ones_like(target_distance) * env_state['dt']

        # 6. 稳定性奖励（姿态稳定）
        roll = env_state.get('roll', torch.zeros_like(target_distance))
        pitch = env_state.get('pitch', torch.zeros_like(target_distance))
        rewards['stability'] = torch.exp(-(roll**2 + pitch**2) / 0.1)

        # 7. 命令平滑度（惩罚急变）
        cmd = env_state['cmd']
        cmd_prev = env_state['cmd_prev']
        cmd_jerk = torch.norm(cmd - cmd_prev, dim=-1)
        rewards['cmd_smoothness'] = torch.exp(-cmd_jerk / 0.5)

        return rewards
```

## 5.3 Avoid专家奖励函数

```python
class AvoidExpertReward:
    """Avoid专家的奖励函数"""

    def __init__(self, cfg):
        self.safe_distance = cfg.get('safe_distance', 0.5)
        self.near_miss_threshold = cfg.get('near_miss_threshold', 0.3)
        self.risk_barrier_scale = cfg.get('risk_barrier_scale', 2.0)

    def compute(self, env_state):
        """
        计算Avoid专家奖励
        """
        rewards = {}

        # 1. 进度奖励（朝向waypoint移动）
        waypoint_rel = env_state['waypoint_rel']
        waypoint_distance = torch.norm(waypoint_rel, dim=-1)
        waypoint_direction = waypoint_rel / (waypoint_distance.unsqueeze(-1) + 1e-6)

        robot_vel_2d = env_state['robot_vel'][:, :2]
        progress = (robot_vel_2d * waypoint_direction).sum(dim=-1)
        rewards['progress'] = torch.clamp(progress, -1.0, 1.0)

        # 2. 碰撞惩罚（最重要）
        rewards['collision'] = -env_state['collision'].float()

        # 3. Near-miss惩罚（低clearance积分）
        clearance = env_state['clearance']
        near_miss = torch.clamp(self.near_miss_threshold - clearance, min=0)
        rewards['near_miss'] = -near_miss

        # 4. 风险屏障（连续风险成本）
        # risk_barrier = max(0, safe_dist - clearance) / safe_dist
        risk_barrier = torch.clamp(
            (self.safe_distance - clearance) / self.safe_distance, 
            min=0
        )
        rewards['risk_barrier'] = -risk_barrier * self.risk_barrier_scale

        # 5. Clearance奖励（鼓励保持距离）
        rewards['clearance_bonus'] = torch.clamp(clearance / self.safe_distance, max=1.0)

        # 6. 稳定性奖励
        roll = env_state.get('roll', torch.zeros_like(clearance))
        pitch = env_state.get('pitch', torch.zeros_like(clearance))
        rewards['stability'] = torch.exp(-(roll**2 + pitch**2) / 0.1)

        # 7. 命令平滑度
        cmd = env_state['cmd']
        cmd_prev = env_state['cmd_prev']
        cmd_jerk = torch.norm(cmd - cmd_prev, dim=-1)
        rewards['cmd_smoothness'] = torch.exp(-cmd_jerk / 0.5)

        return rewards
```

## 5.4 PCR-Net++仲裁网络奖励函数

```python
class PCRNetReward:
    """PCR-Net++仲裁网络的奖励函数"""

    def __init__(self, cfg):
        self.desired_distance = cfg.get('desired_distance', 1.5)
        self.safe_distance = cfg.get('safe_distance', 0.5)
        self.near_miss_threshold = cfg.get('near_miss_threshold', 0.3)
        self.switch_threshold = cfg.get('switch_threshold', 0.15)

    def compute(self, env_state):
        """
        计算PCR-Net++奖励

        关键：同时奖励跟随和避障，惩罚抖动
        """
        rewards = {}

        # ===== 任务奖励 =====

        # 1. 进度奖励（朝向目标移动）
        target_rel_pos = env_state['target_rel_pos']
        target_distance = torch.norm(target_rel_pos, dim=-1)
        target_direction = target_rel_pos / (target_distance.unsqueeze(-1) + 1e-6)

        robot_vel_2d = env_state['robot_vel'][:, :2]
        progress = (robot_vel_2d * target_direction).sum(dim=-1)
        rewards['progress'] = torch.clamp(progress, -1.0, 1.0)

        # 2. 距离保持奖励
        distance_error = torch.abs(target_distance - self.desired_distance)
        rewards['distance_keeping'] = torch.exp(-distance_error / 0.5)

        # ===== 安全奖励 =====

        # 3. 碰撞惩罚
        rewards['collision'] = -env_state['collision'].float()

        # 4. Near-miss惩罚
        clearance = env_state['clearance']
        near_miss = torch.clamp(self.near_miss_threshold - clearance, min=0)
        rewards['near_miss'] = -near_miss

        # 5. 风险屏障
        risk_barrier = torch.clamp(
            (self.safe_distance - clearance) / self.safe_distance,
            min=0
        )
        rewards['risk_barrier'] = -risk_barrier

        # ===== 平滑性奖励（关键！）=====

        # 6. y_eff切换惩罚
        y_eff = env_state['y_eff']
        y_eff_prev = env_state['y_eff_prev']
        y_eff_change = torch.abs(y_eff - y_eff_prev)

        # 惩罚大的切换
        rewards['switch_penalty'] = -torch.where(
            y_eff_change > self.switch_threshold,
            y_eff_change,
            torch.zeros_like(y_eff_change)
        )

        # 7. y_eff平滑度（连续惩罚）
        rewards['y_eff_smoothness'] = -y_eff_change

        # 8. beta平滑度
        beta = env_state['beta']
        beta_prev = env_state['beta_prev']
        beta_change = torch.abs(beta - beta_prev)
        rewards['beta_smoothness'] = -beta_change

        # 9. 命令平滑度
        cmd = env_state['cmd']
        cmd_prev = env_state['cmd_prev']
        cmd_jerk = torch.norm(cmd - cmd_prev, dim=-1)
        rewards['cmd_smoothness'] = torch.exp(-cmd_jerk / 0.3)

        # ===== 效率惩罚 =====

        # 10. 时间惩罚（防止停住）
        rewards['time_penalty'] = -torch.ones_like(target_distance) * env_state['dt']

        return rewards

    # ===== w vs r_risk 的职责分工（必须严格隔离）=====
    #
    # - r_risk（事后）：只评估“实际执行轨迹”的风险（如 near-miss / risk_barrier / clearance_pp）。
    # - w（事前）：评估“如果执行候选命令（尤其 cmd_F）会怎样”，用于提前偏置门控 y_eff。
    #
    # 严格禁止：
    # - 直接用 clearance_pp / r_risk 数值监督 w（会导致 w 退化为 clearance 分类器）。
    #
    def compute_w_auxiliary_loss(
        self,
        w_pred,
        affordance_or_occupancy,
        cmd_F,
        beta,
        horizon_s=0.8,
        dt=0.1,
    ):
        """
        w 的辅助监督损失（推荐：让 w 学到“事前预测”）

        伪标签方案A（推荐）：
        - 在当前帧的 affordance/occupancy 上，对候选命令 cmd_F 做 RiskAlong-lite/轨迹推演，
          得到 horizon 内的 min_clearance_F；
        - conflict_F = 1{min_clearance_F < safe_dist(beta)}；
        - L_w_aux = BCE(w_pred, conflict_F)。

        说明：
        - 这里评估的是“如果执行 cmd_F 会怎样”，而不是“已经执行的轨迹怎样”。
        - 实现必须向量化/GPU友好（禁止 per-env Python 循环）。
        """
        # 计算 beta 对应的安全距离阈值（必须与 post-processor 的 beta 映射一致）
        safe_dist = self.safe_distance  # placeholder: use beta-adjusted safe_dist in real code

        # min_clearance_F = simulate_command_trajectory_or_riskalong_lite(
        #     affordance_or_occupancy, cmd_F, horizon_s=horizon_s, dt=dt
        # )
        # conflict_F = (min_clearance_F < safe_dist).float()
        # loss = F.binary_cross_entropy(w_pred, conflict_F)
        # return loss, conflict_F
        pass
```

## 5.5 奖励权重汇总表

| 阶段            | 奖励项                  | 权重       | 目的          |
| ------------- | -------------------- | -------- | ----------- |
| **Follow专家**  | progress             | 2.0      | 朝目标移动       |
|               | distance_keeping     | 1.5      | 保持跟随距离      |
|               | heading_alignment    | 0.5      | 面向目标        |
|               | collision            | -10.0    | 避免碰撞        |
|               | time_penalty         | -0.01    | 效率          |
|               | stability            | 0.3      | 姿态稳定        |
|               | cmd_smoothness       | 0.2      | 命令平滑        |
| **Avoid专家**   | progress             | 1.0      | 朝waypoint移动 |
|               | collision            | -20.0    | 避免碰撞（更重要）   |
|               | near_miss            | -2.0     | 惩罚近距离       |
|               | risk_barrier         | -1.0     | 连续风险成本      |
|               | clearance_bonus      | 0.5      | 保持距离        |
|               | stability            | 0.3      | 姿态稳定        |
|               | cmd_smoothness       | 0.3      | 命令平滑        |
| **PCR-Net++** | progress             | 1.5      | 朝目标移动       |
|               | distance_keeping     | 1.0      | 保持跟随距离      |
|               | collision            | -15.0    | 避免碰撞        |
|               | near_miss            | -1.5     | 惩罚近距离       |
|               | risk_barrier         | -0.8     | 连续风险成本      |
|               | **switch_penalty**   | **-0.3** | **惩罚大切换**   |
|               | **y_eff_smoothness** | **-0.2** | **y_eff平滑** |
|               | **beta_smoothness**  | **-0.1** | **beta平滑**  |
|               | cmd_smoothness       | 0.3      | 命令平滑        |
|               | time_penalty         | -0.005   | 效率          |

---

# 第六部分：仿真场景设计

## 6.1 场景配置基类

```python
class SceneConfig:
    """场景配置基类"""

    def __init__(self):
        # 地形参数
        self.terrain_type = 'plane'  # 'plane', 'rough', 'slope'
        self.terrain_size = (20.0, 20.0)  # (x, y) in meters

        # 障碍物参数
        self.obstacle_type = 'none'  # 'none', 'poles', 'boxes', 'walls', 'gates'
        self.obstacle_density = 0.0  # obstacles per m²

        # 目标参数
        self.target_type = 'static'  # 'static', 'moving'
        self.target_speed_range = (0.0, 0.0)  # (min, max) m/s

        # 课程参数
        self.difficulty = 0.0  # [0, 1]
```

## 6.2 S0：平地移动目标跟随（预训练底座）

**目的**：先在无障碍平地上把“稳定 1m 跟随 + 保持目标在视野中心 + 六足全向移动跟随”练到位，再 resume 到 S1 强冲突门洞场景，避免把“基础跟随不稳”与“门洞冲突决策”混在一起训练。

**目标运动（difficulty curriculum）**：

- 低难度：匀速前进为主，少量方向扰动
- 中难度：加入后退、左右横移、斜向组合
- 高难度：更高频率的变速 + 变向 + 横移串联（例如加速前进→减速转弯→横移→再加速）

**视野保持**：

- 使用中心窗口约束（soft/hard），连续 `K=5` 个高层步超出 hard 窗口视为丢失并 reset（与训练 loop 的稳定性策略一致）。

---

## 6.3 S1：门洞走廊场景

```python
class S1GateCorridorConfig(SceneConfig):
    """
    S1: 门洞走廊场景

    核心演示场景：人可轻松通过，机器人紧凑通过
    """

    def __init__(self, difficulty=0.5):
        super().__init__()

        self.terrain_type = 'plane'
        self.obstacle_type = 'gates'

        # 走廊参数
        self.corridor_length = 15.0  # m
        self.corridor_width = 2.5    # m

        # 门洞参数（关键！）
        self.gate_count = 2          # 门洞数量
        self.gate_width_range = (0.85, 1.0)  # 课程学习
        self.gate_length = 0.3       # 门洞厚度
        self.gate_spacing_min = 3.0  # 门洞最小间距

        # 墙壁参数
        self.wall_height = 1.0       # m
        self.wall_thickness = 0.1    # m

        # 目标参数
        self.target_type = 'moving'
        self.target_speed_range = (0.4, 1.2)  # m/s（上限对齐训练：1.2）

        # 课程难度
        self.difficulty = difficulty
        self._apply_difficulty()

    def _apply_difficulty(self):
        """根据难度调整参数"""
        d = self.difficulty

        # 门洞宽度：难度越高越窄
        width_range = self.gate_width_range
        self.gate_width = width_range[1] - d * (width_range[1] - width_range[0])

        # 门洞数量：难度越高越多
        self.gate_count = int(1 + d * 2)  # 1-3个

        # 目标速度：难度越高越快
        speed_range = self.target_speed_range
        self.target_speed = speed_range[0] + d * (speed_range[1] - speed_range[0])

    def generate_obstacles(self):
        """生成门洞障碍物"""
        obstacles = []

        # 走廊两侧墙壁
        obstacles.append({
            'type': 'wall',
            'pos': (self.corridor_length / 2, self.corridor_width / 2, self.wall_height / 2),
            'size': (self.corridor_length, self.wall_thickness, self.wall_height),
        })
        obstacles.append({
            'type': 'wall',
            'pos': (self.corridor_length / 2, -self.corridor_width / 2, self.wall_height / 2),
            'size': (self.corridor_length, self.wall_thickness, self.wall_height),
        })

        # 门洞
        gate_positions = self._compute_gate_positions()
        for gate_x in gate_positions:
            # 门洞两侧的墙壁
            gap_half = self.gate_width / 2
            wall_length = (self.corridor_width - self.gate_width) / 2

            # 左侧墙
            obstacles.append({
                'type': 'wall',
                'pos': (gate_x, self.corridor_width / 2 - wall_length / 2, self.wall_height / 2),
                'size': (self.gate_length, wall_length, self.wall_height),
            })
            # 右侧墙
            obstacles.append({
                'type': 'wall',
                'pos': (gate_x, -self.corridor_width / 2 + wall_length / 2, self.wall_height / 2),
                'size': (self.gate_length, wall_length, self.wall_height),
            })

        return obstacles

    def _compute_gate_positions(self):
        """计算门洞位置（均匀分布）"""
        margin = 2.0  # 距离走廊端点的margin
        available_length = self.corridor_length - 2 * margin

        if self.gate_count == 1:
            return [self.corridor_length / 2]
        else:
            spacing = available_length / (self.gate_count + 1)
            return [margin + spacing * (i + 1) for i in range(self.gate_count)]
```

### S1-moving（强制冲突训练版本）

为保证训练中稳定产生 Follow vs Avoid 冲突段，引入 **S1-moving**：移动目标必须穿越门洞（而非随机游走）。目标运动使用 gate-by-gate 阶段机：

- **Approach**：朝下一个门洞中心前进（速度随 difficulty 增大）
- **Align**：进入门洞前的一段距离内，横向收敛到门洞可行区（可选小偏置，随难度增大）
- **Pass**：穿门洞段保持横向稳定，确保“目标可过、人可过、但机器人需精细对齐”
- **Post**：出门洞后短暂扰动（变速/轻微横移），制造决策与视野保持压力

并强制约束：

- 目标始终位于走廊边界内（与走廊宽度/门洞几何一致）
- 目标速度上限与全链路一致（`target_max_speed=1.2 m/s`）

## 6.4 S2：柱阵森林场景

```python
class S2ForestConfig(SceneConfig):
    """
    S2: 柱阵森林场景

    开放域避障 + 跟随速度约束
    """

    def __init__(self, difficulty=0.5):
        super().__init__()

        self.terrain_type = 'plane'
        self.obstacle_type = 'poles'

        # 区域参数
        self.area_size = (15.0, 10.0)  # (x, y) m

        # 柱子参数
        self.pole_radius_range = (0.1, 0.25)  # m
        self.pole_height = 1.0  # m
        self.pole_density_range = (1.5, 4.0)  # poles per m²

        # 目标参数
        self.target_type = 'moving'
        self.target_speed_range = (0.3, 0.8)  # m/s

        # 课程难度
        self.difficulty = difficulty
        self._apply_difficulty()

    def _apply_difficulty(self):
        """根据难度调整参数"""
        d = self.difficulty

        # 柱子密度：难度越高越密
        density_range = self.pole_density_range
        self.pole_density = density_range[0] + d * (density_range[1] - density_range[0])

        # 柱子半径：难度越高越大（更难绕行）
        radius_range = self.pole_radius_range
        self.pole_radius = radius_range[0] + d * (radius_range[1] - radius_range[0])

        # 目标速度
        speed_range = self.target_speed_range
        self.target_speed = speed_range[0] + d * (speed_range[1] - speed_range[0])

    def generate_obstacles(self):
        """生成随机柱子"""
        import numpy as np

        obstacles = []
        area = self.area_size[0] * self.area_size[1]
        num_poles = int(area * self.pole_density)

        # 随机生成柱子位置（避免重叠）
        min_distance = self.pole_radius * 3  # 柱子之间最小距离

        positions = []
        for _ in range(num_poles * 10):  # 多次尝试
            if len(positions) >= num_poles:
                break

            x = np.random.uniform(1.0, self.area_size[0] - 1.0)
            y = np.random.uniform(-self.area_size[1] / 2 + 1.0, self.area_size[1] / 2 - 1.0)

            # 检查与已有柱子的距离
            valid = True
            for px, py in positions:
                if np.sqrt((x - px)**2 + (y - py)**2) < min_distance:
                    valid = False
                    break

            if valid:
                positions.append((x, y))

        # 生成障碍物
        for x, y in positions:
            radius = np.random.uniform(
                self.pole_radius_range[0], 
                self.pole_radius_range[1]
            )
            obstacles.append({
                'type': 'cylinder',
                'pos': (x, y, self.pole_height / 2),
                'radius': radius,
                'height': self.pole_height,
            })

        return obstacles
```

## 6.4 S6：OOD Hold-out场景

```python
class S6OODConfig(SceneConfig):
    """
    S6: 结构化OOD场景（训练中不出现）

    用于泛化性测试
    """

    def __init__(self, ood_type='maze'):
        super().__init__()

        self.ood_type = ood_type  # 'maze', 'cluster', 'nonconvex'
        self._setup_ood()

    def _setup_ood(self):
        """设置OOD场景"""
        if self.ood_type == 'maze':
            # 迷宫式走廊
            self._setup_maze()
        elif self.ood_type == 'cluster':
            # 簇状柱群
            self._setup_cluster()
        elif self.ood_type == 'nonconvex':
            # 非凸障碍（L形、U形）
            self._setup_nonconvex()

    def _setup_maze(self):
        """迷宫场景"""
        self.terrain_type = 'plane'
        self.obstacle_type = 'walls'
        self.area_size = (15.0, 15.0)
        self.wall_pattern = 'maze'  # 预定义迷宫模式

    def _setup_cluster(self):
        """簇状柱群场景"""
        self.terrain_type = 'plane'
        self.obstacle_type = 'poles'
        self.area_size = (15.0, 10.0)
        self.cluster_count = 3
        self.poles_per_cluster = 8
        self.cluster_radius = 2.0

    def _setup_nonconvex(self):
        """非凸障碍场景"""
        self.terrain_type = 'plane'
        self.obstacle_type = 'boxes'
        self.area_size = (15.0, 10.0)
        self.obstacle_shapes = ['L', 'U', 'T']  # 非凸形状
```

## 6.5 移动目标生成器

```python
class MovingTargetGenerator:
    """
    移动目标生成器

    模拟人类行走行为
    """

    def __init__(self, cfg):
        self.speed_range = cfg.get('speed_range', (0.3, 0.8))
        self.turn_rate_max = cfg.get('turn_rate_max', 0.5)  # rad/s
        self.speed_change_prob = cfg.get('speed_change_prob', 0.02)
        self.turn_prob = cfg.get('turn_prob', 0.01)

    def reset(self, num_envs, device):
        """重置目标状态"""
        self.pos = torch.zeros(num_envs, 2, device=device)
        self.heading = torch.zeros(num_envs, device=device)
        self.speed = torch.zeros(num_envs, device=device)

        # 随机初始速度
        self.speed = torch.rand(num_envs, device=device) * (
            self.speed_range[1] - self.speed_range[0]
        ) + self.speed_range[0]

    def step(self, dt, obstacles=None):
        """
        更新目标位置

        Args:
            dt: 时间步长
            obstacles: 障碍物信息（用于避障）
        """
        N = self.pos.shape[0]
        device = self.pos.device

        # 随机速度变化
        speed_change_mask = torch.rand(N, device=device) < self.speed_change_prob
        new_speed = torch.rand(N, device=device) * (
            self.speed_range[1] - self.speed_range[0]
        ) + self.speed_range[0]
        self.speed = torch.where(speed_change_mask, new_speed, self.speed)

        # 随机转向
        turn_mask = torch.rand(N, device=device) < self.turn_prob
        turn_rate = (torch.rand(N, device=device) - 0.5) * 2 * self.turn_rate_max
        self.heading = torch.where(turn_mask, self.heading + turn_rate * dt, self.heading)

        # 更新位置
        vel = torch.stack([
            self.speed * torch.cos(self.heading),
            self.speed * torch.sin(self.heading)
        ], dim=-1)

        self.pos = self.pos + vel * dt

        return self.pos.clone(), vel.clone()
```

---

# 第七部分：完整训练流程

## 7.1 训练阶段总览

```
════════════════════════════════════════════════════════════════════════════════
                        训练阶段总览（预计4-6周）
════════════════════════════════════════════════════════════════════════════════

Week 1-2: Phase 1 - Follow专家训练
    ├── Stage 1.1: 平地 + 静态目标 (1-2天)
    ├── Stage 1.2: 平地 + 移动目标 (2-3天)
    └── Stage 1.3: 引入β风格条件 (1-2天)

Week 2-3: Phase 2 - Avoid专家训练
    ├── Stage 2.1: 平地 + 稀疏障碍 (2天)
    └── Stage 2.2: 平地 + 密集障碍/门洞 (2-3天)

Week 3-5: Phase 3 - PCR-Net++仲裁网络训练（核心）
    ├── Stage 3.1: y-only MoE基线 (2-3天)
    ├── Stage 3.2: y + β (联动Post-Processor) (2-3天)
    └── Stage 3.3: y + w + β (完整PCR-Net++) (3-4天)

Week 5-6: Phase 4 - 场景强化与评估
    ├── Stage 4.1: S1门洞场景强化 (2-3天)
    ├── Stage 4.2: 消融实验 (1-2天)
    ├── Stage 4.3: Pareto曲线实验 (1天)
    └── Stage 4.4: OOD泛化测试 (1天)
════════════════════════════════════════════════════════════════════════════════
```

## 7.2 Phase 1: Follow专家训练

### 7.2.1 Stage 1.1 配置

```python
# configs/follow_stage1_1.yaml

experiment:
  name: "follow_expert_static_target"
  seed: 42

environment:
  scene: "plane"  # 无障碍平地
  num_envs: 4096
  episode_length: 500  # 10s at 50Hz

target:
  type: "static"
  spawn_range: [5.0, 10.0]  # 目标距离范围

robot:
  init_pos_noise: 0.2
  init_heading_noise: 0.3

reward:
  weights:
    progress: 2.0
    distance_keeping: 1.5
    heading_alignment: 0.5
    collision: -10.0
    time_penalty: -0.01
    stability: 0.3
    cmd_smoothness: 0.2
  desired_distance: 1.5

training:
  algorithm: "PPO"
  learning_rate: 3e-4
  num_steps: 24  # steps per update
  num_minibatches: 4
  gamma: 0.99
  lam: 0.95
  clip_range: 0.2
  entropy_coef: 0.01
  max_grad_norm: 1.0
  total_timesteps: 50_000_000  # 约1-2天

curriculum:
  enabled: false  # Stage 1.1不启用课程

checkpoint:
  save_interval: 1000
  eval_interval: 500
```

### 7.2.2 Stage 1.2 配置

```python
# configs/follow_stage1_2.yaml

experiment:
  name: "follow_expert_moving_target"
  seed: 42
  load_checkpoint: "follow_expert_static_target/best.pt"  # 从Stage 1.1加载

environment:
  scene: "plane"
  num_envs: 4096
  episode_length: 1000  # 20s

target:
  type: "moving"
  speed_curriculum:
    - {speed: [0.1, 0.3], duration: 10_000_000}
    - {speed: [0.3, 0.5], duration: 10_000_000}
    - {speed: [0.5, 0.8], duration: 20_000_000}
    - {speed: [0.6, 1.0], duration: 20_000_000}
  turn_rate_max: 0.5

reward:
  weights:
    progress: 2.0
    distance_keeping: 2.0  # 增加权重
    heading_alignment: 0.5
    collision: -10.0
    time_penalty: -0.01
    stability: 0.3
    cmd_smoothness: 0.3
  desired_distance: 1.5
  distance_tolerance: 0.3

training:
  total_timesteps: 60_000_000  # 约2-3天
```

### 7.2.3 Stage 1.3 配置（引入β）

```python
# configs/follow_stage1_3.yaml

experiment:
  name: "follow_expert_with_beta"
  seed: 42
  load_checkpoint: "follow_expert_moving_target/best.pt"

environment:
  scene: "plane"
  num_envs: 4096
  episode_length: 1000

target:
  type: "moving"
  speed_range: [0.4, 1.0]

expert:
  beta_conditioned: true  # 启用β条件化
  beta_sampling: "uniform"  # 训练时随机采样β
  beta_range: [0.0, 1.0]

reward:
  # 根据β调整奖励权重
  beta_modulated_weights:
    progress:
      beta_0: 2.5  # β=0时更重视进度
      beta_1: 1.0  # β=1时降低进度权重
    cmd_smoothness:
      beta_0: 0.2
      beta_1: 0.5  # β=1时更重视平滑

training:
  total_timesteps: 40_000_000

validation:
  # 验证不同β下的行为差异
  beta_sweep: [0.0, 0.25, 0.5, 0.75, 1.0]
```

## 7.3 Phase 2: Avoid专家训练

### 7.3.1 Stage 2.1 配置

```python
# configs/avoid_stage2_1.yaml

experiment:
  name: "avoid_expert_sparse"
  seed: 42

environment:
  scene: "S2_forest"
  scene_config:
    difficulty: 0.3  # 稀疏
    pole_density: 2.0
  num_envs: 4096
  episode_length: 800

waypoint:
  type: "fixed_direction"  # 非人类跟随，固定方向
  direction: [1.0, 0.0]  # +X方向

reward:
  weights:
    progress: 1.0
    collision: -20.0
    near_miss: -2.0
    risk_barrier: -1.0
    clearance_bonus: 0.5
    stability: 0.3
    cmd_smoothness: 0.3
  safe_distance: 0.5
  near_miss_threshold: 0.3

training:
  total_timesteps: 50_000_000
```

### 7.3.2 Stage 2.2 配置

```python
# configs/avoid_stage2_2.yaml

experiment:
  name: "avoid_expert_dense_gate"
  seed: 42
  load_checkpoint: "avoid_expert_sparse/best.pt"

environment:
  scene: "mixed"  # S1 + S2混合
  scene_weights:
    S1_gate: 0.4
    S2_forest: 0.6
  scene_config:
    S1:
      difficulty_curriculum: [0.3, 0.5, 0.7]
      gate_width_range: [0.85, 1.0]
    S2:
      difficulty_curriculum: [0.4, 0.6, 0.8]
      pole_density_range: [2.5, 4.0]
  num_envs: 4096
  episode_length: 1000

expert:
  beta_conditioned: true
  beta_range: [0.0, 1.0]

reward:
  weights:
    progress: 1.0
    collision: -20.0
    near_miss: -2.5
    risk_barrier: -1.5
    clearance_bonus: 0.5
    stability: 0.3
    cmd_smoothness: 0.4

training:
  total_timesteps: 60_000_000
```

## 7.4 Phase 3: PCR-Net++仲裁网络训练

### 7.4.1 Stage 3.1: y-only基线

```python
# configs/pcr_stage3_1_y_only.yaml

experiment:
  name: "pcr_y_only_baseline"
  seed: 42

environment:
  scene: "S2_forest"
  scene_config:
    difficulty: 0.5
  num_envs: 4096
  episode_length: 1000

target:
  type: "moving"
  speed_range: [0.4, 0.8]

experts:
  follow:
    checkpoint: "follow_expert_with_beta/best.pt"
    freeze: true
  avoid:
    checkpoint: "avoid_expert_dense_gate/best.pt"
    freeze: true

pcr_net:
  output: ["y"]  # 仅输出y
  # w和beta固定
  w_fixed: 0.0
  beta_fixed: 0.5

post_processor:
  enabled: true
  beta_linked: false  # 不联动，使用固定beta

reward:
  weights:
    progress: 1.5
    distance_keeping: 1.0
    collision: -15.0
    near_miss: -1.5
    cmd_smoothness: 0.3
    time_penalty: -0.005

training:
  total_timesteps: 40_000_000

metrics:
  # 记录失败模式
  track:
    - near_miss_rate
    - switch_rate
    - cmd_jerk
    - y_saturation_ratio
```

### 7.4.2 Stage 3.2: y + β（联动Post-Processor）

```python
# configs/pcr_stage3_2_y_beta.yaml

experiment:
  name: "pcr_y_beta_linked"
  seed: 42
  load_checkpoint: "pcr_y_only_baseline/best.pt"

environment:
  scene: "mixed"
  scene_weights:
    S1_gate: 0.3
    S2_forest: 0.7
  scene_config:
    S1:
      difficulty: 0.5
    S2:
      difficulty: 0.6
  num_envs: 4096
  episode_length: 1000

target:
  type: "moving"
  speed_range: [0.4, 1.0]

pcr_net:
  output: ["y", "beta"]  # 输出y和beta
  w_fixed: 0.0  # w仍固定

post_processor:
  enabled: true
  beta_linked: true  # 关键：启用β联动
  beta_params:
    safe_dist: [0.35, 1.00]
    max_lin_vel: [1.00, 0.35]
    max_ang_vel: [1.50, 0.50]
    max_delta_lin: [0.15, 0.05]
    max_delta_ang: [0.30, 0.10]
    risk_clamp_gain: [1.0, 3.0]

reward:
  weights:
    progress: 1.5
    distance_keeping: 1.0
    collision: -15.0
    near_miss: -1.5
    risk_barrier: -0.8
    beta_smoothness: -0.1
    cmd_smoothness: 0.3
    time_penalty: -0.005

training:
  total_timesteps: 50_000_000

validation:
  # 验证Pareto曲线
  beta_sweep: [0.0, 0.25, 0.5, 0.75, 1.0]
  metrics:
    - success_rate
    - time_to_goal
    - collision_rate
    - near_miss_rate
    - min_clearance
    - cmd_jerk
```

### 7.4.3 Stage 3.3: 完整PCR-Net++

```python
# configs/pcr_stage3_3_full.yaml

experiment:
  name: "pcr_full_y_w_beta"
  seed: 42
  load_checkpoint: "pcr_y_beta_linked/best.pt"

environment:
  scene: "mixed"
  scene_weights:
    S1_gate: 0.4
    S2_forest: 0.6
  scene_config:
    S1:
      difficulty_curriculum: [0.4, 0.6, 0.8]
    S2:
      difficulty_curriculum: [0.5, 0.7, 0.9]
  num_envs: 4096
  episode_length: 1200

target:
  type: "moving"
  speed_range: [0.3, 1.0]
  turn_rate_max: 0.6

pcr_net:
  output: ["y", "w", "beta"]  # 完整输出

  # w的命令条件化输入（关键！）
  w_input:
    - cmd_F
    - cmd_A
    - delta_cmd
    - risk_F
    - risk_A

  # 融合参数
  lambda_w: 0.4
  gamma_smooth: 0.7
  delta_hyst: 0.08

post_processor:
  enabled: true
  beta_linked: true

  # 条件性融合
  conditional_fusion:
    enabled: true
    tau_consist: 0.3
    tau_danger: 0.7
    tau_temp: 0.3

reward:
  weights:
    progress: 1.5
    distance_keeping: 1.0
    collision: -15.0
    near_miss: -1.5
    risk_barrier: -0.8
    switch_penalty: -0.3
    y_eff_smoothness: -0.2
    beta_smoothness: -0.1
    cmd_smoothness: 0.3
    time_penalty: -0.005

  # w辅助损失（可选）
  w_auxiliary:
    enabled: true
    weight: 0.1

training:
  total_timesteps: 80_000_000

  # 监控
  monitor:
    - w_clearance_correlation  # 检测w退化
    - switch_rate
    - cmd_jerk

validation:
  beta_sweep: [0.0, 0.25, 0.5, 0.75, 1.0]
  w_degradation_check: true  # 检测w是否退化
```

## 7.5 训练脚本框架

```python
# train_pcr_net.py

import torch
import hydra
from omegaconf import DictConfig

from envs.hex_follow_avoid_env import HexFollowAvoidEnv
from models.pcr_net import PCRNetPlusPlus
from models.experts import FollowExpert, AvoidExpert
from trainers.ppo_trainer import PPOTrainer
from utils.logger import WandbLogger
from utils.checkpoint import CheckpointManager

@hydra.main(config_path="configs", config_name="pcr_stage3_3_full")
def main(cfg: DictConfig):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建环境
    env = HexFollowAvoidEnv(
        num_envs=cfg.environment.num_envs,
        scene_config=cfg.environment.scene_config,
        device=device,
    )

    # 加载冻结的专家
    follow_expert = FollowExpert.load(cfg.experts.follow.checkpoint)
    follow_expert.eval()
    for p in follow_expert.parameters():
        p.requires_grad = False

    avoid_expert = AvoidExpert.load(cfg.experts.avoid.checkpoint)
    avoid_expert.eval()
    for p in avoid_expert.parameters():
        p.requires_grad = False

    # 创建PCR-Net++
    pcr_net = PCRNetPlusPlus(
        robot_state_dim=9,
        target_state_dim=6,
        affordance_dim=32,
        history_dim=16,
    ).to(device)

    # 加载checkpoint（如果有）
    if cfg.experiment.get('load_checkpoint'):
        pcr_net.load_state_dict(torch.load(cfg.experiment.load_checkpoint))

    # 创建Post-Processor
    post_processor = BatchCommandPostProcessor(
        num_envs=cfg.environment.num_envs,
        device=device,
    )

    # 创建训练器
    trainer = PPOTrainer(
        env=env,
        pcr_net=pcr_net,
        follow_expert=follow_expert,
        avoid_expert=avoid_expert,
        post_processor=post_processor,
        cfg=cfg.training,
    )

    # 日志
    logger = WandbLogger(
        project="hexapod_follow_avoid",
        name=cfg.experiment.name,
        config=dict(cfg),
    )

    # Checkpoint管理
    ckpt_manager = CheckpointManager(
        save_dir=f"checkpoints/{cfg.experiment.name}",
        save_interval=cfg.checkpoint.save_interval,
    )

    # 训练循环
    for iteration in range(cfg.training.total_iterations):
        # 收集数据
        rollout = trainer.collect_rollout()

        # 更新策略
        metrics = trainer.update(rollout)

        # 记录指标
        logger.log(metrics, step=iteration)

        # 检测w退化
        if cfg.training.monitor.w_clearance_correlation:
            w_corr = compute_w_clearance_correlation(rollout)
            if w_corr > 0.9:
                logger.warning(f"w退化警告：与clearance相关性={w_corr:.3f}")

        # 保存checkpoint
        ckpt_manager.save_if_needed(
            iteration=iteration,
            model=pcr_net,
            metrics=metrics,
        )

        # 定期评估
        if iteration % cfg.checkpoint.eval_interval == 0:
            eval_metrics = trainer.evaluate(
                beta_sweep=cfg.validation.beta_sweep,
            )
            logger.log(eval_metrics, step=iteration, prefix="eval/")

    logger.finish()

if __name__ == "__main__":
    main()
```

---

# 第八部分：评估与实验设计

## 8.1 评估指标定义

```python
class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, cfg):
        self.near_miss_threshold = cfg.get('near_miss_threshold', 0.3)
        self.switch_threshold = cfg.get('switch_threshold', 0.15)

    def compute_all(self, episode_data):
        """
        计算所有指标

        Args:
            episode_data: dict containing trajectory data
        """
        metrics = {}

        # 任务指标
        metrics['success_rate'] = self._success_rate(episode_data)
        metrics['time_to_goal'] = self._time_to_goal(episode_data)

        # 安全指标
        metrics['collision_rate'] = self._collision_rate(episode_data)
        metrics['near_miss_rate'] = self._near_miss_rate(episode_data)
        metrics['min_clearance'] = self._min_clearance(episode_data)
        metrics['near_miss_integral'] = self._near_miss_integral(episode_data)

        # 平滑性指标
        metrics['cmd_jerk'] = self._cmd_jerk(episode_data)
        metrics['switch_rate'] = self._switch_rate(episode_data)
        metrics['y_eff_std'] = self._y_eff_std(episode_data)

        return metrics

    def _success_rate(self, data):
        """成功率"""
        return data['success'].float().mean().item()

    def _time_to_goal(self, data):
        """平均到达时间（仅成功episode）"""
        success_mask = data['success']
        if success_mask.sum() == 0:
            return float('inf')
        return data['episode_length'][success_mask].float().mean().item()

    def _collision_rate(self, data):
        """碰撞率"""
        return data['collision'].float().mean().item()

    def _near_miss_rate(self, data):
        """near-miss帧占比"""
        clearance = data['clearance']  # (T, N)
        near_miss = (clearance < self.near_miss_threshold).float()
        return near_miss.mean().item()

    def _min_clearance(self, data):
        """最小clearance（所有episode）"""
        return data['clearance'].min().item()

    def _near_miss_integral(self, data):
        """near-miss积分"""
        clearance = data['clearance']
        near_miss = torch.clamp(self.near_miss_threshold - clearance, min=0)
        return near_miss.sum(dim=0).mean().item()  # 平均每个episode的积分

    def _cmd_jerk(self, data):
        """命令jerk（RMS）"""
        cmd = data['cmd']  # (T, N, 3)
        cmd_diff = cmd[1:] - cmd[:-1]
        jerk = torch.norm(cmd_diff, dim=-1)
        return jerk.mean().item()

    def _switch_rate(self, data):
        """y_eff切换率"""
        y_eff = data['y_eff']  # (T, N)
        y_eff_diff = torch.abs(y_eff[1:] - y_eff[:-1])
        switches = (y_eff_diff > self.switch_threshold).float()
        return switches.sum(dim=0).mean().item()  # 平均每个episode的切换次数

    def _y_eff_std(self, data):
        """y_eff标准差"""
        return data['y_eff'].std().item()
```

## 8.2 消融实验矩阵

```python
# 消融实验配置
ABLATION_CONFIGS = {
    'y_only': {
        'pcr_output': ['y'],
        'w_fixed': 0.0,
        'beta_fixed': 0.5,
        'beta_linked': False,
    },
    'y_w': {
        'pcr_output': ['y', 'w'],
        'beta_fixed': 0.5,
        'beta_linked': False,
    },
    'y_beta_no_link': {
        'pcr_output': ['y', 'beta'],
        'w_fixed': 0.0,
        'beta_linked': False,  # 关键：不联动
    },
    'y_beta_link': {
        'pcr_output': ['y', 'beta'],
        'w_fixed': 0.0,
        'beta_linked': True,  # 关键：联动
    },
    'y_w_beta_link': {
        'pcr_output': ['y', 'w', 'beta'],
        'beta_linked': True,
    },
    'full_no_smooth': {
        'pcr_output': ['y', 'w', 'beta'],
        'beta_linked': True,
        'smoothing': False,  # 关闭平滑机制
    },
    'full_no_conditional': {
        'pcr_output': ['y', 'w', 'beta'],
        'beta_linked': True,
        'conditional_fusion': False,  # 关闭条件性融合
    },
}
```

## 8.3 Pareto曲线实验

```python
def run_pareto_experiment(model, env, beta_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                          num_episodes=100, seeds=[0, 1, 2, 3, 4]):
    """
    运行Pareto曲线实验

    Returns:
        pareto_data: dict with beta as key, metrics as value
    """
    pareto_data = {}

    for beta in beta_values:
        all_metrics = []

        for seed in seeds:
            # 固定seed
            torch.manual_seed(seed)
            env.reset()

            # 运行评估
            metrics = evaluate_with_fixed_beta(
                model=model,
                env=env,
                beta=beta,
                num_episodes=num_episodes // len(seeds),
            )
            all_metrics.append(metrics)

        # 聚合
        pareto_data[beta] = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        pareto_data[beta]['std'] = {
            key: np.std([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }

    return pareto_data

def plot_pareto_curve(pareto_data, x_metric='time_to_goal', y_metric='min_clearance'):
    """绘制Pareto曲线"""
    import matplotlib.pyplot as plt

    betas = sorted(pareto_data.keys())
    x_values = [pareto_data[b][x_metric] for b in betas]
    y_values = [pareto_data[b][y_metric] for b in betas]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, 'b-o', linewidth=2, markersize=10)

    for i, beta in enumerate(betas):
        plt.annotate(f'β={beta}', (x_values[i], y_values[i]), 
                    textcoords="offset points", xytext=(5, 5))

    plt.xlabel(f'Efficiency ({x_metric})')
    plt.ylabel(f'Safety ({y_metric})')
    plt.title('Risk-Efficiency Pareto Curve')
    plt.grid(True)
    plt.savefig('pareto_curve.png', dpi=150)
    plt.close()
```

## 8.4 w退化检测

```python
def check_w_degradation(model, env, num_episodes=50):
    """
    检测w是否退化为clearance分类器

    Returns:
        degradation_report: dict with analysis results
    """
    w_values = []
    clearance_values = []
    cmd_F_values = []
    cmd_A_values = []

    # 收集数据
    for _ in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                y, w, beta, y_eff = model(obs)
                cmd_F = follow_expert(obs)
                cmd_A = avoid_expert(obs)

            w_values.append(w.cpu().numpy())
            clearance_values.append(obs['clearance'].cpu().numpy())
            cmd_F_values.append(cmd_F.cpu().numpy())
            cmd_A_values.append(cmd_A.cpu().numpy())

            action = ...  # 执行动作
            obs, reward, done, info = env.step(action)

    # 分析
    w_values = np.concatenate(w_values)
    clearance_values = np.concatenate(clearance_values)

    # 1. w与clearance的相关性
    w_clearance_corr = np.corrcoef(w_values.flatten(), clearance_values.flatten())[0, 1]

    # 2. w对候选命令的敏感性
    # 固定状态，改变候选命令，观察w变化
    w_cmd_sensitivity = compute_w_cmd_sensitivity(model, env)

    # 3. 判定
    is_degraded = (w_clearance_corr > 0.9) and (w_cmd_sensitivity < 0.1)

    report = {
        'w_clearance_correlation': w_clearance_corr,
        'w_cmd_sensitivity': w_cmd_sensitivity,
        'is_degraded': is_degraded,
        'recommendation': '需要加强命令条件化输入' if is_degraded else '正常',
    }

    return report
```

---

# 第九部分：行动指南与检查清单

## 9.1 每日检查清单

```markdown
### 每日训练前检查

- [ ] GPU显存是否充足（3090需要预留>20GB）
- [ ] 上一次训练的checkpoint是否保存
- [ ] 配置文件参数是否正确
- [ ] Wandb/日志系统是否正常

### 每日训练中监控

- [ ] reward曲线是否正常上升
- [ ] 碰撞率是否在下降
- [ ] near-miss是否在下降
- [ ] switch_rate是否过高（>2.0表示抖动严重）
- [ ] cmd_jerk是否过大
- [ ] y/w/β的分布是否合理（避免全0或全1）

### 每日训练后分析

- [ ] 保存最新checkpoint
- [ ] 记录关键指标变化
- [ ] 识别潜在问题
- [ ] 规划下一步行动
```

## 9.2 阶段验收标准

```markdown
### Phase 1: Follow专家

**Stage 1.1 验收标准**：
- [ ] 静态目标成功率 > 95%
- [ ] 平均到达时间 < 15s
- [ ] 碰撞率 < 1%

**Stage 1.2 验收标准**：
- [ ] 移动目标成功率 > 85%
- [ ] 跟随距离误差 < 0.5m（均值）
- [ ] 目标丢失率 < 10%

**Stage 1.3 验收标准**：
- [ ] 不同β下行为有明显差异
- [ ] β=0时速度更快，β=1时更平滑
- [ ] 验证图：β vs 平均速度曲线

### Phase 2: Avoid专家

**Stage 2.1 验收标准**：
- [ ] 碰撞率 < 5%
- [ ] near-miss率 < 20%
- [ ] 能穿越稀疏障碍区域

**Stage 2.2 验收标准**：
- [ ] 碰撞率 < 3%
- [ ] near-miss率 < 15%
- [ ] 能穿越门洞（宽度0.9m）
- [ ] 不同β下避障距离有明显差异

### Phase 3: PCR-Net++

**Stage 3.1 (y-only) 验收标准**：
- [ ] 记录基线指标
- [ ] 识别失败模式（滞后、抖动）
- [ ] near-miss率和switch_rate作为后续改进的基准

**Stage 3.2 (y+β) 验收标准**：
- [ ] near-miss率相比y-only下降 > 20%
- [ ] cmd_jerk相比y-only下降 > 15%
- [ ] Pareto曲线有效（单调递增/递减）
- [ ] β=0.5时达到合理的效率-安全平衡

**Stage 3.3 (y+w+β) 验收标准**：
- [ ] switch_rate相比y+β下降 > 15%
- [ ] w退化检测通过（相关性 < 0.8）
- [ ] 事件对齐图显示w的提前性
- [ ] 窄缝场景成功率 > 80%
```

## 9.3 常见问题排查

```markdown
### 问题1：训练不收敛

**症状**：reward震荡或不上升

**排查步骤**：
1. 检查学习率是否过大
2. 检查reward权重是否合理
3. 检查梯度是否爆炸（norm > 10）
4. 检查专家是否正确冻结

**解决方案**：
- 降低学习率（3e-4 → 1e-4）
- 调整reward权重（降低惩罚项）
- 添加梯度裁剪

### 问题2：碰撞率过高

**症状**：碰撞率 > 10%

**排查步骤**：
1. 检查Avoid专家是否正常
2. 检查clearance计算是否正确
3. 检查y是否偏向Follow太多

**解决方案**：
- 增加碰撞惩罚权重
- 检查并修复clearance计算
- 添加y的正则化（偏向Avoid）

### 问题3：switch_rate过高

**症状**：switch_rate > 3.0（每秒切换3次以上）

**排查步骤**：
1. 检查平滑机制是否生效
2. 检查滞回阈值设置
3. 检查switch_penalty权重

**解决方案**：
- 增加gamma（平滑系数）
- 增加delta_hyst（滞回阈值）
- 增加switch_penalty权重

### 问题4：w退化

**症状**：w与clearance高度相关（>0.9）

**排查步骤**：
1. 检查w的输入是否包含候选命令
2. 检查RiskAlong计算是否正确
3. 检查w辅助损失是否生效

**解决方案**：
- 确保cmd_F, cmd_A, risk_F, risk_A在w的输入中
- 增加w辅助损失权重
- 添加w与clearance的解相关正则化

### 问题5：β拉满=停住

**症状**：β=1时速度几乎为0

**排查步骤**：
1. 检查max_lin_vel(β=1)的值
2. 检查是否有进度保底机制
3. 检查time_penalty权重

**解决方案**：
- 提高max_lin_vel(β=1)的下限（至少0.35m/s）
- 增加time_penalty权重
- 添加低速惩罚

### 问题6：Pareto曲线不单调

**症状**：β增大但安全指标反而下降

**排查步骤**：
1. 检查β映射方向是否正确
2. 检查Post-Processor是否正确应用β
3. 检查是否有其他因素干扰

**解决方案**：
- 验证β映射公式
- 打印并检查实际使用的约束参数
- 简化系统，逐一排除干扰因素
```

## 9.4 实验记录模板

```markdown
## 实验记录

**实验名称**: [填写]
**日期**: [填写]
**阶段**: [Phase X / Stage X.X]

### 配置变更
- [列出相比上次实验的配置变更]

### 训练过程
- 开始时间: [填写]
- 结束时间: [填写]
- 总步数: [填写]
- GPU占用: [填写]

### 关键指标
| 指标 | 初始值 | 最终值 | 变化 |
|-----|-------|-------|-----|
| success_rate | | | |
| collision_rate | | | |
| near_miss_rate | | | |
| switch_rate | | | |
| cmd_jerk | | | |

### 观察与分析
- [记录观察到的现象]
- [分析原因]

### 问题与风险
- [列出发现的问题]
- [评估风险]

### 下一步计划
- [列出下一步行动]

### 附件
- [ ] 训练曲线截图
- [ ] Pareto曲线（如适用）
- [ ] 事件对齐图（如适用）
- [ ] Checkpoint路径: [填写]
```

## 9.5 论文实验检查清单

```markdown
### 论文必需实验

**基准对比**：
- [ ] y-only MoE基线
- [ ] 规则仲裁基线
- [ ] APF基线（可选）
- [ ] DWA基线（可选）

**消融实验**：
- [ ] y-only vs y+w vs y+β vs y+w+β
- [ ] β联动 vs β不联动（关键！）
- [ ] 有平滑 vs 无平滑
- [ ] 有条件融合 vs 无条件融合

**Pareto曲线**：
- [ ] β sweep: [0.0, 0.25, 0.5, 0.75, 1.0]
- [ ] 每个β点至少5个seeds
- [ ] 曲线单调性验证

**w退化检测**：
- [ ] w与clearance相关性 < 0.8
- [ ] w对候选命令敏感

**OOD泛化**：
- [ ] S6 hold-out测试
- [ ] 与IID结果对比

**窄缝场景**：
- [ ] 门洞宽度0.85m成功率
- [ ] 录制演示视频

### 论文图表清单

- [ ] Fig.1: 系统架构图
- [ ] Fig.2: y-only失败模式可视化
- [ ] Fig.3: PCR-Net++网络结构
- [ ] Fig.4: β参数映射图
- [ ] Fig.5: 基准对比轨迹
- [ ] Fig.6: Pareto曲线（核心图）
- [ ] Fig.7: y/w/β时序曲线
- [ ] Fig.8: 窄缝场景演示
- [ ] Table I: 基准对比指标
- [ ] Table II: 消融实验指标
- [ ] Table III: β参数映射表
```

---

# 附录A：关键超参数汇总表

| 类别       | 参数                  | 值                       | 说明          |
| -------- | ------------------- | ----------------------- | ----------- |
| **控制频率** | high_level_freq     | 10 Hz                   | 高层决策频率      |
|          | low_level_freq      | 50 Hz                   | 底层控制频率      |
| **跟随距离** | desired_distance    | 1.5 m                   | 期望跟随距离      |
|          | distance_min        | 1.0 m                   | 最小跟随距离      |
|          | distance_max        | 2.0 m                   | 最大跟随距离      |
| **门洞尺寸** | gate_width_range    | [0.85, 1.0] m           | 门洞宽度范围      |
|          | robot_walking_width | 0.70 m                  | 机器人行走宽度     |
| **β参数**  | safe_dist           | [0.35, 1.0] m           | 安全距离范围      |
|          | max_lin_vel         | [1.0, 0.35] m/s         | 最大线速度范围     |
|          | max_ang_vel         | [1.5, 0.5] rad/s        | 最大角速度范围     |
|          | max_delta_lin       | [0.15, 0.05] m/s/step   | 线速度变化率范围    |
|          | max_delta_ang       | [0.30, 0.10] rad/s/step | 角速度变化率范围    |
|          | risk_clamp_gain     | [1.0, 3.0]              | 风险钳制增益范围    |
| **融合参数** | lambda_w            | 0.4                     | w的影响权重      |
|          | gamma_smooth        | 0.7                     | 时序平滑系数      |
|          | delta_hyst          | 0.08                    | 滞回阈值        |
|          | tau_consist         | 0.3                     | 一致性阈值       |
|          | tau_danger          | 0.7                     | 危险阈值        |
|          | tau_temp            | 0.3                     | Softmax温度   |
| **视野保持** | gaze_Kp             | 0.5                     | 视野保持增益      |
| **奖励阈值** | near_miss_threshold | 0.3 m                   | near-miss阈值 |
|          | switch_threshold    | 0.15                    | 切换检测阈值      |

---

**文档结束**

本文档提供了面向IEEE RAL的完整技术方案，包含：

1. 所有核心数学公式的严格定义
2. 完整的代码实现框架
3. 详细的训练流程和配置
4. 精心设计的奖励函数
5. 仿真场景设计
6. 评估实验设计
7. 行动指南和检查清单

请按照Phase顺序执行训练，并使用检查清单确保每个阶段达到验收标准。
