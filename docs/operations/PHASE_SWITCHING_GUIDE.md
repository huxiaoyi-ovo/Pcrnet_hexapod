# Phase 切换配置指南

> 文档状态：当前规范（Operations）。用于阶段切换执行；若与 V7 技术方案冲突，以 V7 为准。

本文档描述如何在不同训练阶段之间切换配置，确保架构清晰且易于维护。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: EGPO Locomotion Training                               │
│  - Algorithm: Expert-Guided Policy Optimization                  │
│  - Goal: Learn stable hexapod locomotion with camera stability   │
│  - Input: Random velocity commands                               │
│  - Output: Locomotion policy (底层运动策略)                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Teacher Navigation Training                            │
│  - Algorithm: Teacher-Student (Teacher phase)                    │
│  - Goal: Learn navigation with GT affordance                     │
│  - Input: GT affordance map + goal position                      │
│  - Output: Teacher policy (高层导航策略)                           │
│  - Constraint: Freeze student (locomotion) policy                │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Student Distillation Training                         │
│  - Algorithm: Teacher-Student (Student phase)                    │
│  - Goal: Distill Teacher knowledge with estimated affordance     │
│  - Input: Estimated affordance map + goal position               │
│  - Output: Student policy (无需GT的导航策略)                       │
│  - Constraint: Freeze student (locomotion) policy                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: EGPO 运动训练

### 目标

训练底层六足运动控制策略，学习：

- 稳定的步态
- 相机稳定性（低抖动、低俯仰角速度、低垂直颠簸）
- 对随机速度指令的跟踪能力

### 配置参数

在 `hex_terrain_config.py` 中设置：

```python
class navigation:
    # Phase 控制
    enable_nav_reward = False          # ✓ 关键：使用 locomotion rewards
    freeze_student_policy = False      # ✓ Phase 1 不冻结策略
    use_adapter = False                # ✓ Phase 1 不使用 Adapter
    use_gt_affordance = True           # 不影响（Phase 1 不使用导航）

class rewards:
    class scales:
        # 核心运动奖励
        tracking_lin_vel = 2.0
        tracking_ang_vel = 1.5

        # 相机稳定性（Sim-to-Real 关键）
        camera_stability = 2.5         # ✓ 关键：训练相机稳定性
        lin_vel_z = -2.0
        ang_vel_xy = -0.05

        # Phase 2/3 参数（Phase 1 忽略）
        nav_stability_weight = 0.3
```

### 训练命令

```bash
cd /home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym
python legged_gym/scripts/train.py --task=hex_terrain --run_name=phase1_locomotion
```

### 预期结果

- **运动能力**: 成功跟踪随机速度指令
- **稳定性指标**: 
  - `camera_stability_reward` > 0.8
  - `lin_vel_z_mean` < 0.05 m/s
  - `ang_vel_xy_mean` < 0.1 rad/s
- **训练时长**: 约 2000 iterations (EGPO BC loss 在 200 iter 后衰减完毕)

### 输出文件

```
logs/hex_ground/<timestamp>/
  ├── model_2000.pt          # 最终训练的策略
  ├── config.json            # 训练配置备份
  └── summaries/             # TensorBoard 日志
```

---

## Phase 2: Teacher 导航训练

### 目标

在 **冻结的底层运动策略** 基础上，训练高层导航策略：

- 使用 **Ground Truth affordance** map
- 学习从当前位置到目标位置的导航
- 保持底层运动策略的相机稳定性

### 配置参数

在 `hex_terrain_config.py` 中设置：

```python
class navigation:
    # Phase 控制
    enable_nav_reward = True           # ✓ 关键：切换到导航奖励
    freeze_student_policy = True       # ✓ 关键：冻结底层策略
    student_fine_tune_lr = 1e-6        # 可选：极小学习率微调
    use_adapter = True                 # ✓ 关键：使用 LocomotionAdapter
    use_gt_affordance = True           # ✓ 关键：Phase 2 使用 GT

    # LocomotionAdapter 参数
    adapter_distance_scale = 2.0       # 距离缩放因子
    adapter_max_ang_vel = 1.0          # 最大角速度 (rad/s)
    adapter_heading_gain = 2.0         # 朝向修正增益

    # 目标生成
    goal_mode = 'random'               # 'random', 'fixed', 'waypoints'
    goal_range_x = [3.0, 15.0]
    goal_range_y = [-8.0, 8.0]
    goal_min_distance = 3.0

class rewards:
    class scales:
        # Phase 1 奖励权重保持不变（但不使用）
        tracking_lin_vel = 2.0
        tracking_ang_vel = 1.5
        camera_stability = 2.5

        # Phase 2/3: 导航时的稳定性保持
        nav_stability_weight = 0.3     # ✓ 可选：额外添加稳定性奖励
```

### 训练命令

```bash
# 加载 Phase 1 训练的模型作为底层策略
python legged_gym/scripts/train.py \
    --task=hex_terrain \
    --run_name=phase2_teacher \
    --load_run=phase1_locomotion \
    --checkpoint=2000
```

### ⚠️ 关键实现要求

#### 1. 策略冻结机制

在 Runner 中实现：

```python
if self.cfg.navigation.freeze_student_policy:
    # 冻结底层策略参数
    for param in self.alg.actor_critic.actor.parameters():
        param.requires_grad = False

    # 如果使用微调而非完全冻结
    if self.cfg.navigation.student_fine_tune_lr > 0:
        for param in self.alg.actor_critic.actor.parameters():
            param.requires_grad = True
        # 为底层策略设置单独的优化器
        self.student_optimizer = torch.optim.Adam(
            self.alg.actor_critic.actor.parameters(),
            lr=self.cfg.navigation.student_fine_tune_lr
        )
```

#### 2. LocomotionAdapter 集成

已在 `hex_terrain.py` 中实现，会自动在 `_get_effective_commands()` 中生效。

#### 3. Teacher 输出接口

Teacher 网络应输出：

- `subgoal_local`: [N, 2] 局部坐标系下的子目标
- `intensity`: [N] 运动强度 [0, 1]

这些会被 `LocomotionAdapter` 转换为底层速度指令。

### 预期结果

- **导航能力**: 成功到达随机目标位置
- **稳定性保持**: `camera_stability` 不应显著下降（通过 `nav_stability_weight` 保持）
- **训练时长**: 约 1000-2000 iterations

---

## Phase 3: Student 蒸馏训练

### 目标

将 Teacher 的导航能力蒸馏到 Student，使其能够：

- 使用 **估计的 affordance** map（而非 GT）
- 匹配 Teacher 的导航性能
- 保持相机稳定性

### 配置参数

在 `hex_terrain_config.py` 中设置：

```python
class navigation:
    # Phase 控制
    enable_nav_reward = True           # ✓ 继续使用导航奖励
    freeze_student_policy = True       # ✓ 继续冻结底层策略
    use_adapter = True                 # ✓ 继续使用 Adapter
    use_gt_affordance = False          # ✓ 关键：Phase 3 使用估计 affordance

    # 其他参数与 Phase 2 相同
```

### 训练命令

```bash
# 加载 Phase 2 的 Teacher 模型
python legged_gym/scripts/train.py \
    --task=hex_terrain \
    --run_name=phase3_student \
    --load_run=phase2_teacher \
    --checkpoint=2000
```

### ⚠️ 关键实现要求

#### 1. 蒸馏损失（推荐升级）

Gemini 建议使用 Beta 分布对齐：

```python
# Teacher 和 Student 输出 Beta 分布参数
teacher_alpha, teacher_beta = teacher_network(obs, gt_affordance)
student_alpha, student_beta = student_network(obs, est_affordance)

# 蒸馏损失：KL 散度
from torch.distributions import Beta
teacher_dist = Beta(teacher_alpha, teacher_beta)
student_dist = Beta(student_alpha, student_beta)
distill_loss = torch.distributions.kl_divergence(teacher_dist, student_dist).mean()

# 或者：参数空间 MSE（备选方案）
distill_loss = F.mse_loss(student_alpha, teacher_alpha.detach()) + \
               F.mse_loss(student_beta, teacher_beta.detach())
```

#### 2. Affordance 估计器

需要一个视觉编码器从深度图估计 affordance：

```python
# 在 Phase 3 中
if self.nav_cfg.use_gt_affordance:
    affordance = self.get_gt_affordance()
else:
    affordance = self.affordance_estimator(depth_image)
```

### 预期结果

- **蒸馏质量**: Student 性能接近 Teacher (> 90%)
- **Sim-to-Real 能力**: 无需 GT affordance 即可导航
- **训练时长**: 约 1000-1500 iterations

---

## 配置速查表

| 参数                      | Phase 1 | Phase 2      | Phase 3      |
| ----------------------- | ------- | ------------ | ------------ |
| `enable_nav_reward`     | `False` | `True`       | `True`       |
| `freeze_student_policy` | `False` | `True`       | `True`       |
| `use_adapter`           | `False` | `True`       | `True`       |
| `use_gt_affordance`     | `True`  | `True`       | `False`      |
| `camera_stability` 权重   | `2.5`   | `0` (固定在参数中) | `0` (固定在参数中) |
| `nav_stability_weight`  | `0`     | `0.3` (可选)   | `0.3` (可选)   |
| 底层策略学习率                 | 正常      | `0` 或 `1e-6` | `0` 或 `1e-6` |

---

## 验证清单

### Phase 1 → Phase 2 切换

- [ ] Phase 1 模型已训练完成（model_2000.pt 存在）
- [ ] `enable_nav_reward = True` 已设置
- [ ] `freeze_student_policy = True` 已设置
- [ ] Runner 中实现了参数冻结逻辑
- [ ] `use_adapter = True` 已设置
- [ ] LocomotionAdapter 工作正常（测试转换结果）
- [ ] 目标生成模式已配置（goal_mode）

### Phase 2 → Phase 3 切换

- [ ] Phase 2 模型已训练完成
- [ ] `use_gt_affordance = False` 已设置
- [ ] Affordance 估计器网络已实现
- [ ] 蒸馏损失已添加到 Runner
- [ ] Teacher 输出 Beta 分布参数（如果使用）
- [ ] Student 输出 Beta 分布参数（如果使用）

---

## 常见问题

### Q1: Phase 2 训练时相机稳定性下降怎么办？

**A**: 增大 `nav_stability_weight` (例如从 0.3 → 0.5)，或者在导航奖励中添加更强的姿态惩罚。

### Q2: LocomotionAdapter 转换的速度指令不合理？

**A**: 调整以下参数：

- `adapter_distance_scale`: 控制速度与距离的关系（↑ = 更保守）
- `adapter_heading_gain`: 控制转向强度（↓ = 更平滑）
- `adapter_max_ang_vel`: 限制最大角速度（↓ = 更稳定）

### Q3: 如何验证底层策略是否真的被冻结？

**A**: 在训练日志中添加：

```python
grad_norm = sum(p.grad.norm() for p in actor.parameters() if p.grad is not None)
print(f"Actor grad norm: {grad_norm}")  # 应该为 0 或极小值
```

### Q4: Phase 3 Student 性能远低于 Teacher？

**A**: 可能原因：

1. Affordance 估计器质量差 → 改进视觉编码器
2. 蒸馏损失权重不足 → 增大蒸馏损失权重
3. 训练时间不够 → 延长训练至 2000+ iterations

---

## 下一步行动

1. **立即开始 Phase 1 训练**
   
   ```bash
   python legged_gym/scripts/train.py --task=hex_terrain --run_name=phase1_locomotion
   ```

2. **在 Phase 1 训练期间，实现以下功能**：
   
   - [ ] Runner 中的策略冻结机制
   - [ ] Teacher/Student 网络输出接口（subgoal, intensity）
   - [ ] Affordance 估计器网络
   - [ ] Beta 分布蒸馏损失（可选，P2 优先级）

3. **Phase 1 完成后，验证稳定性指标**，然后切换到 Phase 2

---

**作者**: Claude (GitHub Copilot)  
**版本**: v1.0  
**日期**: 2025-12-16
