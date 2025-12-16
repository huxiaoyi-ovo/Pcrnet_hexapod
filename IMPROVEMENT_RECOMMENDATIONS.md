# 训练稳定性改进建议

本文档记录针对当前 EGPO + Teacher-Student 架构的改进建议及实现状态。

---

## 1. 伪指令规范化 ⚠️ 部分实现

### 当前状态
✅ 已实现 `torch.nan_to_num()` 处理 NaN
❌ 缺少对极小距离的预防性 clamp

### 问题
当 `dist ≈ 0` 时，`target_dir = goal_vec / (dist + 1e-5)` 可能产生数值尖峰，导致指令抖动。

### 改进方案
```python
def _get_effective_commands(self):
    if getattr(self.nav_cfg, "enable_nav_reward", False):
        goal_vec = self.goal_buf
        dist = torch.norm(goal_vec, dim=1, keepdim=True)
        
        # 🔧 改进：对极小距离设置最小阈值
        dist_safe = torch.clamp(dist, min=0.1)  # 不低于 10cm
        target_dir = goal_vec / dist_safe
        
        # 🔧 改进：速度指令 clamp 到合理范围
        obs_commands = torch.zeros_like(self.commands)
        obs_commands[:, 0] = torch.clamp(target_dir[:, 0] * 0.6, -0.8, 0.8)
        obs_commands[:, 1] = torch.clamp(target_dir[:, 1] * 0.6, -0.8, 0.8)
        obs_commands[:, 2] = torch.clamp(heading_error * 1.5, -1.5, 1.5)
        
        return torch.nan_to_num(obs_commands, nan=0.0)
```

### 配置参数建议
```python
class navigation:
    min_command_distance = 0.1     # 最小指令计算距离 (m)
    max_lin_vel_command = 0.8      # 最大线速度指令 (m/s)
    max_ang_vel_command = 1.5      # 最大角速度指令 (rad/s)
```

---

## 2. 目标距离衰减 ✅ 已实现

### 当前状态
✅ 已在 `_get_effective_commands()` 中实现：
```python
speed_scale = torch.clamp(dist / 1.0, 0.2, 1.0)
```

### 优化建议
可调整衰减参数使其更平滑：

```python
class navigation:
    goal_slowdown_distance = 1.0   # 开始减速的距离 (m)
    goal_min_speed_ratio = 0.2     # 最小速度比例
    goal_slowdown_sharpness = 2.0  # 衰减锐度（tanh 替代 clamp）
```

```python
# 替代当前的 clamp，使用 tanh 实现平滑衰减
speed_scale = 0.5 * (1 + torch.tanh(
    self.nav_cfg.goal_slowdown_sharpness * (dist / self.nav_cfg.goal_slowdown_distance - 0.5)
))
speed_scale = torch.clamp(speed_scale, self.nav_cfg.goal_min_speed_ratio, 1.0)
```

---

## 3. 相机稳定性奖励平滑 ❌ 未实现（建议添加）

### 问题
当前直接使用瞬时 `base_ang_acc`，对传感器噪声和仿真数值误差敏感。

### 改进方案：滑动平均滤波

**在 `_init_extra_buffers()` 中添加**：
```python
# 角加速度历史 buffer（用于平滑）
self.ang_acc_history = torch.zeros(
    self.num_envs, 5, 3,  # 保存最近 5 步
    dtype=torch.float32,
    device=self.device
)
self.ang_acc_history_idx = 0
```

**在 `post_physics_step_separate()` 中更新**：
```python
# 更新角加速度历史
self.ang_acc_history[:, self.ang_acc_history_idx] = self.base_ang_acc.clone()
self.ang_acc_history_idx = (self.ang_acc_history_idx + 1) % 5
```

**修改 `_reward_camera_stability()`**：
```python
def _reward_camera_stability(self):
    # 使用平滑后的角加速度
    ang_acc_smooth = self.ang_acc_history.mean(dim=1)  # (N, 3)
    ang_jitter = torch.sum(torch.square(ang_acc_smooth[:, :2]), dim=1)
    
    # 其他部分保持不变
    ang_wobble = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    z_bobbing = torch.square(self.base_lin_acc[:, 2])
    
    # 🔧 改进：可调权重
    penalty = (ang_jitter * self.cfg.rewards.camera_jitter_weight +
               ang_wobble * self.cfg.rewards.camera_wobble_weight +
               z_bobbing * self.cfg.rewards.camera_bobbing_weight)
    
    return torch.exp(-penalty)
```

**配置参数**：
```python
class rewards:
    class scales:
        camera_stability = 2.5
        camera_jitter_weight = 0.05    # 抖动权重
        camera_wobble_weight = 0.5     # 晃动权重
        camera_bobbing_weight = 0.1    # 颠簸权重
```

---

## 4. 奖励归一化与抗梯度爆炸 ⚠️ 部分实现

### 当前状态
✅ 环境有 `clip_observations` 截断观测
❌ 奖励没有显式截断

### 问题
多个惩罚项叠加可能导致单步奖励 < -50，引起价值函数梯度爆炸。

### 改进方案

**在 `post_physics_step_separate()` 中添加**：
```python
# Phase 1: Locomotion rewards
if not getattr(self.nav_cfg, "enable_nav_reward", False):
    self.compute_reward()
    # 🔧 改进：奖励截断
    self.rew_buf = torch.clamp(
        self.rew_buf, 
        min=self.cfg.rewards.min_reward_clip,
        max=self.cfg.rewards.max_reward_clip
    )
```

**配置参数**：
```python
class rewards:
    min_reward_clip = -10.0   # 最小奖励（防止梯度爆炸）
    max_reward_clip = 10.0    # 最大奖励（防止过拟合单项奖励）
    reward_scale = 1.0        # 全局奖励缩放
```

---

## 5. 指令/观测一致性检查 ✅ 已注意

### 当前状态
✅ `num_privileged_obs = 240` 保持不变
✅ 维度计算已验证：
- obs_buf: 67
- obs_vgf_buf: 30
- obs_terrain_buf: 143
- 总计: 67 + 30 + 143 = 240 ✅

### 验证清单
- [x] 环境配置 `num_privileged_obs = 240`
- [x] Runner storage `[num_obs+30]` = [97]
- [x] 网络输入 `num_obs + 30 + terrain_latent` = 129
- [x] 基类 buffer 分配不会崩溃

---

## 6. 参数可调入口与监控 ⚠️ 部分实现

### 当前状态
✅ 主要权重在 config 中
❌ 缺少详细的稳定性监控指标

### 改进方案：TensorBoard 监控

**在 `post_physics_step_separate()` 中添加日志**：
```python
# 🔧 改进：添加稳定性监控指标
if self.log_dir is not None:
    # 相机稳定性指标
    self.extras["camera_stability"] = {
        "pitch_std": self.base_ang_vel[:, 0].std().item(),
        "roll_std": self.base_ang_vel[:, 1].std().item(),
        "ang_acc_rms": torch.sqrt(torch.mean(self.base_ang_acc[:, :2]**2)).item(),
        "z_acc_std": self.base_lin_acc[:, 2].std().item(),
    }
    
    # Phase 2/3: 导航指标
    if getattr(self.nav_cfg, "enable_nav_reward", False):
        self.extras["navigation"] = {
            "goal_distance_mean": torch.norm(self.goal_buf, dim=1).mean().item(),
            "command_vx_mean": obs_commands[:, 0].mean().item(),
            "command_vy_mean": obs_commands[:, 1].mean().item(),
            "command_omega_mean": obs_commands[:, 2].mean().item(),
        }
```

**在 Runner 的 `log()` 中记录**：
```python
def log(self, locs, width=80, pad=35):
    # ... 现有日志 ...
    
    # 🔧 改进：记录稳定性指标
    if "camera_stability" in self.extras:
        for key, val in self.extras["camera_stability"].items():
            self.writer.add_scalar(f'Camera/{key}', val, it)
    
    if "navigation" in self.extras:
        for key, val in self.extras["navigation"].items():
            self.writer.add_scalar(f'Navigation/{key}', val, it)
```

---

## 7. 配置参数 Sweep 建议

### Phase 1（运动训练）关键参数
```python
# 相机稳定性
camera_stability: [1.0, 2.5, 5.0]
camera_jitter_weight: [0.01, 0.05, 0.1]
camera_wobble_weight: [0.3, 0.5, 0.8]

# 基础运动
tracking_lin_vel: [1.5, 2.0, 3.0]
tracking_ang_vel: [1.0, 1.5, 2.0]
```

### Phase 2/3（导航训练）关键参数
```python
# 导航奖励
goal_approach_scale: [1.0, 2.0, 3.0]
nav_stability_weight: [0.1, 0.3, 0.5]

# 速度调节
goal_slowdown_distance: [0.5, 1.0, 2.0]
goal_min_speed_ratio: [0.1, 0.2, 0.3]
```

---

## 8. 实施优先级

### P0（训练前必须完成）
- [x] 维度验证（已完成）
- [ ] 奖励截断（防止梯度爆炸）
- [ ] 伪指令 clamp（防止数值尖峰）

### P1（Phase 1 训练期间完成）
- [ ] 相机稳定性平滑滤波
- [ ] TensorBoard 稳定性监控
- [ ] 配置参数可调化

### P2（Phase 2 前完成）
- [ ] 距离衰减平滑化（tanh 替代 clamp）
- [ ] 指令一致性单元测试
- [ ] 超参数 Sweep 脚本

---

## 9. 快速实施脚本

### 步骤 1：添加稳定性监控（立即）
```bash
# 在 hex_terrain.py 的 post_physics_step_separate() 末尾添加：
self.extras["camera_pitch_std"] = self.base_ang_vel[:, 0].std().item()
self.extras["camera_roll_std"] = self.base_ang_vel[:, 1].std().item()
```

### 步骤 2：添加奖励截断（训练前）
```bash
# 在 compute_reward() 后添加：
self.rew_buf = torch.clamp(self.rew_buf, -10.0, 10.0)
```

### 步骤 3：启动 Phase 1 训练
```bash
python legged_gym/scripts/train.py --task=hex_terrain --run_name=phase1_v2
```

### 步骤 4：监控 TensorBoard
```bash
tensorboard --logdir=logs/hex_terrain --port=6006
```

关注指标：
- `Camera/pitch_std` < 0.1 rad/s
- `Camera/roll_std` < 0.1 rad/s
- `Camera/ang_acc_rms` < 2.0 rad/s²
- `Rewards/camera_stability` > 0.8

---

**作者**: Claude (GitHub Copilot)  
**版本**: v1.0  
**日期**: 2025-12-16
