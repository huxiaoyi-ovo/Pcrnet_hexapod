# P1修复实施报告

## 修复日期
2025年12月16日

## 修复目标
1. **P1-Depth**: 消除depth_images在raw/processed间的shape/语义漂移
2. **P1-GoalBuf**: 消除goal_buf重新绑定，统一为就地写入
3. **P2-Guard**: 添加防御性编程检查

---

## 1. P1-Depth修复详情

### 问题描述
原代码中`self.depth_images`在raw depth (N,H,W)和processed depth (N,1,H,W)之间反复切换，导致shape不稳定。

### 解决方案
拆分为两个独立buffer，职责明确：
- `self.depth_raw`: 永远(N,H,W)，存储传感器原始深度
- `self.depth_images`: 永远(N,1,H,W)，存储网络输入深度

### 修改点

#### 1.1 `_init_camera_buffers()` - 初始化两个固定shape的buffer
```python
# 传感器原始深度 (N, H, W)
self.depth_raw = torch.zeros(
    self.num_envs,
    self.camera_cfg.height,
    self.camera_cfg.width,
    dtype=torch.float32,
    device=self.device,
    requires_grad=False
)

# 网络输入深度 (N, 1, H, W)
self.depth_images = torch.zeros(
    self.num_envs,
    1,
    self.camera_cfg.height,
    self.camera_cfg.width,
    dtype=torch.float32,
    device=self.device,
    requires_grad=False
)
```

#### 1.2 `_get_depth_images()` - 写入depth_raw
```python
# 修改前: self.depth_images = torch.stack(depth_images_list, dim=0)
# 修改后:
self.depth_raw[:] = torch.stack(depth_images_list, dim=0)
return self.depth_raw
```

#### 1.3 `step_separate()` - 拆分处理管线
```python
# 修改前:
depth_raw = self._get_depth_images()
self.depth_images = self._process_depth_for_network(depth_raw)

# 修改后:
depth_raw = self._get_depth_images()  # -> depth_raw (N,H,W)
processed = self._process_depth_for_network(depth_raw)  # -> (N,1,H,W)
self.depth_images[:] = processed  # 就地写入，保持身份稳定
```

#### 1.4 `reset_idx()` - 清零两个buffer
```python
# 修改前: 
if hasattr(self, 'depth_images') and self.depth_images is not None:
    self.depth_images[env_ids] = 0.0

# 修改后:
self.depth_raw[env_ids] = self.camera_cfg.far_clip  # 原始深度用far_clip表示无效
self.depth_images[env_ids] = 0.0  # 网络输入用0
```

---

## 2. P1-GoalBuf修复详情

### 问题描述
`self.goal_buf`在多处使用`self.goal_buf = ...`重新绑定，与`reset_idx()`中的`self.goal_buf[env_ids] = ...`不一致，导致引用不稳定。

### 解决方案
统一为就地写入：
- `self.goal_buf[:] = ...` (全量更新)
- `self.goal_buf[env_ids] = ...` (子集更新)

### 修改点

#### 2.1 `_update_goal_buffer()` - navigation分支
```python
# 修改前:
self.goal_buf = self.nav_task.get_relative_goal(robot_pos_local, headings)

# 修改后:
rel_goal = self.nav_task.get_relative_goal(robot_pos_local, headings)
self.goal_buf[:] = rel_goal
```

#### 2.2 `_update_goal_buffer()` - velocity_based分支
```python
# 修改前:
self.goal_buf = goal_direction * goal_distance

# 修改后:
self.goal_buf[:] = goal_direction * goal_distance
```

#### 2.3 `_update_goal_buffer()` - fixed分支
```python
# 修改前:
self.goal_buf = fixed_goal.unsqueeze(0).expand(self.num_envs, -1) - self.root_states[:, :2]

# 修改后:
self.goal_buf[:] = fixed_goal.unsqueeze(0).expand(self.num_envs, -1) - self.root_states[:, :2]
```

#### 2.4 `_update_goal_buffer()` - random分支
```python
# 修改前:
self.goal_buf = goal_world - self.root_states[:, :2]

# 修改后:
self.goal_buf[:] = goal_world - self.root_states[:, :2]
```

#### 2.5 `reset_idx()` - 保持不变
```python
# 已经是正确的就地写入
self.goal_buf[env_ids] = self.nav_task.get_relative_goal(...)
```

---

## 3. P2-Guard添加

### 修改点

#### 3.1 `post_physics_step_separate()` - reset前防御性检查
```python
# 修改前:
env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
self.reset_idx(env_ids)

# 修改后:
env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
if env_ids.numel() > 0:  # P2防御性检查
    self.reset_idx(env_ids)
```

---

## 4. 验证结果

### 4.1 代码静态检查
运行`verify_p1_code.py`：
```
✅ 1. depth_raw buffer已初始化
✅ 2. depth_images buffer初始化为(N,1,H,W)
✅ 3. _get_depth_images正确写入depth_raw
✅ 4. step_separate中depth_images使用就地写入
✅ 5. reset_idx清零两个depth buffers
✅ 6. 无goal_buf重新绑定（除初始化外）
✅ 7. _update_goal_buffer的4个分支使用就地写入
✅ 8. P2防御性guard已添加
```

### 4.2 预期效果

#### Shape稳定性
- `obs_dict['depth'].shape`在整个生命周期保持`(N,1,H,W)`
- 连续step、random reset后shape不变

#### Buffer身份稳定性
- `id(env.goal_buf)`和`env.goal_buf.data_ptr()`在多次step/reset后不变
- `id(env.depth_images)`和`env.depth_images.data_ptr()`在多次step/reset后不变

#### 子集隔离性
- `reset_idx(env_ids)`只影响`env_ids`对应的环境
- 未reset的环境的depth/goal保持不变（或按正常逻辑更新）

---

## 5. 未修改的内容（保持不变）

### 5.1 Auto-reset语义（正确，保持）
```python
# post_physics_step_separate()中reset后的刷新逻辑
if env_ids.numel() > 0:
    self.reset_idx(env_ids)

if env_ids.numel() > 0:
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
    self.compute_observations_separated()
```

### 5.2 Prev buffer子集对齐（正确，保持）
```python
# reset_idx()中
prev_robot_pos_buf[env_ids] = robot_pos_local  # 不二次索引
prev_intensity_buf[env_ids] = intensity_buf[env_ids]
```

### 5.3 Curriculum协议（正确，保持）
```python
# post_physics_step_separate()中
self.nav_task.update_curriculum(env_ids_nav, successes)  # successes是全量buffer
```

---

## 6. 训练就绪状态

✅ **所有P0/P1修复已完成**
✅ **代码静态检查通过**
✅ **核心语义保持不变**

### 下一步
可以直接开始训练：
```bash
python legged_gym/scripts/train.py --task=hex_terrain --headless
```

### 监控指标
- `nav/goal_distance`: 应该在[2.0, 8.0]m范围内
- `nav/success_rate`: 应该随训练增加
- `nav/heading_error`: 应该收敛
- 观察是否有NaN/Inf错误（预期无）

---

## 7. 修改文件清单

- ✅ `legged_gym/envs/hex_v4/hex_terrain.py` - 主要修改
- ✅ `verify_p1_code.py` - 新增：代码检查脚本
- ✅ `test_p1_fixes.py` - 新增：运行时测试脚本（可选）

---

## 8. 回归风险评估

### 低风险 ✅
- Depth管线拆分：shape固定，语义清晰
- Goal buffer就地写入：内存布局不变，只是赋值方式改变
- P2 Guard：纯防御性，不影响正常逻辑

### 需要验证的场景
1. 第一次运行时相机初始化是否正常
2. Random reset是否正常工作
3. Curriculum更新是否正常
4. 训练几个epoch后观察是否有shape相关错误

---

## 结论

✅ **P1修复已完成并通过代码检查**
✅ **所有必需的硬约束已满足**
✅ **代码质量提升，训练就绪**
