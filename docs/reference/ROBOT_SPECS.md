# Hex V4 六足机器人物理参数手册（硬指标版）

> 本文只保留“可直接从 URDF / 训练配置 / expert 源码读取”的硬指标；所有推算项、跨模块口径混用项先移除。
> 
> 数据来源（训练主线）：
> 
> - `resources/robots/hex_v4/urdf/hex_ground.urdf`
> - `legged_gym/envs/hex_v4/hex_ground_config.py`
> - `legged_gym/envs/hex_v4/hex_ground.py`
> - `legged_gym/envs/hex_v4/expert.py`
> 
> 整理日期：2026-01-30

---

## 1. URDF：机体/连杆硬指标

### 1.1 机身 (body)

| 参数                      | 数值                               | 单位    | 来源                                                |
| ----------------------- | --------------------------------:| ----- | ------------------------------------------------- |
| 质量                      | 5.5207                           | kg    | `resources/robots/hex_v4/urdf/hex_ground.urdf:11` |
| 碰撞盒尺寸 (x×y×z)           | 0.2 × 0.44 × 0.05                | m     | `resources/robots/hex_v4/urdf/hex_ground.urdf:27` |
| inertial 原点 (xyz)       | (0.0014417, 0.043319, 0.0041358) | m     | `resources/robots/hex_v4/urdf/hex_ground.urdf:10` |
| inertia (Ixx, Iyy, Izz) | (0.03035, 0.0081009, 0.036954)   | kg·m² | `resources/robots/hex_v4/urdf/hex_ground.urdf:12` |

### 1.2 单腿连杆（6腿对称：仅列质量）

| 连杆    | 单个质量 (kg)         | 数量  | 来源示例                                               |
| ----- | -----------------:| ---:| -------------------------------------------------- |
| thigh | 0.496689688064039 | 6   | `resources/robots/hex_v4/urdf/hex_ground.urdf:34`  |
| knee  | 0.108154994189176 | 6   | `resources/robots/hex_v4/urdf/hex_ground.urdf:66`  |
| ankle | 0.540605856314712 | 6   | `resources/robots/hex_v4/urdf/hex_ground.urdf:96`  |
| toe   | 0.01              | 6   | `resources/robots/hex_v4/urdf/hex_ground.urdf:126` |

### 1.3 足端接触几何（toe）

| 参数      | 数值    | 单位  | 来源                                                 |
| ------- | -----:| --- | -------------------------------------------------- |
| toe 球半径 | 0.009 | m   | `resources/robots/hex_v4/urdf/hex_ground.urdf:132` |

### 1.4 整机总质量（由 URDF 各 link mass 求和）

| 参数  | 数值      | 单位  |
| --- | -------:| --- |
| 总质量 | 12.4534 | kg  |

---

## 2. URDF：关节硬指标

> 全部转动关节统一：`effort=27 N·m`，`velocity=5 rad/s`（示例见 `resources/robots/hex_v4/urdf/hex_ground.urdf:60`）。

### 2.1 knee / ankle（各腿一致）

| 关节    | 位置限制 (rad)                    | 最大速度 (rad/s) | 最大扭矩 (N·m) | 来源示例                                               |
| ----- | ----------------------------- | ------------:| ----------:| -------------------------------------------------- |
| knee  | [-2.0943951024, 2.1816615650] | 5.0          | 27         | `resources/robots/hex_v4/urdf/hex_ground.urdf:90`  |
| ankle | [-2.7052603406, 2.4434609528] | 5.0          | 27         | `resources/robots/hex_v4/urdf/hex_ground.urdf:120` |

### 2.2 thigh（髋关节：各腿不对称）

| 腿   | 关节名        | 位置限制 (rad)                    | 来源                                                 |
| --- | ---------- | ----------------------------- | -------------------------------------------------- |
| RF  | j_rf_thigh | [-0.6981317008, 1.5707963268] | `resources/robots/hex_v4/urdf/hex_ground.urdf:55`  |
| RM  | j_rm_thigh | [-0.6981317008, 0.6981317008] | `resources/robots/hex_v4/urdf/hex_ground.urdf:171` |
| RB  | j_rb_thigh | [-1.5707963268, 0.6981317008] | `resources/robots/hex_v4/urdf/hex_ground.urdf:287` |
| LF  | j_lf_thigh | [-1.5707963268, 0.6981317008] | `resources/robots/hex_v4/urdf/hex_ground.urdf:403` |
| LM  | j_lm_thigh | [-0.6981317008, 0.6981317008] | `resources/robots/hex_v4/urdf/hex_ground.urdf:519` |
| LB  | j_lb_thigh | [-0.6981317008, 1.5707963268] | `resources/robots/hex_v4/urdf/hex_ground.urdf:635` |

---

## 3. 训练用 locomotion 指令范围（硬编码）

### 3.1 最大命令范围（训练配置上限）

| 命令          | 范围          | 单位    | 来源                                                |
| ----------- | ----------- | ----- | ------------------------------------------------- |
| lin_vel_x   | [-1.0, 1.0] | m/s   | `legged_gym/envs/hex_v4/hex_ground_config.py:217` |
| lin_vel_y   | [-1.5, 1.5] | m/s   | `legged_gym/envs/hex_v4/hex_ground_config.py:218` |
| ang_vel_yaw | [-2.0, 2.0] | rad/s | `legged_gym/envs/hex_v4/hex_ground_config.py:219` |

### 3.2 curriculum 初始命令范围（env 启动时覆盖）

| 命令          | 初始范围        | 单位    | 来源                                          |
| ----------- | ----------- | ----- | ------------------------------------------- |
| lin_vel_x   | [-0.6, 0.6] | m/s   | `legged_gym/envs/hex_v4/hex_ground.py:1014` |
| lin_vel_y   | [-0.9, 0.9] | m/s   | `legged_gym/envs/hex_v4/hex_ground.py:1015` |
| ang_vel_yaw | [-0.6, 0.6] | rad/s | `legged_gym/envs/hex_v4/hex_ground.py:1016` |

---

## 4. 控制/仿真时间尺度（硬编码）

| 参数                                    | 数值    | 单位    | 来源                                                |
| ------------------------------------- | -----:| ----- | ------------------------------------------------- |
| sim dt                                | 0.005 | s     | `legged_gym/envs/hex_v4/hex_ground_config.py:411` |
| control decimation                    | 4     | steps | `legged_gym/envs/hex_v4/hex_ground_config.py:273` |
| policy/control 频率（由 dt×decimation 得出） | 50    | Hz    | `legged_gym/envs/hex_v4/hex_ground_config.py:411` |
| expert dt                             | 0.02  | s     | `legged_gym/envs/hex_v4/expert.py:17`             |

---

## 5. expert：运动学硬阈值（不做任何推算）

| 参数                    | 数值           | 单位    | 来源                                     |
| --------------------- | ------------:| ----- | -------------------------------------- |
| 足端径向范围 (xy_len)       | [0.14, 0.24] | m     | `legged_gym/envs/hex_v4/expert.py:405` |
| 足端高度范围 (z)            | [-0.2, 0.1]  | m     | `legged_gym/envs/hex_v4/expert.py:409` |
| 足端方向角范围               | [-π/4, π/4]  | rad   | `legged_gym/envs/hex_v4/expert.py:411` |
| 腿间足端最小间距阈值            | 0.12         | m     | `legged_gym/envs/hex_v4/expert.py:424` |
| 抬腿/回摆速度上限 (v_z clamp) | 1.2          | (按实现) | `legged_gym/envs/hex_v4/expert.py:363` |

---

## 6. 训练物理随机化（硬编码）

| 参数             | 数值         | 单位  | 来源                                                |
| -------------- | ----------:| --- | ------------------------------------------------- |
| friction_range | [0.4, 0.8] | -   | `legged_gym/envs/hex_v4/hex_ground_config.py:299` |
