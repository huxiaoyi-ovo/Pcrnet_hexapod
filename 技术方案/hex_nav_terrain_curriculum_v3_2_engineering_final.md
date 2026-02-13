# 六足机器人分层导航训练课程方案（V3.2 Engineering Final）
版本：v3.2（Kinematics-Constrained & Parametric）  
基础：V3.1（导航导向设计） + Expert Review（工程化修正）  
目标：在现有 `RL_hexapod_gym` 框架上，落地 **Gate / Slalom / Gate-on-Slope** 三类导航地形，并以 **课程学习（Terrain Curriculum）** 的方式逐步提升难度，用于后续高层规划器（Subgoal + λ）训练与论文实验。

---

## 0. 关键现状与本方案边界
- 你当前项目的地形生成入口在 `legged_gym/utils/terrain.py`，`Terrain.__init__()` 中的分支是：**curriculum 优先于 selected**（即使你在 cfg 里写了 `selected=True`，但若 `curriculum=True`，也不会走 `selected_terrain()`）。分支顺序是 `if cfg.curriculum: ... elif cfg.selected: ...`。  
- 你当前 `hex_terrain_config.py` 中 **设置了 `selected=True`**，但没有显式 `curriculum=False`，因此实际更可能仍在走 curriculum（需你按本方案修正）。  
- 本方案只覆盖“训练地形与课程”，不改变你已有的深度相机、affordance、以及高层训练脚本结构。

---

## 1. 机器人运动学包络（Kinematic Envelope）
在生成导航地形前，统一用“可通行包络”替代“机身宽度”，避免门洞/蛇形参数拍脑袋。

### 1.1 推荐参数（以 URDF/实物为准）
- 机身静态宽度：`W_body`（m）
- 机身静态长度：`L_body`（m）
- 单腿最大摆动外展量：`W_sw`（m）
- **可通行包络宽度**：`W_env = W_body + 2 * W_sw`（m）
- 标称速度：`v_nom`（m/s）
- 系统反应延迟（感知+决策+执行）：`T_react`（s）
- 最小反应距离：`D_react = v_nom * T_react`（m）

---

## 2. 新增导航地形图元（Terrain Primitives）
三类地形在 `legged_gym/utils/terrain.py` 中实现为函数，并在 `Terrain.make_terrain()` 中通过 `terrain_proportions` 映射到 curriculum 列（columns）。

### 2.1 Gate（门洞 / 窄缝）
训练目标：
- 强制高层输出**精确的朝向（Subgoal direction）**；
- 在难例中迫使 λ 降低（减小速度/摆幅），提升通过率。

参数化：
- 门洞宽度：`W_gap(d) = W_env + margin(d)`
- 余量：`margin(d) = margin_min + (margin_max - margin_min) * (1 - d)`  
  - `d=0` 宽门（更大余量）
  - `d=1` 极窄门（余量接近 `margin_min`）

工程化要点：
- Gate 采用“**横向栅栏 + 中央缺口**”实现，避免机器人从障碍外侧绕过（更像真实门洞）。
- 缺口中心允许随难度 lateral offset（门洞偏置），训练对齐能力。

### 2.2 Slalom（蛇形绕障）
训练目标：
- 训练连续转向与路径平滑；
- 通过“走廊 + 交替障碍柱”强制蛇形路径，而不是绕边走直线。

参数化（在代码内做有界映射）：
- 纵向间距（m）：`S_long(d) = S_max - (S_max - S_min) * d`（下界由 `L_body` 与 `D_react` 约束，避免物理不可行）
- 横向偏移幅值（m）：`Y_offset(d) = Y_max * (0.3 + 0.7*d)`（`Y_max` 由走廊宽度与 `W_env` 约束）

### 2.3 Gate-on-Slope（坡上门，复合难例）
训练目标（论文亮点 Case Study）：
- 同时考验：**调速（λ）** 与 **对齐（Subgoal）** 的协同  
- 在上坡（~20°）上必须对齐穿过窄门：速度太大→震动/打滑→撞门框；对齐不准→撞门框

---

## 3. 课程学习映射（Curriculum Schedule）
在你现有 `Terrain.curiculum()` 中，难度是 `difficulty = i / num_rows`，choice 是 `j / num_cols`。因此：
- **行（rows）控制难度 d**
- **列（cols）控制地形类型（由 proportions 的累计阈值决定）**

推荐：`num_rows = 10`，Stage 划分：
- Stage 0：rows 0–2（d≈0.0–0.2）
- Stage 1：rows 3–6（d≈0.3–0.6）
- Stage 2：rows 7–9（d≈0.7–0.9）

---

## 4. 工程落地（可直接拷贝进项目）

### 4.1 修改 1：扩展 `legged_gym/utils/terrain.py`
你当前 `make_terrain()` 只到 gap/pit（最后 else 是 pit）。我们将其扩展为：
- 保留原 0–6 类（slope/rough/stairs/discrete/stepping/gap/pit）
- 新增 3 类（gate/slalom/gate_on_slope）

#### A) 新增函数：Gate / Slalom / Gate-on-Slope
把下面代码粘贴到 `legged_gym/utils/terrain.py`（建议放在 `pit_terrain` 后面）：

```python
# =========================
# Navigation-Oriented Terrains (V3.2)
# =========================

def _m_to_px(terrain, m: float) -> int:
    return max(1, int(m / terrain.horizontal_scale))

def _m_to_h(terrain, m: float) -> int:
    return int(m / terrain.vertical_scale)

def gate_terrain(
    terrain,
    difficulty: float,
    w_env: float,
    margin_max: float = 0.50,
    margin_min: float = 0.05,
    wall_height: float = 0.60,
    wall_thickness: float = 0.20,
    gate_x_frac: float = 0.65,
    door_offset_max: float = 0.60,
    add_roughness: bool = False,
    rough_min: float = -0.02,
    rough_max: float = 0.02,
):
    """Gate = 横向栅栏 + 中央缺口（门洞），强制通过缺口而不是绕行。"""
    d = float(np.clip(difficulty, 0.0, 1.0))

    if add_roughness:
        terrain_utils.random_uniform_terrain(
            terrain, min_height=rough_min, max_height=rough_max,
            step=0.005, downsampled_scale=0.2
        )

    margin = margin_min + (margin_max - margin_min) * (1.0 - d)
    w_gap = w_env + margin

    wall_h = _m_to_h(terrain, wall_height)
    t_x = _m_to_px(terrain, wall_thickness)
    gap_y = _m_to_px(terrain, w_gap)

    x_center = int(terrain.length * gate_x_frac)
    x1 = max(0, x_center - t_x // 2)
    x2 = min(terrain.length, x_center + t_x // 2)

    max_offset = min(door_offset_max, 0.5 * terrain.width * terrain.horizontal_scale - 0.6 * w_env)
    y_offset = (2.0 * np.random.rand() - 1.0) * max_offset * d
    y_center_m = 0.5 * terrain.width * terrain.horizontal_scale + y_offset
    y_center = int(y_center_m / terrain.horizontal_scale)

    y1 = max(0, y_center - gap_y // 2)
    y2 = min(terrain.width, y_center + gap_y // 2)

    # 栅栏
    terrain.height_field_raw[x1:x2, :] = np.maximum(terrain.height_field_raw[x1:x2, :], wall_h)
    # 门洞缺口
    terrain.height_field_raw[x1:x2, y1:y2] = 0


def slalom_terrain(
    terrain,
    difficulty: float,
    w_env: float,
    l_body: float,
    v_nom: float,
    t_react: float,
    wall_height: float = 0.60,
    wall_thickness: float = 0.20,
    corridor_width_scale: float = 2.8,
    pillar_size_x: float = 0.45,
    pillar_size_y: float = 0.35,
    num_pillars: int = 6,
    add_roughness: bool = False,
    rough_min: float = -0.02,
    rough_max: float = 0.02,
):
    """Slalom = 走廊 + 交替障碍柱。走廊两侧加墙避免绕边直走。"""
    d = float(np.clip(difficulty, 0.0, 1.0))
    d_react = float(v_nom) * float(t_react)

    if add_roughness:
        terrain_utils.random_uniform_terrain(
            terrain, min_height=rough_min, max_height=rough_max,
            step=0.005, downsampled_scale=0.2
        )

    wall_h = _m_to_h(terrain, wall_height)
    wall_t = _m_to_px(terrain, wall_thickness)

    corridor_width = max(corridor_width_scale * w_env, w_env + 0.30)
    corridor_px = _m_to_px(terrain, corridor_width)

    y_center = terrain.width // 2
    y_left = max(0, y_center - corridor_px // 2)
    y_right = min(terrain.width, y_center + corridor_px // 2)

    # 走廊侧墙（把走廊外全抬高）
    terrain.height_field_raw[:, :max(0, y_left - wall_t)] = np.maximum(
        terrain.height_field_raw[:, :max(0, y_left - wall_t)], wall_h
    )
    terrain.height_field_raw[:, min(terrain.width, y_right + wall_t):] = np.maximum(
        terrain.height_field_raw[:, min(terrain.width, y_right + wall_t):], wall_h
    )

    p_x = _m_to_px(terrain, pillar_size_x)
    p_y = _m_to_px(terrain, pillar_size_y)

    s_min = max(1.2 * l_body + 0.8 * d_react, 1.4 * l_body)
    s_max = 2.4 * l_body + 2.0 * d_react
    spacing_m = s_max - (s_max - s_min) * d
    spacing_px = _m_to_px(terrain, spacing_m)

    safety = 0.5 * w_env + 0.05
    max_offset_m = max(0.05, 0.5 * corridor_width - safety)
    offset_m = max_offset_m * (0.3 + 0.7 * d)
    offset_px = _m_to_px(terrain, offset_m)

    x_start = _m_to_px(terrain, 1.0)
    x_end = terrain.length - _m_to_px(terrain, 1.0)
    xs = list(range(x_start, x_end, spacing_px))[:num_pillars]

    for k, xc in enumerate(xs):
        sign = -1 if (k % 2 == 0) else 1
        yc = y_center + sign * offset_px

        x1 = max(0, xc - p_x // 2)
        x2 = min(terrain.length, xc + p_x // 2)
        y1 = max(y_left, yc - p_y // 2)
        y2 = min(y_right, yc + p_y // 2)

        terrain.height_field_raw[x1:x2, y1:y2] = np.maximum(
            terrain.height_field_raw[x1:x2, y1:y2], wall_h
        )


def gate_on_slope_terrain(
    terrain,
    difficulty: float,
    w_env: float,
    slope_angle_deg: float = 20.0,
    platform_size: float = 3.0,
    **gate_kwargs,
):
    """Gate-on-Slope = 在坡面上叠加 Gate（复合难例）。"""
    d = float(np.clip(difficulty, 0.0, 1.0))
    slope = np.tan(np.deg2rad(slope_angle_deg)) * (0.7 + 0.3 * d)
    terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)

    gate_terrain(
        terrain,
        difficulty=d,
        w_env=w_env,
        add_roughness=(d > 0.7),
        **gate_kwargs,
    )
```

#### B) 修改 `Terrain.make_terrain()` 分支
在 `Terrain.make_terrain()` 里，增加 gate/slalom/gate_on_slope 三个分支（并从 cfg 读取参数）。你可以按我之前的 “B) 修改 `Terrain.make_terrain()`” 段落粘贴实现。

---

### 4.2 修改 2：更新 `legged_gym/envs/hex_v4/hex_terrain_config.py`
把 `class terrain(...)` 按下面片段替换（重点：`curriculum=True`, `selected=False`，以及 proportions 扩展到 11 项）：

```python
class terrain(LeggedRobotCfg.terrain):
    mesh_type = "trimesh"
    border_size = 1.0
    terrain_length = 8.0
    terrain_width = 8.0

    curriculum = True
    selected = False
    terrain_kwargs = None

    num_rows = 10
    num_cols = 20

    max_init_terrain_level = 1
    measure_heights = True
    measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    measured_points_y = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    slope_treshold = 0.4

    robot_body_width = 0.25
    robot_body_length = 0.40
    robot_swing_abduction = 0.15
    robot_envelope_width = robot_body_width + 2.0 * robot_swing_abduction

    nominal_speed = 0.5
    reaction_time = 0.4

    gate_margin_max = 0.50
    gate_margin_min = 0.05
    gate_wall_height = 0.60
    gate_wall_thickness = 0.20
    gate_x_frac = 0.65
    gate_door_offset_max = 0.60

    slalom_wall_height = 0.60
    slalom_wall_thickness = 0.20
    slalom_corridor_width_scale = 2.8
    slalom_pillar_size_x = 0.45
    slalom_pillar_size_y = 0.35
    slalom_num_pillars = 6

    gate_on_slope_angle_deg = 20.0

    terrain_proportions = [
        0.10,  # smooth slope
        0.10,  # rough slope
        0.10,  # stairs down
        0.10,  # stairs up
        0.25,  # discrete obstacles
        0.00,  # stepping stones
        0.00,  # gap
        0.00,  # pit
        0.20,  # gate
        0.15,  # slalom
        0.10,  # gate_on_slope
    ]
```

---

## 5. 一步一步改项目（最短闭环）
1) **改 `terrain.py`**：加 3 个新函数 + 扩展 `make_terrain()` 分支  
2) **改 `hex_terrain_config.py`**：强制 `curriculum=True`, `selected=False`，并写入包络/参数  
3) **最小验收**：用 headless + 少量 env 跑 50 步，确认不崩（可参考你已有的 `legged_gym/tests/check_depth_headless.py` 模式）  
4) 再开始 low-level / high-level 训练

---

## 6. 论文实验建议（与 V3.2 地形强绑定）
- Ablation 1：无 Gate/Slalom vs 有 Gate/Slalom（验证“导航导向地形”的必要性）
- Ablation 2：无 Gate-on-Slope vs 有 Gate-on-Slope（验证 λ 与 Subgoal 协同调节）
- 指标：
  - Through-Gate Success Rate（门洞通过率）
  - Slalom Completion Time（蛇形完成时间）
  - Path Smoothness（yaw rate、曲率或 jerk）
  - Collision count（门框/柱体碰撞次数）
