# 调试总结

- 轴向契约必须唯一：world +Y 前进；tile 行(i)=+Y(长度)；heightfield[a,b] 的 a=长度(+Y)、b=宽度(+X)。
- 轴映射只允许在 `Terrain.add_terrain_to_map()` 一处做（含 env_origin_z），禁止分散转置/交换。
- `debug_axis` 必须带自动验收：+Y 单调递增、+X 恒定；失败即硬报错，第一时间暴露轴向错位。
- Isaac Gym 版本可能导致 `SubTerrain.height_field_raw` 轴序互换；用“契约视图 + 显式转置”统一适配。
- S1 机体前进轴为 +Y 时，`heading_offset_rad` 必须为 0；否则会出现朝向垂直走廊并触发 reset-loop。

## S1 2048 并行稳定经验（详细版）

### 1) 轴向与契约（先决条件）

- world +Y 前进；tile 行=+Y（长度）；heightfield[a,b] 的 a=+Y、b=+X，且映射仅在 `Terrain.add_terrain_to_map()` 一处做。
- `debug_axis` 校验必须硬报错（+Y 单调、+X 恒定），避免“看似能跑但坐标错位”的隐性不稳定。

### 2) S1 走廊的区间解耦（spawn / gate / goal）

- gate 必须落在中段，显式排除 spawn 段与 goal 段，避免门洞落到出生/目标区域造成 reset-loop。
- gate 写入 meta 前按 y0 排序，便于复现与诊断。

### 3) spawn 放置逻辑（核心）

- x 采样必须扣掉 `clearance`：`x ∈ [x_center±(half_w - margin - clearance)]`，避免贴墙/进墙。
- y 采样额外避开 gate 禁区（forbidden intervals），即使 gate 理论上不在 spawn 段也加保险。
- 采样失败走 deterministic fallback：`x = x_center`，`y = y_start + spawn_buffer + 0.5*min(spawn_span, 1.0)`。
- 失败仅一次性 warn，保证长跑不中断；如 warn 频繁再升级硬报错。

### 4) 高并行的显存/带宽约束

- Teacher 路径不创建 depth buffer；仅 Student 或显式 `camera_enable` 才创建。
- `step_separate` 在相机关闭时不触碰 depth buffer；相机开启时惰性创建。
- separated 观测加噪按实际 buffer 维度切片，避免与 `env.num_observations` 口径不一致导致崩溃。

### 5) 语义一致性（避免隐性漂移）

- `goal_reached_threshold` 与 `reward_cfg.goal_reach_threshold` 必须一致（统一 0.1）。
- 若启用 `resample_on_reach=True`，强制一致性断言，防止策略行为和评测语义漂移。

### 6) 入口与最小验收流程

- 入口脚本不要硬拦截新 task，错误应来自 env/terrain_type 本身。
- 验收顺序：小并行固定 seed → `debug_axis` → 2048 并行长跑观察 episode_length≈0 的 reset-loop 是否消失。

## 简版 checklist（执行清单）

- 轴向契约唯一，`Terrain.add_terrain_to_map()` 为唯一映射入口
- `debug_axis` 校验通过（+Y 单调、+X 恒定）
- S1 gate 中段采样，spawn/goal 段解耦，gate meta 排序
- S1 spawn：x 扣 clearance，y 避 gate 禁区，失败走 deterministic fallback
- Teacher 不创建 depth buffer；step_separate 相机关闭不触碰 depth
- separated 观测加噪按实际 buffer 维度切片并校验长度
- goal 阈值统一 0.1，resample_on_reach 时强制一致
- 验收流程：小并行固定 seed → debug_axis → 2048 并行长跑
