# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import isaacgym.terrain_utils as terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# Axis contract (classic, single source of truth):
# - World: +Y forward (corridor axis), +X lateral.
# - Tile grid: row i -> +Y (length), col j -> +X (width).
# - Heightfield array: height_field_raw[a, b] where a=length(+Y), b=width(+X).


def _lerp(a, b, t: float) -> float:
    return float(a + (b - a) * float(t))


def _param_range(params_easy: dict, params_hard: dict, key: str, default, t: float):
    return _lerp(params_easy.get(key, default), params_hard.get(key, default), t)


def _int_range(params_easy: dict, params_hard: dict, key: str, default, t: float) -> int:
    return int(round(_param_range(params_easy, params_hard, key, default, t)))


def debug_axis_terrain(terrain, difficulty: float, rng, cfg: LeggedRobotCfg.terrain, seed: int = None):
    """Only vary height along +Y (axis0), keep +X constant for axis calibration."""
    length_px = terrain.length
    width_px = terrain.width
    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale
    height_field = terrain.height_field_raw
    height_field[:] = 0

    params_easy = getattr(cfg, "scene_params_easy", {}) or {}
    params_hard = getattr(cfg, "scene_params_hard", {}) or {}
    step_count = max(1, _int_range(params_easy, params_hard, "step_count", 6, difficulty))
    step_height = _param_range(params_easy, params_hard, "step_height", 0.08, difficulty)
    edge_margin = _param_range(params_easy, params_hard, "edge_margin", 0.8, difficulty)

    margin_cells = max(0, int(round(edge_margin / h_scale)))
    usable_len = max(1, length_px - 2 * margin_cells)
    step_len = max(1, int(np.floor(usable_len / float(step_count))))
    height_cells = max(1, int(round(step_height / v_scale)))

    for k in range(step_count):
        y0 = margin_cells + k * step_len
        y1 = margin_cells + (k + 1) * step_len
        if k == step_count - 1:
            y1 = max(y1, length_px - margin_cells)
        y1 = min(y1, length_px)
        height_field[y0:y1, :] = np.maximum(height_field[y0:y1, :], (k + 1) * height_cells)

    terrain.meta = {
        "scene_type": "debug_axis",
        "params": {
            "step_count": step_count,
            "step_height": float(step_height),
            "edge_margin": float(edge_margin),
        },
        "layout_seed": int(seed or 0),
    }
    return terrain


def s1_corridor_gate_terrain(terrain, difficulty: float, rng, cfg: LeggedRobotCfg.terrain, seed: int = None):
    """S1 corridor + shrinking gates along +Y."""
    length_px = terrain.length
    width_px = terrain.width
    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale
    height_field = terrain.height_field_raw
    height_field[:] = 0

    params_easy = getattr(cfg, "scene_params_easy", {}) or {}
    params_hard = getattr(cfg, "scene_params_hard", {}) or {}

    corridor_width = _param_range(params_easy, params_hard, "corridor_width", 1.6, difficulty)
    door_width = _param_range(params_easy, params_hard, "gate_width", 0.9, difficulty)
    gate_count_min = int(min(params_easy.get("gate_count", 2), params_hard.get("gate_count", 3)))
    gate_count_max = int(max(params_easy.get("gate_count", 2), params_hard.get("gate_count", 3)))
    gate_count = int(rng.randint(gate_count_min, gate_count_max + 1))
    gate_length = _param_range(params_easy, params_hard, "gate_length", 1.0, difficulty)
    gate_length_jitter = _param_range(params_easy, params_hard, "gate_length_jitter", 0.2, difficulty)
    gate_spacing_min = _param_range(params_easy, params_hard, "gate_spacing_min", 0.6, difficulty)
    gate_margin_y = _param_range(params_easy, params_hard, "gate_margin_y", 0.8, difficulty)
    wall_height = _param_range(params_easy, params_hard, "wall_height_m", 0.5, difficulty)
    wall_thickness = _param_range(params_easy, params_hard, "wall_thickness_m", 0.16, difficulty)
    corridor_spawn_buffer = _param_range(params_easy, params_hard, "corridor_spawn_buffer", 0.6, difficulty)
    corridor_spawn_span = _param_range(params_easy, params_hard, "corridor_spawn_span", 2.0, difficulty)
    corridor_goal_min_offset = _param_range(params_easy, params_hard, "corridor_goal_min_offset", 2.0, difficulty)
    corridor_goal_buffer = _param_range(params_easy, params_hard, "corridor_goal_buffer", 0.6, difficulty)
    corridor_goal_margin = _param_range(params_easy, params_hard, "corridor_goal_margin", 0.2, difficulty)

    door_width = min(door_width, corridor_width)
    half_corridor = max(1, int(round(0.5 * corridor_width / h_scale)))
    half_door = max(1, int(round(0.5 * door_width / h_scale)))
    half_door = min(half_door, half_corridor)
    wall_cells = max(1, int(round(wall_height / v_scale)))
    wall_thickness_cells = max(1, int(round(wall_thickness / h_scale)))
    center_x = width_px // 2

    left = max(0, center_x - half_corridor)
    right = min(width_px, center_x + half_corridor)
    left_wall_start = max(0, left - wall_thickness_cells)
    left_wall_end = left
    right_wall_start = right
    right_wall_end = min(width_px, right + wall_thickness_cells)
    if left_wall_end > left_wall_start:
        height_field[:, left_wall_start:left_wall_end] = wall_cells
    if right_wall_end > right_wall_start:
        height_field[:, right_wall_start:right_wall_end] = wall_cells

    length_m = length_px * h_scale
    gate_length = max(0.1, gate_length + rng.uniform(-gate_length_jitter, gate_length_jitter))
    y_min = -0.5 * length_m + gate_margin_y
    y_max = 0.5 * length_m - gate_margin_y
    gate_centers = []
    for _ in range(gate_count):
        placed = False
        for _ in range(50):
            y_center = rng.uniform(y_min, y_max)
            if all(abs(y_center - c) >= (gate_spacing_min + 0.5 * gate_length) for c in gate_centers):
                gate_centers.append(y_center)
                placed = True
                break
        if not placed:
            break

    gates_meta = []
    for y_center in gate_centers:
        y0 = y_center - 0.5 * gate_length
        y1 = y_center + 0.5 * gate_length
        y0_idx = max(0, int(round((y0 + 0.5 * length_m) / h_scale)))
        y1_idx = min(length_px, int(round((y1 + 0.5 * length_m) / h_scale)))
        left_gate = max(left, center_x - half_door)
        right_gate = min(right, center_x + half_door)
        if left_gate > left:
            height_field[y0_idx:y1_idx, left:left_gate] = wall_cells
        if right > right_gate:
            height_field[y0_idx:y1_idx, right_gate:right] = wall_cells
        gates_meta.append({"y0": float(y_center), "length": float(gate_length), "door_width": float(door_width)})

    terrain.meta = {
        "scene_type": "s1_corridor_gate",
        "params": {
            "corridor_length": float(length_m),
            "corridor_width_nom": float(corridor_width),
            "corridor_gates": gates_meta,
            "corridor_x_center": 0.0,
            "corridor_spawn_buffer": float(corridor_spawn_buffer),
            "corridor_spawn_span": float(corridor_spawn_span),
            "corridor_goal_min_offset": float(corridor_goal_min_offset),
            "corridor_goal_buffer": float(corridor_goal_buffer),
            "corridor_goal_margin": float(corridor_goal_margin),
        },
        "layout_seed": int(seed or 0),
        "static_obstacles": [],
    }
    return terrain


def s2_forest_terrain(terrain, difficulty: float, rng, cfg: LeggedRobotCfg.terrain, seed: int = None):
    """S2 forest: poles + blocks with a clear band around x=0."""
    length_px = terrain.length
    width_px = terrain.width
    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale
    height_field = terrain.height_field_raw
    height_field[:] = 0

    params_easy = getattr(cfg, "scene_params_easy", {}) or {}
    params_hard = getattr(cfg, "scene_params_hard", {}) or {}

    count_min = _int_range(params_easy, params_hard, "count_min", 8, difficulty)
    count_max = _int_range(params_easy, params_hard, "count_max", 12, difficulty)
    if count_max < count_min:
        count_max = count_min
    total_count = int(rng.randint(count_min, count_max + 1))

    block_ratio = _param_range(params_easy, params_hard, "block_ratio", 0.2, difficulty)
    block_ratio = float(np.clip(block_ratio, 0.0, 1.0))
    num_blocks = int(round(total_count * block_ratio))
    num_poles = max(0, total_count - num_blocks)

    pole_radius_min = _param_range(params_easy, params_hard, "pole_radius_min", 0.12, difficulty)
    pole_radius_max = _param_range(params_easy, params_hard, "pole_radius_max", 0.18, difficulty)
    pole_height_min = _param_range(params_easy, params_hard, "pole_height_min", 0.30, difficulty)
    pole_height_max = _param_range(params_easy, params_hard, "pole_height_max", 0.35, difficulty)

    block_size_min = _param_range(params_easy, params_hard, "block_size_min", 0.28, difficulty)
    block_size_max = _param_range(params_easy, params_hard, "block_size_max", 0.40, difficulty)
    block_height_min = _param_range(params_easy, params_hard, "block_height_min", 0.30, difficulty)
    block_height_max = _param_range(params_easy, params_hard, "block_height_max", 0.35, difficulty)

    min_dist = _param_range(params_easy, params_hard, "min_dist", 0.45, difficulty)
    clear_band = max(
        float(params_easy.get("spawn_clear", 1.0)),
        float(params_hard.get("spawn_clear", 1.0)),
        float(params_easy.get("goal_clear", 1.0)),
        float(params_hard.get("goal_clear", 1.0)),
    )

    length_m = length_px * h_scale
    width_m = width_px * h_scale
    max_obs = max(pole_radius_max, 0.5 * block_size_max)
    margin = max_obs + 0.1

    centers = []
    shapes = []
    for _ in range(num_poles + num_blocks):
        placed = False
        for _ in range(60):
            x = rng.uniform(-0.5 * width_m + margin, 0.5 * width_m - margin)
            if abs(x) < 0.5 * clear_band:
                continue
            y = rng.uniform(-0.5 * length_m + margin, 0.5 * length_m - margin)
            if min_dist > 0.0:
                if any((x - cx) ** 2 + (y - cy) ** 2 < min_dist ** 2 for cx, cy in centers):
                    continue
            centers.append((x, y))
            placed = True
            break
        if not placed:
            centers.append((None, None))
    centers = [c for c in centers if c[0] is not None]

    shapes = (["pole"] * num_poles) + (["block"] * num_blocks)
    rng.shuffle(shapes)
    shapes = shapes[: len(centers)]
    actual_total = len(centers)
    actual_num_poles = int(sum(1 for s in shapes if s == "pole"))
    actual_num_blocks = int(actual_total - actual_num_poles)
    actual_block_ratio = float(actual_num_blocks / actual_total) if actual_total > 0 else 0.0

    for (x, y), shape in zip(centers, shapes):
        if shape == "pole":
            radius = rng.uniform(pole_radius_min, pole_radius_max)
            height_m = rng.uniform(pole_height_min, pole_height_max)
            rad_cells = max(1, int(round(radius / h_scale)))
            height_cells = max(1, int(round(height_m / v_scale)))
            cx = int(round((x + 0.5 * width_m) / h_scale))
            cy = int(round((y + 0.5 * length_m) / h_scale))
            x1 = max(0, cx - rad_cells)
            x2 = min(width_px, cx + rad_cells + 1)
            y1 = max(0, cy - rad_cells)
            y2 = min(length_px, cy + rad_cells + 1)
            xs = np.arange(x1, x2)
            ys = np.arange(y1, y2)
            grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
            mask = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= rad_cells * rad_cells
            patch = height_field[y1:y2, x1:x2]
            patch[mask] = np.maximum(patch[mask], height_cells)
            height_field[y1:y2, x1:x2] = patch
        else:
            size = rng.uniform(block_size_min, block_size_max)
            height_m = rng.uniform(block_height_min, block_height_max)
            hx = max(1, int(round(0.5 * size / h_scale)))
            hy = max(1, int(round(0.5 * size / h_scale)))
            height_cells = max(1, int(round(height_m / v_scale)))
            cx = int(round((x + 0.5 * width_m) / h_scale))
            cy = int(round((y + 0.5 * length_m) / h_scale))
            x1 = max(0, cx - hx)
            x2 = min(width_px, cx + hx + 1)
            y1 = max(0, cy - hy)
            y2 = min(length_px, cy + hy + 1)
            height_field[y1:y2, x1:x2] = np.maximum(height_field[y1:y2, x1:x2], height_cells)

    terrain.meta = {
        "scene_type": "s2_forest",
        "params": {
            "count_total": int(actual_total),
            "num_poles": int(actual_num_poles),
            "num_blocks": int(actual_num_blocks),
            "block_ratio": float(actual_block_ratio),
            "clear_band": float(clear_band),
        },
        "layout_seed": int(seed or 0),
        "static_obstacles": [],
    }
    return terrain

def fixed_layout_terrain(terrain, difficulty: float, cfg: LeggedRobotCfg.terrain):
    """固定布局地形：中心围挡+通道+低障碍"""
    # 基础网格信息
    width = terrain.width
    length = terrain.length
    h_scale = terrain.horizontal_scale
    v_scale = terrain.vertical_scale
    height_field = terrain.height_field_raw

    # 清空地形
    height_field[:] = 0

    # 结构参数（米）
    ring_half = getattr(cfg, "fixed_layout_ring_half_size", 2.2)
    wall_thickness = getattr(cfg, "fixed_layout_wall_thickness", 0.25)
    gap_min = getattr(cfg, "fixed_layout_gap_min", 0.45)
    gap_max = getattr(cfg, "fixed_layout_gap_max", 1.0)
    gap_buffer = getattr(cfg, "fixed_layout_gap_buffer", 0.1)
    robot_clearance = getattr(cfg, "fixed_layout_robot_clearance", 0.27)
    center_clearance = getattr(cfg, "fixed_layout_center_clearance", 0.6)

    high_min = getattr(cfg, "fixed_layout_high_height_min", 0.25)
    high_max = getattr(cfg, "fixed_layout_high_height_max", 0.35)
    low_min = getattr(cfg, "fixed_layout_low_height_min", 0.08)
    low_max = getattr(cfg, "fixed_layout_low_height_max", 0.12)
    cyl_radius_min = getattr(cfg, "fixed_layout_cyl_radius_min", 0.15)
    cyl_radius_max = getattr(cfg, "fixed_layout_cyl_radius_max", 0.25)
    cyl_offset = getattr(cfg, "fixed_layout_cyl_offset", 0.6)

    # 难度映射
    gap_width = gap_max - (gap_max - gap_min) * difficulty
    min_gap = 2.0 * robot_clearance + gap_buffer
    gap_width = max(gap_width, min_gap)
    passable_width = max(0.0, gap_width - 2.0 * robot_clearance - gap_buffer)
    high_h = np.random.uniform(high_min, high_min + (high_max - high_min) * difficulty)
    low_h = np.random.uniform(low_min, low_min + (low_max - low_min) * difficulty)

    # 转为离散单位
    ring_half_cells = max(1, int(round(ring_half / h_scale)))
    wall_cells = max(1, int(round(wall_thickness / h_scale)))
    gap_cells = max(1, int(round(gap_width / h_scale)))
    clearance_cells = max(0, int(round(center_clearance / h_scale)))
    high_cells = max(1, int(round(high_h / v_scale)))
    low_cells = max(1, int(round(low_h / v_scale)))

    cx = width // 2
    cy = length // 2
    x_min = max(0, cx - ring_half_cells)
    x_max = min(width, cx + ring_half_cells)
    y_min = max(0, cy - ring_half_cells)
    y_max = min(length, cy + ring_half_cells)

    max_gap = max(1, (x_max - x_min) - 2 * wall_cells - 2)
    gap_cells = min(gap_cells, max_gap)
    gap_half = max(1, gap_cells // 2)
    gap_offset_deg = getattr(cfg, "fixed_layout_gap_center_offset_deg", 0.0)
    offset_ratio = 0.0
    if gap_offset_deg > 0.0:
        base_deg = 45.0
        offset_ratio = np.sin(np.deg2rad(min(gap_offset_deg, base_deg))) / np.sin(np.deg2rad(base_deg))
    max_gap_shift_x = max(0, ((x_max - x_min) - 2 * wall_cells - gap_cells) // 2)
    max_gap_shift_y = max(0, ((y_max - y_min) - 2 * wall_cells - gap_cells) // 2)
    gap_shift_x = int(round(np.random.uniform(-1.0, 1.0) * max_gap_shift_x * offset_ratio))
    gap_shift_y = int(round(np.random.uniform(-1.0, 1.0) * max_gap_shift_y * offset_ratio))
    cx_gap = int(np.clip(cx + gap_shift_x, x_min + wall_cells + gap_half, x_max - wall_cells - gap_half))
    cy_gap = int(np.clip(cy + gap_shift_y, y_min + wall_cells + gap_half, y_max - wall_cells - gap_half))

    def fill_rect(x1, x2, y1, y2, height):
        if x2 > x1 and y2 > y1:
            height_field[x1:x2, y1:y2] = height

    # 上下围挡（留通道）
    top_y1 = max(y_min, y_max - wall_cells)
    top_y2 = y_max
    bot_y1 = y_min
    bot_y2 = min(y_max, y_min + wall_cells)
    left_x2 = max(x_min, cx_gap - gap_half)
    right_x1 = min(x_max, cx_gap + gap_half)
    fill_rect(x_min, left_x2, top_y1, top_y2, high_cells)
    fill_rect(right_x1, x_max, top_y1, top_y2, high_cells)
    fill_rect(x_min, left_x2, bot_y1, bot_y2, high_cells)
    fill_rect(right_x1, x_max, bot_y1, bot_y2, high_cells)

    # 左右围挡（留通道）
    left_x1 = x_min
    left_x2 = min(x_max, x_min + wall_cells)
    right_x1 = max(x_min, x_max - wall_cells)
    right_x2 = x_max
    lower_y2 = max(y_min, cy_gap - gap_half)
    upper_y1 = min(y_max, cy_gap + gap_half)
    fill_rect(left_x1, left_x2, y_min, lower_y2, high_cells)
    fill_rect(left_x1, left_x2, upper_y1, y_max, high_cells)
    fill_rect(right_x1, right_x2, y_min, lower_y2, high_cells)
    fill_rect(right_x1, right_x2, upper_y1, y_max, high_cells)

    # 中心清空区
    if clearance_cells > 0:
        c_x1 = max(0, cx - clearance_cells)
        c_x2 = min(width, cx + clearance_cells)
        c_y1 = max(0, cy - clearance_cells)
        c_y2 = min(length, cy + clearance_cells)
        height_field[c_x1:c_x2, c_y1:c_y2] = 0

    def fill_cylinder(center_x: float, center_y: float, radius_m: float, height_cells: int):
        rad_cells = max(1, int(round(radius_m / h_scale)))
        cx_i = int(round(cx + center_x / h_scale))
        cy_i = int(round(cy + center_y / h_scale))
        x1 = max(0, cx_i - rad_cells)
        x2 = min(width, cx_i + rad_cells + 1)
        y1 = max(0, cy_i - rad_cells)
        y2 = min(length, cy_i + rad_cells + 1)
        xs = np.arange(x1, x2)
        ys = np.arange(y1, y2)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        dist = (grid_x - cx_i) ** 2 + (grid_y - cy_i) ** 2
        mask = dist <= rad_cells * rad_cells
        patch = height_field[x1:x2, y1:y2]
        patch[mask] = np.maximum(patch[mask], height_cells)
        height_field[x1:x2, y1:y2] = patch

    def fill_box(center_x: float, center_y: float, size_x: float, size_y: float, height_cells: int):
        half_x = max(1, int(round(0.5 * size_x / h_scale)))
        half_y = max(1, int(round(0.5 * size_y / h_scale)))
        cx_i = int(round(cx + center_x / h_scale))
        cy_i = int(round(cy + center_y / h_scale))
        x1 = max(0, cx_i - half_x)
        x2 = min(width, cx_i + half_x + 1)
        y1 = max(0, cy_i - half_y)
        y2 = min(length, cy_i + half_y + 1)
        if x2 <= x1 or y2 <= y1:
            return
        height_field[x1:x2, y1:y2] = np.maximum(
            height_field[x1:x2, y1:y2],
            height_cells,
        )

    def sample_radius(scale: float = 1.0, radius_limit: float = None):
        radius_max = cyl_radius_min + (cyl_radius_max - cyl_radius_min) * scale
        if radius_limit is not None:
            radius_max = min(radius_max, radius_limit)
        if radius_max < cyl_radius_min:
            radius_max = cyl_radius_min
        radius = np.random.uniform(cyl_radius_min, radius_max)
        max_radius = max(0.05, ring_half - wall_thickness - cyl_offset - robot_clearance)
        return min(radius, max_radius)

    cyl_radius = sample_radius(difficulty)
    cyl_offset_use = max(cyl_offset, cyl_radius + robot_clearance)
    gap_offset_x = (cx_gap - cx) * h_scale
    gap_offset_y = (cy_gap - cy) * h_scale

    gap_centers = [
        (gap_offset_x, ring_half - wall_thickness - cyl_offset_use),   # north
        (gap_offset_x, -ring_half + wall_thickness + cyl_offset_use),  # south
        (ring_half - wall_thickness - cyl_offset_use, gap_offset_y),   # east
        (-ring_half + wall_thickness + cyl_offset_use, gap_offset_y),  # west
    ]
    gap_heights = [high_cells, low_cells, high_cells, low_cells]
    for (gx, gy), h_cells in zip(gap_centers, gap_heights):
        fill_cylinder(gx, gy, cyl_radius, h_cells)

    rand_num_min = getattr(cfg, "fixed_layout_rand_cyl_num_min", 0)
    rand_num_max = getattr(cfg, "fixed_layout_rand_cyl_num_max", 0)
    rand_box_prob = float(getattr(cfg, "fixed_layout_rand_shape_box_prob", 0.5))
    rand_box_min = float(getattr(cfg, "fixed_layout_rand_box_size_min", 0.25))
    rand_box_max = float(getattr(cfg, "fixed_layout_rand_box_size_max", 0.6))
    if rand_num_max > 0 and rand_num_max >= rand_num_min:
        num_float = rand_num_min + (rand_num_max - rand_num_min) * difficulty
        num_rand = int(round(num_float))
        rand_r_min_cfg = getattr(cfg, "fixed_layout_rand_cyl_r_min", 0.0)
        rand_r_max_cfg = getattr(cfg, "fixed_layout_rand_cyl_r_max", ring_half)
        box_max = rand_box_min + (rand_box_max - rand_box_min) * difficulty
        if passable_width > 0.0:
            box_max = min(box_max, passable_width)
            if box_max < rand_box_min:
                box_max = rand_box_min
        for _ in range(num_rand):
            radius_limit = 0.5 * passable_width if passable_width > 0.0 else None
            rand_radius = sample_radius(difficulty, radius_limit=radius_limit)
            min_r = max(center_clearance + robot_clearance + rand_radius, rand_r_min_cfg)
            max_r = min(ring_half - wall_thickness - robot_clearance - rand_radius, rand_r_max_cfg)
            if max_r <= min_r:
                continue
            angle = np.random.uniform(-np.pi, np.pi)
            radius = np.random.uniform(min_r, max_r)
            rx = radius * np.cos(angle)
            ry = radius * np.sin(angle)
            h_cells = high_cells if np.random.rand() < 0.5 else low_cells
            if np.random.rand() < rand_box_prob:
                size_x = np.random.uniform(rand_box_min, box_max)
                size_y = np.random.uniform(rand_box_min, box_max)
                fill_box(rx, ry, size_x, size_y, h_cells)
            else:
                fill_cylinder(rx, ry, rand_radius, h_cells)

    _apply_mixed_overlays(terrain, difficulty, cfg)
    return terrain


def _apply_mixed_overlays(terrain, difficulty: float, cfg: LeggedRobotCfg.terrain):
    if not getattr(cfg, "mixed_enable", False):
        return terrain

    if getattr(cfg, "mixed_roughness_enable", False):
        rough_scale = getattr(cfg, "mixed_roughness_scale", 0.0)
        if rough_scale > 0.0:
            rough_amp = rough_scale * (0.5 + 0.5 * difficulty)
            rough_downsample = getattr(cfg, "mixed_roughness_downsampled_scale", cfg.noise_downsampled_scale)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-rough_amp,
                max_height=rough_amp,
                step=0.005,
                downsampled_scale=rough_downsample,
            )

    if getattr(cfg, "mixed_obstacle_enable", False):
        num_min = getattr(cfg, "mixed_obstacle_num_rects_min", 4)
        num_max = getattr(cfg, "mixed_obstacle_num_rects_max", 12)
        num_rects = int(round(num_min + (num_max - num_min) * difficulty))

        min_size = getattr(cfg, "mixed_obstacle_min_size", 0.3)
        max_size = getattr(cfg, "mixed_obstacle_max_size", 0.6)
        h_min = getattr(cfg, "mixed_obstacle_height_min", 0.12)
        h_max = getattr(cfg, "mixed_obstacle_height_max", 0.28)

        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
        width, length = terrain.height_field_raw.shape

        min_cells = max(1, int(round(min_size / h_scale)))
        max_cells = max(min_cells, int(round(max_size / h_scale)))
        height_cells = max(1, int(round((h_min + (h_max - h_min) * difficulty) / v_scale)))

        for _ in range(num_rects):
            rect_w = np.random.randint(min_cells, max_cells + 1)
            rect_h = np.random.randint(min_cells, max_cells + 1)
            if rect_w >= width or rect_h >= length:
                continue
            x1 = np.random.randint(0, width - rect_w)
            y1 = np.random.randint(0, length - rect_h)
            x2 = x1 + rect_w
            y2 = y1 + rect_h
            terrain.height_field_raw[x1:x2, y1:y2] = np.maximum(
                terrain.height_field_raw[x1:x2, y1:y2],
                height_cells,
            )

    return terrain

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        if int(getattr(cfg, "num_rows", 0) or 0) <= 0 or int(getattr(cfg, "num_cols", 0) or 0) <= 0:
            raise RuntimeError(
                f"classic terrain requires num_rows/num_cols > 0, "
                f"got num_rows={getattr(cfg, 'num_rows', None)}, num_cols={getattr(cfg, 'num_cols', None)}"
            )

        cfg.terrain_proportions = np.array(cfg.terrain_proportions) / np.sum(cfg.terrain_proportions)
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.tile_meta = [[None for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self._check_axis_calib_if_needed()

        self.heightsamples = np.ascontiguousarray(self.height_field_raw, dtype=np.int16)
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )

    def _tile_seed(self, i: int, j: int) -> int:
        base = int(getattr(self.cfg, "terrain_seed", 0) or 0)
        return base + i * 1000 + j * 17
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            seed = self._tile_seed(i, j)
            rng = np.random.RandomState(seed)
            choice = rng.uniform(0.0, 1.0)
            difficulty = rng.uniform(0.0, 1.0)
            terrain = self.make_terrain(choice, difficulty, rng=rng, seed=seed, row=i, col=j)
            self.tile_meta[i][j] = getattr(terrain, "meta", None)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, (self.cfg.num_rows - 1))
                choice = j / self.cfg.num_cols + 0.001
                seed = self._tile_seed(i, j)
                rng = np.random.RandomState(seed)
                terrain = self.make_terrain(choice, difficulty, rng=rng, seed=seed, row=i, col=j)
                self.tile_meta[i][j] = getattr(terrain, "meta", None)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.get('type', None)
        if terrain_type is None:
            raise RuntimeError("selected terrain requires terrain_kwargs['type']")
        terrain_kwargs = {k: v for k, v in self.cfg.terrain_kwargs.items() if k != "type"}
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **terrain_kwargs)
            self.tile_meta[i][j] = getattr(terrain, "meta", None)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty, rng=None, seed=None, row=None, col=None):
        rng = rng if rng is not None else np.random
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        terrain_type = getattr(self.cfg, "terrain_type", None)
        if terrain_type:
            terrain_type = str(terrain_type).lower()
            if terrain_type in ("debug_axis", "calib_axis"):
                return debug_axis_terrain(terrain, difficulty, rng, self.cfg, seed=seed)
            if terrain_type in ("s1", "s1_corridor_gate"):
                return s1_corridor_gate_terrain(terrain, difficulty, rng, self.cfg, seed=seed)
            if terrain_type in ("s2", "s2_forest"):
                return s2_forest_terrain(terrain, difficulty, rng, self.cfg, seed=seed)
            raise RuntimeError(
                f"unsupported terrain_type={terrain_type}. "
                "supported: debug_axis, s1_corridor_gate, s2_forest"
            )
        if getattr(self.cfg, "fixed_layout_enable", False):
            return fixed_layout_terrain(terrain, difficulty, self.cfg)
        #核心参数
        slope = difficulty * 0.35
        # step_height = 0.05 + 0.18 * difficulty #这个递增对六足来说太快了
        step_height = 0.03 + 0.06 * difficulty
        # print("step_height=",step_height)
        # discrete_obstacles_height = 0.05 + difficulty * 0.2
        # discrete_obstacles_height = 0.05 + difficulty * 0.125  # 原值: 5cm→17.5cm 太高
        # 六足实际尺寸: 腿长20.2cm, 身高10cm, 合理障碍物应 <= 6cm
        height_min = getattr(self.cfg, "discrete_obstacles_height_min", 0.02)
        height_max = getattr(self.cfg, "discrete_obstacles_height_max", 0.06)
        discrete_obstacles_height = height_min + (height_max - height_min) * difficulty
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        #cfg参数
        w_env = getattr(self.cfg, "robot_envelope_width", 0.55)
        l_body = getattr(self.cfg, "robot_body_length", 0.40)
        v_nom = getattr(self.cfg, "nominal_speed", 0.5)
        t_react = getattr(self.cfg, "reaction_time", 0.4)

        gate_margin_max = getattr(self.cfg, "gate_margin_max", 0.50)
        gate_margin_min = getattr(self.cfg, "gate_margin_min", 0.05)
        gate_wall_height = getattr(self.cfg, "gate_wall_height", 0.60)
        gate_wall_thickness = getattr(self.cfg, "gate_wall_thickness", 0.20)
        gate_x_frac = getattr(self.cfg, "gate_x_frac", 0.65)
        gate_door_offset_max = getattr(self.cfg, "gate_door_offset_max", 0.60)

        slalom_wall_height = getattr(self.cfg, "slalom_wall_height", 0.60)
        slalom_wall_thickness = getattr(self.cfg, "slalom_wall_thickness", 0.20)
        slalom_corridor_width_scale = getattr(self.cfg, "slalom_corridor_width_scale", 2.8)
        slalom_pillar_size_x = getattr(self.cfg, "slalom_pillar_size_x", 0.45)
        slalom_pillar_size_y = getattr(self.cfg, "slalom_pillar_size_y", 0.35)
        slalom_num_pillars = getattr(self.cfg, "slalom_num_pillars", 6)
        gos_angle = getattr(self.cfg, "gate_on_slope_angle_deg", 20.0)
        # 根据 choice 选择地形类型
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3)

        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)

        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3)

        elif choice < self.proportions[4]:
            # === P2.1: 添加渐进式底噪 ===
            min_size = getattr(self.cfg, "discrete_obstacles_min_size", 1.0)
            max_size = getattr(self.cfg, "discrete_obstacles_max_size", 2.0)
            num_rects_min = getattr(self.cfg, "discrete_obstacles_num_rects_min", None)
            num_rects_max = getattr(self.cfg, "discrete_obstacles_num_rects_max", None)
            if num_rects_min is not None and num_rects_max is not None:
                num_rects = int(round(num_rects_min + (num_rects_max - num_rects_min) * difficulty))
            else:
                num_rects = getattr(self.cfg, "discrete_obstacles_num_rects", 20)
            platform_size = getattr(self.cfg, "discrete_obstacles_platform_size", 2.0)
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                min_size,
                max_size,
                num_rects,
                platform_size=platform_size,
            )
            
            # 获取底噪参数 (从cfg读取)
            noise_min = getattr(self.cfg, 'noise_amplitude_min', 0.005)  # 0.5cm
            noise_max = getattr(self.cfg, 'noise_amplitude_max', 0.020)  # 2cm
            noise_scale = getattr(self.cfg, 'noise_downsampled_scale', 0.3)
            
            # 渐进式底噪 (随难度增加)
            noise_amplitude = noise_min + (noise_max - noise_min) * difficulty
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-noise_amplitude,
                max_height=noise_amplitude,
                step=0.005,
                downsampled_scale=noise_scale
            )

        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=3.)

        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=2.)

        elif choice < self.proportions[7]:
            pit_terrain(terrain, depth=pit_depth, platform_size=2.)

        elif choice < self.proportions[8]:
            gate_terrain(
                terrain, difficulty=difficulty, w_env=w_env,
                margin_max=gate_margin_max, margin_min=gate_margin_min,
                wall_height=gate_wall_height, wall_thickness=gate_wall_thickness,
                gate_x_frac=gate_x_frac, door_offset_max=gate_door_offset_max,
                add_roughness=(difficulty > 0.7),
            )

        elif choice < self.proportions[9]:
            slalom_terrain(
                terrain, difficulty=difficulty, w_env=w_env,
                l_body=l_body, v_nom=v_nom, t_react=t_react,
                wall_height=slalom_wall_height, wall_thickness=slalom_wall_thickness,
                corridor_width_scale=slalom_corridor_width_scale,
                pillar_size_x=slalom_pillar_size_x, pillar_size_y=slalom_pillar_size_y,
                num_pillars=slalom_num_pillars,
                add_roughness=(difficulty > 0.7),
            )

        else:
            # 只在高难度引入 Gate-on-Slope（Stage-2）
            if difficulty < 0.7:
                # 低/中难度时，把这部分概率回退给 gate 或 slalom
                gate_terrain(
                    terrain, difficulty=difficulty, w_env=w_env,
                    margin_max=gate_margin_max, margin_min=gate_margin_min,
                    wall_height=gate_wall_height, wall_thickness=gate_wall_thickness,
                    gate_x_frac=gate_x_frac, door_offset_max=gate_door_offset_max,
                    add_roughness=False,
                )
            else:
                gate_on_slope_terrain(
                    terrain, difficulty=difficulty, w_env=w_env,
                    slope_angle_deg=gos_angle, platform_size=3.0,
                    margin_max=gate_margin_max, margin_min=gate_margin_min,
                    wall_height=gate_wall_height, wall_thickness=gate_wall_thickness,
                    gate_x_frac=gate_x_frac, door_offset_max=gate_door_offset_max,
                )


        return terrain
        
    def _resolve_tile_view(self, tile: np.ndarray):
        expected = (self.length_per_env_pixels, self.width_per_env_pixels)
        alt = (self.width_per_env_pixels, self.length_per_env_pixels)
        if tile.shape == expected:
            tile_view = tile
            map_mode = "none"
        elif tile.shape == alt:
            tile_view = tile.T
            map_mode = "transpose"
            if not getattr(self, "_tile_axis_warned", False):
                print(
                    f"[Warn] SubTerrain axis mismatch: tile_shape={tile.shape}, "
                    f"expected={expected}, alt={alt}. Using transpose."
                )
                self._tile_axis_warned = True
        else:
            raise RuntimeError(
                f"tile shape mismatch: got {tile.shape}, expected {expected} or {alt}"
            )
        if not getattr(self, "_tile_axis_logged", False):
            print(
                "[Terrain] axis_map env_dims="
                f"({self.env_length:.3f}m,{self.env_width:.3f}m) "
                f"px=({self.length_per_env_pixels},{self.width_per_env_pixels}) "
                f"tile_shape={tile.shape} map={map_mode}"
            )
            self._tile_axis_logged = True
        return tile_view, map_mode

    def _check_axis_calib_if_needed(self):
        terrain_type = getattr(self.cfg, "terrain_type", None)
        if terrain_type is None:
            return
        if str(terrain_type).lower() not in ("debug_axis", "calib_axis"):
            return
        # Check axis contract on the first tile (0,0).
        len_px = self.length_per_env_pixels
        wid_px = self.width_per_env_pixels
        if len_px <= 2 or wid_px <= 2:
            raise RuntimeError("axis calib failed: tile resolution too small.")
        row0 = self.border + max(1, int(0.2 * len_px))
        row1 = self.border + max(1, int(0.8 * len_px))
        col0 = self.border + max(1, int(0.5 * wid_px))
        col1 = self.border + max(1, int(0.25 * wid_px))
        col2 = self.border + max(1, int(0.75 * wid_px))
        row0 = min(row0, self.border + len_px - 1)
        row1 = min(row1, self.border + len_px - 1)
        col0 = min(col0, self.border + wid_px - 1)
        col1 = min(col1, self.border + wid_px - 1)
        col2 = min(col2, self.border + wid_px - 1)

        h_low = int(self.height_field_raw[row0, col0])
        h_high = int(self.height_field_raw[row1, col0])
        h_left = int(self.height_field_raw[row0, col1])
        h_right = int(self.height_field_raw[row0, col2])
        tol = 1
        if h_high <= h_low:
            raise RuntimeError(
                "axis calib failed: +Y not increasing. "
                f"h_low={h_low} h_high={h_high} row0={row0} row1={row1} "
                f"col0={col0} len_px={len_px} wid_px={wid_px}"
            )
        if abs(h_left - h_right) > tol:
            raise RuntimeError(
                "axis calib failed: +X not constant. "
                f"h_left={h_left} h_right={h_right} row0={row0} "
                f"col1={col1} col2={col2} tol={tol}"
            )

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        tile = terrain.height_field_raw
        tile_view, _ = self._resolve_tile_view(tile)
        self.height_field_raw[start_x: end_x, start_y:end_y] = tile_view

        env_origin_x = (j + 0.5) * self.env_width
        env_origin_y = (i + 0.5) * self.env_length
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(tile_view[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


# Navigation-Oriented Terrains 
def _m_to_px(terrain, m: float) -> int:
    return max(1, int(m / terrain.horizontal_scale))

def _m_to_h(terrain, m: float) -> int:
    return max(1, int(np.ceil(m / terrain.vertical_scale)))


def _center_slice(center: int, size: int, limit: int):
    """Return [start, end) with guaranteed end > start."""
    size = max(1, int(size))
    half = size // 2
    start = max(0, center - half)
    end = start + size
    if end > limit:
        end = limit
        start = max(0, end - size)
    # 兜底，保证非空
    if end <= start:
        end = min(limit, start + 1)
    return start, end


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
    x1, x2 = _center_slice(x_center, t_x, terrain.length)


    max_offset = min(door_offset_max, 0.5 * terrain.width * terrain.horizontal_scale - 0.6 * w_env)
    y_offset = (2.0 * np.random.rand() - 1.0) * max_offset * d
    y_center_m = 0.5 * terrain.width * terrain.horizontal_scale + y_offset
    y_center = int(y_center_m / terrain.horizontal_scale)

    y1 = max(0, y_center - gap_y // 2)
    y2 = min(terrain.width, y_center + gap_y // 2)

    # 保存基底高度（在加墙之前）
    base_slice = terrain.height_field_raw[x1:x2, :].copy()

    # 栅栏抬高
    terrain.height_field_raw[x1:x2, :] = np.maximum(terrain.height_field_raw[x1:x2, :], wall_h)

    # 门洞缺口：恢复为基底高度（保持坡/粗糙连续）
    terrain.height_field_raw[x1:x2, y1:y2] = base_slice[:, y1:y2]



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

        x1, x2 = _center_slice(xc, p_x, terrain.length)
        y1, y2 = _center_slice(yc, p_y, terrain.width)
        # 夹紧到走廊边界内
        y1 = max(y_left, y1)
        y2 = min(y_right, y2)
        # 兜底，保证非空
        if x2 <= x1 or y2 <= y1:
            continue


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

if __name__=='__main__':
    terrain_kwargs = {
        "type": "random_uniform_terrain",   # 对应 terrain_utils 里的函数名
        "min_height": -0.1,
        "max_height": 0.1,
        "step": 0.005,
        "downsampled_scale": 0.02
    }
    print(terrain_kwargs["type"])
