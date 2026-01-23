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
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.scene_specs = None
        self.scene_generator = None
        self.scene_backend = None
        self.scene_use_heightfield = bool(getattr(cfg, "scene_use_heightfield", False))
        self._scene_heightfield_done = False
        if getattr(cfg, "scene_type", None) or getattr(cfg, "scene_types", None):
            from legged_gym.envs.hex_v4.scene_gen_v2.scene_generator import SceneGenerator
            from legged_gym.envs.hex_v4.scene_gen_v2.backend_heightfield import HeightfieldBackend
            env_dims = {"width": self.env_width, "length": self.env_length}
            robot_env = {"clearance": float(getattr(cfg, "scene_clearance", 0.27))}
            self.scene_generator = SceneGenerator(cfg, env_dims=env_dims, robot_envelope=robot_env)
            if not self.scene_use_heightfield:
                raise RuntimeError("scene_gen_v2 requires heightfield; legacy scene_manager disabled")
            self.scene_specs = [[None for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
            self.scene_backend = HeightfieldBackend(self.env_width, self.env_length,
                                                   self.cfg.horizontal_scale, self.cfg.vertical_scale)
            if cfg.curriculum:
                self.scene_heightfield_curriculum()
            else:
                self.scene_heightfield_randomized()
            self._scene_heightfield_done = True

        if not self._scene_heightfield_done:
            if self.scene_generator is not None:
                raise RuntimeError("scene_gen_v2 requires heightfield; set scene_use_heightfield=True")
            elif cfg.curriculum:
                self.curiculum()
            elif cfg.selected:
                self.selected_terrain()
            else:
                self.randomized_terrain()
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])#对于六足，难度等级太高
            difficulty = np.random.choice([0.1,0.3])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, (self.cfg.num_rows - 1))
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def scene_randomized(self):
        raise RuntimeError("legacy scene_manager disabled; use scene_gen_v2 heightfield")
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            difficulty = np.random.uniform(0.0, 1.0)
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            seed = self._scene_seed(i, j)
            scene_spec = None
            if self.scene_specs is not None:
                self.scene_specs[i][j] = scene_spec
            self.add_terrain_to_map(terrain, i, j)

    def scene_curriculum(self):
        raise RuntimeError("legacy scene_manager disabled; use scene_gen_v2 heightfield")
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, (self.cfg.num_rows - 1))
                terrain = terrain_utils.SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.length_per_env_pixels,
                    vertical_scale=self.cfg.vertical_scale,
                    horizontal_scale=self.cfg.horizontal_scale,
                )
                seed = self._scene_seed(i, j)
                scene_spec = None
                if self.scene_specs is not None:
                    self.scene_specs[i][j] = scene_spec
            self.add_terrain_to_map(terrain, i, j)

    def _scene_seed(self, i: int, j: int) -> int:
        base = int(getattr(self.cfg, "scene_seed", 0) or 0)
        return base + i * 1000 + j * 17

    def scene_heightfield_randomized(self):
        from legged_gym.envs.hex_v4.scene_gen_v2.quantizer import quantize_scene
        scene_type = getattr(self.cfg, "scene_type", None)
        scene_generator = self.scene_generator
        scene_backend = self.scene_backend
        if scene_generator is None or scene_backend is None:
            raise RuntimeError("scene_gen_v2 not initialized")
        for k in range(self.cfg.num_sub_terrains):
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            difficulty = np.random.uniform(0.0, 1.0)
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.length_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale,
            )
            seed = self._scene_seed(i, j)
            rng = np.random.RandomState(seed)
            scene_choice = scene_type
            if scene_choice is None:
                scene_choice = scene_generator._select_scene_type(rng, difficulty)
            scene = scene_generator.sample(scene_choice, difficulty, seed)
            scene = quantize_scene(scene, self.cfg.horizontal_scale, self.cfg.vertical_scale)
            terrain.height_field_raw[:] = scene_backend.render(scene)
            if self.scene_specs is not None:
                self.scene_specs[i][j] = scene
            self.add_terrain_to_map(terrain, i, j)

    def scene_heightfield_curriculum(self):
        from legged_gym.envs.hex_v4.scene_gen_v2.quantizer import quantize_scene
        scene_type = getattr(self.cfg, "scene_type", None)
        scene_generator = self.scene_generator
        scene_backend = self.scene_backend
        if scene_generator is None or scene_backend is None:
            raise RuntimeError("scene_gen_v2 not initialized")
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(1, (self.cfg.num_rows - 1))
                terrain = terrain_utils.SubTerrain(
                    "terrain",
                    width=self.width_per_env_pixels,
                    length=self.length_per_env_pixels,
                    vertical_scale=self.cfg.vertical_scale,
                    horizontal_scale=self.cfg.horizontal_scale,
                )
                seed = self._scene_seed(i, j)
                rng = np.random.RandomState(seed)
                scene_choice = scene_type
                if scene_choice is None:
                    scene_choice = scene_generator._select_scene_type(rng, difficulty)
                scene = scene_generator.sample(scene_choice, difficulty, seed)
                scene = quantize_scene(scene, self.cfg.horizontal_scale, self.cfg.vertical_scale)
                terrain.height_field_raw[:] = scene_backend.render(scene)
                if self.scene_specs is not None:
                    self.scene_specs[i][j] = scene
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
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
        

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
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
