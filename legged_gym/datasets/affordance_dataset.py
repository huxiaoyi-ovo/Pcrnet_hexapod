# -*- coding: utf-8 -*-
"""
datasets/affordance_dataset.py - Affordance 数据集生成与加载脚本（第一视角版）

功能:
1. 第一视角深度生成（基于固定布局障碍场景，含 POV/畸变/噪声）。
2. 生成标签：障碍占用图（occupancy）与可通行间距（passable_gap）。
3. PyTorch Dataset：加载样本并进行轻量增强。

用法:
    1. 生成数据:
       python datasets/affordance_dataset.py --num_samples 20000 --save_dir data/processed --visualize

    2. 训练调用:
       from datasets.affordance_dataset import AffordanceDataset
       dataset = AffordanceDataset(data_path='data/processed/affordance_data.pt', transform=True)
"""

import os
import math
import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, distance_transform_edt

# 默认输出尺寸（如 config 中有 output_size，会覆盖）
IMG_SIZE = 128
MAP_SIZE = 16


def load_hex_ground_cfg():
    try:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        exec_globals = {
            "LEGGED_GYM_ROOT_DIR": base_path,
            "LEGGED_GYM_ENVS_DIR": os.path.join(base_path, "legged_gym", "envs"),
        }

        def _load_code(path, skip_prefixes):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            filtered = []
            for line in lines:
                stripped = line.strip()
                if any(stripped.startswith(prefix) for prefix in skip_prefixes):
                    continue
                filtered.append(line)
            return "".join(filtered)

        base_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'envs', 'base', 'base_config.py')
        robot_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'envs', 'base', 'legged_robot_config.py')
        hex_cfg_path = os.path.join(os.path.dirname(__file__), '..', 'envs', 'hex_v4', 'hex_ground_config.py')

        base_cfg_path = os.path.abspath(base_cfg_path)
        robot_cfg_path = os.path.abspath(robot_cfg_path)
        hex_cfg_path = os.path.abspath(hex_cfg_path)

        base_code = _load_code(base_cfg_path, skip_prefixes=())
        exec(base_code, exec_globals)

        robot_code = _load_code(robot_cfg_path, skip_prefixes=("from .base_config", "from legged_gym"))
        exec(robot_code, exec_globals)

        hex_code = _load_code(hex_cfg_path, skip_prefixes=("from legged_gym",))
        exec(hex_code, exec_globals)

        cfg_cls = exec_globals.get("HexGroundCfg")
        if cfg_cls is None:
            raise RuntimeError("HexGroundCfg not found after exec")
        return cfg_cls()
    except Exception as exc:
        print(f"[Dataset] Warning: load HexGroundCfg failed: {exc}")
        return None


def _get_camera_cfg(cfg):
    cam = None
    if cfg is not None and hasattr(cfg, "sensor") and hasattr(cfg.sensor, "depth_camera"):
        cam = cfg.sensor.depth_camera
    # fallback defaults
    return {
        "enable": getattr(cam, "enable", True),
        "width": getattr(cam, "width", IMG_SIZE),
        "height": getattr(cam, "height", IMG_SIZE),
        "horizontal_fov": getattr(cam, "horizontal_fov", 87.0),
        "near_clip": getattr(cam, "near_clip", 0.05),
        "far_clip": getattr(cam, "far_clip", 5.0),
        "position": getattr(cam, "position", [0.0, 0.22, 0.08]),
        "pitch_deg": getattr(cam, "pitch_deg", 0.0),
        "roll_deg": getattr(cam, "roll_deg", 20.0),
        "yaw_deg": getattr(cam, "yaw_deg", 90.0),
        "output_size": getattr(cam, "output_size", IMG_SIZE),
        "add_noise": getattr(cam, "add_noise", True),
        "noise_level": getattr(cam, "noise_level", 0.02),
        "hole_ratio": getattr(cam, "hole_ratio", 0.05),
        "edge_blur_strength": getattr(cam, "edge_blur_strength", 0.05),
        "distortion_k1": getattr(cam, "distortion_k1", 0.0),
        "distortion_k2": getattr(cam, "distortion_k2", 0.0),
        "distortion_p1": getattr(cam, "distortion_p1", 0.0),
        "distortion_p2": getattr(cam, "distortion_p2", 0.0),
    }


def _get_terrain_cfg(cfg):
    terrain = getattr(cfg, "terrain", None) if cfg is not None else None
    return {
        "terrain_length": getattr(terrain, "terrain_length", 8.0),
        "terrain_width": getattr(terrain, "terrain_width", 8.0),
        "fixed_layout_ring_half_size": getattr(terrain, "fixed_layout_ring_half_size", 2.2),
        "fixed_layout_gap_min": getattr(terrain, "fixed_layout_gap_min", 0.3),
        "fixed_layout_gap_max": getattr(terrain, "fixed_layout_gap_max", 0.7),
        "fixed_layout_gap_buffer": getattr(terrain, "fixed_layout_gap_buffer", 0.1),
        "fixed_layout_robot_clearance": getattr(terrain, "fixed_layout_robot_clearance", 0.27),
        "fixed_layout_wall_thickness": getattr(terrain, "fixed_layout_wall_thickness", 0.25),
        "fixed_layout_center_clearance": getattr(terrain, "fixed_layout_center_clearance", 0.6),
        "fixed_layout_high_height_min": getattr(terrain, "fixed_layout_high_height_min", 0.25),
        "fixed_layout_high_height_max": getattr(terrain, "fixed_layout_high_height_max", 0.35),
        "fixed_layout_low_height_min": getattr(terrain, "fixed_layout_low_height_min", 0.08),
        "fixed_layout_low_height_max": getattr(terrain, "fixed_layout_low_height_max", 0.12),
        "fixed_layout_cyl_radius_min": getattr(terrain, "fixed_layout_cyl_radius_min", 0.15),
        "fixed_layout_cyl_radius_max": getattr(terrain, "fixed_layout_cyl_radius_max", 0.25),
        "fixed_layout_cyl_offset": getattr(terrain, "fixed_layout_cyl_offset", 0.6),
    }


def _get_nav_cfg(cfg):
    nav = getattr(cfg, "navigation", None) if cfg is not None else None
    return {
        "spawn_edge_margin": getattr(nav, "spawn_edge_margin", 0.3),
        "spawn_outside_margin": getattr(nav, "spawn_outside_margin", 0.2),
        "spawn_yaw_jitter_deg": getattr(nav, "spawn_yaw_jitter_deg", 30.0),
        "goal_obstacle_height_threshold": getattr(nav, "goal_obstacle_height_threshold", 0.2),
        "crossable_height_max": getattr(nav, "crossable_height_max", None),
    }


def _get_robot_clearance(cfg):
    asset = getattr(cfg, "asset", None) if cfg is not None else None
    body_shape = getattr(asset, "body_shape", None) if asset is not None else None
    half_width = getattr(body_shape, "y", 0.22)
    return float(half_width) + 0.05


def _rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)

    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    r_y = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float32)
    return r_x @ r_z @ r_y


def _build_base_rays(cam_cfg: Dict[str, float]) -> np.ndarray:
    width = int(cam_cfg["output_size"])
    height = int(cam_cfg["output_size"])
    hfov = math.radians(cam_cfg["horizontal_fov"])
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (height / width))

    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = (height / 2.0) / math.tan(vfov / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - cx) / fx
    y = -(vv - cy) / fy

    r2 = x * x + y * y
    k1 = cam_cfg["distortion_k1"]
    k2 = cam_cfg["distortion_k2"]
    p1 = cam_cfg["distortion_p1"]
    p2 = cam_cfg["distortion_p2"]

    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    # Isaac camera convention: +X forward, +Y left, +Z up
    dirs = np.stack([np.ones_like(x_dist), -x_dist, y_dist], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6
    return dirs / norms


def _sample_robot_pose(terrain_cfg: Dict[str, float], nav_cfg: Dict[str, float]) -> Tuple[np.ndarray, float]:
    half_len = 0.5 * terrain_cfg["terrain_length"]
    half_wid = 0.5 * terrain_cfg["terrain_width"]
    margin = nav_cfg["spawn_edge_margin"]
    ring_half = terrain_cfg.get("fixed_layout_ring_half_size", 0.0)
    outside_margin = nav_cfg.get("spawn_outside_margin", 0.2)
    min_dist = ring_half + outside_margin
    edge_x = max(0.0, half_len - margin, min_dist)
    edge_y = max(0.0, half_wid - margin, min_dist)

    edge = random.randint(0, 3)
    if edge == 0:
        x = -edge_x
        y = random.uniform(-edge_y, edge_y)
    elif edge == 1:
        x = edge_x
        y = random.uniform(-edge_y, edge_y)
    elif edge == 2:
        x = random.uniform(-edge_x, edge_x)
        y = -edge_y
    else:
        x = random.uniform(-edge_x, edge_x)
        y = edge_y

    yaw_to_center = math.atan2(-x, -y)
    jitter = math.radians(nav_cfg["spawn_yaw_jitter_deg"]) * random.uniform(-1.0, 1.0)
    yaw = yaw_to_center + jitter

    return np.array([x, y, 0.0], dtype=np.float32), yaw


def _build_fixed_layout_obstacles(
    terrain_cfg: Dict[str, float],
    difficulty: float,
    clearance: float,
) -> List[Dict[str, float]]:
    ring_half = terrain_cfg["fixed_layout_ring_half_size"]
    wall_thickness = terrain_cfg["fixed_layout_wall_thickness"]
    gap_min = terrain_cfg["fixed_layout_gap_min"]
    gap_max = terrain_cfg["fixed_layout_gap_max"]
    gap_buffer = terrain_cfg.get("fixed_layout_gap_buffer", 0.1)
    robot_clearance = max(terrain_cfg.get("fixed_layout_robot_clearance", 0.27), clearance)
    high_min = terrain_cfg["fixed_layout_high_height_min"]
    high_max = terrain_cfg["fixed_layout_high_height_max"]
    low_min = terrain_cfg["fixed_layout_low_height_min"]
    low_max = terrain_cfg["fixed_layout_low_height_max"]
    cyl_radius_min = terrain_cfg.get("fixed_layout_cyl_radius_min", 0.15)
    cyl_radius_max = terrain_cfg.get("fixed_layout_cyl_radius_max", 0.25)
    cyl_offset = terrain_cfg.get("fixed_layout_cyl_offset", 0.6)

    gap = gap_max - (gap_max - gap_min) * difficulty
    min_gap = 2.0 * robot_clearance + gap_buffer
    gap = max(gap, min_gap)
    high_h = random.uniform(high_min, high_max)
    low_h = random.uniform(low_min, low_max)

    obstacles = []

    x_min, x_max = -ring_half, ring_half
    y_min, y_max = -ring_half, ring_half
    gap_half = gap * 0.5

    # 上下围挡
    for y0, y1 in [(y_max - wall_thickness, y_max), (y_min, y_min + wall_thickness)]:
        obstacles.append({"type": "box", "xmin": x_min, "xmax": -gap_half, "ymin": y0, "ymax": y1, "height": high_h})
        obstacles.append({"type": "box", "xmin": gap_half, "xmax": x_max, "ymin": y0, "ymax": y1, "height": high_h})

    # 左右围挡
    for x0, x1 in [(x_min, x_min + wall_thickness), (x_max - wall_thickness, x_max)]:
        obstacles.append({"type": "box", "xmin": x0, "xmax": x1, "ymin": y_min, "ymax": -gap_half, "height": high_h})
        obstacles.append({"type": "box", "xmin": x0, "xmax": x1, "ymin": gap_half, "ymax": y_max, "height": high_h})

    cyl_radius = random.uniform(cyl_radius_min, cyl_radius_max)
    max_radius = max(0.05, ring_half - wall_thickness - cyl_offset - robot_clearance)
    cyl_radius = min(cyl_radius, max_radius)
    cyl_offset = max(cyl_offset, cyl_radius + robot_clearance)

    gap_centers = [
        (0.0, ring_half - wall_thickness - cyl_offset),   # north
        (0.0, -ring_half + wall_thickness + cyl_offset),  # south
        (ring_half - wall_thickness - cyl_offset, 0.0),   # east
        (-ring_half + wall_thickness + cyl_offset, 0.0),  # west
    ]
    gap_heights = [high_h, low_h, high_h, low_h]
    for (cx, cy), h in zip(gap_centers, gap_heights):
        obstacles.append({
            "type": "cylinder",
            "cx": cx,
            "cy": cy,
            "radius": cyl_radius,
            "height": h,
        })

    return obstacles


def _ray_box_intersect(origin: np.ndarray, direction: np.ndarray, box: Dict[str, float]) -> np.ndarray:
    eps = 1e-6
    dir_safe = np.where(np.abs(direction) < eps, eps, direction)

    t1x = (box["xmin"] - origin[0]) / dir_safe[:, 0]
    t2x = (box["xmax"] - origin[0]) / dir_safe[:, 0]
    t1y = (box["ymin"] - origin[1]) / dir_safe[:, 1]
    t2y = (box["ymax"] - origin[1]) / dir_safe[:, 1]
    t1z = (0.0 - origin[2]) / dir_safe[:, 2]
    t2z = (box["height"] - origin[2]) / dir_safe[:, 2]

    tmin = np.maximum.reduce([np.minimum(t1x, t2x), np.minimum(t1y, t2y), np.minimum(t1z, t2z)])
    tmax = np.minimum.reduce([np.maximum(t1x, t2x), np.maximum(t1y, t2y), np.maximum(t1z, t2z)])

    hit = tmax >= np.maximum(tmin, 0.0)
    t_hit = np.where(hit, tmin, np.inf)
    return t_hit


def _ray_cylinder_intersect(origin: np.ndarray, direction: np.ndarray, cyl: Dict[str, float]) -> np.ndarray:
    cx = cyl["cx"]
    cy = cyl["cy"]
    r = cyl["radius"]
    h = cyl["height"]

    ox = origin[0] - cx
    oy = origin[1] - cy
    dx = direction[:, 0]
    dy = direction[:, 1]
    dz = direction[:, 2]

    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r

    disc = b * b - 4.0 * a * c
    disc = np.where(disc >= 0.0, disc, -1.0)
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

    denom = np.where(np.abs(a) < 1e-6, 1e-6, a * 2.0)
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom

    t_side = np.where(t1 > 0.0, t1, t2)
    z_side = origin[2] + t_side * dz
    valid_side = (t_side > 0.0) & (z_side >= 0.0) & (z_side <= h)
    t_side = np.where(valid_side, t_side, np.inf)

    t_top = np.full_like(t_side, np.inf)
    valid_top = dz < -1e-6
    if np.any(valid_top):
        t_cap = (h - origin[2]) / dz
        x_cap = origin[0] + t_cap * dx - cx
        y_cap = origin[1] + t_cap * dy - cy
        inside = (x_cap * x_cap + y_cap * y_cap) <= r * r
        t_top = np.where((t_cap > 0.0) & inside, t_cap, np.inf)

    return np.minimum(t_side, t_top)


def _render_depth(
    cam_cfg: Dict[str, float],
    base_dirs: np.ndarray,
    obstacles: List[Dict[str, float]],
    robot_pos: np.ndarray,
    robot_yaw: float,
) -> np.ndarray:
    cam_pos_body = np.array(cam_cfg["position"], dtype=np.float32)
    cam_rot_local = _rotation_matrix(
        math.radians(cam_cfg["pitch_deg"]),
        math.radians(cam_cfg["yaw_deg"]),
        math.radians(cam_cfg["roll_deg"]),
    )
    yaw_rot = _rotation_matrix(0.0, robot_yaw, 0.0)

    cam_pos = robot_pos + yaw_rot @ cam_pos_body

    dirs = base_dirs.reshape(-1, 3)
    dirs_world = (yaw_rot @ cam_rot_local @ dirs.T).T
    dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-6)

    # 与地面相交 (z=0)
    plane_t = np.full(dirs_world.shape[0], np.inf, dtype=np.float32)
    down_mask = dirs_world[:, 2] < -1e-6
    plane_t[down_mask] = (0.0 - cam_pos[2]) / dirs_world[down_mask, 2]

    # 与障碍相交
    t_min = np.full(dirs_world.shape[0], np.inf, dtype=np.float32)
    for obs in obstacles:
        if obs.get("type") == "cylinder":
            t_hit = _ray_cylinder_intersect(cam_pos, dirs_world, obs)
        else:
            t_hit = _ray_box_intersect(cam_pos, dirs_world, obs)
        t_min = np.minimum(t_min, t_hit)

    depth = np.minimum(t_min, plane_t)
    depth = np.clip(depth, cam_cfg["near_clip"], cam_cfg["far_clip"])
    depth = depth.reshape(base_dirs.shape[:2])

    if cam_cfg["add_noise"]:
        sigma = cam_cfg["noise_level"] * (cam_cfg["far_clip"] - cam_cfg["near_clip"])
        depth = depth + np.random.normal(0.0, sigma, depth.shape)

    hole_ratio = cam_cfg["hole_ratio"]
    if hole_ratio > 0.0:
        holes = np.random.rand(*depth.shape) < hole_ratio
        depth[holes] = cam_cfg["far_clip"]

    blur = cam_cfg["edge_blur_strength"]
    if blur > 0.0:
        depth = gaussian_filter(depth, sigma=max(0.5, blur * 10.0))

    depth = np.clip(depth, cam_cfg["near_clip"], cam_cfg["far_clip"])
    depth = (depth - cam_cfg["near_clip"]) / (cam_cfg["far_clip"] - cam_cfg["near_clip"]) 
    depth = np.clip(depth, 0.0, 1.0)
    return depth.astype(np.float32)


def _obstacles_to_map(
    obstacles: List[Dict[str, float]],
    robot_pos: np.ndarray,
    robot_yaw: float,
    map_extent: float,
    map_size: int,
    height_threshold: float,
) -> np.ndarray:
    occ = np.zeros((map_size, map_size), dtype=np.float32)

    cell_x = map_extent / map_size
    cell_y = map_extent / map_size
    x_min = -map_extent / 2.0
    y_min = 0.0

    cy = math.cos(-robot_yaw)
    sy = math.sin(-robot_yaw)

    x_centers = x_min + (np.arange(map_size) + 0.5) * cell_x
    y_centers = y_min + (np.arange(map_size) + 0.5) * cell_y
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    rot = np.array([[cy, -sy], [sy, cy]], dtype=np.float32)

    for obs in obstacles:
        if obs["height"] < height_threshold:
            continue
        if obs.get("type") == "cylinder":
            center = np.array([obs["cx"], obs["cy"]], dtype=np.float32) - robot_pos[:2]
            center = center @ rot.T
            dist = (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
            occ[dist <= obs["radius"] ** 2] = 1.0
        else:
            corners = np.array([
                [obs["xmin"], obs["ymin"]],
                [obs["xmin"], obs["ymax"]],
                [obs["xmax"], obs["ymin"]],
                [obs["xmax"], obs["ymax"]],
            ], dtype=np.float32)
            corners -= robot_pos[:2]
            corners = corners @ rot.T
            bx_min, by_min = corners.min(axis=0)
            bx_max, by_max = corners.max(axis=0)

            i0 = int(math.floor((bx_min - x_min) / cell_x))
            i1 = int(math.ceil((bx_max - x_min) / cell_x))
            j0 = int(math.floor((by_min - y_min) / cell_y))
            j1 = int(math.ceil((by_max - y_min) / cell_y))

            i0 = max(0, min(map_size, i0))
            i1 = max(0, min(map_size, i1))
            j0 = max(0, min(map_size, j0))
            j1 = max(0, min(map_size, j1))
            if i1 <= i0 or j1 <= j0:
                continue
            occ[j0:j1, i0:i1] = 1.0

    return occ


def _compute_passable_gap(occ_blocking: np.ndarray, map_extent: float, clearance: float) -> np.ndarray:
    free = 1.0 - occ_blocking
    dist = distance_transform_edt(free) * (map_extent / occ_blocking.shape[0])
    passable = (dist >= clearance) & (free > 0.5)
    return passable.astype(np.float32)


class AffordanceDataset(Dataset):
    """Affordance 数据集（第一视角）"""

    def __init__(self, data_path: str = None, transform: bool = False):
        self.transform = transform
        self.data = []

        if data_path:
            if os.path.exists(data_path):
                print(f"[Dataset] Loading dataset from {data_path}...")
                try:
                    loaded_data = torch.load(data_path)
                    if isinstance(loaded_data, dict) and 'samples' in loaded_data:
                        self.data = loaded_data['samples']
                    else:
                        self.data = loaded_data
                    print(f"[Dataset] Successfully loaded {len(self.data)} samples.")
                except Exception as e:
                    print(f"[Dataset] Error loading data: {e}")
                    raise e
            else:
                raise FileNotFoundError(f"[Dataset] Data file not found at {data_path}.")
        else:
            print("[Dataset] Initialized empty dataset. Use for generation only.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        depth = sample['depth']
        occupancy = sample['occupancy']
        passable_gap = sample['passable_gap']
        low_obstacle = sample.get('low_obstacle', None)

        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        occ_t = torch.from_numpy(occupancy).float()
        gap_t = torch.from_numpy(passable_gap).float()
        if low_obstacle is None:
            low_obstacle = np.zeros_like(passable_gap, dtype=np.float32)
        low_t = torch.from_numpy(low_obstacle).float()

        if self.transform:
            if random.random() > 0.5:
                depth_t = torch.flip(depth_t, [-1])
                occ_t = torch.flip(occ_t, [-1])
                gap_t = torch.flip(gap_t, [-1])
                low_t = torch.flip(low_t, [-1])

        return {
            'depth': depth_t,
            'occupancy': occ_t,
            'passable_gap': gap_t,
            'low_obstacle': low_t,
        }


def generate_synthetic_sample(
    cam_cfg: Dict[str, float],
    terrain_cfg: Dict[str, float],
    nav_cfg: Dict[str, float],
    base_dirs: np.ndarray,
    map_extent: float,
    clearance: float,
) -> Dict[str, np.ndarray]:
    difficulty = random.random()

    robot_pos, robot_yaw = _sample_robot_pose(terrain_cfg, nav_cfg)
    obstacles = _build_fixed_layout_obstacles(terrain_cfg, difficulty, clearance)

    depth = _render_depth(cam_cfg, base_dirs, obstacles, robot_pos, robot_yaw)

    occ_all = _obstacles_to_map(
        obstacles,
        robot_pos,
        robot_yaw,
        map_extent,
        MAP_SIZE,
        height_threshold=0.0,
    )
    crossable_height = nav_cfg.get("crossable_height_max", None)
    if crossable_height is None:
        crossable_height = nav_cfg["goal_obstacle_height_threshold"]
    crossable_height = float(crossable_height)
    occ_blocking = _obstacles_to_map(
        obstacles,
        robot_pos,
        robot_yaw,
        map_extent,
        MAP_SIZE,
        height_threshold=crossable_height,
    )
    occ_crossable = _obstacles_to_map(
        obstacles,
        robot_pos,
        robot_yaw,
        map_extent,
        MAP_SIZE,
        height_threshold=crossable_height,
    )
    passable_gap = _compute_passable_gap(occ_blocking, map_extent, clearance)
    low_obstacle = np.clip(occ_all - occ_crossable, 0.0, 1.0)

    return {
        'depth': depth.astype(np.float32),
        'occupancy': occ_all.astype(np.float32),
        'passable_gap': passable_gap.astype(np.float32),
        'low_obstacle': low_obstacle.astype(np.float32),
        'difficulty': np.array([difficulty], dtype=np.float32),
    }


def visualize_samples(samples: List[Dict], save_path: str = "data_preview.png"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[Warning] Matplotlib import error: {e}. Skipping visualization.")
        return

    num_show = min(5, len(samples))
    fig, axes = plt.subplots(num_show, 5, figsize=(18, 4 * num_show))

    if num_show == 1:
        axes = [axes]

    for i in range(num_show):
        s = samples[i]

        ax = axes[i][0] if num_show > 1 else axes[0]
        im = ax.imshow(s['depth'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title("Depth (ego)")
        fig.colorbar(im, ax=ax)

        ax = axes[i][1] if num_show > 1 else axes[1]
        ax.imshow(s['occupancy'], cmap='gray', vmin=0, vmax=1)
        ax.set_title("Occupancy")

        ax = axes[i][2] if num_show > 1 else axes[2]
        ax.imshow(s['passable_gap'], cmap='Greens', vmin=0, vmax=1)
        ax.set_title("Passable Gap")

        ax = axes[i][3] if num_show > 1 else axes[3]
        ax.imshow(s.get('low_obstacle', np.zeros_like(s['passable_gap'])), cmap='Oranges', vmin=0, vmax=1)
        ax.set_title("Low Obstacle")

        ax = axes[i][4] if num_show > 1 else axes[4]
        gap_ratio = float(np.mean(s['passable_gap']))
        occ_ratio = float(np.mean(s['occupancy']))
        low_ratio = float(np.mean(s.get('low_obstacle', np.zeros_like(s['passable_gap']))))
        ax.text(0.1, 0.6, f"Gap Ratio: {gap_ratio:.2f}\nOcc Ratio: {occ_ratio:.2f}\nLow Ratio: {low_ratio:.2f}", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"[Viz] Preview saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Affordance Dataset (First-Person)")
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='data/processed', help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='affordance_data.pt', help='Output filename')
    parser.add_argument('--visualize', action='store_true', help='Generate a preview image of first 5 samples')
    args = parser.parse_args()

    cfg = load_hex_ground_cfg()
    cam_cfg = _get_camera_cfg(cfg)
    terrain_cfg = _get_terrain_cfg(cfg)
    nav_cfg = _get_nav_cfg(cfg)

    output_size = int(cam_cfg["output_size"])
    cam_cfg["output_size"] = output_size

    base_dirs = _build_base_rays(cam_cfg)
    map_extent = cam_cfg["far_clip"]
    clearance = _get_robot_clearance(cfg)

    if not os.path.exists(args.save_dir):
        print(f"[Main] Creating directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(args.save_dir, args.filename)

    print(f"[Main] Generating {args.num_samples} samples...")
    print(f"[Main] Depth size: {output_size}x{output_size}, Map: {MAP_SIZE}x{MAP_SIZE}, Extent: {map_extent:.2f}m")

    samples = []
    for _ in tqdm(range(args.num_samples), desc="Generating", unit="sample"):
        s = generate_synthetic_sample(cam_cfg, terrain_cfg, nav_cfg, base_dirs, map_extent, clearance)
        samples.append(s)

    occ_ratio = [float(np.mean(s['occupancy'])) for s in samples]
    gap_ratio = [float(np.mean(s['passable_gap'])) for s in samples]
    low_ratio = [float(np.mean(s.get('low_obstacle', 0.0))) for s in samples]

    print("\n" + "=" * 40)
    print("DATASET STATISTICS")
    print("=" * 40)
    print(f"Total Samples: {len(samples)}")
    print(f"Occupancy Ratio: mean={np.mean(occ_ratio):.4f}, std={np.std(occ_ratio):.4f}")
    print(f"Gap Ratio: mean={np.mean(gap_ratio):.4f}, std={np.std(gap_ratio):.4f}")
    print(f"Low Ratio: mean={np.mean(low_ratio):.4f}, std={np.std(low_ratio):.4f}")
    print("=" * 40)

    if args.visualize:
        viz_path = os.path.join(args.save_dir, "dataset_preview.png")
        visualize_samples(samples, viz_path)

    print(f"\n[Main] Saving dataset to {save_path}...")
    torch.save({
        'samples': samples,
        'meta': {
            'num_samples': args.num_samples,
            'img_size': output_size,
            'map_size': MAP_SIZE,
            'map_extent': map_extent,
            'version': 'v4.0_first_person',
            'camera_cfg': cam_cfg,
        }
    }, save_path)

    print("[Main] Done!")


if __name__ == "__main__":
    main()
