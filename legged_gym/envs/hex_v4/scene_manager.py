from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StaticObstacleSpec:
    kind: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    yaw: float = 0.0
    raw_size: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, float]:
        data = {
            "kind": self.kind,
            "position": [float(v) for v in self.position],
            "size": [float(v) for v in self.size],
            "yaw": float(self.yaw),
        }
        if self.raw_size is not None:
            data["raw_size"] = [float(v) for v in self.raw_size]
        return data


@dataclass(frozen=True)
class DynamicObstacleSpec:
    shape: str
    size: Tuple[float, float, float]
    path_start: Tuple[float, float, float]
    path_end: Tuple[float, float, float]
    speed: float
    phase: float
    period: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "shape": self.shape,
            "size": list(self.size),
            "path_start": list(self.path_start),
            "path_end": list(self.path_end),
            "speed": float(self.speed),
            "phase": float(self.phase),
            "period": float(self.period),
        }


@dataclass(frozen=True)
class SceneSpec:
    scene_type: str
    difficulty: float
    params: Dict[str, object]
    static_obstacles: Tuple[StaticObstacleSpec, ...] = ()
    dynamic_template: Optional[Dict[str, float]] = None
    layout_seed: Optional[int] = None
    layout_id: Optional[str] = None
    layout_hash: Optional[str] = None

    def to_meta(self) -> Dict[str, object]:
        return {
            "scene_type": self.scene_type,
            "difficulty": float(self.difficulty),
            "layout_id": self.layout_id,
            "layout_hash": self.layout_hash,
            "layout_seed": self.layout_seed,
            "params": dict(self.params),
            "dynamic_template": dict(self.dynamic_template or {}),
        }


class SceneManager:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.scene_type = getattr(cfg, "scene_type", None)
        self.scene_types = list(getattr(cfg, "scene_types", []) or [])
        if not self.scene_types and self.scene_type:
            self.scene_types = [self.scene_type]
        self.scene_probs_easy = getattr(cfg, "scene_probs_easy", None)
        self.scene_probs_hard = getattr(cfg, "scene_probs_hard", None)
        self.scene_probs = getattr(cfg, "scene_probs", None)

        self.scene_params_easy = getattr(cfg, "scene_params_easy", {})
        self.scene_params_hard = getattr(cfg, "scene_params_hard", {})
        self.scene_seed = int(getattr(cfg, "scene_seed", 13))
        self.scene_margin = float(getattr(cfg, "scene_margin", 0.3))
        self.scene_clearance = float(
            getattr(cfg, "scene_clearance", getattr(cfg, "fixed_layout_robot_clearance", 0.27))
        )
        self.scene_high_dt = float(getattr(cfg, "scene_high_dt", 0.1))
        self.max_dynamic_obstacles = int(getattr(cfg, "scene_dynamic_max", 0))
        self.scene_use_heightfield = bool(getattr(cfg, "scene_use_heightfield", False))

        self.scene_width = float(getattr(cfg, "terrain_width", 8.0))
        self.scene_length = float(getattr(cfg, "terrain_length", 8.0))
        base_block_size = float(getattr(cfg, "scene_static_block_size", 0.4))
        base_block_height = float(getattr(cfg, "scene_static_block_height", 0.35))
        block_sizes = list(getattr(cfg, "scene_static_block_sizes", []) or [])
        block_heights = list(getattr(cfg, "scene_static_block_heights", []) or [])
        if not block_sizes:
            block_sizes = [base_block_size]
        if not block_heights:
            block_heights = [base_block_height] * len(block_sizes)
        if len(block_heights) < len(block_sizes):
            block_heights.extend([block_heights[-1]] * (len(block_sizes) - len(block_heights)))
        self.block_sizes = [float(v) for v in block_sizes]
        self.block_heights = [float(v) for v in block_heights[:len(self.block_sizes)]]
        self.block_size = self.block_sizes[0]
        self.block_height = self.block_heights[0]
        self.wall_block_size = float(getattr(cfg, "scene_static_wall_block_size", self.block_size))
        self.wall_block_height = float(getattr(cfg, "scene_static_wall_block_height", self.block_height))
        self._check_block_coverage()

    def _check_block_coverage(self) -> None:
        scenes = self.scene_types or ([self.scene_type] if self.scene_type else [])
        if not scenes:
            return
        size_min = None
        size_max = None
        height_min = None
        height_max = None
        for scene_type in scenes:
            params_easy = self._resolve_scene_params(self.scene_params_easy, scene_type)
            params_hard = self._resolve_scene_params(self.scene_params_hard, scene_type)
            params_all = [params_easy, params_hard]
            for params in params_all:
                if not params:
                    continue
                if "pole_radius_min" in params or "pole_radius_max" in params:
                    r_min = float(params.get("pole_radius_min", 0.5 * self.block_size))
                    r_max = float(params.get("pole_radius_max", r_min))
                    size_min = r_min * 2.0 if size_min is None else min(size_min, r_min * 2.0)
                    size_max = r_max * 2.0 if size_max is None else max(size_max, r_max * 2.0)
                    h_min = float(params.get("pole_height_min", params.get("pole_height", self.block_height)))
                    h_max = float(params.get("pole_height_max", h_min))
                    height_min = h_min if height_min is None else min(height_min, h_min)
                    height_max = h_max if height_max is None else max(height_max, h_max)
                if "cluster_radius" in params:
                    r_min = float(params.get("cluster_radius", 0.5 * self.block_size))
                    size_min = r_min * 2.0 if size_min is None else min(size_min, r_min * 2.0)
                    size_max = r_min * 2.0 if size_max is None else max(size_max, r_min * 2.0)
                    h_min = float(params.get("obstacle_height", self.block_height))
                    height_min = h_min if height_min is None else min(height_min, h_min)
                    height_max = h_min if height_max is None else max(height_max, h_min)
        if size_min is None or size_max is None:
            return
        block_min = min(self.block_sizes)
        block_max = max(self.block_sizes)
        if block_min > size_min or block_max < size_max:
            raise ValueError(
                f"scene_static_block_sizes 覆盖不足: "
                f"range={block_min:.2f}~{block_max:.2f}, "
                f"required={size_min:.2f}~{size_max:.2f}"
            )
        height_min = self.block_heights[0] if height_min is None else height_min
        height_max = self.block_heights[0] if height_max is None else height_max
        h_min = min(self.block_heights)
        h_max = max(self.block_heights)
        if h_min > height_min or h_max < height_max:
            raise ValueError(
                f"scene_static_block_heights 覆盖不足: "
                f"range={h_min:.2f}~{h_max:.2f}, "
                f"required={height_min:.2f}~{height_max:.2f}"
            )

    @property
    def has_dynamic(self) -> bool:
        if self.max_dynamic_obstacles <= 0:
            return False
        return "s4_crossing" in (self.scene_types or [])

    def seed_for_cell(self, row: int, col: int) -> int:
        return int(self.scene_seed + row * 1000 + col * 17)

    def sample_scene(
        self,
        difficulty: float,
        env_id: int,
        episode_idx: int,
    ) -> SceneSpec:
        layout_seed = int(self.scene_seed + env_id * 10007 + episode_idx * 131)
        rng = np.random.RandomState(layout_seed)
        scene_type = self._select_scene_type(rng, difficulty)
        scene_seed = layout_seed + self._scene_type_index(scene_type) * 7919
        return self.build_scene(
            terrain=None,
            difficulty=difficulty,
            seed=scene_seed,
            scene_type=scene_type,
            layout_seed=layout_seed,
        )

    def build_scene(
        self,
        terrain,
        difficulty: float,
        seed: int,
        scene_type: Optional[str] = None,
        layout_seed: Optional[int] = None,
    ) -> SceneSpec:
        if scene_type is None:
            if self.scene_type:
                scene_type = self.scene_type
            elif self.scene_types:
                scene_type = self.scene_types[0]
            else:
                scene_type = "unknown"

        rng = np.random.RandomState(seed)
        params = self._interpolate_params(difficulty, scene_type)
        static_obstacles = self._build_scene_obstacles(scene_type, params, rng)
        if terrain is not None:
            self._clear_height(terrain)
            if self.scene_use_heightfield:
                self._apply_static_to_heightfield(terrain, static_obstacles)

        dynamic_template = None
        if scene_type == "s4_crossing":
            dynamic_template = {
                "count_min": float(params.get("dynamic_count_min", 2)),
                "count_max": float(params.get("dynamic_count_max", 6)),
                "cross_width": float(params.get("cross_width", 3.0)),
                "cross_span": float(params.get("cross_span", 2.5)),
                "react_steps_min": float(params.get("react_steps_min", 8)),
                "react_steps_max": float(params.get("react_steps_max", 20)),
                "size_xy": float(params.get("dynamic_size_xy", 0.35)),
                "height": float(params.get("dynamic_height", 0.5)),
                "axis": str(params.get("dynamic_axis", "x")),
                "dt_high": float(self.scene_high_dt),
            }

        layout_hash = self._hash_layout(static_obstacles, dynamic_template)
        layout_id = f"{scene_type}-{seed}"

        return SceneSpec(
            scene_type=scene_type,
            difficulty=float(difficulty),
            params=params,
            static_obstacles=tuple(static_obstacles),
            dynamic_template=dynamic_template,
            layout_seed=layout_seed if layout_seed is not None else seed,
            layout_id=layout_id,
            layout_hash=layout_hash,
        )

    def sample_dynamic_obstacles(
        self, scene_spec: SceneSpec, env_id: int, episode_idx: int
    ) -> List[DynamicObstacleSpec]:
        template = scene_spec.dynamic_template or {}
        if not template:
            return []

        base_seed = int(scene_spec.layout_seed or self.scene_seed)
        rng = np.random.RandomState(base_seed + env_id * 97 + episode_idx * 131)
        count_min = int(round(template.get("count_min", 1)))
        count_max = int(round(template.get("count_max", count_min)))
        count = max(count_min, rng.randint(count_min, count_max + 1))
        if self.max_dynamic_obstacles > 0:
            count = min(count, self.max_dynamic_obstacles)

        cross_width = float(template.get("cross_width", 3.0))
        cross_span = float(template.get("cross_span", 2.5))
        react_min = float(template.get("react_steps_min", 8))
        react_max = float(template.get("react_steps_max", 20))
        size_xy = float(template.get("size_xy", 0.35))
        height = float(template.get("height", 0.5))
        axis = str(template.get("axis", "x"))
        dt_high = float(template.get("dt_high", 0.1))

        obstacles: List[DynamicObstacleSpec] = []
        for _ in range(count):
            if axis == "x":
                center_y = rng.uniform(-0.5 * cross_span, 0.5 * cross_span)
                start = (-0.5 * cross_width, center_y, 0.0)
                end = (0.5 * cross_width, center_y, 0.0)
            else:
                center_x = rng.uniform(-0.5 * cross_span, 0.5 * cross_span)
                start = (center_x, -0.5 * cross_width, 0.0)
                end = (center_x, 0.5 * cross_width, 0.0)

            react_steps = rng.uniform(react_min, react_max)
            react_steps = max(1.0, react_steps)
            path_len = float(np.linalg.norm(np.array(end[:2]) - np.array(start[:2])))
            speed = path_len / (react_steps * dt_high)
            speed = max(speed, 0.05)
            period = 2.0 * path_len / speed
            phase = rng.uniform(0.0, period)

            obstacles.append(
                DynamicObstacleSpec(
                    shape="box",
                    size=(size_xy, size_xy, height),
                    path_start=(start[0], start[1], height * 0.5),
                    path_end=(end[0], end[1], height * 0.5),
                    speed=speed,
                    phase=phase,
                    period=period,
                )
            )
        return obstacles

    def build_meta(self, scene_spec: SceneSpec, dynamic_specs: List[DynamicObstacleSpec]) -> Dict[str, object]:
        static_count = len(scene_spec.static_obstacles)
        dynamic_count = len(dynamic_specs)
        area = max(self.scene_width * self.scene_length, 1e-6)
        density_proxy = static_count / area
        min_gap_over_c = self._estimate_min_gap(scene_spec, static_count) / max(self.scene_clearance, 1e-6)
        v_cross_proxy = self._estimate_cross_speed(scene_spec.dynamic_template)

        meta = scene_spec.to_meta()
        meta.update(
            {
                "num_static": static_count,
                "num_dynamic": dynamic_count,
                "density_proxy": float(density_proxy),
                "min_gap_over_c": float(min_gap_over_c),
                "v_cross_proxy": float(v_cross_proxy),
            }
        )
        if static_count > 0:
            size_xy = [max(spec.size[0], spec.size[1]) for spec in scene_spec.static_obstacles]
            heights = [spec.size[2] for spec in scene_spec.static_obstacles]
            raw_sizes = [spec.raw_size for spec in scene_spec.static_obstacles if spec.raw_size is not None]
            raw_size_xy = [max(s[0], s[1]) for s in raw_sizes] if raw_sizes else []
            raw_heights = [s[2] for s in raw_sizes] if raw_sizes else []
            meta.update(
                {
                    "static_size_xy_mean": float(np.mean(size_xy)),
                    "static_size_xy_min": float(np.min(size_xy)),
                    "static_size_xy_max": float(np.max(size_xy)),
                    "static_height_mean": float(np.mean(heights)),
                    "static_height_min": float(np.min(heights)),
                    "static_height_max": float(np.max(heights)),
                }
            )
            if raw_size_xy:
                meta.update(
                    {
                        "static_raw_size_xy_mean": float(np.mean(raw_size_xy)),
                        "static_raw_size_xy_min": float(np.min(raw_size_xy)),
                        "static_raw_size_xy_max": float(np.max(raw_size_xy)),
                        "static_raw_height_mean": float(np.mean(raw_heights)),
                        "static_raw_height_min": float(np.min(raw_heights)),
                        "static_raw_height_max": float(np.max(raw_heights)),
                    }
                )
        if dynamic_specs:
            meta["dynamic_obstacles"] = [spec.to_dict() for spec in dynamic_specs]
        return meta

    def _resolve_scene_params(self, params: Dict[str, object], scene_type: str) -> Dict[str, object]:
        if not isinstance(params, dict) or not params:
            return {}
        values = list(params.values())
        if values and all(isinstance(v, dict) for v in values):
            return params.get(scene_type, {})
        return params

    def _quantize_block(self, size_xy: float, height: float) -> Tuple[float, float, int]:
        sizes = np.asarray(self.block_sizes, dtype=np.float32)
        heights = np.asarray(self.block_heights, dtype=np.float32)
        size_diff = np.abs(sizes - size_xy) / max(size_xy, 1e-6)
        height_diff = np.abs(heights - height) / max(height, 1e-6)
        score = size_diff + height_diff
        idx = int(np.argmin(score))
        size_xy = float(self.block_sizes[idx])
        height = float(self.block_heights[idx])
        return size_xy, height, idx

    def _sample_pole_size(
        self, params: Dict[str, object], rng: np.random.RandomState
    ) -> Tuple[float, float, float, float]:
        radius_min = float(params.get("pole_radius_min", 0.5 * self.block_size))
        radius_max = float(params.get("pole_radius_max", radius_min))
        height_min = float(params.get("pole_height_min", params.get("pole_height", self.block_height)))
        height_max = float(params.get("pole_height_max", height_min))
        radius = rng.uniform(radius_min, radius_max)
        height = rng.uniform(height_min, height_max)
        raw_size_xy = 2.0 * radius
        raw_height = height
        size_xy, height, _ = self._quantize_block(raw_size_xy, height)
        return size_xy, height, raw_size_xy, raw_height

    def _sample_block_size(
        self, params: Dict[str, object], rng: np.random.RandomState
    ) -> Tuple[float, float, float, float]:
        size_min = float(params.get("block_size_min", self.block_size))
        size_max = float(params.get("block_size_max", size_min))
        height_min = float(params.get("block_height_min", self.block_height))
        height_max = float(params.get("block_height_max", height_min))
        size_xy = rng.uniform(size_min, size_max)
        height = rng.uniform(height_min, height_max)
        raw_size_xy = size_xy
        raw_height = height
        size_xy, height, _ = self._quantize_block(size_xy, height)
        return size_xy, height, raw_size_xy, raw_height

    def _sample_obstacle_spec(
        self,
        x: float,
        y: float,
        params: Dict[str, object],
        rng: np.random.RandomState,
        block_ratio: float,
    ) -> StaticObstacleSpec:
        if rng.uniform(0.0, 1.0) < block_ratio:
            size_xy, height, raw_size_xy, raw_height = self._sample_block_size(params, rng)
            kind = "block"
        else:
            size_xy, height, raw_size_xy, raw_height = self._sample_pole_size(params, rng)
            kind = "pole"
        return StaticObstacleSpec(
            kind=kind,
            position=(x, y, 0.5 * height),
            size=(size_xy, size_xy, height),
            raw_size=(raw_size_xy, raw_size_xy, raw_height),
        )

    def _interpolate_params(self, difficulty: float, scene_type: str) -> Dict[str, object]:
        params_easy = self._resolve_scene_params(self.scene_params_easy, scene_type)
        params_hard = self._resolve_scene_params(self.scene_params_hard, scene_type)
        params: Dict[str, object] = {}
        keys = set(params_easy.keys()) | set(params_hard.keys())
        for key in keys:
            if key in params_easy and key in params_hard:
                easy_val = params_easy[key]
                hard_val = params_hard[key]
                try:
                    easy = float(easy_val)
                    hard = float(hard_val)
                except (TypeError, ValueError):
                    params[key] = easy_val
                    continue
                params[key] = easy + (hard - easy) * float(difficulty)
            elif key in params_easy:
                params[key] = params_easy[key]
            else:
                params[key] = params_hard[key]
        return params

    def _select_scene_type(self, rng: np.random.RandomState, difficulty: float) -> str:
        if not self.scene_types:
            return self.scene_type or "unknown"
        if len(self.scene_types) == 1:
            return self.scene_types[0]
        probs = self._get_scene_probs(difficulty)
        idx = int(rng.choice(len(self.scene_types), p=probs))
        return self.scene_types[idx]

    def _scene_type_index(self, scene_type: str) -> int:
        if not self.scene_types:
            return 0
        try:
            return self.scene_types.index(scene_type)
        except ValueError:
            return 0

    def _resolve_probs(self, probs) -> np.ndarray:
        if isinstance(probs, dict):
            arr = [float(probs.get(scene, 0.0)) for scene in self.scene_types]
            return np.asarray(arr, dtype=np.float32)
        return np.asarray(list(probs), dtype=np.float32)

    def _get_scene_probs(self, difficulty: float) -> np.ndarray:
        if self.scene_probs_easy is not None and self.scene_probs_hard is not None:
            easy = self._resolve_probs(self.scene_probs_easy)
            hard = self._resolve_probs(self.scene_probs_hard)
            if easy.shape != hard.shape:
                probs = np.ones(len(self.scene_types), dtype=np.float32)
            else:
                probs = (1.0 - difficulty) * easy + difficulty * hard
        elif self.scene_probs is not None:
            probs = self._resolve_probs(self.scene_probs)
        else:
            probs = np.ones(len(self.scene_types), dtype=np.float32)
        if probs.shape[0] != len(self.scene_types):
            probs = np.ones(len(self.scene_types), dtype=np.float32)
        probs = np.clip(probs, 1e-6, None)
        probs = probs / np.sum(probs)
        return probs

    def _hash_layout(
        self,
        static_obstacles: List[StaticObstacleSpec],
        dynamic_template: Optional[Dict[str, float]],
    ) -> str:
        if not static_obstacles and not dynamic_template:
            return ""
        data = []
        for obs in static_obstacles:
            pos = [round(v, 3) for v in obs.position]
            size = [round(v, 3) for v in obs.size]
            data.append((obs.kind, pos, size))
        payload = {"static": data, "dynamic": dynamic_template}
        digest = hashlib.md5(str(payload).encode("utf-8")).hexdigest()
        return digest

    def _estimate_min_gap(self, scene_spec: SceneSpec, static_count: int) -> float:
        params = scene_spec.params
        if scene_spec.scene_type == "s1_corridor":
            return float(params.get("gap_width_min", params.get("corridor_width", 1.0)))
        if scene_spec.scene_type == "s3_doorway":
            return float(params.get("door_width_min", params.get("room_width", 1.0)))
        if static_count > 0:
            area = max(self.scene_width * self.scene_length, 1e-6)
            return float(np.sqrt(area / static_count))
        return float(self.scene_clearance)

    def _estimate_cross_speed(self, dynamic_template: Optional[Dict[str, float]]) -> float:
        if not dynamic_template:
            return 0.0
        react_min = float(dynamic_template.get("react_steps_min", 8))
        react_max = float(dynamic_template.get("react_steps_max", react_min))
        react = max(1.0, 0.5 * (react_min + react_max))
        cross_width = float(dynamic_template.get("cross_width", 3.0))
        dt_high = float(dynamic_template.get("dt_high", 0.1))
        return cross_width / max(react * dt_high, 1e-6)

    def _clear_height(self, terrain) -> None:
        terrain.height_field_raw[:] = 0

    def _apply_static_to_heightfield(self, terrain, obstacles: List[StaticObstacleSpec]) -> None:
        for obs in obstacles:
            self._fill_rect(
                terrain,
                obs.position[0],
                obs.position[1],
                obs.size[0],
                obs.size[1],
                obs.size[2],
            )

    def _tile_rect(
        self,
        center_x: float,
        center_y: float,
        size_x: float,
        size_y: float,
        kind: str = "block",
    ) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        if size_x <= 0.0 or size_y <= 0.0:
            return obstacles
        if kind == "wall":
            block_size = self.wall_block_size
            block_height = self.wall_block_height
        else:
            block_size = self.block_size
            block_height = self.block_height
        nx = max(1, int(np.ceil(size_x / block_size)))
        ny = max(1, int(np.ceil(size_y / block_size)))
        step_x = size_x / nx
        step_y = size_y / ny
        start_x = center_x - 0.5 * size_x + 0.5 * step_x
        start_y = center_y - 0.5 * size_y + 0.5 * step_y
        for ix in range(nx):
            for iy in range(ny):
                x = start_x + ix * step_x
                y = start_y + iy * step_y
                obstacles.append(
                    StaticObstacleSpec(
                        kind=kind,
                        position=(x, y, 0.5 * block_height),
                        size=(block_size, block_size, block_height),
                    )
                )
        return obstacles

    def _build_scene_obstacles(
        self, scene_type: str, params: Dict[str, object], rng: np.random.RandomState
    ) -> List[StaticObstacleSpec]:
        if scene_type == "s1_corridor":
            return self._build_corridor(params, rng)
        if scene_type == "s2_forest":
            return self._build_forest(params, rng)
        if scene_type == "s3_doorway":
            return self._build_doorway(params, rng)
        if scene_type == "s4_crossing":
            return self._build_crossing(params, rng)
        if scene_type == "s5_transition":
            return self._build_density_transition(params, rng)
        if scene_type == "s6_ood_structured":
            return self._build_structured_ood(params, rng)
        return []

    def _build_corridor(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        width_m = self.scene_width
        length_m = self.scene_length
        corridor_width = float(params.get("corridor_width", 1.4))
        corridor_width = max(corridor_width, 2.0 * self.scene_clearance + 0.1)
        wall_thickness = float(params.get("wall_thickness", self.block_size))
        segment_count = int(round(params.get("wall_segment_count", 0)))
        segment_len = float(params.get("wall_segment_len", 0.0))
        wall_jitter_x = float(params.get("wall_jitter_x", 0.0))

        outer = max(0.0, width_m - corridor_width)
        if outer > 0:
            left_center = -(corridor_width * 0.5 + outer * 0.25)
            right_center = (corridor_width * 0.5 + outer * 0.25)
            if segment_count > 0:
                if segment_len <= 0.0:
                    segment_len = length_m / float(segment_count)
                segment_len = max(1e-3, segment_len)
                start_y = -0.5 * length_m + 0.5 * segment_len
                for idx in range(segment_count):
                    center_y = start_y + idx * segment_len
                    jitter = rng.uniform(-wall_jitter_x, wall_jitter_x) if wall_jitter_x > 0.0 else 0.0
                    obstacles.extend(
                        self._tile_rect(left_center + jitter, center_y, outer * 0.5, segment_len, kind="wall")
                    )
                    obstacles.extend(
                        self._tile_rect(right_center + jitter, center_y, outer * 0.5, segment_len, kind="wall")
                    )
            else:
                obstacles.extend(self._tile_rect(left_center, 0.0, outer * 0.5, length_m, kind="wall"))
                obstacles.extend(self._tile_rect(right_center, 0.0, outer * 0.5, length_m, kind="wall"))

        gate_count = int(round(params.get("gate_count", 2)))
        gate_thickness = float(params.get("gate_thickness", wall_thickness))
        gap_min = float(params.get("gap_width_min", 0.6))
        gap_max = float(params.get("gap_width_max", 1.0))
        gap_min = max(gap_min, 2.0 * self.scene_clearance + 0.05)
        gap_max = max(gap_max, gap_min)
        offset_max = float(params.get("gate_offset_max", 0.3))
        margin_y = float(params.get("gate_margin_y", 0.8))
        block_width = float(params.get("gate_block_width", self.block_size))

        for _ in range(max(0, gate_count)):
            center_y = rng.uniform(-0.5 * length_m + margin_y, 0.5 * length_m - margin_y)
            gap_width = rng.uniform(gap_min, gap_max)
            gap_width = min(gap_width, corridor_width - 0.1)
            remain = corridor_width - gap_width
            if remain <= 0.05:
                continue
            side_width = max(block_width, 0.5 * remain)
            offset = rng.uniform(-offset_max, offset_max)
            left_center = -0.5 * gap_width - 0.5 * side_width + offset
            right_center = 0.5 * gap_width + 0.5 * side_width + offset
            obstacles.extend(self._tile_rect(left_center, center_y, side_width, gate_thickness))
            obstacles.extend(self._tile_rect(right_center, center_y, side_width, gate_thickness))

        obs_min = int(round(params.get("corridor_obstacle_count_min", 0)))
        obs_max = int(round(params.get("corridor_obstacle_count_max", obs_min)))
        if obs_min > 0 or obs_max > 0:
            obs_count = max(obs_min, rng.randint(obs_min, obs_max + 1))
            size_min = float(params.get("corridor_obstacle_size_min", self.block_size))
            size_max = float(params.get("corridor_obstacle_size_max", size_min))
            height_min = float(params.get("corridor_obstacle_height_min", self.block_height))
            height_max = float(params.get("corridor_obstacle_height_max", height_min))
            margin_x = float(params.get("corridor_obstacle_margin_x", 0.1))
            margin_y = float(params.get("corridor_obstacle_margin_y", self.scene_margin))
            clearance = float(params.get("corridor_obstacle_clearance", 0.5 * self.scene_clearance))

            def is_clear(x: float, y: float, size_xy: float) -> bool:
                for spec in obstacles:
                    dx = abs(x - spec.position[0])
                    dy = abs(y - spec.position[1])
                    limit_x = 0.5 * (size_xy + spec.size[0]) + clearance
                    limit_y = 0.5 * (size_xy + spec.size[1]) + clearance
                    if dx < limit_x and dy < limit_y:
                        return False
                return True

            for _ in range(obs_count):
                raw_size = rng.uniform(size_min, size_max)
                raw_height = rng.uniform(height_min, height_max)
                size_xy, height, _ = self._quantize_block(raw_size, raw_height)
                half_x = 0.5 * size_xy + margin_x
                half_y = 0.5 * size_xy + margin_y
                if corridor_width <= 2.0 * half_x:
                    continue
                max_tries = 30
                placed = False
                for _ in range(max_tries):
                    x = rng.uniform(-0.5 * corridor_width + half_x, 0.5 * corridor_width - half_x)
                    y = rng.uniform(-0.5 * length_m + half_y, 0.5 * length_m - half_y)
                    if is_clear(x, y, size_xy):
                        obstacles.append(
                            StaticObstacleSpec(
                                kind="block",
                                position=(x, y, 0.5 * height),
                                size=(size_xy, size_xy, height),
                                raw_size=(raw_size, raw_size, raw_height),
                            )
                        )
                        placed = True
                        break
                if not placed:
                    continue

        return obstacles

    def _build_doorway(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        width_m = self.scene_width
        length_m = self.scene_length
        room_width = float(params.get("room_width", 2.2))
        room_width = max(room_width, 2.0 * self.scene_clearance + 0.2)
        wall_thickness = float(params.get("wall_thickness", self.block_size))

        outer = max(0.0, width_m - room_width)
        if outer > 0:
            left_center = -(room_width * 0.5 + outer * 0.25)
            right_center = (room_width * 0.5 + outer * 0.25)
            obstacles.extend(self._tile_rect(left_center, 0.0, outer * 0.5, length_m, kind="wall"))
            obstacles.extend(self._tile_rect(right_center, 0.0, outer * 0.5, length_m, kind="wall"))

        room_count = int(round(params.get("room_count", params.get("door_count", 2))))
        room_count = max(2, room_count)
        door_thickness = float(params.get("door_thickness", wall_thickness))
        door_min = float(params.get("door_width_min", 0.8))
        door_max = float(params.get("door_width_max", 1.2))
        door_min = max(door_min, 2.0 * self.scene_clearance + 0.05)
        door_max = max(door_max, door_min)
        offset_max = float(params.get("door_offset_max", 0.5))
        block_width = float(params.get("door_block_width", self.block_size))
        jam_count = int(round(params.get("jam_count", 0)))
        jam_size = float(params.get("jam_size", self.block_size))

        room_len = length_m / float(room_count)
        boundary_jitter = float(params.get("room_boundary_jitter", 0.0))
        boundaries = []
        for idx in range(1, room_count):
            base_y = -0.5 * length_m + idx * room_len
            jitter = rng.uniform(-boundary_jitter, boundary_jitter) if boundary_jitter > 0.0 else 0.0
            boundaries.append(base_y + jitter)
        boundaries.sort()
        for center_y in boundaries:
            door_width = rng.uniform(door_min, door_max)
            door_width = min(door_width, room_width - 0.1)
            remain = room_width - door_width
            if remain <= 0.05:
                continue
            side_width = max(block_width, 0.5 * remain)
            offset = rng.uniform(-offset_max, offset_max)
            left_center = -0.5 * door_width - 0.5 * side_width + offset
            right_center = 0.5 * door_width + 0.5 * side_width + offset
            obstacles.extend(self._tile_rect(left_center, center_y, side_width, door_thickness))
            obstacles.extend(self._tile_rect(right_center, center_y, side_width, door_thickness))
            for _ in range(max(0, jam_count)):
                jam_side = rng.choice([-1.0, 1.0])
                jam_x = offset + jam_side * (0.5 * door_width + 0.5 * jam_size)
                jam_y = center_y + rng.uniform(-0.5, 0.5) * door_thickness
                obstacles.extend(self._tile_rect(jam_x, jam_y, jam_size, jam_size))

        room_obs_min = int(round(params.get("room_obstacle_count_min", 0)))
        room_obs_max = int(round(params.get("room_obstacle_count_max", room_obs_min)))
        if room_obs_min > 0 or room_obs_max > 0:
            size_min = float(params.get("room_obstacle_size_min", self.block_size))
            size_max = float(params.get("room_obstacle_size_max", size_min))
            height_min = float(params.get("room_obstacle_height_min", self.block_height))
            height_max = float(params.get("room_obstacle_height_max", height_min))
            margin_x = float(params.get("room_obstacle_margin_x", 0.1))
            margin_y = float(params.get("room_obstacle_margin_y", self.scene_margin))
            clearance = float(params.get("room_obstacle_clearance", 0.5 * self.scene_clearance))

            def is_clear(x: float, y: float, size_xy: float) -> bool:
                for spec in obstacles:
                    dx = abs(x - spec.position[0])
                    dy = abs(y - spec.position[1])
                    limit_x = 0.5 * (size_xy + spec.size[0]) + clearance
                    limit_y = 0.5 * (size_xy + spec.size[1]) + clearance
                    if dx < limit_x and dy < limit_y:
                        return False
                return True

            for room_idx in range(room_count):
                y_min = -0.5 * length_m + room_idx * room_len + margin_y
                y_max = -0.5 * length_m + (room_idx + 1) * room_len - margin_y
                if y_max <= y_min:
                    continue
                count = max(room_obs_min, rng.randint(room_obs_min, room_obs_max + 1))
                for _ in range(count):
                    raw_size = rng.uniform(size_min, size_max)
                    raw_height = rng.uniform(height_min, height_max)
                    size_xy, height, _ = self._quantize_block(raw_size, raw_height)
                    half_x = 0.5 * size_xy + margin_x
                    if room_width <= 2.0 * half_x:
                        continue
                    max_tries = 20
                    for _ in range(max_tries):
                        x = rng.uniform(-0.5 * room_width + half_x, 0.5 * room_width - half_x)
                        y = rng.uniform(y_min, y_max)
                        if is_clear(x, y, size_xy):
                            obstacles.append(
                                StaticObstacleSpec(
                                    kind="block",
                                    position=(x, y, 0.5 * height),
                                    size=(size_xy, size_xy, height),
                                    raw_size=(raw_size, raw_size, raw_height),
                                )
                            )
                            break
        return obstacles

    def _build_forest(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        width_m = self.scene_width
        length_m = self.scene_length
        count_min = int(round(params.get("pole_count_min", 8)))
        count_max = int(round(params.get("pole_count_max", 12)))
        count = max(count_min, rng.randint(count_min, count_max + 1))
        margin = float(params.get("pole_margin", 0.4))
        block_ratio = float(params.get("block_ratio", 0.0))

        for _ in range(count):
            x = rng.uniform(-0.5 * width_m + margin, 0.5 * width_m - margin)
            y = rng.uniform(-0.5 * length_m + margin, 0.5 * length_m - margin)
            obstacles.append(self._sample_obstacle_spec(x, y, params, rng, block_ratio))
        return obstacles

    def _build_crossing(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        static_count = int(round(params.get("static_pole_count", 0)))
        if static_count <= 0:
            return []
        params = dict(params)
        params["pole_count_min"] = static_count
        params["pole_count_max"] = static_count
        return self._build_forest(params, rng)

    def _build_density_transition(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        width_m = self.scene_width
        length_m = self.scene_length
        sparse_count = int(round(params.get("sparse_count", 6)))
        dense_count = int(round(params.get("dense_count", 16)))
        boundary_offset = float(params.get("boundary_offset", 0.0))
        boundary_jitter = float(params.get("boundary_jitter", 0.3))
        sparse_block_ratio = float(params.get("sparse_block_ratio", params.get("block_ratio", 0.0)))
        dense_block_ratio = float(params.get("dense_block_ratio", params.get("block_ratio", sparse_block_ratio)))

        boundary = boundary_offset + rng.uniform(-boundary_jitter, boundary_jitter)
        margin = self.scene_margin

        def sample(count: int, y_min: float, y_max: float, block_ratio: float) -> None:
            if y_max <= y_min or count <= 0:
                return
            for _ in range(count):
                x = rng.uniform(-0.5 * width_m + margin, 0.5 * width_m - margin)
                y = rng.uniform(y_min, y_max)
                obstacles.append(self._sample_obstacle_spec(x, y, params, rng, block_ratio))

        sample(sparse_count, -0.5 * length_m + margin, boundary - margin, sparse_block_ratio)
        sample(dense_count, boundary + margin, 0.5 * length_m - margin, dense_block_ratio)
        return obstacles

    def _build_structured_ood(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        template = str(params.get("ood_template", "mix"))
        if template == "mix":
            template = self._select_ood_template(params, rng)
        params["ood_template_selected"] = template
        if template == "cluster":
            return self._build_ood_cluster(params, rng)
        if template == "maze":
            return self._build_ood_maze(params, rng)
        return self._build_ood_nonconvex(params, rng)

    def _select_ood_template(self, params: Dict[str, object], rng: np.random.RandomState) -> str:
        templates = ["cluster", "nonconvex", "maze"]
        probs = params.get("ood_template_probs")
        if isinstance(probs, dict):
            values = [float(probs.get(name, 0.0)) for name in templates]
        elif isinstance(probs, (list, tuple)):
            values = [float(v) for v in probs]
        else:
            values = []
        if len(values) != len(templates) or sum(values) <= 0.0:
            values = [1.0] * len(templates)
        values = np.asarray(values, dtype=np.float32)
        values = values / np.sum(values)
        return str(rng.choice(templates, p=values))

    def _build_ood_nonconvex(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        u_width = float(params.get("u_width", 2.2))
        u_depth = float(params.get("u_depth", 1.6))
        u_thickness = float(params.get("u_thickness", 0.25))
        u_center_y = float(params.get("u_center_y", -0.5))
        left_x = -0.5 * u_width + 0.5 * u_thickness
        right_x = 0.5 * u_width - 0.5 * u_thickness
        bottom_y = u_center_y - 0.5 * u_depth + 0.5 * u_thickness
        obstacles.extend(self._tile_rect(0.0, bottom_y, u_width, u_thickness, kind="wall"))
        obstacles.extend(self._tile_rect(left_x, u_center_y, u_thickness, u_depth, kind="wall"))
        obstacles.extend(self._tile_rect(right_x, u_center_y, u_thickness, u_depth, kind="wall"))

        l_size = float(params.get("l_size", 1.6))
        l_thickness = float(params.get("l_thickness", 0.25))
        l_center_x = float(params.get("l_center_x", 1.2))
        l_center_y = float(params.get("l_center_y", 1.2))
        obstacles.extend(self._tile_rect(l_center_x, l_center_y, l_thickness, l_size, kind="wall"))
        obstacles.extend(
            self._tile_rect(l_center_x - 0.5 * l_size + 0.5 * l_thickness, l_center_y, l_size, l_thickness, kind="wall")
        )
        return obstacles

    def _build_ood_cluster(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        cluster_count = int(round(params.get("cluster_count", 8)))
        cluster_spread = float(params.get("cluster_spread", 0.8))
        cluster_radius = float(params.get("cluster_radius", 0.18))
        obstacle_height = float(params.get("obstacle_height", self.block_height))
        raw_size_xy = 2.0 * cluster_radius
        raw_height = obstacle_height
        size_xy, height, _ = self._quantize_block(raw_size_xy, obstacle_height)
        cluster_center = np.array(params.get("cluster_center", [-1.5, 1.5]), dtype=np.float32)
        for _ in range(max(0, cluster_count)):
            offset = rng.normal(scale=cluster_spread, size=2)
            x, y = (cluster_center + offset).tolist()
            obstacles.append(
                StaticObstacleSpec(
                    kind="block",
                    position=(x, y, 0.5 * height),
                    size=(size_xy, size_xy, height),
                    raw_size=(raw_size_xy, raw_size_xy, raw_height),
                )
            )
        return obstacles

    def _build_ood_maze(self, params: Dict[str, object], rng: np.random.RandomState) -> List[StaticObstacleSpec]:
        obstacles: List[StaticObstacleSpec] = []
        width_m = self.scene_width
        length_m = self.scene_length
        rows = int(round(params.get("maze_rows", 3)))
        cols = int(round(params.get("maze_cols", 3)))
        rows = max(2, rows)
        cols = max(2, cols)
        gap_width = float(params.get("maze_gap_width", 0.8))
        wall_thickness = float(params.get("maze_wall_thickness", self.block_size))
        wall_margin = float(params.get("maze_margin", self.scene_margin))

        cell_x = width_m / float(cols)
        cell_y = length_m / float(rows)

        def add_wall_segment(center_x: float, center_y: float, size_x: float, size_y: float) -> None:
            if size_x <= 0.0 or size_y <= 0.0:
                return
            obstacles.extend(self._tile_rect(center_x, center_y, size_x, size_y, kind="wall"))

        # Outer boundary walls
        add_wall_segment(0.0, -0.5 * length_m + 0.5 * wall_thickness, width_m, wall_thickness)
        add_wall_segment(0.0, 0.5 * length_m - 0.5 * wall_thickness, width_m, wall_thickness)
        add_wall_segment(-0.5 * width_m + 0.5 * wall_thickness, 0.0, wall_thickness, length_m)
        add_wall_segment(0.5 * width_m - 0.5 * wall_thickness, 0.0, wall_thickness, length_m)

        # Horizontal walls with gaps
        for r in range(1, rows):
            y = -0.5 * length_m + r * cell_y
            gap_center = rng.uniform(-0.5 * width_m + wall_margin, 0.5 * width_m - wall_margin)
            left_len = (gap_center - 0.5 * gap_width) - (-0.5 * width_m)
            right_len = (0.5 * width_m) - (gap_center + 0.5 * gap_width)
            add_wall_segment(-0.5 * width_m + 0.5 * left_len, y, left_len, wall_thickness)
            add_wall_segment(0.5 * width_m - 0.5 * right_len, y, right_len, wall_thickness)

        # Vertical walls with gaps
        for c in range(1, cols):
            x = -0.5 * width_m + c * cell_x
            gap_center = rng.uniform(-0.5 * length_m + wall_margin, 0.5 * length_m - wall_margin)
            bottom_len = (gap_center - 0.5 * gap_width) - (-0.5 * length_m)
            top_len = (0.5 * length_m) - (gap_center + 0.5 * gap_width)
            add_wall_segment(x, -0.5 * length_m + 0.5 * bottom_len, wall_thickness, bottom_len)
            add_wall_segment(x, 0.5 * length_m - 0.5 * top_len, wall_thickness, top_len)

        return obstacles

    def _fill_rect(self, terrain, center_x: float, center_y: float, size_x: float, size_y: float, height_m: float) -> None:
        h_scale = terrain.horizontal_scale
        v_scale = terrain.vertical_scale
        width = terrain.width
        length = terrain.length
        cx = width // 2
        cy = length // 2
        half_x = max(1, int(round(0.5 * size_x / h_scale)))
        half_y = max(1, int(round(0.5 * size_y / h_scale)))
        center_ix = int(round(cx + center_x / h_scale))
        center_iy = int(round(cy + center_y / h_scale))
        x1 = max(0, center_ix - half_x)
        x2 = min(width, center_ix + half_x + 1)
        y1 = max(0, center_iy - half_y)
        y2 = min(length, center_iy + half_y + 1)
        if x2 <= x1 or y2 <= y1:
            return
        height_cells = max(1, int(round(height_m / v_scale)))
        patch = terrain.height_field_raw[x1:x2, y1:y2]
        terrain.height_field_raw[x1:x2, y1:y2] = np.maximum(patch, height_cells)
