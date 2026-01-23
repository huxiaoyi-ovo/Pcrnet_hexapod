from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scene_spec import Box, Cylinder, RectWall, SceneSpec, StaticObstacleSpec, compute_layout_hash


def lerp(a: float, b: float, d: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(d, 0.0, 1.0))


def _resolve_scene_params(params: Optional[Dict[str, Any]], scene_id: str) -> Dict[str, Any]:
    if not params:
        return {}
    if scene_id in params and isinstance(params[scene_id], dict):
        return dict(params[scene_id])
    return dict(params)


def _mix_value(a: Any, b: Any, d: float) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return lerp(float(a), float(b), d)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return type(a)(lerp(float(x), float(y), d) for x, y in zip(a, b))
    return a if d <= 0.5 else b


def _mix_params(easy: Dict[str, Any], hard: Dict[str, Any], d: float) -> Dict[str, Any]:
    keys = set(easy.keys()) | set(hard.keys())
    mixed: Dict[str, Any] = {}
    for key in keys:
        if key in easy and key in hard:
            mixed[key] = _mix_value(easy[key], hard[key], d)
        else:
            mixed[key] = easy.get(key, hard.get(key))
    return mixed


def _sample_points(rng: np.random.RandomState, n: int, bounds: Tuple[Tuple[float, float], Tuple[float, float]],
                   min_dist: float, max_tries: int) -> List[Tuple[float, float]]:
    (x_min, x_max), (y_min, y_max) = bounds
    points: List[Tuple[float, float]] = []
    tries = 0
    while len(points) < n and tries < max_tries:
        tries += 1
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        ok = True
        for px, py in points:
            if (x - px) ** 2 + (y - py) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            points.append((x, y))
    return points


class SceneGenerator:
    _logged = False

    def __init__(self, cfg, env_dims: Optional[Dict[str, float]] = None,
                 robot_envelope: Optional[Dict[str, float]] = None) -> None:
        self.cfg = cfg
        self.env_dims = env_dims
        self.robot_envelope = robot_envelope
        self.scene_type = getattr(cfg, "scene_type", None)
        self.scene_types = list(getattr(cfg, "scene_types", []) or [])
        if not self.scene_types and self.scene_type:
            self.scene_types = [self.scene_type]
        self.scene_probs_easy = getattr(cfg, "scene_probs_easy", None)
        self.scene_probs_hard = getattr(cfg, "scene_probs_hard", None)
        self.scene_probs = getattr(cfg, "scene_probs", None)
        self.scene_seed = int(getattr(cfg, "scene_seed", 13))
        self.scene_margin = float(getattr(cfg, "scene_margin", 0.3))
        self.scene_clearance = float(getattr(cfg, "scene_clearance", 0.27))
        if not SceneGenerator._logged:
            tag = self.scene_type or ",".join(self.scene_types) or "unknown"
            print(f"[Init] 正在使用 scene_gen_v2 ({tag})")
            SceneGenerator._logged = True

    def set_env(self, env_dims: Dict[str, float], robot_envelope: Optional[Dict[str, float]] = None) -> None:
        self.env_dims = env_dims
        if robot_envelope is not None:
            self.robot_envelope = robot_envelope

    @property
    def has_dynamic(self) -> bool:
        return False

    def seed_for_env(self, env_id: int, episode_idx: int) -> int:
        return int(self.scene_seed + env_id * 10007 + episode_idx * 131)

    def _select_scene_type(self, rng: np.random.RandomState, difficulty: float) -> str:
        if self.scene_type:
            return self.scene_type
        if not self.scene_types:
            raise RuntimeError("scene_types is empty")
        probs = None
        if self.scene_probs_easy is not None and self.scene_probs_hard is not None:
            probs_easy = list(self.scene_probs_easy)
            probs_hard = list(self.scene_probs_hard)
            if len(probs_easy) == len(self.scene_types) and len(probs_hard) == len(self.scene_types):
                probs = [lerp(pe, ph, difficulty) for pe, ph in zip(probs_easy, probs_hard)]
        elif self.scene_probs is not None:
            probs = list(self.scene_probs)
        if probs is None or len(probs) != len(self.scene_types):
            probs = [1.0 / len(self.scene_types)] * len(self.scene_types)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / np.sum(probs)
        idx = int(rng.choice(len(self.scene_types), p=probs))
        return self.scene_types[idx]

    def sample(
        self,
        scene_id: str,
        difficulty: float,
        seed: int,
        env_dims: Optional[Dict[str, float]] = None,
        robot_envelope: Optional[Dict[str, float]] = None,
    ) -> SceneSpec:
        rng = np.random.RandomState(int(seed))
        if env_dims is None:
            env_dims = self.env_dims
        if env_dims is None:
            raise RuntimeError("scene_gen_v2 missing env_dims")
        if robot_envelope is None:
            robot_envelope = self.robot_envelope
        if robot_envelope is None:
            robot_envelope = {"clearance": self.scene_clearance}
        if scene_id == "s1_corridor":
            return self._sample_s1(rng, difficulty, seed, env_dims, robot_envelope)
        if scene_id == "s2_forest":
            return self._sample_s2(rng, difficulty, seed, env_dims, robot_envelope)
        raise RuntimeError(f"scene_gen_v2 unsupported scene_id={scene_id}")

    def _sample_s1(
        self,
        rng: np.random.RandomState,
        difficulty: float,
        seed: int,
        env_dims: Dict[str, float],
        robot_envelope: Optional[Dict[str, float]],
    ) -> SceneSpec:
        easy = _resolve_scene_params(getattr(self.cfg, "scene_params_easy", {}), "s1_corridor")
        hard = _resolve_scene_params(getattr(self.cfg, "scene_params_hard", {}), "s1_corridor")
        params = _mix_params(easy, hard, difficulty)
        width_m = float(env_dims["width"])
        length_m = float(env_dims["length"])
        clearance = float((robot_envelope or {}).get("clearance", self.scene_clearance))

        length = float(params.get("corridor_length", length_m))
        width = float(params.get("corridor_width", width_m))
        wall_height = float(params.get("corridor_wall_height", self.cfg.scene_static_wall_block_height))
        center_jitter = float(params.get("corridor_center_x_jitter", 0.0))
        gate_count = int(round(float(params.get("corridor_gate_count", 2))))
        gate_length = float(params.get("corridor_gate_length", 1.0))
        gate_length_jitter = float(params.get("corridor_gate_length_jitter", 0.0))
        gate_width = float(params.get("corridor_gate_width", 0.8))
        gate_width_jitter = float(params.get("corridor_gate_width_jitter", 0.0))
        gate_spacing_min = float(params.get("corridor_gate_spacing_min", 0.8))
        gate_margin_y = float(params.get("corridor_gate_margin_y", 0.6))
        pass_margin = float(params.get("corridor_gate_pass_margin", 0.0))
        spawn_buffer = float(params.get("corridor_spawn_buffer", self.scene_margin))
        spawn_span = float(params.get("corridor_spawn_span", 2.0))
        goal_min_offset = float(params.get("corridor_goal_min_offset", 2.0))
        goal_buffer = float(params.get("corridor_goal_buffer", self.scene_margin))
        goal_margin = float(params.get("corridor_goal_margin", self.scene_margin))

        length = min(length, length_m)
        width = min(width, width_m)
        min_width = max(2.0 * clearance + pass_margin, 0.05)
        gate_count = max(1, gate_count)

        x_center = float(rng.uniform(-center_jitter, center_jitter)) if center_jitter > 1e-6 else 0.0
        wall_thickness = float(params.get("corridor_wall_block_size", getattr(self.cfg, "scene_static_wall_block_size", 0.4)))
        wall_thickness = max(wall_thickness, 0.05)

        gates: List[Dict[str, float]] = []
        centers: List[Tuple[float, float]] = []
        gate_y_start = -0.5 * length
        gate_y_end = 0.5 * length
        y_min = gate_y_start + gate_margin_y
        y_max = gate_y_end - gate_margin_y
        for _ in range(gate_count):
            g_len = max(0.2, gate_length + rng.uniform(-gate_length_jitter, gate_length_jitter))
            g_width = max(min_width, gate_width + rng.uniform(-gate_width_jitter, gate_width_jitter))
            placed = False
            for _ in range(80):
                y_center_abs = float(rng.uniform(y_min, y_max))
                ok = True
                for existing_center, existing_len in centers:
                    if abs(y_center_abs - existing_center) < 0.5 * (existing_len + g_len) + gate_spacing_min:
                        ok = False
                        break
                if ok:
                    centers.append((y_center_abs, g_len))
                    gates.append({"y0": y_center_abs, "length": g_len, "width": g_width})
                    placed = True
                    break
            if not placed and y_max > y_min:
                y_center_abs = float(np.clip(y_min + 0.5 * g_len, y_min, y_max))
                centers.append((y_center_abs, g_len))
                gates.append({"y0": y_center_abs, "length": g_len, "width": g_width})

        half_w = 0.5 * width
        wall_y_start = -0.5 * length_m
        wall_y_end = 0.5 * length_m
        left_wall = RectWall(
            x0=x_center - half_w - wall_thickness,
            x1=x_center - half_w,
            y0=wall_y_start,
            y1=wall_y_end,
            height=wall_height,
        )
        right_wall = RectWall(
            x0=x_center + half_w,
            x1=x_center + half_w + wall_thickness,
            y0=wall_y_start,
            y1=wall_y_end,
            height=wall_height,
        )
        primitives: List[object] = [left_wall, right_wall]
        for gate in gates:
            y_center_abs = float(gate["y0"])
            g_len = float(gate["length"])
            g_width = float(gate["width"])
            y0 = max(gate_y_start, y_center_abs - 0.5 * g_len)
            y1 = min(gate_y_end, y_center_abs + 0.5 * g_len)
            half_gate = 0.5 * g_width
            primitives.append(
                RectWall(
                    x0=x_center - half_gate - wall_thickness,
                    x1=x_center - half_gate,
                    y0=y0,
                    y1=y1,
                    height=wall_height,
                )
            )
            primitives.append(
                RectWall(
                    x0=x_center + half_gate,
                    x1=x_center + half_gate + wall_thickness,
                    y0=y0,
                    y1=y1,
                    height=wall_height,
                )
            )

        spawn_y0 = max(gate_y_start, gate_y_start + spawn_buffer)
        spawn_y1 = min(gate_y_end, max(spawn_y0 + clearance, gate_y_start + spawn_span))
        goal_y1 = gate_y_end
        goal_y0 = max(gate_y_start, gate_y_end - max(goal_buffer, clearance))
        edge_pad_width = max(self.scene_margin, clearance)
        edge_pad_height = wall_height
        edge_pad_enable = bool(params.get("edge_pad_enable", getattr(self.cfg, "scene_edge_pad_enable", False)))
        if not edge_pad_enable:
            edge_pad_width = 0.0
            edge_pad_height = 0.0

        resolved = {
            "clearance": clearance,
            "width_m": width_m,
            "length_m": length_m,
            "corridor_length": length,
            "corridor_width_nom": width,
            "corridor_x_center": x_center,
            "corridor_gates": gates,
            "corridor_wall_height": wall_height,
            "corridor_wall_thickness": wall_thickness,
            "corridor_spawn_buffer": spawn_buffer,
            "corridor_spawn_span": spawn_span,
            "corridor_goal_min_offset": goal_min_offset,
            "corridor_goal_buffer": goal_buffer,
            "corridor_goal_margin": goal_margin,
            "spawn_rect_hf": [x_center - half_w, x_center + half_w, spawn_y0, spawn_y1],
            "goal_rect_hf": [x_center - half_w, x_center + half_w, goal_y0, goal_y1],
            "edge_pad_width": edge_pad_width,
            "edge_pad_height": edge_pad_height,
        }
        scene = SceneSpec(
            scene_id="s1_corridor",
            scene_type="s1_corridor",
            seed=int(seed),
            difficulty=float(difficulty),
            params_resolved=resolved,
            primitives=tuple(primitives),
            static_obstacles=tuple(),
            dynamic_spec=None,
            layout_seed=int(seed),
            layout_id=str(seed),
            layout_hash=None,
        )
        return scene.__class__(
            scene_id=scene.scene_id,
            scene_type=scene.scene_type,
            seed=scene.seed,
            difficulty=scene.difficulty,
            params_resolved=scene.params_resolved,
            primitives=scene.primitives,
            static_obstacles=scene.static_obstacles,
            dynamic_spec=scene.dynamic_spec,
            layout_seed=scene.layout_seed,
            layout_id=scene.layout_id,
            layout_hash=compute_layout_hash(scene),
        )

    def _sample_s2(
        self,
        rng: np.random.RandomState,
        difficulty: float,
        seed: int,
        env_dims: Dict[str, float],
        robot_envelope: Optional[Dict[str, float]],
    ) -> SceneSpec:
        easy = _resolve_scene_params(getattr(self.cfg, "scene_params_easy", {}), "s2_forest")
        hard = _resolve_scene_params(getattr(self.cfg, "scene_params_hard", {}), "s2_forest")
        params = _mix_params(easy, hard, difficulty)
        width_m = float(env_dims["width"])
        length_m = float(env_dims["length"])
        clearance = float((robot_envelope or {}).get("clearance", self.scene_clearance))

        count_min = int(round(float(params.get("pole_count_min", 8))))
        count_max = int(round(float(params.get("pole_count_max", 12))))
        pole_radius_min = float(params.get("pole_radius_min", 0.12))
        pole_radius_max = float(params.get("pole_radius_max", 0.18))
        pole_height_min = float(params.get("pole_height_min", 0.3))
        pole_height_max = float(params.get("pole_height_max", 0.35))
        pole_margin = float(params.get("pole_margin", 0.4))
        block_ratio = float(params.get("block_ratio", 0.2))
        block_size_min = float(params.get("block_size_min", 0.28))
        block_size_max = float(params.get("block_size_max", 0.4))
        block_height_min = float(params.get("block_height_min", 0.3))
        block_height_max = float(params.get("block_height_max", 0.35))

        count_min = max(1, count_min)
        count_max = max(count_min, count_max)
        total_count = int(rng.randint(count_min, count_max + 1))
        min_dist = max(2.0 * clearance, pole_margin)

        bounds = (
            (-0.5 * width_m + pole_margin, 0.5 * width_m - pole_margin),
            (-0.5 * length_m + pole_margin, 0.5 * length_m - pole_margin),
        )
        points: List[Tuple[float, float]] = []
        min_dist_try = min_dist
        for _ in range(4):
            points = _sample_points(rng, total_count, bounds, min_dist_try, max_tries=total_count * 60 + 300)
            if len(points) >= count_min:
                break
            min_dist_try = max(clearance, 0.85 * min_dist_try)
        min_dist = float(min_dist_try)

        primitives: List[object] = []
        static_obs: List[StaticObstacleSpec] = []
        for x, y in points:
            if rng.rand() < block_ratio:
                size = float(rng.uniform(block_size_min, block_size_max))
                height = float(rng.uniform(block_height_min, block_height_max))
                primitives.append(Box(cx=x, cy=y, sx=size, sy=size, height=height))
                static_obs.append(
                    StaticObstacleSpec(
                        kind="block",
                        position=(x, y, 0.5 * height),
                        size=(size, size, height),
                        yaw=0.0,
                        raw_size=(size, size, height),
                    )
                )
            else:
                radius = float(rng.uniform(pole_radius_min, pole_radius_max))
                height = float(rng.uniform(pole_height_min, pole_height_max))
                primitives.append(Cylinder(cx=x, cy=y, radius=radius, height=height))
                static_obs.append(
                    StaticObstacleSpec(
                        kind="pole",
                        position=(x, y, 0.5 * height),
                        size=(2.0 * radius, 2.0 * radius, height),
                        yaw=0.0,
                        raw_size=(2.0 * radius, 2.0 * radius, height),
                    )
                )

        spawn_clear = max(self.scene_margin, 2.0 * clearance)
        y_start = -0.5 * length_m
        y_end = 0.5 * length_m
        spawn_rect = [-0.5 * width_m, 0.5 * width_m, y_start, min(y_end, y_start + spawn_clear)]
        goal_rect = [-0.5 * width_m, 0.5 * width_m, max(y_start, y_end - spawn_clear), y_end]
        edge_pad_width = max(self.scene_margin, clearance)
        edge_pad_height = max(pole_height_max, block_height_max, 0.2)
        edge_pad_enable = bool(params.get("edge_pad_enable", getattr(self.cfg, "scene_edge_pad_enable", False)))
        if not edge_pad_enable:
            edge_pad_width = 0.0
            edge_pad_height = 0.0

        resolved = {
            "clearance": clearance,
            "width_m": width_m,
            "length_m": length_m,
            "count_min": count_min,
            "count_max": count_max,
            "min_dist": min_dist,
            "block_ratio": block_ratio,
            "pole_radius_min": pole_radius_min,
            "pole_radius_max": pole_radius_max,
            "block_size_min": block_size_min,
            "block_size_max": block_size_max,
            "placed": len(points),
            "spawn_rect_hf": spawn_rect,
            "goal_rect_hf": goal_rect,
            "edge_pad_width": edge_pad_width,
            "edge_pad_height": edge_pad_height,
        }
        scene = SceneSpec(
            scene_id="s2_forest",
            scene_type="s2_forest",
            seed=int(seed),
            difficulty=float(difficulty),
            params_resolved=resolved,
            primitives=tuple(primitives),
            static_obstacles=tuple(static_obs),
            dynamic_spec=None,
            layout_seed=int(seed),
            layout_id=str(seed),
            layout_hash=None,
        )
        return scene.__class__(
            scene_id=scene.scene_id,
            scene_type=scene.scene_type,
            seed=scene.seed,
            difficulty=scene.difficulty,
            params_resolved=scene.params_resolved,
            primitives=scene.primitives,
            static_obstacles=scene.static_obstacles,
            dynamic_spec=scene.dynamic_spec,
            layout_seed=scene.layout_seed,
            layout_id=scene.layout_id,
            layout_hash=compute_layout_hash(scene),
        )
