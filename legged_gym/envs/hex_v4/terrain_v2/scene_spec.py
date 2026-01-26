from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class RectWall:
    x0: float
    x1: float
    y0: float
    y1: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "rect_wall",
            "x0": float(self.x0),
            "x1": float(self.x1),
            "y0": float(self.y0),
            "y1": float(self.y1),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    sx: float
    sy: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "box",
            "cx": float(self.cx),
            "cy": float(self.cy),
            "sx": float(self.sx),
            "sy": float(self.sy),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class Cylinder:
    cx: float
    cy: float
    radius: float
    height: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "cylinder",
            "cx": float(self.cx),
            "cy": float(self.cy),
            "radius": float(self.radius),
            "height": float(self.height),
        }


@dataclass(frozen=True)
class StaticObstacleSpec:
    kind: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind),
            "position": [float(v) for v in self.position],
            "size": [float(v) for v in self.size],
            "yaw": float(self.yaw),
        }


@dataclass(frozen=True)
class SceneSpec:
    """SceneSpec for terrain_v2.

    rect_hf convention (in resolved_params):
      - Keys: spawn_rect_hf / goal_rect_hf
      - Format: [x0, x1, y0, y1) in meters (half-open), aligned to horizontal_scale
      - Must be clamped within env_dims (width/length)
    """
    scene_type: str
    seed: int
    difficulty: float
    env_dims: Dict[str, float]
    resolved_params: Dict[str, Any]
    spawn_region: Optional[Tuple[float, float, float, float]]
    goal_region: Optional[Tuple[float, float, float, float]]
    obstacles: Tuple[StaticObstacleSpec, ...]
    primitives: Tuple[object, ...]
    meta: Dict[str, Any]
    layout_seed: Optional[int] = None
    layout_id: Optional[str] = None
    layout_hash: Optional[str] = None
    scene_id: Optional[str] = None

    @property
    def params(self) -> Dict[str, Any]:
        return self.resolved_params

    @property
    def static_obstacles(self) -> Tuple[StaticObstacleSpec, ...]:
        return self.obstacles

    def to_meta(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "seed": int(self.seed),
            "difficulty": float(self.difficulty),
            "env_dims": dict(self.env_dims),
            "layout_seed": self.layout_seed,
            "layout_id": self.layout_id,
            "layout_hash": self.layout_hash,
            "params": dict(self.resolved_params),
            "obstacle_count": len(self.obstacles),
            "spawn_region": None if self.spawn_region is None else list(self.spawn_region),
            "goal_region": None if self.goal_region is None else list(self.goal_region),
            "meta": dict(self.meta),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "seed": int(self.seed),
            "difficulty": float(self.difficulty),
            "env_dims": dict(self.env_dims),
            "resolved_params": dict(self.resolved_params),
            "spawn_region": None if self.spawn_region is None else list(self.spawn_region),
            "goal_region": None if self.goal_region is None else list(self.goal_region),
            "obstacles": [o.to_dict() for o in self.obstacles],
            "primitives": [p.to_dict() for p in self.primitives],
            "meta": dict(self.meta),
            "layout_seed": self.layout_seed,
            "layout_id": self.layout_id,
            "layout_hash": self.layout_hash,
            "scene_id": self.scene_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _json_payload(scene: SceneSpec) -> Dict[str, Any]:
    return {
        "scene_type": scene.scene_type,
        "seed": int(scene.seed),
        "difficulty": float(scene.difficulty),
        "env_dims": scene.env_dims,
        "params": scene.resolved_params,
        "spawn_region": scene.spawn_region,
        "goal_region": scene.goal_region,
        "primitives": [p.to_dict() for p in scene.primitives],
        "obstacles": [o.to_dict() for o in scene.obstacles],
    }


def compute_layout_hash(scene: SceneSpec) -> str:
    payload = _json_payload(scene)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def make_default_spawn_goal_rects(
    env_dims: Dict[str, float],
    margin: float = 0.4,
    spawn_len: float = 1.0,
    goal_len: float = 1.0,
    x_margin: Optional[float] = None,
    y_margin: Optional[float] = None,
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    width_m = float(env_dims["width"])
    length_m = float(env_dims["length"])
    if width_m <= 0.0 or length_m <= 0.0:
        raise RuntimeError(f"invalid env_dims for rect_hf: {env_dims}")
    x_margin = float(margin if x_margin is None else x_margin)
    y_margin = float(margin if y_margin is None else y_margin)
    x0 = -0.5 * width_m + x_margin
    x1 = 0.5 * width_m - x_margin
    y_min = -0.5 * length_m + y_margin
    y_max = 0.5 * length_m - y_margin
    if x1 <= x0 or y_max <= y_min:
        raise RuntimeError(f"rect_hf margin too large: env_dims={env_dims} margin={margin}")
    spawn_len = max(0.0, float(spawn_len))
    goal_len = max(0.0, float(goal_len))
    spawn_y1 = min(y_min + spawn_len, y_max)
    goal_y0 = max(y_max - goal_len, y_min)
    if spawn_y1 <= y_min or y_max <= goal_y0:
        raise RuntimeError(f"rect_hf lengths invalid: spawn_len={spawn_len} goal_len={goal_len}")
    spawn_rect = (x0, x1, y_min, spawn_y1)
    goal_rect = (x0, x1, goal_y0, y_max)
    return spawn_rect, goal_rect
