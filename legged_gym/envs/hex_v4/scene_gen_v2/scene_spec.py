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
class DynamicSpec:
    count_min: int
    count_max: int
    cross_width: float
    cross_span: float
    axis: str
    speed_min: float
    speed_max: float
    period_min: float
    period_max: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count_min": int(self.count_min),
            "count_max": int(self.count_max),
            "cross_width": float(self.cross_width),
            "cross_span": float(self.cross_span),
            "axis": str(self.axis),
            "speed_min": float(self.speed_min),
            "speed_max": float(self.speed_max),
            "period_min": float(self.period_min),
            "period_max": float(self.period_max),
        }


@dataclass(frozen=True)
class StaticObstacleSpec:
    kind: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    yaw: float = 0.0
    raw_size: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
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
class SceneSpec:
    scene_id: str
    scene_type: str
    seed: int
    difficulty: float
    params_resolved: Dict[str, Any]
    primitives: Tuple[object, ...]
    static_obstacles: Tuple[StaticObstacleSpec, ...] = ()
    dynamic_spec: Optional[DynamicSpec] = None
    layout_seed: Optional[int] = None
    layout_id: Optional[str] = None
    layout_hash: Optional[str] = None

    @property
    def params(self) -> Dict[str, Any]:
        return self.params_resolved

    def to_meta(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "seed": int(self.seed),
            "difficulty": float(self.difficulty),
            "layout_seed": self.layout_seed,
            "layout_id": self.layout_id,
            "layout_hash": self.layout_hash,
            "params": dict(self.params_resolved),
            "static_count": len(self.static_obstacles),
            "dynamic_template": None if self.dynamic_spec is None else self.dynamic_spec.to_dict(),
        }


def _json_payload(scene: SceneSpec) -> Dict[str, Any]:
    return {
        "scene_id": scene.scene_id,
        "seed": int(scene.seed),
        "difficulty": float(scene.difficulty),
        "params": scene.params_resolved,
        "primitives": [p.to_dict() for p in scene.primitives],
        "dynamic": None if scene.dynamic_spec is None else scene.dynamic_spec.to_dict(),
    }


def compute_layout_hash(scene: SceneSpec) -> str:
    payload = _json_payload(scene)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
