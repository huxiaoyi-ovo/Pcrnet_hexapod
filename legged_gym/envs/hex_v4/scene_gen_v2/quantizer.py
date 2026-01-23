from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from .scene_spec import Box, Cylinder, RectWall, SceneSpec, StaticObstacleSpec, compute_layout_hash


def _q(val: float, scale: float) -> float:
    return round(float(val) / scale) * scale


def _q_size(val: float, scale: float) -> float:
    return max(scale, _q(val, scale))


def _quantize_rect(rect, h_scale: float):
    x0, x1, y0, y1 = rect
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    sx = abs(x1 - x0)
    sy = abs(y1 - y0)
    sx_q = _q_size(sx, h_scale)
    sy_q = _q_size(sy, h_scale)
    cx_q = _q(cx, h_scale)
    cy_q = _q(cy, h_scale)
    return [cx_q - 0.5 * sx_q, cx_q + 0.5 * sx_q, cy_q - 0.5 * sy_q, cy_q + 0.5 * sy_q]


def _quantize_params(params: Dict[str, Any], h_scale: float, v_scale: float) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(params)
    for key in ("corridor_length", "corridor_width_nom", "corridor_wall_thickness", "corridor_spawn_buffer",
                "corridor_spawn_span", "corridor_goal_buffer", "corridor_goal_margin", "corridor_goal_min_offset"):
        if key in out:
            out[key] = _q(float(out[key]), h_scale)
    if "corridor_wall_height" in out:
        out["corridor_wall_height"] = _q_size(float(out["corridor_wall_height"]), v_scale)
    if "corridor_x_center" in out:
        out["corridor_x_center"] = _q(float(out["corridor_x_center"]), h_scale)
    if "corridor_gates" in out and isinstance(out["corridor_gates"], list):
        gates_q = []
        for gate in out["corridor_gates"]:
            if not isinstance(gate, dict):
                continue
            gates_q.append(
                {
                    "y0": _q(float(gate.get("y0", 0.0)), h_scale),
                    "length": _q_size(float(gate.get("length", h_scale)), h_scale),
                    "width": _q_size(float(gate.get("width", h_scale)), h_scale),
                }
            )
        out["corridor_gates"] = gates_q
    if "spawn_rect_hf" in out and out["spawn_rect_hf"] is not None:
        out["spawn_rect_hf"] = _quantize_rect(out["spawn_rect_hf"], h_scale)
    if "goal_rect_hf" in out and out["goal_rect_hf"] is not None:
        out["goal_rect_hf"] = _quantize_rect(out["goal_rect_hf"], h_scale)
    if "edge_pad_width" in out:
        out["edge_pad_width"] = _q_size(float(out["edge_pad_width"]), h_scale)
    if "edge_pad_height" in out:
        out["edge_pad_height"] = _q_size(float(out["edge_pad_height"]), v_scale)
    return out


def quantize_scene(scene: SceneSpec, h_scale: float, v_scale: float) -> SceneSpec:
    primitives_q: List[object] = []
    for primitive in scene.primitives:
        if isinstance(primitive, RectWall):
            cx = 0.5 * (primitive.x0 + primitive.x1)
            cy = 0.5 * (primitive.y0 + primitive.y1)
            sx = abs(primitive.x1 - primitive.x0)
            sy = abs(primitive.y1 - primitive.y0)
            sx_q = _q_size(sx, h_scale)
            sy_q = _q_size(sy, h_scale)
            cx_q = _q(cx, h_scale)
            cy_q = _q(cy, h_scale)
            height_q = _q_size(primitive.height, v_scale)
            primitives_q.append(
                RectWall(
                    x0=cx_q - 0.5 * sx_q,
                    x1=cx_q + 0.5 * sx_q,
                    y0=cy_q - 0.5 * sy_q,
                    y1=cy_q + 0.5 * sy_q,
                    height=height_q,
                )
            )
        elif isinstance(primitive, Box):
            cx_q = _q(primitive.cx, h_scale)
            cy_q = _q(primitive.cy, h_scale)
            sx_q = _q_size(primitive.sx, h_scale)
            sy_q = _q_size(primitive.sy, h_scale)
            height_q = _q_size(primitive.height, v_scale)
            primitives_q.append(Box(cx=cx_q, cy=cy_q, sx=sx_q, sy=sy_q, height=height_q))
        elif isinstance(primitive, Cylinder):
            cx_q = _q(primitive.cx, h_scale)
            cy_q = _q(primitive.cy, h_scale)
            radius_q = _q_size(primitive.radius, h_scale)
            height_q = _q_size(primitive.height, v_scale)
            primitives_q.append(Cylinder(cx=cx_q, cy=cy_q, radius=radius_q, height=height_q))
        else:
            primitives_q.append(primitive)

    static_q: List[StaticObstacleSpec] = []
    for spec in scene.static_obstacles:
        px, py, pz = spec.position
        sx, sy, sz = spec.size
        pos_q = (_q(px, h_scale), _q(py, h_scale), _q(pz, v_scale))
        size_q = (_q_size(sx, h_scale), _q_size(sy, h_scale), _q_size(sz, v_scale))
        raw_size = spec.raw_size
        if raw_size is not None:
            rx, ry, rz = raw_size
            raw_size = (_q_size(rx, h_scale), _q_size(ry, h_scale), _q_size(rz, v_scale))
        static_q.append(
            StaticObstacleSpec(
                kind=spec.kind,
                position=pos_q,
                size=size_q,
                yaw=spec.yaw,
                raw_size=raw_size,
            )
        )

    params_q = _quantize_params(scene.params_resolved, h_scale, v_scale)
    scene_q = replace(
        scene,
        params_resolved=params_q,
        primitives=tuple(primitives_q),
        static_obstacles=tuple(static_q),
    )
    return replace(scene_q, layout_hash=compute_layout_hash(scene_q))
