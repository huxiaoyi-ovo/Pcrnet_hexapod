from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..scene_spec import RectWall, SceneSpec


def _lerp(a: float, b: float, d: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(d, 0.0, 1.0))


def _q(val: float, scale: float) -> float:
    return round(float(val) / scale) * scale


def _q_size(val: float, scale: float) -> float:
    return max(scale, _q(val, scale))


def _sample_gate_centers(rng: np.random.RandomState, count: int, y_min: float, y_max: float, min_gap: float,
                         lengths: List[float]) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    tries = 0
    while len(centers) < count and tries < 200:
        tries += 1
        idx = len(centers)
        glen = lengths[idx]
        y_center = float(rng.uniform(y_min, y_max))
        ok = True
        for cy, clen in centers:
            if abs(cy - y_center) < 0.5 * (clen + glen) + min_gap:
                ok = False
                break
        if ok:
            centers.append((y_center, glen))
    return centers


def generate_s1_corridor_gate(
    rng: np.random.RandomState,
    difficulty: float,
    seed: int,
    env_dims: Dict[str, float],
    robot_envelope: Dict[str, float],
    params: Dict[str, float],
    h_scale: float,
    v_scale: float,
) -> SceneSpec:
    width_m = float(env_dims["width"])
    length_m = float(env_dims["length"])

    corridor_width = float(params.get("corridor_width", _lerp(1.6, 1.2, difficulty)))
    gate_width = float(params.get("gate_width", _lerp(0.9, 0.65, difficulty)))
    gate_count = int(round(float(params.get("gate_count", _lerp(2.0, 3.0, difficulty)))))
    gate_length = float(params.get("gate_length", _lerp(0.9, 1.2, difficulty)))
    gate_length_jitter = float(params.get("gate_length_jitter", 0.2))
    gate_spacing_min = float(params.get("gate_spacing_min", 0.6))
    gate_margin_y = float(params.get("gate_margin_y", 0.8))

    wall_thickness = float(params.get("wall_thickness_m", 0.16))
    wall_height = float(params.get("wall_height_m", 0.5))

    corridor_length = float(params.get("corridor_length", min(length_m, 0.9 * length_m)))
    corridor_length = min(corridor_length, length_m)
    corridor_width = min(corridor_width, width_m - 2.0 * wall_thickness - 0.05)
    gate_width = min(gate_width, corridor_width)

    corridor_width = max(corridor_width, 0.6)
    gate_width = max(gate_width, 0.4)
    gate_count = max(1, gate_count)

    x_center = float(params.get("corridor_x_center", 0.0))
    y_start = -0.5 * corridor_length
    y_end = 0.5 * corridor_length

    gate_lengths: List[float] = []
    for _ in range(gate_count):
        jitter = float(rng.uniform(-gate_length_jitter, gate_length_jitter))
        gate_lengths.append(max(0.4, gate_length + jitter))

    y_min = y_start + gate_margin_y
    y_max = y_end - gate_margin_y
    centers = _sample_gate_centers(rng, gate_count, y_min, y_max, gate_spacing_min, gate_lengths)
    if len(centers) < gate_count:
        centers = centers[:]
        while len(centers) < gate_count:
            centers.append((y_start + (len(centers) + 1) * (corridor_length / (gate_count + 1)), gate_length))

    half_w = 0.5 * corridor_width
    gate_half = 0.5 * gate_width

    wall_h = _q_size(wall_height, v_scale)
    wall_t = _q_size(wall_thickness, h_scale)

    primitives: List[RectWall] = []
    # Outer walls
    primitives.append(
        RectWall(
            x0=x_center - half_w - wall_t,
            x1=x_center - half_w,
            y0=y_start,
            y1=y_end,
            height=wall_h,
        )
    )
    primitives.append(
        RectWall(
            x0=x_center + half_w,
            x1=x_center + half_w + wall_t,
            y0=y_start,
            y1=y_end,
            height=wall_h,
        )
    )

    gates: List[Dict[str, float]] = []
    for (y_center, glen) in centers:
        y0 = max(y_start, y_center - 0.5 * glen)
        y1 = min(y_end, y_center + 0.5 * glen)
        # Inner protrusions to narrow the corridor
        if gate_half < half_w:
            primitives.append(
                RectWall(
                    x0=x_center - half_w,
                    x1=x_center - gate_half,
                    y0=y0,
                    y1=y1,
                    height=wall_h,
                )
            )
            primitives.append(
                RectWall(
                    x0=x_center + gate_half,
                    x1=x_center + half_w,
                    y0=y0,
                    y1=y1,
                    height=wall_h,
                )
            )
        gates.append({"y0": float(y_center), "length": float(glen), "door_width": float(gate_width)})

    spawn_buffer = float(params.get("corridor_spawn_buffer", 0.6))
    spawn_span = float(params.get("corridor_spawn_span", max(1.5, 0.3 * corridor_length)))
    goal_buffer = float(params.get("corridor_goal_buffer", 0.6))
    goal_min_offset = float(params.get("corridor_goal_min_offset", 2.0))
    goal_margin = float(params.get("corridor_goal_margin", 0.2))

    spawn_region = (
        x_center - (half_w - goal_margin),
        x_center + (half_w - goal_margin),
        y_start + spawn_buffer,
        min(y_start + spawn_span, y_end - spawn_buffer),
    )
    goal_region = (
        x_center - (half_w - goal_margin),
        x_center + (half_w - goal_margin),
        max(y_start + goal_min_offset, y_end - 0.5 * corridor_length),
        y_end - goal_buffer,
    )

    resolved_params = {
        "width_m": width_m,
        "length_m": length_m,
        "corridor_length": corridor_length,
        "corridor_width": corridor_width,
        "corridor_width_nom": corridor_width,
        "corridor_x_center": x_center,
        "corridor_wall_thickness": wall_t,
        "corridor_wall_height": wall_h,
        "corridor_gates": gates,
        "corridor_spawn_buffer": spawn_buffer,
        "corridor_spawn_span": spawn_span,
        "corridor_goal_buffer": goal_buffer,
        "corridor_goal_min_offset": goal_min_offset,
        "corridor_goal_margin": goal_margin,
        "spawn_rect_hf": [float(v) for v in spawn_region],
        "goal_rect_hf": [float(v) for v in goal_region],
        "edge_pad_width": 0.0,
        "edge_pad_height": 0.0,
    }

    meta = {
        "scene_type": "s1_corridor_gate",
        "gate_count": len(gates),
    }

    return SceneSpec(
        scene_type="s1_corridor_gate",
        seed=int(seed),
        difficulty=float(difficulty),
        env_dims={"width": width_m, "length": length_m},
        resolved_params=resolved_params,
        spawn_region=spawn_region,
        goal_region=goal_region,
        obstacles=(),
        primitives=tuple(primitives),
        meta=meta,
        layout_seed=int(seed),
    )
