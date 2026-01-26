from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..scene_spec import RectWall, SceneSpec, make_default_spawn_goal_rects


def _q(val: float, scale: float) -> float:
    return round(float(val) / scale) * scale


def _q_pos(val: float, scale: float) -> float:
    return _q(val, scale)


def _q_size(val: float, scale: float) -> float:
    return max(scale, _q(val, scale))


def generate_debug_axis_calib(
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

    step_count = int(params.get("step_count", 6))
    step_height = float(params.get("step_height", 0.06))
    edge_margin = float(params.get("edge_margin", 0.8))

    step_count = max(2, step_count)
    step_height = max(v_scale, step_height)

    y_start = -0.5 * length_m + edge_margin
    y_end = 0.5 * length_m - edge_margin
    usable = max(h_scale, y_end - y_start)
    step_len = usable / float(step_count)

    primitives = []
    for i in range(step_count):
        y0 = y_start + i * step_len
        y1 = y_start + (i + 1) * step_len
        height = (i + 1) * step_height
        primitives.append(
            RectWall(
                x0=-0.5 * width_m,
                x1=0.5 * width_m,
                y0=_q_pos(y0, h_scale),
                y1=_q_pos(y1, h_scale),
                height=_q_size(height, v_scale),
            )
        )

    spawn_len = float(params.get("spawn_length", 1.0))
    goal_len = float(params.get("goal_length", 1.0))
    spawn_len = max(h_scale, spawn_len)
    goal_len = max(h_scale, goal_len)
    spawn_region, goal_region = make_default_spawn_goal_rects(
        env_dims,
        x_margin=0.1 * width_m,
        y_margin=h_scale,
        spawn_len=spawn_len,
        goal_len=goal_len,
    )

    resolved_params = {
        "width_m": width_m,
        "length_m": length_m,
        "axis": "+Y",
        "step_count": step_count,
        "step_height": step_height,
        "step_length": step_len,
        "spawn_rect_hf": [float(v) for v in spawn_region],
        "goal_rect_hf": [float(v) for v in goal_region],
        "edge_pad_width": 0.0,
        "edge_pad_height": 0.0,
    }

    meta = {
        "scene_type": "debug_axis_calib",
        "axis_rule": "y_only_steps",
    }

    return SceneSpec(
        scene_type="debug_axis_calib",
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
