from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..scene_spec import Box, Cylinder, SceneSpec, StaticObstacleSpec, make_default_spawn_goal_rects


def _lerp(a: float, b: float, d: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(d, 0.0, 1.0))


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


def generate_s2_forest(
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

    count_min = int(round(float(params.get("count_min", _lerp(8, 16, difficulty)))))
    count_max = int(round(float(params.get("count_max", _lerp(12, 24, difficulty)))))

    pole_radius_min = float(params.get("pole_radius_min", _lerp(0.12, 0.14, difficulty)))
    pole_radius_max = float(params.get("pole_radius_max", _lerp(0.18, 0.22, difficulty)))
    pole_height_min = float(params.get("pole_height_min", _lerp(0.30, 0.35, difficulty)))
    pole_height_max = float(params.get("pole_height_max", _lerp(0.35, 0.40, difficulty)))

    block_size_min = float(params.get("block_size_min", _lerp(0.28, 0.30, difficulty)))
    block_size_max = float(params.get("block_size_max", _lerp(0.40, 0.44, difficulty)))
    block_height_min = float(params.get("block_height_min", _lerp(0.30, 0.35, difficulty)))
    block_height_max = float(params.get("block_height_max", _lerp(0.35, 0.40, difficulty)))

    block_ratio = float(params.get("block_ratio", _lerp(0.2, 0.4, difficulty)))
    min_dist = float(params.get("min_dist", 0.45))

    spawn_clear = float(params.get("spawn_clear", 1.0))
    goal_clear = float(params.get("goal_clear", 1.0))

    count_min = max(1, count_min)
    count_max = max(count_min, count_max)
    obstacle_count = int(rng.randint(count_min, count_max + 1))

    x_min = -0.5 * width_m + 0.3
    x_max = 0.5 * width_m - 0.3
    y_min = -0.5 * length_m + spawn_clear
    y_max = 0.5 * length_m - goal_clear
    bounds = ((x_min, x_max), (y_min, y_max))

    points = _sample_points(rng, obstacle_count, bounds, min_dist, max_tries=obstacle_count * 25 + 200)

    primitives = []
    obstacles: List[StaticObstacleSpec] = []
    for x, y in points:
        if rng.rand() < block_ratio:
            size = float(rng.uniform(block_size_min, block_size_max))
            height = float(rng.uniform(block_height_min, block_height_max))
            primitives.append(Box(cx=x, cy=y, sx=size, sy=size, height=height))
            obstacles.append(
                StaticObstacleSpec(
                    kind="box",
                    position=(x, y, 0.5 * height),
                    size=(size, size, height),
                    yaw=0.0,
                )
            )
        else:
            radius = float(rng.uniform(pole_radius_min, pole_radius_max))
            height = float(rng.uniform(pole_height_min, pole_height_max))
            primitives.append(Cylinder(cx=x, cy=y, radius=radius, height=height))
            obstacles.append(
                StaticObstacleSpec(
                    kind="cylinder",
                    position=(x, y, 0.5 * height),
                    size=(2.0 * radius, 2.0 * radius, height),
                    yaw=0.0,
                )
            )

    spawn_region, goal_region = make_default_spawn_goal_rects(
        env_dims,
        x_margin=0.1 * width_m,
        y_margin=h_scale,
        spawn_len=spawn_clear,
        goal_len=goal_clear,
    )

    resolved_params = {
        "width_m": width_m,
        "length_m": length_m,
        "count_min": count_min,
        "count_max": count_max,
        "placed": len(points),
        "pole_radius_min": pole_radius_min,
        "pole_radius_max": pole_radius_max,
        "pole_height_min": pole_height_min,
        "pole_height_max": pole_height_max,
        "block_size_min": block_size_min,
        "block_size_max": block_size_max,
        "block_height_min": block_height_min,
        "block_height_max": block_height_max,
        "block_ratio": block_ratio,
        "spawn_clear": spawn_clear,
        "goal_clear": goal_clear,
        "obstacles": [o.to_dict() for o in obstacles],
        "spawn_rect_hf": [float(v) for v in spawn_region],
        "goal_rect_hf": [float(v) for v in goal_region],
        "edge_pad_width": 0.0,
        "edge_pad_height": 0.0,
    }

    meta = {
        "scene_type": "s2_forest",
        "obstacle_count": len(points),
    }

    return SceneSpec(
        scene_type="s2_forest",
        seed=int(seed),
        difficulty=float(difficulty),
        env_dims={"width": width_m, "length": length_m},
        resolved_params=resolved_params,
        spawn_region=spawn_region,
        goal_region=goal_region,
        obstacles=tuple(obstacles),
        primitives=tuple(primitives),
        meta=meta,
        layout_seed=int(seed),
    )
