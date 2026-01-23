from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .scene_spec import SceneSpec


def _free_mask(heightfield: np.ndarray) -> np.ndarray:
    return heightfield <= 0


def _rect_indices(rect, width_m: float, length_m: float, h_scale: float) -> Optional[Tuple[int, int, int, int]]:
    x0, x1, y0, y1 = rect
    x_min = -0.5 * width_m
    y_min = -0.5 * length_m
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = int(np.floor((x0 - x_min) / h_scale))
    ix1 = int(np.ceil((x1 - x_min) / h_scale))
    iy0 = int(np.floor((y0 - y_min) / h_scale))
    iy1 = int(np.ceil((y1 - y_min) / h_scale))
    width_cells = heightfield_width(width_m, h_scale)
    length_cells = heightfield_length(length_m, h_scale)
    ix0 = max(0, min(width_cells, ix0))
    ix1 = max(0, min(width_cells, ix1))
    iy0 = max(0, min(length_cells, iy0))
    iy1 = max(0, min(length_cells, iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return iy0, iy1, ix0, ix1


def heightfield_width(width_m: float, h_scale: float) -> int:
    return max(1, int(round(width_m / h_scale)))


def heightfield_length(length_m: float, h_scale: float) -> int:
    return max(1, int(round(length_m / h_scale)))


def _estimate_min_free_width(heightfield: np.ndarray, width_m: float, h_scale: float, x_center: float) -> float:
    free = _free_mask(heightfield)
    width_cells = heightfield.shape[1]
    x_min = -0.5 * width_m
    ix = int(round((x_center - x_min) / h_scale))
    ix = max(0, min(width_cells - 1, ix))
    min_width = float(width_m)
    for iy in range(heightfield.shape[0]):
        row = free[iy, :]
        if not row[ix]:
            continue
        left = ix
        right = ix
        while left - 1 >= 0 and row[left - 1]:
            left -= 1
        while right + 1 < width_cells and row[right + 1]:
            right += 1
        span = (right - left + 1) * h_scale
        min_width = min(min_width, span)
    return min_width


def _reachable_from_mask(free: np.ndarray, start_mask: np.ndarray) -> np.ndarray:
    h, w = free.shape[0], free.shape[1]
    visited = np.zeros_like(free, dtype=np.bool_)
    q = deque()
    starts = np.argwhere(start_mask & free)
    for y, x in starts:
        q.append((y, x))
        visited[y, x] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if visited[ny, nx] or not free[ny, nx]:
                continue
            visited[ny, nx] = True
            q.append((ny, nx))
    return visited


def _reachability_between_rects(heightfield: np.ndarray, spawn_rect, goal_rect, width_m: float, length_m: float,
                                h_scale: float) -> bool:
    free = _free_mask(heightfield)
    w_cells = heightfield_width(width_m, h_scale)
    l_cells = heightfield_length(length_m, h_scale)
    spawn_idx = _rect_indices(spawn_rect, width_m, length_m, h_scale) if spawn_rect else None
    goal_idx = _rect_indices(goal_rect, width_m, length_m, h_scale) if goal_rect else None
    if spawn_idx is None or goal_idx is None:
        return False
    spawn_mask = np.zeros((l_cells, w_cells), dtype=np.bool_)
    goal_mask = np.zeros((l_cells, w_cells), dtype=np.bool_)
    sy0, sy1, sx0, sx1 = spawn_idx
    gy0, gy1, gx0, gx1 = goal_idx
    spawn_mask[sy0:sy1, sx0:sx1] = True
    goal_mask[gy0:gy1, gx0:gx1] = True
    visited = _reachable_from_mask(free, spawn_mask)
    return bool(np.any(visited & goal_mask))


def _escape_to_border(heightfield: np.ndarray, start_mask: np.ndarray) -> bool:
    free = _free_mask(heightfield)
    visited = _reachable_from_mask(free, start_mask)
    if visited.size == 0:
        return False
    border = np.zeros_like(visited, dtype=np.bool_)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.any(visited & border))


def _check_s1(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Tuple[bool, Dict[str, Any]]:
    params = scene.params_resolved
    width_m = float(params.get("width_m", 0.0))
    length_m = float(params.get("corridor_length", params.get("length_m", 0.0)))
    x_center = float(params.get("corridor_x_center", 0.0))
    gates = params.get("corridor_gates", [])
    min_gate_width = None
    if isinstance(gates, list) and gates:
        min_gate_width = min(float(g.get("width", width_m)) for g in gates)
    else:
        min_gate_width = float(params.get("corridor_width_nom", width_m))
    min_free_width = _estimate_min_free_width(heightfield, width_m, h_scale, x_center)
    tol = 2.0 * h_scale
    x_min = -0.5 * width_m
    x_center_idx = int(round((x_center - x_min) / h_scale))
    x_center_idx = max(0, min(heightfield.shape[1] - 1, x_center_idx))
    free = _free_mask(heightfield)
    width_cells = heightfield.shape[1]
    widths = []
    for iy in range(heightfield.shape[0]):
        row = free[iy, :]
        if not row[x_center_idx]:
            widths.append(0.0)
            continue
        left = x_center_idx
        right = x_center_idx
        while left - 1 >= 0 and row[left - 1]:
            left -= 1
        while right + 1 < width_cells and row[right + 1]:
            right += 1
        widths.append((right - left + 1) * h_scale)
    narrow_thresh = float(min_gate_width) + 2.0 * h_scale
    narrow_flags = [w <= narrow_thresh for w in widths]
    num_narrow = 0
    prev = False
    for flag in narrow_flags:
        if flag and not prev:
            num_narrow += 1
        prev = flag
    spawn_rect = params.get("spawn_rect_hf", None)
    spawn_idx = _rect_indices(spawn_rect, width_m, length_m, h_scale) if spawn_rect else None
    spawn_mask = np.zeros_like(free, dtype=np.bool_)
    if spawn_idx is not None:
        sy0, sy1, sx0, sx1 = spawn_idx
        spawn_mask[sy0:sy1, sx0:sx1] = True
    else:
        spawn_mask[heightfield.shape[0] // 2, x_center_idx] = True
    outside_escape_ok = not _escape_to_border(heightfield, spawn_mask)
    ok = outside_escape_ok and length_m > 0.1 and min_free_width >= max(0.05, min_gate_width - tol)
    metrics = {
        "gate_count": len(gates) if isinstance(gates, list) else 0,
        "min_gate_width": float(min_gate_width),
        "min_free_width": float(min_free_width),
        "length": float(length_m),
        "num_narrow_segments": int(num_narrow),
        "outside_escape_ok": bool(outside_escape_ok),
    }
    return ok, metrics


def _check_s2(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Tuple[bool, Dict[str, Any]]:
    params = scene.params_resolved
    count_min = int(params.get("count_min", 1))
    count_max = int(params.get("count_max", count_min))
    min_dist_req = float(params.get("min_dist", 0.0))
    static_obs = scene.static_obstacles
    count = len(static_obs)
    min_dist = None
    if count >= 2:
        centers = np.array([[s.position[0], s.position[1]] for s in static_obs], dtype=np.float32)
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-9)
        dist += np.eye(count) * 1e6
        min_dist = float(np.min(dist))
    else:
        min_dist = float("inf")
    width_m = float(params.get("width_m", 0.0))
    length_m = float(params.get("length_m", 0.0))
    spawn_rect = params.get("spawn_rect_hf", None)
    goal_rect = params.get("goal_rect_hf", None)
    reach_ok = False
    if spawn_rect is not None and goal_rect is not None:
        reach_ok = _reachability_between_rects(heightfield, spawn_rect, goal_rect, width_m, length_m, h_scale)
    pass_ratio = float(np.mean(_free_mask(heightfield)))
    center_gap_p10 = None
    if count >= 2:
        nearest = np.min(dist, axis=1)
        center_gap_p10 = float(np.percentile(nearest, 10))
    ok = (count_min <= count <= count_max) and (min_dist >= min_dist_req * 0.8) and reach_ok
    metrics = {
        "count": count,
        "count_min": count_min,
        "count_max": count_max,
        "min_dist": float(min_dist),
        "min_dist_req": float(min_dist_req),
        "reachability_ok": bool(reach_ok),
        "passable_ratio": pass_ratio,
    }
    if center_gap_p10 is not None:
        metrics["center_gap_p10"] = center_gap_p10
    return ok, metrics


def check_scene(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Dict[str, Any]:
    if scene.scene_id == "s1_corridor":
        ok, metrics = _check_s1(scene, heightfield, h_scale)
    elif scene.scene_id == "s2_forest":
        ok, metrics = _check_s2(scene, heightfield, h_scale)
    else:
        ok, metrics = True, {}
    return {"pass": bool(ok), "metrics": metrics}
