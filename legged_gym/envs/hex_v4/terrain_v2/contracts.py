from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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
    width_cells = max(1, int(round(width_m / h_scale)))
    length_cells = max(1, int(round(length_m / h_scale)))
    ix0 = max(0, min(width_cells, ix0))
    ix1 = max(0, min(width_cells, ix1))
    iy0 = max(0, min(length_cells, iy0))
    iy1 = max(0, min(length_cells, iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return iy0, iy1, ix0, ix1


def _rect_in_bounds(rect, width_m: float, length_m: float) -> bool:
    if rect is None:
        return False
    x0, x1, y0, y1 = rect
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0 >= -0.5 * width_m and x1 <= 0.5 * width_m and
            y0 >= -0.5 * length_m and y1 <= 0.5 * length_m and
            (x1 - x0) > 0.0 and (y1 - y0) > 0.0)


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
    if spawn_rect is None or goal_rect is None:
        return False
    free = _free_mask(heightfield)
    spawn_idx = _rect_indices(spawn_rect, width_m, length_m, h_scale)
    goal_idx = _rect_indices(goal_rect, width_m, length_m, h_scale)
    if spawn_idx is None or goal_idx is None:
        return False
    spawn_mask = np.zeros_like(free, dtype=np.bool_)
    goal_mask = np.zeros_like(free, dtype=np.bool_)
    sy0, sy1, sx0, sx1 = spawn_idx
    gy0, gy1, gx0, gx1 = goal_idx
    spawn_mask[sy0:sy1, sx0:sx1] = True
    goal_mask[gy0:gy1, gx0:gx1] = True
    visited = _reachable_from_mask(free, spawn_mask)
    return bool(np.any(visited & goal_mask))


def _check_s1(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Tuple[bool, Dict[str, Any], List[str]]:
    params = scene.params
    corridor_width = float(params.get("corridor_width_nom", params.get("corridor_width", 0.0)))
    gates = params.get("corridor_gates", [])
    gate_count = len(gates) if isinstance(gates, list) else 0
    door_widths = []
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            if gate.get("door_width") is None:
                continue
            door_widths.append(float(gate.get("door_width")))
    min_gate_width = float(min(door_widths)) if door_widths else corridor_width
    max_gate_width = float(max(door_widths)) if door_widths else corridor_width
    mean_gate_width = float(np.mean(door_widths)) if door_widths else corridor_width
    spawn_rect = params.get("spawn_rect_hf", scene.spawn_region)
    goal_rect = params.get("goal_rect_hf", scene.goal_region)
    width_m = float(scene.env_dims.get("width", corridor_width))
    length_m = float(scene.env_dims.get("length", params.get("corridor_length", 0.0)))
    reachable = _reachability_between_rects(heightfield, spawn_rect, goal_rect, width_m, length_m, h_scale)
    coverage_ok = _rect_in_bounds(spawn_rect, width_m, length_m) and _rect_in_bounds(goal_rect, width_m, length_m)

    reasons: List[str] = []
    if corridor_width < 1.2 - 1e-3 or corridor_width > 1.6 + 1e-3:
        reasons.append("corridor_width")
    if gate_count < 2 or gate_count > 3:
        reasons.append("gate_count")
    if gate_count > 0 and len(door_widths) != gate_count:
        reasons.append("door_width_missing")
    if min_gate_width < 0.65 - 1e-3 or min_gate_width > 0.9 + 1e-3:
        reasons.append("gate_width")
    if not reachable:
        reasons.append("reachability")
    if not coverage_ok:
        reasons.append("coverage")

    metrics = {
        "corridor_width": corridor_width,
        "gate_count": gate_count,
        "min_gate_width": min_gate_width,
        "max_gate_width": max_gate_width,
        "mean_gate_width": mean_gate_width,
        "reachability": reachable,
        "coverage_ok": coverage_ok,
    }
    return len(reasons) == 0, metrics, reasons


def _check_s2(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Tuple[bool, Dict[str, Any], List[str]]:
    params = scene.params
    count_min = int(params.get("count_min", 0))
    count_max = int(params.get("count_max", 0))
    placed = int(params.get("placed", 0))
    block_ratio = float(params.get("block_ratio", 0.0))
    spawn_rect = params.get("spawn_rect_hf", scene.spawn_region)
    goal_rect = params.get("goal_rect_hf", scene.goal_region)
    coverage_ok = _rect_in_bounds(spawn_rect, scene.env_dims["width"], scene.env_dims["length"]) and _rect_in_bounds(goal_rect, scene.env_dims["width"], scene.env_dims["length"])

    reasons: List[str] = []
    if placed < count_min or placed > count_max:
        reasons.append("count_range")

    reachable = _reachability_between_rects(heightfield, spawn_rect, goal_rect, scene.env_dims["width"], scene.env_dims["length"], h_scale)
    if not reachable:
        reasons.append("reachability")
    if not coverage_ok:
        reasons.append("coverage")

    metrics = {
        "count_min": count_min,
        "count_max": count_max,
        "placed": placed,
        "block_ratio": block_ratio,
        "reachability": reachable,
        "coverage_ok": coverage_ok,
    }
    return len(reasons) == 0, metrics, reasons


def check_scene(scene: SceneSpec, heightfield: np.ndarray, h_scale: float) -> Dict[str, Any]:
    if scene.scene_type == "s1_corridor_gate":
        ok, metrics, reasons = _check_s1(scene, heightfield, h_scale)
    elif scene.scene_type == "s2_forest":
        ok, metrics, reasons = _check_s2(scene, heightfield, h_scale)
    else:
        ok = False
        metrics = {"scene_type": scene.scene_type}
        reasons = ["unsupported_scene"]
    return {
        "pass": bool(ok),
        "metrics": metrics,
        "reasons": reasons,
    }
