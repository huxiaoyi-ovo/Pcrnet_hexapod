from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .scene_spec import SceneSpec


def _rect_indices(rect, width_m: float, length_m: float, h_scale: float) -> Optional[Tuple[int, int, int, int]]:
    x0, x1, y0, y1 = rect
    x_min = -0.5 * width_m
    y_min = 0.0
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = int(np.floor((x0 - x_min) / h_scale))
    ix1 = int(np.ceil((x1 - x_min) / h_scale))
    iy0 = int(np.floor((y0 - y_min) / h_scale))
    iy1 = int(np.ceil((y1 - y_min) / h_scale))
    ix0 = max(0, min(int(round(width_m / h_scale)), ix0))
    ix1 = max(0, min(int(round(width_m / h_scale)), ix1))
    iy0 = max(0, min(int(round(length_m / h_scale)), iy0))
    iy1 = max(0, min(int(round(length_m / h_scale)), iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, ix1, iy0, iy1


def apply_common_guards(heightfield: np.ndarray, scene: SceneSpec, h_scale: float, v_scale: float) -> np.ndarray:
    params = scene.params_resolved
    width_m = float(params.get("width_m", 0.0))
    length_m = float(params.get("length_m", 0.0))
    if width_m <= 0 or length_m <= 0:
        return heightfield

    edge_pad_width = float(params.get("edge_pad_width", 0.0))
    edge_pad_height = float(params.get("edge_pad_height", 0.0))
    if edge_pad_width > 0.0 and edge_pad_height > 0.0:
        pad_w = max(1, int(round(edge_pad_width / h_scale)))
        pad_h = max(1, int(round(edge_pad_height / v_scale)))
        heightfield[:pad_w, :] = np.maximum(heightfield[:pad_w, :], pad_h)
        heightfield[-pad_w:, :] = np.maximum(heightfield[-pad_w:, :], pad_h)
        heightfield[:, :pad_w] = np.maximum(heightfield[:, :pad_w], pad_h)
        heightfield[:, -pad_w:] = np.maximum(heightfield[:, -pad_w:], pad_h)

    for key in ("spawn_rect_hf", "goal_rect_hf"):
        rect = params.get(key, None)
        if rect is None:
            continue
        idx = _rect_indices(rect, width_m, length_m, h_scale)
        if idx is None:
            continue
        ix0, ix1, iy0, iy1 = idx
        heightfield[ix0:ix1, iy0:iy1] = 0

    return heightfield
