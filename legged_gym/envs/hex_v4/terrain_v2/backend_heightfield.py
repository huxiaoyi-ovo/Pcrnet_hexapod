from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .scene_spec import Box, Cylinder, RectWall, SceneSpec
from .quantizer import quantize_scene


def _rect_indices(x0: float, x1: float, y0: float, y1: float, x_min: float, y_min: float,
                  h_scale: float, width: int, length: int) -> Optional[Tuple[int, int, int, int]]:
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    ix0 = int(math.floor((x0 - x_min) / h_scale))
    ix1 = int(math.ceil((x1 - x_min) / h_scale))
    iy0 = int(math.floor((y0 - y_min) / h_scale))
    iy1 = int(math.ceil((y1 - y_min) / h_scale))
    ix0 = max(0, min(width, ix0))
    ix1 = max(0, min(width, ix1))
    iy0 = max(0, min(length, iy0))
    iy1 = max(0, min(length, iy1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return iy0, iy1, ix0, ix1


def _stamp_box(hf: np.ndarray, x0: float, x1: float, y0: float, y1: float, height_cells: int,
               x_min: float, y_min: float, h_scale: float):
    idx = _rect_indices(x0, x1, y0, y1, x_min, y_min, h_scale, hf.shape[1], hf.shape[0])
    if idx is None:
        return
    iy0, iy1, ix0, ix1 = idx
    hf[iy0:iy1, ix0:ix1] = np.maximum(hf[iy0:iy1, ix0:ix1], height_cells)


def _stamp_cylinder(hf: np.ndarray, cx: float, cy: float, radius: float, height_cells: int,
                    x_min: float, y_min: float, h_scale: float):
    if radius <= 0.0:
        return
    ix = int(round((cx - x_min) / h_scale))
    iy = int(round((cy - y_min) / h_scale))
    r_cells = max(1, int(round(radius / h_scale)))
    x1 = max(0, ix - r_cells)
    x2 = min(hf.shape[1], ix + r_cells + 1)
    y1 = max(0, iy - r_cells)
    y2 = min(hf.shape[0], iy + r_cells + 1)
    if x2 <= x1 or y2 <= y1:
        return
    xs = np.arange(x1, x2) - ix
    ys = np.arange(y1, y2) - iy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    mask = (xx * xx + yy * yy) <= r_cells * r_cells
    patch = hf[y1:y2, x1:x2]
    patch[mask] = np.maximum(patch[mask], height_cells)
    hf[y1:y2, x1:x2] = patch


def _clear_rect(hf: np.ndarray, rect, x_min: float, y_min: float, h_scale: float):
    x0, x1, y0, y1 = rect
    idx = _rect_indices(x0, x1, y0, y1, x_min, y_min, h_scale, hf.shape[1], hf.shape[0])
    if idx is None:
        return
    iy0, iy1, ix0, ix1 = idx
    hf[iy0:iy1, ix0:ix1] = 0


class HeightfieldBackend:
    def __init__(self, width_m: float, length_m: float, horizontal_scale: float, vertical_scale: float) -> None:
        self.width_m = float(width_m)
        self.length_m = float(length_m)
        self.horizontal_scale = float(horizontal_scale)
        self.vertical_scale = float(vertical_scale)

    def render(self, scene: SceneSpec, return_scene: bool = False):
        scene = quantize_scene(scene, self.horizontal_scale, self.vertical_scale)
        width = max(1, int(round(self.width_m / self.horizontal_scale)))
        length = max(1, int(round(self.length_m / self.horizontal_scale)))
        hf = np.zeros((length, width), dtype=np.int16)
        x_min = -0.5 * self.width_m
        y_min = -0.5 * self.length_m

        for primitive in scene.primitives:
            height = float(getattr(primitive, "height", 0.0))
            height_cells = max(1, int(round(height / self.vertical_scale))) if height > 0 else 0
            if isinstance(primitive, RectWall):
                _stamp_box(hf, primitive.x0, primitive.x1, primitive.y0, primitive.y1, height_cells,
                           x_min, y_min, self.horizontal_scale)
            elif isinstance(primitive, Box):
                x0 = primitive.cx - 0.5 * primitive.sx
                x1 = primitive.cx + 0.5 * primitive.sx
                y0 = primitive.cy - 0.5 * primitive.sy
                y1 = primitive.cy + 0.5 * primitive.sy
                _stamp_box(hf, x0, x1, y0, y1, height_cells, x_min, y_min, self.horizontal_scale)
            elif isinstance(primitive, Cylinder):
                _stamp_cylinder(hf, primitive.cx, primitive.cy, primitive.radius, height_cells,
                                x_min, y_min, self.horizontal_scale)

        spawn_rect = scene.params.get("spawn_rect_hf")
        goal_rect = scene.params.get("goal_rect_hf")
        if spawn_rect is None or goal_rect is None:
            raise RuntimeError("terrain_v2 requires spawn_rect_hf/goal_rect_hf (rect_hf) for clearing")
        _clear_rect(hf, spawn_rect, x_min, y_min, self.horizontal_scale)
        _clear_rect(hf, goal_rect, x_min, y_min, self.horizontal_scale)

        edge_pad_width = float(scene.params.get("edge_pad_width", 0.0))
        edge_pad_height = float(scene.params.get("edge_pad_height", 0.0))
        if edge_pad_width > 0.0 and edge_pad_height > 0.0:
            pad_w = max(1, int(round(edge_pad_width / self.horizontal_scale)))
            pad_h = max(1, int(round(edge_pad_height / self.vertical_scale)))
            hf[:pad_w, :] = np.maximum(hf[:pad_w, :], pad_h)
            hf[-pad_w:, :] = np.maximum(hf[-pad_w:, :], pad_h)
            hf[:, :pad_w] = np.maximum(hf[:, :pad_w], pad_h)
            hf[:, -pad_w:] = np.maximum(hf[:, -pad_w:], pad_h)

        hf = hf.astype(np.int16)
        if return_scene:
            return hf, scene
        return hf
