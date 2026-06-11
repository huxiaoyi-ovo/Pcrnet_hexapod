#!/usr/bin/env python3
"""Check RealSense/YOLO inputs before feeding the PCR high-level policy.

This script keeps the real-camera convention aligned with PCR training:

    goal_buf = (x_right, y_forward)

    The camera is assumed to face the robot +Y direction.  By default this is a
checker; with --publish_ros it also publishes the exact PCR ROS1 inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


SIM_ROBOT_BODY_WIDTH_M = 0.25
SIM_ROBOT_BODY_LENGTH_M = 0.40
SIM_ROBOT_SWING_ABDUCTION_M = 0.15
SIM_FIXED_LAYOUT_ROBOT_CLEARANCE_M = 0.27


def resolve_target_forward_offset_m(args: argparse.Namespace) -> float:
    value = getattr(args, "target_forward_offset_m", None)
    if value is None:
        value = args.camera_forward_offset_m
    return float(value)


def resolve_map_forward_offset_m(args: argparse.Namespace) -> float:
    value = getattr(args, "map_forward_offset_m", None)
    if value is None:
        value = args.camera_forward_offset_m
    return float(value)


def update_obstacle_memory(
    raw_occ: np.ndarray,
    memory_state: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    raw_occ = (np.asarray(raw_occ, dtype=np.float32) > 0.5).astype(np.float32)
    if not bool(args.obstacle_memory) or memory_state is None:
        raw_count = int(np.count_nonzero(raw_occ))
        return raw_occ, raw_occ.copy(), {
            "obstacle_memory_enabled": 0.0,
            "obstacle_memory_dt_s": 0.0,
            "obstacle_memory_decay": 0.0,
            "raw_occ_cell_count": float(raw_count),
            "memory_occ_cell_count": float(raw_count),
            "memory_retained_cell_count": 0.0,
            "memory_retained_ratio": 0.0,
            "memory_peak": float(raw_occ.max()) if raw_occ.size else 0.0,
        }

    now = time.monotonic()
    memory = memory_state.get("occ")
    if not isinstance(memory, np.ndarray) or memory.shape != raw_occ.shape:
        memory = np.zeros_like(raw_occ, dtype=np.float32)
        last_stamp = now
    else:
        last_stamp = float(memory_state.get("stamp", now))

    dt = max(0.0, now - last_stamp)
    tau = max(float(args.obstacle_memory_tau_s), 1e-3)
    decay = math.exp(-dt / tau)
    memory = np.maximum(memory * decay, raw_occ).astype(np.float32, copy=False)
    memory_state["occ"] = memory
    memory_state["stamp"] = now

    threshold = float(np.clip(float(args.obstacle_memory_threshold), 0.0, 1.0))
    fused_occ = (memory >= threshold).astype(np.float32)
    retained = (fused_occ > 0.5) & (raw_occ < 0.5)
    memory_count = int(np.count_nonzero(fused_occ))
    retained_count = int(np.count_nonzero(retained))
    return fused_occ, memory.copy(), {
        "obstacle_memory_enabled": 1.0,
        "obstacle_memory_dt_s": float(dt),
        "obstacle_memory_decay": float(decay),
        "raw_occ_cell_count": float(np.count_nonzero(raw_occ)),
        "memory_occ_cell_count": float(memory_count),
        "memory_retained_cell_count": float(retained_count),
        "memory_retained_ratio": float(retained_count / max(memory_count, 1)),
        "memory_peak": float(memory.max()) if memory.size else 0.0,
    }


@dataclass
class TargetEstimate:
    valid: bool
    bbox_fresh: bool = False
    detect_count: int = 0
    detect_best_conf: float = 0.0
    x_right: float = float("nan")
    y_forward: float = float("nan")
    depth_m: float = float("nan")
    bbox_depth_valid_ratio: float = 0.0
    bbox_depth_median_m: float = float("nan")
    target_u: int = -1
    target_v: int = -1
    target_y_forward_raw_m: float = float("nan")
    target_x_cam: float = float("nan")
    target_y_cam: float = float("nan")
    target_z_cam: float = float("nan")
    v_right: float = 0.0
    v_forward: float = 0.0
    conf: float = 0.0
    age_s: float = float("inf")
    bbox_xyxy: Optional[np.ndarray] = None


class OneEuroFilter:
    def __init__(self, te: float = 0.033, min_cutoff: float = 0.6, beta: float = 0.01):
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.te = float(te)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)

    def filter(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)
        alpha_d = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * 1.0 * self.te))
        dx = (float(x) - self.x_prev) / max(self.te, 1e-6)
        edx = self.dx_prev + alpha_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * self.te))
        x_filtered = self.x_prev + alpha * (float(x) - self.x_prev)
        self.x_prev = x_filtered
        self.dx_prev = edx
        return float(x_filtered)


class TargetTracker:
    def __init__(self, args: argparse.Namespace):
        from ultralytics import YOLO

        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:
            cuda_available = False
        device = "cuda" if cuda_available and not args.cpu_yolo else "cpu"
        self.model = YOLO(args.yolo_model).to(device)
        self.device = device
        self.conf = float(args.yolo_conf)
        self.pitch_down_rad = math.radians(float(args.camera_pitch_down_deg))
        self.camera_height_m = float(args.camera_height_m)
        self.forward_offset_m = resolve_target_forward_offset_m(args)
        self.depth_patch = int(args.target_depth_patch)
        self.target_depth_mode = str(args.target_depth_mode)
        self.target_depth_roi_width_frac = float(args.target_depth_roi_width_frac)
        self.max_lost_frames = int(args.max_lost_frames)
        self.last_bbox: Optional[np.ndarray] = None
        self.last_conf: float = 0.0
        self.last_detect_count: int = 0
        self.last_detect_best_conf: float = 0.0
        self.lost_frames = 0
        self.last_valid_time = 0.0
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_time = time.time()
        self.f_x = OneEuroFilter(beta=0.005)
        self.f_y = OneEuroFilter(beta=0.005)
        self.f_vx = OneEuroFilter(min_cutoff=0.1, beta=0.02)
        self.f_vy = OneEuroFilter(min_cutoff=0.1, beta=0.02)

    @staticmethod
    def _pick_largest_person(result) -> Tuple[Optional[np.ndarray], float]:
        if len(result.boxes) <= 0:
            return None, 0.0
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        return boxes[idx].astype(np.int32), float(confs[idx])

    def detect(self, color_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool]:
        results = self.model.predict(
            color_bgr,
            classes=0,
            conf=self.conf,
            verbose=False,
            device=self.device,
            half=(self.device == "cuda"),
        )
        bbox, conf = (None, 0.0)
        self.last_detect_count = 0
        self.last_detect_best_conf = 0.0
        if results and len(results) > 0:
            self.last_detect_count = int(len(results[0].boxes))
            if self.last_detect_count > 0:
                confs = results[0].boxes.conf.detach().cpu().numpy()
                self.last_detect_best_conf = float(np.max(confs))
            bbox, conf = self._pick_largest_person(results[0])

        fresh = bbox is not None
        if fresh:
            self.last_bbox = bbox
            self.last_conf = conf
            self.lost_frames = 0
        elif self.last_bbox is not None and self.lost_frames < self.max_lost_frames:
            bbox = self.last_bbox
            conf = self.last_conf
            self.lost_frames += 1
        else:
            self.last_bbox = None
            self.last_conf = 0.0
        return bbox, conf, fresh

    def estimate(
        self,
        depth_image_raw: np.ndarray,
        depth_scale: float,
        intrin,
        bbox: Optional[np.ndarray],
        conf: float,
        fresh: bool,
    ) -> TargetEstimate:
        if bbox is None:
            return TargetEstimate(
                valid=False,
                bbox_fresh=False,
                detect_count=self.last_detect_count,
                detect_best_conf=self.last_detect_best_conf,
            )
        h, w = depth_image_raw.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        u = int(np.clip((x1 + x2) * 0.5, 0, w - 1))
        v = int(np.clip(y1 + 0.6 * (y2 - y1), 0, h - 1))
        if str(getattr(self, "target_depth_mode", "roi")).lower() == "point":
            half = max(1, self.depth_patch // 2)
            patch = depth_image_raw[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
        else:
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            roi_w = int(round(float(getattr(self, "target_depth_roi_width_frac", 0.35)) * bw))
            rx1 = int(np.clip(u - 0.5 * roi_w, 0, w - 1))
            rx2 = int(np.clip(u + 0.5 * roi_w, 0, w - 1))
            ry1 = int(np.clip(y1 + 0.35 * bh, 0, h - 1))
            ry2 = int(np.clip(y1 + 0.80 * bh, 0, h - 1))
            patch = depth_image_raw[min(ry1, ry2):max(ry1, ry2) + 1, min(rx1, rx2):max(rx1, rx2) + 1]
        valid_patch = patch > 0
        valid_ratio = float(np.mean(valid_patch.astype(np.float32))) if patch.size > 0 else 0.0
        valid_depths = patch[valid_patch].astype(np.float32) * float(depth_scale)
        if valid_depths.size <= 0:
            return TargetEstimate(
                valid=False,
                bbox_fresh=fresh,
                detect_count=self.last_detect_count,
                detect_best_conf=self.last_detect_best_conf,
                conf=conf,
                bbox_xyxy=bbox,
                bbox_depth_valid_ratio=valid_ratio,
                target_u=u,
                target_v=v,
            )

        dist = float(np.median(valid_depths))
        if not (0.1 < dist < 8.0):
            return TargetEstimate(
                valid=False,
                bbox_fresh=fresh,
                detect_count=self.last_detect_count,
                detect_best_conf=self.last_detect_best_conf,
                conf=conf,
                bbox_xyxy=bbox,
                bbox_depth_valid_ratio=valid_ratio,
                bbox_depth_median_m=dist,
                target_u=u,
                target_v=v,
            )

        pt_c = deproject_pixel(intrin, u, v, dist)
        x_right, y_forward, _height = camera_point_to_robot_frame(
            pt_c,
            pitch_down_rad=self.pitch_down_rad,
            camera_height_m=self.camera_height_m,
            forward_offset_m=self.forward_offset_m,
        )
        y_forward_raw = y_forward - self.forward_offset_m
        x_right_f = self.f_x.filter(x_right)
        y_forward_f = self.f_y.filter(y_forward)

        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        v_right = 0.0
        v_forward = 0.0
        if self.last_pos is not None:
            v_right = self.f_vx.filter((x_right_f - self.last_pos[0]) / dt)
            v_forward = self.f_vy.filter((y_forward_f - self.last_pos[1]) / dt)
        self.last_pos = (x_right_f, y_forward_f)
        self.last_time = now
        self.last_valid_time = now
        return TargetEstimate(
            valid=True,
            bbox_fresh=fresh,
            detect_count=self.last_detect_count,
            detect_best_conf=self.last_detect_best_conf,
            x_right=x_right_f,
            y_forward=y_forward_f,
            depth_m=dist,
            bbox_depth_valid_ratio=valid_ratio,
            bbox_depth_median_m=dist,
            target_u=u,
            target_v=v,
            target_y_forward_raw_m=float(y_forward_raw),
            target_x_cam=float(pt_c[0]),
            target_y_cam=float(pt_c[1]),
            target_z_cam=float(pt_c[2]),
            v_right=v_right,
            v_forward=v_forward,
            conf=conf,
            age_s=0.0,
            bbox_xyxy=bbox,
        )


def deproject_pixel(intrin, u: int, v: int, depth_m: float) -> np.ndarray:
    x = (float(u) - float(intrin.ppx)) / float(intrin.fx) * float(depth_m)
    y = (float(v) - float(intrin.ppy)) / float(intrin.fy) * float(depth_m)
    z = float(depth_m)
    return np.asarray([x, y, z], dtype=np.float32)


def camera_point_to_robot_frame(
    point_camera_xyz: np.ndarray,
    *,
    pitch_down_rad: float,
    camera_height_m: float,
    forward_offset_m: float,
) -> Tuple[float, float, float]:
    """Convert RealSense optical frame to robot local frame.

    RealSense optical frame: +X right, +Y down, +Z forward.
    Robot high-level frame: +X right, +Y forward.
    """
    x_cam = float(point_camera_xyz[0])
    y_cam = float(point_camera_xyz[1])
    z_cam = float(point_camera_xyz[2])
    cos_p = math.cos(pitch_down_rad)
    sin_p = math.sin(pitch_down_rad)
    x_right = x_cam
    # Positive pitch_down means the optical +Z ray points downward in body frame.
    y_forward = z_cam * cos_p - y_cam * sin_p + forward_offset_m
    down_body = y_cam * cos_p + z_cam * sin_p
    height_above_ground = float(camera_height_m) - down_body
    return float(x_right), float(y_forward), float(height_above_ground)


def build_local_map_from_depth(
    depth_raw: np.ndarray,
    depth_scale: float,
    intrin,
    args: argparse.Namespace,
    person_bbox: Optional[np.ndarray],
    person_depth_m: float,
    memory_state: Optional[Dict[str, object]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, float],
]:
    map_size = int(args.map_size)
    map_extent = float(args.map_extent_m)
    cell = map_extent / float(map_size)
    robot_clearance_m = resolve_robot_clearance_m(args)
    map_forward_offset_m = resolve_map_forward_offset_m(args)
    radius_cells = clearance_radius_cells(robot_clearance_m, cell)
    policy_visible = build_policy_visible_mask(args, intrin, depth_raw.shape[1], depth_raw.shape[0]).astype(np.float32)
    occ = np.zeros((map_size, map_size), dtype=np.float32)
    observed = np.zeros((map_size, map_size), dtype=np.float32)
    h, w = depth_raw.shape[:2]
    stride = max(1, int(args.depth_stride))
    target_mask = build_target_depth_mask(depth_raw, depth_scale, person_bbox, person_depth_m, args)

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    depth_m = depth_raw[ys, xs].astype(np.float32) * float(depth_scale)
    valid = (depth_m >= float(args.min_depth_m)) & (depth_m <= float(args.max_depth_m))
    depth_invalid_ratio = float(1.0 - np.mean(valid.astype(np.float32))) if valid.size > 0 else 1.0

    if not bool(args.keep_person_in_map) and target_mask.any():
        valid &= ~target_mask[ys, xs]

    xs_v = xs[valid].astype(np.float32)
    ys_v = ys[valid].astype(np.float32)
    z = depth_m[valid].astype(np.float32)
    if z.size > 0:
        x_cam = (xs_v - float(intrin.ppx)) / float(intrin.fx) * z
        y_cam = (ys_v - float(intrin.ppy)) / float(intrin.fy) * z
        cos_p = math.cos(math.radians(float(args.camera_pitch_down_deg)))
        sin_p = math.sin(math.radians(float(args.camera_pitch_down_deg)))
        x_right = x_cam
        y_forward_raw = z * cos_p - y_cam * sin_p
        y_forward = y_forward_raw + map_forward_offset_m
        down_body = y_cam * cos_p + z * sin_p
        height = float(args.camera_height_m) - down_body

        in_map = (
            (x_right >= -0.5 * map_extent)
            & (x_right < 0.5 * map_extent)
            & (y_forward >= 0.0)
            & (y_forward < map_extent)
        )
        self_mask = (
            (np.abs(x_right) <= float(args.self_mask_half_width_m))
            & (y_forward >= 0.0)
            & (y_forward <= float(args.self_mask_forward_m))
        )
        usable_map = in_map & ~self_mask
        ix_seen = np.floor((x_right[usable_map] + 0.5 * map_extent) / cell).astype(np.int32)
        iy_seen = np.floor(y_forward[usable_map] / cell).astype(np.int32)
        ix_seen = np.clip(ix_seen, 0, map_size - 1)
        iy_seen = np.clip(iy_seen, 0, map_size - 1)
        observed[ix_seen, iy_seen] = 1.0

        obstacle_mode = str(args.obstacle_mode).lower()
        if obstacle_mode == "height_band":
            is_obstacle = (
                (height >= float(args.obstacle_min_height_m))
                & (height <= float(args.obstacle_max_height_m))
                & usable_map
            )
        elif obstacle_mode == "non_person_non_ground":
            is_obstacle = (height > float(args.ground_remove_height_m)) & usable_map
        else:
            raise ValueError(f"Unsupported obstacle_mode: {args.obstacle_mode}")
        ix = np.floor((x_right[is_obstacle] + 0.5 * map_extent) / cell).astype(np.int32)
        iy = np.floor(y_forward[is_obstacle] / cell).astype(np.int32)
        ix = np.clip(ix, 0, map_size - 1)
        iy = np.clip(iy, 0, map_size - 1)
        occ[ix, iy] = 1.0
        front_obstacle = (
            is_obstacle
            & (np.abs(x_right) <= float(args.difficulty_front_half_width_m))
            & (y_forward >= float(args.difficulty_front_min_m))
            & (y_forward <= float(args.difficulty_front_max_m))
        )
        if np.any(front_obstacle):
            front_y = y_forward[front_obstacle]
            front_y_raw = y_forward_raw[front_obstacle]
            front_min_m = float(np.min(front_y))
            front_min_raw_m = float(np.min(front_y_raw))
            front_median_m = float(np.median(front_y))
            front_median_raw_m = float(np.median(front_y_raw))
            front_count = int(np.count_nonzero(front_obstacle))
            front_percentile = float(np.clip(float(args.difficulty_front_percentile), 0.0, 100.0))
            if front_count >= int(args.difficulty_front_min_points):
                front_nearest_m = float(np.percentile(front_y, front_percentile))
                front_nearest_raw_m = float(np.percentile(front_y_raw, front_percentile))
                front_risk_valid = True
            else:
                front_nearest_m = float("inf")
                front_nearest_raw_m = float("inf")
                front_risk_valid = False
        else:
            front_min_m = float("inf")
            front_min_raw_m = float("inf")
            front_nearest_m = float("inf")
            front_nearest_raw_m = float("inf")
            front_median_m = float("nan")
            front_median_raw_m = float("nan")
            front_count = 0
            front_risk_valid = False
        num_in_map = int(np.count_nonzero(in_map))
        num_after_self_mask = int(np.count_nonzero(usable_map))
        num_self_masked_points = int(np.count_nonzero(in_map & self_mask))
        num_obstacle_points = int(np.count_nonzero(is_obstacle))
    else:
        front_min_m = float("inf")
        front_min_raw_m = float("inf")
        front_nearest_m = float("inf")
        front_nearest_raw_m = float("inf")
        front_median_m = float("nan")
        front_median_raw_m = float("nan")
        front_count = 0
        front_risk_valid = False
        num_in_map = 0
        num_after_self_mask = 0
        num_self_masked_points = 0
        num_obstacle_points = 0

    raw_occ = occ.copy()
    if radius_cells > 0:
        kernel = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=np.uint8)
        raw_inflated_occ = cv2.dilate((raw_occ > 0.5).astype(np.uint8), kernel, iterations=1)
    else:
        raw_inflated_occ = (raw_occ > 0.5).astype(np.uint8)
    raw_passable = (
        ((raw_inflated_occ <= 0) & (raw_occ < 0.5)).astype(np.float32) * policy_visible
    )
    raw_local_map_2ch = np.stack([raw_occ, raw_passable], axis=0).astype(np.float32)
    current_difficulty_stats = compute_actor_difficulty_stats(
        raw_local_map_2ch,
        map_extent,
        float(args.difficulty_radius_m),
        visible_mask=policy_visible,
        unknown_cost=float(args.unknown_cost),
    )
    actor_difficulty_current_raw = float(current_difficulty_stats["actor_difficulty"])

    occ, memory_occ, memory_stats = update_obstacle_memory(raw_occ, memory_state, args)
    if radius_cells > 0:
        inflated_occ = cv2.dilate((occ > 0.5).astype(np.uint8), kernel, iterations=1)
    else:
        inflated_occ = (occ > 0.5).astype(np.uint8)
    observed_gate_radius_cells = max(0, int(args.observed_gate_radius_cells))
    if observed_gate_radius_cells > 0:
        observed_kernel = np.ones(
            (2 * observed_gate_radius_cells + 1, 2 * observed_gate_radius_cells + 1),
            dtype=np.uint8,
        )
        observed_gate = cv2.dilate((observed > 0.5).astype(np.uint8), observed_kernel, iterations=1).astype(np.float32)
    else:
        observed_gate = (observed > 0.5).astype(np.float32)
    inflated_occ_f = (inflated_occ > 0).astype(np.float32)
    passable_raw = ((inflated_occ <= 0) & (occ < 0.5)).astype(np.float32)
    # Match sim visible_mask semantics with a dense camera-FOV mask.  Sparse
    # depth samples are obstacle evidence, not the definition of visibility.
    passable = passable_raw * policy_visible
    local_map_2ch = np.stack([occ, passable], axis=0).astype(np.float32)
    risk_blocked_map = np.maximum(occ, inflated_occ_f * policy_visible).astype(np.float32)
    difficulty_stats = compute_actor_difficulty_stats(
        local_map_2ch,
        map_extent,
        float(args.difficulty_radius_m),
        visible_mask=policy_visible,
        unknown_cost=float(args.unknown_cost),
    )
    actor_difficulty_map_fused = float(difficulty_stats["actor_difficulty"])
    front_distance_risk = compute_front_distance_risk(
        front_nearest_m,
        min_m=float(args.difficulty_front_min_m),
        max_m=float(args.difficulty_front_max_m),
    )
    actor_difficulty = float(max(actor_difficulty_map_fused, front_distance_risk))
    visible_count = max(float(np.count_nonzero(policy_visible > 0.5)), 1.0)
    blocked_visible = ((policy_visible > 0.5) & (passable < 0.5)).astype(np.float32)
    front_spread_m = (
        float(front_median_m - front_nearest_m)
        if math.isfinite(float(front_median_m)) and math.isfinite(float(front_nearest_m))
        else float("nan")
    )
    debug_stats = {
        "map_forward_offset_m": map_forward_offset_m,
        "front_nearest_obstacle_m": front_nearest_m,
        "front_nearest_obstacle_raw_m": front_nearest_raw_m,
        "front_min_obstacle_m": front_min_m,
        "front_min_obstacle_raw_m": front_min_raw_m,
        "front_median_obstacle_m": front_median_m,
        "front_median_obstacle_raw_m": front_median_raw_m,
        "front_spread_obstacle_m": front_spread_m,
        "front_obstacle_count": float(front_count),
        "front_risk_valid": float(front_risk_valid),
        "front_risk_percentile": float(args.difficulty_front_percentile),
        "front_risk_min_points": float(args.difficulty_front_min_points),
        "actor_difficulty_current_raw": actor_difficulty_current_raw,
        "actor_difficulty_map_raw": actor_difficulty_current_raw,
        "actor_difficulty_map_fused": actor_difficulty_map_fused,
        "actor_difficulty_map_front": actor_difficulty,
        "front_distance_risk": front_distance_risk,
        "nearest_blocked_m": float(difficulty_stats["nearest_blocked_m"]),
        "near_risk": float(difficulty_stats["near_risk"]),
        "weighted_blocked": float(difficulty_stats["weighted_blocked"]),
        "visible_blocked_ratio": float(difficulty_stats["visible_blocked_ratio"]),
        "visible_safety_blocked_ratio": float(difficulty_stats["visible_safety_blocked_ratio"]),
        "visible_free_ratio": float(difficulty_stats["visible_free_ratio"]),
        "visible_weighted_blocked": float(difficulty_stats["visible_weighted_blocked"]),
        "visible_safety_weighted_blocked": float(difficulty_stats["visible_safety_weighted_blocked"]),
        "unknown_weighted": float(difficulty_stats["unknown_weighted"]),
        "valid_depth_ratio": float(1.0 - depth_invalid_ratio),
        "num_sampled_points": float(depth_m.size),
        "num_valid_depth_points": float(z.size),
        "num_points_in_map": float(num_in_map),
        "num_points_after_self_mask": float(num_after_self_mask),
        "num_self_masked_points": float(num_self_masked_points),
        "num_obstacle_points": float(num_obstacle_points),
        "num_raw_obstacle_cells_before_inflation": float(
            np.count_nonzero(raw_occ > 0.5)
        ),
        "num_fused_obstacle_cells_before_inflation": float(
            np.count_nonzero(occ > 0.5)
        ),
        # Legacy field retained for existing bag analysis scripts.
        "num_obstacle_cells_before_inflation": float(np.count_nonzero(occ > 0.5)),
        "num_obstacle_cells_after_inflation": float(np.count_nonzero(inflated_occ > 0)),
        "policy_visible_mean": float(policy_visible.mean()),
        "policy_visible_blocked_mean": float(blocked_visible.sum() / visible_count),
        "policy_visible_free_mean": float(((policy_visible > 0.5) & (passable > 0.5)).sum() / visible_count),
        "inflated_occ_mean": float(inflated_occ_f.mean()),
        "risk_blocked_mean": float(risk_blocked_map.mean()),
        "risk_blocked_visible_mean": float((risk_blocked_map * policy_visible).sum() / visible_count),
        "raw_observed_mean": float(observed.mean()),
        "raw_observed_gate_mean": float(observed_gate.mean()),
    }
    debug_stats.update(memory_stats)
    return (
        local_map_2ch,
        passable.astype(np.float32),
        actor_difficulty,
        target_mask,
        depth_invalid_ratio,
        observed,
        observed_gate,
        policy_visible,
        risk_blocked_map,
        raw_occ,
        memory_occ,
        debug_stats,
    )


def compute_near_field_risk(
    depth_raw: np.ndarray,
    depth_scale: float,
    intrin,
    args: argparse.Namespace,
) -> Dict[str, float]:
    h, w = depth_raw.shape[:2]
    x_frac = float(np.clip(args.near_field_roi_width_frac, 0.05, 1.0))
    y_frac = float(np.clip(args.near_field_roi_height_frac, 0.05, 1.0))
    x1 = int(round(0.5 * (1.0 - x_frac) * w))
    x2 = int(round(0.5 * (1.0 + x_frac) * w))
    y1 = int(round((1.0 - y_frac) * h))
    y2 = h
    roi_raw = depth_raw[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    roi = roi_raw.astype(np.float32) * float(depth_scale)
    valid = (roi >= float(args.min_depth_m)) & (roi <= float(args.near_field_warn_m))
    valid_ratio = float(np.mean(valid.astype(np.float32))) if roi.size > 0 else 0.0
    if np.any(valid):
        map_forward_offset_m = resolve_map_forward_offset_m(args)
        yy, xx = np.mgrid[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        xx_v = xx[valid].astype(np.float32)
        yy_v = yy[valid].astype(np.float32)
        z = roi[valid].astype(np.float32)
        x_cam = (xx_v - float(intrin.ppx)) / float(intrin.fx) * z
        y_cam = (yy_v - float(intrin.ppy)) / float(intrin.fy) * z
        cos_p = math.cos(math.radians(float(args.camera_pitch_down_deg)))
        sin_p = math.sin(math.radians(float(args.camera_pitch_down_deg)))
        x_right = x_cam
        y_forward = z * cos_p - y_cam * sin_p + map_forward_offset_m
        down_body = y_cam * cos_p + z * sin_p
        height = float(args.camera_height_m) - down_body
        near_obstacle = (
            (np.abs(x_right) <= float(args.near_field_half_width_m))
            & (y_forward >= float(args.near_field_min_forward_m))
            & (y_forward <= float(args.near_field_max_forward_m))
            & (height >= float(args.ground_remove_height_m))
            & (height <= float(args.obstacle_max_height_m))
        )
        candidate_count = max(float(z.size), 1.0)
        obstacle_ratio = float(np.count_nonzero(near_obstacle) / candidate_count)
        if np.any(near_obstacle):
            obstacle_y = y_forward[near_obstacle]
            close = obstacle_y <= float(args.near_field_stop_m)
            warning = obstacle_y <= float(args.near_field_warn_m)
            close_ratio = float(np.count_nonzero(close) / candidate_count)
            warning_ratio = float(np.count_nonzero(warning) / candidate_count)
            min_depth = float(np.min(obstacle_y))
            median_depth = float(np.median(obstacle_y))
        else:
            close_ratio = 0.0
            warning_ratio = 0.0
            min_depth = float("nan")
            median_depth = float("nan")
    else:
        obstacle_ratio = 0.0
        close_ratio = 0.0
        warning_ratio = 0.0
        min_depth = float("nan")
        median_depth = float("nan")

    enough_close = close_ratio >= float(args.near_field_close_ratio)
    enough_warning = warning_ratio >= float(args.near_field_warn_ratio)
    if enough_close:
        dist_risk = 1.0
    elif enough_warning:
        dist_risk = float(np.clip(
            (float(args.near_field_warn_m) - min_depth)
            / max(float(args.near_field_warn_m) - float(args.near_field_stop_m), 1e-6),
            0.0,
            1.0,
        ))
    else:
        dist_risk = 0.0
    density_risk = float(np.clip(
        obstacle_ratio / max(float(args.near_field_warn_ratio), 1e-6),
        0.0,
        1.0,
    ))
    risk = float(np.clip(dist_risk * density_risk, 0.0, 1.0))

    return {
        "near_field_risk": float(risk),
        "near_field_valid_ratio": valid_ratio,
        "near_field_obstacle_ratio": obstacle_ratio,
        "near_field_close_ratio": close_ratio,
        "near_field_warning_ratio": warning_ratio,
        "near_field_min_depth_m": min_depth,
        "near_field_median_depth_m": median_depth,
    }


def compute_front_distance_risk(nearest_m: float, *, min_m: float, max_m: float) -> float:
    if not math.isfinite(float(nearest_m)):
        return 0.0
    lo = float(min_m)
    hi = max(float(max_m), lo + 1e-6)
    return float(np.clip((hi - float(nearest_m)) / (hi - lo), 0.0, 1.0))


def build_policy_visible_mask(args: argparse.Namespace, intrin, image_width: int, image_height: int) -> np.ndarray:
    map_size = int(args.map_size)
    map_extent = float(args.map_extent_m)
    cell = map_extent / float(map_size)
    x_centers = np.linspace(-0.5 * map_extent + 0.5 * cell, 0.5 * map_extent - 0.5 * cell, map_size)
    y_centers = np.linspace(0.5 * cell, map_extent - 0.5 * cell, map_size)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")

    pitch = math.radians(float(args.camera_pitch_down_deg))
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    y_forward_from_camera = grid_y - resolve_map_forward_offset_m(args)
    down_body = float(args.camera_height_m)
    z_cam = cos_p * y_forward_from_camera + sin_p * down_body
    y_cam = -sin_p * y_forward_from_camera + cos_p * down_body
    x_cam = grid_x

    u = x_cam / np.maximum(z_cam, 1e-6) * float(intrin.fx) + float(intrin.ppx)
    v = y_cam / np.maximum(z_cam, 1e-6) * float(intrin.fy) + float(intrin.ppy)
    visible = (
        (z_cam >= float(args.min_depth_m))
        & (z_cam <= float(args.max_depth_m))
        & (u >= 0.0)
        & (u < float(image_width))
        & (v >= 0.0)
        & (v < float(image_height))
    )
    return visible.astype(np.float32)


def resolve_robot_clearance_m(args: argparse.Namespace) -> float:
    override = float(args.robot_clearance_m)
    if override > 0.0:
        return override
    body_half_width = 0.5 * float(args.robot_body_width_m)
    swing_margin = float(args.robot_swing_abduction_m)
    depth_margin = float(args.robot_depth_noise_margin_m)
    extra_margin = float(args.robot_extra_safety_margin_m)
    return max(0.0, body_half_width + swing_margin + depth_margin + extra_margin)


def clearance_radius_cells(clearance_m: float, cell_m: float) -> int:
    return int(math.ceil(float(clearance_m) / max(float(cell_m), 1e-6)))


def build_target_depth_mask(
    depth_raw: np.ndarray,
    depth_scale: float,
    person_bbox: Optional[np.ndarray],
    person_depth_m: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Mask only the tracked person's depth layer, not the whole bbox."""
    mask = np.zeros(depth_raw.shape[:2], dtype=np.bool_)
    if person_bbox is None or bool(args.keep_person_in_map):
        return mask
    if not math.isfinite(float(person_depth_m)):
        return mask

    h, w = depth_raw.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person_bbox]
    margin = int(args.person_mask_margin_px)
    x1 = int(np.clip(x1 - margin, 0, w - 1))
    x2 = int(np.clip(x2 + margin, 0, w - 1))
    y1 = int(np.clip(y1 - margin, 0, h - 1))
    y2 = int(np.clip(y2 + margin, 0, h - 1))
    if x2 < x1 or y2 < y1:
        return mask

    patch_depth = depth_raw[y1:y2 + 1, x1:x2 + 1].astype(np.float32) * float(depth_scale)
    valid_depth = patch_depth > 0.0
    close_to_target = np.abs(patch_depth - float(person_depth_m)) <= float(args.target_mask_depth_margin_m)
    mask[y1:y2 + 1, x1:x2 + 1] = valid_depth & close_to_target
    return mask


def compute_actor_difficulty_stats(
    local_map_2ch: np.ndarray,
    map_extent: float,
    radius_m: float,
    *,
    visible_mask: Optional[np.ndarray] = None,
    unknown_cost: float = 0.25,
) -> Dict[str, float]:
    occ = np.clip(local_map_2ch[0], 0.0, 1.0)
    safety = np.clip(local_map_2ch[1], 0.0, 1.0)
    n = occ.shape[0]
    cell = map_extent / float(n)
    x_centers = np.linspace(-0.5 * map_extent + 0.5 * cell, 0.5 * map_extent - 0.5 * cell, n)
    y_centers = np.linspace(0.5 * cell, map_extent - 0.5 * cell, n)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    dist = np.sqrt(grid_x ** 2 + grid_y ** 2)
    radius = max(float(radius_m), 1e-3)
    radial_mask = (dist <= radius).astype(np.float32)
    if visible_mask is None:
        visible = np.ones_like(occ, dtype=np.float32)
    else:
        visible = (np.asarray(visible_mask, dtype=np.float32) > 0.5).astype(np.float32)
        if visible.shape != occ.shape:
            raise ValueError(f"visible_mask shape must be {occ.shape}, got {visible.shape}")
    unknown = 1.0 - visible
    visible_occ = occ * visible
    visible_safety_blocked = np.maximum(occ, 1.0 - safety) * visible
    unknown_risk = unknown * float(np.clip(unknown_cost, 0.0, 1.0))
    blocked = np.maximum(visible_occ, unknown_risk)
    visible_blocked_in_radius = visible_occ * radial_mask
    has_blocked = bool(np.any(visible_blocked_in_radius > 0.5))
    if has_blocked:
        nearest_blocked_m = float(np.min(dist[visible_blocked_in_radius > 0.5]))
        near_risk = float(np.clip((radius - nearest_blocked_m) / radius, 0.0, 1.0))
    else:
        nearest_blocked_m = float("inf")
        near_risk = 0.0
    distance_weight = np.clip((radius - dist) / radius, 0.0, 1.0) ** 2
    weighted_denom = max(float((distance_weight * radial_mask).sum()), 1.0)
    weighted_blocked = float((blocked * distance_weight * radial_mask).sum() / weighted_denom)
    visible_weighted_denom = max(float((distance_weight * radial_mask * visible).sum()), 1.0)
    visible_weighted_blocked = float(
        (visible_occ * distance_weight * radial_mask).sum() / visible_weighted_denom
    )
    visible_safety_weighted_blocked = float(
        (visible_safety_blocked * distance_weight * radial_mask).sum() / visible_weighted_denom
    )
    visible_radial_count = max(float((visible * radial_mask).sum()), 1.0)
    visible_blocked_ratio = float((visible_occ * radial_mask).sum() / visible_radial_count)
    visible_safety_blocked_ratio = float((visible_safety_blocked * radial_mask).sum() / visible_radial_count)
    visible_free_ratio = float(((visible > 0.5) & (safety > 0.5) & (radial_mask > 0.5)).sum() / visible_radial_count)
    unknown_weighted = float((unknown_risk * distance_weight * radial_mask).sum() / weighted_denom)
    # Difficulty follows true obstacle evidence. The inflated safety map remains
    # a policy/safety input, but it should not make far obstacles look critical.
    difficulty = float(np.clip(
        0.55 * visible_weighted_blocked + 0.30 * visible_blocked_ratio + 0.15 * near_risk,
        0.0,
        1.0,
    ))
    return {
        "actor_difficulty": difficulty,
        "nearest_blocked_m": nearest_blocked_m,
        "near_risk": near_risk,
        "weighted_blocked": weighted_blocked,
        "visible_blocked_ratio": visible_blocked_ratio,
        "visible_safety_blocked_ratio": visible_safety_blocked_ratio,
        "visible_free_ratio": visible_free_ratio,
        "visible_weighted_blocked": visible_weighted_blocked,
        "visible_safety_weighted_blocked": visible_safety_weighted_blocked,
        "unknown_weighted": unknown_weighted,
        "unknown_cost": float(np.clip(unknown_cost, 0.0, 1.0)),
    }


def compute_actor_difficulty(
    local_map_2ch: np.ndarray,
    map_extent: float,
    radius_m: float,
    *,
    visible_mask: Optional[np.ndarray] = None,
    unknown_cost: float = 0.25,
) -> float:
    return float(
        compute_actor_difficulty_stats(
            local_map_2ch,
            map_extent,
            radius_m,
            visible_mask=visible_mask,
            unknown_cost=unknown_cost,
        )["actor_difficulty"]
    )


def target_distance_m(target: TargetEstimate) -> float:
    if not target.valid:
        return float("inf")
    return float(math.hypot(float(target.x_right), float(target.y_forward)))


def make_policy_ready_obs(
    target: TargetEstimate,
    local_map_2ch: np.ndarray,
    actor_difficulty: float,
    *,
    risk_blocked_map: Optional[np.ndarray] = None,
    policy_visible_map: Optional[np.ndarray] = None,
    front_distance_risk: float = 0.0,
    depth_invalid_ratio: float,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    use_follow = bool(target.valid)
    if target.valid and math.isfinite(float(target.x_right)) and math.isfinite(float(target.y_forward)):
        goal = np.asarray([[target.x_right, target.y_forward]], dtype=np.float32)
    else:
        goal = np.zeros((1, 2), dtype=np.float32)
    target_distance = target_distance_m(target)
    target_too_close = bool(target.valid and target_distance < float(args.target_min_distance_m))
    depth_invalid = bool(depth_invalid_ratio > float(args.depth_invalid_ratio_stop))
    return {
        "goal": goal,
        "follow_goal": goal.copy(),
        "use_follow_goal": np.asarray([use_follow], dtype=np.bool_),
        "local_map_2ch": local_map_2ch[None, :, :, :].astype(np.float32),
        "risk_blocked_map": (
            np.asarray(risk_blocked_map, dtype=np.float32)[None, :, :]
            if risk_blocked_map is not None
            else np.zeros((1, local_map_2ch.shape[-2], local_map_2ch.shape[-1]), dtype=np.float32)
        ),
        "policy_visible_map": (
            np.asarray(policy_visible_map, dtype=np.float32)[None, :, :]
            if policy_visible_map is not None
            else np.ones((1, local_map_2ch.shape[-2], local_map_2ch.shape[-1]), dtype=np.float32)
        ),
        "front_distance_risk": np.asarray([front_distance_risk], dtype=np.float32),
        "actor_difficulty": np.asarray([actor_difficulty], dtype=np.float32),
        "target_valid": np.asarray([target.valid], dtype=np.bool_),
        "target_lost": np.asarray([not target.valid], dtype=np.bool_),
        "target_too_close": np.asarray([target_too_close], dtype=np.bool_),
        "target_distance_m": np.asarray([target_distance], dtype=np.float32),
        "target_vel": np.asarray([[target.v_right, target.v_forward]], dtype=np.float32),
        "depth_invalid_ratio": np.asarray([depth_invalid_ratio], dtype=np.float32),
        "depth_invalid": np.asarray([depth_invalid], dtype=np.bool_),
    }


def draw_debug(
    color_bgr: np.ndarray,
    target: TargetEstimate,
    local_map_2ch: np.ndarray,
    actor_difficulty: float,
    target_mask: np.ndarray,
    observed_map: np.ndarray,
    depth_invalid_ratio: float,
    args: argparse.Namespace,
) -> np.ndarray:
    rgb_panel = color_bgr.copy()
    if target.bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox_xyxy]
        color = (0, 255, 0) if target.valid else (0, 200, 255)
        cv2.rectangle(rgb_panel, (x1, y1), (x2, y2), color, 2)
    text = (
        f"valid={int(target.valid)} x_right={target.x_right:.2f} "
        f"y_fwd={target.y_forward:.2f} v=({target.v_right:.2f},{target.v_forward:.2f})"
    )
    cv2.putText(rgb_panel, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(rgb_panel, f"difficulty={actor_difficulty:.3f}", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    target_status = (
        f"yoloN={target.detect_count} best={target.detect_best_conf:.2f} "
        f"bboxConf={target.conf:.2f} fresh={int(target.bbox_fresh)} "
        f"depOK={target.bbox_depth_valid_ratio:.2f} depMed={target.bbox_depth_median_m:.2f}"
    )
    cv2.putText(rgb_panel, target_status, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cell_m = float(args.map_extent_m) / float(max(int(args.map_size), 1))
    robot_clearance_m = resolve_robot_clearance_m(args)
    robot_clearance_cells = clearance_radius_cells(robot_clearance_m, cell_m)
    status = (
        f"d={target_distance_m(target):.2f} tooClose={int(target.valid and target_distance_m(target) < float(args.target_min_distance_m))} "
        f"lost={int(not target.valid)} depthBad={depth_invalid_ratio:.2f} "
        f"clr={robot_clearance_m:.2f}m/{robot_clearance_cells}c mode={args.obstacle_mode}"
    )
    cv2.putText(rgb_panel, status, (12, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    occ = (np.clip(local_map_2ch[0], 0.0, 1.0) * 255.0).astype(np.uint8)
    safety = (np.clip(local_map_2ch[1], 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_vis = (target_mask.astype(np.uint8) * 255)
    map_panel_px = max(160, int(args.debug_map_px))
    observed = (np.clip(observed_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    occ_img = cv2.resize(np.flipud(occ.T), (map_panel_px, map_panel_px), interpolation=cv2.INTER_NEAREST)
    clr_img = cv2.resize(np.flipud(safety.T), (map_panel_px, map_panel_px), interpolation=cv2.INTER_NEAREST)
    obs_img = cv2.resize(np.flipud(observed.T), (map_panel_px, map_panel_px), interpolation=cv2.INTER_NEAREST)
    occ_color = np.zeros((map_panel_px, map_panel_px, 3), dtype=np.uint8)
    free_cells = (obs_img > 0) & (clr_img > 127)
    blocked_cells = (obs_img > 0) & (occ_img > 127)
    occ_color[free_cells] = (255, 255, 255)
    occ_color[blocked_cells] = (0, 0, 255)
    clr_color = cv2.cvtColor(clr_img, cv2.COLOR_GRAY2BGR)
    robot_px = (map_panel_px // 2, map_panel_px - 8)
    cv2.circle(occ_color, robot_px, 5, (255, 255, 0), -1)
    cv2.circle(clr_color, robot_px, 5, (0, 255, 255), -1)
    mask_small = cv2.resize(mask_vis, (map_panel_px, map_panel_px), interpolation=cv2.INTER_NEAREST)
    mask_color = cv2.applyColorMap(mask_small, cv2.COLORMAP_OCEAN)

    gap = 12
    label_h = 28
    side_w = 2 * map_panel_px + gap
    side_h = 2 * (map_panel_px + label_h) + gap
    rgb_h, rgb_w = rgb_panel.shape[:2]
    canvas_h = max(rgb_h, side_h)
    canvas_w = rgb_w + gap + side_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:rgb_h, :rgb_w] = rgb_panel

    def paste_panel(img: np.ndarray, x: int, y: int, label: str) -> None:
        canvas[y:y + map_panel_px, x:x + map_panel_px] = img
        cv2.putText(
            canvas,
            label,
            (x + 6, y + map_panel_px + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    x0 = rgb_w + gap
    y0 = 0
    paste_panel(occ_color, x0, y0, "free/obstacle map")
    paste_panel(clr_color, x0 + map_panel_px + gap, y0, "safety map")
    paste_panel(mask_color, x0, y0 + map_panel_px + label_h + gap, "target mask")

    legend_x = x0 + map_panel_px + gap
    legend_y = y0 + map_panel_px + label_h + gap + 30
    cv2.putText(canvas, "policy input", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "map: white=free", (legend_x, legend_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "map: red=obstacle", (legend_x, legend_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "map: black=unknown/blocked", (legend_x, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "safety: white=passable", (legend_x, legend_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "view: bottom=robot, top=forward", (legend_x, legend_y + 146), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas


def prepare_display_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = max(float(args.display_scale), 0.1)
    if abs(scale - 1.0) < 1e-6:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def save_snapshot(
    save_dir: str,
    obs: Dict[str, np.ndarray],
    color_bgr: np.ndarray,
    target_mask: np.ndarray,
    frame_idx: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_dir, f"real_pcr_input_{frame_idx:06d}.npz"),
        **obs,
        target_mask=target_mask.astype(np.uint8),
    )
    cv2.imwrite(os.path.join(save_dir, f"real_pcr_color_{frame_idx:06d}.png"), color_bgr)


class ViewerRecorder:
    """Write draw_debug() frames without blocking the camera/control loop."""

    def __init__(self, rospy, Bool, String, args: argparse.Namespace):
        self.rospy = rospy
        self.args = args
        self.lock = threading.RLock()
        self.frame_queue_limit = max(1, int(args.viewer_record_queue_size))
        # Reserve two slots for stop and shutdown so frame backpressure can
        # never prevent the writer from finalizing the current MP4.
        self.frame_queue = queue.Queue(maxsize=self.frame_queue_limit + 2)
        self.session_dir = ""
        self.active_session = ""
        self.recording_requested = False
        self.accept_frames = False
        self.last_enqueue = 0.0
        self.dropped_frames = 0
        self.session_drop_start = 0
        self.shutdown_started = False
        self.worker = threading.Thread(
            target=self._writer_loop,
            name="pcr_viewer_writer",
            daemon=True,
        )
        self.worker.start()
        self.session_sub = rospy.Subscriber(
            "/pcr/recording_session", String, self._session_callback, queue_size=1
        )
        self.recording_sub = rospy.Subscriber(
            "/pcr/recording", Bool, self._recording_callback, queue_size=1
        )
        rospy.on_shutdown(self.shutdown)

    def _session_callback(self, msg) -> None:
        with self.lock:
            self.session_dir = str(msg.data).strip()
            if self.recording_requested and self.session_dir:
                self._activate_locked(self.session_dir)

    def _recording_callback(self, msg) -> None:
        requested = bool(msg.data)
        with self.lock:
            self.recording_requested = requested
            if requested:
                if self.session_dir:
                    self._activate_locked(self.session_dir)
                else:
                    self.rospy.logwarn(
                        "[ViewerRecorder] recording requested before session path; waiting"
                    )
                return
            old_session = self.active_session
            self.accept_frames = False
            self.active_session = ""
        if old_session:
            self._enqueue_control(("stop", old_session))

    def _activate_locked(self, session_dir: str) -> None:
        if self.accept_frames and self.active_session == session_dir:
            return
        self.active_session = session_dir
        self.accept_frames = True
        self.last_enqueue = 0.0
        self.session_drop_start = self.dropped_frames
        self.rospy.logwarn("[ViewerRecorder] viewer recording armed: %s", session_dir)

    def _enqueue_control(self, item) -> None:
        self.frame_queue.put(item)

    def is_recording(self) -> bool:
        with self.lock:
            return bool(self.accept_frames and self.active_session)

    def enqueue(
        self,
        source_frame_idx: int,
        ros_time: float,
        wall_time: float,
        frame_bgr: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        with self.lock:
            if not self.accept_frames or not self.active_session:
                return
            now_mono = time.monotonic()
            period = 1.0 / max(float(self.args.viewer_video_fps), 1.0)
            if now_mono - self.last_enqueue < period:
                return
            self.last_enqueue = now_mono
            session_dir = self.active_session
            session_drop_start = self.session_drop_start
        packet = (
            "frame",
            session_dir,
            session_drop_start,
            int(source_frame_idx),
            float(ros_time),
            float(wall_time),
            dict(metrics),
            frame_bgr.copy(),
        )
        if self.frame_queue.qsize() >= self.frame_queue_limit:
            with self.lock:
                self.dropped_frames += 1
            self.rospy.logwarn_throttle(
                2.0,
                "[ViewerRecorder] viewer queue full; dropping frames to protect control timing",
            )
            return
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            with self.lock:
                self.dropped_frames += 1
            self.rospy.logwarn_throttle(
                2.0,
                "[ViewerRecorder] viewer queue full; dropping frames to protect control timing",
            )

    def _open_writer(self, video_path: Path, width: int, height: int):
        fps = max(float(self.args.viewer_video_fps), 1.0)
        bitrate = max(int(self.args.viewer_video_bitrate), 100000)
        writer = None
        codec = ""
        gst_inspect = shutil.which("gst-inspect-1.0")
        if gst_inspect is not None:
            try:
                probe = subprocess.run(
                    [gst_inspect, "nvv4l2h264enc"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                probe = None
            if probe is not None and probe.returncode == 0:
                pipeline = (
                    "appsrc ! videoconvert ! video/x-raw,format=BGRx ! "
                    "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                    f"nvv4l2h264enc bitrate={bitrate} ! h264parse ! qtmux ! "
                    f'filesink location="{video_path}" sync=false'
                )
                candidate = cv2.VideoWriter(
                    pipeline,
                    cv2.CAP_GSTREAMER,
                    0,
                    fps,
                    (width, height),
                    True,
                )
                if candidate.isOpened():
                    writer = candidate
                    codec = "nvv4l2h264enc"
                else:
                    candidate.release()
        if writer is None:
            candidate = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                codec = "mp4v"
            else:
                candidate.release()
        return writer, codec

    def _viewer_metadata(
        self,
        session_dir: Path,
        *,
        codec: str,
        width: int,
        height: int,
        frame_count: int,
        dropped_frames: int,
        started_ros: Optional[float],
        started_wall: Optional[float],
        stopped_ros: Optional[float],
        stopped_wall: Optional[float],
    ) -> None:
        video_path = session_dir / "viewer.mp4"
        frame_csv_path = session_dir / "frame_timestamps.csv"
        video_available = bool(frame_count > 0 and video_path.is_file())
        timestamps_available = bool(frame_count > 0 and frame_csv_path.is_file())
        payload = {
            "viewer_video": str(video_path) if video_available else None,
            "frame_timestamps": (
                str(frame_csv_path) if timestamps_available else None
            ),
            "viewer_recording_ok": bool(video_available and timestamps_available),
            "viewer_video_fps": float(self.args.viewer_video_fps),
            "viewer_video_bitrate": int(self.args.viewer_video_bitrate),
            "viewer_codec": codec,
            "viewer_width": int(width),
            "viewer_height": int(height),
            "viewer_frames_written": int(frame_count),
            "viewer_frames_dropped": int(dropped_frames),
            "viewer_started_ros_time": started_ros,
            "viewer_started_wall_time": started_wall,
            "viewer_stopped_ros_time": stopped_ros,
            "viewer_stopped_wall_time": stopped_wall,
            "camera_width": int(self.args.width),
            "camera_height": int(self.args.height),
            "camera_fps_target": int(self.args.fps),
            "camera_height_m": float(self.args.camera_height_m),
            "camera_pitch_down_deg": float(self.args.camera_pitch_down_deg),
            "target_forward_offset_m": float(resolve_target_forward_offset_m(self.args)),
            "map_forward_offset_m": float(resolve_map_forward_offset_m(self.args)),
            "map_size": int(self.args.map_size),
            "map_extent_m": float(self.args.map_extent_m),
            "obstacle_memory_enable": bool(self.args.obstacle_memory),
            "obstacle_memory_tau_s": float(self.args.obstacle_memory_tau_s),
            "obstacle_memory_threshold": float(self.args.obstacle_memory_threshold),
            "yolo_model": str(self.args.yolo_model),
            "yolo_conf": float(self.args.yolo_conf),
            "pcr_policy": str(self.args.viewer_pcr_policy),
            "lowlevel_policy": str(self.args.viewer_lowlevel_policy),
        }
        tmp_path = session_dir / "viewer_session.json.tmp"
        final_path = session_dir / "viewer_session.json"
        try:
            with open(tmp_path, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(str(tmp_path), str(final_path))
        except OSError as exc:
            self.rospy.logerr("[ViewerRecorder] failed to write metadata: %s", exc)

    def _writer_loop(self) -> None:
        writer = None
        csv_file = None
        csv_writer = None
        writer_session = ""
        codec = ""
        width = 0
        height = 0
        frame_count = 0
        started_ros = None
        started_wall = None
        writer_drop_start = 0

        def close_session() -> None:
            nonlocal writer, csv_file, csv_writer, writer_session
            nonlocal codec, width, height, frame_count, started_ros, started_wall
            nonlocal writer_drop_start
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.flush()
                csv_file.close()
            if writer_session:
                with self.lock:
                    dropped = self.dropped_frames - writer_drop_start
                self._viewer_metadata(
                    Path(writer_session),
                    codec=codec,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    dropped_frames=max(0, dropped),
                    started_ros=started_ros,
                    started_wall=started_wall,
                    stopped_ros=float(self.rospy.Time.now().to_sec()),
                    stopped_wall=time.time(),
                )
                self.rospy.logwarn(
                    "[ViewerRecorder] viewer closed: %s frames=%d dropped=%d codec=%s",
                    writer_session,
                    frame_count,
                    max(0, dropped),
                    codec or "none",
                )
            writer = None
            csv_file = None
            csv_writer = None
            writer_session = ""
            codec = ""
            width = 0
            height = 0
            frame_count = 0
            started_ros = None
            started_wall = None
            writer_drop_start = 0

        while True:
            item = self.frame_queue.get()
            try:
                kind = item[0]
                if kind == "shutdown":
                    close_session()
                    return
                if kind == "stop":
                    if writer_session == item[1]:
                        close_session()
                    elif not writer_session:
                        self._viewer_metadata(
                            Path(item[1]),
                            codec="",
                            width=0,
                            height=0,
                            frame_count=0,
                            dropped_frames=0,
                            started_ros=None,
                            started_wall=None,
                            stopped_ros=float(self.rospy.Time.now().to_sec()),
                            stopped_wall=time.time(),
                        )
                    continue

                (
                    _,
                    session_dir,
                    session_drop_start,
                    source_idx,
                    ros_time,
                    wall_time,
                    metrics,
                    frame,
                ) = item
                if writer_session != session_dir:
                    close_session()
                    session_path = Path(session_dir)
                    session_path.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer, codec = self._open_writer(
                        session_path / "viewer.mp4", width, height
                    )
                    if writer is None:
                        self.rospy.logerr(
                            "[ViewerRecorder] no video encoder available; viewer recording disabled"
                        )
                        writer_session = session_dir
                        continue
                    csv_file = open(
                        session_path / "frame_timestamps.csv", "w", newline=""
                    )
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(
                        [
                            "frame_idx",
                            "source_frame_idx",
                            "ros_time",
                            "wall_time",
                            "video_time",
                            "target_valid",
                            "actor_difficulty",
                            "front_distance_risk",
                            "raw_occ_cell_count",
                            "memory_cell_count",
                        ]
                    )
                    writer_session = session_dir
                    writer_drop_start = int(session_drop_start)
                    started_ros = float(ros_time)
                    started_wall = float(wall_time)

                if writer is None or csv_writer is None:
                    continue
                writer.write(frame)
                video_time = float(frame_count) / max(
                    float(self.args.viewer_video_fps), 1.0
                )
                csv_writer.writerow(
                    [
                        frame_count,
                        source_idx,
                        f"{ros_time:.9f}",
                        f"{wall_time:.9f}",
                        f"{video_time:.9f}",
                        int(bool(metrics["target_valid"])),
                        f"{float(metrics['actor_difficulty']):.6f}",
                        f"{float(metrics['front_distance_risk']):.6f}",
                        int(metrics["raw_occ_cell_count"]),
                        int(metrics["memory_cell_count"]),
                    ]
                )
                frame_count += 1
                if frame_count % 10 == 0:
                    csv_file.flush()
            except Exception as exc:
                self.rospy.logerr("[ViewerRecorder] writer error: %s", exc)
                close_session()
            finally:
                self.frame_queue.task_done()

    def shutdown(self) -> None:
        with self.lock:
            if self.shutdown_started:
                return
            self.shutdown_started = True
            self.accept_frames = False
            old_session = self.active_session
            self.active_session = ""
        if old_session:
            self._enqueue_control(("stop", old_session))
        self._enqueue_control(("shutdown",))
        self.worker.join(timeout=10.0)
        if self.worker.is_alive():
            self.rospy.logerr("[ViewerRecorder] writer thread did not stop cleanly")


def setup_ros_publishers(args: argparse.Namespace):
    if not bool(args.publish_ros):
        return None
    try:
        import rospy
        from std_msgs.msg import Bool
        from std_msgs.msg import Float32
        from std_msgs.msg import Float32MultiArray
        from std_msgs.msg import String
    except ImportError as exc:
        raise SystemExit(
            "--publish_ros requires ROS1 Python packages; source the catkin workspace first."
        ) from exc
    rospy.init_node(args.ros_node_name, anonymous=False)
    target_pub = rospy.Publisher(args.target_topic, Float32MultiArray, queue_size=1)
    local_map_pub = rospy.Publisher(args.local_map_topic, Float32MultiArray, queue_size=1)
    risk_blocked_pub = rospy.Publisher(args.risk_blocked_topic, Float32MultiArray, queue_size=1)
    policy_visible_pub = rospy.Publisher(args.policy_visible_topic, Float32MultiArray, queue_size=1)
    raw_occ_pub = rospy.Publisher(args.raw_occ_topic, Float32MultiArray, queue_size=1)
    memory_occ_pub = rospy.Publisher(args.memory_occ_topic, Float32MultiArray, queue_size=1)
    front_distance_risk_pub = rospy.Publisher(args.front_distance_risk_topic, Float32, queue_size=1)
    viewer_recorder = ViewerRecorder(rospy, Bool, String, args)
    print(
        "[RealPCR] ROS publishers ready: "
        f"target={args.target_topic}, local_map={args.local_map_topic}, "
        f"risk_blocked={args.risk_blocked_topic}, policy_visible={args.policy_visible_topic}, "
        f"raw_occ={args.raw_occ_topic}, memory_occ={args.memory_occ_topic}, "
        f"front_distance_risk={args.front_distance_risk_topic}",
        flush=True,
    )
    return (
        rospy,
        Float32,
        Float32MultiArray,
        target_pub,
        local_map_pub,
        risk_blocked_pub,
        policy_visible_pub,
        raw_occ_pub,
        memory_occ_pub,
        front_distance_risk_pub,
        viewer_recorder,
    )


def publish_policy_obs(ros_ctx, obs: Dict[str, np.ndarray]) -> None:
    if ros_ctx is None:
        return
    (
        _rospy,
        Float32,
        Float32MultiArray,
        target_pub,
        local_map_pub,
        risk_blocked_pub,
        policy_visible_pub,
        raw_occ_pub,
        memory_occ_pub,
        front_distance_risk_pub,
        _viewer_recorder,
    ) = ros_ctx
    goal = np.asarray(obs["goal"], dtype=np.float32).reshape(1, 2)[0]
    target_vel = np.asarray(obs["target_vel"], dtype=np.float32).reshape(1, 2)[0]
    target_valid = float(bool(np.asarray(obs["target_valid"]).reshape(-1)[0]))
    target_too_close = float(bool(np.asarray(obs["target_too_close"]).reshape(-1)[0]))
    depth_invalid = float(bool(np.asarray(obs["depth_invalid"]).reshape(-1)[0]))
    actor_difficulty = float(np.asarray(obs["actor_difficulty"], dtype=np.float32).reshape(-1)[0])
    front_distance_risk = float(np.asarray(obs["front_distance_risk"], dtype=np.float32).reshape(-1)[0])

    target_msg = Float32MultiArray()
    target_msg.data = [
        float(goal[0]),
        float(goal[1]),
        float(target_vel[0]),
        float(target_vel[1]),
        target_valid,
        target_too_close,
        depth_invalid,
        actor_difficulty,
        front_distance_risk,
    ]
    local_msg = Float32MultiArray()
    local_msg.data = np.asarray(obs["local_map_2ch"], dtype=np.float32).reshape(-1).tolist()
    risk_msg = Float32MultiArray()
    risk_msg.data = np.asarray(obs["risk_blocked_map"], dtype=np.float32).reshape(-1).tolist()
    visible_msg = Float32MultiArray()
    visible_msg.data = np.asarray(obs["policy_visible_map"], dtype=np.float32).reshape(-1).tolist()
    raw_occ_msg = Float32MultiArray()
    raw_occ_msg.data = np.asarray(obs["raw_occ_map"], dtype=np.float32).reshape(-1).tolist()
    memory_occ_msg = Float32MultiArray()
    memory_occ_msg.data = np.asarray(obs["memory_occ_map"], dtype=np.float32).reshape(-1).tolist()
    front_msg = Float32()
    front_msg.data = front_distance_risk
    target_pub.publish(target_msg)
    local_map_pub.publish(local_msg)
    risk_blocked_pub.publish(risk_msg)
    policy_visible_pub.publish(visible_msg)
    raw_occ_pub.publish(raw_occ_msg)
    memory_occ_pub.publish(memory_occ_msg)
    front_distance_risk_pub.publish(front_msg)

def write_policy_obs_file(path: str, obs: Dict[str, np.ndarray]) -> None:
    if not path:
        return
    goal = np.asarray(obs["goal"], dtype=np.float32).reshape(1, 2)[0]
    target_vel = np.asarray(obs["target_vel"], dtype=np.float32).reshape(1, 2)[0]
    payload = {
        "stamp": time.time(),
        "target_state": [
            float(goal[0]),
            float(goal[1]),
            float(target_vel[0]),
            float(target_vel[1]),
            float(bool(np.asarray(obs["target_valid"]).reshape(-1)[0])),
            float(bool(np.asarray(obs["target_too_close"]).reshape(-1)[0])),
            float(bool(np.asarray(obs["depth_invalid"]).reshape(-1)[0])),
            float(np.asarray(obs["actor_difficulty"], dtype=np.float32).reshape(-1)[0]),
        ],
        "local_map_shape": list(np.asarray(obs["local_map_2ch"], dtype=np.float32).shape),
        "local_map_2ch": np.asarray(obs["local_map_2ch"], dtype=np.float32).reshape(-1).tolist(),
        "risk_blocked_shape": list(np.asarray(obs["risk_blocked_map"], dtype=np.float32).shape),
        "risk_blocked_map": np.asarray(obs["risk_blocked_map"], dtype=np.float32).reshape(-1).tolist(),
        "policy_visible_shape": list(np.asarray(obs["policy_visible_map"], dtype=np.float32).shape),
        "policy_visible_map": np.asarray(obs["policy_visible_map"], dtype=np.float32).reshape(-1).tolist(),
        "raw_occ_shape": list(np.asarray(obs["raw_occ_map"], dtype=np.float32).shape),
        "raw_occ_map": np.asarray(obs["raw_occ_map"], dtype=np.float32).reshape(-1).tolist(),
        "memory_occ_shape": list(np.asarray(obs["memory_occ_map"], dtype=np.float32).shape),
        "memory_occ_map": np.asarray(obs["memory_occ_map"], dtype=np.float32).reshape(-1).tolist(),
        "front_distance_risk": float(np.asarray(obs["front_distance_risk"], dtype=np.float32).reshape(-1)[0]),
    }
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".obs_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense/YOLO PCR input checker.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.35)
    parser.add_argument("--cpu_yolo", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera_pitch_down_deg", type=float, default=10.0)
    parser.add_argument("--camera_forward_offset_m", type=float, default=0.30)
    parser.add_argument("--target_forward_offset_m", type=float, default=None)
    parser.add_argument("--map_forward_offset_m", type=float, default=None)
    parser.add_argument("--camera_height_m", type=float, default=0.45)
    parser.add_argument("--target_depth_mode", type=str, default="roi", choices=["roi", "point"])
    parser.add_argument("--target_depth_roi_width_frac", type=float, default=0.35)
    parser.add_argument("--target_depth_patch", type=int, default=10)
    parser.add_argument("--max_lost_frames", type=int, default=5)
    parser.add_argument("--map_size", type=int, default=32)
    parser.add_argument("--map_extent_m", type=float, default=3.0)
    parser.add_argument("--depth_stride", type=int, default=4)
    parser.add_argument("--observed_gate_radius_cells", type=int, default=2)
    parser.add_argument("--min_depth_m", type=float, default=0.25)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--obstacle_min_height_m", type=float, default=0.08)
    parser.add_argument("--obstacle_max_height_m", type=float, default=0.80)
    parser.add_argument("--obstacle_mode", type=str, default="height_band", choices=["non_person_non_ground", "height_band"])
    parser.add_argument("--ground_remove_height_m", type=float, default=0.04)
    parser.add_argument("--self_mask_forward_m", type=float, default=0.25)
    parser.add_argument("--self_mask_half_width_m", type=float, default=0.45)
    parser.add_argument("--robot_clearance_m", type=float, default=SIM_FIXED_LAYOUT_ROBOT_CLEARANCE_M, help="effective inflation radius; <=0 uses robot geometry")
    parser.add_argument("--robot_body_width_m", type=float, default=SIM_ROBOT_BODY_WIDTH_M)
    parser.add_argument("--robot_body_length_m", type=float, default=SIM_ROBOT_BODY_LENGTH_M)
    parser.add_argument("--robot_swing_abduction_m", type=float, default=SIM_ROBOT_SWING_ABDUCTION_M)
    parser.add_argument("--robot_depth_noise_margin_m", type=float, default=0.03)
    parser.add_argument("--robot_extra_safety_margin_m", type=float, default=0.02)
    parser.add_argument("--unknown_cost", type=float, default=0.25)
    parser.add_argument("--clearance_free_m", type=float, default=0.57)
    parser.add_argument("--difficulty_radius_m", type=float, default=2.0)
    parser.add_argument("--difficulty_front_min_m", type=float, default=0.05)
    parser.add_argument("--difficulty_front_max_m", type=float, default=2.0)
    parser.add_argument("--difficulty_front_half_width_m", type=float, default=0.55)
    parser.add_argument("--difficulty_front_min_points", type=int, default=15)
    parser.add_argument("--difficulty_front_percentile", type=float, default=10.0)
    parser.add_argument("--obstacle_memory", dest="obstacle_memory", action="store_true")
    parser.add_argument("--no_obstacle_memory", dest="obstacle_memory", action="store_false")
    parser.set_defaults(obstacle_memory=True)
    parser.add_argument("--obstacle_memory_tau_s", type=float, default=0.80)
    parser.add_argument("--obstacle_memory_threshold", type=float, default=0.35)
    parser.add_argument("--near_field_stop_m", type=float, default=0.35)
    parser.add_argument("--near_field_warn_m", type=float, default=0.50)
    parser.add_argument("--near_field_min_forward_m", type=float, default=0.05)
    parser.add_argument("--near_field_max_forward_m", type=float, default=0.70)
    parser.add_argument("--near_field_half_width_m", type=float, default=0.45)
    parser.add_argument("--near_field_roi_width_frac", type=float, default=0.45)
    parser.add_argument("--near_field_roi_height_frac", type=float, default=0.35)
    parser.add_argument("--near_field_close_ratio", type=float, default=0.04)
    parser.add_argument("--near_field_warn_ratio", type=float, default=0.12)
    parser.add_argument("--keep_person_in_map", action="store_true")
    parser.add_argument("--person_mask_margin_px", type=int, default=12)
    parser.add_argument("--target_mask_depth_margin_m", type=float, default=0.25)
    parser.add_argument("--target_min_distance_m", type=float, default=0.80)
    parser.add_argument("--depth_invalid_ratio_stop", type=float, default=0.60)
    parser.add_argument("--print_hz", type=float, default=5.0)
    parser.add_argument("--publish_ros", action="store_true", help="publish /pcr target and local-map topics for pcr_realplay.py")
    parser.add_argument("--obs_file", type=str, default="", help="write latest PCR observation JSON for non-ROS laptop checks")
    parser.add_argument("--ros_node_name", type=str, default="real_pcr_input")
    parser.add_argument("--target_topic", type=str, default="/pcr/target_state")
    parser.add_argument("--local_map_topic", type=str, default="/pcr/local_map_2ch")
    parser.add_argument("--risk_blocked_topic", type=str, default="/pcr/risk_blocked_map")
    parser.add_argument("--policy_visible_topic", type=str, default="/pcr/policy_visible_map")
    parser.add_argument("--raw_occ_topic", type=str, default="/pcr/raw_occ_map")
    parser.add_argument("--memory_occ_topic", type=str, default="/pcr/memory_occ_map")
    parser.add_argument("--front_distance_risk_topic", type=str, default="/pcr/front_distance_risk")
    parser.add_argument("--viewer_video_fps", type=float, default=15.0)
    parser.add_argument("--viewer_video_bitrate", type=int, default=4000000)
    parser.add_argument("--viewer_record_queue_size", type=int, default=8)
    parser.add_argument("--viewer_pcr_policy", type=str, default="")
    parser.add_argument("--viewer_lowlevel_policy", type=str, default="")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--display_scale", type=float, default=1.6)
    parser.add_argument("--debug_map_px", type=int, default=320)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is required for this script.") from exc

    tracker = TargetTracker(args)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    hole_filling = rs.hole_filling_filter()

    frame_idx = 0
    last_print = 0.0
    obstacle_memory_state: Dict[str, object] = {}
    window_name = "Real PCR input check"
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    ros_ctx = setup_ros_publishers(args)
    viewer_recorder = None if ros_ctx is None else ros_ctx[-1]
    print(
        "[RealPCR] started. Policy frame: goal_buf=(x_right,y_forward), "
        f"camera forward is robot +Y. publish_ros={bool(args.publish_ros)}"
    )
    cell_m = float(args.map_extent_m) / float(max(int(args.map_size), 1))
    robot_clearance_m = resolve_robot_clearance_m(args)
    robot_clearance_cells = clearance_radius_cells(robot_clearance_m, cell_m)
    print(
        "[RealPCR] safety geometry: "
        f"body_width={float(args.robot_body_width_m):.3f}m "
        f"body_length={float(args.robot_body_length_m):.3f}m "
        f"swing_abduction={float(args.robot_swing_abduction_m):.3f}m "
        f"depth_margin={float(args.robot_depth_noise_margin_m):.3f}m "
        f"extra_margin={float(args.robot_extra_safety_margin_m):.3f}m "
        f"effective_clearance={robot_clearance_m:.3f}m "
        f"cell={cell_m:.4f}m inflation_cells={robot_clearance_cells} "
        f"visible=aligned_camera_intrinsics "
        f"self_mask_y<={float(args.self_mask_forward_m):.2f}m "
        f"self_mask_half_width={float(args.self_mask_half_width_m):.2f}m "
        f"unknown_cost={float(args.unknown_cost):.2f} "
        f"obstacle_memory={bool(args.obstacle_memory)} "
        f"memory_tau={float(args.obstacle_memory_tau_s):.2f}s "
        f"memory_threshold={float(args.obstacle_memory_threshold):.2f}"
    )
    try:
        while True:
            frames = pipe.wait_for_frames(1000)
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f:
                continue
            depth_f = hole_filling.process(depth_f).as_depth_frame()
            color = np.asanyarray(color_f.get_data())
            depth_raw = np.asanyarray(depth_f.get_data())
            near_field_stats = compute_near_field_risk(depth_raw, depth_scale, intrin, args)

            bbox, conf, fresh = tracker.detect(color)
            target = tracker.estimate(depth_raw, depth_scale, intrin, bbox, conf, fresh)
            (
                local_map_2ch,
                _clearance_m,
                actor_difficulty,
                target_mask,
                depth_invalid_ratio,
                observed_map,
                observed_gate_map,
                policy_visible_map,
                risk_blocked_map,
                raw_occ_map,
                memory_occ_map,
                map_debug,
            ) = build_local_map_from_depth(
                depth_raw,
                depth_scale,
                intrin,
                args,
                target.bbox_xyxy,
                target.depth_m,
                memory_state=obstacle_memory_state,
            )
            actor_difficulty_map_front = float(actor_difficulty)
            actor_difficulty = float(max(actor_difficulty_map_front, float(near_field_stats["near_field_risk"])))
            obs = make_policy_ready_obs(
                target,
                local_map_2ch,
                actor_difficulty,
                risk_blocked_map=risk_blocked_map,
                policy_visible_map=policy_visible_map,
                front_distance_risk=float(map_debug["front_distance_risk"]),
                depth_invalid_ratio=depth_invalid_ratio,
                args=args,
            )
            obs["raw_occ_map"] = raw_occ_map[None, :, :].astype(np.float32)
            obs["memory_occ_map"] = memory_occ_map[None, :, :].astype(np.float32)
            publish_policy_obs(ros_ctx, obs)
            write_policy_obs_file(args.obs_file, obs)

            now = time.time()
            if now - last_print >= 1.0 / max(float(args.print_hz), 1e-6):
                target_distance = target_distance_m(target)
                target_too_close = bool(target.valid and target_distance < float(args.target_min_distance_m))
                depth_invalid = bool(depth_invalid_ratio > float(args.depth_invalid_ratio_stop))
                bbox_list = None if bbox is None else [int(v) for v in bbox.tolist()]
                msg = {
                    "target_valid": bool(target.valid),
                    "target_lost": bool(not target.valid),
                    "target_too_close": bool(target_too_close),
                    "target_bbox_exists": bool(bbox is not None),
                    "target_bbox_fresh": bool(fresh),
                    "target_bbox_cached": bool((bbox is not None) and (not fresh)),
                    "yolo_person_count": int(target.detect_count),
                    "yolo_best_conf": float(target.detect_best_conf),
                    "target_bbox_conf": float(conf),
                    "target_bbox_xyxy": bbox_list,
                    "target_bbox_depth_valid_ratio": float(target.bbox_depth_valid_ratio),
                    "target_bbox_depth_median_m": float(target.bbox_depth_median_m),
                    "target_pixel_uv": [int(target.target_u), int(target.target_v)],
                    "target_y_forward_raw_m": float(target.target_y_forward_raw_m),
                    "target_forward_offset_m": float(resolve_target_forward_offset_m(args)),
                    "map_forward_offset_m": float(resolve_map_forward_offset_m(args)),
                    "target_camera_xyz_m": [
                        float(target.target_x_cam),
                        float(target.target_y_cam),
                        float(target.target_z_cam),
                    ],
                    "map_cell_m": float(cell_m),
                    "robot_clearance_m": float(robot_clearance_m),
                    "robot_clearance_cells": int(robot_clearance_cells),
                    "robot_body_width_m": float(args.robot_body_width_m),
                    "robot_body_length_m": float(args.robot_body_length_m),
                    "robot_swing_abduction_m": float(args.robot_swing_abduction_m),
                    "robot_depth_noise_margin_m": float(args.robot_depth_noise_margin_m),
                    "robot_extra_safety_margin_m": float(args.robot_extra_safety_margin_m),
                    "self_mask_forward_m": float(args.self_mask_forward_m),
                    "self_mask_half_width_m": float(args.self_mask_half_width_m),
                    "goal_buf": [float(target.x_right), float(target.y_forward)],
                    "target_distance_m": float(target_distance),
                    "target_depth_m": float(target.depth_m),
                    "target_vel": [float(target.v_right), float(target.v_forward)],
                    "actor_difficulty": float(actor_difficulty),
                    "actor_difficulty_final": float(actor_difficulty),
                    "actor_difficulty_map_front": float(actor_difficulty_map_front),
                    "actor_difficulty_current_raw": float(map_debug["actor_difficulty_current_raw"]),
                    "actor_difficulty_map_raw": float(map_debug["actor_difficulty_map_raw"]),
                    "actor_difficulty_map_fused": float(map_debug["actor_difficulty_map_fused"]),
                    "front_distance_risk": float(map_debug["front_distance_risk"]),
                    "front_nearest_obstacle_m": float(map_debug["front_nearest_obstacle_m"]),
                    "front_nearest_obstacle_raw_m": float(map_debug["front_nearest_obstacle_raw_m"]),
                    "front_min_obstacle_m": float(map_debug["front_min_obstacle_m"]),
                    "front_min_obstacle_raw_m": float(map_debug["front_min_obstacle_raw_m"]),
                    "front_median_obstacle_m": float(map_debug["front_median_obstacle_m"]),
                    "front_median_obstacle_raw_m": float(map_debug["front_median_obstacle_raw_m"]),
                    "front_spread_obstacle_m": float(map_debug["front_spread_obstacle_m"]),
                    "front_obstacle_count": int(map_debug["front_obstacle_count"]),
                    "front_risk_valid": bool(map_debug["front_risk_valid"]),
                    "front_risk_percentile": float(map_debug["front_risk_percentile"]),
                    "front_risk_min_points": int(map_debug["front_risk_min_points"]),
                    "near_field_risk": float(near_field_stats["near_field_risk"]),
                    "near_field_valid_ratio": float(near_field_stats["near_field_valid_ratio"]),
                    "near_field_obstacle_ratio": float(near_field_stats["near_field_obstacle_ratio"]),
                    "near_field_close_ratio": float(near_field_stats["near_field_close_ratio"]),
                    "near_field_warning_ratio": float(near_field_stats["near_field_warning_ratio"]),
                    "near_field_min_depth_m": float(near_field_stats["near_field_min_depth_m"]),
                    "near_field_median_depth_m": float(near_field_stats["near_field_median_depth_m"]),
                    "nearest_blocked_m": float(map_debug["nearest_blocked_m"]),
                    "near_risk": float(map_debug["near_risk"]),
                    "weighted_blocked": float(map_debug["weighted_blocked"]),
                    "visible_blocked_ratio": float(map_debug["visible_blocked_ratio"]),
                    "visible_safety_blocked_ratio": float(map_debug["visible_safety_blocked_ratio"]),
                    "visible_free_ratio": float(map_debug["visible_free_ratio"]),
                    "visible_weighted_blocked": float(map_debug["visible_weighted_blocked"]),
                    "visible_safety_weighted_blocked": float(map_debug["visible_safety_weighted_blocked"]),
                    "unknown_weighted": float(map_debug["unknown_weighted"]),
                    "local_map_shape": list(obs["local_map_2ch"].shape),
                    "occ_mean": float(local_map_2ch[0].mean()),
                    "safety_mean": float(local_map_2ch[1].mean()),
                    "observed_mean": float(observed_map.mean()),
                    "observed_gate_mean": float(observed_gate_map.mean()),
                    "policy_visible_mean": float(policy_visible_map.mean()),
                    "policy_unknown_mean": float((policy_visible_map < 0.5).mean()),
                    "observed_blocked_mean": float(((observed_gate_map > 0.5) & (local_map_2ch[1] < 0.5)).mean()),
                    "policy_unknown_blocked_mean": float(((policy_visible_map < 0.5) & (local_map_2ch[1] < 0.5)).mean()),
                    "policy_visible_blocked_mean": float(map_debug["policy_visible_blocked_mean"]),
                    "policy_visible_free_mean": float(map_debug["policy_visible_free_mean"]),
                    "inflated_occ_mean": float(map_debug["inflated_occ_mean"]),
                    "risk_blocked_mean": float(map_debug["risk_blocked_mean"]),
                    "risk_blocked_visible_mean": float(map_debug["risk_blocked_visible_mean"]),
                    "inflated_blocked_mean": float((local_map_2ch[1] < 0.5).mean()),
                    "num_sampled_points": int(map_debug["num_sampled_points"]),
                    "num_valid_depth_points": int(map_debug["num_valid_depth_points"]),
                    "num_points_in_map": int(map_debug["num_points_in_map"]),
                    "num_points_after_self_mask": int(map_debug["num_points_after_self_mask"]),
                    "num_self_masked_points": int(map_debug["num_self_masked_points"]),
                    "num_obstacle_points": int(map_debug["num_obstacle_points"]),
                    "num_raw_obstacle_cells_before_inflation": int(
                        map_debug["num_raw_obstacle_cells_before_inflation"]
                    ),
                    "num_fused_obstacle_cells_before_inflation": int(
                        map_debug["num_fused_obstacle_cells_before_inflation"]
                    ),
                    "num_obstacle_cells_before_inflation": int(map_debug["num_obstacle_cells_before_inflation"]),
                    "num_obstacle_cells_after_inflation": int(map_debug["num_obstacle_cells_after_inflation"]),
                    "obstacle_memory_enabled": bool(map_debug["obstacle_memory_enabled"]),
                    "obstacle_memory_dt_s": float(map_debug["obstacle_memory_dt_s"]),
                    "obstacle_memory_decay": float(map_debug["obstacle_memory_decay"]),
                    "raw_occ_cell_count": int(map_debug["raw_occ_cell_count"]),
                    "memory_occ_cell_count": int(map_debug["memory_occ_cell_count"]),
                    "memory_retained_cell_count": int(map_debug["memory_retained_cell_count"]),
                    "memory_retained_ratio": float(map_debug["memory_retained_ratio"]),
                    "memory_peak": float(map_debug["memory_peak"]),
                    "target_mask_ratio": float(target_mask.mean()),
                    "depth_invalid_ratio": float(depth_invalid_ratio),
                    "depth_invalid": bool(depth_invalid),
                    "ros1_future_fields": [
                        "goal_buf",
                        "target_vel",
                        "target_valid",
                        "target_lost",
                        "target_too_close",
                        "target_bbox_exists",
                        "target_bbox_fresh",
                        "target_bbox_cached",
                        "yolo_person_count",
                        "yolo_best_conf",
                        "target_bbox_depth_valid_ratio",
                        "target_bbox_depth_median_m",
                        "target_distance_m",
                        "local_map_2ch",
                        "actor_difficulty",
                        "actor_difficulty_final",
                        "actor_difficulty_map_front",
                        "actor_difficulty_current_raw",
                        "actor_difficulty_map_raw",
                        "actor_difficulty_map_fused",
                        "front_distance_risk",
                        "front_nearest_obstacle_m",
                        "front_median_obstacle_m",
                        "front_obstacle_count",
                        "near_field_risk",
                        "near_field_valid_ratio",
                        "near_field_obstacle_ratio",
                        "near_field_close_ratio",
                        "near_field_warning_ratio",
                        "near_field_min_depth_m",
                        "near_field_median_depth_m",
                        "nearest_blocked_m",
                        "near_risk",
                        "weighted_blocked",
                        "visible_blocked_ratio",
                        "visible_safety_blocked_ratio",
                        "visible_free_ratio",
                        "visible_weighted_blocked",
                        "visible_safety_weighted_blocked",
                        "unknown_weighted",
                        "policy_visible_mean",
                        "policy_visible_blocked_mean",
                        "policy_visible_free_mean",
                        "inflated_occ_mean",
                        "risk_blocked_mean",
                        "risk_blocked_visible_mean",
                        "obstacle_memory_enabled",
                        "obstacle_memory_dt_s",
                        "obstacle_memory_decay",
                        "num_raw_obstacle_cells_before_inflation",
                        "num_fused_obstacle_cells_before_inflation",
                        "raw_occ_cell_count",
                        "memory_occ_cell_count",
                        "memory_retained_cell_count",
                        "memory_retained_ratio",
                        "depth_invalid_ratio",
                        "depth_invalid",
                    ],
                }
                print(json.dumps(msg, ensure_ascii=False))
                last_print = now

            if args.save_dir and args.save_every > 0 and frame_idx % int(args.save_every) == 0:
                save_snapshot(args.save_dir, obs, color, target_mask, frame_idx)

            if args.show or (
                viewer_recorder is not None and viewer_recorder.is_recording()
            ):
                debug = draw_debug(
                    color,
                    target,
                    local_map_2ch,
                    actor_difficulty,
                    target_mask,
                    policy_visible_map,
                    depth_invalid_ratio,
                    args,
                )
                if viewer_recorder is not None:
                    viewer_recorder.enqueue(
                        source_frame_idx=frame_idx,
                        ros_time=float(ros_ctx[0].Time.now().to_sec()),
                        wall_time=now,
                        frame_bgr=debug,
                        metrics={
                            "target_valid": float(target.valid),
                            "actor_difficulty": float(actor_difficulty),
                            "front_distance_risk": float(
                                map_debug["front_distance_risk"]
                            ),
                            "raw_occ_cell_count": float(
                                map_debug["raw_occ_cell_count"]
                            ),
                            "memory_cell_count": float(
                                map_debug["memory_occ_cell_count"]
                            ),
                        },
                    )

            if args.show:
                cv2.imshow(window_name, prepare_display_image(debug, args))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                try:
                    window_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    window_visible = 0.0
                if window_visible < 1.0:
                    args.show = False
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
            frame_idx += 1
    finally:
        if viewer_recorder is not None:
            viewer_recorder.shutdown()
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
