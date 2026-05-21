#!/usr/bin/env python3
"""Check RealSense/YOLO inputs before feeding the PCR high-level policy.

This script keeps the real-camera convention aligned with PCR training:

    goal_buf = (x_right, y_forward)

The camera is assumed to face the robot +Y direction.  The script does not
publish ROS messages yet; it prints and optionally saves the exact fields that
should later be carried by ROS1 topics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TargetEstimate:
    valid: bool
    x_right: float = float("nan")
    y_forward: float = float("nan")
    depth_m: float = float("nan")
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
        self.forward_offset_m = float(args.camera_forward_offset_m)
        self.depth_patch = int(args.target_depth_patch)
        self.max_lost_frames = int(args.max_lost_frames)
        self.last_bbox: Optional[np.ndarray] = None
        self.last_conf: float = 0.0
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
        if results and len(results) > 0:
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
    ) -> TargetEstimate:
        if bbox is None:
            return TargetEstimate(valid=False)
        h, w = depth_image_raw.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        u = int(np.clip((x1 + x2) * 0.5, 0, w - 1))
        v = int(np.clip(y2 - 10, 0, h - 1))
        half = max(1, self.depth_patch // 2)
        patch = depth_image_raw[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
        valid_depths = patch[patch > 0].astype(np.float32) * float(depth_scale)
        if valid_depths.size <= 0:
            return TargetEstimate(valid=False, conf=conf, bbox_xyxy=bbox)

        dist = float(np.median(valid_depths))
        if not (0.1 < dist < 8.0):
            return TargetEstimate(valid=False, conf=conf, bbox_xyxy=bbox)

        pt_c = deproject_pixel(intrin, u, v, dist)
        x_right, y_forward, _height = camera_point_to_robot_frame(
            pt_c,
            pitch_down_rad=self.pitch_down_rad,
            camera_height_m=self.camera_height_m,
            forward_offset_m=self.forward_offset_m,
        )
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
            x_right=x_right_f,
            y_forward=y_forward_f,
            depth_m=dist,
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
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    map_size = int(args.map_size)
    map_extent = float(args.map_extent_m)
    cell = map_extent / float(map_size)
    occ = np.zeros((map_size, map_size), dtype=np.float32)
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
        y_forward = z * cos_p - y_cam * sin_p + float(args.camera_forward_offset_m)
        down_body = y_cam * cos_p + z * sin_p
        height = float(args.camera_height_m) - down_body

        is_obstacle = (
            (height >= float(args.obstacle_min_height_m))
            & (height <= float(args.obstacle_max_height_m))
            & (x_right >= -0.5 * map_extent)
            & (x_right < 0.5 * map_extent)
            & (y_forward >= 0.0)
            & (y_forward < map_extent)
        )
        ix = np.floor((x_right[is_obstacle] + 0.5 * map_extent) / cell).astype(np.int32)
        iy = np.floor(y_forward[is_obstacle] / cell).astype(np.int32)
        ix = np.clip(ix, 0, map_size - 1)
        iy = np.clip(iy, 0, map_size - 1)
        occ[ix, iy] = 1.0

    free_mask = (occ < 0.5).astype(np.uint8)
    dist_cells = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3).astype(np.float32)
    clearance_m = dist_cells * cell
    clearance = np.clip(clearance_m / max(float(args.clearance_free_m), 1e-6), 0.0, 1.0)
    local_map_2ch = np.stack([occ, clearance], axis=0).astype(np.float32)
    actor_difficulty = compute_actor_difficulty(local_map_2ch, map_extent, float(args.difficulty_radius_m))
    return local_map_2ch, clearance_m.astype(np.float32), actor_difficulty, target_mask, depth_invalid_ratio


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


def compute_actor_difficulty(local_map_2ch: np.ndarray, map_extent: float, radius_m: float) -> float:
    occ = np.clip(local_map_2ch[0], 0.0, 1.0)
    clearance = np.clip(local_map_2ch[1], 0.0, 1.0)
    n = occ.shape[0]
    cell = map_extent / float(n)
    x_centers = np.linspace(-0.5 * map_extent + 0.5 * cell, 0.5 * map_extent - 0.5 * cell, n)
    y_centers = np.linspace(0.5 * cell, map_extent - 0.5 * cell, n)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    radial_mask = ((grid_x ** 2 + grid_y ** 2) <= max(radius_m, 1e-3) ** 2).astype(np.float32)
    denom = max(float(radial_mask.sum()), 1.0)
    occ_ratio = float((occ * radial_mask).sum() / denom)
    clearance_cost = float(((1.0 - clearance) * radial_mask).sum() / denom)
    return float(np.clip(0.5 * occ_ratio + 0.5 * clearance_cost, 0.0, 1.0))


def target_distance_m(target: TargetEstimate) -> float:
    if not target.valid:
        return float("inf")
    return float(math.hypot(float(target.x_right), float(target.y_forward)))


def make_policy_ready_obs(
    target: TargetEstimate,
    local_map_2ch: np.ndarray,
    actor_difficulty: float,
    *,
    depth_invalid_ratio: float,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
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
        "use_follow_goal": np.asarray(True),
        "local_map_2ch": local_map_2ch[None, :, :, :].astype(np.float32),
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
    depth_invalid_ratio: float,
    args: argparse.Namespace,
) -> np.ndarray:
    canvas = color_bgr.copy()
    if target.bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox_xyxy]
        color = (0, 255, 0) if target.valid else (0, 200, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    text = (
        f"valid={int(target.valid)} x_right={target.x_right:.2f} "
        f"y_fwd={target.y_forward:.2f} v=({target.v_right:.2f},{target.v_forward:.2f})"
    )
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(canvas, f"difficulty={actor_difficulty:.3f}", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    status = (
        f"d={target_distance_m(target):.2f} tooClose={int(target.valid and target_distance_m(target) < float(args.target_min_distance_m))} "
        f"lost={int(not target.valid)} depthBad={depth_invalid_ratio:.2f}"
    )
    cv2.putText(canvas, status, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    occ = (np.clip(local_map_2ch[0], 0.0, 1.0) * 255.0).astype(np.uint8)
    clearance = (np.clip(local_map_2ch[1], 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_vis = (target_mask.astype(np.uint8) * 255)
    occ_img = cv2.resize(occ.T, (180, 180), interpolation=cv2.INTER_NEAREST)
    clr_img = cv2.resize(clearance.T, (180, 180), interpolation=cv2.INTER_NEAREST)
    occ_color = cv2.applyColorMap(occ_img, cv2.COLORMAP_HOT)
    clr_color = cv2.applyColorMap(clr_img, cv2.COLORMAP_VIRIDIS)
    mask_small = cv2.resize(mask_vis, (180, 180), interpolation=cv2.INTER_NEAREST)
    mask_color = cv2.applyColorMap(mask_small, cv2.COLORMAP_OCEAN)
    h, w = canvas.shape[:2]
    if h >= 280 and w >= 390:
        canvas[100:280, 10:190] = occ_color
        canvas[100:280, 200:380] = clr_color
        cv2.putText(canvas, "occ", (14, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, "clear", (204, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if h >= 470 and w >= 190:
        canvas[290:470, 10:190] = mask_color
        cv2.putText(canvas, "target mask", (14, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense/YOLO PCR input checker.")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.6)
    parser.add_argument("--cpu_yolo", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera_pitch_down_deg", type=float, default=10.0)
    parser.add_argument("--camera_forward_offset_m", type=float, default=0.30)
    parser.add_argument("--camera_height_m", type=float, default=0.45)
    parser.add_argument("--target_depth_patch", type=int, default=10)
    parser.add_argument("--max_lost_frames", type=int, default=5)
    parser.add_argument("--map_size", type=int, default=16)
    parser.add_argument("--map_extent_m", type=float, default=3.0)
    parser.add_argument("--depth_stride", type=int, default=4)
    parser.add_argument("--min_depth_m", type=float, default=0.25)
    parser.add_argument("--max_depth_m", type=float, default=3.0)
    parser.add_argument("--obstacle_min_height_m", type=float, default=0.06)
    parser.add_argument("--obstacle_max_height_m", type=float, default=1.20)
    parser.add_argument("--clearance_free_m", type=float, default=0.57)
    parser.add_argument("--difficulty_radius_m", type=float, default=2.0)
    parser.add_argument("--keep_person_in_map", action="store_true")
    parser.add_argument("--person_mask_margin_px", type=int, default=12)
    parser.add_argument("--target_mask_depth_margin_m", type=float, default=0.25)
    parser.add_argument("--target_min_distance_m", type=float, default=0.80)
    parser.add_argument("--depth_invalid_ratio_stop", type=float, default=0.60)
    parser.add_argument("--print_hz", type=float, default=5.0)
    parser.add_argument("--show", action="store_true")
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
    print(
        "[RealPCR] started. Policy frame: goal_buf=(x_right,y_forward), "
        "camera forward is robot +Y. ROS1 is not enabled in this checker."
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

            bbox, conf, _fresh = tracker.detect(color)
            target = tracker.estimate(depth_raw, depth_scale, intrin, bbox, conf)
            local_map_2ch, _clearance_m, actor_difficulty, target_mask, depth_invalid_ratio = build_local_map_from_depth(
                depth_raw,
                depth_scale,
                intrin,
                args,
                target.bbox_xyxy,
                target.depth_m,
            )
            obs = make_policy_ready_obs(
                target,
                local_map_2ch,
                actor_difficulty,
                depth_invalid_ratio=depth_invalid_ratio,
                args=args,
            )

            now = time.time()
            if now - last_print >= 1.0 / max(float(args.print_hz), 1e-6):
                target_distance = target_distance_m(target)
                target_too_close = bool(target.valid and target_distance < float(args.target_min_distance_m))
                depth_invalid = bool(depth_invalid_ratio > float(args.depth_invalid_ratio_stop))
                msg = {
                    "target_valid": bool(target.valid),
                    "target_lost": bool(not target.valid),
                    "target_too_close": bool(target_too_close),
                    "goal_buf": [float(target.x_right), float(target.y_forward)],
                    "target_distance_m": float(target_distance),
                    "target_depth_m": float(target.depth_m),
                    "target_vel": [float(target.v_right), float(target.v_forward)],
                    "actor_difficulty": float(actor_difficulty),
                    "local_map_shape": list(obs["local_map_2ch"].shape),
                    "occ_mean": float(local_map_2ch[0].mean()),
                    "clearance_mean": float(local_map_2ch[1].mean()),
                    "target_mask_ratio": float(target_mask.mean()),
                    "depth_invalid_ratio": float(depth_invalid_ratio),
                    "depth_invalid": bool(depth_invalid),
                    "ros1_future_fields": [
                        "goal_buf",
                        "target_vel",
                        "target_valid",
                        "target_lost",
                        "target_too_close",
                        "target_distance_m",
                        "local_map_2ch",
                        "actor_difficulty",
                        "depth_invalid_ratio",
                        "depth_invalid",
                    ],
                }
                print(json.dumps(msg, ensure_ascii=False))
                last_print = now

            if args.save_dir and args.save_every > 0 and frame_idx % int(args.save_every) == 0:
                save_snapshot(args.save_dir, obs, color, target_mask, frame_idx)

            if args.show:
                debug = draw_debug(
                    color,
                    target,
                    local_map_2ch,
                    actor_difficulty,
                    target_mask,
                    depth_invalid_ratio,
                    args,
                )
                cv2.imshow("Real PCR input check", debug)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
