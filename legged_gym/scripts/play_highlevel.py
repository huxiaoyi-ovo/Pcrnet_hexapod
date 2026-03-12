#!/usr/bin/env python3
"""
Play a high-level (Teacher/Student) planner with Isaac Gym visualization.
"""

import os
import sys
import argparse
import math
import types
import json
import csv
from datetime import datetime
from typing import Optional

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
from isaacgym import gymapi
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.scripts import train_highlevel as th


def _prepare_metrics_dir(args) -> str:
    base = args.metrics_dir
    if base is None or str(base).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join("outputs", "play_highlevel_metrics", f"{args.task}_{stamp}")
    os.makedirs(base, exist_ok=True)
    return base


def _distance_to_oriented_box(center_xy, half_x: float, half_y: float, yaw_rad: float) -> float:
    px = -float(center_xy[0])
    py = -float(center_xy[1])
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    local_x = cos_y * px + sin_y * py
    local_y = -sin_y * px + cos_y * py
    dx = abs(local_x) - float(half_x)
    dy = abs(local_y) - float(half_y)
    outside_x = max(dx, 0.0)
    outside_y = max(dy, 0.0)
    outside = math.hypot(outside_x, outside_y)
    inside = min(max(dx, dy), 0.0)
    return float(max(0.0, outside + inside))


def _prepare_avoid_map_dump_dir(args) -> str:
    base = getattr(args, "avoid_map_dump_dir", None)
    if base is None or str(base).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower() or "default"
        base = os.path.join("outputs", "avoid_map_debug", f"{args.task}_{case}_{stamp}")
    os.makedirs(base, exist_ok=True)
    return base


def _extract_s_avoid_debug_meta(env) -> dict:
    env_impl = getattr(env, "env", None)
    if env_impl is None or not hasattr(env_impl, "s_avoid_active"):
        return {}
    env_id = 0
    active = env_impl.s_avoid_active[env_id]
    if active.numel() == 0 or (not bool(active.any().item())):
        return {}
    slot = int(torch.nonzero(active, as_tuple=False)[0, 0].item())
    robot_xy = env_impl.root_states[env_id, :2]
    obs_xy = env_impl.s_avoid_pos_world[env_id, slot, :2]
    quat = env_impl.root_states[env_id:env_id + 1, 3:7]
    root_yaw = float(env._quat_to_yaw(quat)[0].item())
    ref_xy, ref_yaw = env._get_affordance_reference_pose()
    obs_xy_map = env._world_to_local_xy(
        obs_xy.view(1, 2),
        ref_xy[env_id:env_id + 1],
        ref_yaw[env_id:env_id + 1],
    )[0]
    robot_xy_map = env._world_to_local_xy(
        robot_xy.view(1, 2),
        ref_xy[env_id:env_id + 1],
        ref_yaw[env_id:env_id + 1],
    )[0]
    obs_xy_robot = env._world_to_local_xy(
        obs_xy.view(1, 2),
        robot_xy.view(1, 2),
        ref_yaw[env_id:env_id + 1],
    )[0]
    heading_angle_rad = math.atan2(float(obs_xy_robot[0].item()), float(obs_xy_robot[1].item()))
    body_forward_world = [-math.sin(root_yaw), math.cos(root_yaw)]
    body_plus_y_vs_world_plus_y_rad = math.atan2(body_forward_world[0], body_forward_world[1])
    terrain_cfg = getattr(getattr(getattr(env, "env", None), "cfg", None), "terrain", None)
    target_body_plus_y_deg = 0.0
    if terrain_cfg is not None:
        target_body_plus_y_deg = float(getattr(terrain_cfg, "avoid_spawn_body_plus_y_deg", 0.0))
    target_body_plus_y_rad = math.radians(target_body_plus_y_deg)
    body_plus_y_error_rad = math.atan2(
        math.sin(body_plus_y_vs_world_plus_y_rad - target_body_plus_y_rad),
        math.cos(body_plus_y_vs_world_plus_y_rad - target_body_plus_y_rad),
    )
    world_plus_y_tip = robot_xy + torch.tensor([0.0, 1.0], device=robot_xy.device, dtype=robot_xy.dtype)
    world_plus_y_in_map = env._world_to_local_xy(
        world_plus_y_tip.view(1, 2),
        robot_xy.view(1, 2),
        ref_yaw[env_id:env_id + 1],
    )[0]
    obstacle_center_distance_m = math.hypot(float(obs_xy_map[0].item()), float(obs_xy_map[1].item()))
    obstacle_surface_distance_m = None
    obstacle_shape = "unknown"
    cap_slots = int(getattr(env_impl, "s_avoid_capsule_slot_count", 0))
    box_slots = int(getattr(env_impl, "s_avoid_box_slot_count", 0))
    wall_slots = int(getattr(env_impl, "s_avoid_wall_slot_count", 0))
    if slot < cap_slots:
        obstacle_shape = "capsule"
        radius = float(getattr(env_impl.cfg.terrain, "avoid_capsule_radius", 0.15))
        obstacle_surface_distance_m = max(0.0, obstacle_center_distance_m - radius)
    else:
        quat_world = env_impl.s_avoid_quat_world[env_id, slot].unsqueeze(0)
        yaw_world = float(env._quat_to_yaw(quat_world)[0].item())
        yaw_local = yaw_world - float(ref_yaw[env_id].item())
        if slot < cap_slots + box_slots:
            obstacle_shape = "box"
            half_x = 0.5 * float(getattr(env_impl.cfg.terrain, "avoid_box_size_x", 0.4))
            half_y = 0.5 * float(getattr(env_impl.cfg.terrain, "avoid_box_size_y", 0.4))
            obstacle_surface_distance_m = _distance_to_oriented_box(obs_xy_map, half_x, half_y, yaw_local)
        elif slot < cap_slots + box_slots + wall_slots:
            obstacle_shape = "wall"
            half_x = 0.5 * float(getattr(env_impl.cfg.terrain, "avoid_wall_thickness", 0.12))
            half_y = 0.5 * float(getattr(env_impl.cfg.terrain, "avoid_wall_length", 6.0))
            obstacle_surface_distance_m = _distance_to_oriented_box(obs_xy_map, half_x, half_y, yaw_local)
    obstacle_bearing_rad = math.atan2(float(obs_xy_map[0].item()), float(obs_xy_map[1].item()))
    obstacle_bearing_deg = math.degrees(obstacle_bearing_rad)
    obstacle_center_in_depth_fov = True
    if getattr(env, "camera_fov_rad", None) is not None and env.camera_fov_rad > 0.0:
        delta_bearing = math.atan2(
            math.sin(obstacle_bearing_rad - float(getattr(env, "camera_bearing_rad", 0.0))),
            math.cos(obstacle_bearing_rad - float(getattr(env, "camera_bearing_rad", 0.0))),
        )
        obstacle_center_in_depth_fov = abs(delta_bearing) <= 0.5 * float(env.camera_fov_rad) + 1e-6
    obstacle_center_in_depth_range = True
    if getattr(env, "camera_near", None) is not None and float(env.camera_near) > 0.0:
        obstacle_center_in_depth_range = obstacle_center_in_depth_range and (
            obstacle_center_distance_m >= float(env.camera_near) - 1e-6
        )
    if getattr(env, "camera_far", None) is not None:
        obstacle_center_in_depth_range = obstacle_center_in_depth_range and (
            obstacle_center_distance_m <= float(env.camera_far) + 1e-6
        )
    return {
        "slot": slot,
        "affordance_origin_mode": str(getattr(env, "affordance_origin_mode", "base_center")),
        "camera_mount_xy_local": [
            float(env.affordance_origin_local_xy[0].item()),
            float(env.affordance_origin_local_xy[1].item()),
        ],
        "affordance_origin_xy_world": [
            float(ref_xy[env_id, 0].item()),
            float(ref_xy[env_id, 1].item()),
        ],
        "robot_xy_world": [float(robot_xy[0].item()), float(robot_xy[1].item())],
        "obstacle_xy_world": [float(obs_xy[0].item()), float(obs_xy[1].item())],
        "robot_center_xy_in_map": [
            float(robot_xy_map[0].item()),
            float(robot_xy_map[1].item()),
        ],
        "obstacle_xy_map": [
            float(obs_xy_map[0].item()),
            float(obs_xy_map[1].item()),
        ],
        "obstacle_xy_robot": [
            float(obs_xy_robot[0].item()),
            float(obs_xy_robot[1].item()),
        ],
        "world_plusY_in_robot_frame": [
            float(world_plus_y_in_map[0].item()),
            float(world_plus_y_in_map[1].item()),
        ],
        "body_forward_world_xy": [
            float(body_forward_world[0]),
            float(body_forward_world[1]),
        ],
        "depth_horizontal_fov_deg": float(math.degrees(env.camera_fov_rad)) if getattr(env, "camera_fov_rad", None) is not None else None,
        "depth_vertical_fov_deg": float(math.degrees(env.camera_vertical_fov_rad)) if getattr(env, "camera_vertical_fov_rad", None) is not None else None,
        "depth_tilt_down_deg": float(math.degrees(getattr(env, "camera_tilt_down_rad", 0.0))),
        "depth_camera_height_m": float(getattr(env, "camera_height_m", 0.0)),
        "depth_range_near_m": float(getattr(env, "camera_near", 0.0)),
        "depth_range_far_m": float(getattr(env, "camera_far", 0.0)),
        "obstacle_shape": obstacle_shape,
        "obstacle_center_distance_true_m": float(obstacle_center_distance_m),
        "obstacle_surface_distance_true_m": None if obstacle_surface_distance_m is None else float(obstacle_surface_distance_m),
        "obstacle_center_bearing_deg": float(obstacle_bearing_deg),
        "obstacle_center_in_depth_fov": bool(obstacle_center_in_depth_fov),
        "obstacle_center_in_depth_range": bool(obstacle_center_in_depth_range),
        "target_body_plus_y_vs_world_plusY_rad": float(target_body_plus_y_rad),
        "target_body_plus_y_vs_world_plusY_deg": float(target_body_plus_y_deg),
        "body_plus_y_vs_world_plusY_rad": float(body_plus_y_vs_world_plus_y_rad),
        "body_plus_y_vs_world_plusY_deg": float(math.degrees(body_plus_y_vs_world_plus_y_rad)),
        "body_plus_y_vs_world_plusY_error_rad": float(body_plus_y_error_rad),
        "body_plus_y_vs_world_plusY_error_deg": float(math.degrees(body_plus_y_error_rad)),
        "heading_vs_obstacle_rad": float(heading_angle_rad),
        "heading_vs_obstacle_deg": float(math.degrees(heading_angle_rad)),
        "robot_yaw_rad": float(root_yaw),
    }


def _occupancy_center_from_map(raw_occ: np.ndarray, map_extent: float) -> dict:
    occ_idx = np.argwhere(raw_occ > 0.5)
    pixel_count = int(occ_idx.shape[0])
    if occ_idx.size == 0:
        return {"occupancy_pixel_count": pixel_count}
    cell = float(map_extent) / float(raw_occ.shape[0])
    center = occ_idx.mean(axis=0)
    x_right = -0.5 * float(map_extent) + (float(center[0]) + 0.5) * cell
    y_forward = (float(center[1]) + 0.5) * cell
    dist = np.sqrt(
        (-0.5 * float(map_extent) + (occ_idx[:, 0] + 0.5) * cell) ** 2
        + ((occ_idx[:, 1] + 0.5) * cell) ** 2
    )
    return {
        "occupancy_pixel_count": pixel_count,
        "occupancy_center_idx_xy": [float(center[0]), float(center[1])],
        "occupancy_center_map_xy": [float(x_right), float(y_forward)],
        "occupancy_min_distance_m": float(dist.min()) if dist.size > 0 else None,
    }


def _save_affordance_grid_figure(
    save_path: str,
    tensor: np.ndarray,
    channel_names,
    map_extent: float,
    robot_clearance: float,
    debug_meta: dict,
    *,
    title_prefix: str,
    visible_mask: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as exc:
        print(f"[PlayHigh] ⚠ matplotlib unavailable; skip affordance figure export ({exc}).")
        return

    tensor = np.asarray(tensor, dtype=np.float32)
    channel_count = int(tensor.shape[0])
    fig, axes = plt.subplots(1, channel_count, figsize=(4.6 * channel_count, 4.8), squeeze=False)
    axes = axes[0]
    obstacle_xy = debug_meta.get("obstacle_xy_map", None)
    robot_xy = debug_meta.get("robot_center_xy_in_map", None)
    angle_deg = debug_meta.get("heading_vs_obstacle_deg", None)
    body_world_deg = debug_meta.get("body_plus_y_vs_world_plusY_deg", None)
    body_target_deg = debug_meta.get("target_body_plus_y_vs_world_plusY_deg", None)
    body_error_deg = debug_meta.get("body_plus_y_vs_world_plusY_error_deg", None)
    world_plus_y = debug_meta.get("world_plusY_in_robot_frame", None)
    extent = [-0.5 * float(map_extent), 0.5 * float(map_extent), 0.0, float(map_extent)]
    y_min_plot = -0.35
    if robot_xy is not None and len(robot_xy) == 2:
        y_min_plot = min(y_min_plot, float(robot_xy[1]) - 0.10)
    visible_image = None
    if visible_mask is not None:
        visible_mask = np.asarray(visible_mask, dtype=np.float32)
        if visible_mask.ndim == 2 and visible_mask.shape == tensor.shape[-2:]:
            visible_image = visible_mask.T
    for idx, ax in enumerate(axes):
        name = str(channel_names[idx])
        image = tensor[idx].T
        cmap = "gray" if ("occupancy" in name) else "magma"
        im = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
        if visible_image is not None and "occupancy" in name:
            ax.imshow(
                visible_image,
                origin="lower",
                extent=extent,
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                alpha=0.18,
            )
            ax.contour(
                visible_image,
                levels=[0.5],
                origin="lower",
                extent=extent,
                colors=["deepskyblue"],
                linewidths=1.0,
            )
        ax.set_title(f"{title_prefix}: {name}")
        ax.set_xlabel("x_right (m)")
        ax.set_ylabel("y_forward (m)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(y_min_plot, extent[3])
        ax.scatter([0.0], [0.0], c="cyan", s=28, marker="s")
        if robot_xy is not None and len(robot_xy) == 2:
            robot_x = float(robot_xy[0])
            robot_y = float(robot_xy[1])
            ax.scatter([robot_x], [robot_y], c="white", s=18, marker="o")
            ax.add_patch(Circle((robot_x, robot_y), float(robot_clearance), fill=False, color="cyan", linestyle="--", linewidth=1.2))
            heading_len = 0.55
            ax.plot(
                [robot_x, robot_x],
                [robot_y, robot_y + heading_len],
                color="dodgerblue",
                linewidth=2.2,
            )
            if world_plus_y is not None and len(world_plus_y) == 2:
                world_norm = max(
                    math.hypot(float(world_plus_y[0]), float(world_plus_y[1])),
                    1e-6,
                )
                world_dx = heading_len * float(world_plus_y[0]) / world_norm
                world_dy = heading_len * float(world_plus_y[1]) / world_norm
                ax.plot(
                    [robot_x, robot_x + world_dx],
                    [robot_y, robot_y + world_dy],
                    color="lime",
                    linewidth=2.0,
                )
        else:
            ax.add_patch(Circle((0.0, 0.0), float(robot_clearance), fill=False, color="cyan", linestyle="--", linewidth=1.2))
        if obstacle_xy is not None and len(obstacle_xy) == 2:
            obs_x = float(obstacle_xy[0])
            obs_y = float(obstacle_xy[1])
            ax.scatter([obs_x], [obs_y], c="red", s=36, marker="x")
            if robot_xy is not None and len(robot_xy) == 2:
                ax.plot(
                    [robot_x, obs_x],
                    [robot_y, obs_y],
                    color="purple",
                    linewidth=2.0,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _dump_s_avoid_debug_artifacts(args, env, raw_aff_map: torch.Tensor, local_map_2ch: torch.Tensor) -> None:
    dump_dir = _prepare_avoid_map_dump_dir(args)
    case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower() or "default"
    raw_np = raw_aff_map[0].detach().cpu().numpy().astype(np.float32, copy=False)
    local_np = local_map_2ch[0].detach().cpu().numpy().astype(np.float32, copy=False)
    visible_np = None
    if getattr(env, "affordance_visible_mask", None) is not None:
        visible_np = env.affordance_visible_mask.detach().float().cpu().numpy().astype(np.float32, copy=False)
    np.save(os.path.join(dump_dir, f"{case}_raw_gt_affordance.npy"), raw_np)
    np.save(os.path.join(dump_dir, f"{case}_local_map_2ch.npy"), local_np)
    if visible_np is not None:
        np.save(os.path.join(dump_dir, f"{case}_visible_mask.npy"), visible_np)

    debug_meta = _extract_s_avoid_debug_meta(env)
    occ_meta = _occupancy_center_from_map(raw_np[0], env.affordance_map_extent)
    debug_meta.update({
        "debug_case": case,
        "map_extent_m": float(env.affordance_map_extent),
        "map_size": int(env.affordance_map_size),
        "cell_size_m": float(env.affordance_cell_size),
        "robot_clearance_m": float(env.affordance_clearance),
    })
    debug_meta.update(occ_meta)
    obs_xy = debug_meta.get("obstacle_xy_map", None)
    occ_xy = debug_meta.get("occupancy_center_map_xy", None)
    if obs_xy is not None and occ_xy is not None and len(obs_xy) == 2 and len(occ_xy) == 2:
        debug_meta["occupancy_center_error_map_xy_m"] = [
            float(occ_xy[0] - obs_xy[0]),
            float(occ_xy[1] - obs_xy[1]),
        ]
        debug_meta["occupancy_center_distance_m"] = float(math.hypot(float(occ_xy[0]), float(occ_xy[1])))
        debug_meta["occupancy_center_distance_error_m"] = float(
            debug_meta["occupancy_center_distance_m"] - float(debug_meta.get("obstacle_center_distance_true_m", 0.0))
        )
    occ_min_dist = debug_meta.get("occupancy_min_distance_m", None)
    surf_true_dist = debug_meta.get("obstacle_surface_distance_true_m", None)
    if occ_min_dist is not None and surf_true_dist is not None:
        debug_meta["occupancy_min_distance_error_m"] = float(float(occ_min_dist) - float(surf_true_dist))

    with open(os.path.join(dump_dir, f"{case}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(debug_meta, f, indent=2, ensure_ascii=False)

    _save_affordance_grid_figure(
        os.path.join(dump_dir, f"{case}_raw_gt_affordance.png"),
        raw_np,
        ["occupancy", "passable_gap", "low_obstacle"],
        env.affordance_map_extent,
        env.affordance_clearance,
        debug_meta,
        title_prefix="raw_gt_affordance",
        visible_mask=visible_np,
    )
    _save_affordance_grid_figure(
        os.path.join(dump_dir, f"{case}_local_map_2ch.png"),
        local_np,
        ["occupancy", "clearance_cost"],
        env.affordance_map_extent,
        env.affordance_clearance,
        debug_meta,
        title_prefix="local_map_2ch",
        visible_mask=visible_np,
    )
    if visible_np is not None:
        _save_affordance_grid_figure(
            os.path.join(dump_dir, f"{case}_visible_mask.png"),
            visible_np[None, ...],
            ["visible_mask"],
            env.affordance_map_extent,
            env.affordance_clearance,
            debug_meta,
            title_prefix="debug",
        )

    obs_xy = debug_meta.get("obstacle_xy_map", None)
    occ_xy = debug_meta.get("occupancy_center_map_xy", None)
    err_xy = debug_meta.get("occupancy_center_error_map_xy_m", None)
    robot_xy = debug_meta.get("robot_center_xy_in_map", None)
    heading_angle_deg = debug_meta.get("heading_vs_obstacle_deg", None)
    body_world_deg = debug_meta.get("body_plus_y_vs_world_plusY_deg", None)
    body_target_deg = debug_meta.get("target_body_plus_y_vs_world_plusY_deg", None)
    body_error_deg = debug_meta.get("body_plus_y_vs_world_plusY_error_deg", None)
    occ_pixels = debug_meta.get("occupancy_pixel_count", 0)
    occ_min_dist = debug_meta.get("occupancy_min_distance_m", None)
    surf_true_dist = debug_meta.get("obstacle_surface_distance_true_m", None)
    occ_min_err = debug_meta.get("occupancy_min_distance_error_m", None)
    print(f"[PlayHigh] avoid-map debug dump saved to: {dump_dir}")
    if obs_xy is not None and occ_xy is not None:
        print(
            "[PlayHigh] avoid-map check: origin={} robot_in_map=({:.3f},{:.3f}) obstacle_map=({:.3f},{:.3f}) occ_center=({:.3f},{:.3f}) err=({:.3f},{:.3f}) occ_pixels={} occ_min_dist={:.3f} true_surface_dist={:.3f} dist_err={:.3f} heading_vs_obs={:.1f}deg target_body+y_vs_world+Y={:.1f}deg actual_body+y_vs_world+Y={:.1f}deg body_angle_err={:.1f}deg".format(
                debug_meta.get("affordance_origin_mode", "unknown"),
                0.0 if robot_xy is None else float(robot_xy[0]),
                0.0 if robot_xy is None else float(robot_xy[1]),
                float(obs_xy[0]),
                float(obs_xy[1]),
                float(occ_xy[0]),
                float(occ_xy[1]),
                0.0 if err_xy is None else float(err_xy[0]),
                0.0 if err_xy is None else float(err_xy[1]),
                int(occ_pixels),
                0.0 if occ_min_dist is None else float(occ_min_dist),
                0.0 if surf_true_dist is None else float(surf_true_dist),
                0.0 if occ_min_err is None else float(occ_min_err),
                0.0 if heading_angle_deg is None else float(heading_angle_deg),
                0.0 if body_target_deg is None else float(body_target_deg),
                0.0 if body_world_deg is None else float(body_world_deg),
                0.0 if body_error_deg is None else float(body_error_deg),
            )
        )
    elif obs_xy is not None:
        print(
            "[PlayHigh] avoid-map check: origin={} robot_in_map=({:.3f},{:.3f}) obstacle_map=({:.3f},{:.3f}) occ_pixels={} obstacle_in_fov={} obstacle_in_range={} true_surface_dist={:.3f} heading_vs_obs={:.1f}deg target_body+y_vs_world+Y={:.1f}deg actual_body+y_vs_world+Y={:.1f}deg body_angle_err={:.1f}deg".format(
                debug_meta.get("affordance_origin_mode", "unknown"),
                0.0 if robot_xy is None else float(robot_xy[0]),
                0.0 if robot_xy is None else float(robot_xy[1]),
                float(obs_xy[0]),
                float(obs_xy[1]),
                int(occ_pixels),
                bool(debug_meta.get("obstacle_center_in_depth_fov", False)),
                bool(debug_meta.get("obstacle_center_in_depth_range", False)),
                0.0 if surf_true_dist is None else float(surf_true_dist),
                0.0 if heading_angle_deg is None else float(heading_angle_deg),
                0.0 if body_target_deg is None else float(body_target_deg),
                0.0 if body_world_deg is None else float(body_world_deg),
                0.0 if body_error_deg is None else float(body_error_deg),
            )
        )


def _maybe_apply_e_s_corridor_overrides(args, env_cfg) -> None:
    if env_cfg is None or str(getattr(args, "task", "")) != "e_S_corridor":
        return
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if terrain_cfg is None:
        return
    if getattr(args, "e_s_corridor_width", None) is not None:
        terrain_cfg.e_s_corridor_width = float(args.e_s_corridor_width)
    if getattr(args, "e_s_corridor_wall_thickness", None) is not None:
        terrain_cfg.e_s_corridor_wall_thickness = float(args.e_s_corridor_wall_thickness)
    if getattr(args, "e_s_corridor_curvature", None) is not None:
        terrain_cfg.e_s_corridor_curvature_scale = float(args.e_s_corridor_curvature)


def _maybe_apply_s_avoid_debug_overrides(args, env_cfg) -> None:
    if env_cfg is None or str(getattr(args, "task", "")) != "s_avoid_basic":
        return
    debug_case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower()
    if debug_case == "":
        return
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if terrain_cfg is None:
        return
    terrain_cfg.avoid_map_debug_case = debug_case
    terrain_cfg.avoid_preview_all_stages = False
    target_deg = 0.0 if getattr(args, "avoid_spawn_body_plus_y_deg", None) is None else float(args.avoid_spawn_body_plus_y_deg)
    terrain_cfg.avoid_spawn_body_plus_y_deg = target_deg


def _init_e_s_metrics(args, env) -> dict:
    tol_bearing_rad = math.radians(float(getattr(args, "follow_bearing_tol_deg", 15.0)))
    desired = 1.0
    if hasattr(env, "env") and hasattr(env.env, "nav_cfg") and env.env.nav_cfg is not None:
        desired = float(getattr(env.env.nav_cfg, "follow_distance_desired", 1.0))
    num_envs = int(getattr(env, "num_envs", 1))
    path_len = None
    if hasattr(env, "env") and hasattr(env.env, "e_s_corridor_path_length"):
        path_len = getattr(env.env, "e_s_corridor_path_length")
    return {
        "enabled": str(getattr(args, "task", "")) == "e_S_corridor",
        "metrics_dir": _prepare_metrics_dir(args),
        "follow_dist_tol": float(getattr(args, "follow_dist_tol", 0.25)),
        "follow_bearing_tol_rad": float(tol_bearing_rad),
        "desired_dist": float(desired),
        "autosave_steps": int(max(1, getattr(args, "metrics_autosave_steps", 100))),
        "timeseries": [],
        "episodes": [],
        "current_steps": np.zeros(num_envs, dtype=np.int64),
        "current_follow_hits": np.zeros(num_envs, dtype=np.int64),
        "current_collision": np.zeros(num_envs, dtype=np.bool_),
        "current_collision_steps": np.zeros(num_envs, dtype=np.int64),
        "current_index": np.zeros(num_envs, dtype=np.int64),
        "prev_progress": None,
        "path_length": None if path_len is None else float(path_len),
        "last_export_step": -1,
    }


def _finalize_e_s_episode(metrics: dict, env_id: int, *, success: bool, collided: bool, reason: str) -> None:
    steps = int(metrics["current_steps"][env_id])
    if steps <= 0:
        metrics["current_collision"][env_id] = False
        metrics["current_follow_hits"][env_id] = 0
        metrics["current_steps"][env_id] = 0
        return
    follow_hits = int(metrics["current_follow_hits"][env_id])
    ep_id = int(metrics["current_index"][env_id])
    metrics["episodes"].append(
        {
            "env_id": int(env_id),
            "episode_id": ep_id,
            "steps": steps,
            "tracking_follow_ratio": float(follow_hits / max(steps, 1)),
            "safety_collision_steps": int(metrics["current_collision_steps"][env_id]),
            "safety_collision_step_ratio": float(metrics["current_collision_steps"][env_id] / max(steps, 1)),
            "safety_episode_collided": bool(collided),
            "task_success": bool(success),
            "reason": str(reason),
        }
    )
    metrics["current_index"][env_id] += 1
    metrics["current_collision"][env_id] = False
    metrics["current_follow_hits"][env_id] = 0
    metrics["current_steps"][env_id] = 0
    metrics["current_collision_steps"][env_id] = 0


def _finalize_active_e_s_metrics(metrics: dict, *, reason: str) -> None:
    if not metrics.get("enabled", False):
        return
    num_envs = len(metrics["current_steps"])
    for env_id in range(num_envs):
        if int(metrics["current_steps"][env_id]) <= 0:
            continue
        collided = bool(metrics["current_collision"][env_id])
        _finalize_e_s_episode(
            metrics,
            env_id,
            success=False,
            collided=collided,
            reason=str(reason),
        )


def _update_e_s_metrics(metrics: dict, env, obs_before_step, info, dones, step_idx: int, cmd_tensor) -> None:
    if not metrics.get("enabled", False):
        return
    if obs_before_step is None or "goal" not in obs_before_step:
        return
    goal = obs_before_step["goal"].detach()
    dist = torch.norm(goal[:, :2], dim=1)
    bearing = torch.atan2(goal[:, 0], goal[:, 1])
    follow_ok = (torch.abs(dist - metrics["desired_dist"]) <= metrics["follow_dist_tol"]) & (
        torch.abs(bearing) <= metrics["follow_bearing_tol_rad"]
    )
    collision_mask = None
    if isinstance(info, dict):
        collision_mask = info.get("collision_mask", None)
    if collision_mask is None:
        done_during = info.get("done_during", None) if isinstance(info, dict) else None
        if done_during is None:
            collision_mask = torch.zeros_like(dones, dtype=torch.bool)
        else:
            collision_mask = done_during.to(device=dones.device, dtype=torch.bool)
    else:
        collision_mask = collision_mask.to(device=dones.device, dtype=torch.bool)

    cmd_np = cmd_tensor.detach().cpu().numpy()
    dist_np = dist.detach().cpu().numpy()
    bearing_np = bearing.detach().cpu().numpy()
    follow_np = follow_ok.detach().cpu().numpy().astype(np.int32)
    dones_np = dones.detach().cpu().numpy().astype(np.int32)
    collision_np = collision_mask.detach().cpu().numpy().astype(np.int32)

    progress_tensor = None
    if hasattr(env, "env") and hasattr(env.env, "target_s_corridor_progress"):
        progress_tensor = env.env.target_s_corridor_progress.detach().clone()
        if metrics.get("path_length", None) is None and hasattr(env.env, "e_s_corridor_path_length"):
            metrics["path_length"] = float(env.env.e_s_corridor_path_length)
    progress_np = None if progress_tensor is None else progress_tensor.detach().cpu().numpy()

    num_envs = len(metrics["current_steps"])
    for env_id in range(num_envs):
        metrics["current_steps"][env_id] += 1
        metrics["current_follow_hits"][env_id] += int(follow_np[env_id])
        collided_now = bool(collision_np[env_id] > 0)
        if collided_now:
            metrics["current_collision"][env_id] = True
            metrics["current_collision_steps"][env_id] += 1

    metrics["timeseries"].append(
        {
            "step": int(step_idx),
            "env_id": 0,
            "dist": float(dist_np[0]),
            "bearing_rad": float(bearing_np[0]),
            "bearing_deg": float(bearing_np[0] * 180.0 / math.pi),
            "follow_ok": int(follow_np[0]),
            "collision": int(collision_np[0]),
            "done": int(dones_np[0]),
            "cmd_y": float(cmd_np[0, 1]),
            "cmd_omega": float(cmd_np[0, 2]),
            "progress": float(progress_np[0]) if progress_np is not None else float("nan"),
        }
    )

    prev_progress = metrics.get("prev_progress", None)
    path_len = metrics.get("path_length", None)
    if progress_np is not None and prev_progress is not None and path_len is not None and path_len > 1e-6:
        wrapped = (progress_np + 1e-6 < prev_progress) & (prev_progress > 0.5 * path_len)
    else:
        wrapped = np.zeros(num_envs, dtype=np.bool_)

    for env_id in range(num_envs):
        if bool(dones_np[env_id]):
            collided = bool(metrics["current_collision"][env_id])
            _finalize_e_s_episode(metrics, env_id, success=False, collided=collided, reason="collision_reset")
        elif bool(wrapped[env_id]):
            collided = bool(metrics["current_collision"][env_id])
            _finalize_e_s_episode(metrics, env_id, success=(not collided), collided=collided, reason="path_complete")

    if progress_np is not None:
        metrics["prev_progress"] = progress_np.copy()


def _export_e_s_metrics(metrics: dict, *, final: bool, stop_reason: str) -> None:
    if not metrics.get("enabled", False):
        return
    timeseries = metrics.get("timeseries", [])
    episodes = metrics.get("episodes", [])
    if (not final) and len(timeseries) == 0:
        return
    step_now = int(timeseries[-1]["step"]) if timeseries else -1
    if (not final) and step_now == metrics.get("last_export_step", -1):
        return

    out_dir = metrics["metrics_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ts_path = os.path.join(out_dir, "timeseries.csv")
    ep_path = os.path.join(out_dir, "episodes.csv")
    json_path = os.path.join(out_dir, "summary.json")

    with open(ts_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "env_id", "dist", "bearing_rad", "bearing_deg", "follow_ok", "collision", "done", "cmd_y", "cmd_omega", "progress"],
        )
        writer.writeheader()
        writer.writerows(timeseries)

    with open(ep_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_id",
                "episode_id",
                "steps",
                "tracking_follow_ratio",
                "safety_collision_steps",
                "safety_collision_step_ratio",
                "safety_episode_collided",
                "task_success",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(episodes)

    completed = len(episodes)
    episode_collision_rate = float(sum(int(ep["safety_episode_collided"]) for ep in episodes) / completed) if completed > 0 else 0.0
    total_steps = int(sum(int(ep["steps"]) for ep in episodes))
    total_collision_steps = int(sum(int(ep["safety_collision_steps"]) for ep in episodes))
    collision_step_ratio = float(total_collision_steps / max(total_steps, 1)) if completed > 0 else 0.0
    success_rate = float(sum(int(ep["task_success"]) for ep in episodes) / completed) if completed > 0 else 0.0
    follow_rate = float(sum(ts["follow_ok"] for ts in timeseries) / len(timeseries)) if len(timeseries) > 0 else 0.0
    summary = {
        "task": "e_S_corridor",
        "stop_reason": str(stop_reason),
        "completed_episodes": int(completed),
        "safety_episode_collision_rate": float(episode_collision_rate),
        "safety_collision_step_ratio": float(collision_step_ratio),
        "task_success_rate": float(success_rate),
        "tracking_follow_rate": float(follow_rate),
        "follow_dist_tol": float(metrics["follow_dist_tol"]),
        "follow_bearing_tol_deg": float(metrics["follow_bearing_tol_rad"] * 180.0 / math.pi),
        "desired_dist": float(metrics["desired_dist"]),
        "timeseries_rows": int(len(timeseries)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt

        steps = np.asarray([row["step"] for row in timeseries], dtype=np.float64)
        dist = np.asarray([row["dist"] for row in timeseries], dtype=np.float64)
        bearing_deg = np.asarray([row["bearing_deg"] for row in timeseries], dtype=np.float64)
        collision = np.asarray([row["collision"] for row in timeseries], dtype=np.float64)

        plt.style.use("default")
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8.8, 7.6),
            sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.2, 0.7]},
        )
        fig.suptitle("e_S_corridor Tracking Evaluation", fontsize=13, fontweight="semibold")

        dist_tol = float(metrics["follow_dist_tol"])
        d_des = float(metrics["desired_dist"])
        axes[0].plot(steps, dist, color="#1f77b4", linewidth=1.8, label="distance")
        axes[0].axhline(d_des, color="#222222", linestyle="--", linewidth=1.2, label="desired")
        axes[0].fill_between(
            steps,
            d_des - dist_tol,
            d_des + dist_tol,
            color="#1f77b4",
            alpha=0.12,
            label="follow band",
        )
        axes[0].set_ylabel("Distance (m)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="upper right", frameon=False)

        bearing_tol_deg = float(metrics["follow_bearing_tol_rad"] * 180.0 / math.pi)
        axes[1].plot(steps, bearing_deg, color="#d62728", linewidth=1.8, label="bearing")
        axes[1].axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        axes[1].fill_between(
            steps,
            -bearing_tol_deg,
            bearing_tol_deg,
            color="#d62728",
            alpha=0.12,
            label="follow band",
        )
        axes[1].set_ylabel("Bearing (deg)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right", frameon=False)

        axes[2].plot(steps, collision, color="#111111", linewidth=1.2, label="collision flag")
        axes[2].set_ylabel("Collision")
        axes[2].set_xlabel("Step")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="upper right", frameon=False)

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
        fig.savefig(os.path.join(out_dir, "tracking_timeseries.png"), dpi=260, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, "tracking_timeseries.pdf"), bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7.6, 4.8))
        names = ["Follow", "Success", "EpCollision", "StepCollision"]
        vals = [follow_rate, success_rate, episode_collision_rate, collision_step_ratio]
        colors = ["#4c78a8", "#59a14f", "#e15759", "#f28e2b"]
        bars = ax2.bar(names, vals, color=colors, width=0.58)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("Rate")
        ax2.set_title("e_S_corridor Evaluation Summary")
        ax2.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() * 0.5, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "summary_rates.png"), dpi=260, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, "summary_rates.pdf"), bbox_inches="tight")
        plt.close(fig2)
    except Exception as exc:
        print(f"[PlayHigh] ⚠ metrics plot skipped: {exc}")

    metrics["last_export_step"] = step_now
    print(
        "[PlayHigh][metrics] saved: dir={} follow={:.3f} success={:.3f} ep_collision={:.3f} step_collision={:.3f} completed_eps={} reason={}".format(
            out_dir,
            follow_rate,
            success_rate,
            episode_collision_rate,
            collision_step_ratio,
            completed,
            stop_reason,
        )
    )


def parse_args():
    raw_argv = list(sys.argv)
    parser = argparse.ArgumentParser(description="Play high-level planner in Isaac Gym")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (required): s_follow_basic / s_avoid_basic / s_cylinder / s_calib / e_L_conflict ...",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（None 使用默认）")
    parser.add_argument(
        "--mode",
        type=str,
        default="teacher",
        choices=["teacher", "student"],
        help="Planner mode (teacher or student)",
    )
    parser.add_argument(
        "--low_level_ckpt",
        type=str,
        required=True,
        help="Low-level policy checkpoint path",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="Expert checkpoint path")
    parser.add_argument(
        "--skill",
        type=str,
        default="follow",
        choices=["follow", "avoid", "moe"],
        help="Expert skill: follow / avoid / moe (gate)",
    )
    parser.add_argument("--follow_ckpt", type=str, default=None, help="(moe) Follow expert checkpoint")
    parser.add_argument("--avoid_ckpt", type=str, default=None, help="(moe) Avoid expert checkpoint")
    parser.add_argument("--gate_use_difficulty", action="store_true", help="Gate 使用 difficulty 作为输入（特权信息）")
    parser.add_argument("--vision_ckpt", type=str, default=None, help="Student vision checkpoint path")
    parser.add_argument("--aff_stack", type=int, default=1, help="affordance 堆叠帧数 (短时记忆)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--decimation", type=int, default=5, help="High/low frequency ratio")
    parser.add_argument("--headless", action="store_true", default=False, help="Disable viewer")
    parser.add_argument("--max_steps", type=int, default=0, help="Max steps (0 = infinite)")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--camera_enable", action="store_true", default=False, help="Enable depth camera")
    parser.add_argument("--camera_show", action="store_true", default=False, help="Show depth frames")
    parser.add_argument("--camera_save", action="store_true", default=False, help="Save depth frames")
    parser.add_argument("--camera_dir", type=str, default="outputs/play_highlevel_camera", help="Camera output dir")
    parser.add_argument("--camera_interval", type=int, default=5, help="Camera capture interval")
    parser.add_argument("--camera_env", type=int, default=0, help="Env index for camera output")
    parser.add_argument("--cmd_slew_lin", type=float, default=0.2, help="命令线速度变化率限制")
    parser.add_argument("--cmd_slew_ang", type=float, default=0.4, help="命令角速度变化率限制")
    parser.add_argument("--cmd_safe_dist", type=float, default=None, help="安全距离阈值（None 使用默认 clearance）")
    parser.add_argument("--cmd_free_dist", type=float, default=None, help="安全全速距离（None 使用默认 clearance_free）")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="(V7) 固定风险预算旋钮 beta：0=快/激进，1=安全/保守；None=禁用（保持旧行为）",
    )
    parser.add_argument("--disable_risk_scale", action="store_true", help="禁用 CommandPostProcessor 风险缩放（消融用）")
    parser.add_argument(
        "--debug_cmd",
        dest="debug_cmd",
        action="store_true",
        default=False,
        help="Print high-level command debug info",
    )
    parser.add_argument(
        "--no_debug_cmd",
        dest="debug_cmd",
        action="store_false",
        help="Disable debug output",
    )
    parser.add_argument("--debug_interval", type=int, default=10, help="Debug print interval (steps)")
    parser.add_argument("--debug", action="store_true", help="debug 输出（诊断信息）")
    parser.add_argument("--metrics_dir", type=str, default=None, help="评测统计与绘图输出目录（默认自动创建）")
    parser.add_argument("--metrics_autosave_steps", type=int, default=100, help="统计中间结果自动落盘步数间隔")
    parser.add_argument("--follow_dist_tol", type=float, default=0.25, help="跟随率距离阈值（米）")
    parser.add_argument("--follow_bearing_tol_deg", type=float, default=15.0, help="跟随率朝向阈值（度）")
    parser.add_argument("--e_s_corridor_width", type=float, default=None, help="覆盖 e_S_corridor 的走廊净宽")
    parser.add_argument("--e_s_corridor_wall_thickness", type=float, default=None, help="覆盖 e_S_corridor 的墙厚")
    parser.add_argument("--e_s_corridor_curvature", type=float, default=None, help="覆盖 e_S_corridor 的弯道曲率缩放")
    parser.add_argument(
        "--avoid_map_debug_case",
        type=str,
        default="",
        choices=["", "front", "left", "right", "side_left", "side_right"],
        help="s_avoid_basic 静态验证：机器人固定不动，障碍按机身前方/左前/右前放置",
    )
    parser.add_argument(
        "--avoid_spawn_body_plus_y_deg",
        type=float,
        default=None,
        help="s_avoid_basic 静态验证：强制设置出生时机体 +y 相对世界 +Y 的夹角（度）",
    )
    parser.add_argument(
        "--expert_k_yaw",
        type=float,
        default=None,
        help="S0 expert override: yaw gain k_yaw (None keeps default)",
    )
    parser.add_argument(
        "--expert_heading_lock",
        type=float,
        default=None,
        help="S0 expert override: lock threshold in radians (None keeps default)",
    )
    parser.add_argument(
        "--expert_heading_release",
        type=float,
        default=None,
        help="S0 expert override: release threshold in radians (None keeps default)",
    )
    parser.add_argument(
        "--disable_success_reset",
        action="store_true",
        default=False,
        help="临时关闭 S0 成功即重置（用于长时间跟随观察）",
    )
    parser.add_argument(
        "--keep_success_reset",
        action="store_true",
        default=False,
        help="即使在 S0 + expert-only 模式也保留成功即重置",
    )
    parser.add_argument(
        "--show_reset_reason",
        action="store_true",
        default=False,
        help="打印每次 done 的重置原因归类（success/done_during/timeout/other）",
    )
    parser.add_argument(
        "--show_expert_cmd",
        action="store_true",
        default=False,
        help="Debug: 打印 S0 expert 命令（与训练 EGPO 同口径）",
    )
    parser.add_argument(
        "--use_expert_cmd",
        action="store_true",
        default=False,
        help="兼容旧参数：等价于 --use_follow_expert",
    )
    parser.add_argument(
        "--use_follow_expert",
        action="store_true",
        default=False,
        help="Follow 专家直管输出：高层命令直接由 expert_s0_follow 生成（无需 --teacher_ckpt）",
    )
    parser.add_argument(
        "--force_cmd",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=None,
        help="Debug: 强制覆盖高层命令为固定 [vx, vy, wz]（用于轴向验收）",
    )
    parser.add_argument(
        "--heading_offset_override",
        type=float,
        default=None,
        help="Override heading_offset_rad (radians) for debug alignment",
    )
    parser.add_argument(
        "--heading_offset_flip",
        action="store_true",
        default=False,
        help="Flip heading_offset_rad sign for debug alignment",
    )
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    if not hasattr(args, "physics_engine"):
        args.physics_engine = gymapi.SIM_PHYSX
    if not hasattr(args, "sim_device_type"):
        args.sim_device_type = "cuda"
    if not hasattr(args, "compute_device_id"):
        args.compute_device_id = 0
    if not hasattr(args, "sim_device_id"):
        args.sim_device_id = args.compute_device_id
    if not hasattr(args, "sim_device"):
        if args.sim_device_type == "cuda":
            args.sim_device = f"cuda:{args.sim_device_id}"
        else:
            args.sim_device = "cpu"
    if not hasattr(args, "use_gpu"):
        args.use_gpu = args.sim_device_type == "cuda"
    if not hasattr(args, "use_gpu_pipeline"):
        args.use_gpu_pipeline = args.sim_device_type == "cuda"
    if not hasattr(args, "subscenes"):
        args.subscenes = 0
    if not hasattr(args, "num_threads"):
        args.num_threads = 0
    if not hasattr(args, "rl_device"):
        args.rl_device = args.sim_device

    if args.camera_interval < 1:
        args.camera_interval = 1
    if args.debug_interval < 1:
        args.debug_interval = 1
    if args.debug and "--no_debug_cmd" not in raw_argv:
        args.debug_cmd = True
    if hasattr(th, "normalize_task_name"):
        args.task = th.normalize_task_name(getattr(args, "task", ""))

    return args


def main():
    args = parse_args()
    if args.task == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用 hex_ground / s_avoid_basic..s_ood_holdout / s_calib")
    supported_tasks = (
        "s_follow_basic",
        "s_avoid_basic",
        "s_cylinder",
        "s_narrow_passage",
        "s_step_field",
        "s_dense_obstacles",
        "s_ood_holdout",
        "s_calib",
    )
    if not (args.task in supported_tasks or args.task.startswith("e_")):
        raise ValueError(
            "play_highlevel.py supports only "
            "--task s_follow_basic/s_avoid_basic/s_cylinder/"
            "s_narrow_passage/s_step_field/s_dense_obstacles/s_ood_holdout/s_calib "
            "or e_* paper scenes"
        )
    use_follow_expert = bool(getattr(args, "use_follow_expert", False)) or bool(getattr(args, "use_expert_cmd", False))
    avoid_map_debug_case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower()
    static_avoid_debug = args.task == "s_avoid_basic" and avoid_map_debug_case != ""
    if use_follow_expert and getattr(args, "skill", "follow") != "follow":
        raise ValueError("--use_follow_expert 仅支持 --skill follow")
    if static_avoid_debug and args.num_envs != 1:
        print(f"[PlayHigh] avoid_map_debug_case={avoid_map_debug_case}: forcing num_envs=1 (was {args.num_envs})")
        args.num_envs = 1
    if use_follow_expert and not bool(getattr(args, "show_expert_cmd", False)):
        # In expert takeover mode, always compute and print expert commands.
        args.show_expert_cmd = True
    if bool(getattr(args, "use_expert_cmd", False)) and not bool(getattr(args, "use_follow_expert", False)):
        print("[PlayHigh] ⚠ --use_expert_cmd 已兼容映射到 --use_follow_expert。")
    if (args.expert_heading_lock is not None) and (args.expert_heading_release is not None):
        if not (float(args.expert_heading_release) < float(args.expert_heading_lock)):
            raise ValueError("--expert_heading_release must be < --expert_heading_lock")
    debug = bool(getattr(args, "debug", False))
    def dprint(*vals, **kwargs):
        if debug:
            print(*vals, **kwargs)
    dprint("[PlayHigh] V5 任务需显式传入 --task；hex_terrain 已移除。")
    if getattr(args, "aff_stack", 1) > 1:
        print(f"[PlayHigh] aff_stack={args.aff_stack}: 输入通道数改变，需与 ckpt 训练时一致，否则无法加载。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th.import_modules()
    if args.mode == "student" and not args.vision_ckpt:
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")

    if args.camera_show and args.headless:
        print("[PlayHigh] ⚠ camera_show requested but headless=True. Disabling.")
        args.camera_show = False

    if args.camera_show or args.camera_save:
        args.camera_enable = True
    if args.mode == "student":
        args.camera_enable = True

    camera_cv2 = None
    camera_warned_no_cv2 = False
    if args.camera_show:
        try:
            import cv2
            camera_cv2 = cv2
        except Exception as exc:
            print(f"[PlayHigh] ⚠ cv2 not available ({exc}); disabling camera_show.")
            args.camera_show = False

    if args.camera_save:
        os.makedirs(args.camera_dir, exist_ok=True)

    env_cfg = None
    train_cfg = None
    if getattr(th, "task_registry", None) is not None:
        env_cfg, train_cfg = th.task_registry.get_cfgs(name=args.task)
        if args.seed is not None:
            env_cfg.seed = int(args.seed)
        _maybe_apply_e_s_corridor_overrides(args, env_cfg)
        _maybe_apply_s_avoid_debug_overrides(args, env_cfg)
    env = th.HierarchicalHexapodEnv(args, device, env_cfg=env_cfg, train_cfg=train_cfg)
    if str(getattr(args, "task", "")).startswith("e_"):
        terrain_type_dbg = getattr(getattr(getattr(env, "env", None), "cfg", None), "terrain", None)
        terrain_type_dbg = getattr(terrain_type_dbg, "terrain_type", "unknown")
        moving_mode_dbg = getattr(getattr(getattr(env, "env", None), "nav_cfg", None), "moving_target_mode", "unknown")
        print(
            f"[PlayHigh] e-scene check: task={args.task}, terrain_type={terrain_type_dbg}, moving_target_mode={moving_mode_dbg}"
        )
        if args.task == "e_S_corridor":
            terrain_cfg_dbg = getattr(getattr(env, "env", None), "cfg", None)
            terrain_cfg_dbg = getattr(terrain_cfg_dbg, "terrain", None)
            print(
                "[PlayHigh] e_S_corridor overrides: width={} wall_t={} curvature={}".format(
                    getattr(terrain_cfg_dbg, "e_s_corridor_width", None),
                    getattr(terrain_cfg_dbg, "e_s_corridor_wall_thickness", None),
                    getattr(terrain_cfg_dbg, "e_s_corridor_curvature_scale", None),
                )
            )
    # In S0 expert takeover debugging, default to long-horizon observation:
    # disable success-triggered resets unless explicitly kept.
    auto_disable_success_reset = (args.task == "s_follow_basic" and use_follow_expert)
    disable_success_reset = bool(getattr(args, "disable_success_reset", False)) or (
        auto_disable_success_reset and (not bool(getattr(args, "keep_success_reset", False)))
    )
    if disable_success_reset and hasattr(env, "s0_follow_steps_success"):
        env.s0_follow_steps_success = int(10 ** 9)
        if hasattr(env, "s0_follow_success_bonus"):
            env.s0_follow_success_bonus = 0.0
        print("[PlayHigh] success-reset disabled for S0 follow (long-horizon debug mode).")
    if hasattr(env, "env") and hasattr(env.env, "cfg") and hasattr(env.env.cfg, "terrain"):
        env.env.cfg.terrain.curriculum = False
    if hasattr(env.env, "_update_terrain_curriculum"):
        def _no_update(self, env_ids):
            return
        env.env._update_terrain_curriculum = types.MethodType(_no_update, env.env)
    if hasattr(env.env, "terrain_levels"):
        env.env.terrain_levels.fill_(0)
        if hasattr(env.env, "terrain_origins") and hasattr(env.env, "terrain_types") and hasattr(env.env, "env_origins"):
            env.env.env_origins[:] = env.env.terrain_origins[env.env.terrain_levels, env.env.terrain_types]
        dprint("[PlayHigh] curriculum disabled; start at level 0")
    if hasattr(env, "env") and hasattr(env.env, "debug_viz"):
        env.env.debug_viz = bool(getattr(args, "debug", False)) or static_avoid_debug
    vision_model = None
    s0_expert_fn = None
    if bool(getattr(args, "show_expert_cmd", False)):
        try:
            from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd as _s0_expert_fn
            s0_expert_fn = _s0_expert_fn
            dprint("[PlayHigh] expert cmd debug enabled")
        except Exception as exc:
            print(f"[PlayHigh] ⚠ failed to import S0 expert function: {exc}; disabling --show_expert_cmd.")
            args.show_expert_cmd = False
    if args.mode == "student":
        vision_model = th.AffordanceEstimator(
            depth_channels=1,
            output_size=16,
            max_depth_range=5.0
        ).to(device)
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        vision_model.load_state_dict(state_dict)
        vision_model.eval()
        dprint(f"[PlayHigh] ✓ Vision 加载成功: {args.vision_ckpt}")
    if args.camera_env < 0:
        args.camera_env = 0
    if args.camera_env >= env.num_envs:
        print(f"[PlayHigh] ⚠ camera_env={args.camera_env} out of range; clamping to {env.num_envs - 1}.")
        args.camera_env = env.num_envs - 1
    viewer = getattr(env.env, "viewer", None) if hasattr(env, "env") else None
    input_enabled = viewer is not None and not args.headless
    if not input_enabled and not args.headless:
        print("[PlayHigh] ⚠ viewer not available; keyboard controls disabled.")
    if input_enabled:
        dprint("[PlayHigh] 键盘控制: R=重置, A=降级, D=升级")
        gym = env.env.gym
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET_ENV")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "LEVEL_DOWN")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "LEVEL_UP")
    heading_offset = 0.0
    if hasattr(env, "reward_cfg") and env.reward_cfg is not None:
        heading_offset = float(getattr(env.reward_cfg, "heading_offset_rad", 0.0))
    if args.heading_offset_override is not None:
        heading_offset = float(args.heading_offset_override)
    elif args.heading_offset_flip:
        heading_offset = -heading_offset
    if hasattr(env, "reward_cfg") and env.reward_cfg is not None:
        env.reward_cfg.heading_offset_rad = heading_offset
    if hasattr(env, "env") and hasattr(env.env, "nav_cfg") and env.env.nav_cfg is not None:
        env.env.nav_cfg.heading_offset_rad = heading_offset
    dprint(f"[PlayHigh] heading_offset_rad={heading_offset:.3f} (effective)")
    force_cmd_tensor = None
    if args.force_cmd is not None:
        force_cmd_tensor = torch.tensor(
            [float(args.force_cmd[0]), float(args.force_cmd[1]), float(args.force_cmd[2])],
            device=device,
            dtype=torch.float32,
        ).view(1, 3)
        print(
            "[PlayHigh] force_cmd enabled: "
            f"[{force_cmd_tensor[0,0].item():.3f}, {force_cmd_tensor[0,1].item():.3f}, {force_cmd_tensor[0,2].item():.3f}]"
        )

    def _get_max_level():
        if not hasattr(env.env, "terrain_levels"):
            return 0
        max_level = int(getattr(env.env, "max_terrain_level", 0))
        if max_level <= 0 and hasattr(env.env, "terrain_origins"):
            max_level = int(env.env.terrain_origins.shape[0])
        if max_level <= 0 and hasattr(env.env, "cfg"):
            max_level = int(getattr(env.env.cfg.terrain, "num_rows", 1))
        return max(1, max_level)
    skill = getattr(args, "skill", "follow")

    def _get_aff_maps(current_obs):
        if current_obs is None:
            raise ValueError("obs is None when building affordance map.")
        if args.mode == "student":
            if vision_model is None:
                raise RuntimeError("vision_model is not initialized in student mode.")
            with torch.no_grad():
                vis_out = vision_model(current_obs["depth"], normalize=True)
                raw_aff = torch.stack([
                    vis_out["occupancy"],
                    vis_out["passable_gap"],
                    vis_out["low_obstacle"],
                ], dim=1)
                raw_aff = th.resize_affordance_map(raw_aff, env.affordance_map_size)
        else:
            raw_aff = current_obs["gt_affordance"]
        if skill == "avoid":
            if args.mode == "student":
                policy_aff = th.build_avoid_local_map_2ch(
                    raw_aff,
                    visible_mask=getattr(env, "affordance_visible_mask", None),
                )
            else:
                policy_aff = current_obs.get(
                    "local_map_2ch",
                    th.build_avoid_local_map_2ch(
                        raw_aff,
                        visible_mask=getattr(env, "affordance_visible_mask", None),
                    ),
                )
        else:
            policy_aff = raw_aff
        return raw_aff, policy_aff

    def _get_difficulty(current_obs, current_aff):
        if skill == "avoid" and "actor_difficulty" in current_obs:
            return current_obs["actor_difficulty"]
        if args.mode == "student":
            return th.difficulty_from_gap(current_aff)
        return current_obs["gt_difficulty"]

    obs = env.reset()
    raw_aff_map, aff_map = _get_aff_maps(obs)
    aff_shape = aff_map.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    is_gate = skill == "moe"
    expert_only_mode = use_follow_expert or static_avoid_debug
    policy = None
    follow_policy = None
    avoid_policy = None
    if use_follow_expert:
        print("[PlayHigh] cmd_source=follow_expert (--use_follow_expert)")
    elif static_avoid_debug:
        print(f"[PlayHigh] cmd_source=zero_cmd (--avoid_map_debug_case={avoid_map_debug_case})")
        print("[PlayHigh] static avoid-map debug enabled; robot command is clamped to zero.")
    if not expert_only_mode:
        if not args.teacher_ckpt:
            raise ValueError("非 expert-only 模式必须提供 --teacher_ckpt")
        if is_gate:
            if not args.follow_ckpt or not args.avoid_ckpt:
                raise ValueError("moe 需要 --follow_ckpt 和 --avoid_ckpt")
            policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=obs["state"].shape[1],
                goal_dim=obs["goal"].shape[1],
            ).to(device)
            follow_policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=obs["state"].shape[1],
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            avoid_policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=obs["state"].shape[1],
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            gate_ckpt = torch.load(args.teacher_ckpt, map_location=device)
            gate_state = gate_ckpt["model_state_dict"] if isinstance(gate_ckpt, dict) and "model_state_dict" in gate_ckpt else gate_ckpt
            th.load_high_level_state_dict_compat(policy, gate_state, label="play_gate")
            for model, ckpt_path in [(follow_policy, args.follow_ckpt), (avoid_policy, args.avoid_ckpt)]:
                ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                th.load_high_level_state_dict_compat(
                    model,
                    state_dict,
                    label=f"play_expert:{os.path.basename(ckpt_path)}",
                )
                model.eval()
            policy.eval()
        else:
            policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=obs["state"].shape[1],
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            th.load_high_level_state_dict_compat(policy, state_dict, label="play_policy")
            policy.eval()
    else:
        dprint("[PlayHigh] expert-only takeover enabled; skip policy checkpoint loading.")
    if static_avoid_debug:
        local_map_2ch = obs.get("local_map_2ch", th.build_avoid_local_map_2ch(raw_aff_map))
        _dump_s_avoid_debug_artifacts(args, env, raw_aff_map, local_map_2ch)
    if expert_only_mode and (
        (args.expert_k_yaw is not None)
        or (args.expert_heading_lock is not None)
        or (args.expert_heading_release is not None)
    ):
        print(
            "[PlayHigh] expert overrides: k_yaw={} lock={} release={}".format(
                args.expert_k_yaw, args.expert_heading_lock, args.expert_heading_release
            )
        )
    step_idx = 0
    deterministic = not args.stochastic
    camera_frame_idx = 0
    e_s_metrics = _init_e_s_metrics(args, env)
    stop_reason = "running"
    track_env_idx = min(max(int(args.camera_env), 0), env.num_envs - 1)
    prev_track_pos_world = None
    prev_track_yaw = None
    axis_disp_world_sum = np.zeros(2, dtype=np.float64)
    axis_disp_body_sum = np.zeros(2, dtype=np.float64)
    axis_disp_count = 0

    prev_dist = None
    aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
    aff_stack_fill = torch.ones(env.num_envs, device=device)
    stack_reset_mask = None
    level_up_pressed = False
    level_down_pressed = False
    try:
        while True:
            manual_reset = False
            level_delta = 0
            if input_enabled:
                for evt in env.env.gym.query_viewer_action_events(viewer):
                    if evt.action == "RESET_ENV" and evt.value > 0:
                        manual_reset = True
                    elif evt.action == "LEVEL_DOWN":
                        if evt.value > 0 and not level_down_pressed:
                            level_delta -= 1
                            level_down_pressed = True
                        elif evt.value <= 0:
                            level_down_pressed = False
                    elif evt.action == "LEVEL_UP":
                        if evt.value > 0 and not level_up_pressed:
                            level_delta += 1
                            level_up_pressed = True
                        elif evt.value <= 0:
                            level_up_pressed = False
            if manual_reset or level_delta != 0:
                if level_delta != 0 and hasattr(env.env, "terrain_levels"):
                    env_idx = args.camera_env
                    max_level = _get_max_level()
                    current_level = int(env.env.terrain_levels[env_idx].item())
                    current_level = int(np.clip(current_level, 0, max_level - 1))
                    new_level = int(np.clip(current_level + level_delta, 0, max_level - 1))
                    env.env.terrain_levels[env_idx] = new_level
                    if (hasattr(env.env, "terrain_origins")
                            and hasattr(env.env, "terrain_types")
                            and hasattr(env.env, "env_origins")):
                        env.env.env_origins[env_idx] = env.env.terrain_origins[new_level, env.env.terrain_types[env_idx]]
                    print(f"[PlayHigh] curriculum level -> {new_level}")
                obs = env.reset()
                raw_aff_map, aff_map = _get_aff_maps(obs)
                aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
                aff_stack_fill.fill_(1)
                stack_reset_mask = None
                prev_dist = None
                continue
            obs_before_step = None
            if e_s_metrics.get("enabled", False) and (obs is not None) and ("goal" in obs):
                obs_before_step = {"goal": obs["goal"].detach().clone()}
            reset_mask = stack_reset_mask
            if stack_reset_mask is not None and stack_reset_mask.any():
                _, reset_aff = _get_aff_maps(obs)
                aff_stack_buf[stack_reset_mask] = reset_aff[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                aff_stack_fill[stack_reset_mask] = 1
                stack_reset_mask = None
            raw_aff_map, aff_map = _get_aff_maps(obs)
            aff_stack_buf = torch.roll(aff_stack_buf, shifts=-aff_map.shape[1], dims=1)
            aff_stack_buf[:, -aff_map.shape[1]:, :, :] = aff_map
            if aff_stack > 1:
                if reset_mask is None:
                    aff_stack_fill = torch.clamp(aff_stack_fill + 1, max=aff_stack)
                else:
                    inc_mask = ~reset_mask
                    if inc_mask.any():
                        aff_stack_fill[inc_mask] = torch.clamp(aff_stack_fill[inc_mask] + 1, max=aff_stack)
            else:
                aff_stack_fill.fill_(1)
            difficulty = _get_difficulty(obs, aff_map)
            gate_y = None
            cmd = torch.zeros((env.num_envs, 3), device=device, dtype=torch.float32)
            if not expert_only_mode:
                with torch.no_grad():
                    if is_gate:
                        gate_difficulty = difficulty if args.gate_use_difficulty else torch.zeros_like(difficulty)
                        cmd_f, _ = follow_policy.get_action(
                            aff_stack_buf,
                            obs["state"],
                            obs["goal"],
                            difficulty,
                            deterministic=True,
                        )
                        cmd_a, _ = avoid_policy.get_action(
                            aff_stack_buf,
                            obs["state"],
                            obs["goal"],
                            difficulty,
                            deterministic=True,
                        )
                        gate_y, _ = policy.get_action(
                            aff_stack_buf,
                            obs["state"],
                            obs["goal"],
                            gate_difficulty,
                            deterministic=deterministic,
                        )
                        cmd = gate_y.unsqueeze(-1) * cmd_f + (1.0 - gate_y.unsqueeze(-1)) * cmd_a
                    else:
                        cmd, _ = policy.get_action(
                            aff_stack_buf,
                            obs["state"],
                            obs["goal"],
                            difficulty,
                            deterministic=deterministic,
                        )
            expert_cmd = None
            dircheck_alpha_pre = None
            dircheck_x_pre = None
            dircheck_y_pre = None
            dircheck_omega_pre = None
            dircheck_goal_x_pre = None
            dircheck_goal_y_pre = None
            dircheck_goal_err_pre = None
            if bool(getattr(args, "show_expert_cmd", False)) and s0_expert_fn is not None and getattr(args, "skill", "follow") == "follow":
                quat_e = env.env.root_states[:, 3:7]
                x_q, y_q, z_q, w_q = quat_e[:, 0], quat_e[:, 1], quat_e[:, 2], quat_e[:, 3]
                robot_heading = torch.atan2(
                    2.0 * (w_q * z_q + x_q * y_q),
                    1.0 - 2.0 * (y_q * y_q + z_q * z_q),
                )
                heading_offset_dbg = float(getattr(getattr(env, "reward_cfg", None), "heading_offset_rad", 0.0))
                robot_heading_for_expert = robot_heading + heading_offset_dbg
                robot_heading_for_expert = torch.atan2(
                    torch.sin(robot_heading_for_expert),
                    torch.cos(robot_heading_for_expert),
                )
                target_world_xy = getattr(env.env, "target_world", None)
                if target_world_xy is None:
                    target_world_xy = torch.zeros(env.num_envs, 2, device=device)
                expert_kwargs = {}
                if args.expert_k_yaw is not None:
                    expert_kwargs["k_yaw"] = float(args.expert_k_yaw)
                if args.expert_heading_lock is not None:
                    expert_kwargs["heading_lock_rad"] = float(args.expert_heading_lock)
                if args.expert_heading_release is not None:
                    expert_kwargs["heading_release_rad"] = float(args.expert_heading_release)
                expert_cmd = s0_expert_fn(
                    robot_pos_world_xy=env.env.root_states[:, :2],
                    robot_heading=robot_heading_for_expert,
                    target_world_xy=target_world_xy,
                    target_vel_world_xy=None,
                    target_heading=None,
                    cmd_scale=cmd_scale,
                    reset_mask=reset_mask,
                    **expert_kwargs,
                )
                # Direction-check diagnostic uses the exact pre-step geometry fed to expert.
                delta_w = target_world_xy - env.env.root_states[:, :2]
                cos_h = torch.cos(robot_heading_for_expert)
                sin_h = torch.sin(robot_heading_for_expert)
                x_right = cos_h * delta_w[:, 0] + sin_h * delta_w[:, 1]
                y_forward = -sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
                alpha_pre = torch.atan2(x_right, y_forward)
                dircheck_alpha_pre = alpha_pre
                dircheck_x_pre = x_right
                dircheck_y_pre = y_forward
                dircheck_omega_pre = expert_cmd[:, 2]
                if hasattr(env.env, "goal_buf") and torch.is_tensor(env.env.goal_buf):
                    goal_buf_pre = env.env.goal_buf.detach().clone()
                    if goal_buf_pre.ndim == 2 and goal_buf_pre.shape[1] >= 2:
                        dircheck_goal_x_pre = goal_buf_pre[:, 0]
                        dircheck_goal_y_pre = goal_buf_pre[:, 1]
                        dircheck_goal_err_pre = torch.sqrt(
                            (dircheck_goal_x_pre - x_right) ** 2 + (dircheck_goal_y_pre - y_forward) ** 2
                        )
            if use_follow_expert:
                if expert_cmd is None:
                    raise RuntimeError("--use_follow_expert requires --show_expert_cmd and --skill follow")
                cmd = expert_cmd
            elif static_avoid_debug:
                cmd = torch.zeros_like(cmd)
            if force_cmd_tensor is not None:
                cmd = force_cmd_tensor.expand(env.num_envs, -1)
            env.clearance_override = env._compute_clearance_from_affordance(aff_map)
            env.reward_affordance_override = aff_map
            obs, rewards, dones, info = env.step(cmd, gate_y if is_gate else None)
            _update_e_s_metrics(e_s_metrics, env, obs_before_step, info, dones, step_idx, cmd)
            if e_s_metrics.get("enabled", False) and ((step_idx + 1) % e_s_metrics["autosave_steps"] == 0):
                _export_e_s_metrics(e_s_metrics, final=False, stop_reason="autosave")
            done_during = None
            if info is not None and isinstance(info, dict):
                done_during = info.get("done_during", None)
            if done_during is None:
                done_during = torch.zeros_like(dones, dtype=torch.bool)
            else:
                done_during = done_during.to(device=dones.device, dtype=torch.bool)

            reward_terms = info.get("reward_terms") if isinstance(info, dict) else None
            success_mask = torch.zeros_like(dones, dtype=torch.bool)
            if isinstance(reward_terms, dict) and ("success_bonus" in reward_terms):
                success_bonus = reward_terms["success_bonus"]
                if torch.is_tensor(success_bonus):
                    success_mask = success_bonus.to(device=dones.device) > 0.0

            timeout_mask = torch.zeros_like(dones, dtype=torch.bool)
            if hasattr(env, "no_episode_timeout") and (not bool(getattr(env, "no_episode_timeout", False))):
                ep_len_snapshot = info.get("episode_length", None) if isinstance(info, dict) else None
                if torch.is_tensor(ep_len_snapshot):
                    timeout_mask = ep_len_snapshot.to(device=dones.device) >= int(getattr(env, "max_episode_length", 0))
                    timeout_mask &= dones

            other_done_mask = dones & (~done_during) & (~success_mask) & (~timeout_mask)
            if dones.any():
                stack_reset_mask = dones.clone()
                track_done = bool(dones[track_env_idx].item())
                if track_done:
                    # Avoid post-reset displacement spikes in diagnostics.
                    prev_track_pos_world = None
                    prev_track_yaw = None
                    axis_disp_world_sum[:] = 0.0
                    axis_disp_body_sum[:] = 0.0
                    axis_disp_count = 0
                if args.show_reset_reason or args.debug_cmd or debug:
                    done_n = int(dones.sum().item())
                    phys_n = int(done_during.sum().item())
                    succ_n = int(success_mask.sum().item())
                    tout_n = int(timeout_mask.sum().item())
                    other_n = int(other_done_mask.sum().item())
                    track_reason = "none"
                    if track_done:
                        if bool(done_during[track_env_idx].item()):
                            track_reason = "done_during(physics)"
                        elif bool(success_mask[track_env_idx].item()):
                            track_reason = "success"
                        elif bool(timeout_mask[track_env_idx].item()):
                            track_reason = "timeout"
                        else:
                            track_reason = "other"
                    print(
                        "[PlayHigh][reset] step={} done={} reason(physics/success/timeout/other)={}/{}/{}/{} track_env_reason={}".format(
                            step_idx, done_n, phys_n, succ_n, tout_n, other_n, track_reason
                        )
                    )

            step_dx = 0.0
            step_dy = 0.0
            step_body_x = 0.0
            step_body_y = 0.0
            step_yaw = 0.0
            cmd_omega_track = 0.0
            if hasattr(env.env, "root_states"):
                root = env.env.root_states[track_env_idx]
                pos_xy = root[:2].detach().cpu().numpy()
                quat = root[3:7].detach().cpu().numpy()
                x_q, y_q, z_q, w_q = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                yaw_now = math.atan2(2.0 * (w_q * z_q + x_q * y_q), 1.0 - 2.0 * (y_q * y_q + z_q * z_q))
                if hasattr(env.env, "commands"):
                    cmd_omega_track = float(env.env.commands[track_env_idx, 2].detach().cpu().item())
                else:
                    cmd_omega_track = float(cmd[track_env_idx, 2].detach().cpu().item())
                if prev_track_pos_world is not None and prev_track_yaw is not None:
                    step_dx = float(pos_xy[0] - prev_track_pos_world[0])
                    step_dy = float(pos_xy[1] - prev_track_pos_world[1])
                    cos_h = math.cos(prev_track_yaw)
                    sin_h = math.sin(prev_track_yaw)
                    step_body_x = cos_h * step_dx + sin_h * step_dy
                    step_body_y = -sin_h * step_dx + cos_h * step_dy
                    step_yaw = math.atan2(math.sin(yaw_now - prev_track_yaw), math.cos(yaw_now - prev_track_yaw))
                    axis_disp_world_sum[0] += step_dx
                    axis_disp_world_sum[1] += step_dy
                    axis_disp_body_sum[0] += step_body_x
                    axis_disp_body_sum[1] += step_body_y
                    axis_disp_count += 1
                prev_track_pos_world = pos_xy
                prev_track_yaw = yaw_now

            env_idx = 0
            goal = obs["goal"][env_idx].detach().cpu().numpy()
            goal_dist = float(np.linalg.norm(goal))
            progress = 0.0 if prev_dist is None else float(prev_dist - goal_dist)
            prev_dist = goal_dist

            if args.debug_cmd and step_idx % args.debug_interval == 0:
                env_idx = 0
                cmd_pred = cmd[env_idx].detach().cpu().numpy()
                cmd_exec = None
                if hasattr(env.env, "commands"):
                    cmd_exec = env.env.commands[env_idx, :3].detach().cpu().numpy()
                cmd_show = cmd_exec if cmd_exec is not None else cmd_pred
                cmd_str = "None" if cmd_show is None else np.array2string(cmd_show, precision=3, floatmode="fixed")
                cmd_speed = 0.0 if cmd_show is None else float(np.linalg.norm(cmd_show[:2]))
                reward_total = float(rewards[env_idx].detach().cpu()) if rewards is not None else 0.0
                reward_terms = info.get("reward_terms") if info is not None else None
                reward_approach = 0.0
                reward_heading = 0.0
                reward_time = 0.0
                reward_gate = 0.0
                reward_risk = 0.0
                clearance = 0.0
                risk_scale = 0.0
                gate_val = 0.0
                passable_gate = 0.0
                passable_align = 0.0
                passable_occ_ratio = 0.0
                crossable_gate = 0.0
                crossable_align = 0.0
                crossable_width = 0.0
                if reward_terms is not None:
                    reward_approach = float(reward_terms.get("approach", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    reward_heading = float(reward_terms.get("heading", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    reward_time = float(reward_terms.get("time", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    reward_gate = float(reward_terms.get("gate_smooth", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    reward_risk = float(reward_terms.get("risk_barrier", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    clearance = float(reward_terms.get("clearance", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    risk_scale = float(reward_terms.get("risk_scale", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    passable_gate = float(reward_terms.get("passable_gate", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    passable_align = float(reward_terms.get("passable_align", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    passable_occ_ratio = float(reward_terms.get("passable_occ_ratio", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    crossable_gate = float(reward_terms.get("crossable_gate", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    crossable_align = float(reward_terms.get("crossable_align", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    crossable_width = float(reward_terms.get("crossable_width", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                if is_gate and gate_y is not None:
                    gate_val = float(gate_y[env_idx].detach().cpu())
                yaw_raw = 0.0
                yaw_policy = 0.0
                heading_err_pos = 0.0
                heading_err_neg = 0.0
                bearing_y = 0.0
                goal_raw_dbg = None
                goal_raw_bear_xy = 0.0
                goal_raw_bear_y = 0.0
                goal_world_bear_xy = 0.0
                goal_world_bear_y = 0.0
                if hasattr(env.env, "root_states"):
                    quat = env.env.root_states[env_idx, 3:7].detach().cpu().numpy()
                    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                    yaw_raw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    yaw_policy = math.atan2(math.sin(yaw_raw + heading_offset), math.cos(yaw_raw + heading_offset))
                    if hasattr(env.env, "goal_world"):
                        pos = env.env.root_states[env_idx, :2].detach().cpu().numpy()
                        goal_w = env.env.goal_world[env_idx].detach().cpu().numpy()
                        goal_dir = math.atan2(goal_w[1] - pos[1], goal_w[0] - pos[0])
                        heading_err_pos = math.atan2(math.sin((yaw_raw + 0.5 * math.pi) - goal_dir),
                                                    math.cos((yaw_raw + 0.5 * math.pi) - goal_dir))
                        heading_err_neg = math.atan2(math.sin((yaw_raw - 0.5 * math.pi) - goal_dir),
                                                    math.cos((yaw_raw - 0.5 * math.pi) - goal_dir))
                        delta_x = goal_w[0] - pos[0]
                        delta_y = goal_w[1] - pos[1]
                        rel_x = math.cos(yaw_raw) * delta_x + math.sin(yaw_raw) * delta_y
                        rel_y = -math.sin(yaw_raw) * delta_x + math.cos(yaw_raw) * delta_y
                        goal_world_bear_xy = math.atan2(rel_y, rel_x)
                        goal_world_bear_y = math.atan2(rel_x, rel_y)
                if goal is not None:
                    bearing_y = math.atan2(goal[0], goal[1])
                if hasattr(env.env, "goal_buf"):
                    goal_raw_dbg = env.env.goal_buf[env_idx].detach().cpu().numpy()
                    goal_raw_bear_xy = math.atan2(goal_raw_dbg[1], goal_raw_dbg[0])
                    goal_raw_bear_y = math.atan2(goal_raw_dbg[0], goal_raw_dbg[1])
                goal_bearing = bearing_y
                pass_bearing = 0.0
                cross_bearing = 0.0
                pass_goal_err = 0.0
                cross_goal_err = 0.0
                pass_dir_norm = 0.0
                cross_dir_norm = 0.0
                pass_vis_mean = 0.0
                pass_sector_mean = 0.0
                low_vis_mean = 0.0
                low_sector_mean = 0.0
                vis_ratio = 0.0
                sector_vis_ratio = 0.0
                low_block_ratio = 0.0
                pass_out_sector = 0
                pass_dir_dbg = None
                cross_dir_dbg = None
                pass_gate_dbg = 0.0
                pass_occ_dbg = 0.0
                cross_gate_dbg = 0.0
                cross_width_dbg = 0.0
                debug_aff = None
                if obs is not None:
                    debug_aff = raw_aff_map
                if debug_aff is not None:
                    cross_dir, cross_gate_dbg, cross_width_dbg, low_block_mask = env._compute_low_obstacle_guidance(
                        debug_aff
                    )
                    debug_goal = obs["goal"]
                    pass_dir, pass_gate_dbg, pass_occ_dbg = env._compute_passable_guidance(
                        debug_aff,
                        debug_goal,
                        block_mask=low_block_mask,
                    )
                    pass_gate_dbg = float(pass_gate_dbg[env_idx].detach().cpu())
                    pass_occ_dbg = float(pass_occ_dbg[env_idx].detach().cpu())
                    cross_gate_dbg = float(cross_gate_dbg[env_idx].detach().cpu())
                    cross_width_dbg = float(cross_width_dbg[env_idx].detach().cpu())
                    pass_dir_dbg = pass_dir[env_idx].detach().cpu().numpy()
                    cross_dir_dbg = cross_dir[env_idx].detach().cpu().numpy()
                    pass_dir_norm = float(torch.norm(pass_dir[env_idx]).detach().cpu())
                    cross_dir_norm = float(torch.norm(cross_dir[env_idx]).detach().cpu())
                    if pass_dir_norm > 1e-6:
                        pass_bearing = math.atan2(pass_dir_dbg[0], pass_dir_dbg[1])
                    if cross_dir_norm > 1e-6:
                        cross_bearing = math.atan2(cross_dir_dbg[0], cross_dir_dbg[1])

                    def _angle_diff(a, b):
                        return math.atan2(math.sin(a - b), math.cos(a - b))

                    pass_goal_err = _angle_diff(pass_bearing, goal_bearing)
                    cross_goal_err = _angle_diff(cross_bearing, goal_bearing)

                    passable = debug_aff[env_idx, 1]
                    low_obs = debug_aff[env_idx, 2]
                    visible = env.affordance_visible_mask
                    if visible is None:
                        visible = torch.ones_like(passable, dtype=torch.bool)
                    if visible.device != passable.device:
                        visible = visible.to(passable.device)
                    vis_count = visible.float().sum().clamp_min(1.0)
                    vis_ratio = float((vis_count / float(visible.numel())).detach().cpu())
                    visible_f = visible.float()
                    pass_vis_mean = float((passable * visible_f).sum().div(vis_count).detach().cpu())
                    low_vis_mean = float((low_obs * visible_f).sum().div(vis_count).detach().cpu())

                    sector_deg = 0.0
                    if env.reward_cfg is not None:
                        sector_deg = float(getattr(env.reward_cfg, "passable_sector_deg", 0.0))
                    sector_half = math.radians(sector_deg) * 0.5 if sector_deg > 0.0 else 0.0
                    if sector_half > 0.0:
                        bearing_map = env.affordance_bearing_map
                        if bearing_map.device != passable.device:
                            bearing_map = bearing_map.to(passable.device)
                        angle = torch.atan2(
                            torch.sin(bearing_map - goal_bearing),
                            torch.cos(bearing_map - goal_bearing),
                        )
                        sector_mask = (torch.abs(angle) <= sector_half) & visible
                    else:
                        sector_mask = visible
                    sector_count = sector_mask.float().sum().clamp_min(1.0)
                    sector_vis_ratio = float((sector_count / vis_count).detach().cpu())
                    sector_f = sector_mask.float()
                    pass_sector_mean = float((passable * sector_f).sum().div(sector_count).detach().cpu())
                    low_sector_mean = float((low_obs * sector_f).sum().div(sector_count).detach().cpu())
                    if low_block_mask is not None:
                        low_block_ratio = float(low_block_mask[env_idx].mean().detach().cpu())
                    if sector_half > 0.0:
                        pass_out_sector = int(abs(pass_goal_err) > sector_half)
                aff_delta = 0.0
                aff_std = 0.0
                aff_filled = float(aff_stack_fill[env_idx].item()) / max(aff_stack, 1)
                if aff_stack > 1:
                    base_channels = aff_map.shape[1]
                    stack_h, stack_w = aff_map.shape[2], aff_map.shape[3]
                    stack = aff_stack_buf[env_idx].reshape(aff_stack, base_channels, stack_h, stack_w)
                    aff_delta = (stack[1:] - stack[:-1]).abs().mean().item()
                    aff_std = stack.std(dim=0, unbiased=False).mean().item()
                print(
                    "[PlayHigh] step={} |cmd_xy|={:.3f} progress={:.3f} gate_y={:.3f} reward={:.3f} (approach={:.3f}, heading={:.3f}, time={:.3f}, gate={:.3f}, risk={:.3f}) passable(g/a/o)={:.3f}/{:.3f}/{:.3f} crossable(g/a/w)={:.3f}/{:.3f}/{:.3f} clr={:.3f} risk_scale={:.3f} aff_stack(d/std/fill)={:.3f}/{:.3f}/{:.3f} cmd_pred={} goal={} dist={:.3f} cmd_exec={} yaw_raw={:.3f} yaw_policy={:.3f} bear_y={:.3f} herr(+pi/2)={:.3f} herr(-pi/2)={:.3f}".format(
                        step_idx,
                        cmd_speed,
                        progress,
                        gate_val,
                        reward_total,
                        reward_approach,
                        reward_heading,
                        reward_time,
                        reward_gate,
                        reward_risk,
                        passable_gate,
                        passable_align,
                        passable_occ_ratio,
                        crossable_gate,
                        crossable_align,
                        crossable_width,
                        clearance,
                        risk_scale,
                        aff_delta,
                        aff_std,
                        aff_filled,
                        np.array2string(cmd_pred, precision=3, floatmode="fixed"),
                        np.array2string(goal, precision=3, floatmode="fixed"),
                        goal_dist,
                        cmd_str,
                        yaw_raw,
                        yaw_policy,
                        bearing_y,
                        heading_err_pos,
                        heading_err_neg,
                    )
                )
                print(
                    "[PlayHigh][diag] goal_bear={:.3f} pass_bear={:.3f} cross_bear={:.3f} err_gp={:.3f} err_gc={:.3f} "
                    "pass_dir={} cross_dir={} norm_p/c={:.3f}/{:.3f} pass_gate_dbg={:.3f} pass_occ_dbg={:.3f} "
                    "cross_gate_dbg={:.3f} cross_width_dbg={:.3f} low_block_ratio={:.3f} vis_ratio={:.3f} "
                    "pass_vis/sector={:.3f}/{:.3f} low_vis/sector={:.3f}/{:.3f} sector_vis_ratio={:.3f} "
                    "pass_out_sector={}".format(
                        goal_bearing,
                        pass_bearing,
                        cross_bearing,
                        pass_goal_err,
                        cross_goal_err,
                        "None" if pass_dir_dbg is None else np.array2string(pass_dir_dbg, precision=3, floatmode="fixed"),
                        "None" if cross_dir_dbg is None else np.array2string(cross_dir_dbg, precision=3, floatmode="fixed"),
                        pass_dir_norm,
                        cross_dir_norm,
                        pass_gate_dbg,
                        pass_occ_dbg,
                        cross_gate_dbg,
                        cross_width_dbg,
                        low_block_ratio,
                        vis_ratio,
                        pass_vis_mean,
                        pass_sector_mean,
                        low_vis_mean,
                        low_sector_mean,
                        sector_vis_ratio,
                        pass_out_sector,
                    )
                )
                print(
                    "[PlayHigh][goal] raw={} rot={} bear_raw_xy={:.3f} bear_raw_y={:.3f} "
                    "bear_world_xy={:.3f} bear_world_y={:.3f} bear_policy={:.3f} offset={:.3f}".format(
                        "None" if goal_raw_dbg is None else np.array2string(goal_raw_dbg, precision=3, floatmode="fixed"),
                        "None" if goal is None else np.array2string(goal, precision=3, floatmode="fixed"),
                        goal_raw_bear_xy,
                        goal_raw_bear_y,
                        goal_world_bear_xy,
                        goal_world_bear_y,
                        bearing_y,
                        heading_offset,
                    )
                )
                if expert_cmd is not None:
                    expert_cmd_np = expert_cmd[env_idx].detach().cpu().numpy()
                    print(
                        "[PlayHigh][expert] cmd_expert={} cmd_pred={} cmd_exec={}".format(
                            np.array2string(expert_cmd_np, precision=3, floatmode="fixed"),
                            np.array2string(cmd_pred, precision=3, floatmode="fixed"),
                            cmd_str,
                        )
                    )
                    if (
                        dircheck_alpha_pre is not None
                        and dircheck_x_pre is not None
                        and dircheck_y_pre is not None
                        and dircheck_omega_pre is not None
                    ):
                        a_pre = float(dircheck_alpha_pre[env_idx].detach().cpu().item())
                        x_pre = float(dircheck_x_pre[env_idx].detach().cpu().item())
                        y_pre = float(dircheck_y_pre[env_idx].detach().cpu().item())
                        w_pre = float(dircheck_omega_pre[env_idx].detach().cpu().item())
                        eps = 1e-6
                        sign_match = int(
                            (abs(a_pre) <= eps and abs(w_pre) <= eps)
                            or (a_pre * w_pre < 0.0)
                        )
                        print(
                            "[PlayHigh][dircheck] pre_body(x_right,y_forward)=({:.3f},{:.3f}) pre_alpha={:.3f} pre_omega={:.3f} sign_match={}".format(
                                x_pre,
                                y_pre,
                                a_pre,
                                w_pre,
                                sign_match,
                            )
                        )
                        if (
                            dircheck_goal_x_pre is not None
                            and dircheck_goal_y_pre is not None
                            and dircheck_goal_err_pre is not None
                        ):
                            gx_pre = float(dircheck_goal_x_pre[env_idx].detach().cpu().item())
                            gy_pre = float(dircheck_goal_y_pre[env_idx].detach().cpu().item())
                            ge_pre = float(dircheck_goal_err_pre[env_idx].detach().cpu().item())
                            print(
                                "[PlayHigh][dircheck] pre_goal_buf(x_right,y_forward)=({:.3f},{:.3f}) pre_goal_err={:.6f}".format(
                                    gx_pre,
                                    gy_pre,
                                    ge_pre,
                                )
                            )
                mean_dx = axis_disp_world_sum[0] / max(axis_disp_count, 1)
                mean_dy = axis_disp_world_sum[1] / max(axis_disp_count, 1)
                mean_bx = axis_disp_body_sum[0] / max(axis_disp_count, 1)
                mean_by = axis_disp_body_sum[1] / max(axis_disp_count, 1)
                print(
                    "[PlayHigh][axis] force_cmd={} step_world=({:.4f},{:.4f}) step_body(x_right,y_forward)=({:.4f},{:.4f}) "
                    "step_yaw={:.4f} cmd_omega={:.4f} mean_world=({:.4f},{:.4f}) mean_body=({:.4f},{:.4f}) samples={}".format(
                        "None" if force_cmd_tensor is None else np.array2string(force_cmd_tensor[0].detach().cpu().numpy(), precision=3, floatmode="fixed"),
                        step_dx,
                        step_dy,
                        step_body_x,
                        step_body_y,
                        step_yaw,
                        cmd_omega_track,
                        mean_dx,
                        mean_dy,
                        mean_bx,
                        mean_by,
                        axis_disp_count,
                    )
                )

            if not args.headless:
                env.env.render()

            if args.camera_enable and (step_idx % args.camera_interval == 0):
                depth_np = None
                if hasattr(env.env, "_get_depth_images"):
                    depth = env.env._get_depth_images()
                    depth_np = depth[args.camera_env].detach().cpu().numpy()
                elif hasattr(env.env, "depth_images"):
                    depth = env.env.depth_images
                    depth_np = depth[args.camera_env, 0].detach().cpu().numpy()

                if depth_np is not None:
                    depth_min = float(depth_np.min())
                    depth_max = float(depth_np.max())
                    depth_norm = (depth_np - depth_min) / (max(depth_max - depth_min, 1e-6))
                    depth_u8 = (depth_norm * 255.0).astype("uint8")

                    if args.camera_show and camera_cv2 is not None:
                        depth_vis = camera_cv2.applyColorMap(255 - depth_u8, camera_cv2.COLORMAP_TURBO)
                        camera_cv2.imshow("play_highlevel_depth", depth_vis)
                        camera_cv2.waitKey(1)

                    if args.camera_save:
                        npy_path = os.path.join(args.camera_dir, f"depth_{camera_frame_idx:06d}.npy")
                        png_path = os.path.join(args.camera_dir, f"depth_{camera_frame_idx:06d}.png")
                        np.save(npy_path, depth_np.astype("float32"))
                        if camera_cv2 is not None:
                            depth_vis = camera_cv2.applyColorMap(255 - depth_u8, camera_cv2.COLORMAP_TURBO)
                            camera_cv2.imwrite(png_path, depth_vis)
                        elif not camera_warned_no_cv2:
                            print("[PlayHigh] ⚠ cv2 unavailable; skipping depth PNG output.")
                            camera_warned_no_cv2 = True
                        camera_frame_idx += 1

            step_idx += 1
            if args.max_steps > 0 and step_idx >= args.max_steps:
                stop_reason = "max_steps"
                break

    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("[PlayHigh] keyboard interrupt received; exporting current metrics.")
    finally:
        if e_s_metrics.get("enabled", False):
            _finalize_active_e_s_metrics(e_s_metrics, reason=stop_reason)
            _export_e_s_metrics(e_s_metrics, final=True, stop_reason=stop_reason)


if __name__ == "__main__":
    main()
