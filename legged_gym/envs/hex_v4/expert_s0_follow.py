import math
from typing import Optional, Sequence, Union

import torch


def _as_cmd_scale(
    cmd_scale: Union[Sequence[float], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(cmd_scale):
        scale = cmd_scale.to(device=device, dtype=dtype)
    else:
        scale = torch.tensor(cmd_scale, device=device, dtype=dtype)
    if scale.numel() != 3:
        raise ValueError(f"cmd_scale must have 3 elements, got shape={tuple(scale.shape)}")
    return scale.view(1, 3)


def compute_s0_follow_expert_cmd(
    robot_pos_world_xy: torch.Tensor,
    robot_heading: torch.Tensor,
    target_world_xy: torch.Tensor,
    target_vel_world_xy: Optional[torch.Tensor],
    target_heading: Optional[torch.Tensor],
    cmd_scale: Union[Sequence[float], torch.Tensor],
    *,
    d_des: float = 1.0,
    v_eps: float = 0.05,
    kp_along: float = 0.8,
    kp_perp: float = 0.35,
    kff: float = 1.0,
    v_along_max: float = 1.5,
    v_perp_max: float = 0.12,
    k_yaw: float = 1.2,
    w_max: float = 0.8,
) -> torch.Tensor:
    if robot_pos_world_xy.ndim != 2 or robot_pos_world_xy.shape[1] != 2:
        raise ValueError("robot_pos_world_xy must have shape (N, 2)")
    if target_world_xy.ndim != 2 or target_world_xy.shape[1] != 2:
        raise ValueError("target_world_xy must have shape (N, 2)")
    if robot_heading.ndim != 1 or robot_heading.shape[0] != robot_pos_world_xy.shape[0]:
        raise ValueError("robot_heading must have shape (N,)")

    num_envs = robot_pos_world_xy.shape[0]
    device = robot_pos_world_xy.device
    dtype = robot_pos_world_xy.dtype

    if target_vel_world_xy is not None:
        if target_vel_world_xy.ndim != 2 or target_vel_world_xy.shape != (num_envs, 2):
            raise ValueError("target_vel_world_xy must have shape (N, 2)")
        target_vel_world = target_vel_world_xy.to(device=device, dtype=dtype)
    else:
        target_vel_world = torch.zeros(num_envs, 2, device=device, dtype=dtype)

    if target_heading is not None:
        if target_heading.ndim != 1 or target_heading.shape[0] != num_envs:
            raise ValueError("target_heading must have shape (N,)")
        target_heading_world = target_heading.to(device=device, dtype=dtype)
        heading_dir_world = torch.stack([torch.sin(target_heading_world), torch.cos(target_heading_world)], dim=1)
    else:
        # Fallback for observation-only labels: use geometric target direction in world frame.
        to_target_world = target_world_xy - robot_pos_world_xy
        heading_dir_world = to_target_world / torch.norm(to_target_world, dim=1, keepdim=True).clamp_min(1e-6)
        # Degenerate case: robot and target overlap.
        default_dir = torch.zeros_like(heading_dir_world)
        default_dir[:, 1] = 1.0
        heading_dir_world = torch.where(
            torch.norm(to_target_world, dim=1, keepdim=True) > 1e-6,
            heading_dir_world,
            default_dir,
        )

    target_speed = torch.norm(target_vel_world, dim=1)
    vel_dir_world = target_vel_world / target_speed.unsqueeze(1).clamp_min(1e-6)
    use_vel = target_speed > float(v_eps)
    dir_world = torch.where(use_vel.unsqueeze(1), vel_dir_world, heading_dir_world)
    dir_world = dir_world / torch.norm(dir_world, dim=1, keepdim=True).clamp_min(1e-6)

    shadow_world = target_world_xy - float(d_des) * dir_world
    err_world = shadow_world - robot_pos_world_xy

    cos_heading = torch.cos(robot_heading)
    sin_heading = torch.sin(robot_heading)

    err_body_x = cos_heading * err_world[:, 0] - sin_heading * err_world[:, 1]
    err_body_y = sin_heading * err_world[:, 0] + cos_heading * err_world[:, 1]
    err_body = torch.stack([err_body_x, err_body_y], dim=1)

    dir_body_x = cos_heading * dir_world[:, 0] - sin_heading * dir_world[:, 1]
    dir_body_y = sin_heading * dir_world[:, 0] + cos_heading * dir_world[:, 1]
    dir_body = torch.stack([dir_body_x, dir_body_y], dim=1)
    dir_body = dir_body / torch.norm(dir_body, dim=1, keepdim=True).clamp_min(1e-6)

    perp_body = torch.stack([-dir_body[:, 1], dir_body[:, 0]], dim=1)
    target_vel_body_x = cos_heading * target_vel_world[:, 0] - sin_heading * target_vel_world[:, 1]
    target_vel_body_y = sin_heading * target_vel_world[:, 0] + cos_heading * target_vel_world[:, 1]
    target_vel_body = torch.stack([target_vel_body_x, target_vel_body_y], dim=1)

    err_along = torch.sum(err_body * dir_body, dim=1)
    err_perp = torch.sum(err_body * perp_body, dim=1)
    v_ff_along = torch.sum(target_vel_body * dir_body, dim=1)
    v_ff_perp = torch.sum(target_vel_body * perp_body, dim=1)
    v_along = float(kp_along) * err_along + float(kff) * v_ff_along
    v_perp = float(kp_perp) * err_perp + (0.5 * float(kff)) * v_ff_perp

    dir_body_angle = torch.atan2(dir_body[:, 0], dir_body[:, 1])
    to_shadow_angle = torch.atan2(err_body[:, 0], err_body[:, 1])
    yaw_ref_angle = 0.85 * dir_body_angle + 0.15 * to_shadow_angle
    # Heading-first with catch-up release:
    # large heading error -> rotate first; large tracking error -> allow more translation.
    turn_gain = torch.clamp((torch.abs(yaw_ref_angle) - 0.12) / 0.48, min=0.0, max=1.0)
    track_err = torch.norm(err_world, dim=1)
    catchup_gain = torch.clamp((track_err - 0.25) / 0.75, min=0.0, max=1.0)
    along_floor = 0.25 + 0.45 * catchup_gain
    perp_floor = 0.08 + 0.55 * catchup_gain
    along_scale = torch.maximum(1.0 - 0.85 * turn_gain, along_floor)
    perp_scale = torch.maximum(1.0 - 0.75 * turn_gain, perp_floor)
    yaw_scale = 1.0 + 1.20 * turn_gain + 0.35 * catchup_gain
    v_along = v_along * along_scale
    v_perp = v_perp * perp_scale
    # During sharp turns, avoid reverse push that can cause circling and heading drift.
    v_along_min = -0.2 * (1.0 - turn_gain) * (1.0 - 0.6 * catchup_gain)
    v_along = torch.minimum(v_along, torch.full_like(v_along, float(v_along_max)))
    v_along = torch.maximum(v_along, v_along_min)
    v_perp_limit = float(v_perp_max) * (0.25 + 0.75 * catchup_gain) * (1.0 - 0.50 * turn_gain) + 0.02
    v_perp = torch.minimum(v_perp, v_perp_limit)
    v_perp = torch.maximum(v_perp, -v_perp_limit)
    cmd_xy = v_along.unsqueeze(1) * dir_body + v_perp.unsqueeze(1) * perp_body
    omega = torch.clamp(float(k_yaw) * yaw_scale * yaw_ref_angle, min=-float(w_max), max=float(w_max))

    cmd = torch.stack([cmd_xy[:, 0], cmd_xy[:, 1], omega], dim=1)
    scale = _as_cmd_scale(cmd_scale, device=device, dtype=dtype)
    cmd = torch.clamp(cmd, min=-scale, max=scale)
    return cmd


__all__ = ["compute_s0_follow_expert_cmd"]
