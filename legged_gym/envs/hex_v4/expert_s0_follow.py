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
    kp_perp: float = 0.6,
    kff: float = 0.0,
    v_along_max: float = 0.8,
    v_perp_max: float = 0.15,
    k_yaw: float = 1.0,
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

    if target_vel_world_xy is None:
        target_vel_world_xy = torch.zeros(num_envs, 2, device=device, dtype=dtype)
    if target_heading is None:
        target_heading = torch.zeros(num_envs, device=device, dtype=dtype)

    if target_vel_world_xy.ndim != 2 or target_vel_world_xy.shape != (num_envs, 2):
        raise ValueError("target_vel_world_xy must have shape (N, 2)")
    if target_heading.ndim != 1 or target_heading.shape[0] != num_envs:
        raise ValueError("target_heading must have shape (N,)")

    target_speed = torch.norm(target_vel_world_xy, dim=1)
    vel_dir_world = target_vel_world_xy / target_speed.unsqueeze(1).clamp_min(1e-6)
    heading_dir_world = torch.stack([torch.sin(target_heading), torch.cos(target_heading)], dim=1)
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

    err_along = torch.sum(err_body * dir_body, dim=1)
    err_perp = torch.sum(err_body * perp_body, dim=1)

    v_along = float(kp_along) * err_along + float(kff) * target_speed
    # Allow mild reverse correction when the robot overshoots the desired trailing point.
    v_along = torch.clamp(v_along, min=-0.2, max=float(v_along_max))
    v_perp = torch.clamp(float(kp_perp) * err_perp, min=-float(v_perp_max), max=float(v_perp_max))

    cmd_xy = v_along.unsqueeze(1) * dir_body + v_perp.unsqueeze(1) * perp_body

    dir_body_angle = torch.atan2(dir_body[:, 0], dir_body[:, 1])
    omega = torch.clamp(float(k_yaw) * dir_body_angle, min=-float(w_max), max=float(w_max))

    cmd = torch.stack([cmd_xy[:, 0], cmd_xy[:, 1], omega], dim=1)
    scale = _as_cmd_scale(cmd_scale, device=device, dtype=dtype)
    cmd = torch.clamp(cmd, min=-scale, max=scale)
    return cmd


__all__ = ["compute_s0_follow_expert_cmd"]
