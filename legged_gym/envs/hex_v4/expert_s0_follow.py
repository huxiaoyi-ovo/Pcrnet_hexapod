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


def _world_to_body_xy(
    vec_world_xy: torch.Tensor,
    *,
    cos_heading: torch.Tensor,
    sin_heading: torch.Tensor,
) -> torch.Tensor:
    body_x = cos_heading * vec_world_xy[:, 0] - sin_heading * vec_world_xy[:, 1]
    body_y = sin_heading * vec_world_xy[:, 0] + cos_heading * vec_world_xy[:, 1]
    return torch.stack([body_x, body_y], dim=1)


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
    kp_perp: float = 0.5,
    kff: float = 1.0,
    v_along_max: float = 1.5,
    v_perp_max: float = 0.12,
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

    cos_heading = torch.cos(robot_heading)
    sin_heading = torch.sin(robot_heading)

    # Keep strict observation alignment for S0: only use robot_pos/robot_heading/target_pos.
    # Reject hidden-info inputs to avoid "passed but silently ignored" confusion.
    if target_vel_world_xy is not None:
        raise ValueError("S0 rewritten expert does not use target_vel_world_xy; pass None")
    if target_heading is not None:
        raise ValueError("S0 rewritten expert does not use target_heading; pass None")
    _ = v_eps
    _ = kff

    to_target_world = target_world_xy - robot_pos_world_xy
    to_target_body = _world_to_body_xy(to_target_world, cos_heading=cos_heading, sin_heading=sin_heading)

    dist = torch.norm(to_target_world, dim=1).clamp_min(1e-6)
    dist_err = dist - float(d_des)

    # Longitudinal control: distance regulation (optionally keep the kff hook for future feedforward).
    v_forward = float(kp_along) * dist_err
    v_forward = torch.clamp(v_forward, min=-float(v_along_max), max=float(v_along_max))

    # Lateral control: project robot-relative position onto body lateral axis.
    robot_rel_target_world = robot_pos_world_xy - target_world_xy
    robot_rel_target_body = _world_to_body_xy(robot_rel_target_world, cos_heading=cos_heading, sin_heading=sin_heading)
    lateral_error = robot_rel_target_body[:, 0]
    v_lateral = -float(kp_perp) * lateral_error
    v_lateral = torch.clamp(v_lateral, min=-float(v_perp_max), max=float(v_perp_max))

    # Yaw control: align heading to current target direction.
    heading_error = torch.atan2(to_target_body[:, 0], to_target_body[:, 1])
    omega = -float(k_yaw) * heading_error
    omega = torch.clamp(omega, min=-float(w_max), max=float(w_max))

    forward_axis_body = torch.zeros((num_envs, 2), device=device, dtype=dtype)
    forward_axis_body[:, 1] = 1.0
    lateral_axis_body = torch.zeros((num_envs, 2), device=device, dtype=dtype)
    lateral_axis_body[:, 0] = 1.0
    cmd_xy = v_forward.unsqueeze(1) * forward_axis_body + v_lateral.unsqueeze(1) * lateral_axis_body

    cmd = torch.stack([cmd_xy[:, 0], cmd_xy[:, 1], omega], dim=1)
    scale_cfg = torch.abs(_as_cmd_scale(cmd_scale, device=device, dtype=dtype))
    scale_hard = torch.full((1, 3), 1.4, device=device, dtype=dtype)
    scale = torch.minimum(scale_cfg, scale_hard)
    cmd = torch.clamp(cmd, min=-scale, max=scale)
    return cmd


__all__ = ["compute_s0_follow_expert_cmd"]
