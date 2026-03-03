from typing import Optional, Sequence, Union

import torch

_HEADING_LOCK_STATE = {}


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
    # Keep exact project contract used by goal_buf in hex_ground:
    # heading=0 => world +Y is forward, +X is right.
    # world -> body must use R(-heading) under this contract.
    body_x = cos_heading * vec_world_xy[:, 0] + sin_heading * vec_world_xy[:, 1]
    body_y = -sin_heading * vec_world_xy[:, 0] + cos_heading * vec_world_xy[:, 1]
    return torch.stack([body_x, body_y], dim=1)


def _heading_lock_key(device: torch.device, num_envs: int):
    return str(device), int(num_envs)


def _get_heading_lock_state(device: torch.device, num_envs: int) -> torch.Tensor:
    key = _heading_lock_key(device, num_envs)
    state = _HEADING_LOCK_STATE.get(key, None)
    if state is None or state.shape[0] != num_envs or state.device != device:
        state = torch.zeros(num_envs, device=device, dtype=torch.bool)
    _HEADING_LOCK_STATE[key] = state
    return state


def reset_s0_follow_expert_state():
    _HEADING_LOCK_STATE.clear()


def compute_s0_follow_expert_cmd(
    robot_pos_world_xy: torch.Tensor,
    robot_heading: torch.Tensor,
    target_world_xy: torch.Tensor,
    target_vel_world_xy: Optional[torch.Tensor],
    target_heading: Optional[torch.Tensor],
    cmd_scale: Union[Sequence[float], torch.Tensor],
    reset_mask: Optional[torch.Tensor] = None,
    *,
    d_des: float = 1.0,
    v_eps: float = 0.05,
    kp_along: float = 0.8,
    kp_perp: float = 0.5,
    kff: float = 1.0,
    v_along_max: float = 1.1,
    v_perp_max: float = 0.12,
    k_yaw: float = 1.4,
    w_max: float = 1.2,
    heading_lock_rad: float = 0.35,
    heading_release_rad: float = 0.18,
    allow_backward_bearing_rad: float = 0.30,
    omega_deadzone_rad: float = 0.0,
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
    _ = kp_perp
    _ = v_perp_max

    to_target_world = target_world_xy - robot_pos_world_xy
    to_target_body = _world_to_body_xy(to_target_world, cos_heading=cos_heading, sin_heading=sin_heading)

    # Unicycle-style control with turn-first hysteresis:
    # - alpha: bearing in body frame (x_right, y_forward)
    # - omega: always active for heading alignment
    # - v_forward: distance tracking with cos(alpha), but forced to zero while turning lock is on
    dist = torch.norm(to_target_body, dim=1).clamp_min(1e-6)
    dist_error = dist - float(d_des)
    heading_error = torch.atan2(to_target_body[:, 0], to_target_body[:, 1])
    cos_alpha = torch.cos(heading_error)
    abs_bearing = torch.abs(heading_error)

    # Project runtime validation: +omega is CCW(left turn).
    # With alpha=atan2(x_right, y_forward), turn-toward-target requires omega = -k*alpha.
    omega = -float(k_yaw) * heading_error
    omega = torch.where(
        torch.abs(heading_error) < float(max(0.0, omega_deadzone_rad)),
        torch.zeros_like(omega),
        omega,
    )
    omega = torch.clamp(omega, min=-float(w_max), max=float(w_max))

    lock_th = float(max(1e-3, heading_lock_rad))
    release_th = float(max(1e-3, min(heading_release_rad, lock_th - 1e-3)))
    lock_state = _get_heading_lock_state(device=device, num_envs=num_envs)
    if reset_mask is not None:
        if reset_mask.ndim == 1 and reset_mask.dtype == torch.bool and reset_mask.shape[0] == num_envs:
            lock_state[reset_mask.to(device=device)] = False
        elif reset_mask.ndim == 1 and reset_mask.dtype in (torch.int32, torch.int64):
            lock_state[reset_mask.to(device=device, dtype=torch.long)] = False
        else:
            raise ValueError("reset_mask must be bool mask (N,) or index tensor (K,)")
    enter_lock = abs_bearing > lock_th
    exit_lock = abs_bearing < release_th
    lock_state = torch.where(
        enter_lock,
        torch.ones_like(lock_state, dtype=torch.bool),
        torch.where(exit_lock, torch.zeros_like(lock_state, dtype=torch.bool), lock_state),
    )
    _HEADING_LOCK_STATE[_heading_lock_key(device, num_envs)] = lock_state

    v_forward = float(kp_along) * dist_error * cos_alpha
    allow_backward = abs_bearing < float(max(0.0, allow_backward_bearing_rad))
    v_forward = torch.where((v_forward < 0.0) & (~allow_backward), torch.zeros_like(v_forward), v_forward)
    v_forward = torch.where(lock_state, torch.zeros_like(v_forward), v_forward)
    v_forward = torch.clamp(v_forward, min=-float(v_along_max), max=float(v_along_max))

    # Expert output contract for this debug phase: no lateral command.
    cmd_x = torch.zeros(num_envs, device=device, dtype=dtype)
    cmd_y = v_forward

    cmd = torch.stack([cmd_x, cmd_y, omega], dim=1)
    scale_cfg = torch.abs(_as_cmd_scale(cmd_scale, device=device, dtype=dtype))
    scale_hard = torch.full((1, 3), 1.4, device=device, dtype=dtype)
    scale = torch.minimum(scale_cfg, scale_hard)
    cmd = torch.clamp(cmd, min=-scale, max=scale)
    return cmd


__all__ = ["compute_s0_follow_expert_cmd", "reset_s0_follow_expert_state"]
