"""Shared observed local-map and Avoid actor-state helpers."""

import math

import torch
import torch.nn.functional as F


AVOID_ACTOR_MASK_INDICES = (0, 1, 2, 6, 9)


def _batch_grid(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W) or (H,W), got {tuple(tensor.shape)}")
    return tensor


def canonical_local_map(
    occupancy: torch.Tensor,
    visible_mask: torch.Tensor,
    extent: float = 3.0,
    clearance: float = 0.27,
) -> torch.Tensor:
    """Return observed [occupancy, soft-passability] on an ij [x_right,y_forward] grid."""
    occupancy = _batch_grid(occupancy, "occupancy")
    visible_mask = _batch_grid(visible_mask, "visible_mask")
    if occupancy.shape != visible_mask.shape:
        raise ValueError("occupancy and visible_mask must have the same shape")
    if extent <= 0.0 or clearance < 0.0:
        raise ValueError("extent must be positive and clearance must be non-negative")

    occ_raw = torch.clamp(torch.nan_to_num(occupancy, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    visible = (visible_mask > 0.5).to(dtype=occ_raw.dtype)
    occ = occ_raw * visible
    cell = float(extent) / float(occ_raw.shape[-1])
    radius_cells = int(math.ceil(float(clearance) / cell))
    if radius_cells > 0:
        kernel = 2 * radius_cells + 1
        pooled = F.max_pool2d(occ.unsqueeze(1), kernel, stride=1, padding=radius_cells).squeeze(1)
    else:
        pooled = occ
    p = ((pooled <= 0.5) & (occ < 0.5)).to(dtype=occ.dtype)
    soft = torch.maximum(
        F.avg_pool2d(p.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1),
        p,
    ) * (1.0 - occ)
    soft = soft * visible
    return torch.stack([occ * visible, soft], dim=1)


def canonical_difficulty(local_map: torch.Tensor, extent: float = 3.0, radius: float = 2.0) -> torch.Tensor:
    """Compute the legacy-sim scalar D from the observed two-channel map."""
    if local_map.ndim != 4 or local_map.shape[1] < 2:
        raise ValueError("local_map must have shape (B,2,H,W)")
    if extent <= 0.0 or radius <= 0.0:
        raise ValueError("extent and radius must be positive")
    _, _, height, width = local_map.shape
    cell_x, cell_y = float(extent) / float(height), float(extent) / float(width)
    x = torch.linspace(-0.5 * extent + 0.5 * cell_x, 0.5 * extent - 0.5 * cell_x, height, device=local_map.device, dtype=local_map.dtype)
    y = torch.linspace(0.5 * cell_y, extent - 0.5 * cell_y, width, device=local_map.device, dtype=local_map.dtype)
    grid_x, grid_y = torch.meshgrid(x, y)
    region = ((grid_x.square() + grid_y.square()) <= radius * radius).to(dtype=local_map.dtype)
    denom = region.sum().clamp_min(1.0)
    occ = torch.clamp(torch.nan_to_num(local_map[:, 0], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    soft = torch.clamp(torch.nan_to_num(local_map[:, 1], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return 0.5 * (occ * region).sum(dim=(1, 2)) / denom + 0.5 * ((1.0 - soft) * region).sum(dim=(1, 2)) / denom


def build_avoid_actor_state(raw_state: torch.Tensor, v_drive_nom) -> torch.Tensor:
    """Build the frozen 14-D Avoid actor state without positional leakage."""
    if raw_state.ndim != 2 or raw_state.shape[1] not in (13, 14):
        raise ValueError("raw_state must have shape (B,13) or (B,14)")
    drive = torch.as_tensor(v_drive_nom, device=raw_state.device, dtype=raw_state.dtype)
    if drive.ndim == 0:
        drive = drive.expand(raw_state.shape[0])
    drive = drive.reshape(-1)
    if drive.numel() != raw_state.shape[0]:
        raise ValueError("v_drive_nom must be scalar or have one value per batch item")
    if raw_state.shape[1] == 13:
        state = torch.cat([raw_state, drive.unsqueeze(-1)], dim=-1)
    else:
        state = raw_state.clone()
    state[:, 13] = drive
    state[:, list(AVOID_ACTOR_MASK_INDICES)] = 0.0
    return state


def side_preference(goal: torch.Tensor, valid=None, min_distance: float = 1e-3) -> torch.Tensor:
    """Return [target_x / distance, 0] only for valid forward goals."""
    if goal.ndim != 2 or goal.shape[1] != 2:
        raise ValueError("goal must have shape (B,2) as [x_right,y_forward]")
    if min_distance <= 0.0:
        raise ValueError("min_distance must be positive")
    if not torch.isfinite(goal).all():
        raise ValueError("goal must be finite")
    distance = torch.norm(goal, p=2, dim=-1)
    is_valid = (goal[:, 1] > 0.0) & (distance > float(min_distance))
    if valid is not None:
        is_valid = is_valid & torch.as_tensor(valid, device=goal.device, dtype=torch.bool).reshape(-1)
    side = torch.where(is_valid, goal[:, 0] / distance.clamp_min(float(min_distance)), torch.zeros_like(distance))
    return torch.stack([side, torch.zeros_like(side)], dim=-1)
