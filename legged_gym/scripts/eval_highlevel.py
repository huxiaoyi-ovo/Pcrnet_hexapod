#!/usr/bin/env python3
"""
Independent high-level evaluation pipeline for paper-grade metrics.

Design principles:
1) Frozen policy only (no training-time online stats leakage)
2) Fixed eval protocol (seed + predefined difficulty set)
3) Standard outputs for paper tables (metrics.json + metrics.csv)
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
from isaacgym import gymapi
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd as s0_follow_expert_fn
from legged_gym.scripts import play_highlevel as ph
from legged_gym.scripts import train_highlevel as th


def _to_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def _load_experiment_meta_from_ckpt(path: Optional[str], device: torch.device) -> Optional[Dict]:
    if path is None or str(path).strip() == "":
        return None
    ckpt_obj = torch.load(path, map_location=device)
    if isinstance(ckpt_obj, dict):
        meta = ckpt_obj.get("experiment_meta", None)
        if isinstance(meta, dict):
            return meta
    return None


def _count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
    return total, trainable


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return v


def _json_dumps_compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _trajectory_obstacles_json(env_impl, env_id: int) -> str:
    if (
        env_impl is None
        or not hasattr(env_impl, "s_avoid_active")
        or not hasattr(env_impl, "s_avoid_pos_world")
    ):
        return "[]"
    active = getattr(env_impl, "s_avoid_active", None)
    pos_world = getattr(env_impl, "s_avoid_pos_world", None)
    if not torch.is_tensor(active) or not torch.is_tensor(pos_world):
        return "[]"
    if env_id < 0 or env_id >= int(active.shape[0]):
        return "[]"
    cap_slots = int(getattr(env_impl, "s_avoid_capsule_slot_count", 0))
    cap_slots = max(0, min(cap_slots, int(active.shape[1])))
    box_slots = int(getattr(env_impl, "s_avoid_box_slot_count", 0))
    box_start = cap_slots
    box_end = max(box_start, min(box_start + box_slots, int(active.shape[1])))
    terrain_cfg = getattr(getattr(env_impl, "cfg", None), "terrain", None)
    radius = float(getattr(terrain_cfg, "avoid_capsule_radius", 0.15))
    box_x = float(getattr(terrain_cfg, "avoid_box_size_x", 0.4))
    box_y = float(getattr(terrain_cfg, "avoid_box_size_y", 0.4))
    quat_world = getattr(env_impl, "s_avoid_quat_world", None)
    out = []
    for slot in range(cap_slots):
        if not bool(active[env_id, slot].item()):
            continue
        out.append(
            {
                "slot": int(slot),
                "type": "capsule",
                "x": _safe_float(pos_world[env_id, slot, 0].item()),
                "y": _safe_float(pos_world[env_id, slot, 1].item()),
                "r": radius,
            }
        )
    for slot in range(box_start, box_end):
        if not bool(active[env_id, slot].item()):
            continue
        yaw = 0.0
        if torch.is_tensor(quat_world):
            qz = _safe_float(quat_world[env_id, slot, 2].item())
            qw = _safe_float(quat_world[env_id, slot, 3].item(), 1.0)
            yaw = 2.0 * math.atan2(qz, qw)
        out.append(
            {
                "slot": int(slot),
                "type": "box",
                "x": _safe_float(pos_world[env_id, slot, 0].item()),
                "y": _safe_float(pos_world[env_id, slot, 1].item()),
                "sx": box_x,
                "sy": box_y,
                "yaw": yaw,
            }
        )
    return _json_dumps_compact(out)


def _episode_termination_reason(
    *,
    task_success: bool,
    full_task_success: bool,
    final_collision: bool,
    target_lost_event: bool,
    timeout_or_other: bool,
) -> str:
    if bool(task_success) or bool(full_task_success):
        return "success"
    if bool(final_collision):
        return "collision"
    if bool(target_lost_event):
        return "target_lost"
    if bool(timeout_or_other):
        return "timeout"
    return "follow_lost"


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q))


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _corr_from_sums(sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float, n: int) -> float:
    if n < 2:
        return float("nan")
    nf = float(n)
    cov = sum_xy - (sum_x * sum_y / nf)
    var_x = sum_x2 - (sum_x * sum_x / nf)
    var_y = sum_y2 - (sum_y * sum_y / nf)
    denom = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
    if denom <= 1e-12:
        return float("nan")
    return float(cov / denom)


def _difficulty_list(s: str) -> List[float]:
    if not s:
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    out = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(np.clip(float(token), 0.0, 1.0)))
    if not out:
        out = [0.0, 0.25, 0.5, 0.75, 1.0]
    return out


def _is_pcr_eval_task(args) -> bool:
    return th.is_pcr_line_task_name(str(getattr(args, "task", "")))


def _apply_pcr_play_env_alignment(args, env) -> None:
    """Match the fixed PCR scene view used by play_highlevel."""
    if not _is_pcr_eval_task(args):
        return
    if not hasattr(env, "env") or env.env is None:
        return
    env_impl = env.env
    if hasattr(env_impl, "cfg") and hasattr(env_impl.cfg, "terrain"):
        env_impl.cfg.terrain.curriculum = False
    if hasattr(env_impl, "_update_terrain_curriculum"):
        def _no_terrain_update(self, env_ids):
            return
        env_impl._update_terrain_curriculum = types.MethodType(_no_terrain_update, env_impl)
    if hasattr(env_impl, "terrain_levels"):
        env_impl.terrain_levels.fill_(0)
        if hasattr(env_impl, "terrain_origins") and hasattr(env_impl, "terrain_types") and hasattr(env_impl, "env_origins"):
            env_impl.env_origins[:] = env_impl.terrain_origins[env_impl.terrain_levels, env_impl.terrain_types]
    freeze_stage = bool(getattr(args, "freeze_avoid_stage", False)) or (
        getattr(args, "avoid_stage_override", None) is not None
    )
    if freeze_stage and hasattr(env_impl, "_advance_s_avoid_stage"):
        def _no_stage_advance(self, next_stage):
            return
        env_impl._advance_s_avoid_stage = types.MethodType(_no_stage_advance, env_impl)
    _apply_eval_avoid_stage_override(args, env, verbose=False)
    if hasattr(env_impl, "debug_viz"):
        env_impl.debug_viz = bool(getattr(args, "debug", False)) or (
            _is_pcr_eval_task(args) and not bool(getattr(args, "headless", True))
        )


def _apply_eval_avoid_stage_override(args, env, *, verbose: bool = True) -> None:
    if not _is_pcr_eval_task(args):
        return
    stage_override = getattr(args, "avoid_stage_override", None)
    if stage_override is None:
        return
    if not hasattr(env, "env") or env.env is None:
        return
    env_impl = env.env
    if not hasattr(env_impl, "s_avoid_stage") or not hasattr(env_impl, "s_avoid_stage_per_env"):
        return
    stage_value = int(stage_override)
    if hasattr(env_impl, "cfg") and hasattr(env_impl.cfg, "terrain"):
        setattr(env_impl.cfg.terrain, "pcr_new_force_stage", stage_value)
    env_impl.s_avoid_stage = stage_value
    env_impl.s_avoid_stage_per_env.fill_(stage_value)
    if hasattr(env_impl, "extras") and isinstance(env_impl.extras, dict):
        env_impl.extras["avoid_stage"] = int(stage_value)
    if verbose:
        print(f"[Eval] s_avoid stage override -> {stage_value}", flush=True)


def _apply_pcr_train_runtime_alignment(args, env) -> None:
    """Use PCR train steady-state values for state fields that the policy observes."""
    if not _is_pcr_eval_task(args):
        return
    if not hasattr(env, "forced_forward_speed"):
        return
    env.forced_forward_train_warmup_ratio = 1.0
    env.forced_forward_stage_start_iter = 0
    env.forced_forward_current_iter = 200
    if hasattr(env, "env"):
        env.forced_forward_stage_last = int(getattr(env.env, "s_avoid_stage", 1))
    try:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env._resample_forced_forward_speed(env_ids)
    except Exception:
        pass


RISK_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.000001)
RISK_BIN_LABELS = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")


def _risk_bin_index(risk_f: float) -> Optional[int]:
    if not math.isfinite(risk_f):
        return None
    risk_v = float(np.clip(risk_f, 0.0, 1.0))
    for idx in range(len(RISK_BIN_EDGES) - 1):
        if RISK_BIN_EDGES[idx] <= risk_v < RISK_BIN_EDGES[idx + 1]:
            return idx
    return len(RISK_BIN_LABELS) - 1


def _empty_risk_bin_state() -> list:
    return [
        {
            "steps": 0,
            "gate_y_raw_sum": 0.0,
            "gate_y_raw_sq_sum": 0.0,
            "y_eff_sum": 0.0,
            "y_eff_sq_sum": 0.0,
            "suppression_sum": 0.0,
            "suppression_sq_sum": 0.0,
            "w_sum": 0.0,
            "w_sq_sum": 0.0,
            "signed_w_sum": 0.0,
            "signed_w_active_sum": 0.0,
            "risk_memory_sum": 0.0,
            "risk_f_sum": 0.0,
            "risk_a_sum": 0.0,
            "risk_delta_sum": 0.0,
            "w_support_correction_sum": 0.0,
            "risk_diff_correction_sum": 0.0,
            "near_miss_steps": 0,
        }
        for _ in RISK_BIN_LABELS
    ]


def _rollout_clearance_along_cmd(
    env,
    args,
    aff_map: torch.Tensor,
    cmd_xy: torch.Tensor,
) -> torch.Tensor:
    """Eval-only clearance to occupied cells along a finite command rollout."""
    if aff_map.ndim != 4 or aff_map.size(1) < 1 or cmd_xy.dim() != 2 or cmd_xy.size(1) != 2:
        return env._compute_clearance_along_cmd(aff_map, cmd_xy)

    x_map = getattr(env, "affordance_x_map", None)
    y_map = getattr(env, "affordance_y_map", None)
    if x_map is None or y_map is None:
        return env._compute_clearance_along_cmd(aff_map, cmd_xy)

    device = aff_map.device
    dtype = cmd_xy.dtype
    x_map = x_map.to(device=device, dtype=dtype)
    y_map = y_map.to(device=device, dtype=dtype)
    visible = getattr(env, "affordance_visible_mask", None)
    if visible is None:
        visible = torch.ones_like(x_map, dtype=torch.bool, device=device)
    else:
        visible = visible.to(device=device, dtype=torch.bool)

    occ = aff_map[:, 0] > 0.5
    if occ.shape[-2:] != x_map.shape:
        return env._compute_clearance_along_cmd(aff_map, cmd_xy)

    horizon_s = max(1e-3, float(getattr(args, "conflict_rollout_horizon_s", 1.2)))
    tube_radius = max(1e-3, float(getattr(args, "conflict_rollout_tube_radius_m", 0.25)))
    extent = float(getattr(env, "affordance_map_extent", 2.0))
    fill = torch.full_like(occ, float(extent), dtype=dtype)

    speed = torch.norm(cmd_xy, dim=-1)
    active = speed > 1e-4
    unit = cmd_xy / speed.clamp_min(1e-6).view(-1, 1)
    path_len = torch.clamp(speed * horizon_s, min=0.0, max=extent)

    x = x_map.view(1, *x_map.shape)
    y = y_map.view(1, *y_map.shape)
    ux = unit[:, 0].view(-1, 1, 1)
    uy = unit[:, 1].view(-1, 1, 1)
    s = x * ux + y * uy
    s_clamped = torch.clamp(s, min=0.0)
    s_clamped = torch.minimum(s_clamped, path_len.view(-1, 1, 1))
    closest_x = s_clamped * ux
    closest_y = s_clamped * uy
    dist_to_segment = torch.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
    in_rollout_band = (s >= -tube_radius) & (s <= (path_len.view(-1, 1, 1) + tube_radius))
    valid = occ & visible.view(1, *visible.shape) & in_rollout_band
    dist = torch.where(valid, dist_to_segment, fill)
    clearance = dist.flatten(1).amin(dim=1)
    no_valid = ~valid.flatten(1).any(dim=1)
    clearance = torch.where(no_valid, torch.full_like(clearance, extent), clearance)

    stop_clearance = env._compute_clearance_from_affordance(aff_map)
    return torch.where(active, clearance, stop_clearance)


def _privileged_conflict_diag(
    args,
    env,
    aff_map: torch.Tensor,
    gate_diag: Dict[str, torch.Tensor],
    target_in_fov: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Build eval-only PCR conflict diagnostics from privileged row geometry."""
    risk_ref = gate_diag["risk_F"]
    zero = torch.zeros_like(risk_ref)
    row_valid = gate_diag.get("row_current_valid", zero) > 0.5
    row_front = gate_diag.get("current_row_front_edge", zero)
    row_back = gate_diag.get("current_row_back_edge", zero)
    robot_front = gate_diag.get("robot_front_y", zero)
    robot_rear = gate_diag.get("robot_rear_y", zero)
    cmd_f = gate_diag.get("cmd_f", None)
    cmd_a = gate_diag.get("cmd_a", None)

    follow_forward = zero
    avoid_lateral_abs = zero
    if torch.is_tensor(cmd_f) and cmd_f.shape[-1] >= 2:
        follow_forward = cmd_f[:, 1].to(device=risk_ref.device, dtype=risk_ref.dtype)
    if torch.is_tensor(cmd_a) and cmd_a.shape[-1] >= 1:
        avoid_lateral_abs = torch.abs(cmd_a[:, 0].to(device=risk_ref.device, dtype=risk_ref.dtype))

    pre_m = float(getattr(args, "priv_conflict_pre_m", 0.6))
    post_m = float(getattr(args, "priv_conflict_post_m", 0.3))
    obstacle_window = (
        row_valid
        & (robot_front > (row_front - pre_m))
        & (robot_rear < (row_back + post_m))
    )
    follow_pressure = follow_forward > float(getattr(args, "priv_conflict_follow_thr", 0.20))
    avoid_pressure = avoid_lateral_abs > float(getattr(args, "priv_conflict_avoid_thr", 0.10))
    follow_score = torch.clamp((follow_forward - 0.15) / 0.30, min=0.0, max=1.0)
    avoid_score = torch.clamp((avoid_lateral_abs - 0.05) / 0.25, min=0.0, max=1.0)
    score = obstacle_window.to(dtype=risk_ref.dtype) * follow_score * avoid_score
    high_mask = (
        obstacle_window
        & follow_pressure
        & avoid_pressure
        & (score > float(getattr(args, "priv_conflict_score_thr", 0.25)))
    )
    risk_f_base = gate_diag.get("risk_F", zero)
    risk_a_base = gate_diag.get("risk_A", zero)
    cmd_cos = gate_diag.get("cmd_cos", torch.ones_like(risk_ref))
    cmd_s = torch.zeros_like(cmd_f) if torch.is_tensor(cmd_f) else None
    if torch.is_tensor(cmd_s) and str(getattr(args, "conflict_stop_candidate", "stop")) == "slow":
        slow_ratio = float(getattr(args, "conflict_stop_slow_ratio", 0.2))
        if cmd_s.shape[-1] >= 2:
            cmd_s[:, 1] = torch.clamp(cmd_f[:, 1], min=0.0) * slow_ratio
        if cmd_s.shape[-1] >= 3:
            cmd_s[:, 2] = cmd_f[:, 2]

    if torch.is_tensor(cmd_f) and torch.is_tensor(cmd_a) and torch.is_tensor(cmd_s):
        safe_d, free_d = th._get_effective_safe_free_dist(
            env,
            getattr(args, "beta", None),
            device=risk_ref.device,
            dtype=risk_ref.dtype,
        )
        clearance_rollout_f = _rollout_clearance_along_cmd(env, args, aff_map, cmd_f[:, :2])
        clearance_rollout_a = _rollout_clearance_along_cmd(env, args, aff_map, cmd_a[:, :2])
        clearance_rollout_s = _rollout_clearance_along_cmd(env, args, aff_map, cmd_s[:, :2])
        risk_f = env._risk_from_clearance(clearance_rollout_f, safe_d, free_d)
        risk_a = env._risk_from_clearance(clearance_rollout_a, safe_d, free_d)
        risk_s = env._risk_from_clearance(clearance_rollout_s, safe_d, free_d)
    else:
        clearance_rollout_f = gate_diag.get("clearance_F", zero)
        clearance_rollout_a = gate_diag.get("clearance_A", zero)
        clearance_rollout_s = env._compute_clearance_from_affordance(aff_map)
        risk_f = risk_f_base
        risk_a = risk_a_base
        risk_s = env._risk_from_clearance(
            clearance_rollout_s,
            *th._get_effective_safe_free_dist(
                env,
                getattr(args, "beta", None),
                device=risk_ref.device,
                dtype=risk_ref.dtype,
            ),
        )

    if target_in_fov is None:
        target_recoverable = torch.ones_like(risk_ref, dtype=torch.bool)
    else:
        target_recoverable = target_in_fov.to(device=risk_ref.device, dtype=torch.bool)

    horizon_s = max(1e-3, float(getattr(args, "conflict_rollout_horizon_s", 1.2)))
    target_speed = getattr(getattr(env, "env", None), "target_speed", None)
    if torch.is_tensor(target_speed):
        target_speed = target_speed.to(device=risk_ref.device, dtype=risk_ref.dtype)
    else:
        target_speed = torch.zeros_like(risk_ref)
    cmd_a_lateral_abs = (
        torch.abs(cmd_a[:, 0].to(device=risk_ref.device, dtype=risk_ref.dtype))
        if torch.is_tensor(cmd_a) and cmd_a.shape[-1] >= 1
        else zero
    )
    cmd_s_forward = (
        cmd_s[:, 1].to(device=risk_ref.device, dtype=risk_ref.dtype)
        if torch.is_tensor(cmd_s) and cmd_s.shape[-1] >= 2
        else zero
    )
    lateral_opening_cap = max(1e-3, float(getattr(args, "conflict_utility_lateral_opening_cap_m", 0.45)))
    avoid_lateral_opening = torch.clamp(cmd_a_lateral_abs * horizon_s, max=lateral_opening_cap)
    stop_progress = torch.clamp(cmd_s_forward, min=0.0) * horizon_s
    avoid_target_gap_growth = target_speed * horizon_s
    stop_target_gap_growth = torch.clamp(target_speed - torch.clamp(cmd_s_forward, min=0.0), min=0.0) * horizon_s
    progress_gain = float(getattr(args, "conflict_utility_progress_gain", 0.25))
    lateral_gain = float(getattr(args, "conflict_utility_lateral_gain", 0.35))
    target_gap_cost = float(getattr(args, "conflict_utility_target_gap_cost", 0.35))
    risk_cost = float(getattr(args, "conflict_utility_risk_cost", 1.0))
    utility_a = (
        -risk_cost * risk_a
        + lateral_gain * avoid_lateral_opening
        - target_gap_cost * avoid_target_gap_growth
    )
    utility_s = (
        -risk_cost * risk_s
        + progress_gain * stop_progress
        - target_gap_cost * stop_target_gap_growth
    )

    risk_alt = torch.minimum(risk_a, risk_s)
    unsafe_follow = risk_f > float(getattr(args, "unsafe_conflict_risk_f_thr", 0.25))
    safe_candidate_better = (risk_f - risk_alt) > float(getattr(args, "unsafe_conflict_risk_margin", 0.05))
    command_disagree = cmd_cos < float(getattr(args, "unsafe_conflict_cmd_cos_thr", 0.5))
    unsafe_raw = obstacle_window & unsafe_follow & safe_candidate_better & command_disagree & target_recoverable
    prefer_margin = float(getattr(args, "conflict_utility_margin", 0.03))
    avoid_raw = unsafe_raw & ((utility_a - utility_s) > prefer_margin)
    stop_raw = unsafe_raw & (~avoid_raw)

    phase_code = torch.zeros_like(risk_ref)
    approach = obstacle_window & (robot_front < row_front)
    release = obstacle_window & (robot_rear > row_back)
    inside = obstacle_window & (~approach) & (~release)
    phase_code = torch.where(approach, torch.ones_like(phase_code), phase_code)
    phase_code = torch.where(inside, torch.full_like(phase_code, 2.0), phase_code)
    phase_code = torch.where(release, torch.full_like(phase_code, 3.0), phase_code)
    return {
        "priv_conflict_score": torch.clamp(score, 0.0, 1.0),
        "priv_high_conflict": high_mask.to(dtype=risk_ref.dtype),
        "priv_obstacle_window": obstacle_window.to(dtype=risk_ref.dtype),
        "priv_follow_pressure": follow_pressure.to(dtype=risk_ref.dtype),
        "priv_avoid_pressure": avoid_pressure.to(dtype=risk_ref.dtype),
        "priv_conflict_phase": phase_code,
        "cmd_S": cmd_s if torch.is_tensor(cmd_s) else torch.zeros_like(gate_diag["cmd_f"]),
        "clearance_rollout_F": clearance_rollout_f.to(dtype=risk_ref.dtype),
        "clearance_rollout_A": clearance_rollout_a.to(dtype=risk_ref.dtype),
        "clearance_rollout_S": clearance_rollout_s.to(dtype=risk_ref.dtype),
        "risk_rollout_F": risk_f.to(dtype=risk_ref.dtype),
        "risk_rollout_A": risk_a.to(dtype=risk_ref.dtype),
        "risk_rollout_S": risk_s.to(dtype=risk_ref.dtype),
        "utility_A": utility_a.to(dtype=risk_ref.dtype),
        "utility_S": utility_s.to(dtype=risk_ref.dtype),
        "utility_A_minus_S": (utility_a - utility_s).to(dtype=risk_ref.dtype),
        "avoid_lateral_opening": avoid_lateral_opening.to(dtype=risk_ref.dtype),
        "stop_forward_progress": stop_progress.to(dtype=risk_ref.dtype),
        "avoid_target_gap_growth": avoid_target_gap_growth.to(dtype=risk_ref.dtype),
        "stop_target_gap_growth": stop_target_gap_growth.to(dtype=risk_ref.dtype),
        "target_recoverable_for_conflict": target_recoverable.to(dtype=risk_ref.dtype),
        "unsafe_high_conflict_raw": unsafe_raw.to(dtype=risk_ref.dtype),
        "avoid_high_conflict_raw": avoid_raw.to(dtype=risk_ref.dtype),
        "stop_high_conflict_raw": stop_raw.to(dtype=risk_ref.dtype),
        "unsafe_high_conflict": unsafe_raw.to(dtype=risk_ref.dtype),
        "avoid_high_conflict": avoid_raw.to(dtype=risk_ref.dtype),
        "stop_high_conflict": stop_raw.to(dtype=risk_ref.dtype),
        "unsafe_follow_risk": unsafe_follow.to(dtype=risk_ref.dtype),
        "unsafe_safe_candidate_better": safe_candidate_better.to(dtype=risk_ref.dtype),
        "unsafe_avoid_safer": ((risk_f - risk_a) > float(getattr(args, "unsafe_conflict_risk_margin", 0.05))).to(dtype=risk_ref.dtype),
        "unsafe_stop_safer": ((risk_f - risk_s) > float(getattr(args, "unsafe_conflict_risk_margin", 0.05))).to(dtype=risk_ref.dtype),
        "unsafe_command_disagree": command_disagree.to(dtype=risk_ref.dtype),
    }


def _compute_rule_override_cmd(
    args,
    cmd_f: torch.Tensor,
    cmd_a: torch.Tensor,
    risk_f: torch.Tensor,
    risk_a: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reactive safety rule that replaces learned PCR arbitration at eval time."""
    # Internal high-level command convention:
    # cmd[..., 0] = x_right / lateral
    # cmd[..., 1] = y_forward / forward
    # cmd[..., 2] = yaw
    risk_gap = risk_f - risk_a
    k = float(getattr(args, "rule_k", 8.0))
    margin = float(getattr(args, "rule_margin", 0.10))
    hard_thr = float(getattr(args, "rule_hard_thr", 0.45))
    s_min = float(getattr(args, "rule_s_min", 0.85))
    slow_ratio = float(getattr(args, "rule_slow_ratio", 0.10))
    yaw_keep_loss = float(getattr(args, "rule_yaw_keep_loss", 0.30))

    s = torch.sigmoid(k * (risk_gap - margin))
    s_min_t = torch.full_like(s, s_min)
    s = torch.where(risk_f > hard_thr, torch.maximum(s, s_min_t), s)
    s = torch.clamp(s, 0.0, 1.0)

    follow_scale = torch.clamp(1.0 - s + slow_ratio * s, 0.0, 1.0)
    yaw_scale = torch.clamp(1.0 - yaw_keep_loss * s, 0.0, 1.0)

    cmd = torch.zeros_like(cmd_f)
    if cmd.shape[-1] >= 1 and cmd_a.shape[-1] >= 1:
        cmd[:, 0] = cmd_a[:, 0]
    if cmd.shape[-1] >= 2 and cmd_f.shape[-1] >= 2:
        cmd[:, 1] = follow_scale * cmd_f[:, 1]
    if cmd.shape[-1] >= 3 and cmd_f.shape[-1] >= 3:
        cmd[:, 2] = yaw_scale * cmd_f[:, 2]

    return cmd, {
        "rule_s": s,
        "rule_risk_gap": risk_gap,
        "rule_follow_scale": follow_scale,
        "rule_yaw_scale": yaw_scale,
        "rule_follow_suppression": 1.0 - follow_scale,
    }


def _ensure_delta_y_diag(gate_diag: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gate_y = gate_diag.get("gate_y", gate_diag["gate_y_raw"])
    zero = torch.zeros_like(gate_y)
    delta_y_w = gate_diag.get("w_support_correction", zero)
    delta_y_r = gate_diag.get("risk_diff_correction", zero)
    delta_y_total = gate_diag.get("y_eff", gate_y) - gate_y
    gate_diag.setdefault("delta_y_w_raw", delta_y_w)
    gate_diag.setdefault("delta_y_w_used", delta_y_w)
    gate_diag.setdefault("delta_y_r", delta_y_r)
    gate_diag.setdefault("delta_y_total", delta_y_total)
    return gate_diag


def _apply_disable_delta_y_w_ablation(gate_diag: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    gate_y = gate_diag["gate_y"] if "gate_y" in gate_diag else gate_diag["gate_y_raw"]
    zero = torch.zeros_like(gate_y)
    delta_y_w_raw = gate_diag.get("delta_y_w_raw", gate_diag.get("w_support_correction", zero))
    delta_y_r = gate_diag.get("risk_diff_correction", zero)

    y_eff = torch.clamp(gate_y + delta_y_r, 0.0, 1.0)
    gate_diag["delta_y_w_raw"] = delta_y_w_raw.detach().clone()
    gate_diag["w_support_correction"] = zero
    gate_diag["delta_y_w_used"] = zero
    gate_diag["delta_y_r"] = delta_y_r
    gate_diag["y_eff"] = y_eff
    gate_diag["cmd"] = (
        y_eff.unsqueeze(-1) * gate_diag["cmd_f"]
        + (1.0 - y_eff.unsqueeze(-1)) * gate_diag["cmd_a"]
    )
    gate_diag["delta_y_total"] = y_eff - gate_y
    return gate_diag


def _compute_additive_fusion_diag(
    env,
    args,
    aff_map: torch.Tensor,
    cmd_f: torch.Tensor,
    cmd_a: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Eval-only baseline: lateral Avoid + forward/yaw Follow, no gate policy."""
    diag = th._pcr_gate_command_conflict_diag(env, args, aff_map, cmd_f, cmd_a)
    cmd_a_eff = diag["cmd_a_eff"]
    cmd = cmd_f + cmd_a_eff
    ref = diag["risk_F"]
    nan = torch.full_like(ref, float("nan"))
    zero = torch.zeros_like(ref)
    one = torch.ones_like(ref)
    return {
        "cmd": cmd,
        "cmd_f": cmd_f,
        "cmd_a": cmd_a_eff,
        "gate_y_raw": nan,
        "gate_y": nan,
        "gate_y_safe": nan,
        "y_eff": nan,
        "w": nan,
        "signed_w": nan,
        "signed_w_active": zero,
        "w_support_correction": zero,
        "risk_diff_correction": zero,
        "delta_y_w_raw": nan,
        "delta_y_w_used": nan,
        "delta_y_r": nan,
        "delta_y_total": nan,
        "risk_memory": zero,
        "follow_weight": one,
        "avoid_weight": one,
        "additive_fusion_raw": cmd,
        "additive_lateral_projection": one,
        **diag,
    }


def _resolve_velocity_search_hparams(args) -> Dict[str, float]:
    defaults = {
        "vx_samples": 7,
        "vy_samples": 5,
        "w_samples": 5,
        "rollout_horizon_s": 1.2,
        "rollout_dt_s": 0.15,
        "body_radius_m": 0.32,
        "footprint_radius_m": 0.32,  # legacy alias: body_radius_m
        "min_clearance_margin": 0.08,
        "clear_weight": 3.0,
        "target_dist_weight": 1.0,
        "bearing_weight": 0.35,
        "progress_weight": 0.45,
        "forward_weight": 0.0,
        "forward_min_v": 0.15,
        "smooth_weight": 0.20,
        "target_weight": 1.0,  # legacy alias: target_dist_weight
        "risk_weight": 3.0,    # legacy alias: clear_weight
        "speed_weight": 0.25,  # legacy alias: progress_weight
        "lateral_weight": 0.05,
        "risk_filter_threshold": 0.65,
        "filter_mode": "soft",
        "fallback": "stop",
        "use_dynamic_window": False,
        "acc_lat_max": 1.0,
        "acc_fwd_max": 1.5,
        "acc_yaw_max": 2.0,
    }
    path = str(getattr(args, "velocity_search_hparams_json", "") or "").strip()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"velocity search hparams json must be a dict: {path}")
        if isinstance(loaded.get("hparams", None), dict):
            loaded = loaded["hparams"]
        defaults.update(loaded)
    cli_map = {
        "vx_samples": getattr(args, "velocity_search_vx_samples", None),
        "vy_samples": getattr(args, "velocity_search_vy_samples", None),
        "w_samples": getattr(args, "velocity_search_w_samples", None),
        "rollout_horizon_s": getattr(args, "velocity_search_rollout_horizon_s", None),
        "rollout_dt_s": getattr(args, "velocity_search_rollout_dt_s", None),
        "body_radius_m": getattr(args, "velocity_search_body_radius_m", None),
        "footprint_radius_m": getattr(args, "velocity_search_footprint_radius_m", None),
        "min_clearance_margin": getattr(args, "velocity_search_min_clearance_margin", None),
        "clear_weight": getattr(args, "velocity_search_clear_weight", None),
        "target_dist_weight": getattr(args, "velocity_search_target_dist_weight", None),
        "bearing_weight": getattr(args, "velocity_search_bearing_weight", None),
        "progress_weight": getattr(args, "velocity_search_progress_weight", None),
        "forward_weight": getattr(args, "velocity_search_forward_weight", None),
        "forward_min_v": getattr(args, "velocity_search_forward_min_v", None),
        "smooth_weight": getattr(args, "velocity_search_smooth_weight", None),
        "target_weight": getattr(args, "velocity_search_target_weight", None),
        "risk_weight": getattr(args, "velocity_search_risk_weight", None),
        "speed_weight": getattr(args, "velocity_search_speed_weight", None),
        "lateral_weight": getattr(args, "velocity_search_lateral_weight", None),
        "risk_filter_threshold": getattr(args, "velocity_search_risk_filter_threshold", None),
        "filter_mode": getattr(args, "velocity_search_filter_mode", None),
        "fallback": getattr(args, "velocity_search_fallback", None),
        "acc_lat_max": getattr(args, "velocity_search_acc_lat_max", None),
        "acc_fwd_max": getattr(args, "velocity_search_acc_fwd_max", None),
        "acc_yaw_max": getattr(args, "velocity_search_acc_yaw_max", None),
    }
    for key, value in cli_map.items():
        if value is not None:
            defaults[key] = value
    defaults["vx_samples"] = int(max(3, defaults["vx_samples"]))
    defaults["vy_samples"] = int(max(2, defaults["vy_samples"]))
    defaults["w_samples"] = int(max(3, defaults["w_samples"]))
    defaults["rollout_horizon_s"] = float(max(0.2, defaults["rollout_horizon_s"]))
    defaults["rollout_dt_s"] = float(max(0.05, defaults["rollout_dt_s"]))
    if "body_radius_m" not in defaults or defaults["body_radius_m"] is None:
        defaults["body_radius_m"] = defaults.get("footprint_radius_m", 0.32)
    defaults["body_radius_m"] = float(max(0.05, defaults["body_radius_m"]))
    defaults["footprint_radius_m"] = float(max(0.05, defaults.get("footprint_radius_m", defaults["body_radius_m"])))
    # Use body_radius_m as the planner-facing name; keep footprint_radius_m only for older json files.
    defaults["footprint_radius_m"] = defaults["body_radius_m"]
    defaults["min_clearance_margin"] = float(max(0.0, defaults["min_clearance_margin"]))
    # Keep old json files usable while exposing the paper-facing cost names.
    defaults["clear_weight"] = float(defaults.get("clear_weight", defaults.get("risk_weight", 3.0)))
    defaults["target_dist_weight"] = float(defaults.get("target_dist_weight", defaults.get("target_weight", 1.0)))
    defaults["progress_weight"] = float(defaults.get("progress_weight", defaults.get("speed_weight", 0.25)))
    defaults["forward_weight"] = float(defaults["forward_weight"])
    defaults["forward_min_v"] = float(max(0.0, defaults["forward_min_v"]))
    defaults["bearing_weight"] = float(defaults["bearing_weight"])
    defaults["smooth_weight"] = float(defaults["smooth_weight"])
    defaults["target_weight"] = float(defaults["target_weight"])
    defaults["risk_weight"] = float(defaults["risk_weight"])
    defaults["speed_weight"] = float(defaults["speed_weight"])
    defaults["lateral_weight"] = float(defaults["lateral_weight"])
    defaults["risk_filter_threshold"] = float(defaults["risk_filter_threshold"])
    defaults["filter_mode"] = str(defaults["filter_mode"]).strip().lower()
    if defaults["filter_mode"] not in {"scale", "soft", "none"}:
        raise ValueError(f"invalid velocity_search filter_mode: {defaults['filter_mode']}")
    defaults["fallback"] = str(defaults["fallback"])
    defaults["use_dynamic_window"] = bool(
        defaults.get("use_dynamic_window", False)
        or getattr(args, "velocity_search_use_dynamic_window", False)
    )
    defaults["acc_lat_max"] = float(max(0.0, defaults["acc_lat_max"]))
    defaults["acc_fwd_max"] = float(max(0.0, defaults["acc_fwd_max"]))
    defaults["acc_yaw_max"] = float(max(0.0, defaults["acc_yaw_max"]))
    return defaults


def _velocity_search_rollout_costs(
    env,
    args,
    aff_map: torch.Tensor,
    candidates: torch.Tensor,
    goal_xy: torch.Tensor,
    max_cmd: torch.Tensor,
    hparams: Dict[str, float],
    last_cmd: Optional[torch.Tensor],
    candidate_reachable_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Roll out constant body-frame velocity candidates on the local map."""
    if aff_map.ndim != 4 or aff_map.size(1) < 1:
        raise ValueError(f"aff_map shape invalid for velocity search: {tuple(aff_map.shape)}")
    x_map = getattr(env, "affordance_x_map", None)
    y_map = getattr(env, "affordance_y_map", None)
    if x_map is None or y_map is None:
        raise ValueError("velocity search requires affordance_x_map and affordance_y_map")

    device = aff_map.device
    dtype = candidates.dtype
    n, k, _ = candidates.shape
    x_map = x_map.to(device=device, dtype=dtype)
    y_map = y_map.to(device=device, dtype=dtype)
    if aff_map.shape[-2:] != x_map.shape:
        raise ValueError("velocity search map grid shape mismatch")
    visible = getattr(env, "affordance_visible_mask", None)
    if visible is None:
        visible = torch.ones_like(x_map, dtype=torch.bool, device=device)
    else:
        visible = visible.to(device=device, dtype=torch.bool)
    occ = (aff_map[:, 0] > 0.5) & visible.view(1, *visible.shape)

    horizon_s = float(hparams["rollout_horizon_s"])
    dt = float(hparams["rollout_dt_s"])
    steps = int(max(1, math.ceil(horizon_s / max(dt, 1e-6))))
    radius = float(hparams["body_radius_m"])
    clearance_margin = float(hparams["min_clearance_margin"])
    extent = float(getattr(env, "affordance_map_extent", 3.0))
    x_min = float(x_map.min().detach().cpu().item()) - 0.5 * radius
    x_max = float(x_map.max().detach().cpu().item()) + 0.5 * radius
    y_min = float(y_map.min().detach().cpu().item()) - 0.5 * radius
    y_max = float(y_map.max().detach().cpu().item()) + 0.5 * radius

    vx = candidates[:, :, 0]
    vy = candidates[:, :, 1]
    ww = candidates[:, :, 2]
    x = torch.zeros((n, k), device=device, dtype=dtype)
    y = torch.zeros((n, k), device=device, dtype=dtype)
    yaw = torch.zeros((n, k), device=device, dtype=dtype)
    min_center_dist = torch.full((n, k), float(extent), device=device, dtype=dtype)
    collision = torch.zeros((n, k), device=device, dtype=torch.bool)
    out_of_map_any = torch.zeros((n, k), device=device, dtype=torch.bool)

    grid_x = x_map.view(1, 1, *x_map.shape)
    grid_y = y_map.view(1, 1, *y_map.shape)
    occ_view = occ.view(n, 1, *occ.shape[-2:])
    fill = torch.full((n, k, *x_map.shape), float(extent), device=device, dtype=dtype)

    for _ in range(steps):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        x = x + (vx * c - vy * s) * dt
        y = y + (vx * s + vy * c) * dt
        yaw = yaw + ww * dt
        out_of_map = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
        out_of_map_any = out_of_map_any | out_of_map
        dx = grid_x - x.view(n, k, 1, 1)
        dy = grid_y - y.view(n, k, 1, 1)
        center_dist_map = torch.sqrt(dx * dx + dy * dy + 1e-12)
        selected = torch.where(occ_view, center_dist_map, fill)
        step_center_dist = selected.flatten(2).amin(dim=2)
        has_occ = occ.flatten(1).any(dim=1).view(n, 1)
        step_center_dist = torch.where(has_occ, step_center_dist, torch.full_like(step_center_dist, float(extent)))
        min_center_dist = torch.minimum(min_center_dist, step_center_dist)
        collision = collision | (step_center_dist <= radius)

    predicted_clearance = min_center_dist - radius
    margin_clear = predicted_clearance >= clearance_margin
    feasible = (~collision) & (~out_of_map_any) & margin_clear
    final_x = x
    final_y = y
    final_yaw = yaw

    safe_d, free_d = th._get_effective_safe_free_dist(
        env,
        getattr(args, "beta", None),
        device=device,
        dtype=dtype,
    )
    risk = env._risk_from_clearance(predicted_clearance.clamp_min(0.0), safe_d, free_d)

    goal_norm = torch.norm(goal_xy, dim=-1, keepdim=True)
    default_goal = torch.zeros_like(goal_xy)
    default_goal[:, 1] = 1.0
    goal_dir = torch.where(goal_norm > 1e-6, goal_xy / goal_norm.clamp_min(1e-6), default_goal)
    final_xy = torch.stack([final_x, final_y], dim=-1)
    target_vec = goal_xy.unsqueeze(1) - final_xy
    target_dist = torch.norm(target_vec, dim=-1) / max(extent, 1e-6)
    cy = torch.cos(final_yaw)
    sy = torch.sin(final_yaw)
    rel_x_body = cy * target_vec[:, :, 0] + sy * target_vec[:, :, 1]
    rel_y_body = -sy * target_vec[:, :, 0] + cy * target_vec[:, :, 1]
    bearing_cost = torch.abs(torch.atan2(rel_x_body, rel_y_body)) / math.pi
    progress = torch.sum(final_xy * goal_dir.unsqueeze(1), dim=-1) / max(extent, 1e-6)
    forward_progress = final_y / max(extent, 1e-6)

    if torch.is_tensor(last_cmd) and last_cmd.shape == (n, 3):
        prev = last_cmd.to(device=device, dtype=dtype)
    else:
        prev = torch.zeros((n, 3), device=device, dtype=dtype)
    smooth = torch.norm((candidates - prev.unsqueeze(1)) / max_cmd.view(1, 1, 3).clamp_min(1e-6), dim=-1)
    lateral_penalty = torch.abs(candidates[:, :, 0]) / torch.clamp(max_cmd[0].abs(), min=1e-6)

    cost = (
        float(hparams["clear_weight"]) * risk
        + float(hparams["target_dist_weight"]) * target_dist
        + float(hparams["bearing_weight"]) * bearing_cost
        - float(hparams["progress_weight"]) * progress
        - float(hparams["forward_weight"]) * forward_progress
        + float(hparams["smooth_weight"]) * smooth
        + float(hparams["lateral_weight"]) * lateral_penalty
    )
    if torch.is_tensor(candidate_reachable_mask):
        reachable = candidate_reachable_mask.to(device=device, dtype=torch.bool)
        if reachable.shape != feasible.shape:
            raise ValueError(
                "velocity search dynamic-window mask shape mismatch: "
                f"mask={tuple(reachable.shape)}, feasible={tuple(feasible.shape)}"
            )
        feasible = feasible & reachable

    cost = torch.where(feasible, cost, torch.full_like(cost, float("inf")))
    forward_mask = candidates[:, :, 1] >= float(hparams["forward_min_v"])
    forward_feasible = forward_mask & feasible
    forward_cost = torch.where(forward_feasible, cost, torch.full_like(cost, float("inf")))
    best_forward_idx = torch.argmin(forward_cost, dim=1)
    batch_idx = torch.arange(n, device=device)
    has_forward_feasible = forward_feasible.any(dim=1)
    best_forward_cost = torch.where(
        has_forward_feasible,
        forward_cost[batch_idx, best_forward_idx],
        torch.full((n,), float("inf"), device=device, dtype=dtype),
    )
    best_forward_clearance = torch.where(
        has_forward_feasible,
        predicted_clearance[batch_idx, best_forward_idx],
        torch.full((n,), float("nan"), device=device, dtype=dtype),
    )
    return {
        "cost": cost,
        "risk": risk,
        "predicted_clearance": predicted_clearance,
        "feasible": feasible,
        "collision": collision,
        "out_of_map": out_of_map_any,
        "margin_clear": margin_clear,
        "candidate_count": torch.full((n,), k, device=device, dtype=dtype),
        "feasible_count": feasible.sum(dim=1).to(dtype=dtype),
        "infeasible_count": (~feasible).sum(dim=1).to(dtype=dtype),
        "forward_candidate_count": forward_mask.sum(dim=1).to(dtype=dtype),
        "forward_feasible_count": forward_feasible.sum(dim=1).to(dtype=dtype),
        "forward_infeasible_collision_count": (forward_mask & collision).sum(dim=1).to(dtype=dtype),
        "forward_infeasible_margin_count": (forward_mask & (~margin_clear)).sum(dim=1).to(dtype=dtype),
        "forward_infeasible_out_of_map_count": (forward_mask & out_of_map_any).sum(dim=1).to(dtype=dtype),
        "best_forward_feasible_cost": best_forward_cost,
        "best_forward_feasible_clearance": best_forward_clearance,
    }


def _compute_velocity_search_diag(
    env,
    args,
    aff_map: torch.Tensor,
    cmd_f: torch.Tensor,
    cmd_a: torch.Tensor,
    goal: torch.Tensor,
    hparams: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """DWA-style target-aware velocity-space search baseline; no gate policy."""
    t0 = time.perf_counter()
    diag = th._pcr_gate_command_conflict_diag(env, args, aff_map, cmd_f, cmd_a)
    ref = diag["risk_F"]
    device = cmd_f.device
    dtype = cmd_f.dtype
    max_cmd = getattr(env.post_processor, "max_cmd", None)
    if torch.is_tensor(max_cmd) and max_cmd.numel() >= 3:
        max_cmd = max_cmd.to(device=device, dtype=dtype)
    else:
        max_cmd = torch.tensor([0.4, 0.8, 0.3], device=device, dtype=dtype)
    vx_samples = int(hparams["vx_samples"])
    vy_samples = int(hparams["vy_samples"])
    w_samples = int(hparams["w_samples"])
    def _merge_velocity_samples(base: torch.Tensor, extra_values: list, low: float, high: float) -> torch.Tensor:
        extra = torch.tensor(extra_values, device=device, dtype=dtype).clamp(min=low, max=high)
        vals = torch.cat([base, extra], dim=0)
        vals = torch.round(vals * 10000.0) / 10000.0
        return torch.unique(vals, sorted=True)

    vx_base = torch.linspace(-float(max_cmd[0]), float(max_cmd[0]), vx_samples, device=device, dtype=dtype)
    vy_base = torch.linspace(0.0, float(max_cmd[1]), vy_samples, device=device, dtype=dtype)
    w_base = torch.linspace(-float(max_cmd[2]), float(max_cmd[2]), w_samples, device=device, dtype=dtype)
    vx_vals = _merge_velocity_samples(
        vx_base,
        [-0.45, -0.35, -0.25, -0.15, 0.0, 0.15, 0.25, 0.35, 0.45],
        -float(max_cmd[0]),
        float(max_cmd[0]),
    )
    vy_vals = _merge_velocity_samples(
        vy_base,
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        0.0,
        float(max_cmd[1]),
    )
    w_vals = _merge_velocity_samples(
        w_base,
        [-0.8, -0.4, 0.0, 0.4, 0.8],
        -float(max_cmd[2]),
        float(max_cmd[2]),
    )
    vx_grid, vy_grid, w_grid = torch.meshgrid(vx_vals, vy_vals, w_vals, indexing="ij")
    cand_cmd = torch.stack([vx_grid.reshape(-1), vy_grid.reshape(-1), w_grid.reshape(-1)], dim=-1)
    n = int(cmd_f.shape[0])
    k_before_dynamic_window = int(cand_cmd.shape[0])
    dynamic_window_enabled = bool(hparams.get("use_dynamic_window", False))
    dynamic_window_mask: Optional[torch.Tensor] = None
    if dynamic_window_enabled:
        last_cmd = getattr(env.post_processor, "last_cmd", None)
        if torch.is_tensor(last_cmd) and last_cmd.shape == (n, 3):
            prev_cmd = last_cmd.to(device=device, dtype=dtype)
        else:
            prev_cmd = torch.zeros((n, 3), device=device, dtype=dtype)
        dt_dyn = float(getattr(env, "high_level_dt", hparams.get("rollout_dt_s", 0.1)))
        dt_dyn = max(1e-6, dt_dyn)
        acc_limit = torch.tensor(
            [
                float(hparams.get("acc_lat_max", 1.0)),
                float(hparams.get("acc_fwd_max", 1.5)),
                float(hparams.get("acc_yaw_max", 2.0)),
            ],
            device=device,
            dtype=dtype,
        ) * dt_dyn
        diff = torch.abs(cand_cmd.view(1, k_before_dynamic_window, 3) - prev_cmd.view(n, 1, 3))
        dynamic_window_mask_full = (diff <= acc_limit.view(1, 1, 3) + 1e-6).all(dim=-1)
        keep_any = dynamic_window_mask_full.any(dim=0)
        if bool(keep_any.any().item()):
            cand_cmd = cand_cmd[keep_any]
            dynamic_window_mask = dynamic_window_mask_full[:, keep_any]
        else:
            # Keep one stopped candidate so downstream fallback accounting stays well-defined.
            stop_idx = torch.argmin(torch.norm(cand_cmd, dim=-1))
            cand_cmd = cand_cmd[stop_idx: stop_idx + 1]
            dynamic_window_mask = torch.zeros((n, 1), device=device, dtype=torch.bool)
    k = int(cand_cmd.shape[0])
    cand_cmd_batch = cand_cmd.unsqueeze(0).expand(n, k, 3)
    if dynamic_window_mask is None:
        dynamic_window_after_count = torch.full((n,), k, device=device, dtype=dtype)
        dynamic_window_rejected_count = torch.zeros((n,), device=device, dtype=dtype)
    else:
        dynamic_window_after_count = dynamic_window_mask.sum(dim=1).to(dtype=dtype)
        dynamic_window_rejected_count = (
            float(k_before_dynamic_window) - dynamic_window_after_count
        ).to(dtype=dtype)
    dynamic_window_before_count = torch.full((n,), k_before_dynamic_window, device=device, dtype=dtype)

    goal_xy = goal[:, :2].to(device=device, dtype=dtype)
    try:
        rollout = _velocity_search_rollout_costs(
            env,
            args,
            aff_map,
            cand_cmd_batch,
            goal_xy,
            max_cmd,
            hparams,
            getattr(env.post_processor, "last_cmd", None),
            candidate_reachable_mask=dynamic_window_mask,
        )
        cost = rollout["cost"]
        risk = rollout["risk"]
        predicted_clearance = rollout["predicted_clearance"]
        feasible = rollout["feasible"]
        collision = rollout["collision"]
        out_of_map = rollout["out_of_map"]
        margin_clear = rollout["margin_clear"]
        candidate_count = rollout["candidate_count"]
        feasible_count = rollout["feasible_count"]
        infeasible_count = rollout["infeasible_count"]
    except Exception:
        # Keep eval runnable on older envs, but mark the planner as having no feasible rollout.
        cand_xy_batch = cand_cmd_batch[:, :, :2]
        aff_rep = aff_map.repeat_interleave(k, dim=0)
        cand_flat = cand_xy_batch.reshape(n * k, 2)
        clearance = env._compute_clearance_along_cmd(aff_rep, cand_flat).view(n, k)
        safe_d, free_d = th._get_effective_safe_free_dist(
            env,
            getattr(args, "beta", None),
            device=device,
            dtype=dtype,
        )
        risk = env._risk_from_clearance(clearance, safe_d, free_d)
        predicted_clearance = clearance
        collision = torch.zeros_like(risk, dtype=torch.bool)
        out_of_map = torch.zeros_like(risk, dtype=torch.bool)
        margin_clear = predicted_clearance >= float(hparams["min_clearance_margin"])
        feasible = margin_clear
        if torch.is_tensor(dynamic_window_mask):
            feasible = feasible & dynamic_window_mask.to(device=device, dtype=torch.bool)
        target_norm = torch.norm(cand_xy_batch - goal_xy.unsqueeze(1), dim=-1)
        cost = float(hparams["clear_weight"]) * risk + float(hparams["target_dist_weight"]) * target_norm
        candidate_count = torch.full((n,), k, device=device, dtype=dtype)
        feasible_count = feasible.sum(dim=1).to(dtype=dtype)
        infeasible_count = torch.zeros_like(feasible_count)
        forward_mask = cand_cmd_batch[:, :, 1] >= float(hparams["forward_min_v"])
        forward_feasible = forward_mask & feasible
        forward_candidate_count = forward_mask.sum(dim=1).to(dtype=dtype)
        forward_feasible_count = forward_feasible.sum(dim=1).to(dtype=dtype)
        forward_infeasible_collision_count = torch.zeros_like(forward_feasible_count)
        forward_infeasible_margin_count = (forward_mask & (~margin_clear)).sum(dim=1).to(dtype=dtype)
        forward_infeasible_out_of_map_count = torch.zeros_like(forward_feasible_count)
        forward_cost = torch.where(forward_feasible, cost, torch.full_like(cost, float("inf")))
        best_forward_idx = torch.argmin(forward_cost, dim=1)
        fallback_batch_idx = torch.arange(n, device=device)
        has_forward_feasible = forward_feasible.any(dim=1)
        best_forward_feasible_cost = torch.where(
            has_forward_feasible,
            forward_cost[fallback_batch_idx, best_forward_idx],
            torch.full((n,), float("inf"), device=device, dtype=dtype),
        )
        best_forward_feasible_clearance = torch.where(
            has_forward_feasible,
            predicted_clearance[fallback_batch_idx, best_forward_idx],
            torch.full((n,), float("nan"), device=device, dtype=dtype),
        )
    else:
        forward_candidate_count = rollout["forward_candidate_count"]
        forward_feasible_count = rollout["forward_feasible_count"]
        forward_infeasible_collision_count = rollout["forward_infeasible_collision_count"]
        forward_infeasible_margin_count = rollout["forward_infeasible_margin_count"]
        forward_infeasible_out_of_map_count = rollout["forward_infeasible_out_of_map_count"]
        best_forward_feasible_cost = rollout["best_forward_feasible_cost"]
        best_forward_feasible_clearance = rollout["best_forward_feasible_clearance"]

    finite_cost = torch.isfinite(cost)
    feasible_any = feasible.any(dim=1)
    filter_mode = str(hparams["filter_mode"]).strip().lower()
    safe_mask = feasible & (risk <= float(hparams["risk_filter_threshold"]))
    safe_any = safe_mask.any(dim=1)
    if filter_mode == "none":
        selection_cost = torch.where(
            feasible,
            cost - float(hparams["clear_weight"]) * risk,
            torch.full_like(cost, float("inf")),
        )
    else:
        selection_cost = torch.where(feasible, cost, torch.full_like(cost, float("inf")))
    raw_cost = selection_cost
    raw_idx = torch.argmin(raw_cost, dim=1)
    safe_cost = torch.where(safe_mask, cost, torch.full_like(cost, float("inf")))
    best_safe_idx = torch.argmin(safe_cost, dim=1)
    safest_feasible_risk = torch.where(feasible, risk, torch.full_like(risk, float("inf")))
    safest_idx = torch.argmin(safest_feasible_risk, dim=1)
    fallback_mode = str(hparams["fallback"])
    relaxed_idx = safest_idx if fallback_mode == "safest" else raw_idx
    if filter_mode == "scale":
        after_idx = torch.where(safe_any, best_safe_idx, relaxed_idx)
    else:
        after_idx = raw_idx

    batch_idx = torch.arange(n, device=device)
    cmd_search_raw = torch.zeros_like(cmd_f)
    cmd_search_after_filter = torch.zeros_like(cmd_f)
    cmd_search_raw[:, :3] = cand_cmd_batch[batch_idx, raw_idx, :3]
    cmd_search_after_filter[:, :3] = cand_cmd_batch[batch_idx, after_idx, :3]
    stop_cmd = torch.zeros_like(cmd_search_after_filter)
    fallback_cmd = stop_cmd.clone()
    cmd_search_raw = torch.where(feasible_any.view(-1, 1), cmd_search_raw, stop_cmd)
    cmd_search_after_filter = torch.where(feasible_any.view(-1, 1), cmd_search_after_filter, stop_cmd)
    cmd_safe = cmd_search_after_filter.clone()

    nan = torch.full_like(ref, float("nan"))
    zero = torch.zeros_like(ref)
    raw_risk = torch.where(feasible_any, risk[batch_idx, raw_idx], torch.ones_like(ref))
    safe_risk = torch.where(feasible_any, risk[batch_idx, after_idx], torch.ones_like(ref))
    min_predicted_clearance = torch.where(
        feasible_any,
        torch.where(feasible, predicted_clearance, torch.full_like(predicted_clearance, float("inf"))).amin(dim=1),
        torch.zeros_like(ref),
    )
    best_cost = torch.where(feasible_any, cost[batch_idx, after_idx], torch.full_like(ref, float("inf")))
    actual_candidate_used = feasible_any
    current_clearance = env._compute_clearance_from_affordance(aff_map).to(device=device, dtype=dtype) - float(hparams["body_radius_m"])
    selected_rollout_min_clearance = torch.where(
        actual_candidate_used,
        predicted_clearance[batch_idx, after_idx],
        current_clearance,
    )
    selected_rollout_collision_flag = torch.where(
        actual_candidate_used,
        collision[batch_idx, after_idx].to(dtype=dtype),
        (current_clearance < 0.0).to(dtype=dtype),
    )
    selected_rollout_out_of_map_flag = torch.where(
        actual_candidate_used,
        out_of_map[batch_idx, after_idx].to(dtype=dtype),
        torch.zeros_like(ref),
    )
    filter_changed = (
        torch.norm(cmd_search_after_filter[:, :3] - cmd_search_raw[:, :3], dim=-1) > 1e-6
    ).to(dtype=dtype)
    no_feasible = ~feasible_any
    risk_threshold_relaxed = feasible_any & (~safe_any) & (filter_mode == "scale")
    invalid_cost = feasible_any & (~(finite_cost & feasible).any(dim=1))
    fallback_used = no_feasible.to(dtype=dtype)
    fallback_reason = torch.zeros_like(ref)
    fallback_reason = torch.where(risk_threshold_relaxed, torch.full_like(fallback_reason, 2.0), fallback_reason)
    fallback_reason = torch.where(no_feasible, torch.ones_like(fallback_reason), fallback_reason)
    fallback_reason = torch.where(invalid_cost, torch.full_like(fallback_reason, 5.0), fallback_reason)
    fallback_no_feasible = no_feasible.to(dtype=dtype)
    fallback_risk_threshold = risk_threshold_relaxed.to(dtype=dtype)
    fallback_collision = (no_feasible & collision.all(dim=1)).to(dtype=dtype)
    fallback_margin = (no_feasible & (~margin_clear).all(dim=1)).to(dtype=dtype)
    fallback_out_of_map = (no_feasible & out_of_map.all(dim=1)).to(dtype=dtype)
    fallback_invalid_cost = invalid_cost.to(dtype=dtype)
    compute_time_ms = torch.full_like(ref, 1000.0 * (time.perf_counter() - t0))
    return {
        "cmd": cmd_safe,
        "cmd_f": cmd_f,
        "cmd_a": diag["cmd_a_eff"],
        "gate_y_raw": nan,
        "gate_y": nan,
        "gate_y_safe": nan,
        "y_eff": nan,
        "w": nan,
        "signed_w": nan,
        "signed_w_active": zero,
        "w_support_correction": zero,
        "risk_diff_correction": zero,
        "delta_y_w_raw": nan,
        "delta_y_w_used": nan,
        "delta_y_r": nan,
        "delta_y_total": nan,
        "risk_memory": zero,
        "cmd_search_raw": cmd_search_raw,
        "cmd_search_after_filter": cmd_search_after_filter,
        "cmd_safe": cmd_safe,
        "velocity_search_raw_risk": raw_risk,
        "velocity_search_safe_risk": safe_risk,
        "velocity_search_filter_changed": filter_changed,
        "velocity_search_safe_available": safe_any.to(dtype=dtype),
        "velocity_search_compute_time_ms": compute_time_ms,
        "velocity_search_dynamic_window_enabled": torch.full_like(ref, 1.0 if dynamic_window_enabled else 0.0),
        "velocity_search_candidate_count_before_dynamic_window": dynamic_window_before_count,
        "velocity_search_candidate_count_after_dynamic_window": dynamic_window_after_count,
        "velocity_search_dynamic_window_rejected_count": dynamic_window_rejected_count,
        "velocity_search_candidate_count": candidate_count,
        "velocity_search_feasible_count": feasible_count,
        "velocity_search_infeasible_count": infeasible_count,
        "velocity_search_feasible_count_after_margin": feasible_count,
        "velocity_search_forward_candidate_count": forward_candidate_count,
        "velocity_search_forward_feasible_count": forward_feasible_count,
        "velocity_search_forward_infeasible_collision_count": forward_infeasible_collision_count,
        "velocity_search_forward_infeasible_margin_count": forward_infeasible_margin_count,
        "velocity_search_forward_infeasible_out_of_map_count": forward_infeasible_out_of_map_count,
        "velocity_search_best_forward_feasible_cost": best_forward_feasible_cost,
        "velocity_search_best_forward_feasible_clearance": best_forward_feasible_clearance,
        "velocity_search_fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "fallback_no_feasible": fallback_no_feasible,
        "fallback_risk_threshold": fallback_risk_threshold,
        "fallback_collision": fallback_collision,
        "fallback_margin": fallback_margin,
        "fallback_out_of_map": fallback_out_of_map,
        "fallback_invalid_cost": fallback_invalid_cost,
        "velocity_search_min_predicted_clearance": min_predicted_clearance,
        "velocity_search_best_cost": best_cost,
        "selected_rollout_min_clearance": selected_rollout_min_clearance,
        "selected_rollout_collision_flag": selected_rollout_collision_flag,
        "selected_rollout_out_of_map_flag": selected_rollout_out_of_map_flag,
        "fallback_cmd": fallback_cmd,
        **diag,
    }


def _compute_moe_follow_cmd_from_goal(
    state_tensor: torch.Tensor,
    goal_tensor: torch.Tensor,
    reset_mask: Optional[torch.Tensor],
    cmd_scale: Tuple[float, float, float],
    env_ref=None,
) -> torch.Tensor:
    if state_tensor.ndim != 2 or state_tensor.shape[1] < 3:
        raise ValueError(f"state_tensor shape invalid for analytic follow expert: {tuple(state_tensor.shape)}")
    if goal_tensor.ndim != 2 or goal_tensor.shape[1] < 2:
        raise ValueError(f"goal_tensor shape invalid for analytic follow expert: {tuple(goal_tensor.shape)}")
    robot_pos_world_xy, robot_heading, target_world_xy = th.get_follow_expert_world_inputs(
        env_ref, state_tensor, goal_tensor
    )
    return s0_follow_expert_fn(
        robot_pos_world_xy=robot_pos_world_xy,
        robot_heading=robot_heading,
        target_world_xy=target_world_xy,
        target_vel_world_xy=None,
        target_heading=None,
        cmd_scale=cmd_scale,
        reset_mask=reset_mask,
    )


def _setup_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EpisodeAccumulator:
    step_hl: int = 0
    success: bool = False
    t_success_s: float = float("nan")
    t_collision_s: float = float("nan")
    follow_err_sum: float = 0.0
    follow_err_sq_sum: float = 0.0
    follow_err_count: int = 0
    energy_j: float = 0.0
    distance_m: float = 0.0
    cross_line_dist_end: float = float("nan")
    cross_line_dist_min: float = float("inf")
    episode_collision: bool = False
    progress_reached: bool = False
    progress_ratio_best: float = 0.0
    gate_y_raw_sum: float = 0.0
    y_eff_sum: float = 0.0
    w_sum: float = 0.0
    signed_w_sum: float = 0.0
    signed_w_active_sum: float = 0.0
    clearance_f_sum: float = 0.0
    clearance_a_sum: float = 0.0
    risk_f_sum: float = 0.0
    risk_a_sum: float = 0.0
    clearance_rollout_f_sum: float = 0.0
    clearance_rollout_a_sum: float = 0.0
    clearance_rollout_s_sum: float = 0.0
    risk_rollout_f_sum: float = 0.0
    risk_rollout_a_sum: float = 0.0
    risk_rollout_s_sum: float = 0.0
    risk_rollout_gap_f_min_as_sum: float = 0.0
    w_support_correction_sum: float = 0.0
    risk_diff_correction_sum: float = 0.0
    delta_y_w_raw_sum: float = 0.0
    delta_y_w_used_sum: float = 0.0
    delta_y_r_sum: float = 0.0
    delta_y_total_sum: float = 0.0
    risk_memory_sum: float = 0.0
    rule_s_sum: float = 0.0
    rule_risk_gap_sum: float = 0.0
    rule_follow_scale_sum: float = 0.0
    rule_yaw_scale_sum: float = 0.0
    rule_follow_suppression_sum: float = 0.0
    rule_avoid_conflict_s_sum: float = 0.0
    rule_avoid_conflict_follow_scale_sum: float = 0.0
    rule_avoid_conflict_yaw_scale_sum: float = 0.0
    rule_avoid_conflict_follow_suppression_sum: float = 0.0
    row_not_released_sum: float = 0.0
    row_not_released_w_sum: float = 0.0
    row_not_released_steps: int = 0
    row_released_w_sum: float = 0.0
    row_released_steps: int = 0
    near_miss_steps: int = 0
    gate_switch_count: int = 0
    cmd_jerk_lin_sum: float = 0.0
    cmd_jerk_ang_sum: float = 0.0
    cmd_post_delta_abs_sum: float = 0.0
    cmd_post_delta_x_abs_sum: float = 0.0
    cmd_post_delta_y_abs_sum: float = 0.0
    cmd_post_delta_w_abs_sum: float = 0.0
    cmd_post_delta_y_sum: float = 0.0
    cmd_post_rewrite_steps: int = 0
    cmd_post_samples: int = 0
    velocity_search_steps: int = 0
    velocity_search_filter_change_steps: int = 0
    velocity_search_safe_available_steps: int = 0
    velocity_search_raw_risk_sum: float = 0.0
    velocity_search_safe_risk_sum: float = 0.0
    velocity_search_compute_time_ms_sum: float = 0.0
    velocity_search_dynamic_window_enabled_steps: int = 0
    velocity_search_candidate_count_before_dynamic_window_sum: float = 0.0
    velocity_search_candidate_count_after_dynamic_window_sum: float = 0.0
    velocity_search_dynamic_window_rejected_count_sum: float = 0.0
    velocity_search_candidate_count_sum: float = 0.0
    velocity_search_feasible_count_sum: float = 0.0
    velocity_search_infeasible_count_sum: float = 0.0
    velocity_search_feasible_count_after_margin_sum: float = 0.0
    velocity_search_selected_v_lat_sum: float = 0.0
    velocity_search_selected_v_fwd_sum: float = 0.0
    velocity_search_selected_yaw_sum: float = 0.0
    velocity_search_raw_y_sum: float = 0.0
    velocity_search_after_filter_y_sum: float = 0.0
    velocity_search_cmd_safe_y_sum: float = 0.0
    velocity_search_cmd_safe_y_over_raw_y_sum: float = 0.0
    velocity_search_cmd_safe_y_over_raw_y_steps: int = 0
    velocity_search_forward_candidate_count_sum: float = 0.0
    velocity_search_forward_feasible_count_sum: float = 0.0
    velocity_search_forward_infeasible_collision_count_sum: float = 0.0
    velocity_search_forward_infeasible_margin_count_sum: float = 0.0
    velocity_search_forward_infeasible_out_of_map_count_sum: float = 0.0
    velocity_search_best_forward_feasible_cost_sum: float = 0.0
    velocity_search_best_forward_feasible_clearance_sum: float = 0.0
    velocity_search_best_forward_feasible_steps: int = 0
    actual_base_vel_x_sum: float = 0.0
    actual_base_vel_y_sum: float = 0.0
    actual_base_vel_w_sum: float = 0.0
    actual_base_delta_x_sum: float = 0.0
    actual_base_delta_y_sum: float = 0.0
    row_progress_delta_sum: float = 0.0
    actual_motion_steps: int = 0
    velocity_search_fallback_steps: int = 0
    fallback_no_feasible_steps: int = 0
    fallback_risk_threshold_steps: int = 0
    fallback_collision_steps: int = 0
    fallback_margin_steps: int = 0
    fallback_out_of_map_steps: int = 0
    fallback_invalid_cost_steps: int = 0
    velocity_search_min_predicted_clearance_sum: float = 0.0
    velocity_search_best_cost_sum: float = 0.0
    selected_rollout_min_clearance_sum: float = 0.0
    selected_rollout_collision_steps: int = 0
    selected_rollout_out_of_map_steps: int = 0
    rotate_only_steps: int = 0
    w_trigger_step: int = -1
    w_trigger_progress: float = float("nan")
    gate_region_steps: int = 0
    gate_region_y_eff_sum: float = 0.0
    gate_region_near_miss_steps: int = 0
    high_risk_y_eff_sum: float = 0.0
    high_risk_w_sum: float = 0.0
    high_risk_risk_f_sum: float = 0.0
    high_risk_risk_a_sum: float = 0.0
    high_risk_near_miss_steps: int = 0
    risk_bin_stats: list = field(default_factory=_empty_risk_bin_state)
    conflict_bin_stats: list = field(default_factory=_empty_risk_bin_state)
    priv_conflict_bin_stats: list = field(default_factory=_empty_risk_bin_state)
    priv_conflict_score_sum: float = 0.0
    priv_conflict_steps: int = 0
    priv_obstacle_window_steps: int = 0
    priv_follow_pressure_steps: int = 0
    priv_avoid_pressure_steps: int = 0
    priv_conflict_y_raw_sum: float = 0.0
    priv_conflict_y_eff_sum: float = 0.0
    priv_conflict_w_sum: float = 0.0
    priv_conflict_signed_w_sum: float = 0.0
    priv_conflict_delta_y_sum: float = 0.0
    priv_non_conflict_steps: int = 0
    priv_non_conflict_delta_y_sum: float = 0.0
    priv_window_phase_approach_steps: int = 0
    priv_window_phase_inside_steps: int = 0
    priv_window_phase_release_steps: int = 0
    priv_conflict_phase_approach_steps: int = 0
    priv_conflict_phase_inside_steps: int = 0
    priv_conflict_phase_release_steps: int = 0
    priv_conflict_phase_approach_w_sum: float = 0.0
    priv_conflict_phase_inside_w_sum: float = 0.0
    priv_conflict_phase_release_w_sum: float = 0.0
    priv_conflict_phase_approach_signed_w_sum: float = 0.0
    priv_conflict_phase_inside_signed_w_sum: float = 0.0
    priv_conflict_phase_release_signed_w_sum: float = 0.0
    priv_conflict_phase_approach_delta_y_sum: float = 0.0
    priv_conflict_phase_inside_delta_y_sum: float = 0.0
    priv_conflict_phase_release_delta_y_sum: float = 0.0
    unsafe_conflict_steps: int = 0
    unsafe_follow_risk_steps: int = 0
    unsafe_safe_candidate_better_steps: int = 0
    unsafe_avoid_safer_steps: int = 0
    unsafe_stop_safer_steps: int = 0
    unsafe_command_disagree_steps: int = 0
    unsafe_target_recoverable_steps: int = 0
    unsafe_conflict_y_raw_sum: float = 0.0
    unsafe_conflict_y_eff_sum: float = 0.0
    unsafe_conflict_w_sum: float = 0.0
    unsafe_conflict_signed_w_sum: float = 0.0
    unsafe_conflict_delta_y_sum: float = 0.0
    unsafe_conflict_delta_y_w_raw_sum: float = 0.0
    unsafe_conflict_delta_y_w_used_sum: float = 0.0
    unsafe_conflict_delta_y_r_sum: float = 0.0
    unsafe_conflict_delta_y_total_sum: float = 0.0
    unsafe_non_conflict_steps: int = 0
    unsafe_non_conflict_delta_y_sum: float = 0.0
    unsafe_conflict_phase_approach_steps: int = 0
    unsafe_conflict_phase_inside_steps: int = 0
    unsafe_conflict_phase_release_steps: int = 0
    unsafe_conflict_phase_approach_w_sum: float = 0.0
    unsafe_conflict_phase_inside_w_sum: float = 0.0
    unsafe_conflict_phase_release_w_sum: float = 0.0
    unsafe_conflict_phase_approach_signed_w_sum: float = 0.0
    unsafe_conflict_phase_inside_signed_w_sum: float = 0.0
    unsafe_conflict_phase_release_signed_w_sum: float = 0.0
    unsafe_conflict_phase_approach_delta_y_sum: float = 0.0
    unsafe_conflict_phase_inside_delta_y_sum: float = 0.0
    unsafe_conflict_phase_release_delta_y_sum: float = 0.0
    avoid_conflict_steps: int = 0
    avoid_conflict_y_raw_sum: float = 0.0
    avoid_conflict_y_eff_sum: float = 0.0
    avoid_conflict_w_sum: float = 0.0
    avoid_conflict_signed_w_sum: float = 0.0
    avoid_conflict_delta_y_sum: float = 0.0
    avoid_conflict_delta_y_w_raw_sum: float = 0.0
    avoid_conflict_delta_y_w_used_sum: float = 0.0
    avoid_conflict_delta_y_r_sum: float = 0.0
    avoid_conflict_delta_y_total_sum: float = 0.0
    avoid_conflict_cmd_y_sum: float = 0.0
    avoid_conflict_cmd_y_steps: int = 0
    avoid_conflict_cmd_f_y_sum: float = 0.0
    avoid_conflict_cmd_f_y_steps: int = 0
    avoid_conflict_cmd_a_y_sum: float = 0.0
    avoid_conflict_cmd_a_y_steps: int = 0
    stop_conflict_steps: int = 0
    stop_conflict_y_raw_sum: float = 0.0
    stop_conflict_y_eff_sum: float = 0.0
    stop_conflict_w_sum: float = 0.0
    stop_conflict_signed_w_sum: float = 0.0
    stop_conflict_delta_y_sum: float = 0.0
    target_bearing_abs_sum: float = 0.0
    target_bearing_abs_max: float = 0.0
    target_bearing_abs_samples: list = field(default_factory=list)
    target_in_fov_steps: int = 0
    target_near_fov_edge_steps: int = 0
    target_lost_steps: int = 0
    target_lost_current_steps: int = 0
    target_lost_max_consecutive_steps: int = 0
    target_lost_event: bool = False
    target_conflict_bearing_abs_sum: float = 0.0
    target_conflict_bearing_abs_max: float = 0.0
    target_conflict_in_fov_steps: int = 0
    target_conflict_near_fov_edge_steps: int = 0
    target_conflict_lost_steps: int = 0
    cmd_x_sum: float = 0.0
    cmd_x_sq_sum: float = 0.0
    cmd_x_abs_sum: float = 0.0
    cmd_x_positive_steps: int = 0
    cmd_x_negative_steps: int = 0
    cmd_x_zero_steps: int = 0
    cmd_x_samples: int = 0
    goal_cmd_corr_steps: int = 0
    goal_x_sum: float = 0.0
    goal_y_sum: float = 0.0
    corr_cmd_x_sum: float = 0.0
    corr_cmd_y_sum: float = 0.0
    corr_cmd_yaw_sum: float = 0.0
    goal_x_sq_sum: float = 0.0
    goal_y_sq_sum: float = 0.0
    corr_cmd_x_sq_sum: float = 0.0
    corr_cmd_y_sq_sum: float = 0.0
    corr_cmd_yaw_sq_sum: float = 0.0
    goalx_cmdx_sum: float = 0.0
    goalx_cmdyaw_sum: float = 0.0
    goaly_cmdy_sum: float = 0.0
    goal_x_left_steps: int = 0
    goal_x_right_steps: int = 0
    cmd_yaw_left_sum: float = 0.0
    cmd_yaw_right_sum: float = 0.0
    cmd_x_left_sum: float = 0.0
    cmd_x_right_sum: float = 0.0
    prev_robot_x: float = float("nan")
    prev_robot_y: float = float("nan")
    prev_robot_heading: float = float("nan")
    prev_progress_val: float = float("nan")
    prev_y_eff: Optional[float] = None
    prev_cmd_final: Optional[list] = None
    timeseries: list = field(default_factory=list)
    cmd_f_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cmd_a_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cmd_final_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


class EvalRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _setup_seed(args.seed)
        if not hasattr(self.args, "teacher_ckpt") or getattr(self.args, "teacher_ckpt", None) is None:
            self.args.teacher_ckpt = getattr(self.args, "ckpt", None)
        runtime = ph.build_play_runtime_for_eval(self.args, self.device)
        self.args = runtime.args
        self.env = runtime.env

        self.aff_stack = max(int(getattr(self.args, "aff_stack", 1)), 1)
        self.is_mono_ppo = bool(getattr(self.args, "mono_ppo", False))
        self.aff_stack_buf = None
        self.follow_aff_stack_buf = None
        self.avoid_aff_stack_buf = None
        self.done_prev = None
        self.velocity_search_hparams = (
            _resolve_velocity_search_hparams(self.args)
            if bool(getattr(self.args, "velocity_search", False))
            else {}
        )

        self.mass_kg = self._estimate_robot_mass_kg()
        self.g = 9.81

        self.policy = runtime.policy
        self.avoid_model = runtime.avoid_policy
        self.vision_model = runtime.vision_model
        self.primary_meta = runtime.primary_meta
        self.policy_meta = runtime.policy_meta
        self.aux_checkpoint_meta = runtime.aux_checkpoint_meta
        selected_w_mode = getattr(args, "_eval_selected_w_mode", None)
        if selected_w_mode is not None and str(getattr(self.args, "w_mode", "")) != str(selected_w_mode):
            raise ValueError(
                f"eval_w_mode 被 checkpoint meta 改写: requested={selected_w_mode}, runtime={self.args.w_mode}"
            )
        self.gate_state_dim = runtime.gate_state_dim
        self.gate_goal_dim = getattr(runtime, "gate_goal_dim", None)
        self.avoid_state_dim = runtime.avoid_state_dim

        self.param_info = self._build_param_info()
        self.resolved_protocol = th.build_resolved_protocol(
            self.args,
            self.env,
            primary_ckpt_path=getattr(self.args, "teacher_ckpt", getattr(self.args, "pcr_ckpt", None)),
            primary_meta=self.policy_meta if isinstance(self.policy_meta, dict) else self.primary_meta,
            aux_sources=self.aux_checkpoint_meta,
        )
        self._layout_debug_exported = False

    def _export_eval_layout_debug_once(self) -> None:
        out_dir = str(getattr(self.args, "export_eval_layout_debug", "") or "").strip()
        if not out_dir or self._layout_debug_exported:
            return
        self._layout_debug_exported = True
        env_impl = getattr(self.env, "env", None)
        if (
            env_impl is None
            or not hasattr(env_impl, "s_avoid_active")
            or not hasattr(env_impl, "s_avoid_pos_world")
        ):
            print("[Eval] layout export skipped: s_avoid obstacle state unavailable", flush=True)
            return
        active = env_impl.s_avoid_active
        pos_world = env_impl.s_avoid_pos_world
        quat_world = getattr(env_impl, "s_avoid_quat_world", None)
        if not torch.is_tensor(active) or not torch.is_tensor(pos_world) or active.shape[0] <= 0:
            print("[Eval] layout export skipped: invalid s_avoid obstacle tensors", flush=True)
            return
        os.makedirs(out_dir, exist_ok=True)
        env_id = 0
        origin = torch.zeros(3, device=pos_world.device, dtype=pos_world.dtype)
        if hasattr(env_impl, "env_origins") and torch.is_tensor(env_impl.env_origins):
            origin = env_impl.env_origins[env_id, :3]
        terrain_cfg = getattr(getattr(env_impl, "cfg", None), "terrain", None)
        cap_slots = int(getattr(env_impl, "s_avoid_capsule_slot_count", 0))
        box_slots = int(getattr(env_impl, "s_avoid_box_slot_count", 0))
        slot_count = int(active.shape[1])
        cap_radius = float(getattr(terrain_cfg, "avoid_capsule_radius", 0.15))
        box_x = float(getattr(terrain_cfg, "avoid_box_size_x", 0.4))
        box_y = float(getattr(terrain_cfg, "avoid_box_size_y", 0.4))
        rows = []
        for slot in range(slot_count):
            if not bool(active[env_id, slot].item()):
                continue
            p_world = pos_world[env_id, slot, :3]
            p_local = p_world - origin
            kind = "capsule" if slot < cap_slots else ("box" if slot < cap_slots + box_slots else "wall")
            yaw = 0.0
            if torch.is_tensor(quat_world):
                qz = _safe_float(quat_world[env_id, slot, 2].item())
                qw = _safe_float(quat_world[env_id, slot, 3].item(), 1.0)
                yaw = 2.0 * math.atan2(qz, qw)
            rows.append(
                {
                    "slot": int(slot),
                    "type": kind,
                    "x_local": _safe_float(p_local[0].item()),
                    "y_local": _safe_float(p_local[1].item()),
                    "z_local": _safe_float(p_local[2].item()),
                    "x_world": _safe_float(p_world[0].item()),
                    "y_world": _safe_float(p_world[1].item()),
                    "z_world": _safe_float(p_world[2].item()),
                    "radius": cap_radius if kind == "capsule" else "",
                    "size_x": box_x if kind == "box" else "",
                    "size_y": box_y if kind == "box" else "",
                    "yaw_deg": math.degrees(yaw),
                }
            )
        csv_path = os.path.join(out_dir, "heldout_irregular_rows_layout.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "slot", "type", "x_local", "y_local", "z_local",
                "x_world", "y_world", "z_world", "radius", "size_x", "size_y", "yaw_deg",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, Rectangle
            from matplotlib.transforms import Affine2D

            fig, ax = plt.subplots(figsize=(4.2, 7.0))
            for row in rows:
                x = float(row["x_local"])
                y = float(row["y_local"])
                if row["type"] == "box":
                    sx = float(row["size_x"])
                    sy = float(row["size_y"])
                    yaw_deg = float(row["yaw_deg"])
                    rect = Rectangle(
                        (x - 0.5 * sx, y - 0.5 * sy),
                        sx,
                        sy,
                        facecolor="#bdbdbd",
                        edgecolor="#6f6f6f",
                        linewidth=1.0,
                        alpha=0.85,
                    )
                    rect.set_transform(Affine2D().rotate_deg_around(x, y, yaw_deg) + ax.transData)
                    ax.add_patch(rect)
                else:
                    ax.add_patch(
                        Circle(
                            (x, y),
                            float(row["radius"] or cap_radius),
                            facecolor="#c9c9c9",
                            edgecolor="#707070",
                            linewidth=1.0,
                            alpha=0.9,
                        )
                    )
            row_y = []
            if hasattr(env_impl, "_get_s_avoid_fixed_stage_row_y"):
                row_y = list(env_impl._get_s_avoid_fixed_stage_row_y(4))
            for y in row_y:
                ax.axhline(float(y), color="#e0e0e0", linewidth=0.8, zorder=0)
            ax.set_title("held-out irregular rows")
            ax.set_xlabel("local x / lateral [m]")
            ax.set_ylabel("local y / forward [m]")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.2, 6.3)
            ax.grid(True, color="#eeeeee", linewidth=0.7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "heldout_irregular_rows_layout.png"), dpi=220)
            plt.close(fig)
        except Exception as exc:
            print(f"[Eval] layout PNG export failed: {exc}", flush=True)
        print(f"[Eval] layout export: {out_dir}", flush=True)

    def _estimate_robot_mass_kg(self) -> float:
        try:
            env_impl = self.env.env
            body_props = env_impl.gym.get_actor_rigid_body_properties(env_impl.envs[0], env_impl.actor_handles[0])
            mass = float(sum(float(p.mass) for p in body_props))
            if not np.isfinite(mass) or mass <= 0.0:
                return 15.0
            return mass
        except Exception:
            return 15.0

    def _ckpt_meta(self, ckpt_obj) -> Optional[Dict]:
        if isinstance(ckpt_obj, dict):
            meta = ckpt_obj.get("experiment_meta", None)
            if isinstance(meta, dict):
                return meta
        return None

    def _validate_ckpt_meta(self, ckpt_meta: Optional[Dict], *, expected_skill: str, source_name: str) -> None:
        if not isinstance(ckpt_meta, dict):
            return
        meta_skill = ckpt_meta.get("skill", None)
        if meta_skill is not None and meta_skill != expected_skill:
            raise ValueError(f"{source_name} 的 skill 与当前评测不一致: checkpoint={meta_skill}, expected={expected_skill}")
        meta_mode = ckpt_meta.get("mode", None)
        eval_mode = getattr(self.args, "mode", "teacher")
        if expected_skill != "moe" and meta_mode is not None and meta_mode != eval_mode:
            raise ValueError(f"{source_name} 的 mode 与当前评测不一致: checkpoint={meta_mode}, expected={eval_mode}")

    def _load_models(self) -> None:
        obs = self.env.reset()
        state_dim = int(obs["state"].shape[1])
        goal_dim = int(obs["goal"].shape[1])
        skill = getattr(self.args, "skill", "follow")
        if skill in ("avoid", "moe"):
            aff_shape = obs["local_map_2ch"].shape[1:]
        else:
            aff_shape = obs["gt_affordance"].shape[1:]
        aff_channels = int(aff_shape[0] * self.aff_stack)
        cmd_scale = tuple(float(v) for v in self.env.post_processor.max_cmd.detach().cpu().tolist())

        mode = getattr(self.args, "mode", "teacher")
        if mode == "student" and skill == "moe":
            raise ValueError("当前未实现 Gate 的 student 评测契约，禁止 --mode student --skill moe。")

        if mode == "student":
            if not self.args.vision_ckpt:
                raise ValueError("Student mode requires --vision_ckpt")
            self.vision_model = th.AffordanceEstimator(
                depth_channels=1,
                output_size=th.get_vision_native_output_size(),
                max_depth_range=5.0,
            ).to(self.device)
            ckpt = torch.load(self.args.vision_ckpt, map_location=self.device)
            vision_meta = self._ckpt_meta(ckpt)
            th.validate_vision_runtime_contract(
                self.args,
                self.env,
                source_name="Eval vision checkpoint",
                ckpt_meta=vision_meta,
                strict_meta=True,
            )
            self.vision_model.load_state_dict(_to_state_dict(ckpt))
            self.vision_model.eval()
            self.aux_checkpoint_meta["vision_ckpt"] = {
                "path": os.path.abspath(self.args.vision_ckpt),
                "experiment_meta": vision_meta,
            }

        if skill == "moe":
            if not self.args.ckpt:
                raise ValueError("MoE mode requires --pcr_ckpt for gate policy")
            if not self.args.avoid_ckpt:
                raise ValueError("MoE mode requires --avoid_ckpt; follow side uses analytic expert")

            gate_ckpt = torch.load(self.args.ckpt, map_location=self.device)
            self.policy_meta = self._ckpt_meta(gate_ckpt)
            self.gate_state_dim = th.infer_checkpoint_state_dim(gate_ckpt) or state_dim
            gate_action_dim = th.infer_checkpoint_gate_action_dim(gate_ckpt, self.policy_meta)
            expected_action_dim = 2 if th.is_learned_w_mode(self.args.w_mode) else 1
            if gate_action_dim is not None and int(gate_action_dim) != expected_action_dim:
                raise ValueError(
                    f"gate ckpt actor_output_dim 与当前 eval_w_mode 不一致: "
                    f"checkpoint={gate_action_dim}, expected={expected_action_dim}, w_mode={self.args.w_mode}"
                )
            self.gate_goal_dim = th.infer_checkpoint_goal_dim(gate_ckpt) or (
                goal_dim + (th.LEARNED_W_FEATURE_DIM if expected_action_dim == 2 else 0)
            )
            self.policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=self.gate_state_dim,
                goal_dim=self.gate_goal_dim,
                learned_w=expected_action_dim == 2,
            ).to(self.device)
            self._validate_ckpt_meta(self.policy_meta, expected_skill="moe", source_name="gate ckpt")
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(self.args, self.env),
                self.policy_meta,
                reference_name="current eval runtime",
                candidate_name="gate ckpt",
                strict=True,
            )
            th.load_high_level_state_dict_compat(self.policy, _to_state_dict(gate_ckpt), label="eval_gate")
            self.policy.eval()

            avoid_ckpt = torch.load(self.args.avoid_ckpt, map_location=self.device)
            avoid_meta = self._ckpt_meta(avoid_ckpt)
            self.avoid_state_dim = th.infer_checkpoint_state_dim(avoid_ckpt) or state_dim
            avoid_aff_channels = int(obs["local_map_2ch"].shape[1] * self.aff_stack)
            self.avoid_model = th.CmdVelExpert(
                affordance_channels=avoid_aff_channels,
                state_dim=self.avoid_state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(self.device)
            self._validate_ckpt_meta(avoid_meta, expected_skill="avoid", source_name="avoid expert ckpt")
            th.validate_checkpoint_contract_compatibility(
                self.policy_meta,
                avoid_meta,
                reference_name="gate ckpt",
                candidate_name="avoid expert ckpt",
                strict=True,
            )
            th.load_high_level_state_dict_compat(
                self.avoid_model,
                _to_state_dict(avoid_ckpt),
                label="eval_avoid",
            )
            self.avoid_model.eval()
            self.aux_checkpoint_meta["avoid_ckpt"] = {
                "path": os.path.abspath(self.args.avoid_ckpt),
                "experiment_meta": avoid_meta,
            }
        else:
            if not self.args.ckpt:
                raise ValueError("Follow/Avoid mode requires --pcr_ckpt")
            ckpt = torch.load(self.args.ckpt, map_location=self.device)
            self.policy_meta = self._ckpt_meta(ckpt)
            self.gate_state_dim = th.infer_checkpoint_state_dim(ckpt) or state_dim
            self.policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=self.gate_state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(self.device)
            self._validate_ckpt_meta(self.policy_meta, expected_skill=skill, source_name="policy ckpt")
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(self.args, self.env),
                self.policy_meta,
                reference_name="current eval runtime",
                candidate_name="policy ckpt",
                strict=True,
            )
            th.load_high_level_state_dict_compat(self.policy, _to_state_dict(ckpt), label="eval_policy")
            self.policy.eval()

    def _build_param_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        if self.policy is not None:
            total, trainable = _count_params(self.policy)
            info["policy_total"] = total
            info["policy_trainable"] = trainable

        if self.avoid_model is not None and not self.is_mono_ppo:
            total, trainable = _count_params(self.avoid_model)
            info["avoid_total"] = total
            info["avoid_trainable"] = trainable
        elif self.avoid_model is not None and self.is_mono_ppo:
            total, trainable = _count_params(self.avoid_model)
            info["diagnostic_avoid_total"] = total
            info["diagnostic_avoid_trainable"] = trainable

        if self.vision_model is not None:
            total, trainable = _count_params(self.vision_model)
            info["vision_total"] = total
            info["vision_trainable"] = trainable

        info["high_level_total"] = int(
            info.get("policy_total", 0)
            + info.get("follow_total", 0)
            + info.get("avoid_total", 0)
            + info.get("vision_total", 0)
        )
        info["high_level_trainable"] = int(
            info.get("policy_trainable", 0)
            + info.get("follow_trainable", 0)
            + info.get("avoid_trainable", 0)
            + info.get("vision_trainable", 0)
        )
        return info

    def _build_affordance_bundle(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return ph.compute_play_affordance_bundle(self.args, self.env, obs_dict, self.vision_model)

    def _roll_aff_stack(
        self,
        stack_buf: Optional[torch.Tensor],
        aff_map: torch.Tensor,
        done_prev: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if stack_buf is None:
            return aff_map.repeat(1, self.aff_stack, 1, 1)
        if done_prev is not None and done_prev.any():
            stack_buf = stack_buf.clone()
            stack_buf[done_prev] = aff_map[done_prev].repeat(1, self.aff_stack, 1, 1)
        stack_buf = torch.roll(stack_buf, shifts=-aff_map.shape[1], dims=1)
        stack_buf[:, -aff_map.shape[1] :, :, :] = aff_map
        return stack_buf

    def _policy_step(
        self,
        obs_dict: Dict[str, torch.Tensor],
        aff_stack: torch.Tensor,
        difficulty: torch.Tensor,
        *,
        follow_aff_stack: Optional[torch.Tensor] = None,
        follow_difficulty: Optional[torch.Tensor] = None,
        avoid_aff_stack: Optional[torch.Tensor] = None,
        avoid_difficulty: Optional[torch.Tensor] = None,
        gate_aff_map: Optional[torch.Tensor] = None,
    ):
        state = obs_dict["state"]
        policy_goal = th.get_policy_goal_tensor(obs_dict, self.args.skill)
        goal = torch.zeros_like(policy_goal) if bool(getattr(self.args, "zero_goal", False)) else policy_goal
        avoid_goal = torch.zeros_like(obs_dict["goal"]) if bool(getattr(self.args, "zero_goal", False)) else obs_dict["goal"]
        policy_aff_stack = torch.zeros_like(aff_stack) if bool(getattr(self.args, "zero_local_map", False)) else aff_stack
        difficulty_input = torch.zeros_like(difficulty) if bool(getattr(self.args, "zero_local_map", False)) else difficulty

        if self.args.skill == "moe" and not self.is_mono_ppo:
            if avoid_aff_stack is None or avoid_difficulty is None or gate_aff_map is None:
                raise ValueError("MoE eval requires avoid affordance inputs and gate affordance map.")
            avoid_aff_input = (
                torch.zeros_like(avoid_aff_stack)
                if bool(getattr(self.args, "zero_local_map", False))
                else avoid_aff_stack
            )
            gate_aff_input = (
                torch.zeros_like(gate_aff_map)
                if bool(getattr(self.args, "zero_local_map", False))
                else gate_aff_map
            )
            avoid_difficulty_input = (
                torch.zeros_like(avoid_difficulty)
                if bool(getattr(self.args, "zero_local_map", False))
                else avoid_difficulty
            )
            target_avoid_state_dim = self.avoid_state_dim or int(state.shape[1])
            target_gate_state_dim = self.gate_state_dim or int(state.shape[1])
            expert_state = th.get_moe_expert_state_inputs(
                th.match_state_dim(state, target_avoid_state_dim, label="eval_expert_state")
            )
            gate_state = th.match_state_dim(state, target_gate_state_dim, label="eval_gate_state")
            with torch.no_grad():
                cmd_f = ph._compute_moe_follow_cmd_from_goal(
                    expert_state,
                    goal,
                    self.done_prev,
                    tuple(float(v) for v in self.env.post_processor.max_cmd.detach().cpu().tolist()),
                    env_ref=self.env,
                )
                cmd_a, _ = self.avoid_model.get_action(
                    avoid_aff_input,
                    expert_state,
                    avoid_goal,
                    avoid_difficulty_input,
                    deterministic=True,
                )
                if bool(getattr(self.args, "velocity_search", False)):
                    gate_diag = _compute_velocity_search_diag(
                        self.env,
                        self.args,
                        gate_aff_input,
                        cmd_f,
                        cmd_a,
                        goal,
                        self.velocity_search_hparams,
                    )
                    return gate_diag["cmd"], None, gate_diag
                if bool(getattr(self.args, "additive_fusion", False)):
                    gate_diag = _compute_additive_fusion_diag(
                        self.env,
                        self.args,
                        gate_aff_input,
                        cmd_f,
                        cmd_a,
                    )
                    return gate_diag["cmd"], None, gate_diag
                gate_difficulty = difficulty if self.args.gate_use_difficulty else torch.zeros_like(difficulty)
                if bool(getattr(self.args, "zero_local_map", False)):
                    gate_difficulty = torch.zeros_like(gate_difficulty)
                gate_policy_goal = goal
                if th.is_learned_w_mode(self.args.w_mode):
                    gate_policy_goal, _ = th.build_learned_w_gate_goal(
                        self.env,
                        self.args,
                        goal,
                        gate_aff_input,
                        cmd_f,
                        cmd_a,
                        update_risk_memory=True,
                        state_tensor=gate_state,
                    )
                if getattr(self, "gate_goal_dim", None) is not None:
                    gate_policy_goal = th.match_goal_dim(
                        gate_policy_goal,
                        int(self.gate_goal_dim),
                        label="eval_gate_goal",
                    )
                gate_action, _ = self.policy.get_action(
                    policy_aff_stack,
                    gate_state,
                    gate_policy_goal,
                    gate_difficulty,
                    deterministic=not self.args.stochastic,
                )
                if th.is_learned_w_mode(self.args.w_mode):
                    gate_y_raw = gate_action[:, 0]
                    learned_w = gate_action[:, 1]
                else:
                    gate_y_raw = gate_action
                    learned_w = None
                gate_diag = th.resolve_moe_gate_pcr(
                    self.env,
                    self.args,
                    gate_aff_input,
                    gate_y_raw,
                    cmd_f,
                    cmd_a,
                    learned_w=learned_w,
                )
                gate_diag = _ensure_delta_y_diag(gate_diag)
                if bool(getattr(self.args, "disable_delta_y_w", False)):
                    gate_diag = _apply_disable_delta_y_w_ablation(gate_diag)
                if bool(getattr(self.args, "rule_override", False)):
                    cmd_rule, rule_info = _compute_rule_override_cmd(
                        self.args,
                        gate_diag["cmd_f"],
                        gate_diag["cmd_a"],
                        gate_diag["risk_F"],
                        gate_diag["risk_A"],
                    )
                    follow_scale = rule_info["rule_follow_scale"]
                    zero = torch.zeros_like(follow_scale)
                    one = torch.ones_like(follow_scale)
                    gate_diag["cmd"] = cmd_rule
                    gate_diag["gate_y_raw"] = one
                    gate_diag["gate_y"] = one
                    gate_diag["gate_y_safe"] = one
                    gate_diag["y_eff"] = follow_scale
                    gate_diag["w"] = torch.full_like(follow_scale, 0.5)
                    gate_diag["signed_w"] = zero
                    gate_diag["signed_w_active"] = zero
                    gate_diag["w_support_correction"] = zero
                    gate_diag["risk_diff_correction"] = zero
                    gate_diag["delta_y_w_raw"] = zero
                    gate_diag["delta_y_w_used"] = zero
                    gate_diag["delta_y_r"] = zero
                    gate_diag["delta_y_total"] = follow_scale - one
                    gate_diag.update(rule_info)
            return gate_diag["cmd"], gate_diag["y_eff"], gate_diag

        with torch.no_grad():
            policy_state = th.match_state_dim(
                state,
                self.gate_state_dim or int(state.shape[1]),
                label="eval_policy_state",
            )
            # Mono-PPO output convention:
            # cmd[:, 0] = x_right / lateral, cmd[:, 1] = y_forward / forward, cmd[:, 2] = yaw.
            cmd, _ = self.policy.get_action(
                policy_aff_stack,
                policy_state,
                goal,
                difficulty_input,
                deterministic=not self.args.stochastic,
            )
            if self.is_mono_ppo and self.avoid_model is not None:
                if avoid_aff_stack is None or avoid_difficulty is None:
                    raise ValueError("Mono-PPO conflict diagnostics require avoid affordance inputs.")
                avoid_aff_input = (
                    torch.zeros_like(avoid_aff_stack)
                    if bool(getattr(self.args, "zero_local_map", False))
                    else avoid_aff_stack
                )
                avoid_difficulty_input = (
                    torch.zeros_like(avoid_difficulty)
                    if bool(getattr(self.args, "zero_local_map", False))
                    else avoid_difficulty
                )
                target_avoid_state_dim = self.avoid_state_dim or int(state.shape[1])
                expert_state = th.get_moe_expert_state_inputs(
                    th.match_state_dim(state, target_avoid_state_dim, label="eval_mono_diag_expert_state")
                )
                cmd_f = ph._compute_moe_follow_cmd_from_goal(
                    expert_state,
                    goal,
                    self.done_prev,
                    tuple(float(v) for v in self.env.post_processor.max_cmd.detach().cpu().tolist()),
                    env_ref=self.env,
                )
                cmd_a, _ = self.avoid_model.get_action(
                    avoid_aff_input,
                    expert_state,
                    avoid_goal,
                    avoid_difficulty_input,
                    deterministic=True,
                )
                diag = th._pcr_gate_command_conflict_diag(
                    self.env,
                    self.args,
                    gate_aff_map if gate_aff_map is not None else policy_aff_stack[:, :2],
                    cmd_f,
                    cmd_a,
                )
                nan = torch.full_like(diag["risk_F"], float("nan"))
                zero = torch.zeros_like(diag["risk_F"])
                gate_diag = {
                    "gate_y_raw": nan,
                    "gate_y": nan,
                    "gate_y_safe": nan,
                    "y_eff": nan,
                    "w": nan,
                    "cmd": cmd,
                    "cmd_f": cmd_f,
                    "cmd_a": diag["cmd_a_eff"],
                    "clearance_F": diag["clearance_F"],
                    "clearance_A": diag["clearance_A"],
                    "risk_F": diag["risk_F"],
                    "risk_A": diag["risk_A"],
                    "signed_w": nan,
                    "signed_w_active": nan,
                    "w_support_correction": zero,
                    "risk_diff_correction": zero,
                    "delta_y_w_raw": zero,
                    "delta_y_w_used": zero,
                    "delta_y_r": zero,
                    "delta_y_total": zero,
                    "fusion_formula_version": "none",
                    "row_current_valid": diag["row_current_valid"],
                    "row_not_released": diag["row_not_released"],
                    "current_row_front_edge": diag["current_row_front_edge"],
                    "current_row_back_edge": diag["current_row_back_edge"],
                    "robot_front_y": diag["robot_front_y"],
                    "robot_rear_y": diag["robot_rear_y"],
                    "risk_memory": zero,
                    "cmd_cos": diag["cmd_cos"],
                    "conflict_score": diag["conflict_score"],
                    "post_safe_distance": diag["post_safe_distance"],
                    "post_free_distance": diag["post_free_distance"],
                    "gate_safe_clamp_mask": torch.zeros_like(diag["risk_F"], dtype=torch.bool),
                }
                return cmd, None, gate_diag
        return cmd, None, None

    def _measure_inference_latency_ms(
        self,
        cmd_raw: torch.Tensor,
        aff_map: torch.Tensor,
        policy_elapsed_s: float,
    ) -> float:
        # Measure high-level policy+postprocess latency without altering control state.
        if self.env.post_processor.last_cmd is None:
            self.env.post_processor.reset(self.env.num_envs, self.device)
        last_cmd_backup = self.env.post_processor.last_cmd.detach().clone()
        clearance = self.env._compute_clearance_from_affordance(aff_map)

        t0 = time.perf_counter()
        _ = self.env.post_processor.process(cmd_raw, clearance, beta=self.env.beta_override)
        t1 = time.perf_counter()

        self.env.post_processor.last_cmd = last_cmd_backup
        return 1000.0 * (float(policy_elapsed_s) + (t1 - t0))

    def evaluate(self) -> Dict:
        difficulty_levels = _difficulty_list(self.args.difficulty_levels)
        episodes_total = int(self.args.episodes)
        per_level_target = int(math.ceil(episodes_total / max(1, len(difficulty_levels))))

        latency_ms_samples: List[float] = []
        episode_rows: List[Dict] = []
        timeseries_rows: List[Dict] = []
        dump_timeseries = bool(getattr(self.args, "dump_timeseries", False))
        timeseries_limit = max(0, int(getattr(self.args, "timeseries_episodes", 8)))
        timeseries_stride = max(1, int(getattr(self.args, "timeseries_stride", 1)))
        w_trigger_threshold = float(getattr(self.args, "w_trigger_threshold", 0.5))
        gate_region_risk_threshold = float(getattr(self.args, "gate_region_risk_threshold", 0.5))
        progress_interval_s = max(0.0, float(getattr(self.args, "progress_interval_s", 5.0)))
        target_rgb_half_fov_rad = math.radians(0.5 * float(getattr(self.args, "target_rgb_fov_deg", 69.4)))
        target_fov_margin_rad = math.radians(float(getattr(self.args, "target_fov_margin_deg", 3.0)))
        target_near_edge_margin_rad = math.radians(float(getattr(self.args, "target_near_fov_edge_margin_deg", 5.0)))
        target_in_fov_limit_rad = max(1e-6, target_rgb_half_fov_rad - target_fov_margin_rad)
        target_near_edge_start_rad = max(0.0, target_in_fov_limit_rad - target_near_edge_margin_rad)
        target_lost_k = max(1, int(getattr(self.args, "target_lost_k_eval", 5)))

        global_episode_idx = 0
        eval_start_t = time.perf_counter()
        last_progress_t = eval_start_t

        for level_idx, d in enumerate(difficulty_levels):
            seed_level = int(self.args.seed + level_idx)
            _setup_seed(seed_level)
            print(
                f"[Eval] level {level_idx + 1}/{len(difficulty_levels)} "
                f"difficulty={float(d):.3f} target_episodes={per_level_target}",
                flush=True,
            )

            self.env.set_scene_difficulty_target(float(d))
            self.env._apply_scene_difficulty_for_resets(None)
            ph._maybe_apply_s_avoid_stage_override_runtime(self.args, self.env)
            obs = self.env.reset()
            self._export_eval_layout_debug_once()
            self.aff_stack_buf = None
            self.follow_aff_stack_buf = None
            self.avoid_aff_stack_buf = None
            self.done_prev = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
            unsafe_conflict_streak = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

            acc = [EpisodeAccumulator() for _ in range(self.env.num_envs)]
            done_episodes = 0

            while done_episodes < per_level_target:
                aff_bundle = self._build_affordance_bundle(obs)
                if self.args.skill == "moe":
                    self.aff_stack_buf = self._roll_aff_stack(
                        self.aff_stack_buf, aff_bundle["gate_aff"], self.done_prev
                    )
                    self.follow_aff_stack_buf = self._roll_aff_stack(
                        self.follow_aff_stack_buf, aff_bundle["follow_aff"], self.done_prev
                    )
                    self.avoid_aff_stack_buf = self._roll_aff_stack(
                        self.avoid_aff_stack_buf, aff_bundle["avoid_aff"], self.done_prev
                    )
                    aff_stack = self.aff_stack_buf
                    difficulty_now = aff_bundle["gate_difficulty"]
                else:
                    if self.args.skill == "avoid":
                        actor_aff_map = aff_bundle["avoid_aff"]
                        difficulty_now = aff_bundle["avoid_difficulty"]
                    else:
                        actor_aff_map = aff_bundle["follow_aff"]
                        difficulty_now = aff_bundle["follow_difficulty"]
                    self.aff_stack_buf = self._roll_aff_stack(self.aff_stack_buf, actor_aff_map, self.done_prev)
                    aff_stack = self.aff_stack_buf

                # Pre-step follow error (avoid post-reset contamination on done envs).
                if hasattr(self.env.env, "target_world"):
                    robot_xy = self.env.env.root_states[:, :2]
                    target_xy = self.env.env.target_world
                    d_des = float(getattr(self.env, "s0_follow_d_des", 1.0))
                    dist = torch.norm(target_xy - robot_xy, dim=1)
                    err = torch.abs(dist - d_des)
                else:
                    err = torch.zeros(self.env.num_envs, device=self.device)

                freeze_timer = getattr(self.env.env, "target_freeze_timer", None)
                if freeze_timer is None:
                    valid_follow = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)
                else:
                    valid_follow = freeze_timer <= 1e-6

                follow_goal_obs = obs.get("follow_goal", obs.get("goal", None)) if isinstance(obs, dict) else None
                if torch.is_tensor(follow_goal_obs) and follow_goal_obs.shape[-1] >= 2:
                    target_bearing = torch.atan2(follow_goal_obs[:, 0], follow_goal_obs[:, 1])
                    target_bearing_abs = torch.abs(target_bearing)
                    target_in_fov = target_bearing_abs <= target_in_fov_limit_rad
                    target_near_fov_edge = (
                        (target_bearing_abs > target_near_edge_start_rad)
                        & (target_bearing_abs <= target_in_fov_limit_rad)
                    )
                else:
                    target_bearing = torch.full((self.env.num_envs,), float("nan"), device=self.device)
                    target_bearing_abs = torch.full((self.env.num_envs,), float("nan"), device=self.device)
                    target_in_fov = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                    target_near_fov_edge = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

                t_pol0 = time.perf_counter()
                cmd_raw, gate_y, gate_diag = self._policy_step(
                    obs,
                    aff_stack,
                    difficulty_now,
                    follow_aff_stack=self.follow_aff_stack_buf,
                    follow_difficulty=aff_bundle["follow_difficulty"],
                    avoid_aff_stack=self.avoid_aff_stack_buf,
                    avoid_difficulty=aff_bundle["avoid_difficulty"],
                    gate_aff_map=aff_bundle["gate_aff"],
                )
                t_pol1 = time.perf_counter()
                lat_ms = self._measure_inference_latency_ms(
                    cmd_raw,
                    aff_bundle["gate_aff"] if self.args.skill == "moe" else actor_aff_map,
                    policy_elapsed_s=(t_pol1 - t_pol0),
                )
                latency_ms_samples.append(lat_ms)

                aff_for_post = aff_bundle["gate_aff"] if self.args.skill == "moe" else actor_aff_map
                self.env.clearance_affordance_override = aff_for_post
                self.env.clearance_override = None
                self.env.reward_affordance_override = aff_for_post
                is_additive_fusion = bool(getattr(self.args, "additive_fusion", False))
                is_velocity_search = bool(getattr(self.args, "velocity_search", False))
                gate_y_raw = (
                    gate_diag["gate_y_raw"]
                    if isinstance(gate_diag, dict) and not self.is_mono_ppo and not is_additive_fusion and not is_velocity_search
                    else None
                )
                pcr_risk_override = (
                    gate_diag.get("risk_F", None)
                    if isinstance(gate_diag, dict) and not self.is_mono_ppo
                    else None
                )
                next_obs, rewards, dones, info = self.env.step(
                    cmd_raw,
                    gate_y,
                    gate_y_raw=gate_y_raw,
                    pcr_obstacle_risk_override=pcr_risk_override,
                )
                post_info = info.get("post_info", None) if isinstance(info, dict) else None
                if isinstance(post_info, dict) and isinstance(gate_diag, dict):
                    post_info["gate_y_raw"] = gate_diag["gate_y_raw"].detach().clone()
                    post_info["gate_y"] = gate_diag["gate_y"].detach().clone()
                    post_info["y_eff"] = gate_diag["y_eff"].detach().clone()
                    post_info["w"] = gate_diag["w"].detach().clone()
                    post_info["cmd_F"] = gate_diag["cmd_f"].detach().clone()
                    post_info["cmd_A"] = gate_diag["cmd_a"].detach().clone()
                    post_info["cmd_gate_fused"] = gate_diag["cmd"].detach().clone()
                    post_info["clearance_F"] = gate_diag["clearance_F"].detach().clone()
                    post_info["clearance_A"] = gate_diag["clearance_A"].detach().clone()
                    post_info["risk_F"] = gate_diag["risk_F"].detach().clone()
                    post_info["risk_A"] = gate_diag["risk_A"].detach().clone()
                    post_info["signed_w"] = gate_diag["signed_w"].detach().clone()
                    post_info["signed_w_active"] = gate_diag["signed_w_active"].detach().clone()
                    post_info["w_support_correction"] = gate_diag["w_support_correction"].detach().clone()
                    post_info["risk_diff_correction"] = gate_diag["risk_diff_correction"].detach().clone()
                    post_info["delta_y_w_raw"] = gate_diag["delta_y_w_raw"].detach().clone()
                    post_info["delta_y_w_used"] = gate_diag["delta_y_w_used"].detach().clone()
                    post_info["delta_y_r"] = gate_diag["delta_y_r"].detach().clone()
                    post_info["delta_y_total"] = gate_diag["delta_y_total"].detach().clone()
                    post_info["row_current_valid"] = gate_diag["row_current_valid"].detach().clone()
                    post_info["row_not_released"] = gate_diag["row_not_released"].detach().clone()
                    post_info["risk_memory"] = gate_diag["risk_memory"].detach().clone()
                    post_info["cmd_cos"] = gate_diag["cmd_cos"].detach().clone()
                    post_info["conflict_score"] = gate_diag["conflict_score"].detach().clone()
                    for key in (
                        "rule_s",
                        "rule_risk_gap",
                        "rule_follow_scale",
                        "rule_yaw_scale",
                        "rule_follow_suppression",
                        "cmd_search_raw",
                        "cmd_search_after_filter",
                        "cmd_safe",
                        "velocity_search_raw_risk",
                        "velocity_search_safe_risk",
                        "velocity_search_filter_changed",
                        "velocity_search_safe_available",
                        "velocity_search_compute_time_ms",
                        "velocity_search_dynamic_window_enabled",
                        "velocity_search_candidate_count_before_dynamic_window",
                        "velocity_search_candidate_count_after_dynamic_window",
                        "velocity_search_dynamic_window_rejected_count",
                        "velocity_search_candidate_count",
                        "velocity_search_feasible_count",
                        "velocity_search_infeasible_count",
                        "velocity_search_feasible_count_after_margin",
                        "velocity_search_forward_candidate_count",
                        "velocity_search_forward_feasible_count",
                        "velocity_search_forward_infeasible_collision_count",
                        "velocity_search_forward_infeasible_margin_count",
                        "velocity_search_forward_infeasible_out_of_map_count",
                        "velocity_search_best_forward_feasible_cost",
                        "velocity_search_best_forward_feasible_clearance",
                        "velocity_search_fallback_used",
                        "fallback_reason",
                        "fallback_no_feasible",
                        "fallback_risk_threshold",
                        "fallback_collision",
                        "fallback_margin",
                        "fallback_out_of_map",
                        "fallback_invalid_cost",
                        "velocity_search_min_predicted_clearance",
                        "velocity_search_best_cost",
                        "selected_rollout_min_clearance",
                        "selected_rollout_collision_flag",
                        "selected_rollout_out_of_map_flag",
                        "fallback_cmd",
                    ):
                        if key in gate_diag:
                            post_info[key] = gate_diag[key].detach().clone()
                    priv_diag = _privileged_conflict_diag(
                        self.args,
                        self.env,
                        aff_bundle["gate_aff"],
                        gate_diag,
                        target_in_fov=target_in_fov,
                    )
                    unsafe_raw = priv_diag.get("unsafe_high_conflict_raw", None)
                    if torch.is_tensor(unsafe_raw):
                        unsafe_conflict_streak = torch.where(
                            self.done_prev,
                            torch.zeros_like(unsafe_conflict_streak),
                            unsafe_conflict_streak,
                        )
                        unsafe_conflict_streak = torch.where(
                            unsafe_raw > 0.5,
                            unsafe_conflict_streak + 1,
                            torch.zeros_like(unsafe_conflict_streak),
                        )
                        min_steps = max(1, int(getattr(self.args, "unsafe_conflict_min_steps", 3)))
                        persisted = (unsafe_conflict_streak >= min_steps).to(dtype=unsafe_raw.dtype)
                        priv_diag["unsafe_high_conflict"] = persisted
                        priv_diag["avoid_high_conflict"] = (
                            (priv_diag.get("avoid_high_conflict_raw", persisted) > 0.5)
                            & (persisted > 0.5)
                        ).to(dtype=unsafe_raw.dtype)
                        priv_diag["stop_high_conflict"] = (
                            (priv_diag.get("stop_high_conflict_raw", persisted) > 0.5)
                            & (persisted > 0.5)
                        ).to(dtype=unsafe_raw.dtype)
                    for key, value in priv_diag.items():
                        post_info[key] = value.detach().clone()

                # Step-level energy and distance proxies.
                torques = getattr(self.env.env, "torques", None)
                dof_vel = getattr(self.env.env, "dof_vel", None)
                if torques is None or dof_vel is None:
                    pwr = torch.zeros(self.env.num_envs, device=self.device)
                else:
                    pwr = torch.sum(torch.abs(torques * dof_vel), dim=1)

                base_vel_xy = getattr(self.env.env, "base_lin_vel", None)
                if base_vel_xy is None:
                    ds = torch.zeros(self.env.num_envs, device=self.device)
                else:
                    ds = torch.norm(base_vel_xy[:, :2], dim=1) * float(self.env.high_level_dt)

                reward_terms = info.get("reward_terms", {}) if isinstance(info, dict) else {}
                if reward_terms is None:
                    reward_terms = {}
                success_step = info.get("success_mask", None) if isinstance(info, dict) else None
                if success_step is None:
                    if bool(getattr(self.env.env, "s_avoid_enabled", False)) and hasattr(self.env.env, "_get_s_avoid_episode_success_flags"):
                        env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
                        success_step = self.env.env._get_s_avoid_episode_success_flags(env_ids)
                    else:
                        success_bonus = reward_terms.get("success_bonus", None)
                        if success_bonus is None:
                            success_step = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                        else:
                            success_step = success_bonus > 0.0
                progress_step = info.get("s_avoid_progress_mask", None) if isinstance(info, dict) else None
                if progress_step is None:
                    progress_step = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
                row_success_step = info.get("s_avoid_row_success_mask", None) if isinstance(info, dict) else None
                cross_line_dist = info.get("cross_line_dist", None) if isinstance(info, dict) else None
                if cross_line_dist is None:
                    cross_line_dist = torch.full((self.env.num_envs,), float("nan"), device=self.device)
                episode_collision = info.get("s_avoid_episode_collision", None) if isinstance(info, dict) else None
                if episode_collision is None:
                    episode_collision = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                else:
                    episode_collision = episode_collision.to(device=self.device, dtype=torch.bool)
                if row_success_step is None:
                    progress_step = progress_step.to(device=self.device, dtype=torch.float32)
                    row_success_step = torch.where(
                        episode_collision,
                        torch.zeros_like(progress_step),
                        progress_step,
                    )
                else:
                    row_success_step = row_success_step.to(device=self.device, dtype=torch.float32)

                # Update ongoing episode accumulators.
                for i in range(self.env.num_envs):
                    ai = acc[i]
                    ai.step_hl += 1

                    follow_err_step = float("nan")
                    if bool(valid_follow[i].item()):
                        follow_err_step = _safe_float(err[i].item(), default=0.0)
                        ai.follow_err_sum += follow_err_step
                        ai.follow_err_sq_sum += follow_err_step * follow_err_step
                        ai.follow_err_count += 1

                    ai.energy_j += _safe_float((pwr[i] * float(self.env.high_level_dt)).item(), default=0.0)
                    ai.distance_m += _safe_float(ds[i].item(), default=0.0)
                    progress_val = _safe_float(row_success_step[i].item(), default=0.0)
                    progress_val = float(np.clip(progress_val, 0.0, 1.0))
                    ai.progress_ratio_best = max(ai.progress_ratio_best, progress_val)
                    ai.progress_reached = ai.progress_reached or (progress_val > 0.0)
                    actual_base_vel_x_v = float("nan")
                    actual_base_vel_y_v = float("nan")
                    actual_base_vel_w_v = float("nan")
                    actual_base_delta_x_v = float("nan")
                    actual_base_delta_y_v = float("nan")
                    row_progress_delta_v = float("nan")
                    env_impl_for_motion = getattr(self.env, "env", None)
                    base_lin_vel_t = getattr(env_impl_for_motion, "base_lin_vel", None)
                    base_ang_vel_t = getattr(env_impl_for_motion, "base_ang_vel", None)
                    root_states_t = getattr(env_impl_for_motion, "root_states", None)
                    if torch.is_tensor(base_lin_vel_t) and base_lin_vel_t.shape[0] > i and base_lin_vel_t.shape[1] >= 2:
                        actual_base_vel_x_v = _safe_float(base_lin_vel_t[i, 0].item(), default=float("nan"))
                        actual_base_vel_y_v = _safe_float(base_lin_vel_t[i, 1].item(), default=float("nan"))
                    if torch.is_tensor(base_ang_vel_t) and base_ang_vel_t.shape[0] > i and base_ang_vel_t.shape[1] >= 3:
                        actual_base_vel_w_v = _safe_float(base_ang_vel_t[i, 2].item(), default=float("nan"))
                    current_robot_x_v = float("nan")
                    current_robot_y_v = float("nan")
                    current_heading_v = float("nan")
                    if torch.is_tensor(root_states_t) and root_states_t.shape[0] > i and root_states_t.shape[1] >= 7:
                        current_robot_x_v = _safe_float(root_states_t[i, 0].item(), default=float("nan"))
                        current_robot_y_v = _safe_float(root_states_t[i, 1].item(), default=float("nan"))
                        qx_v = _safe_float(root_states_t[i, 3].item(), default=0.0)
                        qy_v = _safe_float(root_states_t[i, 4].item(), default=0.0)
                        qz_v = _safe_float(root_states_t[i, 5].item(), default=0.0)
                        qw_v = _safe_float(root_states_t[i, 6].item(), default=1.0)
                        current_heading_v = math.atan2(
                            2.0 * (qw_v * qz_v + qx_v * qy_v),
                            1.0 - 2.0 * (qy_v * qy_v + qz_v * qz_v),
                        )
                    if (
                        math.isfinite(current_robot_x_v)
                        and math.isfinite(current_robot_y_v)
                        and math.isfinite(ai.prev_robot_x)
                        and math.isfinite(ai.prev_robot_y)
                    ):
                        heading_for_delta = ai.prev_robot_heading if math.isfinite(ai.prev_robot_heading) else current_heading_v
                        if math.isfinite(heading_for_delta):
                            dx_world = current_robot_x_v - ai.prev_robot_x
                            dy_world = current_robot_y_v - ai.prev_robot_y
                            cos_h = math.cos(heading_for_delta)
                            sin_h = math.sin(heading_for_delta)
                            actual_base_delta_x_v = cos_h * dx_world + sin_h * dy_world
                            actual_base_delta_y_v = -sin_h * dx_world + cos_h * dy_world
                            ai.actual_base_delta_x_sum += actual_base_delta_x_v
                            ai.actual_base_delta_y_sum += actual_base_delta_y_v
                    if math.isfinite(ai.prev_progress_val):
                        row_progress_delta_v = progress_val - ai.prev_progress_val
                        ai.row_progress_delta_sum += row_progress_delta_v
                    if math.isfinite(actual_base_vel_x_v):
                        ai.actual_base_vel_x_sum += actual_base_vel_x_v
                    if math.isfinite(actual_base_vel_y_v):
                        ai.actual_base_vel_y_sum += actual_base_vel_y_v
                    if math.isfinite(actual_base_vel_w_v):
                        ai.actual_base_vel_w_sum += actual_base_vel_w_v
                    if math.isfinite(actual_base_vel_x_v) or math.isfinite(actual_base_delta_y_v):
                        ai.actual_motion_steps += 1
                    ai.prev_robot_x = current_robot_x_v
                    ai.prev_robot_y = current_robot_y_v
                    ai.prev_robot_heading = current_heading_v
                    ai.prev_progress_val = progress_val

                    target_bearing_v = _safe_float(target_bearing[i].item(), default=float("nan"))
                    target_bearing_abs_v = _safe_float(target_bearing_abs[i].item(), default=float("nan"))
                    target_in_fov_v = bool(target_in_fov[i].item())
                    target_near_fov_edge_v = bool(target_near_fov_edge[i].item())
                    goal_x_v = _safe_float(follow_goal_obs[i, 0].item(), default=float("nan"))
                    goal_y_v = _safe_float(follow_goal_obs[i, 1].item(), default=float("nan"))
                    if math.isfinite(target_bearing_abs_v):
                        ai.target_bearing_abs_sum += target_bearing_abs_v
                        ai.target_bearing_abs_max = max(ai.target_bearing_abs_max, target_bearing_abs_v)
                        ai.target_bearing_abs_samples.append(target_bearing_abs_v)
                    ai.target_in_fov_steps += int(target_in_fov_v)
                    ai.target_near_fov_edge_steps += int(target_near_fov_edge_v)
                    if target_in_fov_v:
                        ai.target_lost_current_steps = 0
                    else:
                        ai.target_lost_steps += 1
                        ai.target_lost_current_steps += 1
                        ai.target_lost_max_consecutive_steps = max(
                            ai.target_lost_max_consecutive_steps,
                            ai.target_lost_current_steps,
                        )
                        if ai.target_lost_current_steps >= target_lost_k:
                            ai.target_lost_event = True

                    collision_now = bool(episode_collision[i].item())
                    if collision_now and not ai.episode_collision:
                        ai.t_collision_s = float(ai.step_hl) * float(self.env.high_level_dt)
                    ai.episode_collision = ai.episode_collision or collision_now
                    cross_line_val = _safe_float(cross_line_dist[i].item(), default=float("nan"))
                    ai.cross_line_dist_end = cross_line_val
                    if math.isfinite(cross_line_val):
                        ai.cross_line_dist_min = min(ai.cross_line_dist_min, cross_line_val)

                    if isinstance(post_info, dict):
                        rotate_only_t = post_info.get("rotate_only_active", None)
                        if torch.is_tensor(rotate_only_t):
                            ai.rotate_only_steps += int(bool(rotate_only_t[i].item()))

                        cmd_x_v = float("nan")
                        cmd_y_v = float("nan")
                        cmd_yaw_v = float("nan")
                        cmd_final_t = post_info.get("cmd_exec_mean", None)
                        if cmd_final_t is None:
                            cmd_final_t = post_info.get("cmd_override_final", None)
                        if cmd_final_t is None:
                            cmd_final_t = post_info.get("cmd_post", post_info.get("cmd_slew", cmd_raw))
                        if torch.is_tensor(cmd_final_t):
                            cmd_final_v = [float(x) for x in cmd_final_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_final_sum[j] += cmd_final_v[j]
                            cmd_x_v = _safe_float(cmd_final_v[0], default=float("nan"))
                            cmd_y_v = _safe_float(cmd_final_v[1], default=float("nan"))
                            cmd_yaw_v = _safe_float(cmd_final_v[2], default=float("nan"))
                            cmd_x_eps = float(getattr(self.args, "cmd_x_bias_eps", 0.05))
                            goal_x_bin_thr = float(getattr(self.args, "goal_x_bin_threshold", 0.15))
                            if math.isfinite(cmd_x_v):
                                ai.cmd_x_sum += cmd_x_v
                                ai.cmd_x_sq_sum += cmd_x_v * cmd_x_v
                                ai.cmd_x_abs_sum += abs(cmd_x_v)
                                ai.cmd_x_samples += 1
                                if cmd_x_v > cmd_x_eps:
                                    ai.cmd_x_positive_steps += 1
                                elif cmd_x_v < -cmd_x_eps:
                                    ai.cmd_x_negative_steps += 1
                                else:
                                    ai.cmd_x_zero_steps += 1
                            if (
                                math.isfinite(goal_x_v)
                                and math.isfinite(goal_y_v)
                                and math.isfinite(cmd_x_v)
                                and math.isfinite(cmd_y_v)
                                and math.isfinite(cmd_yaw_v)
                            ):
                                ai.goal_cmd_corr_steps += 1
                                ai.goal_x_sum += goal_x_v
                                ai.goal_y_sum += goal_y_v
                                ai.corr_cmd_x_sum += cmd_x_v
                                ai.corr_cmd_y_sum += cmd_y_v
                                ai.corr_cmd_yaw_sum += cmd_yaw_v
                                ai.goal_x_sq_sum += goal_x_v * goal_x_v
                                ai.goal_y_sq_sum += goal_y_v * goal_y_v
                                ai.corr_cmd_x_sq_sum += cmd_x_v * cmd_x_v
                                ai.corr_cmd_y_sq_sum += cmd_y_v * cmd_y_v
                                ai.corr_cmd_yaw_sq_sum += cmd_yaw_v * cmd_yaw_v
                                ai.goalx_cmdx_sum += goal_x_v * cmd_x_v
                                ai.goalx_cmdyaw_sum += goal_x_v * cmd_yaw_v
                                ai.goaly_cmdy_sum += goal_y_v * cmd_y_v
                                if goal_x_v < -goal_x_bin_thr:
                                    ai.goal_x_left_steps += 1
                                    ai.cmd_yaw_left_sum += cmd_yaw_v
                                    ai.cmd_x_left_sum += cmd_x_v
                                elif goal_x_v > goal_x_bin_thr:
                                    ai.goal_x_right_steps += 1
                                    ai.cmd_yaw_right_sum += cmd_yaw_v
                                    ai.cmd_x_right_sum += cmd_x_v
                            if ai.prev_cmd_final is not None:
                                dx = cmd_final_v[0] - ai.prev_cmd_final[0]
                                dy = cmd_final_v[1] - ai.prev_cmd_final[1]
                                dw = cmd_final_v[2] - ai.prev_cmd_final[2]
                                ai.cmd_jerk_lin_sum += math.hypot(dx, dy)
                                ai.cmd_jerk_ang_sum += abs(dw)
                            ai.prev_cmd_final = cmd_final_v
                        cmd_raw_post_t = post_info.get("cmd_raw", None)
                        cmd_safe_post_t = post_info.get("cmd_override_final", None)
                        if cmd_safe_post_t is None:
                            cmd_safe_post_t = post_info.get("cmd_post", post_info.get("cmd_slew", None))
                        if torch.is_tensor(cmd_raw_post_t) and torch.is_tensor(cmd_safe_post_t):
                            cmd_raw_post_v = [
                                float(x) for x in cmd_raw_post_t[i].detach().cpu().tolist()[:3]
                            ]
                            cmd_safe_post_v = [
                                float(x) for x in cmd_safe_post_t[i].detach().cpu().tolist()[:3]
                            ]
                            deltas = [
                                cmd_safe_post_v[j] - cmd_raw_post_v[j]
                                for j in range(min(3, len(cmd_raw_post_v), len(cmd_safe_post_v)))
                            ]
                            if len(deltas) >= 3:
                                abs_deltas = [abs(v) for v in deltas[:3]]
                                ai.cmd_post_delta_x_abs_sum += abs_deltas[0]
                                ai.cmd_post_delta_y_abs_sum += abs_deltas[1]
                                ai.cmd_post_delta_w_abs_sum += abs_deltas[2]
                                ai.cmd_post_delta_y_sum += deltas[1]
                                ai.cmd_post_delta_abs_sum += sum(abs_deltas)
                                ai.cmd_post_samples += 1
                                if max(abs_deltas) > 1e-6:
                                    ai.cmd_post_rewrite_steps += 1

                    if self.args.skill == "moe" and isinstance(post_info, dict):
                        gate_raw_t = post_info.get("gate_y_raw", None)
                        y_eff_t = post_info.get("y_eff", None)
                        w_t = post_info.get("w", None)
                        signed_w_t = post_info.get("signed_w", None)
                        signed_w_active_t = post_info.get("signed_w_active", None)
                        clr_f_t = post_info.get("clearance_F", None)
                        clr_a_t = post_info.get("clearance_A", None)
                        risk_f_t = post_info.get("risk_F", None)
                        risk_a_t = post_info.get("risk_A", None)
                        w_support_corr_t = post_info.get("w_support_correction", None)
                        risk_diff_corr_t = post_info.get("risk_diff_correction", None)
                        delta_y_w_raw_t = post_info.get("delta_y_w_raw", None)
                        delta_y_w_used_t = post_info.get("delta_y_w_used", None)
                        delta_y_r_t = post_info.get("delta_y_r", None)
                        delta_y_total_t = post_info.get("delta_y_total", None)
                        risk_memory_t = post_info.get("risk_memory", None)
                        rule_s_t = post_info.get("rule_s", None)
                        rule_risk_gap_t = post_info.get("rule_risk_gap", None)
                        rule_follow_scale_t = post_info.get("rule_follow_scale", None)
                        rule_yaw_scale_t = post_info.get("rule_yaw_scale", None)
                        rule_follow_suppression_t = post_info.get("rule_follow_suppression", None)
                        cmd_search_raw_t = post_info.get("cmd_search_raw", None)
                        cmd_search_after_filter_t = post_info.get("cmd_search_after_filter", None)
                        cmd_safe_t = post_info.get("cmd_safe", None)
                        velocity_search_raw_risk_t = post_info.get("velocity_search_raw_risk", None)
                        velocity_search_safe_risk_t = post_info.get("velocity_search_safe_risk", None)
                        velocity_search_filter_changed_t = post_info.get("velocity_search_filter_changed", None)
                        velocity_search_safe_available_t = post_info.get("velocity_search_safe_available", None)
                        velocity_search_compute_time_ms_t = post_info.get("velocity_search_compute_time_ms", None)
                        velocity_search_dynamic_window_enabled_t = post_info.get("velocity_search_dynamic_window_enabled", None)
                        velocity_search_candidate_count_before_dynamic_window_t = post_info.get("velocity_search_candidate_count_before_dynamic_window", None)
                        velocity_search_candidate_count_after_dynamic_window_t = post_info.get("velocity_search_candidate_count_after_dynamic_window", None)
                        velocity_search_dynamic_window_rejected_count_t = post_info.get("velocity_search_dynamic_window_rejected_count", None)
                        velocity_search_candidate_count_t = post_info.get("velocity_search_candidate_count", None)
                        velocity_search_feasible_count_t = post_info.get("velocity_search_feasible_count", None)
                        velocity_search_infeasible_count_t = post_info.get("velocity_search_infeasible_count", None)
                        velocity_search_feasible_count_after_margin_t = post_info.get("velocity_search_feasible_count_after_margin", None)
                        velocity_search_forward_candidate_count_t = post_info.get("velocity_search_forward_candidate_count", None)
                        velocity_search_forward_feasible_count_t = post_info.get("velocity_search_forward_feasible_count", None)
                        velocity_search_forward_infeasible_collision_count_t = post_info.get("velocity_search_forward_infeasible_collision_count", None)
                        velocity_search_forward_infeasible_margin_count_t = post_info.get("velocity_search_forward_infeasible_margin_count", None)
                        velocity_search_forward_infeasible_out_of_map_count_t = post_info.get("velocity_search_forward_infeasible_out_of_map_count", None)
                        velocity_search_best_forward_feasible_cost_t = post_info.get("velocity_search_best_forward_feasible_cost", None)
                        velocity_search_best_forward_feasible_clearance_t = post_info.get("velocity_search_best_forward_feasible_clearance", None)
                        velocity_search_fallback_used_t = post_info.get("velocity_search_fallback_used", None)
                        fallback_reason_t = post_info.get("fallback_reason", None)
                        fallback_no_feasible_t = post_info.get("fallback_no_feasible", None)
                        fallback_risk_threshold_t = post_info.get("fallback_risk_threshold", None)
                        fallback_collision_t = post_info.get("fallback_collision", None)
                        fallback_margin_t = post_info.get("fallback_margin", None)
                        fallback_out_of_map_t = post_info.get("fallback_out_of_map", None)
                        fallback_invalid_cost_t = post_info.get("fallback_invalid_cost", None)
                        velocity_search_min_predicted_clearance_t = post_info.get("velocity_search_min_predicted_clearance", None)
                        velocity_search_best_cost_t = post_info.get("velocity_search_best_cost", None)
                        selected_rollout_min_clearance_t = post_info.get("selected_rollout_min_clearance", None)
                        selected_rollout_collision_flag_t = post_info.get("selected_rollout_collision_flag", None)
                        selected_rollout_out_of_map_flag_t = post_info.get("selected_rollout_out_of_map_flag", None)
                        fallback_cmd_t = post_info.get("fallback_cmd", None)
                        row_not_released_t = post_info.get("row_not_released", None)
                        conflict_t = post_info.get("conflict_score", None)
                        priv_conflict_t = post_info.get("priv_conflict_score", None)
                        priv_high_conflict_t = post_info.get("priv_high_conflict", None)
                        priv_obstacle_window_t = post_info.get("priv_obstacle_window", None)
                        priv_follow_pressure_t = post_info.get("priv_follow_pressure", None)
                        priv_avoid_pressure_t = post_info.get("priv_avoid_pressure", None)
                        priv_conflict_phase_t = post_info.get("priv_conflict_phase", None)
                        unsafe_high_conflict_t = post_info.get("unsafe_high_conflict", None)
                        unsafe_follow_risk_t = post_info.get("unsafe_follow_risk", None)
                        unsafe_safe_candidate_better_t = post_info.get("unsafe_safe_candidate_better", None)
                        unsafe_avoid_safer_t = post_info.get("unsafe_avoid_safer", None)
                        unsafe_stop_safer_t = post_info.get("unsafe_stop_safer", None)
                        unsafe_command_disagree_t = post_info.get("unsafe_command_disagree", None)
                        unsafe_target_recoverable_t = post_info.get("target_recoverable_for_conflict", None)
                        avoid_high_conflict_t = post_info.get("avoid_high_conflict", None)
                        stop_high_conflict_t = post_info.get("stop_high_conflict", None)
                        clr_roll_f_t = post_info.get("clearance_rollout_F", None)
                        clr_roll_a_t = post_info.get("clearance_rollout_A", None)
                        clr_roll_s_t = post_info.get("clearance_rollout_S", None)
                        risk_roll_f_t = post_info.get("risk_rollout_F", None)
                        risk_roll_a_t = post_info.get("risk_rollout_A", None)
                        risk_roll_s_t = post_info.get("risk_rollout_S", None)
                        utility_a_t = post_info.get("utility_A", None)
                        utility_s_t = post_info.get("utility_S", None)
                        utility_a_minus_s_t = post_info.get("utility_A_minus_S", None)
                        avoid_lateral_opening_t = post_info.get("avoid_lateral_opening", None)
                        stop_forward_progress_t = post_info.get("stop_forward_progress", None)
                        avoid_gap_growth_t = post_info.get("avoid_target_gap_growth", None)
                        stop_gap_growth_t = post_info.get("stop_target_gap_growth", None)
                        cmd_f_t = post_info.get("cmd_F", None)
                        cmd_a_t = post_info.get("cmd_A", None)
                        cmd_s_t = post_info.get("cmd_S", None)
                        cmd_raw_post_t = post_info.get("cmd_raw", None)
                        cmd_safe_post_t = post_info.get("cmd_override_final", None)
                        if cmd_safe_post_t is None:
                            cmd_safe_post_t = post_info.get("cmd_post", post_info.get("cmd_slew", None))
                        clearance_pp_t = post_info.get("clearance_pp", None)
                        safe_thr_t = post_info.get("post_safe_distance", None)
                        cmd_f_y_v = (
                            _safe_float(cmd_f_t[i, 1].item(), default=float("nan"))
                            if torch.is_tensor(cmd_f_t) and cmd_f_t.shape[-1] >= 2 else float("nan")
                        )
                        cmd_a_y_v = (
                            _safe_float(cmd_a_t[i, 1].item(), default=float("nan"))
                            if torch.is_tensor(cmd_a_t) and cmd_a_t.shape[-1] >= 2 else float("nan")
                        )

                        gate_raw_v = _safe_float(gate_raw_t[i].item(), default=0.0) if torch.is_tensor(gate_raw_t) else 0.0
                        y_eff_v = _safe_float(y_eff_t[i].item(), default=gate_raw_v) if torch.is_tensor(y_eff_t) else gate_raw_v
                        w_v = _safe_float(w_t[i].item(), default=0.0) if torch.is_tensor(w_t) else 0.0
                        signed_w_v = _safe_float(signed_w_t[i].item(), default=0.0) if torch.is_tensor(signed_w_t) else 0.0
                        signed_w_active_v = _safe_float(signed_w_active_t[i].item(), default=0.0) if torch.is_tensor(signed_w_active_t) else 0.0
                        clr_f_v = _safe_float(clr_f_t[i].item(), default=0.0) if torch.is_tensor(clr_f_t) else 0.0
                        clr_a_v = _safe_float(clr_a_t[i].item(), default=0.0) if torch.is_tensor(clr_a_t) else 0.0
                        risk_f_v = _safe_float(risk_f_t[i].item(), default=0.0) if torch.is_tensor(risk_f_t) else 0.0
                        risk_a_v = _safe_float(risk_a_t[i].item(), default=0.0) if torch.is_tensor(risk_a_t) else 0.0
                        w_support_corr_v = _safe_float(w_support_corr_t[i].item(), default=0.0) if torch.is_tensor(w_support_corr_t) else 0.0
                        risk_diff_corr_v = _safe_float(risk_diff_corr_t[i].item(), default=0.0) if torch.is_tensor(risk_diff_corr_t) else 0.0
                        delta_y_w_raw_v = _safe_float(delta_y_w_raw_t[i].item(), default=w_support_corr_v) if torch.is_tensor(delta_y_w_raw_t) else w_support_corr_v
                        delta_y_w_used_v = _safe_float(delta_y_w_used_t[i].item(), default=w_support_corr_v) if torch.is_tensor(delta_y_w_used_t) else w_support_corr_v
                        delta_y_r_v = _safe_float(delta_y_r_t[i].item(), default=risk_diff_corr_v) if torch.is_tensor(delta_y_r_t) else risk_diff_corr_v
                        delta_y_total_v = _safe_float(delta_y_total_t[i].item(), default=y_eff_v - gate_raw_v) if torch.is_tensor(delta_y_total_t) else (y_eff_v - gate_raw_v)
                        risk_memory_v = _safe_float(risk_memory_t[i].item(), default=0.0) if torch.is_tensor(risk_memory_t) else 0.0
                        rule_s_v = _safe_float(rule_s_t[i].item(), default=float("nan")) if torch.is_tensor(rule_s_t) else float("nan")
                        rule_risk_gap_v = _safe_float(rule_risk_gap_t[i].item(), default=float("nan")) if torch.is_tensor(rule_risk_gap_t) else float("nan")
                        rule_follow_scale_v = _safe_float(rule_follow_scale_t[i].item(), default=float("nan")) if torch.is_tensor(rule_follow_scale_t) else float("nan")
                        rule_yaw_scale_v = _safe_float(rule_yaw_scale_t[i].item(), default=float("nan")) if torch.is_tensor(rule_yaw_scale_t) else float("nan")
                        rule_follow_suppression_v = _safe_float(rule_follow_suppression_t[i].item(), default=float("nan")) if torch.is_tensor(rule_follow_suppression_t) else float("nan")
                        row_not_released_v = _safe_float(row_not_released_t[i].item(), default=0.0) if torch.is_tensor(row_not_released_t) else 0.0
                        conflict_v = _safe_float(conflict_t[i].item(), default=0.0) if torch.is_tensor(conflict_t) else 0.0
                        priv_conflict_v = _safe_float(priv_conflict_t[i].item(), default=0.0) if torch.is_tensor(priv_conflict_t) else 0.0
                        priv_high_conflict_v = _safe_float(priv_high_conflict_t[i].item(), default=0.0) if torch.is_tensor(priv_high_conflict_t) else 0.0
                        priv_obstacle_window_v = _safe_float(priv_obstacle_window_t[i].item(), default=0.0) if torch.is_tensor(priv_obstacle_window_t) else 0.0
                        priv_follow_pressure_v = _safe_float(priv_follow_pressure_t[i].item(), default=0.0) if torch.is_tensor(priv_follow_pressure_t) else 0.0
                        priv_avoid_pressure_v = _safe_float(priv_avoid_pressure_t[i].item(), default=0.0) if torch.is_tensor(priv_avoid_pressure_t) else 0.0
                        priv_conflict_phase_v = int(round(_safe_float(priv_conflict_phase_t[i].item(), default=0.0))) if torch.is_tensor(priv_conflict_phase_t) else 0
                        unsafe_high_conflict_v = _safe_float(unsafe_high_conflict_t[i].item(), default=0.0) if torch.is_tensor(unsafe_high_conflict_t) else 0.0
                        unsafe_follow_risk_v = _safe_float(unsafe_follow_risk_t[i].item(), default=0.0) if torch.is_tensor(unsafe_follow_risk_t) else 0.0
                        unsafe_safe_candidate_better_v = _safe_float(unsafe_safe_candidate_better_t[i].item(), default=0.0) if torch.is_tensor(unsafe_safe_candidate_better_t) else 0.0
                        unsafe_avoid_safer_v = _safe_float(unsafe_avoid_safer_t[i].item(), default=0.0) if torch.is_tensor(unsafe_avoid_safer_t) else 0.0
                        unsafe_stop_safer_v = _safe_float(unsafe_stop_safer_t[i].item(), default=0.0) if torch.is_tensor(unsafe_stop_safer_t) else 0.0
                        unsafe_command_disagree_v = _safe_float(unsafe_command_disagree_t[i].item(), default=0.0) if torch.is_tensor(unsafe_command_disagree_t) else 0.0
                        unsafe_target_recoverable_v = _safe_float(unsafe_target_recoverable_t[i].item(), default=0.0) if torch.is_tensor(unsafe_target_recoverable_t) else 0.0
                        avoid_high_conflict_v = _safe_float(avoid_high_conflict_t[i].item(), default=0.0) if torch.is_tensor(avoid_high_conflict_t) else 0.0
                        stop_high_conflict_v = _safe_float(stop_high_conflict_t[i].item(), default=0.0) if torch.is_tensor(stop_high_conflict_t) else 0.0
                        clr_roll_f_v = _safe_float(clr_roll_f_t[i].item(), default=clr_f_v) if torch.is_tensor(clr_roll_f_t) else clr_f_v
                        clr_roll_a_v = _safe_float(clr_roll_a_t[i].item(), default=clr_a_v) if torch.is_tensor(clr_roll_a_t) else clr_a_v
                        clr_roll_s_v = _safe_float(clr_roll_s_t[i].item(), default=float("nan")) if torch.is_tensor(clr_roll_s_t) else float("nan")
                        risk_roll_f_v = _safe_float(risk_roll_f_t[i].item(), default=risk_f_v) if torch.is_tensor(risk_roll_f_t) else risk_f_v
                        risk_roll_a_v = _safe_float(risk_roll_a_t[i].item(), default=risk_a_v) if torch.is_tensor(risk_roll_a_t) else risk_a_v
                        risk_roll_s_v = _safe_float(risk_roll_s_t[i].item(), default=float("nan")) if torch.is_tensor(risk_roll_s_t) else float("nan")
                        utility_a_v = _safe_float(utility_a_t[i].item(), default=float("nan")) if torch.is_tensor(utility_a_t) else float("nan")
                        utility_s_v = _safe_float(utility_s_t[i].item(), default=float("nan")) if torch.is_tensor(utility_s_t) else float("nan")
                        utility_a_minus_s_v = _safe_float(utility_a_minus_s_t[i].item(), default=float("nan")) if torch.is_tensor(utility_a_minus_s_t) else float("nan")
                        avoid_lateral_opening_v = _safe_float(avoid_lateral_opening_t[i].item(), default=float("nan")) if torch.is_tensor(avoid_lateral_opening_t) else float("nan")
                        stop_forward_progress_v = _safe_float(stop_forward_progress_t[i].item(), default=float("nan")) if torch.is_tensor(stop_forward_progress_t) else float("nan")
                        avoid_gap_growth_v = _safe_float(avoid_gap_growth_t[i].item(), default=float("nan")) if torch.is_tensor(avoid_gap_growth_t) else float("nan")
                        stop_gap_growth_v = _safe_float(stop_gap_growth_t[i].item(), default=float("nan")) if torch.is_tensor(stop_gap_growth_t) else float("nan")
                        clr_pp_v = float("nan")
                        safe_thr_v = float("nan")
                        near_miss_now = False
                        if torch.is_tensor(clearance_pp_t):
                            clr_pp_v = _safe_float(clearance_pp_t[i].item(), default=float("inf"))
                        if safe_thr_t is not None:
                            safe_thr_v = _safe_float(
                                safe_thr_t[i].item() if torch.is_tensor(safe_thr_t) and safe_thr_t.ndim > 0 else float(safe_thr_t),
                                default=float("nan"),
                            )
                        if math.isfinite(clr_pp_v) and math.isfinite(safe_thr_v):
                            near_miss_now = bool(clr_pp_v < safe_thr_v)
                        ai.gate_y_raw_sum += gate_raw_v
                        ai.y_eff_sum += y_eff_v
                        ai.w_sum += w_v
                        ai.signed_w_sum += signed_w_v
                        ai.signed_w_active_sum += signed_w_active_v
                        ai.clearance_f_sum += clr_f_v
                        ai.clearance_a_sum += clr_a_v
                        ai.risk_f_sum += risk_f_v
                        ai.risk_a_sum += risk_a_v
                        ai.clearance_rollout_f_sum += clr_roll_f_v if math.isfinite(clr_roll_f_v) else 0.0
                        ai.clearance_rollout_a_sum += clr_roll_a_v if math.isfinite(clr_roll_a_v) else 0.0
                        ai.clearance_rollout_s_sum += clr_roll_s_v if math.isfinite(clr_roll_s_v) else 0.0
                        ai.risk_rollout_f_sum += risk_roll_f_v if math.isfinite(risk_roll_f_v) else 0.0
                        ai.risk_rollout_a_sum += risk_roll_a_v if math.isfinite(risk_roll_a_v) else 0.0
                        ai.risk_rollout_s_sum += risk_roll_s_v if math.isfinite(risk_roll_s_v) else 0.0
                        if math.isfinite(risk_roll_f_v) and math.isfinite(risk_roll_a_v) and math.isfinite(risk_roll_s_v):
                            ai.risk_rollout_gap_f_min_as_sum += risk_roll_f_v - min(risk_roll_a_v, risk_roll_s_v)
                        ai.w_support_correction_sum += w_support_corr_v
                        ai.risk_diff_correction_sum += risk_diff_corr_v
                        ai.delta_y_w_raw_sum += delta_y_w_raw_v
                        ai.delta_y_w_used_sum += delta_y_w_used_v
                        ai.delta_y_r_sum += delta_y_r_v
                        ai.delta_y_total_sum += delta_y_total_v
                        ai.risk_memory_sum += risk_memory_v
                        if math.isfinite(rule_s_v):
                            ai.rule_s_sum += rule_s_v
                        if math.isfinite(rule_risk_gap_v):
                            ai.rule_risk_gap_sum += rule_risk_gap_v
                        if math.isfinite(rule_follow_scale_v):
                            ai.rule_follow_scale_sum += rule_follow_scale_v
                        if math.isfinite(rule_yaw_scale_v):
                            ai.rule_yaw_scale_sum += rule_yaw_scale_v
                        if math.isfinite(rule_follow_suppression_v):
                            ai.rule_follow_suppression_sum += rule_follow_suppression_v
                        ai.row_not_released_sum += row_not_released_v
                        ai.priv_conflict_score_sum += priv_conflict_v
                        ai.priv_obstacle_window_steps += int(priv_obstacle_window_v > 0.5)
                        ai.priv_follow_pressure_steps += int(priv_follow_pressure_v > 0.5)
                        ai.priv_avoid_pressure_steps += int(priv_avoid_pressure_v > 0.5)
                        ai.unsafe_follow_risk_steps += int(unsafe_follow_risk_v > 0.5)
                        ai.unsafe_safe_candidate_better_steps += int(unsafe_safe_candidate_better_v > 0.5)
                        ai.unsafe_avoid_safer_steps += int(unsafe_avoid_safer_v > 0.5)
                        ai.unsafe_stop_safer_steps += int(unsafe_stop_safer_v > 0.5)
                        ai.unsafe_command_disagree_steps += int(unsafe_command_disagree_v > 0.5)
                        ai.unsafe_target_recoverable_steps += int(unsafe_target_recoverable_v > 0.5)
                        if priv_conflict_phase_v == 1:
                            ai.priv_window_phase_approach_steps += 1
                        elif priv_conflict_phase_v == 2:
                            ai.priv_window_phase_inside_steps += 1
                        elif priv_conflict_phase_v == 3:
                            ai.priv_window_phase_release_steps += 1
                        delta_y_v = y_eff_v - gate_raw_v
                        if priv_high_conflict_v > 0.5:
                            ai.priv_conflict_steps += 1
                            ai.priv_conflict_y_raw_sum += gate_raw_v
                            ai.priv_conflict_y_eff_sum += y_eff_v
                            ai.priv_conflict_w_sum += w_v
                            ai.priv_conflict_signed_w_sum += signed_w_v
                            ai.priv_conflict_delta_y_sum += delta_y_v
                            if math.isfinite(target_bearing_abs_v):
                                ai.target_conflict_bearing_abs_sum += target_bearing_abs_v
                                ai.target_conflict_bearing_abs_max = max(
                                    ai.target_conflict_bearing_abs_max,
                                    target_bearing_abs_v,
                                )
                            ai.target_conflict_in_fov_steps += int(target_in_fov_v)
                            ai.target_conflict_near_fov_edge_steps += int(target_near_fov_edge_v)
                            ai.target_conflict_lost_steps += int(not target_in_fov_v)
                            if priv_conflict_phase_v == 1:
                                ai.priv_conflict_phase_approach_steps += 1
                                ai.priv_conflict_phase_approach_w_sum += w_v
                                ai.priv_conflict_phase_approach_signed_w_sum += signed_w_v
                                ai.priv_conflict_phase_approach_delta_y_sum += delta_y_v
                            elif priv_conflict_phase_v == 2:
                                ai.priv_conflict_phase_inside_steps += 1
                                ai.priv_conflict_phase_inside_w_sum += w_v
                                ai.priv_conflict_phase_inside_signed_w_sum += signed_w_v
                                ai.priv_conflict_phase_inside_delta_y_sum += delta_y_v
                            elif priv_conflict_phase_v == 3:
                                ai.priv_conflict_phase_release_steps += 1
                                ai.priv_conflict_phase_release_w_sum += w_v
                                ai.priv_conflict_phase_release_signed_w_sum += signed_w_v
                                ai.priv_conflict_phase_release_delta_y_sum += delta_y_v
                        else:
                            ai.priv_non_conflict_steps += 1
                            ai.priv_non_conflict_delta_y_sum += delta_y_v
                        if unsafe_high_conflict_v > 0.5:
                            ai.unsafe_conflict_steps += 1
                            ai.unsafe_conflict_y_raw_sum += gate_raw_v
                            ai.unsafe_conflict_y_eff_sum += y_eff_v
                            ai.unsafe_conflict_w_sum += w_v
                            ai.unsafe_conflict_signed_w_sum += signed_w_v
                            ai.unsafe_conflict_delta_y_sum += delta_y_v
                            ai.unsafe_conflict_delta_y_w_raw_sum += delta_y_w_raw_v
                            ai.unsafe_conflict_delta_y_w_used_sum += delta_y_w_used_v
                            ai.unsafe_conflict_delta_y_r_sum += delta_y_r_v
                            ai.unsafe_conflict_delta_y_total_sum += delta_y_total_v
                            if priv_conflict_phase_v == 1:
                                ai.unsafe_conflict_phase_approach_steps += 1
                                ai.unsafe_conflict_phase_approach_w_sum += w_v
                                ai.unsafe_conflict_phase_approach_signed_w_sum += signed_w_v
                                ai.unsafe_conflict_phase_approach_delta_y_sum += delta_y_v
                            elif priv_conflict_phase_v == 2:
                                ai.unsafe_conflict_phase_inside_steps += 1
                                ai.unsafe_conflict_phase_inside_w_sum += w_v
                                ai.unsafe_conflict_phase_inside_signed_w_sum += signed_w_v
                                ai.unsafe_conflict_phase_inside_delta_y_sum += delta_y_v
                            elif priv_conflict_phase_v == 3:
                                ai.unsafe_conflict_phase_release_steps += 1
                                ai.unsafe_conflict_phase_release_w_sum += w_v
                                ai.unsafe_conflict_phase_release_signed_w_sum += signed_w_v
                                ai.unsafe_conflict_phase_release_delta_y_sum += delta_y_v
                        else:
                            ai.unsafe_non_conflict_steps += 1
                            ai.unsafe_non_conflict_delta_y_sum += delta_y_v
                        if avoid_high_conflict_v > 0.5:
                            ai.avoid_conflict_steps += 1
                            ai.avoid_conflict_y_raw_sum += gate_raw_v
                            ai.avoid_conflict_y_eff_sum += y_eff_v
                            ai.avoid_conflict_w_sum += w_v
                            ai.avoid_conflict_signed_w_sum += signed_w_v
                            ai.avoid_conflict_delta_y_sum += delta_y_v
                            ai.avoid_conflict_delta_y_w_raw_sum += delta_y_w_raw_v
                            ai.avoid_conflict_delta_y_w_used_sum += delta_y_w_used_v
                            ai.avoid_conflict_delta_y_r_sum += delta_y_r_v
                            ai.avoid_conflict_delta_y_total_sum += delta_y_total_v
                            if math.isfinite(cmd_y_v):
                                ai.avoid_conflict_cmd_y_sum += cmd_y_v
                                ai.avoid_conflict_cmd_y_steps += 1
                            if math.isfinite(cmd_f_y_v):
                                ai.avoid_conflict_cmd_f_y_sum += cmd_f_y_v
                                ai.avoid_conflict_cmd_f_y_steps += 1
                            if math.isfinite(cmd_a_y_v):
                                ai.avoid_conflict_cmd_a_y_sum += cmd_a_y_v
                                ai.avoid_conflict_cmd_a_y_steps += 1
                            if math.isfinite(rule_s_v):
                                ai.rule_avoid_conflict_s_sum += rule_s_v
                            if math.isfinite(rule_follow_scale_v):
                                ai.rule_avoid_conflict_follow_scale_sum += rule_follow_scale_v
                            if math.isfinite(rule_yaw_scale_v):
                                ai.rule_avoid_conflict_yaw_scale_sum += rule_yaw_scale_v
                            if math.isfinite(rule_follow_suppression_v):
                                ai.rule_avoid_conflict_follow_suppression_sum += rule_follow_suppression_v
                        if stop_high_conflict_v > 0.5:
                            ai.stop_conflict_steps += 1
                            ai.stop_conflict_y_raw_sum += gate_raw_v
                            ai.stop_conflict_y_eff_sum += y_eff_v
                            ai.stop_conflict_w_sum += w_v
                            ai.stop_conflict_signed_w_sum += signed_w_v
                            ai.stop_conflict_delta_y_sum += delta_y_v
                        if row_not_released_v > 0.5:
                            ai.row_not_released_w_sum += w_v
                            ai.row_not_released_steps += 1
                        else:
                            ai.row_released_w_sum += w_v
                            ai.row_released_steps += 1
                        if ai.prev_y_eff is not None and abs(y_eff_v - ai.prev_y_eff) > th.GATE_SWITCH_DY_DEFAULT:
                            ai.gate_switch_count += 1
                        ai.prev_y_eff = y_eff_v
                        if ai.w_trigger_step < 0 and w_v >= w_trigger_threshold:
                            ai.w_trigger_step = int(ai.step_hl)
                            ai.w_trigger_progress = float(progress_val)
                        if risk_f_v >= gate_region_risk_threshold:
                            ai.gate_region_steps += 1
                            ai.gate_region_y_eff_sum += y_eff_v
                            ai.high_risk_y_eff_sum += y_eff_v
                            ai.high_risk_w_sum += w_v
                            ai.high_risk_risk_f_sum += risk_f_v
                            ai.high_risk_risk_a_sum += risk_a_v
                            if near_miss_now:
                                ai.gate_region_near_miss_steps += 1
                                ai.high_risk_near_miss_steps += 1

                        risk_bin_idx = _risk_bin_index(risk_f_v)
                        if risk_bin_idx is not None:
                            bin_state = ai.risk_bin_stats[risk_bin_idx]
                            suppression_v = gate_raw_v - y_eff_v
                            bin_state["steps"] += 1
                            bin_state["gate_y_raw_sum"] += gate_raw_v
                            bin_state["gate_y_raw_sq_sum"] += gate_raw_v * gate_raw_v
                            bin_state["y_eff_sum"] += y_eff_v
                            bin_state["y_eff_sq_sum"] += y_eff_v * y_eff_v
                            bin_state["suppression_sum"] += suppression_v
                            bin_state["suppression_sq_sum"] += suppression_v * suppression_v
                            bin_state["w_sum"] += w_v
                            bin_state["w_sq_sum"] += w_v * w_v
                            bin_state["signed_w_sum"] += signed_w_v
                            bin_state["signed_w_active_sum"] += signed_w_active_v
                            bin_state["risk_memory_sum"] += risk_memory_v
                            bin_state["risk_f_sum"] += risk_f_v
                            bin_state["risk_a_sum"] += risk_a_v
                            bin_state["risk_delta_sum"] += risk_f_v - risk_a_v
                            bin_state["w_support_correction_sum"] += w_support_corr_v
                            bin_state["risk_diff_correction_sum"] += risk_diff_corr_v
                            if near_miss_now:
                                bin_state["near_miss_steps"] += 1
                        conflict_bin_idx = _risk_bin_index(conflict_v)
                        if conflict_bin_idx is not None:
                            bin_state = ai.conflict_bin_stats[conflict_bin_idx]
                            suppression_v = gate_raw_v - y_eff_v
                            bin_state["steps"] += 1
                            bin_state["gate_y_raw_sum"] += gate_raw_v
                            bin_state["gate_y_raw_sq_sum"] += gate_raw_v * gate_raw_v
                            bin_state["y_eff_sum"] += y_eff_v
                            bin_state["y_eff_sq_sum"] += y_eff_v * y_eff_v
                            bin_state["suppression_sum"] += suppression_v
                            bin_state["suppression_sq_sum"] += suppression_v * suppression_v
                            bin_state["w_sum"] += w_v
                            bin_state["w_sq_sum"] += w_v * w_v
                            bin_state["signed_w_sum"] += signed_w_v
                            bin_state["signed_w_active_sum"] += signed_w_active_v
                            bin_state["risk_memory_sum"] += risk_memory_v
                            bin_state["risk_f_sum"] += risk_f_v
                            bin_state["risk_a_sum"] += risk_a_v
                            bin_state["risk_delta_sum"] += risk_f_v - risk_a_v
                            bin_state["w_support_correction_sum"] += w_support_corr_v
                            bin_state["risk_diff_correction_sum"] += risk_diff_corr_v
                            if near_miss_now:
                                bin_state["near_miss_steps"] += 1
                        priv_conflict_bin_idx = _risk_bin_index(priv_conflict_v)
                        if priv_obstacle_window_v > 0.5 and priv_conflict_bin_idx is not None:
                            bin_state = ai.priv_conflict_bin_stats[priv_conflict_bin_idx]
                            suppression_v = gate_raw_v - y_eff_v
                            bin_state["steps"] += 1
                            bin_state["gate_y_raw_sum"] += gate_raw_v
                            bin_state["gate_y_raw_sq_sum"] += gate_raw_v * gate_raw_v
                            bin_state["y_eff_sum"] += y_eff_v
                            bin_state["y_eff_sq_sum"] += y_eff_v * y_eff_v
                            bin_state["suppression_sum"] += suppression_v
                            bin_state["suppression_sq_sum"] += suppression_v * suppression_v
                            bin_state["w_sum"] += w_v
                            bin_state["w_sq_sum"] += w_v * w_v
                            bin_state["signed_w_sum"] += signed_w_v
                            bin_state["signed_w_active_sum"] += signed_w_active_v
                            bin_state["risk_memory_sum"] += risk_memory_v
                            bin_state["risk_f_sum"] += risk_f_v
                            bin_state["risk_a_sum"] += risk_a_v
                            bin_state["risk_delta_sum"] += risk_f_v - risk_a_v
                            bin_state["w_support_correction_sum"] += w_support_corr_v
                            bin_state["risk_diff_correction_sum"] += risk_diff_corr_v
                            if near_miss_now:
                                bin_state["near_miss_steps"] += 1

                        cmd_f_v = [float("nan"), float("nan"), float("nan")]
                        cmd_a_v = [float("nan"), float("nan"), float("nan")]
                        cmd_s_v = [float("nan"), float("nan"), float("nan")]
                        cmd_search_raw_v = [float("nan"), float("nan"), float("nan")]
                        cmd_search_after_filter_v = [float("nan"), float("nan"), float("nan")]
                        cmd_safe_v = [float("nan"), float("nan"), float("nan")]
                        velocity_search_compute_time_ms_v = float("nan")
                        velocity_search_candidate_count_v = float("nan")
                        velocity_search_feasible_count_v = float("nan")
                        velocity_search_infeasible_count_v = float("nan")
                        velocity_search_feasible_count_after_margin_v = float("nan")
                        velocity_search_forward_candidate_count_v = float("nan")
                        velocity_search_forward_feasible_count_v = float("nan")
                        velocity_search_forward_infeasible_collision_count_v = float("nan")
                        velocity_search_forward_infeasible_margin_count_v = float("nan")
                        velocity_search_forward_infeasible_out_of_map_count_v = float("nan")
                        velocity_search_best_forward_feasible_cost_v = float("nan")
                        velocity_search_best_forward_feasible_clearance_v = float("nan")
                        velocity_search_fallback_used_v = float("nan")
                        fallback_reason_v = float("nan")
                        fallback_no_feasible_v = float("nan")
                        fallback_risk_threshold_v = float("nan")
                        fallback_collision_v = float("nan")
                        fallback_margin_v = float("nan")
                        fallback_out_of_map_v = float("nan")
                        fallback_invalid_cost_v = float("nan")
                        velocity_search_min_predicted_clearance_v = float("nan")
                        velocity_search_best_cost_v = float("nan")
                        selected_rollout_min_clearance_v = float("nan")
                        selected_rollout_collision_flag_v = float("nan")
                        selected_rollout_out_of_map_flag_v = float("nan")
                        velocity_search_dynamic_window_enabled_v = float("nan")
                        velocity_search_candidate_count_before_dynamic_window_v = float("nan")
                        velocity_search_candidate_count_after_dynamic_window_v = float("nan")
                        velocity_search_dynamic_window_rejected_count_v = float("nan")
                        fallback_cmd_v = [float("nan"), float("nan"), float("nan")]
                        cmd_raw_post_v = [float("nan"), float("nan"), float("nan")]
                        cmd_safe_post_v = [float("nan"), float("nan"), float("nan")]
                        cmd_post_delta_v = [float("nan"), float("nan"), float("nan")]
                        if torch.is_tensor(cmd_f_t):
                            cmd_f_v = [float(x) for x in cmd_f_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_f_sum[j] += cmd_f_v[j]
                        if torch.is_tensor(cmd_a_t):
                            cmd_a_v = [float(x) for x in cmd_a_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_a_sum[j] += cmd_a_v[j]
                        if torch.is_tensor(cmd_s_t):
                            cmd_s_v = [float(x) for x in cmd_s_t[i].detach().cpu().tolist()[:3]]
                        if torch.is_tensor(cmd_search_raw_t):
                            cmd_search_raw_v = [float(x) for x in cmd_search_raw_t[i].detach().cpu().tolist()[:3]]
                        if torch.is_tensor(cmd_search_after_filter_t):
                            cmd_search_after_filter_v = [
                                float(x) for x in cmd_search_after_filter_t[i].detach().cpu().tolist()[:3]
                            ]
                        if torch.is_tensor(cmd_safe_t):
                            cmd_safe_v = [float(x) for x in cmd_safe_t[i].detach().cpu().tolist()[:3]]
                        if torch.is_tensor(velocity_search_raw_risk_t) and torch.is_tensor(velocity_search_safe_risk_t):
                            raw_risk_v = _safe_float(
                                velocity_search_raw_risk_t[i].item(),
                                default=float("nan"),
                            )
                            safe_risk_v = _safe_float(
                                velocity_search_safe_risk_t[i].item(),
                                default=float("nan"),
                            )
                            if math.isfinite(raw_risk_v) and math.isfinite(safe_risk_v):
                                ai.velocity_search_steps += 1
                                ai.velocity_search_raw_risk_sum += raw_risk_v
                                ai.velocity_search_safe_risk_sum += safe_risk_v
                                if torch.is_tensor(velocity_search_filter_changed_t):
                                    ai.velocity_search_filter_change_steps += int(
                                        bool(velocity_search_filter_changed_t[i].item())
                                    )
                                if torch.is_tensor(velocity_search_safe_available_t):
                                    ai.velocity_search_safe_available_steps += int(
                                        bool(velocity_search_safe_available_t[i].item())
                                    )
                                if torch.is_tensor(velocity_search_compute_time_ms_t):
                                    velocity_search_compute_time_ms_v = _safe_float(
                                        velocity_search_compute_time_ms_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_compute_time_ms_sum += _safe_float(
                                        velocity_search_compute_time_ms_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_dynamic_window_enabled_t):
                                    velocity_search_dynamic_window_enabled_v = _safe_float(
                                        velocity_search_dynamic_window_enabled_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_dynamic_window_enabled_steps += int(
                                        bool(velocity_search_dynamic_window_enabled_t[i].item())
                                    )
                                if torch.is_tensor(velocity_search_candidate_count_before_dynamic_window_t):
                                    velocity_search_candidate_count_before_dynamic_window_v = _safe_float(
                                        velocity_search_candidate_count_before_dynamic_window_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_candidate_count_before_dynamic_window_sum += _safe_float(
                                        velocity_search_candidate_count_before_dynamic_window_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_candidate_count_after_dynamic_window_t):
                                    velocity_search_candidate_count_after_dynamic_window_v = _safe_float(
                                        velocity_search_candidate_count_after_dynamic_window_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_candidate_count_after_dynamic_window_sum += _safe_float(
                                        velocity_search_candidate_count_after_dynamic_window_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_dynamic_window_rejected_count_t):
                                    velocity_search_dynamic_window_rejected_count_v = _safe_float(
                                        velocity_search_dynamic_window_rejected_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_dynamic_window_rejected_count_sum += _safe_float(
                                        velocity_search_dynamic_window_rejected_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_candidate_count_t):
                                    velocity_search_candidate_count_v = _safe_float(
                                        velocity_search_candidate_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_candidate_count_sum += _safe_float(
                                        velocity_search_candidate_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_feasible_count_t):
                                    velocity_search_feasible_count_v = _safe_float(
                                        velocity_search_feasible_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_feasible_count_sum += _safe_float(
                                        velocity_search_feasible_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_infeasible_count_t):
                                    velocity_search_infeasible_count_v = _safe_float(
                                        velocity_search_infeasible_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_infeasible_count_sum += _safe_float(
                                        velocity_search_infeasible_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_feasible_count_after_margin_t):
                                    velocity_search_feasible_count_after_margin_v = _safe_float(
                                        velocity_search_feasible_count_after_margin_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_feasible_count_after_margin_sum += _safe_float(
                                        velocity_search_feasible_count_after_margin_t[i].item(),
                                        default=0.0,
                                    )
                                ai.velocity_search_selected_v_lat_sum += _safe_float(cmd_safe_v[0], default=0.0)
                                ai.velocity_search_selected_v_fwd_sum += _safe_float(cmd_safe_v[1], default=0.0)
                                ai.velocity_search_selected_yaw_sum += _safe_float(cmd_safe_v[2], default=0.0)
                                ai.velocity_search_raw_y_sum += _safe_float(cmd_search_raw_v[1], default=0.0)
                                ai.velocity_search_after_filter_y_sum += _safe_float(cmd_search_after_filter_v[1], default=0.0)
                                ai.velocity_search_cmd_safe_y_sum += _safe_float(cmd_safe_v[1], default=0.0)
                                if math.isfinite(cmd_search_raw_v[1]) and abs(cmd_search_raw_v[1]) > 1e-6:
                                    ai.velocity_search_cmd_safe_y_over_raw_y_sum += _safe_float(
                                        cmd_safe_v[1] / cmd_search_raw_v[1],
                                        default=0.0,
                                    )
                                    ai.velocity_search_cmd_safe_y_over_raw_y_steps += 1
                                if torch.is_tensor(velocity_search_forward_candidate_count_t):
                                    velocity_search_forward_candidate_count_v = _safe_float(
                                        velocity_search_forward_candidate_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_forward_candidate_count_sum += _safe_float(
                                        velocity_search_forward_candidate_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_forward_feasible_count_t):
                                    velocity_search_forward_feasible_count_v = _safe_float(
                                        velocity_search_forward_feasible_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_forward_feasible_count_sum += _safe_float(
                                        velocity_search_forward_feasible_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_forward_infeasible_collision_count_t):
                                    velocity_search_forward_infeasible_collision_count_v = _safe_float(
                                        velocity_search_forward_infeasible_collision_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_forward_infeasible_collision_count_sum += _safe_float(
                                        velocity_search_forward_infeasible_collision_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_forward_infeasible_margin_count_t):
                                    velocity_search_forward_infeasible_margin_count_v = _safe_float(
                                        velocity_search_forward_infeasible_margin_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_forward_infeasible_margin_count_sum += _safe_float(
                                        velocity_search_forward_infeasible_margin_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_forward_infeasible_out_of_map_count_t):
                                    velocity_search_forward_infeasible_out_of_map_count_v = _safe_float(
                                        velocity_search_forward_infeasible_out_of_map_count_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_forward_infeasible_out_of_map_count_sum += _safe_float(
                                        velocity_search_forward_infeasible_out_of_map_count_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_best_forward_feasible_cost_t):
                                    velocity_search_best_forward_feasible_cost_v = _safe_float(
                                        velocity_search_best_forward_feasible_cost_t[i].item(),
                                        default=float("nan"),
                                    )
                                if torch.is_tensor(velocity_search_best_forward_feasible_clearance_t):
                                    velocity_search_best_forward_feasible_clearance_v = _safe_float(
                                        velocity_search_best_forward_feasible_clearance_t[i].item(),
                                        default=float("nan"),
                                    )
                                if math.isfinite(velocity_search_best_forward_feasible_cost_v):
                                    ai.velocity_search_best_forward_feasible_cost_sum += velocity_search_best_forward_feasible_cost_v
                                    ai.velocity_search_best_forward_feasible_steps += 1
                                if math.isfinite(velocity_search_best_forward_feasible_clearance_v):
                                    ai.velocity_search_best_forward_feasible_clearance_sum += velocity_search_best_forward_feasible_clearance_v
                                if torch.is_tensor(velocity_search_fallback_used_t):
                                    velocity_search_fallback_used_v = _safe_float(
                                        velocity_search_fallback_used_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_fallback_steps += int(
                                        bool(velocity_search_fallback_used_t[i].item())
                                    )
                                if torch.is_tensor(fallback_reason_t):
                                    fallback_reason_v = _safe_float(fallback_reason_t[i].item(), default=float("nan"))
                                if torch.is_tensor(fallback_no_feasible_t):
                                    fallback_no_feasible_v = _safe_float(fallback_no_feasible_t[i].item(), default=float("nan"))
                                    ai.fallback_no_feasible_steps += int(bool(fallback_no_feasible_t[i].item()))
                                if torch.is_tensor(fallback_risk_threshold_t):
                                    fallback_risk_threshold_v = _safe_float(fallback_risk_threshold_t[i].item(), default=float("nan"))
                                    ai.fallback_risk_threshold_steps += int(bool(fallback_risk_threshold_t[i].item()))
                                if torch.is_tensor(fallback_collision_t):
                                    fallback_collision_v = _safe_float(fallback_collision_t[i].item(), default=float("nan"))
                                    ai.fallback_collision_steps += int(bool(fallback_collision_t[i].item()))
                                if torch.is_tensor(fallback_margin_t):
                                    fallback_margin_v = _safe_float(fallback_margin_t[i].item(), default=float("nan"))
                                    ai.fallback_margin_steps += int(bool(fallback_margin_t[i].item()))
                                if torch.is_tensor(fallback_out_of_map_t):
                                    fallback_out_of_map_v = _safe_float(fallback_out_of_map_t[i].item(), default=float("nan"))
                                    ai.fallback_out_of_map_steps += int(bool(fallback_out_of_map_t[i].item()))
                                if torch.is_tensor(fallback_invalid_cost_t):
                                    fallback_invalid_cost_v = _safe_float(fallback_invalid_cost_t[i].item(), default=float("nan"))
                                    ai.fallback_invalid_cost_steps += int(bool(fallback_invalid_cost_t[i].item()))
                                if torch.is_tensor(velocity_search_min_predicted_clearance_t):
                                    velocity_search_min_predicted_clearance_v = _safe_float(
                                        velocity_search_min_predicted_clearance_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_min_predicted_clearance_sum += _safe_float(
                                        velocity_search_min_predicted_clearance_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(velocity_search_best_cost_t):
                                    velocity_search_best_cost_v = _safe_float(
                                        velocity_search_best_cost_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.velocity_search_best_cost_sum += _safe_float(
                                        velocity_search_best_cost_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(selected_rollout_min_clearance_t):
                                    selected_rollout_min_clearance_v = _safe_float(
                                        selected_rollout_min_clearance_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.selected_rollout_min_clearance_sum += _safe_float(
                                        selected_rollout_min_clearance_t[i].item(),
                                        default=0.0,
                                    )
                                if torch.is_tensor(selected_rollout_collision_flag_t):
                                    selected_rollout_collision_flag_v = _safe_float(
                                        selected_rollout_collision_flag_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.selected_rollout_collision_steps += int(
                                        bool(selected_rollout_collision_flag_t[i].item())
                                    )
                                if torch.is_tensor(selected_rollout_out_of_map_flag_t):
                                    selected_rollout_out_of_map_flag_v = _safe_float(
                                        selected_rollout_out_of_map_flag_t[i].item(),
                                        default=float("nan"),
                                    )
                                    ai.selected_rollout_out_of_map_steps += int(
                                        bool(selected_rollout_out_of_map_flag_t[i].item())
                                    )
                                if torch.is_tensor(fallback_cmd_t):
                                    fallback_cmd_v = [float(x) for x in fallback_cmd_t[i].detach().cpu().tolist()[:3]]
                        if torch.is_tensor(cmd_raw_post_t) and torch.is_tensor(cmd_safe_post_t):
                            cmd_raw_post_v = [
                                float(x) for x in cmd_raw_post_t[i].detach().cpu().tolist()[:3]
                            ]
                            cmd_safe_post_v = [
                                float(x) for x in cmd_safe_post_t[i].detach().cpu().tolist()[:3]
                            ]
                            if len(cmd_raw_post_v) >= 3 and len(cmd_safe_post_v) >= 3:
                                cmd_post_delta_v = [
                                    cmd_safe_post_v[j] - cmd_raw_post_v[j]
                                    for j in range(3)
                                ]
                        if near_miss_now:
                            ai.near_miss_steps += 1

                        if dump_timeseries and ai.step_hl % timeseries_stride == 0:
                            env_impl = getattr(self.env, "env", None)
                            root_states = getattr(env_impl, "root_states", None)
                            target_world = getattr(env_impl, "target_world", None)
                            robot_x_v = float("nan")
                            robot_y_v = float("nan")
                            target_x_v = float("nan")
                            target_y_v = float("nan")
                            if torch.is_tensor(root_states) and root_states.shape[0] > i and root_states.shape[1] >= 2:
                                robot_x_v = _safe_float(root_states[i, 0].item(), default=float("nan"))
                                robot_y_v = _safe_float(root_states[i, 1].item(), default=float("nan"))
                            if torch.is_tensor(target_world) and target_world.shape[0] > i and target_world.shape[1] >= 2:
                                target_x_v = _safe_float(target_world[i, 0].item(), default=float("nan"))
                                target_y_v = _safe_float(target_world[i, 1].item(), default=float("nan"))
                            cmd_final_v = ai.prev_cmd_final if ai.prev_cmd_final is not None else [float("nan"), float("nan"), float("nan")]
                            ai.timeseries.append(
                                {
                                    "step": int(ai.step_hl),
                                    "step_hl": int(ai.step_hl),
                                    "time_s": float(ai.step_hl) * float(self.env.high_level_dt),
                                    "trajectory_frame": "world_xy_train_play",
                                    "robot_x": robot_x_v,
                                    "robot_y": robot_y_v,
                                    "target_x": target_x_v,
                                    "target_y": target_y_v,
                                    "target_x_right": goal_x_v,
                                    "target_y_forward": goal_y_v,
                                    "obstacles_json": _trajectory_obstacles_json(env_impl, i),
                                    "progress": float(progress_val),
                                    "row_progress_delta": row_progress_delta_v,
                                    "follow_err_m": follow_err_step,
                                    "target_bearing_rad": target_bearing_v,
                                    "target_bearing_abs_rad": target_bearing_abs_v,
                                    "target_bearing_abs_deg": (
                                        target_bearing_abs_v * 180.0 / math.pi
                                        if math.isfinite(target_bearing_abs_v) else float("nan")
                                    ),
                                    "target_in_rgb_fov": int(target_in_fov_v),
                                    "target_near_rgb_fov_edge": int(target_near_fov_edge_v),
                                    "target_lost_current_steps": int(ai.target_lost_current_steps),
                                    "gate_y_raw": gate_raw_v,
                                    "y_eff": y_eff_v,
                                    "w": w_v,
                                    "signed_w": signed_w_v,
                                    "signed_w_active": signed_w_active_v,
                                    "clearance_f": clr_f_v,
                                    "clearance_a": clr_a_v,
                                    "risk_f": risk_f_v,
                                    "risk_a": risk_a_v,
                                    "risk_delta": risk_f_v - risk_a_v,
                                    "clearance_rollout_f": clr_roll_f_v,
                                    "clearance_rollout_a": clr_roll_a_v,
                                    "clearance_rollout_s": clr_roll_s_v,
                                    "risk_rollout_f": risk_roll_f_v,
                                    "risk_rollout_a": risk_roll_a_v,
                                    "risk_rollout_s": risk_roll_s_v,
                                    "risk_rollout_gap_f_min_as": risk_roll_f_v - min(risk_roll_a_v, risk_roll_s_v),
                                    "utility_a": utility_a_v,
                                    "utility_s": utility_s_v,
                                    "utility_a_minus_s": utility_a_minus_s_v,
                                    "avoid_lateral_opening": avoid_lateral_opening_v,
                                    "stop_forward_progress": stop_forward_progress_v,
                                    "avoid_target_gap_growth": avoid_gap_growth_v,
                                    "stop_target_gap_growth": stop_gap_growth_v,
                                    "follow_weight": y_eff_v,
                                    "avoid_weight": 1.0 - y_eff_v,
                                    "csi": gate_raw_v - y_eff_v,
                                    "w_support_correction": w_support_corr_v,
                                    "risk_diff_correction": risk_diff_corr_v,
                                    "delta_y_w_raw": delta_y_w_raw_v,
                                    "delta_y_w_used": delta_y_w_used_v,
                                    "delta_y_r": delta_y_r_v,
                                    "delta_y_total": delta_y_total_v,
                                    "risk_memory": risk_memory_v,
                                    "rule_s": rule_s_v,
                                    "rule_risk_gap": rule_risk_gap_v,
                                    "rule_follow_scale": rule_follow_scale_v,
                                    "rule_yaw_scale": rule_yaw_scale_v,
                                    "rule_follow_suppression": rule_follow_suppression_v,
                                    "row_not_released": row_not_released_v,
                                    "conflict_score": conflict_v,
                                    "priv_conflict_score": priv_conflict_v,
                                    "priv_high_conflict": int(priv_high_conflict_v > 0.5),
                                    "priv_obstacle_window": int(priv_obstacle_window_v > 0.5),
                                    "priv_follow_pressure": int(priv_follow_pressure_v > 0.5),
                                    "priv_avoid_pressure": int(priv_avoid_pressure_v > 0.5),
                                    "priv_conflict_phase": int(priv_conflict_phase_v),
                                    "unsafe_high_conflict": int(unsafe_high_conflict_v > 0.5),
                                    "unsafe_follow_risk": int(unsafe_follow_risk_v > 0.5),
                                    "unsafe_safe_candidate_better": int(unsafe_safe_candidate_better_v > 0.5),
                                    "unsafe_avoid_safer": int(unsafe_avoid_safer_v > 0.5),
                                    "unsafe_stop_safer": int(unsafe_stop_safer_v > 0.5),
                                    "unsafe_command_disagree": int(unsafe_command_disagree_v > 0.5),
                                    "target_recoverable_for_conflict": int(unsafe_target_recoverable_v > 0.5),
                                    "avoid_high_conflict": int(avoid_high_conflict_v > 0.5),
                                    "stop_high_conflict": int(stop_high_conflict_v > 0.5),
                                    "clearance_pp": clr_pp_v,
                                    "near_miss": int(near_miss_now),
                                    "episode_collision": int(ai.episode_collision),
                                    "cmd_f_x": cmd_f_v[0],
                                    "cmd_f_y": cmd_f_v[1],
                                    "cmd_f_w": cmd_f_v[2],
                                    "cmd_a_x": cmd_a_v[0],
                                    "cmd_a_y": cmd_a_v[1],
                                    "cmd_a_w": cmd_a_v[2],
                                    "cmd_s_x": cmd_s_v[0],
                                    "cmd_s_y": cmd_s_v[1],
                                    "cmd_s_w": cmd_s_v[2],
                                    "cmd_search_raw_x": cmd_search_raw_v[0],
                                    "cmd_search_raw_y": cmd_search_raw_v[1],
                                    "cmd_search_raw_w": cmd_search_raw_v[2],
                                    "cmd_search_after_filter_x": cmd_search_after_filter_v[0],
                                    "cmd_search_after_filter_y": cmd_search_after_filter_v[1],
                                    "cmd_search_after_filter_w": cmd_search_after_filter_v[2],
                                    "cmd_safe_x": cmd_safe_v[0],
                                    "cmd_safe_y": cmd_safe_v[1],
                                    "cmd_safe_w": cmd_safe_v[2],
                                    "actual_base_vel_x": actual_base_vel_x_v,
                                    "actual_base_vel_y": actual_base_vel_y_v,
                                    "actual_base_vel_w": actual_base_vel_w_v,
                                    "actual_base_delta_x": actual_base_delta_x_v,
                                    "actual_base_delta_y": actual_base_delta_y_v,
                                    "velocity_search_compute_time_ms": velocity_search_compute_time_ms_v,
                                    "velocity_search_dynamic_window_enabled": velocity_search_dynamic_window_enabled_v,
                                    "velocity_search_candidate_count_before_dynamic_window": velocity_search_candidate_count_before_dynamic_window_v,
                                    "velocity_search_candidate_count_after_dynamic_window": velocity_search_candidate_count_after_dynamic_window_v,
                                    "velocity_search_dynamic_window_rejected_count": velocity_search_dynamic_window_rejected_count_v,
                                    "velocity_search_candidate_count": velocity_search_candidate_count_v,
                                    "velocity_search_feasible_count": velocity_search_feasible_count_v,
                                    "velocity_search_infeasible_count": velocity_search_infeasible_count_v,
                                    "velocity_search_feasible_count_after_margin": velocity_search_feasible_count_after_margin_v,
                                    "velocity_search_forward_candidate_count": velocity_search_forward_candidate_count_v,
                                    "velocity_search_forward_feasible_count": velocity_search_forward_feasible_count_v,
                                    "velocity_search_forward_infeasible_collision_count": velocity_search_forward_infeasible_collision_count_v,
                                    "velocity_search_forward_infeasible_margin_count": velocity_search_forward_infeasible_margin_count_v,
                                    "velocity_search_forward_infeasible_out_of_map_count": velocity_search_forward_infeasible_out_of_map_count_v,
                                    "velocity_search_best_forward_feasible_cost": velocity_search_best_forward_feasible_cost_v,
                                    "velocity_search_best_forward_feasible_clearance": velocity_search_best_forward_feasible_clearance_v,
                                    "velocity_search_fallback_used": velocity_search_fallback_used_v,
                                    "fallback_reason": fallback_reason_v,
                                    "fallback_no_feasible": fallback_no_feasible_v,
                                    "fallback_risk_threshold": fallback_risk_threshold_v,
                                    "fallback_collision": fallback_collision_v,
                                    "fallback_margin": fallback_margin_v,
                                    "fallback_out_of_map": fallback_out_of_map_v,
                                    "fallback_invalid_cost": fallback_invalid_cost_v,
                                    "velocity_search_min_predicted_clearance": velocity_search_min_predicted_clearance_v,
                                    "velocity_search_best_cost": velocity_search_best_cost_v,
                                    "selected_rollout_min_clearance": selected_rollout_min_clearance_v,
                                    "selected_rollout_collision_flag": selected_rollout_collision_flag_v,
                                    "selected_rollout_out_of_map_flag": selected_rollout_out_of_map_flag_v,
                                    "fallback_cmd_x": fallback_cmd_v[0],
                                    "fallback_cmd_y": fallback_cmd_v[1],
                                    "fallback_cmd_w": fallback_cmd_v[2],
                                    "cmd_raw_before_postprocess_x": cmd_raw_post_v[0],
                                    "cmd_raw_before_postprocess_y": cmd_raw_post_v[1],
                                    "cmd_raw_before_postprocess_w": cmd_raw_post_v[2],
                                    "cmd_safe_after_postprocess_x": cmd_safe_post_v[0],
                                    "cmd_safe_after_postprocess_y": cmd_safe_post_v[1],
                                    "cmd_safe_after_postprocess_w": cmd_safe_post_v[2],
                                    "cmd_post_delta_x": cmd_post_delta_v[0],
                                    "cmd_post_delta_y": cmd_post_delta_v[1],
                                    "cmd_post_delta_w": cmd_post_delta_v[2],
                                    "cmd_final_x": cmd_final_v[0],
                                    "cmd_final_y": cmd_final_v[1],
                                    "cmd_final_w": cmd_final_v[2],
                                }
                            )

                    if (not ai.success) and bool(success_step[i].item()):
                        ai.success = True
                        ai.t_success_s = float(ai.step_hl) * float(self.env.high_level_dt)

                done_ids = dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                for i in done_ids:
                    if done_episodes >= per_level_target:
                        break

                    ai = acc[i]
                    follow_mae = float("nan")
                    follow_rmse = float("nan")
                    if ai.follow_err_count > 0:
                        follow_mae = ai.follow_err_sum / float(ai.follow_err_count)
                        follow_rmse = math.sqrt(ai.follow_err_sq_sum / float(ai.follow_err_count))

                    cot = float("nan")
                    if ai.distance_m > 1e-6:
                        cot = ai.energy_j / (self.mass_kg * self.g * ai.distance_m)
                    strict_terminal_success = bool(success_step[i].item())
                    success_event = bool(ai.success)
                    final_collision = bool(ai.episode_collision)
                    row_progress_success = float(np.clip(ai.progress_ratio_best, 0.0, 1.0))
                    full_task_success = bool((row_progress_success >= 1.0 - 1e-6) and (not final_collision))
                    task_success = bool(strict_terminal_success)
                    success_event_and_collision = bool(success_event and final_collision)
                    collision_only = bool(final_collision and row_progress_success <= 1e-6)
                    timeout_or_other = bool((row_progress_success <= 1e-6) and (not final_collision))
                    termination_reason = _episode_termination_reason(
                        task_success=task_success,
                        full_task_success=full_task_success,
                        final_collision=final_collision,
                        target_lost_event=ai.target_lost_event,
                        timeout_or_other=timeout_or_other,
                    )
                    if row_progress_success >= 1.0 - 1e-6:
                        outcome = "full_row_score"
                    elif row_progress_success > 1e-6:
                        outcome = "partial_row_score"
                    elif collision_only:
                        outcome = "collision"
                    else:
                        outcome = "timeout_or_other"

                    denom_steps = float(max(ai.step_hl, 1))
                    gate_region_denom = float(max(ai.gate_region_steps, 1))
                    gate_region_y_eff_mean = (
                        ai.gate_region_y_eff_sum / gate_region_denom
                        if ai.gate_region_steps > 0
                        else float("nan")
                    )
                    gate_region_near_miss_rate = (
                        float(ai.gate_region_near_miss_steps) / gate_region_denom
                        if ai.gate_region_steps > 0
                        else float("nan")
                    )
                    high_risk_steps = int(ai.gate_region_steps)
                    high_risk_denom = float(max(high_risk_steps, 1))
                    high_risk_ratio = float(high_risk_steps) / denom_steps
                    high_risk_y_eff_mean = (
                        ai.high_risk_y_eff_sum / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    high_risk_w_mean = (
                        ai.high_risk_w_sum / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    high_risk_risk_f_mean = (
                        ai.high_risk_risk_f_sum / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    high_risk_risk_a_mean = (
                        ai.high_risk_risk_a_sum / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    high_risk_risk_delta_mean = (
                        (ai.high_risk_risk_f_sum - ai.high_risk_risk_a_sum) / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    high_risk_near_miss_rate = (
                        float(ai.high_risk_near_miss_steps) / high_risk_denom
                        if high_risk_steps > 0
                        else float("nan")
                    )
                    priv_conflict_denom = float(max(ai.priv_conflict_steps, 1))
                    priv_non_conflict_denom = float(max(ai.priv_non_conflict_steps, 1))
                    priv_conflict_delta_y_mean = (
                        ai.priv_conflict_delta_y_sum / priv_conflict_denom
                        if ai.priv_conflict_steps > 0
                        else float("nan")
                    )
                    priv_non_conflict_delta_y_mean = (
                        ai.priv_non_conflict_delta_y_sum / priv_non_conflict_denom
                        if ai.priv_non_conflict_steps > 0
                        else float("nan")
                    )
                    unsafe_conflict_denom = float(max(ai.unsafe_conflict_steps, 1))
                    unsafe_non_conflict_denom = float(max(ai.unsafe_non_conflict_steps, 1))
                    unsafe_conflict_delta_y_mean = (
                        ai.unsafe_conflict_delta_y_sum / unsafe_conflict_denom
                        if ai.unsafe_conflict_steps > 0
                        else float("nan")
                    )
                    unsafe_non_conflict_delta_y_mean = (
                        ai.unsafe_non_conflict_delta_y_sum / unsafe_non_conflict_denom
                        if ai.unsafe_non_conflict_steps > 0
                        else float("nan")
                    )
                    avoid_conflict_denom = float(max(ai.avoid_conflict_steps, 1))
                    stop_conflict_denom = float(max(ai.stop_conflict_steps, 1))
                    avoid_conflict_delta_y_mean = (
                        ai.avoid_conflict_delta_y_sum / avoid_conflict_denom
                        if ai.avoid_conflict_steps > 0
                        else float("nan")
                    )
                    stop_conflict_delta_y_mean = (
                        ai.stop_conflict_delta_y_sum / stop_conflict_denom
                        if ai.stop_conflict_steps > 0
                        else float("nan")
                    )
                    target_bearing_abs_p95 = (
                        _quantile(ai.target_bearing_abs_samples, 0.95)
                        if ai.target_bearing_abs_samples else float("nan")
                    )
                    target_conflict_bearing_abs_mean = (
                        ai.target_conflict_bearing_abs_sum / priv_conflict_denom
                        if ai.priv_conflict_steps > 0 else float("nan")
                    )
                    target_in_fov_rate = ai.target_in_fov_steps / denom_steps
                    l1_safety_success = int(not final_collision)
                    l2_follow_success = int(
                        math.isfinite(follow_mae)
                        and follow_mae <= float(getattr(self.args, "l2_follow_mae_threshold", 0.5))
                        and target_in_fov_rate >= float(getattr(self.args, "l2_target_fov_threshold", 0.90))
                    )
                    l3_progress_success = int(row_progress_success >= 1.0 - 1e-6)
                    cmd_x_denom = float(max(ai.cmd_x_samples, 1))
                    cmd_x_mean = ai.cmd_x_sum / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_sq_mean = ai.cmd_x_sq_sum / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_std = (
                        math.sqrt(max(cmd_x_sq_mean - cmd_x_mean * cmd_x_mean, 0.0))
                        if ai.cmd_x_samples > 0 and math.isfinite(cmd_x_sq_mean) and math.isfinite(cmd_x_mean)
                        else float("nan")
                    )
                    cmd_x_abs_mean = ai.cmd_x_abs_sum / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_positive_rate = ai.cmd_x_positive_steps / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_negative_rate = ai.cmd_x_negative_steps / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_zero_rate = ai.cmd_x_zero_steps / cmd_x_denom if ai.cmd_x_samples > 0 else float("nan")
                    cmd_x_entropy = 0.0
                    if ai.cmd_x_samples > 0:
                        for p_dir in (cmd_x_negative_rate, cmd_x_zero_rate, cmd_x_positive_rate):
                            if math.isfinite(p_dir) and p_dir > 0.0:
                                cmd_x_entropy -= p_dir * math.log(p_dir)
                    else:
                        cmd_x_entropy = float("nan")
                    corr_goalx_cmdx = _corr_from_sums(
                        ai.goal_x_sum,
                        ai.corr_cmd_x_sum,
                        ai.goal_x_sq_sum,
                        ai.corr_cmd_x_sq_sum,
                        ai.goalx_cmdx_sum,
                        ai.goal_cmd_corr_steps,
                    )
                    corr_goalx_cmdyaw = _corr_from_sums(
                        ai.goal_x_sum,
                        ai.corr_cmd_yaw_sum,
                        ai.goal_x_sq_sum,
                        ai.corr_cmd_yaw_sq_sum,
                        ai.goalx_cmdyaw_sum,
                        ai.goal_cmd_corr_steps,
                    )
                    corr_goaly_cmdy = _corr_from_sums(
                        ai.goal_y_sum,
                        ai.corr_cmd_y_sum,
                        ai.goal_y_sq_sum,
                        ai.corr_cmd_y_sq_sum,
                        ai.goaly_cmdy_sum,
                        ai.goal_cmd_corr_steps,
                    )
                    cmd_yaw_left_mean = (
                        ai.cmd_yaw_left_sum / float(ai.goal_x_left_steps)
                        if ai.goal_x_left_steps > 0 else float("nan")
                    )
                    cmd_yaw_right_mean = (
                        ai.cmd_yaw_right_sum / float(ai.goal_x_right_steps)
                        if ai.goal_x_right_steps > 0 else float("nan")
                    )
                    cmd_x_left_mean = (
                        ai.cmd_x_left_sum / float(ai.goal_x_left_steps)
                        if ai.goal_x_left_steps > 0 else float("nan")
                    )
                    cmd_x_right_mean = (
                        ai.cmd_x_right_sum / float(ai.goal_x_right_steps)
                        if ai.goal_x_right_steps > 0 else float("nan")
                    )
                    yaw_directional_response = (
                        cmd_yaw_right_mean - cmd_yaw_left_mean
                        if math.isfinite(cmd_yaw_right_mean) and math.isfinite(cmd_yaw_left_mean)
                        else float("nan")
                    )
                    lateral_directional_response = (
                        cmd_x_right_mean - cmd_x_left_mean
                        if math.isfinite(cmd_x_right_mean) and math.isfinite(cmd_x_left_mean)
                        else float("nan")
                    )
                    cmd_post_denom = float(max(ai.cmd_post_samples, 1))
                    cmd_post_delta_abs_mean = (
                        ai.cmd_post_delta_abs_sum / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    cmd_post_delta_x_abs_mean = (
                        ai.cmd_post_delta_x_abs_sum / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    cmd_post_delta_y_abs_mean = (
                        ai.cmd_post_delta_y_abs_sum / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    cmd_post_delta_y_signed_mean = (
                        ai.cmd_post_delta_y_sum / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    cmd_post_delta_w_abs_mean = (
                        ai.cmd_post_delta_w_abs_sum / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    cmd_post_rewrite_rate = (
                        ai.cmd_post_rewrite_steps / cmd_post_denom
                        if ai.cmd_post_samples > 0 else float("nan")
                    )
                    velocity_search_denom = float(max(ai.velocity_search_steps, 1))
                    velocity_search_filter_change_rate = (
                        ai.velocity_search_filter_change_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_safe_available_rate = (
                        ai.velocity_search_safe_available_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_raw_risk_mean = (
                        ai.velocity_search_raw_risk_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_safe_risk_mean = (
                        ai.velocity_search_safe_risk_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_compute_time_ms_mean = (
                        ai.velocity_search_compute_time_ms_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_dynamic_window_enabled_rate = (
                        ai.velocity_search_dynamic_window_enabled_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_candidate_count_before_dynamic_window_mean = (
                        ai.velocity_search_candidate_count_before_dynamic_window_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_candidate_count_after_dynamic_window_mean = (
                        ai.velocity_search_candidate_count_after_dynamic_window_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_dynamic_window_rejected_count_mean = (
                        ai.velocity_search_dynamic_window_rejected_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_candidate_count_mean = (
                        ai.velocity_search_candidate_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_feasible_count_mean = (
                        ai.velocity_search_feasible_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_infeasible_count_mean = (
                        ai.velocity_search_infeasible_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_feasible_count_after_margin_mean = (
                        ai.velocity_search_feasible_count_after_margin_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_selected_v_lat_mean = (
                        ai.velocity_search_selected_v_lat_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_selected_v_fwd_mean = (
                        ai.velocity_search_selected_v_fwd_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_selected_yaw_mean = (
                        ai.velocity_search_selected_yaw_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_raw_y_mean = (
                        ai.velocity_search_raw_y_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_after_filter_y_mean = (
                        ai.velocity_search_after_filter_y_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_cmd_safe_y_mean = (
                        ai.velocity_search_cmd_safe_y_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_cmd_safe_y_over_raw_y_ratio = (
                        ai.velocity_search_cmd_safe_y_over_raw_y_sum
                        / float(max(ai.velocity_search_cmd_safe_y_over_raw_y_steps, 1))
                        if ai.velocity_search_cmd_safe_y_over_raw_y_steps > 0 else float("nan")
                    )
                    velocity_search_forward_candidate_count_mean = (
                        ai.velocity_search_forward_candidate_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_forward_feasible_count_mean = (
                        ai.velocity_search_forward_feasible_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_forward_infeasible_collision_count_mean = (
                        ai.velocity_search_forward_infeasible_collision_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_forward_infeasible_margin_count_mean = (
                        ai.velocity_search_forward_infeasible_margin_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_forward_infeasible_out_of_map_count_mean = (
                        ai.velocity_search_forward_infeasible_out_of_map_count_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_best_forward_feasible_cost_mean = (
                        ai.velocity_search_best_forward_feasible_cost_sum
                        / float(max(ai.velocity_search_best_forward_feasible_steps, 1))
                        if ai.velocity_search_best_forward_feasible_steps > 0 else float("nan")
                    )
                    velocity_search_best_forward_feasible_clearance_mean = (
                        ai.velocity_search_best_forward_feasible_clearance_sum
                        / float(max(ai.velocity_search_best_forward_feasible_steps, 1))
                        if ai.velocity_search_best_forward_feasible_steps > 0 else float("nan")
                    )
                    actual_motion_denom = float(max(ai.actual_motion_steps, 1))
                    actual_base_vel_x_mean = (
                        ai.actual_base_vel_x_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    actual_base_vel_y_mean = (
                        ai.actual_base_vel_y_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    actual_base_vel_w_mean = (
                        ai.actual_base_vel_w_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    actual_base_delta_x_mean = (
                        ai.actual_base_delta_x_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    actual_base_delta_y_mean = (
                        ai.actual_base_delta_y_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    row_progress_delta_mean = (
                        ai.row_progress_delta_sum / actual_motion_denom
                        if ai.actual_motion_steps > 0 else float("nan")
                    )
                    velocity_search_fallback_rate = (
                        ai.velocity_search_fallback_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_no_feasible_rate = (
                        ai.fallback_no_feasible_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_risk_threshold_rate = (
                        ai.fallback_risk_threshold_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_collision_rate = (
                        ai.fallback_collision_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_margin_rate = (
                        ai.fallback_margin_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_out_of_map_rate = (
                        ai.fallback_out_of_map_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    fallback_invalid_cost_rate = (
                        ai.fallback_invalid_cost_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_min_predicted_clearance_mean = (
                        ai.velocity_search_min_predicted_clearance_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    velocity_search_best_cost_mean = (
                        ai.velocity_search_best_cost_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    selected_rollout_min_clearance_mean = (
                        ai.selected_rollout_min_clearance_sum / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    selected_rollout_collision_rate = (
                        ai.selected_rollout_collision_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    selected_rollout_out_of_map_rate = (
                        ai.selected_rollout_out_of_map_steps / velocity_search_denom
                        if ai.velocity_search_steps > 0 else float("nan")
                    )
                    episode_rows.append(
                        {
                            "episode_id": global_episode_idx,
                            "difficulty": float(d),
                            "success": int(task_success),
                            "task_success": int(task_success),
                            "l1_safety_success": l1_safety_success,
                            "l2_follow_success": l2_follow_success,
                            "l3_progress_success": l3_progress_success,
                            "row_progress_success": row_progress_success,
                            "full_task_success": int(full_task_success),
                            "strict_success": int(strict_terminal_success),
                            "strict_terminal_success": int(strict_terminal_success),
                            "success_event": int(success_event),
                            "success_event_and_collision": int(success_event_and_collision),
                            "time_to_success_s": (
                                float(ai.t_success_s)
                                if success_event and math.isfinite(float(ai.t_success_s))
                                else (float(ai.step_hl) * float(self.env.high_level_dt) if success_event else float("nan"))
                            ),
                            "success_event_time_s": float(ai.t_success_s) if success_event else float("nan"),
                            "episode_collision": int(final_collision),
                            "collision_time_s": float(ai.t_collision_s) if final_collision else float("nan"),
                            "collision_only": int(collision_only),
                            "success_and_collision": int(success_event_and_collision),
                            "timeout_or_other": int(timeout_or_other),
                            "outcome": outcome,
                            "follow_mae_m": follow_mae,
                            "follow_rmse_m": follow_rmse,
                            "cot": cot,
                            "energy_j": ai.energy_j,
                            "distance_m": ai.distance_m,
                            "steps_hl": ai.step_hl,
                            "cross_line_dist_end": ai.cross_line_dist_end,
                            "cross_line_dist_min": ai.cross_line_dist_min if math.isfinite(ai.cross_line_dist_min) else float("nan"),
                            "progress_reached": int(ai.progress_reached),
                            "progress_ratio_best": ai.progress_ratio_best,
                            "gate_y_raw_mean": ai.gate_y_raw_sum / denom_steps,
                            "y_eff_mean": ai.y_eff_sum / denom_steps,
                            "w_mean": ai.w_sum / denom_steps,
                            "signed_w_mean": ai.signed_w_sum / denom_steps,
                            "signed_w_active_mean": ai.signed_w_active_sum / denom_steps,
                            "clearance_f_mean": ai.clearance_f_sum / denom_steps,
                            "clearance_a_mean": ai.clearance_a_sum / denom_steps,
                            "risk_f_mean": ai.risk_f_sum / denom_steps,
                            "risk_a_mean": ai.risk_a_sum / denom_steps,
                            "risk_delta_mean": (ai.risk_f_sum - ai.risk_a_sum) / denom_steps,
                            "clearance_rollout_f_mean": ai.clearance_rollout_f_sum / denom_steps,
                            "clearance_rollout_a_mean": ai.clearance_rollout_a_sum / denom_steps,
                            "clearance_rollout_s_mean": ai.clearance_rollout_s_sum / denom_steps,
                            "risk_rollout_f_mean": ai.risk_rollout_f_sum / denom_steps,
                            "risk_rollout_a_mean": ai.risk_rollout_a_sum / denom_steps,
                            "risk_rollout_s_mean": ai.risk_rollout_s_sum / denom_steps,
                            "risk_rollout_gap_f_min_as_mean": ai.risk_rollout_gap_f_min_as_sum / denom_steps,
                            "w_support_correction_mean": ai.w_support_correction_sum / denom_steps,
                            "risk_diff_correction_mean": ai.risk_diff_correction_sum / denom_steps,
                            "delta_y_w_raw_mean": ai.delta_y_w_raw_sum / denom_steps,
                            "delta_y_w_used_mean": ai.delta_y_w_used_sum / denom_steps,
                            "delta_y_r_mean": ai.delta_y_r_sum / denom_steps,
                            "delta_y_total_mean": ai.delta_y_total_sum / denom_steps,
                            "risk_memory_mean": ai.risk_memory_sum / denom_steps,
                            "rule_s_mean": (
                                ai.rule_s_sum / denom_steps
                                if bool(getattr(self.args, "rule_override", False)) else float("nan")
                            ),
                            "rule_risk_gap_mean": (
                                ai.rule_risk_gap_sum / denom_steps
                                if bool(getattr(self.args, "rule_override", False)) else float("nan")
                            ),
                            "rule_follow_scale_mean": (
                                ai.rule_follow_scale_sum / denom_steps
                                if bool(getattr(self.args, "rule_override", False)) else float("nan")
                            ),
                            "rule_yaw_scale_mean": (
                                ai.rule_yaw_scale_sum / denom_steps
                                if bool(getattr(self.args, "rule_override", False)) else float("nan")
                            ),
                            "rule_follow_suppression_mean": (
                                ai.rule_follow_suppression_sum / denom_steps
                                if bool(getattr(self.args, "rule_override", False)) else float("nan")
                            ),
                            "rule_s_at_avoid_conflict": (
                                ai.rule_avoid_conflict_s_sum / avoid_conflict_denom
                                if bool(getattr(self.args, "rule_override", False)) and ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "rule_follow_scale_at_avoid_conflict": (
                                ai.rule_avoid_conflict_follow_scale_sum / avoid_conflict_denom
                                if bool(getattr(self.args, "rule_override", False)) and ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "rule_yaw_scale_at_avoid_conflict": (
                                ai.rule_avoid_conflict_yaw_scale_sum / avoid_conflict_denom
                                if bool(getattr(self.args, "rule_override", False)) and ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "rule_follow_suppression_at_avoid_conflict": (
                                ai.rule_avoid_conflict_follow_suppression_sum / avoid_conflict_denom
                                if bool(getattr(self.args, "rule_override", False)) and ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_delta_y_w_raw_mean": (
                                ai.avoid_conflict_delta_y_w_raw_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_delta_y_w_used_mean": (
                                ai.avoid_conflict_delta_y_w_used_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_delta_y_r_mean": (
                                ai.avoid_conflict_delta_y_r_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_delta_y_total_mean": (
                                ai.avoid_conflict_delta_y_total_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "row_not_released_rate": ai.row_not_released_sum / denom_steps,
                            "row_not_released_w_mean": (
                                ai.row_not_released_w_sum / float(max(ai.row_not_released_steps, 1))
                                if ai.row_not_released_steps > 0 else float("nan")
                            ),
                            "row_released_w_mean": (
                                ai.row_released_w_sum / float(max(ai.row_released_steps, 1))
                                if ai.row_released_steps > 0 else float("nan")
                            ),
                            "priv_conflict_score_mean": ai.priv_conflict_score_sum / denom_steps,
                            "priv_high_conflict_steps": ai.priv_conflict_steps,
                            "priv_high_conflict_step_rate": ai.priv_conflict_steps / denom_steps,
                            "priv_obstacle_window_rate": ai.priv_obstacle_window_steps / denom_steps,
                            "priv_follow_pressure_rate": ai.priv_follow_pressure_steps / denom_steps,
                            "priv_avoid_pressure_rate": ai.priv_avoid_pressure_steps / denom_steps,
                            "priv_conflict_y_raw_mean": (
                                ai.priv_conflict_y_raw_sum / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_y_eff_mean": (
                                ai.priv_conflict_y_eff_sum / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_w_mean": (
                                ai.priv_conflict_w_sum / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_signed_w_mean": (
                                ai.priv_conflict_signed_w_sum / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_delta_y_mean": priv_conflict_delta_y_mean,
                            "priv_non_conflict_delta_y_mean": priv_non_conflict_delta_y_mean,
                            "conflict_suppression_index": (
                                -priv_conflict_delta_y_mean
                                if math.isfinite(priv_conflict_delta_y_mean) else float("nan")
                            ),
                            "conflict_selective_suppression": (
                                priv_non_conflict_delta_y_mean - priv_conflict_delta_y_mean
                                if math.isfinite(priv_conflict_delta_y_mean)
                                and math.isfinite(priv_non_conflict_delta_y_mean)
                                else float("nan")
                            ),
                            "relative_conflict_modulation": (
                                priv_non_conflict_delta_y_mean - priv_conflict_delta_y_mean
                                if math.isfinite(priv_conflict_delta_y_mean)
                                and math.isfinite(priv_non_conflict_delta_y_mean)
                                else float("nan")
                            ),
                            "priv_window_phase_approach_rate": ai.priv_window_phase_approach_steps / denom_steps,
                            "priv_window_phase_inside_rate": ai.priv_window_phase_inside_steps / denom_steps,
                            "priv_window_phase_release_rate": ai.priv_window_phase_release_steps / denom_steps,
                            "priv_conflict_phase_approach_rate": (
                                ai.priv_conflict_phase_approach_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_inside_rate": (
                                ai.priv_conflict_phase_inside_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_release_rate": (
                                ai.priv_conflict_phase_release_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_approach_steps": ai.priv_conflict_phase_approach_steps,
                            "priv_conflict_phase_inside_steps": ai.priv_conflict_phase_inside_steps,
                            "priv_conflict_phase_release_steps": ai.priv_conflict_phase_release_steps,
                            "priv_conflict_phase_approach_w_mean": (
                                ai.priv_conflict_phase_approach_w_sum / float(ai.priv_conflict_phase_approach_steps)
                                if ai.priv_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_inside_w_mean": (
                                ai.priv_conflict_phase_inside_w_sum / float(ai.priv_conflict_phase_inside_steps)
                                if ai.priv_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_release_w_mean": (
                                ai.priv_conflict_phase_release_w_sum / float(ai.priv_conflict_phase_release_steps)
                                if ai.priv_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_approach_signed_w_mean": (
                                ai.priv_conflict_phase_approach_signed_w_sum / float(ai.priv_conflict_phase_approach_steps)
                                if ai.priv_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_inside_signed_w_mean": (
                                ai.priv_conflict_phase_inside_signed_w_sum / float(ai.priv_conflict_phase_inside_steps)
                                if ai.priv_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_release_signed_w_mean": (
                                ai.priv_conflict_phase_release_signed_w_sum / float(ai.priv_conflict_phase_release_steps)
                                if ai.priv_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_approach_delta_y_mean": (
                                ai.priv_conflict_phase_approach_delta_y_sum / float(ai.priv_conflict_phase_approach_steps)
                                if ai.priv_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_inside_delta_y_mean": (
                                ai.priv_conflict_phase_inside_delta_y_sum / float(ai.priv_conflict_phase_inside_steps)
                                if ai.priv_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "priv_conflict_phase_release_delta_y_mean": (
                                ai.priv_conflict_phase_release_delta_y_sum / float(ai.priv_conflict_phase_release_steps)
                                if ai.priv_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_steps": ai.unsafe_conflict_steps,
                            "unsafe_conflict_step_rate": ai.unsafe_conflict_steps / denom_steps,
                            "unsafe_follow_risk_rate": ai.unsafe_follow_risk_steps / denom_steps,
                            "unsafe_safe_candidate_better_rate": ai.unsafe_safe_candidate_better_steps / denom_steps,
                            "unsafe_avoid_safer_rate": ai.unsafe_avoid_safer_steps / denom_steps,
                            "unsafe_stop_safer_rate": ai.unsafe_stop_safer_steps / denom_steps,
                            "unsafe_command_disagree_rate": ai.unsafe_command_disagree_steps / denom_steps,
                            "unsafe_target_recoverable_rate": ai.unsafe_target_recoverable_steps / denom_steps,
                            "unsafe_conflict_y_raw_mean": (
                                ai.unsafe_conflict_y_raw_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_y_eff_mean": (
                                ai.unsafe_conflict_y_eff_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_w_mean": (
                                ai.unsafe_conflict_w_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_signed_w_mean": (
                                ai.unsafe_conflict_signed_w_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_delta_y_mean": unsafe_conflict_delta_y_mean,
                            "unsafe_conflict_delta_y_w_raw_mean": (
                                ai.unsafe_conflict_delta_y_w_raw_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_delta_y_w_used_mean": (
                                ai.unsafe_conflict_delta_y_w_used_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_delta_y_r_mean": (
                                ai.unsafe_conflict_delta_y_r_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_delta_y_total_mean": (
                                ai.unsafe_conflict_delta_y_total_sum / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_non_conflict_delta_y_mean": unsafe_non_conflict_delta_y_mean,
                            "unsafe_conflict_suppression_index": (
                                -unsafe_conflict_delta_y_mean
                                if math.isfinite(unsafe_conflict_delta_y_mean) else float("nan")
                            ),
                            "unsafe_conflict_selective_suppression": (
                                unsafe_non_conflict_delta_y_mean - unsafe_conflict_delta_y_mean
                                if math.isfinite(unsafe_conflict_delta_y_mean)
                                and math.isfinite(unsafe_non_conflict_delta_y_mean)
                                else float("nan")
                            ),
                            "unsafe_relative_conflict_modulation": (
                                unsafe_non_conflict_delta_y_mean - unsafe_conflict_delta_y_mean
                                if math.isfinite(unsafe_conflict_delta_y_mean)
                                and math.isfinite(unsafe_non_conflict_delta_y_mean)
                                else float("nan")
                            ),
                            "unsafe_conflict_phase_approach_rate": (
                                ai.unsafe_conflict_phase_approach_steps / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_inside_rate": (
                                ai.unsafe_conflict_phase_inside_steps / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_release_rate": (
                                ai.unsafe_conflict_phase_release_steps / unsafe_conflict_denom
                                if ai.unsafe_conflict_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_approach_w_mean": (
                                ai.unsafe_conflict_phase_approach_w_sum / float(ai.unsafe_conflict_phase_approach_steps)
                                if ai.unsafe_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_inside_w_mean": (
                                ai.unsafe_conflict_phase_inside_w_sum / float(ai.unsafe_conflict_phase_inside_steps)
                                if ai.unsafe_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_release_w_mean": (
                                ai.unsafe_conflict_phase_release_w_sum / float(ai.unsafe_conflict_phase_release_steps)
                                if ai.unsafe_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_approach_signed_w_mean": (
                                ai.unsafe_conflict_phase_approach_signed_w_sum / float(ai.unsafe_conflict_phase_approach_steps)
                                if ai.unsafe_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_inside_signed_w_mean": (
                                ai.unsafe_conflict_phase_inside_signed_w_sum / float(ai.unsafe_conflict_phase_inside_steps)
                                if ai.unsafe_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_release_signed_w_mean": (
                                ai.unsafe_conflict_phase_release_signed_w_sum / float(ai.unsafe_conflict_phase_release_steps)
                                if ai.unsafe_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_approach_delta_y_mean": (
                                ai.unsafe_conflict_phase_approach_delta_y_sum / float(ai.unsafe_conflict_phase_approach_steps)
                                if ai.unsafe_conflict_phase_approach_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_inside_delta_y_mean": (
                                ai.unsafe_conflict_phase_inside_delta_y_sum / float(ai.unsafe_conflict_phase_inside_steps)
                                if ai.unsafe_conflict_phase_inside_steps > 0 else float("nan")
                            ),
                            "unsafe_conflict_phase_release_delta_y_mean": (
                                ai.unsafe_conflict_phase_release_delta_y_sum / float(ai.unsafe_conflict_phase_release_steps)
                                if ai.unsafe_conflict_phase_release_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_steps": ai.avoid_conflict_steps,
                            "avoid_conflict_step_rate": ai.avoid_conflict_steps / denom_steps,
                            "avoid_conflict_y_raw_mean": (
                                ai.avoid_conflict_y_raw_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_y_eff_mean": (
                                ai.avoid_conflict_y_eff_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_cmd_y_mean": (
                                ai.avoid_conflict_cmd_y_sum / float(ai.avoid_conflict_cmd_y_steps)
                                if ai.avoid_conflict_cmd_y_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_cmd_f_y_mean": (
                                ai.avoid_conflict_cmd_f_y_sum / float(ai.avoid_conflict_cmd_f_y_steps)
                                if ai.avoid_conflict_cmd_f_y_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_cmd_a_y_mean": (
                                ai.avoid_conflict_cmd_a_y_sum / float(ai.avoid_conflict_cmd_a_y_steps)
                                if ai.avoid_conflict_cmd_a_y_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_w_mean": (
                                ai.avoid_conflict_w_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_signed_w_mean": (
                                ai.avoid_conflict_signed_w_sum / avoid_conflict_denom
                                if ai.avoid_conflict_steps > 0 else float("nan")
                            ),
                            "avoid_conflict_delta_y_mean": avoid_conflict_delta_y_mean,
                            "avoid_conflict_suppression_index": (
                                -avoid_conflict_delta_y_mean
                                if math.isfinite(avoid_conflict_delta_y_mean) else float("nan")
                            ),
                            "stop_conflict_steps": ai.stop_conflict_steps,
                            "stop_conflict_step_rate": ai.stop_conflict_steps / denom_steps,
                            "stop_conflict_y_raw_mean": (
                                ai.stop_conflict_y_raw_sum / stop_conflict_denom
                                if ai.stop_conflict_steps > 0 else float("nan")
                            ),
                            "stop_conflict_y_eff_mean": (
                                ai.stop_conflict_y_eff_sum / stop_conflict_denom
                                if ai.stop_conflict_steps > 0 else float("nan")
                            ),
                            "stop_conflict_w_mean": (
                                ai.stop_conflict_w_sum / stop_conflict_denom
                                if ai.stop_conflict_steps > 0 else float("nan")
                            ),
                            "stop_conflict_signed_w_mean": (
                                ai.stop_conflict_signed_w_sum / stop_conflict_denom
                                if ai.stop_conflict_steps > 0 else float("nan")
                            ),
                            "stop_conflict_delta_y_mean": stop_conflict_delta_y_mean,
                            "stop_conflict_suppression_index": (
                                -stop_conflict_delta_y_mean
                                if math.isfinite(stop_conflict_delta_y_mean) else float("nan")
                            ),
                            "target_bearing_abs_mean": (
                                ai.target_bearing_abs_sum / float(len(ai.target_bearing_abs_samples))
                                if ai.target_bearing_abs_samples else float("nan")
                            ),
                            "target_bearing_abs_p95": target_bearing_abs_p95,
                            "target_bearing_abs_max": ai.target_bearing_abs_max,
                            "target_in_rgb_fov_rate": target_in_fov_rate,
                            "target_near_rgb_fov_edge_rate": ai.target_near_fov_edge_steps / denom_steps,
                            "target_lost_step_rate": ai.target_lost_steps / denom_steps,
                            "target_lost_episode": int(ai.target_lost_event),
                            "target_lost_max_consecutive_steps": ai.target_lost_max_consecutive_steps,
                            "target_bearing_abs_mean_in_priv_conflict": target_conflict_bearing_abs_mean,
                            "target_bearing_abs_max_in_priv_conflict": (
                                ai.target_conflict_bearing_abs_max
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "target_in_fov_rate_in_priv_conflict": (
                                ai.target_conflict_in_fov_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "target_near_fov_edge_rate_in_priv_conflict": (
                                ai.target_conflict_near_fov_edge_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "target_lost_rate_in_priv_conflict": (
                                ai.target_conflict_lost_steps / priv_conflict_denom
                                if ai.priv_conflict_steps > 0 else float("nan")
                            ),
                            "cmd_x_mean": cmd_x_mean,
                            "cmd_x_std": cmd_x_std,
                            "cmd_x_abs_mean": cmd_x_abs_mean,
                            "cmd_x_positive_rate": cmd_x_positive_rate,
                            "cmd_x_negative_rate": cmd_x_negative_rate,
                            "cmd_x_zero_rate": cmd_x_zero_rate,
                            "cmd_x_direction_entropy": cmd_x_entropy,
                            "corr_goalx_cmdx": corr_goalx_cmdx,
                            "corr_goalx_cmdyaw": corr_goalx_cmdyaw,
                            "corr_goaly_cmdy": corr_goaly_cmdy,
                            "cmd_yaw_left_mean": cmd_yaw_left_mean,
                            "cmd_yaw_right_mean": cmd_yaw_right_mean,
                            "cmd_x_left_mean": cmd_x_left_mean,
                            "cmd_x_right_mean": cmd_x_right_mean,
                            "yaw_directional_response": yaw_directional_response,
                            "lateral_directional_response": lateral_directional_response,
                            "cmd_post_delta_abs_mean": cmd_post_delta_abs_mean,
                            "cmd_post_delta_x_abs_mean": cmd_post_delta_x_abs_mean,
                            "cmd_post_delta_y_abs_mean": cmd_post_delta_y_abs_mean,
                            "cmd_post_delta_y_signed_mean": cmd_post_delta_y_signed_mean,
                            "cmd_post_delta_w_abs_mean": cmd_post_delta_w_abs_mean,
                            "cmd_post_rewrite_rate": cmd_post_rewrite_rate,
                            "velocity_search_filter_change_rate": velocity_search_filter_change_rate,
                            "velocity_search_safe_available_rate": velocity_search_safe_available_rate,
                            "velocity_search_raw_risk_mean": velocity_search_raw_risk_mean,
                            "velocity_search_safe_risk_mean": velocity_search_safe_risk_mean,
                            "velocity_search_compute_time_ms_mean": velocity_search_compute_time_ms_mean,
                            "velocity_search_dynamic_window_enabled_rate": velocity_search_dynamic_window_enabled_rate,
                            "velocity_search_candidate_count_before_dynamic_window_mean": velocity_search_candidate_count_before_dynamic_window_mean,
                            "velocity_search_candidate_count_after_dynamic_window_mean": velocity_search_candidate_count_after_dynamic_window_mean,
                            "velocity_search_dynamic_window_rejected_count_mean": velocity_search_dynamic_window_rejected_count_mean,
                            "velocity_search_candidate_count_mean": velocity_search_candidate_count_mean,
                            "velocity_search_feasible_count_mean": velocity_search_feasible_count_mean,
                            "velocity_search_infeasible_count_mean": velocity_search_infeasible_count_mean,
                            "velocity_search_feasible_count_after_margin_mean": velocity_search_feasible_count_after_margin_mean,
                            "velocity_search_selected_v_lat_mean": velocity_search_selected_v_lat_mean,
                            "velocity_search_selected_v_fwd_mean": velocity_search_selected_v_fwd_mean,
                            "velocity_search_selected_yaw_mean": velocity_search_selected_yaw_mean,
                            "velocity_search_raw_y_mean": velocity_search_raw_y_mean,
                            "velocity_search_after_filter_y_mean": velocity_search_after_filter_y_mean,
                            "cmd_safe_y_mean": velocity_search_cmd_safe_y_mean,
                            "cmd_safe_y_over_raw_y_ratio": velocity_search_cmd_safe_y_over_raw_y_ratio,
                            "actual_base_vel_x_mean": actual_base_vel_x_mean,
                            "actual_base_vel_y_mean": actual_base_vel_y_mean,
                            "actual_base_vel_w_mean": actual_base_vel_w_mean,
                            "actual_base_delta_x_mean": actual_base_delta_x_mean,
                            "actual_base_delta_y_mean": actual_base_delta_y_mean,
                            "row_progress_delta_mean": row_progress_delta_mean,
                            "velocity_search_forward_candidate_count_mean": velocity_search_forward_candidate_count_mean,
                            "velocity_search_forward_feasible_count_mean": velocity_search_forward_feasible_count_mean,
                            "velocity_search_forward_infeasible_collision_count_mean": velocity_search_forward_infeasible_collision_count_mean,
                            "velocity_search_forward_infeasible_margin_count_mean": velocity_search_forward_infeasible_margin_count_mean,
                            "velocity_search_forward_infeasible_out_of_map_count_mean": velocity_search_forward_infeasible_out_of_map_count_mean,
                            "velocity_search_best_forward_feasible_cost_mean": velocity_search_best_forward_feasible_cost_mean,
                            "velocity_search_best_forward_feasible_clearance_mean": velocity_search_best_forward_feasible_clearance_mean,
                            "velocity_search_fallback_rate": velocity_search_fallback_rate,
                            "fallback_no_feasible_rate": fallback_no_feasible_rate,
                            "fallback_risk_threshold_rate": fallback_risk_threshold_rate,
                            "fallback_collision_rate": fallback_collision_rate,
                            "fallback_margin_rate": fallback_margin_rate,
                            "fallback_out_of_map_rate": fallback_out_of_map_rate,
                            "fallback_invalid_cost_rate": fallback_invalid_cost_rate,
                            "velocity_search_min_predicted_clearance_mean": velocity_search_min_predicted_clearance_mean,
                            "velocity_search_best_cost_mean": velocity_search_best_cost_mean,
                            "selected_rollout_min_clearance_mean": selected_rollout_min_clearance_mean,
                            "selected_rollout_collision_rate": selected_rollout_collision_rate,
                            "selected_rollout_out_of_map_rate": selected_rollout_out_of_map_rate,
                            "goal_x_left_steps": ai.goal_x_left_steps,
                            "goal_x_right_steps": ai.goal_x_right_steps,
                            "goal_cmd_corr_steps": ai.goal_cmd_corr_steps,
                            "switch_rate": ai.gate_switch_count / denom_steps,
                            "near_miss_rate": ai.near_miss_steps / denom_steps,
                            "rotate_only_rate": ai.rotate_only_steps / denom_steps,
                            "w_trigger_step": ai.w_trigger_step,
                            "w_trigger_progress": ai.w_trigger_progress,
                            "gate_region_steps": ai.gate_region_steps,
                            "gate_region_y_eff_mean": gate_region_y_eff_mean,
                            "gate_region_near_miss_rate": gate_region_near_miss_rate,
                            "high_risk_steps": high_risk_steps,
                            "high_risk_ratio": high_risk_ratio,
                            "high_risk_y_eff_mean": high_risk_y_eff_mean,
                            "high_risk_w_mean": high_risk_w_mean,
                            "high_risk_risk_f_mean": high_risk_risk_f_mean,
                            "high_risk_risk_a_mean": high_risk_risk_a_mean,
                            "high_risk_risk_delta_mean": high_risk_risk_delta_mean,
                            "high_risk_near_miss_rate": high_risk_near_miss_rate,
                            "risk_bin_stats": [dict(bin_state) for bin_state in ai.risk_bin_stats],
                            "conflict_bin_stats": [dict(bin_state) for bin_state in ai.conflict_bin_stats],
                            "priv_conflict_bin_stats": [dict(bin_state) for bin_state in ai.priv_conflict_bin_stats],
                            "cmd_jerk_lin_mean": ai.cmd_jerk_lin_sum / denom_steps,
                            "cmd_jerk_ang_mean": ai.cmd_jerk_ang_sum / denom_steps,
                            "cmd_f_mean_x": ai.cmd_f_sum[0] / denom_steps,
                            "cmd_f_mean_y": ai.cmd_f_sum[1] / denom_steps,
                            "cmd_f_mean_w": ai.cmd_f_sum[2] / denom_steps,
                            "cmd_a_mean_x": ai.cmd_a_sum[0] / denom_steps,
                            "cmd_a_mean_y": ai.cmd_a_sum[1] / denom_steps,
                            "cmd_a_mean_w": ai.cmd_a_sum[2] / denom_steps,
                            "cmd_final_mean_x": ai.cmd_final_sum[0] / denom_steps,
                            "cmd_final_mean_y": ai.cmd_final_sum[1] / denom_steps,
                            "cmd_final_mean_w": ai.cmd_final_sum[2] / denom_steps,
                        }
                    )
                    if dump_timeseries and len(timeseries_rows) < max(1, timeseries_limit) * 100000:
                        if global_episode_idx < timeseries_limit:
                            for ts_row in ai.timeseries:
                                ts_row = dict(ts_row)
                                ts_row["episode_id"] = global_episode_idx
                                ts_row["difficulty"] = float(d)
                                ts_row["episode_termination_reason"] = termination_reason
                                ts_row["episode_task_success"] = int(task_success)
                                ts_row["episode_full_task_success"] = int(full_task_success)
                                ts_row["episode_collision_final"] = int(final_collision)
                                ts_row["episode_target_lost"] = int(ai.target_lost_event)
                                ts_row["episode_row_progress"] = row_progress_success
                                timeseries_rows.append(ts_row)
                    global_episode_idx += 1
                    done_episodes += 1

                    acc[i] = EpisodeAccumulator()

                now_t = time.perf_counter()
                if progress_interval_s > 0.0 and (now_t - last_progress_t) >= progress_interval_s:
                    elapsed = now_t - eval_start_t
                    overall_done = min(global_episode_idx, episodes_total)
                    print(
                        f"[Eval] progress level={level_idx + 1}/{len(difficulty_levels)} "
                        f"difficulty={float(d):.3f} "
                        f"level_eps={done_episodes}/{per_level_target} "
                        f"total_eps={overall_done}/{episodes_total} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    last_progress_t = now_t

                self.done_prev = dones.clone()
                obs = next_obs

        # Trim overshoot to exact requested episode count.
        episode_rows = episode_rows[:episodes_total]

        metrics = self._aggregate_metrics(episode_rows, latency_ms_samples)
        if dump_timeseries:
            metrics["timeseries"] = timeseries_rows
        return metrics

    def _aggregate_metrics(self, rows: List[Dict], latency_ms_samples: List[float]) -> Dict:
        def _clean(values: List[float]) -> List[float]:
            out = []
            for v in values:
                if v is None:
                    continue
                vv = float(v)
                if np.isfinite(vv):
                    out.append(vv)
            return out

        def _high_risk_summary(sub_rows: List[Dict]) -> Dict:
            high_rows = [r for r in sub_rows if int(r.get("high_risk_steps", 0)) > 0]
            n_high = len(high_rows)
            return {
                "high_risk_episode_rate": float(n_high / max(1, len(sub_rows))),
                "high_risk_success_rate": (
                    float(sum(float(r.get("success", 0.0)) for r in high_rows) / max(1, n_high))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_collision_rate": (
                    float(sum(int(r.get("episode_collision", 0)) for r in high_rows) / max(1, n_high))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_progress_rate": (
                    float(sum(int(r.get("progress_reached", 0)) for r in high_rows) / max(1, n_high))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_y_eff_mean": (
                    float(np.mean(_clean([r.get("high_risk_y_eff_mean", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_w_mean": (
                    float(np.mean(_clean([r.get("high_risk_w_mean", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_risk_f_mean": (
                    float(np.mean(_clean([r.get("high_risk_risk_f_mean", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_risk_a_mean": (
                    float(np.mean(_clean([r.get("high_risk_risk_a_mean", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_risk_delta_mean": (
                    float(np.mean(_clean([r.get("high_risk_risk_delta_mean", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
                "high_risk_near_miss_rate_mean": (
                    float(np.mean(_clean([r.get("high_risk_near_miss_rate", float("nan")) for r in high_rows])))
                    if n_high > 0
                    else float("nan")
                ),
            }

        def _risk_bins_summary(sub_rows: List[Dict], stats_key: str = "risk_bin_stats") -> List[Dict]:
            bins = _empty_risk_bin_state()
            episode_sets = [
                {
                    "episodes": set(),
                    "success": set(),
                    "success_score_sum": 0.0,
                    "row_progress_sum": 0.0,
                    "success_event": set(),
                    "collision": set(),
                    "progress": set(),
                }
                for _ in RISK_BIN_LABELS
            ]
            for row_idx, row in enumerate(sub_rows):
                row_bins = row.get(stats_key, [])
                if not isinstance(row_bins, list):
                    continue
                for idx, row_bin in enumerate(row_bins[:len(RISK_BIN_LABELS)]):
                    if not isinstance(row_bin, dict):
                        continue
                    steps = int(row_bin.get("steps", 0) or 0)
                    if steps <= 0:
                        continue
                    bins[idx]["steps"] += steps
                    for key in (
                        "gate_y_raw_sum",
                        "gate_y_raw_sq_sum",
                        "y_eff_sum",
                        "y_eff_sq_sum",
                        "suppression_sum",
                        "suppression_sq_sum",
                        "w_sum",
                        "w_sq_sum",
                        "signed_w_sum",
                        "signed_w_active_sum",
                        "risk_memory_sum",
                        "risk_f_sum",
                        "risk_a_sum",
                        "risk_delta_sum",
                        "w_support_correction_sum",
                        "risk_diff_correction_sum",
                        "near_miss_steps",
                    ):
                        bins[idx][key] += float(row_bin.get(key, 0.0) or 0.0)
                    episode_sets[idx]["episodes"].add(row_idx)
                    task_success_score = float(row.get("task_success", row.get("success", 0.0)) or 0.0)
                    row_progress_score = float(row.get("row_progress_success", 0.0) or 0.0)
                    episode_sets[idx]["success_score_sum"] += task_success_score
                    episode_sets[idx]["row_progress_sum"] += row_progress_score
                    if task_success_score > 1e-6:
                        episode_sets[idx]["success"].add(row_idx)
                    if int(row.get("success_event", 0)):
                        episode_sets[idx]["success_event"].add(row_idx)
                    if int(row.get("episode_collision", 0)):
                        episode_sets[idx]["collision"].add(row_idx)
                    if int(row.get("progress_reached", 0)):
                        episode_sets[idx]["progress"].add(row_idx)
            out = []
            for idx, label in enumerate(RISK_BIN_LABELS):
                steps = int(bins[idx]["steps"])
                episode_count = len(episode_sets[idx]["episodes"])
                gate_y_raw_mean = bins[idx]["gate_y_raw_sum"] / float(steps) if steps > 0 else float("nan")
                y_eff_mean = bins[idx]["y_eff_sum"] / float(steps) if steps > 0 else float("nan")
                suppression_mean = (
                    bins[idx]["suppression_sum"] / float(steps)
                    if bins[idx].get("suppression_sum", 0.0) != 0.0 or bins[idx].get("suppression_sq_sum", 0.0) != 0.0
                    else gate_y_raw_mean - y_eff_mean
                ) if steps > 0 else float("nan")

                def _sem_from_sums(sum_v: float, sq_sum_v: float) -> float:
                    if steps <= 1:
                        return float("nan")
                    mean_v = sum_v / float(steps)
                    var_v = max(0.0, sq_sum_v / float(steps) - mean_v * mean_v)
                    return math.sqrt(var_v / float(steps))

                out.append({
                    "bin": label,
                    "low": float(RISK_BIN_EDGES[idx]),
                    "high": float(min(RISK_BIN_EDGES[idx + 1], 1.0)),
                    "steps": steps,
                    "episode_count": episode_count,
                    "gate_y_raw_mean": gate_y_raw_mean,
                    "gate_y_raw_sem": _sem_from_sums(bins[idx]["gate_y_raw_sum"], bins[idx]["gate_y_raw_sq_sum"]),
                    "y_eff_mean": y_eff_mean,
                    "y_eff_sem": _sem_from_sums(bins[idx]["y_eff_sum"], bins[idx]["y_eff_sq_sum"]),
                    "suppression_mean": suppression_mean,
                    "suppression_sem": _sem_from_sums(bins[idx]["suppression_sum"], bins[idx]["suppression_sq_sum"]),
                    "w_mean": bins[idx]["w_sum"] / float(steps) if steps > 0 else float("nan"),
                    "w_sem": _sem_from_sums(bins[idx]["w_sum"], bins[idx]["w_sq_sum"]),
                    "signed_w_mean": bins[idx]["signed_w_sum"] / float(steps) if steps > 0 else float("nan"),
                    "signed_w_active_mean": bins[idx]["signed_w_active_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_memory_mean": bins[idx]["risk_memory_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_f_mean": bins[idx]["risk_f_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_a_mean": bins[idx]["risk_a_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_delta_mean": bins[idx]["risk_delta_sum"] / float(steps) if steps > 0 else float("nan"),
                    "w_support_correction_mean": bins[idx]["w_support_correction_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_diff_correction_mean": bins[idx]["risk_diff_correction_sum"] / float(steps) if steps > 0 else float("nan"),
                    "near_miss_rate": bins[idx]["near_miss_steps"] / float(steps) if steps > 0 else float("nan"),
                    "success_episode_rate": (
                        episode_sets[idx]["success_score_sum"] / float(episode_count)
                        if episode_count > 0
                        else float("nan")
                    ),
                    "row_progress_success_mean": (
                        episode_sets[idx]["row_progress_sum"] / float(episode_count)
                        if episode_count > 0
                        else float("nan")
                    ),
                    "success_event_episode_rate": (
                        len(episode_sets[idx]["success_event"]) / float(episode_count)
                        if episode_count > 0
                        else float("nan")
                    ),
                    "collision_episode_rate": (
                        len(episode_sets[idx]["collision"]) / float(episode_count)
                        if episode_count > 0
                        else float("nan")
                    ),
                    "progress_episode_rate": (
                        len(episode_sets[idx]["progress"]) / float(episode_count)
                        if episode_count > 0
                        else float("nan")
                    ),
                })
            return out

        def _weighted_step_mean(sub_rows: List[Dict], mean_key: str, steps_key: str, fallback_steps: str = "steps_hl") -> float:
            weighted_sum = 0.0
            weight_sum = 0.0
            for row in sub_rows:
                value = _safe_float(row.get(mean_key, float("nan")), default=float("nan"))
                steps = _safe_float(row.get(steps_key, row.get(fallback_steps, 0.0)), default=0.0)
                if math.isfinite(value) and steps > 0.0:
                    weighted_sum += value * steps
                    weight_sum += steps
            return weighted_sum / weight_sum if weight_sum > 0.0 else float("nan")

        def _priv_conflict_summary(sub_rows: List[Dict]) -> Dict[str, float]:
            total_steps = float(sum(max(0, int(r.get("steps_hl", 0) or 0)) for r in sub_rows))
            high_steps = float(sum(max(0, int(r.get("priv_high_conflict_steps", 0) or 0)) for r in sub_rows))
            non_conflict_steps = max(0.0, total_steps - high_steps)
            delta_conflict = _weighted_step_mean(
                sub_rows,
                "priv_conflict_delta_y_mean",
                "priv_high_conflict_steps",
            )
            weighted_sum = 0.0
            weight_sum = 0.0
            for row in sub_rows:
                value = _safe_float(row.get("priv_non_conflict_delta_y_mean", float("nan")), default=float("nan"))
                steps = max(
                    0.0,
                    _safe_float(row.get("steps_hl", 0.0), default=0.0)
                    - _safe_float(row.get("priv_high_conflict_steps", 0.0), default=0.0),
                )
                if math.isfinite(value) and steps > 0.0:
                    weighted_sum += value * steps
                    weight_sum += steps
            delta_non_conflict = weighted_sum / weight_sum if weight_sum > 0.0 else float("nan")
            visited = [r for r in sub_rows if int(r.get("priv_high_conflict_steps", 0) or 0) > 0]
            return {
                "priv_high_conflict_steps": high_steps,
                "priv_high_conflict_step_rate": high_steps / total_steps if total_steps > 0.0 else float("nan"),
                "priv_non_conflict_steps": non_conflict_steps,
                "priv_conflict_score_mean": _weighted_step_mean(sub_rows, "priv_conflict_score_mean", "steps_hl"),
                "priv_obstacle_window_rate": _weighted_step_mean(sub_rows, "priv_obstacle_window_rate", "steps_hl"),
                "priv_follow_pressure_rate": _weighted_step_mean(sub_rows, "priv_follow_pressure_rate", "steps_hl"),
                "priv_avoid_pressure_rate": _weighted_step_mean(sub_rows, "priv_avoid_pressure_rate", "steps_hl"),
                "priv_conflict_y_raw_mean": _weighted_step_mean(sub_rows, "priv_conflict_y_raw_mean", "priv_high_conflict_steps"),
                "priv_conflict_y_eff_mean": _weighted_step_mean(sub_rows, "priv_conflict_y_eff_mean", "priv_high_conflict_steps"),
                "priv_conflict_w_mean": _weighted_step_mean(sub_rows, "priv_conflict_w_mean", "priv_high_conflict_steps"),
                "priv_conflict_signed_w_mean": _weighted_step_mean(sub_rows, "priv_conflict_signed_w_mean", "priv_high_conflict_steps"),
                "priv_conflict_delta_y_mean": delta_conflict,
                "priv_non_conflict_delta_y_mean": delta_non_conflict,
                "conflict_suppression_index": -delta_conflict if math.isfinite(delta_conflict) else float("nan"),
                "conflict_selective_suppression": (
                    delta_non_conflict - delta_conflict
                    if math.isfinite(delta_conflict) and math.isfinite(delta_non_conflict)
                    else float("nan")
                ),
                "relative_conflict_modulation": (
                    delta_non_conflict - delta_conflict
                    if math.isfinite(delta_conflict) and math.isfinite(delta_non_conflict)
                    else float("nan")
                ),
                "priv_window_phase_approach_rate": _weighted_step_mean(sub_rows, "priv_window_phase_approach_rate", "steps_hl"),
                "priv_window_phase_inside_rate": _weighted_step_mean(sub_rows, "priv_window_phase_inside_rate", "steps_hl"),
                "priv_window_phase_release_rate": _weighted_step_mean(sub_rows, "priv_window_phase_release_rate", "steps_hl"),
                "priv_conflict_phase_approach_rate": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_approach_rate", "priv_high_conflict_steps"
                ),
                "priv_conflict_phase_inside_rate": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_inside_rate", "priv_high_conflict_steps"
                ),
                "priv_conflict_phase_release_rate": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_release_rate", "priv_high_conflict_steps"
                ),
                "priv_conflict_phase_approach_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_approach_w_mean", "priv_conflict_phase_approach_steps"
                ),
                "priv_conflict_phase_inside_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_inside_w_mean", "priv_conflict_phase_inside_steps"
                ),
                "priv_conflict_phase_release_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_release_w_mean", "priv_conflict_phase_release_steps"
                ),
                "priv_conflict_phase_approach_signed_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_approach_signed_w_mean", "priv_conflict_phase_approach_steps"
                ),
                "priv_conflict_phase_inside_signed_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_inside_signed_w_mean", "priv_conflict_phase_inside_steps"
                ),
                "priv_conflict_phase_release_signed_w_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_release_signed_w_mean", "priv_conflict_phase_release_steps"
                ),
                "priv_conflict_phase_approach_delta_y_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_approach_delta_y_mean", "priv_conflict_phase_approach_steps"
                ),
                "priv_conflict_phase_inside_delta_y_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_inside_delta_y_mean", "priv_conflict_phase_inside_steps"
                ),
                "priv_conflict_phase_release_delta_y_mean": _weighted_step_mean(
                    sub_rows, "priv_conflict_phase_release_delta_y_mean", "priv_conflict_phase_release_steps"
                ),
                "priv_conflict_visited_episode_rate": len(visited) / float(max(1, len(sub_rows))),
                "priv_conflict_visited_collision_rate": (
                    sum(int(r.get("episode_collision", 0)) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
                "priv_conflict_visited_row_progress_mean": (
                    sum(float(r.get("success", 0.0) or 0.0) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
            }

        def _unsafe_conflict_summary(sub_rows: List[Dict]) -> Dict[str, float]:
            total_steps = float(sum(max(0, int(r.get("steps_hl", 0) or 0)) for r in sub_rows))
            unsafe_steps = float(sum(max(0, int(r.get("unsafe_conflict_steps", 0) or 0)) for r in sub_rows))
            delta_unsafe = _weighted_step_mean(
                sub_rows,
                "unsafe_conflict_delta_y_mean",
                "unsafe_conflict_steps",
            )
            weighted_sum = 0.0
            weight_sum = 0.0
            for row in sub_rows:
                value = _safe_float(row.get("unsafe_non_conflict_delta_y_mean", float("nan")), default=float("nan"))
                steps = max(
                    0.0,
                    _safe_float(row.get("steps_hl", 0.0), default=0.0)
                    - _safe_float(row.get("unsafe_conflict_steps", 0.0), default=0.0),
                )
                if math.isfinite(value) and steps > 0.0:
                    weighted_sum += value * steps
                    weight_sum += steps
            delta_non_unsafe = weighted_sum / weight_sum if weight_sum > 0.0 else float("nan")
            visited = [r for r in sub_rows if int(r.get("unsafe_conflict_steps", 0) or 0) > 0]
            return {
                "unsafe_conflict_steps": unsafe_steps,
                "unsafe_conflict_step_rate": unsafe_steps / total_steps if total_steps > 0.0 else float("nan"),
                "unsafe_follow_risk_rate": _weighted_step_mean(sub_rows, "unsafe_follow_risk_rate", "steps_hl"),
                "unsafe_safe_candidate_better_rate": _weighted_step_mean(sub_rows, "unsafe_safe_candidate_better_rate", "steps_hl"),
                "unsafe_avoid_safer_rate": _weighted_step_mean(sub_rows, "unsafe_avoid_safer_rate", "steps_hl"),
                "unsafe_stop_safer_rate": _weighted_step_mean(sub_rows, "unsafe_stop_safer_rate", "steps_hl"),
                "unsafe_command_disagree_rate": _weighted_step_mean(sub_rows, "unsafe_command_disagree_rate", "steps_hl"),
                "unsafe_target_recoverable_rate": _weighted_step_mean(sub_rows, "unsafe_target_recoverable_rate", "steps_hl"),
                "unsafe_conflict_y_raw_mean": _weighted_step_mean(sub_rows, "unsafe_conflict_y_raw_mean", "unsafe_conflict_steps"),
                "unsafe_conflict_y_eff_mean": _weighted_step_mean(sub_rows, "unsafe_conflict_y_eff_mean", "unsafe_conflict_steps"),
                "unsafe_conflict_w_mean": _weighted_step_mean(sub_rows, "unsafe_conflict_w_mean", "unsafe_conflict_steps"),
                "unsafe_conflict_signed_w_mean": _weighted_step_mean(sub_rows, "unsafe_conflict_signed_w_mean", "unsafe_conflict_steps"),
                "unsafe_conflict_delta_y_mean": delta_unsafe,
                "unsafe_conflict_delta_y_w_raw_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_delta_y_w_raw_mean", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_delta_y_w_used_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_delta_y_w_used_mean", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_delta_y_r_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_delta_y_r_mean", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_delta_y_total_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_delta_y_total_mean", "unsafe_conflict_steps"
                ),
                "unsafe_non_conflict_delta_y_mean": delta_non_unsafe,
                "unsafe_conflict_suppression_index": -delta_unsafe if math.isfinite(delta_unsafe) else float("nan"),
                "unsafe_conflict_selective_suppression": (
                    delta_non_unsafe - delta_unsafe
                    if math.isfinite(delta_unsafe) and math.isfinite(delta_non_unsafe)
                    else float("nan")
                ),
                "unsafe_relative_conflict_modulation": (
                    delta_non_unsafe - delta_unsafe
                    if math.isfinite(delta_unsafe) and math.isfinite(delta_non_unsafe)
                    else float("nan")
                ),
                "unsafe_conflict_phase_approach_rate": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_approach_rate", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_phase_inside_rate": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_inside_rate", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_phase_release_rate": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_release_rate", "unsafe_conflict_steps"
                ),
                "unsafe_conflict_phase_approach_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_approach_w_mean", "unsafe_conflict_phase_approach_steps"
                ),
                "unsafe_conflict_phase_inside_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_inside_w_mean", "unsafe_conflict_phase_inside_steps"
                ),
                "unsafe_conflict_phase_release_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_release_w_mean", "unsafe_conflict_phase_release_steps"
                ),
                "unsafe_conflict_phase_approach_signed_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_approach_signed_w_mean", "unsafe_conflict_phase_approach_steps"
                ),
                "unsafe_conflict_phase_inside_signed_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_inside_signed_w_mean", "unsafe_conflict_phase_inside_steps"
                ),
                "unsafe_conflict_phase_release_signed_w_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_release_signed_w_mean", "unsafe_conflict_phase_release_steps"
                ),
                "unsafe_conflict_phase_approach_delta_y_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_approach_delta_y_mean", "unsafe_conflict_phase_approach_steps"
                ),
                "unsafe_conflict_phase_inside_delta_y_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_inside_delta_y_mean", "unsafe_conflict_phase_inside_steps"
                ),
                "unsafe_conflict_phase_release_delta_y_mean": _weighted_step_mean(
                    sub_rows, "unsafe_conflict_phase_release_delta_y_mean", "unsafe_conflict_phase_release_steps"
                ),
                "unsafe_conflict_visited_episode_rate": len(visited) / float(max(1, len(sub_rows))),
                "unsafe_conflict_visited_collision_rate": (
                    sum(int(r.get("episode_collision", 0)) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
                "unsafe_conflict_visited_row_progress_mean": (
                    sum(float(r.get("success", 0.0) or 0.0) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
            }

        def _candidate_conflict_summary(sub_rows: List[Dict], prefix: str) -> Dict[str, float]:
            steps_key = f"{prefix}_conflict_steps"
            delta_key = f"{prefix}_conflict_delta_y_mean"
            steps = float(sum(max(0, int(r.get(steps_key, 0) or 0)) for r in sub_rows))
            total_steps = float(sum(max(0, int(r.get("steps_hl", 0) or 0)) for r in sub_rows))
            delta = _weighted_step_mean(sub_rows, delta_key, steps_key)
            visited = [r for r in sub_rows if int(r.get(steps_key, 0) or 0) > 0]
            return {
                steps_key: steps,
                f"{prefix}_conflict_step_rate": steps / total_steps if total_steps > 0.0 else float("nan"),
                f"{prefix}_conflict_y_raw_mean": _weighted_step_mean(
                    sub_rows, f"{prefix}_conflict_y_raw_mean", steps_key
                ),
                f"{prefix}_conflict_y_eff_mean": _weighted_step_mean(
                    sub_rows, f"{prefix}_conflict_y_eff_mean", steps_key
                ),
                f"{prefix}_conflict_w_mean": _weighted_step_mean(
                    sub_rows, f"{prefix}_conflict_w_mean", steps_key
                ),
                f"{prefix}_conflict_signed_w_mean": _weighted_step_mean(
                    sub_rows, f"{prefix}_conflict_signed_w_mean", steps_key
                ),
                delta_key: delta,
                f"{prefix}_conflict_suppression_index": -delta if math.isfinite(delta) else float("nan"),
                f"{prefix}_conflict_visited_episode_rate": len(visited) / float(max(1, len(sub_rows))),
                f"{prefix}_conflict_visited_collision_rate": (
                    sum(int(r.get("episode_collision", 0)) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
                f"{prefix}_conflict_visited_row_progress_mean": (
                    sum(float(r.get("success", 0.0) or 0.0) for r in visited) / float(len(visited))
                    if visited else float("nan")
                ),
            }

        def _target_fov_summary(sub_rows: List[Dict]) -> Dict[str, float]:
            lost_episodes = sum(int(r.get("target_lost_episode", 0) or 0) for r in sub_rows)
            max_lost = _clean([r.get("target_lost_max_consecutive_steps", float("nan")) for r in sub_rows])
            bearing_max_vals = _clean([r.get("target_bearing_abs_max", float("nan")) for r in sub_rows])
            bearing_conflict_max_vals = _clean([
                r.get("target_bearing_abs_max_in_priv_conflict", float("nan")) for r in sub_rows
            ])
            bearing_mean = _weighted_step_mean(sub_rows, "target_bearing_abs_mean", "steps_hl")
            bearing_p95 = _weighted_step_mean(sub_rows, "target_bearing_abs_p95", "steps_hl")
            bearing_conflict_mean = _weighted_step_mean(
                sub_rows, "target_bearing_abs_mean_in_priv_conflict", "priv_high_conflict_steps"
            )
            return {
                "target_bearing_abs_mean": bearing_mean,
                "target_bearing_abs_p95": bearing_p95,
                "target_bearing_abs_max": max(bearing_max_vals) if bearing_max_vals else float("nan"),
                "target_bearing_abs_deg_mean": bearing_mean * 180.0 / math.pi,
                "target_bearing_abs_deg_p95": bearing_p95 * 180.0 / math.pi,
                "target_bearing_abs_deg_max": (
                    max(bearing_max_vals) * 180.0 / math.pi if bearing_max_vals else float("nan")
                ),
                "target_in_rgb_fov_rate": _weighted_step_mean(sub_rows, "target_in_rgb_fov_rate", "steps_hl"),
                "target_near_rgb_fov_edge_rate": _weighted_step_mean(sub_rows, "target_near_rgb_fov_edge_rate", "steps_hl"),
                "target_lost_step_rate": _weighted_step_mean(sub_rows, "target_lost_step_rate", "steps_hl"),
                "target_lost_episode_rate": lost_episodes / float(max(1, len(sub_rows))),
                "target_lost_max_consecutive_steps_mean": float(np.mean(max_lost)) if max_lost else float("nan"),
                "target_lost_max_consecutive_steps_max": max(max_lost) if max_lost else float("nan"),
                "target_bearing_abs_mean_in_priv_conflict": bearing_conflict_mean,
                "target_bearing_abs_deg_mean_in_priv_conflict": bearing_conflict_mean * 180.0 / math.pi,
                "target_bearing_abs_max_in_priv_conflict": (
                    max(bearing_conflict_max_vals) if bearing_conflict_max_vals else float("nan")
                ),
                "target_in_fov_rate_in_priv_conflict": _weighted_step_mean(
                    sub_rows, "target_in_fov_rate_in_priv_conflict", "priv_high_conflict_steps"
                ),
                "target_near_fov_edge_rate_in_priv_conflict": _weighted_step_mean(
                    sub_rows, "target_near_fov_edge_rate_in_priv_conflict", "priv_high_conflict_steps"
                ),
                "target_lost_rate_in_priv_conflict": _weighted_step_mean(
                    sub_rows, "target_lost_rate_in_priv_conflict", "priv_high_conflict_steps"
                ),
            }

        task_success_flags = [int(r.get("task_success", r.get("success", 0))) for r in rows]
        l1_safety_flags = [int(r.get("l1_safety_success", 0)) for r in rows]
        l2_follow_flags = [int(r.get("l2_follow_success", 0)) for r in rows]
        l3_progress_flags = [int(r.get("l3_progress_success", 0)) for r in rows]
        row_progress_scores = [float(r.get("row_progress_success", 0.0)) for r in rows]
        full_task_success_flags = [int(r.get("full_task_success", 0)) for r in rows]
        success_event_flags = [int(r.get("success_event", r["success"])) for r in rows]
        success_event_and_collision_flags = [int(r.get("success_event_and_collision", r.get("success_and_collision", 0))) for r in rows]
        collision_only_flags = [int(r.get("collision_only", 0)) for r in rows]
        timeout_or_other_flags = [int(r.get("timeout_or_other", 0)) for r in rows]
        follow_mae = _clean([r["follow_mae_m"] for r in rows])
        follow_rmse = _clean([r["follow_rmse_m"] for r in rows])
        cot_vals = _clean([r["cot"] for r in rows])
        tts = _clean([r["time_to_success_s"] for r in rows if int(r.get("success_event", 0)) == 1])
        success_event_tts = _clean([
            r.get("success_event_time_s", float("nan"))
            for r in rows
            if int(r.get("success_event", 0)) == 1
        ])
        cross_line_end = _clean([r.get("cross_line_dist_end", float("nan")) for r in rows])
        cross_line_min = _clean([r.get("cross_line_dist_min", float("nan")) for r in rows])
        episode_collision = [int(r.get("episode_collision", 0)) for r in rows]
        progress_flags = [int(r.get("progress_reached", 0)) for r in rows]
        progress_ratio = _clean([r.get("progress_ratio_best", float("nan")) for r in rows])
        gate_y_raw_vals = _clean([r.get("gate_y_raw_mean", float("nan")) for r in rows])
        y_eff_vals = _clean([r.get("y_eff_mean", float("nan")) for r in rows])
        w_vals = _clean([r.get("w_mean", float("nan")) for r in rows])
        signed_w_vals = _clean([r.get("signed_w_mean", float("nan")) for r in rows])
        signed_w_active_vals = _clean([r.get("signed_w_active_mean", float("nan")) for r in rows])
        clearance_f_vals = _clean([r.get("clearance_f_mean", float("nan")) for r in rows])
        clearance_a_vals = _clean([r.get("clearance_a_mean", float("nan")) for r in rows])
        risk_f_vals = _clean([r.get("risk_f_mean", float("nan")) for r in rows])
        risk_a_vals = _clean([r.get("risk_a_mean", float("nan")) for r in rows])
        risk_delta_vals = _clean([r.get("risk_delta_mean", float("nan")) for r in rows])
        risk_rollout_f_vals = _clean([r.get("risk_rollout_f_mean", float("nan")) for r in rows])
        risk_rollout_a_vals = _clean([r.get("risk_rollout_a_mean", float("nan")) for r in rows])
        risk_rollout_s_vals = _clean([r.get("risk_rollout_s_mean", float("nan")) for r in rows])
        risk_rollout_gap_vals = _clean([r.get("risk_rollout_gap_f_min_as_mean", float("nan")) for r in rows])
        w_support_correction_vals = _clean([r.get("w_support_correction_mean", float("nan")) for r in rows])
        risk_diff_correction_vals = _clean([r.get("risk_diff_correction_mean", float("nan")) for r in rows])
        delta_y_w_raw_vals = _clean([r.get("delta_y_w_raw_mean", float("nan")) for r in rows])
        delta_y_w_used_vals = _clean([r.get("delta_y_w_used_mean", float("nan")) for r in rows])
        delta_y_r_vals = _clean([r.get("delta_y_r_mean", float("nan")) for r in rows])
        delta_y_total_vals = _clean([r.get("delta_y_total_mean", float("nan")) for r in rows])
        rule_s_vals = _clean([r.get("rule_s_mean", float("nan")) for r in rows])
        rule_risk_gap_vals = _clean([r.get("rule_risk_gap_mean", float("nan")) for r in rows])
        rule_follow_scale_vals = _clean([r.get("rule_follow_scale_mean", float("nan")) for r in rows])
        rule_yaw_scale_vals = _clean([r.get("rule_yaw_scale_mean", float("nan")) for r in rows])
        rule_follow_suppression_vals = _clean([r.get("rule_follow_suppression_mean", float("nan")) for r in rows])
        rule_s_at_avoid_vals = _clean([r.get("rule_s_at_avoid_conflict", float("nan")) for r in rows])
        rule_follow_scale_at_avoid_vals = _clean([r.get("rule_follow_scale_at_avoid_conflict", float("nan")) for r in rows])
        rule_yaw_scale_at_avoid_vals = _clean([r.get("rule_yaw_scale_at_avoid_conflict", float("nan")) for r in rows])
        rule_follow_suppression_at_avoid_vals = _clean([r.get("rule_follow_suppression_at_avoid_conflict", float("nan")) for r in rows])
        unsafe_delta_y_w_raw_vals = _clean([r.get("unsafe_conflict_delta_y_w_raw_mean", float("nan")) for r in rows])
        unsafe_delta_y_w_used_vals = _clean([r.get("unsafe_conflict_delta_y_w_used_mean", float("nan")) for r in rows])
        unsafe_delta_y_r_vals = _clean([r.get("unsafe_conflict_delta_y_r_mean", float("nan")) for r in rows])
        unsafe_delta_y_total_vals = _clean([r.get("unsafe_conflict_delta_y_total_mean", float("nan")) for r in rows])
        avoid_delta_y_w_raw_vals = _clean([r.get("avoid_conflict_delta_y_w_raw_mean", float("nan")) for r in rows])
        avoid_delta_y_w_used_vals = _clean([r.get("avoid_conflict_delta_y_w_used_mean", float("nan")) for r in rows])
        avoid_delta_y_r_vals = _clean([r.get("avoid_conflict_delta_y_r_mean", float("nan")) for r in rows])
        avoid_delta_y_total_vals = _clean([r.get("avoid_conflict_delta_y_total_mean", float("nan")) for r in rows])
        avoid_cmd_y_vals = _clean([r.get("avoid_conflict_cmd_y_mean", float("nan")) for r in rows])
        avoid_cmd_f_y_vals = _clean([r.get("avoid_conflict_cmd_f_y_mean", float("nan")) for r in rows])
        avoid_cmd_a_y_vals = _clean([r.get("avoid_conflict_cmd_a_y_mean", float("nan")) for r in rows])
        row_not_released_vals = _clean([r.get("row_not_released_rate", float("nan")) for r in rows])
        row_not_released_w_vals = _clean([r.get("row_not_released_w_mean", float("nan")) for r in rows])
        row_released_w_vals = _clean([r.get("row_released_w_mean", float("nan")) for r in rows])
        switch_vals = _clean([r.get("switch_rate", float("nan")) for r in rows])
        near_miss_vals = _clean([r.get("near_miss_rate", float("nan")) for r in rows])
        rotate_only_vals = _clean([r.get("rotate_only_rate", float("nan")) for r in rows])
        w_trigger_steps = _clean([
            r.get("w_trigger_step", float("nan"))
            for r in rows
            if int(r.get("w_trigger_step", -1)) >= 0
        ])
        w_trigger_progress = _clean([r.get("w_trigger_progress", float("nan")) for r in rows])
        gate_region_y_eff = _clean([r.get("gate_region_y_eff_mean", float("nan")) for r in rows])
        gate_region_near_miss = _clean([r.get("gate_region_near_miss_rate", float("nan")) for r in rows])
        cmd_jerk_lin_vals = _clean([r.get("cmd_jerk_lin_mean", float("nan")) for r in rows])
        cmd_jerk_ang_vals = _clean([r.get("cmd_jerk_ang_mean", float("nan")) for r in rows])
        cmd_x_mean_vals = _clean([r.get("cmd_x_mean", float("nan")) for r in rows])
        cmd_x_std_vals = _clean([r.get("cmd_x_std", float("nan")) for r in rows])
        cmd_x_abs_vals = _clean([r.get("cmd_x_abs_mean", float("nan")) for r in rows])
        cmd_x_pos_vals = _clean([r.get("cmd_x_positive_rate", float("nan")) for r in rows])
        cmd_x_neg_vals = _clean([r.get("cmd_x_negative_rate", float("nan")) for r in rows])
        cmd_x_zero_vals = _clean([r.get("cmd_x_zero_rate", float("nan")) for r in rows])
        cmd_x_entropy_vals = _clean([r.get("cmd_x_direction_entropy", float("nan")) for r in rows])
        corr_goalx_cmdx_vals = _clean([r.get("corr_goalx_cmdx", float("nan")) for r in rows])
        corr_goalx_cmdyaw_vals = _clean([r.get("corr_goalx_cmdyaw", float("nan")) for r in rows])
        corr_goaly_cmdy_vals = _clean([r.get("corr_goaly_cmdy", float("nan")) for r in rows])
        cmd_yaw_left_vals = _clean([r.get("cmd_yaw_left_mean", float("nan")) for r in rows])
        cmd_yaw_right_vals = _clean([r.get("cmd_yaw_right_mean", float("nan")) for r in rows])
        cmd_x_left_vals = _clean([r.get("cmd_x_left_mean", float("nan")) for r in rows])
        cmd_x_right_vals = _clean([r.get("cmd_x_right_mean", float("nan")) for r in rows])
        yaw_response_vals = _clean([r.get("yaw_directional_response", float("nan")) for r in rows])
        lateral_response_vals = _clean([r.get("lateral_directional_response", float("nan")) for r in rows])
        cmd_post_delta_abs_vals = _clean([r.get("cmd_post_delta_abs_mean", float("nan")) for r in rows])
        cmd_post_delta_x_abs_vals = _clean([r.get("cmd_post_delta_x_abs_mean", float("nan")) for r in rows])
        cmd_post_delta_y_abs_vals = _clean([r.get("cmd_post_delta_y_abs_mean", float("nan")) for r in rows])
        cmd_post_delta_y_signed_vals = _clean([r.get("cmd_post_delta_y_signed_mean", float("nan")) for r in rows])
        cmd_post_delta_w_abs_vals = _clean([r.get("cmd_post_delta_w_abs_mean", float("nan")) for r in rows])
        cmd_post_rewrite_vals = _clean([r.get("cmd_post_rewrite_rate", float("nan")) for r in rows])
        velocity_search_filter_change_vals = _clean([
            r.get("velocity_search_filter_change_rate", float("nan")) for r in rows
        ])
        velocity_search_safe_available_vals = _clean([
            r.get("velocity_search_safe_available_rate", float("nan")) for r in rows
        ])
        velocity_search_raw_risk_vals = _clean([
            r.get("velocity_search_raw_risk_mean", float("nan")) for r in rows
        ])
        velocity_search_safe_risk_vals = _clean([
            r.get("velocity_search_safe_risk_mean", float("nan")) for r in rows
        ])
        velocity_search_compute_time_vals = _clean([
            r.get("velocity_search_compute_time_ms_mean", float("nan")) for r in rows
        ])
        velocity_search_dynamic_window_enabled_vals = _clean([
            r.get("velocity_search_dynamic_window_enabled_rate", float("nan")) for r in rows
        ])
        velocity_search_candidate_count_before_dynamic_window_vals = _clean([
            r.get("velocity_search_candidate_count_before_dynamic_window_mean", float("nan")) for r in rows
        ])
        velocity_search_candidate_count_after_dynamic_window_vals = _clean([
            r.get("velocity_search_candidate_count_after_dynamic_window_mean", float("nan")) for r in rows
        ])
        velocity_search_dynamic_window_rejected_count_vals = _clean([
            r.get("velocity_search_dynamic_window_rejected_count_mean", float("nan")) for r in rows
        ])
        velocity_search_candidate_count_vals = _clean([
            r.get("velocity_search_candidate_count_mean", float("nan")) for r in rows
        ])
        velocity_search_feasible_count_vals = _clean([
            r.get("velocity_search_feasible_count_mean", float("nan")) for r in rows
        ])
        velocity_search_infeasible_count_vals = _clean([
            r.get("velocity_search_infeasible_count_mean", float("nan")) for r in rows
        ])
        velocity_search_feasible_count_after_margin_vals = _clean([
            r.get("velocity_search_feasible_count_after_margin_mean", float("nan")) for r in rows
        ])
        velocity_search_selected_v_lat_vals = _clean([
            r.get("velocity_search_selected_v_lat_mean", float("nan")) for r in rows
        ])
        velocity_search_selected_v_fwd_vals = _clean([
            r.get("velocity_search_selected_v_fwd_mean", float("nan")) for r in rows
        ])
        velocity_search_selected_yaw_vals = _clean([
            r.get("velocity_search_selected_yaw_mean", float("nan")) for r in rows
        ])
        velocity_search_raw_y_vals = _clean([
            r.get("velocity_search_raw_y_mean", float("nan")) for r in rows
        ])
        velocity_search_after_filter_y_vals = _clean([
            r.get("velocity_search_after_filter_y_mean", float("nan")) for r in rows
        ])
        cmd_safe_y_vals = _clean([r.get("cmd_safe_y_mean", float("nan")) for r in rows])
        cmd_safe_y_over_raw_y_vals = _clean([
            r.get("cmd_safe_y_over_raw_y_ratio", float("nan")) for r in rows
        ])
        actual_base_vel_x_vals = _clean([r.get("actual_base_vel_x_mean", float("nan")) for r in rows])
        actual_base_vel_y_vals = _clean([r.get("actual_base_vel_y_mean", float("nan")) for r in rows])
        actual_base_vel_w_vals = _clean([r.get("actual_base_vel_w_mean", float("nan")) for r in rows])
        actual_base_delta_x_vals = _clean([r.get("actual_base_delta_x_mean", float("nan")) for r in rows])
        actual_base_delta_y_vals = _clean([r.get("actual_base_delta_y_mean", float("nan")) for r in rows])
        row_progress_delta_vals = _clean([r.get("row_progress_delta_mean", float("nan")) for r in rows])
        velocity_search_forward_candidate_count_vals = _clean([
            r.get("velocity_search_forward_candidate_count_mean", float("nan")) for r in rows
        ])
        velocity_search_forward_feasible_count_vals = _clean([
            r.get("velocity_search_forward_feasible_count_mean", float("nan")) for r in rows
        ])
        velocity_search_forward_infeasible_collision_count_vals = _clean([
            r.get("velocity_search_forward_infeasible_collision_count_mean", float("nan")) for r in rows
        ])
        velocity_search_forward_infeasible_margin_count_vals = _clean([
            r.get("velocity_search_forward_infeasible_margin_count_mean", float("nan")) for r in rows
        ])
        velocity_search_forward_infeasible_out_of_map_count_vals = _clean([
            r.get("velocity_search_forward_infeasible_out_of_map_count_mean", float("nan")) for r in rows
        ])
        velocity_search_best_forward_feasible_cost_vals = _clean([
            r.get("velocity_search_best_forward_feasible_cost_mean", float("nan")) for r in rows
        ])
        velocity_search_best_forward_feasible_clearance_vals = _clean([
            r.get("velocity_search_best_forward_feasible_clearance_mean", float("nan")) for r in rows
        ])
        velocity_search_fallback_vals = _clean([
            r.get("velocity_search_fallback_rate", float("nan")) for r in rows
        ])
        fallback_no_feasible_vals = _clean([
            r.get("fallback_no_feasible_rate", float("nan")) for r in rows
        ])
        fallback_risk_threshold_vals = _clean([
            r.get("fallback_risk_threshold_rate", float("nan")) for r in rows
        ])
        fallback_collision_vals = _clean([
            r.get("fallback_collision_rate", float("nan")) for r in rows
        ])
        fallback_margin_vals = _clean([
            r.get("fallback_margin_rate", float("nan")) for r in rows
        ])
        fallback_out_of_map_vals = _clean([
            r.get("fallback_out_of_map_rate", float("nan")) for r in rows
        ])
        fallback_invalid_cost_vals = _clean([
            r.get("fallback_invalid_cost_rate", float("nan")) for r in rows
        ])
        velocity_search_min_clearance_vals = _clean([
            r.get("velocity_search_min_predicted_clearance_mean", float("nan")) for r in rows
        ])
        velocity_search_best_cost_vals = _clean([
            r.get("velocity_search_best_cost_mean", float("nan")) for r in rows
        ])
        selected_rollout_clearance_vals = _clean([
            r.get("selected_rollout_min_clearance_mean", float("nan")) for r in rows
        ])
        selected_rollout_collision_vals = _clean([
            r.get("selected_rollout_collision_rate", float("nan")) for r in rows
        ])
        selected_rollout_out_of_map_vals = _clean([
            r.get("selected_rollout_out_of_map_rate", float("nan")) for r in rows
        ])
        w_clearance_f_corr = _pearson_corr(
            [float(r.get("w_mean", float("nan"))) for r in rows],
            [float(r.get("clearance_f_mean", float("nan"))) for r in rows],
        )
        w_risk_f_corr = _pearson_corr(
            [float(r.get("w_mean", float("nan"))) for r in rows],
            [float(r.get("risk_f_mean", float("nan"))) for r in rows],
        )
        w_risk_delta_corr = _pearson_corr(
            [float(r.get("w_mean", float("nan"))) for r in rows],
            [float(r.get("risk_delta_mean", float("nan"))) for r in rows],
        )

        total_eps = len(rows)
        task_success_eps = int(sum(task_success_flags))
        row_progress_success_sum = float(sum(row_progress_scores))
        full_task_success_eps = int(sum(full_task_success_flags))
        strict_success_eps = int(sum(int(r.get("strict_success", 0)) for r in rows))
        success_event_eps = int(sum(success_event_flags))
        success_event_and_collision_eps = int(sum(success_event_and_collision_flags))
        collision_only_eps = int(sum(collision_only_flags))
        timeout_or_other_eps = int(sum(timeout_or_other_flags))
        collision_eps = int(sum(episode_collision))
        task_success_rate = float(task_success_eps / max(1, total_eps))
        row_progress_success_mean = float(row_progress_success_sum / max(1, total_eps))
        full_task_success_rate = float(full_task_success_eps / max(1, total_eps))
        success_event_rate = float(success_event_eps / max(1, total_eps))
        collision_rate = float(collision_eps / max(1, total_eps))
        success_event_and_collision_rate = float(success_event_and_collision_eps / max(1, total_eps))
        timeout_or_other_rate = float(timeout_or_other_eps / max(1, total_eps))
        collision_only_rate = float(collision_only_eps / max(1, total_eps))
        outcome_total_rate = float("nan")
        high_risk_overall = _high_risk_summary(rows)
        risk_bins_overall = _risk_bins_summary(rows)
        conflict_bins_overall = _risk_bins_summary(rows, "conflict_bin_stats")
        priv_conflict_bins_overall = _risk_bins_summary(rows, "priv_conflict_bin_stats")
        priv_conflict_overall = _priv_conflict_summary(rows)
        unsafe_conflict_overall = _unsafe_conflict_summary(rows)
        avoid_conflict_overall = _candidate_conflict_summary(rows, "avoid")
        stop_conflict_overall = _candidate_conflict_summary(rows, "stop")
        target_fov_overall = _target_fov_summary(rows)

        overall = {
            "episodes": total_eps,
            "success_episodes": task_success_eps,
            "task_success_episodes": task_success_eps,
            "l1_safety_success_rate": float(sum(l1_safety_flags) / max(1, total_eps)),
            "l2_follow_success_rate": float(sum(l2_follow_flags) / max(1, total_eps)),
            "l3_progress_success_rate": float(sum(l3_progress_flags) / max(1, total_eps)),
            "row_progress_success_sum": row_progress_success_sum,
            "full_task_success_episodes": full_task_success_eps,
            "strict_success_episodes": strict_success_eps,
            "success_event_episodes": success_event_eps,
            "success_event_and_collision_episodes": success_event_and_collision_eps,
            "success_and_collision_episodes": success_event_and_collision_eps,
            "timeout_or_other_episodes": timeout_or_other_eps,
            "collision_only_episodes": collision_only_eps,
            "success_rate": task_success_rate,
            "task_success_rate": task_success_rate,
            "row_progress_success_mean": row_progress_success_mean,
            "full_task_success_rate": full_task_success_rate,
            "strict_success_rate": float(strict_success_eps / max(1, total_eps)),
            "success_event_rate": success_event_rate,
            "success_event_and_collision_rate": success_event_and_collision_rate,
            "success_and_collision_rate": success_event_and_collision_rate,
            "timeout_or_other_rate": timeout_or_other_rate,
            "collision_only_rate": collision_only_rate,
            "outcome_total_rate": outcome_total_rate,
            "fail_ratio": float(1.0 - task_success_rate),
            "follow_mae_m_mean": float(np.mean(follow_mae)) if follow_mae else float("nan"),
            "follow_rmse_m_mean": float(np.mean(follow_rmse)) if follow_rmse else float("nan"),
            "time_to_success_s_mean": float(np.mean(tts)) if tts else float("nan"),
            "time_to_success_s_median": float(np.median(tts)) if tts else float("nan"),
            "time_to_success_s_p95": _quantile(tts, 0.95),
            "success_event_time_s_mean": float(np.mean(success_event_tts)) if success_event_tts else float("nan"),
            "success_event_time_s_median": float(np.median(success_event_tts)) if success_event_tts else float("nan"),
            "success_event_time_s_p95": _quantile(success_event_tts, 0.95),
            "cot_mean": float(np.mean(cot_vals)) if cot_vals else float("nan"),
            "cot_median": float(np.median(cot_vals)) if cot_vals else float("nan"),
            "cot_p95": _quantile(cot_vals, 0.95),
            "cross_line_dist_end_mean": float(np.mean(cross_line_end)) if cross_line_end else float("nan"),
            "cross_line_dist_min_mean": float(np.mean(cross_line_min)) if cross_line_min else float("nan"),
            "episode_collision_rate": collision_rate,
            "progress_rate": float(sum(progress_flags) / max(1, total_eps)),
            "progress_ratio_mean": float(np.mean(progress_ratio)) if progress_ratio else 0.0,
            "progress_any_rate": float(sum(progress_flags) / max(1, total_eps)),
            "gate_y_raw_mean": float(np.mean(gate_y_raw_vals)) if gate_y_raw_vals else float("nan"),
            "y_eff_mean": float(np.mean(y_eff_vals)) if y_eff_vals else float("nan"),
            "w_mean": float(np.mean(w_vals)) if w_vals else float("nan"),
            "signed_w_mean": float(np.mean(signed_w_vals)) if signed_w_vals else float("nan"),
            "signed_w_active_mean": (
                float(np.mean(signed_w_active_vals)) if signed_w_active_vals else float("nan")
            ),
            "clearance_f_mean": float(np.mean(clearance_f_vals)) if clearance_f_vals else float("nan"),
            "clearance_a_mean": float(np.mean(clearance_a_vals)) if clearance_a_vals else float("nan"),
            "risk_f_mean": float(np.mean(risk_f_vals)) if risk_f_vals else float("nan"),
            "risk_a_mean": float(np.mean(risk_a_vals)) if risk_a_vals else float("nan"),
            "risk_delta_mean": float(np.mean(risk_delta_vals)) if risk_delta_vals else float("nan"),
            "risk_rollout_f_mean": float(np.mean(risk_rollout_f_vals)) if risk_rollout_f_vals else float("nan"),
            "risk_rollout_a_mean": float(np.mean(risk_rollout_a_vals)) if risk_rollout_a_vals else float("nan"),
            "risk_rollout_s_mean": float(np.mean(risk_rollout_s_vals)) if risk_rollout_s_vals else float("nan"),
            "risk_rollout_gap_f_min_as_mean": float(np.mean(risk_rollout_gap_vals)) if risk_rollout_gap_vals else float("nan"),
            "w_support_correction_mean": float(np.mean(w_support_correction_vals)) if w_support_correction_vals else float("nan"),
            "risk_diff_correction_mean": float(np.mean(risk_diff_correction_vals)) if risk_diff_correction_vals else float("nan"),
            "delta_y_w_raw_mean": float(np.mean(delta_y_w_raw_vals)) if delta_y_w_raw_vals else float("nan"),
            "delta_y_w_used_mean": float(np.mean(delta_y_w_used_vals)) if delta_y_w_used_vals else float("nan"),
            "delta_y_r_mean": float(np.mean(delta_y_r_vals)) if delta_y_r_vals else float("nan"),
            "delta_y_total_mean": float(np.mean(delta_y_total_vals)) if delta_y_total_vals else float("nan"),
            "rule_s_mean": float(np.mean(rule_s_vals)) if rule_s_vals else float("nan"),
            "rule_risk_gap_mean": float(np.mean(rule_risk_gap_vals)) if rule_risk_gap_vals else float("nan"),
            "rule_follow_scale_mean": float(np.mean(rule_follow_scale_vals)) if rule_follow_scale_vals else float("nan"),
            "rule_yaw_scale_mean": float(np.mean(rule_yaw_scale_vals)) if rule_yaw_scale_vals else float("nan"),
            "rule_follow_suppression_mean": (
                float(np.mean(rule_follow_suppression_vals)) if rule_follow_suppression_vals else float("nan")
            ),
            "rule_s_at_avoid_conflict": float(np.mean(rule_s_at_avoid_vals)) if rule_s_at_avoid_vals else float("nan"),
            "rule_follow_scale_at_avoid_conflict": (
                float(np.mean(rule_follow_scale_at_avoid_vals)) if rule_follow_scale_at_avoid_vals else float("nan")
            ),
            "rule_yaw_scale_at_avoid_conflict": (
                float(np.mean(rule_yaw_scale_at_avoid_vals)) if rule_yaw_scale_at_avoid_vals else float("nan")
            ),
            "rule_follow_suppression_at_avoid_conflict": (
                float(np.mean(rule_follow_suppression_at_avoid_vals)) if rule_follow_suppression_at_avoid_vals else float("nan")
            ),
            "unsafe_conflict_delta_y_w_raw_mean": (
                float(np.mean(unsafe_delta_y_w_raw_vals)) if unsafe_delta_y_w_raw_vals else float("nan")
            ),
            "unsafe_conflict_delta_y_w_used_mean": (
                float(np.mean(unsafe_delta_y_w_used_vals)) if unsafe_delta_y_w_used_vals else float("nan")
            ),
            "unsafe_conflict_delta_y_r_mean": (
                float(np.mean(unsafe_delta_y_r_vals)) if unsafe_delta_y_r_vals else float("nan")
            ),
            "unsafe_conflict_delta_y_total_mean": (
                float(np.mean(unsafe_delta_y_total_vals)) if unsafe_delta_y_total_vals else float("nan")
            ),
            "avoid_conflict_delta_y_w_raw_mean": (
                float(np.mean(avoid_delta_y_w_raw_vals)) if avoid_delta_y_w_raw_vals else float("nan")
            ),
            "avoid_conflict_delta_y_w_used_mean": (
                float(np.mean(avoid_delta_y_w_used_vals)) if avoid_delta_y_w_used_vals else float("nan")
            ),
            "avoid_conflict_delta_y_r_mean": (
                float(np.mean(avoid_delta_y_r_vals)) if avoid_delta_y_r_vals else float("nan")
            ),
            "avoid_conflict_delta_y_total_mean": (
                float(np.mean(avoid_delta_y_total_vals)) if avoid_delta_y_total_vals else float("nan")
            ),
            "avoid_conflict_cmd_y_mean": float(np.mean(avoid_cmd_y_vals)) if avoid_cmd_y_vals else float("nan"),
            "avoid_conflict_cmd_f_y_mean": float(np.mean(avoid_cmd_f_y_vals)) if avoid_cmd_f_y_vals else float("nan"),
            "avoid_conflict_cmd_a_y_mean": float(np.mean(avoid_cmd_a_y_vals)) if avoid_cmd_a_y_vals else float("nan"),
            "row_not_released_rate_mean": float(np.mean(row_not_released_vals)) if row_not_released_vals else float("nan"),
            "row_not_released_w_mean": float(np.mean(row_not_released_w_vals)) if row_not_released_w_vals else float("nan"),
            "row_released_w_mean": float(np.mean(row_released_w_vals)) if row_released_w_vals else float("nan"),
            "w_clearance_f_corr": w_clearance_f_corr,
            "w_risk_f_corr": w_risk_f_corr,
            "w_risk_delta_corr": w_risk_delta_corr,
            "w_degen_clearance_like": bool(
                math.isfinite(w_clearance_f_corr)
                and abs(w_clearance_f_corr) > 0.90
                and (
                    (not math.isfinite(w_risk_delta_corr))
                    or abs(w_risk_delta_corr) < abs(w_clearance_f_corr)
                )
            ),
            "w_trigger_rate": float(len(w_trigger_steps) / max(1, total_eps)),
            "w_trigger_step_mean": float(np.mean(w_trigger_steps)) if w_trigger_steps else float("nan"),
            "w_trigger_progress_mean": float(np.mean(w_trigger_progress)) if w_trigger_progress else float("nan"),
            "gate_region_y_eff_mean": float(np.mean(gate_region_y_eff)) if gate_region_y_eff else float("nan"),
            "gate_region_near_miss_rate_mean": float(np.mean(gate_region_near_miss)) if gate_region_near_miss else float("nan"),
            **high_risk_overall,
            **priv_conflict_overall,
            **unsafe_conflict_overall,
            **avoid_conflict_overall,
            **stop_conflict_overall,
            **target_fov_overall,
            "switch_rate_mean": float(np.mean(switch_vals)) if switch_vals else float("nan"),
            "near_miss_rate_mean": float(np.mean(near_miss_vals)) if near_miss_vals else float("nan"),
            "rotate_only_rate_mean": float(np.mean(rotate_only_vals)) if rotate_only_vals else float("nan"),
            "cmd_jerk_lin_mean": float(np.mean(cmd_jerk_lin_vals)) if cmd_jerk_lin_vals else float("nan"),
            "cmd_jerk_ang_mean": float(np.mean(cmd_jerk_ang_vals)) if cmd_jerk_ang_vals else float("nan"),
            "cmd_x_mean": float(np.mean(cmd_x_mean_vals)) if cmd_x_mean_vals else float("nan"),
            "cmd_x_std": float(np.mean(cmd_x_std_vals)) if cmd_x_std_vals else float("nan"),
            "cmd_x_abs_mean": float(np.mean(cmd_x_abs_vals)) if cmd_x_abs_vals else float("nan"),
            "cmd_x_positive_rate": float(np.mean(cmd_x_pos_vals)) if cmd_x_pos_vals else float("nan"),
            "cmd_x_negative_rate": float(np.mean(cmd_x_neg_vals)) if cmd_x_neg_vals else float("nan"),
            "cmd_x_zero_rate": float(np.mean(cmd_x_zero_vals)) if cmd_x_zero_vals else float("nan"),
            "cmd_x_direction_entropy": float(np.mean(cmd_x_entropy_vals)) if cmd_x_entropy_vals else float("nan"),
            "corr_goalx_cmdx": float(np.mean(corr_goalx_cmdx_vals)) if corr_goalx_cmdx_vals else float("nan"),
            "corr_goalx_cmdyaw": float(np.mean(corr_goalx_cmdyaw_vals)) if corr_goalx_cmdyaw_vals else float("nan"),
            "corr_goaly_cmdy": float(np.mean(corr_goaly_cmdy_vals)) if corr_goaly_cmdy_vals else float("nan"),
            "cmd_yaw_left_mean": float(np.mean(cmd_yaw_left_vals)) if cmd_yaw_left_vals else float("nan"),
            "cmd_yaw_right_mean": float(np.mean(cmd_yaw_right_vals)) if cmd_yaw_right_vals else float("nan"),
            "cmd_x_left_mean": float(np.mean(cmd_x_left_vals)) if cmd_x_left_vals else float("nan"),
            "cmd_x_right_mean": float(np.mean(cmd_x_right_vals)) if cmd_x_right_vals else float("nan"),
            "yaw_directional_response": float(np.mean(yaw_response_vals)) if yaw_response_vals else float("nan"),
            "lateral_directional_response": (
                float(np.mean(lateral_response_vals)) if lateral_response_vals else float("nan")
            ),
            "cmd_post_delta_abs_mean": (
                float(np.mean(cmd_post_delta_abs_vals)) if cmd_post_delta_abs_vals else float("nan")
            ),
            "cmd_post_delta_x_abs_mean": (
                float(np.mean(cmd_post_delta_x_abs_vals)) if cmd_post_delta_x_abs_vals else float("nan")
            ),
            "cmd_post_delta_y_abs_mean": (
                float(np.mean(cmd_post_delta_y_abs_vals)) if cmd_post_delta_y_abs_vals else float("nan")
            ),
            "cmd_post_delta_y_signed_mean": (
                float(np.mean(cmd_post_delta_y_signed_vals)) if cmd_post_delta_y_signed_vals else float("nan")
            ),
            "cmd_post_delta_w_abs_mean": (
                float(np.mean(cmd_post_delta_w_abs_vals)) if cmd_post_delta_w_abs_vals else float("nan")
            ),
            "cmd_post_rewrite_rate": (
                float(np.mean(cmd_post_rewrite_vals)) if cmd_post_rewrite_vals else float("nan")
            ),
            "velocity_search_filter_change_rate": (
                float(np.mean(velocity_search_filter_change_vals)) if velocity_search_filter_change_vals else float("nan")
            ),
            "velocity_search_safe_available_rate": (
                float(np.mean(velocity_search_safe_available_vals)) if velocity_search_safe_available_vals else float("nan")
            ),
            "velocity_search_raw_risk_mean": (
                float(np.mean(velocity_search_raw_risk_vals)) if velocity_search_raw_risk_vals else float("nan")
            ),
            "velocity_search_safe_risk_mean": (
                float(np.mean(velocity_search_safe_risk_vals)) if velocity_search_safe_risk_vals else float("nan")
            ),
            "velocity_search_compute_time_ms_mean": (
                float(np.mean(velocity_search_compute_time_vals)) if velocity_search_compute_time_vals else float("nan")
            ),
            "velocity_search_dynamic_window_enabled_rate": (
                float(np.mean(velocity_search_dynamic_window_enabled_vals)) if velocity_search_dynamic_window_enabled_vals else float("nan")
            ),
            "velocity_search_candidate_count_before_dynamic_window_mean": (
                float(np.mean(velocity_search_candidate_count_before_dynamic_window_vals)) if velocity_search_candidate_count_before_dynamic_window_vals else float("nan")
            ),
            "velocity_search_candidate_count_after_dynamic_window_mean": (
                float(np.mean(velocity_search_candidate_count_after_dynamic_window_vals)) if velocity_search_candidate_count_after_dynamic_window_vals else float("nan")
            ),
            "velocity_search_dynamic_window_rejected_count_mean": (
                float(np.mean(velocity_search_dynamic_window_rejected_count_vals)) if velocity_search_dynamic_window_rejected_count_vals else float("nan")
            ),
            "velocity_search_candidate_count_mean": (
                float(np.mean(velocity_search_candidate_count_vals)) if velocity_search_candidate_count_vals else float("nan")
            ),
            "velocity_search_feasible_count_mean": (
                float(np.mean(velocity_search_feasible_count_vals)) if velocity_search_feasible_count_vals else float("nan")
            ),
            "velocity_search_infeasible_count_mean": (
                float(np.mean(velocity_search_infeasible_count_vals)) if velocity_search_infeasible_count_vals else float("nan")
            ),
            "velocity_search_feasible_count_after_margin_mean": (
                float(np.mean(velocity_search_feasible_count_after_margin_vals)) if velocity_search_feasible_count_after_margin_vals else float("nan")
            ),
            "velocity_search_selected_v_lat_mean": (
                float(np.mean(velocity_search_selected_v_lat_vals)) if velocity_search_selected_v_lat_vals else float("nan")
            ),
            "velocity_search_selected_v_fwd_mean": (
                float(np.mean(velocity_search_selected_v_fwd_vals)) if velocity_search_selected_v_fwd_vals else float("nan")
            ),
            "velocity_search_selected_yaw_mean": (
                float(np.mean(velocity_search_selected_yaw_vals)) if velocity_search_selected_yaw_vals else float("nan")
            ),
            "velocity_search_raw_y_mean": (
                float(np.mean(velocity_search_raw_y_vals)) if velocity_search_raw_y_vals else float("nan")
            ),
            "velocity_search_after_filter_y_mean": (
                float(np.mean(velocity_search_after_filter_y_vals)) if velocity_search_after_filter_y_vals else float("nan")
            ),
            "cmd_safe_y_mean": (
                float(np.mean(cmd_safe_y_vals)) if cmd_safe_y_vals else float("nan")
            ),
            "cmd_safe_y_over_raw_y_ratio": (
                float(np.mean(cmd_safe_y_over_raw_y_vals)) if cmd_safe_y_over_raw_y_vals else float("nan")
            ),
            "actual_base_vel_x_mean": (
                float(np.mean(actual_base_vel_x_vals)) if actual_base_vel_x_vals else float("nan")
            ),
            "actual_base_vel_y_mean": (
                float(np.mean(actual_base_vel_y_vals)) if actual_base_vel_y_vals else float("nan")
            ),
            "actual_base_vel_w_mean": (
                float(np.mean(actual_base_vel_w_vals)) if actual_base_vel_w_vals else float("nan")
            ),
            "actual_base_delta_x_mean": (
                float(np.mean(actual_base_delta_x_vals)) if actual_base_delta_x_vals else float("nan")
            ),
            "actual_base_delta_y_mean": (
                float(np.mean(actual_base_delta_y_vals)) if actual_base_delta_y_vals else float("nan")
            ),
            "row_progress_delta_mean": (
                float(np.mean(row_progress_delta_vals)) if row_progress_delta_vals else float("nan")
            ),
            "velocity_search_forward_candidate_count_mean": (
                float(np.mean(velocity_search_forward_candidate_count_vals)) if velocity_search_forward_candidate_count_vals else float("nan")
            ),
            "velocity_search_forward_feasible_count_mean": (
                float(np.mean(velocity_search_forward_feasible_count_vals)) if velocity_search_forward_feasible_count_vals else float("nan")
            ),
            "velocity_search_forward_infeasible_collision_count_mean": (
                float(np.mean(velocity_search_forward_infeasible_collision_count_vals)) if velocity_search_forward_infeasible_collision_count_vals else float("nan")
            ),
            "velocity_search_forward_infeasible_margin_count_mean": (
                float(np.mean(velocity_search_forward_infeasible_margin_count_vals)) if velocity_search_forward_infeasible_margin_count_vals else float("nan")
            ),
            "velocity_search_forward_infeasible_out_of_map_count_mean": (
                float(np.mean(velocity_search_forward_infeasible_out_of_map_count_vals)) if velocity_search_forward_infeasible_out_of_map_count_vals else float("nan")
            ),
            "velocity_search_best_forward_feasible_cost_mean": (
                float(np.mean(velocity_search_best_forward_feasible_cost_vals)) if velocity_search_best_forward_feasible_cost_vals else float("nan")
            ),
            "velocity_search_best_forward_feasible_clearance_mean": (
                float(np.mean(velocity_search_best_forward_feasible_clearance_vals)) if velocity_search_best_forward_feasible_clearance_vals else float("nan")
            ),
            "velocity_search_fallback_rate": (
                float(np.mean(velocity_search_fallback_vals)) if velocity_search_fallback_vals else float("nan")
            ),
            "fallback_no_feasible_rate": (
                float(np.mean(fallback_no_feasible_vals)) if fallback_no_feasible_vals else float("nan")
            ),
            "fallback_risk_threshold_rate": (
                float(np.mean(fallback_risk_threshold_vals)) if fallback_risk_threshold_vals else float("nan")
            ),
            "fallback_collision_rate": (
                float(np.mean(fallback_collision_vals)) if fallback_collision_vals else float("nan")
            ),
            "fallback_margin_rate": (
                float(np.mean(fallback_margin_vals)) if fallback_margin_vals else float("nan")
            ),
            "fallback_out_of_map_rate": (
                float(np.mean(fallback_out_of_map_vals)) if fallback_out_of_map_vals else float("nan")
            ),
            "fallback_invalid_cost_rate": (
                float(np.mean(fallback_invalid_cost_vals)) if fallback_invalid_cost_vals else float("nan")
            ),
            "velocity_search_min_predicted_clearance_mean": (
                float(np.mean(velocity_search_min_clearance_vals)) if velocity_search_min_clearance_vals else float("nan")
            ),
            "velocity_search_best_cost_mean": (
                float(np.mean(velocity_search_best_cost_vals)) if velocity_search_best_cost_vals else float("nan")
            ),
            "selected_rollout_min_clearance_mean": (
                float(np.mean(selected_rollout_clearance_vals)) if selected_rollout_clearance_vals else float("nan")
            ),
            "selected_rollout_collision_rate": (
                float(np.mean(selected_rollout_collision_vals)) if selected_rollout_collision_vals else float("nan")
            ),
            "selected_rollout_out_of_map_rate": (
                float(np.mean(selected_rollout_out_of_map_vals)) if selected_rollout_out_of_map_vals else float("nan")
            ),
            "inference_latency_ms_p50": _quantile(latency_ms_samples, 0.50),
            "inference_latency_ms_p95": _quantile(latency_ms_samples, 0.95),
        }

        by_diff: Dict[str, Dict] = {}
        unique_d = sorted(set(float(r["difficulty"]) for r in rows))
        for d in unique_d:
            sub = [r for r in rows if float(r["difficulty"]) == float(d)]
            succ = [int(r.get("task_success", r.get("success", 0))) for r in sub]
            row_progress_d = _clean([r.get("row_progress_success", float("nan")) for r in sub])
            full_task_success_d = [int(r.get("full_task_success", 0)) for r in sub]
            strict_success_d = [int(r.get("strict_success", 0)) for r in sub]
            success_event_d = [int(r.get("success_event", r["success"])) for r in sub]
            success_event_collision_d = [int(r.get("success_event_and_collision", r.get("success_and_collision", 0))) for r in sub]
            collision_only_d = [int(r.get("collision_only", 0)) for r in sub]
            timeout_or_other_d = [int(r.get("timeout_or_other", 0)) for r in sub]
            mae = _clean([r["follow_mae_m"] for r in sub])
            rmse = _clean([r["follow_rmse_m"] for r in sub])
            cot = _clean([r["cot"] for r in sub])
            tts_d = _clean([r["time_to_success_s"] for r in sub if int(r.get("success_event", 0)) == 1])
            success_event_tts_d = _clean([
                r.get("success_event_time_s", float("nan"))
                for r in sub
                if int(r.get("success_event", 0)) == 1
            ])
            cross_line_end_d = _clean([r.get("cross_line_dist_end", float("nan")) for r in sub])
            cross_line_min_d = _clean([r.get("cross_line_dist_min", float("nan")) for r in sub])
            collision_d = [int(r.get("episode_collision", 0)) for r in sub]
            progress_d = [int(r.get("progress_reached", 0)) for r in sub]
            progress_ratio_d = _clean([r.get("progress_ratio_best", float("nan")) for r in sub])
            gate_y_raw_d = _clean([r.get("gate_y_raw_mean", float("nan")) for r in sub])
            y_eff_d = _clean([r.get("y_eff_mean", float("nan")) for r in sub])
            w_d = _clean([r.get("w_mean", float("nan")) for r in sub])
            signed_w_d = _clean([r.get("signed_w_mean", float("nan")) for r in sub])
            signed_w_active_d = _clean([r.get("signed_w_active_mean", float("nan")) for r in sub])
            clearance_f_d = _clean([r.get("clearance_f_mean", float("nan")) for r in sub])
            clearance_a_d = _clean([r.get("clearance_a_mean", float("nan")) for r in sub])
            risk_f_d = _clean([r.get("risk_f_mean", float("nan")) for r in sub])
            risk_a_d = _clean([r.get("risk_a_mean", float("nan")) for r in sub])
            risk_delta_d = _clean([r.get("risk_delta_mean", float("nan")) for r in sub])
            w_support_correction_d = _clean([r.get("w_support_correction_mean", float("nan")) for r in sub])
            risk_diff_correction_d = _clean([r.get("risk_diff_correction_mean", float("nan")) for r in sub])
            switch_d = _clean([r.get("switch_rate", float("nan")) for r in sub])
            near_miss_d = _clean([r.get("near_miss_rate", float("nan")) for r in sub])
            rotate_only_d = _clean([r.get("rotate_only_rate", float("nan")) for r in sub])
            gate_region_y_eff_d = _clean([r.get("gate_region_y_eff_mean", float("nan")) for r in sub])
            gate_region_near_miss_d = _clean([r.get("gate_region_near_miss_rate", float("nan")) for r in sub])
            high_risk_d = _high_risk_summary(sub)
            priv_conflict_d = _priv_conflict_summary(sub)
            unsafe_conflict_d = _unsafe_conflict_summary(sub)
            avoid_conflict_d = _candidate_conflict_summary(sub, "avoid")
            stop_conflict_d = _candidate_conflict_summary(sub, "stop")
            target_fov_d = _target_fov_summary(sub)
            n = len(sub)
            sr = float(sum(succ) / max(1, n))
            row_progress_mean_d = float(np.mean(row_progress_d)) if row_progress_d else float("nan")
            full_task_success_rate_d = float(sum(full_task_success_d) / max(1, n))
            success_event_rate_d = float(sum(success_event_d) / max(1, n))
            success_event_collision_rate_d = float(sum(success_event_collision_d) / max(1, n))
            timeout_or_other_rate_d = float(sum(timeout_or_other_d) / max(1, n))
            collision_only_rate_d = float(sum(collision_only_d) / max(1, n))
            by_diff[f"{d:.3f}"] = {
                "episodes": n,
                "success_rate": sr,
                "task_success_rate": sr,
                "row_progress_success_mean": row_progress_mean_d,
                "full_task_success_rate": full_task_success_rate_d,
                "strict_success_rate": float(sum(strict_success_d) / max(1, n)),
                "success_event_rate": success_event_rate_d,
                "success_event_and_collision_rate": success_event_collision_rate_d,
                "success_and_collision_rate": success_event_collision_rate_d,
                "timeout_or_other_rate": timeout_or_other_rate_d,
                "collision_only_rate": collision_only_rate_d,
                "outcome_total_rate": float("nan"),
                "fail_ratio": float(1.0 - sr),
                "follow_mae_m_mean": float(np.mean(mae)) if mae else float("nan"),
                "follow_rmse_m_mean": float(np.mean(rmse)) if rmse else float("nan"),
                "time_to_success_s_mean": float(np.mean(tts_d)) if tts_d else float("nan"),
                "time_to_success_s_median": float(np.median(tts_d)) if tts_d else float("nan"),
                "time_to_success_s_p95": _quantile(tts_d, 0.95),
                "success_event_time_s_mean": float(np.mean(success_event_tts_d)) if success_event_tts_d else float("nan"),
                "success_event_time_s_median": float(np.median(success_event_tts_d)) if success_event_tts_d else float("nan"),
                "success_event_time_s_p95": _quantile(success_event_tts_d, 0.95),
                "cot_mean": float(np.mean(cot)) if cot else float("nan"),
                "cot_median": float(np.median(cot)) if cot else float("nan"),
                "cot_p95": _quantile(cot, 0.95),
                "cross_line_dist_end_mean": float(np.mean(cross_line_end_d)) if cross_line_end_d else float("nan"),
                "cross_line_dist_min_mean": float(np.mean(cross_line_min_d)) if cross_line_min_d else float("nan"),
                "episode_collision_rate": float(sum(collision_d) / max(1, n)),
                "progress_rate": float(sum(progress_d) / max(1, n)),
                "progress_ratio_mean": float(np.mean(progress_ratio_d)) if progress_ratio_d else 0.0,
                "progress_any_rate": float(sum(progress_d) / max(1, n)),
                "gate_y_raw_mean": float(np.mean(gate_y_raw_d)) if gate_y_raw_d else float("nan"),
                "y_eff_mean": float(np.mean(y_eff_d)) if y_eff_d else float("nan"),
                "w_mean": float(np.mean(w_d)) if w_d else float("nan"),
                "signed_w_mean": float(np.mean(signed_w_d)) if signed_w_d else float("nan"),
                "signed_w_active_mean": float(np.mean(signed_w_active_d)) if signed_w_active_d else float("nan"),
                "clearance_f_mean": float(np.mean(clearance_f_d)) if clearance_f_d else float("nan"),
                "clearance_a_mean": float(np.mean(clearance_a_d)) if clearance_a_d else float("nan"),
                "risk_f_mean": float(np.mean(risk_f_d)) if risk_f_d else float("nan"),
                "risk_a_mean": float(np.mean(risk_a_d)) if risk_a_d else float("nan"),
                "risk_delta_mean": float(np.mean(risk_delta_d)) if risk_delta_d else float("nan"),
                "w_support_correction_mean": float(np.mean(w_support_correction_d)) if w_support_correction_d else float("nan"),
                "risk_diff_correction_mean": float(np.mean(risk_diff_correction_d)) if risk_diff_correction_d else float("nan"),
                "w_clearance_f_corr": _pearson_corr(
                    [float(r.get("w_mean", float("nan"))) for r in sub],
                    [float(r.get("clearance_f_mean", float("nan"))) for r in sub],
                ),
                "w_risk_delta_corr": _pearson_corr(
                    [float(r.get("w_mean", float("nan"))) for r in sub],
                    [float(r.get("risk_delta_mean", float("nan"))) for r in sub],
                ),
                "gate_region_y_eff_mean": float(np.mean(gate_region_y_eff_d)) if gate_region_y_eff_d else float("nan"),
                "gate_region_near_miss_rate_mean": (
                    float(np.mean(gate_region_near_miss_d)) if gate_region_near_miss_d else float("nan")
                ),
                **high_risk_d,
                **priv_conflict_d,
                **unsafe_conflict_d,
                **avoid_conflict_d,
                **stop_conflict_d,
                **target_fov_d,
                "switch_rate_mean": float(np.mean(switch_d)) if switch_d else float("nan"),
                "near_miss_rate_mean": float(np.mean(near_miss_d)) if near_miss_d else float("nan"),
                "rotate_only_rate_mean": float(np.mean(rotate_only_d)) if rotate_only_d else float("nan"),
            }

        is_additive_fusion = bool(getattr(self.args, "additive_fusion", False))
        is_velocity_search = bool(getattr(self.args, "velocity_search", False))
        result = {
            "protocol": {
                "task": self.args.task,
                "mode": self.args.mode,
                "skill": self.args.skill,
                "seed": int(self.args.seed),
                "difficulty_levels": _difficulty_list(self.args.difficulty_levels),
                "episodes": int(self.args.episodes),
                "num_envs": int(self.args.num_envs),
                "pcr_play_env_alignment": bool(_is_pcr_eval_task(self.args)),
                "pcr_new_curriculum": self.resolved_protocol.get("pcr_new_curriculum", None),
                "generalize": bool(getattr(self.args, "generalize", False)),
                "eval_layout": str(getattr(self.args, "eval_layout", "") or ""),
                "layout_split": (
                    "heldout_test"
                    if str(getattr(self.args, "eval_layout", "") or "") == "heldout_irregular_rows"
                    else "main_test"
                ),
                "layout_used_for_training": False if str(getattr(self.args, "eval_layout", "") or "") == "heldout_irregular_rows" else None,
                "layout_used_for_validation": False if str(getattr(self.args, "eval_layout", "") or "") == "heldout_irregular_rows" else None,
                "layout_family": (
                    "irregular_nonmirror_mixed_shape_rows"
                    if str(getattr(self.args, "eval_layout", "") or "") == "heldout_irregular_rows"
                    else ""
                ),
                "layout_features": (
                    "non-mirror; non-strict alternating; consecutive same-side rows; variable row spacing; "
                    "variable passage width; 11 capsules plus 2 limited-size boxes; no walls; no dead ends; no dynamic obstacles"
                    if str(getattr(self.args, "eval_layout", "") or "") == "heldout_irregular_rows"
                    else ""
                ),
                "avoid_stage_override": None if getattr(self.args, "avoid_stage_override", None) is None else int(self.args.avoid_stage_override),
                "freeze_avoid_stage": bool(getattr(self.args, "freeze_avoid_stage", False)) or (
                    getattr(self.args, "avoid_stage_override", None) is not None
                ),
                "pcr_line_target_speed": (
                    None if getattr(self.args, "pcr_line_target_speed", None) is None
                    else float(self.args.pcr_line_target_speed)
                ),
                "pcr_line_target_speed_scale": (
                    None if getattr(self.args, "pcr_line_target_speed_scale", None) is None
                    else float(self.args.pcr_line_target_speed_scale)
                ),
                "resolved_moving_target_pcr_line_speed": (
                    float(getattr(getattr(self.env.env, "nav_cfg", None), "moving_target_pcr_line_speed"))
                    if getattr(getattr(self.env.env, "nav_cfg", None), "moving_target_pcr_line_speed", None) is not None
                    else None
                ),
                "pcr_forced_forward_train_warmup_ratio": float(getattr(self.env, "forced_forward_train_warmup_ratio", float("nan"))),
                "deterministic_policy": bool(not self.args.stochastic),
                "mass_kg_for_cot": float(self.mass_kg),
                "success_definition": (
                    "success/task_success = env PCR semantic success_mask: final row crossed, "
                    "target distance inside follow band, and no episode collision"
                ),
                "row_progress_success_definition": (
                    "row_progress_success = max episode s_avoid_row_success_mask; each safely completed "
                    "obstacle row contributes 1 / total_rows; diagnostic only"
                ),
                "full_task_success_definition": "row_progress_success == 1.0 and no episode collision",
                "strict_success_definition": "env/play success_mask; equal to task_success for PCR eval",
                "success_event_source": "info.success_mask > s_avoid_episode_success_flags > success_bonus",
                "outcome_categories": ["full_row_score", "partial_row_score", "collision", "timeout_or_other"],
                "aff_stack": int(self.args.aff_stack),
                "camera_enable": bool(getattr(self.args, "camera_enable", False)),
                "camera_interval": None if getattr(self.args, "camera_interval", None) is None else int(self.args.camera_interval),
                "gate_use_difficulty": bool(self.args.gate_use_difficulty),
                "gate_safe_clamp": bool(getattr(self.args, "gate_safe_clamp", False)),
                "gate_safe_max": float(getattr(self.args, "gate_safe_max", 0.3)),
                "beta": None if self.args.beta is None else float(self.args.beta),
                "w_mode": str(self.args.w_mode),
                "policy_variant": (
                    "mono_ppo"
                    if bool(getattr(self.args, "mono_ppo", False))
                    else "additive_fusion"
                    if is_additive_fusion
                    else "velocity_search"
                    if is_velocity_search
                    else "rule_override"
                    if bool(getattr(self.args, "rule_override", False))
                    else "learnedw2_no_delta_y_w"
                    if bool(getattr(self.args, "disable_delta_y_w", False))
                    else {"none": "yonly", "geom": "geomw", "risk_only": "risk_only", "learned": "learnedw", "learnedw2": "learnedw2"}.get(str(self.args.w_mode), str(self.args.w_mode))
                ),
                "risk_only": bool(th.is_risk_only_mode(str(self.args.w_mode))),
                "loaded_checkpoint_variant": (
                    self.policy_meta.get("policy_variant", self.policy_meta.get("trained_w_mode", None))
                    if isinstance(self.policy_meta, dict) else None
                ),
                "uses_gate_policy": bool(
                    self.args.skill == "moe"
                    and not getattr(self.args, "mono_ppo", False)
                    and not is_additive_fusion
                    and not is_velocity_search
                ),
                "uses_learned_w_output": bool(
                    th.is_learned_w_mode(str(self.args.w_mode)) and not is_additive_fusion and not is_velocity_search
                ),
                "uses_learned_w_for_control": bool(
                    th.is_learned_w_mode(str(self.args.w_mode)) and not is_additive_fusion and not is_velocity_search
                    and not bool(getattr(self.args, "disable_delta_y_w", False))
                ),
                "uses_additive_fusion": bool(is_additive_fusion),
                "uses_velocity_search": bool(is_velocity_search),
                "uses_target_aware_velocity_search": bool(is_velocity_search),
                "velocity_search_definition": (
                    "DWA-style target-aware velocity-space local planner: samples high-level "
                    "[v_lat,v_fwd,omega], rolls out constant body-frame commands on the same local map, "
                    "rejects footprint collisions as infeasible, then selects by tracking-clearance cost."
                    if is_velocity_search else ""
                ),
                "uses_lateral_projection": bool(is_additive_fusion),
                "shared_postprocess": True,
                "uses_risk_diff_correction": bool(
                    (th.is_learnedw2_mode(str(self.args.w_mode)) or th.is_risk_only_mode(str(self.args.w_mode)))
                    and not is_additive_fusion
                    and not is_velocity_search
                ),
                "disable_delta_y_w": bool(getattr(self.args, "disable_delta_y_w", False)),
                "delta_y_w_forced_zero": bool(getattr(self.args, "disable_delta_y_w", False)),
                "mono_ppo": bool(getattr(self.args, "mono_ppo", False)),
                "uses_follow_expert": bool(self.args.skill == "moe" and not getattr(self.args, "mono_ppo", False)),
                "uses_avoid_expert": bool(self.args.skill == "moe" and not getattr(self.args, "mono_ppo", False)),
                "uses_pcr_gate": bool(
                    self.args.skill == "moe"
                    and not getattr(self.args, "mono_ppo", False)
                    and not is_additive_fusion
                    and not is_velocity_search
                ),
                "uses_follow_expert_for_metrics": bool(self.args.skill == "moe"),
                "uses_avoid_expert_for_metrics": bool(self.args.skill == "moe" and self.avoid_model is not None),
                "conflict_metrics_available": bool(
                    (not getattr(self.args, "mono_ppo", False))
                    or (self.args.skill == "moe" and self.avoid_model is not None)
                ),
                "mechanism_metrics_available": bool(
                    not getattr(self.args, "mono_ppo", False)
                    and not is_additive_fusion
                    and not is_velocity_search
                ),
                "velocity_search_split": str(getattr(self.args, "velocity_search_split", "")),
                "velocity_search_hparams_source": (
                    "json" if str(getattr(self.args, "velocity_search_hparams_json", "") or "").strip()
                    else "default_or_cli"
                ),
                "velocity_search_hparams_json": str(getattr(self.args, "velocity_search_hparams_json", "") or ""),
                "velocity_search_chosen_hparams": dict(self.velocity_search_hparams) if is_velocity_search else {},
                "velocity_search_validation_tuning_protocol": (
                    "Tune weights only with velocity_search_split=validation_layout; "
                    "main_test and heldout_test must use a saved chosen-hparams JSON without further changes."
                    if is_velocity_search else ""
                ),
                "velocity_search_hparams_frozen_for_eval": bool(
                    is_velocity_search and str(getattr(self.args, "velocity_search_split", "main_test")) != "validation_layout"
                ),
                "mono_ppo_conflict_metrics_note": (
                    "For Mono-PPO, conflict rates are eval-only diagnostics computed from the same "
                    "analytic Follow candidate and diagnostic Avoid candidate; they are not policy inputs."
                    if bool(getattr(self.args, "mono_ppo", False)) else ""
                ),
                "cmd_output_convention": "[x_right, y_forward, yaw]",
                "eval_w_mode": str(self.args.w_mode),
                "trained_w_mode": (
                    self.policy_meta.get("trained_w_mode", self.policy_meta.get("w_mode", None))
                    if isinstance(self.policy_meta, dict) else None
                ),
                "actor_output_dim": (
                    self.policy_meta.get("actor_output_dim", None)
                    if isinstance(self.policy_meta, dict) else None
                ),
                "obs_contract_version": (
                    self.policy_meta.get("obs_contract_version", None)
                    if isinstance(self.policy_meta, dict) else None
                ),
                "fusion_formula_version": (
                    self.policy_meta.get("fusion_formula_version", None)
                    if isinstance(self.policy_meta, dict) else None
                ),
                "w_tau": float(self.args.w_tau),
                "w_blend_mode": str(self.args.w_blend_mode),
                "signed_w_lambda": float(getattr(self.args, "signed_w_lambda", 0.30)),
                "signed_w_gamma_risk": float(getattr(self.args, "signed_w_gamma_risk", 0.15)),
                "signed_w_margin": float(getattr(self.args, "signed_w_margin", 0.05)),
                "w_disable_gate_safe_clamp": bool(self.args.w_disable_gate_safe_clamp),
                "rule_override": bool(getattr(self.args, "rule_override", False)),
                "rule_override_definition": (
                    "Reactive safety baseline: deterministic risk-gap rule replaces PCR arbitration; "
                    "it uses cmd_F/cmd_A/risk_F/risk_A but not learned y/w."
                ),
                "rule_k": float(getattr(self.args, "rule_k", 8.0)),
                "rule_margin": float(getattr(self.args, "rule_margin", 0.10)),
                "rule_hard_thr": float(getattr(self.args, "rule_hard_thr", 0.45)),
                "rule_s_min": float(getattr(self.args, "rule_s_min", 0.85)),
                "rule_slow_ratio": float(getattr(self.args, "rule_slow_ratio", 0.10)),
                "rule_yaw_keep_loss": float(getattr(self.args, "rule_yaw_keep_loss", 0.30)),
                "pcr_w_aux_enable": bool(getattr(self.args, "pcr_w_aux_enable", False)),
                "pcr_w_aux_coef": float(getattr(self.args, "pcr_w_aux_coef", 0.0)),
                "pcr_w_aux_risk_f_threshold": float(getattr(self.args, "pcr_w_aux_risk_f_threshold", 0.4)),
                "pcr_w_aux_risk_margin": float(getattr(self.args, "pcr_w_aux_risk_margin", 0.10)),
                "pcr_w_aux_cmd_cos_threshold": float(getattr(self.args, "pcr_w_aux_cmd_cos_threshold", 0.5)),
                "cmd_slew_lin": float(self.args.cmd_slew_lin),
                "cmd_slew_ang": float(self.args.cmd_slew_ang),
                "cmd_safe_dist": None if self.args.cmd_safe_dist is None else float(self.args.cmd_safe_dist),
                "cmd_free_dist": None if self.args.cmd_free_dist is None else float(self.args.cmd_free_dist),
                "disable_risk_scale": bool(self.args.disable_risk_scale),
                "dump_timeseries": bool(getattr(self.args, "dump_timeseries", False)),
                "timeseries_episodes": int(getattr(self.args, "timeseries_episodes", 8)),
                "timeseries_stride": int(getattr(self.args, "timeseries_stride", 1)),
                "w_trigger_threshold": float(getattr(self.args, "w_trigger_threshold", 0.5)),
                "gate_region_risk_threshold": float(getattr(self.args, "gate_region_risk_threshold", 0.5)),
                "priv_conflict_definition": (
                    "eval-only row-command conflict = row obstacle window AND Follow forward pressure "
                    "AND Avoid lateral pressure"
                ),
                "priv_conflict_follow_thr": float(getattr(self.args, "priv_conflict_follow_thr", 0.20)),
                "priv_conflict_avoid_thr": float(getattr(self.args, "priv_conflict_avoid_thr", 0.10)),
                "priv_conflict_pre_m": float(getattr(self.args, "priv_conflict_pre_m", 0.6)),
                "priv_conflict_post_m": float(getattr(self.args, "priv_conflict_post_m", 0.3)),
                "priv_conflict_score_thr": float(getattr(self.args, "priv_conflict_score_thr", 0.25)),
                "unsafe_conflict_definition": (
                    "eval-only unsafe command conflict = row obstacle window AND rollout risk_F above threshold "
                    "AND risk_F-min(risk_A,risk_S) above margin AND command disagreement AND target recoverable"
                ),
                "avoid_conflict_definition": (
                    "C_avoid = C_unsafe AND utility_A > utility_S + margin; "
                    "utility_A = -risk_cost*risk_A + lateral_gain*lateral_opening - target_gap_cost*target_gap_growth; "
                    "utility_S = -risk_cost*risk_S + progress_gain*forward_progress - target_gap_cost*target_gap_growth"
                ),
                "stop_conflict_definition": "C_stop = C_unsafe AND C_avoid is false",
                "conflict_rollout_horizon_s": float(getattr(self.args, "conflict_rollout_horizon_s", 1.2)),
                "conflict_rollout_tube_radius_m": float(getattr(self.args, "conflict_rollout_tube_radius_m", 0.25)),
                "conflict_utility_risk_cost": float(getattr(self.args, "conflict_utility_risk_cost", 1.0)),
                "conflict_utility_progress_gain": float(getattr(self.args, "conflict_utility_progress_gain", 0.25)),
                "conflict_utility_lateral_gain": float(getattr(self.args, "conflict_utility_lateral_gain", 0.35)),
                "conflict_utility_lateral_opening_cap_m": float(getattr(self.args, "conflict_utility_lateral_opening_cap_m", 0.45)),
                "conflict_utility_target_gap_cost": float(getattr(self.args, "conflict_utility_target_gap_cost", 0.35)),
                "conflict_utility_margin": float(getattr(self.args, "conflict_utility_margin", 0.03)),
                "conflict_stop_candidate": str(getattr(self.args, "conflict_stop_candidate", "stop")),
                "conflict_stop_slow_ratio": float(getattr(self.args, "conflict_stop_slow_ratio", 0.2)),
                "unsafe_conflict_risk_f_thr": float(getattr(self.args, "unsafe_conflict_risk_f_thr", 0.25)),
                "unsafe_conflict_risk_margin": float(getattr(self.args, "unsafe_conflict_risk_margin", 0.05)),
                "unsafe_conflict_cmd_cos_thr": float(getattr(self.args, "unsafe_conflict_cmd_cos_thr", 0.5)),
                "unsafe_conflict_avoid_stop_margin": float(getattr(self.args, "unsafe_conflict_avoid_stop_margin", 0.03)),
                "unsafe_conflict_avoid_stop_margin_status": "legacy unused; use conflict_utility_margin",
                "unsafe_conflict_min_steps": int(getattr(self.args, "unsafe_conflict_min_steps", 3)),
                "priv_conflict_bins_support": "obstacle_window_only",
                "priv_conflict_phase_codes": {"0": "none", "1": "approach", "2": "inside", "3": "release"},
                "target_observability_definition": "bearing=atan2(x_right,y_forward); RGB FOV is used for YOLO-style target visibility",
                "target_rgb_fov_deg": float(getattr(self.args, "target_rgb_fov_deg", 69.4)),
                "target_fov_margin_deg": float(getattr(self.args, "target_fov_margin_deg", 3.0)),
                "target_near_fov_edge_margin_deg": float(getattr(self.args, "target_near_fov_edge_margin_deg", 5.0)),
                "target_lost_k_eval": int(getattr(self.args, "target_lost_k_eval", 5)),
                "l2_follow_mae_threshold": float(getattr(self.args, "l2_follow_mae_threshold", 0.5)),
                "l2_target_fov_threshold": float(getattr(self.args, "l2_target_fov_threshold", 0.90)),
                "cmd_x_bias_eps": float(getattr(self.args, "cmd_x_bias_eps", 0.05)),
                "goal_x_bin_threshold": float(getattr(self.args, "goal_x_bin_threshold", 0.15)),
                "pcr_ckpt": os.path.abspath(self.args.pcr_ckpt) if getattr(self.args, "pcr_ckpt", None) else None,
                "ckpt": os.path.abspath(self.args.ckpt) if getattr(self.args, "ckpt", None) else None,
                "follow_ckpt": os.path.abspath(self.args.follow_ckpt) if getattr(self.args, "follow_ckpt", None) else None,
                "avoid_ckpt": os.path.abspath(self.args.avoid_ckpt) if getattr(self.args, "avoid_ckpt", None) else None,
                "vision_ckpt": os.path.abspath(self.args.vision_ckpt) if getattr(self.args, "vision_ckpt", None) else None,
                "lowlevel_ckpt": os.path.abspath(self.args.lowlevel_ckpt) if getattr(self.args, "lowlevel_ckpt", None) else None,
                "low_level_ckpt": os.path.abspath(self.args.low_level_ckpt) if getattr(self.args, "low_level_ckpt", None) else None,
                "unknown_cli_args": list(getattr(self.args, "_unknown_cli", [])),
                "policy_experiment_meta": self.policy_meta,
            },
            "params": self.param_info,
            "overall": overall,
            "by_difficulty": by_diff,
            "risk_bins": risk_bins_overall,
            "conflict_bins": conflict_bins_overall,
            "priv_conflict_bins": priv_conflict_bins_overall,
            "resolved_protocol": self.resolved_protocol,
            "per_episode": [
                {k: v for k, v in row.items() if k not in ("risk_bin_stats", "conflict_bin_stats", "priv_conflict_bin_stats")}
                for row in rows
            ],
        }
        return result


def _write_outputs(metrics: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    csv_path = os.path.join(out_dir, "metrics.csv")
    timeseries_path = os.path.join(out_dir, "timeseries.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    resolved_protocol = metrics.get("resolved_protocol", None)
    if isinstance(resolved_protocol, dict):
        th.write_resolved_protocol_json(
            os.path.join(out_dir, "resolved_protocol.json"),
            resolved_protocol,
        )
    protocol = metrics.get("protocol", {})
    if isinstance(protocol, dict) and protocol.get("policy_variant") == "velocity_search":
        with open(os.path.join(out_dir, "velocity_search_chosen_hparams.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": protocol.get("velocity_search_split", ""),
                    "hparams": protocol.get("velocity_search_chosen_hparams", {}),
                    "validation_tuning_protocol": protocol.get("velocity_search_validation_tuning_protocol", ""),
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    rows = metrics.get("per_episode", [])
    fieldnames = [
        "episode_id",
        "difficulty",
        "success",
        "task_success",
        "l1_safety_success",
        "l2_follow_success",
        "l3_progress_success",
        "row_progress_success",
        "full_task_success",
        "strict_success",
        "strict_terminal_success",
        "success_event",
        "success_event_and_collision",
        "time_to_success_s",
        "success_event_time_s",
        "episode_collision",
        "collision_time_s",
        "collision_only",
        "success_and_collision",
        "timeout_or_other",
        "outcome",
        "follow_mae_m",
        "follow_rmse_m",
        "cot",
        "energy_j",
        "distance_m",
        "steps_hl",
        "cross_line_dist_end",
        "cross_line_dist_min",
        "progress_reached",
        "progress_ratio_best",
        "gate_y_raw_mean",
        "y_eff_mean",
        "w_mean",
        "signed_w_mean",
        "signed_w_active_mean",
        "clearance_f_mean",
        "clearance_a_mean",
        "risk_f_mean",
        "risk_a_mean",
        "risk_delta_mean",
        "clearance_rollout_f_mean",
        "clearance_rollout_a_mean",
        "clearance_rollout_s_mean",
        "risk_rollout_f_mean",
        "risk_rollout_a_mean",
        "risk_rollout_s_mean",
        "risk_rollout_gap_f_min_as_mean",
        "w_support_correction_mean",
        "risk_diff_correction_mean",
        "delta_y_w_raw_mean",
        "delta_y_w_used_mean",
        "delta_y_r_mean",
        "delta_y_total_mean",
        "risk_memory_mean",
        "rule_s_mean",
        "rule_risk_gap_mean",
        "rule_follow_scale_mean",
        "rule_yaw_scale_mean",
        "rule_follow_suppression_mean",
        "rule_s_at_avoid_conflict",
        "rule_follow_scale_at_avoid_conflict",
        "rule_yaw_scale_at_avoid_conflict",
        "rule_follow_suppression_at_avoid_conflict",
        "unsafe_conflict_delta_y_w_raw_mean",
        "unsafe_conflict_delta_y_w_used_mean",
        "unsafe_conflict_delta_y_r_mean",
        "unsafe_conflict_delta_y_total_mean",
        "avoid_conflict_delta_y_w_raw_mean",
        "avoid_conflict_delta_y_w_used_mean",
        "avoid_conflict_delta_y_r_mean",
        "avoid_conflict_delta_y_total_mean",
        "avoid_conflict_cmd_y_mean",
        "avoid_conflict_cmd_f_y_mean",
        "avoid_conflict_cmd_a_y_mean",
        "row_not_released_rate",
        "row_not_released_w_mean",
        "row_released_w_mean",
        "priv_conflict_score_mean",
        "priv_high_conflict_steps",
        "priv_high_conflict_step_rate",
        "priv_obstacle_window_rate",
            "priv_follow_pressure_rate",
            "priv_avoid_pressure_rate",
        "priv_conflict_y_raw_mean",
        "priv_conflict_y_eff_mean",
        "priv_conflict_w_mean",
        "priv_conflict_signed_w_mean",
        "priv_conflict_delta_y_mean",
        "priv_non_conflict_delta_y_mean",
        "conflict_suppression_index",
        "conflict_selective_suppression",
        "relative_conflict_modulation",
        "priv_window_phase_approach_rate",
        "priv_window_phase_inside_rate",
        "priv_window_phase_release_rate",
        "priv_conflict_phase_approach_rate",
        "priv_conflict_phase_inside_rate",
        "priv_conflict_phase_release_rate",
        "priv_conflict_phase_approach_steps",
        "priv_conflict_phase_inside_steps",
        "priv_conflict_phase_release_steps",
        "priv_conflict_phase_approach_w_mean",
        "priv_conflict_phase_inside_w_mean",
        "priv_conflict_phase_release_w_mean",
        "priv_conflict_phase_approach_signed_w_mean",
        "priv_conflict_phase_inside_signed_w_mean",
        "priv_conflict_phase_release_signed_w_mean",
        "priv_conflict_phase_approach_delta_y_mean",
        "priv_conflict_phase_inside_delta_y_mean",
            "priv_conflict_phase_release_delta_y_mean",
            "unsafe_conflict_steps",
            "unsafe_conflict_step_rate",
            "unsafe_follow_risk_rate",
            "unsafe_safe_candidate_better_rate",
            "unsafe_avoid_safer_rate",
            "unsafe_stop_safer_rate",
            "unsafe_command_disagree_rate",
            "unsafe_target_recoverable_rate",
            "unsafe_conflict_y_raw_mean",
            "unsafe_conflict_y_eff_mean",
            "unsafe_conflict_w_mean",
            "unsafe_conflict_signed_w_mean",
            "unsafe_conflict_delta_y_mean",
            "unsafe_non_conflict_delta_y_mean",
            "unsafe_conflict_suppression_index",
            "unsafe_conflict_selective_suppression",
            "unsafe_relative_conflict_modulation",
            "unsafe_conflict_phase_approach_rate",
            "unsafe_conflict_phase_inside_rate",
            "unsafe_conflict_phase_release_rate",
            "unsafe_conflict_phase_approach_w_mean",
            "unsafe_conflict_phase_inside_w_mean",
            "unsafe_conflict_phase_release_w_mean",
            "unsafe_conflict_phase_approach_signed_w_mean",
            "unsafe_conflict_phase_inside_signed_w_mean",
            "unsafe_conflict_phase_release_signed_w_mean",
            "unsafe_conflict_phase_approach_delta_y_mean",
            "unsafe_conflict_phase_inside_delta_y_mean",
            "unsafe_conflict_phase_release_delta_y_mean",
            "unsafe_conflict_visited_episode_rate",
            "unsafe_conflict_visited_collision_rate",
            "unsafe_conflict_visited_row_progress_mean",
            "avoid_conflict_steps",
            "avoid_conflict_step_rate",
            "avoid_conflict_y_raw_mean",
            "avoid_conflict_y_eff_mean",
            "avoid_conflict_cmd_y_mean",
            "avoid_conflict_cmd_f_y_mean",
            "avoid_conflict_cmd_a_y_mean",
            "avoid_conflict_w_mean",
            "avoid_conflict_signed_w_mean",
            "avoid_conflict_delta_y_mean",
            "avoid_conflict_suppression_index",
            "avoid_conflict_visited_episode_rate",
            "avoid_conflict_visited_collision_rate",
            "avoid_conflict_visited_row_progress_mean",
            "stop_conflict_steps",
            "stop_conflict_step_rate",
            "stop_conflict_y_raw_mean",
            "stop_conflict_y_eff_mean",
            "stop_conflict_w_mean",
            "stop_conflict_signed_w_mean",
            "stop_conflict_delta_y_mean",
            "stop_conflict_suppression_index",
            "stop_conflict_visited_episode_rate",
            "stop_conflict_visited_collision_rate",
            "stop_conflict_visited_row_progress_mean",
            "target_bearing_abs_mean",
        "target_bearing_abs_p95",
        "target_bearing_abs_max",
        "target_bearing_abs_deg_mean",
        "target_bearing_abs_deg_p95",
        "target_bearing_abs_deg_max",
        "target_in_rgb_fov_rate",
        "target_near_rgb_fov_edge_rate",
        "target_lost_step_rate",
        "target_lost_episode",
        "target_lost_episode_rate",
        "target_lost_max_consecutive_steps",
        "target_lost_max_consecutive_steps_mean",
        "target_lost_max_consecutive_steps_max",
        "target_bearing_abs_mean_in_priv_conflict",
        "target_bearing_abs_deg_mean_in_priv_conflict",
        "target_bearing_abs_max_in_priv_conflict",
        "target_in_fov_rate_in_priv_conflict",
        "target_near_fov_edge_rate_in_priv_conflict",
        "target_lost_rate_in_priv_conflict",
        "cmd_x_mean",
        "cmd_x_std",
        "cmd_x_abs_mean",
        "cmd_x_positive_rate",
        "cmd_x_negative_rate",
        "cmd_x_zero_rate",
        "cmd_x_direction_entropy",
        "corr_goalx_cmdx",
        "corr_goalx_cmdyaw",
        "corr_goaly_cmdy",
        "cmd_yaw_left_mean",
        "cmd_yaw_right_mean",
        "cmd_x_left_mean",
        "cmd_x_right_mean",
        "yaw_directional_response",
        "lateral_directional_response",
        "cmd_post_delta_abs_mean",
        "cmd_post_delta_x_abs_mean",
        "cmd_post_delta_y_abs_mean",
        "cmd_post_delta_y_signed_mean",
        "cmd_post_delta_w_abs_mean",
        "cmd_post_rewrite_rate",
        "velocity_search_filter_change_rate",
        "velocity_search_safe_available_rate",
        "velocity_search_raw_risk_mean",
        "velocity_search_safe_risk_mean",
        "velocity_search_compute_time_ms_mean",
        "velocity_search_dynamic_window_enabled_rate",
        "velocity_search_candidate_count_before_dynamic_window_mean",
        "velocity_search_candidate_count_after_dynamic_window_mean",
        "velocity_search_dynamic_window_rejected_count_mean",
        "velocity_search_candidate_count_mean",
        "velocity_search_feasible_count_mean",
        "velocity_search_infeasible_count_mean",
        "velocity_search_feasible_count_after_margin_mean",
        "velocity_search_selected_v_lat_mean",
        "velocity_search_selected_v_fwd_mean",
        "velocity_search_selected_yaw_mean",
        "velocity_search_raw_y_mean",
        "velocity_search_after_filter_y_mean",
        "cmd_safe_y_mean",
        "cmd_safe_y_over_raw_y_ratio",
        "actual_base_vel_x_mean",
        "actual_base_vel_y_mean",
        "actual_base_vel_w_mean",
        "actual_base_delta_x_mean",
        "actual_base_delta_y_mean",
        "row_progress_delta_mean",
        "velocity_search_forward_candidate_count_mean",
        "velocity_search_forward_feasible_count_mean",
        "velocity_search_forward_infeasible_collision_count_mean",
        "velocity_search_forward_infeasible_margin_count_mean",
        "velocity_search_forward_infeasible_out_of_map_count_mean",
        "velocity_search_best_forward_feasible_cost_mean",
        "velocity_search_best_forward_feasible_clearance_mean",
        "velocity_search_fallback_rate",
        "fallback_no_feasible_rate",
        "fallback_risk_threshold_rate",
        "fallback_collision_rate",
        "fallback_margin_rate",
        "fallback_out_of_map_rate",
        "fallback_invalid_cost_rate",
        "velocity_search_min_predicted_clearance_mean",
        "velocity_search_best_cost_mean",
        "selected_rollout_min_clearance_mean",
        "selected_rollout_collision_rate",
        "selected_rollout_out_of_map_rate",
        "goal_x_left_steps",
        "goal_x_right_steps",
        "goal_cmd_corr_steps",
        "switch_rate",
        "near_miss_rate",
        "rotate_only_rate",
        "w_trigger_step",
        "w_trigger_progress",
        "gate_region_steps",
        "gate_region_y_eff_mean",
        "gate_region_near_miss_rate",
        "high_risk_steps",
        "high_risk_ratio",
        "high_risk_y_eff_mean",
        "high_risk_w_mean",
        "high_risk_risk_f_mean",
        "high_risk_risk_a_mean",
        "high_risk_risk_delta_mean",
        "high_risk_near_miss_rate",
        "cmd_jerk_lin_mean",
        "cmd_jerk_ang_mean",
        "cmd_f_mean_x",
        "cmd_f_mean_y",
        "cmd_f_mean_w",
        "cmd_a_mean_x",
        "cmd_a_mean_y",
        "cmd_a_mean_w",
        "cmd_final_mean_x",
        "cmd_final_mean_y",
        "cmd_final_mean_w",
    ]
    csv_rows = []
    for row in rows:
        row_out = dict(row)
        row_out.pop("risk_bin_stats", None)
        row_out.pop("conflict_bin_stats", None)
        row_out.pop("priv_conflict_bin_stats", None)
        csv_rows.append(row_out)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    timeseries_rows = metrics.get("timeseries", [])
    if timeseries_rows:
        ts_fieldnames = [
            "episode_id",
            "difficulty",
            "episode_termination_reason",
            "episode_task_success",
            "episode_full_task_success",
            "episode_collision_final",
            "episode_target_lost",
            "episode_row_progress",
            "step",
            "step_hl",
            "time_s",
            "trajectory_frame",
            "robot_x",
            "robot_y",
            "target_x",
            "target_y",
            "target_x_right",
            "target_y_forward",
            "obstacles_json",
            "progress",
            "row_progress_delta",
            "follow_err_m",
            "target_bearing_rad",
            "target_bearing_abs_rad",
            "target_bearing_abs_deg",
            "target_in_rgb_fov",
            "target_near_rgb_fov_edge",
            "target_lost_current_steps",
            "gate_y_raw",
            "y_eff",
            "w",
            "signed_w",
            "signed_w_active",
            "clearance_f",
            "clearance_a",
            "risk_f",
            "risk_a",
            "risk_delta",
            "clearance_rollout_f",
            "clearance_rollout_a",
            "clearance_rollout_s",
            "risk_rollout_f",
            "risk_rollout_a",
            "risk_rollout_s",
            "risk_rollout_gap_f_min_as",
            "utility_a",
            "utility_s",
            "utility_a_minus_s",
            "avoid_lateral_opening",
            "stop_forward_progress",
            "avoid_target_gap_growth",
            "stop_target_gap_growth",
            "follow_weight",
            "avoid_weight",
            "csi",
            "w_support_correction",
            "risk_diff_correction",
            "delta_y_w_raw",
            "delta_y_w_used",
            "delta_y_r",
            "delta_y_total",
            "risk_memory",
            "rule_s",
            "rule_risk_gap",
            "rule_follow_scale",
            "rule_yaw_scale",
            "rule_follow_suppression",
            "row_not_released",
            "conflict_score",
            "priv_conflict_score",
            "priv_high_conflict",
            "priv_obstacle_window",
            "priv_follow_pressure",
            "priv_avoid_pressure",
            "priv_conflict_phase",
            "unsafe_high_conflict",
            "unsafe_follow_risk",
            "unsafe_safe_candidate_better",
            "unsafe_avoid_safer",
            "unsafe_stop_safer",
            "unsafe_command_disagree",
            "target_recoverable_for_conflict",
            "avoid_high_conflict",
            "stop_high_conflict",
            "clearance_pp",
            "near_miss",
            "episode_collision",
            "cmd_f_x",
            "cmd_f_y",
            "cmd_f_w",
            "cmd_a_x",
            "cmd_a_y",
            "cmd_a_w",
            "cmd_s_x",
            "cmd_s_y",
            "cmd_s_w",
            "cmd_search_raw_x",
            "cmd_search_raw_y",
            "cmd_search_raw_w",
            "cmd_search_after_filter_x",
            "cmd_search_after_filter_y",
            "cmd_search_after_filter_w",
            "cmd_safe_x",
            "cmd_safe_y",
            "cmd_safe_w",
            "actual_base_vel_x",
            "actual_base_vel_y",
            "actual_base_vel_w",
            "actual_base_delta_x",
            "actual_base_delta_y",
            "velocity_search_compute_time_ms",
            "velocity_search_dynamic_window_enabled",
            "velocity_search_candidate_count_before_dynamic_window",
            "velocity_search_candidate_count_after_dynamic_window",
            "velocity_search_dynamic_window_rejected_count",
            "velocity_search_candidate_count",
            "velocity_search_feasible_count",
            "velocity_search_infeasible_count",
            "velocity_search_feasible_count_after_margin",
            "velocity_search_forward_candidate_count",
            "velocity_search_forward_feasible_count",
            "velocity_search_forward_infeasible_collision_count",
            "velocity_search_forward_infeasible_margin_count",
            "velocity_search_forward_infeasible_out_of_map_count",
            "velocity_search_best_forward_feasible_cost",
            "velocity_search_best_forward_feasible_clearance",
            "velocity_search_fallback_used",
            "fallback_reason",
            "fallback_no_feasible",
            "fallback_risk_threshold",
            "fallback_collision",
            "fallback_margin",
            "fallback_out_of_map",
            "fallback_invalid_cost",
            "velocity_search_min_predicted_clearance",
            "velocity_search_best_cost",
            "selected_rollout_min_clearance",
            "selected_rollout_collision_flag",
            "selected_rollout_out_of_map_flag",
            "fallback_cmd_x",
            "fallback_cmd_y",
            "fallback_cmd_w",
            "cmd_raw_before_postprocess_x",
            "cmd_raw_before_postprocess_y",
            "cmd_raw_before_postprocess_w",
            "cmd_safe_after_postprocess_x",
            "cmd_safe_after_postprocess_y",
            "cmd_safe_after_postprocess_w",
            "cmd_post_delta_x",
            "cmd_post_delta_y",
            "cmd_post_delta_w",
            "cmd_final_x",
            "cmd_final_y",
            "cmd_final_w",
        ]
        with open(timeseries_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ts_fieldnames)
            writer.writeheader()
            for row in timeseries_rows:
                writer.writerow(row)
    _write_mechanism_plots(metrics, out_dir)


def _write_mechanism_plots(metrics: Dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Eval] mechanism plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    w_mode = str(metrics.get("protocol", {}).get("w_mode", "none")).lower()

    def _plot_bins(bin_key: str, title: str, xlabel: str, file_stem: str) -> None:
        bins = metrics.get(bin_key, [])
        if not isinstance(bins, list) or len(bins) == 0:
            return
        labels = [str(b.get("bin", "")) for b in bins]
        xs = np.arange(len(labels), dtype=np.float32)

        def vals(key: str) -> List[float]:
            return [_safe_float(item.get(key, float("nan")), default=float("nan")) for item in bins]

        def _set_tight_ylim(ax, series_list: List[List[float]], *, include_zero: bool = False) -> None:
            data = []
            for series in series_list:
                data.extend([float(v) for v in series if math.isfinite(float(v))])
            if not data:
                return
            if include_zero:
                data.append(0.0)
            lo = min(data)
            hi = max(data)
            span = hi - lo
            if span < 1e-9:
                pad = max(abs(hi) * 0.25, 1e-4)
            else:
                pad = max(span * 0.18, 1e-4)
            ax.set_ylim(lo - pad, hi + pad)

        y_raw = vals("gate_y_raw_mean")
        y_eff = vals("y_eff_mean")
        suppression = vals("suppression_mean")
        w = vals("w_mean")
        signed_w = vals("signed_w_mean")
        success = vals("task_success_rate")
        collision = vals("collision_episode_rate")
        steps = vals("steps")
        episodes = vals("episode_count")

        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
        fig.suptitle(title, fontsize=13)

        axes[0, 0].plot(xs, y_raw, marker="o", linewidth=2.0, color="#5b7083", label="y_raw")
        axes[0, 0].plot(xs, y_eff, marker="s", linewidth=2.0, color="#1f77b4", label="y_eff")
        axes[0, 0].set_title("Follow Weight")
        axes[0, 0].set_ylabel("mean")
        axes[0, 0].set_ylim(0.0, 1.0)
        axes[0, 0].legend(loc="best", frameon=False)

        axes[0, 1].plot(xs, suppression, marker="o", linewidth=2.0, color="#ff7f0e", label="y_raw - y_eff")
        if w_mode == "learnedw2":
            axes[0, 1].plot(xs, signed_w, marker="s", linewidth=2.0, color="#9467bd", label="signed_w")
            mod_series = [suppression, signed_w]
        elif w_mode != "none":
            axes[0, 1].plot(xs, w, marker="s", linewidth=2.0, color="#9467bd", label="w")
            mod_series = [suppression, w]
        else:
            mod_series = [suppression]
        axes[0, 1].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.8)
        axes[0, 1].set_title("Conflict Modulation")
        axes[0, 1].set_ylabel("mean")
        _set_tight_ylim(axes[0, 1], mod_series, include_zero=True)
        axes[0, 1].legend(loc="best", frameon=False)

        axes[1, 0].plot(xs, success, marker="o", linewidth=2.0, color="#2ca02c", label="row-progress score")
        axes[1, 0].plot(xs, collision, marker="s", linewidth=2.0, color="#d62728", label="collision")
        axes[1, 0].set_title("Row-Progress Score and Collision")
        axes[1, 0].set_ylabel("score / rate")
        axes[1, 0].set_ylim(0.0, 1.0)
        axes[1, 0].legend(loc="best", frameon=False)

        axes[1, 1].bar(xs - 0.18, steps, width=0.36, color="#9467bd", label="steps")
        axes[1, 1].set_title("Bin Support")
        axes[1, 1].set_ylabel("steps")
        ax_ep = axes[1, 1].twinx()
        ax_ep.bar(xs + 0.18, episodes, width=0.36, color="#8c564b", alpha=0.85, label="episodes")
        ax_ep.set_ylabel("episodes")
        lines_1, labels_1 = axes[1, 1].get_legend_handles_labels()
        lines_2, labels_2 = ax_ep.get_legend_handles_labels()
        axes[1, 1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", frameon=False)

        for ax in axes.reshape(-1):
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel(xlabel)

        fig_path = os.path.join(out_dir, file_stem)
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)
        print(f"[Eval] mechanism plot: {fig_path}", flush=True)

    _plot_bins(
        "risk_bins",
        "PCR Conflict Arbitration Mechanism by Follow-Risk Bin",
        "risk_f bin",
        "mechanism_risk_bins.png",
    )
    _plot_bins(
        "priv_conflict_bins",
        "PCR Conflict Arbitration Mechanism by Privileged Conflict Bin",
        "privileged conflict score bin",
        "mechanism_priv_conflict_bins.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Independent high-level evaluation")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--skill", type=str, default="moe", choices=["follow", "avoid", "moe"])
    parser.add_argument("--mono_ppo", action="store_true", help="PCR external baseline: direct cmd policy under --skill moe")

    parser.add_argument("--pcr_ckpt", type=str, default=None, help="PCR gate policy checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--follow_ckpt", type=str, default=None, help="旧参数保留；当前 moe 不再需要，因为 follow 使用解析式 expert")
    parser.add_argument("--avoid_ckpt", type=str, default=th.DEFAULT_AVOID_CKPT, help=f"avoid checkpoint for moe (default {th.DEFAULT_AVOID_CKPT})")
    parser.add_argument("--vision_ckpt", type=str, default=None, help="vision checkpoint for student mode")
    parser.add_argument("--lowlevel_ckpt", type=str, default=th.DEFAULT_LOWLEVEL_CKPT, help=f"low-level locomotion checkpoint (default {th.DEFAULT_LOWLEVEL_CKPT})")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=25)
    parser.add_argument("--decimation", type=int, default=5)
    parser.add_argument("--aff_stack", type=int, default=1)
    parser.add_argument("--camera_enable", action="store_true", default=False)
    parser.add_argument("--camera_interval", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--difficulty_levels", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--stochastic", action="store_true", help="use stochastic policy sampling")
    parser.add_argument(
        "--generalize",
        action="store_true",
        help="s_pcr_new high-difficulty generalization eval: five rows, faster target, compressed row spacing",
    )
    parser.add_argument(
        "--eval_layout",
        type=str,
        default="",
        choices=["", "heldout_irregular_rows"],
        help="eval-only layout split; heldout_irregular_rows is never used for training or tuning",
    )
    parser.add_argument(
        "--export_eval_layout_debug",
        type=str,
        default="",
        help="optional directory to export the actual eval obstacle layout CSV/PNG",
    )
    parser.add_argument(
        "--avoid_stage_override",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="fix PCR obstacle stage, matching play_highlevel override semantics",
    )
    parser.add_argument(
        "--freeze_avoid_stage",
        action="store_true",
        help="freeze s_avoid stage during eval; implied by --avoid_stage_override",
    )
    parser.add_argument(
        "--pcr_line_target_speed",
        type=float,
        default=None,
        help="override s_pcr_line_avoid_basic scripted target speed [m/s]",
    )
    parser.add_argument(
        "--pcr_line_target_speed_scale",
        type=float,
        default=None,
        help="multiply s_pcr_line_avoid_basic default scripted target speed",
    )

    parser.add_argument("--gate_use_difficulty", action="store_true")
    parser.add_argument("--gate_safe_clamp", action="store_true")
    parser.add_argument("--gate_safe_max", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=None)
    w_group = parser.add_mutually_exclusive_group()
    w_group.add_argument("--yonly", action="store_true", help="evaluate MoE-y without w")
    w_group.add_argument("--wgeom", action="store_true", help="evaluate MoE-y with geometric w")
    w_group.add_argument("--wriskonly", action="store_true", help="evaluate trained Risk-only baseline")
    w_group.add_argument("--wlearned", action="store_true", help="evaluate MoE-y with learned w")
    w_group.add_argument("--wlearned2", action="store_true", help="evaluate learnedw2 signed conflict prior")
    parser.add_argument("--w_mode", type=str, default=None, choices=["none", "geom", "risk_only", "learned", "learnedw2"])
    parser.add_argument("--w_tau", type=float, default=0.25)
    parser.add_argument("--w_blend_mode", type=str, default="multiply", choices=["multiply", "mix"])
    parser.add_argument("--signed_w_lambda", type=float, default=0.30)
    parser.add_argument("--signed_w_gamma_risk", type=float, default=0.15)
    parser.add_argument("--signed_w_margin", type=float, default=0.05)
    parser.add_argument("--w2_lambda", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--w2_risk_gamma", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--w_disable_gate_safe_clamp", action="store_true")
    parser.add_argument(
        "--disable_delta_y_w",
        action="store_true",
        help="Eval-only ablation: keep learned-w raw output/logging but force delta_y_w_used=0 before y_eff.",
    )
    parser.add_argument("--rule_override", action="store_true", help="replace learned PCR arbitration with reactive safety rule")
    parser.add_argument("--additive_fusion", action="store_true", help="eval-only baseline: cmd_F + lateral-projected cmd_A")
    parser.add_argument("--velocity_search", action="store_true", help="eval-only target-aware velocity-space search baseline")
    parser.add_argument(
        "--velocity_search_split",
        choices=("validation_layout", "main_test", "heldout_test"),
        default="main_test",
    )
    parser.add_argument("--velocity_search_hparams_json", type=str, default="")
    parser.add_argument("--velocity_search_vx_samples", type=int, default=None)
    parser.add_argument("--velocity_search_vy_samples", type=int, default=None)
    parser.add_argument("--velocity_search_w_samples", type=int, default=None)
    parser.add_argument("--velocity_search_rollout_horizon_s", type=float, default=None)
    parser.add_argument("--velocity_search_rollout_dt_s", type=float, default=None)
    parser.add_argument("--velocity_search_body_radius_m", type=float, default=None)
    parser.add_argument("--velocity_search_footprint_radius_m", type=float, default=None)
    parser.add_argument("--velocity_search_min_clearance_margin", type=float, default=None)
    parser.add_argument("--velocity_search_clear_weight", type=float, default=None)
    parser.add_argument("--velocity_search_target_dist_weight", type=float, default=None)
    parser.add_argument("--velocity_search_bearing_weight", type=float, default=None)
    parser.add_argument("--velocity_search_progress_weight", type=float, default=None)
    parser.add_argument("--velocity_search_forward_weight", type=float, default=None)
    parser.add_argument("--velocity_search_forward_min_v", type=float, default=None)
    parser.add_argument("--velocity_search_smooth_weight", type=float, default=None)
    parser.add_argument("--velocity_search_target_weight", type=float, default=None)
    parser.add_argument("--velocity_search_risk_weight", type=float, default=None)
    parser.add_argument("--velocity_search_speed_weight", type=float, default=None)
    parser.add_argument("--velocity_search_lateral_weight", type=float, default=None)
    parser.add_argument("--velocity_search_risk_filter_threshold", type=float, default=None)
    parser.add_argument("--velocity_search_filter_mode", choices=("scale", "soft", "none"), default=None)
    parser.add_argument("--velocity_search_fallback", choices=("stop", "safest", "raw"), default=None)
    parser.add_argument("--velocity_search_use_dynamic_window", action="store_true")
    parser.add_argument("--velocity_search_acc_lat_max", type=float, default=None)
    parser.add_argument("--velocity_search_acc_fwd_max", type=float, default=None)
    parser.add_argument("--velocity_search_acc_yaw_max", type=float, default=None)
    parser.add_argument("--rule_k", type=float, default=8.0)
    parser.add_argument("--rule_margin", type=float, default=0.10)
    parser.add_argument("--rule_hard_thr", type=float, default=0.45)
    parser.add_argument("--rule_s_min", type=float, default=0.85)
    parser.add_argument("--rule_slow_ratio", type=float, default=0.10)
    parser.add_argument("--rule_yaw_keep_loss", type=float, default=0.30)
    parser.add_argument("--risk_memory", action="store_true", help="use deployable temporal risk memory in learned-w row slot")
    parser.add_argument("--risk_memory_l_clear", type=float, default=0.40)
    parser.add_argument("--risk_memory_velocity_source", type=str, default="body", choices=["body", "cmd"])
    parser.add_argument("--pcr_w_aux_enable", action="store_true")
    parser.add_argument("--pcr_w_aux_coef", type=float, default=0.05)
    parser.add_argument("--pcr_w_aux_risk_f_threshold", type=float, default=0.4)
    parser.add_argument("--pcr_w_aux_risk_margin", type=float, default=0.10)
    parser.add_argument("--pcr_w_aux_cmd_cos_threshold", type=float, default=0.5)
    parser.add_argument("--cmd_slew_lin", type=float, default=0.2)
    parser.add_argument("--cmd_slew_ang", type=float, default=0.4)
    parser.add_argument("--cmd_safe_dist", type=float, default=None)
    parser.add_argument("--cmd_free_dist", type=float, default=None)
    parser.add_argument("--disable_risk_scale", action="store_true")
    parser.add_argument("--dump_timeseries", action="store_true", help="write step-level mechanism traces")
    parser.add_argument("--timeseries_episodes", type=int, default=8, help="number of completed episodes to trace")
    parser.add_argument("--timeseries_stride", type=int, default=1, help="record one trace row every N high-level steps")
    parser.add_argument("--progress_interval_s", type=float, default=5.0, help="print eval progress every N seconds; <=0 disables")
    parser.add_argument("--w_trigger_threshold", type=float, default=0.5, help="threshold for w trigger timing metrics")
    parser.add_argument(
        "--gate_region_risk_threshold",
        type=float,
        default=0.5,
        help="risk_F threshold used for narrow-gap/high-conflict region summaries",
    )
    parser.add_argument("--target_rgb_fov_deg", type=float, default=69.4)
    parser.add_argument("--target_fov_margin_deg", type=float, default=3.0)
    parser.add_argument("--target_near_fov_edge_margin_deg", type=float, default=5.0)
    parser.add_argument("--target_lost_k_eval", type=int, default=5)
    parser.add_argument("--l2_follow_mae_threshold", type=float, default=0.5)
    parser.add_argument("--l2_target_fov_threshold", type=float, default=0.90)
    parser.add_argument("--cmd_x_bias_eps", type=float, default=0.05)
    parser.add_argument("--goal_x_bin_threshold", type=float, default=0.15)
    parser.add_argument("--priv_conflict_follow_thr", type=float, default=0.20)
    parser.add_argument("--priv_conflict_avoid_thr", type=float, default=0.10)
    parser.add_argument(
        "--priv_conflict_pre_m",
        type=float,
        default=0.6,
        help="front-edge approach window [m] for eval-only privileged conflict evidence",
    )
    parser.add_argument(
        "--priv_conflict_post_m",
        type=float,
        default=0.3,
        help="rear-edge release window [m] for eval-only privileged conflict evidence",
    )
    parser.add_argument("--priv_conflict_score_thr", type=float, default=0.25)
    parser.add_argument("--unsafe_conflict_risk_f_thr", type=float, default=0.25)
    parser.add_argument("--unsafe_conflict_risk_margin", type=float, default=0.05)
    parser.add_argument("--unsafe_conflict_cmd_cos_thr", type=float, default=0.5)
    parser.add_argument(
        "--unsafe_conflict_avoid_stop_margin",
        type=float,
        default=0.03,
        help="legacy unused; C_avoid/C_stop now uses --conflict_utility_margin",
    )
    parser.add_argument("--unsafe_conflict_min_steps", type=int, default=3)
    parser.add_argument("--conflict_rollout_horizon_s", type=float, default=1.2)
    parser.add_argument("--conflict_rollout_tube_radius_m", type=float, default=0.25)
    parser.add_argument("--conflict_utility_risk_cost", type=float, default=1.0)
    parser.add_argument("--conflict_utility_progress_gain", type=float, default=0.25)
    parser.add_argument("--conflict_utility_lateral_gain", type=float, default=0.35)
    parser.add_argument("--conflict_utility_lateral_opening_cap_m", type=float, default=0.45)
    parser.add_argument("--conflict_utility_target_gap_cost", type=float, default=0.35)
    parser.add_argument("--conflict_utility_margin", type=float, default=0.03)
    parser.add_argument("--conflict_stop_candidate", choices=("stop", "slow"), default="stop")
    parser.add_argument("--conflict_stop_slow_ratio", type=float, default=0.2)

    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--viewer", action="store_true", help="open Isaac Gym viewer during eval")
    parser.add_argument("--debug", action="store_true", help="enable debug prints and scene visualization")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/highlevel")

    args, unknown = parser.parse_known_args()
    if bool(getattr(args, "generalize", False)) and str(getattr(args, "task", "")) != "s_pcr_new":
        parser.error("--generalize 仅支持 --task s_pcr_new")
    if bool(getattr(args, "generalize", False)) and getattr(args, "avoid_stage_override", None) not in (None, 4):
        parser.error("--generalize 固定 5 行障碍，只允许省略 --avoid_stage_override 或显式传 4")
    if str(getattr(args, "eval_layout", "") or "").strip():
        if str(getattr(args, "task", "")) != "s_pcr_line_avoid_basic":
            parser.error("--eval_layout 当前只支持 --task s_pcr_line_avoid_basic")
        if getattr(args, "avoid_stage_override", None) not in (None, 4):
            parser.error("--eval_layout heldout_irregular_rows 固定 Stage 4，只允许省略 --avoid_stage_override 或显式传 4")
    if getattr(args, "pcr_line_target_speed", None) is not None and getattr(args, "pcr_line_target_speed_scale", None) is not None:
        parser.error("--pcr_line_target_speed 与 --pcr_line_target_speed_scale 只能二选一")
    if not getattr(args, "pcr_ckpt", None) and getattr(args, "ckpt", None):
        args.pcr_ckpt = args.ckpt
    if not str(getattr(args, "pcr_ckpt", "") or "").strip():
        parser.error("缺少必须的 checkpoint 路径：--pcr_ckpt")

    if bool(getattr(args, "mono_ppo", False)):
        if args.skill != "moe":
            parser.error("--mono_ppo 只支持 --skill moe")
        if args.rule_override:
            parser.error("--mono_ppo 不允许同时启用 --rule_override")
        if args.additive_fusion:
            parser.error("--mono_ppo 不允许同时启用 --additive_fusion")
        if args.velocity_search:
            parser.error("--mono_ppo 不允许同时启用 --velocity_search")
        if any((args.yonly, args.wgeom, args.wriskonly, args.wlearned, args.wlearned2)):
            parser.error("--mono_ppo 不允许同时指定 --yonly/--wgeom/--wriskonly/--wlearned/--wlearned2")
        if args.w_mode is not None and str(args.w_mode).lower() != "none":
            parser.error("--mono_ppo 不使用 w/y 机制，--w_mode 必须为 none")
        selected_w_mode = "none"
    elif args.additive_fusion:
        if args.skill != "moe":
            parser.error("--additive_fusion 只支持 --skill moe")
        if args.rule_override:
            parser.error("--additive_fusion 不允许同时启用 --rule_override")
        if args.velocity_search:
            parser.error("--additive_fusion 不允许同时启用 --velocity_search")
        if any((args.wgeom, args.wriskonly, args.wlearned, args.wlearned2)):
            parser.error("--additive_fusion 不允许同时指定 --wgeom/--wriskonly/--wlearned/--wlearned2")
        if args.w_mode is not None and str(args.w_mode).lower() != "none":
            parser.error("--additive_fusion 不使用 GatePolicy/w，--w_mode 必须为 none")
        selected_w_mode = "none"
    elif args.velocity_search:
        if args.skill != "moe":
            parser.error("--velocity_search 只支持 --skill moe")
        if args.rule_override:
            parser.error("--velocity_search 不允许同时启用 --rule_override")
        if any((args.wgeom, args.wriskonly, args.wlearned, args.wlearned2)):
            parser.error("--velocity_search 不允许同时指定 --wgeom/--wriskonly/--wlearned/--wlearned2")
        if args.w_mode is not None and str(args.w_mode).lower() != "none":
            parser.error("--velocity_search 不使用 GatePolicy/w，--w_mode 必须为 none")
        selected_w_mode = "none"
    elif args.yonly:
        selected_w_mode = "none"
    elif args.wgeom:
        selected_w_mode = "geom"
    elif args.wriskonly:
        selected_w_mode = "risk_only"
    elif args.wlearned:
        selected_w_mode = "learned"
    elif args.wlearned2:
        selected_w_mode = "learnedw2"
    elif args.w_mode is not None:
        selected_w_mode = args.w_mode
    else:
        selected_w_mode, reason = ph._infer_play_w_mode_from_ckpt(args.pcr_ckpt)
        if selected_w_mode is None:
            parser.error(
                f"无法从 --pcr_ckpt 自动识别策略模式：{reason}；请显式加 --yonly / --wgeom / --wriskonly / --wlearned / --wlearned2"
            )
        print(f"[Eval] auto w_mode={selected_w_mode} ({reason})")
        if selected_w_mode == "none" and "actor_output_dim=1 without geom metadata" in reason:
            print("[Eval] 如果这是 geom-w checkpoint，请显式加 --wgeom；一维 ckpt 不能仅靠网络结构区分 yonly/geom-w。")
    if args.w_mode is not None and args.w_mode != selected_w_mode:
        parser.error(
            f"--w_mode={args.w_mode} 与策略模式 --{ {'none': 'yonly', 'geom': 'wgeom', 'risk_only': 'wriskonly', 'learned': 'wlearned', 'learnedw2': 'wlearned2'}[selected_w_mode] } 不一致"
        )
    if bool(getattr(args, "disable_delta_y_w", False)):
        if bool(getattr(args, "rule_override", False)):
            parser.error("--disable_delta_y_w 是 learnedw2 推理消融，不允许同时启用 --rule_override")
        if selected_w_mode != "learnedw2":
            parser.error("--disable_delta_y_w 只允许配合 --wlearned2 / --w_mode learnedw2 使用")
    args.w_mode = selected_w_mode
    args._eval_selected_w_mode = selected_w_mode
    args._runtime_ablation_cli_overrides = {"w_mode": selected_w_mode}
    required_ckpts = ("lowlevel_ckpt",) if bool(getattr(args, "mono_ppo", False)) else ("avoid_ckpt", "lowlevel_ckpt")
    for opt_name in required_ckpts:
        if not str(getattr(args, opt_name, "") or "").strip():
            parser.error(f"缺少必须的 checkpoint 路径：--{opt_name}")
    args.ckpt = args.pcr_ckpt
    args.teacher_ckpt = args.pcr_ckpt
    args.low_level_ckpt = args.lowlevel_ckpt
    if args.camera_interval is not None and args.camera_interval < 1:
        args.camera_interval = 1
    if bool(getattr(args, "viewer", False)):
        args.headless = False
    th.capture_cli_explicit_arg_values(args, parser)
    args._runtime_ablation_cli_overrides["w_mode"] = selected_w_mode
    if not hasattr(args, "physics_engine"):
        args.physics_engine = gymapi.SIM_PHYSX
    if not hasattr(args, "sim_device_type"):
        args.sim_device_type = "cuda"
    if not hasattr(args, "compute_device_id"):
        args.compute_device_id = 0
    if not hasattr(args, "sim_device_id"):
        args.sim_device_id = args.compute_device_id
    if not hasattr(args, "sim_device"):
        args.sim_device = f"cuda:{args.sim_device_id}" if args.sim_device_type == "cuda" else "cpu"
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
    if hasattr(th, "normalize_task_name"):
        args.task = th.normalize_task_name(getattr(args, "task", ""))
    args._unknown_cli = list(unknown)
    sys.argv = [sys.argv[0]] + unknown
    return args


def main():
    args = parse_args()
    runner = EvalRunner(args)
    metrics = runner.evaluate()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if bool(getattr(args, "mono_ppo", False)):
        variant_tag = "mono_ppo"
    elif bool(getattr(args, "additive_fusion", False)):
        variant_tag = "additive_fusion"
    elif bool(getattr(args, "velocity_search", False)):
        variant_tag = f"velocity_search_{str(args.velocity_search_split)}"
    elif bool(getattr(args, "rule_override", False)):
        variant_tag = (
            f"rule_override_k{float(args.rule_k):g}_m{float(args.rule_margin):g}_"
            f"h{float(args.rule_hard_thr):g}_smin{float(args.rule_s_min):g}_"
            f"slow{float(args.rule_slow_ratio):g}_yawloss{float(args.rule_yaw_keep_loss):g}"
        )
    else:
        variant_tag = th.format_pcr_variant_tag(args)
    out_dir = os.path.join(args.output_dir, f"{args.skill}_{args.mode}_{args.task}_{variant_tag}_seed{int(args.seed)}_{ts}")
    _write_outputs(metrics, out_dir)

    overall = metrics["overall"]
    print("=" * 72)
    print("Independent Eval Complete")
    print(f"Output: {out_dir}")
    print("-" * 72)
    print(
        f"Task success / row-progress: "
        f"{overall['task_success_rate']:.4f} / {overall['row_progress_success_mean']:.4f}"
    )
    print(
        "Episode diagnostics full/strict/event/event+collision/collision/zero-progress-timeout: "
        f"{overall['full_task_success_rate']:.4f} / {overall['strict_success_rate']:.4f} / "
        f"{overall['success_event_rate']:.4f} / {overall['success_event_and_collision_rate']:.4f} / "
        f"{overall['episode_collision_rate']:.4f} / {overall['timeout_or_other_rate']:.4f}"
    )
    print(
        "Priv-conflict step/score/CSI/CSS/RCM: "
        f"{overall['priv_high_conflict_step_rate']:.4f} / {overall['priv_conflict_score_mean']:.4f} / "
        f"{overall['conflict_suppression_index']:.4f} / "
        f"{overall['conflict_selective_suppression']:.4f} / "
        f"{overall['relative_conflict_modulation']:.4f}"
    )
    print(
        "Priv-conflict signed_w/delta_y in/out: "
        f"{overall['priv_conflict_signed_w_mean']:.4f} / "
        f"{overall['priv_conflict_delta_y_mean']:.4f} / "
        f"{overall['priv_non_conflict_delta_y_mean']:.4f}"
    )
    print(
        "Priv-conflict phase signed_w approach/inside/release: "
        f"{overall['priv_conflict_phase_approach_signed_w_mean']:.4f} / "
        f"{overall['priv_conflict_phase_inside_signed_w_mean']:.4f} / "
        f"{overall['priv_conflict_phase_release_signed_w_mean']:.4f}"
    )
    print(
        "Priv-conflict phase delta_y approach/inside/release: "
        f"{overall['priv_conflict_phase_approach_delta_y_mean']:.4f} / "
        f"{overall['priv_conflict_phase_inside_delta_y_mean']:.4f} / "
        f"{overall['priv_conflict_phase_release_delta_y_mean']:.4f}"
    )
    print(
        "Unsafe-conflict step/CSI/CSS/RCM: "
        f"{overall['unsafe_conflict_step_rate']:.4f} / "
        f"{overall['unsafe_conflict_suppression_index']:.4f} / "
        f"{overall['unsafe_conflict_selective_suppression']:.4f} / "
        f"{overall['unsafe_relative_conflict_modulation']:.4f}"
    )
    print(
        "Unsafe-conflict signed_w/delta_y in/out: "
        f"{overall['unsafe_conflict_signed_w_mean']:.4f} / "
        f"{overall['unsafe_conflict_delta_y_mean']:.4f} / "
        f"{overall['unsafe_non_conflict_delta_y_mean']:.4f}"
    )
    print(
        "C_avoid/C_stop step/CSI: "
        f"{overall['avoid_conflict_step_rate']:.4f} / "
        f"{overall['avoid_conflict_suppression_index']:.4f} | "
        f"{overall['stop_conflict_step_rate']:.4f} / "
        f"{overall['stop_conflict_suppression_index']:.4f}"
    )
    print(
        "Rollout risk F/A/S/gap: "
        f"{overall['risk_rollout_f_mean']:.4f} / "
        f"{overall['risk_rollout_a_mean']:.4f} / "
        f"{overall['risk_rollout_s_mean']:.4f} / "
        f"{overall['risk_rollout_gap_f_min_as_mean']:.4f}"
    )
    print(
        "Target FOV in/all/lost/maxLost/bearing p95[deg]: "
        f"{overall['target_in_rgb_fov_rate']:.4f} / "
        f"{overall['target_in_fov_rate_in_priv_conflict']:.4f} / "
        f"{overall['target_lost_step_rate']:.4f} / "
        f"{overall['target_lost_max_consecutive_steps_max']:.0f} / "
        f"{overall['target_bearing_abs_deg_p95']:.2f}"
    )
    print(
        "L1/L2/L3/task safety/follow/progress/task: "
        f"{overall['l1_safety_success_rate']:.4f} / "
        f"{overall['l2_follow_success_rate']:.4f} / "
        f"{overall['l3_progress_success_rate']:.4f} / "
        f"{overall['task_success_rate']:.4f}"
    )
    print(
        "Goal-command corr gx-cx/gx-yaw/gy-cy: "
        f"{overall['corr_goalx_cmdx']:.4f} / "
        f"{overall['corr_goalx_cmdyaw']:.4f} / "
        f"{overall['corr_goaly_cmdy']:.4f}"
    )
    print(
        "CmdX bias/std/entropy pos/neg: "
        f"{overall['cmd_x_mean']:.4f} / "
        f"{overall['cmd_x_std']:.4f} / "
        f"{overall['cmd_x_direction_entropy']:.4f} / "
        f"{overall['cmd_x_positive_rate']:.4f} / "
        f"{overall['cmd_x_negative_rate']:.4f}"
    )
    if overall["priv_obstacle_window_rate"] >= 0.95:
        print(
            "[Warn] privileged obstacle window covers almost the full eval trace; "
            "tighten --priv_conflict_pre_m/--priv_conflict_post_m before using mechanism plots as paper evidence."
        )
    print(f"Follow MAE/RMSE [m]: {overall['follow_mae_m_mean']:.4f} / {overall['follow_rmse_m_mean']:.4f}")
    print(
        "Time-to-success [s] mean/med/p95: "
        f"{overall['time_to_success_s_mean']:.4f} / {overall['time_to_success_s_median']:.4f} / {overall['time_to_success_s_p95']:.4f}"
    )
    print(f"CoT mean/med/p95: {overall['cot_mean']:.4f} / {overall['cot_median']:.4f} / {overall['cot_p95']:.4f}")
    print(
        "Inference latency [ms] p50/p95: "
        f"{overall['inference_latency_ms_p50']:.4f} / {overall['inference_latency_ms_p95']:.4f}"
    )
    print(
        "Params total/trainable: "
        f"{metrics['params'].get('high_level_total', 0)} / {metrics['params'].get('high_level_trainable', 0)}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
