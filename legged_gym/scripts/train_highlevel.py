"""
scripts/train_highlevel.py - V5 Command-Space MoE 训练脚本

核心变化:
1) Experts 直接输出 cmd_vel, Gate 输出 y (MoE 仲裁)
2) Command Post-Processor 统一命令限幅/滤波/风险钳制
3) Teacher/Student 仅用于 affordance 蒸馏

用法:
    Follow:
        python scripts/train_highlevel.py --mode teacher --skill follow --task s_follow_basic \\
            --low_level_ckpt agents/fast_2000.pt

    Avoid:
        python scripts/train_highlevel.py --mode teacher --skill avoid --task s_avoid_basic \\
            --low_level_ckpt agents/fast_2000.pt

    Gate:
        python scripts/train_highlevel.py --mode teacher --skill moe --task s_avoid_basic \\
            --low_level_ckpt agents/fast_2000.pt \\
            --avoid_ckpt <AVOID_CKPT>
"""

import os
import sys
import time
import argparse
import math
import copy
import json
import types
import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Optional, Any
from collections import deque

# Use CLI flag to gate debug output before args are parsed.
DEBUG_MODE = "--debug" in sys.argv


def _debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 默认维度定义（实际训练时以环境观测维度为准）
STATE_DIM = 9   # [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
GOAL_DIM = 2    # [goal_x, goal_y] (relative goal_buf)
AFFORDANCE_CHANNELS = 3  # Default GT affordance layout in env
AVOID_LOCAL_MAP_CHANNELS = 2  # [occupancy, clearance_or_cost]
GOAL_TH_DEFAULT = 0.1
GATE_SWITCH_DY_DEFAULT = 0.2
SCENE_CURRIC_ITERS = 1000
EXPERT_INTERFACE_RATIO = 0.2
EXPERT_ALPHA_MIN = 0.0
EXPERT_ALPHA_SCHEDULE = "cosine"
EXPERT_BC_COEF = 2.0

# Task naming upgrade: canonical names use s_*; legacy hex_* stays compatible.
LEGACY_TASK_ALIASES = {
    "hex_s0_follow": "s_follow_basic",
    "hex_s1_large": "s_avoid_basic_large",
    "hex_s2": "s_cylinder",
    "hex_s2_large": "s_cylinder_large",
    "hex_s3": "s_narrow_passage",
    "hex_s3_large": "s_narrow_passage_large",
    "hex_s4": "s_step_field",
    "hex_s4_large": "s_step_field_large",
    "hex_s5": "s_dense_obstacles",
    "hex_s5_large": "s_dense_obstacles_large",
    "hex_s6": "s_ood_holdout",
    "hex_s6_large": "s_ood_holdout_large",
    "hex_mix_gate": "s_mixed_gate",
    "hex_mix_gate_large": "s_mixed_gate_large",
    "hex_calib": "s_calib",
    "hex_debug_plane": "s_debug_plane",
    "hex_debug_heightfield": "s_debug_heightfield",
}
S0_FOLLOW_TASK_NAME = "s_follow_basic"


def normalize_task_name(task_name: str) -> str:
    t = str(task_name).strip()
    if not t:
        return t
    return LEGACY_TASK_ALIASES.get(t, t)


def is_s0_follow_task_name(task_name: str) -> bool:
    return normalize_task_name(task_name) == S0_FOLLOW_TASK_NAME

# 延迟导入占位（便于静态检查）
task_registry: Any = None
TerrainAdaptivePlanner: Any = None
HighLevelPlanner: Any = None
LocomotionAdapter: Any = None
CmdVelExpert: Any = None
GatePolicy: Any = None
CommandPostProcessor: Any = None
AffordanceEstimator: Any = None
NavigationRewardFunction: Any = None
NavigationRewardConfig: Any = None
ActorCritic: Any = None


def difficulty_from_gap(aff_map: torch.Tensor) -> torch.Tensor:
    if aff_map.ndim != 4 or aff_map.size(1) < 2:
        return torch.zeros(aff_map.shape[0], device=aff_map.device)
    gap = torch.nan_to_num(aff_map[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    difficulty = 1.0 - gap.mean(dim=(1, 2))
    return torch.clamp(difficulty, 0.0, 1.0)


def build_avoid_local_map_2ch(
    aff_map: torch.Tensor,
    visible_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a compact avoid map: [occupancy, clearance/cost proxy]."""
    if aff_map.ndim != 4:
        raise ValueError(f"aff_map shape invalid: {tuple(aff_map.shape)}")
    vis = None
    if visible_mask is not None:
        vis = visible_mask
        if vis.ndim == 2:
            vis = vis.unsqueeze(0)
        if vis.ndim != 3:
            raise ValueError(f"visible_mask shape invalid: {tuple(vis.shape)}")
        if vis.shape[-2:] != aff_map.shape[-2:]:
            raise ValueError(
                f"visible_mask spatial shape {tuple(vis.shape[-2:])} does not match aff_map {tuple(aff_map.shape[-2:])}"
            )
        if vis.shape[0] == 1 and aff_map.shape[0] != 1:
            vis = vis.expand(aff_map.shape[0], -1, -1)
        elif vis.shape[0] != aff_map.shape[0]:
            raise ValueError(
                f"visible_mask batch {vis.shape[0]} does not match aff_map batch {aff_map.shape[0]}"
            )
        vis = vis.to(device=aff_map.device)
        vis_f = vis.float()
    if aff_map.size(1) == AVOID_LOCAL_MAP_CHANNELS:
        occ = torch.clamp(torch.nan_to_num(aff_map[:, 0], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        clearance = torch.clamp(torch.nan_to_num(aff_map[:, 1], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        if vis is not None:
            occ = occ * vis_f
            clearance = clearance * vis_f
        return torch.stack([occ, clearance], dim=1)

    occ = torch.nan_to_num(aff_map[:, 0], nan=0.0, posinf=1.0, neginf=0.0)
    occ = torch.clamp(occ, 0.0, 1.0)

    if aff_map.size(1) >= 2:
        free_score = torch.nan_to_num(aff_map[:, 1], nan=0.0, posinf=1.0, neginf=0.0)
    else:
        free_score = 1.0 - occ
    free_score = torch.clamp(free_score, 0.0, 1.0)

    # Smooth the free-space score so the second channel behaves more like
    # a local clearance / safety score instead of a brittle binary mask.
    clearance = F.avg_pool2d(free_score.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    clearance = torch.maximum(clearance, free_score)
    clearance = clearance * (1.0 - occ)
    if vis is not None:
        occ = occ * vis_f
        clearance = clearance * vis_f
    clearance = torch.clamp(clearance, 0.0, 1.0)

    return torch.stack([occ, clearance], dim=1)


def resize_affordance_map(
    aff_map: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    if aff_map.ndim != 4:
        raise ValueError(f"aff_map shape invalid: {tuple(aff_map.shape)}")
    target_size = int(target_size)
    if aff_map.shape[-2:] == (target_size, target_size):
        return aff_map
    return F.interpolate(
        aff_map,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def load_high_level_state_dict_compat(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    label: str = "policy",
) -> None:
    """Load old high-level checkpoints while mirroring actor trunk weights to critic trunk."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} state_dict must be dict, got {type(state_dict)}")

    state = dict(state_dict)
    model_state = model.state_dict()
    mirror_pairs = [
        ("affordance_encoder", "critic_affordance_encoder"),
        ("state_encoder", "critic_state_encoder"),
        ("goal_encoder", "critic_goal_encoder"),
        ("fusion", "critic_fusion"),
    ]
    for src_prefix, dst_prefix in mirror_pairs:
        src_token = src_prefix + "."
        dst_token = dst_prefix + "."
        for key, value in list(state.items()):
            if not key.startswith(src_token):
                continue
            dst_key = dst_token + key[len(src_token):]
            if dst_key in state or dst_key not in model_state:
                continue
            if not torch.is_tensor(value):
                continue
            if model_state[dst_key].shape == value.shape:
                state[dst_key] = value.clone()
                continue
            if (
                value.ndim == 2
                and model_state[dst_key].ndim == 2
                and model_state[dst_key].shape[0] == value.shape[0]
                and model_state[dst_key].shape[1] > value.shape[1]
            ):
                padded = model_state[dst_key].clone()
                padded[:, :value.shape[1]] = value
                state[dst_key] = padded

    # Allow loading old checkpoints when we append a few extra state features
    # (e.g., post-processor command history) to the policy state.
    state_linear_keys = (
        "state_encoder.mlp.0.weight",
        "critic_state_encoder.mlp.0.weight",
    )
    for key in state_linear_keys:
        if key not in state or key not in model_state:
            continue
        value = state[key]
        target = model_state[key]
        if (
            torch.is_tensor(value)
            and value.ndim == 2
            and target.ndim == 2
            and value.shape[0] == target.shape[0]
            and value.shape[1] < target.shape[1]
        ):
            padded = target.clone()
            padded[:, :value.shape[1]] = value
            state[key] = padded

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint incompatible: missing={missing}, unexpected={unexpected}"
        )
    if _module_has_non_finite(model):
        raise RuntimeError(f"{label} checkpoint has non-finite parameters after load")


def apply_goal_occlusion(
    goal: torch.Tensor,
    prev_goal: Optional[torch.Tensor],
    occlude_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prev_goal is None:
        prev_goal = goal.detach()
    occlude_mask = occlude_mask.view(-1, 1)
    masked_goal = torch.where(occlude_mask, prev_goal, goal)
    return masked_goal, masked_goal.detach()


def _get_effective_safe_free_dist(
    env,
    beta: Optional[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, float]:
    post = getattr(env, "post_processor", None)
    safe_default = float(getattr(post, "safe_distance", 0.25))
    free_default = float(getattr(post, "free_distance", 0.6))
    if beta is None or post is None or not hasattr(post, "get_effective_params"):
        return safe_default, free_default
    params = post.get_effective_params(torch.tensor(float(beta), device=device, dtype=dtype))
    safe_t = torch.as_tensor(params.get("safe_distance", safe_default), device=device, dtype=dtype).reshape(-1)
    free_t = torch.as_tensor(params.get("free_distance", free_default), device=device, dtype=dtype).reshape(-1)
    return float(safe_t[0].detach().cpu().item()), float(free_t[0].detach().cpu().item())


def get_moe_expert_state_inputs(state_tensor: torch.Tensor) -> torch.Tensor:
    # Standalone follow/avoid experts were trained with prev_gate_y fixed to 0.
    # Keep the same state contract when they are reused under MoE gating.
    if state_tensor.dim() != 2 or state_tensor.shape[1] <= 9:
        return state_tensor
    expert_state = state_tensor.clone()
    expert_state[:, 9] = 0.0
    return expert_state


def resolve_moe_gate_pcr(
    env,
    args: argparse.Namespace,
    aff_map: torch.Tensor,
    gate_y_raw: torch.Tensor,
    cmd_f: torch.Tensor,
    cmd_a: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    gate_y = gate_y_raw.clone()
    safe_d, free_d = _get_effective_safe_free_dist(
        env,
        getattr(args, "beta", None),
        device=gate_y_raw.device,
        dtype=gate_y_raw.dtype,
    )
    clearance_f = env._compute_clearance_along_cmd(aff_map, cmd_f[:, :2])
    clearance_a = env._compute_clearance_along_cmd(aff_map, cmd_a[:, :2])
    risk_f = env._risk_from_clearance(clearance_f, safe_d, free_d)
    risk_a = env._risk_from_clearance(clearance_a, safe_d, free_d)

    w_mode = str(getattr(args, "w_mode", "none")).lower()
    w_tau = max(float(getattr(args, "w_tau", 0.25)), 1e-6)
    if w_mode == "geom":
        w = torch.exp(-torch.clamp(clearance_f, min=0.0) / w_tau)
        w = torch.clamp(w, 0.0, 1.0)
    else:
        w = torch.zeros_like(gate_y_raw)

    clamp_enabled = bool(getattr(args, "gate_safe_clamp", False))
    if w_mode != "none" and bool(getattr(args, "w_disable_gate_safe_clamp", False)):
        clamp_enabled = False
    clamp_mask = torch.zeros_like(gate_y_raw, dtype=torch.bool)
    if clamp_enabled:
        safe_max = float(getattr(args, "gate_safe_max", 0.3))
        follow_is_riskier = clearance_f <= clearance_a
        clamp_mask = follow_is_riskier & (clearance_f < safe_d)
        gate_y = torch.where(
            clamp_mask,
            torch.minimum(gate_y, torch.full_like(gate_y, safe_max)),
            gate_y,
        )

    blend_mode = str(getattr(args, "w_blend_mode", "multiply")).lower()
    if w_mode == "none":
        y_eff = gate_y
    elif blend_mode == "mix":
        y_eff = torch.clamp(0.5 * gate_y + 0.5 * (1.0 - w), 0.0, 1.0)
    else:
        y_eff = torch.clamp(gate_y * (1.0 - w), 0.0, 1.0)

    cmd = y_eff.unsqueeze(-1) * cmd_f + (1.0 - y_eff.unsqueeze(-1)) * cmd_a
    return {
        "gate_y_raw": gate_y_raw,
        "gate_y": gate_y,
        "gate_y_safe": gate_y,
        "y_eff": y_eff,
        "w": w,
        "cmd": cmd,
        "cmd_f": cmd_f,
        "cmd_a": cmd_a,
        "clearance_F": clearance_f,
        "clearance_A": clearance_a,
        "risk_F": risk_f,
        "risk_A": risk_a,
        "post_safe_distance": torch.full_like(gate_y_raw, safe_d),
        "post_free_distance": torch.full_like(gate_y_raw, free_d),
        "gate_safe_clamp_mask": clamp_mask,
    }


def _get_expert_alpha(local_iteration: int, interface_iters: int) -> float:
    if interface_iters <= 0:
        return 0.0
    progress = min(float(max(local_iteration, 0)) / float(max(interface_iters, 1)), 1.0)
    if EXPERT_ALPHA_SCHEDULE == "cosine":
        alpha = EXPERT_ALPHA_MIN + 0.5 * (1.0 - EXPERT_ALPHA_MIN) * (1.0 + math.cos(math.pi * progress))
    else:
        alpha = EXPERT_ALPHA_MIN + (1.0 - EXPERT_ALPHA_MIN) * (1.0 - progress)
    return max(alpha, EXPERT_ALPHA_MIN)


def _module_has_non_finite(module: nn.Module) -> bool:
    for param in module.parameters():
        if not torch.isfinite(param).all():
            return True
    return False


def _sanitize_module_params(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            if not torch.isfinite(param).all():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)


def _tensor_min_max_text(t: torch.Tensor) -> str:
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return "all_non_finite"
    return f"min={finite.min().item():.6g}, max={finite.max().item():.6g}"


def _assert_finite_tensor(name: str, t: torch.Tensor) -> None:
    if torch.isfinite(t).all():
        return
    raise RuntimeError(f"[NonFinite] {name} has non-finite values ({_tensor_min_max_text(t)})")


def _sanitize_or_fail_action_tensor(
    name: str,
    t: torch.Tensor,
    *,
    allow_recovery: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bad_rows = ~_row_finite_mask(t)
    if not bool(bad_rows.any().item()):
        return t, bad_rows
    if not allow_recovery:
        raise RuntimeError(
            f"[NonFinite-FailFast] {name} has non-finite values ({_tensor_min_max_text(t)}). "
            "Use --allow_nonfinite_recovery to fallback to nan_to_num mode."
        )
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), bad_rows


def _row_finite_mask(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError("_row_finite_mask expects a tensor input.")
    if t.ndim == 0:
        return torch.isfinite(t).view(1)
    if t.ndim == 1:
        return torch.isfinite(t)
    return torch.isfinite(t).reshape(t.shape[0], -1).all(dim=1)


EXPERIMENT_META_REPLAY_KEYS = (
    "aff_stack",
    "beta",
    "cmd_slew_lin",
    "cmd_slew_ang",
    "cmd_safe_dist",
    "cmd_free_dist",
    "disable_risk_scale",
    "gate_use_difficulty",
    "gate_safe_clamp",
    "gate_safe_max",
    "w_mode",
    "w_tau",
    "w_blend_mode",
    "w_disable_gate_safe_clamp",
    "camera_enable",
    "camera_interval",
    "decimation",
)
RUNTIME_ABLATION_ARG_KEYS = (
    "beta",
    "w_mode",
    "w_tau",
    "w_blend_mode",
    "w_disable_gate_safe_clamp",
)
CHECKPOINT_CONTRACT_KEY_PATHS = (
    "aff_stack",
    "decimation",
    "low_level_ckpt",
    "observation_contract.affordance_map_size",
    "observation_contract.affordance_map_extent_m",
    "observation_contract.affordance_origin_mode",
    "observation_contract.camera_enable",
    "observation_contract.camera_interval",
    "observation_contract.camera_width",
    "observation_contract.camera_height",
    "observation_contract.camera_horizontal_fov_deg",
    "observation_contract.camera_vertical_fov_deg",
    "observation_contract.camera_near_m",
    "observation_contract.camera_far_m",
    "observation_contract.camera_fps_cfg",
)
RESOLVED_PROTOCOL_ARG_KEYS = (
    "task",
    "mode",
    "skill",
    "seed",
    "num_envs",
    "num_steps",
    "num_epochs",
    "mini_batch_size",
    "lr",
    "aff_stack",
    "beta",
    "gate_use_difficulty",
    "cmd_slew_lin",
    "cmd_slew_ang",
    "cmd_safe_dist",
    "cmd_free_dist",
    "disable_risk_scale",
    "w_mode",
    "w_tau",
    "w_blend_mode",
    "w_disable_gate_safe_clamp",
    "camera_enable",
    "camera_interval",
    "decimation",
)
VISION_NATIVE_OUTPUT_SIZE = 16


def _coerce_meta_value_like(raw_value: Any, template_value: Any) -> Any:
    if raw_value is None:
        return None
    if isinstance(template_value, bool):
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw_value)
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw_value)
    if isinstance(template_value, float):
        return float(raw_value)
    return raw_value


def apply_experiment_meta_to_args(
    args: argparse.Namespace,
    ckpt_meta: Optional[Dict[str, Any]],
    *,
    context: str,
    overwrite: bool = True,
) -> None:
    if not isinstance(ckpt_meta, dict):
        return
    applied = []
    for key in EXPERIMENT_META_REPLAY_KEYS:
        if key not in ckpt_meta:
            continue
        raw_value = ckpt_meta.get(key, None)
        if raw_value is None:
            continue
        current_value = getattr(args, key, None)
        new_value = _coerce_meta_value_like(raw_value, current_value)
        if (not overwrite) and getattr(args, key, None) is not None:
            continue
        if getattr(args, key, None) != new_value:
            setattr(args, key, new_value)
            applied.append(f"{key}={new_value}")
    if applied:
        print(f"[{context}] replay checkpoint contract: " + ", ".join(applied))


def capture_cli_explicit_arg_values(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    argv: Optional[Any] = None,
) -> Dict[str, Any]:
    if argv is None:
        argv = sys.argv[1:]
    option_to_action: Dict[str, argparse.Action] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_action[option] = action
    explicit_values: Dict[str, Any] = {}
    for token in argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_to_action.get(option, None)
        if action is None:
            continue
        explicit_values[action.dest] = getattr(args, action.dest, None)
    setattr(args, "_cli_explicit_arg_values", dict(explicit_values))
    runtime_overrides = {
        key: explicit_values[key]
        for key in RUNTIME_ABLATION_ARG_KEYS
        if key in explicit_values
    }
    setattr(args, "_runtime_ablation_cli_overrides", runtime_overrides)
    return explicit_values


def apply_runtime_ablation_cli_overrides(
    args: argparse.Namespace,
    ckpt_meta: Optional[Dict[str, Any]],
    *,
    context: str,
) -> None:
    explicit_overrides = dict(getattr(args, "_runtime_ablation_cli_overrides", {}) or {})
    resolved_lines = []
    for key in RUNTIME_ABLATION_ARG_KEYS:
        ckpt_value = ckpt_meta.get(key, None) if isinstance(ckpt_meta, dict) else None
        cli_explicit = key in explicit_overrides
        if cli_explicit:
            setattr(args, key, explicit_overrides[key])
        final_value = getattr(args, key, None)
        if cli_explicit or ckpt_value is not None:
            cli_value = explicit_overrides[key] if cli_explicit else "<unset>"
            resolved_lines.append(
                f"{key}: ckpt={ckpt_value}, cli={cli_value}, final={final_value}"
            )
    if resolved_lines:
        print(f"[{context}] resolved runtime ablation config:")
        for line in resolved_lines:
            print(f"[{context}]   {line}")


def apply_observation_contract_to_env_cfg(
    env_cfg,
    ckpt_meta: Optional[Dict[str, Any]],
    *,
    context: str,
) -> None:
    if not isinstance(ckpt_meta, dict):
        return
    contract = ckpt_meta.get("observation_contract", None)
    if not isinstance(contract, dict):
        return
    cam_cfg = getattr(getattr(env_cfg, "sensor", None), "depth_camera", None)
    nav_cfg = getattr(env_cfg, "navigation", None)
    terrain_cfg = getattr(env_cfg, "terrain", None)
    applied = []
    if cam_cfg is not None:
        camera_fields = {
            "camera_enable": "enable",
            "camera_width": "width",
            "camera_height": "height",
            "camera_horizontal_fov_deg": "horizontal_fov",
            "camera_vertical_fov_deg": "vertical_fov",
            "camera_near_m": "near_clip",
            "camera_far_m": "far_clip",
            "camera_fps_cfg": "fps",
            "camera_interval": "capture_interval",
        }
        for meta_key, attr_name in camera_fields.items():
            if meta_key not in contract or contract[meta_key] is None:
                continue
            current_value = getattr(cam_cfg, attr_name, None)
            new_value = _coerce_meta_value_like(contract[meta_key], current_value)
            if current_value != new_value:
                setattr(cam_cfg, attr_name, new_value)
                applied.append(f"{attr_name}={new_value}")
    map_size = contract.get("affordance_map_size", None)
    if map_size is not None:
        map_size = int(map_size)
        if nav_cfg is not None and getattr(nav_cfg, "affordance_grid_size", None) != map_size:
            nav_cfg.affordance_grid_size = map_size
            applied.append(f"navigation.affordance_grid_size={map_size}")
        if terrain_cfg is not None and hasattr(terrain_cfg, "affordance_grid_size"):
            if getattr(terrain_cfg, "affordance_grid_size", None) != map_size:
                terrain_cfg.affordance_grid_size = map_size
                applied.append(f"terrain.affordance_grid_size={map_size}")
    if applied:
        print(f"[{context}] replay observation contract: " + ", ".join(applied))


def get_vision_native_output_size() -> int:
    return int(VISION_NATIVE_OUTPUT_SIZE)


def _get_nested_meta_value(meta: Optional[Dict[str, Any]], key_path: str) -> Any:
    if not isinstance(meta, dict):
        return None
    current: Any = meta
    for key in str(key_path).split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current.get(key, None)
    return current


def _normalize_contract_compare_value(key_path: str, value: Any) -> Any:
    if isinstance(value, str) and key_path.endswith("_ckpt") and value.strip():
        return os.path.abspath(value)
    return value


def collect_runtime_observation_contract(
    args: argparse.Namespace,
    env,
) -> Dict[str, Any]:
    cam_cfg = getattr(getattr(env, "env", None), "camera_cfg", None)
    low_level_dt = float(getattr(getattr(env, "env", None), "dt", 0.0))
    capture_interval = 1
    if cam_cfg is not None:
        capture_interval = int(getattr(cam_cfg, "capture_interval", 1))
    elif getattr(args, "camera_interval", None) is not None:
        capture_interval = int(getattr(args, "camera_interval", 1))
    capture_interval = max(1, int(capture_interval))
    effective_depth_hz = None
    if low_level_dt > 0.0:
        effective_depth_hz = 1.0 / (low_level_dt * float(capture_interval))
    high_level_dt = float(getattr(env, "high_level_dt", 0.0))
    return {
        "use_avoid_local_map": bool(getattr(args, "skill", "follow") in ("avoid", "moe")),
        "affordance_map_size": int(getattr(env, "affordance_map_size", get_vision_native_output_size())),
        "affordance_map_extent_m": float(getattr(env, "affordance_map_extent", 0.0)),
        "affordance_origin_mode": str(getattr(env, "affordance_origin_mode", "base_center")),
        "camera_enable": bool(getattr(args, "camera_enable", False)),
        "camera_interval": int(capture_interval),
        "camera_width": None if cam_cfg is None else int(getattr(cam_cfg, "width", 0)),
        "camera_height": None if cam_cfg is None else int(getattr(cam_cfg, "height", 0)),
        "camera_horizontal_fov_deg": None if cam_cfg is None else float(getattr(cam_cfg, "horizontal_fov", 0.0)),
        "camera_vertical_fov_deg": None if cam_cfg is None else float(getattr(cam_cfg, "vertical_fov", 0.0)),
        "camera_near_m": None if cam_cfg is None else float(getattr(cam_cfg, "near_clip", 0.0)),
        "camera_far_m": None if cam_cfg is None else float(getattr(cam_cfg, "far_clip", 0.0)),
        "camera_fps_cfg": None if cam_cfg is None else int(getattr(cam_cfg, "fps", 0)),
        "camera_effective_rate_hz": effective_depth_hz,
        "low_level_dt_s": low_level_dt,
        "high_level_dt_s": high_level_dt,
        "high_level_rate_hz": None if high_level_dt <= 0.0 else (1.0 / high_level_dt),
    }


def build_runtime_contract_meta(
    args: argparse.Namespace,
    env,
) -> Dict[str, Any]:
    low_level_ckpt = getattr(args, "low_level_ckpt", None)
    return {
        "aff_stack": int(getattr(args, "aff_stack", 1)),
        "decimation": None if getattr(args, "decimation", None) is None else int(getattr(args, "decimation")),
        "low_level_ckpt": None if not low_level_ckpt else os.path.abspath(str(low_level_ckpt)),
        "observation_contract": collect_runtime_observation_contract(args, env),
    }


def validate_checkpoint_contract_compatibility(
    reference_meta: Optional[Dict[str, Any]],
    candidate_meta: Optional[Dict[str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
    strict: bool = True,
    key_paths: Tuple[str, ...] = CHECKPOINT_CONTRACT_KEY_PATHS,
) -> None:
    if not isinstance(reference_meta, dict) or not isinstance(candidate_meta, dict):
        return
    mismatch_msgs = []
    for key_path in key_paths:
        ref_value = _get_nested_meta_value(reference_meta, key_path)
        cand_value = _get_nested_meta_value(candidate_meta, key_path)
        if ref_value is None or cand_value is None:
            continue
        ref_value = _normalize_contract_compare_value(key_path, ref_value)
        cand_value = _normalize_contract_compare_value(key_path, cand_value)
        if ref_value != cand_value:
            mismatch_msgs.append(f"{key_path}: {reference_name}={ref_value}, {candidate_name}={cand_value}")
    if mismatch_msgs:
        msg = (
            f"{candidate_name} 与 {reference_name} 的训练口径不一致，继续运行会污染实验归因："
            + " | ".join(mismatch_msgs)
        )
        if strict:
            raise ValueError(msg)
        print(f"[Warn] {msg}")


def validate_vision_runtime_contract(
    args: argparse.Namespace,
    env,
    *,
    source_name: str,
    ckpt_meta: Optional[Dict[str, Any]] = None,
    strict_meta: bool = True,
) -> Dict[str, Any]:
    runtime_meta = build_runtime_contract_meta(args, env)
    runtime_contract = runtime_meta["observation_contract"]
    native_size = get_vision_native_output_size()
    runtime_map_size = int(runtime_contract.get("affordance_map_size", native_size))
    if runtime_map_size != native_size:
        print(
            f"[{source_name}] vision native output_size={native_size}, "
            f"runtime affordance_map_size={runtime_map_size}; prediction will be resized before use."
        )
    if isinstance(ckpt_meta, dict):
        validate_checkpoint_contract_compatibility(
            runtime_meta,
            ckpt_meta,
            reference_name="current runtime",
            candidate_name=source_name,
            strict=strict_meta,
        )
    else:
        print(f"[Warn] {source_name} 未包含 experiment_meta；只能校验当前 runtime 与原生输出尺寸。")
    return {
        "vision_native_output_size": native_size,
        "runtime_affordance_map_size": runtime_map_size,
        "resize_before_use": bool(runtime_map_size != native_size),
    }


def build_resolved_protocol(
    args: argparse.Namespace,
    env,
    *,
    primary_ckpt_path: Optional[str],
    primary_meta: Optional[Dict[str, Any]],
    aux_sources: Optional[Dict[str, Dict[str, Any]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_args = {}
    for key in RESOLVED_PROTOCOL_ARG_KEYS:
        if hasattr(args, key):
            runtime_args[key] = getattr(args, key)
    protocol = {
        "sim2real_sensor_target": "Intel RealSense D435i",
        "runtime_args": runtime_args,
        "unknown_cli_args": list(getattr(args, "_unknown_cli", [])),
        "runtime_contract": build_runtime_contract_meta(args, env),
        "primary_checkpoint": {
            "path": None if not primary_ckpt_path else os.path.abspath(str(primary_ckpt_path)),
            "experiment_meta": primary_meta,
        },
        "aux_checkpoints": aux_sources or {},
        "vision_native_output_size": get_vision_native_output_size(),
    }
    if extra:
        protocol.update(extra)
    return protocol


def write_resolved_protocol_json(path: str, protocol: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)


def _actor_grad_has_non_finite(module: nn.Module) -> bool:
    for name, param in module.named_parameters():
        if "value_head" in name:
            continue
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True
    return False


def _get_cuda_device_index(device: torch.device) -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    if not isinstance(device, torch.device) or device.type != "cuda":
        return None
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _cuda_mem_stats_text(device_index: Optional[int]) -> str:
    if device_index is None or not torch.cuda.is_available():
        return "cuda_mem=n/a"
    allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
    return (
        f"cuda_mem[alloc/res/max]={allocated:.1f}/{reserved:.1f}/{max_allocated:.1f} MiB"
    )


def import_modules():
    """延迟导入，带容错处理"""
    global task_registry, TerrainAdaptivePlanner, HighLevelPlanner, LocomotionAdapter
    global AffordanceEstimator, NavigationRewardFunction, NavigationRewardConfig
    global ActorCritic
    
    try:
        # 导入 legged_gym 注册环境
        import legged_gym.envs
        from legged_gym.utils import get_args, task_registry as tr
        task_registry = tr
        _debug_print("[Init] ✓ legged_gym 导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 'legged_gym': {e}")
        print("请确认 legged_gym 已安装并添加到 PYTHONPATH。")
        sys.exit(1)

    try:
        from rsl_rl.algorithms.high_level_planner import TerrainAdaptivePlanner as TAP
        from rsl_rl.algorithms.high_level_planner import LocomotionAdapter as LA
        from rsl_rl.algorithms.high_level_planner import CmdVelExpert as CVE
        from rsl_rl.algorithms.high_level_planner import GatePolicy as GP
        from rsl_rl.algorithms.high_level_planner import CommandPostProcessor as CPP
        from rsl_rl.modules import ActorCritic as AC
        
        # 赋值给全局变量
        # Keep a clearer name for ground tasks; same implementation.
        globals()['TerrainAdaptivePlanner'] = TAP
        globals()['HighLevelPlanner'] = TAP
        globals()['LocomotionAdapter'] = LA
        globals()['CmdVelExpert'] = CVE
        globals()['GatePolicy'] = GP
        globals()['CommandPostProcessor'] = CPP
        globals()['ActorCritic'] = AC
        _debug_print("[Init] ✓ rsl_rl 模块导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 rsl_rl 模块: {e}")
        sys.exit(1)

    try:
        from legged_gym.envs.hex_v4.affordance_estimator import AffordanceEstimator as AE
        globals()['AffordanceEstimator'] = AE
        _debug_print("[Init] ✓ AffordanceEstimator 导入成功")
    except ImportError as e:
        print(f"[Warning] AffordanceEstimator 导入失败: {e}")
        globals()['AffordanceEstimator'] = None

    try:
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardFunction as NRF
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardConfig as NRC
        globals()['NavigationRewardFunction'] = NRF
        globals()['NavigationRewardConfig'] = NRC
        _debug_print("[Init] ✓ NavigationReward 导入成功")
    except ImportError as e:
        print(f"[Warning] NavigationReward 导入失败: {e}")
        globals()['NavigationRewardFunction'] = None
        globals()['NavigationRewardConfig'] = None


class RolloutBuffer:
    """
    PPO Rollout Buffer - 存储一个 rollout 周期内的所有数据
    """
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        state_dim: int,
        goal_dim: int,
        aff_map_shape: Tuple[int, int, int],
        action_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0
        self.action_dim = action_dim

        # 观测（仅保留 PPO 更新所需张量）
        self.states = torch.zeros(num_steps, num_envs, state_dim, device=device)
        self.goals = torch.zeros(num_steps, num_envs, goal_dim, device=device)
        # aff_map_shape 为堆叠后的通道数 (aff_channels, H, W)
        self.aff_maps = torch.zeros(num_steps, num_envs, *aff_map_shape, device=device)
        self.difficulties = torch.zeros(num_steps, num_envs, device=device)
        self.critic_aff_maps = torch.zeros(num_steps, num_envs, *aff_map_shape, device=device)
        self.critic_difficulties = torch.zeros(num_steps, num_envs, device=device)

        # 动作
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        
        # PPO 核心数据
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.action_valid = torch.ones(num_steps, num_envs, device=device, dtype=torch.bool)
        
        # 蒸馏目标 (Student 模式)
        self.teacher_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        # EGPO 目标 (S0 follow expert 命令)
        self.expert_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)

    def add(
        self,
        state,
        goal,
        aff_map,
        difficulty,
        critic_aff_map,
        critic_difficulty,
        action,
        log_prob,
        value,
        reward,
        done,
        action_valid=None,
        teacher_action=None,
        expert_action=None,
    ):
        """添加一步数据"""
        self.states[self.step] = state
        self.goals[self.step] = goal
        self.aff_maps[self.step] = aff_map
        self.difficulties[self.step] = difficulty
        self.critic_aff_maps[self.step] = critic_aff_map
        self.critic_difficulties[self.step] = critic_difficulty
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.squeeze(-1)
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        if action_valid is None:
            self.action_valid[self.step].fill_(True)
        else:
            self.action_valid[self.step] = action_valid.bool()
        
        if teacher_action is not None:
            self.teacher_actions[self.step] = teacher_action
        if expert_action is not None:
            self.expert_actions[self.step] = expert_action
        
        self.step += 1

    def compute_returns(self, next_value: torch.Tensor, gamma: float = 0.99, 
                        gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 和 Returns"""
        advantages = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        
        for t in reversed(range(self.num_steps)):
            current_valid = self.action_valid[t]
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value.squeeze(-1)
                next_valid = torch.ones_like(current_valid, dtype=torch.bool)
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
                next_valid = self.action_valid[t + 1]

            continuation = next_non_terminal * next_valid.float()
            delta = self.rewards[t] + gamma * next_values * continuation - self.values[t]
            last_gae = delta + gamma * gae_lambda * continuation * last_gae
            last_gae = torch.where(current_valid, last_gae, torch.zeros_like(last_gae))
            advantages[t] = torch.where(current_valid, last_gae, torch.zeros_like(last_gae))
        
        returns = advantages + self.values
        return returns, advantages

    def reset(self):
        """重置 buffer"""
        self.step = 0
        self.states.zero_()
        self.goals.zero_()
        self.aff_maps.zero_()
        self.difficulties.zero_()
        self.critic_aff_maps.zero_()
        self.critic_difficulties.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.action_valid.fill_(True)
        self.teacher_actions.zero_()
        self.expert_actions.zero_()


# V5 核心环境包装器 (The Hierarchical Environment Wrapper)
class HierarchicalHexapodEnv:
    """
    V5 分层环境包装器
    
    职责:
    1. 托管 Isaac Gym 底层环境 (s_avoid_basic/s_cylinder)
    2. 托管 Low-Level Controller (冻结的底层策略)
    3. 托管 Command Post-Processor (限幅/滤波/风险钳制)
    4. 托管 Reward Function (V5 gate_smooth + risk_barrier)
    5. 提供 Teacher/Student 两种数据流
    
    状态空间 (V5):
        robot_state: (N, 9) = [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
        goal: (N, 2) = [goal_x, goal_y]
        affordance: (N, 3, 16, 16) = [occupancy, passable_gap, low_obstacle]
        terrain_difficulty: (N,)
    """
    
    def __init__(self, args, device: torch.device, env_cfg=None, train_cfg=None):
        self.args = args
        self.device = device
        self.mode = args.mode
        self.debug = bool(getattr(args, "debug", False))
        if getattr(self.args, "mode", "teacher") == "student":
            setattr(self.args, "camera_enable", True)
        beta_arg = getattr(args, "beta", None)
        self.beta_override = None if beta_arg is None else float(beta_arg)

        from legged_gym.utils import get_args as get_isaac_args
        isaac_args = get_isaac_args()
        for key, value in vars(isaac_args).items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        
        # 初始化 Isaac Gym 环境
        if self.debug:
            print(f"[Env] 创建 Isaac Gym 环境: {args.task}")
        if env_cfg is None:
            env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        # Isolate mutable runtime overrides to avoid cross-task config contamination.
        env_cfg = copy.deepcopy(env_cfg)
        train_cfg = copy.deepcopy(train_cfg) if train_cfg is not None else None
        if getattr(args, "seed", None) is not None:
            env_cfg.seed = int(args.seed)
        if getattr(args, "skill", "follow") == "follow" and hasattr(env_cfg, "navigation"):
            if hasattr(env_cfg.navigation, "follow_goal_force_blocking_line"):
                env_cfg.navigation.goal_force_blocking_line = bool(
                    getattr(env_cfg.navigation, "follow_goal_force_blocking_line")
                )
            else:
                env_cfg.navigation.goal_force_blocking_line = False
            if hasattr(env_cfg.navigation, "follow_goal_force_blocking_prob"):
                env_cfg.navigation.goal_force_blocking_prob = float(
                    getattr(env_cfg.navigation, "follow_goal_force_blocking_prob")
                )
            else:
                env_cfg.navigation.goal_force_blocking_prob = 0.0

        if getattr(args, "camera_enable", False) and hasattr(env_cfg, "sensor"):
            if hasattr(env_cfg.sensor, "depth_camera"):
                env_cfg.sensor.depth_camera.enable = True
                if getattr(args, "camera_interval", None) is not None:
                    env_cfg.sensor.depth_camera.capture_interval = args.camera_interval
        if hasattr(env_cfg, "navigation") and hasattr(env_cfg.navigation, "reward_cfg"):
            reward_goal_th = float(env_cfg.navigation.reward_cfg.get("goal_reach_threshold", GOAL_TH_DEFAULT))
            env_cfg.navigation.goal_reached_threshold = reward_goal_th
            if getattr(env_cfg.navigation, "resample_on_reach", False):
                if abs(env_cfg.navigation.goal_reached_threshold - reward_goal_th) > 1e-6:
                    raise RuntimeError(
                        "goal_reached_threshold must match reward_cfg.goal_reach_threshold when resample_on_reach=True"
                    )

        # 覆盖配置以适配高层训练
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.num_envs = self.env.num_envs
        self.max_episode_length_low = int(self.env.max_episode_length)
        self.max_episode_length = self.max_episode_length_low
        if hasattr(self.env, "debug_viz"):
            self.env.debug_viz = bool(getattr(args, "debug", False))

        cam_cfg = getattr(getattr(env_cfg, "sensor", None), "depth_camera", None)
        nav_cfg = getattr(env_cfg, "navigation", None)
        map_size = getattr(nav_cfg, "affordance_grid_size", None)
        if map_size is None:
            map_size = getattr(env_cfg.terrain, "affordance_grid_size", 16)
        map_size = int(map_size)
        cell_size = getattr(nav_cfg, "affordance_cell_size", None)
        if cell_size is None:
            cell_size = getattr(env_cfg.terrain, "affordance_cell_size", None)
        if cell_size is not None:
            map_extent = float(map_size * cell_size)
        elif cam_cfg is not None and hasattr(cam_cfg, "far_clip"):
            map_extent = float(cam_cfg.far_clip)
        else:
            map_extent = 5.0
        self.affordance_cell_size = float(cell_size) if cell_size is not None else (map_extent / map_size)
        self.affordance_map_size = map_size
        self.affordance_map_extent = map_extent
        self.affordance_origin_mode = "base_center"
        self.affordance_origin_local_xy = torch.zeros(2, device=self.device, dtype=torch.float32)
        clearance = getattr(env_cfg.terrain, "fixed_layout_robot_clearance", None)
        if clearance is None:
            body_shape = getattr(getattr(env_cfg, "asset", None), "body_shape", None)
            if body_shape is not None:
                clearance = math.hypot(float(body_shape.x), float(body_shape.y)) + 0.05
            else:
                clearance = 0.27
        self.affordance_clearance = float(clearance)
        self.affordance_clearance_free = self.affordance_clearance + 0.3
        self.affordance_difficulty_radius = float(
            getattr(env_cfg.navigation, "difficulty_radius_m", 2.0)
        )
        self.affordance_dist_map = self._build_affordance_dist_map()
        crossable_height = getattr(env_cfg.navigation, "crossable_height_max", None)
        if crossable_height is None:
            crossable_height = getattr(env_cfg.navigation, "goal_obstacle_height_threshold", 0.2)
        self.affordance_blocking_height = float(crossable_height)
        self.affordance_crossable_height = float(crossable_height)
        body_shape = getattr(getattr(env_cfg, "asset", None), "body_shape", None)
        if body_shape is not None:
            self.body_width = 2.0 * float(body_shape.y)
        else:
            self.body_width = 0.44
        width_margin = float(getattr(env_cfg.navigation, "crossable_width_margin", 0.0))
        self.crossable_width = self.body_width + width_margin
        self.crossable_sector_deg = float(getattr(env_cfg.navigation, "crossable_sector_deg", 60.0))
        self.camera_fov_rad = None
        self.camera_vertical_fov_rad = None
        self.camera_bearing_rad = 0.0
        self.camera_tilt_down_rad = 0.0
        base_height_m = float(getattr(getattr(env_cfg, "init_state", None), "pos", [0.0, 0.0, 0.1])[2])
        self.camera_height_m = max(1e-3, base_height_m)
        self.camera_near = 0.0
        self.camera_far = self.affordance_map_extent
        use_camera_origin = (
            str(getattr(args, "skill", "")).lower() == "avoid"
            or str(getattr(getattr(env_cfg, "terrain", None), "terrain_type", "")).lower() == "s_avoid_basic"
        )
        if cam_cfg is not None:
            self.camera_fov_rad = math.radians(float(getattr(cam_cfg, "horizontal_fov", 0.0)))
            self.camera_vertical_fov_rad = math.radians(float(getattr(cam_cfg, "vertical_fov", 0.0)))
            self.camera_near = float(getattr(cam_cfg, "near_clip", 0.0))
            self.camera_far = float(getattr(cam_cfg, "far_clip", self.affordance_map_extent))
            yaw_deg = float(getattr(cam_cfg, "yaw_deg", 0.0))
            # Convert camera yaw to bearing_y convention (0 means +Y forward).
            self.camera_bearing_rad = math.radians(yaw_deg) - 0.5 * math.pi
            cam_pos = getattr(cam_cfg, "position", None)
            if cam_pos is not None and len(cam_pos) >= 3:
                self.camera_height_m = max(1e-3, base_height_m + float(cam_pos[2]))
            tilt_down_deg = float(
                getattr(
                    cam_cfg,
                    "tilt_down_deg",
                    getattr(cam_cfg, "roll_deg", getattr(cam_cfg, "pitch_deg", 0.0)),
                )
            )
            self.camera_tilt_down_rad = math.radians(tilt_down_deg)
            if use_camera_origin and cam_pos is not None and len(cam_pos) >= 2:
                self.affordance_origin_mode = "camera_mount"
                self.affordance_origin_local_xy = torch.tensor(
                    [float(cam_pos[0]), float(cam_pos[1])],
                    device=self.device,
                    dtype=torch.float32,
                )
            if cam_pos is not None and len(cam_pos) >= 3:
                base_height = float(getattr(getattr(env_cfg, "init_state", None), "pos", [0.0, 0.0, 0.1])[2])
                self.camera_height_m = max(1e-3, base_height + float(cam_pos[2]))
        # S0 moving-target following: keep target centered using a tighter FOV window.
        self.moving_target_enable = bool(getattr(nav_cfg, "moving_target_enable", False)) if nav_cfg is not None else False
        self.target_fov_soft_scale = float(getattr(nav_cfg, "target_fov_soft_scale", 1.0)) if nav_cfg is not None else 1.0
        self.target_fov_hard_scale = float(getattr(nav_cfg, "target_fov_hard_scale", 1.0)) if nav_cfg is not None else 1.0
        self.target_lost_k = int(getattr(nav_cfg, "target_lost_k", 0)) if nav_cfg is not None else 0
        self.target_center_scale = float(getattr(nav_cfg, "target_center_scale", 0.0)) if nav_cfg is not None else 0.0
        self.target_visible_scale = float(getattr(nav_cfg, "target_visible_scale", 0.0)) if nav_cfg is not None else 0.0
        (self.affordance_x_map,
         self.affordance_y_map,
         self.affordance_bearing_map,
         self.affordance_visible_mask) = self._build_affordance_geometry()
        terrain_type = str(getattr(getattr(env_cfg, "terrain", None), "terrain_type", "")).lower()
        terrain_cfg = getattr(env_cfg, "terrain", None)
        self.use_actor_only_gt_affordance = bool(getattr(terrain_cfg, "avoid_gt_actor_only", False))
        self.require_actor_only_gt_affordance = bool(getattr(terrain_cfg, "avoid_gt_require_scene", False))
        if terrain_type == "s_avoid_basic":
            self.use_actor_only_gt_affordance = True
            self.require_actor_only_gt_affordance = True
        if self.use_actor_only_gt_affordance:
            print("[Env] actor-only GT affordance enabled: scene actors only, no heightfield fallback.")
        if is_s0_follow_task_name(str(getattr(args, "task", "")).lower()):
            if terrain_type not in ("s0_follow_plane", "s0"):
                raise RuntimeError(
                    "s_follow_basic requires terrain_type to be 's0_follow_plane' (or alias 's0'). "
                    f"got terrain_type='{terrain_type or 'unset'}'"
                )
        self.is_s0_follow_task = is_s0_follow_task_name(str(getattr(args, "task", "")).lower()) or (terrain_type in ("s0_follow_plane", "s0"))
        self.scene_difficulty_pending = 0.0
        self.scene_difficulty_override = torch.zeros(
            self.num_envs, device=self.env.device, dtype=torch.float32
        )
        self.env.scene_difficulty_override = self.scene_difficulty_override
        self.s0_follow_d_des = float(getattr(nav_cfg, "follow_distance_desired", 1.0)) if nav_cfg is not None else 1.0
        follow_min = getattr(nav_cfg, "follow_distance_min", None) if nav_cfg is not None else None
        follow_max = getattr(nav_cfg, "follow_distance_max", None) if nav_cfg is not None else None
        if follow_min is None or follow_max is None:
            sigma = float(getattr(nav_cfg, "follow_distance_sigma", 0.25)) if nav_cfg is not None else 0.25
            self.s0_follow_d_min = self.s0_follow_d_des - sigma
            self.s0_follow_d_max = self.s0_follow_d_des + sigma
        else:
            self.s0_follow_d_min = float(min(follow_min, follow_max))
            self.s0_follow_d_max = float(max(follow_min, follow_max))
        self.s0_follow_ahead_margin = 0.2
        self.s0_follow_success_bonus = 10.0
        self.s0_follow_dir_v_eps = 0.05
        success_time_default = float(getattr(env_cfg.env, "episode_length_s", 10.0))
        self.s0_follow_success_time_s = float(
            getattr(nav_cfg, "follow_success_time_s", success_time_default)
        ) if nav_cfg is not None else success_time_default

        if hasattr(self.env, "_resample_commands"):
            def _no_resample(self, env_ids):
                return
            self.env._resample_commands = types.MethodType(_no_resample, self.env)
        
        # 加载 Low-Level Controller
        self._load_low_level_policy(args.low_level_ckpt)
        
        # 初始化 Command Post-Processor (V5)
        max_lin_cmd = float(getattr(nav_cfg, "max_lin_vel_command", 0.8))
        max_ang_cmd = float(getattr(nav_cfg, "max_ang_vel_command", 1.5))
        slew_lin = float(getattr(args, "cmd_slew_lin", 0.2))
        slew_ang = float(getattr(args, "cmd_slew_ang", 0.4))
        safe_dist_arg = getattr(args, "cmd_safe_dist", None)
        free_dist_arg = getattr(args, "cmd_free_dist", None)
        safe_dist = self.affordance_clearance if safe_dist_arg is None else float(safe_dist_arg)
        free_dist = self.affordance_clearance_free if free_dist_arg is None else float(free_dist_arg)
        self.post_processor = CommandPostProcessor(
            max_cmd=(max_lin_cmd, max_lin_cmd, max_ang_cmd),
            max_delta=(slew_lin, slew_lin, slew_ang),
            safe_distance=safe_dist,
            free_distance=free_dist,
            enable_risk_scale=not bool(getattr(args, "disable_risk_scale", False)),
        )
        self.post_processor.reset(self.num_envs, device)
        
        # 初始化 Reward Function
        if NavigationRewardConfig is not None:
            reward_defaults = NavigationRewardConfig()
            reward_kwargs = {k: getattr(reward_defaults, k) for k in reward_defaults.__dict__.keys()}
            reward_cfg_dict = dict(getattr(nav_cfg, "reward_cfg")) if (nav_cfg is not None and hasattr(nav_cfg, "reward_cfg")) else {}
            self.reward_cfg_raw = reward_cfg_dict
            self.terminal_fail_penalty = float(
                self.reward_cfg_raw.get(
                    "terminal_fail_penalty",
                    reward_kwargs.get("collision_penalty", -10.0),
                )
            )
            if nav_cfg is not None and hasattr(nav_cfg, "reward_cfg"):
                for key, value in self.reward_cfg_raw.items():
                    if key not in reward_kwargs:
                        continue
                    if value is not None:
                        reward_kwargs[key] = value
            if reward_kwargs.get("risk_barrier_safe") is None:
                reward_kwargs["risk_barrier_safe"] = self.affordance_clearance
            if reward_kwargs.get("risk_barrier_free") is None:
                reward_kwargs["risk_barrier_free"] = self.affordance_clearance_free
            self.reward_cfg = NavigationRewardConfig(**reward_kwargs)
            self.reward_func = NavigationRewardFunction(self.reward_cfg)
        else:
            self.reward_func = None
            self.reward_cfg_raw = {}
            self.terminal_fail_penalty = -10.0
            print("[Warning] 使用环境原生奖励")
        
        # 状态缓冲区
        self.prev_robot_pos = torch.zeros(self.num_envs, 3, device=device)
        self.prev_gate_y = torch.zeros(self.num_envs, device=device)
        self.reach_given = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.goal_change_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.prev_goal_world = None
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.target_lost_steps = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.stable_follow_steps = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.episode_return_buf = torch.zeros(self.num_envs, device=device)
        self.episode_len_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.clearance_override = None
        self.clearance_affordance_override = None
        self.reward_affordance_override = None
        self.last_obs = None
        self.forced_forward_speed = torch.zeros(self.num_envs, device=device)
        self.prev_nearest_obs_dist = torch.zeros(self.num_envs, device=device)
        self.prev_nearest_obs_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.prev_forward_clearance = torch.zeros(self.num_envs, device=device)
        self.prev_forward_clearance_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.prev_align_center_abs = torch.zeros(self.num_envs, device=device)
        self.prev_align_center_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.prev_cmd_pred_x = torch.zeros(self.num_envs, device=device)
        self.prev_cmd_exec_mean = torch.zeros(self.num_envs, 3, device=device)
        self.rotate_only_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        
        # 频率控制 (High-Level 10Hz, Low-Level 50Hz)
        self.decimation = getattr(args, 'decimation', 5)
        self.high_level_dt = float(self.env.dt) * float(self.decimation)
        self.max_episode_length = max(1, int(np.ceil(self.max_episode_length_low / float(self.decimation))))
        self.max_episode_length_s = float(self.max_episode_length * self.high_level_dt)
        self.no_episode_timeout = bool(getattr(env_cfg.env, "no_episode_timeout", False))
        self.s0_follow_steps_success = max(
            1, int(self.s0_follow_success_time_s / max(1e-6, self.high_level_dt))
        )
        
        if self.debug:
            print(
                f"[Env] 初始化完成: {self.num_envs} envs, decimation={self.decimation}, "
                f"episode_steps(HL)={self.max_episode_length}, episode_s={self.max_episode_length_s:.2f}"
            )

    def _load_low_level_policy(self, ckpt_path: str):
        """加载并冻结底层控制器"""
        if self.debug:
            print(f"[Env] 加载底层策略: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"底层策略不存在: {ckpt_path}")
        
        # 加载 checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # 创建 ActorCritic (需要匹配底层网络结构)
        num_obs = self.env.num_obs
        num_actions = self.env.num_actions
        num_priv_obs = getattr(self.env, 'num_privileged_obs', num_obs)

        def _infer_hidden_dims(state_dict, prefix: str) -> Optional[list]:
            dims = []
            idx = 0
            while True:
                weight_key = f"{prefix}.{idx}.weight"
                if weight_key not in state_dict:
                    break
                out_dim = state_dict[weight_key].shape[0]
                dims.append(out_dim)
                idx += 2
            if len(dims) <= 1:
                return None
            return dims[:-1]

        state_dict = None
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'actor_state_dict' in ckpt:
                state_dict = ckpt['actor_state_dict']
        if state_dict is None and isinstance(ckpt, dict):
            state_dict = ckpt

        actor_hidden_dims = _infer_hidden_dims(state_dict, "actor") if state_dict else None
        critic_hidden_dims = _infer_hidden_dims(state_dict, "critic") if state_dict else None
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
            print("[Warning] 未能推断 actor hidden dims，使用默认 [256, 256, 256]")
        if critic_hidden_dims is None:
            critic_hidden_dims = actor_hidden_dims

        self.low_level_policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_priv_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in ckpt:
            self.low_level_policy.load_state_dict(ckpt['model_state_dict'])
        elif 'actor_state_dict' in ckpt:
            self.low_level_policy.actor.load_state_dict(ckpt['actor_state_dict'])
        else:
            print("[Warning] Checkpoint 格式未知，尝试直接加载")
            self.low_level_policy.load_state_dict(ckpt)
        
        # 冻结参数
        self.low_level_policy.eval()
        for param in self.low_level_policy.parameters():
            param.requires_grad = False
        
        if self.debug:
            print(f"[Env] ✓ 底层策略加载成功 (冻结)")

    def reset(self) -> Dict[str, torch.Tensor]:
        """复位环境"""
        obs, _ = self.env.reset()

        if hasattr(self.env, "commands"):
            self.env.commands[:, :3] = 0.0
            if hasattr(self.env, "commands_scale") and hasattr(self.env, "obs_buf"):
                if self.env.obs_buf.shape[1] >= 3:
                    self.env.obs_buf[:, -3:] = 0.0
        
        self.episode_length_buf.zero_()
        self.prev_gate_y.zero_()
        self.reach_given.zero_()
        self.goal_change_count.zero_()
        self.target_lost_steps.zero_()
        self.stable_follow_steps.zero_()
        self._apply_scene_difficulty_for_resets(None)
        self.reward_affordance_override = None
        self.post_processor.reset(self.num_envs, self.device)
        self._resample_forced_forward_speed(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        self._refresh_depth_images(force=True)
        obs_dict = self._get_high_level_obs()
        self.prev_robot_pos = self.env.root_states[:, :3].clone()
        self.prev_cmd_pred_x.zero_()
        self.prev_cmd_exec_mean.zero_()
        self.rotate_only_active.zero_()
        if hasattr(self.env, "goal_world"):
            self.prev_goal_world = self.env.goal_world.clone()
        
        self.last_obs = obs_dict
        return obs_dict

    def _resample_forced_forward_speed(self, env_ids: torch.Tensor) -> None:
        if env_ids is None or env_ids.numel() == 0:
            return
        if not bool(getattr(self.env, "s_avoid_enabled", False)):
            return
        stage_speed = {
            1: (0.2, 0.5),
            2: (0.2, 0.4),
            3: (0.2, 0.45),
            4: (0.2, 0.35),
        }
        stage_id = int(getattr(self.env, "s_avoid_stage", 1))
        stage_start_iter = int(getattr(self, "forced_forward_stage_start_iter", 0))
        current_iter = int(getattr(self, "forced_forward_current_iter", stage_start_iter))
        stage_iter = max(0, current_iter - stage_start_iter)
        stage_warmup_ratio = min(float(stage_iter) / 100.0, 1.0)
        train_warmup_ratio = float(getattr(self, "forced_forward_train_warmup_ratio", 0.0))
        train_warmup_ratio = min(max(train_warmup_ratio, 0.0), 1.0)
        warmup_ratio = min(stage_warmup_ratio, train_warmup_ratio)
        v_min_base, v_max_base = stage_speed.get(stage_id, (0.2, 0.4))
        v_max = v_min_base + (v_max_base - v_min_base) * warmup_ratio
        self.forced_forward_speed[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * max(v_max - v_min_base, 0.0) + v_min_base
        )

    def set_scene_difficulty_target(self, difficulty: float) -> None:
        """Set curriculum target; it becomes active only for envs after reset."""
        self.scene_difficulty_pending = float(np.clip(float(difficulty), 0.0, 1.0))

    def _normalize_scene_difficulty_override(
        self,
        difficulty_override,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if difficulty_override is None:
            return torch.full(
                (self.num_envs,),
                float(self.scene_difficulty_pending),
                device=self.device,
                dtype=dtype,
            )
        if torch.is_tensor(difficulty_override):
            d_tensor = difficulty_override.to(device=self.device, dtype=dtype)
        else:
            d_tensor = torch.as_tensor(difficulty_override, device=self.device, dtype=dtype)
        if d_tensor.ndim == 0:
            d_vec = torch.full((self.num_envs,), float(d_tensor.item()), device=self.device, dtype=dtype)
        elif d_tensor.numel() == self.num_envs:
            d_vec = d_tensor.reshape(-1)
        else:
            raise ValueError(
                f"scene_difficulty_override shape mismatch: {tuple(d_tensor.shape)} (num_envs={self.num_envs})"
            )
        d_vec = torch.nan_to_num(d_vec, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(d_vec, 0.0, 1.0)

    def _current_scene_difficulty(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        difficulty_override = getattr(self.env, "scene_difficulty_override", self.scene_difficulty_override)
        return self._normalize_scene_difficulty_override(difficulty_override, dtype=dtype)

    def _apply_scene_difficulty_for_resets(self, reset_mask: Optional[torch.Tensor]) -> None:
        pending = float(np.clip(float(self.scene_difficulty_pending), 0.0, 1.0))
        if reset_mask is None:
            self.scene_difficulty_override.fill_(pending)
        else:
            if reset_mask.dtype != torch.bool:
                raise ValueError("reset_mask must be bool tensor")
            if reset_mask.shape[0] != self.num_envs:
                raise ValueError(f"reset_mask shape mismatch: {tuple(reset_mask.shape)}")
            if reset_mask.any():
                self.scene_difficulty_override[reset_mask] = pending
        self.scene_difficulty_override = self._normalize_scene_difficulty_override(
            self.scene_difficulty_override,
            dtype=self.scene_difficulty_override.dtype,
        )
        self.env.scene_difficulty_override = self.scene_difficulty_override

    def _refresh_depth_images(self, force: bool = False) -> None:
        if not hasattr(self.env, "camera_cfg"):
            return
        if self.env.camera_cfg is None:
            return
        if not hasattr(self.env, "depth_images"):
            return

        if not getattr(self.env, "enable_camera", False):
            if hasattr(self.env, "depth_raw") and hasattr(self.env, "_process_depth_for_network"):
                self.env.depth_raw.fill_(self.env.camera_cfg.far_clip)
                processed = self.env._process_depth_for_network(self.env.depth_raw)
                self.env.depth_images[:] = processed
            else:
                self.env.depth_images.fill_(self.env.camera_cfg.far_clip)
            return

        if not force and self.env.common_step_counter % self.env.camera_cfg.capture_interval != 0:
            return

        if hasattr(self.env, "_get_depth_images") and hasattr(self.env, "_process_depth_for_network"):
            depth_raw = self.env._get_depth_images()
            processed = self.env._process_depth_for_network(depth_raw)
            self.env.depth_images[:] = processed

    def _get_affordance_reference_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_xy = self.env.root_states[:, :2]
        ref_yaw = self._quat_to_yaw(self.env.root_states[:, 3:7])
        if not torch.isfinite(ref_yaw).all():
            ref_yaw = ref_yaw.clone()
            ref_yaw[~torch.isfinite(ref_yaw)] = 0.0
        if not torch.isfinite(ref_xy).all():
            ref_xy = ref_xy.clone()
            ref_xy[~torch.isfinite(ref_xy)] = 0.0
        if self.affordance_origin_mode != "camera_mount":
            return ref_xy, ref_yaw

        offset = self.affordance_origin_local_xy.to(device=ref_xy.device, dtype=ref_xy.dtype).view(1, 2)
        cos_h = torch.cos(ref_yaw)
        sin_h = torch.sin(ref_yaw)
        offset_world_x = cos_h * offset[:, 0] - sin_h * offset[:, 1]
        offset_world_y = sin_h * offset[:, 0] + cos_h * offset[:, 1]
        ref_xy = torch.stack([
            ref_xy[:, 0] + offset_world_x,
            ref_xy[:, 1] + offset_world_y,
        ], dim=1)
        return ref_xy, ref_yaw

    def _world_to_local_xy(
        self,
        world_xy: torch.Tensor,
        ref_xy: torch.Tensor,
        ref_yaw: torch.Tensor,
    ) -> torch.Tensor:
        if world_xy.ndim != 2 or world_xy.shape[1] != 2:
            raise ValueError(f"world_xy shape invalid: {tuple(world_xy.shape)}")
        if ref_xy.ndim != 2 or ref_xy.shape[1] != 2:
            raise ValueError(f"ref_xy shape invalid: {tuple(ref_xy.shape)}")
        if ref_yaw.ndim != 1 or ref_yaw.shape[0] != world_xy.shape[0]:
            raise ValueError(f"ref_yaw shape invalid: {tuple(ref_yaw.shape)}")
        dx = world_xy[:, 0] - ref_xy[:, 0]
        dy = world_xy[:, 1] - ref_xy[:, 1]
        cos_h = torch.cos(ref_yaw)
        sin_h = torch.sin(ref_yaw)
        x_local = cos_h * dx + sin_h * dy
        y_local = -sin_h * dx + cos_h * dy
        return torch.stack([x_local, y_local], dim=1)

    def _compute_gt_affordance_from_heightfield(self) -> Optional[torch.Tensor]:
        if getattr(self.env, "height_samples", None) is None:
            return None
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = map_extent / map_size

        x_centers = torch.linspace(
            -map_extent * 0.5 + cell * 0.5,
            map_extent * 0.5 - cell * 0.5,
            map_size,
            device=self.device,
        )
        y_centers = torch.linspace(
            0.0 + cell * 0.5,
            map_extent - cell * 0.5,
            map_size,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="xy")
        x_body = grid_x.reshape(-1).unsqueeze(0)
        y_body = grid_y.reshape(-1).unsqueeze(0)

        ref_xy, yaw = self._get_affordance_reference_pose()
        cos_h = torch.cos(yaw).unsqueeze(1)
        sin_h = torch.sin(yaw).unsqueeze(1)

        x_world = ref_xy[:, 0:1] + cos_h * x_body - sin_h * y_body
        y_world = ref_xy[:, 1:2] + sin_h * x_body + cos_h * y_body

        border = self.env.cfg.terrain.border_size
        scale = self.env.cfg.terrain.horizontal_scale
        max_row = self.env.height_samples.shape[0] - 2
        max_col = self.env.height_samples.shape[1] - 2
        idx_col = torch.clamp(((x_world + border) / scale).long(), 0, max_col)
        idx_row = torch.clamp(((y_world + border) / scale).long(), 0, max_row)

        heights = self.env.height_samples[idx_row, idx_col] * self.env.cfg.terrain.vertical_scale
        heights = heights.view(self.num_envs, map_size, map_size)

        occ_all = (heights > 1e-6).float()
        occ_block = (heights >= self.affordance_blocking_height).float().unsqueeze(1)
        low_mask = (heights > 1e-6) & (heights < self.affordance_crossable_height)
        low_obstacle = low_mask.float()

        radius_cells = int(math.ceil(self.affordance_clearance / cell))
        if radius_cells > 0:
            kernel = 2 * radius_cells + 1
            pooled = F.max_pool2d(occ_block, kernel_size=kernel, stride=1, padding=radius_cells)
            passable = (pooled <= 0.5) & (occ_block < 0.5)
        else:
            passable = occ_block < 0.5
        passable_gap = passable.float().squeeze(1)

        return torch.cat([
            occ_all.unsqueeze(1),
            passable_gap.unsqueeze(1),
            low_obstacle.unsqueeze(1),
        ], dim=1)

    def _compute_gt_affordance_from_scene(self) -> Optional[torch.Tensor]:
        if not hasattr(self.env, "scene_spec_cache"):
            if not hasattr(self.env, "s_avoid_active"):
                return None
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = map_extent / map_size
        if not math.isfinite(cell) or cell <= 0.0:
            if not getattr(self, "_scene_aff_cell_warned", False):
                print(f"[Warn] scene affordance cell invalid: cell={cell}, extent={map_extent}, size={map_size}")
                self._scene_aff_cell_warned = True
            return None
        occ_all = torch.zeros(self.num_envs, map_size, map_size, device=self.device)

        ref_xy, yaw = self._get_affordance_reference_pose()

        x_min = -0.5 * map_extent
        y_min = 0.0

        def rasterize(env_id: int, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
            if not (
                math.isfinite(center_x)
                and math.isfinite(center_y)
                and math.isfinite(size_x)
                and math.isfinite(size_y)
            ):
                if not getattr(self, "_scene_aff_nan_warned", False):
                    print(
                        "[Warn] scene affordance rasterize got NaN/inf "
                        f"(env={env_id}, center=({center_x},{center_y}), size=({size_x},{size_y}))"
                    )
                    self._scene_aff_nan_warned = True
                return
            if size_x <= 0.0 or size_y <= 0.0:
                return
            world_xy = torch.tensor([[center_x, center_y]], device=self.device, dtype=ref_xy.dtype)
            x_body, y_body = self._world_to_local_xy(
                world_xy,
                ref_xy[env_id:env_id + 1],
                yaw[env_id:env_id + 1],
            )[0].tolist()
            x0 = x_body - 0.5 * size_x
            x1 = x_body + 0.5 * size_x
            y0 = y_body - 0.5 * size_y
            y1 = y_body + 0.5 * size_y
            ix0 = int(math.floor((x0 - x_min) / cell))
            ix1 = int(math.ceil((x1 - x_min) / cell))
            iy0 = int(math.floor((y0 - y_min) / cell))
            iy1 = int(math.ceil((y1 - y_min) / cell))
            ix0 = max(0, min(map_size - 1, ix0))
            ix1 = max(0, min(map_size - 1, ix1))
            iy0 = max(0, min(map_size - 1, iy0))
            iy1 = max(0, min(map_size - 1, iy1))
            occ_all[env_id, ix0:ix1 + 1, iy0:iy1 + 1] = 1.0

        def rasterize_actor_bbox(env_id: int, center_x: float, center_y: float, half_x: float, half_y: float, yaw_world: float) -> None:
            if not (math.isfinite(half_x) and math.isfinite(half_y) and math.isfinite(yaw_world)):
                return
            rel_yaw = yaw_world - float(yaw[env_id].item())
            cos_a = abs(math.cos(rel_yaw))
            sin_a = abs(math.sin(rel_yaw))
            size_x = 2.0 * (cos_a * half_x + sin_a * half_y)
            size_y = 2.0 * (sin_a * half_x + cos_a * half_y)
            rasterize(env_id, center_x, center_y, size_x, size_y)

        def get_static_obstacle_field(spec, key: str):
            if isinstance(spec, dict):
                return spec[key]
            return getattr(spec, key)

        if hasattr(self.env, "scene_spec_cache"):
            for env_id in range(self.num_envs):
                scene_spec = self.env.scene_spec_cache[env_id]
                if scene_spec is None:
                    continue
                origin = self.env.env_origins[env_id]
                for spec in scene_spec.static_obstacles:
                    spec_pos = get_static_obstacle_field(spec, "position")
                    spec_size = get_static_obstacle_field(spec, "size")
                    center_x = float(origin[0].item() + spec_pos[0])
                    center_y = float(origin[1].item() + spec_pos[1])
                    rasterize(env_id, center_x, center_y, spec_size[0], spec_size[1])

        if hasattr(self.env, "dynamic_active") and self.env.dynamic_active is not None:
            dyn_size = float(getattr(self.env.cfg.terrain, "scene_dynamic_size", 0.4))
            for env_id in range(self.num_envs):
                if self.env.dynamic_active[env_id].numel() == 0:
                    continue
                origin = self.env.env_origins[env_id]
                t = float(self.env.scene_dyn_time[env_id].item())
                for obs_id, active in enumerate(self.env.dynamic_active[env_id]):
                    if not bool(active.item()):
                        continue
                    period = float(self.env.dynamic_period[env_id, obs_id].item())
                    phase = float(self.env.dynamic_phase[env_id, obs_id].item())
                    path_len = float(self.env.dynamic_path_len[env_id, obs_id].item())
                    half = 0.5 * period if period > 1e-6 else 0.0
                    tau = (t + phase) % period if period > 1e-6 else 0.0
                    progress = tau / half if half > 1e-6 else 0.0
                    if tau > half and half > 1e-6:
                        progress = (period - tau) / half
                    progress = max(0.0, min(1.0, progress))
                    dist = progress * path_len
                    start = self.env.dynamic_start[env_id, obs_id]
                    direction = self.env.dynamic_dir[env_id, obs_id]
                    pos_local = start + direction * dist
                    center_x = float(origin[0].item() + pos_local[0].item())
                    center_y = float(origin[1].item() + pos_local[1].item())
                    rasterize(env_id, center_x, center_y, dyn_size, dyn_size)

        if hasattr(self.env, "s_avoid_active") and self.env.s_avoid_active is not None:
            cap_slots = int(getattr(self.env, "s_avoid_capsule_slot_count", 0))
            box_slots = int(getattr(self.env, "s_avoid_box_slot_count", 0))
            wall_slots = int(getattr(self.env, "s_avoid_wall_slot_count", 0))
            total_slots = int(getattr(self.env, "s_avoid_total_slots", 0))
            cap_r = float(getattr(self.env.cfg.terrain, "avoid_capsule_radius", 0.15))
            box_hx = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_x", 0.4))
            box_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_y", 0.4))
            wall_hx = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_thickness", 0.12))
            wall_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_length", 6.0))
            for env_id in range(self.num_envs):
                for slot in range(total_slots):
                    if not bool(self.env.s_avoid_active[env_id, slot].item()):
                        continue
                    center_x = float(self.env.s_avoid_pos_world[env_id, slot, 0].item())
                    center_y = float(self.env.s_avoid_pos_world[env_id, slot, 1].item())
                    if slot < cap_slots:
                        rasterize(env_id, center_x, center_y, 2.0 * cap_r, 2.0 * cap_r)
                        continue
                    quat = self.env.s_avoid_quat_world[env_id, slot]
                    yaw_world = self._quat_to_yaw(quat.unsqueeze(0))[0].item()
                    if slot < cap_slots + box_slots:
                        rasterize_actor_bbox(env_id, center_x, center_y, box_hx, box_hy, yaw_world)
                    elif slot < cap_slots + box_slots + wall_slots:
                        rasterize_actor_bbox(env_id, center_x, center_y, wall_hx, wall_hy, yaw_world)

        occ_block = occ_all.unsqueeze(1)
        low_obstacle = torch.zeros_like(occ_all)
        radius_cells = int(math.ceil(self.affordance_clearance / cell))
        if radius_cells > 0:
            kernel = 2 * radius_cells + 1
            pooled = F.max_pool2d(occ_block, kernel_size=kernel, stride=1, padding=radius_cells)
            passable = (pooled <= 0.5) & (occ_block < 0.5)
        else:
            passable = occ_block < 0.5
        passable_gap = passable.float().squeeze(1)

        if (
            hasattr(self.env, "nav_cfg")
            and self.env.nav_cfg is not None
            and bool(getattr(self.env.nav_cfg, "affordance_debug", False))
            and self.debug
        ):
            if not getattr(self, "_aff_debug_logged", False):
                mid = map_size // 2
                occ_center = float(occ_all[0, mid, mid].item())
                occ_front = float(occ_all[0, mid, min(map_size - 1, mid + 2)].item())
                occ_right = float(occ_all[0, min(map_size - 1, mid + 2), mid].item())
                print(
                    "[AffDebug] occ(center/front/right)="
                    f"({occ_center:.1f}/{occ_front:.1f}/{occ_right:.1f}) "
                    f"map_extent={map_extent:.2f} cell={cell:.3f}"
                )
                self._aff_debug_logged = True

        return torch.cat([
            occ_all.unsqueeze(1),
            passable_gap.unsqueeze(1),
            low_obstacle.unsqueeze(1),
        ], dim=1)

    def _merge_affordance_maps(
        self,
        aff_primary: Optional[torch.Tensor],
        aff_scene: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if aff_primary is None:
            return aff_scene
        if aff_scene is None:
            return aff_primary
        occ = torch.maximum(aff_primary[:, 0], aff_scene[:, 0])
        passable = torch.minimum(aff_primary[:, 1], aff_scene[:, 1])
        if aff_primary.size(1) >= 3 and aff_scene.size(1) >= 3:
            low = torch.maximum(aff_primary[:, 2], aff_scene[:, 2])
        elif aff_primary.size(1) >= 3:
            low = aff_primary[:, 2]
        elif aff_scene.size(1) >= 3:
            low = aff_scene[:, 2]
        else:
            low = torch.zeros_like(occ)
        return torch.stack([occ, passable, low], dim=1)

    def _build_affordance_dist_map(self) -> torch.Tensor:
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = self.affordance_cell_size
        x_centers = torch.linspace(
            -map_extent * 0.5 + cell * 0.5,
            map_extent * 0.5 - cell * 0.5,
            map_size,
            device=self.device,
        )
        y_centers = torch.linspace(
            0.0 + cell * 0.5,
            map_extent - cell * 0.5,
            map_size,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="xy")
        return torch.sqrt(grid_x ** 2 + grid_y ** 2)

    def _build_affordance_geometry(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = self.affordance_cell_size
        x_centers = torch.linspace(
            -map_extent * 0.5 + cell * 0.5,
            map_extent * 0.5 - cell * 0.5,
            map_size,
            device=self.device,
        )
        y_centers = torch.linspace(
            0.0 + cell * 0.5,
            map_extent - cell * 0.5,
            map_size,
            device=self.device,
        )
        # Keep axis order consistent with scene rasterization: [x_right, y_forward].
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        bearing_y = torch.atan2(grid_x, grid_y)
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        visible = torch.ones_like(dist, dtype=torch.bool)
        if self.camera_fov_rad is not None and self.camera_fov_rad > 0.0:
            angle = torch.atan2(
                torch.sin(bearing_y - self.camera_bearing_rad),
                torch.cos(bearing_y - self.camera_bearing_rad),
            )
            visible = visible & (torch.abs(angle) <= 0.5 * self.camera_fov_rad)
        if self.camera_vertical_fov_rad is not None and self.camera_vertical_fov_rad > 0.0:
            ground_angle_down = torch.atan2(
                torch.full_like(dist, float(self.camera_height_m)),
                torch.clamp(dist, min=1e-6),
            )
            lower = float(self.camera_tilt_down_rad + 0.5 * self.camera_vertical_fov_rad)
            upper = max(0.0, float(self.camera_tilt_down_rad - 0.5 * self.camera_vertical_fov_rad))
            visible = visible & (ground_angle_down >= upper - 1e-6)
            visible = visible & (ground_angle_down <= lower + 1e-6)
        if self.camera_near is not None and self.camera_near > 0.0:
            visible = visible & (dist >= float(self.camera_near) - 1e-6)
        if self.camera_far is not None:
            visible = visible & (dist <= float(self.camera_far) + 1e-6)
        return grid_x, grid_y, bearing_y, visible

    def _compute_clearance_from_affordance(self, aff_map: torch.Tensor) -> torch.Tensor:
        occ = aff_map[:, 0] > 0.5
        dist_map = self.affordance_dist_map
        if dist_map.device != aff_map.device:
            dist_map = dist_map.to(aff_map.device)
        dist = torch.where(occ, dist_map, torch.full_like(dist_map, self.affordance_map_extent))
        min_dist = dist.amin(dim=(1, 2))
        no_obs = ~occ.flatten(1).any(dim=1)
        if no_obs.any():
            min_dist = torch.where(no_obs, torch.full_like(min_dist, self.affordance_map_extent), min_dist)
        return min_dist

    def _compute_objective_difficulty_from_local_map(
        self,
        local_map_2ch: torch.Tensor,
    ) -> torch.Tensor:
        """Objective crowding scalar from full GT local map, independent of FOV masking."""
        if local_map_2ch.ndim != 4 or local_map_2ch.size(1) < 2:
            return difficulty_from_gap(local_map_2ch)

        occ = torch.clamp(
            torch.nan_to_num(local_map_2ch[:, 0], nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        clearance = torch.clamp(
            torch.nan_to_num(local_map_2ch[:, 1], nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        if x_map.device != local_map_2ch.device:
            x_map = x_map.to(local_map_2ch.device)
            y_map = y_map.to(local_map_2ch.device)

        radius = max(float(self.affordance_difficulty_radius), 1e-3)
        radial_mask = ((x_map ** 2 + y_map ** 2) <= (radius ** 2)).float()
        denom = radial_mask.sum().clamp_min(1.0)

        occ_ratio = (occ * radial_mask.unsqueeze(0)).sum(dim=(1, 2)) / denom
        clearance_cost = ((1.0 - clearance) * radial_mask.unsqueeze(0)).sum(dim=(1, 2)) / denom
        difficulty = 0.5 * occ_ratio + 0.5 * clearance_cost
        return torch.clamp(difficulty, 0.0, 1.0)

    def _compute_clearance_along_cmd(
        self,
        aff_map: torch.Tensor,
        cmd_xy: torch.Tensor,
        cone_half_angle_deg: float = 25.0,
    ) -> torch.Tensor:
        """
        A light-weight, command-conditioned clearance proxy.

        Returns the min distance to occupied cells within a cone aligned with cmd_xy.
        This is used as a consistent proxy for (risk_F, risk_A) until RiskAlong is
        implemented.
        """
        if aff_map.ndim != 4 or aff_map.size(1) < 1:
            return torch.full((cmd_xy.shape[0],), self.affordance_map_extent, device=cmd_xy.device)
        if cmd_xy.dim() != 2 or cmd_xy.size(1) != 2:
            raise ValueError("cmd_xy must have shape (N, 2)")

        occ = aff_map[:, 0] > 0.5
        dist_map = self.affordance_dist_map
        bearing_map = self.affordance_bearing_map
        visible = self.affordance_visible_mask
        if dist_map.device != aff_map.device:
            dist_map = dist_map.to(aff_map.device)
        if bearing_map.device != aff_map.device:
            bearing_map = bearing_map.to(aff_map.device)
        if visible is None:
            visible = torch.ones_like(dist_map, dtype=torch.bool)
        elif visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        # cmd bearing is defined w.r.t +Y forward: atan2(x, y)
        cmd_bearing = torch.atan2(cmd_xy[:, 0], cmd_xy[:, 1]).view(-1, 1, 1)
        bearing = bearing_map.view(1, *bearing_map.shape)
        angle = torch.atan2(torch.sin(bearing - cmd_bearing), torch.cos(bearing - cmd_bearing))
        cone_half = math.radians(cone_half_angle_deg)
        sector = (torch.abs(angle) <= cone_half) & visible.view(1, *visible.shape)

        # If cmd is (near) zero, fall back to global clearance.
        cmd_norm = torch.norm(cmd_xy, dim=-1)
        active = cmd_norm > 1e-3
        if not active.any():
            return self._compute_clearance_from_affordance(aff_map)

        dist = torch.where(sector & occ, dist_map.view(1, *dist_map.shape), torch.full_like(dist_map, self.affordance_map_extent).view(1, *dist_map.shape))
        min_dist = dist.amin(dim=(1, 2))
        no_obs = ~(sector & occ).flatten(1).any(dim=1)
        if no_obs.any():
            min_dist = torch.where(no_obs, torch.full_like(min_dist, self.affordance_map_extent), min_dist)
        if (~active).any():
            global_clear = self._compute_clearance_from_affordance(aff_map)
            min_dist = torch.where(active, min_dist, global_clear)
        return min_dist

    def _compute_front_half_obstacle_distance(
        self,
        aff_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Min distance to occupied cells in the front half-plane (y_forward > 0).
        This is used for avoid-mode gating so side-front obstacles can trigger
        avoidance without being polluted by obstacles behind the robot.
        """
        if aff_map.ndim != 4 or aff_map.size(1) < 1:
            return torch.full((aff_map.shape[0],), self.affordance_map_extent, device=aff_map.device)

        occ = aff_map[:, 0] > 0.5
        dist_map = self.affordance_dist_map
        y_map = self.affordance_y_map
        visible = self.affordance_visible_mask
        if dist_map.device != aff_map.device:
            dist_map = dist_map.to(aff_map.device)
        if y_map.device != aff_map.device:
            y_map = y_map.to(aff_map.device)
        if visible is None:
            visible = torch.ones_like(dist_map, dtype=torch.bool)
        elif visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        front_half = (y_map > 0.0) & visible
        dist = torch.where(
            front_half.view(1, *front_half.shape) & occ,
            dist_map.view(1, *dist_map.shape),
            torch.full_like(dist_map, self.affordance_map_extent).view(1, *dist_map.shape),
        )
        min_dist = dist.amin(dim=(1, 2))
        no_obs = ~(front_half.view(1, *front_half.shape) & occ).flatten(1).any(dim=1)
        if no_obs.any():
            min_dist = torch.where(no_obs, torch.full_like(min_dist, self.affordance_map_extent), min_dist)
        return min_dist

    def _compute_front_side_obstacle_distance(
        self,
        aff_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Min occupied distance on the left/right side of the front half-plane.
        This is used to build a simple near-obstacle side-risk teacher for avoid mode.
        """
        if aff_map.ndim != 4 or aff_map.size(1) < 1:
            fill = torch.full((aff_map.shape[0],), self.affordance_map_extent, device=aff_map.device)
            return fill, fill

        occ = aff_map[:, 0] > 0.5
        dist_map = self.affordance_dist_map
        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        visible = self.affordance_visible_mask
        if dist_map.device != aff_map.device:
            dist_map = dist_map.to(aff_map.device)
        if x_map.device != aff_map.device:
            x_map = x_map.to(aff_map.device)
            y_map = y_map.to(aff_map.device)
        if visible is None:
            visible = torch.ones_like(dist_map, dtype=torch.bool)
        elif visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        front_half = (y_map > 0.0) & visible
        left_half = front_half & (x_map < 0.0)
        right_half = front_half & (x_map > 0.0)
        fill_map = torch.full_like(dist_map, self.affordance_map_extent)

        left_dist = torch.where(
            left_half.view(1, *left_half.shape) & occ,
            dist_map.view(1, *dist_map.shape),
            fill_map.view(1, *dist_map.shape),
        )
        right_dist = torch.where(
            right_half.view(1, *right_half.shape) & occ,
            dist_map.view(1, *dist_map.shape),
            fill_map.view(1, *dist_map.shape),
        )
        left_min = left_dist.amin(dim=(1, 2))
        right_min = right_dist.amin(dim=(1, 2))
        no_left = ~(left_half.view(1, *left_half.shape) & occ).flatten(1).any(dim=1)
        no_right = ~(right_half.view(1, *right_half.shape) & occ).flatten(1).any(dim=1)
        if no_left.any():
            left_min = torch.where(no_left, torch.full_like(left_min, self.affordance_map_extent), left_min)
        if no_right.any():
            right_min = torch.where(no_right, torch.full_like(right_min, self.affordance_map_extent), right_min)
        return left_min, right_min

    def _compute_side_candidate_clearance(
        self,
        aff_map: torch.Tensor,
        lateral_probe_x: float = 0.45,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compare two candidate passing channels:
        - left candidate: slight left-shifted forward motion
        - right candidate: slight right-shifted forward motion

        This is a better side teacher for zig-zag avoid scenes than left/right
        nearest-obstacle distance, because centered obstacles can still produce
        asymmetric candidate-channel clearance.
        """
        probe_x = abs(float(lateral_probe_x))
        left_cmd = torch.zeros((aff_map.shape[0], 2), device=aff_map.device, dtype=aff_map.dtype)
        right_cmd = torch.zeros((aff_map.shape[0], 2), device=aff_map.device, dtype=aff_map.dtype)
        left_cmd[:, 0] = -probe_x
        left_cmd[:, 1] = 1.0
        right_cmd[:, 0] = probe_x
        right_cmd[:, 1] = 1.0
        left_clear = self._compute_clearance_along_cmd(aff_map, left_cmd)
        right_clear = self._compute_clearance_along_cmd(aff_map, right_cmd)
        return left_clear, right_clear

    def _compute_nearest_row_gap_target(
        self,
        robot_pos: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        gap_center = torch.zeros(self.num_envs, device=self.device)
        row_y = torch.zeros(self.num_envs, device=self.device)
        valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        gap_left = torch.zeros(self.num_envs, device=self.device)
        gap_right = torch.zeros(self.num_envs, device=self.device)
        gap_left_eff = torch.zeros(self.num_envs, device=self.device)
        gap_right_eff = torch.zeros(self.num_envs, device=self.device)
        eff_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            not hasattr(self.env, "s_avoid_active")
            or self.env.s_avoid_active is None
            or not hasattr(self.env, "s_avoid_pos_world")
            or not hasattr(self.env, "s_avoid_band_x_min")
            or not hasattr(self.env, "s_avoid_band_x_max")
        ):
            return gap_center, row_y, valid, gap_left, gap_right, gap_left_eff, gap_right_eff, eff_valid

        cap_slots = int(getattr(self.env, "s_avoid_capsule_slot_count", 0))
        box_slots = int(getattr(self.env, "s_avoid_box_slot_count", 0))
        wall_slots = int(getattr(self.env, "s_avoid_wall_slot_count", 0))
        total_slots = int(getattr(self.env, "s_avoid_total_slots", 0))
        if total_slots <= 0:
            return gap_center, row_y, valid, gap_left, gap_right, gap_left_eff, gap_right_eff, eff_valid

        cap_half = float(getattr(self.env.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_half = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_y", 0.4))
        wall_hx = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_length", 6.0))
        row_tol = float(getattr(self.reward_cfg, "avoid_row_group_tol", 0.25))
        row_margin = float(getattr(self.reward_cfg, "avoid_row_margin", 0.15))
        row_release_offset = float(getattr(self.reward_cfg, "avoid_row_release_offset", 0.20))
        body_half_length = float(getattr(self.env.cfg.terrain, "avoid_body_half_length", 0.0))
        robot_half_width = 0.5 * float(getattr(self, "body_width", 0.44))
        eff_margin = float(getattr(self.reward_cfg, "avoid_row_effective_margin", 0.0))
        gap_shrink = max(robot_half_width + eff_margin, 0.0)

        def slot_half_extents(env_id: int, slot_id: int) -> Tuple[float, float]:
            if slot_id < cap_slots:
                return cap_half + row_margin, cap_half + row_margin
            quat = self.env.s_avoid_quat_world[env_id, slot_id]
            yaw = self._quat_to_yaw(quat.unsqueeze(0))[0].item()
            cos_yaw = abs(math.cos(yaw))
            sin_yaw = abs(math.sin(yaw))
            if slot_id < cap_slots + box_slots:
                half_x = cos_yaw * box_half + sin_yaw * box_hy
                half_y = sin_yaw * box_half + cos_yaw * box_hy
                return half_x + row_margin, half_y + row_margin
            if slot_id < cap_slots + box_slots + wall_slots:
                half_x = cos_yaw * wall_hx + sin_yaw * wall_hy
                half_y = sin_yaw * wall_hx + cos_yaw * wall_hy
                return half_x + row_margin, half_y + row_margin
            return row_margin, row_margin

        for env_id in range(self.num_envs):
            active_slots = self.env.s_avoid_active[env_id].nonzero(as_tuple=False).flatten()
            if active_slots.numel() == 0:
                continue
            obs_xy = self.env.s_avoid_pos_world[env_id, active_slots, :2]
            row_candidates = []
            rear_y = float(robot_pos[env_id, 1].item()) - body_half_length
            for idx, slot in enumerate(active_slots.tolist()):
                cy = float(obs_xy[idx, 1].item())
                _, half_y = slot_half_extents(env_id, int(slot))
                row_back_edge = cy + half_y
                if rear_y >= (row_back_edge + row_release_offset):
                    continue
                row_candidates.append((slot, cy))
            if len(row_candidates) == 0:
                continue
            nearest_row_y = min(item[1] for item in row_candidates)
            row_slots = [
                int(slot)
                for slot, cy in row_candidates
                if abs(cy - nearest_row_y) <= row_tol
            ]
            if len(row_slots) == 0:
                continue

            band_l = float(self.env.s_avoid_band_x_min[env_id].item())
            band_r = float(self.env.s_avoid_band_x_max[env_id].item())
            if band_r <= band_l:
                continue

            intervals = []
            for slot in row_slots:
                cx = float(self.env.s_avoid_pos_world[env_id, slot, 0].item())
                half_x, _ = slot_half_extents(env_id, int(slot))
                left = max(band_l, cx - half_x)
                right = min(band_r, cx + half_x)
                if right > left:
                    intervals.append((left, right))

            if intervals:
                intervals.sort(key=lambda item: item[0])
                merged = []
                for left, right in intervals:
                    if (not merged) or (left > merged[-1][1]):
                        merged.append([left, right])
                    else:
                        merged[-1][1] = max(merged[-1][1], right)
            else:
                merged = []

            cursor = band_l
            best_width = -1.0
            best_center = 0.5 * (band_l + band_r)
            best_left = band_l
            best_right = band_r
            for left, right in merged:
                if left > cursor:
                    width = left - cursor
                    if width > best_width:
                        best_width = width
                        best_center = 0.5 * (cursor + left)
                        best_left = cursor
                        best_right = left
                cursor = max(cursor, right)
            if band_r > cursor:
                width = band_r - cursor
                if width > best_width:
                    best_width = width
                    best_center = 0.5 * (cursor + band_r)
                    best_left = cursor
                    best_right = band_r

            if best_width <= 1e-6:
                continue
            eff_left = best_left + gap_shrink
            eff_right = best_right - gap_shrink
            gap_center[env_id] = best_center
            row_y[env_id] = nearest_row_y
            gap_left[env_id] = best_left
            gap_right[env_id] = best_right
            gap_left_eff[env_id] = eff_left
            gap_right_eff[env_id] = eff_right
            valid[env_id] = True
            eff_valid[env_id] = eff_right > eff_left
        return gap_center, row_y, valid, gap_left, gap_right, gap_left_eff, gap_right_eff, eff_valid

    def _compute_current_next_row_gap_state(
        self,
        robot_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_gap_center = torch.zeros(self.num_envs, device=self.device)
        current_row_front_edge = torch.zeros(self.num_envs, device=self.device)
        current_row_back_edge = torch.zeros(self.num_envs, device=self.device)
        current_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        next_gap_center = torch.zeros(self.num_envs, device=self.device)
        next_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            not hasattr(self.env, "s_avoid_active")
            or self.env.s_avoid_active is None
            or not hasattr(self.env, "s_avoid_pos_world")
            or not hasattr(self.env, "s_avoid_band_x_min")
            or not hasattr(self.env, "s_avoid_band_x_max")
        ):
            return (
                current_gap_center,
                current_row_front_edge,
                current_row_back_edge,
                current_valid,
                next_gap_center,
                next_valid,
            )

        cap_slots = int(getattr(self.env, "s_avoid_capsule_slot_count", 0))
        box_slots = int(getattr(self.env, "s_avoid_box_slot_count", 0))
        wall_slots = int(getattr(self.env, "s_avoid_wall_slot_count", 0))
        total_slots = int(getattr(self.env, "s_avoid_total_slots", 0))
        if total_slots <= 0:
            return (
                current_gap_center,
                current_row_front_edge,
                current_row_back_edge,
                current_valid,
                next_gap_center,
                next_valid,
            )

        cap_half = float(getattr(self.env.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_half = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_y", 0.4))
        wall_hx = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_length", 6.0))
        row_tol = float(getattr(self.reward_cfg, "avoid_row_group_tol", 0.25))
        row_margin = float(getattr(self.reward_cfg, "avoid_row_margin", 0.15))
        row_release_offset = float(getattr(self.reward_cfg, "avoid_row_release_offset", 0.20))
        body_half_length = float(getattr(self.env.cfg.terrain, "avoid_body_half_length", 0.0))

        def slot_half_extents(env_id: int, slot_id: int) -> Tuple[float, float]:
            if slot_id < cap_slots:
                return cap_half + row_margin, cap_half + row_margin
            quat = self.env.s_avoid_quat_world[env_id, slot_id]
            yaw = self._quat_to_yaw(quat.unsqueeze(0))[0].item()
            cos_yaw = abs(math.cos(yaw))
            sin_yaw = abs(math.sin(yaw))
            if slot_id < cap_slots + box_slots:
                half_x = cos_yaw * box_half + sin_yaw * box_hy
                half_y = sin_yaw * box_half + cos_yaw * box_hy
                return half_x + row_margin, half_y + row_margin
            if slot_id < cap_slots + box_slots + wall_slots:
                half_x = cos_yaw * wall_hx + sin_yaw * wall_hy
                half_y = sin_yaw * wall_hx + cos_yaw * wall_hy
                return half_x + row_margin, half_y + row_margin
            return row_margin, row_margin

        def summarize_row(env_id: int, row_slots: List[int], band_l: float, band_r: float) -> Tuple[float, float, float, bool]:
            intervals = []
            row_front = float("inf")
            row_back = float("-inf")
            for slot in row_slots:
                cx = float(self.env.s_avoid_pos_world[env_id, slot, 0].item())
                cy = float(self.env.s_avoid_pos_world[env_id, slot, 1].item())
                half_x, half_y = slot_half_extents(env_id, int(slot))
                row_front = min(row_front, cy - half_y)
                row_back = max(row_back, cy + half_y)
                left = max(band_l, cx - half_x)
                right = min(band_r, cx + half_x)
                if right > left:
                    intervals.append((left, right))
            if row_back < row_front:
                return 0.0, 0.0, 0.0, False
            if intervals:
                intervals.sort(key=lambda item: item[0])
                merged = []
                for left, right in intervals:
                    if (not merged) or (left > merged[-1][1]):
                        merged.append([left, right])
                    else:
                        merged[-1][1] = max(merged[-1][1], right)
            else:
                merged = []
            cursor = band_l
            best_width = -1.0
            best_center = 0.5 * (band_l + band_r)
            for left, right in merged:
                if left > cursor:
                    width = left - cursor
                    if width > best_width:
                        best_width = width
                        best_center = 0.5 * (cursor + left)
                cursor = max(cursor, right)
            if band_r > cursor:
                width = band_r - cursor
                if width > best_width:
                    best_width = width
                    best_center = 0.5 * (cursor + band_r)
            return best_center, row_front, row_back, best_width > 1e-6

        for env_id in range(self.num_envs):
            active_slots = self.env.s_avoid_active[env_id].nonzero(as_tuple=False).flatten()
            if active_slots.numel() == 0:
                continue
            obs_xy = self.env.s_avoid_pos_world[env_id, active_slots, :2]
            row_candidates = []
            rear_y = float(robot_pos[env_id, 1].item()) - body_half_length
            for idx, slot in enumerate(active_slots.tolist()):
                cy = float(obs_xy[idx, 1].item())
                _, half_y = slot_half_extents(env_id, int(slot))
                row_back_edge = cy + half_y
                if rear_y >= (row_back_edge + row_release_offset):
                    continue
                row_candidates.append((int(slot), cy))
            if len(row_candidates) == 0:
                continue
            row_candidates.sort(key=lambda item: item[1])
            grouped_rows = []
            for slot, cy in row_candidates:
                if (not grouped_rows) or abs(cy - grouped_rows[-1][0]) > row_tol:
                    grouped_rows.append((cy, [slot]))
                else:
                    grouped_rows[-1][1].append(slot)
            if len(grouped_rows) == 0:
                continue

            band_l = float(self.env.s_avoid_band_x_min[env_id].item())
            band_r = float(self.env.s_avoid_band_x_max[env_id].item())
            if band_r <= band_l:
                continue

            curr_center, curr_front, curr_back, curr_ok = summarize_row(
                env_id,
                grouped_rows[0][1],
                band_l,
                band_r,
            )
            if curr_ok:
                current_gap_center[env_id] = curr_center
                current_row_front_edge[env_id] = curr_front
                current_row_back_edge[env_id] = curr_back
                current_valid[env_id] = True

            if len(grouped_rows) > 1:
                next_center, _, _, next_ok = summarize_row(
                    env_id,
                    grouped_rows[1][1],
                    band_l,
                    band_r,
                )
                if next_ok:
                    next_gap_center[env_id] = next_center
                    next_valid[env_id] = True

        return (
            current_gap_center,
            current_row_front_edge,
            current_row_back_edge,
            current_valid,
            next_gap_center,
            next_valid,
        )

    @staticmethod
    def _distance_to_interval(
        x: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        left_err = torch.clamp(left - x, min=0.0)
        right_err = torch.clamp(x - right, min=0.0)
        return left_err + right_err

    def _compute_nearest_row_forward_distance(
        self,
        robot_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist_out = torch.full((self.num_envs,), self.affordance_map_extent, device=self.device)
        valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            not hasattr(self.env, "s_avoid_active")
            or self.env.s_avoid_active is None
            or not hasattr(self.env, "s_avoid_pos_world")
        ):
            return dist_out, valid

        cap_slots = int(getattr(self.env, "s_avoid_capsule_slot_count", 0))
        box_slots = int(getattr(self.env, "s_avoid_box_slot_count", 0))
        wall_slots = int(getattr(self.env, "s_avoid_wall_slot_count", 0))
        total_slots = int(getattr(self.env, "s_avoid_total_slots", 0))
        if total_slots <= 0:
            return dist_out, valid

        cap_half = float(getattr(self.env.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_half = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_box_size_y", 0.4))
        wall_hx = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_hy = 0.5 * float(getattr(self.env.cfg.terrain, "avoid_wall_length", 6.0))
        row_tol = float(getattr(self.reward_cfg, "avoid_row_group_tol", 0.25))
        row_margin = float(getattr(self.reward_cfg, "avoid_row_margin", 0.15))
        row_release_offset = float(getattr(self.reward_cfg, "avoid_row_release_offset", 0.20))
        body_half_length = float(getattr(self.env.cfg.terrain, "avoid_body_half_length", 0.0))

        def slot_half_extents(env_id: int, slot_id: int) -> Tuple[float, float]:
            if slot_id < cap_slots:
                return cap_half + row_margin, cap_half + row_margin
            quat = self.env.s_avoid_quat_world[env_id, slot_id]
            yaw = self._quat_to_yaw(quat.unsqueeze(0))[0].item()
            cos_yaw = abs(math.cos(yaw))
            sin_yaw = abs(math.sin(yaw))
            if slot_id < cap_slots + box_slots:
                half_x = cos_yaw * box_half + sin_yaw * box_hy
                half_y = sin_yaw * box_half + cos_yaw * box_hy
                return half_x + row_margin, half_y + row_margin
            if slot_id < cap_slots + box_slots + wall_slots:
                half_x = cos_yaw * wall_hx + sin_yaw * wall_hy
                half_y = sin_yaw * wall_hx + cos_yaw * wall_hy
                return half_x + row_margin, half_y + row_margin
            return row_margin, row_margin

        for env_id in range(self.num_envs):
            active_slots = self.env.s_avoid_active[env_id].nonzero(as_tuple=False).flatten()
            if active_slots.numel() == 0:
                continue
            obs_xy = self.env.s_avoid_pos_world[env_id, active_slots, :2]
            row_candidates = []
            rear_y = float(robot_pos[env_id, 1].item()) - body_half_length
            for idx, slot in enumerate(active_slots.tolist()):
                cy = float(obs_xy[idx, 1].item())
                _, half_y = slot_half_extents(env_id, int(slot))
                row_back_edge = cy + half_y
                if rear_y >= (row_back_edge + row_release_offset):
                    continue
                row_candidates.append((slot, cy))
            if len(row_candidates) == 0:
                continue
            nearest_row_y = min(item[1] for item in row_candidates)
            row_slots = [
                int(slot)
                for slot, cy in row_candidates
                if abs(cy - nearest_row_y) <= row_tol
            ]
            if len(row_slots) == 0:
                continue

            min_dist = self.affordance_map_extent
            ry = rear_y
            for slot in row_slots:
                cy = float(self.env.s_avoid_pos_world[env_id, slot, 1].item())
                _, half_y = slot_half_extents(env_id, int(slot))
                dist = max((cy - half_y) - ry, 0.0)
                if dist < min_dist:
                    min_dist = dist
            dist_out[env_id] = min_dist
            valid[env_id] = True
        return dist_out, valid

    @staticmethod
    def _risk_from_clearance(
        clearance: torch.Tensor,
        safe_distance: float,
        free_distance: float,
    ) -> torch.Tensor:
        """
        Map clearance to a [0,1] risk scalar: 1 at/below safe_distance, 0 at/above free_distance.
        """
        safe = float(safe_distance)
        free = max(float(free_distance), safe + 1e-6)
        x = (clearance - safe) / (free - safe)
        x = torch.clamp(x, 0.0, 1.0)
        return 1.0 - x

    def _compute_passable_guidance(
        self,
        aff_map: torch.Tensor,
        goal_local: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if aff_map.ndim != 4 or aff_map.size(1) < 2:
            zeros = torch.zeros(aff_map.shape[0], device=aff_map.device)
            return torch.zeros(aff_map.shape[0], 2, device=aff_map.device), zeros, zeros, zeros

        occ = aff_map[:, 0]
        passable = aff_map[:, 1]

        visible = self.affordance_visible_mask
        if visible is None:
            visible = torch.ones_like(passable[0], dtype=torch.bool)
        if visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        bearing_map = self.affordance_bearing_map
        if x_map.device != aff_map.device:
            x_map = x_map.to(aff_map.device)
            y_map = y_map.to(aff_map.device)
            bearing_map = bearing_map.to(aff_map.device)

        visible_f = visible.float()
        passable_vis = passable * visible_f
        if block_mask is not None:
            passable_vis = passable_vis * (1.0 - block_mask)
        right_mask = ((x_map > 0.0) & visible).float()
        left_mask = ((x_map < 0.0) & visible).float()
        right_pass = (passable_vis * right_mask).sum(dim=(1, 2))
        left_pass = (passable_vis * left_mask).sum(dim=(1, 2))
        passable_side = (right_pass - left_pass) / (right_pass + left_pass + 1e-6)
        dir_x = (passable_vis * x_map).sum(dim=(1, 2))
        dir_y = (passable_vis * y_map).sum(dim=(1, 2))
        pass_dir = torch.stack([dir_x, dir_y], dim=1)
        pass_norm = torch.norm(pass_dir, dim=-1, keepdim=True)
        goal_norm = torch.norm(goal_local, dim=-1, keepdim=True)
        goal_dir = goal_local / (goal_norm + 1e-6)
        pass_dir = torch.where(pass_norm > 1e-6, pass_dir / (pass_norm + 1e-6), goal_dir)

        sector_deg = 0.0
        if self.reward_cfg is not None:
            sector_deg = float(getattr(self.reward_cfg, "passable_sector_deg", 0.0))
        sector_half = math.radians(sector_deg) * 0.5 if sector_deg > 0.0 else 0.0
        if sector_half > 0.0:
            goal_bearing = torch.atan2(goal_local[:, 0], goal_local[:, 1])
            angle = torch.atan2(
                torch.sin(bearing_map.unsqueeze(0) - goal_bearing.view(-1, 1, 1)),
                torch.cos(bearing_map.unsqueeze(0) - goal_bearing.view(-1, 1, 1)),
            )
            sector_mask = torch.abs(angle) <= sector_half
            sector_mask = sector_mask & visible.unsqueeze(0)
        else:
            sector_mask = visible.unsqueeze(0)

        sector_f = sector_mask.float()
        occ_ratio = (occ * sector_f).sum(dim=(1, 2)) / (sector_f.sum(dim=(1, 2)) + 1e-6)
        occ_low = 0.0
        occ_high = 1.0
        if self.reward_cfg is not None:
            occ_low = float(getattr(self.reward_cfg, "passable_occ_ratio_low", 0.0))
            occ_high = float(getattr(self.reward_cfg, "passable_occ_ratio_high", 1.0))
        if occ_high > occ_low:
            gate = torch.clamp((occ_ratio - occ_low) / (occ_high - occ_low), 0.0, 1.0)
        else:
            gate = torch.zeros_like(occ_ratio)

        return pass_dir, gate, occ_ratio, passable_side

    def _compute_low_obstacle_guidance(
        self,
        aff_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if aff_map.ndim != 4 or aff_map.size(1) < 3:
            zeros = torch.zeros(aff_map.shape[0], device=aff_map.device)
            return torch.zeros(aff_map.shape[0], 2, device=aff_map.device), zeros, zeros, None

        low_obs = aff_map[:, 2] > 0.5
        visible = self.affordance_visible_mask
        if visible is None:
            visible = torch.ones_like(low_obs[0], dtype=torch.bool)
        if visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        bearing_map = self.affordance_bearing_map
        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        if bearing_map.device != aff_map.device:
            bearing_map = bearing_map.to(aff_map.device)
            x_map = x_map.to(aff_map.device)
            y_map = y_map.to(aff_map.device)

        sector_half = math.radians(self.crossable_sector_deg) * 0.5 if self.crossable_sector_deg > 0.0 else 0.0
        if sector_half > 0.0:
            sector_mask = torch.abs(bearing_map) <= sector_half
            sector_mask = sector_mask & visible
        else:
            sector_mask = visible

        mask = low_obs & sector_mask
        valid = mask.flatten(1).any(dim=1)

        x_map_b = x_map.unsqueeze(0)
        y_map_b = y_map.unsqueeze(0)
        big = torch.full_like(x_map_b, 1e6)
        neg = torch.full_like(x_map_b, -1e6)

        x_min = torch.where(mask, x_map_b, big).amin(dim=(1, 2))
        x_max = torch.where(mask, x_map_b, neg).amax(dim=(1, 2))
        width = torch.where(valid, x_max - x_min, torch.zeros_like(x_min))

        count = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        center_x = torch.where(valid, (x_max + x_min) * 0.5, torch.zeros_like(x_min))
        center_y = torch.where(valid, (mask.float() * y_map_b).sum(dim=(1, 2)) / count, torch.zeros_like(x_min))
        center_dir = torch.stack([center_x, center_y], dim=1)
        center_norm = torch.norm(center_dir, dim=-1, keepdim=True)
        center_dir = torch.where(center_norm > 1e-6, center_dir / center_norm, torch.zeros_like(center_dir))

        width_gate = width < self.crossable_width
        gate = valid & width_gate
        block_mask = mask.float() * (~width_gate).float().view(-1, 1, 1)

        return center_dir, gate.float(), width, block_mask

    def _build_gt_affordance_map(self) -> Optional[torch.Tensor]:
        if hasattr(self.env, 'get_affordance_data'):
            aff_data = self.env.get_affordance_data()
            return aff_data['local_affordance']

        if self.use_actor_only_gt_affordance:
            aff_map = self._compute_gt_affordance_from_scene()
            if aff_map is None and self.require_actor_only_gt_affordance:
                raise RuntimeError(
                    "actor-only GT affordance is required for this task, "
                    "but scene rasterization returned None."
                )
            return aff_map

        aff_height = self._compute_gt_affordance_from_heightfield()
        aff_scene = self._compute_gt_affordance_from_scene()
        return self._merge_affordance_maps(aff_height, aff_scene)

    def _get_high_level_obs(self) -> Dict[str, torch.Tensor]:
        """
        构建高层观测字典
        
        Returns:
            Dict with keys:
            - state: (N, 9+1) robot state + prev_gate_y
            - goal: (N, 2) relative goal
            - gt_affordance: (N, 3, 16, 16) ground truth affordance
            - local_map_2ch: (N, 2, 16, 16) avoid local map
            - gt_difficulty: (N,) difficulty derived from passable_gap
            - depth: (N, 1, H, W) depth image
        """
        obs_dict = {}
        
        # 1. Robot State (V3.6: 9 维)
        if hasattr(self.env, 'robot_state_buf'):
            obs_dict['state'] = self.env.robot_state_buf.clone()
        else:
            # 手动构建
            pos = self.env.root_states[:, :2]
            quat = self.env.root_states[:, 3:7]
            yaw = self._quat_to_yaw(quat)
            lin_vel = self.env.base_lin_vel[:, :2]
            ang_vel = self.env.base_ang_vel[:, 2:3]
            height = self.env.root_states[:, 2:3]
            roll, pitch = self._quat_to_rp(quat)
            
            obs_dict['state'] = torch.cat([
                pos, yaw.unsqueeze(-1), lin_vel, ang_vel, 
                height, roll.unsqueeze(-1), pitch.unsqueeze(-1)
            ], dim=-1)

        # Align yaw to policy heading convention (+Y forward) via heading_offset_rad.
        offset = 0.0
        if self.reward_cfg is not None:
            offset = float(getattr(self.reward_cfg, "heading_offset_rad", 0.0))
            if offset != 0.0:
                state = obs_dict['state']
                yaw = state[:, 2] + offset
                yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
                state = state.clone()
                state[:, 2] = yaw
                obs_dict['state'] = state
        
        # 1.5 添加上一时刻 gate_y（解决门控滤波的非马尔可夫性）
        if hasattr(self, "prev_gate_y"):
            prev_gate_y = self.prev_gate_y.unsqueeze(1)
            obs_dict['state'] = torch.cat([obs_dict['state'], prev_gate_y], dim=1)
        # 1.6 添加上一时刻实际执行命令（解决后处理变化率记忆的非马尔可夫性）
        if hasattr(self, "post_processor") and getattr(self.post_processor, "last_cmd", None) is not None:
            obs_dict['state'] = torch.cat([obs_dict['state'], self.post_processor.last_cmd.detach().clone()], dim=1)
        if bool(getattr(self.env, "s_avoid_enabled", False)) and hasattr(self, "forced_forward_speed"):
            obs_dict['state'] = torch.cat(
                [obs_dict['state'], self.forced_forward_speed.detach().clone().unsqueeze(1)],
                dim=1,
            )

        # 2. Goal (相对坐标)
        if hasattr(self.env, 'goal_buf'):
            obs_dict['goal'] = self.env.goal_buf.clone()
        else:
            obs_dict['goal'] = self.env.commands[:, :2].clone()
        if offset != 0.0:
            goal = obs_dict['goal']
            cos_o = math.cos(offset)
            sin_o = math.sin(offset)
            goal_x = cos_o * goal[:, 0] - sin_o * goal[:, 1]
            goal_y = sin_o * goal[:, 0] + cos_o * goal[:, 1]
            obs_dict['goal'] = torch.stack([goal_x, goal_y], dim=1)
        
        # 3. GT Affordance
        aff_map = self._build_gt_affordance_map()
        if aff_map is not None:
            obs_dict['gt_affordance'] = aff_map
        elif hasattr(self.env, 'measured_heights'):
            heights = self.env.measured_heights
            side_len = int(np.sqrt(heights.shape[1]))

            if side_len ** 2 == heights.shape[1]:
                h_map = heights.view(self.num_envs, 1, side_len, side_len)
                h_map = torch.nn.functional.interpolate(
                    h_map,
                    size=(self.affordance_map_size, self.affordance_map_size),
                    mode='bilinear',
                )
                low = torch.zeros_like(h_map)
                obs_dict['gt_affordance'] = torch.cat([h_map, 1 - torch.abs(h_map), low], dim=1)
            else:
                obs_dict['gt_affordance'] = torch.zeros(
                    self.num_envs,
                    3,
                    self.affordance_map_size,
                    self.affordance_map_size,
                    device=self.device,
                )
        else:
            obs_dict['gt_affordance'] = torch.zeros(
                self.num_envs,
                3,
                self.affordance_map_size,
                self.affordance_map_size,
                device=self.device,
            )
        obs_dict['critic_local_map_2ch'] = build_avoid_local_map_2ch(
            obs_dict['gt_affordance'],
            visible_mask=None,
        )
        obs_dict['local_map_2ch'] = build_avoid_local_map_2ch(
            obs_dict['gt_affordance'],
            visible_mask=self.affordance_visible_mask,
        )
        actor_difficulty = self._compute_objective_difficulty_from_local_map(
            obs_dict['local_map_2ch']
        )
        critic_difficulty = self._compute_objective_difficulty_from_local_map(
            obs_dict['critic_local_map_2ch']
        )
        obs_dict['gt_difficulty'] = difficulty_from_gap(obs_dict['gt_affordance'])
        obs_dict['actor_difficulty'] = actor_difficulty
        obs_dict['critic_difficulty'] = critic_difficulty
        
        # 4. Depth Image
        if hasattr(self.env, 'depth_images'):
            obs_dict['depth'] = self.env.depth_images.clone()
        elif hasattr(self.env, 'depth_buffer'):
            obs_dict['depth'] = self.env.depth_buffer.unsqueeze(1).clone()
        else:
            obs_dict['depth'] = torch.zeros(self.num_envs, 1, 128, 128, device=self.device)
        
        return obs_dict

    def step(
        self,
        cmd_vel: torch.Tensor,
        gate_y: Optional[torch.Tensor] = None,
        *,
        gate_y_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        高层步进 (10Hz)

        Args:
            cmd_vel: (N, 3) [vx, vy, omega] 高层命令
            gate_y: (N,) 当前时刻执行门控权重（仅 Gate 训练使用）
            gate_y_raw: (N,) 当前时刻原始 gate 输出；用于 history 和 gate_smooth

        Returns:
            obs_dict, rewards, dones, info
        """
        # Debug: force a constant forward command along +Y to validate command direction/axis.
        if bool(getattr(self.args, "force_cmd_y", False)):
            v = 0.3
            if cmd_vel is None:
                cmd_vel = torch.zeros(self.num_envs, 3, device=self.device)
            else:
                cmd_vel = cmd_vel.detach()
            cmd_vel = torch.zeros_like(cmd_vel)
            cmd_vel[:, 1] = float(v)
        gate_y_exec = None if gate_y is None else gate_y.to(self.device)
        gate_y_raw_curr = gate_y_exec if gate_y_raw is None else gate_y_raw.to(self.device)

        # 1. Command Post-Processor
        clearance_pp = None
        clearance_aff_map = None
        cmd_preview = None
        cmd_xy_for_clearance = None
        if cmd_vel is not None:
            cmd_preview = self.post_processor.preview_cmd_before_risk(
                cmd_vel,
                beta=self.beta_override,
            )
            cmd_xy_for_clearance = cmd_preview[:, :2]
        if self.clearance_affordance_override is not None:
            clearance_aff_map = self.clearance_affordance_override
        elif self.last_obs is not None and 'gt_affordance' in self.last_obs:
            clearance_aff_map = self.last_obs['gt_affordance']
        if clearance_aff_map is not None and cmd_xy_for_clearance is not None:
            clearance_pp = self._compute_clearance_along_cmd(clearance_aff_map, cmd_xy_for_clearance)
        velocity_cmd, post_info = self.post_processor.process(
            cmd_vel, clearance_pp, beta=self.beta_override
        )
        velocity_cmd_post = velocity_cmd.detach().clone()
        rotate_only_trigger_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(getattr(self.env, "s_avoid_enabled", False)):
            velocity_cmd = velocity_cmd.clone()
            velocity_cmd[:, 1] = torch.clamp(velocity_cmd[:, 1], min=self.forced_forward_speed)
            nav_cfg = getattr(self.env.cfg, "navigation", None)
            yaw_on_deg = float(getattr(nav_cfg, "rotate_only_yaw_on_deg", 15.0))
            yaw_off_deg = float(getattr(nav_cfg, "rotate_only_yaw_off_deg", 8.0))
            yaw_gain = float(getattr(nav_cfg, "rotate_only_yaw_gain", 0.8))
            rotate_gap_margin = float(getattr(nav_cfg, "rotate_only_gap_margin", 0.08))
            row_gate_on = float(getattr(self.reward_cfg, "avoid_row_gate_on", 1.0))
            robot_pos_now = self.env.root_states[:, :3]
            (
                _gap_center_now,
                _row_y_now,
                _gap_valid_now,
                _gap_left_now,
                _gap_right_now,
                gap_left_eff_now,
                gap_right_eff_now,
                gap_eff_valid_now,
            ) = self._compute_nearest_row_gap_target(robot_pos_now)
            row_err_abs_now = self._distance_to_interval(robot_pos_now[:, 0], gap_left_eff_now, gap_right_eff_now)
            row_forward_dist_now, row_forward_valid_now = self._compute_nearest_row_forward_distance(robot_pos_now)
            rotate_in_gap = gap_eff_valid_now & (row_err_abs_now <= rotate_gap_margin)
            rotate_row_released = row_forward_valid_now & (row_forward_dist_now >= row_gate_on)
            rotate_allow = rotate_in_gap | rotate_row_released
            quat_now = self.env.root_states[:, 3:7]
            yaw_now = self._quat_to_yaw(quat_now)
            heading_offset = float(getattr(self.reward_cfg, "heading_offset_rad", 0.0))
            yaw_error = torch.atan2(
                torch.sin(yaw_now + heading_offset),
                torch.cos(yaw_now + heading_offset),
            )
            yaw_deg = torch.abs(yaw_error) * (180.0 / math.pi)
            rotate_only_active = (
                (yaw_deg > yaw_on_deg)
                | (self.rotate_only_active & (yaw_deg > yaw_off_deg))
            ) & rotate_allow
            rotate_only_trigger_event = rotate_only_active & (~self.rotate_only_active)
            omega_cmd = torch.clamp(
                -yaw_gain * yaw_error,
                min=-float(getattr(nav_cfg, "max_ang_vel_command", 0.3)),
                max=float(getattr(nav_cfg, "max_ang_vel_command", 0.3)),
            )
            velocity_cmd[rotate_only_active, 0] = 0.0
            velocity_cmd[rotate_only_active, 1] = 0.0
            velocity_cmd[rotate_only_active, 2] = omega_cmd[rotate_only_active]
            self.rotate_only_active = rotate_only_active.detach().clone()
            if getattr(self.post_processor, "last_cmd", None) is not None:
                self.post_processor.last_cmd = velocity_cmd.detach().clone()
        if clearance_aff_map is not None:
            cmd_preview = velocity_cmd.detach()
            clearance_pp = self._compute_clearance_along_cmd(clearance_aff_map, cmd_preview[:, :2])
        post_info_payload = {}
        if isinstance(post_info, dict):
            post_info_payload.update(post_info)
        if cmd_preview is not None:
            post_info_payload["cmd_preview_for_clearance"] = cmd_preview.detach().clone()
        post_info_payload.setdefault("cmd_raw", cmd_vel)
        post_info_payload["cmd_post"] = velocity_cmd_post.detach().clone()
        post_info_payload["cmd_override_final"] = velocity_cmd.detach().clone()
        post_info_payload["cmd_slew"] = velocity_cmd_post.detach().clone()
        post_info_payload["cmd_clamped"] = velocity_cmd_post.detach().clone()
        if bool(getattr(self.env, "s_avoid_enabled", False)):
            post_info_payload["forced_forward_speed"] = self.forced_forward_speed.detach().clone()
            post_info_payload["rotate_only_active"] = self.rotate_only_active.detach().clone()
        post_info_payload["clearance_pp"] = clearance_pp
        # Prefer per-call effective params (beta-adjusted) when available.
        safe_distance = post_info_payload.get("safe_distance", getattr(self.post_processor, "safe_distance", None))
        free_distance = post_info_payload.get("free_distance", getattr(self.post_processor, "free_distance", None))
        max_cmd = post_info_payload.get("max_cmd", getattr(self.post_processor, "max_cmd", None))
        max_delta = post_info_payload.get("max_delta", getattr(self.post_processor, "max_delta", None))
        risk_clamp_gain = post_info_payload.get("risk_clamp_gain", None)

        if safe_distance is not None:
            post_info_payload["post_safe_distance"] = (
                float(safe_distance.item()) if torch.is_tensor(safe_distance) else float(safe_distance)
            )
        if free_distance is not None:
            post_info_payload["post_free_distance"] = (
                float(free_distance.item()) if torch.is_tensor(free_distance) else float(free_distance)
            )
        if max_cmd is not None:
            if torch.is_tensor(max_cmd):
                post_info_payload["post_max_cmd"] = max_cmd.detach().clone()
            else:
                post_info_payload["post_max_cmd"] = torch.as_tensor(max_cmd, device=self.device)
        if max_delta is not None:
            if torch.is_tensor(max_delta):
                post_info_payload["post_max_delta"] = max_delta.detach().clone()
            else:
                post_info_payload["post_max_delta"] = torch.as_tensor(max_delta, device=self.device)
        if risk_clamp_gain is not None:
            post_info_payload["post_risk_clamp_gain"] = (
                float(risk_clamp_gain.item()) if torch.is_tensor(risk_clamp_gain) else float(risk_clamp_gain)
            )

        # 2. Low-Level 控制循环 (50Hz)
        accumulated_reward = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        cmd_exec_sum = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_exec_steps = torch.zeros(self.num_envs, device=self.device)
        cmd_exec_last = torch.zeros(self.num_envs, 3, device=self.device)

        for _ in range(self.decimation):
            if not active_mask.any():
                break
            if hasattr(self.env, "commands"):
                active_before = active_mask.float()
                velocity_cmd_step = velocity_cmd.detach()
                if (~active_mask).any():
                    velocity_cmd_step = velocity_cmd_step.clone()
                    velocity_cmd_step[~active_mask] = 0.0
                self.env.commands[:, :3] = velocity_cmd_step
                cmd_exec_sum += velocity_cmd_step
                cmd_exec_steps += active_before
                cmd_exec_last = velocity_cmd_step
                if hasattr(self.env, "commands_scale") and hasattr(self.env, "obs_buf"):
                    if self.env.obs_buf.shape[1] >= 3:
                        self.env.obs_buf[:, -3:] = velocity_cmd_step * self.env.commands_scale

            low_level_obs = self.env.obs_buf

            with torch.no_grad():
                actions = self.low_level_policy.act_inference(low_level_obs)
            if (~active_mask).any():
                actions = actions.clone()
                actions[~active_mask] = 0.0

            obs, _, rewards, dones, infos = self.env.step(actions)
            rewards = rewards * active_mask.float()
            accumulated_reward += rewards
            done_any |= dones
            active_mask &= ~dones
        cmd_exec_mean = cmd_exec_sum / cmd_exec_steps.clamp_min(1.0).unsqueeze(-1)
        post_info_payload["cmd_exec_mean"] = cmd_exec_mean.detach().clone()
        post_info_payload["cmd_exec_last"] = cmd_exec_last.detach().clone()
        post_info_payload["cmd_exec_steps"] = cmd_exec_steps.detach().clone()
        
        self.episode_length_buf += 1
        length_snapshot = self.episode_length_buf.clone()

        done_during = done_any.clone()
        manual_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if hasattr(self.env, "goal_world"):
            current_goal_world = self.env.goal_world.clone()
            if self.prev_goal_world is None:
                self.prev_goal_world = current_goal_world.clone()
            goal_delta = torch.norm(current_goal_world - self.prev_goal_world, dim=1)
            goal_changed = goal_delta > 1e-3
            if done_during.any():
                goal_changed = goal_changed & (~done_during)
            self.goal_change_count += goal_changed.long()
            self.prev_goal_world = current_goal_world
        
        # 3. 计算高层奖励
        self._refresh_depth_images()
        reward_obs = self._get_high_level_obs()
        self.last_obs = reward_obs
        robot_pos = self.env.root_states[:, :3]
        
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision_force_max = torch.zeros(self.num_envs, device=self.device)
        collision_threshold = None
        collision_threshold_src = None
        collision_indices_src = None
        contact_norm = None
        collision_debug_indices = None
        obstacle_contact_candidate_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        strict_obstacle_contact_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        obstacle_contact_candidate_min_clearance = torch.full(
            (self.num_envs,), 5.0, dtype=torch.float32, device=self.device
        )
        avoid_band_active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        band_fail_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        nearest_obs = None
        front_half_nearest_obs = None
        front_left_nearest_obs = None
        front_right_nearest_obs = None
        avoid_band_outside_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        clearance = None
        clearance_global = None
        forward_clearance = None
        reward_aff_map = reward_obs['gt_affordance'] if reward_obs is not None else None
        if getattr(self, "reward_affordance_override", None) is not None:
            reward_aff_map = self.reward_affordance_override
            self.reward_affordance_override = None
        if hasattr(self.env, "contact_forces"):
            collision_threshold = getattr(self.env.cfg.terrain, "collision_force_threshold", 1.0)
            collision_threshold_src = "collision_force_threshold"
            indices = getattr(self.env, "penalised_contact_indices", None)
            collision_indices_src = "penalised_contact_indices"
            if indices is not None and indices.numel() > 0:
                penalized_contact_norm = torch.norm(self.env.contact_forces[:, indices, :], dim=-1)
                collision_mask = torch.any(penalized_contact_norm > collision_threshold, dim=1)
                collision_debug_indices = indices
                if hasattr(self.env, "termination_contact_indices"):
                    term_indices = getattr(self.env, "termination_contact_indices", None)
                    if term_indices is not None and term_indices.numel() > 0:
                        collision_debug_indices = torch.unique(
                            torch.cat([indices.to(torch.long), term_indices.to(torch.long)], dim=0),
                            sorted=True,
                        ).to(dtype=torch.long)
                        collision_indices_src = "penalised+termination_contact_indices"
                contact_norm = torch.norm(self.env.contact_forces[:, collision_debug_indices, :], dim=-1)
                collision_force_max = contact_norm.max(dim=1).values
                if bool(getattr(self.env, "s_avoid_enabled", False)):
                    obstacle_contact_margin = float(
                        getattr(self.env.cfg.terrain, "avoid_obstacle_contact_margin", 0.03)
                    )
                    strict_contact_margin = float(
                        getattr(self.env.cfg.terrain, "avoid_strict_contact_margin", 0.01)
                    )
                    nearest_obs_fn = getattr(self.env, "_compute_s_avoid_nearest_obstacle_distance", None)
                    nearest_obs = nearest_obs_fn() if callable(nearest_obs_fn) else None
                    if nearest_obs is not None:
                        robot_hull_clearance = nearest_obs - float(getattr(self, "affordance_clearance", 0.27))
                        obstacle_contact_candidate_min_clearance = torch.nan_to_num(
                            robot_hull_clearance,
                            nan=5.0,
                            posinf=5.0,
                            neginf=-5.0,
                        )
                        strict_obstacle_contact_mask = (
                            obstacle_contact_candidate_min_clearance <= strict_contact_margin
                        )
                        obstacle_contact_candidate_mask = (
                            obstacle_contact_candidate_min_clearance <= obstacle_contact_margin
                        )
                        collision_mask = collision_mask | strict_obstacle_contact_mask
        if self.clearance_override is not None:
            clearance = self.clearance_override
            self.clearance_override = None
        elif reward_aff_map is not None:
            clearance_global = self._compute_clearance_from_affordance(reward_aff_map)
            clearance = self._compute_clearance_along_cmd(reward_aff_map, cmd_exec_mean[:, :2])
            if bool(getattr(self.env, "s_avoid_enabled", False)):
                forward_cmd_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=cmd_exec_mean.dtype)
                forward_cmd_xy[:, 1] = 1.0
                forward_clearance = self._compute_clearance_along_cmd(reward_aff_map, forward_cmd_xy)
                front_half_nearest_obs = self._compute_front_half_obstacle_distance(reward_aff_map)
                front_left_nearest_obs, front_right_nearest_obs = self._compute_front_side_obstacle_distance(
                    reward_aff_map
                )
        self.clearance_affordance_override = None
        passable_dir = None
        passable_gate = None
        passable_occ_ratio = None
        passable_side = None
        crossable_dir = None
        crossable_gate = None
        crossable_width = None
        low_block_mask = None
        if (
            self.reward_cfg is not None
            and (
                getattr(self.reward_cfg, "passable_align_scale", 0.0) != 0.0
                or getattr(self.reward_cfg, "crossable_align_scale", 0.0) != 0.0
                or bool(getattr(self.env, "s_avoid_enabled", False))
            )
            and reward_aff_map is not None
        ):
            crossable_dir, crossable_gate, crossable_width, low_block_mask = self._compute_low_obstacle_guidance(
                reward_aff_map
            )
            passable_dir, passable_gate, passable_occ_ratio, passable_side = self._compute_passable_guidance(
                reward_aff_map,
                reward_obs['goal'],
                block_mask=low_block_mask,
            )

        reward_terms = None
        cmd_pred_x = torch.zeros(self.num_envs, device=self.device, dtype=cmd_exec_mean.dtype)
        if gate_y_exec is None:
            gate_y_exec = torch.zeros(self.num_envs, device=self.device)
        if gate_y_raw_curr is None:
            gate_y_raw_curr = gate_y_exec
        if self.reward_func is not None:
            # Use world-frame linear velocity to match world-frame goal direction in reward terms.
            robot_vel = self.env.root_states[:, 7:10]
            robot_quat = self.env.root_states[:, 3:7]
            if hasattr(self.env, "goal_world"):
                goal_pos = self.env.goal_world.clone()
            else:
                yaw_policy = reward_obs['state'][:, 2]
                cos_h = torch.cos(yaw_policy)
                sin_h = torch.sin(yaw_policy)
                goal_local = reward_obs['goal']
                goal_x = robot_pos[:, 0] + cos_h * goal_local[:, 0] - sin_h * goal_local[:, 1]
                goal_y = robot_pos[:, 1] + sin_h * goal_local[:, 0] + cos_h * goal_local[:, 1]
                goal_pos = torch.stack([goal_x, goal_y], dim=1)
            reward_difficulty = reward_obs['gt_difficulty']
            if 'critic_difficulty' in reward_obs:
                reward_difficulty = reward_obs['critic_difficulty']
            
            reward_dict = self.reward_func.compute_reward(
                robot_pos=robot_pos,
                prev_robot_pos=self.prev_robot_pos,
                goal_pos=goal_pos,
                robot_vel=robot_vel,
                robot_vel_body=self.env.base_lin_vel,
                robot_quat=robot_quat,
                gate_y=gate_y_raw_curr,
                prev_gate_y=self.prev_gate_y,
                terrain_difficulty=reward_difficulty,
                collision_mask=collision_mask,
                clearance=clearance,
                cmd_xy=cmd_exec_mean[:, :2],
                cmd_omega=cmd_exec_mean[:, 2],
                passable_dir=passable_dir,
                passable_gate=passable_gate,
                crossable_dir=crossable_dir,
                crossable_gate=crossable_gate,
                risk_barrier_safe_override=post_info_payload.get("post_safe_distance", None),
                risk_barrier_free_override=post_info_payload.get("post_free_distance", None),
            )
            if clearance is not None:
                reward_dict['clearance'] = clearance
            if clearance_global is not None:
                reward_dict['clearance_global'] = clearance_global
            if passable_occ_ratio is not None:
                reward_dict['passable_occ_ratio'] = passable_occ_ratio
            if crossable_width is not None:
                reward_dict['crossable_width'] = crossable_width
            if bool(getattr(self.env, "s_avoid_enabled", False)):
                stage_ids = getattr(self.env, "s_avoid_stage_per_env", None)
                if stage_ids is None:
                    stage_ids = torch.full(
                        (self.num_envs,),
                        int(getattr(self.env, "s_avoid_stage", 1)),
                        device=self.device,
                        dtype=torch.long,
                    )
                else:
                    stage_ids = stage_ids.to(device=self.device, dtype=torch.long)
                cross_line_local_y = torch.zeros(self.num_envs, device=self.device)
                for stage_value in torch.unique(stage_ids).tolist():
                    stage_id = int(stage_value)
                    last_row_y = float(
                        getattr(self.env.cfg.terrain, f"avoid_stage{stage_id}_last_row_y", 2.0)
                    )
                    cross_line_local_y[stage_ids == stage_id] = last_row_y
                cross_line_world_y = self.env.env_origins[:, 1] + cross_line_local_y
                curr_center_y = robot_pos[:, 1]
                prev_center_y = self.prev_robot_pos[:, 1]
                curr_dist = torch.clamp(cross_line_world_y - curr_center_y, min=0.0)
                prev_dist = torch.clamp(cross_line_world_y - prev_center_y, min=0.0)
                approach_progress = (prev_dist - curr_dist) * float(self.reward_cfg.goal_approach_scale)
                prev_total = reward_dict['total']
                prev_approach = reward_dict.get('approach', torch.zeros_like(approach_progress))
                reward_dict['cross_line_dist'] = curr_dist
                reward_dict['approach'] = approach_progress
                reward_dict['total'] = prev_total - prev_approach + approach_progress

                forward_progress_speed = torch.clamp(
                    (prev_dist - curr_dist) / max(float(self.high_level_dt), 1e-6),
                    min=0.0,
                )
                velocity_reward = forward_progress_speed * float(self.reward_cfg.velocity_scale)
                prev_velocity = reward_dict.get('velocity', torch.zeros_like(velocity_reward))
                reward_dict['velocity'] = velocity_reward
                reward_dict['total'] = reward_dict['total'] - prev_velocity + velocity_reward

                heading_error = reward_obs['state'][:, 2]
                heading_weight = 1.0
                if self.reward_cfg.heading_use_difficulty_gate:
                    heading_weight = torch.clamp(
                        1.0 - reward_difficulty,
                        min=self.reward_cfg.heading_min_weight,
                        max=1.0,
                    )
                heading_reward = (
                    (torch.cos(heading_error) - 0.15)
                    * self.reward_cfg.heading_scale
                    * heading_weight
                )
                if self.reward_cfg.heading_gate_use:
                    progress = prev_dist - curr_dist
                    gate = progress > self.reward_cfg.heading_gate_min_approach
                    if self.reward_cfg.heading_gate_min_speed > 0:
                        speed = torch.norm(robot_vel[:, :2], dim=-1)
                        gate = gate & (speed > self.reward_cfg.heading_gate_min_speed)
                    heading_reward = heading_reward * gate.float()
                prev_heading = reward_dict.get('heading', torch.zeros_like(heading_reward))
                reward_dict['heading'] = heading_reward
                reward_dict['total'] = reward_dict['total'] - prev_heading + heading_reward
                row_lat_scale = float(getattr(self.reward_cfg, "avoid_row_lat_scale", 6.0))
                (
                    _gap_center,
                    _row_y,
                    _gap_valid,
                    _gap_left,
                    _gap_right,
                    gap_left_eff,
                    gap_right_eff,
                    gap_eff_valid,
                ) = self._compute_nearest_row_gap_target(robot_pos)
                row_err_abs = self._distance_to_interval(robot_pos[:, 0], gap_left_eff, gap_right_eff)
                prev_row_err_abs = self._distance_to_interval(self.prev_robot_pos[:, 0], gap_left_eff, gap_right_eff)
                row_lat_reward = (
                    row_lat_scale
                    * (prev_row_err_abs - row_err_abs)
                    * gap_eff_valid.float()
                )
                row_gap_scale = float(getattr(self.reward_cfg, "avoid_row_gap_scale", 8.0))
                row_push_scale = float(getattr(self.reward_cfg, "avoid_row_push_scale", 1.5))
                row_push_margin = float(getattr(self.reward_cfg, "avoid_row_push_margin", 0.25))
                row_cmdx_scale = float(getattr(self.reward_cfg, "avoid_row_cmdx_scale", 0.5))
                row_gate_on = float(getattr(self.reward_cfg, "avoid_row_gate_on", 1.0))
                row_gate_full = float(getattr(self.reward_cfg, "avoid_row_gate_full", 0.4))
                row_forward_dist, row_forward_valid = self._compute_nearest_row_forward_distance(robot_pos)
                gate_row = torch.clamp(
                    (row_gate_on - row_forward_dist) / max(row_gate_on - row_gate_full, 1e-6),
                    min=0.0,
                    max=1.0,
                )
                row_gap_reward = (
                    row_gap_scale
                    * gate_row
                    * (prev_row_err_abs - row_err_abs)
                    * row_forward_valid.float()
                    * gap_eff_valid.float()
                )
                push_err = torch.clamp(
                    row_err_abs / max(row_push_margin, 1e-6),
                    min=0.0,
                    max=1.0,
                )
                row_push_penalty = (
                    -row_push_scale
                    * gate_row
                    * push_err
                    * row_forward_valid.float()
                    * gap_eff_valid.float()
                )
                gap_center_eff = 0.5 * (gap_left_eff + gap_right_eff)
                x_dir_to_gap = torch.sign(gap_center_eff - robot_pos[:, 0])
                out_gap = (row_err_abs > 0).float() * gap_eff_valid.float()
                cmd_raw = post_info_payload.get("cmd_raw", cmd_vel)
                if cmd_raw is None:
                    cmd_raw = torch.zeros_like(cmd_exec_mean)
                elif not torch.is_tensor(cmd_raw):
                    cmd_raw = torch.as_tensor(cmd_raw, device=self.device, dtype=cmd_exec_mean.dtype)
                if cmd_raw.dim() == 1:
                    cmd_raw = cmd_raw.unsqueeze(0)
                cmd_pred_x = cmd_raw[:, 0]
                cmd_x_signed = cmd_pred_x * x_dir_to_gap
                row_cmdx_reward = (
                    row_cmdx_scale
                    * gate_row
                    * out_gap
                    * cmd_x_signed
                    * row_forward_valid.float()
                )
                avoid_smooth_scale = float(getattr(self.reward_cfg, "avoid_smooth_scale", 0.0))
                avoid_smooth_switch_threshold = float(
                    getattr(self.reward_cfg, "avoid_smooth_switch_threshold", 0.05)
                )
                avoid_smooth_switch_unit = float(
                    getattr(self.reward_cfg, "avoid_smooth_switch_unit", 0.001)
                )
                avoid_early_next_gap_threshold = float(
                    getattr(self.reward_cfg, "avoid_early_next_gap_threshold", 0.30)
                )
                avoid_early_next_gap_scale = float(
                    getattr(self.reward_cfg, "avoid_early_next_gap_scale", 1.0)
                )
                heading_event_penalty = float(getattr(self.reward_cfg, "avoid_heading_event_penalty", 0.0))
                row_cmdx_log_mask = row_forward_valid.float() * gap_eff_valid.float()
                row_gate_valid = gate_row * row_forward_valid.float() * gap_eff_valid.float()
                row_gate_active = (
                    (gate_row > 0.05)
                    & row_forward_valid
                    & gap_eff_valid
                ).float()
                row_push_active = (
                    (gate_row > 0.05)
                    & (push_err > 0.05)
                    & row_forward_valid
                    & gap_eff_valid
                ).float()
                row_forward_dist_log = row_forward_dist * row_forward_valid.float() * gap_eff_valid.float()
                inside_gap = (
                    (row_err_abs <= 1e-3)
                    & row_forward_valid
                    & gap_eff_valid
                )
                smooth_switch_valid = (
                    inside_gap
                    & (torch.abs(cmd_pred_x) > avoid_smooth_switch_threshold)
                    & (torch.abs(self.prev_cmd_pred_x) > avoid_smooth_switch_threshold)
                )
                smooth_switch_event = smooth_switch_valid & ((cmd_pred_x * self.prev_cmd_pred_x) < 0.0)
                avoid_smooth_penalty = (
                    -avoid_smooth_scale
                    * avoid_smooth_switch_unit
                    * gate_row
                    * smooth_switch_event.float()
                )
                (
                    current_row_gap_center,
                    current_row_front_edge,
                    current_row_back_edge,
                    current_row_valid,
                    next_row_gap_center,
                    next_row_valid,
                ) = self._compute_current_next_row_gap_state(robot_pos)
                body_half_length = float(getattr(self.env.cfg.terrain, "avoid_body_half_length", 0.0))
                row_release_offset = float(getattr(self.reward_cfg, "avoid_row_release_offset", 0.20))
                robot_front_y = robot_pos[:, 1] + body_half_length
                robot_rear_y = robot_pos[:, 1] - body_half_length
                in_row_through = (
                    current_row_valid
                    & next_row_valid
                    & (robot_front_y >= current_row_front_edge)
                    & (robot_rear_y <= (current_row_back_edge + row_release_offset))
                )
                current_row_x_dir = torch.sign(current_row_gap_center - robot_pos[:, 0])
                next_row_x_dir = torch.sign(next_row_gap_center - robot_pos[:, 0])
                next_row_conflict = (current_row_x_dir * next_row_x_dir) < 0.0
                toward_next_gap_strength = cmd_pred_x * next_row_x_dir
                early_next_gap_active = (
                    in_row_through
                    & next_row_conflict
                    & (toward_next_gap_strength > avoid_early_next_gap_threshold)
                )
                early_next_gap_penalty = (
                    -avoid_early_next_gap_scale
                    * gate_row
                    * torch.clamp(toward_next_gap_strength - avoid_early_next_gap_threshold, min=0.0)
                    * early_next_gap_active.float()
                )
                heading_keep_reward = (
                    -heading_event_penalty
                    * rotate_only_trigger_event.float()
                )
                reward_dict['row_lat'] = row_lat_reward
                reward_dict['row_gap'] = row_gap_reward
                reward_dict['row_push_penalty'] = row_push_penalty
                reward_dict['row_cmdx_reward'] = row_cmdx_reward
                reward_dict['avoid_smooth_penalty'] = avoid_smooth_penalty
                reward_dict['early_next_gap_penalty'] = early_next_gap_penalty
                reward_dict['heading_keep'] = heading_keep_reward
                reward_dict['row_cmdx_signed'] = cmd_x_signed * row_cmdx_log_mask
                reward_dict['row_cmdx_signed_abs'] = cmd_x_signed.abs() * row_cmdx_log_mask
                reward_dict['row_cmdx_signed_pos'] = torch.clamp(cmd_x_signed, min=0.0) * row_cmdx_log_mask
                reward_dict['row_cmdx_signed_neg'] = torch.clamp(cmd_x_signed, max=0.0) * row_cmdx_log_mask
                reward_dict['row_forward_dist'] = row_forward_dist_log
                reward_dict['row_gate'] = row_gate_valid
                reward_dict['row_gate_active'] = row_gate_active
                reward_dict['row_push_active'] = row_push_active
                reward_dict['avoid_smooth_switch_valid'] = smooth_switch_valid.float()
                reward_dict['avoid_smooth_switch_event'] = smooth_switch_event.float()
                reward_dict['early_next_gap_active'] = early_next_gap_active.float()
                reward_dict['total'] = (
                    reward_dict['total']
                    + row_lat_reward
                    + row_gap_reward
                    + row_push_penalty
                    + row_cmdx_reward
                    + avoid_smooth_penalty
                    + early_next_gap_penalty
                    + heading_keep_reward
                )
            total_reward = reward_dict['total']

            # reach_bonus 只在每个 episode 内生效一次
            if (
                self.reward_cfg is not None
                and self.reward_cfg.goal_reach_bonus != 0.0
                and self.reward_cfg.goal_reach_threshold > 0.0
            ):
                dist_to_goal = torch.norm(robot_pos[:, :2] - goal_pos, dim=-1)
                reach_now = dist_to_goal < self.reward_cfg.goal_reach_threshold
                if reach_now.any():
                    done_any |= reach_now
                    manual_reset_mask |= reach_now
                reach_mask = reach_now & (~self.reach_given)
                if reach_now.any():
                    repeated = reach_now & self.reach_given
                    total_reward = total_reward - repeated.float() * self.reward_cfg.goal_reach_bonus
                    reward_dict['reach'] = reach_mask.float() * self.reward_cfg.goal_reach_bonus
                    reward_dict['total'] = total_reward
                self.reach_given |= reach_mask
            reward_terms = reward_dict
        else:
            total_reward = accumulated_reward / self.decimation
            reward_terms = {"total": total_reward}

        avoid_band_scale = 0.0
        if getattr(self.env, "nav_cfg", None) is not None:
            avoid_band_scale = float(getattr(self.env.nav_cfg, "avoid_band_penalty_scale", 0.0))
        if (
            getattr(self.env, "s_avoid_enabled", False)
            and avoid_band_scale != 0.0
            and hasattr(self.env, "s_avoid_band_x_min")
            and hasattr(self.env, "s_avoid_spawn_world_y")
        ):
            band_x_min = self.env.s_avoid_band_x_min
            band_x_max = self.env.s_avoid_band_x_max
            band_y_min = self.env.s_avoid_band_y_min
            band_y_max = self.env.s_avoid_band_y_max
            activate_progress = float(getattr(self.env.nav_cfg, "avoid_band_activate_progress", 0.0))
            if activate_progress <= 0.0:
                avoid_band_active_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            else:
                y_progress = robot_pos[:, 1] - self.env.s_avoid_spawn_world_y
                avoid_band_active_mask = y_progress >= activate_progress
            dx_out = torch.clamp(band_x_min - robot_pos[:, 0], min=0.0) + torch.clamp(
                robot_pos[:, 0] - band_x_max, min=0.0
            )
            avoid_band_outside_dist = torch.where(
                avoid_band_active_mask,
                dx_out,
                torch.zeros_like(dx_out),
            )
            band_term = -avoid_band_scale * avoid_band_outside_dist
            total_reward = total_reward + band_term
            reward_terms["avoid_band_penalty"] = band_term
            reward_terms["total"] = total_reward

            band_fail_margin = float(getattr(self.env.nav_cfg, "avoid_band_fail_margin", 0.30))
            band_fail_mask = avoid_band_active_mask & (
                (robot_pos[:, 0] < (band_x_min - band_fail_margin))
                | (robot_pos[:, 0] > (band_x_max + band_fail_margin))
            )
            if band_fail_mask.any():
                done_any |= band_fail_mask
                manual_reset_mask |= band_fail_mask

        # Add target centering / visibility shaping when explicitly enabled.
        use_target_view_reward = (
            self.camera_fov_rad is not None
            and (
                self.moving_target_enable
                or self.target_center_scale != 0.0
                or self.target_visible_scale != 0.0
            )
        )
        if use_target_view_reward:
            goal_rel = reward_obs.get("goal", None) if isinstance(reward_obs, dict) else None
            if goal_rel is not None:
                bearing_body = torch.atan2(goal_rel[:, 0], goal_rel[:, 1])  # 0 means +Y forward
                bearing_cam = bearing_body - float(self.camera_bearing_rad)
                bearing_cam = torch.atan2(torch.sin(bearing_cam), torch.cos(bearing_cam))
                fov = float(self.camera_fov_rad)
                phys_half = 0.5 * fov
                abs_bearing = torch.abs(bearing_cam)
                phys_visible = abs_bearing <= phys_half

                if self.is_s0_follow_task:
                    difficulty_vec = self._current_scene_difficulty(dtype=abs_bearing.dtype)
                    hard_scale = 1.2 + difficulty_vec * (0.6 - 1.2)
                    soft_scale = 0.5 * hard_scale
                    hard_half_train = torch.clamp(0.5 * fov * hard_scale, min=1e-3)
                    soft_half_train = torch.clamp(0.5 * fov * soft_scale, min=1e-3)
                else:
                    soft_half_train = torch.full_like(
                        abs_bearing,
                        max(1e-3, 0.5 * fov * float(self.target_fov_soft_scale)),
                    )
                    hard_half_train = torch.full_like(
                        abs_bearing,
                        max(1e-3, 0.5 * fov * float(self.target_fov_hard_scale)),
                    )

                # Centering penalty: normalized by the soft window.
                center_margin = abs_bearing / soft_half_train
                # Prevent reward explosion (squared penalty can dominate PPO and blow up KL).
                center_margin = torch.clamp(center_margin, max=3.0)
                r_center = -(center_margin ** 2)
                # Visibility penalty: only active outside hard window.
                visible_excess = torch.clamp(abs_bearing - hard_half_train, min=0.0) / soft_half_train
                visible_excess = torch.clamp(visible_excess, max=3.0)
                r_visible = -(visible_excess ** 2)

                if self.target_center_scale != 0.0:
                    term = float(self.target_center_scale) * r_center
                    total_reward = total_reward + term
                    reward_terms["target_center"] = term
                    reward_terms["total"] = total_reward
                if self.target_visible_scale != 0.0:
                    term = float(self.target_visible_scale) * r_visible
                    total_reward = total_reward + term
                    reward_terms["target_visible"] = term
                    reward_terms["total"] = total_reward

                if self.target_lost_k > 0:
                    out_hard = ~phys_visible
                    # Ignore envs already terminated by physics this high-level step.
                    if done_during.any():
                        out_hard = out_hard & (~done_during)
                        self.target_lost_steps[done_during] = 0
                    self.target_lost_steps = torch.where(
                        out_hard,
                        self.target_lost_steps + 1,
                        torch.zeros_like(self.target_lost_steps),
                    )
                    lost = self.target_lost_steps >= int(self.target_lost_k)
                    if lost.any():
                        # Continuous penalty while target stays lost; do not force reset in S0.
                        total_reward = torch.where(lost, total_reward - 0.3, total_reward)
                        reward_terms["target_lost"] = lost.float() * (-0.3)
                        reward_terms["total"] = total_reward

                if self.is_s0_follow_task and hasattr(self.env, "target_world"):
                    target_world_xy = self.env.target_world
                    robot_world_xy = self.env.root_states[:, :2]
                    target_vel_world_xy = getattr(self.env, "target_vel_world", None)
                    target_heading = getattr(self.env, "target_heading", None)
                    if target_vel_world_xy is None:
                        target_vel_world_xy = torch.zeros_like(target_world_xy)
                    if target_heading is None:
                        target_heading = torch.zeros(self.num_envs, device=self.device)

                    target_speed = torch.norm(target_vel_world_xy, dim=1)
                    vel_dir_world = target_vel_world_xy / target_speed.unsqueeze(1).clamp_min(1e-6)
                    heading_dir_world = torch.stack([torch.sin(target_heading), torch.cos(target_heading)], dim=1)
                    use_vel = target_speed > self.s0_follow_dir_v_eps
                    dir_world = torch.where(use_vel.unsqueeze(1), vel_dir_world, heading_dir_world)
                    dir_world = dir_world / torch.norm(dir_world, dim=1, keepdim=True).clamp_min(1e-6)

                    robot_minus_target = robot_world_xy - target_world_xy
                    ahead = torch.sum(robot_minus_target * dir_world, dim=1)
                    behind = ahead < (-self.s0_follow_ahead_margin)
                    dist_to_target = torch.norm(target_world_xy - robot_world_xy, dim=1)
                    dist_in_band = (dist_to_target >= self.s0_follow_d_min) & (dist_to_target <= self.s0_follow_d_max)
                    target_freeze_timer = getattr(self.env, "target_freeze_timer", None)
                    if target_freeze_timer is None:
                        not_frozen = torch.ones_like(behind, dtype=torch.bool)
                    else:
                        not_frozen = target_freeze_timer <= 1e-6
                    stable_ok = behind & dist_in_band & phys_visible & not_frozen & (~done_during)
                    self.stable_follow_steps = torch.where(
                        stable_ok,
                        self.stable_follow_steps + 1,
                        torch.zeros_like(self.stable_follow_steps),
                    )
                    success = self.stable_follow_steps >= self.s0_follow_steps_success
                    if success.any():
                        done_any |= success
                        manual_reset_mask |= success
                        success_bonus = success.float() * self.s0_follow_success_bonus
                        total_reward = total_reward + success_bonus
                        reward_terms["success_bonus"] = success_bonus
                        reward_terms["total"] = total_reward

        success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        s_avoid_progress_mask = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        cross_line_dist_snapshot = None
        center_y_snapshot = None
        cross_line_y_snapshot = None
        s_avoid_episode_collision_snapshot = None
        if bool(getattr(self.env, "s_avoid_enabled", False)):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if hasattr(self.env, "s_avoid_episode_collision"):
                self.env.s_avoid_episode_collision |= strict_obstacle_contact_mask
                s_avoid_episode_collision_snapshot = self.env.s_avoid_episode_collision.clone()
            if hasattr(self.env, "_get_s_avoid_episode_progress_flags"):
                s_avoid_progress_mask = self.env._get_s_avoid_episode_progress_flags(env_ids)
            if hasattr(self.env, "_get_s_avoid_cross_line_terms"):
                cross_line_dist_snapshot, center_y_snapshot, cross_line_y_snapshot = (
                    self.env._get_s_avoid_cross_line_terms(env_ids)
                )
            elif hasattr(self.env, "_get_s_avoid_cross_line_dist"):
                cross_line_dist_snapshot = self.env._get_s_avoid_cross_line_dist(env_ids)
            if hasattr(self.env, "_get_s_avoid_episode_success_flags"):
                success_mask = self.env._get_s_avoid_episode_success_flags(env_ids)
                done_any |= success_mask
                manual_reset_mask |= success_mask
            if done_during.any() and hasattr(self.env, "s_avoid_terminal_valid"):
                terminal_mask = done_during & self.env.s_avoid_terminal_valid
                if terminal_mask.any():
                    if hasattr(self.env, "s_avoid_terminal_progress_ratio"):
                        s_avoid_progress_mask = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_progress_ratio,
                            s_avoid_progress_mask,
                        )
                    if hasattr(self.env, "s_avoid_terminal_success"):
                        success_mask = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_success,
                            success_mask,
                        )
                    if hasattr(self.env, "s_avoid_terminal_collision"):
                        s_avoid_episode_collision_snapshot = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_collision,
                            s_avoid_episode_collision_snapshot,
                        )
                    if (
                        cross_line_dist_snapshot is not None
                        and hasattr(self.env, "s_avoid_terminal_cross_line_dist")
                    ):
                        cross_line_dist_snapshot = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_cross_line_dist,
                            cross_line_dist_snapshot,
                        )
                    if center_y_snapshot is not None and hasattr(self.env, "s_avoid_terminal_center_y"):
                        center_y_snapshot = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_center_y,
                            center_y_snapshot,
                        )
                    if cross_line_y_snapshot is not None and hasattr(self.env, "s_avoid_terminal_cross_line_y"):
                        cross_line_y_snapshot = torch.where(
                            terminal_mask,
                            self.env.s_avoid_terminal_cross_line_y,
                            cross_line_y_snapshot,
                        )
                    self.env.s_avoid_terminal_valid[terminal_mask] = False
            collision_reset_mask = collision_mask.clone()
            done_any |= collision_reset_mask
            manual_reset_mask |= collision_reset_mask
        
        # 4. 更新缓冲区
        self.prev_robot_pos = robot_pos.clone()
        self.prev_gate_y = gate_y_raw_curr.clone()
        self.prev_cmd_pred_x = cmd_pred_x.detach().clone()
        self.prev_cmd_exec_mean = cmd_exec_mean.detach().clone()
        if nearest_obs is not None:
            self.prev_nearest_obs_dist = torch.nan_to_num(
                nearest_obs,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self.prev_nearest_obs_valid.fill_(True)
            if forward_clearance is not None:
                self.prev_forward_clearance = torch.nan_to_num(
                    forward_clearance,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                self.prev_forward_clearance_valid.fill_(True)
        # done 的环境避免跨 episode 的 shaped reward 污染
        terminal_fail_mask = done_during | band_fail_mask | collision_reset_mask
        if terminal_fail_mask.any():
            terminal_fail_penalty = float(getattr(self, "terminal_fail_penalty", -10.0))
            collision_fail_penalty = min(
                terminal_fail_penalty,
                float(getattr(self.reward_cfg, "collision_penalty", terminal_fail_penalty)),
            )
            terminal_collision_mask = terminal_fail_mask & collision_mask
            safe_reward = torch.full_like(total_reward, terminal_fail_penalty)
            safe_reward = torch.where(
                terminal_collision_mask,
                torch.full_like(total_reward, collision_fail_penalty),
                safe_reward,
            )
            total_reward = torch.where(terminal_fail_mask, safe_reward, total_reward)
            if reward_terms is not None:
                reward_terms["terminal_penalty"] = torch.where(
                    terminal_fail_mask,
                    safe_reward,
                    torch.zeros_like(total_reward),
                )
                for key, value in list(reward_terms.items()):
                    if key in ("total", "terminal_penalty") or (not torch.is_tensor(value)):
                        continue
                    if value.shape[:1] == terminal_fail_mask.shape:
                        reward_terms[key] = torch.where(terminal_fail_mask, torch.zeros_like(value), value)
                reward_terms["total"] = total_reward
        if self.debug and (collision_mask.any() or done_during.any()):
            debug_count = int(getattr(self, "_collision_debug_count", 0))
            if debug_count < 20:
                debug_env = None
                if bool(collision_mask.any().item()):
                    debug_env = int(collision_mask.nonzero(as_tuple=False).flatten()[0].item())
                elif bool(done_during.any().item()):
                    debug_env = int(done_during.nonzero(as_tuple=False).flatten()[0].item())
                if debug_env is not None:
                    if not hasattr(self, "_penalised_contact_body_names"):
                        try:
                            body_names_all = self.env.gym.get_actor_rigid_body_names(
                                self.env.envs[0], self.env.actor_handles[0]
                            )
                        except Exception:
                            body_names_all = []
                        indices_dbg = collision_debug_indices
                        body_name_cache = []
                        if indices_dbg is not None:
                            for idx_v in indices_dbg.detach().cpu().tolist():
                                idx_i = int(idx_v)
                                if 0 <= idx_i < len(body_names_all):
                                    body_name_cache.append(str(body_names_all[idx_i]))
                                else:
                                    body_name_cache.append(f"body_{idx_i}")
                        self._collision_debug_body_names = body_name_cache
                    body_hits = []
                    if contact_norm is not None:
                        env_contact = contact_norm[debug_env]
                        for local_i, force_v in enumerate(env_contact.detach().cpu().tolist()):
                            force_f = float(force_v)
                            if collision_threshold is not None and force_f > float(collision_threshold):
                                body_name = (
                                    self._collision_debug_body_names[local_i]
                                    if local_i < len(self._collision_debug_body_names)
                                    else f"body_{local_i}"
                                )
                                body_hits.append(f"{body_name}:{force_f:.3f}")
                    nearest_slot = -1
                    nearest_obs_dist = -1.0
                    if (
                        hasattr(self.env, "s_avoid_enabled")
                        and bool(getattr(self.env, "s_avoid_enabled", False))
                        and hasattr(self.env, "s_avoid_active")
                        and hasattr(self.env, "s_avoid_pos_world")
                    ):
                        active_mask = self.env.s_avoid_active[debug_env]
                        if bool(active_mask.any().item()):
                            active_slots = active_mask.nonzero(as_tuple=False).flatten()
                            obs_xy = self.env.s_avoid_pos_world[debug_env, active_slots, :2]
                            robot_xy = self.env.root_states[debug_env, :2].unsqueeze(0)
                            dists = torch.norm(obs_xy - robot_xy, dim=1)
                            min_i = int(torch.argmin(dists).item())
                            nearest_slot = int(active_slots[min_i].item())
                            nearest_obs_dist = float(dists[min_i].item())
                    collision_reward_dbg = 0.0
                    if reward_terms is not None and "collision" in reward_terms and torch.is_tensor(reward_terms["collision"]):
                        collision_reward_dbg = float(reward_terms["collision"][debug_env].item())
                    terminal_penalty_dbg = 0.0
                    if reward_terms is not None and "terminal_penalty" in reward_terms and torch.is_tensor(reward_terms["terminal_penalty"]):
                        terminal_penalty_dbg = float(reward_terms["terminal_penalty"][debug_env].item())
                    threshold_dbg = float(collision_threshold) if collision_threshold is not None else -1.0
                    print(
                        "[CollisionDebug] "
                        f"env={debug_env} collision_mask={int(bool(collision_mask[debug_env].item()))} "
                        f"obstacle_contact_candidate={int(bool(obstacle_contact_candidate_mask[debug_env].item()))} "
                        f"done_during={int(bool(done_during[debug_env].item()))} "
                        f"collision_reward={collision_reward_dbg:.3f} "
                        f"terminal_penalty={terminal_penalty_dbg:.3f} "
                        f"force_max={float(collision_force_max[debug_env].item()):.3f} "
                        f"threshold={threshold_dbg:.3f} "
                        f"obs_candidate_min_clearance={float(obstacle_contact_candidate_min_clearance[debug_env].item()):.3f} "
                        f"nearest_active_obs_slot={nearest_slot} nearest_active_obs_dist={nearest_obs_dist:.3f} "
                        f"body_hits={body_hits if body_hits else ['none']}"
                    )
                    self._collision_debug_count = debug_count + 1

        # 4.5 高层 episode 统计（与 breakdown 口径对齐）
        self.episode_return_buf += total_reward
        self.episode_len_buf += 1

        # 5. 处理超时
        timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self.no_episode_timeout:
            timeout = (self.episode_length_buf >= self.max_episode_length) & (~done_any)
        timeout_bootstrap_obs = None
        if timeout.any():
            timeout_bootstrap_obs = {}
            for key, value in reward_obs.items():
                timeout_bootstrap_obs[key] = value.detach().clone() if torch.is_tensor(value) else value
        done_any |= timeout
        manual_reset_mask |= timeout

        # Scene difficulty switches only when an env starts a new episode.
        if done_any.any():
            self._apply_scene_difficulty_for_resets(done_any)

        # 低层已 auto-reset 的 env 不再重复 reset；高层触发的 done 在这里统一 reset。
        manual_reset_mask = manual_reset_mask & (~done_during)
        if manual_reset_mask.any():
            reset_ids = manual_reset_mask.nonzero(as_tuple=False).flatten()
            self.env.reset_idx(reset_ids)
            # reset_idx 仅重置状态，不会刷新 high-level 观测缓存；这里显式刷新一次。
            if hasattr(self.env, "compute_observations"):
                self.env.compute_observations()
            self._refresh_depth_images(force=True)
            self.prev_robot_pos[reset_ids] = self.env.root_states[reset_ids, :3]
            self.prev_cmd_pred_x[reset_ids] = 0.0
            self.prev_cmd_exec_mean[reset_ids] = 0.0
            self.rotate_only_active[reset_ids] = False
            if self.prev_goal_world is not None and hasattr(self.env, "goal_world"):
                self.prev_goal_world[reset_ids] = self.env.goal_world[reset_ids]
            self.prev_nearest_obs_dist[reset_ids] = 0.0
            self.prev_nearest_obs_valid[reset_ids] = False
            self.prev_forward_clearance[reset_ids] = 0.0
            self.prev_forward_clearance_valid[reset_ids] = False
            self.prev_align_center_abs[reset_ids] = 0.0
            self.prev_align_center_valid[reset_ids] = False
        
        goal_change_count_snapshot = self.goal_change_count.clone()
        episode_info = None
        if done_any.any():
            self._resample_forced_forward_speed(done_any.nonzero(as_tuple=False).flatten())
            episode_info = {
                'r': self.episode_return_buf.clone(),
                'l': self.episode_len_buf.clone(),
            }
            self.episode_length_buf[done_any] = 0
            self.prev_gate_y[done_any] = 0
            self.reach_given[done_any] = False
            self.goal_change_count[done_any] = 0
            self.target_lost_steps[done_any] = 0
            self.stable_follow_steps[done_any] = 0
            self.episode_return_buf[done_any] = 0.0
            self.episode_len_buf[done_any] = 0
            self.prev_robot_pos[done_any] = self.env.root_states[done_any, :3]
            self.prev_cmd_pred_x[done_any] = 0.0
            self.prev_cmd_exec_mean[done_any] = 0.0
            self.rotate_only_active[done_any] = False
            self.prev_nearest_obs_dist[done_any] = 0.0
            self.prev_nearest_obs_valid[done_any] = False
            self.prev_forward_clearance[done_any] = 0.0
            self.prev_forward_clearance_valid[done_any] = False
            self.prev_align_center_abs[done_any] = 0.0
            self.prev_align_center_valid[done_any] = False
            if self.prev_goal_world is not None and hasattr(self.env, "goal_world"):
                self.prev_goal_world[done_any] = self.env.goal_world[done_any]
            # 重置 Post-Processor 的变化率限制记忆
            self.post_processor.last_cmd[done_any] = 0.0

        next_obs = self._get_high_level_obs()
        self.last_obs = next_obs

        info = {
            'post_info': post_info_payload,
            'episode_length': length_snapshot,
            'reward_terms': reward_terms,
            'gate_y': gate_y,
            'goal_change_count': goal_change_count_snapshot,
            'episode': episode_info,
            'collision_mask': collision_mask,
            'collision_force_max': collision_force_max,
            'collision_threshold': collision_threshold,
            'collision_threshold_src': collision_threshold_src,
            'collision_indices_src': collision_indices_src,
            'obstacle_contact_candidate_mask': obstacle_contact_candidate_mask,
            'strict_obstacle_contact_mask': strict_obstacle_contact_mask,
            'obstacle_contact_candidate_min_clearance': obstacle_contact_candidate_min_clearance,
            'avoid_band_active_mask': avoid_band_active_mask,
            'avoid_band_outside_dist': avoid_band_outside_dist,
            'clearance': clearance,
            'clearance_global': clearance_global,
            'done_during': done_during,
            'manual_reset_mask': manual_reset_mask,
            'success_mask': success_mask,
            's_avoid_progress_mask': s_avoid_progress_mask,
            'cross_line_dist': cross_line_dist_snapshot,
            'center_y': center_y_snapshot,
            'cross_line_y': cross_line_y_snapshot,
            's_avoid_episode_collision': s_avoid_episode_collision_snapshot,
            'timeout': timeout,
            'timeout_bootstrap_obs': timeout_bootstrap_obs,
            'target_turn_events': getattr(self.env, "target_turn_events", None),
            'target_preturn_events': getattr(self.env, "target_preturn_events", None),
            'target_reflect_events': getattr(self.env, "target_reflect_events", None),
            'target_turn_count': getattr(self.env, "target_turn_count", None),
            'target_preturn_count': getattr(self.env, "target_preturn_count", None),
            'target_reflect_count': getattr(self.env, "target_reflect_count", None),
            'target_reset_dist_error': getattr(self.env, "target_reset_dist_error", None),
            'target_reset_bearing_error': getattr(self.env, "target_reset_bearing_error", None),
        }

        return next_obs, total_reward, done_any, info

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """四元数转 yaw 角"""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _quat_to_rp(self, quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """四元数转 roll, pitch"""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return roll, pitch


# 训练循环 (PPO + Distillation)
def train(args):
    """主训练函数 (V5 Command-Space MoE)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"V5 Command-Space MoE Training")
    print(f"Mode: {args.mode.upper()} | Skill: {getattr(args, 'skill', 'follow')}")
    print(f"Device: {device}")
    print("[Note] Online training stats are for monitoring only; use eval_highlevel.py for paper-report metrics.")
    print(f"{'='*60}\n")
    debug = bool(getattr(args, "debug", False))
    debug_memory = bool(getattr(args, "debug_memory", False)) and torch.cuda.is_available() and device.type == "cuda"
    debug_memory_interval = max(1, int(getattr(args, "debug_memory_interval", 10)))
    debug_memory_device_index = _get_cuda_device_index(device) if debug_memory else None
    def dprint(*vals, **kwargs):
        if debug:
            print(*vals, **kwargs)
    args.task = normalize_task_name(getattr(args, "task", ""))
    skill_name = getattr(args, "skill", "follow")
    if args.task == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用 hex_ground / s_avoid_basic / s_cylinder / s_calib")
    if args.task == "hex_ground":
        print("[Warn] hex_ground 仅作为容器任务，请显式配置 terrain_type 或改用 s_avoid_basic/s_cylinder/s_calib。")
    if args.task in ("s_mixed_gate", "s_mixed_gate_large"):
        print(
            f"[Warn] {args.task} 为实验混合场景容器，当前主线不建议直接用于首轮训练。"
        )
    recommended_by_skill = {
        "follow": ("s_follow_basic",),
        "avoid": ("s_avoid_basic", "s_cylinder", "s_narrow_passage", "s_step_field", "s_dense_obstacles"),
        "moe": ("s_avoid_basic",),
    }
    eval_only_by_skill = {
        "avoid": ("s_ood_holdout", "s_ood_holdout_large"),
    }
    rec_tasks = recommended_by_skill.get(skill_name, tuple())
    eval_only_tasks = eval_only_by_skill.get(skill_name, tuple())
    if args.task in eval_only_tasks:
        if not bool(getattr(args, "allow_eval_only_train", False)):
            raise ValueError(
                f"task={args.task} 属于结构化 OOD hold-out，只允许评测；"
                "如确需临时训练，请显式加 --allow_eval_only_train。"
            )
        print(f"[Warn] task={args.task} 属于结构化 OOD hold-out；当前因 --allow_eval_only_train 被放行。")
    elif rec_tasks and args.task not in rec_tasks:
        rec_text = "/".join(rec_tasks)
        print(f"[Warn] 当前 task={args.task} 不在 {skill_name} 主线推荐列表（{rec_text}）。")
    gate_use_difficulty = bool(getattr(args, "gate_use_difficulty", False))
    moe_use_student_aff = bool(getattr(args, "moe_use_student_aff", False))
    if skill_name == "moe":
        if args.num_steps < 48:
            print(f"[Warn] Gate 训练建议 num_steps>=48，当前 num_steps={args.num_steps}。")
        dprint(f"[Info] Gate difficulty 输入: {'ON' if gate_use_difficulty else 'OFF'}")
        dprint(f"[Info] MoE affordance 来源: {'student' if moe_use_student_aff else args.mode}")
    if getattr(args, "disable_risk_scale", False):
        dprint("[Info] CommandPostProcessor 风险缩放已禁用（消融用）。")
    if getattr(args, "aff_stack", 1) > 1:
        print(f"[Warn] aff_stack={args.aff_stack}: 输入通道数改变，必须使用相同 aff_stack 训练/加载 ckpt；旧 ckpt 不兼容。")
    if args.mode == "student" and not getattr(args, "vision_ckpt", None):
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
    if args.mode == "student":
        args.camera_enable = True
    
    # 导入模块
    import_modules()
    
    skill = getattr(args, "skill", "follow")
    if args.mode == "student" and not getattr(args, "teacher_ckpt", None):
        raise ValueError("Student 模式必须提供 --teacher_ckpt，避免把纯视觉从头训练误记为蒸馏实验。")
    if args.mode == "student" and skill == "moe":
        raise ValueError("当前未实现 Gate 的 student 蒸馏训练链路，禁止使用 --mode student --skill moe。")
    use_avoid_local_map = skill in ("avoid", "moe")

    env_cfg_override = None
    train_cfg_override = None
    if task_registry is not None and not bool(getattr(args, "_entropy_coef_explicit", False)):
        env_cfg_override, train_cfg_override = task_registry.get_cfgs(name=args.task)
        if train_cfg_override is not None:
            algo_cfg = getattr(train_cfg_override, "algorithm", None)
            if algo_cfg is not None and hasattr(algo_cfg, "entropy_coef"):
                args.entropy_coef = float(getattr(algo_cfg, "entropy_coef"))

    # 创建环境
    env = HierarchicalHexapodEnv(args, device, env_cfg=env_cfg_override, train_cfg=train_cfg_override)
    dprint(f"[Main] 环境初始化完成: {env.num_envs} envs")
    
    # 初始 reset（用于确定观测维度）
    obs_dict = env.reset()
    scene_meta = None
    if hasattr(env, "env") and hasattr(env.env, "extras"):
        scene_meta = env.env.extras.get("scene_meta")
    if scene_meta is not None and debug:
        print(f"[Debug] scene_meta: {scene_meta}")
    if debug:
        try:
            if hasattr(env, "env"):
                spacing = getattr(getattr(env.env, "cfg", None), "env", None)
                if spacing is not None:
                    env_spacing = float(getattr(spacing, "env_spacing", 0.0))
                else:
                    env_spacing = 0.0
                if hasattr(env.env, "env_origins"):
                    origins = env.env.env_origins[:3].detach().cpu().tolist()
                else:
                    origins = []
                print(f"[Debug] envs={env.num_envs} env_spacing={env_spacing:.3f} env_origins[:3]={origins}")
            env0 = 0
            base_pos = None
            if hasattr(env, "env") and hasattr(env.env, "root_states"):
                base_pos = env.env.root_states[env0, :3].detach().cpu()
            if base_pos is not None:
                print(f"[Debug] env0 base_pos: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
            if hasattr(env, "env") and hasattr(env.env, "env_origins"):
                origin = env.env.env_origins[env0]
                ox, oy, oz = float(origin[0].item()), float(origin[1].item()), float(origin[2].item())
                print(f"[Debug] env0 origin: ({ox:.3f}, {oy:.3f}, {oz:.3f})")
            else:
                ox = oy = oz = 0.0
            # SceneSpec positions (local)
            scene_spec = None
            if hasattr(env, "env") and hasattr(env.env, "scene_spec_cache"):
                scene_spec = env.env.scene_spec_cache[env0]
            if scene_spec is None:
                print("[Debug] env0 scene_spec: None")
            else:
                num_static = len(scene_spec.static_obstacles)
                print(f"[Debug] env0 scene_spec static_obstacles: {num_static}")
                for idx, spec in enumerate(scene_spec.static_obstacles[:5]):
                    spec_pos = spec["position"] if isinstance(spec, dict) else spec.position
                    spec_size = spec["size"] if isinstance(spec, dict) else spec.size
                    wx = ox + float(spec_pos[0])
                    wy = oy + float(spec_pos[1])
                    wz = oz + float(spec_pos[2])
                    sx, sy, sz = spec_size
                    print(
                        f"[Debug] spec[{idx}] pos=({spec_pos[0]:.3f},{spec_pos[1]:.3f},{spec_pos[2]:.3f}) "
                        f"world=({wx:.3f},{wy:.3f},{wz:.3f}) size=({sx:.3f},{sy:.3f},{sz:.3f})"
                    )
            # Applied actor positions (static)
            applied = []
            if hasattr(env, "env"):
                env_impl = env.env
                all_root_states = getattr(env_impl, "all_root_states", None)
                block_groups = getattr(env_impl, "static_block_groups", [])
                for group in block_groups:
                    active = getattr(env_impl, f"static_{group}_active", None)
                    pos_local = getattr(env_impl, f"static_{group}_pos_local", None)
                    indices = getattr(env_impl, f"static_{group}_actor_indices", None)
                    if active is None or pos_local is None or indices is None:
                        continue
                    active_ids = torch.nonzero(active[env0], as_tuple=False).flatten()
                    for local_id in active_ids.tolist():
                        actor_index = int(indices[env0, local_id].item())
                        pos = pos_local[env0, local_id]
                        applied.append((f"{group}:{local_id}", actor_index, pos))
                active = getattr(env_impl, "static_wall_active", None)
                pos_local = getattr(env_impl, "static_wall_pos_local", None)
                indices = getattr(env_impl, "static_wall_actor_indices", None)
                if active is not None and pos_local is not None and indices is not None:
                    active_ids = torch.nonzero(active[env0], as_tuple=False).flatten()
                    for local_id in active_ids.tolist():
                        actor_index = int(indices[env0, local_id].item())
                        pos = pos_local[env0, local_id]
                        applied.append((f"wall:{local_id}", actor_index, pos))
                if applied:
                    z_values = []
                    print(f"[Debug] env0 applied static obstacles: {len(applied)} (show first 5)")
                    for idx, (label, actor_index, pos) in enumerate(applied[:5]):
                        wx = ox + float(pos[0].item())
                        wy = oy + float(pos[1].item())
                        wz = oz + float(pos[2].item())
                        if all_root_states is not None:
                            actual = all_root_states[actor_index, :3].detach().cpu()
                            ax, ay, az = float(actual[0].item()), float(actual[1].item()), float(actual[2].item())
                            z_values.append(az)
                            print(
                                f"[Debug] actor[{label}] idx={actor_index} "
                                f"local=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                                f"world=({wx:.3f},{wy:.3f},{wz:.3f}) "
                                f"root=({ax:.3f},{ay:.3f},{az:.3f})"
                            )
                        else:
                            print(
                                f"[Debug] actor[{label}] idx={actor_index} "
                                f"local=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                                f"world=({wx:.3f},{wy:.3f},{wz:.3f})"
                            )
                    if z_values:
                        print(f"[Debug] static root z: min={min(z_values):.3f} max={max(z_values):.3f} (ground_z={oz:.3f})")
                else:
                    print("[Debug] env0 applied static obstacles: 0")
        except Exception as exc:
            print(f"[Debug] obstacle position dump failed: {exc}")

    def _get_actor_gt_inputs(current_obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_avoid_local_map:
            return current_obs['local_map_2ch'], current_obs['actor_difficulty']
        return current_obs['gt_affordance'], current_obs['gt_difficulty']

    def _get_critic_gt_inputs(current_obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_avoid_local_map:
            return current_obs['critic_local_map_2ch'], current_obs['critic_difficulty']
        return current_obs['gt_affordance'], current_obs['gt_difficulty']

    def _get_moe_follow_gt_inputs(current_obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return current_obs['gt_affordance'], current_obs['gt_difficulty']

    def _get_moe_avoid_gt_inputs(current_obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return current_obs['local_map_2ch'], current_obs['actor_difficulty']

    def _get_moe_expert_state_inputs(state_tensor: torch.Tensor) -> torch.Tensor:
        return get_moe_expert_state_inputs(state_tensor)

    def _predict_affordance_triplet(
        current_obs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if vision_model is None:
            raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
        with torch.no_grad():
            vis_out = vision_model(current_obs['depth'], normalize=True)
            aff_map_pred = torch.stack([
                vis_out['occupancy'],
                vis_out['passable_gap'],
                vis_out['low_obstacle'],
            ], dim=1)
            aff_map_pred = resize_affordance_map(aff_map_pred, env.affordance_map_size)
            avoid_map_pred = build_avoid_local_map_2ch(
                aff_map_pred,
                visible_mask=env.affordance_visible_mask,
            )
            avoid_difficulty_pred = env._compute_objective_difficulty_from_local_map(avoid_map_pred)
            gt_difficulty_pred = difficulty_from_gap(aff_map_pred)
        return aff_map_pred, avoid_map_pred, avoid_difficulty_pred, gt_difficulty_pred

    def _predict_actor_inputs(current_obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        aff_map_pred, avoid_map_pred, avoid_difficulty_pred, gt_difficulty_pred = _predict_affordance_triplet(current_obs)
        if use_avoid_local_map:
            return avoid_map_pred, avoid_difficulty_pred
        return aff_map_pred, gt_difficulty_pred

    def _predict_moe_expert_inputs(
        current_obs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        aff_map_pred, avoid_map_pred, avoid_difficulty_pred, gt_difficulty_pred = _predict_affordance_triplet(current_obs)
        return aff_map_pred, gt_difficulty_pred, avoid_map_pred, avoid_difficulty_pred

    actor_aff_init, _ = _get_actor_gt_inputs(obs_dict)

    state_dim = obs_dict['state'].shape[1]
    goal_dim = obs_dict['goal'].shape[1]
    aff_shape = actor_aff_init.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack

    # 创建 Policy (V5)
    is_gate = skill == "moe"
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    action_dim = 1 if is_gate else 3
    gate_use_difficulty = bool(getattr(args, "gate_use_difficulty", False))
    moe_use_student_aff = bool(getattr(args, "moe_use_student_aff", False))
    use_follow_expert = bool(getattr(args, "use_follow_expert", False))
    if use_follow_expert and (is_gate or skill != "follow"):
        raise ValueError("--use_follow_expert 仅支持 --skill follow 且 non-gate 路径。")
    if use_follow_expert and bool(getattr(args, "egpo", False)):
        print("[Warn] --use_follow_expert 已开启，--egpo 将被忽略。")
    use_egpo = (
        bool(getattr(args, "egpo", False))
        and (not use_follow_expert)
        and (not is_gate)
        and (skill == "follow")
        and (args.task == S0_FOLLOW_TASK_NAME)
    )
    use_expert_guidance = use_egpo or use_follow_expert
    use_gate_analytic_follow = is_gate
    compute_s0_follow_expert_cmd = None
    if bool(getattr(args, "egpo", False)) and not use_egpo:
        print("[Warn] --egpo 仅在 --skill follow --task s_follow_basic 且 non-gate 路径生效；当前配置下将忽略。")
    if use_expert_guidance or use_gate_analytic_follow:
        from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd as s0_follow_expert_fn
        compute_s0_follow_expert_cmd = s0_follow_expert_fn
        if use_egpo:
            dprint(
                f"[EGPO] enabled: interface_ratio={EXPERT_INTERFACE_RATIO:.2f}, alpha_min={EXPERT_ALPHA_MIN}, "
                f"schedule={EXPERT_ALPHA_SCHEDULE}, bc_coef={EXPERT_BC_COEF}"
            )
        if use_follow_expert:
            print("[Train] cmd_source=follow_expert (--use_follow_expert)")
        if use_gate_analytic_follow:
            print("[Gate] follow_source=analytic_s0_follow")

    run_mode_tag = args.mode + ("_studentaff" if (is_gate and moe_use_student_aff) else "")

    def _compute_moe_follow_cmd_from_goal(
        state_tensor: torch.Tensor,
        goal_tensor: torch.Tensor,
        reset_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if compute_s0_follow_expert_cmd is None:
            raise RuntimeError("Gate 模式下解析式 follow expert 不可用。")
        if state_tensor.ndim != 2 or state_tensor.shape[1] < 3:
            raise ValueError(f"state_tensor shape invalid for analytic follow expert: {tuple(state_tensor.shape)}")
        if goal_tensor.ndim != 2 or goal_tensor.shape[1] < 2:
            raise ValueError(f"goal_tensor shape invalid for analytic follow expert: {tuple(goal_tensor.shape)}")
        robot_pos_world_xy = state_tensor[:, :2]
        robot_heading = torch.atan2(torch.sin(state_tensor[:, 2]), torch.cos(state_tensor[:, 2]))
        goal_x = goal_tensor[:, 0]
        goal_y = goal_tensor[:, 1]
        cos_heading = torch.cos(robot_heading)
        sin_heading = torch.sin(robot_heading)
        delta_world_x = cos_heading * goal_x - sin_heading * goal_y
        delta_world_y = sin_heading * goal_x + cos_heading * goal_y
        target_world_xy = robot_pos_world_xy + torch.stack([delta_world_x, delta_world_y], dim=1)
        return compute_s0_follow_expert_cmd(
            robot_pos_world_xy=robot_pos_world_xy,
            robot_heading=robot_heading,
            target_world_xy=target_world_xy,
            target_vel_world_xy=None,
            target_heading=None,
            cmd_scale=cmd_scale,
            reset_mask=reset_mask,
        )

    def _normalize_meta_value(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_normalize_meta_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _normalize_meta_value(v) for k, v in value.items()}
        return str(value)

    def _build_experiment_meta() -> Dict[str, object]:
        tracked_keys = [
            "task",
            "mode",
            "skill",
            "seed",
            "num_envs",
            "num_steps",
            "num_epochs",
            "mini_batch_size",
            "lr",
            "egpo",
            "egpo_lr_guided",
            "egpo_lr_post",
            "use_follow_expert",
            "moe_use_student_aff",
            "gate_use_difficulty",
            "aff_stack",
            "beta",
            "t1_goal_occlusion",
            "allow_eval_only_train",
            "allow_inexact_resume",
            "teacher_ckpt",
            "vision_ckpt",
            "follow_ckpt",
            "avoid_ckpt",
            "low_level_ckpt",
            "output_dir",
            "save_interval",
            "gamma",
            "gae_lambda",
            "clip_range",
            "value_loss_coef",
            "entropy_coef",
            "max_grad_norm",
            "cmd_slew_lin",
            "cmd_slew_ang",
            "cmd_safe_dist",
            "cmd_free_dist",
            "gate_safe_clamp",
            "gate_safe_max",
            "w_mode",
            "w_tau",
            "w_blend_mode",
            "w_disable_gate_safe_clamp",
            "moe_expert_deterministic",
            "disable_risk_scale",
            "camera_enable",
            "camera_interval",
            "decimation",
        ]
        meta = {key: _normalize_meta_value(getattr(args, key, None)) for key in tracked_keys}
        for key in ("teacher_ckpt", "vision_ckpt", "follow_ckpt", "avoid_ckpt", "low_level_ckpt"):
            if meta.get(key):
                meta[key] = os.path.abspath(str(meta[key]))
        meta["sim2real_sensor_target"] = "Intel RealSense D435i"
        meta["observation_contract"] = collect_runtime_observation_contract(args, env)
        meta["run_mode_tag"] = run_mode_tag
        meta["online_best_metric"] = "online_mean_reward_monitor_only"
        meta["paper_eval_entry"] = "legged_gym/scripts/eval_highlevel.py"
        meta["unknown_cli_args"] = list(getattr(args, "_unknown_cli", []))
        if bool(getattr(env.env, "s_avoid_enabled", False)):
            meta["s_avoid_protocol"] = {
                "goal_mode": "fixed_last_row_plus_margin",
                "goal_y_offset_after_last_row_m": 0.5,
                "goal_buf_mode": "[0.0, cross_line_dist]",
                "success_mode": "cross_line_and_no_collision",
                "success_cross_line_offset_m": 0.3,
                "band_axes": "x_only",
                "band_margin_x_m": float(getattr(env.env.cfg.terrain, "avoid_band_margin_x", 0.0)),
                "forced_forward": {
                    "enabled": True,
                    "stage_speed": {
                        "1": [0.2, 0.8],
                        "2": [0.2, 0.6],
                        "3": [0.2, 0.45],
                        "4": [0.2, 0.35],
                    },
                    "stage_warmup_iters": 50,
                    "train_warmup_iters": 100,
                },
                "risk_barrier_scale": float(getattr(env.reward_cfg, "risk_barrier_scale", 0.0)),
            }
        return meta

    def _check_checkpoint_meta(
        ckpt_meta: Optional[Dict[str, object]],
        *,
        source_name: str,
        strict_source: bool,
    ) -> None:
        if not isinstance(ckpt_meta, dict):
            print(f"[Warn] {source_name} 未包含 experiment_meta；无法校验 teacher/vision 来源一致性。")
            return
        current_meta = _build_experiment_meta()
        hard_keys = ("task", "mode", "skill")
        for key in hard_keys:
            prev_v = ckpt_meta.get(key, None)
            curr_v = current_meta.get(key, None)
            if prev_v is not None and curr_v is not None and prev_v != curr_v:
                raise ValueError(
                    f"{source_name} 与当前运行的 {key} 不一致：checkpoint={prev_v}, current={curr_v}。"
                )
        source_keys = ("teacher_ckpt", "vision_ckpt", "follow_ckpt", "avoid_ckpt", "low_level_ckpt")
        mismatch_msgs = []
        for key in source_keys:
            prev_v = ckpt_meta.get(key, None)
            curr_v = current_meta.get(key, None)
            if prev_v is None or curr_v is None or prev_v == curr_v:
                continue
            mismatch_msgs.append(f"{key}: checkpoint={prev_v}, current={curr_v}")
        if mismatch_msgs:
            msg = (
                f"{source_name} 的来源配置与当前命令不一致，可能污染实验归因："
                + " | ".join(mismatch_msgs)
            )
            if strict_source:
                raise ValueError(msg)
            print(f"[Warn] {msg}")

    def _check_aux_model_meta(
        ckpt_meta: Optional[Dict[str, object]],
        *,
        source_name: str,
        expected_mode: Optional[str] = None,
        expected_skill: Optional[str] = None,
    ) -> None:
        if not isinstance(ckpt_meta, dict):
            return
        if expected_mode is not None:
            meta_mode = ckpt_meta.get("mode", None)
            if meta_mode is not None and meta_mode != expected_mode:
                raise ValueError(
                    f"{source_name} 的 mode 与当前实验预期不一致: checkpoint={meta_mode}, expected={expected_mode}"
                )
        if expected_skill is not None:
            meta_skill = ckpt_meta.get("skill", None)
            if meta_skill is not None and meta_skill != expected_skill:
                raise ValueError(
                    f"{source_name} 的 skill 与当前实验预期不一致: checkpoint={meta_skill}, expected={expected_skill}"
                )

    if is_gate:
        policy = GatePolicy(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
        ).to(device)
    else:
        policy = CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            cmd_scale=cmd_scale,
        ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # 加载 Teacher/Vision 模型 (Student 模式)
    teacher_model = None
    vision_model = None
    follow_model = None
    avoid_model = None

    if args.mode == 'teacher':
        dprint("[Main] Mode: TEACHER. Training from scratch with GT.")
    elif args.mode == 'student':
        dprint("\n[Student] 加载 Teacher 和 Vision 模型...")

        # 加载 Teacher
        if args.teacher_ckpt and not is_gate:
            teacher_model = CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(device)

            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            ckpt_meta = ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None
            _check_aux_model_meta(
                ckpt_meta,
                source_name="Teacher checkpoint",
                expected_mode="teacher",
                expected_skill=skill,
            )
            validate_checkpoint_contract_compatibility(
                build_runtime_contract_meta(args, env),
                ckpt_meta,
                reference_name="current runtime",
                candidate_name="Teacher checkpoint",
                strict=True,
            )
            if 'model_state_dict' in ckpt:
                load_high_level_state_dict_compat(
                    teacher_model,
                    ckpt['model_state_dict'],
                    label="teacher_model",
                )
            else:
                load_high_level_state_dict_compat(
                    teacher_model,
                    ckpt,
                    label="teacher_model",
                )
            teacher_model.eval()

            # 用 Teacher 权重初始化 Student
            policy.load_state_dict(teacher_model.state_dict())
            dprint(f"[Student] ✓ Teacher 加载成功: {args.teacher_ckpt}")

        # 加载 Vision (AffordanceEstimator)
        if args.vision_ckpt and AffordanceEstimator is not None:
            vision_model = AffordanceEstimator(
                depth_channels=1,
                output_size=get_vision_native_output_size(),
                max_depth_range=5.0
            ).to(device)

            ckpt = torch.load(args.vision_ckpt, map_location=device)
            validate_vision_runtime_contract(
                args,
                env,
                source_name="Student vision checkpoint",
                ckpt_meta=ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None,
                strict_meta=True,
            )
            if 'model_state_dict' in ckpt:
                vision_model.load_state_dict(ckpt['model_state_dict'])
            else:
                vision_model.load_state_dict(ckpt)
            vision_model.eval()
            dprint(f"[Student] ✓ Vision 加载成功: {args.vision_ckpt}")
    if args.mode == 'teacher' and is_gate and moe_use_student_aff:
        if not getattr(args, "vision_ckpt", None):
            raise ValueError("moe_use_student_aff 需要提供 --vision_ckpt。")
        if AffordanceEstimator is None:
            raise ValueError("AffordanceEstimator 不可用，无法使用 student affordance。")
        vision_model = AffordanceEstimator(
            depth_channels=1,
            output_size=get_vision_native_output_size(),
            max_depth_range=5.0
        ).to(device)
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        validate_vision_runtime_contract(
            args,
            env,
            source_name="Gate vision checkpoint",
            ckpt_meta=ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None,
            strict_meta=True,
        )
        if 'model_state_dict' in ckpt:
            vision_model.load_state_dict(ckpt['model_state_dict'])
        else:
            vision_model.load_state_dict(ckpt)
        vision_model.eval()
        dprint(f"[Gate] ✓ Vision 加载成功: {args.vision_ckpt}")

    if is_gate:
        if not getattr(args, "avoid_ckpt", None):
            raise ValueError("Gate 模式需要提供 --avoid_ckpt；follow 侧默认使用解析式 expert。")
        if compute_s0_follow_expert_cmd is None:
            raise RuntimeError("Gate 模式下解析式 follow expert 不可用。")
        follow_model = None
        avoid_aff_channels = AVOID_LOCAL_MAP_CHANNELS * aff_stack
        avoid_model = CmdVelExpert(
            affordance_channels=avoid_aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            cmd_scale=cmd_scale,
        ).to(device)
        ckpt = torch.load(args.avoid_ckpt, map_location=device)
        ckpt_meta = ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None
        _check_aux_model_meta(
            ckpt_meta,
            source_name="Expert checkpoint (avoid)",
            expected_skill="avoid",
        )
        validate_checkpoint_contract_compatibility(
            _build_experiment_meta(),
            ckpt_meta,
            reference_name="current gate run",
            candidate_name="Expert checkpoint (avoid)",
            strict=True,
        )
        if 'model_state_dict' in ckpt:
            load_high_level_state_dict_compat(
                avoid_model,
                ckpt['model_state_dict'],
                label=f"expert:{os.path.basename(args.avoid_ckpt)}",
            )
        else:
            load_high_level_state_dict_compat(
                avoid_model,
                ckpt,
                label=f"expert:{os.path.basename(args.avoid_ckpt)}",
            )
        avoid_model.eval()
        for param in avoid_model.parameters():
            param.requires_grad = False
        dprint(f"[Gate] ✓ Follow=analytic expert, Avoid 加载完成: {args.avoid_ckpt}")
        if any(param.requires_grad for param in avoid_model.parameters()):
            raise RuntimeError("Gate 模式下 Avoid expert 仍存在可训练参数。")

    resume_path = getattr(args, "resume", None)
    finetune_path = getattr(args, "finetune_from", None)
    if resume_path and finetune_path:
        raise ValueError("不能同时使用 --resume 与 --finetune_from。")

    start_iteration = 0
    best_reward = float("-inf")
    best_success = float("-inf")
    best_progress = float("-inf")
    best_selection_metric_name = "online_mean_reward_monitor_only"
    if resume_path:
        if not bool(getattr(args, "allow_inexact_resume", False)):
            raise ValueError(
                "当前 checkpoint 不保存 env/curriculum 完整状态，--resume 只属于近似续训；"
                "默认已禁用。若你明确接受，请显式加 --allow_inexact_resume；"
                "若只想继续权重训练，请优先用 --finetune_from。"
            )
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint 不存在: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if os.path.basename(resume_path) == "best_online_reward.pt":
            raise ValueError("best_online_reward.pt 仅用于在线监控挑点，不允许作为 --resume；请改用 --finetune_from。")
        _check_checkpoint_meta(
            ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None,
            source_name="Resume checkpoint",
            strict_source=True,
        )
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        load_high_level_state_dict_compat(policy, state_dict, label="resume_policy")
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                print(f"[Warn] Resume optimizer_state_dict 不兼容，改用新优化器继续训练: {exc}")
        else:
            raise ValueError("Resume checkpoint 未包含 optimizer_state_dict；请改用 --finetune_from。")
        start_iteration = int(ckpt.get("iteration", 0)) + 1 if isinstance(ckpt, dict) else 0
        best_reward = float(
            ckpt.get("best_online_reward", ckpt.get("best_reward", ckpt.get("mean_reward", -float("inf"))))
        ) if isinstance(ckpt, dict) else best_reward
        if isinstance(ckpt, dict):
            best_success = float(ckpt.get("best_online_success", -float("inf")))
            best_progress = float(ckpt.get("best_online_progress", -float("inf")))
            best_selection_metric_name = str(
                ckpt.get("best_online_selection_metric", ckpt.get("selection_metric", best_selection_metric_name))
            )
        if isinstance(ckpt, dict) and "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if isinstance(ckpt, dict) and "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if torch.cuda.is_available() and isinstance(ckpt, dict) and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        log_dir = os.path.dirname(resume_path)
        dprint(f"[Main] Resume: {resume_path}")
    elif finetune_path:
        if not os.path.exists(finetune_path):
            raise FileNotFoundError(f"Finetune checkpoint 不存在: {finetune_path}")
        ckpt = torch.load(finetune_path, map_location=device)
        _check_checkpoint_meta(
            ckpt.get("experiment_meta", None) if isinstance(ckpt, dict) else None,
            source_name="Finetune checkpoint",
            strict_source=False,
        )
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        load_high_level_state_dict_compat(policy, state_dict, label="finetune_policy")
        dprint(f"[Main] Finetune from: {finetune_path}")

    # 创建日志目录
    if resume_path:
        os.makedirs(log_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(args.output_dir, f"{skill}_{run_mode_tag}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[Main] 日志目录: {log_dir}")
    run_meta = _build_experiment_meta()
    run_meta["log_dir"] = os.path.abspath(log_dir)
    with open(os.path.join(log_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    
    # 创建 Rollout Buffer
    aff_map_shape = (aff_channels, aff_shape[1], aff_shape[2])
    buffer = RolloutBuffer(env.num_envs, args.num_steps, state_dim, goal_dim, aff_map_shape, action_dim, device)

    def _build_next_aff_stacks(
        current_actor_stack: Optional[torch.Tensor],
        current_critic_stack: Optional[torch.Tensor],
        next_actor_aff: torch.Tensor,
        next_critic_aff: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_actor_stack is None or current_critic_stack is None:
            return (
                next_actor_aff.repeat(1, aff_stack, 1, 1),
                next_critic_aff.repeat(1, aff_stack, 1, 1),
            )

        actor_stack_next = torch.roll(current_actor_stack, shifts=-next_actor_aff.shape[1], dims=1)
        actor_stack_next[:, -next_actor_aff.shape[1]:, :, :] = next_actor_aff
        critic_stack_next = torch.roll(current_critic_stack, shifts=-next_critic_aff.shape[1], dims=1)
        critic_stack_next[:, -next_critic_aff.shape[1]:, :, :] = next_critic_aff
        return actor_stack_next, critic_stack_next

    def _compute_bootstrap_value_for_obs(
        bootstrap_obs: Dict[str, torch.Tensor],
        current_actor_stack: Optional[torch.Tensor],
        current_critic_stack: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            bootstrap_state = bootstrap_obs['state']
            bootstrap_goal = bootstrap_obs['goal']
            bootstrap_critic_aff_map, bootstrap_critic_difficulty = _get_critic_gt_inputs(bootstrap_obs)
            bootstrap_use_student_aff = args.mode != 'teacher'
            if is_gate and moe_use_student_aff:
                bootstrap_use_student_aff = True
            if bootstrap_use_student_aff:
                bootstrap_aff_map, bootstrap_difficulty = _predict_actor_inputs(bootstrap_obs)
            else:
                bootstrap_aff_map, bootstrap_difficulty = _get_actor_gt_inputs(bootstrap_obs)
            bootstrap_aff_stack, bootstrap_critic_stack = _build_next_aff_stacks(
                current_actor_stack,
                current_critic_stack,
                bootstrap_aff_map,
                bootstrap_critic_aff_map,
            )

            if is_gate:
                gate_difficulty = (
                    bootstrap_difficulty if gate_use_difficulty else torch.zeros_like(bootstrap_difficulty)
                )
                gate_critic_difficulty = (
                    bootstrap_critic_difficulty
                    if gate_use_difficulty else torch.zeros_like(bootstrap_critic_difficulty)
                )
                out = policy(
                    bootstrap_aff_stack,
                    bootstrap_state,
                    bootstrap_goal,
                    gate_difficulty,
                    critic_affordance_map=bootstrap_critic_stack,
                    critic_robot_state=bootstrap_state,
                    critic_goal=bootstrap_goal,
                    critic_terrain_difficulty=gate_critic_difficulty,
                )
            else:
                out = policy(
                    bootstrap_aff_stack,
                    bootstrap_state,
                    bootstrap_goal,
                    bootstrap_difficulty,
                    critic_affordance_map=bootstrap_critic_stack,
                    critic_robot_state=bootstrap_state,
                    critic_goal=bootstrap_goal,
                    critic_terrain_difficulty=bootstrap_critic_difficulty,
                )
        return out.value.squeeze(-1)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    goal_change_counts = deque(maxlen=100)
    episode_collision_flags = deque(maxlen=100)
    if not resume_path:
        best_reward = float("-inf")
    
    # 训练内的 episode 回报统计
    running_returns = torch.zeros(env.num_envs, device=device)
    
    total_iterations = start_iteration + args.num_iterations
    args.log_interval = max(1, int(args.num_iterations) // 20)
    expert_interface_iters = max(1, int(math.ceil(float(args.num_iterations) * float(EXPERT_INTERFACE_RATIO))))
    print(f"\n[Main] 开始训练 (iterations={args.num_iterations}, start={start_iteration})...")
    dprint(f"  - Steps per iteration: {args.num_steps}")
    dprint(f"  - Batch size: {env.num_envs * args.num_steps}")
    if use_egpo:
        dprint(
            f"  - EGPO learning rate (guided/post): "
            f"{args.egpo_lr_guided} / {args.egpo_lr_post}"
        )
    else:
        dprint(f"  - Learning rate: {args.lr}")
    dprint(f"  - Console log interval: {args.log_interval} iters (auto: num_iterations/20)")
    if hasattr(env, "max_episode_length") and hasattr(env, "high_level_dt"):
        dprint(
            f"  - Episode budget (HL): {int(env.max_episode_length)} steps "
            f"(~{float(env.max_episode_length) * float(env.high_level_dt):.2f}s)"
        )
        if hasattr(env, "max_episode_length_low"):
            dprint(
                f"  - Episode budget (LL): {int(env.max_episode_length_low)} steps "
                f"(env.dt={float(getattr(env.env, 'dt', 0.0)):.3f}s, wrapper_decimation={int(env.decimation)})"
            )
    if use_egpo:
        dprint(f"  - Expert interface window: {expert_interface_iters} iters ({EXPERT_INTERFACE_RATIO:.0%} of run)")
    dprint(
        f"  - Non-finite handling: "
        f"{'recovery(skip+sanitize)' if bool(getattr(args, 'allow_nonfinite_recovery', False)) else 'fail-fast'}"
    )
    
    aff_stack_buf = None
    critic_aff_stack_buf = None
    moe_follow_aff_stack_buf = None
    moe_avoid_aff_stack_buf = None
    last_goal_obs = None
    teacher_stack_buf = None
    stack_reset_mask = None
    last_dones = None
    aff_stack_fill = None
    egpo_alpha_half_logged = False
    env.forced_forward_train_warmup_ratio = 0.0
    env.forced_forward_stage_start_iter = 0
    env.forced_forward_current_iter = 0
    env.forced_forward_stage_last = int(getattr(env.env, "s_avoid_stage", 1)) if bool(getattr(env.env, "s_avoid_enabled", False)) else -1
    for iteration in range(start_iteration, total_iterations):
        start_time = time.time()
        local_iteration = max(0, iteration - start_iteration)
        env.forced_forward_train_warmup_ratio = min(float(local_iteration) / 200.0, 1.0)
        if bool(getattr(env.env, "s_avoid_enabled", False)):
            current_stage = int(getattr(env.env, "s_avoid_stage", 1))
            if current_stage != int(getattr(env, "forced_forward_stage_last", current_stage)):
                env.forced_forward_stage_start_iter = local_iteration
                env.forced_forward_stage_last = current_stage
            env.forced_forward_current_iter = local_iteration
        schedule_iteration = iteration
        if env.is_s0_follow_task:
            denom = max(1, int(args.num_iterations) - 1)
            d_scene = float(np.clip(float(schedule_iteration) / float(denom), 0.0, 1.0))
        else:
            d_scene = float(np.clip(float(iteration) / float(SCENE_CURRIC_ITERS), 0.0, 1.0))
        env.set_scene_difficulty_target(d_scene)
        if env.is_s0_follow_task:
            env._apply_scene_difficulty_for_resets(None)
        
        # ============ Rollout Phase ============
        buffer.reset()
        rollout_start = time.time()

        reward_term_keys = [
            'approach',
            'cross_line_dist',
            'reach',
            'heading',
            'passable_align',
            'passable_gate',
            'crossable_align',
            'crossable_gate',
            'target_center',
            'target_visible',
            'target_lost',
            'success_bonus',
            'risk_barrier',
            'gate_smooth',
            'collision',
            'stability',
            'velocity',
            'backward',
            'body_backward',
            'row_lat',
            'row_gap',
            'row_push_penalty',
            'row_cmdx_reward',
            'avoid_smooth_penalty',
            'early_next_gap_penalty',
            'heading_keep',
            'row_cmdx_signed',
            'row_cmdx_signed_abs',
            'row_cmdx_signed_pos',
            'row_cmdx_signed_neg',
            'row_forward_dist',
            'row_gate',
            'row_gate_active',
            'row_push_active',
            'avoid_smooth_switch_valid',
            'avoid_smooth_switch_event',
            'early_next_gap_active',
            'timeout_bootstrap_bonus',
            'turn_penalty',
            'yaw_rate_penalty',
            'avoid_band_penalty',
            'time',
            'terminal_penalty',
            'total',
        ]
        reward_term_sums = {k: torch.zeros((), device=device) for k in reward_term_keys}
        gate_y_sum = torch.zeros((), device=device)
        gate_y_raw_sum = torch.zeros((), device=device)
        gate_y_safe_sum = torch.zeros((), device=device)
        gate_w_sum = torch.zeros((), device=device)
        gate_y_clamp_gap_sum = torch.zeros((), device=device)
        gate_y_w_gap_sum = torch.zeros((), device=device)
        gate_y_gap_sum = torch.zeros((), device=device)
        gate_clearance_f_sum = torch.zeros((), device=device)
        gate_risk_f_sum = torch.zeros((), device=device)
        gate_y_change_sum = torch.zeros((), device=device)
        cmd_speed_sum = torch.zeros((), device=device)
        cmd_pred_x_sum = torch.zeros((), device=device)
        cmd_pred_x_abs_sum = torch.zeros((), device=device)
        cmd_slew_x_sum = torch.zeros((), device=device)
        cmd_slew_x_abs_sum = torch.zeros((), device=device)
        cmd_exec_x_sum = torch.zeros((), device=device)
        cmd_exec_x_abs_sum = torch.zeros((), device=device)
        cmd_x_delta_abs_sum = torch.zeros((), device=device)
        cmd_x_delta_abs_active_sum = torch.zeros((), device=device)
        cmd_x_sign_switch_count = torch.zeros((), device=device)
        cmd_x_sign_switch_active_count = torch.zeros((), device=device)
        cmd_x_switch_valid_count = torch.zeros((), device=device)
        cmd_x_switch_valid_active_count = torch.zeros((), device=device)
        cmd_x_sign_switch_gap_count = torch.zeros((), device=device)
        cmd_x_switch_valid_gap_count = torch.zeros((), device=device)
        cmd_x_gate_active_count = torch.zeros((), device=device)
        body_forward_speed_sum = torch.zeros((), device=device)
        body_backward_speed_sum = torch.zeros((), device=device)
        goal_dist_sum = torch.zeros((), device=device)
        collision_rate_sum = torch.zeros((), device=device)
        obstacle_contact_candidate_rate_sum = torch.zeros((), device=device)
        strict_obstacle_contact_rate_sum = torch.zeros((), device=device)
        obstacle_contact_candidate_min_clearance_sum = torch.zeros((), device=device)
        avoid_band_active_rate_sum = torch.zeros((), device=device)
        avoid_band_outside_dist_sum = torch.zeros((), device=device)
        collision_force_samples = []
        collision_threshold_value = None
        collision_threshold_src_value = None
        collision_indices_src_value = None
        clearance_sum = torch.zeros((), device=device)
        passable_occ_ratio_sum = torch.zeros((), device=device)
        crossable_width_sum = torch.zeros((), device=device)
        risk_scale_sum = torch.zeros((), device=device)
        cmd_jerk_lin_sum = torch.zeros((), device=device)
        cmd_jerk_ang_sum = torch.zeros((), device=device)
        near_miss_excess_sum = torch.zeros((), device=device)
        near_miss_step_sum = torch.zeros((), device=device)
        stuck_step_sum = torch.zeros((), device=device)
        cmd_active_step_sum = torch.zeros((), device=device)
        gate_switch_count = torch.zeros((), device=device)
        prev_y_eff = torch.zeros((env.num_envs,), device=device)
        goal_world_delta_sum = torch.zeros((), device=device)
        target_speed_sum = torch.zeros((), device=device)
        target_turn_event_sum = torch.zeros((), device=device)
        target_preturn_event_sum = torch.zeros((), device=device)
        target_reflect_event_sum = torch.zeros((), device=device)
        target_turn_count_sum = torch.zeros((), device=device)
        target_preturn_count_sum = torch.zeros((), device=device)
        target_reflect_count_sum = torch.zeros((), device=device)
        target_reset_dist_error_sum = torch.zeros((), device=device)
        target_reset_bearing_error_sum = torch.zeros((), device=device)
        aff_stack_delta_sum = torch.zeros((), device=device)
        aff_stack_std_sum = torch.zeros((), device=device)
        aff_stack_filled_sum = torch.zeros((), device=device)
        prev_cmd_slew = torch.zeros((env.num_envs, 3), device=device)
        reset_mask_prev = torch.ones((env.num_envs,), dtype=torch.bool, device=device)
        cmd_x_switch_threshold = 0.02
        expert_action_diff_sum = torch.zeros((), device=device)
        egpo_heading_align_cos_sum = torch.zeros((), device=device)
        egpo_heading_error_deg_samples = []
        egpo_yaw_resp_sign_match_sum = torch.zeros((), device=device)
        egpo_yaw_resp_sign_count = torch.zeros((), device=device)
        egpo_policy_cmd_sum = torch.zeros(3, device=device)
        egpo_expert_cmd_sum = torch.zeros(3, device=device)
        egpo_cmd_count = torch.zeros((), device=device)
        clearance_min_value = None
        avoid_stage_value = 0.0
        avoid_stage_window_value = 0.0
        avoid_shrink_window_value = 0.0
        avoid_stage_completed_eps_value = 0.0
        avoid_corridor_width_value = 0.0
        avoid_collision_rate100_value = 0.0
        avoid_collision_rate50_value = 0.0
        avoid_exposure_rate100_value = 0.0
        avoid_progress_rate_value = 0.0
        avoid_success_rate_value = 0.0
        avoid_row_success_rate_value = 0.0
        avoid_nearest_obstacle_dist_value = 5.0
        avoid_stage_switch_event_value = 0.0
        avoid_stage_switch_from_value = 0.0
        avoid_stage_switch_to_value = 0.0
        avoid_stage_switch_collision_value = 0.0
        avoid_stage_switch_exposure_value = 0.0
        avoid_stage_switch_progress_value = 0.0
        avoid_stage_switch_success_value = 0.0
        avoid_stage_switch_row_success_value = 0.0
        avoid_stage4_shrink_event_value = 0.0
        avoid_stage4_shrink_from_width_value = 0.0
        avoid_stage4_shrink_to_width_value = 0.0
        avoid_goal_sample_retry_mean_value = 0.0
        avoid_goal_sample_fallback_rate_value = 0.0
        avoid_goal_behind_rate_value = 0.0
        avoid_goal_side_rate_value = 0.0
        avoid_preset_retry_mean_value = 0.0
        avoid_preset_sample_fail_mean_value = 0.0
        avoid_preset_passage_fail_mean_value = 0.0
        avoid_preset_min_y_gap_mean_value = 0.0
        avoid_preset_passage_depth_mean_value = 0.0
        avoid_preset_core_depth_mean_value = 0.0
        avoid_goal_retry_total_base = float(getattr(env.env, "_avoid_goal_stats_retry_total", 0.0))
        avoid_goal_retry_count_base = float(getattr(env.env, "_avoid_goal_stats_retry_count", 0.0))
        avoid_goal_fallback_count_base = float(getattr(env.env, "_avoid_goal_stats_fallback_count", 0.0))
        avoid_goal_behind_count_base = float(getattr(env.env, "_avoid_goal_stats_behind_count", 0.0))
        avoid_goal_side_count_base = float(getattr(env.env, "_avoid_goal_stats_side_count", 0.0))
        avoid_goal_count_base = float(getattr(env.env, "_avoid_goal_stats_goal_count", 0.0))
        avoid_goal_count_s123_base = float(getattr(env.env, "_avoid_goal_stats_s123_goal_count", 0.0))
        avoid_goal_behind_count_s123_base = float(getattr(env.env, "_avoid_goal_stats_s123_behind_count", 0.0))
        avoid_goal_side_count_s123_base = float(getattr(env.env, "_avoid_goal_stats_s123_side_count", 0.0))
        avoid_goal_fallback_count_s123_base = float(getattr(env.env, "_avoid_goal_stats_s123_fallback_count", 0.0))
        avoid_goal_count_s4_base = float(getattr(env.env, "_avoid_goal_stats_s4_goal_count", 0.0))
        avoid_goal_behind_count_s4_base = float(getattr(env.env, "_avoid_goal_stats_s4_behind_count", 0.0))
        avoid_goal_side_count_s4_base = float(getattr(env.env, "_avoid_goal_stats_s4_side_count", 0.0))
        avoid_goal_fallback_count_s4_base = float(getattr(env.env, "_avoid_goal_stats_s4_fallback_count", 0.0))
        timeout_bootstrap_count = torch.zeros((), device=device)
        policy_nonfinite_action_count = 0
        expert_nonfinite_action_count = 0
        teacher_nonfinite_action_count = 0
        nonfinite_input_sample_count = 0
        allow_nonfinite_recovery = bool(getattr(args, "allow_nonfinite_recovery", False))
        local_iteration = iteration - start_iteration
        log_memory_this_iter = debug_memory and (local_iteration % debug_memory_interval == 0)
        if log_memory_this_iter and debug_memory_device_index is not None:
            torch.cuda.reset_peak_memory_stats(debug_memory_device_index)
            print(
                f"[MemDebug][iter={iteration}] start "
                f"{_cuda_mem_stats_text(debug_memory_device_index)}"
            )
        expert_alpha_rollout = (
            _get_expert_alpha(schedule_iteration, expert_interface_iters)
            if use_egpo else 0.0
        )
        prev_goal_world_rollout = None
        if hasattr(env.env, "goal_world"):
            prev_goal_world_rollout = env.env.goal_world.clone()
        
        for step in range(args.num_steps):
            # 准备输入
            state = obs_dict['state']
            goal_raw = obs_dict['goal']
            egpo_diag_yaw_before = None
            egpo_diag_cmd_omega = None
            
            use_student_aff = args.mode != 'teacher'
            if is_gate and moe_use_student_aff:
                use_student_aff = True
            if is_gate and args.mode == 'teacher' and moe_use_student_aff:
                print("[Warn] moe_use_student_aff=True 但当前 mode=teacher，仍会使用 student affordance。")
            actor_gt_aff_map, actor_gt_difficulty = _get_actor_gt_inputs(obs_dict)
            critic_aff_map, critic_difficulty = _get_critic_gt_inputs(obs_dict)
            follow_aff_map = None
            follow_difficulty = None
            avoid_aff_map = None
            avoid_difficulty = None
            if is_gate:
                follow_gt_aff_map, follow_gt_difficulty = _get_moe_follow_gt_inputs(obs_dict)
                avoid_gt_aff_map, avoid_gt_difficulty = _get_moe_avoid_gt_inputs(obs_dict)
                if use_student_aff:
                    (
                        follow_aff_map,
                        follow_difficulty,
                        avoid_aff_map,
                        avoid_difficulty,
                    ) = _predict_moe_expert_inputs(obs_dict)
                else:
                    follow_aff_map, follow_difficulty = follow_gt_aff_map, follow_gt_difficulty
                    avoid_aff_map, avoid_difficulty = avoid_gt_aff_map, avoid_gt_difficulty
                aff_map, difficulty = avoid_aff_map, avoid_difficulty
            elif use_student_aff:
                aff_map, difficulty = _predict_actor_inputs(obs_dict)
            else:
                aff_map, difficulty = actor_gt_aff_map, actor_gt_difficulty

            if use_avoid_local_map:
                env.clearance_affordance_override = aff_map
                env.clearance_override = None
                env.reward_affordance_override = aff_map
            elif use_student_aff:
                env.clearance_affordance_override = aff_map
                env.clearance_override = None
                env.reward_affordance_override = aff_map
            else:
                env.clearance_affordance_override = None
                env.clearance_override = None
                env.reward_affordance_override = None

            # 初始化/更新 aff 堆叠
            if aff_stack_buf is None:
                aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
                critic_aff_stack_buf = critic_aff_map.repeat(1, aff_stack, 1, 1)
                aff_stack_fill = torch.ones(env.num_envs, device=device)
                if is_gate:
                    moe_follow_aff_stack_buf = follow_aff_map.repeat(1, aff_stack, 1, 1)
                    moe_avoid_aff_stack_buf = avoid_aff_map.repeat(1, aff_stack, 1, 1)
                if args.mode == 'student' and teacher_model is not None and not is_gate:
                    teacher_stack_buf = actor_gt_aff_map.repeat(1, aff_stack, 1, 1)
            else:
                reset_mask = stack_reset_mask
                if stack_reset_mask is not None and stack_reset_mask.any():
                    aff_stack_buf[stack_reset_mask] = aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    critic_aff_stack_buf[stack_reset_mask] = critic_aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    if is_gate:
                        moe_follow_aff_stack_buf[stack_reset_mask] = follow_aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                        moe_avoid_aff_stack_buf[stack_reset_mask] = avoid_aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    aff_stack_fill[stack_reset_mask] = 1
                    if teacher_stack_buf is not None:
                        teacher_stack_buf[stack_reset_mask] = actor_gt_aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    stack_reset_mask = None
                aff_stack_buf = torch.roll(aff_stack_buf, shifts=-aff_map.shape[1], dims=1)
                aff_stack_buf[:, -aff_map.shape[1]:, :, :] = aff_map
                critic_aff_stack_buf = torch.roll(critic_aff_stack_buf, shifts=-critic_aff_map.shape[1], dims=1)
                critic_aff_stack_buf[:, -critic_aff_map.shape[1]:, :, :] = critic_aff_map
                if is_gate:
                    moe_follow_aff_stack_buf = torch.roll(
                        moe_follow_aff_stack_buf, shifts=-follow_aff_map.shape[1], dims=1
                    )
                    moe_follow_aff_stack_buf[:, -follow_aff_map.shape[1]:, :, :] = follow_aff_map
                    moe_avoid_aff_stack_buf = torch.roll(
                        moe_avoid_aff_stack_buf, shifts=-avoid_aff_map.shape[1], dims=1
                    )
                    moe_avoid_aff_stack_buf[:, -avoid_aff_map.shape[1]:, :, :] = avoid_aff_map
                if teacher_stack_buf is not None:
                    teacher_stack_buf = torch.roll(teacher_stack_buf, shifts=-actor_gt_aff_map.shape[1], dims=1)
                    teacher_stack_buf[:, -actor_gt_aff_map.shape[1]:, :, :] = actor_gt_aff_map
                if aff_stack > 1:
                    if reset_mask is None:
                        aff_stack_fill = torch.clamp(aff_stack_fill + 1, max=aff_stack)
                    else:
                        inc_mask = ~reset_mask
                        if inc_mask.any():
                            aff_stack_fill[inc_mask] = torch.clamp(aff_stack_fill[inc_mask] + 1, max=aff_stack)
                else:
                    aff_stack_fill.fill_(1)

            # 短时记忆诊断：堆叠帧变化/方差/填满比例
            if aff_stack > 1:
                base_channels = aff_map.shape[1]
                stack_h, stack_w = aff_map.shape[2], aff_map.shape[3]
                stack = aff_stack_buf.reshape(env.num_envs, aff_stack, base_channels, stack_h, stack_w)
                delta = (stack[:, 1:] - stack[:, :-1]).abs().mean(dim=(1, 2, 3, 4))
                stack_std = stack.std(dim=1, unbiased=False).mean(dim=(1, 2, 3))
                aff_stack_delta_sum += delta.mean()
                aff_stack_std_sum += stack_std.mean()
                aff_stack_filled_sum += (aff_stack_fill / aff_stack).mean()
            else:
                aff_stack_delta_sum += 0.0
                aff_stack_std_sum += 0.0
                aff_stack_filled_sum += 1.0

            # T1: goal occlusion (weak, input-only)
            if getattr(args, "t1_goal_occlusion", False):
                prob = float(getattr(args, "t1_goal_occlusion_prob", 0.02))
                max_steps = int(getattr(args, "t1_goal_occlusion_len", 10))
                if prob > 0.0 and max_steps > 0:
                    if (
                        not hasattr(env, "t1_occ_steps")
                        or env.t1_occ_steps is None
                        or env.t1_occ_steps.shape[0] != env.num_envs
                    ):
                        env.t1_occ_steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
                    active = env.t1_occ_steps > 0
                    start = (torch.rand(env.num_envs, device=device) < prob) & (~active)
                    if start.any():
                        env.t1_occ_steps[start] = torch.randint(1, max_steps + 1, (start.sum(),), device=device)
                    occ_mask = env.t1_occ_steps > 0
                    goal, last_goal_obs = apply_goal_occlusion(goal_raw, last_goal_obs, occ_mask)
                    env.t1_occ_steps[occ_mask] -= 1
                else:
                    goal = goal_raw
                    last_goal_obs = goal_raw.detach()
            else:
                goal = goal_raw
                last_goal_obs = goal_raw.detach()

            finite_masks = [
                _row_finite_mask(state),
                _row_finite_mask(goal),
                _row_finite_mask(aff_stack_buf),
                _row_finite_mask(critic_aff_stack_buf),
                _row_finite_mask(difficulty),
                _row_finite_mask(critic_difficulty),
            ]
            if is_gate:
                finite_masks.extend([
                    _row_finite_mask(moe_follow_aff_stack_buf),
                    _row_finite_mask(moe_avoid_aff_stack_buf),
                    _row_finite_mask(follow_difficulty),
                    _row_finite_mask(avoid_difficulty),
                ])
            finite_input_mask = finite_masks[0].clone()
            for mask in finite_masks[1:]:
                finite_input_mask &= mask
            if not bool(finite_input_mask.all().item()):
                bad_count = int((~finite_input_mask).sum().item())
                nonfinite_input_sample_count += bad_count
                if not allow_nonfinite_recovery:
                    raise RuntimeError(
                        f"[NonFinite-FailFast] policy inputs contain non-finite values "
                        f"(invalid_envs={bad_count}). Use --allow_nonfinite_recovery to sanitize and mask these samples."
                    )
                state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
                goal = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
                aff_stack_buf = torch.nan_to_num(aff_stack_buf, nan=0.0, posinf=0.0, neginf=0.0)
                critic_aff_stack_buf = torch.nan_to_num(critic_aff_stack_buf, nan=0.0, posinf=0.0, neginf=0.0)
                difficulty = torch.nan_to_num(difficulty, nan=0.0, posinf=0.0, neginf=0.0)
                critic_difficulty = torch.nan_to_num(critic_difficulty, nan=0.0, posinf=0.0, neginf=0.0)
                if is_gate:
                    moe_follow_aff_stack_buf = torch.nan_to_num(
                        moe_follow_aff_stack_buf, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    moe_avoid_aff_stack_buf = torch.nan_to_num(
                        moe_avoid_aff_stack_buf, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    follow_difficulty = torch.nan_to_num(follow_difficulty, nan=0.0, posinf=0.0, neginf=0.0)
                    avoid_difficulty = torch.nan_to_num(avoid_difficulty, nan=0.0, posinf=0.0, neginf=0.0)
            action_valid = finite_input_mask.clone()

            # Policy 决策
            teacher_action = None
            expert_action = None
            gate_y = None
            cmd_used = None
            if is_gate:
                gate_difficulty = difficulty if gate_use_difficulty else torch.zeros_like(difficulty)
                gate_critic_difficulty = critic_difficulty if gate_use_difficulty else torch.zeros_like(critic_difficulty)
                moe_expert_state = _get_moe_expert_state_inputs(state)
                with torch.no_grad():
                    cmd_f = _compute_moe_follow_cmd_from_goal(
                        moe_expert_state,
                        goal,
                        reset_mask_prev,
                    )
                    cmd_a, _ = avoid_model.get_action(
                        moe_avoid_aff_stack_buf, moe_expert_state, goal, avoid_difficulty,
                        deterministic=bool(getattr(args, "moe_expert_deterministic", True))
                    )
                    cmd_f, cmd_f_bad = _sanitize_or_fail_action_tensor(
                        "moe_follow_action",
                        cmd_f.detach(),
                        allow_recovery=allow_nonfinite_recovery,
                    )
                    cmd_a, cmd_a_bad = _sanitize_or_fail_action_tensor(
                        "moe_avoid_action",
                        cmd_a.detach(),
                        allow_recovery=allow_nonfinite_recovery,
                    )
                    expert_nonfinite_action_count += int(cmd_f_bad.sum().item()) + int(cmd_a_bad.sum().item())
                    action_valid = action_valid & (~cmd_f_bad) & (~cmd_a_bad)
                with torch.no_grad():
                    gate_y, _ = policy.get_action(
                        aff_stack_buf,
                        state,
                        goal,
                        gate_difficulty,
                        deterministic=False,
                        critic_affordance_map=critic_aff_stack_buf,
                        critic_robot_state=state,
                        critic_goal=goal,
                        critic_terrain_difficulty=gate_critic_difficulty,
                    )
                    gate_y, gate_y_bad = _sanitize_or_fail_action_tensor(
                        "gate_action",
                        gate_y.detach(),
                        allow_recovery=allow_nonfinite_recovery,
                    )
                    policy_nonfinite_action_count += int(gate_y_bad.sum().item())
                    action_valid = action_valid & (~gate_y_bad)
                    gate_y_raw = gate_y.clone()
                    gate_y_prev = prev_y_eff.clone()
                    gate_diag = resolve_moe_gate_pcr(env, args, aff_map, gate_y_raw, cmd_f, cmd_a)
                    gate_y = gate_diag["gate_y"]
                    y_eff = gate_diag["y_eff"]
                    if bool(gate_diag["gate_safe_clamp_mask"].any().item()):
                        action_valid = action_valid & torch.isclose(gate_y, gate_y_raw, atol=1e-6, rtol=0.0)
                    log_prob, value, _, _ = policy.evaluate_actions(
                        aff_stack_buf,
                        state,
                        goal,
                        gate_difficulty,
                        gate_y,
                        critic_affordance_map=critic_aff_stack_buf,
                        critic_robot_state=state,
                        critic_goal=goal,
                        critic_terrain_difficulty=gate_critic_difficulty,
                    )
                    cmd = gate_diag["cmd"]
                    next_obs, rewards, dones, env_info = env.step(cmd, y_eff, gate_y_raw=gate_y_raw)
                    timeout_mask = env_info.get("timeout", None) if env_info is not None else None
                    timeout_bootstrap_obs = env_info.get("timeout_bootstrap_obs", None) if env_info is not None else None
                    if (
                        timeout_mask is not None
                        and timeout_bootstrap_obs is not None
                        and torch.is_tensor(timeout_mask)
                        and bool(timeout_mask.any().item())
                    ):
                        timeout_values = _compute_bootstrap_value_for_obs(
                            timeout_bootstrap_obs,
                            aff_stack_buf,
                            critic_aff_stack_buf,
                        )
                        timeout_bonus = args.gamma * timeout_values * timeout_mask.float()
                        rewards = rewards + timeout_bonus
                        if env_info is not None and isinstance(env_info, dict):
                            reward_terms_info = env_info.get("reward_terms", None)
                            if isinstance(reward_terms_info, dict):
                                prev_bonus = reward_terms_info.get("timeout_bootstrap_bonus", None)
                                if torch.is_tensor(prev_bonus):
                                    reward_terms_info["timeout_bootstrap_bonus"] = prev_bonus + timeout_bonus.detach()
                                else:
                                    reward_terms_info["timeout_bootstrap_bonus"] = timeout_bonus.detach()
                        timeout_bootstrap_count += timeout_mask.float().sum()
                    rewards = rewards.detach()
                    # Record raw/effective gate and command-conditioned risk proxies for diagnostics.
                    if env_info is not None and isinstance(env_info, dict):
                        post_info = env_info.get("post_info", None)
                        if post_info is not None and isinstance(post_info, dict):
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
                            post_info["post_safe_distance"] = gate_diag["post_safe_distance"].detach().clone()
                            post_info["post_free_distance"] = gate_diag["post_free_distance"].detach().clone()
                action = gate_y.unsqueeze(-1)
                cmd_used = cmd
            else:
                with torch.no_grad():
                    cmd, _ = policy.get_action(
                        aff_stack_buf,
                        state,
                        goal,
                        difficulty,
                        deterministic=False,
                        critic_affordance_map=critic_aff_stack_buf,
                        critic_robot_state=state,
                        critic_goal=goal,
                        critic_terrain_difficulty=critic_difficulty,
                    )
                    cmd, cmd_bad = _sanitize_or_fail_action_tensor(
                        "policy_action",
                        cmd.detach(),
                        allow_recovery=allow_nonfinite_recovery,
                    )
                    policy_nonfinite_action_count += int(cmd_bad.sum().item())
                    action_valid = action_valid & (~cmd_bad)
                    if use_expert_guidance and compute_s0_follow_expert_cmd is not None:
                        policy_cmd_raw = cmd
                        target_world_xy = getattr(env.env, "target_world", None)
                        target_vel_world_xy = getattr(env.env, "target_vel_world", None)
                        target_heading = getattr(env.env, "target_heading", None)
                        quat = env.env.root_states[:, 3:7]
                        x_q, y_q, z_q, w_q = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                        robot_heading = torch.atan2(
                            2.0 * (w_q * z_q + x_q * y_q),
                            1.0 - 2.0 * (y_q * y_q + z_q * z_q),
                        )
                        heading_offset = float(getattr(getattr(env, "reward_cfg", None), "heading_offset_rad", 0.0))
                        robot_heading_for_expert = robot_heading + heading_offset
                        robot_heading_for_expert = torch.atan2(
                            torch.sin(robot_heading_for_expert),
                            torch.cos(robot_heading_for_expert),
                        )
                        if target_world_xy is None:
                            target_world_xy = torch.zeros(env.num_envs, 2, device=device)
                        if target_vel_world_xy is None:
                            target_vel_world_xy = torch.zeros(env.num_envs, 2, device=device)
                        if target_heading is None:
                            target_heading = torch.zeros(env.num_envs, device=device)
                        expert_action = compute_s0_follow_expert_cmd(
                            robot_pos_world_xy=env.env.root_states[:, :2],
                            robot_heading=robot_heading_for_expert,
                            target_world_xy=target_world_xy,
                            # Keep BC labels consistent with policy-observable inputs only.
                            target_vel_world_xy=None,
                            target_heading=None,
                            cmd_scale=cmd_scale,
                            reset_mask=reset_mask_prev,
                        )
                        expert_action, expert_bad = _sanitize_or_fail_action_tensor(
                            "follow_expert_action",
                            expert_action,
                            allow_recovery=allow_nonfinite_recovery,
                        )
                        expert_nonfinite_action_count += int(expert_bad.sum().item())
                        if use_egpo:
                            egpo_policy_cmd_sum += policy_cmd_raw.sum(dim=0)
                            egpo_expert_cmd_sum += expert_action.sum(dim=0)
                            egpo_cmd_count += float(policy_cmd_raw.shape[0])
                        if debug and use_egpo:
                            # Keep debug heading diagnostics on the same branch used by expert labels
                            # in S0: no target vel/heading, geometric fallback direction only.
                            to_target_world = target_world_xy - env.env.root_states[:, :2]
                            dir_world = to_target_world / torch.norm(
                                to_target_world, dim=1, keepdim=True
                            ).clamp_min(1e-6)
                            default_dir = torch.zeros_like(dir_world)
                            default_dir[:, 1] = 1.0
                            dir_world = torch.where(
                                torch.norm(to_target_world, dim=1, keepdim=True) > 1e-6,
                                dir_world,
                                default_dir,
                            )
                            robot_fwd_world = torch.stack(
                                [torch.sin(robot_heading_for_expert), torch.cos(robot_heading_for_expert)], dim=1
                            )
                            align_cos = torch.sum(robot_fwd_world * dir_world, dim=1).clamp(-1.0, 1.0)
                            egpo_heading_align_cos_sum += align_cos.sum()
                            heading_err_deg = torch.rad2deg(torch.acos(align_cos))
                            egpo_heading_error_deg_samples.append(heading_err_deg.detach())
                        if use_egpo:
                            expert_action_diff_sum += torch.norm(cmd - expert_action, dim=1).sum()
                            if expert_alpha_rollout > 0.0:
                                cmd = expert_alpha_rollout * expert_action + (1.0 - expert_alpha_rollout) * cmd
                                action_valid.zero_()
                            if debug:
                                egpo_diag_yaw_before = robot_heading.detach()
                                egpo_diag_cmd_omega = cmd[:, 2].detach()
                        elif use_follow_expert:
                            cmd = expert_action
                            action_valid.zero_()
                    log_prob, value, _, _ = policy.evaluate_actions(
                        aff_stack_buf,
                        state,
                        goal,
                        difficulty,
                        cmd,
                        critic_affordance_map=critic_aff_stack_buf,
                        critic_robot_state=state,
                        critic_goal=goal,
                        critic_terrain_difficulty=critic_difficulty,
                    )

                    if args.mode == 'student' and teacher_model is not None:
                        if teacher_stack_buf is None:
                            teacher_stack_buf = actor_gt_aff_map.repeat(1, aff_stack, 1, 1)
                        t_cmd, _ = teacher_model.get_action(
                            teacher_stack_buf,
                            state, goal,
                            actor_gt_difficulty,
                            deterministic=True
                        )
                        teacher_action, teacher_bad = _sanitize_or_fail_action_tensor(
                            "teacher_action",
                            t_cmd.detach(),
                            allow_recovery=allow_nonfinite_recovery,
                        )
                        teacher_nonfinite_action_count += int(teacher_bad.sum().item())

                    next_obs, rewards, dones, env_info = env.step(cmd)
                    timeout_mask = env_info.get("timeout", None) if env_info is not None else None
                    timeout_bootstrap_obs = env_info.get("timeout_bootstrap_obs", None) if env_info is not None else None
                    if (
                        timeout_mask is not None
                        and timeout_bootstrap_obs is not None
                        and torch.is_tensor(timeout_mask)
                        and bool(timeout_mask.any().item())
                    ):
                        timeout_values = _compute_bootstrap_value_for_obs(
                            timeout_bootstrap_obs,
                            aff_stack_buf,
                            critic_aff_stack_buf,
                        )
                        timeout_bonus = args.gamma * timeout_values * timeout_mask.float()
                        rewards = rewards + timeout_bonus
                        if env_info is not None and isinstance(env_info, dict):
                            reward_terms_info = env_info.get("reward_terms", None)
                            if isinstance(reward_terms_info, dict):
                                prev_bonus = reward_terms_info.get("timeout_bootstrap_bonus", None)
                                if torch.is_tensor(prev_bonus):
                                    reward_terms_info["timeout_bootstrap_bonus"] = prev_bonus + timeout_bonus.detach()
                                else:
                                    reward_terms_info["timeout_bootstrap_bonus"] = timeout_bonus.detach()
                        timeout_bootstrap_count += timeout_mask.float().sum()
                    rewards = rewards.detach()
                    if debug and use_egpo and egpo_diag_yaw_before is not None and egpo_diag_cmd_omega is not None:
                        cmd_omega_eval = egpo_diag_cmd_omega
                        if env_info is not None and isinstance(env_info, dict):
                            post_info = env_info.get("post_info", None)
                            if post_info is not None and isinstance(post_info, dict):
                                cmd_override = post_info.get("cmd_override_final", None)
                                if cmd_override is None:
                                    cmd_override = post_info.get("cmd_exec_mean", None)
                                if cmd_override is not None:
                                    if not torch.is_tensor(cmd_override):
                                        cmd_override = torch.as_tensor(cmd_override, device=device)
                                    if cmd_override.dim() == 1:
                                        cmd_override = cmd_override.unsqueeze(0)
                                    if cmd_override.shape[0] == env.num_envs and cmd_override.shape[1] >= 3:
                                        cmd_omega_eval = cmd_override[:, 2].detach()
                        quat_after = env.env.root_states[:, 3:7]
                        x_a, y_a, z_a, w_a = quat_after[:, 0], quat_after[:, 1], quat_after[:, 2], quat_after[:, 3]
                        yaw_after = torch.atan2(
                            2.0 * (w_a * z_a + x_a * y_a),
                            1.0 - 2.0 * (y_a * y_a + z_a * z_a),
                        )
                        dyaw = torch.atan2(
                            torch.sin(yaw_after - egpo_diag_yaw_before),
                            torch.cos(yaw_after - egpo_diag_yaw_before),
                        )
                        valid = torch.abs(cmd_omega_eval) > 0.05
                        if valid.any():
                            sign_match = (cmd_omega_eval[valid] * dyaw[valid]) > 0.0
                            egpo_yaw_resp_sign_match_sum += sign_match.float().sum()
                            egpo_yaw_resp_sign_count += valid.float().sum()
                action = cmd
                gate_y_prev = None
                cmd_used = cmd
            last_dones = dones.clone()

            # 存储数据
            buffer.add(
                state.detach(),
                goal.detach(),
                aff_stack_buf.detach(),
                difficulty.detach(),
                critic_aff_stack_buf.detach(),
                critic_difficulty.detach(),
                action.detach(),
                log_prob.detach(),
                value.detach(),
                rewards.detach(),
                dones.detach(),
                action_valid.detach(),
                teacher_action.detach() if teacher_action is not None else None,
                expert_action.detach() if expert_action is not None else None,
            )

            # 统计分量与行为
            goal_dist_sum += torch.norm(goal, dim=1).sum()
            if is_gate:
                gate_y_sum += y_eff.sum()
                gate_y_raw_sum += gate_y_raw.sum()
                if 'gate_diag' in locals() and isinstance(gate_diag, dict):
                    gate_w_sum += gate_diag["w"].sum()
                    gate_y_safe = gate_diag.get("gate_y_safe", gate_diag["gate_y"])
                    gate_y_safe_sum += gate_y_safe.sum()
                    gate_y_clamp_gap_sum += torch.abs(gate_y_safe - gate_diag["gate_y_raw"]).sum()
                    gate_y_w_gap_sum += torch.abs(gate_diag["y_eff"] - gate_y_safe).sum()
                    gate_y_gap_sum += torch.abs(gate_diag["y_eff"] - gate_diag["gate_y_raw"]).sum()
                    gate_clearance_f_sum += gate_diag["clearance_F"].sum()
                    gate_risk_f_sum += gate_diag["risk_F"].sum()
                if gate_y_prev is not None:
                    delta_gate = y_eff - gate_y_prev
                    if reset_mask_prev.any():
                        # 使用上一轮 reset_mask_prev，屏蔽跨 episode 的 Δy_eff
                        delta_gate[reset_mask_prev] = 0.0
                    gate_y_change_sum += torch.abs(delta_gate).sum()
                    gate_switch_count += (torch.abs(delta_gate) > GATE_SWITCH_DY_DEFAULT).float().sum()
                    prev_y_eff = y_eff.detach().clone()
            cmd_diag_exec = None
            post_info = env_info.get('post_info', None) if env_info is not None else None
            if post_info is not None and isinstance(post_info, dict):
                cmd_diag_exec = post_info.get('cmd_exec_mean', None)
                if cmd_diag_exec is None:
                    cmd_diag_exec = post_info.get('cmd_override_final', None)
                if cmd_diag_exec is None:
                    cmd_diag_exec = post_info.get('cmd_post', None)
                if cmd_diag_exec is not None and not torch.is_tensor(cmd_diag_exec):
                    cmd_diag_exec = torch.as_tensor(cmd_diag_exec, device=device)
                if torch.is_tensor(cmd_diag_exec) and cmd_diag_exec.dim() == 1:
                    cmd_diag_exec = cmd_diag_exec.unsqueeze(0)
            if cmd_diag_exec is None and hasattr(env.env, "commands"):
                cmd_diag_exec = env.env.commands[:, :3]
            if cmd_diag_exec is None:
                cmd_diag_exec = cmd_used
            if cmd_used is not None:
                cmd_pred_x_sum += cmd_used[:, 0].sum()
                cmd_pred_x_abs_sum += torch.abs(cmd_used[:, 0]).sum()
            cmd_diag_slew = None
            if post_info is not None and isinstance(post_info, dict):
                cmd_diag_slew = post_info.get('cmd_post', None)
                if cmd_diag_slew is None:
                    cmd_diag_slew = post_info.get('cmd_slew', None)
                if cmd_diag_slew is not None and not torch.is_tensor(cmd_diag_slew):
                    cmd_diag_slew = torch.as_tensor(cmd_diag_slew, device=device)
                if torch.is_tensor(cmd_diag_slew) and cmd_diag_slew.dim() == 1:
                    cmd_diag_slew = cmd_diag_slew.unsqueeze(0)
            if cmd_diag_slew is None:
                cmd_diag_slew = cmd_used
            if cmd_diag_slew is not None:
                cmd_slew_x_sum += cmd_diag_slew[:, 0].sum()
                cmd_slew_x_abs_sum += torch.abs(cmd_diag_slew[:, 0]).sum()
            if cmd_diag_exec is not None:
                cmd_speed_sum += torch.norm(cmd_diag_exec[:, :2], dim=1).sum()
                cmd_exec_x_sum += cmd_diag_exec[:, 0].sum()
                cmd_exec_x_abs_sum += torch.abs(cmd_diag_exec[:, 0]).sum()
            if hasattr(env.env, "base_lin_vel"):
                body_lin_vel = env.env.base_lin_vel
                body_forward = body_lin_vel[:, 1]
                body_forward_speed_sum += body_forward.sum()
                body_backward_speed_sum += torch.clamp(-body_forward, min=0.0).sum()
            if hasattr(env.env, "target_speed"):
                target_speed = torch.nan_to_num(env.env.target_speed, nan=0.0, posinf=0.0, neginf=0.0)
                target_speed_sum += target_speed.sum()
            elif hasattr(env.env, "target_vel_world"):
                target_vel = torch.nan_to_num(env.env.target_vel_world, nan=0.0, posinf=0.0, neginf=0.0)
                target_speed_sum += torch.norm(target_vel, dim=1).sum()
            if hasattr(env.env, "goal_world"):
                goal_world_now = torch.nan_to_num(env.env.goal_world, nan=0.0, posinf=0.0, neginf=0.0)
                if prev_goal_world_rollout is not None and prev_goal_world_rollout.shape == goal_world_now.shape:
                    goal_world_delta_sum += torch.norm(goal_world_now - prev_goal_world_rollout, dim=1).sum()
                prev_goal_world_rollout = goal_world_now.clone()
            target_turn_events = env_info.get('target_turn_events', None) if env_info is not None else None
            if target_turn_events is not None:
                target_turn_event_sum += torch.nan_to_num(target_turn_events, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_preturn_events = env_info.get('target_preturn_events', None) if env_info is not None else None
            if target_preturn_events is not None:
                target_preturn_event_sum += torch.nan_to_num(target_preturn_events, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_reflect_events = env_info.get('target_reflect_events', None) if env_info is not None else None
            if target_reflect_events is not None:
                target_reflect_event_sum += torch.nan_to_num(target_reflect_events, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_turn_count = env_info.get('target_turn_count', None) if env_info is not None else None
            if target_turn_count is not None:
                target_turn_count_sum += torch.nan_to_num(target_turn_count, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_preturn_count = env_info.get('target_preturn_count', None) if env_info is not None else None
            if target_preturn_count is not None:
                target_preturn_count_sum += torch.nan_to_num(target_preturn_count, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_reflect_count = env_info.get('target_reflect_count', None) if env_info is not None else None
            if target_reflect_count is not None:
                target_reflect_count_sum += torch.nan_to_num(target_reflect_count, nan=0.0, posinf=0.0, neginf=0.0).sum()
            target_reset_dist_error = env_info.get('target_reset_dist_error', None) if env_info is not None else None
            if target_reset_dist_error is not None:
                target_reset_dist_error_sum += torch.nan_to_num(
                    target_reset_dist_error, nan=0.0, posinf=0.0, neginf=0.0
                ).sum()
            target_reset_bearing_error = env_info.get('target_reset_bearing_error', None) if env_info is not None else None
            if target_reset_bearing_error is not None:
                target_reset_bearing_error_sum += torch.nan_to_num(
                    target_reset_bearing_error, nan=0.0, posinf=0.0, neginf=0.0
                ).sum()
            reward_terms = env_info.get('reward_terms', None)
            if reward_terms is not None:
                for key in reward_term_keys:
                    if key in reward_terms:
                        reward_term_sums[key] += reward_terms[key].sum()
                if 'risk_scale' in reward_terms:
                    risk_scale_sum += reward_terms['risk_scale'].sum()
                if 'passable_occ_ratio' in reward_terms:
                    passable_occ_ratio_sum += reward_terms['passable_occ_ratio'].sum()
                if 'crossable_width' in reward_terms:
                    crossable_width_sum += reward_terms['crossable_width'].sum()
            collision_mask = env_info.get('collision_mask', None) if env_info is not None else None
            if collision_mask is not None:
                collision_rate_sum += collision_mask.float().sum()
            obstacle_contact_candidate_mask = env_info.get('obstacle_contact_candidate_mask', None) if env_info is not None else None
            if obstacle_contact_candidate_mask is not None:
                obstacle_contact_candidate_rate_sum += obstacle_contact_candidate_mask.float().sum()
            strict_obstacle_contact_mask = env_info.get('strict_obstacle_contact_mask', None) if env_info is not None else None
            if strict_obstacle_contact_mask is not None:
                strict_obstacle_contact_rate_sum += strict_obstacle_contact_mask.float().sum()
            obstacle_contact_candidate_min_clearance = env_info.get('obstacle_contact_candidate_min_clearance', None) if env_info is not None else None
            if obstacle_contact_candidate_min_clearance is not None:
                obstacle_contact_candidate_min_clearance_sum += obstacle_contact_candidate_min_clearance.sum()
            avoid_band_active_mask = env_info.get('avoid_band_active_mask', None) if env_info is not None else None
            if avoid_band_active_mask is not None:
                avoid_band_active_rate_sum += avoid_band_active_mask.float().sum()
            avoid_band_outside_dist = env_info.get('avoid_band_outside_dist', None) if env_info is not None else None
            if avoid_band_outside_dist is not None:
                avoid_band_outside_dist_sum += avoid_band_outside_dist.sum()
            collision_force_max = env_info.get('collision_force_max', None) if env_info is not None else None
            if collision_force_max is not None:
                collision_force_samples.append(collision_force_max.detach().cpu())
            collision_threshold = env_info.get('collision_threshold', None) if env_info is not None else None
            if collision_threshold is not None:
                collision_threshold_value = float(collision_threshold)
            collision_threshold_src = env_info.get('collision_threshold_src', None) if env_info is not None else None
            if collision_threshold_src is not None:
                collision_threshold_src_value = collision_threshold_src
            collision_indices_src = env_info.get('collision_indices_src', None) if env_info is not None else None
            if collision_indices_src is not None:
                collision_indices_src_value = collision_indices_src
            clearance = env_info.get('clearance', None) if env_info is not None else None
            if clearance is not None:
                clearance_sum += clearance.sum()
                clearance_min_curr = float(clearance.min().item())
                clearance_min_value = (
                    clearance_min_curr
                    if clearance_min_value is None
                    else min(clearance_min_value, clearance_min_curr)
                )
            if post_info is not None:
                cmd_exec_for_diag = post_info.get('cmd_exec_mean', None)
                if cmd_exec_for_diag is None:
                    cmd_exec_for_diag = post_info.get('cmd_override_final', None)
                if cmd_exec_for_diag is None:
                    cmd_exec_for_diag = post_info.get('cmd_post', None)
                if cmd_exec_for_diag is not None:
                    if not torch.is_tensor(cmd_exec_for_diag):
                        cmd_exec_for_diag = torch.as_tensor(cmd_exec_for_diag, device=device)
                    if cmd_exec_for_diag.dim() == 1:
                        cmd_exec_for_diag = cmd_exec_for_diag.unsqueeze(0)
                    delta_cmd = cmd_exec_for_diag - prev_cmd_slew
                    if reset_mask_prev.any():
                        delta_cmd[reset_mask_prev] = 0.0
                    cmd_jerk_lin_sum += torch.norm(delta_cmd[:, :2], dim=1).sum()
                    cmd_jerk_ang_sum += torch.abs(delta_cmd[:, 2]).sum()
                    delta_cmd_x_abs = torch.abs(delta_cmd[:, 0])
                    cmd_x_delta_abs_sum += delta_cmd_x_abs.sum()
                    if reward_terms is not None and 'row_gate_active' in reward_terms:
                        row_gate_active_mask = reward_terms['row_gate_active'] > 0.5
                    else:
                        row_gate_active_mask = torch.zeros((env.num_envs,), dtype=torch.bool, device=device)
                    cmd_x_gate_active_count += row_gate_active_mask.float().sum()
                    cmd_x_delta_abs_active_sum += delta_cmd_x_abs[row_gate_active_mask].sum()
                    prev_cmd_x = prev_cmd_slew[:, 0]
                    curr_cmd_x = cmd_exec_for_diag[:, 0]
                    switch_valid = (
                        (~reset_mask_prev)
                        & (torch.abs(prev_cmd_x) > cmd_x_switch_threshold)
                        & (torch.abs(curr_cmd_x) > cmd_x_switch_threshold)
                    )
                    sign_switch = switch_valid & ((prev_cmd_x * curr_cmd_x) < 0.0)
                    cmd_x_switch_valid_count += switch_valid.float().sum()
                    cmd_x_sign_switch_count += sign_switch.float().sum()
                    switch_valid_active = switch_valid & row_gate_active_mask
                    sign_switch_active = sign_switch & row_gate_active_mask
                    cmd_x_switch_valid_active_count += switch_valid_active.float().sum()
                    cmd_x_sign_switch_active_count += sign_switch_active.float().sum()
                    if reward_terms is not None and 'avoid_smooth_switch_valid' in reward_terms:
                        switch_valid_gap = reward_terms['avoid_smooth_switch_valid'] > 0.5
                    else:
                        switch_valid_gap = torch.zeros((env.num_envs,), dtype=torch.bool, device=device)
                    if reward_terms is not None and 'avoid_smooth_switch_event' in reward_terms:
                        sign_switch_gap = reward_terms['avoid_smooth_switch_event'] > 0.5
                    else:
                        sign_switch_gap = torch.zeros((env.num_envs,), dtype=torch.bool, device=device)
                    cmd_x_switch_valid_gap_count += switch_valid_gap.float().sum()
                    cmd_x_sign_switch_gap_count += sign_switch_gap.float().sum()
                    prev_cmd_slew = cmd_exec_for_diag.detach()
                clearance_pp = post_info.get('clearance_pp', None)
                if clearance_pp is not None:
                    if not torch.is_tensor(clearance_pp):
                        clearance_pp = torch.as_tensor(clearance_pp, device=device)
                    c_thr = post_info.get('post_safe_distance', None)
                    if c_thr is None:
                        c_thr = getattr(env.post_processor, "safe_distance", None)
                    if c_thr is not None:
                        if torch.is_tensor(c_thr):
                            c_thr_tensor = c_thr.to(device=device)
                        else:
                            c_thr_tensor = torch.tensor(float(c_thr), device=device)
                        near_miss_excess = torch.clamp(c_thr_tensor - clearance_pp, min=0.0)
                        near_miss_excess_sum += near_miss_excess.sum()
                        near_miss_step_sum += (clearance_pp < c_thr_tensor).float().sum()
            if use_avoid_local_map and hasattr(env, "env") and hasattr(env.env, "base_lin_vel"):
                actual_speed = torch.norm(env.env.base_lin_vel[:, :2], dim=1)
                actual_turn_speed = torch.abs(env.env.base_ang_vel[:, 2]) if hasattr(env.env, "base_ang_vel") else None
                cmd_exec = cmd_diag_exec
                if cmd_exec is not None:
                    cmd_speed = torch.norm(cmd_exec[:, :2], dim=1)
                    cmd_turn = torch.abs(cmd_exec[:, 2])
                    cmd_active_lin = cmd_speed > 0.15
                    cmd_active_ang = cmd_turn > 0.20
                    stuck_lin = cmd_active_lin & (actual_speed < 0.05)
                    if actual_turn_speed is None:
                        stuck_ang = torch.zeros_like(stuck_lin)
                    else:
                        stuck_ang = cmd_active_ang & (actual_turn_speed < 0.10)
                    cmd_active = cmd_active_lin | cmd_active_ang
                    stuck_mask = cmd_active & (stuck_lin | stuck_ang)
                    stuck_step_sum += stuck_mask.float().sum()
                    cmd_active_step_sum += cmd_active.float().sum()
                extras = getattr(env.env, "extras", {})
                if isinstance(extras, dict):
                    avoid_stage_value = float(extras.get("avoid_stage", avoid_stage_value))
                    avoid_stage_window_value = float(extras.get("avoid_stage_window", avoid_stage_window_value))
                    avoid_shrink_window_value = float(extras.get("avoid_shrink_window", avoid_shrink_window_value))
                    avoid_stage_completed_eps_value = float(
                        extras.get("avoid_stage_completed_episodes", avoid_stage_completed_eps_value)
                    )
                    avoid_corridor_width_value = float(extras.get("avoid_corridor_width", avoid_corridor_width_value))
                    avoid_collision_rate100_value = float(
                        extras.get("avoid_stage_collision_rate", avoid_collision_rate100_value)
                    )
                    avoid_collision_rate50_value = float(
                        extras.get("avoid_shrink_collision_rate", avoid_collision_rate50_value)
                    )
                    avoid_exposure_rate100_value = float(
                        extras.get("avoid_stage_exposure_rate", avoid_exposure_rate100_value)
                    )
                    avoid_progress_rate_value = float(
                        extras.get("avoid_stage_progress_rate", avoid_progress_rate_value)
                    )
                    avoid_success_rate_value = float(
                        extras.get("avoid_stage_success_rate", avoid_success_rate_value)
                    )
                    avoid_row_success_rate_value = float(
                        extras.get("avoid_stage_row_success_rate", avoid_row_success_rate_value)
                    )
                    avoid_nearest_obstacle_dist_value = float(extras.get("avoid_nearest_obstacle_dist", avoid_nearest_obstacle_dist_value))
                    avoid_preset_retry_mean_value = float(
                        extras.get("avoid_preset_retry_mean", avoid_preset_retry_mean_value)
                    )
                    avoid_preset_sample_fail_mean_value = float(
                        extras.get("avoid_preset_sample_fail_mean", avoid_preset_sample_fail_mean_value)
                    )
                    avoid_preset_passage_fail_mean_value = float(
                        extras.get("avoid_preset_passage_fail_mean", avoid_preset_passage_fail_mean_value)
                    )
                    avoid_preset_min_y_gap_mean_value = float(
                        extras.get("avoid_preset_min_y_gap_mean", avoid_preset_min_y_gap_mean_value)
                    )
                    avoid_preset_passage_depth_mean_value = float(
                        extras.get("avoid_preset_passage_depth_mean", avoid_preset_passage_depth_mean_value)
                    )
                    avoid_preset_core_depth_mean_value = float(
                        extras.get("avoid_preset_core_depth_mean", avoid_preset_core_depth_mean_value)
                    )
                    avoid_stage_switch_event_value = float(
                        extras.get("avoid_stage_switch_event", avoid_stage_switch_event_value)
                    )
                    avoid_stage_switch_from_value = float(
                        extras.get("avoid_stage_switch_from", avoid_stage_switch_from_value)
                    )
                    avoid_stage_switch_to_value = float(
                        extras.get("avoid_stage_switch_to", avoid_stage_switch_to_value)
                    )
                    avoid_stage_switch_collision_value = float(
                        extras.get("avoid_stage_switch_collision_rate", avoid_stage_switch_collision_value)
                    )
                    avoid_stage_switch_exposure_value = float(
                        extras.get("avoid_stage_switch_exposure_rate", avoid_stage_switch_exposure_value)
                    )
                    avoid_stage_switch_progress_value = float(
                        extras.get("avoid_stage_switch_progress_rate", avoid_stage_switch_progress_value)
                    )
                    avoid_stage_switch_success_value = float(
                        extras.get("avoid_stage_switch_success_rate", avoid_stage_switch_success_value)
                    )
                    avoid_stage_switch_row_success_value = float(
                        extras.get("avoid_stage_switch_row_success_rate", avoid_stage_switch_row_success_value)
                    )
                    avoid_stage4_shrink_event_value = float(
                        extras.get("avoid_stage4_shrink_event", avoid_stage4_shrink_event_value)
                    )
                    avoid_stage4_shrink_from_width_value = float(
                        extras.get("avoid_stage4_shrink_from_width", avoid_stage4_shrink_from_width_value)
                    )
                    avoid_stage4_shrink_to_width_value = float(
                        extras.get("avoid_stage4_shrink_to_width", avoid_stage4_shrink_to_width_value)
                    )
            reset_mask_prev = dones.clone()

            # 统计完成的 episode（优先使用高层累计的 episode_return）
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                episode = env_info.get('episode', None) if env_info is not None else None
                if episode is not None:
                    ep_len = episode['l'][done_ids].detach().cpu().tolist()
                    ep_ret = episode['r'][done_ids].detach().cpu().tolist()
                    episode_lengths.extend(ep_len)
                    episode_rewards.extend(ep_ret)
                else:
                    running_returns += rewards
                    ep_len = env_info['episode_length'][done_ids].detach().cpu().tolist()
                    episode_lengths.extend(ep_len)
                    episode_rewards.extend(running_returns[done_ids].detach().cpu().tolist())
                    running_returns[done_ids] = 0.0
                goal_changes = env_info.get('goal_change_count', None)
                if goal_changes is not None:
                    goal_change_counts.extend(goal_changes[done_ids].detach().cpu().tolist())
                if use_avoid_local_map:
                    episode_collision = env_info.get('s_avoid_episode_collision', None) if env_info is not None else None
                    if torch.is_tensor(episode_collision):
                        episode_collision_flags.extend(
                            episode_collision[done_ids].detach().to(dtype=torch.float32).cpu().tolist()
                        )
            else:
                running_returns += rewards
            
            if dones.any():
                stack_reset_mask = dones.clone()
            obs_dict = next_obs
            if dones.any():
                if last_goal_obs is not None:
                    last_goal_obs = last_goal_obs.clone()
                    last_goal_obs[dones] = obs_dict['goal'][dones]

        rollout_time = time.time() - rollout_start
        avoid_goal_retry_total_curr = float(getattr(env.env, "_avoid_goal_stats_retry_total", avoid_goal_retry_total_base))
        avoid_goal_retry_count_curr = float(getattr(env.env, "_avoid_goal_stats_retry_count", avoid_goal_retry_count_base))
        avoid_goal_fallback_count_curr = float(getattr(env.env, "_avoid_goal_stats_fallback_count", avoid_goal_fallback_count_base))
        avoid_goal_behind_count_curr = float(getattr(env.env, "_avoid_goal_stats_behind_count", avoid_goal_behind_count_base))
        avoid_goal_side_count_curr = float(getattr(env.env, "_avoid_goal_stats_side_count", avoid_goal_side_count_base))
        avoid_goal_count_curr = float(getattr(env.env, "_avoid_goal_stats_goal_count", avoid_goal_count_base))
        avoid_goal_count_s123_curr = float(getattr(env.env, "_avoid_goal_stats_s123_goal_count", avoid_goal_count_s123_base))
        avoid_goal_behind_count_s123_curr = float(getattr(env.env, "_avoid_goal_stats_s123_behind_count", avoid_goal_behind_count_s123_base))
        avoid_goal_side_count_s123_curr = float(getattr(env.env, "_avoid_goal_stats_s123_side_count", avoid_goal_side_count_s123_base))
        avoid_goal_fallback_count_s123_curr = float(getattr(env.env, "_avoid_goal_stats_s123_fallback_count", avoid_goal_fallback_count_s123_base))
        avoid_goal_count_s4_curr = float(getattr(env.env, "_avoid_goal_stats_s4_goal_count", avoid_goal_count_s4_base))
        avoid_goal_behind_count_s4_curr = float(getattr(env.env, "_avoid_goal_stats_s4_behind_count", avoid_goal_behind_count_s4_base))
        avoid_goal_side_count_s4_curr = float(getattr(env.env, "_avoid_goal_stats_s4_side_count", avoid_goal_side_count_s4_base))
        avoid_goal_fallback_count_s4_curr = float(getattr(env.env, "_avoid_goal_stats_s4_fallback_count", avoid_goal_fallback_count_s4_base))
        avoid_goal_retry_total_iter = max(0.0, avoid_goal_retry_total_curr - avoid_goal_retry_total_base)
        avoid_goal_retry_count_iter = max(0.0, avoid_goal_retry_count_curr - avoid_goal_retry_count_base)
        avoid_goal_fallback_count_iter = max(0.0, avoid_goal_fallback_count_curr - avoid_goal_fallback_count_base)
        avoid_goal_behind_count_iter = max(0.0, avoid_goal_behind_count_curr - avoid_goal_behind_count_base)
        avoid_goal_side_count_iter = max(0.0, avoid_goal_side_count_curr - avoid_goal_side_count_base)
        avoid_goal_count_iter = max(0.0, avoid_goal_count_curr - avoid_goal_count_base)
        avoid_goal_count_s123_iter = max(0.0, avoid_goal_count_s123_curr - avoid_goal_count_s123_base)
        avoid_goal_behind_count_s123_iter = max(0.0, avoid_goal_behind_count_s123_curr - avoid_goal_behind_count_s123_base)
        avoid_goal_side_count_s123_iter = max(0.0, avoid_goal_side_count_s123_curr - avoid_goal_side_count_s123_base)
        avoid_goal_fallback_count_s123_iter = max(0.0, avoid_goal_fallback_count_s123_curr - avoid_goal_fallback_count_s123_base)
        avoid_goal_count_s4_iter = max(0.0, avoid_goal_count_s4_curr - avoid_goal_count_s4_base)
        avoid_goal_behind_count_s4_iter = max(0.0, avoid_goal_behind_count_s4_curr - avoid_goal_behind_count_s4_base)
        avoid_goal_side_count_s4_iter = max(0.0, avoid_goal_side_count_s4_curr - avoid_goal_side_count_s4_base)
        avoid_goal_fallback_count_s4_iter = max(0.0, avoid_goal_fallback_count_s4_curr - avoid_goal_fallback_count_s4_base)
        if avoid_goal_retry_count_iter > 0.0:
            avoid_goal_sample_retry_mean_value = avoid_goal_retry_total_iter / avoid_goal_retry_count_iter
            avoid_goal_sample_fallback_rate_value = avoid_goal_fallback_count_iter / avoid_goal_retry_count_iter
        else:
            avoid_goal_sample_retry_mean_value = 0.0
            avoid_goal_sample_fallback_rate_value = 0.0
        if avoid_goal_count_iter > 0.0:
            avoid_goal_behind_rate_value = avoid_goal_behind_count_iter / avoid_goal_count_iter
            avoid_goal_side_rate_value = avoid_goal_side_count_iter / avoid_goal_count_iter
        else:
            avoid_goal_behind_rate_value = 0.0
            avoid_goal_side_rate_value = 0.0
        if log_memory_this_iter and debug_memory_device_index is not None:
            print(
                f"[MemDebug][iter={iteration}] after_rollout "
                f"{_cuda_mem_stats_text(debug_memory_device_index)}"
            )
        
        # ============ Update Phase ============
        # 计算最后一步的 value (Bootstrap)
        with torch.no_grad():
            state = obs_dict['state']
            goal = obs_dict['goal']
            if getattr(args, "t1_goal_occlusion", False) and last_goal_obs is not None:
                goal = last_goal_obs
            critic_aff_map_next, critic_difficulty_next = _get_critic_gt_inputs(obs_dict)
            
            bootstrap_use_student_aff = args.mode != 'teacher'
            if is_gate and moe_use_student_aff:
                bootstrap_use_student_aff = True
            if bootstrap_use_student_aff:
                aff_map_next, difficulty = _predict_actor_inputs(obs_dict)
            else:
                aff_map_next, difficulty = _get_actor_gt_inputs(obs_dict)
            
            if aff_stack_buf is None:
                aff_stack_bootstrap = aff_map_next.repeat(1, aff_stack, 1, 1)
                critic_aff_stack_bootstrap = critic_aff_map_next.repeat(1, aff_stack, 1, 1)
            else:
                aff_stack_bootstrap = torch.roll(aff_stack_buf, shifts=-aff_map_next.shape[1], dims=1)
                aff_stack_bootstrap[:, -aff_map_next.shape[1]:, :, :] = aff_map_next
                critic_aff_stack_bootstrap = torch.roll(
                    critic_aff_stack_buf,
                    shifts=-critic_aff_map_next.shape[1],
                    dims=1,
                )
                critic_aff_stack_bootstrap[:, -critic_aff_map_next.shape[1]:, :, :] = critic_aff_map_next
                if last_dones is not None and last_dones.any():
                    done_mask = last_dones.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    aff_stack_bootstrap = torch.where(
                        done_mask,
                        aff_map_next.repeat(1, aff_stack, 1, 1),
                        aff_stack_bootstrap,
                    )
                    critic_aff_stack_bootstrap = torch.where(
                        done_mask,
                        critic_aff_map_next.repeat(1, aff_stack, 1, 1),
                        critic_aff_stack_bootstrap,
                    )

            if is_gate:
                gate_difficulty = difficulty if gate_use_difficulty else torch.zeros_like(difficulty)
                gate_critic_difficulty = (
                    critic_difficulty_next if gate_use_difficulty else torch.zeros_like(critic_difficulty_next)
                )
                out = policy(
                    aff_stack_bootstrap,
                    state,
                    goal,
                    gate_difficulty,
                    critic_affordance_map=critic_aff_stack_bootstrap,
                    critic_robot_state=state,
                    critic_goal=goal,
                    critic_terrain_difficulty=gate_critic_difficulty,
                )
            else:
                out = policy(
                    aff_stack_bootstrap,
                    state,
                    goal,
                    difficulty,
                    critic_affordance_map=critic_aff_stack_bootstrap,
                    critic_robot_state=state,
                    critic_goal=goal,
                    critic_terrain_difficulty=critic_difficulty_next,
                )
            next_value = out.value
        
        # 计算 Returns 和 Advantages
        returns, advantages = buffer.compute_returns(next_value, args.gamma, args.gae_lambda)
        
        # Normalize advantages only on true on-policy samples.
        valid_adv_mask = buffer.action_valid.bool()
        if valid_adv_mask.any():
            valid_advantages = advantages[valid_adv_mask]
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = advantages.clone()
            advantages[valid_adv_mask] = (valid_advantages - adv_mean) / adv_std
            advantages[~valid_adv_mask] = 0.0
        else:
            advantages = torch.zeros_like(advantages)
        _assert_finite_tensor("buffer.actions", buffer.actions)
        _assert_finite_tensor("buffer.log_probs", buffer.log_probs)
        _assert_finite_tensor("buffer.states", buffer.states)
        _assert_finite_tensor("buffer.goals", buffer.goals)
        _assert_finite_tensor("buffer.aff_maps", buffer.aff_maps)
        _assert_finite_tensor("buffer.difficulties", buffer.difficulties)
        _assert_finite_tensor("buffer.critic_aff_maps", buffer.critic_aff_maps)
        _assert_finite_tensor("buffer.critic_difficulties", buffer.critic_difficulties)
        _assert_finite_tensor("returns", returns)
        _assert_finite_tensor("advantages", advantages)
        
        # PPO Update
        total_loss = 0
        policy_loss_sum = 0
        effective_policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        effective_entropy_loss_sum = 0
        distill_loss_sum = 0
        bc_loss_sum = 0
        effective_bc_loss_sum = 0
        expert_log_prob_mean_sum = 0
        approx_kl_sum = 0
        clip_frac_sum = 0
        num_updates = 0
        expert_alpha_update = (
            _get_expert_alpha(schedule_iteration, expert_interface_iters)
            if use_egpo else 0.0
        )
        if use_egpo:
            current_lr = args.egpo_lr_post if expert_alpha_update <= 0.0 else args.egpo_lr_guided
        else:
            current_lr = args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        skipped_nonfinite_updates = 0
        sanitized_param_count = 0
        fail_fast_nonfinite = not bool(getattr(args, "allow_nonfinite_recovery", False))
        
        batch_size = env.num_envs * args.num_steps

        def _handle_nonfinite_event(
            reason: str,
            *,
            epoch: int,
            exc: Optional[Exception] = None,
            try_sanitize: bool = False,
        ) -> None:
            nonlocal skipped_nonfinite_updates, sanitized_param_count
            detail = reason if exc is None else f"{reason}: {exc}"
            if fail_fast_nonfinite:
                raise RuntimeError(
                    f"[NonFinite-FailFast] iter={iteration}, epoch={epoch} {detail}. "
                    "Use --allow_nonfinite_recovery to fallback to skip+sanitize mode."
                )
            skipped_nonfinite_updates += 1
            if try_sanitize and _module_has_non_finite(policy):
                _sanitize_module_params(policy)
                sanitized_param_count += 1
            print(f"[Warn] Skip {detail} at iter={iteration}, epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
        
        for epoch in range(args.num_epochs):
            indices = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, args.mini_batch_size):
                end = min(start + args.mini_batch_size, batch_size)
                mb_indices = indices[start:end]
                
                step_idx = torch.div(mb_indices, env.num_envs, rounding_mode='floor')
                env_idx = torch.remainder(mb_indices, env.num_envs)
                
                # 获取 mini-batch 数据
                mb_actions = buffer.actions[step_idx, env_idx]
                mb_old_log_probs = buffer.log_probs[step_idx, env_idx]
                mb_action_valid = buffer.action_valid[step_idx, env_idx].bool()
                mb_advantages = advantages[step_idx, env_idx]
                mb_returns = returns[step_idx, env_idx]
                
                # 读取 rollout 缓冲区
                mb_states = buffer.states[step_idx, env_idx]
                mb_goals = buffer.goals[step_idx, env_idx]
                mb_aff_maps = buffer.aff_maps[step_idx, env_idx]
                mb_difficulties = buffer.difficulties[step_idx, env_idx]
                mb_critic_aff_maps = buffer.critic_aff_maps[step_idx, env_idx]
                mb_critic_difficulties = buffer.critic_difficulties[step_idx, env_idx]
                mb_actions = torch.nan_to_num(mb_actions, nan=0.0, posinf=0.0, neginf=0.0)
                mb_old_log_probs = torch.nan_to_num(mb_old_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
                mb_advantages = torch.nan_to_num(mb_advantages, nan=0.0, posinf=0.0, neginf=0.0)
                mb_returns = torch.nan_to_num(mb_returns, nan=0.0, posinf=0.0, neginf=0.0)
                mb_states = torch.nan_to_num(mb_states, nan=0.0, posinf=0.0, neginf=0.0)
                mb_goals = torch.nan_to_num(mb_goals, nan=0.0, posinf=0.0, neginf=0.0)
                mb_aff_maps = torch.nan_to_num(mb_aff_maps, nan=0.0, posinf=0.0, neginf=0.0)
                mb_difficulties = torch.nan_to_num(mb_difficulties, nan=0.0, posinf=0.0, neginf=0.0)
                mb_critic_aff_maps = torch.nan_to_num(mb_critic_aff_maps, nan=0.0, posinf=0.0, neginf=0.0)
                mb_critic_difficulties = torch.nan_to_num(mb_critic_difficulties, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 评估当前策略
                if (
                    log_memory_this_iter
                    and debug_memory_device_index is not None
                    and epoch == 0
                    and start == 0
                ):
                    print(
                        f"[MemDebug][iter={iteration}] pre_eval "
                        f"{_cuda_mem_stats_text(debug_memory_device_index)}"
                    )
                try:
                    if is_gate:
                        gate_difficulty = mb_difficulties if gate_use_difficulty else torch.zeros_like(mb_difficulties)
                        gate_critic_difficulty = (
                            mb_critic_difficulties if gate_use_difficulty else torch.zeros_like(mb_critic_difficulties)
                        )
                        new_log_probs, new_values, entropy, _ = policy.evaluate_actions(
                            mb_aff_maps,
                            mb_states,
                            mb_goals,
                            gate_difficulty,
                            mb_actions,
                            critic_affordance_map=mb_critic_aff_maps,
                            critic_robot_state=mb_states,
                            critic_goal=mb_goals,
                            critic_terrain_difficulty=gate_critic_difficulty,
                        )
                    else:
                        new_log_probs, new_values, entropy, _ = policy.evaluate_actions(
                            mb_aff_maps,
                            mb_states,
                            mb_goals,
                            mb_difficulties,
                            mb_actions,
                            critic_affordance_map=mb_critic_aff_maps,
                            critic_robot_state=mb_states,
                            critic_goal=mb_goals,
                            critic_terrain_difficulty=mb_critic_difficulties,
                        )
                except ValueError as exc:
                    _handle_nonfinite_event(
                        "policy.evaluate_actions non-finite minibatch",
                        epoch=epoch,
                        exc=exc,
                        try_sanitize=True,
                    )
                    continue
                if (
                    log_memory_this_iter
                    and debug_memory_device_index is not None
                    and epoch == 0
                    and start == 0
                ):
                    print(
                        f"[MemDebug][iter={iteration}] post_eval "
                        f"{_cuda_mem_stats_text(debug_memory_device_index)}"
                    )
                if (
                    (not torch.isfinite(new_log_probs).all())
                    or (not torch.isfinite(new_values).all())
                    or (not torch.isfinite(entropy).all())
                ):
                    _handle_nonfinite_event(
                        "non-finite policy outputs",
                        epoch=epoch,
                        try_sanitize=True,
                    )
                    continue
                
                # PPO Clipped Loss only on true on-policy actions.
                policy_loss = torch.tensor(0.0, device=device)
                if mb_action_valid.any():
                    log_ratio_raw = new_log_probs[mb_action_valid] - mb_old_log_probs[mb_action_valid]
                    if not torch.isfinite(log_ratio_raw).all():
                        _handle_nonfinite_event(
                            "non-finite log-ratio",
                            epoch=epoch,
                            try_sanitize=False,
                        )
                        continue
                    log_ratio = torch.clamp(log_ratio_raw, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    valid_advantages = mb_advantages[mb_action_valid]
                    surr1 = ratio * valid_advantages
                    surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * valid_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Diagnostics
                    approx_kl_sum += (0.5 * (log_ratio ** 2).mean()).item()
                    clip_frac_sum += (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()
                
                # Value loss only on true on-policy samples.
                value_loss = torch.tensor(0.0, device=device)
                if mb_action_valid.any():
                    value_err = new_values.squeeze(-1)[mb_action_valid] - mb_returns[mb_action_valid]
                    value_loss = 0.5 * (value_err ** 2).mean()
                
                # Entropy Loss
                entropy_loss = torch.tensor(0.0, device=device)
                if mb_action_valid.any():
                    entropy_loss = -entropy[mb_action_valid].mean()
                
                # 蒸馏 Loss (Student 模式)
                distill_loss = torch.tensor(0.0, device=device)
                if args.mode == 'student' and teacher_model is not None and not is_gate:
                    mb_teacher_actions = buffer.teacher_actions[step_idx, env_idx]
                    student_out = policy.forward(
                        mb_aff_maps,
                        mb_states,
                        mb_goals,
                        mb_difficulties,
                        critic_affordance_map=mb_critic_aff_maps,
                        critic_robot_state=mb_states,
                        critic_goal=mb_goals,
                        critic_terrain_difficulty=mb_critic_difficulties,
                    )
                    student_cmd_det = torch.tanh(student_out.cmd_mean) * policy.cmd_scale
                    distill_loss = nn.functional.mse_loss(student_cmd_det, mb_teacher_actions)

                bc_loss = torch.tensor(0.0, device=device)
                expert_log_prob_mean = torch.tensor(0.0, device=device)
                mb_expert_actions = None
                if use_expert_guidance and (~mb_action_valid).any():
                    mb_expert_actions = buffer.expert_actions[step_idx, env_idx][~mb_action_valid]
                    mb_expert_actions = torch.nan_to_num(mb_expert_actions, nan=0.0, posinf=0.0, neginf=0.0)
                if use_expert_guidance and mb_expert_actions is not None:
                    try:
                        expert_log_prob, _, _, _ = policy.evaluate_actions(
                            mb_aff_maps[~mb_action_valid],
                            mb_states[~mb_action_valid],
                            mb_goals[~mb_action_valid],
                            mb_difficulties[~mb_action_valid],
                            mb_expert_actions,
                            critic_affordance_map=mb_critic_aff_maps[~mb_action_valid],
                            critic_robot_state=mb_states[~mb_action_valid],
                            critic_goal=mb_goals[~mb_action_valid],
                            critic_terrain_difficulty=mb_critic_difficulties[~mb_action_valid],
                        )
                    except ValueError as exc:
                        _handle_nonfinite_event(
                            "expert-guided evaluate_actions non-finite minibatch",
                            epoch=epoch,
                            exc=exc,
                            try_sanitize=True,
                        )
                        continue
                    expert_log_prob = torch.nan_to_num(expert_log_prob, nan=-20.0, posinf=20.0, neginf=-20.0)
                    expert_log_prob_mean = torch.clamp(expert_log_prob.mean(), min=-20.0, max=0.0)
                    expert_bc_weight = expert_alpha_update if use_egpo else 1.0
                    bc_loss = (-expert_log_prob_mean) * expert_bc_weight * EXPERT_BC_COEF
                
                # Total loss (full EGPO update path, no hold-phase masking).
                policy_loss_weight = 1.0
                value_loss_weight = args.value_loss_coef
                entropy_loss_weight = args.entropy_coef
                bc_loss_weight = 1.0
                effective_policy_loss = policy_loss_weight * policy_loss
                effective_entropy_loss = entropy_loss_weight * entropy_loss
                effective_bc_loss = bc_loss_weight * bc_loss
                loss_terms = (
                    policy_loss_weight * policy_loss
                    + value_loss_weight * value_loss
                    + entropy_loss_weight * entropy_loss
                    + args.distill_coef * distill_loss
                    + effective_bc_loss
                )
                loss = loss_terms
                if not torch.isfinite(loss):
                    _handle_nonfinite_event(
                        "non-finite loss",
                        epoch=epoch,
                        try_sanitize=False,
                    )
                    continue
                
                # 优化
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if (
                    log_memory_this_iter
                    and debug_memory_device_index is not None
                    and epoch == 0
                    and start == 0
                ):
                    print(
                        f"[MemDebug][iter={iteration}] post_backward "
                        f"{_cuda_mem_stats_text(debug_memory_device_index)}"
                    )
                if _actor_grad_has_non_finite(policy):
                    _handle_nonfinite_event(
                        "non-finite actor gradients before optimizer.step",
                        epoch=epoch,
                        try_sanitize=True,
                    )
                    continue
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    _handle_nonfinite_event(
                        "non-finite grad_norm",
                        epoch=epoch,
                        try_sanitize=True,
                    )
                    continue
                optimizer.step()
                if (
                    log_memory_this_iter
                    and debug_memory_device_index is not None
                    and epoch == 0
                    and start == 0
                ):
                    print(
                        f"[MemDebug][iter={iteration}] post_step "
                        f"{_cuda_mem_stats_text(debug_memory_device_index)}"
                    )
                if _module_has_non_finite(policy):
                    _handle_nonfinite_event(
                        "non-finite policy params after optimizer.step",
                        epoch=epoch,
                        try_sanitize=True,
                    )
                    continue
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                effective_policy_loss_sum += effective_policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                effective_entropy_loss_sum += effective_entropy_loss.item()
                distill_loss_sum += distill_loss.item()
                bc_loss_sum += bc_loss.item()
                effective_bc_loss_sum += effective_bc_loss.item()
                expert_log_prob_mean_sum += expert_log_prob_mean.item()
                num_updates += 1

                del (
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_states,
                    mb_goals,
                    mb_aff_maps,
                    mb_difficulties,
                    mb_critic_aff_maps,
                    mb_critic_difficulties,
                    new_log_probs,
                    new_values,
                    entropy,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    distill_loss,
                    bc_loss,
                    effective_policy_loss,
                    effective_entropy_loss,
                    effective_bc_loss,
                    loss,
                )
        
        update_time = time.time() - start_time - rollout_time
        if log_memory_this_iter and debug_memory_device_index is not None:
            print(
                f"[MemDebug][iter={iteration}] after_update "
                f"{_cuda_mem_stats_text(debug_memory_device_index)}"
            )

        # Explained variance (基于 rollout value 估计)
        valid_ev_mask = buffer.action_valid.view(-1).bool()
        if valid_ev_mask.any():
            returns_flat = returns.view(-1)[valid_ev_mask]
            values_flat = buffer.values.view(-1)[valid_ev_mask]
        else:
            returns_flat = returns.view(-1)
            values_flat = buffer.values.view(-1)
        var_returns = returns_flat.var()
        explained_var = 1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)

        # ============ Logging ============
        iter_time = time.time() - start_time
        fps = batch_size / iter_time
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        sum_reward = float(np.sum(episode_rewards)) if episode_rewards else 0.0
        sum_length = float(np.sum(episode_lengths)) if episode_lengths else 0.0
        mean_goal_changes = np.mean(goal_change_counts) if goal_change_counts else 0
        
        total_samples = env.num_envs * args.num_steps
        reward_term_means = {k: (v / total_samples).item() for k, v in reward_term_sums.items()}
        cmd_speed_mean = (cmd_speed_sum / total_samples).item()
        body_forward_speed_mean = (body_forward_speed_sum / total_samples).item()
        body_backward_speed_mean = (body_backward_speed_sum / total_samples).item()
        gate_y_mean = (gate_y_sum / total_samples).item() if is_gate else 0.0
        gate_y_raw_mean = (gate_y_raw_sum / total_samples).item() if is_gate else 0.0
        gate_y_safe_mean = (gate_y_safe_sum / total_samples).item() if is_gate else 0.0
        gate_w_mean = (gate_w_sum / total_samples).item() if is_gate else 0.0
        gate_y_clamp_gap_mean = (gate_y_clamp_gap_sum / total_samples).item() if is_gate else 0.0
        gate_y_w_gap_mean = (gate_y_w_gap_sum / total_samples).item() if is_gate else 0.0
        gate_y_gap_mean = (gate_y_gap_sum / total_samples).item() if is_gate else 0.0
        gate_clearance_f_mean = (gate_clearance_f_sum / total_samples).item() if is_gate else 0.0
        gate_risk_f_mean = (gate_risk_f_sum / total_samples).item() if is_gate else 0.0
        gate_y_change_mean = (gate_y_change_sum / total_samples).item() if is_gate else 0.0
        risk_scale_mean = (risk_scale_sum / total_samples).item()
        goal_dist_mean = (goal_dist_sum / total_samples).item()
        cmd_pred_x_mean = (cmd_pred_x_sum / total_samples).item()
        cmd_pred_x_abs_mean = (cmd_pred_x_abs_sum / total_samples).item()
        cmd_slew_x_mean = (cmd_slew_x_sum / total_samples).item()
        cmd_slew_x_abs_mean = (cmd_slew_x_abs_sum / total_samples).item()
        cmd_exec_x_mean = (cmd_exec_x_sum / total_samples).item()
        cmd_exec_x_abs_mean = (cmd_exec_x_abs_sum / total_samples).item()
        cmd_x_delta_abs_mean = (cmd_x_delta_abs_sum / total_samples).item()
        cmd_x_delta_abs_active_mean = (
            (cmd_x_delta_abs_active_sum / cmd_x_gate_active_count.clamp_min(1.0)).item()
            if float(cmd_x_gate_active_count.item()) > 0.0 else 0.0
        )
        cmd_x_sign_switch_rate = (
            (cmd_x_sign_switch_count / cmd_x_switch_valid_count.clamp_min(1.0)).item()
            if float(cmd_x_switch_valid_count.item()) > 0.0 else 0.0
        )
        cmd_x_sign_switch_active_rate = (
            (cmd_x_sign_switch_active_count / cmd_x_switch_valid_active_count.clamp_min(1.0)).item()
            if float(cmd_x_switch_valid_active_count.item()) > 0.0 else 0.0
        )
        cmd_x_sign_switch_gap_rate = (
            (cmd_x_sign_switch_gap_count / cmd_x_switch_valid_gap_count.clamp_min(1.0)).item()
            if float(cmd_x_switch_valid_gap_count.item()) > 0.0 else 0.0
        )
        cmd_jerk_lin_mean = (cmd_jerk_lin_sum / total_samples).item()
        cmd_jerk_ang_mean = (cmd_jerk_ang_sum / total_samples).item()
        near_miss_excess_mean = (near_miss_excess_sum / total_samples).item()
        near_miss_ratio = (near_miss_step_sum / total_samples).item()
        stuck_ratio = (
            (stuck_step_sum / cmd_active_step_sum.clamp_min(1.0)).item()
            if use_avoid_local_map else 0.0
        )
        gate_switch_rate_mean = (gate_switch_count / total_samples).item() if is_gate else 0.0
        expert_action_diff_mean = (expert_action_diff_sum / total_samples).item() if use_egpo else 0.0
        egpo_heading_align_cos_mean = 0.0
        egpo_heading_err_p95_deg = 0.0
        egpo_yaw_resp_sign_match_rate = 0.0
        if use_egpo and debug and egpo_heading_error_deg_samples:
            egpo_heading_align_cos_mean = (egpo_heading_align_cos_sum / total_samples).item()
            heading_err_all = torch.cat(egpo_heading_error_deg_samples, dim=0)
            egpo_heading_err_p95_deg = torch.quantile(heading_err_all, 0.95).item()
            if float(egpo_yaw_resp_sign_count.item()) > 0.0:
                egpo_yaw_resp_sign_match_rate = float(
                    (egpo_yaw_resp_sign_match_sum / egpo_yaw_resp_sign_count).item()
                )
        target_speed_mean = (target_speed_sum / total_samples).item()
        goal_world_delta_mean = (goal_world_delta_sum / total_samples).item()
        target_turn_event_rate = (target_turn_event_sum / total_samples).item()
        target_preturn_event_rate = (target_preturn_event_sum / total_samples).item()
        target_reflect_event_rate = (target_reflect_event_sum / total_samples).item()
        target_turn_count_mean = (target_turn_count_sum / total_samples).item()
        target_preturn_count_mean = (target_preturn_count_sum / total_samples).item()
        target_reflect_count_mean = (target_reflect_count_sum / total_samples).item()
        target_reset_dist_error_mean = (target_reset_dist_error_sum / total_samples).item()
        target_reset_bearing_error_deg_mean = (
            (target_reset_bearing_error_sum / total_samples).item() * 180.0 / math.pi
        )
        egpo_expert_log_prob_mean = (expert_log_prob_mean_sum / max(num_updates, 1)) if use_egpo else 0.0
        egpo_policy_cmd_mean = torch.zeros(3, device=device)
        egpo_expert_cmd_mean = torch.zeros(3, device=device)
        if use_egpo and float(egpo_cmd_count.item()) > 0.0:
            denom_cmd = egpo_cmd_count.clamp_min(1.0)
            egpo_policy_cmd_mean = egpo_policy_cmd_sum / denom_cmd
            egpo_expert_cmd_mean = egpo_expert_cmd_sum / denom_cmd
        mean_step_reward = (sum_reward / sum_length) if sum_length > 0 else 0.0
        rollout_mean_step_reward = buffer.rewards.mean().item()
        breakdown_total_mean = reward_term_means.get('total', 0.0)
        reward_residual = mean_step_reward - breakdown_total_mean
        resid_rollout = rollout_mean_step_reward - breakdown_total_mean
        goal_dist_display_mean = goal_dist_mean
        goal_dist_tb_name = 'Stats/GoalDist'
        goal_dist_console_label = 'Goal dist / Cmd / Tgt speed:'
        online_selection_metric = (float(mean_reward),)
        online_selection_metric_name = 'online_mean_reward_monitor_only'
        if use_avoid_local_map:
            goal_dist_display_mean = reward_term_means.get('cross_line_dist', goal_dist_mean)
            goal_dist_tb_name = 'Stats/CrossLineTargetDist'
            goal_dist_console_label = 'Cross-line dist / Cmd speed / Tgt speed:'
            online_selection_metric = (
                float(avoid_success_rate_value),
                float(avoid_progress_rate_value),
                float(mean_reward),
            )
            online_selection_metric_name = 'avoid_success_progress_reward_lexicographic'
        episode_collision_rate_value = (
            float(np.mean(episode_collision_flags))
            if (use_avoid_local_map and episode_collision_flags)
            else 0.0
        )
        collision_rate_mean = (collision_rate_sum / total_samples).item()
        obstacle_contact_candidate_rate_mean = (obstacle_contact_candidate_rate_sum / total_samples).item()
        strict_obstacle_contact_rate_mean = (strict_obstacle_contact_rate_sum / total_samples).item()
        obstacle_contact_candidate_min_clearance_mean = (
            obstacle_contact_candidate_min_clearance_sum / total_samples
        ).item()
        avoid_band_active_rate_mean = (avoid_band_active_rate_sum / total_samples).item()
        avoid_band_outside_dist_mean = (avoid_band_outside_dist_sum / total_samples).item()
        collision_force_mean = 0.0
        collision_force_p95 = 0.0
        if collision_force_samples:
            collision_force_all = torch.cat(collision_force_samples, dim=0)
            collision_force_mean = collision_force_all.mean().item()
            collision_force_p95 = torch.quantile(collision_force_all, 0.95).item()
        clearance_mean = (clearance_sum / total_samples).item()
        clearance_min = clearance_min_value if clearance_min_value is not None else 0.0
        passable_occ_ratio_mean = (passable_occ_ratio_sum / total_samples).item()
        crossable_width_mean = (crossable_width_sum / total_samples).item()
        risk_scale_mean = (risk_scale_sum / total_samples).item()
        aff_stack_delta_mean = (aff_stack_delta_sum / args.num_steps).item()
        aff_stack_std_mean = (aff_stack_std_sum / args.num_steps).item()
        aff_stack_filled_mean = (aff_stack_filled_sum / args.num_steps).item()
        terrain_level_mean = None
        terrain_level_max = None
        if hasattr(env.env, "terrain_levels"):
            terrain_level_mean = env.env.terrain_levels.float().mean().item()
            terrain_level_max = env.env.terrain_levels.max().item()

        # TensorBoard
        writer.add_scalar('Loss/Total', total_loss / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Policy', policy_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/PolicyEffective', effective_policy_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Value', value_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Entropy', entropy_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/EntropyEffective', effective_entropy_loss_sum / max(num_updates, 1), iteration)
        if is_gate:
            writer.add_scalar('Stats/GateYClampGap', gate_y_clamp_gap_mean, iteration)
            writer.add_scalar('Stats/GateYWGap', gate_y_w_gap_mean, iteration)
            writer.add_scalar('Stats/GateYTotalGap', gate_y_gap_mean, iteration)
        if args.mode == 'student':
            writer.add_scalar('Loss/Distill', distill_loss_sum / max(num_updates, 1), iteration)
        if use_egpo:
            writer.add_scalar('Train/ExpertAlpha', expert_alpha_update, iteration)
            writer.add_scalar('Train/EGPO_ExpertLogProbMean', expert_log_prob_mean_sum / max(num_updates, 1), iteration)
            writer.add_scalar('Train/EGPO_BCLoss', bc_loss_sum / max(num_updates, 1), iteration)
            writer.add_scalar('Train/EGPO_BCLossEffective', effective_bc_loss_sum / max(num_updates, 1), iteration)
            writer.add_scalar('Train/ExpertActionDiff', expert_action_diff_mean, iteration)
            if debug:
                writer.add_scalar('Diag/EGPOHeadingAlignCosMean', egpo_heading_align_cos_mean, iteration)
                writer.add_scalar('Diag/EGPOHeadingErrP95Deg', egpo_heading_err_p95_deg, iteration)
                writer.add_scalar('Diag/EGPOYawRespSignMatchRate', egpo_yaw_resp_sign_match_rate, iteration)
        
        writer.add_scalar('Perf/MeanReward', mean_reward, iteration)
        writer.add_scalar('Perf/MeanLength', mean_length, iteration)
        writer.add_scalar('Perf/MeanStepReward', mean_step_reward, iteration)
        writer.add_scalar('Perf/RewardResidual', reward_residual, iteration)
        writer.add_scalar('Perf/RolloutMeanStepReward', rollout_mean_step_reward, iteration)
        writer.add_scalar('Perf/RolloutRewardResidual', resid_rollout, iteration)
        writer.add_scalar('Perf/FPS', fps, iteration)
        writer.add_scalar('Perf/LearningRate', current_lr, iteration)
        writer.add_scalar('Perf/GoalChangeCount', mean_goal_changes, iteration)
        writer.add_scalar('Perf/DifficultyMean', buffer.difficulties.mean().item(), iteration)
        writer.add_scalar('Perf/DifficultyMin', buffer.difficulties.min().item(), iteration)
        writer.add_scalar('Perf/DifficultyMax', buffer.difficulties.max().item(), iteration)
        writer.add_scalar('Perf/CriticDifficultyMean', buffer.critic_difficulties.mean().item(), iteration)
        writer.add_scalar('Perf/CriticDifficultyMin', buffer.critic_difficulties.min().item(), iteration)
        writer.add_scalar('Perf/CriticDifficultyMax', buffer.critic_difficulties.max().item(), iteration)
        if terrain_level_mean is not None:
            writer.add_scalar('Perf/TerrainLevelMean', terrain_level_mean, iteration)
        if terrain_level_max is not None:
            writer.add_scalar('Perf/TerrainLevelMax', terrain_level_max, iteration)
        writer.add_scalar('Diag/ApproxKL', approx_kl_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Diag/ClipFrac', clip_frac_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Diag/ExplainedVar', explained_var.item(), iteration)
        writer.add_scalar('Diag/CollisionRate', collision_rate_mean, iteration)
        writer.add_scalar('Diag/CollisionStepRatio', collision_rate_mean, iteration)
        writer.add_scalar('Diag/EpisodeCollisionRate', episode_collision_rate_value, iteration)
        writer.add_scalar('Diag/StageWindowCollisionRate', avoid_collision_rate100_value if use_avoid_local_map else 0.0, iteration)
        writer.add_scalar('Diag/ObstacleContactCandidateRate', obstacle_contact_candidate_rate_mean, iteration)
        writer.add_scalar('Diag/StrictObstacleContactRate', strict_obstacle_contact_rate_mean, iteration)
        writer.add_scalar('Diag/ObstacleContactCandidateMinClearance', obstacle_contact_candidate_min_clearance_mean, iteration)
        writer.add_scalar('Diag/AvoidBandActiveRate', avoid_band_active_rate_mean, iteration)
        writer.add_scalar('Diag/AvoidBandOutsideDist', avoid_band_outside_dist_mean, iteration)
        writer.add_scalar('Diag/CollisionForceMean', collision_force_mean, iteration)
        writer.add_scalar('Diag/CollisionForceP95', collision_force_p95, iteration)
        writer.add_scalar('Diag/SkippedNonFiniteUpdates', skipped_nonfinite_updates, iteration)
        writer.add_scalar('Diag/SanitizedParamCount', sanitized_param_count, iteration)
        writer.add_scalar('Diag/TimeoutBootstrapCount', timeout_bootstrap_count.item(), iteration)
        writer.add_scalar('Diag/NonFiniteInputSamples', float(nonfinite_input_sample_count), iteration)
        writer.add_scalar('Diag/PolicyNonFiniteActions', policy_nonfinite_action_count, iteration)
        writer.add_scalar('Diag/ExpertNonFiniteActions', expert_nonfinite_action_count, iteration)
        writer.add_scalar('Diag/TeacherNonFiniteActions', teacher_nonfinite_action_count, iteration)
        writer.add_scalar('Diag/AffStackDelta', aff_stack_delta_mean, iteration)
        writer.add_scalar('Diag/AffStackStd', aff_stack_std_mean, iteration)
        writer.add_scalar('Diag/AffStackFilled', aff_stack_filled_mean, iteration)
        if collision_threshold_value is not None:
            writer.add_scalar('Diag/CollisionThreshold', collision_threshold_value, iteration)
        writer.add_scalar('Stats/ClearanceMean', clearance_mean, iteration)
        writer.add_scalar('Stats/ClearanceMin', clearance_min, iteration)
        writer.add_scalar('Stats/PassableOccRatio', passable_occ_ratio_mean, iteration)
        writer.add_scalar('Stats/CrossableWidth', crossable_width_mean, iteration)
        writer.add_scalar('Stats/RiskScale', risk_scale_mean, iteration)
        writer.add_scalar('Stats/CmdJerkLin', cmd_jerk_lin_mean, iteration)
        writer.add_scalar('Stats/CmdJerkAng', cmd_jerk_ang_mean, iteration)
        writer.add_scalar('Stats/NearMissExcess', near_miss_excess_mean, iteration)
        writer.add_scalar('Stats/NearMissRatio', near_miss_ratio, iteration)
        writer.add_scalar('Stats/GateEffSwitchRate', gate_switch_rate_mean, iteration)
        writer.add_scalar('Stats/SceneDifficulty', d_scene, iteration)
        if use_avoid_local_map:
            writer.add_scalar('Avoid/Stage', avoid_stage_value, iteration)
            writer.add_scalar('Avoid/CorridorWidth', avoid_corridor_width_value, iteration)
            writer.add_scalar('Avoid/StageWindow', avoid_stage_window_value, iteration)
            writer.add_scalar('Avoid/Stage4ShrinkWindow', avoid_shrink_window_value, iteration)
            writer.add_scalar('Avoid/StageCompletedEpisodes', avoid_stage_completed_eps_value, iteration)
            writer.add_scalar('Avoid/StageCollisionRate', avoid_collision_rate100_value, iteration)
            writer.add_scalar('Avoid/ShrinkCollisionRate', avoid_collision_rate50_value, iteration)
            writer.add_scalar('Avoid/StageExposureRate', avoid_exposure_rate100_value, iteration)
            writer.add_scalar('Avoid/StageProgressRate', avoid_progress_rate_value, iteration)
            writer.add_scalar('Avoid/StageSuccessRate', avoid_success_rate_value, iteration)
            writer.add_scalar('Avoid/StageRowSuccessRate', avoid_row_success_rate_value, iteration)
            writer.add_scalar('Avoid/NearestObstacleDist', avoid_nearest_obstacle_dist_value, iteration)
            writer.add_scalar('Avoid/PresetRetryMean', avoid_preset_retry_mean_value, iteration)
            writer.add_scalar('Avoid/PresetSampleFailMean', avoid_preset_sample_fail_mean_value, iteration)
            writer.add_scalar('Avoid/PresetPassageFailMean', avoid_preset_passage_fail_mean_value, iteration)
            writer.add_scalar('Avoid/PresetMinYGapMean', avoid_preset_min_y_gap_mean_value, iteration)
            writer.add_scalar('Avoid/PresetPassageDepthMean', avoid_preset_passage_depth_mean_value, iteration)
            writer.add_scalar('Avoid/PresetCoreDepthMean', avoid_preset_core_depth_mean_value, iteration)
            writer.add_scalar('Avoid/StageSwitchEvent', avoid_stage_switch_event_value, iteration)
            writer.add_scalar('Avoid/StageSwitchFrom', avoid_stage_switch_from_value, iteration)
            writer.add_scalar('Avoid/StageSwitchTo', avoid_stage_switch_to_value, iteration)
            writer.add_scalar('Avoid/StageSwitchCollisionRate', avoid_stage_switch_collision_value, iteration)
            writer.add_scalar('Avoid/StageSwitchExposureRate', avoid_stage_switch_exposure_value, iteration)
            writer.add_scalar('Avoid/StageSwitchProgressRate', avoid_stage_switch_progress_value, iteration)
            writer.add_scalar('Avoid/StageSwitchSuccessRate', avoid_stage_switch_success_value, iteration)
            writer.add_scalar('Avoid/StageSwitchRowSuccessRate', avoid_stage_switch_row_success_value, iteration)
            writer.add_scalar('Avoid/Stage4ShrinkEvent', avoid_stage4_shrink_event_value, iteration)
            writer.add_scalar('Avoid/Stage4ShrinkFromWidth', avoid_stage4_shrink_from_width_value, iteration)
            writer.add_scalar('Avoid/Stage4ShrinkToWidth', avoid_stage4_shrink_to_width_value, iteration)
            writer.add_scalar('Avoid/GoalSampleRetryMean', avoid_goal_sample_retry_mean_value, iteration)
            writer.add_scalar('Avoid/GoalSampleFallbackRate', avoid_goal_sample_fallback_rate_value, iteration)
            writer.add_scalar('Avoid/GoalBehindRate', avoid_goal_behind_rate_value, iteration)
            writer.add_scalar('Avoid/GoalSideRate', avoid_goal_side_rate_value, iteration)
            if avoid_goal_count_s123_iter > 0.0:
                writer.add_scalar('Avoid/GoalBehindRate_S123', avoid_goal_behind_count_s123_iter / avoid_goal_count_s123_iter, iteration)
                writer.add_scalar('Avoid/GoalSideRate_S123', avoid_goal_side_count_s123_iter / avoid_goal_count_s123_iter, iteration)
                writer.add_scalar('Avoid/GoalFallbackRate_S123', avoid_goal_fallback_count_s123_iter / avoid_goal_count_s123_iter, iteration)
            if avoid_goal_count_s4_iter > 0.0:
                writer.add_scalar('Avoid/GoalBehindRate_S4', avoid_goal_behind_count_s4_iter / avoid_goal_count_s4_iter, iteration)
                writer.add_scalar('Avoid/GoalSideRate_S4', avoid_goal_side_count_s4_iter / avoid_goal_count_s4_iter, iteration)
                writer.add_scalar('Avoid/GoalFallbackRate_S4', avoid_goal_fallback_count_s4_iter / avoid_goal_count_s4_iter, iteration)
            writer.add_scalar('Avoid/StuckRatio', stuck_ratio, iteration)
        if is_gate:
            writer.add_scalar('Stats/GateYEff', gate_y_mean, iteration)
            writer.add_scalar('Stats/GateYRaw', gate_y_raw_mean, iteration)
            writer.add_scalar('Stats/GateYSafe', gate_y_safe_mean, iteration)
            writer.add_scalar('Stats/GateW', gate_w_mean, iteration)
            writer.add_scalar('Stats/GateFollowClearance', gate_clearance_f_mean, iteration)
            writer.add_scalar('Stats/GateFollowRisk', gate_risk_f_mean, iteration)
            writer.add_scalar('Stats/GateYEffChange', gate_y_change_mean, iteration)
        writer.add_scalar('Stats/CmdSpeed', cmd_speed_mean, iteration)
        writer.add_scalar('Stats/CmdPredX', cmd_pred_x_mean, iteration)
        writer.add_scalar('Stats/CmdPredXAbs', cmd_pred_x_abs_mean, iteration)
        writer.add_scalar('Stats/CmdSlewX', cmd_slew_x_mean, iteration)
        writer.add_scalar('Stats/CmdSlewXAbs', cmd_slew_x_abs_mean, iteration)
        writer.add_scalar('Stats/CmdExecX', cmd_exec_x_mean, iteration)
        writer.add_scalar('Stats/CmdExecXAbs', cmd_exec_x_abs_mean, iteration)
        writer.add_scalar('Stats/CmdXDeltaAbs', cmd_x_delta_abs_mean, iteration)
        writer.add_scalar('Stats/CmdXDeltaAbsActive', cmd_x_delta_abs_active_mean, iteration)
        writer.add_scalar('Stats/CmdXExecSignSwitchRate', cmd_x_sign_switch_rate, iteration)
        writer.add_scalar('Stats/CmdXExecSignSwitchRateGated', cmd_x_sign_switch_active_rate, iteration)
        writer.add_scalar('Stats/CmdXPredSignSwitchRateInGap', cmd_x_sign_switch_gap_rate, iteration)
        if 'row_cmdx_signed' in reward_term_means:
            writer.add_scalar('Stats/CmdXSigned', reward_term_means['row_cmdx_signed'], iteration)
        if 'row_cmdx_signed_abs' in reward_term_means:
            writer.add_scalar('Stats/CmdXSignedAbs', reward_term_means['row_cmdx_signed_abs'], iteration)
        if 'row_cmdx_signed_pos' in reward_term_means:
            writer.add_scalar('Stats/CmdXSignedPos', reward_term_means['row_cmdx_signed_pos'], iteration)
        if 'row_cmdx_signed_neg' in reward_term_means:
            writer.add_scalar('Stats/CmdXSignedNeg', reward_term_means['row_cmdx_signed_neg'], iteration)
        writer.add_scalar('Stats/BodyForwardSpeed', body_forward_speed_mean, iteration)
        writer.add_scalar('Stats/BodyBackwardSpeed', body_backward_speed_mean, iteration)
        writer.add_scalar('Stats/TargetSpeed', target_speed_mean, iteration)
        writer.add_scalar(goal_dist_tb_name, goal_dist_display_mean, iteration)
        writer.add_scalar('Stats/CrossLineDist', reward_term_means.get('cross_line_dist', 0.0), iteration)
        writer.add_scalar('Stats/GoalWorldDelta', goal_world_delta_mean, iteration)
        writer.add_scalar('Stats/TargetTurnEventRate', target_turn_event_rate, iteration)
        writer.add_scalar('Stats/TargetPreTurnEventRate', target_preturn_event_rate, iteration)
        writer.add_scalar('Stats/TargetReflectEventRate', target_reflect_event_rate, iteration)
        writer.add_scalar('Stats/TargetTurnCountMean', target_turn_count_mean, iteration)
        writer.add_scalar('Stats/TargetPreTurnCountMean', target_preturn_count_mean, iteration)
        writer.add_scalar('Stats/TargetReflectCountMean', target_reflect_count_mean, iteration)
        writer.add_scalar('Stats/TargetResetDistError', target_reset_dist_error_mean, iteration)
        writer.add_scalar('Stats/TargetResetBearingErrorDeg', target_reset_bearing_error_deg_mean, iteration)
        if 'row_forward_dist' in reward_term_means:
            writer.add_scalar('Stats/RowForwardDist', reward_term_means['row_forward_dist'], iteration)
        if 'row_gate' in reward_term_means:
            writer.add_scalar('Stats/RowGate', reward_term_means['row_gate'], iteration)
        if 'row_gate_active' in reward_term_means:
            writer.add_scalar('Stats/RowGateActiveRatio', reward_term_means['row_gate_active'], iteration)
        if 'row_push_active' in reward_term_means:
            writer.add_scalar('Stats/RowPushActiveRatio', reward_term_means['row_push_active'], iteration)
        reward_log_keys = list(reward_term_means.keys())
        if use_avoid_local_map:
            reward_log_keys = [
                key
                for key in (
                    'approach',
                    'row_lat',
                    'row_gap',
                    'row_push_penalty',
                    'row_cmdx_reward',
                    'avoid_smooth_penalty',
                    'heading_keep',
                    'row_forward_dist',
                    'row_gate',
                    'row_gate_active',
                    'row_push_active',
                    'timeout_bootstrap_bonus',
                    'avoid_band_penalty',
                    'collision',
                    'stability',
                    'terminal_penalty',
                    'time',
                    'total',
                )
                if key in reward_term_means
            ]
        for key in reward_log_keys:
            writer.add_scalar(f'Reward/{key}', reward_term_means[key], iteration)
        
        # Console
        if (
            use_egpo
            and (not egpo_alpha_half_logged)
            and abs(float(expert_alpha_update) - 0.5) < 0.01
            and float(egpo_cmd_count.item()) > 0.0
        ):
            p = egpo_policy_cmd_mean.detach().cpu().tolist()
            e = egpo_expert_cmd_mean.detach().cpu().tolist()
            print(f"[EGPO][alpha~0.5] Policy cmd mean: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]")
            print(f"[EGPO][alpha~0.5] Expert cmd mean: [{e[0]:.4f}, {e[1]:.4f}, {e[2]:.4f}]")
            egpo_alpha_half_logged = True
        extra_console_log_interval = 30
        should_log_console = (
            iteration % args.log_interval == 0
            or iteration % extra_console_log_interval == 0
        )
        if should_log_console:
            width = 80
            pad = 32
            header = f" Learning iteration {iteration}/{total_iterations} "
            collision_threshold_str = "n/a"
            collision_threshold_src_str = "n/a"
            collision_indices_src_str = "n/a"
            if collision_threshold_value is not None:
                collision_threshold_str = f"{collision_threshold_value:.3f}"
            if collision_threshold_src_value is not None:
                collision_threshold_src_str = str(collision_threshold_src_value)
            if collision_indices_src_value is not None:
                collision_indices_src_str = str(collision_indices_src_value)
            terrain_level_str = "n/a"
            if terrain_level_mean is not None:
                terrain_level_str = f"{terrain_level_mean:.2f}"
            gate_line = ""
            if is_gate:
                gate_line = f"""{'Gate raw/eff/w/Δeff:':>{pad}} {gate_y_raw_mean:.3f} / {gate_y_mean:.3f} / {gate_w_mean:.3f} / {gate_y_change_mean:.3f}\n"""
                gate_line += f"""{'Gate gap(clamp/w/total):':>{pad}} {gate_y_clamp_gap_mean:.3f} / {gate_y_w_gap_mean:.3f} / {gate_y_gap_mean:.3f}\n"""
                gate_line += f"""{'Gate clrF/riskF:':>{pad}} {gate_clearance_f_mean:.3f} / {gate_risk_f_mean:.3f}\n"""
            egpo_line = ""
            if use_egpo:
                egpo_line = (
                    f"""{'EGPO alpha/logp/bc(raw/eff)/diff:':>{pad}} {expert_alpha_update:.3f} / """
                    f"""{egpo_expert_log_prob_mean:.4f} / {bc_loss_sum / max(num_updates, 1):.4f} / """
                    f"""{effective_bc_loss_sum / max(num_updates, 1):.4f} / {expert_action_diff_mean:.3f}\n"""
                )
                if debug:
                    egpo_line += (
                        f"""{'EGPO heading cos/p95deg:':>{pad}} """
                        f"""{egpo_heading_align_cos_mean:.3f} / {egpo_heading_err_p95_deg:.1f}\n"""
                    )
                    egpo_line += (
                        f"""{'EGPO yaw response match:':>{pad}} {egpo_yaw_resp_sign_match_rate:.3f}\n"""
                    )
            avoid_line = ""
            if use_avoid_local_map:
                avoid_line = (
                    f"""{'Avoid stage/corridor/window:':>{pad}} {avoid_stage_value:.0f} / {avoid_corridor_width_value:.3f} / {avoid_stage_window_value:.0f}\n"""
                    f"""{'Avoid stage episodes/shrinkWin:':>{pad}} {avoid_stage_completed_eps_value:.0f} / {avoid_shrink_window_value:.0f}\n"""
                    f"""{'Avoid coll(stageWin/ep/shr):':>{pad}} {avoid_collision_rate100_value:.3f} / {episode_collision_rate_value:.3f} / {avoid_collision_rate50_value:.3f}\n"""
                    f"""{'Avoid exposure/progress/success:':>{pad}} {avoid_exposure_rate100_value:.3f} / {avoid_progress_rate_value:.3f} / {avoid_success_rate_value:.3f}\n"""
                    f"""{'Avoid rowSuccessRatio:':>{pad}} {avoid_row_success_rate_value:.3f}\n"""
                    f"""{'Avoid nearest obs dist:':>{pad}} {avoid_nearest_obstacle_dist_value:.3f}\n"""
                    f"""{'Avoid preset retry/sfail/pfail:':>{pad}} {avoid_preset_retry_mean_value:.3f} / {avoid_preset_sample_fail_mean_value:.3f} / {avoid_preset_passage_fail_mean_value:.3f}\n"""
                    f"""{'Avoid preset ygap/depth/core:':>{pad}} {avoid_preset_min_y_gap_mean_value:.3f} / {avoid_preset_passage_depth_mean_value:.3f} / {avoid_preset_core_depth_mean_value:.3f}\n"""
                    f"""{'Avoid goal retry/fallback:':>{pad}} {avoid_goal_sample_retry_mean_value:.3f} / {avoid_goal_sample_fallback_rate_value:.3f}\n"""
                    f"""{'Avoid goal behind/side:':>{pad}} {avoid_goal_behind_rate_value:.3f} / {avoid_goal_side_rate_value:.3f}\n"""
                    f"""{'Avoid goal S123 behind/side/fb:':>{pad}} {((avoid_goal_behind_count_s123_iter / avoid_goal_count_s123_iter) if avoid_goal_count_s123_iter > 0.0 else 0.0):.3f} / {((avoid_goal_side_count_s123_iter / avoid_goal_count_s123_iter) if avoid_goal_count_s123_iter > 0.0 else 0.0):.3f} / {((avoid_goal_fallback_count_s123_iter / avoid_goal_count_s123_iter) if avoid_goal_count_s123_iter > 0.0 else 0.0):.3f}\n"""
                    f"""{'Avoid goal S4 behind/side/fb:':>{pad}} {((avoid_goal_behind_count_s4_iter / avoid_goal_count_s4_iter) if avoid_goal_count_s4_iter > 0.0 else 0.0):.3f} / {((avoid_goal_side_count_s4_iter / avoid_goal_count_s4_iter) if avoid_goal_count_s4_iter > 0.0 else 0.0):.3f} / {((avoid_goal_fallback_count_s4_iter / avoid_goal_count_s4_iter) if avoid_goal_count_s4_iter > 0.0 else 0.0):.3f}\n"""
                    f"""{'Avoid stuck/near-miss:':>{pad}} {stuck_ratio:.3f} / {near_miss_ratio:.3f}\n"""
                    f"""{'Avoid band out/active:':>{pad}} {avoid_band_outside_dist_mean:.3f} / {avoid_band_active_rate_mean:.3f}\n"""
                          f"""{'Avoid obs-hit cand/strict/min:':>{pad}} {obstacle_contact_candidate_rate_mean:.3f} / {strict_obstacle_contact_rate_mean:.3f} / {obstacle_contact_candidate_min_clearance_mean:.3f}\n"""
                )
            log_string = (f"""{'#' * width}\n"""
                          f"""{header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (rollout {rollout_time:.3f}s, update {update_time:.3f}s)\n"""
                          f"""{'Learning rate:':>{pad}} {current_lr:.2e}\n"""
                          f"""{'Value function loss:':>{pad}} {value_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {policy_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Policy loss (effective):':>{pad}} {effective_policy_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {entropy_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Entropy loss (effective):':>{pad}} {effective_entropy_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                          f"""{'Mean step reward:':>{pad}} {mean_step_reward:.3f} (resid {reward_residual:.3f})\n"""
                          f"""{'Rollout step reward:':>{pad}} {rollout_mean_step_reward:.3f} (resid {resid_rollout:.3f})\n"""
                          f"""{'Mean episode length:':>{pad}} {mean_length:.2f}\n"""
                          f"""{'Goal change count:':>{pad}} {mean_goal_changes:.2f}\n"""
                          f"""{'Curriculum level:':>{pad}} {terrain_level_str}\n"""
                          f"""{'Approx KL / Clip frac:':>{pad}} {approx_kl_sum / max(num_updates, 1):.4f} / {clip_frac_sum / max(num_updates, 1):.3f}\n"""
                          f"""{'Nonfinite skip/sanitize:':>{pad}} {skipped_nonfinite_updates} / {sanitized_param_count}\n"""
                          f"""{'Nonfinite act(pol/exp/tch):':>{pad}} {policy_nonfinite_action_count} / {expert_nonfinite_action_count} / {teacher_nonfinite_action_count}\n"""
                          f"""{'Explained variance:':>{pad}} {explained_var.item():.4f}\n"""
                          f"""{goal_dist_console_label:>{pad}} {goal_dist_display_mean:.3f} / {cmd_speed_mean:.3f} / {target_speed_mean:.3f}\n"""
                          f"""{'CmdX pred/post/exec:':>{pad}} {cmd_pred_x_mean:.3f} / {cmd_slew_x_mean:.3f} / {cmd_exec_x_mean:.3f}\n"""
                          f"""{'CmdX |pred|/|post|/|exec|:':>{pad}} {cmd_pred_x_abs_mean:.3f} / {cmd_slew_x_abs_mean:.3f} / {cmd_exec_x_abs_mean:.3f}\n"""
                          f"""{'CmdX deltaAbs(all/gated):':>{pad}} {cmd_x_delta_abs_mean:.3f} / {cmd_x_delta_abs_active_mean:.3f}\n"""
                          f"""{'CmdX switchRate(exec/gated/predInGap):':>{pad}} {cmd_x_sign_switch_rate:.3f} / {cmd_x_sign_switch_active_rate:.3f} / {cmd_x_sign_switch_gap_rate:.3f}\n"""
                          f"""{'CmdX pred/towardGap/exec:':>{pad}} {cmd_pred_x_mean:.3f} / {reward_term_means.get('row_cmdx_signed', 0.0):.3f} / {cmd_exec_x_mean:.3f}\n"""
                          f"""{'CmdX signed abs/pos/neg:':>{pad}} {reward_term_means.get('row_cmdx_signed_abs', 0.0):.3f} / {reward_term_means.get('row_cmdx_signed_pos', 0.0):.3f} / {reward_term_means.get('row_cmdx_signed_neg', 0.0):.3f}\n"""
                          f"""{'Row fwdDist/gate/gOn/pOn:':>{pad}} {reward_term_means.get('row_forward_dist', 0.0):.3f} / {reward_term_means.get('row_gate', 0.0):.3f} / {reward_term_means.get('row_gate_active', 0.0):.3f} / {reward_term_means.get('row_push_active', 0.0):.3f}\n"""
                          f"""{'Cross-line dist(step mean):':>{pad}} {reward_term_means.get('cross_line_dist', 0.0):.3f}\n"""
                          f"""{'Body fwd / back speed:':>{pad}} {body_forward_speed_mean:.3f} / {body_backward_speed_mean:.3f}\n"""
                          f"""{'Goal world delta:':>{pad}} {goal_world_delta_mean:.3f}\n"""
                          f"""{'Target turn/pre/reflect:':>{pad}} {target_turn_event_rate:.3f} / {target_preturn_event_rate:.3f} / {target_reflect_event_rate:.3f}\n"""
                          f"""{'Target reset err(d/mdeg):':>{pad}} {target_reset_dist_error_mean:.3f} / {target_reset_bearing_error_deg_mean:.3f}\n"""
                          f"""{'Clearance mean/min/risk:':>{pad}} {clearance_mean:.3f} / {clearance_min:.3f} / {risk_scale_mean:.3f}\n"""
                          f"""{gate_line}"""
                          f"""{egpo_line}"""
                          f"""{avoid_line}"""
                          f"""{'AffStack d/std/filled:':>{pad}} {aff_stack_delta_mean:.3f} / {aff_stack_std_mean:.3f} / {aff_stack_filled_mean:.3f}\n"""
                          f"""{'Collision step/episode/stage:':>{pad}} {collision_rate_mean:.3f} / {episode_collision_rate_value:.3f} / {(avoid_collision_rate100_value if use_avoid_local_map else 0.0):.3f}\n"""
                          f"""{'Collision force mean/p95/th:':>{pad}} {collision_force_mean:.3f} / {collision_force_p95:.3f} / {collision_threshold_str} ({collision_threshold_src_str}, idx {collision_indices_src_str})\n"""
                          f"""{'-' * width}\n"""
                          f"""{'Reward(main appr/rowLat/rowGap/rowPush/rowCmdX):':>{pad}} {reward_term_means.get('approach', 0.0):.3f} / {reward_term_means.get('row_lat', 0.0):.3f} / {reward_term_means.get('row_gap', 0.0):.3f} / {reward_term_means.get('row_push_penalty', 0.0):.3f} / {reward_term_means.get('row_cmdx_reward', 0.0):.3f}\n"""
                          f"""{'Reward(smooth/earlyNext/head):':>{pad}} {reward_term_means.get('avoid_smooth_penalty', 0.0):.3f} / {reward_term_means.get('early_next_gap_penalty', 0.0):.3f} / {reward_term_means.get('heading_keep', 0.0):.3f}\n"""
                          f"""{'Reward(timeout-only bootstrap):':>{pad}} {reward_term_means.get('timeout_bootstrap_bonus', 0.0):.3f}\n"""
                          f"""{'Reward(cost band/col/stab/term):':>{pad}} {reward_term_means.get('avoid_band_penalty', 0.0):.3f} / {reward_term_means.get('collision', 0.0):.3f} / {reward_term_means.get('stability', 0.0):.3f} / {reward_term_means.get('terminal_penalty', 0.0):.3f}\n"""
                          f"""{'Reward(time/total):':>{pad}} {reward_term_means.get('time', 0.0):.3f} / {reward_term_means.get('total', 0.0):.3f}\n"""
                          f"""{'#' * width}\n""")
            print(log_string)
        
        # Save checkpoint
        fixed_checkpoint_interval = 100
        should_save_checkpoint = (
            iteration > 0 and (
                iteration % args.save_interval == 0
                or iteration % fixed_checkpoint_interval == 0
            )
        )
        if should_save_checkpoint:
            ckpt_path = os.path.join(log_dir, f'model_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_reward': mean_reward,
                'best_reward': best_reward,
                'best_online_reward': best_reward,
                'best_online_success': best_success,
                'best_online_progress': best_progress,
                'best_online_selection_metric': best_selection_metric_name,
                'selection_metric': 'periodic_checkpoint',
                'experiment_meta': run_meta,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, ckpt_path)
            dprint(f"  Saved: {ckpt_path}")
        
        # Save best online-monitor checkpoint.
        current_best_tuple = (best_success, best_progress, best_reward) if use_avoid_local_map else (best_reward,)
        if online_selection_metric > current_best_tuple and len(episode_rewards) >= 10:
            if use_avoid_local_map:
                best_success, best_progress, best_reward = online_selection_metric
            else:
                best_reward = online_selection_metric[0]
            best_selection_metric_name = online_selection_metric_name
            best_path = os.path.join(log_dir, 'best_online_reward.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': policy.state_dict(),
                'mean_reward': mean_reward,
                'best_reward': best_reward,
                'best_online_reward': best_reward,
                'best_online_success': best_success,
                'best_online_progress': best_progress,
                'best_online_selection_metric': best_selection_metric_name,
                'selection_metric': best_selection_metric_name,
                'experiment_meta': run_meta,
            }, best_path)
            if use_avoid_local_map:
                dprint(
                    f"  ★ New online best (succ/prog/reward): "
                    f"{best_success:.3f} / {best_progress:.3f} / {best_reward:.2f}"
                )
            else:
                dprint(f"  ★ New online best: {mean_reward:.2f}")
    
    # 训练结束
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    if use_avoid_local_map:
        print(
            f"Best Online Monitor (succ/prog/reward): "
            f"{best_success:.3f} / {best_progress:.3f} / {best_reward:.2f}"
        )
    else:
        print(f"Best Online Reward: {best_reward:.2f}")
    print(f"Output: {log_dir}")
    print(f"{'='*60}")
    
    writer.close()
    return policy

    

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Command-Space MoE Training")
    
    # 模式
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['teacher', 'student'],
                        help='训练模式: teacher (GT) 或 student (Vision)')
    parser.add_argument('--skill', type=str, default='follow',
                        choices=['follow', 'avoid', 'moe'],
                        help='训练技能: follow / avoid / moe (gate)')
    
    # 环境
    parser.add_argument('--task', type=str, required=True,
                        help='Isaac Gym 任务名称（必填，例如 s_follow_basic / s_avoid_basic / s_cylinder）')
    parser.add_argument('--allow_eval_only_train', action='store_true',
                        help='仅用于临时调试：允许在结构化 OOD hold-out 场景上训练')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（None 使用默认）')
    parser.add_argument('--aff_stack', type=int, default=1,
                        help='affordance 堆叠帧数 (短时记忆)')
    parser.add_argument('--num_envs', type=int, default=None,
                        help='并行环境数量')
    parser.add_argument('--decimation', type=int, default=5,
                        help='高层/低层频率比 (50Hz / 10Hz = 5)')
    parser.add_argument('--camera_interval', type=int, default=None,
                        help='深度相机采样间隔（按 low-level step 计；None 使用任务默认）')
    
    # Checkpoints
    parser.add_argument('--low_level_ckpt', type=str, required=True,
                        help='底层控制器路径')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='(Student) Teacher 模型路径')
    parser.add_argument('--follow_ckpt', type=str, default=None,
                        help='旧参数保留；当前 moe 默认使用解析式 follow expert，不再需要该项')
    parser.add_argument('--avoid_ckpt', type=str, default=None,
                        help='(Gate) Avoid expert 模型路径')
    parser.add_argument('--vision_ckpt', type=str, default=None,
                        help='(Student) Vision 模型路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练 checkpoint 路径（含优化器与迭代）')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='微调 checkpoint 路径（仅加载权重）')
    parser.add_argument('--allow_inexact_resume', action='store_true',
                        help='允许近似续训：接受 checkpoint 未保存 env/curriculum 完整状态')
    
    # 训练超参数
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='训练迭代次数')
    parser.add_argument('--num_steps', type=int, default=24,
                        help='每次迭代的步数')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='PPO epoch 数')
    parser.add_argument('--mini_batch_size', type=int, default=4096,
                        help='Mini-batch 大小')
    parser.add_argument('--lr', type=float, default=1.5e-5,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.05,
                        help='PPO clip range')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss 系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy 系数')
    parser.add_argument('--distill_coef', type=float, default=1.0,
                        help='蒸馏 loss 系数 (Student 模式)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='梯度裁剪')
    parser.add_argument('--cmd_slew_lin', type=float, default=0.2,
                        help='命令线速度变化率限制')
    parser.add_argument('--cmd_slew_ang', type=float, default=0.4,
                        help='命令角速度变化率限制')
    parser.add_argument('--cmd_safe_dist', type=float, default=None,
                        help='安全距离阈值（None 使用默认 clearance）')
    parser.add_argument('--cmd_free_dist', type=float, default=None,
                        help='安全全速距离（None 使用默认 clearance_free）')
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='(V7) 固定风险预算旋钮 beta：0=快/激进，1=安全/保守；None=禁用（保持旧行为）',
    )
    parser.add_argument('--gate_safe_clamp', action='store_true',
                        help='Gate 训练早期启用安全 clamp')
    parser.add_argument('--gate_safe_max', type=float, default=0.3,
                        help='安全 clamp 的最大 y 值')
    parser.add_argument('--w_mode', type=str, default='none', choices=['none', 'geom'],
                        help='PCR 冲突先验模式：none=y-only，geom=基于 follow 方向 clearance 的解析 w')
    parser.add_argument('--w_tau', type=float, default=0.25,
                        help='w_geom 衰减尺度（米）；越小表示更早把近障碍判为高冲突')
    parser.add_argument('--w_blend_mode', type=str, default='multiply', choices=['multiply', 'mix'],
                        help='w 与 gate_y 的融合方式：multiply=y*(1-w)，mix=0.5*y+0.5*(1-w)')
    parser.add_argument('--w_disable_gate_safe_clamp', action='store_true',
                        help='当 w_mode!=none 时关闭旧 gate_safe_clamp，避免双重安全干预')
    parser.add_argument('--moe_expert_deterministic', action='store_true',
                        help='Gate 训练时专家使用确定性输出')
    parser.add_argument('--gate_use_difficulty', action='store_true',
                        help='Gate 使用 difficulty 作为输入（特权信息）')
    parser.add_argument('--moe_use_student_aff', action='store_true',
                        help='moe 模式强制使用 student affordance（与部署一致）')
    parser.add_argument('--disable_risk_scale', action='store_true',
                        help='禁用 CommandPostProcessor 的风险缩放（消融用）')
    parser.add_argument('--t1_goal_occlusion', action='store_true',
                        help='训练中启用 T1 弱遮挡（仅影响 policy 输入）')
    parser.add_argument('--t1_goal_occlusion_prob', type=float, default=0.02,
                        help='T1 弱遮挡触发概率')
    parser.add_argument('--t1_goal_occlusion_len', type=int, default=10,
                        help='T1 弱遮挡持续步数')
    parser.add_argument('--egpo', action='store_true',
                        help='仅在 s_follow_basic + follow 下启用 EGPO（专家动作混合 + BC）')
    parser.add_argument('--egpo_lr_guided', type=float, default=1.5e-5,
                        help='EGPO 专家接管阶段学习率（alpha>0 时生效）')
    parser.add_argument('--egpo_lr_post', type=float, default=1.5e-4,
                        help='EGPO 接口结束后学习率（alpha<=0 时生效）')
    parser.add_argument('--use_follow_expert', action='store_true',
                        help='follow 专家直管输出：高层命令直接由 expert_s0_follow 生成（无需 --teacher_ckpt）')
    parser.add_argument(
        '--allow_nonfinite_recovery',
        action='store_true',
        help='允许旧行为：遇到非有限值时跳过并尝试sanitize后继续（默认关闭，默认fail-fast）',
    )
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='outputs/planner',
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志间隔（运行时自动覆盖为 num_iterations/20）')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='保存间隔')
    parser.add_argument('--debug', action='store_true',
                        help='debug 输出（场景/障碍位置等）')
    parser.add_argument('--debug_memory', action='store_true',
                        help='打印训练阶段 CUDA 显存诊断（rollout/update/首个mini-batch）')
    parser.add_argument('--debug_memory_interval', type=int, default=10,
                        help='显存诊断打印间隔（按 iteration）')
    parser.add_argument(
        '--force_cmd_y',
        action='store_true',
        help='Debug: 强制全程发送 +Y 前进命令 cmd=[0,+v,0] 以验证命令方向（不依赖策略输出）',
    )
    
    args, unknown = parser.parse_known_args()
    args._entropy_coef_explicit = ("--entropy_coef" in sys.argv)
    args.task = normalize_task_name(getattr(args, "task", ""))
    args._unknown_cli = list(unknown)
    
    # 传递未知参数给 Isaac Gym
    sys.argv = [sys.argv[0]] + unknown
    
    train(args)
