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
from typing import Dict, Optional, Tuple

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
from isaacgym import gymapi
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd as s0_follow_expert_fn
from legged_gym.scripts import train_highlevel as th


def _load_experiment_meta_from_ckpt(path: Optional[str], device: torch.device) -> Optional[dict]:
    if path is None or str(path).strip() == "":
        return None
    ckpt_obj = torch.load(path, map_location=device)
    if isinstance(ckpt_obj, dict):
        meta = ckpt_obj.get("experiment_meta", None)
        if isinstance(meta, dict):
            return meta
    return None


def _ckpt_meta_from_obj(ckpt_obj) -> Optional[dict]:
    if isinstance(ckpt_obj, dict):
        meta = ckpt_obj.get("experiment_meta", None)
        if isinstance(meta, dict):
            return meta
    return None


def _argv_has_option(raw_argv, *option_names: str) -> bool:
    options = set(option_names)
    for token in raw_argv:
        if not isinstance(token, str) or not token.startswith("--"):
            continue
        if token.split("=", 1)[0] in options:
            return True
    return False


def _record_play_runtime_override(args, key: str, value) -> None:
    explicit_values = dict(getattr(args, "_cli_explicit_arg_values", {}) or {})
    explicit_values[key] = value
    setattr(args, "_cli_explicit_arg_values", explicit_values)
    runtime_overrides = dict(getattr(args, "_runtime_ablation_cli_overrides", {}) or {})
    if key in th.RUNTIME_ABLATION_ARG_KEYS:
        runtime_overrides[key] = value
    setattr(args, "_runtime_ablation_cli_overrides", runtime_overrides)


def _selected_play_w_alias(args) -> Optional[str]:
    selected = []
    if bool(getattr(args, "yonly", False)):
        selected.append("none")
    if bool(getattr(args, "wgeom", False)):
        selected.append("geom")
    if bool(getattr(args, "wriskonly", False)):
        selected.append("risk_only")
    if bool(getattr(args, "wlearned", False)):
        selected.append("learned")
    if bool(getattr(args, "wlearned2", False)):
        selected.append("learnedw2")
    if len(selected) > 1:
        raise ValueError("请只保留 --yonly / --wgeom / --wriskonly / --wlearned / --wlearned2 之一")
    return selected[0] if selected else None


def _infer_play_w_mode_from_ckpt(path: Optional[str]) -> Tuple[Optional[str], str]:
    if path is None or str(path).strip() == "":
        return None, "no teacher ckpt"
    if not os.path.exists(path):
        return None, f"ckpt not found: {path}"
    ckpt_obj = torch.load(path, map_location="cpu")
    meta = _ckpt_meta_from_obj(ckpt_obj)
    actor_dim = th.infer_checkpoint_gate_action_dim(ckpt_obj, meta)
    meta_mode = None
    if isinstance(meta, dict):
        for key in ("trained_w_mode", "w_mode", "pcr_w_mode"):
            value = meta.get(key, None)
            if value is None:
                continue
            value = str(value).strip().lower()
            if value in ("none", "geom", "risk_only", "learned", "learnedw2"):
                meta_mode = value
                break
            if value in ("yonly", "moe-y"):
                meta_mode = "none"
                break
    if actor_dim == 2:
        if meta_mode in ("learned", "learnedw2"):
            return meta_mode, f"metadata trained_w_mode={meta_mode}, actor_output_dim=2"
        if meta_mode is not None:
            return "learned", f"actor_output_dim=2 overrides metadata {meta_mode}"
        return "learned", "actor_output_dim=2"
    if actor_dim == 1:
        if meta_mode == "geom":
            return "geom", "metadata trained_w_mode=geom"
        if meta_mode == "risk_only":
            return "risk_only", "metadata trained_w_mode=risk_only"
        if meta_mode in ("none", "learned", "learnedw2"):
            return meta_mode, f"metadata trained_w_mode={meta_mode}"
        return "none", "actor_output_dim=1 without geom metadata; default yonly"
    if meta_mode is not None:
        return meta_mode, f"metadata trained_w_mode={meta_mode}"
    return None, "no usable actor_output_dim or w metadata"


def _apply_play_common_defaults(args, raw_argv) -> None:
    pcr_ckpt = getattr(args, "pcr_ckpt", None)
    teacher_ckpt_explicit = _argv_has_option(raw_argv, "--teacher_ckpt")
    if pcr_ckpt:
        if teacher_ckpt_explicit and getattr(args, "teacher_ckpt", None) and os.path.abspath(str(args.teacher_ckpt)) != os.path.abspath(str(pcr_ckpt)):
            raise ValueError("--pcr_ckpt 与历史参数 --teacher_ckpt 指向不同文件；请只保留 --pcr_ckpt")
        args.teacher_ckpt = pcr_ckpt
    elif getattr(args, "teacher_ckpt", None):
        args.pcr_ckpt = args.teacher_ckpt

    if th.is_pcr_line_task_name(str(getattr(args, "task", ""))):
        if not _argv_has_option(raw_argv, "--skill"):
            args.skill = "moe"
        if not _argv_has_option(raw_argv, "--low_level_ckpt"):
            args.low_level_ckpt = th.DEFAULT_LOWLEVEL_CKPT
        if getattr(args, "skill", None) == "moe" and (not _argv_has_option(raw_argv, "--avoid_ckpt")):
            args.avoid_ckpt = th.DEFAULT_AVOID_CKPT

    selected = _selected_play_w_alias(args)
    w_mode_explicit = _argv_has_option(raw_argv, "--w_mode")
    if selected is not None:
        if w_mode_explicit and str(getattr(args, "w_mode", "none")).lower() != selected:
            raise ValueError(
                f"--w_mode={args.w_mode} 与策略别名不一致；请只保留 --yonly / --wgeom / --wriskonly / --wlearned / --wlearned2 之一"
            )
        args.w_mode = selected
        _record_play_runtime_override(args, "w_mode", selected)
        print(f"[PlayHigh] w_mode={selected} from CLI alias")
        return

    if w_mode_explicit:
        return

    inferred, reason = _infer_play_w_mode_from_ckpt(getattr(args, "pcr_ckpt", None) or getattr(args, "teacher_ckpt", None))
    if inferred is None:
        print(f"[PlayHigh] w_mode auto infer skipped: {reason}; using w_mode={args.w_mode}")
        return
    args.w_mode = inferred
    _record_play_runtime_override(args, "w_mode", inferred)
    print(f"[PlayHigh] auto w_mode={inferred} ({reason})")
    if inferred == "none" and "actor_output_dim=1 without geom metadata" in reason:
        print("[PlayHigh] 如果这是 geom-w checkpoint，请显式加 --wgeom；一维 ckpt 不能仅靠网络结构区分 yonly/geom-w。")


def _validate_expected_ckpt_meta(
    ckpt_meta: Optional[dict],
    *,
    source_name: str,
    expected_skill: Optional[str] = None,
    expected_mode: Optional[str] = None,
) -> None:
    if not isinstance(ckpt_meta, dict):
        return
    if expected_skill is not None:
        meta_skill = ckpt_meta.get("skill", None)
        if meta_skill is not None and meta_skill != expected_skill:
            raise ValueError(f"{source_name} 的 skill 与当前回放预期不一致: checkpoint={meta_skill}, expected={expected_skill}")
    if expected_mode is not None:
        meta_mode = ckpt_meta.get("mode", None)
        if meta_mode is not None and meta_mode != expected_mode:
            raise ValueError(f"{source_name} 的 mode 与当前回放预期不一致: checkpoint={meta_mode}, expected={expected_mode}")


def _compute_moe_follow_cmd_from_goal(
    state_tensor: torch.Tensor,
    goal_tensor: torch.Tensor,
    reset_mask: Optional[torch.Tensor],
    cmd_scale,
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


def compute_play_affordance_bundle(args, env, obs, vision_model=None):
    """Build the same affordance inputs used by play_highlevel."""
    if obs is None:
        raise ValueError("obs is None when building affordance map.")
    skill = getattr(args, "skill", "follow")
    if args.mode == "student":
        if vision_model is None:
            raise RuntimeError("vision_model is not initialized in student mode.")
        with torch.no_grad():
            vis_out = vision_model(obs["depth"], normalize=True)
            raw_aff = torch.stack([
                vis_out["occupancy"],
                vis_out["passable_gap"],
                vis_out["low_obstacle"],
            ], dim=1)
            raw_aff = th.resize_affordance_map(raw_aff, env.affordance_map_size)
        avoid_aff = th.build_avoid_local_map_2ch(
            raw_aff,
            visible_mask=getattr(env, "affordance_visible_mask", None),
        )
        follow_difficulty = th.difficulty_from_gap(raw_aff)
        avoid_difficulty = env._compute_objective_difficulty_from_local_map(avoid_aff)
    else:
        raw_aff = obs["gt_affordance"]
        avoid_aff = obs.get(
            "local_map_2ch",
            th.build_avoid_local_map_2ch(
                raw_aff,
                visible_mask=getattr(env, "affordance_visible_mask", None),
            ),
        )
        follow_difficulty = obs.get("gt_difficulty", th.difficulty_from_gap(raw_aff))
        if "actor_difficulty" in obs:
            avoid_difficulty = obs["actor_difficulty"]
        else:
            avoid_difficulty = env._compute_objective_difficulty_from_local_map(avoid_aff)
    if skill == "avoid":
        policy_aff = avoid_aff
        policy_difficulty = avoid_difficulty
    elif skill == "moe":
        policy_aff = avoid_aff
        policy_difficulty = avoid_difficulty
    else:
        policy_aff = raw_aff
        policy_difficulty = follow_difficulty
    return {
        "raw_aff": raw_aff,
        "policy_aff": policy_aff,
        "policy_difficulty": policy_difficulty,
        "follow_aff": raw_aff,
        "follow_difficulty": follow_difficulty,
        "avoid_aff": avoid_aff,
        "avoid_difficulty": avoid_difficulty,
        "gate_aff": avoid_aff,
        "gate_difficulty": avoid_difficulty,
    }


def _ensure_play_runtime_arg_defaults(args) -> None:
    defaults = {
        "pcr_ckpt": getattr(args, "ckpt", None),
        "teacher_ckpt": getattr(args, "pcr_ckpt", getattr(args, "ckpt", None)),
        "camera_show": False,
        "camera_save": False,
        "camera_dir": "outputs/play_highlevel_camera",
        "camera_env": 0,
        "debug_cmd": False,
        "debug_interval": 10,
        "metrics_dir": None,
        "avoid_map_debug_case": "",
        "dump_teacher_every_s": 0.0,
        "avoid_spawn_body_plus_y_deg": None,
        "avoid_direct_single_obstacle": False,
        "e_s_corridor_width": None,
        "e_s_corridor_wall_thickness": None,
        "e_s_corridor_curvature": None,
        "expert_k_yaw": None,
        "expert_heading_lock": None,
        "expert_heading_release": None,
        "disable_success_reset": False,
        "keep_success_reset": False,
        "show_expert_cmd": False,
        "use_expert_cmd": False,
        "use_follow_expert": False,
        "mono_ppo": False,
        "force_cmd": None,
        "heading_offset_override": None,
        "heading_offset_flip": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if getattr(args, "pcr_ckpt", None) is None and hasattr(args, "ckpt"):
        args.pcr_ckpt = args.ckpt
    if getattr(args, "teacher_ckpt", None) is None and getattr(args, "pcr_ckpt", None) is not None:
        args.teacher_ckpt = args.pcr_ckpt


def build_play_runtime_for_eval(args, device: Optional[torch.device] = None):
    """Create env/models with play_highlevel runtime semantics for eval statistics."""
    _ensure_play_runtime_arg_defaults(args)
    if args.task == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用当前 s_/e_ 任务")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_follow_expert = bool(getattr(args, "use_follow_expert", False)) or bool(getattr(args, "use_expert_cmd", False))
    avoid_map_debug_case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower()
    static_avoid_debug = args.task == "s_avoid_basic" and avoid_map_debug_case != ""
    if use_follow_expert and getattr(args, "skill", "follow") != "follow":
        raise ValueError("--use_follow_expert 仅支持 --skill follow")

    primary_contract_ckpt = None
    if not (use_follow_expert or static_avoid_debug or (getattr(args, "force_cmd", None) is not None)):
        primary_contract_ckpt = getattr(args, "teacher_ckpt", None)
    primary_meta = _load_experiment_meta_from_ckpt(primary_contract_ckpt, device)
    th.apply_experiment_meta_to_args(args, primary_meta, context="PlayHigh")
    th.apply_runtime_ablation_cli_overrides(args, primary_meta, context="PlayHigh")

    th.import_modules()
    if args.mode == "student" and not args.vision_ckpt:
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
    if args.mode == "student" and getattr(args, "skill", "follow") == "moe" and not bool(getattr(args, "mono_ppo", False)):
        raise ValueError("当前未实现 Gate 的 student 回放契约，禁止使用 --mode student --skill moe。")
    if getattr(args, "camera_show", False) and getattr(args, "headless", False):
        print("[PlayHigh] ⚠ camera_show requested but headless=True. Disabling.")
        args.camera_show = False
    if getattr(args, "camera_show", False) or getattr(args, "camera_save", False):
        args.camera_enable = True
    if args.mode == "student":
        args.camera_enable = True

    env_cfg, train_cfg = th.task_registry.get_cfgs(name=args.task)
    if getattr(args, "seed", None) is not None:
        env_cfg.seed = int(args.seed)
    th.apply_observation_contract_to_env_cfg(env_cfg, primary_meta, context="PlayHigh")
    _maybe_apply_e_s_corridor_overrides(args, env_cfg)
    _maybe_apply_s_avoid_debug_overrides(args, env_cfg)
    _maybe_apply_pcr_new_play_overrides(args, env_cfg)
    _maybe_apply_pcr_line_play_overrides(args, env_cfg)
    _maybe_apply_e_l_conflict_debug_overrides(args, env_cfg)

    env = th.HierarchicalHexapodEnv(args, device, env_cfg=env_cfg, train_cfg=train_cfg)
    is_pcr_demo_task = bool(getattr(env, "is_pcr_line_task", False))
    if getattr(args, "camera_interval", None) is None:
        args.camera_interval = int(getattr(getattr(env.env, "camera_cfg", None), "capture_interval", 1))
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
    _maybe_apply_s_avoid_stage_override_runtime(args, env)
    if hasattr(env, "env") and hasattr(env.env, "debug_viz"):
        env.env.debug_viz = bool(getattr(args, "debug", False)) or static_avoid_debug or is_pcr_demo_task

    vision_model = None
    resolved_protocol_aux: Dict[str, Dict] = {}
    if args.mode == "student":
        vision_model = th.AffordanceEstimator(
            depth_channels=1,
            output_size=th.get_vision_native_output_size(),
            max_depth_range=5.0
        ).to(device)
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        vision_meta = _ckpt_meta_from_obj(ckpt)
        th.validate_vision_runtime_contract(
            args,
            env,
            source_name="PlayHigh vision checkpoint",
            ckpt_meta=vision_meta,
            strict_meta=True,
        )
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        vision_model.load_state_dict(state_dict)
        vision_model.eval()
        resolved_protocol_aux["vision_ckpt"] = {
            "path": os.path.abspath(args.vision_ckpt),
            "experiment_meta": vision_meta,
        }

    if getattr(args, "camera_env", 0) < 0:
        args.camera_env = 0
    if args.camera_env >= env.num_envs:
        print(f"[PlayHigh] ⚠ camera_env={args.camera_env} out of range; clamping to {env.num_envs - 1}.")
        args.camera_env = env.num_envs - 1

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

    obs = env.reset()
    aff_bundle = compute_play_affordance_bundle(args, env, obs, vision_model)
    aff_map = aff_bundle["policy_aff"]
    aff_shape = aff_map.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    skill = getattr(args, "skill", "follow")
    is_mono_ppo = bool(getattr(args, "mono_ppo", False))
    if is_mono_ppo and skill != "moe":
        raise ValueError("--mono_ppo 只支持 --skill moe。")
    if is_mono_ppo and args.mode != "teacher":
        raise ValueError("--mono_ppo 当前只支持 --mode teacher。")
    is_gate = skill == "moe" and not is_mono_ppo
    env.disable_pcr_gate_aux = bool(is_mono_ppo)
    expert_only_mode = use_follow_expert or static_avoid_debug or (getattr(args, "force_cmd", None) is not None)
    policy = None
    avoid_policy = None
    gate_state_dim = int(obs["state"].shape[1])
    gate_goal_dim = int(obs["goal"].shape[1])
    avoid_state_dim = int(obs["state"].shape[1])
    if not expert_only_mode:
        if not args.teacher_ckpt:
            raise ValueError("非 expert-only 模式必须提供 --pcr_ckpt")
        if is_gate:
            if not args.avoid_ckpt:
                raise ValueError("moe 需要 --avoid_ckpt；follow 侧默认使用解析式 expert")
            gate_ckpt = torch.load(args.teacher_ckpt, map_location=device)
            gate_meta = _ckpt_meta_from_obj(gate_ckpt)
            gate_state_dim = th.infer_checkpoint_state_dim(gate_ckpt) or gate_state_dim
            gate_action_dim = th.infer_checkpoint_gate_action_dim(gate_ckpt, gate_meta)
            expected_gate_action_dim = 2 if th.is_learned_w_mode(getattr(args, "w_mode", "none")) else 1
            if gate_action_dim is not None and int(gate_action_dim) != expected_gate_action_dim:
                raise ValueError(
                    f"gate ckpt actor_output_dim 与当前 play_w_mode 不一致: "
                    f"checkpoint={gate_action_dim}, expected={expected_gate_action_dim}, w_mode={args.w_mode}"
                )
            gate_goal_dim = th.infer_checkpoint_goal_dim(gate_ckpt) or (
                int(obs["goal"].shape[1]) + (th.LEARNED_W_FEATURE_DIM if expected_gate_action_dim == 2 else 0)
            )
            policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=gate_state_dim,
                goal_dim=gate_goal_dim,
                learned_w=expected_gate_action_dim == 2,
            ).to(device)
            avoid_ckpt = torch.load(args.avoid_ckpt, map_location=device)
            avoid_meta = _ckpt_meta_from_obj(avoid_ckpt)
            avoid_state_dim = th.infer_checkpoint_state_dim(avoid_ckpt) or avoid_state_dim
            avoid_aff_channels = int(aff_bundle["avoid_aff"].shape[1] * aff_stack)
            avoid_policy = th.CmdVelExpert(
                affordance_channels=avoid_aff_channels,
                state_dim=avoid_state_dim,
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            _validate_expected_ckpt_meta(gate_meta, source_name="gate ckpt", expected_skill="moe", expected_mode=args.mode)
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(args, env),
                gate_meta,
                reference_name="current play runtime",
                candidate_name="gate ckpt",
                strict=True,
            )
            gate_state = gate_ckpt["model_state_dict"] if isinstance(gate_ckpt, dict) and "model_state_dict" in gate_ckpt else gate_ckpt
            th.load_high_level_state_dict_compat(policy, gate_state, label="play_gate")
            _validate_expected_ckpt_meta(avoid_meta, source_name="avoid ckpt", expected_skill="avoid", expected_mode=args.mode)
            th.validate_checkpoint_contract_compatibility(
                gate_meta,
                avoid_meta,
                reference_name="gate ckpt",
                candidate_name="avoid ckpt",
                strict=True,
            )
            avoid_state = avoid_ckpt["model_state_dict"] if isinstance(avoid_ckpt, dict) and "model_state_dict" in avoid_ckpt else avoid_ckpt
            th.load_high_level_state_dict_compat(avoid_policy, avoid_state, label=f"play_expert:{os.path.basename(args.avoid_ckpt)}")
            policy.eval()
            avoid_policy.eval()
            policy_meta = gate_meta
            resolved_protocol_aux["teacher_ckpt"] = {
                "path": os.path.abspath(args.teacher_ckpt),
                "experiment_meta": gate_meta,
            }
            resolved_protocol_aux["avoid_ckpt"] = {
                "path": os.path.abspath(args.avoid_ckpt),
                "experiment_meta": avoid_meta,
            }
        else:
            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            policy_meta = _ckpt_meta_from_obj(ckpt)
            gate_state_dim = th.infer_checkpoint_state_dim(ckpt) or gate_state_dim
            policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=gate_state_dim,
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            _validate_expected_ckpt_meta(policy_meta, source_name="policy ckpt", expected_skill=skill, expected_mode=args.mode)
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(args, env),
                policy_meta,
                reference_name="current play runtime",
                candidate_name="policy ckpt",
                strict=True,
            )
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            th.load_high_level_state_dict_compat(policy, state_dict, label="play_policy")
            policy.eval()
            resolved_protocol_aux["teacher_ckpt"] = {
                "path": os.path.abspath(args.teacher_ckpt),
                "experiment_meta": policy_meta,
            }
            if is_mono_ppo and args.avoid_ckpt:
                avoid_ckpt = torch.load(args.avoid_ckpt, map_location=device)
                avoid_meta = _ckpt_meta_from_obj(avoid_ckpt)
                avoid_state_dim = th.infer_checkpoint_state_dim(avoid_ckpt) or avoid_state_dim
                avoid_aff_channels = int(aff_bundle["avoid_aff"].shape[1] * aff_stack)
                avoid_policy = th.CmdVelExpert(
                    affordance_channels=avoid_aff_channels,
                    state_dim=avoid_state_dim,
                    goal_dim=obs["goal"].shape[1],
                    cmd_scale=cmd_scale,
                ).to(device)
                _validate_expected_ckpt_meta(avoid_meta, source_name="diagnostic avoid ckpt", expected_skill="avoid", expected_mode=args.mode)
                th.validate_checkpoint_contract_compatibility(
                    th.build_runtime_contract_meta(args, env),
                    avoid_meta,
                    reference_name="current play runtime",
                    candidate_name="diagnostic avoid ckpt",
                    strict=True,
                )
                avoid_state = avoid_ckpt["model_state_dict"] if isinstance(avoid_ckpt, dict) and "model_state_dict" in avoid_ckpt else avoid_ckpt
                th.load_high_level_state_dict_compat(
                    avoid_policy,
                    avoid_state,
                    label=f"play_diagnostic_avoid:{os.path.basename(args.avoid_ckpt)}",
                )
                avoid_policy.eval()
                resolved_protocol_aux["diagnostic_avoid_ckpt"] = {
                    "path": os.path.abspath(args.avoid_ckpt),
                    "experiment_meta": avoid_meta,
                }
    else:
        policy_meta = primary_meta

    return types.SimpleNamespace(
        args=args,
        device=device,
        env=env,
        obs=obs,
        policy=policy,
        avoid_policy=avoid_policy,
        vision_model=vision_model,
        primary_meta=primary_meta,
        policy_meta=policy_meta,
        aux_checkpoint_meta=resolved_protocol_aux,
        gate_state_dim=gate_state_dim,
        gate_goal_dim=gate_goal_dim,
        avoid_state_dim=avoid_state_dim,
        aff_bundle=aff_bundle,
        heading_offset=heading_offset,
        use_follow_expert=use_follow_expert,
        static_avoid_debug=static_avoid_debug,
        expert_only_mode=expert_only_mode,
    )


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


def _world_point_to_map_xy(env, env_idx: int, world_x: float, world_y: float) -> Tuple[float, float]:
    ref_xy, ref_yaw = env._get_affordance_reference_pose()
    ref_xy = ref_xy[env_idx:env_idx + 1]
    ref_yaw = ref_yaw[env_idx:env_idx + 1]
    world_xy = torch.tensor(
        [[float(world_x), float(world_y)]],
        device=ref_xy.device,
        dtype=ref_xy.dtype,
    )
    local_xy = env._world_to_local_xy(world_xy, ref_xy, ref_yaw)[0]
    return float(local_xy[0].item()), float(local_xy[1].item())


def _prepare_avoid_map_dump_dir(args) -> str:
    base = getattr(args, "avoid_map_dump_dir", None)
    if base is None or str(base).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower() or "default"
        base = os.path.join("outputs", "avoid_map_debug", f"{args.task}_{case}_{stamp}")
    os.makedirs(base, exist_ok=True)
    return base


def _prepare_teacher_dump_dir(args) -> str:
    dump_dir = os.path.join(_prepare_avoid_map_dump_dir(args), "teacher_snapshots")
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir


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


def _extract_s_avoid_band_debug(env, env_id: int = 0) -> dict:
    env_impl = getattr(env, "env", None)
    if env_impl is None or not hasattr(env_impl, "root_states"):
        return {}
    if not hasattr(env_impl, "s_avoid_band_x_min") or not hasattr(env_impl, "s_avoid_band_x_max"):
        return {}
    robot_pos = env_impl.root_states[env_id, :3]
    band_x_min = float(env_impl.s_avoid_band_x_min[env_id].item())
    band_x_max = float(env_impl.s_avoid_band_x_max[env_id].item())
    band_y_min = float(env_impl.s_avoid_band_y_min[env_id].item()) if hasattr(env_impl, "s_avoid_band_y_min") else 0.0
    band_y_max = float(env_impl.s_avoid_band_y_max[env_id].item()) if hasattr(env_impl, "s_avoid_band_y_max") else 0.0
    robot_x = float(robot_pos[0].item())
    robot_y = float(robot_pos[1].item())
    robot_z = float(robot_pos[2].item())
    dx_out = max(band_x_min - robot_x, 0.0) + max(robot_x - band_x_max, 0.0)
    return {
        "robot_x": robot_x,
        "robot_y": robot_y,
        "robot_z": robot_z,
        "band_x_min": band_x_min,
        "band_x_max": band_x_max,
        "band_y_min": band_y_min,
        "band_y_max": band_y_max,
        "dx_out": dx_out,
        "inside_band_x": bool((robot_x >= band_x_min) and (robot_x <= band_x_max)),
    }


def _draw_s_avoid_band_debug_lines(env, viewer, env_id: int = 0) -> None:
    env_impl = getattr(env, "env", None)
    if env_impl is None or viewer is None:
        return
    if not hasattr(env_impl, "gym") or not hasattr(env_impl, "envs"):
        return
    band_dbg = _extract_s_avoid_band_debug(env, env_id=env_id)
    if not band_dbg:
        return
    z = float(band_dbg["robot_z"]) + 0.05
    x0 = float(band_dbg["band_x_min"])
    x1 = float(band_dbg["band_x_max"])
    y0 = float(band_dbg["band_y_min"])
    y1 = float(band_dbg["band_y_max"])
    vertices = np.array(
        [
            x0, y0, z, x1, y0, z,
            x1, y0, z, x1, y1, z,
            x1, y1, z, x0, y1, z,
            x0, y1, z, x0, y0, z,
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            0.2, 1.0, 0.2,
            1.0, 0.2, 0.2,
            0.2, 1.0, 0.2,
            1.0, 0.2, 0.2,
        ],
        dtype=np.float32,
    )
    env_impl.gym.add_lines(viewer, env_impl.envs[env_id], 4, vertices, colors)


def _local_xy_to_world_xy(yaw_world: float, x_right: float, y_forward: float) -> np.ndarray:
    return np.array(
        [
            math.cos(yaw_world) * x_right - math.sin(yaw_world) * y_forward,
            math.sin(yaw_world) * x_right + math.cos(yaw_world) * y_forward,
        ],
        dtype=np.float32,
    )


def _bearing_to_world_dir(yaw_world: float, bearing_rad: float) -> np.ndarray:
    local_x = math.sin(bearing_rad)
    local_y = math.cos(bearing_rad)
    return _local_xy_to_world_xy(yaw_world, local_x, local_y)


def _ground_range_from_camera_ray(
    camera_height_m: float,
    down_angle_rad: float,
    near_clip_m: float,
    far_clip_m: float,
) -> float:
    if camera_height_m <= 1e-6:
        return max(0.0, far_clip_m)
    if down_angle_rad <= 1e-4:
        return max(0.0, far_clip_m)
    slant_hit = camera_height_m / max(math.sin(down_angle_rad), 1e-6)
    slant_use = slant_hit
    if near_clip_m > 0.0:
        slant_use = max(slant_use, near_clip_m)
    if far_clip_m > 0.0:
        slant_use = min(slant_use, far_clip_m)
    return max(0.0, slant_use * math.cos(down_angle_rad))


def _extract_local_map_fov_debug(env, env_id: int = 0) -> dict:
    env_impl = getattr(env, "env", None)
    if env_impl is None or not hasattr(env_impl, "root_states"):
        return {}
    if getattr(env, "camera_fov_rad", None) is None or getattr(env, "camera_vertical_fov_rad", None) is None:
        return {}
    if env.camera_fov_rad <= 0.0 or env.camera_vertical_fov_rad <= 0.0:
        return {}
    root = env_impl.root_states[env_id]
    root_xy = root[:2].detach().cpu().numpy().astype(np.float32, copy=False)
    root_z = float(root[2].item())
    quat = root[3:7].detach().cpu().numpy()
    x_q, y_q, z_q, w_q = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    yaw_world = math.atan2(2.0 * (w_q * z_q + x_q * y_q), 1.0 - 2.0 * (y_q * y_q + z_q * z_q))

    base_height_nominal = float(getattr(getattr(env_impl.cfg, "init_state", None), "pos", [0.0, 0.0, root_z])[2])
    cam_height_abs = float(getattr(env, "camera_height_m", max(root_z, 1e-3)))
    cam_height_offset = cam_height_abs - base_height_nominal
    cam_world_z = root_z + cam_height_offset

    cam_local_xy = getattr(env, "affordance_origin_local_xy", None)
    if torch.is_tensor(cam_local_xy) and cam_local_xy.numel() >= 2:
        cam_local_x = float(cam_local_xy[0].item())
        cam_local_y = float(cam_local_xy[1].item())
    else:
        cam_local_x = 0.0
        cam_local_y = 0.0
    cam_world_xy = root_xy + _local_xy_to_world_xy(yaw_world, cam_local_x, cam_local_y)

    half_hfov = 0.5 * float(env.camera_fov_rad)
    half_vfov = 0.5 * float(env.camera_vertical_fov_rad)
    tilt_down = float(getattr(env, "camera_tilt_down_rad", 0.0))
    near_clip = float(getattr(env, "camera_near", 0.0))
    far_clip = float(getattr(env, "camera_far", 0.0))

    near_down = max(1e-4, tilt_down + half_vfov)
    far_down = max(1e-4, tilt_down - half_vfov)
    near_ground = _ground_range_from_camera_ray(cam_world_z, near_down, near_clip, far_clip)
    far_ground = _ground_range_from_camera_ray(cam_world_z, far_down, near_clip, far_clip)
    if far_ground < near_ground:
        near_ground, far_ground = far_ground, near_ground

    bearing_center = float(getattr(env, "camera_bearing_rad", 0.0))
    bearing_left = bearing_center + half_hfov
    bearing_right = bearing_center - half_hfov
    dir_left = _bearing_to_world_dir(yaw_world, bearing_left)
    dir_right = _bearing_to_world_dir(yaw_world, bearing_right)
    dir_center = _bearing_to_world_dir(yaw_world, bearing_center)

    z_ground = 0.05
    cam_world = np.array([cam_world_xy[0], cam_world_xy[1], cam_world_z], dtype=np.float32)

    def _ground_point(direction_xy: np.ndarray, distance_m: float) -> np.ndarray:
        return np.array(
            [
                cam_world_xy[0] + direction_xy[0] * distance_m,
                cam_world_xy[1] + direction_xy[1] * distance_m,
                z_ground,
            ],
            dtype=np.float32,
        )

    left_near = _ground_point(dir_left, near_ground)
    right_near = _ground_point(dir_right, near_ground)
    left_far = _ground_point(dir_left, far_ground)
    right_far = _ground_point(dir_right, far_ground)
    center_far = _ground_point(dir_center, far_ground)
    return {
        "camera_world": cam_world,
        "left_near": left_near,
        "right_near": right_near,
        "left_far": left_far,
        "right_far": right_far,
        "center_far": center_far,
    }


def _draw_local_map_fov_debug_lines(env, viewer, env_id: int = 0) -> None:
    env_impl = getattr(env, "env", None)
    if env_impl is None or viewer is None:
        return
    if not hasattr(env_impl, "gym") or not hasattr(env_impl, "envs"):
        return
    fov_dbg = _extract_local_map_fov_debug(env, env_id=env_id)
    if not fov_dbg:
        return

    camera_world = np.asarray(fov_dbg["camera_world"], dtype=np.float32)
    left_near = np.asarray(fov_dbg["left_near"], dtype=np.float32)
    right_near = np.asarray(fov_dbg["right_near"], dtype=np.float32)
    left_far = np.asarray(fov_dbg["left_far"], dtype=np.float32)
    right_far = np.asarray(fov_dbg["right_far"], dtype=np.float32)
    center_far = np.asarray(fov_dbg["center_far"], dtype=np.float32)

    segments = [
        (left_near, right_near, (0.2, 0.9, 1.0)),
        (left_far, right_far, (1.0, 0.6, 0.1)),
        (left_near, left_far, (1.0, 0.9, 0.2)),
        (right_near, right_far, (1.0, 0.9, 0.2)),
        (camera_world, left_far, (0.5, 0.8, 1.0)),
        (camera_world, right_far, (0.5, 0.8, 1.0)),
        (camera_world, center_far, (1.0, 1.0, 1.0)),
    ]
    vertices = np.asarray([coord for p0, p1, _ in segments for coord in (*p0, *p1)], dtype=np.float32)
    colors = np.asarray([channel for _, _, color in segments for channel in color], dtype=np.float32)
    env_impl.gym.add_lines(viewer, env_impl.envs[env_id], len(segments), vertices, colors)


def _draw_s_avoid_row_gap_debug_lines(
    env,
    viewer,
    *,
    env_id: int = 0,
    cmd_exec_mean: Optional[np.ndarray] = None,
) -> None:
    env_impl = getattr(env, "env", None)
    if env_impl is None or viewer is None:
        return
    if not hasattr(env_impl, "gym") or not hasattr(env_impl, "envs") or not hasattr(env_impl, "root_states"):
        return
    robot_pos = env_impl.root_states[:, :3]
    (
        _gap_center,
        row_y,
        _gap_valid,
        gap_left,
        gap_right,
        gap_left_eff,
        gap_right_eff,
        gap_eff_valid,
    ) = env._compute_nearest_row_gap_target(robot_pos)
    if not bool(gap_eff_valid[env_id].item()):
        return

    robot = env_impl.root_states[env_id]
    robot_x = float(robot[0].item())
    robot_y = float(robot[1].item())
    robot_z = float(robot[2].item()) + 0.06
    raw_l = float(gap_left[env_id].item())
    raw_r = float(gap_right[env_id].item())
    eff_l = float(gap_left_eff[env_id].item())
    eff_r = float(gap_right_eff[env_id].item())
    row_y_w = float(row_y[env_id].item())
    gap_center = 0.5 * (eff_l + eff_r)

    center_half = 0.10
    robot_half = 0.05
    arrow_len = 0.25
    cmd_x = 0.0
    if cmd_exec_mean is not None and len(cmd_exec_mean) >= 1:
        cmd_x = float(cmd_exec_mean[0])
    arrow_end_x = robot_x + cmd_x * arrow_len / max(abs(cmd_x), 1e-3)
    arrow_end_y = robot_y

    segments = [
        ((raw_l, row_y_w, robot_z), (raw_r, row_y_w, robot_z), (0.95, 0.75, 0.10)),
        ((eff_l, row_y_w, robot_z + 0.02), (eff_r, row_y_w, robot_z + 0.02), (0.10, 0.95, 0.25)),
        ((gap_center, row_y_w - center_half, robot_z + 0.04), (gap_center, row_y_w + center_half, robot_z + 0.04), (0.10, 0.90, 1.00)),
        ((robot_x - robot_half, robot_y, robot_z + 0.01), (robot_x + robot_half, robot_y, robot_z + 0.01), (1.00, 1.00, 1.00)),
        ((robot_x, robot_y - robot_half, robot_z + 0.01), (robot_x, robot_y + robot_half, robot_z + 0.01), (1.00, 1.00, 1.00)),
        ((robot_x, robot_y, robot_z + 0.06), (arrow_end_x, arrow_end_y, robot_z + 0.06), (1.00, 0.20, 0.20)),
    ]
    vertices = np.asarray([coord for p0, p1, _ in segments for coord in (*p0, *p1)], dtype=np.float32)
    colors = np.asarray([channel for _, _, color in segments for channel in color], dtype=np.float32)
    env_impl.gym.add_lines(viewer, env_impl.envs[env_id], len(segments), vertices, colors)


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


def _extract_passable_band_from_row(passable_map: np.ndarray, row_y_map_m: float, map_extent: float) -> dict:
    if passable_map.ndim != 2:
        return {}
    size_x, size_y = passable_map.shape
    if size_x <= 0 or size_y <= 0 or not math.isfinite(map_extent) or map_extent <= 0.0:
        return {}
    cell_x = float(map_extent) / float(size_x)
    cell_y = float(map_extent) / float(size_y)
    if not math.isfinite(row_y_map_m):
        return {}
    row_idx = int(math.floor(float(row_y_map_m) / max(cell_y, 1e-6)))
    row_idx = max(0, min(size_y - 1, row_idx))
    line = np.asarray(passable_map[:, row_idx] > 0.5, dtype=np.bool_)
    if line.size == 0 or not bool(line.any()):
        return {
            "row_idx": row_idx,
            "row_y_center_m": (row_idx + 0.5) * cell_y,
            "band_valid": False,
            "band_count": 0,
        }

    runs = []
    start = None
    for idx, active in enumerate(line.tolist()):
        if active and start is None:
            start = idx
        elif (not active) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, line.size - 1))
    if not runs:
        return {
            "row_idx": row_idx,
            "row_y_center_m": (row_idx + 0.5) * cell_y,
            "band_valid": False,
            "band_count": 0,
        }

    best_start, best_end = max(runs, key=lambda seg: (seg[1] - seg[0] + 1, -abs(((seg[0] + seg[1]) * 0.5) - 0.5 * (line.size - 1))))
    left_m = -0.5 * float(map_extent) + best_start * cell_x
    right_m = -0.5 * float(map_extent) + (best_end + 1) * cell_x
    center_m = 0.5 * (left_m + right_m)
    return {
        "row_idx": row_idx,
        "row_y_center_m": (row_idx + 0.5) * cell_y,
        "band_valid": True,
        "band_count": len(runs),
        "band_left_m": float(left_m),
        "band_right_m": float(right_m),
        "band_center_m": float(center_m),
        "band_width_m": float(max(right_m - left_m, 0.0)),
    }


def _summarize_local_map_support(
    raw_aff_map: torch.Tensor,
    local_map_2ch: torch.Tensor,
    visible_mask: Optional[torch.Tensor],
    map_extent: float,
    row_y_map_m: float,
    robot_x_map_m: float,
) -> dict:
    raw_occ = raw_aff_map[0].detach().cpu().numpy().astype(np.float32, copy=False)
    local_np = local_map_2ch.detach().cpu().numpy().astype(np.float32, copy=False)
    summary = _occupancy_center_from_map(raw_occ, map_extent)
    if raw_aff_map.shape[0] >= 2:
        passable_np = raw_aff_map[1].detach().cpu().numpy().astype(np.float32, copy=False)
        summary.update(_extract_passable_band_from_row(passable_np, row_y_map_m, map_extent))
    else:
        summary.update({"band_valid": False, "band_count": 0})
    row_side = 0.0
    row_delta_m = 0.0
    if bool(summary.get("band_valid", False)):
        band_center_m = float(summary.get("band_center_m", 0.0))
        row_delta_m = band_center_m - float(robot_x_map_m)
        if row_delta_m > 1e-4:
            row_side = 1.0
        elif row_delta_m < -1e-4:
            row_side = -1.0

    size_x, size_y = local_np.shape[1], local_np.shape[2]
    cell_x = float(map_extent) / float(size_x) if size_x > 0 else 0.0
    x_coords = (-0.5 * float(map_extent) + (np.arange(size_x, dtype=np.float32) + 0.5) * cell_x)
    occ = local_np[0]
    clearance = local_np[1] if local_np.shape[0] > 1 else (1.0 - occ)
    if visible_mask is not None:
        vis_np = visible_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        occ = occ * vis_np
        clearance = clearance * vis_np
        vis_bool = vis_np > 0.5
    else:
        vis_bool = np.ones_like(occ, dtype=np.bool_)

    left_mask = (x_coords[:, None] < 0.0) & vis_bool
    right_mask = (x_coords[:, None] > 0.0) & vis_bool
    left_occ = occ[left_mask]
    right_occ = occ[right_mask]
    left_clear = clearance[left_mask]
    right_clear = clearance[right_mask]
    left_occ_mean = float(left_occ.mean()) if left_occ.size > 0 else 0.0
    right_occ_mean = float(right_occ.mean()) if right_occ.size > 0 else 0.0
    left_clear_mean = float(left_clear.mean()) if left_clear.size > 0 else 0.0
    right_clear_mean = float(right_clear.mean()) if right_clear.size > 0 else 0.0
    clear_diff = right_clear_mean - left_clear_mean
    if clear_diff > 1e-4:
        better_side = 1.0
    elif clear_diff < -1e-4:
        better_side = -1.0
    else:
        better_side = 0.0
    summary.update(
        {
            "left_occ_mean": left_occ_mean,
            "right_occ_mean": right_occ_mean,
            "left_clear_mean": left_clear_mean,
            "right_clear_mean": right_clear_mean,
            "clear_diff": float(clear_diff),
            "better_side": float(better_side),
            "grid_size_xy": [int(size_x), int(size_y)],
            "cell_size_m": float(cell_x),
            "row_side": float(row_side),
            "row_delta_m": float(row_delta_m),
            "robot_x_map_m": float(robot_x_map_m),
        }
    )
    return summary


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


def _save_s_avoid_teacher_snapshot(
    save_path: str,
    env,
    raw_aff_map: torch.Tensor,
    local_map_2ch: torch.Tensor,
    *,
    row_y_world: float,
    gap_left_world: float,
    gap_right_world: float,
    gap_left_eff_world: float,
    gap_right_eff_world: float,
    gap_center_eff_world: float,
    row_lat_reward: float,
    row_cmdx_reward: float,
    x_err_now: float,
    x_err_prev: float,
    robot_x_world: float,
    x_dir_to_gap: float,
    cmd_x_signed_gap: float,
    cmd_exec: np.ndarray,
    band_dbg: Optional[dict] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as exc:
        print(f"[PlayHigh] ⚠ matplotlib unavailable; skip teacher snapshot export ({exc}).")
        return

    raw_np = raw_aff_map[0].detach().cpu().numpy().astype(np.float32, copy=False)
    local_np = local_map_2ch[0].detach().cpu().numpy().astype(np.float32, copy=False)
    debug_meta = _extract_s_avoid_debug_meta(env)
    visible_np = None
    if getattr(env, "affordance_visible_mask", None) is not None:
        visible_np = env.affordance_visible_mask.detach().float().cpu().numpy().astype(np.float32, copy=False)

    extent_m = float(env.affordance_map_extent)
    extent = [-0.5 * extent_m, 0.5 * extent_m, 0.0, extent_m]
    robot_xy = debug_meta.get("robot_center_xy_in_map", [0.0, 0.0])
    robot_x = float(robot_xy[0]) if robot_xy is not None and len(robot_xy) == 2 else 0.0
    robot_y = float(robot_xy[1]) if robot_xy is not None and len(robot_xy) == 2 else 0.0
    heading_len = 0.55
    cmd_len_scale = 2.5
    cmd_dx = float(cmd_exec[0]) * cmd_len_scale
    cmd_dy = float(cmd_exec[1]) * cmd_len_scale
    yaw = 0.0
    if hasattr(env.env, "root_states"):
        quat = env.env.root_states[0, 3:7].detach().cpu().numpy()
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        root_xy = env.env.root_states[0, :2].detach().cpu().numpy()
        world_robot_x = float(root_xy[0])
        world_robot_y = float(root_xy[1])
    else:
        world_robot_x = 0.0
        world_robot_y = 0.0

    def _world_to_body(px: float, py: float) -> Tuple[float, float]:
        dx = px - world_robot_x
        dy = py - world_robot_y
        bx = math.cos(yaw) * dx + math.sin(yaw) * dy
        by = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return bx, by

    gap_left_body = _world_to_body(gap_left_world, row_y_world)
    gap_right_body = _world_to_body(gap_right_world, row_y_world)
    gap_left_eff_body = _world_to_body(gap_left_eff_world, row_y_world)
    gap_right_eff_body = _world_to_body(gap_right_eff_world, row_y_world)
    gap_center_eff_body = _world_to_body(gap_center_eff_world, row_y_world)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), squeeze=False)
    axes = axes[0]
    panels = [
        ("teacher_occ", raw_np[0].T, "gray"),
        ("teacher_clear", local_np[1].T, "magma"),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        im = ax.imshow(image, origin="lower", extent=extent, cmap=cmap, vmin=0.0, vmax=1.0)
        if visible_np is not None and title == "teacher_occ":
            ax.imshow(
                visible_np.T,
                origin="lower",
                extent=extent,
                cmap="Greys",
                vmin=0.0,
                vmax=1.0,
                alpha=0.15,
            )
        ax.set_title(title)
        ax.set_xlabel("x_right (m)")
        ax.set_ylabel("y_forward (m)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(-0.35, extent[3])
        ax.scatter([robot_x], [robot_y], c="white", s=20, marker="o")
        ax.add_patch(Circle((robot_x, robot_y), float(env.affordance_clearance), fill=False, color="cyan", linestyle="--", linewidth=1.1))
        ax.plot([robot_x, robot_x], [robot_y, robot_y + heading_len], color="dodgerblue", linewidth=2.0)
        ax.plot(
            [gap_left_body[0], gap_right_body[0]],
            [gap_left_body[1], gap_right_body[1]],
            color="lime",
            linewidth=3.0,
        )
        ax.plot(
            [gap_left_eff_body[0], gap_right_eff_body[0]],
            [gap_left_eff_body[1], gap_right_eff_body[1]],
            color="yellow",
            linewidth=2.0,
            linestyle="--",
        )
        ax.plot(
            [gap_center_eff_body[0], gap_center_eff_body[0]],
            [gap_center_eff_body[1] - 0.18, gap_center_eff_body[1] + 0.18],
            color="deepskyblue",
            linewidth=2.0,
            linestyle=":",
        )
        ax.arrow(
            robot_x,
            robot_y,
            cmd_dx,
            cmd_dy,
            width=0.01,
            head_width=0.05,
            head_length=0.07,
            length_includes_head=True,
            color="red",
            alpha=0.9,
        )
        if band_dbg:
            x0 = float(band_dbg["band_x_min"])
            x1 = float(band_dbg["band_x_max"])
            y0 = float(band_dbg.get("band_y_min", 0.0))
            y1 = float(band_dbg.get("band_y_max", extent_m))
            ax.plot([x0, x0], [y0, y1], color="cyan", linewidth=1.0, alpha=0.8)
            ax.plot([x1, x1], [y0, y1], color="cyan", linewidth=1.0, alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    info = (
        f"raw_gap={gap_left_world:.3f}/{gap_right_world:.3f}\n"
        f"eff_gap={gap_left_eff_world:.3f}/{gap_right_eff_world:.3f} row_y={row_y_world:.3f}\n"
        f"gap_center={gap_center_eff_world:.3f} robot_x={robot_x_world:.3f}\n"
        f"x_dir_to_gap={x_dir_to_gap:.1f} cmd_x_signed_gap={cmd_x_signed_gap:.3f}\n"
        f"x_err prev/now={x_err_prev:.3f}/{x_err_now:.3f}\n"
        f"row_lat={row_lat_reward:.3f} row_cmdx={row_cmdx_reward:.3f}\n"
        f"cmd_exec=({float(cmd_exec[0]):.3f},{float(cmd_exec[1]):.3f})\n"
        f"robot_world=({world_robot_x:.3f},{world_robot_y:.3f})"
    )
    axes[1].text(
        1.03,
        0.98,
        info,
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    if bool(getattr(args, "avoid_direct_single_obstacle", False)) and debug_case == "":
        debug_case = "front"
    if debug_case == "":
        return
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if terrain_cfg is None:
        return
    terrain_cfg.avoid_map_debug_case = debug_case
    terrain_cfg.avoid_preview_all_stages = False
    terrain_cfg.avoid_direct_single_obstacle = bool(getattr(args, "avoid_direct_single_obstacle", False))
    target_deg = 0.0 if getattr(args, "avoid_spawn_body_plus_y_deg", None) is None else float(args.avoid_spawn_body_plus_y_deg)
    terrain_cfg.avoid_spawn_body_plus_y_deg = target_deg


def _maybe_apply_pcr_new_play_overrides(args, env_cfg) -> None:
    if env_cfg is None or str(getattr(args, "task", "")) != "s_pcr_new":
        return
    nav_cfg = getattr(env_cfg, "navigation", None)
    if nav_cfg is None:
        return
    # Play/eval should expose the final mixed 2D distribution immediately.
    # Training keeps the configured episode-based schedule.
    nav_cfg.pcr_new_curriculum_progress_override = 1.0
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if bool(getattr(args, "generalize", False)):
        if terrain_cfg is None:
            raise ValueError("s_pcr_new --generalize requires terrain config.")
        nav_cfg.pcr_new_generalize_enable = True
        nav_cfg.pcr_new_generalize_speed_min = 0.55
        nav_cfg.pcr_new_generalize_speed_max = 0.75
        terrain_cfg.pcr_new_force_stage = 4
        spacing = float(getattr(terrain_cfg, "avoid_fixed_row_y_spacing_scale", 1.0))
        terrain_cfg.pcr_new_generalize_row_spacing_ratio = 0.85
        terrain_cfg.avoid_fixed_row_y_spacing_scale = spacing * terrain_cfg.pcr_new_generalize_row_spacing_ratio
    stage_override = getattr(args, "avoid_stage_override", None)
    if stage_override is not None and terrain_cfg is not None:
        terrain_cfg.pcr_new_force_stage = int(stage_override)


def _maybe_apply_pcr_line_play_overrides(args, env_cfg) -> None:
    if env_cfg is None or str(getattr(args, "task", "")) != "s_pcr_line_avoid_basic":
        return
    nav_cfg = getattr(env_cfg, "navigation", None)
    if nav_cfg is None:
        return
    base_speed = float(getattr(nav_cfg, "moving_target_pcr_line_speed", 0.35))
    speed = getattr(args, "pcr_line_target_speed", None)
    scale = getattr(args, "pcr_line_target_speed_scale", None)
    if speed is None and scale is None:
        return
    if speed is None:
        speed = base_speed * float(scale)
    nav_cfg.moving_target_pcr_line_speed = float(speed)
    print(
        f"[PlayHigh] s_pcr_line_avoid_basic target speed override: "
        f"{base_speed:.3f} -> {float(speed):.3f} m/s"
    )


def _maybe_apply_s_avoid_stage_override_runtime(args, env) -> None:
    if str(getattr(args, "task", "")) not in ("s_avoid_basic", "s_pcr_line_avoid_basic", "s_pcr_new"):
        return
    stage_override = getattr(args, "avoid_stage_override", None)
    if stage_override is None:
        return
    if not hasattr(env, "env") or env.env is None:
        return
    if not hasattr(env.env, "s_avoid_stage") or not hasattr(env.env, "s_avoid_stage_per_env"):
        return
    stage_value = int(stage_override)
    if hasattr(env.env, "cfg") and hasattr(env.env.cfg, "terrain"):
        setattr(env.env.cfg.terrain, "pcr_new_force_stage", stage_value)
    env.env.s_avoid_stage = stage_value
    env.env.s_avoid_stage_per_env.fill_(stage_value)
    if hasattr(env.env, "extras") and isinstance(env.env.extras, dict):
        env.env.extras["avoid_stage"] = int(stage_value)
    print(f"[PlayHigh] s_avoid stage override -> {stage_value}")


def _maybe_apply_e_l_conflict_debug_overrides(args, env_cfg) -> None:
    if env_cfg is None or str(getattr(args, "task", "")) != "e_L_conflict":
        return
    if getattr(args, "force_cmd", None) is None:
        return
    terrain_cfg = getattr(env_cfg, "terrain", None)
    if terrain_cfg is None:
        return
    nav_cfg = getattr(env_cfg, "navigation", None)
    if nav_cfg is None:
        return
    terrain_cfg.e_l_conflict_corner_y = 1.2
    terrain_cfg.e_l_conflict_obstacle_y = 1.05
    print("[PlayHigh] e_L_conflict debug override: obstacle moved closer for force_cmd collision check.")


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
        collision_mask = torch.zeros_like(dones, dtype=torch.bool)
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
        default=th.DEFAULT_LOWLEVEL_CKPT,
        help="Low-level policy checkpoint path",
    )
    parser.add_argument("--pcr_ckpt", type=str, default=None, help="PCR main policy checkpoint path")
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="历史兼容参数；PCR 主策略请优先使用 --pcr_ckpt")
    parser.add_argument("--mono_ppo", action="store_true", help="PCR external baseline: direct cmd policy under --skill moe")
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        choices=["follow", "avoid", "moe"],
        help="Expert skill: follow / avoid / moe (gate); PCR 默认自动使用 moe",
    )
    parser.add_argument("--follow_ckpt", type=str, default=None, help="旧参数保留；当前 moe 不再需要，因为 follow 使用解析式 expert")
    parser.add_argument("--avoid_ckpt", type=str, default=None, help="(moe) Avoid expert checkpoint")
    parser.add_argument("--gate_use_difficulty", action="store_true", help="Gate 使用 actor 局部图计算出的 difficulty 作为输入")
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
    parser.add_argument("--camera_interval", type=int, default=None, help="Camera capture interval")
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
    parser.add_argument("--w_mode", type=str, default="none", choices=["none", "geom", "risk_only", "learned", "learnedw2"], help="PCR w 模式")
    w_alias_group = parser.add_mutually_exclusive_group()
    w_alias_group.add_argument("--yonly", action="store_true", help="PCR MoE-y 回放别名，等价于 --w_mode none")
    w_alias_group.add_argument("--wgeom", action="store_true", help="PCR geom-w 回放别名，等价于 --w_mode geom")
    w_alias_group.add_argument("--wriskonly", action="store_true", help="PCR Risk-only 回放别名，等价于 --w_mode risk_only")
    w_alias_group.add_argument("--wlearned", action="store_true", help="PCR learned-w 回放别名，等价于 --w_mode learned")
    w_alias_group.add_argument("--wlearned2", action="store_true", help="PCR learnedw2 回放别名，等价于 --w_mode learnedw2")
    parser.add_argument("--w_tau", type=float, default=0.25, help="w_geom 衰减尺度（米）")
    parser.add_argument("--w_blend_mode", type=str, default="multiply", choices=["multiply", "mix"], help="w 与 gate_y 的融合方式")
    parser.add_argument("--signed_w_lambda", type=float, default=0.30, help="learnedw2 signed-w 修正系数")
    parser.add_argument("--signed_w_gamma_risk", type=float, default=0.15, help="learnedw2 risk_A-risk_F 安全修正系数")
    parser.add_argument("--signed_w_margin", type=float, default=0.05, help="learnedw2 signed-w 小幅抖动死区")
    parser.add_argument("--w2_lambda", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--w2_risk_gamma", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--w_disable_gate_safe_clamp", action="store_true", help="当 w_mode!=none 时关闭旧 gate_safe_clamp")
    parser.add_argument("--risk_memory", action="store_true", help="learned-w 输入旧 row slot 改用可部署短时 risk_F 记忆")
    parser.add_argument("--risk_memory_l_clear", type=float, default=0.40, help="risk memory 距离衰减释放长度，单位 m")
    parser.add_argument("--risk_memory_velocity_source", type=str, default="body", choices=["body", "cmd"], help="risk memory 衰减速度来源")
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
        "--dump_teacher_every_s",
        type=float,
        default=0.0,
        help="s_avoid_basic: 每隔多少秒导出一次当前真实老师通道图（0=关闭）",
    )
    parser.add_argument(
        "--avoid_spawn_body_plus_y_deg",
        type=float,
        default=None,
        help="s_avoid_basic 静态验证：强制设置出生时机体 +y 相对世界 +Y 的夹角（度）",
    )
    parser.add_argument(
        "--avoid_direct_single_obstacle",
        action="store_true",
        help="s_avoid_basic 碰撞 sanity check：默认固定前方单障碍，直接创建在最终位置，不走池子和后续搬运",
    )
    parser.add_argument(
        "--avoid_stage_override",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="s_avoid_basic 回放阶段覆盖：固定查看指定课程阶段",
    )
    parser.add_argument(
        "--generalize",
        action="store_true",
        help="s_pcr_new 高难泛化评测：5 行障碍、目标速度上移、纵向行距压缩",
    )
    parser.add_argument(
        "--pcr_line_target_speed",
        type=float,
        default=None,
        help="覆盖 s_pcr_line_avoid_basic 脚本目标速度，单位 m/s",
    )
    parser.add_argument(
        "--pcr_line_target_speed_scale",
        type=float,
        default=None,
        help="按倍率覆盖 s_pcr_line_avoid_basic 默认脚本目标速度",
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
        "--zero_goal",
        action="store_true",
        default=False,
        help="Debug: policy 输入前将 goal_buf 置零（只做高层对照，不改环境内部 goal）",
    )
    parser.add_argument(
        "--zero_local_map",
        action="store_true",
        default=False,
        help="Debug: policy 输入前将 local_map / affordance 输入置零（只做高层对照）",
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
        help="Follow 专家直管输出：高层命令直接由 expert_s0_follow 生成（无需 --pcr_ckpt）",
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

    if args.camera_interval is not None and args.camera_interval < 1:
        args.camera_interval = 1
    if args.debug_interval < 1:
        args.debug_interval = 1
    if args.debug and "--no_debug_cmd" not in raw_argv:
        args.debug_cmd = True
    if hasattr(th, "normalize_task_name"):
        args.task = th.normalize_task_name(getattr(args, "task", ""))
    if bool(getattr(args, "generalize", False)) and args.task != "s_pcr_new":
        parser.error("--generalize 仅支持 --task s_pcr_new")
    if bool(getattr(args, "generalize", False)) and getattr(args, "avoid_stage_override", None) not in (None, 4):
        parser.error("--generalize 固定 5 行障碍，只允许省略 --avoid_stage_override 或显式传 4")
    if (
        getattr(args, "pcr_line_target_speed", None) is not None
        and getattr(args, "pcr_line_target_speed_scale", None) is not None
    ):
        parser.error("--pcr_line_target_speed 与 --pcr_line_target_speed_scale 只能二选一")
    if getattr(args, "skill", None) is None:
        args.skill = "follow"
    th.capture_cli_explicit_arg_values(args, parser, argv=raw_argv[1:])
    _apply_play_common_defaults(args, raw_argv)
    if bool(getattr(args, "mono_ppo", False)):
        if args.skill != "moe":
            parser.error("--mono_ppo 只支持 --skill moe")
        if args.mode != "teacher":
            parser.error("--mono_ppo 当前只支持 --mode teacher")
        if any((args.yonly, args.wgeom, args.wriskonly, args.wlearned, args.wlearned2)):
            parser.error("--mono_ppo 不允许同时指定 --yonly/--wgeom/--wriskonly/--wlearned/--wlearned2")
        if str(getattr(args, "w_mode", "none")).lower() != "none":
            parser.error("--mono_ppo 不使用 w/y 机制，--w_mode 必须为 none")

    return args


def main():
    args = parse_args()
    if args.task == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用 hex_ground / s_avoid_basic..s_ood_holdout / s_calib")
    supported_tasks = (
        "s_follow_basic",
        "s_avoid_basic",
        "s_pcr_line_avoid_basic",
        "s_pcr_new",
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
            "--task s_follow_basic/s_avoid_basic/s_pcr_line_avoid_basic/s_pcr_new/s_cylinder/"
            "s_narrow_passage/s_step_field/s_dense_obstacles/s_ood_holdout/s_calib "
            "or e_* paper scenes"
        )
    use_follow_expert = bool(getattr(args, "use_follow_expert", False)) or bool(getattr(args, "use_expert_cmd", False))
    avoid_map_debug_case = str(getattr(args, "avoid_map_debug_case", "")).strip().lower()
    static_avoid_debug = args.task == "s_avoid_basic" and avoid_map_debug_case != ""
    if use_follow_expert and getattr(args, "skill", "follow") != "follow":
        raise ValueError("--use_follow_expert 仅支持 --skill follow")
    primary_contract_ckpt = None
    if not (use_follow_expert or static_avoid_debug or (getattr(args, "force_cmd", None) is not None)):
        primary_contract_ckpt = getattr(args, "teacher_ckpt", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_meta = _load_experiment_meta_from_ckpt(primary_contract_ckpt, device)
    th.apply_experiment_meta_to_args(args, primary_meta, context="PlayHigh")
    th.apply_runtime_ablation_cli_overrides(args, primary_meta, context="PlayHigh")
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
    th.import_modules()
    if args.mode == "student" and not args.vision_ckpt:
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
    if args.mode == "student" and getattr(args, "skill", "follow") == "moe" and not bool(getattr(args, "mono_ppo", False)):
        raise ValueError("当前未实现 Gate 的 student 回放契约，禁止使用 --mode student --skill moe。")

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
        th.apply_observation_contract_to_env_cfg(env_cfg, primary_meta, context="PlayHigh")
        _maybe_apply_e_s_corridor_overrides(args, env_cfg)
        _maybe_apply_s_avoid_debug_overrides(args, env_cfg)
        _maybe_apply_pcr_new_play_overrides(args, env_cfg)
        _maybe_apply_pcr_line_play_overrides(args, env_cfg)
        _maybe_apply_e_l_conflict_debug_overrides(args, env_cfg)
    env = th.HierarchicalHexapodEnv(args, device, env_cfg=env_cfg, train_cfg=train_cfg)
    is_pcr_demo_task = bool(getattr(env, "is_pcr_line_task", False))
    if args.camera_interval is None:
        args.camera_interval = int(getattr(getattr(env.env, "camera_cfg", None), "capture_interval", 1))
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
    _maybe_apply_s_avoid_stage_override_runtime(args, env)
    if hasattr(env, "env") and hasattr(env.env, "debug_viz"):
        env.env.debug_viz = bool(getattr(args, "debug", False)) or static_avoid_debug or is_pcr_demo_task
    if is_pcr_demo_task and not args.headless:
        print("[PlayHigh] PCR demo visualization enabled: moving target point + target/robot trajectories.")
    vision_model = None
    resolved_protocol_aux: Dict[str, Dict] = {}
    s0_expert_fn = s0_follow_expert_fn
    if bool(getattr(args, "show_expert_cmd", False)):
        dprint("[PlayHigh] expert cmd debug enabled")
    if args.mode == "student":
        vision_model = th.AffordanceEstimator(
            depth_channels=1,
            output_size=th.get_vision_native_output_size(),
            max_depth_range=5.0
        ).to(device)
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        vision_meta = _ckpt_meta_from_obj(ckpt)
        th.validate_vision_runtime_contract(
            args,
            env,
            source_name="PlayHigh vision checkpoint",
            ckpt_meta=vision_meta,
            strict_meta=True,
        )
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        vision_model.load_state_dict(state_dict)
        vision_model.eval()
        dprint(f"[PlayHigh] ✓ Vision 加载成功: {args.vision_ckpt}")
        resolved_protocol_aux["vision_ckpt"] = {
            "path": os.path.abspath(args.vision_ckpt),
            "experiment_meta": vision_meta,
        }
    if args.camera_env < 0:
        args.camera_env = 0
    if args.camera_env >= env.num_envs:
        print(f"[PlayHigh] ⚠ camera_env={args.camera_env} out of range; clamping to {env.num_envs - 1}.")
        args.camera_env = env.num_envs - 1
    args.metrics_dir = _prepare_metrics_dir(args)
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

    def _get_aff_bundle(current_obs):
        return compute_play_affordance_bundle(args, env, current_obs, vision_model)

    _maybe_apply_s_avoid_stage_override_runtime(args, env)
    obs = env.reset()
    aff_bundle = _get_aff_bundle(obs)
    raw_aff_map = aff_bundle["raw_aff"]
    aff_map = aff_bundle["policy_aff"]
    aff_shape = aff_map.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    is_mono_ppo = bool(getattr(args, "mono_ppo", False))
    if is_mono_ppo and skill != "moe":
        raise ValueError("--mono_ppo 只支持 --skill moe。")
    if is_mono_ppo and args.mode != "teacher":
        raise ValueError("--mono_ppo 当前只支持 --mode teacher。")
    is_gate = skill == "moe" and not is_mono_ppo
    env.disable_pcr_gate_aux = bool(is_mono_ppo)
    expert_only_mode = use_follow_expert or static_avoid_debug or (force_cmd_tensor is not None)
    policy = None
    avoid_policy = None
    avoid_aff_stack_buf = None
    gate_state_dim = int(obs["state"].shape[1])
    gate_goal_dim = int(obs["goal"].shape[1])
    avoid_state_dim = int(obs["state"].shape[1])
    if use_follow_expert:
        print("[PlayHigh] cmd_source=follow_expert (--use_follow_expert)")
    elif static_avoid_debug:
        print(f"[PlayHigh] cmd_source=zero_cmd (--avoid_map_debug_case={avoid_map_debug_case})")
        print("[PlayHigh] static avoid-map debug enabled; robot command is clamped to zero.")
    elif force_cmd_tensor is not None:
        print("[PlayHigh] cmd_source=force_cmd (--force_cmd)")
    elif is_mono_ppo:
        print("[PlayHigh] cmd_source=mono_ppo_direct_cmd; cmd=[x_right,y_forward,yaw]")
    if not expert_only_mode:
        if not args.teacher_ckpt:
            raise ValueError("非 expert-only 模式必须提供 --pcr_ckpt")
        if is_gate:
            if not args.avoid_ckpt:
                raise ValueError("moe 需要 --avoid_ckpt；follow 侧默认使用解析式 expert")
            gate_ckpt = torch.load(args.teacher_ckpt, map_location=device)
            gate_meta = _ckpt_meta_from_obj(gate_ckpt)
            gate_state_dim = th.infer_checkpoint_state_dim(gate_ckpt) or gate_state_dim
            gate_action_dim = th.infer_checkpoint_gate_action_dim(gate_ckpt, gate_meta)
            expected_gate_action_dim = 2 if th.is_learned_w_mode(getattr(args, "w_mode", "none")) else 1
            if gate_action_dim is not None and int(gate_action_dim) != expected_gate_action_dim:
                raise ValueError(
                    f"gate ckpt actor_output_dim 与当前 play_w_mode 不一致: "
                    f"checkpoint={gate_action_dim}, expected={expected_gate_action_dim}, w_mode={args.w_mode}"
                )
            gate_goal_dim = th.infer_checkpoint_goal_dim(gate_ckpt) or (
                int(obs["goal"].shape[1]) + (th.LEARNED_W_FEATURE_DIM if expected_gate_action_dim == 2 else 0)
            )
            policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=gate_state_dim,
                goal_dim=gate_goal_dim,
                learned_w=expected_gate_action_dim == 2,
            ).to(device)
            ckpt = torch.load(args.avoid_ckpt, map_location=device)
            ckpt_meta = _ckpt_meta_from_obj(ckpt)
            avoid_state_dim = th.infer_checkpoint_state_dim(ckpt) or avoid_state_dim
            avoid_aff_channels = int(aff_bundle["avoid_aff"].shape[1] * aff_stack)
            avoid_policy = th.CmdVelExpert(
                affordance_channels=avoid_aff_channels,
                state_dim=avoid_state_dim,
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            _validate_expected_ckpt_meta(
                gate_meta,
                source_name="gate ckpt",
                expected_skill="moe",
                expected_mode=args.mode,
            )
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(args, env),
                gate_meta,
                reference_name="current play runtime",
                candidate_name="gate ckpt",
                strict=True,
            )
            gate_state = gate_ckpt["model_state_dict"] if isinstance(gate_ckpt, dict) and "model_state_dict" in gate_ckpt else gate_ckpt
            th.load_high_level_state_dict_compat(policy, gate_state, label="play_gate")
            resolved_protocol_aux["teacher_ckpt"] = {
                "path": os.path.abspath(args.teacher_ckpt),
                "experiment_meta": gate_meta,
            }
            _validate_expected_ckpt_meta(
                ckpt_meta,
                source_name="avoid ckpt",
                expected_skill="avoid",
                expected_mode=args.mode,
            )
            th.validate_checkpoint_contract_compatibility(
                gate_meta,
                ckpt_meta,
                reference_name="gate ckpt",
                candidate_name="avoid ckpt",
                strict=True,
            )
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            th.load_high_level_state_dict_compat(
                avoid_policy,
                state_dict,
                label=f"play_expert:{os.path.basename(args.avoid_ckpt)}",
            )
            avoid_policy.eval()
            resolved_protocol_aux["avoid_ckpt"] = {
                "path": os.path.abspath(args.avoid_ckpt),
                "experiment_meta": ckpt_meta,
            }
            policy.eval()
        else:
            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            ckpt_meta = _ckpt_meta_from_obj(ckpt)
            policy_state_dim = th.infer_checkpoint_state_dim(ckpt) or int(obs["state"].shape[1])
            policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=policy_state_dim,
                goal_dim=obs["goal"].shape[1],
                cmd_scale=cmd_scale,
            ).to(device)
            _validate_expected_ckpt_meta(
                ckpt_meta,
                source_name="policy ckpt",
                expected_skill=skill,
                expected_mode=args.mode,
            )
            th.validate_checkpoint_contract_compatibility(
                th.build_runtime_contract_meta(args, env),
                ckpt_meta,
                reference_name="current play runtime",
                candidate_name="policy ckpt",
                strict=True,
            )
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            th.load_high_level_state_dict_compat(policy, state_dict, label="play_policy")
            policy.eval()
            resolved_protocol_aux["teacher_ckpt"] = {
                "path": os.path.abspath(args.teacher_ckpt),
                "experiment_meta": ckpt_meta,
            }
    else:
        dprint("[PlayHigh] expert-only takeover enabled; skip policy checkpoint loading.")
    th.write_resolved_protocol_json(
        os.path.join(args.metrics_dir, "resolved_protocol.json"),
        th.build_resolved_protocol(
            args,
            env,
            primary_ckpt_path=primary_contract_ckpt if primary_contract_ckpt else getattr(args, "teacher_ckpt", None),
            primary_meta=primary_meta,
            aux_sources=resolved_protocol_aux,
        ),
    )
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
    track_ep_cmd_pred_x = []
    track_ep_cmd_exec_x = []
    teacher_dump_interval_s = max(float(getattr(args, "dump_teacher_every_s", 0.0)), 0.0)
    high_level_dt = float(getattr(env, "high_level_dt", float(env.env.dt) * float(args.decimation)))
    teacher_dump_interval_steps = 0
    next_teacher_dump_step = 0
    if teacher_dump_interval_s > 0.0:
        teacher_dump_interval_steps = max(1, int(round(teacher_dump_interval_s / max(high_level_dt, 1e-6))))

    prev_dist = None
    aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
    if is_gate:
        avoid_aff_stack_buf = aff_bundle["avoid_aff"].repeat(1, aff_stack, 1, 1)
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
                _maybe_apply_s_avoid_stage_override_runtime(args, env)
                obs = env.reset()
                aff_bundle = _get_aff_bundle(obs)
                raw_aff_map = aff_bundle["raw_aff"]
                aff_map = aff_bundle["policy_aff"]
                aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
                if is_gate:
                    avoid_aff_stack_buf = aff_bundle["avoid_aff"].repeat(1, aff_stack, 1, 1)
                aff_stack_fill.fill_(1)
                stack_reset_mask = None
                prev_dist = None
                continue
            obs_before_step = None
            if e_s_metrics.get("enabled", False) and (obs is not None) and ("goal" in obs):
                obs_before_step = {"goal": obs["goal"].detach().clone()}
            reset_mask = stack_reset_mask
            if stack_reset_mask is not None and stack_reset_mask.any():
                reset_bundle = _get_aff_bundle(obs)
                reset_aff = reset_bundle["policy_aff"]
                aff_stack_buf[stack_reset_mask] = reset_aff[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                if is_gate:
                    avoid_aff_stack_buf[stack_reset_mask] = reset_bundle["avoid_aff"][stack_reset_mask].repeat(1, aff_stack, 1, 1)
                aff_stack_fill[stack_reset_mask] = 1
                stack_reset_mask = None
            aff_bundle = _get_aff_bundle(obs)
            raw_aff_map = aff_bundle["raw_aff"]
            aff_map = aff_bundle["policy_aff"]
            aff_stack_buf = torch.roll(aff_stack_buf, shifts=-aff_map.shape[1], dims=1)
            aff_stack_buf[:, -aff_map.shape[1]:, :, :] = aff_map
            if is_gate:
                avoid_aff = aff_bundle["avoid_aff"]
                avoid_aff_stack_buf = torch.roll(avoid_aff_stack_buf, shifts=-avoid_aff.shape[1], dims=1)
                avoid_aff_stack_buf[:, -avoid_aff.shape[1]:, :, :] = avoid_aff
            if aff_stack > 1:
                if reset_mask is None:
                    aff_stack_fill = torch.clamp(aff_stack_fill + 1, max=aff_stack)
                else:
                    inc_mask = ~reset_mask
                    if inc_mask.any():
                        aff_stack_fill[inc_mask] = torch.clamp(aff_stack_fill[inc_mask] + 1, max=aff_stack)
            else:
                aff_stack_fill.fill_(1)
            difficulty = aff_bundle["policy_difficulty"]
            difficulty_input = (
                torch.zeros_like(difficulty)
                if bool(getattr(args, "zero_local_map", False))
                else difficulty
            )
            gate_y = None
            gate_diag = None
            avoid_cf_cmds = None
            avoid_cf_feats = None
            cmd = torch.zeros((env.num_envs, 3), device=device, dtype=torch.float32)
            policy_goal = th.get_policy_goal_tensor(obs, skill)
            goal_input = torch.zeros_like(policy_goal) if bool(getattr(args, "zero_goal", False)) else policy_goal
            avoid_goal_input = torch.zeros_like(obs["goal"]) if bool(getattr(args, "zero_goal", False)) else obs["goal"]
            aff_input = torch.zeros_like(aff_stack_buf) if bool(getattr(args, "zero_local_map", False)) else aff_stack_buf
            avoid_aff_input = (
                torch.zeros_like(avoid_aff_stack_buf)
                if is_gate and bool(getattr(args, "zero_local_map", False))
                else avoid_aff_stack_buf
            )
            gate_aff_input = (
                torch.zeros_like(aff_bundle["gate_aff"])
                if is_gate and bool(getattr(args, "zero_local_map", False))
                else aff_bundle["gate_aff"]
            )
            if not expert_only_mode:
                with torch.no_grad():
                    if is_gate:
                        gate_difficulty = aff_bundle["gate_difficulty"] if args.gate_use_difficulty else torch.zeros_like(aff_bundle["gate_difficulty"])
                        if bool(getattr(args, "zero_local_map", False)):
                            gate_difficulty = torch.zeros_like(gate_difficulty)
                        avoid_difficulty_input = (
                            torch.zeros_like(aff_bundle["avoid_difficulty"])
                            if bool(getattr(args, "zero_local_map", False))
                            else aff_bundle["avoid_difficulty"]
                        )
                        expert_state = th.get_moe_expert_state_inputs(
                            th.match_state_dim(obs["state"], avoid_state_dim, label="play_expert_state")
                        )
                        gate_state = th.match_state_dim(obs["state"], gate_state_dim, label="play_gate_state")
                        cmd_f = _compute_moe_follow_cmd_from_goal(
                            expert_state,
                            goal_input,
                            reset_mask,
                            cmd_scale,
                            env_ref=env,
                        )
                        cmd_a, _ = avoid_policy.get_action(
                            avoid_aff_input,
                            expert_state,
                            avoid_goal_input,
                            avoid_difficulty_input,
                            deterministic=True,
                        )
                        gate_policy_goal = goal_input
                        if th.is_learned_w_mode(getattr(args, "w_mode", "none")):
                            gate_policy_goal, _ = th.build_learned_w_gate_goal(
                                env,
                                args,
                                goal_input,
                                gate_aff_input,
                                cmd_f,
                                cmd_a,
                                update_risk_memory=True,
                                state_tensor=gate_state,
                            )
                        gate_policy_goal = th.match_goal_dim(
                            gate_policy_goal,
                            int(gate_goal_dim),
                            label="play_gate_goal",
                        )
                        gate_action, _ = policy.get_action(
                            aff_input,
                            gate_state,
                            gate_policy_goal,
                            gate_difficulty,
                            deterministic=deterministic,
                        )
                        if th.is_learned_w_mode(getattr(args, "w_mode", "none")):
                            gate_y_raw = gate_action[:, 0]
                            learned_w = gate_action[:, 1]
                        else:
                            gate_y_raw = gate_action
                            learned_w = None
                        gate_diag = th.resolve_moe_gate_pcr(
                            env,
                            args,
                            gate_aff_input,
                            gate_y_raw,
                            cmd_f,
                            cmd_a,
                            learned_w=learned_w,
                        )
                        gate_y = gate_diag["y_eff"]
                        cmd = gate_diag["cmd"]
                    else:
                        policy_state = th.match_state_dim(
                            obs["state"],
                            policy_state_dim,
                            label="play_policy_state",
                        )
                        cmd, _ = policy.get_action(
                            aff_input,
                            policy_state,
                            goal_input,
                            difficulty_input,
                            deterministic=deterministic,
                        )
                        if (args.debug_cmd or debug) and skill == "avoid":
                            aff_input_flip = torch.flip(aff_input, dims=[-2])
                            aff_input_zero = torch.zeros_like(aff_input)
                            difficulty_zero = torch.zeros_like(difficulty_input)
                            cmd_flip, _ = policy.get_action(
                                aff_input_flip,
                                policy_state,
                                goal_input,
                                difficulty_input,
                                deterministic=True,
                            )
                            cmd_zero, _ = policy.get_action(
                                aff_input_zero,
                                policy_state,
                                goal_input,
                                difficulty_zero,
                                deterministic=True,
                            )
                            avoid_cf_cmds = {
                                "orig": cmd.detach().clone(),
                                "flip": cmd_flip.detach().clone(),
                                "zero": cmd_zero.detach().clone(),
                            }
                            aff_feat_orig = policy.affordance_encoder(aff_input)
                            aff_feat_flip = policy.affordance_encoder(aff_input_flip)
                            aff_feat_zero = policy.affordance_encoder(aff_input_zero)
                            hidden_orig = policy._encode_hidden(
                                aff_input,
                                obs["state"],
                                goal_input,
                                difficulty_input,
                                critic=False,
                            )
                            hidden_flip = policy._encode_hidden(
                                aff_input_flip,
                                obs["state"],
                                goal_input,
                                difficulty_input,
                                critic=False,
                            )
                            hidden_zero = policy._encode_hidden(
                                aff_input_zero,
                                obs["state"],
                                goal_input,
                                difficulty_zero,
                                critic=False,
                            )
                            avoid_cf_feats = {
                                "aff_orig": aff_feat_orig.detach().clone(),
                                "aff_flip": aff_feat_flip.detach().clone(),
                                "aff_zero": aff_feat_zero.detach().clone(),
                                "hid_orig": hidden_orig.detach().clone(),
                                "hid_flip": hidden_flip.detach().clone(),
                                "hid_zero": hidden_zero.detach().clone(),
                            }
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
            env.clearance_affordance_override = aff_map
            env.clearance_override = None
            env.reward_affordance_override = aff_map
            gate_y_raw_step = gate_diag["gate_y_raw"] if (is_gate and isinstance(gate_diag, dict)) else None
            pcr_risk_override = gate_diag.get("risk_F", None) if isinstance(gate_diag, dict) else None
            obs, rewards, dones, info = env.step(
                cmd,
                gate_y if is_gate else None,
                gate_y_raw=gate_y_raw_step,
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
                post_info["row_current_valid"] = gate_diag["row_current_valid"].detach().clone()
                post_info["row_not_released"] = gate_diag["row_not_released"].detach().clone()
                post_info["risk_memory"] = gate_diag["risk_memory"].detach().clone()
                post_info["signed_w"] = gate_diag["signed_w"].detach().clone()
                post_info["signed_w_active"] = gate_diag["signed_w_active"].detach().clone()
                post_info["w_support_correction"] = gate_diag["w_support_correction"].detach().clone()
                post_info["risk_diff_correction"] = gate_diag["risk_diff_correction"].detach().clone()
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
            manual_reset_mask = None
            if info is not None and isinstance(info, dict):
                manual_reset_mask = info.get("manual_reset_mask", None)
            if manual_reset_mask is None:
                manual_reset_mask = torch.zeros_like(dones, dtype=torch.bool)
            else:
                manual_reset_mask = manual_reset_mask.to(device=dones.device, dtype=torch.bool)
            success_mask = torch.zeros_like(dones, dtype=torch.bool)
            if isinstance(info, dict):
                success_from_info = info.get("success_mask", None)
                if torch.is_tensor(success_from_info):
                    success_mask = success_from_info.to(device=dones.device, dtype=torch.bool)
            if (not success_mask.any()) and isinstance(reward_terms, dict) and ("success_bonus" in reward_terms):
                success_bonus = reward_terms["success_bonus"]
                if torch.is_tensor(success_bonus):
                    success_mask = success_bonus.to(device=dones.device) > 0.0

            reach_mask = torch.zeros_like(dones, dtype=torch.bool)
            if isinstance(reward_terms, dict) and ("reach" in reward_terms):
                reach_term = reward_terms["reach"]
                if torch.is_tensor(reach_term):
                    reach_mask = dones & (reach_term.to(device=dones.device) > 0.0)
            timeout_mask = torch.zeros_like(dones, dtype=torch.bool)
            if isinstance(info, dict):
                timeout_from_info = info.get("timeout", None)
                if torch.is_tensor(timeout_from_info):
                    timeout_mask = timeout_from_info.to(device=dones.device, dtype=torch.bool)
            if (
                (not timeout_mask.any())
                and hasattr(env, "no_episode_timeout")
                and (not bool(getattr(env, "no_episode_timeout", False)))
            ):
                ep_len_snapshot = info.get("episode_length", None) if isinstance(info, dict) else None
                if torch.is_tensor(ep_len_snapshot):
                    timeout_mask = ep_len_snapshot.to(device=dones.device) >= int(getattr(env, "max_episode_length", 0))
            timeout_mask &= dones
            timeout_mask &= (~done_during) & (~success_mask) & (~reach_mask)
            other_done_mask = dones & (~done_during) & (~success_mask) & (~timeout_mask) & (~reach_mask)
            if dones.any():
                stack_reset_mask = dones.clone()
                track_done = bool(dones[track_env_idx].item())
                if track_done:
                    if track_ep_cmd_pred_x or track_ep_cmd_exec_x:
                        pred_arr = np.asarray(track_ep_cmd_pred_x, dtype=np.float64)
                        exec_arr = np.asarray(track_ep_cmd_exec_x, dtype=np.float64)
                        pred_mean = float(pred_arr.mean()) if pred_arr.size > 0 else 0.0
                        pred_std = float(pred_arr.std()) if pred_arr.size > 0 else 0.0
                        exec_mean = float(exec_arr.mean()) if exec_arr.size > 0 else 0.0
                        exec_std = float(exec_arr.std()) if exec_arr.size > 0 else 0.0
                        pred_sign = np.sign(pred_arr[np.abs(pred_arr) > 1.0e-4])
                        exec_sign = np.sign(exec_arr[np.abs(exec_arr) > 1.0e-4])
                        pred_switch = int(np.sum(pred_sign[1:] * pred_sign[:-1] < 0.0)) if pred_sign.size > 1 else 0
                        exec_switch = int(np.sum(exec_sign[1:] * exec_sign[:-1] < 0.0)) if exec_sign.size > 1 else 0
                        pred_switch_rate = float(pred_switch / max(pred_sign.size - 1, 1)) if pred_sign.size > 1 else 0.0
                        exec_switch_rate = float(exec_switch / max(exec_sign.size - 1, 1)) if exec_sign.size > 1 else 0.0
                        print(
                            "[PlayHigh][cmdx-episode] step={} pred_x(mean/std/switch/rate)={:.4f}/{:.4f}/{}/{} exec_x(mean/std/switch/rate)={:.4f}/{:.4f}/{}/{} samples={}".format(
                                step_idx,
                                pred_mean,
                                pred_std,
                                pred_switch,
                                f"{pred_switch_rate:.4f}",
                                exec_mean,
                                exec_std,
                                exec_switch,
                                f"{exec_switch_rate:.4f}",
                                max(pred_arr.size, exec_arr.size),
                            )
                        )
                    track_ep_cmd_pred_x.clear()
                    track_ep_cmd_exec_x.clear()
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
                    reach_n = int(reach_mask.sum().item())
                    other_n = int(other_done_mask.sum().item())
                    track_reason = "none"
                    if track_done:
                        if bool(done_during[track_env_idx].item()):
                            track_reason = "done_during(physics)"
                        elif bool(success_mask[track_env_idx].item()):
                            track_reason = "success"
                        elif bool(timeout_mask[track_env_idx].item()):
                            track_reason = "timeout"
                        elif bool(reach_mask[track_env_idx].item()):
                            track_reason = "reach"
                        else:
                            track_reason = "other"
                    print(
                        "[PlayHigh][reset] step={} done={} reason(physics/success/timeout/reach/other)={}/{}/{}/{}/{} track_env_reason={}".format(
                            step_idx, done_n, phys_n, succ_n, tout_n, reach_n, other_n, track_reason
                        )
                    )

            step_dx = 0.0
            step_dy = 0.0
            step_body_x = 0.0
            step_body_y = 0.0
            step_yaw = 0.0
            cmd_omega_track = 0.0
            band_debug = _extract_s_avoid_band_debug(env, env_id=track_env_idx)
            if input_enabled and debug and getattr(args, "task", "") == "s_avoid_basic":
                env_impl = getattr(env, "env", None)
                post_info_dbg = info.get("post_info") if isinstance(info, dict) else None
                cmd_exec_mean_dbg = None
                if isinstance(post_info_dbg, dict):
                    cmd_exec_mean_t = post_info_dbg.get("cmd_exec_mean", None)
                    if torch.is_tensor(cmd_exec_mean_t):
                        cmd_exec_mean_dbg = cmd_exec_mean_t[track_env_idx].detach().cpu().numpy()
                if env_impl is not None and hasattr(env_impl, "gym"):
                    env_impl.gym.clear_lines(viewer)
                _draw_s_avoid_band_debug_lines(env, viewer, env_id=track_env_idx)
                _draw_local_map_fov_debug_lines(env, viewer, env_id=track_env_idx)
                _draw_s_avoid_row_gap_debug_lines(
                    env,
                    viewer,
                    env_id=track_env_idx,
                    cmd_exec_mean=cmd_exec_mean_dbg,
                )
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
            goal_dist_delta = 0.0 if prev_dist is None else float(prev_dist - goal_dist)
            prev_dist = goal_dist
            if cmd.shape[0] > track_env_idx:
                track_ep_cmd_pred_x.append(float(cmd[track_env_idx, 0].detach().cpu().item()))
            post_info_track = info.get("post_info") if isinstance(info, dict) else None
            cmd_exec_mean_track = None
            if isinstance(post_info_track, dict):
                cmd_exec_mean_t = post_info_track.get("cmd_exec_mean", None)
                if torch.is_tensor(cmd_exec_mean_t) and cmd_exec_mean_t.shape[0] > track_env_idx:
                    cmd_exec_mean_track = float(cmd_exec_mean_t[track_env_idx, 0].detach().cpu().item())
            if cmd_exec_mean_track is None and hasattr(env.env, "commands") and env.env.commands.shape[0] > track_env_idx:
                cmd_exec_mean_track = float(env.env.commands[track_env_idx, 0].detach().cpu().item())
            if cmd_exec_mean_track is not None:
                track_ep_cmd_exec_x.append(cmd_exec_mean_track)

            if args.debug_cmd and step_idx % args.debug_interval == 0:
                env_idx = 0
                cmd_pred = cmd[env_idx].detach().cpu().numpy()
                cmd_exec = None
                cmd_exec_mean = None
                cmd_post = None
                cmd_override_final = None
                rotate_only_active_dbg = 0
                post_info = info.get("post_info") if info is not None else None
                if isinstance(post_info, dict):
                    cmd_exec_mean_t = post_info.get("cmd_exec_mean", None)
                    if torch.is_tensor(cmd_exec_mean_t):
                        cmd_exec_mean = cmd_exec_mean_t[env_idx].detach().cpu().numpy()
                    cmd_post_t = post_info.get("cmd_post", post_info.get("cmd_slew", None))
                    if torch.is_tensor(cmd_post_t):
                        cmd_post = cmd_post_t[env_idx].detach().cpu().numpy()
                    cmd_override_t = post_info.get("cmd_override_final", None)
                    if torch.is_tensor(cmd_override_t):
                        cmd_override_final = cmd_override_t[env_idx].detach().cpu().numpy()
                    rotate_only_t = post_info.get("rotate_only_active", None)
                    if torch.is_tensor(rotate_only_t):
                        rotate_only_active_dbg = int(bool(rotate_only_t[env_idx].item()))
                if hasattr(env.env, "commands"):
                    cmd_exec = env.env.commands[env_idx, :3].detach().cpu().numpy()
                cmd_show = cmd_exec_mean if cmd_exec_mean is not None else (cmd_exec if cmd_exec is not None else cmd_pred)
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
                gate_raw_val = 0.0
                gate_w_val = 0.0
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
                    if isinstance(gate_diag, dict):
                        gate_raw_val = float(gate_diag["gate_y_raw"][env_idx].detach().cpu())
                        gate_w_val = float(gate_diag["w"][env_idx].detach().cpu())
                if is_pcr_demo_task:
                    follow_goal_dbg = obs.get("follow_goal", obs["goal"])[env_idx].detach().cpu().numpy()
                    target_xy_dbg = np.zeros(2, dtype=np.float32)
                    robot_xy_dbg = np.zeros(2, dtype=np.float32)
                    if hasattr(env.env, "target_world"):
                        target_xy_dbg = env.env.target_world[env_idx].detach().cpu().numpy()
                    if hasattr(env.env, "root_states"):
                        robot_xy_dbg = env.env.root_states[env_idx, :2].detach().cpu().numpy()
                    follow_dist_dbg = float(np.linalg.norm(target_xy_dbg - robot_xy_dbg))
                    pcr_core_dbg = 0.0
                    pcr_gate_aux_dbg = 0.0
                    pcr_gap_success_dbg = 0.0
                    pcr_conflict_dbg = 0.0
                    pcr_follow_err_dbg = 0.0
                    pcr_follow_quality_dbg = 0.0
                    row_not_released_dbg = 0.0
                    target_finished_dbg = 0
                    follow_lost_dbg = 0
                    cmd_f_dbg = None
                    cmd_a_dbg = None
                    y_eff_dbg = gate_val
                    if reward_terms is not None:
                        pcr_core_dbg = float(reward_terms.get("pcr_core", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                        pcr_gate_aux_dbg = float(reward_terms.get("pcr_gate_aux", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                        pcr_gap_success_dbg = float(reward_terms.get("pcr_gap_success", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                        pcr_conflict_dbg = float(reward_terms.get("pcr_conflict", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                        pcr_follow_err_dbg = float(reward_terms.get("pcr_follow_err", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                        pcr_follow_quality_dbg = float(reward_terms.get("pcr_follow_quality", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                    if isinstance(info, dict):
                        target_finished_t = info.get("target_line_finished", None)
                        follow_lost_t = info.get("follow_lost_mask", None)
                        if torch.is_tensor(target_finished_t):
                            target_finished_dbg = int(bool(target_finished_t[env_idx].item()))
                        if torch.is_tensor(follow_lost_t):
                            follow_lost_dbg = int(bool(follow_lost_t[env_idx].item()))
                    if isinstance(post_info, dict):
                        cmd_f_t = post_info.get("cmd_F", None)
                        cmd_a_t = post_info.get("cmd_A", None)
                        y_eff_t = post_info.get("y_eff", None)
                        row_not_released_t = post_info.get("row_not_released", None)
                        if torch.is_tensor(cmd_f_t):
                            cmd_f_dbg = cmd_f_t[env_idx].detach().cpu().numpy()
                        if torch.is_tensor(cmd_a_t):
                            cmd_a_dbg = cmd_a_t[env_idx].detach().cpu().numpy()
                        if torch.is_tensor(y_eff_t):
                            y_eff_dbg = float(y_eff_t[env_idx].detach().cpu().item())
                        if torch.is_tensor(row_not_released_t):
                            row_not_released_dbg = float(row_not_released_t[env_idx].detach().cpu().item())
                    cmd_f_str = "None" if cmd_f_dbg is None else np.array2string(cmd_f_dbg, precision=3, floatmode="fixed")
                    cmd_a_str = "None" if cmd_a_dbg is None else np.array2string(cmd_a_dbg, precision=3, floatmode="fixed")
                    target_bearing_deg_dbg = float("nan")
                    target_in_rgb_fov_dbg = 0
                    if follow_goal_dbg is not None and len(follow_goal_dbg) >= 2:
                        target_bearing_rad_dbg = math.atan2(float(follow_goal_dbg[0]), float(follow_goal_dbg[1]))
                        target_bearing_deg_dbg = math.degrees(target_bearing_rad_dbg)
                        target_in_rgb_fov_dbg = int(abs(target_bearing_deg_dbg) <= (69.4 * 0.5 - 3.0))
                    print(
                        "[PlayHigh][PCR] step={} follow_dist={:.3f} follow_goal={} target_xy={} robot_xy={} "
                        "target_bearing={:.1f}deg inFOV={} gate(raw/eff/w)={:.3f}/{:.3f}/{:.3f} "
                        "conflict={:.3f} rowNR={:.1f} "
                        "reward(core/gate/gap)={:.3f}/{:.3f}/{:.3f} follow(err/q)={:.3f}/{:.3f} "
                        "flags(target_finish/follow_lost)={}/{}".format(
                            step_idx,
                            follow_dist_dbg,
                            np.array2string(follow_goal_dbg, precision=3, floatmode="fixed"),
                            np.array2string(target_xy_dbg, precision=3, floatmode="fixed"),
                            np.array2string(robot_xy_dbg, precision=3, floatmode="fixed"),
                            target_bearing_deg_dbg,
                            target_in_rgb_fov_dbg,
                            gate_raw_val,
                            y_eff_dbg,
                            gate_w_val,
                            pcr_conflict_dbg,
                            row_not_released_dbg,
                            pcr_core_dbg,
                            pcr_gate_aux_dbg,
                            pcr_gap_success_dbg,
                            pcr_follow_err_dbg,
                            pcr_follow_quality_dbg,
                            target_finished_dbg,
                            follow_lost_dbg,
                        )
                    )
                    print(
                        "[PlayHigh][PCR-cmd] pred={} exec={} cmd_F={} cmd_A={}".format(
                            np.array2string(cmd_pred, precision=3, floatmode="fixed"),
                            cmd_str,
                            cmd_f_str,
                            cmd_a_str,
                        )
                    )
                if not is_pcr_demo_task:
                    band_robot_x = float(band_debug.get("robot_x", 0.0)) if band_debug else 0.0
                    band_robot_y = float(band_debug.get("robot_y", 0.0)) if band_debug else 0.0
                    band_x_min_dbg = float(band_debug.get("band_x_min", 0.0)) if band_debug else 0.0
                    band_x_max_dbg = float(band_debug.get("band_x_max", 0.0)) if band_debug else 0.0
                    band_dx_out_dbg = float(band_debug.get("dx_out", 0.0)) if band_debug else 0.0
                    band_inside_dbg = int(bool(band_debug.get("inside_band_x", False))) if band_debug else 0
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
                    yaw_err_deg = abs(yaw_policy) * (180.0 / math.pi)
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
                    pass_side_dbg = 0.0
                    cmd_x_dbg = 0.0
                    cmd_x_dir_dbg = 0.0
                    row_lat_dbg = 0.0
                    row_gap_row_y_dbg = 0.0
                    row_gap_left_dbg = 0.0
                    row_gap_right_dbg = 0.0
                    row_gap_left_eff_dbg = 0.0
                    row_gap_right_eff_dbg = 0.0
                    row_x_err_now_dbg = 0.0
                    row_x_err_prev_dbg = 0.0
                    row_forward_dist_dbg = 0.0
                    row_gate_dbg = 0.0
                    row_push_err_dbg = 0.0
                    row_gate_active_dbg = 0
                    center_y_dbg = 0.0
                    cross_line_y_dbg = 0.0
                    cross_line_dist_dbg = 0.0
                    episode_progress_ratio_dbg = 0.0
                    episode_collision_dbg = 0
                    success_dbg = 0
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
                        pass_dir, pass_gate_dbg, pass_occ_dbg, pass_side = env._compute_passable_guidance(
                            debug_aff,
                            debug_goal,
                            block_mask=low_block_mask,
                        )
                        pass_gate_dbg = float(pass_gate_dbg[env_idx].detach().cpu())
                        pass_occ_dbg = float(pass_occ_dbg[env_idx].detach().cpu())
                        pass_side_dbg = float(pass_side[env_idx].detach().cpu())
                        cross_gate_dbg = float(cross_gate_dbg[env_idx].detach().cpu())
                        cross_width_dbg = float(cross_width_dbg[env_idx].detach().cpu())
                        pass_dir_dbg = pass_dir[env_idx].detach().cpu().numpy()
                        cross_dir_dbg = cross_dir[env_idx].detach().cpu().numpy()
                        pass_dir_norm = float(torch.norm(pass_dir[env_idx]).detach().cpu())
                        cross_dir_norm = float(torch.norm(cross_dir[env_idx]).detach().cpu())
                        robot_pos_dbg = env.env.root_states[:, :3]
                        (
                            _row_gap_center_t,
                            row_gap_row_y_t,
                            _row_gap_valid_t,
                            row_gap_left_t,
                            row_gap_right_t,
                            row_gap_left_eff_t,
                            row_gap_right_eff_t,
                            row_gap_eff_valid_t,
                        ) = env._compute_nearest_row_gap_target(
                            robot_pos_dbg
                        )
                        row_forward_dist_t, row_forward_valid_t = env._compute_nearest_row_forward_distance(
                            robot_pos_dbg
                        )
                        row_gap_row_y_dbg = float(row_gap_row_y_t[env_idx].detach().cpu())
                        row_gap_left_dbg = float(row_gap_left_t[env_idx].detach().cpu())
                        row_gap_right_dbg = float(row_gap_right_t[env_idx].detach().cpu())
                        row_gap_left_eff_dbg = float(row_gap_left_eff_t[env_idx].detach().cpu())
                        row_gap_right_eff_dbg = float(row_gap_right_eff_t[env_idx].detach().cpu())
                        row_gap_center_eff_dbg = 0.5 * (row_gap_left_eff_dbg + row_gap_right_eff_dbg)
                        robot_x_dbg = float(robot_pos_dbg[env_idx, 0].detach().cpu())
                        robot_y_dbg = float(robot_pos_dbg[env_idx, 1].detach().cpu())
                        robot_x_map_dbg, _robot_y_map_dbg = _world_point_to_map_xy(
                            env,
                            env_idx,
                            robot_x_dbg,
                            robot_y_dbg,
                        )
                        row_gap_row_y_map_dbg = float("nan")
                        if bool(row_gap_eff_valid_t[env_idx].item()):
                            _row_gap_center_map_dbg, row_gap_row_y_map_dbg = _world_point_to_map_xy(
                                env,
                                env_idx,
                                row_gap_center_eff_dbg,
                                row_gap_row_y_dbg,
                            )
                        if bool(row_gap_eff_valid_t[env_idx].item()):
                            row_x_err_now_dbg = max(row_gap_left_eff_dbg - robot_x_dbg, 0.0) + max(robot_x_dbg - row_gap_right_eff_dbg, 0.0)
                        else:
                            row_x_err_now_dbg = 0.0
                        prev_robot_x_dbg = robot_x_dbg
                        if prev_track_pos_world is not None:
                            prev_robot_x_dbg = float(prev_track_pos_world[0])
                        if bool(row_gap_eff_valid_t[env_idx].item()):
                            row_x_err_prev_dbg = max(row_gap_left_eff_dbg - prev_robot_x_dbg, 0.0) + max(prev_robot_x_dbg - row_gap_right_eff_dbg, 0.0)
                        else:
                            row_x_err_prev_dbg = 0.0
                        row_gate_on_dbg = float(getattr(env.reward_cfg, "avoid_row_gate_on", 1.0)) if env.reward_cfg is not None else 1.0
                        row_gate_full_dbg = float(getattr(env.reward_cfg, "avoid_row_gate_full", 0.4)) if env.reward_cfg is not None else 0.4
                        row_push_margin_dbg = float(getattr(env.reward_cfg, "avoid_row_push_margin", 0.25)) if env.reward_cfg is not None else 0.25
                        row_forward_dist_dbg = float(row_forward_dist_t[env_idx].detach().cpu())
                        row_gate_dbg = float(
                            torch.clamp(
                                (row_gate_on_dbg - row_forward_dist_t[env_idx]) / max(row_gate_on_dbg - row_gate_full_dbg, 1e-6),
                                min=0.0,
                                max=1.0,
                            ).detach().cpu()
                        )
                        row_push_err_dbg = float(
                            torch.clamp(
                                torch.tensor(row_x_err_now_dbg, device=row_forward_dist_t.device) / max(row_push_margin_dbg, 1e-6),
                                min=0.0,
                                max=1.0,
                            ).detach().cpu()
                        )
                        row_gate_active_dbg = int(
                            row_gate_dbg > 0.05
                            and bool(row_forward_valid_t[env_idx].item())
                            and bool(row_gap_eff_valid_t[env_idx].item())
                        )
                        map_support_dbg = {}
                        if obs is not None and "local_map_2ch" in obs and raw_aff_map is not None:
                            visible_dbg = getattr(env, "affordance_visible_mask", None)
                            map_support_dbg = _summarize_local_map_support(
                                raw_aff_map[env_idx],
                                obs["local_map_2ch"][env_idx],
                                visible_dbg,
                                float(getattr(env, "affordance_map_extent", 0.0)),
                                row_gap_row_y_map_dbg,
                                robot_x_map_dbg,
                            )
                        if reward_terms is not None and "row_lat" in reward_terms:
                            row_lat_dbg = float(reward_terms["row_lat"][env_idx].detach().cpu())
                        else:
                            row_lat_dbg = 0.0
                        if reward_terms is not None and "row_cmdx_reward" in reward_terms:
                            row_cmdx_dbg = float(reward_terms["row_cmdx_reward"][env_idx].detach().cpu())
                        else:
                            row_cmdx_dbg = 0.0
                        if isinstance(info, dict):
                            center_y_t = info.get("center_y", info.get("rear_y", None))
                            cross_line_y_t = info.get("cross_line_y", None)
                            cross_line_dist_t = info.get("cross_line_dist", None)
                            progress_ratio_t = info.get("s_avoid_progress_mask", None)
                            episode_collision_t = info.get("s_avoid_episode_collision", None)
                            success_t = info.get("success_mask", None)
                            if torch.is_tensor(center_y_t):
                                center_y_dbg = float(center_y_t[env_idx].detach().cpu())
                            if torch.is_tensor(cross_line_y_t):
                                cross_line_y_dbg = float(cross_line_y_t[env_idx].detach().cpu())
                            if torch.is_tensor(cross_line_dist_t):
                                cross_line_dist_dbg = float(cross_line_dist_t[env_idx].detach().cpu())
                            if torch.is_tensor(progress_ratio_t):
                                episode_progress_ratio_dbg = float(progress_ratio_t[env_idx].detach().cpu())
                            if torch.is_tensor(episode_collision_t):
                                episode_collision_dbg = int(bool(episode_collision_t[env_idx].item()))
                            if torch.is_tensor(success_t):
                                success_dbg = int(bool(success_t[env_idx].item()))
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
                        x_map = env.affordance_x_map
                        if x_map.device != passable.device:
                            x_map = x_map.to(passable.device)
                        right_mask = ((x_map > 0.0) & visible).float()
                        left_mask = ((x_map < 0.0) & visible).float()
                        cmd_x_dbg = 0.0
                        cmd_x_dir_dbg = 0.0
                        if cmd_show is not None:
                            cmd_x_dbg = float(cmd_show[0])
                            cmd_x_dir_dbg = math.tanh(cmd_x_dbg / 0.3)
                        x_dir_to_gap_dbg = 0.0
                        cmd_x_signed_gap_dbg = 0.0
                        if bool(row_gap_eff_valid_t[env_idx].item()):
                            x_delta_to_gap = row_gap_center_eff_dbg - robot_x_dbg
                            if x_delta_to_gap > 1e-6:
                                x_dir_to_gap_dbg = 1.0
                            elif x_delta_to_gap < -1e-6:
                                x_dir_to_gap_dbg = -1.0
                            if cmd_show is not None:
                                cmd_x_signed_gap_dbg = cmd_x_dbg * x_dir_to_gap_dbg
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
                        "[PlayHigh] step={} |cmd_xy|={:.3f} goal_dist_delta={:.3f} gate(raw/eff/w)={:.3f}/{:.3f}/{:.3f} reward={:.3f} (approach={:.3f}, heading={:.3f}, time={:.3f}, gate={:.3f}, risk={:.3f}) row(y/raw_l/raw_r/eff_l/eff_r/c)=({:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}) row_err(prev/now)={:.3f}/{:.3f} row_lat={:.3f} row_cmdx={:.3f} row(fwdDist/gate/pushErr/active)={:.3f}/{:.3f}/{:.3f}/{} x_dir={:.1f} cmd_x_toward_gap={:.3f} clearance={:.3f} risk_scale={:.3f} aff_stack(d/std/fill)={:.3f}/{:.3f}/{:.3f} cmd_pred={} goal={} goal_dist={:.3f} cmd_exec={} cmd_x={:.3f} cmd_x_dir={:.3f} yaw_raw={:.3f} yaw_policy={:.3f} yaw_err_deg={:.1f} rotate_only={} bear_y={:.3f} herr(+pi/2)={:.3f} herr(-pi/2)={:.3f}".format(
                            step_idx,
                            cmd_speed,
                            goal_dist_delta,
                            gate_raw_val,
                            gate_val,
                            gate_w_val,
                            reward_total,
                            reward_approach,
                            reward_heading,
                            reward_time,
                            reward_gate,
                            reward_risk,
                            row_gap_row_y_dbg,
                            row_gap_left_dbg,
                            row_gap_right_dbg,
                            row_gap_left_eff_dbg,
                            row_gap_right_eff_dbg,
                            row_gap_center_eff_dbg,
                            row_x_err_prev_dbg,
                            row_x_err_now_dbg,
                            row_lat_dbg,
                            row_cmdx_dbg,
                            row_forward_dist_dbg,
                            row_gate_dbg,
                            row_push_err_dbg,
                            row_gate_active_dbg,
                            x_dir_to_gap_dbg,
                            cmd_x_signed_gap_dbg,
                            clearance,
                            risk_scale,
                            aff_delta,
                            aff_std,
                            aff_filled,
                            np.array2string(cmd_pred, precision=3, floatmode="fixed"),
                            np.array2string(goal, precision=3, floatmode="fixed"),
                            goal_dist,
                            cmd_str,
                            cmd_x_dbg,
                            cmd_x_dir_dbg,
                            yaw_raw,
                            yaw_policy,
                            yaw_err_deg,
                            rotate_only_active_dbg,
                            bearing_y,
                            heading_err_pos,
                            heading_err_neg,
                        )
                    )
                    cmd_post_str = "None" if cmd_post is None else np.array2string(cmd_post, precision=3, floatmode="fixed")
                    cmd_override_str = "None" if cmd_override_final is None else np.array2string(
                        cmd_override_final, precision=3, floatmode="fixed"
                    )
                    print(
                        "[PlayHigh][cmd] postProcessor={} finalOverride={} execMean={}".format(
                            cmd_post_str,
                            cmd_override_str,
                            cmd_str,
                        )
                    )
                    print(
                        "[PlayHigh][diag] goal_bearing={:.3f} row_gap_width={:.3f} row_x={:.3f} row_err={:.3f} "
                        "row_fwdDist={:.3f} row_gate={:.3f} row_pushErr={:.3f} row_gateActive={} "
                        "gap_center={:.3f} x_dir={:.1f} cmd_x_toward_gap={:.3f} row_cmdx={:.3f} "
                        "pass_gate={:.3f} pass_occ={:.3f} cross_width={:.3f} vis_ratio={:.3f} "
                        "pass_vis/sector={:.3f}/{:.3f} sector_vis_ratio={:.3f}".format(
                            goal_bearing,
                            max(row_gap_right_eff_dbg - row_gap_left_eff_dbg, 0.0),
                            robot_x_dbg,
                            row_x_err_now_dbg,
                            row_forward_dist_dbg,
                            row_gate_dbg,
                            row_push_err_dbg,
                            row_gate_active_dbg,
                            row_gap_center_eff_dbg,
                            x_dir_to_gap_dbg,
                            cmd_x_signed_gap_dbg,
                            row_cmdx_dbg,
                            pass_gate_dbg,
                            pass_occ_dbg,
                            cross_width_dbg,
                            vis_ratio,
                            pass_vis_mean,
                            pass_sector_mean,
                            sector_vis_ratio,
                        )
                    )
                    print(
                        "[PlayHigh][term] center_y={:.3f} cross_line_y={:.3f} cross_line_dist={:.3f} episode_progress={:.3f} success={} episode_collision={}".format(
                            center_y_dbg,
                            cross_line_y_dbg,
                            cross_line_dist_dbg,
                            episode_progress_ratio_dbg,
                            success_dbg,
                            episode_collision_dbg,
                        )
                    )
                    if map_support_dbg:
                        map_occ_center = map_support_dbg.get("occupancy_center_map_xy", [0.0, 0.0])
                        obs_xy_true = _extract_s_avoid_debug_meta(env).get("obstacle_xy_map", [0.0, 0.0])
                        map_band_valid = int(bool(map_support_dbg.get("band_valid", False)))
                        map_better_side = float(map_support_dbg.get("better_side", 0.0))
                        map_row_side = float(map_support_dbg.get("row_side", 0.0))
                        map_row_delta = float(map_support_dbg.get("row_delta_m", 0.0))
                        map_robot_x = float(map_support_dbg.get("robot_x_map_m", 0.0))
                        print(
                            "[PlayHigh][map-gt] occ_true=({:.3f},{:.3f}) occ_gt=({:.3f},{:.3f}) occ_min={:.3f} "
                            "grid(size/cell)={} / {:.3f} row(mapIdx/centerY)={}/ {:.3f} "
                            "band(valid/cnt/l/r/c/w)={}/{}/{:.3f}/{:.3f}/{:.3f}/{:.3f} "
                            "row(robotX/bandDx)={:.3f}/{:.3f} "
                            "side(rowGT/gt/cmd)={:.1f}/{:.1f}/{:.1f}".format(
                                float(obs_xy_true[0]) if len(obs_xy_true) == 2 else 0.0,
                                float(obs_xy_true[1]) if len(obs_xy_true) == 2 else 0.0,
                                float(map_occ_center[0]) if len(map_occ_center) == 2 else 0.0,
                                float(map_occ_center[1]) if len(map_occ_center) == 2 else 0.0,
                                float(map_support_dbg.get("occupancy_min_distance_m", 0.0) or 0.0),
                                map_support_dbg.get("grid_size_xy", [0, 0]),
                                float(map_support_dbg.get("cell_size_m", 0.0)),
                                int(map_support_dbg.get("row_idx", -1)),
                                float(map_support_dbg.get("row_y_center_m", 0.0)),
                                map_band_valid,
                            int(map_support_dbg.get("band_count", 0)),
                            float(map_support_dbg.get("band_left_m", 0.0)),
                            float(map_support_dbg.get("band_right_m", 0.0)),
                            float(map_support_dbg.get("band_center_m", 0.0)),
                            float(map_support_dbg.get("band_width_m", 0.0)),
                            map_robot_x,
                            map_row_delta,
                            map_row_side,
                            x_dir_to_gap_dbg,
                            cmd_x_dir_dbg,
                        )
                    )
                    print(
                        "[PlayHigh][map-local] clr_l/r/diff={:.3f}/{:.3f}/{:.3f} occ_l/r={:.3f}/{:.3f} "
                        "clrSide={:.1f}".format(
                            float(map_support_dbg.get("left_clear_mean", 0.0)),
                            float(map_support_dbg.get("right_clear_mean", 0.0)),
                            float(map_support_dbg.get("clear_diff", 0.0)),
                            float(map_support_dbg.get("left_occ_mean", 0.0)),
                            float(map_support_dbg.get("right_occ_mean", 0.0)),
                            map_better_side,
                        )
                    )
                if avoid_cf_cmds is not None:
                    cmd_orig_dbg = avoid_cf_cmds["orig"][env_idx].detach().cpu().numpy()
                    cmd_flip_dbg = avoid_cf_cmds["flip"][env_idx].detach().cpu().numpy()
                    cmd_zero_dbg = avoid_cf_cmds["zero"][env_idx].detach().cpu().numpy()
                    print(
                        "[PlayHigh][cf] cmd_x orig/flip/zero={:.3f}/{:.3f}/{:.3f} "
                        "cmd_y orig/flip/zero={:.3f}/{:.3f}/{:.3f}".format(
                            float(cmd_orig_dbg[0]),
                            float(cmd_flip_dbg[0]),
                            float(cmd_zero_dbg[0]),
                            float(cmd_orig_dbg[1]),
                            float(cmd_flip_dbg[1]),
                            float(cmd_zero_dbg[1]),
                        )
                    )
                if avoid_cf_feats is not None:
                    aff_orig_dbg = avoid_cf_feats["aff_orig"][env_idx]
                    aff_flip_dbg = avoid_cf_feats["aff_flip"][env_idx]
                    aff_zero_dbg = avoid_cf_feats["aff_zero"][env_idx]
                    hid_orig_dbg = avoid_cf_feats["hid_orig"][env_idx]
                    hid_flip_dbg = avoid_cf_feats["hid_flip"][env_idx]
                    hid_zero_dbg = avoid_cf_feats["hid_zero"][env_idx]
                    aff_df = float(torch.norm(aff_orig_dbg - aff_flip_dbg).detach().cpu())
                    aff_dz = float(torch.norm(aff_orig_dbg - aff_zero_dbg).detach().cpu())
                    hid_df = float(torch.norm(hid_orig_dbg - hid_flip_dbg).detach().cpu())
                    hid_dz = float(torch.norm(hid_orig_dbg - hid_zero_dbg).detach().cpu())
                    print(
                        "[PlayHigh][cf-feat] aff(orig-flip/orig-zero)={:.4f}/{:.4f} "
                        "hid(orig-flip/orig-zero)={:.4f}/{:.4f}".format(
                            aff_df,
                            aff_dz,
                            hid_df,
                            hid_dz,
                        )
                    )
                if not is_pcr_demo_task:
                    if (
                        args.task == "s_avoid_basic"
                        and teacher_dump_interval_steps > 0
                        and raw_aff_map is not None
                        and step_idx >= next_teacher_dump_step
                    ):
                        local_map_2ch_dbg = obs.get("local_map_2ch", th.build_avoid_local_map_2ch(raw_aff_map))
                        teacher_dump_dir = _prepare_teacher_dump_dir(args)
                        teacher_save_path = os.path.join(teacher_dump_dir, f"teacher_step{step_idx:06d}.png")
                        _save_s_avoid_teacher_snapshot(
                            teacher_save_path,
                            env,
                            raw_aff_map[env_idx:env_idx + 1],
                            local_map_2ch_dbg[env_idx:env_idx + 1],
                            row_y_world=row_gap_row_y_dbg,
                            gap_left_world=row_gap_left_dbg,
                            gap_right_world=row_gap_right_dbg,
                            gap_left_eff_world=row_gap_left_eff_dbg,
                            gap_right_eff_world=row_gap_right_eff_dbg,
                            gap_center_eff_world=row_gap_center_eff_dbg,
                            row_lat_reward=row_lat_dbg,
                            row_cmdx_reward=row_cmdx_dbg,
                            x_err_now=row_x_err_now_dbg,
                            x_err_prev=row_x_err_prev_dbg,
                            robot_x_world=robot_x_dbg,
                            x_dir_to_gap=x_dir_to_gap_dbg,
                            cmd_x_signed_gap=cmd_x_signed_gap_dbg,
                            cmd_exec=np.asarray(cmd_show, dtype=np.float32),
                            band_dbg=_extract_s_avoid_band_debug(env, env_idx),
                        )
                        print(f"[PlayHigh] row-gap snapshot saved: {teacher_save_path}")
                        next_teacher_dump_step += teacher_dump_interval_steps
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
                    if band_debug:
                        print(
                            "[PlayHigh][band] robot_xy=({:.3f},{:.3f}) band_x=[{:.3f},{:.3f}] dx_out={:.4f} inside_x={}".format(
                                band_robot_x,
                                band_robot_y,
                                band_x_min_dbg,
                                band_x_max_dbg,
                                band_dx_out_dbg,
                                band_inside_dbg,
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
