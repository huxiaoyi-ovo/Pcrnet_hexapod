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
            "risk_f_sum": 0.0,
            "risk_a_sum": 0.0,
            "risk_delta_sum": 0.0,
            "near_miss_steps": 0,
        }
        for _ in RISK_BIN_LABELS
    ]


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
    clearance_f_sum: float = 0.0
    clearance_a_sum: float = 0.0
    risk_f_sum: float = 0.0
    risk_a_sum: float = 0.0
    near_miss_steps: int = 0
    gate_switch_count: int = 0
    cmd_jerk_lin_sum: float = 0.0
    cmd_jerk_ang_sum: float = 0.0
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
        self.aff_stack_buf = None
        self.follow_aff_stack_buf = None
        self.avoid_aff_stack_buf = None
        self.done_prev = None

        self.mass_kg = self._estimate_robot_mass_kg()
        self.g = 9.81

        self.policy = runtime.policy
        self.avoid_model = runtime.avoid_policy
        self.vision_model = runtime.vision_model
        self.primary_meta = runtime.primary_meta
        self.policy_meta = runtime.policy_meta
        self.aux_checkpoint_meta = runtime.aux_checkpoint_meta
        self.gate_state_dim = runtime.gate_state_dim
        self.avoid_state_dim = runtime.avoid_state_dim

        self.param_info = self._build_param_info()
        self.resolved_protocol = th.build_resolved_protocol(
            self.args,
            self.env,
            primary_ckpt_path=getattr(self.args, "teacher_ckpt", getattr(self.args, "pcr_ckpt", None)),
            primary_meta=self.policy_meta if isinstance(self.policy_meta, dict) else self.primary_meta,
            aux_sources=self.aux_checkpoint_meta,
        )

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
                raise ValueError("MoE mode requires --ckpt for gate policy")
            if not self.args.avoid_ckpt:
                raise ValueError("MoE mode requires --avoid_ckpt; follow side uses analytic expert")

            gate_ckpt = torch.load(self.args.ckpt, map_location=self.device)
            self.policy_meta = self._ckpt_meta(gate_ckpt)
            self.gate_state_dim = th.infer_checkpoint_state_dim(gate_ckpt) or state_dim
            self.policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=self.gate_state_dim,
                goal_dim=goal_dim,
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
                raise ValueError("Follow/Avoid mode requires --ckpt")
            self.policy = th.CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(self.device)
            ckpt = torch.load(self.args.ckpt, map_location=self.device)
            self.policy_meta = self._ckpt_meta(ckpt)
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

        if self.avoid_model is not None:
            total, trainable = _count_params(self.avoid_model)
            info["avoid_total"] = total
            info["avoid_trainable"] = trainable

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

        if self.args.skill == "moe":
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
                gate_difficulty = difficulty if self.args.gate_use_difficulty else torch.zeros_like(difficulty)
                if bool(getattr(self.args, "zero_local_map", False)):
                    gate_difficulty = torch.zeros_like(gate_difficulty)
                gate_y_raw, _ = self.policy.get_action(
                    policy_aff_stack,
                    gate_state,
                    goal,
                    gate_difficulty,
                    deterministic=not self.args.stochastic,
                )
                gate_diag = th.resolve_moe_gate_pcr(self.env, self.args, gate_aff_input, gate_y_raw, cmd_f, cmd_a)
            return gate_diag["cmd"], gate_diag["y_eff"], gate_diag

        with torch.no_grad():
            cmd, _ = self.policy.get_action(
                policy_aff_stack,
                state,
                goal,
                difficulty_input,
                deterministic=not self.args.stochastic,
            )
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
            self.aff_stack_buf = None
            self.follow_aff_stack_buf = None
            self.avoid_aff_stack_buf = None
            self.done_prev = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

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
                gate_y_raw = gate_diag["gate_y_raw"] if isinstance(gate_diag, dict) else None
                pcr_risk_override = gate_diag.get("risk_F", None) if isinstance(gate_diag, dict) else None
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
                cross_line_dist = info.get("cross_line_dist", None) if isinstance(info, dict) else None
                if cross_line_dist is None:
                    cross_line_dist = torch.full((self.env.num_envs,), float("nan"), device=self.device)
                episode_collision = info.get("s_avoid_episode_collision", None) if isinstance(info, dict) else None
                if episode_collision is None:
                    episode_collision = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                else:
                    episode_collision = episode_collision.to(device=self.device, dtype=torch.bool)

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
                    progress_val = _safe_float(progress_step[i].item(), default=0.0)
                    progress_val = float(np.clip(progress_val, 0.0, 1.0))
                    ai.progress_ratio_best = max(ai.progress_ratio_best, progress_val)
                    ai.progress_reached = ai.progress_reached or (progress_val > 0.0)
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

                        cmd_final_t = post_info.get("cmd_exec_mean", None)
                        if cmd_final_t is None:
                            cmd_final_t = post_info.get("cmd_override_final", None)
                        if cmd_final_t is None:
                            cmd_final_t = post_info.get("cmd_post", post_info.get("cmd_slew", cmd_raw))
                        if torch.is_tensor(cmd_final_t):
                            cmd_final_v = [float(x) for x in cmd_final_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_final_sum[j] += cmd_final_v[j]
                            if ai.prev_cmd_final is not None:
                                dx = cmd_final_v[0] - ai.prev_cmd_final[0]
                                dy = cmd_final_v[1] - ai.prev_cmd_final[1]
                                dw = cmd_final_v[2] - ai.prev_cmd_final[2]
                                ai.cmd_jerk_lin_sum += math.hypot(dx, dy)
                                ai.cmd_jerk_ang_sum += abs(dw)
                            ai.prev_cmd_final = cmd_final_v

                    if self.args.skill == "moe" and isinstance(post_info, dict):
                        gate_raw_t = post_info.get("gate_y_raw", None)
                        y_eff_t = post_info.get("y_eff", None)
                        w_t = post_info.get("w", None)
                        clr_f_t = post_info.get("clearance_F", None)
                        clr_a_t = post_info.get("clearance_A", None)
                        risk_f_t = post_info.get("risk_F", None)
                        risk_a_t = post_info.get("risk_A", None)
                        cmd_f_t = post_info.get("cmd_F", None)
                        cmd_a_t = post_info.get("cmd_A", None)
                        clearance_pp_t = post_info.get("clearance_pp", None)
                        safe_thr_t = post_info.get("post_safe_distance", None)

                        gate_raw_v = _safe_float(gate_raw_t[i].item(), default=0.0) if torch.is_tensor(gate_raw_t) else 0.0
                        y_eff_v = _safe_float(y_eff_t[i].item(), default=gate_raw_v) if torch.is_tensor(y_eff_t) else gate_raw_v
                        w_v = _safe_float(w_t[i].item(), default=0.0) if torch.is_tensor(w_t) else 0.0
                        clr_f_v = _safe_float(clr_f_t[i].item(), default=0.0) if torch.is_tensor(clr_f_t) else 0.0
                        clr_a_v = _safe_float(clr_a_t[i].item(), default=0.0) if torch.is_tensor(clr_a_t) else 0.0
                        risk_f_v = _safe_float(risk_f_t[i].item(), default=0.0) if torch.is_tensor(risk_f_t) else 0.0
                        risk_a_v = _safe_float(risk_a_t[i].item(), default=0.0) if torch.is_tensor(risk_a_t) else 0.0
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
                        ai.clearance_f_sum += clr_f_v
                        ai.clearance_a_sum += clr_a_v
                        ai.risk_f_sum += risk_f_v
                        ai.risk_a_sum += risk_a_v
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
                            bin_state["risk_f_sum"] += risk_f_v
                            bin_state["risk_a_sum"] += risk_a_v
                            bin_state["risk_delta_sum"] += risk_f_v - risk_a_v
                            if near_miss_now:
                                bin_state["near_miss_steps"] += 1

                        cmd_f_v = [float("nan"), float("nan"), float("nan")]
                        cmd_a_v = [float("nan"), float("nan"), float("nan")]
                        if torch.is_tensor(cmd_f_t):
                            cmd_f_v = [float(x) for x in cmd_f_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_f_sum[j] += cmd_f_v[j]
                        if torch.is_tensor(cmd_a_t):
                            cmd_a_v = [float(x) for x in cmd_a_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_a_sum[j] += cmd_a_v[j]
                        if near_miss_now:
                            ai.near_miss_steps += 1

                        if dump_timeseries and ai.step_hl % timeseries_stride == 0:
                            cmd_final_v = ai.prev_cmd_final if ai.prev_cmd_final is not None else [float("nan"), float("nan"), float("nan")]
                            ai.timeseries.append(
                                {
                                    "step_hl": int(ai.step_hl),
                                    "time_s": float(ai.step_hl) * float(self.env.high_level_dt),
                                    "progress": float(progress_val),
                                    "follow_err_m": follow_err_step,
                                    "gate_y_raw": gate_raw_v,
                                    "y_eff": y_eff_v,
                                    "w": w_v,
                                    "clearance_f": clr_f_v,
                                    "clearance_a": clr_a_v,
                                    "risk_f": risk_f_v,
                                    "risk_a": risk_a_v,
                                    "risk_delta": risk_f_v - risk_a_v,
                                    "clearance_pp": clr_pp_v,
                                    "near_miss": int(near_miss_now),
                                    "episode_collision": int(ai.episode_collision),
                                    "cmd_f_x": cmd_f_v[0],
                                    "cmd_f_y": cmd_f_v[1],
                                    "cmd_f_w": cmd_f_v[2],
                                    "cmd_a_x": cmd_a_v[0],
                                    "cmd_a_y": cmd_a_v[1],
                                    "cmd_a_w": cmd_a_v[2],
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
                    final_success = bool(success_step[i].item())
                    success_event = bool(ai.success)
                    final_collision = bool(ai.episode_collision)
                    task_success = final_success
                    collision_only = bool(final_collision and not task_success)
                    timeout_or_other = bool((not task_success) and (not final_collision))
                    if task_success:
                        outcome = "success"
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
                    episode_rows.append(
                        {
                            "episode_id": global_episode_idx,
                            "difficulty": float(d),
                            "success": int(task_success),
                            "success_event": int(success_event),
                            "time_to_success_s": (
                                float(ai.t_success_s)
                                if task_success and math.isfinite(float(ai.t_success_s))
                                else (float(ai.step_hl) * float(self.env.high_level_dt) if task_success else float("nan"))
                            ),
                            "success_event_time_s": float(ai.t_success_s) if success_event else float("nan"),
                            "episode_collision": int(final_collision),
                            "collision_time_s": float(ai.t_collision_s) if final_collision else float("nan"),
                            "collision_only": int(collision_only),
                            "success_and_collision": 0,
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
                            "clearance_f_mean": ai.clearance_f_sum / denom_steps,
                            "clearance_a_mean": ai.clearance_a_sum / denom_steps,
                            "risk_f_mean": ai.risk_f_sum / denom_steps,
                            "risk_a_mean": ai.risk_a_sum / denom_steps,
                            "risk_delta_mean": (ai.risk_f_sum - ai.risk_a_sum) / denom_steps,
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
                    float(sum(int(r.get("success", 0)) for r in high_rows) / max(1, n_high))
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

        def _risk_bins_summary(sub_rows: List[Dict]) -> List[Dict]:
            bins = _empty_risk_bin_state()
            episode_sets = [
                {"episodes": set(), "success": set(), "success_event": set(), "collision": set(), "progress": set()}
                for _ in RISK_BIN_LABELS
            ]
            for row_idx, row in enumerate(sub_rows):
                row_bins = row.get("risk_bin_stats", [])
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
                        "risk_f_sum",
                        "risk_a_sum",
                        "risk_delta_sum",
                        "near_miss_steps",
                    ):
                        bins[idx][key] += float(row_bin.get(key, 0.0) or 0.0)
                    episode_sets[idx]["episodes"].add(row_idx)
                    if int(row.get("success", 0)):
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
                    "risk_f_mean": bins[idx]["risk_f_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_a_mean": bins[idx]["risk_a_sum"] / float(steps) if steps > 0 else float("nan"),
                    "risk_delta_mean": bins[idx]["risk_delta_sum"] / float(steps) if steps > 0 else float("nan"),
                    "near_miss_rate": bins[idx]["near_miss_steps"] / float(steps) if steps > 0 else float("nan"),
                    "success_episode_rate": (
                        len(episode_sets[idx]["success"]) / float(episode_count)
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

        success_flags = [int(r["success"]) for r in rows]
        success_event_flags = [int(r.get("success_event", r["success"])) for r in rows]
        collision_only_flags = [int(r.get("collision_only", 0)) for r in rows]
        timeout_or_other_flags = [int(r.get("timeout_or_other", 0)) for r in rows]
        follow_mae = _clean([r["follow_mae_m"] for r in rows])
        follow_rmse = _clean([r["follow_rmse_m"] for r in rows])
        cot_vals = _clean([r["cot"] for r in rows])
        tts = _clean([r["time_to_success_s"] for r in rows if int(r["success"]) == 1])
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
        clearance_f_vals = _clean([r.get("clearance_f_mean", float("nan")) for r in rows])
        clearance_a_vals = _clean([r.get("clearance_a_mean", float("nan")) for r in rows])
        risk_f_vals = _clean([r.get("risk_f_mean", float("nan")) for r in rows])
        risk_a_vals = _clean([r.get("risk_a_mean", float("nan")) for r in rows])
        risk_delta_vals = _clean([r.get("risk_delta_mean", float("nan")) for r in rows])
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
        succ_eps = int(sum(success_flags))
        success_event_eps = int(sum(success_event_flags))
        collision_only_eps = int(sum(collision_only_flags))
        timeout_or_other_eps = int(sum(timeout_or_other_flags))
        collision_eps = int(sum(episode_collision))
        success_rate = float(succ_eps / max(1, total_eps))
        success_event_rate = float(success_event_eps / max(1, total_eps))
        collision_rate = float(collision_eps / max(1, total_eps))
        success_and_collision_rate = 0.0
        timeout_or_other_rate = float(timeout_or_other_eps / max(1, total_eps))
        collision_only_rate = float(collision_only_eps / max(1, total_eps))
        outcome_total_rate = success_rate + collision_only_rate + timeout_or_other_rate
        high_risk_overall = _high_risk_summary(rows)
        risk_bins_overall = _risk_bins_summary(rows)

        overall = {
            "episodes": total_eps,
            "success_episodes": succ_eps,
            "task_success_episodes": succ_eps,
            "success_event_episodes": success_event_eps,
            "success_and_collision_episodes": 0,
            "timeout_or_other_episodes": timeout_or_other_eps,
            "collision_only_episodes": collision_only_eps,
            "success_rate": success_rate,
            "task_success_rate": success_rate,
            "success_event_rate": success_event_rate,
            "success_and_collision_rate": success_and_collision_rate,
            "timeout_or_other_rate": timeout_or_other_rate,
            "collision_only_rate": collision_only_rate,
            "outcome_total_rate": outcome_total_rate,
            "fail_ratio": float(1.0 - success_rate),
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
            "clearance_f_mean": float(np.mean(clearance_f_vals)) if clearance_f_vals else float("nan"),
            "clearance_a_mean": float(np.mean(clearance_a_vals)) if clearance_a_vals else float("nan"),
            "risk_f_mean": float(np.mean(risk_f_vals)) if risk_f_vals else float("nan"),
            "risk_a_mean": float(np.mean(risk_a_vals)) if risk_a_vals else float("nan"),
            "risk_delta_mean": float(np.mean(risk_delta_vals)) if risk_delta_vals else float("nan"),
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
            "switch_rate_mean": float(np.mean(switch_vals)) if switch_vals else float("nan"),
            "near_miss_rate_mean": float(np.mean(near_miss_vals)) if near_miss_vals else float("nan"),
            "rotate_only_rate_mean": float(np.mean(rotate_only_vals)) if rotate_only_vals else float("nan"),
            "cmd_jerk_lin_mean": float(np.mean(cmd_jerk_lin_vals)) if cmd_jerk_lin_vals else float("nan"),
            "cmd_jerk_ang_mean": float(np.mean(cmd_jerk_ang_vals)) if cmd_jerk_ang_vals else float("nan"),
            "inference_latency_ms_p50": _quantile(latency_ms_samples, 0.50),
            "inference_latency_ms_p95": _quantile(latency_ms_samples, 0.95),
        }

        by_diff: Dict[str, Dict] = {}
        unique_d = sorted(set(float(r["difficulty"]) for r in rows))
        for d in unique_d:
            sub = [r for r in rows if float(r["difficulty"]) == float(d)]
            succ = [int(r["success"]) for r in sub]
            success_event_d = [int(r.get("success_event", r["success"])) for r in sub]
            collision_only_d = [int(r.get("collision_only", 0)) for r in sub]
            timeout_or_other_d = [int(r.get("timeout_or_other", 0)) for r in sub]
            mae = _clean([r["follow_mae_m"] for r in sub])
            rmse = _clean([r["follow_rmse_m"] for r in sub])
            cot = _clean([r["cot"] for r in sub])
            tts_d = _clean([r["time_to_success_s"] for r in sub if int(r["success"]) == 1])
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
            clearance_f_d = _clean([r.get("clearance_f_mean", float("nan")) for r in sub])
            clearance_a_d = _clean([r.get("clearance_a_mean", float("nan")) for r in sub])
            risk_f_d = _clean([r.get("risk_f_mean", float("nan")) for r in sub])
            risk_a_d = _clean([r.get("risk_a_mean", float("nan")) for r in sub])
            risk_delta_d = _clean([r.get("risk_delta_mean", float("nan")) for r in sub])
            switch_d = _clean([r.get("switch_rate", float("nan")) for r in sub])
            near_miss_d = _clean([r.get("near_miss_rate", float("nan")) for r in sub])
            rotate_only_d = _clean([r.get("rotate_only_rate", float("nan")) for r in sub])
            gate_region_y_eff_d = _clean([r.get("gate_region_y_eff_mean", float("nan")) for r in sub])
            gate_region_near_miss_d = _clean([r.get("gate_region_near_miss_rate", float("nan")) for r in sub])
            high_risk_d = _high_risk_summary(sub)
            n = len(sub)
            sr = float(sum(succ) / max(1, n))
            success_event_rate_d = float(sum(success_event_d) / max(1, n))
            timeout_or_other_rate_d = float(sum(timeout_or_other_d) / max(1, n))
            collision_only_rate_d = float(sum(collision_only_d) / max(1, n))
            by_diff[f"{d:.3f}"] = {
                "episodes": n,
                "success_rate": sr,
                "task_success_rate": sr,
                "success_event_rate": success_event_rate_d,
                "success_and_collision_rate": 0.0,
                "timeout_or_other_rate": timeout_or_other_rate_d,
                "collision_only_rate": collision_only_rate_d,
                "outcome_total_rate": sr + collision_only_rate_d + timeout_or_other_rate_d,
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
                "clearance_f_mean": float(np.mean(clearance_f_d)) if clearance_f_d else float("nan"),
                "clearance_a_mean": float(np.mean(clearance_a_d)) if clearance_a_d else float("nan"),
                "risk_f_mean": float(np.mean(risk_f_d)) if risk_f_d else float("nan"),
                "risk_a_mean": float(np.mean(risk_a_d)) if risk_a_d else float("nan"),
                "risk_delta_mean": float(np.mean(risk_delta_d)) if risk_delta_d else float("nan"),
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
                "switch_rate_mean": float(np.mean(switch_d)) if switch_d else float("nan"),
                "near_miss_rate_mean": float(np.mean(near_miss_d)) if near_miss_d else float("nan"),
                "rotate_only_rate_mean": float(np.mean(rotate_only_d)) if rotate_only_d else float("nan"),
            }

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
                "avoid_stage_override": None if getattr(self.args, "avoid_stage_override", None) is None else int(self.args.avoid_stage_override),
                "freeze_avoid_stage": bool(getattr(self.args, "freeze_avoid_stage", False)) or (
                    getattr(self.args, "avoid_stage_override", None) is not None
                ),
                "pcr_forced_forward_train_warmup_ratio": float(getattr(self.env, "forced_forward_train_warmup_ratio", float("nan"))),
                "deterministic_policy": bool(not self.args.stochastic),
                "mass_kg_for_cot": float(self.mass_kg),
                "success_definition": "env/play success_mask on done episodes",
                "success_event_source": "info.success_mask > s_avoid_episode_success_flags > success_bonus",
                "outcome_categories": ["success", "collision", "timeout_or_other"],
                "aff_stack": int(self.args.aff_stack),
                "camera_enable": bool(getattr(self.args, "camera_enable", False)),
                "camera_interval": None if getattr(self.args, "camera_interval", None) is None else int(self.args.camera_interval),
                "gate_use_difficulty": bool(self.args.gate_use_difficulty),
                "gate_safe_clamp": bool(getattr(self.args, "gate_safe_clamp", False)),
                "gate_safe_max": float(getattr(self.args, "gate_safe_max", 0.3)),
                "beta": None if self.args.beta is None else float(self.args.beta),
                "w_mode": str(self.args.w_mode),
                "w_tau": float(self.args.w_tau),
                "w_blend_mode": str(self.args.w_blend_mode),
                "w_disable_gate_safe_clamp": bool(self.args.w_disable_gate_safe_clamp),
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
            "resolved_protocol": self.resolved_protocol,
            "per_episode": [
                {k: v for k, v in row.items() if k != "risk_bin_stats"}
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

    rows = metrics.get("per_episode", [])
    fieldnames = [
        "episode_id",
        "difficulty",
        "success",
        "success_event",
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
        "clearance_f_mean",
        "clearance_a_mean",
        "risk_f_mean",
        "risk_a_mean",
        "risk_delta_mean",
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
            "step_hl",
            "time_s",
            "progress",
            "follow_err_m",
            "gate_y_raw",
            "y_eff",
            "w",
            "clearance_f",
            "clearance_a",
            "risk_f",
            "risk_a",
            "risk_delta",
            "clearance_pp",
            "near_miss",
            "episode_collision",
            "cmd_f_x",
            "cmd_f_y",
            "cmd_f_w",
            "cmd_a_x",
            "cmd_a_y",
            "cmd_a_w",
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
    risk_bins = metrics.get("risk_bins", [])
    if not isinstance(risk_bins, list) or len(risk_bins) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Eval] mechanism plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    labels = [str(b.get("bin", "")) for b in risk_bins]
    xs = np.arange(len(labels), dtype=np.float32)

    def vals(key: str) -> List[float]:
        out = []
        for item in risk_bins:
            v = _safe_float(item.get(key, float("nan")), default=float("nan"))
            out.append(v)
        return out

    y_eff = vals("y_eff_mean")
    w_vals = vals("w_mean")
    success = vals("success_episode_rate")
    collision = vals("collision_episode_rate")
    steps = vals("steps")
    episodes = vals("episode_count")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    fig.suptitle("PCR Conflict Arbitration Mechanism by Follow-Risk Bin", fontsize=13)

    axes[0, 0].plot(xs, y_eff, marker="o", linewidth=2.0, color="#1f77b4")
    axes[0, 0].set_title("Executed Follow Weight")
    axes[0, 0].set_ylabel("y_eff mean")
    axes[0, 0].set_ylim(0.0, 1.0)

    axes[0, 1].plot(xs, w_vals, marker="o", linewidth=2.0, color="#ff7f0e")
    axes[0, 1].set_title("Conflict Prior")
    axes[0, 1].set_ylabel("w mean")
    axes[0, 1].set_ylim(0.0, 1.0)

    axes[1, 0].plot(xs, success, marker="o", linewidth=2.0, color="#2ca02c", label="success")
    axes[1, 0].plot(xs, collision, marker="s", linewidth=2.0, color="#d62728", label="collision")
    axes[1, 0].set_title("Episode Outcome")
    axes[1, 0].set_ylabel("rate")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].legend(loc="best", frameon=False)

    axes[1, 1].bar(xs - 0.18, steps, width=0.36, color="#9467bd", label="steps")
    axes[1, 1].bar(xs + 0.18, episodes, width=0.36, color="#8c564b", label="episodes")
    axes[1, 1].set_title("Bin Support")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend(loc="best", frameon=False)

    for ax in axes.reshape(-1):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("risk_f bin")

    fig_path = os.path.join(out_dir, "mechanism_risk_bins.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"[Eval] mechanism plot: {fig_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Independent high-level evaluation")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--skill", type=str, default="follow", choices=["follow", "avoid", "moe"])

    parser.add_argument("--pcr_ckpt", type=str, required=True, help="PCR gate policy checkpoint")
    parser.add_argument("--follow_ckpt", type=str, default=None, help="旧参数保留；当前 moe 不再需要，因为 follow 使用解析式 expert")
    parser.add_argument("--avoid_ckpt", type=str, required=True, help="avoid checkpoint for moe")
    parser.add_argument("--vision_ckpt", type=str, default=None, help="vision checkpoint for student mode")
    parser.add_argument("--lowlevel_ckpt", type=str, required=True, help="low-level locomotion checkpoint")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--decimation", type=int, default=5)
    parser.add_argument("--aff_stack", type=int, default=1)
    parser.add_argument("--camera_enable", action="store_true", default=False)
    parser.add_argument("--camera_interval", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty_levels", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--stochastic", action="store_true", help="use stochastic policy sampling")
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

    parser.add_argument("--gate_use_difficulty", action="store_true")
    parser.add_argument("--gate_safe_clamp", action="store_true")
    parser.add_argument("--gate_safe_max", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=None)
    w_group = parser.add_mutually_exclusive_group(required=True)
    w_group.add_argument("--yonly", action="store_true", help="evaluate MoE-y without w")
    w_group.add_argument("--wgeom", action="store_true", help="evaluate MoE-y with geometric w")
    w_group.add_argument("--wlearned", action="store_true", help="evaluate MoE-y with learned w")
    parser.add_argument("--w_mode", type=str, default=None, choices=["none", "geom", "learned"])
    parser.add_argument("--w_tau", type=float, default=0.25)
    parser.add_argument("--w_blend_mode", type=str, default="multiply", choices=["multiply", "mix"])
    parser.add_argument("--w_disable_gate_safe_clamp", action="store_true")
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

    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--viewer", action="store_true", help="open Isaac Gym viewer during eval")
    parser.add_argument("--debug", action="store_true", help="enable debug prints and scene visualization")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/highlevel")

    args, unknown = parser.parse_known_args()
    if args.yonly:
        selected_w_mode = "none"
    elif args.wgeom:
        selected_w_mode = "geom"
    elif args.wlearned:
        selected_w_mode = "learned"
    else:
        parser.error("必须指定策略模式：--yonly / --wgeom / --wlearned 三选一")
    if args.w_mode is not None and args.w_mode != selected_w_mode:
        parser.error(
            f"--w_mode={args.w_mode} 与策略模式 --{ {'none': 'yonly', 'geom': 'wgeom', 'learned': 'wlearned'}[selected_w_mode] } 不一致"
        )
    args.w_mode = selected_w_mode
    for opt_name in ("pcr_ckpt", "avoid_ckpt", "lowlevel_ckpt"):
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
    out_dir = os.path.join(args.output_dir, f"{args.skill}_{args.mode}_{args.task}_{ts}")
    _write_outputs(metrics, out_dir)

    overall = metrics["overall"]
    print("=" * 72)
    print("Independent Eval Complete")
    print(f"Output: {out_dir}")
    print("-" * 72)
    print(f"Success rate: {overall['success_rate']:.4f} (fail={overall['fail_ratio']:.4f})")
    print(
        "Outcome rates success/collision/timeout (event/collision_all): "
        f"{overall['success_rate']:.4f} / {overall['collision_only_rate']:.4f} / "
        f"{overall['timeout_or_other_rate']:.4f} "
        f"(event={overall['success_event_rate']:.4f}, collision_all={overall['episode_collision_rate']:.4f})"
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
