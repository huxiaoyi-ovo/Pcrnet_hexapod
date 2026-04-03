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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd as s0_follow_expert_fn
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
    robot_pos_world_xy = state_tensor[:, :2]
    robot_heading = torch.atan2(torch.sin(state_tensor[:, 2]), torch.cos(state_tensor[:, 2]))
    target_world_xy = th.get_follow_target_world_xy(env_ref, state_tensor, goal_tensor)
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
    prev_y_eff: Optional[float] = None
    prev_cmd_final: Optional[list] = None
    cmd_f_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cmd_a_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cmd_final_sum: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


class EvalRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        th.import_modules()
        _setup_seed(args.seed)

        primary_meta = _load_experiment_meta_from_ckpt(getattr(args, "ckpt", None), self.device)
        self.primary_meta = primary_meta
        th.apply_experiment_meta_to_args(self.args, primary_meta, context="EvalHigh")
        th.apply_runtime_ablation_cli_overrides(self.args, primary_meta, context="EvalHigh")
        self.args.camera_enable = bool(getattr(self.args, "camera_enable", False)) or (self.args.mode == "student")

        env_cfg, train_cfg = th.task_registry.get_cfgs(name=args.task)
        env_cfg.seed = int(args.seed)
        th.apply_observation_contract_to_env_cfg(env_cfg, primary_meta, context="EvalHigh")
        self.env = th.HierarchicalHexapodEnv(args, self.device, env_cfg=env_cfg, train_cfg=train_cfg)

        self.aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
        self.aff_stack_buf = None
        self.follow_aff_stack_buf = None
        self.avoid_aff_stack_buf = None
        self.done_prev = None

        self.mass_kg = self._estimate_robot_mass_kg()
        self.g = 9.81

        self.policy = None
        self.avoid_model = None
        self.vision_model = None
        self.policy_meta = None
        self.aux_checkpoint_meta: Dict[str, Dict] = {}

        self._load_models()

        self.param_info = self._build_param_info()
        self.resolved_protocol = th.build_resolved_protocol(
            self.args,
            self.env,
            primary_ckpt_path=getattr(self.args, "ckpt", None),
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

            self.policy = th.GatePolicy(
                affordance_channels=aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
            ).to(self.device)
            gate_ckpt = torch.load(self.args.ckpt, map_location=self.device)
            self.policy_meta = self._ckpt_meta(gate_ckpt)
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

            avoid_aff_channels = int(obs["local_map_2ch"].shape[1] * self.aff_stack)
            self.avoid_model = th.CmdVelExpert(
                affordance_channels=avoid_aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(self.device)
            avoid_ckpt = torch.load(self.args.avoid_ckpt, map_location=self.device)
            avoid_meta = self._ckpt_meta(avoid_ckpt)
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
        mode = getattr(self.args, "mode", "teacher")
        if mode == "student":
            with torch.no_grad():
                vis_out = self.vision_model(obs_dict["depth"], normalize=True)
            follow_aff = torch.stack(
                [vis_out["occupancy"], vis_out["passable_gap"], vis_out["low_obstacle"]],
                dim=1,
            )
            follow_aff = th.resize_affordance_map(follow_aff, self.env.affordance_map_size)
            avoid_aff = th.build_avoid_local_map_2ch(
                follow_aff,
                visible_mask=getattr(self.env, "affordance_visible_mask", None),
            )
            follow_difficulty = th.difficulty_from_gap(follow_aff)
            avoid_difficulty = self.env._compute_objective_difficulty_from_local_map(avoid_aff)
        else:
            follow_aff = obs_dict["gt_affordance"]
            avoid_aff = obs_dict.get(
                "local_map_2ch",
                th.build_avoid_local_map_2ch(
                    follow_aff,
                    visible_mask=getattr(self.env, "affordance_visible_mask", None),
                ),
            )
            follow_difficulty = obs_dict.get("gt_difficulty", th.difficulty_from_gap(follow_aff))
            if "actor_difficulty" in obs_dict:
                avoid_difficulty = obs_dict["actor_difficulty"]
            else:
                avoid_difficulty = self.env._compute_objective_difficulty_from_local_map(avoid_aff)

        return {
            "follow_aff": follow_aff,
            "follow_difficulty": follow_difficulty,
            "avoid_aff": avoid_aff,
            "avoid_difficulty": avoid_difficulty,
            "gate_aff": avoid_aff,
            "gate_difficulty": avoid_difficulty,
        }

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
        goal = th.get_policy_goal_tensor(obs_dict, self.args.skill)
        avoid_goal = obs_dict["goal"]

        if self.args.skill == "moe":
            if avoid_aff_stack is None or avoid_difficulty is None or gate_aff_map is None:
                raise ValueError("MoE eval requires avoid affordance inputs and gate affordance map.")
            expert_state = th.get_moe_expert_state_inputs(state)
            with torch.no_grad():
                cmd_f = _compute_moe_follow_cmd_from_goal(
                    expert_state,
                    goal,
                    self.done_prev,
                    tuple(float(v) for v in self.env.post_processor.max_cmd.detach().cpu().tolist()),
                    env_ref=self.env,
                )
                cmd_a, _ = self.avoid_model.get_action(
                    avoid_aff_stack,
                    expert_state,
                    avoid_goal,
                    avoid_difficulty,
                    deterministic=not self.args.stochastic,
                )
                gate_difficulty = difficulty if self.args.gate_use_difficulty else torch.zeros_like(difficulty)
                gate_y_raw, _ = self.policy.get_action(
                    aff_stack,
                    state,
                    goal,
                    gate_difficulty,
                    deterministic=not self.args.stochastic,
                )
                gate_diag = th.resolve_moe_gate_pcr(self.env, self.args, gate_aff_map, gate_y_raw, cmd_f, cmd_a)
            return gate_diag["cmd"], gate_diag["y_eff"], gate_diag

        with torch.no_grad():
            cmd, _ = self.policy.get_action(
                aff_stack,
                state,
                goal,
                difficulty,
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

        global_episode_idx = 0

        for level_idx, d in enumerate(difficulty_levels):
            seed_level = int(self.args.seed + level_idx)
            _setup_seed(seed_level)

            self.env.set_scene_difficulty_target(float(d))
            self.env._apply_scene_difficulty_for_resets(None)
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
                self.env.reward_affordance_override = None
                gate_y_raw = gate_diag["gate_y_raw"] if isinstance(gate_diag, dict) else None
                next_obs, rewards, dones, info = self.env.step(cmd_raw, gate_y, gate_y_raw=gate_y_raw)
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

                    if bool(valid_follow[i].item()):
                        e = _safe_float(err[i].item(), default=0.0)
                        ai.follow_err_sum += e
                        ai.follow_err_sq_sum += e * e
                        ai.follow_err_count += 1

                    ai.energy_j += _safe_float((pwr[i] * float(self.env.high_level_dt)).item(), default=0.0)
                    ai.distance_m += _safe_float(ds[i].item(), default=0.0)
                    progress_val = _safe_float(progress_step[i].item(), default=0.0)
                    progress_val = float(np.clip(progress_val, 0.0, 1.0))
                    ai.progress_ratio_best = max(ai.progress_ratio_best, progress_val)
                    ai.progress_reached = ai.progress_reached or (progress_val > 0.0)
                    ai.episode_collision = ai.episode_collision or bool(episode_collision[i].item())
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

                        if torch.is_tensor(cmd_f_t):
                            cmd_f_v = [float(x) for x in cmd_f_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_f_sum[j] += cmd_f_v[j]
                        if torch.is_tensor(cmd_a_t):
                            cmd_a_v = [float(x) for x in cmd_a_t[i].detach().cpu().tolist()[:3]]
                            for j in range(3):
                                ai.cmd_a_sum[j] += cmd_a_v[j]
                        if torch.is_tensor(clearance_pp_t) and safe_thr_t is not None:
                            safe_thr_v = _safe_float(
                                safe_thr_t[i].item() if torch.is_tensor(safe_thr_t) and safe_thr_t.ndim > 0 else float(safe_thr_t),
                                default=0.0,
                            )
                            clr_pp_v = _safe_float(clearance_pp_t[i].item(), default=float("inf"))
                            if clr_pp_v < safe_thr_v:
                                ai.near_miss_steps += 1

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

                    denom_steps = float(max(ai.step_hl, 1))
                    episode_rows.append(
                        {
                            "episode_id": global_episode_idx,
                            "difficulty": float(d),
                            "success": int(final_success),
                            "time_to_success_s": float(ai.t_success_s) if final_success else float("nan"),
                            "follow_mae_m": follow_mae,
                            "follow_rmse_m": follow_rmse,
                            "cot": cot,
                            "energy_j": ai.energy_j,
                            "distance_m": ai.distance_m,
                            "steps_hl": ai.step_hl,
                            "cross_line_dist_end": ai.cross_line_dist_end,
                            "cross_line_dist_min": ai.cross_line_dist_min if math.isfinite(ai.cross_line_dist_min) else float("nan"),
                            "episode_collision": int(ai.episode_collision),
                            "progress_reached": int(ai.progress_reached),
                            "progress_ratio_best": ai.progress_ratio_best,
                            "gate_y_raw_mean": ai.gate_y_raw_sum / denom_steps,
                            "y_eff_mean": ai.y_eff_sum / denom_steps,
                            "w_mean": ai.w_sum / denom_steps,
                            "clearance_f_mean": ai.clearance_f_sum / denom_steps,
                            "clearance_a_mean": ai.clearance_a_sum / denom_steps,
                            "risk_f_mean": ai.risk_f_sum / denom_steps,
                            "risk_a_mean": ai.risk_a_sum / denom_steps,
                            "switch_rate": ai.gate_switch_count / denom_steps,
                            "near_miss_rate": ai.near_miss_steps / denom_steps,
                            "rotate_only_rate": ai.rotate_only_steps / denom_steps,
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
                    global_episode_idx += 1
                    done_episodes += 1

                    acc[i] = EpisodeAccumulator()

                self.done_prev = dones.clone()
                obs = next_obs

        # Trim overshoot to exact requested episode count.
        episode_rows = episode_rows[:episodes_total]

        metrics = self._aggregate_metrics(episode_rows, latency_ms_samples)
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

        success_flags = [int(r["success"]) for r in rows]
        follow_mae = _clean([r["follow_mae_m"] for r in rows])
        follow_rmse = _clean([r["follow_rmse_m"] for r in rows])
        cot_vals = _clean([r["cot"] for r in rows])
        tts = _clean([r["time_to_success_s"] for r in rows if int(r["success"]) == 1])
        cross_line_end = _clean([r.get("cross_line_dist_end", float("nan")) for r in rows])
        cross_line_min = _clean([r.get("cross_line_dist_min", float("nan")) for r in rows])
        episode_collision = [int(r.get("episode_collision", 0)) for r in rows]
        progress_flags = [int(r.get("progress_reached", 0)) for r in rows]
        progress_ratio = _clean([r.get("progress_ratio_best", float("nan")) for r in rows])
        gate_y_raw_vals = _clean([r.get("gate_y_raw_mean", float("nan")) for r in rows])
        y_eff_vals = _clean([r.get("y_eff_mean", float("nan")) for r in rows])
        w_vals = _clean([r.get("w_mean", float("nan")) for r in rows])
        clearance_f_vals = _clean([r.get("clearance_f_mean", float("nan")) for r in rows])
        risk_f_vals = _clean([r.get("risk_f_mean", float("nan")) for r in rows])
        switch_vals = _clean([r.get("switch_rate", float("nan")) for r in rows])
        near_miss_vals = _clean([r.get("near_miss_rate", float("nan")) for r in rows])
        rotate_only_vals = _clean([r.get("rotate_only_rate", float("nan")) for r in rows])
        cmd_jerk_lin_vals = _clean([r.get("cmd_jerk_lin_mean", float("nan")) for r in rows])
        cmd_jerk_ang_vals = _clean([r.get("cmd_jerk_ang_mean", float("nan")) for r in rows])

        total_eps = len(rows)
        succ_eps = int(sum(success_flags))
        success_rate = float(succ_eps / max(1, total_eps))

        overall = {
            "episodes": total_eps,
            "success_episodes": succ_eps,
            "success_rate": success_rate,
            "fail_ratio": float(1.0 - success_rate),
            "follow_mae_m_mean": float(np.mean(follow_mae)) if follow_mae else float("nan"),
            "follow_rmse_m_mean": float(np.mean(follow_rmse)) if follow_rmse else float("nan"),
            "time_to_success_s_mean": float(np.mean(tts)) if tts else float("nan"),
            "time_to_success_s_median": float(np.median(tts)) if tts else float("nan"),
            "time_to_success_s_p95": _quantile(tts, 0.95),
            "cot_mean": float(np.mean(cot_vals)) if cot_vals else float("nan"),
            "cot_median": float(np.median(cot_vals)) if cot_vals else float("nan"),
            "cot_p95": _quantile(cot_vals, 0.95),
            "cross_line_dist_end_mean": float(np.mean(cross_line_end)) if cross_line_end else float("nan"),
            "cross_line_dist_min_mean": float(np.mean(cross_line_min)) if cross_line_min else float("nan"),
            "episode_collision_rate": float(sum(episode_collision) / max(1, total_eps)),
            "progress_rate": float(sum(progress_flags) / max(1, total_eps)),
            "progress_ratio_mean": float(np.mean(progress_ratio)) if progress_ratio else 0.0,
            "progress_any_rate": float(sum(progress_flags) / max(1, total_eps)),
            "gate_y_raw_mean": float(np.mean(gate_y_raw_vals)) if gate_y_raw_vals else float("nan"),
            "y_eff_mean": float(np.mean(y_eff_vals)) if y_eff_vals else float("nan"),
            "w_mean": float(np.mean(w_vals)) if w_vals else float("nan"),
            "clearance_f_mean": float(np.mean(clearance_f_vals)) if clearance_f_vals else float("nan"),
            "risk_f_mean": float(np.mean(risk_f_vals)) if risk_f_vals else float("nan"),
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
            mae = _clean([r["follow_mae_m"] for r in sub])
            rmse = _clean([r["follow_rmse_m"] for r in sub])
            cot = _clean([r["cot"] for r in sub])
            tts_d = _clean([r["time_to_success_s"] for r in sub if int(r["success"]) == 1])
            cross_line_end_d = _clean([r.get("cross_line_dist_end", float("nan")) for r in sub])
            cross_line_min_d = _clean([r.get("cross_line_dist_min", float("nan")) for r in sub])
            collision_d = [int(r.get("episode_collision", 0)) for r in sub]
            progress_d = [int(r.get("progress_reached", 0)) for r in sub]
            progress_ratio_d = _clean([r.get("progress_ratio_best", float("nan")) for r in sub])
            gate_y_raw_d = _clean([r.get("gate_y_raw_mean", float("nan")) for r in sub])
            y_eff_d = _clean([r.get("y_eff_mean", float("nan")) for r in sub])
            w_d = _clean([r.get("w_mean", float("nan")) for r in sub])
            clearance_f_d = _clean([r.get("clearance_f_mean", float("nan")) for r in sub])
            risk_f_d = _clean([r.get("risk_f_mean", float("nan")) for r in sub])
            switch_d = _clean([r.get("switch_rate", float("nan")) for r in sub])
            near_miss_d = _clean([r.get("near_miss_rate", float("nan")) for r in sub])
            rotate_only_d = _clean([r.get("rotate_only_rate", float("nan")) for r in sub])
            n = len(sub)
            sr = float(sum(succ) / max(1, n))
            by_diff[f"{d:.3f}"] = {
                "episodes": n,
                "success_rate": sr,
                "fail_ratio": float(1.0 - sr),
                "follow_mae_m_mean": float(np.mean(mae)) if mae else float("nan"),
                "follow_rmse_m_mean": float(np.mean(rmse)) if rmse else float("nan"),
                "time_to_success_s_mean": float(np.mean(tts_d)) if tts_d else float("nan"),
                "time_to_success_s_median": float(np.median(tts_d)) if tts_d else float("nan"),
                "time_to_success_s_p95": _quantile(tts_d, 0.95),
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
                "risk_f_mean": float(np.mean(risk_f_d)) if risk_f_d else float("nan"),
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
                "deterministic_policy": bool(not self.args.stochastic),
                "mass_kg_for_cot": float(self.mass_kg),
                "aff_stack": int(self.args.aff_stack),
                "gate_use_difficulty": bool(self.args.gate_use_difficulty),
                "beta": None if self.args.beta is None else float(self.args.beta),
                "cmd_slew_lin": float(self.args.cmd_slew_lin),
                "cmd_slew_ang": float(self.args.cmd_slew_ang),
                "cmd_safe_dist": None if self.args.cmd_safe_dist is None else float(self.args.cmd_safe_dist),
                "cmd_free_dist": None if self.args.cmd_free_dist is None else float(self.args.cmd_free_dist),
                "disable_risk_scale": bool(self.args.disable_risk_scale),
                "ckpt": os.path.abspath(self.args.ckpt) if getattr(self.args, "ckpt", None) else None,
                "follow_ckpt": os.path.abspath(self.args.follow_ckpt) if getattr(self.args, "follow_ckpt", None) else None,
                "avoid_ckpt": os.path.abspath(self.args.avoid_ckpt) if getattr(self.args, "avoid_ckpt", None) else None,
                "vision_ckpt": os.path.abspath(self.args.vision_ckpt) if getattr(self.args, "vision_ckpt", None) else None,
                "low_level_ckpt": os.path.abspath(self.args.low_level_ckpt) if getattr(self.args, "low_level_ckpt", None) else None,
                "unknown_cli_args": list(getattr(self.args, "_unknown_cli", [])),
                "policy_experiment_meta": self.policy_meta,
            },
            "params": self.param_info,
            "overall": overall,
            "by_difficulty": by_diff,
            "resolved_protocol": self.resolved_protocol,
            "per_episode": rows,
        }
        return result


def _write_outputs(metrics: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    csv_path = os.path.join(out_dir, "metrics.csv")

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
        "time_to_success_s",
        "follow_mae_m",
        "follow_rmse_m",
        "cot",
        "energy_j",
        "distance_m",
        "steps_hl",
        "cross_line_dist_end",
        "cross_line_dist_min",
        "episode_collision",
        "progress_reached",
        "progress_ratio_best",
        "gate_y_raw_mean",
        "y_eff_mean",
        "w_mean",
        "clearance_f_mean",
        "clearance_a_mean",
        "risk_f_mean",
        "risk_a_mean",
        "switch_rate",
        "near_miss_rate",
        "rotate_only_rate",
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
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Independent high-level evaluation")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--mode", type=str, default="teacher", choices=["teacher", "student"])
    parser.add_argument("--skill", type=str, default="follow", choices=["follow", "avoid", "moe"])

    parser.add_argument("--ckpt", type=str, required=True, help="policy checkpoint (follow/avoid or gate for moe)")
    parser.add_argument("--follow_ckpt", type=str, default=None, help="旧参数保留；当前 moe 不再需要，因为 follow 使用解析式 expert")
    parser.add_argument("--avoid_ckpt", type=str, default=None, help="avoid checkpoint for moe")
    parser.add_argument("--vision_ckpt", type=str, default=None, help="vision checkpoint for student mode")
    parser.add_argument("--low_level_ckpt", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--decimation", type=int, default=5)
    parser.add_argument("--aff_stack", type=int, default=1)

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty_levels", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--stochastic", action="store_true", help="use stochastic policy sampling")

    parser.add_argument("--gate_use_difficulty", action="store_true")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--w_mode", type=str, default="none", choices=["none", "geom"])
    parser.add_argument("--w_tau", type=float, default=0.25)
    parser.add_argument("--w_blend_mode", type=str, default="multiply", choices=["multiply", "mix"])
    parser.add_argument("--w_disable_gate_safe_clamp", action="store_true")
    parser.add_argument("--cmd_slew_lin", type=float, default=0.2)
    parser.add_argument("--cmd_slew_ang", type=float, default=0.4)
    parser.add_argument("--cmd_safe_dist", type=float, default=None)
    parser.add_argument("--cmd_free_dist", type=float, default=None)
    parser.add_argument("--disable_risk_scale", action="store_true")

    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval/highlevel")

    args, unknown = parser.parse_known_args()
    th.capture_cli_explicit_arg_values(args, parser)
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
