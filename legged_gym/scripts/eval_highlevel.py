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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

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


class EvalRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        th.import_modules()
        _setup_seed(args.seed)

        primary_meta = _load_experiment_meta_from_ckpt(getattr(args, "ckpt", None), self.device)
        self.primary_meta = primary_meta
        th.apply_experiment_meta_to_args(self.args, primary_meta, context="EvalHigh")
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
        self.follow_model = None
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
            if not self.args.follow_ckpt or not self.args.avoid_ckpt:
                raise ValueError("MoE mode requires --follow_ckpt and --avoid_ckpt")

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

            follow_aff_channels = int(obs["gt_affordance"].shape[1] * self.aff_stack)
            self.follow_model = th.CmdVelExpert(
                affordance_channels=follow_aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(self.device)
            follow_ckpt = torch.load(self.args.follow_ckpt, map_location=self.device)
            follow_meta = self._ckpt_meta(follow_ckpt)
            self._validate_ckpt_meta(follow_meta, expected_skill="follow", source_name="follow expert ckpt")
            th.validate_checkpoint_contract_compatibility(
                self.policy_meta,
                follow_meta,
                reference_name="gate ckpt",
                candidate_name="follow expert ckpt",
                strict=True,
            )
            th.load_high_level_state_dict_compat(
                self.follow_model,
                _to_state_dict(follow_ckpt),
                label="eval_follow",
            )
            self.follow_model.eval()
            self.aux_checkpoint_meta["follow_ckpt"] = {
                "path": os.path.abspath(self.args.follow_ckpt),
                "experiment_meta": follow_meta,
            }

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

        if self.follow_model is not None:
            total, trainable = _count_params(self.follow_model)
            info["follow_total"] = total
            info["follow_trainable"] = trainable

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
    ):
        state = obs_dict["state"]
        goal = obs_dict["goal"]

        if self.args.skill == "moe":
            if follow_aff_stack is None or avoid_aff_stack is None or follow_difficulty is None or avoid_difficulty is None:
                raise ValueError("MoE eval requires separate follow/avoid affordance stacks and difficulties.")
            with torch.no_grad():
                cmd_f, _ = self.follow_model.get_action(
                    follow_aff_stack,
                    state,
                    goal,
                    follow_difficulty,
                    deterministic=not self.args.stochastic,
                )
                cmd_a, _ = self.avoid_model.get_action(
                    avoid_aff_stack,
                    state,
                    goal,
                    avoid_difficulty,
                    deterministic=not self.args.stochastic,
                )

                gate_difficulty = difficulty if self.args.gate_use_difficulty else torch.zeros_like(difficulty)
                gate_y, _ = self.policy.get_action(
                    aff_stack,
                    state,
                    goal,
                    gate_difficulty,
                    deterministic=not self.args.stochastic,
                )
                cmd = gate_y.unsqueeze(-1) * cmd_f + (1.0 - gate_y.unsqueeze(-1)) * cmd_a
            return cmd, gate_y

        with torch.no_grad():
            cmd, _ = self.policy.get_action(
                aff_stack,
                state,
                goal,
                difficulty,
                deterministic=not self.args.stochastic,
            )
        return cmd, None

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
                cmd_raw, gate_y = self._policy_step(
                    obs,
                    aff_stack,
                    difficulty_now,
                    follow_aff_stack=self.follow_aff_stack_buf,
                    follow_difficulty=aff_bundle["follow_difficulty"],
                    avoid_aff_stack=self.avoid_aff_stack_buf,
                    avoid_difficulty=aff_bundle["avoid_difficulty"],
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
                next_obs, rewards, dones, info = self.env.step(cmd_raw, gate_y)

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
                    progress_step = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
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
                    ai.progress_reached = ai.progress_reached or bool(progress_step[i].item())
                    ai.episode_collision = ai.episode_collision or bool(episode_collision[i].item())
                    cross_line_val = _safe_float(cross_line_dist[i].item(), default=float("nan"))
                    ai.cross_line_dist_end = cross_line_val
                    if math.isfinite(cross_line_val):
                        ai.cross_line_dist_min = min(ai.cross_line_dist_min, cross_line_val)

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
    parser.add_argument("--follow_ckpt", type=str, default=None, help="follow checkpoint for moe")
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
    parser.add_argument("--cmd_slew_lin", type=float, default=0.2)
    parser.add_argument("--cmd_slew_ang", type=float, default=0.4)
    parser.add_argument("--cmd_safe_dist", type=float, default=None)
    parser.add_argument("--cmd_free_dist", type=float, default=None)
    parser.add_argument("--disable_risk_scale", action="store_true")

    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval/highlevel")

    args, unknown = parser.parse_known_args()
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
