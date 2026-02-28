#!/usr/bin/env python3
"""
Play a high-level (Teacher/Student) planner with Isaac Gym visualization.
"""

import os
import sys
import argparse
import math
import types

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
from isaacgym import gymapi
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.scripts import train_highlevel as th


def parse_args():
    raw_argv = list(sys.argv)
    parser = argparse.ArgumentParser(description="Play high-level planner in Isaac Gym")
    parser.add_argument(
        "--task",
        type=str,
        default="hex_s1",
        help="Task name (hex_s0_follow / hex_s1 / hex_s1_follow_moving / hex_s2 / hex_calib)",
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
        default="logs/hex_s1/Dec31_16-52-59_/model_6000.pt",
        help="Low-level policy checkpoint path",
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Expert checkpoint path")
    parser.add_argument(
        "--skill",
        type=str,
        default="follow",
        choices=["follow", "avoid", "moe"],
        help="Expert skill: follow / avoid / moe (gate)",
    )
    parser.add_argument("--follow_ckpt", type=str, default=None, help="(moe) Follow expert checkpoint")
    parser.add_argument("--avoid_ckpt", type=str, default=None, help="(moe) Avoid expert checkpoint")
    parser.add_argument("--gate_use_difficulty", action="store_true", help="Gate 使用 difficulty 作为输入（特权信息）")
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
    parser.add_argument("--camera_interval", type=int, default=5, help="Camera capture interval")
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

    if args.camera_interval < 1:
        args.camera_interval = 1
    if args.debug_interval < 1:
        args.debug_interval = 1
    if args.debug and "--no_debug_cmd" not in raw_argv:
        args.debug_cmd = True

    return args


def main():
    args = parse_args()
    if args.task == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用 hex_ground / hex_s1..hex_s6 / hex_calib")
    if args.task not in ("hex_s0_follow", "hex_s1", "hex_s1_follow_moving", "hex_s2", "hex_calib"):
        raise ValueError("play_highlevel.py supports only --task hex_s0_follow/hex_s1/hex_s1_follow_moving/hex_s2/hex_calib")
    debug = bool(getattr(args, "debug", False))
    def dprint(*vals, **kwargs):
        if debug:
            print(*vals, **kwargs)
    dprint("[PlayHigh] V5 主线任务默认 hex_s1；hex_terrain 已移除。")
    if getattr(args, "aff_stack", 1) > 1:
        print(f"[PlayHigh] aff_stack={args.aff_stack}: 输入通道数改变，需与 ckpt 训练时一致，否则无法加载。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th.import_modules()
    if args.mode == "student" and not args.vision_ckpt:
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")

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
    env = th.HierarchicalHexapodEnv(args, device, env_cfg=env_cfg, train_cfg=train_cfg)
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
    vision_model = None
    if args.mode == "student":
        vision_model = th.AffordanceEstimator(
            depth_channels=1,
            output_size=16,
            max_depth_range=5.0
        ).to(device)
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        vision_model.load_state_dict(state_dict)
        vision_model.eval()
        dprint(f"[PlayHigh] ✓ Vision 加载成功: {args.vision_ckpt}")
    if args.camera_env < 0:
        args.camera_env = 0
    if args.camera_env >= env.num_envs:
        print(f"[PlayHigh] ⚠ camera_env={args.camera_env} out of range; clamping to {env.num_envs - 1}.")
        args.camera_env = env.num_envs - 1
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
    def _get_aff_map(current_obs):
        if current_obs is None:
            raise ValueError("obs is None when building affordance map.")
        if args.mode == "student":
            if vision_model is None:
                raise RuntimeError("vision_model is not initialized in student mode.")
            with torch.no_grad():
                vis_out = vision_model(current_obs["depth"], normalize=True)
                return torch.stack([
                    vis_out["occupancy"],
                    vis_out["passable_gap"],
                    vis_out["low_obstacle"],
                ], dim=1)
        return current_obs["gt_affordance"]

    def _get_difficulty(current_obs, current_aff):
        if args.mode == "student":
            return th.difficulty_from_gap(current_aff)
        return current_obs["gt_difficulty"]

    obs = env.reset()
    aff_map = _get_aff_map(obs)
    aff_shape = aff_map.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    skill = getattr(args, "skill", "follow")
    is_gate = skill == "moe"
    if is_gate:
        if not args.follow_ckpt or not args.avoid_ckpt:
            raise ValueError("moe 需要 --follow_ckpt 和 --avoid_ckpt")
        policy = th.GatePolicy(
            affordance_channels=aff_channels,
            state_dim=obs["state"].shape[1],
            goal_dim=obs["goal"].shape[1],
        ).to(device)
        follow_policy = th.CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=obs["state"].shape[1],
            goal_dim=obs["goal"].shape[1],
            cmd_scale=cmd_scale,
        ).to(device)
        avoid_policy = th.CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=obs["state"].shape[1],
            goal_dim=obs["goal"].shape[1],
            cmd_scale=cmd_scale,
        ).to(device)
        gate_ckpt = torch.load(args.teacher_ckpt, map_location=device)
        gate_state = gate_ckpt["model_state_dict"] if isinstance(gate_ckpt, dict) and "model_state_dict" in gate_ckpt else gate_ckpt
        policy.load_state_dict(gate_state)
        for model, ckpt_path in [(follow_policy, args.follow_ckpt), (avoid_policy, args.avoid_ckpt)]:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
        policy.eval()
    else:
        policy = th.CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=obs["state"].shape[1],
            goal_dim=obs["goal"].shape[1],
            cmd_scale=cmd_scale,
        ).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict)
        policy.eval()
    step_idx = 0
    deterministic = not args.stochastic
    camera_frame_idx = 0
    track_env_idx = min(max(int(args.camera_env), 0), env.num_envs - 1)
    prev_track_pos_world = None
    prev_track_yaw = None
    axis_disp_world_sum = np.zeros(2, dtype=np.float64)
    axis_disp_body_sum = np.zeros(2, dtype=np.float64)
    axis_disp_count = 0

    prev_dist = None
    aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
    aff_stack_fill = torch.ones(env.num_envs, device=device)
    stack_reset_mask = None
    level_up_pressed = False
    level_down_pressed = False
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
            obs = env.reset()
            aff_map = _get_aff_map(obs)
            aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
            aff_stack_fill.fill_(1)
            stack_reset_mask = None
            prev_dist = None
            continue
        reset_mask = stack_reset_mask
        if stack_reset_mask is not None and stack_reset_mask.any():
            if args.mode == "student":
                reset_aff = _get_aff_map(obs)
            else:
                reset_aff = obs["gt_affordance"]
            aff_stack_buf[stack_reset_mask] = reset_aff[stack_reset_mask].repeat(1, aff_stack, 1, 1)
            aff_stack_fill[stack_reset_mask] = 1
            stack_reset_mask = None
        aff_map = _get_aff_map(obs)
        aff_stack_buf = torch.roll(aff_stack_buf, shifts=-aff_map.shape[1], dims=1)
        aff_stack_buf[:, -aff_map.shape[1]:, :, :] = aff_map
        if aff_stack > 1:
            if reset_mask is None:
                aff_stack_fill = torch.clamp(aff_stack_fill + 1, max=aff_stack)
            else:
                inc_mask = ~reset_mask
                if inc_mask.any():
                    aff_stack_fill[inc_mask] = torch.clamp(aff_stack_fill[inc_mask] + 1, max=aff_stack)
        else:
            aff_stack_fill.fill_(1)
        difficulty = _get_difficulty(obs, aff_map)
        gate_y = None
        with torch.no_grad():
            if is_gate:
                gate_difficulty = difficulty if args.gate_use_difficulty else torch.zeros_like(difficulty)
                cmd_f, _ = follow_policy.get_action(
                    aff_stack_buf,
                    obs["state"],
                    obs["goal"],
                    difficulty,
                    deterministic=True,
                )
                cmd_a, _ = avoid_policy.get_action(
                    aff_stack_buf,
                    obs["state"],
                    obs["goal"],
                    difficulty,
                    deterministic=True,
                )
                gate_y, _ = policy.get_action(
                    aff_stack_buf,
                    obs["state"],
                    obs["goal"],
                    gate_difficulty,
                    deterministic=deterministic,
                )
                cmd = gate_y.unsqueeze(-1) * cmd_f + (1.0 - gate_y.unsqueeze(-1)) * cmd_a
            else:
                cmd, _ = policy.get_action(
                    aff_stack_buf,
                    obs["state"],
                    obs["goal"],
                    difficulty,
                    deterministic=deterministic,
                )
        if force_cmd_tensor is not None:
            cmd = force_cmd_tensor.expand(env.num_envs, -1)
        if args.mode == "student":
            env.clearance_override = env._compute_clearance_from_affordance(aff_map)
            env.reward_affordance_override = aff_map
        obs, rewards, dones, info = env.step(cmd, gate_y if is_gate else None)
        if dones.any():
            stack_reset_mask = dones.clone()

        step_dx = 0.0
        step_dy = 0.0
        step_body_x = 0.0
        step_body_y = 0.0
        if hasattr(env.env, "root_states"):
            root = env.env.root_states[track_env_idx]
            pos_xy = root[:2].detach().cpu().numpy()
            quat = root[3:7].detach().cpu().numpy()
            x_q, y_q, z_q, w_q = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            yaw_now = math.atan2(2.0 * (w_q * z_q + x_q * y_q), 1.0 - 2.0 * (y_q * y_q + z_q * z_q))
            if prev_track_pos_world is not None and prev_track_yaw is not None:
                step_dx = float(pos_xy[0] - prev_track_pos_world[0])
                step_dy = float(pos_xy[1] - prev_track_pos_world[1])
                cos_h = math.cos(prev_track_yaw)
                sin_h = math.sin(prev_track_yaw)
                step_body_x = cos_h * step_dx - sin_h * step_dy
                step_body_y = sin_h * step_dx + cos_h * step_dy
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
        progress = 0.0 if prev_dist is None else float(prev_dist - goal_dist)
        prev_dist = goal_dist

        if args.debug_cmd and step_idx % args.debug_interval == 0:
            env_idx = 0
            cmd_pred = cmd[env_idx].detach().cpu().numpy()
            cmd_exec = None
            if hasattr(env.env, "commands"):
                cmd_exec = env.env.commands[env_idx, :3].detach().cpu().numpy()
            cmd_show = cmd_exec if cmd_exec is not None else cmd_pred
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
                debug_aff = _get_aff_map(obs)
            if debug_aff is not None:
                cross_dir, cross_gate_dbg, cross_width_dbg, low_block_mask = env._compute_low_obstacle_guidance(
                    debug_aff
                )
                debug_goal = obs["goal"]
                pass_dir, pass_gate_dbg, pass_occ_dbg = env._compute_passable_guidance(
                    debug_aff,
                    debug_goal,
                    block_mask=low_block_mask,
                )
                pass_gate_dbg = float(pass_gate_dbg[env_idx].detach().cpu())
                pass_occ_dbg = float(pass_occ_dbg[env_idx].detach().cpu())
                cross_gate_dbg = float(cross_gate_dbg[env_idx].detach().cpu())
                cross_width_dbg = float(cross_width_dbg[env_idx].detach().cpu())
                pass_dir_dbg = pass_dir[env_idx].detach().cpu().numpy()
                cross_dir_dbg = cross_dir[env_idx].detach().cpu().numpy()
                pass_dir_norm = float(torch.norm(pass_dir[env_idx]).detach().cpu())
                cross_dir_norm = float(torch.norm(cross_dir[env_idx]).detach().cpu())
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
                "[PlayHigh] step={} |cmd_xy|={:.3f} progress={:.3f} gate_y={:.3f} reward={:.3f} (approach={:.3f}, heading={:.3f}, time={:.3f}, gate={:.3f}, risk={:.3f}) passable(g/a/o)={:.3f}/{:.3f}/{:.3f} crossable(g/a/w)={:.3f}/{:.3f}/{:.3f} clr={:.3f} risk_scale={:.3f} aff_stack(d/std/fill)={:.3f}/{:.3f}/{:.3f} cmd_pred={} goal={} dist={:.3f} cmd_exec={} yaw_raw={:.3f} yaw_policy={:.3f} bear_y={:.3f} herr(+pi/2)={:.3f} herr(-pi/2)={:.3f}".format(
                    step_idx,
                    cmd_speed,
                    progress,
                    gate_val,
                    reward_total,
                    reward_approach,
                    reward_heading,
                    reward_time,
                    reward_gate,
                    reward_risk,
                    passable_gate,
                    passable_align,
                    passable_occ_ratio,
                    crossable_gate,
                    crossable_align,
                    crossable_width,
                    clearance,
                    risk_scale,
                    aff_delta,
                    aff_std,
                    aff_filled,
                    np.array2string(cmd_pred, precision=3, floatmode="fixed"),
                    np.array2string(goal, precision=3, floatmode="fixed"),
                    goal_dist,
                    cmd_str,
                    yaw_raw,
                    yaw_policy,
                    bearing_y,
                    heading_err_pos,
                    heading_err_neg,
                )
            )
            print(
                "[PlayHigh][diag] goal_bear={:.3f} pass_bear={:.3f} cross_bear={:.3f} err_gp={:.3f} err_gc={:.3f} "
                "pass_dir={} cross_dir={} norm_p/c={:.3f}/{:.3f} pass_gate_dbg={:.3f} pass_occ_dbg={:.3f} "
                "cross_gate_dbg={:.3f} cross_width_dbg={:.3f} low_block_ratio={:.3f} vis_ratio={:.3f} "
                "pass_vis/sector={:.3f}/{:.3f} low_vis/sector={:.3f}/{:.3f} sector_vis_ratio={:.3f} "
                "pass_out_sector={}".format(
                    goal_bearing,
                    pass_bearing,
                    cross_bearing,
                    pass_goal_err,
                    cross_goal_err,
                    "None" if pass_dir_dbg is None else np.array2string(pass_dir_dbg, precision=3, floatmode="fixed"),
                    "None" if cross_dir_dbg is None else np.array2string(cross_dir_dbg, precision=3, floatmode="fixed"),
                    pass_dir_norm,
                    cross_dir_norm,
                    pass_gate_dbg,
                    pass_occ_dbg,
                    cross_gate_dbg,
                    cross_width_dbg,
                    low_block_ratio,
                    vis_ratio,
                    pass_vis_mean,
                    pass_sector_mean,
                    low_vis_mean,
                    low_sector_mean,
                    sector_vis_ratio,
                    pass_out_sector,
                )
            )
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
            mean_dx = axis_disp_world_sum[0] / max(axis_disp_count, 1)
            mean_dy = axis_disp_world_sum[1] / max(axis_disp_count, 1)
            mean_bx = axis_disp_body_sum[0] / max(axis_disp_count, 1)
            mean_by = axis_disp_body_sum[1] / max(axis_disp_count, 1)
            print(
                "[PlayHigh][axis] force_cmd={} step_world=({:.4f},{:.4f}) step_body(x_right,y_forward)=({:.4f},{:.4f}) "
                "mean_world=({:.4f},{:.4f}) mean_body=({:.4f},{:.4f}) samples={}".format(
                    "None" if force_cmd_tensor is None else np.array2string(force_cmd_tensor[0].detach().cpu().numpy(), precision=3, floatmode="fixed"),
                    step_dx,
                    step_dy,
                    step_body_x,
                    step_body_y,
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
            break


if __name__ == "__main__":
    main()
