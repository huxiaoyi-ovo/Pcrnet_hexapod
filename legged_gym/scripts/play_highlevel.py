#!/usr/bin/env python3
"""
Play a high-level (Teacher/Student) planner with Isaac Gym visualization.
"""

import os
import sys
import argparse
import math

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
from isaacgym import gymapi
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.scripts import train_highlevel as th


def parse_args():
    parser = argparse.ArgumentParser(description="Play high-level planner in Isaac Gym")
    parser.add_argument("--task", type=str, default="hex_ground", help="Task name (only hex_ground supported)")
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
        default="logs/hex_ground/Dec31_16-52-59_/model_6000.pt",
        help="Low-level policy checkpoint path",
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--vision_ckpt", type=str, default=None, help="Student vision checkpoint path")
    parser.add_argument("--aff_stack", type=int, default=4, help="affordance 堆叠帧数 (短时记忆)")
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
    parser.add_argument(
        "--debug_cmd",
        dest="debug_cmd",
        action="store_true",
        default=True,
        help="Print high-level command debug info",
    )
    parser.add_argument(
        "--no_debug_cmd",
        dest="debug_cmd",
        action="store_false",
        help="Disable debug output",
    )
    parser.add_argument("--debug_interval", type=int, default=10, help="Debug print interval (steps)")
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

    return args


def main():
    args = parse_args()
    if args.task != "hex_ground":
        raise ValueError("play_highlevel.py currently supports only --task hex_ground")
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

    env = th.HierarchicalHexapodEnv(args, device)
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
        print(f"[PlayHigh] ✓ Vision 加载成功: {args.vision_ckpt}")
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
        print("[PlayHigh] 键盘控制: R=重置, A=降级, D=升级")
        gym = env.env.gym
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET_ENV")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "LEVEL_DOWN")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "LEVEL_UP")
    obs = env.reset()
    heading_offset = 0.0
    if hasattr(env, "reward_cfg") and env.reward_cfg is not None:
        heading_offset = float(getattr(env.reward_cfg, "heading_offset_rad", 0.0))
    print(f"[PlayHigh] heading_offset_rad={heading_offset:.3f} (from reward_cfg)")
    def _get_aff_map(current_obs):
        if args.mode == "student":
            if current_obs is None:
                return None
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

    aff_map = _get_aff_map(obs)
    aff_shape = aff_map.shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack
    # Use clearer alias; implementation is identical to TerrainAdaptivePlanner.
    planner = th.HighLevelPlanner(
        affordance_channels=aff_channels,
        state_dim=obs["state"].shape[1],
        goal_dim=obs["goal"].shape[1],
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    planner.load_state_dict(state_dict)
    planner.eval()
    step_idx = 0
    deterministic = not args.stochastic
    camera_frame_idx = 0

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
                max_level = int(getattr(env.env, "max_terrain_level", 0))
                if max_level <= 0 and hasattr(env.env, "cfg"):
                    max_level = int(getattr(env.env.cfg.terrain, "num_rows", 1))
                current_level = int(env.env.terrain_levels[env_idx].item())
                new_level = int(np.clip(current_level + level_delta, 0, max_level - 1))
                env.env.terrain_levels[env_idx] = new_level
                if hasattr(env.env, "terrain_origins") and hasattr(env.env, "terrain_types"):
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
        with torch.no_grad():
            subgoal, intensity, _ = planner.get_action(
                aff_stack_buf,
                obs["state"],
                obs["goal"],
                difficulty,
                deterministic=deterministic,
            )
        if args.mode == "student":
            env.clearance_override = env._compute_clearance_from_affordance(aff_map)
            env.reward_affordance_override = aff_map
        obs, rewards, dones, info = env.step(subgoal, intensity)
        if dones.any():
            stack_reset_mask = dones.clone()

        env_idx = 0
        goal = obs["goal"][env_idx].detach().cpu().numpy()
        goal_dist = float(np.linalg.norm(goal))
        progress = 0.0 if prev_dist is None else float(prev_dist - goal_dist)
        prev_dist = goal_dist

        if args.debug_cmd and step_idx % args.debug_interval == 0:
            env_idx = 0
            sub = subgoal[env_idx].detach().cpu().numpy()
            inten = float(intensity[env_idx].detach().cpu())
            cmd = None
            if hasattr(env.env, "commands"):
                cmd = env.env.commands[env_idx, :3].detach().cpu().numpy()
            cmd_str = "None" if cmd is None else np.array2string(cmd, precision=3, floatmode="fixed")
            cmd_speed = 0.0 if cmd is None else float(np.linalg.norm(cmd[:2]))
            filtered_intensity = None
            if info is not None and "filtered_intensity" in info:
                filtered_intensity = float(info["filtered_intensity"][env_idx].detach().cpu())
            reward_total = float(rewards[env_idx].detach().cpu()) if rewards is not None else 0.0
            reward_terms = info.get("reward_terms") if info is not None else None
            reward_approach = 0.0
            reward_heading = 0.0
            reward_time = 0.0
            reward_intensity = 0.0
            clearance = 0.0
            intensity_goal_factor = 0.0
            intensity_clear_factor = 0.0
            optimal_intensity = 0.0
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
                reward_intensity = float(reward_terms.get("intensity_match", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                clearance = float(reward_terms.get("clearance", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                intensity_goal_factor = float(reward_terms.get("intensity_goal_factor", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                intensity_clear_factor = float(reward_terms.get("intensity_clear_factor", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                optimal_intensity = float(reward_terms.get("optimal_intensity", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                passable_gate = float(reward_terms.get("passable_gate", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                passable_align = float(reward_terms.get("passable_align", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                passable_occ_ratio = float(reward_terms.get("passable_occ_ratio", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                crossable_gate = float(reward_terms.get("crossable_gate", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                crossable_align = float(reward_terms.get("crossable_align", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
                crossable_width = float(reward_terms.get("crossable_width", torch.zeros(1, device=rewards.device))[env_idx].detach().cpu())
            yaw_raw = 0.0
            yaw_policy = 0.0
            heading_err_pos = 0.0
            heading_err_neg = 0.0
            bearing_y = 0.0
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
            if goal is not None:
                bearing_y = math.atan2(goal[0], goal[1])
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
                "[PlayHigh] step={} |cmd_xy|={:.3f} progress={:.3f} intensity={:.3f}/{:.3f} reward={:.3f} (approach={:.3f}, heading={:.3f}, time={:.3f}, intensity={:.3f}) passable(g/a/o)={:.3f}/{:.3f}/{:.3f} crossable(g/a/w)={:.3f}/{:.3f}/{:.3f} clr={:.3f} fac(g/c)={:.3f}/{:.3f} optI={:.3f} aff_stack(d/std/fill)={:.3f}/{:.3f}/{:.3f} subgoal={} goal={} dist={:.3f} cmd={} yaw_raw={:.3f} yaw_policy={:.3f} bear_y={:.3f} herr(+pi/2)={:.3f} herr(-pi/2)={:.3f}".format(
                    step_idx,
                    cmd_speed,
                    progress,
                    inten,
                    filtered_intensity if filtered_intensity is not None else 0.0,
                    reward_total,
                    reward_approach,
                    reward_heading,
                    reward_time,
                    reward_intensity,
                    passable_gate,
                    passable_align,
                    passable_occ_ratio,
                    crossable_gate,
                    crossable_align,
                    crossable_width,
                    clearance,
                    intensity_goal_factor,
                    intensity_clear_factor,
                    optimal_intensity,
                    aff_delta,
                    aff_std,
                    aff_filled,
                    np.array2string(sub, precision=3, floatmode="fixed"),
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
