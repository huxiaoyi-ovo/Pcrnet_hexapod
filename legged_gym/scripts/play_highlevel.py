#!/usr/bin/env python3
"""
Play a high-level (Teacher/Student) planner with Isaac Gym visualization.
"""

import os
import sys
import argparse

import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
import torch
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.scripts import train_highlevel as th


def parse_args():
    parser = argparse.ArgumentParser(description="Play high-level planner in Isaac Gym")
    parser.add_argument("--task", type=str, default="hex_ground", help="Task name (only hex_ground supported)")
    parser.add_argument(
        "--low_level_ckpt",
        type=str,
        default="logs/hex_ground/Dec31_16-52-59_/model_6000.pt",
        help="Low-level policy checkpoint path",
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True, help="Teacher checkpoint path")
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
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    from legged_gym.utils import get_args as get_isaac_args
    isaac_args = get_isaac_args()
    for key, value in vars(isaac_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if args.camera_interval < 1:
        args.camera_interval = 1

    return args


def main():
    args = parse_args()
    if args.task != "hex_ground":
        raise ValueError("play_highlevel.py currently supports only --task hex_ground")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th.import_modules()

    if args.camera_show and args.headless:
        print("[PlayHigh] ⚠ camera_show requested but headless=True. Disabling.")
        args.camera_show = False

    if args.camera_show or args.camera_save:
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
    # Use clearer alias; implementation is identical to TerrainAdaptivePlanner.
    planner = th.HighLevelPlanner(
        affordance_channels=th.AFFORDANCE_CHANNELS,
        state_dim=th.STATE_DIM,
        goal_dim=th.GOAL_DIM,
    ).to(device)

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    planner.load_state_dict(state_dict)
    planner.eval()

    obs = env.reset()
    step_idx = 0
    deterministic = not args.stochastic
    camera_frame_idx = 0

    while True:
        with torch.no_grad():
            subgoal, intensity, _ = planner.get_action(
                obs["gt_affordance"],
                obs["state"],
                obs["goal"],
                obs["gt_difficulty"],
                deterministic=deterministic,
            )
        obs, _, _, _ = env.step(subgoal, intensity)

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
