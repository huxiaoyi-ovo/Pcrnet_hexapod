#!/usr/bin/env python3
"""Capture real Isaac Gym top-down renders of held-out stage-4 obstacle layouts."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

_PLAY_HIGHLEVEL = None


def _load_play_highlevel():
    global _PLAY_HIGHLEVEL
    if _PLAY_HIGHLEVEL is not None:
        return _PLAY_HIGHLEVEL
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        from legged_gym.scripts import play_highlevel as ph

        _PLAY_HIGHLEVEL = ph
        return ph
    finally:
        sys.argv = old_argv


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), rgb)
        return
    except Exception:
        pass
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return
    except Exception as exc:
        raise RuntimeError(f"Failed to write image: {path}") from exc


def _flatten_low_frequency_lighting(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not bool(getattr(args, "flatten_lighting", True)):
        return rgb
    strength = float(getattr(args, "flatten_strength", 0.75))
    if strength <= 0.0:
        return rgb
    img = rgb.astype(np.float32) / 255.0
    gray = img.mean(axis=2)
    try:
        import cv2

        sigma = max(1.0, float(getattr(args, "flatten_sigma", 95.0)))
        illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    except Exception:
        illum = gray
    illum_mean = float(np.mean(illum))
    gain = illum_mean / np.maximum(illum, 1e-3)
    gain = 1.0 + strength * (gain - 1.0)
    corrected = img * gain[:, :, None]
    return np.clip(np.round(corrected * 255.0), 0, 255).astype(np.uint8)


def _apply_display_tone(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    exposure = float(getattr(args, "display_exposure", 1.0))
    gamma = float(getattr(args, "display_gamma", 1.0))
    if abs(exposure - 1.0) < 1e-6 and abs(gamma - 1.0) < 1e-6:
        return rgb
    img = rgb.astype(np.float32) / 255.0
    img = np.clip(img * max(exposure, 0.0), 0.0, 1.0)
    if gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        img = np.power(img, 1.0 / gamma)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _make_play_args(seed: int, args: argparse.Namespace):
    ph = _load_play_highlevel()
    raw_argv = [
        "export_heldout_layout_isaac_topdown.py",
        "--task",
        "s_pcr_line_avoid_basic",
        "--mode",
        "teacher",
        "--skill",
        "moe",
        "--num_envs",
        "1",
        "--seed",
        str(int(seed)),
        "--low_level_ckpt",
        str(args.low_level_ckpt),
        "--force_cmd",
        "0.0",
        "0.0",
        "0.0",
        "--avoid_stage_override",
        "4",
        "--eval_layout",
        "heldout_irregular_rows",
        "--pcr_line_target_speed",
        "0.60",
    ]
    old_argv = sys.argv
    try:
        sys.argv = raw_argv
        return ph.parse_args()
    finally:
        sys.argv = old_argv


def _move_robot_out_of_frame(env) -> None:
    env_impl = getattr(env, "env", None)
    if env_impl is None or not hasattr(env_impl, "root_states"):
        return
    env_ids = torch.tensor([0], device=env_impl.device, dtype=torch.long)
    env_impl.root_states[0, 0] = float(getattr(env_impl, "env_origins", torch.zeros(1, 3, device=env_impl.device))[0, 0].item())
    env_impl.root_states[0, 1] = -50.0
    env_impl.root_states[0, 2] = 0.60
    env_impl.root_states[0, 7:13] = 0.0
    if hasattr(env_impl, "_sync_robot_root_states"):
        env_impl._sync_robot_root_states(env_ids)


def _set_even_lighting(gym, sim, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "even_lighting", True)):
        return
    try:
        intensity = float(getattr(args, "light_intensity", 0.28))
        ambient = float(getattr(args, "ambient_intensity", 0.78))
        gym.set_light_parameters(
            sim,
            0,
            gymapi.Vec3(intensity, intensity, intensity),
            gymapi.Vec3(ambient, ambient, ambient),
            gymapi.Vec3(0.0, 0.0, -1.0),
        )
    except Exception as exc:
        print(f"[IsaacTopdown] warning: failed to set even lighting: {exc}")


def _capture_seed(seed: int, args: argparse.Namespace) -> Path:
    ph = _load_play_highlevel()
    play_args = _make_play_args(seed, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        runtime = ph.build_play_runtime_for_eval(play_args, device=device)
    finally:
        sys.argv = old_argv
    env = runtime.env
    env_impl = env.env

    if hasattr(env_impl, "debug_viz"):
        env_impl.debug_viz = False
    if hasattr(env_impl, "foot_traj_viz"):
        env_impl.foot_traj_viz = False
    _move_robot_out_of_frame(env)

    gym = env_impl.gym
    sim = env_impl.sim
    env_handle = env_impl.envs[0]
    _set_even_lighting(gym, sim, args)

    camera_props = gymapi.CameraProperties()
    camera_props.width = int(args.width)
    camera_props.height = int(args.height)
    camera_props.horizontal_fov = float(args.horizontal_fov)
    camera_props.near_plane = 0.05
    camera_props.far_plane = max(80.0, float(args.camera_height) + 20.0)
    camera_props.enable_tensors = False
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    center_y = float(args.center_y)
    gym.set_camera_location(
        camera_handle,
        env_handle,
        gymapi.Vec3(0.0, center_y, float(args.camera_height)),
        gymapi.Vec3(0.0, center_y + float(args.lookahead_y), 0.0),
    )

    for _ in range(3):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

    raw = gym.get_camera_image(sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
    rgba = np.asarray(raw, dtype=np.uint8).reshape((int(args.height), int(args.width), 4))
    rgb = rgba[:, :, :3].copy()
    rgb = _flatten_low_frequency_lighting(rgb, args)
    rgb = _apply_display_tone(rgb, args)

    out_path = Path(args.output_dir) / f"heldout_stage4_isaac_topdown_seed{int(seed)}.png"
    _write_rgb(out_path, rgb)
    print(f"[IsaacTopdown] seed={int(seed)} -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="agents/final_paper_outputs_v3/heldout_layout_isaac_topdown")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7001, 7002, 7003, 7004])
    parser.add_argument("--low_level_ckpt", type=str, default="agents/low_level_best.pt")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=2400)
    parser.add_argument("--camera_height", type=float, default=25.0)
    parser.add_argument("--center_y", type=float, default=11.4)
    parser.add_argument("--lookahead_y", type=float, default=0.05)
    parser.add_argument("--horizontal_fov", type=float, default=38.0)
    parser.add_argument("--even_lighting", action="store_true", default=True)
    parser.add_argument("--no_even_lighting", dest="even_lighting", action="store_false")
    parser.add_argument("--light_intensity", type=float, default=0.0)
    parser.add_argument("--ambient_intensity", type=float, default=1.15)
    parser.add_argument("--flatten_lighting", action="store_true", default=True)
    parser.add_argument("--no_flatten_lighting", dest="flatten_lighting", action="store_false")
    parser.add_argument("--flatten_sigma", type=float, default=95.0)
    parser.add_argument("--flatten_strength", type=float, default=0.75)
    parser.add_argument("--display_exposure", type=float, default=1.35)
    parser.add_argument("--display_gamma", type=float, default=1.10)
    parser.add_argument("--single_seed_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not bool(args.single_seed_worker) and len(args.seeds) > 1:
        script = Path(__file__).resolve()
        for seed in args.seeds:
            cmd = [
                sys.executable,
                str(script),
                "--single_seed_worker",
                "--seeds",
                str(int(seed)),
                "--output_dir",
                str(args.output_dir),
                "--low_level_ckpt",
                str(args.low_level_ckpt),
                "--width",
                str(int(args.width)),
                "--height",
                str(int(args.height)),
                "--camera_height",
                str(float(args.camera_height)),
                "--center_y",
                str(float(args.center_y)),
                "--lookahead_y",
                str(float(args.lookahead_y)),
                "--horizontal_fov",
                str(float(args.horizontal_fov)),
            ]
            if bool(args.even_lighting):
                cmd.append("--even_lighting")
            else:
                cmd.append("--no_even_lighting")
            cmd += [
                "--light_intensity",
                str(float(args.light_intensity)),
                "--ambient_intensity",
                str(float(args.ambient_intensity)),
                "--flatten_sigma",
                str(float(args.flatten_sigma)),
                "--flatten_strength",
                str(float(args.flatten_strength)),
                "--display_exposure",
                str(float(args.display_exposure)),
                "--display_gamma",
                str(float(args.display_gamma)),
            ]
            if bool(args.flatten_lighting):
                cmd.append("--flatten_lighting")
            else:
                cmd.append("--no_flatten_lighting")
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return

    for seed in args.seeds:
        _capture_seed(int(seed), args)


if __name__ == "__main__":
    main()
