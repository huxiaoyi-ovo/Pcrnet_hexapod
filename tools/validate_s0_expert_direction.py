#!/usr/bin/env python3
"""
Validate S0 expert direction behavior with user-provided target coordinates.

This script is standalone (no Isaac Gym runtime). It checks:
1) world->body coordinate conversion under project contract
2) bearing alpha = atan2(x_right, y_forward)
3) expert output sign for cmd_y / omega

Project contract used here:
- heading=0 means world +Y is body forward
- body frame is (x_right, y_forward)
- positive omega is CCW (left turn)
"""

import argparse
import importlib.util
import math
import pathlib
from typing import List, Tuple

import torch


def load_expert_func():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    expert_file = repo_root / "legged_gym" / "envs" / "hex_v4" / "expert_s0_follow.py"
    spec = importlib.util.spec_from_file_location("expert_s0_follow_module", str(expert_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {expert_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "compute_s0_follow_expert_cmd", None)
    if fn is None:
        raise RuntimeError("compute_s0_follow_expert_cmd not found in expert_s0_follow.py")
    return fn


def parse_targets(raw: str) -> List[Tuple[float, float]]:
    items = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        items.append((float(x_str), float(y_str)))
    if not items:
        raise ValueError("No valid targets parsed from --targets")
    return items


def world_to_body_contract(delta_world: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    x_right = cos_h * delta_world[:, 0] + sin_h * delta_world[:, 1]
    y_forward = -sin_h * delta_world[:, 0] + cos_h * delta_world[:, 1]
    return torch.stack([x_right, y_forward], dim=1)


def side_label(x_right: float, eps: float = 1e-6) -> str:
    if x_right > eps:
        return "target_on_right"
    if x_right < -eps:
        return "target_on_left"
    return "target_on_centerline"


def sign(v: float, eps: float = 1e-6) -> int:
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def sign_text(s: int) -> str:
    return {1: "+", 0: "0", -1: "-"}[s]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=str,
        default="-1.0,2.0;1.0,2.0;0.0,2.0;0.0,-2.0;2.0,0.0;-2.0,0.0",
        help="Semicolon-separated world targets: x,y;x,y;...",
    )
    parser.add_argument("--robot-x", type=float, default=0.0)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--d-des", type=float, default=1.0)
    parser.add_argument("--kp-along", type=float, default=0.8)
    parser.add_argument("--k-yaw", type=float, default=1.0)
    parser.add_argument("--v-along-max", type=float, default=1.5)
    parser.add_argument("--w-max", type=float, default=0.8)
    parser.add_argument("--omega-deadzone-rad", type=float, default=0.05)
    args = parser.parse_args()
    compute_s0_follow_expert_cmd = load_expert_func()

    targets = parse_targets(args.targets)
    n = len(targets)

    robot_pos = torch.tensor([[args.robot_x, args.robot_y]] * n, dtype=torch.float32)
    robot_heading = torch.full((n,), math.radians(args.heading_deg), dtype=torch.float32)
    target_world = torch.tensor(targets, dtype=torch.float32)

    delta_world = target_world - robot_pos
    body = world_to_body_contract(delta_world, robot_heading)
    alpha = torch.atan2(body[:, 0], body[:, 1])

    cmd = compute_s0_follow_expert_cmd(
        robot_pos_world_xy=robot_pos,
        robot_heading=robot_heading,
        target_world_xy=target_world,
        target_vel_world_xy=None,
        target_heading=None,
        cmd_scale=[1.4, 1.4, 1.4],
        d_des=float(args.d_des),
        kp_along=float(args.kp_along),
        k_yaw=float(args.k_yaw),
        v_along_max=float(args.v_along_max),
        w_max=float(args.w_max),
        omega_deadzone_rad=float(args.omega_deadzone_rad),
    )

    print("=== S0 Expert Direction Validation ===")
    print(
        f"robot_world=({args.robot_x:.3f},{args.robot_y:.3f}) "
        f"heading_deg={args.heading_deg:.2f} d_des={args.d_des:.3f} "
        f"kp_along={args.kp_along:.3f} k_yaw={args.k_yaw:.3f}"
    )
    print("contract: heading=0 => +Y forward, +X right; omega>0 => turn left(CCW)")
    print("")
    print(
        "idx | target_world(x,y) | body(x_right,y_forward) | alpha(rad) | side | "
        "cmd(vx,vy,w) | sign(alpha/w) | dir_match"
    )

    for i in range(n):
        tx, ty = target_world[i].tolist()
        bx, by = body[i].tolist()
        a = float(alpha[i].item())
        vx, vy, w = cmd[i].tolist()
        s_alpha = sign(a)
        s_w = sign(float(w))
        # With current contract (+omega is left turn), turn-toward-target needs omega = -k*alpha.
        dir_match = (s_alpha == 0 and s_w == 0) or (s_alpha == -s_w)
        print(
            f"{i:>3d} | ({tx:+.3f},{ty:+.3f}) | ({bx:+.3f},{by:+.3f}) | {a:+.3f} | "
            f"{side_label(bx):>18s} | ({vx:+.3f},{vy:+.3f},{w:+.3f}) | "
            f"{sign_text(s_alpha)}/{sign_text(s_w)} | {int(dir_match)}"
        )

    print("")
    print("Interpretation:")
    print("- dir_match=1 for all non-deadzone samples => expert direction is self-consistent.")
    print("- If still tracking opposite in play, issue is likely in play/runtime wiring or frame usage.")


if __name__ == "__main__":
    main()
