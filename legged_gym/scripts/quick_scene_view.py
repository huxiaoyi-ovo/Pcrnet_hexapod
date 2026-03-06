#!/usr/bin/env python3
"""Quick scene viewer for registered s_* / e_* tasks."""

import argparse
import copy
import os
import sys
from typing import List, Tuple

import isaacgym  # noqa: F401
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args


def _parse_cli() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Quickly list/open scene envs registered as s_* / e_* tasks."
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Scene task name. Supports repeated flag or comma list, e.g. --task s_cylinder,s_step_field",
    )
    parser.add_argument("--all", action="store_true", help="Run all s_* / e_* scenes.")
    parser.add_argument("--list_scenes", action="store_true", help="Only print all available s_* / e_* scene names.")
    parser.add_argument("--steps", type=int, default=400, help="Steps to run per scene.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed into env config.")
    parser.add_argument("--headless", action="store_true", help="Disable viewer.")
    parser.add_argument("--print_every", type=int, default=50, help="Print interval in steps.")

    args, passthrough = parser.parse_known_args()
    args.steps = max(1, int(args.steps))
    args.num_envs = max(1, int(args.num_envs))
    args.print_every = max(0, int(args.print_every))
    return args, passthrough


def _scene_names() -> List[str]:
    return sorted(
        name
        for name in task_registry.task_classes.keys()
        if name.startswith("s_") or name.startswith("e_")
    )


def _expand_tasks(raw_items: List[str]) -> List[str]:
    tasks: List[str] = []
    for item in raw_items:
        for part in item.split(","):
            name = part.strip()
            if name:
                tasks.append(name)
    return tasks


def _build_isaac_args(passthrough: List[str]):
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0]] + passthrough
        args = get_args()
    finally:
        sys.argv = saved_argv
    return args


def _destroy_env(env) -> None:
    try:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
    except Exception:
        pass
    try:
        if getattr(env, "sim", None) is not None:
            env.gym.destroy_sim(env.sim)
    except Exception:
        pass


def _run_one_scene(task_name: str, quick_args: argparse.Namespace, base_args) -> bool:
    scene_args = copy.deepcopy(base_args)
    scene_args.task = task_name
    scene_num_envs = quick_args.num_envs
    if task_name == "s_avoid_basic":
        scene_num_envs = max(scene_num_envs, 3)
    scene_args.num_envs = scene_num_envs
    scene_args.seed = quick_args.seed
    scene_args.headless = bool(quick_args.headless)

    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.seed = int(quick_args.seed)
    if task_name == "s_avoid_basic":
        # Always preview all 3 curriculum stages for scene inspection:
        # env0->stage1, env1->stage2, env2->stage3 (cycled when num_envs<3).
        setattr(env_cfg.terrain, "avoid_preview_all_stages", True)
    env = None
    try:
        env, _ = task_registry.make_env(name=task_name, args=scene_args, env_cfg=env_cfg)
    except Exception as exc:
        print(f"[QuickScene][ERROR] task={task_name} failed before rollout: {exc}")
        return False

    print(f"\n[QuickScene] Running task={task_name} | num_envs={env.num_envs} | steps={quick_args.steps} | headless={scene_args.headless}")
    if task_name == "s_avoid_basic":
        print("[QuickScene] s_avoid_basic preview mode: showing stage1/2/3 together (no extra flag needed).")

    try:
        obs, _ = env.reset()
        _ = obs  # keep API compatibility
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

        for step in range(quick_args.steps):
            _, _, rewards, dones, _ = env.step(actions)
            if quick_args.print_every > 0 and (step == 0 or (step + 1) % quick_args.print_every == 0):
                rew_mean = float(rewards.mean().item()) if torch.is_tensor(rewards) else float(np.mean(rewards))
                done_ratio = float(dones.float().mean().item()) if torch.is_tensor(dones) else float(np.mean(dones))
                print(
                    f"[QuickScene] task={task_name} step={step + 1}/{quick_args.steps} "
                    f"reward_mean={rew_mean:.3f} done_ratio={done_ratio:.3f}"
                )

            if (not scene_args.headless) and getattr(env, "viewer", None) is not None:
                if env.gym.query_viewer_has_closed(env.viewer):
                    print(f"[QuickScene] Viewer closed early on task={task_name}.")
                    break
        return True
    except Exception as exc:
        print(f"[QuickScene][ERROR] task={task_name} rollout failed: {exc}")
        return False
    finally:
        if env is not None:
            _destroy_env(env)


def main() -> int:
    quick_args, passthrough = _parse_cli()
    all_scenes = _scene_names()

    if quick_args.list_scenes:
        print("[QuickScene] Available s_* / e_* scenes:")
        for name in all_scenes:
            print(name)
        if not quick_args.all and not quick_args.task:
            return 0

    selected_tasks: List[str]
    if quick_args.all:
        selected_tasks = list(all_scenes)
    else:
        selected_tasks = _expand_tasks(quick_args.task)

    if not selected_tasks:
        print("[QuickScene] No scene selected. Use --task <name> or --all.")
        print("[QuickScene] Tip: add --list_scenes to print all scene names.")
        return 1

    unknown = [name for name in selected_tasks if name not in task_registry.task_classes]
    if unknown:
        print("[QuickScene] Unknown task(s):", ", ".join(unknown))
        print("[QuickScene] Use --list_scenes to check valid scene names.")
        return 1

    if len(selected_tasks) > 1 and not quick_args.headless:
        print("[QuickScene] Multi-scene mode requires --headless. For viewer mode, run one --task at a time.")
        return 1

    isaac_args = _build_isaac_args(passthrough)

    failed: List[str] = []
    for task_name in selected_tasks:
        ok = _run_one_scene(task_name=task_name, quick_args=quick_args, base_args=isaac_args)
        if not ok:
            failed.append(task_name)

    if failed:
        print("\n[QuickScene] Done with failures:", ", ".join(failed))
        return 1

    print("\n[QuickScene] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
