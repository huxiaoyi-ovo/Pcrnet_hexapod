#!/usr/bin/env python3
"""Native static task-registry snapshots for s_avoid_clutter; no policy is run."""
import json
import subprocess
import sys
from pathlib import Path

from isaacgym import gymapi  # Isaac Gym must load before torch via legged_gym.

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "avoid_clutter_training_preview_effective_decisions"
IMAGE_DIR = Path("/home/artrc/图片")
SEED = 9173
STAGES = (1, 2, 3, 4)


def _segment_distances(start, end, centers):
    direction = end - start
    length_sq = float(direction @ direction)
    if length_sq < 1e-12:
        nearest = np.repeat(start[None, :], len(centers), axis=0)
    else:
        t = np.clip((centers - start) @ direction / length_sq, 0.0, 1.0)
        nearest = start[None, :] + t[:, None] * direction[None, :]
    return np.linalg.norm(centers - nearest, axis=1)


def _route_min_distance(route, centers):
    return min(float(_segment_distances(start, end, centers).min())
               for start, end in zip(route[:-1], route[1:]))


def _continuous_route(layout, generator):
    """Build a forward-only nominal geometric/dynamic reference through every band."""
    centers = np.asarray(layout.cylinders_xy, dtype=np.float64)
    route = [np.array([0.0, -1.6], dtype=np.float64)]
    previous_exit_y = -1.6
    for gaps, enter_y, exit_y in zip(layout.safe_intervals, layout.band_enter_y, layout.band_exit_y):
        current = route[-1]
        budget = float(generator.v_lat_eff) * max(
            (float(enter_y) - previous_exit_y) / float(layout.v_drive_nom) / float(layout.kappa)
            - float(generator.response_delay),
            0.0,
        )
        candidates = []
        for gap_lo, gap_hi in gaps:
            lo = max(float(gap_lo), float(current[0]) - budget)
            hi = min(float(gap_hi), float(current[0]) + budget)
            interior_lo = lo + float(generator.path_epsilon)
            interior_hi = hi - float(generator.path_epsilon)
            if interior_lo > interior_hi:
                continue
            target_x = min(max(float(current[0]), interior_lo), interior_hi)
            points = [np.array([target_x, float(enter_y)]), np.array([target_x, float(exit_y)])]
            clearance = _route_min_distance([current] + points, centers) - float(layout.combined_radius)
            if clearance > 1e-6:
                candidates.append((clearance, points))
        if not candidates:
            raise RuntimeError("no forward reachable safe-interval point clears every cylinder")
        _, selected = max(candidates, key=lambda item: (item[0], -abs(item[1][0][0] - current[0])))
        route.extend(selected)
        previous_exit_y = float(exit_y)
    route.append(np.array([route[-1][0], float(layout.exit_y)], dtype=np.float64))
    min_center_distance = _route_min_distance(route, centers)
    if min_center_distance - float(layout.combined_radius) <= 1e-6:
        raise RuntimeError("selected nominal route fails expanded-cylinder clearance")
    return route, min_center_distance


def _line_arrays(route, z=0.56):
    count = len(route) - 1
    vertices = np.empty((count, 2), dtype=gymapi.Vec3.dtype)
    colors = np.empty(count, dtype=gymapi.Vec3.dtype)
    for index, (start, end) in enumerate(zip(route[:-1], route[1:])):
        vertices[index][0] = (float(start[0]), float(start[1]), z)
        vertices[index][1] = (float(end[0]), float(end[1]), z)
        colors[index] = (1.0, 0.0, 0.0)
    return vertices, colors


def _git_state():
    def run(*args):
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    return {"head": run("git", "rev-parse", "HEAD"), "status_short": run("git", "status", "--short")}


def _select_decision_layout(env, env_id, stage):
    import torch

    env_ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    for attempt in range(1, 257):
        layout = env.s_avoid_clutter_layouts[env_id]
        mandatory = sum(item.mandatory for item in layout.band_decisions)
        choice = sum(item.choice for item in layout.band_decisions)
        effective = sum(item.effective for item in layout.band_decisions)
        qualifies = (stage <= 2 and mandatory >= 1) or (stage == 3 and choice >= 1) or (stage == 4 and effective >= 2)
        if layout.decision_episode and qualifies:
            return layout, attempt
        env.reset_idx(env_ids)
    raise RuntimeError(f"env {env_id} did not produce a qualifying decision episode in 256 resets")


def _trace_live_actor_positions(env, env_id, layout):
    env.gym.refresh_actor_root_state_tensor(env.sim)
    active = env.s_avoid_active[env_id].detach().cpu().numpy().astype(bool)
    actor_indices = env.s_avoid_actor_indices[env_id, active].detach().cpu().numpy().astype(np.int64)
    live = env.all_root_states[actor_indices, :3].detach().cpu().numpy().copy()
    expected = env.s_avoid_pos_world[env_id, active, :3].detach().cpu().numpy().copy()
    if live.shape[0] != int(layout.cylinders_xy.shape[0]):
        raise RuntimeError("active actor count differs from the selected layout")
    exact = bool(np.array_equal(live, expected))
    max_abs_error = float(np.max(np.abs(live - expected))) if live.size else 0.0
    if max_abs_error > 1e-6:
        raise RuntimeError(f"live actor positions disagree with runtime metadata: {max_abs_error:.3e}")
    return live, {"array_equal": exact, "max_abs_error_m": max_abs_error}


def _camera_from_world_bbox(points_world):
    low = points_world.min(axis=0)
    high = points_world.max(axis=0)
    center = 0.5 * (low + high)
    span = max(float(high[0] - low[0]), float(high[1] - low[1]), 2.0)
    camera = np.array([center[0] - .38 * span, center[1] - .72 * span, max(5.0, .95 * span)], dtype=np.float64)
    target = np.array([center[0], center[1], .20], dtype=np.float64)
    return low, high, camera, target


def main():
    import torch

    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import get_args

    args = get_args()
    args.physics_engine = gymapi.SIM_PHYSX
    args.sim_device = "cpu"
    args.sim_device_id = 0
    args.rl_device = "cpu"
    args.use_gpu = False
    args.use_gpu_pipeline = False
    args.headless = False
    env_cfg, _ = task_registry.get_cfgs("s_avoid_clutter")
    env_cfg.seed = SEED
    env_cfg.env.num_envs = 4
    env_cfg.terrain.avoid_preview_all_stages = True
    env_cfg.terrain.avoid_seed = SEED
    env, _ = task_registry.make_env("s_avoid_clutter", args=args, env_cfg=env_cfg)
    gym, viewer = env.gym, env.viewer
    if viewer is None:
        raise RuntimeError("native viewer was not created")
    try:
        env.reset_idx(torch.arange(env.num_envs, device=env.device, dtype=torch.long))
        OUT.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        records = []
        for env_id, stage in enumerate(STAGES):
            layout, selection_attempts = _select_decision_layout(env, env_id, stage)
            if int(layout.stage) != stage:
                raise RuntimeError(f"env {env_id} resolved stage {layout.stage}, expected {stage}")
            route, min_center_distance = _continuous_route(layout, env.s_avoid_clutter_generator)
            live_positions, actor_trace = _trace_live_actor_positions(env, env_id, layout)
            origin = env.env_origins[env_id].detach().cpu().numpy().astype(np.float64)
            route_array = np.asarray(route, dtype=np.float64)
            route_world = route_array + origin[:2]
            env_origin = gym.get_env_origin(env.envs[env_id])
            gym_env_origin = np.array([env_origin.x, env_origin.y, env_origin.z], dtype=np.float64)
            route_gym_env = route_world - gym_env_origin[:2]
            world_points = np.vstack((live_positions, np.column_stack((route_world, np.zeros(len(route_array))))))
            bbox_low, bbox_high, camera, target = _camera_from_world_bbox(world_points)
            decisions = [{
                "band_index": index,
                "d_min_m": float(item.d_min_m), "d_max_m": float(item.d_max_m),
                "mandatory": bool(item.mandatory), "choice": bool(item.choice), "effective": bool(item.effective),
            } for index, item in enumerate(layout.band_decisions)]
            image_path = IMAGE_DIR / f"pcr_avoid_{stage}_effective_decisions_20260910.png"
            records.append({
                "stage": stage, "env_id": env_id,
                "source_episode": int(env.s_avoid_env_episode_count[env_id].item()) - 1,
                "selection_attempts": selection_attempts,
                "decision_episode": bool(layout.decision_episode),
                "band_count": len(layout.band_exit_y),
                "mandatory_band_count": sum(item["mandatory"] for item in decisions),
                "choice_band_count": sum(item["choice"] for item in decisions),
                "effective_band_count": sum(item["effective"] for item in decisions),
                "band_decisions": decisions,
                "cylinder_count": int(layout.cylinders_xy.shape[0]),
                "v_drive_nom_mps": float(layout.v_drive_nom),
                "generator_margin_m": float(layout.m_clear),
                "combined_radius_m": float(layout.combined_radius),
                "actual_min_route_center_distance_m": min_center_distance,
                "actual_min_route_clearance_m": min_center_distance - float(layout.combined_radius),
                "actor_position_trace": actor_trace,
                "route_semantics": "nominal geometric/dynamic reference, not a learned policy trajectory",
                "route_local_xy_m": route_array.tolist(),
                "route_world_xy_m": route_world.tolist(),
                "gym_env_origin_world_m": gym_env_origin.tolist(),
                "route_gym_env_xy_m": route_gym_env.tolist(),
                "camera_world_m": camera.tolist(),
                "target_world_m": target.tolist(),
                "bbox_world_m": {"min": bbox_low.tolist(), "max": bbox_high.tolist()},
                "screenshot": str(image_path),
            })
        for record in records:
            gym.clear_lines(viewer)
            vertices, colors = _line_arrays(np.asarray(record["route_gym_env_xy_m"], dtype=np.float64), z=.72)
            gym.add_lines(viewer, env.envs[record["env_id"]], len(vertices), vertices, colors)
            gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*record["camera_world_m"]),
                                      gymapi.Vec3(*record["target_world_m"]))
            env.render()
            gym.write_viewer_image_to_file(viewer, record["screenshot"])
        metadata = {
            "task": "s_avoid_clutter", "seed": SEED,
            "sim": {"physics": "PhysX", "sim_device": "cpu", "use_gpu": False,
                    "use_gpu_pipeline": False, "graphics_device_id": 0},
            "event": "effective_decision_static_preview",
            "visual_selection_only": "Static effective-decision layouts for visual review; not a training, policy, or performance result.",
            "events": [{"stage": item["stage"], "band_count": item["band_count"], "mandatory_band_count": item["mandatory_band_count"], "choice_band_count": item["choice_band_count"], "effective_band_count": item["effective_band_count"], "transitions": item["band_decisions"]} for item in records],
            "git": _git_state(), "records": records,
        }
        metadata_path = OUT / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"metadata": str(metadata_path), "images": [item["screenshot"] for item in records]}, indent=2))
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
