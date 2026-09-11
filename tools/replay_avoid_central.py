#!/usr/bin/env python3
"""One-pass u_x=0 counterfactual over the 45 recorded Central layouts.

This is a diagnostic runner.  It reuses the production reset path by replacing
only this process's clutter generator for its first 45 environments.  The
qualification JSON has no historical root, dof, RNG, or low-level state, so it
does not claim bitwise replay of the original episodes.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
FORMAL_SEED = 9174
FORMAL_ENVS = 64
CENTRAL_COUNT = 45
EXPECTED_LOW_LEVEL_SHA256 = "75dfd7aae52b20f08ca9f654c37b2e20ba1e33f8b953737437ad774d122d1481"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _layout_from_snapshot(snapshot):
    from legged_gym.avoid_clutter import BandDecision, ClutterLayout
    return ClutterLayout(
        stage=int(snapshot["stage"]),
        kind=str(snapshot["kind"]),
        decision_episode=bool(snapshot["decision_episode"]),
        cylinders_xy=np.asarray(snapshot["cylinders_xy"], dtype=np.float32),
        band_y=np.asarray(snapshot["band_y"], dtype=np.float32),
        safe_intervals=tuple(tuple(tuple(map(float, interval)) for interval in band) for band in snapshot["safe_intervals"]),
        reachable_intervals=tuple(tuple(tuple(map(float, interval)) for interval in band) for band in snapshot["reachable_intervals"]),
        exit_y=float(snapshot["exit_y"]),
        v_drive_nom=float(snapshot["v_drive_nom"]),
        min_forward_assumption=float(snapshot["min_forward_assumption"]),
        combined_radius=float(snapshot["combined_radius"]),
        m_clear=float(snapshot["m_clear"]),
        kappa=float(snapshot["kappa"]),
        band_enter_y=np.asarray(snapshot["band_enter_y"], dtype=np.float32),
        band_exit_y=np.asarray(snapshot["band_exit_y"], dtype=np.float32),
        coverage=np.asarray(snapshot["coverage"], dtype=np.float32),
        recovery_paths=tuple(
            tuple(tuple(tuple(map(float, point)) for point in path) for path in band)
            for band in snapshot["recovery_paths"]
        ),
        band_decisions=tuple(BandDecision(**item) for item in snapshot.get("band_decisions", ())),
        speed_draws=int(snapshot.get("speed_draws", 1)),
        layout_attempts=int(snapshot.get("layout_attempts", 1)),
        structure_candidates=int(snapshot.get("structure_candidates", 1)),
    )


def _load_central_layouts(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = source.get("central_episodes")
    if not isinstance(rows, list) or len(rows) != CENTRAL_COUNT:
        raise RuntimeError(f"expected exactly {CENTRAL_COUNT} central rows, got {0 if not isinstance(rows, list) else len(rows)}")
    layouts = []
    for index, row in enumerate(rows):
        layout = row.get("layout")
        if row.get("kind") != "central" or not isinstance(layout, dict):
            raise RuntimeError(f"central row {index} has invalid kind/layout")
        if int(layout.get("stage", -1)) != 1 or bool(layout.get("decision_episode")):
            raise RuntimeError(f"central row {index} is not a stage-1 non-decision layout")
        layouts.append(_layout_from_snapshot(layout))
    return source, layouts


def _finite(torch, name, tensor):
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"non-finite {name}")


def _local_root(impl, env_id):
    root = impl.root_states[env_id].detach()
    origin = impl.env_origins[env_id].detach()
    return {
        "root_xyz_world": root[:3].cpu().tolist(),
        "root_xy_env_local": (root[:2] - origin[:2]).cpu().tolist(),
        "root_quat_xyzw": root[3:7].cpu().tolist(),
        "root_lin_vel_world": root[7:10].cpu().tolist(),
        "root_ang_vel_world": root[10:13].cpu().tolist(),
    }


def _body_names(impl, env_id):
    try:
        return list(impl.gym.get_actor_rigid_body_names(impl.envs[env_id], impl.actor_handles[env_id]))
    except Exception:
        return None


def _contact_evidence(torch, impl, env_id, layout, body_names):
    threshold = float(getattr(impl.cfg.terrain, "collision_force_threshold", 1.0))
    groups = {}
    for group, indices in (("termination", impl.termination_contact_indices), ("penalised", impl.penalised_contact_indices)):
        for value in indices.detach().cpu().tolist():
            groups.setdefault(int(value), []).append(group)
    try:
        impl.gym.refresh_rigid_body_state_tensor(impl.sim)
        rigid_body_available = True
    except Exception:
        rigid_body_available = False
    origin = impl.env_origins[env_id, :2].detach().cpu().numpy()
    items = []
    for index, membership in sorted(groups.items()):
        force = impl.contact_forces[env_id, index].detach().cpu().numpy()
        force_norm = float(np.linalg.norm(force))
        if force_norm <= threshold:
            continue
        item = {
            "rigid_body_index": index,
            "rigid_body_name": None if body_names is None or index >= len(body_names) else body_names[index],
            "termination_groups": membership,
            "force_xyz_world": force.tolist(),
            "force_norm": force_norm,
            "threshold": threshold,
        }
        if rigid_body_available and hasattr(impl, "rb_states") and index < impl.rb_states.shape[1]:
            position = impl.rb_states[env_id, index, :3].detach().cpu().numpy()
            item["link_position_world_not_contact_point"] = position.tolist()
            local = position[:2] - origin
            if len(layout.cylinders_xy):
                distances = np.linalg.norm(layout.cylinders_xy - local[None, :], axis=1)
                nearest = int(np.argmin(distances))
                item["geometric_nearest_cylinder_not_contact_pair"] = {
                    "slot": nearest,
                    "distance_m": float(distances[nearest]),
                    "center_xy_env_local": layout.cylinders_xy[nearest].tolist(),
                }
        items.append(item)
    return {"api": "refreshed_net_contact_force_tensor", "items": items, "rigid_body_position_available": rigid_body_available}


def _validate_live_layout(torch, impl, env_id, expected):
    actual = impl.s_avoid_clutter_layouts[env_id]
    if actual is not expected or int(impl.s_avoid_stage_per_env[env_id].item()) != 1:
        raise RuntimeError(f"env {env_id} reset did not retain mapped stage-1 layout")
    checks = {
        "exit_y": float(impl.s_avoid_exit_y[env_id].item()),
        "v_drive_nom": float(impl.s_avoid_v_drive_nom[env_id].item()),
        "band_count": int(impl.s_avoid_band_count[env_id].item()),
        "decision_episode": bool(impl.s_avoid_decision_episode[env_id].item()),
    }
    expected_values = {"exit_y": expected.exit_y, "v_drive_nom": expected.v_drive_nom, "band_count": len(expected.band_exit_y), "decision_episode": expected.decision_episode}
    for key, value in expected_values.items():
        if isinstance(value, float):
            if not math.isclose(checks[key], value, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(f"env {env_id} metadata mismatch for {key}")
        elif checks[key] != value:
            raise RuntimeError(f"env {env_id} metadata mismatch for {key}")
    impl.gym.refresh_actor_root_state_tensor(impl.sim)
    actor_indices = impl.s_avoid_actor_indices[env_id].long()
    live = impl.all_root_states[actor_indices, :3]
    expected_world = torch.as_tensor(expected.cylinders_xy, device=impl.device, dtype=impl.s_avoid_pos_world.dtype)
    expected_world = torch.cat((expected_world, torch.zeros((expected_world.shape[0], 1), device=impl.device, dtype=expected_world.dtype)), dim=1)
    expected_world += impl.env_origins[env_id, :3]
    active = impl.s_avoid_active[env_id]
    if not bool(torch.allclose(live[active], expected_world, rtol=0.0, atol=1e-5)):
        raise RuntimeError(f"env {env_id} live actor coordinates do not match reset layout")
    local_error = live[active, :2] - impl.env_origins[env_id, :2] - torch.as_tensor(expected.cylinders_xy, device=impl.device, dtype=live.dtype)
    return {
        "metadata": checks, "active_cylinders": int(active.sum().item()), "actor_coordinates_verified": True,
        "local_xy_roundtrip_max_abs_error_m": float(local_error.abs().max().item()),
    }


def _run(parsed):
    # Isaac must be imported before Torch in the real diagnostic process.
    from isaacgym import gymapi
    import torch

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TOOLS))
    # train_highlevel's Isaac parser must not consume diagnostic arguments.
    sys.argv = [sys.argv[0]]
    from qualify_avoid_clutter import (
        FORMAL_ITERATION, _assert_checkpoint_finite, _checkpoint_iteration, _runtime_args,
        _validate_checkpoint_metadata,
    )
    from legged_gym.pcr_observation import AVOID_ACTOR_MASK_INDICES
    from legged_gym.pcr_policy_contract import checkpoint_cmd_policy_kwargs
    from legged_gym.scripts import train_highlevel as highlevel

    if parsed.seed != FORMAL_SEED or parsed.num_envs != FORMAL_ENVS:
        raise RuntimeError("counterfactual is fixed to seed 9174 and 64 environments")
    if parsed.max_steps < 500:
        raise RuntimeError("max_steps must include the original 50 s cap")
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual requires CUDA mapped as cuda:0")
    qualification, layouts = _load_central_layouts(parsed.qualification)
    checkpoint_path = Path(parsed.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cuda:0")
    _assert_checkpoint_finite(torch, checkpoint)
    iteration = _checkpoint_iteration(checkpoint)
    if checkpoint_path.name != "model_200.pt" or iteration != FORMAL_ITERATION:
        raise RuntimeError("counterfactual requires formal model_200.pt at iteration 200")
    low_level_path = ROOT / "agents" / "low_level_best.pt"
    low_level_sha = _sha256(low_level_path)
    if low_level_sha != EXPECTED_LOW_LEVEL_SHA256:
        raise RuntimeError("unexpected low-level checkpoint SHA256")
    torch.manual_seed(parsed.seed)
    torch.cuda.manual_seed_all(parsed.seed)
    highlevel.import_modules()
    cfg, _ = highlevel.task_registry.get_cfgs("s_avoid_clutter")
    cfg.seed = parsed.seed
    cfg.terrain.avoid_seed = parsed.seed
    cfg.terrain.avoid_stage12_success_threshold = 1.01
    wrapper = None
    try:
        wrapper = highlevel.HierarchicalHexapodEnv(_runtime_args(ROOT, gymapi, parsed), torch.device("cuda:0"), env_cfg=cfg)
        impl = wrapper.env
        original_generate = impl.s_avoid_clutter_generator.generate

        def mapped_generate(stage, seed, env_id, episode):
            if env_id < CENTRAL_COUNT:
                if int(stage) != 1:
                    raise RuntimeError(f"mapped central env {env_id} changed stage to {stage}")
                return layouts[env_id]
            return original_generate(stage, seed, env_id, episode)

        impl.s_avoid_clutter_generator.generate = mapped_generate
        obs = wrapper.reset()
        layout_validations = [_validate_live_layout(torch, impl, env_id, layout) for env_id, layout in enumerate(layouts)]
        policy_kwargs = checkpoint_cmd_policy_kwargs(checkpoint, tuple(float(item) for item in wrapper.post_processor.max_cmd.detach().cpu()))
        checkpoint_meta = _validate_checkpoint_metadata(checkpoint, policy_kwargs, AVOID_ACTOR_MASK_INDICES)
        actor = highlevel.CmdVelExpert(affordance_channels=2, state_dim=14, goal_dim=2, **policy_kwargs).to(wrapper.device)
        actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
        actor.eval()
        body_names = _body_names(impl, 0)
        contexts = [{
            "env_id": env_id,
            "layout": qualification["central_episodes"][env_id]["layout"],
            "initial_pre_reset": {**_local_root(impl, env_id), "dof_pos": impl.dof_pos[env_id].detach().cpu().tolist(), "dof_vel": impl.dof_vel[env_id].detach().cpu().tolist()},
            "layout_validation": layout_validations[env_id],
            "trace": [], "commands": [], "terminal": None,
        } for env_id, layout in enumerate(layouts)]
        original_reset_idx = impl.reset_idx

        def capture_terminal_reset(env_ids):
            for env_id in env_ids.detach().cpu().tolist():
                if env_id < CENTRAL_COUNT and contexts[env_id]["terminal"] is None:
                    contexts[env_id]["terminal"] = {
                        **_local_root(impl, env_id),
                        "dof_pos": impl.dof_pos[env_id].detach().cpu().tolist(),
                        "dof_vel": impl.dof_vel[env_id].detach().cpu().tolist(),
                        "contact_link_forces": _contact_evidence(torch, impl, env_id, layouts[env_id], body_names),
                    }
            return original_reset_idx(env_ids)

        impl.reset_idx = capture_terminal_reset
        try:
            for step in range(parsed.max_steps):
                with torch.no_grad():
                    policy_action, _ = actor.get_action(obs["local_map_2ch"], obs["state"], obs["goal"], obs["actor_difficulty"], deterministic=True)
                _finite(torch, "policy_action", policy_action)
                forced_action = policy_action.clone()
                forced_action[:, 0] = 0.0
                active_ids = [env_id for env_id, context in enumerate(contexts) if context["terminal"] is None]
                for env_id in active_ids:
                    position = _local_root(impl, env_id)["root_xy_env_local"]
                    contexts[env_id]["trace"].append({"step": step, "x_right": position[0], "y_forward": position[1]})
                next_obs, reward, done, info = wrapper.step(forced_action)
                _finite(torch, "reward", reward)
                raw = info["post_info"]["cmd_raw"]
                executed = info["post_info"]["cmd_exec_mean"]
                _finite(torch, "cmd_raw", raw)
                _finite(torch, "cmd_exec_mean", executed)
                for env_id in active_ids:
                    raw_x = float(raw[env_id, 0].detach().cpu().item())
                    executed_x = float(executed[env_id, 0].detach().cpu().item())
                    raw_y = float(raw[env_id, 1].detach().cpu().item())
                    if abs(raw_x) > 1e-7 or abs(executed_x) > 1e-7:
                        raise RuntimeError(f"u_x=0 override violated for env {env_id} at step {step}")
                    if not math.isclose(raw_y, layouts[env_id].v_drive_nom, rel_tol=0.0, abs_tol=1e-6):
                        raise RuntimeError(f"nominal forward request changed for env {env_id} at step {step}")
                    contexts[env_id]["commands"].append({
                        "step": step,
                        "counterfactual_state_shadow_policy_u_x": float(policy_action[env_id, 0].detach().cpu().item()),
                        "forced_u_x": float(forced_action[env_id, 0].detach().cpu().item()),
                        "raw_u_x": raw_x, "executed_u_x": executed_x,
                        "raw_u_y": raw_y, "executed_u_y": float(executed[env_id, 1].detach().cpu().item()),
                        "layout_nominal_u_y": float(layouts[env_id].v_drive_nom),
                    })
                for env_id in done.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                    if env_id in active_ids and contexts[env_id]["terminal"] is not None:
                        contexts[env_id]["terminal"]["terminal_masks_wrapper"] = {
                            "physical": bool(info["terminal_physical"][env_id].item()),
                            "envelope": bool(info["terminal_envelope"][env_id].item()),
                            "fall": bool(info["terminal_fall"][env_id].item()),
                            "timeout": bool(info["timeout"][env_id].item()),
                            "success": bool(info["success_mask"][env_id].item()),
                        }
                obs = next_obs
                if (step + 1) % 20 == 0:
                    print(json.dumps({"progress_steps": step + 1, "terminal_rows": sum(context["terminal"] is not None for context in contexts)}), flush=True)
                if all(context["terminal"] is not None for context in contexts):
                    break
        finally:
            impl.reset_idx = original_reset_idx
        rows = []
        for context in contexts:
            terminal = context["terminal"]
            rows.append({
                **context,
                "terminal_recorded": terminal is not None,
                "limit_reached_without_terminal": terminal is None,
            })
        summary = {"success": 0, "physical": 0, "envelope": 0, "fall": 0, "timeout": 0, "other": 0, "not_terminal_by_limit": 0}
        for row in rows:
            masks = (row.get("terminal") or {}).get("terminal_masks_wrapper", {})
            if not row["terminal_recorded"]:
                summary["not_terminal_by_limit"] += 1
            elif masks.get("physical"):
                summary["physical"] += 1
            elif masks.get("envelope"):
                summary["envelope"] += 1
            elif masks.get("fall"):
                summary["fall"] += 1
            elif masks.get("success"):
                summary["success"] += 1
            elif masks.get("timeout"):
                summary["timeout"] += 1
            else:
                summary["other"] += 1
        return {
            "status": "completed" if all(row["terminal_recorded"] for row in rows) else "incomplete",
            "counterfactual": "u_x_forced_zero_only",
            "historical_state_limit": "qualification rows did not contain root/dof/RNG/low-level state; this preserves layout and runtime reset semantics only",
            "rows": rows, "summary": summary,
            "provenance": {
                "qualification": str(Path(parsed.qualification).resolve()),
                "qualification_sha256": _sha256(parsed.qualification),
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_iteration": iteration, "checkpoint_meta": checkpoint_meta,
                "low_level_checkpoint": str(low_level_path), "low_level_checkpoint_sha256": low_level_sha,
                "seed": parsed.seed, "terrain_avoid_seed": int(cfg.terrain.avoid_seed),
                "num_envs": parsed.num_envs, "active_central_envs": CENTRAL_COUNT,
                "stage12_success_threshold": float(cfg.terrain.avoid_stage12_success_threshold),
                "script_sha256": _sha256(__file__), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
    finally:
        if wrapper is not None:
            wrapper.env.gym.destroy_sim(wrapper.env.sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qualification", default=str(ROOT / "outputs/avoid_sanity_20260910/qualification_seed9174_20260911.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs/avoid_sanity_20260910/central_zero_lateral_20260911/central_zero_lateral.json"))
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--num_envs", type=int, default=FORMAL_ENVS)
    parser.add_argument("--max_steps", type=int, default=520)
    parsed = parser.parse_args()
    try:
        result = _run(parsed)
        _write_json(parsed.output, result)
        _write_json(Path(parsed.output).parent / "run_manifest.json", result["provenance"])
        _write_json(Path(parsed.output).parent / "status.json", {
            "status": result["status"], "counterfactual": result["counterfactual"],
            "terminal_rows": sum(bool(row["terminal_recorded"]) for row in result["rows"]),
            "total_rows": len(result["rows"]),
        })
    except Exception as exc:
        error = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        _write_json(parsed.output, error)
        _write_json(Path(parsed.output).parent / "status.json", error)
        raise


if __name__ == "__main__":
    main()
