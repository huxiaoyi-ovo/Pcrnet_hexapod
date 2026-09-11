#!/usr/bin/env python3
"""Independent formal qualification for the frozen 1-D Avoid checkpoint.

This script owns evaluation accounting only.  It never changes rewards,
termination, obstacle bands, or curriculum source code.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


PASS_SUCCESS = 0.85
PASS_FAILURE = 0.10
FORMAL_SEED = 9174
FORMAL_ENVS = 64
FORMAL_DECISIONS = 200
FORMAL_ITERATION = 200
EXPECTED_LOW_LEVEL_SHA256 = "75dfd7aae52b20f08ca9f654c37b2e20ba1e33f8b953737437ad774d122d1481"


def decision_quota(num_envs=FORMAL_ENVS, total=FORMAL_DECISIONS):
    """Give the first ``total % num_envs`` environments one extra episode."""
    if num_envs <= 0 or total <= 0:
        raise ValueError("positive quota required")
    base, extra = divmod(total, num_envs)
    return [base + (index < extra) for index in range(num_envs)]


def classify_terminal(physical, envelope, fall, timeout, success):
    """Classify a terminal using the frozen failure-first precedence."""
    failure = bool(physical or envelope or fall)
    if physical:
        return "physical", failure
    if envelope:
        return "envelope", failure
    if fall:
        return "fall", failure
    if success:
        return "success", failure
    if timeout:
        return "timeout", failure
    return "other", failure


def _trace_xy(point):
    if isinstance(point, dict):
        return float(point["y_forward"]), float(point["x_right"])
    y, x = point
    return float(y), float(x)


def _forward_crossings(trace, y_target):
    """Return ordered forward crossings as (segment index, interpolated x)."""
    crossings = []
    for index, (first, second) in enumerate(zip(trace[:-1], trace[1:])):
        y0, x0 = _trace_xy(first)
        y1, x1 = _trace_xy(second)
        if not (math.isfinite(y0) and math.isfinite(x0) and math.isfinite(y1) and math.isfinite(x1)):
            continue
        if y1 <= y0 or y_target < y0 or y_target > y1:
            continue
        alpha = (y_target - y0) / (y1 - y0)
        crossings.append((index, x0 + alpha * (x1 - x0)))
    return crossings


def successful_outer_bypass(trace, band_enter, band_exit, coverage, success):
    """Return ``left``/``right`` only for an observed full same-side crossing.

    A point outside coverage is insufficient: the trace must cross both real
    band boundaries while outside the same real coverage interval.
    """
    if not success or len(trace) < 2 or not (len(band_enter) == len(band_exit) == len(coverage)):
        return None
    for enter, exit_, interval in zip(band_enter, band_exit, coverage):
        if len(interval) != 2:
            continue
        xmin, xmax = float(interval[0]), float(interval[1])
        entry_x = _forward_crossings(trace, float(enter))
        exit_x = _forward_crossings(trace, float(exit_))
        for side, outside in (("left", lambda x: x < xmin), ("right", lambda x: x > xmax)):
            for enter_index, x_enter in entry_x:
                for exit_index, x_exit in exit_x:
                    if exit_index < enter_index or not (outside(x_enter) and outside(x_exit)):
                        continue
                    # Every sampled point between the ordered boundaries must
                    # remain outside the same side.  This rejects a later
                    # return after an ordinary in-coverage traversal.
                    middle = [_trace_xy(point)[1] for point in trace[enter_index + 1:exit_index + 1]]
                    if all(outside(x) for x in middle):
                        return side
    return None


def gate(rows, expected=FORMAL_DECISIONS, quota=None, smoke=False):
    """Compute the formal gate from counted decision rows only."""
    decisions = [row for row in rows if row.get("decision")]
    finite = all(bool(row.get("finite", False)) for row in decisions)
    metadata_correct = all(bool(row.get("metadata_ok", False)) for row in decisions)
    success = sum(row.get("reason") == "success" for row in decisions)
    failure = sum(bool(row.get("failure")) for row in decisions)
    other = sum(row.get("reason") == "other" for row in decisions)
    side_rows = {
        side: [row for row in decisions if row.get("reason") == "success" and row.get("bypass_side") == side]
        for side in ("left", "right")
    }
    repeated = {
        side: {"successes": len(side_rows[side]), "envs": sorted({int(row["env_id"]) for row in side_rows[side]})}
        for side in side_rows
    }
    blocking = [side for side, summary in repeated.items() if summary["successes"] >= 3 and len(summary["envs"]) >= 2]
    n = len(decisions)
    if quota is None:
        quota_ok, quota_counts = n == expected, None
    else:
        quota_counts = [sum(int(row.get("env_id", -1)) == index for row in decisions) for index in range(len(quota))]
        quota_ok = quota_counts == list(quota) and n == expected
    success_rate, failure_rate = success / max(n, 1), failure / max(n, 1)
    reasons = []
    if smoke:
        reasons.append("smoke_not_qualification")
    if not quota_ok:
        reasons.append("quota_incomplete_or_mismatched")
    if not finite:
        reasons.append("nonfinite")
    if not metadata_correct:
        reasons.append("metadata_incorrect")
    if other:
        reasons.append("other_terminal")
    if blocking:
        reasons.append("repeated_outer_bypass_" + "_".join(blocking))
    if success_rate < PASS_SUCCESS:
        reasons.append("success_below_threshold")
    if failure_rate > PASS_FAILURE:
        reasons.append("failure_above_threshold")
    return {
        "decision_episodes": n, "quota": list(quota) if quota is not None else None,
        "quota_counts": quota_counts, "success_rate": success_rate, "failure_rate": failure_rate,
        "finite": finite, "metadata_correct": metadata_correct, "other_terminals": other,
        "outer_bypass": repeated, "formal_blocking_bypass_sides": blocking,
        "pass": not reasons, "reason": "pass" if not reasons else ";".join(reasons),
    }


def _plain(value):
    """Convert known metadata values to JSON primitives without importing NumPy."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    return str(value)


def _raw_numeric_finite(value):
    """Check raw layout values before JSON conversion can stringify NaN/Inf."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_raw_numeric_finite(item) for item in value.values())
    if hasattr(value, "__dict__"):
        return _raw_numeric_finite(vars(value))
    if isinstance(value, (list, tuple)):
        return all(_raw_numeric_finite(item) for item in value)
    if hasattr(value, "tolist"):
        return _raw_numeric_finite(value.tolist())
    if hasattr(value, "item"):
        return _raw_numeric_finite(value.item())
    return True


def _layout_snapshot(layout):
    if layout is None:
        return None
    fields = (
        "stage", "kind", "decision_episode", "cylinders_xy", "band_y", "safe_intervals",
        "reachable_intervals", "exit_y", "v_drive_nom", "min_forward_assumption",
        "combined_radius", "m_clear", "kappa", "band_enter_y", "band_exit_y", "coverage",
        "recovery_paths", "band_decisions", "speed_draws", "layout_attempts", "structure_candidates",
    )
    snapshot = {field: _plain(getattr(layout, field, None)) for field in fields}
    snapshot["raw_numeric_finite"] = _raw_numeric_finite(
        {field: getattr(layout, field, None) for field in fields}
    )
    return snapshot


def _finite_numbers(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    return False


def layout_metadata_ok(snapshot, expected_stage=1):
    """Check captured metadata itself, never config defaults or hard-coded bands."""
    if not isinstance(snapshot, dict) or int(snapshot.get("stage", -1)) != int(expected_stage):
        return False
    if snapshot.get("raw_numeric_finite") is not True:
        return False
    kind, decision = snapshot.get("kind"), bool(snapshot.get("decision_episode"))
    if kind not in ("free", "central", "decision") or decision != (kind == "decision"):
        return False
    enter, exit_, coverage = snapshot.get("band_enter_y"), snapshot.get("band_exit_y"), snapshot.get("coverage")
    if not isinstance(enter, list) or not isinstance(exit_, list) or not isinstance(coverage, list):
        return False
    if not (len(enter) == len(exit_) == len(coverage)):
        return False
    if kind == "free":
        return len(enter) == 0 and _finite_numbers(snapshot)
    if len(enter) == 0:
        return False
    for low, high, interval in zip(enter, exit_, coverage):
        if not (_finite_numbers((low, high, interval)) and float(low) < float(high)):
            return False
        if not isinstance(interval, list) or len(interval) != 2 or float(interval[0]) >= float(interval[1]):
            return False
    return _finite_numbers(snapshot)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_bool(tensor, index):
    return bool(tensor[index].detach().item())


def _assert_finite(torch, name, value):
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite {name}")


def _assert_obs_contract(torch, obs, num_envs):
    expected = {"state": (num_envs, 14), "goal": (num_envs, 2), "local_map_2ch": (num_envs, 2, 32, 32), "actor_difficulty": (num_envs,)}
    for name, shape in expected.items():
        if name not in obs or tuple(obs[name].shape) != shape:
            actual = None if name not in obs else tuple(obs[name].shape)
            raise RuntimeError(f"{name} shape {actual}, expected {shape}")
        _assert_finite(torch, name, obs[name])


def _assert_checkpoint_finite(torch, value, label="checkpoint"):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"non-finite {label}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_checkpoint_finite(torch, item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_checkpoint_finite(torch, item, f"{label}[{index}]")


def _checkpoint_iteration(checkpoint):
    for key in ("iteration", "iter", "current_learning_iteration"):
        if key in checkpoint:
            return int(checkpoint[key])
    return None


def _validate_checkpoint_metadata(checkpoint, policy_kwargs, actor_mask):
    meta = checkpoint.get("experiment_meta")
    if not isinstance(meta, dict):
        raise RuntimeError("checkpoint has no experiment_meta dictionary")
    required = {
        "revision_contract": "avoid_1d_canonical_v1", "task": "s_avoid_clutter", "mode": "teacher",
        "skill": "avoid", "state_dim": 14, "goal_dim": 2, "policy_goal_dim": 2,
        "actor_output_dim": 1, "map_semantics": "canonical_observed_2ch",
        "goal_semantics": "virtual_side_preference", "drive_semantics": "external_nominal_forward_request",
        "decimation": 5,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"checkpoint metadata {key}={meta.get(key)!r}, expected {expected!r}")
    if list(meta.get("actor_state_mask_indices", ())) != list(actor_mask):
        raise RuntimeError("checkpoint actor state mask does not match frozen 1-D contract")
    if int(policy_kwargs.get("action_dim", -1)) != 1:
        raise RuntimeError("checkpoint command head is not 1-D")
    return meta


def _runtime_args(root, gymapi, parsed):
    return SimpleNamespace(
        mode="teacher", skill="avoid", task="s_avoid_clutter", seed=parsed.seed, num_envs=parsed.num_envs,
        decimation=5, low_level_ckpt=str(root / "agents" / "low_level_best.pt"), aff_stack=1,
        physics_engine=gymapi.SIM_PHYSX, sim_device="cuda:0", sim_device_id=0, rl_device="cuda:0",
        use_gpu=True, use_gpu_pipeline=True, headless=True, debug=False, camera_enable=False,
        revision_contract=True, cmd_slew_lin=0.2, cmd_slew_ang=0.4, cmd_safe_dist=None,
        cmd_free_dist=None, beta=None, disable_risk_scale=False, force_cmd_y=False,
    )


def _camera_record(wrapper):
    camera_cfg = getattr(getattr(getattr(wrapper.env, "cfg", None), "sensor", None), "depth_camera", None)
    return {
        "camera_enable": bool(getattr(wrapper.args, "camera_enable", False)),
        "horizontal_fov_deg": None if wrapper.camera_fov_rad is None else math.degrees(wrapper.camera_fov_rad),
        "vertical_fov_deg": None if wrapper.camera_vertical_fov_rad is None else math.degrees(wrapper.camera_vertical_fov_rad),
        "near_clip_m": float(wrapper.camera_near), "far_clip_m": float(wrapper.camera_far),
        "depth_resolution": [
            None if camera_cfg is None else int(getattr(camera_cfg, "width", 0)),
            None if camera_cfg is None else int(getattr(camera_cfg, "height", 0)),
        ],
        "local_map_shape": [2, int(wrapper.affordance_map_size), int(wrapper.affordance_map_size)],
        "local_map_extent_m": float(wrapper.affordance_map_extent),
        "local_map_origin": str(wrapper.affordance_origin_mode),
        "local_map_origin_local_xy": _plain(wrapper.affordance_origin_local_xy),
        "map_source": "observed canonical local_map_2ch (simulation teacher reference)",
    }


def _capture_context(impl, env_id):
    layout = impl.s_avoid_clutter_layouts[env_id]
    snapshot = _layout_snapshot(layout)
    if not layout_metadata_ok(snapshot, expected_stage=1):
        raise RuntimeError(f"invalid stage-1 layout metadata for env {env_id}")
    return {"layout_object": layout, "layout": snapshot, "trace": [], "finite": True}


def _append_local_trace(context, impl, env_id):
    root, origin = impl.root_states[env_id], impl.env_origins[env_id]
    x = float((root[0] - origin[0]).detach().cpu().item())
    y = float((root[1] - origin[1]).detach().cpu().item())
    if not (math.isfinite(x) and math.isfinite(y)):
        raise RuntimeError(f"non-finite env-local trace for env {env_id}")
    context["trace"].append({"x_right": x, "y_forward": y})


def _row_from_terminal(env_id, context, info, counted):
    masks = {
        "physical": _tensor_bool(info["terminal_physical"], env_id),
        "envelope": _tensor_bool(info["terminal_envelope"], env_id),
        "fall": _tensor_bool(info["terminal_fall"], env_id),
        "timeout": _tensor_bool(info["timeout"], env_id),
        "success": _tensor_bool(info["success_mask"], env_id),
    }
    reason, failure = classify_terminal(**masks)
    layout = context["layout"]
    bypass_side = successful_outer_bypass(
        context["trace"], layout["band_enter_y"], layout["band_exit_y"], layout["coverage"],
        masks["success"] and not failure,
    )
    return {
        "env_id": int(env_id), "decision": bool(counted),
        "decision_candidate": bool(layout["decision_episode"]),
        "quota_saturated": bool(layout["decision_episode"] and not counted), "kind": layout["kind"],
        "reason": reason, "failure": failure, "finite": bool(context["finite"]),
        "metadata_ok": layout_metadata_ok(layout, expected_stage=1), "terminal_masks": masks,
        "bypass_side": bypass_side, "trace_coordinate_frame": "env_local_x_right_y_forward",
        "trace": context["trace"], "layout": layout,
    }


def _run(parsed):
    # Isaac must be imported before Torch in the real runner process.
    from isaacgym import gymapi
    import torch

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not parsed.smoke_steps and (parsed.seed != FORMAL_SEED or parsed.num_envs != FORMAL_ENVS):
        raise RuntimeError("formal qualification is fixed at stage-1 seed 9174 with 64 environments")
    if not torch.cuda.is_available():
        raise RuntimeError("qualification requires a CUDA-visible device mapped as cuda:0")
    if parsed.max_steps <= 0 or parsed.smoke_steps < 0:
        raise ValueError("max_steps must be positive and smoke_steps must be non-negative")

    # train_highlevel's Isaac parser must never see runner arguments.
    sys.argv = [sys.argv[0]]
    from legged_gym.pcr_observation import AVOID_ACTOR_MASK_INDICES
    from legged_gym.pcr_policy_contract import checkpoint_cmd_policy_kwargs
    from legged_gym.scripts import train_highlevel as highlevel

    torch.manual_seed(parsed.seed)
    torch.cuda.manual_seed_all(parsed.seed)
    low_level_path = root / "agents" / "low_level_best.pt"
    low_level_sha256 = _sha256(low_level_path)
    if low_level_sha256 != EXPECTED_LOW_LEVEL_SHA256:
        raise RuntimeError(f"unexpected low-level checkpoint SHA256: {low_level_sha256}")
    checkpoint_path = Path(parsed.checkpoint).resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cuda:0")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError("qualification checkpoint lacks model_state_dict")
    _assert_checkpoint_finite(torch, checkpoint)
    iteration = _checkpoint_iteration(checkpoint)
    if not parsed.smoke_steps and (checkpoint_path.name != "model_200.pt" or iteration != FORMAL_ITERATION):
        raise RuntimeError("formal qualification requires model_200.pt with checkpoint iteration 200")

    highlevel.import_modules()
    cfg, _ = highlevel.task_registry.get_cfgs("s_avoid_clutter")
    cfg.seed = parsed.seed
    cfg.terrain.avoid_seed = parsed.seed
    cfg.terrain.avoid_stage12_success_threshold = 1.01
    wrapper = None
    try:
        args = _runtime_args(root, gymapi, parsed)
        wrapper = highlevel.HierarchicalHexapodEnv(args, torch.device("cuda:0"), env_cfg=cfg)
        obs = wrapper.reset()
        _assert_obs_contract(torch, obs, parsed.num_envs)
        policy_kwargs = checkpoint_cmd_policy_kwargs(checkpoint, tuple(float(item) for item in wrapper.post_processor.max_cmd.detach().cpu()))
        checkpoint_meta = _validate_checkpoint_metadata(checkpoint, policy_kwargs, AVOID_ACTOR_MASK_INDICES)
        actor = highlevel.CmdVelExpert(affordance_channels=2, state_dim=14, goal_dim=2, **policy_kwargs).to(wrapper.device)
        # Qualification must not invoke the permissive compatibility loader.
        actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for name, tensor in actor.state_dict().items():
            _assert_finite(torch, f"actor_parameter.{name}", tensor)
        actor.eval()

        quota, counts, rows = decision_quota(parsed.num_envs, FORMAL_DECISIONS), [0] * parsed.num_envs, []
        contexts, steps_completed, termination = [None] * parsed.num_envs, 0, "max_steps"
        for step in range(parsed.max_steps):
            _assert_obs_contract(torch, obs, parsed.num_envs)
            with torch.no_grad():
                action, _ = actor.get_action(obs["local_map_2ch"], obs["state"], obs["goal"], obs["actor_difficulty"], deterministic=True)
            if tuple(action.shape) != (parsed.num_envs, 1):
                raise RuntimeError(f"1-D action shape {tuple(action.shape)} is invalid")
            _assert_finite(torch, "action", action)
            impl = wrapper.env
            for env_id in range(parsed.num_envs):
                if contexts[env_id] is None:
                    contexts[env_id] = _capture_context(impl, env_id)
                elif contexts[env_id]["layout_object"] is not impl.s_avoid_clutter_layouts[env_id]:
                    raise RuntimeError(f"layout changed before terminal accounting for env {env_id}")
                _append_local_trace(contexts[env_id], impl, env_id)

            next_obs, reward, done, info = wrapper.step(action)
            _assert_finite(torch, "reward", reward)
            for name in ("cmd_raw", "cmd_exec_mean", "cmd_override_final", "cmd_preview_for_clearance"):
                _assert_finite(torch, name, info.get("post_info", {}).get(name))
            _assert_obs_contract(torch, next_obs, parsed.num_envs)
            if not torch.is_tensor(done) or tuple(done.shape) != (parsed.num_envs,):
                raise RuntimeError("done mask shape is invalid")
            for name in ("terminal_physical", "terminal_envelope", "terminal_fall", "timeout", "success_mask"):
                if name not in info or not torch.is_tensor(info[name]) or tuple(info[name].shape) != (parsed.num_envs,):
                    raise RuntimeError(f"terminal mask {name} is missing or malformed")
            for env_id in done.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
                context = contexts[env_id]
                if context is None:
                    raise RuntimeError(f"terminal env {env_id} has no old episode context")
                candidate = bool(context["layout"]["decision_episode"])
                counted = candidate and counts[env_id] < quota[env_id]
                row = _row_from_terminal(env_id, context, info, counted)
                rows.append(row)
                if row["reason"] == "other":
                    raise RuntimeError(f"terminal reason other for env {env_id}")
                if counted:
                    counts[env_id] += 1
                contexts[env_id] = None
            obs, steps_completed = next_obs, step + 1
            if parsed.smoke_steps and steps_completed >= parsed.smoke_steps:
                termination = "smoke_steps"
                break
            if counts == quota:
                termination = "quota_complete"
                break

        return {
            "status": "smoke_complete" if parsed.smoke_steps else "completed", "formal": not bool(parsed.smoke_steps),
            "smoke": bool(parsed.smoke_steps), "gate": gate(rows, expected=FORMAL_DECISIONS, quota=quota, smoke=bool(parsed.smoke_steps)),
            "termination": termination, "steps_completed": steps_completed, "max_steps": parsed.max_steps,
            "quota": quota, "quota_counts": counts, "episodes": rows,
            "free_episodes": [row for row in rows if row["kind"] == "free"],
            "central_episodes": [row for row in rows if row["kind"] == "central"],
            "uncounted_decision_episodes": [row for row in rows if row["decision_candidate"] and not row["decision"]],
            "provenance": {
                "task": "s_avoid_clutter", "stage": 1, "seed": parsed.seed, "terrain_avoid_seed": int(cfg.terrain.avoid_seed),
                "stage12_success_threshold": float(cfg.terrain.avoid_stage12_success_threshold), "num_envs": parsed.num_envs,
                "deterministic_policy": True, "policy_observation": {"state": 14, "goal": 2, "local_map_2ch": [2, 32, 32]},
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256, "checkpoint_iteration": iteration,
                "checkpoint_meta": checkpoint_meta, "low_level_checkpoint": str(low_level_path),
                "low_level_checkpoint_sha256": low_level_sha256, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "sim_device": "cuda:0", "rl_device": "cuda:0", "runtime_inputs": _plain(vars(args)), "camera": _camera_record(wrapper),
            },
        }
    finally:
        if wrapper is not None:
            wrapper.env.gym.destroy_sim(wrapper.env.sim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_envs", type=int, default=FORMAL_ENVS)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--smoke_steps", type=int, default=0)
    parsed = parser.parse_args()
    try:
        result = _run(parsed)
        _write_json(parsed.output, result)
        print(json.dumps(result["gate"], sort_keys=True))
        return 0
    except Exception as exc:
        error = {
            "status": "error", "formal": not bool(parsed.smoke_steps), "smoke": bool(parsed.smoke_steps),
            "checkpoint": parsed.checkpoint, "output": parsed.output, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }
        _write_json(parsed.output, error)
        _write_json(Path(parsed.output).with_suffix(Path(parsed.output).suffix + ".error.json"), error)
        raise


if __name__ == "__main__":
    main()
