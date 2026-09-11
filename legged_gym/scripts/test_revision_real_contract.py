#!/usr/bin/env python3
"""CPU checks for the revised real PCR observation and frozen-policy boundary."""
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "src_real" / "interface" / "scripts" / "pcr_real"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEPLOY))

try:
    import cv2  # noqa: F401
except ImportError:
    # The CPU contract fixture only needs the raw policy map; debug inflation
    # can be a no-op when this minimal environment lacks OpenCV.
    sys.modules["cv2"] = SimpleNamespace(dilate=lambda image, _kernel, iterations=1: image)

from legged_gym.pcr_observation import AVOID_ACTOR_MASK_INDICES, canonical_difficulty, canonical_local_map
from legged_gym.pcr_policy_contract import (
    AVOID_1D_REVISION,
    PCR_CANONICAL_REVISION,
    get_avoid_command,
    validate_frozen_avoid_revision,
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime_mod = _load_module("revision_realplay", ROOT / "legged_gym" / "scripts" / "pcr_realplay.py")
planner_mod = _load_module("revision_deployed_planner", DEPLOY / "high_level_planner.py")
builder_mod = _load_module("revision_real_input", ROOT / "legged_gym" / "scripts" / "real_pcr_input_check.py")


def _args(**overrides):
    values = dict(
        map_channels=2,
        map_size=32,
        map_extent_m=3.0,
        input_timeout_s=1.0,
        state_timeout_s=1.0,
        row_timeout_s=1.0,
        row_not_released_default=0.0,
        allow_missing_state=True,
        robot_clearance_m=0.27,
        robot_body_width_m=0.25,
        robot_swing_abduction_m=0.15,
        robot_depth_noise_margin_m=0.03,
        robot_extra_safety_margin_m=0.02,
        difficulty_radius_m=2.0,
        state_dim=9,
        cmd_safe_dist=0.25,
        cmd_free_dist=0.60,
        risk_forward_cmd_thr=0.02,
        risk_memory=True,
        risk_memory_velocity_source="body",
        risk_memory_l_clear=0.4,
        high_level_dt=0.1,
        w_mode="learned",
        w_blend_mode="multiply",
        signed_w_lambda=0.3,
        signed_w_gamma_risk=0.15,
        signed_w_margin=0.05,
        max_cmd_x=0.4,
        max_cmd_y=0.8,
        max_cmd_yaw=0.375,
        max_delta_x_per_s=100.0,
        max_delta_y_per_s=100.0,
        max_delta_yaw_per_s=100.0,
        stop_on_target_lost=True,
        stop_on_depth_invalid=True,
        stop_forward_when_target_too_close=True,
        publish_cmd=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_for_build():
    runtime = runtime_mod.PcrRealplay.__new__(runtime_mod.PcrRealplay)
    runtime.args = _args()
    runtime.torch = torch
    runtime.device = torch.device("cpu")
    runtime.revision_contract = True
    runtime.prev_gate_y = 0.37
    runtime.prev_cmd = np.asarray([0.11, 0.22, -0.33], dtype=np.float32)
    runtime.bridge = runtime_mod.RealPcrPolicyShim(runtime.args, torch, runtime.device)
    runtime.risk_memory = None
    runtime.risk_f_filter = None
    runtime.risk_a_filter = None
    runtime.prev_cmd_stamp = time.time()
    return runtime


def _snapshot(*, state=True, state_stamp=None, visible=True):
    now = time.time()
    local = np.zeros((2, 32, 32), dtype=np.float32)
    local[0, 16, 20] = 1.0
    visible_map = np.ones((1, 32, 32), dtype=np.float32) if visible else None
    raw_state = np.arange(9, dtype=np.float32) if state else None
    return runtime_mod.RealInputSnapshot(
        target=np.asarray([0.2, 1.5, 0.0, 0.0, 1.0], dtype=np.float32),
        local_map_2ch=local,
        policy_visible_map=visible_map,
        state=raw_state,
        target_stamp=now,
        local_map_stamp=now,
        policy_visible_stamp=now,
        state_stamp=now if state_stamp is None else state_stamp,
        row_stamp=now,
    )


def test_revised_tensor_contract():
    runtime = _runtime_for_build()
    snap = _snapshot()
    state, _goal, local, risk, difficulty, *_ = runtime._build_tensors(snap)
    expected = canonical_local_map(
        torch.as_tensor(snap.local_map_2ch[:1]),
        torch.as_tensor(snap.policy_visible_map),
        extent=3.0,
        clearance=0.27,
    )
    assert state.shape == (1, 13)
    assert torch.equal(state[0, :9], torch.arange(9, dtype=torch.float32))
    assert np.isclose(state[0, 9].item(), runtime.prev_gate_y)
    assert torch.allclose(state[0, 10:13], torch.as_tensor(runtime.prev_cmd))
    assert torch.allclose(local, expected)
    assert torch.allclose(risk, expected[:, :1])
    assert torch.allclose(difficulty, canonical_difficulty(expected, extent=3.0, radius=2.0))

    invalid_stamps = _snapshot()
    invalid_stamps.state_stamp = float("nan")
    for stale in (_snapshot(state=False), _snapshot(state_stamp=time.time() - 10.0), _snapshot(visible=False), invalid_stamps):
        try:
            runtime._build_tensors(stale)
        except runtime_mod.RealPcrRuntimeError:
            pass
        else:
            raise AssertionError("revised runtime accepted missing or stale required input")


def test_file_snapshot_preserves_provided_state_stamp():
    runtime = _runtime_for_build()
    now = time.time()
    payload = {
        "stamp": now,
        "robot_state_stamp": now - 0.1,
        "robot_state": list(range(9)),
        "target_state": [0.0, 1.0, 0.0, 0.0, 1.0],
        "local_map_2ch": np.zeros((2, 32, 32), dtype=np.float32).reshape(-1).tolist(),
        "policy_visible_map": np.ones((1, 32, 32), dtype=np.float32).reshape(-1).tolist(),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
        json.dump(payload, handle)
        handle.flush()
        runtime.args.obs_file = handle.name
        snap = runtime._file_snapshot()
    assert np.array_equal(snap.state, np.arange(9, dtype=np.float32))
    assert snap.state_stamp == payload["robot_state_stamp"]


def test_builder_uses_immediate_canonical_map_and_difficulty():
    args = _args(
        camera_pitch_down_deg=10.0,
        camera_height_m=0.45,
        camera_forward_offset_m=0.30,
        target_forward_offset_m=None,
        map_forward_offset_m=None,
        depth_stride=1,
        min_depth_m=0.25,
        max_depth_m=3.0,
        keep_person_in_map=False,
        person_mask_margin_px=0,
        target_mask_depth_margin_m=0.25,
        obstacle_mode="height_band",
        obstacle_min_height_m=0.08,
        obstacle_max_height_m=0.80,
        ground_remove_height_m=0.04,
        self_mask_half_width_m=0.45,
        self_mask_forward_m=0.25,
        observed_gate_radius_cells=0,
        unknown_cost=0.25,
        clearance_free_m=0.57,
        obstacle_memory=True,
        obstacle_memory_tau_s=10.0,
        obstacle_memory_threshold=0.1,
        difficulty_front_half_width_m=0.55,
        difficulty_front_min_m=0.05,
        difficulty_front_max_m=2.0,
        difficulty_front_min_points=1,
        difficulty_front_percentile=10.0,
    )
    intrin = SimpleNamespace(fx=100.0, fy=100.0, ppx=4.0, ppy=4.0)
    depth = np.full((8, 8), 1000, dtype=np.uint16)
    memory = {"occ": np.ones((32, 32), dtype=np.float32), "stamp": time.monotonic()}
    output = builder_mod.build_local_map_from_depth(depth, 0.001, intrin, args, None, float("nan"), memory)
    local_map, _clearance, difficulty, _mask, _invalid, _obs, _gate, visible, _risk, raw_occ, memory_occ, _debug = output
    expected = canonical_local_map(
        torch.as_tensor(raw_occ), torch.as_tensor(visible), extent=3.0, clearance=0.27
    )
    assert torch.allclose(torch.as_tensor(local_map), expected[0])
    assert np.isclose(difficulty, canonical_difficulty(expected, extent=3.0, radius=2.0)[0].item())
    assert np.count_nonzero(memory_occ) >= np.count_nonzero(raw_occ)


def test_writer_only_forwards_provided_state():
    obs = {
        "goal": np.zeros((1, 2), dtype=np.float32),
        "target_vel": np.zeros((1, 2), dtype=np.float32),
        "target_valid": np.asarray([True]),
        "target_too_close": np.asarray([False]),
        "depth_invalid": np.asarray([False]),
        "actor_difficulty": np.zeros(1, dtype=np.float32),
        "local_map_2ch": np.zeros((1, 2, 32, 32), dtype=np.float32),
        "risk_blocked_map": np.zeros((1, 1, 32, 32), dtype=np.float32),
        "policy_visible_map": np.ones((1, 1, 32, 32), dtype=np.float32),
        "raw_occ_map": np.zeros((1, 32, 32), dtype=np.float32),
        "memory_occ_map": np.zeros((1, 32, 32), dtype=np.float32),
        "front_distance_risk": np.zeros(1, dtype=np.float32),
        "robot_state": np.arange(9, dtype=np.float32),
        "robot_state_stamp": np.asarray([123.0]),
    }
    with tempfile.NamedTemporaryFile(mode="r+") as handle:
        builder_mod.write_policy_obs_file(handle.name, obs)
        with open(handle.name, "r") as reader:
            payload = json.load(reader)
    assert payload["robot_state"] == list(range(9))
    assert payload["robot_state_stamp"] == 123.0
    obs.pop("robot_state_stamp")
    with tempfile.NamedTemporaryFile(mode="r+") as handle:
        builder_mod.write_policy_obs_file(handle.name, obs)
        with open(handle.name, "r") as reader:
            payload = json.load(reader)
    assert payload["state_stamp"] == 0.0


def test_actual_policy_step_revised_dryrun():
    runtime = _runtime_for_build()
    runtime.cmd_scale = (1.0, 1.0, 1.0)
    runtime.aff_stack = 1
    runtime.avoid_aff_channels = 2
    runtime.gate_aff_channels = 2
    runtime.avoid_state_dim = 14
    runtime.gate_state_dim = 13
    runtime.avoid_goal_dim = 2
    runtime.gate_goal_dim = 18
    runtime.avoid_model = planner_mod.CmdVelExpert(
        affordance_channels=2, state_dim=14, goal_dim=2, action_dim=1,
        cmd_scale=(0.4,), actor_state_mask_indices=AVOID_ACTOR_MASK_INDICES,
    ).eval()
    runtime.gate_policy = planner_mod.GatePolicy(
        affordance_channels=2, state_dim=13, goal_dim=18, learned_w=True, actor_mask_xy=True,
    ).eval()
    import expert_s0_follow

    calls = []
    original = expert_s0_follow.compute_s0_follow_expert_cmd

    def relative_follow(robot_pos_world_xy, robot_heading, target_world_xy, *args, **kwargs):
        calls.append((robot_pos_world_xy.clone(), robot_heading.clone(), target_world_xy.clone()))
        return torch.tensor([[0.0, 0.25, 0.0]], dtype=torch.float32)

    expert_s0_follow.compute_s0_follow_expert_cmd = relative_follow
    try:
        result = runtime.policy_step(_snapshot())
    finally:
        expert_s0_follow.compute_s0_follow_expert_cmd = original
    assert len(calls) == 1
    assert torch.equal(calls[0][0], torch.zeros_like(calls[0][0]))
    assert torch.equal(calls[0][1], torch.zeros_like(calls[0][1]))
    assert np.isfinite(np.asarray(result["cmd_policy"])).all()
    assert np.isfinite(np.asarray(result["cmd_a"])).all()
    assert runtime.prev_cmd.shape == (3,) and np.isfinite(runtime.prev_cmd).all()
    assert np.isfinite(runtime.prev_gate_y)


def test_actual_revised_models_and_converter():
    avoid = planner_mod.CmdVelExpert(
        affordance_channels=2,
        state_dim=14,
        goal_dim=2,
        action_dim=1,
        cmd_scale=(0.4,),
        actor_state_mask_indices=AVOID_ACTOR_MASK_INDICES,
    )
    gate = planner_mod.GatePolicy(
        affordance_channels=2,
        state_dim=13,
        goal_dim=16,
        learned_w=True,
        actor_mask_xy=True,
    )
    aff = torch.zeros(1, 2, 32, 32)
    raw_state = torch.arange(13, dtype=torch.float32).view(1, -1)
    follow = torch.tensor([[0.0, 0.25, 0.0]])
    target = torch.tensor([[0.3, 1.2]])
    difficulty = torch.zeros(1)
    cmd_a, _ = get_avoid_command(avoid, aff, raw_state, follow, target, difficulty, target_valid=torch.tensor([True]))
    assert cmd_a.shape == (1, 3) and torch.isfinite(cmd_a).all()
    assert cmd_a[0, 1].item() == 0.0 and cmd_a[0, 2].item() == 0.0
    cmd_invalid, _ = get_avoid_command(avoid, aff, raw_state, follow, target, difficulty, target_valid=torch.tensor([False]))
    assert torch.isfinite(cmd_invalid).all()
    gate_a, _ = gate.get_action(aff, raw_state, torch.zeros(1, 16), difficulty, deterministic=True)
    assert gate_a.shape == (1, 2) and torch.isfinite(gate_a).all()

    validate_frozen_avoid_revision({}, {})
    validate_frozen_avoid_revision(
        {"revision_contract": PCR_CANONICAL_REVISION},
        {"revision_contract": AVOID_1D_REVISION},
    )
    try:
        validate_frozen_avoid_revision({"revision_contract": PCR_CANONICAL_REVISION}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("mixed legacy/revised checkpoint pair was accepted")


if __name__ == "__main__":
    test_revised_tensor_contract()
    test_file_snapshot_preserves_provided_state_stamp()
    test_builder_uses_immediate_canonical_map_and_difficulty()
    test_writer_only_forwards_provided_state()
    test_actual_revised_models_and_converter()
    test_actual_policy_step_revised_dryrun()
    print("revision real contract: PASS")
