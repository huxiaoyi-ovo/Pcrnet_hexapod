#!/usr/bin/env python3
"""Nearest assertions for the frozen Track-B eval-only supplement."""

import csv
import inspect
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch

from legged_gym.scripts import eval_highlevel as eh
from legged_gym.scripts import run_track_b_revision_eval as runner
from legged_gym.scripts import train_highlevel as th


class _RiskEnv:
    def __init__(self):
        self.post_processor = SimpleNamespace(safe_distance=0.27, free_distance=0.57)
        self.is_pcr_line_task = False
        self.affordance_map_extent = 3.0
        self.high_level_dt = 0.1
        self.env = SimpleNamespace(base_lin_vel=torch.tensor([[0.0, 0.4, 0.0]]))

    @staticmethod
    def _compute_clearance_along_cmd(_aff_map, cmd_xy):
        return torch.where(cmd_xy[:, 1].abs() > 0.5, torch.full_like(cmd_xy[:, 0], 0.30), torch.full_like(cmd_xy[:, 0], 0.50))

    @staticmethod
    def _risk_from_clearance(clearance, safe_distance, free_distance):
        return torch.clamp((float(free_distance) - clearance) / (float(free_distance) - float(safe_distance)), 0.0, 1.0)


def _risk_args():
    return SimpleNamespace(
        beta=None, pcr_w_aux_enable=False, risk_memory=True, risk_memory_l_clear=0.40,
        risk_memory_velocity_source="body", w_mode="learnedw2", w_tau=0.25,
        gate_safe_clamp=False, w_disable_gate_safe_clamp=True, signed_w_lambda=0.30,
        signed_w_gamma_risk=0.15, signed_w_margin=0.05, w_blend_mode="multiply",
    )


def _legacy_scalar_capsule_reference(occ_all, active_capsules, centers_world, ref_xy, yaw, capsule_radius, map_extent):
    """Frozen 29ee scalar capsule loop, retained only for the equivalence test."""
    out = occ_all.clone()
    map_size = int(occ_all.shape[1])
    cell = float(map_extent) / float(map_size)
    x_min = -0.5 * float(map_extent)
    for env_id in range(int(occ_all.shape[0])):
        for slot in range(int(active_capsules.shape[1])):
            if not bool(active_capsules[env_id, slot].item()):
                continue
            center_x = float(centers_world[env_id, slot, 0].item())
            center_y = float(centers_world[env_id, slot, 1].item())
            if not (math.isfinite(center_x) and math.isfinite(center_y)):
                continue
            world_xy = torch.tensor([[center_x, center_y]], device=ref_xy.device, dtype=ref_xy.dtype)
            dx = world_xy[0, 0] - ref_xy[env_id, 0]
            dy = world_xy[0, 1] - ref_xy[env_id, 1]
            cos_h, sin_h = torch.cos(yaw[env_id]), torch.sin(yaw[env_id])
            x_body = float((cos_h * dx + sin_h * dy).item())
            y_body = float((-sin_h * dx + cos_h * dy).item())
            ix0 = max(0, min(map_size - 1, int(math.floor((x_body - capsule_radius - x_min) / cell))))
            ix1 = max(0, min(map_size - 1, int(math.ceil((x_body + capsule_radius - x_min) / cell))))
            iy0 = max(0, min(map_size - 1, int(math.floor((y_body - capsule_radius) / cell))))
            iy1 = max(0, min(map_size - 1, int(math.ceil((y_body + capsule_radius) / cell))))
            out[env_id, ix0:ix1 + 1, iy0:iy1 + 1] = 1.0
    return out


def _test_capsule_bitwise():
    # The three rows represent base origin, rotated base origin, and camera origin respectively.
    occ = torch.zeros(3, 8, 8)
    occ[0, 0, 0] = 1.0
    active = torch.tensor([[True, False, True], [True, True, False], [True, True, True]])
    centers = torch.tensor([
        [[0.00, 1.00], [0.50, 0.50], [4.00, 1.00]],       # interior/inactive/outside
        [[1.75, 0.00], [0.00, -2.00], [0.00, 0.00]],      # rotated boundary/partial
        [[0.20, 1.20], [-2.50, 1.00], [float("nan"), 1.0]],  # camera-origin/partial/nonfinite
    ])
    ref_xy = torch.tensor([[0.0, 0.0], [0.5, -0.5], [0.2, 0.4]])
    yaw = torch.tensor([0.0, torch.pi / 2.0, -torch.pi / 4.0])
    scalar = _legacy_scalar_capsule_reference(occ, active, centers, ref_xy, yaw, 0.15, 3.0)
    vector = th._vectorized_legacy_capsule_raster(occ, active, centers, ref_xy, yaw, 0.15, 3.0)
    assert torch.equal(scalar, vector), (scalar - vector).nonzero().tolist()


def _test_manifest_path_and_selection():
    fixed = Path("/tmp/fixed")
    risk = Path("/tmp/risk")
    planned = runner.planned_paths(fixed, risk)
    assert len(planned) == 88 and len({str(path) for path in planned}) == 88
    assert runner.PRIMARY_SOURCES["success_rate"] == "task_success_rate"
    assert runner._difficulty_counts(128) == [26, 26, 26, 26, 24]
    assert any("/validation/fixed/y_0.00/speed_0.35/seed_101" in str(path) for path in planned)
    assert any("risk_closer_dsafe_0.22_dfree_0.52" in str(path) for path in planned)
    difficulties = [0.0, 0.25, 0.50, 0.75]
    entries = [
        {
            "canonical_episode_id": episode_id,
            "env_id": episode_id % 2,
            "local_episode_slot": episode_id // 2,
            "manifest_entry_id": f"env_{episode_id % 2:03d}_slot_{episode_id // 2}",
            "layout_id": f"layout_{episode_id}",
            "difficulty": difficulty,
        }
        for episode_id, difficulty in enumerate(difficulties)
    ]
    manifest = {
        "schema": "track_b_manifest_v3", "slots_per_env": 2,
        "difficulty_levels": list(runner.V1_DIFFICULTY_LEVELS),
        "difficulty_counts": {"0.00": 1, "0.25": 1, "0.50": 1, "0.75": 1, "1.00": 0},
        "layout_identity": {
            "normalized_physical_payload": True,
            "float_quantum": runner.LAYOUT_FLOAT_QUANTUM,
            "restore_atol": runner.LAYOUT_RESTORE_FLOAT_ATOL,
            "origin_roundtrip": "same_target_env_generation_and_replay",
        },
        "generation": {"base_task_only": True, "policy_actions": 0, "policy_checkpoints_loaded": False},
        "entries": entries,
    }
    runner.validate_manifest_quota(manifest, 2, 4)
    manifest["entries"].pop()
    try:
        runner.validate_manifest_quota(manifest, 2, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete per-env manifest quota must fail")
    manifest["entries"] = entries
    manifest["generation"]["policy_checkpoints_loaded"] = True
    try:
        runner.validate_manifest_quota(manifest, 2, 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("policy-dependent manifest must fail")
    physical_entry = {"env_id": 0, "difficulty": 0.25, "robot_root_local": [0.0], "manifest_entry_id": "env_000_slot_0", "env_seed": 7}
    archive_entry = dict(physical_entry, layout_id="derived-source-hash", canonical_episode_id=0, source_env_id=4, source_env_seed=99)
    assert eh._track_b_layout_id(physical_entry) == eh._track_b_layout_id(archive_entry)
    local = torch.tensor([0.1234567, -0.7654321, 0.25], dtype=torch.float32)
    target_origin = torch.tensor([100.0, -100.0, 0.0], dtype=torch.float32)
    roundtrip_local = (local + target_origin) - target_origin
    expected_restore = {
        "env_id": 0, "local_episode_slot": 0, "difficulty": 0.25,
        "robot_root_local": local.tolist(), "s_avoid_active": [True],
        "s_avoid_stage": 4, "s_avoid_episode_counter": 1,
        "target_line_finished": False, "manifest_entry_id": "env_000_slot_0",
        "canonical_episode_id": 0, "env_seed": 7,
    }
    expected_restore["layout_id"] = eh._track_b_layout_id(expected_restore)
    same_origin_restore = dict(expected_restore)
    eh._track_b_assert_restore_matches(expected_restore, same_origin_restore)
    cross_origin_restore = dict(expected_restore, robot_root_local=roundtrip_local.tolist())
    cross_origin_restore["layout_id"] = eh._track_b_layout_id(cross_origin_restore)
    assert expected_restore["layout_id"] == cross_origin_restore["layout_id"]
    eh._track_b_assert_restore_matches(expected_restore, cross_origin_restore)
    rows = []
    for y_const in runner.VALIDATION_Y:
        for speed in runner.VALIDATION_SPEEDS:
            rows.append({
                "y_const": y_const, "speed_mps": speed, "episodes": 128,
                "success_rate": 0.8 if y_const in (0.25, 0.30) else 0.7,
                "episode_collision_rate": 0.1 if y_const in (0.25, 0.30) else 0.2,
                "follow_mae_m_mean": 0.2 if y_const in (0.25, 0.30) else 0.3,
                "row_progress_success_mean": 0.5,
            })
    selected = runner._selection(rows)
    assert selected["selected"]["y_const"] == 0.25
    reference = {
        "manifest_sha256": "a",
        "manifest_layout_pairs": [("env_000_slot_0", "x")],
        "canonical_manifest_layout_rows": [(0, "env_000_slot_0", "x", 0.0)],
    }
    try:
        runner._assert_paired(reference, {
            "manifest_sha256": "a",
            "manifest_layout_pairs": [("env_000_slot_0", "wrong")],
            "canonical_manifest_layout_rows": [(0, "env_000_slot_0", "wrong", 0.0)],
        }, "negative")
    except RuntimeError:
        pass
    else:
        raise AssertionError("layout mismatch must fail pairing")


def _test_fixed_risk_and_csv():
    cmd_f = torch.tensor([[0.0, 1.0, 0.3]])
    cmd_a = torch.tensor([[0.6, 0.7, -0.4]])
    fixed = eh._compute_fixed_authority_diag(cmd_f, cmd_a, 0.25)
    assert torch.equal(fixed["cmd_a"], torch.tensor([[0.6, 0.0, 0.0]]))
    assert torch.allclose(fixed["cmd"], 0.25 * cmd_f + 0.75 * fixed["cmd_a"])
    assert torch.isnan(fixed["risk_F"]).all() and torch.isnan(fixed["w"]).all()
    policy_step_source = inspect.getsource(eh.EvalRunner._policy_step)
    assert policy_step_source.index("if self.is_fixed_authority:") < policy_step_source.index("self.gate_policy_call_count += 1")

    env, args = _RiskEnv(), _risk_args()
    env._track_b_risk_mapping_override = (0.22, 0.52)
    eh._install_track_b_risk_mapping_override()
    aff = torch.zeros((1, 2, 2, 2))
    learned_goal, build_diag = th.build_learned_w_gate_goal(env, args, torch.zeros((1, 2)), aff, cmd_f, cmd_a)
    resolved = th.resolve_moe_gate_pcr(env, args, aff, torch.tensor([0.5]), cmd_f, cmd_a, learned_w=torch.tensor([0.7]))
    risk_f = torch.tensor([(0.52 - 0.30) / (0.52 - 0.22)])
    risk_a = torch.tensor([(0.52 - 0.50) / (0.52 - 0.22)])
    assert learned_goal.shape[-1] == 18
    assert torch.allclose(build_diag["risk_F"], risk_f) and torch.allclose(build_diag["risk_A"], risk_a)
    assert torch.allclose(resolved["risk_F"], risk_f) and torch.allclose(resolved["risk_A"], risk_a)
    assert torch.allclose(resolved["risk_diff_correction"], 0.15 * (risk_a - risk_f))
    assert (env.post_processor.safe_distance, env.post_processor.free_distance) == (0.27, 0.57)
    recorded_risk = {
        "safe_distance": 0.22, "free_distance": 0.52,
        "postprocess_safe_distance": 0.27,
        "postprocess_free_distance": 0.5700000000000001,
        "postprocessor_unchanged": True,
    }
    assert runner._risk_mapping_matches(recorded_risk, 0.22, 0.52)
    assert runner._nominal_postprocessor_matches(recorded_risk)
    assert not runner._risk_mapping_matches(dict(recorded_risk, free_distance=0.53), 0.22, 0.52)
    assert not runner._nominal_postprocessor_matches(dict(recorded_risk, postprocess_free_distance=0.58))

    with tempfile.TemporaryDirectory() as temp_dir:
        eh._write_outputs({"per_episode": [{"episode_id": 0, "manifest_entry_id": "env_000_slot_0", "layout_id": "layout"}]}, temp_dir)
        with (Path(temp_dir) / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        assert row["layout_id"] == "layout" and row["manifest_entry_id"] == "env_000_slot_0"


def main():
    _test_capsule_bitwise()
    _test_manifest_path_and_selection()
    _test_fixed_risk_and_csv()
    print("Track-B canonical assertions passed.")


if __name__ == "__main__":
    main()
