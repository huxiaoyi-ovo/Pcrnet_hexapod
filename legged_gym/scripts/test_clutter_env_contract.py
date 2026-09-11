#!/usr/bin/env python3
"""CPU source-contract checks for clutter reset, metrics, and terminal priority."""

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
GROUND = ROOT / "legged_gym/envs/hex_v4/hex_ground.py"
CONFIG = ROOT / "legged_gym/envs/hex_v4/hex_scenes_config.py"


def function_source(source, tree, name):
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            found = node
    if found is None:
        raise AssertionError("missing function: %s" % name)
    return ast.get_source_segment(source, found)


def function_node(tree, name):
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            found = node
    if found is None:
        raise AssertionError("missing function: %s" % name)
    return found


def extract_methods(tree, names):
    module = ast.Module(body=[function_node(tree, name) for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "np": np, "Optional": Optional}
    exec(compile(module, str(GROUND), "exec"), namespace)
    return namespace


def make_history(window=200):
    return {
        "collision": deque(maxlen=window),
        "exposure": deque(maxlen=window),
        "progress": deque(maxlen=window),
        "success": deque(maxlen=window),
        "row_success": deque(maxlen=window),
    }


def run_extracted_behavior(tree):
    methods = extract_methods(
        tree,
        [
            "_update_s_avoid_curriculum",
            "_get_s_avoid_episode_row_pass_counts",
            "_get_s_avoid_episode_row_progress_ratios",
            "_get_s_avoid_episode_success_flags",
        ],
    )

    class Harness:
        _update_s_avoid_curriculum = methods["_update_s_avoid_curriculum"]
        _get_s_avoid_episode_row_pass_counts = methods["_get_s_avoid_episode_row_pass_counts"]
        _get_s_avoid_episode_row_progress_ratios = methods["_get_s_avoid_episode_row_progress_ratios"]
        _get_s_avoid_episode_success_flags = methods["_get_s_avoid_episode_success_flags"]

        def _advance_s_avoid_stage(self, next_stage):
            self.s_avoid_stage = next_stage
            for history in self.s_avoid_stage_metric_hists[next_stage].values():
                history.clear()
            self.s_avoid_stage_completed_episodes[next_stage] = 0

        def _get_s_avoid_cross_line_dist(self, env_ids, stage_ids=None):
            return self.cross_line_dist[env_ids]

    harness = Harness()
    harness.device = "cpu"
    harness.s_avoid_enabled = True
    harness.s_avoid_clutter_enabled = True
    harness.s_avoid_stage = 1
    harness.s_avoid_stage_metric_hists = {stage: make_history() for stage in range(1, 5)}
    harness.s_avoid_stage_completed_episodes = {stage: 0 for stage in range(1, 5)}
    harness.s_avoid_total_completed_episodes = 0
    harness.extras = {}
    harness.cfg = SimpleNamespace(terrain=SimpleNamespace(
        avoid_stage12_success_threshold=.85,
        avoid_stage12_collision_threshold=.10,
    ))

    stages = torch.tensor([1, 1, 2, 1])
    decisions = torch.tensor([True, True, True, False])
    failures = torch.tensor([False, True, True, True])
    exposures = torch.tensor([True, False, True, True])
    progresses = torch.tensor([.25, .75, .5, .5])
    successes = ~failures
    row_successes = torch.tensor([.5, 1.0, .5, .5])
    harness._update_s_avoid_curriculum(
        failures,
        stages,
        episode_exposure_flags=exposures,
        episode_progress_flags=progresses,
        episode_success_flags=successes,
        episode_row_success_flags=row_successes,
        episode_decision_flags=decisions,
    )
    assert harness.s_avoid_stage == 1
    assert harness.s_avoid_total_completed_episodes == 2
    assert harness.s_avoid_stage_completed_episodes[1] == 2
    assert len(harness.s_avoid_stage_metric_hists[1]["collision"]) == 2
    assert len(harness.s_avoid_stage_metric_hists[2]["collision"]) == 0
    assert harness.extras["avoid_stage_completed_episodes"] == 2
    assert harness.extras["avoid_stage_exposure_rate"] == .5
    assert harness.extras["avoid_stage_progress_rate"] == .5
    assert harness.extras["avoid_stage_row_success_rate"] == .75

    for history in harness.s_avoid_stage_metric_hists[1].values():
        history.clear()
    harness.s_avoid_stage_completed_episodes[1] = 0
    harness.s_avoid_total_completed_episodes = 0
    stages = torch.tensor([1] * 200 + [1] * 200 + [2] * 200)
    decisions = torch.tensor([True] * 200 + [False] * 200 + [True] * 200)
    failures = torch.tensor([False] * 200 + [True] * 200 + [True] * 200)
    successes = ~failures
    harness._update_s_avoid_curriculum(failures, stages, episode_success_flags=successes, episode_decision_flags=decisions)
    assert harness.s_avoid_stage == 2
    assert harness.s_avoid_total_completed_episodes == 200
    assert len(harness.s_avoid_stage_metric_hists[2]["collision"]) == 0
    assert harness.extras["avoid_stage_rate_source"] == 2
    assert harness.extras["avoid_stage_completed_episodes"] == 0
    assert harness.extras["avoid_stage_progress_rate"] == 0.0
    assert harness.extras["avoid_stage_row_success_rate"] == 0.0
    assert harness.extras["avoid_stage_success_rate"] == 0.0
    assert harness.extras["avoid_stage_collision_rate"] == 0.0

    harness.root_states = torch.zeros(2, 13)
    harness.env_origins = torch.zeros(2, 3)
    harness.root_states[:, 1] = torch.tensor([.5, -.6])
    harness.s_avoid_stage_per_env = torch.ones(2, dtype=torch.long)
    harness.s_avoid_band_count = torch.tensor([2, 0], dtype=torch.long)
    harness.s_avoid_band_exit_y = torch.tensor([[0., 1., float("inf"), float("inf"), float("inf")], [float("inf")] * 5])
    harness.s_avoid_episode_rows_passed_best = torch.zeros(2, dtype=torch.long)
    harness.s_avoid_exit_y = torch.tensor([1.45, .4])
    progress = harness._get_s_avoid_episode_row_progress_ratios(torch.tensor([0, 1]))
    assert torch.allclose(progress, torch.tensor([.5, .5]))

    harness.cross_line_dist = torch.tensor([0., 0., 0., .2])
    harness.s_avoid_stage_per_env = torch.ones(4, dtype=torch.long)
    harness.s_avoid_episode_physical = torch.tensor([False, False, True, False])
    harness.s_avoid_episode_envelope = torch.tensor([False, True, False, False])
    harness.s_avoid_episode_fall = torch.tensor([False, False, False, True])
    success = harness._get_s_avoid_episode_success_flags(torch.tensor([0, 1, 2, 3]))
    assert torch.equal(success, torch.tensor([True, False, False, False]))


def main():
    ground = GROUND.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    tree = ast.parse(ground)

    assert "avoid_cylinder_slots = 192" in config
    assert "avoid_capsule_slots = 192" in config
    assert "env_spacing = 24.0" in config
    assert "avoid_clutter_y_jitter = .04" in config
    assert "avoid_stage4_window = 200" in config

    init = function_source(ground, tree, "_init_s_avoid_runtime")
    reset = function_source(ground, tree, "_reset_s_avoid_obstacles")
    progress = function_source(ground, tree, "_get_s_avoid_episode_row_progress_ratios")
    curriculum = function_source(ground, tree, "_update_s_avoid_curriculum")
    termination = function_source(ground, tree, "check_termination")
    reset_idx = function_source(ground, tree, "reset_idx")

    assert "s_avoid_band_exit_y" in init and "(self.num_envs, 5)" in init
    assert "{} if self.s_avoid_clutter_enabled" in init
    assert "layout.coverage" in reset and "band_half_width" not in reset.split("if self.s_avoid_clutter_enabled:", 1)[1].split("return", 1)[0]
    assert "cap_z = 0.5 * cap_h" in reset
    assert "s_avoid_band_count" in reset and "layout.band_exit_y" in reset
    assert "supports at most five obstacle bands" in reset
    assert "free = self.s_avoid_band_count[env_ids] == 0" in progress
    assert "self.s_avoid_exit_y[env_ids] + 1.6" in progress
    assert "episode_success_flags is None" in curriculum
    assert "int(stage) != current_stage" in curriculum
    assert "avoid_stage_rate_source" in curriculum
    assert "avoid_stage_progress_rate" in curriculum
    assert "avoid_stage_completed_episodes" in curriculum
    assert "physical_collision" in termination
    assert "self.reset_buf = collision_now.clone()" in termination
    assert "self.reset_buf |= physical_collision" in termination
    assert "episode_collision_now |= strict_penetration" in termination
    assert "self.s_avoid_episode_fall |= fall_now" in termination
    assert "completed_safety_failure" in reset_idx
    assert "terminal_collision if self.s_avoid_clutter_enabled" in reset_idx
    run_extracted_behavior(tree)
    print("clutter env contract: PASS")


if __name__ == "__main__":
    main()
