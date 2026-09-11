#!/usr/bin/env python3
"""Focused CPU checks for the frozen Avoid observation and 1-D policy contract."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legged_gym.pcr_observation import (  # noqa: E402
    build_avoid_actor_state,
    canonical_difficulty,
    canonical_local_map,
    side_preference,
)
from rsl_rl.algorithms.high_level_planner import CmdVelExpert, GatePolicy  # noqa: E402


def main():
    torch.manual_seed(7)
    occupancy = torch.zeros(1, 32, 32)
    visible = torch.ones_like(occupancy)
    visible[:, 18:, :] = 0.0
    hidden_gt = occupancy.clone()
    hidden_gt[:, 22, 20] = 1.0
    observed = canonical_local_map(occupancy, visible)
    assert torch.equal(observed, canonical_local_map(hidden_gt, visible))
    occupancy[:, 12, 20] = 1.0
    axis_map = canonical_local_map(occupancy, torch.ones_like(occupancy), clearance=0.0)
    assert axis_map[0, 0, 12, 20] == 1.0
    fixture_occ = torch.zeros(1, 9, 9)
    fixture_occ[:, 4, 4] = 1.0
    fixture_vis = torch.ones_like(fixture_occ)
    fixture_map = canonical_local_map(fixture_occ, fixture_vis, extent=3.0, clearance=3.0 / 9.0)
    legacy_occ = fixture_occ * fixture_vis
    pooled = F.max_pool2d(legacy_occ.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    legacy_p = ((pooled <= 0.5) & (legacy_occ < 0.5)).to(dtype=legacy_occ.dtype)
    legacy_soft = torch.maximum(F.avg_pool2d(legacy_p.unsqueeze(1), 3, stride=1, padding=1).squeeze(1), legacy_p) * (1.0 - legacy_occ)
    assert fixture_map[0, 0].sum() == 1.0 and fixture_map[0, 1, 4, 5] < 1.0
    assert torch.allclose(fixture_map, torch.stack([legacy_occ, legacy_soft], dim=1))
    difficulty = canonical_difficulty(observed)
    region = torch.zeros(32, 32, dtype=torch.bool)
    cell = 3.0 / 32.0
    x = torch.linspace(-1.5 + cell / 2.0, 1.5 - cell / 2.0, 32)
    y = torch.linspace(cell / 2.0, 3.0 - cell / 2.0, 32)
    grid_x, grid_y = torch.meshgrid(x, y)
    region = grid_x.square() + grid_y.square() <= 4.0
    manual = 0.5 * observed[0, 0][region].mean() + 0.5 * (1.0 - observed[0, 1][region]).mean()
    assert torch.allclose(difficulty[0], manual)

    raw = torch.arange(28, dtype=torch.float32).reshape(2, 14)
    actor_state = build_avoid_actor_state(raw, torch.tensor([0.2, 0.4]))
    assert torch.equal(actor_state[:, [0, 1, 2, 6, 9]], torch.zeros(2, 5))
    assert torch.equal(actor_state[:, 3:6], raw[:, 3:6]) and torch.equal(actor_state[:, 10:13], raw[:, 10:13])
    assert torch.allclose(actor_state[:, 13], torch.tensor([0.2, 0.4]))
    actor_state_13 = build_avoid_actor_state(raw[:, :13], 0.3)
    assert actor_state_13.shape == (2, 14) and torch.allclose(actor_state_13[:, 13], torch.full((2,), 0.3))
    preference = side_preference(torch.tensor([[3.0, 4.0], [1.0, -1.0], [0.0, 0.0]]))
    assert torch.allclose(preference[0], torch.tensor([0.6, 0.0])) and torch.equal(preference[1:], torch.zeros(2, 2))
    for bad_goal, bad_distance in ((torch.tensor([[float("nan"), 1.0]]), 1e-3), (torch.tensor([[1.0, 1.0]]), 0.0)):
        try:
            side_preference(bad_goal, min_distance=bad_distance)
            raise AssertionError("side_preference accepted invalid input")
        except ValueError:
            pass

    affordance = torch.zeros(2, 2, 16, 16)
    state = torch.randn(2, 14)
    changed_state = state.clone()
    changed_state[:, :2] += 10.0
    goal = torch.zeros(2, 2)
    difficulty_input = torch.zeros(2)
    policy = CmdVelExpert(affordance_channels=2, state_dim=14, action_dim=1, cmd_scale=(0.5,), actor_state_mask_indices=(0, 1, 2, 6, 9))
    first = policy.forward(affordance, state, goal, difficulty_input)
    second = policy.forward(affordance, changed_state, goal, difficulty_input)
    assert torch.allclose(first.cmd_mean, second.cmd_mean) and not torch.allclose(first.value, second.value)
    action, _ = policy.get_action(affordance, state, goal, difficulty_input)
    log_prob, value, entropy, _ = policy.evaluate_actions(affordance, state, goal, difficulty_input, action)
    assert action.shape == (2, 1) and log_prob.shape == entropy.shape == (2,) and value.shape == (2, 1)

    legacy = CmdVelExpert(affordance_channels=2, state_dim=14)
    assert legacy.forward(affordance, state, goal, difficulty_input).cmd_mean.shape == (2, 3)

    gate = GatePolicy(affordance_channels=2, state_dim=14, actor_mask_xy=True)
    gate_first = gate.forward(affordance, state, goal, difficulty_input)
    gate_second = gate.forward(affordance, changed_state, goal, difficulty_input)
    assert torch.allclose(gate_first.y_alpha, gate_second.y_alpha) and not torch.allclose(gate_first.value, gate_second.value)
    print("revision observation contract: PASS")


if __name__ == "__main__":
    main()
