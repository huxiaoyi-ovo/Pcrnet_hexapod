#!/usr/bin/env python3
"""Pure-Torch checks for revised Avoid checkpoint loading and PCR reuse."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from legged_gym.pcr_policy_contract import checkpoint_cmd_policy_kwargs, get_avoid_command, validate_frozen_avoid_revision
from rsl_rl.algorithms.high_level_planner import CmdVelExpert


def _checkpoint(model, meta):
    return {"model_state_dict": model.state_dict(), "experiment_meta": meta}


def main():
    torch.manual_seed(11)
    eval_source = (Path(__file__).resolve().parent / "eval_highlevel.py").read_text()
    play_source = (Path(__file__).resolve().parent / "play_highlevel.py").read_text()
    assert "actor_mask_xy=self.policy_meta.get(\"revision_contract\")" in eval_source
    assert "actor_mask_xy=gate_meta.get(\"revision_contract\")" not in eval_source
    assert '"s_avoid_clutter"' in play_source
    assert 'args.skill = "avoid" if str(getattr(args, "task", "")) == "s_avoid_clutter" else "follow"' in play_source
    assert "policy_state[:, 13:14]" in play_source
    assert "Standalone clutter already supplies frozen 14-D" in play_source
    one = CmdVelExpert(affordance_channels=2, state_dim=14, goal_dim=2, action_dim=1, cmd_scale=(.6,), actor_state_mask_indices=(0, 1, 2, 6, 9))
    meta = {"revision_contract": "avoid_1d_canonical_v1", "actor_output_dim": 1, "state_dim": 14, "policy_goal_dim": 2, "actor_state_mask_indices": [0, 1, 2, 6, 9]}
    assert checkpoint_cmd_policy_kwargs(_checkpoint(one, meta), (.6, .6, 1.0))["action_dim"] == 1
    legacy = CmdVelExpert(affordance_channels=2, state_dim=13, goal_dim=2, cmd_scale=(.6, .6, 1.0))
    assert checkpoint_cmd_policy_kwargs(_checkpoint(legacy, {}), (.6, .6, 1.0))["action_dim"] == 3
    try:
        checkpoint_cmd_policy_kwargs(_checkpoint(one, {}), (.6, .6, 1.0))
        raise AssertionError("ambiguous 1-D checkpoint accepted")
    except ValueError:
        pass
    validate_frozen_avoid_revision({}, {})
    validate_frozen_avoid_revision({"revision_contract": "pcr_canonical_v1"}, {"revision_contract": "avoid_1d_canonical_v1"})
    for primary_meta, avoid_meta in (({}, meta), ({"revision_contract": "pcr_canonical_v1"}, {})):
        try:
            validate_frozen_avoid_revision(primary_meta, avoid_meta)
            raise AssertionError("mixed revision pair accepted")
        except ValueError:
            pass
    raw = torch.randn(2, 13)
    follow = torch.tensor([[.1, .37, .2], [-.1, .24, -.2]])
    follow_goal = torch.tensor([[1., 2.], [-1., 3.]])
    cmd, _ = get_avoid_command(one, torch.zeros(2, 2, 16, 16), raw, follow, follow_goal, torch.zeros(2))
    assert cmd.shape == (2, 3) and torch.equal(cmd[:, 1:], torch.zeros(2, 2))
    captured = {}
    original = one.get_action
    def spy(aff, state, goal, difficulty, deterministic=True):
        captured["drive"] = state[:, 13].clone(); captured["goal"] = goal.clone()
        return original(aff, state, goal, difficulty, deterministic=deterministic)
    one.get_action = spy
    get_avoid_command(one, torch.zeros(2, 2, 16, 16), raw, follow, follow_goal, torch.zeros(2))
    assert torch.equal(captured["drive"], follow[:, 1]) and torch.allclose(captured["goal"][:, 0], follow_goal[:, 0] / torch.norm(follow_goal, dim=1))
    get_avoid_command(one, torch.zeros(2, 2, 16, 16), raw, follow, follow_goal, torch.zeros(2), target_valid=torch.tensor([False, True]))
    assert captured["goal"][0, 0] == 0.0
    print("revision policy reuse: PASS")


if __name__ == "__main__":
    main()
