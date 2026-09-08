#!/usr/bin/env python3
"""CPU regression for the historical Avoid 14D high-level state contract.

Only the state-building prefix of _get_high_level_obs is AST-extracted.  The fixture
provides robot_state_buf, prev_gate_y, last_cmd, and forced_forward_speed; goal and
all map/camera paths are intentionally outside this test's scope.
"""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch


def _is_goal_assignment(node):
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        for target in child.targets:
            if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
                continue
            if target.value.id != "obs_dict":
                continue
            key = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
            if isinstance(key, ast.Constant) and key.value == "goal":
                return True
    return False


def load_state_prefix(source):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HierarchicalHexapodEnv")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_get_high_level_obs")
    cut = next(index for index, node in enumerate(method.body) if _is_goal_assignment(node))
    return_state = ast.parse("return obs_dict['state']").body[0]
    prefix = ast.FunctionDef(
        name="build_state_prefix",
        args=method.args,
        body=method.body[:cut] + [return_state],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.Module(body=[prefix], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "Dict": Dict}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["build_state_prefix"]


class Harness:
    def __init__(self, skill, s_avoid_enabled, mono_ppo=False):
        self.args = SimpleNamespace(skill=skill, mono_ppo=mono_ppo)
        self.env = SimpleNamespace(
            robot_state_buf=torch.arange(18, dtype=torch.float32).view(2, 9),
            s_avoid_enabled=s_avoid_enabled,
        )
        self.reward_cfg = None
        self.prev_gate_y = torch.tensor([0.25, 0.75])
        self.post_processor = SimpleNamespace(last_cmd=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.forced_forward_speed = torch.tensor([0.31, 0.47])


def expected_prefix(harness):
    return torch.cat(
        [harness.env.robot_state_buf, harness.prev_gate_y.unsqueeze(1), harness.post_processor.last_cmd], dim=1
    )


def assert_case(build_state_prefix, skill, enabled, expected_dim, label, mono_ppo=False):
    harness = Harness(skill, enabled, mono_ppo=mono_ppo)
    state = build_state_prefix(harness)
    baseline = expected_prefix(harness)
    if state.shape != (2, expected_dim):
        raise AssertionError("{}: got state shape {}, expected (2, {})".format(label, tuple(state.shape), expected_dim))
    if not torch.equal(state[:, :13], baseline):
        raise AssertionError("{}: the original 13 state columns changed".format(label))
    if expected_dim == 14 and not torch.equal(state[:, 13], harness.forced_forward_speed):
        raise AssertionError("{}: forced-forward speed is not the final Avoid state column".format(label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("train_highlevel.py"))
    args = parser.parse_args()
    build_state_prefix = load_state_prefix(args.source)
    assert_case(build_state_prefix, "avoid", True, 14, "Avoid")
    assert_case(build_state_prefix, "moe", True, 13, "Gate")
    assert_case(build_state_prefix, "moe", True, 13, "Mono", mono_ppo=True)
    assert_case(build_state_prefix, "avoid", False, 13, "non-avoid scene")
    assert_case(build_state_prefix, "follow", True, 13, "Follow")
    print("PASS: Avoid high-level state contract ({})".format(args.source))


if __name__ == "__main__":
    main()
