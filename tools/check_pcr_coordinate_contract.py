#!/usr/bin/env python3
"""CPU regression for the PCR Follow body/world coordinate contract."""

import argparse
import ast
import math
from pathlib import Path
from typing import Any

import torch


def load_function(source: Path, name: str, namespace):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parents[1] / "legged_gym" / "scripts" / "train_highlevel.py",
    )
    parser.add_argument(
        "--expert-source",
        type=Path,
        default=Path(__file__).parents[1] / "legged_gym" / "envs" / "hex_v4" / "expert_s0_follow.py",
    )
    args = parser.parse_args()
    helper = load_function(args.source, "get_follow_target_world_xy", {"torch": torch, "Any": Any})
    world_to_body = load_function(args.expert_source, "_world_to_body_xy", {"torch": torch})
    generator = torch.Generator().manual_seed(20260908)
    canonical_heading = torch.tensor(
        [0.0, math.radians(5.0), math.radians(-5.0), math.radians(45.0), math.radians(-45.0), math.pi / 2.0, -math.pi / 2.0]
    )
    random_heading = (torch.rand(16, generator=generator) * 2.0 - 1.0) * math.pi
    heading = torch.cat([canonical_heading, random_heading])
    state = torch.zeros(heading.numel(), 3, dtype=torch.float32)
    state[:, :2] = torch.randn(heading.numel(), 2, generator=generator)
    state[:, 2] = heading
    goal = torch.randn(heading.numel(), 2, generator=generator)
    goal[: canonical_heading.numel()] = torch.tensor([0.0, 2.0])
    target_world = helper(None, state, goal)
    recovered_goal = world_to_body(
        target_world - state[:, :2],
        cos_heading=torch.cos(heading),
        sin_heading=torch.sin(heading),
    )
    if not torch.allclose(recovered_goal, goal, atol=1e-6, rtol=0.0):
        raise SystemExit(
            "PCR Follow round-trip failed: body goal was not preserved\n"
            "recovered={}\nexpected={}".format(recovered_goal, goal)
        )
    canonical_bearing = torch.atan2(
        recovered_goal[: canonical_heading.numel(), 0],
        recovered_goal[: canonical_heading.numel(), 1],
    )
    if not torch.allclose(canonical_bearing, torch.zeros_like(canonical_bearing), atol=1e-6, rtol=0.0):
        raise SystemExit("PCR Follow heading invariance failed: front goal has nonzero bearing")

    anchor_state = torch.tensor([[3.0, 4.0, math.pi / 2.0]], dtype=torch.float32)
    anchor_goal = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
    anchor_target = helper(None, anchor_state, anchor_goal)
    anchor_expected = torch.tensor([[1.0, 4.0]], dtype=torch.float32)
    if not torch.allclose(anchor_target, anchor_expected, atol=1e-6, rtol=0.0):
        raise SystemExit(
            "PCR Follow +90deg anchor failed: target_world={} expected={}".format(anchor_target, anchor_expected)
        )
    print("PCR Follow coordinate contract OK ({})".format(args.source))


if __name__ == "__main__":
    main()
