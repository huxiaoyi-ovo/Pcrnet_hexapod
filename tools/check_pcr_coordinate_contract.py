#!/usr/bin/env python3
"""Check PCR goal_buf coordinate convention.

Project contract:
  - heading=0: body forward is world +Y, body right is world +X.
  - goal_buf = (x_right, y_forward).
"""

import math

import torch


def goal_buf_to_world_delta(goal: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    x_right = goal[:, 0]
    y_forward = goal[:, 1]
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    world_x = cos_h * x_right + sin_h * y_forward
    world_y = -sin_h * x_right + cos_h * y_forward
    return torch.stack([world_x, world_y], dim=1)


def main() -> None:
    goal = torch.tensor(
        [
            [0.0, 1.0],  # straight ahead at heading 0 -> world +Y
            [0.0, 1.0],  # straight ahead at heading +90deg -> world +X
            [1.0, 0.0],  # body right at heading +90deg -> world -Y
        ],
        dtype=torch.float32,
    )
    heading = torch.tensor([0.0, math.pi / 2.0, math.pi / 2.0], dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    actual = goal_buf_to_world_delta(goal, heading)
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise SystemExit(f"PCR coordinate contract failed:\nactual={actual}\nexpected={expected}")
    print("PCR coordinate contract OK")


if __name__ == "__main__":
    main()
