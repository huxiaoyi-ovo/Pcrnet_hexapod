#!/usr/bin/env python3
"""Export top-down views of held-out irregular-row stage-4 layouts."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D


def generate_heldout_stage4(seed: int, terrain_width: float = 12.0):
    stage = 4
    row_y = (0.60, 5.60, 10.60, 15.60, 20.60)
    row_counts = (3, 2, 3, 2, 3)
    capsules = []
    boxes = []
    rng = np.random.RandomState(int(seed) + 910003 + 1009 * stage)
    terrain_half_width = 0.5 * float(terrain_width)
    spawn_margin = 0.2
    x_limit = max(1.80, terrain_half_width - spawn_margin - 0.30)
    gap_x_limit = max(1.55, min(x_limit, 4.20))

    def add_random_obstacle(x: float, y: float, yaw_deg: float) -> None:
        if rng.rand() < 0.5:
            capsules.append((float(x), float(y)))
        else:
            boxes.append((float(x), float(y), float(yaw_deg)))

    for row_idx, (local_y, row_count) in enumerate(zip(row_y, row_counts)):
        left_wide = row_idx in (0, 1, 4)
        if int(row_count) == 3:
            row_x = (-1.65, -0.70, -0.10) if left_wide else (-1.30, -0.70, 0.60)
            if row_idx == 0:
                capsules.extend((float(x), float(local_y)) for x in row_x)
            else:
                boxes.extend(
                    (float(x), float(local_y), float(((-5.0, 7.0, -6.0)[i])))
                    for i, x in enumerate(row_x)
                )
        elif int(row_count) == 2:
            row_x = (-1.65, -0.70) if left_wide else (-1.30, -0.70)
            boxes.extend(
                (float(x), float(local_y), float((6.0, -6.0)[i]))
                for i, x in enumerate(row_x)
            )
        else:
            raise RuntimeError(f"Unsupported row count={int(row_count)}")

    for gap_idx in range(max(0, len(row_y) - 1)):
        y0 = float(row_y[gap_idx])
        y1 = float(row_y[gap_idx + 1])
        gap_count = int(rng.randint(3, 7))
        for _ in range(gap_count):
            gap_x = float(rng.uniform(-gap_x_limit, gap_x_limit))
            gap_y = float(rng.uniform(y0 + 0.75, y1 - 0.75))
            gap_yaw = float(rng.uniform(-16.0, 16.0))
            add_random_obstacle(gap_x, gap_y, gap_yaw)
    return capsules, boxes


def draw_layout(seed: int, output_dir: Path, terrain_width: float, terrain_length: float) -> None:
    capsules, boxes = generate_heldout_stage4(seed, terrain_width=terrain_width)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 11.0))
    half_w = 0.5 * float(terrain_width)
    ax.add_patch(
        Rectangle(
            (-half_w, -2.0),
            terrain_width,
            terrain_length,
            facecolor="#fafafa",
            edgecolor="#303030",
            linewidth=1.4,
        )
    )

    for x, y in capsules:
        ax.add_patch(
            Circle(
                (x, y),
                radius=0.15,
                facecolor="#bfc3c7",
                edgecolor="#555555",
                linewidth=0.8,
                alpha=0.95,
            )
        )

    box_w = 0.34
    box_h = 0.34
    for x, y, yaw_deg in boxes:
        patch = Rectangle(
            (-0.5 * box_w, -0.5 * box_h),
            box_w,
            box_h,
            facecolor="#d6c2a4",
            edgecolor="#5d5146",
            linewidth=0.8,
            alpha=0.95,
        )
        transform = Affine2D().rotate_deg(float(yaw_deg)).translate(float(x), float(y)) + ax.transData
        patch.set_transform(transform)
        ax.add_patch(patch)

    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(-2.0, 23.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x / lateral [m]")
    ax.set_ylabel("world y / forward [m]")
    ax.set_title(f"held-out irregular rows, stage 4, seed {seed}")
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    png_path = output_dir / f"heldout_stage4_topdown_seed{seed}.png"
    pdf_path = output_dir / f"heldout_stage4_topdown_seed{seed}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[heldout-topdown] seed={seed} capsules={len(capsules)} boxes={len(boxes)} -> {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="agents/final_paper_outputs_v3/heldout_layout_topdown")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7001, 7002, 7003, 7004, 7005, 7006])
    parser.add_argument("--terrain_width", type=float, default=12.0)
    parser.add_argument("--terrain_length", type=float, default=28.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for seed in args.seeds:
        draw_layout(int(seed), output_dir, float(args.terrain_width), float(args.terrain_length))


if __name__ == "__main__":
    main()
