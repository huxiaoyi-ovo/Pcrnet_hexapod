#!/usr/bin/env python3
"""Validate and render the frozen RA-L revision held-out layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT = PROJECT_ROOT / "legged_gym/envs/hex_v4/layouts/revision_heldout_mixed_v1.json"
DEFAULT_FIGURE = PROJECT_ROOT / "docs/figures/revision_heldout_mixed_v1_topdown.png"


def _canonical_sha256(layout: dict) -> str:
    payload = {key: value for key, value in layout.items() if key != "sha256"}
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_obstacles(layout: dict, row: int) -> list[dict]:
    return sorted((item for item in layout["obstacles"] if int(item["row"]) == row), key=lambda item: item["x"])


def validate_layout(layout: dict) -> dict:
    if layout.get("layout_id") != "revision_heldout_mixed_v1":
        raise ValueError("layout_id is not revision_heldout_mixed_v1")
    if layout.get("sha256") != _canonical_sha256(layout):
        raise ValueError("stored SHA256 does not match the canonical payload")

    rows = layout["rows"]
    if len(rows) != 5 or len(layout["obstacles"]) != 15:
        raise ValueError("layout must contain exactly five rows and fifteen obstacles")
    sides = [row["opening_side"] for row in rows]
    if sides != ["L", "L", "R", "R", "L"] or all(sides[i] != sides[i + 1] for i in range(4)):
        raise ValueError("opening sequence must be the frozen non-alternating L-L-R-R-L sequence")

    row_y = [float(row["y"]) for row in rows]
    spacings = [round(row_y[i + 1] - row_y[i], 2) for i in range(4)]
    if spacings != [1.68, 2.12, 1.82, 1.94]:
        raise ValueError(f"unexpected row spacings: {spacings}")
    training = layout["training_provenance"]
    support = training["inter_row_spacing_support"]
    outside_spacing = [spacing for spacing in spacings if spacing < support[0] or spacing > support[1]]
    if outside_spacing != [1.68, 2.12]:
        raise ValueError(f"unexpected OOD row spacings: {outside_spacing}")

    diameter = 0.34
    radius = 0.5 * diameter
    measured_gaps = []
    secondary_gaps = []
    for row in rows:
        obstacles = _row_obstacles(layout, int(row["row"]))
        if len(obstacles) != 3:
            raise ValueError(f"row {row['row']} does not have three obstacles")
        gaps = [
            float(obstacles[index + 1]["x"]) - float(obstacles[index]["x"]) - diameter
            for index in range(2)
        ]
        opening_index = 0 if row["opening_side"] == "L" else 1
        measured_gaps.append(round(gaps[opening_index], 2))
        secondary_gaps.append(round(gaps[1 - opening_index], 2))
        center = 0.5 * (float(obstacles[opening_index]["x"]) + float(obstacles[opening_index + 1]["x"]))
        if abs(center - float(row["gap_center_x"])) > 1e-9:
            raise ValueError(f"row {row['row']} gap center does not match its footprints")
    expected_widths = [float(row["gap_width"]) for row in rows]
    if measured_gaps != [round(width, 2) for width in expected_widths]:
        raise ValueError(f"gap widths do not match footprints: {measured_gaps}")
    if secondary_gaps != [0.2] * 5:
        raise ValueError(f"secondary gaps must remain 0.20 m: {secondary_gaps}")

    primitive_counts = Counter(item["primitive"] for item in layout["obstacles"])
    if primitive_counts != Counter({"cube": 5, "cylinder": 5, "sphere": 5}):
        raise ValueError(f"unexpected primitive counts: {primitive_counts}")
    geometry = layout["obstacle_geometry"]
    if geometry["cube"]["side"] != diameter or geometry["cylinder"]["radius"] != radius or geometry["sphere"]["radius"] != radius:
        raise ValueError("primitive footprints are not comparable")

    centers = [float(row["gap_center_x"]) for row in rows]
    lateral_transitions = [abs(centers[i + 1] - centers[i]) for i in range(4)]
    if any(delta > 1.02 for delta in lateral_transitions):
        raise ValueError("a consecutive opening transition exceeds the fixed feasibility bound")
    if any(width <= 0.54 for width in expected_widths):
        raise ValueError("a main passage is below the fixed geometric clearance bound")

    return {
        "spacings": spacings,
        "outside_spacing": outside_spacing,
        "measured_gaps": measured_gaps,
        "secondary_gaps": secondary_gaps,
        "lateral_transitions": [round(value, 2) for value in lateral_transitions],
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "hash": layout["sha256"],
    }


def render_layout(layout: dict, output_png: Path, output_pdf: Optional[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    colors = {"cube": "#0072B2", "cylinder": "#D55E00", "sphere": "#009E73"}
    hatches = {"cube": "///", "cylinder": "...", "sphere": "xx"}
    labels_drawn = set()
    fig, ax = plt.subplots(figsize=(3.5, 6.0), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    radius = 0.17
    for item in layout["obstacles"]:
        kind = item["primitive"]
        label = kind if kind not in labels_drawn else None
        labels_drawn.add(kind)
        if kind == "cube":
            patch = Rectangle((item["x"] - radius, item["y"] - radius), 0.34, 0.34,
                              facecolor=colors[kind], edgecolor="#222222", linewidth=0.8,
                              hatch=hatches[kind], label=label)
        else:
            patch = Circle((item["x"], item["y"]), radius, facecolor=colors[kind],
                           edgecolor="#222222", linewidth=0.8, hatch=hatches[kind], label=label)
        ax.add_patch(patch)

    for index, row in enumerate(layout["rows"]):
        y = float(row["y"])
        ax.axhline(y, color="#BDBDBD", linewidth=0.65, zorder=0)
        ax.text(-1.57, y + 0.10, f"Row {index + 1} ({row['opening_side']})", fontsize=7, va="bottom")
        if index:
            previous_y = float(layout["rows"][index - 1]["y"])
            ax.annotate(f"{y - previous_y:.2f} m", xy=(1.53, 0.5 * (previous_y + y)),
                        ha="right", va="center", fontsize=7, color="#444444")

    start_x, start_y = layout["coordinate_convention"]["start_xy"]
    ax.add_patch(Circle((start_x, start_y), 0.19, facecolor="#666666", edgecolor="#111111", linewidth=0.8, label="Robot start"))
    ax.add_patch(FancyArrowPatch((-0.6, -1.30), (-0.6, 8.92), arrowstyle="->", mutation_scale=10,
                                 linewidth=1.15, color="#444444", label="Target direction"))
    ax.text(-0.55, 8.88, "target", fontsize=7, va="bottom", color="#333333")

    ax.set(xlim=(-1.70, 1.70), ylim=(-1.95, 9.15), xlabel="lateral x [m]", ylabel="forward y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=6.5,
        frameon=True,
        edgecolor="#888888",
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, facecolor="white", transparent=False)
    from PIL import Image

    with Image.open(output_png) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(output_png, dpi=(600, 600))
    if output_pdf is not None:
        fig.savefig(output_pdf, facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_FIGURE.with_suffix(".pdf"))
    args = parser.parse_args()
    with args.layout.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)
    report = validate_layout(layout)
    render_layout(layout, args.output, args.pdf)
    training = layout["training_provenance"]
    center_support = training["gap_center_support"]
    width_support = training["gap_width_support"]
    print(f"PASS layout_id={layout['layout_id']} sha256={report['hash']}")
    print(f"spacings={report['spacings']} training_support={layout['training_provenance']['inter_row_spacing_support']} ood={report['outside_spacing']}")
    print(
        "training_gap_support "
        f"left_center={center_support['left']} right_center={center_support['right']} "
        f"width={width_support}"
    )
    for row in layout["rows"]:
        side = str(row["opening_side"])
        center = float(row["gap_center_x"])
        width = float(row["gap_width"])
        center_range = center_support["left" if side == "L" else "right"]
        center_status = "in_training_support" if center_range[0] <= center <= center_range[1] else "OOD"
        width_status = "in_training_support" if width_support[0] <= width <= width_support[1] else "OOD"
        print(
            f"row={int(row['row'])} side={side} gap_center={center:.2f} "
            f"center_support={center_range} center_status={center_status} "
            f"gap_width={width:.2f} width_support={width_support} width_status={width_status}"
        )
    print(f"secondary_gaps={report['secondary_gaps']} lateral_transitions={report['lateral_transitions']}")
    for obstacle in layout["obstacles"]:
        primitive = str(obstacle["primitive"])
        size = layout["obstacle_geometry"][primitive]
        size_text = ",".join(f"{key}={float(value):.2f}" for key, value in sorted(size.items()))
        print(
            f"obstacle row={int(obstacle['row'])} primitive={primitive} "
            f"position=({float(obstacle['x']):.2f},{float(obstacle['y']):.2f}) size={size_text}"
        )
    print(f"primitive_counts={report['primitive_counts']} figure={args.output} pdf={args.pdf}")
    print(
        "FEASIBILITY PASS "
        "rows=5 passages=5 lateral_transitions_within_bound=true "
        "no_dead_ends=true no_solid_walls=true no_dynamic_or_scattered_clutter=true"
    )


if __name__ == "__main__":
    main()
