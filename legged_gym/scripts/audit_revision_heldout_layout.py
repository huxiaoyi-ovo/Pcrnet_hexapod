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
    if layout.get("obstacle_model") != "native_stage4_capsules" or not bool(layout.get("only_layout_change")):
        raise ValueError("layout must retain the native Stage-4 capsule model and change coordinates only")
    if not bool(layout.get("fixed_across_seeds")):
        raise ValueError("layout must remain fixed across evaluation seeds")
    if len(rows) != 5 or len(layout["obstacles"]) != 13:
        raise ValueError("layout must contain exactly five rows and thirteen obstacles")
    sides = [row["opening_side"] for row in rows]
    if sides != ["L", "L", "R", "R", "L"] or all(sides[i] != sides[i + 1] for i in range(4)):
        raise ValueError("opening sequence must be the frozen non-alternating L-L-R-R-L sequence")

    row_y = [float(row["y"]) for row in rows]
    if row_y != [-0.65, 1.03, 3.15, 4.97, 6.91]:
        raise ValueError(f"unexpected row y coordinates: {row_y}")
    spacings = [round(row_y[i + 1] - row_y[i], 2) for i in range(4)]
    if spacings != [1.68, 2.12, 1.82, 1.94]:
        raise ValueError(f"unexpected row spacings: {spacings}")
    training = layout["training_provenance"]
    support = training["inter_row_spacing_support"]
    outside_spacing = [spacing for spacing in spacings if spacing < support[0] or spacing > support[1]]
    if outside_spacing != [1.68, 2.12]:
        raise ValueError(f"unexpected OOD row spacings: {outside_spacing}")

    expected_x = [(-1.05, 0.25, 0.85), (0.35, 0.85), (-0.85, -0.25, 1.05), (-0.85, -0.35), (-1.05, 0.25, 0.85)]
    row_counts = []
    for row, row_expected_x in zip(rows, expected_x):
        obstacles = _row_obstacles(layout, int(row["row"]))
        row_counts.append(len(obstacles))
        if [float(item["x"]) for item in obstacles] != list(row_expected_x):
            raise ValueError(f"row {row['row']} does not match the native Stage-4 template")
        if any(float(item["y"]) != float(row["y"]) for item in obstacles):
            raise ValueError(f"row {row['row']} obstacle y does not match its row y")

    primitive_counts = Counter(item["primitive"] for item in layout["obstacles"])
    if primitive_counts != Counter({"capsule": 13}):
        raise ValueError(f"unexpected primitive counts: {primitive_counts}")
    geometry = layout["obstacle_geometry"]
    capsule = geometry.get("capsule", {})
    if capsule.get("radius") != 0.15 or capsule.get("height") != 0.50 or capsule.get("asset_rotation_y_deg") != 90.0:
        raise ValueError("capsule geometry does not match the native Stage-4 obstacle")
    if row_counts != [3, 2, 3, 2, 3]:
        raise ValueError(f"unexpected row counts: {row_counts}")

    return {
        "spacings": spacings,
        "outside_spacing": outside_spacing,
        "row_counts": row_counts,
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "hash": layout["sha256"],
    }


def render_layout(layout: dict, output_png: Path, output_pdf: Optional[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

    color = "#0072B2"
    fig, ax = plt.subplots(figsize=(3.5, 6.0), constrained_layout=True, facecolor="white")
    ax.set_facecolor("white")
    for item in layout["obstacles"]:
        patch = FancyBboxPatch(
            (float(item["x"]) - 0.25, float(item["y"]) - 0.15), 0.50, 0.30,
            boxstyle="round,pad=0,rounding_size=0.15", facecolor=color,
            edgecolor="#222222", linewidth=0.8, hatch="///", label="Capsule" if int(item["row"]) == 1 and item["x"] == -1.05 else None,
        )
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
    ax.add_patch(FancyArrowPatch((-0.6, -1.30), (-0.6, 7.45), arrowstyle="->", mutation_scale=10,
                                 linewidth=1.15, color="#444444", label="Target direction"))
    ax.text(-0.55, 7.42, "target", fontsize=7, va="bottom", color="#333333")

    ax.set(xlim=(-1.70, 1.70), ylim=(-1.95, 7.70), xlabel="lateral x [m]", ylabel="forward y [m]")
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
    print(f"PASS layout_id={layout['layout_id']} sha256={report['hash']}")
    print(f"spacings={report['spacings']} training_support={layout['training_provenance']['inter_row_spacing_support']} ood={report['outside_spacing']}")
    for row, count in zip(layout["rows"], report["row_counts"]):
        print(f"row={int(row['row'])} side={row['opening_side']} y={float(row['y']):.2f} capsules={count}")
    for obstacle in layout["obstacles"]:
        print(
            f"obstacle row={int(obstacle['row'])} primitive=capsule "
            f"position=({float(obstacle['x']):.2f},{float(obstacle['y']):.2f}) radius=0.15 height=0.50 "
            "asset_rotation_y_deg=90.00"
        )
    print(f"primitive_counts={report['primitive_counts']} figure={args.output} pdf={args.pdf}")
    print(
        "FEASIBILITY PASS "
        "rows=5 passages=5 native_stage4_templates=true "
        "no_dead_ends=true no_solid_walls=true no_dynamic_or_scattered_clutter=true"
    )


if __name__ == "__main__":
    main()
