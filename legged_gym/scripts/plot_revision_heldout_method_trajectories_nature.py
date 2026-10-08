#!/usr/bin/env python3
"""Render a refined Nature-style overview of held-out method trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from PIL import Image

import plot_revision_heldout_method_trajectories as source_plot


FIG_WIDTH_MM = 110.9
FIG_HEIGHT_MM = 87.0
FIGURE_SIZE_IN = (FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4)
AXES_BOX = [0.105, 7.9 / FIG_HEIGHT_MM, 0.880, 66.5 / FIG_HEIGHT_MM]
PCR_LINEWIDTH = 1.35
BASELINE_LINEWIDTH = 0.95
BASELINE_ALPHA = 0.62
TARGET_COLOR = "#23866F"
OBSTACLE_FACE = "#D8DADC"
OBSTACLE_EDGE = "#A7AAAC"
OBSTACLE_MARKER_DIAMETER_PT = 11.0

NATURE_STYLES = {
    "pcr": {"color": "#174A7E", "label": "Adaptive"},
    "rule_override": {"color": "#5E9272", "label": "Rule-Override"},
    "additive_fusion": {"color": "#8069A6", "label": "Additive-Fusion"},
    "fixed_authority": {"color": "#A66B62", "label": "Fixed-Authority"},
    "geomw": {"color": "#438F98", "label": "Geom-w"},
    "risk_only": {"color": "#BF8538", "label": "Risk-only"},
    "yonly": {"color": "#747474", "label": r"$\alpha$-only"},
}


def _terminal_marker(
    ax,
    x: float,
    y: float,
    reason: str,
    color: str,
) -> None:
    if reason == "success":
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=78,
            facecolor=NATURE_STYLES["pcr"]["color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=22,
            clip_on=False,
        )
        return
    if reason == "collision":
        ax.scatter(
            [x],
            [y],
            marker="X",
            s=30,
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=22,
            clip_on=False,
        )
        return
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=30,
        facecolor="white",
        edgecolor=color,
        linewidth=1.05,
        alpha=0.92,
        zorder=22,
        clip_on=False,
    )


def _add_direction_arrows(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    color: str,
) -> None:
    valid = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(valid) < 8:
        return
    for fraction in (0.28, 0.56, 0.82):
        index = int(round(fraction * (len(valid) - 2)))
        x0, y0 = valid[index]
        x1, y1 = valid[index + 1]
        if abs(x1 - x0) + abs(y1 - y0) < 1e-5:
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 0.85,
                "mutation_scale": 6.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=7,
        )


def _nature_manifest(
    layout_path: Path,
    layout: Dict,
    tracks: Dict[str, Dict],
    svg_path: Path,
    pdf_path: Path,
    png_path: Path,
    tiff_path: Path,
    plot_xlim: Sequence[float],
    plot_ylim: Sequence[float],
) -> Dict:
    manifest = source_plot._build_manifest(
        layout_path,
        layout,
        tracks,
        png_path,
        pdf_path,
        plot_xlim,
        plot_ylim,
    )
    manifest["schema_version"] = 3
    manifest["artifact_role"] = "nature_refined_single_overview_trajectory_comparison"
    manifest["adaptation"] = {
        "reuse_level": "style-only inheritance",
        "scientific_mapping": "trajectory data and layout geometry unchanged; obstacles use their source layout centers",
        "source_plotter": source_plot._relative(
            Path(__file__).with_name("plot_revision_heldout_method_trajectories.py")
        ),
        "source_plotter_sha256": source_plot._sha256(
            Path(__file__).with_name("plot_revision_heldout_method_trajectories.py")
        ),
        "nature_plotter": source_plot._relative(Path(__file__)),
        "nature_plotter_sha256": source_plot._sha256(Path(__file__)),
    }
    manifest["transformations"] = [
        "reused the exact deterministic episode-selection rule from the source plot",
        "subtracted each selected environment world origin to recover layout-local coordinates",
        "omitted the final post-reset sentinel row from each trajectory",
        "used the last pre-reset sample as the terminal-marker location",
        "rotated the display so forward is horizontal",
        "used independent forward and lateral display scales to expand the forward axis",
        "rendered obstacles as fixed-size screen-circular glyphs centered at their source layout coordinates",
        "used a moderately emphasized PCR line without a masking halo",
        "no trajectory coordinate translation, smoothing, interpolation, resampling, or reselection",
    ]
    for track_id, item in manifest["trajectories"].items():
        method = item["method"]
        is_pcr = method == "pcr"
        item["style"] = {
            "color": NATURE_STYLES[method]["color"],
            "linestyle": "-",
            "linewidth": PCR_LINEWIDTH if is_pcr else BASELINE_LINEWIDTH,
            "alpha": 1.0 if is_pcr else BASELINE_ALPHA,
            "white_halo_linewidth": None,
        }

    manifest["figure"].update(
        {
            "svg": source_plot._relative(svg_path),
            "svg_sha256": source_plot._sha256(svg_path),
            "pdf": source_plot._relative(pdf_path),
            "pdf_sha256": source_plot._sha256(pdf_path),
            "png": source_plot._relative(png_path),
            "png_sha256": source_plot._sha256(png_path),
            "tiff": source_plot._relative(tiff_path),
            "tiff_sha256": source_plot._sha256(tiff_path),
            "width_mm": FIG_WIDTH_MM,
            "size_inches": list(FIGURE_SIZE_IN),
            "png_dpi": 600,
            "axes_box_fraction": AXES_BOX,
            "style_profile": "Nature-style refined single overview",
            "font_family": "Arial/Helvetica/DejaVu Sans fallback",
            "minimum_intended_font_size_pt": 5.0,
            "spines": "left and bottom only",
            "grid": "none",
            "panel_alignment": "not applicable: one axes panel",
            "legend_layout": (
                "three centered rows: four primary methods, three remaining "
                "methods, then five semantic glyphs"
            ),
            "row_spacing_annotation": (
                "omitted from this rendering; exact adjacent-row values remain "
                "recorded in layout.row_spacing_m"
            ),
            "axis_aspect": "independent forward and lateral display scales",
            "visual_hierarchy": (
                "Adaptive uses a deep-blue 1.35 pt line without a halo; baselines "
                "use restrained colors, 0.95 pt lines, and alpha 0.62"
            ),
            "obstacle_glyph": "fixed-size screen-circular glyph centered at source layout coordinates",
            "obstacle_glyph_diameter_pt": OBSTACLE_MARKER_DIAMETER_PT,
            "event_glyphs": {
                "start": "black square with white edge",
                "success": "deep-blue star with white edge",
                "collision": "method-color X with white edge",
                "follow_lost_timeout": "white-filled method-color circle",
                "target": "restrained-green dashed line with direction arrows",
            },
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--layout_json", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    with args.layout_json.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)
    method_datasets = {
        method: source_plot._load_method(args.input_root, method)
        for method in source_plot.METHODS
    }
    layout_hashes = {
        data["protocol"].get("layout_sha256") for data in method_datasets.values()
    }
    if layout_hashes != {layout.get("sha256")}:
        raise RuntimeError(
            f"layout hash mismatch: eval={layout_hashes}, JSON={layout.get('sha256')}"
        )

    tracks = source_plot._select_representative_tracks(method_datasets, layout)
    obstacle_radius = float(layout["obstacle_geometry"]["capsule"]["radius"])

    forward_values: List[float] = []
    lateral_values: List[float] = []
    for descriptor in tracks.values():
        for row in descriptor["dataset"]["plot_rows"]:
            forward_values.append(float(row["robot_y_local"]))
            lateral_values.append(float(row["robot_x_local"]))
    for item in layout["obstacles"]:
        lateral = float(item["x"])
        forward = float(item["y"])
        forward_values.extend([forward - obstacle_radius, forward + obstacle_radius])
        lateral_values.extend([lateral - obstacle_radius, lateral + obstacle_radius])
    plot_xlim = (min(forward_values) - 0.10, max(forward_values) + 0.10)
    plot_ylim = (min(lateral_values) - 0.10, max(lateral_values) + 0.10)
    fig = plt.figure(figsize=FIGURE_SIZE_IN, facecolor="white")
    ax = fig.add_axes(AXES_BOX)

    for item in layout["obstacles"]:
        lateral = float(item["x"])
        forward = float(item["y"])
        ax.scatter(
            [forward],
            [lateral],
            marker="o",
            s=OBSTACLE_MARKER_DIAMETER_PT ** 2,
            facecolor=OBSTACLE_FACE,
            edgecolor=OBSTACLE_EDGE,
            linewidth=0.46,
            alpha=0.90,
            zorder=1,
        )

    pcr_rows = tracks["pcr"]["dataset"]["plot_rows"]
    target_forward = [float(row["target_y_local"]) for row in pcr_rows]
    target_lateral = [float(row["target_x_local"]) for row in pcr_rows]
    ax.plot(
        target_forward,
        target_lateral,
        color=TARGET_COLOR,
        linewidth=1.05,
        linestyle=(0, (4.8, 3.0)),
        alpha=0.90,
        zorder=2,
    )
    _add_direction_arrows(ax, target_forward, target_lateral, TARGET_COLOR)

    for track_id in source_plot.TRACK_ORDER:
        if track_id == "pcr":
            continue
        descriptor = tracks[track_id]
        method = descriptor["method"]
        style = NATURE_STYLES[method]
        rows = descriptor["dataset"]["plot_rows"]
        forward = [float(row["robot_y_local"]) for row in rows]
        lateral = [float(row["robot_x_local"]) for row in rows]
        ax.plot(
            forward,
            lateral,
            color=style["color"],
            linewidth=BASELINE_LINEWIDTH,
            alpha=BASELINE_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
        _terminal_marker(
            ax,
            forward[-1],
            lateral[-1],
            descriptor["dataset"]["termination"],
            style["color"],
        )

    pcr_descriptor = tracks["pcr"]
    pcr_forward = [
        float(row["robot_y_local"])
        for row in pcr_descriptor["dataset"]["plot_rows"]
    ]
    pcr_lateral = [
        float(row["robot_x_local"])
        for row in pcr_descriptor["dataset"]["plot_rows"]
    ]
    pcr_line = ax.plot(
        pcr_forward,
        pcr_lateral,
        color=NATURE_STYLES["pcr"]["color"],
        linewidth=PCR_LINEWIDTH,
        alpha=1.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=8,
    )[0]
    _terminal_marker(
        ax,
        pcr_forward[-1],
        pcr_lateral[-1],
        pcr_descriptor["dataset"]["termination"],
        NATURE_STYLES["pcr"]["color"],
    )

    start = pcr_descriptor["dataset"]["plot_rows"][0]
    ax.scatter(
        [float(start["robot_y_local"])],
        [float(start["robot_x_local"])],
        marker="s",
        s=28,
        facecolor="#262626",
        edgecolor="white",
        linewidth=0.5,
        zorder=23,
        clip_on=False,
    )

    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    ax.set_aspect("auto")
    ax.set_xlabel("Forward position, world y (m)", fontsize=8.0, labelpad=3.0)
    ax.set_ylabel("Lateral position, world x (m)", fontsize=8.0, labelpad=3.5)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(which="major", labelsize=7.0, width=0.8, length=2.8, pad=2.0)
    ax.tick_params(which="minor", width=0.55, length=1.6)
    ax.spines["left"].set_color("#292929")
    ax.spines["bottom"].set_color("#292929")

    method_handles = {
        method: Line2D(
            [0],
            [0],
            color=NATURE_STYLES[method]["color"],
            linewidth=PCR_LINEWIDTH if method == "pcr" else 1.2,
            alpha=1.0 if method == "pcr" else 0.82,
            label=NATURE_STYLES[method]["label"],
        )
        for method in source_plot.TRACK_ORDER
    }
    method_row_1 = [
        method_handles[method]
        for method in ("pcr", "rule_override", "additive_fusion", "fixed_authority")
    ]
    method_row_2 = [
        method_handles[method]
        for method in ("geomw", "risk_only", "yonly")
    ]
    status_handles = [
        Line2D([0], [0], color=TARGET_COLOR, linestyle=(0, (4.8, 3.0)), linewidth=1.05, label="Target"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#262626", markeredgecolor="white", markeredgewidth=0.5, markersize=4.6, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=NATURE_STYLES["pcr"]["color"], markeredgecolor="white", markeredgewidth=0.5, markersize=7.2, label="Success"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=0.4, markersize=4.8, label="Collision"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#777777", markeredgewidth=1.0, markersize=4.8, label="Follow lost / timeout"),
    ]
    fig.legend(
        handles=method_row_1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.990),
        ncol=4,
        fontsize=6.2,
        handlelength=1.45,
        handletextpad=0.38,
        columnspacing=1.10,
        labelspacing=0.22,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=method_row_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.954),
        ncol=3,
        fontsize=6.2,
        handlelength=1.45,
        handletextpad=0.38,
        columnspacing=0.78,
        labelspacing=0.22,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.918),
        ncol=5,
        fontsize=6.2,
        handlelength=1.35,
        handletextpad=0.38,
        columnspacing=0.92,
        borderpad=0.0,
        borderaxespad=0.0,
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_prefix.with_suffix(".svg")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    png_path = args.output_prefix.with_suffix(".png")
    tiff_path = args.output_prefix.with_suffix(".tiff")
    json_path = args.output_prefix.with_suffix(".json")
    fig.savefig(svg_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=600, facecolor="white")
    plt.close(fig)
    with Image.open(png_path) as raster:
        raster.save(str(tiff_path), compression="tiff_lzw", dpi=(600, 600))

    manifest = _nature_manifest(
        args.layout_json,
        layout,
        tracks,
        svg_path,
        pdf_path,
        png_path,
        tiff_path,
        plot_xlim,
        plot_ylim,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("selected trajectories:")
    for track_id in source_plot.TRACK_ORDER:
        item = tracks[track_id]
        print(
            f"  {track_id}: env={item['env_id']} "
            f"termination={item['dataset']['termination']} "
            f"collision_row={item['collision_row']}"
        )
    print(f"wrote {svg_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {tiff_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
