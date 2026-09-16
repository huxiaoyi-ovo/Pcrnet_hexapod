#!/usr/bin/env python3
"""Create Nature-style alternatives for the final PCR-Net manuscript figures.

The script intentionally reads the frozen paper CSVs and writes new files.  It
does not overwrite the original manuscript figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_WIDTH_MM = 89.0
SYSTEM_WIDTH_MM = 183.0
MM_PER_INCH = 25.4
EXPORT_DPI = 600
PDF_SUFFIX = ".pdf"
TIFF_SUFFIX = ".tiff"

INK = "#202428"
MUTED = "#687078"
GRID = "#D9DEE2"
PCR_BLUE = "#174A7E"
PCR_LIGHT = "#DDE8F2"
RISK_RED = "#A33A32"
RISK_LIGHT = "#F3E3DF"
WARM = "#C17C3A"
TEAL = "#4D7F80"
GREEN = "#5E7D5C"
PURPLE = "#77628D"
GRAY = "#858A8D"
LIGHT_GRAY = "#EFF2F3"

METHOD_ORDER = [
    "Y-only",
    "Geom-w",
    "Risk-only",
    "Rule-Override",
    "Additive-Fusion",
    "Fixed-Authority",
    "Learned-w",
]
METHOD_STYLE = {
    "Y-only": dict(color=GRAY, marker="o"),
    "Geom-w": dict(color=TEAL, marker="s"),
    "Risk-only": dict(color=WARM, marker="^"),
    "Rule-Override": dict(color=GREEN, marker="D"),
    "Additive-Fusion": dict(color=PURPLE, marker="P"),
    "Fixed-Authority": dict(color="#A66B62", marker="X"),
    "Learned-w": dict(color=PCR_BLUE, marker="*"),
}


def _load_alignment_helper():
    path = Path.home() / ".codex/skills/nature-figure/scripts/audit_panel_alignment.py"
    spec = importlib.util.spec_from_file_location("nature_panel_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load panel-alignment helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_matplotlib_panel_alignment


require_matplotlib_panel_alignment = _load_alignment_helper()


def mm_to_in(value: float) -> float:
    return value / MM_PER_INCH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.3,
            "axes.labelsize": 6.6,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors=INK, pad=1.5)
    axis.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.72, zorder=0)


def panel_label(axis: plt.Axes, text: str, y: float = 1.04) -> None:
    axis.text(
        -0.16,
        y,
        text,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontsize=7.2,
        fontweight="bold",
        clip_on=False,
    )


def line_key(
    axis: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    color: str,
    linewidth: float,
    linestyle: str | tuple = "solid",
) -> None:
    """Draw a compact out-of-axes key with the exact plotted line style."""
    axis.plot(
        [x, x + 0.065],
        [y, y],
        transform=axis.transAxes,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="butt",
        clip_on=False,
        zorder=8,
    )
    axis.text(
        x + 0.082,
        y,
        label,
        transform=axis.transAxes,
        color=color,
        fontsize=5.4,
        ha="left",
        va="center",
        clip_on=False,
        zorder=8,
    )


def parse_pm(value: str) -> tuple[float, float]:
    chunks = value.replace("±", "+/-").split("+/-")
    if len(chunks) != 2:
        raise ValueError(f"Expected mean +/- std, got {value!r}")
    return float(chunks[0].strip()), float(chunks[1].strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_outputs(fig: plt.Figure, prefix: Path) -> dict[str, str]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": prefix.with_suffix(".svg"),
        "pdf": prefix.with_suffix(PDF_SUFFIX),
        "png": prefix.with_suffix(".png"),
        "tiff": prefix.with_suffix(TIFF_SUFFIX),
    }
    fig.savefig(paths["svg"], format="svg")
    fig.savefig(paths["pdf"], format="pdf")
    fig.savefig(paths["png"], format="png", dpi=EXPORT_DPI)
    with Image.open(paths["png"]) as image:
        image.save(str(paths["tiff"]), dpi=(EXPORT_DPI, EXPORT_DPI), compression="tiff_lzw")
    return {key: str(value) for key, value in paths.items()}


def write_manifest(
    path: Path,
    *,
    claim: str,
    archetype: str,
    sources: Iterable[Path],
    outputs: dict[str, str],
    notes: list[str],
) -> None:
    source_rows = [
        {"path": str(source), "sha256": sha256(source)} for source in sources
    ]
    output_rows = {
        key: {"path": value, "sha256": sha256(Path(value))}
        for key, value in outputs.items()
    }
    payload = {
        "schema_version": 1,
        "claim": claim,
        "archetype": archetype,
        "sources": source_rows,
        "outputs": output_rows,
        "notes": notes,
        "rendering": {
            "backend": "Python/Matplotlib",
            "dpi": EXPORT_DPI,
            "editable_vector": True,
            "minimum_nominal_font_pt": 5.4,
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _marker_fill_area_factor(marker: str) -> float:
    from matplotlib.markers import MarkerStyle

    marker_style = MarkerStyle(marker)
    path = marker_style.get_path().transformed(marker_style.get_transform())
    signed_area = 0.0
    for polygon in path.to_polygons(closed_only=True):
        for index in range(len(polygon) - 1):
            x0, y0 = polygon[index]
            x1, y1 = polygon[index + 1]
            signed_area += 0.5 * (float(x0) * float(y1) - float(x1) * float(y0))
    area = abs(signed_area)
    if area <= 1e-9:
        raise RuntimeError(f"Marker {marker!r} has zero fill area")
    return area


def _bbox_is_clear(
    candidate,
    *,
    marker_bboxes,
    label_bboxes,
    gridline_bboxes,
    axes_bbox,
    own_marker_bbox,
) -> bool:
    """Return whether a rendered label bbox has a clear in-axes placement."""
    edge_px = 1.0
    if (
        candidate.x0 < axes_bbox.x0 + edge_px
        or candidate.x1 > axes_bbox.x1 - edge_px
        or candidate.y0 < axes_bbox.y0 + edge_px
        or candidate.y1 > axes_bbox.y1 - edge_px
    ):
        return False
    for marker_bbox in marker_bboxes:
        if marker_bbox is own_marker_bbox:
            continue
        if candidate.overlaps(marker_bbox):
            return False
    for gridline_bbox in gridline_bboxes:
        if (
            candidate.x0 < gridline_bbox.x1 + edge_px
            and candidate.x1 > gridline_bbox.x0 - edge_px
            and candidate.y0 < gridline_bbox.y1 + edge_px
            and candidate.y1 > gridline_bbox.y0 - edge_px
        ):
            return False
    return not any(candidate.overlaps(label_bbox) for label_bbox in label_bboxes)


def place_scatter_labels(axis: plt.Axes, points: list[dict[str, Any]]) -> None:
    """Place direct scatter labels against measured glyph boundaries.

    Candidate locations are evaluated after Matplotlib has rendered the marker
    paths, so heterogeneous marker areas and shapes receive the same small
    screen-space clearance without a hand-written per-method offset table.
    """
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = axis.get_window_extent(renderer)
    marker_bboxes = [point["artist"].get_window_extent(renderer) for point in points]
    gridline_bboxes = [
        line.get_window_extent(renderer)
        for line in (*axis.get_xgridlines(), *axis.get_ygridlines())
        if line.get_visible()
    ]
    gap_px = 2.4 * figure.dpi / 72.0
    directions = [
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
    ]
    placed_bboxes = []
    for point, marker_bbox in zip(points, marker_bboxes):
        candidates = []
        for dx, dy in directions:
            x = (
                marker_bbox.x0 - gap_px
                if dx < 0
                else marker_bbox.x1 + gap_px
                if dx > 0
                else marker_bbox.x0 + marker_bbox.width / 2.0
            )
            y = (
                marker_bbox.y0 - gap_px
                if dy < 0
                else marker_bbox.y1 + gap_px
                if dy > 0
                else marker_bbox.y0 + marker_bbox.height / 2.0
            )
            data_x, data_y = axis.transData.inverted().transform((x, y))
            text = axis.text(
                data_x,
                data_y,
                point["label"],
                transform=axis.transData,
                color=point["color"],
                fontsize=5.5,
                fontweight="bold" if point["method"] == "Learned-w" else "normal",
                ha="right" if dx < 0 else "left" if dx > 0 else "center",
                va="top" if dy < 0 else "bottom" if dy > 0 else "center",
                clip_on=False,
                zorder=7,
            )
            figure.canvas.draw()
            bbox = text.get_window_extent(renderer)
            center_distance = math.hypot(
                bbox.x0 + bbox.width / 2.0 - (marker_bbox.x0 + marker_bbox.width / 2.0),
                bbox.y0 + bbox.height / 2.0 - (marker_bbox.y0 + marker_bbox.height / 2.0),
            )
            is_clear = _bbox_is_clear(
                bbox,
                marker_bboxes=marker_bboxes,
                label_bboxes=placed_bboxes,
                gridline_bboxes=gridline_bboxes,
                axes_bbox=axes_bbox,
                own_marker_bbox=marker_bbox,
            )
            candidates.append((center_distance, text, bbox, is_clear))
            text.remove()
        clear_candidates = [candidate for candidate in candidates if candidate[3]]
        if not clear_candidates:
            raise RuntimeError(f"No collision-free direct-label location for {point['method']}")
        _, text, bbox, _ = min(clear_candidates, key=lambda candidate: candidate[0])
        axis.add_artist(text)
        placed_bboxes.append(bbox)


def plot_main_performance(table_path: Path, scatter_path: Path, output_dir: Path, qa_dir: Path) -> None:
    table_rows = read_csv(table_path)
    scatter_rows = read_csv(scatter_path)
    by_method: dict[str, list[dict[str, str]]] = {name: [] for name in METHOD_ORDER}
    for row in table_rows:
        if row["Method"] in by_method:
            by_method[row["Method"]].append(row)

    width = mm_to_in(FIG_WIDTH_MM)
    fig = plt.figure(figsize=(width, 4.55))
    axis_a = fig.add_axes([0.18, 0.59, 0.75, 0.31])
    axis_b = fig.add_axes([0.18, 0.12, 0.75, 0.31])
    axes = [axis_a, axis_b]
    for axis in axes:
        style_axis(axis)

    axis_a.axvspan(0.33, 0.515, color=PCR_LIGHT, alpha=0.45, zorder=-2)
    axis_a.axvline(0.55, color=MUTED, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    axis_a.text(0.60, 1.025, "above training range", color=MUTED, fontsize=5.4, ha="center", va="bottom")
    for method in METHOD_ORDER:
        rows = sorted(by_method[method], key=lambda row: float(row["Speed"]))
        if not rows:
            continue
        speeds = np.asarray([float(row["Speed"]) for row in rows])
        parsed = [parse_pm(row["Task Success ↑"]) for row in rows]
        means = np.asarray([item[0] for item in parsed])
        errors = np.asarray([item[1] for item in parsed])
        style = METHOD_STYLE[method]
        is_pcr = method == "Learned-w"
        axis_a.errorbar(
            speeds,
            means,
            yerr=errors,
            color=style["color"],
            marker=style["marker"],
            markersize=5.7 if is_pcr else 3.5,
            linewidth=1.65 if is_pcr else 0.85,
            markeredgecolor="white" if is_pcr else style["color"],
            markeredgewidth=0.55,
            capsize=1.8,
            elinewidth=0.7,
            alpha=1.0 if is_pcr else 0.78,
            label="PCR-Net" if is_pcr else method,
            zorder=5 if is_pcr else 3,
        )
    axis_a.set_xlim(0.325, 0.625)
    axis_a.set_ylim(-0.035, 1.07)
    axis_a.set_xticks([0.35, 0.50, 0.60])
    axis_a.set_yticks([0.0, 0.5, 1.0])
    axis_a.set_xlabel("Target speed (m/s)")
    axis_a.set_ylabel("Task success")
    panel_label(axis_a, "a")
    handles, labels = axis_a.get_legend_handles_labels()
    axis_a.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.03, 1.02, 0.2),
        ncol=3,
        mode="expand",
        frameon=False,
        handlelength=1.7,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    scatter_by_method = {row["Method"]: row for row in scatter_rows}
    scatter_points = []
    for method in METHOD_ORDER:
        if method not in scatter_by_method:
            continue
        row = scatter_by_method[method]
        collision = float(row["Collision"])
        mae = float(row["Follow MAE [m]"])
        success = float(row["Task Success"])
        style = METHOD_STYLE[method]
        point_area = 0.30 * float(row["Matplotlib scatter area"])
        artist = axis_b.scatter(
            [collision],
            [mae],
            s=point_area,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.65,
            alpha=1.0 if method == "Learned-w" else 0.82,
            zorder=5 if method == "Learned-w" else 3,
        )
        scatter_points.append(
            {
                "method": method,
                "label": "PCR-Net" if method == "Learned-w" else method,
                "color": style["color"],
                "artist": artist,
            }
        )
        if method == "Learned-w" and not math.isclose(success, 0.953125, abs_tol=0.002):
            raise RuntimeError("Frozen PCR-Net scatter anchor changed")
    axis_b.set_xlim(-0.035, 0.79)
    max_mae = max(float(row["Follow MAE [m]"]) for row in scatter_rows if row["Method"] in METHOD_STYLE)
    axis_b_upper = 1.39 if max_mae <= 1.39 else math.ceil((max_mae + 0.05) * 10.0) / 10.0
    axis_b.set_ylim(0.24, axis_b_upper)
    axis_b.set_xlabel("Collision rate")
    axis_b.set_ylabel("Follow MAE (m)")
    axis_b.set_xticks([0.0, 0.25, 0.50, 0.75])
    axis_b.set_yticks([0.3, 0.8, 1.3] if axis_b_upper == 1.39 else [0.3, 0.8, 1.3, axis_b_upper])
    panel_label(axis_b, "b")
    place_scatter_labels(axis_b, scatter_points)

    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    alignment_json = qa_dir / "final_main_performance_nature.alignment.json"
    require_matplotlib_panel_alignment(
        fig,
        json_out=alignment_json,
        overlay_svg=qa_dir / "final_main_performance_nature.alignment.svg",
        tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        axes=[axis_a, axis_b],
        panel_ids=["a", "b"],
        column_groups=[{"id": "stack", "panels": ["a", "b"]}],
    )
    prefix = output_dir / "final_main_performance_nature"
    outputs = save_outputs(fig, prefix)
    plt.close(fig)
    write_manifest(
        prefix.with_suffix(".json"),
        claim="PCR-Net preserves task success as speed increases and occupies the strongest tested safety-tracking trade-off at 0.60 m/s.",
        archetype="Two-panel claim-escalating quantitative figure",
        sources=[table_path, scatter_path],
        outputs=outputs,
        notes=[
            "Panel a shows mean +/- standard deviation from the frozen three-seed table.",
            "Panel b applies one constant 0.30 display scale to all frozen marker-area values; ordering and encoding are unchanged.",
            "Panel b direct labels are placed from the rendered marker and text bounds with a fixed 2.4 pt edge clearance.",
            (
                "Fixed-Authority was plotted from the same-protocol scatter CSV."
                if "Fixed-Authority" in scatter_by_method
                else "No same-protocol Fixed-Authority row was present in the scatter CSV; no substitute value was plotted."
            ),
            "The 0.60 m/s point is explicitly separated from the training-speed range.",
        ],
    )


def numeric_rows(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        columns[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    return columns


def contiguous_spans(time_s: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    dt = float(np.nanmedian(np.diff(time_s))) if time_s.size > 1 else 0.1
    groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [(float(time_s[group[0]]), float(time_s[group[-1]] + dt)) for group in groups]


def plot_arbitration(csv_path: Path, output_dir: Path, qa_dir: Path) -> None:
    palette = {
        "risk": "#B4473A",
        "pcr": "#1D5B8F",
        "risk_only": "#347C78",
        "raw": "#6D747A",
        "forward": "#A95F1C",
        "conflict": "#F5E8E3",
        "grid": "#E1E6E9",
    }
    data = numeric_rows(csv_path)
    time_s = data["time_s"]
    mask = data["high_risk_conflict"] > 0.5
    spans = contiguous_spans(time_s, mask)
    visible_ranges = {
        "risk_F": (-0.015, 0.75),
        "y": (0.44, 0.75),
        "y_risk": (0.44, 0.75),
        "y_eff": (0.44, 0.75),
        "cmd_safe_x_abs_delta_from_pre_conflict": (-0.025, 0.105),
        "cmd_safe_y": (0.30, 0.75),
    }
    for key, (lower, upper) in visible_ranges.items():
        series_min = float(np.nanmin(data[key]))
        series_max = float(np.nanmax(data[key]))
        if series_min < lower - 1e-9 or series_max > upper + 1e-9:
            raise RuntimeError(
                f"Arbitration axis clips {key}: [{series_min:.4f}, {series_max:.4f}] "
                f"outside [{lower:.4f}, {upper:.4f}]"
            )

    fig = plt.figure(figsize=(mm_to_in(FIG_WIDTH_MM), 4.95))
    axis_a = fig.add_axes([0.18, 0.70, 0.67, 0.20])
    axis_b = fig.add_axes([0.18, 0.40, 0.67, 0.20], sharex=axis_a)
    axis_c = fig.add_axes([0.18, 0.10, 0.67, 0.20], sharex=axis_a)
    axes = [axis_a, axis_b, axis_c]
    for axis in axes:
        style_axis(axis)
        axis.grid(axis="y", color=palette["grid"], linewidth=0.45, alpha=0.82, zorder=0)
        for start, stop in spans:
            axis.axvspan(start, stop, color=palette["conflict"], alpha=0.68, linewidth=0, zorder=-1)

    axis_a.plot(time_s, data["risk_F"], color=palette["risk"], linewidth=1.55, label="Follow-command risk ρF")
    axis_a.axhline(0.25, color=palette["risk"], linewidth=0.75, linestyle=(0, (2, 2)), alpha=0.72, zorder=1)
    axis_a.text(
        0.02,
        0.265,
        "high-risk threshold",
        transform=axis_a.get_yaxis_transform(),
        ha="left",
        va="bottom",
        color=palette["risk"],
        fontsize=5.3,
    )
    axis_a.set_ylim(-0.015, 0.75)
    axis_a.set_yticks([0.0, 0.25, 0.50, 0.75])
    axis_a.set_ylabel("Risk")
    panel_label(axis_a, "a", y=1.26)
    axis_a.text(0.0, 1.26, "Follow-command risk ρF", transform=axis_a.transAxes, ha="left", va="bottom", color=INK, fontsize=7.0, fontweight="bold")

    axis_b.plot(time_s, data["y"], color=palette["raw"], linewidth=0.95, linestyle=(0, (2, 1.4)), label="Raw y")
    axis_b.plot(time_s, data["y_risk"], color=palette["risk_only"], linewidth=1.1, label="Risk-only y + Δyr")
    axis_b.plot(time_s, data["y_eff"], color=palette["pcr"], linewidth=1.55, label="PCR yeff")
    axis_b.fill_between(
        time_s,
        data["y_risk"],
        data["y_eff"],
        where=data["y_eff"] >= data["y_risk"],
        color=palette["pcr"],
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    axis_b.set_ylim(0.44, 0.75)
    axis_b.set_yticks([0.45, 0.55, 0.65, 0.75])
    axis_b.set_ylabel("Follow weight")
    panel_label(axis_b, "b", y=1.26)
    axis_b.text(0.0, 1.26, "Risk-conditioned arbitration", transform=axis_b.transAxes, ha="left", va="bottom", color=INK, fontsize=7.0, fontweight="bold")
    line_key(axis_b, x=0.02, y=1.09, label="Raw y", color=palette["raw"], linewidth=0.95, linestyle=(0, (2, 1.4)))
    line_key(axis_b, x=0.25, y=1.09, label="Risk-only y + Δyr", color=palette["risk_only"], linewidth=1.1)
    line_key(axis_b, x=0.72, y=1.09, label="PCR yeff", color=palette["pcr"], linewidth=1.55)

    twin = axis_c.twinx()
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    twin.spines["right"].set_color(palette["forward"])
    twin.tick_params(axis="y", colors=palette["forward"], labelsize=5.8, width=0.45, length=2.4, pad=1.5)
    axis_c.plot(
        time_s,
        data["cmd_safe_x_abs_delta_from_pre_conflict"],
        color=palette["pcr"],
        linewidth=1.5,
        label="Lateral increase Δ|ux|",
    )
    twin.plot(time_s, data["cmd_safe_y"], color=palette["forward"], linewidth=1.1, linestyle=(0, (5, 1.6)), label="Forward uy")
    axis_c.axhline(0.0, color=palette["grid"], linewidth=0.55, zorder=0)
    axis_c.set_ylim(-0.025, 0.105)
    axis_c.set_yticks([-0.02, 0.00, 0.05, 0.10])
    twin.set_ylim(0.30, 0.75)
    twin.set_yticks([0.30, 0.50, 0.70])
    axis_c.set_ylabel("Δ|ux| (m/s)", color=palette["pcr"])
    twin.set_ylabel("uy (m/s)", color=palette["forward"])
    axis_c.set_xlabel("Time (s)")
    panel_label(axis_c, "c", y=1.26)
    axis_c.text(0.0, 1.26, "Executed motion", transform=axis_c.transAxes, ha="left", va="bottom", color=INK, fontsize=7.0, fontweight="bold")
    line_key(axis_c, x=0.02, y=1.09, label="Lateral increase Δ|ux|", color=palette["pcr"], linewidth=1.5)
    line_key(axis_c, x=0.64, y=1.09, label="Forward uy", color=palette["forward"], linewidth=1.1, linestyle=(0, (5, 1.6)))

    axis_a.tick_params(labelbottom=False)
    axis_b.tick_params(labelbottom=False)
    axis_c.set_xlim(float(time_s[0]), float(time_s[-1]))
    for axis in axes:
        axis.set_xticks([0, 2, 4, 6, 8])

    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=qa_dir / "final_real_robot_arbitration_nature.alignment.json",
        overlay_svg=qa_dir / "final_real_robot_arbitration_nature.alignment.svg",
        tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        axes=axes,
        panel_ids=["a", "b", "c"],
        column_groups=[{"id": "trace-stack", "panels": ["a", "b", "c"]}],
    )
    prefix = output_dir / "final_real_robot_arbitration_nature"
    outputs = save_outputs(fig, prefix)
    plt.close(fig)
    write_manifest(
        prefix.with_suffix(".json"),
        claim="During real-robot conflict intervals, PCR-Net reduces the Follow weight, restores part of it through the learned correction, and increases lateral motion while preserving forward motion.",
        archetype="Three-stage aligned real-robot time-series figure",
        sources=[csv_path],
        outputs=outputs,
        notes=[
            "Conflict intervals are computed directly from the frozen CSV.",
            "Panel a shows the recorded Follow-command risk and its 0.25 high-risk threshold; the redundant front-risk overlay is omitted.",
            "The lateral trace uses the CSV's pre-conflict-baseline-adjusted magnitude.",
            "A color-blind-safe semantic palette is used consistently across all three panels.",
            "Axis limits follow the observed data range with explicit headroom; no trace is clipped.",
            "No smoothing or resampling is applied.",
        ],
    )


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    title: str,
    body: str = "",
    title_color: str = INK,
    body_color: str = MUTED,
    linewidth: float = 0.85,
    zorder: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.007,rounding_size=0.012",
        transform=axis.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * (0.63 if body else 0.50),
        title,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="semibold",
        color=title_color,
        zorder=zorder + 1,
    )
    if body:
        axis.text(
            x + width / 2,
            y + height * 0.30,
            body,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            color=body_color,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    dashed: bool = False,
    connectionstyle: str = "arc3",
    zorder: int = 1,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=7.0,
        linewidth=0.85,
        color=color,
        linestyle=(0, (2.2, 2.0)) if dashed else "solid",
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    axis.add_patch(patch)


def group_label(axis: plt.Axes, x: float, text: str, rate: str, color: str) -> None:
    axis.text(x, 0.955, text, transform=axis.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    axis.text(x, 0.905, rate, transform=axis.transAxes, ha="center", va="center", fontsize=5.5, color=color)


def status_tag(axis: plt.Axes, xy: tuple[float, float], text: str, face: str, edge: str) -> None:
    axis.text(
        xy[0],
        xy[1],
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.3,
        color=edge,
        bbox=dict(boxstyle="round,pad=0.23", facecolor=face, edgecolor=edge, linewidth=0.55),
        zorder=5,
    )


def plot_system_architecture(output_dir: Path) -> None:
    fig = plt.figure(figsize=(mm_to_in(SYSTEM_WIDTH_MM), 3.25))
    axis = fig.add_axes([0.015, 0.04, 0.97, 0.92])
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    group_label(axis, 0.105, "Onboard perception", "30 Hz", TEAL)
    group_label(axis, 0.315, "Orthogonal experts", "shared proposals", PCR_BLUE)
    group_label(axis, 0.615, "PCR-Net arbitration", "10 Hz", PCR_BLUE)
    group_label(axis, 0.885, "Command execution", "50 Hz locomotion", GREEN)

    rounded_box(
        axis,
        (0.025, 0.56),
        0.16,
        0.22,
        face=LIGHT_GRAY,
        edge="#9BA5AC",
        title="D435i depth + YOLOv8",
        body="local geometry + target",
    )
    rounded_box(
        axis,
        (0.025, 0.24),
        0.16,
        0.22,
        face="#E8F0F0",
        edge=TEAL,
        title="Perception state",
        body="occupancy–clearance map Mt\nrelative target state",
    )
    arrow(axis, (0.105, 0.56), (0.105, 0.46), color=TEAL)

    rounded_box(
        axis,
        (0.235, 0.57),
        0.16,
        0.19,
        face="#E9F0F6",
        edge=PCR_BLUE,
        title="Follow expert",
        body="uF = [0, uF,y, uF,ω]",
    )
    rounded_box(
        axis,
        (0.235, 0.27),
        0.16,
        0.19,
        face="#E9F0F6",
        edge=PCR_BLUE,
        title="Avoid expert",
        body="uA = [uA,x, 0, 0]",
    )
    status_tag(axis, (0.315, 0.805), "analytic", "#F7F9FA", MUTED)
    status_tag(axis, (0.315, 0.215), "pretrained + frozen", "#F7F9FA", MUTED)
    arrow(axis, (0.185, 0.38), (0.235, 0.665), color=TEAL, connectionstyle="arc3,rad=-0.18")
    arrow(axis, (0.185, 0.35), (0.235, 0.365), color=TEAL)

    hero = FancyBboxPatch(
        (0.435, 0.16),
        0.36,
        0.69,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor="#F6F9FC",
        edgecolor=PCR_BLUE,
        linewidth=1.1,
        zorder=0,
    )
    axis.add_patch(hero)
    rounded_box(
        axis,
        (0.46, 0.60),
        0.13,
        0.16,
        face=RISK_LIGHT,
        edge=RISK_RED,
        title="Risk query",
        body="ρFdir, ρAdir, mt",
    )
    rounded_box(
        axis,
        (0.635, 0.60),
        0.13,
        0.16,
        face=PCR_LIGHT,
        edge=PCR_BLUE,
        title="GatePolicy",
        body="bounded heads y, w",
    )
    rounded_box(
        axis,
        (0.46, 0.33),
        0.305,
        0.16,
        face=PCR_LIGHT,
        edge=PCR_BLUE,
        title="Effective Follow weight",
        body="yeff = clip(y + Δyw + Δyrdir, 0, 1)",
    )
    rounded_box(
        axis,
        (0.46, 0.19),
        0.305,
        0.09,
        face=PCR_BLUE,
        edge=PCR_BLUE,
        title="umix = yeff uF + (1 − yeff) uA",
        title_color="white",
    )
    status_tag(axis, (0.700, 0.805), "learned gate only", "#E9F0F6", PCR_BLUE)
    arrow(axis, (0.59, 0.68), (0.635, 0.68), color=RISK_RED, dashed=True)
    arrow(axis, (0.70, 0.60), (0.65, 0.49), color=PCR_BLUE)
    arrow(axis, (0.525, 0.60), (0.565, 0.49), color=RISK_RED, dashed=True)
    arrow(axis, (0.612, 0.33), (0.612, 0.28), color=PCR_BLUE)
    arrow(axis, (0.395, 0.665), (0.46, 0.68), color=RISK_RED, dashed=True)
    arrow(axis, (0.395, 0.365), (0.46, 0.65), color=RISK_RED, dashed=True, connectionstyle="arc3,rad=-0.10")
    arrow(axis, (0.395, 0.57), (0.46, 0.245), color=MUTED, connectionstyle="arc3,rad=0.18")
    arrow(axis, (0.395, 0.365), (0.46, 0.225), color=MUTED, connectionstyle="arc3,rad=-0.12")
    axis.plot([0.185, 0.205, 0.435], [0.40, 0.84, 0.84], transform=axis.transAxes, color=TEAL, linewidth=0.85, linestyle=(0, (2.2, 2.0)), zorder=1)
    arrow(axis, (0.435, 0.84), (0.46, 0.72), color=TEAL, dashed=True)

    rounded_box(
        axis,
        (0.825, 0.61),
        0.15,
        0.15,
        face="#E9F1E7",
        edge=GREEN,
        title="Command shaping",
        body="clamp + slew + clearance",
    )
    rounded_box(
        axis,
        (0.825, 0.37),
        0.15,
        0.15,
        face="#E9F1E7",
        edge=GREEN,
        title="Locomotion policy",
        body="velocity → 18 joints",
    )
    rounded_box(
        axis,
        (0.825, 0.17),
        0.15,
        0.11,
        face="#F4F6F3",
        edge=GREEN,
        title="Hexapod",
    )
    status_tag(axis, (0.90, 0.805), "fixed + shared", "#F1F6F0", GREEN)
    arrow(axis, (0.765, 0.235), (0.825, 0.685), color=MUTED, connectionstyle="arc3,rad=-0.18")
    arrow(axis, (0.90, 0.61), (0.90, 0.52), color=GREEN)
    arrow(axis, (0.90, 0.37), (0.90, 0.28), color=GREEN)

    axis.plot([0.47, 0.50], [0.095, 0.095], transform=axis.transAxes, color=MUTED, linewidth=0.9)
    axis.text(0.505, 0.095, "command / observation", transform=axis.transAxes, va="center", fontsize=5.4, color=MUTED)
    axis.plot([0.65, 0.68], [0.095, 0.095], transform=axis.transAxes, color=RISK_RED, linewidth=0.9, linestyle=(0, (2.2, 2.0)))
    axis.text(0.685, 0.095, "risk / context", transform=axis.transAxes, va="center", fontsize=5.4, color=RISK_RED)

    prefix = output_dir / "final_system_architecture_nature"
    outputs = save_outputs(fig, prefix)
    plt.close(fig)
    paper_path = Path.home() / "下载/PCRNet_final_clean/final.tex"
    write_manifest(
        prefix.with_suffix(".json"),
        claim="PCR-Net learns only a bounded scalar arbitration between fixed orthogonal expert proposals; perception, command shaping, and locomotion remain shared components.",
        archetype="Wide asymmetric mechanism schematic with a central hero module",
        sources=[paper_path],
        outputs=outputs,
        notes=[
            "The diagram is redrawn from the manuscript's problem formulation, method, training procedure, and system caption.",
            "Solid arrows indicate command or observation flow; dashed warm arrows indicate risk or context flow.",
            "No implementation detail absent from the manuscript is introduced.",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-table",
        type=Path,
        default=ROOT / "agents/final_paper_outputs_v3/table1_main_performance_stage4.csv",
    )
    parser.add_argument(
        "--main-scatter",
        type=Path,
        default=ROOT / "agents/final_paper_outputs_v3/fig3b_scatter_sized_data_060.csv",
    )
    parser.add_argument(
        "--arbitration-csv",
        type=Path,
        default=ROOT / "agents/final_paper_outputs_v3/fig7_real_robot_arbitration_data.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--qa-dir", type=Path, default=ROOT / "tmp/figure_qa/final_nature_figures")
    parser.add_argument(
        "--figure",
        choices=("all", "main"),
        default="all",
        help="Render only the requested final figure; use main to avoid rewriting other figure outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    plot_main_performance(args.main_table, args.main_scatter, args.output_dir, args.qa_dir)
    if args.figure == "main":
        print(f"Wrote Nature-style main-performance figure to {args.output_dir}")
        return
    plot_arbitration(args.arbitration_csv, args.output_dir, args.qa_dir)
    plot_system_architecture(args.output_dir)
    print(f"Wrote Nature-style figure alternatives to {args.output_dir}")


if __name__ == "__main__":
    main()
