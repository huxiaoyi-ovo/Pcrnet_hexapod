#!/usr/bin/env python3
"""Render the frozen Strong Monolithic PPO budget-capability figure."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy


ROOT = Path(__file__).resolve().parents[2]
MONO_CSV = ROOT / "agents/final_paper_outputs_v3/table_mono_fixed_stage_detailed.csv"
PAPER_AUDIT_CSV = ROOT / "agents/final_paper_outputs_v3/table1_main_performance_stage4_audit.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures"
DEFAULT_QA_DIR = DEFAULT_OUTPUT_DIR / "qa_mono_budget_capability"
FIGURE_NAME = "fig_mono_budget_capability"
FIGURE_SIZE_IN = (7.1, 3.0)
VERTICAL_FIGURE_SIZE_IN = (3.45, 2.25)
EXPORT_DPI = 600

CAPTION = (
    "Fig. X. Mono capability emergence. Frozen-checkpoint probes at curriculum "
    "levels 2–4 use 0.35 m/s with policy updates disabled. Error bars show "
    "sample SD across three evaluation seeds for one training seed; the "
    "vertical lines mark the Adaptive budget and curriculum transition."
)

INK = "#202428"
MUTED = "#687078"
GRID = "#D9DEE2"
STAGE2 = "#7396B5"
STAGE3 = "#4B78A6"
STAGE4 = "#173F6B"
PROPOSED = "#C65D32"
MONO_LIGHT = "#7396B5"
MONO_MID = "#4B78A6"
MONO_DARK = "#173F6B"


def _load_alignment_helper():
    helper = Path("/home/artrc/.codex/skills/nature-figure/scripts/audit_panel_alignment.py")
    spec = importlib.util.spec_from_file_location("mono_panel_alignment", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load panel-alignment helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_matplotlib_panel_alignment


require_matplotlib_panel_alignment = _load_alignment_helper()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6.6,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mono_aggregate(rows: list[dict[str, str]], iteration: int, stage: int, speed: float) -> tuple[float, float]:
    candidates = [
        row
        for row in rows
        if row["record_type"] == "aggregate"
        and int(row["checkpoint_iteration"]) == iteration
        and int(row["stage"]) == stage
        and abs(float(row["target_speed_mps"]) - speed) < 1e-9
        and row["status"] == "complete"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one completed Mono aggregate for {(iteration, stage, speed)}, got {len(candidates)}")
    row = candidates[0]
    return 100.0 * float(row["strict_task_success_rate"]), 100.0 * float(row["strict_task_success_rate_sample_sd"])


def remote_1500_summary() -> dict[int, tuple[float, float]]:
    """Read the only omitted checkpoint directly from its frozen formal metrics."""
    remote = r'''python3 - <<'PY'
import glob, json, statistics
root = "/home/dell/Pcrnet_hexapod_strong_mono_75b5582/outputs/strong_mono_budget_curve_backfill_5f268bd/iter1500"
result = {}
for stage in (2, 3, 4):
    values = []
    for path in sorted(glob.glob(f"{root}/stage{stage}/s_0.35/seed*/**/metrics.json", recursive=True)):
        values.append(json.load(open(path, encoding="utf-8"))["overall"]["strict_success_rate"])
    if len(values) != 3:
        raise RuntimeError(f"stage {stage} expected 3 formal seed metrics, got {len(values)}")
    result[stage] = [statistics.mean(values), statistics.stdev(values)]
print(json.dumps(result, sort_keys=True))
PY'''
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "pcrnet-server", remote],
        check=True,
        text=True,
        capture_output=True,
    )
    raw = json.loads(completed.stdout)
    return {int(stage): (100.0 * values[0], 100.0 * values[1]) for stage, values in raw.items()}


def proposed_summary(rows: list[dict[str, str]]) -> dict[float, tuple[float, float]]:
    output: dict[float, tuple[float, float]] = {}
    for row in rows:
        if row["Method"] == "Learned-w":
            output[float(row["Speed"])] = (
                100.0 * float(row["Task Success Mean"]),
                100.0 * float(row["Task Success Std"]),
            )
    if set(output) != {0.35, 0.5, 0.6}:
        raise RuntimeError("The formal Learned-w three-speed summary is incomplete")
    return output


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.spines["bottom"].set_position(("outward", 2.8))
    axis.tick_params(colors=INK, pad=1.3)
    axis.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.82, zorder=0)
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])


def panel_label(
    fig: plt.Figure,
    axis: plt.Axes,
    letter: str,
    x: float,
    y: float,
) -> None:
    style = {"fontsize": 7.4, "fontweight": "bold", "color": INK, "ha": "left", "va": "bottom", "clip_on": False}
    axis.text(x, y, "(", transform=fig.transFigure, **style)
    axis.text(x, y, letter, transform=offset_copy(fig.transFigure, fig=fig, x=3.0, y=0.0, units="points"), **style)
    axis.text(x, y, ")", transform=offset_copy(fig.transFigure, fig=fig, x=7.5, y=0.0, units="points"), **style)


def errorbar_without_clipping(axis: plt.Axes, *args: object, **kwargs: object):
    container = axis.errorbar(*args, **kwargs)
    for artist in (container.lines[0], *container.lines[1], *container.lines[2]):
        artist.set_clip_on(False)
    return container


def make_figure(qa_dir: Path, layout: str = "horizontal") -> tuple[plt.Figure, dict[str, object]]:
    configure_style()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError(f"Unsupported figure layout: {layout}")
    mono_rows = read_csv(MONO_CSV)
    proposed = proposed_summary(read_csv(PAPER_AUDIT_CSV))
    remote_1500 = remote_1500_summary()
    interactions = np.array([12.288, 18.432, 24.576, 30.720, 36.864, 49.152, 61.440])
    checkpoints = [1000, 1500, 2000, 2500, 3000, 4000, 5000]
    stage_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for stage in (2, 3, 4):
        values, errors = [], []
        for checkpoint in checkpoints:
            mean, sd = remote_1500[stage] if checkpoint == 1500 else mono_aggregate(mono_rows, checkpoint, stage, 0.35)
            values.append(mean)
            errors.append(sd)
        stage_data[stage] = (np.array(values), np.array(errors))

    stage4_data = {
        24.576: [mono_aggregate(mono_rows, 2000, 4, speed) for speed in (0.35, 0.5, 0.6)],
        49.152: [mono_aggregate(mono_rows, 4000, 4, speed) for speed in (0.35, 0.5, 0.6)],
        61.440: [mono_aggregate(mono_rows, 5000, 4, speed) for speed in (0.35, 0.5, 0.6)],
    }

    figure_name = FIGURE_NAME if layout == "horizontal" else f"{FIGURE_NAME}_vertical"
    figure_size = FIGURE_SIZE_IN if layout == "horizontal" else VERTICAL_FIGURE_SIZE_IN
    fig = plt.figure(figsize=figure_size)
    if layout == "horizontal":
        axis_a = fig.add_axes([0.075, 0.295, 0.420, 0.530])
        axis_b = fig.add_axes([0.570, 0.295, 0.420, 0.530])
        panel_a_anchor = (0.005, 0.825)
        panel_b_anchor = (0.500, 0.825)
        for axis in (axis_a, axis_b):
            style_axis(axis)
    else:
        axis_a = fig.add_axes([0.140, 0.400 / 2.25, 0.820, 1.305 / 2.25])
        axis_b = None
        style_axis(axis_a)

    styles = {
        2: dict(color=STAGE2, marker="o", linestyle=(0, (1.2, 1.5)), linewidth=1.0, label="Level 2"),
        3: dict(color=STAGE3, marker="s", linestyle="--", linewidth=1.05, label="Level 3"),
        4: dict(color=STAGE4, marker="D", linestyle="-", linewidth=1.5, label="Level 4"),
    }
    for stage in (2, 3, 4):
        means, errors = stage_data[stage]
        errorbar_without_clipping(
            axis_a,
            interactions,
            means,
            yerr=errors,
            capsize=2.0,
            capthick=0.7,
            elinewidth=0.7,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.0 if stage != 4 else 4.5,
            zorder=4 if stage == 4 else 3,
            **styles[stage],
        )
    axis_a.axvline(24.576, color="#AEB5BA", linewidth=0.65, linestyle=(0, (2.0, 2.0)), zorder=1)
    axis_a.text(11.5, 78.0, "Adaptive high-level\nbudget", color=MUTED, fontsize=5.3, va="bottom", ha="left")
    axis_a.set_xlim(11.3, 62.5)
    axis_a.set_xticks([12.288, 18.432, 24.576, 30.720, 36.864, 49.152, 61.440])
    axis_a.set_xticklabels(["12.3", "18.4", "24.6", "30.7", "36.9", "49.2", "61.4"])
    axis_a.set_xlabel("High-level training interactions (M)")
    axis_a.set_ylabel("Task success (%)")
    axis_a.set_title("Mono capability probes across training budget", fontsize=6.8, fontweight="normal", color=INK, pad=21 if layout == "vertical" else 9.0)
    if layout == "vertical":
        legend_style = dict(fontsize=6.3, handlelength=2.1, handletextpad=0.6, columnspacing=1.4, borderaxespad=0)
    else:
        legend_style = dict(handlelength=1.35, handletextpad=0.45, columnspacing=0.9, borderaxespad=0.25)
    axis_a.legend(
        handles=[
            Line2D([], [], color=styles[2]["color"], marker="o", linestyle=styles[2]["linestyle"], linewidth=styles[2]["linewidth"], markersize=3.8, label="Level 2"),
            Line2D([], [], color=styles[3]["color"], marker="s", linestyle=styles[3]["linestyle"], linewidth=styles[3]["linewidth"], markersize=3.8, label="Level 3"),
            Line2D([], [], color=styles[4]["color"], marker="D", linestyle=styles[4]["linestyle"], linewidth=styles[4]["linewidth"], markersize=4.2, label="Level 4"),
        ],
        loc="lower center" if layout == "vertical" else "upper center",
        bbox_to_anchor=(0.5, 1.045) if layout == "vertical" else (0.5, -0.27),
        ncol=3,
        frameon=False,
        **legend_style,
    )
    if layout == "horizontal":
        panel_label(fig, axis_a, "a", *panel_a_anchor)

    axis_a.axvline(42.394, color="#9EB2C1", linewidth=0.65, linestyle=(0, (2.0, 2.0)), zorder=1)
    axis_a.text(29.5, 80.0, "42.394M:\nCurriculum Level\n2→3 transition", color=MUTED, fontsize=5.3, va="bottom", ha="left")

    if axis_b is not None:
        speeds = np.array([0.35, 0.50, 0.60])
        budget_styles = {
            24.576: dict(color=MONO_LIGHT, marker="o", linestyle=(0, (1.2, 1.5)), linewidth=1.0, label="Mono 24.6M"),
            49.152: dict(color=MONO_MID, marker="s", linestyle="--", linewidth=1.1, label="Mono 49.2M"),
            61.440: dict(color=MONO_DARK, marker="D", linestyle="-", linewidth=1.45, label="Mono 61.4M"),
        }
        for budget, style in budget_styles.items():
            values = np.array([item[0] for item in stage4_data[budget]])
            errors = np.array([item[1] for item in stage4_data[budget]])
            marker_kwargs = {
                "markerfacecolor": "white" if budget == 24.576 else style["color"],
                "markeredgecolor": style["color"] if budget == 24.576 else "white",
                "markeredgewidth": 1.0 if budget == 24.576 else 0.45,
            }
            errorbar_without_clipping(
                axis_b,
                speeds,
                values,
                yerr=errors,
                capsize=2.0,
                capthick=0.7,
                elinewidth=0.7,
                **marker_kwargs,
                markersize=4.2 if budget != 61.440 else 4.6,
                zorder=7 if budget == 24.576 else (5 if budget == 61.440 else 4),
                **style,
            )
        proposed_means = np.array([proposed[speed][0] for speed in speeds])
        proposed_errors = np.array([proposed[speed][1] for speed in speeds])
        errorbar_without_clipping(
            axis_b,
            speeds,
            proposed_means,
            yerr=proposed_errors,
            color=PROPOSED,
            marker="^",
            linestyle="-",
            linewidth=1.2,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.0,
            capthick=0.7,
            elinewidth=0.7,
            zorder=5,
            label="Adaptive",
        )
        axis_b.set_xlim(0.332, 0.618)
        axis_b.set_xticks(speeds)
        axis_b.set_xticklabels(["0.35", "0.50", "0.60"])
        axis_b.set_xlabel("Target speed (m/s)")
        axis_b.set_ylabel("Task success (%)")
        axis_b.set_title("Held out Stage 4 across speed", fontsize=6.8, fontweight="normal", color=INK, pad=9.0)
        axis_b.legend(
            handles=[
                Line2D([], [], color=MONO_LIGHT, marker="o", markerfacecolor="white", markeredgewidth=1.0, linestyle=budget_styles[24.576]["linestyle"], linewidth=budget_styles[24.576]["linewidth"], markersize=3.8, label="Mono 24.6M"),
                Line2D([], [], color=MONO_MID, marker="s", linestyle=budget_styles[49.152]["linestyle"], linewidth=budget_styles[49.152]["linewidth"], markersize=3.8, label="Mono 49.2M"),
                Line2D([], [], color=MONO_DARK, marker="D", linestyle=budget_styles[61.440]["linestyle"], linewidth=budget_styles[61.440]["linewidth"], markersize=4.2, label="Mono 61.4M"),
                Line2D([], [], color=PROPOSED, marker="^", linestyle="-", linewidth=1.2, markersize=5.0, label="Adaptive"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=4,
            frameon=False,
            handlelength=1.35,
            handletextpad=0.45,
            columnspacing=0.85,
            borderaxespad=0.25,
        )
        axis_b.text(0.603, 68.0, "20% above\ntraining range", fontsize=5.25, color=MUTED, ha="right", va="top")
        panel_label(fig, axis_b, "b", *panel_b_anchor)

    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if axis_b is None:
        # A single plot has no inter-panel geometry to compare.
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa_dir / f"{figure_name}.alignment.json",
            overlay_svg=qa_dir / f"{figure_name}.alignment.svg",
            tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
            axes=[axis_a],
            panel_ids=["single"],
        )
    else:
        require_matplotlib_panel_alignment(
            fig,
            json_out=qa_dir / f"{figure_name}.alignment.json",
            overlay_svg=qa_dir / f"{figure_name}.alignment.svg",
            tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
            axes=[axis_a, axis_b],
            panel_ids=["a", "b"],
            row_groups=[{"id": "main-row", "panels": ["a", "b"]}] if layout == "horizontal" else None,
            column_groups=[{"id": "main-column", "panels": ["a", "b"]}] if layout == "vertical" else None,
        )
    data = {
        "mono_source": str(MONO_CSV),
        "proposed_source": str(PAPER_AUDIT_CSV),
        "remote_1500_source": "pcrnet-server:/home/dell/Pcrnet_hexapod_strong_mono_75b5582/outputs/strong_mono_budget_curve_backfill_5f268bd/iter1500",
        "remote_1500_summary_percent_mean_and_sample_sd": {
            str(stage): list(values) for stage, values in remote_1500.items()
        },
        "layout": layout,
        "figure_size_in": list(figure_size),
        "caption": CAPTION,
        "panel_a_protocol": "Mono-only deterministic frozen-checkpoint capability probes at curriculum levels 2-4; no policy updates",
        "panel_a_removed_display": "one Adaptive formal-result marker and its legend entry; the 24.576M budget reference line remains",
        "source_data_changed": False,
        "panel_b_protocol": "formal held-out Stage 4 comparison retained; Mono and Adaptive use independent evaluation seed sets and are not episode-paired",
    }
    if layout == "vertical":
        data["figure_claim"] = "Mono capability emergence from deterministic frozen-checkpoint probes; this is a diagnostic figure, not a formal-performance comparison"
        data["panel_a_protocol"] = "Frozen-checkpoint Mono probes at curriculum levels 2-4 at 0.35 m/s; policy updates disabled; error bars are sample SD over three evaluation seeds for one training seed"
        data.pop("panel_b_protocol")
        data.pop("panel_a_removed_display")
    return fig, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qa-dir", type=Path, default=None)
    parser.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal")
    args = parser.parse_args()
    output_dir = args.output_dir
    qa_dir = args.qa_dir or output_dir / "qa_mono_budget_capability"
    fig, metadata = make_figure(qa_dir, layout=args.layout)
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    figure_name = FIGURE_NAME if args.layout == "horizontal" else f"{FIGURE_NAME}_vertical"
    fig.savefig(output_dir / f"{figure_name}.pdf", format="pdf")
    fig.savefig(output_dir / f"{figure_name}.png", format="png", dpi=EXPORT_DPI)
    (qa_dir / f"{figure_name}.provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
