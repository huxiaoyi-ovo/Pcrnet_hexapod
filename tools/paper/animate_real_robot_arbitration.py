#!/usr/bin/env python3
"""Animate the real-robot PCR arbitration trace from the exported Fig. 7 CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


TOKENS = {
    "risk_f": "#D62728",
    "front_risk": "#7F7F7F",
    "conflict": "#F4B6B6",
    "raw_y": "#6F6F6F",
    "risk_y": "#1F77B4",
    "eff_y": "#D62728",
    "learned_fill": "#CEDFFE",
    "lat": "#1F77B4",
    "fwd": "#D97904",
    "baseline": "#8A8A8A",
    "cursor": "#6F4E9C",
    "highlight": "#F2C94C",
}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    data: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            try:
                data.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                pass
    return {key: np.asarray(value, dtype=float) for key, value in data.items()}


def _spans(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = None
    for idx, active in enumerate(mask.astype(bool)):
        if active and start is None:
            start = float(t[idx])
        if start is not None and ((not active) or idx == len(mask) - 1):
            end_idx = idx if active and idx == len(mask) - 1 else max(idx - 1, 0)
            spans.append((start, float(t[end_idx])))
            start = None
    return spans


def _setup_axis(ax, title: str, ylabel: str, xlim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(*xlim)
    ax.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.tick_params(labelsize=8)


def _key_indices(
    t: np.ndarray,
    risk_f: np.ndarray,
    lat: np.ndarray,
    risk_threshold: float,
    lat_threshold: float,
    max_events: int,
    min_separation_s: float = 2.0,
) -> list[int]:
    """Pick conflict instants with large risk and lateral-command change."""
    risk_span = max(float(np.nanmax(risk_f) - np.nanmin(risk_f)), 1e-6)
    lat_span = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-6)
    score = (risk_f - np.nanmin(risk_f)) / risk_span + 1.8 * (lat - np.nanmin(lat)) / lat_span
    mask = (risk_f >= risk_threshold) & (lat >= lat_threshold)
    candidates = [int(i) for i in np.argsort(score)[::-1] if mask[int(i)]]
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(float(t[idx] - t[j])) >= min_separation_s for j in chosen):
            chosen.append(idx)
        if len(chosen) >= max_events:
            break
    return sorted(chosen)


def build_animation(args: argparse.Namespace) -> None:
    data = _read_csv(Path(args.input_csv))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = data["time_s"]
    risk_f = data["risk_F"]
    front_risk = data.get("front_distance_risk", risk_f)
    raw_y = data["y"]
    y_risk = data["y_risk"]
    y_eff = data["y_eff"]
    delta_w = data["delta_y_w"]
    lat = data["cmd_safe_x_abs_delta_from_pre_conflict"]
    fwd = data["cmd_safe_y"]
    baseline = float(data["cmd_safe_x_abs_pre_conflict_baseline"][0])
    conflict = data.get("high_risk_conflict", np.zeros_like(t)) > 0.5
    key_idxs = []
    key_times: list[float] = []
    if args.annotate:
        key_idxs = _key_indices(
            t,
            risk_f,
            lat,
            args.annotate_risk_threshold,
            args.annotate_lat_threshold,
            args.annotate_max_events,
        )
        key_times = [float(t[idx]) for idx in key_idxs]

    xlim = (float(np.nanmin(t)), float(np.nanmax(t)))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 5.5),
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.96, bottom=0.08, hspace=0.34)

    for ax in axes:
        for a, b in _spans(t, conflict):
            ax.axvspan(a, b, color=TOKENS["conflict"], alpha=0.22, lw=0)
        if args.annotate:
            for key_t in key_times:
                ax.axvspan(
                    key_t - args.annotate_window_s,
                    key_t + args.annotate_window_s,
                    color=TOKENS["highlight"],
                    alpha=0.18,
                    lw=0,
                )

    _setup_axis(axes[0], "(a) D435i-derived risk", "Risk", xlim)
    axes[0].set_ylim(0.0, max(0.78, float(np.nanmax(risk_f)) + 0.05))
    line_risk, = axes[0].plot([], [], color=TOKENS["risk_f"], lw=2.0, label="Follow risk")
    line_front, = axes[0].plot([], [], color=TOKENS["front_risk"], lw=1.5, ls=":", label="Front risk")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    _setup_axis(axes[1], "(b) Arbitration response", "Weight", xlim)
    axes[1].set_ylim(0.42, 0.78)
    line_raw, = axes[1].plot([], [], color=TOKENS["raw_y"], lw=1.5, label="Raw y")
    line_risk_y, = axes[1].plot([], [], color=TOKENS["risk_y"], lw=1.6, ls="--", label=r"Risk-only $y+\Delta y_r$")
    line_eff, = axes[1].plot([], [], color=TOKENS["eff_y"], lw=2.0, label=r"Final $y_{eff}$")
    fill_w = [None]
    axes[1].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    _setup_axis(axes[2], "(c) Command modulation", "Command [m/s]", xlim)
    axes[2].set_ylabel(r"$\Delta |v_x|$ [m/s]", fontsize=9)
    axes[2].set_ylim(
        min(-0.035, float(np.nanmin(lat)) - 0.01),
        max(0.11, float(np.nanmax(lat)) + 0.015),
    )
    line_lat, = axes[2].plot([], [], color=TOKENS["lat"], lw=2.0, label=r"$\Delta |v_x|$")
    axes[2].axhline(0.0, color=TOKENS["baseline"], lw=1.0, ls=":", label="baseline")
    forward_axis = axes[2].twinx()
    forward_axis.set_ylim(
        max(0.0, float(np.nanmin(fwd)) - 0.06),
        float(np.nanmax(fwd)) + 0.06,
    )
    forward_axis.spines["top"].set_visible(False)
    forward_axis.spines["right"].set_color(TOKENS["baseline"])
    forward_axis.tick_params(colors=TOKENS["fwd"], labelsize=8)
    forward_axis.set_ylabel(r"Forward $v_y$ [m/s]", color=TOKENS["fwd"], fontsize=9)
    line_fwd, = forward_axis.plot([], [], color=TOKENS["fwd"], lw=1.7, ls="--", label=r"$v_y$")
    axes[2].set_xlabel("Time [s]", fontsize=9)
    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = forward_axis.get_legend_handles_labels()
    axes[2].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=8,
        frameon=False,
        ncol=3,
        borderaxespad=0.05,
    )

    cursors = [ax.axvline(t[0], color=TOKENS["cursor"], lw=1.4, ls=(0, (2, 2))) for ax in axes]
    time_text = axes[0].text(
        0.985,
        0.90,
        "",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )
    marker_kwargs = dict(
        s=190,
        facecolors="none",
        edgecolors=TOKENS["highlight"],
        linewidths=2.2,
        zorder=9,
    )
    key_marker_risk = axes[0].scatter([], [], **marker_kwargs)
    key_marker_y = axes[1].scatter([], [], **marker_kwargs)
    key_marker_cmd = axes[2].scatter([], [], **marker_kwargs)

    frames = len(t)
    if args.duration_s > 0.0:
        frames = max(2, int(round(float(args.duration_s) * float(args.fps))))

    def update(frame_idx: int):
        if frames <= len(t):
            i = min(frame_idx + 1, len(t))
        else:
            i = min(int(round(1 + frame_idx * (len(t) - 1) / max(frames - 1, 1))), len(t))
        tt = t[:i]
        line_risk.set_data(tt, risk_f[:i])
        line_front.set_data(tt, front_risk[:i])
        line_raw.set_data(tt, raw_y[:i])
        line_risk_y.set_data(tt, y_risk[:i])
        line_eff.set_data(tt, y_eff[:i])
        if fill_w[0] is not None:
            fill_w[0].remove()
        fill_w[0] = axes[1].fill_between(
            tt,
            y_risk[:i],
            y_eff[:i],
            color=TOKENS["learned_fill"],
            alpha=0.55,
            linewidth=0,
            label=r"$\Delta y_w$" if frame_idx == 0 else None,
        )
        line_lat.set_data(tt, lat[:i])
        line_fwd.set_data(tt, fwd[:i])
        for cursor in cursors:
            cursor.set_xdata([float(t[i - 1]), float(t[i - 1])])
        current_t = float(t[i - 1])
        time_text.set_text(f"t = {current_t:.1f} s")
        active_idx = None
        if args.annotate and key_idxs:
            nearest = min(key_idxs, key=lambda idx: abs(float(t[idx]) - current_t))
            if abs(float(t[nearest]) - current_t) <= args.annotate_window_s:
                active_idx = nearest
        if active_idx is None:
            empty = np.empty((0, 2))
            key_marker_risk.set_offsets(empty)
            key_marker_y.set_offsets(empty)
            key_marker_cmd.set_offsets(empty)
        else:
            x = float(t[active_idx])
            key_marker_risk.set_offsets([[x, float(risk_f[active_idx])]])
            key_marker_y.set_offsets([[x, float(y_eff[active_idx])]])
            key_marker_cmd.set_offsets([[x, float(lat[active_idx])]])
        return [
            line_risk,
            line_front,
            line_raw,
            line_risk_y,
            line_eff,
            line_lat,
            line_fwd,
            *cursors,
            time_text,
            key_marker_risk,
            key_marker_y,
            key_marker_cmd,
            fill_w[0],
        ]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / args.fps, blit=False)
    if out.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=args.bitrate,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    anim.save(out, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        default="agents/final_paper_outputs_v3/fig7_real_robot_arbitration_data.csv",
    )
    parser.add_argument(
        "--output",
        default="agents/final_paper_outputs_v3/fig7_real_robot_arbitration_dynamic.mp4",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--duration_s", type=float, default=0.0, help="Override animation duration; <=0 uses one frame per CSV sample.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--bitrate", type=int, default=5000)
    parser.add_argument("--annotate", action="store_true", help="Circle conflict moments and show mechanism subtitles.")
    parser.add_argument("--annotate_risk_threshold", type=float, default=0.45)
    parser.add_argument("--annotate_lat_threshold", type=float, default=0.04)
    parser.add_argument("--annotate_max_events", type=int, default=2)
    parser.add_argument("--annotate_window_s", type=float, default=1.15)
    return parser.parse_args()


if __name__ == "__main__":
    build_animation(parse_args())
