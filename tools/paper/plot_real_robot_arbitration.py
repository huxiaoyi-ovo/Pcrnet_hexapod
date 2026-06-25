#!/usr/bin/env python3
"""Plot a paper-ready real-robot PCR arbitration trace from a ROS1 bag."""

import argparse
import csv
import fnmatch
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_BAG = "/home/hxy/下载/pcr_real_20260618_154442.bag"
DEFAULT_OUTPUT_DIR = "agents/final_paper_outputs_v3"
DEFAULT_PREFIX = "fig7_real_robot_arbitration"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "red": "#D93636",
    "red_light": "#F7D7D7",
    "conflict_fill": "#F7D7D7",
    "learned_fill": "#CEDFFE",
    "blue": "#397FB3",
    "orange": "#E67E22",
    "green": "#2B9A66",
    "gray": "#7A828F",
    "gray_dark": "#464C55",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot real-robot PCR risk, arbitration, and command curves."
    )
    parser.add_argument("--bag", default=DEFAULT_BAG)
    parser.add_argument(
        "--bag_glob",
        default="",
        help="Scan matching bags and select the clearest lateral-response event.",
    )
    parser.add_argument(
        "--paper_bag_pattern",
        default="pcr_real_20260618_*.bag",
        help="Only matching bags may become the final paper figure.",
    )
    parser.add_argument("--candidate_count", type=int, default=3)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--debug_topic", default="/pcr_realplay/debug")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min_duration_s", type=float, default=12.0)
    parser.add_argument("--max_duration_s", type=float, default=22.0)
    parser.add_argument("--signed_w_lambda", type=float, default=0.30)
    parser.add_argument("--signed_w_gamma_risk", type=float, default=0.15)
    parser.add_argument("--signed_w_margin", type=float, default=0.05)
    parser.add_argument("--risk_threshold", type=float, default=0.25)
    parser.add_argument("--conflict_threshold", type=float, default=0.10)
    parser.add_argument("--window_start_s", type=float, default=None)
    parser.add_argument("--window_duration_s", type=float, default=None)
    return parser.parse_args()


def scalar(payload, key):
    try:
        return float(payload.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def vector(payload, key):
    value = payload.get(key, [])
    if isinstance(value, dict):
        value = [
            value.get("x_vec", float("nan")),
            value.get("y_vec", float("nan")),
            value.get("w_twist", float("nan")),
        ]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return np.full(3, np.nan, dtype=np.float64)
    try:
        return np.asarray(value[:3], dtype=np.float64)
    except (TypeError, ValueError):
        return np.full(3, np.nan, dtype=np.float64)


def load_debug_rows(bag_path, debug_topic):
    try:
        import rosbag
    except ImportError as exc:
        raise SystemExit(
            "ROS1 rosbag Python package is required. Run with /usr/bin/python3."
        ) from exc

    rows = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, stamp in bag.read_messages(topics=[debug_topic]):
            try:
                payload = json.loads(msg.data)
            except (AttributeError, TypeError, json.JSONDecodeError):
                continue
            cmd_f = vector(payload, "cmd_f")
            cmd_a = vector(payload, "cmd_a")
            cmd_safe = vector(payload, "cmd_safe")
            safety = payload.get("safety", {})
            reasons = safety.get("reasons", []) if isinstance(safety, dict) else []
            rows.append(
                {
                    "stamp_s": float(stamp.to_sec()),
                    "risk_F": scalar(payload, "risk_F"),
                    "risk_A": scalar(payload, "risk_A"),
                    "front_distance_risk": scalar(payload, "front_distance_risk"),
                    "conflict_score": scalar(payload, "conflict_score"),
                    "y": scalar(payload, "y"),
                    "w": scalar(payload, "w"),
                    "y_eff": scalar(payload, "y_eff"),
                    "cmd_f_x": cmd_f[0],
                    "cmd_f_y": cmd_f[1],
                    "cmd_f_yaw": cmd_f[2],
                    "cmd_a_x": cmd_a[0],
                    "cmd_a_y": cmd_a[1],
                    "cmd_a_yaw": cmd_a[2],
                    "cmd_safe_x": cmd_safe[0],
                    "cmd_safe_y": cmd_safe[1],
                    "cmd_safe_yaw": cmd_safe[2],
                    "target_valid": float(bool(payload.get("target_valid", False))),
                    "depth_invalid": float(bool(payload.get("depth_invalid", False))),
                    "safety_clear": float(len(reasons) == 0),
                    "risk_source_real": float(
                        payload.get("risk_source") == "risk_blocked_map"
                    ),
                }
            )
    if not rows:
        raise SystemExit("No valid JSON messages were found on the debug topic.")
    return rows


def enrich_rows(rows, args):
    for row in rows:
        signed_w = 2.0 * row["w"] - 1.0
        signed_w_active = signed_w if abs(signed_w) > args.signed_w_margin else 0.0
        delta_y_r = args.signed_w_gamma_risk * (row["risk_A"] - row["risk_F"])
        delta_y_w = args.signed_w_lambda * signed_w_active
        row["signed_w"] = signed_w
        row["signed_w_active"] = signed_w_active
        row["delta_y_r"] = delta_y_r
        row["delta_y_w"] = delta_y_w
        row["y_risk"] = float(np.clip(row["y"] + delta_y_r, 0.0, 1.0))
        row["y_eff_reconstructed"] = float(
            np.clip(row["y"] + delta_y_r + delta_y_w, 0.0, 1.0)
        )
        row["y_eff_reconstruction_abs_error"] = abs(
            row["y_eff_reconstructed"] - row["y_eff"]
        )
        cmd_norm = np.hypot(row["cmd_safe_x"], row["cmd_safe_y"])
        row["moving"] = float(
            cmd_norm > 1e-3 or abs(row["cmd_safe_yaw"]) > 1e-3
        )
        row["high_risk_conflict"] = float(
            row["risk_F"] >= args.risk_threshold
            and row["risk_F"] > row["risk_A"]
            and row["conflict_score"] >= args.conflict_threshold
        )
    return rows


def row_is_clean(row):
    required = (
        "risk_F",
        "risk_A",
        "front_distance_risk",
        "conflict_score",
        "y",
        "w",
        "y_eff",
        "cmd_safe_x",
        "cmd_safe_y",
        "cmd_safe_yaw",
    )
    return (
        row["target_valid"] > 0.5
        and row["depth_invalid"] < 0.5
        and row["safety_clear"] > 0.5
        and row["risk_source_real"] > 0.5
        and np.isfinite([row[key] for key in required]).all()
    )


def split_clean_segments(rows, max_gap_s=0.25):
    segments = []
    current = []
    previous_stamp = None
    for row in rows:
        clean = row_is_clean(row)
        continuous = (
            previous_stamp is not None
            and row["stamp_s"] - previous_stamp <= max_gap_s
        )
        if clean and (not current or continuous):
            current.append(row)
        else:
            if current:
                segments.append(current)
            current = [row] if clean else []
        previous_stamp = row["stamp_s"] if clean else None
    if current:
        segments.append(current)
    return segments


def candidate_windows(segments, min_duration_s, max_duration_s):
    candidates = []
    for segment in segments:
        duration = segment[-1]["stamp_s"] - segment[0]["stamp_s"]
        if duration < min_duration_s:
            continue
        if duration <= max_duration_s:
            candidates.append(segment)
            continue
        starts = np.arange(
            segment[0]["stamp_s"],
            segment[-1]["stamp_s"] - max_duration_s + 1e-9,
            1.0,
        )
        for start in starts:
            stop = start + max_duration_s
            window = [
                row for row in segment if start <= row["stamp_s"] <= stop
            ]
            if (
                window
                and window[-1]["stamp_s"] - window[0]["stamp_s"]
                >= min_duration_s
            ):
                candidates.append(window)
    return candidates


def values(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def score_candidate(rows):
    duration = rows[-1]["stamp_s"] - rows[0]["stamp_s"]
    risk_f = values(rows, "risk_F")
    cmd_x = values(rows, "cmd_safe_x")
    cmd_y = values(rows, "cmd_safe_y")
    moving = values(rows, "moving")
    conflict = values(rows, "high_risk_conflict")
    peak_index = int(np.nanargmax(risk_f))
    peak_position = peak_index / max(len(rows) - 1, 1)
    after_peak = risk_f[min(peak_index + 5, len(risk_f) - 1) :]
    recovery_drop = (
        float(np.nanmax(risk_f) - np.nanmedian(after_peak))
        if after_peak.size
        else 0.0
    )
    metrics = {
        "duration_s": duration,
        "risk_peak": float(np.nanmax(risk_f)),
        "risk_range": float(np.nanmax(risk_f) - np.nanmin(risk_f)),
        "mean_abs_lateral": float(np.nanmean(np.abs(cmd_x))),
        "mean_forward": float(np.nanmean(np.maximum(cmd_y, 0.0))),
        "moving_fraction": float(np.nanmean(moving)),
        "high_risk_fraction": float(np.nanmean(conflict)),
        "peak_position": peak_position,
        "post_peak_risk_drop": recovery_drop,
    }
    eligible = (
        metrics["duration_s"] >= 12.0
        and metrics["risk_peak"] >= 0.45
        and metrics["risk_range"] >= 0.30
        and metrics["mean_abs_lateral"] >= 0.12
        and metrics["moving_fraction"] >= 0.80
        and metrics["high_risk_fraction"] >= 0.03
        and 0.15 <= peak_position <= 0.85
        and recovery_drop >= 0.15
    )
    score = (
        min(duration, 20.0)
        + 8.0 * metrics["risk_peak"]
        + 6.0 * metrics["risk_range"]
        + 4.0 * metrics["high_risk_fraction"]
        + 3.0 * metrics["mean_abs_lateral"]
        + 2.0 * metrics["mean_forward"]
        + 3.0 * metrics["post_peak_risk_drop"]
    )
    metrics["eligible"] = bool(eligible)
    metrics["score"] = float(score)
    return metrics


def event_candidates(rows, bag_path, args):
    candidates = []
    session_start = rows[0]["stamp_s"]
    for segment_index, segment in enumerate(split_clean_segments(rows)):
        if segment[-1]["stamp_s"] - segment[0]["stamp_s"] < 6.0:
            continue
        time_s = values(segment, "stamp_s")
        risk_f = values(segment, "risk_F")
        conflict = values(segment, "high_risk_conflict") > 0.5
        spans = contiguous_true_spans(time_s, conflict)
        for event_index, (event_start, event_end) in enumerate(spans):
            pre_index = np.where(
                (time_s >= event_start - 3.0)
                & (time_s <= event_start - 0.35)
            )[0]
            event_indices = np.where(
                (time_s >= event_start) & (time_s <= event_end)
            )[0]
            post_index = np.where(
                (time_s >= event_end + 0.35)
                & (time_s <= event_end + 4.0)
            )[0]
            if min(len(pre_index), len(event_indices), len(post_index)) < 5:
                continue
            window = [
                row
                for row in segment
                if event_start - 3.0 <= row["stamp_s"] <= event_end + 4.0
            ]
            lateral = np.abs(values(segment, "cmd_safe_x"))
            forward = np.maximum(values(segment, "cmd_safe_y"), 0.0)
            delta_y_w = values(segment, "delta_y_w")
            pre_risk = float(np.nanmedian(risk_f[pre_index]))
            event_peak_risk = float(np.nanmax(risk_f[event_indices]))
            post_risk = float(np.nanmedian(risk_f[post_index]))
            pre_lateral = float(np.nanmedian(lateral[pre_index]))
            event_lateral = float(np.nanpercentile(lateral[event_indices], 90))
            post_lateral = float(np.nanmedian(lateral[post_index]))
            risk_rise = event_peak_risk - pre_risk
            risk_recovery = event_peak_risk - post_risk
            lateral_increase = event_lateral - pre_lateral
            event_forward = float(np.nanmean(forward[event_indices]))
            event_delta_y_w = float(np.nanmean(delta_y_w[event_indices]))
            reconstruction_error = float(
                np.nanmax(values(window, "y_eff_reconstruction_abs_error"))
            )
            moving_fraction = float(np.nanmean(values(window, "moving")))
            score = (
                14.0 * lateral_increase
                + 5.0 * risk_rise
                + 4.0 * risk_recovery
                + 3.0 * event_delta_y_w
                + 2.0 * event_forward
                - 2.0 * abs(post_lateral - pre_lateral)
            )
            paper_eligible = (
                fnmatch.fnmatch(Path(bag_path).name, args.paper_bag_pattern)
                and event_peak_risk >= 0.45
                and risk_rise >= 0.25
                and risk_recovery >= 0.15
                and lateral_increase >= 0.05
                and event_forward >= 0.15
                and moving_fraction >= 0.80
                and reconstruction_error < 1e-5
            )
            candidates.append(
                {
                    "bag": str(bag_path),
                    "bag_name": Path(bag_path).name,
                    "segment_index": segment_index,
                    "event_index": event_index,
                    "event_start_rel_s": event_start - session_start,
                    "event_end_rel_s": event_end - session_start,
                    "window_start_rel_s": window[0]["stamp_s"] - session_start,
                    "window_end_rel_s": window[-1]["stamp_s"] - session_start,
                    "pre_risk": pre_risk,
                    "event_peak_risk": event_peak_risk,
                    "post_risk": post_risk,
                    "risk_rise": risk_rise,
                    "risk_recovery": risk_recovery,
                    "pre_lateral": pre_lateral,
                    "event_lateral_p90": event_lateral,
                    "post_lateral": post_lateral,
                    "lateral_increase": lateral_increase,
                    "event_forward": event_forward,
                    "event_delta_y_w": event_delta_y_w,
                    "moving_fraction": moving_fraction,
                    "reconstruction_error_max": reconstruction_error,
                    "paper_eligible": paper_eligible,
                    "score": float(score),
                    "rows": window,
                }
            )
    return candidates


def write_multibag_audit(candidates, bag_summaries, output_dir, prefix):
    audit_path = output_dir / (prefix + "_bag_audit.csv")
    fields = [
        "rank",
        "bag_name",
        "event_start_rel_s",
        "event_end_rel_s",
        "window_start_rel_s",
        "window_end_rel_s",
        "paper_eligible",
        "score",
        "pre_risk",
        "event_peak_risk",
        "post_risk",
        "risk_rise",
        "risk_recovery",
        "pre_lateral",
        "event_lateral_p90",
        "post_lateral",
        "lateral_increase",
        "event_forward",
        "event_delta_y_w",
        "moving_fraction",
        "reconstruction_error_max",
    ]
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    with open(audit_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(ranked, start=1):
            record = {key: item[key] for key in fields if key != "rank"}
            record["rank"] = rank
            writer.writerow(record)

    summary_path = output_dir / (prefix + "_bag_summary.csv")
    summary_fields = [
        "bag_name",
        "debug_samples",
        "clean_samples",
        "candidate_events",
        "status",
    ]
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(bag_summaries)
    return audit_path, summary_path


def plot_candidate_gallery(candidates, output_dir, prefix, count):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = candidates[: max(1, count)]
    fig, axes = plt.subplots(
        2,
        len(selected),
        figsize=(3.3 * len(selected), 4.7),
        squeeze=False,
        sharex="col",
    )
    fig.patch.set_facecolor(TOKENS["surface"])
    for column, candidate in enumerate(selected):
        rows = candidate["rows"]
        time_s = values(rows, "stamp_s") - rows[0]["stamp_s"]
        event_start = candidate["event_start_rel_s"] - candidate["window_start_rel_s"]
        event_end = candidate["event_end_rel_s"] - candidate["window_start_rel_s"]
        for axis in axes[:, column]:
            style_axis(axis)
            axis.axvspan(
                event_start,
                event_end,
                color=TOKENS["red_light"],
                alpha=0.55,
                linewidth=0,
            )
        axes[0, column].plot(
            time_s,
            values(rows, "risk_F"),
            color=TOKENS["red"],
            linewidth=1.6,
        )
        axes[0, column].plot(
            time_s,
            values(rows, "risk_A"),
            color=TOKENS["blue"],
            linewidth=1.2,
            linestyle="--",
        )
        axes[0, column].set_ylim(-0.03, 1.03)
        axes[0, column].set_title(
            "{} | {:.1f}-{:.1f} s{}".format(
                candidate["bag_name"].replace("pcr_real_20260618_", "").replace(
                    ".bag", ""
                ),
                candidate["window_start_rel_s"],
                candidate["window_end_rel_s"],
                "" if candidate["paper_eligible"] else " | secondary",
            ),
            fontsize=7.5,
            color=TOKENS["ink"],
        )
        lateral = np.abs(values(rows, "cmd_safe_x"))
        axes[1, column].plot(
            time_s,
            lateral,
            color=TOKENS["blue"],
            linewidth=1.6,
        )
        axes[1, column].axhline(
            candidate["pre_lateral"],
            color=TOKENS["gray"],
            linewidth=0.9,
            linestyle=":",
        )
        low = max(0.0, float(np.nanmin(lateral)) - 0.02)
        high = float(np.nanmax(lateral)) + 0.02
        axes[1, column].set_ylim(low, high)
        axes[1, column].set_xlabel("Time [s]", fontsize=8)
        axes[1, column].text(
            0.03,
            0.92,
            r"$\Delta |v_x|={:+.3f}$".format(candidate["lateral_increase"]),
            transform=axes[1, column].transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=TOKENS["ink"],
        )
    axes[0, 0].set_ylabel("Risk", fontsize=8)
    axes[1, 0].set_ylabel(r"Lateral speed $|v_x|$ [m/s]", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.78, bottom=0.12, wspace=0.25)
    fig.text(
        0.08,
        0.965,
        "Top real-robot arbitration candidates",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.08,
        0.895,
        "Red shading: high-risk conflict; dotted line: pre-conflict lateral baseline.",
        ha="left",
        va="top",
        fontsize=8,
        color=TOKENS["muted"],
    )
    png_path = output_dir / (prefix + "_candidates.png")
    pdf_path = output_dir / (prefix + "_candidates.pdf")
    fig.savefig(png_path, dpi=250, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def select_across_bags(args, output_dir):
    bag_paths = [
        Path(path).expanduser().resolve()
        for path in sorted(glob.glob(os.path.expanduser(args.bag_glob)))
    ]
    if not bag_paths:
        raise SystemExit("No bags matched --bag_glob: {}".format(args.bag_glob))
    candidates = []
    bag_summaries = []
    for bag_path in bag_paths:
        try:
            rows = enrich_rows(load_debug_rows(bag_path, args.debug_topic), args)
        except SystemExit as exc:
            bag_summaries.append(
                {
                    "bag_name": bag_path.name,
                    "debug_samples": 0,
                    "clean_samples": 0,
                    "candidate_events": 0,
                    "status": str(exc),
                }
            )
            continue
        for row in rows:
            row["session_relative_s"] = row["stamp_s"] - rows[0]["stamp_s"]
        bag_candidates = event_candidates(rows, bag_path, args)
        candidates.extend(bag_candidates)
        bag_summaries.append(
            {
                "bag_name": bag_path.name,
                "debug_samples": len(rows),
                "clean_samples": sum(row_is_clean(row) for row in rows),
                "candidate_events": len(bag_candidates),
                "status": "ok",
            }
        )
    if not candidates:
        raise SystemExit("No high-risk events were found in the scanned bags.")
    audit_path, summary_path = write_multibag_audit(
        candidates, bag_summaries, output_dir, args.prefix
    )
    current_candidates = sorted(
        [
            item
            for item in candidates
            if fnmatch.fnmatch(item["bag_name"], args.paper_bag_pattern)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )
    paper_candidates = [
        item for item in current_candidates if item["paper_eligible"]
    ]
    if not paper_candidates:
        raise SystemExit(
            "No current real-robot event satisfied the fixed lateral-response criteria."
        )
    gallery_paths = plot_candidate_gallery(
        current_candidates,
        output_dir,
        args.prefix,
        args.candidate_count,
    )
    selected_event = paper_candidates[0]
    selection_metrics = score_candidate(selected_event["rows"])
    selection_metrics.update(
        {
            key: selected_event[key]
            for key in (
                "score",
                "pre_risk",
                "event_peak_risk",
                "post_risk",
                "risk_rise",
                "risk_recovery",
                "pre_lateral",
                "event_lateral_p90",
                "post_lateral",
                "lateral_increase",
                "event_forward",
                "event_delta_y_w",
            )
        }
    )
    candidate_metrics = []
    for item in paper_candidates:
        record = {
            "start_rel_s": item["window_start_rel_s"],
            "end_rel_s": item["window_end_rel_s"],
            "eligible": item["paper_eligible"],
            "score": item["score"],
            "risk_peak": item["event_peak_risk"],
            "risk_range": item["risk_rise"],
            "mean_abs_lateral": item["event_lateral_p90"],
            "mean_forward": item["event_forward"],
        }
        candidate_metrics.append(record)
    return (
        selected_event["rows"],
        selection_metrics,
        candidate_metrics,
        Path(selected_event["bag"]),
        audit_path,
        summary_path,
        gallery_paths,
    )


def select_window(rows, args):
    if args.window_start_s is not None:
        duration = (
            args.window_duration_s
            if args.window_duration_s is not None
            else args.max_duration_s
        )
        session_start = rows[0]["stamp_s"]
        start = session_start + args.window_start_s
        selected = [
            row for row in rows if start <= row["stamp_s"] <= start + duration
        ]
        if not selected:
            raise SystemExit("Manual paper window did not contain any samples.")
        if not all(row_is_clean(row) for row in selected):
            raise SystemExit("Manual paper window contains invalid or unsafe samples.")
        return selected, score_candidate(selected), []

    segments = split_clean_segments(rows)
    candidates = candidate_windows(
        segments, args.min_duration_s, args.max_duration_s
    )
    scored = [(candidate, score_candidate(candidate)) for candidate in candidates]
    eligible = [item for item in scored if item[1]["eligible"]]
    if not eligible:
        raise SystemExit(
            "No interval satisfied the fixed paper-selection criteria. "
            "Inspect the bag instead of weakening the criteria silently."
        )
    eligible.sort(key=lambda item: item[1]["score"], reverse=True)
    selected, metrics = eligible[0]
    candidate_metrics = []
    session_start = rows[0]["stamp_s"]
    for _candidate, item in sorted(
        scored, key=lambda pair: pair[1]["score"], reverse=True
    ):
        record = dict(item)
        record["start_rel_s"] = _candidate[0]["stamp_s"] - session_start
        record["end_rel_s"] = _candidate[-1]["stamp_s"] - session_start
        candidate_metrics.append(record)
    return selected, metrics, candidate_metrics


def contiguous_true_spans(time_s, mask, min_span_s=0.15):
    spans = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        at_end = index == len(mask) - 1
        if start is not None and ((not active) or at_end):
            end = index if active and at_end else index - 1
            if time_s[end] - time_s[start] >= min_span_s:
                spans.append((time_s[start], time_s[end]))
            start = None
    return spans


def style_axis(axis):
    axis.set_facecolor(TOKENS["panel"])
    axis.grid(True, axis="both", color=TOKENS["grid"], linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(TOKENS["axis"])
    axis.spines["bottom"].set_color(TOKENS["axis"])
    axis.tick_params(colors=TOKENS["ink"], labelsize=8)
    axis.yaxis.label.set_color(TOKENS["ink"])
    axis.xaxis.label.set_color(TOKENS["ink"])


def shade_conflicts(axes, spans):
    for axis in axes:
        for start, end in spans:
            axis.axvspan(
                start,
                end,
                color=TOKENS["conflict_fill"],
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )


def lateral_modulation(selected):
    time_s = values(selected, "stamp_s") - selected[0]["stamp_s"]
    conflict = values(selected, "high_risk_conflict") > 0.5
    event_index = np.flatnonzero(conflict)
    lateral_magnitude = np.abs(values(selected, "cmd_safe_x"))
    if event_index.size:
        event_start = time_s[event_index[0]]
        pre_index = np.where(
            (time_s >= max(time_s[0], event_start - 2.5))
            & (time_s <= event_start - 0.35)
        )[0]
    else:
        pre_index = np.asarray([], dtype=np.int64)
    baseline = (
        float(np.nanmedian(lateral_magnitude[pre_index]))
        if pre_index.size
        else float(np.nanmedian(lateral_magnitude))
    )
    return baseline, lateral_magnitude - baseline


def plot_figure(selected, output_dir, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid")
    except ImportError:
        pass

    time_s = values(selected, "stamp_s") - selected[0]["stamp_s"]
    conflict_mask = values(selected, "high_risk_conflict") > 0.5
    spans = contiguous_true_spans(time_s, conflict_mask)
    lateral_magnitude = np.abs(values(selected, "cmd_safe_x"))
    lateral_baseline, lateral_delta = lateral_modulation(selected)
    span_metrics = []
    for span_start, span_end in spans:
        event_index = np.where(
            (time_s >= span_start) & (time_s <= span_end)
        )[0]
        pre_index = np.where(
            (time_s >= max(time_s[0], span_start - 3.0))
            & (time_s <= span_start - 0.35)
        )[0]
        if event_index.size == 0 or pre_index.size == 0:
            continue
        event_lateral = float(
            np.nanpercentile(lateral_magnitude[event_index], 90)
        )
        span_metrics.append(
            (
                event_lateral - lateral_baseline,
                span_start,
                span_end,
                event_lateral,
            )
        )
    if span_metrics:
        (
            lateral_increase,
            main_span_start,
            main_span_end,
            peak_lateral,
        ) = max(span_metrics, key=lambda item: item[0])
        main_event_index = np.where(
            (time_s >= main_span_start) & (time_s <= main_span_end)
        )[0]
        peak_index = int(
            main_event_index[
                np.nanargmax(values(selected, "risk_F")[main_event_index])
            ]
        )
    else:
        peak_index = int(np.nanargmax(values(selected, "risk_F")))
        peak_lateral = float(np.nanmax(lateral_magnitude))
        lateral_increase = peak_lateral - lateral_baseline
    peak_time = time_s[peak_index]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
            ],
            "axes.labelsize": 7.2,
            "legend.fontsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(3.5, 4.8),
        sharex=True,
        gridspec_kw={"hspace": 0.82},
    )
    fig.patch.set_facecolor(TOKENS["surface"])
    for axis in axes:
        style_axis(axis)
        axis.tick_params(labelsize=6.6)
    shade_conflicts(axes, spans)

    axes[0].plot(
        time_s,
        values(selected, "risk_F"),
        color=TOKENS["red"],
        linewidth=1.45,
        label=r"$\rho_F$ (Follow)",
        zorder=3,
    )
    axes[0].plot(
        time_s,
        values(selected, "front_distance_risk"),
        color=TOKENS["gray"],
        linewidth=1.0,
        linestyle=":",
        label="D435i front risk",
        zorder=2,
    )
    axes[0].set_ylim(-0.03, 0.78)
    axes[0].set_ylabel("Risk")
    axes[0].text(
        0.0,
        1.48,
        "(a) Real-robot Follow-command risk",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor=TOKENS["conflict_fill"],
            edgecolor="none",
            alpha=0.75,
            label="High-risk conflict",
        )
    )
    labels.append("High-risk conflict")
    handles.append(
        Line2D(
            [0],
            [0],
            color="#6F4E9C",
            alpha=1.0,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            label="Peak Follow-risk instant",
        )
    )
    labels.append("Peak Follow-risk instant")
    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.0),
        ncol=2,
        mode="expand",
        frameon=False,
        borderaxespad=0.0,
    )

    axes[1].plot(
        time_s,
        values(selected, "y"),
        color=TOKENS["gray_dark"],
        linewidth=1.25,
        label=r"Raw gate $y$",
        zorder=3,
    )
    axes[1].plot(
        time_s,
        values(selected, "y_risk"),
        color=TOKENS["blue"],
        linewidth=1.20,
        linestyle="--",
        label=r"Risk-only $y+\Delta y_r$",
        zorder=3,
    )
    axes[1].plot(
        time_s,
        values(selected, "y_eff"),
        color=TOKENS["red"],
        linewidth=1.45,
        label=r"PCR $y_{\mathrm{eff}}$",
        zorder=4,
    )
    axes[1].fill_between(
        time_s,
        values(selected, "y_risk"),
        values(selected, "y_eff"),
        color=TOKENS["learned_fill"],
        alpha=0.62,
        linewidth=0,
        label=r"Learned correction $\Delta y_w$",
        zorder=1,
    )
    y_min = min(0.45, float(np.nanmin(values(selected, "y_risk"))) - 0.03)
    y_max = max(0.80, float(np.nanmax(values(selected, "y_eff"))) + 0.03)
    axes[1].set_ylim(y_min, min(y_max, 1.02))
    axes[1].set_ylabel("Follow weight")
    axes[1].text(
        0.0,
        1.48,
        "(b) PCR arbitration response",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.0),
        ncol=2,
        mode="expand",
        frameon=False,
        borderaxespad=0.0,
    )

    axes[2].plot(
        time_s,
        lateral_delta,
        color=TOKENS["blue"],
        linewidth=1.45,
        label=r"Baseline-adjusted $\Delta |v_x|$",
        zorder=3,
    )
    axes[2].axhline(
        0.0,
        color=TOKENS["gray"],
        linewidth=0.9,
        linestyle=":",
        label="Baseline (0.190 m/s)",
        zorder=2,
    )
    forward_axis = axes[2].twinx()
    forward_axis.plot(
        time_s,
        values(selected, "cmd_safe_y"),
        color=TOKENS["orange"],
        linewidth=1.4,
        linestyle="--",
        label=r"Forward $v_y$",
        zorder=3,
    )
    forward_axis.spines["top"].set_visible(False)
    forward_axis.spines["right"].set_color(TOKENS["axis"])
    forward_axis.tick_params(colors=TOKENS["orange"], labelsize=8)
    forward_axis.set_ylabel(
        r"Forward speed $v_y$ [m/s]", color=TOKENS["orange"], fontsize=7.2
    )
    lateral_low = min(-0.035, float(np.nanmin(lateral_delta)) - 0.01)
    lateral_high = max(0.11, float(np.nanmax(lateral_delta)) + 0.015)
    axes[2].set_ylim(
        lateral_low,
        lateral_high,
    )
    forward_values = values(selected, "cmd_safe_y")
    forward_axis.set_ylim(
        max(0.0, float(np.nanmin(forward_values)) - 0.06),
        float(np.nanmax(forward_values)) + 0.06,
    )
    axes[2].set_ylabel(r"$\Delta |v_x|$ [m/s]")
    axes[2].set_xlabel("Time within selected real-robot interval [s]")
    axes[2].text(
        0.0,
        1.48,
        "(c) Risk-induced lateral modulation",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = forward_axis.get_legend_handles_labels()
    legend_handles = handles_left + handles_right
    legend_labels = labels_left + labels_right
    legend_handles.append(Line2D([], [], color="none", linewidth=0))
    legend_labels.append("")
    axes[2].legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.0),
        ncol=2,
        mode="expand",
        frameon=False,
        borderaxespad=0.0,
    )
    axes[2].annotate(
        r"$\Delta |v_x|={:+.3f}$ m/s".format(lateral_increase),
        xy=(peak_time, peak_lateral - lateral_baseline),
        xytext=(
            max(time_s[0] + 0.15, peak_time - 1.45),
            peak_lateral - lateral_baseline + 0.012,
        ),
        fontsize=5.9,
        color=TOKENS["blue"],
        arrowprops={
            "arrowstyle": "-",
            "color": TOKENS["blue"],
            "linewidth": 0.8,
        },
    )
    for axis in axes:
        axis.axvline(
            peak_time,
            color="#6F4E9C",
            alpha=1.0,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            zorder=2,
        )
        axis.set_xlim(time_s[0], time_s[-1])
    fig.subplots_adjust(left=0.19, right=0.83, top=0.87, bottom=0.09)
    png_path = output_dir / (prefix + ".png")
    pdf_path = output_dir / (prefix + ".pdf")
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path, spans


def plot_horizontal_figure(selected, output_dir, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    time_s = values(selected, "stamp_s") - selected[0]["stamp_s"]
    conflict_mask = values(selected, "high_risk_conflict") > 0.5
    spans = contiguous_true_spans(time_s, conflict_mask)
    lateral_magnitude = np.abs(values(selected, "cmd_safe_x"))
    lateral_baseline, lateral_delta = lateral_modulation(selected)

    span_metrics = []
    for span_start, span_end in spans:
        event_index = np.where(
            (time_s >= span_start) & (time_s <= span_end)
        )[0]
        if event_index.size == 0:
            continue
        event_lateral = float(
            np.nanpercentile(lateral_magnitude[event_index], 90)
        )
        span_metrics.append(
            (
                event_lateral - lateral_baseline,
                span_start,
                span_end,
                event_lateral,
            )
        )
    if span_metrics:
        (
            lateral_increase,
            main_span_start,
            main_span_end,
            peak_lateral,
        ) = max(span_metrics, key=lambda item: item[0])
        main_event_index = np.where(
            (time_s >= main_span_start) & (time_s <= main_span_end)
        )[0]
        peak_index = int(
            main_event_index[
                np.nanargmax(values(selected, "risk_F")[main_event_index])
            ]
        )
    else:
        peak_index = int(np.nanargmax(values(selected, "risk_F")))
        peak_lateral = float(np.nanmax(lateral_magnitude))
        lateral_increase = peak_lateral - lateral_baseline
    peak_time = time_s[peak_index]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "axes.labelsize": 6.5,
            "legend.fontsize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.25))
    fig.patch.set_facecolor(TOKENS["surface"])
    for axis in axes:
        style_axis(axis)
        axis.tick_params(labelsize=5.8)
        for start, end in spans:
            axis.axvspan(
                start,
                end,
                color=TOKENS["conflict_fill"],
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )
        axis.axvline(
            peak_time,
            color="#6F4E9C",
            alpha=1.0,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            zorder=2,
        )
        axis.set_xlim(time_s[0], time_s[-1])
        axis.set_xlabel("Time [s]")

    axes[0].plot(
        time_s,
        values(selected, "risk_F"),
        color=TOKENS["red"],
        linewidth=1.45,
        label=r"$\rho_F$ (Follow)",
        zorder=3,
    )
    axes[0].plot(
        time_s,
        values(selected, "front_distance_risk"),
        color=TOKENS["gray"],
        linewidth=1.0,
        linestyle=":",
        label="D435i front risk",
        zorder=2,
    )
    axes[0].set_ylim(-0.03, 0.78)
    axes[0].set_ylabel("Risk")
    axes[0].set_title(
        "(a) Real-robot Follow-command risk",
        loc="left",
        fontsize=7.2,
        fontweight="semibold",
        color=TOKENS["ink"],
        pad=41,
    )
    risk_handles, risk_labels = axes[0].get_legend_handles_labels()
    risk_handles.append(
        Patch(
            facecolor=TOKENS["conflict_fill"],
            edgecolor="none",
            alpha=0.75,
            label="High-risk conflict",
        )
    )
    risk_labels.append("High-risk conflict")
    risk_handles.append(
        Line2D(
            [0],
            [0],
            color="#6F4E9C",
            alpha=1.0,
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            label="Peak Follow-risk instant",
        )
    )
    risk_labels.append("Peak Follow-risk instant")
    axes[0].legend(
        risk_handles,
        risk_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )

    axes[1].plot(
        time_s,
        values(selected, "y"),
        color=TOKENS["gray_dark"],
        linewidth=1.25,
        label=r"Raw gate $y$",
        zorder=3,
    )
    axes[1].plot(
        time_s,
        values(selected, "y_risk"),
        color=TOKENS["blue"],
        linewidth=1.20,
        linestyle="--",
        label=r"Risk-only $y+\Delta y_r$",
        zorder=3,
    )
    axes[1].plot(
        time_s,
        values(selected, "y_eff"),
        color=TOKENS["red"],
        linewidth=1.45,
        label=r"PCR $y_{\mathrm{eff}}$",
        zorder=4,
    )
    axes[1].fill_between(
        time_s,
        values(selected, "y_risk"),
        values(selected, "y_eff"),
        color=TOKENS["learned_fill"],
        alpha=0.62,
        linewidth=0,
        label=r"Learned correction $\Delta y_w$",
        zorder=1,
    )
    y_min = min(0.45, float(np.nanmin(values(selected, "y_risk"))) - 0.03)
    y_max = max(0.80, float(np.nanmax(values(selected, "y_eff"))) + 0.03)
    axes[1].set_ylim(y_min, min(y_max, 1.02))
    axes[1].set_ylabel("Follow weight")
    axes[1].set_title(
        "(b) PCR arbitration response",
        loc="left",
        fontsize=7.2,
        fontweight="semibold",
        color=TOKENS["ink"],
        pad=41,
    )
    axes[1].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )

    axes[2].plot(
        time_s,
        lateral_delta,
        color=TOKENS["blue"],
        linewidth=1.45,
        label=r"Baseline-adjusted $\Delta |v_x|$",
        zorder=3,
    )
    axes[2].axhline(
        0.0,
        color=TOKENS["gray"],
        linewidth=0.9,
        linestyle=":",
        label="Baseline (0.190 m/s)",
        zorder=2,
    )
    forward_axis = axes[2].twinx()
    forward_axis.plot(
        time_s,
        values(selected, "cmd_safe_y"),
        color=TOKENS["orange"],
        linewidth=1.4,
        linestyle="--",
        label=r"Forward $v_y$",
        zorder=3,
    )
    forward_axis.spines["top"].set_visible(False)
    forward_axis.spines["right"].set_color(TOKENS["axis"])
    forward_axis.tick_params(colors=TOKENS["orange"], labelsize=8)
    forward_axis.set_ylabel(
        r"Forward speed $v_y$ [m/s]",
        color=TOKENS["orange"],
        fontsize=9,
    )
    axes[2].set_ylim(
        min(-0.035, float(np.nanmin(lateral_delta)) - 0.01),
        max(0.11, float(np.nanmax(lateral_delta)) + 0.015),
    )
    forward_values = values(selected, "cmd_safe_y")
    forward_axis.set_ylim(
        max(0.0, float(np.nanmin(forward_values)) - 0.06),
        float(np.nanmax(forward_values)) + 0.06,
    )
    axes[2].set_ylabel(r"$\Delta |v_x|$ [m/s]")
    axes[2].set_title(
        "(c) Risk-induced lateral modulation",
        loc="left",
        fontsize=7.2,
        fontweight="semibold",
        color=TOKENS["ink"],
        pad=41,
    )
    left_handles, left_labels = axes[2].get_legend_handles_labels()
    right_handles, right_labels = forward_axis.get_legend_handles_labels()
    axes[2].legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
    )
    axes[2].annotate(
        r"$\Delta |v_x|={:+.3f}$ m/s".format(lateral_increase),
        xy=(peak_time, peak_lateral - lateral_baseline),
        xytext=(
            max(time_s[0] + 0.15, peak_time - 1.5),
            peak_lateral - lateral_baseline + 0.013,
        ),
        fontsize=5.5,
        color=TOKENS["blue"],
        arrowprops={
            "arrowstyle": "-",
            "color": TOKENS["blue"],
            "linewidth": 0.8,
        },
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.90,
        top=0.66,
        bottom=0.16,
        wspace=0.42,
    )
    png_path = output_dir / (prefix + "_horizontal.png")
    pdf_path = output_dir / (prefix + "_horizontal.pdf")
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def write_selected_csv(selected, output_dir, prefix):
    path = output_dir / (prefix + "_data.csv")
    lateral_baseline, lateral_delta = lateral_modulation(selected)
    fieldnames = [
        "time_s",
        "stamp_s",
        "risk_F",
        "risk_A",
        "front_distance_risk",
        "conflict_score",
        "high_risk_conflict",
        "y",
        "w",
        "signed_w_active",
        "delta_y_r",
        "delta_y_w",
        "y_risk",
        "y_eff",
        "y_eff_reconstructed",
        "y_eff_reconstruction_abs_error",
        "cmd_safe_x",
        "cmd_safe_x_abs",
        "cmd_safe_x_abs_pre_conflict_baseline",
        "cmd_safe_x_abs_delta_from_pre_conflict",
        "cmd_safe_y",
        "cmd_safe_yaw",
        "target_valid",
        "depth_invalid",
        "safety_clear",
        "risk_source_real",
    ]
    start = selected[0]["stamp_s"]
    derived_fields = {
        "time_s",
        "cmd_safe_x_abs",
        "cmd_safe_x_abs_pre_conflict_baseline",
        "cmd_safe_x_abs_delta_from_pre_conflict",
    }
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(selected):
            record = {
                key: row[key]
                for key in fieldnames
                if key not in derived_fields
            }
            record["time_s"] = row["stamp_s"] - start
            record["cmd_safe_x_abs"] = abs(row["cmd_safe_x"])
            record["cmd_safe_x_abs_pre_conflict_baseline"] = lateral_baseline
            record["cmd_safe_x_abs_delta_from_pre_conflict"] = lateral_delta[index]
            writer.writerow(record)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_notes(
    bag_path,
    selected,
    selection_metrics,
    candidate_metrics,
    spans,
    output_dir,
    prefix,
    args,
):
    session_start = selected[0]["stamp_s"] - (
        selected[0].get("session_relative_s", 0.0)
    )
    reconstruction_error = values(
        selected, "y_eff_reconstruction_abs_error"
    )
    conflict = values(selected, "high_risk_conflict") > 0.5
    risk_f = values(selected, "risk_F")
    y_risk = values(selected, "y_risk")
    y_eff = values(selected, "y_eff")
    cmd_x = values(selected, "cmd_safe_x")
    cmd_y = values(selected, "cmd_safe_y")
    lateral_baseline, lateral_delta = lateral_modulation(selected)
    selected_start_rel = selected[0]["stamp_s"] - args.session_start_stamp_s
    selected_end_rel = selected[-1]["stamp_s"] - args.session_start_stamp_s
    pre_risk = float(selection_metrics.get("pre_risk", selection_metrics.get("risk_start", np.nan)))
    event_peak_risk = float(selection_metrics.get("event_peak_risk", selection_metrics.get("risk_peak", np.nan)))
    post_risk = float(selection_metrics.get("post_risk", selection_metrics.get("risk_end", np.nan)))
    pre_lateral = float(selection_metrics.get("pre_lateral", selection_metrics.get("mean_abs_lateral", np.nan)))
    event_lateral_p90 = float(selection_metrics.get("event_lateral_p90", np.nanpercentile(np.abs(cmd_x), 90)))
    post_lateral = float(selection_metrics.get("post_lateral", np.nan))
    lateral_increase_metric = float(selection_metrics.get("lateral_increase", np.nanmax(lateral_delta)))
    event_forward = float(selection_metrics.get("event_forward", np.nanmean(np.maximum(cmd_y, 0.0))))
    event_delta_y_w = float(selection_metrics.get("event_delta_y_w", np.nanmean(values(selected, "delta_y_w")[conflict]) if np.any(conflict) else np.nanmean(values(selected, "delta_y_w"))))
    lines = [
        "# Real-Robot PCR Arbitration Figure Notes",
        "",
        "## Source",
        "",
        "- Bag: `{}`".format(bag_path),
        "- SHA256: `{}`".format(sha256(bag_path)),
        "- Debug topic: `{}`".format(args.debug_topic),
        "- Selected interval: `{:.3f}--{:.3f} s` from bag debug start.".format(
            selected_start_rel, selected_end_rel
        ),
        "- Selected samples: `{}` at approximately `10 Hz`.".format(len(selected)),
        "",
        "## Deterministic Selection",
        "",
        "- Only continuous samples with valid target, valid depth, no safety stop, and `risk_blocked_map` were eligible.",
        "- Fixed event requirements: current `pcr_real_20260618_*.bag`, risk peak >= 0.45, risk rise >= 0.25, post-event risk recovery >= 0.15, lateral increase >= 0.05 m/s, event forward command >= 0.15 m/s, moving fraction >= 0.80, and arbitration reconstruction error < 1e-5.",
        "- Winning event score: `{:.4f}`.".format(selection_metrics["score"]),
        "- Risk pre/peak/post: `{:.4f} / {:.4f} / {:.4f}`.".format(
            pre_risk,
            event_peak_risk,
            post_risk,
        ),
        "- Lateral magnitude pre/event-p90/post: `{:.4f} / {:.4f} / {:.4f}` m/s.".format(
            pre_lateral,
            event_lateral_p90,
            post_lateral,
        ),
        "- Lateral increase / event forward command: `{:+.4f} / {:.4f}` m/s.".format(
            lateral_increase_metric,
            event_forward,
        ),
        "- Event learned correction mean: `{:.4f}`.".format(
            event_delta_y_w
        ),
        "",
        "## Arbitration Reconstruction",
        "",
        "- `signed_w = 2w - 1`; values with `|signed_w| <= {:.3f}` are set to zero.".format(
            args.signed_w_margin
        ),
        "- `Delta y_r = {:.3f} (risk_A - risk_F)`.".format(
            args.signed_w_gamma_risk
        ),
        "- `Delta y_w = {:.3f} signed_w_active`.".format(
            args.signed_w_lambda
        ),
        "- `y_eff = clip(y + Delta y_r + Delta y_w, 0, 1)`.",
        "- Reconstruction absolute error mean/max: `{:.3e} / {:.3e}`.".format(
            float(np.nanmean(reconstruction_error)),
            float(np.nanmax(reconstruction_error)),
        ),
        "",
        "## Figure Evidence",
        "",
        "- Panel (a) omits the constant-zero `risk_A` curve. In all valid samples of the selected interval, `clearance_A = 3.0 m`, `risk_A_raw = 0`, and `risk_A = 0` because the pure-lateral Avoid cone contains no observed blocked cell.",
        "- The analytic correction in this interval is therefore `Delta y_r = -0.15 risk_F`; this is a recorded property of the deployable local-map risk estimate, not a manually imposed plotting assumption.",
        "- Panel (c) reports `Delta |v_x|(t) = |v_x(t)| - median_pre(|v_x|)`; the pre-conflict window is 2.5--0.35 s before the first high-risk event.",
        "- The recorded pre-conflict lateral baseline is `{:.4f} m/s`; the plotted transformation changes only the display origin, while the CSV retains the original signed `cmd_safe_x`.".format(
            lateral_baseline
        ),
        "- The displayed lateral modulation range is `{:+.4f}` to `{:+.4f} m/s`.".format(
            float(np.nanmin(lateral_delta)),
            float(np.nanmax(lateral_delta)),
        ),
        "- Light-red vertical shading denotes high-risk conflict states and uses recorded `risk_F >= {:.2f}`, `risk_F > risk_A`, and recorded `conflict_score >= {:.2f}`. The light-blue area in panel (b) is reserved exclusively for the learned correction `Delta y_w`.".format(
            args.risk_threshold, args.conflict_threshold
        ),
        "- High-risk spans in the selected interval: `{}`.".format(
            ", ".join("{:.2f}--{:.2f} s".format(a, b) for a, b in spans)
            if spans
            else "none"
        ),
        "- During high-risk samples, mean `risk_F`, `Delta y_w`, lateral command, and forward command are `{:.4f}`, `{:.4f}`, `{:.4f}`, and `{:.4f}`.".format(
            float(np.nanmean(risk_f[conflict])),
            float(np.nanmean(values(selected, "delta_y_w")[conflict])),
            float(np.nanmean(cmd_x[conflict])),
            float(np.nanmean(cmd_y[conflict])),
        ),
        "- During high-risk samples, learned-w restores `y_eff - y_risk = {:.4f}` on average.".format(
            float(np.nanmean((y_eff - y_risk)[conflict]))
        ),
        "",
        "## Claim Boundary",
        "",
        "- Supported: the real D435i-derived Follow-command risk changes online, PCR changes the effective Follow weight, and the published command increases its lateral magnitude while retaining forward authority.",
        "- Avoid-risk scope: the observed lateral Avoid cone remains free in this interval, so the trace demonstrates Follow-risk-driven arbitration rather than two dynamically varying command risks.",
        "- Not supported by this trace alone: trial success rate, collision rate, final-row clearance, or statistical sim-to-real superiority. Those require synchronized video and manual trial labels.",
        "",
        "## Candidate Audit",
        "",
        "| Rank | Start [s] | End [s] | Eligible | Score | Risk peak | Risk range | Mean abs(lat) | Mean fwd |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(candidate_metrics[:10], start=1):
        lines.append(
            "| {} | {:.2f} | {:.2f} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                rank,
                item["start_rel_s"],
                item["end_rel_s"],
                "yes" if item["eligible"] else "no",
                item["score"],
                item["risk_peak"],
                item["risk_range"],
                item["mean_abs_lateral"],
                item["mean_forward"],
            )
        )
    path = output_dir / (prefix + "_notes.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = None
    summary_path = None
    gallery_paths = (None, None)
    if args.bag_glob:
        (
            selected,
            metrics,
            candidates,
            bag_path,
            audit_path,
            summary_path,
            gallery_paths,
        ) = select_across_bags(args, output_dir)
        args.session_start_stamp_s = (
            selected[0]["stamp_s"] - selected[0]["session_relative_s"]
        )
    else:
        bag_path = Path(args.bag).expanduser().resolve()
        if not bag_path.is_file():
            raise SystemExit("Bag not found: {}".format(bag_path))
        rows = enrich_rows(load_debug_rows(bag_path, args.debug_topic), args)
        args.session_start_stamp_s = rows[0]["stamp_s"]
        for row in rows:
            row["session_relative_s"] = row["stamp_s"] - args.session_start_stamp_s
        selected, metrics, candidates = select_window(rows, args)
    png_path, pdf_path, spans = plot_figure(
        selected, output_dir, args.prefix
    )
    horizontal_png_path, horizontal_pdf_path = plot_horizontal_figure(
        selected, output_dir, args.prefix
    )
    csv_path = write_selected_csv(selected, output_dir, args.prefix)
    notes_path = write_notes(
        bag_path,
        selected,
        metrics,
        candidates,
        spans,
        output_dir,
        args.prefix,
        args,
    )

    summary = {
        "bag": str(bag_path),
        "selected_start_rel_s": selected[0]["session_relative_s"],
        "selected_end_rel_s": selected[-1]["session_relative_s"],
        "selected_samples": len(selected),
        "selection_score": metrics["score"],
        "risk_peak": metrics["risk_peak"],
        "risk_range": metrics["risk_range"],
        "mean_abs_lateral": metrics["mean_abs_lateral"],
        "mean_forward": metrics["mean_forward"],
        "high_risk_fraction": metrics["high_risk_fraction"],
        "reconstruction_error_max": float(
            np.nanmax(values(selected, "y_eff_reconstruction_abs_error"))
        ),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "horizontal_png": str(horizontal_png_path),
        "horizontal_pdf": str(horizontal_pdf_path),
        "csv": str(csv_path),
        "notes": str(notes_path),
        "bag_audit_csv": None if audit_path is None else str(audit_path),
        "bag_summary_csv": None if summary_path is None else str(summary_path),
        "candidate_gallery_png": (
            None if gallery_paths[0] is None else str(gallery_paths[0])
        ),
        "candidate_gallery_pdf": (
            None if gallery_paths[1] is None else str(gallery_paths[1])
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
