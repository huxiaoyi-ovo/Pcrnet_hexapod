#!/usr/bin/env python3
"""Export PCR debug curves from a recorded real-robot session."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


SCALAR_FIELDS = [
    "risk_F",
    "risk_A",
    "risk_F_raw",
    "risk_A_raw",
    "y",
    "y_eff",
    "w",
    "front_distance_risk",
    "risk_memory",
    "conflict_score",
    "actor_difficulty",
    "clearance_F",
    "clearance_A",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-ready PCR curves and reference the session viewer MP4."
    )
    parser.add_argument("bag", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--debug_topic", type=str, default="/pcr_realplay/debug")
    return parser.parse_args()


def vector_value(payload: Dict, key: str, index: int) -> float:
    value = payload.get(key, [])
    if isinstance(value, dict):
        component = ("x_vec", "y_vec", "w_twist")[index]
        try:
            return float(value.get(component, float("nan")))
        except (TypeError, ValueError):
            return float("nan")
    if not isinstance(value, (list, tuple)) or len(value) <= index:
        return float("nan")
    try:
        return float(value[index])
    except (TypeError, ValueError):
        return float("nan")


def scalar_value(payload: Dict, key: str) -> float:
    try:
        return float(payload.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def debug_row(payload: Dict, stamp_s: float) -> Dict[str, float]:
    row = {"stamp_s": float(stamp_s)}
    for key in SCALAR_FIELDS:
        row[key] = scalar_value(payload, key)
    for prefix, key in (
        ("cmd_f", "cmd_f"),
        ("cmd_a", "cmd_a"),
        ("cmd_policy", "cmd_policy"),
        ("cmd_safe", "cmd_safe"),
        ("usr_command", "usr_command"),
    ):
        row[f"{prefix}_x"] = vector_value(payload, key, 0)
        row[f"{prefix}_y"] = vector_value(payload, key, 1)
        row[f"{prefix}_yaw"] = vector_value(payload, key, 2)
    for key in ("target_valid", "target_lost", "target_too_close", "row_not_released"):
        row[key] = float(bool(payload.get(key, False)))
    return row


def write_debug_csv(rows: List[Dict[str, float]], output_dir: Path) -> Path:
    path = output_dir / "pcr_debug.csv"
    if not rows:
        path.write_text("")
        return path
    start = rows[0]["stamp_s"]
    fieldnames = ["time_s"] + [key for key in rows[0].keys() if key != "stamp_s"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["time_s"] = float(item.pop("stamp_s")) - start
            writer.writerow(item)
    return path


def plot_curves(rows: List[Dict[str, float]], output_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = np.asarray([row["stamp_s"] - rows[0]["stamp_s"] for row in rows])

    def values(key: str) -> np.ndarray:
        return np.asarray([row.get(key, float("nan")) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_s, values("risk_F"), label="risk_F", linewidth=1.8)
    axes[0].plot(time_s, values("risk_A"), label="risk_A", linewidth=1.8)
    axes[0].plot(
        time_s,
        values("front_distance_risk"),
        label="front distance risk",
        linewidth=1.2,
        alpha=0.8,
    )
    axes[0].set_ylabel("Risk")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3, loc="upper right")

    axes[1].plot(time_s, values("y"), label="y", linewidth=1.4)
    axes[1].plot(time_s, values("y_eff"), label="y_eff", linewidth=1.8)
    axes[1].plot(time_s, values("w"), label="w", linewidth=1.4)
    axes[1].plot(
        time_s,
        values("row_not_released"),
        label="row_not_released",
        linewidth=1.0,
        alpha=0.7,
    )
    axes[1].set_ylabel("Arbitration")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=4, loc="upper right")

    axes[2].plot(time_s, values("cmd_safe_x"), label="safe lateral", linewidth=1.6)
    axes[2].plot(time_s, values("cmd_safe_y"), label="safe forward", linewidth=1.6)
    axes[2].plot(time_s, values("cmd_safe_yaw"), label="safe yaw", linewidth=1.6)
    axes[2].plot(
        time_s,
        values("cmd_a_x"),
        label="avoid lateral",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Command")
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=4, loc="upper right")

    fig.tight_layout()
    png_path = output_dir / "pcr_curves.png"
    pdf_path = output_dir / "pcr_curves.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    try:
        import rosbag
    except ImportError as exc:
        raise SystemExit("ROS1 rosbag Python package is required; source ROS first.") from exc

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"bag not found: {bag_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else bag_path.parent / f"{bag_path.stem}_export"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _topic, msg, stamp in bag.read_messages(topics=[args.debug_topic]):
            try:
                payload = json.loads(msg.data)
            except (AttributeError, TypeError, json.JSONDecodeError):
                continue
            rows.append(debug_row(payload, stamp.to_sec()))

    csv_path = write_debug_csv(rows, output_dir)
    curve_path = plot_curves(rows, output_dir)
    viewer_path = bag_path.parent / "viewer.mp4"
    frame_csv_path = bag_path.parent / "frame_timestamps.csv"
    viewer_frame_count = 0
    if frame_csv_path.is_file():
        with open(frame_csv_path, newline="") as handle:
            viewer_frame_count = max(sum(1 for _line in handle) - 1, 0)
    summary = {
        "bag": str(bag_path),
        "debug_samples": len(rows),
        "viewer_frames": viewer_frame_count,
        "debug_csv": str(csv_path),
        "curve_png": None if curve_path is None else str(curve_path),
        "viewer_video": str(viewer_path) if viewer_path.is_file() else None,
        "viewer_timestamps": (
            str(frame_csv_path) if frame_csv_path.is_file() else None
        ),
    }
    with open(output_dir / "export_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
