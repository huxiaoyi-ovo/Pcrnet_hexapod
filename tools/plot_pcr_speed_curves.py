#!/usr/bin/env python3
"""Plot PCR speed-conflict and speed-performance curves."""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_ORDER = ["yonly", "geomw", "learnedw"]
METHOD_LABEL = {
    "yonly": "Y-only",
    "geomw": "Geom-w",
    "learnedw": "Learned-w",
}
METHOD_COLOR = {
    "yonly": "#6b7280",
    "geomw": "#2563eb",
    "learnedw": "#dc2626",
}


def _safe_float(value, default=float("nan")):
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _method_from_row(row):
    method = str(row.get("Method", "")).strip().lower()
    if method:
        return method
    w_mode = str(row.get("w_mode", "")).strip().lower()
    source = str(row.get("source", "")).lower()
    if "learnedw" in source or w_mode in ("learned", "learnedw2"):
        return "learnedw"
    if "geomw" in source or w_mode == "geom":
        return "geomw"
    if "yonly" in source or w_mode == "none":
        return "yonly"
    return "unknown"


def _speed_from_row(row):
    speed = str(row.get("Speed", "")).strip()
    if speed:
        return _safe_float(speed)
    source = str(row.get("source", row.get("Source", "")))
    for token in ("s_0.35", "s_0.5", "s_0.50", "s_0.6", "s_0.60"):
        if token in source:
            return _safe_float(token.replace("s_", ""))
    return float("nan")


def _read_performance(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = _method_from_row(row)
            speed = _speed_from_row(row)
            if method not in METHOD_ORDER or not math.isfinite(speed):
                continue
            rows.append(
                {
                    "speed": speed,
                    "method": method,
                    "task_success_mean": _safe_float(row.get("Task Success Mean")),
                    "task_success_std": _safe_float(row.get("Task Success Std"), 0.0),
                    "collision_mean": _safe_float(row.get("Collision Mean")),
                    "collision_std": _safe_float(row.get("Collision Std"), 0.0),
                    "follow_mae_mean": _safe_float(row.get("Follow MAE Mean")),
                    "follow_mae_std": _safe_float(row.get("Follow MAE Std"), 0.0),
                    "unsafe_rate_mean": _safe_float(row.get("Unsafe Rate Mean")),
                    "unsafe_rate_std": _safe_float(row.get("Unsafe Rate Std"), 0.0),
                    "c_avoid_rate_mean": _safe_float(row.get("C_avoid Rate Mean")),
                    "c_avoid_rate_std": _safe_float(row.get("C_avoid Rate Std"), 0.0),
                }
            )
    return rows


def _mean_std(values):
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan"), 0
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0, len(clean)
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(max(var, 0.0)), len(clean)


def _read_forward_risk(paths):
    grouped = defaultdict(list)
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = _method_from_row(row)
                speed = _speed_from_row(row)
                forward_risk = _safe_float(row.get("risk_rollout_f_mean"))
                if method not in METHOD_ORDER or not math.isfinite(speed) or not math.isfinite(forward_risk):
                    continue
                grouped[(speed, method)].append(forward_risk)
    out = {}
    for key, vals in grouped.items():
        mean, std, count = _mean_std(vals)
        out[key] = {
            "forward_risk_mean": mean,
            "forward_risk_std": std,
            "forward_risk_n": count,
        }
    return out


def _write_plot_data(path, perf_rows, risk_rows):
    fields = [
        "Speed",
        "Method",
        "Success Rate",
        "Success Rate Std",
        "Collision Rate",
        "Collision Rate Std",
        "Tracking MAE",
        "Tracking MAE Std",
        "Conflict Rate",
        "Conflict Rate Std",
        "Avoidance Choice Rate",
        "Avoidance Choice Rate Std",
        "Forward Risk",
        "Forward Risk Std",
        "Forward Risk N",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(perf_rows, key=lambda r: (r["speed"], METHOD_ORDER.index(r["method"]))):
            risk = risk_rows.get((row["speed"], row["method"]), {})
            out = {
                "Speed": f"{row['speed']:.2f}",
                "Method": METHOD_LABEL[row["method"]],
                "Success Rate": row.get("task_success_mean", ""),
                "Success Rate Std": row.get("task_success_std", ""),
                "Collision Rate": row.get("collision_mean", ""),
                "Collision Rate Std": row.get("collision_std", ""),
                "Tracking MAE": row.get("follow_mae_mean", ""),
                "Tracking MAE Std": row.get("follow_mae_std", ""),
                "Conflict Rate": row.get("unsafe_rate_mean", ""),
                "Conflict Rate Std": row.get("unsafe_rate_std", ""),
                "Avoidance Choice Rate": row.get("c_avoid_rate_mean", ""),
                "Avoidance Choice Rate Std": row.get("c_avoid_rate_std", ""),
                "Forward Risk": risk.get("forward_risk_mean", ""),
                "Forward Risk Std": risk.get("forward_risk_std", ""),
                "Forward Risk N": risk.get("forward_risk_n", ""),
            }
            writer.writerow(out)


def _series(rows, method, key_mean, key_std):
    selected = sorted([r for r in rows if r["method"] == method], key=lambda r: r["speed"])
    xs = [r["speed"] for r in selected]
    ys = [r[key_mean] for r in selected]
    es = [r.get(key_std, 0.0) for r in selected]
    return xs, ys, es


def _draw_line(ax, xs, ys, es, method, label=None):
    pairs = [(x, y, e) for x, y, e in zip(xs, ys, es) if math.isfinite(y)]
    if not pairs:
        return
    xs, ys, es = zip(*pairs)
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        marker="o",
        linewidth=2.0,
        capsize=3.0,
        color=METHOD_COLOR[method],
        label=label or METHOD_LABEL[method],
    )


def plot_conflict(perf_rows, risk_rows, output):
    risk_plot_rows = []
    for row in perf_rows:
        risk = risk_rows.get((row["speed"], row["method"]), {})
        if risk:
            merged = dict(row)
            merged.update(risk)
            risk_plot_rows.append(merged)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True)
    for method in METHOD_ORDER:
        xs, ys, es = _series(perf_rows, method, "unsafe_rate_mean", "unsafe_rate_std")
        _draw_line(axes[0], xs, ys, es, method)
        xs, ys, es = _series(risk_plot_rows, method, "forward_risk_mean", "forward_risk_std")
        _draw_line(axes[1], xs, ys, es, method)

    axes[0].set_title("Conflict Rate")
    axes[0].set_ylabel("Rate")
    axes[1].set_title("Forward Risk")
    axes[1].set_ylabel("Risk")
    for ax in axes:
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_xticks([0.35, 0.50, 0.60])
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0.0)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_performance(perf_rows, output):
    specs = [
        ("Success Rate", "task_success_mean", "task_success_std", "Rate", (0.0, 1.05)),
        ("Collision Rate", "collision_mean", "collision_std", "Rate", (0.0, None)),
        ("Tracking MAE", "follow_mae_mean", "follow_mae_std", "Distance Error (m)", (0.0, None)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    for ax, (title, mean_key, std_key, ylabel, ylim) in zip(axes, specs):
        for method in METHOD_ORDER:
            xs, ys, es = _series(perf_rows, method, mean_key, std_key)
            _draw_line(ax, xs, ys, es, method)
        ax.set_title(title)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.35, 0.50, 0.60])
        ax.grid(True, alpha=0.25)
        if ylim[1] is None:
            ax.set_ylim(bottom=ylim[0])
        else:
            ax.set_ylim(*ylim)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PCR speed curves")
    parser.add_argument(
        "--performance_csv",
        default="agents/eval_data_seed23/pcr_main_table/pcr_main_table_3seed_aggregate_20260529_manual_seed1.csv",
    )
    parser.add_argument(
        "--risk_csv",
        action="append",
        default=None,
    )
    parser.add_argument("--output_dir", default="agents/eval_data_seed23/pcr_main_table")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.risk_csv is None:
        args.risk_csv = [
            "agents/eval_data_seed23/pcr_main_table/pcr_main_table_all_metrics_20260529_032025.csv",
            "agents/eval_data/pcr_main_table/pcr_main_table_all_metrics_with_rollout_20260529.csv",
        ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_rows = _read_performance(args.performance_csv)
    risk_rows = _read_forward_risk(args.risk_csv)

    data_path = out_dir / "speed_curve_plot_data.csv"
    conflict_path = out_dir / "fig_speed_conflict_curve.png"
    performance_path = out_dir / "fig_speed_performance_curve.png"

    _write_plot_data(data_path, perf_rows, risk_rows)
    plot_conflict(perf_rows, risk_rows, conflict_path)
    plot_performance(perf_rows, performance_path)

    print(f"Plot data: {data_path}")
    print(f"Speed-conflict figure: {conflict_path}")
    print(f"Speed-performance figure: {performance_path}")


if __name__ == "__main__":
    main()
