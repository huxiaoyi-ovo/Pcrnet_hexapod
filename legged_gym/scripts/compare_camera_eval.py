#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare camera_eval_metrics.csv results across runs.

Usage examples:
  python compare_camera_eval.py logs/hex_ground/camera_eval
  python compare_camera_eval.py logs/hex_ground/camera_eval --per-cmd
  python compare_camera_eval.py /path/to/run/camera_eval_metrics.csv
  python compare_camera_eval.py logs/hex_ground/camera_eval --rank-by diff_mean_mean
  python compare_camera_eval.py logs/hex_ground/camera_eval --out camera_eval_summary.csv
"""

import argparse
import csv
import math
import os
import statistics
from typing import Dict, List, Tuple


def _is_number(val: str) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _compute_stats(vals: List[float]) -> Dict[str, float]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "std": float("nan"),
        }
    vals_sorted = sorted(vals)
    return {
        "mean": _safe_mean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "p95": _percentile(vals_sorted, 95.0),
        "std": statistics.pstdev(vals_sorted) if len(vals_sorted) > 1 else 0.0,
    }


def _collect_csvs(paths: List[str]) -> List[str]:
    csvs: List[str] = []
    for p in paths:
        if os.path.isfile(p):
            if p.endswith(".csv"):
                csvs.append(p)
            continue
        if os.path.isdir(p):
            direct = os.path.join(p, "camera_eval_metrics.csv")
            if os.path.isfile(direct):
                csvs.append(direct)
                continue
            for root, _, files in os.walk(p):
                if "camera_eval_metrics.csv" in files:
                    csvs.append(os.path.join(root, "camera_eval_metrics.csv"))
        else:
            raise FileNotFoundError(f"Path not found: {p}")
    uniq = sorted(set(csvs))
    return uniq


def _read_metrics(csv_path: str) -> Tuple[Dict[str, List[float]], List[str]]:
    metrics = {
        "diff_mean": [],
        "grad_diff_mean": [],
        "diff_rms": [],
        "depth_std": [],
        "cmd_speed": [],
        "base_speed": [],
        "track_err_lin": [],
        "track_err_yaw": [],
        "speed_ratio": [],
        "energy": [],
        "base_ang_vel_xy_rms": [],
        "base_ang_acc_xy_rms": [],
        "roll": [],
        "pitch": [],
        "depth_invalid_ratio": [],
        "depth_clip_ratio": [],
        "depth_grad_mean": [],
        "feature_count": [],
        "edge_density": [],
    }
    cmds: List[str] = []
    with open(csv_path, "r", encoding="ascii", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = row.get("cmd", "")
            cmds.append(cmd)
            for key in metrics:
                val = row.get(key, "")
                if val is None or not _is_number(val):
                    continue
                fval = float(val)
                if math.isfinite(fval):
                    metrics[key].append(fval)
    return metrics, cmds


def _label_from_path(csv_path: str) -> str:
    parent = os.path.basename(os.path.dirname(csv_path))
    return parent or os.path.basename(csv_path)


def _format_float(val: float, width: int = 10) -> str:
    if val is None or not math.isfinite(val):
        return "nan".rjust(width)
    return f"{val:.6f}".rjust(width)


def _normalize(values: List[float]) -> List[float]:
    finite_vals = [v for v in values if math.isfinite(v)]
    if not finite_vals:
        return [float("nan") for _ in values]
    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if math.isclose(vmax, vmin):
        return [0.0 if math.isfinite(v) else float("nan") for v in values]
    normed = []
    for v in values:
        if not math.isfinite(v):
            normed.append(float("nan"))
        else:
            normed.append((v - vmin) / (vmax - vmin))
    return normed


def _per_cmd_stats(csv_path: str) -> Dict[str, Dict[str, float]]:
    per_cmd: Dict[str, List[float]] = {}
    per_cmd_g: Dict[str, List[float]] = {}
    per_cmd_r: Dict[str, List[float]] = {}
    per_cmd_s: Dict[str, List[float]] = {}
    per_cmd_t: Dict[str, List[float]] = {}
    per_cmd_ty: Dict[str, List[float]] = {}
    per_cmd_e: Dict[str, List[float]] = {}
    with open(csv_path, "r", encoding="ascii", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmd = row.get("cmd", "")
            per_cmd.setdefault(cmd, [])
            per_cmd_g.setdefault(cmd, [])
            per_cmd_r.setdefault(cmd, [])
            per_cmd_s.setdefault(cmd, [])
            per_cmd_t.setdefault(cmd, [])
            per_cmd_ty.setdefault(cmd, [])
            per_cmd_e.setdefault(cmd, [])
            for key, bucket in (
                ("diff_mean", per_cmd),
                ("grad_diff_mean", per_cmd_g),
                ("diff_rms", per_cmd_r),
                ("depth_std", per_cmd_s),
                ("track_err_lin", per_cmd_t),
                ("track_err_yaw", per_cmd_ty),
                ("energy", per_cmd_e),
            ):
                val = row.get(key, "")
                if val is None or not _is_number(val):
                    continue
                fval = float(val)
                if math.isfinite(fval):
                    bucket[cmd].append(fval)
    out: Dict[str, Dict[str, float]] = {}
    for cmd in per_cmd:
        out[cmd] = {
            "count": float(len(per_cmd[cmd])),
            "diff_mean": _safe_mean(per_cmd[cmd]),
            "grad_diff_mean": _safe_mean(per_cmd_g[cmd]),
            "diff_rms": _safe_mean(per_cmd_r[cmd]),
            "depth_std": _safe_mean(per_cmd_s[cmd]),
            "track_err_lin": _safe_mean(per_cmd_t[cmd]),
            "track_err_yaw": _safe_mean(per_cmd_ty[cmd]),
            "energy": _safe_mean(per_cmd_e[cmd]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare camera_eval_metrics.csv across runs",
        epilog=(
            "Examples:\n"
            "  compare_camera_eval.py logs/hex_ground/camera_eval\n"
            "  compare_camera_eval.py logs/hex_ground/camera_eval --per-cmd\n"
            "  compare_camera_eval.py /path/to/run/camera_eval_metrics.csv\n"
            "  compare_camera_eval.py logs/hex_ground/camera_eval --rank-by diff_mean_mean\n"
            "  compare_camera_eval.py logs/hex_ground/camera_eval --out camera_eval_summary.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="CSV files or directories to scan")
    parser.add_argument("--per-cmd", action="store_true", help="Print per-command statistics")
    parser.add_argument(
        "--rank-by",
        default="composite",
        choices=[
            "composite",
            "diff_mean_mean",
            "grad_diff_mean_mean",
            "diff_rms_p95",
            "tracking_mean",
            "energy_mean",
        ],
        help="Ranking key (lower is better)",
    )
    parser.add_argument(
        "--weights",
        default="0.5,0.3,0.2",
        help="Composite weights: diff_mean_mean,grad_diff_mean_mean,diff_rms_p95",
    )
    parser.add_argument(
        "--tracking-weight",
        type=float,
        default=0.5,
        help="Composite tracking weight (lower tracking error is better)",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=0.2,
        help="Composite energy weight (lower energy is better)",
    )
    parser.add_argument(
        "--tracking-yaw-scale",
        type=float,
        default=0.5,
        help="Scale applied to track_err_yaw when combining tracking error",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    csvs = _collect_csvs(args.paths)
    if not csvs:
        print("No camera_eval_metrics.csv files found.")
        return 1

    try:
        w_parts = [float(x.strip()) for x in args.weights.split(",")]
        if len(w_parts) != 3:
            raise ValueError
    except Exception:
        print("Invalid --weights. Expected three comma-separated floats, e.g. 0.5,0.3,0.2")
        return 1

    runs = []
    for csv_path in csvs:
        metrics, _ = _read_metrics(csv_path)
        stat_keys = [
            "diff_mean",
            "grad_diff_mean",
            "diff_rms",
            "depth_std",
            "cmd_speed",
            "base_speed",
            "track_err_lin",
            "track_err_yaw",
            "speed_ratio",
            "energy",
            "base_ang_vel_xy_rms",
            "base_ang_acc_xy_rms",
            "roll",
            "pitch",
            "depth_invalid_ratio",
            "depth_clip_ratio",
            "depth_grad_mean",
            "feature_count",
            "edge_density",
        ]
        stats = {key: _compute_stats(metrics[key]) for key in stat_keys}
        runs.append(
            {
                "name": _label_from_path(csv_path),
                "path": csv_path,
                "count": len(metrics["diff_mean"]),
                "stats": stats,
            }
        )

    diff_mean_mean = [r["stats"]["diff_mean"]["mean"] for r in runs]
    grad_mean_mean = [r["stats"]["grad_diff_mean"]["mean"] for r in runs]
    diff_rms_p95 = [r["stats"]["diff_rms"]["p95"] for r in runs]
    tracking_mean = []
    energy_mean = [r["stats"]["energy"]["mean"] for r in runs]

    for r in runs:
        t_lin = r["stats"]["track_err_lin"]["mean"]
        t_yaw = r["stats"]["track_err_yaw"]["mean"]
        if math.isfinite(t_lin) and math.isfinite(t_yaw):
            tracking_mean.append(t_lin + args.tracking_yaw_scale * t_yaw)
        else:
            tracking_mean.append(float("nan"))

    def _normalize_with_missing(values, weight):
        normed = _normalize(values)
        has_finite = any(math.isfinite(v) for v in values)
        if not has_finite or weight <= 0.0:
            return normed, False
        return [v if math.isfinite(v) else 1.0 for v in normed], True

    norm_diff_mean = _normalize(diff_mean_mean)
    norm_grad_mean = _normalize(grad_mean_mean)
    norm_diff_rms = _normalize(diff_rms_p95)

    norm_tracking, tracking_enabled = _normalize_with_missing(tracking_mean, args.tracking_weight)
    norm_energy, energy_enabled = _normalize_with_missing(energy_mean, args.energy_weight)
    tracking_weight = args.tracking_weight if tracking_enabled else 0.0
    energy_weight = args.energy_weight if energy_enabled else 0.0
    if args.tracking_weight > 0.0 and not tracking_enabled:
        print("Warning: tracking metrics missing; tracking weight ignored.")
    if args.energy_weight > 0.0 and not energy_enabled:
        print("Warning: energy metrics missing; energy weight ignored.")

    for idx, run in enumerate(runs):
        score = (
            w_parts[0] * norm_diff_mean[idx]
            + w_parts[1] * norm_grad_mean[idx]
            + w_parts[2] * norm_diff_rms[idx]
            + tracking_weight * (norm_tracking[idx] if idx < len(norm_tracking) else 0.0)
            + energy_weight * (norm_energy[idx] if idx < len(norm_energy) else 0.0)
        )
        run["composite"] = score
        run["rank_keys"] = {
            "composite": score,
            "diff_mean_mean": diff_mean_mean[idx],
            "grad_diff_mean_mean": grad_mean_mean[idx],
            "diff_rms_p95": diff_rms_p95[idx],
            "tracking_mean": tracking_mean[idx],
            "energy_mean": energy_mean[idx],
        }

    def _sort_key(val: float) -> float:
        return val if math.isfinite(val) else float("inf")

    runs_sorted = sorted(runs, key=lambda r: _sort_key(r["rank_keys"][args.rank_by]))

    print("\nRanking (lower is better):")
    header = (
        "rank  name                           n  diff_mean_mean  grad_mean_mean  diff_rms_p95  track_err_mean  energy_mean  depth_std_mean  composite"
    )
    print(header)
    for rank, run in enumerate(runs_sorted, start=1):
        stats = run["stats"]
        row = (
            f"{str(rank).rjust(4)}  "
            f"{run['name'][:30].ljust(30)}  "
            f"{str(run['count']).rjust(3)}  "
            f"{_format_float(stats['diff_mean']['mean'])}  "
            f"{_format_float(stats['grad_diff_mean']['mean'])}  "
            f"{_format_float(stats['diff_rms']['p95'])}  "
            f"{_format_float(run['rank_keys']['tracking_mean'])}  "
            f"{_format_float(run['rank_keys']['energy_mean'])}  "
            f"{_format_float(stats['depth_std']['mean'])}  "
            f"{_format_float(run['composite'])}"
        )
        print(row)

    print("\nDetails (mean/median/p95):")
    for run in runs_sorted:
        s = run["stats"]
        print(f"\n{run['name']}  ({run['path']})")
        print(
            "  diff_mean     "
            f"mean={s['diff_mean']['mean']:.6f}  "
            f"median={s['diff_mean']['median']:.6f}  "
            f"p95={s['diff_mean']['p95']:.6f}"
        )
        print(
            "  grad_diff_mean"
            f" mean={s['grad_diff_mean']['mean']:.6f}  "
            f"median={s['grad_diff_mean']['median']:.6f}  "
            f"p95={s['grad_diff_mean']['p95']:.6f}"
        )
        print(
            "  diff_rms      "
            f"mean={s['diff_rms']['mean']:.6f}  "
            f"median={s['diff_rms']['median']:.6f}  "
            f"p95={s['diff_rms']['p95']:.6f}"
        )
        print(
            "  depth_std     "
            f"mean={s['depth_std']['mean']:.6f}  "
            f"median={s['depth_std']['median']:.6f}  "
            f"p95={s['depth_std']['p95']:.6f}"
        )
        print(
            "  track_err_lin "
            f"mean={s['track_err_lin']['mean']:.6f}  "
            f"median={s['track_err_lin']['median']:.6f}  "
            f"p95={s['track_err_lin']['p95']:.6f}"
        )
        print(
            "  track_err_yaw "
            f"mean={s['track_err_yaw']['mean']:.6f}  "
            f"median={s['track_err_yaw']['median']:.6f}  "
            f"p95={s['track_err_yaw']['p95']:.6f}"
        )
        print(
            "  energy        "
            f"mean={s['energy']['mean']:.6f}  "
            f"median={s['energy']['median']:.6f}  "
            f"p95={s['energy']['p95']:.6f}"
        )
        print(
            "  ang_vel_xy    "
            f"mean={s['base_ang_vel_xy_rms']['mean']:.6f}  "
            f"median={s['base_ang_vel_xy_rms']['median']:.6f}  "
            f"p95={s['base_ang_vel_xy_rms']['p95']:.6f}"
        )
        print(
            "  ang_acc_xy    "
            f"mean={s['base_ang_acc_xy_rms']['mean']:.6f}  "
            f"median={s['base_ang_acc_xy_rms']['median']:.6f}  "
            f"p95={s['base_ang_acc_xy_rms']['p95']:.6f}"
        )
        print(
            "  roll          "
            f"mean={s['roll']['mean']:.6f}  "
            f"std={s['roll']['std']:.6f}  "
            f"p95={s['roll']['p95']:.6f}"
        )
        print(
            "  pitch         "
            f"mean={s['pitch']['mean']:.6f}  "
            f"std={s['pitch']['std']:.6f}  "
            f"p95={s['pitch']['p95']:.6f}"
        )
        print(
            "  depth_invalid "
            f"mean={s['depth_invalid_ratio']['mean']:.6f}  "
            f"median={s['depth_invalid_ratio']['median']:.6f}  "
            f"p95={s['depth_invalid_ratio']['p95']:.6f}"
        )
        print(
            "  depth_clip    "
            f"mean={s['depth_clip_ratio']['mean']:.6f}  "
            f"median={s['depth_clip_ratio']['median']:.6f}  "
            f"p95={s['depth_clip_ratio']['p95']:.6f}"
        )
        print(
            "  depth_grad    "
            f"mean={s['depth_grad_mean']['mean']:.6f}  "
            f"median={s['depth_grad_mean']['median']:.6f}  "
            f"p95={s['depth_grad_mean']['p95']:.6f}"
        )
        print(
            "  feature_count "
            f"mean={s['feature_count']['mean']:.6f}  "
            f"median={s['feature_count']['median']:.6f}  "
            f"p95={s['feature_count']['p95']:.6f}"
        )
        print(
            "  edge_density  "
            f"mean={s['edge_density']['mean']:.6f}  "
            f"median={s['edge_density']['median']:.6f}  "
            f"p95={s['edge_density']['p95']:.6f}"
        )

        if args.per_cmd:
            per_cmd = _per_cmd_stats(run["path"])
            if per_cmd:
                print("  per-cmd (mean):")
                cmd_header = (
                    "    cmd          count  diff_mean  grad_mean  diff_rms  depth_std"
                    "  track_lin  track_yaw  energy"
                )
                print(cmd_header)
                for cmd in sorted(per_cmd.keys()):
                    item = per_cmd[cmd]
                    print(
                        f"    {cmd[:10].ljust(10)}  "
                        f"{int(item['count']):>5d}  "
                        f"{item['diff_mean']:.6f}  "
                        f"{item['grad_diff_mean']:.6f}  "
                        f"{item['diff_rms']:.6f}  "
                        f"{item['depth_std']:.6f}  "
                        f"{item['track_err_lin']:.6f}  "
                        f"{item['track_err_yaw']:.6f}  "
                        f"{item['energy']:.6f}"
                    )

    if args.out:
        with open(args.out, "w", encoding="ascii", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "path",
                    "n",
                    "diff_mean_mean",
                    "diff_mean_median",
                    "diff_mean_p95",
                    "grad_diff_mean_mean",
                    "grad_diff_mean_median",
                    "grad_diff_mean_p95",
                    "diff_rms_mean",
                    "diff_rms_median",
                    "diff_rms_p95",
                    "depth_std_mean",
                    "depth_std_median",
                    "depth_std_p95",
                    "track_err_lin_mean",
                    "track_err_lin_median",
                    "track_err_lin_p95",
                    "track_err_yaw_mean",
                    "track_err_yaw_median",
                    "track_err_yaw_p95",
                    "energy_mean",
                    "energy_median",
                    "energy_p95",
                    "base_ang_vel_xy_rms_mean",
                    "base_ang_vel_xy_rms_median",
                    "base_ang_vel_xy_rms_p95",
                    "base_ang_acc_xy_rms_mean",
                    "base_ang_acc_xy_rms_median",
                    "base_ang_acc_xy_rms_p95",
                    "roll_mean",
                    "roll_std",
                    "roll_p95",
                    "pitch_mean",
                    "pitch_std",
                    "pitch_p95",
                    "depth_invalid_ratio_mean",
                    "depth_invalid_ratio_median",
                    "depth_invalid_ratio_p95",
                    "depth_clip_ratio_mean",
                    "depth_clip_ratio_median",
                    "depth_clip_ratio_p95",
                    "depth_grad_mean_mean",
                    "depth_grad_mean_median",
                    "depth_grad_mean_p95",
                    "feature_count_mean",
                    "feature_count_median",
                    "feature_count_p95",
                    "edge_density_mean",
                    "edge_density_median",
                    "edge_density_p95",
                    "tracking_mean",
                    "energy_mean",
                    "composite",
                ]
            )
            for run in runs_sorted:
                s = run["stats"]
                writer.writerow(
                    [
                        run["name"],
                        run["path"],
                        run["count"],
                        s["diff_mean"]["mean"],
                        s["diff_mean"]["median"],
                        s["diff_mean"]["p95"],
                        s["grad_diff_mean"]["mean"],
                        s["grad_diff_mean"]["median"],
                        s["grad_diff_mean"]["p95"],
                        s["diff_rms"]["mean"],
                        s["diff_rms"]["median"],
                        s["diff_rms"]["p95"],
                        s["depth_std"]["mean"],
                        s["depth_std"]["median"],
                        s["depth_std"]["p95"],
                        s["track_err_lin"]["mean"],
                        s["track_err_lin"]["median"],
                        s["track_err_lin"]["p95"],
                        s["track_err_yaw"]["mean"],
                        s["track_err_yaw"]["median"],
                        s["track_err_yaw"]["p95"],
                        s["energy"]["mean"],
                        s["energy"]["median"],
                        s["energy"]["p95"],
                        s["base_ang_vel_xy_rms"]["mean"],
                        s["base_ang_vel_xy_rms"]["median"],
                        s["base_ang_vel_xy_rms"]["p95"],
                        s["base_ang_acc_xy_rms"]["mean"],
                        s["base_ang_acc_xy_rms"]["median"],
                        s["base_ang_acc_xy_rms"]["p95"],
                        s["roll"]["mean"],
                        s["roll"]["std"],
                        s["roll"]["p95"],
                        s["pitch"]["mean"],
                        s["pitch"]["std"],
                        s["pitch"]["p95"],
                        s["depth_invalid_ratio"]["mean"],
                        s["depth_invalid_ratio"]["median"],
                        s["depth_invalid_ratio"]["p95"],
                        s["depth_clip_ratio"]["mean"],
                        s["depth_clip_ratio"]["median"],
                        s["depth_clip_ratio"]["p95"],
                        s["depth_grad_mean"]["mean"],
                        s["depth_grad_mean"]["median"],
                        s["depth_grad_mean"]["p95"],
                        s["feature_count"]["mean"],
                        s["feature_count"]["median"],
                        s["feature_count"]["p95"],
                        s["edge_density"]["mean"],
                        s["edge_density"]["median"],
                        s["edge_density"]["p95"],
                        run["rank_keys"]["tracking_mean"],
                        run["rank_keys"]["energy_mean"],
                        run["composite"],
                    ]
                )
        print(f"\nSaved summary to: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
