#!/usr/bin/env python3
"""Build final paper tables and simple figures from completed eval outputs."""

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_ORDER = ["yonly", "geomw", "risk_only", "rule_override", "learnedw"]
METHOD_LABELS = {
    "yonly": "Y-only",
    "geomw": "Geom-w",
    "risk_only": "Risk-only",
    "rule_override": "Rule-Override",
    "learnedw": "Learned-w",
    "mono_ppo": "Mono-PPO",
}


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown(path: str, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(max(var, 0.0))


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(float(mean)):
        return "N/A"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _file_stamp(path: str) -> str:
    if not path:
        return "missing"
    if not os.path.exists(path):
        return "missing"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def _infer_speed(row: Dict[str, str]) -> str:
    for key in ("Speed", "speed", "resolved_moving_target_pcr_line_speed"):
        if key in row and row.get(key, "") not in ("", None):
            v = _safe_float(row.get(key))
            if math.isfinite(v):
                return f"{v:.2f}"
    src = str(row.get("source", "") or row.get("Source", ""))
    m = re.search(r"/s_(0\.35|0\.5|0\.50|0\.6|0\.60)(?:/|$)", src)
    if m:
        return f"{float(m.group(1)):.2f}"
    return ""


def _infer_method(row: Dict[str, str]) -> str:
    text = " ".join(
        str(row.get(k, ""))
        for k in ("Method", "method", "policy_variant", "w_mode", "source", "Source")
    ).lower()
    if "risk_only" in text or "risk-only" in text:
        return "risk_only"
    if "rule_override" in text or "rule-override" in text:
        return "rule_override"
    if "mono_ppo" in text or "mono-ppo" in text:
        return "mono_ppo"
    if "learnedw2" in text or "learnedw" in text or "learned" in text:
        return "learnedw"
    if "geomw" in text or "geom" in text:
        return "geomw"
    if "yonly" in text or re.search(r"\bnone\b", text):
        return "yonly"
    return str(row.get("Method", "") or row.get("policy_variant", "") or "unknown")


def _method_sort(method: str) -> Tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def _seed(row: Dict[str, str]) -> str:
    return str(row.get("seed", "") or row.get("Seed", "")).strip()


def _latest_by_key(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    selected: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        speed = _infer_speed(row)
        method = _infer_method(row)
        seed = _seed(row)
        key = (speed, method, seed)
        src = str(row.get("source", "") or row.get("Source", ""))
        mtime = os.path.getmtime(src) if os.path.exists(src) else 0.0
        prev = selected.get(key)
        if prev is None or mtime >= float(prev.get("_mtime", 0.0)):
            copy = dict(row)
            copy["_speed"] = speed
            copy["_method"] = method
            copy["_seed"] = seed
            copy["_mtime"] = mtime
            selected[key] = copy
    return list(selected.values())


def _raw_rows_from_all_csv(paths: Sequence[str], methods: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        for row in _read_csv(path):
            method = _infer_method(row)
            if method not in methods:
                continue
            speed = _infer_speed(row)
            if not speed:
                continue
            copy = dict(row)
            copy["_speed"] = speed
            copy["_method"] = method
            copy["_seed"] = _seed(row)
            rows.append(copy)
    return _latest_by_key(rows)


def _aggregate_raw(
    rows: Sequence[Dict[str, str]],
    metrics: Sequence[Tuple[str, str]],
    methods: Sequence[str],
) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        speed = row.get("_speed") or _infer_speed(row)
        method = row.get("_method") or _infer_method(row)
        if method not in methods:
            continue
        groups[(speed, method)].append(row)

    out: List[Dict] = []
    for (speed, method), group in sorted(
        groups.items(), key=lambda kv: (float(kv[0][0] or 0.0), _method_sort(kv[0][1]))
    ):
        agg = {
            "Speed": speed,
            "Method": METHOD_LABELS.get(method, method),
            "method_key": method,
            "Seeds": ",".join(sorted({_seed(r) for r in group if _seed(r)})),
            "Num Runs": len(group),
        }
        for title, key in metrics:
            mean, std = _mean_std(_safe_float(r.get(key, "")) for r in group)
            agg[f"{title} Mean"] = mean
            agg[f"{title} Std"] = std
            agg[title] = _fmt_pm(mean, std)
        out.append(agg)
    return out


def _build_table1(args) -> List[Dict]:
    rows: List[Dict[str, str]] = []
    rows.extend(
        _raw_rows_from_all_csv(
            [args.internal_all_csv],
            methods=["yonly", "geomw", "learnedw"],
        )
    )
    rows.extend(_raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"]))
    rows.extend(_raw_rows_from_all_csv([args.rule_all_csv], methods=["rule_override"]))
    metrics = [
        ("Task Success", "success_rate"),
        ("Row-progress", "progress_ratio_mean"),
        ("Collision", "episode_collision_rate"),
        ("Follow MAE", "follow_mae_m_mean"),
    ]
    table = _aggregate_raw(rows, metrics, METHOD_ORDER)
    fields = [
        "Speed",
        "Method",
        "Seeds",
        "Num Runs",
        "Task Success",
        "Row-progress",
        "Collision",
        "Follow MAE",
    ]
    _write_csv(os.path.join(args.output_dir, "table1_main_performance_stage4.csv"), table, fields)
    _write_markdown(os.path.join(args.output_dir, "table1_main_performance_stage4.md"), table, fields)
    return table


def _aggregate_csv_lookup(paths: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            speed = str(row.get("Speed", "")).strip()
            method = _infer_method(row)
            if not speed or method == "unknown":
                continue
            lookup[(f"{_safe_float(speed):.2f}", method)] = row
    return lookup


def _delta_lookup(rows: Sequence[Dict[str, str]], methods: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = row.get("_method") or _infer_method(row)
        speed = row.get("_speed") or _infer_speed(row)
        if method in methods and speed:
            grouped[(speed, method)].append(row)
    keys = [
        "delta_y_r_mean",
        "unsafe_conflict_delta_y_w_used_mean",
        "unsafe_conflict_delta_y_r_mean",
        "unsafe_conflict_delta_y_total_mean",
        "avoid_conflict_delta_y_w_used_mean",
        "avoid_conflict_delta_y_r_mean",
        "avoid_conflict_delta_y_total_mean",
    ]
    for key, group in grouped.items():
        out[key] = {}
        for metric_key in keys:
            mean, _ = _mean_std(_safe_float(r.get(metric_key, "")) for r in group)
            out[key][metric_key] = mean
    return out


def _risk_only_source_mode(args, rows: Sequence[Dict[str, str]]) -> str:
    requested = str(getattr(args, "risk_only_source_mode", "auto")).strip().lower()
    if requested == "trained":
        return requested
    for row in rows:
        if (row.get("_method") or _infer_method(row)) != "risk_only":
            continue
        text = " ".join(
            str(row.get(k, ""))
            for k in ("source", "Source", "policy_variant", "method", "Method", "Notes")
        ).lower()
        if "risk_only_learnedw2" in text or "eval-time" in text or "forced" in text:
            return "legacy_eval_only"
    return "trained"


def _build_table2(args) -> List[Dict]:
    aggregate_lookup = _aggregate_csv_lookup(
        [args.internal_aggregate_csv, args.risk_aggregate_csv, args.rule_aggregate_csv, args.learnedw_diag_aggregate_csv]
    )
    delta_rows: List[Dict[str, str]] = []
    delta_rows.extend(_raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"]))
    delta_rows.extend(_raw_rows_from_all_csv([args.learnedw_diag_all_csv], methods=["learnedw"]))
    deltas = _delta_lookup(delta_rows, methods=["risk_only", "learnedw"])
    risk_only_mode = _risk_only_source_mode(args, delta_rows)

    out: List[Dict] = []
    for speed in ("0.35", "0.50", "0.60"):
        for method in ("yonly", "geomw", "risk_only", "learnedw"):
            agg = aggregate_lookup.get((speed, method), {})
            d = deltas.get((speed, method), {})
            row = {
                "Speed": speed,
                "Method": METHOD_LABELS.get(method, method),
                "Unsafe Rate": _fmt(_safe_float(agg.get("Unsafe Rate Mean", ""))),
                "C_avoid Rate": _fmt(_safe_float(agg.get("C_avoid Rate Mean", ""))),
                "CSI@C_avoid": _fmt(_safe_float(agg.get("CSI@C_avoid Mean", ""))),
                "Delta y_w@C_avoid": (
                    _fmt(d.get("avoid_conflict_delta_y_w_used_mean", float("nan")), digits=4) if method in ("risk_only", "learnedw") else "N/A"
                ),
                "Delta y_r (all)": (
                    _fmt(d.get("delta_y_r_mean", float("nan")), digits=4) if method in ("risk_only", "learnedw") else "N/A"
                ),
                "Delta y_r@C_unsafe": (
                    _fmt(d.get("unsafe_conflict_delta_y_r_mean", float("nan")), digits=4) if method in ("risk_only", "learnedw") else "N/A"
                ),
                "Delta y_r@C_avoid": (
                    _fmt(d.get("avoid_conflict_delta_y_r_mean", float("nan")), digits=4) if method in ("risk_only", "learnedw") else "N/A"
                ),
                "Delta y_total@C_avoid": (
                    _fmt(d.get("avoid_conflict_delta_y_total_mean", float("nan")), digits=4) if method in ("risk_only", "learnedw") else "N/A"
                ),
                "Notes": _table2_note(method, risk_only_mode=risk_only_mode),
            }
            out.append(row)
    fields = [
        "Speed",
        "Method",
        "Unsafe Rate",
        "C_avoid Rate",
        "CSI@C_avoid",
        "Delta y_w@C_avoid",
        "Delta y_r (all)",
        "Delta y_r@C_unsafe",
        "Delta y_r@C_avoid",
        "Delta y_total@C_avoid",
        "Notes",
    ]
    _write_csv(os.path.join(args.output_dir, "table2_mechanism_ablation.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "table2_mechanism_ablation.md"), out, fields)
    return out


def _table2_note(method: str, risk_only_mode: str = "auto") -> str:
    if method == "risk_only":
        if risk_only_mode == "trained":
            return "trained from scratch with risk-difference term Δy_r only; no learned-w channel"
        return "LEGACY eval-only source; rerun trained Risk-only before using this row."
    if method == "learnedw":
        return "Delta terms from learned-w diagnostic eval."
    if method in ("yonly", "geomw"):
        return "No learned-w correction term."
    return ""


def _find_metrics_json(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    if os.path.isfile(path) and path.endswith("metrics.json"):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        if "metrics.json" in files:
            out.append(os.path.join(root, "metrics.json"))
    return sorted(out)


def _metrics_row(path: str) -> Optional[Dict]:
    files = _find_metrics_json(path)
    if not files:
        return None
    latest = max(files, key=lambda p: os.path.getmtime(p))
    with open(latest, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    overall = metrics.get("overall", {})
    return {
        "success_rate": overall.get("success_rate", ""),
        "progress_ratio_mean": overall.get("progress_ratio_mean", ""),
        "episode_collision_rate": overall.get("episode_collision_rate", ""),
        "l1_safety_success_rate": overall.get("l1_safety_success_rate", ""),
        "l2_follow_success_rate": overall.get("l2_follow_success_rate", ""),
        "l3_progress_success_rate": overall.get("l3_progress_success_rate", ""),
        "target_in_rgb_fov_rate": overall.get("target_in_rgb_fov_rate", ""),
        "target_bearing_abs_deg_p95": overall.get("target_bearing_abs_deg_p95", ""),
        "cmd_x_mean": overall.get("cmd_x_mean", ""),
        "Source": latest,
    }


def _stage4_mono_row(path: str) -> Optional[Dict]:
    rows = _read_csv(path)
    candidates = [r for r in rows if _infer_method(r) == "mono_ppo" and _infer_speed(r) == "0.35"]
    if not candidates:
        return None
    return candidates[0]


def _build_table3(args) -> List[Dict]:
    stage_sources = {
        "2": _metrics_row(args.mono_stage2_path),
        "3": _metrics_row(args.mono_stage3_path),
        "4": _stage4_mono_row(args.mono_stage4_all_csv),
    }
    out: List[Dict] = []
    missing_stages: List[str] = []
    for stage, row in stage_sources.items():
        if row is None:
            missing_stages.append(stage)
            continue
        out.append(
            {
                "Stage": stage,
                "Task Success": _fmt(_safe_float(row.get("success_rate", ""))),
                "Row-progress": _fmt(_safe_float(row.get("progress_ratio_mean", ""))),
                "Collision": _fmt(_safe_float(row.get("episode_collision_rate", ""))),
                "L1 Safety": _fmt(_safe_float(row.get("l1_safety_success_rate", ""))),
                "L2 Follow": _fmt(_safe_float(row.get("l2_follow_success_rate", ""))),
                "L3 Progress": _fmt(_safe_float(row.get("l3_progress_success_rate", ""))),
                "FOV-in": _fmt(_safe_float(row.get("target_in_rgb_fov_rate", ""))),
                "Bearing p95": _fmt(_safe_float(row.get("target_bearing_abs_deg_p95", "")), digits=2),
                "Cmd-x Mean": _fmt(_safe_float(row.get("cmd_x_mean", ""))),
                "Source": row.get("Source", row.get("source", "")),
            }
        )
    if missing_stages and not args.allow_incomplete_mono_stage_probe:
        raise RuntimeError(
            "Mono-PPO stage probe is incomplete; missing stage(s): "
            + ", ".join(missing_stages)
            + ". Re-run with --allow_incomplete_mono_stage_probe only for local dry-runs."
        )
    for stage in missing_stages:
        print(f"[Warn] Mono-PPO stage {stage} metrics not found; skipping.")
    fields = [
        "Stage",
        "Task Success",
        "Row-progress",
        "Collision",
        "L1 Safety",
        "L2 Follow",
        "L3 Progress",
        "FOV-in",
        "Bearing p95",
        "Cmd-x Mean",
        "Source",
    ]
    _write_csv(os.path.join(args.output_dir, "table3_mono_ppo_stage_probe.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "table3_mono_ppo_stage_probe.md"), out, fields[:-1])
    return out


def _plot_fig4(args, table1: Sequence[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable; skipping Fig.4 ({exc})")
        return

    metrics = [
        ("Task Success Mean", "Task Success"),
        ("Collision Mean", "Collision"),
        ("Follow MAE Mean", "Follow MAE [m]"),
    ]
    colors = {
        "yonly": "#7f7f7f",
        "geomw": "#1f77b4",
        "risk_only": "#ff7f0e",
        "rule_override": "#2ca02c",
        "learnedw": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    for ax, (metric_key, ylabel) in zip(axes, metrics):
        for method in METHOD_ORDER:
            pts = [
                r for r in table1
                if r.get("method_key") == method and math.isfinite(_safe_float(r.get(metric_key, "")))
            ]
            pts = sorted(pts, key=lambda r: _safe_float(r.get("Speed", "")))
            if not pts:
                continue
            x = [_safe_float(r["Speed"]) for r in pts]
            y = [_safe_float(r[metric_key]) for r in pts]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=METHOD_LABELS.get(method, method),
                color=colors.get(method),
            )
        ax.set_xlabel("Target speed [m/s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.set_xticks([0.35, 0.50, 0.60])
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.output_dir, f"fig4_speed_curves_stage4.{ext}"), dpi=300)
    plt.close(fig)


def _copy_fig5(args) -> None:
    if not args.learnedw_mechanism_dir or not os.path.isdir(args.learnedw_mechanism_dir):
        print("[Warn] learned-w mechanism directory not found; Fig.5 copy skipped.")
        return
    copied = []
    for name in ("mechanism_risk_bins.png", "mechanism_priv_conflict_bins.png"):
        src = os.path.join(args.learnedw_mechanism_dir, name)
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, "fig5_" + name)
            shutil.copy2(src, dst)
            copied.append(dst)
    if copied:
        with open(os.path.join(args.output_dir, "fig5_note.md"), "w", encoding="utf-8") as f:
            f.write("# Fig.5 mechanism source\n\n")
            f.write("Copied mechanism plots. Rename labels in the manuscript figure as:\n")
            f.write("- risk_F: Forward risk\n")
            f.write("- risk_A: Lateral-avoid risk\n")
            f.write("- y: Follow weight\n")
            f.write("- y_eff: Effective follow weight\n")
            f.write("- conflict mask: Conflict window\n")
    else:
        print("[Warn] no mechanism_*.png files found in learned-w mechanism directory.")


def _write_manifest(args, table1: Sequence[Dict], table2: Sequence[Dict], table3: Sequence[Dict]) -> None:
    path = os.path.join(args.output_dir, "MANIFEST.md")
    risk_rows = _raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"])
    risk_only_mode = _risk_only_source_mode(args, risk_rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Final Paper Outputs\n\n")
        f.write("Generated files:\n\n")
        for name in [
            "table1_main_performance_stage4.csv",
            "table1_main_performance_stage4.md",
            "fig4_speed_curves_stage4.png",
            "fig4_speed_curves_stage4.pdf",
            "table2_mechanism_ablation.csv",
            "table2_mechanism_ablation.md",
            "table3_mono_ppo_stage_probe.csv",
            "table3_mono_ppo_stage_probe.md",
        ]:
            f.write(f"- {name}\n")
        f.write("\nRow counts:\n\n")
        f.write(f"- Table I: {len(table1)} rows\n")
        f.write(f"- Table II: {len(table2)} rows\n")
        f.write(f"- Table III: {len(table3)} rows\n")
        f.write("\nSource files and paths:\n\n")
        for label, src in [
            ("internal_all_csv", args.internal_all_csv),
            ("internal_aggregate_csv", args.internal_aggregate_csv),
            ("risk_all_csv", args.risk_all_csv),
            ("risk_aggregate_csv", args.risk_aggregate_csv),
            ("rule_all_csv", args.rule_all_csv),
            ("rule_aggregate_csv", args.rule_aggregate_csv),
            ("learnedw_diag_all_csv", args.learnedw_diag_all_csv),
            ("learnedw_diag_aggregate_csv", args.learnedw_diag_aggregate_csv),
            ("mono_stage2_path", args.mono_stage2_path),
            ("mono_stage3_path", args.mono_stage3_path),
            ("mono_stage4_all_csv", args.mono_stage4_all_csv),
            ("learnedw_mechanism_dir", args.learnedw_mechanism_dir),
        ]:
            f.write(f"- {label}: `{src}` (mtime={_file_stamp(src)})\n")
        f.write("\nRisk-only source mode:\n\n")
        f.write(f"- requested: `{args.risk_only_source_mode}`\n")
        f.write(f"- resolved: `{risk_only_mode}`\n")
        f.write("\nValidation checklist:\n\n")
        f.write("- Fig.4: legend must contain Y-only / Geom-w / Risk-only / Rule-Override / Learned-w.\n")
        f.write("- Table I: expected 15 rows = 3 speeds x 5 methods; Mono-PPO is intentionally excluded.\n")
        if risk_only_mode == "trained":
            f.write("- Table II: Risk-only note should say trained from scratch; Delta y_r@C_avoid should be non-zero.\n")
            f.write("- Table II: if trained Risk-only Delta y_r@C_avoid is 0.000, inspect the trained source and C_avoid window before using the table.\n")
        else:
            f.write("- Table II: current Risk-only source is legacy eval-only; rerun trained Risk-only and do not use this table in the paper.\n")
        f.write("- Table II: Risk-only Delta y_w@C_avoid should be 0.000; Learned-w Delta y_w@C_avoid should be non-zero.\n")
        f.write("- Table III: expected stages 2, 3, and 4; one-row stage4 output is not a valid paper table.\n")
        f.write("- Source timestamps: verify all source paths are the intended final eval outputs.\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final PCR paper tables and figures.")
    parser.add_argument("--output_dir", default="agents/final_paper_outputs")
    parser.add_argument(
        "--internal_all_csv",
        default="agents/eval_data_seed23/pcr_main_table/pcr_main_table_all_metrics_with_rollout_20260529.csv",
    )
    parser.add_argument(
        "--internal_aggregate_csv",
        default="agents/eval_data_seed23/pcr_main_table/pcr_main_table_3seed_aggregate_20260529_manual_seed1.csv",
    )
    parser.add_argument(
        "--risk_all_csv",
        default="agents/eval_data_risk_only/pcr_main_table/pcr_main_table_all_metrics_20260601_223036.csv",
    )
    parser.add_argument(
        "--risk_aggregate_csv",
        default="agents/eval_data_risk_only/pcr_main_table/pcr_main_table_aggregate_20260601_223036.csv",
    )
    parser.add_argument(
        "--risk_only_source_mode",
        choices=("auto", "trained"),
        default="auto",
        help=(
            "How to label Risk-only in Table II. Risk-only is expected to be a "
            "from-scratch risk-difference-only policy."
        ),
    )
    parser.add_argument(
        "--rule_all_csv",
        default="agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_all_metrics_20260601_190358.csv",
    )
    parser.add_argument(
        "--rule_aggregate_csv",
        default="agents/eval_data_rule_override_current/pcr_main_table/pcr_main_table_aggregate_20260601_190358.csv",
    )
    parser.add_argument(
        "--learnedw_diag_all_csv",
        default="agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_all_metrics_20260602_141845.csv",
    )
    parser.add_argument(
        "--learnedw_diag_aggregate_csv",
        default="agents/eval_data_learnedw_diag/pcr_main_table/pcr_main_table_aggregate_20260602_141845.csv",
    )
    parser.add_argument(
        "--mono_stage2_path",
        default="agents/eval_data_mono_targetview_stage_probe/stage2_s035",
    )
    parser.add_argument(
        "--mono_stage3_path",
        default="agents/eval_data_mono_targetview_stage_probe/stage3_s035",
    )
    parser.add_argument(
        "--mono_stage4_all_csv",
        default="agents/eval_data_mono_targetview/pcr_main_table/pcr_main_table_all_metrics_20260604_160305.csv",
    )
    parser.add_argument(
        "--learnedw_mechanism_dir",
        default="agents/eval_data/s_0.6/moe_teacher_s_pcr_line_avoid_basic_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_seed1_20260526_204857",
    )
    parser.add_argument(
        "--allow_incomplete_mono_stage_probe",
        action="store_true",
        help="Allow Table III generation when Mono-PPO stage2/3/4 metrics are incomplete. Use only for local dry-runs.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    table1 = _build_table1(args)
    table2 = _build_table2(args)
    table3 = _build_table3(args)
    _plot_fig4(args, table1)
    _copy_fig5(args)
    _write_manifest(args, table1, table2, table3)
    print(f"Final paper outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
