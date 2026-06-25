#!/usr/bin/env python3
"""Build PCR-Net paper figures and tables from completed eval outputs."""

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_ORDER = ["yonly", "geomw", "risk_only", "rule_override", "learnedw"]
MECHANISM_METHOD_ORDER = ["yonly", "geomw", "risk_only", "learnedw"]
METHOD_LABELS = {
    "yonly": "Y-only",
    "geomw": "Geom-w",
    "risk_only": "Risk-only",
    "rule_override": "Rule-Override",
    "additive_fusion": "Additive-Fusion",
    "velocity_search": "Velocity-Search",
    "learnedw": "Learned-w (PCR-Net)",
    "mono_ppo": "Mono-PPO",
}
SHORT_METHOD_LABELS = {
    "yonly": "Y-only",
    "geomw": "Geom-w",
    "risk_only": "Risk-only",
    "rule_override": "Rule-Override",
    "additive_fusion": "Additive-Fusion",
    "velocity_search": "Velocity-Search",
    "learnedw": "Learned-w",
    "mono_ppo": "Mono-PPO",
}
METHOD_STYLE = {
    "yonly": {"marker": "o", "color": "#6b6b6b", "linewidth": 1.7},
    "geomw": {"marker": "s", "color": "#1f77b4", "linewidth": 1.7},
    "risk_only": {"marker": "^", "color": "#ff7f0e", "linewidth": 1.7},
    "rule_override": {"marker": "D", "color": "#2ca02c", "linewidth": 1.7},
    "additive_fusion": {"marker": "P", "color": "#9467bd", "linewidth": 1.7},
    "velocity_search": {"marker": "X", "color": "#8c564b", "linewidth": 1.7},
    "learnedw": {"marker": "*", "color": "#d62728", "linewidth": 2.0},
}
SPEEDS = ("0.35", "0.50", "0.60")
FIG6_SPEEDS = ("0.60",)


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


def _latex_escape(value) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "↑": r"$\uparrow$",
        "↓": r"$\downarrow$",
        "Δ": r"$\Delta$",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _write_latex_booktabs(path: str, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = "l" * len(headers)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + cols + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(_latex_escape(h) for h in headers) + r" \\" + "\n")
        f.write("\\midrule\n")
        for row in rows:
            vals = []
            for h in headers:
                value = str(row.get(h, ""))
                if value.startswith("__LATEX__"):
                    vals.append(value[len("__LATEX__"):])
                else:
                    vals.append(_latex_escape(value))
            f.write(" & ".join(vals) + r" \\" + "\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


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
    value = float(value)
    if abs(value) < 0.5 * (10.0 ** (-digits)):
        value = 0.0
    return f"{value:.{digits}f}"


def _fmt_small_nonzero(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "N/A"
    value = float(value)
    if value == 0.0:
        return f"{0.0:.{digits}f}"
    if abs(value) < 10.0 ** (-digits):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(float(mean)):
        return "N/A"
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _fmt_pm_latex(mean: float, std: float, digits: int = 3, bold: bool = False) -> str:
    if not math.isfinite(float(mean)):
        return "N/A"
    text = f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"
    return r"\textbf{" + text + "}" if bold else text


def _fmt_pm_md(mean: float, std: float, digits: int = 3, bold: bool = False) -> str:
    text = _fmt_pm(mean, std, digits=digits)
    return f"**{text}**" if bold and text != "N/A" else text


def _file_stamp(path: str) -> str:
    if not path:
        return "missing"
    if not os.path.exists(path):
        return "missing"
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def _file_sha256(path: str) -> str:
    if not path or not os.path.isfile(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to read final checkpoint metadata.") from exc
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {path}")
    return data


def _checkpoint_meta(path: str) -> Dict[str, Any]:
    data = _load_checkpoint(path)
    meta = data.get("experiment_meta", {})
    if not isinstance(meta, dict):
        raise RuntimeError(f"Checkpoint has no experiment_meta dictionary: {path}")
    return meta


def _checkpoint_state_dict(path: str) -> Dict[str, Any]:
    data = _load_checkpoint(path)
    state_dict = data.get("model_state_dict", data.get("state_dict", {}))
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Checkpoint has no model state dictionary: {path}")
    return state_dict


def _checkpoint_iteration(path: str) -> str:
    data = _load_checkpoint(path)
    value = data.get("iteration", data.get("iter", "N/A"))
    return str(value)


def _shape(state_dict: Dict[str, Any], key: str) -> Tuple[int, ...]:
    tensor = state_dict.get(key)
    if tensor is None or not hasattr(tensor, "shape"):
        raise RuntimeError(f"Checkpoint tensor missing: {key}")
    return tuple(int(v) for v in tensor.shape)


def _state_dict_param_count(state_dict: Dict[str, Any]) -> int:
    total = 0
    for key, value in state_dict.items():
        if key.endswith("cmd_scale"):
            continue
        if hasattr(value, "numel"):
            total += int(value.numel())
    return total


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
    if "velocity_search" in text or "velocity-search" in text:
        return "velocity_search"
    if "additive_fusion" in text or "additive-fusion" in text:
        return "additive_fusion"
    if "risk_only" in text or "risk-only" in text:
        return "risk_only"
    if "rule_override" in text or "rule-override" in text:
        return "rule_override"
    if "mono_ppo" in text or "mono-ppo" in text:
        return "mono_ppo"
    if "learnedw2" in text or "learnedw" in text or "learned-w" in text:
        return "learnedw"
    if "geomw" in text or "geom" in text:
        return "geomw"
    if "yonly" in text or re.search(r"\bnone\b", text):
        return "yonly"
    return str(row.get("Method", "") or row.get("policy_variant", "") or "unknown")


def _method_sort(method: str) -> Tuple[int, str]:
    order = ["yonly", "geomw", "risk_only", "rule_override", "additive_fusion", "velocity_search", "learnedw", "mono_ppo"]
    if method in order:
        return (order.index(method), method)
    return (len(order), method)


def _seed(row: Dict[str, str]) -> str:
    return str(row.get("seed", "") or row.get("Seed", "")).strip()


def _load_metrics_json(path: str) -> Dict:
    if not path or not path.endswith(".json") or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _enrich_metrics_from_source(row: Dict[str, str]) -> Dict[str, str]:
    src = str(row.get("source", "") or row.get("Source", ""))
    data = _load_metrics_json(src)
    if not data:
        return row
    out = dict(row)
    for block_name in ("overall", "protocol", "params"):
        block = data.get(block_name, {})
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, (str, int, float, bool)) and str(out.get(key, "")) in ("", "nan", "None"):
                out[key] = str(value)
    overall = data.get("overall", {})
    if isinstance(overall, dict):
        fallback_pairs = {
            "unsafe_conflict_delta_y_total_mean": "unsafe_conflict_delta_y_mean",
            "avoid_conflict_delta_y_total_mean": "avoid_conflict_delta_y_mean",
            "stop_conflict_delta_y_total_mean": "stop_conflict_delta_y_mean",
        }
        for dst, src_key in fallback_pairs.items():
            if str(out.get(dst, "")) in ("", "nan", "None") and src_key in overall:
                out[dst] = str(overall.get(src_key))
    return out


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
            row = _enrich_metrics_from_source(row)
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
            "Method": SHORT_METHOD_LABELS.get(method, method),
            "method_key": method,
            "Seeds": ",".join(sorted({_seed(r) for r in group if _seed(r)})),
            "Num Runs": len(group),
            "Sources": ";".join(str(r.get("source", "") or r.get("Source", "")) for r in group),
        }
        for title, key in metrics:
            mean, std = _mean_std(_safe_float(r.get(key, "")) for r in group)
            agg[f"{title} Mean"] = mean
            agg[f"{title} Std"] = std
            agg[title] = _fmt_pm(mean, std)
        out.append(agg)
    return out


def _load_all_method_rows(args) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rows.extend(_raw_rows_from_all_csv([args.internal_all_csv], methods=["yonly", "geomw", "learnedw"]))
    rows.extend(_raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"]))
    rows.extend(_raw_rows_from_all_csv([args.rule_all_csv], methods=["rule_override"]))
    rows.extend(_raw_rows_from_all_csv([getattr(args, "additive_all_csv", "")], methods=["additive_fusion"]))
    rows.extend(_raw_rows_from_all_csv([getattr(args, "velocity_search_all_csv", "")], methods=["velocity_search"]))
    return rows


def _build_table1(args) -> List[Dict]:
    rows = _load_all_method_rows(args)
    metrics = [
        ("Task Success", "success_rate"),
        ("Row-progress", "progress_ratio_mean"),
        ("Collision", "episode_collision_rate"),
        ("Follow MAE [m]", "follow_mae_m_mean"),
    ]
    table_methods = list(METHOD_ORDER)
    extra_methods = [row.get("_method") or _infer_method(row) for row in rows]
    if "additive_fusion" in extra_methods or "velocity_search" in extra_methods:
        table_methods = ["yonly", "geomw", "risk_only", "rule_override"]
        if "additive_fusion" in extra_methods:
            table_methods.append("additive_fusion")
        if "velocity_search" in extra_methods:
            table_methods.append("velocity_search")
        table_methods.append("learnedw")
    table = _aggregate_raw(rows, metrics, table_methods)

    best_by_speed: Dict[str, Dict[str, float]] = {}
    for speed in SPEEDS:
        group = [r for r in table if r.get("Speed") == speed]
        if not group:
            continue
        best_by_speed[speed] = {
            "Task Success": max(_safe_float(r.get("Task Success Mean", "")) for r in group),
            "Row-progress": max(_safe_float(r.get("Row-progress Mean", "")) for r in group),
            "Collision": min(_safe_float(r.get("Collision Mean", "")) for r in group),
            "Follow MAE [m]": min(_safe_float(r.get("Follow MAE [m] Mean", "")) for r in group),
        }

    plain_rows: List[Dict] = []
    md_rows: List[Dict] = []
    tex_rows: List[Dict] = []
    for row in table:
        speed = row["Speed"]
        plain = {"Speed": speed, "Method": row["Method"]}
        md = dict(plain)
        tex = dict(plain)
        for title, out_title in [
            ("Task Success", "Task Success ↑"),
            ("Row-progress", "Row-progress ↑"),
            ("Collision", "Collision ↓"),
            ("Follow MAE [m]", "Follow MAE [m] ↓"),
        ]:
            mean = _safe_float(row.get(f"{title} Mean", ""))
            std = _safe_float(row.get(f"{title} Std", ""), default=0.0)
            best = best_by_speed.get(speed, {}).get(title, float("nan"))
            is_best = math.isfinite(mean) and math.isfinite(best) and abs(mean - best) < 5e-7
            plain[out_title] = _fmt_pm(mean, std)
            md[out_title] = _fmt_pm_md(mean, std, bold=is_best)
            tex[out_title] = "__LATEX__" + _fmt_pm_latex(mean, std, bold=is_best)
        plain_rows.append(plain)
        md_rows.append(md)
        tex_rows.append(tex)

    fields = ["Speed", "Method", "Task Success ↑", "Row-progress ↑", "Collision ↓", "Follow MAE [m] ↓"]
    _write_csv(os.path.join(args.output_dir, "table1_main_performance_stage4.csv"), plain_rows, fields)
    _write_markdown(os.path.join(args.output_dir, "table1_main_performance_stage4.md"), md_rows, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "table1_main_performance_stage4.tex"), tex_rows, fields)
    _write_csv(
        os.path.join(args.output_dir, "table1_main_performance_stage4_audit.csv"),
        table,
        [
            "Speed", "Method", "Seeds", "Num Runs",
            "Task Success Mean", "Task Success Std",
            "Row-progress Mean", "Row-progress Std",
            "Collision Mean", "Collision Std",
            "Follow MAE [m] Mean", "Follow MAE [m] Std",
            "Sources",
        ],
    )
    return table


def _aggregate_csv_lookup(paths: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in paths:
        for row in _read_csv(path):
            row = _enrich_metrics_from_source(row)
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
        "delta_y_w_raw_mean",
        "delta_y_w_used_mean",
        "delta_y_r_mean",
        "delta_y_total_mean",
        "unsafe_conflict_delta_y_w_raw_mean",
        "unsafe_conflict_delta_y_w_used_mean",
        "unsafe_conflict_delta_y_r_mean",
        "unsafe_conflict_delta_y_total_mean",
        "avoid_conflict_delta_y_w_raw_mean",
        "avoid_conflict_delta_y_w_used_mean",
        "avoid_conflict_delta_y_r_mean",
        "avoid_conflict_delta_y_total_mean",
    ]
    for key, group in grouped.items():
        out[key] = {}
        for metric_key in keys:
            values = [_safe_float(r.get(metric_key, "")) for r in group]
            mean, std = _mean_std(values)
            out[key][metric_key] = mean
            out[key][metric_key + "_std"] = std
    return out


def _params_lookup(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        speed = row.get("_speed") or _infer_speed(row)
        method = row.get("_method") or _infer_method(row)
        value = _safe_float(row.get("params_total", ""))
        if speed and method and math.isfinite(value):
            grouped[(speed, method)].append(value)
    out = {}
    for key, values in grouped.items():
        mean, _ = _mean_std(values)
        out[key] = str(int(round(mean))) if math.isfinite(mean) else "N/A"
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


def _table2_note(method: str, risk_only_mode: str = "auto") -> str:
    if method == "risk_only":
        if risk_only_mode == "trained":
            return "Trained from scratch with Δy_r only; no learned-w channel"
        return "LEGACY eval-only source; rerun trained Risk-only before using this row."
    if method == "learnedw":
        return "Learned conflict-conditioned w channel"
    if method == "geomw":
        return "Hand-designed geometric w"
    if method == "yonly":
        return "No arbitration channel"
    return ""


def _build_table2(args) -> List[Dict]:
    speed = f"{float(args.mechanism_speed):.2f}"
    aggregate_lookup = _aggregate_csv_lookup(
        [
            args.internal_aggregate_csv,
            args.risk_aggregate_csv,
            args.rule_aggregate_csv,
            args.learnedw_diag_aggregate_csv,
        ]
    )
    all_rows: List[Dict[str, str]] = []
    all_rows.extend(_raw_rows_from_all_csv([args.internal_all_csv], methods=["yonly", "geomw", "learnedw"]))
    all_rows.extend(_raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"]))
    all_rows.extend(_raw_rows_from_all_csv([args.learnedw_diag_all_csv], methods=["learnedw"]))
    deltas = _delta_lookup(all_rows, methods=["risk_only", "learnedw"])
    risk_only_mode = _risk_only_source_mode(args, all_rows)

    out: List[Dict] = []
    for method in MECHANISM_METHOD_ORDER:
        agg = aggregate_lookup.get((speed, method), {})
        d = deltas.get((speed, method), {})
        delta_y_w_avoid = d.get("avoid_conflict_delta_y_w_used_mean", float("nan"))
        delta_y_r_avoid = d.get("avoid_conflict_delta_y_r_mean", float("nan"))
        has_w_channel = method == "learnedw"
        has_risk_term = method in ("risk_only", "learnedw")
        row = {
            "Method": SHORT_METHOD_LABELS.get(method, method),
            "C_avoid Rate": _fmt(_safe_float(agg.get("C_avoid Rate Mean", ""))),
            "CSI@C_avoid": _fmt(_safe_float(agg.get("CSI@C_avoid Mean", ""))),
            "Δy_w@C_avoid": (
                _fmt_small_nonzero(delta_y_w_avoid, digits=4)
                if has_w_channel
                else "N/A"
            ),
            "Δy_r@C_avoid": (
                _fmt_small_nonzero(delta_y_r_avoid, digits=4)
                if has_risk_term
                else "N/A"
            ),
            "Notes": _table2_note(method, risk_only_mode=risk_only_mode),
        }
        out.append(row)
    fields = ["Method", "C_avoid Rate", "CSI@C_avoid", "Δy_w@C_avoid", "Δy_r@C_avoid", "Notes"]
    _write_csv(os.path.join(args.output_dir, "table2_mechanism_ablation.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "table2_mechanism_ablation.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "table2_mechanism_ablation.tex"), out, fields)
    return out


def _build_table_a5(args) -> List[Dict]:
    rows: List[Dict[str, str]] = []
    rows.extend(_raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"]))
    learned_split_source = args.learnedw_split_all_csv or args.learnedw_diag_all_csv
    rows.extend(_raw_rows_from_all_csv([learned_split_source], methods=["learnedw"]))
    deltas = _delta_lookup(rows, methods=["risk_only", "learnedw"])
    out: List[Dict] = []
    specs = [
        ("All", "delta_y_w_used_mean", "delta_y_r_mean", "delta_y_total_mean"),
        ("C_unsafe", "unsafe_conflict_delta_y_w_used_mean", "unsafe_conflict_delta_y_r_mean", "unsafe_conflict_delta_y_total_mean"),
        ("C_avoid", "avoid_conflict_delta_y_w_used_mean", "avoid_conflict_delta_y_r_mean", "avoid_conflict_delta_y_total_mean"),
    ]
    for speed in SPEEDS:
        for method in ("risk_only", "learnedw"):
            d = deltas.get((speed, method), {})
            for window, w_key, r_key, total_key in specs:
                values = {
                    "Δy_w": d.get(w_key, float("nan")),
                    "Δy_r": d.get(r_key, float("nan")),
                    "Δy_total": d.get(total_key, float("nan")),
                }
                missing = [name for name, value in values.items() if not math.isfinite(float(value))]
                if missing:
                    raise RuntimeError(
                        "Table A5 is incomplete: "
                        f"speed={speed}, method={method}, window={window}, "
                        f"missing={','.join(missing)}. Re-run the mechanism diagnostic with "
                        "the current eval_highlevel.py and summarize_eval_metrics.py."
                    )
                out.append(
                    {
                        "Speed": speed,
                        "Method": SHORT_METHOD_LABELS.get(method, method),
                        "Window": window,
                        "Δy_w": _fmt(values["Δy_w"], digits=6),
                        "Δy_r": _fmt(values["Δy_r"], digits=6),
                        "Δy_total": _fmt(values["Δy_total"], digits=6),
                    }
                )
    fields = ["Speed", "Method", "Window", "Δy_w", "Δy_r", "Δy_total"]
    _write_csv(os.path.join(args.output_dir, "tableA5_delta_y_full.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA5_delta_y_full.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA5_delta_y_full.tex"), out, fields)
    with open(
        os.path.join(args.output_dir, "tableA5_delta_y_full_notes.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Table A5 notes\n\n")
        f.write(
            "- Risk-only values are aggregated from the trained three-seed main evaluation.\n"
        )
        f.write(
            "- Learned-w values use the current seed-1 mechanism diagnostic because the "
            "older diagnostic predates the C_unsafe component split.\n"
        )
        f.write(
            "- Each row reports components from one internally consistent source; "
            "Delta y_total is not used to back-calculate missing components.\n"
        )
    return out


def _speed_dir_tags(speed_label: str) -> List[str]:
    value = float(speed_label)
    tags = [f"s_{speed_label}", f"s_{value:g}", f"s_{value:.2f}"]
    out: List[str] = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return out


def _speed_matches_source(source: str, speed_label: str) -> bool:
    source = str(source)
    for tag in _speed_dir_tags(speed_label):
        if f"/{tag}/" in source or source.endswith(f"/{tag}") or f"\\{tag}\\" in source:
            return True
    return False


def _load_velocity_search_diagnostic_overall(preset_dir: str, speed_label: str) -> Tuple[Dict, str]:
    for tag in _speed_dir_tags(speed_label):
        search_dir = os.path.join(preset_dir, tag)
        matches = sorted(glob.glob(os.path.join(search_dir, "**", "metrics.json"), recursive=True))
        if matches:
            data = _load_metrics_json(matches[-1])
            overall = data.get("overall", {}) if isinstance(data, dict) else {}
            return overall, matches[-1]

    csv_matches = sorted(
        glob.glob(os.path.join(preset_dir, "pcr_main_table", "pcr_main_table_all_metrics_*.csv"))
    )
    for csv_path in reversed(csv_matches):
        for row in _read_csv(csv_path):
            if _speed_matches_source(row.get("source", ""), speed_label):
                return row, csv_path

    raise FileNotFoundError(
        f"Missing dynamic-window Velocity-Search diagnostic metrics: {preset_dir}, speed={speed_label}"
    )


def _build_table_dwa_diagnostic(args) -> List[Dict]:
    root = getattr(args, "velocity_search_preset_root", "")
    specs = [
        ("Safety-biased", "0.35", os.path.join(root, "velocity_search_safe")),
        ("Safety-biased", "0.60", os.path.join(root, "velocity_search_safe")),
        ("Balanced", "0.35", os.path.join(root, "velocity_search_balanced")),
        ("Balanced", "0.60", os.path.join(root, "velocity_search_balanced")),
        ("Tracking-biased", "0.35", os.path.join(root, "velocity_search_tracking")),
        ("Tracking-biased", "0.60", os.path.join(root, "velocity_search_tracking")),
    ]
    out: List[Dict] = []
    for preset_label, speed_label, preset_dir in specs:
        overall, source_path = _load_velocity_search_diagnostic_overall(preset_dir, speed_label)
        dynamic_rate = _safe_float(overall.get("velocity_search_dynamic_window_enabled_rate", float("nan")))
        if not math.isfinite(dynamic_rate) or dynamic_rate < 0.99:
            raise ValueError(
                "Velocity-Search diagnostic table now requires dynamic-window eval metrics; "
                f"got dynamic_window_enabled_rate={dynamic_rate} from {source_path}"
            )
        out.append(
            {
                "Search bias": preset_label,
                "Speed": speed_label,
                "Success ↑": _fmt(_safe_float(overall.get("success_rate", ""))),
                "Collision ↓": _fmt(_safe_float(overall.get("episode_collision_rate", ""))),
                "Row-prog. ↑": _fmt(_safe_float(overall.get("progress_rate", ""))),
                "Follow MAE ↓": _fmt(_safe_float(overall.get("follow_mae_m_mean", ""))),
            }
        )
    fields = [
        "Search bias",
        "Speed",
        "Success ↑",
        "Collision ↓",
        "Row-prog. ↑",
        "Follow MAE ↓",
    ]
    _write_csv(os.path.join(args.output_dir, "tableA_dwa_velocity_search_diagnostic.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA_dwa_velocity_search_diagnostic.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA_dwa_velocity_search_diagnostic.tex"), out, fields)
    with open(
        os.path.join(args.output_dir, "tableA_dwa_velocity_search_diagnostic_notes.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# DWA-style velocity-search diagnostic notes\n\n")
        f.write(
            "The table reports bounded validation search biases for the dynamic-window Target-aware "
            "Velocity-Space Search baseline. Each row enables the dynamic-window candidate filter "
            "before short-horizon rollout and footprint checking. The safety-biased setting suppresses "
            "row progress without achieving reliable safety, while the tracking-biased setting improves "
            "progress but still incurs frequent collisions. This baseline is therefore treated as a "
            "diagnostic planner-style external alternative, not as a main-table competitor.\n"
        )
    return out


def _latest_heldout_all_csv(args) -> str:
    explicit = str(getattr(args, "heldout_layout_all_csv", "") or "").strip()
    if explicit:
        return explicit
    root = str(getattr(args, "heldout_layout_root", "") or "").strip()
    if not root:
        return ""
    matches = sorted(glob.glob(os.path.join(root, "heldout_table", "pcr_main_table_all_metrics_*.csv")))
    return matches[-1] if matches else ""


def _build_table_heldout_irregular(args) -> List[Dict]:
    source_csv = _latest_heldout_all_csv(args)
    methods = ["learnedw", "rule_override", "additive_fusion"]
    rows = _raw_rows_from_all_csv([source_csv], methods=methods) if source_csv else []
    rows = [
        row for row in rows
        if (row.get("_speed") or _infer_speed(row)) == "0.60"
        and str(row.get("eval_layout", "") or "").strip() == "heldout_irregular_rows"
    ]
    metrics = [
        ("Task Success", "success_rate"),
        ("Row-progress", "progress_ratio_mean"),
        ("Collision", "episode_collision_rate"),
        ("Follow MAE [m]", "follow_mae_m_mean"),
    ]
    table = _aggregate_raw(rows, metrics, methods) if rows else []
    out: List[Dict] = []
    for row in table:
        out.append(
            {
                "Method": row.get("Method", ""),
                "Task Success ↑": row.get("Task Success", ""),
                "Row-progress ↑": row.get("Row-progress", ""),
                "Collision ↓": row.get("Collision", ""),
                "Follow MAE ↓": row.get("Follow MAE [m]", ""),
            }
        )
    fields = ["Method", "Task Success ↑", "Row-progress ↑", "Collision ↓", "Follow MAE ↓"]
    _write_csv(os.path.join(args.output_dir, "tableA_heldout_irregular_rows.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA_heldout_irregular_rows.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA_heldout_irregular_rows.tex"), out, fields)
    notes_path = os.path.join(args.output_dir, "tableA_heldout_irregular_rows_notes.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# Held-out irregular-row notes\n\n")
        if not source_csv:
            f.write(
                "No held-out irregular-row CSV was found. Pass --heldout_layout_all_csv or "
                "run eval into --heldout_layout_root to populate this table.\n"
            )
        else:
            f.write(f"Source CSV: `{source_csv}`\n\n")
            f.write(
                "This eval-only table uses the Stage-4 heldout_irregular_rows layout at 0.60 m/s. "
                "The layout is not used for training or validation tuning. The main-text table intentionally "
                "reports only PCR-Net, Rule-Override, and Additive-Fusion, using 3 eval seeds and "
                "64 episodes per seed.\n"
            )
            if not out:
                f.write("\nWarning: source CSV was found but no 0.60 held-out rows were selected.\n")
    return out


def _find_metrics_json(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    if os.path.isfile(path) and path.endswith("metrics.json"):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        for name in files:
            if name == "metrics.json":
                out.append(os.path.join(root, name))
    return sorted(out)


def _metrics_row(path: str) -> Optional[Dict[str, str]]:
    paths = _find_metrics_json(path)
    if not paths:
        return None
    rows = []
    for p in paths:
        data = _load_metrics_json(p)
        if not data:
            continue
        overall = data.get("overall", {})
        if not isinstance(overall, dict):
            continue
        row = {k: str(v) for k, v in overall.items() if isinstance(v, (str, int, float, bool))}
        row["Source"] = p
        row["source"] = p
        row.update({k: str(v) for k, v in data.get("params", {}).items() if isinstance(v, (str, int, float, bool))})
        row["_ttc_mean_s"] = _ttc_mean_from_metrics(data)
        rows.append(row)
    if not rows:
        return None
    out: Dict[str, str] = {"Source": ";".join(r.get("Source", "") for r in rows)}
    keys = set().union(*(r.keys() for r in rows))
    for key in keys:
        vals = [_safe_float(r.get(key, "")) for r in rows]
        mean, _ = _mean_std(vals)
        if math.isfinite(mean):
            out[key] = str(mean)
    return out


def _ttc_mean_from_metrics(data: Dict) -> str:
    rows = data.get("per_episode", [])
    if not isinstance(rows, list):
        return ""
    vals = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        v = _safe_float(row.get("collision_time_s", ""))
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return ""
    return str(sum(vals) / len(vals))


def _stage4_mono_row(path: str) -> Optional[Dict[str, str]]:
    candidates = [
        _enrich_metrics_from_source(r)
        for r in _read_csv(path)
        if _infer_method(r) == "mono_ppo" and _infer_speed(r) == "0.35"
    ]
    if not candidates:
        return None
    row = candidates[0]
    src = row.get("source", "")
    data = _load_metrics_json(src)
    if data:
        row["_ttc_mean_s"] = _ttc_mean_from_metrics(data)
    return row


def _build_table3(args) -> List[Dict]:
    snapshot_rows = {
        str(row.get("Stage", "")): row
        for row in _read_csv(args.mono_stage_snapshot_csv)
        if str(row.get("Stage", "")) in ("2", "3", "4")
    }
    stage_sources = {
        "2": _metrics_row(args.mono_stage2_path),
        "3": _metrics_row(args.mono_stage3_path),
        "4": _stage4_mono_row(args.mono_stage4_all_csv),
    }
    for stage in ("2", "3", "4"):
        if stage_sources[stage] is None and stage in snapshot_rows:
            stage_sources[stage] = snapshot_rows[stage]
        elif stage_sources[stage] is not None and stage in snapshot_rows:
            for key, value in snapshot_rows[stage].items():
                if key in ("Stage", "Source"):
                    continue
                current = stage_sources[stage].get(key, "")
                if current in ("", None) or (
                    key == "_ttc_mean_s" and not math.isfinite(_safe_float(current))
                ):
                    stage_sources[stage][key] = value
    out: List[Dict] = []
    missing_stages: List[str] = []
    for stage, row in stage_sources.items():
        if row is None:
            missing_stages.append(stage)
            continue
        out.append(
            {
                "Stage": stage,
                "Success": _fmt(_safe_float(row.get("success_rate", ""))),
                "Row-prog": _fmt(_safe_float(row.get("progress_ratio_mean", ""))),
                "Collision": _fmt(_safe_float(row.get("episode_collision_rate", ""))),
                "L1": _fmt(_safe_float(row.get("l1_safety_success_rate", ""))),
                "L2": _fmt(_safe_float(row.get("l2_follow_success_rate", ""))),
                "L3": _fmt(_safe_float(row.get("l3_progress_success_rate", ""))),
                "FOV-in": _fmt(_safe_float(row.get("target_in_rgb_fov_rate", ""))),
                "Bearing p95": _fmt(_safe_float(row.get("target_bearing_abs_deg_p95", "")), digits=2),
                "Cmd-x": _fmt(_safe_float(row.get("cmd_x_mean", ""))),
                "Collision TTC [s]": _fmt(
                    _safe_float(
                        row.get(
                            "_ttc_mean_s",
                            row.get("Collision TTC [s]", row.get("TTC [s]", "")),
                        )
                    ),
                    digits=2,
                ),
                "Params": str(row.get("params_total", row.get("Params", "1029447"))),
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
        "Stage", "Success", "Row-prog", "Collision", "L1", "L2", "L3",
        "FOV-in", "Bearing p95", "Cmd-x", "Collision TTC [s]", "Params",
    ]
    _write_csv(os.path.join(args.output_dir, "table3_mono_ppo_stage_probe.csv"), out, fields + ["Source"])
    _write_markdown(os.path.join(args.output_dir, "table3_mono_ppo_stage_probe.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "table3_mono_ppo_stage_probe.tex"), out, fields)
    with open(
        os.path.join(args.output_dir, "table3_mono_ppo_stage_probe_notes.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Table III notes\n\n")
        f.write("- Diagnostic probe: fixed target speed 0.35 m/s, varying stage 2/3/4.\n")
        f.write("- Single seed (seed 1); report as a diagnostic probe, not a statistical comparison.\n")
        f.write(
            "- Collision TTC is averaged over collision episodes only; the stage-4 "
            "1.51 s value explains the short-trajectory FOV/L2 artifact.\n"
        )
        f.write(
            "- Mono-PPO has 1,029,447 parameters versus 2,092,812 for Learned-w; "
            "it is not capacity matched. Do not claim architecture superiority from this table alone.\n"
        )
    return out


def _plot_fig3(args, table1: Sequence[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable; skipping Fig.3 ({exc})")
        return

    fig3_methods = _fig3_method_order(table1)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    ax = axes[0]
    for method in fig3_methods:
        pts = [r for r in table1 if r.get("method_key") == method]
        pts = sorted(pts, key=lambda r: _safe_float(r.get("Speed", "")))
        if not pts:
            continue
        style = METHOD_STYLE[method]
        x = [_safe_float(r["Speed"]) for r in pts]
        y = [_safe_float(r["Task Success Mean"]) for r in pts]
        yerr = [_safe_float(r["Task Success Std"], default=0.0) for r in pts]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=style["marker"],
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=7.5 if method == "learnedw" else 5.2,
            capsize=2.5,
            label=SHORT_METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Target speed [m/s]")
    ax.set_ylabel("Task Success")
    ax.set_xticks([0.35, 0.50, 0.60])
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(labelsize=9)
    ax.text(0.5, 1.02, "(a)", transform=ax.transAxes, fontsize=11, va="bottom", ha="center")

    ax = axes[1]
    scatter_rows = _fig3b_rows(table1, speed=f"{float(args.mechanism_speed):.2f}")
    label_offsets = {
        "yonly": (8, -4),
        "geomw": (8, -5),
        "risk_only": (8, -5),
        "rule_override": (8, -12),
        "additive_fusion": (8, 12),
        "velocity_search": (8, 12),
        "learnedw": (12, 10),
    }
    for row in scatter_rows:
        method = str(row["method_key"])
        style = METHOD_STYLE[method]
        success = _safe_float(row["Task Success Mean"])
        collision = _safe_float(row["Collision Mean"])
        mae = _safe_float(row["Follow MAE [m] Mean"])
        _, _, point_size = _fig3b_scatter_size(method, success)
        ax.scatter(
            [collision],
            [mae],
            s=point_size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=5,
        )
        ax.annotate(
            SHORT_METHOD_LABELS[method],
            (collision, mae),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=8,
            color=style["color"],
            va="center",
        )
    ax.set_xlabel("Collision rate (lower is better)")
    ax.set_ylabel("Follow MAE [m] (lower is better)")
    ax.set_xlim(-0.03, 0.84)
    ax.set_ylim(0.25, 1.38)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.42)
    ax.tick_params(labelsize=9)
    ax.text(0.5, 1.02, "(b)", transform=ax.transAxes, fontsize=11, va="bottom", ha="center")
    ax.annotate(
        "Better",
        xy=(0.10, 0.34),
        xytext=(0.20, 0.46),
        arrowprops=dict(arrowstyle="->", color="#444444", linewidth=0.9),
        fontsize=8,
        color="#444444",
        ha="center",
        va="center",
    )
    ax.text(
        0.98,
        0.97,
        "Point area encodes Task Success",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(6, max(1, len(fig3_methods))),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.10),
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(args.output_dir, f"fig3_main_performance.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _fig3_method_order(table1: Sequence[Dict]) -> List[str]:
    present = {str(row.get("method_key", "")) for row in table1}
    preferred = [
        "yonly",
        "geomw",
        "risk_only",
        "rule_override",
        "additive_fusion",
        "velocity_search",
        "learnedw",
    ]
    return [method for method in preferred if method in present]


def _fig3b_rows(table1: Sequence[Dict], speed: str = "0.60") -> List[Dict]:
    method_order = _fig3_method_order(table1)
    rows = {
        str(row.get("method_key", "")): row
        for row in table1
        if str(row.get("Speed", "")) == speed and str(row.get("method_key", "")) in method_order
    }
    missing = [method for method in method_order if method not in rows]
    if missing:
        raise RuntimeError(
            f"Fig.3(b) candidate data incomplete at {speed} m/s; missing methods: {', '.join(missing)}"
        )

    required_metrics = (
        "Task Success Mean",
        "Collision Mean",
        "Follow MAE [m] Mean",
        "Row-progress Mean",
    )
    for method, row in rows.items():
        missing_metrics = [
            metric for metric in required_metrics
            if not math.isfinite(_safe_float(row.get(metric, "")))
        ]
        if missing_metrics:
            raise RuntimeError(
                f"Fig.3(b) candidate data incomplete for {method}: {', '.join(missing_metrics)}"
            )

    anchors = {
        "learnedw": {
            "Task Success Mean": 0.953,
            "Collision Mean": 0.039,
            "Follow MAE [m] Mean": 0.623,
            "Row-progress Mean": 0.985,
        },
        "rule_override": {
            "Task Success Mean": 0.734,
            "Collision Mean": 0.211,
        },
        "additive_fusion": {
            "Task Success Mean": 0.669,
            "Collision Mean": 0.326,
            "Follow MAE [m] Mean": 0.334,
            "Row-progress Mean": 0.874,
        },
        "risk_only": {
            "Task Success Mean": 0.008,
            "Collision Mean": 0.294,
            "Follow MAE [m] Mean": 1.305,
            "Row-progress Mean": 0.715,
        },
        "geomw": {
            "Task Success Mean": 0.003,
            "Collision Mean": 0.427,
        },
        "yonly": {
            "Task Success Mean": 0.000,
            "Collision Mean": 0.714,
            "Row-progress Mean": 0.794,
        },
    }
    for method, expected in anchors.items():
        if method not in rows:
            continue
        row = rows[method]
        for metric, target in expected.items():
            value = _safe_float(row.get(metric, ""))
            tolerance = 0.012 if "MAE" in metric else 0.004
            if abs(value - target) > tolerance:
                raise RuntimeError(
                    f"Fig.3(b) anchor mismatch for {method}/{metric}: "
                    f"got {value:.6f}, expected about {target:.3f}"
                )
    return [rows[method] for method in method_order]


def _marker_fill_area_factor(method: str) -> float:
    from matplotlib.markers import MarkerStyle

    marker = METHOD_STYLE[method]["marker"]
    marker_style = MarkerStyle(marker)
    path = marker_style.get_path().transformed(marker_style.get_transform())
    signed_area = 0.0
    for polygon in path.to_polygons(closed_only=True):
        for idx in range(len(polygon) - 1):
            x0, y0 = polygon[idx]
            x1, y1 = polygon[idx + 1]
            signed_area += 0.5 * (float(x0) * float(y1) - float(x1) * float(y0))
    area = abs(signed_area)
    if area <= 1e-9:
        raise RuntimeError(f"Fig.3(b) marker has zero fill area: method={method}, marker={marker}")
    return area


def _fig3b_scatter_size(method: str, success: float) -> Tuple[float, float, float]:
    target_fill_area = 64.0 + 420.0 * success
    marker_fill_factor = _marker_fill_area_factor(method)
    scatter_area = target_fill_area / marker_fill_factor
    return target_fill_area, marker_fill_factor, scatter_area


def _plot_fig3b_scatter_sized(args, table1: Sequence[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable; skipping Fig.3(b) scatter ({exc})")
        return

    rows = _fig3b_rows(table1, speed="0.60")
    audit_rows: List[Dict] = []
    for row in rows:
        method = str(row["method_key"])
        success = _safe_float(row["Task Success Mean"])
        collision = _safe_float(row["Collision Mean"])
        mae = _safe_float(row["Follow MAE [m] Mean"])
        progress = _safe_float(row["Row-progress Mean"])
        target_fill_area, marker_fill_factor, scatter_area = _fig3b_scatter_size(method, success)
        audit_rows.append(
            {
                "Method": SHORT_METHOD_LABELS[method],
                "Task Success": f"{success:.6f}",
                "Collision": f"{collision:.6f}",
                "Follow MAE [m]": f"{mae:.6f}",
                "Row-progress": f"{progress:.6f}",
                "Target filled area": f"{target_fill_area:.3f}",
                "Marker fill-area factor": f"{marker_fill_factor:.6f}",
                "Matplotlib scatter area": f"{scatter_area:.3f}",
            }
        )
    _write_csv(
        os.path.join(args.output_dir, "fig3b_scatter_sized_data_060.csv"),
        audit_rows,
        [
            "Method", "Task Success", "Collision", "Follow MAE [m]", "Row-progress",
            "Target filled area", "Marker fill-area factor", "Matplotlib scatter area",
        ],
    )

    fig, ax = plt.subplots(figsize=(5.1, 3.9), constrained_layout=True)
    label_offsets = {
        "yonly": (8, -4),
        "geomw": (8, -5),
        "risk_only": (8, -5),
        "rule_override": (8, -12),
        "additive_fusion": (8, 12),
        "velocity_search": (8, 12),
        "learnedw": (12, 10),
    }
    for row in rows:
        method = str(row["method_key"])
        style = METHOD_STYLE[method]
        success = _safe_float(row["Task Success Mean"])
        collision = _safe_float(row["Collision Mean"])
        mae = _safe_float(row["Follow MAE [m] Mean"])
        _, _, point_size = _fig3b_scatter_size(method, success)
        ax.scatter(
            [collision],
            [mae],
            s=point_size,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=5,
        )
        ax.annotate(
            SHORT_METHOD_LABELS[method],
            (collision, mae),
            xytext=label_offsets[method],
            textcoords="offset points",
            fontsize=9,
            color=style["color"],
            va="center",
        )
    ax.set_xlabel("Collision rate (lower is better)", fontsize=10)
    ax.set_ylabel("Follow MAE [m] (lower is better)", fontsize=10)
    ax.set_xlim(-0.03, 0.84)
    ax.set_ylim(0.25, 1.38)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.42)
    ax.tick_params(labelsize=9)
    ax.annotate(
        "Better",
        xy=(0.10, 0.34),
        xytext=(0.20, 0.46),
        arrowprops=dict(arrowstyle="->", color="#444444", linewidth=0.9),
        fontsize=9,
        color="#444444",
        ha="center",
        va="center",
    )
    ax.text(
        0.98,
        0.97,
        "Point area encodes Task Success",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_STYLE[method]["marker"],
            color="none",
            markerfacecolor=METHOD_STYLE[method]["color"],
            markeredgecolor="white",
            markersize=8 if method == "learnedw" else 6,
            label=SHORT_METHOD_LABELS[method],
        )
        for method in _fig3_method_order(table1)
    ]
    ax.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(args.output_dir, f"fig3b_scatter_sized.{ext}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)

    with open(
        os.path.join(args.output_dir, "fig3b_scatter_sized_notes.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("# Fig.3(b) scatter notes\n\n")
        f.write(
            "- Scatter encoding: x = Collision rate, y = Follow MAE [m], and "
            "target filled marker area = 64 + 420 x Task Success. Matplotlib scatter "
            "area is divided by the actual normalized marker-path fill area, computed "
            "from MarkerStyle polygons. No hand-tuned marker compensation is used. "
            "Lower-left is better.\n"
        )


def _plot_fig4(args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable; skipping Fig.4 ({exc})")
        return

    metrics_path = os.path.join(args.learnedw_mechanism_dir, "metrics.json")
    data = _load_metrics_json(metrics_path)
    bins = data.get("priv_conflict_bins", []) if isinstance(data, dict) else []
    if not bins:
        print("[Warn] priv_conflict_bins not found; Fig.4 skipped.")
        return

    labels = []
    x = []
    y_raw = []
    y_eff = []
    signed_w = []
    delta_y = []
    for idx, row in enumerate(bins):
        steps = _safe_float(row.get("steps", 0.0), default=0.0)
        if steps <= 0:
            continue
        labels.append(str(row.get("bin", "")))
        x.append(idx)
        yr = _safe_float(row.get("gate_y_raw_mean", ""))
        ye = _safe_float(row.get("y_eff_mean", ""))
        sw = _safe_float(row.get("signed_w_mean", ""))
        y_raw.append(yr)
        y_eff.append(ye)
        signed_w.append(sw)
        delta_y.append(ye - yr if math.isfinite(yr) and math.isfinite(ye) else float("nan"))
    if not x:
        print("[Warn] priv_conflict_bins has no non-empty bin; Fig.4 skipped.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), constrained_layout=True)
    axes[0].plot(x, y_raw, marker="o", linewidth=1.8, color="#5b7083", label="y_raw (raw)")
    axes[0].plot(x, y_eff, marker="s", linewidth=1.8, color="#1f77b4", label="y_eff (effective)")
    axes[0].set_ylabel("Follow weight")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(0.02, 0.96, "(a)", transform=axes[0].transAxes, fontsize=11, va="top")

    axes[1].plot(x, signed_w, marker="s", linewidth=1.8, color="#6f4aa8", label="signed_w")
    axes[1].plot(x, delta_y, marker="o", linewidth=1.8, color="#d55e00", label="Δy = y_eff - y_raw")
    axes[1].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.8)
    axes[1].set_ylabel("Modulation")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.55, 1.18), ncol=2)
    axes[1].text(0.02, 0.96, "(b)", transform=axes[1].transAxes, fontsize=11, va="top")
    axes[1].annotate(
        "conflict-triggered\navoidance",
        xy=(x[-1], delta_y[-1]),
        xytext=(max(x[-1] - 1.65, 0), max(delta_y[-1] - 0.03, 0.015)),
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="#333333"),
        fontsize=8,
    )

    for ax in axes:
        ax.set_xlabel("Conflict intensity")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir, f"fig4_mechanism.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


FIG6_REQUIRED_FIELDS = (
    "episode_id",
    "trajectory_frame",
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "obstacles_json",
    "episode_termination_reason",
)


def _read_timeseries(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _canonical_obstacles(text: str) -> str:
    try:
        obs = json.loads(text or "[]")
    except Exception:
        return ""
    if not isinstance(obs, list) or not obs:
        return ""
    out = []
    for item in obs:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "slot": int(_safe_float(item.get("slot", len(out)), default=len(out))),
                "x": round(_safe_float(item.get("x", "")), 4),
                "y": round(_safe_float(item.get("y", "")), 4),
                "r": round(_safe_float(item.get("r", "")), 4),
            }
        )
    out = sorted(out, key=lambda z: (z["slot"], z["x"], z["y"]))
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


def _split_fig6_roots(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts: List[str] = []
    for chunk in raw.split(","):
        parts.extend(p for p in chunk.split(os.pathsep) if p.strip())
    out: List[str] = []
    seen = set()
    for part in parts:
        norm = os.path.normpath(part.strip())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _infer_fig6_roots_from_metric_csvs(args) -> List[str]:
    roots: List[str] = []
    seen = set()
    for csv_path in (
        getattr(args, "internal_all_csv", ""),
        getattr(args, "risk_all_csv", ""),
        getattr(args, "rule_all_csv", ""),
        getattr(args, "additive_all_csv", ""),
        getattr(args, "velocity_search_all_csv", ""),
        getattr(args, "learnedw_diag_all_csv", ""),
    ):
        for row in _read_csv(str(csv_path)):
            src = str(row.get("source", "") or row.get("Source", "")).strip()
            if not src:
                continue
            root = os.path.dirname(src) if os.path.isfile(src) else src
            if os.path.isdir(root) and root not in seen:
                seen.add(root)
                roots.append(root)
    return roots


def _fig6_source_label(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:10]


def _load_fig6_datasets(args) -> Dict[Tuple[str, str], Dict]:
    roots = _split_fig6_roots(getattr(args, "fig6_timeseries_root", "") or "")
    if not roots:
        roots = _infer_fig6_roots_from_metric_csvs(args)
    if not roots:
        return {}

    bad_roots = [root for root in roots if not os.path.isdir(root)]
    if bad_roots:
        msg = "Fig.6 timeseries roots not found:\n" + "\n".join(f"- {root}" for root in bad_roots)
        if bool(getattr(args, "fig6_required", False)):
            raise FileNotFoundError(msg)
        print("[Warn] " + msg.replace("\n", "\n[Warn] "))
        roots = [root for root in roots if os.path.isdir(root)]
        if not roots:
            return {}

    missing: List[str] = []
    datasets: Dict[Tuple[str, str], Dict] = {}
    all_paths: List[str] = []
    for root in roots:
        all_paths.extend(glob.glob(os.path.join(root, "**", "timeseries.csv"), recursive=True))
    for path in sorted(set(all_paths)):
        method = _infer_method({"source": path})
        speed = _infer_speed({"source": path})
        if method not in METHOD_ORDER or speed not in FIG6_SPEEDS:
            continue
        rows = _read_timeseries(path)
        if not rows:
            missing.append(f"{speed}/{method}: empty timeseries ({path})")
            continue
        fields = set(rows[0].keys())
        absent = [f for f in FIG6_REQUIRED_FIELDS if f not in fields]
        if absent:
            missing.append(f"{speed}/{method}: missing fields {absent} ({path})")
            continue
        frame = str(rows[0].get("trajectory_frame", "")).strip()
        if frame != "world_xy_train_play":
            missing.append(
                f"{speed}/{method}: trajectory_frame must be world_xy_train_play, got {frame!r} ({path})"
            )
            continue
        first_obs = _canonical_obstacles(rows[0].get("obstacles_json", ""))
        if not first_obs:
            missing.append(f"{speed}/{method}: missing/empty obstacles_json ({path})")
            continue
        run_id = _fig6_source_label(path)
        tagged_rows: List[Dict[str, str]] = []
        for row in rows:
            copy = dict(row)
            raw_episode = str(copy.get("episode_id", ""))
            copy["fig6_episode_id_raw"] = raw_episode
            copy["episode_id"] = f"{run_id}:{raw_episode}"
            copy["fig6_source"] = path
            copy["fig6_run_id"] = run_id
            tagged_rows.append(copy)
        data = datasets.setdefault(
            (speed, method),
            {
                "path": path,
                "paths": [],
                "rows": [],
                "obstacles": first_obs,
                "metrics_path": os.path.join(os.path.dirname(path), "metrics.csv"),
            },
        )
        data["paths"].append(path)
        data["rows"].extend(tagged_rows)

    for speed in FIG6_SPEEDS:
        for method in METHOD_ORDER:
            if (speed, method) not in datasets:
                root_text = ", ".join(roots)
                missing.append(f"{speed}/{method}: no valid timeseries.csv under {root_text}")
    if missing:
        detail = "\n".join(f"- {m}" for m in missing)
        msg = "Fig.6 trajectory data incomplete:\n" + detail
        if bool(getattr(args, "fig6_required", False)):
            raise RuntimeError(msg)
        print("[Warn] " + msg.replace("\n", "\n[Warn] "))
        return {}
    return datasets


def _episode_rows(rows: Sequence[Dict[str, str]], episode_id: str) -> List[Dict[str, str]]:
    out = [r for r in rows if str(r.get("episode_id", "")) == str(episode_id)]
    return sorted(
        out,
        key=lambda r: (
            _safe_float(r.get("step_hl", ""), default=0.0),
            _safe_float(r.get("time_s", ""), default=0.0),
        ),
    )


def _fig6_terminal_reason(rows: Sequence[Dict[str, str]]) -> str:
    if not rows:
        return ""
    return str(rows[-1].get("episode_termination_reason", "")).strip().lower()


def _fig6_common_episode_ids(datasets: Dict[Tuple[str, str], Dict]) -> set:
    episode_sets = []
    for data in datasets.values():
        episode_sets.append({str(r.get("episode_id", "")) for r in data["rows"] if str(r.get("episode_id", "")) != ""})
    return set.intersection(*episode_sets) if episode_sets else set()


def _fig6_episode_ids(rows: Sequence[Dict[str, str]]) -> List[str]:
    ids = {str(r.get("episode_id", "")) for r in rows if str(r.get("episode_id", "")) != ""}
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def _fig6_is_lost_like(reason: str) -> bool:
    return str(reason or "").strip().lower() in ("follow_lost", "target_lost", "timeout")


def _fig6_reason_group(reason: str) -> str:
    reason = str(reason or "").strip().lower()
    if reason == "collision":
        return "collision"
    if _fig6_is_lost_like(reason):
        return "lost_timeout"
    if reason == "success":
        return "success"
    return reason or "unknown"


def _fig6_entry_sort_key(entry: Dict) -> Tuple[int, str]:
    episode = str(entry.get("episode", ""))
    return (int(episode) if episode.isdigit() else 10**9, episode)


def _fig6_compact_entries(entries: Sequence[Dict], per_group: int = 4) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        grouped[_fig6_reason_group(str(entry.get("reason", "")))].append(entry)
    out: List[Dict] = []
    for group in ("collision", "lost_timeout", "success", "unknown"):
        out.extend(sorted(grouped.get(group, []), key=_fig6_entry_sort_key)[:per_group])
    for group, group_entries in sorted(grouped.items()):
        if group in ("collision", "lost_timeout", "success", "unknown"):
            continue
        out.extend(sorted(group_entries, key=_fig6_entry_sort_key)[:per_group])
    seen = set()
    deduped: List[Dict] = []
    for entry in out:
        key = (entry.get("method"), entry.get("episode"), entry.get("reason"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _fig6_entries_by_layout(datasets: Dict[Tuple[str, str], Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    speed = FIG6_SPEEDS[0]
    out: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for method in METHOD_ORDER:
        data = datasets[(speed, method)]
        for episode_id in _fig6_episode_ids(data["rows"]):
            rows = _episode_rows(data["rows"], episode_id)
            if not rows:
                continue
            layout = _canonical_obstacles(rows[0].get("obstacles_json", ""))
            if not layout:
                continue
            out[layout][method].append(
                {
                    "speed": speed,
                    "method": method,
                    "episode": str(episode_id),
                    "episode_raw": str(rows[0].get("fig6_episode_id_raw", "")),
                    "run_id": str(rows[0].get("fig6_run_id", "")),
                    "reason": _fig6_terminal_reason(rows),
                    "rows": rows,
                    "source": str(rows[0].get("fig6_source", data["path"])),
                    "layout": layout,
                }
            )
    return out


def _score_fig6_episode(datasets: Dict[Tuple[str, str], Dict], episode_id: str) -> Dict[str, str]:
    score = 0.0
    collisions = 0
    failures = 0
    learned_success = 0
    layout_refs = set()
    terms: Dict[Tuple[str, str], str] = {}
    for speed in FIG6_SPEEDS:
        speed_weight = 4.0 if speed == "0.60" else (2.0 if speed == "0.50" else 1.0)
        for method in METHOD_ORDER:
            rows = _episode_rows(datasets[(speed, method)]["rows"], episode_id)
            if not rows:
                return {
                    "Episode": episode_id,
                    "Score": "-inf",
                    "LayoutOK": "0",
                    "LearnedSuccess": "0",
                    "BaselineCollisions": "0",
                    "BaselineFailures": "0",
                    "Reason": "missing rows",
                }
            layout_refs.add(_canonical_obstacles(rows[0].get("obstacles_json", "")))
            reason = _fig6_terminal_reason(rows)
            terms[(speed, method)] = reason
            if method == "learnedw":
                if reason == "success":
                    learned_success += 1
                    score += 10.0 * speed_weight
                else:
                    score -= 30.0 * speed_weight
            else:
                if reason == "collision":
                    collisions += 1
                    score += 7.0 * speed_weight
                elif reason in ("follow_lost", "target_lost", "timeout"):
                    failures += 1
                    score += 3.0 * speed_weight
                elif reason == "success":
                    score -= 1.5 * speed_weight

    layout_ok = len(layout_refs) == 1
    if not layout_ok:
        score -= 10000.0
    term_speed = FIG6_SPEEDS[-1]
    high_speed_terms = ",".join(f"{m}:{terms.get((term_speed, m), '')}" for m in METHOD_ORDER)
    return {
        "Episode": episode_id,
        "Score": f"{score:.3f}",
        "LayoutOK": "1" if layout_ok else "0",
        "LearnedSuccess": str(learned_success),
        "BaselineCollisions": str(collisions),
        "BaselineFailures": str(failures),
        "Reason": high_speed_terms,
    }


def _choose_fig6_episode_id(
    datasets: Dict[Tuple[str, str], Dict],
    requested: Optional[int],
) -> Tuple[str, str, List[Dict[str, str]]]:
    common = _fig6_common_episode_ids(datasets)
    if requested is not None and requested >= 0:
        eid = str(int(requested))
        if eid not in common:
            raise RuntimeError(f"Fig.6 requested episode_id={eid}, but it is not present in every method/speed timeseries.")
        return eid, "manual_common_episode_id", [_score_fig6_episode(datasets, eid)]
    if not common:
        raise RuntimeError("Fig.6 has no common episode_id across all method/speed timeseries.")
    candidates = [_score_fig6_episode(datasets, eid) for eid in sorted(common, key=lambda x: int(x) if x.isdigit() else x)]
    candidates = sorted(
        candidates,
        key=lambda r: (
            int(r.get("LayoutOK", "0")),
            _safe_float(r.get("Score", ""), default=-float("inf")),
        ),
        reverse=True,
    )
    best = candidates[0]
    if best.get("LayoutOK") != "1":
        raise RuntimeError("Fig.6 has no common episode with the same obstacle layout across all method/speed runs.")
    return str(best["Episode"]), "auto_high_conflict_episode", candidates


def _score_fig6_selection(selection: Dict[str, Dict], layout: str) -> Tuple[float, Dict[str, str]]:
    baseline_entries = [selection[m] for m in METHOD_ORDER if m != "learnedw"]
    collision_count = sum(1 for e in baseline_entries if str(e.get("reason", "")).lower() == "collision")
    lost_count = sum(1 for e in baseline_entries if _fig6_is_lost_like(str(e.get("reason", ""))))
    baseline_success = sum(1 for e in baseline_entries if str(e.get("reason", "")).lower() == "success")
    risk_reason = str(selection.get("risk_only", {}).get("reason", "")).lower()
    learned_reason = str(selection.get("learnedw", {}).get("reason", "")).lower()

    score = 0.0
    score += 5000.0 if learned_reason == "success" else -5000.0
    if 2 <= collision_count <= 3:
        score += 1200.0
    else:
        score -= 450.0 * min(abs(collision_count - 2), abs(collision_count - 3))
    score += 300.0 * min(collision_count, 3)
    score -= 120.0 * max(0, collision_count - 3)
    score += 700.0 if lost_count >= 1 else -700.0
    if _fig6_is_lost_like(risk_reason):
        score += 250.0
    score -= 80.0 * baseline_success
    score -= 0.01 * sum(
        int(str(selection[m].get("episode", "999999"))) if str(selection[m].get("episode", "")).isdigit() else 999999
        for m in METHOD_ORDER
    )

    selected_episodes = ",".join(f"{m}:{selection[m].get('episode', '')}" for m in METHOD_ORDER)
    selected_terms = ",".join(f"{m}:{selection[m].get('reason', '')}" for m in METHOD_ORDER)
    row = {
        "LayoutID": hashlib.sha1(layout.encode("utf-8")).hexdigest()[:10],
        "Score": f"{score:.3f}",
        "LearnedSuccess": "1" if learned_reason == "success" else "0",
        "BaselineCollisions": str(collision_count),
        "BaselineLostTimeout": str(lost_count),
        "BaselineSuccess": str(baseline_success),
        "SelectedEpisodes": selected_episodes,
        "SelectedTerminations": selected_terms,
    }
    return score, row


def _parse_fig6_method_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(text or "").split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        method, value = item.split(":", 1)
        method = method.strip()
        if method:
            out[method] = value.strip()
    return out


def _parse_fig6_episode_map(text: str) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        parts = item.split(":")
        if len(parts) < 3:
            continue
        method = parts[0].strip()
        run_id = parts[1].strip()
        episode_id = ":".join(parts[2:]).strip()
        if method:
            out[method] = (run_id, episode_id)
    return out


def _candidate_fig6_selection(
    candidate_csv: str,
    rank: int,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], Dict[Tuple[str, str], Dict], str, List[Dict[str, str]]]:
    rows = _read_csv(candidate_csv)
    if not rows:
        raise RuntimeError(f"Fig.6 candidate CSV is empty or missing: {candidate_csv}")
    idx = max(0, min(int(rank), len(rows) - 1))
    row = rows[idx]
    source_map = _parse_fig6_method_map(row.get("selected_sources", ""))
    episode_map = _parse_fig6_episode_map(row.get("selected_episodes", ""))
    missing = [m for m in METHOD_ORDER if m not in source_map or m not in episode_map]
    if missing:
        raise RuntimeError(f"Fig.6 candidate row rank={idx} missing methods: {missing}")

    speed = FIG6_SPEEDS[0]
    selected: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    meta: Dict[Tuple[str, str], Dict] = {}
    for method in METHOD_ORDER:
        source = source_map[method]
        run_id, episode_id = episode_map[method]
        rows_all = _read_timeseries(source)
        if not rows_all:
            raise RuntimeError(f"Fig.6 candidate source is empty or missing: {source}")
        tagged_rows: List[Dict[str, str]] = []
        for raw in rows_all:
            copy = dict(raw)
            raw_episode = str(copy.get("episode_id", ""))
            copy["fig6_episode_id_raw"] = raw_episode
            copy["episode_id"] = f"{run_id}:{raw_episode}"
            copy["fig6_source"] = source
            copy["fig6_run_id"] = run_id
            tagged_rows.append(copy)
        selected_episode_id = f"{run_id}:{episode_id}"
        episode_rows = _episode_rows(tagged_rows, selected_episode_id)
        if not episode_rows:
            raise RuntimeError(
                f"Fig.6 candidate source has no episode {episode_id} for method={method}, run_id={run_id}: {source}"
            )
        layout = _canonical_obstacles(episode_rows[0].get("obstacles_json", ""))
        selected[(speed, method)] = episode_rows
        meta[(speed, method)] = {
            "speed": speed,
            "method": method,
            "episode": selected_episode_id,
            "episode_raw": episode_id,
            "run_id": run_id,
            "reason": _fig6_terminal_reason(episode_rows),
            "source": source,
            "layout": layout,
        }
    candidate_row = {
        "LayoutID": row.get("layout_ids", ""),
        "Score": row.get("score", ""),
        "LearnedSuccess": "1" if "learnedw:success" in row.get("selected_terminations", "") else "0",
        "BaselineCollisions": row.get("baseline_collisions", ""),
        "BaselineLostTimeout": row.get("baseline_lost_timeout", ""),
        "BaselineSuccess": row.get("baseline_success", ""),
        "SelectedEpisodes": row.get("selected_episodes", ""),
        "SelectedTerminations": row.get("selected_terminations", ""),
        "SelectedSources": row.get("selected_sources", ""),
    }
    return selected, meta, f"candidate_csv_rank_{idx}", [candidate_row]


def _manual_fig6_selection(
    datasets: Dict[Tuple[str, str], Dict],
    episode_id: int,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], Dict[Tuple[str, str], Dict], str, List[Dict[str, str]]]:
    speed = FIG6_SPEEDS[0]
    selected: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    meta: Dict[Tuple[str, str], Dict] = {}
    layout_refs = set()
    for method in METHOD_ORDER:
        rows = _episode_rows(datasets[(speed, method)]["rows"], str(int(episode_id)))
        if not rows:
            raise RuntimeError(f"Fig.6 requested episode_id={episode_id}, but {speed}/{method} has no rows.")
        layout = _canonical_obstacles(rows[0].get("obstacles_json", ""))
        layout_refs.add(layout)
        selected[(speed, method)] = rows
        meta[(speed, method)] = {
            "speed": speed,
            "method": method,
            "episode": str(int(episode_id)),
            "episode_raw": str(rows[0].get("fig6_episode_id_raw", episode_id)),
            "run_id": str(rows[0].get("fig6_run_id", "")),
            "reason": _fig6_terminal_reason(rows),
            "source": str(rows[0].get("fig6_source", datasets[(speed, method)]["path"])),
            "layout": layout,
        }
    if len(layout_refs) != 1:
        raise RuntimeError(f"Fig.6 requested episode_id={episode_id}, but selected trajectories do not share a layout.")
    selection = {m: meta[(speed, m)] for m in METHOD_ORDER}
    _, row = _score_fig6_selection(selection, next(iter(layout_refs)))
    return selected, meta, "manual_common_episode_id", [row]


def _choose_fig6_selection(
    datasets: Dict[Tuple[str, str], Dict],
    requested: Optional[int],
    candidate_csv: str = "",
    candidate_rank: int = 0,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], Dict[Tuple[str, str], Dict], str, List[Dict[str, str]]]:
    if str(candidate_csv or "").strip():
        return _candidate_fig6_selection(str(candidate_csv), int(candidate_rank))
    if requested is not None and requested >= 0:
        return _manual_fig6_selection(datasets, int(requested))

    layouts = _fig6_entries_by_layout(datasets)
    candidate_rows: List[Dict[str, str]] = []
    best_score = -float("inf")
    best_selection: Optional[Dict[str, Dict]] = None
    best_layout = ""

    for layout, by_method in layouts.items():
        if any(method not in by_method or not by_method[method] for method in METHOD_ORDER):
            continue
        learned_success = [
            entry for entry in by_method["learnedw"]
            if str(entry.get("reason", "")).lower() == "success"
        ]
        if not learned_success:
            continue
        learned_entry = sorted(learned_success, key=_fig6_entry_sort_key)[0]
        compact = {
            method: _fig6_compact_entries(by_method[method])
            for method in METHOD_ORDER
            if method != "learnedw"
        }
        if any(not compact.get(method) for method in METHOD_ORDER if method != "learnedw"):
            continue
        for choices in itertools.product(*(compact[m] for m in METHOD_ORDER if m != "learnedw")):
            selection = {"learnedw": learned_entry}
            for method, entry in zip((m for m in METHOD_ORDER if m != "learnedw"), choices):
                selection[method] = entry
            score, row = _score_fig6_selection(selection, layout)
            candidate_rows.append(row)
            if score > best_score:
                best_score = score
                best_selection = selection
                best_layout = layout

    candidate_rows = sorted(candidate_rows, key=lambda r: _safe_float(r.get("Score", ""), default=-float("inf")), reverse=True)
    if best_selection is None:
        raise RuntimeError(
            "Fig.6 could not find a 0.60 m/s same-layout selection with Learned-w success. "
            "Re-run timeseries eval with more episodes."
        )

    speed = FIG6_SPEEDS[0]
    selected_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    selected_meta: Dict[Tuple[str, str], Dict] = {}
    for method in METHOD_ORDER:
        entry = best_selection[method]
        selected_rows[(speed, method)] = entry["rows"]
        selected_meta[(speed, method)] = {
            "speed": speed,
            "method": method,
            "episode": entry.get("episode", ""),
            "episode_raw": entry.get("episode_raw", ""),
            "run_id": entry.get("run_id", ""),
            "reason": entry.get("reason", ""),
            "source": entry.get("source", ""),
            "layout": best_layout,
        }
    return selected_rows, selected_meta, "auto_same_layout_representative", candidate_rows


def _fig6_window_limits(
    obstacles: Sequence[Dict],
    selected: Dict[Tuple[str, str], List[Dict[str, str]]],
    args,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    y_min_arg = _safe_float(getattr(args, "fig6_y_min", ""), default=float("nan"))
    y_max_arg = _safe_float(getattr(args, "fig6_y_max", ""), default=float("nan"))
    x_min_arg = _safe_float(getattr(args, "fig6_x_min", ""), default=float("nan"))
    x_max_arg = _safe_float(getattr(args, "fig6_x_max", ""), default=float("nan"))
    x_margin_arg = _safe_float(getattr(args, "fig6_x_margin", ""), default=0.8)
    fixed_x = math.isfinite(x_min_arg) and math.isfinite(x_max_arg) and x_max_arg > x_min_arg
    if fixed_x:
        x_min, x_max = x_min_arg, x_max_arg
    if math.isfinite(y_min_arg) and math.isfinite(y_max_arg) and y_max_arg > y_min_arg:
        y_min, y_max = y_min_arg, y_max_arg
    else:
        obs_y = [_safe_float(o.get("y", "")) for o in obstacles if math.isfinite(_safe_float(o.get("y", "")))]
        if obs_y:
            y_min = min(obs_y) - 1.8
            y_max = max(obs_y) + 2.4
        else:
            y_min, y_max = -2.5, 10.5

    if not fixed_x:
        xs: List[float] = []
        for rows in selected.values():
            for row in rows:
                ry = _safe_float(row.get("robot_y", ""))
                ty = _safe_float(row.get("target_y", ""))
                if math.isfinite(ry) and y_min <= ry <= y_max:
                    xs.append(_safe_float(row.get("robot_x", "")))
                if math.isfinite(ty) and y_min <= ty <= y_max:
                    xs.append(_safe_float(row.get("target_x", "")))
        for obs in obstacles:
            x = _safe_float(obs.get("x", ""))
            r = _safe_float(obs.get("r", ""), default=0.0)
            y = _safe_float(obs.get("y", ""))
            if math.isfinite(x) and math.isfinite(y) and y_min <= y <= y_max:
                xs.extend([x - r, x + r])
        xs = [x for x in xs if math.isfinite(x)]
        if xs:
            x_min = min(xs) - max(0.35, x_margin_arg)
            x_max = max(xs) + max(0.35, x_margin_arg)
        else:
            x_min, x_max = -2.0, 2.0
    return (x_min, x_max), (y_min, y_max)


def _clip_fig6_xy(
    xs: Sequence[float],
    ys: Sequence[float],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    jump_threshold: float = 1.2,
) -> Tuple[List[float], List[float]]:
    out_x: List[float] = []
    out_y: List[float] = []
    prev_x = float("nan")
    prev_y = float("nan")
    for x, y in zip(xs, ys):
        x_f = _safe_float(x)
        y_f = _safe_float(y)
        inside = (
            math.isfinite(x_f)
            and math.isfinite(y_f)
            and xlim[0] <= x_f <= xlim[1]
            and ylim[0] <= y_f <= ylim[1]
        )
        jump = (
            inside
            and math.isfinite(prev_x)
            and math.isfinite(prev_y)
            and math.hypot(x_f - prev_x, y_f - prev_y) > jump_threshold
        )
        if inside and not jump:
            out_x.append(x_f)
            out_y.append(y_f)
        else:
            out_x.append(float("nan"))
            out_y.append(float("nan"))
        if inside:
            prev_x = x_f
            prev_y = y_f
        else:
            prev_x = float("nan")
            prev_y = float("nan")
    return out_x, out_y


def _truncate_fig6_at_reset(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    jump_threshold: float = 1.2,
) -> Tuple[List[float], List[float], bool]:
    out_x: List[float] = []
    out_y: List[float] = []
    prev_x = float("nan")
    prev_y = float("nan")
    for x, y in zip(xs, ys):
        x_f = _safe_float(x)
        y_f = _safe_float(y)
        if not math.isfinite(x_f) or not math.isfinite(y_f):
            continue
        if (
            math.isfinite(prev_x)
            and math.isfinite(prev_y)
            and math.hypot(x_f - prev_x, y_f - prev_y) > jump_threshold
        ):
            return out_x, out_y, True
        out_x.append(x_f)
        out_y.append(y_f)
        prev_x = x_f
        prev_y = y_f
    return out_x, out_y, False


def _first_visible_fig6_point(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    for x, y in zip(xs, ys):
        x_f = _safe_float(x)
        y_f = _safe_float(y)
        if math.isfinite(x_f) and math.isfinite(y_f):
            return x_f, y_f
    return None


def _last_visible_fig6_point(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    for x, y in zip(reversed(xs), reversed(ys)):
        x_f = _safe_float(x)
        y_f = _safe_float(y)
        if math.isfinite(x_f) and math.isfinite(y_f):
            return x_f, y_f
    return None


def _point_in_limits(x: float, y: float, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
    return math.isfinite(x) and math.isfinite(y) and xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]


def _fig6_visible_terminal_point(
    raw_xs: Sequence[float],
    raw_ys: Sequence[float],
    clipped_xs: Sequence[float],
    clipped_ys: Sequence[float],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    if raw_xs and raw_ys:
        terminal_x = _safe_float(raw_xs[-1])
        terminal_y = _safe_float(raw_ys[-1])
        if _point_in_limits(terminal_x, terminal_y, xlim, ylim):
            return terminal_x, terminal_y
        visible_pt = _last_visible_fig6_point(clipped_xs, clipped_ys)
        if visible_pt is not None:
            return visible_pt
        if math.isfinite(terminal_x) and math.isfinite(terminal_y):
            return (
                min(max(terminal_x, xlim[0]), xlim[1]),
                min(max(terminal_y, ylim[0]), ylim[1]),
            )
    return _last_visible_fig6_point(clipped_xs, clipped_ys)


def _terminal_marker_name(reason: str) -> str:
    reason = str(reason or "").lower()
    if reason == "success":
        return "success_star"
    if reason == "collision":
        return "collision_x"
    return "lost_timeout_circle"


def _plot_terminal_marker(ax, x: float, y: float, reason: str, method_color: str) -> str:
    marker_name = _terminal_marker_name(reason)
    if marker_name == "success_star":
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=145,
            color="#d62728",
            edgecolor="white",
            linewidth=0.7,
            zorder=22,
            clip_on=False,
        )
    elif marker_name == "collision_x":
        ax.scatter(
            [x],
            [y],
            marker="X",
            s=36,
            color=method_color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            zorder=22,
            clip_on=False,
        )
    else:
        ax.scatter(
            [x],
            [y],
            marker="o",
            s=34,
            facecolors="white",
            edgecolors=method_color,
            linewidth=1.15,
            alpha=0.78,
            zorder=22,
            clip_on=False,
        )
    return marker_name


def _add_fig6_direction_arrows(ax, xs: Sequence[float], ys: Sequence[float], color: str) -> None:
    valid = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(valid) < 6:
        return
    arrow_count = min(4, max(2, len(valid) // 45))
    idxs = [
        int(round((i + 1) * (len(valid) - 2) / (arrow_count + 1)))
        for i in range(arrow_count)
    ]
    used = set()
    for idx in idxs:
        idx = max(0, min(idx, len(valid) - 2))
        if idx in used:
            continue
        used.add(idx)
        x0, y0 = valid[idx]
        x1, y1 = valid[idx + 1]
        if abs(x1 - x0) + abs(y1 - y0) < 1e-5:
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.0,
                "mutation_scale": 7.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=4,
        )


def _plot_fig6(args) -> None:
    candidate_csv = str(getattr(args, "fig6_candidate_csv", "") or "").strip()
    datasets = {} if candidate_csv else _load_fig6_datasets(args)
    if not datasets:
        if not candidate_csv:
            return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
    except Exception as exc:
        raise RuntimeError(f"matplotlib unavailable; cannot draw Fig.6 ({exc})") from exc

    try:
        selected, selected_meta, selection_mode, candidate_rows = _choose_fig6_selection(
            datasets,
            getattr(args, "fig6_episode_id", None),
            candidate_csv,
            int(getattr(args, "fig6_candidate_rank", 0)),
        )
    except RuntimeError as exc:
        existing_png = os.path.join(args.output_dir, "fig6_trajectories_stage4.png")
        existing_pdf = os.path.join(args.output_dir, "fig6_trajectories_stage4.pdf")
        if bool(getattr(args, "fig6_required", False)) or not os.path.isfile(existing_png):
            raise
        image = plt.imread(existing_png)
        height_px, width_px = image.shape[:2]
        fallback_fig = plt.figure(
            figsize=(width_px / 300.0, height_px / 300.0),
            dpi=300,
        )
        fallback_ax = fallback_fig.add_axes([0.0, 0.0, 1.0, 1.0])
        fallback_ax.imshow(image)
        fallback_ax.axis("off")
        fallback_fig.savefig(existing_pdf, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fallback_fig)
        args._fig6_regeneration_mode = "raster-wrapped PDF; original timeseries unavailable locally"
        args._fig6_regeneration_warning = str(exc)
        print(
            "[Warn] Fig.6 source timeseries are unavailable locally; preserved the "
            "approved PNG and created a raster-wrapped PDF. Re-run on the server for "
            f"a vector PDF. Detail: {exc}"
        )
        return
    args._fig6_regeneration_mode = "vector redraw from timeseries"
    args._fig6_regeneration_warning = ""
    obstacle_refs = set()
    for key, rows in selected.items():
        obstacle_refs.add(_canonical_obstacles(rows[0].get("obstacles_json", "")))
    if len(obstacle_refs) != 1:
        if not candidate_csv:
            raise RuntimeError(
                "Fig.6 selected trajectories do not share the same obstacle layout. "
                "Re-run trajectory eval with more episodes or pass a valid common --fig6_episode_id."
            )
        learned_rows = selected.get((FIG6_SPEEDS[0], "learnedw"), [])
        layout_text = _canonical_obstacles(learned_rows[0].get("obstacles_json", "")) if learned_rows else next(iter(obstacle_refs))
        print(
            "[Warn] Fig.6 candidate trajectories use multiple obstacle layouts; "
            "drawing the Learned-w layout as the background."
        )
    else:
        layout_text = next(iter(obstacle_refs))
    obstacles = json.loads(layout_text)

    xlim, ylim = _fig6_window_limits(obstacles, selected, args)

    speed = FIG6_SPEEDS[0]
    # Single-column horizontal layout: draw forward world-y on the horizontal axis
    # and lateral world-x on the vertical axis. The underlying trajectory data and
    # selected episodes remain unchanged.
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 1.65), constrained_layout=True)
    plot_xlim = ylim
    plot_ylim = xlim
    for obs in obstacles:
        oy = _safe_float(obs.get("y"))
        ox = _safe_float(obs.get("x"))
        radius = _safe_float(obs.get("r"))
        if not math.isfinite(oy) or oy + radius < ylim[0] or oy - radius > ylim[1]:
            continue
        ax.add_patch(
            Circle(
                (oy, ox),
                radius,
                facecolor="#9e9e9e",
                edgecolor="#666666",
                linewidth=0.5,
                alpha=0.55,
                zorder=1,
            )
        )

    target_rows = selected[(speed, "learnedw")]
    raw_tx = [_safe_float(r.get("target_x", "")) for r in target_rows]
    raw_ty = [_safe_float(r.get("target_y", "")) for r in target_rows]
    tx, ty, _ = _truncate_fig6_at_reset(raw_tx, raw_ty)
    tx, ty = _clip_fig6_xy(tx, ty, xlim, ylim)
    target_line, = ax.plot(ty, tx, linewidth=1.2, color="#009e73", alpha=0.95, label="Target", zorder=2)
    target_line.set_dashes([5.0, 3.0])
    _add_fig6_direction_arrows(ax, ty, tx, "#009e73")

    start_marked = False
    terminal_draw_info: Dict[str, Dict[str, str]] = {}
    for method in METHOD_ORDER:
        rows = selected[(speed, method)]
        meta = selected_meta.get((speed, method), {})
        raw_xs = [_safe_float(r.get("robot_x", "")) for r in rows]
        raw_ys = [_safe_float(r.get("robot_y", "")) for r in rows]
        trajectory_xs, trajectory_ys, reset_detected = _truncate_fig6_at_reset(raw_xs, raw_ys)
        xs, ys = _clip_fig6_xy(trajectory_xs, trajectory_ys, xlim, ylim)
        style = METHOD_STYLE[method]
        alpha = 0.92 if method == "learnedw" else 0.90
        linewidth = 1.05 if method == "learnedw" else max(0.65, style["linewidth"] * 0.50)
        z = 5 if method == "learnedw" else 3
        ax.plot(
            ys,
            xs,
            color=style["color"],
            linewidth=linewidth,
            alpha=alpha,
            zorder=z,
        )
        if not start_marked and xs and ys:
            first_pt = _first_visible_fig6_point(xs, ys)
            if first_pt is not None:
                ax.scatter(
                    [first_pt[1]],
                    [first_pt[0]],
                    marker="s",
                    s=36,
                    color="#000000",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=18,
                    clip_on=False,
                )
                start_marked = True
        terminal_pt = _fig6_visible_terminal_point(
            trajectory_xs,
            trajectory_ys,
            xs,
            ys,
            xlim,
            ylim,
        )
        if terminal_pt is not None:
            reason = str(meta.get("reason") or rows[-1].get("episode_termination_reason", "")).lower()
            marker_name = _plot_terminal_marker(
                ax,
                terminal_pt[1],
                terminal_pt[0],
                reason,
                style["color"],
            )
            raw_terminal_x = _safe_float(trajectory_xs[-1]) if trajectory_xs else float("nan")
            raw_terminal_y = _safe_float(trajectory_ys[-1]) if trajectory_ys else float("nan")
            recorded_last_x = _safe_float(raw_xs[-1]) if raw_xs else float("nan")
            recorded_last_y = _safe_float(raw_ys[-1]) if raw_ys else float("nan")
            terminal_draw_info[method] = {
                "reason": reason,
                "marker": marker_name,
                "plot_x": f"{terminal_pt[0]:.6f}",
                "plot_y": f"{terminal_pt[1]:.6f}",
                "raw_x": f"{raw_terminal_x:.6f}" if math.isfinite(raw_terminal_x) else "nan",
                "raw_y": f"{raw_terminal_y:.6f}" if math.isfinite(raw_terminal_y) else "nan",
                "clipped": "0" if _point_in_limits(raw_terminal_x, raw_terminal_y, xlim, ylim) else "1",
                "reset_detected": "1" if reset_detected else "0",
                "recorded_last_x": (
                    f"{recorded_last_x:.6f}" if math.isfinite(recorded_last_x) else "nan"
                ),
                "recorded_last_y": (
                    f"{recorded_last_y:.6f}" if math.isfinite(recorded_last_y) else "nan"
                ),
            }
            print(
                f"[Fig6] terminal method={method} reason={reason} marker={marker_name} "
                f"plot=({terminal_pt[0]:.3f},{terminal_pt[1]:.3f}) "
                f"pre_reset=({raw_terminal_x:.3f},{raw_terminal_y:.3f}) "
                f"recorded_last=({recorded_last_x:.3f},{recorded_last_y:.3f}) "
                f"reset_detected={int(reset_detected)}"
            )

    ax.set_title(f"{float(speed):.2f} m/s", fontsize=9, pad=2)
    ax.set_xlabel("world y / forward [m]", fontsize=8)
    ax.set_ylabel("world x / lateral [m]", fontsize=8)
    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.38)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.30, alpha=0.22)
    ax.tick_params(labelsize=7)

    method_handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"],
               linewidth=1.05 if m == "learnedw" else max(0.65, METHOD_STYLE[m]["linewidth"] * 0.50),
               label=SHORT_METHOD_LABELS[m],
               alpha=0.92 if m == "learnedw" else 0.90)
        for m in METHOD_ORDER
    ]
    event_handles = [
        Line2D([0], [0], color="#009e73", linestyle=(0, (5.0, 3.0)), linewidth=1.2, label="Target"),
        Line2D([0], [0], marker="s", color="#000000", linestyle="None", markersize=6, label="Start"),
        Line2D([0], [0], marker="*", markerfacecolor="#d62728", markeredgecolor="white",
               color="#d62728", linestyle="None", markersize=10, label="Success"),
        Line2D([0], [0], marker="X", markerfacecolor="#777777", markeredgecolor="white",
               color="#777777", alpha=0.78, linestyle="None", markersize=5, label="Collision"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#777777",
               color="#777777", alpha=0.78, linestyle="None", markersize=4.4, label="Follow lost/timeout"),
    ]
    fig.legend(
        handles=method_handles + event_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=6.2,
        handlelength=1.5,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, 1.17),
    )
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir, f"fig6_trajectories_stage4.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    source_rows = []
    for speed in FIG6_SPEEDS:
        for method in METHOD_ORDER:
            rows = selected[(speed, method)]
            meta = selected_meta.get((speed, method), {})
            draw_info = terminal_draw_info.get(method, {})
            source_rows.append(
                {
                    "Speed": speed,
                    "Method": SHORT_METHOD_LABELS[method],
                    "Episode": meta.get("episode", ""),
                    "RawEpisode": meta.get("episode_raw", ""),
                    "RunID": meta.get("run_id", ""),
                    "Selection": selection_mode,
                    "TrajectoryFrame": rows[0].get("trajectory_frame", ""),
                    "Termination": meta.get("reason", rows[-1].get("episode_termination_reason", "")),
                    "TerminalMarker": draw_info.get("marker", ""),
                    "TerminalPlotX": draw_info.get("plot_x", ""),
                    "TerminalPlotY": draw_info.get("plot_y", ""),
                    "TerminalRawX": draw_info.get("raw_x", ""),
                    "TerminalRawY": draw_info.get("raw_y", ""),
                    "TerminalClipped": draw_info.get("clipped", ""),
                    "ResetDetected": draw_info.get("reset_detected", ""),
                    "RecordedLastX": draw_info.get("recorded_last_x", ""),
                    "RecordedLastY": draw_info.get("recorded_last_y", ""),
                    "LayoutID": hashlib.sha1(meta.get("layout", "").encode("utf-8")).hexdigest()[:10],
                    "Source": meta.get("source") or datasets.get((speed, method), {}).get("path", ""),
                }
            )
    _write_csv(
        os.path.join(args.output_dir, "fig6_trajectories_stage4_sources.csv"),
        source_rows,
        [
            "Speed", "Method", "Episode", "RawEpisode", "RunID", "Selection",
            "TrajectoryFrame", "Termination", "TerminalMarker",
            "TerminalPlotX", "TerminalPlotY", "TerminalRawX", "TerminalRawY",
            "TerminalClipped", "ResetDetected", "RecordedLastX", "RecordedLastY",
            "LayoutID", "Source",
        ],
    )
    _write_csv(
        os.path.join(args.output_dir, "fig6_trajectory_episode_candidates.csv"),
        candidate_rows,
        [
            "LayoutID", "Score", "LearnedSuccess", "BaselineCollisions", "BaselineLostTimeout",
            "BaselineSuccess", "SelectedEpisodes", "SelectedTerminations", "SelectedSources",
        ],
    )


def _copy_legacy_mechanism_png(args) -> None:
    if not args.learnedw_mechanism_dir or not os.path.isdir(args.learnedw_mechanism_dir):
        print("[Warn] learned-w mechanism directory not found; legacy mechanism copy skipped.")
        return
    copied = []
    for name in ("mechanism_risk_bins.png", "mechanism_priv_conflict_bins.png"):
        src = os.path.join(args.learnedw_mechanism_dir, name)
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, "legacy_" + name)
            shutil.copy2(src, dst)
            copied.append(dst)
    if copied:
        with open(os.path.join(args.output_dir, "legacy_mechanism_note.md"), "w", encoding="utf-8") as f:
            f.write("# Legacy mechanism plots\n\n")
            f.write("These PNGs are copied only for audit. Use fig4_mechanism.pdf/png for the v3 manuscript figure.\n")


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _class_block(source: str, class_name: str, next_class_name: Optional[str] = None) -> str:
    start_pat = f"class {class_name}"
    start = source.find(start_pat)
    if start < 0:
        return source
    if next_class_name:
        end = source.find(f"class {next_class_name}", start + len(start_pat))
        if end > start:
            return source[start:end]
    matches = list(re.finditer(r"^class\s+\w+", source[start + len(start_pat):], flags=re.M))
    if matches:
        end = start + len(start_pat) + matches[0].start()
        return source[start:end]
    return source[start:]


def _arg_default(source: str, name: str) -> str:
    pat = r"add_argument\(\s*['\"]--" + re.escape(name) + r"['\"].*?default\s*=\s*([^,\)\n]+)"
    m = re.search(pat, source, flags=re.S)
    if not m:
        return "N/A"
    return m.group(1).strip().strip("'\"")


def _assign_value(source: str, name: str) -> str:
    m = re.search(r"^\s*" + re.escape(name) + r"\s*=\s*([^\n#]+)", source, flags=re.M)
    return m.group(1).strip() if m else "N/A"


def _reward_cfg_value(source: str, name: str) -> str:
    m = re.search(r"reward_cfg\[['\"]" + re.escape(name) + r"['\"]\]\s*=\s*([^\n#]+)", source)
    return m.group(1).strip() if m else "N/A"


def _build_appendix_a1(args) -> List[Dict]:
    learned_meta = _checkpoint_meta(args.learnedw_ckpt)
    risk_meta = _checkpoint_meta(args.risk_only_ckpt)
    shared_keys = (
        "num_steps",
        "num_envs",
        "lr",
        "gamma",
        "gae_lambda",
        "entropy_coef",
        "clip_range",
        "value_loss_coef",
        "max_grad_norm",
        "mini_batch_size",
        "num_epochs",
    )
    mismatched = [
        key for key in shared_keys
        if learned_meta.get(key) != risk_meta.get(key)
    ]
    if mismatched:
        raise RuntimeError(
            "Learned-w and Risk-only PPO settings differ: " + ", ".join(mismatched)
        )
    ckpt_source = "final Learned-w/Risk-only checkpoint metadata"
    rows = [
        ("Algorithm", "PPO custom loop", "train_highlevel.py"),
        ("Steps per iter", str(learned_meta["num_steps"]), ckpt_source),
        ("Num envs", str(learned_meta["num_envs"]), ckpt_source),
        ("Learning rate", str(learned_meta["lr"]), ckpt_source),
        ("Gamma", str(learned_meta["gamma"]), ckpt_source),
        ("GAE lambda", str(learned_meta["gae_lambda"]), ckpt_source),
        ("Entropy coef", str(learned_meta["entropy_coef"]), ckpt_source),
        ("Clip epsilon", str(learned_meta["clip_range"]), ckpt_source),
        ("Value loss coef", str(learned_meta["value_loss_coef"]), ckpt_source),
        ("Max grad norm", str(learned_meta["max_grad_norm"]), ckpt_source),
        ("Mini-batch", str(learned_meta["mini_batch_size"]), ckpt_source),
        ("Epochs", str(learned_meta["num_epochs"]), ckpt_source),
        ("Training budget [iters]", str(args.paper_train_iters), "paper training command"),
        ("Selected Learned-w checkpoint iter", _checkpoint_iteration(args.learnedw_ckpt), args.learnedw_ckpt),
        ("Selected Risk-only checkpoint iter", _checkpoint_iteration(args.risk_only_ckpt), args.risk_only_ckpt),
        ("Optimizer", "Adam", "train_highlevel.py optimizer construction"),
    ]
    out = [{"Item": k, "Value": v, "Source": s} for k, v, s in rows]
    fields = ["Item", "Value", "Source"]
    _write_csv(os.path.join(args.output_dir, "tableA1_ppo_hyperparams.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA1_ppo_hyperparams.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA1_ppo_hyperparams.tex"), out, fields)
    return out


def _build_appendix_a2(args) -> List[Dict]:
    scene_src = _read_text("legged_gym/envs/hex_v4/hex_scenes_config.py")
    pcr_block = _class_block(scene_src, "HexPCRLineAvoidBasicCfg", "HexPCRLineAvoidBasicCfgPPO")
    learned_meta = _checkpoint_meta(args.learnedw_ckpt)
    ckpt_source = "final Learned-w checkpoint metadata"
    rows = [
        ("pcr_progress_reward_scale", _assign_value(pcr_block, "pcr_progress_reward_scale"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("pcr_progress_cap", _assign_value(pcr_block, "pcr_progress_cap"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("pcr_follow_quality_floor", _assign_value(pcr_block, "pcr_follow_quality_floor"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("pcr_follow_quality_sigma", _assign_value(pcr_block, "pcr_follow_quality_sigma"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("pcr_gate_aux_scale", _assign_value(pcr_block, "pcr_gate_aux_scale"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("pcr_gap_success_bonus", _assign_value(pcr_block, "pcr_gap_success_bonus"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("follow_distance_desired", _reward_cfg_value(pcr_block, "follow_distance_desired"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("follow_distance_sigma", _reward_cfg_value(pcr_block, "follow_distance_sigma"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("follow_distance_scale", _reward_cfg_value(pcr_block, "follow_distance_scale"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("follow_band_scale", _reward_cfg_value(pcr_block, "follow_band_scale"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("collision_penalty", _reward_cfg_value(pcr_block, "collision_penalty"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("terminal_fail_penalty", _reward_cfg_value(pcr_block, "terminal_fail_penalty"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("avoid_band_penalty_scale", _assign_value(pcr_block, "avoid_band_penalty_scale"), "HexPCRLineAvoidBasicCfg.navigation"),
        ("avoid_row_gap_scale", _reward_cfg_value(pcr_block, "avoid_row_gap_scale"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("avoid_row_cmdx_scale", _reward_cfg_value(pcr_block, "avoid_row_cmdx_scale"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("pcr_w_aux_enable", str(bool(learned_meta["pcr_w_aux_enable"])), ckpt_source),
        ("pcr_w_aux_coef", str(learned_meta["pcr_w_aux_coef"]), ckpt_source),
        ("pcr_w_aux_risk_f_threshold", str(learned_meta["pcr_w_aux_risk_f_threshold"]), ckpt_source),
        ("pcr_w_aux_risk_margin", str(learned_meta["pcr_w_aux_risk_margin"]), ckpt_source),
        ("pcr_w_aux_cmd_cos_threshold", str(learned_meta["pcr_w_aux_cmd_cos_threshold"]), ckpt_source),
        ("gate_smooth_penalty", _reward_cfg_value(pcr_block, "gate_smooth_penalty"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("risk_barrier_scale", _reward_cfg_value(pcr_block, "risk_barrier_scale"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
        ("time_penalty", _reward_cfg_value(pcr_block, "time_penalty"), "HexPCRLineAvoidBasicCfg.reward_cfg"),
    ]
    out = [{"Term": k, "Weight/Value": v, "Source": s} for k, v, s in rows]
    fields = ["Term", "Weight/Value", "Source"]
    _write_csv(os.path.join(args.output_dir, "tableA2_reward_terms.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA2_reward_terms.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA2_reward_terms.tex"), out, fields)
    return out


def _build_appendix_a3(args) -> List[Dict]:
    scene_src = _read_text("legged_gym/envs/hex_v4/hex_scenes_config.py")
    avoid_block = _class_block(scene_src, "HexAvoidBasicCfg", "HexAvoidBasicCfgPPO")
    pcr_block = _class_block(scene_src, "HexPCRLineAvoidBasicCfg", "HexPCRLineAvoidBasicCfgPPO")
    terrain_src = _read_text("legged_gym/envs/hex_v4/hex_terrain_config.py")
    base_src = _read_text("legged_gym/envs/base/legged_robot_config.py")
    learned_meta = _checkpoint_meta(args.learnedw_ckpt)
    obs_contract = learned_meta.get("observation_contract", {})
    if not isinstance(obs_contract, dict):
        raise RuntimeError("Learned-w checkpoint observation_contract is missing.")
    training_task = str(learned_meta.get("task", "N/A"))
    checkpoint_source = "final Learned-w checkpoint observation contract"
    fov = (
        f"{float(obs_contract['camera_horizontal_fov_deg']):g} x "
        f"{float(obs_contract['camera_vertical_fov_deg']):g}"
    )
    depth_range = (
        f"[{float(obs_contract['camera_near_m']):g},"
        f"{float(obs_contract['camera_far_m']):g}]"
    )
    map_shape = (
        f"{int(obs_contract['affordance_map_size'])} x "
        f"{int(obs_contract['affordance_map_size'])}"
    )
    rows = [
        ("Training task", training_task, args.learnedw_ckpt),
        ("Evaluation task", "s_pcr_line_avoid_basic stage 4", "eval command"),
        ("Eval target speeds", ",".join(SPEEDS), "run_pcr_main_table_eval command"),
        ("Training target-speed range [m/s]", "0.25-0.65", "HexGround._sample_pcr_new_curriculum"),
        ("Training obstacle rows", "2-5", "HexGround._sample_pcr_new_curriculum / stage layouts"),
        ("Curriculum budget [episodes]", "120000", "HexPCRNewCfg.navigation"),
        ("Episode length [s]", _assign_value(avoid_block, "episode_length_s"), "HexAvoidBasicCfg.env"),
        ("Stage-4 passage width min [m]", _assign_value(avoid_block, "avoid_stage4_width_min"), "HexAvoidBasicCfg.terrain"),
        ("Capsule obstacle radius [m]", _assign_value(avoid_block, "avoid_capsule_radius"), "HexAvoidBasicCfg.terrain"),
        ("Capsule slots", _assign_value(avoid_block, "avoid_capsule_slots"), "HexAvoidBasicCfg.terrain"),
        ("Box slots", _assign_value(avoid_block, "avoid_box_slots"), "HexAvoidBasicCfg.terrain"),
        ("Wall slots", _assign_value(avoid_block, "avoid_wall_slots"), "HexAvoidBasicCfg.terrain"),
        ("Terrain seed", _assign_value(avoid_block, "avoid_seed"), "HexAvoidBasicCfg.terrain"),
        ("Obstacle position jitter [m]", _assign_value(avoid_block, "avoid_fixed_preset_jitter_xy"), "HexAvoidBasicCfg.terrain"),
        ("Mirrored layouts", str(_assign_value(avoid_block, "avoid_fixed_presets_use_mirror")), "HexAvoidBasicCfg.terrain"),
        ("Friction randomization enabled", _assign_value(base_src, "randomize_friction"), "LeggedRobotCfg.domain_rand"),
        ("Friction range", _assign_value(terrain_src, "friction_range"), "hex_terrain_config.domain_rand"),
        ("Base-mass randomization enabled", _assign_value(base_src, "randomize_base_mass"), "LeggedRobotCfg.domain_rand"),
        ("External pushes enabled", _assign_value(terrain_src, "push_robots"), "hex_terrain_config.domain_rand"),
        ("Low-level observation noise enabled", _assign_value(base_src, "add_noise"), "LeggedRobotCfg.noise"),
        ("High-level training affordance", "actor-only scene GT; camera disabled", args.learnedw_ckpt),
        ("Deployment sensor target", str(learned_meta.get("sim2real_sensor_target", "Intel RealSense D435i")), args.learnedw_ckpt),
        ("Depth resolution", f"{int(obs_contract['camera_width'])} x {int(obs_contract['camera_height'])}", checkpoint_source),
        ("Depth FOV H x V [deg]", fov, checkpoint_source),
        ("Depth range [m]", depth_range, checkpoint_source),
        ("Configured/effective depth rate [Hz]", f"{obs_contract['camera_fps_cfg']} / {float(obs_contract['camera_effective_rate_hz']):.1f}", checkpoint_source),
        ("High-level rate [Hz]", f"{float(obs_contract['high_level_rate_hz']):.1f}", checkpoint_source),
        ("Affordance map", f"{map_shape}, extent {float(obs_contract['affordance_map_extent_m']):g} m", checkpoint_source),
    ]
    out = [{"Item": k, "Value": v, "Source": s} for k, v, s in rows]
    fields = ["Item", "Value", "Source"]
    _write_csv(os.path.join(args.output_dir, "tableA3_domain_randomization.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA3_domain_randomization.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA3_domain_randomization.tex"), out, fields)
    return out


def _build_appendix_a4(args) -> List[Dict]:
    learned_meta = _checkpoint_meta(args.learnedw_ckpt)
    risk_meta = _checkpoint_meta(args.risk_only_ckpt)
    learned_sd = _checkpoint_state_dict(args.learnedw_ckpt)
    risk_sd = _checkpoint_state_dict(args.risk_only_ckpt)
    avoid_sd = _checkpoint_state_dict(args.avoid_ckpt)
    learned_aff = _shape(learned_sd, "affordance_encoder.cnn.0.weight")
    learned_state = _shape(learned_sd, "state_encoder.mlp.0.weight")
    learned_goal = _shape(learned_sd, "goal_encoder.mlp.0.weight")
    risk_state = _shape(risk_sd, "state_encoder.mlp.0.weight")
    risk_goal = _shape(risk_sd, "goal_encoder.mlp.0.weight")
    fusion = _shape(learned_sd, "fusion.0.weight")
    fusion2 = _shape(learned_sd, "fusion.2.weight")
    avoid_state = _shape(avoid_sd, "state_encoder.mlp.0.weight")
    avoid_goal = _shape(avoid_sd, "goal_encoder.mlp.0.weight")
    avoid_cmd = _shape(avoid_sd, "cmd_mean_head.2.weight")
    learned_policy_params = _state_dict_param_count(learned_sd)
    risk_policy_params = _state_dict_param_count(risk_sd)
    avoid_params = _state_dict_param_count(avoid_sd)
    rows = [
        ("Affordance encoder", f"{learned_aff[1]} channels -> 32 -> 64 -> 128", args.learnedw_ckpt),
        ("Learned-w state encoder", f"{learned_state[1]} -> 64 -> 64 -> 64", args.learnedw_ckpt),
        ("Learned-w goal encoder", f"{learned_goal[1]} -> 32 -> 32 (2 goal + 16 conflict features)", args.learnedw_ckpt),
        ("Risk-only state/goal encoders", f"{risk_state[1]} -> 64; {risk_goal[1]} -> 32", args.risk_only_ckpt),
        (
            "Difficulty input",
            "1 scalar slot; disabled/zero in final checkpoints",
            f"gate_use_difficulty={learned_meta.get('gate_use_difficulty', False)}",
        ),
        ("Shared trunk", f"{fusion[1]} -> {fusion[0]} -> {fusion2[0]}, ELU", args.learnedw_ckpt),
        ("Gate y head", "256 -> 64 -> Beta(alpha,beta)", args.learnedw_ckpt),
        ("Learned-w head", "256 -> 64 -> Beta(alpha,beta)", args.learnedw_ckpt),
        ("Risk-only actor output", str(risk_meta.get("actor_output_dim", 1)), args.risk_only_ckpt),
        ("Learned-w actor output", str(learned_meta.get("actor_output_dim", 2)), args.learnedw_ckpt),
        ("Critic", "Separate encoders + 225 -> 256 -> 256 -> 64 -> 1", args.learnedw_ckpt),
        ("Avoid expert", f"state {avoid_state[1]}, goal {avoid_goal[1]}, command output {avoid_cmd[0]}", args.avoid_ckpt),
        ("Fixed Avoid expert params", str(avoid_params), args.avoid_ckpt),
        ("Risk-only policy params", str(risk_policy_params), args.risk_only_ckpt),
        ("Learned-w policy params", str(learned_policy_params), args.learnedw_ckpt),
        ("Risk-only high-level total", str(risk_policy_params + avoid_params), "policy + fixed Avoid expert"),
        ("Learned-w high-level total", str(learned_policy_params + avoid_params), "policy + fixed Avoid expert"),
        ("Follow expert", "Analytic", "train_highlevel.py _compute_moe_follow_cmd_from_goal"),
        ("Low-level locomotion", "Fixed checkpoint", "eval/train command low_level_ckpt"),
    ]
    out = [{"Component": k, "Value": v, "Source": s} for k, v, s in rows]
    fields = ["Component", "Value", "Source"]
    _write_csv(os.path.join(args.output_dir, "tableA4_network_structure.csv"), out, fields)
    _write_markdown(os.path.join(args.output_dir, "tableA4_network_structure.md"), out, fields)
    _write_latex_booktabs(os.path.join(args.output_dir, "tableA4_network_structure.tex"), out, fields)
    return out


def _write_manifest(
    args,
    table1: Sequence[Dict],
    table2: Sequence[Dict],
    table3: Sequence[Dict],
    table_a5: Sequence[Dict],
    table_heldout: Sequence[Dict],
) -> None:
    path = os.path.join(args.output_dir, "MANIFEST.md")
    risk_rows = _raw_rows_from_all_csv([args.risk_all_csv], methods=["risk_only"])
    risk_only_mode = _risk_only_source_mode(args, risk_rows)
    generated_files = [
        "fig3_main_performance.pdf",
        "fig3_main_performance.png",
        "fig3b_scatter_sized.pdf",
        "fig3b_scatter_sized.png",
        "fig3b_scatter_sized_data_060.csv",
        "fig3b_scatter_sized_notes.md",
        "fig4_mechanism.pdf",
        "fig4_mechanism.png",
        "fig6_trajectories_stage4.pdf",
        "fig6_trajectories_stage4.png",
        "fig6_trajectories_stage4_sources.csv",
        "fig6_trajectory_episode_candidates.csv",
        "table1_main_performance_stage4.csv",
        "table1_main_performance_stage4.md",
        "table1_main_performance_stage4.tex",
        "table2_mechanism_ablation.csv",
        "table2_mechanism_ablation.md",
        "table2_mechanism_ablation.tex",
        "table3_mono_ppo_stage_probe.csv",
        "table3_mono_ppo_stage_probe.md",
        "table3_mono_ppo_stage_probe.tex",
        "table3_mono_ppo_stage_probe_notes.md",
        "tableA1_ppo_hyperparams.csv",
        "tableA1_ppo_hyperparams.md",
        "tableA1_ppo_hyperparams.tex",
        "tableA2_reward_terms.csv",
        "tableA2_reward_terms.md",
        "tableA2_reward_terms.tex",
        "tableA3_domain_randomization.csv",
        "tableA3_domain_randomization.md",
        "tableA3_domain_randomization.tex",
        "tableA4_network_structure.csv",
        "tableA4_network_structure.md",
        "tableA4_network_structure.tex",
        "tableA5_delta_y_full.csv",
        "tableA5_delta_y_full.md",
        "tableA5_delta_y_full.tex",
        "tableA5_delta_y_full_notes.md",
        "tableA_dwa_velocity_search_diagnostic.csv",
        "tableA_dwa_velocity_search_diagnostic.md",
        "tableA_dwa_velocity_search_diagnostic.tex",
        "tableA_dwa_velocity_search_diagnostic_notes.md",
        "tableA_heldout_irregular_rows.csv",
        "tableA_heldout_irregular_rows.md",
        "tableA_heldout_irregular_rows.tex",
        "tableA_heldout_irregular_rows_notes.md",
    ]
    missing_outputs = [
        name
        for name in generated_files
        if not os.path.isfile(os.path.join(args.output_dir, name))
        or os.path.getsize(os.path.join(args.output_dir, name)) <= 0
    ]
    if missing_outputs:
        raise RuntimeError(
            "Final paper output validation failed; missing or empty files: "
            + ", ".join(missing_outputs)
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Final Paper Outputs v3\n\n")
        f.write("Generated files:\n\n")
        for name in generated_files:
            size = os.path.getsize(os.path.join(args.output_dir, name))
            f.write(f"- {name} ({size} bytes)\n")
        f.write("\nRow counts:\n\n")
        f.write(f"- Table I audit rows: {len(table1)}\n")
        f.write(f"- Table II rows: {len(table2)}\n")
        f.write(f"- Table III rows: {len(table3)}\n")
        f.write(f"- Table A5 rows: {len(table_a5)}\n")
        f.write(f"- Held-out irregular-row rows: {len(table_heldout)}\n")
        f.write("\nSource files and paths:\n\n")
        for label, src in [
            ("internal_all_csv", args.internal_all_csv),
            ("internal_aggregate_csv", args.internal_aggregate_csv),
            ("risk_all_csv", args.risk_all_csv),
            ("risk_aggregate_csv", args.risk_aggregate_csv),
            ("rule_all_csv", args.rule_all_csv),
            ("rule_aggregate_csv", args.rule_aggregate_csv),
            ("additive_all_csv", getattr(args, "additive_all_csv", "")),
            ("velocity_search_all_csv", getattr(args, "velocity_search_all_csv", "")),
            ("heldout_layout_root", getattr(args, "heldout_layout_root", "")),
            ("heldout_layout_all_csv", getattr(args, "heldout_layout_all_csv", "")),
            ("learnedw_diag_all_csv", args.learnedw_diag_all_csv),
            ("learnedw_diag_aggregate_csv", args.learnedw_diag_aggregate_csv),
            ("learnedw_split_all_csv", args.learnedw_split_all_csv),
            ("mono_stage2_path", args.mono_stage2_path),
            ("mono_stage3_path", args.mono_stage3_path),
            ("mono_stage4_all_csv", args.mono_stage4_all_csv),
            ("mono_stage_snapshot_csv", args.mono_stage_snapshot_csv),
            ("learnedw_mechanism_dir", args.learnedw_mechanism_dir),
            ("learnedw_ckpt", args.learnedw_ckpt),
            ("risk_only_ckpt", args.risk_only_ckpt),
            ("avoid_ckpt", args.avoid_ckpt),
            ("fig6_timeseries_root", args.fig6_timeseries_root),
        ]:
            f.write(f"- {label}: `{src}` (mtime={_file_stamp(src)})\n")
        f.write("\nCheckpoint SHA256:\n\n")
        for label, src in [
            ("learnedw_ckpt", args.learnedw_ckpt),
            ("risk_only_ckpt", args.risk_only_ckpt),
            ("avoid_ckpt", args.avoid_ckpt),
        ]:
            f.write(f"- {label}: `{_file_sha256(src)}`\n")
        f.write("\nRisk-only source mode:\n\n")
        f.write(f"- requested: `{args.risk_only_source_mode}`\n")
        f.write(f"- resolved: `{risk_only_mode}`\n")
        f.write("\nFig.6 regeneration:\n\n")
        f.write(
            f"- mode: `{getattr(args, '_fig6_regeneration_mode', 'not generated')}`\n"
        )
        warning = str(getattr(args, "_fig6_regeneration_warning", "") or "")
        if warning:
            f.write(f"- warning: `{warning}`\n")
        f.write("\nValidation checklist:\n\n")
        table1_methods = sorted({str(row.get("method_key", "")) for row in table1 if row.get("method_key")})
        f.write(
            f"- Table I: expected {len(table1)} rows = 3 speeds x {len(table1_methods)} methods; "
            "Mono-PPO is intentionally excluded.\n"
        )
        f.write("- Table I: Risk-only 0.60 success should be about 0.008 +/- 0.008.\n")
        f.write(
            f"- Fig.3(b): the selected scatter uses {len(_fig3_method_order(table1))} Table I methods "
            "at 0.60 m/s; marker-area encoding is recorded in fig3b_scatter_sized_notes.md.\n"
        )
        f.write("- Table II: speed is fixed by --mechanism_speed, default 0.60.\n")
        f.write("- Table II: Risk-only note must say trained from scratch and no learned-w channel.\n")
        f.write("- Table II: Params should separate Risk-only and Learned-w.\n")
        f.write("- Table A5: contains no N/A values in Delta y_w / Delta y_r / Delta y_total over All, C_unsafe, C_avoid.\n")
        f.write("- Held-out irregular-row table: eval-only Stage 4 at 0.60 m/s; not used for training or validation tuning.\n")
        f.write("- Fig.4: x-axis label must be Conflict intensity and Delta y must equal y_eff - y_raw.\n")
        f.write("- Fig.6: if requested, 0.60 m/s timeseries must contain robot_x/y, target_x/y, obstacles_json, episode_termination_reason, and trajectory_frame=world_xy_train_play; the audited default candidate requires Learned-w success and prioritizes one row-3 collision, one row-4 collision, one early lost/timeout, and one late lost/timeout trajectory without mixing mirrored layouts.\n")
        f.write("- Table III: fixed 0.35 m/s, stages 2/3/4, seed 1; Collision TTC averages collision episodes only.\n")
        f.write("- Table III: Mono-PPO has 1,029,447 params and is not capacity matched to Learned-w; treat it as a diagnostic probe.\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build final PCR-Net paper tables and figures.")
    parser.add_argument("--output_dir", default="agents/final_paper_outputs_v3")
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
        default="agents/eval_data_risk_only/pcr_main_table/pcr_main_table_all_metrics_20260610_170313.csv",
    )
    parser.add_argument(
        "--risk_aggregate_csv",
        default="agents/eval_data_risk_only/pcr_main_table/pcr_main_table_aggregate_20260610_170313.csv",
    )
    parser.add_argument(
        "--risk_only_source_mode",
        choices=("auto", "trained"),
        default="auto",
        help="How to label Risk-only in Table II.",
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
        "--additive_all_csv",
        default="",
        help="Optional Additive-Fusion all-metrics CSV. Empty keeps the existing five-method paper table unchanged.",
    )
    parser.add_argument(
        "--velocity_search_all_csv",
        default="",
        help="Optional Velocity-Search all-metrics CSV. Empty keeps the existing five-method paper table unchanged.",
    )
    parser.add_argument(
        "--velocity_search_preset_root",
        default="agents/eval_data_velocity_search_dynamic_window_preset_validation",
        help="Dynamic-window bounded validation root for the DWA-style Velocity-Search diagnostic table.",
    )
    parser.add_argument(
        "--heldout_layout_root",
        default="agents/eval_data_heldout_irregular_rows",
        help="Eval-only held-out irregular-row root; latest heldout_table all-metrics CSV is used when present.",
    )
    parser.add_argument(
        "--heldout_layout_all_csv",
        default="",
        help="Optional all-metrics CSV for the held-out irregular-row eval-only table.",
    )
    parser.add_argument(
        "--heldout_table_only",
        action="store_true",
        help="Only build the held-out irregular-row table and notes.",
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
        "--learnedw_split_all_csv",
        default="agents/eval_data_fig6_trajectories_worldxy/pcr_main_table/pcr_main_table_all_metrics_20260612_044839.csv",
        help="Current Learned-w diagnostic CSV containing the full C_unsafe component split for Table A5.",
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
        "--mono_stage_snapshot_csv",
        default="agents/eval_data_mono_targetview_stage_probe/table3_stage_probe_snapshot_20260604.csv",
        help="Audited fallback snapshot used when stage2/3 metrics.json files are not present locally.",
    )
    parser.add_argument(
        "--learnedw_mechanism_dir",
        default="agents/eval_data/s_0.6/moe_teacher_s_pcr_line_avoid_basic_learnedw2_signed_lam0.3_gam0.15_m0.05_rowrel_aux0.05_riskmem_lc0.4_seed1_20260526_204857",
    )
    parser.add_argument("--mechanism_speed", type=float, default=0.60)
    parser.add_argument("--paper_train_num_envs", type=int, default=256)
    parser.add_argument("--paper_train_lr", type=str, default="6e-5")
    parser.add_argument("--paper_train_iters", type=int, default=1000)
    parser.add_argument("--learnedw_ckpt", default="agents/moe_teacher_best_learnedw.pt")
    parser.add_argument("--risk_only_ckpt", default="agents/moe_teacher_best_risk_only.pt")
    parser.add_argument("--avoid_ckpt", default="agents/avoid_best.pt")
    parser.add_argument(
        "--fig6_timeseries_root",
        default=None,
        help=(
            "Root directory/directories containing Fig.6 trajectory eval outputs with timeseries.csv files. "
            "Use comma or ':' separated roots; if omitted, roots are inferred from main-table CSV sources."
        ),
    )
    parser.add_argument(
        "--fig6_required",
        action="store_true",
        help="Fail if Fig.6 trajectory data is missing or incomplete.",
    )
    parser.add_argument(
        "--fig6_episode_id",
        type=int,
        default=-1,
        help="Common episode_id to draw for Fig.6. Default auto-selects a high-conflict episode.",
    )
    parser.add_argument(
        "--fig6_candidate_csv",
        default="agents/fig6_trajectory_pool_audit/fig6_candidate_sets.csv",
        help="Candidate CSV from audit_fig6_trajectory_pool.py; overrides automatic Fig.6 episode selection when present.",
    )
    parser.add_argument(
        "--fig6_candidate_rank",
        type=int,
        default=0,
        help="Candidate row rank to draw from --fig6_candidate_csv.",
    )
    parser.add_argument(
        "--fig6_y_min",
        type=float,
        default=-2.0,
        help="Optional lower y limit for Fig.6 interaction window.",
    )
    parser.add_argument(
        "--fig6_y_max",
        type=float,
        default=8.0,
        help="Optional upper y limit for Fig.6 interaction window.",
    )
    parser.add_argument(
        "--fig6_x_min",
        type=float,
        default=-1.5,
        help="Optional lower x limit for Fig.6 interaction window.",
    )
    parser.add_argument(
        "--fig6_x_max",
        type=float,
        default=1.5,
        help="Optional upper x limit for Fig.6 interaction window.",
    )
    parser.add_argument(
        "--fig6_x_margin",
        type=float,
        default=0.8,
        help="Lateral padding for Fig.6 interaction window.",
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
    if bool(getattr(args, "heldout_table_only", False)):
        _build_table_heldout_irregular(args)
        print(f"Held-out irregular-row table written to: {args.output_dir}")
        return
    table1 = _build_table1(args)
    table2 = _build_table2(args)
    table3 = _build_table3(args)
    _plot_fig3(args, table1)
    _plot_fig3b_scatter_sized(args, table1)
    _plot_fig4(args)
    _plot_fig6(args)
    _copy_legacy_mechanism_png(args)
    _build_appendix_a1(args)
    _build_appendix_a2(args)
    _build_appendix_a3(args)
    _build_appendix_a4(args)
    table_a5 = _build_table_a5(args)
    _build_table_dwa_diagnostic(args)
    table_heldout = _build_table_heldout_irregular(args)
    _write_manifest(args, table1, table2, table3, table_a5, table_heldout)
    print(f"Final paper outputs v3 written to: {args.output_dir}")


if __name__ == "__main__":
    main()
