#!/usr/bin/env python3
"""Aggregate multiple eval metrics.json files into CSV tables."""

import os
import csv
import json
import math
import argparse
from typing import Dict, List, Optional, Tuple


PAPER_METRICS: List[Tuple[str, str]] = [
    ("Task Success", "task_success_rate"),
    ("Collision", "episode_collision_rate"),
    ("Follow MAE", "follow_mae_m_mean"),
    ("Unsafe Rate", "unsafe_conflict_step_rate"),
    ("C_avoid Rate", "avoid_conflict_step_rate"),
    ("CSI@C_avoid", "avoid_conflict_suppression_index"),
]
METHOD_ORDER = [
    "yonly",
    "geomw",
    "risk_only",
    "rule_override",
    "additive_fusion",
    "velocity_search",
    "mono_ppo",
    "learnedw",
]


def _find_metrics_json(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isfile(p) and p.endswith("metrics.json"):
            out.append(p)
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f == "metrics.json":
                        out.append(os.path.join(root, f))
    return sorted(set(out))


def _flatten(j: Dict, src_path: str) -> Dict:
    protocol = j.get("protocol", {})
    overall = j.get("overall", {})
    params = j.get("params", {})
    row = {
        "source": src_path,
        "task": protocol.get("task", ""),
        "mode": protocol.get("mode", ""),
        "skill": protocol.get("skill", ""),
        "seed": protocol.get("seed", ""),
        "policy_variant": protocol.get("policy_variant", ""),
        "velocity_search_split": protocol.get("velocity_search_split", ""),
        "velocity_search_hparams_source": protocol.get("velocity_search_hparams_source", ""),
        "velocity_search_hparams_json": protocol.get("velocity_search_hparams_json", ""),
        "velocity_search_hparams_frozen_for_eval": protocol.get("velocity_search_hparams_frozen_for_eval", ""),
        "beta": protocol.get("beta", ""),
        "w_mode": protocol.get("w_mode", ""),
        "w_tau": protocol.get("w_tau", ""),
        "w_blend_mode": protocol.get("w_blend_mode", ""),
        "episodes": overall.get("episodes", ""),
        "success_rate": overall.get("success_rate", ""),
        "l1_safety_success_rate": overall.get("l1_safety_success_rate", ""),
        "l2_follow_success_rate": overall.get("l2_follow_success_rate", ""),
        "l3_progress_success_rate": overall.get("l3_progress_success_rate", ""),
        "fail_ratio": overall.get("fail_ratio", ""),
        "follow_mae_m_mean": overall.get("follow_mae_m_mean", ""),
        "follow_rmse_m_mean": overall.get("follow_rmse_m_mean", ""),
        "tts_mean_s": overall.get("time_to_success_s_mean", ""),
        "tts_median_s": overall.get("time_to_success_s_median", ""),
        "tts_p95_s": overall.get("time_to_success_s_p95", ""),
        "cot_mean": overall.get("cot_mean", ""),
        "cot_median": overall.get("cot_median", ""),
        "cot_p95": overall.get("cot_p95", ""),
        "episode_collision_rate": overall.get("episode_collision_rate", ""),
        "progress_rate": overall.get("progress_rate", ""),
        "progress_ratio_mean": overall.get("progress_ratio_mean", ""),
        "gate_y_raw_mean": overall.get("gate_y_raw_mean", ""),
        "y_eff_mean": overall.get("y_eff_mean", ""),
        "w_mean": overall.get("w_mean", ""),
        "clearance_f_mean": overall.get("clearance_f_mean", ""),
        "clearance_a_mean": overall.get("clearance_a_mean", ""),
        "risk_f_mean": overall.get("risk_f_mean", ""),
        "risk_a_mean": overall.get("risk_a_mean", ""),
        "risk_delta_mean": overall.get("risk_delta_mean", ""),
        "risk_rollout_f_mean": overall.get("risk_rollout_f_mean", ""),
        "risk_rollout_a_mean": overall.get("risk_rollout_a_mean", ""),
        "risk_rollout_s_mean": overall.get("risk_rollout_s_mean", ""),
        "risk_rollout_gap_f_min_as_mean": overall.get("risk_rollout_gap_f_min_as_mean", ""),
        "rule_s_mean": overall.get("rule_s_mean", ""),
        "rule_follow_scale_mean": overall.get("rule_follow_scale_mean", ""),
        "rule_yaw_scale_mean": overall.get("rule_yaw_scale_mean", ""),
        "rule_follow_suppression_mean": overall.get("rule_follow_suppression_mean", ""),
        "rule_s_at_avoid_conflict": overall.get("rule_s_at_avoid_conflict", ""),
        "rule_follow_scale_at_avoid_conflict": overall.get("rule_follow_scale_at_avoid_conflict", ""),
        "rule_yaw_scale_at_avoid_conflict": overall.get("rule_yaw_scale_at_avoid_conflict", ""),
        "rule_follow_suppression_at_avoid_conflict": overall.get("rule_follow_suppression_at_avoid_conflict", ""),
        "delta_y_w_raw_mean": overall.get("delta_y_w_raw_mean", ""),
        "delta_y_w_used_mean": overall.get("delta_y_w_used_mean", ""),
        "delta_y_r_mean": overall.get("delta_y_r_mean", ""),
        "delta_y_total_mean": overall.get("delta_y_total_mean", ""),
        "unsafe_conflict_delta_y_w_raw_mean": overall.get("unsafe_conflict_delta_y_w_raw_mean", ""),
        "unsafe_conflict_delta_y_w_used_mean": overall.get("unsafe_conflict_delta_y_w_used_mean", ""),
        "unsafe_conflict_delta_y_r_mean": overall.get("unsafe_conflict_delta_y_r_mean", ""),
        "unsafe_conflict_delta_y_total_mean": overall.get("unsafe_conflict_delta_y_total_mean", ""),
        "avoid_conflict_delta_y_w_raw_mean": overall.get("avoid_conflict_delta_y_w_raw_mean", ""),
        "avoid_conflict_delta_y_w_used_mean": overall.get("avoid_conflict_delta_y_w_used_mean", ""),
        "avoid_conflict_delta_y_r_mean": overall.get("avoid_conflict_delta_y_r_mean", ""),
        "avoid_conflict_delta_y_total_mean": overall.get("avoid_conflict_delta_y_total_mean", ""),
        "switch_rate_mean": overall.get("switch_rate_mean", ""),
        "near_miss_rate_mean": overall.get("near_miss_rate_mean", ""),
        "cmd_jerk_lin_mean": overall.get("cmd_jerk_lin_mean", ""),
        "cmd_jerk_ang_mean": overall.get("cmd_jerk_ang_mean", ""),
        "target_in_rgb_fov_rate": overall.get("target_in_rgb_fov_rate", ""),
        "target_lost_step_rate": overall.get("target_lost_step_rate", ""),
        "target_lost_episode_rate": overall.get("target_lost_episode_rate", ""),
        "target_bearing_abs_deg_mean": overall.get("target_bearing_abs_deg_mean", ""),
        "target_bearing_abs_deg_p95": overall.get("target_bearing_abs_deg_p95", ""),
        "cmd_x_mean": overall.get("cmd_x_mean", ""),
        "cmd_x_std": overall.get("cmd_x_std", ""),
        "cmd_x_abs_mean": overall.get("cmd_x_abs_mean", ""),
        "cmd_x_positive_rate": overall.get("cmd_x_positive_rate", ""),
        "cmd_x_negative_rate": overall.get("cmd_x_negative_rate", ""),
        "cmd_x_zero_rate": overall.get("cmd_x_zero_rate", ""),
        "cmd_x_direction_entropy": overall.get("cmd_x_direction_entropy", ""),
        "corr_goalx_cmdx": overall.get("corr_goalx_cmdx", ""),
        "corr_goalx_cmdyaw": overall.get("corr_goalx_cmdyaw", ""),
        "corr_goaly_cmdy": overall.get("corr_goaly_cmdy", ""),
        "cmd_yaw_left_mean": overall.get("cmd_yaw_left_mean", ""),
        "cmd_yaw_right_mean": overall.get("cmd_yaw_right_mean", ""),
        "cmd_x_left_mean": overall.get("cmd_x_left_mean", ""),
        "cmd_x_right_mean": overall.get("cmd_x_right_mean", ""),
        "yaw_directional_response": overall.get("yaw_directional_response", ""),
        "lateral_directional_response": overall.get("lateral_directional_response", ""),
        "cmd_post_delta_abs_mean": overall.get("cmd_post_delta_abs_mean", ""),
        "cmd_post_delta_x_abs_mean": overall.get("cmd_post_delta_x_abs_mean", ""),
        "cmd_post_delta_y_abs_mean": overall.get("cmd_post_delta_y_abs_mean", ""),
        "cmd_post_delta_y_signed_mean": overall.get("cmd_post_delta_y_signed_mean", ""),
        "cmd_post_delta_w_abs_mean": overall.get("cmd_post_delta_w_abs_mean", ""),
        "cmd_post_rewrite_rate": overall.get("cmd_post_rewrite_rate", ""),
        "velocity_search_filter_change_rate": overall.get("velocity_search_filter_change_rate", ""),
        "velocity_search_safe_available_rate": overall.get("velocity_search_safe_available_rate", ""),
        "velocity_search_raw_risk_mean": overall.get("velocity_search_raw_risk_mean", ""),
        "velocity_search_safe_risk_mean": overall.get("velocity_search_safe_risk_mean", ""),
        "velocity_search_compute_time_ms_mean": overall.get("velocity_search_compute_time_ms_mean", ""),
        "velocity_search_dynamic_window_enabled_rate": overall.get("velocity_search_dynamic_window_enabled_rate", ""),
        "velocity_search_candidate_count_before_dynamic_window_mean": overall.get("velocity_search_candidate_count_before_dynamic_window_mean", ""),
        "velocity_search_candidate_count_after_dynamic_window_mean": overall.get("velocity_search_candidate_count_after_dynamic_window_mean", ""),
        "velocity_search_dynamic_window_rejected_count_mean": overall.get("velocity_search_dynamic_window_rejected_count_mean", ""),
        "velocity_search_candidate_count_mean": overall.get("velocity_search_candidate_count_mean", ""),
        "velocity_search_feasible_count_mean": overall.get("velocity_search_feasible_count_mean", ""),
        "velocity_search_infeasible_count_mean": overall.get("velocity_search_infeasible_count_mean", ""),
        "velocity_search_feasible_count_after_margin_mean": overall.get("velocity_search_feasible_count_after_margin_mean", ""),
        "velocity_search_selected_v_lat_mean": overall.get("velocity_search_selected_v_lat_mean", ""),
        "velocity_search_selected_v_fwd_mean": overall.get("velocity_search_selected_v_fwd_mean", ""),
        "velocity_search_selected_yaw_mean": overall.get("velocity_search_selected_yaw_mean", ""),
        "velocity_search_raw_y_mean": overall.get("velocity_search_raw_y_mean", ""),
        "velocity_search_after_filter_y_mean": overall.get("velocity_search_after_filter_y_mean", ""),
        "cmd_safe_y_mean": overall.get("cmd_safe_y_mean", ""),
        "cmd_safe_y_over_raw_y_ratio": overall.get("cmd_safe_y_over_raw_y_ratio", ""),
        "actual_base_vel_x_mean": overall.get("actual_base_vel_x_mean", ""),
        "actual_base_vel_y_mean": overall.get("actual_base_vel_y_mean", ""),
        "actual_base_vel_w_mean": overall.get("actual_base_vel_w_mean", ""),
        "actual_base_delta_x_mean": overall.get("actual_base_delta_x_mean", ""),
        "actual_base_delta_y_mean": overall.get("actual_base_delta_y_mean", ""),
        "row_progress_delta_mean": overall.get("row_progress_delta_mean", ""),
        "velocity_search_forward_candidate_count_mean": overall.get("velocity_search_forward_candidate_count_mean", ""),
        "velocity_search_forward_feasible_count_mean": overall.get("velocity_search_forward_feasible_count_mean", ""),
        "velocity_search_forward_infeasible_collision_count_mean": overall.get("velocity_search_forward_infeasible_collision_count_mean", ""),
        "velocity_search_forward_infeasible_margin_count_mean": overall.get("velocity_search_forward_infeasible_margin_count_mean", ""),
        "velocity_search_forward_infeasible_out_of_map_count_mean": overall.get("velocity_search_forward_infeasible_out_of_map_count_mean", ""),
        "velocity_search_best_forward_feasible_cost_mean": overall.get("velocity_search_best_forward_feasible_cost_mean", ""),
        "velocity_search_best_forward_feasible_clearance_mean": overall.get("velocity_search_best_forward_feasible_clearance_mean", ""),
        "velocity_search_fallback_rate": overall.get("velocity_search_fallback_rate", ""),
        "fallback_no_feasible_rate": overall.get("fallback_no_feasible_rate", ""),
        "fallback_risk_threshold_rate": overall.get("fallback_risk_threshold_rate", ""),
        "fallback_collision_rate": overall.get("fallback_collision_rate", ""),
        "fallback_margin_rate": overall.get("fallback_margin_rate", ""),
        "fallback_out_of_map_rate": overall.get("fallback_out_of_map_rate", ""),
        "fallback_invalid_cost_rate": overall.get("fallback_invalid_cost_rate", ""),
        "velocity_search_min_predicted_clearance_mean": overall.get("velocity_search_min_predicted_clearance_mean", ""),
        "velocity_search_best_cost_mean": overall.get("velocity_search_best_cost_mean", ""),
        "selected_rollout_min_clearance_mean": overall.get("selected_rollout_min_clearance_mean", ""),
        "selected_rollout_collision_rate": overall.get("selected_rollout_collision_rate", ""),
        "selected_rollout_out_of_map_rate": overall.get("selected_rollout_out_of_map_rate", ""),
        "w_clearance_f_corr": overall.get("w_clearance_f_corr", ""),
        "w_risk_f_corr": overall.get("w_risk_f_corr", ""),
        "w_risk_delta_corr": overall.get("w_risk_delta_corr", ""),
        "w_degen_clearance_like": overall.get("w_degen_clearance_like", ""),
        "w_trigger_rate": overall.get("w_trigger_rate", ""),
        "w_trigger_step_mean": overall.get("w_trigger_step_mean", ""),
        "w_trigger_progress_mean": overall.get("w_trigger_progress_mean", ""),
        "gate_region_y_eff_mean": overall.get("gate_region_y_eff_mean", ""),
        "gate_region_near_miss_rate_mean": overall.get("gate_region_near_miss_rate_mean", ""),
        "latency_p50_ms": overall.get("inference_latency_ms_p50", ""),
        "latency_p95_ms": overall.get("inference_latency_ms_p95", ""),
        "params_total": params.get("high_level_total", ""),
        "params_trainable": params.get("high_level_trainable", ""),
    }
    return row


def _safe_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(max(var, 0.0))


def _format_float(v: float, digits: int = 3) -> str:
    if not math.isfinite(float(v)):
        return ""
    return f"{float(v):.{digits}f}"


def _format_pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(float(mean)):
        return ""
    return f"{float(mean):.{digits}f} +/- {float(std):.{digits}f}"


def _method_from_protocol(protocol: Dict, src_path: str = "") -> str:
    variant = str(protocol.get("policy_variant", "") or "").strip().lower()
    w_mode = str(protocol.get("w_mode", "") or "").strip().lower()
    text = f"{variant} {w_mode} {src_path}".lower()
    if "velocity_search" in text or "velocity-search" in text:
        return "velocity_search"
    if "additive_fusion" in text or "additive-fusion" in text:
        return "additive_fusion"
    if "risk_only" in text or "risk-only" in text or bool(protocol.get("risk_only", False)):
        return "risk_only"
    if "rule_override" in text or "rule-override" in text:
        return "rule_override"
    if "mono_ppo" in text or "mono-ppo" in text:
        return "mono_ppo"
    if "learnedw2" in text or "learnedw" in text or w_mode in ("learned", "learnedw2"):
        return "learnedw"
    if "geomw" in text or w_mode == "geom":
        return "geomw"
    if "yonly" in text or w_mode == "none":
        return "yonly"
    return variant or w_mode or "unknown"


def _speed_from_protocol(protocol: Dict) -> str:
    for key in (
        "resolved_moving_target_pcr_line_speed",
        "pcr_line_target_speed",
    ):
        v = protocol.get(key, None)
        if v is None:
            continue
        f = _safe_float(v)
        if math.isfinite(f):
            return f"{f:.2f}"
    return ""


def _primary_checkpoint(metrics: Dict) -> str:
    protocol = metrics.get("protocol", {})
    for key in ("pcr_ckpt", "ckpt"):
        val = protocol.get(key, "")
        if val:
            return str(val)
    resolved = metrics.get("resolved_protocol", {})
    primary = resolved.get("primary_checkpoint", {}) if isinstance(resolved, dict) else {}
    if isinstance(primary, dict) and primary.get("path"):
        return str(primary["path"])
    return ""


def _paper_row(metrics: Dict, src_path: str) -> Dict:
    protocol = metrics.get("protocol", {})
    overall = metrics.get("overall", {})
    method = _method_from_protocol(protocol, src_path)
    conflict_metrics_available = bool(protocol.get("conflict_metrics_available", method != "mono_ppo"))
    mechanism_metrics_available = bool(protocol.get("mechanism_metrics_available", method != "mono_ppo"))
    row = {
        "Speed": _speed_from_protocol(protocol),
        "Method": method,
        "Seed": protocol.get("seed", ""),
        "Num Episodes": overall.get("episodes", protocol.get("episodes", "")),
        "Checkpoint": _primary_checkpoint(metrics),
        "Source": src_path,
    }
    for title, key in PAPER_METRICS:
        if title in ("Unsafe Rate", "C_avoid Rate") and not conflict_metrics_available:
            row[title] = ""
        elif title == "CSI@C_avoid" and not mechanism_metrics_available:
            row[title] = ""
        else:
            row[title] = overall.get(key, "")
    return row


def _dedupe_latest(rows: List[Dict]) -> List[Dict]:
    selected: Dict[Tuple[str, str, str], Dict] = {}
    for row in rows:
        key = (str(row.get("Speed", "")), str(row.get("Method", "")), str(row.get("Seed", "")))
        src = str(row.get("Source", ""))
        mtime = os.path.getmtime(src) if os.path.exists(src) else 0.0
        prev = selected.get(key)
        if prev is None:
            row["_mtime"] = mtime
            selected[key] = row
            continue
        if mtime >= float(prev.get("_mtime", 0.0)):
            row["_mtime"] = mtime
            selected[key] = row
    out = []
    for row in selected.values():
        row.pop("_mtime", None)
        out.append(row)
    return sorted(out, key=lambda r: (float(r.get("Speed", 0.0) or 0.0), _method_sort_key(str(r.get("Method", ""))), int(r.get("Seed", 0) or 0)))


def _method_sort_key(method: str) -> Tuple[int, str]:
    m = str(method).strip().lower()
    try:
        return METHOD_ORDER.index(m), m
    except ValueError:
        return len(METHOD_ORDER), m


def _write_paper_tables(metric_files: List[str], args) -> None:
    rows = []
    for p in metric_files:
        with open(p, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(_paper_row(metrics, p))
    if args.paper_dedupe_latest:
        rows = _dedupe_latest(rows)

    single_output = args.paper_single_output or _suffix_path(args.output, "_paper_single.csv")
    aggregate_output = args.paper_aggregate_output or _suffix_path(args.output, "_paper_aggregate.csv")
    markdown_output = args.paper_markdown_output or _suffix_path(args.output, "_paper_aggregate.md")

    _write_csv(single_output, rows, _paper_single_fieldnames())
    aggregate_rows = _aggregate_paper_rows(rows)
    _write_csv(aggregate_output, aggregate_rows, _paper_aggregate_fieldnames())
    _write_markdown_table(markdown_output, aggregate_rows)
    print(f"Paper single-seed CSV written: {single_output}")
    print(f"Paper aggregate CSV written: {aggregate_output}")
    print(f"Paper aggregate Markdown written: {markdown_output}")


def _suffix_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    if not ext:
        return path + suffix
    return root + suffix


def _paper_single_fieldnames() -> List[str]:
    return [
        "Speed",
        "Method",
        "Seed",
        *[title for title, _ in PAPER_METRICS],
        "Num Episodes",
        "Checkpoint",
        "Source",
    ]


def _paper_aggregate_fieldnames() -> List[str]:
    fields = ["Speed", "Method", "Seeds", "Num Runs", "Num Episodes Total"]
    for title, _ in PAPER_METRICS:
        fields.extend([f"{title} Mean", f"{title} Std", f"{title}"])
    fields.append("Sources")
    return fields


def _aggregate_paper_rows(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (str(row.get("Speed", "")), str(row.get("Method", "")))
        groups.setdefault(key, []).append(row)

    out = []
    for (speed, method), group in sorted(groups.items(), key=lambda kv: (float(kv[0][0] or 0.0), _method_sort_key(kv[0][1]))):
        agg = {
            "Speed": speed,
            "Method": method,
            "Seeds": ",".join(str(r.get("Seed", "")) for r in sorted(group, key=lambda r: int(r.get("Seed", 0) or 0))),
            "Num Runs": len(group),
            "Num Episodes Total": sum(int(float(r.get("Num Episodes", 0) or 0)) for r in group),
            "Sources": ";".join(str(r.get("Source", "")) for r in group),
        }
        for title, _ in PAPER_METRICS:
            mean, std = _mean_std([_safe_float(r.get(title, "")) for r in group])
            agg[f"{title} Mean"] = _format_float(mean)
            agg[f"{title} Std"] = _format_float(std)
            agg[title] = _format_pm(mean, std)
        out.append(agg)
    return out


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown_table(path: str, rows: List[Dict]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    headers = ["Speed", "Method", *[title for title, _ in PAPER_METRICS]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---:"] + ["---"] + ["---:"] * len(PAPER_METRICS)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval metrics into CSV")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories containing them")
    parser.add_argument("--output", type=str, default="outputs/eval/eval_summary.csv")
    parser.add_argument("--paper_main_table", action="store_true", help="also write PCR paper main-table CSV/Markdown")
    parser.add_argument("--paper_single_output", type=str, default=None)
    parser.add_argument("--paper_aggregate_output", type=str, default=None)
    parser.add_argument("--paper_markdown_output", type=str, default=None)
    parser.add_argument("--paper_dedupe_latest", action="store_true", default=True)
    parser.add_argument("--no_paper_dedupe_latest", action="store_false", dest="paper_dedupe_latest")
    args = parser.parse_args()

    metric_files = _find_metrics_json(args.paths)
    if not metric_files:
        raise RuntimeError("No metrics.json found in given paths")

    rows = []
    for p in metric_files:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        rows.append(_flatten(j, p))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Summary CSV written: {args.output}")
    print(f"Runs aggregated: {len(rows)}")
    if args.paper_main_table:
        _write_paper_tables(metric_files, args)


if __name__ == "__main__":
    main()
