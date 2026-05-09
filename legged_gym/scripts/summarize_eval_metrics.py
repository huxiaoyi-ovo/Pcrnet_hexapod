#!/usr/bin/env python3
"""Aggregate multiple eval metrics.json files into a single CSV table."""

import os
import csv
import json
import argparse
from typing import Dict, List


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
        "beta": protocol.get("beta", ""),
        "w_mode": protocol.get("w_mode", ""),
        "w_tau": protocol.get("w_tau", ""),
        "w_blend_mode": protocol.get("w_blend_mode", ""),
        "episodes": overall.get("episodes", ""),
        "success_rate": overall.get("success_rate", ""),
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
        "switch_rate_mean": overall.get("switch_rate_mean", ""),
        "near_miss_rate_mean": overall.get("near_miss_rate_mean", ""),
        "cmd_jerk_lin_mean": overall.get("cmd_jerk_lin_mean", ""),
        "cmd_jerk_ang_mean": overall.get("cmd_jerk_ang_mean", ""),
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


def main():
    parser = argparse.ArgumentParser(description="Summarize eval metrics into CSV")
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories containing them")
    parser.add_argument("--output", type=str, default="outputs/eval/eval_summary.csv")
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


if __name__ == "__main__":
    main()
