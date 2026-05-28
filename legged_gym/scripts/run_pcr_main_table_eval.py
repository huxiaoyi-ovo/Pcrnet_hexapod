#!/usr/bin/env python3
"""Run the PCR main-table eval grid and write paper-ready summaries."""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


METHODS: Dict[str, Dict] = {
    "yonly": {
        "ckpt_arg": "yonly_ckpt",
        "flags": ["--yonly"],
    },
    "geomw": {
        "ckpt_arg": "geomw_ckpt",
        "flags": ["--wgeom", "--w_tau", "0.15", "--w_blend_mode", "multiply", "--w_disable_gate_safe_clamp"],
    },
    "learnedw": {
        "ckpt_arg": "learnedw_ckpt",
        "flags": [
            "--wlearned2",
            "--w_blend_mode",
            "multiply",
            "--signed_w_lambda",
            "0.30",
            "--signed_w_gamma_risk",
            "0.15",
            "--signed_w_margin",
            "0.05",
            "--w_disable_gate_safe_clamp",
            "--risk_memory",
            "--risk_memory_l_clear",
            "0.40",
            "--risk_memory_velocity_source",
            "body",
            "--pcr_w_aux_enable",
            "--pcr_w_aux_coef",
            "0.05",
            "--pcr_w_aux_risk_f_threshold",
            "0.25",
            "--pcr_w_aux_risk_margin",
            "0.05",
            "--pcr_w_aux_cmd_cos_threshold",
            "0.5",
        ],
    },
}


def _parse_csv_floats(s: str) -> List[float]:
    out = []
    for token in str(s).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out = []
    for token in str(s).split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _parse_csv_methods(s: str) -> List[str]:
    out = []
    for token in str(s).split(","):
        method = token.strip().lower()
        if not method:
            continue
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}; expected one of {sorted(METHODS)}")
        out.append(method)
    return out


def _speed_tag(speed: float) -> str:
    return f"{float(speed):.2f}".rstrip("0").rstrip(".")


def _run(cmd: List[str], *, dry_run: bool, continue_on_error: bool) -> None:
    print("[PCRMainTableEval] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError:
        if not continue_on_error:
            raise
        print("[PCRMainTableEval] failed; continuing because --continue_on_error is set", flush=True)


def _eval_cmd(args, *, seed: int, speed: float, method: str) -> List[str]:
    method_cfg = METHODS[method]
    output_dir = os.path.join(args.output_root, f"s_{_speed_tag(speed)}")
    ckpt = getattr(args, method_cfg["ckpt_arg"])
    cmd = [
        sys.executable,
        "legged_gym/scripts/eval_highlevel.py",
        "--task",
        "s_pcr_line_avoid_basic",
        "--mode",
        "teacher",
        "--skill",
        "moe",
        "--pcr_ckpt",
        ckpt,
        "--avoid_ckpt",
        args.avoid_ckpt,
        "--lowlevel_ckpt",
        args.lowlevel_ckpt,
        "--num_envs",
        str(args.num_envs),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(seed),
        "--output_dir",
        output_dir,
        "--avoid_stage_override",
        "4",
        "--freeze_avoid_stage",
        "--pcr_line_target_speed",
        f"{float(speed):.2f}",
        "--dump_timeseries",
        "--timeseries_episodes",
        str(args.timeseries_episodes),
    ]
    cmd.extend(method_cfg["flags"])
    return cmd


def _summary_cmd(args, speeds: List[float]) -> List[str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_dir = args.summary_dir
    aggregate_path = os.path.join(summary_dir, f"pcr_main_table_aggregate_{ts}.csv")
    single_path = os.path.join(summary_dir, f"pcr_main_table_single_seed_{ts}.csv")
    markdown_path = os.path.join(summary_dir, f"pcr_main_table_aggregate_{ts}.md")
    all_metrics_path = os.path.join(summary_dir, f"pcr_main_table_all_metrics_{ts}.csv")
    paths = [os.path.join(args.output_root, f"s_{_speed_tag(speed)}") for speed in speeds]
    for token in str(args.extra_summary_paths or "").split(","):
        token = token.strip()
        if token:
            paths.append(token)
    return [
        sys.executable,
        "legged_gym/scripts/summarize_eval_metrics.py",
        *paths,
        "--output",
        all_metrics_path,
        "--paper_main_table",
        "--paper_single_output",
        single_path,
        "--paper_aggregate_output",
        aggregate_path,
        "--paper_markdown_output",
        markdown_path,
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCR main-table eval grid")
    parser.add_argument("--seeds", type=str, default="2,3")
    parser.add_argument("--speeds", type=str, default="0.35,0.50,0.60")
    parser.add_argument("--methods", type=str, default="yonly,geomw,learnedw")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--timeseries_episodes", type=int, default=64)
    parser.add_argument("--output_root", type=str, default="agents/eval_data_seed23")
    parser.add_argument("--summary_dir", type=str, default="agents/eval_data_seed23/pcr_main_table")
    parser.add_argument("--extra_summary_paths", type=str, default="", help="extra metrics dirs/files to include in final table")
    parser.add_argument("--yonly_ckpt", type=str, default="agents/moe_teacher_best_yonly.pt")
    parser.add_argument("--geomw_ckpt", type=str, default="agents/moe_teacher_best_w0.15.pt")
    parser.add_argument("--learnedw_ckpt", type=str, default="agents/moe_teacher_best_learnedw.pt")
    parser.add_argument("--avoid_ckpt", type=str, default="agents/avoid_best.pt")
    parser.add_argument("--lowlevel_ckpt", type=str, default="agents/low_level_best.pt")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = _parse_csv_ints(args.seeds)
    speeds = _parse_csv_floats(args.speeds)
    methods = _parse_csv_methods(args.methods)

    if not args.summary_only:
        for seed in seeds:
            for speed in speeds:
                for method in methods:
                    _run(
                        _eval_cmd(args, seed=seed, speed=speed, method=method),
                        dry_run=args.dry_run,
                        continue_on_error=args.continue_on_error,
                    )

    _run(_summary_cmd(args, speeds), dry_run=args.dry_run, continue_on_error=False)


if __name__ == "__main__":
    main()
