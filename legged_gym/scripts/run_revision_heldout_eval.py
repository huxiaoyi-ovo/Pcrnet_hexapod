#!/usr/bin/env python3
"""Prepare the fixed RA-L revision held-out evaluation grid without training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYOUT_ID = "revision_heldout_mixed_v1"
METHODS: Dict[str, Dict] = {
    "learnedw": {"ckpt": "learnedw_ckpt", "flags": ["--wlearned2", "--risk_memory", "--risk_memory_l_clear", "0.40", "--risk_memory_velocity_source", "body", "--pcr_w_aux_enable", "--pcr_w_aux_coef", "0.05", "--pcr_w_aux_risk_f_threshold", "0.25", "--pcr_w_aux_risk_margin", "0.05", "--pcr_w_aux_cmd_cos_threshold", "0.5"]},
    "additive_fusion": {"ckpt": "yonly_ckpt", "flags": ["--yonly", "--additive_fusion"]},
    "rule_override": {"ckpt": "yonly_ckpt", "flags": ["--yonly", "--rule_override", "--rule_k", "8", "--rule_margin", "0.10", "--rule_hard_thr", "0.45", "--rule_s_min", "0.85", "--rule_slow_ratio", "0.10", "--rule_yaw_keep_loss", "0.30"]},
    "mono_ppo": {"ckpt": "mono_ppo_ckpt", "flags": ["--mono_ppo"]},
}


def _command(args, method: str, speed: float, seed: int) -> List[str]:
    spec = METHODS[method]
    return [
        sys.executable, "legged_gym/scripts/eval_highlevel.py", "--task", "s_pcr_line_avoid_basic",
        "--mode", "teacher", "--skill", "moe", "--pcr_ckpt", str(getattr(args, spec["ckpt"])),
        "--avoid_ckpt", args.avoid_ckpt, "--lowlevel_ckpt", args.lowlevel_ckpt,
        "--num_envs", str(args.num_envs), "--episodes", "128", "--seed", str(seed),
        "--output_dir", os.path.join(args.output_root, method, f"speed_{speed:.2f}"),
        "--eval_layout", LAYOUT_ID, "--avoid_stage_override", "4", "--freeze_avoid_stage",
        "--pcr_line_target_speed", f"{speed:.2f}", "--headless", *spec["flags"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--output_root", default="agents/eval_revision_heldout_mixed_v1")
    parser.add_argument("--learnedw_ckpt", default="agents/moe_teacher_best_learnedw.pt")
    parser.add_argument("--yonly_ckpt", default="agents/moe_teacher_best_yonly.pt")
    parser.add_argument("--mono_ppo_ckpt", default="agents/mono_ppo_best.pt")
    parser.add_argument("--avoid_ckpt", default="agents/avoid_best.pt")
    parser.add_argument("--lowlevel_ckpt", default="agents/low_level_best.pt")
    args = parser.parse_args()
    jobs = [(method, speed, seed) for method in METHODS for speed in (0.35, 0.50, 0.60) for seed in (1, 2, 3)]
    if len(jobs) != 36:
        raise RuntimeError(f"fixed protocol must contain 36 jobs, got {len(jobs)}")
    print(f"[RevisionHeldoutEval] layout={LAYOUT_ID} jobs={len(jobs)} episodes=128")
    for method, speed, seed in jobs:
        command = _command(args, method, speed, seed)
        print("[RevisionHeldoutEval] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
