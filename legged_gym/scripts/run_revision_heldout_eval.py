#!/usr/bin/env python3
"""Prepare the fixed RA-L revision held-out evaluation grid without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYOUT_ID = "revision_heldout_mixed_v1"
LAYOUT_PATH = os.path.join(
    PROJECT_ROOT, "legged_gym", "envs", "hex_v4", "layouts", f"{LAYOUT_ID}.json"
)
METHODS: Dict[str, Dict] = {
    "learnedw": {"ckpt": "learnedw_ckpt", "flags": ["--wlearned2", "--risk_memory", "--risk_memory_l_clear", "0.40", "--risk_memory_velocity_source", "body", "--pcr_w_aux_enable", "--pcr_w_aux_coef", "0.05", "--pcr_w_aux_risk_f_threshold", "0.25", "--pcr_w_aux_risk_margin", "0.05", "--pcr_w_aux_cmd_cos_threshold", "0.5"]},
    "additive_fusion": {"ckpt": "yonly_ckpt", "flags": ["--yonly", "--additive_fusion"]},
    "rule_override": {"ckpt": "yonly_ckpt", "flags": ["--yonly", "--rule_override", "--rule_k", "8", "--rule_margin", "0.10", "--rule_hard_thr", "0.45", "--rule_s_min", "0.85", "--rule_slow_ratio", "0.10", "--rule_yaw_keep_loss", "0.30"]},
    "mono_ppo": {"ckpt": "mono_ppo_ckpt", "flags": ["--mono_ppo"]},
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layout_metadata() -> Dict[str, str]:
    with open(LAYOUT_PATH, "r", encoding="utf-8") as handle:
        layout = json.load(handle)
    payload = {key: value for key, value in layout.items() if key != "sha256"}
    actual_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_hash = str(layout.get("sha256", ""))
    if layout.get("layout_id") != LAYOUT_ID or expected_hash != actual_hash:
        raise RuntimeError(
            f"layout identity mismatch: id={layout.get('layout_id')!r}, "
            f"stored_sha256={expected_hash}, actual_sha256={actual_hash}"
        )
    return {"layout_id": LAYOUT_ID, "layout_sha256": actual_hash}


def _metrics_files(path: str) -> List[str]:
    found: List[str] = []
    if not os.path.isdir(path):
        return found
    for root, _, files in os.walk(path):
        if "metrics.json" in files:
            found.append(os.path.join(root, "metrics.json"))
    return sorted(found)


def _validate_metrics(
    path: str,
    args,
    method: str,
    speed: float,
    seed: int,
    layout_sha256: str,
) -> None:
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    protocol = metrics.get("protocol", {})
    overall = metrics.get("overall", {})
    per_episode = metrics.get("per_episode", [])
    expected_quota = [
        128 // args.num_envs + int(env_id < 128 % args.num_envs)
        for env_id in range(args.num_envs)
    ]
    counts = [0 for _ in range(args.num_envs)]
    for row in per_episode:
        env_id = int(row.get("env_id", -1))
        if env_id < 0 or env_id >= args.num_envs:
            raise RuntimeError(f"invalid env_id in {path}: {env_id}")
        counts[env_id] += 1

    checks = {
        "eval_layout": protocol.get("eval_layout") == LAYOUT_ID,
        "layout_sha256": protocol.get("layout_sha256") == layout_sha256,
        "seed": int(protocol.get("seed", -1)) == seed,
        "difficulty_levels": protocol.get("difficulty_levels") == [0.0],
        "episode_sampling": protocol.get("episode_sampling") == "balanced_per_env_quota",
        "num_envs": int(protocol.get("num_envs", -1)) == args.num_envs,
        "episodes": int(overall.get("episodes", -1)) == 128 and len(per_episode) == 128,
        "per_env_quota": counts == expected_quota,
        "recorded_episodes_by_env": protocol.get("recorded_episodes_by_env") == expected_quota,
        "speed": abs(float(protocol.get("resolved_moving_target_pcr_line_speed", -1.0)) - speed) < 1e-9,
        "pcr_ckpt": os.path.abspath(str(protocol.get("pcr_ckpt", "")))
        == os.path.abspath(str(getattr(args, METHODS[method]["ckpt"]))),
        "avoid_ckpt": os.path.abspath(str(protocol.get("avoid_ckpt", "")))
        == os.path.abspath(args.avoid_ckpt),
        "lowlevel_ckpt": os.path.abspath(str(protocol.get("lowlevel_ckpt", "")))
        == os.path.abspath(args.lowlevel_ckpt),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"result validation failed for {path}: {', '.join(failed)}")


def _command(args, method: str, speed: float, seed: int) -> List[str]:
    spec = METHODS[method]
    return [
        sys.executable, "legged_gym/scripts/eval_highlevel.py", "--task", "s_pcr_line_avoid_basic",
        "--mode", "teacher", "--skill", "moe", "--pcr_ckpt", str(getattr(args, spec["ckpt"])),
        "--avoid_ckpt", args.avoid_ckpt, "--lowlevel_ckpt", args.lowlevel_ckpt,
        "--num_envs", str(args.num_envs), "--episodes", "128", "--seed", str(seed),
        "--difficulty_levels", "0.0", "--balanced_env_episodes",
        "--output_dir", os.path.join(args.output_root, method, f"speed_{speed:.2f}"),
        "--eval_layout", LAYOUT_ID, "--avoid_stage_override", "4", "--freeze_avoid_stage",
        "--pcr_line_target_speed", f"{speed:.2f}", "--headless", *spec["flags"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--learnedw_ckpt", default="agents/moe_teacher_best_learnedw.pt")
    parser.add_argument("--yonly_ckpt", default="agents/moe_teacher_best_yonly.pt")
    parser.add_argument("--mono_ppo_ckpt", default="agents/mono_ppo_best.pt")
    parser.add_argument("--avoid_ckpt", default="agents/avoid_best.pt")
    parser.add_argument("--lowlevel_ckpt", default="agents/low_level_best.pt")
    args = parser.parse_args()
    layout_meta = _layout_metadata()
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.output_root is None:
        args.output_root = os.path.join(
            "agents", f"eval_{LAYOUT_ID}_{layout_meta['layout_sha256'][:12]}"
        )
    jobs = [(method, speed, seed) for method in METHODS for speed in (0.35, 0.50, 0.60) for seed in (1, 2, 3)]
    if len(jobs) != 36:
        raise RuntimeError(f"fixed protocol must contain 36 jobs, got {len(jobs)}")
    print(
        f"[RevisionHeldoutEval] layout={LAYOUT_ID} sha256={layout_meta['layout_sha256']} "
        f"jobs={len(jobs)} episodes=128 sampling=balanced_per_env_quota"
    )
    if not args.dry_run:
        if os.path.exists(args.output_root) and os.listdir(args.output_root):
            raise RuntimeError(
                f"output root must be new and empty to prevent stale-result mixing: {args.output_root}"
            )
        os.makedirs(args.output_root, exist_ok=True)
        checkpoint_paths = {
            name: os.path.abspath(str(getattr(args, name)))
            for name in ("learnedw_ckpt", "yonly_ckpt", "mono_ppo_ckpt", "avoid_ckpt", "lowlevel_ckpt")
        }
        missing = [path for path in checkpoint_paths.values() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"missing checkpoints: {missing}")
        manifest = {
            "layout_id": LAYOUT_ID,
            "layout_sha256": layout_meta["layout_sha256"],
            "episodes_per_cell": 128,
            "num_envs": args.num_envs,
            "difficulty_levels": [0.0],
            "episode_sampling": "balanced_per_env_quota",
            "methods": list(METHODS),
            "speeds": [0.35, 0.50, 0.60],
            "seeds": [1, 2, 3],
            "checkpoints": {
                name: {"path": path, "sha256": _sha256_file(path)}
                for name, path in checkpoint_paths.items()
            },
        }
        with open(os.path.join(args.output_root, "RUN_MANIFEST.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    for method, speed, seed in jobs:
        command = _command(args, method, speed, seed)
        print("[RevisionHeldoutEval] " + " ".join(command), flush=True)
        if not args.dry_run:
            output_dir = os.path.join(args.output_root, method, f"speed_{speed:.2f}")
            before = set(_metrics_files(output_dir))
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            created = set(_metrics_files(output_dir)) - before
            if len(created) != 1:
                raise RuntimeError(
                    f"expected exactly one new metrics.json for {method}/{speed:.2f}/seed{seed}, "
                    f"got {sorted(created)}"
                )
            result_path = next(iter(created))
            _validate_metrics(
                result_path, args, method, speed, seed, layout_meta["layout_sha256"]
            )
            print(f"[RevisionHeldoutEval] validated {result_path}", flush=True)


if __name__ == "__main__":
    main()
