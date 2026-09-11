#!/usr/bin/env python3
"""Fail-closed, eval-only launcher for the frozen Track-B supplement."""

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "29ee797e208cece7c92beef22cb6ad5bcca4b1a3"
TENSORIZATION_SOURCE_COMMIT = "951cfbba44b34f5302ee5b6099f13e0f6f46344c"
VALIDATION_Y = tuple(i / 20.0 for i in range(21))
VALIDATION_SPEEDS = (0.35, 0.50)
FINAL_SPEEDS = (0.35, 0.50, 0.60)
FINAL_SEEDS = (1, 2, 3)
RISK_PAIRS = (
    ("nominal", 0.27, 0.57),
    ("closer", 0.22, 0.52),
    ("farther", 0.32, 0.62),
    ("narrow", 0.27, 0.47),
    ("wider", 0.27, 0.67),
)
V1_DIFFICULTY_LEVELS = (0.0, 0.25, 0.50, 0.75, 1.0)
V1_DIFFICULTY_CLI = "0.0,0.25,0.5,0.75,1.0"
LAYOUT_FLOAT_QUANTUM = 1.0e-5
LAYOUT_RESTORE_FLOAT_ATOL = 2.0e-5
PRIMARY_SOURCES = {
    "success_rate": "task_success_rate",
    "episode_collision_rate": "episode_collision_rate",
    "follow_mae_m_mean": "follow_mae_m_mean",
    "row_progress_success_mean": "row_progress_success_mean",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def _json_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _difficulty_counts(episodes: int):
    per_level = int(math.ceil(float(episodes) / float(len(V1_DIFFICULTY_LEVELS))))
    remaining = int(episodes)
    counts = []
    for _ in V1_DIFFICULTY_LEVELS:
        count = min(per_level, remaining)
        counts.append(count)
        remaining -= count
    if remaining != 0:
        raise RuntimeError("Track-B V1 difficulty allocation did not consume all episodes")
    return counts


def _matches_distance(actual, expected: float) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-9)
    except (TypeError, ValueError):
        return False


def _risk_mapping_matches(risk: dict, safe_d: float, free_d: float) -> bool:
    return (
        _matches_distance(risk.get("safe_distance"), safe_d)
        and _matches_distance(risk.get("free_distance"), free_d)
    )


def _nominal_postprocessor_matches(risk: dict) -> bool:
    return (
        _matches_distance(risk.get("postprocess_safe_distance"), 0.27)
        and _matches_distance(risk.get("postprocess_free_distance"), 0.57)
        and risk.get("postprocessor_unchanged") is True
    )


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    contract = {"schema": "track_b_csv_contract_v1", "rows": len(rows), "rows_sha256": _json_sha256(rows)}
    if path.exists() or contract_path.exists():
        if not path.is_file() or not contract_path.is_file():
            raise RuntimeError(f"partial/mismatched Track-B CSV output: {path}")
        with contract_path.open(encoding="utf-8") as handle:
            if json.load(handle) != contract:
                raise RuntimeError(f"Track-B CSV contract mismatch: {path}")
        return
    _dump_json(contract_path, contract)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _speed_tag(speed: float) -> str:
    return f"{float(speed):.2f}"


def _condition(method: str, y_const=None, risk_pair=None) -> str:
    if method == "fixed":
        if y_const is None:
            raise RuntimeError("Fixed condition lacks y")
        return f"y_{float(y_const):.2f}"
    if method == "additive":
        return "default"
    if risk_pair is None:
        return "nominal"
    name, safe_d, free_d = risk_pair
    return f"risk_{name}_dsafe_{safe_d:.2f}_dfree_{free_d:.2f}"


def _raw_path(root: Path, stage: str, method: str, speed: float, seed: int, *, y_const=None, risk_pair=None, condition_name=None) -> Path:
    condition = str(condition_name) if condition_name is not None else _condition(method, y_const, risk_pair)
    return root / "raw" / stage / method / condition / f"speed_{_speed_tag(speed)}" / f"seed_{int(seed)}"


def planned_paths(fixed_root: Path, risk_root: Path):
    paths = []
    for speed in VALIDATION_SPEEDS:
        for y_const in VALIDATION_Y:
            paths.append(_raw_path(fixed_root, "validation", "fixed", speed, 101, y_const=y_const))
    for speed in FINAL_SPEEDS:
        for seed in FINAL_SEEDS:
            for method in ("additive", "fixed", "pcr"):
                paths.append(_raw_path(
                    fixed_root, "final", method, speed, seed,
                    condition_name="y_frozen_selection" if method == "fixed" else None,
                ))
    for speed in FINAL_SPEEDS:
        for risk_pair in RISK_PAIRS:
            paths.append(_raw_path(risk_root, "risk", "pcr", speed, 101, risk_pair=risk_pair))
    paths.extend((
        _raw_path(fixed_root, "smoke", "fixed", 0.35, 101, y_const=0.50),
        _raw_path(fixed_root, "smoke", "additive", 0.35, 101),
        _raw_path(fixed_root, "smoke", "pcr", 0.35, 101),
        _raw_path(risk_root, "smoke", "pcr", 0.35, 101, risk_pair=RISK_PAIRS[1]),
    ))
    if len(paths) != 88 or len({str(path) for path in paths}) != 88:
        raise RuntimeError("Track-B planned paths are not exactly 88 unique locations")
    return paths


def validate_manifest_quota(manifest: dict, num_envs: int, episodes: int) -> None:
    """Pure admission check for immutable `(env_id, local_episode_slot)` pairing."""
    if manifest.get("schema") != "track_b_manifest_v3":
        raise RuntimeError("Track-B manifest schema mismatch")
    slots = int(manifest.get("slots_per_env", 0))
    if slots < 1 or episodes != num_envs * slots:
        raise RuntimeError("Track-B manifest per-env quota mismatch")
    entries = manifest.get("entries", [])
    expected = {(env_id, slot) for env_id in range(num_envs) for slot in range(slots)}
    got = {(int(entry.get("env_id", -1)), int(entry.get("local_episode_slot", -1))) for entry in entries}
    ids = [str(entry.get("manifest_entry_id", "")) for entry in entries]
    layout_ids = [str(entry.get("layout_id", "")) for entry in entries]
    expected_ids = {f"env_{env_id:03d}_slot_{slot}" for env_id, slot in expected}
    canonical_ids = [int(entry.get("canonical_episode_id", -1)) for entry in entries]
    expected_counts = _difficulty_counts(episodes)
    actual_counts = [sum(float(entry.get("difficulty", float("nan"))) == difficulty for entry in entries) for difficulty in V1_DIFFICULTY_LEVELS]
    expected_count_map = {f"{difficulty:.2f}": count for difficulty, count in zip(V1_DIFFICULTY_LEVELS, expected_counts)}
    layout_identity = {
        "normalized_physical_payload": True,
        "float_quantum": LAYOUT_FLOAT_QUANTUM,
        "restore_atol": LAYOUT_RESTORE_FLOAT_ATOL,
        "origin_roundtrip": "same_target_env_generation_and_replay",
    }
    generation = manifest.get("generation", {})
    if (
        got != expected or len(entries) != episodes or len(set(ids)) != episodes or set(ids) != expected_ids or not all(ids) or not all(layout_ids)
        or set(canonical_ids) != set(range(episodes))
        or tuple(float(value) for value in manifest.get("difficulty_levels", [])) != V1_DIFFICULTY_LEVELS
        or actual_counts != expected_counts or manifest.get("difficulty_counts") != expected_count_map
        or manifest.get("layout_identity") != layout_identity
        or generation.get("base_task_only") is not True
        or generation.get("policy_actions") != 0
        or generation.get("policy_checkpoints_loaded") is not False
    ):
        raise RuntimeError("Track-B manifest does not bind exactly one complete record per env/slot")


def _checkpoint_hashes(args) -> dict:
    return {str(Path(path).resolve()): _sha256(Path(path)) for path in (args.pcr_ckpt, args.avoid_ckpt, args.lowlevel_ckpt)}


def _root_contract(args, kind: str) -> dict:
    return {
        "schema": "track_b_root_contract_v3",
        "kind": kind,
        "base_commit": BASE_COMMIT,
        "tensorization_source_commit": TENSORIZATION_SOURCE_COMMIT,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "checkpoint_sha256": args.checkpoint_hashes,
        "checkpoint_paths": {key: str(Path(key)) for key in args.checkpoint_hashes},
        "formal_run": {"episodes": 128, "num_envs": 64, "difficulty_levels": list(V1_DIFFICULTY_LEVELS), "difficulty_counts": _difficulty_counts(128), "slots_per_env": 2},
        "smoke_run": {"episodes": 2, "num_envs": 2, "difficulty_levels": list(V1_DIFFICULTY_LEVELS), "difficulty_counts": _difficulty_counts(2), "slots_per_env": 1},
        "validation": {"speeds": list(VALIDATION_SPEEDS), "seeds": [101], "y_grid": list(VALIDATION_Y)},
        "final": {"methods": ["additive", "fixed", "pcr"], "speeds": list(FINAL_SPEEDS), "seeds": list(FINAL_SEEDS)},
        "risk": {"speeds": list(FINAL_SPEEDS), "seeds": [101], "pairs": [list(pair) for pair in RISK_PAIRS]},
        "zero_training": True,
        "selection_rule": "pooled equal-count episodes over 0.35 and 0.50: max task_success_rate, then min episode_collision_rate, then min follow_mae_m_mean, then smaller numeric y",
        "pairing": {
            "binding": "immutable per-env/local-slot manifest_entry_id plus canonical_episode_id with final normalized layout_id equality",
            "layout_float_quantum": LAYOUT_FLOAT_QUANTUM,
            "restore_float_atol": LAYOUT_RESTORE_FLOAT_ATOL,
            "origin_roundtrip": "same_target_env_generation_and_replay",
        },
    }


def _prepare_root(root: Path, contract: dict, paths) -> None:
    if root.exists() and any(root.iterdir()):
        saved = root / "provenance.json"
        if not saved.is_file():
            raise RuntimeError(f"nonempty root without Track-B provenance: {root}")
        with saved.open(encoding="utf-8") as handle:
            if json.load(handle) != contract:
                raise RuntimeError(f"root contract mismatch: {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    _dump_json(root / "provenance.json", contract)
    _dump_json(root / "created.json", {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root_contract_sha256": _json_sha256(contract)})
    own_paths = [str(path) for path in paths if str(path).startswith(str(root) + os.sep)]
    _dump_json(root / "planned_paths.json", {"count": len(own_paths), "paths": own_paths})


def _frozen_pcr_flags():
    return [
        "--wlearned2", "--w_blend_mode", "multiply", "--signed_w_lambda", "0.30",
        "--signed_w_gamma_risk", "0.15", "--signed_w_margin", "0.05",
        "--w_disable_gate_safe_clamp", "--risk_memory", "--risk_memory_l_clear", "0.40",
        "--risk_memory_velocity_source", "body", "--pcr_w_aux_enable", "--pcr_w_aux_coef", "0.05",
        "--pcr_w_aux_risk_f_threshold", "0.25", "--pcr_w_aux_risk_margin", "0.05",
        "--pcr_w_aux_cmd_cos_threshold", "0.5",
    ]


def _base_eval_command(args, speed: float, seed: int):
    return [
        args.python, "legged_gym/scripts/eval_highlevel.py", "--task", "s_pcr_line_avoid_basic",
        "--mode", "teacher", "--skill", "moe", "--pcr_ckpt", args.pcr_ckpt,
        "--avoid_ckpt", args.avoid_ckpt, "--lowlevel_ckpt", args.lowlevel_ckpt,
        "--num_envs", str(args.num_envs), "--episodes", str(args.episodes), "--seed", str(seed),
        "--difficulty_levels", args.difficulty_levels, "--avoid_stage_override", "4", "--freeze_avoid_stage",
        "--pcr_line_target_speed", _speed_tag(speed), "--headless",
    ]


def _assert_eval_only(cmd, hashes: dict = None) -> None:
    joined = " ".join(cmd)
    if "legged_gym/scripts/eval_highlevel.py" not in joined:
        raise RuntimeError("launcher permits eval_highlevel.py only")
    if any(token in joined for token in ("train_highlevel.py", "optimizer", "--resume", "--max_iterations")):
        raise RuntimeError("launcher rejected a training-related token")
    if hashes is not None:
        for path, expected in hashes.items():
            if _sha256(Path(path)) != expected:
                raise RuntimeError(f"checkpoint changed: {path}")


def _manifest_path(root: Path, stage: str, speed: float, seed: int) -> Path:
    return root / "manifests" / stage / f"speed_{_speed_tag(speed)}" / f"seed_{int(seed)}.json"


def _ensure_manifest(args, root: Path, stage: str, speed: float, seed: int) -> tuple:
    path = _manifest_path(root, stage, speed, seed)
    if not path.exists():
        # This generation command deliberately has no checkpoint arguments:
        # the eval entry builds only the base task and exits before any policy.
        cmd = [
            args.python, "legged_gym/scripts/eval_highlevel.py", "--task", "s_pcr_line_avoid_basic",
            "--mode", "teacher", "--skill", "moe", "--num_envs", str(args.num_envs),
            "--episodes", str(args.episodes), "--seed", str(seed), "--difficulty_levels", args.difficulty_levels,
            "--avoid_stage_override", "4", "--freeze_avoid_stage", "--pcr_line_target_speed", _speed_tag(speed),
            "--headless", "--track_b_manifest_generate", str(path),
        ]
        _assert_eval_only(cmd)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError("Track-B manifest missing or writable")
    digest = _sha256(path)
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest_quota(manifest, int(args.num_envs), int(args.episodes))
    return path, digest


def _point_contract(args, *, stage: str, method: str, speed: float, seed: int, y_const, risk_pair, manifest_sha: str) -> dict:
    return {
        "schema": "track_b_point_contract_v1",
        "git_commit": _git("rev-parse", "HEAD"),
        "checkpoint_sha256": args.checkpoint_hashes,
        "stage": stage,
        "method": method,
        "condition": _condition(method, y_const, risk_pair),
        "speed_mps": float(speed),
        "seed": int(seed),
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "difficulty_levels": str(args.difficulty_levels),
        "manifest_sha256": manifest_sha,
        "fixed_y_const": None if y_const is None else float(y_const),
        "risk_pair": None if risk_pair is None else list(risk_pair),
        "selection_sha256": getattr(args, "selection_sha256", None),
        "zero_training": True,
    }


def _run_one(args, root: Path, *, stage: str, method: str, speed: float, seed: int, y_const=None, risk_pair=None):
    output = _raw_path(root, stage, method, speed, seed, y_const=y_const, risk_pair=risk_pair)
    manifest, manifest_sha = _ensure_manifest(args, root, stage, speed, seed)
    point_contract = _point_contract(
        args, stage=stage, method=method, speed=speed, seed=seed, y_const=y_const,
        risk_pair=risk_pair, manifest_sha=manifest_sha,
    )
    point_contract_path = output.parent / f"{output.name}.point_contract.json"
    if output.exists() or point_contract_path.exists():
        if not output.is_dir() or not point_contract_path.is_file():
            raise RuntimeError(f"partial Track-B point refuses resume: {output}")
        with point_contract_path.open(encoding="utf-8") as handle:
            if json.load(handle) != point_contract:
                raise RuntimeError(f"Track-B point contract mismatch: {output}")
        return _row_from_metrics(args, output / "metrics.json", stage, method, speed, seed, y_const, risk_pair, manifest_sha)
    cmd = _base_eval_command(args, speed, seed) + [
        "--track_b_manifest", str(manifest), "--track_b_manifest_sha256", manifest_sha, "--output_dir", str(output),
    ]
    if stage != "validation":
        cmd += ["--dump_timeseries", "--timeseries_episodes", str(args.episodes), "--timeseries_stride", "1"]
    if method == "fixed":
        cmd += ["--fixed_authority", "--fixed_y_const", f"{float(y_const):.2f}", "--wlearned2"]
    elif method == "additive":
        cmd += ["--additive_fusion", "--wlearned2"]
    elif method == "pcr":
        cmd += _frozen_pcr_flags()
        if risk_pair is not None:
            _, safe_d, free_d = risk_pair
            cmd += ["--risk_mapping_safe_dist", f"{safe_d:.2f}", "--risk_mapping_free_dist", f"{free_d:.2f}"]
    else:
        raise RuntimeError(f"unsupported method {method}")
    _assert_eval_only(cmd, args.checkpoint_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    _dump_json(point_contract_path, point_contract)
    with (root / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "command": cmd}) + "\n")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    _assert_eval_only(cmd, args.checkpoint_hashes)
    return _row_from_metrics(args, output / "metrics.json", stage, method, speed, seed, y_const, risk_pair, manifest_sha)


def _row_from_metrics(args, metrics_path: Path, stage: str, method: str, speed: float, seed: int, y_const, risk_pair, manifest_sha: str):
    if not metrics_path.is_file():
        raise RuntimeError(f"missing metrics: {metrics_path}")
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    episodes = metrics.get("per_episode", [])
    if len(episodes) != int(args.episodes):
        raise RuntimeError("Track-B point lacks exactly requested complete episodes")
    pairs = [(str(row.get("manifest_entry_id", "")), str(row.get("layout_id", ""))) for row in episodes]
    canonical_rows = [
        (int(row.get("episode_id", -1)), str(row.get("manifest_entry_id", "")), str(row.get("layout_id", "")), float(row.get("difficulty", float("nan"))))
        for row in episodes
    ]
    if (
        not all(entry and layout for entry, layout in pairs)
        or len({entry for entry, _ in pairs}) != int(args.episodes)
        or {item[0] for item in canonical_rows} != set(range(int(args.episodes)))
    ):
        raise RuntimeError("Track-B point lacks unique manifest/layout provenance")
    pairs = sorted(pairs)
    canonical_rows = sorted(canonical_rows)
    if [sum(item[3] == difficulty for item in canonical_rows) for difficulty in V1_DIFFICULTY_LEVELS] != _difficulty_counts(int(args.episodes)):
        raise RuntimeError("Track-B point lacks the frozen V1 difficulty counts")
    resolved = metrics.get("resolved_protocol", {})
    if resolved.get("track_b_manifest", {}).get("sha256") != manifest_sha:
        raise RuntimeError("metrics manifest digest mismatch")
    if method == "fixed":
        fixed = resolved.get("track_b_fixed_authority", {})
        if fixed.get("gate_policy_call_count") != 0 or fixed.get("dynamic_fields_consumed") is not False:
            raise RuntimeError("Fixed Gate-bypass assertion failed")
    if risk_pair is not None:
        _, safe_d, free_d = risk_pair
        risk = resolved.get("track_b_risk_mapping", {})
        if not _risk_mapping_matches(risk, safe_d, free_d):
            raise RuntimeError("risk override did not reach learned-w mapping")
        if not _nominal_postprocessor_matches(risk):
            raise RuntimeError("risk override changed postprocessor distances")
    overall = metrics.get("overall", {})
    if stage != "validation" and not metrics_path.with_name("timeseries.csv").is_file():
        raise RuntimeError("Track-B raw proposal/post/termination trace is missing")
    row = {
        "stage": stage, "method": method, "condition": _condition(method, y_const, risk_pair),
        "speed_mps": speed, "seed": seed, "episodes": int(args.episodes), "y_const": y_const,
        "risk_setting": None if risk_pair is None else risk_pair[0],
        "risk_mapping_safe_distance": None if risk_pair is None else risk_pair[1],
        "risk_mapping_free_distance": None if risk_pair is None else risk_pair[2],
        "manifest_sha256": manifest_sha, "metrics_json": str(metrics_path),
        "timeseries_csv": str(metrics_path.with_name("timeseries.csv")), "manifest_layout_pairs": pairs,
        "canonical_manifest_layout_rows": canonical_rows,
    }
    for output_key, source_key in PRIMARY_SOURCES.items():
        value = overall.get(source_key)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"missing finite overall metric {source_key}")
        row[output_key] = float(value)
    return row


def _assert_paired(reference: dict, candidate: dict, label: str) -> None:
    if (
        reference["manifest_sha256"] != candidate["manifest_sha256"]
        or reference["manifest_layout_pairs"] != candidate["manifest_layout_pairs"]
        or reference["canonical_manifest_layout_rows"] != candidate["canonical_manifest_layout_rows"]
    ):
        raise RuntimeError(f"paired manifest/layout assertion failed: {label}")


def _selection(rows):
    candidates = []
    for y_const in VALIDATION_Y:
        items = [row for row in rows if row["y_const"] == y_const]
        if len(items) != len(VALIDATION_SPEEDS) or {item["speed_mps"] for item in items} != set(VALIDATION_SPEEDS):
            raise RuntimeError("incomplete Fixed validation candidate")
        total = sum(int(item["episodes"]) for item in items)
        candidates.append({
            "y_const": y_const, "episodes_total": total,
            **{metric: sum(float(item[metric]) * int(item["episodes"]) for item in items) / total for metric in PRIMARY_SOURCES},
        })
    selected = min(candidates, key=lambda row: (-row["success_rate"], row["episode_collision_rate"], row["follow_mae_m_mean"], row["y_const"]))
    return {
        "selection_rule": "pooled equal-count episodes at speeds 0.35 and 0.50; maximize task_success_rate, then minimize episode_collision_rate, then follow_mae_m_mean, then choose smaller numeric y",
        "metric_sources": PRIMARY_SOURCES,
        "selected": selected,
        "candidates": sorted(candidates, key=lambda row: row["y_const"]),
    }


def _selection_contract(args) -> dict:
    return {
        "schema": "track_b_selection_contract_v1",
        "git_commit": _git("rev-parse", "HEAD"),
        "checkpoint_sha256": args.checkpoint_hashes,
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "difficulty_levels": str(args.difficulty_levels),
        "validation_speeds": list(VALIDATION_SPEEDS),
        "validation_y": list(VALIDATION_Y),
        "selection_rule": "pooled equal-count episodes at speeds 0.35 and 0.50; maximize task_success_rate, then minimize episode_collision_rate, then follow_mae_m_mean, then choose smaller numeric y",
    }


def _summary(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    result = []
    for group, items in sorted(grouped.items()):
        row = dict(zip(group_fields, group))
        row["runs"] = len(items)
        row["episodes_total"] = sum(int(item["episodes"]) for item in items)
        for metric in PRIMARY_SOURCES:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = sum(values) / len(values)
            row[f"{metric}_std"] = math.sqrt(sum((x - row[f"{metric}_mean"]) ** 2 for x in values) / len(values))
        result.append(row)
    return result


def _load_selection(args, root: Path) -> tuple:
    path = root / "fixed_authority_selection.json"
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise RuntimeError("final stage requires frozen selection JSON")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_contract") != _selection_contract(args):
        raise RuntimeError("frozen selection contract mismatch")
    selected = payload.get("selected", {})
    y_const = selected.get("y_const")
    if y_const not in VALIDATION_Y:
        raise RuntimeError("invalid frozen y selection")
    return float(y_const), _sha256(path)


def _run_smokes(args, fixed_root: Path, risk_root: Path):
    smoke = argparse.Namespace(**vars(args))
    smoke.episodes, smoke.num_envs, smoke.difficulty_levels = 2, 2, V1_DIFFICULTY_CLI
    smoke.checkpoint_hashes = args.checkpoint_hashes
    rows = [
        _run_one(smoke, fixed_root, stage="smoke", method="fixed", speed=0.35, seed=101, y_const=0.50),
        _run_one(smoke, fixed_root, stage="smoke", method="additive", speed=0.35, seed=101),
        _run_one(smoke, fixed_root, stage="smoke", method="pcr", speed=0.35, seed=101),
        _run_one(smoke, risk_root, stage="smoke", method="pcr", speed=0.35, seed=101, risk_pair=RISK_PAIRS[1]),
    ]
    _assert_paired(rows[0], rows[1], "smoke fixed/additive")
    _assert_paired(rows[0], rows[2], "smoke fixed/pcr")
    _write_csv(fixed_root / "track_b_smoke.csv", rows[:3])
    _write_csv(risk_root / "track_b_risk_smoke.csv", rows[3:])


def _run_validation(args, fixed_root: Path):
    rows, refs = [], {}
    for speed in VALIDATION_SPEEDS:
        for y_const in VALIDATION_Y:
            row = _run_one(args, fixed_root, stage="validation", method="fixed", speed=speed, seed=101, y_const=y_const)
            if speed in refs:
                _assert_paired(refs[speed], row, f"validation speed={speed:.2f} y={y_const:.2f}")
            else:
                refs[speed] = row
            rows.append(row)
    _write_csv(fixed_root / "fixed_authority_validation.csv", rows)
    selection_path = fixed_root / "fixed_authority_selection.json"
    selection_payload = _selection(rows)
    selection_payload["selection_contract"] = _selection_contract(args)
    selection_payload["validation_rows_sha256"] = _json_sha256(rows)
    if selection_path.exists():
        if stat.S_IMODE(selection_path.stat().st_mode) & 0o222:
            raise RuntimeError("frozen selection JSON became writable")
        with selection_path.open(encoding="utf-8") as handle:
            if json.load(handle) != selection_payload:
                raise RuntimeError("frozen selection JSON does not match validation evidence")
        return
    _dump_json(selection_path, selection_payload)
    os.chmod(selection_path, 0o444)


def _run_final(args, fixed_root: Path):
    selected_y, selection_sha256 = _load_selection(args, fixed_root)
    final_args = argparse.Namespace(**vars(args))
    final_args.selection_sha256 = selection_sha256
    rows, refs = [], {}
    for speed in FINAL_SPEEDS:
        for seed in FINAL_SEEDS:
            for method in ("additive", "fixed", "pcr"):
                row = _run_one(final_args, fixed_root, stage="final", method=method, speed=speed, seed=seed, y_const=selected_y if method == "fixed" else None)
                key = (speed, seed)
                if key in refs:
                    _assert_paired(refs[key], row, f"final speed={speed:.2f} seed={seed} method={method}")
                else:
                    refs[key] = row
                rows.append(row)
    _write_csv(fixed_root / "additive_fixed_pcr_main_eval.csv", rows)
    _write_csv(fixed_root / "additive_fixed_pcr_summary.csv", _summary(rows, ("method", "speed_mps")))


def _run_risk(args, risk_root: Path):
    rows, refs = [], {}
    for speed in FINAL_SPEEDS:
        for risk_pair in RISK_PAIRS:
            row = _run_one(args, risk_root, stage="risk", method="pcr", speed=speed, seed=101, risk_pair=risk_pair)
            if speed in refs:
                _assert_paired(refs[speed], row, f"risk speed={speed:.2f} setting={risk_pair[0]}")
            else:
                refs[speed] = row
            rows.append(row)
    _write_csv(risk_root / "risk_threshold_sensitivity.csv", rows)
    _write_csv(risk_root / "risk_threshold_sensitivity_summary.csv", _summary(rows, ("risk_setting", "speed_mps", "risk_mapping_safe_distance", "risk_mapping_free_distance")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "validation", "final", "risk", "formal_sequence"), required=True)
    parser.add_argument("--fixed_output_root", required=True)
    parser.add_argument("--risk_output_root", required=True)
    parser.add_argument("--pcr_ckpt", required=True)
    parser.add_argument("--avoid_ckpt", required=True)
    parser.add_argument("--lowlevel_ckpt", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--difficulty_levels", default=V1_DIFFICULTY_CLI)
    return parser.parse_args()


def main():
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Track-B launcher requires CUDA_VISIBLE_DEVICES=0")
    if _git("status", "--porcelain"):
        raise RuntimeError("Track-B launcher requires a clean canonical worktree")
    if _git("rev-parse", "--abbrev-ref", "HEAD") != "codex/track-b-canonical-20260912":
        raise RuntimeError("Track-B launcher requires the dedicated canonical branch")
    if _git("rev-list", "--count", f"{BASE_COMMIT}..HEAD") != "1":
        raise RuntimeError("Track-B launcher requires exactly one canonical commit atop 29ee")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=PROJECT_ROOT).returncode != 0:
        raise RuntimeError("Track-B launcher must run from the exact canonical base-derived commit")
    if args.phase != "smoke" and (args.num_envs != 64 or args.episodes != 128 or args.difficulty_levels != V1_DIFFICULTY_CLI):
        raise RuntimeError("formal Track-B is frozen at 64 envs x 2 slots and the historical V1 difficulty list")
    if args.episodes % args.num_envs != 0:
        raise RuntimeError("episodes must divide exactly into per-env manifest slots")
    args.checkpoint_hashes = _checkpoint_hashes(args)
    fixed_root, risk_root = Path(args.fixed_output_root), Path(args.risk_output_root)
    paths = planned_paths(fixed_root, risk_root)
    _prepare_root(fixed_root, _root_contract(args, "fixed_authority"), paths)
    _prepare_root(risk_root, _root_contract(args, "risk_sensitivity"), paths)
    if args.phase == "smoke":
        _run_smokes(args, fixed_root, risk_root)
    elif args.phase == "validation":
        _run_validation(args, fixed_root)
    elif args.phase == "final":
        _run_final(args, fixed_root)
    elif args.phase == "risk":
        _run_risk(args, risk_root)
    else:
        _run_validation(args, fixed_root)
        _run_final(args, fixed_root)
        _run_risk(args, risk_root)


if __name__ == "__main__":
    main()
