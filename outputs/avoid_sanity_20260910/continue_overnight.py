#!/usr/bin/env python3
"""Fail-closed remote controller for the approved Avoid Sanity continuation."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SANITY_ROOT = Path("/home/dell/Pcrnet_hexapod_runs/avoid_sanity_20260910_220619")
SANITY_PID = 2852404
SANITY_STARTTIME = "269172016"
WATCHDOG_DEADLINE = 1789071406
SANITY_OUTPUT = SANITY_ROOT / "training_output" / "avoid_sanity_stage1_seed9173_20260910_220639"
SANITY_MODEL = SANITY_OUTPUT / "model_200.pt"
LOW_LEVEL_SHA256 = "75dfd7aae52b20f08ca9f654c37b2e20ba1e33f8b953737437ad774d122d1481"
FORMAL_SEED = 9174
FORMAL_ENVS = 64
FORMAL_DECISIONS = 200


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_status(root, status, **extra):
    payload = {"status": status, "timestamp": now(), **extra}
    atomic_json(Path(root) / "controller_status.json", payload)
    return payload


def proc_state(pid, expected_starttime):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return "missing"
    fields = stat.read_text(encoding="utf-8").split()
    if len(fields) < 22 or fields[21] != str(expected_starttime):
        return "reused"
    return "zombie" if fields[2] == "Z" else "running"


def checkpoint_probe(python_exe, checkpoint):
    """Use the approved Torch runtime only for a read-only checkpoint probe."""
    code = r'''
import json, math, sys, torch
obj = torch.load(sys.argv[1], map_location="cpu")
def finite(v):
    if torch.is_tensor(v):
        return (not (v.is_floating_point() or v.is_complex())) or bool(torch.isfinite(v).all().item())
    if isinstance(v, dict): return all(finite(x) for x in v.values())
    if isinstance(v, (list, tuple)): return all(finite(x) for x in v)
    return True
meta = obj.get("experiment_meta", {}) if isinstance(obj, dict) else {}
print(json.dumps({"is_dict": isinstance(obj, dict), "has_state": isinstance(obj.get("model_state_dict"), dict) if isinstance(obj, dict) else False, "finite": finite(obj), "iteration": obj.get("iteration") if isinstance(obj, dict) else None, "meta": meta}, sort_keys=True))
'''
    result = subprocess.run(
        [str(python_exe), "-c", code, str(checkpoint)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError("checkpoint probe failed: " + result.stderr.strip()[-1000:])
    return json.loads(result.stdout)


def checkpoint_ok(probe, iteration):
    meta = probe.get("meta", {})
    required = {
        "revision_contract": "avoid_1d_canonical_v1", "task": "s_avoid_clutter",
        "mode": "teacher", "skill": "avoid", "state_dim": 14, "goal_dim": 2,
        "policy_goal_dim": 2, "actor_output_dim": 1,
        "actor_state_mask_indices": [0, 1, 2, 6, 9],
        "map_semantics": "canonical_observed_2ch",
    }
    return (
        probe.get("is_dict") is True and probe.get("has_state") is True and probe.get("finite") is True
        and probe.get("iteration") == iteration and all(meta.get(k) == v for k, v in required.items())
    )


def sanity_artifacts_ok(python_exe):
    train_log = SANITY_ROOT / "train.log"
    if not train_log.is_file() or "Training Complete!" not in train_log.read_text(encoding="utf-8", errors="replace"):
        return False, "sanity_missing_completion_marker", None
    if not SANITY_MODEL.is_file():
        return False, "sanity_missing_model_200", None
    probe = checkpoint_probe(python_exe, SANITY_MODEL)
    if not checkpoint_ok(probe, 200):
        return False, "sanity_checkpoint_contract_invalid", probe
    return True, "sanity_complete", probe


def wait_sanity(root, python_exe, poll_seconds=30):
    """Wait only for the named Sanity process; a zombie is a completed process."""
    while True:
        watchdog_path = SANITY_ROOT / "replacement_watchdog.json"
        watchdog = load_json(watchdog_path) if watchdog_path.is_file() else {}
        if watchdog.get("status") == "deadline_signalled":
            return False, "watchdog_deadline_signalled", {"watchdog": watchdog}
        state = proc_state(SANITY_PID, SANITY_STARTTIME)
        write_status(root, "sanity_running", sanity_pid=SANITY_PID, sanity_process_state=state, watchdog=watchdog)
        if state in ("missing", "zombie"):
            ok, reason, probe = sanity_artifacts_ok(python_exe)
            return ok, reason, {"watchdog": watchdog, "checkpoint_probe": probe, "process_state": state}
        if state == "reused":
            return False, "sanity_pid_identity_mismatch", {"watchdog": watchdog}
        if time.time() >= WATCHDOG_DEADLINE:
            return False, "sanity_deadline_elapsed_without_completion", {"watchdog": watchdog}
        time.sleep(poll_seconds)


def stop_own_process(process, grace_seconds=20):
    """Terminate only a child process group that this controller created."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_seconds)


def run_own(command, env, log_path, timeout_seconds, cwd=None, on_start=None):
    """Run one owned child process group and leave its complete log as evidence."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True, cwd=None if cwd is None else str(cwd),
            )
            if on_start is not None:
                on_start(process.pid)
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_own_process(process)
            raise RuntimeError("owned child timeout")
        finally:
            if process is not None and process.poll() is None:
                stop_own_process(process)
    return returncode


def runtime_env(root):
    conda_root = "/home/dell/miniconda3/envs/isaac_gym"
    isaac_python = "/home/dell/isaacgym/python"
    inherited_python = os.environ.get("PYTHONPATH", "")
    inherited_library = os.environ.get("LD_LIBRARY_PATH", "")
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=":".join(filter(None, [str(root), isaac_python, inherited_python])),
        LD_LIBRARY_PATH=":".join(filter(None, [f"{conda_root}/lib", inherited_library])),
    )


def recompute_gate(qualification, model_path, root):
    """Recompute formal release conditions from raw qualification evidence."""
    reasons = []
    rows = qualification.get("episodes", [])
    decisions = [row for row in rows if row.get("decision")]
    quota = [4] * 8 + [3] * 56
    counts = [sum(row.get("env_id") == i for row in decisions) for i in range(64)]
    success = sum(row.get("reason") == "success" for row in decisions)
    failure = sum(bool(row.get("failure")) for row in decisions)
    finite = all(row.get("finite") is True for row in decisions)
    metadata = all(row.get("metadata_ok") is True for row in decisions)
    side_block = []
    for side in ("left", "right"):
        subset = [row for row in decisions if row.get("reason") == "success" and row.get("bypass_side") == side]
        if len(subset) >= 3 and len({row.get("env_id") for row in subset}) >= 2:
            side_block.append(side)
    provenance = qualification.get("provenance", {})
    if qualification.get("status") != "completed" or qualification.get("formal") is not True or qualification.get("smoke"):
        reasons.append("qualification_not_formal_completed")
    if len(decisions) != FORMAL_DECISIONS or counts != quota:
        reasons.append("decision_quota_invalid")
    if success / max(1, len(decisions)) < 0.85 or failure / max(1, len(decisions)) > 0.10:
        reasons.append("rate_gate_failed")
    if not finite or not metadata or any(row.get("reason") == "other" for row in decisions):
        reasons.append("row_integrity_failed")
    if side_block:
        reasons.append("repeated_outer_bypass")
    if provenance.get("checkpoint_sha256") != sha256(model_path) or provenance.get("checkpoint_iteration") != 200:
        reasons.append("qualification_checkpoint_mismatch")
    if provenance.get("seed") != FORMAL_SEED or provenance.get("terrain_avoid_seed") != FORMAL_SEED or provenance.get("stage") != 1:
        reasons.append("qualification_stage_seed_mismatch")
    if provenance.get("low_level_checkpoint_sha256") != LOW_LEVEL_SHA256:
        reasons.append("low_level_sha_mismatch")
    manifest = load_json(Path(root) / "outputs" / "avoid_sanity_20260910" / "source_manifest.json")
    pre_review = load_json(Path(root) / "outputs" / "avoid_sanity_20260910" / "pre_review.json")
    files = manifest.get("files", [])
    canonical = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if not files or manifest.get("manifest_sha256") != hashlib.sha256(canonical).hexdigest():
        reasons.append("source_manifest_invalid")
    if manifest.get("manifest_sha256") != pre_review.get("approved_source_manifest_sha256"):
        reasons.append("unreviewed_source_manifest")
    for entry in files:
        path = Path(root) / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            reasons.append("source_sha_mismatch")
            break
    if sha256(Path(root) / "agents" / "low_level_best.pt") != LOW_LEVEL_SHA256:
        reasons.append("source_low_level_sha_mismatch")
    if not {"tools/qualify_avoid_clutter.py", "outputs/avoid_sanity_20260910/continue_overnight.py"}.issubset(
        {entry.get("path") for entry in files}
    ):
        reasons.append("source_manifest_missing_controller_or_runner")
    if provenance.get("num_envs") != FORMAL_ENVS or provenance.get("deterministic_policy") is not True:
        reasons.append("qualification_runtime_contract_mismatch")
    if provenance.get("stage12_success_threshold") != 1.01:
        reasons.append("qualification_stage_freeze_mismatch")
    return {
        "pass": not reasons, "reasons": reasons, "decision_count": len(decisions), "quota_counts": counts,
        "success_rate": success / max(1, len(decisions)), "failure_rate": failure / max(1, len(decisions)),
        "formal_blocking_bypass_sides": side_block, "checkpoint_sha256": sha256(model_path),
    }


def formal_command(root, python_exe, model_path):
    return [
        str(python_exe), "-u", str(Path(root) / "legged_gym" / "scripts" / "train_highlevel.py"),
        "--task", "s_avoid_clutter", "--skill", "avoid", "--revision_contract", "--seed", "9173",
        "--num_envs", "128", "--num_iterations", "1001", "--num_steps", "24", "--save_interval", "200",
        "--finetune_from", str(model_path), "--low_level_ckpt", str(Path(root) / "agents" / "low_level_best.pt"),
        "--output_dir", str(Path(root) / "formal_output"),
        "--run_name", "avoid_formal_seed9173", "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--headless",
    ]


def formal_run_dirs(root):
    output = Path(root) / "formal_output"
    if not output.is_dir():
        return set()
    return {
        path.resolve() for path in output.iterdir()
        if path.is_dir() and path.name.startswith("avoid_formal_seed9173_") and (path / "run_meta.json").is_file()
    }


def locate_new_formal_run(root, before):
    new_dirs = formal_run_dirs(root) - set(before)
    if len(new_dirs) != 1:
        raise RuntimeError(f"expected one new formal run directory, got {sorted(map(str, new_dirs))}")
    run_dir = new_dirs.pop()
    meta = load_json(run_dir / "run_meta.json")
    if Path(meta.get("log_dir", "")).resolve() != run_dir:
        raise RuntimeError("formal run_meta log_dir does not match the new run directory")
    return run_dir, run_dir / "model_1000.pt"


def run_controller(root, python_exe, arm_formal=False, poll_seconds=30):
    root = Path(root)
    ok, reason, sanity_evidence = wait_sanity(root, python_exe, poll_seconds=poll_seconds)
    if not ok:
        write_status(root, "no_go", reason=reason, sanity=sanity_evidence)
        return 2
    model_path = SANITY_MODEL
    qualification_path = root / "qualification.json"
    env = runtime_env(root)
    command = [
        str(python_exe), "-u", str(root / "tools" / "qualify_avoid_clutter.py"),
        "--checkpoint", str(model_path), "--output", str(qualification_path),
        "--num_envs", "64", "--seed", "9174", "--max_steps", "6000",
    ]
    write_status(root, "qualifying", command=command, checkpoint=str(model_path), cuda_visible_devices="1")
    try:
        returncode = run_own(
            command, env, root / "qualification.log", 2 * 60 * 60, cwd=root,
            on_start=lambda pid: write_status(root, "qualifying", child_pid=pid, command=command, checkpoint=str(model_path), cuda_visible_devices="1"),
        )
    except Exception as exc:
        write_status(root, "error", reason="qualification_controller_error", error=str(exc))
        return 2
    if returncode != 0 or not qualification_path.is_file():
        write_status(root, "no_go", reason="qualification_failed", returncode=returncode)
        return 2
    qualification = load_json(qualification_path)
    review = recompute_gate(qualification, model_path, root)
    atomic_json(root / "final_review.json", review)
    if not review["pass"]:
        write_status(root, "no_go", reason="final_review_failed", final_review=review)
        return 2
    if not arm_formal:
        write_status(root, "reviewing", reason="formal_not_armed", final_review=review)
        return 0
    command = formal_command(root, python_exe, model_path)
    env = runtime_env(root)
    before = formal_run_dirs(root)
    write_status(root, "formal_running", command=command, checkpoint=str(model_path), fresh_optimizer=True, cuda_visible_devices="1")
    try:
        returncode = run_own(
            command, env, root / "formal_output" / "train.log", 24 * 60 * 60, cwd=root,
            on_start=lambda pid: write_status(root, "formal_running", child_pid=pid, command=command, checkpoint=str(model_path), fresh_optimizer=True, cuda_visible_devices="1"),
        )
    except Exception as exc:
        write_status(root, "error", reason="formal_controller_error", error=str(exc))
        return 2
    marker = root / "formal_output" / "train.log"
    if returncode != 0 or not marker.is_file() or "Training Complete!" not in marker.read_text(encoding="utf-8", errors="replace"):
        write_status(root, "error", reason="formal_completion_missing", returncode=returncode)
        return 2
    try:
        run_dir, model_1000 = locate_new_formal_run(root, before)
    except Exception as exc:
        write_status(root, "error", reason="formal_run_location_invalid", error=str(exc))
        return 2
    probe = checkpoint_probe(python_exe, model_1000)
    if not checkpoint_ok(probe, 1000):
        write_status(root, "error", reason="formal_checkpoint_invalid", checkpoint_probe=probe)
        return 2
    write_status(root, "completed", formal_model=str(model_1000), formal_run_dir=str(run_dir), checkpoint_probe=probe)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--arm-formal", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        code = run_controller(args.root, args.python, arm_formal=args.arm_formal, poll_seconds=args.poll_seconds)
    except KeyboardInterrupt:
        write_status(args.root, "error", reason="controller_interrupted")
        code = 130
    except Exception as exc:
        write_status(args.root, "error", reason="controller_unhandled_exception", error=str(exc))
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
