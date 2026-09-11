#!/usr/bin/env python3
"""Pure standard-library checks for the overnight controller gate sequence."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("overnight", HERE / "continue_overnight.py")
overnight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overnight)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    model = root / "model_200.pt"
    model.write_bytes(b"checkpoint")
    runner = root / "tools" / "qualify_avoid_clutter.py"
    controller = root / "outputs" / "avoid_sanity_20260910" / "continue_overnight.py"
    low_level = root / "agents" / "low_level_best.pt"
    for path, payload in ((runner, b"runner"), (controller, b"controller"), (low_level, b"low-level")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    entries = [
        {"path": str(path.relative_to(root)), "sha256": overnight.sha256(path), "size": path.stat().st_size}
        for path in (low_level, controller, runner)
    ]
    entries.sort(key=lambda item: item["path"])
    manifest = {
        "files": entries,
        "manifest_sha256": overnight.hashlib.sha256(
            json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    write_json(root / "outputs" / "avoid_sanity_20260910" / "source_manifest.json", manifest)
    write_json(root / "outputs" / "avoid_sanity_20260910" / "pre_review.json", {
        "approved_source_manifest_sha256": manifest["manifest_sha256"]
    })
    quota = [4] * 8 + [3] * 56
    rows = []
    for env_id, count in enumerate(quota):
        rows.extend({
            "env_id": env_id, "decision": True, "finite": True, "metadata_ok": True,
            "reason": "success", "failure": False, "bypass_side": None,
        } for _ in range(count))
    original_low_level_sha = overnight.LOW_LEVEL_SHA256
    overnight.LOW_LEVEL_SHA256 = overnight.sha256(low_level)
    qualification = {
        "status": "completed", "formal": True, "smoke": False, "episodes": rows,
        "provenance": {
            "checkpoint_sha256": overnight.sha256(model), "checkpoint_iteration": 200,
            "seed": 9174, "terrain_avoid_seed": 9174, "stage": 1,
            "low_level_checkpoint_sha256": overnight.LOW_LEVEL_SHA256,
            "num_envs": 64, "deterministic_policy": True, "stage12_success_threshold": 1.01,
        },
    }
    assert overnight.recompute_gate(qualification, model, root)["pass"]
    qualification["provenance"]["checkpoint_sha256"] = "wrong"
    assert "qualification_checkpoint_mismatch" in overnight.recompute_gate(qualification, model, root)["reasons"]
    qualification["provenance"]["checkpoint_sha256"] = overnight.sha256(model)
    write_json(root / "outputs" / "avoid_sanity_20260910" / "pre_review.json", {
        "approved_source_manifest_sha256": "unreviewed"
    })
    assert "unreviewed_source_manifest" in overnight.recompute_gate(qualification, model, root)["reasons"]

    # A failed Sanity gate must not call either qualification or formal launch.
    original_wait, original_run = overnight.wait_sanity, overnight.run_own
    calls = []
    overnight.wait_sanity = lambda *args, **kwargs: (False, "missing_checkpoint", {})
    overnight.run_own = lambda *args, **kwargs: calls.append(args) or 0
    assert overnight.run_controller(root, sys.executable, arm_formal=True, poll_seconds=0) == 2
    assert calls == [] and json.loads((root / "controller_status.json").read_text())["status"] == "no_go"
    overnight.wait_sanity, overnight.run_own = original_wait, original_run
    overnight.LOW_LEVEL_SHA256 = original_low_level_sha

    # Timeout termination is scoped to the controller's own process group.
    try:
        overnight.run_own([sys.executable, "-c", "import time; time.sleep(30)"], {}, root / "child.log", 0.1)
        raise AssertionError("timeout unexpectedly returned")
    except RuntimeError as exc:
        assert str(exc) == "owned child timeout"

command = overnight.formal_command("/fresh", "/env/python3", "/sanity/model_200.pt")
assert command[:2] == ["/env/python3", "-u"] and "--finetune_from" in command
print("continue overnight controller: PASS")
