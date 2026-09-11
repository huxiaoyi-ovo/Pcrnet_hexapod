#!/usr/bin/env python3
"""Build and verify a fresh remote snapshot for the approved GPU smoke only."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
REMOTE = "pcrnet-server"
REMOTE_BASE = "/home/dell/Pcrnet_hexapod_runs"
INCLUDE = (
    "legged_gym", "rsl_rl", "resources", "agents/low_level_best.pt",
    "tools/qualify_avoid_clutter.py", "tools/test_qualify_avoid_clutter.py",
    "tools/smoke_avoid_clutter_training.py",
    "outputs/avoid_sanity_20260910/continue_overnight.py",
    "outputs/avoid_sanity_20260910/test_continue_overnight.py",
    "outputs/avoid_sanity_20260910/deploy_overnight.py",
)
REVIEW_FILE = "outputs/avoid_sanity_20260910/pre_review.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root=ROOT):
    files = []
    for relative in INCLUDE:
        item = Path(root) / relative
        if item.is_file():
            candidates = [item]
        elif item.is_dir():
            candidates = [path for path in item.rglob("*") if path.is_file()]
        else:
            raise FileNotFoundError(item)
        for path in candidates:
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo"):
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: str(path.relative_to(root)))


def manifest(files, root=ROOT):
    entries = [
        {"path": str(path.relative_to(root)), "sha256": sha256(path), "size": path.stat().st_size}
        for path in files
    ]
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {"files": entries, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}


def build_bundle(bundle):
    files = source_files()
    data = manifest(files)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "source_manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with tarfile.open(bundle, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        archive.add(
            OUTPUT / "source_manifest.json",
            arcname="outputs/avoid_sanity_20260910/source_manifest.json",
            recursive=False,
        )
        review_path = ROOT / REVIEW_FILE
        if review_path.is_file():
            archive.add(review_path, arcname=REVIEW_FILE, recursive=False)
    return data


def run(command):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def deploy(bundle, snapshot):
    remote = f"{REMOTE}:{snapshot}"
    run(["ssh", REMOTE, "mkdir", "-p", snapshot])
    run(["scp", str(bundle), remote + "/"])
    remote_bundle = f"{snapshot}/{bundle.name}"
    local_sha = sha256(bundle)
    # The manifest's field order is JSON, so remote verification is performed
    # by a tiny standard-library reader rather than assuming line ordering.
    verify_py = (
        "import hashlib,json,pathlib; root=pathlib.Path('" + snapshot + "'); "
        "m=json.load(open(root/'outputs/avoid_sanity_20260910/source_manifest.json')); "
        "assert all(hashlib.sha256((root/e['path']).read_bytes()).hexdigest()==e['sha256'] for e in m['files']); "
        "print('snapshot manifest: PASS')"
    )
    remote_script = (
        f"set -eu; test \"$(sha256sum {shlex.quote(remote_bundle)} | awk '{{print $1}}')\" = {shlex.quote(local_sha)}; "
        f"tar -xzf {shlex.quote(remote_bundle)} -C {shlex.quote(snapshot)}; "
        f"python3 -c {shlex.quote(verify_py)}"
    )
    run(["ssh", REMOTE, "bash -lc " + shlex.quote(remote_script)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true", help="perform the approved ssh/scp transfer")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    bundle = OUTPUT / f"avoid_overnight_bundle_{args.tag}.tar.gz"
    data = build_bundle(bundle)
    record = {
        "status": "built", "bundle": str(bundle), "bundle_sha256": sha256(bundle),
        "bundle_bytes": bundle.stat().st_size, "source_manifest_sha256": data["manifest_sha256"],
        "source_files": len(data["files"]), "remote": None,
    }
    if args.deploy:
        snapshot = f"{REMOTE_BASE}/avoid_overnight_20260910_{args.tag}"
        deploy(bundle, snapshot)
        record.update({"status": "deployed", "remote": snapshot})
    (OUTPUT / "deployment_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
