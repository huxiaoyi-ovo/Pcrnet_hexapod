"""Pure data accessors for the frozen RA-L revision held-out layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


LAYOUT_ID = "revision_heldout_mixed_v1"
_LAYOUT_PATH = Path(__file__).with_name("layouts") / f"{LAYOUT_ID}.json"


def canonical_layout_payload(layout: Dict[str, Any]) -> str:
    """Return the stable JSON representation used for the layout SHA256."""
    payload = {key: value for key, value in layout.items() if key != "sha256"}
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_layout_sha256(layout: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_layout_payload(layout).encode("utf-8")).hexdigest()


def load_revision_heldout_layout() -> Dict[str, Any]:
    with _LAYOUT_PATH.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)
    if layout.get("layout_id") != LAYOUT_ID:
        raise RuntimeError(f"unexpected revision layout id: {layout.get('layout_id')!r}")
    expected_hash = str(layout.get("sha256", ""))
    actual_hash = canonical_layout_sha256(layout)
    if expected_hash != actual_hash:
        raise RuntimeError(f"revision held-out layout hash mismatch: expected={expected_hash}, actual={actual_hash}")
    if layout.get("obstacle_model") != "native_stage4_capsules" or not bool(layout.get("only_layout_change")):
        raise RuntimeError("revision held-out layout must retain the native Stage-4 capsule model")
    if len(layout.get("rows", ())) != 5 or len(layout.get("obstacles", ())) != 13:
        raise RuntimeError("revision held-out layout must contain five rows and thirteen capsules")
    if any(obstacle.get("primitive") != "capsule" for obstacle in layout["obstacles"]):
        raise RuntimeError("revision held-out layout must contain capsules only")
    capsule = layout.get("obstacle_geometry", {}).get("capsule", {})
    if capsule.get("radius") != 0.15 or capsule.get("height") != 0.50 or capsule.get("asset_rotation_y_deg") != 90.0:
        raise RuntimeError("revision held-out layout must retain the native Stage-4 capsule geometry and rotation")
    return layout


def revision_heldout_layout_metadata() -> Dict[str, str]:
    layout = load_revision_heldout_layout()
    return {
        "layout_id": str(layout["layout_id"]),
        "layout_sha256": str(layout["sha256"]),
        "layout_path": str(_LAYOUT_PATH),
    }
