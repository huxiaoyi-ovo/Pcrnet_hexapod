#!/usr/bin/env python3
"""Pure-Python checks for qualification accounting; no Isaac or Torch import."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from legged_gym.avoid_clutter import AvoidClutterGenerator
from qualify_avoid_clutter import (
    _layout_snapshot,
    classify_terminal,
    decision_quota,
    gate,
    layout_metadata_ok,
    successful_outer_bypass,
)


quota = decision_quota()
assert len(quota) == 64 and quota[:8] == [4] * 8 and quota[8:] == [3] * 56
assert sum(quota) == 200

# Failure always wins over an inconsistent concurrent success flag.
assert classify_terminal(True, False, False, False, True) == ("physical", True)
assert classify_terminal(False, True, False, False, True) == ("envelope", True)
assert classify_terminal(False, False, True, False, True) == ("fall", True)
assert classify_terminal(False, False, False, False, True) == ("success", False)

# A single max-x point is not evidence.  Both real band boundaries must cross.
coverage = [[-1.0, 1.0]]
assert successful_outer_bypass([(0.0, 2.0), (0.05, 2.0)], [0.1], [0.3], coverage, True) is None
assert successful_outer_bypass([(0.0, 2.0), (0.4, 2.0)], [0.1], [0.3], coverage, True) == "right"
assert successful_outer_bypass([(0.0, -2.0), (0.4, -2.0)], [0.1], [0.3], coverage, True) == "left"
assert successful_outer_bypass([(0.0, 2.0), (0.4, -2.0)], [0.1], [0.3], coverage, True) is None
# A later outside return cannot relabel an earlier in-coverage traverse.
assert successful_outer_bypass(
    [(0.0, 0.0), (0.4, 0.0), (0.5, 2.0), (0.8, 2.0)], [0.1], [0.3], coverage, True
) is None

# Use the real deterministic generator rather than hand-written layout shapes.
layouts = {}
generator = AvoidClutterGenerator()
for env_id in range(64):
    for episode_id in range(60):
        layout = generator.generate(1, 9174, env_id, episode_id)
        layouts.setdefault(layout.kind, layout)
    if len(layouts) == 3:
        break
assert set(layouts) == {"free", "central", "decision"}
for kind, layout in layouts.items():
    snapshot = _layout_snapshot(layout)
    assert layout_metadata_ok(snapshot), kind
assert len(layouts["free"].band_enter_y) == 0
bad_snapshot = _layout_snapshot(layouts["decision"])
bad_snapshot["raw_numeric_finite"] = False
assert not layout_metadata_ok(bad_snapshot)


def success_row(env_id, side=None):
    return {
        "env_id": env_id, "decision": True, "finite": True, "metadata_ok": True,
        "reason": "success", "failure": False, "bypass_side": side,
    }


rows = []
for env_id, count in enumerate(quota):
    rows.extend(success_row(env_id) for _ in range(count))
assert gate(rows, quota=quota)["pass"]

# A single success episode outside a side is recorded but cannot block release.
rows[0]["bypass_side"] = "left"
single = gate(rows, quota=quota)
assert single["pass"] and single["formal_blocking_bypass_sides"] == []

# Three observations need both a shared side and at least two environments.
rows[1]["bypass_side"] = "left"
rows[4]["bypass_side"] = "left"
repeated = gate(rows, quota=quota)
assert not repeated["pass"] and repeated["formal_blocking_bypass_sides"] == ["left"]
for row in rows:
    row["bypass_side"] = None
rows[0]["bypass_side"] = rows[1]["bypass_side"] = rows[2]["bypass_side"] = "right"
assert gate(rows, quota=quota)["pass"]  # one environment alone cannot block.

# A smoke run can execute safely but can never become a formal PASS.
assert not gate(rows, quota=quota, smoke=True)["pass"]
print("qualify helpers: PASS")
