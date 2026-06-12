#!/usr/bin/env python3
"""Audit trajectory timeseries files for the final Fig.6 pool."""

import argparse
import csv
import glob
import hashlib
import heapq
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import permutations, product
from typing import Dict, Iterable, List, Sequence, Tuple


METHOD_ORDER = ["yonly", "geomw", "risk_only", "rule_override", "learnedw"]
METHOD_LABELS = {
    "yonly": "Y-only",
    "geomw": "Geom-w",
    "risk_only": "Risk-only",
    "rule_override": "Rule-Override",
    "learnedw": "Learned-w",
}
REQUIRED_FIELDS = (
    "episode_id",
    "trajectory_frame",
    "robot_x",
    "robot_y",
    "target_x",
    "target_y",
    "obstacles_json",
    "episode_termination_reason",
)
LOST_LIKE = {"follow_lost", "target_lost", "timeout"}
FIG6_FAILURE_CATEGORIES = (
    "collision_row3",
    "collision_row4",
    "lost_early",
    "lost_late",
)


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown(path: str, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def _infer_method(path: str) -> str:
    text = path.lower()
    if "risk_only" in text or "risk-only" in text:
        return "risk_only"
    if "rule_override" in text or "rule-override" in text:
        return "rule_override"
    if "learnedw2" in text or "learnedw" in text or "learned-w" in text:
        return "learnedw"
    if "geomw" in text or "geom-w" in text:
        return "geomw"
    if "yonly" in text or "y-only" in text:
        return "yonly"
    return "unknown"


def _infer_speed(path: str, rows: Sequence[Dict[str, str]]) -> str:
    for row in rows[:10]:
        for key in ("Speed", "speed", "resolved_moving_target_pcr_line_speed"):
            value = row.get(key, "")
            v = _safe_float(value)
            if math.isfinite(v):
                return f"{v:.2f}"
    m = re.search(r"/s_(0\.35|0\.5|0\.50|0\.6|0\.60)(?:/|$)", path)
    if m:
        return f"{float(m.group(1)):.2f}"
    m = re.search(r"speed[_-]?(0\.35|0\.5|0\.50|0\.6|0\.60)", path.lower())
    if m:
        return f"{float(m.group(1)):.2f}"
    return ""


def _infer_seed(path: str) -> str:
    m = re.search(r"seed(\d+)", path.lower())
    return m.group(1) if m else ""


def _run_id(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:10]


def _episode_ids(rows: Sequence[Dict[str, str]]) -> List[str]:
    ids = {str(r.get("episode_id", "")) for r in rows if str(r.get("episode_id", "")) != ""}
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def _episode_rows(rows: Sequence[Dict[str, str]], episode_id: str) -> List[Dict[str, str]]:
    out = [r for r in rows if str(r.get("episode_id", "")) == str(episode_id)]
    return sorted(
        out,
        key=lambda r: (
            _safe_float(r.get("step_hl", ""), default=0.0),
            _safe_float(r.get("time_s", ""), default=0.0),
        ),
    )


def _load_obstacles(text: str) -> List[Dict[str, float]]:
    try:
        raw = json.loads(text or "[]")
    except Exception:
        return []
    out = []
    if not isinstance(raw, list):
        return out
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        x = _safe_float(item.get("x", ""))
        y = _safe_float(item.get("y", ""))
        r = _safe_float(item.get("r", ""), default=0.15)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        out.append(
            {
                "slot": int(_safe_float(item.get("slot", idx), default=idx)),
                "x": round(x, 4),
                "y": round(y, 4),
                "r": round(r, 4),
            }
        )
    return sorted(out, key=lambda z: (z["slot"], z["x"], z["y"]))


def _layout_id(obstacles: Sequence[Dict[str, float]]) -> str:
    text = json.dumps(list(obstacles), separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _layout_summary(obstacles: Sequence[Dict[str, float]]) -> Dict[str, str]:
    if not obstacles:
        return {
            "layout_id": "",
            "layout_side": "unknown",
            "layout_x_mean": "nan",
            "layout_x_abs_mean": "nan",
            "layout_y_min": "nan",
            "layout_y_max": "nan",
            "layout_signature": "",
            "obstacle_count": "0",
        }
    xs = [float(o["x"]) for o in obstacles]
    ys = [float(o["y"]) for o in obstacles]
    x_mean = sum(xs) / len(xs)
    x_abs_mean = sum(abs(x) for x in xs) / len(xs)
    if x_mean > 0.08:
        side = "right_heavy"
    elif x_mean < -0.08:
        side = "left_heavy"
    else:
        side = "balanced"
    bins: Dict[int, List[float]] = defaultdict(list)
    for obs in obstacles:
        bins[int(round(float(obs["y"]) * 2.0))].append(float(obs["x"]))
    parts = []
    for y_bin in sorted(bins):
        row_xs = bins[y_bin]
        left = sum(1 for x in row_xs if x < -0.08)
        mid = sum(1 for x in row_xs if -0.08 <= x <= 0.08)
        right = sum(1 for x in row_xs if x > 0.08)
        parts.append(f"{left}{mid}{right}")
    return {
        "layout_id": _layout_id(obstacles),
        "layout_side": side,
        "layout_x_mean": f"{x_mean:.4f}",
        "layout_x_abs_mean": f"{x_abs_mean:.4f}",
        "layout_y_min": f"{min(ys):.4f}",
        "layout_y_max": f"{max(ys):.4f}",
        "layout_signature": "-".join(parts),
        "obstacle_count": str(len(obstacles)),
    }


def _obstacle_row_centers(
    obstacles: Sequence[Dict[str, float]],
    *,
    cluster_gap: float = 0.6,
) -> List[float]:
    ys = sorted(float(obs["y"]) for obs in obstacles if math.isfinite(float(obs["y"])))
    if not ys:
        return []
    clusters: List[List[float]] = [[ys[0]]]
    for y in ys[1:]:
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(y - center) <= cluster_gap:
            clusters[-1].append(y)
        else:
            clusters.append([y])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _truncate_episode_at_reset(
    rows: Sequence[Dict[str, str]],
    *,
    jump_threshold: float = 1.2,
) -> Tuple[List[Dict[str, str]], bool]:
    kept: List[Dict[str, str]] = []
    prev_x = float("nan")
    prev_y = float("nan")
    for row in rows:
        x = _safe_float(row.get("robot_x", ""))
        y = _safe_float(row.get("robot_y", ""))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if (
            math.isfinite(prev_x)
            and math.isfinite(prev_y)
            and math.hypot(x - prev_x, y - prev_y) > jump_threshold
        ):
            return kept, True
        kept.append(row)
        prev_x = x
        prev_y = y
    return kept, False


def _nearest_obstacle_row(y: float, row_centers: Sequence[float]) -> Tuple[int, float]:
    if not math.isfinite(y) or not row_centers:
        return 0, float("nan")
    index = min(range(len(row_centers)), key=lambda i: abs(y - row_centers[i]))
    return index + 1, abs(y - row_centers[index])


def _failure_category(entry: Dict) -> str:
    reason = str(entry.get("termination", "")).lower()
    row_index = int(_safe_float(entry.get("terminal_row", ""), default=0.0))
    if reason == "collision" and row_index in (3, 4):
        return f"collision_row{row_index}"
    if reason in LOST_LIKE:
        if row_index and row_index <= 3:
            return "lost_early"
        if row_index >= 4:
            return "lost_late"
    return _reason_group(reason)


def _mirror_ok(entries: Sequence[Dict], strict_same_side: bool = False) -> bool:
    sides = {str(e.get("layout_side", "")) for e in entries if str(e.get("layout_side", "")) != "unknown"}
    if not sides:
        return False
    if strict_same_side:
        return len(sides) == 1
    return not ("left_heavy" in sides and "right_heavy" in sides)


def _reason_group(reason: str) -> str:
    reason = str(reason or "").strip().lower()
    if reason == "collision":
        return "collision"
    if reason in LOST_LIKE:
        return "lost_timeout"
    if reason == "success":
        return "success"
    return reason or "unknown"


def _score_candidate(entries: Dict[str, Dict], strict_same_side: bool = False) -> Tuple[float, Dict]:
    learned_reason = str(entries["learnedw"].get("termination", "")).lower()
    baseline = [entries[m] for m in METHOD_ORDER if m != "learnedw"]
    collisions = sum(1 for e in baseline if str(e.get("termination", "")).lower() == "collision")
    lost = sum(1 for e in baseline if str(e.get("termination", "")).lower() in LOST_LIKE)
    baseline_success = sum(1 for e in baseline if str(e.get("termination", "")).lower() == "success")
    mirror_ok = _mirror_ok(list(entries.values()), strict_same_side=strict_same_side)
    layout_ids = {str(e.get("layout_id", "")) for e in entries.values()}
    layout_sides = {str(e.get("layout_side", "")) for e in entries.values()}
    layout_signatures = {str(e.get("layout_signature", "")) for e in entries.values()}
    failure_categories = [_failure_category(e) for e in baseline]
    exact_failure_pattern = sorted(failure_categories) == sorted(FIG6_FAILURE_CATEGORIES)

    score = 0.0
    score += 5000.0 if learned_reason == "success" else -5000.0
    score += 8000.0 if exact_failure_pattern else 0.0
    if 2 <= collisions <= 3:
        score += 1300.0
    else:
        score -= 450.0 * min(abs(collisions - 2), abs(collisions - 3))
    score += 250.0 * min(collisions, 3)
    score += 800.0 if lost >= 1 else -800.0
    score -= 120.0 * baseline_success
    if mirror_ok:
        score += 500.0
    else:
        score -= 3000.0
    if len(layout_ids) == 1:
        score += 400.0
    if len(layout_signatures) == 1:
        score += 200.0

    row = {
        "score": f"{score:.3f}",
        "mirror_ok": "1" if mirror_ok else "0",
        "baseline_collisions": str(collisions),
        "baseline_lost_timeout": str(lost),
        "baseline_success": str(baseline_success),
        "exact_failure_pattern": "1" if exact_failure_pattern else "0",
        "layout_ids": ",".join(sorted(layout_ids)),
        "layout_sides": ",".join(sorted(layout_sides)),
        "layout_signatures": ",".join(sorted(layout_signatures)),
        "selected_episodes": ",".join(
            f"{m}:{entries[m].get('run_id')}:{entries[m].get('episode_id')}" for m in METHOD_ORDER
        ),
        "selected_terminations": ",".join(f"{m}:{entries[m].get('termination')}" for m in METHOD_ORDER),
        "selected_failure_categories": ",".join(
            f"{m}:{_failure_category(entries[m])}" for m in METHOD_ORDER
        ),
        "selected_terminal_rows": ",".join(
            f"{m}:{entries[m].get('terminal_row')}" for m in METHOD_ORDER
        ),
        "selected_terminal_y": ",".join(
            f"{m}:{entries[m].get('terminal_y')}" for m in METHOD_ORDER
        ),
        "selected_time_end": ",".join(
            f"{m}:{entries[m].get('terminal_time')}" for m in METHOD_ORDER
        ),
        "selected_sources": ";".join(f"{m}:{entries[m].get('source')}" for m in METHOD_ORDER),
    }
    return score, row


def _compact_entries(entries: Sequence[Dict], per_group: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        grouped[_reason_group(str(entry.get("termination", "")))].append(entry)
    out: List[Dict] = []
    for group in ("collision", "lost_timeout", "success", "unknown"):
        out.extend(sorted(grouped.get(group, []), key=lambda e: (e.get("run_id", ""), str(e.get("episode_id", ""))))[:per_group])
    return out


def _category_pool(entries: Sequence[Dict], category: str, limit: int) -> List[Dict]:
    matched = [entry for entry in entries if _failure_category(entry) == category]
    if category == "lost_early":
        matched.sort(
            key=lambda e: (
                int(_safe_float(e.get("terminal_row", ""), default=99.0)),
                _safe_float(e.get("terminal_y", ""), default=999.0),
                _safe_float(e.get("terminal_time", ""), default=999.0),
            )
        )
    elif category == "lost_late":
        matched.sort(
            key=lambda e: (
                -int(_safe_float(e.get("terminal_row", ""), default=0.0)),
                -_safe_float(e.get("terminal_y", ""), default=-999.0),
                -_safe_float(e.get("terminal_time", ""), default=-999.0),
            )
        )
    else:
        matched.sort(
            key=lambda e: (
                _safe_float(e.get("terminal_row_error", ""), default=999.0),
                _safe_float(e.get("terminal_time", ""), default=999.0),
            )
        )
    return matched[:limit]


def _learned_success_pool(entries: Sequence[Dict], per_side: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for entry in entries:
        if str(entry.get("termination", "")).lower() == "success":
            grouped[str(entry.get("layout_side", "unknown"))].append(entry)
    out: List[Dict] = []
    for side in sorted(grouped):
        out.extend(
            sorted(
                grouped[side],
                key=lambda e: (
                    _safe_float(e.get("terminal_time", ""), default=999.0),
                    e.get("run_id", ""),
                    str(e.get("episode_id", "")),
                ),
            )[:per_side]
        )
    return out


def _scan_timeseries(path: str) -> Tuple[Dict, List[Dict]]:
    rows = _read_csv(path)
    fields = set(rows[0].keys()) if rows else set()
    missing = [f for f in REQUIRED_FIELDS if f not in fields]
    method = _infer_method(path)
    speed = _infer_speed(path, rows)
    seed = _infer_seed(path)
    run_id = _run_id(path)
    episodes = _episode_ids(rows) if not missing else []
    termination_counter = Counter()
    episode_entries: List[Dict] = []
    layout_ids = set()
    layout_sides = Counter()

    for episode_id in episodes:
        erows = _episode_rows(rows, episode_id)
        if not erows:
            continue
        reason = str(erows[-1].get("episode_termination_reason", "")).strip().lower() or "unknown"
        termination_counter[reason] += 1
        obstacles = _load_obstacles(erows[0].get("obstacles_json", ""))
        layout = _layout_summary(obstacles)
        obstacle_rows = _obstacle_row_centers(obstacles)
        trajectory_rows, reset_detected = _truncate_episode_at_reset(erows)
        terminal_row_data = trajectory_rows[-1] if trajectory_rows else erows[-1]
        terminal_x = _safe_float(terminal_row_data.get("robot_x", ""))
        terminal_y = _safe_float(terminal_row_data.get("robot_y", ""))
        terminal_time = _safe_float(terminal_row_data.get("time_s", ""))
        start_y = _safe_float(trajectory_rows[0].get("robot_y", "")) if trajectory_rows else float("nan")
        terminal_row, terminal_row_error = _nearest_obstacle_row(terminal_y, obstacle_rows)
        if obstacle_rows and math.isfinite(start_y) and obstacle_rows[-1] > start_y:
            terminal_progress = (terminal_y - start_y) / (obstacle_rows[-1] - start_y)
        else:
            terminal_progress = float("nan")
        layout_ids.add(layout["layout_id"])
        layout_sides[layout["layout_side"]] += 1
        entry = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "speed": speed,
            "seed": seed,
            "run_id": run_id,
            "episode_id": str(episode_id),
            "termination": reason,
            "source": path,
            "rows": str(len(erows)),
            "time_start": erows[0].get("time_s", ""),
            "time_end": erows[-1].get("time_s", ""),
            "terminal_time": f"{terminal_time:.4f}" if math.isfinite(terminal_time) else "nan",
            "terminal_x": f"{terminal_x:.4f}" if math.isfinite(terminal_x) else "nan",
            "terminal_y": f"{terminal_y:.4f}" if math.isfinite(terminal_y) else "nan",
            "terminal_progress": (
                f"{terminal_progress:.4f}" if math.isfinite(terminal_progress) else "nan"
            ),
            "terminal_row": str(terminal_row),
            "terminal_row_error": (
                f"{terminal_row_error:.4f}" if math.isfinite(terminal_row_error) else "nan"
            ),
            "failure_category": "",
            "reset_detected": "1" if reset_detected else "0",
            "obstacle_row_centers": ",".join(f"{y:.4f}" for y in obstacle_rows),
        }
        entry.update(layout)
        entry["failure_category"] = _failure_category(entry)
        episode_entries.append(entry)

    run_row = {
        "run_id": run_id,
        "method": method,
        "speed": speed,
        "seed": seed,
        "source": path,
        "row_count": str(len(rows)),
        "episode_count": str(len(episodes)),
        "field_ok": "1" if not missing else "0",
        "missing_fields": ",".join(missing),
        "trajectory_frame": rows[0].get("trajectory_frame", "") if rows else "",
        "layout_count": str(len([x for x in layout_ids if x])),
        "layout_sides": ",".join(f"{k}:{v}" for k, v in sorted(layout_sides.items())),
    }
    for reason in ("success", "collision", "target_lost", "timeout", "follow_lost", "unknown"):
        run_row[reason] = str(termination_counter.get(reason, 0))
    return run_row, episode_entries


def _build_method_summary(episode_rows: Sequence[Dict], speed: str) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in episode_rows:
        if row.get("speed") == speed:
            grouped[(row.get("speed", ""), row.get("method", ""))].append(row)
    out: List[Dict] = []
    for (sp, method), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], METHOD_ORDER.index(kv[0][1]) if kv[0][1] in METHOD_ORDER else 99)):
        counter = Counter(str(r.get("termination", "")) for r in rows)
        sides = Counter(str(r.get("layout_side", "")) for r in rows)
        out.append(
            {
                "speed": sp,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "episode_count": str(len(rows)),
                "success": str(counter.get("success", 0)),
                "collision": str(counter.get("collision", 0)),
                "target_lost": str(counter.get("target_lost", 0)),
                "timeout": str(counter.get("timeout", 0)),
                "follow_lost": str(counter.get("follow_lost", 0)),
                "lost_timeout_total": str(sum(counter.get(k, 0) for k in LOST_LIKE)),
                "layout_count": str(len({r.get("layout_id", "") for r in rows if r.get("layout_id", "")})),
                "layout_sides": ",".join(f"{k}:{v}" for k, v in sorted(sides.items())),
            }
        )
    return out


def _build_candidate_sets(
    episode_rows: Sequence[Dict],
    speed: str,
    per_method_limit: int,
    top_k: int,
    strict_same_side: bool,
) -> List[Dict]:
    by_method: Dict[str, List[Dict]] = defaultdict(list)
    for row in episode_rows:
        if row.get("speed") == speed and row.get("method") in METHOD_ORDER:
            by_method[row["method"]].append(row)
    if any(not by_method.get(method) for method in METHOD_ORDER):
        return []

    learned_pool = _learned_success_pool(by_method["learnedw"], per_side=per_method_limit)
    if not learned_pool:
        return []
    baseline_methods = [m for m in METHOD_ORDER if m != "learnedw"]
    category_pools = {
        (method, category): _category_pool(
            by_method[method],
            category,
            per_method_limit,
        )
        for method in baseline_methods
        for category in FIG6_FAILURE_CATEGORIES
    }
    ranked: List[Tuple[float, int, Dict]] = []
    serial = 0

    def consider(entries: Dict[str, Dict]) -> None:
        nonlocal serial
        if not _mirror_ok(list(entries.values()), strict_same_side=strict_same_side):
            return
        score, row = _score_candidate(entries, strict_same_side=strict_same_side)
        item = (score, serial, row)
        serial += 1
        if len(ranked) < top_k:
            heapq.heappush(ranked, item)
        elif score > ranked[0][0]:
            heapq.heapreplace(ranked, item)

    for learned in learned_pool:
        for categories in permutations(FIG6_FAILURE_CATEGORIES):
            pools = [
                category_pools[(method, category)]
                for method, category in zip(baseline_methods, categories)
            ]
            if any(not pool for pool in pools):
                continue
            for choices in product(*pools):
                entries = {"learnedw": learned}
                entries.update(dict(zip(baseline_methods, choices)))
                consider(entries)

    if ranked:
        return [item[2] for item in sorted(ranked, key=lambda x: (x[0], x[1]), reverse=True)]

    print(
        "[Warn] No exact Fig.6 failure pattern found; falling back to broad "
        "2-3 collision plus lost/timeout ranking."
    )
    broad_pools = {
        method: _compact_entries(by_method[method], per_group=per_method_limit)
        for method in baseline_methods
    }
    for learned in learned_pool:
        for choices in product(*(broad_pools[m] for m in baseline_methods)):
            entries = {"learnedw": learned}
            entries.update(dict(zip(baseline_methods, choices)))
            consider(entries)
    return [item[2] for item in sorted(ranked, key=lambda x: (x[0], x[1]), reverse=True)]


def _write_report(path: str, run_rows: Sequence[Dict], summary_rows: Sequence[Dict], candidate_rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fig.6 Trajectory Pool Audit\n\n")
        f.write("## Method Summary\n\n")
        headers = [
            "speed", "method", "episode_count", "success", "collision",
            "target_lost", "timeout", "follow_lost", "lost_timeout_total",
            "layout_count", "layout_sides",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in summary_rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
        f.write("\n## Top Fig.6 Candidate Sets\n\n")
        c_headers = [
            "score", "exact_failure_pattern", "mirror_ok", "baseline_collisions",
            "baseline_lost_timeout", "layout_sides", "selected_failure_categories",
            "selected_terminal_rows", "selected_terminal_y", "selected_episodes",
        ]
        f.write("| " + " | ".join(c_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(c_headers)) + " |\n")
        for row in candidate_rows[:20]:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in c_headers) + " |\n")
        f.write("\n## Run Count\n\n")
        f.write(f"- Runs scanned: {len(run_rows)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit trajectory timeseries files for Fig.6.")
    parser.add_argument("--root", action="append", default=["agents"], help="Root directory to scan; repeatable.")
    parser.add_argument("--speed", default="0.60", help="Speed used for Fig.6 candidate ranking.")
    parser.add_argument("--output_dir", default="agents/fig6_trajectory_pool_audit")
    parser.add_argument("--per_method_limit", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--strict_same_side", action="store_true", help="Require all selected layouts to share side class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [os.path.normpath(r) for r in args.root if str(r).strip()]
    paths: List[str] = []
    for root in roots:
        paths.extend(glob.glob(os.path.join(root, "**", "timeseries.csv"), recursive=True))
    paths = sorted(set(paths))

    run_rows: List[Dict] = []
    episode_rows: List[Dict] = []
    for path in paths:
        run_row, entries = _scan_timeseries(path)
        run_rows.append(run_row)
        episode_rows.extend(entries)

    speed = f"{float(args.speed):.2f}"
    summary_rows = _build_method_summary(episode_rows, speed=speed)
    candidate_rows = _build_candidate_sets(
        episode_rows,
        speed=speed,
        per_method_limit=max(1, int(args.per_method_limit)),
        top_k=max(1, int(args.top_k)),
        strict_same_side=bool(args.strict_same_side),
    )

    run_fields = [
        "run_id", "method", "speed", "seed", "source", "row_count", "episode_count",
        "field_ok", "missing_fields", "trajectory_frame", "layout_count", "layout_sides",
        "success", "collision", "target_lost", "timeout", "follow_lost", "unknown",
    ]
    episode_fields = [
        "method", "method_label", "speed", "seed", "run_id", "episode_id", "termination",
        "source", "rows", "time_start", "time_end", "terminal_time",
        "terminal_x", "terminal_y", "terminal_progress", "terminal_row",
        "terminal_row_error", "failure_category", "reset_detected",
        "obstacle_row_centers", "layout_id", "layout_side",
        "layout_x_mean", "layout_x_abs_mean", "layout_y_min", "layout_y_max",
        "layout_signature", "obstacle_count",
    ]
    summary_fields = [
        "speed", "method", "method_label", "episode_count", "success", "collision",
        "target_lost", "timeout", "follow_lost", "lost_timeout_total",
        "layout_count", "layout_sides",
    ]
    candidate_fields = [
        "score", "exact_failure_pattern", "mirror_ok", "baseline_collisions",
        "baseline_lost_timeout", "baseline_success", "layout_ids", "layout_sides",
        "layout_signatures", "selected_episodes", "selected_terminations",
        "selected_failure_categories", "selected_terminal_rows", "selected_terminal_y",
        "selected_time_end", "selected_sources",
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "fig6_runs.csv"), run_rows, run_fields)
    _write_csv(os.path.join(args.output_dir, "fig6_episodes.csv"), episode_rows, episode_fields)
    _write_csv(os.path.join(args.output_dir, "fig6_method_summary.csv"), summary_rows, summary_fields)
    _write_markdown(os.path.join(args.output_dir, "fig6_method_summary.md"), summary_rows, summary_fields)
    _write_csv(os.path.join(args.output_dir, "fig6_candidate_sets.csv"), candidate_rows, candidate_fields)
    _write_markdown(os.path.join(args.output_dir, "fig6_candidate_sets.md"), candidate_rows[:30], candidate_fields)
    _write_report(os.path.join(args.output_dir, "README.md"), run_rows, summary_rows, candidate_rows)

    print(f"[Fig6Audit] timeseries files scanned: {len(paths)}")
    print(f"[Fig6Audit] runs written: {len(run_rows)}")
    print(f"[Fig6Audit] episodes written: {len(episode_rows)}")
    print(f"[Fig6Audit] candidate sets written: {len(candidate_rows)}")
    print(f"[Fig6Audit] output_dir: {args.output_dir}")
    if candidate_rows:
        top = candidate_rows[0]
        print(
            "[Fig6Audit] top candidate: "
            f"score={top.get('score')} collisions={top.get('baseline_collisions')} "
            f"lost={top.get('baseline_lost_timeout')} mirror_ok={top.get('mirror_ok')} "
            f"exact={top.get('exact_failure_pattern')} "
            f"categories={top.get('selected_failure_categories')}"
        )
    else:
        print("[Fig6Audit] no complete Fig.6 candidate set found.")


if __name__ == "__main__":
    main()
