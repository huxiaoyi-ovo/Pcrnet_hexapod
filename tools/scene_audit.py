import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
from isaacgym import terrain_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from legged_gym.utils.terrain import (
    debug_axis_terrain,
    s1_corridor_gate_terrain,
    s2_forest_terrain,
)

SCENE_CFG_MAP = {
    "debug_axis": ("legged_gym.envs.hex_v4.hex_scenes_config", "HexCalibCfg"),
    "s1_corridor_gate": ("legged_gym.envs.hex_v4.hex_scenes_config", "HexS1Cfg"),
    "s2_forest": ("legged_gym.envs.hex_v4.hex_scenes_config", "HexS2Cfg"),
}


def _parse_list(value: str, cast=float) -> List:
    if value is None:
        return []
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def _load_cfg(scene_id: str):
    if scene_id not in SCENE_CFG_MAP:
        raise RuntimeError(f"unsupported scene_id={scene_id}")
    mod_name, cls_name = SCENE_CFG_MAP[scene_id]
    module = __import__(mod_name, fromlist=[cls_name])
    cfg_cls = getattr(module, cls_name)
    return cfg_cls.terrain


def _make_subterrain(cfg):
    width_px = int(cfg.terrain_width / cfg.horizontal_scale)
    length_px = int(cfg.terrain_length / cfg.horizontal_scale)
    return terrain_utils.SubTerrain(
        "terrain",
        width=width_px,
        length=length_px,
        vertical_scale=cfg.vertical_scale,
        horizontal_scale=cfg.horizontal_scale,
    )


def _run_scene(scene_id: str, difficulty: float, seed: int) -> Dict:
    cfg = _load_cfg(scene_id)
    rng = np.random.RandomState(seed)
    terrain = _make_subterrain(cfg)
    if scene_id == "debug_axis":
        terrain = debug_axis_terrain(terrain, difficulty, rng, cfg, seed=seed)
    elif scene_id == "s1_corridor_gate":
        terrain = s1_corridor_gate_terrain(terrain, difficulty, rng, cfg, seed=seed)
    elif scene_id == "s2_forest":
        terrain = s2_forest_terrain(terrain, difficulty, rng, cfg, seed=seed)
    else:
        raise RuntimeError(f"unsupported scene_id={scene_id}")

    meta = getattr(terrain, "meta", {}) or {}
    params = meta.get("params", {}) or {}
    metrics: Dict[str, float] = {}

    if scene_id == "s1_corridor_gate":
        gates = params.get("corridor_gates", []) or []
        door_widths = [float(g.get("door_width", 0.0)) for g in gates if g.get("door_width") is not None]
        metrics["corridor_width"] = float(params.get("corridor_width_nom", 0.0))
        metrics["gate_count"] = float(len(gates))
        if door_widths:
            metrics["door_width_mean"] = float(sum(door_widths) / len(door_widths))
            metrics["door_width_min"] = float(min(door_widths))
            metrics["door_width_max"] = float(max(door_widths))
    elif scene_id == "s2_forest":
        metrics["count_total"] = float(params.get("count_total", 0))
        metrics["num_poles"] = float(params.get("num_poles", 0))
        metrics["num_blocks"] = float(params.get("num_blocks", 0))
        metrics["block_ratio"] = float(params.get("block_ratio", 0.0))

    return {
        "scene_id": scene_id,
        "difficulty": float(difficulty),
        "seed": int(seed),
        "metrics": metrics,
        "meta": meta,
    }


def _summarize(results: List[Dict]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for item in results:
        scene_id = item["scene_id"]
        if scene_id not in summary:
            summary[scene_id] = {
                "total": 0,
                "door_width_samples": [],
                "gate_count_samples": [],
                "count_total_samples": [],
                "block_ratio_samples": [],
            }
        summary[scene_id]["total"] += 1
        metrics = item.get("metrics", {}) or {}
        if "door_width_mean" in metrics:
            summary[scene_id]["door_width_samples"].append(metrics["door_width_mean"])
        if "gate_count" in metrics:
            summary[scene_id]["gate_count_samples"].append(metrics["gate_count"])
        if "count_total" in metrics:
            summary[scene_id]["count_total_samples"].append(metrics["count_total"])
        if "block_ratio" in metrics:
            summary[scene_id]["block_ratio_samples"].append(metrics["block_ratio"])

    for scene_id, item in summary.items():
        door_samples = item.get("door_width_samples", [])
        if door_samples:
            item["door_width_stats"] = {
                "mean": float(sum(door_samples) / len(door_samples)),
                "min": float(min(door_samples)),
                "max": float(max(door_samples)),
                "count": int(len(door_samples)),
            }
        gate_samples = item.get("gate_count_samples", [])
        if gate_samples:
            item["gate_count_stats"] = {
                "mean": float(sum(gate_samples) / len(gate_samples)),
                "min": float(min(gate_samples)),
                "max": float(max(gate_samples)),
                "count": int(len(gate_samples)),
            }
        count_samples = item.get("count_total_samples", [])
        if count_samples:
            item["count_total_stats"] = {
                "mean": float(sum(count_samples) / len(count_samples)),
                "min": float(min(count_samples)),
                "max": float(max(count_samples)),
                "count": int(len(count_samples)),
            }
        ratio_samples = item.get("block_ratio_samples", [])
        if ratio_samples:
            item["block_ratio_stats"] = {
                "mean": float(sum(ratio_samples) / len(ratio_samples)),
                "min": float(min(ratio_samples)),
                "max": float(max(ratio_samples)),
                "count": int(len(ratio_samples)),
            }

        item.pop("door_width_samples", None)
        item.pop("gate_count_samples", None)
        item.pop("count_total_samples", None)
        item.pop("block_ratio_samples", None)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, default="s1_corridor_gate,s2_forest")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--difficulties", type=str, default="0,0.5,1")
    parser.add_argument("--golden-count", type=int, default=5)
    parser.add_argument("--write-golden", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    scenes = _parse_list(args.scenes, cast=str)
    diffs = _parse_list(args.difficulties, cast=float)
    if args.seeds:
        seeds = _parse_list(args.seeds, cast=int)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    results: List[Dict] = []
    for scene_id in scenes:
        for seed in seeds:
            for diff in diffs:
                results.append(_run_scene(scene_id, diff, seed))

    axis_details: Dict[str, Dict] = {}
    for scene_id in scenes:
        try:
            cfg = _load_cfg(scene_id)
            length_m = float(cfg.terrain_length)
            width_m = float(cfg.terrain_width)
            h_scale = float(cfg.horizontal_scale)
            axis_details[scene_id] = {
                "env_length_m": length_m,
                "env_width_m": width_m,
                "length_px": int(round(length_m / h_scale)),
                "width_px": int(round(width_m / h_scale)),
                "horizontal_scale": h_scale,
            }
        except Exception as exc:
            axis_details[scene_id] = {"error": str(exc)}

    summary = _summarize(results)
    for scene_id in scenes:
        if scene_id not in summary:
            continue
        print(f"[Summary] {scene_id} total={summary[scene_id]['total']}")

    report = {
        "scenes": scenes,
        "seed_start": args.seed_start,
        "seed_count": len(seeds),
        "difficulties": diffs,
        "axis_convention": {
            "tile_axis0": "+Y(length)",
            "tile_axis1": "+X(width)",
            "note": "classic tile builder",
        },
        "axis_details": axis_details,
        "summary": summary,
        "samples": results,
    }

    if args.write_golden:
        golden: Dict[str, List[int]] = {scene: [] for scene in scenes}
        for scene_id in scenes:
            golden[scene_id] = seeds[: max(0, int(args.golden_count))]
        with open(args.write_golden, "w", encoding="utf-8") as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
