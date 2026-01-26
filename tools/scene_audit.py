import argparse
import importlib.util
import json
import os
import sys
import types
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCENE_CFG_MAP = {
    "debug_axis_calib": ("legged_gym.envs.hex_v4.hex_scenes_config", "HexCalibCfg"),
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


def _ensure_pkg(name: str):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _load_module(name: str, path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_legged_gym():
    pkg = _ensure_pkg("legged_gym")
    pkg.LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    pkg.LEGGED_GYM_ENVS_DIR = os.path.join(pkg.LEGGED_GYM_ROOT_DIR, "legged_gym", "envs")
    _ensure_pkg("legged_gym.envs")
    _ensure_pkg("legged_gym.envs.base")
    _ensure_pkg("legged_gym.envs.hex_v4")
    _ensure_pkg("legged_gym.envs.hex_v4.terrain_v2")

    base_base = os.path.join(ROOT, "legged_gym", "envs", "base", "base_config.py")
    _load_module("legged_gym.envs.base.base_config", base_base)
    base_cfg = os.path.join(ROOT, "legged_gym", "envs", "base", "legged_robot_config.py")
    _load_module("legged_gym.envs.base.legged_robot_config", base_cfg)
    hex_ground_cfg = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_ground_config.py")
    _load_module("legged_gym.envs.hex_v4.hex_ground_config", hex_ground_cfg)

    scene_spec = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "terrain_v2", "scene_spec.py")
    _load_module("legged_gym.envs.hex_v4.terrain_v2.scene_spec", scene_spec)
    generator = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "terrain_v2", "scene_generator.py")
    _load_module("legged_gym.envs.hex_v4.terrain_v2.scene_generator", generator)
    backend = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "terrain_v2", "backend_heightfield.py")
    _load_module("legged_gym.envs.hex_v4.terrain_v2.backend_heightfield", backend)
    contracts = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "terrain_v2", "contracts.py")
    _load_module("legged_gym.envs.hex_v4.terrain_v2.contracts", contracts)

    for mod_name, _ in SCENE_CFG_MAP.values():
        file_path = os.path.join(ROOT, *mod_name.split(".")) + ".py"
        _load_module(mod_name, file_path)


def _load_cfg(scene_id: str):
    if scene_id not in SCENE_CFG_MAP:
        raise RuntimeError(f"unsupported scene_id={scene_id}")
    _bootstrap_legged_gym()
    mod_name, cls_name = SCENE_CFG_MAP[scene_id]
    module = sys.modules.get(mod_name)
    cfg_cls = getattr(module, cls_name, None) if module is not None else None
    if cfg_cls is None:
        raise RuntimeError(f"config class {cls_name} not found in {mod_name}")
    return cfg_cls.terrain


def _run_scene(scene_id: str, difficulty: float, seed: int) -> Dict:
    cfg = _load_cfg(scene_id)
    env_dims = {"width": float(cfg.terrain_width), "length": float(cfg.terrain_length)}
    from legged_gym.envs.hex_v4.terrain_v2.scene_generator import SceneGenerator
    from legged_gym.envs.hex_v4.terrain_v2.backend_heightfield import HeightfieldBackend
    from legged_gym.envs.hex_v4.terrain_v2.contracts import check_scene

    generator = SceneGenerator(cfg, env_dims=env_dims, robot_envelope={"clearance": float(getattr(cfg, "scene_clearance", 0.27))})
    backend = HeightfieldBackend(env_dims["width"], env_dims["length"], cfg.horizontal_scale, cfg.vertical_scale)
    scene = generator.sample(scene_id, difficulty, seed)
    heightfield, scene = backend.render(scene, return_scene=True)
    result = check_scene(scene, heightfield, cfg.horizontal_scale)
    return {
        "scene_id": scene_id,
        "difficulty": float(difficulty),
        "seed": int(seed),
        "pass": bool(result["pass"]),
        "metrics": result["metrics"],
        "reasons": result.get("reasons", []),
    }


def _summarize(results: List[Dict]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for item in results:
        scene_id = item["scene_id"]
        if scene_id not in summary:
            summary[scene_id] = {
                "pass": 0,
                "total": 0,
                "reasons": {},
                "door_width_samples": [],
            }
        summary[scene_id]["total"] += 1
        if item["pass"]:
            summary[scene_id]["pass"] += 1
        for reason in item.get("reasons", []):
            summary[scene_id]["reasons"][reason] = summary[scene_id]["reasons"].get(reason, 0) + 1
        metrics = item.get("metrics", {}) or {}
        mean_gate_width = metrics.get("mean_gate_width", None)
        if mean_gate_width is not None:
            summary[scene_id]["door_width_samples"].append(float(mean_gate_width))
    for scene_id, item in summary.items():
        samples = item.get("door_width_samples", [])
        if samples:
            item["door_width_stats"] = {
                "mean": float(sum(samples) / max(1, len(samples))),
                "min": float(min(samples)),
                "max": float(max(samples)),
                "count": int(len(samples)),
            }
        item.pop("door_width_samples", None)
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
        item = summary[scene_id]
        pass_rate = item["pass"] / max(1, item["total"])
        print(f"[Summary] {scene_id} pass_rate={pass_rate:.2%} ({item['total']} cases)")

    report = {
        "scenes": scenes,
        "seed_start": args.seed_start,
        "seed_count": len(seeds),
        "difficulties": diffs,
        "axis_convention": {
            "tile_axis0": "+Y(length)",
            "tile_axis1": "+X(width)",
            "note": "scene_audit uses backend tiles only (no SubTerrain buffer).",
        },
        "axis_details": axis_details,
        "summary": summary,
        "samples": results,
    }

    if args.write_golden:
        golden: Dict[str, List[int]] = {scene: [] for scene in scenes}
        for scene_id in scenes:
            for seed in seeds:
                pass_all = True
                for diff in diffs:
                    row = _run_scene(scene_id, diff, seed)
                    if not row["pass"]:
                        pass_all = False
                        break
                if pass_all:
                    golden[scene_id].append(int(seed))
                if len(golden[scene_id]) >= args.golden_count:
                    break
        with open(args.write_golden, "w", encoding="utf-8") as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)
        report["golden_seeds"] = golden
        print(f"[OK] golden saved: {args.write_golden}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[OK] report saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
