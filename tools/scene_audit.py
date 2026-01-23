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
    "s1_corridor": ("legged_gym.envs.hex_v4.hex_s1_config", "HexS1Cfg"),
    "s2_forest": ("legged_gym.envs.hex_v4.hex_s2_config", "HexS2Cfg"),
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
    _ensure_pkg("legged_gym.envs.hex_v4.scene_gen_v2")

    base_base = os.path.join(ROOT, "legged_gym", "envs", "base", "base_config.py")
    _load_module("legged_gym.envs.base.base_config", base_base)
    base_cfg = os.path.join(ROOT, "legged_gym", "envs", "base", "legged_robot_config.py")
    _load_module("legged_gym.envs.base.legged_robot_config", base_cfg)
    hex_ground_cfg = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_ground_config.py")
    _load_module("legged_gym.envs.hex_v4.hex_ground_config", hex_ground_cfg)

    scene_spec = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "scene_spec.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.scene_spec", scene_spec)
    quantizer = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "quantizer.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.quantizer", quantizer)
    guards = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "guards.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.guards", guards)
    generator = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "scene_generator.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.scene_generator", generator)
    backend = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "backend_heightfield.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.backend_heightfield", backend)
    contracts = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_gen_v2", "contracts.py")
    _load_module("legged_gym.envs.hex_v4.scene_gen_v2.contracts", contracts)

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
    from legged_gym.envs.hex_v4.scene_gen_v2.scene_generator import SceneGenerator
    from legged_gym.envs.hex_v4.scene_gen_v2.backend_heightfield import HeightfieldBackend
    from legged_gym.envs.hex_v4.scene_gen_v2.contracts import check_scene

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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, default="s1_corridor,s2_forest")
    parser.add_argument("--seeds", type=str, default="101,102,103,104,105")
    parser.add_argument("--difficulties", type=str, default="0,0.5,1")
    parser.add_argument("--golden-count", type=int, default=3)
    parser.add_argument("--write-golden", type=str, default="")
    args = parser.parse_args()

    scenes = _parse_list(args.scenes, cast=str)
    seeds = _parse_list(args.seeds, cast=int)
    diffs = _parse_list(args.difficulties, cast=float)
    if not scenes:
        scenes = ["s1_corridor", "s2_forest"]

    results = []
    for scene_id in scenes:
        for seed in seeds:
            for diff in diffs:
                results.append(_run_scene(scene_id, diff, seed))

    for item in results:
        metrics = item["metrics"]
        metric_str = ", ".join([f"{k}={metrics[k]}" for k in sorted(metrics.keys())])
        print(
            f"[{item['scene_id']}] seed={item['seed']} diff={item['difficulty']:.2f} "
            f"pass={item['pass']} {metric_str}"
        )

    for scene_id in scenes:
        items = [r for r in results if r["scene_id"] == scene_id]
        if not items:
            continue
        pass_rate = sum(1 for r in items if r["pass"]) / max(1, len(items))
        print(f"[Summary] {scene_id} pass_rate={pass_rate:.2%} ({len(items)} cases)")

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
        print(f"[OK] golden saved: {args.write_golden}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
