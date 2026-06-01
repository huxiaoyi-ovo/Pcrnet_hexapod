# Debug helper for S2 scene config using classic builder.
#   Run: python3 tools/debug_s2_scene.py

import importlib.util
import os
import sys
import types

import numpy as np
from isaacgym import terrain_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_scenes_config.py")


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


def load_hex_s2_cfg(path: str):
    _ensure_pkg("legged_gym")
    _ensure_pkg("legged_gym.envs")
    _ensure_pkg("legged_gym.envs.base")
    _ensure_pkg("legged_gym.envs.hex_v4")
    base_base = os.path.join(ROOT, "legged_gym", "envs", "base", "base_config.py")
    _load_module("legged_gym.envs.base.base_config", base_base)
    base_cfg = os.path.join(ROOT, "legged_gym", "envs", "base", "legged_robot_config.py")
    _load_module("legged_gym.envs.base.legged_robot_config", base_cfg)
    hex_ground_cfg = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_ground_config.py")
    _load_module("legged_gym.envs.hex_v4.hex_ground_config", hex_ground_cfg)
    _load_module("legged_gym.envs.hex_v4.hex_scenes_config", path)
    mod = sys.modules.get("legged_gym.envs.hex_v4.hex_scenes_config")
    return getattr(mod, "HexS2Cfg", None)


def main() -> int:
    if not os.path.exists(CFG_PATH):
        print(f"[Err] hex_scenes_config.py not found: {CFG_PATH}")
        return 1

    HexS2Cfg = load_hex_s2_cfg(CFG_PATH)
    if HexS2Cfg is None:
        print("[Err] HexS2Cfg not found in config.")
        return 1

    from legged_gym.utils.terrain import s2_forest_terrain

    cfg = HexS2Cfg.terrain
    seed = int(getattr(cfg, "terrain_seed", 0) or 0)
    rng = np.random.RandomState(seed)

    width_px = int(cfg.terrain_width / cfg.horizontal_scale)
    length_px = int(cfg.terrain_length / cfg.horizontal_scale)
    terrain = terrain_utils.SubTerrain(
        "terrain",
        width=width_px,
        length=length_px,
        vertical_scale=cfg.vertical_scale,
        horizontal_scale=cfg.horizontal_scale,
    )
    terrain = s2_forest_terrain(terrain, difficulty=0.5, rng=rng, cfg=cfg, seed=seed)

    meta = getattr(terrain, "meta", {}) or {}
    params = meta.get("params", {}) or {}
    print("[S2] scene_type:", meta.get("scene_type"))
    print("[S2] count_total:", params.get("count_total"))
    print("[S2] num_poles:", params.get("num_poles"))
    print("[S2] num_blocks:", params.get("num_blocks"))
    print("[S2] block_ratio:", params.get("block_ratio"))
    print("[S2] scene_params_easy keys:", sorted(getattr(cfg, "scene_params_easy", {}).keys()))
    print("[S2] scene_params_hard keys:", sorted(getattr(cfg, "scene_params_hard", {}).keys()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
