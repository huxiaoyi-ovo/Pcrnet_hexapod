# Debug helper for S2 scene config without importing Isaac Gym.
# Run: python3 tools/debug_s2_scene.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_s2_config.py")
SCENE_PATH = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "scene_manager.py")


def load_hex_s2_cfg(path: str):
    ns = {}
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, ns)
    return ns.get("HexS2Cfg")


def main() -> int:
    if not os.path.exists(CFG_PATH):
        print(f"[Err] hex_s2_config.py not found: {CFG_PATH}")
        return 1

    HexS2Cfg = load_hex_s2_cfg(CFG_PATH)
    if HexS2Cfg is None:
        print("[Err] HexS2Cfg not found in config.")
        return 1

    terrain = HexS2Cfg.terrain
    print("[S2] scene_type:", getattr(terrain, "scene_type", None))
    print("[S2] scene_use_actors:", getattr(terrain, "scene_use_actors", None))
    print("[S2] scene_static_max:", getattr(terrain, "scene_static_max", None))
    print("[S2] scene_static_block_sizes:", getattr(terrain, "scene_static_block_sizes", None))
    print("[S2] scene_static_block_heights:", getattr(terrain, "scene_static_block_heights", None))
    print("[S2] scene_params_easy keys:", sorted(getattr(terrain, "scene_params_easy", {}).keys()))
    print("[S2] scene_params_hard keys:", sorted(getattr(terrain, "scene_params_hard", {}).keys()))

    # Optional: try to sample a scene without importing isaacgym.
    # This uses SceneManager directly; if it fails, it will print a warning.
    try:
        sys.path.insert(0, ROOT)
        from legged_gym.envs.hex_v4.scene_manager import SceneManager

        sm = SceneManager(terrain)
        spec = sm.sample_scene(0.5, env_id=0, episode_idx=0)
        print("[S2] sampled scene_type:", spec.scene_type)
        print("[S2] num_static:", len(spec.static_obstacles))
        if spec.static_obstacles:
            print("[S2] first3:", spec.static_obstacles[:3])
    except Exception as exc:
        print("[Warn] SceneManager sample failed:", repr(exc))
        print("[Warn] This does not block config check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
