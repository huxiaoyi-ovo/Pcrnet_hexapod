# Debug helper for S2 scene config without importing Isaac Gym.
# Run: python3 tools/debug_s2_scene.py

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "legged_gym", "envs", "hex_v4", "hex_s2_config.py")


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
    print("[S2] scene_static_block_sizes:", getattr(terrain, "scene_static_block_sizes", None))
    print("[S2] scene_static_block_heights:", getattr(terrain, "scene_static_block_heights", None))
    print("[S2] scene_params_easy keys:", sorted(getattr(terrain, "scene_params_easy", {}).keys()))
    print("[S2] scene_params_hard keys:", sorted(getattr(terrain, "scene_params_hard", {}).keys()))

    try:
        sys.path.insert(0, ROOT)
        from legged_gym.envs.hex_v4.scene_gen_v2.scene_generator import SceneGenerator
        from legged_gym.envs.hex_v4.scene_gen_v2.backend_heightfield import HeightfieldBackend
        from legged_gym.envs.hex_v4.scene_gen_v2.contracts import check_scene

        env_dims = {"width": float(terrain.terrain_width), "length": float(terrain.terrain_length)}
        gen = SceneGenerator(terrain, env_dims=env_dims, robot_envelope={"clearance": float(getattr(terrain, "scene_clearance", 0.27))})
        backend = HeightfieldBackend(env_dims["width"], env_dims["length"], terrain.horizontal_scale, terrain.vertical_scale)
        spec = gen.sample("s2_forest", 0.5, seed=terrain.scene_seed)
        hf, spec = backend.render(spec, return_scene=True)
        result = check_scene(spec, hf, terrain.horizontal_scale)
        print("[S2] sampled scene_type:", spec.scene_type)
        print("[S2] num_static:", len(spec.static_obstacles))
        print("[S2] contract pass:", result["pass"])
    except Exception as exc:
        print("[Warn] scene_gen_v2 sample failed:", repr(exc))
        print("[Warn] This does not block config check.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
