from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS5Cfg(HexGroundCfg):
    class terrain(HexGroundCfg.terrain):
        fixed_layout_enable = False
        scene_type = "s5_transition"
        scene_seed = 105
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 60
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.24, 0.44]
        scene_static_block_heights = [0.3, 0.35]
        scene_params_easy = {
            "pole_radius_min": 0.12,
            "pole_radius_max": 0.18,
            "pole_height": 0.3,
            "sparse_count": 6,
            "dense_count": 14,
            "boundary_offset": 0.0,
            "boundary_jitter": 0.4,
        }
        scene_params_hard = {
            "pole_radius_min": 0.14,
            "pole_radius_max": 0.22,
            "pole_height": 0.35,
            "sparse_count": 4,
            "dense_count": 20,
            "boundary_offset": 0.2,
            "boundary_jitter": 0.6,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS5CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s5"
