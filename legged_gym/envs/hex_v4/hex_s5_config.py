from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS5Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s5_sparse_dense"
        scene_seed = 105
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_params_easy = {
            "pole_radius_min": 0.12,
            "pole_radius_max": 0.18,
            "pole_height": 0.3,
            "block_size_min": 0.28,
            "block_size_max": 0.4,
            "block_height_min": 0.3,
            "block_height_max": 0.35,
            "sparse_count": 6,
            "dense_count": 14,
            "boundary_offset": 0.0,
            "boundary_jitter": 0.4,
            "sparse_block_ratio": 0.1,
            "dense_block_ratio": 0.3,
        }
        scene_params_hard = {
            "pole_radius_min": 0.14,
            "pole_radius_max": 0.22,
            "pole_height": 0.35,
            "block_size_min": 0.3,
            "block_size_max": 0.44,
            "block_height_min": 0.35,
            "block_height_max": 0.4,
            "sparse_count": 4,
            "dense_count": 20,
            "boundary_offset": 0.2,
            "boundary_jitter": 0.6,
            "sparse_block_ratio": 0.15,
            "dense_block_ratio": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (30.0, 30.0),
            "width_mul": (20.0, 20.0),
            "split_range": (0.35, 0.65),
            "sparse_count": (20, 60),
            "dense_count": (80, 180),
            "min_dist_sparse_mul": (3.5, 2.8),
            "min_dist_dense_mul": (2.5, 1.8),
            "block_size_mul": (1.5, 2.5),
            "pole_radius_mul": (0.6, 1.0),
            "block_ratio": 0.5,
            "spawn_clear_mul": 3.0,
            "obs_height_mul": (2.0, 3.0),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS5CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s5"


class HexS5LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s5_sparse_dense"
        scene_seed = 105
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_params_easy = {
            "pole_radius_min": 0.12,
            "pole_radius_max": 0.18,
            "pole_height": 0.3,
            "block_size_min": 0.28,
            "block_size_max": 0.4,
            "block_height_min": 0.3,
            "block_height_max": 0.35,
            "sparse_count": 10,
            "dense_count": 14,
            "boundary_offset": 0.0,
            "boundary_jitter": 0.4,
            "sparse_block_ratio": 0.1,
            "dense_block_ratio": 0.3,
        }
        scene_params_hard = {
            "pole_radius_min": 0.14,
            "pole_radius_max": 0.22,
            "pole_height": 0.35,
            "block_size_min": 0.3,
            "block_size_max": 0.44,
            "block_height_min": 0.35,
            "block_height_max": 0.4,
            "sparse_count": 18,
            "dense_count": 30,
            "boundary_offset": 0.2,
            "boundary_jitter": 0.6,
            "sparse_block_ratio": 0.15,
            "dense_block_ratio": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (30.0, 30.0),
            "width_mul": (20.0, 20.0),
            "split_range": (0.35, 0.65),
            "sparse_count": (20, 60),
            "dense_count": (80, 180),
            "min_dist_sparse_mul": (3.5, 2.8),
            "min_dist_dense_mul": (2.5, 1.8),
            "block_size_mul": (1.5, 2.5),
            "pole_radius_mul": (0.6, 1.0),
            "block_ratio": 0.5,
            "spawn_clear_mul": 3.0,
            "obs_height_mul": (2.0, 3.0),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS5LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s5_large"
