from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS6Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s6_ood_cluster"
        scene_seed = 106
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.32, 0.4, 0.48]
        scene_static_block_heights = [0.35, 0.4, 0.45]
        scene_static_wall_block_size = 0.8
        scene_static_wall_block_height = 0.4
        scene_params_easy = {
            "ood_template": "mix",
            "ood_template_probs": {"cluster": 1.0, "nonconvex": 1.0, "maze": 1.0},
            "u_width": 2.0,
            "u_depth": 1.4,
            "u_thickness": 0.25,
            "l_size": 1.4,
            "l_thickness": 0.25,
            "maze_rows": 3,
            "maze_cols": 3,
            "maze_gap_width": 0.9,
            "maze_wall_thickness": 0.25,
            "cluster_count": 6,
            "cluster_radius": 0.18,
            "cluster_spread": 0.6,
            "obstacle_height": 0.35,
        }
        scene_params_hard = {
            "ood_template": "mix",
            "ood_template_probs": {"cluster": 1.0, "nonconvex": 1.0, "maze": 1.0},
            "u_width": 2.4,
            "u_depth": 1.7,
            "u_thickness": 0.3,
            "l_size": 1.8,
            "l_thickness": 0.3,
            "maze_rows": 4,
            "maze_cols": 4,
            "maze_gap_width": 0.8,
            "maze_wall_thickness": 0.3,
            "cluster_count": 10,
            "cluster_radius": 0.22,
            "cluster_spread": 0.8,
            "obstacle_height": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (20.0, 30.0),
            "width_mul": (16.0, 20.0),
            "cluster_count": (3, 6),
            "cluster_radius_mul": (4.0, 6.0),
            "cluster_size": (10, 30),
            "cluster_sigma_mul": (1.5, 2.5),
            "block_size_mul": (1.5, 2.5),
            "pole_radius_mul": (0.6, 1.0),
            "block_ratio": 0.5,
            "obs_height_mul": (2.0, 3.0),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS6CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s6"


class HexS6LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s6_ood_cluster"
        scene_seed = 106
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.32, 0.4, 0.48]
        scene_static_block_heights = [0.35, 0.4, 0.45]
        scene_static_wall_block_size = 1.0
        scene_static_wall_block_height = 0.4
        scene_params_easy = {
            "ood_template": "cluster",
            "ood_template_probs": {"cluster": 1.0, "nonconvex": 1.0, "maze": 1.0},
            "u_width": 2.0,
            "u_depth": 1.4,
            "u_thickness": 0.25,
            "l_size": 1.4,
            "l_thickness": 0.25,
            "maze_rows": 3,
            "maze_cols": 3,
            "maze_gap_width": 0.9,
            "maze_wall_thickness": 0.25,
            "cluster_count": 6,
            "cluster_radius": 0.18,
            "cluster_spread": 0.6,
            "obstacle_height": 0.35,
        }
        scene_params_hard = {
            "ood_template": "cluster",
            "ood_template_probs": {"cluster": 1.0, "nonconvex": 1.0, "maze": 1.0},
            "u_width": 2.2,
            "u_depth": 1.6,
            "u_thickness": 0.3,
            "l_size": 1.6,
            "l_thickness": 0.3,
            "maze_rows": 3,
            "maze_cols": 3,
            "maze_gap_width": 0.8,
            "maze_wall_thickness": 0.3,
            "cluster_count": 8,
            "cluster_radius": 0.22,
            "cluster_spread": 0.8,
            "obstacle_height": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (20.0, 30.0),
            "width_mul": (16.0, 20.0),
            "cluster_count": (3, 6),
            "cluster_radius_mul": (4.0, 6.0),
            "cluster_size": (10, 30),
            "cluster_sigma_mul": (1.5, 2.5),
            "block_size_mul": (1.5, 2.5),
            "pole_radius_mul": (0.6, 1.0),
            "block_ratio": 0.5,
            "obs_height_mul": (2.0, 3.0),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS6LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s6_large"
