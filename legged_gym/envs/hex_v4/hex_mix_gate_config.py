from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexMixGateCfg(HexGroundCfg):
    class terrain(HexGroundCfg.terrain):
        fixed_layout_enable = False
        scene_types = ["s3_forest", "s4_crossing", "s5_transition"]
        scene_seed = 120
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 80
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.24, 0.44]
        scene_static_block_heights = [0.3, 0.35]

        scene_probs_easy = {
            "s3_forest": 0.2,
            "s4_crossing": 0.6,
            "s5_transition": 0.2,
        }
        scene_probs_hard = {
            "s3_forest": 0.3,
            "s4_crossing": 0.4,
            "s5_transition": 0.3,
        }

        scene_params_easy = {
            "s3_forest": {
                "pole_count_min": 8,
                "pole_count_max": 12,
                "pole_radius_min": 0.12,
                "pole_radius_max": 0.18,
                "pole_height": 0.3,
                "pole_margin": 0.4,
            },
            "s4_crossing": {
                "cross_width": 3.0,
                "cross_span": 2.6,
                "dynamic_count_min": 2,
                "dynamic_count_max": 4,
                "dynamic_size_xy": 0.4,
                "dynamic_height": 0.5,
                "react_steps_min": 12,
                "react_steps_max": 20,
                "dynamic_axis": "x",
                "static_pole_count": 0,
            },
            "s5_transition": {
                "pole_radius_min": 0.12,
                "pole_radius_max": 0.18,
                "pole_height": 0.3,
                "sparse_count": 6,
                "dense_count": 14,
                "boundary_offset": 0.0,
                "boundary_jitter": 0.4,
            },
        }
        scene_params_hard = {
            "s3_forest": {
                "pole_count_min": 16,
                "pole_count_max": 24,
                "pole_radius_min": 0.14,
                "pole_radius_max": 0.22,
                "pole_height": 0.35,
                "pole_margin": 0.3,
            },
            "s4_crossing": {
                "cross_width": 3.4,
                "cross_span": 3.0,
                "dynamic_count_min": 4,
                "dynamic_count_max": 6,
                "dynamic_size_xy": 0.4,
                "dynamic_height": 0.5,
                "react_steps_min": 8,
                "react_steps_max": 12,
                "dynamic_axis": "x",
                "static_pole_count": 0,
            },
            "s5_transition": {
                "pole_radius_min": 0.14,
                "pole_radius_max": 0.22,
                "pole_height": 0.35,
                "sparse_count": 4,
                "dense_count": 20,
                "boundary_offset": 0.2,
                "boundary_jitter": 0.6,
            },
        }

        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexMixGateCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_mix_gate"
