from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS4Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s4_crossing"
        scene_seed = 104
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_params_easy = {
            "cross_width": 3.0,
            "cross_span": 2.6,
            "dynamic_count_min": 2,
            "dynamic_count_max": 4,
            "dynamic_size_xy": 0.4,
            "dynamic_height": 0.5,
            "react_steps_min": 12,
            "react_steps_max": 20,
            "dynamic_axis": "x",
        }
        scene_params_hard = {
            "cross_width": 3.4,
            "cross_span": 3.0,
            "dynamic_count_min": 4,
            "dynamic_count_max": 6,
            "dynamic_size_xy": 0.4,
            "dynamic_height": 0.5,
            "react_steps_min": 8,
            "react_steps_max": 12,
            "dynamic_axis": "x",
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS4CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s4"


class HexS4LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s4_crossing"
        scene_seed = 104
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_params_easy = {
            "cross_width": 3.0,
            "cross_span": 2.6,
            "dynamic_count_min": 1,
            "dynamic_count_max": 2,
            "dynamic_size_xy": 0.4,
            "dynamic_height": 0.5,
            "react_steps_min": 12,
            "react_steps_max": 20,
            "dynamic_axis": "x",
            "static_pole_count": 0,
        }
        scene_params_hard = {
            "cross_width": 3.4,
            "cross_span": 3.0,
            "dynamic_count_min": 2,
            "dynamic_count_max": 2,
            "dynamic_size_xy": 0.4,
            "dynamic_height": 0.5,
            "react_steps_min": 8,
            "react_steps_max": 12,
            "dynamic_axis": "x",
            "static_pole_count": 4,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS4LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s4_large"
