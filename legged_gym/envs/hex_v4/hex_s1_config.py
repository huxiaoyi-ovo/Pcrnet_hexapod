from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS1Cfg(HexGroundCfg):
    class terrain(HexGroundCfg.terrain):
        fixed_layout_enable = False
        scene_type = "s1_corridor"
        scene_seed = 101
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 80
        scene_static_block_max = 40
        scene_static_wall_max = 160
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_wall_block_size = 1.6
        scene_params_easy = {
            "corridor_width": 1.6,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "wall_segment_count": 16,
            "gate_count": 2,
            "gate_thickness": 0.25,
            "gap_width_min": 0.8,
            "gap_width_max": 1.1,
            "gate_offset_max": 0.3,
            "gate_margin_y": 0.8,
            "gate_block_width": 0.4,
        }
        scene_params_hard = {
            "corridor_width": 1.1,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "wall_segment_count": 20,
            "gate_count": 3,
            "gate_thickness": 0.3,
            "gap_width_min": 0.6,
            "gap_width_max": 0.9,
            "gate_offset_max": 0.4,
            "gate_margin_y": 0.6,
            "gate_block_width": 0.4,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS1CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1"


class HexS1FollowCfg(HexGroundCfg):
    class terrain(HexGroundCfg.terrain):
        fixed_layout_enable = False
        scene_type = "s1_corridor"
        scene_seed = 101
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 80
        scene_static_block_max = 40
        scene_static_wall_max = 160
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_wall_block_size = 1.6
        scene_params_easy = {
            "corridor_width": 1.6,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "wall_segment_count": 16,
            "gate_count": 2,
            "gate_thickness": 0.25,
            "gap_width_min": 0.8,
            "gap_width_max": 1.1,
            "gate_offset_max": 0.3,
            "gate_margin_y": 0.8,
            "gate_block_width": 0.4,
        }
        scene_params_hard = {
            "corridor_width": 1.1,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "wall_segment_count": 20,
            "gate_count": 3,
            "gate_thickness": 0.3,
            "gap_width_min": 0.6,
            "gap_width_max": 0.9,
            "gate_offset_max": 0.4,
            "gate_margin_y": 0.6,
            "gate_block_width": 0.4,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4

    class navigation(HexGroundCfg.navigation):
        goal_force_blocking_line = False
        goal_force_blocking_prob = 0.0


class HexS1FollowCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_follow"
