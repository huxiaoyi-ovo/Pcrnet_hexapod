from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS2Cfg(HexGroundCfg):
    class terrain(HexGroundCfg.terrain):
        fixed_layout_enable = False
        scene_type = "s2_doorway"
        scene_seed = 102
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 80
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_params_easy = {
            "room_width": 2.6,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "door_count": 2,
            "door_thickness": 0.25,
            "door_width_min": 0.9,
            "door_width_max": 1.3,
            "door_offset_max": 0.5,
            "door_margin_y": 0.8,
            "jam_count": 0,
            "jam_size": 0.35,
            "door_block_width": 0.4,
        }
        scene_params_hard = {
            "room_width": 2.1,
            "wall_height": 0.35,
            "wall_thickness": 0.25,
            "door_count": 3,
            "door_thickness": 0.3,
            "door_width_min": 0.7,
            "door_width_max": 1.0,
            "door_offset_max": 0.6,
            "door_margin_y": 0.6,
            "jam_count": 1,
            "jam_size": 0.4,
            "door_block_width": 0.4,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS2CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s2"
