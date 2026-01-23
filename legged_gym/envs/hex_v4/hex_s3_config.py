from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS3Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s3_doorway"
        scene_seed = 103
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
        scene_static_wall_block_size = 0.8
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "room_width": 6.5,
            "wall_thickness": 0.25,
            "room_count": 3,
            "room_boundary_jitter": 0.4,
            "door_thickness": 0.25,
            "door_width_min": 0.9,
            "door_width_max": 1.3,
            "door_offset_max": 0.5,
            "door_block_width": 0.4,
            "jam_count": 0,
            "jam_size": 0.35,
            "room_obstacle_count_min": 0,
            "room_obstacle_count_max": 1,
            "room_obstacle_size_min": 0.28,
            "room_obstacle_size_max": 0.38,
            "room_obstacle_height_min": 0.3,
            "room_obstacle_height_max": 0.35,
        }
        scene_params_hard = {
            "room_width": 6.0,
            "wall_thickness": 0.25,
            "room_count": 4,
            "room_boundary_jitter": 0.6,
            "door_thickness": 0.3,
            "door_width_min": 0.7,
            "door_width_max": 1.0,
            "door_offset_max": 0.6,
            "door_block_width": 0.4,
            "jam_count": 1,
            "jam_size": 0.4,
            "room_obstacle_count_min": 1,
            "room_obstacle_count_max": 2,
            "room_obstacle_size_min": 0.3,
            "room_obstacle_size_max": 0.44,
            "room_obstacle_height_min": 0.35,
            "room_obstacle_height_max": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (24.0, 32.0),
            "width_mul": (18.0, 22.0),
            "wall_thickness_mul": 2.0,
            "door_width_mul": (4.0, 2.2),
            "wall_count": (2, 5),
            "outer_walls": True,
            "door_zigzag": True,
            "wall_height_mul": (2.5, 3.5),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS3CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s3"


class HexS3LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        scene_type = "s3_doorway"
        scene_seed = 103
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
        scene_static_wall_block_size = 1.0
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "room_width": 2.6,
            "wall_thickness": 0.25,
            "room_count": 3,
            "room_boundary_jitter": 0.4,
            "door_thickness": 0.25,
            "door_width_min": 0.9,
            "door_width_max": 1.3,
            "door_offset_max": 0.5,
            "door_block_width": 1.0,
            "jam_count": 0,
            "jam_size": 0.35,
            "room_obstacle_count_min": 0,
            "room_obstacle_count_max": 1,
            "room_obstacle_size_min": 0.28,
            "room_obstacle_size_max": 0.38,
            "room_obstacle_height_min": 0.3,
            "room_obstacle_height_max": 0.35,
        }
        scene_params_hard = {
            "room_width": 2.1,
            "wall_thickness": 0.25,
            "room_count": 3,
            "room_boundary_jitter": 0.6,
            "door_thickness": 0.3,
            "door_width_min": 0.7,
            "door_width_max": 1.0,
            "door_offset_max": 0.6,
            "door_block_width": 1.0,
            "jam_count": 1,
            "jam_size": 0.4,
            "room_obstacle_count_min": 1,
            "room_obstacle_count_max": 1,
            "room_obstacle_size_min": 0.3,
            "room_obstacle_size_max": 0.44,
            "room_obstacle_height_min": 0.35,
            "room_obstacle_height_max": 0.4,
        }
        scene_cfg = {
            "clearance": 0.27,
            "length_mul": (24.0, 32.0),
            "width_mul": (18.0, 22.0),
            "wall_thickness_mul": 2.0,
            "door_width_mul": (4.0, 2.2),
            "wall_count": (2, 5),
            "outer_walls": True,
            "door_zigzag": True,
            "wall_height_mul": (2.5, 3.5),
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS3LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s3_large"
