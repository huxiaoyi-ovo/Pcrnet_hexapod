from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS3Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
        fixed_layout_enable = False
        scene_type = "s3_doorway"
        scene_seed = 103
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 120
        scene_static_block_max = 40
        scene_static_wall_max = 160
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_static_wall_block_size = 0.8
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
            "room_width": 2.1,
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
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS3CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s3"
