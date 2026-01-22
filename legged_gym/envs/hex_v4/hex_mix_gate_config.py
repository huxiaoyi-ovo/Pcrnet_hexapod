from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexMixGateCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
        fixed_layout_enable = False
        scene_types = ["s3_doorway", "s4_crossing", "s5_transition"]
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
        scene_static_block_max = 40
        scene_static_wall_max = 120
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_static_wall_block_size = 0.8
        scene_static_wall_block_height = 0.35

        scene_probs_easy = {
            "s3_doorway": 0.2,
            "s4_crossing": 0.6,
            "s5_transition": 0.2,
        }
        scene_probs_hard = {
            "s3_doorway": 0.3,
            "s4_crossing": 0.4,
            "s5_transition": 0.3,
        }

        scene_params_easy = {
            "s3_doorway": {
                "room_width": 2.4,
                "wall_thickness": 0.25,
                "room_count": 3,
                "room_boundary_jitter": 0.4,
                "door_thickness": 0.25,
                "door_width_min": 0.9,
                "door_width_max": 1.2,
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
            },
        }
        scene_params_hard = {
            "s3_doorway": {
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
            },
        }

        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexMixGateCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_mix_gate"


class HexMixGateLargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
        fixed_layout_enable = False
        scene_types = ["s3_doorway", "s4_crossing", "s5_transition"]
        scene_seed = 120
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_actors = True
        scene_use_heightfield = False
        scene_resample_on_reset = True
        scene_resample_on_level_change = True
        scene_static_max = 0
        scene_static_block_max = 32
        scene_static_wall_max = 40
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_static_wall_block_size = 1.0
        scene_static_wall_block_height = 0.35
        scene_actor_budget = 80

        scene_probs_easy = {
            "s3_doorway": 0.2,
            "s4_crossing": 0.6,
            "s5_transition": 0.2,
        }
        scene_probs_hard = {
            "s3_doorway": 0.3,
            "s4_crossing": 0.4,
            "s5_transition": 0.3,
        }

        scene_params_easy = {
            "s3_doorway": {
                "room_width": 6.5,
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
            },
            "s4_crossing": {
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
            },
            "s5_transition": {
                "pole_radius_min": 0.12,
                "pole_radius_max": 0.18,
                "pole_height": 0.3,
                "block_size_min": 0.28,
                "block_size_max": 0.4,
                "block_height_min": 0.3,
                "block_height_max": 0.35,
                "sparse_count": 8,
                "dense_count": 12,
                "boundary_offset": 0.0,
                "boundary_jitter": 0.4,
                "sparse_block_ratio": 0.1,
                "dense_block_ratio": 0.3,
            },
        }
        scene_params_hard = {
            "s3_doorway": {
                "room_width": 6.0,
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
            },
            "s4_crossing": {
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
            },
            "s5_transition": {
                "pole_radius_min": 0.14,
                "pole_radius_max": 0.22,
                "pole_height": 0.35,
                "block_size_min": 0.3,
                "block_size_max": 0.44,
                "block_height_min": 0.35,
                "block_height_max": 0.4,
                "sparse_count": 12,
                "dense_count": 20,
                "boundary_offset": 0.2,
                "boundary_jitter": 0.6,
                "sparse_block_ratio": 0.15,
                "dense_block_ratio": 0.4,
            },
        }

        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexMixGateLargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_mix_gate_large"
