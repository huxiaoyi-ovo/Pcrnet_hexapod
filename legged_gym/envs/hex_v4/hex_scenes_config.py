from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexDebugPlaneCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 8.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
        terrain_v2_enable = False
        scene_use_heightfield = False
        debug_allow_plane = True
        curriculum = False
        num_rows = 1
        num_cols = 1


class HexDebugPlaneCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_debug_plane"


class HexDebugHeightfieldCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 8.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_v2_enable = True
        scene_type = "debug_axis_calib"
        scene_seed = 7
        scene_use_heightfield = True
        debug_allow_plane = False
        curriculum = False
        num_rows = 0
        num_cols = 0


class HexDebugHeightfieldCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_debug_heightfield"


class HexCalibCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_length = 12.0
        terrain_width = 6.0
        terrain_v2_enable = True
        scene_type = "debug_axis_calib"
        scene_seed = 7
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 0.6
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "step_count": 6,
            "step_height": 0.08,
            "edge_margin": 0.8,
            "spawn_length": 1.0,
            "goal_length": 1.0,
        }
        scene_params_hard = {
            "step_count": 6,
            "step_height": 0.08,
            "edge_margin": 0.8,
            "spawn_length": 1.0,
            "goal_length": 1.0,
        }
        num_rows = 0
        num_cols = 0
        terrain_v2_max_rows = 10
        terrain_proportions = [1.0]
        max_init_terrain_level = 0


class HexCalibCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_calib"


class HexS1Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s1_corridor_gate"
        scene_seed = 101
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 0.6
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "corridor_width": 1.6,
            "gate_width": 0.9,
            "gate_count": 2,
            "gate_length": 1.0,
            "gate_length_jitter": 0.2,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.8,
            "wall_thickness_m": 0.16,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.0,
            "corridor_goal_min_offset": 2.0,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        scene_params_hard = {
            "corridor_width": 1.2,
            "gate_width": 0.65,
            "gate_count": 3,
            "gate_length": 1.2,
            "gate_length_jitter": 0.25,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.6,
            "wall_thickness_m": 0.14,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.5,
            "corridor_goal_min_offset": 2.5,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        num_rows = 0
        num_cols = 0
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS1CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1"


class HexS1FollowCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s1_corridor_gate"
        scene_seed = 101
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 0.6
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "corridor_width": 1.6,
            "gate_width": 0.9,
            "gate_count": 2,
            "gate_length": 1.0,
            "gate_length_jitter": 0.2,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.8,
            "wall_thickness_m": 0.16,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.0,
            "corridor_goal_min_offset": 2.0,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        scene_params_hard = {
            "corridor_width": 1.2,
            "gate_width": 0.65,
            "gate_count": 3,
            "gate_length": 1.2,
            "gate_length_jitter": 0.25,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.6,
            "wall_thickness_m": 0.14,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.5,
            "corridor_goal_min_offset": 2.5,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        num_rows = 0
        num_cols = 0
        terrain_proportions = [1.0]
        max_init_terrain_level = 4

    class navigation(HexGroundCfg.navigation):
        goal_force_blocking_line = False
        goal_force_blocking_prob = 0.0


class HexS1FollowCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_follow"


class HexS1LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s1_corridor_gate"
        scene_seed = 101
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 1.0
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "corridor_width": 1.6,
            "gate_width": 0.9,
            "gate_count": 2,
            "gate_length": 1.0,
            "gate_length_jitter": 0.2,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.8,
            "wall_thickness_m": 0.2,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.0,
            "corridor_goal_min_offset": 2.0,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        scene_params_hard = {
            "corridor_width": 1.2,
            "gate_width": 0.65,
            "gate_count": 3,
            "gate_length": 1.2,
            "gate_length_jitter": 0.25,
            "gate_spacing_min": 0.6,
            "gate_margin_y": 0.6,
            "wall_thickness_m": 0.18,
            "wall_height_m": 0.5,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.5,
            "corridor_goal_min_offset": 2.5,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
        }
        num_rows = 0
        num_cols = 0
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS1LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_large"


class HexS2Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s2_forest"
        scene_seed = 102
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.24, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_params_easy = {
            "count_min": 8,
            "count_max": 12,
            "pole_radius_min": 0.12,
            "pole_radius_max": 0.18,
            "pole_height_min": 0.30,
            "pole_height_max": 0.35,
            "block_ratio": 0.2,
            "block_size_min": 0.28,
            "block_size_max": 0.40,
            "block_height_min": 0.30,
            "block_height_max": 0.35,
            "min_dist": 0.45,
            "spawn_clear": 1.0,
            "goal_clear": 1.0,
        }
        scene_params_hard = {
            "count_min": 16,
            "count_max": 24,
            "pole_radius_min": 0.14,
            "pole_radius_max": 0.22,
            "pole_height_min": 0.35,
            "pole_height_max": 0.40,
            "block_ratio": 0.4,
            "block_size_min": 0.30,
            "block_size_max": 0.44,
            "block_height_min": 0.35,
            "block_height_max": 0.40,
            "min_dist": 0.40,
            "spawn_clear": 1.0,
            "goal_clear": 1.0,
        }
        num_rows = 0
        num_cols = 0
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS2CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s2"


class HexS2LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s2_forest"
        scene_seed = 102
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.24, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_params_easy = {
            "count_min": 24,
            "count_max": 32,
            "pole_radius_min": 0.12,
            "pole_radius_max": 0.18,
            "pole_height_min": 0.30,
            "pole_height_max": 0.35,
            "block_ratio": 0.5,
            "block_size_min": 0.28,
            "block_size_max": 0.40,
            "block_height_min": 0.30,
            "block_height_max": 0.35,
            "min_dist": 0.40,
            "spawn_clear": 1.0,
            "goal_clear": 1.0,
        }
        scene_params_hard = {
            "count_min": 40,
            "count_max": 56,
            "pole_radius_min": 0.14,
            "pole_radius_max": 0.22,
            "pole_height_min": 0.35,
            "pole_height_max": 0.40,
            "block_ratio": 0.57,
            "block_size_min": 0.30,
            "block_size_max": 0.44,
            "block_height_min": 0.35,
            "block_height_max": 0.40,
            "min_dist": 0.36,
            "spawn_clear": 1.0,
            "goal_clear": 1.0,
        }
        num_rows = 0
        num_cols = 0
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS2LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s2_large"


class HexS3Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s3_doorway_rooms"
        scene_seed = 103
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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
        terrain_v2_enable = True
        scene_type = "s3_doorway_rooms"
        scene_seed = 103
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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


class HexS4Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s4_crossing"
        scene_seed = 104
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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
        terrain_v2_enable = True
        scene_type = "s4_crossing"
        scene_seed = 104
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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


class HexS5Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s5_sparse_dense"
        scene_seed = 105
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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
        terrain_v2_enable = True
        scene_type = "s5_sparse_dense"
        scene_seed = 105
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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


class HexS6Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_type = "s6_structured_ood"
        scene_seed = 106
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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
        terrain_v2_enable = True
        scene_type = "s6_structured_ood"
        scene_seed = 106
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
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


class HexMixGateCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_types = ["s3_doorway_rooms", "s4_crossing", "s5_sparse_dense"]
        scene_seed = 120
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_static_wall_block_size = 0.8
        scene_static_wall_block_height = 0.35

        scene_probs_easy = {
            "s3_doorway_rooms": 0.2,
            "s4_crossing": 0.6,
            "s5_sparse_dense": 0.2,
        }
        scene_probs_hard = {
            "s3_doorway_rooms": 0.3,
            "s4_crossing": 0.4,
            "s5_sparse_dense": 0.3,
        }

        scene_params_easy = {
            "s3_doorway_rooms": {
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
            "s5_sparse_dense": {
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
            "s3_doorway_rooms": {
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
            "s5_sparse_dense": {
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
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_v2_enable = True
        scene_types = ["s3_doorway_rooms", "s4_crossing", "s5_sparse_dense"]
        scene_seed = 120
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
        scene_use_heightfield = True
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = [0.28, 0.36, 0.44]
        scene_static_block_heights = [0.3, 0.35, 0.4]
        scene_static_wall_block_size = 1.0
        scene_static_wall_block_height = 0.35

        scene_probs_easy = {
            "s3_doorway_rooms": 0.2,
            "s4_crossing": 0.6,
            "s5_sparse_dense": 0.2,
        }
        scene_probs_hard = {
            "s3_doorway_rooms": 0.3,
            "s4_crossing": 0.4,
            "s5_sparse_dense": 0.3,
        }

        scene_params_easy = {
            "s3_doorway_rooms": {
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
            "s5_sparse_dense": {
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
            "s3_doorway_rooms": {
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
            "s5_sparse_dense": {
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
