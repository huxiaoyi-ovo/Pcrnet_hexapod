from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexDebugPlaneCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 8.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
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
        terrain_type = "debug_axis"
        terrain_seed = 7
        debug_allow_plane = False
        curriculum = False
        num_rows = 1
        num_cols = 1


class HexDebugHeightfieldCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_debug_heightfield"


class HexCalibCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_type = "debug_axis"
        terrain_seed = 7
        fixed_layout_enable = False
        terrain_length = 12.0
        terrain_width = 6.0
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_dynamic_max = 0
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
        num_rows = 5
        num_cols = 10
        terrain_proportions = [1.0]
        max_init_terrain_level = 0


class HexCalibCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_calib"


class HexS0FollowCfg(HexGroundCfg):
    """S0: flat ground + moving target, train stable following before S1 resume."""
    class env(HexGroundCfg.env):
        env_spacing = 10.0
        episode_length_s = 10
    class terrain(HexGroundCfg.terrain):
        # S0 is a flat, obstacle-free pretrain bed: use plane to ensure env origins are
        # separated by env_spacing (avoid overlapping origins and improve stability).
        mesh_type = "plane"
        debug_allow_plane = True
        terrain_type = "s0_follow_plane"
        terrain_seed = 1
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_resample_on_reset = False
        scene_resample_on_level_change = False
        num_rows = 5
        num_cols = 10
        terrain_proportions = [1.0]
        # Start from the easiest curriculum level for S0 pretraining.
        max_init_terrain_level = 0
    class navigation(HexGroundCfg.navigation):
        # Disable edge spawn: S0 wants centered tracking, not ring spawns.
        spawn_edge_enable = False
        heading_offset_rad = 0.0
        # S0: lock spawn yaw close to +Y forward (heading=0) to keep the moving target in view.
        spawn_yaw_jitter_deg = 2.0
        # Keep legacy goal sampling off; goal_world will be overridden by moving target.
        goal_mode = "fixed"
        fixed_goal = [0.0, 0.0]
        resample_on_reach = False
        # Moving target curriculum (difficulty controls switching frequency/complexity).
        moving_target_enable = True
        moving_target_v_max = 1.2
        # Make early training learnable: slow + gentle, then scale up with difficulty.
        moving_target_v_min = 0.05
        moving_target_v_typical = 0.5
        moving_target_turn_rate_max = 0.35
        moving_target_accel_max = 1.0
        moving_target_cmd_period_slow = 1.2
        moving_target_cmd_period_fast = 0.5
        moving_target_freeze_s = 0.3
        # Turn curriculum: max heading change ramps from small-angle to 90deg.
        moving_target_turn_deg_easy = 10.0
        moving_target_turn_deg_hard = 90.0
        # Early-turn near boundary to avoid hard reflection flips.
        moving_target_preturn_lookahead_s = 0.8
        moving_target_preturn_deg_min = 25.0
        moving_target_preturn_deg_max = 90.0
        moving_target_preturn_center_bias = 0.6
        # Speed variation (non-constant target speed).
        moving_target_speed_span_easy = 0.06
        moving_target_speed_span_hard = 0.35
        moving_target_speed_wave_amp = 0.15
        moving_target_speed_wave_rate_slow = 0.6
        moving_target_speed_wave_rate_fast = 1.8

        # High-level command bounds (must cover the target speed envelope).
        max_lin_vel_command = 1.4
        max_ang_vel_command = 1.5
        # Following distance contract (paper + code unified).
        follow_distance_desired = 1.0
        follow_distance_min = 0.7
        follow_distance_max = 1.4
        # Target-in-view contract: use camera FOV but enforce a tighter center window.
        target_fov_soft_scale = 0.35
        target_fov_hard_scale = 0.70
        # S0 pretraining should not be dominated by early "lost" resets; keep it as shaping only.
        target_lost_k = 0
        # Reward weights for view centering (penalties are negative values).
        # Keep this small: too strong encourages early-stage "spin in place" solutions.
        target_center_scale = 0.2
        # Visibility penalty can easily dominate early training when the target drifts out of the
        # (tight) soft/hard window; keep it small for S0 and rely on shaping/curriculum.
        target_visible_scale = 0.2
        reward_cfg = dict(HexGroundCfg.navigation.reward_cfg)
        reward_cfg["heading_offset_rad"] = 0.0
        # Follow-mode reward: track a desired distance instead of "reach goal".
        reward_cfg["follow_enable"] = True
        reward_cfg["follow_distance_desired"] = 1.0
        reward_cfg["follow_distance_sigma"] = 0.20
        reward_cfg["follow_distance_scale"] = 12.0
        reward_cfg["follow_band_scale"] = 4.0
        # Disable reach logic in S0 (we don't want to "reach target point").
        reward_cfg["goal_reach_threshold"] = 0.0
        reward_cfg["goal_reach_bonus"] = 0.0
        # Use view-centering instead of heading-to-goal for orientation.
        reward_cfg["heading_scale"] = 0.0
        # S0 is a clean following pretrain bed: disable navigation/obstacle terms to avoid noisy gradients.
        reward_cfg["passable_align_scale"] = 0.0
        reward_cfg["crossable_align_scale"] = 0.0
        reward_cfg["gate_smooth_penalty"] = 0.0
        reward_cfg["risk_barrier_scale"] = 0.0
        reward_cfg["time_penalty"] = 0.0
        # Disable collision penalty in S0 until collision_mask is fully validated (avoid "don't move" collapse).
        reward_cfg["collision_penalty"] = 0.0


class HexS0FollowCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s0_follow"


class HexS1Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_type = "s1_corridor_gate"
        terrain_seed = 101
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
            "gate_width": 1.0,
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
            "gate_width": 0.85,
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
        num_rows = 5
        num_cols = 10
        terrain_proportions = [1.0]
        max_init_terrain_level = 4
    class navigation(HexGroundCfg.navigation):
        heading_offset_rad = 0.0
        reward_cfg = dict(HexGroundCfg.navigation.reward_cfg)
        reward_cfg["heading_offset_rad"] = 0.0


class HexS1CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1"


class HexS1FollowCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_type = "s1_corridor_gate"
        terrain_seed = 101
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
            "gate_width": 1.0,
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
            "gate_width": 0.85,
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
        num_rows = 5
        num_cols = 10
        terrain_proportions = [1.0]
        max_init_terrain_level = 4

    class navigation(HexGroundCfg.navigation):
        goal_force_blocking_line = False
        goal_force_blocking_prob = 0.0
        heading_offset_rad = 0.0
        reward_cfg = dict(HexGroundCfg.navigation.reward_cfg)
        reward_cfg["heading_offset_rad"] = 0.0


class HexS1FollowCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_follow"


class HexS1FollowMovingCfg(HexS1FollowCfg):
    """
    S1-moving: corridor gates + moving target that must traverse the gates.

    This is used to force consistent follow-vs-avoid conflicts.
    """
    class navigation(HexS1FollowCfg.navigation):
        # Moving target scripted by gates (force conflicts).
        moving_target_enable = True
        moving_target_mode = "s1_gate_script"
        moving_target_v_max = 1.2
        moving_target_v_typical = 0.6
        moving_target_turn_rate_max = 0.0  # scripted mode does not use turn_rate
        moving_target_accel_max = 3.0
        moving_target_margin = 0.25

        # Following distance contract (paper + code unified).
        follow_distance_desired = 1.0
        follow_distance_min = 0.7
        follow_distance_max = 1.4

        # Target-in-view contract (use camera FOV but enforce a tighter center window).
        target_fov_soft_scale = 0.35
        target_fov_hard_scale = 0.70
        target_lost_k = 5
        target_center_scale = 2.0
        target_visible_scale = 1.0

        reward_cfg = dict(HexS1FollowCfg.navigation.reward_cfg)
        reward_cfg["heading_offset_rad"] = 0.0
        reward_cfg["follow_enable"] = True
        reward_cfg["follow_distance_desired"] = 1.0
        reward_cfg["follow_distance_sigma"] = 0.25
        reward_cfg["follow_distance_scale"] = 6.0
        # Disable reach logic in moving-target follow.
        reward_cfg["goal_reach_threshold"] = 0.0
        reward_cfg["goal_reach_bonus"] = 0.0
        # Use view-centering instead of heading-to-goal for orientation.
        reward_cfg["heading_scale"] = 0.0


class HexS1FollowMovingCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_follow_moving"


class HexS1LargeCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_type = "s1_corridor_gate"
        terrain_seed = 101
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
            "gate_width": 1.0,
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
            "gate_width": 0.85,
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
        num_rows = 5
        num_cols = 10
        terrain_proportions = [1.0]
        max_init_terrain_level = 4
    class navigation(HexGroundCfg.navigation):
        heading_offset_rad = 0.0
        reward_cfg = dict(HexGroundCfg.navigation.reward_cfg)
        reward_cfg["heading_offset_rad"] = 0.0


class HexS1LargeCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1_large"


class HexS2Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        terrain_type = "s2_forest"
        terrain_seed = 201
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
            "layout_modes": ["poisson", "cluster", "lane"],
            "layout_mode_probs": [0.5, 0.25, 0.25],
            "cluster_count_min": 2,
            "cluster_count_max": 4,
            "cluster_sigma": 0.6,
            "lane_cycles": 1.0,
            "lane_amplitude": 0.8,
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
            "layout_modes": ["poisson", "cluster", "lane"],
            "layout_mode_probs": [0.35, 0.35, 0.30],
            "cluster_count_min": 3,
            "cluster_count_max": 5,
            "cluster_sigma": 0.55,
            "lane_cycles": 1.2,
            "lane_amplitude": 0.9,
        }
        num_rows = 5
        num_cols = 10
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
        terrain_type = "s2_forest"
        terrain_seed = 201
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
            "layout_modes": ["poisson", "cluster", "lane"],
            "layout_mode_probs": [0.5, 0.25, 0.25],
            "cluster_count_min": 3,
            "cluster_count_max": 5,
            "cluster_sigma": 0.7,
            "lane_cycles": 1.0,
            "lane_amplitude": 0.9,
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
            "layout_modes": ["poisson", "cluster", "lane"],
            "layout_mode_probs": [0.35, 0.35, 0.30],
            "cluster_count_min": 4,
            "cluster_count_max": 6,
            "cluster_sigma": 0.6,
            "lane_cycles": 1.2,
            "lane_amplitude": 1.0,
        }
        num_rows = 5
        num_cols = 10
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
        terrain_type = "s3_doorway_rooms"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
        terrain_type = "s3_doorway_rooms"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
        terrain_type = "s4_crossing"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
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
        terrain_type = "s4_crossing"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
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
        terrain_type = "s5_sparse_dense"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
        terrain_type = "s5_sparse_dense"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
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
        terrain_type = "s6_ood_cluster"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
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
        terrain_type = "s6_ood_cluster"
        fixed_layout_enable = False
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 0
        scene_holdout = True
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
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 6
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
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
        scene_clearance = 0.27
        scene_margin = 0.3
        scene_high_dt = 0.1
        scene_dynamic_max = 2
        scene_dynamic_size = 0.4
        scene_dynamic_height = 0.5
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
