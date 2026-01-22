from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexS1Cfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
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
        scene_static_max = 0
        scene_static_block_max = 0
        scene_static_wall_max = 120
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 0.6
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "corridor_length": 6.5,
            "corridor_width": 1.6,
            "corridor_wall_height": 0.35,
            "corridor_center_x_jitter": 0.05,
            "corridor_gate_count": 2,
            "corridor_gate_length": 1.0,
            "corridor_gate_length_jitter": 0.2,
            "corridor_gate_width": 0.9,
            "corridor_gate_width_jitter": 0.1,
            "corridor_gate_spacing_min": 0.8,
            "corridor_gate_margin_y": 0.8,
            "corridor_gate_pass_margin": 0.05,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.0,
            "corridor_goal_min_offset": 2.0,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
            "corridor_obstacle_count_min": 0,
            "corridor_obstacle_count_max": 0,
        }
        scene_params_hard = {
            "corridor_length": 8.0,
            "corridor_width": 1.2,
            "corridor_wall_height": 0.35,
            "corridor_center_x_jitter": 0.08,
            "corridor_gate_count": 3,
            "corridor_gate_length": 1.2,
            "corridor_gate_length_jitter": 0.2,
            "corridor_gate_width": 0.65,
            "corridor_gate_width_jitter": 0.12,
            "corridor_gate_spacing_min": 0.8,
            "corridor_gate_margin_y": 0.6,
            "corridor_gate_pass_margin": 0.05,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.5,
            "corridor_goal_min_offset": 2.5,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
            "corridor_obstacle_count_min": 0,
            "corridor_obstacle_count_max": 0,
        }
        num_cols = 3
        terrain_proportions = [1.0]
        max_init_terrain_level = 4


class HexS1CfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_s1"


class HexS1FollowCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "plane"
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
        scene_static_max = 0
        scene_static_block_max = 0
        scene_static_wall_max = 120
        scene_static_block_size = 0.4
        scene_static_block_height = 0.35
        scene_static_block_sizes = []
        scene_static_block_heights = []
        scene_static_wall_block_size = 0.6
        scene_static_wall_block_height = 0.35
        scene_params_easy = {
            "corridor_length": 6.5,
            "corridor_width": 1.6,
            "corridor_wall_height": 0.35,
            "corridor_center_x_jitter": 0.05,
            "corridor_gate_count": 2,
            "corridor_gate_length": 1.0,
            "corridor_gate_length_jitter": 0.2,
            "corridor_gate_width": 0.9,
            "corridor_gate_width_jitter": 0.1,
            "corridor_gate_spacing_min": 0.8,
            "corridor_gate_margin_y": 0.8,
            "corridor_gate_pass_margin": 0.05,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.0,
            "corridor_goal_min_offset": 2.0,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
            "corridor_obstacle_count_min": 0,
            "corridor_obstacle_count_max": 0,
        }
        scene_params_hard = {
            "corridor_length": 8.0,
            "corridor_width": 1.2,
            "corridor_wall_height": 0.35,
            "corridor_center_x_jitter": 0.08,
            "corridor_gate_count": 3,
            "corridor_gate_length": 1.2,
            "corridor_gate_length_jitter": 0.2,
            "corridor_gate_width": 0.65,
            "corridor_gate_width_jitter": 0.12,
            "corridor_gate_spacing_min": 0.8,
            "corridor_gate_margin_y": 0.6,
            "corridor_gate_pass_margin": 0.05,
            "corridor_spawn_buffer": 0.6,
            "corridor_spawn_span": 2.5,
            "corridor_goal_min_offset": 2.5,
            "corridor_goal_buffer": 0.6,
            "corridor_goal_margin": 0.2,
            "corridor_obstacle_count_min": 0,
            "corridor_obstacle_count_max": 0,
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
