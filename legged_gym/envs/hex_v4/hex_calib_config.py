from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO


class HexCalibCfg(HexGroundCfg):
    class env(HexGroundCfg.env):
        env_spacing = 12.0
    class terrain(HexGroundCfg.terrain):
        mesh_type = "heightfield"
        fixed_layout_enable = False
        terrain_length = 12.0
        terrain_width = 6.0
        scene_type = "calib_axis"
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
        num_rows = 1
        num_cols = 1
        terrain_proportions = [1.0]
        max_init_terrain_level = 0


class HexCalibCfgPPO(HexGroundCfgPPO):
    class runner(HexGroundCfgPPO.runner):
        experiment_name = "hex_calib"
