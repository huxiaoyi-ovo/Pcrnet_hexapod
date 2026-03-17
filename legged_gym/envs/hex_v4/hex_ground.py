
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO
from legged_gym.envs.hex_v4.hex_scenes_config import HexDebugPlaneCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertGround
from legged_gym.envs.hex_v4.scene_spec import SceneSpec
import torch
import numpy as np
from typing import List, Optional, Tuple
from collections import deque

from isaacgym import gymtorch,gymapi,gymutil
from legged_gym.utils import get_args,class_to_dict
from legged_gym.utils.helpers import parse_sim_params
from legged_gym.utils.terrain import (
    _get_e_s_corridor_geom as terrain_get_e_s_corridor_geom,
    _build_e_s_corridor_centerline as terrain_build_e_s_corridor_centerline,
)
from isaacgym.torch_utils import torch_rand_float,quat_rotate_inverse

import math
import time
class HexGround(LeggedRobot):
    def __init__(self,cfg:HexGroundCfg,sim_params,physics_engine,sim_device,headless):
        # 相机配置（在调用父类初始化前设置）
        self.camera_cfg = None
        self.enable_camera = False
        self._use_camera_in_headless = False
        self.camera_handles = []
        self.nav_cfg = getattr(cfg, "navigation", None)
        if hasattr(cfg, "sensor") and hasattr(cfg.sensor, "depth_camera"):
            self.camera_cfg = cfg.sensor.depth_camera
            self.enable_camera = bool(self.camera_cfg.enable)
            self._use_camera_in_headless = headless and self.enable_camera
        terrain_type = getattr(cfg.terrain, "terrain_type", None) if hasattr(cfg, "terrain") else None
        debug_allow_plane = bool(getattr(cfg.terrain, "debug_allow_plane", False))
        mesh_type = getattr(cfg.terrain, "mesh_type", None)
        terrain_type_str = str(terrain_type).strip().lower() if terrain_type is not None else ""
        if terrain_type_str == "s_avoid_basic" and mesh_type not in ("plane", "none"):
            raise RuntimeError("s_avoid_basic requires cfg.terrain.mesh_type='plane' (or 'none').")
        if debug_allow_plane:
            if mesh_type not in ("plane", "none"):
                raise RuntimeError("debug_allow_plane requires cfg.terrain.mesh_type='plane' (or 'none').")
        else:
            if mesh_type not in ("heightfield", "trimesh"):
                raise RuntimeError("hex_ground requires cfg.terrain.mesh_type in {'heightfield','trimesh'} for classic terrain.")
            if not terrain_type:
                raise RuntimeError(
                    "hex_ground 是容器任务，必须显式设置 terrain_type。"
                    "建议使用: --task s_avoid_basic 或 --task s_debug_plane。"
                    "示例: python legged_gym/scripts/train.py --task s_avoid_basic --num_envs 2048; "
                    "或 python legged_gym/scripts/train_highlevel.py "
                    "--mode teacher --skill follow --task s_avoid_basic --low_level_ckpt agents/fast_2000.pt"
                )
        super().__init__(cfg,sim_params,physics_engine,sim_device,headless)
        self.cfg:HexGroundCfg = cfg
        if terrain_type in ("e_l_conflict", "e_l_confilct", "e_l_conflict_turn") and self.nav_cfg is not None:
            moving_mode = str(getattr(self.nav_cfg, "moving_target_mode", "")).strip().lower()
            if moving_mode in ("e_l_confilct_script",):
                moving_mode = "e_l_conflict_script"
            if moving_mode != "e_l_conflict_script":
                raise RuntimeError(
                    "e_L_conflict requires navigation.moving_target_mode='e_l_conflict_script', "
                    f"got '{moving_mode or 'unset'}'"
                )
            geom = self._get_e_l_conflict_turn_path()
            print(
                "[Scene] e_L_conflict active: "
                f"straight x={geom['x_line']:.3f}, y:{geom['start_y']:.3f}->{geom['turn_entry_y']:.3f}, "
                f"arc_r={geom['turn_r']:.3f}, then y={geom['y_line']:.3f}, x:{geom['obs_x']:.3f}->{geom['end_x']:.3f}, "
                f"speed={geom['speed']:.3f}, straight_len={geom['straight_len']:.3f}, spawn_gap={geom['spawn_gap']:.3f}"
            )
            print(
                "[Scene] e_L_conflict obstacle spec: "
                f"center_local=({geom['obs_x']:.3f},{geom['obs_y']:.3f}), radius={geom['obs_r']:.3f}, "
                f"clearance={max(0.0, geom['turn_r'] - geom['obs_r']):.3f}"
            )
        if terrain_type == "e_s_corridor" and self.nav_cfg is not None:
            moving_mode = str(getattr(self.nav_cfg, "moving_target_mode", "")).strip().lower()
            if moving_mode != "e_s_corridor_script":
                raise RuntimeError(
                    "e_S_corridor requires navigation.moving_target_mode='e_s_corridor_script', "
                    f"got '{moving_mode or 'unset'}'"
                )
            geom = self._get_e_s_corridor_geometry()
            cache = self._build_e_s_corridor_cache()
            print(
                "[Scene] e_S_corridor active: "
                f"width={geom['corridor_width']:.3f}, amp={geom['amplitude']:.3f}, "
                f"y:{geom['start_y']:.3f}->{geom['end_y']:.3f}, "
                f"straight_in={geom['straight_in']:.3f}, bend={geom['bend_length']:.3f}, "
                f"straight_out={geom['straight_out']:.3f}, speed={geom['speed']:.3f}, "
                f"spawn_gap={geom['spawn_gap']:.3f}"
            )
            print(
                "[Scene] e_S_corridor wall spec: "
                f"mesh=analytic_trimesh, wall_thickness={geom['wall_thickness']:.3f}, "
                f"wall_height={geom['wall_height']:.3f}, path_segments={cache['segments_per_side']}"
            )
        self.debug_viz = False
        self.foot_traj_viz=False
        terrain_obj = getattr(self, "terrain", None)
        self.scene_generator = None
        self.scene_manager = None
        self.scene_specs = getattr(terrain_obj, "tile_meta", None)
        self.scene_meta = [None] * self.num_envs
        self.scene_episode_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.scene_dyn_time = torch.zeros(self.num_envs, device=self.device)
        self.scene_spec_cache = [None] * self.num_envs
        self.scene_level_cache = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.scene_difficulty_override = None
        self.scene_margin = float(getattr(self.cfg.terrain, "scene_margin", 0.3))
        self.scene_clearance = float(getattr(self.cfg.terrain, "scene_clearance", 0.27))
        self._init_scene_runtime()
        #额外初始化电机类，可以计理想力矩或模拟的仿真力矩
        self.actuator=Actuator(self.cfg,self.device)
        #额外初始化专家类，可以在step时，提供专家动作参考
        # if self.cfg.env.gen_expert_actions:
        self.expert=ExpertGround(self.cfg,self.device,self.cfg.env.num_envs)

        #额外初始化相机类
        cam_prop=gymapi.CameraProperties()
        # print("sim_params.use_gpu_pipline=",sim_params.use_gpu_pipline)
        if self.camera_cfg is not None and self.enable_camera:
            self._init_camera_buffers()

    def create_sim(self):
        """重写create_sim以在headless模式下支持相机"""
        if self._use_camera_in_headless:
            self.graphics_device_id = self.sim_device_id
        super().create_sim()
        if self.camera_cfg is not None:
            self.cameras_created = False
    #当返回的观测是分离时，重写这个函数，否则进行注释
    # def reset(self):
    #     self.reset_idx(torch.arange(self.num_envs, device=self.device))
    #     obs,obs_vfg,obs_terrain,_,_,_ = self.step(torch.zeros_like(self.actions))
    #     return obs,obs_vfg,obs_terrain
    def reset_separate(self):
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        obs_dict, _, _, _ = self.step_separate(torch.zeros_like(self.actions))
        return obs_dict

    def _draw_debug_vis(self):
        # In viewer debug mode, prefer visualizing moving-target/robot trajectories over
        # the default height-sampling points (which would also clear lines every frame).
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            debug_case = str(getattr(self.cfg.terrain, "avoid_map_debug_case", "")).strip().lower()
            if debug_case:
                self._debug_draw_s_avoid_map_case()
                return
        if self.viewer and self.enable_viewer_sync and self.debug_viz and self._moving_target_enabled():
            self._debug_draw_target_and_robot_trajectories()
            return
        if self.viewer and self.enable_viewer_sync and self.debug_viz and self.s_avoid_enabled:
            self._debug_draw_goal_and_robot_trajectories()
            return

        terrain_obj = getattr(self, "terrain", None)
        if terrain_obj is None or not hasattr(terrain_obj, "cfg"):
            return
        super()._draw_debug_vis()

    def _debug_draw_target_and_robot_trajectories(self):
        """Draw moving target (point + trajectory) and robot trajectory for all envs.

        Notes:
        - This is for interactive debugging only; it is guarded by viewer + debug_viz.
        - We draw incrementally at scene dt (~10Hz) and periodically clear lines to avoid
          unbounded accumulation.
        """
        if self.viewer is None or not hasattr(self, "envs"):
            return
        if not hasattr(self, "target_world"):
            return

        if not hasattr(self, "_viz_traj_time_accum"):
            self._viz_traj_time_accum = 0.0
            self._viz_traj_tick = 0
            # Default off: keep line clearing aligned with reset_idx.
            # Set cfg.terrain.debug_viz_clear_every > 0 only when periodic clear is desired.
            self._viz_traj_clear_every = int(getattr(self.cfg.terrain, "debug_viz_clear_every", 0))
            self._viz_prev_robot = np.zeros((self.num_envs, 3), dtype=np.float32)
            self._viz_prev_target = np.zeros((self.num_envs, 3), dtype=np.float32)
            self._viz_prev_valid = np.zeros((self.num_envs,), dtype=np.bool_)
            self._viz_color_robot = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            self._viz_color_target = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            self._viz_target_sphere = gymutil.WireframeSphereGeometry(
                0.04, 6, 6, color=(0.0, 1.0, 0.0)
            )

        self._viz_traj_time_accum += float(self.dt)
        dt_high = float(getattr(self.cfg.terrain, "scene_high_dt", 0.1))
        if self._viz_traj_time_accum + 1e-9 < dt_high:
            return
        self._viz_traj_time_accum = 0.0
        self._viz_traj_tick += 1

        if self._viz_traj_clear_every > 0 and self._viz_traj_tick % self._viz_traj_clear_every == 0:
            self.gym.clear_lines(self.viewer)
            self._viz_prev_valid[:] = False

        # Pull positions to CPU for rendering.
        robot_pos = self.root_states[:, 0:3].detach().cpu().numpy().astype(np.float32, copy=False)
        target_xy = self.target_world.detach().cpu().numpy().astype(np.float32, copy=False)

        target_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        target_pos[:, 0:2] = target_xy[:, 0:2]
        # Lift target a bit above ground for visibility.
        target_pos[:, 2] = robot_pos[:, 2] + 0.06

        # Draw per-env incremental segments.
        for i in range(self.num_envs):
            if not self._viz_prev_valid[i]:
                self._viz_prev_robot[i] = robot_pos[i]
                self._viz_prev_target[i] = target_pos[i]
                self._viz_prev_valid[i] = True
            else:
                v_robot = np.stack([self._viz_prev_robot[i], robot_pos[i]], axis=0)
                v_target = np.stack([self._viz_prev_target[i], target_pos[i]], axis=0)
                self.gym.add_lines(self.viewer, self.envs[i], 1, v_robot, self._viz_color_robot)
                self.gym.add_lines(self.viewer, self.envs[i], 1, v_target, self._viz_color_target)
                self._viz_prev_robot[i] = robot_pos[i]
                self._viz_prev_target[i] = target_pos[i]

            # Draw current target position marker.
            pose = gymapi.Transform(
                gymapi.Vec3(float(target_pos[i, 0]), float(target_pos[i, 1]), float(target_pos[i, 2])),
                r=None,
            )
            gymutil.draw_lines(self._viz_target_sphere, self.gym, self.viewer, self.envs[i], pose)

    def _debug_draw_goal_and_robot_trajectories(self):
        """Draw avoid-task local-goal point/trajectory and robot trajectory."""
        if self.viewer is None or not hasattr(self, "envs"):
            return
        if not hasattr(self, "goal_world"):
            return

        if not hasattr(self, "_viz_goal_traj_time_accum"):
            self._viz_goal_traj_time_accum = 0.0
            self._viz_goal_traj_tick = 0
            self._viz_goal_traj_clear_every = int(getattr(self.cfg.terrain, "debug_viz_clear_every", 0))
            self._viz_goal_prev_robot = np.zeros((self.num_envs, 3), dtype=np.float32)
            self._viz_goal_prev_point = np.zeros((self.num_envs, 3), dtype=np.float32)
            self._viz_goal_prev_valid = np.zeros((self.num_envs,), dtype=np.bool_)
            self._viz_goal_color_robot = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            self._viz_goal_color_point = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
            self._viz_goal_sphere = gymutil.WireframeSphereGeometry(
                0.04, 6, 6, color=(0.0, 1.0, 0.0)
            )

        self._viz_goal_traj_time_accum += float(self.dt)
        dt_high = float(getattr(self.cfg.terrain, "scene_high_dt", 0.1))
        if self._viz_goal_traj_time_accum + 1e-9 < dt_high:
            return
        self._viz_goal_traj_time_accum = 0.0
        self._viz_goal_traj_tick += 1

        if self._viz_goal_traj_clear_every > 0 and self._viz_goal_traj_tick % self._viz_goal_traj_clear_every == 0:
            self.gym.clear_lines(self.viewer)
            self._viz_goal_prev_valid[:] = False

        robot_pos = self.root_states[:, 0:3].detach().cpu().numpy().astype(np.float32, copy=False)
        goal_xy = self.goal_world.detach().cpu().numpy().astype(np.float32, copy=False)

        goal_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        goal_pos[:, 0:2] = goal_xy[:, 0:2]
        goal_pos[:, 2] = robot_pos[:, 2] + 0.06

        for i in range(self.num_envs):
            if not self._viz_goal_prev_valid[i]:
                self._viz_goal_prev_robot[i] = robot_pos[i]
                self._viz_goal_prev_point[i] = goal_pos[i]
                self._viz_goal_prev_valid[i] = True
            else:
                v_robot = np.stack([self._viz_goal_prev_robot[i], robot_pos[i]], axis=0)
                v_goal = np.stack([self._viz_goal_prev_point[i], goal_pos[i]], axis=0)
                self.gym.add_lines(self.viewer, self.envs[i], 1, v_robot, self._viz_goal_color_robot)
                self.gym.add_lines(self.viewer, self.envs[i], 1, v_goal, self._viz_goal_color_point)
                self._viz_goal_prev_robot[i] = robot_pos[i]
                self._viz_goal_prev_point[i] = goal_pos[i]

            pose = gymapi.Transform(
                gymapi.Vec3(float(goal_pos[i, 0]), float(goal_pos[i, 1]), float(goal_pos[i, 2])),
                r=None,
            )
            gymutil.draw_lines(self._viz_goal_sphere, self.gym, self.viewer, self.envs[i], pose)

    def _debug_draw_thick_line(self, env_id: int, p0: np.ndarray, p1: np.ndarray, color, width: float = 0.014):
        direction = p1[:2] - p0[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            offsets = [np.zeros(3, dtype=np.float32)]
        else:
            perp = np.array([-direction[1], direction[0]], dtype=np.float32) / norm
            offsets = [
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([perp[0] * width, perp[1] * width, 0.0], dtype=np.float32),
                np.array([-perp[0] * width, -perp[1] * width, 0.0], dtype=np.float32),
            ]
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
        for off in offsets:
            verts = np.stack([p0 + off, p1 + off], axis=0).astype(np.float32)
            self.gym.add_lines(self.viewer, self.envs[env_id], 1, verts, color_arr)

    def _debug_draw_s_avoid_map_case(self):
        if self.viewer is None or not hasattr(self, "envs") or self.num_envs <= 0:
            return
        self.gym.clear_lines(self.viewer)
        if not hasattr(self, "_viz_debug_obs_sphere"):
            self._viz_debug_obs_sphere = gymutil.WireframeSphereGeometry(
                0.07, 8, 8, color=(1.0, 0.0, 1.0)
            )
            self._viz_debug_fwd_sphere = gymutil.WireframeSphereGeometry(
                0.05, 8, 8, color=(0.0, 1.0, 1.0)
            )
        env_id = 0
        robot_pos = self.root_states[env_id, :3].detach().cpu().numpy().astype(np.float32, copy=False)
        quat = self.root_states[env_id, 3:7].detach().cpu()
        x_q, y_q, z_q, w_q = [float(v.item()) for v in quat]
        yaw = math.atan2(
            2.0 * (w_q * z_q + x_q * y_q),
            1.0 - 2.0 * (y_q * y_q + z_q * z_q),
        )
        fwd = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        right = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        z_up = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        p0 = robot_pos + z_up
        p_fwd = p0 + 0.45 * fwd
        p_right = p0 + 0.30 * right
        self._debug_draw_thick_line(env_id, p0, p_fwd, color=(0.0, 1.0, 1.0), width=0.018)
        self._debug_draw_thick_line(env_id, p0, p_right, color=(1.0, 1.0, 0.0), width=0.014)
        gymutil.draw_lines(
            self._viz_debug_fwd_sphere,
            self.gym,
            self.viewer,
            self.envs[env_id],
            gymapi.Transform(gymapi.Vec3(float(p_fwd[0]), float(p_fwd[1]), float(p_fwd[2])), r=None),
        )

        active = self.s_avoid_active[env_id]
        if bool(active.any().item()):
            slot = int(torch.nonzero(active, as_tuple=False)[0, 0].item())
            obs_pos = self.s_avoid_pos_world[env_id, slot, :3].detach().cpu().numpy().astype(np.float32, copy=False)
            obs_marker = obs_pos.copy()
            obs_marker[2] = p0[2]
            self._debug_draw_thick_line(env_id, p0, obs_marker, color=(1.0, 0.0, 1.0), width=0.018)
            gymutil.draw_lines(
                self._viz_debug_obs_sphere,
                self.gym,
                self.viewer,
                self.envs[env_id],
                gymapi.Transform(gymapi.Vec3(float(obs_pos[0]), float(obs_pos[1]), float(obs_marker[2])), r=None),
            )
    
    def _pre_create_envs(self):
        self.robot_actor_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.static_scene_actor_handles = None
        self.static_scene_actor_indices = None
        self.e_conflict_static_asset = None
        self.e_conflict_static_shape = None
        self.e_conflict_capsule_half_height = None
        self.e_conflict_obstacle_pose_local = None
        self.e_s_corridor_wall_asset = None
        self.e_s_corridor_wall_pose_local = None
        self.e_s_corridor_wall_handles = None
        self.e_s_corridor_wall_indices = None
        self.e_s_corridor_path_s_tensor = None
        self.e_s_corridor_path_pos_tensor = None
        self.e_s_corridor_path_tan_tensor = None
        self.e_s_corridor_path_length = None
        self.dynamic_actor_handles = None
        self.dynamic_actor_indices = None
        self.dynamic_asset = None
        self.s_avoid_enabled = False
        self.s_avoid_actor_handles = None
        self.s_avoid_actor_indices = None
        self.s_avoid_capsule_asset = None
        self.s_avoid_box_asset = None
        self.s_avoid_wall_asset = None
        self.s_avoid_capsule_slot_count = 0
        self.s_avoid_box_slot_count = 0
        self.s_avoid_wall_slot_count = 0
        self.s_avoid_total_slots = 0
        self.s_avoid_capsule_quat = None
        self.s_avoid_identity_quat = None
        self.s_avoid_direct_single_obstacle = False

        terrain_type = str(getattr(self.cfg.terrain, "terrain_type", "")).lower()
        if terrain_type in ("e_l_conflict", "e_l_confilct", "e_l_conflict_turn"):
            obs_x, obs_y, obs_r, obs_h = self._get_e_l_conflict_obstacle_local()

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.collapse_fixed_joints = True

            # Prefer capsule (closest built-in primitive to a cylinder), fallback to box.
            self.e_conflict_static_asset = None
            if hasattr(self.gym, "create_capsule"):
                try:
                    half_height = max(1e-3, 0.5 * obs_h - obs_r)
                    self.e_conflict_capsule_half_height = float(half_height)
                    self.e_conflict_static_asset = self.gym.create_capsule(
                        self.sim, obs_r, half_height, asset_options
                    )
                    self.e_conflict_static_shape = "capsule"
                except Exception:
                    self.e_conflict_static_asset = None
                    self.e_conflict_capsule_half_height = None
            if self.e_conflict_static_asset is None:
                dia = max(1e-3, 2.0 * obs_r)
                self.e_conflict_static_asset = self.gym.create_box(
                    self.sim, dia, dia, max(1e-3, obs_h), asset_options
                )
                self.e_conflict_static_shape = "box"
                self.e_conflict_capsule_half_height = None

            self.e_conflict_obstacle_pose_local = (float(obs_x), float(obs_y), float(0.5 * obs_h))
            self.static_scene_actor_handles = [None for _ in range(self.num_envs)]
            self.static_scene_actor_indices = np.zeros((self.num_envs,), dtype=np.int32)
            print(
                "[Scene] e_L_conflict obstacle actor: "
                f"shape={self.e_conflict_static_shape}, center_local=({obs_x:.3f},{obs_y:.3f}), "
                f"radius={obs_r:.3f}, height={obs_h:.3f}"
            )

        if terrain_type == "e_s_corridor" and str(getattr(self.cfg.terrain, "mesh_type", "")).lower() in ("plane", "none"):
            geom = self._get_e_s_corridor_geometry()
            cache = self._build_e_s_corridor_cache()

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.collapse_fixed_joints = True

            self.e_s_corridor_wall_asset = self.gym.create_box(
                self.sim,
                float(geom["wall_thickness"]),
                float(geom["segment_length"]),
                float(geom["wall_height"]),
                asset_options,
            )
            self.e_s_corridor_wall_pose_local = cache["wall_poses_local"]
            wall_count = len(self.e_s_corridor_wall_pose_local)
            self.e_s_corridor_wall_handles = [
                [None for _ in range(wall_count)] for _ in range(self.num_envs)
            ]
            self.e_s_corridor_wall_indices = np.zeros((self.num_envs, wall_count), dtype=np.int32)
            print(
                "[Scene] e_S_corridor wall actor pool: "
                f"count={wall_count}, segment_len={geom['segment_length']:.3f}, "
                f"width={geom['corridor_width']:.3f}, amp={geom['amplitude']:.3f}"
            )

        if terrain_type == "s_avoid_basic":
            self.s_avoid_direct_single_obstacle = bool(
                getattr(self.cfg.terrain, "avoid_direct_single_obstacle", False)
            )
            fixed_asset_options = gymapi.AssetOptions()
            fixed_asset_options.fix_base_link = True
            fixed_asset_options.disable_gravity = True
            fixed_asset_options.collapse_fixed_joints = True

            pooled_asset_options = gymapi.AssetOptions()
            pooled_asset_options.fix_base_link = False
            pooled_asset_options.disable_gravity = True
            pooled_asset_options.collapse_fixed_joints = True
            pooled_asset_options.linear_damping = float(
                getattr(self.cfg.terrain, "avoid_pooled_linear_damping", 50.0)
            )
            pooled_asset_options.angular_damping = float(
                getattr(self.cfg.terrain, "avoid_pooled_angular_damping", 50.0)
            )

            cap_r = float(getattr(self.cfg.terrain, "avoid_capsule_radius", 0.15))
            cap_h = float(getattr(self.cfg.terrain, "avoid_capsule_height", 0.5))
            cap_half_h = max(1e-3, 0.5 * cap_h - cap_r)
            box_x = float(getattr(self.cfg.terrain, "avoid_box_size_x", 0.4))
            box_y = float(getattr(self.cfg.terrain, "avoid_box_size_y", 0.4))
            box_z = float(getattr(self.cfg.terrain, "avoid_box_size_z", 0.5))
            wall_t = float(getattr(self.cfg.terrain, "avoid_wall_thickness", 0.12))
            wall_l = float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))
            wall_h = float(getattr(self.cfg.terrain, "avoid_wall_height", 0.5))

            if self.s_avoid_direct_single_obstacle:
                self.s_avoid_capsule_asset = self.gym.create_capsule(self.sim, cap_r, cap_half_h, fixed_asset_options)
                self.s_avoid_box_asset = self.gym.create_box(self.sim, box_x, box_y, box_z, fixed_asset_options)
                self.s_avoid_wall_asset = self.gym.create_box(self.sim, wall_t, wall_l, wall_h, fixed_asset_options)
                self.s_avoid_capsule_slot_count = 1
                self.s_avoid_box_slot_count = 0
                self.s_avoid_wall_slot_count = 0
            else:
                self.s_avoid_capsule_asset = self.gym.create_capsule(self.sim, cap_r, cap_half_h, pooled_asset_options)
                self.s_avoid_box_asset = self.gym.create_box(self.sim, box_x, box_y, box_z, pooled_asset_options)
                self.s_avoid_wall_asset = self.gym.create_box(self.sim, wall_t, wall_l, wall_h, pooled_asset_options)
                self.s_avoid_capsule_slot_count = int(getattr(self.cfg.terrain, "avoid_capsule_slots", 6))
                self.s_avoid_box_slot_count = int(getattr(self.cfg.terrain, "avoid_box_slots", 2))
                self.s_avoid_wall_slot_count = int(getattr(self.cfg.terrain, "avoid_wall_slots", 2))
            self.s_avoid_total_slots = (
                self.s_avoid_capsule_slot_count
                + self.s_avoid_box_slot_count
                + self.s_avoid_wall_slot_count
            )

            self.s_avoid_actor_handles = [
                [None for _ in range(self.s_avoid_total_slots)] for _ in range(self.num_envs)
            ]
            self.s_avoid_actor_indices = np.zeros((self.num_envs, self.s_avoid_total_slots), dtype=np.int32)
            self.s_avoid_identity_quat = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            self.s_avoid_capsule_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
            self.s_avoid_enabled = True
            print(
                "[Scene] s_avoid_basic obstacle pool: "
                f"capsule_slots={self.s_avoid_capsule_slot_count}, "
                f"box_slots={self.s_avoid_box_slot_count}, "
                f"wall_slots={self.s_avoid_wall_slot_count}, "
                f"capsule_d={2.0 * cap_r:.2f}, box=({box_x:.2f},{box_y:.2f},{box_z:.2f}), "
                f"wall=({wall_t:.2f},{wall_l:.2f},{wall_h:.2f})"
            )
            if self.s_avoid_direct_single_obstacle:
                print("[Scene] s_avoid_basic direct-single-obstacle debug enabled")
            else:
                pooled_mass = float(getattr(self.cfg.terrain, "avoid_pooled_actor_mass", 1000.0))
                print(
                    "[Scene] s_avoid_basic pooled obstacle body: "
                    f"fix_base_link=False, mass={pooled_mass:.1f}, "
                    f"lin_damp={pooled_asset_options.linear_damping:.1f}, "
                    f"ang_damp={pooled_asset_options.angular_damping:.1f}"
                )

        terrain_obj = getattr(self, "terrain", None)
        self.scene_generator = getattr(terrain_obj, "scene_generator", None)
        if self.scene_generator is not None and self.scene_generator.has_dynamic:
            max_dyn = int(getattr(self.cfg.terrain, "scene_dynamic_max", 0))
            if max_dyn > 0:
                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                asset_options.disable_gravity = True
                asset_options.collapse_fixed_joints = True
                size_xy = float(getattr(self.cfg.terrain, "scene_dynamic_size", 0.35))
                height = float(getattr(self.cfg.terrain, "scene_dynamic_height", 0.5))
                self.dynamic_asset = self.gym.create_box(self.sim, size_xy, size_xy, height, asset_options)
                self.dynamic_actor_handles = [[None for _ in range(max_dyn)] for _ in range(self.num_envs)]
                self.dynamic_actor_indices = np.zeros((self.num_envs, max_dyn), dtype=np.int32)

    def _get_s_avoid_debug_case(self) -> str:
        debug_case = str(getattr(self.cfg.terrain, "avoid_map_debug_case", "")).strip().lower()
        if self.s_avoid_direct_single_obstacle and debug_case == "":
            return "front"
        return debug_case

    def _get_s_avoid_debug_local_pose(self):
        cam_cfg = getattr(getattr(self.cfg, "sensor", None), "depth_camera", None)
        cam_y = 0.0
        if cam_cfg is not None and hasattr(cam_cfg, "position") and len(cam_cfg.position) >= 2:
            cam_y = float(cam_cfg.position[1])
        local_x = 0.0
        local_y = 1.35
        debug_case = self._get_s_avoid_debug_case()
        if debug_case == "left":
            local_x = -0.60
        elif debug_case == "right":
            local_x = 0.60
        elif debug_case == "side_left":
            local_x = -1.35
            local_y = cam_y
        elif debug_case == "side_right":
            local_x = 1.35
            local_y = cam_y
        stage12_spawn_y = -1.6
        return local_x, stage12_spawn_y + local_y

    def _post_create_envs(self):
        if not hasattr(self, "robot_actor_indices"):
            return
        indices = np.asarray(self.robot_actor_indices, dtype=np.int32)
        if indices.shape[0] != self.num_envs:
            raise RuntimeError(
                f"robot_actor_indices mismatch: got {indices.shape[0]}, expected {self.num_envs}"
            )
        if np.any(indices < 0):
            raise RuntimeError(f"robot_actor_indices invalid: {indices}")
        if np.any(np.diff(indices) <= 0):
            raise RuntimeError(f"robot_actor_indices not strictly increasing: {indices}")

    def _scene_group_id(self, env_id: int) -> int:
        return int(env_id) + 1

    def _scene_collision_filter(self) -> int:
        scene_filter = int(getattr(self.cfg.terrain, "scene_collision_filter", 0))
        if scene_filter < 0 or scene_filter >= (1 << 31):
            scene_filter = 0
        return scene_filter

    def _apply_actor_collision_filter(self, env_handle, actor_handle, target_filter: int, env_id: int, debug_tag: str = ""):
        try:
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            before_filters = [int(getattr(prop, "filter", 0)) for prop in shape_props]
            for prop in shape_props:
                prop.filter = target_filter
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            if getattr(self, "debug_viz", False) and env_id == 0 and debug_tag.startswith("s_avoid_obs_"):
                debug_count = int(getattr(self, "_s_avoid_filter_debug_count", 0))
                if debug_count < 40:
                    after_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
                    after_filters = [int(getattr(prop, "filter", 0)) for prop in after_props]
                    actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
                    print(
                        f"[Debug][s_avoid_filter] tag={debug_tag} actor_index={actor_index} "
                        f"target={int(target_filter)} before={before_filters} after={after_filters}"
                    )
                    self._s_avoid_filter_debug_count = debug_count + 1
            if getattr(self, "debug_viz", False) and env_id == 0 and debug_tag:
                flag = f"_{debug_tag}_filter_logged"
                if not getattr(self, flag, False):
                    print(f"[Debug] {debug_tag} shape filter={target_filter}")
                    setattr(self, flag, True)
        except Exception:
            if getattr(self, "debug_viz", False):
                tag = debug_tag or "unnamed_actor"
                flag = f"_{tag}_filter_warned"
                if not getattr(self, flag, False):
                    print(f"[Debug] {tag} shape filter update failed")
                    setattr(self, flag, True)

    def _configure_s_avoid_pooled_body(self, env_handle, actor_handle, env_id: int):
        if not self.s_avoid_enabled or self.s_avoid_direct_single_obstacle:
            return
        target_mass = float(getattr(self.cfg.terrain, "avoid_pooled_actor_mass", 1000.0))
        try:
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            changed = False
            for prop in body_props:
                if float(getattr(prop, "mass", 0.0)) < target_mass:
                    prop.mass = target_mass
                    changed = True
            if changed:
                self.gym.set_actor_rigid_body_properties(
                    env_handle, actor_handle, body_props, recomputeInertia=True
                )
            if getattr(self, "debug_viz", False) and env_id == 0:
                flag = f"_s_avoid_body_mass_logged_{id(actor_handle)}"
                if not getattr(self, flag, False):
                    masses = [float(getattr(prop, "mass", 0.0)) for prop in body_props]
                    print(f"[Debug] s_avoid pooled actor masses={masses}")
                    setattr(self, flag, True)
        except Exception:
            if getattr(self, "debug_viz", False) and env_id == 0:
                if not getattr(self, "_s_avoid_body_props_warned", False):
                    print("[Debug] s_avoid pooled body property update failed")
                    self._s_avoid_body_props_warned = True

    def _update_s_avoid_debug_colors(self, env_ids: torch.Tensor):
        if (
            not self.s_avoid_enabled
            or env_ids.numel() == 0
            or self.viewer is None
            or not self.enable_viewer_sync
            or not self.debug_viz
        ):
            return
        cap_slots = int(self.s_avoid_capsule_slot_count)
        box_end = cap_slots + int(self.s_avoid_box_slot_count)
        color_active_capsule = gymapi.Vec3(0.15, 0.90, 0.25)
        color_active_box = gymapi.Vec3(0.95, 0.78, 0.18)
        color_active_wall = gymapi.Vec3(0.92, 0.20, 0.20)
        color_inactive = gymapi.Vec3(0.35, 0.35, 0.35)
        for env_id in env_ids.tolist():
            handles = self.s_avoid_actor_handles[env_id]
            active = self.s_avoid_active[env_id]
            for slot, actor_handle in enumerate(handles):
                if actor_handle is None:
                    continue
                if bool(active[slot].item()):
                    if slot < cap_slots:
                        color = color_active_capsule
                    elif slot < box_end:
                        color = color_active_box
                    else:
                        color = color_active_wall
                else:
                    color = color_inactive
                try:
                    self.gym.set_rigid_body_color(
                        self.envs[env_id],
                        actor_handle,
                        0,
                        gymapi.MESH_VISUAL,
                        color,
                    )
                except Exception:
                    if getattr(self, "debug_viz", False) and env_id == 0:
                        if not getattr(self, "_s_avoid_color_warned", False):
                            print("[Debug] s_avoid obstacle color update failed")
                            self._s_avoid_color_warned = True

    def _get_s_avoid_park_local_pose(self, slot: int) -> Tuple[float, float, float]:
        half_w = 0.5 * float(getattr(self.cfg.terrain, "terrain_width", 12.0))
        half_l = 0.5 * float(getattr(self.cfg.terrain, "terrain_length", 12.0))
        lane_x = half_w + 8.0 + 1.8 * float(slot)
        lane_y = -half_l - 8.0 - 1.2 * float(slot)
        lane_z = 0.25
        return lane_x, lane_y, lane_z

    def _on_create_robot(self, env_id, env_handle, actor_handle):
        if hasattr(self, "robot_actor_indices"):
            self.robot_actor_indices[env_id] = self.gym.get_actor_index(
                env_handle, actor_handle, gymapi.DOMAIN_SIM
            )
        # 机器人与障碍碰撞过滤（避免 actor 场景穿墙）
        scene_filter = self._scene_collision_filter()
        group_id = self._scene_group_id(env_id)
        if getattr(self, "debug_viz", False) and env_id == 0 and not getattr(self, "_robot_group_logged", False):
            print(f"[Debug] robot collision_group={group_id}, scene_filter={scene_filter}")
            self._robot_group_logged = True
        self._apply_actor_collision_filter(env_handle, actor_handle, scene_filter, env_id, debug_tag="robot")

    def _create_env_actors(self, env_id, env_handle):
        group_id = self._scene_group_id(env_id)
        scene_filter = self._scene_collision_filter()
        create_filter = scene_filter

        if self.e_conflict_static_asset is not None and self.e_conflict_obstacle_pose_local is not None:
            ox, oy, oz = self.e_conflict_obstacle_pose_local
            env_origin = self.env_origins[env_id]
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                float(env_origin[0].item() + ox),
                float(env_origin[1].item() + oy),
                float(env_origin[2].item() + oz),
            )
            if self.e_conflict_static_shape == "capsule":
                # Isaac Gym capsule major axis is not guaranteed to align with world +Z.
                # Rotate to vertical so obstacle height matches the configured value.
                pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
            static_handle = self.gym.create_actor(
                env_handle,
                self.e_conflict_static_asset,
                pose,
                "static_obs_0",
                group_id,
                create_filter,
                0,
            )
            self._apply_actor_collision_filter(env_handle, static_handle, create_filter, env_id, debug_tag="static")
            if self.static_scene_actor_handles is not None:
                self.static_scene_actor_handles[env_id] = static_handle
            if self.static_scene_actor_indices is not None:
                static_index = self.gym.get_actor_index(env_handle, static_handle, gymapi.DOMAIN_SIM)
                self.static_scene_actor_indices[env_id] = static_index

        if self.e_s_corridor_wall_asset is not None and self.e_s_corridor_wall_pose_local is not None:
            env_origin = self.env_origins[env_id]
            for slot, (wx, wy, wz, yaw) in enumerate(self.e_s_corridor_wall_pose_local):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(
                    float(env_origin[0].item() + wx),
                    float(env_origin[1].item() + wy),
                    float(env_origin[2].item() + wz),
                )
                pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), float(yaw))
                wall_handle = self.gym.create_actor(
                    env_handle,
                    self.e_s_corridor_wall_asset,
                    pose,
                    f"e_s_wall_{slot}",
                    group_id,
                    create_filter,
                    0,
                )
                self._apply_actor_collision_filter(env_handle, wall_handle, create_filter, env_id, debug_tag="e_s_wall")
                if self.e_s_corridor_wall_handles is not None:
                    self.e_s_corridor_wall_handles[env_id][slot] = wall_handle
                if self.e_s_corridor_wall_indices is not None:
                    wall_index = self.gym.get_actor_index(env_handle, wall_handle, gymapi.DOMAIN_SIM)
                    self.e_s_corridor_wall_indices[env_id, slot] = wall_index

        if self.s_avoid_enabled and self.s_avoid_actor_handles is not None:
            for slot in range(self.s_avoid_total_slots):
                if slot < self.s_avoid_capsule_slot_count:
                    asset = self.s_avoid_capsule_asset
                    pose = gymapi.Transform()
                    pose.r = self.s_avoid_capsule_quat
                elif slot < self.s_avoid_capsule_slot_count + self.s_avoid_box_slot_count:
                    asset = self.s_avoid_box_asset
                    pose = gymapi.Transform()
                    pose.r = self.s_avoid_identity_quat
                else:
                    asset = self.s_avoid_wall_asset
                    pose = gymapi.Transform()
                    pose.r = self.s_avoid_identity_quat
                if self.s_avoid_direct_single_obstacle:
                    env_origin = self.env_origins[env_id]
                    local_x, local_y = self._get_s_avoid_debug_local_pose()
                    cap_h = float(getattr(self.cfg.terrain, "avoid_capsule_height", 0.5))
                    pose.p = gymapi.Vec3(
                        float(env_origin[0].item() + local_x),
                        float(env_origin[1].item() + local_y),
                        float(env_origin[2].item() + 0.5 * cap_h),
                    )
                else:
                    env_origin = self.env_origins[env_id]
                    park_x, park_y, park_z = self._get_s_avoid_park_local_pose(slot)
                    pose.p = gymapi.Vec3(
                        float(env_origin[0].item() + park_x),
                        float(env_origin[1].item() + park_y),
                        float(env_origin[2].item() + park_z),
                    )
                actor_handle = self.gym.create_actor(
                    env_handle,
                    asset,
                    pose,
                    f"s_avoid_obs_{slot}",
                    group_id,
                    scene_filter,
                    0,
                )
                self._apply_actor_collision_filter(
                    env_handle, actor_handle, scene_filter, env_id, debug_tag=f"s_avoid_obs_{slot}"
                )
                self._configure_s_avoid_pooled_body(env_handle, actor_handle, env_id)
                self.s_avoid_actor_handles[env_id][slot] = actor_handle
                actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
                self.s_avoid_actor_indices[env_id, slot] = actor_index

        if self.dynamic_asset is None or self.dynamic_actor_indices is None:
            return
        max_dyn = int(self.dynamic_actor_indices.shape[1])
        for obs_id in range(max_dyn):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, -5.0)
            actor_handle = self.gym.create_actor(
                env_handle,
                self.dynamic_asset,
                pose,
                f"dyn_obs_{obs_id}",
                group_id,
                create_filter,
                0,
            )
            self._apply_actor_collision_filter(env_handle, actor_handle, create_filter, env_id, debug_tag="dyn")
            self.dynamic_actor_handles[env_id][obs_id] = actor_handle
            actor_index = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
            self.dynamic_actor_indices[env_id, obs_id] = actor_index

    def _init_camera_buffers(self):
        """初始化相机图像接收buffer"""
        self.depth_raw = torch.zeros(
            self.num_envs,
            self.camera_cfg.height,
            self.camera_cfg.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.depth_images = torch.zeros(
            self.num_envs,
            1,
            self.camera_cfg.height,
            self.camera_cfg.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.rgb_images = None

    def _create_depth_cameras(self):
        """为所有环境创建深度相机"""
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_cfg.width
        camera_props.height = self.camera_cfg.height
        camera_props.enable_tensors = True
        camera_props.horizontal_fov = self.camera_cfg.horizontal_fov
        camera_props.near_plane = self.camera_cfg.near_clip
        camera_props.far_plane = self.camera_cfg.far_clip

        self.camera_handles = []
        for env_idx in range(self.num_envs):
            env_handle = self.envs[env_idx]
            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            if camera_handle == -1:
                continue
            robot_handle = self.actor_handles[env_idx]
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(
                self.camera_cfg.position[0],
                self.camera_cfg.position[1],
                self.camera_cfg.position[2],
            )
            pitch_rad = np.deg2rad(self.camera_cfg.pitch_deg)
            yaw_rad = np.deg2rad(self.camera_cfg.yaw_deg)
            roll_rad = np.deg2rad(self.camera_cfg.roll_deg)
            pitch_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), pitch_rad)
            yaw_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaw_rad)
            roll_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), roll_rad)
            local_transform.r = pitch_q * yaw_q * roll_q
            body_handle = self.gym.find_actor_rigid_body_handle(
                env_handle,
                robot_handle,
                "body",
            )
            if body_handle == -1:
                continue
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                body_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM,
            )
            self.camera_handles.append(camera_handle)
        self.depth_debug_count = 0

    def _get_depth_images(self):
        """获取所有环境的深度图像 (num_envs, H, W)"""
        if self.enable_camera and not self.cameras_created:
            self._create_depth_cameras()
            self.cameras_created = True
            if len(self.camera_handles) > 0:
                try:
                    self.gym.fetch_results(self.sim, True)
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    if isinstance(self.device, str) and ('cuda' in self.device or 'gpu' in self.device):
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                except Exception:
                    self.enable_camera = False
                    self.depth_raw.fill_(self.camera_cfg.far_clip)
                    return self.depth_raw

        if not self.enable_camera or len(self.camera_handles) == 0:
            self.depth_raw.fill_(self.camera_cfg.far_clip)
            return self.depth_raw
        try:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        except Exception:
            self.depth_raw.fill_(self.camera_cfg.far_clip)
            return self.depth_raw

        self.gym.start_access_image_tensors(self.sim)
        depth_images_list = []
        H = self.camera_cfg.height
        W = self.camera_cfg.width
        far = self.camera_cfg.far_clip
        near = self.camera_cfg.near_clip
        for env_idx in range(self.num_envs):
            if env_idx >= len(self.camera_handles):
                depth_images_list.append(torch.full((H, W), far, dtype=torch.float32, device=self.device))
                continue
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                self.envs[env_idx],
                self.camera_handles[env_idx],
                gymapi.IMAGE_DEPTH,
            )
            depth_image = gymtorch.wrap_tensor(depth_tensor)
            if (depth_image < 0).any():
                depth_image = -depth_image
            invalid_mask = ~torch.isfinite(depth_image)
            if invalid_mask.any():
                depth_image = depth_image.clone()
                depth_image[invalid_mask] = far
            depth_image = depth_image.clamp(near, far)
            depth_images_list.append(depth_image)
        self.gym.end_access_image_tensors(self.sim)
        self.depth_raw[:] = torch.stack(depth_images_list, dim=0)
        return self.depth_raw

    def _get_rgb_images(self, normalize: bool = True, channels_last: bool = False):
        """获取所有环境的 RGB 图像"""
        if self.enable_camera and not self.cameras_created:
            self._create_depth_cameras()
            self.cameras_created = True
            try:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
            except Exception:
                return torch.zeros(
                    self.num_envs,
                    3,
                    self.camera_cfg.height,
                    self.camera_cfg.width,
                    dtype=torch.float32 if normalize else torch.uint8,
                    device=self.device,
                )

        if not self.enable_camera or len(self.camera_handles) == 0:
            return torch.zeros(
                self.num_envs,
                3 if not channels_last else self.camera_cfg.height,
                self.camera_cfg.height if not channels_last else self.camera_cfg.width,
                self.camera_cfg.width if not channels_last else 3,
                dtype=torch.float32 if normalize else torch.uint8,
                device=self.device,
            )

        try:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        except Exception:
            return torch.zeros(
                self.num_envs,
                3 if not channels_last else self.camera_cfg.height,
                self.camera_cfg.height if not channels_last else self.camera_cfg.width,
                self.camera_cfg.width if not channels_last else 3,
                dtype=torch.float32 if normalize else torch.uint8,
                device=self.device,
            )

        self.gym.start_access_image_tensors(self.sim)
        rgb_list = []
        H = self.camera_cfg.height
        W = self.camera_cfg.width
        for env_idx in range(self.num_envs):
            if env_idx >= len(self.camera_handles):
                blank = torch.zeros(H, W, 3, dtype=torch.float32 if normalize else torch.uint8, device=self.device)
                rgb_list.append(blank)
                continue
            color_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                self.envs[env_idx],
                self.camera_handles[env_idx],
                gymapi.IMAGE_COLOR,
            )
            color_image = gymtorch.wrap_tensor(color_tensor)
            rgb = color_image[..., :3]
            if normalize:
                rgb = rgb.to(torch.float32) / 255.0
            if not channels_last:
                rgb = rgb.permute(2, 0, 1).contiguous()
            rgb_list.append(rgb)
        self.gym.end_access_image_tensors(self.sim)
        if channels_last:
            self.rgb_images = torch.stack(rgb_list, dim=0)
        else:
            self.rgb_images = torch.stack(rgb_list, dim=0)
        return self.rgb_images

    def _process_depth_for_network(self, depth_images):
        """预处理深度图，用于神经网络输入"""
        depth_normalized = (depth_images - self.camera_cfg.near_clip) / (
            self.camera_cfg.far_clip - self.camera_cfg.near_clip
        )
        depth_normalized = depth_normalized.unsqueeze(1)
        if hasattr(self.camera_cfg, 'output_size'):
            depth_normalized = torch.nn.functional.interpolate(
                depth_normalized,
                size=(self.camera_cfg.output_size, self.camera_cfg.output_size),
                mode='bilinear',
                align_corners=False
            )
        return depth_normalized

    def _init_scene_runtime(self):
        if self.s_avoid_enabled:
            self._init_s_avoid_runtime()
        if self.scene_generator is None or not self.scene_generator.has_dynamic:
            return
        self._init_dynamic_runtime()

    def _init_dynamic_runtime(self):
        if self.dynamic_actor_indices is None:
            return
        if isinstance(self.dynamic_actor_indices, list):
            self.dynamic_actor_indices = np.array(self.dynamic_actor_indices, dtype=np.int32)
        if isinstance(self.dynamic_actor_indices, np.ndarray):
            self.dynamic_actor_indices = torch.tensor(
                self.dynamic_actor_indices, device=self.device, dtype=torch.int32
            )
        max_dyn = int(self.dynamic_actor_indices.shape[1]) if self.dynamic_actor_indices is not None else 0
        if max_dyn <= 0:
            return
        self.dynamic_actor_indices_flat = self.dynamic_actor_indices.reshape(-1).contiguous()
        self.dynamic_actor_indices_flat_long = self.dynamic_actor_indices_flat.to(torch.long)
        self.dynamic_active = torch.zeros(self.num_envs, max_dyn, device=self.device, dtype=torch.bool)
        self.dynamic_start = torch.zeros(self.num_envs, max_dyn, 3, device=self.device)
        self.dynamic_dir = torch.zeros(self.num_envs, max_dyn, 3, device=self.device)
        self.dynamic_path_len = torch.zeros(self.num_envs, max_dyn, device=self.device)
        self.dynamic_phase = torch.zeros(self.num_envs, max_dyn, device=self.device)
        self.dynamic_period = torch.ones(self.num_envs, max_dyn, device=self.device)
        self.dynamic_height = torch.zeros(self.num_envs, max_dyn, device=self.device)
        self.dynamic_quat = torch.zeros(self.num_envs, max_dyn, 4, device=self.device)
        self.dynamic_quat[..., 3] = 1.0

    def _init_s_avoid_runtime(self):
        if self.s_avoid_actor_indices is None:
            return
        if isinstance(self.s_avoid_actor_indices, np.ndarray):
            self.s_avoid_actor_indices = torch.tensor(
                self.s_avoid_actor_indices, device=self.device, dtype=torch.int32
            )
        self.s_avoid_actor_indices_flat = self.s_avoid_actor_indices.reshape(-1).contiguous()
        self.s_avoid_actor_indices_flat_long = self.s_avoid_actor_indices_flat.to(torch.long)
        self.s_avoid_active = torch.zeros(
            (self.num_envs, self.s_avoid_total_slots), device=self.device, dtype=torch.bool
        )
        self.s_avoid_pos_world = torch.zeros(
            (self.num_envs, self.s_avoid_total_slots, 3), device=self.device, dtype=torch.float
        )
        self.s_avoid_quat_world = torch.zeros(
            (self.num_envs, self.s_avoid_total_slots, 4), device=self.device, dtype=torch.float
        )
        self.s_avoid_quat_world[..., 3] = 1.0
        self.s_avoid_band_x_min = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_band_x_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_band_y_min = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_band_y_max = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_spawn_world_y = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_episode_collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.s_avoid_episode_exposed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.s_avoid_episode_goal_init_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_episode_goal_best_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.s_avoid_env_episode_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.s_avoid_stage_per_env = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.s_avoid_stage = 1
        self.s_avoid_total_completed_episodes = 0
        stage12_window = int(
            getattr(
                self.cfg.terrain,
                "avoid_stage12_window",
                getattr(self.cfg.terrain, "avoid_stage_switch_window", 100),
            )
        )
        stage23_window = int(getattr(self.cfg.terrain, "avoid_stage23_window", stage12_window))
        stage34_window = int(getattr(self.cfg.terrain, "avoid_stage34_window", stage23_window))
        stage3_window = int(getattr(self.cfg.terrain, "avoid_stage3_shrink_window", 100))
        self.s_avoid_stage_metric_hists = {
            1: self._make_s_avoid_metric_history(stage12_window),
            2: self._make_s_avoid_metric_history(stage23_window),
            3: self._make_s_avoid_metric_history(stage34_window),
            4: self._make_s_avoid_metric_history(stage3_window),
        }
        self.s_avoid_stage_completed_episodes = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
        self.s_avoid_corridor_width = float(getattr(self.cfg.terrain, "avoid_stage3_width_start", 1.2))
        self.s_avoid_last_shrink_stage_episode = 0
        self.s_avoid_stage_presets = self._build_s_avoid_stage_presets()
        self.extras["avoid_stage"] = int(self.s_avoid_stage)
        self.extras["avoid_stage_collision_rate"] = 0.0
        self.extras["avoid_shrink_collision_rate"] = 0.0
        self.extras["avoid_stage_exposure_rate"] = 0.0
        self.extras["avoid_stage_progress_rate"] = 0.0
        self.extras["avoid_stage_success_rate"] = 0.0
        self.extras["avoid_corridor_width"] = float(self.s_avoid_corridor_width)
        self.extras["avoid_completed_episodes"] = 0
        self.extras["avoid_stage_completed_episodes"] = 0
        self.extras["avoid_stage_window"] = int(stage12_window)
        self.extras["avoid_shrink_window"] = int(stage3_window)
        self.extras["avoid_nearest_obstacle_dist"] = 5.0
        self.extras["avoid_stage_switch_event"] = 0.0
        self.extras["avoid_stage_switch_from"] = 0.0
        self.extras["avoid_stage_switch_to"] = 0.0
        self.extras["avoid_stage_switch_collision_rate"] = 0.0
        self.extras["avoid_stage_switch_exposure_rate"] = 0.0
        self.extras["avoid_stage_switch_progress_rate"] = 0.0
        self.extras["avoid_stage_switch_success_rate"] = 0.0
        self.extras["avoid_stage3_shrink_event"] = 0.0
        self.extras["avoid_stage3_shrink_from_width"] = float(self.s_avoid_corridor_width)
        self.extras["avoid_stage3_shrink_to_width"] = float(self.s_avoid_corridor_width)

    def _get_s_avoid_stage_template(self):
        total_slots = int(self.s_avoid_total_slots)
        cap_slots = int(self.s_avoid_capsule_slot_count)
        pos = np.zeros((total_slots, 3), dtype=np.float32)
        for slot in range(total_slots):
            park_x, park_y, park_z = self._get_s_avoid_park_local_pose(slot)
            pos[slot] = np.array([park_x, park_y, park_z], dtype=np.float32)
        quat = np.zeros((total_slots, 4), dtype=np.float32)
        quat[:, 3] = 1.0
        if cap_slots > 0:
            quat[:cap_slots] = np.array(
                [self.s_avoid_capsule_quat.x, self.s_avoid_capsule_quat.y, self.s_avoid_capsule_quat.z, self.s_avoid_capsule_quat.w],
                dtype=np.float32,
            )
        active = np.zeros((total_slots,), dtype=np.bool_)
        return active, pos, quat

    def _make_s_avoid_preset(
        self,
        *,
        stage: int,
        rng: np.random.RandomState,
        cap_count: int = 0,
        box_count: int = 0,
        spacing: float,
        core_count: int,
        spawn_clear: float,
        cap_r: float,
        box_r: float,
        cap_z: float,
        box_z: float,
        half_extent: float,
        stage12_core_half_width: float,
        stage12_core_y_min: float,
        stage12_core_y_max: float,
        stage12_band_half_width: float,
        stage12_band_y_min: float,
        stage12_band_y_max: float,
        stage12_forbidden,
    ):
        active, pos, quat = self._get_s_avoid_stage_template()
        cap_slots = int(self.s_avoid_capsule_slot_count)
        box_slots = int(self.s_avoid_box_slot_count)
        total_count = int(cap_count + box_count)
        if total_count <= 0:
            return dict(active=active, pos=pos, quat=quat)
        safe_r = max(cap_r, box_r)
        points = self._sample_s_avoid_points(
            rng=rng,
            count=min(total_count, max(0, int(core_count))),
            half_extent=half_extent,
            min_spacing=spacing,
            spawn_clearance=spawn_clear + safe_r,
            x_range=(-stage12_core_half_width, stage12_core_half_width),
            y_range=(stage12_core_y_min, stage12_core_y_max),
            avoid_zones=stage12_forbidden,
        )
        points = self._sample_s_avoid_points(
            rng=rng,
            count=total_count,
            half_extent=half_extent,
            min_spacing=spacing,
            spawn_clearance=spawn_clear + safe_r,
            x_range=(-stage12_band_half_width, stage12_band_half_width),
            y_range=(stage12_band_y_min, stage12_band_y_max),
            avoid_zones=stage12_forbidden,
            existing_points=points,
        )
        for i in range(min(cap_count, cap_slots)):
            x, y = points[i]
            active[i] = True
            pos[i] = np.array([x, y, cap_z], dtype=np.float32)
        for j in range(min(box_count, box_slots)):
            x, y = points[cap_count + j]
            slot = cap_slots + j
            active[slot] = True
            pos[slot] = np.array([x, y, box_z], dtype=np.float32)
            yaw = float(rng.uniform(-math.pi, math.pi))
            quat[slot] = np.array(
                [0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)],
                dtype=np.float32,
            )
        return dict(active=active, pos=pos, quat=quat)

    def _build_s_avoid_stage_presets(self):
        if not self.s_avoid_enabled:
            return {}
        seed0 = int(getattr(self.cfg.terrain, "avoid_seed", 7001))
        cap_slots = int(self.s_avoid_capsule_slot_count)
        box_slots = int(self.s_avoid_box_slot_count)
        cap_r = float(getattr(self.cfg.terrain, "avoid_capsule_radius", 0.15))
        cap_h = float(getattr(self.cfg.terrain, "avoid_capsule_height", 0.5))
        cap_z = 0.5 * cap_h
        box_h = float(getattr(self.cfg.terrain, "avoid_box_size_z", 0.5))
        box_z = 0.5 * box_h
        box_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_y", 0.4))
        box_r = math.sqrt(box_hx * box_hx + box_hy * box_hy)
        half_extent = 0.5 * min(float(self.cfg.terrain.terrain_width), float(self.cfg.terrain.terrain_length))
        half_extent = max(half_extent - float(getattr(self.cfg.terrain, "avoid_spawn_extra_margin", 0.1)), 0.6)
        spawn_clear = float(getattr(self.cfg.terrain, "avoid_spawn_clearance", 0.5))
        stage12_band_half_width = float(getattr(self.cfg.terrain, "avoid_stage12_band_half_width", 1.05))
        stage12_band_y_min = float(getattr(self.cfg.terrain, "avoid_stage12_band_y_min", -0.5))
        stage12_band_y_max = float(getattr(self.cfg.terrain, "avoid_stage12_band_y_max", 2.4))
        stage12_core_half_width = float(getattr(self.cfg.terrain, "avoid_stage12_core_half_width", 0.4))
        stage12_core_y_min = float(getattr(self.cfg.terrain, "avoid_stage12_core_y_min", 0.0))
        stage12_core_y_max = float(getattr(self.cfg.terrain, "avoid_stage12_core_y_max", 1.6))
        stage12_spawn_y = -1.6
        stage12_forbidden = [
            (0.0, stage12_spawn_y, spawn_clear + max(cap_r, box_r)),
        ]

        stage1_count = int(getattr(self.cfg.terrain, "avoid_stage1_preset_count", 48))
        stage15_count = int(getattr(self.cfg.terrain, "avoid_stage15_preset_count", 48))
        stage2_count = int(getattr(self.cfg.terrain, "avoid_stage2_preset_count", 56))
        stage1_min = int(getattr(self.cfg.terrain, "avoid_stage1_count_min", 3))
        stage1_max = int(getattr(self.cfg.terrain, "avoid_stage1_count_max", 5))
        stage15_min = int(getattr(self.cfg.terrain, "avoid_stage15_count_min", 5))
        stage15_max = int(getattr(self.cfg.terrain, "avoid_stage15_count_max", 6))
        stage2_min = int(getattr(self.cfg.terrain, "avoid_stage2_count_min", 6))
        stage2_max = int(getattr(self.cfg.terrain, "avoid_stage2_count_max", 8))
        stage1_spacing = float(getattr(self.cfg.terrain, "avoid_stage1_min_spacing", 1.2))
        stage15_spacing = float(getattr(self.cfg.terrain, "avoid_stage15_min_spacing", 0.9))
        stage2_spacing = float(getattr(self.cfg.terrain, "avoid_stage2_min_spacing", 0.7))
        stage1_core = int(getattr(self.cfg.terrain, "avoid_stage1_core_count", 1))
        stage15_core = int(getattr(self.cfg.terrain, "avoid_stage15_core_count", 2))
        stage2_core = int(getattr(self.cfg.terrain, "avoid_stage2_core_count", 2))

        presets = {1: [], 2: [], 3: []}
        for preset_idx in range(max(1, stage1_count)):
            rng = np.random.RandomState(seed0 + 100000 + 97 * preset_idx)
            cap_count = int(rng.randint(stage1_min, min(stage1_max, cap_slots) + 1))
            presets[1].append(
                self._make_s_avoid_preset(
                    stage=1,
                    rng=rng,
                    cap_count=cap_count,
                    box_count=0,
                    spacing=stage1_spacing,
                    core_count=stage1_core,
                    spawn_clear=spawn_clear,
                    cap_r=cap_r,
                    box_r=box_r,
                    cap_z=cap_z,
                    box_z=box_z,
                    half_extent=half_extent,
                    stage12_core_half_width=stage12_core_half_width,
                    stage12_core_y_min=stage12_core_y_min,
                    stage12_core_y_max=stage12_core_y_max,
                    stage12_band_half_width=stage12_band_half_width,
                    stage12_band_y_min=stage12_band_y_min,
                    stage12_band_y_max=stage12_band_y_max,
                    stage12_forbidden=stage12_forbidden,
                )
            )
        for preset_idx in range(max(1, stage15_count)):
            rng = np.random.RandomState(seed0 + 200000 + 97 * preset_idx)
            cap_count = int(rng.randint(stage15_min, min(stage15_max, cap_slots) + 1))
            presets[2].append(
                self._make_s_avoid_preset(
                    stage=2,
                    rng=rng,
                    cap_count=cap_count,
                    box_count=0,
                    spacing=stage15_spacing,
                    core_count=stage15_core,
                    spawn_clear=spawn_clear,
                    cap_r=cap_r,
                    box_r=box_r,
                    cap_z=cap_z,
                    box_z=box_z,
                    half_extent=half_extent,
                    stage12_core_half_width=stage12_core_half_width,
                    stage12_core_y_min=stage12_core_y_min,
                    stage12_core_y_max=stage12_core_y_max,
                    stage12_band_half_width=stage12_band_half_width,
                    stage12_band_y_min=stage12_band_y_min,
                    stage12_band_y_max=stage12_band_y_max,
                    stage12_forbidden=stage12_forbidden,
                )
            )
        for preset_idx in range(max(1, stage2_count)):
            rng = np.random.RandomState(seed0 + 300000 + 97 * preset_idx)
            total_count = int(rng.randint(stage2_min, min(stage2_max, cap_slots + box_slots) + 1))
            min_box = min(box_slots, 2 if box_slots > 0 else 0)
            box_count = min(box_slots, max(min_box, total_count - cap_slots))
            cap_count = max(0, min(cap_slots, total_count - box_count))
            box_count = min(box_slots, max(0, total_count - cap_count))
            presets[3].append(
                self._make_s_avoid_preset(
                    stage=3,
                    rng=rng,
                    cap_count=cap_count,
                    box_count=box_count,
                    spacing=stage2_spacing,
                    core_count=stage2_core,
                    spawn_clear=spawn_clear,
                    cap_r=cap_r,
                    box_r=box_r,
                    cap_z=cap_z,
                    box_z=box_z,
                    half_extent=half_extent,
                    stage12_core_half_width=stage12_core_half_width,
                    stage12_core_y_min=stage12_core_y_min,
                    stage12_core_y_max=stage12_core_y_max,
                    stage12_band_half_width=stage12_band_half_width,
                    stage12_band_y_min=stage12_band_y_min,
                    stage12_band_y_max=stage12_band_y_max,
                    stage12_forbidden=stage12_forbidden,
                )
            )
        return presets

    def _get_s_avoid_spawn_local_y(self, stage: int) -> float:
        wall_l = float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))
        if int(stage) == 4:
            return -0.5 * wall_l + 0.8
        return -1.6

    def _compute_s_avoid_band_world(
        self,
        env_id: int,
        stage: int,
        active: np.ndarray,
        pos_local: np.ndarray,
        quat: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        margin_x = float(getattr(self.cfg.terrain, "avoid_band_margin_x", 0.45))
        margin_y = float(getattr(self.cfg.terrain, "avoid_band_margin_y", 0.30))
        activate_progress = float(getattr(self.nav_cfg, "avoid_band_activate_progress", 0.50))
        goal_y_range = getattr(self.cfg.navigation, "goal_range_y", [1.2, 2.8])
        goal_y_max = float(goal_y_range[1]) if len(goal_y_range) >= 2 else 2.8

        cap_slots = int(self.s_avoid_capsule_slot_count)
        box_slots = int(self.s_avoid_box_slot_count)
        box_end = cap_slots + box_slots

        cap_r = float(getattr(self.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_y", 0.4))
        wall_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))

        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")

        for slot in range(int(self.s_avoid_total_slots)):
            if not bool(active[slot]):
                continue
            cx = float(pos_local[slot, 0])
            cy = float(pos_local[slot, 1])
            if slot < cap_slots:
                half_x = cap_r
                half_y = cap_r
            else:
                qz = float(quat[slot, 2])
                qw = float(quat[slot, 3])
                yaw = 2.0 * math.atan2(qz, qw)
                cos_yaw = abs(math.cos(yaw))
                sin_yaw = abs(math.sin(yaw))
                if slot < box_end:
                    half_x = cos_yaw * box_hx + sin_yaw * box_hy
                    half_y = sin_yaw * box_hx + cos_yaw * box_hy
                else:
                    half_x = cos_yaw * wall_hx + sin_yaw * wall_hy
                    half_y = sin_yaw * wall_hx + cos_yaw * wall_hy
            x_min = min(x_min, cx - half_x)
            x_max = max(x_max, cx + half_x)
            y_min = min(y_min, cy - half_y)
            y_max = max(y_max, cy + half_y)

        spawn_local_y = self._get_s_avoid_spawn_local_y(stage)
        band_front_start_y = spawn_local_y + activate_progress
        if not math.isfinite(x_min) or not math.isfinite(y_min):
            half_w = 0.5 * float(self.cfg.terrain.terrain_width)
            env_origin_x = float(self.env_origins[env_id, 0].item())
            env_origin_y = float(self.env_origins[env_id, 1].item())
            return (
                env_origin_x - half_w,
                env_origin_x + half_w,
                env_origin_y + band_front_start_y,
                env_origin_y + max(goal_y_max + margin_y, band_front_start_y + 0.5),
            )

        x_half = max(abs(x_min), abs(x_max)) + margin_x
        band_y_min_local = min(y_min - margin_y, band_front_start_y)
        band_y_max_local = max(y_max + margin_y, goal_y_max + margin_y)
        env_origin_x = float(self.env_origins[env_id, 0].item())
        env_origin_y = float(self.env_origins[env_id, 1].item())
        return (
            env_origin_x - x_half,
            env_origin_x + x_half,
            env_origin_y + band_y_min_local,
            env_origin_y + band_y_max_local,
        )


    def _get_scene_spec(self, env_id: int):
        if self.scene_specs is None:
            return None
        level = int(self.terrain_levels[env_id].item()) if hasattr(self, "terrain_levels") else 0
        col = int(self.terrain_types[env_id].item()) if hasattr(self, "terrain_types") else 0
        level = max(0, min(level, len(self.scene_specs) - 1))
        col = max(0, min(col, len(self.scene_specs[level]) - 1))
        return self.scene_specs[level][col]

    def _scene_meta_params(self, meta: dict) -> dict:
        scene_type = meta.get("scene_type", None)
        params: dict = {}
        if scene_type == "s1_corridor_gate":
            length = float(meta.get("L", self.cfg.terrain.terrain_length))
            width_nom = float(meta.get("W0", 0.5 * self.cfg.terrain.terrain_width)) * 2.0
            gate_width = float(meta.get("Wg", 0.5 * width_nom)) * 2.0
            gates_meta = meta.get("gates", []) or []
            gates = []
            for gate in gates_meta:
                try:
                    y0, y1 = gate
                except Exception:
                    continue
                y0 = float(y0)
                y1 = float(y1)
                length_gate = abs(y1 - y0)
                y_center = 0.5 * (y0 + y1) - 0.5 * length
                gates.append({"y0": y_center, "length": length_gate, "door_width": gate_width})
            params.update(
                {
                    "corridor_length": length,
                    "corridor_width_nom": width_nom,
                    "corridor_gates": gates,
                    "corridor_x_center": 0.0,
                }
            )
        elif scene_type == "s3_doorway_rooms":
            params["room_width"] = float(meta.get("W", self.cfg.terrain.terrain_width))
        return params

    def _scene_spec_from_meta(self, meta: dict, env_id: int) -> Optional[SceneSpec]:
        if not isinstance(meta, dict):
            return None
        scene_type = meta.get("scene_type", None) or "unknown"
        params = meta.get("params", {}) or {}
        static_obstacles = meta.get("static_obstacles", []) or []
        layout_seed = meta.get("layout_seed", None)
        return SceneSpec(
            scene_type=scene_type,
            params=dict(params),
            static_obstacles=list(static_obstacles),
            layout_seed=layout_seed,
        )

    @staticmethod
    def _normalize_difficulty_override_tensor(
        difficulty_override,
        *,
        expected_num: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(difficulty_override):
            d_tensor = difficulty_override.to(device=device, dtype=dtype)
        else:
            d_tensor = torch.as_tensor(difficulty_override, device=device, dtype=dtype)
        if d_tensor.ndim == 0:
            d_vec = torch.full((expected_num,), float(d_tensor.item()), device=device, dtype=dtype)
        elif d_tensor.numel() == expected_num:
            d_vec = d_tensor.reshape(-1)
        else:
            raise ValueError(
                f"scene_difficulty_override shape mismatch: {tuple(d_tensor.shape)} "
                f"(expected {expected_num})"
            )
        d_vec = torch.nan_to_num(d_vec, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(d_vec, 0.0, 1.0)

    def _current_scene_difficulty(
        self,
        *,
        expected_num: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if device is None:
            device = self.device
        num = int(self.num_envs if expected_num is None else expected_num)

        terrain_type = str(getattr(self.cfg.terrain, "terrain_type", "")).lower()
        difficulty_override = getattr(self, "scene_difficulty_override", None)
        if difficulty_override is not None and terrain_type in ("s0_follow_plane", "s0"):
            return self._normalize_difficulty_override_tensor(
                difficulty_override,
                expected_num=num,
                device=device,
                dtype=dtype,
            )

        if hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level"):
            denom = max(1, int(self.max_terrain_level))
            d = self.terrain_levels[:num].to(device=device, dtype=dtype) / float(denom)
            d = torch.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
            return torch.clamp(d, 0.0, 1.0)

        return torch.zeros(num, device=device, dtype=dtype)

    def _get_scene_difficulty(self, env_id: int) -> float:
        if not hasattr(self, "terrain_levels") or self.terrain_levels.numel() == 0:
            return 0.0
        env_id = int(max(0, min(int(env_id), self.terrain_levels.numel() - 1)))
        if hasattr(self, "max_terrain_level"):
            denom = max(1, int(self.max_terrain_level))
            d = float(self.terrain_levels[env_id].item()) / float(denom)
        else:
            d = 0.0
        if not np.isfinite(d):
            d = 0.0
        return float(np.clip(d, 0.0, 1.0))

    def _fill_dynamic_buffers(self, env_id: int, dynamic_specs):
        if self.dynamic_actor_indices is None:
            return
        self.dynamic_active[env_id].fill_(False)
        self.dynamic_start[env_id].zero_()
        self.dynamic_dir[env_id].zero_()
        self.dynamic_path_len[env_id].zero_()
        self.dynamic_phase[env_id].zero_()
        self.dynamic_period[env_id].fill_(1.0)
        self.dynamic_height[env_id].zero_()

        max_dyn = int(self.dynamic_actor_indices.shape[1])
        for idx, spec in enumerate(dynamic_specs[:max_dyn]):
            start = torch.tensor(spec.path_start, device=self.device, dtype=torch.float32)
            end = torch.tensor(spec.path_end, device=self.device, dtype=torch.float32)
            delta = end - start
            delta_xy = delta.clone()
            delta_xy[2] = 0.0
            path_len = torch.norm(delta_xy[:2])
            direction = delta_xy / (path_len + 1e-6)
            self.dynamic_active[env_id, idx] = True
            self.dynamic_start[env_id, idx] = start
            self.dynamic_dir[env_id, idx] = direction
            self.dynamic_path_len[env_id, idx] = path_len
            self.dynamic_phase[env_id, idx] = float(spec.phase)
            self.dynamic_period[env_id, idx] = float(spec.period)
            self.dynamic_height[env_id, idx] = float(spec.size[2])

    def _s_avoid_collision_rate(self, history: deque) -> float:
        if history is None or len(history) == 0:
            return 0.0
        return float(sum(history) / len(history))

    def _make_s_avoid_metric_history(self, window_size: int):
        maxlen = max(1, int(window_size))
        return {
            "collision": deque(maxlen=maxlen),
            "exposure": deque(maxlen=maxlen),
            "progress": deque(maxlen=maxlen),
            "success": deque(maxlen=maxlen),
        }

    def _clear_s_avoid_metric_history(self, stage: int) -> None:
        stage = int(stage)
        stage_hists = self.s_avoid_stage_metric_hists.get(stage, None)
        if stage_hists is None:
            return
        for hist in stage_hists.values():
            hist.clear()
        self.s_avoid_stage_completed_episodes[stage] = 0
        if stage == 3:
            self.s_avoid_last_shrink_stage_episode = 0

    def _get_s_avoid_stage_metric_rates(self, stage: int) -> Tuple[float, float, float, float]:
        stage = int(stage)
        stage_hists = self.s_avoid_stage_metric_hists.get(stage, None)
        if stage_hists is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            self._s_avoid_collision_rate(stage_hists["collision"]),
            self._s_avoid_collision_rate(stage_hists["exposure"]),
            self._s_avoid_collision_rate(stage_hists["progress"]),
            self._s_avoid_collision_rate(stage_hists["success"]),
        )

    def _get_s_avoid_stage_window_size(self, stage: int) -> int:
        stage = int(stage)
        stage_hists = self.s_avoid_stage_metric_hists.get(stage, None)
        if stage_hists is None:
            return 0
        return int(stage_hists["collision"].maxlen)

    def _advance_s_avoid_stage(self, next_stage: int) -> None:
        next_stage = int(next_stage)
        if next_stage <= int(self.s_avoid_stage):
            return
        self._clear_s_avoid_metric_history(next_stage)
        self.s_avoid_stage = next_stage

    def _compute_s_avoid_nearest_obstacle_distance(self) -> Optional[torch.Tensor]:
        if not self.s_avoid_enabled or self.s_avoid_actor_indices is None:
            return None
        if self.s_avoid_total_slots <= 0:
            return None

        robot_xy = self.root_states[:, :2]
        inf = torch.full((self.num_envs,), 1.0e6, device=self.device, dtype=torch.float32)
        nearest = inf.clone()
        active = self.s_avoid_active

        cap_slots = int(getattr(self.cfg.terrain, "avoid_capsule_slots", 0))
        box_slots = int(getattr(self.cfg.terrain, "avoid_box_slots", 0))
        cap_r = float(getattr(self.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_y", 0.4))
        wall_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))

        for slot in range(int(self.s_avoid_total_slots)):
            active_slot = active[:, slot]
            if not bool(active_slot.any().item()):
                continue

            delta = robot_xy - self.s_avoid_pos_world[:, slot, :2]
            if slot < cap_slots:
                clearance = torch.norm(delta, dim=1) - cap_r
            else:
                quat = self.s_avoid_quat_world[:, slot]
                yaw = 2.0 * torch.atan2(quat[:, 2], quat[:, 3])
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
                local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
                if slot < cap_slots + box_slots:
                    hx, hy = box_hx, box_hy
                else:
                    hx, hy = wall_hx, wall_hy
                dx = torch.abs(local_x) - hx
                dy = torch.abs(local_y) - hy
                outside = torch.norm(torch.stack([torch.clamp(dx, min=0.0), torch.clamp(dy, min=0.0)], dim=1), dim=1)
                inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
                clearance = outside + inside

            clearance = torch.where(active_slot, clearance, inf)
            nearest = torch.minimum(nearest, clearance)

        nearest = torch.where(nearest >= 1.0e5, torch.full_like(nearest, 5.0), nearest)
        return nearest

    def _update_s_avoid_curriculum(
        self,
        episode_collision_flags: torch.Tensor,
        episode_stage_ids: Optional[torch.Tensor] = None,
        episode_exposure_flags: Optional[torch.Tensor] = None,
        episode_progress_flags: Optional[torch.Tensor] = None,
        episode_success_flags: Optional[torch.Tensor] = None,
    ):
        if not self.s_avoid_enabled or episode_collision_flags.numel() == 0:
            return
        flags = episode_collision_flags.detach().to(device="cpu", dtype=torch.bool).tolist()
        if episode_stage_ids is None:
            episode_stage_ids = torch.full_like(episode_collision_flags, int(self.s_avoid_stage), dtype=torch.long)
        stage_ids = episode_stage_ids.detach().to(device="cpu", dtype=torch.long).tolist()
        if episode_exposure_flags is None:
            episode_exposure_flags = torch.ones_like(episode_collision_flags, dtype=torch.bool)
        exposure_flags = episode_exposure_flags.detach().to(device="cpu", dtype=torch.bool).tolist()
        if episode_progress_flags is None:
            episode_progress_flags = torch.ones_like(episode_collision_flags, dtype=torch.bool)
        progress_flags = episode_progress_flags.detach().to(device="cpu", dtype=torch.bool).tolist()
        if episode_success_flags is None:
            episode_success_flags = torch.ones_like(episode_collision_flags, dtype=torch.bool)
        success_flags = episode_success_flags.detach().to(device="cpu", dtype=torch.bool).tolist()
        for stage_id, flag, exposed, progressed, succeeded in zip(
            stage_ids,
            flags,
            exposure_flags,
            progress_flags,
            success_flags,
        ):
            stage_hists = self.s_avoid_stage_metric_hists.get(int(stage_id), None)
            if stage_hists is None:
                continue
            stage_hists["collision"].append(1.0 if bool(flag) else 0.0)
            stage_hists["exposure"].append(1.0 if bool(exposed) else 0.0)
            stage_hists["progress"].append(1.0 if bool(progressed) else 0.0)
            stage_hists["success"].append(1.0 if bool(succeeded) else 0.0)
            self.s_avoid_stage_completed_episodes[int(stage_id)] = (
                int(self.s_avoid_stage_completed_episodes.get(int(stage_id), 0)) + 1
            )
        self.s_avoid_total_completed_episodes += len(flags)

        current_stage = int(self.s_avoid_stage)
        stage_hists = self.s_avoid_stage_metric_hists.get(current_stage, None)
        if stage_hists is None:
            return
        rate_stage, exposure_stage, progress_stage, success_stage = self._get_s_avoid_stage_metric_rates(current_stage)
        stage_window_size = self._get_s_avoid_stage_window_size(current_stage)
        stage_completed_eps = int(self.s_avoid_stage_completed_episodes.get(current_stage, 0))
        shrink_th = float(getattr(self.cfg.terrain, "avoid_stage3_shrink_collision_threshold", 0.08))
        shrink_success_th = float(getattr(self.cfg.terrain, "avoid_stage3_shrink_success_threshold", 0.60))
        shrink_step = float(getattr(self.cfg.terrain, "avoid_stage3_shrink_step", 0.05))
        width_min = float(getattr(self.cfg.terrain, "avoid_stage3_width_min", 0.85))
        shrink_cooldown = int(getattr(self.cfg.terrain, "avoid_stage3_shrink_cooldown_episodes", 50))

        switched = False
        old_stage = current_stage
        self.extras["avoid_stage_switch_event"] = 0.0
        self.extras["avoid_stage3_shrink_event"] = 0.0
        if (
            current_stage == 1
            and stage_completed_eps >= int(getattr(self.cfg.terrain, "avoid_stage12_min_episodes", 200))
            and len(stage_hists["collision"]) >= stage_hists["collision"].maxlen
            and exposure_stage >= float(getattr(self.cfg.terrain, "avoid_stage12_exposure_threshold", 0.60))
            and progress_stage >= float(getattr(self.cfg.terrain, "avoid_stage12_progress_threshold", 0.35))
            and success_stage >= float(getattr(self.cfg.terrain, "avoid_stage12_success_threshold", 0.30))
            and rate_stage < float(getattr(self.cfg.terrain, "avoid_stage12_collision_threshold", 0.05))
        ):
            self._advance_s_avoid_stage(2)
            switched = True
        elif (
            current_stage == 2
            and stage_completed_eps >= int(getattr(self.cfg.terrain, "avoid_stage23_min_episodes", 200))
            and len(stage_hists["collision"]) >= stage_hists["collision"].maxlen
            and exposure_stage >= float(getattr(self.cfg.terrain, "avoid_stage23_exposure_threshold", 0.60))
            and progress_stage >= float(getattr(self.cfg.terrain, "avoid_stage23_progress_threshold", 0.35))
            and success_stage >= float(getattr(self.cfg.terrain, "avoid_stage23_success_threshold", 0.30))
            and rate_stage < float(getattr(self.cfg.terrain, "avoid_stage23_collision_threshold", 0.05))
        ):
            self._advance_s_avoid_stage(3)
            switched = True
        elif (
            current_stage == 3
            and stage_completed_eps >= int(getattr(self.cfg.terrain, "avoid_stage34_min_episodes", 200))
            and len(stage_hists["collision"]) >= stage_hists["collision"].maxlen
            and exposure_stage >= float(getattr(self.cfg.terrain, "avoid_stage34_exposure_threshold", 0.60))
            and progress_stage >= float(getattr(self.cfg.terrain, "avoid_stage34_progress_threshold", 0.35))
            and success_stage >= float(getattr(self.cfg.terrain, "avoid_stage34_success_threshold", 0.30))
            and rate_stage < float(getattr(self.cfg.terrain, "avoid_stage34_collision_threshold", 0.05))
        ):
            self._advance_s_avoid_stage(4)
            switched = True

        if switched:
            self.extras["avoid_stage_switch_event"] = 1.0
            self.extras["avoid_stage_switch_from"] = float(old_stage)
            self.extras["avoid_stage_switch_to"] = float(self.s_avoid_stage)
            self.extras["avoid_stage_switch_collision_rate"] = float(rate_stage)
            self.extras["avoid_stage_switch_exposure_rate"] = float(exposure_stage)
            self.extras["avoid_stage_switch_progress_rate"] = float(progress_stage)
            self.extras["avoid_stage_switch_success_rate"] = float(success_stage)
            print(
                f"[s_avoid_basic] stage {old_stage}->{self.s_avoid_stage} | "
                f"stage_episodes={stage_completed_eps}, total_episodes={self.s_avoid_total_completed_episodes}, "
                f"window={stage_window_size}, collision={rate_stage:.3f}, exposure={exposure_stage:.3f}, "
                f"progress={progress_stage:.3f}, success={success_stage:.3f}"
            )

        current_stage = int(self.s_avoid_stage)
        rate_stage, exposure_stage, progress_stage, success_stage = self._get_s_avoid_stage_metric_rates(current_stage)
        stage_window_size = self._get_s_avoid_stage_window_size(current_stage)
        stage_completed_eps = int(self.s_avoid_stage_completed_episodes.get(current_stage, 0))
        shrink_rate = 0.0
        if self.s_avoid_stage == 4:
            shrink_rate = rate_stage
            shrink_success = success_stage
            stage3_hists = self.s_avoid_stage_metric_hists.get(4, None)
            enough_window = (
                stage3_hists is not None
                and len(stage3_hists["collision"]) >= stage3_hists["collision"].maxlen
            )
            enough_cooldown = (
                stage_completed_eps - self.s_avoid_last_shrink_stage_episode
            ) >= shrink_cooldown
            if (
                enough_window
                and enough_cooldown
                and shrink_rate < shrink_th
                and shrink_success >= shrink_success_th
                and self.s_avoid_corridor_width > width_min + 1e-6
            ):
                old_width = float(self.s_avoid_corridor_width)
                self.s_avoid_corridor_width = max(width_min, old_width - shrink_step)
                self.s_avoid_last_shrink_stage_episode = stage_completed_eps
                self.extras["avoid_stage3_shrink_event"] = 1.0
                self.extras["avoid_stage3_shrink_from_width"] = float(old_width)
                self.extras["avoid_stage3_shrink_to_width"] = float(self.s_avoid_corridor_width)
                print(
                    f"[s_avoid_basic] stage3 corridor width {old_width:.2f}->{self.s_avoid_corridor_width:.2f} "
                    f"(stage_episodes={stage_completed_eps}, window={stage_window_size}, "
                    f"collision={shrink_rate:.3f}, success={shrink_success:.3f})"
                )
        elif self.extras.get("avoid_stage3_shrink_event", 0.0) == 0.0:
            self.extras["avoid_stage3_shrink_from_width"] = float(self.s_avoid_corridor_width)
            self.extras["avoid_stage3_shrink_to_width"] = float(self.s_avoid_corridor_width)

        self.extras["avoid_stage"] = int(self.s_avoid_stage)
        self.extras["avoid_stage_collision_rate"] = float(rate_stage)
        self.extras["avoid_shrink_collision_rate"] = float(shrink_rate)
        self.extras["avoid_stage_exposure_rate"] = float(exposure_stage)
        self.extras["avoid_stage_progress_rate"] = float(progress_stage)
        self.extras["avoid_stage_success_rate"] = float(success_stage)
        self.extras["avoid_corridor_width"] = float(self.s_avoid_corridor_width)
        self.extras["avoid_completed_episodes"] = int(self.s_avoid_total_completed_episodes)
        self.extras["avoid_stage_completed_episodes"] = int(stage_completed_eps)
        self.extras["avoid_stage_window"] = int(stage_window_size)
        self.extras["avoid_shrink_window"] = int(getattr(self.cfg.terrain, "avoid_stage3_shrink_window", 100))

    def _reset_s_avoid_episode_progress(self, env_ids: torch.Tensor) -> None:
        if (not self.s_avoid_enabled) or env_ids.numel() == 0 or (not hasattr(self, "goal_world")):
            return
        goal_dist = torch.norm(self.goal_world[env_ids] - self.root_states[env_ids, :2], dim=1)
        self.s_avoid_episode_goal_init_dist[env_ids] = goal_dist
        self.s_avoid_episode_goal_best_dist[env_ids] = goal_dist

    def _get_s_avoid_episode_progress_flags(
        self,
        env_ids: torch.Tensor,
        stage_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=torch.bool)
        if stage_ids is None:
            stage_ids = self.s_avoid_stage_per_env[env_ids]
        init_dist = self.s_avoid_episode_goal_init_dist[env_ids]
        best_dist = self.s_avoid_episode_goal_best_dist[env_ids]
        progress = torch.clamp(init_dist - best_dist, min=0.0)
        progress_delta12 = float(getattr(self.cfg.terrain, "avoid_stage12_progress_delta", 0.25))
        progress_delta23 = float(getattr(self.cfg.terrain, "avoid_stage23_progress_delta", progress_delta12))
        progress_delta34 = float(getattr(self.cfg.terrain, "avoid_stage34_progress_delta", progress_delta23))
        thresholds = torch.full_like(progress, progress_delta34)
        thresholds = torch.where(
            stage_ids.to(device=self.device) == 1,
            torch.full_like(progress, progress_delta12),
            thresholds,
        )
        thresholds = torch.where(
            stage_ids.to(device=self.device) == 2,
            torch.full_like(progress, progress_delta23),
            thresholds,
        )
        return progress >= thresholds

    def _get_s_avoid_episode_success_flags(
        self,
        env_ids: torch.Tensor,
        stage_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=torch.bool)
        if stage_ids is None:
            stage_ids = self.s_avoid_stage_per_env[env_ids]
        best_dist = self.s_avoid_episode_goal_best_dist[env_ids]
        success_dist12 = float(getattr(self.cfg.terrain, "avoid_stage12_success_distance", 0.80))
        success_dist23 = float(getattr(self.cfg.terrain, "avoid_stage23_success_distance", success_dist12))
        success_dist34 = float(getattr(self.cfg.terrain, "avoid_stage34_success_distance", success_dist23))
        thresholds = torch.full_like(best_dist, success_dist34)
        thresholds = torch.where(
            stage_ids.to(device=self.device) == 1,
            torch.full_like(best_dist, success_dist12),
            thresholds,
        )
        thresholds = torch.where(
            stage_ids.to(device=self.device) == 2,
            torch.full_like(best_dist, success_dist23),
            thresholds,
        )
        return best_dist <= thresholds

    def _resolve_s_avoid_stage(self, env_id: int, episode_idx: Optional[int] = None) -> int:
        debug_case = str(getattr(self.cfg.terrain, "avoid_map_debug_case", "")).strip().lower()
        if debug_case:
            return 1
        preview_all = bool(getattr(self.cfg.terrain, "avoid_preview_all_stages", False))
        if preview_all:
            if self.num_envs >= 4:
                return int(env_id % 4) + 1
            if episode_idx is None:
                episode_idx = int(self.s_avoid_env_episode_count[env_id].item())
            return int(episode_idx % 4) + 1
        return int(self.s_avoid_stage)

    def _reset_s_avoid_robot_pose(self, env_ids: torch.Tensor):
        if not self.s_avoid_enabled or env_ids.numel() == 0:
            return

        wall_l = float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))
        stage3_spawn_y = -0.5 * wall_l + 0.8
        stage12_spawn_y = -1.6
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        stage_vals = self.s_avoid_stage_per_env[env_ids]
        stage3_mask = stage_vals == 4
        if stage3_mask.any():
            stage3_ids = env_ids[stage3_mask]
            self.root_states[stage3_ids, 1] += stage3_spawn_y
        stage12_mask = ~stage3_mask
        if stage12_mask.any():
            stage12_ids = env_ids[stage12_mask]
            self.root_states[stage12_ids, 1] += stage12_spawn_y
        self.s_avoid_spawn_world_y[env_ids] = self.root_states[env_ids, 1]

        body_plus_y_target_deg = float(getattr(self.cfg.terrain, "avoid_spawn_body_plus_y_deg", 0.0))
        body_plus_y_target_rad = math.radians(body_plus_y_target_deg)
        yaw = torch.full(
            (len(env_ids),),
            -body_plus_y_target_rad,
            device=self.device,
            dtype=torch.float32,
        )
        qz = torch.sin(0.5 * yaw)
        qw = torch.cos(0.5 * yaw)
        quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
        self.root_states[env_ids, 3:7] = quat
        self._sync_robot_root_states(env_ids)
        debug_case = str(getattr(self.cfg.terrain, "avoid_map_debug_case", "")).strip().lower()
        if debug_case and env_ids.numel() > 0:
            env0 = int(env_ids[0].item())
            active = self.s_avoid_active[env0]
            if bool(active.any().item()):
                slot = int(torch.nonzero(active, as_tuple=False)[0, 0].item())
                robot_xy = self.root_states[env0, :2]
                obs_xy = self.s_avoid_pos_world[env0, slot, :2]
                delta = obs_xy - robot_xy
                cos_h = math.cos(body_plus_y_target_rad)
                sin_h = math.sin(body_plus_y_target_rad)
                local_x = float(cos_h * delta[0].item() + sin_h * delta[1].item())
                local_y = float(-sin_h * delta[0].item() + cos_h * delta[1].item())
                ok = False
                if debug_case == "front":
                    ok = abs(local_x) < 0.15 and local_y > 0.8
                elif debug_case == "left":
                    ok = local_x < -0.25 and local_y > 0.8
                elif debug_case == "right":
                    ok = local_x > 0.25 and local_y > 0.8
                elif debug_case == "side_left":
                    ok = local_x < -0.8 and abs(local_y) < 0.4
                elif debug_case == "side_right":
                    ok = local_x > 0.8 and abs(local_y) < 0.4
                print(
                    f"[AvoidMapDebug] case={debug_case} env={env0} "
                    f"target_body+y_vs_world+Y_deg={body_plus_y_target_deg:.1f} "
                    f"set_root_yaw_deg={float(math.degrees(-body_plus_y_target_rad)):.1f} "
                    f"robot_xy=({float(robot_xy[0].item()):.3f},{float(robot_xy[1].item()):.3f}) "
                    f"obs_xy=({float(obs_xy[0].item()):.3f},{float(obs_xy[1].item()):.3f}) "
                    f"local_xy(x_right,y_forward)=({local_x:.3f},{local_y:.3f}) "
                    f"ok={int(ok)}"
                )

    def _resample_s_avoid_goals(self, env_ids: torch.Tensor):
        if not self.s_avoid_enabled or env_ids.numel() == 0:
            return

        wall_l = float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))
        goal_stage3_y = 0.5 * wall_l - 0.8
        goal_x_min = float(getattr(self.cfg.navigation, "goal_range_x", [-1.6, 1.6])[0])
        goal_x_max = float(getattr(self.cfg.navigation, "goal_range_x", [-1.6, 1.6])[1])
        goal_y_min = float(getattr(self.cfg.navigation, "goal_range_y", [1.2, 2.8])[0])
        goal_y_max = float(getattr(self.cfg.navigation, "goal_range_y", [1.2, 2.8])[1])
        goal_max_tries = int(getattr(self.cfg.navigation, "goal_sample_max_tries", 32))
        goal_clearance = float(getattr(self.cfg.terrain, "avoid_spawn_clearance", 0.5))
        cap_slots = int(getattr(self.cfg.terrain, "avoid_capsule_slots", 0))
        box_slots = int(getattr(self.cfg.terrain, "avoid_box_slots", 0))
        cap_r = float(getattr(self.cfg.terrain, "avoid_capsule_radius", 0.15))
        box_hx = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_x", 0.4))
        box_hy = 0.5 * float(getattr(self.cfg.terrain, "avoid_box_size_y", 0.4))
        box_r = math.sqrt(box_hx * box_hx + box_hy * box_hy)
        slot_radii = torch.zeros(self.s_avoid_total_slots, device=self.device, dtype=torch.float32)
        if cap_slots > 0:
            slot_radii[:cap_slots] = cap_r
        if box_slots > 0:
            slot_radii[cap_slots:cap_slots + box_slots] = box_r

        goal_local = torch.zeros((len(env_ids), 2), device=self.device, dtype=torch.float32)
        stage_vals = self.s_avoid_stage_per_env[env_ids]

        stage3_mask = stage_vals == 4
        if stage3_mask.any():
            goal_local[stage3_mask, 1] = goal_stage3_y

        stage12_mask = ~stage3_mask
        if stage12_mask.any():
            stage12_rows = torch.nonzero(stage12_mask, as_tuple=False).flatten()
            stage12_ids = env_ids[stage12_mask]
            for row_idx, env_id in zip(stage12_rows.tolist(), stage12_ids.tolist()):
                origin_xy = self.env_origins[env_id, :2]
                active = self.s_avoid_active[env_id]
                obstacle_pos = self.s_avoid_pos_world[env_id, :, :2]
                valid_goal = None
                fallback_goal = None
                for _ in range(goal_max_tries):
                    candidate_local = torch.tensor(
                        [
                            float(np.random.uniform(goal_x_min, goal_x_max)),
                            float(np.random.uniform(goal_y_min, goal_y_max)),
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    fallback_goal = candidate_local
                    candidate_world = origin_xy + candidate_local
                    if bool(active.any().item()):
                        active_pos = obstacle_pos[active]
                        active_r = slot_radii[active]
                        dist = torch.norm(active_pos - candidate_world.unsqueeze(0), dim=1)
                        if bool((dist < (active_r + goal_clearance)).any().item()):
                            continue
                    valid_goal = candidate_local
                    break
                if valid_goal is None:
                    valid_goal = fallback_goal if fallback_goal is not None else torch.tensor(
                        [0.0, max(goal_y_min, 1.6)],
                        device=self.device,
                        dtype=torch.float32,
                    )
                goal_local[row_idx] = valid_goal

        self.goal_world[env_ids] = self.env_origins[env_ids, :2] + goal_local

    def _sample_s_avoid_points(
        self,
        *,
        rng: np.random.RandomState,
        count: int,
        half_extent: float,
        min_spacing: float,
        spawn_clearance: float,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        avoid_zones: Optional[List[Tuple[float, float, float]]] = None,
        existing_points: Optional[List[Tuple[float, float]]] = None,
        max_tries: int = 600,
    ):
        if count <= 0:
            return []
        points = list(existing_points or [])
        if len(points) >= count:
            return points[:count]
        if x_range is None:
            x_range = (-half_extent, half_extent)
        if y_range is None:
            y_range = (-half_extent, half_extent)
        avoid_zones = list(avoid_zones or [])
        spacing = float(min_spacing)
        for _ in range(4):
            points = list(existing_points or [])
            for _try in range(max_tries):
                if len(points) >= count:
                    break
                x = rng.uniform(float(x_range[0]), float(x_range[1]))
                y = rng.uniform(float(y_range[0]), float(y_range[1]))
                if (x * x + y * y) ** 0.5 < spawn_clearance:
                    continue
                blocked = False
                for cx, cy, cr in avoid_zones:
                    dx = x - float(cx)
                    dy = y - float(cy)
                    if (dx * dx + dy * dy) ** 0.5 < float(cr):
                        blocked = True
                        break
                if blocked:
                    continue
                ok = True
                for px, py in points:
                    dx = x - px
                    dy = y - py
                    if (dx * dx + dy * dy) ** 0.5 < spacing:
                        ok = False
                        break
                if ok:
                    points.append((x, y))
            if len(points) >= count:
                return points[:count]
            spacing = max(0.2, spacing * 0.85)

        while len(points) < count:
            x = rng.uniform(float(x_range[0]), float(x_range[1]))
            y = rng.uniform(float(y_range[0]), float(y_range[1]))
            if (x * x + y * y) ** 0.5 < spawn_clearance:
                continue
            blocked = False
            for cx, cy, cr in avoid_zones:
                dx = x - float(cx)
                dy = y - float(cy)
                if (dx * dx + dy * dy) ** 0.5 < float(cr):
                    blocked = True
                    break
            if blocked:
                continue
            points.append((x, y))
        return points[:count]

    def _reset_s_avoid_obstacles(self, env_ids: torch.Tensor):
        if not self.s_avoid_enabled or env_ids.numel() == 0 or self.s_avoid_actor_indices is None:
            return

        total_slots = int(self.s_avoid_total_slots)
        cap_slots = int(self.s_avoid_capsule_slot_count)
        box_slots = int(self.s_avoid_box_slot_count)
        wall_slots = int(self.s_avoid_wall_slot_count)

        cap_h = float(getattr(self.cfg.terrain, "avoid_capsule_height", 0.5))
        wall_h = float(getattr(self.cfg.terrain, "avoid_wall_height", 0.5))
        wall_t = float(getattr(self.cfg.terrain, "avoid_wall_thickness", 0.12))
        wall_l = float(getattr(self.cfg.terrain, "avoid_wall_length", 6.0))
        seed0 = int(getattr(self.cfg.terrain, "avoid_seed", 7001))

        cap_z = 0.5 * cap_h
        wall_z = 0.5 * wall_h
        stage12_spawn_y = -1.6
        debug_case = self._get_s_avoid_debug_case()

        if self.s_avoid_direct_single_obstacle:
            local_x, local_y = self._get_s_avoid_debug_local_pose()
            flat_indices = self.s_avoid_actor_indices[env_ids].reshape(-1).contiguous()
            flat_indices_long = flat_indices.to(torch.long)
            root_states = getattr(self, "all_root_states", None)
            for env_id in env_ids.tolist():
                self.s_avoid_env_episode_count[env_id] += 1
                active = np.zeros((total_slots,), dtype=np.bool_)
                pos = np.zeros((total_slots, 3), dtype=np.float32)
                pos[:, 2] = -5.0
                quat = np.zeros((total_slots, 4), dtype=np.float32)
                quat[:, 3] = 1.0
                active[0] = True
                pos[0] = np.array([local_x, local_y, cap_z], dtype=np.float32)
                quat[0] = np.array(
                    [
                        self.s_avoid_capsule_quat.x,
                        self.s_avoid_capsule_quat.y,
                        self.s_avoid_capsule_quat.z,
                        self.s_avoid_capsule_quat.w,
                    ],
                    dtype=np.float32,
                )
                env_origin = self.env_origins[env_id, :3].detach().cpu().numpy()
                pos_world = pos.copy()
                pos_world[:, 0] += env_origin[0]
                pos_world[:, 1] += env_origin[1]
                pos_world[:, 2] += env_origin[2]
                band_x_min, band_x_max, band_y_min, band_y_max = self._compute_s_avoid_band_world(
                    env_id=env_id,
                    stage=1,
                    active=active,
                    pos_local=pos,
                    quat=quat,
                )
                self.s_avoid_active[env_id] = torch.from_numpy(active).to(device=self.device)
                self.s_avoid_pos_world[env_id] = torch.from_numpy(pos_world).to(device=self.device)
                self.s_avoid_quat_world[env_id] = torch.from_numpy(quat).to(device=self.device)
                self.s_avoid_band_x_min[env_id] = band_x_min
                self.s_avoid_band_x_max[env_id] = band_x_max
                self.s_avoid_band_y_min[env_id] = band_y_min
                self.s_avoid_band_y_max[env_id] = band_y_max
            if root_states is not None and flat_indices.numel() > 0:
                pos_flat = self.s_avoid_pos_world[env_ids].reshape(-1, 3)
                quat_flat = self.s_avoid_quat_world[env_ids].reshape(-1, 4)
                root_states[flat_indices_long, :3] = pos_flat
                root_states[flat_indices_long, 3:7] = quat_flat
                root_states[flat_indices_long, 7:13] = 0.0
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(root_states),
                    gymtorch.unwrap_tensor(flat_indices),
                    int(flat_indices.numel()),
                )
            return

        for env_id in env_ids.tolist():
            episode_idx = int(self.s_avoid_env_episode_count[env_id].item())
            self.s_avoid_env_episode_count[env_id] += 1
            stage = self._resolve_s_avoid_stage(env_id, episode_idx=episode_idx)
            self.s_avoid_stage_per_env[env_id] = int(stage)

            active, pos, quat = self._get_s_avoid_stage_template()

            if debug_case:
                if cap_slots > 0:
                    cam_cfg = getattr(getattr(self.cfg, "sensor", None), "depth_camera", None)
                    cam_y = 0.0
                    if cam_cfg is not None and hasattr(cam_cfg, "position") and len(cam_cfg.position) >= 2:
                        cam_y = float(cam_cfg.position[1])
                    local_x = 0.0
                    local_y = 1.35
                    if debug_case == "left":
                        local_x = -0.60
                    elif debug_case == "right":
                        local_x = 0.60
                    elif debug_case == "side_left":
                        local_x = -1.35
                        local_y = cam_y
                    elif debug_case == "side_right":
                        local_x = 1.35
                        local_y = cam_y
                    active[0] = True
                    pos[0] = np.array([local_x, stage12_spawn_y + local_y, cap_z], dtype=np.float32)
                env_origin = self.env_origins[env_id, :3].detach().cpu().numpy()
                pos_world = pos.copy()
                pos_world[:, 0] += env_origin[0]
                pos_world[:, 1] += env_origin[1]
                pos_world[:, 2] += env_origin[2]
                band_x_min, band_x_max, band_y_min, band_y_max = self._compute_s_avoid_band_world(
                    env_id=env_id,
                    stage=stage,
                    active=active,
                    pos_local=pos,
                    quat=quat,
                )
                self.s_avoid_active[env_id] = torch.from_numpy(active).to(device=self.device)
                self.s_avoid_pos_world[env_id] = torch.from_numpy(pos_world).to(device=self.device)
                self.s_avoid_quat_world[env_id] = torch.from_numpy(quat).to(device=self.device)
                self.s_avoid_band_x_min[env_id] = band_x_min
                self.s_avoid_band_x_max[env_id] = band_x_max
                self.s_avoid_band_y_min[env_id] = band_y_min
                self.s_avoid_band_y_max[env_id] = band_y_max
                continue

            if stage in (1, 2, 3):
                stage_presets = self.s_avoid_stage_presets.get(int(stage), [])
                if len(stage_presets) > 0:
                    preset_idx = int((seed0 + env_id * 10007 + episode_idx * 131) % len(stage_presets))
                    preset = stage_presets[preset_idx]
                    active = np.array(preset["active"], copy=True)
                    pos = np.array(preset["pos"], copy=True)
                    quat = np.array(preset["quat"], copy=True)
            else:
                width = float(self.s_avoid_corridor_width)
                left_x = -0.5 * width - 0.5 * wall_t
                right_x = 0.5 * width + 0.5 * wall_t
                left_slot = cap_slots + box_slots
                right_slot = left_slot + 1
                if wall_slots >= 2:
                    active[left_slot] = True
                    active[right_slot] = True
                    pos[left_slot] = np.array([left_x, 0.0, wall_z], dtype=np.float32)
                    pos[right_slot] = np.array([right_x, 0.0, wall_z], dtype=np.float32)

            env_origin = self.env_origins[env_id, :3].detach().cpu().numpy()
            pos_world = pos.copy()
            pos_world[:, 0] += env_origin[0]
            pos_world[:, 1] += env_origin[1]
            pos_world[:, 2] += env_origin[2]
            band_x_min, band_x_max, band_y_min, band_y_max = self._compute_s_avoid_band_world(
                env_id=env_id,
                stage=stage,
                active=active,
                pos_local=pos,
                quat=quat,
            )

            self.s_avoid_active[env_id] = torch.from_numpy(active).to(device=self.device)
            self.s_avoid_pos_world[env_id] = torch.from_numpy(pos_world).to(device=self.device)
            self.s_avoid_quat_world[env_id] = torch.from_numpy(quat).to(device=self.device)
            self.s_avoid_band_x_min[env_id] = band_x_min
            self.s_avoid_band_x_max[env_id] = band_x_max
            self.s_avoid_band_y_min[env_id] = band_y_min
            self.s_avoid_band_y_max[env_id] = band_y_max

        self._sync_s_avoid_obstacles(env_ids)
    def _sync_s_avoid_obstacles(self, env_ids: torch.Tensor):
        if not self.s_avoid_enabled or env_ids.numel() == 0 or self.s_avoid_actor_indices is None:
            return
        slot_n = int(self.s_avoid_total_slots)
        flat_indices = self.s_avoid_actor_indices[env_ids].reshape(-1).contiguous()
        flat_indices_long = flat_indices.to(torch.long)
        pos_flat = self.s_avoid_pos_world[env_ids].reshape(-1, 3)
        quat_flat = self.s_avoid_quat_world[env_ids].reshape(-1, 4)

        root_states = getattr(self, "all_root_states", None)
        if root_states is None:
            return
        root_states[flat_indices_long, :3] = pos_flat
        root_states[flat_indices_long, 3:7] = quat_flat
        root_states[flat_indices_long, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(flat_indices),
            int(flat_indices.numel()),
        )
        self._update_s_avoid_debug_colors(env_ids)

        scene_filter = self._scene_collision_filter()
        for env_id in env_ids.tolist():
            env_active = self.s_avoid_active[env_id]
            handles = self.s_avoid_actor_handles[env_id]
            for slot in range(slot_n):
                filter_v = scene_filter
                self._apply_actor_collision_filter(
                    self.envs[env_id], handles[slot], filter_v, env_id, debug_tag=f"s_avoid_obs_{slot}"
                )
                if getattr(self, "debug_viz", False) and env_id == 0:
                    sync_count = int(getattr(self, "_s_avoid_sync_debug_count", 0))
                    if sync_count < 40:
                        actor_index = int(self.s_avoid_actor_indices[env_id, slot].item())
                        pos = self.s_avoid_pos_world[env_id, slot]
                        print(
                            "[Debug][s_avoid_sync] "
                            f"slot={slot} actor_index={actor_index} active={int(bool(env_active[slot].item()))} "
                            f"target_filter={int(filter_v)} root_xyz=({float(pos[0].item()):.3f},"
                            f"{float(pos[1].item()):.3f},{float(pos[2].item()):.3f})"
                        )
                        self._s_avoid_sync_debug_count = sync_count + 1

    def _sync_robot_root_states(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        root_states = getattr(self, "all_root_states", self.root_states)
        if getattr(self, "robot_actor_indices_long", None) is None:
            env_ids_int32 = env_ids.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )
            return
        actor_ids = self.robot_actor_indices_long[env_ids]
        root_states[actor_ids] = self.root_states[env_ids]
        actor_ids_int32 = actor_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )

    def _reset_scene(self, env_ids: torch.Tensor, force_resample: Optional[torch.Tensor] = None):
        if len(env_ids) == 0:
            return
        if self.scene_specs is None:
            return
        env_id_list = env_ids.tolist()
        for env_id in env_id_list:
            self.scene_episode_count[env_id] += 1
            raw_spec = self._get_scene_spec(env_id)
            if isinstance(raw_spec, SceneSpec):
                scene_spec = raw_spec
            else:
                scene_spec = self._scene_spec_from_meta(raw_spec, env_id)
            self.scene_spec_cache[env_id] = scene_spec
            if scene_spec is not None:
                self.scene_meta[env_id] = scene_spec.to_meta()
        sample_id = int(env_ids[0].item())
        if self.scene_meta[sample_id] is not None:
            self.extras["scene_meta"] = self.scene_meta[sample_id]

    def _update_dynamic_obstacles(self):
        if self.dynamic_actor_indices is None:
            return
        self.scene_dyn_time += self.dt
        self._sync_dynamic_obstacles()

    def _moving_target_enabled(self) -> bool:
        return self.nav_cfg is not None and bool(getattr(self.nav_cfg, "moving_target_enable", False))

    def _moving_target_mode(self) -> str:
        if self.nav_cfg is None:
            return "disabled"
        mode = getattr(self.nav_cfg, "moving_target_mode", None)
        if mode is None:
            return "random_plane"
        mode = str(mode).strip().lower()
        if mode in ("e_l_confilct_script",):
            return "e_l_conflict_script"
        return mode

    def _reset_moving_target_s1(self, env_ids: torch.Tensor):
        """Reset scripted gate-traversal moving target for S1 corridor."""
        if env_ids.numel() == 0:
            return
        if self.scene_spec_cache is None:
            return

        # Fill per-env gate arrays (small loop on reset only).
        max_gates = int(getattr(self, "_s1_gate_max", 4))
        for env_id in env_ids.tolist():
            scene_spec = self.scene_spec_cache[env_id]
            if scene_spec is None or scene_spec.scene_type != "s1_corridor_gate":
                self.s1_gate_count[env_id] = 0
                self.s1_gate_idx[env_id] = 0
                self.s1_path_dir[env_id] = 1.0
                continue
            params = scene_spec.params or {}
            length = float(params.get("corridor_length", self.cfg.terrain.terrain_length))
            width_nom = float(params.get("corridor_width_nom", params.get("corridor_width", self.cfg.terrain.terrain_width)))
            self.s1_corridor_half_len[env_id] = 0.5 * length
            self.s1_corridor_half_w[env_id] = 0.5 * width_nom

            gates = params.get("corridor_gates", []) or []
            if not isinstance(gates, list):
                gates = []
            count = min(len(gates), max_gates)
            self.s1_gate_count[env_id] = int(count)
            if count <= 0:
                self.s1_gate_idx[env_id] = 0
                self.s1_path_dir[env_id] = 1.0
                continue

            # Deterministic, per-env RNG for gate bias. RandomState(seed) is independent of global numpy state.
            seed = int((scene_spec.layout_seed or 0) + env_id * 131)
            rng = np.random.RandomState(seed)

            # Difficulty in [0,1] from curriculum (per-env).
            d = self._get_scene_difficulty(env_id)
            bias_base = 0.06 + 0.06 * float(np.clip(d, 0.0, 1.0))  # meters

            for j in range(max_gates):
                if j < count:
                    g = gates[j]
                    y0 = float(g.get("y0", 0.0))
                    glen = float(g.get("length", 1.0))
                    door_w = float(g.get("door_width", width_nom))
                    door_half = 0.5 * min(width_nom, door_w)

                    # Small lateral bias inside the doorway opening.
                    margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.25))
                    bias_max = max(0.0, min(bias_base, door_half - margin - 0.02))
                    bias = float(rng.uniform(-bias_max, bias_max)) if bias_max > 0 else 0.0

                    self.s1_gate_y[env_id, j] = y0
                    self.s1_gate_len[env_id, j] = glen
                    self.s1_gate_door_half[env_id, j] = door_half
                    self.s1_gate_bias_x[env_id, j] = bias
                    self.s1_gate_post_side[env_id, j] = float(rng.choice([-1.0, 1.0]))
                else:
                    self.s1_gate_y[env_id, j] = 0.0
                    self.s1_gate_len[env_id, j] = 0.0
                    self.s1_gate_door_half[env_id, j] = 0.0
                    self.s1_gate_bias_x[env_id, j] = 0.0
                    self.s1_gate_post_side[env_id, j] = 1.0

            self.s1_gate_idx[env_id] = 0
            self.s1_path_dir[env_id] = 1.0

        # Initialize target position in front of robot, clamped to corridor bounds.
        robot_local = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
        desired = float(getattr(self.nav_cfg, "follow_distance_desired", 1.0))
        target_local = robot_local.clone()
        target_local[:, 1] += desired

        margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.25))
        half_len = self.s1_corridor_half_len[env_ids]
        half_w = self.s1_corridor_half_w[env_ids]
        x_min = -half_w + margin
        x_max = half_w - margin
        y_min = -half_len + margin
        y_max = half_len - margin
        target_local[:, 0] = torch.clamp(target_local[:, 0], x_min, x_max)
        target_local[:, 1] = torch.clamp(target_local[:, 1], y_min, y_max)

        self.target_world[env_ids] = self.env_origins[env_ids, :2] + target_local
        self.target_vel_world[env_ids].zero_()
        self.goal_world[env_ids] = self.target_world[env_ids]

    def _update_moving_target_s1(self, dt: float, d: torch.Tensor):
        """
        Scripted S1 gate traversal.

        Target moves along +Y, aligns to a slightly biased doorway center in advance,
        passes the gate, and then performs a small lateral move after the gate.
        """
        # If gates are not available, do nothing.
        gate_count = self.s1_gate_count
        active = gate_count > 0
        if not active.any():
            return

        pos_local = self.target_world - self.env_origins[:, :2]
        x = pos_local[:, 0]
        y = pos_local[:, 1]

        dir_y = self.s1_path_dir  # +1 or -1
        dir_y = torch.where(active, dir_y, torch.ones_like(dir_y))

        # Gather current gate parameters.
        idx = torch.clamp(self.s1_gate_idx, 0, self.s1_gate_y.shape[1] - 1).to(torch.long)
        gather_idx = idx.view(-1, 1)
        y_gate = torch.gather(self.s1_gate_y, 1, gather_idx).squeeze(1)
        gate_len = torch.gather(self.s1_gate_len, 1, gather_idx).squeeze(1).clamp(min=0.1)
        x_bias = torch.gather(self.s1_gate_bias_x, 1, gather_idx).squeeze(1)
        post_side = torch.gather(self.s1_gate_post_side, 1, gather_idx).squeeze(1)

        y_f = dir_y * y
        gate_y_f = dir_y * y_gate
        half_len = 0.5 * gate_len

        align_dist = 1.0 + 0.6 * d
        post_dist = 0.8 + 0.4 * d
        align_start = gate_y_f - half_len - align_dist
        pass_start = gate_y_f - half_len
        pass_end = gate_y_f + half_len
        post_end = gate_y_f + half_len + post_dist

        # Stage masks in forward coordinate.
        m_approach = y_f < align_start
        m_align = (y_f >= align_start) & (y_f < pass_start)
        m_pass = (y_f >= pass_start) & (y_f <= pass_end)
        m_post = (y_f > pass_end) & (y_f <= post_end)

        x_center = torch.zeros_like(x)
        x_post_amp = 0.06 + 0.06 * d
        x_post = post_side * x_post_amp
        x_des = torch.where(m_align | m_pass, x_bias, torch.where(m_post, x_post, x_center))

        v_max = float(getattr(self.nav_cfg, "moving_target_v_max", 1.2))
        v_typ = float(getattr(self.nav_cfg, "moving_target_v_typical", 0.6))
        v_base = torch.clamp(v_typ + (v_max - v_typ) * (0.2 + 0.8 * d), 0.2, v_max)
        v_align = 0.85 * v_base
        v_pass = 0.75 * v_base
        vy_mag = torch.where(m_align, v_align, torch.where(m_pass, v_pass, v_base))
        vy = dir_y * vy_mag

        kp_x = 2.0 + 1.0 * d
        v_lat_max = 0.22 + 0.10 * d
        vx = torch.clamp(kp_x * (x_des - x), -v_lat_max, v_lat_max)

        vel_local = torch.stack([vx, vy], dim=-1)
        pos_local = pos_local + vel_local * float(dt)

        # Corridor boundary clamp (account for narrower doorways).
        margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.25))
        half_nom = self.s1_corridor_half_w
        inside = torch.abs(pos_local[:, 1].unsqueeze(1) - self.s1_gate_y) <= 0.5 * self.s1_gate_len.clamp(min=0.0)
        door_half = self.s1_gate_door_half
        # Prefer the current gate's doorway constraint to avoid being overly conservative.
        # Edge-case fallback: if we are inside any gate (but not the current one), use the most
        # conservative doorway width (min over gates) to avoid passing through blocked regions.
        ar = torch.arange(self.s1_gate_y.shape[1], device=self.device).view(1, -1)
        curr_mask = (idx.view(-1, 1) == ar)
        inside_curr = inside & curr_mask
        in_curr_gate = inside_curr.any(dim=1)
        curr_half = torch.gather(door_half, 1, idx.view(-1, 1)).squeeze(1)
        half_gate = torch.where(inside, door_half, torch.full_like(door_half, 1e9))
        half_min = half_gate.min(dim=1).values
        in_any_gate = inside.any(dim=1)
        half_w = torch.where(
            in_curr_gate,
            torch.minimum(half_nom, curr_half),
            torch.where(in_any_gate, torch.minimum(half_nom, half_min), half_nom),
        )
        half_w = torch.clamp(half_w, min=margin + 0.05)

        pos_local[:, 0] = torch.clamp(pos_local[:, 0], -half_w + margin, half_w - margin)

        # Y boundary: clamp and flip direction at ends.
        half_len_corr = self.s1_corridor_half_len
        y_min = -half_len_corr + margin
        y_max = half_len_corr - margin
        hit_hi = pos_local[:, 1] > y_max
        hit_lo = pos_local[:, 1] < y_min
        if hit_hi.any() or hit_lo.any():
            pos_local[:, 1] = torch.clamp(pos_local[:, 1], y_min, y_max)
            dir_y = torch.where(hit_hi, -torch.ones_like(dir_y), dir_y)
            dir_y = torch.where(hit_lo, torch.ones_like(dir_y), dir_y)
            # Reset to the nearest end gate when flipping.
            last_idx = torch.clamp(gate_count - 1, min=0).to(torch.long)
            idx = torch.where(hit_hi, last_idx, idx)
            idx = torch.where(hit_lo, torch.zeros_like(idx), idx)

        # Advance gate index after the post segment, if another gate exists in current direction.
        advance = y_f > post_end
        dir_int = dir_y.to(torch.long)
        next_idx = idx + dir_int
        valid_next = (next_idx >= 0) & (next_idx < gate_count)
        idx = torch.where(advance & valid_next, next_idx, idx)

        # Commit state.
        self.s1_path_dir = torch.where(active, dir_y, self.s1_path_dir)
        self.s1_gate_idx = torch.where(active, idx, self.s1_gate_idx)
        self.target_world = self.env_origins[:, :2] + pos_local
        self.target_vel_world = vel_local
        self.goal_world[:] = self.target_world

    def _reset_moving_target_s0_circle(self, env_ids: torch.Tensor):
        """Reset scripted S0 right-turn circular target trajectory."""
        if env_ids.numel() == 0:
            return
        radius = float(getattr(self.nav_cfg, "moving_target_circle_radius", 1.2))
        speed = float(getattr(self.nav_cfg, "moving_target_circle_speed", 0.28))
        center_x = float(getattr(self.nav_cfg, "moving_target_circle_center_x", radius))
        center_y = float(getattr(self.nav_cfg, "moving_target_circle_center_y", 1.0))
        phase_deg = float(getattr(self.nav_cfg, "moving_target_circle_start_phase_deg", 180.0))
        clockwise = bool(getattr(self.nav_cfg, "moving_target_circle_clockwise", True))
        phase0 = math.radians(phase_deg)
        theta_dot = (-1.0 if clockwise else 1.0) * (speed / max(radius, 1e-6))

        theta = torch.full((len(env_ids),), phase0, device=self.device, dtype=torch.float32)
        x_local = center_x + radius * torch.cos(theta)
        y_local = center_y + radius * torch.sin(theta)
        target_local = torch.stack([x_local, y_local], dim=1)

        vel_x = theta_dot * (-radius * torch.sin(theta))
        vel_y = theta_dot * (radius * torch.cos(theta))
        vel_local = torch.stack([vel_x, vel_y], dim=1)
        heading = torch.atan2(vel_x, vel_y)

        self.target_speed_phase[env_ids] = theta
        self.target_heading[env_ids] = heading
        self.target_heading_des[env_ids] = heading
        self.target_speed[env_ids] = float(speed)
        self.target_speed_des[env_ids] = float(speed)
        self.target_cmd_timer[env_ids] = 0.0
        freeze_s = float(getattr(self.nav_cfg, "moving_target_freeze_s", 0.0))
        self.target_freeze_timer[env_ids] = freeze_s
        self.target_turn_events[env_ids] = 0.0
        self.target_preturn_events[env_ids] = 0.0
        self.target_reflect_events[env_ids] = 0.0

        self.target_world[env_ids] = self.env_origins[env_ids, :2] + target_local
        self.target_vel_world[env_ids] = vel_local
        self.goal_world[env_ids] = self.target_world[env_ids]

    def _update_moving_target_s0_circle(self, dt: float):
        """Update scripted S0 right-turn circular target trajectory."""
        radius = float(getattr(self.nav_cfg, "moving_target_circle_radius", 1.2))
        speed = float(getattr(self.nav_cfg, "moving_target_circle_speed", 0.28))
        center_x = float(getattr(self.nav_cfg, "moving_target_circle_center_x", radius))
        center_y = float(getattr(self.nav_cfg, "moving_target_circle_center_y", 1.0))
        clockwise = bool(getattr(self.nav_cfg, "moving_target_circle_clockwise", True))
        theta_dot = (-1.0 if clockwise else 1.0) * (speed / max(radius, 1e-6))

        self.target_turn_events.zero_()
        self.target_preturn_events.zero_()
        self.target_reflect_events.zero_()

        active_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if hasattr(self, "target_freeze_timer"):
            frozen = self.target_freeze_timer > 0.0
            if frozen.any():
                self.target_freeze_timer = torch.clamp(self.target_freeze_timer - dt, min=0.0)
                self.target_vel_world[frozen].zero_()
                self.goal_world[frozen] = self.target_world[frozen]
                active_mask = ~frozen
                if not active_mask.any():
                    return

        theta = self.target_speed_phase
        theta_next = torch.atan2(torch.sin(theta + theta_dot * dt), torch.cos(theta + theta_dot * dt))
        theta = torch.where(active_mask, theta_next, theta)
        self.target_speed_phase = theta

        x_local = center_x + radius * torch.cos(theta)
        y_local = center_y + radius * torch.sin(theta)
        target_local = torch.stack([x_local, y_local], dim=1)

        vel_x = theta_dot * (-radius * torch.sin(theta))
        vel_y = theta_dot * (radius * torch.cos(theta))
        vel_local = torch.stack([vel_x, vel_y], dim=1)
        vel_local = torch.where(active_mask.unsqueeze(1), vel_local, torch.zeros_like(vel_local))
        heading = torch.atan2(vel_x, vel_y)
        heading = torch.where(active_mask, heading, self.target_heading)

        self.target_heading = heading
        self.target_heading_des = heading
        self.target_speed = torch.where(active_mask, torch.full_like(self.target_speed, float(speed)), self.target_speed)
        self.target_speed_des = self.target_speed.clone()
        self.target_world = self.env_origins[:, :2] + target_local
        self.target_vel_world = vel_local
        self.goal_world[:] = self.target_world

    def _ensure_e_l_conflict_buffers(self):
        if hasattr(self, "target_lturn_stage"):
            return
        self.target_lturn_stage = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long, requires_grad=False
        )
        self.target_lturn_hold_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float, requires_grad=False
        )
        self.target_lturn_theta = torch.full(
            (self.num_envs,), math.pi, device=self.device, dtype=torch.float, requires_grad=False
        )

    def _get_e_s_corridor_geometry(self):
        geom = dict(terrain_get_e_s_corridor_geom(self.cfg.terrain))
        segment_length = float(getattr(self.cfg.terrain, "e_s_corridor_segment_length", 0.55))
        segment_overlap = float(getattr(self.cfg.terrain, "e_s_corridor_segment_overlap", 0.80))
        speed = float(getattr(self.nav_cfg, "moving_target_s_corridor_speed", 0.50))
        spawn_gap = float(getattr(self.nav_cfg, "moving_target_s_corridor_spawn_gap", 1.00))
        path_samples = int(getattr(self.nav_cfg, "moving_target_s_corridor_path_samples", 801))

        segment_length = max(0.2, segment_length)
        segment_overlap = float(np.clip(segment_overlap, 0.5, 0.95))
        speed = max(0.05, speed)
        spawn_gap = max(0.2, spawn_gap)
        path_samples = max(201, path_samples)

        geom["segment_length"] = float(segment_length)
        geom["segment_overlap"] = float(segment_overlap)
        geom["speed"] = float(speed)
        geom["spawn_gap"] = float(spawn_gap)
        geom["path_samples"] = int(path_samples)
        return geom

    def _build_e_s_corridor_cache(self):
        cache = getattr(self, "_e_s_corridor_cache", None)
        if cache is not None:
            return cache

        geom = self._get_e_s_corridor_geometry()
        segment_length = float(geom["segment_length"])
        segment_overlap = float(geom["segment_overlap"])
        corridor_half = 0.5 * float(geom["corridor_width"])
        wall_h = float(geom["wall_height"])
        wall_z = 0.5 * wall_h
        direct_mesh_meta = getattr(getattr(self, "terrain", None), "direct_mesh_meta", None)
        if isinstance(direct_mesh_meta, dict) and ("centerline_local" in direct_mesh_meta):
            pos = np.asarray(direct_mesh_meta["centerline_local"], dtype=np.float64)
            tan = np.asarray(direct_mesh_meta["tangent_local"], dtype=np.float64)
            arc_s = np.asarray(direct_mesh_meta["arc_s_local"], dtype=np.float64)
        else:
            _, pos, tan, arc_s = terrain_build_e_s_corridor_centerline(
                self.cfg.terrain,
                num_samples=int(geom["path_samples"]),
            )

        spacing = max(0.15, segment_length * segment_overlap)
        sample_s = np.arange(0.0, arc_s[-1] + 0.5 * spacing, spacing, dtype=np.float64)
        cx = np.interp(sample_s, arc_s, pos[:, 0])
        cy = np.interp(sample_s, arc_s, pos[:, 1])
        tx = np.interp(sample_s, arc_s, tan[:, 0])
        ty = np.interp(sample_s, arc_s, tan[:, 1])
        tnorm = np.clip(np.sqrt(tx * tx + ty * ty), 1e-8, None)
        tx = tx / tnorm
        ty = ty / tnorm
        nx = ty
        ny = -tx
        wall_poses_local = []
        for side_sign in (-1.0, 1.0):
            bx = cx + side_sign * corridor_half * nx
            by = cy + side_sign * corridor_half * ny
            for i in range(sample_s.shape[0] - 1):
                mx = 0.5 * (bx[i] + bx[i + 1])
                my = 0.5 * (by[i] + by[i + 1])
                dx = bx[i + 1] - bx[i]
                dy = by[i + 1] - by[i]
                yaw = math.atan2(-dx, dy)
                wall_poses_local.append((float(mx), float(my), wall_z, float(yaw)))

        cache = {
            "geom": geom,
            "arc_s": arc_s,
            "pos": pos,
            "tan": tan,
            "wall_poses_local": wall_poses_local,
            "segments_per_side": int(max(0, sample_s.shape[0] - 1)),
            "path_length": float(arc_s[-1]),
        }
        self._e_s_corridor_cache = cache
        return cache

    def _ensure_e_s_corridor_tensors(self):
        if self.e_s_corridor_path_s_tensor is not None:
            return
        cache = self._build_e_s_corridor_cache()
        self.e_s_corridor_path_s_tensor = torch.tensor(
            cache["arc_s"], device=self.device, dtype=torch.float32
        )
        self.e_s_corridor_path_pos_tensor = torch.tensor(
            cache["pos"], device=self.device, dtype=torch.float32
        )
        self.e_s_corridor_path_tan_tensor = torch.tensor(
            cache["tan"], device=self.device, dtype=torch.float32
        )
        self.e_s_corridor_path_length = float(cache["path_length"])

    def _ensure_e_s_corridor_buffers(self):
        if hasattr(self, "target_s_corridor_progress"):
            return
        self.target_s_corridor_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
        )

    def _interp_e_s_corridor_path(self, progress: torch.Tensor):
        self._ensure_e_s_corridor_tensors()
        loop = bool(getattr(self.nav_cfg, "moving_target_s_corridor_loop", True))
        total = max(float(self.e_s_corridor_path_length), 1e-6)
        if loop:
            progress = torch.remainder(progress, total)
        else:
            progress = torch.clamp(progress, 0.0, total)

        s = self.e_s_corridor_path_s_tensor
        pos = self.e_s_corridor_path_pos_tensor
        tan = self.e_s_corridor_path_tan_tensor

        idx1 = torch.searchsorted(s, progress, right=True)
        idx1 = torch.clamp(idx1, 1, s.shape[0] - 1)
        idx0 = idx1 - 1
        s0 = s[idx0]
        s1 = s[idx1]
        w = ((progress - s0) / (s1 - s0).clamp_min(1e-6)).unsqueeze(1)

        pos_i = pos[idx0] * (1.0 - w) + pos[idx1] * w
        tan_i = tan[idx0] * (1.0 - w) + tan[idx1] * w
        tan_i = tan_i / tan_i.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return pos_i, tan_i, progress

    def _reset_moving_target_e_s_corridor(self, env_ids: torch.Tensor):
        """Reset scripted moving target for e_S_corridor."""
        if env_ids.numel() == 0:
            return
        self._ensure_e_s_corridor_buffers()
        speed = float(getattr(self.nav_cfg, "moving_target_s_corridor_speed", 0.50))

        progress = torch.zeros(len(env_ids), device=self.device, dtype=torch.float32)
        target_local, tan_local, progress = self._interp_e_s_corridor_path(progress)
        vel_local = tan_local * speed
        heading = torch.atan2(tan_local[:, 0], tan_local[:, 1])

        self.target_s_corridor_progress[env_ids] = progress
        self.target_heading[env_ids] = heading
        self.target_heading_des[env_ids] = heading
        self.target_speed[env_ids] = speed
        self.target_speed_des[env_ids] = speed
        self.target_cmd_timer[env_ids] = 0.0
        self.target_speed_phase[env_ids] = 0.0
        self.target_freeze_timer[env_ids] = 0.0
        self.target_turn_events[env_ids] = 0.0
        self.target_preturn_events[env_ids] = 0.0
        self.target_reflect_events[env_ids] = 0.0

        self.target_world[env_ids] = self.env_origins[env_ids, :2] + target_local
        self.target_vel_world[env_ids] = vel_local
        self.goal_world[env_ids] = self.target_world[env_ids]

    def _update_moving_target_e_s_corridor(self, dt: float):
        """Update scripted S-corridor target along centerline with approximately constant speed."""
        self._ensure_e_s_corridor_buffers()
        speed = float(getattr(self.nav_cfg, "moving_target_s_corridor_speed", 0.50))
        progress = self.target_s_corridor_progress + speed * float(dt)
        target_local, tan_local, progress = self._interp_e_s_corridor_path(progress)
        vel_local = tan_local * speed
        heading = torch.atan2(tan_local[:, 0], tan_local[:, 1])

        self.target_s_corridor_progress[:] = progress
        self.target_heading[:] = heading
        self.target_heading_des[:] = heading
        self.target_speed[:] = speed
        self.target_speed_des[:] = speed
        self.target_world[:] = self.env_origins[:, :2] + target_local
        self.target_vel_world[:] = vel_local
        self.goal_world[:] = self.target_world
        self.target_turn_events.zero_()
        self.target_preturn_events.zero_()
        self.target_reflect_events.zero_()

    def _get_e_l_conflict_obstacle_local(self):
        obs_r = float(getattr(self.cfg.terrain, "e_l_conflict_obstacle_radius", 0.15))
        obs_h = float(getattr(self.cfg.terrain, "e_l_conflict_obstacle_height", 0.45))
        use_inner_tangent = bool(getattr(self.cfg.terrain, "e_l_conflict_obstacle_inner_tangent", True))
        if use_inner_tangent:
            corner_x = float(getattr(self.cfg.terrain, "e_l_conflict_corner_x", 0.0))
            corner_y = float(getattr(self.cfg.terrain, "e_l_conflict_corner_y", 4.0))
            obs_x = corner_x + obs_r
            obs_y = corner_y - obs_r
        else:
            obs_x = float(getattr(self.cfg.terrain, "e_l_conflict_obstacle_x", 0.15))
            obs_y = float(getattr(self.cfg.terrain, "e_l_conflict_obstacle_y", 3.85))

        # Keep obstacle inside local tile bounds.
        half_len = 0.5 * float(getattr(self.cfg.terrain, "terrain_length", 12.0))
        half_wid = 0.5 * float(getattr(self.cfg.terrain, "terrain_width", 12.0))
        margin = 0.05
        x_min = -half_wid + obs_r + margin
        x_max = half_wid - obs_r - margin
        y_min = -half_len + obs_r + margin
        y_max = half_len - obs_r - margin
        obs_x = float(np.clip(obs_x, x_min, x_max))
        obs_y = float(np.clip(obs_y, y_min, y_max))
        return obs_x, obs_y, obs_r, obs_h

    def _get_e_l_conflict_turn_path(self):
        obs_x, obs_y, obs_r, _ = self._get_e_l_conflict_obstacle_local()
        start_y_base = float(getattr(self.nav_cfg, "moving_target_lturn_start_y", 1.0))
        straight_extra = float(getattr(self.nav_cfg, "moving_target_lturn_straight_extra", 0.0))
        start_y = start_y_base - max(0.0, straight_extra)
        end_x = float(getattr(self.nav_cfg, "moving_target_lturn_end_x", 3.0))
        speed = float(getattr(self.nav_cfg, "moving_target_lturn_speed", 0.85))
        hold_s = float(getattr(self.nav_cfg, "moving_target_lturn_hold_s", 0.0))
        loop = bool(getattr(self.nav_cfg, "moving_target_lturn_loop", True))
        clear_m = float(getattr(self.nav_cfg, "moving_target_lturn_clearance", 0.05))
        turn_after = float(getattr(self.nav_cfg, "moving_target_lturn_turn_after_obstacle", 0.10))
        spawn_gap = float(getattr(self.nav_cfg, "moving_target_lturn_spawn_gap", 0.5))
        turn_r = max(obs_r + clear_m, obs_r + 1e-3)
        x_line = obs_x - turn_r
        turn_entry_y = obs_y + obs_r + max(0.0, turn_after)

        # Keep path inside tile bounds.
        half_len = 0.5 * float(getattr(self.cfg.terrain, "terrain_length", 12.0))
        half_wid = 0.5 * float(getattr(self.cfg.terrain, "terrain_width", 12.0))
        margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.3))
        x_line = float(np.clip(x_line, -half_wid + margin, half_wid - margin))
        turn_center_y = float(
            np.clip(turn_entry_y, -half_len + margin + turn_r, half_len - margin - turn_r)
        )
        y_line = turn_center_y + turn_r
        end_x = float(np.clip(end_x, x_line + 0.2, half_wid - margin))
        start_y = float(np.clip(start_y, -half_len + margin, turn_center_y - 0.2))

        return {
            "obs_x": obs_x,
            "obs_y": obs_y,
            "obs_r": obs_r,
            "turn_r": turn_r,
            "x_line": x_line,
            "turn_center_y": turn_center_y,
            "turn_entry_y": turn_center_y,
            "y_line": y_line,
            "start_y": start_y,
            "end_x": end_x,
            "speed": speed,
            "hold_s": hold_s,
            "loop": loop,
            "spawn_gap": max(0.0, spawn_gap),
            "straight_len": max(0.0, turn_center_y - start_y),
        }

    def _reset_moving_target_e_l_conflict(self, env_ids: torch.Tensor):
        """Reset scripted L-turn moving target used by e_L_conflict scene."""
        if env_ids.numel() == 0:
            return
        self._ensure_e_l_conflict_buffers()
        geom = self._get_e_l_conflict_turn_path()
        x0 = float(geom["x_line"])
        y0 = float(geom["start_y"])
        speed = float(geom["speed"])

        target_local = torch.zeros((len(env_ids), 2), device=self.device, dtype=torch.float32)
        target_local[:, 0] = x0
        target_local[:, 1] = y0

        vel_local = torch.zeros_like(target_local)
        vel_local[:, 1] = speed  # +Y straight segment

        heading = torch.zeros(len(env_ids), device=self.device, dtype=torch.float32)

        self.target_lturn_stage[env_ids] = 0
        self.target_lturn_hold_timer[env_ids] = 0.0
        self.target_lturn_theta[env_ids] = math.pi

        self.target_heading[env_ids] = heading
        self.target_heading_des[env_ids] = heading
        self.target_speed[env_ids] = speed
        self.target_speed_des[env_ids] = speed
        self.target_cmd_timer[env_ids] = 0.0
        self.target_speed_phase[env_ids] = 0.0
        self.target_freeze_timer[env_ids] = 0.0
        self.target_turn_events[env_ids] = 0.0
        self.target_preturn_events[env_ids] = 0.0
        self.target_reflect_events[env_ids] = 0.0

        self.target_world[env_ids] = self.env_origins[env_ids, :2] + target_local
        self.target_vel_world[env_ids] = vel_local
        self.goal_world[env_ids] = self.target_world[env_ids]

    def _update_moving_target_e_l_conflict(self, dt: float):
        """Update scripted L-turn target: straight -> rounded right turn -> straight."""
        self._ensure_e_l_conflict_buffers()
        geom = self._get_e_l_conflict_turn_path()
        x0 = float(geom["x_line"])
        y0 = float(geom["start_y"])
        end_x = float(geom["end_x"])
        speed = float(geom["speed"])
        hold_s = float(geom["hold_s"])
        loop = bool(geom["loop"])
        obs_x = float(geom["obs_x"])
        turn_center_y = float(geom["turn_center_y"])
        turn_entry_y = float(geom["turn_entry_y"])
        turn_r = float(geom["turn_r"])
        y_line = float(geom["y_line"])

        self.target_turn_events.zero_()
        self.target_preturn_events.zero_()
        self.target_reflect_events.zero_()

        pos_local = self.target_world - self.env_origins[:, :2]
        vel_local = torch.zeros_like(pos_local)
        heading = self.target_heading.clone()
        stage = self.target_lturn_stage

        # Stage 0: move straight along +Y to the arc entry.
        m0 = stage == 0
        if m0.any():
            pos_local[m0, 0] = x0
            pos_local[m0, 1] += speed * dt
            vel_local[m0, 1] = speed
            heading[m0] = 0.0
            reached_entry = m0 & (pos_local[:, 1] >= turn_entry_y)
            if reached_entry.any():
                pos_local[reached_entry, 1] = turn_entry_y
                stage[reached_entry] = 1
                self.target_lturn_theta[reached_entry] = math.pi
                self.target_turn_events[reached_entry] = 1.0

        # Stage 1: quarter-circle right turn around obstacle center.
        m1 = stage == 1
        if m1.any():
            w_arc = speed / max(turn_r, 1e-6)
            theta_curr = self.target_lturn_theta[m1]
            theta_next = torch.clamp(theta_curr - w_arc * dt, min=0.5 * math.pi, max=math.pi)
            self.target_lturn_theta[m1] = theta_next

            pos_local[m1, 0] = obs_x + turn_r * torch.cos(theta_next)
            pos_local[m1, 1] = turn_center_y + turn_r * torch.sin(theta_next)
            vel_local[m1, 0] = speed * torch.sin(theta_next)
            vel_local[m1, 1] = -speed * torch.cos(theta_next)
            heading[m1] = torch.atan2(vel_local[m1, 0], vel_local[m1, 1])

            finished_arc_local = theta_next <= (0.5 * math.pi + 1e-4)
            if finished_arc_local.any():
                finished_idx = torch.nonzero(m1, as_tuple=False).flatten()[finished_arc_local]
                pos_local[finished_idx, 0] = obs_x
                pos_local[finished_idx, 1] = y_line
                vel_local[finished_idx, 0] = speed
                vel_local[finished_idx, 1] = 0.0
                heading[finished_idx] = 0.5 * math.pi
                stage[finished_idx] = 2

        # Stage 2: move straight along +X.
        m2 = stage == 2
        if m2.any():
            pos_local[m2, 1] = y_line
            pos_local[m2, 0] += speed * dt
            vel_local[m2, 0] = speed
            heading[m2] = 0.5 * math.pi
            reached_end = m2 & (pos_local[:, 0] >= end_x)
            if reached_end.any():
                pos_local[reached_end, 0] = end_x
                vel_local[reached_end].zero_()
                self.target_lturn_hold_timer[reached_end] = max(0.0, hold_s)
                stage[reached_end] = 3

        # Stage 3: optional hold and loop restart.
        m3 = stage == 3
        if m3.any():
            heading[m3] = 0.5 * math.pi
            vel_local[m3].zero_()
            if loop:
                self.target_lturn_hold_timer[m3] = torch.clamp(self.target_lturn_hold_timer[m3] - dt, min=0.0)
                restart = m3 & (self.target_lturn_hold_timer <= 0.0)
                if restart.any():
                    pos_local[restart, 0] = x0
                    pos_local[restart, 1] = y0
                    vel_local[restart, 1] = speed
                    heading[restart] = 0.0
                    stage[restart] = 0
                    self.target_lturn_theta[restart] = math.pi

        self.target_lturn_stage = stage
        self.target_heading = heading
        self.target_heading_des = heading
        self.target_speed[:] = speed
        self.target_speed_des[:] = speed
        self.target_world = self.env_origins[:, :2] + pos_local
        self.target_vel_world = vel_local
        self.goal_world[:] = self.target_world

    def _reset_moving_target(self, env_ids: torch.Tensor):
        """Reset moving target state for selected envs."""
        if env_ids.numel() == 0 or not self._moving_target_enabled():
            return
        moving_mode = self._moving_target_mode()
        if moving_mode == "s1_gate_script":
            self._reset_moving_target_s1(env_ids)
            return
        if moving_mode == "s0_circle_right":
            self._reset_moving_target_s0_circle(env_ids)
            return
        if moving_mode == "e_l_conflict_script":
            self._reset_moving_target_e_l_conflict(env_ids)
            return
        if moving_mode == "e_s_corridor_script":
            self._reset_moving_target_e_s_corridor(env_ids)
            return
        # Place target straight ahead of the robot (camera center) at reset.
        robot_local = self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
        desired = float(getattr(self.nav_cfg, "follow_distance_desired", 1.0))

        # Heading from root quat (world +Y forward contract): heading=0 means +Y.
        quat = self.root_states[env_ids, 3:7]
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        fwd = torch.stack([torch.sin(heading), torch.cos(heading)], dim=-1)  # +Y forward when heading=0
        target_local = robot_local + desired * fwd  # no lateral bias

        # Clamp into scene bounds.
        half_len = 0.5 * float(self.cfg.terrain.terrain_length)
        half_wid = 0.5 * float(self.cfg.terrain.terrain_width)
        margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.6))
        x_min = -half_wid + margin
        x_max = half_wid - margin
        y_min = -half_len + margin
        y_max = half_len - margin
        target_local[:, 0] = torch.clamp(target_local[:, 0], x_min, x_max)
        target_local[:, 1] = torch.clamp(target_local[:, 1], y_min, y_max)

        # Initialize heading/speed. Keep target heading consistent with robot spawn yaw.
        self.target_heading[env_ids] = heading
        self.target_heading_des[env_ids] = heading
        v_typ = float(getattr(self.nav_cfg, "moving_target_v_typical", 0.6))
        self.target_speed[env_ids] = v_typ
        self.target_speed_des[env_ids] = v_typ
        self.target_cmd_timer[env_ids] = 0.0
        self.target_speed_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * math.pi)
        freeze_s = float(getattr(self.nav_cfg, "moving_target_freeze_s", 0.0))
        self.target_freeze_timer[env_ids] = freeze_s
        self.target_turn_events[env_ids] = 0.0
        self.target_preturn_events[env_ids] = 0.0
        self.target_reflect_events[env_ids] = 0.0

        # Store world state and expose as goal_world for high-level.
        self.target_world[env_ids] = self.env_origins[env_ids, :2] + target_local
        self.target_vel_world[env_ids].zero_()
        self.goal_world[env_ids] = self.target_world[env_ids]

    def _update_moving_target(self):
        """Vectorized moving target update (called at low-level frequency)."""
        if not self._moving_target_enabled():
            return
        device = self.device
        # Update at high-level scene dt for performance (default 10Hz).
        if not hasattr(self, "_moving_target_time_accum"):
            self._moving_target_time_accum = 0.0
        self._moving_target_time_accum += float(self.dt)
        dt_high = float(getattr(self.cfg.terrain, "scene_high_dt", 0.1))
        if self._moving_target_time_accum + 1e-9 < dt_high:
            return
        dt = float(self._moving_target_time_accum)
        self._moving_target_time_accum = 0.0

        moving_mode = self._moving_target_mode()
        if moving_mode == "s1_gate_script":
            d = self._current_scene_difficulty(expected_num=self.num_envs, device=device, dtype=torch.float32)
            self._update_moving_target_s1(dt=dt, d=d)
            return
        if moving_mode == "s0_circle_right":
            self._update_moving_target_s0_circle(dt=dt)
            return
        if moving_mode == "e_l_conflict_script":
            self._update_moving_target_e_l_conflict(dt=dt)
            return
        if moving_mode == "e_s_corridor_script":
            self._update_moving_target_e_s_corridor(dt=dt)
            return

        d = self._current_scene_difficulty(expected_num=self.num_envs, device=device, dtype=torch.float32)
        active_mask = torch.ones(self.num_envs, device=device, dtype=torch.bool)

        # Freeze after reset for early-stage learnability.
        if hasattr(self, "target_freeze_timer"):
            frozen = self.target_freeze_timer > 0.0
            if frozen.any():
                self.target_freeze_timer = torch.clamp(self.target_freeze_timer - dt, min=0.0)
                self.target_vel_world[frozen].zero_()
                self.goal_world[frozen] = self.target_world[frozen]
                active_mask = ~frozen
                if active_mask.any():
                    d = d.clone()
                    d[frozen] = 0.0
                else:
                    return

        v_max = float(getattr(self.nav_cfg, "moving_target_v_max", 1.2))
        v_min = float(getattr(self.nav_cfg, "moving_target_v_min", 0.05))
        v_typ = float(getattr(self.nav_cfg, "moving_target_v_typical", 0.6))
        turn_rate_max = float(getattr(self.nav_cfg, "moving_target_turn_rate_max", 1.0))
        accel_max = float(getattr(self.nav_cfg, "moving_target_accel_max", 2.0))
        turn_deg_easy = float(getattr(self.nav_cfg, "moving_target_turn_deg_easy", 10.0))
        turn_deg_hard = float(getattr(self.nav_cfg, "moving_target_turn_deg_hard", 90.0))
        turn_deg_hard = float(np.clip(turn_deg_hard, 0.0, 90.0))
        period_fast = float(getattr(self.nav_cfg, "moving_target_cmd_period_fast", 0.6))
        period_slow = float(getattr(self.nav_cfg, "moving_target_cmd_period_slow", 2.0))
        speed_span_easy = float(getattr(self.nav_cfg, "moving_target_speed_span_easy", 0.06))
        speed_span_hard = float(getattr(self.nav_cfg, "moving_target_speed_span_hard", 0.35))
        speed_wave_amp_cfg = float(getattr(self.nav_cfg, "moving_target_speed_wave_amp", 0.15))
        speed_wave_rate_slow = float(getattr(self.nav_cfg, "moving_target_speed_wave_rate_slow", 0.6))
        speed_wave_rate_fast = float(getattr(self.nav_cfg, "moving_target_speed_wave_rate_fast", 1.8))
        preturn_lookahead_s = float(getattr(self.nav_cfg, "moving_target_preturn_lookahead_s", 0.8))
        preturn_deg_min = float(getattr(self.nav_cfg, "moving_target_preturn_deg_min", 25.0))
        preturn_deg_max = float(getattr(self.nav_cfg, "moving_target_preturn_deg_max", 90.0))
        preturn_deg_max = float(np.clip(preturn_deg_max, 0.0, 90.0))
        preturn_center_bias = float(getattr(self.nav_cfg, "moving_target_preturn_center_bias", 0.6))
        preturn_center_bias = float(np.clip(preturn_center_bias, 0.0, 1.0))

        self.target_turn_events.zero_()
        self.target_preturn_events.zero_()
        self.target_reflect_events.zero_()

        # More frequent command changes at higher difficulty.
        period = period_slow + (period_fast - period_slow) * d
        period = torch.clamp(period, min=0.05)

        # Scene bounds.
        half_len = 0.5 * float(self.cfg.terrain.terrain_length)
        half_wid = 0.5 * float(self.cfg.terrain.terrain_width)
        margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.6))
        x_min = -half_wid + margin
        x_max = half_wid - margin
        y_min = -half_len + margin
        y_max = half_len - margin

        pos_local_curr = self.target_world - self.env_origins[:, :2]
        dir_x_curr = torch.sin(self.target_heading)
        dir_y_curr = torch.cos(self.target_heading)
        eps = 1e-6
        dist_to_x = torch.where(
            dir_x_curr >= 0.0,
            (x_max - pos_local_curr[:, 0]) / (torch.abs(dir_x_curr) + eps),
            (pos_local_curr[:, 0] - x_min) / (torch.abs(dir_x_curr) + eps),
        )
        dist_to_y = torch.where(
            dir_y_curr >= 0.0,
            (y_max - pos_local_curr[:, 1]) / (torch.abs(dir_y_curr) + eps),
            (pos_local_curr[:, 1] - y_min) / (torch.abs(dir_y_curr) + eps),
        )
        dist_to_boundary_along_heading = torch.minimum(dist_to_x, dist_to_y)
        lookahead_dist = torch.clamp(self.target_speed, min=v_min) * preturn_lookahead_s + margin * 0.25
        preturn_needed = dist_to_boundary_along_heading <= lookahead_dist

        # Countdown.
        self.target_cmd_timer[active_mask] -= dt
        need_cmd = active_mask & ((self.target_cmd_timer <= 0.0) | preturn_needed)
        if need_cmd.any():
            turn_deg_max = turn_deg_easy + (turn_deg_hard - turn_deg_easy) * d
            turn_deg_max = torch.clamp(turn_deg_max, min=0.0, max=90.0)
            turn_rad_max = torch.deg2rad(turn_deg_max)
            rand_delta = (torch.rand(self.num_envs, device=device) * 2.0 - 1.0) * turn_rad_max

            center_vec = -pos_local_curr
            cross_z = dir_x_curr * center_vec[:, 1] - dir_y_curr * center_vec[:, 0]
            sign_to_center = torch.sign(cross_z)
            random_sign = torch.where(
                torch.rand(self.num_envs, device=device) > 0.5,
                torch.ones(self.num_envs, device=device),
                -torch.ones(self.num_envs, device=device),
            )
            sign_to_center = torch.where(torch.abs(sign_to_center) < 1e-4, random_sign, sign_to_center)

            preturn_deg = preturn_deg_min + (preturn_deg_max - preturn_deg_min) * d
            preturn_deg = torch.clamp(preturn_deg, min=0.0, max=90.0)
            preturn_delta = sign_to_center * torch.deg2rad(preturn_deg)
            preturn_delta = (
                (1.0 - preturn_center_bias) * rand_delta + preturn_center_bias * preturn_delta
            )

            preturn_mask = need_cmd & preturn_needed
            delta_heading = torch.where(preturn_mask, preturn_delta, rand_delta)
            theta_des = self.target_heading + delta_heading
            theta_des = torch.atan2(torch.sin(theta_des), torch.cos(theta_des))

            speed_span = speed_span_easy + (speed_span_hard - speed_span_easy) * d
            speed_span = torch.clamp(speed_span, min=0.0, max=max(0.0, v_max - v_min))
            v_cmd = v_typ + (torch.rand(self.num_envs, device=device) * 2.0 - 1.0) * speed_span
            # Large heading changes slow down target for smoother arcs.
            turn_scale = 1.0 - 0.35 * (
                torch.abs(delta_heading) / torch.clamp(turn_rad_max, min=1e-3)
            )
            turn_scale = torch.clamp(turn_scale, min=0.55, max=1.0)
            v_cmd = torch.clamp(v_cmd * turn_scale, v_min, v_max)

            self.target_heading_des = torch.where(need_cmd, theta_des, self.target_heading_des)
            self.target_speed_des = torch.where(need_cmd, v_cmd, self.target_speed_des)

            timer_default = period * (0.7 + 0.6 * torch.rand(self.num_envs, device=device))
            timer_preturn = torch.clamp(0.35 + 0.35 * (1.0 - d), min=0.2)
            next_timer = torch.where(preturn_mask, timer_preturn, timer_default)
            self.target_cmd_timer = torch.where(need_cmd, next_timer, self.target_cmd_timer)

            turn_events = need_cmd.float()
            preturn_events = preturn_mask.float()
            self.target_turn_events[:] = turn_events
            self.target_preturn_events[:] = preturn_events
            self.target_turn_count += turn_events
            self.target_preturn_count += preturn_events

        # Smoothly turn toward desired heading.
        dtheta = torch.atan2(
            torch.sin(self.target_heading_des - self.target_heading),
            torch.cos(self.target_heading_des - self.target_heading),
        )
        max_turn = (0.3 + 0.7 * d) * turn_rate_max * dt
        dtheta = torch.clamp(dtheta, -max_turn, max_turn)
        heading_next = self.target_heading + dtheta
        heading_next = torch.atan2(torch.sin(heading_next), torch.cos(heading_next))
        self.target_heading = torch.where(active_mask, heading_next, self.target_heading)

        # Smoothly change speed with a bounded low-frequency wave (avoid long constant-speed segments).
        wave_rate = speed_wave_rate_slow + (speed_wave_rate_fast - speed_wave_rate_slow) * d
        speed_phase_next = self.target_speed_phase + wave_rate * dt
        speed_phase_next = torch.atan2(torch.sin(speed_phase_next), torch.cos(speed_phase_next))
        self.target_speed_phase = torch.where(active_mask, speed_phase_next, self.target_speed_phase)
        wave_amp = speed_wave_amp_cfg * (0.25 + 0.75 * d)
        speed_des_wave = torch.clamp(self.target_speed_des + wave_amp * torch.sin(self.target_speed_phase), v_min, v_max)

        dv = speed_des_wave - self.target_speed
        max_dv = (0.4 + 0.6 * d) * accel_max * dt
        dv = torch.clamp(dv, -max_dv, max_dv)
        speed_next = torch.clamp(self.target_speed + dv, 0.0, v_max)
        self.target_speed = torch.where(active_mask, speed_next, self.target_speed)

        # Integrate position in local env frame.
        dir_x = torch.sin(self.target_heading)
        dir_y = torch.cos(self.target_heading)
        vel_local = torch.stack([self.target_speed * dir_x, self.target_speed * dir_y], dim=-1)
        pos_local = pos_local_curr + vel_local * dt
        pos_local = torch.where(active_mask.unsqueeze(1), pos_local, pos_local_curr)

        # Keep within bounds. Reflection remains as a fallback only.
        hit_x = active_mask & ((pos_local[:, 0] < x_min) | (pos_local[:, 0] > x_max))
        hit_y = active_mask & ((pos_local[:, 1] < y_min) | (pos_local[:, 1] > y_max))
        reflect_mask = hit_x | hit_y
        if hit_x.any():
            pos_local[:, 0] = torch.clamp(pos_local[:, 0], x_min, x_max)
            self.target_heading = torch.where(hit_x, -self.target_heading, self.target_heading)
            self.target_heading_des = torch.where(hit_x, -self.target_heading_des, self.target_heading_des)
        if hit_y.any():
            pos_local[:, 1] = torch.clamp(pos_local[:, 1], y_min, y_max)
            self.target_heading = torch.where(hit_y, math.pi - self.target_heading, self.target_heading)
            self.target_heading_des = torch.where(hit_y, math.pi - self.target_heading_des, self.target_heading_des)
        self.target_heading = torch.atan2(torch.sin(self.target_heading), torch.cos(self.target_heading))
        self.target_heading_des = torch.atan2(torch.sin(self.target_heading_des), torch.cos(self.target_heading_des))
        if reflect_mask.any():
            reflect_events = reflect_mask.float()
            self.target_reflect_events[:] = reflect_events
            self.target_reflect_count += reflect_events

        # Rebuild velocity from (possibly reflected) heading.
        dir_x = torch.sin(self.target_heading)
        dir_y = torch.cos(self.target_heading)
        vel_local = torch.stack([self.target_speed * dir_x, self.target_speed * dir_y], dim=-1)
        vel_local = torch.where(active_mask.unsqueeze(1), vel_local, torch.zeros_like(vel_local))

        # Commit world state.
        self.target_world = self.env_origins[:, :2] + pos_local
        self.target_vel_world = vel_local
        self.goal_world[:] = self.target_world

    def _sync_dynamic_obstacles(self):
        if self.dynamic_actor_indices is None:
            return
        t = self.scene_dyn_time.view(-1, 1, 1)
        period = self.dynamic_period.unsqueeze(-1).clamp(min=1e-3)
        phase = self.dynamic_phase.unsqueeze(-1)
        half = 0.5 * period
        tau = torch.remainder(t + phase, period)
        progress = torch.where(tau <= half, tau / half, (period - tau) / half).clamp(0.0, 1.0)
        dist = progress * self.dynamic_path_len.unsqueeze(-1)
        pos_local = self.dynamic_start + self.dynamic_dir * dist
        pos_local[..., 2] = self.dynamic_height * 0.5
        pos_world = pos_local + self.env_origins[:, None, :3]
        inactive_mask = ~self.dynamic_active
        if inactive_mask.any():
            pos_world[inactive_mask] = 0.0
            pos_world[inactive_mask, 2] = -5.0
        pos_flat = pos_world.reshape(-1, 3)
        quat_flat = self.dynamic_quat.reshape(-1, 4)
        indices = self.dynamic_actor_indices_flat
        indices_long = self.dynamic_actor_indices_flat_long
        root_states = getattr(self, "all_root_states", self.root_states)
        root_states[indices_long, :3] = pos_flat
        root_states[indices_long, 3:7] = quat_flat
        root_states[indices_long, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(indices),
            len(indices),
        )
        # 同步动态障碍碰撞过滤：active 开碰撞，inactive 关碰撞
        if self.dynamic_actor_handles is not None and self.dynamic_active is not None:
            scene_filter = self._scene_collision_filter()
            for env_id in range(self.num_envs):
                env_handles = self.dynamic_actor_handles[env_id]
                active_mask = self.dynamic_active[env_id].tolist()
                for local_id, actor_handle in enumerate(env_handles):
                    target_filter = scene_filter if active_mask[local_id] else 0
                    self._apply_actor_collision_filter(
                        self.envs[env_id], actor_handle, target_filter, env_id, debug_tag=""
                    )

    def _maybe_resample_scene_columns(self, env_ids: torch.Tensor):
        return

    def _reset_root_states(self, env_ids):
        """重置root状态，slalom地形时固定入口位姿"""
        if len(env_ids) == 0:
            return
        terrain_type = getattr(self.cfg.terrain, "terrain_type", None)
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
            if not terrain_type and hasattr(self, "terrain_types"):
                proportions = []
                running = 0.0
                for p in self.cfg.terrain.terrain_proportions:
                    running += float(p)
                    proportions.append(running)
                slalom_low = proportions[-2] if len(proportions) > 1 else 0.0
                slalom_high = proportions[-1] if proportions else 0.0
                choice = self.terrain_types[env_ids].float() / float(self.cfg.terrain.num_cols) + 0.001
                slalom_mask = (choice >= slalom_low) & (choice < slalom_high)
                if slalom_mask.any():
                    slalom_env_ids = env_ids[slalom_mask]
                    x_offset = -0.5 * self.cfg.terrain.terrain_length + 1.0
                    y_offset = 0.0
                    self.root_states[slalom_env_ids] = self.base_init_state
                    self.root_states[slalom_env_ids, :3] += self.env_origins[slalom_env_ids]
                    self.root_states[slalom_env_ids, 0] += x_offset
                    self.root_states[slalom_env_ids, 1] += y_offset
                    yaw = -0.5 * math.pi
                    qz = math.sin(yaw / 2.0)
                    qw = math.cos(yaw / 2.0)
                    quat = torch.tensor([0.0, 0.0, qz, qw], device=self.device)
                    self.root_states[slalom_env_ids, 3:7] = quat.repeat(len(slalom_env_ids), 1)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.nav_cfg is not None and getattr(self.nav_cfg, "spawn_edge_enable", False):
            half_len = 0.5 * self.cfg.terrain.terrain_length
            half_wid = 0.5 * self.cfg.terrain.terrain_width
            margin = getattr(self.nav_cfg, "spawn_edge_margin", 0.3)
            ring_half = getattr(self.cfg.terrain, "fixed_layout_ring_half_size", 0.0)
            outside_margin = getattr(self.nav_cfg, "spawn_outside_margin", 0.2)
            min_dist = ring_half + outside_margin
            edge_x = max(0.0, half_len - margin, min_dist)
            edge_y = max(0.0, half_wid - margin, min_dist)
            edge_x = min(edge_x, half_len)
            edge_y = min(edge_y, half_wid)

            num = len(env_ids)
            edges = torch.randint(0, 4, (num,), device=self.device)
            rand_x = torch_rand_float(-edge_x, edge_x, (num, 1), device=self.device).squeeze(1)
            rand_y = torch_rand_float(-edge_y, edge_y, (num, 1), device=self.device).squeeze(1)

            local_x = rand_x.clone()
            local_y = rand_y.clone()
            mask_left = edges == 0
            mask_right = edges == 1
            mask_bottom = edges == 2
            mask_top = edges == 3
            local_x[mask_left] = -edge_x
            local_x[mask_right] = edge_x
            local_y[mask_bottom] = -edge_y
            local_y[mask_top] = edge_y

            self.root_states[env_ids, 0] = self.env_origins[env_ids, 0] + local_x
            self.root_states[env_ids, 1] = self.env_origins[env_ids, 1] + local_y

            yaw_to_center = torch.atan2(-local_y, -local_x)
            jitter_deg = getattr(self.nav_cfg, "spawn_yaw_jitter_deg", 30.0)
            jitter = torch_rand_float(
                -math.radians(jitter_deg),
                math.radians(jitter_deg),
                (num, 1),
                device=self.device,
            ).squeeze(1)
            yaw = yaw_to_center + jitter
            qz = torch.sin(0.5 * yaw)
            qw = torch.cos(0.5 * yaw)
            quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
            self.root_states[env_ids, 3:7] = quat
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.1, 0.1, (len(env_ids), 6), device=self.device
        )
        self._sync_robot_root_states(env_ids)

    def _scene_spawn_bounds(self, scene_spec: Optional[SceneSpec]):
        if scene_spec is None:
            return None
        if scene_spec.scene_type == "s3_doorway_rooms":
            width = float(scene_spec.params.get("room_width", self.cfg.terrain.terrain_width))
        else:
            return None
        length = float(self.cfg.terrain.terrain_length)
        return width, length

    def _scene_goal_ranges(self, scene_spec: Optional[SceneSpec]):
        bounds = self._scene_spawn_bounds(scene_spec)
        if bounds is None:
            return None
        width, length = bounds
        if self.nav_cfg is not None:
            margin = float(getattr(self.nav_cfg, "goal_scene_margin", self.scene_margin))
        else:
            margin = float(self.scene_margin)
        if width <= 2.0 * margin or length <= 2.0 * margin:
            return None
        range_x = (-0.5 * width + margin, 0.5 * width - margin)
        range_y = (-0.5 * length + margin, 0.5 * length - margin)
        return range_x, range_y

    def _corridor_half_width_at_y(self, scene_spec: SceneSpec, y_local: float) -> float:
        params = scene_spec.params or {}
        width_nom = float(params.get("corridor_width_nom", params.get("corridor_width", self.cfg.terrain.terrain_width)))
        half_nom = 0.5 * width_nom
        gates = params.get("corridor_gates", [])
        if not isinstance(gates, list) or not gates:
            return half_nom
        half_min = half_nom
        for gate in gates:
            y0 = float(gate.get("y0", 0.0))
            length = float(gate.get("length", 0.0))
            door_width = gate.get("door_width", None)
            if door_width is None:
                continue
            if abs(y_local - y0) <= 0.5 * length:
                half_min = min(half_min, 0.5 * min(width_nom, float(door_width)))
        return half_min

    def _is_scene_spawn_clear(self, scene_spec: SceneSpec, x_local: float, y_local: float, clearance: float) -> bool:
        for spec in scene_spec.static_obstacles:
            if hasattr(spec, "position") and hasattr(spec, "size"):
                pos = spec.position
                size = spec.size
            elif isinstance(spec, dict):
                pos = spec.get("position", None)
                size = spec.get("size", None)
                if size is None and spec.get("type", "") == "cylinder":
                    r = float(spec.get("radius", 0.0))
                    h = float(spec.get("height", 0.0))
                    size = (2.0 * r, 2.0 * r, h)
                if pos is None or size is None:
                    continue
            else:
                continue
            dx = abs(x_local - float(pos[0]))
            dy = abs(y_local - float(pos[1]))
            limit_x = 0.5 * float(size[0]) + clearance
            limit_y = 0.5 * float(size[1]) + clearance
            if dx < limit_x and dy < limit_y:
                return False
        return True

    def _apply_scene_spawn(self, env_ids: torch.Tensor):
        if self.scene_spec_cache is None or env_ids.numel() == 0:
            return
        if self.nav_cfg is not None:
            margin = float(getattr(self.nav_cfg, "spawn_scene_margin", self.scene_margin))
            clearance = float(getattr(self.nav_cfg, "spawn_scene_clearance", self.scene_clearance))
            max_tries = int(getattr(self.nav_cfg, "spawn_scene_max_tries", 30))
        else:
            margin = float(self.scene_margin)
            clearance = float(self.scene_clearance)
            max_tries = 30

        updated = []
        for env_id in env_ids.tolist():
            scene_spec = self.scene_spec_cache[env_id]
            if scene_spec is None:
                terrain_type = getattr(self.cfg.terrain, "terrain_type", None)
                if terrain_type is not None and str(terrain_type).lower() in ("s1", "s1_corridor_gate"):
                    mesh_type = getattr(self.cfg.terrain, "mesh_type", None)
                    raise RuntimeError(
                        "S1 spawn requires scene_spec_cache but got None. "
                        f"mesh_type={mesh_type} terrain_type={terrain_type}. "
                        "请确保使用 classic heightfield 且 tile_meta 由 Terrain 生成。"
                    )
            seed = 0
            if scene_spec is not None:
                seed = int((scene_spec.layout_seed or 0) + env_id * 131)
            rng = np.random.RandomState(seed)
            if scene_spec is not None and scene_spec.scene_type == "s1_corridor_gate":
                params = scene_spec.params or {}
                length = float(params.get("corridor_length", self.cfg.terrain.terrain_length))
                y_start = -0.5 * length
                y_end = 0.5 * length
                spawn_buffer = float(params.get("corridor_spawn_buffer", margin))
                spawn_span = float(params.get("corridor_spawn_span", max(1.0, 0.3 * length)))
                x_center = float(params.get("corridor_x_center", 0.0))
                width_nom = float(params.get("corridor_width_nom", self.cfg.terrain.terrain_width))
                door_width = None
                gates = params.get("corridor_gates", []) or []
                if gates:
                    try:
                        door_width = float(gates[0].get("door_width", None))
                    except Exception:
                        door_width = None
                x_local = 0.0
                y_local = 0.0
                placed = False
                if not hasattr(self, "_s1_spawn_warned"):
                    self._s1_spawn_warned = False
                def _warn_once(msg: str):
                    if not self._s1_spawn_warned:
                        print(msg)
                        self._s1_spawn_warned = True
                y_min = y_start + spawn_buffer
                y_max = min(y_start + spawn_span, y_end - spawn_buffer)
                forbid_pad = max(margin, clearance)
                forbidden = []
                if gates:
                    for gate in gates:
                        try:
                            gy = float(gate.get("y0", 0.0))
                            gl = float(gate.get("length", 0.0))
                        except Exception:
                            continue
                        if gl <= 0.0:
                            continue
                        forbidden.append((gy - 0.5 * gl - forbid_pad, gy + 0.5 * gl + forbid_pad))
                for _ in range(max_tries):
                    if y_max <= y_min:
                        _warn_once(
                            "[Warn] S1 spawn invalid y-range, falling back. "
                            f"y_min={y_min:.3f} y_max={y_max:.3f} "
                            f"length={length:.3f} spawn_buffer={spawn_buffer:.3f} spawn_span={spawn_span:.3f}"
                        )
                        break
                    y_local = rng.uniform(y_min, y_max)
                    if forbidden:
                        hit = False
                        for y0, y1 in forbidden:
                            if y0 <= y_local <= y1:
                                hit = True
                                break
                        if hit:
                            continue
                    half_w = self._corridor_half_width_at_y(scene_spec, y_local)
                    usable_half = half_w - margin - clearance
                    if usable_half <= 0.0:
                        continue
                    x_min = x_center - usable_half
                    x_max = x_center + usable_half
                    x_local = rng.uniform(x_min, x_max)
                    if self._is_scene_spawn_clear(scene_spec, x_local, y_local, clearance):
                        placed = True
                        break
                if not placed:
                    _warn_once(
                        "[Warn] S1 spawn fallback triggered; using deterministic safe point. "
                        f"length={length:.3f} spawn_buffer={spawn_buffer:.3f} spawn_span={spawn_span:.3f} "
                        f"margin={margin:.3f} clearance={clearance:.3f} door_width={door_width}"
                    )
                    span_mid = 0.5 * min(spawn_span, 1.0)
                    y_local = y_start + spawn_buffer + span_mid
                    if y_max > y_min:
                        y_local = float(np.clip(y_local, y_min, y_max))
                    if forbidden:
                        inside = False
                        for y0, y1 in forbidden:
                            if y0 <= y_local <= y1:
                                inside = True
                                break
                        if inside:
                            if y_min < y_max and all(not (y0 <= y_min <= y1) for y0, y1 in forbidden):
                                y_local = y_min
                            elif y_min < y_max and all(not (y0 <= y_max <= y1) for y0, y1 in forbidden):
                                y_local = y_max
                    x_local = x_center
                self.root_states[env_id] = self.base_init_state
                self.root_states[env_id, :3] += self.env_origins[env_id]
                self.root_states[env_id, 0] += x_local
                self.root_states[env_id, 1] += y_local
                updated.append(env_id)
                continue
            if scene_spec is not None and scene_spec.scene_type == "s2_forest":
                params = scene_spec.params or {}
                clear_band = float(params.get("clear_band", 1.0))
                length = float(self.cfg.terrain.terrain_length)
                safe_half_width = max(0.0, 0.5 * clear_band - margin)
                y_min = -0.5 * length + margin
                y_max = 0.5 * length - margin
                if y_max <= y_min:
                    y_local = 0.0
                else:
                    y_local = rng.uniform(y_min, y_max)
                if safe_half_width <= 0.0:
                    x_local = 0.0
                else:
                    x_local = rng.uniform(-safe_half_width, safe_half_width)
                self.root_states[env_id] = self.base_init_state
                self.root_states[env_id, :3] += self.env_origins[env_id]
                self.root_states[env_id, 0] += x_local
                self.root_states[env_id, 1] += y_local
                updated.append(env_id)
                continue
            bounds = self._scene_spawn_bounds(scene_spec)
            if bounds is None:
                continue
            width, length = bounds
            if width <= 2.0 * margin or length <= 2.0 * margin:
                continue
            x_local = 0.0
            y_local = 0.0
            placed = False
            for _ in range(max_tries):
                x_local = rng.uniform(-0.5 * width + margin, 0.5 * width - margin)
                y_local = rng.uniform(-0.5 * length + margin, 0.5 * length - margin)
                if scene_spec is None or self._is_scene_spawn_clear(scene_spec, x_local, y_local, clearance):
                    placed = True
                    break
            if not placed:
                x_local = float(np.clip(x_local, -0.5 * width + margin, 0.5 * width - margin))
                y_local = float(np.clip(y_local, -0.5 * length + margin, 0.5 * length - margin))
            self.root_states[env_id] = self.base_init_state
            self.root_states[env_id, :3] += self.env_origins[env_id]
            self.root_states[env_id, 0] += x_local
            self.root_states[env_id, 1] += y_local
            updated.append(env_id)
        if updated:
            self._sync_robot_root_states(torch.tensor(updated, device=self.device, dtype=torch.long))


    def step(self,actions):
        #因为返回的观测改变了，因此需要重新定义step函数
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # print("------------>self torqures.shape:",self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        # return self.obs_buf, self.obs_vgf_buf, self.obs_terrain_buf, self.rew_buf, self.reset_buf, self.extras

    def step_separate(self,actions):
        #因为返回的观测改变了，因此需要重新定义step函数
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # print("------------>self torqures.shape:",self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step_separate()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.obs_vgf_buf = torch.clip(self.obs_vgf_buf, -clip_obs, clip_obs)
        self.obs_terrain_buf = torch.clip(self.obs_terrain_buf, -clip_obs, clip_obs)

        if self.enable_camera and self.camera_cfg is not None:
            if not hasattr(self, "depth_raw") or not hasattr(self, "depth_images"):
                self._init_camera_buffers()
            if self.common_step_counter % self.camera_cfg.capture_interval == 0:
                depth_raw = self._get_depth_images()
                processed = self._process_depth_for_network(depth_raw)
                self.depth_images[:] = processed

        obs_dict = self._build_obs_dict()
        return obs_dict, self.rew_buf, self.reset_buf, self.extras

    # def _create_envs(self):
    #     super()._create_envs()
    #     print("dof names=",self.dof_names)
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        #因为要对command ranges 进行修改，所以重新定义这个函数
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"]=[-0.6,0.6]
            self.command_ranges["lin_vel_y"]=[-0.9,0.9]
            self.command_ranges["ang_vel_yaw"]=[-0.6,0.6]
        

    def _compute_torques(self, actions):
        #重新定义力矩计算，调用电机类
        action_scaled = actions * self.cfg.control.action_scale
        pos_err = (action_scaled+self.default_dof_pos) - self.dof_pos
        vel_err = -self.dof_vel
        torques = torch.clip(self.actuator.get_torques(pos_err,vel_err),
                                  min=-self.torque_limits,
                                  max=self.torque_limits)
        # print("pos_err\n",pos_err[0].reshape(6,3))
        # print("vel_err\n",vel_err[0].reshape(6,3))
        # print("torques\n",torques[0].reshape(6,3))
        # if (torch.abs(pos_err[0])>1.0).any():
        #     print("reach max pos err")
        #     exit(0)
        # if (torques[0]==27.0).any():   
        #     print("reach max exit")
        #     exit(0)
            
        return torques

    def _post_physics_step_callback(self):
        if self._moving_target_enabled():
            self._update_moving_target()
        self._update_dynamic_obstacles()
        super()._post_physics_step_callback()
    

    def post_physics_step(self):
        #添加了base_lin_acc的计算，添加了IMU加速度计算，添加了分开式的观测计算,所以需要重写基类的这个函数
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._refresh_robot_root_states()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        #添加的部分，模拟加速度计的输出，所以要减去重力加速度，这里加速度的单位是g
        root_acc = ((self.root_states[:,7:10]-self.last_root_vel[:,:3])/self.cfg.sim.dt)/9.81 - self.gravity_vec
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, root_acc)

        root_ang_acc = (self.root_states[:,10:13]-self.last_root_vel[:,3:])/self.cfg.sim.dt
        self.base_ang_acc[:] = quat_rotate_inverse(self.base_quat, root_ang_acc)
        #根据IMU安装的位置，根据基座质心计算IMU质心处加速度的大小
        self.IMU_lin_acc = self.base_lin_acc + (self.base_ang_acc.cross(self.IMU_pos,dim=1) + self.base_ang_vel.cross(self.base_ang_vel.cross(self.IMU_pos,dim=1),dim=1))/9.81

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        #原来的观测计算方式
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        #新的观测计算方式
        # self.compute_observations_separated()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        self.last_contacts = (self.contact_forces[:, self.feet_indices, 2] > foot_contact_threshold)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()        

        if self.viewer and self.foot_traj_viz:
            self._draw_foot_end_trajectory()
        
        
    def post_physics_step_separate(self):
        #添加了base_lin_acc的计算，添加了IMU加速度计算，添加了分开式的观测计算,所以需要重写基类的这个函数
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._refresh_robot_root_states()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        #添加的部分，模拟加速度计的输出，所以要减去重力加速度，这里加速度的单位是g
        root_acc = ((self.root_states[:,7:10]-self.last_root_vel[:,:3])/self.cfg.sim.dt)/9.81 - self.gravity_vec
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, root_acc)

        root_ang_acc = (self.root_states[:,10:13]-self.last_root_vel[:,3:])/self.cfg.sim.dt
        self.base_ang_acc[:] = quat_rotate_inverse(self.base_quat, root_ang_acc)
        #根据IMU安装的位置，根据基座质心计算IMU质心处加速度的大小
        self.IMU_lin_acc = self.base_lin_acc + (self.base_ang_acc.cross(self.IMU_pos,dim=1) + self.base_ang_vel.cross(self.base_ang_vel.cross(self.IMU_pos,dim=1),dim=1))/9.81

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        #原来的观测计算方式
        #新的观测计算方式
        self.compute_observations_separated()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()        

        if self.viewer and self.foot_traj_viz:
            self._draw_foot_end_trajectory()

    def check_termination(self):
        contact_threshold = getattr(self.cfg.terrain, "collision_force_threshold", 1.0)
        collision_now = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_threshold,
            dim=1,
        )
        self.reset_buf = collision_now.clone()
        if self.s_avoid_enabled and hasattr(self, "s_avoid_episode_collision"):
            self.s_avoid_episode_collision |= collision_now
            nearest_obs = self._compute_s_avoid_nearest_obstacle_distance()
            if nearest_obs is not None:
                exposure_dist = float(getattr(self.cfg.terrain, "avoid_obstacle_exposure_distance", 1.8))
                self.s_avoid_episode_exposed |= nearest_obs < exposure_dist
                self.extras["avoid_nearest_obstacle_dist"] = float(nearest_obs.min().item())
            if hasattr(self, "goal_world") and hasattr(self, "s_avoid_episode_goal_best_dist"):
                goal_dist = torch.norm(self.goal_world - self.root_states[:, :2], dim=1)
                self.s_avoid_episode_goal_best_dist = torch.minimum(self.s_avoid_episode_goal_best_dist, goal_dist)

        height_threshold = getattr(self.cfg.env, "termination_height_threshold", None)
        if height_threshold is not None:
            if hasattr(self, "measured_heights"):
                height = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.025 - self.measured_heights, min=-1, max=1.0)
                base_height = torch.mean(height, dim=1)
            else:
                base_height = self.root_states[:, 2]
            self.reset_buf |= base_height < height_threshold

        max_tilt_deg = getattr(self.cfg.env, "termination_max_tilt_deg", None)
        if max_tilt_deg is not None:
            cos_max_tilt = math.cos(math.radians(max_tilt_deg))
            self.reset_buf |= self.projected_gravity[:, 2] > -cos_max_tilt

        no_episode_timeout = bool(getattr(self.cfg.env, "no_episode_timeout", False))
        if no_episode_timeout:
            self.time_out_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        else:
            self.time_out_buf = self.episode_length_buf > self.max_episode_length
            self.reset_buf |= self.time_out_buf
        
      


    def reset_idx(self, env_ids:torch.Tensor):
        prev_levels = None
        if hasattr(self, "terrain_levels") and len(env_ids) > 0:
            prev_levels = self.terrain_levels[env_ids].clone()
        if self.s_avoid_enabled and len(env_ids) > 0 and hasattr(self, "s_avoid_episode_collision"):
            completed_mask = self.episode_length_buf[env_ids] > 0
            if bool(completed_mask.any().item()):
                completed_env_ids = env_ids[completed_mask]
                completed_stage_ids = self.s_avoid_stage_per_env[completed_env_ids].clone()
                completed_flags = self.s_avoid_episode_collision[completed_env_ids].clone()
                completed_exposed = self.s_avoid_episode_exposed[completed_env_ids].clone()
                completed_progress = self._get_s_avoid_episode_progress_flags(
                    completed_env_ids,
                    stage_ids=completed_stage_ids,
                )
                completed_success = self._get_s_avoid_episode_success_flags(
                    completed_env_ids,
                    stage_ids=completed_stage_ids,
                )
                self._update_s_avoid_curriculum(
                    completed_flags,
                    completed_stage_ids,
                    completed_exposed,
                    completed_progress,
                    completed_success,
                )
            self.s_avoid_episode_collision[env_ids] = False
            self.s_avoid_episode_exposed[env_ids] = False
            self.s_avoid_episode_goal_init_dist[env_ids] = 0.0
            self.s_avoid_episode_goal_best_dist[env_ids] = 0.0
        self._maybe_resample_scene_columns(env_ids)
        super().reset_idx(env_ids)

        if len(env_ids) !=0:
            terrain_type = str(getattr(self.cfg.terrain, "terrain_type", "")).lower()
            is_s0_follow_plane = terrain_type in ("s0_follow_plane", "s0")
            force_resample = None
            if prev_levels is not None and bool(getattr(self.cfg.terrain, "scene_resample_on_level_change", False)):
                level_changed = self.terrain_levels[env_ids] != prev_levels
                force_resample = level_changed
            self._reset_scene(env_ids, force_resample=force_resample)
            if prev_levels is not None:
                self.scene_level_cache[env_ids] = self.terrain_levels[env_ids]
            self._apply_scene_spawn(env_ids)
            if self.s_avoid_enabled:
                self._reset_s_avoid_obstacles(env_ids)
                self._reset_s_avoid_robot_pose(env_ids)
            self._resample_nav_goals(env_ids)
            if self.s_avoid_enabled:
                self._reset_s_avoid_episode_progress(env_ids)
            if self._moving_target_enabled():
                # For S0, we re-place the target after locking spawn yaw; avoid double-reset here.
                if not is_s0_follow_plane:
                    self._reset_moving_target(env_ids)
            if hasattr(self, "target_turn_count"):
                self.target_turn_count[env_ids] = 0.0
                self.target_preturn_count[env_ids] = 0.0
                self.target_reflect_count[env_ids] = 0.0
                self.target_turn_events[env_ids] = 0.0
                self.target_preturn_events[env_ids] = 0.0
                self.target_reflect_events[env_ids] = 0.0
                self.target_reset_dist_error[env_ids] = 0.0
                self.target_reset_bearing_error[env_ids] = 0.0
            if hasattr(self, "goal_world"):
                yaw_offset = float(getattr(self.nav_cfg, "heading_offset_rad", 0.0)) if self.nav_cfg is not None else 0.0
                jitter_deg = float(getattr(self.nav_cfg, "spawn_yaw_jitter_deg", 0.0)) if self.nav_cfg is not None else 0.0

                s1_ids = []
                s0_ids = []
                e_ids = []
                e_s_ids = []
                if is_s0_follow_plane:
                    # S0 may run without scene_spec_cache (e.g., mesh_type=plane), so treat all
                    # env_ids as S0 for spawn-yaw locking and moving-target placement.
                    s0_ids = env_ids.tolist()
                elif self.scene_spec_cache is not None:
                    for env_id in env_ids.tolist():
                        scene_spec = self.scene_spec_cache[env_id]
                        if scene_spec is not None and scene_spec.scene_type == "s1_corridor_gate":
                            s1_ids.append(env_id)
                        elif scene_spec is not None and scene_spec.scene_type == "s0_follow_plane":
                            s0_ids.append(env_id)
                        elif scene_spec is not None and scene_spec.scene_type == "e_l_conflict_turn":
                            e_ids.append(env_id)
                        elif scene_spec is not None and scene_spec.scene_type == "e_s_corridor":
                            e_s_ids.append(env_id)
                if (not e_ids) and self._moving_target_enabled() and self._moving_target_mode() == "e_l_conflict_script":
                    e_ids = env_ids.tolist()
                if (not e_s_ids) and self._moving_target_enabled() and self._moving_target_mode() == "e_s_corridor_script":
                    e_s_ids = env_ids.tolist()
                if s1_ids:
                    s1_tensor = torch.tensor(s1_ids, device=self.device, dtype=torch.long)
                    jitter = torch_rand_float(
                        -math.radians(jitter_deg),
                        math.radians(jitter_deg),
                        (len(s1_ids), 1),
                        device=self.device,
                    ).squeeze(1)
                    yaw = yaw_offset + jitter
                    qz = torch.sin(0.5 * yaw)
                    qw = torch.cos(0.5 * yaw)
                    quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                    self.root_states[s1_tensor, 3:7] = quat
                    self._sync_robot_root_states(s1_tensor)

                other_ids = env_ids
                if s0_ids:
                    # S0: lock spawn yaw to +Y forward with small jitter, then re-place the target
                    # to keep robot/target consistent (avoid "reset mismatch" and early target loss).
                    s0_tensor = torch.tensor(s0_ids, device=self.device, dtype=torch.long)
                    # Force-reset robot root state for S0. This avoids a failure mode where
                    # moving target resets (buffer) but the robot remains at the last episode pose.
                    self.root_states[s0_tensor] = self.base_init_state
                    self.root_states[s0_tensor, :3] += self.env_origins[s0_tensor]
                    self.root_states[s0_tensor, 7:13] = 0.0
                    jitter = torch_rand_float(
                        -math.radians(jitter_deg),
                        math.radians(jitter_deg),
                        (len(s0_ids), 1),
                        device=self.device,
                    ).squeeze(1)
                    yaw = yaw_offset + jitter
                    qz = torch.sin(0.5 * yaw)
                    qw = torch.cos(0.5 * yaw)
                    quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                    self.root_states[s0_tensor, 3:7] = quat
                    self._sync_robot_root_states(s0_tensor)
                    if self._moving_target_enabled():
                        self._reset_moving_target(s0_tensor)
                    if hasattr(self, "target_world"):
                        delta = self.target_world[s0_tensor] - self.root_states[s0_tensor, :2]
                        cos_h = torch.cos(yaw)
                        sin_h = torch.sin(yaw)
                        x_r = cos_h * delta[:, 0] + sin_h * delta[:, 1]
                        y_f = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
                        bearing = torch.atan2(x_r, y_f)
                        desired = float(getattr(self.nav_cfg, "follow_distance_desired", 1.0))
                        dist_err = torch.abs(torch.norm(delta, dim=1) - desired)
                        self.target_reset_dist_error[s0_tensor] = dist_err
                        self.target_reset_bearing_error[s0_tensor] = torch.abs(bearing)

                    if getattr(self, "debug_viz", False) and hasattr(self, "target_world"):
                        if not getattr(self, "_s0_reset_align_warned", False):
                            delta = self.target_world[s0_tensor] - self.root_states[s0_tensor, :2]
                            cos_h = torch.cos(yaw)
                            sin_h = torch.sin(yaw)
                            x_r = cos_h * delta[:, 0] + sin_h * delta[:, 1]
                            y_f = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
                            bearing = torch.atan2(x_r, y_f)
                            too_far = torch.abs(bearing) > 0.35  # ~20deg
                            behind = y_f < 0.0
                            if bool((too_far | behind).any().item()):
                                import warnings
                                warnings.warn(
                                    f"[S0 reset] target not centered/forward for some envs: "
                                    f"bearing_deg(p95)={float(torch.quantile(torch.abs(bearing), 0.95).item() * 180.0 / math.pi):.1f}, "
                                    f"behind_frac={float(behind.float().mean().item()):.3f}.",
                                    stacklevel=1,
                                )
                            self._s0_reset_align_warned = True

                if s1_ids or s0_ids:
                    skip = set(s1_ids) | set(s0_ids)
                    rest = [env_id for env_id in env_ids.tolist() if env_id not in skip]
                    if rest:
                        other_ids = torch.tensor(rest, device=self.device, dtype=torch.long)
                    else:
                        other_ids = None

                if e_ids and hasattr(self, "target_world"):
                    e_tensor = torch.tensor(e_ids, device=self.device, dtype=torch.long)
                    # Rebuild robot spawn for e_L_conflict: face target with fixed start gap.
                    self.root_states[e_tensor] = self.base_init_state
                    self.root_states[e_tensor, :3] += self.env_origins[e_tensor]
                    self.root_states[e_tensor, 7:13] = 0.0

                    geom = self._get_e_l_conflict_turn_path()
                    spawn_gap = float(geom.get("spawn_gap", 0.5))
                    target_xy = self.target_world[e_tensor]
                    target_vel = self.target_vel_world[e_tensor]
                    vel_norm = torch.norm(target_vel, dim=1, keepdim=True)
                    dir_world = torch.zeros_like(target_vel)
                    dir_world[:, 1] = 1.0
                    valid_vel = vel_norm > 1e-6
                    dir_world = torch.where(valid_vel, target_vel / vel_norm.clamp_min(1e-6), dir_world)
                    robot_xy = target_xy - spawn_gap * dir_world

                    half_len = 0.5 * float(self.cfg.terrain.terrain_length)
                    half_wid = 0.5 * float(self.cfg.terrain.terrain_width)
                    margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.3))
                    x_min = self.env_origins[e_tensor, 0] - half_wid + margin
                    x_max = self.env_origins[e_tensor, 0] + half_wid - margin
                    y_min = self.env_origins[e_tensor, 1] - half_len + margin
                    y_max = self.env_origins[e_tensor, 1] + half_len - margin
                    robot_xy[:, 0] = torch.clamp(robot_xy[:, 0], x_min, x_max)
                    robot_xy[:, 1] = torch.clamp(robot_xy[:, 1], y_min, y_max)
                    self.root_states[e_tensor, :2] = robot_xy

                    delta = target_xy - robot_xy
                    yaw = torch.atan2(delta[:, 0], delta[:, 1])
                    qz = torch.sin(0.5 * yaw)
                    qw = torch.cos(0.5 * yaw)
                    quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                    self.root_states[e_tensor, 3:7] = quat
                    self._sync_robot_root_states(e_tensor)

                    cos_h = torch.cos(yaw)
                    sin_h = torch.sin(yaw)
                    x_r = cos_h * delta[:, 0] + sin_h * delta[:, 1]
                    y_f = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
                    bearing = torch.atan2(x_r, y_f)
                    self.target_reset_dist_error[e_tensor] = torch.abs(torch.norm(delta, dim=1) - spawn_gap)
                    self.target_reset_bearing_error[e_tensor] = torch.abs(bearing)
                    if not getattr(self, "_e_l_conflict_spawn_logged", False):
                        d_mean = float(torch.norm(delta, dim=1).mean().item())
                        b_mean = float(torch.abs(bearing).mean().item() * 180.0 / math.pi)
                        print(
                            f"[Scene] e_L_conflict spawn check: mean_dist={d_mean:.3f} (target {spawn_gap:.3f}), "
                            f"mean_bearing_deg={b_mean:.2f}"
                        )
                        self._e_l_conflict_spawn_logged = True

                if e_s_ids and hasattr(self, "target_world"):
                    e_s_tensor = torch.tensor(e_s_ids, device=self.device, dtype=torch.long)
                    self.root_states[e_s_tensor] = self.base_init_state
                    self.root_states[e_s_tensor, :3] += self.env_origins[e_s_tensor]
                    self.root_states[e_s_tensor, 7:13] = 0.0

                    geom = self._get_e_s_corridor_geometry()
                    spawn_gap = float(geom.get("spawn_gap", 1.0))
                    target_xy = self.target_world[e_s_tensor]
                    target_vel = self.target_vel_world[e_s_tensor]
                    vel_norm = torch.norm(target_vel, dim=1, keepdim=True)
                    dir_world = torch.zeros_like(target_vel)
                    dir_world[:, 1] = 1.0
                    valid_vel = vel_norm > 1e-6
                    dir_world = torch.where(valid_vel, target_vel / vel_norm.clamp_min(1e-6), dir_world)
                    robot_xy = target_xy - spawn_gap * dir_world

                    half_len = 0.5 * float(self.cfg.terrain.terrain_length)
                    half_wid = 0.5 * float(self.cfg.terrain.terrain_width)
                    margin = float(getattr(self.nav_cfg, "moving_target_margin", 0.3))
                    x_min = self.env_origins[e_s_tensor, 0] - half_wid + margin
                    x_max = self.env_origins[e_s_tensor, 0] + half_wid - margin
                    y_min = self.env_origins[e_s_tensor, 1] - half_len + margin
                    y_max = self.env_origins[e_s_tensor, 1] + half_len - margin
                    robot_xy[:, 0] = torch.clamp(robot_xy[:, 0], x_min, x_max)
                    robot_xy[:, 1] = torch.clamp(robot_xy[:, 1], y_min, y_max)
                    self.root_states[e_s_tensor, :2] = robot_xy

                    delta = target_xy - robot_xy
                    yaw = torch.atan2(delta[:, 0], delta[:, 1])
                    qz = torch.sin(0.5 * yaw)
                    qw = torch.cos(0.5 * yaw)
                    quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                    self.root_states[e_s_tensor, 3:7] = quat
                    self._sync_robot_root_states(e_s_tensor)

                    cos_h = torch.cos(yaw)
                    sin_h = torch.sin(yaw)
                    x_r = cos_h * delta[:, 0] + sin_h * delta[:, 1]
                    y_f = -sin_h * delta[:, 0] + cos_h * delta[:, 1]
                    bearing = torch.atan2(x_r, y_f)
                    self.target_reset_dist_error[e_s_tensor] = torch.abs(torch.norm(delta, dim=1) - spawn_gap)
                    self.target_reset_bearing_error[e_s_tensor] = torch.abs(bearing)
                    if not getattr(self, "_e_s_corridor_spawn_logged", False):
                        d_mean = float(torch.norm(delta, dim=1).mean().item())
                        b_mean = float(torch.abs(bearing).mean().item() * 180.0 / math.pi)
                        print(
                            f"[Scene] e_S_corridor spawn check: mean_dist={d_mean:.3f} (target {spawn_gap:.3f}), "
                            f"mean_bearing_deg={b_mean:.2f}"
                        )
                        self._e_s_corridor_spawn_logged = True

                if s1_ids or s0_ids or e_ids or e_s_ids:
                    skip = set(s1_ids) | set(s0_ids) | set(e_ids) | set(e_s_ids)
                    rest = [env_id for env_id in env_ids.tolist() if env_id not in skip]
                    if rest:
                        other_ids = torch.tensor(rest, device=self.device, dtype=torch.long)
                    else:
                        other_ids = None

                if (not self.s_avoid_enabled) and other_ids is not None and other_ids.numel() > 0:
                    # 出生时朝向目标点：用 heading_offset 对齐策略朝向，并加入 yaw 抖动
                    goal_delta = self.goal_world[other_ids] - self.root_states[other_ids, :2]
                    jitter = torch_rand_float(
                        -math.radians(jitter_deg),
                        math.radians(jitter_deg),
                        (len(other_ids), 1),
                        device=self.device,
                    ).squeeze(1)
                    yaw = torch.atan2(goal_delta[:, 1], goal_delta[:, 0]) - yaw_offset + jitter
                    qz = torch.sin(0.5 * yaw)
                    qw = torch.cos(0.5 * yaw)
                    quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                    self.root_states[other_ids, 3:7] = quat
                    self._sync_robot_root_states(other_ids)
            self.get_expert_actions()
            clear_on_reset = bool(getattr(self.cfg.terrain, "debug_viz_clear_on_reset", False))
            clear_debug_target_traj = (
                self.viewer is not None
                and bool(getattr(self, "debug_viz", False))
                and self._moving_target_enabled()
            )
            clear_debug_goal_traj = (
                self.viewer is not None
                and bool(getattr(self, "debug_viz", False))
                and self.s_avoid_enabled
                and hasattr(self, "goal_world")
                and (not self._moving_target_enabled())
            )
            # Default behavior: do NOT clear global viewer lines on each reset.
            # Global clear can make trajectories disappear before the currently viewed env resets.
            if self.viewer is not None:
                need_clear = False
                if clear_on_reset and bool(getattr(self, "foot_traj_viz", False)):
                    need_clear = True
                if clear_on_reset and bool(getattr(self, "debug_viz", False)) and self._moving_target_enabled():
                    need_clear = True
                if clear_debug_target_traj:
                    need_clear = True
                if clear_debug_goal_traj:
                    need_clear = True
                if need_clear:
                    self.gym.clear_lines(self.viewer)
            # Reset per-env debug trajectory state to avoid cross-episode line segments.
            if hasattr(self, "_viz_prev_valid"):
                ids = env_ids.detach().cpu().numpy()
                self._viz_prev_valid[ids] = False
                if clear_debug_target_traj:
                    self._viz_prev_valid[:] = False
                    if hasattr(self, "_viz_traj_tick"):
                        self._viz_traj_tick = 0
            if hasattr(self, "_viz_goal_prev_valid"):
                ids = env_ids.detach().cpu().numpy()
                self._viz_goal_prev_valid[ids] = False
                if clear_debug_goal_traj:
                    self._viz_goal_prev_valid[:] = False
                    if hasattr(self, "_viz_goal_traj_tick"):
                        self._viz_goal_traj_tick = 0
        
            # print("reset ids=",env_ids)
            # print("resample commands\n",self.commands)
        # if len(env_ids) !=0:
            # print("------------------->reset env_ids=",env_ids)

        #需要额外重设上一次的基座线速度，设置为0，设置上一次碰撞，设置为0
        # self.last_root_vel[env_ids] = 0.
        # self.last_contacts[env_ids] = 0.
        #TODO 基类环境中，没有重置上一次碰撞，原因？
    
    def get_expert_actions(self):
        #这个是专家参与动作交互，所以很多状态不需要再次判断
        #计算专家动作,输给专家的指令是[reset,vx,vy,vz,omega_z]
        command = torch.stack([self.reset_buf.clone(),self.commands[:,0],self.commands[:,1],torch.zeros_like(self.reset_buf),self.commands[:,2]],dim=1)
        expert_dofs = self.expert.ProcessCommand(command,self.dof_pos,self.dof_vel) #此时的actions还是关节角度的绝对位置，要进行转化
        self.expert_actions = ((expert_dofs-self.default_dof_pos)/self.cfg.control.action_scale).detach()  
        return self.expert_actions
    
    

    def _reset_dofs(self,env_ids):
        #不给初始关节角度添加随机值，因此重写
        self.dof_pos[env_ids]= self.default_dof_pos
        self.dof_vel[env_ids]= 0.
        if self.robot_actor_indices_int32 is None:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
        else:
            env_ids_int32 = self.robot_actor_indices_int32[env_ids]
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def compute_observations(self):
        #先记录观测的最大、最小、平均值
        
        #自定义的观测变了，需要重新定义观测
        # self.obs_buf = torch.cat([self.base_quat*self.obs_scales.quat,
        #                           self.base_ang_vel*self.obs_scales.ang_vel,
        #                           self.base_lin_acc*self.obs_scales.lin_acc,
        #                           (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.torques*self.obs_scales.dof_torque,
        #                           self.commands*self.commands_scale],dim=-1)
        # print("self.last_actions\n",self.last_actions)
        # self.obs_buf = torch.cat([self.last_actions*self.obs_scales.actions,
        #                           (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.torques*self.obs_scales.dof_torque,
        #                           self.commands*self.commands_scale],dim=-1) 
        #地形信息也可以被演员拿到
        # print("measure_heights\n",self.measured_heights[0,:20])
        # print("\n\n")
        #减去0.025是减去身体的厚度的一半
        if self.cfg.terrain.measure_heights:
            height=torch.clip((self.root_states[:,2].unsqueeze(1)-0.025-self.measured_heights),min=-1.0,max=1.0)
        else:
            print("-----------Not measure height, modify the obs----------------\n")
            exit(0)
        # print("measure_heights\n",height[0,:20])
            
        # self.obs_buf = torch.cat([self.last_actions*self.obs_scales.actions,
        #                           (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.torques*self.obs_scales.dof_torque,
        #                           self.commands*self.commands_scale,
        #                           height*self.obs_scales.height_measurements],dim=-1) 

        self.obs_buf = torch.cat([self.last_actions*self.obs_scales.actions,
                                  (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.torques*self.obs_scales.dof_torque,
                                  self.commands*self.commands_scale],dim=-1)         
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec[:self.cfg.env.num_observations]
        if self.privileged_obs_buf is not None:
            priv_part = torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,
                                   self.projected_gravity*self.obs_scales.gravity,
                                   self.contact_forces[:,self.feet_indices,2]*self.obs_scales.contact_force,
                                   height*self.obs_scales.height_measurements],dim=-1)
            # priv_part = torch.cat([height*self.obs_scales.height_measurements],dim=-1)            
            if self.add_noise:
                priv_part += (2*torch.rand_like(priv_part)-1)*self.noise_scale_vec[self.cfg.env.num_observations:]
            self.privileged_obs_buf = torch.cat([self.obs_buf, priv_part],dim=-1)
        # print("base_lin_vel=",self.privileged_obs_buf[0,75:78])
        # print("projected gravity=",self.privileged_obs_buf[0,78:81])
        # print("contact force=",self.privileged_obs_buf[0,81:87])
        # print("base lin acc=",self.base_lin_acc)
        # print("IMU lin acc=",self.IMU_lin_acc)
        # print("\n")
        # print("base_lin_vel=",self.obs_buf[0,75:78])


        # if self.add_noise:
        #     self.obs_buf += (2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec[:self.cfg.env.num_observations]
        #     if self.privileged_obs_buf is not None:
        #         self.privileged_obs_buf += (2*torch.rand_like(self.privileged_obs_buf)-1)*self.noise_scale_vec

        self._extract_robot_state()
        self._update_goal_buffer()

    def compute_observations_separated(self):
        #返回分为 obs(机器人本体可以获取的观测), obs_vgf(特权信息， 基座线速度，重力加速度，足端z方向力), obs_terrain(地形高度信息)
        height=torch.clip((self.root_states[:,2].unsqueeze(1)-0.025-self.measured_heights),min=-1.0,max=1.0)

        self.obs_buf = torch.cat([self.base_quat*self.obs_scales.quat,
                                  self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.base_lin_acc*self.obs_scales.lin_acc,
                                  (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.torques*self.obs_scales.dof_torque,
                                  self.commands*self.commands_scale],dim=-1)
        # self.obs_buf = torch.cat([(self.last_actions*self.obs_scales.actions,
        #                           self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.torques*self.obs_scales.dof_torque,
        #                           self.commands*self.commands_scale],dim=-1)
        #为了对collision相关损失进行更加精准的预测，将刚体碰撞力放入观测中
        # collision=torch.sum(1.*(torch.norm(self.contact_forces[:,self.penalised_contact_indices,:],dim=-1)>0.1),dim=1).unsqueeze(1)
        self.obs_vgf_buf = torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,
                                      self.projected_gravity*self.obs_scales.gravity,
                                      self.contact_forces[:,self.feet_indices,2]*self.obs_scales.contact_force],dim=-1)
        
        # self.obs_vgf_buf = torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,
        #                               self.projected_gravity*self.obs_scales.gravity,
        #                               self.contact_forces[:,self.feet_indices,2]*self.obs_scales.contact_force,
        #                               torch.norm(self.contact_forces[:,self.penalised_contact_indices,:])],dim=-1), dim=-1)        
        self.obs_terrain_buf = height*self.obs_scales.height_measurements

        if self.add_noise:
            # NOTE: separated obs dims are not aligned with cfg.env.num_observations.
            obs_dim = int(self.obs_buf.shape[1])
            vgf_dim = int(self.obs_vgf_buf.shape[1])
            terrain_dim = int(self.obs_terrain_buf.shape[1])
            needed = obs_dim + vgf_dim + terrain_dim
            if self.noise_scale_vec.shape[0] < needed:
                raise RuntimeError(
                    f"noise_scale_vec too short for separated obs: "
                    f"len={self.noise_scale_vec.shape[0]} needed={needed}"
                )
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:obs_dim]
            self.obs_vgf_buf += (2 * torch.rand_like(self.obs_vgf_buf) - 1) * self.noise_scale_vec[obs_dim:obs_dim + vgf_dim]
            self.obs_terrain_buf += (2 * torch.rand_like(self.obs_terrain_buf) - 1) * self.noise_scale_vec[obs_dim + vgf_dim:obs_dim + vgf_dim + terrain_dim]

        self._extract_robot_state()
        self._update_goal_buffer()

    def _build_obs_dict(self):
        depth_images = self.depth_images if hasattr(self, "depth_images") else None
        return {
            'proprioception': self.obs_buf,
            'privileged': self.obs_vgf_buf,
            'terrain': self.obs_terrain_buf,
            'depth': depth_images,
            'robot_state': self.robot_state_buf,
            'goal': self.goal_buf,
        }

    def get_observations_separated(self):
        return self._build_obs_dict()

    def _extract_robot_state(self):
        """提取高层使用的机器人状态 (num_envs, 9)"""
        quat = self.base_quat
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        yaw = torch.atan2(
            2.0 * (w*z + x*y),
            1.0 - 2.0 * (y*y + z*z)
        )
        roll = torch.atan2(
            2.0 * (w*x + y*z),
            1.0 - 2.0 * (x*x + y*y)
        )
        pitch = torch.asin(torch.clamp(2.0 * (w*y - z*x), -1.0, 1.0))

        pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
        pos_y = self.root_states[:, 1] - self.env_origins[:, 1]

        if isinstance(self.measured_heights, torch.Tensor) and self.measured_heights.numel() > 0:
            height = self.root_states[:, 2] - torch.mean(self.measured_heights, dim=1)
        else:
            height = self.root_states[:, 2]

        self.robot_state_buf = torch.stack([
            pos_x,
            pos_y,
            yaw,
            self.base_lin_vel[:, 0],
            self.base_lin_vel[:, 1],
            self.base_ang_vel[:, 2],
            height,
            roll,
            pitch
        ], dim=1)

    def _update_goal_buffer(self):
        if self.nav_cfg is None:
            return
        if not hasattr(self, "goal_world"):
            return
        goal_world = self.goal_world
        delta_world = goal_world - self.root_states[:, :2]
        heading = self.robot_state_buf[:, 2]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        # Project-wide contract (align with S1):
        # - heading=0 means world +Y is forward
        # - goal_buf is (x_right, y_forward) so bearing = atan2(x_right, y_forward)
        #
        # Project world delta to body (x_right, y_forward) using R(-heading).
        # heading=0 => forward=(0,+1), right=(+1,0).
        x_right = cos_h * delta_world[:, 0] + sin_h * delta_world[:, 1]
        y_forward = -sin_h * delta_world[:, 0] + cos_h * delta_world[:, 1]
        self.goal_buf[:] = torch.stack([x_right, y_forward], dim=1)

        if getattr(self.nav_cfg, "resample_on_reach", False):
            dist = torch.norm(delta_world, dim=1)
            reached = dist < getattr(self.nav_cfg, "goal_reached_threshold", 0.1)
            if reached.any():
                self._resample_nav_goals(reached.nonzero(as_tuple=False).flatten())

    def _line_has_obstacle(
        self,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        samples = int(getattr(self.nav_cfg, "goal_line_samples", 16))
        if samples <= 0:
            return torch.zeros(start_xy.shape[0], dtype=torch.bool, device=self.device)
        if self.height_samples is None:
            return torch.zeros(start_xy.shape[0], dtype=torch.bool, device=self.device)

        t = torch.linspace(0.0, 1.0, samples, device=self.device).view(1, -1, 1)
        points = start_xy.unsqueeze(1) + (goal_xy - start_xy).unsqueeze(1) * t

        border = self.cfg.terrain.border_size
        scale = self.cfg.terrain.horizontal_scale
        pts = points + border
        idx = (pts / scale).long()
        col = torch.clamp(idx[..., 0], 0, self.height_samples.shape[1] - 2)
        row = torch.clamp(idx[..., 1], 0, self.height_samples.shape[0] - 2)
        heights = self.height_samples[row, col] * self.cfg.terrain.vertical_scale

        threshold = getattr(self.nav_cfg, "goal_obstacle_height_threshold", 0.08)
        return (heights >= threshold).any(dim=1)

    def _resample_nav_goals(self, env_ids: torch.Tensor):
        if self.nav_cfg is None or env_ids.numel() == 0:
            return
        if self.s_avoid_enabled:
            self._resample_s_avoid_goals(env_ids)
            return
        goal_mode = getattr(self.nav_cfg, "goal_mode", "random")
        if goal_mode == "fixed":
            fixed_goal = torch.tensor(self.nav_cfg.fixed_goal, device=self.device)
            self.goal_world[env_ids] = fixed_goal.unsqueeze(0) + self.env_origins[env_ids, :2]
            return
        if goal_mode != "random":
            return
        self._sample_random_goals(env_ids)

    def _sample_random_goals(self, env_ids: torch.Tensor):
        max_tries = int(getattr(self.nav_cfg, "goal_sample_max_tries", 20))
        min_dist = float(getattr(self.nav_cfg, "goal_min_distance", 2.0))
        range_x = getattr(self.nav_cfg, "goal_range_x", [2.0, 5.0])
        range_y = getattr(self.nav_cfg, "goal_range_y", [-3.0, 3.0])
        allow_fallback = bool(getattr(self.nav_cfg, "goal_allow_fallback", True))
        force_blocking = bool(getattr(self.nav_cfg, "goal_force_blocking_line", False))
        force_prob = float(getattr(self.nav_cfg, "goal_force_blocking_prob", 1.0))

        corridor_envs = []
        s1_envs = []
        other_envs = []
        if self.scene_spec_cache is not None:
            for env_id in env_ids.tolist():
                scene_spec = self.scene_spec_cache[env_id]
                if scene_spec is not None and scene_spec.scene_type == "s1_corridor_gate":
                    s1_envs.append(env_id)
                elif self._scene_goal_ranges(scene_spec) is not None:
                    corridor_envs.append(env_id)
                else:
                    other_envs.append(env_id)
        else:
            other_envs = env_ids.tolist()

        for env_id in s1_envs:
            scene_spec = self.scene_spec_cache[env_id]
            if scene_spec is None:
                continue
            params = scene_spec.params or {}
            length = float(params.get("corridor_length", self.cfg.terrain.terrain_length))
            y_start = -0.5 * length
            y_end = 0.5 * length
            goal_buffer = float(params.get("corridor_goal_buffer", self.scene_margin))
            goal_min_offset = float(params.get("corridor_goal_min_offset", min_dist))
            x_center = float(params.get("corridor_x_center", 0.0))
            margin = float(params.get("corridor_goal_margin", self.scene_margin))

            pending = torch.tensor([env_id], device=self.device, dtype=torch.long)
            # 先检查可用 y 区间，避免 y_max <= y_min 导致卡死
            root_world = self.root_states[pending, :2]
            root_local = root_world - self.env_origins[pending, :2]
            y_min = torch.clamp(root_local[:, 1] + goal_min_offset, min=y_start + goal_buffer)
            y_max = torch.full_like(y_min, y_end - goal_buffer)
            if torch.any(y_max <= y_min):
                if env_id not in other_envs:
                    other_envs.append(env_id)
                continue
            for _ in range(max_tries):
                rand_y = torch_rand_float(0.0, 1.0, (1, 1), device=self.device).squeeze(1)
                y_local = y_min + (y_max - y_min) * rand_y
                y_val = float(y_local.item())
                half_w = self._corridor_half_width_at_y(scene_spec, y_val)
                x_min = x_center - (half_w - margin)
                x_max = x_center + (half_w - margin)
                if x_max <= x_min:
                    continue
                rand_x = torch_rand_float(x_min, x_max, (1, 1), device=self.device).squeeze(1)
                goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, y_local], dim=1)
                pos = self.root_states[pending, :2]
                dist = torch.norm(goal_world - pos, dim=1)
                dist_ok = dist >= min_dist
                blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
                if force_blocking and force_prob > 0.0:
                    force_mask = torch.rand_like(dist_ok.float()) < force_prob
                else:
                    force_mask = torch.zeros_like(dist_ok, dtype=torch.bool)
                ok = dist_ok & (blocked | ~force_mask)
                if ok.any():
                    self.goal_world[pending[ok]] = goal_world[ok]
                    pending = pending[~ok]
                    break
            if pending.numel() > 0 and allow_fallback:
                for _ in range(max_tries):
                    rand_y = torch_rand_float(0.0, 1.0, (1, 1), device=self.device).squeeze(1)
                    y_local = y_min + (y_max - y_min) * rand_y
                    y_val = float(y_local.item())
                    half_w = self._corridor_half_width_at_y(scene_spec, y_val)
                    x_min = x_center - (half_w - margin)
                    x_max = x_center + (half_w - margin)
                    if x_max <= x_min:
                        continue
                    rand_x = torch_rand_float(x_min, x_max, (1, 1), device=self.device).squeeze(1)
                    goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, y_local], dim=1)
                    pos = self.root_states[pending, :2]
                    dist = torch.norm(goal_world - pos, dim=1)
                    dist_ok = dist >= min_dist
                    blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
                    if force_blocking and force_prob > 0.0:
                        force_mask = torch.rand_like(dist_ok.float()) < force_prob
                    else:
                        force_mask = torch.zeros_like(dist_ok, dtype=torch.bool)
                    ok = dist_ok & (blocked | ~force_mask)
                    if ok.any():
                        self.goal_world[pending[ok]] = goal_world[ok]
                        pending = pending[~ok]
                        break
            if pending.numel() > 0 and allow_fallback:
                if env_id not in other_envs:
                    other_envs.append(env_id)

        for env_id in corridor_envs:
            scene_spec = self.scene_spec_cache[env_id]
            ranges = self._scene_goal_ranges(scene_spec)
            if ranges is None:
                continue
            range_x_scene, range_y_scene = ranges
            pending = torch.tensor([env_id], device=self.device, dtype=torch.long)
            for _ in range(max_tries):
                rand_x = torch_rand_float(range_x_scene[0], range_x_scene[1], (1, 1), device=self.device).squeeze(1)
                rand_y = torch_rand_float(range_y_scene[0], range_y_scene[1], (1, 1), device=self.device).squeeze(1)
                goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, rand_y], dim=1)
                pos = self.root_states[pending, :2]
                dist = torch.norm(goal_world - pos, dim=1)
                dist_ok = dist >= min_dist
                blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
                if force_blocking and force_prob > 0.0:
                    force_mask = torch.rand_like(dist_ok.float()) < force_prob
                else:
                    force_mask = torch.zeros_like(dist_ok, dtype=torch.bool)
                ok = dist_ok & (blocked | ~force_mask)
                if ok.any():
                    self.goal_world[pending[ok]] = goal_world[ok]
                    pending = pending[~ok]
                    break
            if pending.numel() > 0 and allow_fallback:
                for _ in range(max_tries):
                    rand_x = torch_rand_float(range_x_scene[0], range_x_scene[1], (1, 1), device=self.device).squeeze(1)
                    rand_y = torch_rand_float(range_y_scene[0], range_y_scene[1], (1, 1), device=self.device).squeeze(1)
                    goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, rand_y], dim=1)
                    pos = self.root_states[pending, :2]
                    dist = torch.norm(goal_world - pos, dim=1)
                    dist_ok = dist >= min_dist
                    blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
                    if force_blocking and force_prob > 0.0:
                        force_mask = torch.rand_like(blocked.float()) < force_prob
                    else:
                        force_mask = torch.zeros_like(blocked, dtype=torch.bool)
                    ok = dist_ok & (blocked | ~force_mask)
                    if ok.any():
                        self.goal_world[pending[ok]] = goal_world[ok]
                        pending = pending[~ok]
                        break

        if not other_envs:
            return

        pending = torch.tensor(other_envs, device=self.device, dtype=torch.long)
        for _ in range(max_tries):
            if pending.numel() == 0:
                break
            num = pending.shape[0]
            rand_x = torch_rand_float(range_x[0], range_x[1], (num, 1), device=self.device).squeeze(1)
            rand_y = torch_rand_float(range_y[0], range_y[1], (num, 1), device=self.device).squeeze(1)
            goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, rand_y], dim=1)
            pos = self.root_states[pending, :2]
            dist = torch.norm(goal_world - pos, dim=1)
            dist_ok = dist >= min_dist
            blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
            if force_blocking and force_prob > 0.0:
                force_mask = torch.rand_like(dist_ok.float()) < force_prob
            else:
                force_mask = torch.zeros_like(dist_ok, dtype=torch.bool)
            ok = dist_ok & (blocked | ~force_mask)
            if ok.any():
                self.goal_world[pending[ok]] = goal_world[ok]
            pending = pending[~ok]

        if pending.numel() > 0 and allow_fallback:
            for _ in range(max_tries):
                if pending.numel() == 0:
                    break
                num = pending.shape[0]
                rand_x = torch_rand_float(range_x[0], range_x[1], (num, 1), device=self.device).squeeze(1)
                rand_y = torch_rand_float(range_y[0], range_y[1], (num, 1), device=self.device).squeeze(1)
                goal_world = self.env_origins[pending, :2] + torch.stack([rand_x, rand_y], dim=1)
                pos = self.root_states[pending, :2]
                dist = torch.norm(goal_world - pos, dim=1)
                dist_ok = dist >= min_dist
                blocked = self._line_has_obstacle(pos, goal_world, env_ids=pending)
                if force_blocking and force_prob > 0.0:
                    force_mask = torch.rand_like(blocked.float()) < force_prob
                else:
                    force_mask = torch.zeros_like(blocked, dtype=torch.bool)
                ok = dist_ok & (blocked | ~force_mask)
                if ok.any():
                    self.goal_world[pending[ok]] = goal_world[ok]
                pending = pending[~ok]

    def _init_buffers(self):
        super()._init_buffers()
        #额外添加专家参考动作
        self.expert_actions = torch.zeros(self.num_envs,self.num_actions,dtype=torch.float,device=self.device,requires_grad=False)
        #Additional add buffers for base_lin_acc
        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        #额外添加IMU安装处的加速度
        self.IMU_lin_acc = torch.zeros_like(self.base_lin_vel)
        #额外添加IMU质心在机器人坐标系下的坐标 #TODO 放到cfg参数中
        self.IMU_pos = torch.tensor([[0,-0.015,0.039625]],dtype=torch.float,device=self.device,requires_grad=False)
        #额外添加角加速度
        self.base_ang_acc = torch.zeros_like(self.base_ang_vel)
        #额外添加上一次的接触力
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        # 高层导航缓冲
        self.robot_state_buf = torch.zeros(
            self.num_envs, 9, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.goal_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.goal_world = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Moving target buffers (S0 mainline; always allocated for simplicity).
        self.target_world = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_vel_world = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_heading = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_heading_des = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_speed = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_speed_des = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_cmd_timer = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Freeze timer after reset to keep the target stationary (S0 learnability).
        self.target_freeze_timer = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_speed_phase = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Per-step diagnostics for moving-target behavior.
        self.target_turn_events = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_preturn_events = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_reflect_events = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # Per-episode counters (reset in reset_idx for corresponding envs).
        self.target_turn_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_preturn_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_reflect_count = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_reset_dist_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.target_reset_bearing_error = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        # S1 moving target scripted gate traversal (always allocated; used only when enabled).
        self._s1_gate_max = 4
        self.s1_gate_y = torch.zeros(
            self.num_envs, self._s1_gate_max, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_gate_len = torch.zeros(
            self.num_envs, self._s1_gate_max, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_gate_door_half = torch.zeros(
            self.num_envs, self._s1_gate_max, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_gate_bias_x = torch.zeros(
            self.num_envs, self._s1_gate_max, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_gate_post_side = torch.ones(
            self.num_envs, self._s1_gate_max, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_gate_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.s1_gate_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.s1_path_dir = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_corridor_half_len = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.s1_corridor_half_w = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        #设置记录观测的最大和最小和平均值，用于判断观测归一化合理程度
        # self.priv_obs_min = torch.zeros_like(self.privileged_obs_buf[0])
        # self.priv_obs_max = torch.zeros_like(self.privileged_obs_buf[0])
        # self.priv_obs_mean = torch.zeros_like(self.privileged_obs_buf[0])

        #添加用于更新刚体位置的
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs,-1,13)

        #用于标记摆动过程中是否达到swing_init_point附近
        self.reach_swing_init = torch.zeros(self.num_envs,6,dtype=torch.bool,device=self.device)
        self.reach_stance_init = torch.zeros(self.num_envs,6,dtype=torch.bool,device=self.device)
        #用于标记获取reach奖励的时间
        self.reach_rew_time = torch.zeros(self.num_envs,6,dtype=torch.float,device=self.device)
        # #记录swing_init_pos的角度位置信息
        # self.default_dof_swing_pos = torch.zeros(self.num_actions,dtype=torch.float,device=self.device)
        # # TODO 从cfg中读取这部分参数
        # for i in range(self.num_dofs):
        #     name=self.dof_names[i]
        #     angle=self.cfg.init_state.default_swing_init_angles[name]
        #     self.default_dof_swing_pos[i]=angle

        # #
        # self.default_dof_swing_pos = self.default_dof_swing_pos.unsqueeze(0)


    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        if self.privileged_obs_buf is None:
            noise_vec = torch.zeros_like(self.obs_buf[0])
        else:
            noise_vec = torch.zeros_like(self.privileged_obs_buf[0])

        print("---------->noise_vec.shape=",noise_vec.shape)

        #[quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)]
        # noise_vec[:4] = noise_level * noise_scales.quat * self.obs_scales.quat
        # noise_vec[4:7] = noise_level * noise_scales.ang_vel * self.obs_scales.ang_vel
        # noise_vec[7:10] = noise_level * noise_scales.lin_acc * self.obs_scales.lin_acc
        # noise_vec[10:28] = noise_level * noise_scales.dof_pos * self.obs_scales.dof_pos
        # noise_vec[28:46] = noise_level * noise_scales.dof_vel * self.obs_scales.dof_vel
        # noise_vec[46:64] = noise_level * noise_scales.dof_torque * self.obs_scales.dof_torque
        # noise_vec[64:67] = 0.0 #command
        #[last_action(18), dof_pos(18), dof_vel(18), dof_torque(18), command(3), measured_hegiths(143)]
        noise_vec[:18]=0.0
        noise_vec[18:36]=noise_level * noise_scales.dof_pos*self.obs_scales.dof_pos
        noise_vec[36:54]=noise_level * noise_scales.dof_vel*self.obs_scales.dof_vel
        noise_vec[54:72]=noise_level * noise_scales.dof_torque*self.obs_scales.dof_torque
        noise_vec[72:75]=0.0
        #地形信息actor也可以拿到
        # noise_vec[75:] = noise_level * noise_scales.height_measurements * self.obs_scales.height_measurements
        
        if self.privileged_obs_buf is not None:
            #[lin_vel(3), gravity(3), contact_force(6) ,measured_heights(187)]
            noise_vec[75:78] = noise_level * noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[78:81] = noise_level * noise_scales.gravity * self.obs_scales.gravity
            noise_vec[81:87] = noise_level * noise_scales.contact_force * self.obs_scales.contact_force
            noise_vec[87:] = noise_level * noise_scales.height_measurements * self.obs_scales.height_measurements

            # noise_vec[67:70] = noise_level * noise_scales.lin_vel * self.obs_scales.lin_vel
            # noise_vec[70:73] = noise_level * noise_scales.gravity * self.obs_scales.gravity
            # noise_vec[73:79] = noise_level * noise_scales.contact_force * self.obs_scales.contact_force
            # noise_vec[79:] = noise_level * noise_scales.height_measurements * self.obs_scales.height_measurements

            # noise_vec[75:] = noise_level * noise_scales.height_measurements * self.obs_scales.height_measurements
        return noise_vec
    
    def _resample_commands(self, env_ids):
        for i, key in enumerate(['lin_vel_x','lin_vel_y','ang_vel_yaw']):
            self.commands[env_ids, i] = torch_rand_float(self.command_ranges[key][0], self.command_ranges[key][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids,i] *= torch.abs(self.commands[env_ids,i])>0.05
            x=self.commands[env_ids,i]

            x[x<self.command_ranges[key][0]*0.8]=self.command_ranges[key][0]
            x[x>self.command_ranges[key][1]*0.8]=self.command_ranges[key][1]
            self.commands[env_ids,i]=x   
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.1).unsqueeze(1)

    def _update_terrain_curriculum(self, env_ids):
        #重新设计地形更新的规则
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :3] - self.env_origins[env_ids, :3], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.3) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]        
    def update_command_curriculum(self, env_ids):
        #对 vx vy omega都进行阶段性更新
        ave_lin_rew = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])/self.max_episode_length
        ave_ang_rew = torch.mean(self.episode_sums["tracking_ang_vel"][env_ids])/self.max_episode_length
        if ave_lin_rew > 0.8 * self.reward_scales["tracking_lin_vel"]:
            #升级 vx vy
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1,self.cfg.commands.ranges.lin_vel_x[0],0)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.15, self.cfg.commands.ranges.lin_vel_y[0], 0)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0,self.cfg.commands.ranges.lin_vel_x[1])
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.15, 0,self.cfg.commands.ranges.lin_vel_y[1])
        elif ave_lin_rew < 0.6 * self.reward_scales["tracking_lin_vel"]:
            #降级 vx vy
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] + 0.1, -10,-0.2)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] + 0.15, -10,-0.3)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] - 0.1, 0.2,10)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] - 0.15, 0.3,10)

        if ave_ang_rew > 0.8 * self.reward_scales["tracking_ang_vel"]:
            #升级 omega
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.3, self.cfg.commands.ranges.ang_vel_yaw[0],0)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.3, 0,self.cfg.commands.ranges.ang_vel_yaw[1])
        elif ave_ang_rew < 0.6 * self.reward_scales["tracking_ang_vel"]:
            #降级 omega
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] + 0.3, -10,-0.6)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] - 0.3, 0.6,10)
    
    def _draw_foot_end_trajectory(self):
        # self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        feet_states=self.rb_states[:,self.feet_indices,:]
        colors = [(1,1,0), (1,0,0), (0,1,0), (0,0,1), (1,0,1), (0,1,1)]
        sphere_geoms = [gymutil.WireframeSphereGeometry(0.005, 4, 4, color=c) for c in colors]
        
        for i in range(self.num_envs):
            for j in range(6):
                x=feet_states[i,j][0]
                y=feet_states[i,j][1]
                z=feet_states[i,j][2]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geoms[j], self.gym, self.viewer, self.envs[i], sphere_pose) 


    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        #重设六足的足端腾空时间不少于0.18s
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.last_contacts = contact #放到post_physics_step后面计算，因为reward中还有其他奖励要使用
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # print("feet_air_time=",self.feet_air_time[0])
        # print("first_contact=",first_contact[0])
        rew_airTime = torch.sum((self.feet_air_time - 0.18) * first_contact, dim=1) # reward only on first contact with the ground
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.2)
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > cmd_threshold # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_footend_pos_xy(self):
        """单次分段奖励"""
        # # swing或者stance阶段，只要靠近一次init point就给奖励，只给一次
        # # 区分swing或者stance
        # contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts) 
        # self.expert.kin.ForwardKin(self.dof_pos.view(-1,3),self.expert.B_e_cur_flat)
        # dist=torch.norm(self.expert.B_e_cur[...,0:2]-self.expert.swing_init_point[:,0:2],dim=-1)

        # self.reach_stance_init[~contact_filt]=False
        # self.reach_swing_init[contact_filt]=False

        # # rew=torch.exp(-dist/(0.14*0.2))
        # reach=(dist<0.01) & (~contact_filt)
        # #靠近范围 没有获得过 距离上一次获得奖励时间高于0.18s
        # get_rew_mask = reach & (~self.reach_swing_init) & (self.reach_rew_time>0.22)
        # # rew[~(reach &(~self.reach_swing_init))] =0.0
        # swing_reward = torch.sum( get_rew_mask, dim=1)
        # self.reach_swing_init[reach]=True
        # self.reach_rew_time[self.reach_swing_init] += self.dt
        # self.reach_rew_time[get_rew_mask]=0.0



        # reach=(dist<0.015) & (contact_filt)
        # get_rew_mask = reach & (~self.reach_stance_init) & (self.reach_rew_time>0.25)
        # # rew[~(reach &(~self.reach_stance_init))] =0.0
        # stance_reward = torch.sum( get_rew_mask, dim=1)
        # self.reach_stance_init[reach]=True
        # self.reach_rew_time[self.reach_stance_init] += self.dt
        # self.reach_rew_time[get_rew_mask]=0.0

        # for i in range(6):
        #     print(f"gaits={float(self.expert.gaits[0,i])}, dist={dist[0,i]}, reach_stance={self.reach_stance_init[0,i]}")
        # print(f"stance_rew={stance_reward[0]}")
        # print("\n")
        # time.sleep(0.5)
        # rew = (stance_reward+swing_reward) * (torch.norm(self.commands[:,:3])>0.2)
        # rew = (swing_reward) * (torch.norm(self.commands[:,:3])>0.2)
        # return rew

        """持续奖励"""
        xy_dist=torch.norm(self.expert.B_e_cur[...,0:2]-self.expert.swing_init_point[:,0:2],dim=-1).sum(dim=-1)
        # xy_dist=(xy_dist*(~contact_filt)).sum(dim=1) #只计算swing状态下的contact_filt
        # xy_dist[xy_dist<0.2]=0.2
        # print("xy_dist=",xy_dist[0])
        # rew=(torch.exp(-xy_dist/0.12)*(~contact_filt))/(torch.sum(~contact_filt,dim=1)+1e-6)
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.2)
        return torch.exp(-xy_dist/(0.4*0.5)) * (torch.norm(self.commands[:, :3], dim=1) > cmd_threshold)

    def _reward_swing(self):
        #估计摆动时，靠近设置的初始点，来避免长期运动带来的累计误差
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        # 
        # print("reach_swing_init=",self.reach_swing_init[0])
        # print("contact_filt=",contact_filt[0])
        self.reach_swing_init[contact_filt]=False
        err=torch.norm( (self.dof_pos-self.default_dof_swing_pos).view(self.num_envs,6,3), dim=-1)
        # print("dof_pos\n",self.dof_pos[0].view(6,3))
        # print("defualt_swing_pos\n",self.default_dof_swing_pos[0].view(6,3))
        # print("err=",err[0])
        # print("--------------\n")
        self.reach_swing_init[err<0.15]=True
        
        # swing && 没有达到过目标点
        reaching_mask=(~contact_filt) & (~self.reach_swing_init) #N*6
        # swing && 达到目标带你
        leaving_mask=(~contact_filt) & (self.reach_swing_init)

        weight=torch.sigmoid((err-0.15)*10)
        weight[leaving_mask] = 1.0-weight[leaving_mask]

        reaching_rew=torch.exp(-(err+0.2)/0.6)
        leaving_rew=torch.tanh((err+0.2)/0.6)

        smooth_rew=weight*reaching_rew+(1.0-weight)*leaving_rew
        smooth_rew[contact_filt]=0.0
        smooth_rew=torch.sum(smooth_rew,dim=1)/(torch.sum(~contact_filt,dim=1)+1e-6)
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.2)
        smooth_rew *= torch.norm(self.commands[:,:3],dim=1) > cmd_threshold
        return smooth_rew
        
        # print("valid_mask=",reaching_mask[0])
        # print("err=",err[0])
        rew = torch.exp(-err/0.8)*reaching_mask
        reached_mask=(~contact_filt) & self.reach_swing_init
        rew[reached_mask]=math.exp(-0.15/0.8)
        # print("rew\n",rew[0])
        leg_rew=torch.sum(rew,dim=1)/(torch.sum(~contact_filt,dim=1)+1e-6)
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.2)
        leg_rew *= torch.norm(self.commands[:,:3],dim=1) > cmd_threshold
        #命令为0时，设置静止

        # print("leg_rew=",leg_rew[0])
        # print("---------------\n")
        return leg_rew
    
    def _reward_mirror(self):
        #LB[0:3] LF[3:6] LM[6:9] RB[9:12] RF[12:15] RM[15:18]
        # print("LB LF action dist=",self.actions[:,0]-self.actions[:,3])
        # print("RB RF action dist=",torch.norm(self.actions[:,9:12]-self.actions[:,12:15],dim=1,p=1))
        dist = torch.norm(self.actions[:,0:3]-self.actions[:,3:6],dim=1,p=1)+\
               torch.norm(self.actions[:,9:12]-self.actions[:,12:15],dim=1,p=1)
        return dist

    
    def _reward_stand_still(self):
        # print("default_dof_pos\n",self.default_dof_pos)
        # print("self.dof_pos\n",self.dof_pos[0])
        # print("err=",torch.abs(self.dof_pos-self.default_dof_pos)[0])
        # print("err sum=",torch.sum(torch.abs(self.dof_pos-self.default_dof_pos),dim=1)[0])
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.2)
        return torch.sum(torch.abs(self.dof_pos-self.default_dof_pos),dim=1)\
            *(torch.norm(self.commands[:,:3],dim=1) < cmd_threshold)

    def _zero_cmd_mask(self):
        cmd_threshold = getattr(self.cfg.rewards, "zero_cmd_threshold", 0.1)
        return torch.norm(self.commands[:, :3], dim=1) < cmd_threshold

    def _reward_zero_cmd_feet_contact(self):
        mask = self._zero_cmd_mask()
        if not torch.any(mask):
            return torch.zeros_like(mask, dtype=torch.float)
        contact_threshold = getattr(self.cfg.rewards, "zero_cmd_contact_force_threshold", 1.0)
        contact = self.contact_forces[:, self.feet_indices, 2] > contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)
        all_contact = contact_filt.all(dim=1)
        penalty = (~all_contact).float()
        return penalty * mask.float()

    def _reward_zero_cmd_dof_vel(self):
        mask = self._zero_cmd_mask()
        if not torch.any(mask):
            return torch.zeros_like(mask, dtype=torch.float)
        eps = getattr(self.cfg.rewards, "zero_cmd_dof_vel_eps", 0.05)
        excess = torch.clamp(torch.abs(self.dof_vel) - eps, min=0.0)
        penalty = torch.max(excess, dim=1).values
        return penalty * mask.float()

    def _reward_zero_cmd_action_rate(self):
        mask = self._zero_cmd_mask()
        if not torch.any(mask):
            return torch.zeros_like(mask, dtype=torch.float)
        eps = getattr(self.cfg.rewards, "zero_cmd_action_rate_eps", 0.05)
        excess = torch.clamp(torch.abs(self.actions - self.last_actions) - eps, min=0.0)
        penalty = torch.max(excess, dim=1).values
        return penalty * mask.float()

    def _reward_zero_cmd_base_vel(self):
        mask = self._zero_cmd_mask()
        if not torch.any(mask):
            return torch.zeros_like(mask, dtype=torch.float)
        lin_eps = getattr(self.cfg.rewards, "zero_cmd_base_lin_vel_eps", 0.05)
        ang_eps = getattr(self.cfg.rewards, "zero_cmd_base_ang_vel_eps", 0.05)
        lin_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        ang_speed = torch.abs(self.base_ang_vel[:, 2])
        lin_excess = torch.clamp(lin_speed - lin_eps, min=0.0)
        ang_excess = torch.clamp(ang_speed - ang_eps, min=0.0)
        return (lin_excess + ang_excess) * mask.float()

    def _reward_base_height(self):
        #修改成正的奖励，越靠近目标值，奖励越高
        # print("in reward_base_height, base_height=",torch.mean(self.root_states[:,2].unsqueeze(1)-self.measured_heights,dim=1))
        # print("robot_states z=",self.root_states[0,2])
        # print("self.measured_heights=",self.measured_heights.mean())
        height = torch.clip(self.root_states[:,2].unsqueeze(1)-0.025-self.measured_heights, min=-1, max=1.0)
        base_height = torch.mean(height,dim=1)
        err = torch.abs(base_height-self.cfg.rewards.base_height_target)
        # print("err = ",err)
        reward = torch.exp(-err/0.04)
        low_height_threshold = getattr(self.cfg.rewards, "low_height_penalty_threshold", None)
        if low_height_threshold is not None:
            low_height_penalty = getattr(self.cfg.rewards, "low_height_penalty_value", -1.0)
            reward = torch.where(base_height < low_height_threshold, torch.full_like(reward, low_height_penalty), reward)
        return reward
        # return torch.abs(base_height-self.cfg.rewards.base_height_target)
        # return super()._reward_base_height()

    def _reward_feet_contact_forces_increase(self):
        # print("fee_contact_force=\n",self.contact_forces[0,self.feet_indices,:])
        # print("last fee_contact_force=\n",self.last_contact_forces[0,self.feet_indices,:])


        feet_force_dt = (self.contact_forces - self.last_contact_forces)[:,self.feet_indices,:]
        feet_force_dt *= (feet_force_dt>0.0) #只获取增加的接触力，也就是与足端与地面碰撞时
        # print("feet_force_dt\n",feet_force_dt[0])
        # print("feet_force_delt \n",feet_force_dt[0])
        feet_force_dt = torch.norm(feet_force_dt,dim=-1)
        # print("torch.norm(feet_force_dt)\n",feet_force_dt[0])
        # print("feet_force_dt \n",feet_force_dt[0])
        self.last_contact_forces = self.contact_forces.clone()
        return feet_force_dt.sum(dim=1)

    def _reward_CoT(self):
        return torch.sum(torch.abs(self.torques*self.dof_vel),dim=1)

    def _reward_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:,self.feet_indices,:2],dim=2)>\
                         torch.abs(self.contact_forces[:,self.feet_indices,2]),dim=1)

    def _reward_camera_wobble_y(self):
        return torch.square(self.base_ang_vel[:, 1])

    # def _reward_tracking_lin_vel(self):
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     lin_vel_error *= lin_vel_error>0.1 #小于0.1的速度误差对机器人来说一样，可以鼓励优化其他部分而不是牺牲自然状态追求高精度的速度跟踪
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw) 
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     ang_vel_error *=ang_vel_error>0.2
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)    
    def _reward_tracking_dof(self):
        action_scaled = self.actions * self.cfg.control.action_scale
        pos_err = (action_scaled+self.default_dof_pos) - self.dof_pos
        pos_err *= pos_err>0.15

        return torch.square(pos_err).sum(dim=1)
        

if __name__ == '__main__':
    args = get_args()
    cfg = HexDebugPlaneCfg()
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = HexGround(cfg, sim_params, args.physics_engine, args.sim_device, args.headless)
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step(torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device))
        
