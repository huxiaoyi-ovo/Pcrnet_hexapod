
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO
from legged_gym.envs.hex_v4.hex_scenes_config import HexDebugPlaneCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertGround
from legged_gym.envs.hex_v4.scene_spec import SceneSpec
import torch
import numpy as np
from typing import Optional

from isaacgym import gymtorch,gymapi,gymutil
from legged_gym.utils import get_args,class_to_dict
from legged_gym.utils.helpers import parse_sim_params
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
        if hasattr(cfg, "sensor") and hasattr(cfg.sensor, "depth_camera"):
            self.camera_cfg = cfg.sensor.depth_camera
            self.enable_camera = bool(self.camera_cfg.enable)
            self._use_camera_in_headless = headless and self.enable_camera
        terrain_type = getattr(cfg.terrain, "terrain_type", None) if hasattr(cfg, "terrain") else None
        debug_allow_plane = bool(getattr(cfg.terrain, "debug_allow_plane", False))
        mesh_type = getattr(cfg.terrain, "mesh_type", None)
        if debug_allow_plane:
            if mesh_type not in ("plane", "none"):
                raise RuntimeError("debug_allow_plane requires cfg.terrain.mesh_type='plane' (or 'none').")
        else:
            if mesh_type != "heightfield":
                raise RuntimeError("hex_ground requires cfg.terrain.mesh_type='heightfield' for classic terrain.")
            if not terrain_type:
                raise RuntimeError(
                    "hex_ground 是容器任务，必须显式设置 terrain_type。"
                    "建议使用: --task hex_s1 或 --task hex_debug_plane。"
                    "示例: python legged_gym/scripts/train.py --task hex_s1 --num_envs 2048; "
                    "或 python legged_gym/scripts/train_highlevel.py "
                    "--mode teacher --skill follow --task hex_s1 --low_level_ckpt agents/fast_2000.pt"
                )
        super().__init__(cfg,sim_params,physics_engine,sim_device,headless)
        self.cfg:HexGroundCfg = cfg
        self.nav_cfg = getattr(cfg, "navigation", None)
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
        if self.viewer and self.enable_viewer_sync and self.debug_viz and self._moving_target_enabled():
            self._debug_draw_target_and_robot_trajectories()
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
    
    def _pre_create_envs(self):
        self.robot_actor_indices = np.zeros(self.num_envs, dtype=np.int32)
        self.dynamic_actor_handles = None
        self.dynamic_actor_indices = None
        self.dynamic_asset = None
        terrain_obj = getattr(self, "terrain", None)
        self.scene_generator = getattr(terrain_obj, "scene_generator", None)
        if self.scene_generator is None or not self.scene_generator.has_dynamic:
            return
        max_dyn = int(getattr(self.cfg.terrain, "scene_dynamic_max", 0))
        if max_dyn <= 0:
            return
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        size_xy = float(getattr(self.cfg.terrain, "scene_dynamic_size", 0.35))
        height = float(getattr(self.cfg.terrain, "scene_dynamic_height", 0.5))
        self.dynamic_asset = self.gym.create_box(self.sim, size_xy, size_xy, height, asset_options)
        self.dynamic_actor_handles = [[None for _ in range(max_dyn)] for _ in range(self.num_envs)]
        self.dynamic_actor_indices = np.zeros((self.num_envs, max_dyn), dtype=np.int32)

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
        return env_id + 1

    def _scene_collision_filter(self) -> int:
        scene_filter = int(getattr(self.cfg.terrain, "scene_collision_filter", 0xFFFFFFFF))
        if scene_filter >= (1 << 31):
            scene_filter = -1
        return scene_filter

    def _apply_actor_collision_filter(self, env_handle, actor_handle, target_filter: int, env_id: int, debug_tag: str = ""):
        try:
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            for prop in shape_props:
                prop.filter = target_filter
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            if getattr(self, "debug_viz", False) and env_id == 0 and debug_tag:
                flag = f"_{debug_tag}_filter_logged"
                if not getattr(self, flag, False):
                    print(f"[Debug] {debug_tag} shape filter={target_filter}")
                    setattr(self, flag, True)
        except Exception:
            if getattr(self, "debug_viz", False) and debug_tag:
                flag = f"_{debug_tag}_filter_warned"
                if not getattr(self, flag, False):
                    print(f"[Debug] {debug_tag} shape filter update failed")
                    setattr(self, flag, True)

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
        if self.dynamic_asset is None or self.dynamic_actor_indices is None:
            return
        group_id = self._scene_group_id(env_id)
        scene_filter = self._scene_collision_filter()
        create_filter = scene_filter
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

    def _get_scene_difficulty(self, env_id: int) -> float:
        if hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level"):
            denom = max(1, int(self.max_terrain_level))
            return float(self.terrain_levels[env_id].item()) / denom
        return 0.0

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
        return str(mode)

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

    def _reset_moving_target(self, env_ids: torch.Tensor):
        """Reset moving target state for selected envs."""
        if env_ids.numel() == 0 or not self._moving_target_enabled():
            return
        if self._moving_target_mode() == "s1_gate_script":
            self._reset_moving_target_s1(env_ids)
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
            if hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level"):
                denom = max(1, int(self.max_terrain_level))
                d = self.terrain_levels.float() / float(denom)
            else:
                d = torch.zeros(self.num_envs, device=device)
            d = torch.clamp(d, 0.0, 1.0)
            self._update_moving_target_s1(dt=dt, d=d)
            return

        difficulty_override = getattr(self, "scene_difficulty_override", None)
        terrain_type = str(getattr(self.cfg.terrain, "terrain_type", "")).lower()
        if difficulty_override is not None and terrain_type in ("s0_follow_plane", "s0"):
            if torch.is_tensor(difficulty_override):
                difficulty_tensor = difficulty_override.to(device=device, dtype=torch.float32)
                if difficulty_tensor.ndim == 0:
                    difficulty_value = float(difficulty_tensor.item())
                    difficulty_value = float(np.clip(difficulty_value, 0.0, 1.0))
                    d = torch.full((self.num_envs,), difficulty_value, device=device)
                elif difficulty_tensor.numel() == self.num_envs:
                    d = torch.clamp(difficulty_tensor.reshape(-1), 0.0, 1.0)
                else:
                    raise ValueError(
                        f"scene_difficulty_override shape mismatch: {tuple(difficulty_tensor.shape)}"
                    )
            else:
                difficulty_value = float(difficulty_override)
                difficulty_value = float(np.clip(difficulty_value, 0.0, 1.0))
                d = torch.full((self.num_envs,), difficulty_value, device=device)
        elif hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level"):
            denom = max(1, int(self.max_terrain_level))
            d = self.terrain_levels.float() / float(denom)
        else:
            d = torch.zeros(self.num_envs, device=device)
        d = torch.clamp(d, 0.0, 1.0)

        # Freeze after reset for early-stage learnability.
        if hasattr(self, "target_freeze_timer"):
            frozen = self.target_freeze_timer > 0.0
            if frozen.any():
                self.target_freeze_timer = torch.clamp(self.target_freeze_timer - dt, min=0.0)
                self.target_vel_world[frozen].zero_()
                self.goal_world[frozen] = self.target_world[frozen]
                if (~frozen).any():
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
        self.target_cmd_timer -= dt
        need_cmd = (self.target_cmd_timer <= 0.0) | preturn_needed
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
        self.target_heading = self.target_heading + dtheta
        self.target_heading = torch.atan2(torch.sin(self.target_heading), torch.cos(self.target_heading))

        # Smoothly change speed with a bounded low-frequency wave (avoid long constant-speed segments).
        wave_rate = speed_wave_rate_slow + (speed_wave_rate_fast - speed_wave_rate_slow) * d
        self.target_speed_phase = self.target_speed_phase + wave_rate * dt
        self.target_speed_phase = torch.atan2(torch.sin(self.target_speed_phase), torch.cos(self.target_speed_phase))
        wave_amp = speed_wave_amp_cfg * (0.25 + 0.75 * d)
        speed_des_wave = torch.clamp(self.target_speed_des + wave_amp * torch.sin(self.target_speed_phase), v_min, v_max)

        dv = speed_des_wave - self.target_speed
        max_dv = (0.4 + 0.6 * d) * accel_max * dt
        dv = torch.clamp(dv, -max_dv, max_dv)
        self.target_speed = torch.clamp(self.target_speed + dv, 0.0, v_max)

        # Integrate position in local env frame.
        dir_x = torch.sin(self.target_heading)
        dir_y = torch.cos(self.target_heading)
        vel_local = torch.stack([self.target_speed * dir_x, self.target_speed * dir_y], dim=-1)
        pos_local = pos_local_curr + vel_local * dt

        # Keep within bounds. Reflection remains as a fallback only.
        hit_x = (pos_local[:, 0] < x_min) | (pos_local[:, 0] > x_max)
        hit_y = (pos_local[:, 1] < y_min) | (pos_local[:, 1] > y_max)
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
            dx = abs(x_local - float(spec.position[0]))
            dy = abs(y_local - float(spec.position[1]))
            limit_x = 0.5 * float(spec.size[0]) + clearance
            limit_y = 0.5 * float(spec.size[1]) + clearance
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
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > contact_threshold, dim=1)

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

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        
      


    def reset_idx(self, env_ids:torch.Tensor):
        prev_levels = None
        if hasattr(self, "terrain_levels") and len(env_ids) > 0:
            prev_levels = self.terrain_levels[env_ids].clone()
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
            self._resample_nav_goals(env_ids)
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
                        x_r = cos_h * delta[:, 0] - sin_h * delta[:, 1]
                        y_f = sin_h * delta[:, 0] + cos_h * delta[:, 1]
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
                            x_r = cos_h * delta[:, 0] - sin_h * delta[:, 1]
                            y_f = sin_h * delta[:, 0] + cos_h * delta[:, 1]
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

                if other_ids is not None and other_ids.numel() > 0:
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
            # Default behavior: do NOT clear global viewer lines on each reset.
            # Global clear can make trajectories disappear before the currently viewed env resets.
            if self.viewer is not None and clear_on_reset:
                need_clear = False
                if bool(getattr(self, "foot_traj_viz", False)):
                    need_clear = True
                if bool(getattr(self, "debug_viz", False)) and self._moving_target_enabled():
                    need_clear = True
                if need_clear:
                    self.gym.clear_lines(self.viewer)
            # Reset per-env debug trajectory state to avoid cross-episode line segments.
            if hasattr(self, "_viz_prev_valid"):
                ids = env_ids.detach().cpu().numpy()
                self._viz_prev_valid[ids] = False
                if clear_on_reset and getattr(self, "debug_viz", False) and self.viewer is not None and self._moving_target_enabled():
                    self._viz_prev_valid[:] = False
                    if hasattr(self, "_viz_traj_tick"):
                        self._viz_traj_tick = 0
        
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
        # Build world-frame right/forward unit vectors from heading and project delta.
        # heading=0 => forward=(0,+1), right=(+1,0).
        x_right = cos_h * delta_world[:, 0] - sin_h * delta_world[:, 1]
        y_forward = sin_h * delta_world[:, 0] + cos_h * delta_world[:, 1]
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
        
