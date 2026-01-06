
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_ground_config import HexGroundCfg, HexGroundCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertGround
import torch
import numpy as np

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
        super().__init__(cfg,sim_params,physics_engine,sim_device,headless)
        self.cfg:HexGroundCfg = cfg
        self.nav_cfg = getattr(cfg, "navigation", None)
        self.debug_viz = False
        self.foot_traj_viz=False
        #额外初始化电机类，可以计理想力矩或模拟的仿真力矩
        self.actuator=Actuator(self.cfg,self.device)
        #额外初始化专家类，可以在step时，提供专家动作参考
        # if self.cfg.env.gen_expert_actions:
        self.expert=ExpertGround(self.cfg,self.device,self.cfg.env.num_envs)

        #额外初始化相机类
        cam_prop=gymapi.CameraProperties()
        # print("sim_params.use_gpu_pipline=",sim_params.use_gpu_pipline)
        if self.camera_cfg is not None:
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
    
    def _create_envs(self):
        super()._create_envs()
        #打印一些值进行查看

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

    def _reset_root_states(self, env_ids):
        """重置root状态，slalom地形时固定入口位姿"""
        if len(env_ids) == 0:
            return
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
            if hasattr(self, "terrain_types"):
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
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )


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

        if self.camera_cfg is not None:
            if not self.enable_camera:
                self.depth_raw.fill_(self.camera_cfg.far_clip)
                processed = self._process_depth_for_network(self.depth_raw)
                self.depth_images[:] = processed
            elif self.common_step_counter % self.camera_cfg.capture_interval == 0:
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
    

    def post_physics_step(self):
        #添加了base_lin_acc的计算，添加了IMU加速度计算，添加了分开式的观测计算,所以需要重写基类的这个函数
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

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
        super().reset_idx(env_ids)

        if len(env_ids) !=0:
            self._resample_nav_goals(env_ids)
            if hasattr(self, "goal_world"):
                # 出生时朝向目标点（不使用 yaw 扰动）
                goal_delta = self.goal_world[env_ids] - self.root_states[env_ids, :2]
                yaw = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
                qz = torch.sin(0.5 * yaw)
                qw = torch.cos(0.5 * yaw)
                quat = torch.stack([torch.zeros_like(qz), torch.zeros_like(qz), qz, qw], dim=1)
                self.root_states[env_ids, 3:7] = quat
                env_ids_int32 = env_ids.to(dtype=torch.int32)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
            self.get_expert_actions()
            #可视化的轨迹线条清楚
            if self.viewer and self.foot_traj_viz:
                self.gym.clear_lines(self.viewer)            
        
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
        env_ids_int32=env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))

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
            self.obs_buf += (2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec[:self.num_obs]
            self.obs_vgf_buf += (2*torch.rand_like(self.obs_vgf_buf)-1)*self.noise_scale_vec[self.num_obs : self.num_obs+12]
            self.obs_terrain_buf += (2*torch.rand_like(self.obs_terrain_buf)-1)*self.noise_scale_vec[self.num_obs+12:]

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
        dx_body = cos_h * delta_world[:, 0] + sin_h * delta_world[:, 1]
        dy_body = -sin_h * delta_world[:, 0] + cos_h * delta_world[:, 1]
        self.goal_buf[:] = torch.stack([dx_body, dy_body], dim=1)

        if getattr(self.nav_cfg, "resample_on_reach", False):
            dist = torch.norm(delta_world, dim=1)
            reached = dist < getattr(self.nav_cfg, "goal_reached_threshold", 0.5)
            if reached.any():
                self._resample_nav_goals(reached.nonzero(as_tuple=False).flatten())

    def _line_has_obstacle(self, start_xy: torch.Tensor, goal_xy: torch.Tensor) -> torch.Tensor:
        if self.height_samples is None:
            return torch.zeros(start_xy.shape[0], dtype=torch.bool, device=self.device)
        samples = int(getattr(self.nav_cfg, "goal_line_samples", 16))
        if samples <= 0:
            return torch.zeros(start_xy.shape[0], dtype=torch.bool, device=self.device)

        t = torch.linspace(0.0, 1.0, samples, device=self.device).view(1, -1, 1)
        points = start_xy.unsqueeze(1) + (goal_xy - start_xy).unsqueeze(1) * t

        border = self.cfg.terrain.border_size
        scale = self.cfg.terrain.horizontal_scale
        pts = points + border
        idx = (pts / scale).long()
        px = torch.clamp(idx[..., 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clamp(idx[..., 1], 0, self.height_samples.shape[1] - 2)
        heights = self.height_samples[px, py] * self.cfg.terrain.vertical_scale

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

        pending = env_ids.clone()
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
            blocked = self._line_has_obstacle(pos, goal_world)
            ok = dist_ok & blocked
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
                if dist_ok.any():
                    self.goal_world[pending[dist_ok]] = goal_world[dist_ok]
                pending = pending[~dist_ok]

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
    args=get_args()
    cfg = HexGroundCfg()

    sim_params = {"sim":class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args,sim_params)
    env = HexGround(cfg,sim_params,args.physics_engine,args.sim_device,args.headless)
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step(torch.zeros(env.num_envs,env.num_actions,dtype=torch.float,device=env.device))
        
