
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.hex_v4.hex_terrain_config import HexTerrainCfg, HexTerrainCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.actuator import Actuator
from legged_gym.envs.hex_v4.expert import ExpertGround
import torch
import numpy as np

from isaacgym import gymtorch,gymapi,gymutil
from legged_gym.utils import get_args,class_to_dict
from legged_gym.utils.helpers import parse_sim_params
from legged_gym.utils.my_math import quat_apply_yaw
from isaacgym.torch_utils import torch_rand_float,quat_rotate_inverse

import math
import time
from legged_gym.envs.hex_v4.navigation_env import NavigationRewardFunction, NavigationRewardConfig, NavigationTaskManager


# ==================== LocomotionAdapter ====================
# 解耦高层导航策略（subgoal, intensity）与底层运动策略（velocity commands）
# 在 Phase 2/3 中使用，将 Teacher/Student 输出转换为底层可执行的速度指令
class LocomotionAdapter:
    """
    将高层导航指令 (subgoal_local, intensity) 转换为底层速度指令 (vx, vy, omega)
    
    设计理念：
    - subgoal_local: [dx, dy] 机器人局部坐标系下的子目标位置
    - intensity: [0, 1] 期望的运动强度（影响速度大小）
    - 输出: [vx, vy, omega] 前向速度、侧向速度、角速度
    
    转换逻辑：
    1. 计算朝向误差 theta = atan2(dy, dx)
    2. 计算目标距离 dist = sqrt(dx^2 + dy^2)
    3. 根据距离和强度计算速度：
       - vx = intensity * max_speed * cos(theta) * distance_factor
       - vy = intensity * max_speed * sin(theta) * distance_factor
       - omega = heading_gain * theta (朝向修正)
    """
    
    @staticmethod
    def convert(subgoal_local: torch.Tensor, 
                intensity: torch.Tensor, 
                max_lin_vel: float = 0.7,
                max_ang_vel: float = 1.0,
                distance_scale: float = 2.0,
                heading_gain: float = 2.0,
                device='cuda:0') -> torch.Tensor:
        """
        将 (subgoal, intensity) 转换为 (vx, vy, omega)
        
        Args:
            subgoal_local: [num_envs, 2] 局部坐标系下的子目标 (dx, dy)
            intensity: [num_envs] 运动强度 [0, 1]
            max_lin_vel: 最大线速度 (m/s)
            max_ang_vel: 最大角速度 (rad/s)
            distance_scale: 距离缩放因子（调节速度与距离的关系）
            heading_gain: 朝向修正增益
            device: 计算设备
            
        Returns:
            commands: [num_envs, 3] 速度指令 (vx, vy, omega)
        """
        dx = subgoal_local[:, 0]  # x方向距离
        dy = subgoal_local[:, 1]  # y方向距离
        
        # 计算极坐标
        dist = torch.sqrt(dx**2 + dy**2 + 1e-6)  # 避免除零
        theta = torch.atan2(dy, dx)  # 朝向角
        
        # 距离因子：远距离时加速，近距离时减速
        # 使用 tanh 实现平滑过渡
        distance_factor = torch.tanh(dist / distance_scale)
        
        # 速度大小 = intensity * max_speed * distance_factor
        speed = intensity * max_lin_vel * distance_factor
        
        # 分解为前向和侧向速度
        vx = speed * torch.cos(theta)
        vy = speed * torch.sin(theta)
        
        # 角速度：朝向修正（限幅到 max_ang_vel）
        omega = torch.clamp(heading_gain * theta, -max_ang_vel, max_ang_vel)
        
        # 组装输出
        commands = torch.stack([vx, vy, omega], dim=1)
        return commands

class HexTerrain(LeggedRobot):
    def __init__(self,cfg:HexTerrainCfg,sim_params,physics_engine,sim_device,headless):
        # 相机配置（在调用父类初始化前设置）
        self.camera_cfg = cfg.sensor.depth_camera
        self.nav_cfg = cfg.navigation
        
        # 如果在headless模式下启用相机，需要保持graphics_device_id为GPU
        # 而不是-1（CPU）
        self._use_camera_in_headless = headless and self.camera_cfg.enable
        
        self.enable_camera = self.camera_cfg.enable
        camera_width = self.camera_cfg.width
        camera_height = self.camera_cfg.height
        self.camera_handles = []
        
        # 调用父类初始化
        super().__init__(cfg,sim_params,physics_engine,sim_device,headless)
        
        # 在headless模式下启用相机：重新设置graphics_device_id
        if self._use_camera_in_headless:
            print("[Camera] Enabling cameras in headless mode with GPU graphics")
            # 重新设置graphics_device_id为GPU而不是-1
            self.graphics_device_id = self.sim_device_id
            # 重新创建sim以使用正确的graphics_device_id
            # 注意：这需要在prepare_sim之前完成
        
        self.cfg:HexTerrainCfg = cfg
        self.debug_viz = False
        self.foot_traj_viz=False
        #额外初始化电机类，可以计理想力矩或模拟的仿真力矩
        self.actuator=Actuator(self.cfg,self.device)
        #额外初始化专家类，可以在step时，提供专家动作参考
        # if self.cfg.env.gen_expert_actions:
        self.expert=ExpertGround(self.cfg,self.device,self.cfg.env.num_envs)
        self._init_contact_debug_indices()

        #额外初始化相机类
        cam_prop=gymapi.CameraProperties()
        # print("sim_params.use_gpu_pipline=",sim_params.use_gpu_pipline)

        # 创建相机后处理buffer
        self._init_camera_buffers()
        self._init_navigation_buffers()#新增初始化额外的observation buffer
        
        # Navigation reward modules
        self.nav_reward_fn = NavigationRewardFunction(NavigationRewardConfig())
        self.nav_task = NavigationTaskManager(
            num_envs=self.num_envs,
            device=self.device,
            map_size=(self.cfg.terrain.terrain_length, self.cfg.terrain.terrain_width),
            goal_reach_threshold=self.nav_cfg.goal_reached_threshold if hasattr(self.nav_cfg, "goal_reached_threshold") else 0.1,
            max_episode_length=int(self.max_episode_length),
            curriculum_enabled=True,
        )

        # buffers for nav reward
        self.prev_robot_pos_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_intensity_buf = torch.zeros(self.num_envs, device=self.device)
        self.intensity_buf = torch.ones(self.num_envs, device=self.device)  # 先占位：后面接高层λ输出

        # 确保 root_states 有有效数据（防御性编程）
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # init nav goals & prev buffers (local frame)
        env_ids_all = torch.arange(self.num_envs, device=self.device)
        robot_pos_local = self.root_states[:, :3] - self.env_origins

        self.nav_task.reset_goals(env_ids_all, robot_pos_local, min_distance=2.0, max_distance=8.0)
        self.prev_robot_pos_buf[:] = robot_pos_local
        self.prev_intensity_buf[:] = self.intensity_buf

        # init goal_buf for observations
        headings = self._yaw_from_quat(self.root_states[:, 3:7])
        self.goal_buf = self.nav_task.get_relative_goal(robot_pos_local, headings)

        self._train_iter = 0
        self._expert_interface_iter = None



    def _compute_collision_mask(self, threshold: float = None) -> torch.Tensor:
        """统一的collision事件判定
        
        【P0.4封装】:
        - 使用统一阈值 cfg.terrain.collision_force_threshold
        - 只检测非足端碰撞 (penalised_contact_indices)
        - 足端大力不触发collision事件（这是正常运动）
        
        Returns:
            collision_mask: (N,) bool - 发生碰撞为True
        """
        if threshold is None:
            threshold = getattr(self.cfg.terrain, 'collision_penalty_threshold', None)
            if threshold is None:
                threshold = getattr(self.cfg.terrain, 'collision_force_threshold', 1.0)
        # 只检测非足端刚体的碰撞力
        rb_force = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], 
            dim=-1
        )
        return (rb_force > threshold).any(dim=1)
    
    def _yaw_from_quat(self, quat: torch.Tensor) -> torch.Tensor:
        """从四元数提取yaw角度
        
        【格式约定 - CRITICAL】:
        - 输入: quat 格式 [x, y, z, w] (Isaac Gym 标准)
        - 输出: yaw 弧度，绕 z 轴旋转角
        - 公式: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        
        【全链路一致性】:
        - self.root_states[:, 3:7] → [x,y,z,w]
        - self.base_quat → [x,y,z,w]
        - quat_rotate_inverse(self.base_quat, ...) → 接受 [x,y,z,w]
        - 所有 euler 转换必须使用相同顺序
        
        Args:
            quat: (N,4) [x,y,z,w] - Isaac Gym标准格式
        
        Returns:
            yaw: (N,) yaw角度（弧度）
        """
        # 格式断言建议: assert quat.shape[-1] == 4, "quat must be [x,y,z,w]"
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

    def reset_separate(self):
        """重置环境并返回观测字典"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs_dict, _, _, _ = self.step_separate(torch.zeros_like(self.actions))
        return obs_dict
    
    def create_sim(self):
        """重写create_sim以在headless模式下支持相机"""
        
        # 在headless模式下启用相机：设置graphics_device_id为GPU
        if self._use_camera_in_headless:
            print("[Camera] Configuring GPU graphics for headless mode with cameras")
            self.graphics_device_id = self.sim_device_id
        
        # 调用父类创建环境和actors
        super().create_sim()
        
        # 延迟相机创建，避免与GPU初始化冲突
        # 相机将在第一次调用_get_depth_images时创建
        self.cameras_created = False
    
    def _create_envs(self):
        super()._create_envs()
        # 相机创建移到create_sim中

    def _init_contact_debug_indices(self):
        """Cache rigid-body indices for debug statistics (knee/thigh contacts)."""
        try:
            rb_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        except Exception:
            rb_names = []
        self._rb_names = rb_names

        def _indices_for(substr: str):
            return [i for i, n in enumerate(rb_names) if substr in n]

        knee = _indices_for("knee")
        thigh = _indices_for("thigh")
        self._knee_contact_indices = (
            torch.tensor(knee, dtype=torch.long, device=self.device) if len(knee) else None
        )
        self._thigh_contact_indices = (
            torch.tensor(thigh, dtype=torch.long, device=self.device) if len(thigh) else None
        )

        collision_debug_names = []
        for name in getattr(self.cfg.asset, "penalize_contacts_on", []):
            collision_debug_names.extend([rb_name for rb_name in rb_names if name in rb_name])
        for name in getattr(self.cfg.asset, "terminate_after_contacts_on", []):
            collision_debug_names.extend([rb_name for rb_name in rb_names if name in rb_name])
        collision_debug_names = list(dict.fromkeys(collision_debug_names))
        collision_debug_tensor_indices = []
        for rb_name in collision_debug_names:
            rb_index = None
            if hasattr(self.gym, "find_actor_rigid_body_index"):
                try:
                    rb_index = int(
                        self.gym.find_actor_rigid_body_index(
                            self.envs[0], self.actor_handles[0], rb_name, gymapi.DOMAIN_ENV
                        )
                    )
                except Exception:
                    rb_index = None
            if rb_index is None or rb_index < 0:
                try:
                    rb_index = int(rb_names.index(rb_name))
                except ValueError:
                    rb_index = -1
            if rb_index >= 0:
                collision_debug_tensor_indices.append(rb_index)
        self._collision_debug_rb_tensor_indices = (
            torch.tensor(collision_debug_tensor_indices, dtype=torch.long, device=self.device)
            if len(collision_debug_tensor_indices)
            else None
        )

    def _init_navigation_buffers(self):
        """初始化导航和EGPO观测buffers
        
        ============================================================
        EGPO Encoder 观测架构 (完整说明)
        ============================================================
        
        观测空间组成:
        1. obs_buf (67维) - 本体观测，可部署
           [quat(4), ang_vel(3), lin_acc(3), dof_pos(18), 
            dof_vel(18), torque(18), command(3)]
        
        2. obs_vgf_buf (30维) - 特权物理观测
           [base_lin_vel(3), projected_gravity(3), rb_contact_force(24)]
           训练: 真值，部署: Estimator估计 (MLP: 67→30)
        
        3. obs_terrain_buf (143维) - 特权地形观测
           [Raycast高度图: 11×13=143点]
           训练: Raycast采样，部署: LSTM估计 (历史obs→32)
        
        数据流向:
        ┌─────────────────────────────────────────────────────────┐
        │ 训练阶段 (Training)                                      │
        ├─────────────────────────────────────────────────────────┤
        │ obs(67) + obs_vgf(30) → [97] ──→ Storage → Actor/Critic│
        │ obs_terrain(143) → CNN Encoder → [32 latent] ──→ Concat │
        │ 最终输入: [67+30+32] = 129维                             │
        └─────────────────────────────────────────────────────────┘
        
        ┌─────────────────────────────────────────────────────────┐
        │ 部署阶段 (Deployment)                                    │
        ├─────────────────────────────────────────────────────────┤
        │ obs(67) → Estimator → obs_vgf_est(30)                   │
        │ obs_history(67+30)×20 → LSTM → terrain_latent_est(32)  │
        │ 最终输入: [67+30+32] = 129维 (无需Raycast)              │
        └─────────────────────────────────────────────────────────┘
        
        Runner硬编码 (expert_guided_encoder_runner.py:51):
          actor_obs_shape = [97]   # obs + vgf 拼接层
          critic_obs_shape = [143] # terrain for CNN encoder
        """
        print("[EGPO] Initializing observation buffers...")
        
        # ============================================================
        # 特权观测 Buffer 1: VGF (Velocity-Gravity-Force)
        # ============================================================
        self.obs_vgf_buf = torch.zeros(
            self.num_envs,
            30,  # [lin_vel(3) + gravity(3) + contact_force(24)]
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # ============================================================
        # 特权观测 Buffer 2: Terrain (Raycast Height Map)
        # ============================================================
        self.obs_terrain_buf = torch.zeros(
            self.num_envs,
            143,  # [11×13 height map]
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # 机器人状态buffer（高层用的精简版）
        self.robot_state_buf = torch.zeros(
            self.num_envs,
            9,  # [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # 目标位置buffer（相对机器人的目标位置）
        self.goal_buf = torch.zeros(
            self.num_envs,
            2,  # [goal_x, goal_y] 相对位置
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # === P0.3: Raw统计buffer (不受scale/dt影响的原始质量指标) ===
        # [camera_stability_sum, base_height_sum, collision_count, distance_traveled,
        #  camera_jitter_sum, camera_wobble_sum, camera_bobbing_sum]
        self.episode_raw_stats = torch.zeros(
            self.num_envs,
            7,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # === Command/Expert诊断统计 ===
        # [cmd_norm_sum, cmd_nonzero_count, base_speed_sum, expert_action_norm_sum, expert_action_update_count,
        #  cmd_abs_x_sum, cmd_abs_y_sum, base_abs_x_sum, base_abs_y_sum, upright_count, swing_frac_sum]
        self.episode_cmd_stats = torch.zeros(
            self.num_envs,
            11,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        self._expert_action_updated = False
        self._expert_action_norm_buf = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # === Debug stats for failure modes (episode accumulation) ===
        # [knee_contact_steps, thigh_contact_steps, dof_pos_limit_violation_sum, nonfoot_contact_steps]
        self.episode_debug_stats = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # 深度图buffer（无论是否启用相机，都初始化为零张量，避免None导致训练崩溃）
        self.depth_images = torch.zeros(
            self.num_envs,
            1,
            self.camera_cfg.height,
            self.camera_cfg.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        print(f"[Nav] Buffers initialized:")
        print(f"  - robot_state: {self.robot_state_buf.shape}")
        print(f"  - goal: {self.goal_buf.shape}")


    def _create_depth_cameras(self):
        """为所有环境创建深度相机"""
        
        # 定义相机属性 
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.camera_cfg.width              # 分辨率宽度
        camera_props.height = self.camera_cfg.height            # 分辨率高度
        camera_props.enable_tensors = True                      # 启用tensor输出
        
        # FOV设置
        camera_props.horizontal_fov = self.camera_cfg.horizontal_fov  # 水平视场角（度）
        
        # 深度范围
        camera_props.near_plane = self.camera_cfg.near_clip    # 近裁剪面
        camera_props.far_plane = self.camera_cfg.far_clip      # 远裁剪面
        
        print(f"[Camera] Creating {self.num_envs} depth cameras...")
        print(f"  - Resolution: {camera_props.width}×{camera_props.height}")
        print(f"  - FOV: {camera_props.horizontal_fov}°")
        print(f"  - Depth range: [{camera_props.near_plane}, {camera_props.far_plane}]m")
        
        # 为每个环境创建相机 
        self.camera_handles = []
        
        for env_idx in range(self.num_envs):
            env_handle = self.envs[env_idx]
            
            # 创建相机传感器
            camera_handle = self.gym.create_camera_sensor(
                env_handle, 
                camera_props
            )
            
            if camera_handle == -1:
                print(f"[Camera Error] Failed to create camera for env {env_idx}")
                continue
            
            # 获取机器人actor handle
            robot_handle = self.actor_handles[env_idx]
            
            # 设置相机位置（相对机器人base）
            
            local_transform = gymapi.Transform()
            
            # 位置：机器人前方0.25m，高度0.10m
            local_transform.p = gymapi.Vec3(
                self.camera_cfg.position[0],  # x: 左右
                self.camera_cfg.position[1],  # y: 前后
                self.camera_cfg.position[2]   # z: 高度
            )

            # 方向：根据pitch, yaw, roll角度设置四元数
            pitch_rad = np.deg2rad(self.camera_cfg.pitch_deg)  
            yaw_rad   = np.deg2rad(self.camera_cfg.yaw_deg)
            roll_rad  = np.deg2rad(self.camera_cfg.roll_deg)

            # 
            pitch_q = gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), pitch_rad)
            # 
            yaw_q   = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), yaw_rad)
            #
            roll_q  = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), roll_rad)

            # 组合（注意乘法次序，右乘后者）
            local_transform.r = pitch_q * yaw_q * roll_q
            
            # 附加相机到机器人base link
            body_handle = self.gym.find_actor_rigid_body_handle(
                env_handle,
                robot_handle,
                "body"  # hex_v4 的 base link 名称是 "body"
            )
            
            if body_handle == -1:
                print(f"[Camera Error] Failed to find body handle for env {env_idx}")
                continue
            
            # 打印调试信息（仅第一个环境）
            if env_idx == 0:
                print(f"[Camera Debug] Attaching camera at position: {local_transform.p}")
                print(f"[Camera Debug] Pitch angle: {self.camera_cfg.pitch_deg}°")
            
            self.gym.attach_camera_to_body(
                camera_handle,
                env_handle,
                body_handle,
                local_transform,
                gymapi.FOLLOW_TRANSFORM  # 相机跟随刚体运动
            )
            
            self.camera_handles.append(camera_handle)
        
        print(f"[Camera] Created {len(self.camera_handles)} cameras successfully!")
        
        # 初始化调试计数器
        self.depth_debug_count = 0

    def _init_camera_buffers(self):
        """初始化相机图像接收buffer
        
        【P1-Depth管线语义稳定】:
        - depth_raw: 永远 (N, H, W) - 传感器原始深度
        - depth_images: 永远 (N, 1, H, W) - 网络输入深度
        - 两者shape固定，禁止None或shape漂移
        """
        # 传感器原始深度 (N, H, W)
        self.depth_raw = torch.zeros(
            self.num_envs,
            self.camera_cfg.height,
            self.camera_cfg.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # 网络输入深度 (N, 1, H, W)
        self.depth_images = torch.zeros(
            self.num_envs,
            1,
            self.camera_cfg.height,
            self.camera_cfg.width,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )
        
        # 归一化的深度图（用于网络输入，暂未使用）
        # Shape: (num_envs, 1, height, width)
        self.depth_normalized = None
        # RGB 图像缓存 (num_envs, 3, H, W) 或 (num_envs, H, W, 3)
        self.rgb_images = None

    def _get_depth_images(self):
        """
        获取所有环境的深度图像
        
        返回:
            depth_images: torch.Tensor, shape (num_envs, height, width)
                         单位：米，包含距离信息
        """
        # 延迟创建相机（第一次调用时）
        if self.enable_camera and not self.cameras_created:
            print("[Camera] Creating cameras on first depth request...")
            self._create_depth_cameras()
            self.cameras_created = True
            if len(self.camera_handles) > 0:
                print("[Camera] Initializing camera rendering...")
                try:
                    self.gym.fetch_results(self.sim, True)
                    self.gym.step_graphics(self.sim)
                    self.gym.render_all_camera_sensors(self.sim)
                    if isinstance(self.device, str) and ('cuda' in self.device or 'gpu' in self.device):
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    print("[Camera] Camera rendering initialized")
                except Exception as e:
                    print(f"[Camera Warning] Initial render failed: {e}")
                    self.enable_camera = False
                    # P1-Depth: 填充depth_raw并返回，保持语义一致
                    self.depth_raw.fill_(self.camera_cfg.far_clip)
                    return self.depth_raw

        if not self.enable_camera or len(self.camera_handles) == 0:
            # P1-Depth: 填充depth_raw并返回，保持语义一致
            self.depth_raw.fill_(self.camera_cfg.far_clip)
            return self.depth_raw
        try:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
        except Exception as e:
            print(f"[Camera Error] Rendering failed: {e}")
            # P1-Depth: 填充depth_raw并返回，保持语义一致
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
                gymapi.IMAGE_DEPTH
            )
            depth_image = gymtorch.wrap_tensor(depth_tensor)
            # Isaac Gym 官方说明: IMAGE_DEPTH 为“negative distance from camera to pixel in view direction” 
            # 因此这里将其取反转换为常规正向距离表示，若已有正值则不动；只对出现负值的图像执行一次。
            if (depth_image < 0).any():
                depth_image = -depth_image
                if env_idx == 0 and hasattr(self, 'depth_debug_count') and self.depth_debug_count < 3:
                    print("[Depth Debug] Inverted negative depth to positive metric distances")

            if env_idx == 0 and hasattr(self, 'depth_debug_count') and self.depth_debug_count < 3:
                invalid = ~torch.isfinite(depth_image)
                valid = torch.isfinite(depth_image)
                total = depth_image.numel()
                valid_count = valid.sum().item()
                print(f"[Depth Debug] Raw depth min={depth_image[valid].min().item() if valid_count>0 else float('nan'):.3f}, "
                      f"max={depth_image[valid].max().item() if valid_count>0 else float('nan'):.3f}, "
                      f"mean={depth_image[valid].mean().item() if valid_count>0 else float('nan'):.3f}")
                print(f"[Depth Debug] Valid pixels: {valid_count}/{total} ({100*valid_count/total:.1f}%), invalid={invalid.sum().item()}")
                self.depth_debug_count += 1

            invalid_mask = ~torch.isfinite(depth_image)
            if invalid_mask.any():
                depth_image = depth_image.clone()
                depth_image[invalid_mask] = far

            depth_image = depth_image.clamp(near, far)
            depth_images_list.append(depth_image)

        self.gym.end_access_image_tensors(self.sim)

        # 写入原始深度buffer (N, H, W)
        self.depth_raw[:] = torch.stack(depth_images_list, dim=0)
        return self.depth_raw

    def _get_rgb_images(self, normalize: bool = True, channels_last: bool = False):
        """获取所有环境的 RGB 图像。

        参数:
            normalize: 是否将图像归一化到 [0,1] 浮点。
            channels_last: 若为 True, 返回形状 (num_envs, H, W, 3)，否则 (num_envs, 3, H, W)。

        返回:
            rgb_images: torch.Tensor
                形状 (num_envs, 3, H, W) 或 (num_envs, H, W, 3)
                dtype float32 (normalize=True) 或 uint8 (normalize=False)
        """
        # 延迟创建相机（与深度逻辑一致）
        if self.enable_camera and not self.cameras_created:
            print("[Camera] Creating cameras on first RGB request...")
            self._create_depth_cameras()
            self.cameras_created = True
            try:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
            except Exception as e:
                print(f"[Camera Warning] Initial RGB render failed: {e}")
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
        except Exception as e:
            print(f"[Camera Error] RGB rendering failed: {e}")
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
                # 填充空白图 (黑色)
                blank = torch.zeros(H, W, 3, dtype=torch.float32 if normalize else torch.uint8, device=self.device)
                rgb_list.append(blank)
                continue
            color_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                self.envs[env_idx],
                self.camera_handles[env_idx],
                gymapi.IMAGE_COLOR,
            )
            color_image = gymtorch.wrap_tensor(color_tensor)  # (H, W, 4) RGBA uint8
            # 去除 alpha 通道
            rgb = color_image[..., :3]
            if normalize:
                # 转为 float32 并归一化
                rgb = rgb.to(torch.float32) / 255.0
            if not channels_last:
                # 转换为 (3, H, W)
                rgb = rgb.permute(2, 0, 1).contiguous()
            rgb_list.append(rgb)
        self.gym.end_access_image_tensors(self.sim)

        if channels_last:
            self.rgb_images = torch.stack(rgb_list, dim=0)  # (N, H, W, 3)
        else:
            self.rgb_images = torch.stack(rgb_list, dim=0)  # (N, 3, H, W)
        return self.rgb_images
    
    def _process_depth_for_network(self, depth_images):
        """
        预处理深度图，用于神经网络输入
        
        参数:
            depth_images: (num_envs, height, width)
            
        返回:
            depth_normalized: (num_envs, 1, height, width), 范围[0, 1]
        """
        # 归一化到[0, 1]
        depth_normalized = (depth_images - self.camera_cfg.near_clip) / \
                          (self.camera_cfg.far_clip - self.camera_cfg.near_clip)
        
        # 添加channel维度
        depth_normalized = depth_normalized.unsqueeze(1)
        
        # 可选：调整分辨率（如果网络需要不同大小）
        if hasattr(self.camera_cfg, 'output_size'):
            depth_normalized = torch.nn.functional.interpolate(
                depth_normalized,
                size=(self.camera_cfg.output_size, self.camera_cfg.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        return depth_normalized



    def step_separate(self,actions):
        #因为返回的观测改变了，因此需要重新定义step函数
        #新增获取深度图像的部分
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        返回:
            obs_dict: 观测字典
            rewards: 奖励
            dones: 结束标志
            infos: 额外信息
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
        # EGPO: 分离观测结构，clip各自的buffer而不是privileged_obs_buf
        self.obs_vgf_buf = torch.clip(self.obs_vgf_buf, -clip_obs, clip_obs)
        self.obs_terrain_buf = torch.clip(self.obs_terrain_buf, -clip_obs, clip_obs)

        #  更新深度图 (P1-Depth: 拆分管线，固定shape)
        if self.enable_camera and (self.common_step_counter % self.camera_cfg.capture_interval == 0):
            # 获取原始深度 -> depth_raw (N, H, W)
            depth_raw = self._get_depth_images()
            # 预处理为网络输入 -> depth_images (N, 1, H, W)
            processed = self._process_depth_for_network(depth_raw)
            # 就地写入，保持buffer身份稳定
            self.depth_images[:] = processed
        
        # 构造观测字典 (原来是tuple，现在是字典)
        obs_dict = {
            'proprioception': self.obs_buf,           # (num_envs, 67)
            'privileged': self.obs_vgf_buf,          # (num_envs, 30)
            'terrain': self.obs_terrain_buf,          # (num_envs, 187)
            'depth': self.depth_images,               # (num_envs, 1, H, W)
            'robot_state': self.robot_state_buf,      # (num_envs, 9)
            'goal': self.goal_buf                     # (num_envs, 2)
        }
        
        return obs_dict, self.rew_buf, self.reset_buf, self.extras

    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        #因为要对command ranges 进行修改，所以重新定义这个函数
        if self.cfg.commands.curriculum:
            self.command_ranges["lin_vel_x"]=[-0.4,0.4]
            self.command_ranges["lin_vel_y"]=[-0.6,0.6]
            self.command_ranges["ang_vel_yaw"]=[-0.4,0.4]
        

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

        # 添加的部分，模拟加速度计的输出，所以要减去重力加速度，这里加速度的单位是g
        # 注意：last_root_vel 在每个 control step 更新一次，因此差分时间应使用 self.dt (= sim_dt * decimation)
        root_acc = ((self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt) / 9.81 - self.gravity_vec
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, root_acc)

        root_ang_acc = (self.root_states[:, 10:13] - self.last_root_vel[:, 3:]) / self.dt
        self.base_ang_acc[:] = quat_rotate_inverse(self.base_quat, root_ang_acc)
        #根据IMU安装的位置，根据基座质心计算IMU质心处加速度的大小
        self.IMU_lin_acc = self.base_lin_acc + (self.base_ang_acc.cross(self.IMU_pos,dim=1) + self.base_ang_vel.cross(self.base_ang_vel.cross(self.IMU_pos,dim=1),dim=1))/9.81

        self._post_physics_step_callback()

        # compute observations first (we need robot_state/goal for nav)
        self.compute_observations_separated()

        # base termination (fall/illegal contacts etc.)
        self.check_termination()

        # === Command/Expert诊断统计（每步累加） ===
        with torch.no_grad():
            # === A: 非足端接触滞后触发（供 collision reward 使用） ===
            collision_penalty_threshold = getattr(self.cfg.terrain, "collision_penalty_threshold", None)
            nonfoot_now = self._compute_collision_mask(threshold=collision_penalty_threshold)
            self.nonfoot_contact_streak[nonfoot_now] += 1
            self.nonfoot_contact_streak[~nonfoot_now] = 0
            h_steps = int(getattr(self.cfg.rewards, "nonfoot_contact_hysteresis_steps", 1))
            h_steps = max(1, h_steps)
            self.nonfoot_contact_trigger = self.nonfoot_contact_streak >= h_steps
            self.episode_nonfoot_trigger_steps += self.nonfoot_contact_trigger.float()

            # === B1: jitter 饱和比例 + 足端冲击（episode累加）===
            raw_ang_jitter = torch.sum(torch.square(self.base_ang_acc[:, :2]), dim=1)  # (rad/s^2)^2
            jitter_cap = float(getattr(self.cfg.rewards, "camera_jitter_cap", 50.0))
            self.episode_jitter_sat_steps += (raw_ang_jitter >= jitter_cap).float()
            self.episode_raw_ang_jitter_sum += raw_ang_jitter

            feet_force_norm = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
            dF = torch.clamp(feet_force_norm - self._prev_feet_force_norm, min=0.0)
            self.episode_foot_dF_sum += dF.mean(dim=1)
            self._prev_feet_force_norm[:] = feet_force_norm

            cmd = self._get_effective_commands()
            cmd_norm = torch.norm(cmd, dim=1)
            cmd_nonzero_th = getattr(self.cfg.commands, 'nonzero_threshold', 0.05)
            cmd_nonzero = cmd_norm > cmd_nonzero_th
            base_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
            self.episode_cmd_stats[:, 0] += cmd_norm
            self.episode_cmd_stats[:, 1] += cmd_nonzero.float()
            self.episode_cmd_stats[:, 2] += base_speed

            # x/y command distribution and realized motion (user mapping: y=前后, x=横移)
            self.episode_cmd_stats[:, 5] += torch.abs(cmd[:, 0])
            self.episode_cmd_stats[:, 6] += torch.abs(cmd[:, 1])
            self.episode_cmd_stats[:, 7] += torch.abs(self.base_lin_vel[:, 0])
            self.episode_cmd_stats[:, 8] += torch.abs(self.base_lin_vel[:, 1])

            upright_cos_min = getattr(self.cfg.rewards, "upright_cos_min", 0.75)
            upright = self.projected_gravity[:, 2] < -upright_cos_min
            self.episode_cmd_stats[:, 9] += upright.float()

            foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
            foot_contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
            contact_filt = torch.logical_or(foot_contact, self.last_contacts)
            first_contact = (self.feet_air_time > 0.0) & contact_filt
            swing_frac = (~foot_contact).float().mean(dim=1)
            self.episode_cmd_stats[:, 10] += swing_frac
            # foot contact timing stats
            self.episode_foot_contact_events += first_contact.float().sum(dim=1)
            self.episode_foot_contact_time_sum += (self.feet_air_time * first_contact.float()).sum(dim=1)
            contact_fz = torch.abs(self.contact_forces[:, self.feet_indices, 2])
            self.episode_foot_contact_fz_sum += (contact_fz * contact_filt.float()).sum(dim=1)
            self.episode_foot_contact_fz_count += contact_filt.float().sum(dim=1)
            if self._expert_action_updated:
                self.episode_cmd_stats[:, 3] += self._expert_action_norm_buf
                self.episode_cmd_stats[:, 4] += 1.0
                self._expert_action_updated = False
                
            # === Failure mode debug stats ===
            contact_th = getattr(self.cfg.terrain, "collision_penalty_threshold", 0.5)
            # knee/thigh contacts (if indices are available)
            if self._knee_contact_indices is not None:
                knee_force = torch.norm(self.contact_forces[:, self._knee_contact_indices, :], dim=-1)
                self.episode_debug_stats[:, 0] += (knee_force > contact_th).any(dim=1).float()
            if self._thigh_contact_indices is not None:
                thigh_force = torch.norm(self.contact_forces[:, self._thigh_contact_indices, :], dim=-1)
                self.episode_debug_stats[:, 1] += (thigh_force > contact_th).any(dim=1).float()
            # dof pos limit violations (soft limits from URDF)
            out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clamp(max=0.0)
            out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clamp(min=0.0)
            self.episode_debug_stats[:, 2] += torch.sum(out_of_limits, dim=1)
            # any non-foot contact (penalized bodies)
            self.episode_debug_stats[:, 3] += self._compute_collision_mask(threshold=contact_th).float()

            # one-time debug: verify penalised_contact_indices actually cover knee/thigh bodies
            if self.common_step_counter == 1 and hasattr(self, "extras"):
                try:
                    penalised = set(self.penalised_contact_indices.detach().cpu().tolist())
                except Exception:
                    penalised = set()
                knee = (
                    set(self._knee_contact_indices.detach().cpu().tolist())
                    if self._knee_contact_indices is not None
                    else set()
                )
                thigh = (
                    set(self._thigh_contact_indices.detach().cpu().tolist())
                    if self._thigh_contact_indices is not None
                    else set()
                )
                self.extras["debug/contact/penalised_count"] = float(len(penalised))
                self.extras["debug/contact/knee_count"] = float(len(knee))
                self.extras["debug/contact/thigh_count"] = float(len(thigh))
                self.extras["debug/contact/knee_overlap_penalised"] = float(len(knee & penalised))
                self.extras["debug/contact/thigh_overlap_penalised"] = float(len(thigh & penalised))

        if getattr(self.nav_cfg, "enable_nav_reward", False):
            # === [P0.2 数据一致性] Phase 2: Navigation Reward 分支 ===
            # 【GPT澄清】P0.2的真正含义：
            # - 目标：确保 policy/expert 对齐路径使用同源指令
            # - 必须统一：obs command、expert command、BC对齐用的command
            # - 无需统一：Phase 2的reward已被override，不依赖commands
            # - 当前设计：self.commands仅用于日志诊断(raw vs effective对比)
            
            # [Fix] Real Intensity Calculation - Prevent Reward Hacking
            # Use actual physics speed (body frame), not terrain difficulty
            speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)  # (N,)
            max_speed = getattr(self.nav_cfg, 'max_speed_for_intensity', 1.0)
            self.intensity_buf = torch.clamp(speed_xy / max_speed, 0.0, 1.0)
            
            # === P0.4: Collision mask (使用封装函数) ===
            collision_penalty_threshold = getattr(self.cfg.terrain, 'collision_penalty_threshold', None)
            collision_mask = self._compute_collision_mask(threshold=collision_penalty_threshold)
            collision_term_threshold = getattr(self.cfg.terrain, 'collision_force_threshold', 1.0)
            collision_mask_term = self._compute_collision_mask(threshold=collision_term_threshold)

            # terrain difficulty: use terrain level normalized to [0,1]
            # terrain_levels 在 LeggedRobot 里一般存在（课程学习用）
            if hasattr(self, "terrain_levels"):
                denom = max(1, int(self.max_terrain_level - 1))
                terrain_difficulty = torch.clamp(self.terrain_levels.float() / denom, 0.0, 1.0)
            else:
                terrain_difficulty = torch.zeros(self.num_envs, device=self.device)

            # --- local robot pos for navigation ---
            robot_pos_local = self.root_states[:, :3] - self.env_origins  # (N,3)
            goal_local = self.nav_task.goal_positions                     # (N,2) 约定：存 local goal

            # compute nav reward (使用 local 坐标)
            rew_dict = self.nav_reward_fn.compute_reward(
                robot_pos=robot_pos_local,
                prev_robot_pos=self.prev_robot_pos_buf,   # 也存 local
                goal_pos=goal_local,
                robot_vel=self.root_states[:, 7:10],      # 速度不受平移影响
                robot_quat=self.base_quat,                # [x,y,z,w] 格式
                intensity=self.intensity_buf,
                prev_intensity=self.prev_intensity_buf,
                terrain_difficulty=terrain_difficulty,
                collision_mask=collision_mask,
            )

            # override reward
            self.rew_buf = rew_dict["total"]
            
            # === P0.3: 维护 Phase 2/3 raw统计（保证curriculum正常工作） ===
            # 【GPT审查修正】raw统计不乘scale/dt，直接累加质量指标
            with torch.no_grad():
                # 1. camera_stability: [0,1]质量指标，每步累加
                camera_quality, ang_jitter, ang_wobble, z_bobbing, penalty = self._reward_camera_stability(
                    return_terms=True
                )
                self.episode_raw_stats[:, 0] += camera_quality
                self.episode_raw_stats[:, 4] += ang_jitter
                self.episode_raw_stats[:, 5] += ang_wobble
                self.episode_raw_stats[:, 6] += z_bobbing
                
                # 2. base_height: [0,1]质量指标，每步累加
                height_quality, _ = self._base_height_quality()
                self.episode_raw_stats[:, 1] += height_quality
                
                # 3. collision_count: 事件计数，每次碰撞+1
                collision_events = collision_mask.float()
                self.episode_raw_stats[:, 2] += collision_events
                
                # 4. distance_traveled: 沿指令方向的位移投影累加(米)
                delta_xy = robot_pos_local[:, :2] - self.prev_robot_pos_buf[:, :2]
                cmd_xy = self._get_effective_commands()[:, :2]
                cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
                cmd_dir_body = torch.where(cmd_norm > 1e-6, cmd_xy / cmd_norm, torch.zeros_like(cmd_xy))
                cmd_dir_world = quat_apply_yaw(
                    self.base_quat, torch.cat([cmd_dir_body, torch.zeros_like(cmd_dir_body[:, :1])], dim=1)
                )[:, :2]
                step_progress = torch.sum(delta_xy * cmd_dir_world, dim=1)
                self.episode_raw_stats[:, 3] += torch.clamp(step_progress, min=0.0)
            
            # Phase 2/3: 可选的稳定性保持（避免导航时相机质量退化）
            # 如果配置中启用了 nav_stability_weight，额外添加 camera_stability 奖励
            if hasattr(self.cfg.rewards, 'nav_stability_weight'):
                nav_stability_weight = getattr(self.cfg.rewards, 'nav_stability_weight', 0.0)
                if nav_stability_weight and nav_stability_weight > 0:
                    # 复用 Phase 1 的 camera_stability 奖励函数 (这里是reward shaping，可以乘scale*dt)
                    camera_rew_shaped = self._reward_camera_stability() * self.cfg.rewards.scales.camera_stability * self.dt
                    self.rew_buf += nav_stability_weight * camera_rew_shaped
                    # 日志字段改名避免与raw混淆
                    self.extras["nav_rew"]["camera_stability_shaped"] = camera_rew_shaped.mean().item()

            # nav termination: reached goal / timeout / collision (使用 local 坐标)
            dones_nav, successes, info_nav = self.nav_task.check_termination(
                robot_positions=robot_pos_local,
                collision_mask=collision_mask_term,
            )
            self.reset_buf |= dones_nav

            # update curriculum for completed episodes
            env_ids_nav = dones_nav.nonzero(as_tuple=False).flatten()
            if env_ids_nav.numel() > 0:
                self.nav_task.update_curriculum(env_ids_nav, successes)

            # logging
            self.extras["nav_rew"] = {k: v.mean().item() for k, v in rew_dict.items() if k != "total"}
            self.extras["nav_success_rate"] = successes.float().mean().item()
            self.extras["nav_collision_soft_rate"] = collision_mask.float().mean().item()
            self.extras["nav_collision_hard_rate"] = collision_mask_term.float().mean().item()
            # Step A-P1: 添加intensity监控日志
            self.extras["intensity_mean"] = self.intensity_buf.mean().item()
            self.extras["speed_xy_mean"] = speed_xy.mean().item()
            self.extras["terrain_difficulty_mean"] = terrain_difficulty.mean().item()
            
            # 导航指标监控
            self.extras["goal_distance_mean"] = torch.norm(self.goal_buf, dim=1).mean().item()
            effective_cmd = self._get_effective_commands()
            self.extras["command_vx_mean"] = effective_cmd[:, 0].mean().item()
            self.extras["command_vy_mean"] = effective_cmd[:, 1].mean().item()
            self.extras["command_omega_mean"] = effective_cmd[:, 2].mean().item()
            
            # [P0.2 数据一致性] 诊断日志：记录raw commands vs effective commands
            # 【GPT澄清】这是合理的诊断功能，用于监控Phase 2中goal指引的效果
            # self.commands = 基础随机指令，effective_cmd = goal导出的导航指令
            # 对比两者差异有助于理解导航策略是否正常工作
            self.extras["raw_cmd_vx_mean"] = self.commands[:, 0].mean().item()
            self.extras["raw_cmd_vy_mean"] = self.commands[:, 1].mean().item()
            self.extras["raw_cmd_omega_mean"] = self.commands[:, 2].mean().item()

            # goal_buf 已经在 compute_observations_separated() -> _update_goal_buffer() 中更新
            # 无需重复更新

            # update prev buffers (local) - 在reward计算之后更新
            self.prev_robot_pos_buf[:] = robot_pos_local
            self.prev_intensity_buf[:] = self.intensity_buf
            
            # 【GPT建议】Phase 2数据链完整性断言（debug模式）
            if getattr(self.cfg.env, 'debug_mode', False):
                assert torch.isfinite(self.goal_buf).all(), \
                    f"[Phase2-Debug] goal_buf has NaN/Inf! mean={self.goal_buf.mean()}, max={self.goal_buf.max()}"
                assert torch.isfinite(effective_cmd).all(), \
                    f"[Phase2-Debug] effective_cmd has NaN/Inf! mean={effective_cmd.mean()}, max={effective_cmd.max()}"
                assert torch.isfinite(self.rew_buf).all(), \
                    f"[Phase2-Debug] rew_buf has NaN/Inf! mean={self.rew_buf.mean()}, min={self.rew_buf.min()}, max={self.rew_buf.max()}"

        else:
            # default locomotion reward
            self.compute_reward()
            
            # === Phase 1: 维护 raw 统计（保证curriculum正常工作）===
            # 注意：基类compute_reward()不维护episode_raw_stats，需要在这里补充
            with torch.no_grad():
                # 1. camera_stability: [0,1]质量指标
                camera_quality, ang_jitter, ang_wobble, z_bobbing, penalty = self._reward_camera_stability(
                    return_terms=True
                )
                self.episode_raw_stats[:, 0] += camera_quality
                self.episode_raw_stats[:, 4] += ang_jitter
                self.episode_raw_stats[:, 5] += ang_wobble
                self.episode_raw_stats[:, 6] += z_bobbing
                
                # 2. base_height: [0,1]质量指标
                height_quality, _ = self._base_height_quality()
                self.episode_raw_stats[:, 1] += height_quality
                
                # 3. collision_count: 使用封装函数
                collision_mask = self._compute_collision_mask()
                self.episode_raw_stats[:, 2] += collision_mask.float()
                
                # 4. distance_traveled: 沿指令方向的位移投影累加
                robot_pos_local = self.root_states[:, :3] - self.env_origins
                delta_xy = robot_pos_local[:, :2] - self.prev_robot_pos_buf[:, :2]
                cmd_xy = self._get_effective_commands()[:, :2]
                cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
                cmd_dir_body = torch.where(cmd_norm > 1e-6, cmd_xy / cmd_norm, torch.zeros_like(cmd_xy))
                cmd_dir_world = quat_apply_yaw(
                    self.base_quat, torch.cat([cmd_dir_body, torch.zeros_like(cmd_dir_body[:, :1])], dim=1)
                )[:, :2]
                step_progress = torch.sum(delta_xy * cmd_dir_world, dim=1)
                self.episode_raw_stats[:, 3] += torch.clamp(step_progress, min=0.0)
                
                # 更新prev_robot_pos_buf用于下一步距离计算
                self.prev_robot_pos_buf[:] = robot_pos_local
            
            # Phase 1数据链完整性断言（debug模式）
            if getattr(self.cfg.env, 'debug_mode', False):
                assert torch.isfinite(self.rew_buf).all(), \
                    f"[Phase1-Debug] rew_buf has NaN/Inf! mean={self.rew_buf.mean()}, min={self.rew_buf.min()}, max={self.rew_buf.max()}"
            
        
        # Curriculum统计有效性断言（每500步检查一次，避免性能影响）
        if getattr(self.cfg.env, 'debug_mode', False) and self.common_step_counter % 500 == 0:
            # 确保raw统计链没有断掉（episode至少有一个非零变化）
            # episode_raw_stats[:, 0-3]应该在递增（camera/height质量、collision次数、距离）
            assert (self.episode_raw_stats[:, 3] > 0).any() or self.common_step_counter < 100, \
                f"[Curriculum-Debug] distance_traveled统计全为0! 可能统计链断裂。raw_stats mean={self.episode_raw_stats.mean(0)}"
            assert (self.episode_raw_stats[:, 2] >= 0).all(), \
                f"[Curriculum-Debug] collision_count出现负值! 统计损坏。raw_stats min={self.episode_raw_stats.min(0)[0]}"
            # camera/height质量应在合理范围[0, max_episode_length]
            max_expected = self.max_episode_length_s / self.cfg.sim.dt
            assert (self.episode_raw_stats[:, 0] <= max_expected * 2).all(), \
                f"[Curriculum-Debug] camera_stability统计异常大! max={self.episode_raw_stats[:, 0].max()}, expected<{max_expected}"
            
            # Phase 1 相机稳定性监控
            self.extras["camera_pitch_std"] = self.base_ang_vel[:, 0].std().item()
            self.extras["camera_roll_std"] = self.base_ang_vel[:, 1].std().item()
            self.extras["camera_ang_acc_rms"] = torch.sqrt(
                torch.mean(self.base_ang_acc[:, :2]**2)
            ).item()
            self.extras["camera_z_acc_std"] = self.base_lin_acc[:, 2].std().item()

        # reset environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # P2-Guard: 防御性检查，虽然基类有guard但此处显式添加
        if env_ids.numel() > 0:
            self.reset_idx(env_ids)

        # Auto-reset后刷新观测，确保返回的obs与环境state一致
        if env_ids.numel() > 0:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.compute_observations_separated()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # Update contact history for reward contact filtering (PhysX meshes can be noisy)
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        self.last_contacts = (torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold)

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()        

        if self.viewer and self.foot_traj_viz:
            self._draw_foot_end_trajectory()
        



    def set_train_progress(self, train_iter, expert_interface_iter=None):
        self._train_iter = int(train_iter)
        if expert_interface_iter is not None:
            self._expert_interface_iter = int(expert_interface_iter)

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

    def _get_effective_commands(self):
        """
        Get effective commands (random in Phase 1, goal-directed in Phase 2/3)
        
        【EGPO Critical】:
        - Both policy and expert MUST see the same commands
        - Phase 1 (enable_nav_reward=False): Random locomotion commands
        - Phase 2/3 (enable_nav_reward=True): Goal-derived navigation commands
          - 如果 use_adapter=True，使用 LocomotionAdapter 转换高层指令
          - 如果 use_adapter=False，直接从 goal_buf 推导速度指令（原逻辑）
        
        Returns:
            commands: (N, 3) [vx, vy, omega_z]
        """
        if getattr(self.nav_cfg, "enable_nav_reward", False):
            # Phase 2/3: Derive commands from goal
            if getattr(self.nav_cfg, "use_adapter", False):
                # 使用 LocomotionAdapter：需要高层策略提供 (subgoal, intensity)
                # 这里假设 self.goal_buf 是 subgoal_local (局部坐标系)
                # self.intensity_buf 是高层策略输出的强度
                
                # 如果 intensity_buf 未初始化，使用距离推导的默认值
                if not hasattr(self, 'intensity_buf'):
                    goal_dist = torch.norm(self.goal_buf, dim=1)
                    self.intensity_buf = torch.clamp(goal_dist / 5.0, 0.0, 1.0)
                
                obs_commands = LocomotionAdapter.convert(
                    subgoal_local=self.goal_buf,
                    intensity=self.intensity_buf,
                    max_lin_vel=0.7,
                    max_ang_vel=getattr(self.nav_cfg, "adapter_max_ang_vel", 1.0),
                    distance_scale=getattr(self.nav_cfg, "adapter_distance_scale", 2.0),
                    heading_gain=getattr(self.nav_cfg, "adapter_heading_gain", 2.0),
                    device=self.device
                )
            else:
                # 原逻辑：直接从 goal_buf 推导速度指令
                goal_vec = self.goal_buf
                dist = torch.norm(goal_vec, dim=1, keepdim=True)
                
                # 安全处理：防止极小距离导致数值尖峰
                min_dist = getattr(self.nav_cfg, 'min_command_distance', 0.1)
                dist_safe = torch.clamp(dist, min=min_dist)
                
                # Distance-based speed scaling: prevent jitter at goal
                slowdown_dist = getattr(self.nav_cfg, 'goal_slowdown_distance', 1.0)
                min_speed = getattr(self.nav_cfg, 'goal_min_speed_ratio', 0.2)
                speed_scale = torch.clamp(dist / slowdown_dist, min_speed, 1.0)
                
                target_dir = goal_vec / dist_safe
                
                obs_commands = torch.zeros_like(self.commands)
                # Linear velocity towards goal with clamp
                max_lin_vel = getattr(self.nav_cfg, 'max_lin_vel_command', 0.8)
                obs_commands[:, 0] = torch.clamp(
                    target_dir[:, 0] * max_lin_vel * speed_scale.squeeze(-1),
                    -max_lin_vel, max_lin_vel
                )
                obs_commands[:, 1] = torch.clamp(
                    target_dir[:, 1] * max_lin_vel * speed_scale.squeeze(-1),
                    -max_lin_vel, max_lin_vel
                )
                # Angular velocity to correct heading
                heading_error = torch.atan2(target_dir[:, 1], target_dir[:, 0])
                max_ang_vel = getattr(self.nav_cfg, 'max_ang_vel_command', 1.5)
                obs_commands[:, 2] = torch.clamp(
                    heading_error * 1.5, -max_ang_vel, max_ang_vel
                )
            
            # Safety: replace NaN with zeros
            return torch.nan_to_num(obs_commands, nan=0.0)
        else:
            # Phase 1: Random commands from base class
            return self.commands
    
    def reset_idx(self, env_ids:torch.Tensor):
        """重置指定环境
        
        【重要 - Auto-reset 语义】:
        - 此函数在 post_physics_step_separate() 中被调用
        - 重置后必须刷新观测（由 post_physics_step_separate 负责）
        - 保证返回 s0 观测与重置后状态一致
        - 避免 obs/state 时序错位导致训练不稳定
        
        【四元数格式】:
        - self.root_states[:, 3:7] 为 [x,y,z,w]
        - 所有 euler 转换使用一致格式
        """
        if len(env_ids) != 0:
            ep_len = self.episode_length_buf[env_ids].float().clamp_min(1.0)
            ep_camera_quality = (self.episode_raw_stats[env_ids, 0] / ep_len).mean().item()
            ep_height_quality = (self.episode_raw_stats[env_ids, 1] / ep_len).mean().item()
            ep_collision_count = self.episode_raw_stats[env_ids, 2].mean().item()
            ep_distance_traveled = self.episode_raw_stats[env_ids, 3].mean().item()
            ep_camera_jitter = (self.episode_raw_stats[env_ids, 4] / ep_len).mean().item()
            ep_camera_wobble = (self.episode_raw_stats[env_ids, 5] / ep_len).mean().item()
            ep_camera_bobbing = (self.episode_raw_stats[env_ids, 6] / ep_len).mean().item()
            ep_cmd_norm_mean = (self.episode_cmd_stats[env_ids, 0] / ep_len).mean().item()
            ep_cmd_nonzero_frac = (self.episode_cmd_stats[env_ids, 1] / ep_len).mean().item()
            ep_base_speed_mean = (self.episode_cmd_stats[env_ids, 2] / ep_len).mean().item()
            expert_update_count = self.episode_cmd_stats[env_ids, 4]
            expert_denom = torch.clamp(expert_update_count, min=1.0)
            ep_expert_action_norm_mean = (self.episode_cmd_stats[env_ids, 3] / expert_denom).mean().item()
            ep_expert_update_frac = (expert_update_count / ep_len).mean().item()

            cmd_abs_x_mean = (self.episode_cmd_stats[env_ids, 5] / ep_len).mean().item()
            cmd_abs_y_mean = (self.episode_cmd_stats[env_ids, 6] / ep_len).mean().item()
            base_abs_x_mean = (self.episode_cmd_stats[env_ids, 7] / ep_len).mean().item()
            base_abs_y_mean = (self.episode_cmd_stats[env_ids, 8] / ep_len).mean().item()
            upright_frac = (self.episode_cmd_stats[env_ids, 9] / ep_len).mean().item()
            swing_frac = (self.episode_cmd_stats[env_ids, 10] / ep_len).mean().item()

            jitter_w = getattr(self.cfg.rewards, 'camera_jitter_weight', 0.05)
            wobble_w = getattr(self.cfg.rewards, 'camera_wobble_weight', 0.5)
            bobbing_w = getattr(self.cfg.rewards, 'camera_bobbing_weight', 0.1)
            ep_camera_penalty = ep_camera_jitter * jitter_w + ep_camera_wobble * wobble_w + ep_camera_bobbing * bobbing_w

            ep_camera_raw_ang_jitter_mean = (
                (self.episode_raw_ang_jitter_sum[env_ids] / ep_len).mean().item()
                if hasattr(self, "episode_raw_ang_jitter_sum")
                else 0.0
            )
            ep_camera_raw_ang_acc_xy_rms = float(math.sqrt(max(ep_camera_raw_ang_jitter_mean, 0.0)))
            ep_camera_jitter_sat_frac = (
                (self.episode_jitter_sat_steps[env_ids] / ep_len).mean().item()
                if hasattr(self, "episode_jitter_sat_steps")
                else 0.0
            )
            ep_foot_impact_df_mean = (
                (self.episode_foot_dF_sum[env_ids] / ep_len).mean().item()
                if hasattr(self, "episode_foot_dF_sum")
                else 0.0
            )
            ep_foot_contact_event_rate = (
                (self.episode_foot_contact_events[env_ids] / (ep_len * float(self.feet_indices.shape[0]))).mean().item()
                if hasattr(self, "episode_foot_contact_events")
                else 0.0
            )
            ep_foot_air_time_at_contact = (
                (self.episode_foot_contact_time_sum[env_ids] /
                 torch.clamp(self.episode_foot_contact_events[env_ids], min=1.0)).mean().item()
                if hasattr(self, "episode_foot_contact_time_sum")
                else 0.0
            )
            ep_foot_contact_fz_mean = (
                (self.episode_foot_contact_fz_sum[env_ids] /
                 torch.clamp(self.episode_foot_contact_fz_count[env_ids], min=1.0)).mean().item()
                if hasattr(self, "episode_foot_contact_fz_sum")
                else 0.0
            )
            ep_nonfoot_contact_trigger_frac = (
                (self.episode_nonfoot_trigger_steps[env_ids] / ep_len).mean().item()
                if hasattr(self, "episode_nonfoot_trigger_steps")
                else 0.0
            )

            base_ang_acc_xy_rms = torch.sqrt(torch.mean(self.base_ang_acc[env_ids, :2] ** 2)).item()
            base_ang_vel_xy_rms = torch.sqrt(torch.mean(self.base_ang_vel[env_ids, :2] ** 2)).item()
            base_lin_acc_z_rms = torch.sqrt(torch.mean(self.base_lin_acc[env_ids, 2] ** 2)).item()

            self.extras["ep_camera_quality"] = ep_camera_quality
            self.extras["ep_height_quality"] = ep_height_quality
            self.extras["ep_collision_count"] = ep_collision_count
            self.extras["ep_distance_traveled"] = ep_distance_traveled
            self.extras["ep_camera_jitter"] = ep_camera_jitter
            self.extras["ep_camera_wobble"] = ep_camera_wobble
            self.extras["ep_camera_bobbing"] = ep_camera_bobbing
            self.extras["ep_camera_penalty"] = ep_camera_penalty
            self.extras["ep_camera_raw_ang_acc_xy_rms"] = ep_camera_raw_ang_acc_xy_rms
            self.extras["ep_camera_jitter_sat_frac"] = ep_camera_jitter_sat_frac
            self.extras["ep_foot_impact_df_mean"] = ep_foot_impact_df_mean
            self.extras["ep_foot_contact_event_rate"] = ep_foot_contact_event_rate
            self.extras["ep_foot_air_time_at_contact"] = ep_foot_air_time_at_contact
            self.extras["ep_foot_contact_fz_mean"] = ep_foot_contact_fz_mean
            self.extras["ep_nonfoot_contact_trigger_frac"] = ep_nonfoot_contact_trigger_frac
            self.extras["ep_base_ang_acc_xy_rms"] = base_ang_acc_xy_rms
            self.extras["ep_base_ang_vel_xy_rms"] = base_ang_vel_xy_rms
            self.extras["ep_base_lin_acc_z_rms"] = base_lin_acc_z_rms
            self.extras["ep_cmd_norm_mean"] = ep_cmd_norm_mean
            self.extras["ep_cmd_nonzero_frac"] = ep_cmd_nonzero_frac
            self.extras["ep_base_speed_mean"] = ep_base_speed_mean
            self.extras["ep_expert_action_norm_mean"] = ep_expert_action_norm_mean
            self.extras["ep_expert_update_frac"] = ep_expert_update_frac
            self.extras["ep_cmd_abs_x_mean"] = cmd_abs_x_mean
            self.extras["ep_cmd_abs_y_mean"] = cmd_abs_y_mean
            self.extras["ep_base_abs_x_mean"] = base_abs_x_mean
            self.extras["ep_base_abs_y_mean"] = base_abs_y_mean
            self.extras["ep_upright_frac"] = upright_frac
            self.extras["ep_swing_frac"] = swing_frac

        super().reset_idx(env_ids)

        if len(env_ids) !=0:
            if "episode" in self.extras:
                self.extras["episode"]["camera_jitter_mean"] = ep_camera_jitter
                self.extras["episode"]["camera_wobble_mean"] = ep_camera_wobble
                self.extras["episode"]["camera_bobbing_mean"] = ep_camera_bobbing
                self.extras["episode"]["camera_penalty_mean"] = ep_camera_penalty
                self.extras["episode"]["camera_raw_ang_acc_xy_rms"] = ep_camera_raw_ang_acc_xy_rms
                self.extras["episode"]["camera_jitter_sat_frac"] = ep_camera_jitter_sat_frac
                self.extras["episode"]["foot_impact_df_mean"] = ep_foot_impact_df_mean
                self.extras["episode"]["foot_contact_event_rate"] = ep_foot_contact_event_rate
                self.extras["episode"]["foot_air_time_at_contact"] = ep_foot_air_time_at_contact
                self.extras["episode"]["foot_contact_fz_mean"] = ep_foot_contact_fz_mean
                self.extras["episode"]["nonfoot_contact_trigger_frac"] = ep_nonfoot_contact_trigger_frac
                self.extras["episode"]["base_ang_acc_xy_rms"] = base_ang_acc_xy_rms
                self.extras["episode"]["base_ang_vel_xy_rms"] = base_ang_vel_xy_rms
                self.extras["episode"]["base_lin_acc_z_rms"] = base_lin_acc_z_rms
                self.extras["episode"]["ep_camera_quality"] = ep_camera_quality
                self.extras["episode"]["ep_height_quality"] = ep_height_quality
                self.extras["episode"]["ep_collision_count"] = ep_collision_count
                self.extras["episode"]["ep_distance_traveled"] = ep_distance_traveled
                self.extras["episode"]["ep_cmd_norm_mean"] = ep_cmd_norm_mean
                self.extras["episode"]["ep_cmd_nonzero_frac"] = ep_cmd_nonzero_frac
                self.extras["episode"]["ep_base_speed_mean"] = ep_base_speed_mean
                self.extras["episode"]["ep_expert_action_norm_mean"] = ep_expert_action_norm_mean
                self.extras["episode"]["ep_expert_update_frac"] = ep_expert_update_frac
                self.extras["episode"]["cmd_abs_x_mean"] = cmd_abs_x_mean
                self.extras["episode"]["cmd_abs_y_mean"] = cmd_abs_y_mean
                self.extras["episode"]["base_abs_x_mean"] = base_abs_x_mean
                self.extras["episode"]["base_abs_y_mean"] = base_abs_y_mean
                self.extras["episode"]["upright_frac"] = upright_frac
                self.extras["episode"]["swing_frac"] = swing_frac
                # failure mode diagnostics
                ep_knee_contact_frac = (self.episode_debug_stats[env_ids, 0] / ep_len).mean().item()
                ep_thigh_contact_frac = (self.episode_debug_stats[env_ids, 1] / ep_len).mean().item()
                ep_dof_limit_violation = (self.episode_debug_stats[env_ids, 2] / ep_len).mean().item()
                ep_nonfoot_contact_frac = (self.episode_debug_stats[env_ids, 3] / ep_len).mean().item()
                self.extras["episode"]["knee_contact_frac"] = ep_knee_contact_frac
                self.extras["episode"]["thigh_contact_frac"] = ep_thigh_contact_frac
                self.extras["episode"]["dof_pos_limit_violation"] = ep_dof_limit_violation
                self.extras["episode"]["nonfoot_contact_frac"] = ep_nonfoot_contact_frac
            self.get_expert_actions()
            #可视化的轨迹线条清楚
            if self.viewer and self.foot_traj_viz:
                self.gym.clear_lines(self.viewer)            
        
            # ========== 关键重置 (修复遗漏的buffer重置) ==========
            # 1. 【GPT-Fix】重置速度历史 - 同步到当前速度避免差分尖峰
            # 修复前: last_root_vel[env_ids] = 0. → 导致episode第一步产生巨大加速度尖峰
            # 修复后: 同步到reset后的实际速度，避免假尖峰污染camera_quality
            self.last_root_vel[env_ids, :3] = self.root_states[env_ids, 7:10]   # lin_vel
            self.last_root_vel[env_ids, 3:] = self.root_states[env_ids, 10:13]  # ang_vel
            
            # 2. 【GPT建议】重置接触历史 - 同步到当前接触力，防止差分尖峰（原理同上）
            self.last_contact_forces[env_ids] = self.contact_forces[env_ids]
            
            # 3. 重置其他状态历史
            self.last_contacts[env_ids] = 0.
            if hasattr(self, "nonfoot_contact_streak"):
                self.nonfoot_contact_streak[env_ids] = 0
            if hasattr(self, "nonfoot_contact_trigger"):
                self.nonfoot_contact_trigger[env_ids] = False
            
            # 3. 重置角加速度 - 确保 camera_stability 奖励正常
            self.base_ang_acc[env_ids] = 0.
            
            # 4. 重置摆动相位状态 - 确保 swing 奖励正常
            self.reach_swing_init[env_ids] = False
            self.reach_rew_time[env_ids] = 0.0
            
            # 5. 【P0.3】重置 raw 统计 - 确保 curriculum 正常工作
            self.episode_raw_stats[env_ids] = 0.0
            self.episode_cmd_stats[env_ids] = 0.0
            self.episode_debug_stats[env_ids] = 0.0
            if hasattr(self, "episode_nonfoot_trigger_steps"):
                self.episode_nonfoot_trigger_steps[env_ids] = 0.0
            if hasattr(self, "episode_jitter_sat_steps"):
                self.episode_jitter_sat_steps[env_ids] = 0.0
            if hasattr(self, "episode_raw_ang_jitter_sum"):
                self.episode_raw_ang_jitter_sum[env_ids] = 0.0
            if hasattr(self, "episode_foot_dF_sum"):
                self.episode_foot_dF_sum[env_ids] = 0.0
            if hasattr(self, "episode_foot_contact_events"):
                self.episode_foot_contact_events[env_ids] = 0.0
            if hasattr(self, "episode_foot_contact_time_sum"):
                self.episode_foot_contact_time_sum[env_ids] = 0.0
            if hasattr(self, "episode_foot_contact_fz_sum"):
                self.episode_foot_contact_fz_sum[env_ids] = 0.0
            if hasattr(self, "episode_foot_contact_fz_count"):
                self.episode_foot_contact_fz_count[env_ids] = 0.0
            if hasattr(self, "_prev_feet_force_norm"):
                self._prev_feet_force_norm[env_ids] = torch.norm(
                    self.contact_forces[env_ids][:, self.feet_indices, :], dim=-1
                )
            # =====================================================

        # 重置深度图为零（避免使用旧数据）
        # P1-Depth: 清零两个buffer，shape固定无需None检查
        self.depth_raw[env_ids] = self.camera_cfg.far_clip  # 原始深度用far_clip表示无效
        self.depth_images[env_ids] = 0.0  # 网络输入用0
        
        # 重置目标位置（使用 local 坐标，传递子集）
        robot_pos_local = self.root_states[env_ids, :3] - self.env_origins[env_ids]
        self.nav_task.reset_goals(env_ids, robot_pos_local, min_distance=2.0, max_distance=8.0)

        # 同步 goal_buf（local -> robot frame）- 只更新 env_ids
        headings = self._yaw_from_quat(self.root_states[env_ids, 3:7])
        self.goal_buf[env_ids] = self.nav_task.get_relative_goal(robot_pos_local, headings, env_ids)

        # 同步 prev buffer（local）
        # 注意: robot_pos_local 已经是子集，直接赋值，不要再加 [env_ids]
        self.prev_robot_pos_buf[env_ids] = robot_pos_local
        self.prev_intensity_buf[env_ids] = self.intensity_buf[env_ids]

        
    def get_expert_actions(self):
        """
        Get expert actions for BC loss calculation
        
        【EGPO Architecture Critical】:
        - Expert actions are used for Behavior Cloning loss
        - Expert MUST see the SAME commands as the policy
        - Otherwise BC loss will pull policy in wrong direction
        
        【Implementation】:
        - Phase 1: Expert uses random commands (same as policy)
        - Phase 2: Expert uses goal-derived commands (same as policy)
        """
        # Sync expert with policy commands (CRITICAL for EGPO)
        cmd_to_use = self._get_effective_commands()
        
        # Build expert command: [reset, vx, vy, vz, omega_z]
        command = torch.stack([
            self.reset_buf.clone(),
            cmd_to_use[:, 0],
            cmd_to_use[:, 1],
            torch.zeros_like(self.reset_buf),
            cmd_to_use[:, 2]
        ], dim=1)
        
        expert_dofs = self.expert.ProcessCommand(command, self.dof_pos, self.dof_vel)
        self.expert_actions = ((expert_dofs - self.default_dof_pos) / 
                               self.cfg.control.action_scale).detach()
        self._expert_action_norm_buf = torch.norm(self.expert_actions, dim=1)
        self._expert_action_updated = True
        return self.expert_actions
    
    def get_observations_dict(self):
        """
        获取当前的观测字典
        用于训练时获取观测
        """
        foot_contact_forces = None
        if hasattr(self, "contact_forces"):
            foot_contact_forces = self.contact_forces[:, self.feet_indices, :]
        obs_dict = {
            'proprioception': self.obs_buf,
            'privileged': self.obs_vgf_buf,
            'terrain': self.obs_terrain_buf,
            'depth': self.depth_images if hasattr(self, 'depth_images') else None,
            'robot_state': self.robot_state_buf,
            'goal': self.goal_buf,
            'foot_contact_forces': foot_contact_forces
        }
        return obs_dict

    def _reset_dofs(self,env_ids):
        #不给初始关节角度添加随机值，因此重写
        self.dof_pos[env_ids]= self.default_dof_pos
        self.dof_vel[env_ids]= 0.
        env_ids_int32=env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32),
                                              len(env_ids_int32))


    def compute_observations_separated(self):
        """计算EGPO Encoder的分离式观测
        
        ============================================================
        EGPO观测计算 - 核心函数
        ============================================================
        
        输出三个独立的观测buffer:
        
        1. self.obs_buf (67维) - 本体观测
           [quat(4), ang_vel(3), lin_acc(3), dof_pos(18), 
            dof_vel(18), torque(18), command(3)]
           - 可在实际机器人上获取
           - 用于: Actor/Critic输入的一部分
        
        2. self.obs_vgf_buf (30维) - 特权物理观测
           [base_lin_vel(3), projected_gravity(3), rb_contact_force(24)]
           - 训练: 仿真器提供真值
           - 部署: Estimator从obs_buf估计
           - 用于: 拼接到obs_buf形成97维输入
        
        3. self.obs_terrain_buf (143维) - 特权地形观测
           [Raycast高度图: 11×13点]
           - 训练: Raycast直接采样地形
           - 部署: LSTM从历史观测估计
           - 用于: CNN Encoder提取32维terrain_latent
        
        数据流向Runner:
        obs_dict = {
            'proprioception': obs_buf (67),
            'privileged': obs_vgf_buf (30),
            'terrain': obs_terrain_buf (143)
        }
        
        Runner处理:
        obs_splice = concat([obs, obs_vgf]) = 97维
        terrain_latent = CNN_encoder(obs_terrain) = 32维
        network_input = concat([obs_splice, terrain_latent]) = 129维
        
        【格式约定】:
        - self.base_quat: [x,y,z,w] 格式（Isaac Gym标准）
        
        【Auto-reset保证】:
        - reset_idx()后必须调用此函数刷新观测
        ============================================================
        """
        # ============================================================
        # 计算地形相对高度 (143维)
        # ============================================================
        height=torch.clip((self.root_states[:,2].unsqueeze(1)-0.025-self.measured_heights),min=-1.0,max=1.0)

        # === P0.2: Commands\u5355\u4e00\u4e8b\u5b9e\u6e90 ===
        # \u5f3a\u5236\u4f7f\u7528 _get_effective_commands() \u907f\u514dBC loss\u6f02\u79fb
        obs_commands = self._get_effective_commands()

        # \u56db\u5143\u6570\u89c2\u6d4b: [x,y,z,w] \u683c\u5f0f
        # \u5173\u952e\uff1a\u4f7f\u7528 obs_commands \u800c\u975e self.commands
        self.obs_buf = torch.cat([self.base_quat*self.obs_scales.quat,
                                  self.base_ang_vel*self.obs_scales.ang_vel,
                                  self.base_lin_acc*self.obs_scales.lin_acc,
                                  (self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
                                  self.dof_vel*self.obs_scales.dof_vel,
                                  self.torques*self.obs_scales.dof_torque,
                                  obs_commands*self.commands_scale],dim=-1)
        # self.obs_buf = torch.cat([(self.last_actions*self.obs_scales.actions,
        #                           self.dof_pos-self.default_dof_pos)*self.obs_scales.dof_pos,
        #                           self.dof_vel*self.obs_scales.dof_vel,
        #                           self.torques*self.obs_scales.dof_torque,
        #                           self.commands*self.commands_scale],dim=-1)
        #为了对collision相关损失进行更加精准的预测，将刚体碰撞力放入观测中
        # 【P0.4维度说明】:
        # - obs看到的contact: penalised(3) + feet(6) = 9个刚体的force norm
        # - collision事件只检测penalised（非足端碰撞），见_compute_collision_mask()
        # - 这是设计意图：足端大力是正常运动，不触发collision事件
        rb_force_norm = torch.norm(self.contact_forces[:,torch.cat([self.penalised_contact_indices,self.feet_indices]),:], dim=-1)
        self.obs_vgf_buf = torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,
                                      self.projected_gravity*self.obs_scales.gravity,
                                      rb_force_norm*self.obs_scales.contact_force,], dim=-1)
                                    #   self.contact_forces[:,self.feet_indices,2]*self.obs_scales.contact_force],dim=-1)
        
        # self.obs_vgf_buf = torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,
        #                               self.projected_gravity*self.obs_scales.gravity,
        #                               self.contact_forces[:,self.feet_indices,2]*self.obs_scales.contact_force,
        #                               torch.norm(self.contact_forces[:,self.penalised_contact_indices,:])],dim=-1), dim=-1)        
        self.obs_terrain_buf = height*self.obs_scales.height_measurements

        
        if self.add_noise:
            self.obs_buf += (2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec[:self.num_obs]
            self.obs_vgf_buf += (2*torch.rand_like(self.obs_vgf_buf)-1)*self.noise_scale_vec[self.num_obs : self.num_obs+30]
            self.obs_terrain_buf += (2*torch.rand_like(self.obs_terrain_buf)-1)*self.noise_scale_vec[self.num_obs+30:]

        #新增提取机器人状态
        self._extract_robot_state()
        #新增更新目标位置
        self._update_goal_buffer()
        #添加噪声

    def _extract_robot_state(self):
        """
        从完整观测中提取高层需要的机器人状态
        
        【格式约定 - CRITICAL】:
        - 四元数格式: [x, y, z, w] (Isaac Gym 标准)
        - 与 _yaw_from_quat() 保持完全一致
        - 与 quat_rotate_inverse() 输入格式一致
        
        【输出状态】:
        - [x, y, z, vx, vy, yaw, roll, pitch, yaw_rate]
        - 所有欧拉角从相同四元数格式派生
        
        Shape: (num_envs, 9)
        """
        # 提取yaw角（从quaternion）
        # quaternion: [x, y, z, w] - Isaac Gym标准格式
        quat = self.base_quat
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = torch.atan2(
            2.0 * (w*z + x*y),
            1.0 - 2.0 * (y*y + z*z)
        )
        
        # roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
        roll = torch.atan2(
            2.0 * (w*x + y*z),
            1.0 - 2.0 * (x*x + y*y)
        )
        
        # pitch = asin(2*(w*y - z*x))
        pitch = torch.asin(torch.clamp(2.0 * (w*y - z*x), -1.0, 1.0))
        
        # 机器人相对起点的位置
        pos_x = self.root_states[:, 0] - self.env_origins[:, 0]
        pos_y = self.root_states[:, 1] - self.env_origins[:, 1]
        
        # 机器人高度（相对地面）
        if isinstance(self.measured_heights, torch.Tensor) and self.measured_heights.numel() > 0:
            height = self.root_states[:, 2] - torch.mean(self.measured_heights, dim=1)
        else:
            height = self.root_states[:, 2]  # 使用绝对高度
        
        self.robot_state_buf = torch.stack([
            pos_x,                      # 0: x位置
            pos_y,                      # 1: y位置
            yaw,                        # 2: yaw角
            self.base_lin_vel[:, 0],    # 3: vx
            self.base_lin_vel[:, 1],    # 4: vy
            self.base_ang_vel[:, 2],    # 5: omega
            height,                     # 6: height
            roll,                       # 7: roll
            pitch                       # 8: pitch
        ], dim=1)

    def _update_goal_buffer(self):
        """
        更新目标位置 buffer

        约定：
        - self.goal_buf 始终表示 “机器人坐标系下的相对 goal (x,y)”
        - 当启用导航奖励时：goal 由 NavigationTaskManager 维护（world-goal），这里仅做坐标变换同步
        - 当未启用导航奖励时：使用旧的 goal_mode 生成逻辑（但修正 random 为相对坐标）
        """
        # 启用导航奖励：以 nav_task 为准
        if getattr(self.nav_cfg, "enable_nav_reward", False):
            # nav_task 维护的是 local 坐标系 goal_positions: (N,2)
            if not hasattr(self, "nav_task"):
                return

            # 需要机器人heading（yaw）。
            # 确保 robot_state_buf 是最新的：调用顺序上 compute_observations_separated 会先 _extract_robot_state()
            headings = self.robot_state_buf[:, 2]

            # local -> robot frame relative goal
            robot_pos_local = self.root_states[:, :3] - self.env_origins
            # P1-GoalBuf: 就地写入，禁止重新绑定
            rel_goal = self.nav_task.get_relative_goal(robot_pos_local, headings)
            self.goal_buf[:] = rel_goal
            return

        # === Phase 1: 未启用导航奖励，沿用旧逻辑（但统一为相对坐标） ===
        # [P0.2 数据一致性] 【GPT澄清】此分支中self.commands是正确的指令来源
        # - Phase 1: self.commands = 真实指令（基类随机采样+curriculum）
        # - tracking reward应该跟踪self.commands（不是effective_commands）
        # - obs/expert使用effective_commands只是为了Phase 2一致性（Phase 1中两者相同）
        if self.nav_cfg.goal_mode == 'velocity_based':
            goal_distance = self.nav_cfg.goal_distance
            vel_norm = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
            vel_norm = torch.clamp(vel_norm, min=0.1)
            goal_direction = self.commands[:, :2] / vel_norm
            # P1-GoalBuf: 就地写入
            self.goal_buf[:] = goal_direction * goal_distance  # 相对坐标（机器人坐标系近似/或世界系相对偏移）

        elif self.nav_cfg.goal_mode == 'fixed':
            fixed_goal = torch.tensor(self.nav_cfg.fixed_goal, device=self.device)  # world
            # 转成“世界系相对位移”
            self.goal_buf[:] = fixed_goal.unsqueeze(0).expand(self.num_envs, -1) - self.root_states[:, :2]

        elif self.nav_cfg.goal_mode == 'random':
            # 先采样 world goal
            goal_world = torch.zeros(self.num_envs, 2, device=self.device)
            goal_world[:, 0] = torch.rand(self.num_envs, device=self.device) * \
                (self.nav_cfg.goal_range_x[1] - self.nav_cfg.goal_range_x[0]) + self.nav_cfg.goal_range_x[0]
            goal_world[:, 1] = torch.rand(self.num_envs, device=self.device) * \
                (self.nav_cfg.goal_range_y[1] - self.nav_cfg.goal_range_y[0]) + self.nav_cfg.goal_range_y[0]

            # 再转成相对坐标（世界系相对位移）(P1-GoalBuf: 就地写入)
            self.goal_buf[:] = goal_world - self.root_states[:, :2]


    def get_observations_separated(self):
        return self.obs_buf, self.obs_vgf_buf, self.obs_terrain_buf

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

        # === 非足端接触惩罚滞后（防噪声误触发） ===
        self.nonfoot_contact_streak = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device, requires_grad=False
        )
        self.nonfoot_contact_trigger = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.episode_nonfoot_trigger_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )

        # === B1: 相机/接触冲击诊断统计（episode累加） ===
        self.episode_jitter_sat_steps = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.episode_raw_ang_jitter_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.episode_foot_dF_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        # === Feet contact timing stats (episode accumulation) ===
        self.episode_foot_contact_events = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.episode_foot_contact_time_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.episode_foot_contact_fz_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.episode_foot_contact_fz_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._prev_feet_force_norm = torch.zeros(
            self.num_envs,
            int(self.feet_indices.shape[0]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
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
        """生成EGPO Encoder的噪声向量
        
        ============================================================
        EGPO噪声配置 - 分离结构
        ============================================================
        
        噪声向量维度: 240 = obs(67) + vgf(30) + terrain(143)
        
        结构:
        [0:67]   - obs_buf噪声: [quat(4), ang_vel(3), lin_acc(3), 
                                  dof_pos(18), dof_vel(18), torque(18), 
                                  command(3)]
        [67:97]  - obs_vgf_buf噪声: [lin_vel(3), gravity(3), 
                                      contact_force(24)]
        [97:240] - obs_terrain_buf噪声: [height_measurements(143)]
        
        Domain Randomization:
        - obs: 传感器噪声模拟（IMU、编码器）
        - vgf: 物理量估计误差（速度、接触力）
        - terrain: 地形感知噪声（高度图误差）
        
        训练效果:
        - 提高策略鲁棒性
        - 帮助Estimator和LSTM学习噪声抑制
        - 平滑sim-to-real转换
        ============================================================
        """
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # EGPO使用分离的观测结构: obs(67) + vgf(30) + terrain(143) = 240
        # 创建完整的噪声向量，而不是依赖 privileged_obs_buf 的大小
        noise_vec = torch.zeros(67 + 30 + 143, dtype=torch.float, device=self.device, requires_grad=False)

        print("---------->noise_vec.shape=",noise_vec.shape)

        # [quat(4), ang_vel(3), lin_acc(3), dof_pos(18), dof_vel(18), dof_torque(18), command(3)]
        noise_vec[:4] = noise_level * noise_scales.quat * self.obs_scales.quat
        noise_vec[4:7] = noise_level * noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_level * noise_scales.lin_acc * self.obs_scales.lin_acc
        noise_vec[10:28] = noise_level * noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[28:46] = noise_level * noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[46:64] = noise_level * noise_scales.dof_torque * self.obs_scales.dof_torque
        noise_vec[64:67] = 0.0 #command
        
        # === P2.2: 移除条件门控，分离结构下vgf/terrain必然存在 ===
        # 【GPT审查强化】此处假设分离结构已确保vgf和terrain可用
        # [lin_vel(3), gravity(3), contact_force(24), measured_heights(143)]
        # 注: obs_vgf_buf总维度30，contact_force实际计算见Line 1189
        noise_vec[67:70] = noise_level * noise_scales.lin_vel * self.obs_scales.lin_vel
        noise_vec[70:73] = noise_level * noise_scales.gravity * self.obs_scales.gravity
        noise_vec[73:97] = noise_level * noise_scales.contact_force * self.obs_scales.contact_force
        noise_vec[97:] = noise_level * noise_scales.height_measurements * self.obs_scales.height_measurements
        
        return noise_vec
    
    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        #这样随机采样出来的有绝对的只有x或者只有y
        self.commands[env_ids,0] *= torch.abs(self.commands[env_ids,0])>0.1
        self.commands[env_ids,1] *= torch.abs(self.commands[env_ids,1])>0.2
        self.commands[env_ids,2] *= torch.abs(self.commands[env_ids,2])>0.2
        # self.commands[:,1].fill_(0.7)
        # Avoid extra norm-based zeroing (can unintentionally suppress small-but-valid pure-x lateral commands)

    def _update_terrain_curriculum(self, env_ids):
        """=== P1.1: Curriculum重构 - 基于raw统计的稳定升级 ===
        
        【问题诊断】:
        原实现依赖 episode_sums (= raw * scale * dt)
        → 改变scale或dt会破坏阈值稳定性
        → 单次成功立即升级，导致抖动
        
        【解决方案】:
        1. 使用 episode_raw_stats (物理量，不受scale影响)
        2. 多指标质量评分 (4项中至少3项达标)
        3. 软升级机制 (连续2次通过才升级)
        4. 配置参数化 (阈值可调)
        """
        if not self.init_done:
            return
        
        # 初始化升级计数器 (首次调用时)
        if not hasattr(self, 'terrain_pass_count'):
            self.terrain_pass_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # 获取配置阈值
        cfg_t = self.cfg.terrain
        stability_th = getattr(cfg_t, 'curriculum_stability_threshold', 0.7)
        height_th = getattr(cfg_t, 'curriculum_height_threshold', 0.7)
        collision_th = getattr(cfg_t, 'curriculum_collision_threshold', 5.0)
        distance_floor = getattr(cfg_t, 'curriculum_distance_threshold', 0.0)
        distance_k = getattr(cfg_t, 'curriculum_distance_k', 0.8)
        min_cmd = getattr(cfg_t, 'curriculum_min_command', 0.1)
        quality_score = getattr(cfg_t, 'curriculum_quality_score', 3.0)
        consecutive_passes = getattr(cfg_t, 'curriculum_consecutive_passes', 2)
        cap_start = getattr(cfg_t, 'curriculum_expert_level_cap_start', 0)
        cap_end = getattr(cfg_t, 'curriculum_expert_level_cap_end', -1)
        freeze_iters = getattr(cfg_t, 'curriculum_post_expert_freeze_iters', 0)

        train_iter = getattr(self, "_train_iter", None)
        expert_iter = getattr(self, "_expert_interface_iter", None)
        cap_level = None
        if train_iter is not None and expert_iter is not None and expert_iter > 0:
            if cap_end < 0:
                cap_end = self.max_terrain_level - 1
            progress = min(float(train_iter) / expert_iter, 1.0)
            cap_level = int(math.floor(cap_start + (cap_end - cap_start) * progress))
            cap_level = max(0, min(cap_level, self.max_terrain_level - 1))
        freeze_upgrading = False
        if train_iter is not None and expert_iter is not None and freeze_iters > 0:
            freeze_upgrading = train_iter >= expert_iter and train_iter < expert_iter + freeze_iters
        
        # 从 episode_raw_stats 提取指标 [camera_stability, base_height, collision_count, distance]
        # 【GPT审查修正】用实际步数而非max_episode_length求均值
        episode_length = self.episode_length_buf[env_ids].float()
        camera_stability = self.episode_raw_stats[env_ids, 0] / (episode_length + 1e-6)
        base_height = self.episode_raw_stats[env_ids, 1] / (episode_length + 1e-6)
        collision_count = self.episode_raw_stats[env_ids, 2]  # 总事件数
        
        # 质量评分 (4项布尔指标)
        pass_stability = camera_stability > stability_th
        pass_height = base_height > height_th
        pass_collision = collision_count < collision_th
        
        # 距离指标 (基于episode累计距离，且必须对齐指令强度)
        distance_traveled = self.episode_raw_stats[env_ids, 3]
        cmd_norm_mean = self.episode_cmd_stats[env_ids, 0] / (episode_length + 1e-6)
        episode_time = episode_length * self.dt
        target_distance = torch.clamp(cmd_norm_mean * episode_time * distance_k, min=distance_floor)
        pass_distance = (cmd_norm_mean >= min_cmd) & (distance_traveled > target_distance)
        
        # 计算质量分数 (4项中满足几项)
        score = pass_stability.float() + pass_height.float() + pass_collision.float() + pass_distance.float()
        
        # 软升级: 连续通过计数
        current_pass = (score >= quality_score) & pass_distance
        if freeze_upgrading:
            current_pass = torch.zeros_like(current_pass, dtype=torch.bool)
        self.terrain_pass_count[env_ids] = torch.where(
            current_pass,
            self.terrain_pass_count[env_ids] + 1,
            torch.zeros_like(self.terrain_pass_count[env_ids])  # 失败则清零
        )
        
        # 升级条件: 连续N次通过
        move_up = self.terrain_pass_count[env_ids] >= consecutive_passes
        if freeze_upgrading:
            move_up = torch.zeros_like(move_up)
        
        # 降级条件: 分数低于2分
        move_down = (score < 2.0) & ~move_up
        
        # 执行升降级
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        if cap_level is not None:
            self.terrain_levels[env_ids] = torch.clamp(self.terrain_levels[env_ids], 0, cap_level)
        
        # 重置升级计数器 (升级后清零，避免连续跳级)
        self.terrain_pass_count[env_ids] = torch.where(
            move_up,
            torch.zeros_like(self.terrain_pass_count[env_ids]),
            self.terrain_pass_count[env_ids]
        )

        # TB日志：curriculum质量与通过率
        self.extras["curr_camera_quality_mean"] = camera_stability.mean().item()
        self.extras["curr_height_quality_mean"] = base_height.mean().item()
        self.extras["curr_collision_mean"] = collision_count.mean().item()
        self.extras["curr_distance_mean"] = distance_traveled.mean().item()
        self.extras["curr_cmd_norm_mean"] = cmd_norm_mean.mean().item()
        self.extras["curr_target_distance_mean"] = target_distance.mean().item()
        self.extras["curr_pass_distance_rate"] = pass_distance.float().mean().item()
        self.extras["curr_score_mean"] = score.mean().item()
        self.extras["curr_pass_rate"] = current_pass.float().mean().item()
        self.extras["curr_move_up_rate"] = move_up.float().mean().item()
        
        # 更新环境原点
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
        """Reward reasonable swing time, but avoid 'very long air-time' reward hacking.

        - Uses contact filtering for mesh noise.
        - Rewards only on first contact event.
        - Saturates and adds penalty when air time is too long.
        - Gates reward when body is not upright or when knee/thigh are in contact.
        """
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.) * contact_filt

        # update timers
        self.feet_air_time += self.dt

        # shaping parameters (seconds)
        t_target = getattr(self.cfg.rewards, "feet_air_time_target_s", 0.18)
        t_max = getattr(self.cfg.rewards, "feet_air_time_max_s", 0.45)
        long_penalty = getattr(self.cfg.rewards, "feet_air_time_long_penalty", 1.0)
        over_cap = getattr(self.cfg.rewards, "feet_air_time_over_cap_s", 0.60)
        cmd_th = getattr(self.cfg.rewards, "feet_air_time_cmd_threshold", 0.2)

        t = self.feet_air_time
        t_good = torch.clamp(torch.clamp(t, max=t_max) - t_target, min=0.0)
        t_over = torch.clamp(t - t_max, min=0.0, max=over_cap)
        per_foot = t_good - long_penalty * t_over

        rew_air_time = torch.sum(per_foot * first_contact, dim=1)
        cmd_gate = (torch.norm(self.commands[:, :3], dim=1) > cmd_th)
        rew_air_time *= cmd_gate

        # Gate: only count when body is upright and no knee/thigh contact is happening
        upright_cos_min = getattr(self.cfg.rewards, "upright_cos_min", 0.75)
        # projected_gravity is gravity expressed in base frame; upright => gravity points to -Z in base frame.
        upright = self.projected_gravity[:, 2] < -upright_cos_min
        collision_th = getattr(self.cfg.terrain, "collision_penalty_threshold", None)
        no_nonfoot_contact = ~self._compute_collision_mask(threshold=collision_th)
        rew_air_time *= (upright & no_nonfoot_contact).float()

        # Penalize excessive air-time even without toe contact (avoid "never touch" loophole).
        miss_penalty = (t_over * (~contact_filt)).mean(dim=1) * long_penalty
        miss_penalty *= cmd_gate
        miss_penalty *= upright.float()
        rew_air_time -= miss_penalty

        # reset on (filtered) contact
        self.feet_air_time *= ~contact_filt
        return rew_air_time
    
    def _reward_footend_pos_xy(self):
        """Continuous foot placement shaping (swing legs only), with deadzone for obstacle tolerance."""
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)
        swing_mask = ~contact_filt  # (N,6)

        # forward kinematics for current foot positions (body/leg base frame)
        self.expert.kin.ForwardKin(self.dof_pos.view(-1, 3), self.expert.B_e_cur_flat)
        dist_xy = torch.norm(
            self.expert.B_e_cur[..., 0:2] - self.expert.swing_init_point[:, 0:2],
            dim=-1,
        )  # (N,6)

        # allow larger deviations on harder terrain (curriculum)
        if hasattr(self, "terrain_levels") and hasattr(self, "max_terrain_level"):
            denom = max(1, int(self.max_terrain_level - 1))
            difficulty = torch.clamp(self.terrain_levels.float() / denom, 0.0, 1.0).unsqueeze(1)
        else:
            difficulty = torch.zeros(self.num_envs, 1, device=self.device)

        deadzone_base = getattr(self.cfg.rewards, "foot_xy_deadzone_base", 0.03)
        deadzone_extra = getattr(self.cfg.rewards, "foot_xy_deadzone_per_difficulty", 0.03)
        sigma_base = getattr(self.cfg.rewards, "foot_xy_sigma_base", 0.10)
        sigma_extra = getattr(self.cfg.rewards, "foot_xy_sigma_per_difficulty", 0.08)
        reward_min = getattr(self.cfg.rewards, "foot_xy_reward_min", 0.0)
        reward_max = getattr(self.cfg.rewards, "foot_xy_reward_max", 1.0)

        deadzone = deadzone_base + deadzone_extra * difficulty  # (N,1)
        sigma = sigma_base + sigma_extra * difficulty           # (N,1)

        dist_excess = torch.clamp(dist_xy - deadzone, min=0.0)
        per_leg_quality = torch.exp(-torch.square(dist_excess / (sigma + 1e-6)))
        per_leg_quality = torch.clamp(per_leg_quality, reward_min, reward_max)

        # average only across swing legs to avoid over-constraining stance/obstacle contacts
        swing_count = swing_mask.sum(dim=1).clamp_min(1)
        quality = (per_leg_quality * swing_mask.float()).sum(dim=1) / swing_count

        # only apply when commands are non-trivial
        quality *= (torch.norm(self.commands[:, :3], dim=1) > 0.2).float()

        # Gate: disable shaping when posture is abnormal or when non-foot bodies are touching (anti-hack)
        upright_cos_min = getattr(self.cfg.rewards, "upright_cos_min", 0.75)
        upright = self.projected_gravity[:, 2] < -upright_cos_min
        collision_th = getattr(self.cfg.terrain, "collision_penalty_threshold", None)
        no_nonfoot_contact = ~self._compute_collision_mask(threshold=collision_th)
        quality *= (upright & no_nonfoot_contact).float()
        return quality

    def _reward_swing(self):
        #估计摆动时，靠近设置的初始点，来避免长期运动带来的累计误差
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts) 
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
        smooth_rew *= torch.norm(self.commands[:,:3],dim=1)>0.2
        return smooth_rew
        
        # print("valid_mask=",reaching_mask[0])
        # print("err=",err[0])
        rew = torch.exp(-err/0.8)*reaching_mask
        reached_mask=(~contact_filt) & self.reach_swing_init
        rew[reached_mask]=math.exp(-0.15/0.8)
        # print("rew\n",rew[0])
        leg_rew=torch.sum(rew,dim=1)/(torch.sum(~contact_filt,dim=1)+1e-6)
        leg_rew *= torch.norm(self.commands[:,:3],dim=1)>0.2
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
        """零指令时惩罚基座移动（修复版）
        
        问题诊断 (2025-12-17):
        原实现惩罚关节偏离 → 策略学会"永远移动"避免关节误差
        导致 stand_still: -0.76，Mean Reward崩溃到0.0
        
        正确实现: 直接惩罚基座速度
        """
        # 基座水平速度 (x, y)
        base_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        
        # 零指令判断（更严格的阈值）
        is_zero_command = (torch.norm(self.commands[:, :3], dim=1) < 0.1)
        
        # 返回速度惩罚（零指令时生效）
        return base_speed * is_zero_command

    def _base_height_quality(self):
        #修改成正的奖励，越靠近目标值，奖励越高
        # print("in reward_base_height, base_height=",torch.mean(self.root_states[:,2].unsqueeze(1)-self.measured_heights,dim=1))
        # print("robot_states z=",self.root_states[0,2])
        # print("self.measured_heights=",self.measured_heights.mean())
        height = torch.clip(self.root_states[:,2].unsqueeze(1)-0.025-self.measured_heights, min=-1, max=1.0)
        base_height = torch.mean(height,dim=1)
        err = torch.abs(base_height-self.cfg.rewards.base_height_target)
        # print("err = ",err)
        quality = torch.exp(-err/0.03)
        return quality, base_height

    def _reward_base_height(self):
        quality, base_height = self._base_height_quality()
        reward = quality
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

    def _reward_collision(self):
        """Penalize non-foot contacts on unique penalised bodies.

        Note: `cfg.asset.penalize_contacts_on` may include duplicated substrings (e.g. "knee") to
        shape observation dimensions; reward should not double-count the same rigid body.
        """
        if not hasattr(self, "_penalised_contact_indices_unique"):
            self._penalised_contact_indices_unique = torch.unique(self.penalised_contact_indices)

        collision_threshold = getattr(self.cfg.terrain, "collision_penalty_threshold", None)
        if collision_threshold is None:
            collision_threshold = getattr(self.cfg.terrain, "collision_force_threshold", 1.0)

        # A: 使用滞后触发（由 post_physics_step_separate() 统一更新），防止接触噪声误罚
        return self.nonfoot_contact_trigger.float()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error *= lin_vel_error>0.1 #小于0.1的速度误差对机器人来说一样，可以鼓励优化其他部分而不是牺牲自然状态追求高精度的速度跟踪
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error *=ang_vel_error>0.2
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)    
    def _reward_tracking_dof(self):
        action_scaled = self.actions * self.cfg.control.action_scale
        pos_err = (action_scaled+self.default_dof_pos) - self.dof_pos
        pos_err *= pos_err>0.15

        return torch.square(pos_err).sum(dim=1)

    def _reward_tripod_gait(self):
        """Encourage expert-style tripod gait (3 legs stance, 3 legs swing).

        Uses the same A/B leg grouping as `ExpertGround`:
        - A group: [0,1,5]
        - B group: [2,3,4]

        Reward is high when either:
        - A is in contact (stance) and B is in swing (no contact), or
        - B is in contact (stance) and A is in swing (no contact).
        """
        if not hasattr(self, "expert") or not hasattr(self.expert, "A_group_index"):
            return torch.zeros(self.num_envs, device=self.device)

        foot_contact_threshold = getattr(self.cfg.rewards, "feet_contact_force_threshold", 1.0)
        contact = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > foot_contact_threshold
        contact_filt = torch.logical_or(contact, self.last_contacts)  # (N,6)

        A = self.expert.A_group_index
        B = self.expert.B_group_index

        stance_w = float(getattr(self.cfg.rewards, "tripod_stance_weight", 0.4))
        swing_w = float(getattr(self.cfg.rewards, "tripod_swing_weight", 0.6))
        cmd_th = float(getattr(self.cfg.rewards, "tripod_cmd_threshold", 0.2))

        # Pattern A stance / B swing
        stance_A = contact_filt[:, A].float().mean(dim=1)
        swing_B = (~contact_filt[:, B]).float().mean(dim=1)
        qA = stance_w * stance_A + swing_w * swing_B

        # Pattern B stance / A swing
        stance_B = contact_filt[:, B].float().mean(dim=1)
        swing_A = (~contact_filt[:, A]).float().mean(dim=1)
        qB = stance_w * stance_B + swing_w * swing_A

        quality = torch.maximum(qA, qB)

        # Only apply when commands are non-trivial
        quality *= (torch.norm(self.commands[:, :3], dim=1) > cmd_th).float()

        # Gate: disable shaping when posture is abnormal or when non-foot bodies are touching
        upright_cos_min = getattr(self.cfg.rewards, "upright_cos_min", 0.75)
        upright = self.projected_gravity[:, 2] < -upright_cos_min
        collision_th = getattr(self.cfg.terrain, "collision_penalty_threshold", None)
        no_nonfoot_contact = ~self._compute_collision_mask(threshold=collision_th)
        quality *= (upright & no_nonfoot_contact).float()

        return quality
    
    def _reward_camera_stability(self, return_terms=False):
        """相机稳定性奖励 - 通过惩罚机身抖动提升视觉质量
        
        【Sim-to-Real关键】:
        - Motion Blur根源: 角加速度（高频抖动）
        - 画面倾斜根源: pitch/roll角速度（持续晃动）
        - 垂直颠簸: z轴线加速度（深度估计误差）
        
        【实现逻辑】:
        - Jitter (抖动): 惩罚 base_ang_acc (rad/s²), 尤其xy分量
        - Wobble (晃动): 惩罚 base_ang_vel (rad/s), pitch/roll
        - Bobbing (颠簸): 惩罚 base_lin_acc (g), z分量
        
        【GPT-Fix】:
        - 对各项penalty加clamp，防止偶发尖峰污染整个episode的mean
        - 从config读取cap阈值，保证可复现可调试
        - 添加层级式诊断日志
        
        【GPT-Warning】:
        - base_lin_acc单位是g（见Line 794: /9.81），不是m/s²
        - 因此z_bobbing的量纲是g²，与角加速度(rad/s²)²量纲不同
        - clamp值需基于此单位设置（bobbing_cap=100对应~10g）
        
        Returns:
            reward: (N,) 稳定性奖励，使用exp函数确保平滑梯度
        """
        # 1. 高频抖动（Motion Blur的直接原因）- 只关心pitch/roll加速度
        # 单位: (rad/s²)²
        ang_jitter = torch.sum(torch.square(self.base_ang_acc[:, :2]), dim=1)
        
        # 2. 持续晃动（画面倾斜）- pitch/roll角速度  
        # 单位: (rad/s)²
        ang_wobble = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        
            # 3. 垂直颠簸（影响深度估计）
            # base_lin_acc 为“加速度计读数”（单位:g，包含重力），因此用 (a_z - 1g)^2 衡量上下颠簸与失重/跳跃
        z_bobbing = torch.square(self.base_lin_acc[:, 2] - 1.0)
        
        # 【GPT建议】从config读取cap阈值，避免硬编码暗参
        # 使得调参可复现、可追溯（cfg文件记录所有超参）
        jitter_cap = getattr(self.cfg.rewards, 'camera_jitter_cap', 50.0)
        wobble_cap = getattr(self.cfg.rewards, 'camera_wobble_cap', 4.0)
        bobbing_cap = getattr(self.cfg.rewards, 'camera_bobbing_cap', 100.0)
        
        ang_jitter = torch.clamp(ang_jitter, 0, jitter_cap)
        ang_wobble = torch.clamp(ang_wobble, 0, wobble_cap)
        z_bobbing = torch.clamp(z_bobbing, 0, bobbing_cap)
        
        # 组合惩罚（使用可配置权重）
        jitter_w = getattr(self.cfg.rewards, 'camera_jitter_weight', 0.05)
        wobble_w = getattr(self.cfg.rewards, 'camera_wobble_weight', 0.5)
        bobbing_w = getattr(self.cfg.rewards, 'camera_bobbing_weight', 0.1)
        penalty = ang_jitter * jitter_w + ang_wobble * wobble_w + z_bobbing * bobbing_w
        
        # 【GPT建议】层级式日志键名，提升logger兼容性（很多logger对 '/' 分层有更好支持）
        if self.common_step_counter % 500 == 0 and hasattr(self, 'extras'):
            self.extras['debug/camera/ang_jitter_mean'] = ang_jitter.mean().item()
            self.extras['debug/camera/ang_wobble_mean'] = ang_wobble.mean().item()
            self.extras['debug/camera/z_bobbing_mean'] = z_bobbing.mean().item()
            self.extras['debug/camera/penalty_mean'] = penalty.mean().item()
            self.extras['debug/camera/base_ang_acc_xy_std'] = self.base_ang_acc[:, :2].std().item()
            self.extras['debug/camera/base_lin_acc_z_std'] = self.base_lin_acc[:, 2].std().item()
        
        # 使用exp确保平滑梯度和正向奖励；并对异常姿态做门控，防止刷分
        camera_quality = torch.exp(-penalty)
        upright_cos_min = getattr(self.cfg.rewards, "upright_cos_min", 0.75)
        upright = self.projected_gravity[:, 2] < -upright_cos_min
        camera_quality = torch.where(upright, camera_quality, torch.zeros_like(camera_quality))
        
        # 【诊断日志】记录最终quality分布（层级式键名提升可见性）
        if self.common_step_counter % 500 == 0 and hasattr(self, 'extras'):
            self.extras['debug/camera/quality_mean'] = camera_quality.mean().item()
            self.extras['debug/camera/quality_min'] = camera_quality.min().item()
            self.extras['debug/camera/quality_max'] = camera_quality.max().item()
        
        if return_terms:
            return camera_quality, ang_jitter, ang_wobble, z_bobbing, penalty
        return camera_quality
        

if __name__ == '__main__':
    args=get_args()
    cfg = HexTerrainCfg()

    sim_params = {"sim":class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args,sim_params)
    env = HexTerrain(cfg,sim_params,args.physics_engine,args.sim_device,args.headless)
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.step(torch.zeros(env.num_envs,env.num_actions,dtype=torch.float,device=env.device))
        
