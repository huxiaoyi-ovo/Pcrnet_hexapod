"""
scripts/train_highlevel.py - V3.6 高层规划器训练脚本

版本: v3.6 (Production Ready with Squashed Gaussian + Beta)
作者: 胡潇屹
日期: 2025年12月

核心改动 (相比 V3.3):
1. state_dim=9: 匹配 hex_terrain.py 中的 robot_state_buf 定义
   [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
2. 使用 V3.6 TerrainAdaptivePlanner: Squashed Gaussian + Beta 分布
3. 集成完整 PPO 训练循环 (GAE, Clipped Loss)
4. 支持 LocomotionAdapter 的 Slew Rate Limiting

依赖:
- legged_gym (必须在 PYTHONPATH 中)
- rsl_rl (必须在 PYTHONPATH 中)
- rsl_rl.algorithms.high_level_planner (V3.6)
- legged_gym.envs.hex_v4.affordance_estimator (Phase 1)
- legged_gym.scripts.navigation_env (V2 Reward)

用法:
    1. 训练 Teacher:
       python scripts/train_highlevel.py --mode teacher --task hex_terrain \\
           --low_level_ckpt agents/fast_2000.pt

    2. 训练 Student (Distillation):
       python scripts/train_highlevel.py --mode student --task hex_terrain \\
           --low_level_ckpt agents/fast_2000.pt \\
           --teacher_ckpt outputs/planner/teacher/best_model.pt \\
           --vision_ckpt outputs/train_v4_fix/run_xxx/best_model.pt
"""

import os
import sys
import time
import argparse
import math
import types
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Optional
from collections import deque

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# V3.6 状态维度定义 (匹配 hex_terrain.py robot_state_buf)
STATE_DIM = 9   # [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
GOAL_DIM = 2    # [goal_x, goal_y] (相对坐标)
AFFORDANCE_CHANNELS = 2  # [occupancy, passable_gap]


def difficulty_from_gap(aff_map: torch.Tensor) -> torch.Tensor:
    if aff_map.ndim != 4 or aff_map.size(1) < 2:
        return torch.zeros(aff_map.shape[0], device=aff_map.device)
    gap = aff_map[:, 1]
    difficulty = 1.0 - gap.mean(dim=(1, 2))
    return torch.clamp(difficulty, 0.0, 1.0)


def import_modules():
    """延迟导入，带容错处理"""
    global task_registry, TerrainAdaptivePlanner, LocomotionAdapter
    global AffordanceEstimator, NavigationRewardFunction, NavigationRewardConfig
    global ActorCritic
    
    try:
        # 导入 legged_gym 注册环境
        import legged_gym.envs
        from legged_gym.utils import get_args, task_registry as tr
        task_registry = tr
        print("[Init] ✓ legged_gym 导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 'legged_gym': {e}")
        print("请确认 legged_gym 已安装并添加到 PYTHONPATH。")
        sys.exit(1)

    try:
        from rsl_rl.algorithms.high_level_planner import TerrainAdaptivePlanner as TAP
        from rsl_rl.algorithms.high_level_planner import LocomotionAdapter as LA
        from rsl_rl.modules import ActorCritic as AC
        
        # 赋值给全局变量
        globals()['TerrainAdaptivePlanner'] = TAP
        globals()['LocomotionAdapter'] = LA
        globals()['ActorCritic'] = AC
        print("[Init] ✓ rsl_rl 模块导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 rsl_rl 模块: {e}")
        sys.exit(1)

    try:
        from legged_gym.envs.hex_v4.affordance_estimator import AffordanceEstimator as AE
        globals()['AffordanceEstimator'] = AE
        print("[Init] ✓ AffordanceEstimator 导入成功")
    except ImportError as e:
        print(f"[Warning] AffordanceEstimator 导入失败: {e}")
        globals()['AffordanceEstimator'] = None

    try:
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardFunction as NRF
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardConfig as NRC
        globals()['NavigationRewardFunction'] = NRF
        globals()['NavigationRewardConfig'] = NRC
        print("[Init] ✓ NavigationReward 导入成功")
    except ImportError as e:
        print(f"[Warning] NavigationReward 导入失败: {e}")
        globals()['NavigationRewardFunction'] = None
        globals()['NavigationRewardConfig'] = None


class RolloutBuffer:
    """
    PPO Rollout Buffer - 存储一个 rollout 周期内的所有数据
    """
    def __init__(self, num_envs: int, num_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0
        
        # 观测 (存储字典的列表)
        self.obs_list = []
        
        # 动作
        self.subgoals = torch.zeros(num_steps, num_envs, 3, device=device)
        self.intensities = torch.zeros(num_steps, num_envs, device=device)
        
        # PPO 核心数据
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        
        # 蒸馏目标 (Student 模式)
        self.teacher_subgoals = torch.zeros(num_steps, num_envs, 3, device=device)
        self.teacher_intensities = torch.zeros(num_steps, num_envs, device=device)

    def add(self, obs_dict, subgoal, intensity, log_prob, value, reward, done,
            teacher_subgoal=None, teacher_intensity=None):
        """添加一步数据"""
        self.obs_list.append({k: v.clone() for k, v in obs_dict.items()})
        self.subgoals[self.step] = subgoal
        self.intensities[self.step] = intensity
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.squeeze(-1)
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        
        if teacher_subgoal is not None:
            self.teacher_subgoals[self.step] = teacher_subgoal
            self.teacher_intensities[self.step] = teacher_intensity
        
        self.step += 1

    def compute_returns(self, next_value: torch.Tensor, gamma: float = 0.99, 
                        gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 和 Returns"""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = next_value.squeeze(-1)
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + self.values
        return returns, advantages

    def reset(self):
        """重置 buffer"""
        self.step = 0
        self.obs_list = []
        self.subgoals.zero_()
        self.intensities.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.teacher_subgoals.zero_()
        self.teacher_intensities.zero_()


# V3 核心环境包装器 (The Hierarchical Environment Wrapper)
class HierarchicalHexapodEnv:
    """
    V3.6 分层环境包装器
    
    职责:
    1. 托管 Isaac Gym 底层环境 (hex_terrain)
    2. 托管 Low-Level Controller (冻结的底层策略)
    3. 托管 Locomotion Adapter (V3.6 Slew Rate Limiting)
    4. 托管 Reward Function (V2 强度适配奖励)
    5. 提供 Teacher/Student 两种数据流
    
    状态空间 (V3.6):
        robot_state: (N, 9) = [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
        goal: (N, 2) = [goal_x, goal_y]
        affordance: (N, 2, 16, 16) = [occupancy, passable_gap]
        terrain_difficulty: (N,)
    """
    
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.mode = args.mode
        
        # 初始化 Isaac Gym 环境
        print(f"[Env] 创建 Isaac Gym 环境: {args.task}")
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        if args.task == "hex_ground" and hasattr(env_cfg, "navigation"):
            env_cfg.navigation.goal_reached_threshold = 0.1

        # 覆盖配置以适配高层训练
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
        
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.num_envs = self.env.num_envs
        self.max_episode_length = self.env.max_episode_length

        cam_cfg = getattr(getattr(env_cfg, "sensor", None), "depth_camera", None)
        nav_cfg = getattr(env_cfg, "navigation", None)
        map_size = getattr(nav_cfg, "affordance_grid_size", None)
        if map_size is None:
            map_size = getattr(env_cfg.terrain, "affordance_grid_size", 16)
        map_size = int(map_size)
        cell_size = getattr(nav_cfg, "affordance_cell_size", None)
        if cell_size is None:
            cell_size = getattr(env_cfg.terrain, "affordance_cell_size", None)
        if cell_size is not None:
            map_extent = float(map_size * cell_size)
        elif cam_cfg is not None and hasattr(cam_cfg, "far_clip"):
            map_extent = float(cam_cfg.far_clip)
        else:
            map_extent = 5.0
        self.affordance_map_size = map_size
        self.affordance_map_extent = map_extent
        self.affordance_clearance = float(getattr(env_cfg.terrain, "fixed_layout_robot_clearance", 0.27))
        self.affordance_blocking_height = float(getattr(env_cfg.navigation, "goal_obstacle_height_threshold", 0.2))

        if hasattr(self.env, "_resample_commands"):
            def _no_resample(self, env_ids):
                return
            self.env._resample_commands = types.MethodType(_no_resample, self.env)
        
        # 加载 Low-Level Controller
        self._load_low_level_policy(args.low_level_ckpt)
        
        # 初始化 Locomotion Adapter (V3.6 with Slew Rate Limiting)
        self.adapter = LocomotionAdapter(
            max_linear_vel=0.5,
            max_angular_vel=0.8,
            min_speed_factor=0.4,
            max_intensity_change=0.1  # V3.6 关键参数
        )
        self.adapter.reset(self.num_envs, device)
        
        # 初始化 Reward Function
        if NavigationRewardConfig is not None:
            reward_kwargs = dict(
                goal_approach_scale=2.0,
                goal_reach_bonus=10.0,
                intensity_match_bonus=0.2,
                intensity_smooth_penalty=-0.05,
            )
            if args.task == "hex_ground":
                reward_kwargs.update(
                    goal_approach_scale=3.0,
                    goal_reach_threshold=0.1,
                    heading_scale=0.2,
                    heading_use_difficulty_gate=True,
                    heading_min_weight=0.2,
                    stability_scale=0.01,
                    time_penalty=-0.03,
                )
            self.reward_cfg = NavigationRewardConfig(**reward_kwargs)
            self.reward_func = NavigationRewardFunction(self.reward_cfg)
        else:
            self.reward_func = None
            print("[Warning] 使用环境原生奖励")
        
        # 状态缓冲区
        self.prev_robot_pos = torch.zeros(self.num_envs, 3, device=device)
        self.prev_intensity = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        
        # 频率控制 (High-Level 10Hz, Low-Level 50Hz)
        self.decimation = getattr(args, 'decimation', 5)
        
        print(f"[Env] 初始化完成: {self.num_envs} envs, decimation={self.decimation}")

    def _load_low_level_policy(self, ckpt_path: str):
        """加载并冻结底层控制器"""
        print(f"[Env] 加载底层策略: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"底层策略不存在: {ckpt_path}")
        
        # 加载 checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # 创建 ActorCritic (需要匹配底层网络结构)
        num_obs = self.env.num_obs
        num_actions = self.env.num_actions
        num_priv_obs = getattr(self.env, 'num_privileged_obs', num_obs)
        
        self.low_level_policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_priv_obs,
            num_actions=num_actions,
            actor_hidden_dims=[256, 256, 256],
            critic_hidden_dims=[256, 256, 256],
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in ckpt:
            self.low_level_policy.load_state_dict(ckpt['model_state_dict'])
        elif 'actor_state_dict' in ckpt:
            self.low_level_policy.actor.load_state_dict(ckpt['actor_state_dict'])
        else:
            print("[Warning] Checkpoint 格式未知，尝试直接加载")
            self.low_level_policy.load_state_dict(ckpt)
        
        # 冻结参数
        self.low_level_policy.eval()
        for param in self.low_level_policy.parameters():
            param.requires_grad = False
        
        print(f"[Env] ✓ 底层策略加载成功 (冻结)")

    def reset(self) -> Dict[str, torch.Tensor]:
        """复位环境"""
        obs, _ = self.env.reset()

        if hasattr(self.env, "commands"):
            self.env.commands[:, :3] = 0.0
            if hasattr(self.env, "commands_scale") and hasattr(self.env, "obs_buf"):
                if self.env.obs_buf.shape[1] >= 3:
                    self.env.obs_buf[:, -3:] = 0.0
        
        self.episode_length_buf.zero_()
        self.prev_intensity.zero_()
        self.adapter.reset(self.num_envs, self.device)

        self._refresh_depth_images(force=True)
        obs_dict = self._get_high_level_obs()
        self.prev_robot_pos = self.env.root_states[:, :3].clone()
        
        return obs_dict

    def _refresh_depth_images(self, force: bool = False) -> None:
        if not hasattr(self.env, "camera_cfg"):
            return
        if self.env.camera_cfg is None:
            return
        if not hasattr(self.env, "depth_images"):
            return

        if not getattr(self.env, "enable_camera", False):
            if hasattr(self.env, "depth_raw") and hasattr(self.env, "_process_depth_for_network"):
                self.env.depth_raw.fill_(self.env.camera_cfg.far_clip)
                processed = self.env._process_depth_for_network(self.env.depth_raw)
                self.env.depth_images[:] = processed
            else:
                self.env.depth_images.fill_(self.env.camera_cfg.far_clip)
            return

        if not force and self.env.common_step_counter % self.env.camera_cfg.capture_interval != 0:
            return

        if hasattr(self.env, "_get_depth_images") and hasattr(self.env, "_process_depth_for_network"):
            depth_raw = self.env._get_depth_images()
            processed = self.env._process_depth_for_network(depth_raw)
            self.env.depth_images[:] = processed

    def _compute_gt_affordance_from_heightfield(self) -> Optional[torch.Tensor]:
        if getattr(self.env, "height_samples", None) is None:
            return None
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = map_extent / map_size

        x_centers = torch.linspace(
            -map_extent * 0.5 + cell * 0.5,
            map_extent * 0.5 - cell * 0.5,
            map_size,
            device=self.device,
        )
        y_centers = torch.linspace(
            0.0 + cell * 0.5,
            map_extent - cell * 0.5,
            map_size,
            device=self.device,
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="xy")
        x_body = grid_x.reshape(-1).unsqueeze(0)
        y_body = grid_y.reshape(-1).unsqueeze(0)

        robot_xy = self.env.root_states[:, :2]
        if hasattr(self.env, "robot_state_buf"):
            yaw = self.env.robot_state_buf[:, 2]
        else:
            yaw = self._quat_to_yaw(self.env.root_states[:, 3:7])
        cos_h = torch.cos(yaw).unsqueeze(1)
        sin_h = torch.sin(yaw).unsqueeze(1)

        x_world = robot_xy[:, 0:1] + cos_h * x_body - sin_h * y_body
        y_world = robot_xy[:, 1:2] + sin_h * x_body + cos_h * y_body

        border = self.env.cfg.terrain.border_size
        scale = self.env.cfg.terrain.horizontal_scale
        max_x = self.env.height_samples.shape[0] - 2
        max_y = self.env.height_samples.shape[1] - 2
        idx_x = torch.clamp(((x_world + border) / scale).long(), 0, max_x)
        idx_y = torch.clamp(((y_world + border) / scale).long(), 0, max_y)

        heights = self.env.height_samples[idx_x, idx_y] * self.env.cfg.terrain.vertical_scale
        heights = heights.view(self.num_envs, map_size, map_size)

        occ_all = (heights > 1e-6).float()
        occ_block = (heights >= self.affordance_blocking_height).float().unsqueeze(1)

        radius_cells = int(math.ceil(self.affordance_clearance / cell))
        if radius_cells > 0:
            kernel = 2 * radius_cells + 1
            pooled = F.max_pool2d(occ_block, kernel_size=kernel, stride=1, padding=radius_cells)
            passable = (pooled <= 0.5) & (occ_block < 0.5)
        else:
            passable = occ_block < 0.5
        passable_gap = passable.float().squeeze(1)

        return torch.cat([occ_all.unsqueeze(1), passable_gap.unsqueeze(1)], dim=1)

    def _get_high_level_obs(self) -> Dict[str, torch.Tensor]:
        """
        构建高层观测字典
        
        Returns:
            Dict with keys:
            - state: (N, 9) robot state
            - goal: (N, 2) relative goal
            - gt_affordance: (N, 2, 16, 16) ground truth affordance
            - gt_difficulty: (N,) difficulty derived from passable_gap
            - depth: (N, 1, H, W) depth image
        """
        obs_dict = {}
        
        # 1. Robot State (V3.6: 9 维)
        if hasattr(self.env, 'robot_state_buf'):
            obs_dict['state'] = self.env.robot_state_buf.clone()
        else:
            # 手动构建
            pos = self.env.root_states[:, :2]
            quat = self.env.root_states[:, 3:7]
            yaw = self._quat_to_yaw(quat)
            lin_vel = self.env.base_lin_vel[:, :2]
            ang_vel = self.env.base_ang_vel[:, 2:3]
            height = self.env.root_states[:, 2:3]
            roll, pitch = self._quat_to_rp(quat)
            
            obs_dict['state'] = torch.cat([
                pos, yaw.unsqueeze(-1), lin_vel, ang_vel, 
                height, roll.unsqueeze(-1), pitch.unsqueeze(-1)
            ], dim=-1)
        
        # 2. Goal (相对坐标)
        if hasattr(self.env, 'goal_buf'):
            obs_dict['goal'] = self.env.goal_buf.clone()
        else:
            obs_dict['goal'] = self.env.commands[:, :2].clone()
        
        # 3. GT Affordance
        if hasattr(self.env, 'get_affordance_data'):
            aff_data = self.env.get_affordance_data()
            obs_dict['gt_affordance'] = aff_data['local_affordance']
        else:
            aff_map = self._compute_gt_affordance_from_heightfield()
            if aff_map is not None:
                obs_dict['gt_affordance'] = aff_map
            elif hasattr(self.env, 'measured_heights'):
                heights = self.env.measured_heights
                side_len = int(np.sqrt(heights.shape[1]))
                
                if side_len ** 2 == heights.shape[1]:
                    h_map = heights.view(self.num_envs, 1, side_len, side_len)
                    h_map = torch.nn.functional.interpolate(h_map, size=(16, 16), mode='bilinear')
                    obs_dict['gt_affordance'] = torch.cat([h_map, 1 - torch.abs(h_map)], dim=1)
                else:
                    obs_dict['gt_affordance'] = torch.zeros(self.num_envs, 2, 16, 16, device=self.device)
            else:
                obs_dict['gt_affordance'] = torch.zeros(self.num_envs, 2, 16, 16, device=self.device)
        obs_dict['gt_difficulty'] = difficulty_from_gap(obs_dict['gt_affordance'])
        
        # 4. Depth Image
        if hasattr(self.env, 'depth_images'):
            obs_dict['depth'] = self.env.depth_images.clone()
        elif hasattr(self.env, 'depth_buffer'):
            obs_dict['depth'] = self.env.depth_buffer.unsqueeze(1).clone()
        else:
            obs_dict['depth'] = torch.zeros(self.num_envs, 1, 128, 128, device=self.device)
        
        return obs_dict

    def step(self, subgoal: torch.Tensor, intensity: torch.Tensor
             ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        高层步进 (10Hz)
        
        Args:
            subgoal: (N, 3) [dx, dy, dyaw] 子目标
            intensity: (N,) 运动强度 [0, 1]
        
        Returns:
            obs_dict, rewards, dones, info
        """
        # 1. Adapter: 动作映射 (带 Slew Rate Limiting)
        velocity_cmd, adapter_info = self.adapter.convert(subgoal, intensity)

        # 2. Low-Level 控制循环 (50Hz)
        accumulated_reward = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for _ in range(self.decimation):
            if hasattr(self.env, "commands"):
                self.env.commands[:, :3] = velocity_cmd.detach()
                if hasattr(self.env, "commands_scale") and hasattr(self.env, "obs_buf"):
                    if self.env.obs_buf.shape[1] >= 3:
                        self.env.obs_buf[:, -3:] = velocity_cmd.detach() * self.env.commands_scale

            low_level_obs = self.env.obs_buf
            
            with torch.no_grad():
                actions = self.low_level_policy.act_inference(low_level_obs)
            
            obs, _, rewards, dones, infos = self.env.step(actions)
            accumulated_reward += rewards
            done_any |= dones
        
        self.episode_length_buf += 1
        
        # 3. 计算高层奖励
        self._refresh_depth_images()
        current_obs = self._get_high_level_obs()
        robot_pos = self.env.root_states[:, :3]
        
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self.env, "contact_forces") and hasattr(self.env, "termination_contact_indices"):
            contact_threshold = getattr(self.env.cfg.terrain, "collision_force_threshold", 1.0)
            collision_mask = torch.any(
                torch.norm(self.env.contact_forces[:, self.env.termination_contact_indices, :], dim=-1) > contact_threshold,
                dim=1,
            )

        if self.reward_func is not None:
            robot_vel = self.env.base_lin_vel
            robot_quat = self.env.root_states[:, 3:7]
            if hasattr(self.env, "goal_world"):
                goal_pos = self.env.goal_world.clone()
            else:
                yaw = current_obs['state'][:, 2]
                cos_h = torch.cos(yaw)
                sin_h = torch.sin(yaw)
                goal_local = current_obs['goal']
                goal_x = robot_pos[:, 0] + cos_h * goal_local[:, 0] - sin_h * goal_local[:, 1]
                goal_y = robot_pos[:, 1] + sin_h * goal_local[:, 0] + cos_h * goal_local[:, 1]
                goal_pos = torch.stack([goal_x, goal_y], dim=1)
            filtered_intensity = adapter_info['filtered_intensity'].squeeze(-1)
            
            reward_dict = self.reward_func.compute_reward(
                robot_pos=robot_pos,
                prev_robot_pos=self.prev_robot_pos,
                goal_pos=goal_pos,
                robot_vel=robot_vel,
                robot_quat=robot_quat,
                intensity=filtered_intensity,
                prev_intensity=self.prev_intensity,
                terrain_difficulty=current_obs['gt_difficulty'],
                collision_mask=collision_mask,
            )
            total_reward = reward_dict['total']
        else:
            total_reward = accumulated_reward / self.decimation
        
        # 4. 更新缓冲区
        self.prev_robot_pos = robot_pos.clone()
        self.prev_intensity = adapter_info['filtered_intensity'].squeeze(-1).clone()
        
        # 5. 处理超时
        timeout = self.episode_length_buf >= self.max_episode_length
        done_any |= timeout
        
        if done_any.any():
            self.episode_length_buf[done_any] = 0
            self.prev_intensity[done_any] = 0
            # ★ 修复幽灵动量问题: 重置 Adapter 的变化率限制记忆
            # 防止 Reset 后机器人因残留的 last_intensity 记忆而"猛冲"
            self.adapter.last_intensity[done_any] = 0.0
        
        info = {
            'adapter_info': adapter_info,
            'episode_length': self.episode_length_buf.clone(),
        }
        
        return current_obs, total_reward, done_any, info

    def _quat_to_yaw(self, quat: torch.Tensor) -> torch.Tensor:
        """四元数转 yaw 角"""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def _quat_to_rp(self, quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """四元数转 roll, pitch"""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return roll, pitch


# 训练循环 (PPO + Distillation)
def train(args):
    """主训练函数 (V3.6 完整 PPO 实现)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"V3.6 High-Level Planner Training")
    print(f"Mode: {args.mode.upper()}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 导入模块
    import_modules()
    
    # 创建环境
    env = HierarchicalHexapodEnv(args, device)
    print(f"[Main] 环境初始化完成: {env.num_envs} envs")
    
    # 创建 Planner (V3.6)
    planner = TerrainAdaptivePlanner(
        affordance_channels=AFFORDANCE_CHANNELS,
        state_dim=STATE_DIM,
        goal_dim=GOAL_DIM,
    ).to(device)
    
    optimizer = optim.Adam(planner.parameters(), lr=args.lr)
    
    # 加载 Teacher/Vision 模型 (Student 模式)
    teacher_model = None
    vision_model = None
    
    if args.mode == 'teacher':
        print("[Main] Mode: TEACHER. Training from scratch with GT.")
        
    elif args.mode == 'student':
        print("\n[Student] 加载 Teacher 和 Vision 模型...")
        
        # 加载 Teacher
        if args.teacher_ckpt:
            teacher_model = TerrainAdaptivePlanner(
                affordance_channels=AFFORDANCE_CHANNELS,
                state_dim=STATE_DIM,
                goal_dim=GOAL_DIM,
            ).to(device)
            
            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            if 'model_state_dict' in ckpt:
                teacher_model.load_state_dict(ckpt['model_state_dict'])
            else:
                teacher_model.load_state_dict(ckpt)
            teacher_model.eval()
            
            # 用 Teacher 权重初始化 Student
            planner.load_state_dict(teacher_model.state_dict())
            print(f"[Student] ✓ Teacher 加载成功: {args.teacher_ckpt}")
        
        # 加载 Vision (AffordanceEstimator)
        if args.vision_ckpt and AffordanceEstimator is not None:
            vision_model = AffordanceEstimator(
                depth_channels=1,
                output_size=16,
                max_depth_range=5.0
            ).to(device)
            
            ckpt = torch.load(args.vision_ckpt, map_location=device)
            if 'model_state_dict' in ckpt:
                vision_model.load_state_dict(ckpt['model_state_dict'])
            else:
                vision_model.load_state_dict(ckpt)
            vision_model.eval()
            print(f"[Student] ✓ Vision 加载成功: {args.vision_ckpt}")
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, f"{args.mode}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[Main] 日志目录: {log_dir}")
    
    # 创建 Rollout Buffer
    buffer = RolloutBuffer(env.num_envs, args.num_steps, device)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_reward = float('-inf')
    
    # 初始 reset
    obs_dict = env.reset()
    
    print(f"\n[Main] 开始训练 ({args.num_iterations} iterations)...")
    print(f"  - Steps per iteration: {args.num_steps}")
    print(f"  - Batch size: {env.num_envs * args.num_steps}")
    print(f"  - Learning rate: {args.lr}")
    
    for iteration in range(args.num_iterations):
        start_time = time.time()
        
        # ============ Rollout Phase ============
        buffer.reset()
        
        for step in range(args.num_steps):
            # 准备输入
            state = obs_dict['state']
            goal = obs_dict['goal']
            
            if args.mode == 'teacher':
                aff_map = obs_dict['gt_affordance']
                difficulty = obs_dict['gt_difficulty']
            else:
                # Student: 使用 Vision 模型预测
                if vision_model is not None:
                    with torch.no_grad():
                        vis_out = vision_model(obs_dict['depth'], normalize=True)
                        aff_map = torch.stack([
                            vis_out['occupancy'], 
                            vis_out['passable_gap']
                        ], dim=1)
                        difficulty = difficulty_from_gap(aff_map)
                else:
                    aff_map = obs_dict['gt_affordance']
                    difficulty = obs_dict['gt_difficulty']
            
            # Planner 决策
            with torch.no_grad():
                subgoal, intensity, info = planner.get_action(
                    aff_map, state, goal, difficulty, deterministic=False
                )
                
                log_prob, value, _, _ = planner.evaluate_actions(
                    aff_map, state, goal, difficulty, subgoal, intensity
                )
            
            # Teacher 目标 (Student 模式)
            teacher_subgoal = None
            teacher_intensity = None
            if args.mode == 'student' and teacher_model is not None:
                with torch.no_grad():
                    t_sub, t_int, _ = teacher_model.get_action(
                        obs_dict['gt_affordance'],
                        state, goal,
                        obs_dict['gt_difficulty'],
                        deterministic=True
                    )
                    teacher_subgoal = t_sub
                    teacher_intensity = t_int
            
            # 环境步进
            next_obs, rewards, dones, env_info = env.step(subgoal, intensity)
            
            # 存储数据
            buffer.add(
                obs_dict, subgoal, intensity, log_prob.sum(dim=-1), value,
                rewards, dones, teacher_subgoal, teacher_intensity
            )
            
            # 统计完成的 episode
            for i in range(env.num_envs):
                if dones[i]:
                    ep_len = env_info['episode_length'][i].item()
                    episode_lengths.append(ep_len)
                    episode_rewards.append(rewards[i].item() * max(ep_len, 1))
            
            obs_dict = next_obs
        
        # ============ Update Phase ============
        # 计算最后一步的 value (Bootstrap)
        with torch.no_grad():
            state = obs_dict['state']
            goal = obs_dict['goal']
            
            if args.mode == 'teacher':
                aff_map = obs_dict['gt_affordance']
                difficulty = obs_dict['gt_difficulty']
            else:
                if vision_model is not None:
                    vis_out = vision_model(obs_dict['depth'], normalize=True)
                    aff_map = torch.stack([vis_out['occupancy'], vis_out['passable_gap']], dim=1)
                    difficulty = difficulty_from_gap(aff_map)
                else:
                    aff_map = obs_dict['gt_affordance']
                    difficulty = obs_dict['gt_difficulty']
            
            out = planner(aff_map, state, goal, difficulty)
            next_value = out.value
        
        # 计算 Returns 和 Advantages
        returns, advantages = buffer.compute_returns(next_value, args.gamma, args.gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        distill_loss_sum = 0
        num_updates = 0
        
        batch_size = env.num_envs * args.num_steps
        
        for epoch in range(args.num_epochs):
            indices = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, args.mini_batch_size):
                end = min(start + args.mini_batch_size, batch_size)
                mb_indices = indices[start:end]
                
                step_idx = mb_indices // env.num_envs
                env_idx = mb_indices % env.num_envs
                
                # 获取 mini-batch 数据
                mb_subgoals = buffer.subgoals[step_idx, env_idx]
                mb_intensities = buffer.intensities[step_idx, env_idx]
                mb_old_log_probs = buffer.log_probs[step_idx, env_idx]
                mb_advantages = advantages[step_idx, env_idx]
                mb_returns = returns[step_idx, env_idx]
                
                # 重新构造观测
                mb_states = torch.stack([
                    buffer.obs_list[s.item()]['state'][e.item()] 
                    for s, e in zip(step_idx, env_idx)
                ])
                mb_goals = torch.stack([
                    buffer.obs_list[s.item()]['goal'][e.item()] 
                    for s, e in zip(step_idx, env_idx)
                ])
                mb_aff_maps = torch.stack([
                    buffer.obs_list[s.item()]['gt_affordance'][e.item()] 
                    for s, e in zip(step_idx, env_idx)
                ])
                mb_difficulties = torch.stack([
                    buffer.obs_list[s.item()]['gt_difficulty'][e.item()] 
                    for s, e in zip(step_idx, env_idx)
                ])
                
                # 评估当前策略
                new_log_probs, new_values, entropy, _ = planner.evaluate_actions(
                    mb_aff_maps, mb_states, mb_goals, mb_difficulties,
                    mb_subgoals, mb_intensities
                )
                new_log_probs = new_log_probs.sum(dim=-1)
                
                # PPO Clipped Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * ((new_values.squeeze(-1) - mb_returns) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = -entropy.mean()
                
                # 蒸馏 Loss (Student 模式)
                distill_loss = torch.tensor(0.0, device=device)
                if args.mode == 'student' and teacher_model is not None:
                    mb_teacher_subgoals = buffer.teacher_subgoals[step_idx, env_idx]
                    mb_teacher_intensities = buffer.teacher_intensities[step_idx, env_idx]
                    
                    distill_loss = (
                        nn.functional.mse_loss(mb_subgoals, mb_teacher_subgoals) +
                        nn.functional.mse_loss(mb_intensities, mb_teacher_intensities)
                    )
                
                # 总 Loss
                loss = (
                    policy_loss + 
                    args.value_loss_coef * value_loss + 
                    args.entropy_coef * entropy_loss +
                    args.distill_coef * distill_loss
                )
                
                # 优化
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(planner.parameters(), args.max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                distill_loss_sum += distill_loss.item()
                num_updates += 1
        
        # ============ Logging ============
        iter_time = time.time() - start_time
        fps = batch_size / iter_time
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        
        # TensorBoard
        writer.add_scalar('Loss/Total', total_loss / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Policy', policy_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Value', value_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Entropy', entropy_sum / max(num_updates, 1), iteration)
        if args.mode == 'student':
            writer.add_scalar('Loss/Distill', distill_loss_sum / max(num_updates, 1), iteration)
        
        writer.add_scalar('Perf/MeanReward', mean_reward, iteration)
        writer.add_scalar('Perf/MeanLength', mean_length, iteration)
        writer.add_scalar('Perf/FPS', fps, iteration)
        
        # Console
        if iteration % args.log_interval == 0:
            print(f"\nIter {iteration}/{args.num_iterations}")
            print(f"  Reward: {mean_reward:.2f} | Length: {mean_length:.1f}")
            print(f"  Loss: {total_loss/max(num_updates,1):.4f} (P:{policy_loss_sum/max(num_updates,1):.4f} "
                  f"V:{value_loss_sum/max(num_updates,1):.4f} E:{entropy_sum/max(num_updates,1):.4f})")
            print(f"  FPS: {fps:.0f} | Time: {iter_time:.1f}s")
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(log_dir, f'model_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': planner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_reward': mean_reward,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
        
        # Save best
        if mean_reward > best_reward and len(episode_rewards) >= 10:
            best_reward = mean_reward
            best_path = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': planner.state_dict(),
                'mean_reward': mean_reward,
            }, best_path)
            print(f"  ★ New best: {mean_reward:.2f}")
    
    # 训练结束
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Output: {log_dir}")
    print(f"{'='*60}")
    
    writer.close()
    return planner

    

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3.6 High-Level Planner Training")
    
    # 模式
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['teacher', 'student'],
                        help='训练模式: teacher (GT) 或 student (Vision)')
    
    # 环境
    parser.add_argument('--task', type=str, default='hex_terrain',
                        help='Isaac Gym 任务名称')
    parser.add_argument('--num_envs', type=int, default=4096,
                        help='并行环境数量')
    parser.add_argument('--decimation', type=int, default=5,
                        help='高层/低层频率比 (50Hz / 10Hz = 5)')
    
    # Checkpoints
    parser.add_argument('--low_level_ckpt', type=str, required=True,
                        help='底层控制器路径')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='(Student) Teacher 模型路径')
    parser.add_argument('--vision_ckpt', type=str, default=None,
                        help='(Student) Vision 模型路径')
    
    # 训练超参数
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='训练迭代次数')
    parser.add_argument('--num_steps', type=int, default=24,
                        help='每次迭代的步数')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='PPO epoch 数')
    parser.add_argument('--mini_batch_size', type=int, default=4096,
                        help='Mini-batch 大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss 系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy 系数')
    parser.add_argument('--distill_coef', type=float, default=1.0,
                        help='蒸馏 loss 系数 (Student 模式)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='梯度裁剪')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='outputs/planner',
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='保存间隔')
    
    args, unknown = parser.parse_known_args()
    
    # 传递未知参数给 Isaac Gym
    sys.argv = [sys.argv[0]] + unknown
    
    train(args)
