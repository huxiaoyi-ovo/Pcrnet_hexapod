"""
scripts/train_highlevel.py - V5 Command-Space MoE 训练脚本

核心变化:
1) Experts 直接输出 cmd_vel, Gate 输出 y (MoE 仲裁)
2) Command Post-Processor 统一命令限幅/滤波/风险钳制
3) Teacher/Student 仅用于 affordance 蒸馏

用法:
    Follow / Avoid:
        python scripts/train_highlevel.py --mode teacher --skill follow --task hex_terrain \\
            --low_level_ckpt agents/fast_2000.pt

    Gate:
        python scripts/train_highlevel.py --mode teacher --skill moe --task hex_terrain \\
            --low_level_ckpt agents/fast_2000.pt \\
            --follow_ckpt outputs/planner/follow/best_model.pt \\
            --avoid_ckpt outputs/planner/avoid/best_model.pt
"""

import os
import sys
import time
import argparse
import math
import types
import isaacgym  # noqa: F401  # ensure isaacgym is imported before torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Optional, Any
from collections import deque

# Use CLI flag to gate debug output before args are parsed.
DEBUG_MODE = "--debug" in sys.argv


def _debug_print(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 默认维度定义（实际训练时以环境观测维度为准）
STATE_DIM = 9   # [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
GOAL_DIM = 2    # [goal_x, goal_y] (相对坐标)
AFFORDANCE_CHANNELS = 3  # [occupancy, passable_gap, low_obstacle]

# 延迟导入占位（便于静态检查）
task_registry: Any = None
TerrainAdaptivePlanner: Any = None
HighLevelPlanner: Any = None
LocomotionAdapter: Any = None
CmdVelExpert: Any = None
GatePolicy: Any = None
CommandPostProcessor: Any = None
AffordanceEstimator: Any = None
NavigationRewardFunction: Any = None
NavigationRewardConfig: Any = None
ActorCritic: Any = None


def difficulty_from_gap(aff_map: torch.Tensor) -> torch.Tensor:
    if aff_map.ndim != 4 or aff_map.size(1) < 2:
        return torch.zeros(aff_map.shape[0], device=aff_map.device)
    gap = aff_map[:, 1]
    difficulty = 1.0 - gap.mean(dim=(1, 2))
    return torch.clamp(difficulty, 0.0, 1.0)


def apply_goal_occlusion(
    goal: torch.Tensor,
    prev_goal: Optional[torch.Tensor],
    occlude_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prev_goal is None:
        prev_goal = goal.detach()
    occlude_mask = occlude_mask.view(-1, 1)
    masked_goal = torch.where(occlude_mask, prev_goal, goal)
    return masked_goal, masked_goal.detach()


def import_modules():
    """延迟导入，带容错处理"""
    global task_registry, TerrainAdaptivePlanner, HighLevelPlanner, LocomotionAdapter
    global AffordanceEstimator, NavigationRewardFunction, NavigationRewardConfig
    global ActorCritic
    
    try:
        # 导入 legged_gym 注册环境
        import legged_gym.envs
        from legged_gym.utils import get_args, task_registry as tr
        task_registry = tr
        _debug_print("[Init] ✓ legged_gym 导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 'legged_gym': {e}")
        print("请确认 legged_gym 已安装并添加到 PYTHONPATH。")
        sys.exit(1)

    try:
        from rsl_rl.algorithms.high_level_planner import TerrainAdaptivePlanner as TAP
        from rsl_rl.algorithms.high_level_planner import LocomotionAdapter as LA
        from rsl_rl.algorithms.high_level_planner import CmdVelExpert as CVE
        from rsl_rl.algorithms.high_level_planner import GatePolicy as GP
        from rsl_rl.algorithms.high_level_planner import CommandPostProcessor as CPP
        from rsl_rl.modules import ActorCritic as AC
        
        # 赋值给全局变量
        # Keep a clearer name for ground tasks; same implementation.
        globals()['TerrainAdaptivePlanner'] = TAP
        globals()['HighLevelPlanner'] = TAP
        globals()['LocomotionAdapter'] = LA
        globals()['CmdVelExpert'] = CVE
        globals()['GatePolicy'] = GP
        globals()['CommandPostProcessor'] = CPP
        globals()['ActorCritic'] = AC
        _debug_print("[Init] ✓ rsl_rl 模块导入成功")
    except ImportError as e:
        print(f"\n[Error] 无法导入 rsl_rl 模块: {e}")
        sys.exit(1)

    try:
        from legged_gym.envs.hex_v4.affordance_estimator import AffordanceEstimator as AE
        globals()['AffordanceEstimator'] = AE
        _debug_print("[Init] ✓ AffordanceEstimator 导入成功")
    except ImportError as e:
        print(f"[Warning] AffordanceEstimator 导入失败: {e}")
        globals()['AffordanceEstimator'] = None

    try:
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardFunction as NRF
        from legged_gym.envs.hex_v4.navigation_env import NavigationRewardConfig as NRC
        globals()['NavigationRewardFunction'] = NRF
        globals()['NavigationRewardConfig'] = NRC
        _debug_print("[Init] ✓ NavigationReward 导入成功")
    except ImportError as e:
        print(f"[Warning] NavigationReward 导入失败: {e}")
        globals()['NavigationRewardFunction'] = None
        globals()['NavigationRewardConfig'] = None


class RolloutBuffer:
    """
    PPO Rollout Buffer - 存储一个 rollout 周期内的所有数据
    """
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        state_dim: int,
        goal_dim: int,
        aff_map_shape: Tuple[int, int, int],
        action_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.step = 0
        self.action_dim = action_dim

        # 观测（仅保留 PPO 更新所需张量）
        self.states = torch.zeros(num_steps, num_envs, state_dim, device=device)
        self.goals = torch.zeros(num_steps, num_envs, goal_dim, device=device)
        # aff_map_shape 为堆叠后的通道数 (aff_channels, H, W)
        self.aff_maps = torch.zeros(num_steps, num_envs, *aff_map_shape, device=device)
        self.difficulties = torch.zeros(num_steps, num_envs, device=device)

        # 动作
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        
        # PPO 核心数据
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        
        # 蒸馏目标 (Student 模式)
        self.teacher_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)

    def add(
        self,
        state,
        goal,
        aff_map,
        difficulty,
        action,
        log_prob,
        value,
        reward,
        done,
        teacher_action=None,
    ):
        """添加一步数据"""
        self.states[self.step] = state
        self.goals[self.step] = goal
        self.aff_maps[self.step] = aff_map
        self.difficulties[self.step] = difficulty
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value.squeeze(-1)
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        
        if teacher_action is not None:
            self.teacher_actions[self.step] = teacher_action
        
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
        self.states.zero_()
        self.goals.zero_()
        self.aff_maps.zero_()
        self.difficulties.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.teacher_actions.zero_()


# V5 核心环境包装器 (The Hierarchical Environment Wrapper)
class HierarchicalHexapodEnv:
    """
    V5 分层环境包装器
    
    职责:
    1. 托管 Isaac Gym 底层环境 (hex_terrain)
    2. 托管 Low-Level Controller (冻结的底层策略)
    3. 托管 Command Post-Processor (限幅/滤波/风险钳制)
    4. 托管 Reward Function (V5 gate_smooth + risk_barrier)
    5. 提供 Teacher/Student 两种数据流
    
    状态空间 (V5):
        robot_state: (N, 9) = [pos_x, pos_y, yaw, vx, vy, omega, height, roll, pitch]
        goal: (N, 2) = [goal_x, goal_y]
        affordance: (N, 3, 16, 16) = [occupancy, passable_gap, low_obstacle]
        terrain_difficulty: (N,)
    """
    
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.mode = args.mode
        self.debug = bool(getattr(args, "debug", False))

        from legged_gym.utils import get_args as get_isaac_args
        isaac_args = get_isaac_args()
        for key, value in vars(isaac_args).items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        
        # 初始化 Isaac Gym 环境
        if self.debug:
            print(f"[Env] 创建 Isaac Gym 环境: {args.task}")
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        if getattr(args, "skill", "follow") == "follow" and hasattr(env_cfg, "navigation"):
            if hasattr(env_cfg.navigation, "follow_goal_force_blocking_line"):
                env_cfg.navigation.goal_force_blocking_line = bool(
                    getattr(env_cfg.navigation, "follow_goal_force_blocking_line")
                )
            else:
                env_cfg.navigation.goal_force_blocking_line = False
            if hasattr(env_cfg.navigation, "follow_goal_force_blocking_prob"):
                env_cfg.navigation.goal_force_blocking_prob = float(
                    getattr(env_cfg.navigation, "follow_goal_force_blocking_prob")
                )
            else:
                env_cfg.navigation.goal_force_blocking_prob = 0.0

        if getattr(args, "camera_enable", False) and hasattr(env_cfg, "sensor"):
            if hasattr(env_cfg.sensor, "depth_camera"):
                env_cfg.sensor.depth_camera.enable = True
                if getattr(args, "camera_interval", None) is not None:
                    env_cfg.sensor.depth_camera.capture_interval = args.camera_interval

        if args.task == "hex_ground" and hasattr(env_cfg, "navigation"):
            env_cfg.navigation.goal_reached_threshold = 0.1

        # 覆盖配置以适配高层训练
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, args.num_envs)
        
        self.env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        self.num_envs = self.env.num_envs
        self.max_episode_length = self.env.max_episode_length
        if hasattr(self.env, "debug_viz"):
            self.env.debug_viz = bool(getattr(args, "debug", False))

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
        self.affordance_cell_size = float(cell_size) if cell_size is not None else (map_extent / map_size)
        self.affordance_map_size = map_size
        self.affordance_map_extent = map_extent
        clearance = getattr(env_cfg.terrain, "fixed_layout_robot_clearance", None)
        if clearance is None:
            body_shape = getattr(getattr(env_cfg, "asset", None), "body_shape", None)
            if body_shape is not None:
                clearance = math.hypot(float(body_shape.x), float(body_shape.y)) + 0.05
            else:
                clearance = 0.27
        self.affordance_clearance = float(clearance)
        self.affordance_clearance_free = self.affordance_clearance + 0.3
        self.affordance_dist_map = self._build_affordance_dist_map()
        crossable_height = getattr(env_cfg.navigation, "crossable_height_max", None)
        if crossable_height is None:
            crossable_height = getattr(env_cfg.navigation, "goal_obstacle_height_threshold", 0.2)
        self.affordance_blocking_height = float(crossable_height)
        self.affordance_crossable_height = float(crossable_height)
        body_shape = getattr(getattr(env_cfg, "asset", None), "body_shape", None)
        if body_shape is not None:
            self.body_width = 2.0 * float(body_shape.y)
        else:
            self.body_width = 0.44
        width_margin = float(getattr(env_cfg.navigation, "crossable_width_margin", 0.0))
        self.crossable_width = self.body_width + width_margin
        self.crossable_sector_deg = float(getattr(env_cfg.navigation, "crossable_sector_deg", 60.0))
        self.camera_fov_rad = None
        self.camera_bearing_rad = 0.0
        self.camera_far = self.affordance_map_extent
        if cam_cfg is not None:
            self.camera_fov_rad = math.radians(float(getattr(cam_cfg, "horizontal_fov", 0.0)))
            self.camera_far = float(getattr(cam_cfg, "far_clip", self.affordance_map_extent))
            yaw_deg = float(getattr(cam_cfg, "yaw_deg", 0.0))
            # Convert camera yaw to bearing_y convention (0 means +Y forward).
            self.camera_bearing_rad = math.radians(yaw_deg) - 0.5 * math.pi
        (self.affordance_x_map,
         self.affordance_y_map,
         self.affordance_bearing_map,
         self.affordance_visible_mask) = self._build_affordance_geometry()

        if hasattr(self.env, "_resample_commands"):
            def _no_resample(self, env_ids):
                return
            self.env._resample_commands = types.MethodType(_no_resample, self.env)
        
        # 加载 Low-Level Controller
        self._load_low_level_policy(args.low_level_ckpt)
        
        # 初始化 Command Post-Processor (V5)
        max_lin_cmd = float(getattr(nav_cfg, "max_lin_vel_command", 0.8))
        max_ang_cmd = float(getattr(nav_cfg, "max_ang_vel_command", 1.5))
        slew_lin = float(getattr(args, "cmd_slew_lin", 0.2))
        slew_ang = float(getattr(args, "cmd_slew_ang", 0.4))
        safe_dist_arg = getattr(args, "cmd_safe_dist", None)
        free_dist_arg = getattr(args, "cmd_free_dist", None)
        safe_dist = self.affordance_clearance if safe_dist_arg is None else float(safe_dist_arg)
        free_dist = self.affordance_clearance_free if free_dist_arg is None else float(free_dist_arg)
        self.post_processor = CommandPostProcessor(
            max_cmd=(max_lin_cmd, max_lin_cmd, max_ang_cmd),
            max_delta=(slew_lin, slew_lin, slew_ang),
            safe_distance=safe_dist,
            free_distance=free_dist,
            enable_risk_scale=not bool(getattr(args, "disable_risk_scale", False)),
        )
        self.post_processor.reset(self.num_envs, device)
        
        # 初始化 Reward Function
        if NavigationRewardConfig is not None:
            reward_defaults = NavigationRewardConfig()
            reward_kwargs = {k: getattr(reward_defaults, k) for k in reward_defaults.__dict__.keys()}
            if nav_cfg is not None and hasattr(nav_cfg, "reward_cfg"):
                for key, value in dict(getattr(nav_cfg, "reward_cfg")).items():
                    if value is not None:
                        reward_kwargs[key] = value
            if reward_kwargs.get("risk_barrier_safe") is None:
                reward_kwargs["risk_barrier_safe"] = self.affordance_clearance
            if reward_kwargs.get("risk_barrier_free") is None:
                reward_kwargs["risk_barrier_free"] = self.affordance_clearance_free
            self.reward_cfg = NavigationRewardConfig(**reward_kwargs)
            self.reward_func = NavigationRewardFunction(self.reward_cfg)
        else:
            self.reward_func = None
            print("[Warning] 使用环境原生奖励")
        
        # 状态缓冲区
        self.prev_robot_pos = torch.zeros(self.num_envs, 3, device=device)
        self.prev_gate_y = torch.zeros(self.num_envs, device=device)
        self.reach_given = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.goal_change_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.prev_goal_world = None
        self.episode_length_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.episode_return_buf = torch.zeros(self.num_envs, device=device)
        self.episode_len_buf = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self.clearance_override = None
        self.reward_affordance_override = None
        self.last_obs = None
        
        # 频率控制 (High-Level 10Hz, Low-Level 50Hz)
        self.decimation = getattr(args, 'decimation', 5)
        
        if self.debug:
            print(f"[Env] 初始化完成: {self.num_envs} envs, decimation={self.decimation}")

    def _load_low_level_policy(self, ckpt_path: str):
        """加载并冻结底层控制器"""
        if self.debug:
            print(f"[Env] 加载底层策略: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"底层策略不存在: {ckpt_path}")
        
        # 加载 checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # 创建 ActorCritic (需要匹配底层网络结构)
        num_obs = self.env.num_obs
        num_actions = self.env.num_actions
        num_priv_obs = getattr(self.env, 'num_privileged_obs', num_obs)

        def _infer_hidden_dims(state_dict, prefix: str) -> Optional[list]:
            dims = []
            idx = 0
            while True:
                weight_key = f"{prefix}.{idx}.weight"
                if weight_key not in state_dict:
                    break
                out_dim = state_dict[weight_key].shape[0]
                dims.append(out_dim)
                idx += 2
            if len(dims) <= 1:
                return None
            return dims[:-1]

        state_dict = None
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'actor_state_dict' in ckpt:
                state_dict = ckpt['actor_state_dict']
        if state_dict is None and isinstance(ckpt, dict):
            state_dict = ckpt

        actor_hidden_dims = _infer_hidden_dims(state_dict, "actor") if state_dict else None
        critic_hidden_dims = _infer_hidden_dims(state_dict, "critic") if state_dict else None
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
            print("[Warning] 未能推断 actor hidden dims，使用默认 [256, 256, 256]")
        if critic_hidden_dims is None:
            critic_hidden_dims = actor_hidden_dims

        self.low_level_policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_priv_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
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
        
        if self.debug:
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
        self.prev_gate_y.zero_()
        self.reach_given.zero_()
        self.goal_change_count.zero_()
        self.reward_affordance_override = None
        self.post_processor.reset(self.num_envs, self.device)

        self._refresh_depth_images(force=True)
        obs_dict = self._get_high_level_obs()
        self.prev_robot_pos = self.env.root_states[:, :3].clone()
        if hasattr(self.env, "goal_world"):
            self.prev_goal_world = self.env.goal_world.clone()
        
        self.last_obs = obs_dict
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
        if getattr(self.env.cfg.terrain, "scene_use_actors", False):
            scene_aff = self._compute_gt_affordance_from_scene()
            if scene_aff is not None:
                return scene_aff
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
        low_mask = (heights > 1e-6) & (heights < self.affordance_crossable_height)
        low_obstacle = low_mask.float()

        radius_cells = int(math.ceil(self.affordance_clearance / cell))
        if radius_cells > 0:
            kernel = 2 * radius_cells + 1
            pooled = F.max_pool2d(occ_block, kernel_size=kernel, stride=1, padding=radius_cells)
            passable = (pooled <= 0.5) & (occ_block < 0.5)
        else:
            passable = occ_block < 0.5
        passable_gap = passable.float().squeeze(1)

        return torch.cat([
            occ_all.unsqueeze(1),
            passable_gap.unsqueeze(1),
            low_obstacle.unsqueeze(1),
        ], dim=1)

    def _compute_gt_affordance_from_scene(self) -> Optional[torch.Tensor]:
        if not hasattr(self.env, "scene_spec_cache"):
            return None
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = map_extent / map_size
        if not math.isfinite(cell) or cell <= 0.0:
            if not getattr(self, "_scene_aff_cell_warned", False):
                print(f"[Warn] scene affordance cell invalid: cell={cell}, extent={map_extent}, size={map_size}")
                self._scene_aff_cell_warned = True
            return None
        occ_all = torch.zeros(self.num_envs, map_size, map_size, device=self.device)

        robot_xy = self.env.root_states[:, :2]
        if hasattr(self.env, "robot_state_buf"):
            yaw = self.env.robot_state_buf[:, 2]
        else:
            yaw = self._quat_to_yaw(self.env.root_states[:, 3:7])
        if not torch.isfinite(yaw).all():
            yaw = yaw.clone()
            yaw[~torch.isfinite(yaw)] = 0.0
        if not torch.isfinite(robot_xy).all():
            robot_xy = robot_xy.clone()
            robot_xy[~torch.isfinite(robot_xy)] = 0.0
        cos_h = torch.cos(yaw)
        sin_h = torch.sin(yaw)

        x_min = -0.5 * map_extent
        y_min = 0.0

        def rasterize(env_id: int, center_x: float, center_y: float, size_x: float, size_y: float) -> None:
            if not (
                math.isfinite(center_x)
                and math.isfinite(center_y)
                and math.isfinite(size_x)
                and math.isfinite(size_y)
            ):
                if not getattr(self, "_scene_aff_nan_warned", False):
                    print(
                        "[Warn] scene affordance rasterize got NaN/inf "
                        f"(env={env_id}, center=({center_x},{center_y}), size=({size_x},{size_y}))"
                    )
                    self._scene_aff_nan_warned = True
                return
            if size_x <= 0.0 or size_y <= 0.0:
                return
            dx = center_x - float(robot_xy[env_id, 0].item())
            dy = center_y - float(robot_xy[env_id, 1].item())
            x_body = float(cos_h[env_id].item()) * dx + float(sin_h[env_id].item()) * dy
            y_body = -float(sin_h[env_id].item()) * dx + float(cos_h[env_id].item()) * dy
            x0 = x_body - 0.5 * size_x
            x1 = x_body + 0.5 * size_x
            y0 = y_body - 0.5 * size_y
            y1 = y_body + 0.5 * size_y
            ix0 = int(math.floor((x0 - x_min) / cell))
            ix1 = int(math.ceil((x1 - x_min) / cell))
            iy0 = int(math.floor((y0 - y_min) / cell))
            iy1 = int(math.ceil((y1 - y_min) / cell))
            ix0 = max(0, min(map_size - 1, ix0))
            ix1 = max(0, min(map_size - 1, ix1))
            iy0 = max(0, min(map_size - 1, iy0))
            iy1 = max(0, min(map_size - 1, iy1))
            occ_all[env_id, ix0:ix1 + 1, iy0:iy1 + 1] = 1.0

        for env_id in range(self.num_envs):
            scene_spec = self.env.scene_spec_cache[env_id]
            if scene_spec is None:
                continue
            origin = self.env.env_origins[env_id]
            for spec in scene_spec.static_obstacles:
                center_x = float(origin[0].item() + spec.position[0])
                center_y = float(origin[1].item() + spec.position[1])
                rasterize(env_id, center_x, center_y, spec.size[0], spec.size[1])

        if hasattr(self.env, "dynamic_active") and self.env.dynamic_active is not None:
            dyn_size = float(getattr(self.env.cfg.terrain, "scene_dynamic_size", 0.4))
            for env_id in range(self.num_envs):
                if self.env.dynamic_active[env_id].numel() == 0:
                    continue
                origin = self.env.env_origins[env_id]
                t = float(self.env.scene_dyn_time[env_id].item())
                for obs_id, active in enumerate(self.env.dynamic_active[env_id]):
                    if not bool(active.item()):
                        continue
                    period = float(self.env.dynamic_period[env_id, obs_id].item())
                    phase = float(self.env.dynamic_phase[env_id, obs_id].item())
                    path_len = float(self.env.dynamic_path_len[env_id, obs_id].item())
                    half = 0.5 * period if period > 1e-6 else 0.0
                    tau = (t + phase) % period if period > 1e-6 else 0.0
                    progress = tau / half if half > 1e-6 else 0.0
                    if tau > half and half > 1e-6:
                        progress = (period - tau) / half
                    progress = max(0.0, min(1.0, progress))
                    dist = progress * path_len
                    start = self.env.dynamic_start[env_id, obs_id]
                    direction = self.env.dynamic_dir[env_id, obs_id]
                    pos_local = start + direction * dist
                    center_x = float(origin[0].item() + pos_local[0].item())
                    center_y = float(origin[1].item() + pos_local[1].item())
                    rasterize(env_id, center_x, center_y, dyn_size, dyn_size)

        occ_block = occ_all.unsqueeze(1)
        low_obstacle = torch.zeros_like(occ_all)
        radius_cells = int(math.ceil(self.affordance_clearance / cell))
        if radius_cells > 0:
            kernel = 2 * radius_cells + 1
            pooled = F.max_pool2d(occ_block, kernel_size=kernel, stride=1, padding=radius_cells)
            passable = (pooled <= 0.5) & (occ_block < 0.5)
        else:
            passable = occ_block < 0.5
        passable_gap = passable.float().squeeze(1)

        if (
            hasattr(self.env, "nav_cfg")
            and self.env.nav_cfg is not None
            and bool(getattr(self.env.nav_cfg, "affordance_debug", False))
            and self.debug
        ):
            if not getattr(self, "_aff_debug_logged", False):
                mid = map_size // 2
                occ_center = float(occ_all[0, mid, mid].item())
                occ_front = float(occ_all[0, mid, min(map_size - 1, mid + 2)].item())
                occ_right = float(occ_all[0, min(map_size - 1, mid + 2), mid].item())
                print(
                    "[AffDebug] occ(center/front/right)="
                    f"({occ_center:.1f}/{occ_front:.1f}/{occ_right:.1f}) "
                    f"map_extent={map_extent:.2f} cell={cell:.3f}"
                )
                self._aff_debug_logged = True

        return torch.cat([
            occ_all.unsqueeze(1),
            passable_gap.unsqueeze(1),
            low_obstacle.unsqueeze(1),
        ], dim=1)

    def _build_affordance_dist_map(self) -> torch.Tensor:
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = self.affordance_cell_size
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
        return torch.sqrt(grid_x ** 2 + grid_y ** 2)

    def _build_affordance_geometry(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        map_size = self.affordance_map_size
        map_extent = self.affordance_map_extent
        cell = self.affordance_cell_size
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
        bearing_y = torch.atan2(grid_x, grid_y)
        dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        visible = torch.ones_like(dist, dtype=torch.bool)
        if self.camera_fov_rad is not None and self.camera_fov_rad > 0.0:
            angle = torch.atan2(
                torch.sin(bearing_y - self.camera_bearing_rad),
                torch.cos(bearing_y - self.camera_bearing_rad),
            )
            visible = visible & (torch.abs(angle) <= 0.5 * self.camera_fov_rad)
        if self.camera_far is not None:
            visible = visible & (dist <= float(self.camera_far) + 1e-6)
        return grid_x, grid_y, bearing_y, visible

    def _compute_clearance_from_affordance(self, aff_map: torch.Tensor) -> torch.Tensor:
        occ = aff_map[:, 0] > 0.5
        dist_map = self.affordance_dist_map
        if dist_map.device != aff_map.device:
            dist_map = dist_map.to(aff_map.device)
        dist = torch.where(occ, dist_map, torch.full_like(dist_map, self.affordance_map_extent))
        min_dist = dist.amin(dim=(1, 2))
        no_obs = ~occ.flatten(1).any(dim=1)
        if no_obs.any():
            min_dist = torch.where(no_obs, torch.full_like(min_dist, self.affordance_map_extent), min_dist)
        return min_dist

    def _compute_passable_guidance(
        self,
        aff_map: torch.Tensor,
        goal_local: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if aff_map.ndim != 4 or aff_map.size(1) < 2:
            zeros = torch.zeros(aff_map.shape[0], device=aff_map.device)
            return torch.zeros(aff_map.shape[0], 2, device=aff_map.device), zeros, zeros

        occ = aff_map[:, 0]
        passable = aff_map[:, 1]

        visible = self.affordance_visible_mask
        if visible is None:
            visible = torch.ones_like(passable[0], dtype=torch.bool)
        if visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        bearing_map = self.affordance_bearing_map
        if x_map.device != aff_map.device:
            x_map = x_map.to(aff_map.device)
            y_map = y_map.to(aff_map.device)
            bearing_map = bearing_map.to(aff_map.device)

        visible_f = visible.float()
        passable_vis = passable * visible_f
        if block_mask is not None:
            passable_vis = passable_vis * (1.0 - block_mask)
        dir_x = (passable_vis * x_map).sum(dim=(1, 2))
        dir_y = (passable_vis * y_map).sum(dim=(1, 2))
        pass_dir = torch.stack([dir_x, dir_y], dim=1)
        pass_norm = torch.norm(pass_dir, dim=-1, keepdim=True)
        goal_norm = torch.norm(goal_local, dim=-1, keepdim=True)
        goal_dir = goal_local / (goal_norm + 1e-6)
        pass_dir = torch.where(pass_norm > 1e-6, pass_dir / (pass_norm + 1e-6), goal_dir)

        sector_deg = 0.0
        if self.reward_cfg is not None:
            sector_deg = float(getattr(self.reward_cfg, "passable_sector_deg", 0.0))
        sector_half = math.radians(sector_deg) * 0.5 if sector_deg > 0.0 else 0.0
        if sector_half > 0.0:
            goal_bearing = torch.atan2(goal_local[:, 0], goal_local[:, 1])
            angle = torch.atan2(
                torch.sin(bearing_map.unsqueeze(0) - goal_bearing.view(-1, 1, 1)),
                torch.cos(bearing_map.unsqueeze(0) - goal_bearing.view(-1, 1, 1)),
            )
            sector_mask = torch.abs(angle) <= sector_half
            sector_mask = sector_mask & visible.unsqueeze(0)
        else:
            sector_mask = visible.unsqueeze(0)

        sector_f = sector_mask.float()
        occ_ratio = (occ * sector_f).sum(dim=(1, 2)) / (sector_f.sum(dim=(1, 2)) + 1e-6)
        occ_low = 0.0
        occ_high = 1.0
        if self.reward_cfg is not None:
            occ_low = float(getattr(self.reward_cfg, "passable_occ_ratio_low", 0.0))
            occ_high = float(getattr(self.reward_cfg, "passable_occ_ratio_high", 1.0))
        if occ_high > occ_low:
            gate = torch.clamp((occ_ratio - occ_low) / (occ_high - occ_low), 0.0, 1.0)
        else:
            gate = torch.zeros_like(occ_ratio)

        return pass_dir, gate, occ_ratio

    def _compute_low_obstacle_guidance(
        self,
        aff_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if aff_map.ndim != 4 or aff_map.size(1) < 3:
            zeros = torch.zeros(aff_map.shape[0], device=aff_map.device)
            return torch.zeros(aff_map.shape[0], 2, device=aff_map.device), zeros, zeros, None

        low_obs = aff_map[:, 2] > 0.5
        visible = self.affordance_visible_mask
        if visible is None:
            visible = torch.ones_like(low_obs[0], dtype=torch.bool)
        if visible.device != aff_map.device:
            visible = visible.to(aff_map.device)

        bearing_map = self.affordance_bearing_map
        x_map = self.affordance_x_map
        y_map = self.affordance_y_map
        if bearing_map.device != aff_map.device:
            bearing_map = bearing_map.to(aff_map.device)
            x_map = x_map.to(aff_map.device)
            y_map = y_map.to(aff_map.device)

        sector_half = math.radians(self.crossable_sector_deg) * 0.5 if self.crossable_sector_deg > 0.0 else 0.0
        if sector_half > 0.0:
            sector_mask = torch.abs(bearing_map) <= sector_half
            sector_mask = sector_mask & visible
        else:
            sector_mask = visible

        mask = low_obs & sector_mask
        valid = mask.flatten(1).any(dim=1)

        x_map_b = x_map.unsqueeze(0)
        y_map_b = y_map.unsqueeze(0)
        big = torch.full_like(x_map_b, 1e6)
        neg = torch.full_like(x_map_b, -1e6)

        x_min = torch.where(mask, x_map_b, big).amin(dim=(1, 2))
        x_max = torch.where(mask, x_map_b, neg).amax(dim=(1, 2))
        width = torch.where(valid, x_max - x_min, torch.zeros_like(x_min))

        count = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        center_x = torch.where(valid, (x_max + x_min) * 0.5, torch.zeros_like(x_min))
        center_y = torch.where(valid, (mask.float() * y_map_b).sum(dim=(1, 2)) / count, torch.zeros_like(x_min))
        center_dir = torch.stack([center_x, center_y], dim=1)
        center_norm = torch.norm(center_dir, dim=-1, keepdim=True)
        center_dir = torch.where(center_norm > 1e-6, center_dir / center_norm, torch.zeros_like(center_dir))

        width_gate = width < self.crossable_width
        gate = valid & width_gate
        block_mask = mask.float() * (~width_gate).float().view(-1, 1, 1)

        return center_dir, gate.float(), width, block_mask

    def _get_high_level_obs(self) -> Dict[str, torch.Tensor]:
        """
        构建高层观测字典
        
        Returns:
            Dict with keys:
            - state: (N, 9+1) robot state + prev_gate_y
            - goal: (N, 2) relative goal
            - gt_affordance: (N, 3, 16, 16) ground truth affordance
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

        # Align yaw to policy heading convention (+Y forward) via heading_offset_rad.
        offset = 0.0
        if self.reward_cfg is not None:
            offset = float(getattr(self.reward_cfg, "heading_offset_rad", 0.0))
            if offset != 0.0:
                state = obs_dict['state']
                yaw = state[:, 2] + offset
                yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
                state = state.clone()
                state[:, 2] = yaw
                obs_dict['state'] = state
        
        # 1.5 添加上一时刻 gate_y（解决门控滤波的非马尔可夫性）
        if hasattr(self, "prev_gate_y"):
            prev_gate_y = self.prev_gate_y.unsqueeze(1)
            obs_dict['state'] = torch.cat([obs_dict['state'], prev_gate_y], dim=1)

        # 2. Goal (相对坐标)
        if hasattr(self.env, 'goal_buf'):
            obs_dict['goal'] = self.env.goal_buf.clone()
        else:
            obs_dict['goal'] = self.env.commands[:, :2].clone()
        if offset != 0.0:
            goal = obs_dict['goal']
            cos_o = math.cos(offset)
            sin_o = math.sin(offset)
            goal_x = cos_o * goal[:, 0] - sin_o * goal[:, 1]
            goal_y = sin_o * goal[:, 0] + cos_o * goal[:, 1]
            obs_dict['goal'] = torch.stack([goal_x, goal_y], dim=1)
        
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
                    low = torch.zeros_like(h_map)
                    obs_dict['gt_affordance'] = torch.cat([h_map, 1 - torch.abs(h_map), low], dim=1)
                else:
                    obs_dict['gt_affordance'] = torch.zeros(self.num_envs, 3, 16, 16, device=self.device)
            else:
                obs_dict['gt_affordance'] = torch.zeros(self.num_envs, 3, 16, 16, device=self.device)
        obs_dict['gt_difficulty'] = difficulty_from_gap(obs_dict['gt_affordance'])
        
        # 4. Depth Image
        if hasattr(self.env, 'depth_images'):
            obs_dict['depth'] = self.env.depth_images.clone()
        elif hasattr(self.env, 'depth_buffer'):
            obs_dict['depth'] = self.env.depth_buffer.unsqueeze(1).clone()
        else:
            obs_dict['depth'] = torch.zeros(self.num_envs, 1, 128, 128, device=self.device)
        
        return obs_dict

    def step(
        self,
        cmd_vel: torch.Tensor,
        gate_y: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        高层步进 (10Hz)

        Args:
            cmd_vel: (N, 3) [vx, vy, omega] 高层命令
            gate_y: (N,) 门控权重（仅 Gate 训练使用）

        Returns:
            obs_dict, rewards, dones, info
        """
        # 1. Command Post-Processor
        clearance_pp = None
        if self.clearance_override is not None:
            clearance_pp = self.clearance_override
        elif self.last_obs is not None and 'gt_affordance' in self.last_obs:
            clearance_pp = self._compute_clearance_from_affordance(self.last_obs['gt_affordance'])
        velocity_cmd, post_info = self.post_processor.process(cmd_vel, clearance_pp)

        # 2. Low-Level 控制循环 (50Hz)
        accumulated_reward = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        active_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for _ in range(self.decimation):
            if not active_mask.any():
                break
            if hasattr(self.env, "commands"):
                velocity_cmd_step = velocity_cmd.detach()
                if (~active_mask).any():
                    velocity_cmd_step = velocity_cmd_step.clone()
                    velocity_cmd_step[~active_mask] = 0.0
                self.env.commands[:, :3] = velocity_cmd_step
                if hasattr(self.env, "commands_scale") and hasattr(self.env, "obs_buf"):
                    if self.env.obs_buf.shape[1] >= 3:
                        self.env.obs_buf[:, -3:] = velocity_cmd_step * self.env.commands_scale

            low_level_obs = self.env.obs_buf

            with torch.no_grad():
                actions = self.low_level_policy.act_inference(low_level_obs)
            if (~active_mask).any():
                actions = actions.clone()
                actions[~active_mask] = 0.0

            obs, _, rewards, dones, infos = self.env.step(actions)
            rewards = rewards * active_mask.float()
            accumulated_reward += rewards
            done_any |= dones
            active_mask &= ~dones
        
        self.episode_length_buf += 1
        length_snapshot = self.episode_length_buf.clone()

        done_during = done_any.clone()

        if hasattr(self.env, "goal_world"):
            current_goal_world = self.env.goal_world.clone()
            if self.prev_goal_world is None:
                self.prev_goal_world = current_goal_world.clone()
            goal_delta = torch.norm(current_goal_world - self.prev_goal_world, dim=1)
            goal_changed = goal_delta > 1e-3
            if done_during.any():
                goal_changed = goal_changed & (~done_during)
            self.goal_change_count += goal_changed.long()
            self.prev_goal_world = current_goal_world
        
        # 3. 计算高层奖励
        self._refresh_depth_images()
        reward_obs = self._get_high_level_obs()
        self.last_obs = reward_obs
        robot_pos = self.env.root_states[:, :3]
        
        collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision_force_max = torch.zeros(self.num_envs, device=self.device)
        collision_threshold = None
        collision_threshold_src = None
        collision_indices_src = None
        clearance = None
        reward_aff_map = reward_obs['gt_affordance'] if reward_obs is not None else None
        override_used = getattr(self, "reward_affordance_override", None) is not None
        if override_used:
            reward_aff_map = self.reward_affordance_override
            self.reward_affordance_override = None
        if hasattr(self.env, "contact_forces"):
            collision_threshold = getattr(self.env.cfg.terrain, "collision_force_threshold", 1.0)
            collision_threshold_src = "collision_force_threshold"
            indices = getattr(self.env, "penalised_contact_indices", None)
            collision_indices_src = "penalised_contact_indices"
            if indices is not None and indices.numel() > 0:
                contact_norm = torch.norm(self.env.contact_forces[:, indices, :], dim=-1)
                collision_force_max = contact_norm.max(dim=1).values
                collision_mask = torch.any(contact_norm > collision_threshold, dim=1)
        if self.clearance_override is not None:
            clearance = self.clearance_override
            self.clearance_override = None
        elif reward_aff_map is not None:
            clearance = self._compute_clearance_from_affordance(reward_aff_map)
        passable_dir = None
        passable_gate = None
        passable_occ_ratio = None
        crossable_dir = None
        crossable_gate = None
        crossable_width = None
        low_block_mask = None
        if (
            self.reward_cfg is not None
            and (
                getattr(self.reward_cfg, "passable_align_scale", 0.0) != 0.0
                or getattr(self.reward_cfg, "crossable_align_scale", 0.0) != 0.0
            )
            and reward_aff_map is not None
        ):
            crossable_dir, crossable_gate, crossable_width, low_block_mask = self._compute_low_obstacle_guidance(
                reward_aff_map
            )
            passable_dir, passable_gate, passable_occ_ratio = self._compute_passable_guidance(
                reward_aff_map,
                reward_obs['goal'],
                block_mask=low_block_mask,
            )

        reward_terms = None
        if gate_y is None:
            gate_y = torch.zeros(self.num_envs, device=self.device)
        gate_y = gate_y.to(self.device)
        if self.reward_func is not None:
            robot_vel = self.env.base_lin_vel
            robot_quat = self.env.root_states[:, 3:7]
            if hasattr(self.env, "goal_world"):
                goal_pos = self.env.goal_world.clone()
            else:
                yaw_policy = reward_obs['state'][:, 2]
                cos_h = torch.cos(yaw_policy)
                sin_h = torch.sin(yaw_policy)
                goal_local = reward_obs['goal']
                goal_x = robot_pos[:, 0] + cos_h * goal_local[:, 0] - sin_h * goal_local[:, 1]
                goal_y = robot_pos[:, 1] + sin_h * goal_local[:, 0] + cos_h * goal_local[:, 1]
                goal_pos = torch.stack([goal_x, goal_y], dim=1)
            reward_difficulty = reward_obs['gt_difficulty']
            if override_used and reward_aff_map is not None:
                reward_difficulty = difficulty_from_gap(reward_aff_map)
            
            reward_dict = self.reward_func.compute_reward(
                robot_pos=robot_pos,
                prev_robot_pos=self.prev_robot_pos,
                goal_pos=goal_pos,
                robot_vel=robot_vel,
                robot_quat=robot_quat,
                gate_y=gate_y,
                prev_gate_y=self.prev_gate_y,
                terrain_difficulty=reward_difficulty,
                collision_mask=collision_mask,
                clearance=clearance,
                cmd_xy=velocity_cmd[:, :2],
                passable_dir=passable_dir,
                passable_gate=passable_gate,
                crossable_dir=crossable_dir,
                crossable_gate=crossable_gate,
            )
            if clearance is not None:
                reward_dict['clearance'] = clearance
            if passable_occ_ratio is not None:
                reward_dict['passable_occ_ratio'] = passable_occ_ratio
            if crossable_width is not None:
                reward_dict['crossable_width'] = crossable_width
            total_reward = reward_dict['total']

            # reach_bonus 只在每个 episode 内生效一次
            if self.reward_cfg is not None:
                dist_to_goal = torch.norm(robot_pos[:, :2] - goal_pos, dim=-1)
                reach_now = dist_to_goal < self.reward_cfg.goal_reach_threshold
                if reach_now.any():
                    done_any |= reach_now
                reach_mask = reach_now & (~self.reach_given)
                if reach_now.any():
                    repeated = reach_now & self.reach_given
                    total_reward = total_reward - repeated.float() * self.reward_cfg.goal_reach_bonus
                    reward_dict['reach'] = reach_mask.float() * self.reward_cfg.goal_reach_bonus
                    reward_dict['total'] = total_reward
                self.reach_given |= reach_mask
            reward_terms = reward_dict
        else:
            total_reward = accumulated_reward / self.decimation
            reward_terms = {"total": total_reward}
        
        # 4. 更新缓冲区
        self.prev_robot_pos = robot_pos.clone()
        self.prev_gate_y = gate_y.clone()
        
        # done 的环境避免跨 episode 的 shaped reward 污染
        if done_during.any():
            safe_reward = accumulated_reward / self.decimation
            total_reward = torch.where(done_during, safe_reward, total_reward)
            if reward_terms is not None:
                reward_terms["total"] = total_reward

        # 4.5 高层 episode 统计（与 breakdown 口径对齐）
        self.episode_return_buf += total_reward
        self.episode_len_buf += 1

        # 5. 处理超时
        timeout = self.episode_length_buf >= self.max_episode_length
        done_any |= timeout
        
        episode_info = None
        if done_any.any():
            episode_info = {
                'r': self.episode_return_buf.clone(),
                'l': self.episode_len_buf.clone(),
            }
            self.episode_length_buf[done_any] = 0
            self.prev_gate_y[done_any] = 0
            self.reach_given[done_any] = False
            self.goal_change_count[done_any] = 0
            self.episode_return_buf[done_any] = 0.0
            self.episode_len_buf[done_any] = 0
            if self.prev_goal_world is not None and hasattr(self.env, "goal_world"):
                self.prev_goal_world[done_any] = self.env.goal_world[done_any]
            # 重置 Post-Processor 的变化率限制记忆
            self.post_processor.last_cmd[done_any] = 0.0

        next_obs = self._get_high_level_obs()

        info = {
            'post_info': post_info,
            'episode_length': length_snapshot,
            'reward_terms': reward_terms,
            'gate_y': gate_y,
            'goal_change_count': self.goal_change_count.clone(),
            'episode': episode_info,
            'collision_mask': collision_mask,
            'collision_force_max': collision_force_max,
            'collision_threshold': collision_threshold,
            'collision_threshold_src': collision_threshold_src,
            'collision_indices_src': collision_indices_src,
            'clearance': clearance,
        }

        return next_obs, total_reward, done_any, info

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
    """主训练函数 (V5 Command-Space MoE)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"V5 Command-Space MoE Training")
    print(f"Mode: {args.mode.upper()} | Skill: {getattr(args, 'skill', 'follow')}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    debug = bool(getattr(args, "debug", False))
    def dprint(*vals, **kwargs):
        if debug:
            print(*vals, **kwargs)
    if args.task != "hex_ground":
        print(f"[Warn] V5 主线默认任务为 hex_ground，当前 task={args.task}（hex_terrain 视为 legacy）。")
    gate_use_difficulty = bool(getattr(args, "gate_use_difficulty", False))
    moe_use_student_aff = bool(getattr(args, "moe_use_student_aff", False))
    if getattr(args, "skill", "follow") == "moe":
        if args.num_steps < 48:
            print(f"[Warn] Gate 训练建议 num_steps>=48，当前 num_steps={args.num_steps}。")
        dprint(f"[Info] Gate difficulty 输入: {'ON' if gate_use_difficulty else 'OFF'}")
        dprint(f"[Info] MoE affordance 来源: {'student' if moe_use_student_aff else args.mode}")
    if getattr(args, "disable_risk_scale", False):
        dprint("[Info] CommandPostProcessor 风险缩放已禁用（消融用）。")
    if getattr(args, "aff_stack", 1) > 1:
        print(f"[Warn] aff_stack={args.aff_stack}: 输入通道数改变，必须使用相同 aff_stack 训练/加载 ckpt；旧 ckpt 不兼容。")
    if args.mode == "student" and not getattr(args, "vision_ckpt", None):
        raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
    if args.mode == "student":
        args.camera_enable = True
    
    # 导入模块
    import_modules()
    
    # 创建环境
    env = HierarchicalHexapodEnv(args, device)
    dprint(f"[Main] 环境初始化完成: {env.num_envs} envs")
    
    # 初始 reset（用于确定观测维度）
    obs_dict = env.reset()
    scene_meta = None
    if hasattr(env, "env") and hasattr(env.env, "extras"):
        scene_meta = env.env.extras.get("scene_meta")
    if scene_meta is not None and debug:
        print(f"[Debug] scene_meta: {scene_meta}")
    if debug:
        try:
            if hasattr(env, "env"):
                spacing = getattr(getattr(env.env, "cfg", None), "env", None)
                if spacing is not None:
                    env_spacing = float(getattr(spacing, "env_spacing", 0.0))
                else:
                    env_spacing = 0.0
                if hasattr(env.env, "env_origins"):
                    origins = env.env.env_origins[:3].detach().cpu().tolist()
                else:
                    origins = []
                print(f"[Debug] envs={env.num_envs} env_spacing={env_spacing:.3f} env_origins[:3]={origins}")
            env0 = 0
            base_pos = None
            if hasattr(env, "env") and hasattr(env.env, "root_states"):
                base_pos = env.env.root_states[env0, :3].detach().cpu()
            if base_pos is not None:
                print(f"[Debug] env0 base_pos: ({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})")
            if hasattr(env, "env") and hasattr(env.env, "env_origins"):
                origin = env.env.env_origins[env0]
                ox, oy, oz = float(origin[0].item()), float(origin[1].item()), float(origin[2].item())
                print(f"[Debug] env0 origin: ({ox:.3f}, {oy:.3f}, {oz:.3f})")
            else:
                ox = oy = oz = 0.0
            # SceneSpec positions (local)
            scene_spec = None
            if hasattr(env, "env") and hasattr(env.env, "scene_spec_cache"):
                scene_spec = env.env.scene_spec_cache[env0]
            if scene_spec is None:
                print("[Debug] env0 scene_spec: None")
            else:
                num_static = len(scene_spec.static_obstacles)
                print(f"[Debug] env0 scene_spec static_obstacles: {num_static}")
                for idx, spec in enumerate(scene_spec.static_obstacles[:5]):
                    wx = ox + float(spec.position[0])
                    wy = oy + float(spec.position[1])
                    wz = oz + float(spec.position[2])
                    sx, sy, sz = spec.size
                    print(
                        f"[Debug] spec[{idx}] pos=({spec.position[0]:.3f},{spec.position[1]:.3f},{spec.position[2]:.3f}) "
                        f"world=({wx:.3f},{wy:.3f},{wz:.3f}) size=({sx:.3f},{sy:.3f},{sz:.3f})"
                    )
            # Applied actor positions (static)
            applied = []
            if hasattr(env, "env"):
                env_impl = env.env
                all_root_states = getattr(env_impl, "all_root_states", None)
                block_groups = getattr(env_impl, "static_block_groups", [])
                for group in block_groups:
                    active = getattr(env_impl, f"static_{group}_active", None)
                    pos_local = getattr(env_impl, f"static_{group}_pos_local", None)
                    indices = getattr(env_impl, f"static_{group}_actor_indices", None)
                    if active is None or pos_local is None or indices is None:
                        continue
                    active_ids = torch.nonzero(active[env0], as_tuple=False).flatten()
                    for local_id in active_ids.tolist():
                        actor_index = int(indices[env0, local_id].item())
                        pos = pos_local[env0, local_id]
                        applied.append((f"{group}:{local_id}", actor_index, pos))
                active = getattr(env_impl, "static_wall_active", None)
                pos_local = getattr(env_impl, "static_wall_pos_local", None)
                indices = getattr(env_impl, "static_wall_actor_indices", None)
                if active is not None and pos_local is not None and indices is not None:
                    active_ids = torch.nonzero(active[env0], as_tuple=False).flatten()
                    for local_id in active_ids.tolist():
                        actor_index = int(indices[env0, local_id].item())
                        pos = pos_local[env0, local_id]
                        applied.append((f"wall:{local_id}", actor_index, pos))
                if applied:
                    z_values = []
                    print(f"[Debug] env0 applied static obstacles: {len(applied)} (show first 5)")
                    for idx, (label, actor_index, pos) in enumerate(applied[:5]):
                        wx = ox + float(pos[0].item())
                        wy = oy + float(pos[1].item())
                        wz = oz + float(pos[2].item())
                        if all_root_states is not None:
                            actual = all_root_states[actor_index, :3].detach().cpu()
                            ax, ay, az = float(actual[0].item()), float(actual[1].item()), float(actual[2].item())
                            z_values.append(az)
                            print(
                                f"[Debug] actor[{label}] idx={actor_index} "
                                f"local=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                                f"world=({wx:.3f},{wy:.3f},{wz:.3f}) "
                                f"root=({ax:.3f},{ay:.3f},{az:.3f})"
                            )
                        else:
                            print(
                                f"[Debug] actor[{label}] idx={actor_index} "
                                f"local=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                                f"world=({wx:.3f},{wy:.3f},{wz:.3f})"
                            )
                    if z_values:
                        print(f"[Debug] static root z: min={min(z_values):.3f} max={max(z_values):.3f} (ground_z={oz:.3f})")
                else:
                    print("[Debug] env0 applied static obstacles: 0")
        except Exception as exc:
            print(f"[Debug] obstacle position dump failed: {exc}")
    state_dim = obs_dict['state'].shape[1]
    goal_dim = obs_dict['goal'].shape[1]
    aff_shape = obs_dict['gt_affordance'].shape[1:]
    aff_stack = max(int(getattr(args, "aff_stack", 1)), 1)
    aff_channels = aff_shape[0] * aff_stack

    # 创建 Policy (V5)
    skill = getattr(args, "skill", "follow")
    is_gate = skill == "moe"
    cmd_scale = tuple(float(v) for v in env.post_processor.max_cmd.detach().cpu().tolist())
    action_dim = 1 if is_gate else 3
    gate_use_difficulty = bool(getattr(args, "gate_use_difficulty", False))
    moe_use_student_aff = bool(getattr(args, "moe_use_student_aff", False))

    if is_gate:
        policy = GatePolicy(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
        ).to(device)
    else:
        policy = CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            cmd_scale=cmd_scale,
        ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # 加载 Teacher/Vision 模型 (Student 模式)
    teacher_model = None
    vision_model = None
    follow_model = None
    avoid_model = None

    if args.mode == 'teacher':
        dprint("[Main] Mode: TEACHER. Training from scratch with GT.")
    elif args.mode == 'student':
        dprint("\n[Student] 加载 Teacher 和 Vision 模型...")

        # 加载 Teacher
        if args.teacher_ckpt and not is_gate:
            teacher_model = CmdVelExpert(
                affordance_channels=aff_channels,
                state_dim=state_dim,
                goal_dim=goal_dim,
                cmd_scale=cmd_scale,
            ).to(device)

            ckpt = torch.load(args.teacher_ckpt, map_location=device)
            if 'model_state_dict' in ckpt:
                teacher_model.load_state_dict(ckpt['model_state_dict'])
            else:
                teacher_model.load_state_dict(ckpt)
            teacher_model.eval()

            # 用 Teacher 权重初始化 Student
            policy.load_state_dict(teacher_model.state_dict())
            dprint(f"[Student] ✓ Teacher 加载成功: {args.teacher_ckpt}")

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
            dprint(f"[Student] ✓ Vision 加载成功: {args.vision_ckpt}")
    if args.mode == 'teacher' and is_gate and moe_use_student_aff:
        if not getattr(args, "vision_ckpt", None):
            raise ValueError("moe_use_student_aff 需要提供 --vision_ckpt。")
        if AffordanceEstimator is None:
            raise ValueError("AffordanceEstimator 不可用，无法使用 student affordance。")
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
        dprint(f"[Gate] ✓ Vision 加载成功: {args.vision_ckpt}")

    if is_gate:
        if not getattr(args, "follow_ckpt", None) or not getattr(args, "avoid_ckpt", None):
            raise ValueError("Gate 模式需要提供 --follow_ckpt 和 --avoid_ckpt。")
        follow_model = CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            cmd_scale=cmd_scale,
        ).to(device)
        avoid_model = CmdVelExpert(
            affordance_channels=aff_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            cmd_scale=cmd_scale,
        ).to(device)
        for model, ckpt_path in [(follow_model, args.follow_ckpt), (avoid_model, args.avoid_ckpt)]:
            ckpt = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        dprint(f"[Gate] ✓ Follow/Avoid 加载完成: {args.follow_ckpt}, {args.avoid_ckpt}")
        if any(param.requires_grad for param in follow_model.parameters()):
            raise RuntimeError("Gate 模式下 Follow expert 仍存在可训练参数。")
        if any(param.requires_grad for param in avoid_model.parameters()):
            raise RuntimeError("Gate 模式下 Avoid expert 仍存在可训练参数。")

    resume_path = getattr(args, "resume", None)
    finetune_path = getattr(args, "finetune_from", None)
    if resume_path and finetune_path:
        raise ValueError("不能同时使用 --resume 与 --finetune_from。")

    start_iteration = 0
    best_reward = float("-inf")
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint 不存在: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict)
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print("[Warn] Resume checkpoint 未包含 optimizer_state_dict。")
        start_iteration = int(ckpt.get("iteration", 0)) + 1 if isinstance(ckpt, dict) else 0
        best_reward = float(ckpt.get("best_reward", ckpt.get("mean_reward", -float("inf")))) if isinstance(ckpt, dict) else best_reward
        if isinstance(ckpt, dict) and "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if isinstance(ckpt, dict) and "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if torch.cuda.is_available() and isinstance(ckpt, dict) and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        log_dir = os.path.dirname(resume_path)
        dprint(f"[Main] Resume: {resume_path}")
    elif finetune_path:
        if not os.path.exists(finetune_path):
            raise FileNotFoundError(f"Finetune checkpoint 不存在: {finetune_path}")
        ckpt = torch.load(finetune_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict)
        dprint(f"[Main] Finetune from: {finetune_path}")

    # 创建日志目录
    if resume_path:
        os.makedirs(log_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(args.output_dir, f"{skill}_{args.mode}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"[Main] 日志目录: {log_dir}")
    
    # 创建 Rollout Buffer
    aff_map_shape = (aff_channels, aff_shape[1], aff_shape[2])
    buffer = RolloutBuffer(env.num_envs, args.num_steps, state_dim, goal_dim, aff_map_shape, action_dim, device)
    
    # 训练统计
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    goal_change_counts = deque(maxlen=100)
    if not resume_path:
        best_reward = float("-inf")
    
    # 训练内的 episode 回报统计
    running_returns = torch.zeros(env.num_envs, device=device)
    
    total_iterations = start_iteration + args.num_iterations
    print(f"\n[Main] 开始训练 (iterations={args.num_iterations}, start={start_iteration})...")
    dprint(f"  - Steps per iteration: {args.num_steps}")
    dprint(f"  - Batch size: {env.num_envs * args.num_steps}")
    dprint(f"  - Learning rate: {args.lr}")
    
    aff_stack_buf = None
    last_goal_obs = None
    teacher_stack_buf = None
    stack_reset_mask = None
    last_dones = None
    aff_stack_fill = None
    for iteration in range(start_iteration, total_iterations):
        start_time = time.time()
        
        # ============ Rollout Phase ============
        buffer.reset()
        rollout_start = time.time()

        reward_term_keys = [
            'approach',
            'reach',
            'heading',
            'passable_align',
            'passable_gate',
            'crossable_align',
            'crossable_gate',
            'risk_barrier',
            'gate_smooth',
            'collision',
            'stability',
            'velocity',
            'time',
            'total',
        ]
        reward_term_sums = {k: torch.zeros((), device=device) for k in reward_term_keys}
        gate_y_sum = torch.zeros((), device=device)
        gate_y_change_sum = torch.zeros((), device=device)
        cmd_speed_sum = torch.zeros((), device=device)
        goal_dist_sum = torch.zeros((), device=device)
        collision_rate_sum = torch.zeros((), device=device)
        collision_force_samples = []
        collision_threshold_value = None
        collision_threshold_src_value = None
        collision_indices_src_value = None
        clearance_sum = torch.zeros((), device=device)
        passable_occ_ratio_sum = torch.zeros((), device=device)
        crossable_width_sum = torch.zeros((), device=device)
        risk_scale_sum = torch.zeros((), device=device)
        aff_stack_delta_sum = torch.zeros((), device=device)
        aff_stack_std_sum = torch.zeros((), device=device)
        aff_stack_filled_sum = torch.zeros((), device=device)
        
        for step in range(args.num_steps):
            # 准备输入
            state = obs_dict['state']
            goal_raw = obs_dict['goal']
            
            use_student_aff = args.mode != 'teacher'
            if is_gate and moe_use_student_aff:
                use_student_aff = True
            if is_gate and args.mode == 'teacher' and moe_use_student_aff:
                print("[Warn] moe_use_student_aff=True 但当前 mode=teacher，仍会使用 student affordance。")
            if use_student_aff:
                # Student: 使用 Vision 模型预测
                if vision_model is None:
                    raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
                with torch.no_grad():
                    vis_out = vision_model(obs_dict['depth'], normalize=True)
                    aff_map = torch.stack([
                        vis_out['occupancy'],
                        vis_out['passable_gap'],
                        vis_out['low_obstacle'],
                    ], dim=1)
                    difficulty = difficulty_from_gap(aff_map)
                env.clearance_override = env._compute_clearance_from_affordance(aff_map)
                env.reward_affordance_override = aff_map
            else:
                aff_map = obs_dict['gt_affordance']
                difficulty = obs_dict['gt_difficulty']
                env.clearance_override = None
                env.reward_affordance_override = None

            # 初始化/更新 aff 堆叠
            if aff_stack_buf is None:
                aff_stack_buf = aff_map.repeat(1, aff_stack, 1, 1)
                aff_stack_fill = torch.ones(env.num_envs, device=device)
            else:
                reset_mask = stack_reset_mask
                if stack_reset_mask is not None and stack_reset_mask.any():
                    aff_stack_buf[stack_reset_mask] = aff_map[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    aff_stack_fill[stack_reset_mask] = 1
                    if teacher_stack_buf is not None:
                        teacher_aff = obs_dict['gt_affordance']
                        teacher_stack_buf[stack_reset_mask] = teacher_aff[stack_reset_mask].repeat(1, aff_stack, 1, 1)
                    stack_reset_mask = None
                aff_stack_buf = torch.roll(aff_stack_buf, shifts=-aff_map.shape[1], dims=1)
                aff_stack_buf[:, -aff_map.shape[1]:, :, :] = aff_map
                if aff_stack > 1:
                    if reset_mask is None:
                        aff_stack_fill = torch.clamp(aff_stack_fill + 1, max=aff_stack)
                    else:
                        inc_mask = ~reset_mask
                        if inc_mask.any():
                            aff_stack_fill[inc_mask] = torch.clamp(aff_stack_fill[inc_mask] + 1, max=aff_stack)
                else:
                    aff_stack_fill.fill_(1)

            # 短时记忆诊断：堆叠帧变化/方差/填满比例
            if aff_stack > 1:
                base_channels = aff_map.shape[1]
                stack_h, stack_w = aff_map.shape[2], aff_map.shape[3]
                stack = aff_stack_buf.reshape(env.num_envs, aff_stack, base_channels, stack_h, stack_w)
                delta = (stack[:, 1:] - stack[:, :-1]).abs().mean(dim=(1, 2, 3, 4))
                stack_std = stack.std(dim=1, unbiased=False).mean(dim=(1, 2, 3))
                aff_stack_delta_sum += delta.mean()
                aff_stack_std_sum += stack_std.mean()
                aff_stack_filled_sum += (aff_stack_fill / aff_stack).mean()
            else:
                aff_stack_delta_sum += 0.0
                aff_stack_std_sum += 0.0
                aff_stack_filled_sum += 1.0

            # T1: goal occlusion (weak, input-only)
            if getattr(args, "t1_goal_occlusion", False):
                prob = float(getattr(args, "t1_goal_occlusion_prob", 0.02))
                max_steps = int(getattr(args, "t1_goal_occlusion_len", 10))
                if prob > 0.0 and max_steps > 0:
                    if (
                        not hasattr(env, "t1_occ_steps")
                        or env.t1_occ_steps is None
                        or env.t1_occ_steps.shape[0] != env.num_envs
                    ):
                        env.t1_occ_steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
                    active = env.t1_occ_steps > 0
                    start = (torch.rand(env.num_envs, device=device) < prob) & (~active)
                    if start.any():
                        env.t1_occ_steps[start] = torch.randint(1, max_steps + 1, (start.sum(),), device=device)
                    occ_mask = env.t1_occ_steps > 0
                    goal, last_goal_obs = apply_goal_occlusion(goal_raw, last_goal_obs, occ_mask)
                    env.t1_occ_steps[occ_mask] -= 1
                else:
                    goal = goal_raw
                    last_goal_obs = goal_raw.detach()
            else:
                goal = goal_raw
                last_goal_obs = goal_raw.detach()

            # Policy 决策
            teacher_action = None
            gate_y = None
            cmd_used = None
            if is_gate:
                gate_difficulty = difficulty if gate_use_difficulty else torch.zeros_like(difficulty)
                with torch.no_grad():
                    cmd_f, _ = follow_model.get_action(
                        aff_stack_buf, state, goal, difficulty,
                        deterministic=bool(getattr(args, "moe_expert_deterministic", True))
                    )
                    cmd_a, _ = avoid_model.get_action(
                        aff_stack_buf, state, goal, difficulty,
                        deterministic=bool(getattr(args, "moe_expert_deterministic", True))
                    )
                    cmd_f = cmd_f.detach()
                    cmd_a = cmd_a.detach()
                gate_y, _ = policy.get_action(
                    aff_stack_buf, state, goal, gate_difficulty, deterministic=False
                )
                gate_y_prev = env.prev_gate_y.clone()
                if getattr(args, "gate_safe_clamp", False):
                    safe_max = float(getattr(args, "gate_safe_max", 0.3))
                    clearance_gate = env._compute_clearance_from_affordance(aff_map)
                    gate_y = torch.where(
                        clearance_gate < env.post_processor.safe_distance,
                        torch.minimum(gate_y, torch.full_like(gate_y, safe_max)),
                        gate_y,
                    )
                log_prob, value, _, _ = policy.evaluate_actions(
                    aff_stack_buf, state, goal, gate_difficulty, gate_y
                )
                cmd = gate_y.unsqueeze(-1) * cmd_f + (1.0 - gate_y.unsqueeze(-1)) * cmd_a
                next_obs, rewards, dones, env_info = env.step(cmd, gate_y)
                action = gate_y.unsqueeze(-1)
                cmd_used = cmd
            else:
                cmd, _ = policy.get_action(
                    aff_stack_buf, state, goal, difficulty, deterministic=False
                )
                log_prob, value, _, _ = policy.evaluate_actions(
                    aff_stack_buf, state, goal, difficulty, cmd
                )

                if args.mode == 'student' and teacher_model is not None:
                    with torch.no_grad():
                        teacher_aff = obs_dict['gt_affordance']
                        if teacher_stack_buf is None:
                            teacher_stack_buf = teacher_aff.repeat(1, aff_stack, 1, 1)
                        else:
                            teacher_stack_buf = torch.roll(teacher_stack_buf, shifts=-teacher_aff.shape[1], dims=1)
                            teacher_stack_buf[:, -teacher_aff.shape[1]:, :, :] = teacher_aff
                        t_cmd, _ = teacher_model.get_action(
                            teacher_stack_buf,
                            state, goal,
                            obs_dict['gt_difficulty'],
                            deterministic=True
                        )
                        teacher_action = t_cmd

                next_obs, rewards, dones, env_info = env.step(cmd)
                action = cmd
                gate_y_prev = None
                cmd_used = cmd
            last_dones = dones.clone()

            # 存储数据
            buffer.add(
                state.detach(),
                goal.detach(),
                aff_stack_buf.detach(),
                difficulty.detach(),
                action.detach(),
                log_prob.detach(),
                value.detach(),
                rewards.detach(),
                dones.detach(),
                teacher_action.detach() if teacher_action is not None else None,
            )

            # 统计分量与行为
            goal_dist_sum += torch.norm(goal, dim=1).sum()
            if is_gate:
                gate_y_sum += gate_y.sum()
                if gate_y_prev is not None:
                    gate_y_change_sum += torch.abs(gate_y - gate_y_prev).sum()
            if hasattr(env.env, "commands"):
                cmd_speed_sum += torch.norm(env.env.commands[:, :2], dim=1).sum()
            else:
                cmd_speed_sum += torch.norm(cmd_used[:, :2], dim=1).sum()
            reward_terms = env_info.get('reward_terms', None)
            if reward_terms is not None:
                for key in reward_term_keys:
                    if key in reward_terms:
                        reward_term_sums[key] += reward_terms[key].sum()
                if 'risk_scale' in reward_terms:
                    risk_scale_sum += reward_terms['risk_scale'].sum()
                if 'passable_occ_ratio' in reward_terms:
                    passable_occ_ratio_sum += reward_terms['passable_occ_ratio'].sum()
                if 'crossable_width' in reward_terms:
                    crossable_width_sum += reward_terms['crossable_width'].sum()
            collision_mask = env_info.get('collision_mask', None) if env_info is not None else None
            if collision_mask is not None:
                collision_rate_sum += collision_mask.float().sum()
            collision_force_max = env_info.get('collision_force_max', None) if env_info is not None else None
            if collision_force_max is not None:
                collision_force_samples.append(collision_force_max.detach().cpu())
            collision_threshold = env_info.get('collision_threshold', None) if env_info is not None else None
            if collision_threshold is not None:
                collision_threshold_value = float(collision_threshold)
            collision_threshold_src = env_info.get('collision_threshold_src', None) if env_info is not None else None
            if collision_threshold_src is not None:
                collision_threshold_src_value = collision_threshold_src
            collision_indices_src = env_info.get('collision_indices_src', None) if env_info is not None else None
            if collision_indices_src is not None:
                collision_indices_src_value = collision_indices_src
            clearance = env_info.get('clearance', None) if env_info is not None else None
            if clearance is not None:
                clearance_sum += clearance.sum()

            # 统计完成的 episode（优先使用高层累计的 episode_return）
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                episode = env_info.get('episode', None) if env_info is not None else None
                if episode is not None:
                    ep_len = episode['l'][done_ids].detach().cpu().tolist()
                    ep_ret = episode['r'][done_ids].detach().cpu().tolist()
                    episode_lengths.extend(ep_len)
                    episode_rewards.extend(ep_ret)
                else:
                    running_returns += rewards
                    ep_len = env_info['episode_length'][done_ids].detach().cpu().tolist()
                    episode_lengths.extend(ep_len)
                    episode_rewards.extend(running_returns[done_ids].detach().cpu().tolist())
                    running_returns[done_ids] = 0.0
                goal_changes = env_info.get('goal_change_count', None)
                if goal_changes is not None:
                    goal_change_counts.extend(goal_changes[done_ids].detach().cpu().tolist())
            else:
                running_returns += rewards
            
            if dones.any():
                stack_reset_mask = dones.clone()
            obs_dict = next_obs
            if dones.any():
                if last_goal_obs is not None:
                    last_goal_obs = last_goal_obs.clone()
                    last_goal_obs[dones] = obs_dict['goal'][dones]

        rollout_time = time.time() - rollout_start
        
        # ============ Update Phase ============
        # 计算最后一步的 value (Bootstrap)
        with torch.no_grad():
            state = obs_dict['state']
            goal = obs_dict['goal']
            
            if args.mode == 'teacher':
                aff_map_next = obs_dict['gt_affordance']
                difficulty = obs_dict['gt_difficulty']
            else:
                if vision_model is None:
                    raise ValueError("Student 模式必须提供 --vision_ckpt，以确保仅使用相机输入。")
                with torch.no_grad():
                    vis_out = vision_model(obs_dict['depth'], normalize=True)
                    aff_map_next = torch.stack([
                        vis_out['occupancy'],
                        vis_out['passable_gap'],
                        vis_out['low_obstacle'],
                    ], dim=1)
                    difficulty = difficulty_from_gap(aff_map_next)
            
            if aff_stack_buf is None:
                aff_stack_bootstrap = aff_map_next.repeat(1, aff_stack, 1, 1)
            else:
                aff_stack_bootstrap = torch.roll(aff_stack_buf, shifts=-aff_map_next.shape[1], dims=1)
                aff_stack_bootstrap[:, -aff_map_next.shape[1]:, :, :] = aff_map_next
                if last_dones is not None and last_dones.any():
                    done_mask = last_dones.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    aff_stack_bootstrap = torch.where(
                        done_mask,
                        aff_map_next.repeat(1, aff_stack, 1, 1),
                        aff_stack_bootstrap,
                    )

            if is_gate:
                gate_difficulty = difficulty if gate_use_difficulty else torch.zeros_like(difficulty)
                out = policy(aff_stack_bootstrap, state, goal, gate_difficulty)
            else:
                out = policy(aff_stack_bootstrap, state, goal, difficulty)
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
        approx_kl_sum = 0
        clip_frac_sum = 0
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
                mb_actions = buffer.actions[step_idx, env_idx]
                mb_old_log_probs = buffer.log_probs[step_idx, env_idx]
                mb_advantages = advantages[step_idx, env_idx]
                mb_returns = returns[step_idx, env_idx]
                
                # 读取 rollout 缓冲区
                mb_states = buffer.states[step_idx, env_idx]
                mb_goals = buffer.goals[step_idx, env_idx]
                mb_aff_maps = buffer.aff_maps[step_idx, env_idx]
                mb_difficulties = buffer.difficulties[step_idx, env_idx]
                
                # 评估当前策略
                if is_gate:
                    gate_difficulty = mb_difficulties if gate_use_difficulty else torch.zeros_like(mb_difficulties)
                    new_log_probs, new_values, entropy, _ = policy.evaluate_actions(
                        mb_aff_maps, mb_states, mb_goals, gate_difficulty,
                        mb_actions
                    )
                else:
                    new_log_probs, new_values, entropy, _ = policy.evaluate_actions(
                        mb_aff_maps, mb_states, mb_goals, mb_difficulties,
                        mb_actions
                    )
                
                # PPO Clipped Loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Diagnostics
                log_ratio = new_log_probs - mb_old_log_probs
                approx_kl_sum += (0.5 * (log_ratio ** 2).mean()).item()
                clip_frac_sum += (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()
                
                # Value Loss
                value_loss = 0.5 * ((new_values.squeeze(-1) - mb_returns) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = -entropy.mean()
                
                # 蒸馏 Loss (Student 模式)
                distill_loss = torch.tensor(0.0, device=device)
                if args.mode == 'student' and teacher_model is not None and not is_gate:
                    mb_teacher_actions = buffer.teacher_actions[step_idx, env_idx]
                    distill_loss = nn.functional.mse_loss(mb_actions, mb_teacher_actions)
                
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
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                distill_loss_sum += distill_loss.item()
                num_updates += 1
        
        update_time = time.time() - start_time - rollout_time

        # Explained variance (基于 rollout value 估计)
        returns_flat = returns.view(-1)
        values_flat = buffer.values.view(-1)
        var_returns = returns_flat.var()
        explained_var = 1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)

        # ============ Logging ============
        iter_time = time.time() - start_time
        fps = batch_size / iter_time
        
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0
        sum_reward = float(np.sum(episode_rewards)) if episode_rewards else 0.0
        sum_length = float(np.sum(episode_lengths)) if episode_lengths else 0.0
        mean_goal_changes = np.mean(goal_change_counts) if goal_change_counts else 0
        
        total_samples = env.num_envs * args.num_steps
        reward_term_means = {k: (v / total_samples).item() for k, v in reward_term_sums.items()}
        cmd_speed_mean = (cmd_speed_sum / total_samples).item()
        gate_y_mean = (gate_y_sum / total_samples).item() if is_gate else 0.0
        gate_y_change_mean = (gate_y_change_sum / total_samples).item() if is_gate else 0.0
        risk_scale_mean = (risk_scale_sum / total_samples).item()
        goal_dist_mean = (goal_dist_sum / total_samples).item()
        mean_step_reward = (sum_reward / sum_length) if sum_length > 0 else 0.0
        rollout_mean_step_reward = buffer.rewards.mean().item()
        breakdown_total_mean = reward_term_means.get('total', 0.0)
        reward_residual = mean_step_reward - breakdown_total_mean
        resid_rollout = rollout_mean_step_reward - breakdown_total_mean
        collision_rate_mean = (collision_rate_sum / total_samples).item()
        collision_force_mean = 0.0
        collision_force_p95 = 0.0
        if collision_force_samples:
            collision_force_all = torch.cat(collision_force_samples, dim=0)
            collision_force_mean = collision_force_all.mean().item()
            collision_force_p95 = torch.quantile(collision_force_all, 0.95).item()
        clearance_mean = (clearance_sum / total_samples).item()
        passable_occ_ratio_mean = (passable_occ_ratio_sum / total_samples).item()
        crossable_width_mean = (crossable_width_sum / total_samples).item()
        risk_scale_mean = (risk_scale_sum / total_samples).item()
        aff_stack_delta_mean = (aff_stack_delta_sum / args.num_steps).item()
        aff_stack_std_mean = (aff_stack_std_sum / args.num_steps).item()
        aff_stack_filled_mean = (aff_stack_filled_sum / args.num_steps).item()
        terrain_level_mean = None
        terrain_level_max = None
        if hasattr(env.env, "terrain_levels"):
            terrain_level_mean = env.env.terrain_levels.float().mean().item()
            terrain_level_max = env.env.terrain_levels.max().item()

        # TensorBoard
        writer.add_scalar('Loss/Total', total_loss / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Policy', policy_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Value', value_loss_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Loss/Entropy', entropy_sum / max(num_updates, 1), iteration)
        if args.mode == 'student':
            writer.add_scalar('Loss/Distill', distill_loss_sum / max(num_updates, 1), iteration)
        
        writer.add_scalar('Perf/MeanReward', mean_reward, iteration)
        writer.add_scalar('Perf/MeanLength', mean_length, iteration)
        writer.add_scalar('Perf/MeanStepReward', mean_step_reward, iteration)
        writer.add_scalar('Perf/RewardResidual', reward_residual, iteration)
        writer.add_scalar('Perf/RolloutMeanStepReward', rollout_mean_step_reward, iteration)
        writer.add_scalar('Perf/RolloutRewardResidual', resid_rollout, iteration)
        writer.add_scalar('Perf/FPS', fps, iteration)
        writer.add_scalar('Perf/GoalChangeCount', mean_goal_changes, iteration)
        writer.add_scalar('Perf/DifficultyMean', buffer.difficulties.mean().item(), iteration)
        writer.add_scalar('Perf/DifficultyMin', buffer.difficulties.min().item(), iteration)
        writer.add_scalar('Perf/DifficultyMax', buffer.difficulties.max().item(), iteration)
        if terrain_level_mean is not None:
            writer.add_scalar('Perf/TerrainLevelMean', terrain_level_mean, iteration)
        if terrain_level_max is not None:
            writer.add_scalar('Perf/TerrainLevelMax', terrain_level_max, iteration)
        writer.add_scalar('Diag/ApproxKL', approx_kl_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Diag/ClipFrac', clip_frac_sum / max(num_updates, 1), iteration)
        writer.add_scalar('Diag/ExplainedVar', explained_var.item(), iteration)
        writer.add_scalar('Diag/CollisionRate', collision_rate_mean, iteration)
        writer.add_scalar('Diag/CollisionForceMean', collision_force_mean, iteration)
        writer.add_scalar('Diag/CollisionForceP95', collision_force_p95, iteration)
        writer.add_scalar('Diag/AffStackDelta', aff_stack_delta_mean, iteration)
        writer.add_scalar('Diag/AffStackStd', aff_stack_std_mean, iteration)
        writer.add_scalar('Diag/AffStackFilled', aff_stack_filled_mean, iteration)
        if collision_threshold_value is not None:
            writer.add_scalar('Diag/CollisionThreshold', collision_threshold_value, iteration)
        writer.add_scalar('Stats/ClearanceMin', clearance_mean, iteration)
        writer.add_scalar('Stats/PassableOccRatio', passable_occ_ratio_mean, iteration)
        writer.add_scalar('Stats/CrossableWidth', crossable_width_mean, iteration)
        writer.add_scalar('Stats/RiskScale', risk_scale_mean, iteration)
        if is_gate:
            writer.add_scalar('Stats/GateY', gate_y_mean, iteration)
            writer.add_scalar('Stats/GateYChange', gate_y_change_mean, iteration)
        writer.add_scalar('Stats/CmdSpeed', cmd_speed_mean, iteration)
        writer.add_scalar('Stats/GoalDist', goal_dist_mean, iteration)
        for key, value in reward_term_means.items():
            writer.add_scalar(f'Reward/{key}', value, iteration)
        
        # Console
        if iteration % args.log_interval == 0:
            width = 80
            pad = 32
            header = f" Learning iteration {iteration}/{total_iterations} "
            collision_threshold_str = "n/a"
            collision_threshold_src_str = "n/a"
            collision_indices_src_str = "n/a"
            if collision_threshold_value is not None:
                collision_threshold_str = f"{collision_threshold_value:.3f}"
            if collision_threshold_src_value is not None:
                collision_threshold_src_str = str(collision_threshold_src_value)
            if collision_indices_src_value is not None:
                collision_indices_src_str = str(collision_indices_src_value)
            terrain_level_str = "n/a"
            if terrain_level_mean is not None:
                terrain_level_str = f"{terrain_level_mean:.2f}"
            gate_line = ""
            if is_gate:
                gate_line = f"""{'Gate y / Δy:':>{pad}} {gate_y_mean:.3f} / {gate_y_change_mean:.3f}\n"""
            log_string = (f"""{'#' * width}\n"""
                          f"""{header.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (rollout {rollout_time:.3f}s, update {update_time:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {value_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {policy_loss_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Entropy loss:':>{pad}} {entropy_sum / max(num_updates, 1):.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                          f"""{'Mean step reward:':>{pad}} {mean_step_reward:.3f} (resid {reward_residual:.3f})\n"""
                          f"""{'Rollout step reward:':>{pad}} {rollout_mean_step_reward:.3f} (resid {resid_rollout:.3f})\n"""
                          f"""{'Mean episode length:':>{pad}} {mean_length:.2f}\n"""
                          f"""{'Goal change count:':>{pad}} {mean_goal_changes:.2f}\n"""
                          f"""{'Curriculum level:':>{pad}} {terrain_level_str}\n"""
                          f"""{'Approx KL / Clip frac:':>{pad}} {approx_kl_sum / max(num_updates, 1):.4f} / {clip_frac_sum / max(num_updates, 1):.3f}\n"""
                          f"""{'Explained variance:':>{pad}} {explained_var.item():.4f}\n"""
                          f"""{'Goal dist / Cmd speed:':>{pad}} {goal_dist_mean:.3f} / {cmd_speed_mean:.3f}\n"""
                          f"""{'Clearance / RiskScale:':>{pad}} {clearance_mean:.3f} / {risk_scale_mean:.3f}\n"""
                          f"""{gate_line}"""
                          f"""{'Passable gate/align:':>{pad}} {reward_term_means.get('passable_gate', 0.0):.3f} / {reward_term_means.get('passable_align', 0.0):.3f} (occ {passable_occ_ratio_mean:.3f})\n"""
                          f"""{'Crossable gate/align:':>{pad}} {reward_term_means.get('crossable_gate', 0.0):.3f} / {reward_term_means.get('crossable_align', 0.0):.3f} (width {crossable_width_mean:.3f})\n"""
                          f"""{'AffStack d/std/filled:':>{pad}} {aff_stack_delta_mean:.3f} / {aff_stack_std_mean:.3f} / {aff_stack_filled_mean:.3f}\n"""
                          f"""{'Collision rate/force:':>{pad}} {collision_rate_mean:.3f} / {collision_force_mean:.3f} (p95 {collision_force_p95:.3f}, th {collision_threshold_str} {collision_threshold_src_str}, idx {collision_indices_src_str})\n"""
                          f"""{'-' * width}\n"""
                          f"""{'Reward(approach/reach/heading):':>{pad}} {reward_term_means.get('approach', 0.0):.3f} / {reward_term_means.get('reach', 0.0):.3f} / {reward_term_means.get('heading', 0.0):.3f}\n"""
                          f"""{'Reward(gate/risk/col):':>{pad}} {reward_term_means.get('gate_smooth', 0.0):.3f} / {reward_term_means.get('risk_barrier', 0.0):.3f} / {reward_term_means.get('collision', 0.0):.3f}\n"""
                          f"""{'Reward(velocity/time/total):':>{pad}} {reward_term_means.get('velocity', 0.0):.3f} / {reward_term_means.get('time', 0.0):.3f} / {reward_term_means.get('total', 0.0):.3f}\n"""
                          f"""{'#' * width}\n""")
            print(log_string)
        
        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            ckpt_path = os.path.join(log_dir, f'model_{iteration}.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean_reward': mean_reward,
                'best_reward': best_reward,
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, ckpt_path)
            dprint(f"  Saved: {ckpt_path}")
        
        # Save best
        if mean_reward > best_reward and len(episode_rewards) >= 10:
            best_reward = mean_reward
            best_path = os.path.join(log_dir, 'best_model.pt')
            torch.save({
                'iteration': iteration,
                'model_state_dict': policy.state_dict(),
                'mean_reward': mean_reward,
                'best_reward': best_reward,
            }, best_path)
            dprint(f"  ★ New best: {mean_reward:.2f}")
    
    # 训练结束
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Output: {log_dir}")
    print(f"{'='*60}")
    
    writer.close()
    return policy

    

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V5 Command-Space MoE Training")
    
    # 模式
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['teacher', 'student'],
                        help='训练模式: teacher (GT) 或 student (Vision)')
    parser.add_argument('--skill', type=str, default='follow',
                        choices=['follow', 'avoid', 'moe'],
                        help='训练技能: follow / avoid / moe (gate)')
    
    # 环境
    parser.add_argument('--task', type=str, default='hex_ground',
                        help='Isaac Gym 任务名称')
    parser.add_argument('--aff_stack', type=int, default=1,
                        help='affordance 堆叠帧数 (短时记忆)')
    parser.add_argument('--num_envs', type=int, default=4096,
                        help='并行环境数量')
    parser.add_argument('--decimation', type=int, default=5,
                        help='高层/低层频率比 (50Hz / 10Hz = 5)')
    
    # Checkpoints
    parser.add_argument('--low_level_ckpt', type=str, required=True,
                        help='底层控制器路径')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='(Student) Teacher 模型路径')
    parser.add_argument('--follow_ckpt', type=str, default=None,
                        help='(Gate) Follow expert 模型路径')
    parser.add_argument('--avoid_ckpt', type=str, default=None,
                        help='(Gate) Avoid expert 模型路径')
    parser.add_argument('--vision_ckpt', type=str, default=None,
                        help='(Student) Vision 模型路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练 checkpoint 路径（含优化器与迭代）')
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='微调 checkpoint 路径（仅加载权重）')
    
    # 训练超参数
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='训练迭代次数')
    parser.add_argument('--num_steps', type=int, default=24,
                        help='每次迭代的步数')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='PPO epoch 数')
    parser.add_argument('--mini_batch_size', type=int, default=4096,
                        help='Mini-batch 大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.1,
                        help='PPO clip range')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='Value loss 系数')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy 系数')
    parser.add_argument('--distill_coef', type=float, default=1.0,
                        help='蒸馏 loss 系数 (Student 模式)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='梯度裁剪')
    parser.add_argument('--cmd_slew_lin', type=float, default=0.2,
                        help='命令线速度变化率限制')
    parser.add_argument('--cmd_slew_ang', type=float, default=0.4,
                        help='命令角速度变化率限制')
    parser.add_argument('--cmd_safe_dist', type=float, default=None,
                        help='安全距离阈值（None 使用默认 clearance）')
    parser.add_argument('--cmd_free_dist', type=float, default=None,
                        help='安全全速距离（None 使用默认 clearance_free）')
    parser.add_argument('--gate_safe_clamp', action='store_true',
                        help='Gate 训练早期启用安全 clamp')
    parser.add_argument('--gate_safe_max', type=float, default=0.3,
                        help='安全 clamp 的最大 y 值')
    parser.add_argument('--moe_expert_deterministic', action='store_true',
                        help='Gate 训练时专家使用确定性输出')
    parser.add_argument('--gate_use_difficulty', action='store_true',
                        help='Gate 使用 difficulty 作为输入（特权信息）')
    parser.add_argument('--moe_use_student_aff', action='store_true',
                        help='moe 模式强制使用 student affordance（与部署一致）')
    parser.add_argument('--disable_risk_scale', action='store_true',
                        help='禁用 CommandPostProcessor 的风险缩放（消融用）')
    parser.add_argument('--t1_goal_occlusion', action='store_true',
                        help='训练中启用 T1 弱遮挡（仅影响 policy 输入）')
    parser.add_argument('--t1_goal_occlusion_prob', type=float, default=0.02,
                        help='T1 弱遮挡触发概率')
    parser.add_argument('--t1_goal_occlusion_len', type=int, default=10,
                        help='T1 弱遮挡持续步数')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='outputs/planner',
                        help='输出目录')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志间隔')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='保存间隔')
    parser.add_argument('--debug', action='store_true',
                        help='debug 输出（场景/障碍位置等）')
    
    args, unknown = parser.parse_known_args()
    
    # 传递未知参数给 Isaac Gym
    sys.argv = [sys.argv[0]] + unknown
    
    train(args)
