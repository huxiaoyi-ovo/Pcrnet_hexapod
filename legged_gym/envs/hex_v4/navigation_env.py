"""
navigation_env.py - 导航任务环境与奖励函数 V2

改动说明 (相比V1):
- 移除: 步态效率奖励 (gait_optimal_bonus, gait_switch_penalty)
- 新增: 运动强度适配奖励 (intensity_match, intensity_smooth)
- 修改: 使用连续的地形难度而非离散步态

创新点:
- 强度-地形匹配奖励: 鼓励在简单地形用高强度，复杂地形用低强度
- 强度平滑惩罚: 避免运动强度剧烈波动
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class NavigationRewardConfig:
    """导航奖励配置 V2"""

    # 目标接近奖励
    goal_approach_scale: float = 2.0      # 接近目标的奖励系数
    goal_reach_bonus: float = 10.0        # 到达目标的额外奖励
    goal_reach_threshold: float = 0.3     # 到达判定阈值 (m)

    # 碰撞惩罚
    collision_penalty: float = -10.0

    # 朝向奖励
    heading_scale: float = 0.5            # 朝向目标的奖励
    heading_use_difficulty_gate: bool = False  # 根据地形难度弱化朝向奖励
    heading_min_weight: float = 0.2       # 朝向奖励最小权重 (仅门控开启时生效)
    heading_gate_use: bool = False        # 速度/进度门控
    heading_gate_min_speed: float = 0.05  # 低速门控阈值
    heading_gate_min_approach: float = 0.0  # 进度门控阈值

    # ★运动强度奖励 (替代步态效率)★
    intensity_match_bonus: float = 0.2    # 强度匹配地形的奖励
    intensity_mismatch_penalty: float = -0.1  # 强度不匹配的惩罚
    intensity_smooth_penalty: float = -0.05   # 强度变化惩罚
    max_intensity_change: float = 0.2     # 每步最大强度变化
    intensity_gate_use: bool = False      # 强度匹配门控
    intensity_gate_min_speed: float = 0.05  # 强度门控速度阈值
    intensity_gate_min_approach: float = 0.0  # 强度门控进度阈值

    # 时间惩罚
    time_penalty: float = -0.01           # 每步时间惩罚

    # 稳定性奖励
    stability_scale: float = 0.1
    pitch_threshold: float = 0.3          # rad, 约17度
    roll_threshold: float = 0.3

    # 速度奖励
    velocity_scale: float = 0.1


class NavigationRewardFunction:
    """导航任务奖励函数 V2 ★核心设计★

    奖励结构:
    1. 主奖励 (目标导向):
       - 接近目标: r = scale × (d_{t-1} - d_t)
       - 到达奖励: r = bonus if d < threshold

    2. 强度适配 (地形自适应) ★新增★:
       - 最优强度 = 1 - 地形难度
       - 强度匹配奖励/惩罚
       - 强度平滑惩罚

    3. 安全性:
       - 碰撞惩罚
       - 姿态稳定性

    4. 效率:
       - 时间惩罚

    强度-地形关系:
    地形难度 d   →  最优强度 λ* = 1 - d

    d ≈ 0 (简单) →  λ* ≈ 1 (激进，快速通过)
    d ≈ 0.5 (中等) → λ* ≈ 0.5 (平衡)
    d ≈ 1 (困难) →  λ* ≈ 0 (保守，谨慎通过)
    """

    def __init__(self, cfg: NavigationRewardConfig = None):
        self.cfg = cfg or NavigationRewardConfig()

    def compute_reward(
        self,
        robot_pos: torch.Tensor,           # (N, 3) 当前位置
        prev_robot_pos: torch.Tensor,      # (N, 3) 上一位置
        goal_pos: torch.Tensor,            # (N, 2) 目标位置
        robot_vel: torch.Tensor,           # (N, 3) 速度
        robot_quat: torch.Tensor,          # (N, 4) 姿态四元数
        intensity: torch.Tensor,           # (N,) 当前运动强度 ★新增★
        prev_intensity: torch.Tensor,      # (N,) 上一运动强度 ★新增★
        terrain_difficulty: torch.Tensor,  # (N,) 地形难度 ★修改★
        collision_mask: torch.Tensor,      # (N,) 是否碰撞
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合奖励

        Returns:
            dict: 各项奖励分量和总奖励
        """
        device = robot_pos.device
        num_envs = robot_pos.shape[0]

        rewards = {}

        # 1. 目标接近奖励
        dist_to_goal = torch.norm(robot_pos[:, :2] - goal_pos, dim=-1)
        prev_dist = torch.norm(prev_robot_pos[:, :2] - goal_pos, dim=-1)

        # 进度奖励: 距离减少则正奖励
        approach_reward = (prev_dist - dist_to_goal) * self.cfg.goal_approach_scale
        rewards['approach'] = approach_reward

        # 到达目标奖励
        reached = dist_to_goal < self.cfg.goal_reach_threshold
        reach_reward = reached.float() * self.cfg.goal_reach_bonus
        rewards['reach'] = reach_reward

        # 2. 朝向奖励
        heading = self._quat_to_heading(robot_quat)
        goal_direction = torch.atan2(
            goal_pos[:, 1] - robot_pos[:, 1],
            goal_pos[:, 0] - robot_pos[:, 0]
        )
        heading_error = self._angle_diff(heading, goal_direction)
        heading_weight = 1.0
        if self.cfg.heading_use_difficulty_gate:
            heading_weight = torch.clamp(
                1.0 - terrain_difficulty,
                min=self.cfg.heading_min_weight,
                max=1.0,
            )
        heading_reward = torch.cos(heading_error) * self.cfg.heading_scale * heading_weight
        if self.cfg.heading_gate_use:
            progress = prev_dist - dist_to_goal
            gate = progress > self.cfg.heading_gate_min_approach
            if self.cfg.heading_gate_min_speed > 0:
                speed = torch.norm(robot_vel[:, :2], dim=-1)
                gate = gate & (speed > self.cfg.heading_gate_min_speed)
            heading_reward = heading_reward * gate.float()
        rewards['heading'] = heading_reward

        # 3. 运动强度适配奖励 ★新增★
        intensity_rewards = self._compute_intensity_reward(
            intensity, prev_intensity, terrain_difficulty
        )
        if self.cfg.intensity_gate_use:
            progress = prev_dist - dist_to_goal
            gate = progress > self.cfg.intensity_gate_min_approach
            if self.cfg.intensity_gate_min_speed > 0:
                speed = torch.norm(robot_vel[:, :2], dim=-1)
                gate = gate & (speed > self.cfg.intensity_gate_min_speed)
            intensity_rewards['intensity_match'] = intensity_rewards['intensity_match'] * gate.float()
        rewards.update(intensity_rewards)

        # 4. 碰撞惩罚
        collision_reward = collision_mask.float() * self.cfg.collision_penalty
        rewards['collision'] = collision_reward

        # 5. 稳定性奖励
        stability_reward = self._compute_stability_reward(robot_quat)
        rewards['stability'] = stability_reward

        # 6. 速度奖励
        vel_towards_goal = self._compute_vel_towards_goal(robot_vel, robot_pos, goal_pos)
        velocity_reward = vel_towards_goal * self.cfg.velocity_scale
        rewards['velocity'] = velocity_reward

        # 7. 时间惩罚
        time_reward = torch.ones(num_envs, device=device) * self.cfg.time_penalty
        rewards['time'] = time_reward

        # 总奖励
        total_reward = (
            rewards['approach'] +
            rewards['reach'] +
            rewards['heading'] +
            rewards['intensity_match'] +    # ★新增★
            rewards['intensity_smooth'] +   # ★新增★
            rewards['collision'] +
            rewards['stability'] +
            rewards['velocity'] +
            rewards['time']
        )
        rewards['total'] = total_reward

        return rewards

    def _compute_intensity_reward(
        self,
        intensity: torch.Tensor,
        prev_intensity: torch.Tensor,
        terrain_difficulty: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算运动强度适配奖励 ★新增★

        原理:
        1. 计算最优强度 = 1 - 地形难度
        2. 评估当前强度与最优强度的匹配程度
        3. 惩罚强度剧烈变化
        """
        # 最优强度: 地形越简单，强度应越高
        optimal_intensity = 1.0 - terrain_difficulty

        # 强度匹配误差
        intensity_error = torch.abs(intensity - optimal_intensity)

        # 匹配奖励: 误差越小，奖励越高
        # error=0 -> reward=0.2, error=1 -> reward=-0.1
        match_reward = torch.where(
            intensity_error < 0.2,
            torch.full_like(intensity_error, self.cfg.intensity_match_bonus),
            self.cfg.intensity_match_bonus - intensity_error * 0.3
        )

        # 强度变化惩罚
        intensity_change = torch.abs(intensity - prev_intensity)
        smooth_penalty = torch.where(
            intensity_change > self.cfg.max_intensity_change,
            self.cfg.intensity_smooth_penalty * intensity_change,
            torch.zeros_like(intensity_change)
        )

        return {
            'intensity_match': match_reward,
            'intensity_smooth': smooth_penalty,
            'optimal_intensity': optimal_intensity,  # 用于日志
            'intensity_error': intensity_error,      # 用于日志
        }

    def _compute_stability_reward(self, robot_quat: torch.Tensor) -> torch.Tensor:
        """计算姿态稳定性奖励
        
        【P0.1修复】格式约定: robot_quat = [x, y, z, w] (Isaac Gym标准)
        
        【全链路一致性 - CRITICAL】:
        - HexTerrain.base_quat → [x,y,z,w]
        - HexTerrain._yaw_from_quat() → 接受 [x,y,z,w]
        - 此函数也必须使用 [x,y,z,w]
        - 禁止改回 [w,x,y,z]，否则会导致silent bug
        """
        # === P0.1: 修正四元数格式 [x,y,z,w] ===
        x, y, z, w = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]

        # 计算roll和pitch (使用正确的公式)
        roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = torch.asin(torch.clamp(2*(w*y - z*x), -1, 1))

        # 惩罚过大的倾斜
        roll_penalty = torch.where(
            torch.abs(roll) > self.cfg.roll_threshold,
            -torch.abs(roll) * self.cfg.stability_scale,
            torch.zeros_like(roll)
        )
        pitch_penalty = torch.where(
            torch.abs(pitch) > self.cfg.pitch_threshold,
            -torch.abs(pitch) * self.cfg.stability_scale,
            torch.zeros_like(pitch)
        )

        return roll_penalty + pitch_penalty

    def _compute_vel_towards_goal(
        self,
        robot_vel: torch.Tensor,
        robot_pos: torch.Tensor,
        goal_pos: torch.Tensor
    ) -> torch.Tensor:
        """计算朝向目标的速度分量"""
        to_goal = goal_pos - robot_pos[:, :2]
        to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
        to_goal_dir = to_goal / (to_goal_norm + 1e-6)

        vel_xy = robot_vel[:, :2]
        vel_towards = (vel_xy * to_goal_dir).sum(dim=-1)

        return torch.clamp(vel_towards / 0.7, -1, 1)  # 归一化

    def _quat_to_heading(self, quat: torch.Tensor) -> torch.Tensor:
        """四元数转航向角
        
        【格式约定 - CRITICAL】: quat = [x, y, z, w] (Isaac Gym 标准)
        
        【禁止修改】:
        - 不要改回 [w,x,y,z]，否则会导致silent bug
        - HexTerrain._yaw_from_quat() 使用相同格式
        """
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    @staticmethod
    def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """计算角度差 (处理wraparound)"""
        diff = a - b
        return torch.atan2(torch.sin(diff), torch.cos(diff))


class NavigationTaskManager:
    """导航任务管理器

    功能:
    1. 目标生成与管理
    2. Episode终止条件判断
    3. Curriculum Learning支持
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        map_size: Tuple[float, float] = (10.0, 10.0),
        goal_reach_threshold: float = 0.3,
        max_episode_length: int = 500,
        curriculum_enabled: bool = True,
    ):
        self.num_envs = num_envs
        self.device = device
        self.map_size = map_size
        self.goal_reach_threshold = goal_reach_threshold
        self.max_episode_length = max_episode_length
        self.curriculum_enabled = curriculum_enabled

        # 状态
        self.goal_positions = torch.zeros(num_envs, 2, device=device)
        self.episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Curriculum
        self.difficulty_level = torch.zeros(num_envs, device=device)
        self.success_history = torch.zeros(num_envs, 10, device=device)
        self.success_idx = torch.zeros(num_envs, device=device, dtype=torch.long)

    def reset_goals(
        self,
        env_ids: torch.Tensor,
        robot_positions: torch.Tensor,
        min_distance: float = 2.0,
        max_distance: float = 8.0,
    ):
        """重置目标位置"""
        num_reset = len(env_ids)

        if self.curriculum_enabled:
            # 根据difficulty调整距离
            difficulties = self.difficulty_level[env_ids]
            distances = min_distance + difficulties * (max_distance - min_distance)
        else:
            distances = torch.rand(num_reset, device=self.device) * (max_distance - min_distance) + min_distance

        # 随机方向
        angles = torch.rand(num_reset, device=self.device) * 2 * np.pi

        # 计算目标位置 (注意: robot_positions 已经是子集，形状为 (num_reset, 2 或 3))
        new_goals = torch.zeros(num_reset, 2, device=self.device)
        new_goals[:, 0] = robot_positions[:, 0] + distances * torch.cos(angles)
        new_goals[:, 1] = robot_positions[:, 1] + distances * torch.sin(angles)

        # 限制在地图范围内
        new_goals[:, 0] = torch.clamp(new_goals[:, 0], -self.map_size[0]/2, self.map_size[0]/2)
        new_goals[:, 1] = torch.clamp(new_goals[:, 1], -self.map_size[1]/2, self.map_size[1]/2)

        self.goal_positions[env_ids] = new_goals
        self.episode_lengths[env_ids] = 0

    def check_termination(
        self,
        robot_positions: torch.Tensor,
        collision_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """检查终止条件"""
        self.episode_lengths += 1

        # 到达目标
        dist_to_goal = torch.norm(robot_positions[:, :2] - self.goal_positions, dim=-1)
        reached_goal = dist_to_goal < self.goal_reach_threshold

        # 超时
        timeout = self.episode_lengths >= self.max_episode_length

        # 终止
        dones = reached_goal | timeout | collision_mask
        successes = reached_goal

        info = {
            'reached_goal': reached_goal,
            'timeout': timeout,
            'collision': collision_mask,
            'episode_length': self.episode_lengths.clone(),
        }

        return dones, successes, info

    def update_curriculum(self, env_ids: torch.Tensor, successes: torch.Tensor):
        """更新curriculum难度
        
        注意: successes 参数是全量buffer (num_envs,)，需要用 env_ids 索引
        """
        if not self.curriculum_enabled:
            return

        # 更新成功历史 (successes 是全量buffer，需要索引)
        idx = self.success_idx[env_ids] % 10
        self.success_history[env_ids, idx] = successes[env_ids].float()
        self.success_idx[env_ids] += 1

        # 计算成功率
        success_rate = self.success_history[env_ids].mean(dim=-1)

        # 调整难度
        difficulty_delta = torch.zeros_like(success_rate)
        difficulty_delta = torch.where(success_rate > 0.7, torch.tensor(0.1, device=self.device), difficulty_delta)
        difficulty_delta = torch.where(success_rate < 0.3, torch.tensor(-0.1, device=self.device), difficulty_delta)

        self.difficulty_level[env_ids] = torch.clamp(
            self.difficulty_level[env_ids] + difficulty_delta, 0.0, 1.0
        )

    def get_relative_goal(
        self, 
        robot_positions: torch.Tensor, 
        robot_headings: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """获取机器人坐标系下的相对目标位置
        
        Args:
            robot_positions: 机器人位置 (N, 2/3) 或 (len(env_ids), 2/3)
            robot_headings: 机器人朝向 (N,) 或 (len(env_ids),)
            env_ids: 可选。如果为None，认为robot_positions是全量；否则是子集
        
        Returns:
            relative_goal: 相对目标 (N, 2) 或 (len(env_ids), 2)
        """
        # 世界坐标系下的目标偏移
        if env_ids is not None:
            # 子集模式：仅计算指定 env_ids 的目标
            goal_pos = self.goal_positions[env_ids]
        else:
            # 全量模式
            goal_pos = self.goal_positions
        
        delta = goal_pos - robot_positions[:, :2]

        # 旋转到机器人坐标系
        cos_h = torch.cos(-robot_headings)
        sin_h = torch.sin(-robot_headings)

        relative_goal = torch.zeros_like(delta)
        relative_goal[:, 0] = cos_h * delta[:, 0] - sin_h * delta[:, 1]
        relative_goal[:, 1] = sin_h * delta[:, 0] + cos_h * delta[:, 1]

        return relative_goal
