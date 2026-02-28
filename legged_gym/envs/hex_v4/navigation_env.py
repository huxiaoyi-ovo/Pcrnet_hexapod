"""
navigation_env.py - 导航任务环境与奖励函数 V5

改动说明:
- 退役 intensity 速度缩放语义，改为 MoE 门控 y
- 新增: risk barrier 连续风险成本
- 新增: gate smooth 惩罚，限制门控抖动
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class NavigationRewardConfig:
    """导航奖励配置 V5"""

    # 目标接近奖励
    goal_approach_scale: float = 2.0      # 接近目标的奖励系数
    goal_reach_bonus: float = 10.0        # 到达目标的额外奖励
    goal_reach_threshold: float = 0.1     # 到达判定阈值 (m)

    # Follow-mode (moving target) reward: track a desired distance instead of "reach goal".
    follow_enable: bool = False
    follow_distance_desired: float = 1.0
    follow_distance_sigma: float = 0.25
    follow_distance_scale: float = 6.0
    follow_band_scale: float = 1.0  # scale for the Gaussian band reward term
    follow_outside_penalty: float = -1.0  # extra penalty when far outside follow band

    # 碰撞惩罚
    collision_penalty: float = -10.0

    # 朝向奖励
    heading_scale: float = 0.5            # 朝向目标的奖励
    heading_offset_rad: float = 0.0       # 朝向参考偏移 (用于前向轴修正)
    heading_use_difficulty_gate: bool = False  # 根据地形难度弱化朝向奖励
    heading_min_weight: float = 0.2       # 朝向奖励最小权重 (仅门控开启时生效)
    heading_gate_use: bool = False        # 速度/进度门控
    heading_gate_min_speed: float = 0.05  # 低速门控阈值
    heading_gate_min_approach: float = 0.0  # 进度门控阈值

    # 可通行区域引导
    passable_align_scale: float = 0.0
    passable_occ_ratio_low: float = 0.2
    passable_occ_ratio_high: float = 0.6
    passable_sector_deg: float = 60.0
    crossable_align_scale: float = 0.0
    crossable_gate_min_speed: float = 0.05

    # ★门控平滑与风险成本（MoE）★
    gate_smooth_penalty: float = -0.05     # 门控变化惩罚
    gate_max_change: float = 0.2           # 每步门控变化阈值
    risk_barrier_scale: float = -0.5       # 连续风险成本系数
    risk_barrier_safe: float = 0.25        # 安全距离阈值 (m)
    risk_barrier_free: float = 0.6         # 认为安全的距离 (m)
    risk_barrier_tau: float = 0.1          # 风险成本斜率

    # 时间惩罚
    time_penalty: float = -0.01           # 每步时间惩罚

    # 稳定性奖励
    stability_scale: float = 0.1
    pitch_threshold: float = 0.3          # rad, 约17度
    roll_threshold: float = 0.3

    # 速度奖励
    velocity_scale: float = 0.1
    backward_scale: float = 0.0
    turn_penalty_scale: float = 0.0


class NavigationRewardFunction:
    """导航任务奖励函数 V5 ★核心设计★

    奖励结构:
    1. 主奖励 (目标导向):
       - 接近目标: r = scale × (d_{t-1} - d_t)
       - 到达奖励: r = bonus if d < threshold

    2. MoE 门控:
       - gate smooth 惩罚 (限制 y 抖动)

    3. 安全性:
       - risk barrier 连续风险成本
       - 碰撞惩罚
       - 姿态稳定性

    4. 效率:
       - 时间惩罚
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
        terrain_difficulty: torch.Tensor,  # (N,) 地形难度 ★修改★
        collision_mask: torch.Tensor,      # (N,) 是否碰撞
        clearance: Optional[torch.Tensor] = None,  # (N,) 前方最小障碍距离
        cmd_xy: Optional[torch.Tensor] = None,  # (N, 2) 实际执行的平移指令
        passable_dir: Optional[torch.Tensor] = None,  # (N, 2) 可通行方向
        passable_gate: Optional[torch.Tensor] = None,  # (N,) 可通行门控
        crossable_dir: Optional[torch.Tensor] = None,  # (N, 2) 低障中心方向
        crossable_gate: Optional[torch.Tensor] = None,  # (N,) 低障门控
        cmd_omega: Optional[torch.Tensor] = None,  # (N,) 实际执行角速度命令
        gate_y: Optional[torch.Tensor] = None,  # (N,) 门控权重
        prev_gate_y: Optional[torch.Tensor] = None,  # (N,) 上一门控
        intensity: Optional[torch.Tensor] = None,       # (N,) 兼容字段（将被 gate_y 替代）
        prev_intensity: Optional[torch.Tensor] = None,  # (N,) 兼容字段
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

        if self.cfg.follow_enable:
            # Follow-mode: reward reduction of distance error w.r.t desired following distance.
            desired = float(self.cfg.follow_distance_desired)
            sigma = max(float(self.cfg.follow_distance_sigma), 1e-3)
            err = torch.abs(dist_to_goal - desired)
            prev_err = torch.abs(prev_dist - desired)
            approach_reward = (prev_err - err) * float(self.cfg.follow_distance_scale)
            # Encourage staying near the band center.
            band_scale = float(getattr(self.cfg, "follow_band_scale", 1.0))
            band_reward = band_scale * torch.exp(-0.5 * (err / sigma) ** 2)
            rewards['approach'] = approach_reward + band_reward
            rewards['reach'] = torch.zeros_like(dist_to_goal)
        else:
            # 进度奖励: 距离减少则正奖励
            approach_reward = (prev_dist - dist_to_goal) * self.cfg.goal_approach_scale
            rewards['approach'] = approach_reward

            # 到达目标奖励
            reached = dist_to_goal < self.cfg.goal_reach_threshold
            reach_reward = reached.float() * self.cfg.goal_reach_bonus
            rewards['reach'] = reach_reward

        # 2. 朝向奖励
        heading = self._quat_to_heading(robot_quat)
        if self.cfg.heading_offset_rad != 0.0:
            # Align heading reference (e.g., +Y forward) without changing goal frame.
            heading = heading + self.cfg.heading_offset_rad
            heading = torch.atan2(torch.sin(heading), torch.cos(heading))
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
        heading_reward = (torch.cos(heading_error) - 0.15) * self.cfg.heading_scale * heading_weight
        if self.cfg.heading_gate_use:
            progress = prev_dist - dist_to_goal
            gate = progress > self.cfg.heading_gate_min_approach
            if self.cfg.heading_gate_min_speed > 0:
                speed = torch.norm(robot_vel[:, :2], dim=-1)
                gate = gate & (speed > self.cfg.heading_gate_min_speed)
            heading_reward = heading_reward * gate.float()
        rewards['heading'] = heading_reward

        # 2.5 可通行区域引导奖励
        passable_align = torch.zeros(num_envs, device=device)
        if cmd_xy is not None and passable_dir is not None and self.cfg.passable_align_scale != 0.0:
            cmd_norm = torch.norm(cmd_xy, dim=-1, keepdim=True)
            pass_norm = torch.norm(passable_dir, dim=-1, keepdim=True)
            cmd_unit = cmd_xy / (cmd_norm + 1e-6)
            pass_unit = passable_dir / (pass_norm + 1e-6)
            align = torch.sum(cmd_unit * pass_unit, dim=-1)
            gate = passable_gate if passable_gate is not None else torch.ones_like(align)
            gate = gate * (cmd_norm.squeeze(-1) > 1e-2).float()
            passable_align = align * gate * self.cfg.passable_align_scale
        rewards['passable_align'] = passable_align
        if passable_gate is not None:
            rewards['passable_gate'] = passable_gate

        crossable_align = torch.zeros(num_envs, device=device)
        if cmd_xy is not None and crossable_dir is not None and self.cfg.crossable_align_scale != 0.0:
            cmd_norm = torch.norm(cmd_xy, dim=-1, keepdim=True)
            cross_norm = torch.norm(crossable_dir, dim=-1, keepdim=True)
            cmd_unit = cmd_xy / (cmd_norm + 1e-6)
            cross_unit = crossable_dir / (cross_norm + 1e-6)
            align = torch.sum(cmd_unit * cross_unit, dim=-1)
            gate = crossable_gate if crossable_gate is not None else torch.ones_like(align)
            gate = gate * (cmd_norm.squeeze(-1) > self.cfg.crossable_gate_min_speed).float()
            roll, pitch = self._quat_to_rp(robot_quat)
            pose_gate = (torch.abs(roll) < self.cfg.roll_threshold) & (torch.abs(pitch) < self.cfg.pitch_threshold)
            crossable_align = align * gate * pose_gate.float() * self.cfg.crossable_align_scale
        rewards['crossable_align'] = crossable_align
        if crossable_gate is not None:
            rewards['crossable_gate'] = crossable_gate

        # 3. Gate smooth + Risk barrier
        if gate_y is None:
            gate_y = intensity if intensity is not None else torch.zeros(num_envs, device=device)
        if prev_gate_y is None:
            prev_gate_y = prev_intensity if prev_intensity is not None else gate_y
        gate_change = torch.abs(gate_y - prev_gate_y)
        gate_smooth = torch.where(
            gate_change > self.cfg.gate_max_change,
            self.cfg.gate_smooth_penalty * gate_change,
            torch.zeros_like(gate_change),
        )
        rewards['gate_smooth'] = gate_smooth

        risk_barrier = torch.zeros(num_envs, device=device)
        risk_scale = None
        if clearance is not None and self.cfg.risk_barrier_scale != 0.0:
            safe = self.cfg.risk_barrier_safe
            free = max(self.cfg.risk_barrier_free, safe + 1e-6)
            tau = max(self.cfg.risk_barrier_tau, 1e-3)
            risk = torch.exp(-(clearance - safe) / tau)
            risk_barrier = self.cfg.risk_barrier_scale * risk
            risk_scale = torch.clamp((clearance - safe) / (free - safe), 0.0, 1.0)
        rewards['risk_barrier'] = risk_barrier
        if risk_scale is not None:
            rewards['risk_scale'] = risk_scale

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

        # 6.5 后退惩罚（朝目标方向速度为负时惩罚）
        backward_penalty = torch.zeros(num_envs, device=device)
        if self.cfg.backward_scale != 0.0:
            backward_speed = torch.clamp(-vel_towards_goal, min=0.0)
            backward_penalty = -backward_speed * self.cfg.backward_scale
        rewards['backward'] = backward_penalty

        # 6.6 无意义转向惩罚（已基本对准目标时，仍给大角速度命令）
        turn_penalty = torch.zeros(num_envs, device=device)
        if self.cfg.turn_penalty_scale != 0.0:
            if cmd_omega is None:
                omega_cmd = torch.zeros(num_envs, device=device)
            else:
                omega_cmd = cmd_omega.view(-1)
            aligned = torch.abs(heading_error) < 0.2  # about 11.5 deg
            omega_excess = torch.clamp(torch.abs(omega_cmd) - 0.2, min=0.0)
            turn_penalty = -omega_excess * aligned.float() * self.cfg.turn_penalty_scale
        rewards['turn_penalty'] = turn_penalty

        # 7. 时间惩罚
        time_reward = torch.ones(num_envs, device=device) * self.cfg.time_penalty
        rewards['time'] = time_reward

        # 总奖励
        total_reward = (
            rewards['approach'] +
            rewards['reach'] +
            rewards['heading'] +
            rewards['passable_align'] +
            rewards['crossable_align'] +
            rewards['gate_smooth'] +
            rewards['risk_barrier'] +
            rewards['collision'] +
            rewards['stability'] +
            rewards['velocity'] +
            rewards['backward'] +
            rewards['turn_penalty'] +
            rewards['time']
        )
        rewards['total'] = total_reward

        return rewards

    def _compute_stability_reward(self, robot_quat: torch.Tensor) -> torch.Tensor:
        """计算姿态稳定性奖励
        
        【P0.1修复】格式约定: robot_quat = [x, y, z, w] (Isaac Gym标准)
        
        【全链路一致性 - CRITICAL】:
        - HexGround.base_quat → [x,y,z,w]
        - HexGround._yaw_from_quat() → 接受 [x,y,z,w]
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
        - HexGround._yaw_from_quat() 使用相同格式
        """
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    @staticmethod
    def _quat_to_rp(quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """四元数转 roll/pitch (quat=[x,y,z,w])"""
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1))
        return roll, pitch

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
        goal_reach_threshold: float = 0.1,
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
