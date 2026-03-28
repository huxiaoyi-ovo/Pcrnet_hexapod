"""
V3.6 地形自适应高层规划器
变更摘要:
1. 状态空间优化: 默认 state_dim=9 (匹配 V2 方案的高层抽象)，移除底层冗余关节信息。
2. 数学严谨性: 保持 TanhSquashedGaussian (Subgoal) 和 Beta (Intensity) 分布。
3. 物理安全性: 保持 Slew Rate Limiting 保护真机电机。

注意: 
训练脚本 (train_highlevel.py) 中的环境包装器必须构造对应的 9 维向量：
State = [Pos(3), LinVel(3), Heading(1), AngVel(1), Intensity(1)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
from typing import Tuple, Dict, Optional, NamedTuple
import numpy as np


class PlannerOutput(NamedTuple):
    """
    规划器前向传播输出结构
    """
    subgoal_mean: torch.Tensor      # (B, 3) Pre-tanh Gaussian Mean
    subgoal_std: torch.Tensor       # (B, 3) Gaussian Std
    intensity_alpha: torch.Tensor   # (B, 1) Beta Alpha (>0)
    intensity_beta: torch.Tensor    # (B, 1) Beta Beta (>0)
    value: torch.Tensor             # (B, 1) Value Estimate


class AffordanceCNNEncoder(nn.Module):
    """
    Affordance 地图编码器 (CNN)
    Input: (B, C, H, W) [Occupancy, Passable, LowOb]
    Output: (B, 128)
    """
    def __init__(self, in_channels: int = 2, out_features: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            # 16x16 -> 8x8
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),

            # 4x4 -> 4x4 (保留空间信息)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.fc = nn.Linear(128 * 4 * 4, out_features)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, affordance_map: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = affordance_map.shape
        # Upstream affordance maps use spatial order [x_right, y_forward],
        # i.e. height stores x and width stores y.
        coord_x = torch.linspace(-1.0, 1.0, height, device=affordance_map.device)
        coord_y = torch.linspace(-1.0, 1.0, width, device=affordance_map.device)
        xx, yy = torch.meshgrid(coord_x, coord_y, indexing="ij")
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        x = torch.cat([affordance_map, coord], dim=1)
        x = self.cnn(x)
        return self.fc(x)


class StateEncoder(nn.Module):
    """
    机器人状态编码器 (MLP)
    处理高层抽象状态 (e.g., Velocity, Orientation)
    """
    def __init__(self, state_dim: int, out_features: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, out_features),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class GoalEncoder(nn.Module):
    """目标编码器 (MLP)"""
    def __init__(self, goal_dim: int = 2, out_features: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(goal_dim, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, out_features),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return self.mlp(goal)


class TerrainAdaptivePlanner(nn.Module):
    """
    V3.6 地形自适应规划器 (Actor-Critic)
    """
    def __init__(
        self,
        affordance_channels: int = 2,
        state_dim: int = 9,     # ✅ 修正: 默认为 9 (High-Level Abstract State)
        goal_dim: int = 2,
        subgoal_dim: int = 3,
        hidden_dim: int = 256,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ):
        super().__init__()
        self.subgoal_dim = subgoal_dim
        self.min_std = min_std
        self.max_std = max_std

        # --- Encoders ---
        self.affordance_encoder = AffordanceCNNEncoder(affordance_channels, 128)
        self.state_encoder = StateEncoder(state_dim, 64)
        self.goal_encoder = GoalEncoder(goal_dim, 32)

        # Fusion: CNN(128) + State(64) + Goal(32) + Diff(1) = 225
        fusion_dim = 128 + 64 + 32 + 1 

        # --- Backbone ---
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # --- Actor Heads ---
        
        # A. Subgoal (Squashed Gaussian parameters)
        self.subgoal_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, subgoal_dim) 
        )
        self.subgoal_std_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, subgoal_dim),
            nn.Softplus()
        )

        # B. Intensity (Beta parameters)
        # Alpha, Beta 必须 > 0，加 1.0 偏移保证分布稳定
        self.intensity_alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.intensity_beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        # --- Critic Head ---
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        # 物理量缩放 (m, m, rad)
        # subgoal 范围：xy 用于线速度命令，yaw 保持相对保守
        self.register_buffer('subgoal_scale', torch.tensor([1.0, 1.0, 0.5]))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 缩小策略输出层的初始权重，增加初始熵
        nn.init.orthogonal_(self.subgoal_mean_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.intensity_alpha_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.intensity_beta_head[-2].weight, gain=0.01)

    def forward(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
    ) -> PlannerOutput:
        
        # 1. 编码与融合
        aff_feat = self.affordance_encoder(affordance_map)
        state_feat = self.state_encoder(robot_state)
        goal_feat = self.goal_encoder(goal)
        
        if terrain_difficulty.dim() == 1:
            terrain_difficulty = terrain_difficulty.unsqueeze(-1)
        
        fused = torch.cat([aff_feat, state_feat, goal_feat, terrain_difficulty], dim=-1)
        hidden = self.fusion(fused)

        # 2. 计算分布参数
        # Subgoal
        subgoal_mean = self.subgoal_mean_head(hidden)
        subgoal_std = self.subgoal_std_head(hidden)
        subgoal_std = torch.clamp(subgoal_std, self.min_std, self.max_std)

        # Intensity (Add 1.0 offset for stability)
        intensity_alpha = self.intensity_alpha_head(hidden) + 1.0
        intensity_beta = self.intensity_beta_head(hidden) + 1.0

        # Value
        value = self.value_head(hidden)

        return PlannerOutput(
            subgoal_mean=subgoal_mean,
            subgoal_std=subgoal_std,
            intensity_alpha=intensity_alpha,
            intensity_beta=intensity_beta,
            value=value
        )

    def get_action(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        采样动作 (Rollout 阶段)
        """
        out = self.forward(affordance_map, robot_state, goal, terrain_difficulty)

        if deterministic:
            # 1. Subgoal (Mean -> Tanh)
            subgoal_raw = out.subgoal_mean
            subgoal = torch.tanh(subgoal_raw) * self.subgoal_scale
            
            # 2. Intensity (Beta Mean)
            intensity = out.intensity_alpha / (out.intensity_alpha + out.intensity_beta)
            intensity = intensity.squeeze(-1)
        else:
            # 1. Subgoal (Sample Normal -> Tanh)
            subgoal_dist = Normal(out.subgoal_mean, out.subgoal_std)
            subgoal_raw = subgoal_dist.rsample() # rsample 保留梯度
            subgoal = torch.tanh(subgoal_raw) * self.subgoal_scale
            
            # 2. Intensity (Sample Beta)
            intensity_dist = Beta(out.intensity_alpha, out.intensity_beta)
            intensity = intensity_dist.sample().squeeze(-1)

        info = {
            'value': out.value,
            'subgoal_mean': out.subgoal_mean,
            'subgoal_std': out.subgoal_std,
            'intensity_alpha': out.intensity_alpha,
            'intensity_beta': out.intensity_beta
        }

        return subgoal, intensity, info

    def evaluate_actions(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        subgoal_action: torch.Tensor,
        intensity_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作 (PPO Update 阶段)
        包含严格的 Tanh Jacobian 修正
        """
        out = self.forward(affordance_map, robot_state, goal, terrain_difficulty)

        # -----------------------------------------------------------
        # 1. Subgoal LogProb (Squashed Gaussian Correction)
        # -----------------------------------------------------------
        # 反推原始的高斯分布值 (Pre-tanh action)
        subgoal_norm = subgoal_action / self.subgoal_scale
        subgoal_norm = torch.clamp(subgoal_norm, -0.999999, 0.999999)
        subgoal_raw = torch.atanh(subgoal_norm)
        
        subgoal_dist = Normal(out.subgoal_mean, out.subgoal_std)
        log_prob_raw = subgoal_dist.log_prob(subgoal_raw)
        
        # Jacobian Correction
        log_det_jacobian = 2.0 * (np.log(2.0) - subgoal_raw - F.softplus(-2.0 * subgoal_raw))
        
        subgoal_log_prob = (log_prob_raw - log_det_jacobian).sum(dim=-1)
        subgoal_entropy_raw = subgoal_dist.rsample()
        subgoal_entropy_log_prob_raw = subgoal_dist.log_prob(subgoal_entropy_raw)
        subgoal_entropy_log_det = 2.0 * (
            np.log(2.0) - subgoal_entropy_raw - F.softplus(-2.0 * subgoal_entropy_raw)
        )
        subgoal_entropy = -(subgoal_entropy_log_prob_raw - subgoal_entropy_log_det).sum(dim=-1)

        # -----------------------------------------------------------
        # 2. Intensity LogProb (Beta)
        # -----------------------------------------------------------
        intensity_dist = Beta(out.intensity_alpha, out.intensity_beta)
        
        eps = 1e-6
        intensity_clamped = torch.clamp(intensity_action, eps, 1.0 - eps)
        
        if intensity_clamped.dim() == 1:
            intensity_clamped = intensity_clamped.unsqueeze(-1)
            
        intensity_log_prob = intensity_dist.log_prob(intensity_clamped).sum(dim=-1)
        intensity_entropy = intensity_dist.entropy().sum(dim=-1)

        # 3. 合并
        total_log_prob = subgoal_log_prob + intensity_log_prob
        total_entropy = subgoal_entropy + intensity_entropy

        return total_log_prob, out.value, total_entropy, None


class LocomotionAdapter:
    """
    V3.6 运动适配器 (Expert Enhanced)
    职责: 将高层指令映射为底层控制指令
    特性: Slew Rate Limiting & Exponential Mapping
    """
    
    def __init__(
        self,
        max_linear_vel: float = 0.5,
        max_angular_vel: float = 0.8,
        min_speed_factor: float = 0.4,
        min_turn_factor: float = 0.5,
        max_intensity_change: float = 0.1, # 建议: 每0.1秒变化不超过10%
    ):
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_speed_factor = min_speed_factor
        self.min_turn_factor = min_turn_factor
        
        self.max_intensity_change = max_intensity_change
        self.last_intensity = None 

    def reset(self, num_envs, device):
        """Episode重置时调用"""
        self.last_intensity = torch.zeros(num_envs, 1, device=device)

    def convert(
        self,
        subgoal: torch.Tensor,
        intensity: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行映射与滤波
        """
        if intensity.dim() == 0: intensity = intensity.unsqueeze(0)
        if intensity.dim() == 1: intensity = intensity.unsqueeze(-1)

        # 1. 初始化
        if self.last_intensity is None or self.last_intensity.shape[0] != intensity.shape[0]:
            self.last_intensity = torch.zeros_like(intensity)

        # 2. 变化率限制 (Slew Rate Limiting)
        delta = self.max_intensity_change
        intensity_filtered = torch.clamp(
            intensity,
            self.last_intensity - delta,
            self.last_intensity + delta
        )
        intensity_filtered = torch.clamp(intensity_filtered, 0.0, 1.0)
        
        # 更新记忆
        self.last_intensity = intensity_filtered.detach()

        # 3. 非线性指数映射
        intensity_sq = torch.square(intensity_filtered)
        
        speed_f = self.min_speed_factor + (1.0 - self.min_speed_factor) * intensity_sq
        turn_f = self.min_turn_factor + (1.0 - self.min_turn_factor) * intensity_sq

        # 4. 指令生成
        cmd_vx = subgoal[:, 0:1]
        cmd_vy = subgoal[:, 1:2]
        cmd_omega = subgoal[:, 2:3]

        vx = torch.clamp(cmd_vx * speed_f, -self.max_linear_vel, self.max_linear_vel)
        vy = torch.clamp(cmd_vy * speed_f, -self.max_linear_vel, self.max_linear_vel)
        omega = torch.clamp(cmd_omega * turn_f * 1.5, -self.max_angular_vel, self.max_angular_vel)

        velocity_cmd = torch.cat([vx, vy, omega], dim=-1)

        info = {
            'speed_factor': speed_f,
            'raw_intensity': intensity,
            'filtered_intensity': intensity_filtered
        }
        
        return velocity_cmd, info

    def convert_numpy(
        self,
        subgoal: np.ndarray,
        intensity: float,
    ) -> Tuple[np.ndarray, Dict]:
        """
        NumPy版本，用于真机部署
        
        Args:
            subgoal: (3,) 子目标 [dx, dy, dyaw]
            intensity: float 运动强度 ∈ [0, 1]
        
        Returns:
            velocity_cmd: (3,) [vx, vy, omega]
            info: 调试信息
        """
        # 变化率限制
        if not hasattr(self, '_last_intensity_np'):
            self._last_intensity_np = 0.0
        
        delta = self.max_intensity_change
        intensity_filtered = np.clip(
            intensity,
            self._last_intensity_np - delta,
            self._last_intensity_np + delta
        )
        intensity_filtered = np.clip(intensity_filtered, 0.0, 1.0)
        self._last_intensity_np = intensity_filtered
        
        # 非线性映射
        intensity_sq = intensity_filtered ** 2
        speed_f = self.min_speed_factor + (1.0 - self.min_speed_factor) * intensity_sq
        turn_f = self.min_turn_factor + (1.0 - self.min_turn_factor) * intensity_sq
        
        # 速度命令
        vx = np.clip(subgoal[0] * speed_f, -self.max_linear_vel, self.max_linear_vel)
        vy = np.clip(subgoal[1] * speed_f, -self.max_linear_vel, self.max_linear_vel)
        omega = np.clip(subgoal[2] * turn_f * 1.5, -self.max_angular_vel, self.max_angular_vel)
        
        velocity_cmd = np.array([vx, vy, omega])
        
        # 运动风格判定
        if intensity_filtered < 0.33:
            style = 'Conservative'
        elif intensity_filtered < 0.67:
            style = 'Balanced'
        else:
            style = 'Aggressive'
        
        info = {
            'speed_factor': speed_f,
            'filtered_intensity': intensity_filtered,
            'style': style
        }
        
        return velocity_cmd, info


class CmdVelOutput(NamedTuple):
    cmd_mean: torch.Tensor
    cmd_std: torch.Tensor
    value: torch.Tensor


class CmdVelExpert(nn.Module):
    """
    Command-space expert policy.
    Output: cmd_vel = [vx, vy, omega] (tanh-squashed Gaussian).
    """
    def __init__(
        self,
        affordance_channels: int = 3,
        state_dim: int = 9,
        goal_dim: int = 2,
        hidden_dim: int = 256,
        min_std: float = 0.01,
        max_std: float = 1.0,
        cmd_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.affordance_encoder = AffordanceCNNEncoder(affordance_channels, 128)
        self.state_encoder = StateEncoder(state_dim, 64)
        self.goal_encoder = GoalEncoder(goal_dim, 32)
        self.critic_affordance_encoder = AffordanceCNNEncoder(affordance_channels, 128)
        self.critic_state_encoder = StateEncoder(state_dim, 64)
        self.critic_goal_encoder = GoalEncoder(goal_dim, 32)

        fusion_dim = 128 + 64 + 32 + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.critic_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.cmd_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3)
        )
        self.cmd_std_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3),
            nn.Softplus()
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        self.register_buffer("cmd_scale", torch.tensor(cmd_scale))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.cmd_mean_head[-1].weight, gain=0.01)

    @staticmethod
    def _format_difficulty(terrain_difficulty: torch.Tensor) -> torch.Tensor:
        if terrain_difficulty.dim() == 1:
            terrain_difficulty = terrain_difficulty.unsqueeze(-1)
        return terrain_difficulty

    def _encode_hidden(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        *,
        critic: bool = False,
    ) -> torch.Tensor:
        terrain_difficulty = self._format_difficulty(terrain_difficulty)
        if critic:
            aff_feat = self.critic_affordance_encoder(affordance_map)
            state_feat = self.critic_state_encoder(robot_state)
            goal_feat = self.critic_goal_encoder(goal)
            fusion = self.critic_fusion
        else:
            aff_feat = self.affordance_encoder(affordance_map)
            state_feat = self.state_encoder(robot_state)
            goal_feat = self.goal_encoder(goal)
            fusion = self.fusion
        fused = torch.cat([aff_feat, state_feat, goal_feat, terrain_difficulty], dim=-1)
        return fusion(fused)

    def forward(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> CmdVelOutput:
        actor_hidden = self._encode_hidden(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic=False,
        )
        critic_hidden = actor_hidden
        if (
            critic_affordance_map is not None
            or critic_robot_state is not None
            or critic_goal is not None
            or critic_terrain_difficulty is not None
        ):
            critic_hidden = self._encode_hidden(
                affordance_map if critic_affordance_map is None else critic_affordance_map,
                robot_state if critic_robot_state is None else critic_robot_state,
                goal if critic_goal is None else critic_goal,
                terrain_difficulty if critic_terrain_difficulty is None else critic_terrain_difficulty,
                critic=True,
            )

        cmd_mean = self.cmd_mean_head(actor_hidden)
        cmd_std = self.cmd_std_head(actor_hidden)
        cmd_std = torch.clamp(cmd_std, self.min_std, self.max_std)
        value = self.value_head(critic_hidden)

        return CmdVelOutput(cmd_mean=cmd_mean, cmd_std=cmd_std, value=value)

    def get_action(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        deterministic: bool = False,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        out = self.forward(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic_affordance_map=critic_affordance_map,
            critic_robot_state=critic_robot_state,
            critic_goal=critic_goal,
            critic_terrain_difficulty=critic_terrain_difficulty,
        )
        if deterministic:
            cmd_raw = out.cmd_mean
        else:
            cmd_dist = Normal(out.cmd_mean, out.cmd_std)
            cmd_raw = cmd_dist.rsample()
        cmd = torch.tanh(cmd_raw) * self.cmd_scale
        info = {
            "value": out.value,
            "cmd_mean": out.cmd_mean,
            "cmd_std": out.cmd_std,
        }
        return cmd, info

    def evaluate_actions(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        cmd_action: torch.Tensor,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        out = self.forward(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic_affordance_map=critic_affordance_map,
            critic_robot_state=critic_robot_state,
            critic_goal=critic_goal,
            critic_terrain_difficulty=critic_terrain_difficulty,
        )

        zero_scale_mask = self.cmd_scale.abs() <= 1e-6
        safe_cmd_scale = torch.where(zero_scale_mask, torch.ones_like(self.cmd_scale), self.cmd_scale)
        cmd_norm = cmd_action / safe_cmd_scale
        if zero_scale_mask.any():
            cmd_norm = torch.where(zero_scale_mask.view(1, -1), torch.zeros_like(cmd_norm), cmd_norm)
        cmd_norm = torch.clamp(cmd_norm, -0.999999, 0.999999)
        cmd_raw = torch.atanh(cmd_norm)

        cmd_dist = Normal(out.cmd_mean, out.cmd_std)
        log_prob_raw = cmd_dist.log_prob(cmd_raw)
        log_det_jacobian = 2.0 * (np.log(2.0) - cmd_raw - F.softplus(-2.0 * cmd_raw))
        if zero_scale_mask.any():
            active_dim_mask = (~zero_scale_mask).to(device=cmd_raw.device, dtype=cmd_raw.dtype).view(1, -1)
            log_prob_raw = log_prob_raw * active_dim_mask
            log_det_jacobian = log_det_jacobian * active_dim_mask
        cmd_log_prob = (log_prob_raw - log_det_jacobian).sum(dim=-1)
        cmd_entropy_raw = cmd_dist.rsample()
        cmd_entropy_log_prob_raw = cmd_dist.log_prob(cmd_entropy_raw)
        cmd_entropy_log_det = 2.0 * (np.log(2.0) - cmd_entropy_raw - F.softplus(-2.0 * cmd_entropy_raw))
        if zero_scale_mask.any():
            active_dim_mask = (~zero_scale_mask).to(device=cmd_entropy_raw.device, dtype=cmd_entropy_raw.dtype).view(1, -1)
            cmd_entropy_log_prob_raw = cmd_entropy_log_prob_raw * active_dim_mask
            cmd_entropy_log_det = cmd_entropy_log_det * active_dim_mask
        cmd_entropy = -(cmd_entropy_log_prob_raw - cmd_entropy_log_det).sum(dim=-1)

        return cmd_log_prob, out.value, cmd_entropy, None


class GateOutput(NamedTuple):
    y_alpha: torch.Tensor
    y_beta: torch.Tensor
    value: torch.Tensor


class GatePolicy(nn.Module):
    """
    Gate policy for MoE arbitration. Output y in [0,1] using Beta.
    """
    def __init__(
        self,
        affordance_channels: int = 3,
        state_dim: int = 9,
        goal_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.affordance_encoder = AffordanceCNNEncoder(affordance_channels, 128)
        self.state_encoder = StateEncoder(state_dim, 64)
        self.goal_encoder = GoalEncoder(goal_dim, 32)
        self.critic_affordance_encoder = AffordanceCNNEncoder(affordance_channels, 128)
        self.critic_state_encoder = StateEncoder(state_dim, 64)
        self.critic_goal_encoder = GoalEncoder(goal_dim, 32)

        fusion_dim = 128 + 64 + 32 + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.critic_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        self.y_alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.y_beta_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.y_alpha_head[-2].weight, gain=0.01)
        nn.init.orthogonal_(self.y_beta_head[-2].weight, gain=0.01)

    @staticmethod
    def _format_difficulty(terrain_difficulty: torch.Tensor) -> torch.Tensor:
        if terrain_difficulty.dim() == 1:
            terrain_difficulty = terrain_difficulty.unsqueeze(-1)
        return terrain_difficulty

    def _encode_hidden(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        *,
        critic: bool = False,
    ) -> torch.Tensor:
        terrain_difficulty = self._format_difficulty(terrain_difficulty)
        if critic:
            aff_feat = self.critic_affordance_encoder(affordance_map)
            state_feat = self.critic_state_encoder(robot_state)
            goal_feat = self.critic_goal_encoder(goal)
            fusion = self.critic_fusion
        else:
            aff_feat = self.affordance_encoder(affordance_map)
            state_feat = self.state_encoder(robot_state)
            goal_feat = self.goal_encoder(goal)
            fusion = self.fusion
        fused = torch.cat([aff_feat, state_feat, goal_feat, terrain_difficulty], dim=-1)
        return fusion(fused)

    def forward(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> GateOutput:
        actor_hidden = self._encode_hidden(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic=False,
        )
        critic_hidden = actor_hidden
        if (
            critic_affordance_map is not None
            or critic_robot_state is not None
            or critic_goal is not None
            or critic_terrain_difficulty is not None
        ):
            critic_hidden = self._encode_hidden(
                affordance_map if critic_affordance_map is None else critic_affordance_map,
                robot_state if critic_robot_state is None else critic_robot_state,
                goal if critic_goal is None else critic_goal,
                terrain_difficulty if critic_terrain_difficulty is None else critic_terrain_difficulty,
                critic=True,
            )

        y_alpha = self.y_alpha_head(actor_hidden) + 1.0
        y_beta = self.y_beta_head(actor_hidden) + 1.0
        value = self.value_head(critic_hidden)
        return GateOutput(y_alpha=y_alpha, y_beta=y_beta, value=value)

    def get_action(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        deterministic: bool = False,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        out = self.forward(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic_affordance_map=critic_affordance_map,
            critic_robot_state=critic_robot_state,
            critic_goal=critic_goal,
            critic_terrain_difficulty=critic_terrain_difficulty,
        )
        if deterministic:
            y = out.y_alpha / (out.y_alpha + out.y_beta)
            y = y.squeeze(-1)
        else:
            y_dist = Beta(out.y_alpha, out.y_beta)
            y = y_dist.sample().squeeze(-1)
        info = {
            "value": out.value,
            "y_alpha": out.y_alpha,
            "y_beta": out.y_beta,
        }
        return y, info

    def evaluate_actions(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        y_action: torch.Tensor,
        critic_affordance_map: Optional[torch.Tensor] = None,
        critic_robot_state: Optional[torch.Tensor] = None,
        critic_goal: Optional[torch.Tensor] = None,
        critic_terrain_difficulty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        out = self.forward(
            affordance_map,
            robot_state,
            goal,
            terrain_difficulty,
            critic_affordance_map=critic_affordance_map,
            critic_robot_state=critic_robot_state,
            critic_goal=critic_goal,
            critic_terrain_difficulty=critic_terrain_difficulty,
        )
        y_dist = Beta(out.y_alpha, out.y_beta)
        eps = 1e-6
        y_clamped = torch.clamp(y_action, eps, 1.0 - eps)
        if y_clamped.dim() == 1:
            y_clamped = y_clamped.unsqueeze(-1)
        y_log_prob = y_dist.log_prob(y_clamped).sum(dim=-1)
        y_entropy = y_dist.entropy().sum(dim=-1)
        return y_log_prob, out.value, y_entropy, None


class CommandPostProcessor:
    """
    Post-processor for cmd_vel:
    1) clamp to limits
    2) slew-rate limiting
    3) risk-based scaling (clearance)
    """
    def __init__(
        self,
        max_cmd: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        max_delta: Tuple[float, float, float] = (0.2, 0.2, 0.4),
        safe_distance: float = 0.25,
        free_distance: float = 0.6,
        enable_risk_scale: bool = True,
    ):
        # Keep float32 by default; in process() we will align dtype with cmd.dtype.
        self.max_cmd = torch.tensor(max_cmd, dtype=torch.float32)
        self.max_delta = torch.tensor(max_delta, dtype=torch.float32)
        self.safe_distance = safe_distance
        self.free_distance = max(free_distance, safe_distance + 1e-6)
        self.enable_risk_scale = enable_risk_scale
        self.last_cmd = None
        # Beta-controlled "constraint family" endpoints (V7 defaults).
        # Only used when process(..., beta=...) is provided; otherwise old behavior is preserved.
        self._beta_safe_dist = (0.35, 1.00)
        self._beta_max_lin = (1.00, 0.35)
        self._beta_max_ang = (1.50, 0.50)
        self._beta_max_delta_lin = (0.15, 0.05)
        self._beta_max_delta_ang = (0.30, 0.10)
        self._beta_risk_gain = (1.0, 3.0)
        self._beta_free_margin = 0.25

    def reset(self, num_envs: int, device: torch.device):
        self.last_cmd = torch.zeros(num_envs, 3, device=device)

    def _normalize_beta(self, beta: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        beta_t = beta
        if not torch.is_tensor(beta_t):
            beta_t = torch.tensor(beta_t, device=device, dtype=dtype)
        beta_t = beta_t.to(device=device, dtype=dtype)
        if beta_t.dim() == 0:
            beta_t = beta_t.view(1)
        return torch.clamp(beta_t, 0.0, 1.0)

    def _compute_beta_params(
        self,
        beta_t: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict:
        safe0, safe1 = self._beta_safe_dist
        v_fast, v_safe = self._beta_max_lin
        w_fast, w_safe = self._beta_max_ang
        dv_fast, dv_safe = self._beta_max_delta_lin
        dw_fast, dw_safe = self._beta_max_delta_ang
        g0, g1 = self._beta_risk_gain

        safe_distance = (safe0 + (safe1 - safe0) * beta_t).to(device=device, dtype=dtype)
        free_distance = safe_distance + float(self._beta_free_margin)

        vmax = (v_fast + (v_safe - v_fast) * beta_t).to(device=device, dtype=dtype)
        wmax = (w_fast + (w_safe - w_fast) * beta_t).to(device=device, dtype=dtype)
        max_cmd = torch.stack([vmax, vmax, wmax], dim=-1)

        dvmax = (dv_fast + (dv_safe - dv_fast) * beta_t).to(device=device, dtype=dtype)
        dwmax = (dw_fast + (dw_safe - dw_fast) * beta_t).to(device=device, dtype=dtype)
        max_delta = torch.stack([dvmax, dvmax, dwmax], dim=-1)

        risk_clamp_gain = (g0 + (g1 - g0) * beta_t).to(device=device, dtype=dtype)

        return {
            "safe_distance": safe_distance,
            "free_distance": free_distance,
            "max_cmd": max_cmd,
            "max_delta": max_delta,
            "risk_clamp_gain": risk_clamp_gain,
        }

    def get_effective_params(self, beta: torch.Tensor) -> Dict:
        """
        Return beta-adjusted constraint parameters without mutating internal state.

        beta=0 -> fast/aggressive, beta=1 -> safe/conservative.
        """
        beta_t = self._normalize_beta(beta, device=torch.device("cpu"), dtype=torch.float32)
        return self._compute_beta_params(beta_t, device=beta_t.device, dtype=beta_t.dtype)

    def get_effective_safe_distance(self, beta: torch.Tensor) -> torch.Tensor:
        return self.get_effective_params(beta)["safe_distance"]

    def preview_cmd_before_risk(
        self,
        cmd: torch.Tensor,
        beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cmd.dim() == 1:
            cmd = cmd.unsqueeze(0)

        device = cmd.device
        max_cmd = self.max_cmd.to(device=device, dtype=cmd.dtype)
        max_delta = self.max_delta.to(device=device, dtype=cmd.dtype)

        if beta is not None:
            beta_t = self._normalize_beta(beta, device=device, dtype=cmd.dtype)
            beta_params = self._compute_beta_params(beta_t, device=device, dtype=cmd.dtype)
            max_cmd = beta_params["max_cmd"]
            max_delta = beta_params["max_delta"]

        cmd_clamped = torch.clamp(cmd, -max_cmd, max_cmd)
        if self.last_cmd is None or self.last_cmd.shape != cmd_clamped.shape:
            last_cmd = torch.zeros_like(cmd_clamped)
        else:
            last_cmd = self.last_cmd.to(device=device, dtype=cmd.dtype)
        delta = torch.clamp(cmd_clamped - last_cmd, -max_delta, max_delta)
        return last_cmd + delta

    def process(
        self,
        cmd: torch.Tensor,
        clearance: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if cmd.dim() == 1:
            cmd = cmd.unsqueeze(0)

        device = cmd.device
        max_cmd = self.max_cmd.to(device=device, dtype=cmd.dtype)
        max_delta = self.max_delta.to(device=device, dtype=cmd.dtype)
        safe_distance = self.safe_distance
        free_distance = self.free_distance
        risk_clamp_gain = None

        # Beta override: beta=0 -> fast/aggressive, beta=1 -> safe/conservative.
        if beta is not None:
            beta_t = self._normalize_beta(beta, device=device, dtype=cmd.dtype)
            beta_params = self._compute_beta_params(beta_t, device=device, dtype=cmd.dtype)
            safe_distance = beta_params["safe_distance"]
            free_distance = beta_params["free_distance"]
            max_cmd = beta_params["max_cmd"]
            max_delta = beta_params["max_delta"]
            risk_clamp_gain = beta_params["risk_clamp_gain"]

        # 1) Clamp to limits
        cmd_clamped = torch.clamp(cmd, -max_cmd, max_cmd)

        # 2) Slew-rate limiting
        if self.last_cmd is None or self.last_cmd.shape != cmd_clamped.shape:
            self.last_cmd = torch.zeros_like(cmd_clamped)
        delta = torch.clamp(cmd_clamped - self.last_cmd, -max_delta, max_delta)
        cmd_slew_pre_scale = self.last_cmd + delta
        cmd_exec = cmd_slew_pre_scale

        # 3) Risk-based scaling
        scale = None
        if clearance is not None and self.enable_risk_scale:
            clearance = clearance.to(device)
            safe = safe_distance
            free = free_distance
            if torch.is_tensor(safe):
                safe = safe.to(device=device, dtype=clearance.dtype)
            else:
                safe = torch.tensor(float(safe), device=device, dtype=clearance.dtype)
            if torch.is_tensor(free):
                free = free.to(device=device, dtype=clearance.dtype)
            else:
                free = torch.tensor(float(free), device=device, dtype=clearance.dtype)
            free = torch.maximum(free, safe + 1e-6)
            scale = torch.clamp(
                (clearance - safe) / (free - safe),
                0.0,
                1.0,
            )
            if risk_clamp_gain is None:
                cmd_exec = cmd_exec * scale.unsqueeze(-1)
            else:
                gain = risk_clamp_gain
                if gain.dim() == 1:
                    gain = gain.view(-1, 1)
                cmd_exec = cmd_exec * torch.pow(scale.unsqueeze(-1), gain)
        self.last_cmd = cmd_exec.detach()

        info = {
            "cmd_raw": cmd,
            "cmd_clamped": cmd_clamped,
            "cmd_slew_pre_scale": cmd_slew_pre_scale,
            "cmd_slew": cmd_exec,
            "risk_scale": scale,
        }
        if beta is not None:
            info["beta"] = beta_t
            info["safe_distance"] = safe_distance
            info["free_distance"] = free_distance
            info["max_cmd"] = max_cmd
            info["max_delta"] = max_delta
            if risk_clamp_gain is not None:
                info["risk_clamp_gain"] = risk_clamp_gain
        return cmd_exec, info

    def reset_numpy(self):
        """重置NumPy状态（真机部署时每个Episode开始调用）"""
        self.last_cmd = None


def test_planner_v36():
    """
    V3.6 高层规划器完整测试函数
    验证前向传播、动作采样、动作评估、适配器的正确性
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Testing TerrainAdaptivePlanner V3.6 on {device}")
    print(f"{'='*60}")

    # 创建模型
    planner = TerrainAdaptivePlanner(
        affordance_channels=2,
        state_dim=9,
        goal_dim=2,
        subgoal_dim=3,
        hidden_dim=256
    ).to(device)

    # 测试输入
    batch_size = 32
    affordance_map = torch.rand(batch_size, 2, 16, 16, device=device)
    robot_state = torch.randn(batch_size, 9, device=device)
    goal = torch.randn(batch_size, 2, device=device)
    terrain_difficulty = torch.rand(batch_size, device=device)

    print("\n[1] Input Shapes:")
    print(f"    Affordance map: {affordance_map.shape}")
    print(f"    Robot state:    {robot_state.shape}")
    print(f"    Goal:           {goal.shape}")
    print(f"    Terrain diff:   {terrain_difficulty.shape}")

    # 前向传播
    output = planner(affordance_map, robot_state, goal, terrain_difficulty)
    print("\n[2] Forward Output Shapes:")
    print(f"    Subgoal mean:    {output.subgoal_mean.shape}")
    print(f"    Subgoal std:     {output.subgoal_std.shape}")
    print(f"    Intensity alpha: {output.intensity_alpha.shape}")
    print(f"    Intensity beta:  {output.intensity_beta.shape}")
    print(f"    Value:           {output.value.shape}")

    # 动作采样 (随机)
    subgoal, intensity, info = planner.get_action(
        affordance_map, robot_state, goal, terrain_difficulty, deterministic=False
    )
    print("\n[3] Sampled Actions (Stochastic):")
    print(f"    Subgoal sample:   {subgoal[0].detach().cpu().numpy()}")
    print(f"    Intensity sample: {intensity[:5].detach().cpu().numpy()}")
    print(f"    Intensity range:  [{intensity.min():.3f}, {intensity.max():.3f}]")

    # 动作采样 (确定性)
    subgoal_det, intensity_det, _ = planner.get_action(
        affordance_map, robot_state, goal, terrain_difficulty, deterministic=True
    )
    print("\n[4] Sampled Actions (Deterministic):")
    print(f"    Subgoal mean:     {subgoal_det[0].detach().cpu().numpy()}")
    print(f"    Intensity mean:   {intensity_det[:5].detach().cpu().numpy()}")

    # 动作评估
    log_prob, value, entropy, _ = planner.evaluate_actions(
        affordance_map, robot_state, goal, terrain_difficulty,
        subgoal, intensity
    )
    print("\n[5] Action Evaluation:")
    print(f"    Log prob mean:    {log_prob.mean().item():.4f}")
    print(f"    Log prob std:     {log_prob.std().item():.4f}")
    print(f"    Entropy mean:     {entropy.mean().item():.4f}")
    print(f"    Value mean:       {value.mean().item():.4f}")

    # 检查数值稳定性
    assert not torch.isnan(log_prob).any(), "NaN in log_prob!"
    assert not torch.isnan(entropy).any(), "NaN in entropy!"
    assert not torch.isinf(log_prob).any(), "Inf in log_prob!"
    print("\n[6] Numerical Stability: PASSED")

    # LocomotionAdapter 测试
    adapter = LocomotionAdapter()
    adapter.reset(batch_size, device)
    
    velocity_cmd, adapt_info = adapter.convert(subgoal, intensity)
    print("\n[7] LocomotionAdapter (Torch):")
    print(f"    Velocity cmd shape: {velocity_cmd.shape}")
    print(f"    Speed factors:      {adapt_info['speed_factor'][:3].squeeze().cpu().numpy()}")
    print(f"    Filtered intensity: {adapt_info['filtered_intensity'][:3].squeeze().cpu().numpy()}")

    # 测试变化率限制
    print("\n[8] Slew Rate Limiting Test:")
    sudden_intensity = torch.ones(batch_size, device=device)
    _, info2 = adapter.convert(subgoal, sudden_intensity)
    max_change = (info2['filtered_intensity'] - adapt_info['filtered_intensity']).abs().max().item()
    print(f"    Max intensity change: {max_change:.4f} (limit: {adapter.max_intensity_change})")
    assert max_change <= adapter.max_intensity_change + 1e-6, "Rate limiting failed!"
    print("    Rate limiting: PASSED")

    # NumPy版本测试 (真机部署)
    print("\n[9] NumPy Deployment Test:")
    adapter.reset_numpy()
    subgoal_np = np.array([0.3, -0.1, 0.2])
    intensity_np = 0.7
    vel_np, info_np = adapter.convert_numpy(subgoal_np, intensity_np)
    print(f"    Velocity cmd: {vel_np}")
    print(f"    Style:        {info_np['style']}")
    
    # 测试NumPy变化率限制
    vel_np2, info_np2 = adapter.convert_numpy(subgoal_np, 0.0)  # 突然变为0
    intensity_change = abs(info_np2['filtered_intensity'] - info_np['filtered_intensity'])
    print(f"    NumPy rate limit test: change={intensity_change:.3f} (limit={adapter.max_intensity_change})")
    assert intensity_change <= adapter.max_intensity_change + 1e-6, "NumPy rate limiting failed!"
    print("    NumPy rate limiting: PASSED")

    # 参数统计
    num_params = sum(p.numel() for p in planner.parameters())
    trainable_params = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    print(f"\n[10] Model Statistics:")
    print(f"    Total parameters:     {num_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # 模型保存/加载测试
    print("\n[11] Save/Load Test:")
    state_dict = planner.state_dict()
    planner2 = TerrainAdaptivePlanner(state_dim=9).to(device)
    planner2.load_state_dict(state_dict)
    
    # 验证加载后输出一致
    with torch.no_grad():
        out1 = planner(affordance_map, robot_state, goal, terrain_difficulty)
        out2 = planner2(affordance_map, robot_state, goal, terrain_difficulty)
        diff = (out1.value - out2.value).abs().max().item()
    print(f"    Value diff after reload: {diff:.6f}")
    assert diff < 1e-5, "Save/Load failed!"
    print("    Save/Load: PASSED")

    print(f"\n{'='*60}")
    print("All V3.6 Tests PASSED!")
    print(f"{'='*60}\n")
    
    return planner


if __name__ == "__main__":
    test_planner_v36()
