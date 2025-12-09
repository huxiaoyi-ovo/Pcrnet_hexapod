"""
hierarchical_ppo.py - 分层PPO算法 V2

改动说明 (相比V1):
- 移除: 离散步态动作处理 (Categorical分布)
- 修改: 双连续动作空间 (两个Gaussian分布)
- 简化: 单一熵系数 (无需区分连续/离散)

与V1的关键区别:
- V1: subgoal (连续) + gait (离散Categorical)
- V2: subgoal (连续) + intensity (连续Gaussian)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class HierarchicalPPOConfig:
    """分层PPO配置 V2"""

    # 训练参数
    num_learning_epochs: int = 5       # 每次更新的epoch数
    num_mini_batches: int = 4          # mini-batch数量

    # 学习率
    learning_rate: float = 3e-4

    # PPO参数
    clip_param: float = 0.2            # PPO clip参数
    value_loss_coef: float = 0.5       # 价值损失系数
    entropy_coef: float = 0.01         # 熵系数 ★简化: 单一系数★
    max_grad_norm: float = 1.0         # 梯度裁剪

    # GAE参数
    gamma: float = 0.99                # 折扣因子
    lam: float = 0.95                  # GAE lambda

    # 其他
    normalize_advantage: bool = True
    desired_kl: float = 0.01           # 目标KL散度


class RolloutStorage:
    """经验回放存储 V2

    存储高层规划器的rollout数据

    改动:
    - 移除: gaits, gait_log_probs
    - 新增: intensities, intensity_log_probs
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions: int,
        affordance_shape: Tuple[int, ...],
        state_dim: int,
        goal_dim: int,
        subgoal_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_transitions = num_transitions
        self.device = device

        # Observations
        self.affordance_maps = torch.zeros(
            num_transitions, num_envs, *affordance_shape, device=device
        )
        self.robot_states = torch.zeros(
            num_transitions, num_envs, state_dim, device=device
        )
        self.goals = torch.zeros(
            num_transitions, num_envs, goal_dim, device=device
        )
        self.terrain_difficulties = torch.zeros(
            num_transitions, num_envs, device=device
        )  # ★修改★

        # Actions
        self.subgoals = torch.zeros(
            num_transitions, num_envs, subgoal_dim, device=device
        )
        self.intensities = torch.zeros(
            num_transitions, num_envs, device=device
        )  # ★修改: gaits -> intensities★

        # Log probs
        self.subgoal_log_probs = torch.zeros(
            num_transitions, num_envs, device=device
        )
        self.intensity_log_probs = torch.zeros(
            num_transitions, num_envs, device=device
        )  # ★修改★

        # Values and rewards
        self.values = torch.zeros(num_transitions, num_envs, device=device)
        self.rewards = torch.zeros(num_transitions, num_envs, device=device)
        self.dones = torch.zeros(num_transitions, num_envs, device=device)

        # Computed quantities
        self.advantages = torch.zeros(num_transitions, num_envs, device=device)
        self.returns = torch.zeros(num_transitions, num_envs, device=device)

        self.step = 0

    def add_transition(
        self,
        affordance_map: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
        terrain_difficulty: torch.Tensor,
        subgoal: torch.Tensor,
        intensity: torch.Tensor,              # ★修改★
        subgoal_log_prob: torch.Tensor,
        intensity_log_prob: torch.Tensor,     # ★修改★
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        """添加一个transition"""
        self.affordance_maps[self.step] = affordance_map
        self.robot_states[self.step] = robot_state
        self.goals[self.step] = goal
        self.terrain_difficulties[self.step] = terrain_difficulty
        self.subgoals[self.step] = subgoal
        self.intensities[self.step] = intensity
        self.subgoal_log_probs[self.step] = subgoal_log_prob
        self.intensity_log_probs[self.step] = intensity_log_prob
        self.values[self.step] = value.squeeze(-1)
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()

        self.step += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float):
        """计算GAE和returns"""
        last_gae = 0
        for step in reversed(range(self.num_transitions)):
            if step == self.num_transitions - 1:
                next_value = last_value.squeeze(-1)
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            # TD error
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            # GAE
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, num_mini_batches: int):
        """生成mini-batch"""
        total_samples = self.num_envs * self.num_transitions
        batch_size = total_samples // num_mini_batches

        # Flatten all data
        affordance_flat = self.affordance_maps.reshape(-1, *self.affordance_maps.shape[2:])
        state_flat = self.robot_states.reshape(-1, self.robot_states.shape[-1])
        goal_flat = self.goals.reshape(-1, self.goals.shape[-1])
        terrain_flat = self.terrain_difficulties.reshape(-1)
        subgoal_flat = self.subgoals.reshape(-1, self.subgoals.shape[-1])
        intensity_flat = self.intensities.reshape(-1)
        subgoal_lp_flat = self.subgoal_log_probs.reshape(-1)
        intensity_lp_flat = self.intensity_log_probs.reshape(-1)
        value_flat = self.values.reshape(-1)
        advantage_flat = self.advantages.reshape(-1)
        return_flat = self.returns.reshape(-1)

        # Shuffle
        indices = torch.randperm(total_samples, device=self.device)

        for start in range(0, total_samples, batch_size):
            end = min(start + batch_size, total_samples)
            batch_idx = indices[start:end]

            yield {
                'affordance_map': affordance_flat[batch_idx],
                'robot_state': state_flat[batch_idx],
                'goal': goal_flat[batch_idx],
                'terrain_difficulty': terrain_flat[batch_idx],
                'subgoal': subgoal_flat[batch_idx],
                'intensity': intensity_flat[batch_idx],
                'old_subgoal_log_prob': subgoal_lp_flat[batch_idx],
                'old_intensity_log_prob': intensity_lp_flat[batch_idx],
                'old_value': value_flat[batch_idx],
                'advantage': advantage_flat[batch_idx],
                'return': return_flat[batch_idx],
            }

    def clear(self):
        """清空存储"""
        self.step = 0


class HierarchicalPPO:
    """分层PPO训练器 V2

    关键特性:
    1. 双连续动作空间 (subgoal + intensity)
    2. 两个动作都使用Gaussian分布
    3. 统一的熵系数

    与V1的区别:
    - V1: 混合动作空间 (连续 + 离散)
    - V2: 纯连续动作空间 (连续 + 连续)
    """

    def __init__(
        self,
        planner: nn.Module,
        cfg: HierarchicalPPOConfig = None,
        device: torch.device = None,
    ):
        self.planner = planner
        self.cfg = cfg or HierarchicalPPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam(planner.parameters(), lr=self.cfg.learning_rate)

        self.learning_iteration = 0

    def update(self, storage: RolloutStorage) -> Dict[str, float]:
        """执行PPO更新"""

        # 标准化advantage
        if self.cfg.normalize_advantage:
            advantages = storage.advantages.reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            storage.advantages = advantages.reshape(storage.num_transitions, storage.num_envs)

        # 统计
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        approx_kl_sum = 0

        for epoch in range(self.cfg.num_learning_epochs):
            for batch in storage.get_batches(self.cfg.num_mini_batches):
                # 评估动作
                subgoal_lp, intensity_lp, entropy, values = self.planner.evaluate_actions(
                    affordance_map=batch['affordance_map'],
                    robot_state=batch['robot_state'],
                    goal=batch['goal'],
                    subgoal=batch['subgoal'],
                    intensity=batch['intensity'],
                    terrain_difficulty=batch['terrain_difficulty'],
                )

                # ===== 子目标 PPO Loss =====
                subgoal_ratio = torch.exp(subgoal_lp - batch['old_subgoal_log_prob'])
                subgoal_surr1 = subgoal_ratio * batch['advantage']
                subgoal_surr2 = torch.clamp(
                    subgoal_ratio, 
                    1 - self.cfg.clip_param, 
                    1 + self.cfg.clip_param
                ) * batch['advantage']
                subgoal_loss = -torch.min(subgoal_surr1, subgoal_surr2).mean()

                # ===== 运动强度 PPO Loss ★修改★ =====
                intensity_ratio = torch.exp(intensity_lp - batch['old_intensity_log_prob'])
                intensity_surr1 = intensity_ratio * batch['advantage']
                intensity_surr2 = torch.clamp(
                    intensity_ratio,
                    1 - self.cfg.clip_param,
                    1 + self.cfg.clip_param
                ) * batch['advantage']
                intensity_loss = -torch.min(intensity_surr1, intensity_surr2).mean()

                # 组合policy loss
                policy_loss = subgoal_loss + intensity_loss

                # ===== Value Loss =====
                value_pred = values.squeeze(-1)
                value_loss = (value_pred - batch['return']).pow(2).mean()

                # ===== Entropy Bonus =====
                entropy_loss = -entropy.mean()

                # ===== Total Loss =====
                total_loss = (
                    policy_loss +
                    self.cfg.value_loss_coef * value_loss +
                    self.cfg.entropy_coef * entropy_loss
                )

                # 更新
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.planner.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # 统计
                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()

                # 近似KL散度
                with torch.no_grad():
                    approx_kl = 0.5 * ((subgoal_lp - batch['old_subgoal_log_prob']).pow(2).mean() +
                                       (intensity_lp - batch['old_intensity_log_prob']).pow(2).mean())
                    approx_kl_sum += approx_kl.item()

        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        self.learning_iteration += 1

        return {
            'total_loss': total_loss_sum / num_updates,
            'policy_loss': policy_loss_sum / num_updates,
            'value_loss': value_loss_sum / num_updates,
            'entropy': entropy_sum / num_updates,
            'approx_kl': approx_kl_sum / num_updates,
        }


class TeacherStudentTrainer:
    """Teacher-Student蒸馏训练器 V2

    训练流程:
    Phase 1: Teacher使用GT Affordance训练
    Phase 2: Student使用Estimated Affordance + 蒸馏损失训练

    蒸馏损失:
    - 子目标: MSE(student_mean, teacher_mean)
    - 运动强度: MSE(student_intensity, teacher_intensity) ★修改★
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        affordance_estimator: nn.Module,
        device: torch.device,
        distill_weight: float = 0.5,
        learning_rate: float = 3e-4,
    ):
        self.teacher = teacher
        self.student = student
        self.affordance_estimator = affordance_estimator
        self.device = device
        self.distill_weight = distill_weight

        # 冻结Teacher
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()

        # 冻结Affordance Estimator
        for param in affordance_estimator.parameters():
            param.requires_grad = False
        affordance_estimator.eval()

        self.student_optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    def distill_step(
        self,
        depth: torch.Tensor,
        gt_affordance_map: torch.Tensor,
        gt_terrain_difficulty: torch.Tensor,
        robot_state: torch.Tensor,
        goal: torch.Tensor,
    ) -> Dict[str, float]:
        """执行一步蒸馏"""

        # Teacher输出 (使用GT affordance)
        with torch.no_grad():
            teacher_output = self.teacher(
                gt_affordance_map, robot_state, goal, gt_terrain_difficulty
            )
            teacher_subgoal = teacher_output.subgoal_mean
            teacher_intensity = teacher_output.intensity_mean

        # Student输出 (使用estimated affordance)
        with torch.no_grad():
            est_affordance = self.affordance_estimator(depth)

        est_aff_map = torch.stack([
            est_affordance['occupancy'],
            est_affordance['traversability']
        ], dim=1)
        est_difficulty = est_affordance['terrain_difficulty']

        student_output = self.student(
            est_aff_map, robot_state, goal, est_difficulty
        )

        # ===== Distillation Losses =====

        # 子目标MSE
        subgoal_loss = F.mse_loss(student_output.subgoal_mean, teacher_subgoal)

        # ★运动强度MSE★
        intensity_loss = F.mse_loss(
            student_output.intensity_mean, 
            teacher_intensity
        )

        # 总损失
        loss = subgoal_loss + self.distill_weight * intensity_loss

        # 更新
        self.student_optimizer.zero_grad()
        loss.backward()
        self.student_optimizer.step()

        return {
            'distill_loss': loss.item(),
            'subgoal_loss': subgoal_loss.item(),
            'intensity_loss': intensity_loss.item(),
        }