#!/usr/bin/env python3
"""
监控相机稳定性相关的实际数值
用于验证Gemini关于角加速度过大的假设
"""

import torch
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import task_registry

class MonitorArgs:
    task = 'hex_terrain'
    sim_device = 'cuda:0'
    rl_device = 'cuda:0'
    physics_engine = isaacgym.gymapi.SIM_PHYSX
    headless = True
    num_envs = 128
    use_gpu = True
    use_gpu_pipeline = True

if __name__ == '__main__':
    print("=== 相机稳定性数值监控 ===\n")
    
    args = MonitorArgs()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # 重置环境
    obs = env.reset()
    
    # 执行一些步骤以获得稳定数据
    for i in range(100):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rewards, dones, infos = env.step(actions)
    
    # 采集统计数据
    stats = {
        'base_ang_acc': env.base_ang_acc,
        'base_ang_vel': env.base_ang_vel,
        'base_lin_acc': env.base_lin_acc,
    }
    
    print("=== 角加速度 (base_ang_acc) ===")
    ang_acc_xy = torch.sum(torch.square(stats['base_ang_acc'][:, :2]), dim=1)
    print(f"  ang_acc_xy 均值: {ang_acc_xy.mean().item():.2f}")
    print(f"  ang_acc_xy 最大: {ang_acc_xy.max().item():.2f}")
    print(f"  ang_acc_xy 最小: {ang_acc_xy.min().item():.2f}")
    print(f"  ang_acc_xy 标准差: {ang_acc_xy.std().item():.2f}")
    
    print("\n=== 角速度 (base_ang_vel) ===")
    ang_vel_xy = torch.sum(torch.square(stats['base_ang_vel'][:, :2]), dim=1)
    print(f"  ang_vel_xy 均值: {ang_vel_xy.mean().item():.4f}")
    print(f"  ang_vel_xy 最大: {ang_vel_xy.max().item():.4f}")
    
    print("\n=== 线加速度 z (base_lin_acc) ===")
    lin_acc_z = torch.square(stats['base_lin_acc'][:, 2])
    print(f"  lin_acc_z 均值: {lin_acc_z.mean().item():.4f}")
    print(f"  lin_acc_z 最大: {lin_acc_z.max().item():.4f}")
    
    print("\n=== 组合惩罚计算 ===")
    jitter_w = 0.05
    wobble_w = 0.5
    bobbing_w = 0.1
    
    penalty = ang_acc_xy * jitter_w + ang_vel_xy * wobble_w + lin_acc_z * bobbing_w
    reward = torch.exp(-penalty)
    
    print(f"  penalty 均值: {penalty.mean().item():.2f}")
    print(f"  penalty 最大: {penalty.max().item():.2f}")
    print(f"  reward 均值: {reward.mean().item():.4f}")
    print(f"  reward 最小: {reward.min().item():.6f}")
    
    print("\n=== Gemini假设验证 ===")
    print(f"  Gemini预测: ang_acc_xy ~ 100")
    print(f"  实际测量: ang_acc_xy ~ {ang_acc_xy.mean().item():.2f}")
    print(f"  Gemini预测: penalty ~ 5.3")
    print(f"  实际测量: penalty ~ {penalty.mean().item():.2f}")
    
    if ang_acc_xy.mean().item() > 50:
        print("\n⚠️ 验证: Gemini的诊断正确！角加速度过大！")
        print(f"   建议: camera_jitter_weight 从 0.05 降到 0.005")
    else:
        print("\n✅ 当前权重可能合理，需进一步观察训练曲线")
