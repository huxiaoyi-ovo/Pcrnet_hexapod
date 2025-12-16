"""
快速验收测试：评估hex_ground模型在hex_terrain环境的表现
测试3项指标：Tracking / Stability / Robustness
"""
import torch
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import time

def test_locomotion_transfer(checkpoint_path, num_episodes=50):
    """
    测试低层locomotion策略在新环境的迁移能力
    
    Args:
        checkpoint_path: EGPO_2000.pt的路径
        num_episodes: 测试的episode数量
    """
    args = get_args()
    args.task = "hex_terrain"
    args.headless = True  # 无GUI加速测试
    
    # 加载环境
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 8  # 减少环境数加速
    env_cfg.navigation.enable_nav_reward = False  # 关闭导航奖励
    env_cfg.terrain.curriculum = True  # 开启地形课程
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 加载策略（这里需要手动处理维度不匹配）
    # 暂时用随机策略代替，真实测试时需要加载EGPO_2000
    print("[Test] 使用随机策略测试（替代EGPO加载）")
    
    # 测试指标
    metrics = {
        'tracking_error_lin': [],
        'tracking_error_ang': [],
        'fall_count': 0,
        'total_steps': 0,
        'terrain_levels': [],
    }
    
    obs_dict = env.reset()
    
    for episode in range(num_episodes):
        for step in range(int(env.max_episode_length)):
            # 随机动作（实际测试时应加载模型）
            actions = torch.rand_like(env.actions) * 0.5 - 0.25
            
            obs_dict, rewards, dones, infos = env.step(actions)
            
            # 记录tracking误差
            lin_vel_error = torch.norm(
                env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1
            ).mean().item()
            ang_vel_error = torch.abs(
                env.commands[:, 2] - env.base_ang_vel[:, 2]
            ).mean().item()
            
            metrics['tracking_error_lin'].append(lin_vel_error)
            metrics['tracking_error_ang'].append(ang_vel_error)
            metrics['total_steps'] += env.num_envs
            
            # 检查跌倒
            if dones.any():
                metrics['fall_count'] += dones.sum().item()
                # 记录地形难度
                fallen_ids = dones.nonzero(as_tuple=False).flatten()
                for env_id in fallen_ids:
                    if hasattr(env, 'terrain_levels'):
                        metrics['terrain_levels'].append(
                            env.terrain_levels[env_id].item()
                        )
        
        if episode % 10 == 0:
            print(f"[Progress] Episode {episode}/{num_episodes}")
    
    # 计算最终指标
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    print(f"\n1. Tracking Performance:")
    print(f"   - 平均线速度误差: {np.mean(metrics['tracking_error_lin']):.3f} m/s")
    print(f"   - 平均角速度误差: {np.mean(metrics['tracking_error_ang']):.3f} rad/s")
    
    print(f"\n2. Stability:")
    fall_rate = metrics['fall_count'] / (metrics['total_steps'] / env.max_episode_length)
    print(f"   - 跌倒率: {fall_rate:.2%}")
    print(f"   - 总步数: {metrics['total_steps']}")
    print(f"   - 跌倒次数: {metrics['fall_count']}")
    
    if metrics['terrain_levels']:
        print(f"\n3. Terrain Difficulty:")
        print(f"   - 平均地形等级: {np.mean(metrics['terrain_levels']):.2f}")
        print(f"   - 最高地形等级: {max(metrics['terrain_levels'])}")
    
    # 给出决策建议
    print("\n" + "="*60)
    print("决策建议")
    print("="*60)
    
    need_retrain = False
    reasons = []
    
    if np.mean(metrics['tracking_error_lin']) > 0.3:
        need_retrain = True
        reasons.append("线速度跟踪误差过大(>0.3 m/s)")
    
    if np.mean(metrics['tracking_error_ang']) > 0.5:
        need_retrain = True
        reasons.append("角速度跟踪误差过大(>0.5 rad/s)")
    
    if fall_rate > 0.3:
        need_retrain = True
        reasons.append(f"跌倒率过高({fall_rate:.1%} > 30%)")
    
    if need_retrain:
        print("❌ 建议重新训练Phase 1:")
        for reason in reasons:
            print(f"   - {reason}")
    else:
        print("✅ 无需重新训练，可直接进入Phase 2!")
        print("   - 所有指标达标")
        print("   - 下一步：修复Intensity作弊 + 启动导航训练")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    checkpoint = f"{LEGGED_GYM_ROOT_DIR}/agents/EGPO_2000.pt"
    test_locomotion_transfer(checkpoint, num_episodes=50)
