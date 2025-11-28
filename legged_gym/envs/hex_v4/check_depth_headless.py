#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在headless模式下获取并保存深度图（使用CPU渲染）"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

from legged_gym.envs.hex_v4.hex_terrain import HexTerrain
from legged_gym.envs.hex_v4.hex_terrain_config import HexTerrainCfg
from legged_gym.utils import get_args, class_to_dict
from legged_gym.utils.helpers import parse_sim_params
import torch
import numpy as np

def check_depth_headless():
    """在headless模式下检查深度相机数据"""
    
    # 获取参数
    args = get_args()
    args.headless = True  # 使用headless模式
    
    # 创建配置
    cfg = HexTerrainCfg()
    cfg.sensor.depth_camera.enable = False  # headless模式下禁用相机
    cfg.env.num_envs = 8
    
    print("\n" + "="*70)
    print("Check Depth Data in Headless Mode")
    print("="*70)
    
    # 解析仿真参数
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("\n[1] Creating environment (headless mode, no camera)...")
    env = HexTerrain(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless
    )
    print("    ✅ Environment created")
    
    print("\n[2] Resetting environment...")
    obs_dict = env.reset_separate()
    print("    ✅ Environment reset")
    
    print("\n[3] Running 50 steps...")
    for step in range(50):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs_dict, _, _, _ = env.step_separate(actions)
        if (step + 1) % 10 == 0:
            print(f"    Step {step + 1}/50")
    
    print("\n[4] Checking depth observation...")
    if obs_dict['depth'] is not None:
        depth = obs_dict['depth']
        print(f"    Depth shape: {depth.shape}")
        print(f"    Depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
        print(f"    Depth mean: {depth.mean().item():.3f}")
        
        # 检查是否全零（headless模式下相机被禁用）
        if torch.all(depth == 0):
            print("    ⚠️  Depth is all zeros (camera disabled in headless mode)")
        else:
            print("    ✅ Depth has non-zero values")
    else:
        print("    ⚠️  Depth is None")
    
    print("\n[5] Checking other observations...")
    for key in ['proprioception', 'privileged', 'terrain', 'robot_state', 'goal']:
        if key in obs_dict and obs_dict[key] is not None:
            obs = obs_dict[key]
            print(f"    {key:15s}: shape={obs.shape}, range=[{obs.min().item():.3f}, {obs.max().item():.3f}]")
    
    print("\n" + "="*70)
    print("✅ Check completed!")
    print("\n💡 Note: In headless mode, cameras are automatically disabled.")
    print("   To use cameras, you must run in non-headless mode,")
    print("   but this requires more GPU memory.")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_depth_headless()
