#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化深度相机输出"""

import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)
print(f"[Debug] Project root added to path: {project_root}")

from legged_gym.envs.hex_v4.hex_terrain import HexTerrain
from legged_gym.envs.hex_v4.hex_terrain_config import HexTerrainCfg
from legged_gym.utils import get_args, class_to_dict
from legged_gym.utils.helpers import parse_sim_params
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端

def visualize_depth():
    """可视化深度相机输出"""
    
    # 获取参数
    args = get_args()
    args.headless = True  # 必须非headless才能使用相机
    
    # 创建配置
    cfg = HexTerrainCfg()
    cfg.sensor.depth_camera.enable = True
    cfg.env.num_envs = 1  # 只用2个环境，节省显存
    cfg.sensor.depth_camera.width = 64  # 降低分辨率
    cfg.sensor.depth_camera.height = 64
    cfg.sensor.depth_camera.capture_interval = 10  # 每步都采集
    
    # 减少地形复杂度以节省显存
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.terrain.curriculum = False
    
    print("\n" + "="*70)
    print("Depth Camera Visualization")
    print("="*70)
    
    # 解析仿真参数
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    
    print("\n[1] Creating environment...")
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
    
    # 创建matplotlib窗口（显示2个环境）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    print("\n[3] Starting visualization (press Ctrl+C to stop)...")
    print("    - Window shows depth images from 2 environments")
    print("    - Brighter = closer, Darker = farther")
    print("    - Blue/Purple = very close, Red/Yellow = far away")
    
    plt.ion()  # 开启交互模式
    
    step_count = 0
    try:
        while True:
            # 执行一步（机器人保持静止）
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs_dict, rewards, dones, infos = env.step_separate(actions)
            
            # 获取深度图
            if obs_dict['depth'] is not None:
                # depth shape: (num_envs, 1, H, W)
                depth_images = obs_dict['depth'].cpu().numpy()
                
                # 更新每个子图
                for i in range(min(2, env.num_envs)):
                    depth = depth_images[i, 0]  # (H, W)
                    
                    # 转换为可视化（取绝对值，因为Isaac Gym返回负值）
                    depth_vis = np.abs(depth)
                    
                    # 清空并重绘
                    axes[i].clear()
                    
                    # 使用jet colormap: 蓝色=近，红色=远
                    im = axes[i].imshow(depth_vis, cmap='jet', vmin=0.05, vmax=5.0)
                    axes[i].set_title(f'Env {i} - Step {step_count}\n'
                                     f'Range: [{depth_vis.min():.2f}, {depth_vis.max():.2f}]m',
                                     fontsize=10)
                    axes[i].axis('off')
                    
                    # 添加colorbar（仅第一次）
                    if step_count == 0:
                        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                
                # 添加总标题
                fig.suptitle(f'Depth Camera Visualization - Step {step_count}\n'
                           f'Camera: pos={cfg.sensor.depth_camera.position}, '
                           f'pitch={cfg.sensor.depth_camera.pitch_deg}°',
                           fontsize=12)
                
                plt.pause(0.01)  # 短暂暂停以更新显示
            
            step_count += 1
            
            # 每50步打印一次统计信息
            if step_count % 50 == 0:
                if obs_dict['depth'] is not None:
                    depth_np = np.abs(obs_dict['depth'].cpu().numpy())
                    print(f"    Step {step_count}: depth range [{depth_np.min():.3f}, {depth_np.max():.3f}]m, "
                          f"mean={depth_np.mean():.3f}m")
    
    except KeyboardInterrupt:
        print("\n\n[4] Visualization stopped by user")
    
    plt.ioff()
    plt.show()  # 保持窗口打开
    
    print("\n" + "="*70)
    print("Visualization completed")
    print("="*70 + "\n")

if __name__ == "__main__":
    visualize_depth()
