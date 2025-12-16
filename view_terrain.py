# -*- coding: utf-8 -*-
"""
查看hex_terrain的地形分布
不加载模型，只显示地形
"""

from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch

def view_terrain(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 配置：更少的机器人，更多地形可视化
    env_cfg.env.num_envs = 1  # 只需要1个机器人
    env_cfg.terrain.num_rows = 10  # 保持原配置，看全部难度
    env_cfg.terrain.num_cols = 20  # 保持原配置，看全部地形类型
    env_cfg.terrain.curriculum = False  # 关闭curriculum便于观察
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # 创建环境
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    
    # 打印地形信息
    print("\n" + "="*60)
    print("HEX_TERRAIN 地形配置信息")
    print("="*60)
    print("地形网格: {} rows x {} cols = {} 块地形".format(
        env_cfg.terrain.num_rows, 
        env_cfg.terrain.num_cols,
        env_cfg.terrain.num_rows * env_cfg.terrain.num_cols
    ))
    print("地形尺寸: {} x {} 米".format(
        env_cfg.terrain.terrain_length,
        env_cfg.terrain.terrain_width
    ))
    print("\n地形类型分布:")
    terrain_types = [
        "Smooth Slope",
        "Rough Slope", 
        "Stairs A",
        "Stairs B",
        "Discrete Obstacles",
        "Stepping Stones",
        "Gap",
        "Pit",
        "Gate (导航地形)",
        "Slalom (导航地形)"
    ]
    proportions = env_cfg.terrain.terrain_proportions
    for i, (t_type, prop) in enumerate(zip(terrain_types, proportions)):
        if prop > 0:
            print("  [{:2d}] {:20s} - {:5.1f}%".format(i, t_type, prop * 100))
    
    print("\n导航地形占比: {:.1f}% (Gate + Slalom)".format(
        (proportions[8] + proportions[9]) * 100
    ))
    print("基础地形占比: {:.1f}%".format(
        sum(proportions[:8]) * 100
    ))
    
    print("\n" + "="*60)
    print("控制说明:")
    print("  - 使用鼠标拖动旋转视角")
    print("  - 滚轮缩放")
    print("  - 方向键移动摄像机")
    print("  - ESC 退出")
    print("="*60 + "\n")
    
    # 简单的随机action让机器人不要一直站着不动
    for i in range(100000):
        with torch.inference_mode():
            # 给一个小的随机命令，让机器人稍微动一下（便于观察地形）
            if i % 500 == 0:
                env.commands[:, 0] = torch.rand(1, device=env.device) * 0.3 - 0.15  # x: -0.15~0.15
                env.commands[:, 1] = torch.rand(1, device=env.device) * 0.2 - 0.1   # y: -0.1~0.1
                env.commands[:, 2] = torch.rand(1, device=env.device) * 0.4 - 0.2   # yaw: -0.2~0.2
            
            # 简单的PD控制，保持站立
            actions = torch.zeros(1, env.num_actions, device=env.device)
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == '__main__':
    args = get_args()
    args.task = "hex_terrain"  # 强制使用hex_terrain
    view_terrain(args)
