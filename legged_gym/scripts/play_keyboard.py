"""
scripts/play_keyboard.py - 键盘控制六足机器人 (基于 fast_2000.pt)

功能:
1. 加载 Isaac Gym 环境和底层策略 (fast_2000.pt).
2. 监听键盘输入，实时修改 env.commands (vx, vy, omega).
3. 将指令输入 Policy，驱动机器人移动.

按键说明:
  [↑/↓] : 前进/后退 (Linear X)
  [←/→] : 左移/右移 (Linear Y)
  [A/D] : 左转/右转 (Angular Yaw)
  [Space]: 刹车 (Stop)
  [R]    : 重置 (Reset)
"""

import os
import sys
import time
import isaacgym
from isaacgym import gymapi, gymutil
import numpy as np
import torch

# 导入 legged_gym 模块
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules import ActorCritic

def play(args):
    if getattr(args, "task", None) == "hex_terrain":
        raise RuntimeError("hex_terrain 已移除，请改用 hex_ground / s_avoid_basic..s_ood_holdout / s_calib")
    # 1. 准备环境配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 强制修改配置以适应 Play 模式
    env_cfg.env.num_envs = 1  # 只控制一个机器人
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False # 关闭课程，随机地形
    env_cfg.noise.add_noise = False    # 关闭观测噪声
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # 创建环境
    print(f"Creating environment: {args.task}")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 2. 加载模型 (fast_2000.pt)
    device = env.device
    load_path = args.load_run
    print(f"Loading model from: {load_path}")
    
    if not os.path.exists(load_path):
        print(f"[Error] Model file not found: {load_path}")
        sys.exit(1)

    # 初始化网络 (必须与训练时的结构一致)
    # 根据 hex_ground 配置，假设是 [512, 256, 128]
    policy = ActorCritic(
        num_actor_obs=env.num_obs,
        num_critic_obs=env.num_privileged_obs,
        num_actions=env.num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
    ).to(device)

    # 加载权重
    checkpoint = torch.load(load_path, map_location=device)
    # 兼容处理 - 根据checkpoint内容判断加载方式
    if 'model_state_dict' in checkpoint:
        # 完整的模型state_dict (包含actor和critic)
        policy.load_state_dict(checkpoint['model_state_dict'])
    elif 'actor_state_dict' in checkpoint:
        # 只有actor的state_dict
        policy.actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        # checkpoint直接是state_dict，需要判断是完整模型还是只有actor
        # 检查键名格式
        sample_key = list(checkpoint.keys())[0]
        
        if 'actor.' in sample_key and not any('critic' in key for key in checkpoint.keys()):
            # 键名带有 "actor." 前缀，但没有critic权重
            # 需要提取actor部分并去掉前缀
            print("[Info] Extracting actor weights from checkpoint")
            actor_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('actor.'):
                    # 去掉 "actor." 前缀
                    new_key = key.replace('actor.', '')
                    actor_state_dict[new_key] = value
            policy.actor.load_state_dict(actor_state_dict)
        elif any('critic' in key for key in checkpoint.keys()):
            # 包含critic，使用strict=False加载完整模型（忽略缺失的critic键）
            print("[Info] Loading model with strict=False (critic weights may be missing)")
            policy.load_state_dict(checkpoint, strict=False)
        else:
            # 直接是actor的权重，不带前缀
            print("[Info] Loading actor weights directly")
            policy.actor.load_state_dict(checkpoint)
    
    policy.eval()

    # 3. 设置键盘监听
    # Isaac Gym 的 viewer 提供了订阅键盘事件的接口
    
    class KeyboardController:
        def __init__(self):
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_yaw = 0.0
            self.step_size = 0.1 # 每次按键增加的速度量
            self.max_vel = 1.0
            self.reset_flag = False

        def print_instructions(self):
            print("\n" + "="*30)
            print("   Hexapod Keyboard Control")
            print("="*30)
            print(" [Up/Down]    : Vel X (+/-)")
            print(" [Left/Right] : Vel Y (+/-)")
            print(" [A/D]        : Turn  (+/-)")
            print(" [Space]      : Stop")
            print(" [R]          : Reset Env")
            print("="*30 + "\n")

        def update(self, gym, viewer):
            # 获取所有按下的键
            # 注意: Isaac Gym 的 Python API 对键盘支持有限，
            # 我们这里通过 subscribe_viewer_keyboard_event 来处理事件
            pass # 实际逻辑在主循环处理

    controller = KeyboardController()
    controller.print_instructions()

    # 注册按键回调
    gym = env.gym
    viewer = env.viewer
    
    # 订阅需要的按键
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "UP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "DOWN")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "LEFT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "RIGHT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "TURN_LEFT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "TURN_RIGHT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "STOP")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "RESET")

    # 4. 主循环
    obs, _ = env.reset()  # 返回 (obs, privileged_obs)
    
    while not gym.query_viewer_has_closed(viewer):
        # 处理键盘事件
        events = gym.query_viewer_action_events(viewer)
        for evt in events:
            if evt.action == "UP" and evt.value > 0:
                controller.vel_x += controller.step_size
            elif evt.action == "DOWN" and evt.value > 0:
                controller.vel_x -= controller.step_size
            elif evt.action == "LEFT" and evt.value > 0:
                controller.vel_y += controller.step_size # Isaac Gym 坐标系: 左通常是 +Y
            elif evt.action == "RIGHT" and evt.value > 0:
                controller.vel_y -= controller.step_size
            elif evt.action == "TURN_LEFT" and evt.value > 0: # A
                controller.vel_yaw += controller.step_size
            elif evt.action == "TURN_RIGHT" and evt.value > 0: # D
                controller.vel_yaw -= controller.step_size
            elif evt.action == "STOP" and evt.value > 0:
                controller.vel_x = 0.0
                controller.vel_y = 0.0
                controller.vel_yaw = 0.0
            elif evt.action == "RESET" and evt.value > 0:
                controller.reset_flag = True

        # 限制速度范围 (根据 hex_ground_config.py 中的 commands.ranges)
        controller.vel_x = np.clip(controller.vel_x, -1.0, 1.0)
        controller.vel_y = np.clip(controller.vel_y, -1.5, 1.5)
        controller.vel_yaw = np.clip(controller.vel_yaw, -2.0, 2.0)

        # 手动覆盖环境的 commands
        # commands: [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # 注意：需要将物理单位转换为环境使用的归一化单位(如果有缩放)，
        # 但通常 legged_gym 的 commands 直接就是物理量 (m/s, rad/s)
        env.commands[:, 0] = controller.vel_x
        env.commands[:, 1] = controller.vel_y
        env.commands[:, 2] = controller.vel_yaw

        # 处理重置
        if controller.reset_flag:
            obs, _ = env.reset()
            controller.reset_flag = False
            controller.vel_x = 0; controller.vel_y = 0; controller.vel_yaw = 0
        
        # 推理
        with torch.no_grad():
            actions = policy.act(obs.detach())

        # 执行
        obs, _, _, _, _ = env.step(actions.detach())
        
        # 简单显示当前指令
        # print(f"\rCmd: Vx={controller.vel_x:.1f} Vy={controller.vel_y:.1f} W={controller.vel_yaw:.1f}", end="")

if __name__ == '__main__':
    # 构造参数
    # 如果您没有使用 argparse，可以手动设置类
    args = get_args()
    
    # 默认值覆盖 (如果没有在命令行提供)
    if not hasattr(args, 'task') or not args.task:
        args.task = "hex_debug_plane"
    if not hasattr(args, 'load_run') or not args.load_run:
        args.load_run = "agents/fast_2000.pt" # 默认路径
    
    play(args)
