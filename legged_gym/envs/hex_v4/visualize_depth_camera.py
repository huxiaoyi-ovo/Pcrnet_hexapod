#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化深度相机输出

示例运行命令:
    # 基本运行（窗口显示 + 保存 PNG 和 CSV）
    python legged_gym/envs/hex_v4/visualize_depth_camera.py --frames 50

    # 仅保存（无窗口）
    python legged_gym/envs/hex_v4/visualize_depth_camera.py --frames 50 --no_show
    /home/hxy/anaconda3/envs/hexapod_rl_env/bin/python legged_gym/envs/hex_v4/visualize_depth_camera.py --frames 5 --no_show

    # 自定义分辨率 + 色图
    python legged_gym/envs/hex_v4/visualize_depth_camera.py --frames 30 --width 128 --height 128 --colormap inferno

    # 自定义输出目录
    python legged_gym/envs/hex_v4/visualize_depth_camera.py --frames 20 --save_dir logs/camera_visualizations/depth_test

参数说明:
    --frames N        保存 / (可视化) 步数
    --no_show         禁用窗口，仅写文件
    --width/--height  覆盖相机分辨率
    --colormap name   保存与显示使用的色图 (默认 viridis，可选 inferno/plasma/magma 等)
    --save_dir path   输出目录 (默认 logs/camera_visualizations/depth)

说明:
    * 脚本已强制使用 CPU PhysX + 关闭 GPU pipeline 来减少 CUDA 错误日志。
    * 每帧会保存 depth_XXXX_env0.png 及写入 depth_stats.csv。
"""

import sys
import os
import csv

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
from isaacgym import gymapi
matplotlib.use('TkAgg')  # 使用交互式后端

# 固定随机种子，确保与 RGB 脚本的机器人初始姿态一致
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 统一使用 RGB 测试脚本中的障碍物生成函数，避免重复代码与参数差异
from legged_gym.envs.hex_v4.visualize_rgb_camera import add_front_obstacle

def extract_local_args():
    """从 sys.argv 中抽取本脚本自定义参数并移除，避免 get_args() 报未知参数。

    支持参数：
      --save_dir <path>    输出目录，默认 logs/camera_visualizations/depth
      --frames <N>         保存帧数，默认 100（同时更新窗口）
      --interval <N>       采样间隔（保留占位，目前未用），默认 1
      --width <W>          覆盖相机宽度
      --height <H>         覆盖相机高度
      --colormap <name>    保存图像使用的 colormap（jet/magma 等），默认 jet
      --no_show            不弹出窗口，仅保存文件
    """
    local = {
        'save_dir': 'logs/camera_visualizations/depth',
        'frames': 100,
        'interval': 1,
        'width': None,
        'height': None,
        'colormap': 'viridis',
        'no_show': False,
    }
    argv = sys.argv
    i = 1
    remove = set()
    while i < len(argv):
        tok = argv[i]
        if tok == '--save_dir' and i + 1 < len(argv):
            local['save_dir'] = argv[i+1]; remove.update({i, i+1}); i += 2; continue
        if tok == '--frames' and i + 1 < len(argv):
            local['frames'] = int(argv[i+1]); remove.update({i, i+1}); i += 2; continue
        if tok == '--interval' and i + 1 < len(argv):
            local['interval'] = int(argv[i+1]); remove.update({i, i+1}); i += 2; continue
        if tok == '--width' and i + 1 < len(argv):
            local['width'] = int(argv[i+1]); remove.update({i, i+1}); i += 2; continue
        if tok == '--height' and i + 1 < len(argv):
            local['height'] = int(argv[i+1]); remove.update({i, i+1}); i += 2; continue
        if tok == '--colormap' and i + 1 < len(argv):
            local['colormap'] = argv[i+1]; remove.update({i, i+1}); i += 2; continue
        if tok == '--no_show':
            local['no_show'] = True; remove.add(i); i += 1; continue
        i += 1
    if remove:
        sys.argv = [arg for idx, arg in enumerate(argv) if idx not in remove]

    class LocalArgs: pass
    ns = LocalArgs()
    for k, v in local.items():
        setattr(ns, k, v)
    return ns


def visualize_depth():
    """可视化深度相机输出"""
    
    # 获取参数（先提取本地参数，剥离后再调用 get_args）
    local_args = extract_local_args()
    args = get_args()
    # 关闭 PhysX GPU 与 GPU pipeline，减少 CUDA 错误日志与显存占用
    if hasattr(args, 'use_gpu'):
        args.use_gpu = False
    if hasattr(args, 'use_gpu_pipeline'):
        args.use_gpu_pipeline = False
    args.headless = True  # 使用 headless 配合相机传感器渲染
    
    # 创建配置
    cfg = HexTerrainCfg()
    cfg.sensor.depth_camera.enable = True
    cfg.env.num_envs = 1  # 只用2个环境，节省显存
    # 可通过命令行覆盖分辨率
    cfg.sensor.depth_camera.width = local_args.width or 64  # 降低分辨率
    cfg.sensor.depth_camera.height = local_args.height or 64
    cfg.sensor.depth_camera.capture_interval = 10  # 每步都采集
    
    # 使用平面地形，与RGB脚本保持一致，确保环境原点在(0,0,0)附近
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
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

    # 在机器人前方添加一个红色障碍物（默认 0.8m 前方，尺寸 0.3x0.3x0.2m）
    add_front_obstacle(env, env_id=0, distance=0.8, size_xyz=(0.3, 0.3, 0.2), color=(1.0, 0.0, 0.0))
    # 渲染一次以确保相机更新
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    
    # 创建matplotlib窗口（显示2个环境）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()
    
    print("\n[3] Starting visualization (press Ctrl+C to stop)...")
    print(f"    - Window shows depth images from up to 2 envs (actual={env.num_envs})")
    print("    - Brighter = closer, Darker = farther")
    print("    - Blue/Purple = very close, Red/Yellow = far away")
    
    # 输出目录与 CSV 统计
    os.makedirs(local_args.save_dir, exist_ok=True)
    stats_path = os.path.join(local_args.save_dir, 'depth_stats.csv')
    csv_file = open(stats_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'env', 'min', 'max', 'mean', 'valid_pixels', 'invalid_pixels'])

    if not local_args.no_show:
        plt.ion()  # 开启交互模式
    
    step_count = 0
    try:
        while True:
            # 直接抓取原始深度（单位：米，已在环境函数中转为正值）
            depth_raw = env._get_depth_images().cpu().numpy()  # (N, H, W)
            
            # 更新每个子图
            for i in range(min(2, env.num_envs)):
                depth = depth_raw[i]  # (H, W)
                # 保存 PNG
                png_path = os.path.join(local_args.save_dir, f'depth_{step_count:04d}_env{i}.png')
                try:
                    plt.imsave(png_path, depth, cmap=local_args.colormap,
                               vmin=cfg.sensor.depth_camera.near_clip,
                               vmax=cfg.sensor.depth_camera.far_clip)
                except Exception as e:
                    print(f"[Warn] Failed to save {png_path}: {e}")

                # 计算统计并写入 CSV
                dmin = float(np.nanmin(depth))
                dmax = float(np.nanmax(depth))
                dmean = float(np.nanmean(depth))
                invalid = int(np.count_nonzero(~np.isfinite(depth)))
                total = depth.size
                valid = int(total - invalid)
                csv_writer.writerow([step_count, i, dmin, dmax, dmean, valid, invalid])

                # 绘制到窗口（如启用）
                if not local_args.no_show:
                    axes[i].clear()
                    im = axes[i].imshow(depth, cmap=local_args.colormap,
                                        vmin=cfg.sensor.depth_camera.near_clip,
                                        vmax=cfg.sensor.depth_camera.far_clip)
                    axes[i].set_title(
                        f'Env {i} - Step {step_count}\nRange: [{dmin:.2f}, {dmax:.2f}]m',
                        fontsize=10
                    )
                    axes[i].axis('off')
                    if step_count == 0:
                        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            if not local_args.no_show:
                fig.suptitle(
                    f'Depth Camera Visualization - Step {step_count}\n'
                    f'Camera: pos={cfg.sensor.depth_camera.position}, pitch={cfg.sensor.depth_camera.pitch_deg}°',
                    fontsize=12
                )
                plt.pause(0.01)

            step_count += 1
            if step_count % 50 == 0:
                dr = depth_raw
                print(f"    Step {step_count}: depth range [{np.nanmin(dr):.3f}, {np.nanmax(dr):.3f}]m, mean={np.nanmean(dr):.3f}m")
            # 若达到保存帧数限制则退出循环
            if step_count >= local_args.frames:
                break

    except KeyboardInterrupt:
        print("\n\n[4] Visualization stopped by user")
    
    # 关闭 CSV 文件
    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass

    if not local_args.no_show:
        plt.ioff()
        plt.show()  # 保持窗口打开
    
    print("\n" + "="*70)
    print("Visualization completed")
    print("="*70 + "\n")

if __name__ == "__main__":
    visualize_depth()
