#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一同步显示与保存 RGB 与 Depth 图像的测试脚本

示例运行:
    # 保存并显示 (默认分辨率 128x128)
    python legged_gym/envs/hex_v4/visualize_rgb_depth_combined.py --frames 20

    # 仅保存，不显示窗口
    python legged_gym/envs/hex_v4/visualize_rgb_depth_combined.py --frames 20 --no_show

    # 自定义分辨率与色图
    python legged_gym/envs/hex_v4/visualize_rgb_depth_combined.py --frames 10 --width 128 --height 128 --colormap magma

    # 启用遮罩叠加 (将近处区域用红色透明覆盖)
    python legged_gym/envs/hex_v4/visualize_rgb_depth_combined.py --frames 10 --overlay --overlay_thresh 0.3

参数:
    --frames N            采集帧数 (默认 50)
    --save_dir path       输出目录 (默认 logs/camera_visualizations/combined)
    --width/--height      相机分辨率覆盖 (默认 128x128)
    --colormap name       深度图色图 (默认 jet)
    --no_show             不显示窗口
    --overlay             在 RGB 上叠加近距离遮罩
    --overlay_thresh F    归一化深度阈值 (0~1), 默认 0.25
    --channels_last       RGB 保存为 (H,W,3)
    --no_normalize        RGB 不归一化 (uint8)

说明:
    * 使用单环境、CPU PhysX、关闭 GPU pipeline，减轻资源占用。
    * 障碍物放置与其它脚本保持一致 (前方 +Y 方向 distance=0.8)。
    * 每帧保存: rgb_XXXX.png, depth_XXXX.png, combined_XXXX.png (拼接), 可选 overlay。
    * 统计写入 combined_stats.csv。
"""

import sys
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# 项目根路径加入 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)
print(f"[Debug] Project root: {project_root}")

from legged_gym.envs.hex_v4.hex_ground import HexGround
from legged_gym.envs.hex_v4.hex_scenes_config import HexDebugHeightfieldCfg
from legged_gym.utils import get_args, class_to_dict
from legged_gym.utils.helpers import parse_sim_params
import torch  # Isaac Gym 要求 torch 在其后导入

matplotlib.use('TkAgg')  # 若 --no_show 则不会弹窗


def extract_local_args():
    local = {
        'frames': 50,
        'save_dir': 'logs/camera_visualizations/combined',
        'width': 128,
        'height': 128,
        'colormap': 'jet',
        'no_show': False,
        'overlay': False,
        'overlay_thresh': 0.25,
        'channels_last': False,
        'no_normalize': False,
    }
    argv = sys.argv
    i = 1
    remove = set()
    while i < len(argv):
        tok = argv[i]
        if tok == '--frames' and i + 1 < len(argv):
            local['frames'] = int(argv[i+1]); remove.update({i,i+1}); i += 2; continue
        if tok == '--save_dir' and i + 1 < len(argv):
            local['save_dir'] = argv[i+1]; remove.update({i,i+1}); i += 2; continue
        if tok == '--width' and i + 1 < len(argv):
            local['width'] = int(argv[i+1]); remove.update({i,i+1}); i += 2; continue
        if tok == '--height' and i + 1 < len(argv):
            local['height'] = int(argv[i+1]); remove.update({i,i+1}); i += 2; continue
        if tok == '--colormap' and i + 1 < len(argv):
            local['colormap'] = argv[i+1]; remove.update({i,i+1}); i += 2; continue
        if tok == '--no_show':
            local['no_show'] = True; remove.add(i); i += 1; continue
        if tok == '--overlay':
            local['overlay'] = True; remove.add(i); i += 1; continue
        if tok == '--overlay_thresh' and i + 1 < len(argv):
            local['overlay_thresh'] = float(argv[i+1]); remove.update({i,i+1}); i += 2; continue
        if tok == '--channels_last':
            local['channels_last'] = True; remove.add(i); i += 1; continue
        if tok == '--no_normalize':
            local['no_normalize'] = True; remove.add(i); i += 1; continue
        i += 1
    if remove:
        sys.argv = [arg for idx, arg in enumerate(argv) if idx not in remove]
    class LocalArgs: pass
    ns = LocalArgs()
    for k,v in local.items(): setattr(ns,k,v)
    return ns


def prepare_env(local_args):
    args = get_args()
    # 关闭 GPU pipeline 使用 CPU
    if hasattr(args,'use_gpu'): args.use_gpu = False
    if hasattr(args,'use_gpu_pipeline'): args.use_gpu_pipeline = False
    args.headless = True

    cfg = HexDebugHeightfieldCfg()
    cam_cfg = cfg.sensor.depth_camera
    cam_cfg.enable = True
    cam_cfg.width = local_args.width
    cam_cfg.height = local_args.height
    cam_cfg.capture_interval = 10
    cfg.env.num_envs = 1
    
    # 使用 debug heightfield 任务配置（非平整高度场）
    cfg.control.use_actuator_net = False

    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    env = HexGround(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    return env, args, cfg


def save_rgb(rgb_tensor, path, normalize=True, channels_last=False):
    if channels_last:
        arr = rgb_tensor.cpu().numpy()
    else:
        arr = rgb_tensor.cpu().numpy().transpose(1,2,0)
    if normalize and arr.dtype != np.uint8:
        arr = np.clip(arr*255.0,0,255).astype(np.uint8)
    elif not normalize and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_depth(depth_tensor, path, near, far, colormap):
    depth = depth_tensor.cpu().numpy()
    norm = (depth - near) / (far - near)
    norm = np.clip(norm, 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps.get(colormap)
    except Exception:
        cmap = matplotlib.cm.get_cmap(colormap)
    rgba = cmap(norm)  # (H,W,4)
    rgb = (rgba[...,:3]*255.0).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def apply_overlay(rgb_arr_uint8, depth_tensor, near, far, thresh):
    depth = depth_tensor.cpu().numpy()
    norm = (depth - near)/(far-near)
    mask = norm < thresh
    if mask.sum() == 0:
        return rgb_arr_uint8
    # 红色半透明叠加
    overlay = rgb_arr_uint8.copy()
    overlay[mask] = (overlay[mask]*0.3 + np.array([255,0,0])*0.7).astype(np.uint8)
    return overlay


def main():
    local_args = extract_local_args()
    env, args, cfg = prepare_env(local_args)
    normalize = not local_args.no_normalize
    channels_last = local_args.channels_last

    os.makedirs(local_args.save_dir, exist_ok=True)
    print(f"[Info] Saving images to: {local_args.save_dir}")

    # 复位（地形已在 prepare_env 中通过官方 pyramid_stairs_terrain 生成）
    env.reset_separate()
    print(f"[Terrain] Using official pyramid_stairs_terrain with step_height=-0.08m (descending)")
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    # Matplotlib
    if not local_args.no_show:
        plt.ion()
        fig, (ax_rgb, ax_depth) = plt.subplots(1,2, figsize=(8,4))

    stats_path = os.path.join(local_args.save_dir, 'combined_stats.csv')
    csv_file = open(stats_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['frame','rgb_min','rgb_max','rgb_mean','depth_min','depth_max','depth_mean'])

    near = cfg.sensor.depth_camera.near_clip
    far  = cfg.sensor.depth_camera.far_clip

    for frame in range(local_args.frames):
        depth_all = env._get_depth_images()  # (1,H,W)
        rgb_all = env._get_rgb_images(normalize=normalize, channels_last=channels_last)
        depth = depth_all[0]
        rgb = rgb_all[0]

        # 保存单独图片
        rgb_path = os.path.join(local_args.save_dir, f'rgb_{frame:04d}.png')
        depth_path = os.path.join(local_args.save_dir, f'depth_{frame:04d}.png')
        save_rgb(rgb, rgb_path, normalize=normalize, channels_last=channels_last)
        save_depth(depth, depth_path, near, far, local_args.colormap)

        # 构造 combined 横向拼接
        if channels_last:
            rgb_arr = rgb.cpu().numpy()
            if normalize:
                rgb_uint8 = (np.clip(rgb_arr,0,1)*255).astype(np.uint8)
            else:
                rgb_uint8 = rgb_arr.astype(np.uint8)
        else:
            rgb_arr = rgb.cpu().numpy().transpose(1,2,0)
            if normalize:
                rgb_uint8 = (np.clip(rgb_arr,0,1)*255).astype(np.uint8)
            else:
                rgb_uint8 = rgb_arr.astype(np.uint8)

        # 深度伪彩
        norm = (depth.cpu().numpy()-near)/(far-near)
        norm = np.clip(norm,0,1)
        try:
            cmap = matplotlib.colormaps.get(local_args.colormap)
        except Exception:
            cmap = matplotlib.cm.get_cmap(local_args.colormap)
        depth_rgb = (cmap(norm)[...,:3]*255).astype(np.uint8)

        # 可选 overlay
        if local_args.overlay:
            rgb_overlay = apply_overlay(rgb_uint8, depth, near, far, local_args.overlay_thresh)
        else:
            rgb_overlay = rgb_uint8

        combined = np.concatenate([rgb_overlay, depth_rgb], axis=1)
        combined_path = os.path.join(local_args.save_dir, f'combined_{frame:04d}.png')
        Image.fromarray(combined).save(combined_path)

        # 统计
        rgb_min = int(rgb_uint8.min())
        rgb_max = int(rgb_uint8.max())
        rgb_mean = float(rgb_uint8.mean())
        depth_valid = depth[torch.isfinite(depth)]
        depth_min = float(depth_valid.min().item()) if depth_valid.numel()>0 else float('nan')
        depth_max = float(depth_valid.max().item()) if depth_valid.numel()>0 else float('nan')
        depth_mean = float(depth_valid.mean().item()) if depth_valid.numel()>0 else float('nan')
        writer.writerow([frame,rgb_min,rgb_max,f"{rgb_mean:.3f}",f"{depth_min:.3f}",f"{depth_max:.3f}",f"{depth_mean:.3f}"])

        print(f"[Frame {frame}] saved: {os.path.basename(rgb_path)}, {os.path.basename(depth_path)}, {os.path.basename(combined_path)}")

        if not local_args.no_show:
            ax_rgb.clear(); ax_depth.clear()
            ax_rgb.imshow(rgb_uint8); ax_rgb.set_title(f'RGB {frame}') ; ax_rgb.axis('off')
            ax_depth.imshow(depth_rgb); ax_depth.set_title(f'Depth {frame}') ; ax_depth.axis('off')
            fig.suptitle(f'RGB & Depth Combined - frame {frame}\nObstacle front +Y 0.8m | overlay={local_args.overlay}')
            plt.pause(0.01)

    try:
        csv_file.flush(); csv_file.close()
    except Exception:
        pass

    if not local_args.no_show:
        plt.ioff(); plt.show()

    print(f"[Done] Combined capture finished. Stats in: {stats_path}")


if __name__ == '__main__':
    main()
