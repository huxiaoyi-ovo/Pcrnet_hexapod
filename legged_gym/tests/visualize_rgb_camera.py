#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGB 相机测试脚本：创建单环境，抓取并保存若干帧 RGB 图像。

示例运行命令:
    # 基本运行（默认分辨率）保存 10 帧
    python legged_gym/envs/hex_v4/visualize_rgb_camera.py --frames 10

    # 自定义分辨率（低分辨率更稳定）
    python legged_gym/envs/hex_v4/visualize_rgb_camera.py --frames 5 --width 128 --height 128
    /home/hxy/anaconda3/envs/hexapod_rl_env/bin/python legged_gym/envs/hex_v4/visualize_rgb_camera.py --frames 3 --width 128 --height 128

    # 保存未归一化 uint8 通道顺序 (H,W,3)
    python legged_gym/envs/hex_v4/visualize_rgb_camera.py --frames 5 --channels_last --no_normalize

输出:
    PNG: logs/camera_visualizations/rgb/rgb_XXX.png
    CSV: logs/camera_visualizations/rgb/rgb_stats.csv

参数:
    --frames N          保存帧数
    --save_dir path     输出目录 (默认 logs/camera_visualizations/rgb)
    --interval N        采样间隔 (预留, 当前不使用)
    --channels_last     返回 (H,W,3) 格式保存
    --no_normalize      不做 [0,1] 归一化 (直接 uint8)
    --width/--height    覆盖分辨率

注意:
    * 512x512 在当前环境可能导致 139 (segfault)，建议使用 <=256 或 128x128。
    * 脚本关闭 PhysX GPU 与 GPU pipeline 以降低显存占用。
"""
import sys
import os
import time
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)
print(f"[Debug] Project root: {project_root}")

from legged_gym.envs.hex_v4.hex_ground import HexGround
from legged_gym.envs.hex_v4.hex_scenes_config import HexDebugPlaneCfg
from legged_gym.utils import get_args, class_to_dict
from legged_gym.utils.helpers import parse_sim_params
import torch
import numpy as np
from PIL import Image
from isaacgym import gymapi

# 固定随机种子，确保与 Depth 脚本的机器人初始姿态一致
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def extract_local_args():
    """从 sys.argv 中抽取本脚本自定义参数并移除，避免 get_args() 报未知参数。"""
    # 自定义参数默认值
    local = {
        'frames': 10,
        'save_dir': 'logs/camera_visualizations/rgb',
        'interval': 1,
        'channels_last': False,
        'no_normalize': False,
        'width': None,
        'height': None,
    }
    argv = sys.argv
    i = 1
    # 收集要删除的 index
    remove_indices = set()
    while i < len(argv):
        tok = argv[i]
        if tok == '--frames' and i + 1 < len(argv):
            local['frames'] = int(argv[i+1]); remove_indices.update({i, i+1}); i += 2; continue
        if tok == '--save_dir' and i + 1 < len(argv):
            local['save_dir'] = argv[i+1]; remove_indices.update({i, i+1}); i += 2; continue
        if tok == '--interval' and i + 1 < len(argv):
            local['interval'] = int(argv[i+1]); remove_indices.update({i, i+1}); i += 2; continue
        if tok == '--channels_last':
            local['channels_last'] = True; remove_indices.add(i); i += 1; continue
        if tok == '--no_normalize':
            local['no_normalize'] = True; remove_indices.add(i); i += 1; continue
        if tok == '--width' and i + 1 < len(argv):
            local['width'] = int(argv[i+1]); remove_indices.update({i, i+1}); i += 2; continue
        if tok == '--height' and i + 1 < len(argv):
            local['height'] = int(argv[i+1]); remove_indices.update({i, i+1}); i += 2; continue
        i += 1
    # 重建 sys.argv (保留顺序)
    if remove_indices:
        sys.argv = [arg for idx, arg in enumerate(argv) if idx not in remove_indices]
    # 简单的 Namespace 替代
    class LocalArgs: pass
    ns = LocalArgs()
    for k, v in local.items():
        setattr(ns, k, v)
    return ns


def prepare_env():
    # 先抽取并剥离本脚本参数，再调用全局 get_args()
    local_args = extract_local_args()
    args = get_args()
    # 尝试关闭 PhysX GPU 与 GPU pipeline 以减少显存占用，只保留图形设备用于相机
    if hasattr(args, 'use_gpu'):
        args.use_gpu = False
    if hasattr(args, 'use_gpu_pipeline'):
        args.use_gpu_pipeline = False
    args.headless = True  # 强制 headless

    # 配置相机与环境
    cfg = HexDebugPlaneCfg()
    cam_cfg = cfg.sensor.depth_camera
    cam_cfg.enable = True
    # 分辨率：若命令行未指定，则采用配置默认（通常为 128x128）
    cam_cfg.width = local_args.width or cam_cfg.width
    cam_cfg.height = local_args.height or cam_cfg.height
    cam_cfg.capture_interval = 10  # 与深度测试保持一致，降低渲染频率
    cfg.env.num_envs = 1
    # 使用 debug plane 任务配置（无需手动修改 mesh_type）
    cfg.terrain.measure_heights = True  # 保持高度测量以兼容噪声向量尺寸
    # 关闭执行器网络以减少显存占用
    cfg.control.use_actuator_net = False

    # 构造仿真参数
    sim_params = {"sim": class_to_dict(cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    env = HexGround(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    return env, local_args


def add_front_obstacle(env, env_id: int = 0, distance: float = 0.8,
                       size_xyz=(0.3, 0.3, 0.2), color=(1.0, 0.0, 0.0)):
    """在机器人正前方添加一个固定障碍物（盒子）。

    参数:
        env_id: 目标环境索引（默认 0）。
        distance: 沿 +Y 方向（机器人正前方）的距离（米）。
        size_xyz: (x, y, z) 尺寸（米）。
        color: (r, g, b) 颜色，范围 0-1。
    """
    if env_id < 0 or env_id >= env.num_envs:
        print(f"[Warn] Invalid env_id {env_id}, num_envs={env.num_envs}")
        return

    # 创建盒子资产（固定基座，作为静态障碍）
    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = True
    box_asset = env.gym.create_box(env.sim, size_xyz[0], size_xyz[1], size_xyz[2], asset_opts)

    # 放置在机器人正前方 +Y 方向
    origin = env.env_origins[env_id]
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(float(origin[0]), float(origin[1] + distance), float(origin[2] + size_xyz[2] * 0.5))
    pose.r = gymapi.Quat(0, 0, 0, 1)

    actor = env.gym.create_actor(env.envs[env_id], box_asset, pose, f"obstacle_{env_id}", env_id, 0)
    print(f"[Obstacle] Added at position: x={origin[0]:.2f}, y={origin[1]+distance:.2f}, z={origin[2]+size_xyz[2]*0.5:.2f}")

    # 给刚体着色（索引 0）
    try:
        env.gym.set_rigid_body_color(env.envs[env_id], actor, 0, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(color[0], color[1], color[2]))
    except Exception as e:
        print(f"[Warn] set_rigid_body_color failed: {e}")


def add_front_pit(env, env_id: int = 0, distance: float = 0.5,
                  pit_size=(0.4, 0.3), pit_depth=0.15, wall_thickness=0.02,
                  color=(0.3, 0.3, 0.3)):
    """在机器人正前方添加一个坑洞效果（用低矮围墙围起来，中间留空）。

    注意: 由于平面地形会挡住地下的物体，我们通过在地面上放置矮墙围成边框，
    中间不放任何物体，这样深度相机会看到更远的地面距离（模拟坑洞效果）。
    
    为了让效果更明显，我们还在机器人与坑洞之间放一段地面，坑洞区域无地面覆盖。

    参数:
        env_id: 目标环境索引（默认 0）。
        distance: 沿 +Y 方向（机器人正前方）到坑洞中心的距离（米）。
        pit_size: (width_x, length_y) 坑洞的水平尺寸（米）。
        pit_depth: 围墙高度（米），用于视觉标识坑洞边界。
        wall_thickness: 墙壁厚度（米）。
        color: (r, g, b) 墙壁颜色，范围 0-1。
    """
    if env_id < 0 or env_id >= env.num_envs:
        print(f"[Warn] Invalid env_id {env_id}, num_envs={env.num_envs}")
        return

    origin = env.env_origins[env_id]
    pit_center_x = float(origin[0])
    pit_center_y = float(origin[1] + distance)
    ground_z = float(origin[2])

    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = True

    width_x, length_y = pit_size
    wall_h = pit_depth

    # 四面矮墙放在地面上 (z = wall_h/2, 即底部在 z=0)
    # 这样形成一个可见的边框，中间是空的
    walls = [
        # (size_x, size_y, size_z, offset_x, offset_y, offset_z)
        # 前墙 (+Y 边) - 绿色
        (width_x + 2*wall_thickness, wall_thickness, wall_h,
         0, length_y/2 + wall_thickness/2, wall_h/2),
        # 后墙 (-Y 边) - 绿色  
        (width_x + 2*wall_thickness, wall_thickness, wall_h,
         0, -length_y/2 - wall_thickness/2, wall_h/2),
        # 左墙 (-X 边)
        (wall_thickness, length_y, wall_h,
         -width_x/2 - wall_thickness/2, 0, wall_h/2),
        # 右墙 (+X 边)
        (wall_thickness, length_y, wall_h,
         width_x/2 + wall_thickness/2, 0, wall_h/2),
    ]
    
    # 墙壁颜色: 前后用绿色，左右用指定颜色
    wall_colors = [
        (0.0, 0.8, 0.0),  # 前墙 - 绿色
        (0.0, 0.8, 0.0),  # 后墙 - 绿色
        color,             # 左墙
        color,             # 右墙
    ]

    for i, (sx, sy, sz, ox, oy, oz) in enumerate(walls):
        box_asset = env.gym.create_box(env.sim, sx, sy, sz, asset_opts)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(pit_center_x + ox, pit_center_y + oy, ground_z + oz)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        actor = env.gym.create_actor(env.envs[env_id], box_asset, pose, f"pit_wall_{env_id}_{i}", env_id, 0)
        c = wall_colors[i]
        try:
            env.gym.set_rigid_body_color(env.envs[env_id], actor, 0, gymapi.MESH_VISUAL,
                                         gymapi.Vec3(c[0], c[1], c[2]))
        except Exception:
            pass

    print(f"[Pit Frame] Added at y={pit_center_y:.2f}, size={pit_size}, wall_height={pit_depth}m (green border marks pit area)")


def add_descending_stairs(env, env_id: int = 0, start_distance: float = 0.3,
                          num_steps: int = 5, step_width: float = 0.6,
                          step_depth: float = 0.15, step_height: float = 0.05,
                          first_step_elevation: float = 0.1,
                          colors=None):
    """在机器人正前方添加向下的阶梯。

    阶梯从机器人前方开始，第一级高出地面，后续台阶逐级下降。

    参数:
        env_id: 目标环境索引（默认 0）。
        start_distance: 第一级台阶起始边沿到机器人的距离（米）。
        num_steps: 台阶数量。
        step_width: 台阶宽度（X 方向，米）。
        step_depth: 每级台阶的深度（Y 方向，米）。
        step_height: 每级台阶的高度差（米）。
        first_step_elevation: 第一级台阶顶部高出地面的高度（米）。
        colors: 每级台阶的颜色列表，None 则使用渐变色。
    """
    if env_id < 0 or env_id >= env.num_envs:
        print(f"[Warn] Invalid env_id {env_id}, num_envs={env.num_envs}")
        return

    origin = env.env_origins[env_id]
    base_x = float(origin[0])
    base_y = float(origin[1])
    ground_z = float(origin[2])

    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = True

    # 默认渐变色：从浅灰到深灰
    if colors is None:
        colors = []
        for i in range(num_steps):
            gray = 0.8 - (i / num_steps) * 0.6  # 从 0.8 渐变到 0.2
            colors.append((gray, gray, gray))

    print(f"[Stairs] Creating {num_steps} descending steps starting at y={base_y + start_distance:.2f}")
    print(f"  First step elevation: {first_step_elevation}m above ground")

    for i in range(num_steps):
        # 每级台阶的位置
        # Y 位置：从 start_distance 开始，每级增加 step_depth
        step_y = base_y + start_distance + step_depth * (i + 0.5)
        
        # Z 位置：第一级顶部高出地面 first_step_elevation，后续逐级下降
        # 台阶 i 的顶部 z = ground_z + first_step_elevation - i * step_height
        step_top_z = ground_z + first_step_elevation - i * step_height
        # 台阶的厚度：确保底部不会低于地面太多
        step_thickness = first_step_elevation + step_height  # 固定厚度
        step_center_z = step_top_z - step_thickness / 2

        # 创建台阶方块
        box_asset = env.gym.create_box(env.sim, step_width, step_depth, step_thickness, asset_opts)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(base_x, step_y, step_center_z)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        actor = env.gym.create_actor(env.envs[env_id], box_asset, pose, 
                                     f"stair_{env_id}_{i}", env_id, 0)
        
        # 设置颜色
        color = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
        try:
            env.gym.set_rigid_body_color(env.envs[env_id], actor, 0, gymapi.MESH_VISUAL,
                                         gymapi.Vec3(color[0], color[1], color[2]))
        except Exception:
            pass

        print(f"  Step {i+1}: y={step_y:.3f}, top_z={step_top_z:.3f}, color=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")

    total_drop = (num_steps - 1) * step_height
    total_length = num_steps * step_depth
    print(f"[Stairs] Total: {num_steps} steps, length={total_length:.2f}m, drop={total_drop:.2f}m")


def save_rgb_tensor(rgb_tensor, path, normalize=True, channels_last=False):
    """保存单张 RGB 图像。
    rgb_tensor: (3,H,W) or (H,W,3), float32 [0,1] 或 uint8 [0,255]
    """
    if not channels_last:
        if rgb_tensor.dim() != 3:
            raise ValueError('Expect (3,H,W) tensor when channels_last=False')
        rgb_np = rgb_tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
    else:
        rgb_np = rgb_tensor.cpu().numpy()

    if normalize and rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
    elif not normalize and rgb_np.dtype != np.uint8:
        # 强制转 uint8 以防类型不符
        rgb_np = rgb_np.astype(np.uint8)

    Image.fromarray(rgb_np).save(path)


def main():
    env, local_args = prepare_env()
    normalize = not local_args.no_normalize
    channels_last = local_args.channels_last

    os.makedirs(local_args.save_dir, exist_ok=True)
    print(f"[Info] Saving RGB frames to: {local_args.save_dir}")
    # 打开 CSV 统计文件
    stats_path = os.path.join(local_args.save_dir, 'rgb_stats.csv')
    csv_file = open(stats_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'shape', 'channels_last', 'normalize', 'min', 'max', 'mean'])

    # 环境复位
    obs_dict = env.reset_separate()
    print("[Info] Environment reset done")

    # 在机器人前方添加一个红色障碍物（默认 0.8m 前方，尺寸 0.3x0.3x0.2m）
    add_front_obstacle(env, env_id=0, distance=0.8, size_xyz=(0.3, 0.3, 0.2), color=(1.0, 0.0, 0.0))
    # 渲染一次以确保相机更新
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    # 不再推进物理步，以避免新增障碍物后 root_state 索引形状不匹配问题。

    captured = 0
    step = 0
    t0 = time.time()
    while captured < local_args.frames:
        # 直接渲染相机并抓取图像（不推进物理）
        rgb = env._get_rgb_images(normalize=normalize, channels_last=channels_last)
        if rgb is None:
            print("[Warn] RGB capture returned None")
            break
        single = rgb[0]
        path = os.path.join(local_args.save_dir, f'rgb_{captured:03d}.png')
        save_rgb_tensor(single, path, normalize=normalize, channels_last=channels_last)
        # 打印统计
        if normalize:
            stats_min = float(single.min().item())
            stats_max = float(single.max().item())
            stats_mean = float(single.mean().item())
        else:
            stats_min = int(single.min().item())
            stats_max = int(single.max().item())
            stats_mean = float(single.float().mean().item())
        shape = tuple(single.shape)
        # 写入 CSV
        csv_writer.writerow([captured, str(shape), channels_last, normalize, stats_min, stats_max, stats_mean])
        print(f"[Frame {captured}] shape={shape} min={stats_min} max={stats_max} mean={stats_mean:.3f} saved={path}")
        captured += 1
        step += 1

    dt = time.time() - t0
    print(f"[Done] Captured {captured} frames in {dt:.2f}s")
    print("示例查看命令:")
    print(f"  eog {local_args.save_dir}/rgb_000.png  (若有桌面环境)")

    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
