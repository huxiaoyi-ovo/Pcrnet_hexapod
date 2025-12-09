# -*- coding: utf-8 -*-
"""
datasets/affordance_dataset.py - Affordance 任务数据集生成与加载脚本 (V3 完整版)

功能:
1. Procedural Generation: 生成合成的六足机器人地形导航数据 (Depth Images + Labels)。
2. PyTorch Dataset: 提供标准的 Dataset 类，支持数据增强。
3. V3 Heuristics: 模拟 V3 方案中的 'Self-Supervised' 难度标签生成过程（Phase 1 几何近似）。

依赖:
    pip install torch numpy scipy tqdm matplotlib

用法:
    1. 生成数据: 
       python datasets/affordance_dataset.py --num_samples 20000 --save_dir data/processed --visualize
    
    2. 训练调用: 
       from datasets.affordance_dataset import AffordanceDataset
       dataset = AffordanceDataset(data_path='data/processed/affordance_data.pt', transform=True)
"""

import os
import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 设置随机种子以保证可复现性 (仅影响生成过程，训练时通常由 Trainer 控制)
# np.random.seed(42)
# torch.manual_seed(42)

# 常量定义 (地形生成参数)
IMG_SIZE = 128
MAP_SIZE = 16
MAX_DEPTH_RANGE = 5.0  # 假设深度相机最大感知距离为 5.0 米

# 地形类型概率分布
TERRAIN_PROBS = {
    'flat': 0.2,      # 平地
    'obstacle': 0.3,  # 离散障碍物
    'slope': 0.2,     # 斜坡
    'rough': 0.3      # 崎岖不平
}

# 难度计算权重 (V3 Phase 1 Heuristic)
DIFF_WEIGHT_BASE = 0.4
DIFF_WEIGHT_OBS = 0.4
DIFF_WEIGHT_ROUGH = 2.0


class AffordanceDataset(Dataset):
    """
    六足机器人 Affordance 数据集 (V3)
    
    包含:
    - 输入: Depth Image (1, 128, 128) -> 归一化到 [0, 1]
    - 输出: 
        1. Occupancy Map (16, 16) -> [0, 1] 概率
        2. Traversability Map (16, 16) -> [0, 1] 分数
        3. Terrain Difficulty (1,) -> [0, 1] 标量
    """

    def __init__(self, data_path: str = None, transform: bool = False):
        """
        Args:
            data_path (str): 已生成数据的路径 (.pt 文件).
            transform (bool): 是否启用随机数据增强 (Flip/Rotate).
        """
        self.transform = transform
        self.data = []
        
        if data_path:
            if os.path.exists(data_path):
                print(f"[Dataset] Loading dataset from {data_path}...")
                try:
                    loaded_data = torch.load(data_path)
                    if isinstance(loaded_data, dict) and 'samples' in loaded_data:
                        self.data = loaded_data['samples']
                    else:
                        # 兼容直接保存 list 的情况
                        self.data = loaded_data
                    print(f"[Dataset] Successfully loaded {len(self.data)} samples.")
                except Exception as e:
                    print(f"[Dataset] Error loading data: {e}")
                    raise e
            else:
                raise FileNotFoundError(f"[Dataset] Data file not found at {data_path}. Please run generation first.")
        else:
            print("[Dataset] Initialized empty dataset. Use for generation only.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本，并执行转换和增强
        """
        sample = self.data[idx]
        
        # 1. 解包数据 (保持原始 numpy float32)
        depth = sample['depth']                 # (128, 128)
        occupancy = sample['occupancy']         # (16, 16)
        traversability = sample['traversability'] # (16, 16)
        difficulty = sample['difficulty']       # (1,)

        # 2. 转换为 Tensor
        depth_t = torch.from_numpy(depth).float()
        occ_t = torch.from_numpy(occupancy).float()
        trav_t = torch.from_numpy(traversability).float()
        diff_t = torch.from_numpy(difficulty).float()

        # 3. 增加通道维度 (C, H, W) -> (1, 128, 128)
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0)

        # 4. 数据增强 (Data Augmentation)
        # 注意: 这是一个空间映射任务，Depth翻转时，Map Label 也必须翻转
        if self.transform:
            # Random Horizontal Flip
            if random.random() > 0.5:
                depth_t = torch.flip(depth_t, [-1])
                occ_t = torch.flip(occ_t, [-1])
                trav_t = torch.flip(trav_t, [-1])
            
            # Random Vertical Flip (对于俯视/前视地图也是合理的增强)
            if random.random() > 0.5:
                depth_t = torch.flip(depth_t, [-2])
                occ_t = torch.flip(occ_t, [-2])
                trav_t = torch.flip(trav_t, [-2])

            # Random Rotation 90 deg (可选，增加鲁棒性)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                depth_t = torch.rot90(depth_t, k, [-2, -1])
                occ_t = torch.rot90(occ_t, k, [-2, -1])
                trav_t = torch.rot90(trav_t, k, [-2, -1])

        # 5. 返回字典
        return {
            'depth': depth_t,
            'occupancy': occ_t,
            'traversability': trav_t,
            'terrain_difficulty': diff_t
        }


# 合成数据生成核心逻辑 (Procedural Generation)

def generate_terrain_depth(img_size: int = 128) -> Tuple[np.ndarray, str, float]:
    """
    生成单一地形的高度图/深度图
    Returns:
        depth_map: (img_size, img_size) float32, range [0, 1]
        terrain_type: str
        base_diff: float (基础难度系数)
    """
    depth_map = np.zeros((img_size, img_size), dtype=np.float32)
    
    # 根据概率选择地形
    types = list(TERRAIN_PROBS.keys())
    probs = list(TERRAIN_PROBS.values())
    terrain_type = np.random.choice(types, p=probs)
    
    base_diff = 0.0

    if terrain_type == 'flat':
        # 平地：极低噪声
        depth_map += np.random.normal(0, 0.005, (img_size, img_size))
        base_diff = 0.05
        
    elif terrain_type == 'obstacle':
        # 障碍物：随机放置矩形块 (模拟石头、箱子)
        # 基础底噪
        depth_map += np.random.normal(0, 0.01, (img_size, img_size))
        
        num_obs = np.random.randint(3, 12)
        for _ in range(num_obs):
            # 随机位置和大小
            cx, cy = np.random.randint(0, img_size, 2)
            w = np.random.randint(5, 20) # width radius
            h = np.random.randint(5, 20) # height radius
            
            # 随机高度 (0.2 ~ 0.8)
            height = np.random.uniform(0.2, 0.8)
            
            # 叠加
            x_min, x_max = max(0, cx-w), min(img_size, cx+w)
            y_min, y_max = max(0, cy-h), min(img_size, cy+h)
            depth_map[x_min:x_max, y_min:y_max] = np.maximum(depth_map[x_min:x_max, y_min:y_max], height)
            
        base_diff = 0.6
            
    elif terrain_type == 'slope':
        # 斜坡：生成随机梯度的平面
        x = np.linspace(0, 1, img_size)
        y = np.linspace(0, 1, img_size)
        X, Y = np.meshgrid(x, y)
        
        # 随机坡度方向
        theta = np.random.uniform(0, 2*np.pi)
        slope_steepness = np.random.uniform(0.3, 1.0)
        
        plane = (X * np.cos(theta) + Y * np.sin(theta)) * slope_steepness
        # 归一化偏移，保证大部分在 [0,1]
        plane = plane - np.min(plane)
        plane = plane / (np.max(plane) + 1e-6) * slope_steepness
        
        depth_map += plane
        depth_map += np.random.normal(0, 0.01, (img_size, img_size))
        base_diff = 0.3 + slope_steepness * 0.4
        
    elif terrain_type == 'rough':
        # 崎岖地形：高斯滤波后的噪声 (Perlin-like)
        noise = np.random.rand(img_size, img_size)
        # 使用不同的 sigma 混合不同频率的噪声
        low_freq = gaussian_filter(noise, sigma=8.0)
        high_freq = gaussian_filter(np.random.rand(img_size, img_size), sigma=2.0)
        
        depth_map = low_freq * 0.7 + high_freq * 0.3
        # 归一化对比度拉伸
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-6)
        # 随机高度缩放
        depth_map *= np.random.uniform(0.4, 0.9)
        
        base_diff = 0.8

    # 最终截断和归一化 (模拟传感器输出范围)
    depth_map = np.clip(depth_map, 0.0, 1.0)
    
    return depth_map, terrain_type, base_diff


def compute_labels_from_depth(
    depth_map: np.ndarray, 
    map_size: int = 16,
    base_difficulty: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    根据深度图计算几何标签和 V3 难度近似值
    
    Args:
        depth_map: (128, 128)
        map_size: 输出地图尺寸 (16)
        base_difficulty: 地形类型的基础难度
    
    Returns:
        occupancy: (16, 16)
        traversability: (16, 16)
        difficulty: float
    """
    img_size = depth_map.shape[0]
    block_size = img_size // map_size
    
    occupancy = np.zeros((map_size, map_size), dtype=np.float32)
    traversability = np.zeros((map_size, map_size), dtype=np.float32)
    
    # 计算局部统计量
    # 我们可以使用 view_as_blocks 或者简单的 reshape 来加速，
    # 但为了逻辑清晰，这里使用循环，对于生成脚本来说性能足够。
    
    total_roughness = 0.0
    
    for i in range(map_size):
        for j in range(map_size):
            # 提取 8x8 的 patch
            r_start, r_end = i*block_size, (i+1)*block_size
            c_start, c_end = j*block_size, (j+1)*block_size
            patch = depth_map[r_start:r_end, c_start:c_end]
            
            # 统计特征
            h_min = np.min(patch)
            h_max = np.max(patch)
            h_std = np.std(patch)
            h_mean = np.mean(patch)
            
            # 1. Occupancy 逻辑
            # 判定条件：高度差过大(台阶) 或 局部粗糙度过高
            step_height = h_max - h_min
            
            # 阈值：标准差 > 0.05 或 高度差 > 0.3 视为障碍
            if h_std > 0.05 or step_height > 0.25:
                occupancy[i, j] = 1.0
            else:
                occupancy[i, j] = 0.0
            
            # 2. Traversability 逻辑
            # 越平滑分数越高。Traversability = 1.0 - normalized_roughness
            # 主要受标准差影响
            roughness_score = np.clip(h_std * 8.0, 0.0, 1.0)
            traversability[i, j] = 1.0 - roughness_score
            
            total_roughness += h_std

    # 3. V3 Terrain Difficulty Calculation (Heuristic)
    # 公式: D = w1 * Base + w2 * ObsRatio + w3 * GlobalRoughness
    
    obs_ratio = np.mean(occupancy)
    global_roughness = np.std(depth_map) # 全局标准差
    
    # 组合
    difficulty = (
        base_difficulty * DIFF_WEIGHT_BASE + 
        obs_ratio * DIFF_WEIGHT_OBS + 
        global_roughness * DIFF_WEIGHT_ROUGH
    )
    
    # 添加随机扰动 (模拟物理测量的不确定性)
    difficulty += np.random.normal(0, 0.03)
    
    # 严格截断到 [0, 1]
    difficulty = np.clip(difficulty, 0.0, 1.0)
    
    return occupancy, traversability, difficulty


def generate_synthetic_sample(img_size: int = 128, map_size: int = 16) -> Dict[str, np.ndarray]:
    """生成单个完整样本的封装函数"""
    
    # 1. 生成地形
    depth, t_type, base_diff = generate_terrain_depth(img_size)
    
    # 2. 计算标签
    occ, trav, diff_val = compute_labels_from_depth(depth, map_size, base_diff)
    
    return {
        'depth': depth.astype(np.float32),
        'occupancy': occ.astype(np.float32),
        'traversability': trav.astype(np.float32),
        'difficulty': np.array([diff_val], dtype=np.float32),
        'type': t_type # 仅用于调试或可视化
    }


# 主程序与工具函数

def visualize_samples(samples: List[Dict], save_path: str = "data_preview.png"):
    """
    可视化生成的样本，用于人工检查
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用无界面backend
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[Warning] Matplotlib import error: {e}. Skipping visualization.")
        return
    except Exception as e:
        print(f"[Warning] Matplotlib error: {e}. Skipping visualization.")
        import traceback
        traceback.print_exc()
        return

    num_show = min(5, len(samples))
    fig, axes = plt.subplots(num_show, 4, figsize=(16, 4 * num_show))
    
    if num_show == 1: axes = [axes] # Handle single sample case
    
    for i in range(num_show):
        s = samples[i]
        
        # Depth
        ax = axes[i][0] if num_show > 1 else axes[0]
        im = ax.imshow(s['depth'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Depth ({s['type']})")
        fig.colorbar(im, ax=ax)
        
        # Occupancy
        ax = axes[i][1] if num_show > 1 else axes[1]
        ax.imshow(s['occupancy'], cmap='gray', vmin=0, vmax=1)
        ax.set_title("Occupancy")
        
        # Traversability
        ax = axes[i][2] if num_show > 1 else axes[2]
        ax.imshow(s['traversability'], cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("Traversability")
        
        # Info
        ax = axes[i][3] if num_show > 1 else axes[3]
        ax.text(0.1, 0.5, f"Difficulty: {s['difficulty'][0]:.2f}\nType: {s['type']}", fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"[Viz] Preview saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate V3 Affordance Dataset (Production Ready)")
    parser.add_argument('--num_samples', type=int, default=20000, help='Number of samples to generate')
    parser.add_argument('--save_dir', type=str, default='data/processed', help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='affordance_data.pt', help='Output filename')
    parser.add_argument('--visualize', action='store_true', help='Generate a preview image of first 5 samples')
    args = parser.parse_args()

    # 1. 路径检查
    if not os.path.exists(args.save_dir):
        print(f"[Main] Creating directory: {args.save_dir}")
        os.makedirs(args.save_dir, exist_ok=True)
    
    save_path = os.path.join(args.save_dir, args.filename)
    
    # 2. 生成循环
    print(f"[Main] Starting generation of {args.num_samples} samples...")
    print(f"[Main] Config: Depth {IMG_SIZE}x{IMG_SIZE} -> Map {MAP_SIZE}x{MAP_SIZE}")
    
    samples = []
    
    # 使用 tqdm 显示进度
    for _ in tqdm(range(args.num_samples), desc="Generating", unit="sample"):
        # 生成样本 (去除 debug 用的 type 字段以节省空间，或者保留看需求)
        s = generate_synthetic_sample(IMG_SIZE, MAP_SIZE)
        
        # 如果不需要 type 字段进入 Dataset，可以在这里 pop 掉
        # s.pop('type') 
        # 但保留它对于分析数据分布很有用，占空间很小，建议保留
        
        samples.append(s)

    # 3. 统计信息
    diffs = [s['difficulty'][0] for s in samples]
    types = [s['type'] for s in samples]
    
    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Total Samples: {len(samples)}")
    print(f"Difficulty:")
    print(f"  Mean: {np.mean(diffs):.4f}")
    print(f"  Std:  {np.std(diffs):.4f}")
    print(f"  Min:  {np.min(diffs):.4f}")
    print(f"  Max:  {np.max(diffs):.4f}")
    print(f"Terrain Distribution:")
    for t in TERRAIN_PROBS.keys():
        count = types.count(t)
        print(f"  {t}: {count} ({count/len(samples)*100:.1f}%)")
    print("="*40)

    # 4. 可视化 (可选)
    if args.visualize:
        viz_path = os.path.join(args.save_dir, "dataset_preview.png")
        visualize_samples(samples, viz_path)

    # 5. 保存
    print(f"\n[Main] Saving dataset to {save_path}...")
    # 保存为一个字典，方便后续扩展 meta 信息
    torch.save({
        'samples': samples,
        'meta': {
            'num_samples': args.num_samples,
            'img_size': IMG_SIZE,
            'map_size': MAP_SIZE,
            'version': 'v3.0'
        }
    }, save_path)
    
    print("[Main] Done! Dataset ready for Phase 1 training.")


if __name__ == "__main__":
    main()