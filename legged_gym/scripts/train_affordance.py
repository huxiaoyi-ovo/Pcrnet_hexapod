"""
Affordance Estimator 训练脚本 (V3)

功能:
1. 加载合成数据集 (AffordanceDataset).
2. 训练 AffordanceEstimator 网络 (ResNet-18 Encoder + 2 Heads).
3. 监控核心指标: Occupancy BCE, Passable Gap BCE.
4. 可视化验证: 自动保存每个 Epoch 的预测对比图.

用法:
   1.  准备数据:
    
    ```bash
    python datasets/affordance_dataset.py --num_samples 20000 --save_dir data/processed
    ```

2.  开始训练:
    在项目根目录下运行：
    ```bash
    python scripts/train_affordance.py \
        --data_path data/processed/affordance_data.pt \
        --batch_size 64 \
        --epochs 50 \
        --output_dir outputs/train_phase1
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到 path，以便导入 modules 和 datasets
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets.affordance_dataset import AffordanceDataset

# 直接导入 affordance_estimator 模块，避免触发 envs/__init__.py 中的 Isaac Gym 导入
import importlib.util
_estimator_path = os.path.join(os.path.dirname(__file__), '..', 'envs', 'hex_v4', 'affordance_estimator.py')
_spec = importlib.util.spec_from_file_location("affordance_estimator", _estimator_path)
_estimator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_estimator_module)
AffordanceEstimator = _estimator_module.AffordanceEstimator
AffordanceLoss = _estimator_module.AffordanceLoss


# ==========================================
# 配置类 (Configuration)
# ==========================================

class TrainConfig:
    def __init__(self, args):
        self.data_path = args.data_path
        self.output_dir = args.output_dir
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_workers = args.num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 损失权重配置
        self.loss_weights = {
            'occupancy': 10.0,      # 占用预测比较稀疏，权重加大
            'passable_gap': 5.0,
        }
        
        # 数据参数 (对应合成数据脚本)
        self.depth_range = 1.0      # 合成数据已归一化到 [0, 1]

# ==========================================
# 辅助函数 (Utils)
# ==========================================

def save_visualization(
    depth: torch.Tensor, 
    targets: Dict[str, torch.Tensor], 
    preds: Dict[str, torch.Tensor], 
    epoch: int, 
    save_dir: str,
    suffix: str = ""
):
    """
    生成验证集的可视化对比图 (V3 视觉一致性检查)
    展示: Depth | GT Occ | Pred Occ | GT Gap | Pred Gap
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取 Batch 中的第一个样本转为 Numpy
    img_depth = depth[0, 0].cpu().numpy()
    gt_occ = targets['occupancy'][0].cpu().numpy()
    pred_occ = preds['occupancy'][0].detach().cpu().numpy()
    gt_gap = targets['passable_gap'][0].cpu().numpy()
    pred_gap = preds['passable_gap'][0].detach().cpu().numpy()

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Depth Input
    ax = axes[0, 0]
    im = ax.imshow(img_depth, cmap='viridis', vmin=0, vmax=1)
    ax.set_title("Input Depth")
    plt.colorbar(im, ax=ax)
    
    # 2. Occupancy (GT vs Pred)
    ax = axes[0, 1]
    ax.imshow(gt_occ, cmap='gray', vmin=0, vmax=1)
    ax.set_title("GT Occupancy")
    
    ax = axes[0, 2]
    ax.imshow(pred_occ, cmap='gray', vmin=0, vmax=1)
    ax.set_title("Pred Occupancy")
    
    # 3. Passable Gap (GT vs Pred)
    ax = axes[1, 1]
    ax.imshow(gt_gap, cmap='Greens', vmin=0, vmax=1)
    ax.set_title("GT Passable Gap")
    
    ax = axes[1, 2]
    ax.imshow(pred_gap, cmap='Greens', vmin=0, vmax=1)
    ax.set_title("Pred Passable Gap")
    
    # 4. Info Panel
    ax = axes[1, 0]
    occ_err = float(np.mean(np.abs(gt_occ - pred_occ)))
    gap_err = float(np.mean(np.abs(gt_gap - pred_gap)))
    text_info = (
        f"Epoch: {epoch}\n"
        f"Occ MAE: {occ_err:.4f}\n"
        f"Gap MAE: {gap_err:.4f}"
    )
    ax.text(0.5, 0.5, text_info, ha='center', va='center', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    filename = f"val_epoch_{epoch:03d}_{suffix}.png" if suffix else f"val_epoch_{epoch:03d}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# ==========================================
# 训练与验证循环
# ==========================================

def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: AffordanceLoss, 
    optimizer: optim.Optimizer, 
    cfg: TrainConfig
) -> Dict[str, float]:
    
    model.train()
    total_loss = 0
    metrics = {'loss_occ': 0, 'loss_gap': 0}
    
    for batch_idx, batch in enumerate(loader):
        # 1. 数据搬运
        depth = batch['depth'].to(cfg.device)
        targets = {
            'occupancy': batch['occupancy'].to(cfg.device),
            'passable_gap': batch['passable_gap'].to(cfg.device),
        }
        
        # 2. 前向传播
        # 数据已经是 [0, 1] 范围，不需要再次归一化
        preds = model(depth, normalize=False)
        
        # 3. 计算损失
        loss_dict = criterion(preds, targets)
        loss = loss_dict['total']
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 记录
        total_loss += loss.item()
        metrics['loss_occ'] += loss_dict['loss_occupancy'].item()
        metrics['loss_gap'] += loss_dict['loss_passable_gap'].item()
    
    # 平均化
    num_batches = len(loader)
    return {
        'total': total_loss / num_batches,
        'loss_occ': metrics['loss_occ'] / num_batches,
        'loss_gap': metrics['loss_gap'] / num_batches,
    }

@torch.no_grad()
def validate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: AffordanceLoss, 
    cfg: TrainConfig,
    epoch: int,
    visualize: bool = False
) -> Dict[str, float]:
    
    model.eval()
    total_loss = 0
    gap_ratio_sum = 0
    occ_ratio_sum = 0
    
    # 收集所有样本用于筛选最高难度
    all_samples = [] if visualize else None
    
    for batch_idx, batch in enumerate(loader):
        depth = batch['depth'].to(cfg.device)
        targets = {
            'occupancy': batch['occupancy'].to(cfg.device),
            'passable_gap': batch['passable_gap'].to(cfg.device),
        }
        
        preds = model(depth, normalize=False)
        loss_dict = criterion(preds, targets)
        
        total_loss += loss_dict['total'].item()
        gap_ratio_sum += targets['passable_gap'].mean().item()
        occ_ratio_sum += targets['occupancy'].mean().item()
        
        # 收集样本信息用于后续筛选最高难度
        if visualize:
            for i in range(depth.size(0)):
                all_samples.append({
                    'depth': depth[i:i+1],
                    'targets': {
                        'occupancy': targets['occupancy'][i:i+1],
                        'passable_gap': targets['passable_gap'][i:i+1],
                    },
                    'preds': {
                        'occupancy': preds['occupancy'][i:i+1],
                        'passable_gap': preds['passable_gap'][i:i+1],
                    },
                    'gap_ratio': targets['passable_gap'][i].mean().item()
                })
    
    # 可视化: 保存平地和最高难度的样本
    if visualize and all_samples:
        # 按可通行间距比例排序
        all_samples.sort(key=lambda x: x['gap_ratio'])
        
        # 间距最少 (更拥挤)
        tight_sample = all_samples[0]
        save_visualization(
            tight_sample['depth'], 
            tight_sample['targets'], 
            tight_sample['preds'], 
            epoch, 
            os.path.join(cfg.output_dir, "viz_val"),
            suffix="tight"
        )
        
        # 间距最多
        open_sample = all_samples[-1]
        save_visualization(
            open_sample['depth'], 
            open_sample['targets'], 
            open_sample['preds'], 
            epoch, 
            os.path.join(cfg.output_dir, "viz_val"),
            suffix="open"
        )
            
    num_batches = len(loader)
    return {
        'val_loss': total_loss / num_batches,
        'val_gap_ratio': gap_ratio_sum / num_batches,
        'val_occ_ratio': occ_ratio_sum / num_batches,
    }

# ==========================================
# 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Affordance Estimator (Occ + Gap)")
    parser.add_argument('--data_path', type=str, required=True, help='Path to .pt dataset file')
    parser.add_argument('--output_dir', type=str, default='outputs/train_v3', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    cfg = TrainConfig(args)
    
    # 1. 准备目录与日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 更新 cfg.output_dir 为实际的 run_dir，供后续可视化使用
    cfg.output_dir = run_dir
    
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    
    print(f"[Train] Output directory: {run_dir}")
    print(f"[Train] Device: {cfg.device}")

    # 2. 数据加载
    print(f"[Train] Loading dataset from {cfg.data_path}...")
    full_dataset = AffordanceDataset(data_path=cfg.data_path, transform=True)
    
    # 划分训练/验证集 (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 关闭验证集的数据增强 (transform逻辑在dataset内部，这里简单起见，假设 dataset 默认为 True
    # 严格来说应该实例化两个 Dataset 对象，但这需要修改 Dataset 类接口或重新加载。
    # Phase 1 预训练可以容忍验证集有少量 Augmentation。)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    print(f"[Train] Dataset splitted: {train_size} train, {val_size} val")

    # 3. 模型初始化
    # 注意: 因为合成数据集已经是 [0, 1]，所以 max_depth_range 设为 1.0
    model = AffordanceEstimator(depth_channels=1, max_depth_range=1.0).to(cfg.device)
    
    # 4. 损失函数与优化器
    criterion = AffordanceLoss(
        occupancy_weight=cfg.loss_weights['occupancy'],
        passable_gap_weight=cfg.loss_weights['passable_gap'],
    ).to(cfg.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # 5. 从checkpoint恢复 (如果指定)
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        print(f"[Train] Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"[Train] Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
    
    print(f"[Train] Start training for {cfg.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        # --- Train ---
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg)
        
        # --- Validate ---
        val_metrics = validate(model, val_loader, criterion, cfg, epoch, visualize=True)
        
        # --- Logging ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Print
        print(f"Epoch [{epoch}/{cfg.epochs}] "
              f"Train Loss: {train_metrics['total']:.4f} (Occ: {train_metrics['loss_occ']:.3f}) | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Gap Ratio: {val_metrics['val_gap_ratio']:.3f} | "
              f"LR: {current_lr:.2e}")
        
        # Tensorboard
        writer.add_scalar('Loss/Train', train_metrics['total'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metric/Gap_Ratio', val_metrics['val_gap_ratio'], epoch)
        writer.add_scalar('Metric/Occ_Ratio', val_metrics['val_occ_ratio'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # --- Checkpoint ---
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_path = os.path.join(run_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(args)
            }, save_path)
            print(f"  -> Best model saved to {save_path}")

    total_time = time.time() - start_time
    print(f"\n[Train] Done! Total time: {total_time/60:.1f} min.")
    print(f"[Train] Best Val Loss: {best_val_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    main()



