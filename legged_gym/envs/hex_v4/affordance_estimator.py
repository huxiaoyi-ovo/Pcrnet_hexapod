"""
Affordance 感知网络（双通道版本）

功能:
1. Depth Encoder: ResNet-18 风格的共享特征提取器。
2. Multi-Head Decoders:
   - Occupancy Head: 预测障碍物占用 (16x16)
   - Passable Gap Head: 预测可通行间距 (16x16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
# 基础模块 
class ResNetBlock(nn.Module):
    """ResNet 基础残差块 (Pre-activation 结构)"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DepthEncoder(nn.Module):
    """
    深度图编码器 (ResNet-18 Modified)
    Input: (B, 1, 128, 128)
    Output: (B, 256, 8, 8)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Initial Conv: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool: 64x64 -> 32x32
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        # Layer 1: 32x32 -> 32x32
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        # Layer 2: 32x32 -> 16x16
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        # Layer 3: 16x16 -> 8x8
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.out_channels = 256
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 解码头 (Decoders)

class OccupancyHead(nn.Module):
    """
    占用预测头
    Input: (B, 256, 8, 8)
    Output: (B, 16, 16) Probability [0,1]
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            # Upsample: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Refine
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.decoder(features)
        return out.squeeze(1)


class PassableGapHead(nn.Module):
    """
    可通行间距预测头
    Input: (B, 256, 8, 8)
    Output: (B, 16, 16) Score [0,1]
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            # Upsample: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Refine
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.decoder(features)
        return out.squeeze(1)


class LowObstacleHead(nn.Module):
    """
    可跨越低障预测头
    Input: (B, 256, 8, 8)
    Output: (B, 16, 16) Score [0,1]
    """
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            # Upsample: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Refine
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.decoder(features)
        return out.squeeze(1)


# 主网络

class AffordanceEstimator(nn.Module):
    """
    Affordance Estimator (Main Module)
    """
    def __init__(self, depth_channels: int = 1, max_depth_range: float = 5.0, output_size: int = 16):
        super().__init__()
        self.max_depth_range = max_depth_range
        if output_size != 16:
            raise ValueError(f"Only output_size=16 is supported, got {output_size}.")

        # Encoder
        self.encoder = DepthEncoder(in_channels=depth_channels)
        
        # Heads
        self.occupancy_head = OccupancyHead(self.encoder.out_channels)
        self.passable_gap_head = PassableGapHead(self.encoder.out_channels)
        self.low_obstacle_head = LowObstacleHead(self.encoder.out_channels)

    def forward(
        self, 
        depth: torch.Tensor, 
        return_features: bool = False,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            depth: (B, 1, 128, 128) 深度图
            return_features: 是否返回中间特征
            normalize: 是否执行 V3 的输入归一化 (Input Normalization)
        
        Returns:
            Dict containing 'occupancy', 'passable_gap', 'low_obstacle'
        """
        
        # V3 CODE OPTIMIZATION: Input Normalization
        # 如果输入是原始米单位 (0-5m)，则归一化到 [0, 1]。
        # 如果训练数据已经是 [0, 1] (合成数据)，则 normalize 应设为 False，或 max_depth_range 设为 1.0。
        # 为了鲁棒性，这里默认执行 clamp，防止异常值。
        x = depth
        if normalize:
            x = torch.clamp(x, 0.0, self.max_depth_range)
            if self.max_depth_range > 1.0:
                x = x / self.max_depth_range

        # Encode
        features = self.encoder(x)

        # Decode
        occupancy = self.occupancy_head(features)
        passable_gap = self.passable_gap_head(features)
        low_obstacle = self.low_obstacle_head(features)

        outputs = {
            'occupancy': occupancy,             # (B, 16, 16)
            'passable_gap': passable_gap,       # (B, 16, 16)
            'low_obstacle': low_obstacle,       # (B, 16, 16)
        }

        if return_features:
            outputs['features'] = features

        return outputs


# 损失函数

class AffordanceLoss(nn.Module):
    """
    三通道损失函数
    L_total = w1 * L_occ + w2 * L_gap + w3 * L_low
    """
    def __init__(
        self,
        occupancy_weight: float = 1.0,
        passable_gap_weight: float = 1.0,
        low_obstacle_weight: float = 1.0,
    ):
        super().__init__()
        self.weights = {
            'occupancy': occupancy_weight,
            'passable_gap': passable_gap_weight,
            'low_obstacle': low_obstacle_weight,
        }
        
        # Loss Criteria
        self.bce_loss = nn.BCELoss()  # For Occupancy & Passable Gap (0/1 probability)

    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Occupancy Loss
        l_occ = self.bce_loss(predictions['occupancy'], targets['occupancy'])

        # 2. Passable Gap Loss
        l_gap = self.bce_loss(predictions['passable_gap'], targets['passable_gap'])

        # 3. Low Obstacle Loss
        l_low = self.bce_loss(predictions['low_obstacle'], targets['low_obstacle'])

        # Weighted Sum
        total_loss = (
            self.weights['occupancy'] * l_occ +
            self.weights['passable_gap'] * l_gap +
            self.weights['low_obstacle'] * l_low
        )
        
        return {
            'total': total_loss,
            'loss_occupancy': l_occ,
            'loss_passable_gap': l_gap,
            'loss_low_obstacle': l_low,
        }


# 自测模块 


if __name__ == "__main__":
    print("[Test] Initializing AffordanceEstimator...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate Model
    model = AffordanceEstimator(max_depth_range=1.0).to(device) # dataset is 0-1
    print(f"       Device: {device}")
    
    # 2. Create Dummy Input
    batch_size = 4
    dummy_depth = torch.rand(batch_size, 1, 128, 128).to(device)
    
    # 3. Forward Pass
    print("[Test] Running forward pass...")
    outputs = model(dummy_depth, return_features=True, normalize=True)
    
    # 4. Check Shapes
    print("\n[Check] Output Shapes:")
    for k, v in outputs.items():
        print(f"       {k:<20}: {list(v.shape)}")
        
    assert outputs['occupancy'].shape == (batch_size, 16, 16)
    assert outputs['passable_gap'].shape == (batch_size, 16, 16)
    assert outputs['low_obstacle'].shape == (batch_size, 16, 16)
    
    # 5. Test Loss
    print("\n[Test] Calculating loss...")
    criterion = AffordanceLoss()
    targets = {
        'occupancy': torch.randint(0, 2, (batch_size, 16, 16)).float().to(device),
        'passable_gap': torch.randint(0, 2, (batch_size, 16, 16)).float().to(device),
        'low_obstacle': torch.randint(0, 2, (batch_size, 16, 16)).float().to(device),
    }
    
    losses = criterion(outputs, targets)
    for k, v in losses.items():
        print(f"       {k:<20}: {v.item():.4f}")
        
    print("\n[Success] AffordanceEstimator V3 is ready for training.")
