#!/bin/bash
# 清理GPU显存脚本

echo "========================================"
echo "清理GPU显存"
echo "========================================"
echo ""
echo "当前GPU使用情况："
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 杀死VS Code GPU进程（保留主进程）
echo "关闭VS Code GPU进程..."
pkill -f "code.*gpu-process" 2>/dev/null || true

# 等待进程清理
sleep 2

echo ""
echo "清理后的GPU使用情况："
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 清理PyTorch缓存
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"PyTorch CUDA cache cleared")
EOF

echo ""
echo "清理完成！现在运行深度图像测试："
echo "/home/hxy/anaconda3/envs/hexapod_rl_env/bin/python legged_gym/envs/hex_v4/save_depth_images.py"
