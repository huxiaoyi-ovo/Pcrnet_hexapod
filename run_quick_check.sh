#!/bin/bash
# Quick Check执行脚本

echo "============================================================"
echo "科研级Quick Check - 开始执行"
echo "============================================================"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda未找到，尝试直接运行..."
    python3 quick_check_camera.py
    exit $?
fi

# 激活环境并运行
echo "激活hexapod_rl_env环境..."
source ~/.bashrc
conda activate hexapod_rl_env 2>/dev/null || source activate hexapod_rl_env 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ 环境激活成功"
    python quick_check_camera.py
else
    echo "⚠️  环境激活失败，尝试直接运行..."
    python3 quick_check_camera.py
fi
