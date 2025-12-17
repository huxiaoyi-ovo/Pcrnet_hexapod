#!/bin/bash
# Phase 1 修复版训练脚本
# 融合三方AI建议的最优方案

echo "=== EGPO Phase 1 修复版训练 ==="
echo "修复内容:"
echo "  ✓ Stand Still奖励函数（根因修复）"
echo "  ✓ 惩罚权重增强（-2.0 → -5.0）"
echo ""
echo "监控重点:"
echo "  1. stand_still应降到 <-0.1"
echo "  2. mean_reward应恢复到 >6.0"
echo "  3. terrain_level应自然提升"
echo ""

# 主训练命令
python legged_gym/scripts/train.py \
    --task=hex_terrain \
    --run_name=phase1_v2_integrated \
    --num_envs=4096 \
    --headless

echo ""
echo "训练完成后执行以下检查:"
echo "  1. 监控相机稳定性: python monitor_camera_values.py"
echo "  2. 根据监控结果决定是否调整 camera_jitter_weight"
echo "  3. 检查 TensorBoard: tensorboard --logdir logs/hex_terrain/"
