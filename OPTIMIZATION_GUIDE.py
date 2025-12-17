# Phase 1 次要优化配置
# 仅在P0修复后奖励仍不理想时使用

class OptionalFixes:
    """
    基于三方AI建议的可选优化
    """
    
    # 来源: Codex建议
    # 如果机器人过于保守（不敢移动）
    def reduce_energy_penalties(self):
        """降低能耗相关惩罚"""
        action_rate = -0.03      # 从-0.05降低
        dof_acc = -2.0e-7        # 从-3.0e-7降低
        torques = -0.01          # 如果配置中有单独torques项
    
    # 来源: Gemini建议（仅在监控脚本验证后）
    # 如果 monitor_camera_values.py 显示 ang_acc_xy > 50
    def reduce_camera_jitter_weight(self):
        """降低相机抖动惩罚权重"""
        camera_jitter_weight = 0.005  # 从0.05降低10倍
        # 保持其他权重不变
        camera_wobble_weight = 0.5
        camera_bobbing_weight = 0.1
    
    # 来源: Gemini建议
    # 如果地形课程在修复stand_still后仍不提升
    def adjust_terrain_curriculum(self):
        """调整地形课程参数"""
        # 检查 hex_terrain.py 中的 _update_terrain_curriculum
        # 降低升级门槛: distance > env_length/2 → distance > env_length/3
        # 或者强制初始化到更高难度
        max_init_terrain_level = 3  # 从1提升到3
    
    # 来源: Codex建议
    # 如果希望机器人更多静止任务
    def adjust_command_distribution(self):
        """调整指令分布"""
        # 增加零速度指令的概率
        # 或降低速度指令范围
        command_ranges = {
            "lin_vel_x": [-0.4, 0.4],  # 从[-0.6, 0.6]缩小
            "lin_vel_y": [-0.6, 0.6],  # 从[-0.9, 0.9]缩小
            "ang_vel_yaw": [-0.4, 0.4] # 从[-0.6, 0.6]缩小
        }


# ============================================================
# 使用决策树
# ============================================================

"""
Step 1: 运行修复版训练
    → python train_phase1_fixed.sh

Step 2: 观察iter=500的结果
    
    Case A: stand_still <-0.1 且 mean_reward >6
        → ✅ 修复成功！继续训练到2000
        → 进入Step 3

    Case B: stand_still仍然很负
        → ⚠️ 检查是否正确应用了代码修复
        → 重新检查 _reward_stand_still 实现
    
    Case C: mean_reward仍然接近0
        → 考虑应用 reduce_energy_penalties()

Step 3: 运行相机监控
    → python monitor_camera_values.py
    
    Case A: ang_acc_xy > 50
        → 应用 reduce_camera_jitter_weight()
        → 重新训练
    
    Case B: ang_acc_xy < 50
        → 当前权重合理，无需修改

Step 4: 检查地形课程
    
    Case A: terrain_level在iter=1000时 >3
        → ✅ 地形课程正常
    
    Case B: terrain_level仍然=0
        → 应用 adjust_terrain_curriculum()
        
Step 5: Phase 1完成标准
    ✓ mean_reward >6
    ✓ stand_still <-0.1
    ✓ terrain_level >3
    ✓ camera_stability >0.05
    ✓ episode_length >400
    
    → 进入Phase 2训练
"""
