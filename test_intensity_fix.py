#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step A-P1: Intensity作弊修复验收测试

验证目标：
1. intensity不再等于1-terrain_difficulty（作弊消除）
2. intensity在[0,1]范围内（数值稳定）
3. intensity与speed_xy正相关（物理意义正确）
"""

import sys
import os
sys.path.append(os.getcwd())

# Isaac Gym要求：先导入isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import numpy as np

def test_intensity_fix():
    """验收测试：确认intensity作弊已修复"""
    print("\n" + "="*70)
    print("Step A-P1: Intensity作弊修复验收测试")
    print("="*70)
    
    args = get_args()
    args.task = 'hex_terrain'
    args.headless = True
    args.num_envs = 64
    
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 64
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.navigation.enable_nav_reward = True  # 启用导航模式
    
    print(f"\n创建环境...")
    print(f"  - num_envs: {args.num_envs}")
    print(f"  - enable_nav_reward: True")
    print(f"  - max_speed_for_intensity: {env_cfg.navigation.max_speed_for_intensity}")
    
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    actions = torch.zeros(64, env.num_actions, device=env.device)
    
    # 收集100步数据
    print(f"\n运行100步，收集数据...")
    intensities = []
    terrain_diffs = []
    speeds = []
    expected_old = []
    
    for i in range(100):
        obs_dict, _, _, info = env.step_separate(actions)
        
        if 'intensity_mean' in env.extras:
            intensities.append(env.extras['intensity_mean'])
        if 'terrain_difficulty_mean' in env.extras:
            td = env.extras['terrain_difficulty_mean']
            terrain_diffs.append(td)
            expected_old.append(1.0 - td)
        if 'speed_xy_mean' in env.extras:
            speeds.append(env.extras['speed_xy_mean'])
        
        if i % 25 == 0:
            print(f"  Step {i:3d}: intensity={env.extras.get('intensity_mean', 0):.3f}, "
                  f"speed={env.extras.get('speed_xy_mean', 0):.3f}, "
                  f"terrain_diff={env.extras.get('terrain_difficulty_mean', 0):.3f}")
    
    # 转换为numpy数组
    intensities = np.array(intensities)
    terrain_diffs = np.array(terrain_diffs)
    speeds = np.array(speeds)
    expected_old = np.array(expected_old)
    
    print("\n" + "="*70)
    print("验收结果")
    print("="*70)
    
    passed = 0
    total = 3
    
    # 检查1: intensity不再等于1-difficulty
    print("\n【检查1】作弊消除检查")
    if len(expected_old) > 0:
        diff_from_old = np.abs(intensities - expected_old).mean()
        print(f"  Intensity vs (1-terrain_difficulty) 平均差异: {diff_from_old:.4f}")
        if diff_from_old < 0.1:
            print(f"  ❌ 作弊仍存在！intensity几乎等于1-difficulty (差异<0.1)")
        else:
            print(f"  ✅ 作弊已消除！intensity与1-difficulty显著不同")
            passed += 1
    else:
        print("  ⚠️  未收集到terrain_difficulty数据")
    
    # 检查2: intensity范围
    print("\n【检查2】数值稳定性检查")
    int_min, int_max = intensities.min(), intensities.max()
    int_mean = intensities.mean()
    print(f"  Intensity统计: min={int_min:.3f}, max={int_max:.3f}, mean={int_mean:.3f}")
    if int_min >= 0 and int_max <= 1:
        print(f"  ✅ Intensity在[0,1]范围内")
        passed += 1
    else:
        print(f"  ❌ Intensity超出范围！")
    
    # 检查3: intensity与速度正相关
    print("\n【检查3】物理意义检查")
    if len(speeds) > 0:
        corr = np.corrcoef(intensities, speeds)[0, 1]
        print(f"  Intensity-Speed相关系数: {corr:.3f}")
        if corr > 0.5:
            print(f"  ✅ Intensity正确反映速度（高度正相关）")
            passed += 1
        elif corr > 0:
            print(f"  ⚠️  相关性较弱但为正（可能速度变化小）")
            passed += 0.5
        else:
            print(f"  ❌ 相关性错误（应该正相关）")
    else:
        print("  ⚠️  未收集到speed数据")
    
    # 额外信息
    print("\n【额外信息】")
    print(f"  平均速度: {speeds.mean():.3f} m/s")
    print(f"  平均地形难度: {terrain_diffs.mean():.3f}")
    print(f"  平均intensity: {int_mean:.3f}")
    print(f"  旧作弊值(1-diff): {expected_old.mean():.3f}")
    print(f"  两者差异: {abs(int_mean - expected_old.mean()):.3f}")
    
    # 最终结论
    print("\n" + "="*70)
    print(f"验收得分: {passed}/{total}")
    print("="*70)
    
    if passed >= 2.5:
        print("\n🎉 修复成功！Intensity作弊已消除，数值稳定且物理意义正确！")
        print("   可以继续进行Phase 2训练。")
        return 0
    elif passed >= 2:
        print("\n✅ 修复基本成功！主要检查通过。")
        print("   建议微调max_speed_for_intensity参数。")
        return 0
    else:
        print("\n❌ 修复可能存在问题，请检查代码。")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_intensity_fix()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
