#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1修复代码检查
检查所有关键修改点是否正确
"""

import re

def check_file():
    file_path = "legged_gym/envs/hex_v4/hex_terrain.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("="*60)
    print("P1修复代码检查")
    print("="*60)
    
    # 检查1: depth_raw初始化
    if 'self.depth_raw = torch.zeros' in content:
        print("✅ 1. depth_raw buffer已初始化")
    else:
        print("❌ 1. depth_raw buffer未找到")
    
    # 检查2: depth_images初始化为固定shape
    if re.search(r'self\.depth_images = torch\.zeros\(\s*self\.num_envs,\s*1,', content):
        print("✅ 2. depth_images buffer初始化为(N,1,H,W)")
    else:
        print("❌ 2. depth_images初始化不正确")
    
    # 检查3: _get_depth_images写入depth_raw
    if 'self.depth_raw[:] = torch.stack(depth_images_list' in content:
        print("✅ 3. _get_depth_images正确写入depth_raw")
    else:
        print("❌ 3. _get_depth_images未写入depth_raw")
    
    # 检查4: step_separate使用就地写入depth_images
    if 'self.depth_images[:] = processed' in content:
        print("✅ 4. step_separate中depth_images使用就地写入")
    else:
        print("❌ 4. step_separate中depth_images未使用就地写入")
    
    # 检查5: reset_idx清零depth buffers
    if 'self.depth_raw[env_ids]' in content and 'self.depth_images[env_ids]' in content:
        print("✅ 5. reset_idx清零两个depth buffers")
    else:
        print("❌ 5. reset_idx未正确清零depth buffers")
    
    # 检查6: goal_buf重新绑定检查
    # 排除初始化行（158行和91行）
    lines = content.split('\n')
    goal_rebind_lines = []
    for i, line in enumerate(lines, 1):
        if re.search(r'self\.goal_buf\s*=\s*[^=]', line):
            # 排除初始化
            if i not in [91, 158]:
                goal_rebind_lines.append((i, line.strip()))
    
    if len(goal_rebind_lines) == 0:
        print("✅ 6. 无goal_buf重新绑定（除初始化外）")
    else:
        print(f"❌ 6. 发现{len(goal_rebind_lines)}处goal_buf重新绑定:")
        for line_no, line_text in goal_rebind_lines:
            print(f"    Line {line_no}: {line_text[:60]}")
    
    # 检查7: _update_goal_buffer使用就地写入
    in_update_func = False
    update_uses_inplace = []
    for i, line in enumerate(lines, 1):
        if 'def _update_goal_buffer' in line:
            in_update_func = True
        elif in_update_func and line.strip().startswith('def '):
            in_update_func = False
        
        if in_update_func and 'self.goal_buf[:]' in line:
            update_uses_inplace.append(i)
    
    if len(update_uses_inplace) >= 3:  # nav, velocity, fixed/random
        print(f"✅ 7. _update_goal_buffer的{len(update_uses_inplace)}个分支使用就地写入")
    else:
        print(f"⚠️  7. _update_goal_buffer仅{len(update_uses_inplace)}个分支使用就地写入")
    
    # 检查8: P2防御性guard
    if 'if env_ids.numel() > 0:' in content and 'self.reset_idx(env_ids)' in content:
        # 检查是否在reset前有guard
        reset_idx_line = None
        guard_before_reset = False
        for i, line in enumerate(lines):
            if 'self.reset_idx(env_ids)' in line:
                reset_idx_line = i
                # 检查前5行是否有guard
                for j in range(max(0, i-5), i):
                    if 'if env_ids.numel() > 0:' in lines[j]:
                        guard_before_reset = True
                        break
        
        if guard_before_reset:
            print("✅ 8. P2防御性guard已添加")
        else:
            print("⚠️  8. P2防御性guard可能未正确添加")
    else:
        print("⚠️  8. P2防御性guard未找到")
    
    print("\n" + "="*60)
    print("代码检查完成")
    print("="*60)

if __name__ == "__main__":
    check_file()
