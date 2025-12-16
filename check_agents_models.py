# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import os

agents_dir = "/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym/agents"
models = ["EGPO_2000.pt", "EGPO_3000.pt", "base_line200.pt", "fast_2000.pt"]

print("="*70)
print("Check agents models")
print("="*70)

for model_name in models:
    model_path = os.path.join(agents_dir, model_name)
    if not os.path.exists(model_path):
        print("\n[SKIP] {}: File not found".format(model_name))
        continue
    
    print("\n" + "="*70)
    print("Model: {}".format(model_name))
    print("="*70)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print("Format: Dict")
            print("Keys: {}".format(list(checkpoint.keys())))
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("[OK] Full checkpoint (model + optimizer + iter)")
                if 'iter' in checkpoint:
                    print("   Iterations: {}".format(checkpoint['iter']))
            else:
                state_dict = checkpoint
                print("[WARN] Direct state_dict (may be actor-only)")
        else:
            state_dict = checkpoint
            print("Format: Direct state_dict")
        
        print("\nNetwork structure:")
        
        actor_keys = [k for k in state_dict.keys() if 'actor' in k.lower()]
        if actor_keys:
            first_layer_key = None
            for key in sorted(actor_keys):
                if '.0.weight' in key or 'actor.weight' in key:
                    first_layer_key = key
                    break
            
            if first_layer_key and first_layer_key in state_dict:
                input_dim = state_dict[first_layer_key].shape[1]
                output_dim = state_dict[first_layer_key].shape[0]
                print("  Actor first layer: {}".format(first_layer_key))
                print("  Input dim: {}".format(input_dim))
                print("  Hidden dim: {}".format(output_dim))
                
                if input_dim == 75:
                    print("\n[RESULT] Task: hex_ground")
                    print("   - 75 dims = last_actions(18) + dof(18+18+18) + cmd(3)")
                elif input_dim == 67:
                    print("\n[RESULT] Task: hex_terrain")
                    print("   - 67 dims = quat(4) + ang_vel(3) + lin_acc(3) + dof(18+18+18) + cmd(3)")
                else:
                    print("\n[UNKNOWN] Input dim: {}".format(input_dim))
        
        critic_keys = [k for k in state_dict.keys() if 'critic' in k.lower()]
        if critic_keys:
            print("\n[OK] Has Critic ({} params)".format(len(critic_keys)))
            for key in sorted(critic_keys):
                if '.0.weight' in key:
                    critic_input_dim = state_dict[key].shape[1]
                    print("   Critic input dim: {}".format(critic_input_dim))
                    break
        else:
            print("\n[MISSING] No Critic (actor-only)")
        
        if 'std' in state_dict:
            print("[OK] Has std parameter")
        else:
            print("[WARN] Missing std parameter")
        
        print("\nTotal parameters: {}".format(len(state_dict)))
        
    except Exception as e:
        print("[ERROR] Failed to load: {}".format(e))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nInput dimension mapping:")
print("  - 75 dims -> hex_ground task (with last_actions)")
print("  - 67 dims -> hex_terrain task (with IMU)")
print("\nRecommendation:")
print("  - If no 67-dim model: need to train hex_terrain Phase 1")
print("  - If has 67-dim with Critic: ready for Phase 2")
print("="*70)
