# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
import types
import math


def play(args):
    # 使用全局变量，如果未定义则使用默认值
    global EXPORT_POLICY, RECORD_FRAMES, MOVE_CAMERA
    try:
        EXPORT_POLICY
    except NameError:
        EXPORT_POLICY = False
    try:
        RECORD_FRAMES
    except NameError:
        RECORD_FRAMES = False
    try:
        MOVE_CAMERA
    except NameError:
        MOVE_CAMERA = False
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    camera_mode = getattr(args, "camera_mode", "none")
    camera_interval = max(1, int(getattr(args, "camera_interval", 5)))
    camera_env = int(getattr(args, "camera_env", 0))
    camera_env_random = camera_env < 0
    camera_save = bool(getattr(args, "camera_save", False))
    camera_show = bool(getattr(args, "camera_show", False))
    camera_dir = getattr(args, "camera_dir", None)
    camera_active = camera_mode in ("depth", "rgb", "both")

    if camera_active:
        if hasattr(env_cfg, "sensor") and hasattr(env_cfg.sensor, "depth_camera"):
            env_cfg.sensor.depth_camera.enable = True
        else:
            print("[Play] ⚠ Camera requested but env has no depth_camera config. Disabling.")
            camera_active = False

    # play 默认严格复现训练配置（除非显式打开 --play_overrides）
    if getattr(args, "play_overrides", False):
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
    mesh_type_override = getattr(args, "terrain_mesh", None)
    if mesh_type_override is not None:
        mesh_type = str(mesh_type_override).lower()
        if mesh_type not in ("plane", "heightfield", "trimesh"):
            raise ValueError("Invalid --terrain_mesh. Use one of: plane, heightfield, trimesh.")
        env_cfg.terrain.mesh_type = mesh_type

    if getattr(args, "num_envs", None) is None and getattr(env_cfg.env, "num_envs", 0) > 64:
        print(f"[Play] ⚠ env_cfg.env.num_envs={env_cfg.env.num_envs} (training setting). "
              f"建议显式传入 `--num_envs=1` 以降低可视化/显存压力。")

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy（先load再reset，便于后续按 checkpoint 同步课程进度）
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # 同步训练进度（让课程 cap/freeze 与训练一致）
    train_iter = getattr(args, "train_iter", None)
    if train_iter is None:
        train_iter = getattr(args, "checkpoint", None)
    if train_iter is None:
        train_iter = 0
    expert_interface_iter = getattr(getattr(ppo_runner, "alg", None), "expert_interface_iter", None)
    if hasattr(env, "set_train_progress") and expert_interface_iter is not None:
        env.set_train_progress(int(train_iter), int(expert_interface_iter))

    # 检查是否需要分离观测（以 env 能力 + alg 接口为准）
    has_separated_reset = hasattr(env, "reset_separate") and hasattr(env, "step_separate")
    has_encoder_interface = hasattr(getattr(ppo_runner, "alg", None), "encode_obs")
    algorithm_name = getattr(getattr(train_cfg, "runner", None), "algorithm_class_name", "")
    use_separated_obs = has_separated_reset and has_encoder_interface

    print(f"[Play] Architecture detection:")
    print(f"  - algorithm_class_name: {algorithm_name}")
    print(f"  - has_separated_reset: {has_separated_reset}")
    print(f"  - has_encoder_interface: {has_encoder_interface}")
    print(f"  - use_separated_obs: {use_separated_obs}")

    # reset（与训练环境一致）
    if use_separated_obs:
        obs_dict = env.reset_separate()
        obs = obs_dict['proprioception']
        obs_vgf = obs_dict['privileged']
        obs_terrain = obs_dict['terrain']
    else:
        obs, _ = env.reset()
    
    # 根据架构类型获取不同的inference方法
    policy = None  # 初始化为None，避免未定义错误
    if use_separated_obs:
        # EGPOEncoder 架构: 需要 encode_obs 获取 terrain_latent
        ppo_runner.alg.actor_critic.eval()

        # 【关键】确保 storage 已初始化，并显式重置 obs_hist/dones
        if not hasattr(ppo_runner.alg, "storage") or ppo_runner.alg.storage is None:
            # 维度与训练 runner 保持一致：actor_obs=[num_obs+30]，critic_obs=[11*13]
            ppo_runner.alg.init_storage(env.num_envs, ppo_runner.num_steps_per_env, [env.num_obs + 30], [11 * 13], [env.num_actions])

        storage = ppo_runner.alg.storage
        storage.clear()
        if hasattr(storage, "obs_hist"):
            storage.obs_hist.zero_()
        if hasattr(storage, "dones"):
            storage.dones.zero_()
        if hasattr(storage, "num_transitions_per_env") and hasattr(storage, "num_hist"):
            storage.valid_hist_index = storage.num_transitions_per_env + storage.num_hist - 1
        print("[Play] ✓ Reset storage obs_hist/dones")
        
        # EGPOEncoder 架构不支持JIT导出（需要encode_obs流程）
        if EXPORT_POLICY:
            print("[Play] ⚠ EXPORT_POLICY disabled for EGPO architecture (requires encode_obs flow)")
            EXPORT_POLICY = False
        
        print("[Play] ✓ Using EGPOEncoder architecture with separated observations")
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("[Play] ✓ Using standard policy inference")
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    camera_frame_idx = 0
    camera_cfg = getattr(env, "camera_cfg", None)
    camera_cv2 = None
    camera_warned = False

    if camera_active:
        if camera_env_random:
            print("[Play] camera_env<0: random env will be sampled each capture")
        elif camera_env >= env.num_envs:
            print(f"[Play] ⚠ camera_env={camera_env} out of range, clamping to 0")
            camera_env = 0
        if camera_show and getattr(args, "headless", False):
            print("[Play] ⚠ camera_show requested but headless=True. Disabling.")
            camera_show = False
        if camera_show:
            try:
                import cv2 as _cv2
                camera_cv2 = _cv2
            except Exception as exc:
                print(f"[Play] ⚠ camera_show requires cv2: {exc}. Disabling.")
                camera_show = False

    if camera_save and camera_dir is None:
        camera_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            'camera_frames',
        )
    if camera_save and camera_dir:
        os.makedirs(camera_dir, exist_ok=True)

    def _normalize_depth(depth_np):
        if camera_cfg is None:
            depth_norm = depth_np
        else:
            near = float(getattr(camera_cfg, "near_clip", 0.0))
            far = float(getattr(camera_cfg, "far_clip", 1.0))
            denom = max(far - near, 1e-6)
            depth_norm = (depth_np - near) / denom
        return np.clip(depth_norm, 0.0, 1.0)

    def _write_image(path, img):
        if camera_cv2 is not None:
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = camera_cv2.cvtColor(img, camera_cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            return bool(camera_cv2.imwrite(path, bgr))
        try:
            import imageio.v2 as imageio
            imageio.imwrite(path, img)
            return True
        except Exception:
            return False

    #设置变量，用于统计v g f估计的平均误差
    v_ave_err = 0
    g_ave_err = 0
    f_ave_err = 0

    # 可选：键盘覆盖 commands / 上帝模式（默认关闭）
    keyboard_enabled = bool(getattr(args, "keyboard_cmds", False))
    god_enabled = keyboard_enabled or bool(getattr(args, "god_mode", False))
    if (keyboard_enabled or god_enabled) and getattr(env, "viewer", None) is None:
        print("[Play] ⚠ keyboard/god mode requested but viewer is None (headless). Disabling input control.")
        keyboard_enabled = False
        god_enabled = False
    input_enabled = keyboard_enabled or god_enabled

    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_yaw = 0.0
    cmd_step = 0.1
    reset_requested = False
    god_active = bool(getattr(args, "god_mode", False))
    god_env = int(getattr(getattr(env_cfg, "viewer", None), "ref_env", 0))
    god_env = int(np.clip(god_env, 0, env.num_envs - 1))
    god_pos = None
    god_yaw = 0.0
    god_speed = 0.6
    god_yaw_speed = 1.5
    god_keys = {
        "GOD_FWD": False,
        "GOD_BACK": False,
        "GOD_LEFT": False,
        "GOD_RIGHT": False,
        "GOD_YAW_LEFT": False,
        "GOD_YAW_RIGHT": False,
    }

    if input_enabled:
        if keyboard_enabled and hasattr(env, "_resample_commands"):
            # 禁用随机指令重采样，确保键盘指令覆盖随机指令
            def _no_resample(self, env_ids):
                return
            env._resample_commands = types.MethodType(_no_resample, env)

        gym = env.gym
        viewer = env.viewer

        if keyboard_enabled:
            print("\n[Play] Keyboard control enabled:")
            print("  - ↑/↓: vx max +/- (override)")
            print("  - ←/→: vy max +/- (override)")
            print("  - A/D: yaw max +/- (override)")
            print("  - Q/E: yaw max +/- (rotate in place, override)")
            print("  - Space: stop")
            print("  - R: reset")
            if god_enabled:
                print("  - G: toggle god mode (WASD move, Q/E yaw)")

            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_UP, "CMD_VX_UP")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_DOWN, "CMD_VX_DOWN")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_LEFT, "CMD_VY_LEFT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_RIGHT, "CMD_VY_RIGHT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "CMD_YAW_LEFT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "CMD_YAW_RIGHT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "CMD_YAW_LEFT_ONLY")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "CMD_YAW_RIGHT_ONLY")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "CMD_STOP")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "CMD_RESET")

        if god_enabled:
            print("\n[Play] God mode available:")
            print("  - W/S: move forward/back (body heading)")
            print("  - A/D: move left/right (body heading)")
            print("  - Q/E: yaw + / -")
            print("  - G: toggle god mode on/off")
            if keyboard_enabled:
                print("  - Note: keyboard commands are paused while god mode is active.")

            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_W, "GOD_FWD")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "GOD_BACK")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "GOD_LEFT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "GOD_RIGHT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "GOD_YAW_LEFT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_E, "GOD_YAW_RIGHT")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "GOD_TOGGLE")

    def _clamp_cmd(val, rng, default_min, default_max):
        if rng is None:
            return float(np.clip(val, default_min, default_max))
        return float(np.clip(val, float(rng[0]), float(rng[1])))

    def _cmd_range(rng, default_min, default_max):
        if rng is None:
            return float(default_min), float(default_max)
        return float(rng[0]), float(rng[1])

    def _yaw_from_quat(q):
        siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        return math.atan2(siny_cosp, cosy_cosp)

    def _quat_from_yaw(yaw):
        half = 0.5 * yaw
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    for i in range(10*int(env.max_episode_length)):
        with torch.inference_mode():
            if input_enabled:
                cmd_ranges = {}
                vx_min, vx_max = -1.0, 1.0
                vy_min, vy_max = -1.0, 1.0
                yaw_min, yaw_max = -1.0, 1.0
                if keyboard_enabled:
                    cmd_ranges = getattr(env, "command_ranges", {})
                    vx_min, vx_max = _cmd_range(cmd_ranges.get("lin_vel_x"), -1.0, 1.0)
                    vy_min, vy_max = _cmd_range(cmd_ranges.get("lin_vel_y"), -1.0, 1.0)
                    yaw_min, yaw_max = _cmd_range(cmd_ranges.get("ang_vel_yaw"), -1.0, 1.0)

                events = env.gym.query_viewer_action_events(env.viewer)
                for evt in events:
                    if god_enabled and evt.action == "GOD_TOGGLE" and evt.value > 0:
                        god_active = not god_active
                        god_pos = None
                        for key in god_keys:
                            god_keys[key] = False
                        state = "ON" if god_active else "OFF"
                        print(f"[Play] God mode {state}")
                        continue
                    if god_enabled and god_active and evt.action in god_keys:
                        god_keys[evt.action] = evt.value > 0
                        continue
                    if evt.action == "CMD_RESET" and evt.value > 0:
                        reset_requested = True
                        continue
                    if not keyboard_enabled or god_active:
                        continue
                    if evt.value <= 0:
                        continue
                    if evt.action == "CMD_VX_UP":
                        cmd_vx, cmd_vy, cmd_yaw = vx_max, 0.0, 0.0
                    elif evt.action == "CMD_VX_DOWN":
                        cmd_vx, cmd_vy, cmd_yaw = vx_min, 0.0, 0.0
                    elif evt.action == "CMD_VY_LEFT":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, vy_max, 0.0
                    elif evt.action == "CMD_VY_RIGHT":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, vy_min, 0.0
                    elif evt.action == "CMD_YAW_LEFT":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, yaw_max
                    elif evt.action == "CMD_YAW_RIGHT":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, yaw_min
                    elif evt.action == "CMD_YAW_LEFT_ONLY":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, yaw_max
                    elif evt.action == "CMD_YAW_RIGHT_ONLY":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, yaw_min
                    elif evt.action == "CMD_STOP":
                        cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, 0.0

            if keyboard_enabled and not god_active:
                cmd_ranges = getattr(env, "command_ranges", {})
                vx_min, vx_max = _cmd_range(cmd_ranges.get("lin_vel_x"), -1.0, 1.0)
                vy_min, vy_max = _cmd_range(cmd_ranges.get("lin_vel_y"), -1.0, 1.0)
                yaw_min, yaw_max = _cmd_range(cmd_ranges.get("ang_vel_yaw"), -1.0, 1.0)

                # clamp to training command ranges when available
                cmd_vx = _clamp_cmd(cmd_vx, cmd_ranges.get("lin_vel_x"), -1.0, 1.0)
                cmd_vy = _clamp_cmd(cmd_vy, cmd_ranges.get("lin_vel_y"), -1.0, 1.0)
                cmd_yaw = _clamp_cmd(cmd_yaw, cmd_ranges.get("ang_vel_yaw"), -1.0, 1.0)

                # apply command override
                cmd_tensor = torch.tensor([cmd_vx, cmd_vy, cmd_yaw], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
                env.commands[:, :3] = cmd_tensor
                # 刷新观测以反映最新 commands（严格使用 env 自身的观测函数，避免覆盖 obs_buf 结构）
                if use_separated_obs and hasattr(env, "compute_observations_separated"):
                    env.compute_observations_separated()
                    obs, obs_vgf, obs_terrain = env.get_observations_separated()
                else:
                    env.compute_observations()
                    obs = env.get_observations()

            if god_enabled and god_active:
                if god_pos is None:
                    pos_np = env.root_states[god_env, :3].detach().cpu().numpy()
                    quat_np = env.root_states[god_env, 3:7].detach().cpu().numpy()
                    god_pos = [float(pos_np[0]), float(pos_np[1]), float(pos_np[2])]
                    god_yaw = _yaw_from_quat(quat_np)

                fwd_step = 0.0
                side_step = 0.0
                dyaw = 0.0
                if god_keys["GOD_FWD"]:
                    fwd_step += god_speed * env.dt
                if god_keys["GOD_BACK"]:
                    fwd_step -= god_speed * env.dt
                if god_keys["GOD_LEFT"]:
                    side_step += god_speed * env.dt
                if god_keys["GOD_RIGHT"]:
                    side_step -= god_speed * env.dt
                if god_keys["GOD_YAW_LEFT"]:
                    dyaw += god_yaw_speed * env.dt
                if god_keys["GOD_YAW_RIGHT"]:
                    dyaw -= god_yaw_speed * env.dt

                if fwd_step != 0.0 or side_step != 0.0:
                    forward = (math.cos(god_yaw), math.sin(god_yaw))
                    left = (-math.sin(god_yaw), math.cos(god_yaw))
                    dx = forward[0] * fwd_step + left[0] * side_step
                    dy = forward[1] * fwd_step + left[1] * side_step
                    god_pos[0] += dx
                    god_pos[1] += dy
                if dyaw != 0.0:
                    god_yaw += dyaw

                quat = _quat_from_yaw(god_yaw)
                env.root_states[god_env, 0:3] = torch.tensor(god_pos, device=env.device)
                env.root_states[god_env, 3:7] = torch.tensor(quat, device=env.device)
                env.root_states[god_env, 7:13] = 0.0
                env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
                env.commands[god_env, :3] = 0.0

            if reset_requested:
                reset_requested = False
                cmd_vx, cmd_vy, cmd_yaw = 0.0, 0.0, 0.0
                god_pos = None
                if use_separated_obs:
                    obs_dict = env.reset_separate()
                    obs = obs_dict['proprioception']
                    obs_vgf = obs_dict['privileged']
                    obs_terrain = obs_dict['terrain']
                    # reset 后同步清空 history，避免把上一局残留带入下一局
                    if hasattr(ppo_runner.alg, "storage") and ppo_runner.alg.storage is not None:
                        storage = ppo_runner.alg.storage
                        storage.clear()
                        if hasattr(storage, "obs_hist"):
                            storage.obs_hist.zero_()
                        if hasattr(storage, "dones"):
                            storage.dones.zero_()
                        if hasattr(storage, "num_transitions_per_env") and hasattr(storage, "num_hist"):
                            storage.valid_hist_index = storage.num_transitions_per_env + storage.num_hist - 1
                else:
                    obs, _ = env.reset()

            if god_enabled and god_active:
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            elif use_separated_obs:
                # EGPO架构: 使用encode_obs获取terrain_latent，然后调用act_inference
                if getattr(args, "allow_fallback", False):
                    try:
                        obs_splice, obs_terrain_latent = ppo_runner.alg.encode_obs(obs, obs_vgf, obs_terrain)
                    except Exception as e:
                        print(f"[Play] ⚠ encode_obs failed: {e} (fallback enabled)")
                        obs_splice = torch.cat([obs, obs_vgf], dim=-1)
                        obs_terrain_latent = ppo_runner.alg.actor_critic.encode_terrain(obs_terrain)
                else:
                    obs_splice, obs_terrain_latent = ppo_runner.alg.encode_obs(obs, obs_vgf, obs_terrain)
                actions = ppo_runner.alg.actor_critic.act_inference(obs_splice, obs_terrain_latent)
            else:
                actions = policy(obs)
            # actions,obs_vgf_estimate,obs_terrain_lstm_estimates,obs_terrain_lstm = ppo_runner.alg.act_inference(obs,obs_terrain)
            #计算估计误差
            # v_err = torch.norm( (obs_vgf_estimate[:,:3]-obs_vgf[:,:3])/env.obs_scales.lin_vel, dim=1 ).mean()
            # g_err = torch.norm( (obs_vgf_estimate[:,3:6]-obs_vgf[:,3:6])/env.obs_scales.gravity, dim=1).mean()
            # f_err = torch.norm( (obs_vgf_estimate[:,6:]-obs_vgf[:,6:])/env.obs_scales.contact_force, dim=1).mean()
            
            # v_ave_err+=v_err.item()
            # g_ave_err+=g_err.item()
            # f_ave_err+=f_err.item()

            #打印估计值和输出值
            # if i%100==0:
            #     print(f"real v={obs_vgf[0,:3]/env.obs_scales.lin_vel}, est v={obs_vgf_estimate[0,:3]/env.obs_scales.lin_vel}; ")
            #     print(f"real g={obs_vgf_estimate[0,3:6]/env.obs_scales.gravity}, est g={obs_vgf[0,3:6]/env.obs_scales.gravity}")
            #     print(f"real f={obs_vgf_estimate[0,6:]/env.obs_scales.contact_force}, est f={obs_vgf[0,6:]/env.obs_scales.contact_force}")
                # print(f"real terrain lstm\n{obs_terrain_lstm}\n estimates terrain lstm\n{obs_terrain_lstm_estimates}\n")
            
            # obs,obs_vgf,obs_terrain,rew,dones,infos=env.step_separate(actions)
            # ppo_runner.alg.process_inference_env_step(dones)

            if use_separated_obs:
                obs_dict, rews, dones, infos = env.step_separate(actions.detach())
                obs = obs_dict['proprioception']
                obs_vgf = obs_dict['privileged']
                obs_terrain = obs_dict['terrain']
                if hasattr(ppo_runner.alg, "process_inference_env_step"):
                    ppo_runner.alg.process_inference_env_step(dones)
            else:
                obs, _, rews, dones, infos = env.step(actions.detach())


            if camera_active and (i % camera_interval == 0):
                capture_env = camera_env
                if camera_env_random:
                    capture_env = int(np.random.randint(0, env.num_envs))
                depth_np = None
                rgb_np = None
                if camera_mode in ("depth", "both"):
                    if hasattr(env, "_get_depth_images"):
                        depth = env._get_depth_images()
                        depth_np = depth[capture_env].detach().cpu().numpy()
                    elif not camera_warned:
                        print("[Play] ⚠ env has no _get_depth_images; depth capture disabled.")
                        camera_warned = True
                if camera_mode in ("rgb", "both"):
                    if hasattr(env, "_get_rgb_images"):
                        rgb = env._get_rgb_images(normalize=False, channels_last=True)
                        rgb_np = rgb[capture_env].detach().cpu().numpy()
                    elif not camera_warned:
                        print("[Play] ⚠ env has no _get_rgb_images; rgb capture disabled.")
                        camera_warned = True

                depth_vis = None
                if depth_np is not None:
                    depth_norm = _normalize_depth(depth_np)
                    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
                    if camera_cv2 is not None:
                        depth_vis = camera_cv2.applyColorMap(255 - depth_u8, camera_cv2.COLORMAP_TURBO)
                    else:
                        depth_vis = depth_u8

                if camera_show and camera_cv2 is not None:
                    if rgb_np is not None:
                        camera_cv2.imshow("camera_rgb", camera_cv2.cvtColor(rgb_np, camera_cv2.COLOR_RGB2BGR))
                    if depth_vis is not None:
                        camera_cv2.imshow("camera_depth", depth_vis)
                    camera_cv2.waitKey(1)

                if camera_save and camera_dir:
                    if rgb_np is not None:
                        _write_image(os.path.join(camera_dir, f"rgb_{camera_frame_idx:06d}.png"), rgb_np)
                    if depth_np is not None:
                        np.save(os.path.join(camera_dir, f"depth_{camera_frame_idx:06d}.npy"), depth_np)
                        if depth_vis is not None:
                            _write_image(os.path.join(camera_dir, f"depth_{camera_frame_idx:06d}.png"), depth_vis)
                    camera_frame_idx += 1

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                    }
                )
            elif i==stop_state_log:
                logger.plot_states()
            if  0 < i < stop_rew_log:
                if infos["episode"]:
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes>0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i==stop_rew_log:
                logger.print_rewards()
                #打印平均误差
                # print(f"v err={v_ave_err/(i+1)}, g err={g_ave_err/(i+1)}, f err={f_ave_err/(i+1)}")

    if camera_cv2 is not None:
        camera_cv2.destroyAllWindows()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
