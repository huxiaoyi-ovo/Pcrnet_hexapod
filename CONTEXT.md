# 项目上下文交接文档

## 1. 项目概述

- 项目名称：`RL_hexapod_gym`
- 技术栈：Isaac Gym + PyTorch + `rsl_rl` PPO，高层避障策略在 `legged_gym/scripts/train_highlevel.py`，高层网络在 `rsl_rl/algorithms/high_level_planner.py`。
- 核心目标：训练六足机器人高层避障专家，默认观测为 `state + goal_buf + local_map_2ch`，输出 `cmd_vel`，在单卡 3090 约束下拿到可比较、可复现、可支撑论文的结果，并逐步靠近 Sim2Real 口径。

## 2. 当前进展

- 已完成的模块/功能：
  - 坐标口径已统一：`heading=0` 对齐 world `+Y`，`goal_buf=(x_right, y_forward)`，`bearing=atan2(x_right, y_forward)`。
  - 避障主线已固定为 `state + goal_buf + local_map_2ch`，`local_map_2ch` 只保留 `occupancy + clearance/cost`。
  - `s_avoid_basic` 的课程升级逻辑已收紧：不再只看 `collision100 + exposure100`，还加入了 `progress100 + success100`，并把 `avoid_stage_switch_min_episodes` 提到 `1600`。
  - 已验证控制链路前向没有写反：`play_highlevel --force_cmd 0 0.5 0` 时，机体系 `y_forward` 为正且距离下降。
  - 已修训练代码中的一批不必要梯度追踪：rollout 阶段高层 `policy.get_action / evaluate_actions` 已放进 `torch.no_grad()`；update 改为 `optimizer.zero_grad(set_to_none=True)`；新增 `--debug_memory` 显存诊断打点。
- 正在进行的任务（当前卡在哪）：
  - 当前主问题不是奖励本身，而是训练跑到几十轮后，PPO update 阶段在 `critic_affordance_encoder -> conv2d` 处 OOM。
  - 现象：`num_envs=128, num_steps=24, lr=2e-5, mini_batch_size=128` 时，训练方向已经明显学对（如 `iter 50` 出现 `Body fwd/back = 0.255/0.014`，`Goal dist=3.169`），但后续仍会在 update 阶段因显存耗尽崩溃。
  - 当前最怀疑的是训练循环里仍存在逐轮累积的 CUDA 占用；已加诊断，但还没拿到 `MemDebug` 日志。
- 下一步计划：
  - 用当前诊断版命令重跑，抓 `MemDebug` 日志，确认显存是在 `after_rollout / pre_eval / post_backward / after_update` 哪一段单调上涨。
  - 如果确认是训练循环累积，占优先级最高的是继续排查 update 阶段张量生命周期，而不是再改奖励。
  - 真正的双卡训练还没实现；若单卡诊断后仍无法稳定，需要改代码实现 `sim_device` 与 `rl_device` 真正分开。

## 3. 代码约定（必须遵守）

- 命名规范：
  - 用户解释始终中文；代码与代码注释始终英文。
  - 坐标和动作口径固定：`cmd[0]=x_right`，`cmd[1]=y_forward`，`cmd[2]=omega`，`+omega=左转`。
- 文件结构约定：
  - 优先改现有文件，不默认新增类或大结构。
  - 当前高频相关文件：
    - `legged_gym/scripts/train_highlevel.py`
    - `legged_gym/envs/hex_v4/hex_ground.py`
    - `legged_gym/envs/hex_v4/hex_scenes_config.py`
    - `rsl_rl/algorithms/high_level_planner.py`
- 注释/文档风格：
  - 面向实验，不写空泛工程化表述。
  - 重大方向调整才记 `TODO_LOG.md`；小参数微调默认不记。
- 其他强制约定：
  - 每次新任务先输出 `Updated Plan`，未获“执行”前不改文件/不跑有副作用命令。
  - 回答用户给出指标时，首句格式固定为：`主矛盾是___，建议改___，预期效果是___`。
  - 解释只围绕实验逻辑：奖励、训练动力学、观测/动作影响。

## 4. 关键设计决策（及原因）

- 避障专家默认观测固定为 `state + goal_buf + local_map_2ch`，不再把 `low_obstacle` 作为默认必需输入。原因：当前部署主线要对齐 `D435i depth -> local_map_2ch -> avoid expert`。
- `goal_buf` 统一解释为“局部子目标”，不要求必须是人的真值位置。原因：训练和部署都需要兼容路径点/跟踪器/外部定位来源。
- 当前 reward 主线保留：
  - `goal_approach_scale=2.5`
  - `risk_barrier_scale=-1.2`
  - `collision_penalty=-25.0`
  - `heading_scale=0.20`
  - `yaw_rate_penalty=-0.05`
  - `target_center_scale=0.10`
  - `body_backward_scale=3.0`
  - `backward_scale=0.0`
    原因：这版已经把策略从“倒着逃跑”拉到“正向避障”的趋势上，不能再频繁改奖励。
- 课程升级不再只看“遇到障碍但没撞”，而要求“低碰撞 + 真遇到障碍 + 有效推进 + 近目标成功率达标”。原因：避免靠保守退让提前升到 stage2。
- 当前不直接上真正双卡改动，先做单卡显存诊断。原因：先用最小实验确认显存累积的具体位置，再决定是否值得改训练设备流。

## 5. 已知问题 & 注意事项

- 当前的 bug 或技术债：
  - 训练能学对，但几十轮后仍可能在 PPO update 的 `critic_affordance_encoder -> conv2d` 处 OOM。
  - 当前代码还不支持真正的 `sim_device / rl_device` 分卡训练；`CUDA_VISIBLE_DEVICES=1` 只是把整个进程搬到物理 GPU1。
  - `train_highlevel.py` 里旧的 `Actor MLP` 打印是冻结低层策略，不是高层 CNN；不要被日志误导。
- 需要特别小心的地方：
  - 不要再用 `5 iter / 20 env` 这种短冒烟来判断奖励方向，它只能查硬错误。
  - 不要一边改奖励一边改学习率/mini-batch/课程门槛，否则无法解释结果。
  - 现在训练方向已经变好，优先保住这条线；显存问题应先靠诊断定位，不要先大改网络结构。

## 6. 新窗口启动指令

在新窗口中，请以“机器人科研实验搭档 + 编程主力”的身份继续工作，严格遵守本仓库 `AGENTS.md`：先给 `Updated Plan`，等待用户“执行”；所有解释用中文、代码注释用英文；优先围绕当前实验主线工作，不扩展范围。当前首要任务不是改奖励，也不是改网络结构，而是基于现有训练方向已经转正这一事实，继续审查 `train_highlevel.py` 中所有与梯度传播、张量生命周期、CUDA 占用累积有关的代码，利用 `--debug_memory` 输出定位 update 阶段显存上涨的具体位置；只有在定位完成后，才决定是否需要进一步改训练逻辑或实现真正分卡训练。

## 7. 关键信息缺口

- 最新这版 `--debug_memory` 的实际输出日志还没有拿到，这是下一步定位显存累积根因必须补齐的信息。
