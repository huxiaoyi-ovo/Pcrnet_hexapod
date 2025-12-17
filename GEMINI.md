# 六足机器人强化学习项目 (RL Hexapod Gym) - 项目背景与指南

> **当前版本**: 基于技术方案 V4 (底层 EGPO + 高层 Teacher-Student)
> **核心目标**: 构建一个可在多样复杂地形（平地、斜坡、楼梯、崎岖地形）上完成目标导航的六足机器人系统，并具备 Sim-to-Real 能力。

## 1. 项目概览

本项目是一个基于 **NVIDIA Isaac Gym** 的分层强化学习系统。它通过大规模并行仿真训练六足机器人，使其具备底层地形适应能力和高层视觉导航能力。

### 1.1 核心架构 (Hierarchical Architecture)

系统采用 **双层架构**，通过接口层解耦：

*   **底层运动策略 (Low-Level Locomotion)**
    *   **算法**: **EGPO (Expert-Guided Policy Optimization)** + Encoder。
    *   **输入**: 本体感觉 (Proprioception) + 特权信息 (Privileged Info) / 地形隐变量 (Terrain Latent) + 速度指令。
    *   **输出**: 18维关节动作。
    *   **目标**: 稳定行走、地形适应、视觉稳定性（减少相机抖动）。
    *   **特点**: 训练初期利用运动学专家策略进行引导，加速收敛。

*   **高层导航策略 (High-Level Navigation)**
    *   **算法**: **Teacher-Student** 框架。
    *   **Teacher (Phase 2)**: 使用真值 (Ground Truth) 可通行性 (Affordance) 训练，建立性能上限。
    *   **Student (Phase 3)**: 使用感知模型估计的 Affordance 进行蒸馏训练，适应真实传感器噪声。
    *   **输出**: 局部子目标 (Subgoal) + 运动强度 (Intensity $\lambda$)。

*   **感知层 (Perception)**
    *   **Affordance Estimator**: 将深度图 (Depth Image) 转换为占用栅格 (Occupancy) 和可通行性图 (Traversability)。

## 2. 目录结构说明

项目主要分为环境定义 (`legged_gym`) 和 算法实现 (`rsl_rl`) 两部分：

### `legged_gym/` (环境与机器人逻辑)
*   **`envs/hex_v4/`**: **核心目录**。
    *   `hex_ground.py` / `hex_terrain.py`: 环境逻辑实现。
    *   `hex_ground_config.py` / `hex_terrain_config.py`: **关键配置文件** (奖励权重、观测空间、物理参数)。
    *   `expert.py`: EGPO 使用的运动学专家控制器。
*   `scripts/`: 运行脚本。
    *   `train.py`: 训练入口。
    *   `play.py`: 推理/可视化入口。

### `rsl_rl/` (RL 算法库)
*   `algorithms/`: `ppo.py` (PPO实现), `EGPO.py` (专家引导逻辑)。
*   `modules/`: 神经网络架构 (`ActorCritic`, `ActorCriticRecurrent`, `ActorCriticEncoder` 等)。
*   `runners/`: 训练循环控制 (`on_policy_runner.py`, `expert_guided_runner.py`)。

### 其他重要目录
*   `技术方案/`: 详细的技术文档 (V4 完整版、底层/高层设计)。
*   `logs/`: 训练日志与模型权重 (`.pt` 文件)。
*   `resources/`: 机器人 URDF 模型与网格文件。

## 3. 训练流程 (Phased Training)

根据 V4 技术方案，训练分为三个阶段：

### Phase 1: 底层运动控制 (Locomotion)
*   **目标**: 训练机器人在复杂地形上稳定行走，并保证相机平稳。
*   **关键配置**: `enable_nav_reward=False` (关闭导航奖励), 使用随机速度指令。
*   **运行命令**:
    ```bash
    # 使用 Phase 1 修复版配置 (推荐)
    bash train_phase1_fixed.sh
    
    # 或手动运行
    python legged_gym/scripts/train.py --task=hex_terrain --num_envs=4096 --headless
    ```
*   **监控**: 关注 `stand_still` 奖励（应降低）和 `mean_reward`（应上升）。

### Phase 2: 高层 Teacher 导航 (Navigation Teacher)
*   **目标**: 获得基于完美感知的导航策略上限。
*   **配置**: 冻结底层网络，启用导航奖励。
*   **输入**: GT Affordance + 机器人状态 + 目标。

### Phase 3: 高层 Student 导航 (Navigation Student)
*   **目标**: Sim-to-Real 准备，使用估计的感知信息。
*   **方法**: 蒸馏 Teacher 的策略 (Subgoal + Intensity)。

## 4. 常用开发命令

*   **启动训练 (Headless)**:
    ```bash
    python legged_gym/scripts/train.py --task=hex_terrain --num_envs=4096 --headless
    ```
*   **恢复训练**:
    ```bash
    python legged_gym/scripts/train.py --task=hex_terrain --resume --load_run=<run_id>
    ```
*   **可视化/测试 (Play)**:
    ```bash
    python legged_gym/scripts/play.py --task=hex_terrain --load_run=<run_id>
    ```
*   **监控 TensorBoard**:
    ```bash
    tensorboard --logdir logs/hex_terrain/
    ```

## 5. 开发规范与注意事项

1.  **特权信息 (Privileged Info)**:
    *   Critic 网络可以看到“上帝视角”信息（如地形高度图、接触力），而 Actor 只能看到传感器数据。
    *   Sim-to-Real 的关键在于 Actor 必须依赖 **Proprioception** (本体感觉) 和 **History** (历史信息) 或 **Estimator** (估计器)。

2.  **配置文件**:
    *   所有的参数调整（奖励权重、PID 参数、环境物理属性）应在 `*_config.py` 类中修改。
    *   **严禁**硬编码关键参数在代码逻辑中。

3.  **坐标系**:
    *   确保所有方向向量（如 `goal`）与机器人基座坐标系一致。

4.  **近期修复 (Phase 1 Fixes)**:
    *   修复了 `Stand Still` 奖励函数的逻辑。
    *   增强了惩罚项权重。
    *   请优先使用 `train_phase1_fixed.sh` 进行底层训练。