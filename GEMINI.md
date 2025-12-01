# Hexapod Robot Reinforcement Learning Project (Isaac Gym)

## Project Overview

This project implements a Reinforcement Learning (RL) framework for controlling a hexapod robot using NVIDIA's Isaac Gym. It utilizes proximal policy optimization (PPO) and Expert-Guided Policy Optimization (EGPO) to train robust locomotion policies on various terrains. The system is designed for massive parallelism on GPU, enabling the training of thousands of environment instances simultaneously.

**Key Features:**
*   **Simulator:** NVIDIA Isaac Gym (GPU-accelerated physics).
*   **Algorithms:** PPO (Proximal Policy Optimization), EGPO (Expert-Guided Policy Optimization).
*   **Architecture:** Teacher-Student framework for Sim-to-Real transfer.
    *   **Privileged Information:** Used during training (Teacher).
    *   **Estimator/Encoder:** Used during deployment (Student) to estimate privileged states (velocity, gravity, contact forces, terrain).
*   **Parallelism:** Supports training 4096+ environments in parallel.
*   **Terrain Adaptation:** Curriculum learning with various terrain types (slopes, stairs, rough terrain).

## Directory Structure

*   `legged_gym/`: Main package containing environment definitions and scripts.
    *   `envs/`: Environment logic.
        *   `hex_v4/`: Specific environment for the Hexapod robot (`hex_ground.py`, `hex_ground_config.py`).
    *   `scripts/`: Entry points for training and evaluation (`train.py`, `play.py`).
    *   `utils/`: Helper functions (math, terrain generation, task registry).
    *   `resources/`: Robot assets (URDFs, meshes) and neural network weights.
*   `rsl_rl/`: RL algorithm implementations.
    *   `algorithms/`: PPO, EGPO implementations.
    *   `runners/`: Training loop runners.
    *   `modules/`: Neural network architectures (Actor-Critic, Recurrent, Encoder).
*   `logs/`: Training logs, TensorBoard events, and exported policies.

## Getting Started

### 1. Training
To start training the hexapod policy:

```bash
python legged_gym/scripts/train.py --task hex_ground
```

*   **Configuration:** The training behavior is controlled by `legged_gym/envs/hex_v4/hex_ground_config.py`. Key parameters include `num_envs`, `episode_length_s`, and reward scales.
*   **Headless Mode:** To run without the visualizer (faster), add `--headless`.

### 2. Evaluation (Play)
To visualize a trained policy:

```bash
python legged_gym/scripts/play.py --task hex_ground
```

*   **Model Loading:** By default, it loads the latest run from `logs/hex_ground/`. You can specify a run using `--load_run <run_id>` or specific checkpoint.
*   **Export:** The script can export the policy to a JIT module for C++ deployment if `EXPORT_POLICY = True` is set in the script.

## Configuration

The project relies heavily on configuration classes defined in `*_config.py` files.

*   **`HexGroundCfg`:** Defines environment parameters (physics, terrain, noise, sensors).
    *   `env`: Number of environments, observation space dimensions.
    *   `terrain`: Terrain types, proportions, and curriculum settings.
    *   `rewards`: Reward functions and their weights (scales).
    *   `control`: PD gains, action scaling, and actuator network settings.
*   **`HexGroundCfgPPO`:** Defines training parameters.
    *   `runner`: Algorithm selection (`policy_class_name`, `algorithm_class_name`), experiment name.
    *   `algorithm`: Learning rate, schedule, PPO clip parameters.

## Development Guidelines

*   **Modifying Rewards:** Adjust weights in `HexGroundCfg.rewards.scales` to shape the robot's behavior.
*   **New Terrains:** Modify `HexGroundCfg.terrain` to enable/disable terrain types or adjust difficulty.
*   **Sim-to-Real:** The project uses an actuator network (`resources/actuator_nets/`) to model real-world actuator dynamics. Ensure this path is correct in the config.

## Reference
For a detailed technical summary in Chinese, please refer to `overview.md`.
