# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Training Commands

```bash
# Basic training - use specific scene tasks, not hex_ground directly
python legged_gym/scripts/train.py --task hex_s1 --num_envs 4096
python legged_gym/scripts/train.py --task hex_s2 --num_envs 2048

# Debug training on flat ground (fastest for testing)
python legged_gym/scripts/train.py --task hex_debug_plane --num_envs 256

# High-level navigation training (requires low-level checkpoint)
python legged_gym/scripts/train_highlevel.py --mode teacher --skill follow --task hex_s1 --num_envs 256 --low_level_ckpt logs/hex_ground/Dec31_16-52-59_/model_6000.pt

# Expert-guided training (EGPO)
python legged_gym/scripts/train.py --task hex_s1 --expert_guided
```

### Testing and Visualization

```bash
# Test trained policy
python legged_gym/scripts/play.py --task hex_s1 --load_run logs/hex_s1/[timestamp]

# Camera debugging
python legged_gym/tests/visualize_rgb_depth_combined.py --frames 3 --task hex_debug_heightfield

# Scene audit - validate terrain generation
python tools/scene_audit.py --scenes s1_corridor_gate --seed 42

# Quick environment test
python legged_gym/tests/test_env.py --task hex_s1
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir=logs --port=6006
```

## Development Protocol

### Documentation Maintenance Rules

- **TODO_LOG.md**: Update BEFORE major implementation. Record only major changes/strategy shifts.
- **CHANGELOG_CN.md**: Deprecated. Record major finalized changes in TODO_LOG milestone entries.
- **PROJECT_OVERVIEW_CN.md**: Update "Idea" section ONLY when user explicitly requests to memorize an idea.
- **docs/NAVIGATION.md**: Keep document routing up to date when paths change.

### Coding Workflow

1. **Provide detailed plan** before ANY changes and wait for explicit user approval
2. **List TODO items** after approval and execute step-by-step
3. **Update TODO_LOG.md** before implementation
4. **Record major finalized changes in TODO_LOG milestones**
5. **For docs-governance tasks, avoid touching core code directories** (`legged_gym/`, `rsl_rl/`, `resources/`, `tools/`)

## Project Core Requirements & Constraints

### Core Goal

Train high-quality policies on a single RTX 3090 (2048/4096 parallel) with sufficient simulation quality for strong generalization, transfer, and Sim2Real potential. Aim for RAL-level paper with clear contributions, strong baselines, and reproducible protocols.

### "High Simulation Quality" Definition

- **Task Quality**: Scene distribution must strictly match technical spec (S1-S6 + T1/T2). Must be verifiable with metrics (passability, reachability, collision statistics).
- **Physics Quality**: Contact/friction/mass must be controllable. Policies must NOT rely on simulation exploits (penetration, topological loopholes, wrong collision shapes, reward hacking).
- **Sensing Quality**: Vision/depth for high-level must have stable pipeline and controllable noise. Training/evaluation sensing setup must be reproducible and reportable.

### Critical Performance Constraints

- **Hardware**: Single RTX 3090, target 2048/4096 parallel training. Throughput and stability are hard constraints.
- **Scene Implementation**: Prefer "throughput-friendly" representations (heightfield, weak geometry). Only introduce expensive simulation (many actors, complex collisions) when it clearly supports paper conclusions and is cost-controllable.

### Generalization/Sim2Real Constraints

- **OOD Evaluation**: MUST have strict out-of-distribution evaluation - S6 structured OOD hold-out (not seen in training) as regression test benchmark.
- **Robustness Matrix**: MUST test sensing degradation (noise/blur/dropout), dynamics randomization (friction/mass/thrust perturbation), and policy degradation curves. Avoid only working in "clean simulation".
- **Design Principle**: Always avoid "simulation-specific shortcuts" (overfitting to render features, relying on unrealistic contact/friction boundaries).

### RAL-grade Evidence Requirements

- **Reproducible**: Fixed seeds, fixed evaluation episodes, unified training budget. Report mean/variance or confidence intervals for key results.
- **Strong Baselines**: Include paradigm-level baselines (end-to-end, rule arbitration, classical baselines) with fair budget.
- **Scene Validation**: Complete offline/online "scene distribution contract" validation before large-scale training to confirm the task is actually happening.

### Observation and Layer Boundaries

- **Low-level locomotion**: Does NOT use depth camera. Terrain input is privileged heightmap/sampling.
- **High-level**: Uses affordance/target state to output cmd_vel and gate y. Uses Command Post-Processor for stable interface ensuring training/deployment consistency.

## Role Mode Switch (Ask Every New Session)

**Opening question**: "这次你希望我扮演哪个角色？(1) 决策辅助 (2) 编程主力。回复 1 或 2。"

### (1) Decision Assistant

**Priorities**: Research purpose > Controllable distribution & reproducibility > Measurable metrics & evaluation protocols > Throughput/scale > Realism/detail.

**Output**: Macro design focus - problem definition/trade-offs/metrics/experiment matrix/risk boundaries/next decisions. Avoid implementation details by default.

### (2) Programming Lead

**Priorities**: Correct implementation (semantics & interface contracts) > Project connectivity (runnable, reproducible, trainable/evaluable) > Simplicity & maintainability (minimum complexity, clear structure) > Performance (only key optimizations with evidence).

**Style**: Implement with research-level clean structure. Avoid patch stacking, flag stacking, or assertion-style "defensive patching" that leads to code bloat. Prefer eliminating root causes and unified abstractions.

## Debugging Summary

### Coordinate System Contract

- **Unique contract**: World +Y forward; Tile row(i) = +Y(length); heightfield[a,b] where a=length(+Y), b=width(+X).
- **Single mapping point**: Axis mapping only at `Terrain.add_terrain_to_map()` (including env_origin_z). No distributed transposes/swaps allowed.
- **debug_axis validation**: Must have automatic acceptance test - +Y monotonic increasing, +X constant. Failures hard error to expose misalignment immediately.
- **Isaac Gym compatibility**: SubTerrain.height_field_raw axis order may swap between versions. Use "contract view + explicit transpose" for unified adaptation.

### S1 Corridor Critical Settings

- When robot forward axis is +Y, `heading_offset_rad` MUST be 0. Otherwise robot orientation becomes perpendicular to corridor and triggers reset-loop.
- **NEVER** change heading_offset without verifying corridor navigation works correctly.

## High-Level Architecture

### Project Structure

The codebase consists of two main packages:

- **legged_gym**: Environment definitions and utilities
- **rsl_rl**: Reinforcement learning algorithms (PPO, EGPO)

### Key Architectural Patterns

1. **Task Registry System**: All tasks are centrally registered in `legged_gym/envs/__init__.py`. Never instantiate environments directly - use the registry.

2. **Configuration Hierarchy**: Each task has two config classes:
   
   - `*Cfg`: Environment configuration (physics, rewards, etc.)
   - `*CfgPPO`: Training configuration (learning rates, batch sizes, etc.)

3. **Environment Inheritance**:
   
   ```
   LeggedRobot (base class)
   └── HexGround (hexapod-specific)
   ```

4. **Two-Stage Learning**:
   
   - Stage 1: Low-level locomotion with privileged information
   - Stage 2: High-level navigation using vision/estimators

### Critical Implementation Details

1. **Coordinate System**: The project enforces +Y forward, +X right coordinate system. This is critical for S1 corridor navigation - heading_offset is always 0.

2. **Terrain System**: Currently uses "classic" heightfield-based terrain generation. The deprecated terrain_v2 has been removed. Always use `mesh_type="heightfield"`.

3. **Scene Generation**: Each scene (S1-S6) has specific parameters defined in `hex_scenes_config.py`. S1 uses corridor_width, gate_width, wall_thickness.

4. **Expert Guidance (EGPO)**: When enabled, actions are interpolated: `action = α*expert + (1-α)*RL` where α decays from 1.0 to 0.1 over 200 iterations.

5. **Environment Spacing**: Default 12m spacing between parallel environments. The terrain grid auto-expands to accommodate num_envs.

### Common Pitfalls to Avoid

1. **Never use hex_ground directly** - it's a container class. Always specify a terrain_type like hex_s1.

2. **Don't modify terrain.py directly** - use the scene generation system through SceneSpec.

3. **Always check heading_offset** - should be 0 for S1 tasks to maintain +Y forward.

4. **Use debug tasks for quick iteration** - hex_debug_plane is much faster than terrain tasks.

5. **Validate with scene_audit** before training on new scene configurations.

### Key Files for Understanding Architecture

- `legged_gym/envs/base/legged_robot.py`: Base environment class
- `legged_gym/envs/hex_v4/hex_ground.py`: Hexapod environment implementation
- `legged_gym/envs/hex_v4/hex_scenes_config.py`: All scene configurations
- `legged_gym/utils/task_registry.py`: Task registration system
- `rsl_rl/runners/on_policy_runner.py`: Training loop implementation
- `rsl_rl/algorithms/EGPO.py`: Expert-guided policy optimization
