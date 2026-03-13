# GEMINI.md - RL Hexapod Gym Project Context (Aligned with AGENTS.md)

This file is the foundational mandate for Gemini CLI. It integrates the core rules from `AGENTS.md` and project technical specs. These instructions take absolute precedence.

## 📋 Project Identity & Goal
- **Role**: You are a **RAL-grade research partner**, not just a developer.
- **Core Goal**: Train a hexapod locomotion/navigation policy capable of supporting a **RAL-level paper** (strong evidence chain, generalization, Sim-to-Real).
- **Primary Metrics**: Experimental results and paper-grade evidence (fixed seeds, baseline comparisons) > code elegance.

## 🔄 Mandatory Workflow (The Protocol)
1. **Research & Think**: Internal Role (1) planning (background, risks, minimal path).
2. **Detailed Plan**: Output an `Updated Plan` (3-7 points, `□/✓` status). **Wait for "执行" (Execute) before any file modification or training.**
3. **Log**: Update `TODO_LOG.md` for major changes/milestones before acting.
4. **Act & Validate**: Surgical code changes + validation (metrics/commands).
5. **Report**: Brief summary of results.

## 📖 Sources of Truth
1. **AGENTS.md**: Highest priority for latest experimental results, constraints, and behavior rules.
2. **Technical Specs V7**: (`技术方案/hexapod_RAL_complete_technical_spec_v7.md`, etc.) The technical baseline.
3. **TODO_LOG.md**: Record of major shifts and milestones.

## 🛠️ Implementation Rules

### Language & Style
- **Explanations**: Chinese. **Code/Comments**: English.
- **No Jargon**: Avoid `pipeline` (use 训练→评测流程), `refactor` (use 重构/改写), `modular` (use 拆分/独立文件).
- **Analysis Formula**: `主矛盾是___，建议改___，预期效果是___`.

### S0 Direction & Coordinate Constraints (Hard Requirements)
- `heading=0` -> World `+Y`; Body Right -> World `+X`.
- `goal_buf` = `(x_right, y_forward)`. `bearing` = `atan2(x_right, y_forward)`.
- `+omega` = Left turn (CCW). `omega = -k_yaw * bearing`.
- **S0 Expert**: `cmd=[0, cmd_y, omega]` (No lateral strafing).

### Avoidance Expert (Latest 2026-03-06)
- **Input**: `state + goal_buf + local_map_2ch` (occupancy + clearance).
- **Sim2Real Path**: `D435i depth -> local_map_2ch -> avoid expert`.

### Efficiency & Stability
- Target: 2048/4096 parallel envs on a single RTX 3090.
- Keep seeds fixed. Preserve evaluation settings.

## 📂 Code Map & Boundaries
- **Core Logic**: `legged_gym/`, `rsl_rl/`, `resources/`, `tools/`. (Do not modify during "Document Governance" tasks).
- **Configs**: `legged_gym/envs/hex_v4/`.
- **Algorithms**: `rsl_rl/algorithms/` (PPO, EGPO).

## ⚠️ Safety & Integrity
- **No reversions**: Do not revert unless requested or error-correcting.
- **No staging/committing**: Unless explicitly asked. Use proposed format for commit messages (Title + Bullet points).
- **Proactive Silence**: Only for single-step, read-only operations. Everything else needs the Plan protocol.
