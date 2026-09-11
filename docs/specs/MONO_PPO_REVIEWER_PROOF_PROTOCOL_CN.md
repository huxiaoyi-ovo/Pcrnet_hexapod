# Reviewer-proof Mono-PPO 公平比较协议

## 适用范围

正式 Mono-PPO 与 PCR 对照固定使用 `s_pcr_new --revision_contract`。两者使用同一 canonical local map、同一 actor 的绝对 `x_right/y_forward` 掩码、同一 critic 基础观测语义与同一任务奖励。正式 Mono 与默认 PCR Gate 的 actor/critic difficulty 槽均固定为零，不把 local-map 派生 scalar 当作额外 engineered 输入；完整 critic map 和未 mask state 仍保留。训练、bootstrap、timeout bootstrap、PPO update 和 checkpoint 驱动的评测都遵循此口径，metadata 明确记录 actor/critic difficulty semantics。PCR 的 learned-w 输入、Avoid expert 与 Gate 是待比较方法本身的内容，不构成 Mono 的额外输入。

## 共同任务奖励

保留 `pcr_core`、`row_lat`、`row_gap`、`row_push`、`row_cmdx`、`avoid_smooth`、`early_next_gap`、`pcr_gap_success`、碰撞、终止、时间、稳定性及其余共享状态/动作项。`revision_contract` 下关闭 `approach`、`follow_outside`、Mono target-center/visible/lost、Gate 内部的 `gate_smooth`、`pcr_gate_aux` 与 `pcr_yaw_suppress`。这不是删除 PCR 的 learned-w：`w_aux` 仅是 learned-w 的方法专属辅助损失，明确不计入 task reward。

## Strong Mono-PPO

Mono 使用 3-D tanh-bounded diagonal-Gaussian `CmdVelExpert`，输出 `[x_right, y_forward, yaw]`。正式默认值为：`256 env × 48 steps`、`gamma=.99`、`GAE=.95`、`clip=.20`、5 epochs、4 minibatches（每 batch 3072）、`lr=3e-4`、adaptive KL（`desired_kl=.01`）、entropy `.01`、value coefficient `1.0`、max gradient norm `1.0`、clipped value loss。KL 使用 rollout 保存的旧 Gaussian mean/std 与当前 mean/std 的精确对角 Gaussian KL；仅 Mono 使用该调度，原 Gate/PCR 更新保持原行为。

这是 fail-closed 的冻结 profile：正式 Reviewer Mono 仅允许 training seeds `1/2/3`，上述 env、steps、iterations、epochs、mini-batch、PPO 参数、schedule、clipped value loss 与 `pcr_avoid_pretrain_interactions` 任一项被 CLI 覆盖都会直接报错，不能仍标为 reviewer-proof。`revision_contract` 下的正式 PCR/Mono 同样拒绝 `--gate_use_difficulty`，保证 actor/critic difficulty 都为零。

## 预算与选择

一个 gate-stage interaction reference 为 `256 × 48 × 1000 = 12,288,000` transitions。它只是 Gate 阶段的交互成本参照，绝不称为“matched PCR budget”。官方 Avoid run 的已恢复成本边界为 `512 × 24 × 1000 = 12,288,000` transitions；其 best checkpoint 出现 iteration 不改变完整 Avoid run 的成本。故 PCR 总高层学习成本为 `12,288,000 + 12,288,000 = 24,576,000` transitions。

每个 Mono seed 必须从 completed iteration 0 连续完成固定 3000 iterations（`36,864,000` transitions，3 个 gate-stage references）。不采用 plateau、early stop、online-best checkpoint 或 validation 选择最终模型；formal profile 不写 `best_online_reward.pt`，在线指标仅记录到日志。正式 Reviewer Mono 禁止 `--resume`，因为 checkpoint 未保存 env/curriculum/physics 的精确状态；中断后必须同一 seed 从 0 重跑，不能把近似续训写成正式结果。reviewer Mono 专用保存严格按 completed iteration 的每 100 次触发，避免旧周期逻辑的 0/1-based 歧义；三个正式角色节点为：completed 1000 的 `gate-stage interaction reference`（12,288,000）、completed 2000 的 `PCR-total-interaction reference`（24,576,000）和 completed 3000 的 `full-training`（36,864,000，固定最终 checkpoint）。`pcr_avoid_pretrain_interactions=12,288,000` 写入 metadata/config hash，并要求 PCR-total reference 精确落在 completed iteration；不整除或超出 3BG 时训练直接报错，不静默近似。正式启动还要求 Git commit 可解析且源码树干净；dirty tree 或 unknown commit 直接拒绝，避免用“旧 commit + dirty=true”冒充固定 code provenance。

学习曲线只允许 offline diagnostic validation：`--reviewer_mono_validation` 固定为 `.35/.50 m/s`、保留 RNG seeds `101,102`、`num_envs=64`、`episodes=128`、difficulty levels `0,.25,.5,.75,1`、timeseries `64/stride=1`、基础 layout 和 `mono_ppo` 单一方法；它明确不含 `.60 m/s` 或 `heldout_irregular_rows`，不参与停训、checkpoint 选择或 Full 选择。`.60 m/s` 与 held-out layout 只进入最终 test。

保留 tanh-bounded Gaussian actor。其 log-prob 和 Monte-Carlo squashed entropy 都包含 active action dimensions 的 `log|cmd_scale|` Jacobian；zero-scale dimension 被 mask。该常数不改变 PPO ratio/梯度，但修正绝对 logprob 与 entropy 的记录。

若启用 `--mono_ppo_reward_audit`，审计会逐项打印 `approach`、`follow_outside`、`gate_smooth`、`pcr_gate_aux` 与 `pcr_yaw_suppress`，并在 revision contract 下标记 legacy follow 已移除；这些项应为零，作为运行资格检查证据。

最终每个方法使用 3 个训练 seed；最终评测共享同一 layout bank 和目标速度 `.35/.50/.60 m/s`。报告环境交互数、实际成功 `optimizer.step()` 累计次数、单 GPU wall-time hours（不是能耗测量）、可训练参数量与推理总参数量。每个 reviewer Mono iteration 记录 cumulative environment interactions、cumulative optimizer updates、cumulative wall time、latent Gaussian action std mean、`|action/cmd_scale|>=.95` saturation rate 及 raw→clamped postprocessor rate；统计字段会写入 checkpoint，但正式 profile 禁止 resume，不能将其称为精确续训状态。`run_meta.json` 和 checkpoint 记录 source Git commit/dirty 状态、关键 PPO 参数、奖励契约版本、配置 SHA-256、预算及参数统计。预算 marker 优先硬链接周期 checkpoint；若文件系统不支持硬链接则复制同一文件，名称与 completed-iteration 语义不变。历史 checkpoint/结果不进入本协议的正式主表。
