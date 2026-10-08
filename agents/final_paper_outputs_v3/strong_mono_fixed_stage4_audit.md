# Strong Mono fixed Stage-4 audit

Frozen on 2026-09-17. This note uses only the original fixed `s_pcr_line_avoid_basic`
Stage-4 evaluation. `revision_heldout_mixed_v1` is reference-only and is excluded
from every result and conclusion below.

## Evaluation contract

- Policy: deterministic Strong Mono (`mono_ppo`), one training seed (`seed=1`).
- Scene: `s_pcr_line_avoid_basic`, `avoid_stage_override=4`, frozen stage.
- Evaluation: seeds 101/102/103, 128 episodes per cell, 64 parallel environments,
  difficulty levels `0.0,0.25,0.5,0.75,1.0`.
- Strict success: final row crossed, target inside the follow band, and no episode collision.
- Completed-2000 checkpoint: `model_completed_iter_2000.pt`, SHA-256
  `93f4f6b155ff7576b83fa6c7d82045dbdb06a1c5fd12810bbb9885a6498479a2`,
  clean source commit `5f268bde5447b6a9279491cc821ca60d723a2fda`.
- New six-cell output:
  `/home/dell/Pcrnet_hexapod_strong_mono_75b5582/outputs/strong_mono_budget_matched_2000_stage4_5f268bd`.

## Completed-2000 Stage-4 result

Mean +/- sample SD over three evaluation seeds:

| Speed | Strict success | Collision | Row progress | Follow MAE |
| --- | ---: | ---: | ---: | ---: |
| 0.50 m/s | 0.00 +/- 0.00% | 94.53 +/- 3.13% | 9.38 +/- 1.65% | 0.362 +/- 0.015 m |
| 0.60 m/s | 0.00 +/- 0.00% | 91.15 +/- 3.52% | 9.32 +/- 3.29% | 0.537 +/- 0.027 m |

Per-seed strict success is zero in all six cells. The protocol validator checked
all six `(speed, seed)` pairs, 768 total episodes, deterministic inference, the
fixed Stage-4 override, checkpoint path, clean source commit, and episode count.

Together with the pre-existing 0.35 m/s result, completed 2000 now has a complete
three-speed Stage-4 row. This checkpoint contains 24.576M transitions, exactly
matching the 24.576M total high-level task-learning budget of Avoid pretraining
(12.288M) plus arbitration learning (12.288M).

## Curriculum transitions

The TensorBoard event files give the following exact first-observed mastery stages:

| Mastery transition | PPO iteration | Interactions |
| --- | ---: | ---: |
| Initial stage 0 | 0 | 0 |
| 0 -> 1 | 1210 | 14,868,480 |
| 1 -> 2 | 1236 | 15,187,968 |
| 2 -> 3 | 3450 | 42,393,600 |

The final plot annotation must therefore use 42.394M interactions, not the earlier
approximation of 42.5M.

## Frozen Strong Mono configuration

- Observation: canonical observed local map with two semantic channels
  (occupancy and soft passability), 32x32 over 3 m; the CNN appends two coordinate
  channels internally. State dimension is 13, with actor indices 0 and 1 masked;
  goal dimension is 2; actor and critic difficulty inputs are zeroed.
- Action: three direct command components `[x_right, y_forward, yaw]` from a
  tanh-squashed diagonal Gaussian.
- Network: map CNN `4 -> 32 -> 64 -> 128`, state MLP `13 -> 64 -> 64 -> 64`,
  goal MLP `2 -> 32 -> 32`, fusion `225 -> 256 -> 256`; separate actor/critic
  encoders, Gaussian mean/std heads, and scalar value head.
- Trainable parameters: 1,029,447.
- PPO: 256 environments, 48 steps/iteration, 5 epochs, 4 mini-batches of 3,072,
  Adam, initial LR `3e-4` with adaptive KL target `0.01`, clip `0.2`, entropy
  coefficient `0.01`, value coefficient `1.0`, gamma `0.99`, GAE lambda `0.95`,
  max gradient norm `1.0`, clipped value loss enabled.
- Update protection: LR change capped at 1.5x per iteration, KL early stop at
  `0.02`, full-iteration rollback at KL `0.05`.
- Reward: Strong Mono metadata records `pcr_common_task_reward_v1` and excludes
  method-specific approach, target-view, gate-smooth, gate-auxiliary, and
  yaw-suppression terms. The legacy Learned-w checkpoint records its `w` auxiliary
  settings but not a complete frozen reward table, so its exact reward weights
  must be cited from the corresponding training source/config rather than inferred
  from this checkpoint alone.
- Curriculum: transition openings at 6.144M/12.288M/18.432M; 2,048 frontier
  episodes; success >= 0.50, row progress >= 0.70, collision <= 0.30; 20% next-level
  probe while the frontier gate is unmet; stage weights are
  `(1,0,0,0)`, `(0.4,0.6,0,0)`, `(0.2,0.3,0.5,0)`, `(0.1,0.2,0.3,0.4)`.
- Exact accepted optimizer-update counters: completed 2000 = 23,154; completed
  3000 = 33,952; completed 4000 = 46,486; completed 5000 = 60,082.

## Arbitration policy comparison

- Learned-w arbitration observation: the same two semantic map channels plus two
  internal coordinate channels, state dimension 13, and goal/features dimension
  18 (2 goal values + 16 command/conflict features).
- Learned-w action: two Beta-distributed scalars `(y,w)` controlling arbitration;
  it does not emit direct velocity commands.
- Its CNN/state/fusion widths match Strong Mono; the larger goal input and extra
  Beta head produce 1,063,237 trainable policy parameters, only 3.28% more than
  Strong Mono's policy.
- The deployed Learned-w high-level system additionally contains the fixed Avoid
  expert (1,029,575 trainable parameters when trained; frozen during arbitration
  learning), giving 2,092,812 stored high-level parameters. The Follow expert is
  analytic and the low-level locomotion policy is fixed.
- Arbitration PPO: 256 environments, 48 steps/iteration, 5 full-batch epochs,
  Adam LR `6e-5`, clip `0.05`, entropy coefficient `0.04`, value coefficient
  `0.5`, gamma `0.99`, GAE lambda `0.95`, max gradient norm `0.5`.
- The arbitration run budget is 1,000 iterations = 12.288M interactions = 5,000
  optimizer-step opportunities. The selected final checkpoint is iteration 709
  (710 completed iterations, 3,550 optimizer-step opportunities); the legacy
  checkpoint does not store an accepted-update counter, so an exact accepted
  count must not be claimed.

## Claim boundary

The comparison is budget-matched at completed 2000, not architecture-identical.
Strong Mono receives a direct-command action space and a curriculum specifically
repaired for monolithic learning; Learned-w retains fixed experts and learns only
their arbitration. The defensible paper claim is therefore about task-specific
interaction cost and cross-condition behavior, not superiority under identical
parameter count or identical action space.
