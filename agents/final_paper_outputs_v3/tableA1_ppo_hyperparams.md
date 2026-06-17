| Item | Value | Source |
| --- | --- | --- |
| Algorithm | PPO custom loop | train_highlevel.py |
| Steps per iter | 48 | final Learned-w/Risk-only checkpoint metadata |
| Num envs | 256 | final Learned-w/Risk-only checkpoint metadata |
| Learning rate | 6e-05 | final Learned-w/Risk-only checkpoint metadata |
| Gamma | 0.99 | final Learned-w/Risk-only checkpoint metadata |
| GAE lambda | 0.95 | final Learned-w/Risk-only checkpoint metadata |
| Entropy coef | 0.04 | final Learned-w/Risk-only checkpoint metadata |
| Clip epsilon | 0.05 | final Learned-w/Risk-only checkpoint metadata |
| Value loss coef | 0.5 | final Learned-w/Risk-only checkpoint metadata |
| Max grad norm | 0.5 | final Learned-w/Risk-only checkpoint metadata |
| Mini-batch | 12288 | final Learned-w/Risk-only checkpoint metadata |
| Epochs | 5 | final Learned-w/Risk-only checkpoint metadata |
| Training budget [iters] | 1000 | paper training command |
| Selected Learned-w checkpoint iter | 709 | agents/moe_teacher_best_learnedw.pt |
| Selected Risk-only checkpoint iter | 388 | agents/moe_teacher_best_risk_only.pt |
| Optimizer | Adam | train_highlevel.py optimizer construction |
