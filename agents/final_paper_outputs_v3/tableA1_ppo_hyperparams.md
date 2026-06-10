| Item | Value | Source |
| --- | --- | --- |
| Algorithm | PPO custom loop | train_highlevel.py |
| Steps per iter | 24 | train_highlevel.py --num_steps default |
| Num envs | 256 | paper command argument |
| Learning rate | 1.5e-5 | train_highlevel.py --lr default |
| Gamma | 0.99 | train_highlevel.py --gamma default |
| GAE lambda | 0.95 | train_highlevel.py --gae_lambda default |
| Entropy coef | 0.01 | train_highlevel.py --entropy_coef default |
| Clip epsilon | 0.05 | train_highlevel.py --clip_range default |
| Value loss coef | 0.5 | train_highlevel.py --value_loss_coef default |
| Max grad norm | 0.5 | train_highlevel.py --max_grad_norm default |
| Mini-batch | 4096 | train_highlevel.py --mini_batch_size default |
| Epochs | 2 | train_highlevel.py --num_epochs default |
| Total iters | 1000 | train_highlevel.py --num_iterations default |
| Optimizer | Adam | train_highlevel.py optimizer construction |
