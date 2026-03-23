from pathlib import Path
files = [
    Path('/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym/legged_gym/envs/hex_v4/hex_ground.py'),
    Path('/home/hxy/RL_GYM_PROJECTS/RL_hexapod_gym/legged_gym/scripts/train_highlevel.py'),
]
patterns = [
    's_avoid_episode_goal_best_dist',
    's_avoid_episode_goal_init_dist',
    'reach_given',
    'goal_buf',
    'goal_world',
]
for p in files:
    print(f'FILE: {p}')
    lines = p.read_text().splitlines()
    for pat in patterns:
        hits = [i+1 for i,l in enumerate(lines) if pat in l]
        if hits:
            print(pat, hits)
    print()
