import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# from ..utils.helpers import par
import torch

args=get_args()
env,env_cfg = task_registry.make_env(name=args.task, args=args)
# print(env._reward_support_height())


# ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
# ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
#test

