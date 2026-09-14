#!/usr/bin/env python3
"""CPU runtime checks for the production 1-D clutter Avoid step."""
import ast
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from legged_gym.avoid_reward import avoid_clutter_reward
from legged_gym.pcr_observation import build_avoid_actor_state
from rsl_rl.algorithms.high_level_planner import CmdVelExpert, CommandPostProcessor


def _extract_runtime_methods():
    source = (Path(__file__).resolve().parent / "train_highlevel.py").read_text()
    tree = ast.parse(source)
    wanted = {"_resample_avoid_clutter_context", "_advance_avoid_clutter_context", "_step_avoid_clutter"}
    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "HierarchicalHexapodEnv":
            methods = [copy.deepcopy(item) for item in node.body if isinstance(item, ast.FunctionDef) and item.name in wanted]
            break
    assert {method.name for method in methods} == wanted
    namespace = {"torch": torch, "math": __import__("math"), "avoid_clutter_reward": avoid_clutter_reward}
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "train_highlevel.py", "exec"), namespace)
    return namespace


def _actual_save_checkpoint_condition(iteration, total_iterations, save_interval, checkpoint_iterations=()):
    source = (Path(__file__).resolve().parent / "train_highlevel.py").read_text()
    tree = ast.parse(source)
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        fixed_index = None
        checkpoint_index = None
        for index, node in enumerate(body):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "fixed_checkpoint_interval"
                for target in node.targets
            ):
                fixed_index = index
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "should_save_checkpoint" for target in node.targets):
                continue
            checkpoint_index = index
            break
        if checkpoint_index is not None:
            if fixed_index is None or fixed_index >= checkpoint_index:
                raise AssertionError("checkpoint condition must retain its fixed interval setup")
            namespace = {
                "iteration": iteration,
                "total_iterations": total_iterations,
                "args": SimpleNamespace(
                    save_interval=save_interval,
                    checkpoint_iterations=list(checkpoint_iterations),
                ),
                "is_mono_ppo": False,
            }
            module = ast.Module(
                body=[copy.deepcopy(item) for item in body[fixed_index:checkpoint_index + 1]],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), "train_highlevel.py", "exec"), namespace)
            return bool(namespace["should_save_checkpoint"])
    raise AssertionError("missing checkpoint save condition")


def _resume_rng_restore_source():
    source = (Path(__file__).resolve().parent / "train_highlevel.py").read_text()
    required = [
        'torch.set_rng_state(ckpt["torch_rng_state"].cpu())',
        'torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])',
    ]
    assert all(item in source for item in required)


def _extract_clutter_risk_scale_contract():
    source = (Path(__file__).resolve().parent / "train_highlevel.py").read_text()
    tree = ast.parse(source)
    wanted = {"normalize_task_name", "is_avoid_clutter_task_name", "enforce_avoid_clutter_risk_scale_contract"}
    functions = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {function.name for function in functions} == wanted
    namespace = {"Any": object, "LEGACY_TASK_ALIASES": {}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "train_highlevel.py", "exec"), namespace)
    return namespace["enforce_avoid_clutter_risk_scale_contract"]


class MockPostProcessor:
    def __init__(self, slow_forward=False):
        self.last_cmd = torch.full((1, 3), 0.15)
        self.max_cmd = torch.tensor([0.6, 0.6, 1.0])
        self.slow_forward = slow_forward

    def preview_cmd_before_risk(self, cmd, beta=None):
        return cmd.clone()

    def process(self, cmd, clearance, beta=None):
        executed = cmd.clone()
        if self.slow_forward:
            executed[:, 1] = 0.1
        self.last_cmd = executed.detach().clone()
        return executed, {"mock": True}


class MockLowLevel:
    def act_inference(self, obs):
        return torch.zeros(obs.shape[0], 2)


class MockEnv:
    def __init__(self, mode):
        self.mode = mode
        self.root_states = torch.zeros(1, 13)
        self.env_origins = torch.zeros(1, 3)
        self.s_avoid_exit_y = torch.tensor([0.2])
        self.s_avoid_v_drive_nom = torch.tensor([0.3])
        self.s_avoid_stage_per_env = torch.tensor([4])
        self.commands = torch.zeros(1, 3)
        self.commands_scale = torch.tensor([2.0, 3.0, 4.0])
        self.obs_buf = torch.zeros(1, 8)
        self.s_avoid_terminal_physical = torch.zeros(1, dtype=torch.bool)
        self.s_avoid_terminal_envelope = torch.zeros(1, dtype=torch.bool)
        self.s_avoid_terminal_fall = torch.zeros(1, dtype=torch.bool)
        self.s_avoid_terminal_success = torch.zeros(1, dtype=torch.bool)
        self.s_avoid_terminal_center_y = torch.zeros(1)
        self.reset_count = 0

    def step(self, action):
        self.root_states[:, 1] += self.commands[:, 1]
        done = torch.zeros(1, dtype=torch.bool)
        if self.mode == "failure_cross":
            done[:] = True
            self.s_avoid_terminal_physical[:] = True
            self.s_avoid_terminal_center_y[:] = self.s_avoid_exit_y + 1.0
            self.root_states[:, 1] = -9.0  # Low-level auto-reset overwrites root state.
        if self.mode == "fall_failure":
            done[:] = True
            self.s_avoid_terminal_fall[:] = True
        return self.obs_buf.clone(), torch.zeros(1), torch.zeros(1), done, {}

    def reset_idx(self, ids):
        self.reset_count += int(ids.numel())
        self.root_states[ids, 1] = -1.0

    def compute_observations(self):
        pass


class Harness:
    def __init__(self, mode="alive", slow_forward=False, clearance_side=0.0):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.beta_override = None
        self.env = MockEnv(mode)
        self.post_processor = MockPostProcessor(slow_forward)
        self.low_level_policy = MockLowLevel()
        self.decimation = 1
        self.high_level_dt = 0.1
        self.max_episode_length = 500
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.episode_return_buf = torch.zeros(1)
        self.episode_len_buf = torch.zeros(1, dtype=torch.long)
        self.forced_forward_speed = torch.tensor([0.3])
        self.avoid_virtual_bearing = torch.zeros(1)
        self.avoid_virtual_goal_valid = torch.ones(1, dtype=torch.bool)
        self.avoid_context_elapsed = torch.zeros(1, dtype=torch.long)
        self.avoid_context_hold_steps = torch.ones(1, dtype=torch.long)
        self.avoid_context_ramp_steps = torch.ones(1, dtype=torch.long)
        self.avoid_context_start_speed = self.forced_forward_speed.clone()
        self.avoid_context_target_speed = torch.tensor([0.15])
        self.avoid_context_start_bearing = torch.zeros(1)
        self.avoid_context_target_bearing = torch.zeros(1)
        self.clearance_side = float(clearance_side)
        self.last_obs = self._get_high_level_obs()

    def _compute_clearance_along_cmd(self, visible_map, cmd):
        side = cmd[:, 0].abs() > 0.01
        forward = (cmd[:, 0].abs() <= 0.01) & (cmd[:, 1] > 0.01)
        side_clearance = torch.where(
            (cmd[:, 0] * self.clearance_side > 0.0) | (self.clearance_side == 0.0),
            torch.full_like(cmd[:, 0], 0.65),
            torch.full_like(cmd[:, 0], 0.3),
        )
        return torch.where(side, side_clearance, torch.where(forward, torch.full_like(cmd[:, 0], 0.5), torch.ones_like(cmd[:, 0])))

    def _get_high_level_obs(self):
        return {
            "state": self.env.root_states[:, 1:2].clone(),
            "goal": torch.ones(1, 1),
            "drive": self.forced_forward_speed.clone(),
            "local_map_2ch": torch.zeros(1, 2, 4, 4),
        }

    def _refresh_depth_images(self, force=False):
        pass

    def _reset_idx(self, ids):
        self.env.reset_idx(ids)


def main():
    torch.manual_seed(7)
    raw = torch.randn(4, 13, requires_grad=True)
    actor = build_avoid_actor_state(raw, torch.tensor([.2, .3, .4, .5]))
    assert actor.shape == (4, 14) and torch.all(actor[:, [0, 1, 2, 6, 9]] == 0)
    policy_cmd_scale = (0.6,)
    policy = CmdVelExpert(affordance_channels=2, state_dim=14, goal_dim=2, action_dim=1, cmd_scale=policy_cmd_scale, actor_state_mask_indices=(0, 1, 2, 6, 9))
    action, _ = policy.get_action(torch.zeros(4, 2, 16, 16), actor, torch.zeros(4, 2), torch.zeros(4))
    log_prob, value, entropy, _ = policy.evaluate_actions(torch.zeros(4, 2, 16, 16), actor, torch.zeros(4, 2), torch.zeros(4), action)
    (log_prob.mean() + value.mean() + entropy.mean()).backward()
    assert action.shape == (4, 1) and torch.isfinite(action).all()
    assert all(torch.isfinite(param.grad).all() for param in policy.parameters() if param.grad is not None)

    assert _actual_save_checkpoint_condition(100, 1000, 200)
    assert _actual_save_checkpoint_condition(200, 1000, 200)
    assert not _actual_save_checkpoint_condition(998, 1000, 200)
    assert _actual_save_checkpoint_condition(999, 1000, 200)
    assert _actual_save_checkpoint_condition(0, 1, 200)
    assert _actual_save_checkpoint_condition(3999, 5000, 200, checkpoint_iterations=(3999,))
    assert not _actual_save_checkpoint_condition(3998, 5000, 200, checkpoint_iterations=(3999,))
    assert _actual_save_checkpoint_condition(4999, 5000, 200, checkpoint_iterations=(3999,))

    _resume_rng_restore_source()
    cpu_rng = torch.get_rng_state()
    torch.set_rng_state(cpu_rng.cpu())

    enforce_clutter_contract = _extract_clutter_risk_scale_contract()
    clutter_args = SimpleNamespace(task="s_avoid_clutter", disable_risk_scale=False)
    assert enforce_clutter_contract(clutter_args, context="test")
    assert clutter_args.disable_risk_scale
    other_args = SimpleNamespace(task="s_avoid_basic", disable_risk_scale=False)
    assert not enforce_clutter_contract(other_args, context="test")
    assert not other_args.disable_risk_scale

    post = CommandPostProcessor(
        max_cmd=(0.6, 0.6, 1.0),
        max_delta=(0.2, 0.2, 0.4),
        safe_distance=0.27,
        free_distance=0.57,
        enable_risk_scale=not clutter_args.disable_risk_scale,
    )
    post.reset(1, torch.device("cpu"))
    post.last_cmd[:] = torch.tensor([[0.2, 0.3, 0.0]])
    near_cmd = torch.tensor([[0.8, 0.6, 0.0]])
    native_clamp = torch.clamp
    # The local CPU fixture uses torch 1.8, which lacks tensor min/max support
    # for torch.clamp. Keep the production process path under test.
    def _compat_tensor_clamp(value, minimum=None, maximum=None, *, out=None):
        if out is None and torch.is_tensor(minimum) and torch.is_tensor(maximum):
            return torch.minimum(torch.maximum(value, minimum), maximum)
        return native_clamp(value, minimum, maximum, out=out)
    torch.clamp = _compat_tensor_clamp
    try:
        near_exec, near_info = post.process(near_cmd, clearance=torch.tensor([0.0]))
    finally:
        torch.clamp = native_clamp
    assert near_info["risk_scale"] is None
    assert torch.allclose(near_info["cmd_clamped"], torch.tensor([[0.6, 0.6, 0.0]]))
    assert torch.allclose(near_exec, torch.tensor([[0.4, 0.5, 0.0]]))
    assert torch.allclose(near_exec, near_info["cmd_slew_pre_scale"])

    def reward_clearance(current, straight, *, failure=False, success=False):
        return avoid_clutter_reward(
            torch.zeros(1), torch.tensor([current]), torch.tensor([straight]), torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool),
            torch.tensor([failure]), torch.tensor([success]),
        )["clearance"].item()

    assert reward_clearance(.8, .8) == 0.0
    assert reward_clearance(.285, .57) == 0.0
    assert reward_clearance(.2, .8) == 0.0
    assert abs(reward_clearance(.57, .285) - .0075) < 1e-8
    assert abs(reward_clearance(.1425, .285) + .00375) < 1e-8
    assert reward_clearance(.285, .285) == 0.0
    assert reward_clearance(.8, .3, success=True) == 0.0
    assert reward_clearance(.1, .3, failure=True) == 0.0

    for name, method in _extract_runtime_methods().items():
        setattr(Harness, name, method)
    slow = Harness(slow_forward=True)
    _, _, _, slow_info = slow._step_avoid_clutter(torch.tensor([[0.2]]))
    assert torch.allclose(slow.env.obs_buf[0, -3:], torch.tensor([0.4, 0.3, 0.0]))
    assert slow_info["post_info"]["cmd_exec_mean"][0, 1] == 0.1
    assert slow_info["reward_terms"]["smooth"][0] < 0.0

    # Stage-1 must preserve the virtual preference observation while omitting
    # only its reward contribution.  Stage-2 retains the frozen tie-break.
    stage1 = Harness()
    stage1.env.s_avoid_stage_per_env[:] = 1
    stage1.env.s_avoid_exit_y[:] = 99.0
    stage1_goal_before = stage1.last_obs["goal"].clone()
    stage1_next, _, _, stage1_info = stage1._step_avoid_clutter(torch.tensor([[0.2]]))
    assert stage1_info["reward_terms"]["preference"][0] == 0.0
    assert torch.equal(stage1_goal_before, stage1_next["goal"])

    stage2 = Harness()
    stage2.env.s_avoid_stage_per_env[:] = 2
    stage2.env.s_avoid_exit_y[:] = 99.0
    stage2_goal_before = stage2.last_obs["goal"].clone()
    stage2_next, _, _, stage2_info = stage2._step_avoid_clutter(torch.tensor([[0.2]]))
    assert stage2_info["reward_terms"]["preference"][0] != 0.0
    assert torch.equal(stage2_goal_before, stage2_next["goal"])

    left = Harness(clearance_side=-1.0)
    right = Harness(clearance_side=1.0)
    left.env.s_avoid_exit_y[:] = right.env.s_avoid_exit_y[:] = 99.0
    _, _, _, left_info = left._step_avoid_clutter(torch.tensor([[-0.2]]))
    _, _, _, right_info = right._step_avoid_clutter(torch.tensor([[0.2]]))
    left_wrong = Harness(clearance_side=-1.0)
    right_wrong = Harness(clearance_side=1.0)
    left_wrong.env.s_avoid_exit_y[:] = right_wrong.env.s_avoid_exit_y[:] = 99.0
    _, _, _, left_wrong_info = left_wrong._step_avoid_clutter(torch.tensor([[0.2]]))
    _, _, _, right_wrong_info = right_wrong._step_avoid_clutter(torch.tensor([[-0.2]]))
    assert left_info["reward_terms"]["clearance"][0] > 0.0
    assert right_info["reward_terms"]["clearance"][0] > 0.0
    assert left_wrong_info["reward_terms"]["clearance"][0] < 0.0
    assert right_wrong_info["reward_terms"]["clearance"][0] < 0.0

    alive = Harness()
    next_obs, _, done, alive_info = alive._step_avoid_clutter(torch.tensor([[0.2]]))
    assert done.item() and alive_info["success_mask"].item() and alive.env.reset_count == 1
    assert next_obs["state"][0, 0] == -1.0 and alive.last_obs is next_obs and alive.avoid_context_elapsed[0] == 0

    failed = Harness(mode="failure_cross")
    _, reward, done, failed_info = failed._step_avoid_clutter(torch.tensor([[0.2]]))
    assert done.item() and reward.item() == -20.0 and not failed_info["success_mask"].item()
    assert failed_info["collision_mask"].item() and failed.env.reset_count == 0
    assert failed_info["s_avoid_episode_collision"].item()

    fallen = Harness(mode="fall_failure")
    _, reward, done, fallen_info = fallen._step_avoid_clutter(torch.tensor([[0.2]]))
    assert done.item() and reward.item() == -20.0 and fallen_info["terminal_fall"].item()
    assert not fallen_info["collision_mask"].item()
    assert not fallen_info["s_avoid_episode_collision"].item()

    timed = Harness()
    timed.env.s_avoid_exit_y[:] = 99.0
    timed.episode_length_buf[:] = 499
    next_obs, _, done, timeout_info = timed._step_avoid_clutter(torch.tensor([[0.0]]))
    assert done.item() and timeout_info["timeout"].item() and timed.env.reset_count == 1
    assert timeout_info["timeout_bootstrap_obs"]["state"][0, 0] > next_obs["state"][0, 0]
    assert timeout_info["timeout_bootstrap_obs"]["drive"][0] == 0.15 and next_obs["drive"][0] == 0.3

    reset_speed = Harness()
    reset_speed.forced_forward_speed[:] = 0.5
    reset_speed.env.s_avoid_v_drive_nom[:] = 0.2
    reset_speed.env.s_avoid_stage_per_env[:] = 1
    reset_speed._resample_avoid_clutter_context(torch.tensor([0]), episode_reset=True)
    assert reset_speed.forced_forward_speed[0] == 0.2 and reset_speed.avoid_context_target_speed[0] == 0.2
    print("avoid clutter training: PASS")


if __name__ == "__main__":
    main()
