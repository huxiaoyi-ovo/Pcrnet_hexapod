#!/usr/bin/env python3
"""Finite CPU-PhysX wrapper smoke for the frozen 1-D Avoid baseline.

This creates the real task-registry environment and applies random, untrained
high-level actions through the frozen low-level checkpoint.  It deliberately
does not construct an optimizer, a rollout buffer, or a PPO update.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from isaacgym import gymapi  # Isaac Gym must load before torch.
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 9173
NUM_ENVS = 4
HIGH_LEVEL_STEPS = 24
OUTPUT_PATH = ROOT / "outputs" / "avoid_clutter_training_smoke" / "smoke_20260910.json"
EXPECTED_LOW_LEVEL_SHA256 = "75dfd7aae52b20f08ca9f654c37b2e20ba1e33f8b953737437ad774d122d1481"


def _finite(name, tensor):
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"non-finite {name}")


def _args():
    return SimpleNamespace(
        mode="teacher",
        skill="avoid",
        task="s_avoid_clutter",
        seed=SEED,
        num_envs=NUM_ENVS,
        decimation=5,
        low_level_ckpt=str(ROOT / "agents" / "low_level_best.pt"),
        aff_stack=1,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device="cpu",
        sim_device_id=0,
        rl_device="cpu",
        use_gpu=False,
        use_gpu_pipeline=False,
        headless=True,
        debug=False,
        camera_enable=False,
        revision_contract=True,
        cmd_slew_lin=0.2,
        cmd_slew_ang=0.4,
        cmd_safe_dist=None,
        cmd_free_dist=None,
        beta=None,
        disable_risk_scale=False,
        force_cmd_y=False,
    )


def main():
    # train_highlevel's internal Isaac argument parser must not see harness args.
    sys.argv = [sys.argv[0]]
    from legged_gym.pcr_observation import AVOID_ACTOR_MASK_INDICES
    from legged_gym.scripts import train_highlevel as highlevel

    torch.manual_seed(SEED)
    checkpoint_path = ROOT / "agents" / "low_level_best.pt"
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_sha256 != EXPECTED_LOW_LEVEL_SHA256:
        raise RuntimeError(f"unexpected low-level checkpoint SHA256: {checkpoint_sha256}")
    highlevel.import_modules()
    wrapper = highlevel.HierarchicalHexapodEnv(_args(), torch.device("cpu"))
    obs = wrapper.reset()
    expected = {
        "state": (NUM_ENVS, 14),
        "goal": (NUM_ENVS, 2),
        "actor_difficulty": (NUM_ENVS,),
    }
    for name, shape in expected.items():
        if tuple(obs[name].shape) != shape:
            raise RuntimeError(f"{name} shape {tuple(obs[name].shape)}, expected {shape}")
        _finite(name, obs[name])
    if tuple(obs["local_map_2ch"].shape[:2]) != (NUM_ENVS, 2):
        raise RuntimeError(f"local_map_2ch shape {tuple(obs['local_map_2ch'].shape)}")
    _finite("local_map_2ch", obs["local_map_2ch"])

    max_lateral = float(wrapper.post_processor.max_cmd[0].item())
    actor = highlevel.CmdVelExpert(
        affordance_channels=2,
        state_dim=14,
        goal_dim=2,
        cmd_scale=(max_lateral,),
        action_dim=1,
        actor_state_mask_indices=AVOID_ACTOR_MASK_INDICES,
    ).to(wrapper.device).eval()
    done_total = 0
    for _ in range(HIGH_LEVEL_STEPS):
        with torch.no_grad():
            action, _ = actor.get_action(
                obs["local_map_2ch"], obs["state"], obs["goal"], obs["actor_difficulty"]
            )
        if tuple(action.shape) != (NUM_ENVS, 1):
            raise RuntimeError(f"action shape {tuple(action.shape)}")
        _finite("action", action)
        obs, reward, done, info = wrapper.step(action)
        command = info["post_info"]["cmd_exec_mean"]
        if tuple(command.shape) != (NUM_ENVS, 3):
            raise RuntimeError(f"command shape {tuple(command.shape)}")
        _finite("command", command)
        _finite("reward", reward)
        _finite("next_state", obs["state"])
        done_total += int(done.sum().item())

    result = {
        "task": "s_avoid_clutter",
        "seed": SEED,
        "num_envs": NUM_ENVS,
        "high_level_steps": HIGH_LEVEL_STEPS,
        "physics": "CPU PhysX",
        "low_level_checkpoint": str(checkpoint_path),
        "low_level_checkpoint_sha256": checkpoint_sha256,
        "high_level_actor": "random CmdVelExpert initialization (untrained)",
        "state_shape": list(obs["state"].shape),
        "action_shape": [NUM_ENVS, 1],
        "command_shape": [NUM_ENVS, 3],
        "finite": True,
        "done_total": done_total,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": result, "output": str(OUTPUT_PATH)}, sort_keys=True))
    wrapper.env.gym.destroy_sim(wrapper.env.sim)


if __name__ == "__main__":
    main()
