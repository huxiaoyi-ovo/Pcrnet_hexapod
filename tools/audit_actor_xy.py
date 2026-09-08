#!/usr/bin/env python3
from __future__ import annotations

"""Read-only actor-state x/y intervention audit for an existing eval rollout.

The wrapped policy call returns the original live action unchanged.  Every Nth
call is copied under ``torch.no_grad`` and evaluated deterministically with
only actor ``state[:, 0:2]`` changed.  The copies never call the environment,
Follow expert, risk update, or ``env.step``.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
XY_VARIANTS: Tuple[Tuple[str, int, float], ...] = (
    ("x_minus_0p25", 0, -0.25),
    ("x_plus_0p25", 0, 0.25),
    ("y_minus_0p50", 1, -0.50),
    ("y_plus_0p50", 1, 0.50),
    ("y_abs_0", 1, 0.0),
    ("y_abs_2", 1, 2.0),
    ("y_abs_4", 1, 4.0),
    ("y_abs_6", 1, 6.0),
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tensor_copy(tensor: torch.Tensor, rows: int) -> np.ndarray:
    return tensor[:rows].detach().cpu().numpy().copy()


def _require_finite(label: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{label}: non-finite diagnostic tensor")


@dataclass
class ActorCollector:
    label: str
    sample_every: int
    max_frames: int
    calls: int = 0
    captured: int = 0
    records: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    def _append(self, key: str, value: torch.Tensor, rows: int) -> None:
        self.records.setdefault(key, []).append(_tensor_copy(value, rows))

    def wrap(self, original):
        def wrapped(module, affordance_map, robot_state, goal, terrain_difficulty,
                    deterministic=False, **kwargs):
            # Live action is evaluated first and returned without any changed input.
            live_action, live_info = original(
                affordance_map, robot_state, goal, terrain_difficulty,
                deterministic=deterministic, **kwargs
            )
            _require_finite(f"{self.label}: live action", live_action)
            self.calls += 1
            if self.calls % self.sample_every != 0 or self.captured >= self.max_frames:
                return live_action, live_info

            remaining = self.max_frames - self.captured
            rows = min(int(robot_state.shape[0]), remaining)
            if rows <= 0:
                return live_action, live_info
            # Clones are required: no live actor, critic, map, goal, difficulty, or
            # caller tensor is mutated by this diagnostic.
            actor_state = robot_state.detach().clone()
            _require_finite(f"{self.label}: actor state", actor_state)
            _require_finite(f"{self.label}: map", affordance_map)
            _require_finite(f"{self.label}: goal", goal)
            _require_finite(f"{self.label}: difficulty", terrain_difficulty)
            state_before = actor_state.detach().clone()
            map_before = affordance_map.detach().clone()
            goal_before = goal.detach().clone()
            difficulty_before = terrain_difficulty.detach().clone()
            probe_kwargs = dict(kwargs)
            # Keep the critic on the original actor state when the live call did
            # not supply a dedicated critic state.  Otherwise preserve its input.
            if probe_kwargs.get("critic_robot_state", None) is None:
                probe_kwargs["critic_robot_state"] = actor_state
            critic_before = probe_kwargs.get("critic_robot_state", None)
            if torch.is_tensor(critic_before):
                critic_before = critic_before.detach().clone()

            with torch.no_grad():
                baseline, _ = original(
                    affordance_map, actor_state, goal, terrain_difficulty,
                    deterministic=True, **probe_kwargs
                )
                _require_finite(f"{self.label}: deterministic baseline", baseline)
                if deterministic and not torch.allclose(
                    live_action[:rows], baseline[:rows], atol=1e-6, rtol=1e-6
                ):
                    raise RuntimeError(
                        f"{self.label}: deterministic baseline replay differs from live action"
                    )
                variants: Dict[str, torch.Tensor] = {}
                for name, axis, value in XY_VARIANTS:
                    changed_state = actor_state.clone()
                    if name.startswith("y_abs_"):
                        changed_state[:, axis] = value
                    else:
                        changed_state[:, axis] += value
                    variants[name], _ = original(
                        affordance_map, changed_state, goal, terrain_difficulty,
                        deterministic=True, **probe_kwargs
                    )
                    _require_finite(f"{self.label}: {name}", variants[name])

            if not torch.equal(actor_state, state_before):
                raise RuntimeError(f"{self.label}: actor state input was mutated")
            if not torch.equal(affordance_map, map_before):
                raise RuntimeError(f"{self.label}: map input was mutated")
            if not torch.equal(goal, goal_before) or not torch.equal(terrain_difficulty, difficulty_before):
                raise RuntimeError(f"{self.label}: goal or difficulty input was mutated")
            if torch.is_tensor(critic_before) and not torch.equal(probe_kwargs.get("critic_robot_state"), critic_before):
                raise RuntimeError(f"{self.label}: critic state input was mutated")

            self._append("state", actor_state, rows)
            self._append("affordance_map", affordance_map, rows)
            self._append("goal", goal, rows)
            self._append("difficulty", terrain_difficulty, rows)
            self._append("baseline_live", live_action, rows)
            self._append("baseline_deterministic", baseline, rows)
            for name, action in variants.items():
                self._append(name, action, rows)
            self.captured += rows
            return live_action, live_info

        return MethodType(wrapped, original.__self__)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            key: np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float32)
            for key, values in self.records.items()
        }


def _action_summary(label: str, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    baseline = arrays.get("baseline_deterministic", np.empty((0,), dtype=np.float32))
    if baseline.size == 0:
        return {"actor": label, "frames": 0}
    summary: Dict[str, Any] = {"actor": label, "frames": int(baseline.shape[0]), "variants": {}}
    state = arrays.get("state")
    if state is not None and state.ndim == 2 and state.shape[1] >= 2:
        summary["captured_state_xy_min_m"] = state[:, :2].min(axis=0).tolist()
        summary["captured_state_xy_max_m"] = state[:, :2].max(axis=0).tolist()
    for name, _axis, _value in XY_VARIANTS:
        changed = arrays[name]
        delta = changed - baseline
        abs_delta = np.abs(delta)
        item: Dict[str, Any] = {
            "action_delta_abs": {
                "mean": abs_delta.mean(axis=0).tolist(),
                "median": np.median(abs_delta, axis=0).tolist(),
                "p90": np.percentile(abs_delta, 90, axis=0).tolist(),
                "p95": np.percentile(abs_delta, 95, axis=0).tolist(),
                "max": abs_delta.max(axis=0).tolist(),
            }
        }
        if label == "avoid":
            lateral = baseline[:, 0]
            active = np.abs(lateral) > 0.02
            near_zero = ~active
            signs_changed = np.sign(changed[:, 0]) != np.sign(lateral)
            item["lateral_sign_flip"] = {
                "active_denominator": int(active.sum()),
                "active_rate": float(signs_changed[active].mean()) if active.any() else None,
                "near_zero_denominator": int(near_zero.sum()),
                "near_zero_rate": float(signs_changed[near_zero].mean()) if near_zero.any() else None,
            }
        else:
            item["gate_y_delta_abs"] = {
                stat: values[0] for stat, values in item["action_delta_abs"].items()
            }
            if baseline.ndim == 2 and baseline.shape[1] > 1:
                item["gate_w_delta_abs"] = {
                    stat: values[1] for stat, values in item["action_delta_abs"].items()
                }
        summary["variants"][name] = item
    return summary


def _checkpoint_args(eval_args: Sequence[str]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for option, label in (
        ("--ckpt", "gate_ckpt"),
        ("--pcr_ckpt", "gate_ckpt"),
        ("--avoid_ckpt", "avoid_ckpt"),
        ("--low_level_ckpt", "lowlevel_ckpt"),
    ):
        for idx, token in enumerate(eval_args[:-1]):
            if token == option:
                path = os.path.abspath(eval_args[idx + 1])
                result[label] = {"path": path, "sha256": _sha256(path) if os.path.isfile(path) else "missing"}
    return result


def _self_test() -> None:
    class Toy:
        def get_action(self, affordance_map, robot_state, goal, terrain_difficulty, deterministic=False, **kwargs):
            return robot_state[:, :2].clone(), {}

    collector = ActorCollector("avoid", sample_every=1, max_frames=4)
    toy = Toy()
    original = Toy.get_action
    Toy.get_action = _make_class_wrapper(original, collector)
    state = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    state_before = state.clone()
    amap = torch.zeros(4, 2, 2, 2)
    goal = torch.ones(4, 2)
    difficulty = torch.zeros(4)
    live, _ = toy.get_action(amap, state, goal, difficulty, deterministic=True)
    if not torch.equal(live, state[:, :2]) or not torch.equal(state, state_before):
        raise AssertionError("self-test: live action or caller state changed")
    arrays = collector.arrays()
    if not np.array_equal(arrays["x_plus_0p25"][:, 0], arrays["baseline_deterministic"][:, 0] + 0.25):
        raise AssertionError("self-test: x intervention changed an unexpected field")
    if not np.array_equal(arrays["y_abs_4"][:, 0], arrays["baseline_deterministic"][:, 0]):
        raise AssertionError("self-test: absolute y changed x")
    if not np.array_equal(arrays["y_abs_4"][:, 1], np.full(4, 4.0, dtype=np.float32)):
        raise AssertionError("self-test: absolute y was not applied")
    Toy.get_action = original
    print("audit_actor_xy self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-source", required=False, default="", help="absolute path to eval_highlevel.py")
    parser.add_argument("--audit-output", default="", help="new directory for .npz/.json only")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER, help="pass existing eval arguments after --")
    args = parser.parse_args()
    if args.eval_args and args.eval_args[0] == "--":
        args.eval_args = args.eval_args[1:]
    if args.sample_every <= 0 or args.max_frames <= 0:
        parser.error("--sample-every and --max-frames must be positive")
    if not args.self_test and (not args.eval_source or not args.audit_output):
        parser.error("--eval-source and --audit-output are required outside --self-test")
    return args


def main() -> None:
    global torch
    args = parse_args()
    if args.self_test:
        import torch as torch_module
        torch = torch_module
        _self_test()
        return

    eval_source = Path(args.eval_source).expanduser().resolve()
    if not eval_source.is_file():
        raise SystemExit(f"eval source not found: {eval_source}")
    project_root = eval_source.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import isaacgym  # noqa: F401  # Isaac Gym must initialize before torch.
    import torch as torch_module
    torch = torch_module
    from legged_gym.scripts import train_highlevel as th

    collectors = {
        "avoid": ActorCollector("avoid", args.sample_every, args.max_frames),
        "gate": ActorCollector("gate", args.sample_every, args.max_frames),
    }
    original_avoid = th.CmdVelExpert.get_action
    original_gate = th.GatePolicy.get_action
    th.CmdVelExpert.get_action = _make_class_wrapper(original_avoid, collectors["avoid"])
    th.GatePolicy.get_action = _make_class_wrapper(original_gate, collectors["gate"])

    sys.argv = [str(eval_source)] + list(args.eval_args)
    try:
        runpy.run_path(str(eval_source), run_name="__main__")
    finally:
        th.CmdVelExpert.get_action = original_avoid
        th.GatePolicy.get_action = original_gate

    output_dir = Path(args.audit_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "purpose": "read-only actor x/y intervention; live actions were returned unchanged",
        "eval_source": str(eval_source),
        "eval_source_git_sha": _git_sha(project_root),
        "diagnostic_script_sha256": _sha256(os.path.abspath(__file__)),
        "eval_args": list(args.eval_args),
        "sample_every_get_action_batches": args.sample_every,
        "max_frames_per_actor": args.max_frames,
        "variants": [{"name": name, "axis": axis, "value_m": value} for name, axis, value in XY_VARIANTS],
        "checkpoints": _checkpoint_args(args.eval_args),
        "actors": {},
    }
    for label, collector in collectors.items():
        arrays = collector.arrays()
        np.savez_compressed(output_dir / f"{label}_actor_xy.npz", **arrays)
        metadata["actors"][label] = {
            "get_action_calls": collector.calls,
            "captured_frames": collector.captured,
            "summary": _action_summary(label, arrays),
        }
    with open(output_dir / "audit_actor_xy_summary.json", "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _make_class_wrapper(original, collector: ActorCollector):
    def class_wrapper(module, *args, **kwargs):
        return collector.wrap(original.__get__(module, type(module)))(*args, **kwargs)
    return class_wrapper


if __name__ == "__main__":
    main()
