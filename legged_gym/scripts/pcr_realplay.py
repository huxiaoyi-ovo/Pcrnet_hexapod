#!/usr/bin/env python3
"""Run PCR learned-w policy with real ROS1 inputs.

This is the real-robot entry point.  It intentionally keeps the same policy
semantics used by PCR training:

    goal_buf = (x_right, y_forward)
    cmd = [cmd_x_right, cmd_y_forward, cmd_yaw]

ROS Twist publishing maps this to:

    linear.x  <- cmd_y_forward
    linear.y  <- cmd_x_right
    angular.z <- cmd_yaw

The default is dry-run: no motion command is published unless --publish_cmd is
set explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class RealInputSnapshot:
    target: Optional[np.ndarray] = None
    local_map_2ch: Optional[np.ndarray] = None
    state: Optional[np.ndarray] = None
    row_not_released: Optional[float] = None
    target_too_close: bool = False
    depth_invalid: bool = False
    target_stamp: float = 0.0
    local_map_stamp: float = 0.0
    state_stamp: float = 0.0
    row_stamp: float = 0.0


class RealPcrRuntimeError(RuntimeError):
    pass


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("pcr_realplay.py requires torch in the active Python environment.") from exc
    return torch


def _to_state_dict(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"checkpoint object must be dict-like, got {type(ckpt_obj)}")


def _ckpt_meta(ckpt_obj: Any) -> Dict[str, Any]:
    if isinstance(ckpt_obj, dict) and isinstance(ckpt_obj.get("experiment_meta", None), dict):
        return dict(ckpt_obj["experiment_meta"])
    return {}


def _load_ckpt(path: str, torch_mod, device):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch_mod.load(path, map_location=device)


def _infer_state_dim(ckpt_obj: Any, torch_mod) -> Optional[int]:
    state = _to_state_dict(ckpt_obj)
    for key in ("state_encoder.mlp.0.weight", "module.state_encoder.mlp.0.weight"):
        value = state.get(key, None)
        if torch_mod.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def _infer_goal_dim(ckpt_obj: Any, torch_mod) -> Optional[int]:
    state = _to_state_dict(ckpt_obj)
    for key in ("goal_encoder.mlp.0.weight", "module.goal_encoder.mlp.0.weight"):
        value = state.get(key, None)
        if torch_mod.is_tensor(value) and value.ndim == 2:
            return int(value.shape[1])
    return None


def _infer_affordance_channels(ckpt_obj: Any, torch_mod) -> Optional[int]:
    state = _to_state_dict(ckpt_obj)
    for key in ("affordance_encoder.cnn.0.weight", "module.affordance_encoder.cnn.0.weight"):
        value = state.get(key, None)
        if torch_mod.is_tensor(value) and value.ndim == 4:
            return int(value.shape[1]) - 2
    return None


def _infer_gate_action_dim(ckpt_obj: Any, meta: Dict[str, Any], torch_mod) -> Optional[int]:
    dim = meta.get("actor_output_dim", None)
    if dim is not None:
        return int(dim)
    state = _to_state_dict(ckpt_obj)
    for key in ("w_alpha_head.2.weight", "module.w_alpha_head.2.weight"):
        value = state.get(key, None)
        if torch_mod.is_tensor(value):
            return 2
    for key in ("y_alpha_head.2.weight", "module.y_alpha_head.2.weight"):
        value = state.get(key, None)
        if torch_mod.is_tensor(value):
            return 1
    return None


def _load_high_level_state_dict_compat(model, state_dict: Dict[str, Any], torch_mod, label: str) -> None:
    state = dict(state_dict)
    model_state = model.state_dict()
    mirror_pairs = [
        ("affordance_encoder", "critic_affordance_encoder"),
        ("state_encoder", "critic_state_encoder"),
        ("goal_encoder", "critic_goal_encoder"),
        ("fusion", "critic_fusion"),
    ]
    for src_prefix, dst_prefix in mirror_pairs:
        src_token = src_prefix + "."
        dst_token = dst_prefix + "."
        for key, value in list(state.items()):
            if not key.startswith(src_token):
                continue
            dst_key = dst_token + key[len(src_token):]
            if dst_key in state or dst_key not in model_state:
                continue
            if torch_mod.is_tensor(value) and model_state[dst_key].shape == value.shape:
                state[dst_key] = value.clone()

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RealPcrRuntimeError(f"{label} checkpoint incompatible: missing={missing}, unexpected={unexpected}")
    for param in model.parameters():
        if not torch_mod.isfinite(param).all():
            raise RealPcrRuntimeError(f"{label} checkpoint has non-finite parameters")


def _match_2d_tensor(tensor, target_dim: int, torch_mod, *, label: str):
    if tensor.dim() != 2:
        raise ValueError(f"{label} shape invalid: {tuple(tensor.shape)}")
    current = int(tensor.shape[1])
    target = int(target_dim)
    if current == target:
        return tensor
    if current < target:
        pad = torch_mod.zeros(tensor.shape[0], target - current, device=tensor.device, dtype=tensor.dtype)
        return torch_mod.cat([tensor, pad], dim=1)
    raise ValueError(f"{label} dim mismatch: runtime={current}, checkpoint={target}; refuse to truncate")


def _sanitize_array(values: np.ndarray, *, shape: Tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _actor_difficulty_from_local_map(local_map_2ch: np.ndarray, map_extent_m: float, radius_m: float) -> float:
    occ = np.clip(local_map_2ch[0], 0.0, 1.0)
    clearance = np.clip(local_map_2ch[1], 0.0, 1.0)
    n = int(occ.shape[0])
    cell = float(map_extent_m) / float(max(n, 1))
    x_centers = np.linspace(-0.5 * map_extent_m + 0.5 * cell, 0.5 * map_extent_m - 0.5 * cell, n)
    y_centers = np.linspace(0.5 * cell, map_extent_m - 0.5 * cell, n)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    mask = ((grid_x ** 2 + grid_y ** 2) <= max(float(radius_m), 1e-3) ** 2).astype(np.float32)
    denom = max(float(mask.sum()), 1.0)
    occ_ratio = float((occ * mask).sum() / denom)
    clearance_cost = float(((1.0 - clearance) * mask).sum() / denom)
    return float(np.clip(0.5 * occ_ratio + 0.5 * clearance_cost, 0.0, 1.0))


class RealPcrPolicyShim:
    """Small environment-like object required by PCR gate helpers."""

    def __init__(self, args: argparse.Namespace, torch_mod, device):
        self.args = args
        self.torch = torch_mod
        self.device = device
        self.affordance_map_size = int(args.map_size)
        self.affordance_map_extent = float(args.map_extent_m)
        self.affordance_difficulty_radius = float(args.difficulty_radius_m)
        self.is_pcr_line_task = False
        self.row_not_released_value = float(args.row_not_released_default)
        self._dist_map = self._build_dist_map()
        self._x_map, self._y_map, self._bearing_map = self._build_geometry()

    def _build_geometry(self):
        n = self.affordance_map_size
        extent = self.affordance_map_extent
        cell = extent / float(n)
        x = self.torch.linspace(-0.5 * extent + 0.5 * cell, 0.5 * extent - 0.5 * cell, n, device=self.device)
        y = self.torch.linspace(0.5 * cell, extent - 0.5 * cell, n, device=self.device)
        grid_x, grid_y = self.torch.meshgrid(x, y, indexing="ij")
        bearing = self.torch.atan2(grid_x, grid_y)
        return grid_x, grid_y, bearing

    def _build_dist_map(self):
        n = self.affordance_map_size
        extent = self.affordance_map_extent
        cell = extent / float(n)
        x = self.torch.linspace(-0.5 * extent + 0.5 * cell, 0.5 * extent - 0.5 * cell, n, device=self.device)
        y = self.torch.linspace(0.5 * cell, extent - 0.5 * cell, n, device=self.device)
        grid_x, grid_y = self.torch.meshgrid(x, y, indexing="ij")
        return self.torch.sqrt(grid_x ** 2 + grid_y ** 2)

    @staticmethod
    def _risk_from_clearance(clearance, safe_distance: float, free_distance: float):
        safe = float(safe_distance)
        free = max(float(free_distance), safe + 1e-6)
        x = (clearance - safe) / (free - safe)
        return 1.0 - x.clamp(0.0, 1.0)

    def _compute_clearance_along_cmd(self, aff_map, cmd_xy, cone_half_angle_deg: float = 25.0):
        if aff_map.ndim != 4 or aff_map.shape[1] < 1:
            raise ValueError(f"aff_map shape invalid: {tuple(aff_map.shape)}")
        occ = aff_map[:, 0] > 0.5
        dist = self._dist_map.to(device=aff_map.device, dtype=aff_map.dtype)
        bearing = self._bearing_map.to(device=aff_map.device, dtype=aff_map.dtype)
        out = []
        cone = math.radians(float(cone_half_angle_deg))
        for i in range(cmd_xy.shape[0]):
            cmd = cmd_xy[i]
            speed = self.torch.norm(cmd)
            if float(speed.detach().cpu().item()) < 1e-6:
                out.append(self.torch.full((), self.affordance_map_extent, device=aff_map.device, dtype=aff_map.dtype))
                continue
            cmd_bearing = self.torch.atan2(cmd[0], cmd[1])
            diff = self.torch.atan2(self.torch.sin(bearing - cmd_bearing), self.torch.cos(bearing - cmd_bearing))
            sector = self.torch.abs(diff) <= cone
            selected = occ[i] & sector
            if bool(selected.any().detach().cpu().item()):
                out.append(dist[selected].min())
            else:
                out.append(self.torch.full((), self.affordance_map_extent, device=aff_map.device, dtype=aff_map.dtype))
        return self.torch.stack(out, dim=0)


class PcrRealplay:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.torch = _load_torch()
        self.device = self.torch.device(args.device if args.device else ("cuda" if self.torch.cuda.is_available() else "cpu"))
        self.snapshot = RealInputSnapshot()
        self.prev_cmd = np.zeros(3, dtype=np.float32)
        self.prev_cmd_stamp = time.time()
        self.risk_memory = None
        self.bridge = RealPcrPolicyShim(args, self.torch, self.device)
        self._load_models()

        self.rospy = None
        self.cmd_pub = None
        self.debug_pub = None
        self._ros_msg_classes = {}
        if not args.fake_input:
            self._setup_ros()

    def _load_models(self) -> None:
        from rsl_rl.algorithms.high_level_planner import CmdVelExpert, GatePolicy

        for path_name in ("pcr_ckpt", "avoid_ckpt"):
            path = getattr(self.args, path_name)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"missing required --{path_name}: {path}")
        if self.args.lowlevel_ckpt and (not os.path.exists(self.args.lowlevel_ckpt)):
            raise FileNotFoundError(f"lowlevel checkpoint hint not found: {self.args.lowlevel_ckpt}")

        gate_ckpt = _load_ckpt(self.args.pcr_ckpt, self.torch, self.device)
        gate_meta = _ckpt_meta(gate_ckpt)
        trained_w_mode = str(gate_meta.get("trained_w_mode", gate_meta.get("w_mode", "learned"))).lower()
        if trained_w_mode not in ("learned", "learnedw2"):
            raise ValueError(f"pcr_realplay requires learned-w checkpoint, got trained_w_mode={trained_w_mode}")
        requested_w_mode = str(self.args.w_mode).lower()
        if requested_w_mode == "auto":
            self.args.w_mode = trained_w_mode
        elif requested_w_mode != trained_w_mode:
            raise ValueError(
                f"--w_mode={self.args.w_mode} mismatches checkpoint trained_w_mode={trained_w_mode}"
            )
        for key in (
            "w_blend_mode",
            "signed_w_lambda",
            "signed_w_gamma_risk",
            "signed_w_margin",
            "risk_memory",
            "risk_memory_l_clear",
            "risk_memory_velocity_source",
        ):
            if key in gate_meta and gate_meta[key] is not None:
                setattr(self.args, key, gate_meta[key])
        action_dim = _infer_gate_action_dim(gate_ckpt, gate_meta, self.torch)
        if action_dim is not None and int(action_dim) != 2:
            raise ValueError(f"pcr ckpt is not learned-w: actor_output_dim={action_dim}")
        self.gate_state_dim = _infer_state_dim(gate_ckpt, self.torch) or int(self.args.state_dim)
        self.gate_goal_dim = _infer_goal_dim(gate_ckpt, self.torch) or (2 + 16)
        gate_aff_channels = _infer_affordance_channels(gate_ckpt, self.torch)
        if gate_aff_channels is None:
            gate_aff_channels = int(self.args.map_channels * self.args.aff_stack)
        self.gate_aff_channels = int(gate_aff_channels)

        self.gate_policy = GatePolicy(
            affordance_channels=self.gate_aff_channels,
            state_dim=self.gate_state_dim,
            goal_dim=self.gate_goal_dim,
            learned_w=True,
        ).to(self.device)
        _load_high_level_state_dict_compat(
            self.gate_policy,
            _to_state_dict(gate_ckpt),
            self.torch,
            label="pcr_gate",
        )
        self.gate_policy.eval()

        avoid_ckpt = _load_ckpt(self.args.avoid_ckpt, self.torch, self.device)
        self.avoid_state_dim = _infer_state_dim(avoid_ckpt, self.torch) or self.gate_state_dim
        self.avoid_goal_dim = _infer_goal_dim(avoid_ckpt, self.torch) or 2
        avoid_aff_channels = _infer_affordance_channels(avoid_ckpt, self.torch)
        if avoid_aff_channels is None:
            avoid_aff_channels = int(self.args.map_channels * self.args.aff_stack)
        self.avoid_aff_channels = int(avoid_aff_channels)
        cmd_scale = tuple(float(v) for v in self.args.cmd_scale.split(","))
        if len(cmd_scale) != 3:
            raise ValueError("--cmd_scale must contain three comma-separated values")
        self.cmd_scale = cmd_scale
        self.avoid_model = CmdVelExpert(
            affordance_channels=self.avoid_aff_channels,
            state_dim=self.avoid_state_dim,
            goal_dim=self.avoid_goal_dim,
            cmd_scale=cmd_scale,
        ).to(self.device)
        _load_high_level_state_dict_compat(
            self.avoid_model,
            _to_state_dict(avoid_ckpt),
            self.torch,
            label="avoid_expert",
        )
        self.avoid_model.eval()

        max_needed = max(self.gate_aff_channels, self.avoid_aff_channels)
        if max_needed % int(self.args.map_channels) != 0:
            raise ValueError(
                f"checkpoint affordance channels={max_needed} is not divisible by map_channels={self.args.map_channels}"
            )
        self.aff_stack = max_needed // int(self.args.map_channels)
        print(
            "[PCRRealplay] loaded models: "
            f"gate_state_dim={self.gate_state_dim}, gate_goal_dim={self.gate_goal_dim}, "
            f"gate_aff_channels={self.gate_aff_channels}, avoid_aff_channels={self.avoid_aff_channels}, "
            f"aff_stack={self.aff_stack}, lowlevel_hint={self.args.lowlevel_ckpt}",
            flush=True,
        )

    def _setup_ros(self) -> None:
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Float32, Float32MultiArray, String
        except ImportError as exc:
            raise SystemExit("ROS1 Python packages are required unless --fake_input is used.") from exc
        self.rospy = rospy
        self._ros_msg_classes = {
            "Twist": Twist,
            "Float32": Float32,
            "Float32MultiArray": Float32MultiArray,
            "String": String,
        }
        rospy.init_node(self.args.ros_node_name, anonymous=False)
        rospy.Subscriber(self.args.target_topic, Float32MultiArray, self._target_cb, queue_size=1)
        rospy.Subscriber(self.args.local_map_topic, Float32MultiArray, self._local_map_cb, queue_size=1)
        if self.args.state_topic:
            rospy.Subscriber(self.args.state_topic, Float32MultiArray, self._state_cb, queue_size=1)
        if self.args.row_not_released_topic:
            rospy.Subscriber(self.args.row_not_released_topic, Float32, self._row_cb, queue_size=1)
        self.debug_pub = rospy.Publisher(self.args.debug_topic, String, queue_size=1)
        if self.args.publish_cmd:
            self.cmd_pub = rospy.Publisher(self.args.cmd_topic, Twist, queue_size=1)
        print(
            "[PCRRealplay] ROS ready: "
            f"target={self.args.target_topic}, local_map={self.args.local_map_topic}, "
            f"state={self.args.state_topic or '<zeros>'}, publish_cmd={self.args.publish_cmd}",
            flush=True,
        )

    def _target_cb(self, msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if data.size < 5:
            print(
                f"[PCRRealplay][Warn] target message too short: got {data.size}, expected >=5; command will stop.",
                flush=True,
            )
            self.snapshot.target = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.snapshot.target_too_close = False
            self.snapshot.depth_invalid = True
            self.snapshot.target_stamp = time.time()
            return
        if not np.isfinite(data[:5]).all():
            print(
                "[PCRRealplay][Warn] target message has non-finite values; command will stop.",
                flush=True,
            )
            self.snapshot.target = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.snapshot.target_too_close = False
            self.snapshot.depth_invalid = True
            self.snapshot.target_stamp = time.time()
            return
        self.snapshot.target = data[:5].copy()
        self.snapshot.target_too_close = bool(data[5] > 0.5) if data.size >= 6 else False
        self.snapshot.depth_invalid = bool(data[6] > 0.5) if data.size >= 7 else False
        self.snapshot.target_stamp = time.time()

    def _local_map_cb(self, msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        expected = int(self.args.map_channels * self.args.map_size * self.args.map_size)
        if data.size != expected:
            print(
                f"[PCRRealplay][Warn] local_map message size mismatch: got {data.size}, expected {expected}; command will stop.",
                flush=True,
            )
            self.snapshot.local_map_2ch = None
            self.snapshot.local_map_stamp = 0.0
            return
        self.snapshot.local_map_2ch = data.reshape(self.args.map_channels, self.args.map_size, self.args.map_size).copy()
        self.snapshot.local_map_stamp = time.time()

    def _state_cb(self, msg) -> None:
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if data.size <= 0:
            print("[PCRRealplay][Warn] state message is empty; command will use zero state if allowed.", flush=True)
            self.snapshot.state = None
            self.snapshot.state_stamp = 0.0
            return
        if not np.isfinite(data).all():
            print("[PCRRealplay][Warn] state message has non-finite values; command will use zero state if allowed.", flush=True)
            self.snapshot.state = None
            self.snapshot.state_stamp = 0.0
            return
        self.snapshot.state = data.copy()
        self.snapshot.state_stamp = time.time()

    def _row_cb(self, msg) -> None:
        self.snapshot.row_not_released = float(msg.data)
        self.snapshot.row_stamp = time.time()

    def _fake_snapshot(self) -> RealInputSnapshot:
        target = np.asarray([self.args.fake_x_right, self.args.fake_y_forward, 0.0, 0.0, 1.0], dtype=np.float32)
        local_map = np.zeros((self.args.map_channels, self.args.map_size, self.args.map_size), dtype=np.float32)
        local_map[1, :, :] = 1.0
        if self.args.fake_obstacle:
            ix = int(np.clip(round((self.args.fake_obstacle_x + 0.5 * self.args.map_extent_m) / self.args.map_extent_m * self.args.map_size), 0, self.args.map_size - 1))
            iy = int(np.clip(round(self.args.fake_obstacle_y / self.args.map_extent_m * self.args.map_size), 0, self.args.map_size - 1))
            local_map[0, ix, iy] = 1.0
            local_map[1, max(0, ix - 1):min(self.args.map_size, ix + 2), max(0, iy - 1):min(self.args.map_size, iy + 2)] = 0.0
        state = np.zeros((int(self.args.state_dim),), dtype=np.float32)
        now = time.time()
        return RealInputSnapshot(
            target=target,
            local_map_2ch=local_map,
            state=state,
            row_not_released=float(self.args.row_not_released_default),
            target_too_close=False,
            depth_invalid=False,
            target_stamp=now,
            local_map_stamp=now,
            state_stamp=now,
            row_stamp=now,
        )

    def _get_snapshot(self) -> RealInputSnapshot:
        if self.args.fake_input:
            return self._fake_snapshot()
        return self.snapshot

    def _build_tensors(self, snap: RealInputSnapshot):
        now = time.time()
        if snap.target is None or (now - snap.target_stamp) > float(self.args.input_timeout_s):
            raise RealPcrRuntimeError("target input missing or stale")
        if snap.local_map_2ch is None or (now - snap.local_map_stamp) > float(self.args.input_timeout_s):
            raise RealPcrRuntimeError("local_map_2ch input missing or stale")
        target = np.asarray(snap.target, dtype=np.float32).reshape(-1)
        if target.size < 5:
            raise RealPcrRuntimeError("target input must be [x_right,y_forward,v_right,v_forward,valid]")
        if not np.isfinite(target[:5]).all():
            raise RealPcrRuntimeError("target input contains non-finite values")
        target_valid = bool(target[4] > 0.5)
        goal_np = np.asarray([[target[0], target[1]]], dtype=np.float32)
        local_np = _sanitize_array(
            snap.local_map_2ch,
            shape=(self.args.map_channels, self.args.map_size, self.args.map_size),
            name="local_map_2ch",
        )
        if snap.state is None or (now - snap.state_stamp) > float(self.args.state_timeout_s):
            if not self.args.allow_missing_state:
                raise RealPcrRuntimeError("state input missing or stale")
            state_np = np.zeros((self.gate_state_dim,), dtype=np.float32)
        else:
            state_np = np.asarray(snap.state, dtype=np.float32).reshape(-1)
            if not np.isfinite(state_np).all():
                raise RealPcrRuntimeError("state input contains non-finite values")

        row_not_released = snap.row_not_released
        if row_not_released is None or (now - snap.row_stamp) > float(self.args.row_timeout_s):
            row_not_released = float(self.args.row_not_released_default)
        self.bridge.row_not_released_value = float(np.clip(row_not_released, 0.0, 1.0))

        actor_difficulty = _actor_difficulty_from_local_map(
            local_np,
            float(self.args.map_extent_m),
            float(self.args.difficulty_radius_m),
        )

        torch_mod = self.torch
        goal = torch_mod.as_tensor(goal_np, device=self.device, dtype=torch_mod.float32)
        local = torch_mod.as_tensor(local_np[None, :, :, :], device=self.device, dtype=torch_mod.float32)
        state = torch_mod.as_tensor(state_np[None, :], device=self.device, dtype=torch_mod.float32)
        difficulty = torch_mod.as_tensor([actor_difficulty], device=self.device, dtype=torch_mod.float32)
        return (
            state,
            goal,
            local,
            difficulty,
            target_valid,
            bool(snap.target_too_close),
            bool(snap.depth_invalid),
            actor_difficulty,
        )

    def _stack_map(self, local_map):
        repeat = max(1, int(self.aff_stack))
        return local_map.repeat(1, repeat, 1, 1)

    def _update_risk_memory(self, risk_f, cmd_f, state_tensor):
        torch_mod = self.torch
        if not bool(getattr(self.args, "risk_memory", False)):
            return torch_mod.zeros_like(risk_f)
        if self.risk_memory is None or self.risk_memory.shape != risk_f.shape:
            self.risk_memory = torch_mod.zeros_like(risk_f)
        source = str(getattr(self.args, "risk_memory_velocity_source", "body")).lower()
        if source == "body" and state_tensor.shape[1] >= 5:
            v_forward = state_tensor[:, 4].to(device=risk_f.device, dtype=risk_f.dtype)
        elif cmd_f.shape[1] >= 2:
            v_forward = cmd_f[:, 1].to(device=risk_f.device, dtype=risk_f.dtype)
        else:
            v_forward = torch_mod.zeros_like(risk_f)
        dt = max(float(getattr(self.args, "high_level_dt", 0.10)), 1e-6)
        l_clear = max(float(getattr(self.args, "risk_memory_l_clear", 0.40)), 1e-6)
        delta_s = torch_mod.clamp(v_forward, min=0.0) * dt
        decay = torch_mod.exp(-delta_s / l_clear)
        self.risk_memory = torch_mod.maximum(torch_mod.clamp(risk_f, 0.0, 1.0), self.risk_memory * decay).detach()
        return self.risk_memory

    def _learned_w_goal(self, base_goal, aff_map, cmd_f, cmd_a, state_tensor):
        torch_mod = self.torch
        cmd_a_eff = cmd_a.clone()
        if cmd_a_eff.shape[-1] >= 2:
            cmd_a_eff[:, 1] = 0.0
        if cmd_a_eff.shape[-1] >= 3:
            cmd_a_eff[:, 2] = 0.0
        clearance_f = self.bridge._compute_clearance_along_cmd(aff_map, cmd_f[:, :2])
        clearance_a = self.bridge._compute_clearance_along_cmd(aff_map, cmd_a_eff[:, :2])
        risk_f = self.bridge._risk_from_clearance(clearance_f, self.args.cmd_safe_dist, self.args.cmd_free_dist)
        risk_a = self.bridge._risk_from_clearance(clearance_a, self.args.cmd_safe_dist, self.args.cmd_free_dist)
        lin_f = cmd_f[:, :2]
        lin_a = cmd_a_eff[:, :2]
        norm_f = torch_mod.norm(lin_f, dim=1)
        norm_a = torch_mod.norm(lin_a, dim=1)
        denom = torch_mod.clamp(norm_f * norm_a, min=1e-6)
        cmd_cos = torch_mod.sum(lin_f * lin_a, dim=1) / denom
        cmd_cos = torch_mod.where((norm_f > 1e-6) & (norm_a > 1e-6), cmd_cos, torch_mod.ones_like(cmd_cos))
        cmd_cos = torch_mod.clamp(cmd_cos, -1.0, 1.0)
        conflict_score = torch_mod.clamp(risk_f - risk_a, min=0.0) * (1.0 - cmd_cos) * 0.5
        row_not_released = torch_mod.full_like(risk_f, float(self.bridge.row_not_released_value))
        risk_memory = self._update_risk_memory(risk_f, cmd_f, state_tensor)
        row_actor = risk_memory
        scale = max(float(self.args.map_extent_m), 1e-6)
        features = torch_mod.cat(
            [
                cmd_f[:, :3],
                cmd_a_eff[:, :3],
                cmd_f[:, :3] - cmd_a_eff[:, :3],
                risk_f.unsqueeze(-1),
                risk_a.unsqueeze(-1),
                row_actor.unsqueeze(-1),
                cmd_cos.unsqueeze(-1),
                torch_mod.clamp(conflict_score, 0.0, 1.0).unsqueeze(-1),
                torch_mod.clamp(clearance_f, 0.0, scale).unsqueeze(-1) / scale,
                torch_mod.clamp(clearance_a, 0.0, scale).unsqueeze(-1) / scale,
            ],
            dim=-1,
        )
        goal = torch_mod.cat([base_goal, features], dim=-1)
        diag = {
            "cmd_a_eff": cmd_a_eff,
            "clearance_F": clearance_f,
            "clearance_A": clearance_a,
            "risk_F": risk_f,
            "risk_A": risk_a,
            "risk_memory": risk_memory,
            "row_not_released": row_not_released,
            "cmd_cos": cmd_cos,
            "conflict_score": torch_mod.clamp(conflict_score, 0.0, 1.0),
        }
        return goal, diag

    def _resolve_gate(self, gate_y_raw, cmd_f, cmd_a, learned_w, diag):
        torch_mod = self.torch
        if gate_y_raw.dim() == 2 and gate_y_raw.shape[-1] == 1:
            gate_y_raw = gate_y_raw.squeeze(-1)
        if learned_w.dim() == 2 and learned_w.shape[-1] == 1:
            learned_w = learned_w.squeeze(-1)
        w = torch_mod.clamp(learned_w, 0.0, 1.0)
        gate_y = torch_mod.clamp(gate_y_raw, 0.0, 1.0)
        cmd_a_eff = diag["cmd_a_eff"]
        signed_w = torch_mod.zeros_like(w)
        signed_w_active = torch_mod.zeros_like(w)
        w_corr = torch_mod.zeros_like(w)
        risk_corr = torch_mod.zeros_like(w)
        if str(self.args.w_mode).lower() == "learnedw2":
            signed_w = 2.0 * w - 1.0
            signed_margin = float(getattr(self.args, "signed_w_margin", 0.05))
            signed_w_active = torch_mod.where(
                torch_mod.abs(signed_w) > signed_margin,
                signed_w,
                torch_mod.zeros_like(signed_w),
            )
            w_corr = float(self.args.signed_w_lambda) * signed_w_active
            risk_corr = float(self.args.signed_w_gamma_risk) * (diag["risk_A"] - diag["risk_F"])
            y_eff = torch_mod.clamp(gate_y + w_corr + risk_corr, 0.0, 1.0)
        elif str(self.args.w_blend_mode).lower() == "mix":
            y_eff = torch_mod.clamp(0.5 * gate_y + 0.5 * (1.0 - w), 0.0, 1.0)
        else:
            y_eff = torch_mod.clamp(gate_y * (1.0 - w), 0.0, 1.0)
        cmd = y_eff.unsqueeze(-1) * cmd_f + (1.0 - y_eff.unsqueeze(-1)) * cmd_a_eff
        return {
            "cmd": cmd,
            "gate_y_raw": gate_y_raw,
            "gate_y": gate_y,
            "w": w,
            "signed_w": signed_w,
            "signed_w_active": signed_w_active,
            "w_support_correction": w_corr,
            "risk_diff_correction": risk_corr,
            "y_eff": y_eff,
            "cmd_f": cmd_f,
            "cmd_a": cmd_a_eff,
            **diag,
        }

    def policy_step(self, snap: RealInputSnapshot) -> Dict[str, Any]:
        from legged_gym.envs.hex_v4.expert_s0_follow import compute_s0_follow_expert_cmd

        state, goal, local_map, difficulty, target_valid, target_too_close, depth_invalid, actor_difficulty = self._build_tensors(snap)
        local_stack = self._stack_map(local_map)
        torch_mod = self.torch
        with torch_mod.no_grad():
            expert_state = _match_2d_tensor(state, self.avoid_state_dim, torch_mod, label="real_expert_state")
            gate_state = _match_2d_tensor(state, self.gate_state_dim, torch_mod, label="real_gate_state")
            robot_pos = expert_state[:, :2]
            robot_heading = expert_state[:, 2]
            cos_h = torch_mod.cos(robot_heading)
            sin_h = torch_mod.sin(robot_heading)
            # goal = (x_right, y_forward), heading=0 => forward is world +Y.
            delta_world_x = cos_h * goal[:, 0] + sin_h * goal[:, 1]
            delta_world_y = -sin_h * goal[:, 0] + cos_h * goal[:, 1]
            target_world = robot_pos + torch_mod.stack([delta_world_x, delta_world_y], dim=1)
            cmd_f = compute_s0_follow_expert_cmd(
                robot_pos_world_xy=robot_pos,
                robot_heading=robot_heading,
                target_world_xy=target_world,
                target_vel_world_xy=None,
                target_heading=None,
                cmd_scale=self.cmd_scale,
                reset_mask=None,
            )
            avoid_goal = _match_2d_tensor(goal, self.avoid_goal_dim, torch_mod, label="real_avoid_goal")
            cmd_a, _ = self.avoid_model.get_action(
                local_stack[:, : self.avoid_aff_channels, :, :],
                expert_state,
                avoid_goal,
                difficulty,
                deterministic=True,
            )
            gate_goal, diag = self._learned_w_goal(goal, local_map, cmd_f, cmd_a, gate_state)
            gate_goal = _match_2d_tensor(gate_goal, self.gate_goal_dim, torch_mod, label="real_gate_goal")
            gate_action, _ = self.gate_policy.get_action(
                local_stack[:, : self.gate_aff_channels, :, :],
                gate_state,
                gate_goal,
                difficulty,
                deterministic=True,
            )
            if gate_action.dim() != 2 or gate_action.shape[-1] != 2:
                raise RealPcrRuntimeError(f"learned-w gate action shape invalid: {tuple(gate_action.shape)}")
            gate_diag = self._resolve_gate(
                gate_action[:, 0],
                cmd_f,
                cmd_a,
                gate_action[:, 1],
                diag,
            )

        cmd_np = gate_diag["cmd"][0].detach().cpu().numpy().astype(np.float32)
        safe_cmd, safety = self._apply_safety(
            cmd_np,
            target_valid,
            target_too_close=target_too_close,
            depth_invalid=depth_invalid,
        )
        result = {
            "cmd_policy": cmd_np,
            "cmd_safe": safe_cmd,
            "target_valid": bool(target_valid),
            "target_too_close": bool(target_too_close),
            "depth_invalid": bool(depth_invalid),
            "actor_difficulty": float(actor_difficulty),
            "safety": safety,
        }
        for key in (
            "gate_y_raw",
            "gate_y",
            "w",
            "signed_w",
            "signed_w_active",
            "y_eff",
            "w_support_correction",
            "risk_diff_correction",
            "cmd_f",
            "cmd_a",
            "clearance_F",
            "clearance_A",
            "risk_F",
            "risk_A",
            "risk_memory",
            "row_not_released",
            "cmd_cos",
            "conflict_score",
        ):
            val = gate_diag[key]
            arr = val[0].detach().cpu().numpy()
            result[key] = arr.astype(np.float32).tolist() if np.ndim(arr) > 0 else float(arr)
        return result

    def _apply_safety(
        self,
        cmd: np.ndarray,
        target_valid: bool,
        *,
        target_too_close: bool,
        depth_invalid: bool,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        cmd = np.asarray(cmd, dtype=np.float32).reshape(3)
        reasons = []
        if not np.isfinite(cmd).all():
            cmd = np.zeros(3, dtype=np.float32)
            reasons.append("nonfinite_cmd")
        if not target_valid and bool(self.args.stop_on_target_lost):
            cmd[:] = 0.0
            reasons.append("target_lost")
        if bool(depth_invalid) and bool(self.args.stop_on_depth_invalid):
            cmd[:] = 0.0
            reasons.append("depth_invalid")
        if bool(target_too_close) and bool(self.args.stop_forward_when_target_too_close):
            cmd[1] = min(float(cmd[1]), 0.0)
            reasons.append("target_too_close")

        limits = np.asarray([self.args.max_cmd_x, self.args.max_cmd_y, self.args.max_cmd_yaw], dtype=np.float32)
        cmd = np.clip(cmd, -limits, limits)

        now = time.time()
        dt = max(now - self.prev_cmd_stamp, 1e-3)
        delta_limits = np.asarray(
            [
                self.args.max_delta_x_per_s * dt,
                self.args.max_delta_y_per_s * dt,
                self.args.max_delta_yaw_per_s * dt,
            ],
            dtype=np.float32,
        )
        delta = np.clip(cmd - self.prev_cmd, -delta_limits, delta_limits)
        safe = self.prev_cmd + delta
        self.prev_cmd = safe.astype(np.float32)
        self.prev_cmd_stamp = now
        if reasons:
            safe[:] = 0.0
            self.prev_cmd[:] = 0.0
        return safe.astype(np.float32), {"reasons": reasons, "dry_run": not bool(self.args.publish_cmd)}

    def publish_or_print(self, result: Dict[str, Any]) -> None:
        cmd = np.asarray(result["cmd_safe"], dtype=np.float32).reshape(3)
        payload = {
            "cmd_policy": np.asarray(result["cmd_policy"], dtype=np.float32).round(4).tolist(),
            "cmd_safe": cmd.round(4).tolist(),
            "ros_twist": {
                "linear_x_forward": float(cmd[1]),
                "linear_y_left": float(-cmd[0]) if self.args.ros_linear_y_left_positive else float(cmd[0]),
                "angular_z": float(cmd[2]),
            },
            "y": float(result["gate_y"]),
            "w": float(result["w"]),
            "y_eff": float(result["y_eff"]),
            "risk_F": float(result["risk_F"]),
            "risk_A": float(result["risk_A"]),
            "risk_memory": float(result["risk_memory"]),
            "row_not_released": float(result["row_not_released"]),
            "target_valid": bool(result["target_valid"]),
            "target_too_close": bool(result.get("target_too_close", False)),
            "depth_invalid": bool(result.get("depth_invalid", False)),
            "safety": result["safety"],
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

        if self.rospy is not None and self.debug_pub is not None:
            msg = self._ros_msg_classes["String"]()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.debug_pub.publish(msg)
        if self.rospy is not None and self.cmd_pub is not None and self.args.publish_cmd:
            twist = self._ros_msg_classes["Twist"]()
            twist.linear.x = float(cmd[1])
            twist.linear.y = float(-cmd[0]) if self.args.ros_linear_y_left_positive else float(cmd[0])
            twist.angular.z = float(cmd[2])
            self.cmd_pub.publish(twist)

    def run_once(self) -> Dict[str, Any]:
        snap = self._get_snapshot()
        result = self.policy_step(snap)
        self.publish_or_print(result)
        return result

    def spin(self) -> None:
        if self.args.fake_input:
            for _ in range(max(1, int(self.args.fake_steps))):
                self.run_once()
                time.sleep(1.0 / max(float(self.args.rate_hz), 1e-6))
            return
        if self.rospy is None:
            raise RealPcrRuntimeError("ROS is not initialized")
        rate = self.rospy.Rate(float(self.args.rate_hz))
        while not self.rospy.is_shutdown():
            try:
                self.run_once()
            except RealPcrRuntimeError as exc:
                zero = {
                    "cmd_policy": np.zeros(3, dtype=np.float32),
                    "cmd_safe": np.zeros(3, dtype=np.float32),
                    "target_valid": False,
                    "target_too_close": False,
                    "depth_invalid": True,
                    "gate_y": 0.0,
                    "w": 0.0,
                    "y_eff": 0.0,
                    "risk_F": 0.0,
                    "risk_A": 0.0,
                    "risk_memory": 0.0,
                    "row_not_released": 0.0,
                    "safety": {"reasons": [str(exc)], "dry_run": not bool(self.args.publish_cmd)},
                }
                self.publish_or_print(zero)
            rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCR learned-w real robot ROS1 runner")
    parser.add_argument("--pcr_ckpt", required=True, type=str, help="learned-w gate checkpoint")
    parser.add_argument("--avoid_ckpt", default="agents/avoid_best.pt", type=str)
    parser.add_argument("--lowlevel_ckpt", default="agents/low_level_best.pt", type=str, help="path hint for deployment record")
    parser.add_argument("--w_mode", default="auto", choices=["auto", "learned", "learnedw2"])
    parser.add_argument("--w_blend_mode", default="multiply", choices=["multiply", "mix"])
    parser.add_argument("--signed_w_lambda", type=float, default=0.30)
    parser.add_argument("--signed_w_gamma_risk", type=float, default=0.15)
    parser.add_argument("--signed_w_margin", type=float, default=0.05)
    parser.add_argument("--w2_lambda", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--w2_risk_gamma", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--risk_memory", action="store_true", help="use deployable temporal risk memory in learned-w row slot")
    parser.add_argument("--risk_memory_l_clear", type=float, default=0.40)
    parser.add_argument("--risk_memory_velocity_source", type=str, default="body", choices=["body", "cmd"])
    parser.add_argument("--high_level_dt", type=float, default=0.10)
    parser.add_argument("--cmd_scale", default="1.0,1.0,1.0")
    parser.add_argument("--device", default="", help="cuda, cuda:0, or cpu; default auto")

    parser.add_argument("--fake_input", action="store_true", help="run without ROS using synthetic policy inputs")
    parser.add_argument("--fake_steps", type=int, default=3)
    parser.add_argument("--fake_x_right", type=float, default=0.0)
    parser.add_argument("--fake_y_forward", type=float, default=1.8)
    parser.add_argument("--fake_obstacle", action="store_true")
    parser.add_argument("--fake_obstacle_x", type=float, default=0.0)
    parser.add_argument("--fake_obstacle_y", type=float, default=1.0)

    parser.add_argument("--ros_node_name", default="pcr_realplay")
    parser.add_argument("--target_topic", default="/pcr/target_state", help="Float32MultiArray: [x_right,y_forward,v_right,v_forward,valid,target_too_close,depth_invalid]; last two fields optional")
    parser.add_argument("--local_map_topic", default="/pcr/local_map_2ch", help="Float32MultiArray: flattened (2,16,16)")
    parser.add_argument("--state_topic", default="", help="optional Float32MultiArray robot state; zeros if omitted")
    parser.add_argument("--row_not_released_topic", default="", help="optional Float32 diagnostic only; learned-w actor always receives zero row-release feature")
    parser.add_argument("--cmd_topic", default="/cmd_vel")
    parser.add_argument("--debug_topic", default="/pcr_realplay/debug")
    parser.add_argument("--publish_cmd", action="store_true", help="actually publish Twist; default is dry-run")
    parser.add_argument("--ros_linear_y_left_positive", action="store_true", help="publish Twist.linear.y = -cmd_x_right")

    parser.add_argument("--map_channels", type=int, default=2)
    parser.add_argument("--map_size", type=int, default=16)
    parser.add_argument("--map_extent_m", type=float, default=3.0)
    parser.add_argument("--difficulty_radius_m", type=float, default=2.0)
    parser.add_argument("--aff_stack", type=int, default=1, help="fallback only; checkpoint usually overrides")
    parser.add_argument("--state_dim", type=int, default=9, help="fallback only; checkpoint usually overrides")
    parser.add_argument("--row_not_released_default", type=float, default=0.0, help="diagnostic only; not fed to learned-w actor")
    parser.add_argument("--cmd_safe_dist", type=float, default=0.25)
    parser.add_argument("--cmd_free_dist", type=float, default=0.60)

    parser.add_argument("--rate_hz", type=float, default=10.0)
    parser.add_argument("--input_timeout_s", type=float, default=0.5)
    parser.add_argument("--state_timeout_s", type=float, default=1.0)
    parser.add_argument("--row_timeout_s", type=float, default=1.0)
    parser.add_argument("--allow_missing_state", action="store_true", default=True)
    parser.add_argument("--stop_on_target_lost", action="store_true", default=True)
    parser.add_argument("--stop_on_depth_invalid", action="store_true", default=True)
    parser.add_argument("--stop_forward_when_target_too_close", action="store_true", default=True)
    parser.add_argument("--max_cmd_x", type=float, default=0.20)
    parser.add_argument("--max_cmd_y", type=float, default=0.25)
    parser.add_argument("--max_cmd_yaw", type=float, default=0.60)
    parser.add_argument("--max_delta_x_per_s", type=float, default=0.30)
    parser.add_argument("--max_delta_y_per_s", type=float, default=0.35)
    parser.add_argument("--max_delta_yaw_per_s", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = PcrRealplay(args)
    runner.spin()


if __name__ == "__main__":
    main()
