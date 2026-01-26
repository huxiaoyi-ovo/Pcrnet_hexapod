from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from .scene_spec import SceneSpec, compute_layout_hash
from .generators.debug_axis_calib import generate_debug_axis_calib
from .generators.s1_corridor_gate import generate_s1_corridor_gate
from .generators.s2_forest import generate_s2_forest
from .generators.s3_doorway_rooms import generate_s3_doorway_rooms
from .generators.s4_crossing import generate_s4_crossing
from .generators.s5_sparse_dense import generate_s5_sparse_dense
from .generators.s6_structured_ood import generate_s6_structured_ood


def _lerp(a: float, b: float, d: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(d, 0.0, 1.0))


def _resolve_scene_params(params: Optional[Dict[str, Any]], scene_id: str) -> Dict[str, Any]:
    if not params:
        return {}
    if scene_id in params and isinstance(params[scene_id], dict):
        return dict(params[scene_id])
    return dict(params)


def _mix_value(a: Any, b: Any, d: float) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _lerp(float(a), float(b), d)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return type(a)(_lerp(float(x), float(y), d) for x, y in zip(a, b))
    return a if d <= 0.5 else b


def _mix_params(easy: Dict[str, Any], hard: Dict[str, Any], d: float) -> Dict[str, Any]:
    keys = set(easy.keys()) | set(hard.keys())
    mixed: Dict[str, Any] = {}
    for key in keys:
        if key in easy and key in hard:
            mixed[key] = _mix_value(easy[key], hard[key], d)
        else:
            mixed[key] = easy.get(key, hard.get(key))
    return mixed


class SceneGenerator:
    _logged = False

    def __init__(self, cfg, env_dims: Optional[Dict[str, float]] = None, robot_envelope: Optional[Dict[str, float]] = None) -> None:
        self.cfg = cfg
        self.env_dims = env_dims
        self.robot_envelope = robot_envelope
        self.scene_type = getattr(cfg, "scene_type", None)
        self.scene_types = list(getattr(cfg, "scene_types", []) or [])
        if not self.scene_types and self.scene_type:
            self.scene_types = [self.scene_type]
        self.scene_probs_easy = getattr(cfg, "scene_probs_easy", None)
        self.scene_probs_hard = getattr(cfg, "scene_probs_hard", None)
        self.scene_probs = getattr(cfg, "scene_probs", None)
        self.scene_seed = int(getattr(cfg, "scene_seed", 13))
        self.scene_clearance = float(getattr(cfg, "scene_clearance", 0.27))
        self.h_scale = float(getattr(cfg, "horizontal_scale", 0.1))
        self.v_scale = float(getattr(cfg, "vertical_scale", 0.01))
        if not SceneGenerator._logged:
            tag = self.scene_type or ",".join(self.scene_types) or "unknown"
            print(f"[Init] 正在使用 terrain_v2 ({tag})")
            SceneGenerator._logged = True

    def set_env(self, env_dims: Dict[str, float], robot_envelope: Optional[Dict[str, float]] = None) -> None:
        self.env_dims = env_dims
        if robot_envelope is not None:
            self.robot_envelope = robot_envelope

    @property
    def has_dynamic(self) -> bool:
        # Stage A: dynamic obstacles are not enabled yet.
        return False

    def seed_for_env(self, env_id: int, episode_idx: int) -> int:
        return int(self.scene_seed + env_id * 10007 + episode_idx * 131)

    def seed_for_cell(self, row: int, col: int) -> int:
        return int(self.scene_seed + row * 1000 + col * 17)

    def _select_scene_type(self, rng: np.random.RandomState, difficulty: float) -> str:
        if self.scene_type:
            return self.scene_type
        if not self.scene_types:
            raise RuntimeError("scene_types is empty")
        probs = None
        if self.scene_probs_easy is not None and self.scene_probs_hard is not None:
            if isinstance(self.scene_probs_easy, dict) and isinstance(self.scene_probs_hard, dict):
                probs_easy = [float(self.scene_probs_easy.get(t, 0.0)) for t in self.scene_types]
                probs_hard = [float(self.scene_probs_hard.get(t, 0.0)) for t in self.scene_types]
            else:
                probs_easy = list(self.scene_probs_easy)
                probs_hard = list(self.scene_probs_hard)
            if len(probs_easy) == len(self.scene_types) and len(probs_hard) == len(self.scene_types):
                probs = [_lerp(pe, ph, difficulty) for pe, ph in zip(probs_easy, probs_hard)]
        elif self.scene_probs is not None:
            if isinstance(self.scene_probs, dict):
                probs = [float(self.scene_probs.get(t, 0.0)) for t in self.scene_types]
            else:
                probs = list(self.scene_probs)
        if probs is None or len(probs) != len(self.scene_types):
            probs = [1.0 / len(self.scene_types)] * len(self.scene_types)
        probs = np.asarray(probs, dtype=np.float64)
        total = float(np.sum(probs))
        if total <= 0.0 or not np.isfinite(total):
            probs = np.asarray([1.0 / len(self.scene_types)] * len(self.scene_types), dtype=np.float64)
        else:
            probs = probs / total
        idx = int(rng.choice(len(self.scene_types), p=probs))
        return self.scene_types[idx]

    def _resolve_params(self, scene_id: str, difficulty: float) -> Dict[str, Any]:
        easy = _resolve_scene_params(getattr(self.cfg, "scene_params_easy", {}), scene_id)
        hard = _resolve_scene_params(getattr(self.cfg, "scene_params_hard", {}), scene_id)
        return _mix_params(easy, hard, difficulty)

    def sample(
        self,
        scene_id: str,
        difficulty: float,
        seed: int,
        env_dims: Optional[Dict[str, float]] = None,
        robot_envelope: Optional[Dict[str, float]] = None,
    ) -> SceneSpec:
        rng = np.random.RandomState(int(seed))
        if env_dims is None:
            env_dims = self.env_dims
        if env_dims is None:
            raise RuntimeError("terrain_v2 missing env_dims")
        if robot_envelope is None:
            robot_envelope = self.robot_envelope
        if robot_envelope is None:
            robot_envelope = {"clearance": self.scene_clearance}
        params = self._resolve_params(scene_id, difficulty)

        if scene_id == "debug_axis_calib":
            scene = generate_debug_axis_calib(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s1_corridor_gate":
            scene = generate_s1_corridor_gate(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s2_forest":
            scene = generate_s2_forest(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s3_doorway_rooms":
            scene = generate_s3_doorway_rooms(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s4_crossing":
            scene = generate_s4_crossing(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s5_sparse_dense":
            scene = generate_s5_sparse_dense(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        elif scene_id == "s6_structured_ood":
            scene = generate_s6_structured_ood(
                rng,
                difficulty,
                seed,
                env_dims,
                robot_envelope,
                params,
                self.h_scale,
                self.v_scale,
            )
        else:
            raise RuntimeError(f"terrain_v2 unsupported scene_type={scene_id}")

        scene_id_tag = f"{scene.scene_type}-{scene.seed}"
        scene = replace(scene, scene_id=scene_id_tag)
        scene = replace(scene, layout_hash=compute_layout_hash(scene))
        return scene
