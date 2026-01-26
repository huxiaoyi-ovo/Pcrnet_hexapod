from __future__ import annotations

from typing import Dict

import numpy as np

from ..scene_spec import SceneSpec


def generate_s6_structured_ood(
    rng: np.random.RandomState,
    difficulty: float,
    seed: int,
    env_dims: Dict[str, float],
    robot_envelope: Dict[str, float],
    params: Dict[str, float],
    h_scale: float,
    v_scale: float,
) -> SceneSpec:
    raise RuntimeError("terrain_v2: s6_structured_ood not implemented yet (Stage B)")
