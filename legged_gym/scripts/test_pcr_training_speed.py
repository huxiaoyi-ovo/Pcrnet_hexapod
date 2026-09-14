#!/usr/bin/env python3
"""CPU regression for legacy and reviewer-proof Strong Mono s_pcr_new curricula."""

import argparse
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)

    def to(self, device=None, dtype=None):
        return self

    def numel(self):
        return len(self.values)

    def item(self):
        if len(self.values) != 1:
            raise ValueError("item requires exactly one value")
        return self.values[0]

    def copy_(self, source):
        self.values = list(source.values)
        return self


class FakeTorch:
    Tensor = Tensor
    long = "long"
    float32 = "float32"
    bool = "bool"

    @staticmethod
    def as_tensor(values, device=None, dtype=None):
        if dtype == FakeTorch.long:
            return Tensor(int(v) for v in values)
        return Tensor(float(v) for v in values)

    @staticmethod
    def ones_like(tensor, dtype=None):
        return Tensor([1] * tensor.numel())

    @staticmethod
    def mode(tensor):
        values, counts = np.unique(np.asarray(tensor.values), return_counts=True)
        return SimpleNamespace(values=Tensor([values[int(np.argmax(counts))]]))


METHODS = {
    "_s_avoid_collision_rate",
    "_pcr_new_curriculum_progress",
    "_pcr_new_curriculum_weights",
    "_pcr_new_strong_mono_open_stage",
    "_pcr_new_strong_mono_level_rates",
    "_pcr_new_strong_mono_frontier_competent",
    "_advance_pcr_new_strong_mono_transitions",
    "_pcr_new_strong_mono_sampling_weights",
    "_sample_pcr_new_curriculum",
    "_update_s_avoid_curriculum",
    "export_pcr_new_curriculum_state",
    "import_pcr_new_curriculum_state",
    "_update_pcr_new_curriculum_extras",
}


def load_methods(source: Path):
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    ground = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HexGround")
    selected = [node for node in ground.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    missing = METHODS - {node.name for node in selected}
    if missing:
        raise AssertionError("source is missing methods: {}".format(sorted(missing)))
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "torch": FakeTorch, "Tuple": Tuple, "Optional": Optional, "deque": deque}
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in METHODS}


def make_hist(window=2048):
    return {
        "collision": deque(maxlen=window),
        "exposure": deque(maxlen=window),
        "progress": deque(maxlen=window),
        "success": deque(maxlen=window),
        "row_success": deque(maxlen=window),
    }


class CurriculumHarness:
    def __init__(self, methods, progress=0.0, strong=False, num_envs=8):
        for name, method in methods.items():
            setattr(type(self), name, method)
        self.num_envs = int(num_envs)
        self.device = "cpu"
        self.s_avoid_enabled = True
        self.s_avoid_clutter_enabled = False
        self.s_avoid_stage = 1
        self.s_avoid_total_completed_episodes = 0
        self.pcr_new_curriculum_enabled = True
        self.pcr_new_strong_mono_curriculum_enabled = bool(strong)
        self.pcr_new_strong_mono_transitions = 0
        self.pcr_new_strong_mono_mastery_stage = 0
        self.pcr_new_strong_mono_openings = (6144000, 12288000, 18432000)
        self.pcr_new_strong_mono_stage_weights = (
            (1.00, 0.00, 0.00, 0.00),
            (0.40, 0.60, 0.00, 0.00),
            (0.20, 0.30, 0.50, 0.00),
            (0.10, 0.20, 0.30, 0.40),
        )
        self.pcr_new_strong_mono_window = 2048
        self.pcr_new_strong_mono_success_threshold = 0.50
        self.pcr_new_strong_mono_row_success_threshold = 0.70
        self.pcr_new_strong_mono_collision_threshold = 0.30
        self.pcr_new_strong_mono_probe_ratio = 0.20
        self.pcr_new_strong_mono_level_hists = {level: make_hist() for level in range(4)}
        self.s_avoid_env_episode_count = Tensor([0] * self.num_envs)
        self.pcr_new_curriculum_level = Tensor([0] * self.num_envs)
        self.pcr_new_target_speed = Tensor([0.35] * self.num_envs)
        self.s_avoid_stage_per_env = Tensor([1] * self.num_envs)
        self.cfg = SimpleNamespace(terrain=SimpleNamespace(avoid_seed=7001, pcr_new_force_stage=None))
        self.nav_cfg = SimpleNamespace(
            pcr_new_curriculum_progress_override=progress,
            pcr_new_curriculum_total_episodes=120000,
            pcr_new_generalize_enable=False,
            pcr_new_force_target_speed=None,
        )
        self.extras = {}
        # State/sampling smoke is CPU-only and intentionally does not execute
        # Isaac-backed tensor logging; the production method is syntax-checked.
        self._update_pcr_new_curriculum_extras = lambda: None


def expected_weights(progress):
    if progress < 0.25:
        return (0.70, 0.30, 0.00, 0.00)
    if progress < 0.50:
        return (0.30, 0.40, 0.30, 0.00)
    if progress < 0.75:
        return (0.15, 0.30, 0.35, 0.20)
    return (0.10, 0.20, 0.30, 0.40)


def expected_level(env_id, episode_idx, progress):
    rng = np.random.RandomState(7001 + env_id * 10007 + episode_idx * 131 + 7919)
    return int(rng.choice(np.arange(4, dtype=np.int64), p=expected_weights(progress)))


def assert_legacy_sampler_unchanged(methods):
    level3_seen = False
    for progress in (0.0, 0.30, 0.60, 0.90):
        harness = CurriculumHarness(methods, progress=progress, strong=False)
        for env_id in range(16):
            for episode_idx in range(16):
                stage, speed, level = harness._sample_pcr_new_curriculum(env_id, episode_idx)
                if level != expected_level(env_id, episode_idx, progress):
                    raise AssertionError("legacy curriculum seed/weights changed")
                if level in (0, 1) and stage != 1:
                    raise AssertionError("legacy L0/L1 stage changed")
                if level == 2 and stage != 2:
                    raise AssertionError("legacy L2 stage changed")
                if level == 3:
                    level3_seen = True
                    if stage not in (3, 4):
                        raise AssertionError("legacy L3 stage changed")
                if speed > 0.50 + 1e-12:
                    raise AssertionError("legacy training speed exceeds 0.50 m/s")
    if not level3_seen:
        raise AssertionError("legacy test grid did not exercise L3")


def sample_levels(harness, count=20000):
    return np.asarray([harness._sample_pcr_new_curriculum(i % 64, i // 64)[2] for i in range(count)])


def append_level_outcomes(harness, level, success, row_success, collision, count=2048, decision=False):
    harness._update_s_avoid_curriculum(
        Tensor([bool(collision)] * count),
        Tensor([1] * count),
        Tensor([int(level)] * count),
        Tensor([bool(decision)] * count),
        Tensor([1.0] * count),
        Tensor([float(success)] * count),
        Tensor([float(row_success)] * count),
        Tensor([True] * count),
    )


def assert_strong_mono_contract(methods):
    harness = CurriculumHarness(methods, strong=True)
    if np.any(sample_levels(harness) != 0):
        raise AssertionError("Strong Mono C0 must sample only L0")
    harness._advance_pcr_new_strong_mono_transitions(6144000 - 1)
    if harness._pcr_new_strong_mono_open_stage() != 0:
        raise AssertionError("transition opening occurred early")
    harness._advance_pcr_new_strong_mono_transitions(1)
    if harness._pcr_new_strong_mono_open_stage() != 1 or harness.pcr_new_strong_mono_mastery_stage != 0:
        raise AssertionError("first opening/mastery boundary is incorrect")
    levels = sample_levels(harness)
    probe_ratio = float(np.mean(levels == 1))
    if not (0.18 <= probe_ratio <= 0.22) or np.any((levels != 0) & (levels != 1)):
        raise AssertionError("next-level probe is not deterministic 20% sampling: {}".format(probe_ratio))

    append_level_outcomes(harness, level=0, success=0.0, row_success=0.0, collision=1.0)
    if len(harness.pcr_new_strong_mono_level_hists[0]["success"]) != 2048:
        raise AssertionError("non-clutter Strong Mono episodes were filtered by decision_episode")
    harness._advance_pcr_new_strong_mono_transitions(6144000)
    if harness._pcr_new_strong_mono_open_stage() != 2 or harness.pcr_new_strong_mono_mastery_stage != 0:
        raise AssertionError("failure episodes changed availability or mastery")

    clean = CurriculumHarness(methods, strong=True)
    clean._advance_pcr_new_strong_mono_transitions(6144000)
    append_level_outcomes(clean, level=1, success=1.0, row_success=1.0, collision=0.0)
    clean._advance_pcr_new_strong_mono_transitions(0)
    if clean.pcr_new_strong_mono_mastery_stage != 0:
        raise AssertionError("L1 metrics advanced the L0 frontier")
    append_level_outcomes(clean, level=0, success=1.0, row_success=1.0, collision=0.0)
    clean._advance_pcr_new_strong_mono_transitions(0)
    if clean.pcr_new_strong_mono_mastery_stage != 1:
        raise AssertionError("qualified L0 frontier did not advance exactly one stage")

    state = clean.export_pcr_new_curriculum_state()
    restored = CurriculumHarness(methods, strong=True)
    if not restored.import_pcr_new_curriculum_state(state):
        raise AssertionError("Strong Mono curriculum state did not import")
    if restored.export_pcr_new_curriculum_state() != state:
        raise AssertionError("Strong Mono curriculum state roundtrip changed state")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "envs/hex_v4/hex_ground.py",
    )
    args = parser.parse_args()
    methods = load_methods(args.source)
    assert_legacy_sampler_unchanged(methods)
    assert_strong_mono_contract(methods)
    print("PASS: legacy and Strong Mono s_pcr_new curriculum ({})".format(args.source))


if __name__ == "__main__":
    main()
