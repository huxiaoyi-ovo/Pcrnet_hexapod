#!/usr/bin/env python3
"""CPU regression for the s_pcr_new training curriculum target-speed ceiling."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np


METHODS = {
    "_pcr_new_curriculum_progress",
    "_pcr_new_curriculum_weights",
    "_sample_pcr_new_curriculum",
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
    namespace = {"np": np, "Tuple": Tuple}
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in METHODS}


class CurriculumHarness:
    def __init__(self, methods, progress):
        for name, method in methods.items():
            setattr(type(self), name, method)
        self.pcr_new_curriculum_enabled = True
        self.s_avoid_total_completed_episodes = 0
        self.cfg = SimpleNamespace(terrain=SimpleNamespace(avoid_seed=7001, pcr_new_force_stage=None))
        self.nav_cfg = SimpleNamespace(
            pcr_new_curriculum_progress_override=progress,
            pcr_new_curriculum_total_episodes=120000,
            pcr_new_generalize_enable=False,
            pcr_new_force_target_speed=None,
        )


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


def assert_training_sampler(methods):
    level3_seen = False
    for progress in (0.0, 0.30, 0.60, 0.90):
        harness = CurriculumHarness(methods, progress)
        for env_id in range(16):
            for episode_idx in range(16):
                stage, speed, level = harness._sample_pcr_new_curriculum(env_id, episode_idx)
                expected = expected_level(env_id, episode_idx, progress)
                if level != expected:
                    raise AssertionError("curriculum seed/weights changed for env {}, episode {}".format(env_id, episode_idx))
                if level in (0, 1) and stage != 1:
                    raise AssertionError("level {} must retain stage 1".format(level))
                if level == 2 and stage != 2:
                    raise AssertionError("level 2 must retain stage 2")
                if level == 3:
                    level3_seen = True
                    if stage not in (3, 4):
                        raise AssertionError("level 3 must retain stage 3/4 sampling")
                if speed > 0.50 + 1e-12:
                    raise AssertionError("training target speed exceeds 0.50 m/s: {}".format(speed))
    if not level3_seen:
        raise AssertionError("test grid did not exercise level 3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "envs/hex_v4/hex_ground.py",
    )
    args = parser.parse_args()
    assert_training_sampler(load_methods(args.source))
    print("PASS: s_pcr_new training target speed ({})".format(args.source))


if __name__ == "__main__":
    main()
