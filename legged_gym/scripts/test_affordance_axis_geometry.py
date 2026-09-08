#!/usr/bin/env python3
"""CPU geometry regression for the affordance-map [x_right, y_forward] contract.

Run against the working tree, or pass an older source file with --source.  Methods
are extracted from the source AST so Isaac Gym is neither imported nor initialized.
"""

import argparse
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


METHODS = {
    "_quat_to_yaw",
    "_get_affordance_reference_pose",
    "_world_to_local_xy",
    "_compute_gt_affordance_from_heightfield",
    "_compute_gt_affordance_from_scene",
    "_build_affordance_dist_map",
    "_build_affordance_geometry",
    "_compute_clearance_from_affordance",
    "_compute_clearance_along_cmd",
}


class TorchMeshgridCompat:
    """Preserve explicit xy/ij source semantics on the local torch 1.8 CPU fixture."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def meshgrid(self, *tensors, **kwargs):
        indexing = kwargs.pop("indexing", "ij")
        if kwargs or len(tensors) != 2:
            return torch.meshgrid(*tensors, **kwargs)
        grid_x, grid_y = torch.meshgrid(*tensors)
        if indexing == "ij":
            return grid_x, grid_y
        if indexing == "xy":
            return grid_x.transpose(0, 1), grid_y.transpose(0, 1)
        raise ValueError("unsupported meshgrid indexing: {}".format(indexing))


def supports_meshgrid_indexing():
    try:
        torch.meshgrid(torch.zeros(1), torch.zeros(1), indexing="ij")
    except TypeError:
        return False
    return True


def load_methods(source: Path) -> Dict[str, object]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HierarchicalHexapodEnv"
    )
    selected = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    missing = METHODS - {node.name for node in selected}
    if missing:
        raise AssertionError("source is missing methods: {}".format(sorted(missing)))
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch if supports_meshgrid_indexing() else TorchMeshgridCompat(),
        "F": F,
        "math": math,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return {name: namespace[name] for name in METHODS}


class Harness:
    def __init__(self, methods, env, yaw):
        for name, method in methods.items():
            setattr(type(self), name, method)
        self.env = env
        self.num_envs = env.root_states.shape[0]
        self.device = torch.device("cpu")
        self.affordance_map_size = 32
        self.affordance_map_extent = 3.0
        self.affordance_cell_size = self.affordance_map_extent / self.affordance_map_size
        self.affordance_origin_mode = "base_center"
        self.affordance_origin_local_xy = torch.zeros(2, device=self.device)
        self.affordance_clearance = 0.0
        self.affordance_blocking_height = 0.2
        self.affordance_crossable_height = 0.2
        self.env.root_states[:, 5] = torch.sin(yaw * 0.5)
        self.env.root_states[:, 6] = torch.cos(yaw * 0.5)
        self.camera_fov_rad = None
        self.camera_bearing_rad = 0.0
        self.camera_vertical_fov_rad = None
        self.camera_tilt_down_rad = 0.0
        self.camera_near = None
        self.camera_far = None
        self.affordance_dist_map = self._build_affordance_dist_map()
        (
            self.affordance_x_map,
            self.affordance_y_map,
            self.affordance_bearing_map,
            self.affordance_visible_mask,
        ) = self._build_affordance_geometry()

def make_env(num_envs, scene_specs=None, height_samples=None):
    terrain = SimpleNamespace(border_size=2.0, horizontal_scale=0.05, vertical_scale=1.0)
    env = SimpleNamespace(
        root_states=torch.zeros(num_envs, 7),
        env_origins=torch.zeros(num_envs, 3),
        cfg=SimpleNamespace(terrain=terrain),
    )
    if scene_specs is not None:
        env.scene_spec_cache = scene_specs
    if height_samples is not None:
        env.height_samples = height_samples
    return env


def local_to_world(local_xy, yaw):
    x_local, y_local = local_xy
    return (
        math.cos(yaw) * x_local - math.sin(yaw) * y_local,
        math.sin(yaw) * x_local + math.cos(yaw) * y_local,
    )


def expected_distance(harness, ix, iy):
    return math.hypot(
        float(harness.affordance_x_map[ix, iy].item()),
        float(harness.affordance_y_map[ix, iy].item()),
    )


def assert_close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("{}: got {:.6f}, expected {:.6f}".format(label, float(actual), float(expected)))


def test_counterexample_and_scene_queries(methods):
    front = (16, 5)
    right = (21, 0)
    local_cells = [(front, 0.0), (right, 0.0), (front, math.pi / 2.0), (right, math.pi / 2.0)]
    specs = []
    for (ix, iy), yaw in local_cells:
        x = -1.5 + (ix + 0.5) * (3.0 / 32.0)
        y = (iy + 0.5) * (3.0 / 32.0)
        world_x, world_y = local_to_world((x, y), yaw)
        specs.append(SimpleNamespace(static_obstacles=[SimpleNamespace(position=(world_x, world_y), size=(0.01, 0.01))]))
    yaw = torch.tensor([item[1] for item in local_cells])
    harness = Harness(methods, make_env(4, scene_specs=specs), yaw)

    # The historical mismatch reads 1.833526 m here; the physical cell is 0.517751 m.
    assert_close(harness.affordance_dist_map[16, 5], expected_distance(harness, 16, 5), "known counterexample")
    assert_close(harness.affordance_dist_map[16, 5], 0.517751, "known counterexample value")

    scene = harness._compute_gt_affordance_from_scene()
    commands = []
    opposite_commands = []
    expected = []
    for env_id, ((ix, iy), _) in enumerate(local_cells):
        if (ix, iy) == front:
            commands.append([0.0, 1.0])
            opposite_commands.append([1.0, 0.0])
        else:
            # The obstacle stays in-cell at y > 0; the command itself remains pure rightward.
            commands.append([1.0, 0.0])
            opposite_commands.append([0.0, 1.0])
        if scene[env_id, 0, ix, iy].item() != 1.0:
            raise AssertionError("scene raster missed env {} cell [{}, {}]".format(env_id, ix, iy))
        expected.append(expected_distance(harness, ix, iy))
    queried = harness._compute_clearance_along_cmd(scene, torch.tensor(commands))
    for env_id, value in enumerate(expected):
        assert_close(queried[env_id], value, "scene query env {}".format(env_id))
    opposite = harness._compute_clearance_along_cmd(scene, torch.tensor(opposite_commands))
    if not torch.allclose(opposite, torch.full_like(opposite, 3.0)):
        raise AssertionError("front/right cross-queries must not enter the 25 degree cone")


def test_heightfield_scene_alignment(methods):
    ix, iy = 16, 5
    x = -1.5 + (ix + 0.5) * (3.0 / 32.0)
    y = (iy + 0.5) * (3.0 / 32.0)
    height_samples = torch.zeros(200, 200)
    height_samples[int((y + 2.0) / 0.05), int((x + 2.0) / 0.05)] = 1.0
    spec = SimpleNamespace(static_obstacles=[SimpleNamespace(position=(x, y), size=(0.01, 0.01))])
    harness = Harness(methods, make_env(1, scene_specs=[spec], height_samples=height_samples), torch.zeros(1))
    heightfield = harness._compute_gt_affordance_from_heightfield()
    scene = harness._compute_gt_affordance_from_scene()
    if heightfield[0, 0, ix, iy].item() != 1.0 or scene[0, 0, ix, iy].item() != 1.0:
        raise AssertionError("heightfield and scene must share [x_right, y_forward] cell [{}, {}]".format(ix, iy))
    assert_close(harness.affordance_dist_map[ix, iy], expected_distance(harness, ix, iy), "heightfield/grid distance")


def test_camera_mount_scene_reference(methods):
    # Server Git 49235d2 depth-camera position: [0.00, 0.22, 0.08].
    camera_offset = (0.0, 0.22)
    ix, iy = 16, 5
    cell_local = (-1.5 + (ix + 0.5) * (3.0 / 32.0), (iy + 0.5) * (3.0 / 32.0))
    yaw = torch.tensor([0.0, math.pi / 2.0])
    specs = []
    for heading in yaw.tolist():
        cam_x, cam_y = local_to_world(camera_offset, heading)
        cell_x, cell_y = local_to_world(cell_local, heading)
        specs.append(
            SimpleNamespace(
                static_obstacles=[SimpleNamespace(position=(cam_x + cell_x, cam_y + cell_y), size=(0.01, 0.01))]
            )
        )
    harness = Harness(methods, make_env(2, scene_specs=specs), yaw)
    harness.affordance_origin_mode = "camera_mount"
    harness.affordance_origin_local_xy = torch.tensor(camera_offset)
    ref_xy, ref_yaw = harness._get_affordance_reference_pose()
    for env_id, heading in enumerate(yaw.tolist()):
        expected_ref = local_to_world(camera_offset, heading)
        assert_close(ref_xy[env_id, 0], expected_ref[0], "camera mount x env {}".format(env_id))
        assert_close(ref_xy[env_id, 1], expected_ref[1], "camera mount y env {}".format(env_id))
        assert_close(ref_yaw[env_id], heading, "camera mount yaw env {}".format(env_id))
    scene = harness._compute_gt_affordance_from_scene()
    if not torch.all(scene[:, 0, ix, iy] == 1.0):
        raise AssertionError("camera-mount scene raster missed the expected local cell")
    queried = harness._compute_clearance_along_cmd(scene, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    expected = torch.full_like(queried, expected_distance(harness, ix, iy))
    if not torch.allclose(queried, expected):
        raise AssertionError("camera-mount command clearance disagrees with the local cell distance")


def test_empty_and_zero_command_fallback(methods):
    harness = Harness(methods, make_env(2, scene_specs=[SimpleNamespace(static_obstacles=[])] * 2), torch.zeros(2))
    empty = torch.zeros(2, 3, 32, 32)
    active = harness._compute_clearance_along_cmd(empty, torch.tensor([[0.0, 1.0], [1.0, 0.1]]))
    zero = harness._compute_clearance_along_cmd(empty, torch.zeros(2, 2))
    if not torch.allclose(active, torch.full_like(active, 3.0)) or not torch.allclose(zero, active):
        raise AssertionError("empty-map and zero-command fallbacks must return map extent")

    occupied = empty.clone()
    occupied[:, 0, 16, 5] = 1.0
    zero_with_obstacle = harness._compute_clearance_along_cmd(occupied, torch.zeros(2, 2))
    global_clear = harness._compute_clearance_from_affordance(occupied)
    if not torch.allclose(zero_with_obstacle, global_clear):
        raise AssertionError("zero command must use global clearance")
    mixed = harness._compute_clearance_along_cmd(occupied, torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
    if not torch.allclose(mixed[0:1], global_clear[0:1]) or not torch.allclose(mixed[1:2], global_clear[1:2]):
        raise AssertionError("mixed zero/active batch must preserve each command fallback")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("train_highlevel.py"))
    args = parser.parse_args()
    methods = load_methods(args.source)
    test_counterexample_and_scene_queries(methods)
    test_heightfield_scene_alignment(methods)
    test_camera_mount_scene_reference(methods)
    test_empty_and_zero_command_fallback(methods)
    meshgrid_backend = "native" if supports_meshgrid_indexing() else "compat"
    print("PASS: affordance axis geometry ({}, meshgrid={})".format(args.source, meshgrid_backend))


if __name__ == "__main__":
    main()
