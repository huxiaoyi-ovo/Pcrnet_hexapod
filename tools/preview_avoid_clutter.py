#!/usr/bin/env python3
"""Render three static, meter-scale Avoid clutter layout previews in Isaac Gym."""

import json
import math
from pathlib import Path

from isaacgym import gymapi


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "avoid_clutter_design_preview" / "spawn_fixed"
CYLINDER_URDF = OUT.parent / "cylinder.urdf"
DOMAIN = {"x": [-1.6, 1.6], "y": [0.0, 10.0]}
CYLINDER_RADIUS = 0.15
CYLINDER_HEIGHT = 0.50
W_EFF = 0.54  # 2 * current nominal 0.27 m collision-envelope radius.
ROBOT_Y = 0.45
PREVIEW_FORWARD_SHIFT = 1.6
CAMERA_MOUNT_Y = 0.22
DEPTH_HORIZONTAL_FOV_DEG = 87.0
DEPTH_NEAR_M = 0.28
DEPTH_FAR_M = 3.0


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    scale = dx * dx + dy * dy
    if scale == 0.0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / scale))
    return math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy))


def closest_path_point(polyline, y_value):
    for start, end in zip(polyline, polyline[1:]):
        if min(start[1], end[1]) <= y_value <= max(start[1], end[1]) and end[1] != start[1]:
            t = (y_value - start[1]) / (end[1] - start[1])
            return (start[0] + t * (end[0] - start[0]), y_value), (end[0] - start[0], end[1] - start[1])
    raise ValueError("anchor y is outside the forward-only polyline")


def paired_obstacles(polyline, anchor_ys, offset):
    obstacles = []
    for y_value in anchor_ys:
        point, tangent = closest_path_point(polyline, y_value)
        length = math.hypot(*tangent)
        normal = (-tangent[1] / length, tangent[0] / length)
        obstacles.extend([
            (point[0] + offset * normal[0], point[1] + offset * normal[1]),
            (point[0] - offset * normal[0], point[1] - offset * normal[1]),
        ])
    return obstacles


def make_layouts():
    definitions = [
        ("easy", 0.25, 0.20, 1,
         [(0.0, 0.45), (0.0, 1.45), (0.42, 2.45), (0.42, 7.55)],
         [0.95, 2.95, 5.45], 0.79),
        ("medium", 0.15, 0.35, 3,
         [(0.0, 0.45), (0.0, 1.20), (-0.48, 2.00), (-0.48, 2.80),
          (0.48, 3.70), (0.48, 4.50), (-0.42, 5.40), (-0.42, 7.55)],
         [0.82, 2.45, 4.15, 6.35], 0.68),
        ("hard", 0.075, 0.50, 5,
         [(0.0, 0.45), (0.0, 1.05), (-0.50, 1.75), (-0.50, 2.40),
          (0.50, 3.10), (0.50, 3.75), (-0.50, 4.45), (-0.50, 5.10),
          (0.50, 5.80), (0.50, 6.45), (-0.45, 7.15), (-0.45, 7.55)],
         [0.78, 2.08, 3.43, 4.78, 6.12], 0.59),
    ]
    layouts = []
    for name, margin, speed, decisions, polyline, anchors, offset in definitions:
        shifted_polyline = [(x, y + PREVIEW_FORWARD_SHIFT) for x, y in polyline]
        shifted_polyline.insert(0, (0.0, ROBOT_Y))
        layouts.append({
            "name": name,
            "margin_m": margin,
            "suggested_forward_speed_mps": speed,
            "decision_count": decisions,
            "polyline": shifted_polyline,
            "obstacles": paired_obstacles(shifted_polyline, [y + PREVIEW_FORWARD_SHIFT for y in anchors], offset),
        })
    return layouts


def check_layout(layout):
    polyline = layout["polyline"]
    margin = layout["margin_m"]
    required = CYLINDER_RADIUS + W_EFF / 2.0 + margin
    if any(end[1] <= start[1] for start, end in zip(polyline, polyline[1:])):
        raise ValueError(f"{layout['name']}: polyline is not forward-only")
    min_clearance = min(
        point_segment_distance(obstacle, start, end)
        for obstacle in layout["obstacles"]
        for start, end in zip(polyline, polyline[1:])
    )
    if min_clearance + 1e-9 < required:
        raise ValueError(f"{layout['name']}: clearance {min_clearance:.3f} < {required:.3f}")
    for i, first in enumerate(layout["obstacles"]):
        for second in layout["obstacles"][i + 1:]:
            if math.dist(first, second) < 2.0 * CYLINDER_RADIUS:
                raise ValueError(f"{layout['name']}: overlapping cylinders")
    half_width = W_EFF / 2.0
    for start, end in zip(polyline, polyline[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        normal = (-dy / length, dx / length)
        for point in (start, end):
            for sign in (-1.0, 1.0):
                x, y = point[0] + sign * half_width * normal[0], point[1] + sign * half_width * normal[1]
                if not (DOMAIN["x"][0] <= x <= DOMAIN["x"][1] and DOMAIN["y"][0] <= y <= DOMAIN["y"][1]):
                    raise ValueError(f"{layout['name']}: route boundary leaves display domain")
    layout["geometry_check"] = {
        "minimum_centerline_clearance_m": round(min_clearance, 6),
        "required_centerline_clearance_m": round(required, 6),
        "obstacle_overlap": False,
        "route_boundary_in_domain": True,
    }
    first_group = sorted(layout["obstacles"], key=lambda point: point[1])[:2]
    visibility = []
    for x, y in first_group:
        forward = y - (ROBOT_Y + CAMERA_MOUNT_Y)
        lateral_limit = forward * math.tan(math.radians(DEPTH_HORIZONTAL_FOV_DEG / 2.0))
        if not (DEPTH_NEAR_M <= forward <= DEPTH_FAR_M and abs(x) <= lateral_limit):
            raise ValueError(f"{layout['name']}: first obstacle group is outside the nominal depth view")
        visibility.append({"x_m": round(x, 6), "forward_from_mount_m": round(forward, 6)})
    layout["first_group_depth_visibility_check"] = {
        "camera_mount_y_m": CAMERA_MOUNT_Y,
        "horizontal_fov_deg": DEPTH_HORIZONTAL_FOV_DEG,
        "near_far_m": [DEPTH_NEAR_M, DEPTH_FAR_M],
        "first_group": visibility,
    }


def add_marker(gym, sim, env, start, end, lateral, color):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    marker_options = gymapi.AssetOptions()
    marker_options.fix_base_link = True
    marker_options.disable_gravity = True
    asset = gym.create_box(sim, length, 0.04, 0.018, marker_options)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3((start[0] + end[0]) / 2.0 + lateral * (-dy / length),
                         (start[1] + end[1]) / 2.0 + lateral * (dx / length), 0.018)
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), math.atan2(dy, dx))
    actor = gym.create_actor(env, asset, pose, "route_marker", 0, 0)
    gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))


def create_scene(gym, sim, env, layout, cylinder_asset, robot_asset):
    for obstacle in layout["obstacles"]:
        pose = gymapi.Transform(p=gymapi.Vec3(obstacle[0], obstacle[1], CYLINDER_HEIGHT / 2.0))
        actor = gym.create_actor(env, cylinder_asset, pose, "cylinder", 0, 0)
        gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.30, 0.33, 0.37))
    for start, end in zip(layout["polyline"], layout["polyline"][1:]):
        add_marker(gym, sim, env, start, end, 0.0, (0.90, 0.05, 0.05))
    robot_pose = gymapi.Transform(p=gymapi.Vec3(0.0, ROBOT_Y, 0.32))
    robot = gym.create_actor(env, robot_asset, robot_pose, "static_hexapod", 0, 0)
    gym.set_rigid_body_color(env, robot, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.20, 0.43, 0.68))
    states = gym.get_actor_dof_states(env, robot, gymapi.STATE_ALL)
    names = gym.get_asset_dof_names(robot_asset)
    for index, name in enumerate(names):
        if name.endswith("thigh"):
            states["pos"][index] = 0.5 if ("rf" in name or "lb" in name) else (-0.5 if ("lf" in name or "rb" in name) else 0.0)
        elif name.endswith("knee"):
            states["pos"][index] = 0.67
        elif name.endswith("ankle"):
            states["pos"][index] = -2.2
    gym.set_actor_dof_states(env, robot, states, gymapi.STATE_POS)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    layouts = make_layouts()
    for layout in layouts:
        check_layout(layout)
    metadata = {
        "purpose": "static Isaac Gym preview only; not a training generator or policy output",
        "coordinates": "+x right, +y forward, meters",
        "domain_m": DOMAIN,
        "cylinder": {"radius_m": CYLINDER_RADIUS, "height_m": CYLINDER_HEIGHT},
        "route": {
            "meaning": "reserved center route, not a strategy output or optimal route",
            "W_eff_m": W_EFF,
            "W_eff_note": "nominal 2*0.27 m collision-envelope clearance illustration; not a leg-swing-accurate robot guarantee",
            "red_markers": "reserved center route only; the nominal envelope is geometry-checked but not drawn",
            "preview_forward_shift_m": PREVIEW_FORWARD_SHIFT,
            "spawn_note": "robot stays at y=0.45; the old route and obstacles are shifted forward with a straight entry added",
        },
        "limits": "Suggested speeds are non-dynamic; no FOV, lateral-response, or reaction-time calibration is claimed.",
        "layouts": layouts,
    }
    (OUT / "layouts.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    params.physx.num_position_iterations = 4
    params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym CPU PhysX sim creation failed")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    fixed = gymapi.AssetOptions()
    fixed.fix_base_link = True
    fixed.disable_gravity = True
    cylinder_asset = gym.load_asset(sim, str(CYLINDER_URDF.parent), CYLINDER_URDF.name, fixed)
    robot_asset = gym.load_asset(sim, str(ROOT / "resources" / "robots" / "hex_v4" / "urdf"), "hex_ground.urdf", fixed)
    if cylinder_asset is None or robot_asset is None:
        raise RuntimeError("Isaac Gym could not load a required static asset")
    lower = gymapi.Vec3(-2.0, -0.5, -0.2)
    upper = gymapi.Vec3(2.0, 10.5, 1.5)
    envs = [gym.create_env(sim, lower, upper, 3) for _ in layouts]
    for env, layout in zip(envs, layouts):
        create_scene(gym, sim, env, layout, cylinder_asset, robot_asset)

    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280
    camera_props.height = 1600
    camera_props.horizontal_fov = 32.0
    camera_props.near_plane = 0.05
    camera_props.far_plane = 40.0
    cameras = [gym.create_camera_sensor(env, camera_props) for env in envs]
    for env, camera in zip(envs, cameras):
        gym.set_camera_location(camera, env, gymapi.Vec3(3.7, -3.6, 16.0), gymapi.Vec3(0.0, 5.0, 0.0))
    for _ in range(3):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
    for env, camera, layout in zip(envs, cameras, layouts):
        gym.write_camera_image_to_file(sim, env, camera, gymapi.IMAGE_COLOR, str(OUT / f"{layout['name']}.png"))
    gym.destroy_sim(sim)
    print(f"Rendered {len(layouts)} Isaac Gym previews to {OUT}")


if __name__ == "__main__":
    main()
