#!/usr/bin/env python3
"""Plot outcome-conditioned representative trajectories on the held-out layout."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator


FIGURE_SCALE = 1.5
AXES_BOX = [0.065, 0.07, 0.88, 0.77]
PCR_LINEWIDTH = 1.25 * FIGURE_SCALE
BASELINE_LINEWIDTH = 0.70 * FIGURE_SCALE
PCR_ALPHA = 1.0
BASELINE_ALPHA = 0.58
PCR_HALO_LINEWIDTH = PCR_LINEWIDTH + 1.0 * FIGURE_SCALE
PROFILE_GRID_M = np.linspace(-1.75, 6.85, 87)
PROFILE_TERMINAL_WEIGHT = 0.16
GREEDY_METHOD_ORDER = [
    "rule_override",
    "additive_fusion",
    "fixed_authority",
    "geomw",
    "yonly",
]
TRACK_ORDER = [
    "pcr",
    "rule_override",
    "additive_fusion",
    "fixed_authority",
    "geomw",
    "risk_only",
    "yonly",
]

METHODS = {
    "pcr": {
        "label": "PCR",
        "variant": "learnedw2",
        "color": "#d62728",
        "linestyle": "-",
        "marker": None,
    },
    "rule_override": {
        "label": "Rule-Override",
        "variant": "rule_override",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": None,
    },
    "additive_fusion": {
        "label": "Additive-Fusion",
        "variant": "additive_fusion",
        "color": "#9467bd",
        "linestyle": "-",
        "marker": None,
    },
    "fixed_authority": {
        "label": "Fixed-Authority",
        "variant": "fixed_authority",
        "color": "#8c564b",
        "linestyle": "-",
        "marker": None,
    },
    "geomw": {
        "label": "Geom-w",
        "variant": "geomw",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": None,
    },
    "risk_only": {
        "label": "Risk-only",
        "variant": "risk_only",
        "color": "#ff7f0e",
        "linestyle": "-",
        "marker": None,
    },
    "yonly": {
        "label": "Y-only",
        "variant": "yonly",
        "color": "#6b6b6b",
        "linestyle": "-",
        "marker": None,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_file(pattern: str) -> Path:
    matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        raise RuntimeError(f"expected one file for {pattern!r}, found {len(matches)}")
    return matches[0]


def _load_method(input_root: Path, method: str) -> Dict:
    method_root = input_root / method
    timeseries_path = _one_file(str(method_root / "**" / "timeseries.csv"))
    metrics_path = _one_file(str(method_root / "**" / "metrics.json"))
    with timeseries_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    episodes = metrics.get("per_episode", [])
    if not rows or len(episodes) != 64:
        raise RuntimeError(f"{method}: expected a 64-environment pool")
    episode_by_env = {int(row["env_id"]): row for row in episodes}
    if sorted(episode_by_env) != list(range(64)):
        raise RuntimeError(f"{method}: expected one completed episode from every env_id")
    rows_by_episode: Dict[int, List[Dict]] = {}
    for row in rows:
        rows_by_episode.setdefault(int(row["episode_id"]), []).append(row)
    for episode_rows in rows_by_episode.values():
        episode_rows.sort(key=lambda row: (float(row["time_s"]), int(row["step_hl"])))

    protocol = metrics.get("protocol", {})
    expected_variant = METHODS[method]["variant"]
    if str(protocol.get("policy_variant")) != expected_variant:
        raise RuntimeError(
            f"{method}: policy_variant={protocol.get('policy_variant')!r}, "
            f"expected {expected_variant!r}"
        )
    if int(protocol.get("seed", -1)) != 1:
        raise RuntimeError(f"{method}: expected seed 1")
    if not math.isclose(
        float(protocol.get("resolved_moving_target_pcr_line_speed", float("nan"))),
        0.60,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(f"{method}: expected target speed 0.60 m/s")
    if method == "pcr" and not bool(protocol.get("pcr_lateral_direction_memory", False)):
        raise RuntimeError("PCR trace must use the approved lateral-direction memory")
    if method == "fixed_authority":
        if not math.isclose(
            float(protocol.get("fixed_authority_y_const", float("nan"))),
            0.50,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Fixed-Authority must use y*=0.50")
        if int(protocol.get("fixed_authority_gate_policy_call_count", -1)) != 0:
            raise RuntimeError("Fixed-Authority must bypass the Gate policy")
    return {
        "method": method,
        "timeseries_path": timeseries_path,
        "metrics_path": metrics_path,
        "protocol": protocol,
        "episode_by_env": episode_by_env,
        "rows_by_episode": rows_by_episode,
    }


def _select_method_episode(dataset: Dict, env_id: int, layout: Dict) -> Dict:
    episode = dataset["episode_by_env"][env_id]
    episode_id = int(episode["episode_id"])
    rows = dataset["rows_by_episode"].get(episode_id, [])
    if len(rows) < 2:
        raise RuntimeError(f"{dataset['method']}: missing trajectory for episode_id={episode_id}")

    observed = json.loads(rows[0]["obstacles_json"])
    if len(observed) != len(layout["obstacles"]):
        raise RuntimeError(f"{dataset['method']}: obstacle count mismatch")
    origin_x_values = []
    origin_y_values = []
    for index, expected in enumerate(layout["obstacles"]):
        item = observed[index]
        if int(item.get("slot", index)) != index:
            raise RuntimeError(f"{dataset['method']}: obstacle slot order mismatch")
        origin_x_values.append(float(item["x"]) - float(expected["x"]))
        origin_y_values.append(float(item["y"]) - float(expected["y"]))
    origin_x = float(sorted(origin_x_values)[len(origin_x_values) // 2])
    origin_y = float(sorted(origin_y_values)[len(origin_y_values) // 2])
    residual = max(
        max(abs(value - origin_x) for value in origin_x_values),
        max(abs(value - origin_y) for value in origin_y_values),
    )
    if residual > 2e-5:
        raise RuntimeError(f"{dataset['method']}: layout is not a pure env-origin translation")

    local_rows = []
    for row in rows:
        local = dict(row)
        local["robot_x_local"] = float(row["robot_x"]) - origin_x
        local["robot_y_local"] = float(row["robot_y"]) - origin_y
        local["target_x_local"] = float(row["target_x"]) - origin_x
        local["target_y_local"] = float(row["target_y"]) - origin_y
        local_rows.append(local)

    start_xy = tuple(float(value) for value in layout["coordinate_convention"]["start_xy"])
    last = local_rows[-1]
    previous = local_rows[-2]
    last_to_start = math.hypot(
        last["robot_x_local"] - start_xy[0],
        last["robot_y_local"] - start_xy[1],
    )
    reset_jump = math.hypot(
        last["robot_x_local"] - previous["robot_x_local"],
        last["robot_y_local"] - previous["robot_y_local"],
    )
    if not (last_to_start < 0.05 and reset_jump > 1.0):
        raise RuntimeError(f"{dataset['method']}: expected one post-reset sentinel row")
    reason = str(last.get("episode_termination_reason", "")).strip().lower()
    if reason not in {"success", "collision", "target_lost", "timeout", "follow_lost"}:
        raise RuntimeError(f"{dataset['method']}: unsupported termination reason {reason!r}")
    return {
        **dataset,
        "episode": episode,
        "episode_id": episode_id,
        "rows": rows,
        "plot_rows": local_rows[:-1],
        "termination": reason,
        "reset_row_omitted": True,
        "reset_jump_m": reset_jump,
        "env_origin_world_xy_m": [origin_x, origin_y],
        "env_id": env_id,
    }


def _row_centers(layout: Dict) -> Dict[int, float]:
    centers: Dict[int, float] = {}
    for obstacle in layout["obstacles"]:
        row = int(obstacle["row"])
        y = float(obstacle["y"])
        if row in centers and not math.isclose(centers[row], y, abs_tol=1e-9):
            raise RuntimeError(f"row {row} has inconsistent forward coordinates")
        centers[row] = y
    return dict(sorted(centers.items()))


def _collision_row(dataset: Dict, row_centers: Dict[int, float]) -> Optional[int]:
    if dataset["termination"] != "collision":
        return None
    terminal_y = float(dataset["plot_rows"][-1]["robot_y_local"])
    return min(row_centers, key=lambda row: abs(terminal_y - row_centers[row]))


def _trajectory_profile(dataset: Dict) -> np.ndarray:
    rows = dataset["plot_rows"]
    lateral = np.asarray([row["robot_x_local"] for row in rows], dtype=float)
    forward = np.asarray([row["robot_y_local"] for row in rows], dtype=float)
    return np.asarray(
        [
            lateral[int(np.argmin(np.abs(forward - sample_y)))]
            if sample_y <= float(np.max(forward)) + 0.05
            else np.nan
            for sample_y in PROFILE_GRID_M
        ],
        dtype=float,
    )


def _trajectory_distance(left: Dict, right: Dict) -> float:
    mask = np.isfinite(left["profile"]) & np.isfinite(right["profile"])
    lateral_distance = float(
        np.mean(np.abs(left["profile"][mask] - right["profile"][mask]))
    )
    terminal_distance = PROFILE_TERMINAL_WEIGHT * abs(
        left["terminal_y_m"] - right["terminal_y_m"]
    )
    return lateral_distance + terminal_distance


def _describe_candidate(dataset: Dict, row_centers: Dict[int, float]) -> Dict:
    terminal = dataset["plot_rows"][-1]
    return {
        "dataset": dataset,
        "method": dataset["method"],
        "env_id": int(dataset["env_id"]),
        "profile": _trajectory_profile(dataset),
        "terminal_y_m": float(terminal["robot_y_local"]),
        "collision_row": _collision_row(dataset, row_centers),
        "task_success": int(dataset["episode"].get("task_success", 0)),
    }


def _select_representative_tracks(method_datasets: Dict[str, Dict], layout: Dict) -> Dict[str, Dict]:
    centers = _row_centers(layout)
    candidates: Dict[str, List[Dict]] = {}
    for method, dataset in method_datasets.items():
        candidates[method] = [
            _describe_candidate(_select_method_episode(dataset, env_id, layout), centers)
            for env_id in range(64)
        ]

    pcr_candidates = [item for item in candidates["pcr"] if item["task_success"] == 1]
    if not pcr_candidates:
        raise RuntimeError("no successful PCR trajectory in the 64-environment pool")
    pcr = max(
        pcr_candidates,
        key=lambda item: (
            float(np.nanmax(item["profile"]) - np.nanmin(item["profile"])),
            -item["env_id"],
        ),
    )

    row4_risk = [
        item
        for item in candidates["risk_only"]
        if item["task_success"] == 0
        and item["dataset"]["termination"] == "collision"
        and item["collision_row"] == 4
    ]
    if len(row4_risk) < 2:
        raise RuntimeError("fewer than two Risk-only row-4 collisions in the candidate pool")
    risk_a, risk_b = max(
        itertools.combinations(row4_risk, 2),
        key=lambda pair: (
            _trajectory_distance(pair[0], pair[1]),
            -pair[0]["env_id"],
            -pair[1]["env_id"],
        ),
    )
    risk_a, risk_b = sorted((risk_a, risk_b), key=lambda item: item["env_id"])

    selected = [pcr, risk_a, risk_b]
    selected_by_method: Dict[str, Dict] = {}
    for method in GREEDY_METHOD_ORDER:
        eligible = [
            item
            for item in candidates[method]
            if item["task_success"] == 0
            and item["dataset"]["termination"] != "success"
            and not (
                item["dataset"]["termination"] == "collision"
                and item["collision_row"] == 4
            )
        ]
        if not eligible:
            raise RuntimeError(f"{method}: no eligible failed representative trajectory")
        chosen = max(
            eligible,
            key=lambda item: (
                min(_trajectory_distance(item, previous) for previous in selected),
                -item["env_id"],
            ),
        )
        selected.append(chosen)
        selected_by_method[method] = chosen

    comparison_tracks = [pcr, *selected_by_method.values()]
    risk_only = max(
        (risk_a, risk_b),
        key=lambda item: (
            float(np.mean([
                _trajectory_distance(item, previous)
                for previous in comparison_tracks
            ])),
            min(
                _trajectory_distance(item, previous)
                for previous in comparison_tracks
            ),
            -item["env_id"],
        ),
    )

    tracks = {
        "pcr": pcr,
        "rule_override": selected_by_method["rule_override"],
        "additive_fusion": selected_by_method["additive_fusion"],
        "fixed_authority": selected_by_method["fixed_authority"],
        "geomw": selected_by_method["geomw"],
        "risk_only": risk_only,
        "yonly": selected_by_method["yonly"],
    }
    row4_collision_tracks = [
        track_id
        for track_id, item in tracks.items()
        if item["dataset"]["termination"] == "collision" and item["collision_row"] == 4
    ]
    if row4_collision_tracks != ["risk_only"]:
        raise RuntimeError(f"unexpected row-4 collision tracks: {row4_collision_tracks}")
    if tracks["pcr"]["task_success"] != 1:
        raise RuntimeError("selected PCR trajectory is not successful")
    if any(
        item["task_success"] != 0
        for track_id, item in tracks.items()
        if track_id != "pcr"
    ):
        raise RuntimeError("a selected baseline trajectory is not a task failure")
    return tracks


def _terminal_marker(
    ax,
    x: float,
    y: float,
    reason: str,
    color: str,
    size_scale: float = 1.0,
) -> None:
    if reason == "success":
        ax.scatter(
            [x], [y], marker="*", s=145 * size_scale ** 2, color="#d62728",
            edgecolor="white", linewidth=0.7 * size_scale, zorder=22, clip_on=False,
        )
        return
    if reason == "collision":
        ax.scatter(
            [x], [y], marker="X", s=36 * size_scale ** 2, color=color,
            edgecolor="white", linewidth=0.45 * size_scale, alpha=0.78,
            zorder=22, clip_on=False,
        )
        return
    ax.scatter(
        [x], [y], marker="o", s=34 * size_scale ** 2, facecolor="white",
        edgecolor=color, linewidth=1.15 * size_scale, alpha=0.78,
        zorder=22, clip_on=False,
    )


def _add_direction_arrows(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    color: str,
) -> None:
    valid = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(valid) < 6:
        return
    arrow_count = min(4, max(2, len(valid) // 45))
    indexes = [
        int(round((index + 1) * (len(valid) - 2) / (arrow_count + 1)))
        for index in range(arrow_count)
    ]
    used = set()
    for index in indexes:
        index = max(0, min(index, len(valid) - 2))
        if index in used:
            continue
        used.add(index)
        x0, y0 = valid[index]
        x1, y1 = valid[index + 1]
        if abs(x1 - x0) + abs(y1 - y0) < 1e-5:
            continue
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.0 * FIGURE_SCALE,
                "mutation_scale": 7.0 * FIGURE_SCALE,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=7,
        )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _build_manifest(
    layout_path: Path,
    layout: Dict,
    tracks: Dict[str, Dict],
    png_path: Path,
    pdf_path: Path,
    plot_xlim: Sequence[float],
    plot_ylim: Sequence[float],
) -> Dict:
    trajectories = {}
    for track_id in TRACK_ORDER:
        descriptor = tracks[track_id]
        dataset = descriptor["dataset"]
        method = descriptor["method"]
        style = METHODS[method]
        rows = dataset["plot_rows"]
        episode = dataset["episode"]
        trajectories[track_id] = {
            "method": method,
            "label": style["label"],
            "style": {
                "color": style["color"],
                "linestyle": style["linestyle"],
                "marker": style["marker"],
                "linewidth": (
                    PCR_LINEWIDTH if method == "pcr" else BASELINE_LINEWIDTH
                ),
                "alpha": PCR_ALPHA if method == "pcr" else BASELINE_ALPHA,
                "solid_capstyle": "round",
                "white_halo_linewidth": (
                    PCR_HALO_LINEWIDTH if method == "pcr" else None
                ),
            },
            "termination": dataset["termination"],
            "terminal_glyph": (
                "success_star"
                if dataset["termination"] == "success"
                else "collision_x"
                if dataset["termination"] == "collision"
                else "follow_lost_timeout_circle"
            ),
            "collision_row": descriptor["collision_row"],
            "env_id": int(dataset["env_id"]),
            "episode_id_in_completion_order": int(dataset["episode_id"]),
            "task_success": int(episode.get("task_success", 0)),
            "episode_collision": int(episode.get("episode_collision", 0)),
            "row_progress": float(episode.get("progress_ratio_best", float("nan"))),
            "raw_point_count": len(dataset["rows"]),
            "plotted_point_count": len(rows),
            "terminal_marker_basis": "last recorded pre-reset sample",
            "post_reset_sentinel_omitted": bool(dataset["reset_row_omitted"]),
            "post_reset_jump_m": float(dataset["reset_jump_m"]),
            "env_origin_world_xy_m": dataset["env_origin_world_xy_m"],
            "source": {
                "timeseries_csv": _relative(dataset["timeseries_path"]),
                "timeseries_sha256": _sha256(dataset["timeseries_path"]),
                "metrics_json": _relative(dataset["metrics_path"]),
                "metrics_sha256": _sha256(dataset["metrics_path"]),
                "pcr_ckpt": dataset["protocol"].get("pcr_ckpt"),
                "avoid_ckpt": dataset["protocol"].get("avoid_ckpt"),
                "lowlevel_ckpt": dataset["protocol"].get("lowlevel_ckpt"),
            },
            "points": [
                {
                    "time_s": float(row["time_s"]),
                    "robot_xy_m": [row["robot_x_local"], row["robot_y_local"]],
                    "target_xy_m": [row["target_x_local"], row["target_y_local"]],
                }
                for row in rows
            ],
        }

    centers = _row_centers(layout)
    row_spacings = [
        centers[row + 1] - centers[row]
        for row in range(min(centers), max(centers))
    ]
    return {
        "schema_version": 2,
        "artifact_role": "outcome_conditioned_representative_trajectory_comparison",
        "selection_rule": {
            "candidate_pool": "seed-1, 64 environments, fixed layout and 0.60 m/s",
            "outcomes": "one successful PCR trajectory and failed baseline trajectories",
            "row4_requirement": "one failed Risk-only row-4 collision trajectory is shown",
            "diversity": (
                "PCR maximizes lateral range; two maximally separated Risk-only row-4 "
                "candidates preserve the prior greedy selection context; the displayed "
                "Risk-only member maximizes mean distance to the displayed non-Risk methods"
            ),
            "distance_definition": (
                "mean absolute lateral-profile separation on the common forward grid plus "
                f"{PROFILE_TERMINAL_WEIGHT} times terminal forward separation"
            ),
            "forward_profile_grid_m": PROFILE_GRID_M.tolist(),
            "greedy_method_order": GREEDY_METHOD_ORDER,
        },
        "protocol": {
            "layout_id": layout["layout_id"],
            "layout_version": int(layout["version"]),
            "layout_sha256": layout["sha256"],
            "seed": 1,
            "target_speed_mps": 0.60,
            "num_envs": 64,
            "episodes_per_method": 64,
            "methods": [METHODS[key]["label"] for key in METHODS],
            "mono_ppo_included": False,
            "selected_env_ids_by_track": {
                track_id: int(tracks[track_id]["env_id"]) for track_id in TRACK_ORDER
            },
        },
        "layout": {
            "source": _relative(layout_path),
            "source_sha256": _sha256(layout_path),
            "obstacles": layout["obstacles"],
            "row_centers_forward_m": centers,
            "row_spacing_m": row_spacings,
        },
        "transformations": [
            "selected representative episodes independently within each method pool",
            "subtracted each selected environment world origin to recover layout-local coordinates",
            "omitted the final post-reset sentinel row from each trajectory",
            "used the last pre-reset sample as the terminal-marker location",
            "rotated the display so forward is horizontal",
            "kept forward and lateral coordinates at the original Fig. 6 physical 1:1 scale",
            "rendered obstacles as physical-radius circles in the original Fig. 6 style",
            "matched start, terminal, and target glyphs to the original Fig. 6 style",
            "added only the adjacent-row spacing dimension arrows and labels",
            "cropped the view to robot-trajectory and obstacle extents with 0.10 m padding",
            "no per-trajectory translation, smoothing, interpolation, or resampling in the plotted data",
        ],
        "trajectories": trajectories,
        "figure": {
            "png": _relative(png_path),
            "png_sha256": _sha256(png_path),
            "pdf": _relative(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "matplotlib_version": matplotlib.__version__,
            "size_inches": [8.0 * FIGURE_SCALE, 3.4 * FIGURE_SCALE],
            "png_dpi": 300,
            "global_visual_scale": FIGURE_SCALE,
            "axes_box_fraction": AXES_BOX,
            "speed_annotation": None,
            "display_transform": "plot_x=forward_y; plot_y=lateral_x",
            "lateral_display_scale": 1.0,
            "view": "single_overview",
            "xlim": [float(value) for value in plot_xlim],
            "ylim": [float(value) for value in plot_ylim],
            "obstacle_glyph": "physical-radius circle centered at the obstacle position",
            "axis_limits_basis": "robot trajectories and obstacle footprints plus 0.10 m padding",
            "row_spacing_annotation": "only visual addition relative to the original Fig. 6 style",
            "visual_hierarchy": (
                "PCR uses a narrow white halo and full opacity; baselines use thin "
                "semi-transparent solid lines; coordinates and samples are unchanged"
            ),
            "event_glyphs": {
                "start": "black square with white edge",
                "success": "red star with white edge",
                "collision": "method-color X with white edge; gray X in legend",
                "follow_lost_timeout": (
                    "white-filled method-color circle; gray outline circle in legend"
                ),
                "target": "green dashed line with direction arrows",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--layout_json", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    with args.layout_json.open("r", encoding="utf-8") as handle:
        layout = json.load(handle)
    method_datasets = {
        method: _load_method(args.input_root, method) for method in METHODS
    }
    layout_hashes = {
        data["protocol"].get("layout_sha256") for data in method_datasets.values()
    }
    if layout_hashes != {layout.get("sha256")}:
        raise RuntimeError(
            f"layout hash mismatch: eval={layout_hashes}, JSON={layout.get('sha256')}"
        )
    tracks = _select_representative_tracks(method_datasets, layout)
    centers = _row_centers(layout)
    obstacle_radius = float(layout["obstacle_geometry"]["capsule"]["radius"])
    forward_values: List[float] = []
    lateral_values: List[float] = []
    for descriptor in tracks.values():
        for row in descriptor["dataset"]["plot_rows"]:
            forward_values.append(float(row["robot_y_local"]))
            lateral_values.append(float(row["robot_x_local"]))
    for item in layout["obstacles"]:
        forward = float(item["y"])
        lateral = float(item["x"])
        forward_values.extend([forward - obstacle_radius, forward + obstacle_radius])
        lateral_values.extend([lateral - obstacle_radius, lateral + obstacle_radius])
    plot_padding = 0.10
    plot_xlim = (
        min(forward_values) - plot_padding,
        max(forward_values) + plot_padding,
    )
    plot_ylim = (
        min(lateral_values) - plot_padding,
        max(lateral_values) + plot_padding,
    )

    fig = plt.figure(figsize=(8.0 * FIGURE_SCALE, 3.4 * FIGURE_SCALE))
    ax = fig.add_axes(AXES_BOX)
    row_items = list(centers.items())
    for (_, left), (_, right) in zip(row_items[:-1], row_items[1:]):
        ax.annotate(
            "",
            xy=(right, 1.025),
            xytext=(left, 1.025),
            xycoords=ax.get_xaxis_transform(),
            textcoords=ax.get_xaxis_transform(),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#555555",
                "linewidth": 0.8 * FIGURE_SCALE,
                "mutation_scale": 10.0 * FIGURE_SCALE,
            },
            annotation_clip=False,
            zorder=25,
        )
        ax.text(
            0.5 * (left + right),
            1.035,
            f"{right - left:.2f} m",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.2 * FIGURE_SCALE,
            color="#333333",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.8 * FIGURE_SCALE,
            },
            clip_on=False,
            zorder=26,
        )

    pcr_rows = tracks["pcr"]["dataset"]["plot_rows"]

    def draw_obstacles(axis, alpha: float) -> None:
        for item in layout["obstacles"]:
            axis.add_patch(
                Circle(
                    (float(item["y"]), float(item["x"])),
                    radius=obstacle_radius,
                    facecolor="#9e9e9e",
                    edgecolor="#666666",
                    linewidth=0.5 * FIGURE_SCALE,
                    alpha=alpha,
                    zorder=1,
                )
            )

    def draw_target(axis, alpha: float) -> None:
        target_forward = [row["target_y_local"] for row in pcr_rows]
        target_lateral = [row["target_x_local"] for row in pcr_rows]
        axis.plot(
            target_forward,
            target_lateral,
            color="#009e73",
            linewidth=1.2 * FIGURE_SCALE,
            linestyle=(0, (5.0, 3.0)),
            alpha=alpha,
            zorder=2,
        )
        _add_direction_arrows(axis, target_forward, target_lateral, "#009e73")

    def draw_tracks(
        axis,
        terminal_size: float,
        show_start: bool,
    ) -> None:
        for track_id in TRACK_ORDER:
            descriptor = tracks[track_id]
            dataset = descriptor["dataset"]
            method = descriptor["method"]
            style = METHODS[method]
            rows = dataset["plot_rows"]
            forward = [row["robot_y_local"] for row in rows]
            lateral = [row["robot_x_local"] for row in rows]
            linewidth = PCR_LINEWIDTH if method == "pcr" else BASELINE_LINEWIDTH
            alpha = PCR_ALPHA if method == "pcr" else BASELINE_ALPHA
            line, = axis.plot(
                forward,
                lateral,
                color=style["color"],
                linestyle="-",
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5 if method == "pcr" else 3,
            )
            if method == "pcr":
                line.set_path_effects([
                    path_effects.Stroke(
                        linewidth=PCR_HALO_LINEWIDTH,
                        foreground="white",
                        alpha=0.94,
                    ),
                    path_effects.Normal(),
                ])
            _terminal_marker(
                axis,
                forward[-1],
                lateral[-1],
                dataset["termination"],
                style["color"],
                size_scale=terminal_size,
            )
        if show_start:
            start = tracks["pcr"]["dataset"]["plot_rows"][0]
            axis.scatter(
                [float(start["robot_y_local"])],
                [float(start["robot_x_local"])],
                marker="s",
                s=36 * FIGURE_SCALE ** 2,
                facecolor="#000000",
                edgecolor="white",
                linewidth=0.7 * FIGURE_SCALE,
                zorder=21,
                clip_on=False,
            )

    draw_obstacles(ax, alpha=0.38)
    draw_target(ax, alpha=0.95)
    draw_tracks(ax, terminal_size=FIGURE_SCALE, show_start=True)

    line_handles: List[Line2D] = []
    legend_methods = set()
    for track_id in TRACK_ORDER:
        descriptor = tracks[track_id]
        dataset = descriptor["dataset"]
        method = descriptor["method"]
        style = METHODS[method]
        linewidth = PCR_LINEWIDTH if method == "pcr" else BASELINE_LINEWIDTH
        alpha = PCR_ALPHA if method == "pcr" else BASELINE_ALPHA
        if method not in legend_methods:
            legend_methods.add(method)
            line_handles.append(
                Line2D(
                    [0], [0], color=style["color"], linestyle="-",
                    linewidth=linewidth, alpha=alpha, label=style["label"],
                )
            )

    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world y / forward [m]", fontsize=8 * FIGURE_SCALE)
    ax.set_ylabel("world x / lateral [m]", fontsize=8 * FIGURE_SCALE)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(
        True, which="major", linestyle="--",
        linewidth=0.40 * FIGURE_SCALE, alpha=0.26,
    )
    ax.grid(
        True, which="minor", linestyle=":",
        linewidth=0.25 * FIGURE_SCALE, alpha=0.12,
    )
    ax.tick_params(
        which="major",
        labelsize=7 * FIGURE_SCALE,
        width=0.8 * FIGURE_SCALE,
        length=3.5 * FIGURE_SCALE,
    )
    ax.tick_params(
        which="minor",
        width=0.6 * FIGURE_SCALE,
        length=2.0 * FIGURE_SCALE,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8 * FIGURE_SCALE)
    ax.set_axisbelow(True)
    status_handles = [
        Line2D([0], [0], color="#009e73", linestyle=(0, (5.0, 3.0)), linewidth=1.2 * FIGURE_SCALE, label="Target"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#000000", markeredgecolor="white", markeredgewidth=0.7 * FIGURE_SCALE, markersize=6 * FIGURE_SCALE, label="Start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728", markeredgecolor="white", markeredgewidth=0.7 * FIGURE_SCALE, markersize=10 * FIGURE_SCALE, label="Success"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=0.45 * FIGURE_SCALE, markersize=6 * FIGURE_SCALE, alpha=0.78, label="Collision"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#777777", markeredgewidth=1.15 * FIGURE_SCALE, markersize=6 * FIGURE_SCALE, alpha=0.78, label="Follow lost/timeout"),
    ]
    fig.legend(
        handles=line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=7,
        frameon=False,
        fontsize=6.0 * FIGURE_SCALE,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=0.65,
    )
    fig.legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=6.0 * FIGURE_SCALE,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=0.65,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    json_path = args.output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    manifest = _build_manifest(
        args.layout_json,
        layout,
        tracks,
        png_path,
        pdf_path,
        plot_xlim,
        plot_ylim,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("selected trajectories:")
    for track_id in TRACK_ORDER:
        item = tracks[track_id]
        print(
            f"  {track_id}: env={item['env_id']} "
            f"termination={item['dataset']['termination']} "
            f"collision_row={item['collision_row']}"
        )
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
