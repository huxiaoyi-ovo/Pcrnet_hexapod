#!/usr/bin/env python3
"""Generate code-grounded neural-network architecture diagrams as SVG."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_assets" / "network_architectures"

NAVY = "#182B49"
EDGE = "#263B64"
INPUT = "#BFE3C0"
ENCODER = "#C9D8F0"
TRUNK = "#C9C8F2"
HEAD = "#F4D49B"
OUTPUT = "#F2C1BC"
CRITIC = "#D9D1E8"
MAP = "#BBDDE5"
TEXT = "#172033"
MUTED = "#526079"


def setup(title: str, *, width: float = 16.0, height: float = 9.0):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.set_title(title, fontsize=19, color=TEXT, pad=12, fontweight="semibold")
    return fig, ax


def save(fig, filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def arrow(ax, start, end, *, color: str = EDGE, alpha: float = 0.9, lw: float = 1.25):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def neuron_layer(
    ax,
    x: float,
    y: float,
    dim: int,
    label: str,
    *,
    color: str,
    radius: float = 0.16,
    span: float = 1.45,
    label_position: str = "below",
    note: str | None = None,
    annotate: bool = True,
) -> list[tuple[float, float]]:
    if dim <= 5:
        ys = [y + (i - (dim - 1) / 2) * min(0.48, span / max(dim - 1, 1)) for i in range(dim)]
        dot_y = None
    else:
        ys = [y - span / 2, y - span / 4, y + span / 4, y + span / 2]
        dot_y = y

    for cy in ys:
        ax.add_patch(Circle((x, cy), radius, facecolor=color, edgecolor=NAVY, linewidth=1.45, zorder=3))
    if dot_y is not None:
        ax.text(x, dot_y, r"$\vdots$", ha="center", va="center", fontsize=14, color=NAVY)

    if annotate:
        label_y = y - span / 2 - 0.43 if label_position == "below" else y + span / 2 + 0.38
        va = "top" if label_position == "below" else "bottom"
        ax.text(x, label_y, label, ha="center", va=va, fontsize=9.2, color=TEXT, fontweight="semibold")
        ax.text(x, label_y - 0.22 if label_position == "below" else label_y + 0.22, f"{dim}-D", ha="center", va=va, fontsize=8.3, color=MUTED)
        if note:
            note_y = label_y - 0.43 if label_position == "below" else label_y + 0.43
            ax.text(x, note_y, note, ha="center", va=va, fontsize=7.5, color=MUTED)
    return [(x, cy) for cy in ys]


def connect_layers(ax, left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]], *, alpha: float = 0.28):
    for x1, y1 in left:
        for x2, y2 in right:
            arrow(ax, (x1 + 0.15, y1), (x2 - 0.15, y2), alpha=alpha, lw=0.65)


def feature_stack(
    ax,
    x: float,
    y: float,
    label: str,
    shape: str,
    *,
    color: str = MAP,
    width: float = 0.62,
    height: float = 0.86,
    depth: int = 4,
    note: str | None = None,
) -> tuple[float, float]:
    offset = 0.055
    for idx in reversed(range(depth)):
        ax.add_patch(
            Rectangle(
                (x - width / 2 + idx * offset, y - height / 2 + idx * offset),
                width,
                height,
                facecolor=color,
                edgecolor=NAVY,
                linewidth=1.15,
                alpha=0.96,
                zorder=2 + idx,
            )
        )
    center_x = x + (depth - 1) * offset / 2
    ax.text(center_x, y - height / 2 - 0.26, label, ha="center", va="top", fontsize=8.8, color=TEXT, fontweight="semibold")
    ax.text(center_x, y - height / 2 - 0.47, shape, ha="center", va="top", fontsize=7.8, color=MUTED)
    if note:
        ax.text(center_x, y - height / 2 - 0.66, note, ha="center", va="top", fontsize=7.2, color=MUTED)
    return center_x + width / 2 + (depth - 1) * offset, y


def connect_points(ax, left: tuple[float, float], right: tuple[float, float]):
    arrow(ax, left, right, alpha=0.8, lw=1.15)


def branch_to_layer(ax, starts: Iterable[tuple[float, float]], layer: Sequence[tuple[float, float]]):
    target_x = layer[0][0] - 0.16
    target_y = sum(y for _, y in layer) / len(layer)
    for sx, sy in starts:
        arrow(ax, (sx, sy), (target_x, target_y), alpha=0.62, lw=0.9)


def draw_shared_encoders(ax, *, state_dim: int, goal_dim: int, goal_label: str):
    map_y = 7.15
    p0 = feature_stack(ax, 0.65, map_y, "map + coords", "4 x 32 x 32", note="occupancy / clearance + x,y")
    p1 = feature_stack(ax, 1.85, map_y, "Conv1 + ELU", "32 x 16 x 16", note="k3 / s2")
    p2 = feature_stack(ax, 3.00, map_y, "Conv2 + ELU", "64 x 8 x 8", note="k3 / s2")
    p3 = feature_stack(ax, 4.15, map_y, "Conv3 + ELU", "128 x 8 x 8", note="k3 / s1")
    for left, right in zip((p0, p1, p2), (p1, p2, p3)):
        connect_points(ax, left, (right[0] - 0.65, right[1]))
    map_fc = neuron_layer(ax, 5.25, map_y, 128, "map feature", color=ENCODER, note="pool 4x4 + FC")
    branch_to_layer(ax, [p3], map_fc)

    state0 = neuron_layer(ax, 0.75, 4.45, state_dim, "robot state", color=INPUT)
    state1 = neuron_layer(ax, 2.05, 4.45, 64, "state FC1", color=ENCODER, note="ELU")
    state2 = neuron_layer(ax, 3.35, 4.45, 64, "state FC2", color=ENCODER, note="ELU")
    state3 = neuron_layer(ax, 4.65, 4.45, 64, "state feature", color=ENCODER)
    connect_layers(ax, state0, state1)
    connect_layers(ax, state1, state2)
    connect_layers(ax, state2, state3)

    goal0 = neuron_layer(ax, 0.75, 1.75, goal_dim, goal_label, color=INPUT)
    goal1 = neuron_layer(ax, 2.35, 1.75, 32, "goal FC1", color=ENCODER, note="ELU")
    goal2 = neuron_layer(ax, 3.75, 1.75, 32, "goal feature", color=ENCODER)
    connect_layers(ax, goal0, goal1)
    connect_layers(ax, goal1, goal2)

    difficulty = neuron_layer(ax, 5.10, 1.15, 1, "difficulty", color=INPUT, span=0.5, note="zero in final ckpt")
    return map_fc, state3, goal2, difficulty


def draw_critic_inset(ax, *, x0: float = 6.25, y: float = 0.75):
    ax.text(x0 - 0.1, y + 0.64, "training critic (separate encoders)", fontsize=8.8, color=MUTED, ha="left")
    layers = [
        neuron_layer(ax, x0, y, 225, "critic fusion", color=CRITIC, span=0.75),
        neuron_layer(ax, x0 + 1.15, y, 256, "FC", color=CRITIC, span=0.75),
        neuron_layer(ax, x0 + 2.30, y, 256, "FC", color=CRITIC, span=0.75),
        neuron_layer(ax, x0 + 3.45, y, 64, "FC", color=CRITIC, span=0.75),
        neuron_layer(ax, x0 + 4.55, y, 1, "value", color=OUTPUT, span=0.4),
    ]
    for left, right in zip(layers, layers[1:]):
        connect_layers(ax, left, right, alpha=0.18)


def draw_gate(*, learned_w: bool, filename: str, title: str, state_dim: int, goal_dim: int, goal_label: str) -> Path:
    fig, ax = setup(title, width=15.5, height=9.2)
    map_fc, state_fc, goal_fc, difficulty = draw_shared_encoders(
        ax, state_dim=state_dim, goal_dim=goal_dim, goal_label=goal_label
    )
    concat = neuron_layer(ax, 6.45, 4.65, 225, "fusion", color=TRUNK, note="128 + 64 + 32 + 1")
    for branch in (map_fc, state_fc, goal_fc, difficulty):
        branch_to_layer(ax, branch, concat)
    h1 = neuron_layer(ax, 7.85, 4.65, 256, "FC1", color=TRUNK, note="ELU")
    h2 = neuron_layer(ax, 9.20, 4.65, 256, "FC2", color=TRUNK, note="ELU")
    connect_layers(ax, concat, h1)
    connect_layers(ax, h1, h2)

    specs = [
        (6.85, r"$\alpha_y$", "Beta shape"),
        (5.35, r"$\beta_y$", "Beta shape"),
    ]
    if learned_w:
        specs += [
            (3.85, r"$\alpha_w$", "Beta shape"),
            (2.35, r"$\beta_w$", "Beta shape"),
        ]
    ax.text(10.55, 7.72, "head\n64-D + ELU", ha="center", va="top", fontsize=8.5, color=TEXT, fontweight="semibold")
    ax.text(11.85, 7.72, "output\n1-D, Softplus + 1", ha="center", va="top", fontsize=8.5, color=TEXT, fontweight="semibold")
    for row_y, symbol, note in specs:
        head = neuron_layer(ax, 10.55, row_y, 64, "", color=HEAD, span=0.78, annotate=False)
        out = neuron_layer(ax, 11.85, row_y, 1, "", color=OUTPUT, span=0.45, annotate=False)
        branch_to_layer(ax, h2, head)
        connect_layers(ax, head, out, alpha=0.34)
        ax.text(12.16, row_y, symbol, fontsize=10.2, color=TEXT, va="center", ha="left")

    if learned_w:
        ax.text(13.55, 6.10, r"$y \sim \mathrm{Beta}(\alpha_y,\beta_y)$", fontsize=10, color=NAVY, ha="center")
        ax.text(13.55, 3.10, r"$w \sim \mathrm{Beta}(\alpha_w,\beta_w)$", fontsize=10, color=NAVY, ha="center")
    else:
        ax.text(13.40, 6.10, r"$y \sim \mathrm{Beta}(\alpha_y,\beta_y)$", fontsize=10, color=NAVY, ha="center")
        ax.text(13.40, 5.65, "shared by Y-only and Risk-only checkpoints", fontsize=7.6, color=MUTED, ha="center")

    draw_critic_inset(ax)
    return save(fig, filename)


def draw_cmd_policy(*, filename: str, title: str, state_dim: int, output_note: str) -> Path:
    fig, ax = setup(title, width=15.2, height=9.2)
    map_fc, state_fc, goal_fc, difficulty = draw_shared_encoders(
        ax, state_dim=state_dim, goal_dim=2, goal_label="local goal"
    )
    concat = neuron_layer(ax, 6.45, 4.65, 225, "fusion", color=TRUNK, note="128 + 64 + 32 + 1")
    for branch in (map_fc, state_fc, goal_fc, difficulty):
        branch_to_layer(ax, branch, concat)
    h1 = neuron_layer(ax, 7.85, 4.65, 256, "FC1", color=TRUNK, note="ELU")
    h2 = neuron_layer(ax, 9.20, 4.65, 256, "FC2", color=TRUNK, note="ELU")
    connect_layers(ax, concat, h1)
    connect_layers(ax, h1, h2)

    ax.text(10.65, 7.05, "head: 64-D + ELU", ha="center", fontsize=8.5, color=TEXT, fontweight="semibold")
    ax.text(12.00, 7.05, "output: 3-D", ha="center", fontsize=8.5, color=TEXT, fontweight="semibold")
    mean_h = neuron_layer(ax, 10.65, 5.85, 64, "", color=HEAD, span=1.0, annotate=False)
    mean_o = neuron_layer(ax, 12.00, 5.85, 3, "", color=OUTPUT, span=1.0, annotate=False)
    std_h = neuron_layer(ax, 10.65, 3.30, 64, "", color=HEAD, span=1.0, annotate=False)
    std_o = neuron_layer(ax, 12.00, 3.30, 3, "", color=OUTPUT, span=1.0, annotate=False)
    for head in (mean_h, std_h):
        branch_to_layer(ax, h2, head)
    connect_layers(ax, mean_h, mean_o, alpha=0.34)
    connect_layers(ax, std_h, std_o, alpha=0.34)
    ax.text(12.38, 5.85, r"$\mu_x,\mu_y,\mu_\omega$  (tanh)", ha="left", va="center", fontsize=8.4, color=TEXT)
    ax.text(12.38, 3.30, r"$\sigma_x,\sigma_y,\sigma_\omega$  (Softplus)", ha="left", va="center", fontsize=8.4, color=TEXT)
    ax.text(13.70, 4.78, "tanh-squashed\nGaussian command", ha="center", fontsize=10.2, color=NAVY, fontweight="semibold")
    ax.text(13.70, 4.10, r"$[x_{\mathrm{right}},\,y_{\mathrm{forward}},\,\omega]$", ha="center", fontsize=10.2, color=TEXT)
    ax.text(13.70, 2.45, output_note, ha="center", fontsize=7.8, color=MUTED)
    draw_critic_inset(ax)
    return save(fig, filename)


def draw_low_level() -> Path:
    fig, ax = setup("Fixed Low-Level Locomotion Actor-Critic", width=14.3, height=7.4)
    ax.text(0.15, 6.18, "Actor", fontsize=13, color=NAVY, fontweight="semibold")
    actor_dims = [75, 512, 256, 128, 18]
    actor_names = ["observation", "FC1", "FC2", "FC3", "joint targets"]
    actor_colors = [INPUT, TRUNK, TRUNK, TRUNK, OUTPUT]
    actor_layers = []
    for idx, (dim, name, color) in enumerate(zip(actor_dims, actor_names, actor_colors)):
        actor_layers.append(neuron_layer(ax, 1.25 + idx * 2.55, 5.05, dim, name, color=color, span=1.75, note="ELU" if 0 < idx < 4 else None))
    for left, right in zip(actor_layers, actor_layers[1:]):
        connect_layers(ax, left, right, alpha=0.25)

    ax.text(0.15, 2.95, "Critic (training only)", fontsize=13, color=NAVY, fontweight="semibold")
    critic_dims = [230, 512, 256, 128, 1]
    critic_names = ["privileged obs.", "FC1", "FC2", "FC3", "value"]
    critic_layers = []
    for idx, (dim, name) in enumerate(zip(critic_dims, critic_names)):
        color = INPUT if idx == 0 else OUTPUT if idx == 4 else CRITIC
        critic_layers.append(neuron_layer(ax, 1.25 + idx * 2.55, 1.75, dim, name, color=color, span=1.75, note="ELU" if 0 < idx < 4 else None))
    for left, right in zip(critic_layers, critic_layers[1:]):
        connect_layers(ax, left, right, alpha=0.22)
    ax.text(12.95, 6.65, "checkpoint: agents/low_level_best.pt", ha="right", fontsize=8.2, color=MUTED)
    return save(fig, "fixed_low_level_locomotion.svg")


def draw_affordance_estimator() -> Path:
    fig, ax = setup("Optional Student Affordance Estimator", width=16.5, height=8.4)
    y = 4.25
    stacks = [
        feature_stack(ax, 0.70, y, "depth", "1 x 128 x 128", color=INPUT, width=0.48, height=1.42),
        feature_stack(ax, 2.05, y, "Conv7 + pool", "64 x 32 x 32", note="s2, then max-pool"),
        feature_stack(ax, 3.55, y, "ResBlock x2", "64 x 32 x 32"),
        feature_stack(ax, 5.05, y, "ResBlock x2", "128 x 16 x 16", note="first block s2"),
        feature_stack(ax, 6.55, y, "ResBlock x2", "256 x 8 x 8", note="first block s2"),
    ]
    for left, right in zip(stacks, stacks[1:]):
        connect_points(ax, left, (right[0] - 0.66, right[1]))

    rows = [
        (6.55, "occupancy"),
        (4.25, "passable gap"),
        (1.95, "low obstacle"),
    ]
    for row_y, name in rows:
        up = feature_stack(ax, 9.00, row_y, "TransConv", "128 x 16 x 16", color=HEAD, note="k4 / s2")
        refine = feature_stack(ax, 10.70, row_y, "Conv + ReLU", "64 x 16 x 16", color=HEAD, note="k3 / s1")
        out = feature_stack(ax, 12.40, row_y, name, "1 x 16 x 16", color=OUTPUT, width=0.42, height=1.15, depth=2, note="1x1 Conv + Sigmoid")
        connect_points(ax, (stacks[-1][0], stacks[-1][1]), (up[0] - 0.66, up[1]))
        connect_points(ax, up, (refine[0] - 0.66, refine[1]))
        connect_points(ax, refine, (out[0] - 0.52, out[1]))
    ax.text(14.75, 4.45, "three dense\n16 x 16 predictions", ha="center", va="center", fontsize=11, color=NAVY, fontweight="semibold")
    ax.text(14.75, 3.62, "optional student branch;\nnot used in final teacher-mode PCR tables", ha="center", va="top", fontsize=8, color=MUTED)
    return save(fig, "optional_affordance_estimator.svg")


def main() -> None:
    outputs = [
        draw_gate(
            learned_w=True,
            filename="pcr_gate_learnedw.svg",
            title="PCR Learned-w GatePolicy",
            state_dim=13,
            goal_dim=18,
            goal_label="goal + risk features",
        ),
        draw_gate(
            learned_w=False,
            filename="pcr_gate_base_variants.svg",
            title="PCR Base GatePolicy (Y-only / Risk-only shared network)",
            state_dim=13,
            goal_dim=2,
            goal_label="target goal",
        ),
        draw_cmd_policy(
            filename="avoid_expert.svg",
            title="Learned Avoid Expert (CmdVelExpert)",
            state_dim=14,
            output_note="PCR consumes the lateral expert component",
        ),
        draw_cmd_policy(
            filename="mono_ppo.svg",
            title="Monolithic PPO High-Level Policy",
            state_dim=13,
            output_note="direct command without expert arbitration",
        ),
        draw_low_level(),
        draw_affordance_estimator(),
    ]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
