#!/usr/bin/env python3
"""Draw a code-faithful PCR high-level decision flowchart."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


COLORS = {
    "input": ("#F3F4F6", "#4B5563"),
    "analytic": ("#E5E7EB", "#374151"),
    "learned": ("#DCEAFE", "#1D4ED8"),
    "learned_green": ("#DDF7E8", "#047857"),
    "fusion": ("#DBEAFE", "#075985"),
    "safety": ("#FEF3C7", "#B45309"),
    "ros": ("#FCE7F3", "#BE185D"),
    "dark": ("#111827", "#111827"),
}


def setup_font() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def wrap(text: str, width: int = 27) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def box(ax, x, y, w, h, title, body, kind="analytic", tag=None, title_size=11.5, body_size=9.0):
    fill, edge = COLORS[kind]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.55,
        facecolor=fill,
        edgecolor=edge,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.048, title, ha="left", va="top", fontsize=title_size, weight="bold", color="#111827", zorder=3)
    if body:
        ax.text(x + 0.018, y + h - 0.105, body, ha="left", va="top", fontsize=body_size, color="#1F2937", linespacing=1.18, zorder=3)
    if tag:
        tw = 0.078 if len(tag) <= 7 else 0.103
        th = 0.032
        tx = x + w - tw - 0.016
        ty = y + h - th - 0.018
        tag_patch = FancyBboxPatch(
            (tx, ty),
            tw,
            th,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.8,
            facecolor="#FFFFFF",
            edgecolor=edge,
            zorder=4,
        )
        ax.add_patch(tag_patch)
        ax.text(tx + tw / 2, ty + th / 2, tag, ha="center", va="center", fontsize=7.5, color=edge, weight="bold", zorder=5)
    return patch


def arrow(ax, start, end, *, color="#111827", lw=1.55, dashed=False, rad=0.0, label=None, label_offset=(0, 0), zorder=1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=8.0, color=color, ha="center", va="center", zorder=6)
    return patch


def tiny_grid(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#1F2937", edgecolor="#111827", linewidth=1.0, zorder=3))
    cells = [
        (0, 5, "#EF4444"), (1, 5, "#EF4444"), (2, 5, "#EF4444"),
        (5, 4, "#EF4444"), (6, 4, "#EF4444"),
        (3, 1, "#FDE047"), (4, 1, "#FDE047"), (5, 1, "#FDE047"),
        (2, 2, "#FDE047"), (3, 2, "#FDE047"), (4, 2, "#FDE047"),
        (1, 3, "#FDE047"), (2, 3, "#FDE047"),
    ]
    nx = ny = 8
    for i in range(nx + 1):
        xx = x + w * i / nx
        ax.plot([xx, xx], [y, y + h], color="#4B5563", linewidth=0.4, zorder=4)
    for j in range(ny + 1):
        yy = y + h * j / ny
        ax.plot([x, x + w], [yy, yy], color="#4B5563", linewidth=0.4, zorder=4)
    for cx, cy, c in cells:
        ax.add_patch(Rectangle((x + w * cx / nx, y + h * cy / ny), w / nx, h / ny, facecolor=c, edgecolor="none", zorder=5))
    ax.arrow(x + w * 0.5, y + h * 0.08, 0, h * 0.13, width=0.002, head_width=0.018, head_length=0.018, color="#FFFFFF", zorder=6)


def draw() -> None:
    setup_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(19.0, 10.4), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.965, "PCR-Net High-Level Decision Flow", fontsize=24, weight="bold", ha="left", va="top", color="#111827")
    ax.text(
        0.035,
        0.928,
        "Code path: real_pcr_input_check.py -> pcr_realplay.py -> /usr/command_pcr",
        fontsize=12.2,
        ha="left",
        va="top",
        color="#4B5563",
    )

    # Column captions.
    captions = [
        (0.055, "Real-robot inputs"),
        (0.300, "Mid-level experts"),
        (0.550, "PCR-Net arbitration"),
        (0.830, "Execution command"),
    ]
    for x, txt in captions:
        ax.text(x, 0.885, txt.upper(), fontsize=12, weight="bold", ha="center", va="center", color="#111827")

    # Inputs.
    box(
        ax,
        0.035,
        0.635,
        0.20,
        0.205,
        "Target state",
        "from /pcr/target_state\n"
        "goal_buf = [x_right, y_forward]\n"
        "valid, too_close, depth_invalid\n"
        "actor_difficulty",
        "input",
    )
    box(
        ax,
        0.035,
        0.385,
        0.20,
        0.205,
        "Local map",
        "from /pcr/local_map_2ch\n"
        "ch0: occupancy\n"
        "ch1: passable / safety\n"
        "32 x 32 local grid",
        "input",
    )
    tiny_grid(ax, 0.169, 0.410, 0.045, 0.070)
    box(
        ax,
        0.035,
        0.165,
        0.20,
        0.165,
        "Risk inputs",
        "risk_blocked_map\n"
        "policy_visible_map\n"
        "front_distance_risk\n"
        "short-horizon risk_memory",
        "input",
    )

    # Experts.
    box(
        ax,
        0.290,
        0.650,
        0.205,
        0.155,
        "Follow expert",
        "inputs: target state + robot state\n"
        "output: u_F = [0, y, yaw]\n"
        "forward + turning only",
        "analytic",
        tag="analytic",
    )
    box(
        ax,
        0.290,
        0.405,
        0.205,
        0.170,
        "Avoid expert",
        "inputs: local_map_2ch + state\n"
        "       + goal_buf + difficulty\n"
        "raw: [x, y, yaw]\n"
        "PCR uses u_A = [x, 0, 0]",
        "learned_green",
        tag="learned",
    )

    # Arbitration.
    box(
        ax,
        0.545,
        0.690,
        0.255,
        0.160,
        "(1) Command-risk query",
        "inputs: risk map + u_F + u_A\n"
        "outputs: risk_F, risk_A,\n"
        "clearance_F/A, cmd_cos,\n"
        "conflict features + risk_memory",
        "analytic",
        tag="analytic",
    )
    box(
        ax,
        0.545,
        0.460,
        0.255,
        0.175,
        "(2) GatePolicy learned-w",
        "inputs: local map + state\n"
        "       + gate_goal\n"
        "gate_goal = goal_buf + 16 conflict features\n"
        "outputs: y, w",
        "learned",
        tag="learned",
    )
    box(
        ax,
        0.545,
        0.218,
        0.255,
        0.190,
        "(3) Conflict-aware fusion",
        "signed_w = 2w - 1\n"
        "y_eff = clip(y + lambda deadband(signed_w)\n"
        "             + gamma(risk_A - risk_F), 0, 1)\n"
        "u_mix = y_eff u_F + (1-y_eff) u_A",
        "fusion",
        tag="analytic",
        body_size=8.0,
    )

    # Safety/output.
    box(
        ax,
        0.845,
        0.505,
        0.135,
        0.160,
        "Safety limiter",
        "target lost\n"
        "depth invalid\n"
        "target too close\n"
        "clip + rate limit",
        "safety",
        tag="fixed",
    )
    box(
        ax,
        0.845,
        0.220,
        0.135,
        0.185,
        "ROS command",
        "/usr/command_pcr\n"
        "joy_command:\n"
        "x_vec, y_vec,\n"
        "w_twist",
        "ros",
        tag="10 Hz",
    )

    # Main arrows.
    arrow(ax, (0.235, 0.740), (0.290, 0.735), label="goal", label_offset=(0.0, 0.022), zorder=6)
    arrow(ax, (0.235, 0.500), (0.290, 0.495), label="map", label_offset=(0.0, 0.020), zorder=6)
    arrow(ax, (0.235, 0.720), (0.290, 0.500), color="#047857", rad=-0.15, label="goal", label_offset=(-0.018, -0.022), zorder=6)

    # Expert commands go into a small command bus before PCR uses them in two places.
    bus = FancyBboxPatch(
        (0.508, 0.482),
        0.025,
        0.140,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.0,
        facecolor="#FFFFFF",
        edgecolor="#111827",
        zorder=7,
    )
    ax.add_patch(bus)
    ax.text(0.5205, 0.552, "u_F\nu_A", fontsize=8.0, ha="center", va="center", color="#111827", zorder=8)
    arrow(ax, (0.495, 0.735), (0.508, 0.600), label="", rad=0.05, zorder=6)
    arrow(ax, (0.495, 0.493), (0.508, 0.505), label="", zorder=6)
    arrow(ax, (0.533, 0.605), (0.545, 0.780), label="cmds", label_offset=(0.012, 0.020), zorder=6)
    arrow(ax, (0.533, 0.495), (0.545, 0.320), label="cmds", label_offset=(0.013, -0.020), zorder=6)

    arrow(ax, (0.235, 0.488), (0.545, 0.545), color="#4B5563", dashed=True, rad=0.07, label="map", label_offset=(0.000, 0.032), zorder=0)
    arrow(ax, (0.235, 0.248), (0.545, 0.772), color="#6B7280", dashed=True, rad=-0.10, label="risk maps", label_offset=(-0.045, 0.035), zorder=0)
    arrow(ax, (0.672, 0.690), (0.672, 0.635), color="#6B7280", dashed=True, label="risk / conflict", label_offset=(0.072, 0.000), zorder=6)
    arrow(ax, (0.672, 0.460), (0.672, 0.408), color="#111827", label="y, w", label_offset=(0.040, 0.000), zorder=6)

    arrow(ax, (0.800, 0.315), (0.845, 0.560), label="u_mix", label_offset=(0.035, 0.020), zorder=6)
    arrow(ax, (0.912, 0.505), (0.912, 0.405), label="u_exec", label_offset=(0.055, 0.000), zorder=6)

    # Notes.
    ax.text(
        0.035,
        0.070,
        "Legend:",
        fontsize=11.5,
        weight="bold",
        color="#111827",
        ha="left",
        va="center",
    )
    legend_items = [
        ("learned", "learned policy module"),
        ("analytic", "fixed analytic computation"),
        ("safety", "fixed deployment guard"),
        ("ros", "published real-robot command"),
    ]
    lx = 0.095
    for kind, txt in legend_items:
        fill, edge = COLORS[kind]
        ax.add_patch(Rectangle((lx, 0.056), 0.018, 0.025, facecolor=fill, edgecolor=edge, linewidth=1.1))
        ax.text(lx + 0.024, 0.0685, txt, fontsize=9.4, ha="left", va="center", color="#374151")
        lx += 0.195
    ax.text(
        0.035,
        0.030,
        "Solid arrows are command / decision flow. Dashed arrows are risk or conflict information.",
        fontsize=9.5,
        color="#4B5563",
        ha="left",
        va="center",
    )

    png_path = OUT_DIR / "pcr_highlevel_flowchart.png"
    pdf_path = OUT_DIR / "pcr_highlevel_flowchart.pdf"
    svg_path = OUT_DIR / "pcr_highlevel_flowchart.svg"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    print(svg_path)


if __name__ == "__main__":
    draw()
