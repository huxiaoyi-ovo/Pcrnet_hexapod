#!/usr/bin/env python3
"""Draw the PCR w mechanism figure from completed eval outputs."""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np


METHOD_STYLE = {
    "yonly": {
        "color": "#4C6272",
        "marker": "o",
        "label": "MoE-y",
    },
    "geomw": {
        "color": "#D95F02",
        "marker": "s",
        "label": "MoE-y + geom-w",
    },
    "learnedw": {
        "color": "#7570B3",
        "marker": "^",
        "label": "MoE-y + learned-w",
    },
}


OUTCOME_FIELDS = [
    ("row_progress_success_mean", "Row-progress score", "#2A9D8F"),
    ("episode_collision_rate", "Collision rate", "#D95F02"),
]


def _load_metrics(eval_dir: str) -> Dict:
    path = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    bins = metrics.get("risk_bins", None)
    if not isinstance(bins, list) or len(bins) == 0:
        raise ValueError(
            f"{path} has no risk_bins. Re-run eval_highlevel.py with the latest code "
            "before drawing the publication mechanism figure."
        )
    _validate_outcome_fields(metrics, path)
    return metrics


def _validate_outcome_fields(metrics: Dict, path: str) -> None:
    overall = metrics.get("overall", {})
    missing = [key for key, _, _ in OUTCOME_FIELDS if key not in overall]
    if missing:
        raise ValueError(
            f"{path} misses row-progress/collision fields {missing}. "
            "Re-run eval_highlevel.py after the row-progress score fix."
        )


def _finite_float(v, default=float("nan")) -> float:
    try:
        out = float(v)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _bin_labels(metrics: Dict, bin_key: str = "risk_bins") -> List[str]:
    labels = []
    for item in metrics[bin_key]:
        label = str(item.get("bin", ""))
        if not label:
            low = _finite_float(item.get("low", float("nan")))
            high = _finite_float(item.get("high", float("nan")))
            label = f"{low:.2f}-{high:.2f}" if math.isfinite(low) and math.isfinite(high) else ""
        labels.append(label)
    return labels


def _series(
    metrics: Dict,
    key: str,
    sem_key: str,
    min_steps: int,
    bin_key: str = "risk_bins",
) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    sems = []
    for item in metrics.get(bin_key, []):
        steps = int(_finite_float(item.get("steps", 0), 0.0))
        if steps < min_steps:
            values.append(float("nan"))
            sems.append(float("nan"))
            continue
        values.append(_finite_float(item.get(key, float("nan"))))
        sems.append(_finite_float(item.get(sem_key, float("nan"))))
    return np.asarray(values, dtype=np.float64), np.asarray(sems, dtype=np.float64)


def _overall_rate(metrics: Dict, key: str) -> float:
    return _finite_float(metrics.get("overall", {}).get(key, float("nan")))


def _write_plot_data(
    out_dir: str,
    labels: List[str],
    methods: List[Tuple[str, str, Dict]],
    min_steps: int,
) -> None:
    rows = []
    for method_key, metrics, method_name in methods:
        for label, item in zip(labels, metrics["risk_bins"]):
            steps = int(_finite_float(item.get("steps", 0), 0.0))
            if steps < min_steps:
                continue
            rows.append({
                "method": method_name,
                "method_key": method_key,
                "risk_bin": label,
                "steps": steps,
                "episode_count": int(_finite_float(item.get("episode_count", 0), 0.0)),
                "gate_y_raw_mean": _finite_float(item.get("gate_y_raw_mean", float("nan"))),
                "y_eff_mean": _finite_float(item.get("y_eff_mean", float("nan"))),
                "suppression_mean": _finite_float(item.get("suppression_mean", float("nan"))),
                "w_mean": _finite_float(item.get("w_mean", float("nan"))),
                "row_progress_score": _finite_float(item.get("success_episode_rate", float("nan"))),
                "success_episode_rate": _finite_float(item.get("success_episode_rate", float("nan"))),
                "success_event_episode_rate": _finite_float(item.get("success_event_episode_rate", float("nan"))),
                "collision_episode_rate": _finite_float(item.get("collision_episode_rate", float("nan"))),
            })
    out_path = os.path.join(out_dir, "pcr_w_mechanism_plot_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _safe_tag(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "run"


def _method_tag(method_key: str, metrics: Dict, label: str) -> str:
    protocol = metrics.get("resolved_protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}
    runtime_args = protocol.get("runtime_args", {})
    if not isinstance(runtime_args, dict):
        runtime_args = {}

    def _protocol_value(key: str, default=None):
        if key in protocol:
            return protocol.get(key, default)
        return runtime_args.get(key, default)

    w_mode = str(_protocol_value("w_mode", "") or "")
    if w_mode == "none" or method_key == "yonly":
        return "yonly"
    if w_mode == "geom" or method_key == "geomw":
        return f"geomw_w{float(_protocol_value('w_tau', 0.25)):g}"
    if w_mode == "learned" or method_key == "learnedw":
        parts = ["learnedw"]
        if bool(_protocol_value("pcr_w_aux_enable", False)):
            parts.append(f"rowrel_aux{float(_protocol_value('pcr_w_aux_coef', 0.0)):g}")
        return "_".join(parts)
    return _safe_tag(label).lower()


def draw_figure(args) -> Tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yonly = _load_metrics(args.yonly_dir)
    geomw = _load_metrics(args.geomw_dir)
    learnedw = _load_metrics(args.learnedw_dir) if args.learnedw_dir else None

    labels = _bin_labels(yonly)
    geomw_labels = _bin_labels(geomw)
    if labels != geomw_labels:
        raise ValueError(f"risk bin mismatch: y-only={labels}, geom-w={geomw_labels}")
    if learnedw is not None:
        learnedw_labels = _bin_labels(learnedw)
        if labels != learnedw_labels:
            raise ValueError(f"risk bin mismatch: y-only={labels}, learned-w={learnedw_labels}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    x = np.arange(len(labels), dtype=np.float64)
    if learnedw is None:
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
        axes = np.asarray(axes).reshape(1, 3)
        ax_y, ax_supp, ax_outcome = axes[0, 0], axes[0, 1], axes[0, 2]
        ax_conflict = None
    else:
        fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.2), constrained_layout=True)
        ax_y, ax_supp, ax_conflict, ax_outcome = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    methods = [
        ("yonly", yonly, args.yonly_label or METHOD_STYLE["yonly"]["label"]),
        ("geomw", geomw, args.geomw_label or METHOD_STYLE["geomw"]["label"]),
    ]
    if learnedw is not None:
        methods.append(("learnedw", learnedw, args.learnedw_label or METHOD_STYLE["learnedw"]["label"]))

    out_dir = args.out_dir
    if bool(getattr(args, "auto_subdir", True)):
        tags = [_method_tag(method_key, metrics, label) for method_key, metrics, label in methods]
        out_dir = os.path.join(args.out_dir, "_vs_".join(tags) + "_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    for method_key, metrics, label in methods:
        style = METHOD_STYLE[method_key]
        y_eff, y_eff_sem = _series(metrics, "y_eff_mean", "y_eff_sem", args.min_steps)
        suppression, suppression_sem = _series(metrics, "suppression_mean", "suppression_sem", args.min_steps)

        ax_y.errorbar(
            x,
            y_eff,
            yerr=y_eff_sem,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.0,
            capsize=2.5,
        )
        ax_supp.errorbar(
            x,
            suppression,
            yerr=suppression_sem,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.0,
            capsize=2.5,
        )

    ax_y.set_title("A. Executed follow weight")
    ax_y.set_ylabel(r"$y_{\mathrm{eff}}$")
    ax_y.set_xlabel(r"Follow-command risk $r_F$")
    ax_y.set_ylim(args.ymin, args.ymax)

    ax_supp.set_title("B. Follow suppression")
    ax_supp.set_ylabel(r"$\Delta y = y_{\mathrm{raw}} - y_{\mathrm{eff}}$")
    ax_supp.set_xlabel(r"Follow-command risk $r_F$")
    ax_supp.set_ylim(args.suppression_ymin, args.suppression_ymax)

    if ax_conflict is not None:
        conflict_labels = _bin_labels(yonly, "conflict_bins")
        for _, metrics, _ in methods:
            if _bin_labels(metrics, "conflict_bins") != conflict_labels:
                raise ValueError("conflict bin mismatch across methods")
        conflict_x = np.arange(len(conflict_labels), dtype=np.float64)
        for method_key, metrics, label in methods:
            style = METHOD_STYLE[method_key]
            w_mean, w_sem = _series(metrics, "w_mean", "w_sem", args.min_steps, bin_key="conflict_bins")
            ax_conflict.errorbar(
                conflict_x,
                w_mean,
                yerr=w_sem,
                label=label,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5.0,
                capsize=2.5,
            )
        ax_conflict.set_title("C. Conflict-conditioned weight")
        ax_conflict.set_ylabel(r"$w$")
        ax_conflict.set_xlabel(r"Command conflict score")
        ax_conflict.set_ylim(args.w_ymin, args.w_ymax)
        ax_conflict.set_xticks(conflict_x)
        ax_conflict.set_xticklabels(conflict_labels, rotation=25, ha="right")

    outcome_methods = [
        (args.yonly_label or METHOD_STYLE["yonly"]["label"], yonly),
        (args.geomw_label or METHOD_STYLE["geomw"]["label"], geomw),
    ]
    if learnedw is not None:
        outcome_methods.append((args.learnedw_label or METHOD_STYLE["learnedw"]["label"], learnedw))
    outcome_x = np.arange(len(outcome_methods), dtype=np.float64)
    bar_width = 0.34
    offsets = (
        np.arange(len(OUTCOME_FIELDS), dtype=np.float64) - (len(OUTCOME_FIELDS) - 1.0) / 2.0
    ) * bar_width
    for field_idx, (key, label, color) in enumerate(OUTCOME_FIELDS):
        values = np.asarray([_overall_rate(metrics, key) for _, metrics in outcome_methods], dtype=np.float64)
        ax_outcome.bar(
            outcome_x + offsets[field_idx],
            values,
            width=bar_width,
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            label=label,
        )
        for xi, value in zip(outcome_x + offsets[field_idx], values):
            if math.isfinite(value) and value >= args.outcome_label_min:
                ax_outcome.text(
                    xi,
                    min(0.98, value + 0.025),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#202020",
                )

    ax_outcome.set_title("D. Row-progress task score" if learnedw is not None else "C. Row-progress task score")
    ax_outcome.set_ylabel("Episode-level score / rate")
    ax_outcome.set_xticks(outcome_x)
    ax_outcome.set_xticklabels([name for name, _ in outcome_methods], rotation=18, ha="right")
    ax_outcome.set_ylim(0.0, 1.0)

    line_axes = [ax_y, ax_supp] + ([ax_conflict] if ax_conflict is not None else [])
    for ax in line_axes:
        if ax is ax_conflict:
            ax.set_xticks(conflict_x)
            ax.set_xticklabels(conflict_labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False, loc="best")

    ax_outcome.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.8)
    ax_outcome.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    if args.title:
        fig.suptitle(args.title, fontsize=11, y=1.03)

    png_path = os.path.join(out_dir, f"{args.name}.png")
    pdf_path = os.path.join(out_dir, f"{args.name}.pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    _write_plot_data(out_dir, labels, methods, args.min_steps)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description="Draw PCR w mechanism figure from eval outputs.")
    parser.add_argument("--yonly_dir", required=True, help="Eval output dir for MoE-y.")
    parser.add_argument("--geomw_dir", required=True, help="Eval output dir for MoE-y + geom-w.")
    parser.add_argument("--learnedw_dir", default=None, help="Optional eval output dir for MoE-y + learned-w.")
    parser.add_argument("--out_dir", default="figures", help="Output directory for paper figures.")
    parser.add_argument("--no_auto_subdir", dest="auto_subdir", action="store_false", help="Write directly into out_dir instead of creating a tagged timestamp subfolder.")
    parser.set_defaults(auto_subdir=True)
    parser.add_argument("--name", default="pcr_w_mechanism", help="Output file stem.")
    parser.add_argument("--yonly_label", default="MoE-y", help="Legend label for y-only.")
    parser.add_argument("--geomw_label", default="MoE-y + geom-w", help="Legend label for geom-w.")
    parser.add_argument("--learnedw_label", default="MoE-y + learned-w", help="Legend label for learned-w.")
    parser.add_argument("--min_steps", type=int, default=50, help="Hide risk bins with fewer steps.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    parser.add_argument("--suppression_ymin", type=float, default=0.0)
    parser.add_argument("--suppression_ymax", type=float, default=0.45)
    parser.add_argument("--w_ymin", type=float, default=0.0)
    parser.add_argument("--w_ymax", type=float, default=1.0)
    parser.add_argument("--rate_ymax", type=float, default=0.5)
    parser.add_argument("--outcome_label_min", type=float, default=0.055, help="Hide tiny stacked-bar labels below this rate.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    return parser.parse_args()


def main():
    args = parse_args()
    png_path, pdf_path = draw_figure(args)
    print(f"[PCR w figure] PNG: {png_path}")
    print(f"[PCR w figure] PDF: {pdf_path}")


if __name__ == "__main__":
    main()
