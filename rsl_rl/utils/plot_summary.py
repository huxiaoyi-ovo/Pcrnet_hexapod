import os


def generate_training_summary(log_dir, tags, output_path=None, title=None):
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PlotSummary] Skipped (missing deps): {exc}")
        return None

    if not log_dir or not os.path.isdir(log_dir):
        print(f"[PlotSummary] log_dir not found: {log_dir}")
        return None

    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
    except Exception as exc:
        print(f"[PlotSummary] Failed to load events: {exc}")
        return None

    scalar_tags = set(ea.Tags().get("scalars", []))

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()

    for idx, tag_item in enumerate(tags):
        ax = axes[idx]
        if isinstance(tag_item, (list, tuple)):
            tag, label = tag_item
        else:
            tag, label = tag_item, tag_item

        if tag in scalar_tags:
            scalars = ea.Scalars(tag)
            if scalars:
                steps = [x.step for x in scalars]
                values = [x.value for x in scalars]
                ax.plot(steps, values, linewidth=1.2)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")

        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(tags):]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if output_path is None:
        output_path = os.path.join(log_dir, "training_summary.png")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[PlotSummary] Saved: {output_path}")
    return output_path
