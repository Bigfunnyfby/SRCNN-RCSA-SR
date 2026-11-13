# -*- coding: utf-8 -*-
"""Draw the RCSA-SR architecture diagram (schematic, not a result figure)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")


def box(ax, x, y, w, h, text, fc="#e8f1fb", ec="#2b6cb0", fs=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", lw=1.2, connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, linewidth=lw, color=color,
                                 connectionstyle=connectionstyle))


def main():
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # main pipeline
    box(ax, 0.20, 2.75, 1.45, 0.85, "LR input\n(bicubic ↑)", fc="#f5f5f5", ec="#666666")
    box(ax, 2.00, 2.75, 1.55, 0.85, "Head Conv\n5×5, 64")
    box(ax, 4.00, 2.55, 3.55, 1.25, "4 × Residual Block\n(64 channels)", fc="#fff4e6", ec="#c05621", fs=10.5)
    box(ax, 8.00, 2.75, 1.55, 0.85, "Tail Conv\n5×5, 1")
    ax.add_patch(plt.Circle((10.15, 3.175), 0.23, fill=False, linewidth=1.4, color="#333333"))
    ax.text(10.15, 3.175, "+", ha="center", va="center", fontsize=16)
    box(ax, 10.95, 2.75, 1.55, 0.85, "HR output", fc="#f5f5f5", ec="#666666")

    arrow(ax, 1.65, 3.175, 1.98, 3.175)
    arrow(ax, 3.55, 3.175, 3.98, 3.175)
    arrow(ax, 7.55, 3.175, 7.98, 3.175)
    arrow(ax, 9.55, 3.175, 9.90, 3.175)
    arrow(ax, 10.38, 3.175, 10.93, 3.175)
    # global skip
    ax.plot([0.92, 0.92, 10.15], [2.73, 2.22, 2.22], color="#2b6cb0", linewidth=1.6)
    arrow(ax, 10.15, 2.22, 10.15, 2.93, color="#2b6cb0", lw=1.6)
    ax.text(5.4, 2.10, "global residual connection", color="#2b6cb0", fontsize=9, ha="center")

    # expanded residual block
    ax.text(0.25, 1.48, "Residual block detail:", fontsize=10, fontweight="bold")
    y = 0.62
    box(ax, 0.30, y, 1.35, 0.62, "Conv 3×3\n+ ReLU", fs=8.5)
    box(ax, 2.00, y, 1.35, 0.62, "Conv 3×3", fs=8.5)
    box(ax, 3.70, y, 1.35, 0.62, "SE block\n(64 → 8 → 64)", fs=8.0, fc="#e6fffa", ec="#2c7a7b")
    box(ax, 5.40, y, 1.35, 0.62, "× 0.5", fs=8.5, fc="#faf5ff", ec="#6b46c1")
    ax.add_patch(plt.Circle((7.20, y + 0.31), 0.22, fill=False, linewidth=1.4, color="#333333"))
    ax.text(7.20, y + 0.31, "+", ha="center", va="center", fontsize=15)
    box(ax, 7.75, y, 1.35, 0.62, "output", fs=8.5, fc="#f5f5f5", ec="#666666")
    for x1, x2 in [(1.65, 1.98), (3.35, 3.68), (5.05, 5.38), (6.75, 6.96), (7.42, 7.73)]:
        arrow(ax, x1, y + 0.31, x2, y + 0.31)
    arrow(ax, 8.42, y - 0.02, 7.20, y - 0.02, color="#6b46c1",
          connectionstyle="arc3,rad=-0.35", lw=1.2)
    ax.text(7.85, y - 0.36, "identity", color="#6b46c1", fontsize=8.5, ha="center")

    fig.tight_layout()
    out = os.path.join(FIG, "fig_architecture.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
