#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Draw a compact quantitative-effect plot for Fig.1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import patches

from plot_vop_shortpaper_figures import PALETTE, build_paper_metrics_bundle, runtime_to_fps, setup_style, soften_spines


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = GAME4LOC_DIR / "figures" / "vop_shortpaper_20260419_review"


METHOD_SPECS = [
    ("dense DKM", "Dense DKM", PALETTE["dense"], "^", 210, 6, 8, 1.0),
    ("LoFTR", "LoFTR", PALETTE["loftr"], "D", 170, 6, 6, 0.95),
    ("sparse", "Sparse", PALETTE["sparse"], "o", 190, 6, -12, 1.0),
    ("sparse + rotate90", "Sparse+Rotate", PALETTE["rotate"], "o", 195, 6, 8, 1.0),
    ("sparse + VOP (ours)", "Ours", PALETTE["ours"], "o", 255, 8, -12, 1.0),
]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in (".png", ".pdf"):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        saved.append(path)
    return saved


def draw_badge(ax: plt.Axes, xywh: tuple[float, float, float, float], *, number: str, caption: str, edge: str, fill: str) -> None:
    x, y, w, h = xywh
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h * 0.62,
        number,
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=edge,
        transform=ax.transAxes,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.25,
        caption,
        ha="center",
        va="center",
        fontsize=11.4,
        color=PALETTE["text"],
        transform=ax.transAxes,
    )


def annotate_point(ax: plt.Axes, x: float, y: float, label: str, color: str, dx: float, dy: float, *, fontsize: float = 13.6) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        fontweight="bold" if label == "Ours" else "normal",
        path_effects=[pe.withStroke(linewidth=3.4, foreground="white", alpha=0.92)],
    )


def main() -> None:
    setup_style()
    metrics = build_paper_metrics_bundle()["GTA same-area"]

    dense = metrics["dense DKM"]
    sparse = metrics["sparse"]
    rotate = metrics["sparse + rotate90"]
    ours = metrics["sparse + VOP (ours)"]

    speedup_vs_dense = dense["mean_total_s"] / ours["mean_total_s"]
    dis_gain_vs_sparse = sparse["dis1_m"] - ours["dis1_m"]
    ma20_gain_vs_sparse = ours["ma20_pct"] - sparse["ma20_pct"]

    fig = plt.figure(figsize=(8.0, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.38, 1.0], hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    ax_top.set_axis_off()
    ax_top.text(
        0.00,
        0.98,
        "Quantitative effect on GTA-UAV same-area",
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=17.5,
        fontweight="bold",
        color=PALETTE["text"],
    )
    draw_badge(
        ax_top,
        (0.00, 0.10, 0.29, 0.62),
        number=f"{speedup_vs_dense:.1f}x",
        caption="faster than Dense",
        edge=PALETTE["callout_green"],
        fill="#EEF8F0",
    )
    draw_badge(
        ax_top,
        (0.35, 0.10, 0.29, 0.62),
        number=f"-{dis_gain_vs_sparse:.1f} m",
        caption="Dis@1 vs Sparse",
        edge=PALETTE["ours"],
        fill="#FCF0ED",
    )
    draw_badge(
        ax_top,
        (0.70, 0.10, 0.29, 0.62),
        number=f"+{ma20_gain_vs_sparse:.1f} pp",
        caption="MA@20 over Sparse",
        edge=PALETTE["rotate"],
        fill="#EEF4FB",
    )

    soften_spines(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Frames per second (FPS, log scale)", fontsize=14.8)
    ax.set_ylabel("Dis@1 (m)", fontsize=14.8)
    ax.grid(True, which="both", linestyle="-", linewidth=0.65, alpha=0.50)
    ax.tick_params(axis="both", labelsize=12.8)

    xs: list[float] = []
    ys: list[float] = []
    for method_name, label, color, marker, size, dx, dy, alpha in METHOD_SPECS:
        point = metrics[method_name]
        x = runtime_to_fps(point["mean_total_s"])
        y = point["dis1_m"]
        xs.append(x)
        ys.append(y)
        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.15 if label == "Ours" else 0.9,
            alpha=alpha,
            zorder=4 if label == "Ours" else 3,
        )
        annotate_point(ax, x, y, label, color, dx, dy, fontsize=14.6 if label == "Ours" else 13.2)

    ax.set_xlim(min(xs) * 0.72, max(xs) * 1.40)
    ax.set_ylim(max(ys) * 1.16, min(ys) * 0.80)

    ax.annotate(
        "Better",
        xy=(0.94, 0.94),
        xytext=(0.80, 0.83),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color=PALETTE["callout_green"],
            linewidth=2.2,
            mutation_scale=14,
            shrinkA=1,
            shrinkB=1,
        ),
        ha="left",
        va="center",
        fontsize=13.8,
        fontweight="bold",
        color=PALETTE["callout_green"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=5,
    )

    saved = save_figure(fig, OUTPUT_DIR, "fig01_quantitative_effect_gta_mini")
    plt.close(fig)
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
