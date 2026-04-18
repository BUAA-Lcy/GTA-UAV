#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Generate a submission-style VOP framework draft figure."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from plot_vop_shortpaper_figures import (
    PALETTE,
    load_rotated_query,
    load_square_image,
    prettify_angle,
    save_figure,
    setup_style,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
DATA_DIR = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC_DIR / "figures" / "vop_framework_draft_20260414",
        help="Output directory for the generated figure.",
    )
    return parser.parse_args()


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    fc: str = "white",
    ec: str = PALETTE["axis"],
    title_color: str = PALETTE["text"],
    body_color: str = PALETTE["axis"],
    lw: float = 1.8,
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.04 * w,
        y + h - 0.28 * h,
        title,
        ha="left",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + 0.04 * w,
        y + 0.12 * h,
        body,
        ha="left",
        va="bottom",
        fontsize=10.2,
        color=body_color,
        linespacing=1.33,
    )


def add_arrow(
    ax: plt.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = PALETTE["muted"],
    lw: float = 1.8,
    ms: float = 18.0,
) -> None:
    ax.add_patch(
        patches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_demo_posterior(angles_deg: Sequence[float]) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=float)

    def wrapped_gaussian(center: float, sigma: float, weight: float) -> np.ndarray:
        diff = np.abs(((angles - center + 180.0) % 360.0) - 180.0)
        return weight * np.exp(-(diff**2) / (2.0 * sigma**2))

    posterior = (
        wrapped_gaussian(20.0, 22.0, 1.00)
        + wrapped_gaussian(-20.0, 18.0, 0.60)
        + wrapped_gaussian(45.0, 16.0, 0.42)
        + 0.10
    )
    posterior = posterior / posterior.sum()
    return posterior


def axes_midpoint(ax: plt.Axes) -> tuple[float, float]:
    pos = ax.get_position()
    return pos.x0 + pos.width / 2.0, pos.y0 + pos.height / 2.0


def axes_union_bounds(axes: Sequence[plt.Axes]) -> tuple[float, float, float, float]:
    x0 = min(ax.get_position().x0 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    return x0, y0, x1, y1


def create_framework_draft(output_dir: Path) -> List[Path]:
    query_path = DATA_DIR / "drone" / "images" / "02_0987.JPG"
    satellite_path = DATA_DIR / "satellite" / "02_6_009_022.png"

    fig = plt.figure(figsize=(16.2, 8.9))
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[0.92, 1.48],
        width_ratios=[1.28, 1.28, 1.08, 1.18, 1.14],
        hspace=0.16,
        wspace=0.20,
    )

    fig.suptitle(
        "VOP-Guided Post-Retrieval Fine Localization Framework",
        x=0.015,
        y=0.985,
        ha="left",
        fontsize=21.5,
        fontweight="bold",
        color=PALETTE["text"],
    )
    fig.text(
        0.017,
        0.942,
        "Retrieval stays fixed. VOP only proposes useful orientations for sparse geometric verification.",
        ha="left",
        va="top",
        fontsize=12.4,
        color=PALETTE["muted"],
    )

    ax_train = fig.add_subplot(gs[0, :])
    ax_train.set_axis_off()
    ax_train.set_xlim(0, 1)
    ax_train.set_ylim(0, 1)

    ax_train.text(0.01, 0.88, "Training stage", fontsize=14.8, fontweight="bold", color=PALETTE["muted"])

    train_specs = [
        (
            0.04,
            0.43,
            0.18,
            0.38,
            "Teacher construction",
            "Fix retrieval top-1\nSweep discrete angles\nRecord localization outcome",
            "#F7F4EC",
            "#C28A2C",
        ),
        (
            0.28,
            0.43,
            0.22,
            0.38,
            "Useful-angle supervision",
            "Useful set: d(theta) <= best + delta\nPair-confidence weighting by best distance",
            "#F6F1FB",
            "#6D5EAC",
        ),
        (
            0.56,
            0.43,
            0.16,
            0.38,
            "Train VOP",
            "Frozen retrieval features\nPredict posterior over 36 angles",
            "#FBEFEB",
            PALETTE["ours"],
        ),
        (
            0.78,
            0.43,
            0.17,
            0.38,
            "Checkpoint",
            "Lightweight orientation proposer\nUsed only after retrieval",
            "#EDF5EC",
            PALETTE["success"],
        ),
    ]
    for x, y, w, h, title, body, fc, ec in train_specs:
        add_box(ax_train, x, y, w, h, title, body, fc=fc, ec=ec)

    add_arrow(ax_train, 0.22, 0.62, 0.28, 0.62)
    add_arrow(ax_train, 0.50, 0.62, 0.56, 0.62)
    add_arrow(ax_train, 0.72, 0.62, 0.78, 0.62)

    ax_train.text(
        0.04,
        0.26,
        "Current paper line: top-k useful angle hypotheses + geometry verification",
        fontsize=12.6,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax_train.text(
        0.04,
        0.16,
        "Not a brute-force sweep and not a retrieval redesign.",
        fontsize=11.1,
        color=PALETTE["muted"],
    )

    sub_pair = gs[1, 0:2].subgridspec(1, 2, wspace=0.08)
    query_img = load_square_image(query_path, size=420)
    satellite_img = load_square_image(satellite_path, size=420)

    ax_query = fig.add_subplot(sub_pair[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_axis_off()
    ax_query.set_title("1. Query UAV view", loc="left", fontsize=14.4, fontweight="bold")
    ax_query.text(
        0.03,
        0.03,
        "02_0987.JPG",
        transform=ax_query.transAxes,
        fontsize=10.0,
        color="white",
        bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.48), ec="none"),
    )

    ax_sat = fig.add_subplot(sub_pair[0, 1])
    ax_sat.imshow(satellite_img)
    ax_sat.set_axis_off()
    ax_sat.set_title("2. Retrieved top-1 tile", loc="left", fontsize=14.4, fontweight="bold")
    ax_sat.text(
        0.03,
        0.03,
        "02_6_009_022.png",
        transform=ax_sat.transAxes,
        fontsize=10.0,
        color="white",
        bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.48), ec="none"),
    )
    ax_sat.text(
        0.03,
        0.96,
        "Retrieval fixed",
        transform=ax_sat.transAxes,
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
    )

    ax_pair_note = fig.add_subplot(gs[1, 0:2], frame_on=False)
    ax_pair_note.set_axis_off()
    ax_pair_note.text(
        0.00,
        1.07,
        "Inference stage",
        transform=ax_pair_note.transAxes,
        fontsize=14.8,
        fontweight="bold",
        color=PALETTE["muted"],
    )
    ax_pair_note.text(
        0.00,
        1.00,
        "Paper7 real sample pair used only for illustration of the post-retrieval pipeline.",
        transform=ax_pair_note.transAxes,
        fontsize=11.0,
        color=PALETTE["muted"],
    )

    ax_posterior = fig.add_subplot(gs[1, 2], projection="polar")
    candidate_angles_deg = list(np.linspace(-180, 170, 36))
    posterior = build_demo_posterior(candidate_angles_deg)
    topk = 4
    topk_idx = np.argsort(posterior)[-topk:]
    selected_angles = [candidate_angles_deg[idx] for idx in topk_idx]
    selected_angles_sorted = sorted(selected_angles)
    best_angle = 20.0

    angles_rad = np.deg2rad(candidate_angles_deg)
    ax_posterior.bar(
        angles_rad,
        posterior,
        width=np.deg2rad(8.6),
        color=PALETTE["rotate"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    ax_posterior.scatter(
        np.deg2rad(selected_angles_sorted),
        [posterior[candidate_angles_deg.index(angle)] for angle in selected_angles_sorted],
        s=72,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    ax_posterior.set_theta_zero_location("N")
    ax_posterior.set_theta_direction(-1)
    ax_posterior.set_title("3. VOP posterior over useful angles", loc="left", fontsize=14.4, fontweight="bold", pad=20)
    ax_posterior.set_ylim(0.0, posterior.max() * 1.28)
    ax_posterior.set_rlabel_position(110)
    ax_posterior.tick_params(labelsize=8.6)
    ax_posterior.grid(color=PALETTE["grid"], alpha=0.82)
    ax_posterior.text(
        0.50,
        -0.12,
        "Red markers: retained top-k hypotheses",
        transform=ax_posterior.transAxes,
        ha="center",
        va="top",
        fontsize=9.8,
        color=PALETTE["muted"],
    )

    sub_hyp = gs[1, 3].subgridspec(2, 2, wspace=0.12, hspace=0.16)
    hypothesis_angles = [-20.0, 0.0, 20.0, 40.0]
    verification_text = {
        -20.0: "secondary",
        0.0: "kept",
        20.0: "best",
        40.0: "useful",
    }
    hyp_axes = []
    for idx, angle in enumerate(hypothesis_angles):
        ax = fig.add_subplot(sub_hyp[idx // 2, idx % 2])
        hyp_axes.append(ax)
        ax.imshow(load_rotated_query(query_path, angle, size=245))
        ax.set_axis_off()
        is_best = math.isclose(angle, best_angle)
        border_color = PALETTE["ours"] if is_best else PALETTE["rotate"]
        border = patches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            transform=ax.transAxes,
            fill=False,
            linewidth=2.5 if is_best else 1.8,
            edgecolor=border_color,
            clip_on=False,
        )
        ax.add_patch(border)
        ax.text(
            0.03,
            0.95,
            prettify_angle(angle),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.3,
            fontweight="bold",
            color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90),
        )
        ax.text(
            0.03,
            0.05,
            verification_text[angle],
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9.1,
            fontweight="bold" if is_best else "normal",
            color=border_color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90),
        )
        if is_best:
            ax.text(
                0.96,
                0.95,
                "selected",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=9.2,
                fontweight="bold",
                color=PALETTE["ours"],
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
            )

    hyp_left = hyp_axes[0].get_position()
    hyp_right = hyp_axes[1].get_position()
    fig.text(
        hyp_left.x0,
        max(hyp_left.y1, hyp_right.y1) + 0.020,
        "4. Verify a shortlist of top-k hypotheses",
        ha="left",
        va="bottom",
        fontsize=14.4,
        fontweight="bold",
        color=PALETTE["text"],
    )
    fig.text(
        hyp_left.x0,
        max(hyp_left.y1, hyp_right.y1) - 0.005,
        "Rotate only a few VOP-supported candidates, then let geometry choose the final one.",
        ha="left",
        va="bottom",
        fontsize=10.6,
        color=PALETTE["muted"],
    )

    ax_final = fig.add_subplot(gs[1, 4])
    ax_final.imshow(satellite_img)
    ax_final.set_axis_off()
    ax_final.set_title("5. Geometry selection and final localization", loc="left", fontsize=14.4, fontweight="bold")
    ax_final.scatter(
        [satellite_img.shape[1] * 0.53],
        [satellite_img.shape[0] * 0.47],
        s=170,
        color=PALETTE["ours"],
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
    )
    ax_final.text(
        0.03,
        0.95,
        "best angle: +20°",
        transform=ax_final.transAxes,
        va="top",
        ha="left",
        fontsize=10.6,
        fontweight="bold",
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
    )
    ax_final.text(
        0.03,
        0.04,
        "Sparse matching\nHomography verification\nReturn final pose on the tile",
        transform=ax_final.transAxes,
        va="bottom",
        ha="left",
        fontsize=10.1,
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.93),
    )

    pair_box = ax_sat.get_position()
    posterior_box = ax_posterior.get_position()
    hyp_box = axes_union_bounds(hyp_axes)
    final_box = ax_final.get_position()
    arrow_y = 0.40
    for (x0, y0), (x1, y1) in (
        ((pair_box.x1 + 0.006, arrow_y), (posterior_box.x0 - 0.008, arrow_y)),
        ((posterior_box.x1 + 0.008, arrow_y), (hyp_box[0] - 0.008, arrow_y)),
        ((hyp_box[2] + 0.008, arrow_y), (final_box.x0 - 0.008, arrow_y)),
    ):
        fig.add_artist(
            patches.FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.6,
                color=PALETTE["muted"],
            )
        )

    posterior_mid = axes_midpoint(ax_posterior)
    checkpoint_center_x = 0.815
    checkpoint_bottom_y = 0.565
    fig.add_artist(
        patches.FancyArrowPatch(
            (checkpoint_center_x, checkpoint_bottom_y),
            (posterior_mid[0] + 0.01, 0.602),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=1.8,
            color=PALETTE["ours"],
            connectionstyle="arc3,rad=0.16",
        )
    )

    fig.text(
        0.018,
        0.012,
        "Example images are from UAV-VisLoc same-area-paper7. Posterior and shortlisted hypotheses are illustrative for the framework draft.",
        fontsize=10.0,
        color=PALETTE["muted"],
    )

    return save_figure(fig, output_dir, "fig_vop_framework_submission_draft")


def main() -> None:
    args = parse_args()
    setup_style()
    saved = create_framework_draft(args.output_dir)
    print("Generated files:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
