#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Generate pair-specific VOP visualization assets for a Paper7 UAV/satellite pair."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
if str(GAME4LOC_DIR) not in sys.path:
    sys.path.insert(0, str(GAME4LOC_DIR))

from game4loc.dataset.visloc import get_transforms
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import load_vop_checkpoint
from game4loc.orientation.vop import rotate_feature_map
from plot_vop_shortpaper_figures import (
    PALETTE,
    load_rotated_query,
    load_square_image,
    prettify_angle,
    save_figure,
    setup_style,
)

DATA_ROOT = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset"
DEFAULT_RETRIEVAL_CKPT = (
    GAME4LOC_DIR
    / "work_dir"
    / "visloc"
    / "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    / "0409152642"
    / "weights_e10_0.6527.pth"
)
DEFAULT_VOP_CKPT = (
    GAME4LOC_DIR
    / "work_dir"
    / "paper7_main_table_runs"
    / "visloc_paper7_20260411"
    / "artifacts"
    / "vop_samearea_paper7_useful5_weight30_e6.pth"
)
DEFAULT_META_JSON = DATA_ROOT / "same-area-paper7-drone2sate-test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-name", type=str, default="01_0409.JPG")
    parser.add_argument("--gallery-name", type=str, default="01_5_005_022.png")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--retrieval-checkpoint", type=Path, default=DEFAULT_RETRIEVAL_CKPT)
    parser.add_argument("--vop-checkpoint", type=Path, default=DEFAULT_VOP_CKPT)
    parser.add_argument("--pairs-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC_DIR / "figures" / "pair_vop_assets_20260414",
    )
    return parser.parse_args()


def rotate_query_tensor(query_tensor: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(float(angle_deg)) < 1e-6:
        return query_tensor
    fill_value = -1.0
    if query_tensor.ndim == 3:
        return TF.rotate(
            query_tensor,
            angle=float(angle_deg),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=[fill_value] * int(query_tensor.shape[0]),
        )
    if query_tensor.ndim == 4:
        rotated = [
            TF.rotate(
                sample,
                angle=float(angle_deg),
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=[fill_value] * int(sample.shape[0]),
            )
            for sample in query_tensor
        ]
        return torch.stack(rotated, dim=0)
    raise ValueError(f"Unsupported query tensor shape: {tuple(query_tensor.shape)}")


def load_pair_metadata(meta_json: Path, query_name: str, gallery_name: str) -> Dict[str, object]:
    records = json.loads(meta_json.read_text(encoding="utf-8"))
    for item in records:
        if item.get("drone_img_name") != query_name:
            continue
        if gallery_name in item.get("pair_pos_sate_img_list", []):
            return item
    raise ValueError(f"Could not find pair metadata for query={query_name} gallery={gallery_name} in {meta_json}")


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_model_and_tensors(
    query_path: Path,
    gallery_path: Path,
    retrieval_checkpoint: Path,
    device: str,
) -> Dict[str, object]:
    model = DesModel(
        "vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
        pretrained=False,
        img_size=384,
        share_weights=True,
    )
    model.load_state_dict(torch.load(retrieval_checkpoint, map_location="cpu"), strict=False)
    model = model.to(device).eval()

    data_config = model.get_config()
    val_transforms, _, _ = get_transforms((384, 384), mean=data_config["mean"], std=data_config["std"])

    query_rgb = load_rgb_image(query_path)
    gallery_rgb = load_rgb_image(gallery_path)
    query_tensor = val_transforms(image=query_rgb)["image"]
    gallery_tensor = val_transforms(image=gallery_rgb)["image"]

    return {
        "model": model,
        "query_rgb": query_rgb,
        "gallery_rgb": gallery_rgb,
        "query_tensor": query_tensor,
        "gallery_tensor": gallery_tensor,
    }


def compute_similarity_map(gallery_map: torch.Tensor, query_map: torch.Tensor, angle_deg: float) -> np.ndarray:
    query_rot = rotate_feature_map(query_map, float(angle_deg))
    sim = (gallery_map[0] * query_rot[0]).sum(dim=0).detach().cpu().numpy()
    sim = sim.astype(np.float32)
    sim -= float(sim.min())
    denom = float(sim.max())
    if denom > 1e-6:
        sim /= denom
    return sim


def similarity_overlay(base_rgb: np.ndarray, similarity_map: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heatmap = cv2.resize(similarity_map, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.clip(255.0 * heatmap, 0, 255).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_rgb.astype(np.float32) + (1.0 - alpha) * base_rgb.astype(np.float32)).clip(0, 255)
    return overlay.astype(np.uint8)


def project_center_pixel(H: np.ndarray | None, image_shape: Sequence[int]) -> tuple[float, float] | None:
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    center = np.array([width / 2.0, height / 2.0, 1.0], dtype=np.float64).reshape(3, 1)
    proj = H @ center
    denom = float(proj[2, 0])
    if (not np.isfinite(denom)) or abs(denom) < 1e-6:
        return None
    x = float(proj[0, 0] / denom)
    y = float(proj[1, 0] / denom)
    if (not np.isfinite(x)) or (not np.isfinite(y)):
        return None
    return x, y


def pick_best_geometry_candidate(candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return max(
        candidates,
        key=lambda item: (
            int(item.get("inliers", 0)),
            float(item.get("inlier_ratio", 0.0)),
        ),
    )


def run_relaxed_sparse_verification(
    gallery_tensor: torch.Tensor,
    query_tensor: torch.Tensor,
    candidate_angles: Sequence[float],
    output_dir: Path,
    device: str,
) -> List[Dict[str, object]]:
    vis_dir = output_dir / "match_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Visualization-only relaxed sparse setting: more keypoints and upsampled scales
    # to expose candidate differences on a single illustrative sample.
    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        sparse_allow_upsample=True,
        sparse_scales=(0.8, 1.0, 1.6, 2.4),
        sparse_sp_detection_threshold=0.0001,
        sparse_sp_max_num_keypoints=8192,
        sparse_min_inliers=0,
        sparse_min_inlier_ratio=0.0,
        sparse_save_final_vis=True,
        sparse_save_final_vis_dir=str(vis_dir),
        sparse_save_final_vis_max=64,
    )

    candidate_records: List[Dict[str, object]] = []
    for rank, angle_deg in enumerate(candidate_angles, start=1):
        rotated_query = rotate_query_tensor(query_tensor, float(angle_deg))
        _ = matcher.est_center(
            gallery_tensor,
            rotated_query,
            center_xy0=(0.0, 0.0),
            tl_xy0=(0.0, 0.0),
            rotate=0.0,
            case_name=f"rank{rank}_{angle_deg:+.1f}",
            save_final_vis=True,
        )
        match_info = matcher.get_last_match_info() or {}
        angle_results = matcher.get_last_angle_results() or []
        selected_result = angle_results[0] if angle_results else {}
        homography = selected_result.get("homography")
        projected_xy = project_center_pixel(homography, gallery_tensor.shape[-2:])
        candidate_records.append(
            {
                "rank": int(rank),
                "angle_deg": float(angle_deg),
                "inliers": int(match_info.get("inliers", 0)),
                "n_kept": int(match_info.get("n_kept", 0)),
                "inlier_ratio": float(match_info.get("inlier_ratio", 0.0)),
                "final_vis_path": match_info.get("final_vis_path"),
                "homography": None if homography is None else np.asarray(homography, dtype=np.float64),
                "projected_xy": projected_xy,
            }
        )
    return candidate_records


def add_candidate_card(
    ax: plt.Axes,
    query_path: Path,
    angle_deg: float,
    prob: float,
    inliers: int,
    *,
    is_geom_selected: bool = False,
    is_vop_top1: bool = False,
) -> None:
    ax.imshow(load_rotated_query(query_path, float(angle_deg), size=250))
    ax.set_axis_off()
    border_color = PALETTE["ours"] if is_geom_selected else PALETTE["rotate"]
    border = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, linewidth=2.4 if is_geom_selected else 1.6, edgecolor=border_color, clip_on=False)
    ax.add_patch(border)
    ax.text(
        0.03,
        0.95,
        prettify_angle(float(angle_deg)),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90),
    )
    ax.text(
        0.03,
        0.05,
        f"p={prob:.3f}\ninliers={inliers}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=border_color,
        fontweight="bold" if is_geom_selected else "normal",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.90),
    )
    if is_vop_top1:
        ax.text(
            0.97,
            0.95,
            "VOP top-1",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.9,
            fontweight="bold",
            color=PALETTE["rotate"],
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.90),
        )
    if is_geom_selected:
        ax.text(
            0.97,
            0.05,
            "geometry selected",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.9,
            fontweight="bold",
            color=PALETTE["ours"],
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.90),
        )


def draw_projected_point(image_rgb: np.ndarray, projected_xy: tuple[float, float] | None, title: str) -> np.ndarray:
    image = image_rgb.copy()
    if projected_xy is not None:
        x, y = projected_xy
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(round(x)), int(round(y))), 10, (217, 80, 62), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, (int(round(x)), int(round(y))), 10, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    return image


def create_feature_comparison_figure(
    output_dir: Path,
    query_rgb: np.ndarray,
    gallery_rgb: np.ndarray,
    sim_top1_overlay: np.ndarray,
    sim_best_overlay: np.ndarray,
    top1_angle: float,
    best_angle: float,
) -> List[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.2), constrained_layout=True)
    panels = [
        (query_rgb, "Query image"),
        (gallery_rgb, "Retrieved satellite tile"),
        (sim_top1_overlay, f"Feature similarity @ {prettify_angle(top1_angle)}"),
        (sim_best_overlay, f"Feature similarity @ {prettify_angle(best_angle)}"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12.4, fontweight="bold")
        ax.set_axis_off()
    fig.suptitle(
        "Feature rotation and cross-view comparison on the selected Paper7 pair",
        x=0.01,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        -0.02,
        "The two right panels use retrieval feature maps encoded by VOP. Brighter regions indicate stronger local agreement after feature rotation.",
        fontsize=10.0,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "pair01_feature_rotation_comparison")


def create_posterior_figure(
    output_dir: Path,
    query_path: Path,
    candidate_angles: Sequence[float],
    candidate_probs: Sequence[float],
    candidate_records: Sequence[Dict[str, object]],
    top1_angle: float,
    best_angle: float,
) -> List[Path]:
    fig = plt.figure(figsize=(13.8, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.18)

    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    all_angles = np.deg2rad(candidate_angles)
    all_probs = np.asarray(candidate_probs, dtype=float)
    ax_polar.bar(
        all_angles,
        all_probs,
        width=np.deg2rad(8.6),
        color=PALETTE["rotate"],
        alpha=0.74,
        edgecolor="white",
        linewidth=0.35,
    )
    topk_angles = [float(item["angle_deg"]) for item in candidate_records]
    topk_probs = [float(candidate_probs[list(candidate_angles).index(float(item["angle_deg"]))]) for item in candidate_records]
    ax_polar.scatter(
        np.deg2rad(topk_angles),
        topk_probs,
        s=70,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("Actual VOP posterior", loc="left", fontsize=13.6, fontweight="bold", pad=18)
    ax_polar.grid(color=PALETTE["grid"], alpha=0.82)
    ax_polar.tick_params(labelsize=8.5)

    subgrid = gs[0, 1].subgridspec(2, 2, wspace=0.12, hspace=0.16)
    best_angle_norm = float(best_angle)
    top1_angle_norm = float(top1_angle)
    for idx, item in enumerate(candidate_records):
        ax = fig.add_subplot(subgrid[idx // 2, idx % 2])
        angle_deg = float(item["angle_deg"])
        prob = float(candidate_probs[list(candidate_angles).index(angle_deg)])
        add_candidate_card(
            ax,
            query_path=query_path,
            angle_deg=angle_deg,
            prob=prob,
            inliers=int(item["inliers"]),
            is_geom_selected=abs(angle_deg - best_angle_norm) < 1e-6,
            is_vop_top1=abs(angle_deg - top1_angle_norm) < 1e-6,
        )

    fig.suptitle(
        "Posterior-to-top-k verification: VOP proposes, geometry selects",
        x=0.01,
        ha="left",
        fontsize=16.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.02,
        "This pair is useful for the framework figure because VOP top-1 and geometry-selected angle are different.",
        fontsize=10.0,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "pair01_vop_posterior_topk")


def create_pipeline_step3_posterior_only(
    output_dir: Path,
    candidate_angles: Sequence[float],
    candidate_probs: Sequence[float],
    candidate_records: Sequence[Dict[str, object]],
) -> List[Path]:
    fig = plt.figure(figsize=(5.6, 4.9))
    ax_polar = fig.add_subplot(111, projection="polar")

    all_angles = np.deg2rad(candidate_angles)
    all_probs = np.asarray(candidate_probs, dtype=float)
    bars = ax_polar.bar(
        all_angles,
        all_probs,
        width=np.deg2rad(8.6),
        color=PALETTE["rotate"],
        alpha=0.74,
        edgecolor="white",
        linewidth=0.35,
    )

    topk_angles = [float(item["angle_deg"]) for item in candidate_records]
    selected_indices = [list(candidate_angles).index(angle) for angle in topk_angles]
    for idx in selected_indices:
        bars[idx].set_facecolor(PALETTE["ours"])
        bars[idx].set_alpha(0.90)
        bars[idx].set_edgecolor("white")
        bars[idx].set_linewidth(0.45)

    uniform = np.full_like(all_probs, 1.0 / len(all_probs))
    ax_polar.plot(all_angles, uniform, color=PALETTE["muted"], linewidth=1.2, alpha=0.65, zorder=3)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("3. VOP posterior over angles", loc="left", fontsize=13.8, fontweight="bold", pad=18)
    ax_polar.set_rlabel_position(112)
    ax_polar.set_ylim(0.0, all_probs.max() * 1.2)
    ax_polar.tick_params(labelsize=8.8)
    ax_polar.grid(color=PALETTE["grid"], alpha=0.82)
    ax_polar.text(
        0.5,
        -0.10,
        "Highlighted bars = selected top-k hypotheses",
        transform=ax_polar.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "pair01_vop_step3_posterior_only")


def create_pipeline_summary_figure(
    output_dir: Path,
    query_path: Path,
    gallery_path: Path,
    gallery_rgb: np.ndarray,
    candidate_angles: Sequence[float],
    candidate_probs: Sequence[float],
    candidate_records: Sequence[Dict[str, object]],
    top1_angle: float,
    best_angle: float,
    final_projected_xy: tuple[float, float] | None,
    selected_match_vis_path: str | None,
) -> List[Path]:
    fig = plt.figure(figsize=(15.8, 6.1))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.02, 1.22, 1.1], wspace=0.16)

    query_img = load_square_image(query_path, size=400)
    gallery_img = load_square_image(gallery_path, size=400)

    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_axis_off()
    ax_query.set_title("1. Query UAV view", loc="left", fontsize=13.6, fontweight="bold")
    ax_query.text(
        0.03,
        0.03,
        query_path.name,
        transform=ax_query.transAxes,
        fontsize=9.7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.18", fc=(0, 0, 0, 0.48), ec="none"),
    )

    ax_gallery = fig.add_subplot(gs[0, 1])
    ax_gallery.imshow(gallery_img)
    ax_gallery.set_axis_off()
    ax_gallery.set_title("2. Retrieved top-1 tile", loc="left", fontsize=13.6, fontweight="bold")
    ax_gallery.text(
        0.03,
        0.03,
        gallery_path.name,
        transform=ax_gallery.transAxes,
        fontsize=9.7,
        color="white",
        bbox=dict(boxstyle="round,pad=0.18", fc=(0, 0, 0, 0.48), ec="none"),
    )

    ax_polar = fig.add_subplot(gs[0, 2], projection="polar")
    all_angles = np.deg2rad(candidate_angles)
    all_probs = np.asarray(candidate_probs, dtype=float)
    ax_polar.bar(
        all_angles,
        all_probs,
        width=np.deg2rad(8.6),
        color=PALETTE["rotate"],
        alpha=0.74,
        edgecolor="white",
        linewidth=0.35,
    )
    ax_polar.scatter(
        np.deg2rad([float(item["angle_deg"]) for item in candidate_records]),
        [float(candidate_probs[list(candidate_angles).index(float(item["angle_deg"]))]) for item in candidate_records],
        s=62,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_title("3. Actual VOP posterior", loc="left", fontsize=13.6, fontweight="bold", pad=20)
    ax_polar.tick_params(labelsize=8.4)
    ax_polar.grid(color=PALETTE["grid"], alpha=0.82)
    ax_polar.text(
        0.5,
        -0.12,
        "Red markers = retained top-k candidates",
        transform=ax_polar.transAxes,
        ha="center",
        va="top",
        fontsize=9.1,
        color=PALETTE["muted"],
    )

    subgrid = gs[0, 3].subgridspec(2, 2, wspace=0.10, hspace=0.14)
    for idx, item in enumerate(candidate_records):
        ax = fig.add_subplot(subgrid[idx // 2, idx % 2])
        angle_deg = float(item["angle_deg"])
        prob = float(candidate_probs[list(candidate_angles).index(angle_deg)])
        add_candidate_card(
            ax,
            query_path=query_path,
            angle_deg=angle_deg,
            prob=prob,
            inliers=int(item["inliers"]),
            is_geom_selected=abs(angle_deg - float(best_angle)) < 1e-6,
            is_vop_top1=abs(angle_deg - float(top1_angle)) < 1e-6,
        )
    fig.text(
        fig.axes[-4].get_position().x0,
        fig.axes[-4].get_position().y1 + 0.02,
        "4. Verify only a useful top-k set",
        fontsize=13.6,
        fontweight="bold",
        color=PALETTE["text"],
    )

    ax_final = fig.add_subplot(gs[0, 4])
    ax_final.imshow(draw_projected_point(gallery_rgb, final_projected_xy, ""))
    ax_final.set_axis_off()
    ax_final.set_title("5. Geometry-selected output", loc="left", fontsize=13.6, fontweight="bold")
    ax_final.text(
        0.03,
        0.95,
        f"VOP top-1: {prettify_angle(top1_angle)}\ngeometry: {prettify_angle(best_angle)}",
        transform=ax_final.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        fontweight="bold",
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
    )
    ax_final.text(
        0.03,
        0.04,
        "For this illustration, top-k candidates are verified\nwith relaxed sparse settings to expose geometry differences.",
        transform=ax_final.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.9,
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
    )

    fig.suptitle(
        "Pair-specific VOP pipeline visualization on the Paper7 sample (01_0409 -> 01_5_005_022)",
        x=0.01,
        ha="left",
        fontsize=17.8,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.01,
        -0.015,
        "This summary figure is intended for the method/framework diagram. Posterior is real; top-k geometry verification uses visualization-friendly relaxed sparse settings on this single pair.",
        fontsize=9.8,
        color=PALETTE["muted"],
    )

    saved = save_figure(fig, output_dir, "pair01_vop_pipeline_summary")

    if selected_match_vis_path and Path(selected_match_vis_path).is_file():
        match_image = Image.open(selected_match_vis_path).convert("RGB")
        match_out = output_dir / "pair01_selected_match_visual.png"
        match_image.save(match_out)
        saved.append(match_out)

    return saved


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    setup_style()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_meta = load_pair_metadata(args.pairs_json, args.query_name, args.gallery_name)
    query_path = DATA_ROOT / pair_meta["drone_img_dir"] / args.query_name
    gallery_path = DATA_ROOT / pair_meta["sate_img_dir"] / args.gallery_name

    prepared = prepare_model_and_tensors(
        query_path=query_path,
        gallery_path=gallery_path,
        retrieval_checkpoint=args.retrieval_checkpoint,
        device=device,
    )
    model = prepared["model"]
    query_rgb = prepared["query_rgb"]
    gallery_rgb = prepared["gallery_rgb"]
    query_tensor = prepared["query_tensor"]
    gallery_tensor = prepared["gallery_tensor"]

    vop = load_vop_checkpoint(str(args.vop_checkpoint), device=device)
    candidate_angles = getattr(vop, "candidate_angles_deg", None) or [0.0]
    posterior = vop.predict_posterior(
        retrieval_model=model,
        gallery_img=gallery_tensor,
        query_img=query_tensor,
        candidate_angles_deg=candidate_angles,
        device=device,
        gallery_branch="img2",
        query_branch="img1",
    )
    probs = np.asarray(posterior["probs"], dtype=np.float64)
    order = list(np.argsort(probs)[::-1][: max(1, min(int(args.topk), len(candidate_angles)))])
    topk_angles = [float(candidate_angles[idx]) for idx in order]

    candidate_records = run_relaxed_sparse_verification(
        gallery_tensor=gallery_tensor,
        query_tensor=query_tensor,
        candidate_angles=topk_angles,
        output_dir=output_dir,
        device=device,
    )
    best_geometry = pick_best_geometry_candidate(candidate_records)
    top1_angle = float(posterior["top_angle_deg"])
    best_angle = float(best_geometry["angle_deg"])

    with torch.no_grad():
        gallery_batch = gallery_tensor.unsqueeze(0).to(device)
        query_batch = query_tensor.unsqueeze(0).to(device)
        gallery_map = model.extract_feature_map(gallery_batch, branch="img2")
        query_map = model.extract_feature_map(query_batch, branch="img1")
        gallery_map, query_map = vop.encode(gallery_map, query_map)
        sim_top1 = compute_similarity_map(gallery_map, query_map, top1_angle)
        sim_best = compute_similarity_map(gallery_map, query_map, best_angle)

    sim_top1_overlay = similarity_overlay(gallery_rgb, sim_top1)
    sim_best_overlay = similarity_overlay(gallery_rgb, sim_best)

    saved_paths: List[Path] = []
    saved_paths.extend(
        create_feature_comparison_figure(
            output_dir=output_dir,
            query_rgb=query_rgb,
            gallery_rgb=gallery_rgb,
            sim_top1_overlay=sim_top1_overlay,
            sim_best_overlay=sim_best_overlay,
            top1_angle=top1_angle,
            best_angle=best_angle,
        )
    )
    saved_paths.extend(
        create_posterior_figure(
            output_dir=output_dir,
            query_path=query_path,
            candidate_angles=candidate_angles,
            candidate_probs=probs.tolist(),
            candidate_records=candidate_records,
            top1_angle=top1_angle,
            best_angle=best_angle,
        )
    )
    saved_paths.extend(
        create_pipeline_step3_posterior_only(
            output_dir=output_dir,
            candidate_angles=candidate_angles,
            candidate_probs=probs.tolist(),
            candidate_records=candidate_records,
        )
    )
    saved_paths.extend(
        create_pipeline_summary_figure(
            output_dir=output_dir,
            query_path=query_path,
            gallery_path=gallery_path,
            gallery_rgb=gallery_rgb,
            candidate_angles=candidate_angles,
            candidate_probs=probs.tolist(),
            candidate_records=candidate_records,
            top1_angle=top1_angle,
            best_angle=best_angle,
            final_projected_xy=best_geometry.get("projected_xy"),
            selected_match_vis_path=best_geometry.get("final_vis_path"),
        )
    )

    summary = {
        "query_name": args.query_name,
        "gallery_name": args.gallery_name,
        "device": device,
        "phi1_deg": pair_meta.get("drone_metadata", {}).get("phi1"),
        "vop_top1_angle_deg": top1_angle,
        "vop_top1_prob": float(posterior["top_prob"]),
        "vop_entropy": float(posterior["entropy"]),
        "vop_concentration": float(posterior["concentration"]),
        "topk_candidates": [
            {
                "rank": int(item["rank"]),
                "angle_deg": float(item["angle_deg"]),
                "posterior_prob": float(probs[list(candidate_angles).index(float(item["angle_deg"]))]),
                "n_kept": int(item["n_kept"]),
                "inliers": int(item["inliers"]),
                "inlier_ratio": float(item["inlier_ratio"]),
                "projected_xy": None
                if item.get("projected_xy") is None
                else [float(item["projected_xy"][0]), float(item["projected_xy"][1])],
                "final_vis_path": item.get("final_vis_path"),
            }
            for item in candidate_records
        ],
        "geometry_selected_angle_deg": best_angle,
        "geometry_selected_projected_xy": None
        if best_geometry.get("projected_xy") is None
        else [float(best_geometry["projected_xy"][0]), float(best_geometry["projected_xy"][1])],
    }
    summary_path = output_dir / "pair01_vop_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    saved_paths.append(summary_path)

    print("Generated files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
