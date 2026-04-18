#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Generate real example figures for offline teacher signal construction."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from geopy.distance import geodesic
from PIL import Image
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
if str(GAME4LOC_DIR) not in sys.path:
    sys.path.insert(0, str(GAME4LOC_DIR))

from game4loc.dataset.visloc import get_transforms, tile2sate
from game4loc.evaluate.visloc import project_match_center_from_h
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.matcher.sparse_sp_lg import SparseSpLgMatcher
from game4loc.models.model import DesModel
from game4loc.orientation import build_rotation_angle_list, compute_entropy
from plot_vop_shortpaper_figures import PALETTE, save_figure, setup_style


DATA_ROOT = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset"
DEFAULT_RETRIEVAL_CKPT = (
    GAME4LOC_DIR
    / "work_dir"
    / "visloc"
    / "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    / "0409152642"
    / "weights_e10_0.6527.pth"
)
DEFAULT_META_JSON = DATA_ROOT / "same-area-paper7-drone2sate-test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-name", type=str, default="01_0409.JPG")
    parser.add_argument("--gallery-name", type=str, default="01_5_005_022.png")
    parser.add_argument("--pairs-json", type=Path, default=DEFAULT_META_JSON)
    parser.add_argument("--retrieval-checkpoint", type=Path, default=DEFAULT_RETRIEVAL_CKPT)
    parser.add_argument("--rotate-step", type=float, default=10.0)
    parser.add_argument("--temperature-m", type=float, default=25.0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC_DIR / "figures" / "teacher_signal_example_20260414",
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
    raise ValueError(f"Unsupported query tensor shape: {tuple(query_tensor.shape)}")


def soft_distribution_from_distances(distances_m: List[float], temperature_m: float) -> List[float]:
    distances = torch.tensor(distances_m, dtype=torch.float32)
    valid_mask = torch.isfinite(distances)
    logits = torch.full_like(distances, fill_value=-40.0)
    logits[valid_mask] = -distances[valid_mask] / max(float(temperature_m), 1e-6)
    return torch.softmax(logits, dim=0).tolist()


def tensor_to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def load_pair_metadata(meta_json: Path, query_name: str, gallery_name: str) -> Dict[str, object]:
    records = json.loads(meta_json.read_text(encoding="utf-8"))
    for item in records:
        if item.get("drone_img_name") != query_name:
            continue
        if gallery_name in item.get("pair_pos_sate_img_list", []):
            return item
    raise ValueError(f"Could not find pair metadata for query={query_name}, gallery={gallery_name}")


def build_example_record(args: argparse.Namespace) -> Dict[str, object]:
    pair_meta = load_pair_metadata(args.pairs_json, args.query_name, args.gallery_name)
    query_path = DATA_ROOT / pair_meta["drone_img_dir"] / args.query_name
    gallery_path = DATA_ROOT / pair_meta["sate_img_dir"] / args.gallery_name

    center_latlon, topleft_latlon = tile2sate(args.gallery_name)
    gallery_center_xy = (float(center_latlon[1]), float(center_latlon[0]))
    gallery_topleft_xy = (float(topleft_latlon[1]), float(topleft_latlon[0]))
    query_eval_loc = (float(pair_meta["drone_loc_lat_lon"][0]), float(pair_meta["drone_loc_lat_lon"][1]))

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    model = DesModel(
        "vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
        pretrained=False,
        img_size=384,
        share_weights=True,
    )
    state_dict = torch.load(args.retrieval_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    data_config = model.get_config()
    val_transforms, _, _ = get_transforms((384, 384), mean=data_config["mean"], std=data_config["std"])

    query_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    gallery_bgr = cv2.imread(str(gallery_path), cv2.IMREAD_COLOR)
    if query_bgr is None or gallery_bgr is None:
        raise FileNotFoundError("Failed to load the example pair images.")

    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    gallery_rgb = cv2.cvtColor(gallery_bgr, cv2.COLOR_BGR2RGB)
    query_tensor = val_transforms(image=query_rgb)["image"]
    gallery_tensor = val_transforms(image=gallery_rgb)["image"]

    matcher = GimDKM(device=device, match_mode="sparse")
    _ = matcher.est_center(
        gallery_tensor,
        query_tensor,
        gallery_center_xy,
        gallery_topleft_xy,
        yaw0=None,
        yaw1=None,
        rotate=float(args.rotate_step),
        case_name=f"{args.query_name}_teacher_curve",
    )

    angle_results = [item for item in (matcher.get_last_angle_results() or []) if int(item.get("phase", 0)) == 1]
    angle_results = sorted(angle_results, key=lambda item: float(item.get("search_angle", 0.0)))

    expected_angles = build_rotation_angle_list(args.rotate_step)
    angle_to_distance = {}
    enriched_results: List[Dict[str, object]] = []
    for item in angle_results:
        rot_angle = float(item.get("rot_angle", 0.0))
        loc_xy = project_match_center_from_h(
            item.get("homography"),
            gallery_tensor,
            gallery_center_xy,
            gallery_topleft_xy,
        )
        if loc_xy is None:
            distance_m = float("inf")
        else:
            pred_latlon = (float(loc_xy[1]), float(loc_xy[0]))
            distance_m = float(geodesic(query_eval_loc, pred_latlon).meters)
        angle_to_distance[round(rot_angle, 6)] = distance_m
        item_copy = dict(item)
        item_copy["distance_m"] = distance_m
        item_copy["projected_xy"] = None if loc_xy is None else [float(loc_xy[0]), float(loc_xy[1])]
        enriched_results.append(item_copy)

    distances_m = [float(angle_to_distance.get(round(float(angle), 6), float("inf"))) for angle in expected_angles]
    target_probs = soft_distribution_from_distances(distances_m, temperature_m=float(args.temperature_m))

    finite_pairs = [(idx, dist) for idx, dist in enumerate(distances_m) if math.isfinite(dist)]
    if not finite_pairs:
        raise RuntimeError("All teacher distances are invalid for this pair.")
    best_index = min(finite_pairs, key=lambda item: item[1])[0]
    sorted_finite = sorted(finite_pairs, key=lambda item: item[1])
    second_distance = float(sorted_finite[1][1]) if len(sorted_finite) > 1 else float("inf")
    distance_gap = second_distance - float(distances_m[best_index]) if math.isfinite(second_distance) else float("inf")
    entropy = float(compute_entropy(torch.tensor(target_probs, dtype=torch.float32).unsqueeze(0))[0].item())

    best_angle = float(expected_angles[best_index])

    match_vis_dir = args.output_dir / "match_vis"
    match_vis_dir.mkdir(parents=True, exist_ok=True)
    best_matcher = GimDKM(
        device=device,
        match_mode="sparse",
        sparse_save_final_vis=True,
        sparse_save_final_vis_dir=str(match_vis_dir),
        sparse_save_final_vis_max=8,
    )
    rotated_query_tensor = rotate_query_tensor(query_tensor, best_angle)
    _ = best_matcher.est_center(
        gallery_tensor,
        rotated_query_tensor,
        gallery_center_xy,
        gallery_topleft_xy,
        yaw0=None,
        yaw1=None,
        rotate=0.0,
        case_name=f"{args.query_name}_best_{best_angle:+.1f}",
        save_final_vis=True,
    )
    best_match_info = best_matcher.get_last_match_info() or {}
    best_match_vis = best_match_info.get("final_vis_path")

    gallery_vis_tensor = gallery_tensor.to(device)[None, ...] * 0.5 + 0.5
    query_vis_tensor = query_tensor.to(device)[None, ...] * 0.5 + 0.5

    sparse_vis_matcher = SparseSpLgMatcher(device=device)
    _, _, _, _, best_sparse_stats, _, _ = sparse_vis_matcher._run_matching_for_angles(
        gallery_vis_tensor,
        query_vis_tensor,
        yaw0=None,
        yaw1=None,
        candidate_angles=[best_angle],
        reproj_threshold=float(sparse_vis_matcher.sparse_ransac_reproj_threshold),
        run_name="paper_real_example",
    )

    return {
        "query_name": args.query_name,
        "gallery_name": args.gallery_name,
        "query_path": query_path,
        "gallery_path": gallery_path,
        "query_rgb": query_rgb,
        "gallery_rgb": gallery_rgb,
        "query_vis_tensor": query_vis_tensor,
        "gallery_vis_tensor": gallery_vis_tensor,
        "expected_angles": [float(angle) for angle in expected_angles],
        "distances_m": [float(value) for value in distances_m],
        "target_probs": [float(value) for value in target_probs],
        "best_index": int(best_index),
        "best_angle_deg": best_angle,
        "best_distance_m": float(distances_m[best_index]),
        "second_distance_m": float(second_distance),
        "distance_gap_m": float(distance_gap),
        "target_entropy": float(entropy),
        "pair_meta": pair_meta,
        "best_match_vis": best_match_vis,
        "best_match_info": best_match_info,
        "best_sparse_stats": best_sparse_stats,
        "device": device,
        "temperature_m": float(args.temperature_m),
        "enriched_results": enriched_results,
    }


def make_clean_match_vis(record: Dict[str, object], output_dir: Path) -> Path | None:
    best_sparse_stats = record.get("best_sparse_stats")
    if isinstance(best_sparse_stats, dict):
        mk0 = best_sparse_stats.get("mk0")
        mk1 = best_sparse_stats.get("mk1")
        image1_vis_rot = best_sparse_stats.get("image1_vis_rot")
        inlier_mask = best_sparse_stats.get("h_mask")
        if isinstance(mk0, np.ndarray) and isinstance(mk1, np.ndarray) and mk0.shape[0] > 0:
            img0 = tensor_to_uint8_rgb(record["gallery_vis_tensor"])
            if isinstance(image1_vis_rot, torch.Tensor):
                img1 = tensor_to_uint8_rgb(image1_vis_rot)
            else:
                img1 = tensor_to_uint8_rgb(record["query_vis_tensor"])

            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            gap = 18
            canvas = np.full((max(h0, h1), w0 + gap + w1, 3), fill_value=255, dtype=np.uint8)
            canvas[:h0, :w0] = img0
            canvas[:h1, w0 + gap : w0 + gap + w1] = img1

            inlier_bool = None
            if inlier_mask is not None:
                inlier_bool = np.asarray(inlier_mask).reshape(-1).astype(bool)
                if inlier_bool.shape[0] != mk0.shape[0]:
                    inlier_bool = None

            draw_mask = inlier_bool if inlier_bool is not None else np.ones((mk0.shape[0],), dtype=bool)
            draw_indices = np.flatnonzero(draw_mask)
            if draw_indices.size == 0:
                draw_indices = np.arange(mk0.shape[0], dtype=np.int32)
            if draw_indices.size > 60:
                draw_indices = draw_indices[np.linspace(0, draw_indices.size - 1, num=60, dtype=np.int32)]

            for idx in draw_indices:
                x0, y0 = int(round(float(mk0[idx, 0]))), int(round(float(mk0[idx, 1])))
                x1, y1 = int(round(float(mk1[idx, 0]))), int(round(float(mk1[idx, 1])))
                x1 += w0 + gap
                color = (41, 156, 90)
                cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (x1, y1), 2, color, -1, lineType=cv2.LINE_AA)

            caption = (
                f"teacher-best angle = {record['best_angle_deg']:+.0f} deg"
                f" | matches = {int(mk0.shape[0])}"
                f" | inliers = {int(draw_mask.sum())}"
            )
            cv2.putText(
                canvas,
                caption,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2,
                lineType=cv2.LINE_AA,
            )

            out_path = output_dir / "teacher_pair_geometric_matcher_real.png"
            Image.fromarray(canvas).save(out_path)
            return out_path

    match_vis_path = record.get("best_match_vis")
    if not match_vis_path or not Path(match_vis_path).is_file():
        return None
    image = Image.open(match_vis_path).convert("RGB")
    out_path = output_dir / "teacher_pair_geometric_matcher_real.png"
    image.save(out_path)
    return out_path


def create_error_curve_figure(record: Dict[str, object], output_dir: Path) -> List[Path]:
    angles = np.asarray(record["expected_angles"], dtype=float)
    distances = np.asarray(record["distances_m"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(angles, distances, color=PALETTE["dense"], linewidth=2.2, marker="o", markersize=4.5)
    ax.scatter(
        [record["best_angle_deg"]],
        [record["best_distance_m"]],
        s=90,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.1,
        zorder=5,
    )
    ax.axhline(record["best_distance_m"] + 5.0, color=PALETTE["rotate"], linestyle="--", linewidth=1.4, alpha=0.85)
    ax.text(
        record["best_angle_deg"],
        record["best_distance_m"] + 3.0,
        f"best {record['best_distance_m']:.1f} m @ {record['best_angle_deg']:+.0f}°",
        fontsize=9.5,
        color=PALETTE["ours"],
        ha="left",
        va="bottom",
    )
    ax.set_title("Localization Error Curve", loc="left", fontsize=15.0, fontweight="bold")
    ax.set_xlabel("Candidate angle (deg)")
    ax.set_ylabel("Localization error (m)")
    ax.set_xlim(float(angles.min()), float(angles.max()))
    ax.grid(color=PALETTE["grid"], alpha=0.75)
    fig.suptitle(
        "Real teacher error curve on the Paper7 example pair",
        x=0.01,
        ha="left",
        fontsize=16.2,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Computed by the actual offline teacher procedure: fix retrieval top-1, sweep angles, and measure final localization error after geometric verification.",
        fontsize=9.9,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "teacher_pair_localization_error_curve")


def create_soft_target_figure(record: Dict[str, object], output_dir: Path) -> List[Path]:
    angles = np.asarray(record["expected_angles"], dtype=float)
    targets = np.asarray(record["target_probs"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = np.full((len(angles),), PALETTE["rotate"], dtype=object)
    colors[int(record["best_index"])] = PALETTE["ours"]
    ax.bar(angles, targets, width=7.6, color=colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_title("Soft Orientation Target", loc="left", fontsize=15.0, fontweight="bold")
    ax.set_xlabel("Candidate angle (deg)")
    ax.set_ylabel("Target probability")
    ax.set_xlim(float(angles.min()) - 5.0, float(angles.max()) + 5.0)
    ax.grid(color=PALETTE["grid"], alpha=0.75)
    ax.text(
        0.02,
        0.96,
        f"temperature = {record['temperature_m']:.0f} m\nentropy = {record['target_entropy']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.92),
    )
    fig.suptitle(
        "Real soft target derived from the teacher error curve",
        x=0.01,
        ha="left",
        fontsize=16.2,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "This is the exact soft profile used by the original soft-target teacher line: softmax(-distance / temperature).",
        fontsize=9.9,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "teacher_pair_soft_orientation_target")


def create_soft_target_clean_figure(record: Dict[str, object], output_dir: Path) -> List[Path]:
    angles = np.asarray(record["expected_angles"], dtype=float)
    targets = np.asarray(record["target_probs"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = np.full((len(angles),), PALETTE["rotate"], dtype=object)
    colors[int(record["best_index"])] = PALETTE["ours"]
    ax.bar(angles, targets, width=7.6, color=colors, alpha=0.88, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Probability")
    ax.set_xlim(float(angles.min()) - 5.0, float(angles.max()) + 5.0)
    ax.grid(color=PALETTE["grid"], alpha=0.75)
    saved = save_figure(fig, output_dir, "teacher_pair_soft_orientation_target_clean")
    saved.extend(save_figure(fig, output_dir, "teacher_pair_soft_orientation_target_v1"))
    return saved


def build_periodic_smooth_curve(
    angles: np.ndarray,
    targets: np.ndarray,
    *,
    dense_count: int = 1200,
    sigma_deg: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(angles, dtype=float)
    targets = np.asarray(targets, dtype=float)

    dense_angles = np.linspace(float(angles.min()), float(angles.max()), dense_count)
    circular_diff = ((dense_angles[:, None] - angles[None, :] + 180.0) % 360.0) - 180.0
    kernel = np.exp(-0.5 * (circular_diff / sigma_deg) ** 2)
    smooth_targets = (kernel * targets[None, :]).sum(axis=1)
    return dense_angles, smooth_targets


def create_soft_target_smooth_figure(record: Dict[str, object], output_dir: Path) -> List[Path]:
    angles = np.asarray(record["expected_angles"], dtype=float)
    targets = np.asarray(record["target_probs"], dtype=float)
    best_idx = int(record["best_index"])
    dense_angles, smooth_targets = build_periodic_smooth_curve(angles, targets, sigma_deg=9.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.fill_between(dense_angles, smooth_targets, 0.0, color=PALETTE["rotate"], alpha=0.16, zorder=1)
    ax.plot(dense_angles, smooth_targets, color=PALETTE["rotate"], linewidth=2.6, zorder=3)
    ax.scatter(
        angles,
        targets,
        s=34,
        color=PALETTE["rotate"],
        edgecolors="white",
        linewidths=0.6,
        alpha=0.95,
        zorder=4,
    )
    ax.scatter(
        [angles[best_idx]],
        [targets[best_idx]],
        s=96,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.0,
        zorder=5,
    )
    ax.axvline(float(angles[best_idx]), color=PALETTE["ours"], linewidth=1.2, alpha=0.9, zorder=2)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Probability")
    ax.set_xlim(float(angles.min()) - 5.0, float(angles.max()) + 5.0)
    ax.set_ylim(0.0, max(float(smooth_targets.max()), float(targets.max())) * 1.12)
    ax.grid(color=PALETTE["grid"], alpha=0.75)
    return save_figure(fig, output_dir, "teacher_pair_soft_orientation_target_v2_smooth")


def create_triptych_figure(record: Dict[str, object], output_dir: Path, clean_match_vis: Path | None) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)

    if clean_match_vis is not None and clean_match_vis.is_file():
        axes[0].imshow(Image.open(clean_match_vis).convert("RGB"))
    else:
        axes[0].imshow(record["gallery_rgb"])
    axes[0].set_title("Geometric Matcher", fontsize=13.8, fontweight="bold")
    axes[0].set_axis_off()

    angles = np.asarray(record["expected_angles"], dtype=float)
    distances = np.asarray(record["distances_m"], dtype=float)
    axes[1].plot(angles, distances, color=PALETTE["dense"], linewidth=2.0)
    axes[1].scatter(
        [record["best_angle_deg"]],
        [record["best_distance_m"]],
        s=70,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    axes[1].set_title("Localization Error Curve", fontsize=13.8, fontweight="bold")
    axes[1].set_xlabel("Angle (deg)")
    axes[1].set_ylabel("Error (m)")
    axes[1].grid(color=PALETTE["grid"], alpha=0.75)

    targets = np.asarray(record["target_probs"], dtype=float)
    bar_colors = np.full((len(angles),), PALETTE["rotate"], dtype=object)
    bar_colors[int(record["best_index"])] = PALETTE["ours"]
    axes[2].bar(angles, targets, width=7.6, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.4)
    axes[2].set_title("Soft Orientation Target", fontsize=13.8, fontweight="bold")
    axes[2].set_xlabel("Angle (deg)")
    axes[2].set_ylabel("Probability")
    axes[2].grid(color=PALETTE["grid"], alpha=0.75)

    fig.suptitle(
        "Offline Teacher Signal Construction: real example from the selected Paper7 pair",
        x=0.01,
        ha="left",
        fontsize=17.0,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "teacher_pair_triptych")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record = build_example_record(args)
    clean_match_vis = make_clean_match_vis(record, args.output_dir)

    saved_paths: List[Path] = []
    if clean_match_vis is not None:
        saved_paths.append(clean_match_vis)
    saved_paths.extend(create_error_curve_figure(record, args.output_dir))
    saved_paths.extend(create_soft_target_figure(record, args.output_dir))
    saved_paths.extend(create_soft_target_clean_figure(record, args.output_dir))
    saved_paths.extend(create_soft_target_smooth_figure(record, args.output_dir))
    saved_paths.extend(create_triptych_figure(record, args.output_dir, clean_match_vis))

    summary_path = args.output_dir / "teacher_pair_summary.json"
    summary = {
        "query_name": record["query_name"],
        "gallery_name": record["gallery_name"],
        "phi1_deg": record["pair_meta"].get("drone_metadata", {}).get("phi1"),
        "best_angle_deg": record["best_angle_deg"],
        "best_distance_m": record["best_distance_m"],
        "second_distance_m": record["second_distance_m"],
        "distance_gap_m": record["distance_gap_m"],
        "target_entropy": record["target_entropy"],
        "temperature_m": record["temperature_m"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    saved_paths.append(summary_path)

    print("Generated files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
