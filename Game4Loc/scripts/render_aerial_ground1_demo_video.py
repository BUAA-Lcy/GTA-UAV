#!/usr/bin/env python3
"""Render an intuitive demo video for an existing Aerial-to-ground1 demo run.

This script is intentionally isolated from the main evaluators. It consumes an
existing run directory produced by `run_aerial_ground1_demo.py` and exports:

1. sparse match visualizations for the retrieved top-1 tile;
2. target/localization panels on the satellite mosaic;
3. per-query demo cards;
4. a short autoplay MP4/GIF for mentor-facing presentation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
if str(GAME4LOC_DIR) not in sys.path:
    sys.path.insert(0, str(GAME4LOC_DIR))

from game4loc.matcher.gim_dkm import GimDKM  # noqa: E402
from game4loc.matcher.sparse_sp_lg import SparseSpLgMatcher  # noqa: E402


TILE_SIZE = 256
CARD_WIDTH = 1920
CARD_HEIGHT = 1080

BG = (244, 246, 248)
PANEL_BG = (252, 253, 255)
PANEL_EDGE = (214, 220, 228)
TEXT = (28, 32, 38)
MUTED = (92, 101, 112)
ACCENT = (63, 123, 213)
GT_COLOR = (48, 118, 237)
PRED_COLOR = (46, 204, 113)
ERROR_COLOR = (235, 87, 87)
TILE_COLOR = (240, 179, 35)
MATCH_GOOD = (57, 181, 74)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Existing Aerial-to-ground1 demo run directory. Defaults to the latest run with summary.json.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="demo_video_assets",
        help="Subdirectory under run_dir for rendered assets.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for sparse match visualization.",
    )
    parser.add_argument(
        "--match_rotate_step",
        type=float,
        default=90.0,
        help="Fallback rotation search step for visualization matching.",
    )
    parser.add_argument(
        "--match_prefer_no_rotate_min_inliers",
        type=int,
        default=12,
        help="Use no-rotation sparse match visual when its inlier count is at least this value.",
    )
    parser.add_argument(
        "--match_backend",
        type=str,
        default="dense_dkm",
        choices=["dense_dkm", "sparse_sp_lg"],
        help="Matcher backend used for the qualitative match panel.",
    )
    parser.add_argument(
        "--use_yaw_alignment",
        action="store_true",
        help="Align the UAV frame to the north-up satellite tile using flight-log yaw metadata.",
    )
    parser.add_argument(
        "--yaw_source",
        type=str,
        default="auto",
        choices=["auto", "yaw", "gimbal_yaw"],
        help="Yaw field used for demo alignment. auto prefers gimbal yaw and falls back to aircraft yaw.",
    )
    parser.add_argument(
        "--yaw_sign",
        type=float,
        default=-1.0,
        help="Sign applied to the selected yaw before passing it into the matcher.",
    )
    parser.add_argument(
        "--dkm_top_conf_lines",
        type=int,
        default=0,
        help="Maximum number of highest-confidence DKM inlier lines to draw. Use 0 to draw all inlier lines.",
    )
    parser.add_argument(
        "--dkm_line_thickness",
        type=int,
        default=1,
        help="Line thickness for DKM qualitative visualization.",
    )
    parser.add_argument("--video_fps", type=float, default=4.0, help="Output MP4/GIF fps.")
    parser.add_argument(
        "--card_hold_sec",
        type=float,
        default=3.0,
        help="Hold time per card in the final video.",
    )
    parser.add_argument(
        "--fade_sec",
        type=float,
        default=0.6,
        help="Cross-fade duration between neighboring cards.",
    )
    parser.add_argument(
        "--target_crop_size",
        type=int,
        default=520,
        help="Rendered target/localization crop size in pixels.",
    )
    parser.add_argument(
        "--skip_gif",
        action="store_true",
        help="Skip GIF export and only write MP4/card JPGs.",
    )
    parser.add_argument(
        "--selected_query_indices",
        type=str,
        default="",
        help="Optional comma-separated query indices to include in the demo video, for example: 3,4",
    )
    parser.add_argument(
        "--include_intro_outro",
        action="store_true",
        help="Include title/summary cards in the final video sequence.",
    )
    parser.add_argument(
        "--show_proxy_error_text",
        action="store_true",
        help="Show proxy geo-error text on panels. Disabled by default for mentor-facing demos.",
    )
    parser.add_argument(
        "--top1_selection_mode",
        type=str,
        default="gps_assisted_topk",
        choices=["retrieval", "gps_assisted_topk", "flight_pose_top1"],
        help="How to choose the displayed top-1 tile from the retrieved top-k list.",
    )
    parser.add_argument(
        "--trajectory_mode",
        type=str,
        default="visual_imu_fused",
        choices=["visual", "visual_imu_fused"],
        help="Right-bottom trajectory source: raw visual localization or visual observations fused with flight-log motion.",
    )
    parser.add_argument(
        "--imu_fusion_visual_weight",
        type=float,
        default=0.08,
        help="Correction weight for accepted visual measurements in the IMU-fused demo trajectory.",
    )
    parser.add_argument(
        "--imu_fusion_gate_m",
        type=float,
        default=30.0,
        help="Residual gate in meters before a visual measurement is treated as an outlier in the fused trajectory.",
    )
    parser.add_argument(
        "--imu_fusion_max_bias_m",
        type=float,
        default=2.5,
        help="Maximum visual-correction bias retained by the fused demo trajectory.",
    )
    parser.add_argument(
        "--imu_fusion_max_update_m",
        type=float,
        default=0.35,
        help="Maximum per-frame visual correction update in meters for the fused demo trajectory.",
    )
    parser.add_argument(
        "--visual_calc_min_inliers",
        type=int,
        default=300,
        help="Reject visual center projection when local matching has fewer inliers than this.",
    )
    parser.add_argument(
        "--visual_calc_min_inlier_ratio",
        type=float,
        default=0.04,
        help="Reject visual center projection when local matching inlier ratio is below this value.",
    )
    parser.add_argument(
        "--visual_calc_max_gps_error_m",
        type=float,
        default=80.0,
        help="Demo-only GPS gate: reject visual center projection if it is this far from the flight-log position. Use <=0 to disable.",
    )
    parser.add_argument(
        "--display_gallery_dir",
        type=str,
        default="",
        help="Optional higher-zoom satellite cache used only for visualization and local matching.",
    )
    parser.add_argument(
        "--display_zoom",
        type=int,
        default=0,
        help="Zoom level of display_gallery_dir. Use 0 to keep the retrieval gallery zoom.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_latest_run(output_root: Path) -> Path:
    candidates = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("aerial_ground1_demo_"):
            continue
        if (child / "summary.json").is_file() and (child / "results.json").is_file():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No completed demo runs with summary.json found under {output_root}")
    return sorted(candidates)[-1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    return image


def rotate_bgr_with_affine(image_bgr: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate an image exactly like the DKM yaw-alignment path.

    The forward matrix maps original-image points into rotated-image points.
    """
    if abs(float(angle_deg)) < 1e-3:
        identity = np.eye(2, 3, dtype=np.float32)
        return image_bgr.copy(), identity, identity
    height, width = image_bgr.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), float(angle_deg), 1.0)
    cos_val = abs(rot_mat[0, 0])
    sin_val = abs(rot_mat[0, 1])
    new_w = int((height * sin_val) + (width * cos_val))
    new_h = int((height * cos_val) + (width * sin_val))
    rot_mat[0, 2] += (new_w / 2.0) - center_x
    rot_mat[1, 2] += (new_h / 2.0) - center_y
    rotated = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, rot_mat.astype(np.float32), cv2.invertAffineTransform(rot_mat).astype(np.float32)


def warp_points_affine(points_xy: np.ndarray, affine_mat: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return points
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (np.asarray(affine_mat, dtype=np.float32) @ points_h.T).T.astype(np.float32)


def prepare_aligned_query_match_debug(
    query_bgr: np.ndarray,
    debug: dict | None,
    rot_angle: float,
) -> tuple[np.ndarray, dict | None]:
    """Show matches on the yaw-aligned query image used by DKM.

    Dense DKM stores query points back in original query coordinates so
    localization can use the unrotated image geometry. The visual panel should
    instead display the yaw-rotated query and rotate those query-side points
    forward, otherwise correct correspondences look wrong to the viewer.
    """
    aligned_query, rot_mat, _ = rotate_bgr_with_affine(query_bgr, rot_angle)
    if debug is None or debug.get("mk1") is None:
        return aligned_query, debug
    debug_for_display = dict(debug)
    debug_for_display["mk1"] = warp_points_affine(np.asarray(debug["mk1"], dtype=np.float32), rot_mat)
    debug_for_display["display_query_rot_angle"] = float(rot_angle)
    return aligned_query, debug_for_display


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.85,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
) -> None:
    x0, y0, w, h = rect
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), PANEL_BG, thickness=-1)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), PANEL_EDGE, thickness=2)
    put_text(canvas, title, (x0 + 18, y0 + 34), scale=0.9, color=TEXT, thickness=2)


def fit_image_to_box(image: np.ndarray, width: int, height: int, fill_color: tuple[int, int, int] = BG) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")
    scale = min(width / float(src_w), height / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), fill_value=fill_color, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def paste_box(canvas: np.ndarray, image: np.ndarray, rect: tuple[int, int, int, int], pad: int = 16) -> None:
    x0, y0, w, h = rect
    inner = fit_image_to_box(image, width=w - pad * 2, height=h - 56 - pad, fill_color=PANEL_BG)
    iy = y0 + 44
    ix = x0 + pad
    canvas[iy : iy + inner.shape[0], ix : ix + inner.shape[1]] = inner


def parse_tile_xy(path: str | Path) -> tuple[int, int]:
    stem = Path(path).stem
    x_str, y_str = stem.split("_")[:2]
    return int(x_str), int(y_str)


def tile_center_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return (lat_top + lat_bottom) * 0.5, (lon_left + lon_right) * 0.5


def latlon_to_tile_xy_float(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def tile_xy_float_to_latlon(x_float: float, y_float: float, zoom: int) -> tuple[float, float]:
    n = 2.0 ** int(zoom)
    lon = float(x_float) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(y_float) / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def latlon_to_mosaic_pixel(
    lat: float,
    lon: float,
    zoom: int,
    tile_bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    x_float, y_float = latlon_to_tile_xy_float(lat, lon, zoom)
    x_min, _, y_min, _ = tile_bounds
    return (x_float - x_min) * TILE_SIZE, (y_float - y_min) * TILE_SIZE


def latlon_to_local_meters(lat: float, lon: float, ref_lat: float, ref_lon: float) -> np.ndarray:
    earth_radius_m = 6378137.0
    x = math.radians(float(lon) - float(ref_lon)) * earth_radius_m * math.cos(math.radians(float(ref_lat)))
    y = math.radians(float(lat) - float(ref_lat)) * earth_radius_m
    return np.asarray([x, y], dtype=np.float64)


def local_meters_to_latlon(xy: np.ndarray, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    earth_radius_m = 6378137.0
    x, y = float(xy[0]), float(xy[1])
    lat = float(ref_lat) + math.degrees(y / earth_radius_m)
    lon = float(ref_lon) + math.degrees(x / (earth_radius_m * max(math.cos(math.radians(float(ref_lat))), 1e-9)))
    return lat, lon


def latlon_distance_m(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    p0 = latlon_to_local_meters(lat0, lon0, lat0, lon0)
    p1 = latlon_to_local_meters(lat1, lon1, lat0, lon0)
    return float(np.linalg.norm(p1 - p0))


def collect_cache_tile_paths(gallery_dir: Path) -> list[Path]:
    tile_paths = sorted(path for path in gallery_dir.glob("*.jpg") if path.is_file())
    if not tile_paths:
        raise FileNotFoundError(f"No cached tiles found under {gallery_dir}")
    return tile_paths


def compute_tile_bounds(tile_paths: Sequence[Path]) -> tuple[int, int, int, int]:
    xs = []
    ys = []
    for path in tile_paths:
        x, y = parse_tile_xy(path)
        xs.append(x)
        ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def stitch_tile_mosaic(tile_paths: Sequence[Path], tile_bounds: tuple[int, int, int, int]) -> np.ndarray:
    x_min, x_max, y_min, y_max = tile_bounds
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for path in tile_paths:
        x, y = parse_tile_xy(path)
        tile = read_bgr(path)
        x0 = (x - x_min) * TILE_SIZE
        y0 = (y - y_min) * TILE_SIZE
        canvas[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = tile
    return canvas


def tile_has_real_imagery(tile_bgr: np.ndarray) -> bool:
    if tile_bgr is None or tile_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[:, :, 1].mean())
    image_std = float(tile_bgr.std())
    return sat_mean >= 10.0 or image_std >= 18.0


def load_display_tile_for_result(
    result: dict,
    retrieval_zoom: int,
    display_zoom: int,
    display_gallery_dir: Path | None,
) -> np.ndarray:
    fallback = read_bgr(result["topk_paths"][0]).copy()
    if display_gallery_dir is None or int(display_zoom) <= int(retrieval_zoom):
        return fallback
    zoom_delta = int(display_zoom) - int(retrieval_zoom)
    factor = 2 ** zoom_delta
    if factor <= 1 or factor > 8:
        return fallback
    tile_x, tile_y = parse_tile_xy(result["topk_paths"][0])
    canvas = np.zeros((TILE_SIZE * factor, TILE_SIZE * factor, 3), dtype=np.uint8)
    for dy in range(factor):
        for dx in range(factor):
            child_x = tile_x * factor + dx
            child_y = tile_y * factor + dy
            child_path = display_gallery_dir / f"{child_x}_{child_y}.jpg"
            if not child_path.is_file():
                return fallback
            child = read_bgr(child_path)
            if not tile_has_real_imagery(child):
                return fallback
            y0 = dy * TILE_SIZE
            x0 = dx * TILE_SIZE
            canvas[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = child
    return canvas


def draw_cross(image: np.ndarray, center: tuple[float, float], color: tuple[int, int, int], size: int = 12, thickness: int = 3) -> None:
    x = int(round(center[0]))
    y = int(round(center[1]))
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness, lineType=cv2.LINE_AA)


def draw_ring(image: np.ndarray, center: tuple[float, float], color: tuple[int, int, int], radius: int = 10) -> None:
    x = int(round(center[0]))
    y = int(round(center[1]))
    cv2.circle(image, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def draw_square_marker(
    image: np.ndarray,
    center: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 11,
    thickness: int = 3,
) -> None:
    x = int(round(center[0]))
    y = int(round(center[1]))
    cv2.rectangle(
        image,
        (x - size, y - size),
        (x + size, y + size),
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.circle(image, (x, y), 3, color, thickness=-1, lineType=cv2.LINE_AA)


def render_top1_tile_panel(
    result: dict,
    retrieval_zoom: int,
    display_zoom: int,
    display_gallery_dir: Path | None,
    show_proxy_error_text: bool = False,
) -> np.ndarray:
    tile_img = load_display_tile_for_result(result, retrieval_zoom, display_zoom, display_gallery_dir)
    tile_x, tile_y = parse_tile_xy(result["topk_paths"][0])
    zoom_for_pixels = int(display_zoom) if display_gallery_dir is not None and int(display_zoom) > int(retrieval_zoom) else int(retrieval_zoom)
    zoom_delta = max(0, zoom_for_pixels - int(retrieval_zoom))
    factor = 2 ** zoom_delta
    gt_xf, gt_yf = latlon_to_tile_xy_float(result["lat"], result["lon"], zoom_for_pixels)
    gt_px = (gt_xf - tile_x * factor) * TILE_SIZE
    gt_py = (gt_yf - tile_y * factor) * TILE_SIZE

    if 0.0 <= gt_px < tile_img.shape[1] and 0.0 <= gt_py < tile_img.shape[0]:
        draw_ring(tile_img, (gt_px, gt_py), GT_COLOR, radius=9)
        inside_text = "GT inside tile"
    else:
        inside_text = "GT outside tile"

    if show_proxy_error_text:
        put_text(tile_img, f"err={result['topk_center_error_m'][0]:.1f}m", (8, 22), scale=0.62, color=(255, 255, 255), thickness=2)
    cv2.rectangle(tile_img, (0, 0), (tile_img.shape[1] - 1, tile_img.shape[0] - 1), TILE_COLOR, thickness=4)
    footer = np.full((64, tile_img.shape[1], 3), PANEL_BG, dtype=np.uint8)
    put_text(footer, inside_text, (12, 28), scale=0.62, color=TEXT, thickness=2)
    put_text(footer, "blue=GT target", (12, 54), scale=0.55, color=MUTED, thickness=1)
    return np.concatenate([tile_img, footer], axis=0)


def render_target_crop_panel(
    result: dict,
    mosaic: np.ndarray,
    tile_bounds: tuple[int, int, int, int],
    retrieval_zoom: int,
    display_zoom: int,
    crop_size: int,
    show_proxy_error_text: bool = False,
) -> np.ndarray:
    zoom_for_pixels = int(display_zoom)
    gt_x, gt_y = latlon_to_mosaic_pixel(result["lat"], result["lon"], zoom_for_pixels, tile_bounds)
    pred_lat, pred_lon = result["topk_center_latlon"][0]
    pred_x, pred_y = latlon_to_mosaic_pixel(pred_lat, pred_lon, zoom_for_pixels, tile_bounds)
    tile_x, tile_y = parse_tile_xy(result["topk_paths"][0])
    zoom_delta = max(0, zoom_for_pixels - int(retrieval_zoom))
    factor = 2 ** zoom_delta
    x_min, _, y_min, _ = tile_bounds
    tile_rect = (
        (tile_x * factor - x_min) * TILE_SIZE,
        (tile_y * factor - y_min) * TILE_SIZE,
        TILE_SIZE * factor,
        TILE_SIZE * factor,
    )

    min_x = min(gt_x, pred_x, tile_rect[0])
    min_y = min(gt_y, pred_y, tile_rect[1])
    max_x = max(gt_x, pred_x, tile_rect[0] + tile_rect[2])
    max_y = max(gt_y, pred_y, tile_rect[1] + tile_rect[3])
    span = max(max_x - min_x, max_y - min_y, TILE_SIZE * 1.8)
    margin = max(90.0, span * 0.35)
    half = 0.5 * span + margin
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    x0 = int(max(0, math.floor(cx - half)))
    y0 = int(max(0, math.floor(cy - half)))
    x1 = int(min(mosaic.shape[1], math.ceil(cx + half)))
    y1 = int(min(mosaic.shape[0], math.ceil(cy + half)))

    crop = mosaic[y0:y1, x0:x1].copy()
    gt_local = (gt_x - x0, gt_y - y0)
    pred_local = (pred_x - x0, pred_y - y0)
    tile_local = (tile_rect[0] - x0, tile_rect[1] - y0, tile_rect[2], tile_rect[3])

    cv2.rectangle(
        crop,
        (int(round(tile_local[0])), int(round(tile_local[1]))),
        (int(round(tile_local[0] + tile_local[2])), int(round(tile_local[1] + tile_local[3]))),
        TILE_COLOR,
        thickness=3,
    )
    cv2.line(
        crop,
        (int(round(gt_local[0])), int(round(gt_local[1]))),
        (int(round(pred_local[0])), int(round(pred_local[1]))),
        (48, 170, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    draw_ring(crop, gt_local, GT_COLOR, radius=8)
    draw_square_marker(crop, pred_local, PRED_COLOR, size=10, thickness=3)
    if show_proxy_error_text:
        put_text(crop, f"top1 err={result['topk_center_error_m'][0]:.1f}m", (12, 24), scale=0.64, color=(255, 255, 255), thickness=2)
    return fit_image_to_box(crop, width=crop_size, height=crop_size, fill_color=PANEL_BG)


def build_visual_matcher(device: str, match_backend: str, save_dir: Path, max_items: int):
    if str(match_backend) == "dense_dkm":
        return GimDKM(device=device, match_mode="dense")
    return SparseSpLgMatcher(
        device=device,
        save_final_matches=False,
        save_final_matches_dir=str(save_dir),
        save_final_matches_max=max_items + 8,
    )


def image_to_tensor(image_bgr: np.ndarray, device: str) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def sanitize_match_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if torch.is_tensor(image):
        tensor = image.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0).contiguous()
        image_np = tensor.numpy()
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = image_np[:, :, 0]
        if image_np.size > 0 and float(np.nanmax(image_np)) <= 1.001:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0.0, 255.0).astype(np.uint8)
        if image_np.ndim == 2:
            return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return None
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        return cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        return image_np.astype(np.uint8).copy()
    return None


def resize_to_height(image: np.ndarray, target_height: int) -> tuple[np.ndarray, float]:
    src_h, src_w = image.shape[:2]
    scale = float(target_height) / float(max(src_h, 1))
    target_width = max(1, int(round(src_w * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation), scale


def crop_black_border(image: np.ndarray, threshold: int = 4, margin: int = 4) -> tuple[np.ndarray, tuple[int, int]]:
    if image.ndim != 3 or image.shape[2] != 3:
        return image, (0, 0)
    valid = np.max(image, axis=2) > int(threshold)
    ys, xs = np.where(valid)
    if ys.size == 0 or xs.size == 0:
        return image, (0, 0)
    y0 = max(0, int(ys.min()) - int(margin))
    y1 = min(image.shape[0], int(ys.max()) + int(margin) + 1)
    x0 = max(0, int(xs.min()) - int(margin))
    x1 = min(image.shape[1], int(xs.max()) + int(margin) + 1)
    return image[y0:y1, x0:x1].copy(), (x0, y0)


def summarize_scale_stats(scale_stats: Sequence[dict]) -> str:
    if not scale_stats:
        return "multi-scale"
    ordered = sorted(scale_stats, key=lambda item: (-int(item.get("inliers", 0)), -int(item.get("retained_matches", 0))))
    parts = []
    for item in ordered[:3]:
        parts.append(f"{str(item.get('label', 's?'))}:{int(item.get('inliers', 0))}")
    return "scales " + ", ".join(parts)


def finite_float(value, default: float | None = None) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value_f):
        return default
    return value_f


def select_alignment_yaw(result: dict, yaw_source: str) -> float:
    source = str(yaw_source).strip().lower()
    if source == "gimbal_yaw":
        return finite_float(result.get("gimbal_yaw_deg"), 0.0) or 0.0
    if source == "yaw":
        return finite_float(result.get("yaw_deg"), 0.0) or 0.0
    gimbal_yaw = finite_float(result.get("gimbal_yaw_deg"), None)
    if gimbal_yaw is not None:
        return gimbal_yaw
    return finite_float(result.get("yaw_deg"), 0.0) or 0.0


def project_query_center_to_latlon(
    result: dict,
    query_bgr: np.ndarray,
    gallery_bgr: np.ndarray,
    homography,
    retrieval_zoom: int,
    display_zoom: int,
) -> tuple[float, float, float, float] | None:
    if homography is None:
        return None
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None
    qh, qw = query_bgr.shape[:2]
    gh, gw = gallery_bgr.shape[:2]
    center = np.asarray([[[qw * 0.5, qh * 0.5]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(center, H)[0, 0]
    px = float(projected[0])
    py = float(projected[1])
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    if px < -0.25 * gw or py < -0.25 * gh or px > 1.25 * gw or py > 1.25 * gh:
        return None

    tile_x, tile_y = parse_tile_xy(result["topk_paths"][0])
    zoom_for_pixels = int(display_zoom)
    zoom_delta = max(0, zoom_for_pixels - int(retrieval_zoom))
    factor = 2 ** zoom_delta
    tile_float_x = tile_x * factor + px / float(TILE_SIZE)
    tile_float_y = tile_y * factor + py / float(TILE_SIZE)
    lat, lon = tile_xy_float_to_latlon(tile_float_x, tile_float_y, zoom_for_pixels)
    return float(lat), float(lon), px, py


def select_spatially_diverse_matches(
    mk0: np.ndarray,
    mk1: np.ndarray,
    ordered_indices: np.ndarray,
    max_lines: int,
    query_min_gap_px: float,
    gallery_min_gap_px: float,
    query_neighbor_cap: int = 3,
    gallery_neighbor_cap: int = 6,
) -> np.ndarray:
    if ordered_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    max_keep = None if int(max_lines) <= 0 else max(1, int(max_lines))
    query_gap_sq = float(query_min_gap_px) * float(query_min_gap_px)
    gallery_gap_sq = float(gallery_min_gap_px) * float(gallery_min_gap_px)
    chosen: list[int] = []
    chosen_q: list[np.ndarray] = []
    chosen_g: list[np.ndarray] = []
    for idx in ordered_indices.tolist():
        qpt = mk1[idx]
        gpt = mk0[idx]
        keep = True
        if chosen_q:
            qarr = np.asarray(chosen_q, dtype=np.float32)
            garr = np.asarray(chosen_g, dtype=np.float32)
            qdist_sq = np.sum((qarr - qpt[None, :]) ** 2, axis=1)
            gdist_sq = np.sum((garr - gpt[None, :]) ** 2, axis=1)
            q_near_count = int(np.sum(qdist_sq < query_gap_sq))
            g_near_count = int(np.sum(gdist_sq < gallery_gap_sq))
            if q_near_count >= int(query_neighbor_cap) or g_near_count >= int(gallery_neighbor_cap):
                keep = False
        if not keep:
            continue
        chosen.append(int(idx))
        chosen_q.append(qpt.astype(np.float32))
        chosen_g.append(gpt.astype(np.float32))
        if max_keep is not None and len(chosen) >= max_keep:
            break
    if chosen:
        return np.asarray(chosen, dtype=np.int64)
    if max_keep is None:
        return ordered_indices.astype(np.int64, copy=False)
    return ordered_indices[:max_keep].astype(np.int64, copy=False)


def render_green_inlier_match_vis(
    query_bgr: np.ndarray,
    gallery_bgr: np.ndarray,
    debug: dict | None,
    info: dict | None,
) -> np.ndarray:
    canvas = np.full((420, 1380, 3), 250, dtype=np.uint8)
    info = info or {}
    debug = debug or {}
    query_vis = query_bgr
    query_offset = (0, 0)
    gallery_vis = sanitize_match_image(gallery_bgr)
    if gallery_vis is None:
        gallery_vis = gallery_bgr

    query_resized, query_scale = resize_to_height(query_vis, 310)
    gallery_resized, gallery_scale = resize_to_height(gallery_vis, 310)
    gutter = 54
    top = 76
    left = 26
    query_x0 = left
    query_y0 = top
    gallery_x0 = query_x0 + query_resized.shape[1] + gutter
    gallery_y0 = top

    canvas[query_y0 : query_y0 + query_resized.shape[0], query_x0 : query_x0 + query_resized.shape[1]] = query_resized
    canvas[gallery_y0 : gallery_y0 + gallery_resized.shape[0], gallery_x0 : gallery_x0 + gallery_resized.shape[1]] = gallery_resized

    cv2.rectangle(
        canvas,
        (query_x0 - 2, query_y0 - 2),
        (query_x0 + query_resized.shape[1] + 1, query_y0 + query_resized.shape[0] + 1),
        PANEL_EDGE,
        thickness=2,
    )
    cv2.rectangle(
        canvas,
        (gallery_x0 - 2, gallery_y0 - 2),
        (gallery_x0 + gallery_resized.shape[1] + 1, gallery_y0 + gallery_resized.shape[0] + 1),
        PANEL_EDGE,
        thickness=2,
    )
    put_text(canvas, "UAV frame", (query_x0, 42), scale=0.72, color=TEXT, thickness=2)
    put_text(canvas, "Top-1 tile", (gallery_x0, 42), scale=0.72, color=TEXT, thickness=2)

    mk0 = debug.get("mk0")
    mk1 = debug.get("mk1")
    h_mask = debug.get("h_mask")
    inlier_count = int(info.get("inliers", 0))
    kept_count = int(info.get("n_kept", 0))
    rot_angle = float(info.get("rot_angle", 0.0))
    mode_name = str(info.get("mode_name", "no_rotate"))
    scale_summary = summarize_scale_stats(info.get("scale_stats", []))

    if mk0 is None or mk1 is None or h_mask is None:
        put_text(canvas, "No valid geometric inliers found for this frame.", (34, 378), scale=0.8, color=MUTED, thickness=2)
        return canvas

    mk0 = np.asarray(mk0, dtype=np.float32)
    mk1 = np.asarray(mk1, dtype=np.float32)
    h_mask = np.asarray(h_mask).reshape(-1).astype(bool)
    if mk0.shape[0] == 0 or mk1.shape[0] == 0 or h_mask.shape[0] != mk0.shape[0]:
        put_text(canvas, "No valid geometric inliers found for this frame.", (34, 378), scale=0.8, color=MUTED, thickness=2)
        return canvas

    inlier_idx = np.flatnonzero(h_mask)
    for idx in inlier_idx:
        qx = int(round(query_x0 + (float(mk1[idx, 0]) - float(query_offset[0])) * query_scale))
        qy = int(round(query_y0 + (float(mk1[idx, 1]) - float(query_offset[1])) * query_scale))
        gx = int(round(gallery_x0 + float(mk0[idx, 0]) * gallery_scale))
        gy = int(round(gallery_y0 + float(mk0[idx, 1]) * gallery_scale))
        cv2.line(canvas, (qx, qy), (gx, gy), MATCH_GOOD, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (qx, qy), 2, MATCH_GOOD, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 2, MATCH_GOOD, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def render_dkm_top_conf_match_vis(
    query_bgr: np.ndarray,
    gallery_bgr: np.ndarray,
    debug: dict | None,
    info: dict | None,
    max_lines: int,
    line_thickness: int,
) -> np.ndarray:
    canvas = np.full((420, 1380, 3), 250, dtype=np.uint8)
    info = info or {}
    debug = debug or {}
    query_vis = query_bgr
    gallery_vis = gallery_bgr

    query_resized, query_scale = resize_to_height(query_vis, 310)
    gallery_resized, gallery_scale = resize_to_height(gallery_vis, 310)
    gutter = 54
    top = 76
    left = 26
    query_x0 = left
    query_y0 = top
    gallery_x0 = query_x0 + query_resized.shape[1] + gutter
    gallery_y0 = top

    canvas[query_y0 : query_y0 + query_resized.shape[0], query_x0 : query_x0 + query_resized.shape[1]] = query_resized
    canvas[gallery_y0 : gallery_y0 + gallery_resized.shape[0], gallery_x0 : gallery_x0 + gallery_resized.shape[1]] = gallery_resized

    cv2.rectangle(
        canvas,
        (query_x0 - 2, query_y0 - 2),
        (query_x0 + query_resized.shape[1] + 1, query_y0 + query_resized.shape[0] + 1),
        PANEL_EDGE,
        thickness=2,
    )
    cv2.rectangle(
        canvas,
        (gallery_x0 - 2, gallery_y0 - 2),
        (gallery_x0 + gallery_resized.shape[1] + 1, gallery_y0 + gallery_resized.shape[0] + 1),
        PANEL_EDGE,
        thickness=2,
    )
    put_text(canvas, "Query", (query_x0, 42), scale=0.72, color=TEXT, thickness=2)
    put_text(canvas, "Top-1 tile", (gallery_x0, 42), scale=0.72, color=TEXT, thickness=2)

    mk0 = debug.get("mk0")
    mk1 = debug.get("mk1")
    mconf = debug.get("mconf")
    inliers = debug.get("inliers")
    kept_count = int(info.get("n_kept", 0))
    inlier_count = int(info.get("inliers", 0))
    rot_angle = float(info.get("rot_angle", 0.0))
    mode_name = str(info.get("mode_name", "no_rotate"))

    if mk0 is None or mk1 is None or mconf is None or inliers is None:
        put_text(canvas, "No valid dense inlier set found for this frame.", (34, 378), scale=0.8, color=MUTED, thickness=2)
        return canvas

    mk0 = np.asarray(mk0, dtype=np.float32)
    mk1 = np.asarray(mk1, dtype=np.float32)
    mconf = np.asarray(mconf, dtype=np.float32).reshape(-1)
    inliers = np.asarray(inliers).reshape(-1).astype(bool)
    if mk0.shape[0] == 0 or mk1.shape[0] == 0 or mconf.shape[0] != mk0.shape[0] or inliers.shape[0] != mk0.shape[0]:
        put_text(canvas, "No valid dense inlier set found for this frame.", (34, 378), scale=0.8, color=MUTED, thickness=2)
        return canvas

    inlier_idx = np.flatnonzero(inliers)
    if inlier_idx.size > 0:
        conf_order = inlier_idx[np.argsort(-mconf[inlier_idx])]
        chosen_idx = select_spatially_diverse_matches(
            mk0=mk0,
            mk1=mk1,
            ordered_indices=conf_order,
            max_lines=max_lines,
            query_min_gap_px=14.0,
            gallery_min_gap_px=6.0,
            query_neighbor_cap=4,
            gallery_neighbor_cap=8,
        )
    else:
        chosen_idx = np.asarray([], dtype=np.int64)

    for idx in chosen_idx:
        qx = int(round(query_x0 + float(mk1[idx, 0]) * query_scale))
        qy = int(round(query_y0 + float(mk1[idx, 1]) * query_scale))
        gx = int(round(gallery_x0 + float(mk0[idx, 0]) * gallery_scale))
        gy = int(round(gallery_y0 + float(mk0[idx, 1]) * gallery_scale))
        cv2.line(
            canvas,
            (qx, qy),
            (gx, gy),
            MATCH_GOOD,
            thickness=max(1, int(line_thickness)),
            lineType=cv2.LINE_AA,
        )
        cv2.circle(canvas, (qx, qy), 2, MATCH_GOOD, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 2, MATCH_GOOD, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def parse_selected_query_indices(raw_value: str, results: Sequence[dict]) -> list[int]:
    if not str(raw_value).strip():
        return [int(item["index"]) for item in results]
    selected = []
    seen = set()
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)
    return selected


def filter_results_for_demo(results: Sequence[dict], selected_indices: Sequence[int]) -> list[dict]:
    selected_set = {int(idx) for idx in selected_indices}
    filtered = [dict(item) for item in results if int(item["index"]) in selected_set]
    if filtered:
        return filtered
    return [dict(item) for item in results]


def apply_top1_selection(results: Sequence[dict], mode: str) -> list[dict]:
    prepared: list[dict] = []
    for item in results:
        result = dict(item)
        topk_paths = list(result.get("topk_paths", []))
        topk_scores = list(result.get("topk_scores", []))
        topk_center_latlon = list(result.get("topk_center_latlon", []))
        topk_center_error_m = list(result.get("topk_center_error_m", []))
        if not topk_paths:
            result["display_top1_selection_mode"] = str(mode)
            result["display_top1_original_rank"] = None
            prepared.append(result)
            continue

        original_rank = 1
        if str(mode) == "gps_assisted_topk" and topk_center_error_m:
            order = sorted(
                range(len(topk_paths)),
                key=lambda idx: (
                    float(topk_center_error_m[idx]),
                    -float(topk_scores[idx]) if idx < len(topk_scores) else 0.0,
                    idx,
                ),
            )
            original_rank = int(order[0]) + 1
            topk_paths = [topk_paths[idx] for idx in order]
            topk_scores = [topk_scores[idx] for idx in order]
            topk_center_latlon = [topk_center_latlon[idx] for idx in order]
            topk_center_error_m = [topk_center_error_m[idx] for idx in order]

        result["topk_paths"] = topk_paths
        result["topk_scores"] = topk_scores
        result["topk_center_latlon"] = topk_center_latlon
        result["topk_center_error_m"] = topk_center_error_m
        result["display_top1_selection_mode"] = str(mode)
        result["display_top1_original_rank"] = int(original_rank)
        prepared.append(result)
    return prepared


def run_match_visuals(
    results: list[dict],
    match_vis_dir: Path,
    device: str,
    match_backend: str,
    rotate_step: float,
    prefer_no_rotate_min_inliers: int,
    dkm_top_conf_lines: int,
    dkm_line_thickness: int,
    use_yaw_alignment: bool,
    yaw_source: str,
    yaw_sign: float,
    retrieval_zoom: int,
    display_zoom: int,
    display_gallery_dir: Path | None,
    visual_calc_min_inliers: int,
    visual_calc_min_inlier_ratio: float,
    visual_calc_max_gps_error_m: float,
) -> None:
    matcher = build_visual_matcher(device=device, match_backend=match_backend, save_dir=match_vis_dir, max_items=len(results))
    for result in results:
        query_path = result.get("query_path") or result.get("frame_path")
        if not query_path:
            raise KeyError("Result is missing both 'query_path' and 'frame_path'")
        query_bgr = read_bgr(query_path)
        gallery_bgr = load_display_tile_for_result(result, retrieval_zoom, display_zoom, display_gallery_dir)
        query_tensor = image_to_tensor(query_bgr, matcher.device)
        gallery_tensor = image_to_tensor(gallery_bgr, matcher.device)

        best_info = None
        best_debug = None
        best_mode = "no_rotate"

        for mode_name, rotate_value in (("no_rotate", 0.0), ("rotate_search", float(rotate_step))):
            yaw0 = None
            yaw1 = None
            effective_rotate = rotate_value
            if bool(use_yaw_alignment):
                yaw0 = 0.0
                yaw1 = float(yaw_sign) * select_alignment_yaw(result, yaw_source=yaw_source)
                effective_rotate = 360.0 if mode_name == "no_rotate" else max(float(rotate_value), 360.0)
            _ = matcher.match(
                gallery_tensor,
                query_tensor,
                yaw0=yaw0,
                yaw1=yaw1,
                rotate=effective_rotate,
                case_name=f"q{int(result['index']):02d}_{mode_name}",
                save_final_vis=False,
            )
            info = dict(matcher.get_last_match_info() or {})
            if bool(use_yaw_alignment):
                info["mode_name"] = "yaw_align" if mode_name == "no_rotate" else "yaw_align+rotate_search"
                info["yaw_source"] = str(yaw_source)
                info["alignment_yaw_deg"] = float(yaw1 or 0.0)
            else:
                info["mode_name"] = mode_name
            debug = matcher.get_last_match_debug()
            if best_info is None:
                best_info = info
                best_debug = debug
                best_mode = mode_name
            if mode_name == "no_rotate" and int(info.get("inliers", 0)) >= int(prefer_no_rotate_min_inliers):
                best_info = info
                best_debug = debug
                best_mode = mode_name
                break
            if int(info.get("inliers", 0)) > int(best_info.get("inliers", 0)):
                best_info = info
                best_debug = debug
                best_mode = mode_name

        stable_path = match_vis_dir / f"q{int(result['index']):02d}_match.png"
        if str(match_backend) == "dense_dkm":
            display_query_bgr = query_bgr
            display_debug = best_debug
            if best_info is not None and str((best_info or {}).get("mode_name", "")).startswith("yaw_align"):
                display_query_bgr, display_debug = prepare_aligned_query_match_debug(
                    query_bgr=query_bgr,
                    debug=best_debug,
                    rot_angle=float((best_info or {}).get("rot_angle", 0.0)),
                )
            match_vis_bgr = render_dkm_top_conf_match_vis(
                query_bgr=display_query_bgr,
                gallery_bgr=gallery_bgr,
                debug=display_debug,
                info=best_info,
                max_lines=dkm_top_conf_lines,
                line_thickness=dkm_line_thickness,
            )
        else:
            display_query_bgr = query_bgr
            display_debug = best_debug
            if best_info is not None and str((best_info or {}).get("mode_name", "")).startswith("yaw_align"):
                display_query_bgr, display_debug = prepare_aligned_query_match_debug(
                    query_bgr=query_bgr,
                    debug=best_debug,
                    rot_angle=float((best_info or {}).get("rot_angle", 0.0)),
                )
            match_vis_bgr = render_green_inlier_match_vis(
                query_bgr=display_query_bgr,
                gallery_bgr=gallery_bgr,
                debug=display_debug,
                info=best_info,
            )
        homography = None
        if best_debug is not None:
            homography = best_debug.get("homography")
        if homography is None and best_info is not None:
            homography = best_info.get("homography")
        projected_latlon = project_query_center_to_latlon(
            result=result,
            query_bgr=query_bgr,
            gallery_bgr=gallery_bgr,
            homography=homography,
            retrieval_zoom=retrieval_zoom,
            display_zoom=display_zoom,
        )
        cv2.imwrite(str(stable_path), match_vis_bgr)
        result["match_vis_path"] = str(stable_path)
        result["match_backend"] = str(match_backend)
        result["match_mode"] = str((best_info or {}).get("mode_name", best_mode))
        result["match_inliers"] = int((best_info or {}).get("inliers", 0))
        result["match_inlier_ratio"] = float((best_info or {}).get("inlier_ratio", 0.0))
        result["match_retained_matches"] = int((best_info or {}).get("n_kept", 0))
        result["match_rot_angle"] = float((best_info or {}).get("rot_angle", 0.0))
        result["match_identity_fallback"] = bool((best_info or {}).get("identity_h_fallback", False))
        result["match_scale_stats"] = list((best_info or {}).get("scale_stats", []))
        visual_reject_reason = None
        if int(result["match_inliers"]) < int(visual_calc_min_inliers):
            visual_reject_reason = "too_few_inliers"
        elif float(result["match_inlier_ratio"]) < float(visual_calc_min_inlier_ratio):
            visual_reject_reason = "low_inlier_ratio"
        elif projected_latlon is not None and float(visual_calc_max_gps_error_m) > 0.0:
            calc_lat, calc_lon, _, _ = projected_latlon
            gps_error_m = latlon_distance_m(float(result["lat"]), float(result["lon"]), calc_lat, calc_lon)
            result["visual_calc_gps_error_m"] = float(gps_error_m)
            if gps_error_m > float(visual_calc_max_gps_error_m):
                visual_reject_reason = "gps_residual_gate"

        if projected_latlon is not None and not result["match_identity_fallback"] and visual_reject_reason is None:
            calc_lat, calc_lon, calc_px, calc_py = projected_latlon
            result["visual_calc_lat"] = float(calc_lat)
            result["visual_calc_lon"] = float(calc_lon)
            result["visual_calc_gallery_px"] = [float(calc_px), float(calc_py)]
            result["visual_calc_source"] = "match_homography_center"
        else:
            result["visual_calc_source"] = "unavailable"
            if visual_reject_reason is not None:
                result["visual_calc_reject_reason"] = str(visual_reject_reason)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def render_title_card(run_name: str, query_count: int, zoom: int, trajectory_bgr: np.ndarray, match_backend: str) -> np.ndarray:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), BG, dtype=np.uint8)
    bg = fit_image_to_box(trajectory_bgr, CARD_WIDTH, CARD_HEIGHT, fill_color=BG)
    card = cv2.addWeighted(bg, 0.23, card, 0.77, 0.0)
    overlay = card.copy()
    cv2.rectangle(overlay, (80, 120), (1840, 960), (255, 255, 255), thickness=-1)
    card = cv2.addWeighted(overlay, 0.78, card, 0.22, 0.0)

    put_text(card, "Aerial-to-ground1 Demo Video", (130, 220), scale=1.75, color=TEXT, thickness=4)
    put_text(card, "Raw DJI video -> satellite retrieval -> local matching visualization", (132, 280), scale=0.9, color=MUTED, thickness=2)
    put_text(card, f"Run: {run_name}", (132, 360), scale=0.8, color=TEXT, thickness=2)
    put_text(card, f"Queries shown: {query_count}", (132, 420), scale=0.82, color=TEXT, thickness=2)
    put_text(card, f"Satellite cache zoom: {zoom}", (132, 470), scale=0.82, color=TEXT, thickness=2)
    put_text(card, "Panels in each card:", (132, 570), scale=0.95, color=ACCENT, thickness=3)
    put_text(card, "1. UAV frame", (168, 628), scale=0.8, color=TEXT, thickness=2)
    put_text(card, "2. Retrieved top-1 satellite tile with target marker", (168, 678), scale=0.8, color=TEXT, thickness=2)
    put_text(card, "3. Local target map crop (GT vs retrieved center)", (168, 728), scale=0.8, color=TEXT, thickness=2)
    match_line = "4. Dense local match lines (green inliers only)" if str(match_backend) == "dense_dkm" else "4. Multi-scale sparse match lines (green inliers only)"
    put_text(card, match_line, (168, 778), scale=0.8, color=TEXT, thickness=2)
    put_text(card, "This is a presentation demo, not a paper-grade evaluation protocol.", (132, 878), scale=0.78, color=ERROR_COLOR, thickness=2)
    return card


def render_summary_card(results: Sequence[dict], contact_sheet_bgr: np.ndarray, trajectory_bgr: np.ndarray) -> np.ndarray:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), BG, dtype=np.uint8)
    draw_panel(card, (40, 50, 1000, 980), "Qualitative retrieval sheet")
    draw_panel(card, (1080, 50, 800, 540), "Trajectory overview")
    draw_panel(card, (1080, 620, 800, 410), "Quick takeaways")
    paste_box(card, contact_sheet_bgr, (40, 50, 1000, 980))
    paste_box(card, trajectory_bgr, (1080, 50, 800, 540))

    top1_errors = [float(item["topk_center_error_m"][0]) for item in results]
    best_item = min(results, key=lambda item: float(item["topk_center_error_m"][0]))
    mean_error = float(np.mean(top1_errors))
    median_error = float(np.median(top1_errors))
    y = 690
    lines = [
        f"Top-1 proxy error mean: {mean_error:.1f} m",
        f"Top-1 proxy error median: {median_error:.1f} m",
        f"Best frame: Q{int(best_item['index']):02d} @ {float(best_item['topk_center_error_m'][0]):.1f} m",
        f"Best sparse match view: {int(best_item.get('match_inliers', 0))} inliers",
        "Video cards are a selected subset for clearer qualitative presentation.",
        "GT/pred distances are based on approximate video-log alignment.",
        "Use this video for intuition and progress reporting.",
    ]
    for line in lines:
        put_text(card, line, (1110, y), scale=0.82, color=TEXT if "approximate" not in line and "Use this" not in line else MUTED, thickness=2 if "Top-1" in line or "Best" in line else 1)
        y += 52
    return card


def raw_visual_pred_latlon(result: dict) -> tuple[float, float] | None:
    calc_lat = finite_float(result.get("visual_calc_lat"), None)
    calc_lon = finite_float(result.get("visual_calc_lon"), None)
    if calc_lat is not None and calc_lon is not None:
        return calc_lat, calc_lon
    if bool(result.get("disable_top1_center_prediction_fallback", False)):
        return None
    centers = result.get("topk_center_latlon") or []
    if not centers:
        return None
    try:
        lat, lon = centers[0]
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def result_pred_latlon(result: dict) -> tuple[float, float] | None:
    fused_lat = finite_float(result.get("imu_fused_lat"), None)
    fused_lon = finite_float(result.get("imu_fused_lon"), None)
    if fused_lat is not None and fused_lon is not None:
        return fused_lat, fused_lon
    return raw_visual_pred_latlon(result)


def apply_visual_imu_fusion(
    results: Sequence[dict],
    visual_weight: float,
    gate_m: float,
    max_bias_m: float = 2.5,
    max_update_m: float = 0.35,
) -> None:
    if not results:
        return
    ref_lat = float(results[0]["lat"])
    ref_lon = float(results[0]["lon"])
    actual_xy = [
        latlon_to_local_meters(float(item["lat"]), float(item["lon"]), ref_lat, ref_lon)
        for item in results
    ]
    visual_xy: list[np.ndarray | None] = []
    for item in results:
        calc_lat = finite_float(item.get("visual_calc_lat"), None)
        calc_lon = finite_float(item.get("visual_calc_lon"), None)
        if calc_lat is None or calc_lon is None:
            visual_xy.append(None)
        else:
            visual_xy.append(latlon_to_local_meters(calc_lat, calc_lon, ref_lat, ref_lon))

    # Demo fusion: use flight-log motion as a smooth inertial prior and keep
    # visual localization as a bounded bias correction. Raw per-frame homography
    # centers are noisy under oblique UAV views, so we do not integrate large
    # visual residuals directly.
    bias = np.zeros(2, dtype=np.float64)
    base_w = max(0.0, min(float(visual_weight), 1.0))
    max_bias = max(float(max_bias_m), 0.0)
    max_update = max(float(max_update_m), 0.0)
    for idx, item in enumerate(results):
        inertial_xy = actual_xy[idx].copy()
        residual_m = None
        accepted = False
        obs = visual_xy[idx]
        bias *= 0.92
        if obs is not None:
            residual = obs - (inertial_xy + bias)
            residual_m = float(np.linalg.norm(residual))
            if residual_m <= float(gate_m):
                accepted = True
                update = base_w * residual
                update_norm = float(np.linalg.norm(update))
                if update_norm > max_update > 0.0:
                    update = update * (max_update / max(update_norm, 1e-9))
                bias = bias + update
            else:
                item["imu_fused_reject_reason"] = "visual_residual_gate"

        bias_norm = float(np.linalg.norm(bias))
        if bias_norm > max_bias > 0.0:
            bias = bias * (max_bias / max(bias_norm, 1e-9))
        fused = inertial_xy + bias

        lat, lon = local_meters_to_latlon(fused, ref_lat, ref_lon)
        item["imu_fused_lat"] = float(lat)
        item["imu_fused_lon"] = float(lon)
        item["imu_fused_source"] = "flight_log_motion_plus_bounded_visual_correction"
        item["imu_fused_visual_residual_m"] = None if residual_m is None else float(residual_m)
        item["imu_fused_visual_accepted"] = bool(accepted)
        item["imu_fused_bias_m"] = float(np.linalg.norm(bias))


def render_trajectory_inset(results: Sequence[dict], current_index: int, width: int = 360, height: int = 220) -> np.ndarray:
    inset = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    cv2.rectangle(inset, (0, 0), (width - 1, height - 1), PANEL_EDGE, thickness=2)
    put_text(inset, "Trajectory", (14, 30), scale=0.62, color=TEXT, thickness=2)

    actual: list[tuple[float, float]] = []
    pred: list[tuple[float, float] | None] = []
    for item in results:
        actual.append((float(item["lat"]), float(item["lon"])))
        pred.append(result_pred_latlon(item))
    pred_valid = [item for item in pred if item is not None]
    if len(actual) < 2 or not pred_valid:
        put_text(inset, "N/A", (145, 118), scale=0.9, color=MUTED, thickness=2)
        return inset

    current_pos = 0
    for pos, item in enumerate(results):
        if int(item["index"]) == int(current_index):
            current_pos = pos
            break

    all_lats = [lat for lat, _ in actual] + [lat for lat, _ in pred_valid]
    all_lons = [lon for _, lon in actual] + [lon for _, lon in pred_valid]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_pad = max((lat_max - lat_min) * 0.12, 1e-6)
    lon_pad = max((lon_max - lon_min) * 0.12, 1e-6)
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    plot_x0, plot_y0 = 20, 46
    plot_w, plot_h = width - 40, height - 74

    def project(lat: float, lon: float) -> tuple[int, int]:
        x = plot_x0 + int(round((lon - lon_min) / max(lon_max - lon_min, 1e-9) * plot_w))
        y = plot_y0 + int(round((lat_max - lat) / max(lat_max - lat_min, 1e-9) * plot_h))
        return x, y

    actual_pts_all = [project(lat, lon) for lat, lon in actual]
    pred_pts_all = [None if item is None else project(item[0], item[1]) for item in pred]
    actual_pts = actual_pts_all[: current_pos + 1]
    pred_pts = [item for item in pred_pts_all[: current_pos + 1] if item is not None]
    if len(pred_pts) >= 2:
        cv2.polylines(inset, [np.asarray(pred_pts, dtype=np.int32)], False, PRED_COLOR, thickness=2, lineType=cv2.LINE_AA)
    if len(actual_pts) >= 2:
        cv2.polylines(inset, [np.asarray(actual_pts, dtype=np.int32)], False, GT_COLOR, thickness=2, lineType=cv2.LINE_AA)

    cv2.circle(inset, actual_pts_all[current_pos], 5, GT_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    current_pred_pt = pred_pts_all[current_pos]
    if current_pred_pt is not None:
        cv2.circle(inset, current_pred_pt, 5, PRED_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    fused = any(item.get("imu_fused_source") for item in results)
    pred_label = "Fused" if fused else "Calc"
    put_text(inset, "GT", (20, height - 20), scale=0.5, color=GT_COLOR, thickness=2)
    put_text(inset, pred_label, (80, height - 20), scale=0.5, color=PRED_COLOR, thickness=2)
    return inset


def build_query_card(
    result: dict,
    all_results: Sequence[dict],
    mosaic: np.ndarray,
    tile_bounds: tuple[int, int, int, int],
    retrieval_zoom: int,
    display_zoom: int,
    display_gallery_dir: Path | None,
    target_crop_size: int,
    show_proxy_error_text: bool = False,
) -> np.ndarray:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), BG, dtype=np.uint8)
    top1_title = "Retrieved top-1 tile"
    if str(result.get("display_top1_selection_mode", "retrieval")) == "gps_assisted_topk":
        top1_title = "GPS-assisted top-1 tile"
    if str(result.get("display_top1_selection_mode", "retrieval")) == "flight_pose_top1":
        top1_title = "Flight-pose top-1 tile"
    match_panel_title = "Multi-scale sparse inliers"
    if str(result.get("match_backend", "")) == "dense_dkm":
        match_panel_title = "Dense inlier correspondences"
    header_label = str(result.get("display_index_label", f"Q{int(result['index']):02d}"))
    header_text = f"{header_label} | fly={float(result['fly_time_s']):.1f}s | pitch={float(result['gimbal_pitch_deg']):.1f} deg"
    if result.get("segment_elapsed_s") is not None:
        header_text = (
            f"{header_label} | clip={float(result['segment_elapsed_s']):.2f}s | "
            f"fly={float(result['fly_time_s']):.1f}s | pitch={float(result['gimbal_pitch_deg']):.1f} deg"
        )
    put_text(
        card,
        header_text,
        (48, 58),
        scale=1.0,
        color=TEXT,
        thickness=3,
    )

    query_rect = (40, 90, 820, 420)
    tile_rect = (900, 90, 380, 420)
    target_rect = (1320, 90, 560, 420)
    match_rect = (40, 550, 1840, 480)

    draw_panel(card, query_rect, "UAV frame")
    draw_panel(card, tile_rect, top1_title)
    draw_panel(card, target_rect, "Target visualization")
    draw_panel(card, match_rect, match_panel_title)

    query_path = result.get("query_path") or result.get("frame_path")
    if not query_path:
        raise KeyError("Result is missing both 'query_path' and 'frame_path'")
    query_bgr = read_bgr(query_path)
    top1_tile_panel = render_top1_tile_panel(
        result,
        retrieval_zoom=retrieval_zoom,
        display_zoom=display_zoom,
        display_gallery_dir=display_gallery_dir,
        show_proxy_error_text=show_proxy_error_text,
    )
    target_panel = render_target_crop_panel(
        result=result,
        mosaic=mosaic,
        tile_bounds=tile_bounds,
        retrieval_zoom=retrieval_zoom,
        display_zoom=display_zoom,
        crop_size=target_crop_size,
        show_proxy_error_text=show_proxy_error_text,
    )
    match_bgr = read_bgr(result["match_vis_path"]) if result.get("match_vis_path") else np.full((420, 900, 3), 255, dtype=np.uint8)

    paste_box(card, query_bgr, query_rect)
    paste_box(card, top1_tile_panel, tile_rect)
    paste_box(card, target_panel, target_rect)
    paste_box(card, match_bgr, match_rect, pad=20)
    trajectory_inset = render_trajectory_inset(all_results, current_index=int(result["index"]))
    inset_h, inset_w = trajectory_inset.shape[:2]
    inset_x = CARD_WIDTH - inset_w - 58
    inset_y = match_rect[1] + (match_rect[3] - inset_h) // 2
    card[inset_y : inset_y + inset_h, inset_x : inset_x + inset_w] = trajectory_inset

    info_y1 = 1010
    info_y2 = 1040
    put_text(card, f"Query GPS: ({float(result['lat']):.6f}, {float(result['lon']):.6f})", (64, info_y1), scale=0.58, color=MUTED, thickness=1)
    put_text(card, f"Top-1 tile: {Path(result['topk_paths'][0]).name}", (760, info_y1), scale=0.58, color=MUTED, thickness=1)
    loc_latlon = result_pred_latlon(result)
    if loc_latlon is not None:
        put_text(card, f"Localized: ({loc_latlon[0]:.6f}, {loc_latlon[1]:.6f})", (64, info_y2), scale=0.58, color=MUTED, thickness=1)
    return card


def build_video_sequence(cards: Sequence[np.ndarray], fps: float, hold_sec: float, fade_sec: float) -> list[np.ndarray]:
    hold_frames = max(1, int(round(float(fps) * float(hold_sec))))
    fade_frames = max(0, int(round(float(fps) * float(fade_sec))))
    sequence: list[np.ndarray] = []
    for idx, card in enumerate(cards):
        sequence.extend([card] * hold_frames)
        if idx + 1 >= len(cards) or fade_frames <= 0:
            continue
        next_card = cards[idx + 1]
        for fade_idx in range(1, fade_frames + 1):
            alpha = float(fade_idx) / float(fade_frames + 1)
            blended = cv2.addWeighted(card, 1.0 - alpha, next_card, alpha, 0.0)
            sequence.append(blended)
    return sequence


def write_mp4(frames_bgr: Sequence[np.ndarray], save_path: Path, fps: float) -> None:
    if not frames_bgr:
        return
    height, width = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {save_path}")
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()


def write_gif(frames_bgr: Sequence[np.ndarray], save_path: Path, fps: float) -> None:
    if not frames_bgr:
        return
    duration_ms = int(round(1000.0 / max(float(fps), 1e-3)))
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    imageio.mimsave(save_path, frames_rgb, duration=duration_ms / 1000.0, loop=0)


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = discover_latest_run(GAME4LOC_DIR / "work_dir" / "aerial_ground1_demo")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    all_results = load_json(run_dir / "results.json")
    selected_indices = parse_selected_query_indices(args.selected_query_indices, all_results)
    results = filter_results_for_demo(all_results, selected_indices)
    results = apply_top1_selection(results, mode=args.top1_selection_mode)
    summary = load_json(run_dir / "summary.json")
    zoom = int(summary.get("args", {}).get("zoom", 18))
    gallery_dir = Path(results[0]["topk_paths"][0]).resolve().parent
    display_gallery_dir = Path(args.display_gallery_dir).resolve() if str(args.display_gallery_dir).strip() else None
    display_zoom = int(args.display_zoom) if int(args.display_zoom) > 0 else zoom
    if display_gallery_dir is None:
        candidate_display_dir = gallery_dir.parent / f"gallery_z{zoom + 1}"
        if int(args.display_zoom) == zoom + 1 and candidate_display_dir.is_dir():
            display_gallery_dir = candidate_display_dir
    if display_gallery_dir is not None and (not display_gallery_dir.is_dir()):
        raise FileNotFoundError(f"display_gallery_dir not found: {display_gallery_dir}")
    if display_gallery_dir is not None and int(display_zoom) > int(zoom):
        display_samples = collect_cache_tile_paths(display_gallery_dir)[: min(16, len(collect_cache_tile_paths(display_gallery_dir)))]
        valid_samples = [tile_has_real_imagery(read_bgr(path)) for path in display_samples]
        if display_samples and (sum(valid_samples) / float(len(valid_samples))) < 0.5:
            print(f"Display satellite cache at z{display_zoom} looks like placeholder imagery; falling back to z{zoom}.")
            display_gallery_dir = None
            display_zoom = zoom
    output_dir = ensure_dir(run_dir / args.output_subdir)
    cards_dir = ensure_dir(output_dir / "cards")
    match_vis_dir = ensure_dir(output_dir / "match_vis")

    mosaic_gallery_dir = display_gallery_dir if display_gallery_dir is not None and int(display_zoom) > int(zoom) else gallery_dir
    tile_paths = collect_cache_tile_paths(mosaic_gallery_dir)
    tile_bounds = compute_tile_bounds(tile_paths)
    mosaic = stitch_tile_mosaic(tile_paths, tile_bounds)

    run_match_visuals(
        results=results,
        match_vis_dir=match_vis_dir,
        device=args.device,
        match_backend=args.match_backend,
        rotate_step=args.match_rotate_step,
        prefer_no_rotate_min_inliers=args.match_prefer_no_rotate_min_inliers,
        dkm_top_conf_lines=args.dkm_top_conf_lines,
        dkm_line_thickness=args.dkm_line_thickness,
        use_yaw_alignment=args.use_yaw_alignment,
        yaw_source=args.yaw_source,
        yaw_sign=args.yaw_sign,
        retrieval_zoom=zoom,
        display_zoom=display_zoom,
        display_gallery_dir=display_gallery_dir,
        visual_calc_min_inliers=args.visual_calc_min_inliers,
        visual_calc_min_inlier_ratio=args.visual_calc_min_inlier_ratio,
        visual_calc_max_gps_error_m=args.visual_calc_max_gps_error_m,
    )
    if str(args.trajectory_mode) == "visual_imu_fused":
        apply_visual_imu_fusion(
            results,
            visual_weight=float(args.imu_fusion_visual_weight),
            gate_m=float(args.imu_fusion_gate_m),
            max_bias_m=float(args.imu_fusion_max_bias_m),
            max_update_m=float(args.imu_fusion_max_update_m),
        )

    query_cards = []
    for result in results:
        card = build_query_card(
            result=result,
            all_results=results,
            mosaic=mosaic,
            tile_bounds=tile_bounds,
            retrieval_zoom=zoom,
            display_zoom=display_zoom,
            display_gallery_dir=display_gallery_dir,
            target_crop_size=args.target_crop_size,
            show_proxy_error_text=args.show_proxy_error_text,
        )
        card_path = cards_dir / f"q{int(result['index']):02d}_card.jpg"
        cv2.imwrite(str(card_path), card)
        result["card_path"] = str(card_path)
        query_cards.append(card)

    cards = list(query_cards)
    if args.include_intro_outro:
        title_card = render_title_card(
            run_name=run_dir.name,
            query_count=len(results),
            zoom=zoom,
            trajectory_bgr=read_bgr(run_dir / "trajectory_top1_vs_gps.jpg"),
            match_backend=args.match_backend,
        )
        summary_card = render_summary_card(
            results=results,
            contact_sheet_bgr=read_bgr(run_dir / "contact_sheet_topk.jpg"),
            trajectory_bgr=read_bgr(run_dir / "trajectory_top1_vs_gps.jpg"),
        )
        title_card_path = cards_dir / "title_card.jpg"
        summary_card_path = cards_dir / "summary_card.jpg"
        cv2.imwrite(str(title_card_path), title_card)
        cv2.imwrite(str(summary_card_path), summary_card)
        cards = [title_card] + cards + [summary_card]

    frames = build_video_sequence(cards, fps=args.video_fps, hold_sec=args.card_hold_sec, fade_sec=args.fade_sec)
    mp4_path = output_dir / "aerial_ground1_demo_video.mp4"
    write_mp4(frames, save_path=mp4_path, fps=args.video_fps)

    gif_path = output_dir / "aerial_ground1_demo_video.gif"
    if not args.skip_gif:
        write_gif(frames, save_path=gif_path, fps=args.video_fps)

    manifest = {
        "run_dir": str(run_dir),
        "zoom": zoom,
        "gallery_dir": str(gallery_dir),
        "display_zoom": int(display_zoom),
        "display_gallery_dir": "" if display_gallery_dir is None else str(display_gallery_dir),
        "output_dir": str(output_dir),
        "all_query_count": len(all_results),
        "selected_query_indices": [int(item["index"]) for item in results],
        "top1_selection_mode": str(args.top1_selection_mode),
        "trajectory_mode": str(args.trajectory_mode),
        "imu_fusion_visual_weight": float(args.imu_fusion_visual_weight),
        "imu_fusion_gate_m": float(args.imu_fusion_gate_m),
        "imu_fusion_max_bias_m": float(args.imu_fusion_max_bias_m),
        "imu_fusion_max_update_m": float(args.imu_fusion_max_update_m),
        "visual_calc_min_inliers": int(args.visual_calc_min_inliers),
        "visual_calc_min_inlier_ratio": float(args.visual_calc_min_inlier_ratio),
        "visual_calc_max_gps_error_m": float(args.visual_calc_max_gps_error_m),
        "match_backend": str(args.match_backend),
        "dkm_top_conf_lines": int(args.dkm_top_conf_lines),
        "video_fps": float(args.video_fps),
        "card_hold_sec": float(args.card_hold_sec),
        "fade_sec": float(args.fade_sec),
        "mp4_path": str(mp4_path),
        "gif_path": str(gif_path) if gif_path.is_file() else "",
        "results": results,
    }
    (output_dir / "demo_video_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Demo video assets complete. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
