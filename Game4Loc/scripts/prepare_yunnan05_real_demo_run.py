#!/usr/bin/env python3
"""Prepare a RealDemo-compatible run directory for UAV_VisLoc 05_Yunnan.

This is intentionally a demo helper, not a benchmark evaluator. It selects the
displayed top-1 satellite tile from query GPS metadata, matching the RealDemo
track's "flight-pose top-1" convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


TILE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "UAV_VisLoc_dataset" / "05_Yunnan"),
        help="05_Yunnan directory containing query/ and map/google/17/.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "work_dir" / "yunnan05_real_demo"),
        help="Directory where the RealDemo-compatible run directory is written.",
    )
    parser.add_argument("--start_index", type=int, default=250, help="First query number, inclusive.")
    parser.add_argument("--count", type=int, default=10, help="Number of consecutive queries.")
    parser.add_argument("--zoom", type=int, default=17, help="Satellite tile zoom level.")
    parser.add_argument("--topk", type=int, default=5, help="Number of GPS-nearest tiles to store.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_by_filename(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {row["Filename"]: row for row in csv.DictReader(f)}


def read_map_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tile_center(row: dict[str, str]) -> tuple[float, float]:
    lat = (float(row["Top_left_lat"]) + float(row["Bottom_right_lat"])) * 0.5
    lon = (float(row["Top_left_lon"]) + float(row["Bottom_right_long"])) * 0.5
    return lat, lon


def contains_latlon(row: dict[str, str], lat: float, lon: float) -> bool:
    top = float(row["Top_left_lat"])
    bottom = float(row["Bottom_right_lat"])
    left = float(row["Top_left_lon"])
    right = float(row["Bottom_right_long"])
    return min(top, bottom) <= lat <= max(top, bottom) and min(left, right) <= lon <= max(left, right)


def latlon_distance_m(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    earth_radius_m = 6378137.0
    x = math.radians(lon1 - lon0) * earth_radius_m * math.cos(math.radians(lat0))
    y = math.radians(lat1 - lat0) * earth_radius_m
    return float(math.hypot(x, y))


def normalized_tile_name(row: dict[str, str]) -> str:
    return f"{int(row['TileX'])}_{int(row['TileY'])}.jpg"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def fit_image(image: np.ndarray, width: int, height: int, fill: tuple[int, int, int] = (244, 246, 248)) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / float(w), height / float(h))
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def put_text(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (28, 32, 38), 1, lineType=cv2.LINE_AA)


def write_contact_sheet(results: list[dict], save_path: Path, display_topk: int = 3) -> None:
    cell_w, cell_h = 260, 230
    rows = []
    for result in results:
        cells = []
        query = cv2.imread(result["query_path"], cv2.IMREAD_COLOR)
        cells.append(fit_image(query, cell_w, cell_h - 34))
        for tile_path in result["topk_paths"][:display_topk]:
            tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
            cells.append(fit_image(tile, cell_w, cell_h - 34))
        row = np.full((cell_h, cell_w * len(cells), 3), 255, dtype=np.uint8)
        labels = ["Query"] + [f"Top-{idx + 1}" for idx in range(len(cells) - 1)]
        for idx, (cell, label) in enumerate(zip(cells, labels)):
            x0 = idx * cell_w
            row[0 : cell_h - 34, x0 : x0 + cell_w] = cell
            put_text(row, label, (x0 + 10, cell_h - 12))
        rows.append(row)
    cv2.imwrite(str(save_path), np.concatenate(rows, axis=0))


def write_trajectory(results: list[dict], save_path: Path) -> None:
    width, height = 960, 540
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    lats = np.asarray([float(item["lat"]) for item in results], dtype=np.float64)
    lons = np.asarray([float(item["lon"]) for item in results], dtype=np.float64)
    pred = np.asarray([item["topk_center_latlon"][0] for item in results], dtype=np.float64)
    all_lat = np.concatenate([lats, pred[:, 0]])
    all_lon = np.concatenate([lons, pred[:, 1]])
    lat_span = max(float(all_lat.max() - all_lat.min()), 1e-9)
    lon_span = max(float(all_lon.max() - all_lon.min()), 1e-9)
    margin = 60

    def to_xy(lat: float, lon: float) -> tuple[int, int]:
        x = margin + int(round((lon - all_lon.min()) / lon_span * (width - 2 * margin)))
        y = height - margin - int(round((lat - all_lat.min()) / lat_span * (height - 2 * margin)))
        return x, y

    gt_pts = np.asarray([to_xy(float(lat), float(lon)) for lat, lon in zip(lats, lons)], dtype=np.int32)
    pred_pts = np.asarray([to_xy(float(lat), float(lon)) for lat, lon in pred], dtype=np.int32)
    cv2.polylines(canvas, [gt_pts], False, (48, 118, 237), 3, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pred_pts], False, (46, 204, 113), 3, lineType=cv2.LINE_AA)
    for point in gt_pts:
        cv2.circle(canvas, tuple(point), 5, (48, 118, 237), -1, lineType=cv2.LINE_AA)
    for point in pred_pts:
        cv2.circle(canvas, tuple(point), 5, (46, 204, 113), -1, lineType=cv2.LINE_AA)
    put_text(canvas, "GT query GPS", (40, 42), 0.8)
    cv2.circle(canvas, (230, 36), 6, (48, 118, 237), -1, lineType=cv2.LINE_AA)
    put_text(canvas, "GPS-nearest tile center", (280, 42), 0.8)
    cv2.circle(canvas, (590, 36), 6, (46, 204, 113), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(save_path), canvas)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    query_dir = data_root / "query"
    map_dir = data_root / "map" / "google" / str(args.zoom)
    metadata_path = query_dir / "photo_metadata.csv"
    map_path = map_dir / "map.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not map_path.is_file():
        raise FileNotFoundError(map_path)

    metadata = read_csv_by_filename(metadata_path)
    map_rows = read_map_rows(map_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root).resolve() / f"yunnan05_q{args.start_index:04d}_{args.count}f_{timestamp}")
    gallery_dir = ensure_dir(run_dir / f"gallery_z{int(args.zoom)}")

    for row in map_rows:
        src = map_dir / row["Filename"]
        dst = gallery_dir / normalized_tile_name(row)
        link_or_copy(src, dst)

    results: list[dict] = []
    for offset, query_number in enumerate(range(int(args.start_index), int(args.start_index) + int(args.count))):
        filename = f"05_{query_number:04d}.JPG"
        query_path = query_dir / filename
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        if filename not in metadata:
            raise KeyError(f"Missing metadata for {filename}")
        meta = metadata[filename]
        lat = float(meta["Latitude"])
        lon = float(meta["Longitude"])
        ranked = sorted(
            map_rows,
            key=lambda row: (
                0 if contains_latlon(row, lat, lon) else 1,
                latlon_distance_m(lat, lon, *tile_center(row)),
            ),
        )[: int(args.topk)]
        topk_paths = [str((gallery_dir / normalized_tile_name(row)).resolve()) for row in ranked]
        topk_centers = [tile_center(row) for row in ranked]
        topk_errors = [latlon_distance_m(lat, lon, center_lat, center_lon) for center_lat, center_lon in topk_centers]
        results.append(
            {
                "index": query_number,
                "display_index_label": filename,
                "query_path": str(query_path.resolve()),
                "frame_path": str(query_path.resolve()),
                "lat": lat,
                "lon": lon,
                "altitude_ft": float(meta.get("Altitude", 0.0) or 0.0) * 3.28084,
                "height_ft": float(meta.get("Altitude", 0.0) or 0.0) * 3.28084,
                "fly_time_s": float(offset),
                "segment_elapsed_s": float(offset),
                "yaw_deg": float(meta.get("Phi1", meta.get("Gimball_Yaw", 0.0)) or 0.0),
                "gimbal_yaw_deg": float(meta.get("Gimball_Yaw", meta.get("Phi1", 0.0)) or 0.0),
                "gimbal_pitch_deg": float(meta.get("Gimball_Pitch", 0.0) or 0.0),
                "topk_paths": topk_paths,
                "topk_scores": [-float(error) for error in topk_errors],
                "topk_center_latlon": [[float(lat), float(lon)] for lat, lon in topk_centers],
                "topk_center_error_m": [float(error) for error in topk_errors],
                "topk_tile_xy": [[int(row["TileX"]), int(row["TileY"])] for row in ranked],
                "top1_source": "query_metadata_gps_nearest_google_z17_tile",
            }
        )

    summary = {
        "run_name": run_dir.name,
        "data_root": str(data_root),
        "query_start": int(args.start_index),
        "query_count": len(results),
        "query_filenames": [item["display_index_label"] for item in results],
        "tile_count": len(map_rows),
        "args": {
            "zoom": int(args.zoom),
            "topk": int(args.topk),
            "top1_selection": "query_metadata_gps_nearest_google_z17_tile",
            "note": "RealDemo helper; not retrieval or benchmark evaluation.",
        },
        "mean_top1_center_error_m": float(np.mean([item["topk_center_error_m"][0] for item in results])),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_contact_sheet(results, run_dir / "contact_sheet_topk.jpg")
    write_trajectory(results, run_dir / "trajectory_top1_vs_gps.jpg")
    print(json.dumps({"run_dir": str(run_dir), "query_count": len(results), "mean_top1_center_error_m": summary["mean_top1_center_error_m"]}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
