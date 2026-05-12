#!/usr/bin/env python3
"""Continuous real-video demo runner for the Aerial-to-ground1 dataset.

This script extracts every frame from a short contiguous DJI video segment,
runs the existing Game4Loc retrieval model on those frames, and exports a run
directory that is compatible with `render_aerial_ground1_demo_video.py`.

Unlike the earlier sampled-frame demo, this path keeps a consecutive clip with
no skipped frames inside the chosen segment so the final report can show a
direct real-video qualitative result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from run_aerial_ground1_demo import (
    DEFAULT_TILE_URL,
    PROJECT_ROOT,
    align_abs_time_to_log_row,
    build_model,
    build_video_infos,
    discover_inputs,
    ensure_dir,
    expand_latlon_bbox,
    extract_features,
    gallery_tile_centers,
    haversine_m,
    nearest_gallery_tile,
    prepare_gallery_tiles,
    read_flight_log,
    retrieve_topk,
    select_video_log_rows,
    tile_grid_from_bbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "Aerial-to-ground1"),
        help="Directory containing the raw DJI videos and DJIFlightRecord CSV.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "work_dir" / "aerial_ground1_demo"),
        help="Directory where demo artifacts will be written.",
    )
    parser.add_argument(
        "--gallery_cache_dir",
        type=str,
        default=None,
        help="Optional persistent cache directory for downloaded satellite tiles.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
        help="Game4Loc retrieval backbone name.",
    )
    parser.add_argument(
        "--checkpoint_start",
        type=str,
        default=str(PROJECT_ROOT / "pretrained" / "visloc" / "vit_base_eva_visloc_same_area_0407.pth"),
        help="Checkpoint used for retrieval inference.",
    )
    parser.add_argument("--img_size", type=int, default=384, help="Model input image size.")
    parser.add_argument("--batch_size", type=int, default=64, help="Feature extraction batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--zoom", type=int, default=18, help="Satellite tile zoom level.")
    parser.add_argument(
        "--tile_margin_m",
        type=float,
        default=120.0,
        help="Extra gallery margin around the flight track in meters.",
    )
    parser.add_argument(
        "--segment_abs_start_sec",
        type=float,
        default=281.45,
        help="Absolute start time in the concatenated videos for the continuous clip.",
    )
    parser.add_argument(
        "--segment_duration_sec",
        type=float,
        default=1.5,
        help="Clip duration in seconds. All frames inside this segment are kept consecutively.",
    )
    parser.add_argument(
        "--query_save_width",
        type=int,
        default=1280,
        help="Resize extracted video frames to this width before saving.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k retrieval count to store.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--tile_url_template",
        type=str,
        default=DEFAULT_TILE_URL,
        help="Satellite tile URL template with {z}, {x}, {y}.",
    )
    parser.add_argument(
        "--force_redownload_tiles",
        action="store_true",
        help="Redownload existing cached tiles.",
    )
    return parser.parse_args()


def maybe_resize(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / float(width)
    target_height = int(round(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_frame_indices(local_start_s: float, duration_s: float, fps: float, frame_count: int) -> List[int]:
    start_frame = int(round(max(local_start_s, 0.0) * fps))
    frame_total = max(1, int(round(max(duration_s, 0.0) * fps)))
    end_frame = min(frame_count, start_frame + frame_total)
    if end_frame <= start_frame:
        end_frame = min(frame_count, start_frame + 1)
    return list(range(start_frame, end_frame))


def extract_consecutive_frames(
    video_path: Path,
    frame_indices: List[int],
    video_abs_start_s: float,
    fps: float,
    total_video_duration_s: float,
    flight_rows,
    query_dir: Path,
    query_save_width: int,
) -> list[dict]:
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")

    ensure_dir(query_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    first_frame = int(frame_indices[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(first_frame))

    samples: list[dict] = []
    try:
        for seq_index, frame_index in enumerate(frame_indices, start=1):
            if frame_index != first_frame + seq_index - 1:
                raise ValueError("frame_indices must be consecutive for sequential extraction")

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")

            frame = maybe_resize(frame, target_width=query_save_width)
            abs_time_s = float(video_abs_start_s + frame_index / fps)
            aligned_row = align_abs_time_to_log_row(
                abs_time_s=abs_time_s,
                video_total_duration_s=total_video_duration_s,
                video_rows=flight_rows,
            )
            segment_elapsed_s = float((frame_index - first_frame) / fps)
            local_time_s = float(frame_index / fps)

            frame_name = (
                f"f{seq_index:04d}_src{frame_index:06d}_"
                f"clip{segment_elapsed_s:05.2f}_"
                f"fly{aligned_row.fly_time_s:07.2f}.jpg"
            )
            frame_path = query_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            samples.append(
                {
                    "index": int(seq_index),
                    "frame_index": int(frame_index),
                    "abs_time_s": abs_time_s,
                    "video_local_time_s": local_time_s,
                    "segment_elapsed_s": segment_elapsed_s,
                    "fly_time_s": float(aligned_row.fly_time_s),
                    "lat": float(aligned_row.lat),
                    "lon": float(aligned_row.lon),
                    "height_ft": float(aligned_row.height_ft),
                    "altitude_ft": float(aligned_row.altitude_ft),
                    "yaw_deg": float(aligned_row.yaw_deg),
                    "gimbal_pitch_deg": float(aligned_row.gimbal_pitch_deg),
                    "gimbal_yaw_deg": float(aligned_row.gimbal_yaw_deg),
                    "video_path": str(video_path),
                    "frame_path": str(frame_path),
                    "display_index_label": f"F{seq_index:03d}",
                }
            )
    finally:
        cap.release()
    return samples


def build_summary(
    run_dir: Path,
    args: argparse.Namespace,
    video_info: dict,
    segment: dict,
    gallery_count: int,
    results: list[dict],
) -> None:
    top1_errors = [float(item["topk_center_error_m"][0]) for item in results]
    topk_best_errors = [float(min(item["topk_center_error_m"])) for item in results]
    summary = {
        "dataset": "Aerial-to-ground1",
        "demo_type": "continuous_clip",
        "note": (
            "This demo keeps every frame inside one selected contiguous clip. "
            "GPS/error numbers are still proxy values because MP4-to-flight-log "
            "alignment is approximated from the video/log time coverage."
        ),
        "args": vars(args),
        "segment": segment,
        "video_info": video_info,
        "query_count": len(results),
        "gallery_count": int(gallery_count),
        "top1_center_error_mean_m": float(np.mean(top1_errors)) if top1_errors else None,
        "top1_center_error_median_m": float(np.median(top1_errors)) if top1_errors else None,
        "topk_best_center_error_mean_m": float(np.mean(topk_best_errors)) if topk_best_errors else None,
        "queries": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Aerial-to-ground1 Continuous Clip Demo",
        "",
        "This run keeps one contiguous real DJI video clip without skipping frames inside the chosen segment.",
        "",
        f"- Source video: `{Path(video_info['path']).name}`",
        f"- Segment absolute start: {segment['requested_abs_start_s']:.3f} s",
        f"- Segment actual local start: {segment['actual_local_start_s']:.3f} s",
        f"- Segment duration: {segment['actual_duration_s']:.3f} s",
        f"- Consecutive frames: {len(results)}",
        f"- Source FPS: {segment['source_fps']:.6f}",
        f"- Gallery tiles (zoom {args.zoom}): {gallery_count}",
        f"- Proxy top-1 center error mean: {summary['top1_center_error_mean_m']:.2f} m" if top1_errors else "- Proxy top-1 center error mean: N/A",
        f"- Proxy top-k best center error mean: {summary['topk_best_center_error_mean_m']:.2f} m" if topk_best_errors else "- Proxy top-k best center error mean: N/A",
        "",
        "## Note",
        "",
        summary["note"],
        "",
        "## Main Files",
        "",
        "- `queries/`: all consecutive extracted frames from the chosen clip",
        "- `results.json`: per-frame retrieval dump",
        "- `summary.json`: machine-readable metadata",
        "",
        "## First/Last Frames",
        "",
        "| Frame | clip time (s) | flyTime (s) | Top-1 error (m) | Top-1 tile |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    preview_items = []
    if results:
        preview_items.extend(results[:2])
        if len(results) > 4:
            preview_items.extend(results[len(results) // 2 - 1: len(results) // 2 + 1])
        if len(results) > 2:
            preview_items.extend(results[-2:])
    seen = set()
    for result in preview_items:
        key = int(result["index"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {result['display_index_label']} | {float(result['segment_elapsed_s']):.2f} | "
            f"{float(result['fly_time_s']):.1f} | {float(result['topk_center_error_m'][0]):.1f} | "
            f"`{Path(result['topk_paths'][0]).name}` |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = time.strftime("aerial_ground1_continuous_demo_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / run_name)
    query_dir = ensure_dir(run_dir / "queries")
    if args.gallery_cache_dir:
        gallery_dir = ensure_dir(Path(args.gallery_cache_dir).resolve())
    else:
        gallery_dir = ensure_dir(output_root / "cache" / f"gallery_z{args.zoom}")

    video_paths, csv_path = discover_inputs(data_root)
    flight_rows = read_flight_log(csv_path)
    video_rows = select_video_log_rows(flight_rows)
    video_infos = build_video_infos(video_paths)
    total_video_duration_s = video_infos[-1].end_time_s

    segment_abs_start_s = float(max(args.segment_abs_start_sec, 0.0))
    segment_duration_s = float(max(args.segment_duration_sec, 1e-3))
    target_video_info = None
    for info in video_infos:
        if info.start_time_s <= segment_abs_start_s < info.end_time_s:
            target_video_info = info
            break
    if target_video_info is None:
        raise ValueError(f"segment_abs_start_sec={segment_abs_start_s:.3f} is outside the available videos")

    local_start_s = segment_abs_start_s - target_video_info.start_time_s
    if local_start_s < 0 or local_start_s >= target_video_info.duration_s:
        raise ValueError("Chosen segment start falls outside the target video")

    frame_indices = build_frame_indices(
        local_start_s=local_start_s,
        duration_s=segment_duration_s,
        fps=float(target_video_info.fps),
        frame_count=int(target_video_info.frame_count),
    )
    samples = extract_consecutive_frames(
        video_path=Path(target_video_info.path).resolve(),
        frame_indices=frame_indices,
        video_abs_start_s=float(target_video_info.start_time_s),
        fps=float(target_video_info.fps),
        total_video_duration_s=float(total_video_duration_s),
        flight_rows=video_rows,
        query_dir=query_dir,
        query_save_width=int(args.query_save_width),
    )
    (run_dir / "query_manifest.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    lat_min, lon_min, lat_max, lon_max = expand_latlon_bbox(flight_rows, margin_m=args.tile_margin_m)
    tile_bounds = tile_grid_from_bbox(lat_min, lon_min, lat_max, lon_max, zoom=args.zoom)
    gallery_paths = prepare_gallery_tiles(
        gallery_dir=gallery_dir,
        zoom=args.zoom,
        tile_bounds=tile_bounds,
        url_template=args.tile_url_template,
        force_redownload=args.force_redownload_tiles,
    )

    model, val_transforms = build_model(args)
    query_paths = [item["frame_path"] for item in samples]
    query_features = extract_features(
        image_paths=query_paths,
        model=model,
        transform=val_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    gallery_features = extract_features(
        image_paths=gallery_paths,
        model=model,
        transform=val_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    top_indices, top_scores = retrieve_topk(query_features=query_features, gallery_features=gallery_features, topk=args.topk)
    centers = gallery_tile_centers(gallery_paths=gallery_paths, zoom=args.zoom)

    results = []
    for sample, sample_indices, sample_scores in zip(samples, top_indices, top_scores):
        nearest_idx, nearest_error_m = nearest_gallery_tile(sample["lat"], sample["lon"], centers)
        topk_paths = [gallery_paths[int(i)] for i in sample_indices]
        topk_scores = [float(v) for v in sample_scores]
        topk_center_latlon = [centers[int(i)] for i in sample_indices]
        topk_center_error_m = [
            haversine_m(sample["lat"], sample["lon"], tile_lat, tile_lon)
            for tile_lat, tile_lon in topk_center_latlon
        ]

        result = dict(sample)
        result.update(
            {
                "nearest_gallery_tile_path": gallery_paths[nearest_idx],
                "nearest_gallery_tile_center_error_m": float(nearest_error_m),
                "topk_paths": topk_paths,
                "topk_scores": topk_scores,
                "topk_center_latlon": [(float(lat), float(lon)) for lat, lon in topk_center_latlon],
                "topk_center_error_m": [float(v) for v in topk_center_error_m],
            }
        )
        results.append(result)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if samples:
        actual_local_start_s = float(samples[0]["video_local_time_s"])
        actual_abs_start_s = float(samples[0]["abs_time_s"])
        actual_duration_s = float(samples[-1]["segment_elapsed_s"] + 1.0 / target_video_info.fps)
    else:
        actual_local_start_s = local_start_s
        actual_abs_start_s = segment_abs_start_s
        actual_duration_s = 0.0

    segment = {
        "requested_abs_start_s": segment_abs_start_s,
        "requested_duration_s": segment_duration_s,
        "actual_abs_start_s": actual_abs_start_s,
        "actual_local_start_s": actual_local_start_s,
        "actual_duration_s": actual_duration_s,
        "source_fps": float(target_video_info.fps),
        "source_frame_count": len(samples),
    }
    build_summary(
        run_dir=run_dir,
        args=args,
        video_info={
            "path": str(target_video_info.path),
            "fps": float(target_video_info.fps),
            "frame_count": int(target_video_info.frame_count),
            "width": int(target_video_info.width),
            "height": int(target_video_info.height),
            "duration_s": float(target_video_info.duration_s),
            "start_time_s": float(target_video_info.start_time_s),
            "end_time_s": float(target_video_info.end_time_s),
        },
        segment=segment,
        gallery_count=len(gallery_paths),
        results=results,
    )
    print(f"Continuous demo ready: {run_dir}")


if __name__ == "__main__":
    main()
