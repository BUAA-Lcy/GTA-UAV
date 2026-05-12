#!/usr/bin/env python3
"""Fast mentor-facing demo for raw DJI aerial videos with GPS/pose logs.

This script builds a lightweight end-to-end retrieval demo for the raw
`Aerial-to-ground1` drop:

1. parse DJI flight logs;
2. sample frames from one or more MP4 files;
3. download the surrounding satellite tile gallery on demand;
4. run a pretrained Game4Loc retrieval model;
5. export a contact sheet, trajectory figure, and markdown summary.

The design goal is speed-to-demo, not a paper-grade benchmark. The per-frame
GPS/error numbers are reported as proxy values because MP4-to-flight-log
alignment is approximated by sequential coverage over the continuous
`CAMERA.isVideo` interval.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import albumentations as A
import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game4loc.models.model import DesModel  # noqa: E402


TILE_SIZE = 256
DEFAULT_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


@dataclass
class FlightRow:
    fly_time_s: float
    lat: float
    lon: float
    height_ft: float
    altitude_ft: float
    yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float
    is_video: bool


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    start_time_s: float
    end_time_s: float


@dataclass
class QuerySample:
    index: int
    abs_time_s: float
    fly_time_s: float
    lat: float
    lon: float
    height_ft: float
    altitude_ft: float
    yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float
    video_path: str
    frame_path: str


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self) -> int:
        return len(self.image_paths)


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
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--zoom", type=int, default=18, help="Satellite tile zoom level.")
    parser.add_argument(
        "--tile_margin_m",
        type=float,
        default=120.0,
        help="Extra gallery margin around the flight track in meters.",
    )
    parser.add_argument(
        "--sample_interval_sec",
        type=float,
        default=30.0,
        help="Nominal interval between sampled query frames.",
    )
    parser.add_argument(
        "--sample_start_sec",
        type=float,
        default=0.0,
        help="Absolute start time within the concatenated videos for query sampling.",
    )
    parser.add_argument(
        "--sample_end_sec",
        type=float,
        default=None,
        help="Absolute end time within the concatenated videos for query sampling.",
    )
    parser.add_argument(
        "--sample_max_frames",
        type=int,
        default=10,
        help="Maximum number of sampled query frames to keep in the report.",
    )
    parser.add_argument(
        "--query_save_width",
        type=int,
        default=1280,
        help="Resize sampled video frames to this width before saving.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k retrieval count to store.")
    parser.add_argument(
        "--display_topk",
        type=int,
        default=3,
        help="Number of gallery results to display per query in the contact sheet.",
    )
    parser.add_argument(
        "--query_max_gimbal_pitch_deg",
        type=float,
        default=None,
        help="Optional filter: only sample frames aligned to rows with gimbal pitch <= this threshold.",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only prepare gallery tiles and extracted queries without retrieval.",
    )
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_flight_log(csv_path: Path) -> List[FlightRow]:
    rows: List[FlightRow] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
        if not first_line.startswith("sep="):
            f.seek(0)
        reader = csv.DictReader(f)
        for record in reader:
            lat = record.get("OSD.latitude", "")
            lon = record.get("OSD.longitude", "")
            fly_time_s = record.get("OSD.flyTime [s]", "")
            if lat in {"", None} or lon in {"", None} or fly_time_s in {"", None}:
                continue
            rows.append(
                FlightRow(
                    fly_time_s=float(fly_time_s),
                    lat=float(lat),
                    lon=float(lon),
                    height_ft=float(record.get("OSD.height [ft]", 0.0) or 0.0),
                    altitude_ft=float(record.get("OSD.altitude [ft]", 0.0) or 0.0),
                    yaw_deg=float(record.get("OSD.yaw [360]", 0.0) or 0.0),
                    gimbal_pitch_deg=float(record.get("GIMBAL.pitch", 0.0) or 0.0),
                    gimbal_yaw_deg=float(record.get("GIMBAL.yaw [360]", 0.0) or 0.0),
                    is_video=str2bool(record.get("CAMERA.isVideo", "")),
                )
            )
    if not rows:
        raise RuntimeError(f"No valid GPS rows found in flight log: {csv_path}")
    return rows


def discover_inputs(data_root: Path) -> tuple[List[Path], Path]:
    videos = sorted(data_root.glob("*_video.mp4"))
    csv_files = sorted(data_root.glob("DJIFlightRecord*.csv"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {data_root}")
    if not csv_files:
        raise FileNotFoundError(f"No DJIFlightRecord CSV found under {data_root}")
    return videos, csv_files[0]


def load_video_info(video_path: Path, start_time_s: float) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}: fps={fps}, frames={frame_count}")
    duration_s = frame_count / fps
    return VideoInfo(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=duration_s,
        start_time_s=start_time_s,
        end_time_s=start_time_s + duration_s,
    )


def build_video_infos(video_paths: Sequence[Path]) -> List[VideoInfo]:
    infos: List[VideoInfo] = []
    current_start = 0.0
    for video_path in video_paths:
        info = load_video_info(video_path, start_time_s=current_start)
        infos.append(info)
        current_start = info.end_time_s
    return infos


def find_video_for_time(video_infos: Sequence[VideoInfo], abs_time_s: float) -> VideoInfo:
    for info in video_infos:
        if abs_time_s <= info.end_time_s or info is video_infos[-1]:
            return info
    return video_infos[-1]


def capture_frame(video_path: str, local_time_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for frame extraction: {video_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, max(local_time_s, 0.0) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_idx = int(round(max(local_time_s, 0.0) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode frame at t={local_time_s:.3f}s from {video_path}")
    return frame


def maybe_resize(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / float(w)
    target_height = int(round(h * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def uniform_sample_times(
    total_duration_s: float,
    interval_s: float,
    max_frames: int,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> List[float]:
    if total_duration_s <= 0:
        return [0.0]

    start_s = float(max(0.0, start_s))
    end_s = float(total_duration_s if end_s is None else min(max(end_s, start_s), total_duration_s))
    window_duration_s = max(end_s - start_s, 0.0)
    if window_duration_s <= 0:
        return [start_s]

    times = list(np.arange(start_s, end_s + 1e-6, max(interval_s, 1e-3), dtype=np.float64))
    if len(times) <= 1:
        times = [start_s + window_duration_s * 0.5]

    if max_frames > 0 and len(times) > max_frames:
        indices = np.linspace(0, len(times) - 1, num=max_frames)
        indices = sorted({int(round(v)) for v in indices})
        times = [times[index] for index in indices]

    if times and abs(times[0] - start_s) < 1e-8 and window_duration_s > 1.0:
        times[0] = min(start_s + interval_s * 0.5, start_s + window_duration_s * 0.1)

    return [float(min(max(t, start_s), end_s)) for t in times]


def select_video_log_rows(rows: Sequence[FlightRow]) -> List[FlightRow]:
    video_rows = [row for row in rows if row.is_video]
    return video_rows if video_rows else list(rows)


def align_abs_time_to_log_row(
    abs_time_s: float,
    video_total_duration_s: float,
    video_rows: Sequence[FlightRow],
) -> FlightRow:
    if not video_rows:
        raise ValueError("video_rows must not be empty")

    if video_total_duration_s <= 0:
        return video_rows[0]

    log_start = video_rows[0].fly_time_s
    log_end = video_rows[-1].fly_time_s
    scaled_time = log_start + abs_time_s * ((log_end - log_start) / video_total_duration_s)

    fly_times = [row.fly_time_s for row in video_rows]
    insert_idx = bisect.bisect_left(fly_times, scaled_time)
    if insert_idx <= 0:
        return video_rows[0]
    if insert_idx >= len(video_rows):
        return video_rows[-1]

    prev_row = video_rows[insert_idx - 1]
    next_row = video_rows[insert_idx]
    if abs(prev_row.fly_time_s - scaled_time) <= abs(next_row.fly_time_s - scaled_time):
        return prev_row
    return next_row


def log_row_to_abs_time(
    row: FlightRow,
    video_rows: Sequence[FlightRow],
    total_video_duration_s: float,
) -> float:
    if not video_rows:
        return 0.0
    log_start = video_rows[0].fly_time_s
    log_end = video_rows[-1].fly_time_s
    if log_end <= log_start or total_video_duration_s <= 0:
        return 0.0
    ratio = (row.fly_time_s - log_start) / (log_end - log_start)
    return float(min(max(ratio, 0.0), 1.0) * total_video_duration_s)


def select_query_sample_times(
    flight_rows: Sequence[FlightRow],
    total_video_duration_s: float,
    interval_s: float,
    max_frames: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_gimbal_pitch_deg: float | None = None,
) -> List[float]:
    video_rows = select_video_log_rows(flight_rows)
    base_times = uniform_sample_times(
        total_duration_s=total_video_duration_s,
        interval_s=interval_s,
        max_frames=max_frames,
        start_s=start_s,
        end_s=end_s,
    )
    if max_gimbal_pitch_deg is None:
        return base_times

    candidate_times = []
    actual_end_s = total_video_duration_s if end_s is None else end_s
    for row in video_rows:
        if row.gimbal_pitch_deg > max_gimbal_pitch_deg:
            continue
        abs_time = log_row_to_abs_time(
            row=row,
            video_rows=video_rows,
            total_video_duration_s=total_video_duration_s,
        )
        if start_s <= abs_time <= actual_end_s:
            candidate_times.append(abs_time)

    if not candidate_times:
        raise RuntimeError(
            "No candidate frames remain after applying the gimbal-pitch filter. "
            "Try a larger --sample_end_sec or a looser --query_max_gimbal_pitch_deg."
        )

    candidate_times = sorted(candidate_times)
    if max_frames > 0 and len(candidate_times) > max_frames:
        indices = np.linspace(0, len(candidate_times) - 1, num=max_frames)
        candidate_times = [candidate_times[int(round(idx))] for idx in indices]

    deduped = []
    min_gap = max(interval_s * 0.33, 1.0)
    for t in candidate_times:
        if not deduped or abs(t - deduped[-1]) >= min_gap:
            deduped.append(t)
    return deduped if deduped else [candidate_times[len(candidate_times) // 2]]


def meters_to_lat_delta(meters: float) -> float:
    return meters / 111_320.0


def meters_to_lon_delta(meters: float, latitude_deg: float) -> float:
    return meters / (111_320.0 * max(math.cos(math.radians(latitude_deg)), 1e-6))


def expand_latlon_bbox(rows: Sequence[FlightRow], margin_m: float) -> tuple[float, float, float, float]:
    lats = [row.lat for row in rows]
    lons = [row.lon for row in rows]
    lat_center = 0.5 * (min(lats) + max(lats))
    lat_margin = meters_to_lat_delta(margin_m)
    lon_margin = meters_to_lon_delta(margin_m, lat_center)
    return (
        min(lats) - lat_margin,
        min(lons) - lon_margin,
        max(lats) + lat_margin,
        max(lons) + lon_margin,
    )


def latlon_to_tile_xy(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def latlon_to_tile_xy_float(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def tile_center_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n))))
    return (lat_top + lat_bottom) * 0.5, (lon_left + lon_right) * 0.5


def tile_grid_from_bbox(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    zoom: int,
) -> tuple[int, int, int, int]:
    xa, ya = latlon_to_tile_xy(lat_min, lon_min, zoom)
    xb, yb = latlon_to_tile_xy(lat_max, lon_max, zoom)
    x_min, x_max = sorted((xa, xb))
    y_min, y_max = sorted((ya, yb))
    return x_min, x_max, y_min, y_max


def download_tile(x: int, y: int, zoom: int, save_path: Path, url_template: str, retries: int = 5) -> None:
    url = url_template.format(z=zoom, x=x, y=y)
    headers = {"User-Agent": "Game4Loc-AerialGround1-Demo/1.0"}
    last_error = None
    for attempt in range(1, int(retries) + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            save_path.write_bytes(payload)
            return
        except Exception as exc:
            last_error = exc
            if attempt >= int(retries):
                break
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(str(last_error))


def prepare_gallery_tiles(
    gallery_dir: Path,
    zoom: int,
    tile_bounds: tuple[int, int, int, int],
    url_template: str,
    force_redownload: bool = False,
) -> List[str]:
    ensure_dir(gallery_dir)
    x_min, x_max, y_min, y_max = tile_bounds
    gallery_paths: List[str] = []

    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"Preparing gallery tiles: zoom={zoom}, count={total}")
    download_start = time.time()
    for idx, x in enumerate(range(x_min, x_max + 1), start=1):
        for y in range(y_min, y_max + 1):
            save_path = gallery_dir / f"{x}_{y}.jpg"
            gallery_paths.append(str(save_path))
            if save_path.exists() and not force_redownload:
                continue
            try:
                download_tile(x=x, y=y, zoom=zoom, save_path=save_path, url_template=url_template)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to download tile z={zoom}, x={x}, y={y}: {exc}") from exc
        if idx % max(1, math.ceil((x_max - x_min + 1) / 10)) == 0:
            elapsed = time.time() - download_start
            print(f"  downloaded column {idx}/{x_max - x_min + 1} in {elapsed:.1f}s")

    return gallery_paths


def extract_query_samples(
    video_infos: Sequence[VideoInfo],
    flight_rows: Sequence[FlightRow],
    sample_times_s: Sequence[float],
    query_dir: Path,
    query_save_width: int,
) -> List[QuerySample]:
    ensure_dir(query_dir)
    video_rows = select_video_log_rows(flight_rows)
    total_video_duration_s = video_infos[-1].end_time_s if video_infos else 0.0
    samples: List[QuerySample] = []

    for index, abs_time_s in enumerate(sample_times_s, start=1):
        info = find_video_for_time(video_infos, abs_time_s)
        local_time_s = min(max(abs_time_s - info.start_time_s, 0.0), max(info.duration_s - 1e-3, 0.0))
        frame = capture_frame(info.path, local_time_s=local_time_s)
        frame = maybe_resize(frame, target_width=query_save_width)
        aligned_row = align_abs_time_to_log_row(
            abs_time_s=abs_time_s,
            video_total_duration_s=total_video_duration_s,
            video_rows=video_rows,
        )

        frame_name = (
            f"q{index:02d}_t{abs_time_s:07.2f}_"
            f"fly{aligned_row.fly_time_s:07.2f}_"
            f"lat{aligned_row.lat:.6f}_lon{aligned_row.lon:.6f}.jpg"
        )
        frame_path = query_dir / frame_name
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        samples.append(
            QuerySample(
                index=index,
                abs_time_s=float(abs_time_s),
                fly_time_s=aligned_row.fly_time_s,
                lat=aligned_row.lat,
                lon=aligned_row.lon,
                height_ft=aligned_row.height_ft,
                altitude_ft=aligned_row.altitude_ft,
                yaw_deg=aligned_row.yaw_deg,
                gimbal_pitch_deg=aligned_row.gimbal_pitch_deg,
                gimbal_yaw_deg=aligned_row.gimbal_yaw_deg,
                video_path=info.path,
                frame_path=str(frame_path),
            )
        )
    return samples


def build_model(args: argparse.Namespace) -> tuple[DesModel, object]:
    model = DesModel(
        model_name=args.model,
        pretrained=False,
        img_size=args.img_size,
    )
    state_dict = torch.load(args.checkpoint_start, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    data_config = model.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    val_transforms = A.Compose(
        [
            A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.Normalize(mean, std),
            ToTensorV2(),
        ]
    )

    model = model.to(args.device)
    model.eval()
    return model, val_transforms


def extract_features(
    image_paths: Sequence[str],
    model: DesModel,
    transform,
    batch_size: int,
    num_workers: int,
    device: str,
) -> torch.Tensor:
    dataset = ImagePathDataset(image_paths=image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    features: List[torch.Tensor] = []
    use_amp = device.startswith("cuda")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                embedding = model(img1=batch)
            embedding = F.normalize(embedding, dim=-1)
            features.append(embedding.float().cpu())
    return torch.cat(features, dim=0)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_m * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def gallery_tile_centers(gallery_paths: Sequence[str], zoom: int) -> List[tuple[float, float]]:
    centers = []
    for path in gallery_paths:
        stem = Path(path).stem
        x_str, y_str = stem.split("_")[:2]
        centers.append(tile_center_latlon(int(x_str), int(y_str), zoom))
    return centers


def retrieve_topk(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = query_features @ gallery_features.T
    top_scores, top_indices = torch.topk(similarity, k=min(topk, gallery_features.shape[0]), dim=1)
    return top_indices.numpy(), top_scores.numpy()


def nearest_gallery_tile(
    lat: float,
    lon: float,
    gallery_centers: Sequence[tuple[float, float]],
) -> tuple[int, float]:
    best_index = -1
    best_distance = float("inf")
    for idx, (tile_lat, tile_lon) in enumerate(gallery_centers):
        distance = haversine_m(lat, lon, tile_lat, tile_lon)
        if distance < best_distance:
            best_distance = distance
            best_index = idx
    return best_index, best_distance


def save_query_manifest(samples: Sequence[QuerySample], save_path: Path) -> None:
    manifest = [asdict(sample) for sample in samples]
    save_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_contact_sheet(
    results: Sequence[dict],
    save_path: Path,
    display_topk: int,
) -> None:
    if not results:
        return

    display_topk = max(1, min(display_topk, len(results[0]["topk_paths"])))
    rows = len(results)
    cols = 1 + display_topk
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, result in enumerate(results):
        query_img = read_rgb(result["query_path"])
        query_ax = axes[row_idx, 0]
        query_ax.imshow(query_img)
        query_ax.set_title(
            f"Q{result['index']:02d}\n"
            f"fly={result['fly_time_s']:.1f}s\n"
            f"pitch={result['gimbal_pitch_deg']:.1f} deg"
        )
        query_ax.axis("off")

        for rank in range(display_topk):
            ax = axes[row_idx, rank + 1]
            gallery_img = read_rgb(result["topk_paths"][rank])
            ax.imshow(gallery_img)
            top_err = result["topk_center_error_m"][rank]
            ax.set_title(f"Top-{rank + 1}\nerr={top_err:.1f} m")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def stitch_tile_mosaic(
    gallery_dir: Path,
    tile_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    x_min, x_max, y_min, y_max = tile_bounds
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = gallery_dir / f"{x}_{y}.jpg"
            tile = cv2.imread(str(tile_path))
            if tile is None:
                continue
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            x_offset = (x - x_min) * TILE_SIZE
            y_offset = (y - y_min) * TILE_SIZE
            canvas[y_offset:y_offset + TILE_SIZE, x_offset:x_offset + TILE_SIZE] = tile
    return canvas


def latlon_to_mosaic_pixel(
    lat: float,
    lon: float,
    zoom: int,
    tile_bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    x_float, y_float = latlon_to_tile_xy_float(lat_deg=lat, lon_deg=lon, zoom=zoom)
    x_min, _, y_min, _ = tile_bounds
    pixel_x = (x_float - x_min) * TILE_SIZE
    pixel_y = (y_float - y_min) * TILE_SIZE
    return pixel_x, pixel_y


def render_trajectory_plot(
    results: Sequence[dict],
    gallery_dir: Path,
    tile_bounds: tuple[int, int, int, int],
    zoom: int,
    save_path: Path,
) -> None:
    if not results:
        return

    mosaic = stitch_tile_mosaic(gallery_dir=gallery_dir, tile_bounds=tile_bounds)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(mosaic)

    for result in results:
        gt_x, gt_y = latlon_to_mosaic_pixel(result["lat"], result["lon"], zoom=zoom, tile_bounds=tile_bounds)
        pred_lat, pred_lon = result["topk_center_latlon"][0]
        pred_x, pred_y = latlon_to_mosaic_pixel(pred_lat, pred_lon, zoom=zoom, tile_bounds=tile_bounds)
        ax.plot([gt_x, pred_x], [gt_y, pred_y], color="#f59e0b", linewidth=1.3, alpha=0.85)
        ax.scatter(gt_x, gt_y, s=60, c="#2563eb", marker="o", edgecolors="white", linewidths=0.8)
        ax.scatter(pred_x, pred_y, s=60, c="#dc2626", marker="x", linewidths=2.0)
        ax.text(gt_x + 8, gt_y - 8, f"Q{result['index']:02d}", color="white", fontsize=8, weight="bold")

    ax.set_title("Aerial-to-ground1 Demo: GPS track vs retrieved top-1 tile center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_results(
    results: Sequence[dict],
    run_dir: Path,
    args: argparse.Namespace,
    video_infos: Sequence[VideoInfo],
    gallery_count: int,
) -> None:
    top1_errors = [item["topk_center_error_m"][0] for item in results]
    topk_best_errors = [min(item["topk_center_error_m"]) for item in results]

    summary = {
        "dataset": "Aerial-to-ground1",
        "note": (
            "Per-frame GPS numbers are proxy values because MP4-to-flight-log "
            "alignment is approximated by sequential coverage over the "
            "continuous CAMERA.isVideo interval."
        ),
        "args": vars(args),
        "video_infos": [asdict(info) for info in video_infos],
        "query_count": len(results),
        "gallery_count": gallery_count,
        "top1_center_error_mean_m": float(np.mean(top1_errors)) if top1_errors else None,
        "top1_center_error_median_m": float(np.median(top1_errors)) if top1_errors else None,
        "topk_best_center_error_mean_m": float(np.mean(topk_best_errors)) if topk_best_errors else None,
        "recall_top1_within_100m": float(np.mean([err <= 100.0 for err in top1_errors])) if top1_errors else None,
        "recall_top1_within_200m": float(np.mean([err <= 200.0 for err in top1_errors])) if top1_errors else None,
        "queries": results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Aerial-to-ground1 Demo Summary",
        "",
        "This is a fast retrieval-only demo built from raw DJI videos and the flight log.",
        "",
        f"- Query frames: {len(results)}",
        f"- Gallery tiles (zoom {args.zoom}): {gallery_count}",
        f"- Proxy top-1 center error (mean): {summary['top1_center_error_mean_m']:.2f} m" if top1_errors else "- Proxy top-1 center error (mean): N/A",
        f"- Proxy top-1 center error (median): {summary['top1_center_error_median_m']:.2f} m" if top1_errors else "- Proxy top-1 center error (median): N/A",
        f"- Proxy top-k best center error (mean): {summary['topk_best_center_error_mean_m']:.2f} m" if topk_best_errors else "- Proxy top-k best center error (mean): N/A",
        f"- Top-1 within 100 m: {summary['recall_top1_within_100m'] * 100.0:.1f}%" if top1_errors else "- Top-1 within 100 m: N/A",
        f"- Top-1 within 200 m: {summary['recall_top1_within_200m'] * 100.0:.1f}%" if top1_errors else "- Top-1 within 200 m: N/A",
        "",
        "## Alignment Note",
        "",
        summary["note"],
        "",
        "## Main Artifacts",
        "",
        "- `queries/`: sampled query frames",
        f"- Cached satellite tiles: `{args.gallery_cache_dir or str((Path(args.output_root) / 'cache' / f'gallery_z{args.zoom}').resolve())}`",
        "- `contact_sheet_topk.jpg`: query/top-k qualitative sheet",
        "- `trajectory_top1_vs_gps.jpg`: GPS path vs retrieved top-1 centers",
        "- `summary.json`: machine-readable result dump",
        "",
        "## Query Snapshot",
        "",
        "| Query | flyTime (s) | Gimbal pitch | Top-1 error (m) | Top-1 tile |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| Q{result['index']:02d} | {result['fly_time_s']:.1f} | "
            f"{result['gimbal_pitch_deg']:.1f} | {result['topk_center_error_m'][0]:.1f} | "
            f"`{Path(result['topk_paths'][0]).name}` |"
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = time.strftime("aerial_ground1_demo_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(output_root / run_name)
    query_dir = ensure_dir(run_dir / "queries")
    if args.gallery_cache_dir:
        gallery_dir = ensure_dir(Path(args.gallery_cache_dir).resolve())
    else:
        gallery_dir = ensure_dir(output_root / "cache" / f"gallery_z{args.zoom}")

    video_paths, csv_path = discover_inputs(data_root)
    flight_rows = read_flight_log(csv_path)
    video_infos = build_video_infos(video_paths)
    total_video_duration_s = video_infos[-1].end_time_s
    sample_times_s = select_query_sample_times(
        flight_rows=flight_rows,
        total_video_duration_s=total_video_duration_s,
        interval_s=args.sample_interval_sec,
        max_frames=args.sample_max_frames,
        start_s=args.sample_start_sec,
        end_s=args.sample_end_sec,
        max_gimbal_pitch_deg=args.query_max_gimbal_pitch_deg,
    )

    lat_min, lon_min, lat_max, lon_max = expand_latlon_bbox(flight_rows, margin_m=args.tile_margin_m)
    tile_bounds = tile_grid_from_bbox(lat_min, lon_min, lat_max, lon_max, zoom=args.zoom)
    gallery_paths = prepare_gallery_tiles(
        gallery_dir=gallery_dir,
        zoom=args.zoom,
        tile_bounds=tile_bounds,
        url_template=args.tile_url_template,
        force_redownload=args.force_redownload_tiles,
    )

    query_samples = extract_query_samples(
        video_infos=video_infos,
        flight_rows=flight_rows,
        sample_times_s=sample_times_s,
        query_dir=query_dir,
        query_save_width=args.query_save_width,
    )
    save_query_manifest(query_samples, run_dir / "query_manifest.json")

    if args.download_only:
        print(f"Prepared queries and gallery only. Output: {run_dir}")
        return

    model, val_transforms = build_model(args)
    query_paths = [sample.frame_path for sample in query_samples]
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
    for sample, sample_indices, sample_scores in zip(query_samples, top_indices, top_scores):
        nearest_idx, nearest_error_m = nearest_gallery_tile(sample.lat, sample.lon, centers)

        topk_paths = [gallery_paths[int(i)] for i in sample_indices]
        topk_scores = [float(v) for v in sample_scores]
        topk_center_latlon = [centers[int(i)] for i in sample_indices]
        topk_center_error_m = [
            haversine_m(sample.lat, sample.lon, tile_lat, tile_lon)
            for tile_lat, tile_lon in topk_center_latlon
        ]

        results.append(
            {
                "index": sample.index,
                "abs_time_s": sample.abs_time_s,
                "fly_time_s": sample.fly_time_s,
                "lat": sample.lat,
                "lon": sample.lon,
                "height_ft": sample.height_ft,
                "altitude_ft": sample.altitude_ft,
                "yaw_deg": sample.yaw_deg,
                "gimbal_pitch_deg": sample.gimbal_pitch_deg,
                "gimbal_yaw_deg": sample.gimbal_yaw_deg,
                "query_path": sample.frame_path,
                "video_path": sample.video_path,
                "nearest_gallery_tile_path": gallery_paths[nearest_idx],
                "nearest_gallery_tile_center_error_m": float(nearest_error_m),
                "topk_paths": topk_paths,
                "topk_scores": topk_scores,
                "topk_center_latlon": [(float(lat), float(lon)) for lat, lon in topk_center_latlon],
                "topk_center_error_m": [float(v) for v in topk_center_error_m],
            }
        )

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    render_contact_sheet(results=results, save_path=run_dir / "contact_sheet_topk.jpg", display_topk=args.display_topk)
    render_trajectory_plot(
        results=results,
        gallery_dir=gallery_dir,
        tile_bounds=tile_bounds,
        zoom=args.zoom,
        save_path=run_dir / "trajectory_top1_vs_gps.jpg",
    )
    summarize_results(
        results=results,
        run_dir=run_dir,
        args=args,
        video_infos=video_infos,
        gallery_count=len(gallery_paths),
    )

    print(f"Demo complete. Output directory: {run_dir}")


if __name__ == "__main__":
    main()
