#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Generate publication-quality figures for the VOP short paper draft."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
REPO_ROOT = GAME4LOC_DIR.parent

LOG_DIR = GAME4LOC_DIR / "Log"
DATA_DIR = GAME4LOC_DIR / "data"
WORK_DIR = GAME4LOC_DIR / "work_dir"
PAPER_PATH = REPO_ROOT / "Paper.zh-CN.md"
SUPERVISION_PAPER_PATH = REPO_ROOT / "Paper.md"


@dataclass(frozen=True)
class FigureMethod:
    name: str
    short_label: str
    log_path: Path | None
    dataset: str
    color: str
    highlight: bool = False
    marker: str = "o"
    label_dx: float = 4
    label_dy: float = 6


PALETTE = {
    "ours": "#C84C3A",
    "dense": "#25262B",
    "rotate": "#3E7CB1",
    "sparse": "#8A9099",
    "teacher": "#3F8C84",
    "clean30": "#C28A2C",
    "useful": "#6D5EAC",
    "grid": "#D9DCE3",
    "axis": "#4A4F57",
    "text": "#1F2329",
    "muted": "#6D737D",
    "success": "#5A9A5A",
    "useful_fill": "#D8EED8",
    "loftr": "#8F6B2E",
    "top2": "#5C948A",
    "callout": "#A85D2A",
    "callout_green": "#2E6B3F",
}


VISLOC_0304_METHODS = [
    FigureMethod(
        name="dense DKM, no rotate",
        short_label="DKM",
        log_path=None,
        dataset="03/04",
        color=PALETTE["dense"],
        marker="^",
        label_dx=-24,
        label_dy=4,
    ),
    FigureMethod(
        name="LoFTR",
        short_label="LoFTR",
        log_path=None,
        dataset="03/04",
        color=PALETTE["loftr"],
        marker="D",
        label_dx=6,
        label_dy=6,
    ),
    FigureMethod(
        name="SuperPoint",
        short_label="Sparse",
        log_path=None,
        dataset="03/04",
        color=PALETTE["sparse"],
        label_dx=6,
        label_dy=-14,
    ),
    FigureMethod(
        name="SuperPoint + Rotate",
        short_label="Sparse+Rotate",
        log_path=None,
        dataset="03/04",
        color=PALETTE["rotate"],
        label_dx=6,
        label_dy=6,
    ),
    FigureMethod(
        name="Ours (SuperPoint + VOP)",
        short_label="Ours",
        log_path=None,
        dataset="03/04",
        color=PALETTE["ours"],
        highlight=True,
        label_dx=8,
        label_dy=8,
    ),
]

GTA_METHODS = [
    FigureMethod(
        name="dense DKM",
        short_label="DKM",
        log_path=None,
        dataset="GTA same-area",
        color=PALETTE["dense"],
        marker="^",
        label_dx=6,
        label_dy=5,
    ),
    FigureMethod(
        name="sparse",
        short_label="Sparse",
        log_path=None,
        dataset="GTA same-area",
        color=PALETTE["sparse"],
        label_dx=6,
        label_dy=-14,
    ),
    FigureMethod(
        name="sparse + rotate90",
        short_label="Sparse+Rotate",
        log_path=None,
        dataset="GTA same-area",
        color=PALETTE["rotate"],
        label_dx=6,
        label_dy=6,
    ),
    FigureMethod(
        name="LoFTR",
        short_label="LoFTR",
        log_path=None,
        dataset="GTA same-area",
        color=PALETTE["loftr"],
        marker="D",
        label_dx=6,
        label_dy=6,
    ),
    FigureMethod(
        name="sparse + VOP (ours)",
        short_label="Ours",
        log_path=None,
        dataset="GTA same-area",
        color=PALETTE["ours"],
        highlight=True,
        label_dx=7,
        label_dy=-14,
    ),
]

GTA_THRESHOLD_METHODS = [method for method in GTA_METHODS if method.name != "dense DKM"]

SUPERVISION_METHODS = [
    FigureMethod(
        name="current teacher, top-2",
        short_label="Top-2",
        log_path=None,
        dataset="03/04",
        color=PALETTE["top2"],
    ),
    FigureMethod(
        name="current teacher, top-4",
        short_label="Top-4",
        log_path=None,
        dataset="03/04",
        color=PALETTE["teacher"],
    ),
    FigureMethod(
        name="clean30, top-4",
        short_label="Clean30",
        log_path=None,
        dataset="03/04",
        color=PALETTE["clean30"],
    ),
    FigureMethod(
        name="useful5-weight30, top-4 (ours)",
        short_label="Ours",
        log_path=None,
        dataset="03/04",
        color=PALETTE["ours"],
        highlight=True,
    ),
]

MAIN_SAMPLE = {
    "cache_path": WORK_DIR / "vop" / "topk_analysis" / "cache_same_area_full.json",
    "eval_path": WORK_DIR / "vop" / "topk_analysis" / "posterior_k4.json",
    "query_name": "03_0079.JPG",
}

MECHANISM_FIG = {
    "cache_path": WORK_DIR / "vop" / "topk_analysis" / "cache_same_area_full.json",
    "eval_path": WORK_DIR / "vop" / "topk_analysis" / "posterior_k4.json",
    "angle_shape_path": WORK_DIR / "vop" / "angle_curve_shape_0408.json",
    "samples": [
        ("Sharp useful set", "04_0125.JPG"),
        ("Broad useful interval", "03_0011.JPG"),
        ("Multimodal useful set", "03_0570.JPG"),
    ],
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": PALETTE["axis"],
            "axes.linewidth": 0.9,
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.65,
            "grid.alpha": 0.6,
            "xtick.color": PALETTE["axis"],
            "ytick.color": PALETTE["axis"],
            "text.color": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "legend.frameon": False,
            "legend.fontsize": 9.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC_DIR / "figures" / "vop_shortpaper_20260411",
        help="Output directory for the generated figures.",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in (".png", ".pdf"):
        out_path = output_dir / f"{stem}{suffix}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        saved.append(out_path)
    return saved


def soften_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_float(pattern: str, text: str, default: float | None = None) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match:
        return float(match.group(1))
    return default


def parse_log_metrics(log_path: Path) -> Dict[str, float]:
    text = read_text(log_path)
    metrics: Dict[str, float] = {}

    metrics["dis1_m"] = extract_float(r"Dis@1:\s*([0-9.]+)", text) or math.nan
    metrics["dis3_m"] = extract_float(r"Dis@3:\s*([0-9.]+)", text) or math.nan
    metrics["dis5_m"] = extract_float(r"Dis@5:\s*([0-9.]+)", text) or math.nan
    metrics["ma3_pct"] = extract_float(r"MA@3m:\s*([0-9.]+)", text) or math.nan
    metrics["ma5_pct"] = extract_float(r"MA@5m:\s*([0-9.]+)", text) or math.nan
    metrics["ma10_pct"] = extract_float(r"MA@10m:\s*([0-9.]+)", text) or math.nan
    metrics["ma20_pct"] = extract_float(r"MA@20m:\s*([0-9.]+)", text) or math.nan
    metrics["fallback_pct"] = extract_float(r"fallback=\d+\(([0-9.]+)%\)", text) or math.nan
    metrics["worse_pct"] = (
        extract_float(r"worse[-_ ]than[-_ ]coarse=\d+\(([0-9.]+)%\)", text) or math.nan
    )
    metrics["mean_inliers"] = extract_float(r"mean_inliers=([0-9.]+)", text) or math.nan
    metrics["mean_total_s"] = (
        extract_float(r"mean_total(?:_time)?=([0-9.]+)s/query", text)
        or extract_float(r"with_match 阶段平均耗时:.*?总计=([0-9.]+)s", text)
        or math.nan
    )
    metrics["mean_vop_s"] = extract_float(r"mean_(?:vop_forward|posterior_time)=([0-9.]+)s/query", text)
    return metrics


def extract_markdown_table_after_heading(text: str, heading_fragment: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    start_idx = next(
        idx for idx, line in enumerate(lines) if line.strip().startswith("## ") and heading_fragment in line
    )
    table_lines: List[str] = []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif table_lines:
            break
    if len(table_lines) < 3:
        raise ValueError(f"Could not find a markdown table after heading containing: {heading_fragment}")

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: List[Dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def get_row_name(row: Mapping[str, str]) -> str | None:
    for key in ("Variant", "变体", "方法"):
        if key in row:
            value = row[key].strip()
            if value:
                return value
    return None


def parse_numeric_cell(value: str) -> float:
    cleaned = value.strip().replace("%", "").replace("s", "")
    return float(cleaned)


def convert_table_row_to_metrics(row: Mapping[str, str]) -> Dict[str, float]:
    header_map = {
        "dis@1": "dis1_m",
        "dis@3": "dis3_m",
        "dis@5": "dis5_m",
        "ma@3": "ma3_pct",
        "ma@3m": "ma3_pct",
        "ma@5": "ma5_pct",
        "ma@5m": "ma5_pct",
        "ma@10": "ma10_pct",
        "ma@10m": "ma10_pct",
        "ma@20": "ma20_pct",
        "ma@20m": "ma20_pct",
        "fallback": "fallback_pct",
        "worse-than-coarse": "worse_pct",
        "total time": "mean_total_s",
        "mean total time": "mean_total_s",
        "mean inliers": "mean_inliers",
        "recall@1": "recall1_pct",
        "recall@5": "recall5_pct",
        "recall@10": "recall10_pct",
        "map": "map_pct",
    }
    metrics: Dict[str, float] = {}
    for header, raw_value in row.items():
        if header.strip().lower() in {"variant", "变体", "方法"}:
            continue
        metric_key = header_map.get(header.strip().lower())
        if metric_key is None:
            continue
        metrics[metric_key] = parse_numeric_cell(raw_value)
    return metrics


def load_table_metrics(table_heading_fragment: str, paper_path: Path = PAPER_PATH) -> Dict[str, Dict[str, float]]:
    rows = extract_markdown_table_after_heading(read_text(paper_path), table_heading_fragment)
    metrics_map: Dict[str, Dict[str, float]] = {}
    for row in rows:
        row_name = get_row_name(row)
        if not row_name:
            continue
        metrics_map[row_name] = convert_table_row_to_metrics(row)
    return metrics_map


def parse_summary_metrics(summary_path: Path) -> Dict[str, float]:
    text = read_text(summary_path)
    metrics: Dict[str, float] = {}
    bullet_patterns = {
        "dis1_m": r"Dis@1=([0-9.]+)",
        "dis3_m": r"Dis@3=([0-9.]+)",
        "dis5_m": r"Dis@5=([0-9.]+)",
        "ma3_pct": r"MA@3m=([0-9.]+)",
        "ma5_pct": r"MA@5m=([0-9.]+)",
        "ma10_pct": r"MA@10m=([0-9.]+)",
        "ma20_pct": r"MA@20m=([0-9.]+)",
        "fallback_pct": r"fallback=\d+\s+\(([0-9.]+)%\)",
        "worse_pct": r"worse_than_coarse=\d+\s+\(([0-9.]+)%\)",
        "mean_inliers": r"mean_inliers=([0-9.]+)",
        "mean_total_s": r"mean_matcher=([0-9.]+)s/query",
    }
    for key, pattern in bullet_patterns.items():
        value = extract_float(pattern, text)
        if value is not None:
            metrics[key] = value
    return metrics


def build_paper_metrics_bundle() -> Dict[str, Dict[str, Dict[str, float]]]:
    visloc_metrics = load_table_metrics("8.2 UAV-VisLoc", PAPER_PATH)
    gta_metrics = load_table_metrics("8.1 GTA Same-Area")
    gta_detail_sources = {
        "sparse": LOG_DIR / "vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log",
        "sparse + rotate90": LOG_DIR / "vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log",
        "sparse + VOP (ours)": LOG_DIR / "vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log",
    }
    for method_name, log_path in gta_detail_sources.items():
        detailed_metrics = parse_log_metrics(log_path)
        detailed_metrics.update(gta_metrics[method_name])
        gta_metrics[method_name] = detailed_metrics
    gta_metrics["LoFTR"] = parse_summary_metrics(
        WORK_DIR / "loftr_baseline_runs" / "gta_samearea_loftr_20260411" / "summary.md"
    )
    supervision_metrics = load_table_metrics("8.2 VisLoc Table", SUPERVISION_PAPER_PATH)
    return {
        "03/04": visloc_metrics,
        "GTA same-area": gta_metrics,
        "supervision": {method.name: supervision_metrics[method.name] for method in SUPERVISION_METHODS},
    }


def collect_metrics(methods: Sequence[FigureMethod]) -> Dict[str, Dict[str, float]]:
    return {method.name: parse_log_metrics(method.log_path) for method in methods if method.log_path is not None}


def load_square_image(path: Path, size: int = 360) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(247, 247, 247))
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas)


def load_rotated_query(path: Path, angle_deg: float, size: int = 220) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.LANCZOS)
    rotated = image.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(248, 248, 248))
    rotated = ImageOps.contain(rotated, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(248, 248, 248))
    left = (size - rotated.width) // 2
    top = (size - rotated.height) // 2
    canvas.paste(rotated, (left, top))
    return np.asarray(canvas)


def prettify_angle(angle_deg: float) -> str:
    return f"{angle_deg:+.0f}°".replace("-", "−")


def reorder_angles(candidate_angles_deg: Sequence[float]) -> Tuple[List[float], List[int]]:
    order = sorted(
        range(len(candidate_angles_deg)),
        key=lambda idx: ((candidate_angles_deg[idx] + 180.0) % 360.0) - 180.0,
    )
    return [candidate_angles_deg[idx] for idx in order], order


def build_useful_mask(distances_sorted: Sequence[float], delta_m: float = 5.0) -> np.ndarray:
    distances = np.asarray(distances_sorted, dtype=float)
    return distances <= (np.nanmin(distances) + delta_m)


def run_components(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    mask_arr = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    n = len(mask_arr)
    i = 0
    while i < n:
        if not mask_arr[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask_arr[j + 1]:
            j += 1
        segments.append((i, j))
        i = j + 1
    if len(segments) > 1 and mask_arr[0] and mask_arr[-1]:
        first = segments[0]
        last = segments[-1]
        segments = [(last[0], first[1] + n)] + segments[1:-1]
    return segments


def load_cache_record(cache_path: Path, eval_path: Path, query_name: str) -> Dict[str, object]:
    cache = json.loads(read_text(cache_path))
    eval_json = json.loads(read_text(eval_path))
    cache_record = next(record for record in cache["records"] if record["query_name"] == query_name)
    eval_record = next(record for record in eval_json["records"] if record["query_name"] == query_name)

    angles_deg = cache["config"]["candidate_angles_deg"]
    angles_sorted, order = reorder_angles(angles_deg)
    distances_sorted = [cache_record["oracle_distances_m"][idx] for idx in order]
    posterior_sorted = [cache_record["posterior_probs"][idx] for idx in order]
    useful_mask = build_useful_mask(distances_sorted, delta_m=5.0)

    angle_to_sorted_idx = {
        round(float(angle), 4): sorted_idx for sorted_idx, angle in enumerate(angles_sorted)
    }
    selected_sorted_idx = [angle_to_sorted_idx[round(float(angle), 4)] for angle in eval_record["selected_angles_deg"]]
    selected_oracle_errors = [distances_sorted[idx] for idx in selected_sorted_idx]
    best_selected_rank = int(np.argmin(selected_oracle_errors))

    return {
        "angles_sorted_deg": angles_sorted,
        "distances_sorted_m": distances_sorted,
        "posterior_sorted": posterior_sorted,
        "useful_mask": useful_mask.tolist(),
        "segments": run_components(useful_mask),
        "cache_record": cache_record,
        "eval_record": eval_record,
        "selected_sorted_idx": selected_sorted_idx,
        "selected_oracle_errors": selected_oracle_errors,
        "best_selected_rank": best_selected_rank,
    }


def add_panel_header(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["muted"],
    )
    ax.text(
        0.095,
        1.08,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        color=PALETTE["text"],
    )


def annotate_scatter_point(ax: plt.Axes, x: float, y: float, label: str, color: str, dx: float = 4, dy: float = 6) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.6,
        color=color,
        fontweight="bold" if label == "Ours" else "normal",
        path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.92)],
    )


def runtime_to_fps(runtime_s: float) -> float:
    if not math.isfinite(runtime_s) or runtime_s <= 0:
        return math.nan
    return 1.0 / runtime_s


def compute_pareto_frontier(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    valid_points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if not valid_points:
        return []

    sorted_points = sorted(valid_points, key=lambda item: item[0])
    frontier: List[Tuple[float, float]] = []
    best_dis1 = math.inf
    for fps, dis1 in reversed(sorted_points):
        if dis1 < best_dis1 - 1e-9:
            frontier.append((fps, dis1))
            best_dis1 = dis1
    frontier.reverse()
    return frontier


def draw_sparse_dense_frontier(
    ax: plt.Axes,
    frontier_points: Sequence[Tuple[float, float]],
    *,
    label_text: str = "Empirical sparse-to-dense frontier",
    fit_mode: str = "quadratic",
    label_fraction: float = 0.48,
    label_y_offset: float = 3.0,
) -> None:
    if len(frontier_points) < 2:
        return

    frontier_points = sorted(frontier_points, key=lambda item: item[0])
    log_frontier_fps = np.log10([point[0] for point in frontier_points])
    frontier_dis1 = np.array([point[1] for point in frontier_points], dtype=float)

    if fit_mode == "piecewise":
        sampled_log_segments = []
        sampled_dis_segments = []
        for idx in range(len(frontier_points) - 1):
            is_last = idx == len(frontier_points) - 2
            sampled_log_segments.append(
                np.linspace(log_frontier_fps[idx], log_frontier_fps[idx + 1], 60, endpoint=is_last)
            )
            sampled_dis_segments.append(
                np.linspace(frontier_dis1[idx], frontier_dis1[idx + 1], 60, endpoint=is_last)
            )
        sampled_log_fps = np.concatenate(sampled_log_segments)
        sampled_dis1 = np.concatenate(sampled_dis_segments)
    elif len(frontier_points) >= 3:
        coeffs = np.polyfit(log_frontier_fps, frontier_dis1, deg=2)
        sampled_log_fps = np.linspace(log_frontier_fps[0], log_frontier_fps[-1], 200)
        sampled_dis1 = np.polyval(coeffs, sampled_log_fps)
    else:
        sampled_log_fps = np.linspace(log_frontier_fps[0], log_frontier_fps[-1], 120)
        sampled_dis1 = np.interp(sampled_log_fps, log_frontier_fps, frontier_dis1)

    frontier_fps = np.power(10.0, sampled_log_fps)
    ax.plot(
        frontier_fps,
        sampled_dis1,
        color=PALETTE["muted"],
        linewidth=1.8,
        linestyle=(0, (4.5, 3.2)),
        alpha=0.95,
        zorder=2,
    )

    label_idx = int(len(sampled_log_fps) * label_fraction)
    ax.text(
        frontier_fps[label_idx],
        sampled_dis1[label_idx] + label_y_offset,
        label_text,
        fontsize=9.1,
        color=PALETTE["muted"],
        style="italic",
        ha="left",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.9)],
    )


def relabel_methods(
    methods: Sequence[FigureMethod],
    label_overrides: Mapping[str, str],
    offset_overrides: Mapping[str, Tuple[float, float]] | None = None,
) -> List[FigureMethod]:
    relabeled: List[FigureMethod] = []
    for method in methods:
        changes: Dict[str, object] = {}
        if method.name in label_overrides:
            changes["short_label"] = label_overrides[method.name]
        if offset_overrides and method.name in offset_overrides:
            dx, dy = offset_overrides[method.name]
            changes["label_dx"] = dx
            changes["label_dy"] = dy
        relabeled.append(replace(method, **changes) if changes else method)
    return relabeled


def create_main_pipeline_figure(output_dir: Path) -> List[Path]:
    fig = plt.figure(figsize=(15.2, 5.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.15, 1.35, 1.08], wspace=0.24)

    sample = load_cache_record(
        MAIN_SAMPLE["cache_path"],
        MAIN_SAMPLE["eval_path"],
        MAIN_SAMPLE["query_name"],
    )
    cache_record = sample["cache_record"]
    eval_record = sample["eval_record"]

    query_path = DATA_DIR / "UAV_VisLoc_dataset" / "drone" / "images" / cache_record["query_name"]
    gallery_path = DATA_DIR / "UAV_VisLoc_dataset" / "satellite" / cache_record["gallery_name"]
    query_img = load_square_image(query_path, size=420)
    gallery_img = load_square_image(gallery_path, size=420)

    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_axis_off()
    ax_query.set_title("1. Query UAV view", loc="left", fontweight="bold")
    ax_query.text(
        0.02,
        0.03,
        cache_record["query_name"],
        transform=ax_query.transAxes,
        fontsize=9.2,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.45), ec="none"),
    )

    ax_gallery = fig.add_subplot(gs[0, 1])
    ax_gallery.imshow(gallery_img)
    ax_gallery.set_axis_off()
    ax_gallery.set_title("2. Retrieved top-1 tile", loc="left", fontweight="bold")
    ax_gallery.text(
        0.02,
        0.03,
        cache_record["gallery_name"],
        transform=ax_gallery.transAxes,
        fontsize=9.2,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.45), ec="none"),
    )

    ax_posterior = fig.add_subplot(gs[0, 2], projection="polar")
    angles_rad = np.deg2rad(sample["angles_sorted_deg"])
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    posterior_width = np.deg2rad(9.0)
    ax_posterior.bar(
        angles_rad,
        posterior,
        width=posterior_width,
        color=PALETTE["rotate"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    selected_indices = sample["selected_sorted_idx"]
    ax_posterior.scatter(
        angles_rad[selected_indices],
        posterior[selected_indices],
        s=62,
        color=PALETTE["ours"],
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
    )
    uniform = np.full_like(posterior, 1.0 / len(posterior))
    ax_posterior.plot(angles_rad, uniform, color=PALETTE["muted"], linewidth=1.2, alpha=0.65, zorder=3)
    ax_posterior.set_theta_zero_location("N")
    ax_posterior.set_theta_direction(-1)
    ax_posterior.set_title("3. VOP posterior over angles", loc="left", fontweight="bold", pad=18)
    ax_posterior.set_rlabel_position(112)
    ax_posterior.set_ylim(0.0, posterior.max() * 1.2)
    ax_posterior.tick_params(labelsize=8.5)
    ax_posterior.grid(color=PALETTE["grid"], alpha=0.85)
    ax_posterior.text(
        0.5,
        -0.12,
        "Highlighted bars = selected top-4 hypotheses",
        transform=ax_posterior.transAxes,
        ha="center",
        va="top",
        fontsize=9.1,
        color=PALETTE["muted"],
    )

    subgrid = gs[0, 3].subgridspec(2, 2, wspace=0.12, hspace=0.18)
    oracle_errors = sample["selected_oracle_errors"]
    candidate_axes: List[plt.Axes] = []
    for idx, (sorted_idx, angle_deg, oracle_error) in enumerate(
        zip(selected_indices, eval_record["selected_angles_deg"], oracle_errors)
    ):
        ax = fig.add_subplot(subgrid[idx // 2, idx % 2])
        candidate_axes.append(ax)
        candidate_img = load_rotated_query(query_path, float(angle_deg), size=250)
        ax.imshow(candidate_img)
        ax.set_axis_off()
        is_best = idx == sample["best_selected_rank"]
        border_color = PALETTE["ours"] if is_best else PALETTE["rotate"]
        rect = patches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            transform=ax.transAxes,
            fill=False,
            linewidth=2.4 if is_best else 1.6,
            edgecolor=border_color,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            0.03,
            0.95,
            prettify_angle(float(angle_deg)),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.2,
            fontweight="bold",
            color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
        )
        ax.text(
            0.03,
            0.05,
            f"verified: {oracle_error:.1f} m",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9.2,
            fontweight="bold" if is_best else "normal",
            color=border_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
        )

    top_left = candidate_axes[0].get_position()
    top_right = candidate_axes[1].get_position()
    fig.text(
        top_left.x0,
        max(top_left.y1, top_right.y1) + 0.022,
        "4. Sparse verification on top-k hypotheses",
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        color=PALETTE["text"],
    )

    ax_final = fig.add_subplot(gs[0, 4])
    ax_final.imshow(gallery_img)
    ax_final.set_axis_off()
    ax_final.set_title("5. Final localization output", loc="left", fontweight="bold")
    best_angle = float(eval_record["selected_angles_deg"][sample["best_selected_rank"]])
    best_oracle_error = float(oracle_errors[sample["best_selected_rank"]])
    final_error = float(eval_record["final_error_m"])
    ax_final.scatter([gallery_img.shape[1] / 2], [gallery_img.shape[0] / 2], s=135, color=PALETTE["ours"], edgecolor="white", linewidth=1.3)
    ax_final.text(
        0.03,
        0.95,
        f"selected angle: {prettify_angle(best_angle)}",
        transform=ax_final.transAxes,
        va="top",
        ha="left",
        fontsize=10.4,
        fontweight="bold",
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92),
    )
    ax_final.text(
        0.03,
        0.05,
        f"query final error: {final_error:.1f} m\nbest single-angle sweep: {best_oracle_error:.1f} m",
        transform=ax_final.transAxes,
        va="bottom",
        ha="left",
        fontsize=9.5,
        color=PALETTE["text"],
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92),
    )

    arrow_y = 0.93
    for x0, x1 in ((0.17, 0.31), (0.37, 0.51), (0.59, 0.77), (0.83, 0.94)):
        fig.add_artist(
            patches.FancyArrowPatch(
                (x0, arrow_y),
                (x1, arrow_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color=PALETTE["muted"],
            )
        )

    fig.suptitle(
        "VOP-guided sparse fine localization pipeline",
        x=0.01,
        ha="left",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    return save_figure(fig, output_dir, "fig01_vop_pipeline_example")


def create_report_overview_flowchart(output_dir: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(15.2, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def add_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        *,
        fc: str = "white",
        ec: str = PALETTE["axis"],
        lw: float = 1.4,
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
            x + 0.02 * w,
            y + h - 0.23 * h,
            title,
            ha="left",
            va="center",
            fontsize=12.8,
            fontweight="bold",
            color=PALETTE["text"],
        )
        ax.text(
            x + 0.02 * w,
            y + 0.10 * h,
            body,
            ha="left",
            va="bottom",
            fontsize=9.7,
            color=PALETTE["axis"],
            linespacing=1.35,
        )

    def add_arrow(x0: float, y0: float, x1: float, y1: float, *, color: str = PALETTE["muted"]) -> None:
        ax.add_patch(
            patches.FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.6,
                color=color,
            )
        )

    fig.suptitle(
        "VOP-guided sparse fine localization framework",
        x=0.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    ax.text(0.02, 0.90, "Training stage", fontsize=13, fontweight="bold", color=PALETTE["muted"])
    ax.text(0.02, 0.47, "Inference stage", fontsize=13, fontweight="bold", color=PALETTE["muted"])

    add_box(
        0.05,
        0.71,
        0.18,
        0.15,
        "Teacher construction",
        "Fix retrieval top-1\nSweep discrete angles\nRecord localization outcome",
        fc="#F7F4EC",
        ec="#C28A2C",
    )
    add_box(
        0.29,
        0.71,
        0.21,
        0.15,
        "Useful-angle supervision",
        "Useful-angle set: d(theta) <= best + delta\nPair-confidence weighting by best distance",
        fc="#F6F1FB",
        ec="#6D5EAC",
    )
    add_box(
        0.56,
        0.71,
        0.16,
        0.15,
        "Train VOP",
        "Frozen retrieval features\nPredict posterior over discrete angles",
        fc="#FBEFEB",
        ec=PALETTE["ours"],
    )
    add_box(
        0.78,
        0.71,
        0.17,
        0.15,
        "Checkpoint",
        "Lightweight useful-angle proposer\nUsed at inference only after retrieval",
        fc="#EDF5EC",
        ec=PALETTE["success"],
    )

    add_arrow(0.23, 0.785, 0.29, 0.785)
    add_arrow(0.50, 0.785, 0.56, 0.785)
    add_arrow(0.72, 0.785, 0.78, 0.785)

    add_box(
        0.04,
        0.25,
        0.15,
        0.16,
        "Input pair",
        "UAV query image\nSatellite gallery\nRetrieval already fixed",
        fc="white",
    )
    add_box(
        0.23,
        0.25,
        0.16,
        0.16,
        "Retrieval top-1",
        "Keep retrieval unchanged\nSelect one candidate tile",
        fc="white",
    )
    add_box(
        0.43,
        0.25,
        0.17,
        0.16,
        "VOP posterior",
        "Predict posterior over\n36 discrete orientations",
        fc="#FBEFEB",
        ec=PALETTE["ours"],
    )
    add_box(
        0.64,
        0.25,
        0.16,
        0.16,
        "Top-k hypotheses",
        "Keep k=4 useful angles\nAllocate sparse matching budget",
        fc="#EEF5FB",
        ec=PALETTE["rotate"],
    )
    add_box(
        0.84,
        0.25,
        0.12,
        0.16,
        "Geometry selection",
        "Sparse matching\nHomography verification\nChoose best result",
        fc="#F2F5F8",
        ec=PALETTE["axis"],
    )

    add_arrow(0.19, 0.34, 0.23, 0.34)
    add_arrow(0.39, 0.34, 0.43, 0.34)
    add_arrow(0.60, 0.34, 0.64, 0.34)
    add_arrow(0.80, 0.34, 0.84, 0.34)
    add_arrow(0.865, 0.72, 0.515, 0.43, color=PALETTE["ours"])

    return save_figure(fig, output_dir, "fig00_report_framework_overview")


def create_tradeoff_figure(output_dir: Path, metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]]) -> List[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    panel_specs = [
        ("GTA-UAV same-area", GTA_METHODS, metrics_bundle["GTA same-area"]),
        ("UAV-VisLoc 03/04", VISLOC_0304_METHODS, metrics_bundle["03/04"]),
    ]

    for ax, (title, methods, metrics_map) in zip(axes, panel_specs):
        soften_spines(ax)
        ax.set_xscale("log")
        ax.set_xlabel("Mean total runtime per query (s, log scale)")
        ax.set_ylabel("MA@20 (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, which="both", linestyle="-", linewidth=0.65, alpha=0.55)

        for method in methods:
            metrics = metrics_map[method.name]
            x = metrics["mean_total_s"]
            y = metrics["ma20_pct"]
            size = 255 if method.highlight else 180
            edge_lw = 1.8 if method.highlight else 0.9
            ax.scatter(
                x,
                y,
                s=size,
                color=method.color,
                marker=method.marker,
                edgecolor="white",
                linewidth=edge_lw,
                zorder=4 if method.highlight else 3,
            )
            annotate_scatter_point(
                ax,
                x,
                y,
                method.short_label,
                method.color,
                dx=method.label_dx,
                dy=method.label_dy,
            )

        xs = [metrics_map[m.name]["mean_total_s"] for m in methods]
        ys = [metrics_map[m.name]["ma20_pct"] for m in methods]
        ax.set_xlim(min(xs) * 0.72, max(xs) * 1.45)
        ax.set_ylim(0, max(ys) * 1.18)

        ax.annotate(
            "better",
            xy=(ax.get_xlim()[0] * 1.03, ax.get_ylim()[1] * 0.93),
            xytext=(ax.get_xlim()[0] * 1.03, ax.get_ylim()[1] * 0.73),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], linewidth=1.2),
            ha="left",
            va="center",
            fontsize=9.6,
            color=PALETTE["muted"],
        )
        ax.annotate(
            "faster",
            xy=(ax.get_xlim()[0] * 1.25, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06),
            xytext=(ax.get_xlim()[0] * 1.75, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], linewidth=1.2),
            ha="left",
            va="center",
            fontsize=9.6,
            color=PALETTE["muted"],
        )

    fig.suptitle(
        "Runtime-robustness trade-off under the current paper benchmarks",
        x=0.01,
        ha="left",
        fontsize=16.2,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "fig02_accuracy_speed_tradeoff")


def create_dis1_efficiency_figure(output_dir: Path, metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]]) -> List[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.05),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.12, 1.0]},
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.86, wspace=0.12)
    panel_specs = [
        ("GTA-UAV", GTA_METHODS, metrics_bundle["GTA same-area"]),
        ("UAV-VisLoc", VISLOC_0304_METHODS, metrics_bundle["03/04"]),
    ]

    for ax, (title, methods, metrics_map) in zip(axes, panel_specs):
        soften_spines(ax)
        ax.set_xscale("log")
        ax.set_xlabel("Frames per second (FPS, log scale)")
        ax.set_ylabel("Dis@1 (m)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, which="both", linestyle="-", linewidth=0.65, alpha=0.55)

        scatter_points: Dict[str, Tuple[float, float]] = {}
        for method in methods:
            metrics = metrics_map[method.name]
            x = runtime_to_fps(metrics["mean_total_s"])
            y = metrics["dis1_m"]
            scatter_points[method.name] = (x, y)
            size = 255 if method.highlight else 180
            edge_lw = 1.8 if method.highlight else 0.9
            ax.scatter(
                x,
                y,
                s=size,
                color=method.color,
                marker=method.marker,
                edgecolor="white",
                linewidth=edge_lw,
                zorder=4 if method.highlight else 3,
            )
            annotate_scatter_point(
                ax,
                x,
                y,
                method.short_label,
                method.color,
                dx=method.label_dx,
                dy=method.label_dy,
            )

        baseline_frontier_points = compute_pareto_frontier(
            [scatter_points[method.name] for method in methods if not method.highlight]
        )
        draw_sparse_dense_frontier(ax, baseline_frontier_points)

        xs = [runtime_to_fps(metrics_map[m.name]["mean_total_s"]) for m in methods]
        ys = [metrics_map[m.name]["dis1_m"] for m in methods]
        ax.set_xlim(min(xs) * 0.72, max(xs) * 1.45)
        ax.set_ylim(max(ys) * 1.18, min(ys) * 0.78)

        ax.annotate(
            "Better",
            xy=(0.96, 0.93),
            xytext=(0.73, 0.71),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color=PALETTE["ours"],
                linewidth=2.5,
                mutation_scale=17,
                shrinkA=2,
                shrinkB=1,
            ),
            ha="left",
            va="center",
            fontsize=11.4,
            fontweight="bold",
            color=PALETTE["ours"],
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=5,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markersize=9,
            markerfacecolor=PALETTE["dense"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Dense DKM",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["loftr"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="LoFTR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["sparse"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Sparse",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["rotate"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Sparse+Rotate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=9.5,
            markerfacecolor=PALETTE["ours"],
            markeredgecolor="white",
            markeredgewidth=1.1,
            label="Ours",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
        fontsize=10.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    return save_figure(fig, output_dir, "fig02_dis1_runtime_tradeoff")


def create_dis1_efficiency_figure_sp_lg_frontier(
    output_dir: Path,
    metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]],
) -> List[Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.9, 10.2),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.08, top=0.90, hspace=0.18)

    label_overrides = {
        "SuperPoint": "SP+LG",
        "sparse": "SP+LG",
        "SuperPoint + Rotate": "SP+LG+Rotate",
        "sparse + rotate90": "SP+LG+Rotate",
    }
    offset_overrides = {
        "SuperPoint": (6, -14),
        "sparse": (6, -14),
        "SuperPoint + Rotate": (7, 8),
        "sparse + rotate90": (7, 8),
    }
    panel_specs = [
        ("GTA-UAV", relabel_methods(GTA_METHODS, label_overrides, offset_overrides), metrics_bundle["GTA same-area"]),
        ("UAV-VisLoc", relabel_methods(VISLOC_0304_METHODS, label_overrides, offset_overrides), metrics_bundle["03/04"]),
    ]

    for ax, (title, methods, metrics_map) in zip(axes, panel_specs):
        soften_spines(ax)
        ax.set_xscale("log")
        ax.set_xlabel("Frames per second (FPS, log scale)")
        ax.set_ylabel("Dis@1 (m)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, which="both", linestyle="-", linewidth=0.65, alpha=0.55)

        scatter_points: Dict[str, Tuple[float, float]] = {}
        for method in methods:
            metrics = metrics_map[method.name]
            x = runtime_to_fps(metrics["mean_total_s"])
            y = metrics["dis1_m"]
            scatter_points[method.name] = (x, y)
            size = 255 if method.highlight else 180
            edge_lw = 1.8 if method.highlight else 0.9
            ax.scatter(
                x,
                y,
                s=size,
                color=method.color,
                marker=method.marker,
                edgecolor="white",
                linewidth=edge_lw,
                zorder=4 if method.highlight else 3,
            )
            annotate_scatter_point(
                ax,
                x,
                y,
                method.short_label,
                method.color,
                dx=method.label_dx,
                dy=method.label_dy,
            )

        baseline_frontier_points = compute_pareto_frontier(
            [scatter_points[method.name] for method in methods if not method.highlight]
        )
        draw_sparse_dense_frontier(
            ax,
            baseline_frontier_points,
            label_text="Baseline Pareto frontier",
            fit_mode="piecewise",
            label_fraction=0.26,
            label_y_offset=1.8,
        )

        xs = [runtime_to_fps(metrics_map[m.name]["mean_total_s"]) for m in methods]
        ys = [metrics_map[m.name]["dis1_m"] for m in methods]
        ax.set_xlim(min(xs) * 0.72, max(xs) * 1.45)
        ax.set_ylim(max(ys) * 1.18, min(ys) * 0.78)

        ax.annotate(
            "Better",
            xy=(0.92, 0.93),
            xytext=(0.81, 0.81),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color=PALETTE["callout_green"],
                linewidth=2.1,
                mutation_scale=13,
                shrinkA=1,
                shrinkB=1,
            ),
            ha="left",
            va="center",
            fontsize=10.8,
            fontweight="bold",
            color=PALETTE["callout_green"],
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=5,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markersize=9,
            markerfacecolor=PALETTE["dense"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Dense DKM",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["loftr"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="LoFTR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["sparse"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="SP+LG",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["rotate"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="SP+LG+Rotate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=9.5,
            markerfacecolor=PALETTE["ours"],
            markeredgecolor="white",
            markeredgewidth=1.1,
            label="Ours",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=5,
        frameon=False,
        fontsize=10.0,
        handletextpad=0.5,
        columnspacing=1.25,
    )
    return save_figure(fig, output_dir, "fig02b_dis1_runtime_tradeoff_sp_lg_frontier")


def create_main_pipeline_posterior_only_figure(output_dir: Path) -> List[Path]:
    sample = load_cache_record(
        MAIN_SAMPLE["cache_path"],
        MAIN_SAMPLE["eval_path"],
        MAIN_SAMPLE["query_name"],
    )

    fig = plt.figure(figsize=(5.8, 5.0))
    ax_posterior = fig.add_subplot(111, projection="polar")
    angles_rad = np.deg2rad(sample["angles_sorted_deg"])
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    posterior_width = np.deg2rad(9.0)
    bars = ax_posterior.bar(
        angles_rad,
        posterior,
        width=posterior_width,
        color=PALETTE["rotate"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    selected_indices = sample["selected_sorted_idx"]
    for idx in selected_indices:
        bars[idx].set_facecolor(PALETTE["ours"])
        bars[idx].set_alpha(0.88)
        bars[idx].set_edgecolor("white")
        bars[idx].set_linewidth(0.4)

    uniform = np.full_like(posterior, 1.0 / len(posterior))
    ax_posterior.plot(angles_rad, uniform, color=PALETTE["muted"], linewidth=1.2, alpha=0.65, zorder=3)
    ax_posterior.set_theta_zero_location("N")
    ax_posterior.set_theta_direction(-1)
    ax_posterior.set_title("3. VOP posterior over angles", loc="left", fontweight="bold", pad=18)
    ax_posterior.set_rlabel_position(112)
    ax_posterior.set_ylim(0.0, posterior.max() * 1.2)
    ax_posterior.tick_params(labelsize=8.5)
    ax_posterior.grid(color=PALETTE["grid"], alpha=0.85)
    ax_posterior.text(
        0.5,
        -0.11,
        "Highlighted bars = selected top-4 hypotheses",
        transform=ax_posterior.transAxes,
        ha="center",
        va="top",
        fontsize=9.1,
        color=PALETTE["muted"],
    )
    return save_figure(fig, output_dir, "fig01_step3_vop_posterior_only")


def create_dis1_efficiency_vertical_figure(
    output_dir: Path,
    metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]],
) -> List[Path]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 10.2),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.90, hspace=0.18)

    panel_specs = [
        ("GTA-UAV", GTA_METHODS, metrics_bundle["GTA same-area"]),
        ("UAV-VisLoc", VISLOC_0304_METHODS, metrics_bundle["03/04"]),
    ]

    for ax, (title, methods, metrics_map) in zip(axes, panel_specs):
        soften_spines(ax)
        ax.set_xscale("log")
        ax.set_xlabel("Mean total runtime per query (s, log scale)")
        ax.set_ylabel("Dis@1 (m)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, which="both", linestyle="-", linewidth=0.65, alpha=0.55)

        for method in methods:
            metrics = metrics_map[method.name]
            x = metrics["mean_total_s"]
            y = metrics["dis1_m"]
            size = 255 if method.highlight else 180
            edge_lw = 1.8 if method.highlight else 0.9
            ax.scatter(
                x,
                y,
                s=size,
                color=method.color,
                marker=method.marker,
                edgecolor="white",
                linewidth=edge_lw,
                zorder=4 if method.highlight else 3,
            )
            annotate_scatter_point(
                ax,
                x,
                y,
                method.short_label,
                method.color,
                dx=method.label_dx,
                dy=method.label_dy,
            )

        xs = [metrics_map[m.name]["mean_total_s"] for m in methods]
        ys = [metrics_map[m.name]["dis1_m"] for m in methods]
        ax.set_xlim(min(xs) * 0.72, max(xs) * 1.45)
        ax.set_ylim(max(ys) * 1.18, min(ys) * 0.78)

        ax.annotate(
            "better",
            xy=(ax.get_xlim()[0] * 1.03, ax.get_ylim()[1] * 1.02),
            xytext=(ax.get_xlim()[0] * 1.03, ax.get_ylim()[1] * 1.55),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], linewidth=1.2),
            ha="left",
            va="center",
            fontsize=9.6,
            color=PALETTE["muted"],
        )
        ax.annotate(
            "faster",
            xy=(ax.get_xlim()[0] * 1.25, ax.get_ylim()[0] - (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.06),
            xytext=(ax.get_xlim()[0] * 1.75, ax.get_ylim()[0] - (ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.06),
            arrowprops=dict(arrowstyle="-|>", color=PALETTE["muted"], linewidth=1.2),
            ha="left",
            va="center",
            fontsize=9.6,
            color=PALETTE["muted"],
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markersize=9,
            markerfacecolor=PALETTE["dense"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Dense DKM",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["loftr"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="LoFTR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["sparse"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Sparse",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=PALETTE["rotate"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label="Sparse+Rotate",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=9.5,
            markerfacecolor=PALETTE["ours"],
            markeredgecolor="white",
            markeredgewidth=1.1,
            label="Ours",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=False,
        fontsize=10.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    return save_figure(fig, output_dir, "fig02c_dis1_runtime_tradeoff_vertical")


def create_threshold_curve_figure(output_dir: Path, metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]]) -> List[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    thresholds = np.array([3, 5, 10, 20], dtype=float)
    panel_specs = [
        ("GTA-UAV same-area", GTA_THRESHOLD_METHODS, metrics_bundle["GTA same-area"]),
        ("UAV-VisLoc 03/04", VISLOC_0304_METHODS, metrics_bundle["03/04"]),
    ]

    for ax, (title, methods, metrics_map) in zip(axes, panel_specs):
        soften_spines(ax)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Localization threshold (m)")
        ax.set_ylabel("Success rate (%)")
        ax.set_xticks(thresholds)
        ax.set_xlim(2.5, 20.5)

        for method in methods:
            metrics = metrics_map[method.name]
            ys = np.array(
                [metrics["ma3_pct"], metrics["ma5_pct"], metrics["ma10_pct"], metrics["ma20_pct"]],
                dtype=float,
            )
            lw = 2.7 if method.highlight else 2.1
            ms = 7.5 if method.highlight else 6.1
            ls = "-" if method.highlight else ("--" if "Dense" in method.name else "-")
            alpha = 1.0 if method.highlight else 0.92
            ax.plot(
                thresholds,
                ys,
                color=method.color,
                linewidth=lw,
                linestyle=ls,
                marker=method.marker,
                markersize=ms,
                alpha=alpha,
                label=method.short_label,
            )

        ax.set_ylim(0, 55)
        ax.legend(ncol=2 if len(methods) >= 4 else 1, loc="upper left")

    fig.suptitle(
        "Threshold success curves on the GTA main benchmark and the 03/04 support split",
        x=0.01,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "fig03_threshold_success_curves")


def plot_angle_surface_panel(
    ax: plt.Axes,
    cache_path: Path,
    eval_path: Path,
    title: str,
    query_name: str,
    *,
    show_left_ylabel: bool,
    show_right_ylabel: bool,
) -> None:
    sample = load_cache_record(cache_path, eval_path, query_name)
    angles = np.asarray(sample["angles_sorted_deg"], dtype=float)
    distances = np.asarray(sample["distances_sorted_m"], dtype=float)
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    useful_mask = np.asarray(sample["useful_mask"], dtype=bool)

    soften_spines(ax)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Candidate angle (deg)")
    ax.set_ylabel("Final localization error (m)" if show_left_ylabel else "")
    ax.set_xlim(-180, 170)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["−180", "−90", "0", "90", "180"])

    for start_idx, end_idx in sample["segments"]:
        start = angles[start_idx]
        end = angles[min(end_idx, len(angles) - 1)]
        if end_idx >= len(angles):
            end = angles[-1]
        ax.axvspan(start - 5.0, end + 5.0, color=PALETTE["useful_fill"], alpha=0.85, zorder=0)

    ax.plot(
        angles,
        distances,
        color=PALETTE["dense"],
        linewidth=2.0,
        marker="o",
        markersize=3.8,
        zorder=3,
    )
    best_idx = int(np.argmin(distances))
    ax.scatter(
        [angles[best_idx]],
        [distances[best_idx]],
        s=52,
        color=PALETTE["success"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )

    for idx in sample["selected_sorted_idx"]:
        ax.axvline(
            angles[idx],
            color=PALETTE["ours"],
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    ax2 = ax.twinx()
    ax2.plot(
        angles,
        posterior,
        color=PALETTE["rotate"],
        linewidth=2.1,
        linestyle="-",
        zorder=4,
    )
    ax2.fill_between(
        angles,
        posterior,
        0.0,
        color=PALETTE["rotate"],
        alpha=0.12,
        zorder=1,
    )
    ax2.set_ylabel("VOP posterior" if show_right_ylabel else "", color=PALETTE["rotate"])
    ax2.tick_params(axis="y", colors=PALETTE["rotate"], labelright=show_right_ylabel)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(PALETTE["rotate"])
    ax2.grid(False)


def create_mechanism_figure(output_dir: Path) -> List[Path]:
    fig = plt.figure(figsize=(13.6, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.55, 1.0], hspace=0.36, wspace=0.28)

    for idx, (title, query_name) in enumerate(MECHANISM_FIG["samples"]):
        ax = fig.add_subplot(gs[0, idx])
        plot_angle_surface_panel(
            ax,
            MECHANISM_FIG["cache_path"],
            MECHANISM_FIG["eval_path"],
            title,
            query_name,
            show_left_ylabel=(idx == 0),
            show_right_ylabel=(idx == len(MECHANISM_FIG["samples"]) - 1),
        )

    summary = json.loads(read_text(MECHANISM_FIG["angle_shape_path"]))
    oracle_stats = summary["eval_oracle"]["by_delta_m"]
    deltas = [1, 3, 5]
    multimodal = [oracle_stats[str(delta)]["multimodal_ratio"] * 100.0 for delta in deltas]
    multiangle = [oracle_stats[str(delta)]["count_ge_2_ratio"] * 100.0 for delta in deltas]
    mean_count = [oracle_stats[str(delta)]["effective_angle_count_mean"] for delta in deltas]

    ax_sum = fig.add_subplot(gs[1, :])
    soften_spines(ax_sum)
    x = np.arange(len(deltas), dtype=float)
    width = 0.32
    bars1 = ax_sum.bar(
        x - width / 2,
        multiangle,
        width,
        color=PALETTE["rotate"],
        label=">= 2 useful angles",
    )
    bars2 = ax_sum.bar(
        x + width / 2,
        multimodal,
        width,
        color=PALETTE["ours"],
        label="multimodal useful set",
    )
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels([f"+{delta} m" for delta in deltas])
    ax_sum.set_ylabel("Query ratio (%)")
    ax_sum.set_xlabel("Tolerance around the oracle best angle")
    ax_sum.set_ylim(0, max(max(multiangle), max(multimodal)) * 1.35)
    ax_sum.set_title("Dataset-level evidence: useful-angle sets are often not single sharp modes", loc="left", fontweight="bold")

    ax_sum2 = ax_sum.twinx()
    ax_sum2.plot(
        x,
        mean_count,
        color=PALETTE["dense"],
        marker="o",
        linewidth=2.2,
        label="mean useful-angle count",
    )
    ax_sum2.set_ylabel("Mean useful-angle count", color=PALETTE["dense"])
    ax_sum2.tick_params(axis="y", colors=PALETTE["dense"])
    ax_sum2.spines["top"].set_visible(False)
    ax_sum2.spines["right"].set_color(PALETTE["dense"])
    ax_sum2.grid(False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax_sum.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.2,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=PALETTE["text"],
            )
    for xi, yi in zip(x, mean_count):
        ax_sum2.text(
            xi,
            yi + 0.08,
            f"{yi:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=PALETTE["dense"],
        )

    legend_items = [
        patches.Patch(color=PALETTE["rotate"], label=">= 2 useful angles"),
        patches.Patch(color=PALETTE["ours"], label="multimodal useful set"),
        Line2D([0], [0], color=PALETTE["dense"], marker="o", linewidth=2.2, label="mean useful-angle count"),
    ]
    ax_sum.legend(handles=legend_items, loc="upper left", ncol=3)

    fig.suptitle(
        "Why a posterior over useful angles is the right object",
        x=0.01,
        ha="left",
        fontsize=16.2,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "fig04a_angle_surface_mechanism")


def create_supervision_figure(output_dir: Path, metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]]) -> List[Path]:
    metrics_map = metrics_bundle["supervision"]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.2), constrained_layout=True, sharey=True)
    methods = SUPERVISION_METHODS
    y_positions = np.arange(len(methods))[::-1]

    specs = [
        ("Dis@1 (m)", "dis1_m", "lower is better"),
        ("MA@20 (%)", "ma20_pct", "higher is better"),
        ("Fallback (%)", "fallback_pct", "lower is better"),
        ("Worse-than-coarse (%)", "worse_pct", "lower is better"),
    ]

    for ax, (title, key, direction) in zip(axes, specs):
        soften_spines(ax)
        values = [metrics_map[method.name][key] for method in methods]
        padding = (max(values) - min(values)) * 0.18 or 1.0
        ax.set_xlim(min(values) - padding, max(values) + padding)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(direction)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

        for method, y in zip(methods, y_positions):
            value = metrics_map[method.name][key]
            ax.hlines(y, min(values) - padding, value, color=PALETTE["grid"], linewidth=1.0, zorder=1)
            ax.scatter(
                value,
                y,
                s=160 if method.highlight else 120,
                color=method.color,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
            ax.text(
                value,
                y + 0.16,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color=method.color,
                fontweight="bold" if method.highlight else "normal",
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.92)],
            )

        if ax is axes[0]:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([method.short_label for method in methods])
        else:
            ax.set_yticks(y_positions)
            ax.tick_params(labelleft=False)

    fig.suptitle(
        "Supervision comparison on 03/04: useful-angle weighting remains the best sparse line",
        x=0.01,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
    )
    return save_figure(fig, output_dir, "fig04b_supervision_ablation")


def dump_metrics_snapshot(
    output_dir: Path,
    metrics_bundle: Mapping[str, Mapping[str, Dict[str, float]]],
) -> Path:
    snapshot = {}
    for group_name, metrics_map in metrics_bundle.items():
        snapshot[group_name] = metrics_map
    out_path = output_dir / "figure_metrics_snapshot.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    setup_style()

    metrics_bundle = build_paper_metrics_bundle()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    saved_paths.extend(create_report_overview_flowchart(output_dir))
    saved_paths.extend(create_main_pipeline_figure(output_dir))
    saved_paths.extend(create_tradeoff_figure(output_dir, metrics_bundle))
    saved_paths.extend(create_dis1_efficiency_figure(output_dir, metrics_bundle))
    saved_paths.extend(create_dis1_efficiency_vertical_figure(output_dir, metrics_bundle))
    saved_paths.extend(create_threshold_curve_figure(output_dir, metrics_bundle))
    saved_paths.extend(create_mechanism_figure(output_dir))
    saved_paths.extend(create_supervision_figure(output_dir, metrics_bundle))
    saved_paths.append(dump_metrics_snapshot(output_dir, metrics_bundle))

    print("Generated files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
