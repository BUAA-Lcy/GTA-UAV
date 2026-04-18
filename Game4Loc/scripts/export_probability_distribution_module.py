#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Export a compact, real probability-distribution illustration module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from visualize_teacher_signal_example import (
    DEFAULT_META_JSON,
    DEFAULT_RETRIEVAL_CKPT,
    build_example_record,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
REPO_ROOT = GAME4LOC_DIR.parent

QUERY_IMAGE = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset" / "drone" / "images" / "01_0409.JPG"
GALLERY_IMAGE = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset" / "satellite" / "01_5_005_022.png"
OUTPUT_PATH = REPO_ROOT / "outputs_schematic" / "png_modules" / "posterior" / "probability_distribution_module.png"


def prepare_hint_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    image = ImageEnhance.Color(image).enhance(0.08)
    image = ImageEnhance.Contrast(image).enhance(0.72)
    image = ImageEnhance.Brightness(image).enhance(1.10)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.6))
    image = image.convert("RGBA")
    rgba = np.asarray(image).copy()
    rgba[..., 3] = 42
    return rgba


def circular_smooth(values: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    radius = max(int(round(3.0 * sigma)), 1)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[-radius:], values, values[:radius]])
    smooth = np.convolve(padded, kernel, mode="same")[radius:-radius]
    return smooth


def low_pass_periodic(values: np.ndarray, keep: int = 4) -> np.ndarray:
    coeffs = np.fft.rfft(values)
    coeffs[keep + 1 :] = 0
    smooth = np.fft.irfft(coeffs, n=len(values))
    return np.clip(smooth, 0.0, None)


def build_real_distribution() -> tuple[np.ndarray, np.ndarray]:
    args = SimpleNamespace(
        query_name="01_0409.JPG",
        gallery_name="01_5_005_022.png",
        pairs_json=DEFAULT_META_JSON,
        retrieval_checkpoint=DEFAULT_RETRIEVAL_CKPT,
        rotate_step=10.0,
        temperature_m=25.0,
        device="",
        output_dir=REPO_ROOT / "outputs_schematic" / "_tmp_prob_module",
    )
    record = build_example_record(args)
    angles = np.mod(np.asarray(record["expected_angles"], dtype=float), 360.0)
    probs = np.asarray(record["target_probs"], dtype=float)
    order = np.argsort(angles)
    angles = angles[order]
    probs = probs[order]

    probs = low_pass_periodic(probs, keep=5)
    dense_x = np.linspace(0.0, 360.0, 1440, endpoint=False)
    dense_y = np.interp(dense_x, angles, probs, period=360.0)
    dense_y = circular_smooth(dense_y, sigma=9.5)
    dense_y = np.clip(dense_y, 0.0, None)
    if dense_y.max() > 0:
        dense_y = dense_y / dense_y.max()
    dense_y = 0.12 + 0.70 * dense_y
    return dense_x, dense_y


def cluster_peak_regions(angles: np.ndarray, probs: np.ndarray, threshold_ratio: float = 0.70) -> list[tuple[float, float]]:
    threshold = float(np.max(probs) * threshold_ratio)
    strong_angles = [float(a) for a, p in zip(angles, probs) if float(p) >= threshold]
    if not strong_angles:
        return []
    strong_angles = sorted(strong_angles)
    groups: list[list[float]] = [[strong_angles[0]]]
    for angle in strong_angles[1:]:
        if angle - groups[-1][-1] <= 20.0:
            groups[-1].append(angle)
        else:
            groups.append([angle])
    if len(groups) > 1 and (groups[0][0] + 360.0 - groups[-1][-1] <= 20.0):
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    spans: list[tuple[float, float]] = []
    for group in groups:
        left = group[0] - 8.0
        right = group[-1] + 8.0
        spans.append((left, right))
    spans.sort(key=lambda item: ((item[0] + item[1]) / 2.0) % 360.0)
    return spans


def circular_distance(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def top_two_peaks(x: np.ndarray, y: np.ndarray) -> list[float]:
    prev = np.roll(y, 1)
    nxt = np.roll(y, -1)
    peak_idx = np.where((y >= prev) & (y >= nxt))[0]
    if peak_idx.size == 0:
        return []
    ordered = sorted(peak_idx.tolist(), key=lambda idx: float(y[idx]), reverse=True)
    chosen: list[int] = []
    for idx in ordered:
        angle = float(x[idx])
        if all(circular_distance(angle, float(x[old])) >= 55.0 for old in chosen):
            chosen.append(idx)
        if len(chosen) == 2:
            break
    return [float(x[idx]) for idx in chosen]


def add_span(ax: plt.Axes, span: tuple[float, float], *, color: str, alpha: float) -> None:
    left, right = span
    if left < 0.0:
        right_chunk = 360.0 - (left + 360.0)
        left_chunk = right
        if right_chunk >= 12.0:
            ax.axvspan(left + 360.0, 360.0, color=color, alpha=alpha, zorder=1)
        if left_chunk >= 12.0:
            ax.axvspan(0.0, right, color=color, alpha=alpha, zorder=1)
    elif right > 360.0:
        right_chunk = 360.0 - left
        left_chunk = right - 360.0
        if right_chunk >= 12.0:
            ax.axvspan(left, 360.0, color=color, alpha=alpha, zorder=1)
        if left_chunk >= 12.0:
            ax.axvspan(0.0, right - 360.0, color=color, alpha=alpha, zorder=1)
    else:
        ax.axvspan(left, right, color=color, alpha=alpha, zorder=1)


def main() -> None:
    for path in (QUERY_IMAGE, GALLERY_IMAGE, DEFAULT_META_JSON, DEFAULT_RETRIEVAL_CKPT):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dense_x, dense_y = build_real_distribution()

    raw_angles = np.array(
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0,
         180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0, 330.0, 340.0, 350.0],
        dtype=float,
    )
    # Recompute with the same real pair so the highlighted peak regions follow the actual distribution.
    args = SimpleNamespace(
        query_name="01_0409.JPG",
        gallery_name="01_5_005_022.png",
        pairs_json=DEFAULT_META_JSON,
        retrieval_checkpoint=DEFAULT_RETRIEVAL_CKPT,
        rotate_step=10.0,
        temperature_m=25.0,
        device="",
        output_dir=REPO_ROOT / "outputs_schematic" / "_tmp_prob_module",
    )
    record = build_example_record(args)
    raw_probs = np.asarray(record["target_probs"], dtype=float)
    raw_angles = np.mod(np.asarray(record["expected_angles"], dtype=float), 360.0)
    order = np.argsort(raw_angles)
    raw_angles = raw_angles[order]
    raw_probs = raw_probs[order]
    peak_spans = cluster_peak_regions(raw_angles, raw_probs)
    smooth_peaks = top_two_peaks(dense_x, dense_y)

    fig = plt.figure(figsize=(4.8, 2.7), dpi=360, facecolor="white")
    ax = fig.add_axes([0.09, 0.30, 0.86, 0.50], facecolor="white")

    uav_hint = prepare_hint_image(QUERY_IMAGE, (300, 205))
    sat_hint = prepare_hint_image(GALLERY_IMAGE, (300, 205))
    ax.imshow(uav_hint, extent=(46, 126, 0.22, 0.80), aspect="auto", interpolation="bilinear", zorder=0)
    ax.imshow(sat_hint, extent=(234, 314, 0.22, 0.80), aspect="auto", interpolation="bilinear", zorder=0)

    if smooth_peaks:
        add_span(ax, (smooth_peaks[0] - 18.0, smooth_peaks[0] + 18.0), color="#D7ECD4", alpha=0.62)
    if len(smooth_peaks) >= 2:
        add_span(ax, (smooth_peaks[1] - 15.0, smooth_peaks[1] + 15.0), color="#F2DEC8", alpha=0.68)
    elif len(peak_spans) >= 2:
        add_span(ax, peak_spans[0], color="#F2DEC8", alpha=0.68)

    ax.plot(dense_x, dense_y, color="#26292F", linewidth=2.2, solid_capstyle="round", zorder=3)

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(0.0, 0.90)
    ax.set_yticks([])
    ax.set_xticks([0.0, 360.0])
    ax.set_xticklabels(["0°", "360°"], fontsize=10.0, color="#383C43")
    ax.set_xlabel("Probability distribution", fontsize=10.7, color="#2F333A", labelpad=7)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("#666B74")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=0)

    fig.savefig(OUTPUT_PATH, dpi=360, facecolor="white", transparent=False)
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
