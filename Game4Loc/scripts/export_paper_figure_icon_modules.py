#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Export a no-text, icon-like VOP paper module library grounded in real assets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
REPO_ROOT = GAME4LOC_DIR.parent
if str(GAME4LOC_DIR) not in sys.path:
    sys.path.insert(0, str(GAME4LOC_DIR))

from game4loc.matcher.sparse_sp_lg import SparseSpLgMatcher
from plot_vop_shortpaper_figures import (
    MECHANISM_FIG,
    PALETTE as BASE_PALETTE,
    build_paper_metrics_bundle,
    load_cache_record,
    load_rotated_query,
    setup_style,
)
from visualize_paper7_pair_vop import draw_projected_point
from visualize_teacher_signal_example import (
    DEFAULT_META_JSON,
    DEFAULT_RETRIEVAL_CKPT,
    build_example_record,
)

from export_paper_figure_modules import (
    DATA_ROOT,
    FIG_ROOT,
    FONT_BOLD,
    FONT_REGULAR,
    PAIR_ROOT,
    PALETTE,
    SHORTPAPER_ROOT,
    TEACHER_ROOT,
    VISLOC_CKPT,
    VOP_CKPT,
    crop_to_alpha,
    draw_arrow_icon,
    draw_drone_icon,
    draw_pin_icon,
    fig_to_image,
    fit_inside,
    load_font,
    make_round_mask,
    numpy_to_pca_rgb,
    prepare_main_feature_bundle,
    tensor_from_rgb,
    text_size,
)


OUTPUT_ROOT = REPO_ROOT / "outputs_schematic"
PNG_ROOT = OUTPUT_ROOT / "png_modules"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
PREVIEW_MODULE_ROOT = PREVIEW_ROOT / "module_previews"
MANIFEST_ROOT = OUTPUT_ROOT / "manifest"


@dataclass
class ModuleEntry:
    file_name: str
    category: str
    description: str
    source_file: str
    derived_from_real_image: bool
    transparent_background: bool
    recommended_usage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for the icon-like module library.",
    )
    return parser.parse_args()


def ensure_dirs(output_root: Path) -> None:
    png_root = output_root / "png_modules"
    preview_root = output_root / "previews"
    preview_module_root = preview_root / "module_previews"
    manifest_root = output_root / "manifest"
    for category in ("input", "backbone", "features", "posterior", "verification", "results", "icons"):
        (png_root / category).mkdir(parents=True, exist_ok=True)
        (preview_module_root / category).mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)


def rounded_thumb(
    image: Image.Image,
    *,
    size: Tuple[int, int] = (260, 260),
    accent: str = PALETTE["panel_edge"],
    radius: int = 26,
    border: int = 4,
    fill: Tuple[int, int, int, int] = (255, 255, 255, 245),
) -> Image.Image:
    image = ImageOps.fit(image.convert("RGBA"), size, method=Image.Resampling.LANCZOS)
    thumb = Image.new("RGBA", size, (0, 0, 0, 0))
    thumb.alpha_composite(image, (0, 0))
    thumb.putalpha(make_round_mask(size, radius))

    pad = 14
    canvas = Image.new("RGBA", (size[0] + pad * 2, size[1] + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (1, 1, canvas.width - 2, canvas.height - 2),
        radius=radius + 8,
        fill=fill,
        outline=accent,
        width=border,
    )
    canvas.alpha_composite(thumb, (pad, pad))
    return crop_to_alpha(canvas, padding=4)


def feature_thumb(
    tensor: torch.Tensor,
    *,
    accent: str,
    size: Tuple[int, int] = (220, 220),
) -> Image.Image:
    rgb = numpy_to_pca_rgb(tensor)
    image = Image.fromarray(rgb).convert("RGBA").resize(size, Image.Resampling.NEAREST)
    return rounded_thumb(image, size=size, accent=accent, radius=24, border=4)


def make_preview_version(module_img: Image.Image, caption: str) -> Image.Image:
    module_img = crop_to_alpha(module_img, padding=3)
    caption_font = load_font(18, bold=False)
    dummy = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    cap_w, cap_h = text_size(draw, caption, caption_font)
    canvas = Image.new(
        "RGBA",
        (max(module_img.width + 28, cap_w + 30), module_img.height + cap_h + 42),
        (0, 0, 0, 0),
    )
    x = (canvas.width - module_img.width) // 2
    y = 12
    canvas.alpha_composite(module_img, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((4, 4, canvas.width - 5, canvas.height - 5), radius=18, outline="#CBD5E1", width=2)
    draw.text(((canvas.width - cap_w) / 2, module_img.height + 18), caption, font=caption_font, fill="#5B6675")
    return canvas


def save_module(
    output_root: Path,
    image: Image.Image,
    *,
    category: str,
    stem: str,
    description: str,
    source_file: str,
    derived_from_real_image: bool,
    recommended_usage: str,
    manifest_entries: List[ModuleEntry],
) -> Path:
    png_root = output_root / "png_modules"
    preview_root = output_root / "previews" / "module_previews"
    image = crop_to_alpha(image, padding=6)
    out_path = png_root / category / f"{stem}.png"
    image.save(out_path)
    make_preview_version(image, stem).save(preview_root / category / f"{stem}_preview.png")
    manifest_entries.append(
        ModuleEntry(
            file_name=str(out_path),
            category=category,
            description=description,
            source_file=source_file,
            derived_from_real_image=bool(derived_from_real_image),
            transparent_background=True,
            recommended_usage=recommended_usage,
        )
    )
    return out_path


def contact_sheet(paths: Sequence[Path], *, title: str, columns: int = 4) -> Image.Image:
    title_font = load_font(32, bold=True)
    caption_font = load_font(16, bold=False)
    thumbs = [Image.open(path).convert("RGBA") for path in paths]
    previews = [make_preview_version(thumb, path.stem) for thumb, path in zip(thumbs, paths)]
    tile_w = max(tile.width for tile in previews)
    tile_h = max(tile.height for tile in previews)
    rows = int(math.ceil(len(previews) / float(columns)))
    width = columns * tile_w + (columns + 1) * 20
    height = rows * tile_h + (rows + 1) * 20 + 58
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), title, font=title_font, fill="#1F2937")
    for idx, tile in enumerate(previews):
        row, col = divmod(idx, columns)
        x = 20 + col * tile_w
        y = 64 + row * tile_h
        canvas.alpha_composite(tile, (x, y))
    return canvas


def simple_panel(size: Tuple[int, int], *, accent: str, fill: Tuple[int, int, int, int] = (255, 255, 255, 245), radius: int = 28, border: int = 4) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3), radius=radius, fill=fill, outline=accent, width=border)
    return canvas


def draw_network_stack(size: Tuple[int, int] = (120, 120), accent: str = PALETTE["axis"]) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    rects = [
        (24, 32, 74, 92),
        (34, 22, 84, 82),
        (44, 12, 94, 72),
    ]
    fills = [(232, 238, 247, 255), (220, 231, 246, 255), (206, 224, 245, 255)]
    for rect, fill in zip(rects, fills):
        draw.rounded_rectangle(rect, radius=12, fill=fill, outline=accent, width=3)
    return crop_to_alpha(canvas, padding=3)


def draw_rotation_icon_no_text(size: Tuple[int, int] = (180, 180)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    cx, cy = size[0] // 2, size[1] // 2
    radius = 58
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), start=25, end=325, fill=PALETTE["rotate"], width=8)
    draw.polygon([(cx + 6, cy - radius - 10), (cx + 36, cy - radius + 2), (cx + 4, cy - radius + 16)], fill=PALETTE["rotate"])
    for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        x = cx + int(round((radius + 18) * math.cos(angle)))
        y = cy + int(round((radius + 18) * math.sin(angle)))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(125, 148, 180, 160))
    return crop_to_alpha(canvas, padding=4)


def draw_softmax_icon(size: Tuple[int, int] = (250, 150)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bar_xs = [24, 50, 76, 102]
    bar_heights = [28, 52, 38, 22]
    for x, h in zip(bar_xs, bar_heights):
        draw.rounded_rectangle((x, size[1] - 24 - h, x + 16, size[1] - 24), radius=7, fill="#8CB6DD")
    draw.line((132, size[1] // 2, 170, size[1] // 2), fill=PALETTE["muted"], width=5)
    draw.polygon([(168, size[1] // 2 - 12), (194, size[1] // 2), (168, size[1] // 2 + 12)], fill=PALETTE["muted"])
    pts = []
    curve_x = np.linspace(195, size[0] - 18, 60)
    curve_y = 110 - 54 * np.exp(-((curve_x - 214) / 23.0) ** 2) - 26 * np.exp(-((curve_x - 240) / 18.0) ** 2)
    for x, y in zip(curve_x, curve_y):
        pts.append((float(x), float(y)))
    draw.line(pts, fill=PALETTE["rotate"], width=6, joint="curve")
    return crop_to_alpha(canvas, padding=4)


def draw_topk_icon(size: Tuple[int, int] = (210, 92)) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    xs = [20, 56, 92, 128, 164]
    for idx, x in enumerate(xs):
        fill = PALETTE["ours"] if idx < 4 else (160, 168, 178, 90)
        outline = PALETTE["ours"] if idx < 4 else "#A7AFBA"
        draw.rounded_rectangle((x, 20, x + 24, 72), radius=9, fill=fill, outline=outline, width=3)
    return crop_to_alpha(canvas, padding=4)


def make_clean_match_vis_no_text(
    stats: Mapping[str, object],
    gallery_tensor: torch.Tensor,
    query_tensor: torch.Tensor,
) -> Image.Image:
    mk0 = stats.get("mk0")
    mk1 = stats.get("mk1")
    image1_vis_rot = stats.get("image1_vis_rot")
    h_mask = stats.get("h_mask")
    if not isinstance(mk0, np.ndarray) or not isinstance(mk1, np.ndarray) or mk0.shape[0] == 0:
        raise RuntimeError("No sparse matches available for clean visualization.")

    def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
        array = tensor.detach().cpu().float().numpy()
        if array.ndim == 4:
            array = array[0]
        array = np.clip(array.transpose(1, 2, 0), 0.0, 1.0)
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)

    img0 = tensor_to_uint8_rgb(gallery_tensor)
    img1 = tensor_to_uint8_rgb(image1_vis_rot if isinstance(image1_vis_rot, torch.Tensor) else query_tensor)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    gap = 18
    canvas = np.full((max(h0, h1), w0 + gap + w1, 3), 255, dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap : w0 + gap + w1] = img1

    draw_mask = np.ones((mk0.shape[0],), dtype=bool)
    if h_mask is not None:
        mask = np.asarray(h_mask).reshape(-1).astype(bool)
        if mask.shape[0] == mk0.shape[0]:
            draw_mask = mask
    draw_indices = np.flatnonzero(draw_mask)
    if draw_indices.size == 0:
        draw_indices = np.arange(mk0.shape[0], dtype=np.int32)
    if draw_indices.size > 70:
        draw_indices = draw_indices[np.linspace(0, draw_indices.size - 1, num=70, dtype=np.int32)]

    for idx in draw_indices:
        x0, y0 = int(round(float(mk0[idx, 0]))), int(round(float(mk0[idx, 1])))
        x1, y1 = int(round(float(mk1[idx, 0]))), int(round(float(mk1[idx, 1])))
        x1 += w0 + gap
        color = (38, 138, 86)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, lineType=cv2.LINE_AA)

    return Image.fromarray(canvas).convert("RGBA")


def run_sparse_visuals_no_text(query_path: Path, gallery_path: Path, best_angle: float, device: str) -> Dict[str, Image.Image]:
    gallery_img = Image.open(gallery_path).convert("RGB")
    query_img = Image.open(query_path).convert("RGB")
    gallery_tensor = tensor_from_rgb(gallery_img, device=device)
    query_tensor = tensor_from_rgb(query_img, device=device)

    matcher = SparseSpLgMatcher(device=device)
    outputs: Dict[str, Image.Image] = {}
    for key, angle in (("sparse", 0.0), ("ours", float(best_angle))):
        _, _, _, _, stats, _, _ = matcher._run_matching_for_angles(
            gallery_tensor,
            query_tensor,
            yaw0=None,
            yaw1=None,
            candidate_angles=[float(angle)],
            reproj_threshold=float(matcher.sparse_ransac_reproj_threshold),
            run_name=f"icon_{key}",
        )
        outputs[key] = make_clean_match_vis_no_text(stats, gallery_tensor, query_tensor)
    return outputs


def build_input_pair_icon(query_image: Image.Image, gallery_image: Image.Image) -> Image.Image:
    q = fit_inside(rounded_thumb(query_image, size=(220, 220), accent=PALETTE["rotate"]), (240, 240))
    g = fit_inside(rounded_thumb(gallery_image, size=(220, 220), accent=PALETTE["dense"]), (240, 240))
    canvas = Image.new("RGBA", (560, 260), (0, 0, 0, 0))
    canvas.alpha_composite(q, (0, 10))
    arrow = crop_to_alpha(draw_arrow_icon(size=(120, 56)), padding=0)
    canvas.alpha_composite(fit_inside(arrow, (110, 56)), (228, 92))
    canvas.alpha_composite(g, (320, 10))
    return crop_to_alpha(canvas, padding=4)


def build_backbone_icon(query_image: Image.Image, gallery_image: Image.Image, fq_tile: Image.Image, fg_tile: Image.Image) -> Image.Image:
    canvas = simple_panel((820, 300), accent=PALETTE["panel_edge"])
    q = fit_inside(rounded_thumb(query_image, size=(130, 130), accent=PALETTE["rotate"]), (150, 150))
    g = fit_inside(rounded_thumb(gallery_image, size=(130, 130), accent=PALETTE["dense"]), (150, 150))
    fq = fit_inside(fq_tile, (150, 150))
    fg = fit_inside(fg_tile, (150, 150))
    net = fit_inside(draw_network_stack(), (120, 120))
    canvas.alpha_composite(q, (36, 34))
    canvas.alpha_composite(g, (36, 156))
    canvas.alpha_composite(net, (350, 92))
    canvas.alpha_composite(fq, (596, 34))
    canvas.alpha_composite(fg, (596, 156))
    draw = ImageDraw.Draw(canvas)
    for y0, y1 in ((96, 118), (218, 182)):
        draw.line((200, y0, 350, 118 if y0 < 150 else 182), fill=PALETTE["muted"], width=5)
    for x0, y0, x1, y1 in ((470, 118, 596, 96), (470, 182, 596, 218)):
        draw.line((x0, y0, x1, y1), fill=PALETTE["muted"], width=5)
        draw.polygon([(x1 - 10, y1 - 12), (x1 + 18, y1), (x1 - 10, y1 + 12)], fill=PALETTE["muted"])
    return crop_to_alpha(canvas, padding=6)


def build_rotation_candidates_icon(query_image: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (360, 360), (0, 0, 0, 0))
    base_query = ImageOps.fit(query_image.convert("RGBA"), (188, 188), method=Image.Resampling.LANCZOS)
    thumb = rounded_thumb(base_query, size=(188, 188), accent=PALETTE["rotate"])
    canvas.alpha_composite(thumb, ((canvas.width - thumb.width) // 2, (canvas.height - thumb.height) // 2))
    draw = ImageDraw.Draw(canvas)
    cx, cy = canvas.width // 2, canvas.height // 2
    ring_r = 132
    draw.arc((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r), start=35, end=330, fill=PALETTE["rotate"], width=7)
    draw.polygon([(cx + 22, cy - ring_r - 10), (cx + 54, cy - ring_r + 2), (cx + 20, cy - ring_r + 22)], fill=PALETTE["rotate"])
    for angle in np.linspace(0, 2 * math.pi, 18, endpoint=False):
        x = cx + int(round(ring_r * math.cos(angle)))
        y = cy + int(round(ring_r * math.sin(angle)))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(116, 145, 182, 170))
    return crop_to_alpha(canvas, padding=6)


def build_concat_icon() -> Image.Image:
    canvas = Image.new("RGBA", (250, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    colors = [PALETTE["rotate"], PALETTE["teacher"], PALETTE["clean30"], PALETTE["useful"]]
    ys = [26, 58, 90, 122]
    for y, color in zip(ys, colors):
        draw.rounded_rectangle((28, y, 98, y + 24), radius=8, fill=color)
        draw.line((98, y + 12, 156, 74), fill=color, width=5)
    draw.rounded_rectangle((160, 50, 212, 98), radius=14, fill=PALETTE["axis"])
    return crop_to_alpha(canvas, padding=4)


def build_head_icon(fusion_preview: Image.Image) -> Image.Image:
    preview = fit_inside(fusion_preview, (120, 120))
    canvas = Image.new("RGBA", (500, 180), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    preview_box = simple_panel((146, 146), accent=PALETTE["panel_edge"], radius=20, border=3)
    preview_box.alpha_composite(preview, ((preview_box.width - preview.width) // 2, (preview_box.height - preview.height) // 2))
    canvas.alpha_composite(preview_box, (0, 17))
    blocks = [
        (190, 40, 86, 98, PALETTE["orange_fill"], PALETTE["clean30"]),
        (306, 48, 72, 82, PALETTE["green_fill"], PALETTE["green"]),
        (402, 52, 64, 74, PALETTE["purple_fill"], PALETTE["useful"]),
    ]
    for x, y, w, h, fill, outline in blocks:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=outline, width=3)
    for x0, x1, y in ((146, 190, 90), (276, 306, 90), (378, 402, 90)):
        draw.line((x0, y, x1, y), fill=PALETTE["muted"], width=5)
        draw.polygon([(x1 - 4, y - 11), (x1 + 16, y), (x1 - 4, y + 11)], fill=PALETTE["muted"])
    return crop_to_alpha(canvas, padding=4)


def build_posterior_icon(sample: Mapping[str, object]) -> Image.Image:
    angles = np.deg2rad(sample["angles_sorted_deg"])
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    selected_indices = sample["selected_sorted_idx"]

    fig = plt.figure(figsize=(3.3, 3.3))
    ax = fig.add_subplot(111, projection="polar")
    bars = ax.bar(
        angles,
        posterior,
        width=np.deg2rad(9.0),
        color=PALETTE["rotate"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    for idx in selected_indices:
        bars[idx].set_facecolor(PALETTE["ours"])
        bars[idx].set_alpha(0.95)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.grid(color="#D7DCE5", alpha=0.7, linewidth=0.7)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_topk_selected_angles_icon(query_path: Path, selected_angles: Sequence[float], best_rank: int) -> Image.Image:
    canvas = Image.new("RGBA", (420, 420), (0, 0, 0, 0))
    slots = [(0, 0), (212, 0), (0, 212), (212, 212)]
    for idx, (angle, (x, y)) in enumerate(zip(selected_angles[:4], slots)):
        rotated = Image.fromarray(load_rotated_query(query_path, float(angle), size=160)).convert("RGBA")
        accent = PALETTE["ours"] if idx == best_rank else PALETTE["rotate"]
        tile = rounded_thumb(rotated, size=(170, 170), accent=accent, radius=22, border=5 if idx == best_rank else 3)
        canvas.alpha_composite(tile, (x, y))
    return crop_to_alpha(canvas, padding=4)


def build_vop_icon(
    fg_tile: Image.Image,
    fqrot_tile: Image.Image,
    prod_tile: Image.Image,
    diff_tile: Image.Image,
    posterior_tile: Image.Image,
) -> Image.Image:
    canvas = simple_panel((1020, 320), accent=PALETTE["ours"], radius=30)
    tiles = [fit_inside(tile, (130, 130)) for tile in (fg_tile, fqrot_tile, prod_tile, diff_tile)]
    xs = [26, 176, 326, 476]
    for x, tile in zip(xs, tiles):
        slot = simple_panel((138, 138), accent=PALETTE["panel_edge"], radius=18, border=2)
        slot.alpha_composite(tile, ((slot.width - tile.width) // 2, (slot.height - tile.height) // 2))
        canvas.alpha_composite(slot, (x, 92))
    head = fit_inside(build_head_icon(prod_tile), (270, 170))
    canvas.alpha_composite(head, (638, 72))
    posterior = fit_inside(posterior_tile, (210, 210))
    canvas.alpha_composite(posterior, (812, 56))
    draw = ImageDraw.Draw(canvas)
    for x0, x1 in ((614, 638), (786, 812)):
        draw.line((x0, 160, x1, 160), fill=PALETTE["muted"], width=5)
        draw.polygon([(x1 - 4, 148), (x1 + 16, 160), (x1 - 4, 172)], fill=PALETTE["muted"])
    return crop_to_alpha(canvas, padding=6)


def build_verification_box_icon(match_module: Image.Image) -> Image.Image:
    thumb = fit_inside(match_module, (660, 260))
    canvas = simple_panel((720, 330), accent=PALETTE["green"], radius=28)
    canvas.alpha_composite(thumb, ((canvas.width - thumb.width) // 2, (canvas.height - thumb.height) // 2))
    return crop_to_alpha(canvas, padding=6)


def build_result_icons(pair_summary: Mapping[str, object]) -> Tuple[Image.Image, Image.Image]:
    gallery_name = pair_summary["gallery_name"]
    gallery_path = DATA_ROOT / "satellite" / gallery_name
    gallery_rgb = np.asarray(Image.open(gallery_path).convert("RGB"))
    projected = pair_summary.get("geometry_selected_projected_xy")
    projected_xy = None if projected is None else (float(projected[0]), float(projected[1]))
    result_rgb = draw_projected_point(gallery_rgb, projected_xy, "")
    basemap = rounded_thumb(Image.fromarray(gallery_rgb).convert("RGBA"), size=(340, 340), accent=PALETTE["dense"], radius=28)
    result = rounded_thumb(Image.fromarray(result_rgb).convert("RGBA"), size=(340, 340), accent=PALETTE["ours"], radius=28)
    return basemap, result


def build_tradeoff_icon(metrics_bundle: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Image.Image:
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.6), constrained_layout=True)
    datasets = [
        (
            "GTA same-area",
            axes[0],
            [
                ("dense DKM", "^", "#2C2F35", 80),
                ("LoFTR", "D", "#9B6F25", 58),
                ("sparse", "o", "#99A1AC", 72),
                ("sparse + rotate90", "o", PALETTE["rotate"], 76),
                ("sparse + VOP (ours)", "o", PALETTE["ours"], 88),
            ],
        ),
        (
            "03/04",
            axes[1],
            [
                ("dense DKM, no rotate", "^", "#2C2F35", 80),
                ("LoFTR", "D", "#9B6F25", 58),
                ("SuperPoint", "o", "#99A1AC", 72),
                ("SuperPoint + Rotate", "o", PALETTE["rotate"], 76),
                ("Ours (SuperPoint + VOP)", "o", PALETTE["ours"], 88),
            ],
        ),
    ]
    for dataset_name, ax, method_specs in datasets:
        dataset = metrics_bundle[dataset_name]
        points = []
        for method_name, marker, color, size in method_specs:
            if method_name not in dataset:
                continue
            m = dataset[method_name]
            fps = 1.0 / max(float(m["mean_total_s"]), 1e-6)
            dis = float(m["dis1_m"])
            points.append((fps, dis, marker, color, size, method_name))
        if not points:
            continue
        frontier = sorted(points, key=lambda item: item[0])
        xs = [p[0] for p in frontier]
        ys = [p[1] for p in frontier]
        ax.plot(xs, ys, linestyle=(0, (4, 3)), linewidth=1.3, color="#8B95A3", zorder=1)
        for fps, dis, marker, color, size, method_name in points:
            ax.scatter(fps, dis, marker=marker, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(
            "",
            xy=(max(xs) * 1.08, min(ys) - 8.0),
            xytext=(max(xs) * 0.84, min(ys) + 2.0),
            arrowprops=dict(arrowstyle="-|>", color="#1F6B46", lw=1.8, mutation_scale=11),
        )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.35, color="#DDE3EA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False, length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_facecolor("white")
        ax.set_xlim(min(xs) * 0.8, max(xs) * 1.25)
        ax.set_ylim(max(ys) + 12.0, min(ys) - 12.0)
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_error_curve_icon(teacher_record: Mapping[str, object]) -> Image.Image:
    angles = np.asarray(teacher_record["expected_angles"], dtype=float)
    distances = np.asarray(teacher_record["distances_m"], dtype=float)
    best_idx = int(teacher_record["best_index"])
    finite = np.isfinite(distances)
    ymax = float(np.nanmax(distances[finite])) if np.any(finite) else 1.0

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(angles, distances, color="#2F3137", linewidth=2.2, marker="o", markersize=2.8)
    ax.scatter([angles[best_idx]], [distances[best_idx]], s=78, color=PALETTE["ours"], edgecolor="white", linewidth=1.0, zorder=5)
    ax.axhline(distances[best_idx], linestyle="--", linewidth=1.4, color=PALETTE["rotate"], alpha=0.75)
    ax.set_xlim(float(np.nanmin(angles)), float(np.nanmax(angles)))
    ax.set_ylim(0.0, ymax * 1.05)
    ax.grid(True, alpha=0.28, color="#DDE3EA")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_soft_target_icon(teacher_record: Mapping[str, object]) -> Image.Image:
    angles = np.asarray(teacher_record["expected_angles"], dtype=float)
    probs = np.asarray(teacher_record["target_probs"], dtype=float)
    best_idx = int(teacher_record["best_index"])

    dense_x = np.linspace(float(np.min(angles)), float(np.max(angles)), 900)
    dense_y = np.interp(dense_x, angles, probs)
    kernel = np.exp(-0.5 * (np.arange(-18, 19) / 6.0) ** 2)
    kernel = kernel / kernel.sum()
    dense_y = np.convolve(dense_y, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.fill_between(dense_x, dense_y, 0.0, color="#A8C6E4", alpha=0.45)
    ax.plot(dense_x, dense_y, color="#447FB7", linewidth=2.8)
    ax.scatter(angles, probs, s=28, color="#4E84BA", edgecolor="white", linewidth=0.8, zorder=4)
    ax.axvline(angles[best_idx], color=PALETTE["ours"], linewidth=1.6, alpha=0.9)
    ax.scatter([angles[best_idx]], [probs[best_idx]], s=150, color=PALETTE["ours"], edgecolor="white", linewidth=1.2, zorder=5)
    ax.set_xlim(float(np.min(angles)), float(np.max(angles)))
    ax.set_ylim(0.0, max(float(np.max(dense_y)) * 1.12, 0.02))
    ax.grid(True, alpha=0.28, color="#DDE3EA")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_useful_angle_set_icon(sample: Mapping[str, object]) -> Image.Image:
    angles = np.asarray(sample["angles_sorted_deg"], dtype=float)
    distances = np.asarray(sample["distances_sorted_m"], dtype=float)
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    useful_mask = np.asarray(sample["useful_mask"], dtype=bool)
    selected_idx = np.asarray(sample["selected_sorted_idx"], dtype=int)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax2 = ax.twinx()
    ax.plot(angles, distances, color="#2F3137", linewidth=2.0, marker="o", markersize=2.6, zorder=3)
    ax2.plot(angles, posterior, color=PALETTE["rotate"], linewidth=2.0, alpha=0.95, zorder=2)
    ax2.fill_between(angles, posterior, 0.0, color="#A8C6E4", alpha=0.25, zorder=1)
    for idx in np.flatnonzero(useful_mask):
        x0 = angles[idx] - 5.0
        x1 = angles[idx] + 5.0
        ax.axvspan(x0, x1, color="#D7F0D6", alpha=0.45, zorder=0)
    for idx in selected_idx:
        ax.axvline(angles[idx], color=PALETTE["ours"], linewidth=1.4, alpha=0.82, zorder=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.28, color="#DDE3EA")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax2.tick_params(labelright=False, length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax2.set_ylabel("")
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_pair_confidence_icon() -> Image.Image:
    x = np.linspace(0.0, 100.0, 300)
    y = 1.0 / (1.0 + np.exp((x - 30.0) / 10.0))
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.plot(x, y, color=PALETTE["green"], linewidth=2.8)
    ax.fill_between(x, y, 0.0, color="#B8DFC3", alpha=0.35)
    ax.axvline(30.0, color="#8893A1", linestyle="--", linewidth=1.3)
    ax.scatter([30.0], [0.5], s=44, color=PALETTE["green"], zorder=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.28, color="#DDE3EA")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.05)
    return crop_to_alpha(fig_to_image(fig), padding=2)


def build_teacher_record(output_root: Path, device: str) -> Dict[str, object]:
    tmp_dir = output_root / "_tmp_teacher_record"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        query_name="01_0409.JPG",
        gallery_name="01_5_005_022.png",
        pairs_json=DEFAULT_META_JSON,
        retrieval_checkpoint=DEFAULT_RETRIEVAL_CKPT,
        rotate_step=10.0,
        temperature_m=25.0,
        device=device,
        output_dir=tmp_dir,
    )
    return build_example_record(args)


def write_manifest(output_root: Path, entries: Sequence[ModuleEntry]) -> None:
    manifest_path = output_root / "manifest" / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(entry) for entry in entries], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_readme(output_root: Path, entries: Sequence[ModuleEntry], missing_inputs: Sequence[str]) -> None:
    total = len(entries)
    real_count = sum(1 for entry in entries if entry.derived_from_real_image)
    redraw_count = total - real_count
    lines = [
        "# Schematic Paper Figure Modules",
        "",
        "This folder contains the no-text, icon-like module library for the VOP paper figures.",
        "",
        f"- Total exported modules: `{total}`",
        f"- From real local images or real computed results: `{real_count}`",
        f"- Redrawn / vectorized schematic modules: `{redraw_count}`",
        "",
        "## Notes",
        "",
        "- All module PNGs use transparent backgrounds.",
        "- Module images are intentionally text-free so they can be re-labeled in PPT / Illustrator later.",
        "- Preview sheets keep light captions only for human inspection.",
        "",
        "## Missing Inputs",
        "",
    ]
    if missing_inputs:
        for item in missing_inputs:
            lines.append(f"- Missing: `{item}`")
    else:
        lines.append("- No required input file is missing.")
    lines += [
        "",
        "## Module Inventory",
        "",
    ]
    for entry in entries:
        lines += [
            f"### `{Path(entry.file_name).name}`",
            f"- category: `{entry.category}`",
            f"- source: `{entry.source_file}`",
            f"- real-derived: `{entry.derived_from_real_image}`",
            f"- usage: {entry.recommended_usage}",
            f"- description: {entry.description}",
            "",
        ]
    (output_root / "manifest" / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    ensure_dirs(output_root)
    setup_style()

    required_sources = [
        VISLOC_CKPT,
        VOP_CKPT,
        PAIR_ROOT / "pair01_vop_summary.json",
        DEFAULT_META_JSON,
        DEFAULT_RETRIEVAL_CKPT,
    ]
    missing_inputs = [str(path) for path in required_sources if not Path(path).is_file()]
    if missing_inputs:
        print("Missing required inputs:")
        for item in missing_inputs:
            print(item)
        return

    manifest_entries: List[ModuleEntry] = []
    saved_paths: Dict[str, Path] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main_bundle = prepare_main_feature_bundle(device=device)
    main_sample = main_bundle["sample"]
    selected_angles = [float(angle) for angle in main_sample["eval_record"]["selected_angles_deg"]]
    best_rank = int(main_sample["best_selected_rank"])
    sparse_visuals = run_sparse_visuals_no_text(
        query_path=main_bundle["query_path"],
        gallery_path=main_bundle["gallery_path"],
        best_angle=float(main_bundle["best_angle"]),
        device=device,
    )
    pair_summary = json.loads((PAIR_ROOT / "pair01_vop_summary.json").read_text(encoding="utf-8"))
    teacher_record = build_teacher_record(output_root=output_root, device=device)
    mechanism_query = MECHANISM_FIG["samples"][2][1]
    useful_sample = load_cache_record(MECHANISM_FIG["cache_path"], MECHANISM_FIG["eval_path"], mechanism_query)
    metrics_bundle = build_paper_metrics_bundle()

    query_image = main_bundle["query_image"]
    gallery_image = main_bundle["gallery_image"]

    query_tile = rounded_thumb(query_image, size=(300, 300), accent=PALETTE["rotate"], radius=28)
    gallery_tile = rounded_thumb(gallery_image, size=(300, 300), accent=PALETTE["dense"], radius=28)
    input_pair = build_input_pair_icon(query_image, gallery_image)
    basemap_icon, result_icon = build_result_icons(pair_summary)

    fq_icon = feature_thumb(main_bundle["raw_query_map"], accent=PALETTE["rotate"], size=(240, 240))
    fg_icon = feature_thumb(main_bundle["raw_gallery_map"], accent=PALETTE["dense"], size=(240, 240))
    fqrot_icon = feature_thumb(main_bundle["enc_query_rot"], accent=PALETTE["teacher"], size=(240, 240))
    fg_enc_icon = feature_thumb(main_bundle["enc_gallery_map"], accent=PALETTE["rotate"], size=(220, 220))
    fqrot_enc_icon = feature_thumb(main_bundle["enc_query_rot"], accent=PALETTE["teacher"], size=(220, 220))
    prod_icon = feature_thumb(main_bundle["enc_gallery_map"] * main_bundle["enc_query_rot"], accent=PALETTE["clean30"], size=(220, 220))
    diff_icon = feature_thumb(torch.abs(main_bundle["enc_gallery_map"] - main_bundle["enc_query_rot"]), accent=PALETTE["useful"], size=(220, 220))

    posterior_icon = build_posterior_icon(main_sample)
    topk_icon = build_topk_selected_angles_icon(main_bundle["query_path"], selected_angles, best_rank)
    rotation_candidates_icon = build_rotation_candidates_icon(query_image)
    concat_icon = build_concat_icon()
    head_icon = build_head_icon(prod_icon)
    backbone_icon = build_backbone_icon(query_image, gallery_image, fq_icon, fg_icon)
    vop_icon = build_vop_icon(fg_enc_icon, fqrot_enc_icon, prod_icon, diff_icon, posterior_icon)

    sparse_match_icon = rounded_thumb(sparse_visuals["sparse"], size=(520, 250), accent=PALETTE["sparse"], radius=20)
    ours_match_icon = rounded_thumb(sparse_visuals["ours"], size=(520, 250), accent=PALETTE["ours"], radius=20)
    compare_icon = build_input_pair_icon(sparse_match_icon, ours_match_icon)
    verification_box_icon = build_verification_box_icon(ours_match_icon)

    tradeoff_icon = build_tradeoff_icon(metrics_bundle)
    error_curve_icon = build_error_curve_icon(teacher_record)
    soft_target_icon = build_soft_target_icon(teacher_record)
    useful_angle_icon = build_useful_angle_set_icon(useful_sample)
    pair_conf_icon = build_pair_confidence_icon()

    drone_icon = crop_to_alpha(draw_drone_icon(size=(210, 160)), padding=0)
    pin_icon = crop_to_alpha(draw_pin_icon(size=(120, 170)), padding=0)
    arrow_icon = crop_to_alpha(draw_arrow_icon(size=(130, 52)), padding=0)
    rotation_icon = draw_rotation_icon_no_text()
    softmax_icon = draw_softmax_icon()
    topk_chip = draw_topk_icon()

    save_specs = [
        ("input", "uav_input_tile_icon", query_tile, "Real UAV input tile, simplified as a text-free icon module.", str(main_bundle["query_path"]), True, "Use for the UAV query image in a paper diagram."),
        ("input", "satellite_input_tile_icon", gallery_tile, "Real satellite top-1 tile, simplified as a text-free icon module.", str(main_bundle["gallery_path"]), True, "Use for the gallery / retrieved satellite input block."),
        ("input", "input_pair_icon", input_pair, "Simplified side-by-side input pair icon built from the real query-gallery pair.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']}", True, "Use when a compact pair module is easier than placing q and g separately."),
        ("input", "final_localization_basemap_icon", basemap_icon, "Raw satellite basemap tile used under the final localization result.", str(DATA_ROOT / 'satellite' / pair_summary['gallery_name']), True, "Use as the result-stage satellite basemap block."),
        ("backbone", "frozen_backbone_icon", backbone_icon, "Minimal frozen-backbone schematic driven by the real query, gallery, and feature maps.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']} | {VISLOC_CKPT}", True, "Use as the backbone stage icon without any embedded labels."),
        ("backbone", "learnable_vop_module_icon", vop_icon, "Minimal VOP schematic built from real encoded maps and the real posterior.", str(VOP_CKPT), True, "Use as the VOP stage outer box in a diagram."),
        ("features", "feature_map_f_q_icon", fq_icon, "Raw query feature map visualized with PCA colors.", str(VISLOC_CKPT), True, "Use as the F_q feature block."),
        ("features", "feature_map_f_g_icon", fg_icon, "Raw gallery feature map visualized with PCA colors.", str(VISLOC_CKPT), True, "Use as the F_g feature block."),
        ("features", "feature_map_f_q_rot_icon", fqrot_icon, "Rotated query feature map from the real selected VOP angle.", str(VOP_CKPT), True, "Use as the F_{q,rot} feature block."),
        ("features", "fusion_block_f_g_icon", fg_enc_icon, "Encoded gallery feature tile for the VOP fusion stage.", str(VOP_CKPT), True, "Use as the [F_g] fusion input."),
        ("features", "fusion_block_f_q_rot_icon", fqrot_enc_icon, "Encoded rotated-query feature tile for the VOP fusion stage.", str(VOP_CKPT), True, "Use as the [F_{q,rot}] fusion input."),
        ("features", "fusion_block_f_g_mul_f_q_rot_icon", prod_icon, "Element-wise product fusion tile from the real VOP sample.", str(VOP_CKPT), True, "Use as the [F_g ⊙ F_{q,rot}] block."),
        ("features", "fusion_block_abs_f_g_minus_f_q_rot_icon", diff_icon, "Absolute-difference fusion tile from the real VOP sample.", str(VOP_CKPT), True, "Use as the [|F_g - F_{q,rot}|] block."),
        ("features", "feature_concat_icon", concat_icon, "No-text schematic icon for feature concatenation / fusion.", "vector redraw", False, "Place between fusion inputs and the VOP head."),
        ("features", "conv_relu_gap_head_icon", head_icon, "Minimal head icon showing the fusion preview feeding a three-block head.", str(VOP_CKPT), True, "Use as the learnable VOP scoring head."),
        ("features", "rotation_candidates_icon", rotation_candidates_icon, "Real query tile wrapped by a rotation orbit schematic.", str(main_bundle["query_path"]), True, "Use for the candidate-angle generation step."),
        ("posterior", "posterior_probability_distribution_icon", posterior_icon, "Real VOP posterior rendered as a no-text polar icon.", str(MECHANISM_FIG["eval_path"]), True, "Use for the posterior output block."),
        ("posterior", "topk_selected_angles_icon", topk_icon, "Real top-k selected angle hypotheses shown as rotated query thumbnails.", str(main_bundle["query_path"]), True, "Use for the selected top-k hypotheses step."),
        ("posterior", "soft_orientation_target_icon", soft_target_icon, "Real teacher soft target re-rendered without titles or axis text.", str(DEFAULT_META_JSON), True, "Use in the supervision branch of a figure."),
        ("posterior", "useful_angle_set_icon", useful_angle_icon, "Real multimodal useful-angle example re-rendered as a no-text plot icon.", mechanism_query, True, "Use when illustrating multimodal useful-angle structure."),
        ("posterior", "pair_confidence_weight_icon", pair_conf_icon, "Minimal pair-weight sigmoid icon from the current training recipe.", "real training recipe: center=30, scale=10", False, "Use in training-supervision schematics."),
        ("verification", "superpoint_lightglue_geometric_verification_icon", verification_box_icon, "Verification-stage panel containing the real VOP-guided sparse match visual.", str(main_bundle["gallery_path"]), True, "Use as the downstream geometric verification stage."),
        ("verification", "sparse_no_vop_match_icon", sparse_match_icon, "Real sparse matching visual without VOP prior.", str(main_bundle["query_path"]), True, "Use for the plain sparse branch."),
        ("verification", "ours_vop_guided_match_icon", ours_match_icon, "Real VOP-guided sparse matching visual.", str(main_bundle["query_path"]), True, "Use for the Ours branch."),
        ("verification", "match_visual_comparison_icon", compare_icon, "Compact side-by-side comparison between sparse and VOP-guided matches.", str(main_bundle["query_path"]), True, "Use when one block should summarize the match-quality difference."),
        ("results", "final_localization_result_icon", result_icon, "Final localization result with the real projected pin on the satellite tile.", str(PAIR_ROOT / "pair01_vop_summary.json"), True, "Use as the final result block."),
        ("results", "accuracy_efficiency_tradeoff_icon", tradeoff_icon, "Minimal no-text redraw of the current runtime-vs-Dis@1 plot.", str(SHORTPAPER_ROOT / "fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png"), True, "Use as a compact quantitative inset."),
        ("results", "localization_error_curve_icon", error_curve_icon, "No-text redraw of the real teacher localization error curve.", str(DEFAULT_META_JSON), True, "Use as the mechanism-analysis curve block."),
        ("icons", "uav_topview_icon", drone_icon, "Standalone UAV icon without text.", "vector redraw", False, "Use as a small pictogram in the input stage."),
        ("icons", "red_pin_icon", pin_icon, "Standalone red location pin icon.", "vector redraw", False, "Use as a reusable localization marker."),
        ("icons", "arrow_right_icon", arrow_icon, "Standalone right arrow icon.", "vector redraw", False, "Use between modules when hand-assembling a figure."),
        ("icons", "rotation_arrow_icon", rotation_icon, "Standalone rotation arrow icon without any label text.", "vector redraw", False, "Use near query rotation modules."),
        ("icons", "softmax_icon", softmax_icon, "Standalone softmax-style icon without text.", "vector redraw", False, "Use near posterior-generation stages."),
        ("icons", "topk_chip_icon", topk_chip, "Standalone top-k selection icon without text.", "vector redraw", False, "Use near selected-angle stages."),
    ]

    for category, stem, image, description, source_file, derived_from_real_image, recommended_usage in save_specs:
        saved_paths[stem] = save_module(
            output_root,
            image,
            category=category,
            stem=stem,
            description=description,
            source_file=source_file,
            derived_from_real_image=derived_from_real_image,
            recommended_usage=recommended_usage,
            manifest_entries=manifest_entries,
        )

    preview_groups = {
        "assembly_preview_pipeline": [
            saved_paths["uav_input_tile_icon"],
            saved_paths["satellite_input_tile_icon"],
            saved_paths["frozen_backbone_icon"],
            saved_paths["rotation_candidates_icon"],
            saved_paths["learnable_vop_module_icon"],
            saved_paths["superpoint_lightglue_geometric_verification_icon"],
            saved_paths["final_localization_result_icon"],
        ],
        "assembly_preview_features": [
            saved_paths["feature_map_f_q_icon"],
            saved_paths["feature_map_f_g_icon"],
            saved_paths["feature_map_f_q_rot_icon"],
            saved_paths["fusion_block_f_g_icon"],
            saved_paths["fusion_block_f_q_rot_icon"],
            saved_paths["fusion_block_f_g_mul_f_q_rot_icon"],
            saved_paths["fusion_block_abs_f_g_minus_f_q_rot_icon"],
            saved_paths["feature_concat_icon"],
            saved_paths["conv_relu_gap_head_icon"],
            saved_paths["posterior_probability_distribution_icon"],
            saved_paths["topk_selected_angles_icon"],
        ],
        "assembly_preview_analysis": [
            saved_paths["accuracy_efficiency_tradeoff_icon"],
            saved_paths["localization_error_curve_icon"],
            saved_paths["soft_orientation_target_icon"],
            saved_paths["useful_angle_set_icon"],
            saved_paths["pair_confidence_weight_icon"],
            saved_paths["uav_topview_icon"],
            saved_paths["red_pin_icon"],
            saved_paths["rotation_arrow_icon"],
            saved_paths["softmax_icon"],
            saved_paths["topk_chip_icon"],
        ],
    }
    for stem, paths in preview_groups.items():
        title = stem.replace("_", " ").title()
        contact_sheet(paths, title=title, columns=4).save(output_root / "previews" / f"{stem}.png")

    write_manifest(output_root, manifest_entries)
    write_readme(output_root, manifest_entries, missing_inputs)

    print("Generated files:")
    for key in sorted(saved_paths):
        print(saved_paths[key])
    print(output_root / "manifest" / "README.md")
    print(output_root / "manifest" / "manifest.json")


if __name__ == "__main__":
    main()
