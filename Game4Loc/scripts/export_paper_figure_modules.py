#!/home/lcy/miniconda3/envs/gtauav/bin/python
"""Export a reusable PNG module library for the VOP paper figures."""

from __future__ import annotations

import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
GAME4LOC_DIR = SCRIPT_DIR.parent
REPO_ROOT = GAME4LOC_DIR.parent
if str(GAME4LOC_DIR) not in sys.path:
    sys.path.insert(0, str(GAME4LOC_DIR))

from game4loc.dataset.visloc import get_transforms
from game4loc.matcher.sparse_sp_lg import SparseSpLgMatcher
from game4loc.models.model import DesModel
from game4loc.orientation import load_vop_checkpoint
from game4loc.orientation.vop import rotate_feature_map
from plot_vop_shortpaper_figures import MAIN_SAMPLE, PALETTE as BASE_PALETTE
from plot_vop_shortpaper_figures import build_paper_metrics_bundle, load_cache_record, setup_style
from visualize_paper7_pair_vop import draw_projected_point

OUTPUT_ROOT = REPO_ROOT / "outputs"
PNG_ROOT = OUTPUT_ROOT / "png_modules"
PREVIEW_ROOT = OUTPUT_ROOT / "previews"
PREVIEW_MODULE_ROOT = PREVIEW_ROOT / "module_previews"
MANIFEST_ROOT = OUTPUT_ROOT / "manifest"

DATA_ROOT = GAME4LOC_DIR / "data" / "UAV_VisLoc_dataset"
FIG_ROOT = GAME4LOC_DIR / "figures"
PAIR_ROOT = FIG_ROOT / "pair_vop_assets_20260414"
TEACHER_ROOT = FIG_ROOT / "teacher_signal_example_20260414"
SHORTPAPER_ROOT = FIG_ROOT / "vop_shortpaper_20260411"
REVIEW_ROOT = FIG_ROOT / "vop_shortpaper_20260415_review"

VISLOC_CKPT = GAME4LOC_DIR / "pretrained" / "visloc" / "vit_base_eva_visloc_same_area_0407.pth"
VOP_CKPT = GAME4LOC_DIR / "work_dir" / "vop" / "vop_0409_useful5_weight30_e6.pth"

FONT_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]

PALETTE = {
    **BASE_PALETTE,
    "panel_fill": "#FFFFFF",
    "panel_edge": "#D7DCE5",
    "title": "#1F2329",
    "subtitle": "#5F6772",
    "green": "#2E6B3F",
    "blue_fill": "#DDEBF8",
    "orange_fill": "#FDEADF",
    "green_fill": "#E5F3E8",
    "purple_fill": "#F0EBFA",
    "light_gray": "#EDF1F5",
}


@dataclass
class ModuleEntry:
    file_name: str
    category: str
    description: str
    source_file: str
    derived_from_real_image: bool
    transparent_background: bool
    recommended_usage: str


def find_font(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return None


FONT_REGULAR = find_font(FONT_REGULAR_CANDIDATES)
FONT_BOLD = find_font(FONT_BOLD_CANDIDATES) or FONT_REGULAR


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_path = FONT_BOLD if bold else FONT_REGULAR
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def ensure_dirs() -> None:
    for category in ("input", "backbone", "features", "posterior", "verification", "results", "icons"):
        (PNG_ROOT / category).mkdir(parents=True, exist_ok=True)
        (PREVIEW_MODULE_ROOT / category).mkdir(parents=True, exist_ok=True)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)


def rgba(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGBA")


def alpha_bbox(image: Image.Image) -> Tuple[int, int, int, int] | None:
    alpha = image.convert("RGBA").getchannel("A")
    return alpha.getbbox()


def crop_to_alpha(image: Image.Image, padding: int = 8) -> Image.Image:
    image = image.convert("RGBA")
    bbox = alpha_bbox(image)
    if bbox is None:
        return image
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    return image.crop((left, top, right, bottom))


def make_round_mask(size: Tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def fit_inside(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    img = image.convert("RGBA").copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    return img


def card_from_content(
    content: Image.Image,
    *,
    title: str,
    subtitle: str | None = None,
    accent: str = PALETTE["panel_edge"],
    fill: str = PALETTE["panel_fill"],
    padding: int = 22,
    title_gap: int = 14,
) -> Image.Image:
    content = content.convert("RGBA")
    title_font = load_font(28, bold=True)
    sub_font = load_font(18, bold=False)
    dummy = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    title_w, title_h = text_size(draw, title, title_font)
    subtitle_w = subtitle_h = 0
    if subtitle:
        subtitle_w, subtitle_h = text_size(draw, subtitle, sub_font)

    width = max(content.width + padding * 2, title_w + padding * 2, subtitle_w + padding * 2)
    header_h = padding + title_h + (subtitle_h + 8 if subtitle else 0)
    height = header_h + title_gap + content.height + padding
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    box_top = header_h + title_gap
    draw.rounded_rectangle(
        (2, box_top, width - 3, height - 3),
        radius=22,
        fill=fill,
        outline=accent,
        width=3,
    )
    draw.text((padding, padding // 2), title, font=title_font, fill=PALETTE["title"])
    if subtitle:
        draw.text((padding, padding // 2 + title_h + 4), subtitle, font=sub_font, fill=PALETTE["subtitle"])

    content_x = (width - content.width) // 2
    content_y = box_top + (height - box_top - content.height) // 2
    canvas.alpha_composite(content, (content_x, content_y))
    return crop_to_alpha(canvas, padding=4)


def image_tile(
    image: Image.Image,
    *,
    title: str,
    accent: str,
    tag: str | None = None,
    footer: str | None = None,
    size: Tuple[int, int] = (360, 360),
) -> Image.Image:
    image = ImageOps.fit(image.convert("RGBA"), size, method=Image.Resampling.LANCZOS)
    rounded = Image.new("RGBA", size, (0, 0, 0, 0))
    rounded.alpha_composite(image, (0, 0))
    rounded.putalpha(make_round_mask(size, 26))

    canvas = Image.new("RGBA", (size[0] + 34, size[1] + 34), (0, 0, 0, 0))
    box = ImageDraw.Draw(canvas)
    box.rounded_rectangle((1, 1, canvas.width - 2, canvas.height - 2), radius=30, fill="white", outline=accent, width=4)
    canvas.alpha_composite(rounded, (17, 17))
    draw = ImageDraw.Draw(canvas)

    if tag:
        tag_font = load_font(26, bold=True)
        tag_w, tag_h = text_size(draw, tag, tag_font)
        pill_w = tag_w + 22
        draw.rounded_rectangle((28, 28, 28 + pill_w, 28 + tag_h + 12), radius=18, fill=(255, 255, 255, 220))
        draw.text((39, 33), tag, font=tag_font, fill=PALETTE["title"])

    if footer:
        footer_font = load_font(20, bold=False)
        footer_w, footer_h = text_size(draw, footer, footer_font)
        x0 = canvas.width - footer_w - 32
        y0 = canvas.height - footer_h - 24
        draw.rounded_rectangle((x0 - 10, y0 - 6, x0 + footer_w + 10, y0 + footer_h + 6), radius=16, fill=(255, 255, 255, 220))
        draw.text((x0, y0), footer, font=footer_font, fill=accent)

    return card_from_content(canvas, title=title, accent=accent)


def fig_to_image(fig: plt.Figure, *, dpi: int = 320) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def vector_icon_canvas(size: Tuple[int, int] = (256, 256)) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def draw_arrow_icon(size: Tuple[int, int] = (220, 88)) -> Image.Image:
    img, draw = vector_icon_canvas(size)
    draw.rounded_rectangle((1, 1, size[0] - 2, size[1] - 2), radius=24, fill="white", outline=PALETTE["panel_edge"], width=3)
    y = size[1] // 2
    draw.line((32, y, size[0] - 56, y), fill=PALETTE["muted"], width=7)
    draw.polygon([(size[0] - 58, y - 16), (size[0] - 22, y), (size[0] - 58, y + 16)], fill=PALETTE["muted"])
    return crop_to_alpha(img, padding=4)


def draw_rotation_icon(size: Tuple[int, int] = (220, 220)) -> Image.Image:
    img, draw = vector_icon_canvas(size)
    draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3), radius=30, fill="white", outline=PALETTE["panel_edge"], width=3)
    cx, cy = size[0] // 2, size[1] // 2
    radius = 64
    draw.arc((cx - radius, cy - radius, cx + radius, cy + radius), start=35, end=325, fill=PALETTE["rotate"], width=8)
    draw.polygon([(cx + 10, cy - radius - 8), (cx + 40, cy - radius + 8), (cx + 8, cy - radius + 20)], fill=PALETTE["rotate"])
    font = load_font(28, bold=True)
    w, h = text_size(draw, "36 angles", font)
    draw.text((cx - w // 2, cy - h // 2), "36 angles", font=font, fill=PALETTE["title"])
    return crop_to_alpha(img, padding=4)


def draw_pin_icon(size: Tuple[int, int] = (160, 220)) -> Image.Image:
    img, draw = vector_icon_canvas(size)
    cx = size[0] // 2
    draw.ellipse((28, 20, size[0] - 28, 120), fill=PALETTE["ours"], outline="white", width=6)
    draw.ellipse((58, 50, size[0] - 58, 90), fill="white")
    draw.polygon([(cx, size[1] - 14), (42, 88), (size[0] - 42, 88)], fill=PALETTE["ours"])
    draw.line((cx, 110, cx, size[1] - 28), fill=PALETTE["ours"], width=22)
    return crop_to_alpha(img, padding=4)


def draw_drone_icon(size: Tuple[int, int] = (260, 200)) -> Image.Image:
    img, draw = vector_icon_canvas(size)
    draw.rounded_rectangle((2, 2, size[0] - 3, size[1] - 3), radius=26, fill="white", outline=PALETTE["panel_edge"], width=3)
    cx, cy = size[0] // 2, size[1] // 2
    arm = 58
    body = 20
    rotor_r = 18
    draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=PALETTE["axis"], width=8)
    draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill=PALETTE["axis"], width=8)
    draw.rounded_rectangle((cx - body, cy - 28, cx + body, cy + 28), radius=12, fill=PALETTE["rotate"])
    for dx, dy in ((-arm, -arm), (arm, arm), (-arm, arm), (arm, -arm)):
        draw.ellipse((cx + dx - rotor_r, cy + dy - rotor_r, cx + dx + rotor_r, cy + dy + rotor_r), outline=PALETTE["axis"], width=6)
        draw.ellipse((cx + dx - 5, cy + dy - 5, cx + dx + 5, cy + dy + 5), fill=PALETTE["axis"])
    return crop_to_alpha(img, padding=4)


def draw_badge(text: str, *, fill: str, text_fill: str = "white") -> Image.Image:
    font = load_font(24, bold=True)
    dummy = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)
    w, h = text_size(draw, text, font)
    img = Image.new("RGBA", (w + 34, h + 22), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=18, fill=fill)
    draw.text((17, 11), text, font=font, fill=text_fill)
    return img


def numpy_to_pca_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    c, h, w = arr.shape
    flat = arr.reshape(c, h * w).T
    flat = flat - flat.mean(axis=0, keepdims=True)
    if flat.shape[0] <= 1:
        rgb = np.zeros((h * w, 3), dtype=np.float32)
    else:
        _, _, vh = np.linalg.svd(flat, full_matrices=False)
        basis = vh[: min(3, vh.shape[0])]
        proj = flat @ basis.T
        if proj.shape[1] < 3:
            proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode="constant")
        rgb = proj[:, :3]
    rgb = rgb.reshape(h, w, 3)
    rgb -= rgb.min(axis=(0, 1), keepdims=True)
    denom = rgb.max(axis=(0, 1), keepdims=True)
    denom[denom < 1e-6] = 1.0
    rgb = rgb / denom
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def feature_tile(tensor: torch.Tensor, *, label: str, accent: str) -> Image.Image:
    rgb = numpy_to_pca_rgb(tensor)
    image = Image.fromarray(rgb).convert("RGBA").resize((240, 240), Image.Resampling.NEAREST)
    image = ImageOps.expand(image, border=12, fill="white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, image.width - 1, image.height - 1), radius=20, outline=accent, width=4)
    return card_from_content(image, title=label, accent=accent)


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def make_preview_version(module_img: Image.Image) -> Image.Image:
    module_img = crop_to_alpha(module_img, padding=2)
    canvas = Image.new("RGBA", (module_img.width + 24, module_img.height + 24), (0, 0, 0, 0))
    canvas.alpha_composite(module_img, (12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((6, 6, canvas.width - 7, canvas.height - 7), radius=20, outline="#CBD5E1", width=3)
    return canvas


def extract_top_panel(image: Image.Image, index: int, count: int = 3) -> Image.Image:
    image = image.convert("RGBA")
    w, h = image.size
    left = int(w * 0.12) + index * int(w * 0.26)
    panel_w = int(w * 0.225)
    top = int(h * 0.12)
    bottom = int(h * 0.64)
    return image.crop((left, top, left + panel_w, bottom))


def clean_plot_card(path: Path, *, title: str, accent: str, crop_bbox: Tuple[int, int, int, int] | None = None) -> Image.Image:
    image = load_rgba(path)
    if crop_bbox is not None:
        image = image.crop(crop_bbox)
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white_bg.alpha_composite(image)
    return card_from_content(white_bg, title=title, accent=accent)


def build_tradeoff_module() -> Image.Image:
    return clean_plot_card(
        SHORTPAPER_ROOT / "fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png",
        title="Accuracy-Efficiency Tradeoff",
        accent=PALETTE["green"],
    )


def build_error_curve_module() -> Image.Image:
    return clean_plot_card(
        TEACHER_ROOT / "teacher_pair_localization_error_curve.png",
        title="Localization Error Curve",
        accent=PALETTE["dense"],
    )


def build_soft_target_module() -> Image.Image:
    return clean_plot_card(
        TEACHER_ROOT / "teacher_pair_soft_orientation_target_v2_smooth.png",
        title="Soft Orientation Target",
        accent=PALETTE["rotate"],
    )


def build_useful_angle_panel_module() -> Image.Image:
    source = load_rgba(REVIEW_ROOT / "fig04a_angle_surface_mechanism.png")
    panel = extract_top_panel(source, index=2)
    return card_from_content(panel, title="Useful-Angle Set", subtitle="multimodal real example", accent=PALETTE["useful"])


def build_pair_confidence_curve_module() -> Image.Image:
    x = np.linspace(0.0, 100.0, 256)
    y = 1.0 / (1.0 + np.exp((x - 30.0) / 10.0))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(x, y, color=PALETTE["green"], linewidth=3.0)
    ax.fill_between(x, y, 0.0, color=PALETTE["green"], alpha=0.16)
    ax.axvline(30.0, color=PALETTE["muted"], linestyle="--", linewidth=1.4)
    ax.scatter([30.0], [0.5], color=PALETTE["green"], s=40, zorder=3)
    ax.set_xlabel("Best teacher distance $d^*$ (m)")
    ax.set_ylabel("Pair weight")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.4)
    ax.set_title("Pair-confidence weighting", loc="left", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    image = fig_to_image(fig)
    return card_from_content(image, title="Pair Confidence Weight", subtitle="real training recipe: center=30, scale=10", accent=PALETTE["green"])


def tensor_from_rgb(image: Image.Image, size: Tuple[int, int] = (384, 384), device: str = "cpu") -> torch.Tensor:
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))[None, ...].to(device)
    return tensor


def make_clean_match_vis(stats: Mapping[str, object], gallery_tensor: torch.Tensor, query_tensor: torch.Tensor, *, caption: str) -> Image.Image:
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
    gap = 20
    canvas = np.full((max(h0, h1), w0 + gap + w1, 3), 255, dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0 + gap : w0 + gap + w1] = img1

    inlier_mask = None
    if h_mask is not None:
        inlier_mask = np.asarray(h_mask).reshape(-1).astype(bool)
        if inlier_mask.shape[0] != mk0.shape[0]:
            inlier_mask = None
    draw_mask = inlier_mask if inlier_mask is not None else np.ones((mk0.shape[0],), dtype=bool)
    draw_indices = np.flatnonzero(draw_mask)
    if draw_indices.size == 0:
        draw_indices = np.arange(mk0.shape[0], dtype=np.int32)
    if draw_indices.size > 80:
        draw_indices = draw_indices[np.linspace(0, draw_indices.size - 1, num=80, dtype=np.int32)]

    for idx in draw_indices:
        x0, y0 = int(round(float(mk0[idx, 0]))), int(round(float(mk0[idx, 1])))
        x1, y1 = int(round(float(mk1[idx, 0]))), int(round(float(mk1[idx, 1])))
        x1 += w0 + gap
        color = (41, 156, 90)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(canvas, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 34, 34), 2, lineType=cv2.LINE_AA)
    return Image.fromarray(canvas).convert("RGBA")


def run_sparse_visuals(query_path: Path, gallery_path: Path, best_angle: float, device: str) -> Dict[str, Image.Image]:
    gallery_img = Image.open(gallery_path).convert("RGB")
    query_img = Image.open(query_path).convert("RGB")
    gallery_tensor = tensor_from_rgb(gallery_img, device=device)
    query_tensor = tensor_from_rgb(query_img, device=device)

    matcher = SparseSpLgMatcher(device=device)
    outputs = {}
    for key, angle in (("sparse", 0.0), ("ours", float(best_angle))):
        _, _, _, _, stats, _, _ = matcher._run_matching_for_angles(
            gallery_tensor,
            query_tensor,
            yaw0=None,
            yaw1=None,
            candidate_angles=[float(angle)],
            reproj_threshold=float(matcher.sparse_ransac_reproj_threshold),
            run_name=f"module_{key}",
        )
        caption = (
            f"{'Sparse (no VOP)' if key == 'sparse' else 'Ours'}"
            f" | angle = {angle:+.0f}°"
            f" | matches = {int(stats.get('n_kept', 0))}"
            f" | inliers = {int(np.asarray(stats.get('h_mask')).sum()) if stats.get('h_mask') is not None else 0}"
        )
        outputs[key] = make_clean_match_vis(stats, gallery_tensor, query_tensor, caption=caption)
    return outputs


def prepare_main_feature_bundle(device: str) -> Dict[str, object]:
    sample = load_cache_record(MAIN_SAMPLE["cache_path"], MAIN_SAMPLE["eval_path"], MAIN_SAMPLE["query_name"])
    query_name = sample["cache_record"]["query_name"]
    gallery_name = sample["cache_record"]["gallery_name"]
    query_path = DATA_ROOT / "drone" / "images" / query_name
    gallery_path = DATA_ROOT / "satellite" / gallery_name

    model = DesModel("vit_base_patch16_rope_reg1_gap_256.sbb_in1k", pretrained=False, img_size=384, share_weights=True)
    model.load_state_dict(torch.load(VISLOC_CKPT, map_location="cpu"), strict=False)
    model = model.to(device).eval()
    data_config = model.get_config()
    val_transforms, _, _ = get_transforms((384, 384), mean=data_config["mean"], std=data_config["std"])

    query_rgb = np.asarray(Image.open(query_path).convert("RGB"))
    gallery_rgb = np.asarray(Image.open(gallery_path).convert("RGB"))
    query_tensor = val_transforms(image=query_rgb)["image"].unsqueeze(0).to(device)
    gallery_tensor = val_transforms(image=gallery_rgb)["image"].unsqueeze(0).to(device)

    vop = load_vop_checkpoint(str(VOP_CKPT), device=device)
    best_angle = float(sample["eval_record"]["selected_angles_deg"][sample["best_selected_rank"]])

    with torch.no_grad():
        raw_gallery_map = model.extract_feature_map(gallery_tensor, branch="img2")
        raw_query_map = model.extract_feature_map(query_tensor, branch="img1")
        enc_gallery_map, enc_query_map = vop.encode(raw_gallery_map, raw_query_map)
        enc_query_rot = rotate_feature_map(enc_query_map, best_angle)

    return {
        "sample": sample,
        "query_path": query_path,
        "gallery_path": gallery_path,
        "query_image": Image.open(query_path).convert("RGBA"),
        "gallery_image": Image.open(gallery_path).convert("RGBA"),
        "raw_gallery_map": raw_gallery_map.detach().cpu(),
        "raw_query_map": raw_query_map.detach().cpu(),
        "enc_gallery_map": enc_gallery_map.detach().cpu(),
        "enc_query_map": enc_query_map.detach().cpu(),
        "enc_query_rot": enc_query_rot.detach().cpu(),
        "best_angle": best_angle,
    }


def build_posterior_plot(sample: Mapping[str, object]) -> Image.Image:
    angles = np.deg2rad(sample["angles_sorted_deg"])
    posterior = np.asarray(sample["posterior_sorted"], dtype=float)
    selected_indices = sample["selected_sorted_idx"]
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    bars = ax.bar(
        angles,
        posterior,
        width=np.deg2rad(9.0),
        color=PALETTE["rotate"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.35,
    )
    for idx in selected_indices:
        bars[idx].set_facecolor(PALETTE["ours"])
        bars[idx].set_alpha(0.90)
    uniform = np.full_like(posterior, 1.0 / len(posterior))
    ax.plot(angles, uniform, color=PALETTE["muted"], linewidth=1.1, alpha=0.65)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("VOP posterior", loc="left", fontweight="bold", pad=18)
    ax.set_rlabel_position(112)
    ax.set_ylim(0.0, posterior.max() * 1.2)
    ax.tick_params(labelsize=8.2)
    ax.grid(color=PALETTE["grid"], alpha=0.82)
    ax.text(0.5, -0.11, "highlighted = selected top-4", transform=ax.transAxes, ha="center", va="top", fontsize=9.0, color=PALETTE["muted"])
    return card_from_content(fig_to_image(fig), title="Posterior Probability Distribution", accent=PALETTE["rotate"])


def build_topk_angles_module(selected_angles: Sequence[float]) -> Image.Image:
    font = load_font(26, bold=True)
    tag_font = load_font(20, bold=False)
    chip_w = 138
    chip_h = 64
    gap = 14
    width = len(selected_angles) * chip_w + (len(selected_angles) - 1) * gap
    canvas = Image.new("RGBA", (width, chip_h + 30), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for idx, angle in enumerate(selected_angles):
        x0 = idx * (chip_w + gap)
        fill = PALETTE["ours"] if idx == 0 else PALETTE["blue_fill"]
        outline = PALETTE["ours"] if idx == 0 else PALETTE["rotate"]
        text_fill = "white" if idx == 0 else PALETTE["rotate"]
        draw.rounded_rectangle((x0, 12, x0 + chip_w, 12 + chip_h), radius=20, fill=fill, outline=outline, width=3)
        angle_text = f"{angle:+.0f}°"
        tw, th = text_size(draw, angle_text, font)
        draw.text((x0 + (chip_w - tw) / 2, 22), angle_text, font=font, fill=text_fill)
        rank_text = f"rank {idx + 1}"
        rw, _ = text_size(draw, rank_text, tag_font)
        draw.text((x0 + (chip_w - rw) / 2, 22 + th + 4), rank_text, font=tag_font, fill=text_fill)
    return card_from_content(canvas, title="Top-k Selected Angles", accent=PALETTE["ours"], subtitle="real selected hypotheses from the main sample")


def build_rotation_module(query_image: Image.Image) -> Image.Image:
    image = ImageOps.fit(query_image.convert("RGBA"), (250, 250), method=Image.Resampling.LANCZOS)
    base = Image.new("RGBA", (330, 330), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.ellipse((40, 40, 290, 290), outline=PALETTE["rotate"], width=8)
    base.alpha_composite(image, (40, 40))
    draw.arc((24, 24, 306, 306), start=40, end=320, fill=PALETTE["rotate"], width=8)
    draw.polygon([(214, 36), (252, 44), (227, 72)], fill=PALETTE["rotate"])
    badge = draw_badge("36 candidate angles", fill=PALETTE["rotate"])
    base.alpha_composite(badge, ((base.width - badge.width) // 2, 274))
    return card_from_content(base, title="Rotation Operation", accent=PALETTE["rotate"], subtitle="query rotation prior to hypothesis scoring")


def build_concat_symbol_module() -> Image.Image:
    canvas = Image.new("RGBA", (360, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    colors = [PALETTE["rotate"], PALETTE["teacher"], PALETTE["clean30"], PALETTE["useful"]]
    xs = [26, 96, 166, 236]
    for x, color in zip(xs, colors):
        draw.rounded_rectangle((x, 56, x + 54, 146), radius=16, fill=color)
    draw.line((306, 36, 306, 166), fill=PALETTE["axis"], width=4)
    draw.line((306, 36, 326, 36), fill=PALETTE["axis"], width=4)
    draw.line((306, 166, 326, 166), fill=PALETTE["axis"], width=4)
    draw.rounded_rectangle((332, 60, 350, 142), radius=8, fill=PALETTE["axis"])
    font = load_font(26, bold=True)
    draw.text((72, 20), "concat", font=font, fill=PALETTE["title"])
    return card_from_content(canvas, title="Feature Concatenation / Fusion", accent=PALETTE["axis"])


def build_head_box_module(fusion_preview: Image.Image) -> Image.Image:
    fusion_preview = fit_inside(fusion_preview, (180, 180))
    canvas = Image.new("RGBA", (560, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    box_specs = [
        (210, 42, 112, "1×1 Conv", PALETTE["orange_fill"], PALETTE["clean30"]),
        (340, 42, 94, "ReLU", PALETTE["green_fill"], PALETTE["green"]),
        (450, 42, 88, "GAP", PALETTE["purple_fill"], PALETTE["useful"]),
    ]
    draw.rounded_rectangle((0, 20, 180, 200), radius=24, fill="white", outline=PALETTE["panel_edge"], width=3)
    canvas.alpha_composite(fusion_preview, ((180 - fusion_preview.width) // 2, (220 - fusion_preview.height) // 2))
    for x, y, w, text, fill, outline in box_specs:
        draw.rounded_rectangle((x, y, x + w, y + 120), radius=22, fill=fill, outline=outline, width=3)
        font = load_font(24, bold=True)
        tw, th = text_size(draw, text, font)
        draw.text((x + (w - tw) / 2, y + 50 - th / 2), text, font=font, fill=PALETTE["title"])
    for x0, x1 in ((184, 210), (322, 340), (434, 450)):
        draw.line((x0, 102, x1, 102), fill=PALETTE["muted"], width=5)
        draw.polygon([(x1 - 3, 88), (x1 + 18, 102), (x1 - 3, 116)], fill=PALETTE["muted"])
    return card_from_content(canvas, title="1×1 Conv → ReLU → GAP", accent=PALETTE["panel_edge"], subtitle="actual fusion map preview + learnable VOP head")


def build_backbone_box(query_tile: Image.Image, gallery_tile: Image.Image, fq_tile: Image.Image, fg_tile: Image.Image) -> Image.Image:
    thumbs = [fit_inside(img, (150, 150)) for img in (query_tile, gallery_tile)]
    feats = [fit_inside(img, (150, 150)) for img in (fq_tile, fg_tile)]
    canvas = Image.new("RGBA", (760, 320), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((2, 2, 758, 318), radius=28, fill="white", outline=PALETTE["panel_edge"], width=3)
    font = load_font(26, bold=True)
    sub_font = load_font(20, bold=False)
    draw.text((24, 18), "Frozen ViT-Base Backbone", font=font, fill=PALETTE["title"])
    draw.text((24, 54), "real UAV / satellite pair -> real retrieval feature maps", font=sub_font, fill=PALETTE["subtitle"])
    xs = [26, 196, 426, 596]
    for x, tile in zip(xs, thumbs + feats):
        draw.rounded_rectangle((x, 106, x + 138, 244), radius=22, fill="white", outline=PALETTE["panel_edge"], width=2)
        canvas.alpha_composite(tile, (x + (138 - tile.width) // 2, 106 + (138 - tile.height) // 2))
    labels = ["q", "g", "F_q", "F_g"]
    for x, label in zip(xs, labels):
        badge = draw_badge(label, fill=PALETTE["rotate"] if label in {"q", "g"} else PALETTE["useful"], text_fill="white")
        canvas.alpha_composite(badge, (x + 16, 252))
    for x0, x1 in ((166, 196), (366, 426), (566, 596)):
        draw.line((x0, 174, x1, 174), fill=PALETTE["muted"], width=5)
        draw.polygon([(x1 - 4, 160), (x1 + 16, 174), (x1 - 4, 188)], fill=PALETTE["muted"])
    return crop_to_alpha(canvas, padding=6)


def build_vop_outer_box(fg_tile: Image.Image, fqrot_tile: Image.Image, prod_tile: Image.Image, diff_tile: Image.Image, posterior_tile: Image.Image) -> Image.Image:
    tiles = [fit_inside(img, (128, 128)) for img in (fg_tile, fqrot_tile, prod_tile, diff_tile)]
    posterior_tile = fit_inside(posterior_tile, (200, 200))
    canvas = Image.new("RGBA", (1060, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((2, 2, 1058, 358), radius=30, fill="white", outline=PALETTE["ours"], width=4)
    title_font = load_font(28, bold=True)
    sub_font = load_font(20, bold=False)
    draw.text((24, 18), "Learnable VOP Module", font=title_font, fill=PALETTE["title"])
    draw.text((24, 54), "real projected features, real pairwise fusion, real posterior", font=sub_font, fill=PALETTE["subtitle"])
    positions = [28, 176, 324, 472]
    labels = ["[F_g]", "[F_{q,rot}]", "[F_g ⊙\nF_{q,rot}]", "[|F_g -\nF_{q,rot}|]"]
    for x, tile, label in zip(positions, tiles, labels):
        draw.rounded_rectangle((x, 118, x + 128, 246), radius=20, fill="white", outline=PALETTE["panel_edge"], width=2)
        canvas.alpha_composite(tile, (x + (128 - tile.width) // 2, 118 + (128 - tile.height) // 2))
        font = load_font(16, bold=True)
        lines = label.split("\n")
        text_y = 254
        for line in lines:
            tw, th = text_size(draw, line, font)
            draw.text((x + (128 - tw) / 2, text_y), line, font=font, fill=PALETTE["title"])
            text_y += th + 2
    head = build_head_box_module(prod_tile)
    head_small = fit_inside(head, (260, 220))
    canvas.alpha_composite(head_small, (642, 106))
    canvas.alpha_composite(posterior_tile, (838, 92))
    badge = draw_badge("posterior", fill=PALETTE["ours"])
    canvas.alpha_composite(badge, (864, 294))
    for x0, x1 in ((606, 642), (900, 838)):
        draw.line((x0, 184, x1, 184), fill=PALETTE["muted"], width=5)
        if x1 > x0:
            draw.polygon([(x1 - 4, 170), (x1 + 16, 184), (x1 - 4, 198)], fill=PALETTE["muted"])
        else:
            draw.polygon([(x1 + 4, 170), (x1 - 16, 184), (x1 + 4, 198)], fill=PALETTE["muted"])
    return crop_to_alpha(canvas, padding=6)


def build_verification_box(match_module: Image.Image) -> Image.Image:
    thumb = fit_inside(match_module, (600, 250))
    canvas = Image.new("RGBA", (690, 360), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((2, 2, 688, 358), radius=28, fill="white", outline=PALETTE["green"], width=4)
    title_font = load_font(28, bold=True)
    sub_font = load_font(20, bold=False)
    draw.text((24, 18), "SuperPoint + LightGlue → Geometric Verification", font=title_font, fill=PALETTE["title"])
    draw.text((24, 54), "real sparse matches after a selected rotation hypothesis", font=sub_font, fill=PALETTE["subtitle"])
    canvas.alpha_composite(thumb, ((canvas.width - thumb.width) // 2, 96))
    return crop_to_alpha(canvas, padding=6)


def build_final_result_module(pair_summary: Mapping[str, object]) -> Tuple[Image.Image, Image.Image]:
    gallery_name = pair_summary["gallery_name"]
    gallery_path = DATA_ROOT / "satellite" / gallery_name
    gallery_rgb = np.asarray(Image.open(gallery_path).convert("RGB"))
    projected = pair_summary.get("geometry_selected_projected_xy")
    if projected is not None:
        projected = (float(projected[0]), float(projected[1]))
    result_rgb = draw_projected_point(gallery_rgb, projected, "")
    image = Image.fromarray(result_rgb).convert("RGBA")
    image = ImageOps.fit(image, (420, 420), method=Image.Resampling.LANCZOS)
    basemap = image_tile(image, title="Final Localization Basemap", accent=PALETTE["ours"], tag="g")
    footer = f"angle = {pair_summary['geometry_selected_angle_deg']:+.0f}°"
    result = image_tile(image, title="Final Localization Result", accent=PALETTE["ours"], tag="result", footer=footer)
    return basemap, result


def module_stem(name: str) -> str:
    return name.lower().replace(" ", "_").replace("+", "plus").replace("×", "x").replace("→", "to").replace("/", "_").replace(",", "")


def save_module(
    image: Image.Image,
    *,
    category: str,
    stem: str,
    description: str,
    source_file: str,
    derived_from_real_image: bool,
    recommended_usage: str,
    manifest_entries: List[ModuleEntry],
) -> None:
    image = crop_to_alpha(image, padding=6)
    out_path = PNG_ROOT / category / f"{stem}.png"
    image.save(out_path)
    preview = make_preview_version(image)
    preview.save(PREVIEW_MODULE_ROOT / category / f"{stem}_preview.png")
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


def contact_sheet(paths: Sequence[Path], *, title: str, columns: int = 4) -> Image.Image:
    tiles = [load_rgba(path) for path in paths]
    tile_previews = [make_preview_version(tile) for tile in tiles]
    thumb_size = (420, 340)
    captions = [path.stem for path in paths]
    title_font = load_font(34, bold=True)
    cap_font = load_font(18, bold=False)
    rows = int(math.ceil(len(tile_previews) / float(columns)))
    cell_w, cell_h = thumb_size[0] + 36, thumb_size[1] + 56
    width = columns * cell_w + 40
    height = 90 + rows * cell_h + 30
    sheet = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 18), title, font=title_font, fill=PALETTE["title"])
    for idx, (tile, caption) in enumerate(zip(tile_previews, captions)):
        row, col = divmod(idx, columns)
        x = 20 + col * cell_w
        y = 74 + row * cell_h
        thumb = fit_inside(tile, thumb_size)
        sheet.alpha_composite(thumb, (x + (thumb_size[0] - thumb.width) // 2, y))
        draw.text((x + 4, y + thumb_size[1] + 10), caption, font=cap_font, fill=PALETTE["subtitle"])
    return sheet


def pipeline_preview(module_paths: Mapping[str, Path]) -> Image.Image:
    order = [
        module_paths["input_pair_q_g"],
        module_paths["frozen_vit_base_backbone_box"],
        module_paths["learnable_vop_module_box"],
        module_paths["topk_selected_angles"],
        module_paths["superpoint_lightglue_geometric_verification_box"],
        module_paths["final_localization_result"],
    ]
    images = [fit_inside(load_rgba(path), (340, 300)) for path in order]
    canvas = Image.new("RGBA", (2160, 520), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    draw.text((24, 18), "Assembly Preview: Core Pipeline Modules", font=title_font, fill=PALETTE["title"])
    x = 24
    y = 120
    arrow = draw_arrow_icon((92, 56))
    for idx, img in enumerate(images):
        canvas.alpha_composite(img, (x, y + (300 - img.height) // 2))
        x += 340
        if idx != len(images) - 1:
            canvas.alpha_composite(arrow, (x + 10, y + 120))
            x += 110
    return canvas


def feature_preview(module_paths: Mapping[str, Path]) -> Image.Image:
    order = [
        module_paths["feature_map_f_q"],
        module_paths["feature_map_f_g"],
        module_paths["feature_map_f_q_rot"],
        module_paths["fusion_block_f_g"],
        module_paths["fusion_block_f_q_rot"],
        module_paths["fusion_block_f_g_mul_f_q_rot"],
        module_paths["fusion_block_abs_f_g_minus_f_q_rot"],
        module_paths["feature_concatenation_fusion_symbol"],
        module_paths["conv_relu_gap_head_box"],
        module_paths["posterior_probability_distribution"],
    ]
    paths = [module_paths[key] for key in [
        "feature_map_f_q",
        "feature_map_f_g",
        "feature_map_f_q_rot",
        "fusion_block_f_g",
        "fusion_block_f_q_rot",
        "fusion_block_f_g_mul_f_q_rot",
        "fusion_block_abs_f_g_minus_f_q_rot",
        "feature_concatenation_fusion_symbol",
        "conv_relu_gap_head_box",
        "posterior_probability_distribution",
    ]]
    return contact_sheet(paths, title="Assembly Preview: Feature / Posterior Modules", columns=5)


def results_preview(module_paths: Mapping[str, Path]) -> Image.Image:
    paths = [module_paths[key] for key in [
        "accuracy_efficiency_tradeoff_plot",
        "localization_error_curve",
        "soft_orientation_target",
        "useful_angle_set_module",
        "pair_confidence_weight_curve",
        "sparse_no_vop_match_visual",
        "ours_vop_guided_match_visual",
    ]]
    return contact_sheet(paths, title="Assembly Preview: Plots / Supervision / Verification", columns=3)


def write_manifest(entries: Sequence[ModuleEntry], missing_items: Sequence[str]) -> None:
    manifest = [entry.__dict__ for entry in entries]
    (MANIFEST_ROOT / "manifest.json").write_text(json.dumps({"modules": manifest, "missing_inputs": list(missing_items)}, indent=2, ensure_ascii=False), encoding="utf-8")


def write_readme(entries: Sequence[ModuleEntry], missing_items: Sequence[str]) -> None:
    total = len(entries)
    real = [entry for entry in entries if entry.derived_from_real_image]
    redraw = [entry for entry in entries if not entry.derived_from_real_image]
    lines = [
        "# PNG Module Library",
        "",
        f"- Total exported modules: {total}",
        f"- Modules derived from local real imagery / real experiment figures: {len(real)}",
        f"- Vectorized / redrawn modules: {len(redraw)}",
        "",
        "## Exported Modules",
        "",
    ]
    for entry in entries:
        lines.append(f"- `{Path(entry.file_name).name}`")
        lines.append(f"  meaning: {entry.description}")
        lines.append(f"  category: {entry.category}")
        lines.append(f"  source: {entry.source_file}")
        lines.append(f"  derived_from_real_image: {str(entry.derived_from_real_image).lower()}")
        lines.append(f"  recommended_usage: {entry.recommended_usage}")
    lines.extend([
        "",
        "## Real-source Modules",
        "",
    ])
    for entry in real:
        lines.append(f"- `{Path(entry.file_name).name}` <- {entry.source_file}")
    lines.extend([
        "",
        "## Redrawn / Vectorized Modules",
        "",
    ])
    for entry in redraw:
        lines.append(f"- `{Path(entry.file_name).name}` <- {entry.source_file}")
    lines.extend([
        "",
        "## Missing Inputs / Notes",
        "",
    ])
    if missing_items:
        for item in missing_items:
            lines.append(f"- {item}")
    else:
        lines.append("- No critical source files were missing. No PPT/PPTX source deck was found; SVG drafts and figure PNGs were used instead.")
        lines.append("- No local drone icon asset was found, so a vectorized UAV top-view icon was generated to match the paper style.")
    (MANIFEST_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_dirs()

    required_sources = [
        VISLOC_CKPT,
        VOP_CKPT,
        PAIR_ROOT / "pair01_vop_summary.json",
        TEACHER_ROOT / "teacher_pair_summary.json",
        SHORTPAPER_ROOT / "fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png",
        REVIEW_ROOT / "fig04a_angle_surface_mechanism.png",
    ]
    missing_items = [str(path) for path in required_sources if not path.is_file()]
    if missing_items:
        print("Missing required inputs:")
        for item in missing_items:
            print(item)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_entries: List[ModuleEntry] = []
    module_paths: Dict[str, Path] = {}

    main_bundle = prepare_main_feature_bundle(device=device)
    main_sample = main_bundle["sample"]
    query_image = main_bundle["query_image"]
    gallery_image = main_bundle["gallery_image"]

    selected_angles = [float(angle) for angle in main_sample["eval_record"]["selected_angles_deg"]]
    sparse_visuals = run_sparse_visuals(
        query_path=main_bundle["query_path"],
        gallery_path=main_bundle["gallery_path"],
        best_angle=float(main_bundle["best_angle"]),
        device=device,
    )

    pair_summary = json.loads((PAIR_ROOT / "pair01_vop_summary.json").read_text(encoding="utf-8"))
    basemap_module, final_result_module = build_final_result_module(pair_summary)

    query_tile = image_tile(query_image, title="UAV Input Tile", accent=PALETTE["rotate"], tag="q")
    gallery_tile = image_tile(gallery_image, title="Satellite Input Tile", accent=PALETTE["dense"], tag="g")

    pair_canvas = Image.new("RGBA", (780, 390), (0, 0, 0, 0))
    pair_canvas.alpha_composite(fit_inside(query_tile, (360, 340)), (10, 20))
    pair_canvas.alpha_composite(fit_inside(gallery_tile, (360, 340)), (410, 20))
    input_pair_module = card_from_content(pair_canvas, title="Input Pair (q, g)", accent=PALETTE["panel_edge"], subtitle="real UAV / satellite pair")

    fq_tile = feature_tile(main_bundle["raw_query_map"], label="F_q", accent=PALETTE["rotate"])
    fg_tile = feature_tile(main_bundle["raw_gallery_map"], label="F_g", accent=PALETTE["dense"])
    fqrot_tile = feature_tile(main_bundle["enc_query_rot"], label="F_{q,rot}", accent=PALETTE["teacher"])
    fg_enc_tile = feature_tile(main_bundle["enc_gallery_map"], label="[F_g]", accent=PALETTE["rotate"])
    fqrot_enc_tile = feature_tile(main_bundle["enc_query_rot"], label="[F_{q,rot}]", accent=PALETTE["teacher"])
    prod_tile = feature_tile(main_bundle["enc_gallery_map"] * main_bundle["enc_query_rot"], label="[F_g ⊙ F_{q,rot}]", accent=PALETTE["clean30"])
    diff_tile = feature_tile(torch.abs(main_bundle["enc_gallery_map"] - main_bundle["enc_query_rot"]), label="[|F_g - F_{q,rot}|]", accent=PALETTE["useful"])

    posterior_module = build_posterior_plot(main_sample)
    topk_module = build_topk_angles_module(selected_angles)
    rotation_module = build_rotation_module(query_image)
    concat_module = build_concat_symbol_module()
    head_module = build_head_box_module(prod_tile)
    backbone_box = build_backbone_box(query_tile, gallery_tile, fq_tile, fg_tile)
    vop_box = build_vop_outer_box(fg_enc_tile, fqrot_enc_tile, prod_tile, diff_tile, posterior_module)

    sparse_match_module = image_tile(sparse_visuals["sparse"], title="Sparse (no VOP) Match Visual", accent=PALETTE["sparse"], tag="0°")
    ours_match_module = image_tile(sparse_visuals["ours"], title="Ours (VOP-guided) Match Visual", accent=PALETTE["ours"], tag=f"{main_bundle['best_angle']:+.0f}°")

    compare_canvas = Image.new("RGBA", (1520, 420), (0, 0, 0, 0))
    compare_canvas.alpha_composite(fit_inside(sparse_match_module, (730, 360)), (0, 30))
    compare_canvas.alpha_composite(fit_inside(ours_match_module, (730, 360)), (790, 30))
    verification_compare = card_from_content(compare_canvas, title="Sparse Matching vs VOP-guided Matching", accent=PALETTE["green"])
    verification_box = build_verification_box(ours_match_module)

    tradeoff_module = build_tradeoff_module()
    error_curve_module = build_error_curve_module()
    soft_target_module = build_soft_target_module()
    useful_module = build_useful_angle_panel_module()
    pair_conf_module = build_pair_confidence_curve_module()

    save_specs = [
        ("input", "uav_input_tile", query_tile, "Real UAV query tile used in the main VOP pipeline example.", str(main_bundle["query_path"]), True, "Drag into the leftmost input position in the paper figure."),
        ("input", "satellite_input_tile", gallery_tile, "Real retrieved satellite tile used in the main VOP pipeline example.", str(main_bundle["gallery_path"]), True, "Drag into the gallery/top-1 input position."),
        ("input", "input_pair_q_g", input_pair_module, "Combined real input pair module with q/g labels.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']}", True, "Use as a compact q-g pair block in PPT/Illustrator."),
        ("input", "final_localization_basemap", basemap_module, "Real satellite basemap with the final localization pin position.", str(DATA_ROOT / 'satellite' / pair_summary['gallery_name']), True, "Use as the base map before adding extra result annotations."),
        ("backbone", "frozen_vit_base_backbone_box", backbone_box, "Frozen ViT-Base backbone box with real input thumbnails and real feature maps.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']} | {VISLOC_CKPT}", True, "Use as the backbone stage in the method figure."),
        ("backbone", "learnable_vop_module_box", vop_box, "Learnable VOP module outer box with real pairwise feature maps and real posterior.", f"{VISLOC_CKPT} | {VOP_CKPT} | {MAIN_SAMPLE['cache_path']}", True, "Use as the central VOP block in the method figure."),
        ("verification", "superpoint_lightglue_geometric_verification_box", verification_box, "Sparse verification box with real SP+LG match visual.", str(main_bundle["gallery_path"]), True, "Use as the downstream verification stage."),
        ("features", "feature_map_f_q", fq_tile, "Real backbone feature map F_q visualized by PCA projection.", f"{main_bundle['query_path']} | {VISLOC_CKPT}", True, "Use as an individual feature tile or within the backbone box."),
        ("features", "feature_map_f_g", fg_tile, "Real backbone feature map F_g visualized by PCA projection.", f"{main_bundle['gallery_path']} | {VISLOC_CKPT}", True, "Use as an individual feature tile or within the backbone box."),
        ("features", "feature_map_f_q_rot", fqrot_tile, "Real rotated query feature map F_{q,rot} after VOP rotation.", f"{main_bundle['query_path']} | {VOP_CKPT}", True, "Use before the pairwise fusion stage."),
        ("features", "fusion_block_f_g", fg_enc_tile, "Real encoded feature tile [F_g] used inside the VOP pairwise head.", f"{main_bundle['gallery_path']} | {VOP_CKPT}", True, "Use as one of the four pairwise fusion inputs."),
        ("features", "fusion_block_f_q_rot", fqrot_enc_tile, "Real encoded feature tile [F_{q,rot}] used inside the VOP pairwise head.", f"{main_bundle['query_path']} | {VOP_CKPT}", True, "Use as one of the four pairwise fusion inputs."),
        ("features", "fusion_block_f_g_mul_f_q_rot", prod_tile, "Real multiplicative fusion tile [F_g ⊙ F_{q,rot}].", f"{main_bundle['query_path']} | {main_bundle['gallery_path']} | {VOP_CKPT}", True, "Use as one of the four pairwise fusion inputs."),
        ("features", "fusion_block_abs_f_g_minus_f_q_rot", diff_tile, "Real absolute-difference fusion tile [|F_g - F_{q,rot}|].", f"{main_bundle['query_path']} | {main_bundle['gallery_path']} | {VOP_CKPT}", True, "Use as one of the four pairwise fusion inputs."),
        ("features", "feature_concatenation_fusion_symbol", concat_module, "Reusable concat / fusion symbol module for the four VOP feature inputs.", "redrawn vector icon matched to the paper palette", False, "Place between the four feature tiles and the prediction head."),
        ("features", "conv_relu_gap_head_box", head_module, "Prediction head module showing 1×1 Conv → ReLU → GAP.", f"real fusion preview from {VOP_CKPT}", True, "Use as the VOP prediction head."),
        ("features", "candidate_angles_rotation_module", rotation_module, "Rotation operation block based on the real UAV query tile.", str(main_bundle["query_path"]), True, "Use to explain 36 candidate angles and feature rotation."),
        ("posterior", "posterior_probability_distribution", posterior_module, "Real posterior distribution on the main sample with highlighted top-4 angles.", f"{MAIN_SAMPLE['cache_path']} | {MAIN_SAMPLE['eval_path']}", True, "Use as the posterior module in the pipeline figure."),
        ("posterior", "topk_selected_angles", topk_module, "Real top-k selected angle badges from the main sample.", f"{MAIN_SAMPLE['eval_path']}", False, "Use after the posterior module to denote selected hypotheses."),
        ("posterior", "soft_orientation_target", soft_target_module, "Soft orientation target from the real teacher example.", str(TEACHER_ROOT / "teacher_pair_soft_orientation_target_v2_smooth.png"), True, "Use in training supervision figures."),
        ("posterior", "useful_angle_set_module", useful_module, "Useful-angle set module cropped from the real mechanism figure.", str(REVIEW_ROOT / "fig04a_angle_surface_mechanism.png"), True, "Use to explain multimodal useful-angle structure."),
        ("posterior", "pair_confidence_weight_curve", pair_conf_module, "Pair-confidence weighting curve derived from the real training recipe.", str(GAME4LOC_DIR / "train_vop.py"), False, "Use alongside useful-angle supervision in the training section."),
        ("verification", "sparse_no_vop_match_visual", sparse_match_module, "Real SP+LG match visualization without VOP guidance on the main sample.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']}", True, "Use as the plain sparse baseline visual."),
        ("verification", "ours_vop_guided_match_visual", ours_match_module, "Real VOP-guided SP+LG match visualization on the main sample.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']} | best angle {main_bundle['best_angle']:+.0f}", True, "Use as the VOP-guided verification visual."),
        ("verification", "match_visual_comparison_dual", verification_compare, "Side-by-side comparison between no-VOP sparse matching and VOP-guided matching.", f"{main_bundle['query_path']} | {main_bundle['gallery_path']}", True, "Use when a compact comparison panel is needed."),
        ("results", "final_localization_result", final_result_module, "Real final localization result with the actual projected pin on the satellite tile.", str(PAIR_ROOT / "pair01_vop_summary.json"), True, "Use as the final result panel."),
        ("results", "accuracy_efficiency_tradeoff_plot", tradeoff_module, "Accuracy-efficiency tradeoff plot from the current paper figure set.", str(SHORTPAPER_ROOT / "fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png"), True, "Use as the standalone quantitative comparison plot."),
        ("results", "localization_error_curve", error_curve_module, "Real teacher localization error curve on the Paper7 pair.", str(TEACHER_ROOT / "teacher_pair_localization_error_curve.png"), True, "Use as a supervision / motivation plot."),
        ("icons", "uav_topview_icon", draw_drone_icon(), "Reusable vectorized UAV top-view icon.", "redrawn vector icon because no local UAV icon asset exists", False, "Use as a small visual cue in diagrams or legends."),
        ("icons", "red_pin_icon", draw_pin_icon(), "Reusable red localization pin icon.", "redrawn vector icon matched to the paper palette", False, "Use on top of localization maps or result panels."),
        ("icons", "arrow_right_icon", draw_arrow_icon(), "Reusable straight arrow icon.", "redrawn vector icon matched to the paper palette", False, "Use between adjacent modules in PowerPoint or Illustrator."),
        ("icons", "rotation_arrow_icon", draw_rotation_icon(), "Reusable rotation / 36-angle icon.", "redrawn vector icon matched to the paper palette", False, "Use near rotation or candidate-angle annotations."),
        ("icons", "topk_badge", draw_badge("top-k", fill=PALETTE["ours"]), "Reusable top-k badge.", "redrawn vector badge", False, "Use to annotate selected hypotheses."),
        ("icons", "softmax_badge", draw_badge("softmax", fill=PALETTE["rotate"]), "Reusable softmax badge.", "redrawn vector badge", False, "Use between the head and the posterior distribution."),
    ]

    for category, stem, image, description, source_file, derived, usage in save_specs:
        save_module(
            image,
            category=category,
            stem=stem,
            description=description,
            source_file=source_file,
            derived_from_real_image=derived,
            recommended_usage=usage,
            manifest_entries=manifest_entries,
        )
        module_paths[stem] = PNG_ROOT / category / f"{stem}.png"

    preview_images = {
        "assembly_preview_pipeline.png": pipeline_preview(module_paths),
        "assembly_preview_features.png": feature_preview(module_paths),
        "assembly_preview_results.png": results_preview(module_paths),
    }
    for name, image in preview_images.items():
        image.save(PREVIEW_ROOT / name)

    write_manifest(manifest_entries, missing_items)
    write_readme(manifest_entries, missing_items)

    print("Exported module library:")
    for entry in manifest_entries:
        print(entry.file_name)
    print("Preview files:")
    for name in preview_images:
        print(PREVIEW_ROOT / name)
    print("Manifest:")
    print(MANIFEST_ROOT / "manifest.json")
    print(MANIFEST_ROOT / "README.md")


if __name__ == "__main__":
    main()
