#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
DATA_PATH = REPORT_DIR / "advisor_update_data.json"


PALETTE = {
    "bg": "#F8FAFC",
    "panel": "#FFFFFF",
    "grid": "#E2E8F0",
    "axis": "#94A3B8",
    "text": "#0F172A",
    "muted": "#475569",
    "old": "#2563EB",
    "expanded": "#EA580C",
    "retrieval": "#64748B",
    "rotate": "#0F766E",
    "vop": "#B45309",
    "vop2": "#7C3AED",
    "oracle": "#16A34A",
    "random": "#DC2626",
    "highlight": "#FACC15",
    "good": "#15803D",
    "bad": "#B91C1C",
}


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_summary_txt(path: Path) -> Dict[str, Dict[str, float]]:
    content = path.read_text(encoding="utf-8")
    blocks: Dict[str, Dict[str, float]] = {}
    current = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = re.match(r"^\[(.+)\]$", line)
        if match:
            current = match.group(1)
            blocks.setdefault(current, {})
            continue
        if not current or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"Recall@1", "Recall@5", "Recall@10", "AP", "Best_Recall@1"}:
            try:
                blocks[current][key] = float(value)
            except ValueError:
                pass
    return {name: values for name, values in blocks.items() if values}


def parse_retrieval_eval_log(path: Path) -> List[Dict[str, float]]:
    pattern = re.compile(
        r"Recall@1:\s*([0-9.]+)\s*-\s*Recall@5:\s*([0-9.]+)\s*-\s*Recall@10:\s*([0-9.]+)\s*-\s*Recall@top1:\s*([0-9.]+)\s*-\s*AP:\s*([0-9.]+)\s*-\s*SDM@1:\s*([0-9.]+)\s*-\s*SDM@3:\s*([0-9.]+)\s*-\s*SDM@5:\s*([0-9.]+)\s*-\s*Dis@1:\s*([0-9.]+)\s*-\s*Dis@3:\s*([0-9.]+)\s*-\s*Dis@5:\s*([0-9.]+)"
    )
    records: List[Dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values = [float(item) for item in match.groups()]
        records.append(
            {
                "Recall@1": values[0],
                "Recall@5": values[1],
                "Recall@10": values[2],
                "Recall@top1": values[3],
                "AP": values[4],
                "SDM@1_pct": values[5] * 100.0,
                "SDM@3_pct": values[6] * 100.0,
                "SDM@5_pct": values[7] * 100.0,
                "Dis@1_m": values[8],
                "Dis@3_m": values[9],
                "Dis@5_m": values[10],
            }
        )
    return records


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((float(v) - mu) ** 2 for v in values) / (len(values) - 1))


def scene_mean(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    by_scene: Dict[str, List[float]] = {}
    for record in records:
        scene = str(record["query_name"])[:2]
        by_scene.setdefault(scene, []).append(float(record["final_error_m"]))
    return {scene: mean(values) for scene, values in sorted(by_scene.items())}


def scene_counts_from_query_names(names: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name in names:
        scene = str(name)[:2]
        counts[scene] = counts.get(scene, 0) + 1
    return counts


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str = "") -> None:
        self.width = int(width)
        self.height = int(height)
        self.parts: List[str] = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{self.width}" height="{self.height}" '
                f'viewBox="0 0 {self.width} {self.height}" '
                f'fill="none" role="img" aria-label="{esc(title)}">'
            ),
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">',
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["axis"]}"/>',
            "</marker>",
            "</defs>",
            (
                f'<rect x="0" y="0" width="{self.width}" height="{self.height}" '
                f'fill="{PALETTE["bg"]}"/>'
            ),
            (
                '<style>'
                'text{font-family:"Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei","PingFang SC","Hiragino Sans GB","DejaVu Sans","Arial",sans-serif;}'
                "</style>"
            ),
        ]

    def add(self, chunk: str) -> None:
        self.parts.append(chunk)

    def panel(self, x: float, y: float, w: float, h: float, radius: float = 18.0) -> None:
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{PALETTE["panel"]}" stroke="{PALETTE["grid"]}" stroke-width="1.5"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        label: object,
        *,
        size: int = 18,
        fill: str | None = None,
        anchor: str = "start",
        weight: str = "400",
    ) -> None:
        color = fill or PALETTE["text"]
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{esc(label)}</text>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str | None = None,
        width: float = 1.0,
        dash: str | None = None,
        marker_end: bool = False,
    ) -> None:
        attrs = [
            f'x1="{x1}"',
            f'y1="{y1}"',
            f'x2="{x2}"',
            f'y2="{y2}"',
            f'stroke="{stroke or PALETTE["axis"]}"',
            f'stroke-width="{width}"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker_end:
            attrs.append('marker-end="url(#arrow)"')
        self.add(f"<line {' '.join(attrs)}/>")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        radius: float = 8.0,
        stroke: str | None = None,
    ) -> None:
        extra = ""
        if stroke:
            extra = f' stroke="{stroke}" stroke-width="1.2"'
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"{extra}/>'
        )

    def circle(self, cx: float, cy: float, r: float, *, fill: str, stroke: str | None = None) -> None:
        extra = ""
        if stroke:
            extra = f' stroke="{stroke}" stroke-width="1.2"'
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{extra}/>')

    def close(self) -> str:
        return "\n".join(self.parts + ["</svg>"])


def write_svg(path: Path, svg: SVG) -> None:
    path.write_text(svg.close(), encoding="utf-8")


def draw_panel_title(svg: SVG, x: float, y: float, title: str, subtitle: str) -> None:
    svg.text(x, y, title, size=28, weight="700")
    svg.text(x, y + 28, subtitle, size=15, fill=PALETTE["muted"])


def draw_horizontal_scale(
    svg: SVG,
    x: float,
    y: float,
    width: float,
    max_value: float,
    ticks: Sequence[float],
    label: str,
) -> None:
    svg.line(x, y, x + width, y, stroke=PALETTE["axis"], width=1.4)
    for tick in ticks:
        px = x + width * float(tick) / max_value
        svg.line(px, y - 6, px, y + 6, stroke=PALETTE["axis"], width=1.0)
        svg.line(px, y - 220, px, y, stroke=PALETTE["grid"], width=1.0, dash="4 6")
        svg.text(px, y + 24, f"{tick:g}", size=12, fill=PALETTE["muted"], anchor="middle")
    svg.text(x + width, y + 50, label, size=13, fill=PALETTE["muted"], anchor="end")


def chart_problem_background(output_path: Path) -> None:
    svg = SVG(1420, 700, "课题背景与切入点")
    svg.panel(20, 20, 1380, 660)
    draw_panel_title(
        svg,
        48,
        62,
        "课题背景：为什么检索后细定位仍然困难",
        "这是一个两阶段任务：检索先缩小搜索范围，细定位再在候选卫星图上恢复更准确的位置。",
    )
    panels = [
        (50, 130, 400, 520, "任务位置"),
        (500, 130, 400, 520, "当前主要瓶颈"),
        (950, 130, 400, 520, "我们的切入思路"),
    ]
    for x, y, w, h, title in panels:
        svg.rect(x, y, w, h, fill=PALETTE["panel"], stroke=PALETTE["grid"], radius=20)
        svg.text(x + 24, y + 38, title, size=24, weight="700")

    left_boxes = [
        (82, 200, 336, 76, "1 输入无人机查询图像", "#EFF6FF", "#93C5FD"),
        (82, 304, 336, 76, "2 检索模型在卫星图库中排序", "#EFF6FF", "#93C5FD"),
        (82, 408, 336, 76, "3 得到 top-K 候选，当前细定位主要接收 top-1", "#EFF6FF", "#93C5FD"),
        (82, 512, 336, 76, "4 细定位负责从候选图中恢复更准的位置", "#EFF6FF", "#93C5FD"),
    ]
    for x, y, w, h, label, fill, stroke in left_boxes:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, radius=16)
        svg.text(x + 20, y + 44, label, size=18, weight="600")
    for arrow in [(250, 276, 250, 304), (250, 380, 250, 408), (250, 484, 250, 512)]:
        svg.line(*arrow, width=2.0, marker_end=True)

    mid_boxes = [
        (532, 200, 336, 92, "方向歧义\n同一 query 在多个角度下都可能看起来部分合理", "#FEF3C7", "#FCD34D"),
        (532, 330, 336, 92, "对应点不稳定\n错误匹配会直接污染后续几何估计", "#FEF3C7", "#FCD34D"),
        (532, 460, 336, 92, "几何模型敏感\n角度一旦错了，RANSAC 与投影都容易连锁失败", "#FEF3C7", "#FCD34D"),
    ]
    for x, y, w, h, label, fill, stroke in mid_boxes:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, radius=16)
        for idx, line in enumerate(label.split("\n")):
            svg.text(x + 18, y + 34 + idx * 24, line, size=18 if idx == 0 else 16, weight="700" if idx == 0 else "500")

    right_boxes = [
        (982, 200, 336, 92, "不要只靠穷举试角度\n方向问题需要被显式建模", "#FFF7ED", "#FDBA74"),
        (982, 330, 336, 92, "在匹配前先预测方向后验\n让 VOP 给出少量高概率角度", "#FFF7ED", "#FDBA74"),
        (982, 460, 336, 92, "只让 top-k 角度进入匹配与几何验证\n减少无效搜索并提高候选质量", "#ECFDF5", "#86EFAC"),
    ]
    for x, y, w, h, label, fill, stroke in right_boxes:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, radius=16)
        for idx, line in enumerate(label.split("\n")):
            svg.text(x + 18, y + 34 + idx * 24, line, size=18 if idx == 0 else 16, weight="700" if idx == 0 else "500")

    svg.text(50, 670, "核心定位：VOP 插在检索与细定位之间，目标不是替代检索，而是把方向建模前置。", size=16, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_vop_method(output_path: Path) -> None:
    svg = SVG(1400, 620, "什么是 VOP")
    svg.panel(20, 20, 1360, 580)
    draw_panel_title(
        svg,
        48,
        62,
        "VOP 的训练与使用方式",
        "训练阶段生成角度教师分布；推理阶段只保留少量高概率角度进入匹配器。",
    )
    svg.text(60, 150, "训练阶段", size=24, weight="700", fill=PALETTE["old"])
    svg.text(60, 370, "推理阶段", size=24, weight="700", fill=PALETTE["expanded"])

    train_boxes = [
        (60, 170, 170, 82, "top-1 检索到的\n卫星图像", "#EFF6FF", "#93C5FD"),
        (280, 170, 170, 82, "无人机查询\n图像", "#EFF6FF", "#93C5FD"),
        (500, 170, 200, 82, "将查询图旋转为\n36 个角度（步长 10°）", "#FEF3C7", "#FCD34D"),
        (750, 170, 250, 82, "统计每个角度下的\n最终定位误差", "#FEF3C7", "#FCD34D"),
        (1050, 170, 260, 82, "把角度-误差曲线\n转成软教师分布", "#FFF7ED", "#FDBA74"),
    ]
    infer_boxes = [
        (60, 390, 170, 82, "top-1 检索到的\n卫星图像", "#EFF6FF", "#93C5FD"),
        (280, 390, 170, 82, "无人机查询\n图像", "#EFF6FF", "#93C5FD"),
        (500, 390, 200, 82, "来自检索骨干的\n特征图", "#ECFDF5", "#86EFAC"),
        (750, 390, 250, 82, "VOP 预测候选角度上的\n后验分布", "#FFF7ED", "#FDBA74"),
        (1050, 390, 260, 82, "保留 top-k 个角度，\n只在其上运行匹配器", "#ECFDF5", "#86EFAC"),
    ]
    for x, y, w, h, label, fill, stroke in train_boxes + infer_boxes:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, radius=16)
        for idx, line in enumerate(str(label).split("\n")):
            svg.text(x + w / 2, y + 32 + idx * 20, line, size=17, anchor="middle", weight="600")
    for arrow in [
        (230, 211, 280, 211),
        (450, 211, 500, 211),
        (700, 211, 750, 211),
        (1000, 211, 1050, 211),
        (230, 431, 280, 431),
        (450, 431, 500, 431),
        (700, 431, 750, 431),
        (1000, 431, 1050, 431),
    ]:
        svg.line(*arrow, width=2.2, marker_end=True)
    svg.text(1050, 285, "教师目标是软分布，不是独热标签", size=16, fill=PALETTE["muted"])
    svg.text(1050, 500, "这里用方向打分替代\n穷举式旋转搜索", size=16, fill=PALETTE["muted"])
    svg.text(
        60,
        548,
        "VOP 的作用：显式建模方向不确定性，并直接复用现有检索骨干作为特征编码器。",
        size=15,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_vop_teacher_stats(data: Dict[str, object], output_path: Path) -> None:
    useful = data["vop_old_compact"]["useful_structure"]
    diag = data["vop_useful_diag"]
    size_values = [float(useful[key]["size_mean"]) for key in ("1m", "3m", "5m")]
    multi_values = [float(useful[key]["multimodal_ratio"]) * 100.0 for key in ("1m", "3m", "5m")]
    entropy_bins = diag["teacher_stats"]["entropy_bins"]
    gap_counts = diag["teacher_stats"]["distance_gap_counts"]
    svg = SVG(1400, 700, "VOP 教师统计")
    svg.panel(20, 20, 1360, 660)
    draw_panel_title(
        svg,
        48,
        62,
        "VOP 学到的到底是什么方向目标",
        "教师分布通常带有歧义，而且经常是多区间的，因此用后验分布比单角度分类更合适。",
    )
    left_x = 60
    left_y = 200
    left_w = 580
    right_x = 730
    right_y = 200
    right_w = 600
    svg.text(left_x, 150, "原始 03/04 协议下的有效角结构", size=22, weight="700")
    svg.text(right_x, 150, "教师分布统计", size=22, weight="700")
    labels = ["1m", "3m", "5m"]
    for idx, label in enumerate(labels):
        x = left_x + 50 + idx * 170
        svg.text(x + 35, 512, label, size=14, anchor="middle", fill=PALETTE["muted"])
        size_h = 220 * size_values[idx] / 4.0
        multi_h = 220 * multi_values[idx] / 50.0
        svg.rect(x, 470 - size_h, 36, size_h, fill=PALETTE["vop2"])
        svg.rect(x + 46, 470 - multi_h, 36, multi_h, fill=PALETTE["vop"])
        svg.text(x + 18, 485 - size_h, f"{size_values[idx]:.2f}", size=12, anchor="middle", fill=PALETTE["vop2"], weight="600")
        svg.text(x + 64, 485 - multi_h, f"{multi_values[idx]:.1f}%", size=12, anchor="middle", fill=PALETTE["vop"], weight="600")
    svg.text(left_x + 30, 255, "平均有效角集合大小", size=13, fill=PALETTE["vop2"])
    svg.text(left_x + 30, 278, "多区间比例", size=13, fill=PALETTE["vop"])
    svg.line(left_x + 20, 470, left_x + 520, 470, stroke=PALETTE["axis"], width=1.3)
    svg.line(left_x + 20, 250, left_x + 20, 470, stroke=PALETTE["axis"], width=1.3)
    svg.text(left_x, 540, "在 5m 容差下，42.2% 的查询已经出现多区间有效角集合。", size=14, fill=PALETTE["muted"])

    entropy_labels = list(entropy_bins.keys())
    entropy_values = [int(entropy_bins[key]) for key in entropy_labels]
    max_entropy = max(entropy_values)
    for idx, (label, value) in enumerate(zip(entropy_labels, entropy_values)):
        x = right_x + idx * 120
        h = 180 * value / max_entropy
        svg.rect(x, 420 - h, 70, h, fill=PALETTE["old"])
        svg.text(x + 35, 440, label.replace("[", "").replace(")", "").replace("]", ""), size=11, anchor="middle", fill=PALETTE["muted"])
        svg.text(x + 35, 410 - h, value, size=12, anchor="middle", fill=PALETTE["old"], weight="600")
    svg.line(right_x - 10, 420, right_x + 5 * 120 + 80, 420, stroke=PALETTE["axis"], width=1.3)
    svg.text(right_x, 465, "教师熵分箱（502 个训练对）", size=14, fill=PALETTE["muted"])
    gap_items = [(">=1m", gap_counts["ge_1m"]), (">=3m", gap_counts["ge_3m"]), (">=5m", gap_counts["ge_5m"]), (">=10m", gap_counts["ge_10m"])]
    max_gap = max(v for _, v in gap_items)
    for idx, (label, value) in enumerate(gap_items):
        x = right_x + idx * 130
        h = 120 * value / max_gap
        svg.rect(x, 610 - h, 80, h, fill=PALETTE["rotate"])
        svg.text(x + 40, 630, label, size=12, anchor="middle", fill=PALETTE["muted"])
        svg.text(x + 40, 600 - h, value, size=12, anchor="middle", fill=PALETTE["rotate"], weight="600")
    svg.text(right_x, 662, "在 502 个训练对中，有 161 个样本的角度差距达到 >=5m，说明监督并不简单。", size=14, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_vop_k_curve(data: Dict[str, object], output_path: Path) -> None:
    compact = data["vop_old_compact"]
    ks = [1, 2, 3, 4, 5]
    dis_values = [float(compact[f"posterior_k{k}"]["Dis@1_m"]) for k in ks]
    time_values = [float(compact[f"posterior_k{k}"]["runtime_total"]) for k in ks]
    cov_values = [float(compact[f"posterior_k{k}"]["oracle_best_cov"]) * 100.0 for k in ks]
    svg = SVG(1360, 620, "VOP 的 k 值曲线")
    svg.panel(20, 20, 1320, 580)
    draw_panel_title(
        svg,
        48,
        62,
        "原始 03/04 协议：VOP 的 prior-topk 折中",
        "随着 k 增大，VOP 持续降低 Dis@1，但耗时也随之上升。这是最主要的实用折中。",
    )
    x0 = 110
    x1 = 1270
    y_bottom = 520
    left_margin = 140
    plot_w = 1080
    for y in [180, 320, 460]:
        svg.line(left_margin, y, left_margin + plot_w, y, stroke=PALETTE["grid"], width=1.0, dash="4 6")
    for i, k in enumerate(ks):
        x = left_margin + i * plot_w / (len(ks) - 1)
        svg.line(x, 150, x, y_bottom, stroke=PALETTE["grid"], width=1.0, dash="4 6")
        svg.text(x, 548, f"k={k}", size=13, anchor="middle", fill=PALETTE["muted"])
    svg.line(left_margin, y_bottom, left_margin + plot_w, y_bottom, stroke=PALETTE["axis"], width=1.4)
    svg.text(70, 155, "Dis@1（米）", size=14, fill=PALETTE["vop2"])
    svg.text(70, 180, "70", size=12, fill=PALETTE["muted"])
    svg.text(70, 320, "55", size=12, fill=PALETTE["muted"])
    svg.text(70, 460, "40", size=12, fill=PALETTE["muted"])
    dis_min, dis_max = 40.0, 70.0
    time_min, time_max = 0.10, 0.50
    cov_min, cov_max = 0.0, 25.0

    def map_y(value: float, vmin: float, vmax: float, top: float, bottom: float) -> float:
        ratio = (float(value) - vmin) / max(vmax - vmin, 1e-8)
        return bottom - ratio * (bottom - top)

    points_dis = []
    points_time = []
    points_cov = []
    for i, k in enumerate(ks):
        x = left_margin + i * plot_w / (len(ks) - 1)
        points_dis.append((x, map_y(dis_values[i], dis_min, dis_max, 160, 480)))
        points_time.append((x, map_y(time_values[i], time_min, time_max, 160, 480)))
        points_cov.append((x, map_y(cov_values[i], cov_min, cov_max, 160, 480)))
    svg.add(
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
            " ".join(f"{x},{y}" for x, y in points_dis),
            PALETTE["vop2"],
        )
    )
    svg.add(
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
            " ".join(f"{x},{y}" for x, y in points_time),
            PALETTE["vop"],
        )
    )
    svg.add(
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
            " ".join(f"{x},{y}" for x, y in points_cov),
            PALETTE["rotate"],
        )
    )
    for (x, y), value in zip(points_dis, dis_values):
        svg.circle(x, y, 6, fill=PALETTE["vop2"])
        svg.text(x, y - 12, f"{value:.1f}", size=12, anchor="middle", fill=PALETTE["vop2"], weight="600")
    for (x, y), value in zip(points_time, time_values):
        svg.circle(x, y, 6, fill=PALETTE["vop"])
        svg.text(x, y + 22, f"{value:.3f}s", size=12, anchor="middle", fill=PALETTE["vop"], weight="600")
    for (x, y), value in zip(points_cov, cov_values):
        svg.circle(x, y, 6, fill=PALETTE["rotate"])
        svg.text(x + 12, y - 6, f"{value:.1f}%", size=12, fill=PALETTE["rotate"], weight="600")
    legend = [
        ("Dis@1", PALETTE["vop2"]),
        ("单次耗时", PALETTE["vop"]),
        ("最优角覆盖率", PALETTE["rotate"]),
    ]
    lx = 880
    for idx, (label, color) in enumerate(legend):
        yy = 96 + idx * 24
        svg.rect(lx, yy - 12, 18, 18, fill=color, radius=4)
        svg.text(lx + 28, yy + 2, label, size=13, fill=PALETTE["muted"])
    svg.text(48, 590, "原始小范围结果表明：k=4/k=5 精度最好，但 k=3 已经是不错的速度-精度折中。", size=14, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_vop_controls_old(data: Dict[str, object], output_path: Path) -> None:
    compact = data["vop_old_compact"]
    random_dis = float(compact["random_k3"]["Dis@1_m"]["mean"])
    random_cov = float(compact["random_k3"]["oracle_best_cov"]["mean"]) * 100.0
    random_time = float(compact["random_k3"]["runtime_total"]["mean"])
    items = [
        ("后验选择 k=3", float(compact["posterior_k3"]["Dis@1_m"]), float(compact["posterior_k3"]["oracle_best_cov"]) * 100.0, float(compact["posterior_k3"]["runtime_total"]), PALETTE["vop2"]),
        ("均匀选择 k=3", float(compact["uniform_k3"]["Dis@1_m"]), float(compact["uniform_k3"]["oracle_best_cov"]) * 100.0, float(compact["uniform_k3"]["runtime_total"]), "#CA8A04"),
        ("随机选择 k=3", random_dis, random_cov, random_time, PALETTE["random"]),
        ("理想上界 k=3", float(compact["oracle_k3"]["Dis@1_m"]), float(compact["oracle_k3"]["oracle_best_cov"]) * 100.0, float(compact["oracle_k3"]["runtime_total"]), PALETTE["oracle"]),
    ]
    dis_max = 65.0
    cov_max = 100.0
    svg = SVG(1280, 620, "原始协议机制控制")
    svg.panel(24, 24, 1232, 572)
    draw_panel_title(
        svg,
        52,
        68,
        "原始 03/04 协议：k=3 的机制控制",
        "后验选择优于均匀/随机选择，而理想上界说明方向预测仍有提升空间。",
    )
    left_x = 250
    left_y = 535
    left_w = 420
    right_x = 770
    right_y = 535
    right_w = 420
    draw_horizontal_scale(svg, left_x, left_y, left_w, dis_max, [0, 20, 40, 60], "Dis@1（米）")
    draw_horizontal_scale(svg, right_x, right_y, right_w, cov_max, [0, 25, 50, 75, 100], "最优角覆盖率（%）")
    row_start = 170
    row_gap = 84
    bar_h = 24
    for idx, (label, dis1, cov, runtime, color) in enumerate(items):
        y = row_start + idx * row_gap
        svg.text(220, y + 18, label, size=18, anchor="end", weight="600")
        svg.rect(left_x, y, left_w * dis1 / dis_max, bar_h, fill=color)
        svg.text(left_x + left_w * dis1 / dis_max + 10, y + 18, f"{dis1:.1f}", size=14, fill=color, weight="600")
        svg.rect(right_x, y, right_w * cov / cov_max, bar_h, fill=color)
        svg.text(right_x + right_w * cov / cov_max + 10, y + 18, f"{cov:.1f}%", size=14, fill=color, weight="600")
        svg.text(220, y + 42, f"耗时={runtime:.3f}s", size=12, anchor="end", fill=PALETTE["muted"])
    svg.text(52, 578, "这是最直接的证据：在原始小范围设定下，VOP 学到的不是随机方向信号。", size=14, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_vop_scene_0304(data: Dict[str, object], output_path: Path) -> None:
    scene_data = data["vop_scene_0304"]
    counts = data["vop_scene_counts"]
    methods = ["posterior_k1", "posterior_k3", "posterior_k4", "posterior_k5", "oracle_k3"]
    labels = {
        "posterior_k1": "k=1",
        "posterior_k3": "k=3",
        "posterior_k4": "k=4",
        "posterior_k5": "k=5",
        "oracle_k3": "理想上界 k=3",
    }
    colors = {
        "posterior_k1": "#475569",
        "posterior_k3": PALETTE["vop"],
        "posterior_k4": PALETTE["vop2"],
        "posterior_k5": "#9333EA",
        "oracle_k3": PALETTE["oracle"],
    }
    svg = SVG(1360, 620, "场景 03 和 04")
    svg.panel(20, 20, 1320, 580)
    draw_panel_title(
        svg,
        48,
        62,
        "原始小范围设定：场景 03 与场景 04",
        f"场景 03 有 {counts['03']} 个查询，场景 04 有 {counts['04']} 个查询。",
    )
    panels = [("03", 70, 170, 560, 360), ("04", 720, 170, 560, 360)]
    vmax = 80.0
    for scene, x, y, w, h in panels:
        svg.rect(x, y, w, h, fill=PALETTE["panel"], stroke=PALETTE["grid"], radius=18)
        svg.text(x + 24, y + 34, f"场景 {scene}", size=22, weight="700")
        base_x = x + 140
        base_y = y + 300
        plot_w = 360
        draw_horizontal_scale(svg, base_x, base_y, plot_w, vmax, [0, 20, 40, 60, 80], "Dis@1（米）")
        for idx, method in enumerate(methods):
            yy = y + 70 + idx * 48
            value = float(scene_data[method][scene])
            svg.text(base_x - 18, yy + 15, labels[method], size=14, anchor="end", fill=PALETTE["muted"])
            svg.rect(base_x, yy, plot_w * value / vmax, 22, fill=colors[method])
            svg.text(base_x + plot_w * value / vmax + 10, yy + 16, f"{value:.1f}", size=13, fill=colors[method], weight="600")
    svg.text(48, 590, "结论：VOP 在场景 03 上尤其有效；场景 04 也有改善，但残余歧义更大。", size=14, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_vop_variant_progress(data: Dict[str, object], output_path: Path) -> None:
    current = data["vop_current_diag"]["eval_diagnostics"]
    useful = data["vop_useful_diag"]["eval_diagnostics"]
    items = [
        ("top-1 命中率", float(current["posterior_top1_hit_rate"]) * 100.0, float(useful["posterior_top1_hit_rate"]) * 100.0, 12.0, "%"),
        ("top-3 覆盖率", float(current["posterior_top3_coverage"]) * 100.0, float(useful["posterior_top3_coverage"]) * 100.0, 30.0, "%"),
        ("平均最终误差", float(current["final_error_m"]["mean"]), float(useful["final_error_m"]["mean"]), 70.0, "m"),
    ]
    svg = SVG(1280, 540, "监督版本对比")
    svg.panel(24, 24, 1232, 492)
    draw_panel_title(
        svg,
        52,
        68,
        "改成有效角监督后发生了什么",
        "原始小范围设定下：当前 VOP 与有效角 5m top-3 版 VOP 的对比。",
    )
    legend_y = 112
    svg.rect(52, legend_y, 18, 18, fill=PALETTE["old"], radius=4)
    svg.text(78, legend_y + 15, "当前 rank-CE 版 VOP", size=14, fill=PALETTE["muted"])
    svg.rect(290, legend_y, 18, 18, fill=PALETTE["vop2"], radius=4)
    svg.text(316, legend_y + 15, "有效角 5m top-3 版 VOP", size=14, fill=PALETTE["muted"])
    row_start = 170
    row_gap = 100
    bar_h = 24
    for idx, (label, old_v, new_v, vmax, suffix) in enumerate(items):
        y = row_start + idx * row_gap
        svg.text(250, y + 18, label, size=18, anchor="end", weight="600")
        base_x = 280
        plot_w = 860
        draw_horizontal_scale(svg, base_x, y + 58, plot_w, vmax, [0, vmax / 3, 2 * vmax / 3, vmax], "")
        svg.rect(base_x, y, plot_w * old_v / vmax, bar_h, fill=PALETTE["old"])
        svg.rect(base_x, y + 30, plot_w * new_v / vmax, bar_h, fill=PALETTE["vop2"])
        svg.text(base_x + plot_w * old_v / vmax + 10, y + 18, f"{old_v:.1f}{suffix}", size=14, fill=PALETTE["old"], weight="600")
        svg.text(base_x + plot_w * new_v / vmax + 10, y + 48, f"{new_v:.1f}{suffix}", size=14, fill=PALETTE["vop2"], weight="600")
    svg.text(52, 506, "这说明我们不只是加了一个模块，也在改进 VOP 背后的监督设计。", size=14, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_protocol_gap(data: Dict[str, object], output_path: Path) -> None:
    old_table = data["old_vs_expanded"]["attribution_table"]["old_protocol"]
    exp_table = data["old_vs_expanded"]["attribution_table"]["expanded_protocol"]
    items = [
        ("Retrieval only", old_table["retrieval_only"]["Dis@1_m"], exp_table["retrieval_only"]["Dis@1_m"]),
        ("Best post-retrieval", old_table["prior_topk5_best"]["Dis@1_m"], exp_table["prior_topk4"]["Dis@1_m"]),
    ]
    old_q = old_table["retrieval_only"]["query_count"]
    exp_q = exp_table["retrieval_only"]["query_count"]
    vmax = 350.0
    svg = SVG(1200, 520, "Old vs expanded protocol gap")
    svg.panel(24, 24, 1152, 472)
    draw_panel_title(
        svg,
        52,
        68,
        "Old main-table protocol vs expanded strict-pos protocol",
        f"Dis@1 (lower is better). Query count: old={old_q}, expanded={exp_q}.",
    )
    plot_x = 280
    plot_y = 440
    plot_w = 840
    draw_horizontal_scale(svg, plot_x, plot_y, plot_w, vmax, [0, 50, 100, 150, 200, 250, 300, 350], "meters")
    legend_y = 116
    svg.rect(52, legend_y, 18, 18, fill=PALETTE["old"], radius=4)
    svg.text(78, legend_y + 15, "Old main-table protocol", size=14, fill=PALETTE["muted"])
    svg.rect(276, legend_y, 18, 18, fill=PALETTE["expanded"], radius=4)
    svg.text(302, legend_y + 15, "Expanded strict-pos protocol", size=14, fill=PALETTE["muted"])
    base_y = 185
    row_gap = 150
    bar_h = 30
    for idx, (label, old_value, exp_value) in enumerate(items):
        block_y = base_y + idx * row_gap
        svg.text(238, block_y + 22, label, size=18, anchor="end", weight="600")
        old_w = plot_w * float(old_value) / vmax
        exp_w = plot_w * float(exp_value) / vmax
        svg.rect(plot_x, block_y, old_w, bar_h, fill=PALETTE["old"])
        svg.rect(plot_x, block_y + 42, exp_w, bar_h, fill=PALETTE["expanded"])
        svg.text(plot_x + old_w + 10, block_y + 21, f"{old_value:.1f} m", size=14, fill=PALETTE["old"], weight="600")
        svg.text(plot_x + exp_w + 10, block_y + 63, f"{exp_value:.1f} m", size=14, fill=PALETTE["expanded"], weight="600")
    gap = float(data["old_vs_expanded"]["stage_gap_decomposition"]["coarse_gap_share_of_final_gap_pct"])
    svg.text(
        52,
        474,
        f"Key message: about {gap:.1f}% of the old-to-expanded final gap already comes from the coarse retrieval stage.",
        size=14,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_expanded_methods(data: Dict[str, object], output_path: Path) -> None:
    methods = data["expanded_formal"]["summary"]["methods"]
    order = [
        ("retrieval_only", "retrieval"),
        ("no_rotate", "no_rotate"),
        ("rotate90", "rotate90"),
        ("prior_topk2", "prior_topk2"),
        ("prior_topk4", "prior_topk4"),
    ]
    labels = {
        "retrieval": "Retrieval only",
        "no_rotate": "No rotate",
        "rotate90": "Rotate 90",
        "prior_topk2": "VOP prior top-2",
        "prior_topk4": "VOP prior top-4",
    }
    colors = {
        "retrieval": PALETTE["retrieval"],
        "no_rotate": "#334155",
        "rotate90": PALETTE["rotate"],
        "prior_topk2": PALETTE["vop"],
        "prior_topk4": PALETTE["vop2"],
    }
    dis_max = 340.0
    time_max = 0.40
    svg = SVG(1280, 640, "Expanded formal comparison")
    svg.panel(24, 24, 1232, 592)
    draw_panel_title(
        svg,
        52,
        68,
        "Expanded strict-pos formal comparison",
        "377 queries. Left: Dis@1. Right: time per query. Lower is better on both axes.",
    )
    left_x = 250
    left_y = 555
    left_w = 420
    right_x = 770
    right_y = 555
    right_w = 420
    draw_horizontal_scale(svg, left_x, left_y, left_w, dis_max, [0, 100, 200, 300], "Dis@1 (m)")
    draw_horizontal_scale(svg, right_x, right_y, right_w, time_max, [0.0, 0.1, 0.2, 0.3, 0.4], "seconds / query")
    row_start = 170
    row_gap = 78
    bar_h = 24
    for idx, (key, color_key) in enumerate(order):
        y = row_start + idx * row_gap
        label = labels[color_key]
        info = methods[key]
        dis_value = float(info["metrics"]["Dis@1_m"])
        time_value = float(info["runtime_s_per_query"]["total"])
        fallback = float(info["stats"]["fallback_ratio_pct"])
        worse = float(info["stats"]["worse_than_coarse_ratio_pct"])
        svg.text(220, y + 18, label, size=17, anchor="end", weight="600")
        svg.rect(left_x, y, left_w * dis_value / dis_max, bar_h, fill=colors[color_key])
        svg.text(left_x + left_w * dis_value / dis_max + 10, y + 18, f"{dis_value:.1f}", size=14, fill=colors[color_key], weight="600")
        svg.rect(right_x, y, right_w * time_value / time_max, bar_h, fill=colors[color_key])
        svg.text(right_x + right_w * time_value / time_max + 10, y + 18, f"{time_value:.3f}s", size=14, fill=colors[color_key], weight="600")
        svg.text(220, y + 40, f"fb={fallback:.1f}%  worse={worse:.1f}%", size=12, anchor="end", fill=PALETTE["muted"])
    svg.text(
        52,
        598,
        "Observation: prior_topk4 is the best current expanded-split variant, but its absolute gain over retrieval is still small.",
        size=14,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_mechanism_control(data: Dict[str, object], output_path: Path) -> None:
    control = data["stage_b"]["mechanism_control_k3"]
    items = [
        ("Posterior", float(control["posterior"]["metrics"]["Dis@1_m"]), float(control["posterior"]["topk_oracle_best_coverage"]) * 100.0, float(control["posterior"]["runtime_s_per_query"]["total"]), PALETTE["vop2"]),
        ("Uniform", float(control["uniform"]["metrics"]["Dis@1_m"]), float(control["uniform"]["topk_oracle_best_coverage"]) * 100.0, float(control["uniform"]["runtime_s_per_query"]["total"]), "#CA8A04"),
        ("Random", float(control["random"]["Dis@1_m"]["mean"]), float(control["random"]["topk_oracle_best_coverage"]["mean"]) * 100.0, float(control["random"]["runtime_total_s_per_query"]["mean"]), PALETTE["random"]),
        ("Oracle", float(control["oracle"]["metrics"]["Dis@1_m"]), float(control["oracle"]["topk_oracle_best_coverage"]) * 100.0, float(control["oracle"]["runtime_s_per_query"]["total"]), PALETTE["oracle"]),
    ]
    dis_max = 330.0
    cov_max = 100.0
    svg = SVG(1280, 620, "Mechanism control k=3")
    svg.panel(24, 24, 1232, 572)
    draw_panel_title(
        svg,
        52,
        68,
        "Mechanism control for top-k orientation selection (k=3)",
        "Posterior already beats uniform/random, but the gap to the oracle still shows missing orientation information.",
    )
    left_x = 250
    left_y = 535
    left_w = 420
    right_x = 770
    right_y = 535
    right_w = 420
    draw_horizontal_scale(svg, left_x, left_y, left_w, dis_max, [0, 100, 200, 300], "Dis@1 (m)")
    draw_horizontal_scale(svg, right_x, right_y, right_w, cov_max, [0, 25, 50, 75, 100], "oracle-best coverage (%)")
    row_start = 170
    row_gap = 84
    bar_h = 24
    for idx, (label, dis1, cov, runtime, color) in enumerate(items):
        y = row_start + idx * row_gap
        svg.text(220, y + 18, label, size=18, anchor="end", weight="600")
        svg.rect(left_x, y, left_w * dis1 / dis_max, bar_h, fill=color)
        svg.text(left_x + left_w * dis1 / dis_max + 10, y + 18, f"{dis1:.1f}", size=14, fill=color, weight="600")
        svg.rect(right_x, y, right_w * cov / cov_max, bar_h, fill=color)
        svg.text(right_x + right_w * cov / cov_max + 10, y + 18, f"{cov:.1f}%", size=14, fill=color, weight="600")
        svg.text(220, y + 42, f"time={runtime:.3f}s", size=12, anchor="end", fill=PALETTE["muted"])
    svg.text(
        52,
        578,
        "Honest read: VOP is useful as a selector, but current posterior confidence remains too flat on the expanded protocol.",
        size=14,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_scene_breakdown(data: Dict[str, object], output_path: Path) -> None:
    scene_stats = data["old_vs_expanded"]["expanded_scene_stats"]
    scenes = ["01", "02", "03", "04", "08"]
    vmax = 1000.0
    svg = SVG(1280, 660, "Expanded scene breakdown")
    svg.panel(24, 24, 1232, 612)
    draw_panel_title(
        svg,
        52,
        68,
        "Scene-wise breakdown on the expanded strict-pos protocol",
        "Rows show Dis@1. Scene 08 remains the dominant failure case.",
    )
    plot_x = 310
    plot_y = 585
    plot_w = 900
    draw_horizontal_scale(svg, plot_x, plot_y, plot_w, vmax, [0, 250, 500, 750, 1000], "Dis@1 (m)")
    legend_y = 116
    legend = [
        ("Retrieval only", PALETTE["retrieval"]),
        ("Rotate 90", PALETTE["rotate"]),
        ("VOP prior top-4", PALETTE["vop2"]),
    ]
    lx = 52
    for label, color in legend:
        svg.rect(lx, legend_y, 18, 18, fill=color, radius=4)
        svg.text(lx + 26, legend_y + 15, label, size=14, fill=PALETTE["muted"])
        lx += 190
    row_start = 165
    row_gap = 86
    bar_h = 18
    offsets = [0, 24, 48]
    for idx, scene in enumerate(scenes):
        y = row_start + idx * row_gap
        stats = scene_stats[scene]
        q_count = int(stats["query_count"])
        values = [
            ("Retrieval only", float(stats["retrieval"]["Dis@1_m"]), PALETTE["retrieval"]),
            ("Rotate 90", float(stats["methods"]["rotate90"]["Dis@1_m"]), PALETTE["rotate"]),
            ("VOP prior top-4", float(stats["methods"]["prior_topk4"]["Dis@1_m"]), PALETTE["vop2"]),
        ]
        svg.text(280, y + 18, f"Scene {scene}", size=18, anchor="end", weight="600")
        svg.text(280, y + 38, f"q={q_count}", size=12, anchor="end", fill=PALETTE["muted"])
        for offset, (_, value, color) in zip(offsets, values):
            yy = y + offset
            svg.rect(plot_x, yy, plot_w * value / vmax, bar_h, fill=color)
            svg.text(plot_x + plot_w * value / vmax + 10, yy + 14, f"{value:.1f}", size=13, fill=color, weight="600")
        if scene == "08":
            svg.rect(34, y - 10, 1188, 74, fill="none", radius=12, stroke=PALETTE["highlight"])
    svg.text(
        52,
        622,
        "Scene 03/04 benefit clearly from post-retrieval refinement; scene 08 suggests the bottleneck is mostly upstream of fine localization.",
        size=14,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_pose_retrieval(data: Dict[str, object], output_path: Path) -> None:
    abcd = data["pose_abcd_large"]
    cd = data["pose_cd_main"]
    methods = [
        ("A", "无对齐 + 无姿态", float(abcd["A_noalign_nopose"]["Recall@1"]), PALETTE["retrieval"]),
        ("B", "无对齐 + 有姿态", float(abcd["B_noalign_pose"]["Recall@1"]), "#F97316"),
        ("C", "有对齐 + 无姿态", float(abcd["C_align_nopose"]["Recall@1"]), PALETTE["rotate"]),
        ("D", "有对齐 + 有姿态", float(abcd["D_align_pose"]["Recall@1"]), PALETTE["vop2"]),
    ]
    c_values = [
        float(cd["C_align_nopose_seed1"]["Recall@1"]),
        float(cd["C_align_nopose_seed2"]["Recall@1"]),
        float(cd["C_align_nopose_seed3"]["Recall@1"]),
    ]
    d_values = [
        float(cd["D_align_pose_residual_seed1"]["Recall@1"]),
        float(cd["D_align_pose_residual_seed2"]["Recall@1"]),
        float(cd["D_align_pose_residual_seed3"]["Recall@1"]),
    ]
    svg = SVG(1280, 620, "检索侧姿态分支消融")
    svg.panel(24, 24, 1232, 572)
    draw_panel_title(
        svg,
        52,
        68,
        "检索侧姿态分支（仅作背景，不是细定位主线）",
        "对齐明显有帮助，但姿态门控检索目前还不够稳定，暂不适合作为主投稿贡献。",
    )
    left_x = 70
    left_y = 520
    left_w = 520
    vmax = 75.0
    draw_horizontal_scale(svg, left_x + 150, left_y, left_w - 170, vmax, [0, 15, 30, 45, 60, 75], "Recall@1（%）")
    row_start = 170
    row_gap = 78
    bar_h = 22
    for idx, (short, label, value, color) in enumerate(methods):
        y = row_start + idx * row_gap
        svg.text(left_x + 110, y + 16, short, size=18, anchor="end", weight="700")
        svg.text(left_x + 120, y + 16, label, size=14, fill=PALETTE["muted"])
        bar_x = left_x + 150
        bar_w = (left_w - 170) * value / vmax
        svg.rect(bar_x, y, bar_w, bar_h, fill=color)
        svg.text(bar_x + bar_w + 8, y + 16, f"{value:.1f}", size=13, fill=color, weight="600")
    right_panel_x = 680
    right_panel_w = 520
    svg.text(right_panel_x, 150, "对齐检索下的 3 次重复稳定性", size=22, weight="700")
    svg.text(right_panel_x, 178, "圆点表示单个随机种子，横条表示 Recall@1 的均值。", size=14, fill=PALETTE["muted"])
    axis_x = right_panel_x + 110
    axis_y = 500
    plot_w = 360
    draw_horizontal_scale(svg, axis_x, axis_y, plot_w, vmax, [0, 15, 30, 45, 60, 75], "Recall@1（%）")
    mean_c = mean(c_values)
    mean_d = mean(d_values)
    rows = [
        ("有对齐 + 无姿态", c_values, mean_c, PALETTE["rotate"]),
        ("有对齐 + 姿态（残差）", d_values, mean_d, PALETTE["vop2"]),
    ]
    for idx, (label, values, avg, color) in enumerate(rows):
        y = 250 + idx * 120
        svg.text(axis_x - 20, y + 16, label, size=16, anchor="end", weight="600")
        svg.rect(axis_x, y, plot_w * avg / vmax, 24, fill=color)
        svg.text(axis_x + plot_w * avg / vmax + 8, y + 17, f"均值={avg:.2f}", size=13, fill=color, weight="600")
        for dot_idx, value in enumerate(values):
            cy = y + 54
            cx = axis_x + plot_w * float(value) / vmax
            svg.circle(cx, cy, 6, fill=color, stroke=PALETTE["panel"])
            svg.text(cx, cy + 22, f"s{dot_idx + 1}:{value:.1f}", size=11, fill=PALETTE["muted"], anchor="middle")
    svg.text(
        52,
        588,
        "结论：保留带对齐的检索作为背景信息，但不要把姿态门控检索当成主投稿路线。",
        size=14,
        fill=PALETTE["muted"],
    )
    write_svg(output_path, svg)


def chart_paper_compare(data: Dict[str, object], output_path: Path) -> None:
    target = data["paper_table4_target"]
    current = data["paper7_current_repro_latest"]
    split_counts = data["split_counts"]
    svg = SVG(1420, 700, "与原论文主表的对比")
    svg.panel(20, 20, 1380, 660)
    draw_panel_title(
        svg,
        48,
        62,
        "与原论文主表的对应关系",
        "原论文 Table 4 是检索主表。当前最可比的是本地 same-area-paper7 的 retrieval-only 复现，而不是 03/04 子集上的 VOP 细定位结果。",
    )
    svg.rect(50, 128, 1320, 140, fill=PALETTE["panel"], stroke=PALETTE["grid"], radius=18)
    notes = [
        "原论文 Table 4 指标：R@1、R@5、AP、SDM@3、Dis@1；本质上是检索主表，不是细定位主表。",
        f"本地 paper7 协议当前严格位置测试查询数为 {split_counts['legacy_strict_pos_queries']}；实际有效测试场景主要是 01/02/03/04/08。",
        "因此当前 03/04 上的 VOP 结果可以说明方法有效，但不能直接写成“超过原论文主表”。",
    ]
    for idx, line in enumerate(notes):
        svg.text(78, 172 + idx * 28, line, size=17, fill=PALETTE["muted"])

    left_x = 70
    left_y = 330
    left_w = 620
    svg.text(left_x, 308, "可直接比较的检索指标", size=24, weight="700")
    metrics = [
        ("R@1", float(target["R@1"]), float(current["Recall@1"]), 100.0),
        ("R@5", float(target["R@5"]), float(current["Recall@5"]), 100.0),
        ("AP", float(target["AP"]), float(current["AP"]), 100.0),
        ("SDM@3", float(target["SDM@3"]), float(current["SDM@3_pct"]), 100.0),
    ]
    legend_y = 300
    svg.rect(left_x + 330, legend_y - 12, 18, 18, fill=PALETTE["old"], radius=4)
    svg.text(left_x + 356, legend_y + 3, "原论文 Table 4", size=14, fill=PALETTE["muted"])
    svg.rect(left_x + 500, legend_y - 12, 18, 18, fill=PALETTE["vop2"], radius=4)
    svg.text(left_x + 526, legend_y + 3, "当前本地 paper7 复现", size=14, fill=PALETTE["muted"])
    row_gap = 78
    bar_h = 20
    plot_x = left_x + 170
    plot_w = 470
    draw_horizontal_scale(svg, plot_x, 640, plot_w, 100.0, [0, 25, 50, 75, 100], "百分比（越高越好）")
    for idx, (label, target_v, current_v, vmax) in enumerate(metrics):
        y = left_y + idx * row_gap
        svg.text(left_x + 130, y + 18, label, size=18, anchor="end", weight="600")
        svg.rect(plot_x, y, plot_w * target_v / vmax, bar_h, fill=PALETTE["old"])
        svg.rect(plot_x, y + 28, plot_w * current_v / vmax, bar_h, fill=PALETTE["vop2"])
        svg.text(plot_x + plot_w * target_v / vmax + 8, y + 16, f"{target_v:.2f}", size=13, fill=PALETTE["old"], weight="600")
        svg.text(plot_x + plot_w * current_v / vmax + 8, y + 44, f"{current_v:.2f}", size=13, fill=PALETTE["vop2"], weight="600")

    right_x = 760
    svg.text(right_x, 308, "Dis@1 对比与解读", size=24, weight="700")
    dis_target = float(target["Dis@1_m"])
    dis_current = float(current["Dis@1_m"])
    vmax = max(dis_target, dis_current) * 1.1
    draw_horizontal_scale(svg, right_x + 120, 520, 520, vmax, [0, 100, 200, 300], "米（越低越好）")
    svg.text(right_x + 100, 392, "原论文", size=18, anchor="end", weight="600")
    svg.rect(right_x + 120, 372, 520 * dis_target / vmax, 24, fill=PALETTE["old"])
    svg.text(right_x + 120 + 520 * dis_target / vmax + 8, 390, f"{dis_target:.2f}m", size=14, fill=PALETTE["old"], weight="600")
    svg.text(right_x + 100, 442, "当前复现", size=18, anchor="end", weight="600")
    svg.rect(right_x + 120, 422, 520 * dis_current / vmax, 24, fill=PALETTE["vop2"])
    svg.text(right_x + 120 + 520 * dis_current / vmax + 8, 440, f"{dis_current:.2f}m", size=14, fill=PALETTE["vop2"], weight="600")
    gap_r1 = float(current["Recall@1"]) - float(target["R@1"])
    gap_ap = float(current["AP"]) - float(target["AP"])
    gap_dis = dis_current - dis_target
    svg.text(right_x, 576, f"当前本地复现（最新已完成评估）相对原论文：R@1 {gap_r1:+.2f}，AP {gap_ap:+.2f}，Dis@1 {gap_dis:+.2f}m。", size=16, fill=PALETTE["muted"])
    svg.text(right_x, 606, "结论：paper7 retrieval 复现还没有对齐原论文主表，所以 VOP 当前更适合先作为方法证据，而不是直接拿去对主表宣称提升。", size=16, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def chart_pipeline_flow(output_path: Path) -> None:
    svg = SVG(1460, 660, "从检索到定位的完整流程")
    svg.panel(20, 20, 1420, 620)
    draw_panel_title(
        svg,
        48,
        60,
        "从检索到定位的完整算法流程",
        "在线推理时，VOP 插在 top-1 候选进入匹配之前，用来缩小角度搜索空间。",
    )
    boxes = [
        (50, 170, 165, 82, "1 输入\n无人机查询图像", PALETTE["panel"], PALETTE["grid"]),
        (255, 170, 180, 82, "2 检索模型\n排序卫星图库", PALETTE["panel"], PALETTE["grid"]),
        (475, 170, 180, 82, "3 得到 top-K\n卫星候选", PALETTE["panel"], PALETTE["grid"]),
        (695, 170, 200, 82, "4 取 top-1 候选\n进入细定位", PALETTE["panel"], PALETTE["grid"]),
        (220, 360, 200, 82, "5 提取检索特征\nquery / gallery", "#ECFDF5", "#86EFAC"),
        (460, 360, 220, 82, "6 VOP 预测角度后验\n选出 top-k", "#FFF7ED", "#FDBA74"),
        (720, 360, 240, 82, "7 对每个候选角度\n旋转 query 并跑匹配器", "#FFF7ED", "#FDBA74"),
        (1000, 360, 220, 82, "8 比较几何质量\n内点数 / 内点率选优", "#ECFDF5", "#86EFAC"),
        (1260, 360, 150, 82, "9 输出最终位置", PALETTE["panel"], PALETTE["grid"]),
        (1010, 170, 390, 82, "结果统计与诊断\nDis@K / 内点 / 回退率 / 耗时", "#EFF6FF", "#93C5FD"),
    ]
    for x, y, w, h, label, fill, stroke in boxes:
        svg.rect(x, y, w, h, fill=fill, radius=16, stroke=stroke)
        lines = str(label).split("\n")
        base = y + 34 - 12 * (len(lines) - 1) / 2
        for idx, line in enumerate(lines):
            svg.text(x + w / 2, base + idx * 22, line, size=17, anchor="middle", weight="600")
    arrows = [
        (215, 211, 255, 211),
        (435, 211, 475, 211),
        (655, 211, 695, 211),
        (795, 252, 795, 310),
        (795, 310, 320, 310),
        (320, 310, 320, 360),
        (420, 401, 460, 401),
        (680, 401, 720, 401),
        (960, 401, 1000, 401),
        (1220, 401, 1260, 401),
        (895, 211, 1010, 211),
        (1320, 252, 1320, 310),
        (1320, 310, 1335, 310),
        (1110, 252, 1110, 360),
    ]
    for x1, y1, x2, y2 in arrows:
        svg.line(x1, y1, x2, y2, width=2.2, marker_end=True)
    svg.text(220, 478, "当前主线：VOP top-k 方案", size=20, weight="700", fill=PALETTE["vop2"])
    svg.text(220, 506, "含义是先选少量角度，再分别做真实匹配与几何验证。", size=16, fill=PALETTE["muted"])
    svg.text(220, 548, "与 baseline 的区别：不是对所有角度穷举搜索，而是先做方向筛选。", size=16, fill=PALETTE["muted"])
    svg.text(1000, 478, "可选分支：置信度判别器", size=20, weight="700", fill=PALETTE["old"])
    svg.text(1000, 506, "它用于在多个候选细定位结果之间再做一次接受/拒绝。", size=16, fill=PALETTE["muted"])
    svg.text(1000, 548, "但当前阶段不建议把它作为主贡献来讲。", size=16, fill=PALETTE["muted"])
    write_svg(output_path, svg)


def build_summary() -> Dict[str, object]:
    pose_abcd_large = parse_summary_txt(ROOT / "Game4Loc/work_dir/visloc_pose_abcd_large_20260403_195252/summary.txt")
    pose_cd_main = parse_summary_txt(ROOT / "Game4Loc/work_dir/visloc_pose_cd_main_20260404_011657/summary.txt")
    pose_gate_ablation = parse_summary_txt(ROOT / "Game4Loc/work_dir/visloc_pose_gate_ablation_20260404_081711/summary.txt")
    paper7_retrieval_log = ROOT / "Game4Loc/work_dir/expanded_logs/paper7_retrieval_20260409_152630/train_paper7.stdout.log"
    paper7_retrieval_history = parse_retrieval_eval_log(paper7_retrieval_log)
    paper7_current_repro_latest = paper7_retrieval_history[-1] if paper7_retrieval_history else {}
    phase_a = read_json(ROOT / "Game4Loc/work_dir/vop/topk_analysis/phaseA_official_main_table_0408.json")
    vop_old_compact = read_json(ROOT / "Game4Loc/work_dir/vop/topk_analysis/summary_compact_0408.json")
    vop_current_diag = read_json(ROOT / "Game4Loc/work_dir/vop/diagnostics_0407_full_rankce_current.json")
    vop_useful_diag = read_json(ROOT / "Game4Loc/work_dir/vop/diagnostics_0408_useful5_top3.json")
    cache_same_area = read_json(ROOT / "Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json")
    stage_b = read_json(ROOT / "Game4Loc/work_dir/expanded_summaries/stage_b_20260409_092856/stage_b_summary.json")
    old_vs_expanded = read_json(
        ROOT / "Game4Loc/work_dir/expanded_summaries/protocol_attribution_20260409_124652/old_vs_expanded_attribution_summary.json"
    )
    expanded_formal = read_json(
        ROOT / "Game4Loc/work_dir/expanded_summaries/protocol_attribution_20260409_124652/expanded_formal_attribution.json"
    )
    expanded_split = read_json(ROOT / "Game4Loc/data/UAV_VisLoc_dataset/same-area-expanded-split-summary.json")
    legacy_split = read_json(ROOT / "Game4Loc/data/UAV_VisLoc_dataset/same-area-paper7-split-summary.json")
    vop_old = read_json(ROOT / "Game4Loc/work_dir/vop/summary_0408_effective_angle_topk.json")

    c_values = [
        float(pose_cd_main["C_align_nopose_seed1"]["Recall@1"]),
        float(pose_cd_main["C_align_nopose_seed2"]["Recall@1"]),
        float(pose_cd_main["C_align_nopose_seed3"]["Recall@1"]),
    ]
    d_values = [
        float(pose_cd_main["D_align_pose_residual_seed1"]["Recall@1"]),
        float(pose_cd_main["D_align_pose_residual_seed2"]["Recall@1"]),
        float(pose_cd_main["D_align_pose_residual_seed3"]["Recall@1"]),
    ]
    stage_gap = old_vs_expanded["stage_gap_decomposition"]
    old_scene_records = {}
    for name in ["posterior_k1", "posterior_k3", "posterior_k4", "posterior_k5", "uniform_k3", "oracle_k3"]:
        payload = read_json(ROOT / f"Game4Loc/work_dir/vop/topk_analysis/{name}.json")
        old_scene_records[name] = scene_mean(payload["records"])
    old_scene_counts = scene_counts_from_query_names(rec["query_name"] for rec in cache_same_area["records"])
    summary = {
        "sources": {
            "pose_abcd_large": str(ROOT / "Game4Loc/work_dir/visloc_pose_abcd_large_20260403_195252/summary.txt"),
            "pose_cd_main": str(ROOT / "Game4Loc/work_dir/visloc_pose_cd_main_20260404_011657/summary.txt"),
            "pose_gate_ablation": str(ROOT / "Game4Loc/work_dir/visloc_pose_gate_ablation_20260404_081711/summary.txt"),
            "paper7_retrieval_log": str(paper7_retrieval_log),
            "phase_a": str(ROOT / "Game4Loc/work_dir/vop/topk_analysis/phaseA_official_main_table_0408.json"),
            "vop_old_compact": str(ROOT / "Game4Loc/work_dir/vop/topk_analysis/summary_compact_0408.json"),
            "vop_current_diag": str(ROOT / "Game4Loc/work_dir/vop/diagnostics_0407_full_rankce_current.json"),
            "vop_useful_diag": str(ROOT / "Game4Loc/work_dir/vop/diagnostics_0408_useful5_top3.json"),
            "cache_same_area_full": str(ROOT / "Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json"),
            "stage_b": str(ROOT / "Game4Loc/work_dir/expanded_summaries/stage_b_20260409_092856/stage_b_summary.json"),
            "old_vs_expanded": str(
                ROOT / "Game4Loc/work_dir/expanded_summaries/protocol_attribution_20260409_124652/old_vs_expanded_attribution_summary.json"
            ),
            "expanded_formal": str(
                ROOT / "Game4Loc/work_dir/expanded_summaries/protocol_attribution_20260409_124652/expanded_formal_attribution.json"
            ),
            "expanded_split": str(ROOT / "Game4Loc/data/UAV_VisLoc_dataset/same-area-expanded-split-summary.json"),
            "legacy_split": str(ROOT / "Game4Loc/data/UAV_VisLoc_dataset/same-area-paper7-split-summary.json"),
            "vop_old_summary": str(ROOT / "Game4Loc/work_dir/vop/summary_0408_effective_angle_topk.json"),
        },
        "pose_abcd_large": pose_abcd_large,
        "pose_cd_main": pose_cd_main,
        "pose_gate_ablation": pose_gate_ablation,
        "paper7_current_repro_latest": paper7_current_repro_latest,
        "paper_table4_target": {
            "R@1": 80.20,
            "R@5": 96.53,
            "AP": 87.83,
            "SDM@3": 85.46,
            "Dis@1_m": 122.87,
        },
        "pose_cd_main_stats": {
            "align_no_pose_mean_r1": mean(c_values),
            "align_no_pose_std_r1": std(c_values),
            "align_pose_mean_r1": mean(d_values),
            "align_pose_std_r1": std(d_values),
        },
        "phase_a": phase_a,
        "vop_old_compact": vop_old_compact,
        "vop_current_diag": vop_current_diag,
        "vop_useful_diag": vop_useful_diag,
        "vop_scene_0304": old_scene_records,
        "vop_scene_counts": old_scene_counts,
        "stage_b": stage_b,
        "old_vs_expanded": old_vs_expanded,
        "expanded_formal": expanded_formal,
        "split_counts": {
            "expanded_strict_pos_queries": int(expanded_split["overall"]["test_pos_queries"]),
            "expanded_test_queries": int(expanded_split["overall"]["test_queries"]),
            "legacy_strict_pos_queries": int(legacy_split["overall"]["test_pos_queries"]),
            "legacy_test_queries": int(legacy_split["overall"]["test_queries"]),
        },
        "vop_old_summary": vop_old,
        "executive_points": [
            "VOP is a visual orientation posterior: it predicts a small set of likely angles and lets the matcher avoid brute-force rotation search.",
            "On the original 03/04 small-range setting, VOP prior-topk consistently outperforms random and uniform angle selection, so it is learning a meaningful orientation signal.",
            "Retrieval-side pose gating was tried systematically, but it is not stable enough yet; it should stay as supporting context rather than the paper's main claim.",
        ],
        "stage_gap_digest": {
            "coarse_gap_old_to_expanded_m": float(stage_gap["coarse_gap_old_to_expanded_m"]),
            "best_final_gap_old_to_expanded_m": float(stage_gap["best_final_gap_old_to_expanded_m"]),
            "coarse_gap_share_pct": float(stage_gap["coarse_gap_share_of_final_gap_pct"]),
        },
    }
    return summary


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    DATA_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    chart_problem_background(FIG_DIR / "00_problem_background.svg")
    chart_pipeline_flow(FIG_DIR / "01_full_pipeline.svg")
    chart_vop_method(FIG_DIR / "02_vop_method.svg")
    chart_vop_teacher_stats(summary, FIG_DIR / "03_vop_teacher_stats.svg")
    chart_vop_k_curve(summary, FIG_DIR / "04_vop_k_curve.svg")
    chart_vop_controls_old(summary, FIG_DIR / "05_vop_controls_old.svg")
    chart_vop_scene_0304(summary, FIG_DIR / "06_vop_scene_0304.svg")
    chart_vop_variant_progress(summary, FIG_DIR / "07_vop_variant_progress.svg")
    chart_pose_retrieval(summary, FIG_DIR / "08_retrieval_attempts.svg")
    chart_paper_compare(summary, FIG_DIR / "09_paper_compare.svg")
    print(f"Wrote summary to {DATA_PATH}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
