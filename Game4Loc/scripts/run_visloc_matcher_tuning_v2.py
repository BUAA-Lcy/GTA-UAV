import csv
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "Log" / "visloc_matcher_tuning_v2"
CONDA_BIN = Path("/home/lcy/miniconda3/bin/conda")
PYTHON_CMD = [str(CONDA_BIN), "run", "--no-capture-output", "-n", "gtauav", "python"]

BASE_ARGS = [
    "eval_visloc.py",
    "--data_root", "./data/UAV_VisLoc_dataset",
    "--test_pairs_meta_file", "same-area-drone2sate-test.json",
    "--model", "vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
    "--checkpoint_start", "./pretrained/visloc/vit_base_eva_visloc_same_area_best.pth",
    "--with_match",
    "--sparse",
    "--use_yaw",
    "--rotate", "360",
    "--sparse_phase1_min_inliers", "0",
    "--gpu_ids", "0",
    "--num_workers", "0",
    "--multi_scale", "True",
]


@dataclass(frozen=True)
class RunConfig:
    stage: str
    name: str
    query_limit: int
    sparse_ransac_method: str = "RANSAC"
    sparse_ransac_reproj_threshold: float = 20.0
    sparse_ransac_confidence: float = 0.99
    sparse_ransac_max_iter: int = 1000
    sparse_sp_detection_threshold: float = 0.003
    sparse_sp_max_num_keypoints: int = 2048
    sparse_sp_nms_radius: int = 4
    sparse_sp_remove_borders: int = 4
    sparse_sp_max_edge: int = 1024
    sparse_lg_filter_threshold: float = 0.0
    sparse_scales: str = "1.0,0.8,0.6,1.2"
    multi_scale: bool = True
    sparse_max_matches_per_scale: int = 1024
    sparse_max_total_matches: int = 4096
    sparse_min_inliers: int = 15
    sparse_min_inlier_ratio: float = 0.001


def metric_from_output(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def run_eval(run_dir: Path, run_id: int, config: RunConfig) -> dict:
    name_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", config.name).strip("_") or "run"
    log_path = run_dir / f"{run_id:02d}_{config.stage}_{name_slug}.log"

    cmd = list(PYTHON_CMD) + list(BASE_ARGS) + [
        "--query_limit", str(config.query_limit),
        "--sparse_ransac_method", config.sparse_ransac_method,
        "--sparse_ransac_reproj_threshold", str(config.sparse_ransac_reproj_threshold),
        "--sparse_ransac_confidence", str(config.sparse_ransac_confidence),
        "--sparse_ransac_max_iter", str(config.sparse_ransac_max_iter),
        "--sparse_sp_detection_threshold", str(config.sparse_sp_detection_threshold),
        "--sparse_sp_max_num_keypoints", str(config.sparse_sp_max_num_keypoints),
        "--sparse_sp_nms_radius", str(config.sparse_sp_nms_radius),
        "--sparse_sp_remove_borders", str(config.sparse_sp_remove_borders),
        "--sparse_sp_max_edge", str(config.sparse_sp_max_edge),
        "--sparse_lg_filter_threshold", str(config.sparse_lg_filter_threshold),
        "--sparse_scales", config.sparse_scales,
        "--multi_scale", "True" if config.multi_scale else "False",
        "--sparse_max_matches_per_scale", str(config.sparse_max_matches_per_scale),
        "--sparse_max_total_matches", str(config.sparse_max_total_matches),
        "--sparse_min_inliers", str(config.sparse_min_inliers),
        "--sparse_min_inlier_ratio", str(config.sparse_min_inlier_ratio),
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = proc.stdout
    log_path.write_text(output, encoding="utf-8")
    return {
        "stage": config.stage,
        "name": config.name,
        "query_limit": config.query_limit,
        "ransac_method": config.sparse_ransac_method,
        "reproj": config.sparse_ransac_reproj_threshold,
        "ransac_conf": config.sparse_ransac_confidence,
        "ransac_iter": config.sparse_ransac_max_iter,
        "sp_det": config.sparse_sp_detection_threshold,
        "sp_kpts": config.sparse_sp_max_num_keypoints,
        "sp_nms": config.sparse_sp_nms_radius,
        "sp_edge": config.sparse_sp_max_edge,
        "lg_filter": config.sparse_lg_filter_threshold,
        "scales": config.sparse_scales,
        "multi_scale": config.multi_scale,
        "max_per_scale": config.sparse_max_matches_per_scale,
        "max_total": config.sparse_max_total_matches,
        "min_inliers": config.sparse_min_inliers,
        "min_ratio": config.sparse_min_inlier_ratio,
        "started_at": started,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "dis1": metric_from_output(r"Dis@1:\s*([0-9.]+)", output),
        "sdm1": metric_from_output(r"SDM@1:\s*([0-9.]+)", output),
        "recall1": metric_from_output(r"Recall@1:\s*([0-9.]+)", output),
        "match_total_s": metric_from_output(r"with_match 阶段平均耗时: .*总计=([0-9.]+)s", output),
    }


def score_key(row: dict):
    return (
        float("inf") if row.get("dis1") is None else row["dis1"],
        float("inf") if row.get("match_total_s") is None else row["match_total_s"],
    )


def append_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "stage", "name", "query_limit", "ransac_method", "reproj", "ransac_conf", "ransac_iter",
        "sp_det", "sp_kpts", "sp_nms", "sp_edge", "lg_filter", "scales", "multi_scale",
        "max_per_scale", "max_total", "min_inliers", "min_ratio",
        "dis1", "sdm1", "recall1", "match_total_s", "returncode", "log_path", "started_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def print_top(rows: list[dict], title: str, limit: int = 5) -> None:
    good = [row for row in rows if row.get("returncode") == 0 and row.get("dis1") is not None]
    good.sort(key=score_key)
    print(f"\n{title}")
    for row in good[:limit]:
        print(
            f"  {row['name']}: Dis@1={row['dis1']:.4f}, match_total={row['match_total_s']:.6f}s, "
            f"sp_kpts={row['sp_kpts']}, sp_nms={row['sp_nms']}, sp_edge={row['sp_edge']}, "
            f"caps=({row['max_per_scale']}/{row['max_total']}), min_h=({row['min_inliers']}/{row['min_ratio']})"
        )


def best(rows: list[dict]) -> dict:
    good = [row for row in rows if row.get("returncode") == 0 and row.get("dis1") is not None]
    good.sort(key=score_key)
    if not good:
        raise RuntimeError("No successful rows found.")
    return good[0]


def cfg_from_row(stage: str, name: str, query_limit: int, row: dict) -> RunConfig:
    return RunConfig(
        stage=stage,
        name=name,
        query_limit=query_limit,
        sparse_ransac_method=row["ransac_method"],
        sparse_ransac_reproj_threshold=float(row["reproj"]),
        sparse_ransac_confidence=float(row["ransac_conf"]),
        sparse_ransac_max_iter=int(row["ransac_iter"]),
        sparse_sp_detection_threshold=float(row["sp_det"]),
        sparse_sp_max_num_keypoints=int(row["sp_kpts"]),
        sparse_sp_nms_radius=int(row["sp_nms"]),
        sparse_sp_remove_borders=4,
        sparse_sp_max_edge=int(row["sp_edge"]),
        sparse_lg_filter_threshold=float(row["lg_filter"]),
        sparse_scales=str(row["scales"]),
        multi_scale=bool(row["multi_scale"]),
        sparse_max_matches_per_scale=int(row["max_per_scale"]),
        sparse_max_total_matches=int(row["max_total"]),
        sparse_min_inliers=int(row["min_inliers"]),
        sparse_min_inlier_ratio=float(row["min_ratio"]),
    )


def main() -> int:
    run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.csv"
    search_limit = 120
    rows: list[dict] = []
    run_id = 1

    base_cfg = RunConfig(stage="baseline", name="base_best_round1", query_limit=search_limit)
    base_row = run_eval(run_dir, run_id, base_cfg)
    rows.append(base_row)
    run_id += 1
    append_summary(summary, rows)

    stage1_cfgs = [
        RunConfig(stage="stage1_kpts", name=f"kpts{k}_nms{nms}", query_limit=search_limit,
                  sparse_sp_max_num_keypoints=k, sparse_sp_nms_radius=nms)
        for k in (1536, 2048, 3072)
        for nms in (3, 4, 5)
    ]
    stage1_rows = []
    for cfg in stage1_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage1_rows.append(row)
        rows.append(row)
        run_id += 1
        append_summary(summary, rows)
    best_stage1 = best(stage1_rows + [base_row])
    print_top(stage1_rows + [base_row], "Top after Stage 1 (Keypoints)")

    stage2_cfgs = [
        cfg_from_row("stage2_capacity", f"edge{edge}_cap{per}_{tot}", search_limit, best_stage1 | {
            "sp_edge": edge,
            "max_per_scale": per,
            "max_total": tot,
        })
        for edge, per, tot in (
            (896, 768, 3072),
            (1024, 1024, 4096),
            (1280, 1536, 6144),
            (1408, 2048, 8192),
        )
    ]
    stage2_rows = []
    for cfg in stage2_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage2_rows.append(row)
        rows.append(row)
        run_id += 1
        append_summary(summary, rows)
    best_stage2 = best(stage2_rows + [best_stage1])
    print_top(stage2_rows + [best_stage1], "Top after Stage 2 (Capacity)")

    stage3_cfgs = [
        cfg_from_row("stage3_quality", f"minh{mi}_r{mr}", search_limit, best_stage2 | {
            "min_inliers": mi,
            "min_ratio": mr,
        })
        for mi, mr in (
            (10, 0.0005),
            (12, 0.0008),
            (15, 0.0010),
            (20, 0.0015),
        )
    ]
    stage3_rows = []
    for cfg in stage3_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage3_rows.append(row)
        rows.append(row)
        run_id += 1
        append_summary(summary, rows)
    best_stage3 = best(stage3_rows + [best_stage2])
    print_top(stage3_rows + [best_stage2], "Top after Stage 3 (Quality Gate)")

    stage4_cfgs = [
        cfg_from_row("stage4_ransac", f"conf{conf}_iter{iters}", search_limit, best_stage3 | {
            "ransac_conf": conf,
            "ransac_iter": iters,
        })
        for conf, iters in (
            (0.95, 500),
            (0.99, 1000),
            (0.999, 2000),
        )
    ]
    stage4_rows = []
    for cfg in stage4_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage4_rows.append(row)
        rows.append(row)
        run_id += 1
        append_summary(summary, rows)
    best_stage4 = best(stage4_rows + [best_stage3])
    print_top(stage4_rows + [best_stage3], "Top after Stage 4 (RANSAC Confidence)")

    finalists = []
    seen = set()
    for row in (base_row, best_stage1, best_stage2, best_stage3, best_stage4):
        key = (
            row["ransac_method"], float(row["reproj"]), float(row["ransac_conf"]), int(row["ransac_iter"]),
            float(row["sp_det"]), int(row["sp_kpts"]), int(row["sp_nms"]), int(row["sp_edge"]),
            float(row["lg_filter"]), str(row["scales"]), bool(row["multi_scale"]),
            int(row["max_per_scale"]), int(row["max_total"]), int(row["min_inliers"]), float(row["min_ratio"]),
        )
        if key in seen:
            continue
        seen.add(key)
        finalists.append(cfg_from_row("stage5_full", f"full_{row['name']}", 0, row))

    full_rows = []
    for cfg in finalists:
        row = run_eval(run_dir, run_id, cfg)
        full_rows.append(row)
        rows.append(row)
        run_id += 1
        append_summary(summary, rows)
    best_full = best(full_rows)
    print_top(full_rows, "Top after Stage 5 (Full set)")

    best_txt = run_dir / "best_result.txt"
    best_txt.write_text(
        "\n".join([
            f"best_name={best_full['name']}",
            f"dis1={best_full['dis1']}",
            f"sdm1={best_full['sdm1']}",
            f"recall1={best_full['recall1']}",
            f"match_total_s={best_full['match_total_s']}",
            f"ransac_method={best_full['ransac_method']}",
            f"reproj={best_full['reproj']}",
            f"ransac_conf={best_full['ransac_conf']}",
            f"ransac_iter={best_full['ransac_iter']}",
            f"sp_det={best_full['sp_det']}",
            f"sp_kpts={best_full['sp_kpts']}",
            f"sp_nms={best_full['sp_nms']}",
            f"sp_edge={best_full['sp_edge']}",
            f"lg_filter={best_full['lg_filter']}",
            f"scales={best_full['scales']}",
            f"multi_scale={best_full['multi_scale']}",
            f"max_per_scale={best_full['max_per_scale']}",
            f"max_total={best_full['max_total']}",
            f"min_inliers={best_full['min_inliers']}",
            f"min_ratio={best_full['min_ratio']}",
            f"log_path={best_full['log_path']}",
            f"summary_csv={summary}",
        ]),
        encoding="utf-8",
    )

    print("\nBest full-set result")
    print(
        f"  {best_full['name']}: Dis@1={best_full['dis1']:.4f}, SDM@1={best_full['sdm1']:.4f}, "
        f"Recall@1={best_full['recall1']:.4f}, match_total={best_full['match_total_s']:.6f}s"
    )
    print(f"  summary: {summary}")
    print(f"  best:    {best_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
