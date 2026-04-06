import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "Log" / "visloc_matcher_tuning"
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
    "--multi_scale", "True",
    "--num_workers", "0",
]


@dataclass(frozen=True)
class RunConfig:
    stage: str
    name: str
    query_limit: int
    sparse_ransac_method: str = "USAC_MAGSAC"
    sparse_ransac_reproj_threshold: float = 20.0
    sparse_sp_detection_threshold: float = 0.005
    sparse_lg_filter_threshold: float = 0.0
    sparse_scales: str = "1.0,0.75,0.5,1.25"
    multi_scale: bool = True


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
        "--sparse_sp_detection_threshold", str(config.sparse_sp_detection_threshold),
        "--sparse_lg_filter_threshold", str(config.sparse_lg_filter_threshold),
        "--sparse_scales", config.sparse_scales,
        "--multi_scale", "True" if config.multi_scale else "False",
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

    result = {
        "stage": config.stage,
        "name": config.name,
        "query_limit": config.query_limit,
        "ransac_method": config.sparse_ransac_method,
        "reproj": config.sparse_ransac_reproj_threshold,
        "sp_det": config.sparse_sp_detection_threshold,
        "lg_filter": config.sparse_lg_filter_threshold,
        "scales": config.sparse_scales,
        "multi_scale": config.multi_scale,
        "started_at": started,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "dis1": metric_from_output(r"Dis@1:\s*([0-9.]+)", output),
        "sdm1": metric_from_output(r"SDM@1:\s*([0-9.]+)", output),
        "recall1": metric_from_output(r"Recall@1:\s*([0-9.]+)", output),
        "match_total_s": metric_from_output(r"with_match 阶段平均耗时: .*总计=([0-9.]+)s", output),
    }
    return result


def score_key(result: dict):
    dis1 = result.get("dis1")
    match_total_s = result.get("match_total_s")
    sdm1 = result.get("sdm1")
    return (
        float("inf") if dis1 is None else dis1,
        float("inf") if match_total_s is None else match_total_s,
        float("inf") if sdm1 is None else -sdm1,
    )


def append_summary(summary_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "stage",
        "name",
        "query_limit",
        "ransac_method",
        "reproj",
        "sp_det",
        "lg_filter",
        "scales",
        "multi_scale",
        "dis1",
        "sdm1",
        "recall1",
        "match_total_s",
        "returncode",
        "log_path",
        "started_at",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
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
            f"  {row['name']}: Dis@1={row['dis1']:.4f}, SDM@1={row['sdm1']:.4f}, "
            f"Recall@1={row['recall1']:.4f}, match_total={row['match_total_s']:.6f}s"
        )


def best_result(rows: list[dict]) -> dict:
    good = [row for row in rows if row.get("returncode") == 0 and row.get("dis1") is not None]
    good.sort(key=score_key)
    if not good:
        raise RuntimeError("No successful runs were found in the current stage.")
    return good[0]


def main() -> int:
    run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"

    search_limit = 100
    all_rows: list[dict] = []
    run_id = 1

    baseline_cfg = RunConfig(stage="baseline", name="baseline_default", query_limit=search_limit)
    baseline_row = run_eval(run_dir, run_id, baseline_cfg)
    all_rows.append(baseline_row)
    run_id += 1
    append_summary(summary_path, all_rows)

    stage1_cfgs = [
        RunConfig(stage="stage1_ransac", name=f"ransac_{method.lower()}_rp{int(reproj)}", query_limit=search_limit,
                  sparse_ransac_method=method, sparse_ransac_reproj_threshold=reproj)
        for method in ("USAC_MAGSAC", "USAC_FAST", "RANSAC")
        for reproj in (10.0, 20.0, 40.0)
    ]
    stage1_rows = []
    for cfg in stage1_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage1_rows.append(row)
        all_rows.append(row)
        run_id += 1
        append_summary(summary_path, all_rows)
    best_stage1 = best_result(stage1_rows + [baseline_row])
    print_top(stage1_rows + [baseline_row], "Top after Stage 1 (RANSAC)")

    stage2_cfgs = [
        RunConfig(
            stage="stage2_sp_lg",
            name=f"sp{sp_det:.3f}_lg{lg_filter:.2f}",
            query_limit=search_limit,
            sparse_ransac_method=best_stage1["ransac_method"],
            sparse_ransac_reproj_threshold=float(best_stage1["reproj"]),
            sparse_sp_detection_threshold=sp_det,
            sparse_lg_filter_threshold=lg_filter,
            sparse_scales=str(best_stage1["scales"]),
            multi_scale=bool(best_stage1["multi_scale"]),
        )
        for sp_det in (0.003, 0.005, 0.008)
        for lg_filter in (0.0, 0.05, 0.10)
    ]
    stage2_rows = []
    for cfg in stage2_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage2_rows.append(row)
        all_rows.append(row)
        run_id += 1
        append_summary(summary_path, all_rows)
    best_stage2 = best_result(stage2_rows + [best_stage1])
    print_top(stage2_rows + [best_stage1], "Top after Stage 2 (SP/LG)")

    scale_presets = [
        ("default4", True, "1.0,0.75,0.5,1.25"),
        ("compact3", True, "1.0,0.75,0.5"),
        ("balanced4", True, "1.0,0.8,0.6,1.2"),
        ("single1", False, "1.0"),
    ]
    stage3_cfgs = [
        RunConfig(
            stage="stage3_scales",
            name=f"scale_{preset_name}",
            query_limit=search_limit,
            sparse_ransac_method=best_stage2["ransac_method"],
            sparse_ransac_reproj_threshold=float(best_stage2["reproj"]),
            sparse_sp_detection_threshold=float(best_stage2["sp_det"]),
            sparse_lg_filter_threshold=float(best_stage2["lg_filter"]),
            sparse_scales=scales,
            multi_scale=multi_scale,
        )
        for preset_name, multi_scale, scales in scale_presets
    ]
    stage3_rows = []
    for cfg in stage3_cfgs:
        row = run_eval(run_dir, run_id, cfg)
        stage3_rows.append(row)
        all_rows.append(row)
        run_id += 1
        append_summary(summary_path, all_rows)
    best_stage3 = best_result(stage3_rows + [best_stage2])
    print_top(stage3_rows + [best_stage2], "Top after Stage 3 (Scales)")

    full_candidates = []
    seen = set()
    for row in (baseline_row, best_stage1, best_stage2, best_stage3):
        key = (
            row["ransac_method"],
            float(row["reproj"]),
            float(row["sp_det"]),
            float(row["lg_filter"]),
            str(row["scales"]),
            bool(row["multi_scale"]),
        )
        if key in seen:
            continue
        seen.add(key)
        full_candidates.append(
            RunConfig(
                stage="stage4_full",
                name=f"full_{row['name']}",
                query_limit=0,
                sparse_ransac_method=row["ransac_method"],
                sparse_ransac_reproj_threshold=float(row["reproj"]),
                sparse_sp_detection_threshold=float(row["sp_det"]),
                sparse_lg_filter_threshold=float(row["lg_filter"]),
                sparse_scales=str(row["scales"]),
                multi_scale=bool(row["multi_scale"]),
            )
        )

    stage4_rows = []
    for cfg in full_candidates:
        row = run_eval(run_dir, run_id, cfg)
        stage4_rows.append(row)
        all_rows.append(row)
        run_id += 1
        append_summary(summary_path, all_rows)
    print_top(stage4_rows, "Top after Stage 4 (Full set)")

    best_full = best_result(stage4_rows)
    final_txt = run_dir / "best_result.txt"
    final_txt.write_text(
        "\n".join(
            [
                f"best_name={best_full['name']}",
                f"dis1={best_full['dis1']}",
                f"sdm1={best_full['sdm1']}",
                f"recall1={best_full['recall1']}",
                f"match_total_s={best_full['match_total_s']}",
                f"ransac_method={best_full['ransac_method']}",
                f"reproj={best_full['reproj']}",
                f"sp_det={best_full['sp_det']}",
                f"lg_filter={best_full['lg_filter']}",
                f"scales={best_full['scales']}",
                f"multi_scale={best_full['multi_scale']}",
                f"log_path={best_full['log_path']}",
                f"summary_csv={summary_path}",
            ]
        ),
        encoding="utf-8",
    )

    print("\nBest full-set result")
    print(
        f"  {best_full['name']}: Dis@1={best_full['dis1']:.4f}, SDM@1={best_full['sdm1']:.4f}, "
        f"Recall@1={best_full['recall1']:.4f}, match_total={best_full['match_total_s']:.6f}s"
    )
    print(f"  summary: {summary_path}")
    print(f"  best:    {final_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
