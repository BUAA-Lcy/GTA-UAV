import argparse
import os
import re
from pathlib import Path


RESULTS_MARKER = "Parsed Results:"


def parse_metrics_from_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    eval_matches = re.findall(
        r"Recall@1:\s*([0-9.]+)\s*-\s*Recall@5:\s*([0-9.]+)\s*-\s*Recall@10:\s*([0-9.]+).*?AP:\s*([0-9.]+)",
        text,
    )
    best_matches = re.findall(r"训练完成，最佳 Recall@1=([0-9.]+)", text)

    result = {
        "recall1": None,
        "recall5": None,
        "recall10": None,
        "ap": None,
        "best_recall1": None,
    }

    if eval_matches:
        recall1, recall5, recall10, ap = eval_matches[-1]
        result["recall1"] = float(recall1)
        result["recall5"] = float(recall5)
        result["recall10"] = float(recall10)
        result["ap"] = float(ap)

    if best_matches:
        result["best_recall1"] = float(best_matches[-1]) * 100.0

    return result


def find_latest_log(log_dir: Path):
    logs = sorted(log_dir.glob("*.log"))
    return logs[-1] if logs else None


def find_latest_weight(ckpt_dir: Path):
    weights = sorted(ckpt_dir.rglob("weights_end.pth"))
    return weights[-1] if weights else None


def format_value(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def build_results_section(run_root: Path):
    log_root = run_root / "logs"
    ckpt_root = run_root / "checkpoints"

    experiment_dirs = []
    if log_root.exists():
        experiment_dirs = sorted([p for p in log_root.iterdir() if p.is_dir()])

    lines = [RESULTS_MARKER, ""]
    if not experiment_dirs:
        lines.append("No experiment log directories found.")
        lines.append("")
        return "\n".join(lines)

    for exp_dir in experiment_dirs:
        name = exp_dir.name
        latest_log = find_latest_log(exp_dir)
        latest_weight = find_latest_weight(ckpt_root / name)
        metrics = parse_metrics_from_log(latest_log) if latest_log else {}

        lines.append(f"[{name}]")
        lines.append(f"latest_log={latest_log if latest_log else ''}")
        lines.append(f"weights_end={latest_weight if latest_weight else ''}")
        lines.append(f"Recall@1={format_value(metrics.get('recall1'))}")
        lines.append(f"Recall@5={format_value(metrics.get('recall5'))}")
        lines.append(f"Recall@10={format_value(metrics.get('recall10'))}")
        lines.append(f"AP={format_value(metrics.get('ap'))}")
        lines.append(f"Best_Recall@1={format_value(metrics.get('best_recall1'))}")
        lines.append("")

    return "\n".join(lines)


def refresh_summary(summary_path: Path, run_root: Path):
    if summary_path.exists():
        original = summary_path.read_text(encoding="utf-8", errors="ignore")
    else:
        original = ""

    prefix = original.split(RESULTS_MARKER, 1)[0].rstrip()
    results_section = build_results_section(run_root)

    if prefix:
        new_text = f"{prefix}\n\n{results_section}\n"
    else:
        new_text = f"{results_section}\n"

    summary_path.write_text(new_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Rebuild score summary for a UAV-VisLoc multi-run directory.")
    parser.add_argument("--run_root", type=str, required=True, help="Run root containing logs/ and checkpoints/")
    parser.add_argument("--summary_file", type=str, default=None, help="Summary file to update. Defaults to <run_root>/summary.txt")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    summary_path = Path(args.summary_file).resolve() if args.summary_file else run_root / "summary.txt"

    refresh_summary(summary_path, run_root)
    print(f"[OK] summary refreshed: {summary_path}")


if __name__ == "__main__":
    main()
