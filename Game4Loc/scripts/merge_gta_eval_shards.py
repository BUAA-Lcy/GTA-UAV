#!/home/lcy/miniconda3/envs/gtauav/bin/python
import argparse
import glob
import math
import os
import re
from dataclasses import dataclass


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"


@dataclass
class ShardMetrics:
    path: str
    start: int
    end: int
    count: int
    total_queries: int
    recall1: float
    recall5: float
    recall10: float
    map_score: float
    sdm1: float
    sdm3: float
    sdm5: float
    dis1: float
    dis3: float
    dis5: float
    ma3: float
    ma5: float
    ma10: float
    ma20: float
    fallback: int
    worse_than_coarse: int
    identity_fallback: int
    out_of_bounds: int
    projection_invalid: int
    mean_retained: float
    mean_inliers: float
    mean_inlier_ratio: float
    mean_vop_time: float
    mean_matcher_time: float
    mean_total_time: float


def _search(pattern: str, text: str, path: str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Failed to parse pattern from {path}: {pattern}")
    return match


def _parse_log(path: str) -> ShardMetrics:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    slice_match = re.search(
        r"本次评估查询切片: start=(\d+) end=(\d+) count=(\d+) / total=(\d+)",
        text,
        flags=re.MULTILINE,
    )
    if slice_match is not None:
        start, end, count, total_queries = (int(slice_match.group(i)) for i in range(1, 5))
    else:
        total_match = _search(r"测试查询图像总数: (\d+)", text, path)
        query_match = _search(r"最终匹配统计\(按查询汇总\): .* queries=(\d+)", text, path)
        start = 0
        count = int(query_match.group(1))
        end = start + count
        total_queries = int(total_match.group(1))

    metric_match = _search(
        rf"Recall@1: ({FLOAT_RE}) - Recall@5: ({FLOAT_RE}) - Recall@10: ({FLOAT_RE}) - mAP: ({FLOAT_RE}) - "
        rf"SDM@1: ({FLOAT_RE}) - SDM@3: ({FLOAT_RE}) - SDM@5: ({FLOAT_RE}) - "
        rf"Dis@1: ({FLOAT_RE}) - Dis@3: ({FLOAT_RE}) - Dis@5: ({FLOAT_RE})",
        text,
        path,
    )
    ma_match = _search(
        rf"MA@3m: ({FLOAT_RE}) - MA@5m: ({FLOAT_RE}) - MA@10m: ({FLOAT_RE}) - MA@20m: ({FLOAT_RE})",
        text,
        path,
    )
    robust_match = _search(
        r"稳健性统计\(按查询汇总\): fallback=(\d+)\([\d.]+%\) worse-than-coarse=(\d+)\([\d.]+%\) "
        r"identity-H fallback=(\d+)\([\d.]+%\) out-of-bounds=(\d+)\([\d.]+%\) projection-invalid=(\d+)\([\d.]+%\)",
        text,
        path,
    )
    match_stats = _search(
        rf"最终匹配统计\(按查询汇总\): mean_retained_matches=({FLOAT_RE}) mean_inliers=({FLOAT_RE}) "
        rf"mean_inlier_ratio=({FLOAT_RE}) queries=(\d+)",
        text,
        path,
    )
    runtime_match = _search(
        rf"细定位耗时\(按查询汇总\): mean_vop_forward_time=({FLOAT_RE})s/query "
        rf"mean_matcher_time=({FLOAT_RE})s/query mean_total_time=({FLOAT_RE})s/query",
        text,
        path,
    )

    queries_from_stats = int(match_stats.group(4))
    if queries_from_stats != count:
        raise ValueError(
            f"Shard query count mismatch in {path}: slice_count={count} queries_from_stats={queries_from_stats}"
        )

    values = [float(metric_match.group(i)) for i in range(1, 11)]
    ma_values = [float(ma_match.group(i)) for i in range(1, 5)]
    robust_values = [int(robust_match.group(i)) for i in range(1, 6)]
    match_values = [float(match_stats.group(i)) for i in range(1, 4)]
    runtime_values = [float(runtime_match.group(i)) for i in range(1, 4)]

    return ShardMetrics(
        path=path,
        start=start,
        end=end,
        count=count,
        total_queries=total_queries,
        recall1=values[0],
        recall5=values[1],
        recall10=values[2],
        map_score=values[3],
        sdm1=values[4],
        sdm3=values[5],
        sdm5=values[6],
        dis1=values[7],
        dis3=values[8],
        dis5=values[9],
        ma3=ma_values[0],
        ma5=ma_values[1],
        ma10=ma_values[2],
        ma20=ma_values[3],
        fallback=robust_values[0],
        worse_than_coarse=robust_values[1],
        identity_fallback=robust_values[2],
        out_of_bounds=robust_values[3],
        projection_invalid=robust_values[4],
        mean_retained=match_values[0],
        mean_inliers=match_values[1],
        mean_inlier_ratio=match_values[2],
        mean_vop_time=runtime_values[0],
        mean_matcher_time=runtime_values[1],
        mean_total_time=runtime_values[2],
    )


def _weighted_mean(shards, attr: str, total_count: int) -> float:
    return sum(getattr(shard, attr) * shard.count for shard in shards) / max(total_count, 1)


def _render_summary(shards):
    if not shards:
        raise ValueError("No shard logs matched.")

    total_queries = shards[0].total_queries
    for shard in shards:
        if shard.total_queries != total_queries:
            raise ValueError("Shard total query counts do not agree.")

    shards = sorted(shards, key=lambda item: item.start)
    covered = 0
    missing = []
    overlaps = []
    cursor = 0
    for shard in shards:
        if shard.start > cursor:
            missing.append((cursor, shard.start))
        if shard.start < cursor:
            overlaps.append((shard.start, min(cursor, shard.end)))
        cursor = max(cursor, shard.end)
        covered += shard.count
    if cursor < total_queries:
        missing.append((cursor, total_queries))

    merged_count = sum(shard.count for shard in shards)
    if merged_count != covered:
        raise ValueError(f"Covered query mismatch: merged_count={merged_count} covered={covered}")

    lines = []
    lines.append("# GTA Eval Shard Merge")
    lines.append("")
    lines.append(f"- shard_count: `{len(shards)}`")
    lines.append(f"- merged_queries: `{merged_count}` / `{total_queries}`")
    lines.append(f"- fully_covered: `{'yes' if (merged_count == total_queries and not missing and not overlaps) else 'no'}`")
    if missing:
        missing_text = ", ".join(f"[{start}, {end})" for start, end in missing)
        lines.append(f"- missing_ranges: `{missing_text}`")
    if overlaps:
        overlap_text = ", ".join(f"[{start}, {end})" for start, end in overlaps)
        lines.append(f"- overlap_ranges: `{overlap_text}`")
    lines.append("")
    lines.append("## Merged Metrics")
    lines.append(
        "- retrieval: "
        f"Recall@1={_weighted_mean(shards, 'recall1', merged_count):.4f}, "
        f"Recall@5={_weighted_mean(shards, 'recall5', merged_count):.4f}, "
        f"Recall@10={_weighted_mean(shards, 'recall10', merged_count):.4f}, "
        f"mAP={_weighted_mean(shards, 'map_score', merged_count):.4f}, "
        f"SDM@1={_weighted_mean(shards, 'sdm1', merged_count):.4f}, "
        f"SDM@3={_weighted_mean(shards, 'sdm3', merged_count):.4f}, "
        f"SDM@5={_weighted_mean(shards, 'sdm5', merged_count):.4f}"
    )
    lines.append(
        "- fine_localization: "
        f"Dis@1={_weighted_mean(shards, 'dis1', merged_count):.4f}, "
        f"Dis@3={_weighted_mean(shards, 'dis3', merged_count):.4f}, "
        f"Dis@5={_weighted_mean(shards, 'dis5', merged_count):.4f}"
    )
    lines.append(
        "- threshold_success: "
        f"MA@3={_weighted_mean(shards, 'ma3', merged_count):.4f}, "
        f"MA@5={_weighted_mean(shards, 'ma5', merged_count):.4f}, "
        f"MA@10={_weighted_mean(shards, 'ma10', merged_count):.4f}, "
        f"MA@20={_weighted_mean(shards, 'ma20', merged_count):.4f}"
    )
    fallback = sum(shard.fallback for shard in shards)
    worse = sum(shard.worse_than_coarse for shard in shards)
    identity = sum(shard.identity_fallback for shard in shards)
    oob = sum(shard.out_of_bounds for shard in shards)
    invalid = sum(shard.projection_invalid for shard in shards)
    lines.append(
        "- robustness: "
        f"fallback={fallback} ({100.0 * fallback / max(merged_count, 1):.2f}%), "
        f"worse-than-coarse={worse} ({100.0 * worse / max(merged_count, 1):.2f}%), "
        f"identity-H fallback={identity} ({100.0 * identity / max(merged_count, 1):.2f}%), "
        f"out-of-bounds={oob} ({100.0 * oob / max(merged_count, 1):.2f}%), "
        f"projection-invalid={invalid} ({100.0 * invalid / max(merged_count, 1):.2f}%)"
    )
    lines.append(
        "- match_stats: "
        f"mean_retained_matches={_weighted_mean(shards, 'mean_retained', merged_count):.4f}, "
        f"mean_inliers={_weighted_mean(shards, 'mean_inliers', merged_count):.4f}, "
        f"mean_inlier_ratio={_weighted_mean(shards, 'mean_inlier_ratio', merged_count):.4f}"
    )
    lines.append(
        "- runtime: "
        f"mean_vop_forward_time={_weighted_mean(shards, 'mean_vop_time', merged_count):.6f}s/query, "
        f"mean_matcher_time={_weighted_mean(shards, 'mean_matcher_time', merged_count):.6f}s/query, "
        f"mean_total_time={_weighted_mean(shards, 'mean_total_time', merged_count):.6f}s/query"
    )
    lines.append("")
    lines.append("## Shards")
    for shard in shards:
        lines.append(
            f"- `{os.path.basename(shard.path)}`: range=[{shard.start}, {shard.end}) count={shard.count} "
            f"Dis@1={shard.dis1:.4f} MA@20={shard.ma20:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Merge sharded GTA evaluator logs.")
    parser.add_argument("--log_glob", type=str, required=True, help="Glob pattern for shard app logs.")
    parser.add_argument("--output_path", type=str, default="", help="Optional markdown output path.")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.log_glob))
    shards = [_parse_log(path) for path in paths]
    summary = _render_summary(shards)
    print(summary)
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            handle.write(summary + "\n")


if __name__ == "__main__":
    main()
