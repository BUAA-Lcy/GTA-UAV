import argparse
import json
import math
import os
import time

import numpy as np
from geopy.distance import geodesic

from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
from game4loc.evaluate.visloc import _rotate_query_tensor
from game4loc.matcher.gim_dkm import GimDKM


USEFUL_DELTAS_M = (1.0, 3.0, 5.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate top-k localization strategies from a cached oracle/posterior file.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_pairs_meta_file", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--sate_img_dir", type=str, default="satellite")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--strategy", type=str, default="posterior", choices=("posterior", "random", "uniform", "oracle"))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def to_serializable(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def distribution_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def prefer_match_candidate(inliers, inlier_ratio, best_inliers, best_ratio):
    if inliers > best_inliers:
        return True
    if inliers == best_inliers and inlier_ratio > best_ratio:
        return True
    return False


def build_uniform_indices(num_angles: int, topk: int):
    if topk >= num_angles:
        return list(range(num_angles))
    raw = np.floor(np.arange(topk, dtype=np.float64) * float(num_angles) / float(topk)).astype(np.int64)
    unique = []
    seen = set()
    for idx in raw.tolist():
        if idx not in seen:
            unique.append(int(idx))
            seen.add(int(idx))
    if len(unique) < topk:
        for idx in range(num_angles):
            if idx not in seen:
                unique.append(int(idx))
                seen.add(int(idx))
            if len(unique) >= topk:
                break
    return unique[:topk]


def select_indices(strategy, topk, num_angles, posterior_probs, oracle_distances, rng):
    if strategy == "posterior":
        return list(np.argsort(np.asarray(posterior_probs, dtype=np.float64))[::-1][:topk])
    if strategy == "random":
        return [int(x) for x in rng.choice(num_angles, size=topk, replace=False).tolist()]
    if strategy == "uniform":
        return build_uniform_indices(num_angles, topk)
    if strategy == "oracle":
        order = np.argsort(np.asarray(oracle_distances, dtype=np.float64))
        return [int(x) for x in order[:topk].tolist()]
    raise ValueError(f"Unknown strategy: {strategy}")


def build_useful_mask(oracle_distances, best_distance, delta_m):
    threshold = float(best_distance) + float(delta_m)
    return [bool(math.isfinite(float(dist)) and float(dist) <= threshold) for dist in oracle_distances]


def circular_components(mask):
    n = len(mask)
    if n == 0:
        return 0, []
    if not any(mask):
        return 0, []
    if all(mask):
        return 1, [n]
    first_false = next(idx for idx, flag in enumerate(mask) if not flag)
    ordered = mask[first_false + 1 :] + mask[: first_false + 1]
    component_lengths = []
    current_len = 0
    for flag in ordered:
        if flag:
            current_len += 1
        elif current_len > 0:
            component_lengths.append(current_len)
            current_len = 0
    if current_len > 0:
        component_lengths.append(current_len)
    return len(component_lengths), component_lengths


def summarize_useful_structure(records, delta_key):
    sizes = [record["useful_sets"][delta_key]["size"] for record in records]
    components = [record["useful_sets"][delta_key]["components"] for record in records]
    widths = [record["useful_sets"][delta_key]["max_interval_width_deg"] for record in records]
    return {
        "size": distribution_stats(sizes),
        "components": distribution_stats(components),
        "max_interval_width_deg": distribution_stats(widths),
        "single_interval_ratio": float(np.mean([comp == 1 for comp in components])) if components else float("nan"),
        "multimodal_ratio": float(np.mean([comp > 1 for comp in components])) if components else float("nan"),
        "count_ge_2_ratio": float(np.mean([size >= 2 for size in sizes])) if sizes else float("nan"),
        "count_ge_3_ratio": float(np.mean([size >= 3 for size in sizes])) if sizes else float("nan"),
    }


def main():
    args = parse_args()
    device = args.device or "cuda"
    with open(args.cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    candidate_angles = [float(x) for x in cache["config"]["candidate_angles_deg"]]
    oracle_rotate_step = float(cache["config"]["oracle_rotate_step"])
    topk = max(1, min(int(args.topk), len(candidate_angles)))
    rng = np.random.default_rng(int(args.random_seed))

    val_transforms, _, _ = get_transforms((args.img_size, args.img_size), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    query_dataset = VisLocDatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.test_pairs_meta_file,
        view="drone",
        mode="pos",
        transforms=val_transforms,
        query_mode="D2S",
    )
    gallery_dataset = VisLocDatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.test_pairs_meta_file,
        view="sate",
        transforms=val_transforms,
        sate_img_dir=args.sate_img_dir,
        query_mode="D2S",
    )

    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        logger=None,
        sparse_save_final_vis=False,
    )

    eval_records = []
    for record in cache["records"]:
        query_index = int(record["query_index"])
        gallery_index = int(record["gallery_index"])
        query_sample = query_dataset[query_index]
        gallery_sample = gallery_dataset[gallery_index]
        query_loc = tuple(float(x) for x in record["query_loc_xy"])
        gallery_center_lon_lat = tuple(float(x) for x in record["gallery_center_lon_lat"])
        gallery_topleft_lon_lat = tuple(float(x) for x in record["gallery_topleft_lon_lat"])
        oracle_distances = [float(x) for x in record["oracle_distances_m"]]
        best_distance = float(np.min(np.asarray(oracle_distances, dtype=np.float64)))
        best_index = int(np.argmin(np.asarray(oracle_distances, dtype=np.float64)))

        selected_indices = select_indices(
            strategy=args.strategy,
            topk=topk,
            num_angles=len(candidate_angles),
            posterior_probs=record["posterior_probs"],
            oracle_distances=oracle_distances,
            rng=rng,
        )

        best_match_loc_lon_lat = None
        best_match_info = None
        best_match_inliers = -1
        best_match_ratio = -1.0
        best_match_angle = None
        match_time = 0.0
        for cand_index in selected_indices:
            cand_angle = float(candidate_angles[int(cand_index)])
            rotated_query = _rotate_query_tensor(query_sample, cand_angle)
            t_match = time.perf_counter()
            candidate_loc_lon_lat = matcher.est_center(
                gallery_sample,
                rotated_query,
                gallery_center_lon_lat,
                gallery_topleft_lon_lat,
                yaw0=None,
                yaw1=None,
                rotate=0.0,
                case_name=f"{record['query_name']}_{args.strategy}_top{topk}_{cand_angle:.1f}",
            )
            match_time += time.perf_counter() - t_match
            candidate_match_info = matcher.get_last_match_info() or {}
            cand_kept = int(candidate_match_info.get("n_kept", 0))
            cand_inliers = int(candidate_match_info.get("inliers", 0))
            cand_ratio = float(cand_inliers) / float(max(cand_kept, 1))
            if prefer_match_candidate(cand_inliers, cand_ratio, best_match_inliers, best_match_ratio):
                best_match_loc_lon_lat = candidate_loc_lon_lat
                best_match_info = dict(candidate_match_info)
                best_match_inliers = cand_inliers
                best_match_ratio = cand_ratio
                best_match_angle = cand_angle

        final_lat_lon = (best_match_loc_lon_lat[1], best_match_loc_lon_lat[0])
        final_error_m = float(geodesic(query_loc, final_lat_lon).meters)
        n_kept = int((best_match_info or {}).get("n_kept", 0))
        inliers = int((best_match_info or {}).get("inliers", 0))
        inlier_ratio = float(inliers) / float(max(n_kept, 1))

        useful_sets = {}
        for delta in USEFUL_DELTAS_M:
            key = f"{int(delta)}m"
            useful_mask = build_useful_mask(oracle_distances, best_distance, delta)
            useful_indices = [idx for idx, flag in enumerate(useful_mask) if flag]
            comp_count, comp_lengths = circular_components(useful_mask)
            selected_hits = [idx for idx in selected_indices if idx in useful_indices]
            useful_sets[key] = {
                "size": int(len(useful_indices)),
                "components": int(comp_count),
                "max_interval_width_deg": float(max(comp_lengths) * oracle_rotate_step) if comp_lengths else 0.0,
                "selected_hits_any": bool(len(selected_hits) > 0),
                "selected_set_recall": float(len(selected_hits)) / float(max(len(useful_indices), 1)),
            }

        eval_records.append(
            {
                "query_index": query_index,
                "query_name": record["query_name"],
                "selected_indices": [int(x) for x in selected_indices],
                "selected_angles_deg": [float(candidate_angles[idx]) for idx in selected_indices],
                "oracle_best_in_topk": bool(best_index in selected_indices),
                "final_error_m": float(final_error_m),
                "dis3_m": float(record["coarse_dis3_m"]),
                "dis5_m": float(record["coarse_dis5_m"]),
                "n_kept": int(n_kept),
                "inliers": int(inliers),
                "inlier_ratio": float(inlier_ratio),
                "num_hypotheses": int(len(selected_indices)),
                "vop_time_s": float(record["vop_time_s"]) if args.strategy == "posterior" else 0.0,
                "match_time_s": float(match_time),
                "posterior_top_prob": float(record["posterior_top_prob"]),
                "posterior_entropy": float(record["posterior_entropy"]),
                "posterior_concentration": float(record["posterior_concentration"]),
                "useful_sets": useful_sets,
            }
        )

    dis1 = [item["final_error_m"] for item in eval_records]
    dis3 = [item["dis3_m"] for item in eval_records]
    dis5 = [item["dis5_m"] for item in eval_records]
    kept = [item["n_kept"] for item in eval_records]
    inliers = [item["inliers"] for item in eval_records]
    ratios = [item["inlier_ratio"] for item in eval_records]
    vop_times = [item["vop_time_s"] for item in eval_records]
    match_times = [item["match_time_s"] for item in eval_records]

    useful_summary = {}
    for delta in USEFUL_DELTAS_M:
        key = f"{int(delta)}m"
        any_hits = [float(item["useful_sets"][key]["selected_hits_any"]) for item in eval_records]
        set_recall = [float(item["useful_sets"][key]["selected_set_recall"]) for item in eval_records]
        covered = [item for item in eval_records if item["useful_sets"][key]["selected_hits_any"]]
        missed = [item for item in eval_records if not item["useful_sets"][key]["selected_hits_any"]]
        useful_summary[key] = {
            "any_hit_rate": float(np.mean(any_hits)),
            "set_recall_mean": float(np.mean(set_recall)),
            "covered_error_m": distribution_stats([item["final_error_m"] for item in covered]),
            "missed_error_m": distribution_stats([item["final_error_m"] for item in missed]),
            "structure": summarize_useful_structure(eval_records, key),
        }

    summary = {
        "strategy": args.strategy,
        "topk": int(topk),
        "uniform_indices": build_uniform_indices(len(candidate_angles), topk),
        "uniform_angles_deg": [float(candidate_angles[idx]) for idx in build_uniform_indices(len(candidate_angles), topk)],
        "query_count": int(len(eval_records)),
        "metrics": {
            "Dis@1_m": float(np.mean(dis1)),
            "Dis@3_m": float(np.mean(dis3)),
            "Dis@5_m": float(np.mean(dis5)),
        },
        "runtime_s_per_query": {
            "vop_forward": float(np.mean(vop_times)),
            "matcher": float(np.mean(match_times)),
            "total": float(np.mean(np.asarray(vop_times) + np.asarray(match_times))),
        },
        "topk_oracle_best_coverage": float(np.mean([float(item["oracle_best_in_topk"]) for item in eval_records])),
        "posterior_stats": {
            "mean_top_prob": float(np.mean([item["posterior_top_prob"] for item in eval_records])),
            "mean_entropy": float(np.mean([item["posterior_entropy"] for item in eval_records])),
            "mean_concentration": float(np.mean([item["posterior_concentration"] for item in eval_records])),
        },
        "match_stats": {
            "mean_hypotheses": float(np.mean([item["num_hypotheses"] for item in eval_records])),
            "retained_matches": float(np.mean(kept)),
            "mean_inliers": float(np.mean(inliers)),
            "mean_inlier_ratio": float(np.mean(ratios)),
        },
        "useful_coverage": useful_summary,
    }

    output = {
        "config": {
            "cache_path": args.cache_path,
            "strategy": args.strategy,
            "topk": int(topk),
            "random_seed": int(args.random_seed),
        },
        "summary": summary,
        "records": eval_records,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)

    print(f"Saved cached evaluation to {args.output_path}")
    print(json.dumps(to_serializable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
