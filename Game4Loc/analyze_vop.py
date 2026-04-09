import argparse
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from geopy.distance import geodesic
from torch.utils.data import DataLoader

from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
from game4loc.evaluate.visloc import _rotate_query_tensor, predict, project_match_center_from_h
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import compute_entropy, load_vop_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze VOP teacher separability and eval-time orientation diagnostics.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_pairs_meta_file", type=str, required=True)
    parser.add_argument("--teacher_cache", type=str, required=True)
    parser.add_argument("--orientation_checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--checkpoint_start", type=str, default="")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sate_img_dir", type=str, default="satellite")
    parser.add_argument("--query_limit", type=int, default=0)
    parser.add_argument("--temperature_m", type=float, default=-1.0)
    parser.add_argument("--localization_topk", type=int, default=1)
    parser.add_argument("--useful_delta_m", type=float, default=5.0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def soft_distribution_from_distances(distances_m, temperature_m: float):
    distances = torch.tensor(distances_m, dtype=torch.float32)
    valid_mask = torch.isfinite(distances)
    logits = torch.full_like(distances, fill_value=-40.0)
    logits[valid_mask] = -distances[valid_mask] / max(float(temperature_m), 1e-6)
    probs = torch.softmax(logits, dim=0)
    return probs


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
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def fixed_bin_counts(values, edges):
    arr = np.asarray(values, dtype=np.float64)
    counts = {}
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= edges[-1]:
            mask = (arr >= left) & (arr <= right)
            label = f"[{left:.2f}, {right:.2f}]"
        else:
            mask = (arr >= left) & (arr < right)
            label = f"[{left:.2f}, {right:.2f})"
        counts[label] = int(mask.sum())
    return counts


def summarize_teacher_cache(teacher_cache):
    records = list(teacher_cache["records"])
    entropies = []
    best_probs = []
    prob_margins = []
    distance_gaps = []
    for record in records:
        probs = torch.tensor(record["target_probs"], dtype=torch.float32)
        entropy = float(compute_entropy(probs.unsqueeze(0))[0].item())
        sorted_probs = torch.sort(probs, descending=True).values
        entropies.append(entropy)
        best_probs.append(float(sorted_probs[0].item()))
        prob_margins.append(float((sorted_probs[0] - sorted_probs[1]).item()))
        gap = float(record.get("distance_gap_m", float("inf")))
        if math.isfinite(gap):
            distance_gaps.append(gap)

    return {
        "count": int(len(records)),
        "entropy": distribution_stats(entropies),
        "best_prob": distribution_stats(best_probs),
        "top1_top2_margin": distribution_stats(prob_margins),
        "distance_gap_m": distribution_stats(distance_gaps),
        "distance_gap_counts": {
            "ge_1m": int(sum(gap >= 1.0 for gap in distance_gaps)),
            "ge_3m": int(sum(gap >= 3.0 for gap in distance_gaps)),
            "ge_5m": int(sum(gap >= 5.0 for gap in distance_gaps)),
            "ge_10m": int(sum(gap >= 10.0 for gap in distance_gaps)),
        },
        "entropy_bins": fixed_bin_counts(entropies, [0.0, 0.4, 0.6, 0.7, 0.8, 1.0]),
    }


def safe_corr(x_values, y_values):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return float("nan")
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_bucket_summary(records):
    if not records:
        return {"count": 0}
    final_errors = np.asarray([float(item["final_error_m"]) for item in records], dtype=np.float64)
    top1_hits = np.asarray([float(item["top1_hit"]) for item in records], dtype=np.float64)
    top3_hits = np.asarray([float(item["top3_hit"]) for item in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "mean_dis1_m": float(final_errors.mean()),
        "median_dis1_m": float(np.median(final_errors)),
        "top1_hit_rate": float(top1_hits.mean()),
        "top3_coverage": float(top3_hits.mean()),
    }


def split_by_median(records, key):
    values = np.asarray([float(item[key]) for item in records], dtype=np.float64)
    threshold = float(np.median(values))
    low = [item for item in records if float(item[key]) <= threshold]
    high = [item for item in records if float(item[key]) > threshold]
    return threshold, low, high


def split_by_tertiles(records, key):
    values = np.asarray([float(item[key]) for item in records], dtype=np.float64)
    q1 = float(np.quantile(values, 1.0 / 3.0))
    q2 = float(np.quantile(values, 2.0 / 3.0))
    low = [item for item in records if float(item[key]) <= q1]
    mid = [item for item in records if q1 < float(item[key]) <= q2]
    high = [item for item in records if float(item[key]) > q2]
    return (q1, q2), low, mid, high


def prefer_match_candidate(inliers, inlier_ratio, best_inliers, best_ratio):
    if inliers > best_inliers:
        return True
    if inliers == best_inliers and inlier_ratio > best_ratio:
        return True
    return False


def build_eval_diagnostics(records):
    top1_hits = np.asarray([float(item["top1_hit"]) for item in records], dtype=np.float64)
    top3_hits = np.asarray([float(item["top3_hit"]) for item in records], dtype=np.float64)
    near_1 = np.asarray([float(item["within_1m"]) for item in records], dtype=np.float64)
    near_3 = np.asarray([float(item["within_3m"]) for item in records], dtype=np.float64)
    near_5 = np.asarray([float(item["within_5m"]) for item in records], dtype=np.float64)
    topk_near_1 = np.asarray([float(item.get("topk_within_1m", False)) for item in records], dtype=np.float64)
    topk_near_3 = np.asarray([float(item.get("topk_within_3m", False)) for item in records], dtype=np.float64)
    topk_near_5 = np.asarray([float(item.get("topk_within_5m", False)) for item in records], dtype=np.float64)
    final_errors = np.asarray([float(item["final_error_m"]) for item in records], dtype=np.float64)
    top_probs = np.asarray([float(item["posterior_top_prob"]) for item in records], dtype=np.float64)
    entropies = np.asarray([float(item["posterior_entropy"]) for item in records], dtype=np.float64)

    hit_group = [item for item in records if item["top1_hit"]]
    miss_group = [item for item in records if not item["top1_hit"]]
    entropy_thresholds, entropy_low, entropy_mid, entropy_high = split_by_tertiles(records, "oracle_entropy")
    gap_threshold, gap_small, gap_big = split_by_median(records, "oracle_distance_gap_m")
    kept_threshold, kept_few, kept_many = split_by_median(records, "n_kept")
    ratio_threshold, ratio_low, ratio_high = split_by_median(records, "inlier_ratio")
    prob_threshold, prob_low, prob_high = split_by_median(records, "posterior_top_prob")

    return {
        "count": int(len(records)),
        "posterior_top1_hit_rate": float(top1_hits.mean()),
        "posterior_top3_coverage": float(top3_hits.mean()),
        "posterior_top1_within_best_plus": {
            "1m": float(near_1.mean()),
            "3m": float(near_3.mean()),
            "5m": float(near_5.mean()),
        },
        "posterior_topk_within_best_plus": {
            "1m": float(topk_near_1.mean()),
            "3m": float(topk_near_3.mean()),
            "5m": float(topk_near_5.mean()),
        },
        "final_error_m": distribution_stats(final_errors),
        "hit_vs_miss_final_error_m": {
            "hit": distribution_stats([item["final_error_m"] for item in hit_group]),
            "miss": distribution_stats([item["final_error_m"] for item in miss_group]),
        },
        "posterior_hit_correlation": {
            "top_prob_vs_hit": safe_corr(top_probs, top1_hits),
            "entropy_vs_hit": safe_corr(entropies, top1_hits),
        },
        "posterior_hit_group_stats": {
            "hit": {
                "count": int(len(hit_group)),
                "mean_top_prob": float(np.mean([item["posterior_top_prob"] for item in hit_group])) if hit_group else float("nan"),
                "mean_entropy": float(np.mean([item["posterior_entropy"] for item in hit_group])) if hit_group else float("nan"),
            },
            "miss": {
                "count": int(len(miss_group)),
                "mean_top_prob": float(np.mean([item["posterior_top_prob"] for item in miss_group])) if miss_group else float("nan"),
                "mean_entropy": float(np.mean([item["posterior_entropy"] for item in miss_group])) if miss_group else float("nan"),
            },
        },
        "bucketed_failure_modes": {
            "oracle_entropy_low_mid_high": {
                "thresholds": {"q33": entropy_thresholds[0], "q67": entropy_thresholds[1]},
                "low": build_bucket_summary(entropy_low),
                "mid": build_bucket_summary(entropy_mid),
                "high": build_bucket_summary(entropy_high),
            },
            "oracle_distance_gap_small_big": {
                "threshold_median_m": gap_threshold,
                "small": build_bucket_summary(gap_small),
                "big": build_bucket_summary(gap_big),
            },
            "retained_matches_few_many": {
                "threshold_median": kept_threshold,
                "few": build_bucket_summary(kept_few),
                "many": build_bucket_summary(kept_many),
            },
            "inlier_ratio_low_high": {
                "threshold_median": ratio_threshold,
                "low": build_bucket_summary(ratio_low),
                "high": build_bucket_summary(ratio_high),
            },
            "vop_top_prob_low_high": {
                "threshold_median": prob_threshold,
                "low": build_bucket_summary(prob_low),
                "high": build_bucket_summary(prob_high),
            },
        },
        "runtime_s_per_query": {
            "vop_forward": float(np.mean([item["vop_time_s"] for item in records])),
            "prior_single_matcher": float(np.mean([item["prior_single_match_time_s"] for item in records])),
            "prior_single_total": float(np.mean([item["vop_time_s"] + item["prior_single_match_time_s"] for item in records])),
            "oracle_rotate10_matcher": float(np.mean([item["oracle_match_time_s"] for item in records])),
        },
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    teacher_cache = torch.load(args.teacher_cache, map_location="cpu")
    teacher_stats = summarize_teacher_cache(teacher_cache)
    temperature_m = float(args.temperature_m if args.temperature_m > 0 else teacher_cache.get("temperature_m", 25.0))

    model = DesModel(args.model, pretrained=True, img_size=args.img_size, share_weights=True)
    if args.checkpoint_start:
        state_dict = torch.load(args.checkpoint_start, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    data_config = model.get_config()
    img_size = (args.img_size, args.img_size)
    val_transforms, _, _ = get_transforms(img_size, mean=data_config["mean"], std=data_config["std"])

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

    if args.query_limit > 0:
        query_indices = list(range(min(args.query_limit, len(query_dataset))))
    else:
        query_indices = list(range(len(query_dataset)))

    query_loader = DataLoader(
        [query_dataset[idx] for idx in query_indices],
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    predict_cfg = SimpleNamespace(device=device, verbose=True, normalize_features=True)
    with torch.no_grad():
        query_features = predict(predict_cfg, model, query_loader)
        gallery_features = predict(predict_cfg, model, gallery_loader)
        scores = query_features @ gallery_features.T
        top1_indices = torch.argmax(scores, dim=1).detach().cpu().tolist()

    orientation_model = load_vop_checkpoint(args.orientation_checkpoint, device=device)
    candidate_angles = getattr(orientation_model, "candidate_angles_deg", None)
    if not candidate_angles:
        raise RuntimeError(f"Orientation checkpoint {args.orientation_checkpoint} does not contain candidate angles.")
    oracle_rotate_step = abs(float(candidate_angles[1] - candidate_angles[0])) if len(candidate_angles) > 1 else 0.0
    localization_topk = max(1, int(args.localization_topk))

    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        logger=None,
        sparse_save_final_vis=False,
    )

    eval_records = []
    for local_idx, query_index in enumerate(query_indices):
        top1_index = int(top1_indices[local_idx])
        query_sample = query_dataset[query_index]
        gallery_sample = gallery_dataset[top1_index]
        query_name = query_dataset.images_name[query_index]
        gallery_name = gallery_dataset.images_name[top1_index]
        query_loc = query_dataset.images_center_loc_xy[query_index]
        gallery_center_lat, gallery_center_lon = gallery_dataset.images_center_loc_xy[top1_index]
        gallery_topleft_lat, gallery_topleft_lon = gallery_dataset.images_topleft_loc_xy[top1_index]
        gallery_center_lon_lat = (gallery_center_lon, gallery_center_lat)
        gallery_topleft_lon_lat = (gallery_topleft_lon, gallery_topleft_lat)

        t_oracle = time.perf_counter()
        _ = matcher.est_center(
            gallery_sample,
            query_sample,
            gallery_center_lon_lat,
            gallery_topleft_lon_lat,
            yaw0=None,
            yaw1=None,
            rotate=oracle_rotate_step,
            case_name=f"{query_name}_oracle",
        )
        oracle_match_time = time.perf_counter() - t_oracle
        angle_results = [item for item in matcher.get_last_angle_results() if int(item.get("phase", 0)) == 1]
        angle_results = sorted(angle_results, key=lambda item: float(item.get("search_angle", 0.0)))
        oracle_angles = [float(item.get("rot_angle", 0.0)) for item in angle_results]
        if oracle_angles != list(candidate_angles):
            raise ValueError(f"Unexpected oracle angles for {query_name}: {oracle_angles} vs {candidate_angles}")

        oracle_distances = []
        for angle_result in angle_results:
            loc_lon_lat = project_match_center_from_h(
                angle_result.get("homography"),
                gallery_sample,
                gallery_center_lon_lat,
                gallery_topleft_lon_lat,
            )
            if loc_lon_lat is None:
                oracle_distances.append(float("inf"))
                continue
            pred_lat_lon = (float(loc_lon_lat[1]), float(loc_lon_lat[0]))
            oracle_distances.append(float(geodesic(query_loc, pred_lat_lon).meters))

        oracle_dist_tensor = torch.tensor(oracle_distances, dtype=torch.float32)
        finite_mask = torch.isfinite(oracle_dist_tensor)
        if not finite_mask.any():
            continue
        sorted_finite = torch.sort(oracle_dist_tensor[finite_mask]).values
        best_distance = float(sorted_finite[0].item())
        second_distance = float(sorted_finite[1].item()) if sorted_finite.numel() > 1 else float("inf")
        best_index = int(torch.argmin(torch.nan_to_num(oracle_dist_tensor, nan=float("inf"), posinf=float("inf"))).item())
        teacher_probs = soft_distribution_from_distances(oracle_distances, temperature_m=temperature_m)
        teacher_entropy = float(compute_entropy(teacher_probs.unsqueeze(0))[0].item())

        t_vop = time.perf_counter()
        posterior = orientation_model.predict_posterior(
            retrieval_model=model,
            gallery_img=gallery_sample,
            query_img=query_sample,
            candidate_angles_deg=candidate_angles,
            device=device,
            gallery_branch="img2",
            query_branch="img1",
        )
        vop_time = time.perf_counter() - t_vop

        top_index = int(posterior["top_index"])
        sorted_indices = list(np.argsort(np.asarray(posterior["probs"], dtype=np.float64))[::-1])
        top3_indices = sorted_indices[:3]
        topk_indices = sorted_indices[:localization_topk]
        predicted_oracle_distance = float(oracle_distances[top_index])
        topk_oracle_distances = [float(oracle_distances[idx]) for idx in topk_indices]

        prior_match_time = 0.0
        best_match_loc_lon_lat = None
        best_match_info = None
        best_match_inliers = -1
        best_match_ratio = -1.0
        best_match_angle = None
        for cand_index in topk_indices:
            rotated_query = _rotate_query_tensor(query_sample, candidate_angles[cand_index])
            t_prior = time.perf_counter()
            match_loc_lon_lat = matcher.est_center(
                gallery_sample,
                rotated_query,
                gallery_center_lon_lat,
                gallery_topleft_lon_lat,
                yaw0=None,
                yaw1=None,
                rotate=0.0,
                case_name=f"{query_name}_prior_top{localization_topk}",
            )
            prior_match_time += time.perf_counter() - t_prior
            match_info = matcher.get_last_match_info() or {}
            n_kept = int(match_info.get("n_kept", 0))
            inliers = int(match_info.get("inliers", 0))
            inlier_ratio = float(inliers) / float(max(n_kept, 1))
            if prefer_match_candidate(inliers, inlier_ratio, best_match_inliers, best_match_ratio):
                best_match_loc_lon_lat = match_loc_lon_lat
                best_match_info = match_info
                best_match_inliers = inliers
                best_match_ratio = inlier_ratio
                best_match_angle = float(candidate_angles[cand_index])

        match_loc_lon_lat = best_match_loc_lon_lat
        match_info = best_match_info or {}
        final_lat_lon = (match_loc_lon_lat[1], match_loc_lon_lat[0])
        final_error_m = float(geodesic(query_loc, final_lat_lon).meters)
        n_kept = int(match_info.get("n_kept", 0))
        inliers = int(match_info.get("inliers", 0))
        inlier_ratio = float(inliers) / float(max(n_kept, 1))

        eval_records.append(
            {
                "query_index": int(query_index),
                "query_name": query_name,
                "gallery_index": top1_index,
                "gallery_name": gallery_name,
                "oracle_best_index": best_index,
                "oracle_best_angle_deg": float(candidate_angles[best_index]),
                "oracle_best_distance_m": best_distance,
                "oracle_second_distance_m": second_distance,
                "oracle_distance_gap_m": float(second_distance - best_distance) if math.isfinite(second_distance) else float("inf"),
                "oracle_entropy": teacher_entropy,
                "oracle_distances_m": [float(x) for x in oracle_distances],
                "posterior_top_index": top_index,
                "posterior_top_angle_deg": float(posterior["top_angle_deg"]),
                "posterior_top_prob": float(posterior["top_prob"]),
                "posterior_entropy": float(posterior["entropy"]),
                "posterior_concentration": float(posterior["concentration"]),
                "posterior_top3_indices": [int(idx) for idx in top3_indices],
                "posterior_topk_indices": [int(idx) for idx in topk_indices],
                "top1_hit": bool(top_index == best_index),
                "top3_hit": bool(best_index in top3_indices),
                "within_1m": bool(math.isfinite(predicted_oracle_distance) and predicted_oracle_distance <= best_distance + 1.0),
                "within_3m": bool(math.isfinite(predicted_oracle_distance) and predicted_oracle_distance <= best_distance + 3.0),
                "within_5m": bool(math.isfinite(predicted_oracle_distance) and predicted_oracle_distance <= best_distance + 5.0),
                "topk_within_1m": bool(any(math.isfinite(x) and x <= best_distance + 1.0 for x in topk_oracle_distances)),
                "topk_within_3m": bool(any(math.isfinite(x) and x <= best_distance + 3.0 for x in topk_oracle_distances)),
                "topk_within_5m": bool(any(math.isfinite(x) and x <= best_distance + 5.0 for x in topk_oracle_distances)),
                "predicted_oracle_distance_m": predicted_oracle_distance,
                "final_error_m": final_error_m,
                "selected_match_angle_deg": best_match_angle,
                "n_kept": n_kept,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "vop_time_s": float(vop_time),
                "prior_single_match_time_s": float(prior_match_time),
                "oracle_match_time_s": float(oracle_match_time),
            }
        )

    output = {
        "teacher_stats": teacher_stats,
        "eval_diagnostics": build_eval_diagnostics(eval_records),
        "eval_records": eval_records,
        "config": {
            "data_root": args.data_root,
            "test_pairs_meta_file": args.test_pairs_meta_file,
            "teacher_cache": args.teacher_cache,
            "orientation_checkpoint": args.orientation_checkpoint,
            "checkpoint_start": args.checkpoint_start,
            "temperature_m": temperature_m,
            "oracle_rotate_step": oracle_rotate_step,
            "localization_topk": localization_topk,
            "useful_delta_m": float(args.useful_delta_m),
        },
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)

    print(f"Saved diagnostics to {args.output_path}")
    print(json.dumps(to_serializable(output["teacher_stats"]), ensure_ascii=False, indent=2))
    print(json.dumps(to_serializable(output["eval_diagnostics"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
