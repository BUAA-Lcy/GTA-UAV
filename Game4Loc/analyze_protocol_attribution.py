import argparse
import json
import math
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from geopy.distance import geodesic
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run old-vs-expanded attribution analysis on existing checkpoints and optional VOP cache."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_pairs_meta_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--checkpoint_start", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default="")
    parser.add_argument(
        "--methods",
        type=str,
        default="retrieval_only,no_rotate,rotate90,prior_topk2,prior_topk4",
        help="Comma-separated methods from: retrieval_only,no_rotate,rotate90,prior_topkK",
    )
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sate_img_dir", type=str, default="satellite")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output_path", type=str, required=True)
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
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def parse_methods(methods_raw):
    methods = []
    for item in str(methods_raw).split(","):
        item = item.strip()
        if not item:
            continue
        methods.append(item)
    if not methods:
        raise ValueError("No methods requested.")
    return methods


def build_query_meta_map(data_root, pairs_meta_file):
    with open(os.path.join(data_root, pairs_meta_file), "r", encoding="utf-8") as f:
        meta = json.load(f)
    output = {}
    for item in meta:
        query_name = item["drone_img_name"]
        output[query_name] = {
            "strict_pos_count": int(len(item.get("pair_pos_sate_img_list") or [])),
            "semipos_count": int(len(item.get("pair_pos_semipos_sate_img_list") or [])),
            "drone_loc_lat_lon": item.get("drone_loc_lat_lon"),
            "drone_metadata_raw": item.get("drone_metadata") or {},
        }
    return output


def build_cache_record_map(cache_path):
    if not str(cache_path).strip():
        return None, None
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    record_map = {record["query_name"]: record for record in cache["records"]}
    candidate_angles = [float(x) for x in cache["config"].get("candidate_angles_deg") or []]
    return record_map, candidate_angles


def prefer_match_candidate(inliers, inlier_ratio, best_inliers, best_ratio):
    if inliers > best_inliers:
        return True
    if inliers == best_inliers and inlier_ratio > best_ratio:
        return True
    return False


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def method_sort_key(method_name):
    if method_name == "retrieval_only":
        return (0, 0)
    if method_name == "no_rotate":
        return (1, 0)
    if method_name == "rotate90":
        return (2, 0)
    if method_name.startswith("prior_topk"):
        return (3, int(method_name.replace("prior_topk", "")))
    return (99, 0)


def summarize_method(query_records, method_name, gallery_count):
    method_records = [record["methods"][method_name] for record in query_records if method_name in record["methods"]]
    output = {
        "query_count": int(len(method_records)),
        "gallery_count": int(gallery_count),
    }
    if not method_records:
        return output

    final_errors = np.asarray([record["final_error_m"] for record in method_records], dtype=np.float64)
    dis3 = np.asarray([record["dis3_m"] for record in method_records], dtype=np.float64)
    dis5 = np.asarray([record["dis5_m"] for record in method_records], dtype=np.float64)

    output["metrics"] = {
        "Dis@1_m": float(final_errors.mean()),
        "Dis@3_m": float(dis3.mean()),
        "Dis@5_m": float(dis5.mean()),
    }

    runtimes = np.asarray([record.get("runtime_total_s", 0.0) for record in method_records], dtype=np.float64)
    output["runtime_s_per_query"] = {
        "vop_forward": float(np.mean([record.get("runtime_vop_s", 0.0) for record in method_records])),
        "matcher": float(np.mean([record.get("runtime_matcher_s", 0.0) for record in method_records])),
        "total": float(runtimes.mean()),
    }

    output["stats"] = {
        "fallback_count": int(sum(bool(record.get("fallback_to_center", False)) for record in method_records)),
        "fallback_ratio_pct": float(np.mean([bool(record.get("fallback_to_center", False)) for record in method_records]) * 100.0),
        "worse_than_coarse_count": int(sum(bool(record.get("worse_than_coarse", False)) for record in method_records)),
        "worse_than_coarse_ratio_pct": float(np.mean([bool(record.get("worse_than_coarse", False)) for record in method_records]) * 100.0),
        "out_of_bounds_count": int(sum(bool(record.get("out_of_bounds", False)) for record in method_records)),
        "projection_invalid_count": int(sum(bool(record.get("projection_invalid", False)) for record in method_records)),
        "identity_h_fallback_count": int(sum(bool(record.get("identity_h_fallback", False)) for record in method_records)),
        "mean_hypotheses": float(np.mean([record.get("hypotheses_evaluated", 1) for record in method_records])),
    }

    retained = [safe_float(record.get("retained_matches")) for record in method_records]
    inliers = [safe_float(record.get("inliers")) for record in method_records]
    ratios = [safe_float(record.get("inlier_ratio")) for record in method_records]
    retained = [v for v in retained if v is not None]
    inliers = [v for v in inliers if v is not None]
    ratios = [v for v in ratios if v is not None]
    output["match_stats"] = {
        "retained_matches": None if not retained else float(np.mean(retained)),
        "mean_inliers": None if not inliers else float(np.mean(inliers)),
        "mean_inlier_ratio": None if not ratios else float(np.mean(ratios)),
    }

    return output


def summarize_scenes(query_records, method_names):
    scene_buckets = defaultdict(list)
    for record in query_records:
        scene_buckets[record["scene"]].append(record)

    scene_summary = {}
    for scene_name, records in sorted(scene_buckets.items()):
        retrieval_ap = [item["retrieval"]["ap"] for item in records if item["retrieval"]["ap"] is not None]
        scene_item = {
            "query_count": int(len(records)),
            "retrieval": {
                "Recall@1_pct": float(np.mean([item["retrieval"]["recall1"] for item in records]) * 100.0),
                "AP_pct": None if not retrieval_ap else float(np.mean(retrieval_ap) * 100.0),
                "Dis@1_m": float(np.mean([item["retrieval"]["coarse_dis1_m"] for item in records])),
            },
            "methods": {},
        }
        for method_name in method_names:
            method_records = [item["methods"][method_name] for item in records if method_name in item["methods"]]
            if not method_records:
                continue
            scene_item["methods"][method_name] = {
                "Dis@1_m": float(np.mean([item["final_error_m"] for item in method_records])),
                "fallback_ratio_pct": float(np.mean([bool(item.get("fallback_to_center", False)) for item in method_records]) * 100.0),
                "worse_than_coarse_ratio_pct": float(np.mean([bool(item.get("worse_than_coarse", False)) for item in method_records]) * 100.0),
            }
        scene_summary[scene_name] = scene_item
    return scene_summary


def run_single_match(matcher, query_sample, gallery_sample, gallery_center_lon_lat, gallery_topleft_lon_lat, query_loc, coarse_dis3_m, coarse_dis5_m, rotate_step):
    t_match = time.perf_counter()
    loc_lon_lat = matcher.est_center(
        gallery_sample,
        query_sample,
        gallery_center_lon_lat,
        gallery_topleft_lon_lat,
        yaw0=None,
        yaw1=None,
        rotate=rotate_step,
        case_name=None,
    )
    match_time = time.perf_counter() - t_match
    match_info = dict(matcher.get_last_match_info() or {})
    final_lat_lon = (float(loc_lon_lat[1]), float(loc_lon_lat[0]))
    final_error_m = float(geodesic(query_loc, final_lat_lon).meters)
    retained_matches = float(match_info.get("n_kept", 0))
    inliers = float(match_info.get("inliers", 0))
    inlier_ratio = float(match_info.get("inlier_ratio", 0.0))
    if retained_matches <= 0 and inliers > 0:
        retained_matches = inliers
    if retained_matches > 0 and inlier_ratio <= 0:
        inlier_ratio = float(inliers) / float(max(retained_matches, 1.0))
    return {
        "final_error_m": float(final_error_m),
        "dis3_m": float(coarse_dis3_m),
        "dis5_m": float(coarse_dis5_m),
        "runtime_vop_s": 0.0,
        "runtime_matcher_s": float(match_time),
        "runtime_total_s": float(match_time),
        "hypotheses_evaluated": int(max(1, len(matcher.get_last_angle_results() or []))),
        "retained_matches": float(retained_matches),
        "inliers": float(inliers),
        "inlier_ratio": float(inlier_ratio),
        "fallback_to_center": bool(match_info.get("fallback_to_center", False)),
        "identity_h_fallback": bool(match_info.get("identity_h_fallback", False)),
        "out_of_bounds": bool(match_info.get("out_of_bounds", False)),
        "projection_invalid": bool(match_info.get("projection_invalid", False)),
        "fallback_reason": match_info.get("fallback_reason"),
        "selected_angle_deg": float(match_info.get("rot_angle", 0.0)),
        "selected_rank": 1,
    }


def run_prior_topk(
    matcher,
    rotate_query_tensor,
    query_sample,
    gallery_sample,
    gallery_center_lon_lat,
    gallery_topleft_lon_lat,
    query_loc,
    coarse_dis3_m,
    coarse_dis5_m,
    cache_record,
    candidate_angles,
    topk,
):
    posterior_probs = np.asarray(cache_record["posterior_probs"], dtype=np.float64)
    sorted_indices = [int(x) for x in np.argsort(posterior_probs)[::-1][:topk].tolist()]
    best_match_loc_lon_lat = None
    best_match_info = None
    best_match_inliers = -1
    best_match_ratio = -1.0
    best_rank = None
    best_angle = None
    total_match_time = 0.0
    candidate_infos = []

    for rank_j, cand_index in enumerate(sorted_indices):
        cand_angle = float(candidate_angles[cand_index])
        rotated_query = rotate_query_tensor(query_sample, cand_angle)
        t_match = time.perf_counter()
        candidate_loc_lon_lat = matcher.est_center(
            gallery_sample,
            rotated_query,
            gallery_center_lon_lat,
            gallery_topleft_lon_lat,
            yaw0=None,
            yaw1=None,
            rotate=0.0,
            case_name=None,
        )
        total_match_time += time.perf_counter() - t_match
        candidate_match_info = dict(matcher.get_last_match_info() or {})
        cand_kept = float(candidate_match_info.get("n_kept", 0))
        cand_inliers = float(candidate_match_info.get("inliers", 0))
        cand_ratio = float(candidate_match_info.get("inlier_ratio", 0.0))
        if cand_kept <= 0 and cand_inliers > 0:
            cand_kept = cand_inliers
        if cand_kept > 0 and cand_ratio <= 0:
            cand_ratio = float(cand_inliers) / float(max(cand_kept, 1.0))
        cand_lat_lon = (float(candidate_loc_lon_lat[1]), float(candidate_loc_lon_lat[0]))
        cand_error_m = float(geodesic(query_loc, cand_lat_lon).meters)
        candidate_infos.append(
            {
                "candidate_rank": int(rank_j + 1),
                "candidate_index": int(cand_index),
                "candidate_angle_deg": float(cand_angle),
                "candidate_prob": float(posterior_probs[cand_index]),
                "candidate_error_m": float(cand_error_m),
                "retained_matches": float(cand_kept),
                "inliers": float(cand_inliers),
                "inlier_ratio": float(cand_ratio),
                "fallback_to_center": bool(candidate_match_info.get("fallback_to_center", False)),
                "out_of_bounds": bool(candidate_match_info.get("out_of_bounds", False)),
                "projection_invalid": bool(candidate_match_info.get("projection_invalid", False)),
                "fallback_reason": candidate_match_info.get("fallback_reason"),
            }
        )
        if prefer_match_candidate(cand_inliers, cand_ratio, best_match_inliers, best_match_ratio):
            best_match_loc_lon_lat = candidate_loc_lon_lat
            best_match_info = dict(candidate_match_info)
            best_match_inliers = cand_inliers
            best_match_ratio = cand_ratio
            best_rank = int(rank_j + 1)
            best_angle = float(cand_angle)

    if best_match_loc_lon_lat is None:
        coarse_lat_lon = (float(gallery_center_lon_lat[1]), float(gallery_center_lon_lat[0]))
        best_match_loc_lon_lat = (float(coarse_lat_lon[1]), float(coarse_lat_lon[0]))
        best_match_info = {
            "n_kept": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "fallback_to_center": True,
            "identity_h_fallback": True,
            "out_of_bounds": False,
            "projection_invalid": False,
            "fallback_reason": "all_failed",
        }
        best_rank = None
        best_angle = None

    final_lat_lon = (float(best_match_loc_lon_lat[1]), float(best_match_loc_lon_lat[0]))
    final_error_m = float(geodesic(query_loc, final_lat_lon).meters)
    retained_matches = float(best_match_info.get("n_kept", 0))
    inliers = float(best_match_info.get("inliers", 0))
    inlier_ratio = float(best_match_info.get("inlier_ratio", 0.0))
    if retained_matches <= 0 and inliers > 0:
        retained_matches = inliers
    if retained_matches > 0 and inlier_ratio <= 0:
        inlier_ratio = float(inliers) / float(max(retained_matches, 1.0))

    return {
        "final_error_m": float(final_error_m),
        "dis3_m": float(coarse_dis3_m),
        "dis5_m": float(coarse_dis5_m),
        "runtime_vop_s": float(cache_record.get("vop_time_s", 0.0)),
        "runtime_matcher_s": float(total_match_time),
        "runtime_total_s": float(cache_record.get("vop_time_s", 0.0) + total_match_time),
        "hypotheses_evaluated": int(topk),
        "retained_matches": float(retained_matches),
        "inliers": float(inliers),
        "inlier_ratio": float(inlier_ratio),
        "fallback_to_center": bool(best_match_info.get("fallback_to_center", False)),
        "identity_h_fallback": bool(best_match_info.get("identity_h_fallback", False)),
        "out_of_bounds": bool(best_match_info.get("out_of_bounds", False)),
        "projection_invalid": bool(best_match_info.get("projection_invalid", False)),
        "fallback_reason": best_match_info.get("fallback_reason"),
        "selected_angle_deg": best_angle,
        "selected_rank": best_rank,
        "posterior_top_prob": float(cache_record.get("posterior_top_prob", 0.0)),
        "posterior_entropy": float(cache_record.get("posterior_entropy", 0.0)),
        "posterior_concentration": float(cache_record.get("posterior_concentration", 0.0)),
        "candidate_records": candidate_infos,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    methods = parse_methods(args.methods)
    cache_record_map, candidate_angles = build_cache_record_map(args.cache_path)
    query_meta_map = build_query_meta_map(args.data_root, args.test_pairs_meta_file)

    from game4loc.dataset.visloc import VisLocDatasetEval, get_transforms
    from game4loc.evaluate.visloc import _rotate_query_tensor, predict
    from game4loc.matcher.gim_dkm import GimDKM
    from game4loc.models.model import DesModel

    model = DesModel(args.model, pretrained=True, img_size=args.img_size, share_weights=True)
    state_dict = torch.load(args.checkpoint_start, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    data_config = model.get_config()
    val_transforms, _, _ = get_transforms(
        (args.img_size, args.img_size),
        mean=data_config["mean"],
        std=data_config["std"],
    )

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

    query_loader = DataLoader(
        [query_dataset[idx] for idx in range(len(query_dataset))],
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
        all_scores = (query_features @ gallery_features.T).detach().cpu().numpy()

    gallery_names = list(gallery_dataset.images_name)
    gallery_mask_by_scene = {}
    for idx, gallery_name in enumerate(gallery_names):
        scene_name = str(gallery_name).split("_")[0]
        gallery_mask_by_scene.setdefault(scene_name, np.zeros(len(gallery_names), dtype=np.float32))
        gallery_mask_by_scene[scene_name][idx] = 1.0

    query_matches = []
    for query_name in query_dataset.images_name:
        query_matches.append(np.array([gallery_names.index(pair) for pair in query_dataset.pairs_drone2sate_dict[query_name]], dtype=np.int64))

    matcher = None
    if any(method != "retrieval_only" for method in methods):
        matcher = GimDKM(
            device=device,
            match_mode="sparse",
            logger=None,
            sparse_save_final_vis=False,
        )

    query_records = []
    for query_index, query_name in enumerate(query_dataset.images_name):
        scene_name = str(query_name).split("_")[0]
        masked_score = all_scores[query_index] * gallery_mask_by_scene[scene_name]
        ranking = np.argsort(masked_score)[::-1]
        top1_index = int(ranking[0])
        good_index = np.isin(ranking, query_matches[query_index])
        y_true = good_index.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)
        ap = None
        if np.sum(y_true) > 0:
            ap = float(average_precision_score(y_true, y_scores))

        query_loc = tuple(float(x) for x in query_dataset.images_center_loc_xy[query_index])
        gallery_center_lat, gallery_center_lon = gallery_dataset.images_center_loc_xy[top1_index]
        gallery_topleft_lat, gallery_topleft_lon = gallery_dataset.images_topleft_loc_xy[top1_index]
        coarse_dis1_m = float(geodesic(query_loc, (gallery_center_lat, gallery_center_lon)).meters)
        coarse_dis3_m = 0.0
        coarse_dis5_m = 0.0
        for rank_k, dst_name in ((3, "coarse_dis3_m"), (5, "coarse_dis5_m")):
            dist_sum = 0.0
            for rank_index in ranking[:rank_k]:
                gallery_lat, gallery_lon = gallery_dataset.images_center_loc_xy[int(rank_index)]
                dist_sum += float(geodesic(query_loc, (gallery_lat, gallery_lon)).meters)
            if dst_name == "coarse_dis3_m":
                coarse_dis3_m = dist_sum / float(rank_k)
            else:
                coarse_dis5_m = dist_sum / float(rank_k)

        cache_record = None
        if cache_record_map is not None:
            cache_record = cache_record_map.get(query_name)
            if cache_record is None:
                raise KeyError(f"Missing cache record for query {query_name}")

        query_pose_meta = query_dataset.images_pose_metadata[query_index] or {}
        query_json_meta = query_meta_map.get(query_name, {})
        strict_pos_count = int(query_json_meta.get("strict_pos_count", 0))
        semipos_count = int(query_json_meta.get("semipos_count", 0))
        useful_angle_size_5m = None
        useful_angle_size_3m = None
        useful_angle_size_1m = None
        oracle_best_distance_m = None
        if cache_record is not None and "oracle_distances_m" in cache_record:
            oracle_arr = np.asarray(cache_record["oracle_distances_m"], dtype=np.float64)
            finite_mask = np.isfinite(oracle_arr)
            if np.any(finite_mask):
                oracle_best_distance_m = float(np.min(oracle_arr[finite_mask]))
                useful_angle_size_1m = int(np.sum(oracle_arr[finite_mask] <= oracle_best_distance_m + 1.0))
                useful_angle_size_3m = int(np.sum(oracle_arr[finite_mask] <= oracle_best_distance_m + 3.0))
                useful_angle_size_5m = int(np.sum(oracle_arr[finite_mask] <= oracle_best_distance_m + 5.0))

        record = {
            "query_index": int(query_index),
            "query_name": query_name,
            "scene": scene_name,
            "gallery_count": int(len(gallery_dataset)),
            "query_loc_lat_lon": [float(query_loc[0]), float(query_loc[1])],
            "pose_metadata": {
                "height": safe_float(query_pose_meta.get("height")),
                "omega": safe_float(query_pose_meta.get("omega")),
                "kappa": safe_float(query_pose_meta.get("kappa")),
                "phi1": safe_float(query_pose_meta.get("phi1")),
                "phi2": safe_float(query_pose_meta.get("phi2")),
                "drone_yaw": safe_float(query_pose_meta.get("drone_yaw")),
                "cam_yaw": safe_float(query_pose_meta.get("cam_yaw")),
            },
            "query_attributes": {
                "strict_pos_count": int(strict_pos_count),
                "semipos_count": int(semipos_count),
                "useful_angle_size_1m": useful_angle_size_1m,
                "useful_angle_size_3m": useful_angle_size_3m,
                "useful_angle_size_5m": useful_angle_size_5m,
                "oracle_best_distance_m": oracle_best_distance_m,
            },
            "retrieval": {
                "top1_gallery_index": int(top1_index),
                "top1_gallery_name": gallery_names[top1_index],
                "ranking_top5": [int(x) for x in ranking[:5].tolist()],
                "recall1": bool(top1_index in set(query_matches[query_index].tolist())),
                "ap": ap,
                "coarse_dis1_m": float(coarse_dis1_m),
                "coarse_dis3_m": float(coarse_dis3_m),
                "coarse_dis5_m": float(coarse_dis5_m),
            },
            "methods": {},
        }

        if cache_record is not None and gallery_names[top1_index] != cache_record["gallery_name"]:
            raise ValueError(
                f"Top1 gallery mismatch for {query_name}: retrieval={gallery_names[top1_index]} cache={cache_record['gallery_name']}"
            )

        if "retrieval_only" in methods:
            record["methods"]["retrieval_only"] = {
                "final_error_m": float(coarse_dis1_m),
                "dis3_m": float(coarse_dis3_m),
                "dis5_m": float(coarse_dis5_m),
                "runtime_vop_s": 0.0,
                "runtime_matcher_s": 0.0,
                "runtime_total_s": 0.0,
                "hypotheses_evaluated": 0,
                "retained_matches": None,
                "inliers": None,
                "inlier_ratio": None,
                "fallback_to_center": False,
                "identity_h_fallback": False,
                "out_of_bounds": False,
                "projection_invalid": False,
                "fallback_reason": None,
                "selected_angle_deg": None,
                "selected_rank": None,
                "worse_than_coarse": False,
            }

        if matcher is not None and any(method != "retrieval_only" for method in methods):
            gallery_sample = gallery_dataset[top1_index]
            query_sample = query_dataset[query_index]
            gallery_center_lon_lat = (float(gallery_center_lon), float(gallery_center_lat))
            gallery_topleft_lon_lat = (float(gallery_topleft_lon), float(gallery_topleft_lat))

            if "no_rotate" in methods:
                output = run_single_match(
                    matcher,
                    query_sample,
                    gallery_sample,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                    query_loc,
                    coarse_dis3_m,
                    coarse_dis5_m,
                    rotate_step=0.0,
                )
                output["worse_than_coarse"] = bool(output["final_error_m"] > coarse_dis1_m + 1e-6)
                record["methods"]["no_rotate"] = output

            if "rotate90" in methods:
                output = run_single_match(
                    matcher,
                    query_sample,
                    gallery_sample,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                    query_loc,
                    coarse_dis3_m,
                    coarse_dis5_m,
                    rotate_step=90.0,
                )
                output["worse_than_coarse"] = bool(output["final_error_m"] > coarse_dis1_m + 1e-6)
                record["methods"]["rotate90"] = output

            for method_name in methods:
                if not method_name.startswith("prior_topk"):
                    continue
                if cache_record is None or not candidate_angles:
                    raise RuntimeError(f"Method {method_name} requires --cache_path with posterior_probs and candidate_angles.")
                topk = int(method_name.replace("prior_topk", ""))
                output = run_prior_topk(
                    matcher,
                    _rotate_query_tensor,
                    query_sample,
                    gallery_sample,
                    gallery_center_lon_lat,
                    gallery_topleft_lon_lat,
                    query_loc,
                    coarse_dis3_m,
                    coarse_dis5_m,
                    cache_record,
                    candidate_angles,
                    topk,
                )
                output["worse_than_coarse"] = bool(output["final_error_m"] > coarse_dis1_m + 1e-6)
                record["methods"][method_name] = output

        query_records.append(record)

    retrieval_ap = [record["retrieval"]["ap"] for record in query_records if record["retrieval"]["ap"] is not None]
    recall1 = np.mean([record["retrieval"]["recall1"] for record in query_records]) * 100.0
    coarse_dis1 = np.mean([record["retrieval"]["coarse_dis1_m"] for record in query_records])
    coarse_dis3 = np.mean([record["retrieval"]["coarse_dis3_m"] for record in query_records])
    coarse_dis5 = np.mean([record["retrieval"]["coarse_dis5_m"] for record in query_records])

    summary = {
        "retrieval": {
            "query_count": int(len(query_records)),
            "gallery_count": int(len(gallery_dataset)),
            "Recall@1_pct": float(recall1),
            "AP_pct": None if not retrieval_ap else float(np.mean(retrieval_ap) * 100.0),
            "Dis@1_m": float(coarse_dis1),
            "Dis@3_m": float(coarse_dis3),
            "Dis@5_m": float(coarse_dis5),
            "coarse_error_distribution_m": distribution_stats([record["retrieval"]["coarse_dis1_m"] for record in query_records]),
        },
        "methods": {},
        "scenes": summarize_scenes(query_records, methods),
    }
    for method_name in sorted(methods, key=method_sort_key):
        summary["methods"][method_name] = summarize_method(query_records, method_name, len(gallery_dataset))

    output = {
        "config": {
            "data_root": os.path.abspath(args.data_root),
            "test_pairs_meta_file": args.test_pairs_meta_file,
            "checkpoint_start": os.path.abspath(args.checkpoint_start),
            "cache_path": os.path.abspath(args.cache_path) if args.cache_path else "",
            "methods": methods,
            "device": device,
        },
        "summary": summary,
        "records": query_records,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)

    print(f"Saved attribution analysis to {args.output_path}")
    print(json.dumps(to_serializable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
