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
from game4loc.orientation import load_vop_checkpoint


USEFUL_DELTAS_M = (1.0, 3.0, 5.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze top-k useful angle hypotheses and mechanism controls for fine localization."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_pairs_meta_file", type=str, required=True)
    parser.add_argument("--orientation_checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--checkpoint_start", type=str, default="")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size_eval", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sate_img_dir", type=str, default="satellite")
    parser.add_argument("--query_limit", type=int, default=0)
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
        "q90": float(np.quantile(arr, 0.90)),
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
    mask = []
    threshold = float(best_distance) + float(delta_m)
    for dist in oracle_distances:
        valid = math.isfinite(float(dist)) and float(dist) <= threshold
        mask.append(bool(valid))
    return mask


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
    max_widths = [record["useful_sets"][delta_key]["max_interval_width_deg"] for record in records]
    return {
        "size": distribution_stats(sizes),
        "components": distribution_stats(components),
        "max_interval_width_deg": distribution_stats(max_widths),
        "single_interval_ratio": float(np.mean([comp == 1 for comp in components])) if components else float("nan"),
        "multimodal_ratio": float(np.mean([comp > 1 for comp in components])) if components else float("nan"),
        "count_ge_2_ratio": float(np.mean([size >= 2 for size in sizes])) if sizes else float("nan"),
        "count_ge_3_ratio": float(np.mean([size >= 3 for size in sizes])) if sizes else float("nan"),
    }


def aggregate(records, strategy, topk, candidate_angles):
    dis1 = [record["final_error_m"] for record in records]
    dis3 = [record["dis3_m"] for record in records]
    dis5 = [record["dis5_m"] for record in records]
    kept = [record["n_kept"] for record in records]
    inliers = [record["inliers"] for record in records]
    ratios = [record["inlier_ratio"] for record in records]
    vop_time = [record["vop_time_s"] for record in records]
    matcher_time = [record["match_time_s"] for record in records]
    top_prob = [record["posterior_top_prob"] for record in records]
    entropy = [record["posterior_entropy"] for record in records]
    concentration = [record["posterior_concentration"] for record in records]
    oracle_hit = [float(record["oracle_best_in_topk"]) for record in records]

    useful_summary = {}
    for delta in USEFUL_DELTAS_M:
        key = f"{int(delta)}m"
        useful_any = [float(record["useful_sets"][key]["selected_hits_any"]) for record in records]
        useful_recall = [float(record["useful_sets"][key]["selected_set_recall"]) for record in records]
        covered_records = [record for record in records if record["useful_sets"][key]["selected_hits_any"]]
        missed_records = [record for record in records if not record["useful_sets"][key]["selected_hits_any"]]
        useful_summary[key] = {
            "any_hit_rate": float(np.mean(useful_any)),
            "set_recall_mean": float(np.mean(useful_recall)),
            "covered_error_m": distribution_stats([item["final_error_m"] for item in covered_records]),
            "missed_error_m": distribution_stats([item["final_error_m"] for item in missed_records]),
            "structure": summarize_useful_structure(records, key),
        }

    return {
        "strategy": strategy,
        "topk": int(topk),
        "candidate_angles_deg": [float(x) for x in candidate_angles],
        "uniform_indices": build_uniform_indices(len(candidate_angles), topk),
        "uniform_angles_deg": [float(candidate_angles[idx]) for idx in build_uniform_indices(len(candidate_angles), topk)],
        "query_count": int(len(records)),
        "metrics": {
            "Dis@1_m": float(np.mean(dis1)),
            "Dis@3_m": float(np.mean(dis3)),
            "Dis@5_m": float(np.mean(dis5)),
        },
        "runtime_s_per_query": {
            "vop_forward": float(np.mean(vop_time)),
            "matcher": float(np.mean(matcher_time)),
            "total": float(np.mean(np.asarray(vop_time) + np.asarray(matcher_time))),
        },
        "topk_oracle_best_coverage": float(np.mean(oracle_hit)),
        "posterior_stats": {
            "mean_top_prob": float(np.mean(top_prob)),
            "mean_entropy": float(np.mean(entropy)),
            "mean_concentration": float(np.mean(concentration)),
        },
        "match_stats": {
            "mean_hypotheses": float(np.mean([record["num_hypotheses"] for record in records])),
            "retained_matches": float(np.mean(kept)),
            "mean_inliers": float(np.mean(inliers)),
            "mean_inlier_ratio": float(np.mean(ratios)),
        },
        "useful_coverage": useful_summary,
    }


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    orientation_model = load_vop_checkpoint(args.orientation_checkpoint, device=device)
    candidate_angles = getattr(orientation_model, "candidate_angles_deg", None)
    if not candidate_angles:
        raise RuntimeError(f"Orientation checkpoint {args.orientation_checkpoint} does not contain candidate angles.")
    if len(candidate_angles) < 2:
        raise RuntimeError("Candidate angle set must contain at least two angles.")

    topk = max(1, min(int(args.topk), len(candidate_angles)))
    oracle_rotate_step = abs(float(candidate_angles[1] - candidate_angles[0]))
    rng = np.random.default_rng(int(args.random_seed))

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
        all_scores = (query_features @ gallery_features.T).detach().cpu().numpy()

    gallery_mapi_idx = {}
    gallery_names = list(gallery_dataset.images_name)
    for idx, gallery_name in enumerate(gallery_names):
        map_prefix = str(gallery_name).split("_")[0]
        gallery_mapi_idx.setdefault(map_prefix, np.zeros(len(gallery_names), dtype=np.float32))
        gallery_mapi_idx[map_prefix][idx] = 1.0

    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        logger=None,
        sparse_save_final_vis=False,
    )

    eval_records = []
    for local_idx, query_index in enumerate(query_indices):
        query_name = query_dataset.images_name[query_index]
        map_prefix = str(query_name).split("_")[0]
        map_mask = gallery_mapi_idx.get(map_prefix)
        if map_mask is None:
            raise KeyError(f"Cannot find gallery map mask for query {query_name}")
        score = all_scores[local_idx] * map_mask
        ranking = np.argsort(score)[::-1]
        top1_index = int(ranking[0])
        query_sample = query_dataset[query_index]
        gallery_sample = gallery_dataset[top1_index]
        gallery_name = gallery_dataset.images_name[top1_index]
        query_loc = query_dataset.images_center_loc_xy[query_index]
        gallery_center_lat, gallery_center_lon = gallery_dataset.images_center_loc_xy[top1_index]
        gallery_topleft_lat, gallery_topleft_lon = gallery_dataset.images_topleft_loc_xy[top1_index]
        gallery_center_lon_lat = (gallery_center_lon, gallery_center_lat)
        gallery_topleft_lon_lat = (gallery_topleft_lon, gallery_topleft_lat)

        _ = matcher.est_center(
            gallery_sample,
            query_sample,
            gallery_center_lon_lat,
            gallery_topleft_lon_lat,
            yaw0=None,
            yaw1=None,
            rotate=oracle_rotate_step,
            case_name=f"{query_name}_oracle_grid",
        )
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

        oracle_dist_arr = np.asarray(oracle_distances, dtype=np.float64)
        finite_mask = np.isfinite(oracle_dist_arr)
        if not np.any(finite_mask):
            continue
        best_index = int(np.argmin(oracle_dist_arr))
        best_distance = float(oracle_dist_arr[best_index])

        posterior_probs = None
        posterior_top_prob = 0.0
        posterior_entropy = 0.0
        posterior_concentration = 0.0
        vop_time = 0.0
        if args.strategy == "posterior":
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
            posterior_probs = posterior["probs"]
            posterior_top_prob = float(posterior["top_prob"])
            posterior_entropy = float(posterior["entropy"])
            posterior_concentration = float(posterior["concentration"])

        selected_indices = select_indices(
            strategy=args.strategy,
            topk=topk,
            num_angles=len(candidate_angles),
            posterior_probs=posterior_probs,
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
                case_name=f"{query_name}_{args.strategy}_top{topk}_{cand_angle:.1f}",
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

        if best_match_loc_lon_lat is None:
            continue

        final_lat_lon = (best_match_loc_lon_lat[1], best_match_loc_lon_lat[0])
        final_error_m = float(geodesic(query_loc, final_lat_lon).meters)
        n_kept = int((best_match_info or {}).get("n_kept", 0))
        inliers = int((best_match_info or {}).get("inliers", 0))
        inlier_ratio = float(inliers) / float(max(n_kept, 1))

        gallery_center_distances = []
        for rank_k in (1, 3, 5):
            dist_sum = 0.0
            for idx in ranking[:rank_k]:
                gallery_lat, gallery_lon = gallery_dataset.images_center_loc_xy[int(idx)]
                dist_sum += float(geodesic(query_loc, (gallery_lat, gallery_lon)).meters)
            gallery_center_distances.append(dist_sum / float(rank_k))

        useful_sets = {}
        for delta in USEFUL_DELTAS_M:
            key = f"{int(delta)}m"
            useful_mask = build_useful_mask(oracle_distances, best_distance, delta)
            useful_indices = [idx for idx, flag in enumerate(useful_mask) if flag]
            comp_count, comp_lengths = circular_components(useful_mask)
            selected_hits = [idx for idx in selected_indices if idx in useful_indices]
            useful_sets[key] = {
                "size": int(len(useful_indices)),
                "indices": [int(idx) for idx in useful_indices],
                "components": int(comp_count),
                "component_lengths_bins": [int(x) for x in comp_lengths],
                "max_interval_width_deg": float(max(comp_lengths) * oracle_rotate_step) if comp_lengths else 0.0,
                "selected_hits_any": bool(len(selected_hits) > 0),
                "selected_hit_indices": [int(idx) for idx in selected_hits],
                "selected_set_recall": float(len(selected_hits)) / float(max(len(useful_indices), 1)),
            }

        eval_records.append(
            {
                "query_index": int(query_index),
                "query_name": query_name,
                "gallery_index": int(top1_index),
                "gallery_name": gallery_name,
                "oracle_best_index": int(best_index),
                "oracle_best_angle_deg": float(candidate_angles[best_index]),
                "oracle_best_distance_m": float(best_distance),
                "oracle_distances_m": [float(x) for x in oracle_distances],
                "selected_indices": [int(idx) for idx in selected_indices],
                "selected_angles_deg": [float(candidate_angles[idx]) for idx in selected_indices],
                "selected_match_angle_deg": float(best_match_angle),
                "oracle_best_in_topk": bool(best_index in selected_indices),
                "final_error_m": float(final_error_m),
                "dis3_m": float(gallery_center_distances[1]),
                "dis5_m": float(gallery_center_distances[2]),
                "n_kept": int(n_kept),
                "inliers": int(inliers),
                "inlier_ratio": float(inlier_ratio),
                "num_hypotheses": int(len(selected_indices)),
                "vop_time_s": float(vop_time),
                "match_time_s": float(match_time),
                "posterior_top_prob": float(posterior_top_prob),
                "posterior_entropy": float(posterior_entropy),
                "posterior_concentration": float(posterior_concentration),
                "useful_sets": useful_sets,
            }
        )

    output = {
        "config": {
            "data_root": args.data_root,
            "test_pairs_meta_file": args.test_pairs_meta_file,
            "orientation_checkpoint": args.orientation_checkpoint,
            "checkpoint_start": args.checkpoint_start,
            "strategy": args.strategy,
            "topk": int(topk),
            "random_seed": int(args.random_seed),
            "oracle_rotate_step": float(oracle_rotate_step),
        },
        "summary": aggregate(eval_records, args.strategy, topk, candidate_angles),
        "records": eval_records,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)

    print(f"Saved analysis to {args.output_path}")
    print(json.dumps(to_serializable(output["summary"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
