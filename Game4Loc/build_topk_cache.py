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
from game4loc.evaluate.visloc import predict, project_match_center_from_h
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import load_vop_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Build cached oracle curves and VOP posteriors for top-k experiments.")
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


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    orientation_model = load_vop_checkpoint(args.orientation_checkpoint, device=device)
    candidate_angles = getattr(orientation_model, "candidate_angles_deg", None)
    if not candidate_angles or len(candidate_angles) < 2:
        raise RuntimeError("Orientation checkpoint must contain a valid candidate angle set.")
    oracle_rotate_step = abs(float(candidate_angles[1] - candidate_angles[0]))

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

    records = []
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
        query_loc = query_dataset.images_center_loc_xy[query_index]
        gallery_name = gallery_dataset.images_name[top1_index]
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
            case_name=f"{query_name}_oracle_grid",
        )
        oracle_time = time.perf_counter() - t_oracle
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

        coarse_dis3 = 0.0
        coarse_dis5 = 0.0
        for rank_k, attr_name in ((3, "coarse_dis3_m"), (5, "coarse_dis5_m")):
            dist_sum = 0.0
            for idx in ranking[:rank_k]:
                gallery_lat, gallery_lon = gallery_dataset.images_center_loc_xy[int(idx)]
                dist_sum += float(geodesic(query_loc, (gallery_lat, gallery_lon)).meters)
            if rank_k == 3:
                coarse_dis3 = dist_sum / float(rank_k)
            else:
                coarse_dis5 = dist_sum / float(rank_k)

        records.append(
            {
                "query_index": int(query_index),
                "query_name": query_name,
                "query_loc_xy": [float(query_loc[0]), float(query_loc[1])],
                "gallery_index": int(top1_index),
                "gallery_name": gallery_name,
                "gallery_center_lon_lat": [float(gallery_center_lon), float(gallery_center_lat)],
                "gallery_topleft_lon_lat": [float(gallery_topleft_lon), float(gallery_topleft_lat)],
                "ranking_top5": [int(x) for x in ranking[:5].tolist()],
                "coarse_dis3_m": float(coarse_dis3),
                "coarse_dis5_m": float(coarse_dis5),
                "oracle_distances_m": [float(x) for x in oracle_distances],
                "oracle_time_s": float(oracle_time),
                "posterior_probs": [float(x) for x in posterior["probs"]],
                "posterior_top_prob": float(posterior["top_prob"]),
                "posterior_entropy": float(posterior["entropy"]),
                "posterior_concentration": float(posterior["concentration"]),
                "vop_time_s": float(vop_time),
            }
        )

    output = {
        "config": {
            "data_root": args.data_root,
            "test_pairs_meta_file": args.test_pairs_meta_file,
            "orientation_checkpoint": args.orientation_checkpoint,
            "checkpoint_start": args.checkpoint_start,
            "oracle_rotate_step": float(oracle_rotate_step),
            "candidate_angles_deg": [float(x) for x in candidate_angles],
            "query_count": int(len(records)),
        },
        "records": records,
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(output), f, ensure_ascii=False, indent=2)

    print(f"Saved cache to {args.output_path}")
    print(json.dumps(to_serializable(output["config"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
