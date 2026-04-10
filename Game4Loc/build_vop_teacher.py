import argparse
import json
import math
import os
from typing import List

import cv2
import torch
import albumentations as A
from geopy.distance import geodesic
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from game4loc.dataset.gta import sate2loc
from game4loc.dataset.visloc import tile2sate
from game4loc.evaluate.visloc import project_match_center_from_h
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import build_rotation_angle_list, compute_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Build soft teacher targets for VOP.")
    parser.add_argument("--dataset", type=str, default="visloc", choices=("visloc", "gta"))
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pairs_meta_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--checkpoint_start", type=str, default="")
    parser.add_argument("--rotate_step", type=float, default=10.0)
    parser.add_argument("--temperature_m", type=float, default=25.0)
    parser.add_argument("--query_limit", type=int, default=0)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def build_eval_transform(img_size: int, mean, std):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_visloc_pair_records(data_root: str, pairs_meta_file: str, query_limit: int = 0) -> List[dict]:
    with open(os.path.join(data_root, pairs_meta_file), "r", encoding="utf-8") as f:
        pairs_meta = json.load(f)

    records = []
    for item in pairs_meta:
        positive_tiles = item.get("pair_pos_sate_img_list") or []
        if not positive_tiles:
            continue
        gallery_name = positive_tiles[0]
        gallery_center_lat_lon, gallery_topleft_lat_lon = tile2sate(gallery_name)
        records.append(
            {
                "dataset": "visloc",
                "query_name": item["drone_img_name"],
                "gallery_name": gallery_name,
                "query_path": os.path.join(data_root, item["drone_img_dir"], item["drone_img_name"]),
                "gallery_path": os.path.join(data_root, item["sate_img_dir"], gallery_name),
                "query_eval_loc": [float(item["drone_loc_lat_lon"][0]), float(item["drone_loc_lat_lon"][1])],
                "gallery_center_xy": [float(gallery_center_lat_lon[1]), float(gallery_center_lat_lon[0])],
                "gallery_topleft_xy": [float(gallery_topleft_lat_lon[1]), float(gallery_topleft_lat_lon[0])],
                "distance_space": "geodesic",
            }
        )
        if query_limit > 0 and len(records) >= query_limit:
            break
    return records


def load_gta_pair_records(data_root: str, pairs_meta_file: str, query_limit: int = 0) -> List[dict]:
    with open(os.path.join(data_root, pairs_meta_file), "r", encoding="utf-8") as f:
        pairs_meta = json.load(f)

    records = []
    for item in pairs_meta:
        positive_tiles = item.get("pair_pos_sate_img_list") or []
        if not positive_tiles:
            continue
        gallery_name = positive_tiles[0]
        tile_zoom, offset, tile_x, tile_y = gallery_name.replace(".png", "").split("_")
        gallery_center_x, gallery_center_y, gallery_topleft_x, gallery_topleft_y = sate2loc(
            int(tile_zoom),
            int(offset),
            int(tile_x),
            int(tile_y),
        )
        records.append(
            {
                "dataset": "gta",
                "query_name": item["drone_img_name"],
                "gallery_name": gallery_name,
                "query_path": os.path.join(data_root, item["drone_img_dir"], item["drone_img_name"]),
                "gallery_path": os.path.join(data_root, item["sate_img_dir"], gallery_name),
                "query_eval_loc": [float(item["drone_loc_x_y"][0]), float(item["drone_loc_x_y"][1])],
                "gallery_center_xy": [float(gallery_center_x), float(gallery_center_y)],
                "gallery_topleft_xy": [float(gallery_topleft_x), float(gallery_topleft_y)],
                "distance_space": "xy",
            }
        )
        if query_limit > 0 and len(records) >= query_limit:
            break
    return records


def load_pair_records(dataset: str, data_root: str, pairs_meta_file: str, query_limit: int = 0) -> List[dict]:
    dataset_key = str(dataset).strip().lower()
    if dataset_key == "visloc":
        return load_visloc_pair_records(data_root, pairs_meta_file, query_limit=query_limit)
    if dataset_key == "gta":
        return load_gta_pair_records(data_root, pairs_meta_file, query_limit=query_limit)
    raise ValueError(f"Unsupported dataset: {dataset}")


def compute_distance_m(distance_space: str, query_eval_loc, projected_xy) -> float:
    if projected_xy is None:
        return float("inf")

    if str(distance_space).lower() == "geodesic":
        pred_lat_lon = (float(projected_xy[1]), float(projected_xy[0]))
        query_lat_lon = (float(query_eval_loc[0]), float(query_eval_loc[1]))
        return float(geodesic(query_lat_lon, pred_lat_lon).meters)

    if str(distance_space).lower() == "xy":
        dx = float(query_eval_loc[0]) - float(projected_xy[0])
        dy = float(query_eval_loc[1]) - float(projected_xy[1])
        return float(math.hypot(dx, dy))

    raise ValueError(f"Unsupported distance space: {distance_space}")


def load_rgb_tensor(image_path: str, transform):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image=image)["image"]


def soft_distribution_from_distances(distances_m, temperature_m: float):
    distances = torch.tensor(distances_m, dtype=torch.float32)
    valid_mask = torch.isfinite(distances)
    logits = torch.full_like(distances, fill_value=-40.0)
    logits[valid_mask] = -distances[valid_mask] / max(float(temperature_m), 1e-6)
    return torch.softmax(logits, dim=0).tolist()


def summarize_teacher_records(records):
    if not records:
        return {}

    entropies = []
    best_probs = []
    prob_margins = []
    distance_gaps = []
    best_distances = []
    informative_count = 0

    for record in records:
        probs = torch.tensor(record["target_probs"], dtype=torch.float32)
        entropies.append(float(compute_entropy(probs.unsqueeze(0))[0].item()))
        sorted_probs = torch.sort(probs, descending=True).values
        best_probs.append(float(sorted_probs[0].item()))
        prob_margins.append(float((sorted_probs[0] - sorted_probs[1]).item()))

        best_distance = float(record.get("best_distance_m", float("inf")))
        second_distance = float(record.get("second_distance_m", float("inf")))
        best_distances.append(best_distance)
        if math.isfinite(second_distance) and math.isfinite(best_distance):
            gap = second_distance - best_distance
            distance_gaps.append(float(gap))
            if gap >= 5.0:
                informative_count += 1

    summary = {
        "count": int(len(records)),
        "entropy_mean": float(sum(entropies) / len(entropies)),
        "entropy_min": float(min(entropies)),
        "entropy_max": float(max(entropies)),
        "best_prob_mean": float(sum(best_probs) / len(best_probs)),
        "best_prob_min": float(min(best_probs)),
        "best_prob_max": float(max(best_probs)),
        "prob_margin_mean": float(sum(prob_margins) / len(prob_margins)),
        "prob_margin_min": float(min(prob_margins)),
        "prob_margin_max": float(max(prob_margins)),
        "best_distance_mean_m": float(sum(best_distances) / len(best_distances)),
        "distance_gap_mean_m": float(sum(distance_gaps) / len(distance_gaps)) if distance_gaps else float("nan"),
        "distance_gap_ge_5m_count": int(informative_count),
    }
    return summary


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = DesModel(args.model, pretrained=True, img_size=args.img_size, share_weights=True)
    if args.checkpoint_start:
        state_dict = torch.load(args.checkpoint_start, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    data_config = model.get_config()
    val_transforms = build_eval_transform(args.img_size, mean=data_config["mean"], std=data_config["std"])

    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        logger=None,
        sparse_save_final_vis=False,
    )
    records = load_pair_records(args.dataset, args.data_root, args.pairs_meta_file, query_limit=args.query_limit)
    expected_angles = build_rotation_angle_list(args.rotate_step)
    teacher_records = []

    for record in tqdm(records, desc="Building VOP teacher"):
        gallery_tensor = load_rgb_tensor(record["gallery_path"], val_transforms)
        query_tensor = load_rgb_tensor(record["query_path"], val_transforms)

        gallery_center_xy = (float(record["gallery_center_xy"][0]), float(record["gallery_center_xy"][1]))
        gallery_topleft_xy = (float(record["gallery_topleft_xy"][0]), float(record["gallery_topleft_xy"][1]))
        _ = matcher.est_center(
            gallery_tensor,
            query_tensor,
            gallery_center_xy,
            gallery_topleft_xy,
            yaw0=None,
            yaw1=None,
            rotate=args.rotate_step,
            case_name=record["query_name"],
        )
        angle_results = matcher.get_last_angle_results()
        angle_results = [item for item in angle_results if int(item.get("phase", 0)) == 1]
        angle_results = sorted(angle_results, key=lambda item: float(item.get("search_angle", 0.0)))

        distances_m = []
        candidate_angles = []
        for angle_result in angle_results:
            rot_angle = float(angle_result.get("rot_angle", 0.0))
            candidate_angles.append(rot_angle)
            loc_xy = project_match_center_from_h(
                angle_result.get("homography"),
                gallery_tensor,
                gallery_center_xy,
                gallery_topleft_xy,
            )
            if loc_xy is None:
                distances_m.append(float("inf"))
                continue
            distances_m.append(
                compute_distance_m(
                    distance_space=record["distance_space"],
                    query_eval_loc=record["query_eval_loc"],
                    projected_xy=loc_xy,
                )
            )

        if candidate_angles != expected_angles:
            raise ValueError(
                f"Unexpected angle list for {record['query_name']}: got {candidate_angles}, expected {expected_angles}"
            )

        target_probs = soft_distribution_from_distances(distances_m, temperature_m=args.temperature_m)
        target_tensor = torch.tensor(target_probs, dtype=torch.float32)
        best_index = int(target_tensor.argmax().item())
        distance_tensor = torch.tensor(distances_m, dtype=torch.float32)
        finite_mask = torch.isfinite(distance_tensor)
        finite_distances = distance_tensor[finite_mask]
        best_distance = float("inf")
        second_distance = float("inf")
        if finite_distances.numel() > 0:
            sorted_distances = torch.sort(finite_distances).values
            best_distance = float(sorted_distances[0].item())
            if sorted_distances.numel() > 1:
                second_distance = float(sorted_distances[1].item())
        teacher_records.append(
            {
                **record,
                "candidate_angles_deg": candidate_angles,
                "distances_m": distances_m,
                "target_probs": target_probs,
                "best_angle_deg": float(candidate_angles[best_index]),
                "best_index": int(best_index),
                "best_distance_m": best_distance,
                "second_distance_m": second_distance,
                "distance_gap_m": float(second_distance - best_distance)
                if math.isfinite(best_distance) and math.isfinite(second_distance)
                else float("inf"),
            }
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    teacher_summary = summarize_teacher_records(teacher_records)
    torch.save(
        {
            "records": teacher_records,
            "rotate_step": float(args.rotate_step),
            "temperature_m": float(args.temperature_m),
            "dataset": str(args.dataset),
            "pairs_meta_file": args.pairs_meta_file,
            "img_size": int(args.img_size),
            "model": args.model,
            "checkpoint_start": args.checkpoint_start,
            "summary": teacher_summary,
        },
        args.output_path,
    )
    print(f"Saved {len(teacher_records)} teacher records to {args.output_path}")
    if teacher_summary:
        print(f"Teacher summary: {teacher_summary}")


if __name__ == "__main__":
    main()
