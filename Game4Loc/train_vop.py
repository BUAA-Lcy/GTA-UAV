import argparse
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from game4loc.models.model import DesModel
from game4loc.orientation import VisualOrientationPosterior, build_rotation_angle_list, compute_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="Train the visual orientation posterior head.")
    parser.add_argument("--teacher_cache", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--checkpoint_start", type=str, default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--neg_prob", type=float, default=0.25)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--ranking_weight", type=float, default=0.5)
    parser.add_argument("--ranking_margin", type=float, default=0.2)
    parser.add_argument("--ranking_gap_m", type=float, default=5.0)
    parser.add_argument("--ranking_gap_scale_m", type=float, default=20.0)
    parser.add_argument("--ce_weight", type=float, default=0.5)
    parser.add_argument("--ce_gap_m", type=float, default=5.0)
    parser.add_argument("--ce_entropy_max", type=float, default=0.8)
    parser.add_argument("--filter_entropy_max", type=float, default=1.0)
    parser.add_argument("--filter_gap_m", type=float, default=0.0)
    parser.add_argument("--filter_best_distance_max", type=float, default=float("inf"))
    parser.add_argument("--partial_unfreeze", type=str, default="none", choices=("none", "last_block"))
    parser.add_argument("--supervision_mode", type=str, default="posterior", choices=("posterior", "useful_bce"))
    parser.add_argument("--useful_delta_m", type=float, default=5.0)
    parser.add_argument(
        "--pair_weight_mode",
        type=str,
        default="uniform",
        choices=("uniform", "best_distance_sigmoid"),
    )
    parser.add_argument("--pair_weight_center_m", type=float, default=30.0)
    parser.add_argument("--pair_weight_scale_m", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_transform(img_size: int, mean, std):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def load_rgb_tensor(image_path: str, transform):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image=image)["image"]


class VOPDataset(Dataset):
    def __init__(
        self,
        records,
        transform,
        neg_prob=0.25,
        pair_weight_mode="uniform",
        pair_weight_center_m=30.0,
        pair_weight_scale_m=10.0,
    ):
        self.records = list(records)
        self.transform = transform
        self.neg_prob = float(neg_prob)
        self.pair_weight_mode = str(pair_weight_mode).lower()
        self.pair_weight_center_m = float(pair_weight_center_m)
        self.pair_weight_scale_m = float(pair_weight_scale_m)
        self.gallery_pool = [record["gallery_path"] for record in self.records]
        self.uniform_target = None
        if self.records:
            num_angles = len(self.records[0]["target_probs"])
            self.uniform_target = torch.full((num_angles,), 1.0 / max(num_angles, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.records)

    def compute_pair_weight(self, record):
        if self.pair_weight_mode == "uniform":
            return 1.0
        if self.pair_weight_mode == "best_distance_sigmoid":
            best_distance = float(record.get("best_distance_m", float("inf")))
            if not math.isfinite(best_distance):
                return 0.0
            scale = max(self.pair_weight_scale_m, 1e-6)
            value = 1.0 / (1.0 + math.exp((best_distance - self.pair_weight_center_m) / scale))
            return float(min(max(value, 0.0), 1.0))
        raise ValueError(f"Unsupported pair_weight_mode: {self.pair_weight_mode}")

    def __getitem__(self, index):
        record = self.records[index]
        query_img = load_rgb_tensor(record["query_path"], self.transform)
        distances = torch.tensor(record["distances_m"], dtype=torch.float32)
        best_index = int(record.get("best_index", int(torch.argmin(torch.nan_to_num(distances, nan=float("inf"), posinf=float("inf"))).item())))
        distance_gap = float(record.get("distance_gap_m", float("inf")))
        target_entropy = float(compute_entropy(torch.tensor(record["target_probs"], dtype=torch.float32).unsqueeze(0))[0].item())
        pair_weight = float(self.compute_pair_weight(record))

        use_negative = self.neg_prob > 0.0 and random.random() < self.neg_prob and len(self.records) > 1
        if use_negative:
            neg_index = random.randrange(len(self.records) - 1)
            if neg_index >= index:
                neg_index += 1
            gallery_path = self.records[neg_index]["gallery_path"]
            target = self.uniform_target.clone()
            distances = torch.full_like(distances, float("inf"))
            ranking_mask = torch.zeros_like(target, dtype=torch.bool)
            ranking_weight = torch.zeros_like(target, dtype=torch.float32)
            best_index = -1
            distance_gap = float("inf")
            target_entropy = 1.0
            pair_weight = 1.0
        else:
            gallery_path = record["gallery_path"]
            target = torch.tensor(record["target_probs"], dtype=torch.float32)
            ranking_mask = torch.isfinite(distances)
            ranking_weight = torch.zeros_like(target, dtype=torch.float32)

        gallery_img = load_rgb_tensor(gallery_path, self.transform)
        return (
            query_img,
            gallery_img,
            target,
            distances,
            ranking_mask,
            ranking_weight,
            best_index,
            distance_gap,
            target_entropy,
            torch.tensor(pair_weight, dtype=torch.float32),
        )


def compute_top1_angle_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred_idx = torch.argmax(logits, dim=1)
    target_idx = torch.argmax(targets, dim=1)
    return float((pred_idx == target_idx).float().mean().item())


def build_ranking_targets(distances: torch.Tensor, best_indices: torch.Tensor, ranking_gap_m: float, ranking_gap_scale_m: float):
    batch_size, num_angles = distances.shape
    finite_mask = torch.isfinite(distances)
    best_distances = torch.full((batch_size,), float("inf"), dtype=distances.dtype, device=distances.device)
    valid_best = best_indices >= 0
    valid_rows = torch.nonzero(valid_best, as_tuple=False).squeeze(1)
    if valid_rows.numel() > 0:
        best_distances[valid_rows] = distances[valid_rows, best_indices[valid_rows]]

    gap = distances - best_distances[:, None]
    informative = finite_mask & valid_best[:, None] & (gap >= float(ranking_gap_m))
    informative &= torch.arange(num_angles, device=distances.device)[None, :] != best_indices[:, None]
    weights = torch.zeros_like(distances)
    if informative.any():
        scaled_gap = gap.clamp_min(0.0) / max(float(ranking_gap_scale_m), 1e-6)
        weights[informative] = scaled_gap[informative].clamp(min=0.25, max=2.0)
    return informative, weights


def compute_ranking_loss(logits: torch.Tensor, best_indices: torch.Tensor, informative_mask: torch.Tensor, weights: torch.Tensor, ranking_margin: float):
    valid_best = best_indices >= 0
    if not valid_best.any():
        return logits.new_tensor(0.0), 0

    best_logits = torch.zeros((logits.shape[0], 1), dtype=logits.dtype, device=logits.device)
    valid_rows = torch.nonzero(valid_best, as_tuple=False).squeeze(1)
    best_logits[valid_rows, 0] = logits[valid_rows, best_indices[valid_rows]]
    diff = best_logits - logits
    margin = float(ranking_margin) * weights.clamp_min(1.0)
    loss_tensor = F.relu(margin - diff)
    effective = informative_mask & valid_best[:, None]
    pair_count = int(effective.sum().item())
    if pair_count <= 0:
        return logits.new_tensor(0.0), 0
    loss = (loss_tensor * weights * effective.float()).sum() / effective.float().sum().clamp_min(1.0)
    return loss, pair_count


def compute_ce_loss(
    logits: torch.Tensor,
    best_indices: torch.Tensor,
    distance_gaps: torch.Tensor,
    target_entropies: torch.Tensor,
    ce_gap_m: float,
    ce_entropy_max: float,
):
    valid = best_indices >= 0
    valid &= torch.isfinite(distance_gaps)
    valid &= distance_gaps >= float(ce_gap_m)
    valid &= target_entropies <= float(ce_entropy_max)
    sample_count = int(valid.sum().item())
    if sample_count <= 0:
        return logits.new_tensor(0.0), 0
    loss = F.cross_entropy(logits[valid], best_indices[valid], reduction="mean")
    return loss, sample_count


def build_useful_targets(distances: torch.Tensor, best_indices: torch.Tensor, useful_delta_m: float):
    batch_size, num_angles = distances.shape
    targets = torch.zeros((batch_size, num_angles), dtype=distances.dtype, device=distances.device)
    valid_mask = torch.isfinite(distances)

    valid_best = best_indices >= 0
    valid_rows = torch.nonzero(valid_best, as_tuple=False).squeeze(1)
    if valid_rows.numel() > 0:
        best_distances = distances[valid_rows, best_indices[valid_rows]]
        useful_mask = valid_mask[valid_rows] & (
            distances[valid_rows] <= (best_distances[:, None] + float(useful_delta_m))
        )
        targets[valid_rows] = useful_mask.float()

    neg_rows = torch.nonzero(~valid_best, as_tuple=False).squeeze(1)
    if neg_rows.numel() > 0:
        valid_mask[neg_rows] = True

    return targets, valid_mask


def compute_useful_bce_loss(
    logits: torch.Tensor,
    useful_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
):
    if useful_targets.numel() == 0:
        return logits.new_tensor(0.0)

    pos_count = useful_targets.sum(dim=1, keepdim=True)
    valid_count = valid_mask.float().sum(dim=1, keepdim=True)
    neg_count = (valid_count - pos_count).clamp_min(0.0)
    pos_weight = torch.where(
        pos_count > 0,
        (neg_count / pos_count.clamp_min(1.0)).clamp(min=1.0, max=10.0),
        torch.ones_like(pos_count),
    )
    weights = torch.where(useful_targets > 0.5, pos_weight, torch.ones_like(useful_targets))
    loss_map = F.binary_cross_entropy_with_logits(logits, useful_targets, reduction="none")
    weight_mask = weights * valid_mask.float()
    if sample_weights is not None:
        weight_mask = weight_mask * sample_weights[:, None].clamp_min(0.0)
    denom = weight_mask.sum().clamp_min(1.0)
    return (loss_map * weight_mask).sum() / denom


def compute_useful_topk_metrics(logits: torch.Tensor, useful_targets: torch.Tensor, topk: int = 3):
    if useful_targets.numel() == 0:
        return 0.0, 0.0
    topk = max(1, min(int(topk), logits.shape[1]))
    top1_idx = torch.argmax(logits, dim=1)
    top1_hit = useful_targets.gather(1, top1_idx[:, None]).squeeze(1) > 0.5
    topk_idx = torch.topk(logits, k=topk, dim=1).indices
    topk_hit = useful_targets.gather(1, topk_idx).max(dim=1).values > 0.5
    return float(top1_hit.float().mean().item()), float(topk_hit.float().mean().item())


def run_epoch(
    model,
    backbone,
    loader,
    optimizer,
    candidate_angles,
    device,
    entropy_weight=0.1,
    ranking_weight=0.5,
    ranking_margin=0.2,
    ranking_gap_m=5.0,
    ranking_gap_scale_m=20.0,
    ce_weight=0.5,
    ce_gap_m=5.0,
    ce_entropy_max=0.8,
    supervision_mode="posterior",
    useful_delta_m=5.0,
    train_backbone=False,
    train=True,
):
    total_loss = 0.0
    total_kl = 0.0
    total_entropy = 0.0
    total_rank = 0.0
    total_ce = 0.0
    total_acc = 0.0
    total_useful_top1 = 0.0
    total_useful_top3 = 0.0
    total_count = 0
    total_rank_pairs = 0
    total_ce_samples = 0

    model.train(train)
    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for query_img, gallery_img, target_probs, distances, _, _, best_index, distance_gap, target_entropy_scalar, pair_weight in iterator:
        query_img = query_img.to(device=device, non_blocking=True)
        gallery_img = gallery_img.to(device=device, non_blocking=True)
        target_probs = target_probs.to(device=device, non_blocking=True)
        distances = distances.to(device=device, non_blocking=True)
        best_index = best_index.to(device=device, non_blocking=True, dtype=torch.long)
        distance_gap = distance_gap.to(device=device, non_blocking=True, dtype=torch.float32)
        target_entropy_scalar = target_entropy_scalar.to(device=device, non_blocking=True, dtype=torch.float32)
        pair_weight = pair_weight.to(device=device, non_blocking=True, dtype=torch.float32)

        with torch.set_grad_enabled(bool(train and train_backbone)):
            gallery_map = backbone.extract_feature_map(gallery_img, branch="img2")
            query_map = backbone.extract_feature_map(query_img, branch="img1")

        logits = model(gallery_map, query_map, candidate_angles)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        target_entropy = compute_entropy(target_probs)
        pred_entropy = compute_entropy(probs)
        useful_targets, useful_valid_mask = build_useful_targets(
            distances=distances,
            best_indices=best_index,
            useful_delta_m=useful_delta_m,
        )

        if str(supervision_mode).lower() == "useful_bce":
            loss_kl = logits.new_tensor(0.0)
            loss_entropy = logits.new_tensor(0.0)
            loss_rank = logits.new_tensor(0.0)
            loss_ce = compute_useful_bce_loss(
                logits=logits,
                useful_targets=useful_targets,
                valid_mask=useful_valid_mask,
                sample_weights=pair_weight,
            )
            pair_count = 0
            ce_sample_count = 0
        else:
            ranking_mask, ranking_weights = build_ranking_targets(
                distances=distances,
                best_indices=best_index,
                ranking_gap_m=ranking_gap_m,
                ranking_gap_scale_m=ranking_gap_scale_m,
            )

            loss_kl = F.kl_div(log_probs, target_probs, reduction="batchmean")
            loss_entropy = F.l1_loss(pred_entropy, target_entropy)
            loss_rank, pair_count = compute_ranking_loss(
                logits=logits,
                best_indices=best_index,
                informative_mask=ranking_mask,
                weights=ranking_weights,
                ranking_margin=ranking_margin,
            )
            loss_ce, ce_sample_count = compute_ce_loss(
                logits=logits,
                best_indices=best_index,
                distance_gaps=distance_gap,
                target_entropies=target_entropy_scalar,
                ce_gap_m=ce_gap_m,
                ce_entropy_max=ce_entropy_max,
            )
        loss = (
            loss_kl
            + float(entropy_weight) * loss_entropy
            + float(ranking_weight) * loss_rank
            + float(ce_weight) * loss_ce
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = int(query_img.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_kl += float(loss_kl.item()) * batch_size
        total_entropy += float(loss_entropy.item()) * batch_size
        total_rank += float(loss_rank.item()) * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_acc += compute_top1_angle_accuracy(logits.detach(), target_probs) * batch_size
        useful_top1, useful_top3 = compute_useful_topk_metrics(logits.detach(), useful_targets.detach(), topk=3)
        total_useful_top1 += useful_top1 * batch_size
        total_useful_top3 += useful_top3 * batch_size
        total_count += batch_size
        total_rank_pairs += pair_count
        total_ce_samples += ce_sample_count

    total_count = max(total_count, 1)
    return {
        "loss": total_loss / total_count,
        "kl": total_kl / total_count,
        "entropy": total_entropy / total_count,
        "rank": total_rank / total_count,
        "ce": total_ce / total_count,
        "top1_angle_acc": total_acc / total_count,
        "useful_top1_hit": total_useful_top1 / total_count,
        "useful_top3_coverage": total_useful_top3 / total_count,
        "rank_pairs": total_rank_pairs,
        "ce_samples": total_ce_samples,
    }


def summarize_teacher(records):
    if not records:
        return {}
    entropies = []
    best_probs = []
    prob_margins = []
    distance_gaps = []
    best_distances = []
    informative = 0
    for record in records:
        probs = torch.tensor(record["target_probs"], dtype=torch.float32)
        entropies.append(float(compute_entropy(probs.unsqueeze(0))[0].item()))
        sorted_probs = torch.sort(probs, descending=True).values
        best_probs.append(float(sorted_probs[0].item()))
        prob_margins.append(float((sorted_probs[0] - sorted_probs[1]).item()))
        best_distance = float(record.get("best_distance_m", float("inf")))
        if math.isfinite(best_distance):
            best_distances.append(best_distance)
        gap = float(record.get("distance_gap_m", float("inf")))
        if math.isfinite(gap):
            distance_gaps.append(gap)
            if gap >= 5.0:
                informative += 1
    return {
        "count": int(len(records)),
        "entropy_mean": float(sum(entropies) / len(entropies)),
        "best_prob_mean": float(sum(best_probs) / len(best_probs)),
        "prob_margin_mean": float(sum(prob_margins) / len(prob_margins)),
        "best_distance_mean_m": float(sum(best_distances) / len(best_distances)) if best_distances else float("nan"),
        "distance_gap_mean_m": float(sum(distance_gaps) / len(distance_gaps)) if distance_gaps else float("nan"),
        "distance_gap_ge_5m_count": int(informative),
    }


def summarize_pair_weights(records, pair_weight_mode: str, pair_weight_center_m: float, pair_weight_scale_m: float):
    if not records:
        return {}
    dataset = VOPDataset(
        records=[],
        transform=None,
        neg_prob=0.0,
        pair_weight_mode=pair_weight_mode,
        pair_weight_center_m=pair_weight_center_m,
        pair_weight_scale_m=pair_weight_scale_m,
    )
    weights = [dataset.compute_pair_weight(record) for record in records]
    arr = np.asarray(weights, dtype=np.float64)
    return {
        "mode": str(pair_weight_mode),
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
        "ge_0.5_ratio": float(np.mean(arr >= 0.5)) if arr.size else float("nan"),
        "ge_0.8_ratio": float(np.mean(arr >= 0.8)) if arr.size else float("nan"),
    }


def filter_teacher_records(records, filter_entropy_max: float, filter_gap_m: float, filter_best_distance_max: float):
    kept = []
    entropy_max = float(filter_entropy_max)
    gap_min = float(filter_gap_m)
    best_distance_max = float(filter_best_distance_max)
    for record in records:
        probs = torch.tensor(record["target_probs"], dtype=torch.float32)
        entropy = float(compute_entropy(probs.unsqueeze(0))[0].item())
        gap = float(record.get("distance_gap_m", float("inf")))
        best_distance = float(record.get("best_distance_m", float("inf")))
        if entropy > entropy_max:
            continue
        if math.isfinite(gap) and gap < gap_min:
            continue
        if not math.isfinite(gap) and gap_min > 0.0:
            continue
        if math.isfinite(best_distance_max):
            if (not math.isfinite(best_distance)) or best_distance >= best_distance_max:
                continue
        kept.append(record)
    return kept


def unfreeze_backbone_scope(backbone_model, scope: str):
    scope = str(scope).strip().lower()
    if scope == "none":
        return []
    if scope != "last_block":
        raise ValueError(f"Unsupported partial unfreeze scope: {scope}")
    if not hasattr(backbone_model, "blocks") or len(backbone_model.blocks) <= 0:
        raise AttributeError("Backbone does not expose blocks for last_block unfreezing.")

    unfrozen_names = []
    block_idx = len(backbone_model.blocks) - 1
    for name, param in backbone_model.blocks[-1].named_parameters():
        param.requires_grad = True
        unfrozen_names.append(f"blocks.{block_idx}.{name}")
    return unfrozen_names


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    teacher_cache = torch.load(args.teacher_cache, map_location="cpu")
    records = list(teacher_cache["records"])
    if not records:
        raise RuntimeError("Teacher cache is empty.")
    print(f"Teacher summary (raw): {teacher_cache.get('summary', summarize_teacher(records))}")

    filtered_records = filter_teacher_records(
        records,
        filter_entropy_max=args.filter_entropy_max,
        filter_gap_m=args.filter_gap_m,
        filter_best_distance_max=args.filter_best_distance_max,
    )
    if not filtered_records:
        raise RuntimeError(
            "Teacher filtering removed all samples. "
            f"filter_entropy_max={args.filter_entropy_max}, "
            f"filter_gap_m={args.filter_gap_m}, "
            f"filter_best_distance_max={args.filter_best_distance_max}"
        )
    if len(filtered_records) != len(records):
        print(
            "Teacher filter applied: "
            f"kept {len(filtered_records)}/{len(records)} "
            f"(entropy<={args.filter_entropy_max}, gap>={args.filter_gap_m}m, "
            f"best_distance<{args.filter_best_distance_max}m)"
        )
        print(f"Teacher summary (filtered): {summarize_teacher(filtered_records)}")
    records = filtered_records
    print(
        "Pair-weight summary: "
        f"{summarize_pair_weights(records, args.pair_weight_mode, args.pair_weight_center_m, args.pair_weight_scale_m)}"
    )

    candidate_angles = teacher_cache.get("records", [])[0]["candidate_angles_deg"]
    expected_angles = build_rotation_angle_list(float(teacher_cache["rotate_step"]))
    if candidate_angles != expected_angles:
        raise ValueError(f"Unexpected candidate angles in teacher cache: {candidate_angles}")

    val_count = max(1, int(len(records) * float(args.val_ratio)))
    random.shuffle(records)
    val_records = records[:val_count]
    train_records = records[val_count:]
    if not train_records:
        train_records = val_records

    backbone = DesModel(args.model, pretrained=True, img_size=args.img_size, share_weights=True)
    if args.checkpoint_start:
        state_dict = torch.load(args.checkpoint_start, map_location="cpu")
        backbone.load_state_dict(state_dict, strict=False)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone_module = backbone.model if getattr(backbone, "share_weights", False) else backbone.model1
    unfrozen_names = unfreeze_backbone_scope(backbone_module, args.partial_unfreeze)
    train_backbone = len(unfrozen_names) > 0
    if train_backbone:
        print(f"Partial unfreeze: {args.partial_unfreeze} -> {len(unfrozen_names)} parameter tensors")
        print(f"Partial unfreeze sample params: {unfrozen_names[:4]}")
    else:
        print("Partial unfreeze: none")

    data_config = backbone.get_config()
    val_transforms = build_eval_transform(args.img_size, mean=data_config["mean"], std=data_config["std"])
    train_dataset = VOPDataset(
        train_records,
        transform=val_transforms,
        neg_prob=args.neg_prob,
        pair_weight_mode=args.pair_weight_mode,
        pair_weight_center_m=args.pair_weight_center_m,
        pair_weight_scale_m=args.pair_weight_scale_m,
    )
    val_dataset = VOPDataset(
        val_records,
        transform=val_transforms,
        neg_prob=0.0,
        pair_weight_mode=args.pair_weight_mode,
        pair_weight_center_m=args.pair_weight_center_m,
        pair_weight_scale_m=args.pair_weight_scale_m,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    sample_query, sample_gallery, _, _, _, _, _, _, _, _ = train_dataset[0]
    with torch.no_grad():
        sample_map = backbone.extract_feature_map(sample_gallery.unsqueeze(0).to(device), branch="img2")
    vop = VisualOrientationPosterior(in_channels=int(sample_map.shape[1]), hidden_dim=args.hidden_dim).to(device)
    trainable_params = list(vop.parameters()) + [param for param in backbone.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            vop,
            backbone,
            train_loader,
            optimizer,
            candidate_angles,
            device,
            entropy_weight=args.entropy_weight,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            ranking_gap_m=args.ranking_gap_m,
            ranking_gap_scale_m=args.ranking_gap_scale_m,
            ce_weight=args.ce_weight,
            ce_gap_m=args.ce_gap_m,
            ce_entropy_max=args.ce_entropy_max,
            supervision_mode=args.supervision_mode,
            useful_delta_m=args.useful_delta_m,
            train_backbone=train_backbone,
            train=True,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                vop,
                backbone,
                val_loader,
                optimizer,
                candidate_angles,
                device,
                entropy_weight=args.entropy_weight,
                ranking_weight=args.ranking_weight,
                ranking_margin=args.ranking_margin,
                ranking_gap_m=args.ranking_gap_m,
                ranking_gap_scale_m=args.ranking_gap_scale_m,
                ce_weight=args.ce_weight,
                ce_gap_m=args.ce_gap_m,
                ce_entropy_max=args.ce_entropy_max,
                supervision_mode=args.supervision_mode,
                useful_delta_m=args.useful_delta_m,
                train_backbone=False,
                train=False,
            )
        print(
            f"epoch={epoch} "
            f"train_loss={train_stats['loss']:.4f} train_kl={train_stats['kl']:.4f} train_rank={train_stats['rank']:.4f} train_ce={train_stats['ce']:.4f} "
            f"train_acc={train_stats['top1_angle_acc']:.4f} train_useful_top1={train_stats['useful_top1_hit']:.4f} train_useful_top3={train_stats['useful_top3_coverage']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_kl={val_stats['kl']:.4f} val_rank={val_stats['rank']:.4f} val_ce={val_stats['ce']:.4f} "
            f"val_acc={val_stats['top1_angle_acc']:.4f} val_useful_top1={val_stats['useful_top1_hit']:.4f} val_useful_top3={val_stats['useful_top3_coverage']:.4f} "
            f"val_rank_pairs={val_stats['rank_pairs']} val_ce_samples={val_stats['ce_samples']}"
        )
        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            best_state = {k: v.detach().cpu() for k, v in vop.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in vop.state_dict().items()}

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "in_channels": int(sample_map.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "candidate_angles_deg": candidate_angles,
            "rotate_step": float(teacher_cache["rotate_step"]),
            "model": args.model,
            "checkpoint_start": args.checkpoint_start,
            "img_size": int(args.img_size),
            "ranking_weight": float(args.ranking_weight),
            "ranking_margin": float(args.ranking_margin),
            "ranking_gap_m": float(args.ranking_gap_m),
            "ranking_gap_scale_m": float(args.ranking_gap_scale_m),
            "ce_weight": float(args.ce_weight),
            "ce_gap_m": float(args.ce_gap_m),
            "ce_entropy_max": float(args.ce_entropy_max),
            "filter_entropy_max": float(args.filter_entropy_max),
            "filter_gap_m": float(args.filter_gap_m),
            "filter_best_distance_max": float(args.filter_best_distance_max),
            "filtered_record_count": int(len(records)),
            "partial_unfreeze": args.partial_unfreeze,
            "unfrozen_names": unfrozen_names,
            "supervision_mode": str(args.supervision_mode),
            "useful_delta_m": float(args.useful_delta_m),
            "pair_weight_mode": str(args.pair_weight_mode),
            "pair_weight_center_m": float(args.pair_weight_center_m),
            "pair_weight_scale_m": float(args.pair_weight_scale_m),
        },
        args.output_path,
    )
    print(f"Saved VOP checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
