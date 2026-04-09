import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_angle_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def build_rotation_angle_list(step_deg: float) -> List[float]:
    step_deg = abs(float(step_deg))
    if step_deg <= 1e-6 or step_deg >= 360.0:
        return [0.0]
    n_steps = int(math.floor((360.0 - 1e-6) / step_deg)) + 1
    return [normalize_angle_deg(round(i * step_deg, 6)) for i in range(n_steps)]


def compute_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    denom = math.log(max(int(probs.shape[-1]), 2))
    return -(probs * probs.log()).sum(dim=-1) / denom


def compute_resultant_length(probs: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    angles_rad = torch.deg2rad(angles_deg.to(device=probs.device, dtype=probs.dtype))
    cos_term = torch.cos(angles_rad)[None, :]
    sin_term = torch.sin(angles_rad)[None, :]
    mean_cos = (probs * cos_term).sum(dim=-1)
    mean_sin = (probs * sin_term).sum(dim=-1)
    return torch.sqrt(mean_cos.square() + mean_sin.square())


def circular_mean_deg(probs: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    angles_rad = torch.deg2rad(angles_deg.to(device=probs.device, dtype=probs.dtype))
    mean_cos = (probs * torch.cos(angles_rad)[None, :]).sum(dim=-1)
    mean_sin = (probs * torch.sin(angles_rad)[None, :]).sum(dim=-1)
    angles = torch.rad2deg(torch.atan2(mean_sin, mean_cos))
    return ((angles + 180.0) % 360.0) - 180.0


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def rotate_feature_map(feature_map: torch.Tensor, angle_deg: float) -> torch.Tensor:
    if abs(float(angle_deg)) < 1e-6:
        return feature_map
    bsz, _, height, width = feature_map.shape
    dtype = feature_map.dtype
    device = feature_map.device
    theta = math.radians(float(angle_deg))
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
        ],
        dtype=dtype,
        device=device,
    ).unsqueeze(0).repeat(bsz, 1, 1)
    grid = F.affine_grid(rotation, feature_map.size(), align_corners=False)
    return F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class VisualOrientationPosterior(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.gallery_proj = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True),
        )

    def encode(self, gallery_map: torch.Tensor, query_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gallery_map = F.normalize(self.gallery_proj(gallery_map), dim=1)
        query_map = F.normalize(self.query_proj(query_map), dim=1)
        return gallery_map, query_map

    def forward(
        self,
        gallery_map: torch.Tensor,
        query_map: torch.Tensor,
        candidate_angles_deg: Sequence[float],
    ) -> torch.Tensor:
        gallery_map, query_map = self.encode(gallery_map, query_map)
        logits = []
        for angle_deg in candidate_angles_deg:
            query_rot = rotate_feature_map(query_map, float(angle_deg))
            pair_map = torch.cat(
                [
                    gallery_map,
                    query_rot,
                    gallery_map * query_rot,
                    torch.abs(gallery_map - query_rot),
                ],
                dim=1,
            )
            logits.append(self.head(pair_map).mean(dim=(2, 3)).squeeze(1))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def predict_posterior(
        self,
        retrieval_model: nn.Module,
        gallery_img: torch.Tensor,
        query_img: torch.Tensor,
        candidate_angles_deg: Sequence[float],
        device: str,
        gallery_branch: str = "img2",
        query_branch: str = "img1",
    ) -> Dict[str, object]:
        retrieval_model = _unwrap_model(retrieval_model)
        self.eval()
        gallery_img = gallery_img.to(device=device, non_blocking=True).unsqueeze(0)
        query_img = query_img.to(device=device, non_blocking=True).unsqueeze(0)
        gallery_map = retrieval_model.extract_feature_map(gallery_img, branch=gallery_branch)
        query_map = retrieval_model.extract_feature_map(query_img, branch=query_branch)
        logits = self(gallery_map, query_map, candidate_angles_deg)
        probs = torch.softmax(logits, dim=-1)
        angles_tensor = torch.tensor(candidate_angles_deg, dtype=probs.dtype, device=probs.device)
        top_idx = int(torch.argmax(probs[0]).item())
        entropy = float(compute_entropy(probs)[0].item())
        concentration = float(compute_resultant_length(probs, angles_tensor)[0].item())
        mean_angle = float(circular_mean_deg(probs, angles_tensor)[0].item())
        return {
            "candidate_angles_deg": [float(angle) for angle in candidate_angles_deg],
            "logits": logits[0].detach().cpu().tolist(),
            "probs": probs[0].detach().cpu().tolist(),
            "top_index": top_idx,
            "top_angle_deg": float(candidate_angles_deg[top_idx]),
            "top_prob": float(probs[0, top_idx].item()),
            "entropy": entropy,
            "concentration": concentration,
            "mean_angle_deg": mean_angle,
        }


def select_angle_result_with_vop(
    angle_results: Sequence[Dict[str, object]],
    posterior: Dict[str, object],
    mode: str = "fusion",
    fusion_weight: float = 0.5,
) -> Optional[Dict[str, object]]:
    if not angle_results:
        return None

    angle_to_prob = {}
    for angle_deg, prob in zip(posterior.get("candidate_angles_deg", []), posterior.get("probs", [])):
        angle_to_prob[round(normalize_angle_deg(float(angle_deg)), 6)] = float(prob)

    valid_results = []
    for angle_result in angle_results:
        if str(angle_result.get("status", "")) != "ok":
            continue
        if angle_result.get("homography") is None:
            continue
        angle_key = round(normalize_angle_deg(float(angle_result.get("rot_angle", 0.0))), 6)
        prior_prob = angle_to_prob.get(angle_key)
        if prior_prob is None:
            continue
        candidate = dict(angle_result)
        candidate["vop_prob"] = prior_prob
        valid_results.append(candidate)

    if not valid_results:
        return None

    if mode == "single":
        return max(valid_results, key=lambda item: (float(item["vop_prob"]), float(item.get("inliers", 0.0))))

    geometry_scores = []
    for item in valid_results:
        score = item.get("score")
        if score is None:
            score = float(item.get("inliers", 0.0)) + 100.0 * float(item.get("ratio", 0.0))
        geometry_scores.append(float(score))
    geo_tensor = torch.tensor(geometry_scores, dtype=torch.float32)
    if geo_tensor.numel() <= 1 or float(geo_tensor.max().item() - geo_tensor.min().item()) < 1e-6:
        geo_norm = torch.ones_like(geo_tensor)
    else:
        geo_norm = (geo_tensor - geo_tensor.min()) / (geo_tensor.max() - geo_tensor.min())

    prior_tensor = torch.tensor([float(item["vop_prob"]) for item in valid_results], dtype=torch.float32)
    fusion_tensor = (1.0 - float(fusion_weight)) * geo_norm + float(fusion_weight) * prior_tensor
    best_index = int(torch.argmax(fusion_tensor).item())
    best_item = dict(valid_results[best_index])
    best_item["fusion_score"] = float(fusion_tensor[best_index].item())
    best_item["geometry_score_norm"] = float(geo_norm[best_index].item())
    return best_item


def load_vop_checkpoint(checkpoint_path: str, device: str = "cpu") -> VisualOrientationPosterior:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state_dict.")
    model = VisualOrientationPosterior(
        in_channels=int(checkpoint["in_channels"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.candidate_angles_deg = [float(angle) for angle in checkpoint.get("candidate_angles_deg", [])]
    model = model.to(device)
    model.eval()
    return model
