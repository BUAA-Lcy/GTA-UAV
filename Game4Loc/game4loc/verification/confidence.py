import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


CONFIDENCE_FEATURE_NAMES: Tuple[str, ...] = (
    "candidate_rank",
    "candidate_prob",
    "top_prob",
    "entropy",
    "concentration",
    "log_retained_matches",
    "log_inliers",
    "inlier_ratio",
    "geometry_score",
    "identity_h_fallback",
    "fallback_to_center",
    "out_of_bounds",
    "projection_invalid",
)


def compute_geometry_score(inliers: float, inlier_ratio: float, offset: Optional[float] = 25.0) -> float:
    if offset is None:
        return float(inliers) + 100.0 * float(inlier_ratio)
    return float(inlier_ratio) * max(float(inliers) - float(offset), 0.0)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _as_bool_float(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def build_confidence_candidate_record(
    *,
    candidate_rank: int,
    candidate_angle_deg: float,
    candidate_prob: float,
    top_prob: float,
    entropy: float,
    concentration: float,
    match_info: Optional[Dict[str, object]],
    coarse_error_m: Optional[float],
    candidate_error_m: Optional[float],
    score_offset: Optional[float] = 25.0,
) -> Dict[str, object]:
    info = dict(match_info or {})
    retained_matches = _as_float(info.get("n_kept", 0.0), default=0.0)
    inliers = _as_float(info.get("inliers", 0.0), default=0.0)
    inlier_ratio = _as_float(info.get("inlier_ratio", 0.0), default=0.0)
    if retained_matches > 0.0 and inlier_ratio <= 0.0:
        inlier_ratio = float(inliers) / float(max(retained_matches, 1.0))
    geometry_score = compute_geometry_score(inliers, inlier_ratio, offset=score_offset)

    record = {
        "candidate_rank": int(candidate_rank),
        "candidate_angle_deg": _as_float(candidate_angle_deg, default=0.0),
        "candidate_prob": _as_float(candidate_prob, default=0.0),
        "top_prob": _as_float(top_prob, default=0.0),
        "entropy": _as_float(entropy, default=1.0),
        "concentration": _as_float(concentration, default=0.0),
        "retained_matches": retained_matches,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "geometry_score": geometry_score,
        "identity_h_fallback": _as_bool_float(info.get("identity_h_fallback", False)),
        "fallback_to_center": _as_bool_float(info.get("fallback_to_center", False)),
        "out_of_bounds": _as_bool_float(info.get("out_of_bounds", False)),
        "projection_invalid": _as_bool_float(info.get("projection_invalid", False)),
        "coarse_error_m": None if coarse_error_m is None else _as_float(coarse_error_m, default=0.0),
        "candidate_error_m": None if candidate_error_m is None else _as_float(candidate_error_m, default=0.0),
    }
    record["label_improve_over_coarse"] = bool(
        record["coarse_error_m"] is not None
        and record["candidate_error_m"] is not None
        and float(record["candidate_error_m"]) + 1e-6 < float(record["coarse_error_m"])
    )
    return record


def _feature_vector_from_record(record: Dict[str, object], feature_names: Sequence[str]) -> np.ndarray:
    feature_map = {
        "candidate_rank": _as_float(record.get("candidate_rank", 0.0), default=0.0),
        "candidate_prob": _as_float(record.get("candidate_prob", 0.0), default=0.0),
        "top_prob": _as_float(record.get("top_prob", 0.0), default=0.0),
        "entropy": _as_float(record.get("entropy", 1.0), default=1.0),
        "concentration": _as_float(record.get("concentration", 0.0), default=0.0),
        "log_retained_matches": math.log1p(max(_as_float(record.get("retained_matches", 0.0), default=0.0), 0.0)),
        "log_inliers": math.log1p(max(_as_float(record.get("inliers", 0.0), default=0.0), 0.0)),
        "inlier_ratio": _as_float(record.get("inlier_ratio", 0.0), default=0.0),
        "geometry_score": _as_float(record.get("geometry_score", 0.0), default=0.0),
        "identity_h_fallback": _as_float(record.get("identity_h_fallback", 0.0), default=0.0),
        "fallback_to_center": _as_float(record.get("fallback_to_center", 0.0), default=0.0),
        "out_of_bounds": _as_float(record.get("out_of_bounds", 0.0), default=0.0),
        "projection_invalid": _as_float(record.get("projection_invalid", 0.0), default=0.0),
    }
    return np.asarray([feature_map[name] for name in feature_names], dtype=np.float32)


@dataclass
class LinearConfidenceVerifier:
    feature_names: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weight: np.ndarray
    bias: float
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float32).reshape(-1)
        self.feature_std = np.asarray(self.feature_std, dtype=np.float32).reshape(-1)
        self.weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        self.bias = float(self.bias)
        self.threshold = float(self.threshold)

    def _normalize(self, feats: np.ndarray) -> np.ndarray:
        std = np.where(np.abs(self.feature_std) < 1e-6, 1.0, self.feature_std)
        return (feats - self.feature_mean) / std

    def score_records(self, records: Sequence[Dict[str, object]]) -> np.ndarray:
        if len(records) == 0:
            return np.zeros((0,), dtype=np.float32)
        feats = np.stack([_feature_vector_from_record(record, self.feature_names) for record in records], axis=0)
        feats = self._normalize(feats)
        logits = feats @ self.weight + self.bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.astype(np.float32)

    def score_record(self, record: Dict[str, object]) -> float:
        scores = self.score_records([record])
        return float(scores[0]) if scores.size > 0 else 0.0


def select_candidate_with_confidence(
    records: Sequence[Dict[str, object]],
    verifier: LinearConfidenceVerifier,
) -> Optional[Dict[str, object]]:
    if len(records) == 0:
        return None
    scores = verifier.score_records(records)
    best_index = int(np.argmax(scores))
    selected = dict(records[best_index])
    selected["confidence_score"] = float(scores[best_index])
    selected["confidence_accept"] = bool(float(scores[best_index]) >= float(verifier.threshold))
    return selected


def save_confidence_verifier(
    path: str,
    *,
    feature_names: Sequence[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    weight: np.ndarray,
    bias: float,
    threshold: float = 0.5,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "type": "linear_confidence_verifier",
        "feature_names": [str(name) for name in feature_names],
        "feature_mean": np.asarray(feature_mean, dtype=np.float32).reshape(-1).tolist(),
        "feature_std": np.asarray(feature_std, dtype=np.float32).reshape(-1).tolist(),
        "weight": np.asarray(weight, dtype=np.float32).reshape(-1).tolist(),
        "bias": float(bias),
        "threshold": float(threshold),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_confidence_verifier(path: str) -> LinearConfidenceVerifier:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("type") != "linear_confidence_verifier":
        raise ValueError(f"Unsupported confidence verifier type in {path}: {checkpoint.get('type')}")
    return LinearConfidenceVerifier(
        feature_names=tuple(checkpoint["feature_names"]),
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        weight=np.asarray(checkpoint["weight"], dtype=np.float32),
        bias=float(checkpoint["bias"]),
        threshold=float(checkpoint.get("threshold", 0.5)),
    )
