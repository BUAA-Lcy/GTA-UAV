import argparse
import json
import math
import os
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from game4loc.verification import (
    CONFIDENCE_FEATURE_NAMES,
    save_confidence_verifier,
)
from game4loc.verification.confidence import _feature_vector_from_record


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight confidence-aware verifier for prior_topk fine localization.")
    parser.add_argument("--dump_path", type=str, required=True, help="JSON dump produced by eval_visloc.py --confidence_dump_path")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the trained verifier checkpoint")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs for the linear verifier")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val_stride", type=int, default=5, help="Use every N-th query for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Acceptance threshold saved into the verifier checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu()
    labels_cpu = labels.detach().cpu()
    preds = (probs >= 0.5).to(torch.float32)
    correct = float((preds == labels_cpu).float().mean().item()) if labels_cpu.numel() > 0 else 0.0
    tp = float(((preds == 1) & (labels_cpu == 1)).sum().item())
    fp = float(((preds == 1) & (labels_cpu == 0)).sum().item())
    fn = float(((preds == 0) & (labels_cpu == 1)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": correct,
        "precision": precision,
        "recall": recall,
        "positive_rate": float(labels_cpu.mean().item()) if labels_cpu.numel() > 0 else 0.0,
    }


def main():
    args = parse_args()
    set_seed(int(args.seed))

    with open(args.dump_path, "r", encoding="utf-8") as f:
        dump_records = json.load(f)

    if not isinstance(dump_records, list) or len(dump_records) == 0:
        raise ValueError(f"No query records found in dump: {args.dump_path}")

    train_queries: List[Dict[str, object]] = []
    val_queries: List[Dict[str, object]] = []
    for idx, query_record in enumerate(dump_records):
        if (idx % max(int(args.val_stride), 2)) == 0:
            val_queries.append(query_record)
        else:
            train_queries.append(query_record)

    if len(train_queries) == 0 or len(val_queries) == 0:
        raise ValueError("Need non-empty train and validation query splits for confidence verifier training.")

    def flatten_records(query_records: Sequence[Dict[str, object]]):
        feature_list = []
        label_list = []
        for query_record in query_records:
            for candidate in query_record.get("candidates", []):
                feature_list.append(_feature_vector_from_record(candidate, CONFIDENCE_FEATURE_NAMES))
                label_list.append(1.0 if bool(candidate.get("label_improve_over_coarse", False)) else 0.0)
        if len(feature_list) == 0:
            raise ValueError("No candidate records available for training.")
        features = np.stack(feature_list, axis=0).astype(np.float32)
        labels = np.asarray(label_list, dtype=np.float32)
        return features, labels

    train_x_np, train_y_np = flatten_records(train_queries)
    val_x_np, val_y_np = flatten_records(val_queries)

    feature_mean = train_x_np.mean(axis=0)
    feature_std = train_x_np.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    train_x = torch.from_numpy((train_x_np - feature_mean) / feature_std)
    train_y = torch.from_numpy(train_y_np)
    val_x = torch.from_numpy((val_x_np - feature_mean) / feature_std)
    val_y = torch.from_numpy(val_y_np)

    model = nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_count = float(train_y.sum().item())
    neg_count = float(train_y.numel() - train_y.sum().item())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_x).squeeze(1)
        train_loss = criterion(train_logits, train_y)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x).squeeze(1)
            val_loss = float(criterion(val_logits, val_y).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "weight": model.weight.detach().cpu().clone().reshape(-1),
                "bias": float(model.bias.detach().cpu().item()),
            }

    if best_state is None:
        raise RuntimeError("Failed to train the confidence verifier.")

    weight = best_state["weight"].numpy()
    bias = float(best_state["bias"])

    train_logits_np = ((train_x_np - feature_mean) / feature_std) @ weight + bias
    val_logits_np = ((val_x_np - feature_mean) / feature_std) @ weight + bias
    train_metrics = _compute_binary_metrics(torch.from_numpy(train_logits_np), torch.from_numpy(train_y_np))
    val_metrics = _compute_binary_metrics(torch.from_numpy(val_logits_np), torch.from_numpy(val_y_np))

    metadata = {
        "dump_path": os.path.abspath(args.dump_path),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_stride": int(args.val_stride),
        "seed": int(args.seed),
        "train_queries": len(train_queries),
        "val_queries": len(val_queries),
        "train_candidates": int(train_y_np.shape[0]),
        "val_candidates": int(val_y_np.shape[0]),
        "train_positive_rate": float(train_y_np.mean()),
        "val_positive_rate": float(val_y_np.mean()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": float(best_val_loss),
    }

    save_confidence_verifier(
        args.output_path,
        feature_names=CONFIDENCE_FEATURE_NAMES,
        feature_mean=feature_mean,
        feature_std=feature_std,
        weight=weight,
        bias=bias,
        threshold=float(args.threshold),
        metadata=metadata,
    )

    print("Saved confidence verifier:", os.path.abspath(args.output_path))
    print("Train queries:", len(train_queries), "Val queries:", len(val_queries))
    print("Train candidates:", int(train_y_np.shape[0]), "Val candidates:", int(val_y_np.shape[0]))
    print("Train positive rate:", float(train_y_np.mean()))
    print("Val positive rate:", float(val_y_np.mean()))
    print("Train metrics:", train_metrics)
    print("Val metrics:", val_metrics)
    print("Best val loss:", float(best_val_loss))
    print("Feature names:", list(CONFIDENCE_FEATURE_NAMES))


if __name__ == "__main__":
    main()
