#!/usr/bin/env python3
"""Train the SIVR token-sequence classifier on collected Qwen trajectories.

The official SIVR ``var`` input is a variable-length sequence whose token
features are [circular variance, covariance log-determinant, output entropy].
This adaptation keeps that representation and the released Transformer
dimensions, but evaluates it with grouped out-of-fold predictions so that
multiple draws of the same grounding example cannot cross a fold boundary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for module_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from ceor.io import atomic_json as _write_json


def _write_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[np.ndarray], labels: Sequence[int]):
        self.sequences = [torch.as_tensor(value, dtype=torch.float32) for value in sequences]
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def collate_sequences(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [int(sequence.shape[0]) for sequence, _ in batch]
    width = int(batch[0][0].shape[1])
    values = torch.zeros(len(batch), max(lengths), width, dtype=torch.float32)
    mask = torch.ones(len(batch), max(lengths), dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.float32)
    for index, (sequence, label) in enumerate(batch):
        values[index, : sequence.shape[0]] = sequence
        mask[index, : sequence.shape[0]] = False
        labels[index] = label
    return values, mask, labels


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence exceeds positional-encoding maximum length.")
        return values + self.encoding[:, : values.shape[1]]


class AttentionPool(nn.Module):
    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        scores = self.scorer(hidden).squeeze(-1).masked_fill(pad_mask, -1e9)
        weights = scores.softmax(dim=1)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)


class SIVRTransformerClassifier(nn.Module):
    """Released SIVR dimensions with a registered (trainable) attention pool."""

    def __init__(
        self,
        input_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pool: str = "attn",
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_name = str(pool)
        self.attention_pool = AttentionPool(d_model) if pool == "attn" else None
        self.output = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, values: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.position(self.input_projection(values))
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        if self.pool_name == "attn":
            pooled = self.attention_pool(hidden, pad_mask)
        elif self.pool_name == "mean":
            valid = (~pad_mask).unsqueeze(-1)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        else:
            raise ValueError(f"Unknown pool: {self.pool_name!r}.")
        return self.output(pooled).squeeze(-1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Ignoring incomplete JSONL tail at line {line_number}.", file=sys.stderr)
                break
    if not rows:
        raise ValueError(f"No complete records in {path}.")
    return rows


def token_sequence(draw: Mapping[str, Any], view: str) -> np.ndarray:
    trajectory = draw["trajectory"]
    sivr = trajectory["views"][view]["sivr"]
    output = trajectory["output"]
    columns = (
        sivr["circular_variance_per_token"],
        sivr["covariance_logdet_mean_per_token"],
        output["entropy_per_token"],
    )
    lengths = {len(column) for column in columns}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError("SIVR per-token features have inconsistent lengths.")
    values = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("SIVR sequence contains non-finite values.")
    return values


def build_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    view: str,
    label_mode: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    baseline = {
        str(record["sample_id"]): int(record["grounding"]["hallucination"])
        for record in records if record["condition"] == "baseline"
    }
    selected = [record for record in records if record["condition"] == condition]
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    metadata: list[dict[str, Any]] = []
    for record in selected:
        sample_id = str(record["sample_id"])
        for draw in record["draws"]:
            if label_mode == "own_draw":
                label = int(float(draw["accuracy"]) < 0.5)
            elif label_mode == "fixed_baseline_majority":
                if sample_id not in baseline:
                    continue
                label = baseline[sample_id]
            else:
                raise ValueError(f"Unknown label mode: {label_mode!r}.")
            sequences.append(token_sequence(draw, view))
            labels.append(label)
            groups.append(str(record.get("group_id", sample_id)))
            metadata.append({
                "sample_id": sample_id,
                "group_id": str(record.get("group_id", sample_id)),
                "draw": int(draw["draw"]),
                "iou": float(draw["iou"]),
                "accuracy": float(draw["accuracy"]),
            })
    return sequences, np.asarray(labels, dtype=int), np.asarray(groups), metadata


def fit_scaler(sequences: Sequence[np.ndarray], indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([sequences[int(index)] for index in indices], axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def fpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        return float("nan")
    false_positive, true_positive, _ = roc_curve(labels, scores)
    eligible = false_positive[true_positive >= 0.95]
    return float(np.min(eligible)) if eligible.size else 1.0


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if np.unique(labels).size != 2:
        return {key: float("nan") for key in ("accuracy", "auroc", "aupr", "fpr95")}
    return {
        "accuracy": float(accuracy_score(labels, scores >= 0.5)),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": fpr95(labels, scores),
    }


def train_fold(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, std = fit_scaler(sequences, train_indices)
    normalized = [(sequence - mean) / std for sequence in sequences]
    train_data = SequenceDataset(
        [normalized[int(index)] for index in train_indices], labels[train_indices]
    )
    validation_data = SequenceDataset(
        [normalized[int(index)] for index in validation_indices], labels[validation_indices]
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_sequences,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
    )
    model = SIVRTransformerClassifier(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(args.epochs):
        model.train()
        for values, mask, targets in train_loader:
            values, mask, targets = values.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values, mask), targets)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for values, mask, _ in validation_loader:
            probability = torch.sigmoid(model(values.to(device), mask.to(device)))
            predictions.append(probability.cpu().numpy())
    return np.concatenate(predictions)


def grouped_oof(
    sequences: Sequence[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    folds = min(int(args.folds), len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two distinct example groups are required.")
    predictions = np.full(len(labels), np.nan, dtype=float)
    valid_folds = 0
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if np.unique(labels[train_indices]).size != 2:
            continue
        predictions[validation_indices] = train_fold(
            sequences,
            labels,
            train_indices,
            validation_indices,
            args=args,
            seed=int(seed) + fold * 1009,
            device=device,
        )
        valid_folds += 1
    return predictions, valid_folds


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else input_dir / "sivr_sequence_classifier"
    )
    preflight = json.loads((input_dir / "preflight.json").read_text(encoding="utf-8"))
    available_views = list(preflight["identity"]["layer_views"])
    views = available_views if args.views == ["all"] else list(dict.fromkeys(args.views))
    invalid = sorted(set(views) - set(available_views))
    if invalid:
        raise ValueError(f"Unknown views {invalid}; available views: {available_views}.")
    records = read_jsonl(input_dir / "records.jsonl")
    available_conditions = sorted({str(record["condition"]) for record in records})
    conditions = available_conditions if args.conditions == ["all"] else list(dict.fromkeys(args.conditions))
    invalid_conditions = sorted(set(conditions) - set(available_conditions))
    if invalid_conditions:
        raise ValueError(f"Unknown conditions: {invalid_conditions}.")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for view in views:
            sequences, labels, groups, metadata = build_examples(
                records,
                condition=condition,
                view=view,
                label_mode=args.label_mode,
            )
            if not sequences or np.unique(labels).size != 2:
                continue
            for seed in args.seeds:
                predictions, valid_folds = grouped_oof(
                    sequences, labels, groups, args=args, seed=int(seed), device=device
                )
                valid = np.isfinite(predictions)
                result = metrics(labels[valid], predictions[valid])
                rows.append({
                    "condition": condition,
                    "view": view,
                    "seed": int(seed),
                    "label_mode": args.label_mode,
                    "num_sequences": int(valid.sum()),
                    "num_groups": int(len(np.unique(groups[valid]))),
                    "valid_folds": int(valid_folds),
                    **result,
                })
                for index in np.flatnonzero(valid):
                    prediction_rows.append({
                        "condition": condition,
                        "view": view,
                        "seed": int(seed),
                        "label": int(labels[index]),
                        "score": float(predictions[index]),
                        **metadata[int(index)],
                    })
    aggregate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for view in views:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["view"] == view
            ]
            if selected:
                aggregate_rows.append({
                    "condition": condition,
                    "view": view,
                    "num_seeds": len(selected),
                    **{
                        f"{metric}_{statistic}": float(function([
                            row[metric] for row in selected
                        ]))
                        for metric in ("accuracy", "auroc", "aupr", "fpr95")
                        for statistic, function in (("mean", np.mean), ("std", np.std))
                    },
                })
        for seed in args.seeds:
            seed_rows = [
                row for row in rows
                if row["condition"] == condition and int(row["seed"]) == int(seed)
            ]
            by_view = {str(row["view"]): row for row in seed_rows}
            if "key3" not in by_view:
                continue
            random_rows = [
                row for view, row in by_view.items() if view.startswith("random3_")
            ]
            for metric in ("accuracy", "auroc", "aupr", "fpr95"):
                key_value = float(by_view["key3"][metric])
                full_value = (
                    float(by_view["full"][metric]) if "full" in by_view else float("nan")
                )
                random_values = np.asarray(
                    [float(row[metric]) for row in random_rows], dtype=float
                )
                comparison_rows.append({
                    "condition": condition,
                    "seed": int(seed),
                    "metric": metric,
                    "full": full_value,
                    "key3": key_value,
                    "key3_minus_full": key_value - full_value,
                    "num_random3": len(random_values),
                    "random3_mean": (
                        float(np.mean(random_values)) if len(random_values) else float("nan")
                    ),
                    "key3_minus_random3_mean": (
                        key_value - float(np.mean(random_values))
                        if len(random_values) else float("nan")
                    ),
                    "key3_better_random_fraction": (
                        float(np.mean(random_values >= key_value))
                        if metric == "fpr95" and len(random_values)
                        else float(np.mean(random_values <= key_value))
                        if len(random_values) else float("nan")
                    ),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "metrics_seed_summary.csv", aggregate_rows)
    _write_csv(output_dir / "key_vs_random.csv", comparison_rows)
    _write_csv(output_dir / "oof_predictions.csv", prediction_rows)
    summary = {
        "method": "SIVR var-only token-sequence Transformer",
        "official_dimensions": {
            "input_features": ["circular_variance", "covariance_logdet_mean", "entropy"],
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "pool": args.pool,
        },
        "evaluation": "grouped out-of-fold",
        "pooling_note": (
            "The released repository constructs AttnPool inside forward; this implementation "
            "registers it once so its parameters are trainable."
        ),
        "input_dir": str(input_dir),
        "conditions": conditions,
        "views": views,
        "seeds": list(map(int, args.seeds)),
        "metrics": str(output_dir / "metrics.csv"),
        "metrics_seed_summary": str(output_dir / "metrics_seed_summary.csv"),
        "key_vs_random": str(output_dir / "key_vs_random.csv"),
        "oof_predictions": str(output_dir / "oof_predictions.csv"),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--conditions", nargs="+", default=["baseline"])
    parser.add_argument("--views", nargs="+", default=["full", "key3"])
    parser.add_argument(
        "--label-mode",
        choices=("own_draw", "fixed_baseline_majority"),
        default="own_draw",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool", choices=("attn", "mean"), default="attn")
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds>=2, epochs>=1, and batch-size>=1 are required.")
    run(args)


if __name__ == "__main__":
    main()
