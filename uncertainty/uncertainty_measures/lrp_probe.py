"""Latent Representation Probing (LRP) for VLM abstention.

Reproduces the three probe designs in Yao et al., *Reading Between the Lines:
Abstaining from VLM-Generated OCR Errors via Latent Representation Probes*
(2025): concatenated probes, visual-focused attention probes, and top-layer
ensembles.  The implementation accepts the paper's soft correctness labels and
returns error probabilities/scores for compatibility with this repository.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .paper_probe_features import (
    LRP_ATTENTION_FEATURE_NAME,
    LRP_HIDDEN_FEATURE_NAME,
    LRP_VISUAL_ATTENTION_FEATURE_NAME,
)


LRP_VARIANTS = (
    "concat_hidden",
    "visual_attention",
    "ensemble_hidden",
    "concat_attention",
    "ensemble_attention",
)


class FourLayerMLP(nn.Module):
    """Four linear layers, matching the paper's MLP probe depth."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim < 1 or len(hidden_dims) != 3 or min(hidden_dims) < 1:
            raise ValueError("LRP MLP needs one positive input and three hidden widths.")
        dimensions = [int(input_dim), *(int(value) for value in hidden_dims)]
        blocks: List[nn.Module] = []
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
            blocks.extend(
                [
                    nn.Linear(input_width, output_width),
                    nn.GELU(),
                    nn.LayerNorm(output_width),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(dimensions[-1], 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("LRP MLP input must be [batch, features].")
        return self.output(self.hidden(values.float())).squeeze(-1)


class LayerwiseMLP(nn.Module):
    """Independent four-layer probes trained jointly for every hidden layer."""

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ):
        super().__init__()
        self.probes = nn.ModuleList(
            [
                FourLayerMLP(input_dim, hidden_dims=hidden_dims, dropout=dropout)
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != len(self.probes):
            raise ValueError("Layerwise LRP input must be [batch, layers, features].")
        return torch.stack(
            [probe(values[:, index, :]) for index, probe in enumerate(self.probes)],
            dim=1,
        )


class AttentionPatternTransformer(nn.Module):
    """Four-block Transformer over variable-length input-token patterns."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim < 1 or model_dim < 1 or num_layers < 1 or num_heads < 1:
            raise ValueError("Invalid LRP attention Transformer dimensions.")
        if model_dim % num_heads:
            raise ValueError("LRP Transformer model_dim must be divisible by num_heads.")
        self.input_projection = nn.Linear(int(input_dim), int(model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(num_heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(model_dim))
        self.output = nn.Linear(int(model_dim), 1)

    @staticmethod
    def _sinusoidal_positions(
        length: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(width, 1))
        )
        encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if width > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )
        return encoding.to(dtype=dtype)

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("LRP attention input must be [batch, tokens, features].")
        hidden = self.input_projection(values.float())
        hidden = hidden + self._sinusoidal_positions(
            hidden.shape[1],
            hidden.shape[2],
            device=hidden.device,
            dtype=hidden.dtype,
        ).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.output(self.norm(pooled)).squeeze(-1)


@dataclass(frozen=True)
class LRPTrainingConfig:
    variants: Tuple[str, ...] = LRP_VARIANTS
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 6
    min_delta: float = 1e-4
    mlp_hidden_dims: Tuple[int, int, int] = (256, 128, 32)
    transformer_model_dim: int = 64
    transformer_layers: int = 4
    transformer_heads: int = 4
    dropout: float = 0.1
    top_k_layers: int = 5
    random_seed: int = 10
    device: str = "auto"

    def validate(self) -> None:
        unknown = sorted(set(self.variants) - set(LRP_VARIANTS))
        if unknown or not self.variants:
            raise ValueError(f"Unknown or empty LRP variants: {unknown}")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid LRP training configuration.")
        if self.weight_decay < 0 or not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Invalid LRP regularization/validation configuration.")
        if self.patience < 0 or self.min_delta < 0 or self.top_k_layers < 1:
            raise ValueError("Invalid LRP early stopping or top-K setting.")
        if len(self.mlp_hidden_dims) != 3 or min(self.mlp_hidden_dims) < 1:
            raise ValueError("LRP MLP needs exactly three positive hidden widths.")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested LRP device `{value}`, but CUDA is unavailable.")
    return device


def _as_finite_tensor(value: Any, *, name: str, sample_index: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Invalid `{name}` in LRP sample {sample_index}.")
    return tensor


@dataclass
class LRPFeatureBatch:
    hidden: torch.Tensor
    attention: List[torch.Tensor]
    visual_attention: torch.Tensor

    @property
    def num_layers(self) -> int:
        return int(self.hidden.shape[1])

    @property
    def num_heads(self) -> int:
        return int(self.visual_attention.shape[2])

    @property
    def hidden_dim(self) -> int:
        return int(self.hidden.shape[2])


def lrp_feature_batch(
    features: Sequence[Mapping[str, Any]],
    *,
    require_visual: bool = True,
) -> LRPFeatureBatch:
    hidden_rows: List[torch.Tensor] = []
    attention_rows: List[torch.Tensor] = []
    visual_rows: List[torch.Tensor] = []
    for sample_index, sample in enumerate(features):
        missing = [
            name
            for name in (
                LRP_HIDDEN_FEATURE_NAME,
                LRP_ATTENTION_FEATURE_NAME,
                LRP_VISUAL_ATTENTION_FEATURE_NAME,
            )
            if sample.get(name) is None
        ]
        if missing:
            raise ValueError(
                f"Missing LRP features {missing} in sample {sample_index}; "
                "regenerate with --collect_lrp_probe."
            )
        hidden = _as_finite_tensor(
            sample[LRP_HIDDEN_FEATURE_NAME],
            name=LRP_HIDDEN_FEATURE_NAME,
            sample_index=sample_index,
        )
        attention = _as_finite_tensor(
            sample[LRP_ATTENTION_FEATURE_NAME],
            name=LRP_ATTENTION_FEATURE_NAME,
            sample_index=sample_index,
        )
        visual = _as_finite_tensor(
            sample[LRP_VISUAL_ATTENTION_FEATURE_NAME],
            name=LRP_VISUAL_ATTENTION_FEATURE_NAME,
            sample_index=sample_index,
        )
        if hidden.ndim != 2 or attention.ndim != 3 or visual.ndim != 2:
            raise ValueError("Unexpected LRP feature rank.")
        if hidden.shape[0] != attention.shape[0] or visual.shape != attention.shape[:2]:
            raise ValueError("LRP layer/head dimensions are inconsistent.")
        if require_visual and sample.get("lrp_visual_attention_available") is False:
            raise ValueError(
                f"LRP sample {sample_index} has no resolved visual tokens; "
                "the visual-focused paper probe would be undefined."
            )
        hidden_rows.append(hidden)
        attention_rows.append(attention)
        visual_rows.append(visual)
    if not hidden_rows:
        raise ValueError("No LRP features were supplied.")
    hidden_shapes = {tuple(value.shape) for value in hidden_rows}
    attention_prefixes = {tuple(value.shape[:2]) for value in attention_rows}
    visual_shapes = {tuple(value.shape) for value in visual_rows}
    if len(hidden_shapes) != 1 or len(attention_prefixes) != 1 or len(visual_shapes) != 1:
        raise ValueError("LRP model dimensions differ across samples.")
    return LRPFeatureBatch(
        hidden=torch.stack(hidden_rows, dim=0),
        attention=attention_rows,
        visual_attention=torch.stack(visual_rows, dim=0),
    )


def _concat_attention_sequences(batch: LRPFeatureBatch) -> List[torch.Tensor]:
    return [
        value.permute(2, 0, 1).reshape(value.shape[2], -1)
        for value in batch.attention
    ]


def _layer_attention_sequences(
    batch: LRPFeatureBatch, layer_index: int
) -> List[torch.Tensor]:
    return [value[layer_index].transpose(0, 1) for value in batch.attention]


def _split_indices(
    error_targets: np.ndarray,
    validation_fraction: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    classes = (error_targets >= 0.5).astype(np.int64)
    counts = np.bincount(classes, minlength=2)
    if error_targets.size < 6 or int(counts.min()) < 2:
        raise ValueError(
            "LRP requires at least six training samples and two samples from "
            "each correctness class."
        )
    fit, holdout = train_test_split(
        np.arange(error_targets.size),
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=classes,
    )
    return np.asarray(fit), np.asarray(holdout)


def abstention_metrics(
    correctness: Sequence[float],
    error_scores: Sequence[float],
    *,
    error_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Eqs. (16)-(19) with soft correctness labels."""
    correct = np.asarray(correctness, dtype=np.float64).reshape(-1)
    errors = np.asarray(error_scores, dtype=np.float64).reshape(-1)
    if correct.shape != errors.shape or correct.size == 0:
        raise ValueError("Abstention correctness and score arrays must align.")
    if not np.isfinite(correct).all() or not np.isfinite(errors).all():
        raise ValueError("Abstention inputs must be finite.")
    correct = np.clip(correct, 0.0, 1.0)
    answer = errors < float(error_threshold)
    abstain = ~answer
    effective_reliability = np.mean((2.0 * correct - 1.0) * answer)
    abstention_accuracy = np.mean(
        answer * correct + abstain * (1.0 - correct)
    )
    reliable_accuracy = (
        float(np.sum(correct * answer) / np.sum(answer))
        if np.any(answer)
        else float("nan")
    )
    abstention_precision = (
        float(np.sum((1.0 - correct) * abstain) / np.sum(abstain))
        if np.any(abstain)
        else float("nan")
    )
    return {
        "effective_reliability": float(effective_reliability),
        "abstention_accuracy": float(abstention_accuracy),
        "reliable_accuracy": reliable_accuracy,
        "abstention_precision": abstention_precision,
        "coverage": float(np.mean(answer)),
        "error_threshold": float(error_threshold),
        "confidence_threshold": float(1.0 - error_threshold),
    }


def select_abstention_error_threshold(
    correctness: Sequence[float], error_scores: Sequence[float]
) -> Tuple[float, Dict[str, float]]:
    """Select tau on a train-only holdout by maximum abstention accuracy."""
    errors = np.asarray(error_scores, dtype=np.float64).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], errors)))
    best_threshold = 0.5
    best_metrics = abstention_metrics(
        correctness, errors, error_threshold=best_threshold
    )
    for threshold in candidates:
        metrics = abstention_metrics(
            correctness, errors, error_threshold=float(threshold)
        )
        if (
            metrics["abstention_accuracy"] > best_metrics["abstention_accuracy"] + 1e-12
            or (
                abs(metrics["abstention_accuracy"] - best_metrics["abstention_accuracy"])
                <= 1e-12
                and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _fixed_loader(
    values: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    config: LRPTrainingConfig,
    *,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.random_seed + seed_offset)
    return DataLoader(
        TensorDataset(values[indices], labels[indices]),
        batch_size=min(config.batch_size, len(indices)),
        shuffle=True,
        generator=generator,
    )


def _sequence_collate(rows):
    sequences, labels = zip(*rows)
    lengths = torch.tensor([value.shape[0] for value in sequences], dtype=torch.long)
    maximum = int(lengths.max().item())
    width = int(sequences[0].shape[1])
    padded = torch.zeros(len(sequences), maximum, width, dtype=torch.float32)
    mask = torch.ones(len(sequences), maximum, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = int(sequence.shape[0])
        padded[index, :length] = sequence
        mask[index, :length] = False
    return padded, mask, torch.stack(labels)


class _SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, indices):
        self.sequences = sequences
        self.labels = labels
        self.indices = list(int(value) for value in indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        return self.sequences[source], self.labels[source]


def _sequence_loader(
    sequences: Sequence[torch.Tensor],
    labels: torch.Tensor,
    indices: np.ndarray,
    config: LRPTrainingConfig,
    *,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.random_seed + seed_offset)
    return DataLoader(
        _SequenceDataset(sequences, labels, indices),
        batch_size=min(config.batch_size, len(indices)),
        shuffle=True,
        generator=generator,
        collate_fn=_sequence_collate,
    )


def _pad_sequences(sequences: Sequence[torch.Tensor]):
    dummy_labels = [torch.tensor(0.0) for _ in sequences]
    values, mask, _ = _sequence_collate(list(zip(sequences, dummy_labels)))
    return values, mask


def _fit_fixed(
    model: nn.Module,
    train_values: torch.Tensor,
    labels: torch.Tensor,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    config: LRPTrainingConfig,
    device: torch.device,
    *,
    layerwise: bool = False,
    seed_offset: int = 0,
) -> Tuple[nn.Module, np.ndarray, Dict[str, Any]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = _fixed_loader(
        train_values, labels, fit_indices, config, seed_offset=seed_offset
    )
    holdout_x = train_values[holdout_indices]
    holdout_y = labels[holdout_indices]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        batches = 0
        for values, targets in loader:
            values = values.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            if layerwise:
                targets = targets[:, None].expand_as(logits)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            logits = _fixed_logits(
                model,
                holdout_x,
                device,
                batch_size=config.batch_size,
            )
            target = (
                holdout_y[:, None].expand_as(logits) if layerwise else holdout_y
            )
            validation_loss = float(criterion(logits, target).item())
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total / max(batches, 1),
                "holdout_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("LRP fixed probe produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    holdout_scores = torch.sigmoid(
        _fixed_logits(
            model,
            holdout_x,
            device,
            batch_size=config.batch_size,
        )
    ).numpy()
    return model.cpu(), holdout_scores, {
        "best_epoch": best_epoch,
        "best_holdout_loss": best_loss,
        "training_history": history,
    }


def _fixed_logits(
    model: nn.Module,
    values: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            rows.append(model(batch).detach().cpu())
    return torch.cat(rows, dim=0)


def _score_fixed(
    model: nn.Module,
    values: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    return torch.sigmoid(
        _fixed_logits(model, values, device, batch_size=batch_size)
    ).numpy()


def _fit_sequence(
    model: AttentionPatternTransformer,
    train_sequences: Sequence[torch.Tensor],
    labels: torch.Tensor,
    fit_indices: np.ndarray,
    holdout_indices: np.ndarray,
    config: LRPTrainingConfig,
    device: torch.device,
    *,
    seed_offset: int = 0,
) -> Tuple[AttentionPatternTransformer, np.ndarray, Dict[str, Any]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = _sequence_loader(
        train_sequences, labels, fit_indices, config, seed_offset=seed_offset
    )
    holdout_sequences = [train_sequences[int(index)] for index in holdout_indices]
    holdout_y = labels[holdout_indices]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        batches = 0
        for values, mask, targets in loader:
            values, mask, targets = values.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values, mask), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            holdout_logits = _sequence_logits(
                model,
                holdout_sequences,
                device,
                batch_size=config.batch_size,
            )
            validation_loss = float(
                criterion(holdout_logits, holdout_y).item()
            )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total / max(batches, 1),
                "holdout_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("LRP attention probe produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    holdout_scores = torch.sigmoid(
        _sequence_logits(
            model,
            holdout_sequences,
            device,
            batch_size=config.batch_size,
        )
    ).numpy()
    return model.cpu(), holdout_scores, {
        "best_epoch": best_epoch,
        "best_holdout_loss": best_loss,
        "training_history": history,
    }


def _sequence_logits(
    model: AttentionPatternTransformer,
    sequences: Sequence[torch.Tensor],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(sequences), int(batch_size)):
            values, mask = _pad_sequences(
                sequences[start : start + int(batch_size)]
            )
            rows.append(
                model(values.to(device), mask.to(device)).detach().cpu()
            )
    return torch.cat(rows, dim=0)


def _score_sequence(
    model: AttentionPatternTransformer,
    sequences: Sequence[torch.Tensor],
    device: torch.device,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    return torch.sigmoid(
        _sequence_logits(model, sequences, device, batch_size=batch_size)
    ).numpy()


def _layer_abstention_accuracies(
    error_targets: np.ndarray, layer_scores: np.ndarray
) -> np.ndarray:
    decisions = layer_scores >= 0.5
    return np.mean(
        np.where(decisions, error_targets[:, None], 1.0 - error_targets[:, None]),
        axis=0,
    )


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def fit_lrp_probes_and_score(
    train_features: Sequence[Mapping[str, Any]],
    train_correctness: Sequence[float],
    eval_features: Sequence[Mapping[str, Any]],
    *,
    eval_correctness: Optional[Sequence[float]] = None,
    config: Optional[LRPTrainingConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Fit requested LRP variants and return repository-style error scores."""
    config = config or LRPTrainingConfig()
    config.validate()
    require_visual = "visual_attention" in config.variants
    train = lrp_feature_batch(train_features, require_visual=require_visual)
    evaluation = lrp_feature_batch(eval_features, require_visual=require_visual)
    if (
        train.hidden.shape[1:] != evaluation.hidden.shape[1:]
        or train.visual_attention.shape[1:] != evaluation.visual_attention.shape[1:]
    ):
        raise ValueError("LRP train/eval model dimensions differ.")
    correctness = np.asarray(train_correctness, dtype=np.float32).reshape(-1)
    if correctness.shape[0] != train.hidden.shape[0]:
        raise ValueError("LRP feature and correctness counts differ.")
    if not np.isfinite(correctness).all():
        raise ValueError("LRP correctness labels must be finite.")
    correctness = np.clip(correctness, 0.0, 1.0)
    error_targets = 1.0 - correctness
    fit_indices, holdout_indices = _split_indices(
        error_targets, config.validation_fraction, config.random_seed
    )
    labels = torch.from_numpy(error_targets)
    device = _resolve_device(config.device)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    scores: Dict[str, np.ndarray] = {}
    variants_metadata: Dict[str, Any] = {}
    checkpoint_variants: Dict[str, Any] = {}

    def finish_variant(
        name: str,
        values: np.ndarray,
        holdout_scores: np.ndarray,
        training_metadata: Dict[str, Any],
        checkpoint_spec: Dict[str, Any],
        *,
        fixed_threshold: Optional[float] = None,
    ) -> None:
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise FloatingPointError(f"LRP variant `{name}` produced invalid scores.")
        if fixed_threshold is None:
            threshold, holdout_metrics = select_abstention_error_threshold(
                correctness[holdout_indices], holdout_scores
            )
        else:
            threshold = float(fixed_threshold)
            holdout_metrics = abstention_metrics(
                correctness[holdout_indices],
                holdout_scores,
                error_threshold=threshold,
            )
        binary = (error_targets[holdout_indices] >= 0.5).astype(np.int64)
        holdout_auroc = (
            float(roc_auc_score(binary, holdout_scores))
            if np.unique(binary).size == 2
            else float("nan")
        )
        measure_name = f"lrp_{name}"
        scores[measure_name] = values.astype(np.float64)
        variant_meta = {
            **training_metadata,
            "measure_name": measure_name,
            "holdout_auroc": holdout_auroc,
            "selected_error_threshold": threshold,
            "holdout_abstention_metrics": holdout_metrics,
        }
        if eval_correctness is not None:
            variant_meta["eval_abstention_metrics"] = abstention_metrics(
                eval_correctness, values, error_threshold=threshold
            )
        variants_metadata[name] = variant_meta
        checkpoint_spec["error_threshold"] = threshold
        checkpoint_variants[name] = checkpoint_spec

    if "concat_hidden" in config.variants:
        input_dim = train.num_layers * train.hidden_dim
        model = FourLayerMLP(
            input_dim, config.mlp_hidden_dims, dropout=config.dropout
        )
        train_values = train.hidden.flatten(start_dim=1)
        eval_values = evaluation.hidden.flatten(start_dim=1)
        model, holdout_scores, training_meta = _fit_fixed(
            model,
            train_values,
            labels,
            fit_indices,
            holdout_indices,
            config,
            device,
        )
        eval_scores = _score_fixed(
            model, eval_values, device, batch_size=config.batch_size
        )
        finish_variant(
            "concat_hidden",
            eval_scores,
            holdout_scores,
            training_meta,
            {
                "kind": "fixed_mlp",
                "input_dim": input_dim,
                "hidden_dims": tuple(config.mlp_hidden_dims),
                "dropout": config.dropout,
                "state_dict": _cpu_state(model),
            },
        )

    if "visual_attention" in config.variants:
        input_dim = train.num_layers * train.num_heads
        model = FourLayerMLP(
            input_dim, config.mlp_hidden_dims, dropout=config.dropout
        )
        train_values = train.visual_attention.flatten(start_dim=1)
        eval_values = evaluation.visual_attention.flatten(start_dim=1)
        model, holdout_scores, training_meta = _fit_fixed(
            model,
            train_values,
            labels,
            fit_indices,
            holdout_indices,
            config,
            device,
            seed_offset=1,
        )
        eval_scores = _score_fixed(
            model, eval_values, device, batch_size=config.batch_size
        )
        finish_variant(
            "visual_attention",
            eval_scores,
            holdout_scores,
            training_meta,
            {
                "kind": "fixed_mlp",
                "input_dim": input_dim,
                "hidden_dims": tuple(config.mlp_hidden_dims),
                "dropout": config.dropout,
                "state_dict": _cpu_state(model),
            },
        )

    if "ensemble_hidden" in config.variants:
        model = LayerwiseMLP(
            train.num_layers,
            train.hidden_dim,
            config.mlp_hidden_dims,
            config.dropout,
        )
        model, holdout_layer_scores, training_meta = _fit_fixed(
            model,
            train.hidden,
            labels,
            fit_indices,
            holdout_indices,
            config,
            device,
            layerwise=True,
            seed_offset=2,
        )
        layer_accuracy = _layer_abstention_accuracies(
            error_targets[holdout_indices], holdout_layer_scores
        )
        top_k = min(config.top_k_layers, train.num_layers)
        selected = np.argsort(layer_accuracy)[::-1][:top_k]
        eval_layer_scores = _score_fixed(
            model, evaluation.hidden, device, batch_size=config.batch_size
        )
        # The fraction of selected probes voting "error" preserves Eq. (14)'s
        # majority decision at 0.5 while retaining useful ranking granularity.
        eval_scores = np.mean(eval_layer_scores[:, selected] >= 0.5, axis=1)
        holdout_scores = np.mean(
            holdout_layer_scores[:, selected] >= 0.5, axis=1
        )
        training_meta.update(
            {
                "selected_layers": selected.astype(int).tolist(),
                "layer_holdout_abstention_accuracy": layer_accuracy.tolist(),
                "aggregation": "error-vote fraction; >=0.5 is majority abstention",
            }
        )
        finish_variant(
            "ensemble_hidden",
            eval_scores,
            holdout_scores,
            training_meta,
            {
                "kind": "layerwise_mlp",
                "num_layers": train.num_layers,
                "input_dim": train.hidden_dim,
                "hidden_dims": tuple(config.mlp_hidden_dims),
                "dropout": config.dropout,
                "selected_layers": selected.astype(int).tolist(),
                "state_dict": _cpu_state(model),
            },
            fixed_threshold=0.5,
        )

    if "concat_attention" in config.variants:
        input_dim = train.num_layers * train.num_heads
        model = AttentionPatternTransformer(
            input_dim,
            model_dim=config.transformer_model_dim,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            dropout=config.dropout,
        )
        train_sequences = _concat_attention_sequences(train)
        eval_sequences = _concat_attention_sequences(evaluation)
        model, holdout_scores, training_meta = _fit_sequence(
            model,
            train_sequences,
            labels,
            fit_indices,
            holdout_indices,
            config,
            device,
            seed_offset=3,
        )
        eval_scores = _score_sequence(
            model, eval_sequences, device, batch_size=config.batch_size
        )
        finish_variant(
            "concat_attention",
            eval_scores,
            holdout_scores,
            training_meta,
            {
                "kind": "attention_transformer",
                "input_dim": input_dim,
                "model_dim": config.transformer_model_dim,
                "num_layers": config.transformer_layers,
                "num_heads": config.transformer_heads,
                "dropout": config.dropout,
                "state_dict": _cpu_state(model),
            },
        )

    if "ensemble_attention" in config.variants:
        layer_models = []
        holdout_columns = []
        eval_columns = []
        layer_training = []
        for layer_index in range(train.num_layers):
            model = AttentionPatternTransformer(
                train.num_heads,
                model_dim=config.transformer_model_dim,
                num_layers=config.transformer_layers,
                num_heads=config.transformer_heads,
                dropout=config.dropout,
            )
            model, holdout_scores, training_meta = _fit_sequence(
                model,
                _layer_attention_sequences(train, layer_index),
                labels,
                fit_indices,
                holdout_indices,
                config,
                device,
                seed_offset=100 + layer_index,
            )
            eval_scores = _score_sequence(
                model,
                _layer_attention_sequences(evaluation, layer_index),
                device,
                batch_size=config.batch_size,
            )
            layer_models.append(model)
            holdout_columns.append(holdout_scores)
            eval_columns.append(eval_scores)
            layer_training.append(training_meta)
        holdout_layer_scores = np.stack(holdout_columns, axis=1)
        eval_layer_scores = np.stack(eval_columns, axis=1)
        layer_accuracy = _layer_abstention_accuracies(
            error_targets[holdout_indices], holdout_layer_scores
        )
        top_k = min(config.top_k_layers, train.num_layers)
        selected = np.argsort(layer_accuracy)[::-1][:top_k]
        eval_scores = np.mean(eval_layer_scores[:, selected] >= 0.5, axis=1)
        holdout_scores = np.mean(
            holdout_layer_scores[:, selected] >= 0.5, axis=1
        )
        finish_variant(
            "ensemble_attention",
            eval_scores,
            holdout_scores,
            {
                "selected_layers": selected.astype(int).tolist(),
                "layer_holdout_abstention_accuracy": layer_accuracy.tolist(),
                "layer_training": layer_training,
                "aggregation": "error-vote fraction; >=0.5 is majority abstention",
            },
            {
                "kind": "layerwise_attention_transformer",
                "input_dim": train.num_heads,
                "model_dim": config.transformer_model_dim,
                "num_layers": config.transformer_layers,
                "num_heads": config.transformer_heads,
                "dropout": config.dropout,
                "selected_layers": selected.astype(int).tolist(),
                "state_dicts": [_cpu_state(model) for model in layer_models],
            },
            fixed_threshold=0.5,
        )

    metadata: Dict[str, Any] = {
        "method": "Latent Representation Probing (LRP)",
        "paper": "Yao et al. (2025), arXiv:2511.19806v1",
        "training_config": asdict(config),
        "label_semantics": "soft correctness y in [0,1]; probes train on error target 1-y",
        "score_semantics": "OCR/VQA error risk; higher means abstain",
        "train_samples": int(train.hidden.shape[0]),
        "fit_samples": int(fit_indices.size),
        "holdout_samples": int(holdout_indices.size),
        "eval_samples": int(evaluation.hidden.shape[0]),
        "num_layers": train.num_layers,
        "num_heads": train.num_heads,
        "hidden_dim": train.hidden_dim,
        "architecture_notes": {
            "mlp_depth": "four Linear layers (three hidden plus output)",
            "attention_position_encoding": "fixed sinusoidal",
            "attention_pooling": "padding-masked mean over input-token positions",
            "ensemble_hidden_parameter_sharing": "none",
        },
        "variants": variants_metadata,
    }
    checkpoint = {
        "format_version": 1,
        "model_dimensions": {
            "num_layers": train.num_layers,
            "num_heads": train.num_heads,
            "hidden_dim": train.hidden_dim,
        },
        "variants": checkpoint_variants,
        "metadata": metadata,
    }
    return scores, metadata, checkpoint


def save_lrp_probe_checkpoint(path: str, checkpoint: Mapping[str, Any]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), destination)
    return str(destination)


def load_lrp_probe_checkpoint(path: str) -> Dict[str, Any]:
    return dict(torch.load(path, map_location="cpu", weights_only=False))


def _load_variant_model(spec: Mapping[str, Any]) -> nn.Module:
    kind = spec["kind"]
    if kind == "fixed_mlp":
        model = FourLayerMLP(
            int(spec["input_dim"]),
            hidden_dims=tuple(spec["hidden_dims"]),
            dropout=float(spec["dropout"]),
        )
    elif kind == "layerwise_mlp":
        model = LayerwiseMLP(
            int(spec["num_layers"]),
            int(spec["input_dim"]),
            tuple(spec["hidden_dims"]),
            float(spec["dropout"]),
        )
    elif kind == "attention_transformer":
        model = AttentionPatternTransformer(
            int(spec["input_dim"]),
            model_dim=int(spec["model_dim"]),
            num_layers=int(spec["num_layers"]),
            num_heads=int(spec["num_heads"]),
            dropout=float(spec["dropout"]),
        )
    else:
        raise ValueError(f"Cannot load LRP model kind `{kind}` as one module.")
    model.load_state_dict(spec["state_dict"])
    return model


def score_lrp_checkpoint(
    checkpoint: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Apply a saved LRP checkpoint without access to correctness labels."""
    if int(batch_size) < 1:
        raise ValueError("LRP checkpoint scoring batch_size must be positive.")
    require_visual = "visual_attention" in checkpoint["variants"]
    batch = lrp_feature_batch(features, require_visual=require_visual)
    expected = checkpoint.get("model_dimensions", {})
    actual = {
        "num_layers": batch.num_layers,
        "num_heads": batch.num_heads,
        "hidden_dim": batch.hidden_dim,
    }
    mismatched = {
        name: (expected.get(name), value)
        for name, value in actual.items()
        if expected.get(name) is not None and int(expected[name]) != int(value)
    }
    if mismatched:
        raise ValueError(f"LRP checkpoint/feature dimensions differ: {mismatched}")
    resolved = _resolve_device(device)
    output: Dict[str, np.ndarray] = {}
    for name, spec in checkpoint["variants"].items():
        kind = spec["kind"]
        if name == "concat_hidden":
            values = batch.hidden.flatten(start_dim=1)
            scores = _score_fixed(
                _load_variant_model(spec),
                values,
                resolved,
                batch_size=batch_size,
            )
        elif name == "visual_attention":
            values = batch.visual_attention.flatten(start_dim=1)
            scores = _score_fixed(
                _load_variant_model(spec),
                values,
                resolved,
                batch_size=batch_size,
            )
        elif name == "ensemble_hidden":
            layer_scores = _score_fixed(
                _load_variant_model(spec),
                batch.hidden,
                resolved,
                batch_size=batch_size,
            )
            selected = np.asarray(spec["selected_layers"], dtype=int)
            scores = np.mean(layer_scores[:, selected] >= 0.5, axis=1)
        elif name == "concat_attention":
            scores = _score_sequence(
                _load_variant_model(spec),
                _concat_attention_sequences(batch),
                resolved,
                batch_size=batch_size,
            )
        elif name == "ensemble_attention":
            layer_scores = []
            for layer_index, state in enumerate(spec["state_dicts"]):
                layer_spec = dict(spec)
                layer_spec["kind"] = "attention_transformer"
                layer_spec["state_dict"] = state
                model = _load_variant_model(layer_spec)
                layer_scores.append(
                    _score_sequence(
                        model,
                        _layer_attention_sequences(batch, layer_index),
                        resolved,
                        batch_size=batch_size,
                    )
                )
            matrix = np.stack(layer_scores, axis=1)
            selected = np.asarray(spec["selected_layers"], dtype=int)
            scores = np.mean(matrix[:, selected] >= 0.5, axis=1)
        else:
            raise ValueError(f"Unknown LRP checkpoint variant `{name}` ({kind}).")
        output[f"lrp_{name}"] = np.asarray(scores, dtype=np.float64)
    return output
