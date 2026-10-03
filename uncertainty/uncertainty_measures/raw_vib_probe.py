"""Minimal probes on the unmodified VIB-Probe input tensor.

Both ablations remove the VIB encoder, residual MLP, Gaussian bottleneck,
reparameterization, KL objective, and latent decoder.  The linear variant uses
one affine readout of the flattened pre-o_proj attention-head outputs; the
shallow variant inserts exactly one GELU hidden layer before that readout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .paper_probe_features import VIB_FEATURE_NAME
from .vib_probe import vib_feature_tensor


@dataclass(frozen=True)
class RawVIBLinearProbeConfig:
    """Optimization settings for the encoder-free minimal probes."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 12
    min_delta: float = 1e-4
    random_seed: int = 10
    device: str = "auto"
    hidden_dim: int = 0
    dropout: float = 0.0

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Raw VIB epochs and batch size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid Raw VIB optimizer configuration.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Raw VIB validation fraction must be in (0,1).")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("Invalid Raw VIB early-stopping configuration.")
        if self.hidden_dim < 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("Invalid Raw VIB shallow architecture configuration.")


class RawVIBLinearProbe(nn.Module):
    """Raw affine readout, optionally with exactly one hidden GELU layer."""

    def __init__(
        self,
        input_shape: Sequence[int],
        *,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        if len(self.input_shape) != 3 or min(self.input_shape) < 1:
            raise ValueError("Raw VIB input shape must contain [layers,heads,dim].")
        self.hidden_dim = int(hidden_dim)
        self.dropout_probability = float(dropout)
        if self.hidden_dim < 0 or not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("Invalid Raw VIB hidden dimension or dropout.")
        flattened_dim = int(np.prod(self.input_shape))
        if self.hidden_dim:
            self.input_projection = nn.Linear(flattened_dim, self.hidden_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(self.dropout_probability)
            self.readout = nn.Linear(self.hidden_dim, 1)
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            nn.init.xavier_uniform_(self.readout.weight)
        else:
            self.input_projection = nn.Identity()
            self.activation = nn.Identity()
            self.dropout = nn.Identity()
            self.readout = nn.Linear(flattened_dim, 1)
            nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        expected = self.input_shape
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                "Raw VIB input must be [batch,layers,heads,head_dim]="
                f"[batch,{expected}], got {tuple(values.shape)}."
            )
        hidden = self.input_projection(values.float().flatten(start_dim=1))
        hidden = self.dropout(self.activation(hidden))
        return self.readout(hidden).squeeze(-1)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested Raw VIB device `{value}`, but CUDA is unavailable.")
    return device


def _split_indices(
    labels: np.ndarray,
    validation_fraction: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    classes = labels.astype(np.int64)
    counts = np.bincount(classes, minlength=2)
    if labels.size < 6 or int(counts.min()) < 2:
        raise ValueError("Raw VIB probe requires both task-error classes.")
    fit, holdout = train_test_split(
        np.arange(labels.size),
        test_size=float(validation_fraction),
        random_state=int(random_seed),
        stratify=classes,
    )
    return np.asarray(fit), np.asarray(holdout)


def _score_logits(
    model: RawVIBLinearProbe,
    values: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            rows.append(model(batch).detach().cpu())
    return torch.cat(rows, dim=0)


def _holdout_bce(
    model: RawVIBLinearProbe,
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    logits = _score_logits(
        model, values, device=device, batch_size=batch_size
    )
    return float(F.binary_cross_entropy_with_logits(logits, labels).item())


def fit_raw_vib_linear_probe_and_score(
    train_features: Sequence[Mapping[str, Any]],
    train_error_targets: Sequence[float],
    eval_features: Sequence[Mapping[str, Any]],
    *,
    config: Optional[RawVIBLinearProbeConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any], RawVIBLinearProbe]:
    """Fit the raw-input logistic probe using train-only early stopping."""
    config = config or RawVIBLinearProbeConfig()
    config.validate()
    train_values = vib_feature_tensor(
        train_features, feature_name=VIB_FEATURE_NAME
    )
    eval_values = vib_feature_tensor(
        eval_features, feature_name=VIB_FEATURE_NAME
    )
    if tuple(train_values.shape[1:]) != tuple(eval_values.shape[1:]):
        raise ValueError("Raw VIB train/eval feature shapes differ.")
    labels = np.asarray(train_error_targets, dtype=np.float32).reshape(-1)
    if labels.shape[0] != train_values.shape[0]:
        raise ValueError("Raw VIB feature and label counts differ.")
    if (
        not np.isfinite(labels).all()
        or not np.all(np.isin(labels, (0.0, 1.0)))
    ):
        raise ValueError("Raw VIB targets must be finite binary values.")

    fit_indices, holdout_indices = _split_indices(
        labels, config.validation_fraction, config.random_seed
    )
    torch.manual_seed(int(config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.random_seed))
    device = _resolve_device(config.device)
    model = RawVIBLinearProbe(
        train_values.shape[1:],
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
    ).to(device)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    architecture_name = "shallow" if model.hidden_dim else "linear"
    logging.info(
        "RAW VIB %s MODEL | input_shape=%s | flattened_dim=%d | "
        "hidden_dim=%d | dropout=%.3f | parameters=%d | bottleneck=none "
        "| KL=none",
        architecture_name.upper(),
        model.input_shape,
        int(np.prod(model.input_shape)),
        model.hidden_dim,
        model.dropout_probability,
        parameter_count,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    label_tensor = torch.from_numpy(labels)
    generator = torch.Generator().manual_seed(int(config.random_seed))
    loader = DataLoader(
        TensorDataset(
            train_values[fit_indices], label_tensor[fit_indices]
        ),
        batch_size=min(int(config.batch_size), int(fit_indices.size)),
        shuffle=True,
        generator=generator,
    )
    holdout_values = train_values[holdout_indices]
    holdout_labels = label_tensor[holdout_indices]
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(int(config.epochs)):
        model.train()
        epoch_loss = 0.0
        count = 0
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(batch_values), batch_labels
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(batch_labels.shape[0])
            count += int(batch_labels.shape[0])
        holdout_loss = _holdout_bce(
            model,
            holdout_values,
            holdout_labels,
            device=device,
            batch_size=config.batch_size,
        )
        history.append({
            "epoch": epoch + 1,
            "train_bce": epoch_loss / max(count, 1),
            "holdout_bce": holdout_loss,
        })
        if holdout_loss < best_loss - float(config.min_delta):
            best_loss = holdout_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if config.patience and stale_epochs >= int(config.patience):
                break
    if best_state is None:
        raise RuntimeError("Raw VIB probe did not produce a checkpoint.")
    model.load_state_dict(best_state)
    eval_logits = _score_logits(
        model, eval_values, device=device, batch_size=config.batch_size
    )
    holdout_logits = _score_logits(
        model, holdout_values, device=device, batch_size=config.batch_size
    )
    probabilities = torch.sigmoid(eval_logits).numpy().astype(np.float64)
    holdout_probabilities = torch.sigmoid(holdout_logits).numpy()
    holdout_targets = labels[holdout_indices].astype(np.int64)

    if model.hidden_dim:
        weights = model.input_projection.weight.detach().cpu().reshape(
            model.hidden_dim, *model.input_shape
        )
        head_norm = torch.linalg.vector_norm(weights.float(), dim=(0, 3))
        layer_norm = torch.linalg.vector_norm(weights.float(), dim=(0, 2, 3))
    else:
        weights = model.readout.weight.detach().cpu().reshape(model.input_shape)
        head_norm = torch.linalg.vector_norm(weights.float(), dim=-1)
        layer_norm = torch.linalg.vector_norm(weights.float(), dim=(1, 2))
    flat_order = torch.argsort(head_norm.flatten(), descending=True)
    top_heads = []
    num_heads = int(model.input_shape[1])
    for flat_index in flat_order[: min(20, flat_order.numel())].tolist():
        layer_index = int(flat_index // num_heads)
        head_index = int(flat_index % num_heads)
        top_heads.append({
            "layer": layer_index,
            "head": head_index,
            "weight_l2": float(head_norm[layer_index, head_index]),
        })
    metadata: Dict[str, Any] = {
        "method": (
            "raw_vib_shallow_probe" if model.hidden_dim
            else "raw_vib_linear_probe"
        ),
        "definition": (
            "flatten(vib_attention_head_outputs) -> Linear -> GELU -> "
            "Dropout -> affine logit"
            if model.hidden_dim
            else "flatten(vib_attention_head_outputs) -> affine logit"
        ),
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout_probability,
        "removed_components": [
            (
                "VIB deep encoder MLP (replaced by one hidden layer)"
                if model.hidden_dim else "VIB encoder MLP"
            ),
            "residual blocks",
            "Gaussian mu/log-variance bottleneck",
            "latent reparameterization",
            "KL regularization",
            "latent decoder",
        ],
        "feature_name": VIB_FEATURE_NAME,
        "input_shape": list(model.input_shape),
        "flattened_dim": int(np.prod(model.input_shape)),
        "parameter_count": parameter_count,
        "training_config": asdict(config),
        "train_samples": int(train_values.shape[0]),
        "fit_samples": int(fit_indices.size),
        "holdout_samples": int(holdout_indices.size),
        "eval_samples": int(eval_values.shape[0]),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_holdout_bce": best_loss,
        "holdout_auroc": float(
            roc_auc_score(holdout_targets, holdout_probabilities)
        ),
        "history": history,
        "layer_weight_l2": layer_norm.tolist(),
        "top_heads_by_weight_l2": top_heads,
    }
    logging.info(
        "RAW VIB %s RESULT | epochs=%d | best_epoch=%d | "
        "holdout_bce=%.6f | holdout_auroc=%.4f | parameters=%d",
        architecture_name.upper(),
        len(history),
        best_epoch,
        best_loss,
        metadata["holdout_auroc"],
        parameter_count,
    )
    return probabilities, metadata, model.cpu()


def fit_raw_vib_shallow_probe_and_score(
    train_features: Sequence[Mapping[str, Any]],
    train_error_targets: Sequence[float],
    eval_features: Sequence[Mapping[str, Any]],
    *,
    config: Optional[RawVIBLinearProbeConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any], RawVIBLinearProbe]:
    """Fit the one-hidden-layer raw-input ablation."""
    config = config or RawVIBLinearProbeConfig(
        epochs=50,
        learning_rate=1e-4,
        patience=8,
        hidden_dim=128,
        dropout=0.1,
    )
    if int(config.hidden_dim) < 1:
        raise ValueError("Raw VIB shallow probe requires hidden_dim >= 1.")
    return fit_raw_vib_linear_probe_and_score(
        train_features,
        train_error_targets,
        eval_features,
        config=config,
    )


def save_raw_vib_linear_probe_checkpoint(
    path: str,
    model: RawVIBLinearProbe,
    metadata: Mapping[str, Any],
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "input_shape": list(model.input_shape),
        "hidden_dim": int(model.hidden_dim),
        "dropout": float(model.dropout_probability),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "metadata": dict(metadata),
    }, destination)
    return str(destination)


def load_raw_vib_linear_probe_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> Tuple[RawVIBLinearProbe, Dict[str, Any]]:
    resolved = _resolve_device(device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    model = RawVIBLinearProbe(
        payload["input_shape"],
        hidden_dim=int(payload.get("hidden_dim", 0)),
        dropout=float(payload.get("dropout", 0.0)),
    ).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
