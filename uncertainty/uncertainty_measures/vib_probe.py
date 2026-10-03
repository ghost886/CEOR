"""VIB-Probe hallucination detection and head-level intervention.

This is a project-native reproduction of Zhang et al., *VIB-Probe: Detecting
and Mitigating Hallucinations in Vision-Language Models via Variational
Information Bottleneck* (2026).  Scores follow the repository convention:
zero means faithful/correct and one means hallucinated/error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .paper_probe_features import VIB_FEATURE_NAME


VIB_INPUT_ADAPTERS = ("flatten", "head_projection", "factorized")
VIB_FEATURE_LAYOUTS = ("layer_head", "vector")
VIB_LAST_TOKEN_FEATURE_NAME = "last_token_h"
VIB_PRIOR_TYPES = ("standard_normal", "gaussian_mixture")

# Controlled parameter-count ablations.  The VIB loss, encoder, bottleneck,
# split, optimizer, and early stopping remain unchanged; only the raw
# [layer, head, head_dim] input adapter differs.
VIB_SIZE_ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "head_projection_16": {
        "input_adapter": "head_projection",
        "projected_head_dim": 16,
    },
    "factorized_8x8x32": {
        "input_adapter": "factorized",
        "factorized_feature_dim": 32,
        "factorized_num_heads": 8,
        "factorized_num_layers": 8,
    },
}


@dataclass(frozen=True)
class VIBProbeConfig:
    """Shape and architecture of the paper's VIB detector."""

    num_layers: int
    num_heads: int
    head_dim: int
    encoder_dims: Tuple[int, int, int] = (1024, 512, 256)
    latent_dim: int = 256
    residual_blocks: int = 2
    input_adapter: str = "flatten"
    projected_head_dim: int = 16
    factorized_feature_dim: int = 32
    factorized_num_heads: int = 8
    factorized_num_layers: int = 8
    prior_type: str = "standard_normal"
    mixture_components: int = 5
    mixture_readout: bool = False
    mixture_mean_init_scale: float = 0.05

    def validate(self) -> None:
        if min(self.num_layers, self.num_heads, self.head_dim) < 1:
            raise ValueError("VIB input dimensions must be positive.")
        if tuple(self.encoder_dims) != (1024, 512, 256):
            # Alternative sizes remain useful for unit tests and ablations.
            if len(self.encoder_dims) != 3 or min(self.encoder_dims) < 1:
                raise ValueError("VIB encoder_dims must contain three positive widths.")
        if self.latent_dim < 1 or self.residual_blocks < 0:
            raise ValueError("Invalid VIB bottleneck configuration.")
        if self.input_adapter not in VIB_INPUT_ADAPTERS:
            raise ValueError(
                f"Unknown VIB input adapter `{self.input_adapter}`; "
                f"expected one of {VIB_INPUT_ADAPTERS}."
            )
        if self.projected_head_dim < 1:
            raise ValueError("VIB projected_head_dim must be positive.")
        if min(
            self.factorized_feature_dim,
            self.factorized_num_heads,
            self.factorized_num_layers,
        ) < 1:
            raise ValueError("VIB factorized adapter dimensions must be positive.")
        if self.prior_type not in VIB_PRIOR_TYPES:
            raise ValueError(
                f"Unknown VIB prior `{self.prior_type}`; expected one of "
                f"{VIB_PRIOR_TYPES}."
            )
        if self.mixture_components < 1:
            raise ValueError("VIB mixture_components must be positive.")
        if self.mixture_mean_init_scale < 0:
            raise ValueError("VIB mixture_mean_init_scale must be non-negative.")
        if self.mixture_readout and self.prior_type != "gaussian_mixture":
            raise ValueError(
                "VIB mixture_readout requires prior_type='gaussian_mixture'."
            )


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear_1 = nn.Linear(width, width)
        self.linear_2 = nn.Linear(width, width)
        self.norm_1 = nn.LayerNorm(width)
        self.norm_2 = nn.LayerNorm(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm_1(F.gelu(self.linear_1(values)))
        values = self.linear_2(values)
        return self.norm_2(F.gelu(values + residual))


class VIBProbe(nn.Module):
    """Head-output encoder, Gaussian bottleneck, and linear risk decoder.

    The paper-faithful baseline flattens the complete raw
    ``[layer, head, head_dim]`` tensor before its ``1024 -> 512 -> 256``
    encoder.  Controlled size ablations can instead apply a shared per-head
    projection or learned feature/head/layer factorization.  Every adapter
    remains differentiable with respect to every raw head output, preserving
    gradient-times-activation mitigation.
    """

    def __init__(self, config: VIBProbeConfig):
        super().__init__()
        config.validate()
        self.config = config
        layers, heads, head_dim = (
            config.num_layers,
            config.num_heads,
            config.head_dim,
        )
        self.feature_projection: nn.Module
        self.head_compressor: nn.Module
        self.layer_compressor: nn.Module
        if config.input_adapter == "flatten":
            self.feature_projection = nn.Identity()
            self.head_compressor = nn.Identity()
            self.layer_compressor = nn.Identity()
            adapter_output_dim = layers * heads * head_dim
        elif config.input_adapter == "head_projection":
            projected_dim = int(config.projected_head_dim)
            self.feature_projection = nn.Sequential(
                nn.Linear(head_dim, projected_dim),
                nn.GELU(),
                nn.LayerNorm(projected_dim),
            )
            self.head_compressor = nn.Identity()
            self.layer_compressor = nn.Identity()
            adapter_output_dim = layers * heads * projected_dim
        else:
            projected_dim = int(config.factorized_feature_dim)
            self.feature_projection = nn.Sequential(
                nn.Linear(head_dim, projected_dim),
                nn.GELU(),
                nn.LayerNorm(projected_dim),
            )
            # These two shared linear maps retain learned layer/head identity
            # while avoiding one enormous dense map from L*H*d_head.
            self.head_compressor = nn.Linear(
                heads, int(config.factorized_num_heads)
            )
            self.layer_compressor = nn.Linear(
                layers, int(config.factorized_num_layers)
            )
            adapter_output_dim = (
                int(config.factorized_num_layers)
                * int(config.factorized_num_heads)
                * projected_dim
            )
        self.adapter_output_dim = int(adapter_output_dim)
        dimensions = [self.adapter_output_dim, *config.encoder_dims]
        blocks: List[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            blocks.extend(
                [nn.Linear(input_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim)]
            )
        self.encoder = nn.Sequential(*blocks)
        self.residual = nn.Sequential(
            *[_ResidualBlock(config.encoder_dims[-1]) for _ in range(config.residual_blocks)]
        )
        self.mu = nn.Linear(config.encoder_dims[-1], config.latent_dim)
        self.log_variance = nn.Linear(config.encoder_dims[-1], config.latent_dim)
        if config.prior_type == "gaussian_mixture":
            components = int(config.mixture_components)
            latent_dim = int(config.latent_dim)
            self.prior_component_means = nn.Parameter(
                torch.empty(components, latent_dim)
            )
            self.prior_component_log_variances = nn.Parameter(
                torch.zeros(components, latent_dim)
            )
            self.prior_mixture_logits = nn.Parameter(torch.zeros(components))
        else:
            self.register_parameter("prior_component_means", None)
            self.register_parameter("prior_component_log_variances", None)
            self.register_parameter("prior_mixture_logits", None)
        classifier_dim = int(config.latent_dim)
        if config.mixture_readout:
            classifier_dim += int(config.mixture_components)
        self.classifier = nn.Linear(classifier_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.prior_component_means is not None:
            nn.init.normal_(
                self.prior_component_means,
                mean=0.0,
                std=float(self.config.mixture_mean_init_scale),
            )
            nn.init.zeros_(self.prior_component_log_variances)
            nn.init.zeros_(self.prior_mixture_logits)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (
            self.config.num_layers,
            self.config.num_heads,
            self.config.head_dim,
        )

    def encode(
        self, head_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if head_outputs.ndim != 4 or tuple(head_outputs.shape[1:]) != self.input_shape:
            raise ValueError(
                "VIB input must have shape [batch, layers, heads, head_dim] = "
                f"[batch, {self.input_shape}], got {tuple(head_outputs.shape)}."
            )
        if not bool(torch.isfinite(head_outputs).all()):
            raise ValueError("VIB input contains NaN or infinity.")
        adapted = self.feature_projection(head_outputs.float())
        if self.config.input_adapter == "factorized":
            # [B,L,H,D] -> mix H -> [B,L,H',D].
            adapted = self.head_compressor(
                adapted.transpose(-1, -2)
            ).transpose(-1, -2)
            # Move L to the last axis, mix it, then restore [B,L',H',D].
            adapted = self.layer_compressor(
                adapted.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)
        hidden = self.encoder(adapted.flatten(start_dim=1))
        hidden = self.residual(hidden)
        mu = self.mu(hidden)
        log_variance = self.log_variance(hidden).clamp(min=-12.0, max=8.0)
        return mu, log_variance

    def forward(
        self,
        head_outputs: torch.Tensor,
        *,
        sample: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_variance = self.encode(head_outputs)
        if sample is None:
            sample = self.training
        if sample:
            std = torch.exp(0.5 * log_variance)
            latent = mu + std * torch.randn_like(std)
        else:
            latent = mu
        classifier_features = latent
        if self.config.mixture_readout:
            responsibilities = self.prototype_responsibilities(mu, log_variance)
            classifier_features = torch.cat(
                [classifier_features, responsibilities], dim=-1
            )
        return self.classifier(classifier_features).squeeze(-1), mu, log_variance

    def prototype_responsibilities(
        self,
        mu: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Return posterior affinity to each learned Gaussian prototype."""
        if self.config.prior_type != "gaussian_mixture":
            raise ValueError("Prototype responsibilities require a mixture prior.")
        component_kl = gaussian_kl_to_components(
            mu,
            log_variance,
            self.prior_component_means,
            self.prior_component_log_variances,
        )
        log_weights = F.log_softmax(self.prior_mixture_logits, dim=0)
        return F.softmax(log_weights.unsqueeze(0) - component_kl, dim=-1)

    def information_kl(
        self,
        mu: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """KL regularizer for either the standard or mixture bottleneck."""
        if self.config.prior_type == "standard_normal":
            return gaussian_kl(mu, log_variance)
        return gaussian_mixture_kl_upper_bound(
            mu,
            log_variance,
            self.prior_component_means,
            self.prior_component_log_variances,
            self.prior_mixture_logits,
        )


def gaussian_kl(mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Closed-form ``KL(q(z|v) || N(0, I))`` for each sample."""
    return 0.5 * torch.sum(
        mu.square() + torch.exp(log_variance) - log_variance - 1.0,
        dim=-1,
    )


def gaussian_kl_to_components(
    posterior_mu: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    component_means: torch.Tensor,
    component_log_variances: torch.Tensor,
) -> torch.Tensor:
    """Analytic ``KL(q_i || p_k)`` for every sample/component pair."""
    if posterior_mu.shape != posterior_log_variance.shape:
        raise ValueError("Posterior mean/log-variance shapes must match.")
    if component_means.shape != component_log_variances.shape:
        raise ValueError("Mixture mean/log-variance shapes must match.")
    if posterior_mu.ndim != 2 or component_means.ndim != 2:
        raise ValueError("Gaussian KL inputs must be rank-two tensors.")
    if posterior_mu.shape[-1] != component_means.shape[-1]:
        raise ValueError("Posterior and mixture latent dimensions must match.")
    posterior_log_variance = posterior_log_variance.clamp(-12.0, 8.0)
    component_log_variances = component_log_variances.clamp(-6.0, 4.0)
    mean_delta = (
        posterior_mu.unsqueeze(1) - component_means.unsqueeze(0)
    ).square()
    variance_ratio = (
        posterior_log_variance.unsqueeze(1)
        - component_log_variances.unsqueeze(0)
    ).exp()
    scaled_mean_delta = mean_delta * torch.exp(
        -component_log_variances.unsqueeze(0)
    )
    return 0.5 * torch.sum(
        component_log_variances.unsqueeze(0)
        - posterior_log_variance.unsqueeze(1)
        + variance_ratio
        + scaled_mean_delta
        - 1.0,
        dim=-1,
    )


def gaussian_mixture_kl_upper_bound(
    posterior_mu: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    component_means: torch.Tensor,
    component_log_variances: torch.Tensor,
    mixture_logits: torch.Tensor,
) -> torch.Tensor:
    """Stable variational upper bound for ``KL(q || sum_k pi_k p_k)``.

    ``-log sum_k pi_k exp(-KL(q || p_k))`` is analytic, non-negative, and
    exactly recovers the existing standard-Gaussian KL when the mixture has a
    single zero-mean/unit-variance component.
    """
    component_kl = gaussian_kl_to_components(
        posterior_mu,
        posterior_log_variance,
        component_means,
        component_log_variances,
    )
    if mixture_logits.ndim != 1 or mixture_logits.shape[0] != component_kl.shape[1]:
        raise ValueError("Mixture logits must contain one value per component.")
    log_weights = F.log_softmax(mixture_logits, dim=0)
    return -torch.logsumexp(log_weights.unsqueeze(0) - component_kl, dim=-1)


@dataclass(frozen=True)
class VIBProbeTrainingConfig:
    """Published optimizer defaults plus train-only checkpoint selection."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    beta: float = 3e-4
    beta_warmup_fraction: float = 0.2
    validation_fraction: float = 0.2
    patience: int = 8
    min_delta: float = 1e-4
    random_seed: int = 10
    device: str = "auto"
    encoder_dims: Tuple[int, int, int] = (1024, 512, 256)
    latent_dim: int = 256
    residual_blocks: int = 2
    input_adapter: str = "flatten"
    projected_head_dim: int = 16
    factorized_feature_dim: int = 32
    factorized_num_heads: int = 8
    factorized_num_layers: int = 8
    prior_type: str = "standard_normal"
    mixture_components: int = 5
    mixture_readout: bool = False
    mixture_mean_init_scale: float = 0.05

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("VIB epochs and batch size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.beta < 0:
            raise ValueError("Invalid VIB optimizer or beta configuration.")
        if not 0.0 <= self.beta_warmup_fraction <= 1.0:
            raise ValueError("VIB beta_warmup_fraction must be in [0, 1].")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("VIB validation_fraction must be in (0, 1).")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("Invalid VIB early-stopping configuration.")
        VIBProbeConfig(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            encoder_dims=tuple(self.encoder_dims),
            latent_dim=int(self.latent_dim),
            residual_blocks=int(self.residual_blocks),
            input_adapter=str(self.input_adapter),
            projected_head_dim=int(self.projected_head_dim),
            factorized_feature_dim=int(self.factorized_feature_dim),
            factorized_num_heads=int(self.factorized_num_heads),
            factorized_num_layers=int(self.factorized_num_layers),
            prior_type=str(self.prior_type),
            mixture_components=int(self.mixture_components),
            mixture_readout=bool(self.mixture_readout),
            mixture_mean_init_scale=float(self.mixture_mean_init_scale),
        ).validate()


def vib_probe_parameter_count(config: VIBProbeConfig) -> int:
    """Return the exact parameter count without allocating the network."""
    config.validate()

    def linear(input_dim: int, output_dim: int, *, bias: bool = True) -> int:
        return input_dim * output_dim + (output_dim if bias else 0)

    layers, heads, head_dim = (
        int(config.num_layers),
        int(config.num_heads),
        int(config.head_dim),
    )
    adapter_parameters = 0
    if config.input_adapter == "flatten":
        adapter_output_dim = layers * heads * head_dim
    elif config.input_adapter == "head_projection":
        projected_dim = int(config.projected_head_dim)
        adapter_parameters = linear(head_dim, projected_dim) + 2 * projected_dim
        adapter_output_dim = layers * heads * projected_dim
    else:
        projected_dim = int(config.factorized_feature_dim)
        adapter_parameters = (
            linear(head_dim, projected_dim)
            + 2 * projected_dim
            + linear(heads, int(config.factorized_num_heads))
            + linear(layers, int(config.factorized_num_layers))
        )
        adapter_output_dim = (
            int(config.factorized_num_layers)
            * int(config.factorized_num_heads)
            * projected_dim
        )

    dimensions = [adapter_output_dim, *map(int, config.encoder_dims)]
    encoder_parameters = sum(
        linear(input_dim, output_dim) + 2 * output_dim
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
    )
    width = int(config.encoder_dims[-1])
    residual_parameters = int(config.residual_blocks) * (
        2 * linear(width, width) + 4 * width
    )
    bottleneck_parameters = 2 * linear(width, int(config.latent_dim))
    prior_parameters = 0
    classifier_input_dim = int(config.latent_dim)
    if config.prior_type == "gaussian_mixture":
        prior_parameters = int(config.mixture_components) * (
            2 * int(config.latent_dim) + 1
        )
        if config.mixture_readout:
            classifier_input_dim += int(config.mixture_components)
    classifier_parameters = linear(classifier_input_dim, 1)
    return int(
        adapter_parameters
        + encoder_parameters
        + residual_parameters
        + bottleneck_parameters
        + prior_parameters
        + classifier_parameters
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested VIB device `{value}`, but CUDA is unavailable.")
    return device


def vib_feature_tensor(
    features: Sequence[Mapping[str, Any]],
    feature_name: str = VIB_FEATURE_NAME,
    feature_layout: str = "layer_head",
) -> torch.Tensor:
    """Stack one VIB input source into ``[N, layers, heads, dim]``.

    The paper input already has ``[layers, heads, head_dim]`` layout.  A
    conventional probe vector such as ``last_token_h`` is represented as one
    synthetic layer and one synthetic head, so it can use the exact same VIB
    encoder, latent bottleneck, loss, and training loop.
    """
    if feature_layout not in VIB_FEATURE_LAYOUTS:
        raise ValueError(
            f"Unknown VIB feature layout `{feature_layout}`; expected one of "
            f"{VIB_FEATURE_LAYOUTS}."
        )
    rows = []
    for sample_index, sample in enumerate(features):
        value = sample.get(feature_name)
        if value is None:
            raise ValueError(
                f"Missing `{feature_name}` in sample {sample_index}; regenerate "
                "with the corresponding probe-feature collection enabled."
            )
        value = torch.as_tensor(value).detach().float().cpu()
        if feature_layout == "vector":
            value = value.reshape(-1).unsqueeze(0).unsqueeze(0)
        if value.ndim != 3 or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid VIB feature tensor in sample {sample_index}.")
        rows.append(value)
    if not rows:
        raise ValueError("No VIB features were supplied.")
    shapes = {tuple(value.shape) for value in rows}
    if len(shapes) != 1:
        raise ValueError(f"VIB feature shapes differ: {sorted(shapes)}")
    return torch.stack(rows, dim=0)


def _training_split(
    labels: np.ndarray, validation_fraction: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    classes = (labels >= 0.5).astype(np.int64)
    counts = np.bincount(classes, minlength=2)
    if labels.size < 6 or int(counts.min()) < 2:
        raise ValueError(
            "VIB-Probe requires at least six training samples and two samples "
            "from each correctness class."
        )
    indices = np.arange(labels.size)
    fit, holdout = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=classes,
    )
    return np.asarray(fit), np.asarray(holdout)


def _deterministic_vib_logits(
    model: VIBProbe,
    values: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Score CPU-backed features without moving the full dataset to the GPU."""
    rows: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            rows.append(model(batch, sample=False)[0].detach().cpu())
    return torch.cat(rows, dim=0)


def _deterministic_vib_responsibilities(
    model: VIBProbe,
    values: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return CPU-backed prototype responsibilities for a mixture VIB."""
    if model.config.prior_type != "gaussian_mixture":
        raise ValueError("Responsibilities are only defined for mixture VIB.")
    rows: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch = values[start : start + int(batch_size)].to(device)
            mu, log_variance = model.encode(batch)
            rows.append(
                model.prototype_responsibilities(mu, log_variance).detach().cpu()
            )
    return torch.cat(rows, dim=0)


def _mixture_responsibility_diagnostics(
    responsibilities: torch.Tensor,
    *,
    error_targets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Summarize component use, collapse, and optional component error rates."""
    values = responsibilities.detach().float().cpu().numpy()
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Mixture responsibilities must have shape [samples, K].")
    usage = np.mean(values, axis=0)
    usage_safe = np.clip(usage, 1e-12, 1.0)
    usage_entropy = float(-np.sum(usage_safe * np.log(usage_safe)))
    assignments = np.argmax(values, axis=1)
    counts = np.bincount(assignments, minlength=values.shape[1])
    sample_entropy = -np.sum(
        np.clip(values, 1e-12, 1.0) * np.log(np.clip(values, 1e-12, 1.0)),
        axis=1,
    )
    diagnostics: Dict[str, Any] = {
        "component_usage": usage.tolist(),
        "hard_assignment_counts": counts.astype(int).tolist(),
        "hard_active_component_count": int(np.sum(counts > 0)),
        "active_component_count_at_1pct": int(np.sum(usage >= 0.01)),
        "effective_component_count": float(np.exp(usage_entropy)),
        "mean_responsibility_entropy": float(np.mean(sample_entropy)),
        "mean_max_responsibility": float(np.mean(np.max(values, axis=1))),
    }
    if error_targets is not None:
        targets = np.asarray(error_targets, dtype=np.float64).reshape(-1)
        if targets.shape[0] != values.shape[0]:
            raise ValueError("Mixture responsibilities and error targets must align.")
        diagnostics["hard_component_error_rate"] = [
            float(np.mean(targets[assignments == component]))
            if np.any(assignments == component)
            else float("nan")
            for component in range(values.shape[1])
        ]
    return diagnostics


def _vib_holdout_loss(
    model: VIBProbe,
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    beta: float,
) -> Tuple[float, float, float]:
    """Return sample-weighted total, BCE, and KL validation losses."""
    bce_sum = 0.0
    kl_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), int(batch_size)):
            batch_x = values[start : start + int(batch_size)].to(device)
            batch_y = labels[start : start + int(batch_size)].to(device)
            logits, mu, log_variance = model(batch_x, sample=False)
            bce_sum += float(
                F.binary_cross_entropy_with_logits(
                    logits, batch_y, reduction="sum"
                ).item()
            )
            kl_sum += float(model.information_kl(mu, log_variance).sum().item())
            count += int(batch_x.shape[0])
    if count < 1:
        raise ValueError("VIB holdout split is empty.")
    bce = bce_sum / count
    kl = kl_sum / count
    return bce + float(beta) * kl, bce, kl


def fit_vib_probe_and_score(
    train_features: Sequence[Mapping[str, Any]],
    train_error_targets: Sequence[float],
    eval_features: Sequence[Mapping[str, Any]],
    *,
    config: Optional[VIBProbeTrainingConfig] = None,
    feature_name: str = VIB_FEATURE_NAME,
    feature_layout: str = "layer_head",
) -> Tuple[np.ndarray, Dict[str, Any], VIBProbe]:
    """Train VIB only on the train split and return eval error probabilities."""
    config = config or VIBProbeTrainingConfig()
    config.validate()
    x_train = vib_feature_tensor(
        train_features,
        feature_name=feature_name,
        feature_layout=feature_layout,
    )
    x_eval = vib_feature_tensor(
        eval_features,
        feature_name=feature_name,
        feature_layout=feature_layout,
    )
    if tuple(x_train.shape[1:]) != tuple(x_eval.shape[1:]):
        raise ValueError("VIB train/eval feature shapes differ.")
    labels = np.asarray(train_error_targets, dtype=np.float32).reshape(-1)
    if labels.shape[0] != x_train.shape[0]:
        raise ValueError("VIB feature and label counts differ.")
    if not np.isfinite(labels).all() or np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("VIB targets must be finite values in [0, 1].")

    fit_indices, holdout_indices = _training_split(
        labels, config.validation_fraction, config.random_seed
    )
    torch.manual_seed(int(config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.random_seed))
    device = _resolve_device(config.device)
    shape = tuple(int(value) for value in x_train.shape[1:])
    model_config = VIBProbeConfig(
        num_layers=shape[0],
        num_heads=shape[1],
        head_dim=shape[2],
        encoder_dims=tuple(config.encoder_dims),
        latent_dim=int(config.latent_dim),
        residual_blocks=int(config.residual_blocks),
        input_adapter=str(config.input_adapter),
        projected_head_dim=int(config.projected_head_dim),
        factorized_feature_dim=int(config.factorized_feature_dim),
        factorized_num_heads=int(config.factorized_num_heads),
        factorized_num_layers=int(config.factorized_num_layers),
        prior_type=str(config.prior_type),
        mixture_components=int(config.mixture_components),
        mixture_readout=bool(config.mixture_readout),
        mixture_mean_init_scale=float(config.mixture_mean_init_scale),
    )
    model = VIBProbe(model_config).to(device)
    parameter_count = int(sum(value.numel() for value in model.parameters()))
    expected_parameter_count = vib_probe_parameter_count(model_config)
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            "VIB parameter-count accounting mismatch: "
            f"model={parameter_count}, expected={expected_parameter_count}."
        )
    logging.info(
        "VIB MODEL | feature=%s | layout=%s | adapter=%s | prior=%s "
        "| mixture_components=%d | mixture_readout=%s | input_shape=%s "
        "| adapter_output_dim=%d "
        "| parameters=%d | estimated_fp32=%.2f MiB",
        feature_name,
        feature_layout,
        model_config.input_adapter,
        model_config.prior_type,
        model_config.mixture_components,
        model_config.mixture_readout,
        shape,
        model.adapter_output_dim,
        parameter_count,
        parameter_count * 4.0 / (1024.0 ** 2),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(int(config.random_seed))
    loader = DataLoader(
        TensorDataset(
            x_train[fit_indices],
            torch.from_numpy(labels[fit_indices]),
        ),
        batch_size=min(int(config.batch_size), len(fit_indices)),
        shuffle=True,
        generator=generator,
    )
    holdout_x = x_train[holdout_indices]
    holdout_y = torch.from_numpy(labels[holdout_indices])
    steps_per_epoch = max(len(loader), 1)
    warmup_steps = max(
        int(math.ceil(config.epochs * steps_per_epoch * config.beta_warmup_fraction)),
        1,
    )

    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale_epochs = 0
    global_step = 0
    history: List[Dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            beta = float(config.beta) * min((global_step + 1) / warmup_steps, 1.0)
            optimizer.zero_grad(set_to_none=True)
            logits, mu, log_variance = model(batch_x, sample=True)
            prediction_loss = criterion(logits, batch_y)
            kl_loss = model.information_kl(mu, log_variance).mean()
            loss = prediction_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
            global_step += 1

        holdout_loss, holdout_prediction, holdout_kl = _vib_holdout_loss(
            model,
            holdout_x,
            holdout_y,
            device=device,
            batch_size=config.batch_size,
            beta=config.beta,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss / max(batches, 1),
                "holdout_loss": holdout_loss,
                "holdout_bce": float(holdout_prediction),
                "holdout_kl": float(holdout_kl),
            }
        )
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
            if config.patience and stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("VIB-Probe did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    eval_logits = _deterministic_vib_logits(
        model, x_eval, device=device, batch_size=config.batch_size
    )
    training_logits_tensor = _deterministic_vib_logits(
        model, x_train, device=device, batch_size=config.batch_size
    )
    eval_scores = torch.sigmoid(eval_logits).numpy()
    training_logits = training_logits_tensor.numpy()
    holdout_scores = torch.sigmoid(
        training_logits_tensor[holdout_indices]
    ).numpy()
    if not np.isfinite(eval_scores).all():
        raise FloatingPointError("VIB-Probe produced non-finite probabilities.")
    mixture_metadata: Optional[Dict[str, Any]] = None
    if model_config.prior_type == "gaussian_mixture":
        train_responsibilities = _deterministic_vib_responsibilities(
            model,
            x_train,
            device=device,
            batch_size=config.batch_size,
        )
        eval_responsibilities = _deterministic_vib_responsibilities(
            model,
            x_eval,
            device=device,
            batch_size=config.batch_size,
        )
        prior_weights = F.softmax(
            model.prior_mixture_logits.detach().cpu(), dim=0
        )
        component_variances = torch.exp(
            model.prior_component_log_variances.detach().cpu().clamp(-6.0, 4.0)
        )
        mixture_metadata = {
            "components": int(model_config.mixture_components),
            "readout_uses_responsibilities": bool(model_config.mixture_readout),
            "kl_approximation": (
                "-logsumexp(log_pi_k - KL(q(z|x)||p_k(z)))"
            ),
            "prior_weights": prior_weights.tolist(),
            "component_mean_l2_norm": torch.linalg.vector_norm(
                model.prior_component_means.detach().cpu(), dim=-1
            ).tolist(),
            "component_mean_variance": component_variances.mean(dim=-1).tolist(),
            "train": _mixture_responsibility_diagnostics(
                train_responsibilities,
                error_targets=labels,
            ),
            "eval": _mixture_responsibility_diagnostics(eval_responsibilities),
            "eval_prototype_responsibilities": (
                eval_responsibilities.numpy().astype(np.float32).tolist()
            ),
        }
        logging.info(
            "VIB MIXTURE | components=%d | readout=%s | train_effective=%.3f "
            "| eval_effective=%.3f | train_active=%d | eval_active=%d",
            model_config.mixture_components,
            model_config.mixture_readout,
            mixture_metadata["train"]["effective_component_count"],
            mixture_metadata["eval"]["effective_component_count"],
            mixture_metadata["train"]["active_component_count_at_1pct"],
            mixture_metadata["eval"]["active_component_count_at_1pct"],
        )
    holdout_binary = (labels[holdout_indices] >= 0.5).astype(np.int64)
    holdout_auroc = (
        float(roc_auc_score(holdout_binary, holdout_scores))
        if np.unique(holdout_binary).size == 2
        else float("nan")
    )
    threshold_logit = float(np.mean(training_logits))
    baseline_config = VIBProbeConfig(
        num_layers=shape[0],
        num_heads=shape[1],
        head_dim=shape[2],
        encoder_dims=tuple(config.encoder_dims),
        latent_dim=int(config.latent_dim),
        residual_blocks=int(config.residual_blocks),
        input_adapter="flatten",
        prior_type=str(config.prior_type),
        mixture_components=int(config.mixture_components),
        mixture_readout=bool(config.mixture_readout),
        mixture_mean_init_scale=float(config.mixture_mean_init_scale),
    )
    baseline_parameter_count = vib_probe_parameter_count(baseline_config)
    parameter_reduction_fraction = (
        1.0 - parameter_count / baseline_parameter_count
        if baseline_parameter_count else 0.0
    )
    metadata: Dict[str, Any] = {
        "method": "VIB-Probe",
        "paper": "Zhang et al. (2026), arXiv:2601.05547v2",
        "feature_name": feature_name,
        "feature_layout": feature_layout,
        "input_shape": list(shape),
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "label_semantics": "0=faithful/correct, 1=hallucinated/error",
        "score_semantics": "hallucination/error probability",
        "train_samples": int(x_train.shape[0]),
        "fit_samples": int(fit_indices.size),
        "holdout_samples": int(holdout_indices.size),
        "eval_samples": int(x_eval.shape[0]),
        "best_epoch": int(best_epoch),
        "best_holdout_loss": float(best_loss),
        "holdout_auroc": holdout_auroc,
        "training_history": history,
        # Appendix B.3 triggers mitigation at the mean training logit.
        "mitigation_threshold_logit": threshold_logit,
        "mitigation_top_fraction": 0.05,
        "mitigation_strength": 0.001,
        "parameter_count": parameter_count,
        "baseline_flatten_parameter_count": int(baseline_parameter_count),
        "parameter_reduction_fraction": float(parameter_reduction_fraction),
        "estimated_fp32_parameter_mib": float(
            parameter_count * 4.0 / (1024.0 ** 2)
        ),
        "input_adapter": model_config.input_adapter,
        "adapter_output_dim": int(model.adapter_output_dim),
        "prior_type": model_config.prior_type,
        "mixture_components": int(model_config.mixture_components),
        "mixture_readout": bool(model_config.mixture_readout),
        "mixture_diagnostics": mixture_metadata,
        "input_adapter_description": {
            "flatten": (
                "flatten complete L*H*d_head tensor, then project to encoder"
            ),
            "head_projection": (
                "shared per-head feature projection, preserve every L/H slot, "
                "then flatten"
            ),
            "factorized": (
                "shared feature projection plus learned head-axis and layer-axis "
                "compression, then flatten"
            ),
        }[model_config.input_adapter],
    }
    logging.info(
        "VIB RESULT | adapter=%s | epochs=%d | best_epoch=%d "
        "| holdout_auroc=%.6f | parameters=%d | reduction_vs_flatten=%.2f%%",
        model_config.input_adapter,
        len(history),
        best_epoch,
        holdout_auroc,
        parameter_count,
        100.0 * parameter_reduction_fraction,
    )
    return eval_scores.astype(np.float64), metadata, model.cpu()


@dataclass(frozen=True)
class VIBMitigationConfig:
    """Inference-time intervention defaults from Appendix B.3."""

    threshold_logit: Optional[float] = None
    top_fraction: float = 0.05
    strength: float = 0.001
    minimum_scale: Optional[float] = None

    def validate(self) -> None:
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("VIB mitigation top_fraction must be in (0, 1].")
        if self.strength < 0:
            raise ValueError("VIB mitigation strength must be non-negative.")
        if self.minimum_scale is not None and self.minimum_scale > 1.0:
            raise ValueError("VIB minimum_scale cannot exceed 1.")


def vib_head_attribution_and_scales(
    model: VIBProbe,
    head_outputs: torch.Tensor,
    *,
    top_fraction: float = 0.05,
    strength: float = 0.001,
    minimum_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Implement Eqs. (13)-(14) and return ``[layers, heads]`` scales."""
    config = VIBMitigationConfig(
        top_fraction=top_fraction,
        strength=strength,
        minimum_scale=minimum_scale,
    )
    config.validate()
    if head_outputs.ndim != 4 or head_outputs.shape[0] != 1:
        raise ValueError("VIB mitigation currently requires one sample at a time.")
    if not head_outputs.requires_grad:
        head_outputs.requires_grad_(True)
    model.eval()
    risk_logit = model(head_outputs, sample=False)[0][0]
    gradients = torch.autograd.grad(
        risk_logit, head_outputs, retain_graph=False, create_graph=False
    )[0]
    signed_sensitivity = (gradients * head_outputs).sum(dim=-1)[0]
    importance = signed_sensitivity.abs()
    total_heads = int(importance.numel())
    selected_count = max(1, int(math.ceil(total_heads * float(top_fraction))))
    flat_indices = torch.topk(
        importance.flatten(), k=selected_count, largest=True
    ).indices
    mask = torch.zeros(total_heads, dtype=torch.bool, device=importance.device)
    mask[flat_indices] = True
    mask = mask.reshape_as(importance)
    scales = torch.ones_like(importance)
    updates = 1.0 - float(strength) * F.relu(signed_sensitivity)
    if minimum_scale is not None:
        updates = updates.clamp_min(float(minimum_scale))
    scales = torch.where(mask, updates, scales)
    selected = []
    for flat_index in flat_indices.detach().cpu().tolist():
        layer = int(flat_index // importance.shape[1])
        head = int(flat_index % importance.shape[1])
        selected.append(
            {
                "layer": layer,
                "head": head,
                "importance": float(importance[layer, head].detach().cpu()),
                "signed_sensitivity": float(
                    signed_sensitivity[layer, head].detach().cpu()
                ),
                "scale": float(scales[layer, head].detach().cpu()),
            }
        )
    details = {
        "risk_logit": float(risk_logit.detach().cpu()),
        "risk_probability": float(torch.sigmoid(risk_logit).detach().cpu()),
        "selected_head_count": selected_count,
        "total_head_count": total_heads,
        "selected_heads": selected,
    }
    return scales.detach(), details


def save_vib_probe_checkpoint(
    path: str,
    model: VIBProbe,
    metadata: Mapping[str, Any],
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": asdict(model.config),
        "metadata": dict(metadata),
    }
    torch.save(payload, destination)
    return str(destination)


def load_vib_probe_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> Tuple[VIBProbe, Dict[str, Any]]:
    resolved = _resolve_device(device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    model_config = dict(payload["model_config"])
    model_config["encoder_dims"] = tuple(model_config["encoder_dims"])
    model = VIBProbe(VIBProbeConfig(**model_config)).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
