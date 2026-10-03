"""Layer-wise answer-token decision and hidden-state commitment metrics.

The hidden trajectory is causally aligned: state ``h[l, t]`` is the decoder
state at the query position which predicts answer token ``y[t]``.  This avoids
the common one-token teacher-forcing leak where the state after seeing ``y[t]``
is used to score ``y[t]`` itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from uncertainty.uncertainty_measures.internal_trajectory import coe_metrics


EPS = 1e-12


@dataclass(frozen=True)
class CommitmentThresholds:
    """Thresholds use normalized entropy and natural-log JSD/KL units."""

    entropy: float = 0.30
    margin: float = 0.50
    target_top_k: int = 5
    jsd_to_final: float = 0.05
    hidden_cosine: float = 0.95

    def validate(self) -> None:
        if not 0.0 <= self.entropy <= 1.0:
            raise ValueError("entropy threshold must be in [0, 1].")
        if not 0.0 <= self.margin <= 1.0:
            raise ValueError("margin threshold must be in [0, 1].")
        if self.target_top_k < 1:
            raise ValueError("target_top_k must be positive.")
        if not 0.0 <= self.jsd_to_final <= math.log(2.0):
            raise ValueError("jsd_to_final must be in [0, log(2)].")
        if not -1.0 <= self.hidden_cosine <= 1.0:
            raise ValueError("hidden_cosine must be in [-1, 1].")


def normalized_energy_entropy(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Entropy of squared activation energy, normalized to [0, 1]."""
    energy = values.float().square()
    total = energy.sum(dim=dim, keepdim=True)
    probabilities = energy / total.clamp_min(EPS)
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum(dim=dim)
    size = values.shape[dim]
    if size < 2:
        return torch.zeros_like(entropy)
    return entropy / math.log(size)


def first_sustained_depth(
    condition: torch.Tensor, layer_ids: Sequence[int]
) -> torch.Tensor:
    """First physical layer satisfying a condition through all later layers.

    ``condition`` has shape ``[layers, tokens]``.  Tokens that never commit are
    encoded as ``-1`` rather than being silently assigned the final layer.
    """
    values = torch.as_tensor(condition, dtype=torch.bool)
    if values.ndim != 2 or values.shape[0] != len(layer_ids):
        raise ValueError("condition must have shape [layers, tokens].")
    sustained = torch.flip(
        torch.cumprod(torch.flip(values.to(torch.int8), dims=(0,)), dim=0),
        dims=(0,),
    ).bool()
    result = torch.full((values.shape[1],), -1, dtype=torch.int64)
    for index, layer_id in enumerate(layer_ids):
        newly = sustained[index] & result.eq(-1)
        result[newly] = int(layer_id)
    return result


def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _project_log_probs(
    states: torch.Tensor,
    *,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_device, norm_dtype = _module_device_dtype(final_norm)
    head_device, _ = _module_device_dtype(lm_head)
    values = states.to(device=norm_device, dtype=norm_dtype)
    values = final_norm(values)
    if values.device != head_device:
        values = values.to(head_device)
    logits = lm_head(values).float()
    log_probs = F.log_softmax(logits, dim=-1)
    return logits.detach().cpu(), log_probs.detach().cpu()


def _jsd(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p, q = log_p.exp(), log_q.exp()
    mixture = 0.5 * (p + q)
    log_mixture = mixture.clamp_min(EPS).log()
    return 0.5 * (
        (p * (log_p - log_mixture)).sum(dim=-1)
        + (q * (log_q - log_mixture)).sum(dim=-1)
    )


def _depth_summary(depths: torch.Tensor, final_layer: int) -> dict[str, float]:
    valid = depths[depths.ge(0)].float()
    return {
        "committed_fraction": float(valid.numel() / max(depths.numel(), 1)),
        "depth_mean_committed": float(valid.mean()) if valid.numel() else float("nan"),
        "depth_median_committed": float(valid.median()) if valid.numel() else float("nan"),
        "relative_depth_mean_committed": (
            float(valid.mean() / max(final_layer, 1)) if valid.numel() else float("nan")
        ),
    }


@torch.inference_mode()
def compute_decision_commitment_features(
    hidden_trajectory: Any,
    target_token_ids: Any,
    *,
    final_norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    layer_ids: Sequence[int] | None = None,
    head_energy_entropy: Any | None = None,
    head_layer_ids: Sequence[int] | None = None,
    thresholds: CommitmentThresholds | None = None,
) -> dict[str, Any]:
    """Compute memory-bounded logit-lens and residual-stream trajectories.

    Parameters
    ----------
    hidden_trajectory:
        Tensor ``[answer_tokens, decoder_layers, hidden_dim]`` containing the
        post-block state at the *predictive query* for each target token.
    target_token_ids:
        Cached/generated answer tokens, shape ``[answer_tokens]``.

    Full vocabulary distributions are projected one layer at a time and never
    retained in the returned artifact.
    """
    hidden = torch.as_tensor(hidden_trajectory).detach().float().cpu()
    targets = torch.as_tensor(target_token_ids, dtype=torch.long).reshape(-1).cpu()
    if hidden.ndim != 3 or min(hidden.shape) < 1:
        raise ValueError("hidden_trajectory must be [tokens, layers, hidden_dim].")
    if targets.shape[0] != hidden.shape[0]:
        raise ValueError("One target token id is required per hidden-state token.")
    ids = list(range(hidden.shape[1])) if layer_ids is None else [int(x) for x in layer_ids]
    if len(ids) != hidden.shape[1] or len(set(ids)) != len(ids):
        raise ValueError("layer_ids must uniquely identify every decoder layer.")
    config = thresholds or CommitmentThresholds()
    config.validate()

    layers, tokens = hidden.shape[1], hidden.shape[0]
    final_logits, final_log_probs = _project_log_probs(
        hidden[:, -1], final_norm=final_norm, lm_head=lm_head
    )
    vocabulary = final_logits.shape[-1]
    if targets.min() < 0 or targets.max() >= vocabulary:
        raise ValueError("A target token id is outside the output vocabulary.")

    entropy = torch.empty((layers, tokens), dtype=torch.float32)
    entropy_normalized = torch.empty_like(entropy)
    top1_probability = torch.empty_like(entropy)
    top2_probability = torch.empty_like(entropy)
    margin = torch.empty_like(entropy)
    target_probability = torch.empty_like(entropy)
    target_rank = torch.empty((layers, tokens), dtype=torch.int32)
    top1_token_id = torch.empty((layers, tokens), dtype=torch.int32)
    jsd_to_final = torch.empty_like(entropy)
    kl_final_to_layer = torch.empty_like(entropy)
    adjacent_jsd = torch.full_like(entropy, float("nan"))
    previous_log_probs: torch.Tensor | None = None

    for layer_index in range(layers):
        if layer_index == layers - 1:
            logits, log_probs = final_logits, final_log_probs
        else:
            logits, log_probs = _project_log_probs(
                hidden[:, layer_index], final_norm=final_norm, lm_head=lm_head
            )
        probabilities = log_probs.exp()
        top_values, top_indices = probabilities.topk(k=2, dim=-1)
        target_logits = logits.gather(1, targets[:, None]).squeeze(1)
        entropy[layer_index] = -(probabilities * log_probs).sum(dim=-1)
        entropy_normalized[layer_index] = entropy[layer_index] / math.log(vocabulary)
        top1_probability[layer_index] = top_values[:, 0]
        top2_probability[layer_index] = top_values[:, 1]
        margin[layer_index] = top_values[:, 0] - top_values[:, 1]
        target_probability[layer_index] = probabilities.gather(
            1, targets[:, None]
        ).squeeze(1)
        target_rank[layer_index] = (
            logits.gt(target_logits[:, None]).sum(dim=-1) + 1
        ).to(torch.int32)
        top1_token_id[layer_index] = top_indices[:, 0].to(torch.int32)
        jsd_to_final[layer_index] = _jsd(log_probs, final_log_probs)
        kl_final_to_layer[layer_index] = (
            final_log_probs.exp() * (final_log_probs - log_probs)
        ).sum(dim=-1)
        if previous_log_probs is not None:
            adjacent_jsd[layer_index] = _jsd(log_probs, previous_log_probs)
        previous_log_probs = log_probs
        del logits, log_probs, probabilities, top_values, top_indices, target_logits

    hidden_by_layer = hidden.permute(1, 0, 2)
    hidden_cosine_to_final = F.cosine_similarity(
        hidden_by_layer, hidden_by_layer[-1:].expand_as(hidden_by_layer), dim=-1
    )
    deltas = hidden_by_layer[1:] - hidden_by_layer[:-1]
    hidden_delta_norm = torch.linalg.vector_norm(deltas, dim=-1)
    if layers > 2:
        hidden_delta_cosine = F.cosine_similarity(deltas[1:], deltas[:-1], dim=-1)
    else:
        hidden_delta_cosine = torch.empty((0, tokens), dtype=torch.float32)
    hidden_entropy = normalized_energy_entropy(hidden_by_layer, dim=-1)

    final_top1 = top1_token_id[-1]
    depths = {
        "entropy_commitment_depth": first_sustained_depth(
            entropy_normalized.le(config.entropy), ids
        ),
        "margin_commitment_depth": first_sustained_depth(
            margin.ge(config.margin), ids
        ),
        "target_topk_commitment_depth": first_sustained_depth(
            target_rank.le(config.target_top_k), ids
        ),
        "vocab_convergence_depth": first_sustained_depth(
            jsd_to_final.le(config.jsd_to_final), ids
        ),
        "hidden_convergence_depth": first_sustained_depth(
            hidden_cosine_to_final.ge(config.hidden_cosine), ids
        ),
        "top1_commitment_depth": first_sustained_depth(
            top1_token_id.eq(final_top1[None, :]), ids
        ),
    }
    depth_summaries = {
        name: _depth_summary(values, ids[-1]) for name, values in depths.items()
    }
    token_flip_count = top1_token_id[1:].ne(top1_token_id[:-1]).sum(dim=0).to(torch.int16)

    output: dict[str, Any] = {
        "schema_version": 1,
        "alignment": "causal_predictive_query_for_each_answer_token",
        "layer_ids": torch.tensor(ids, dtype=torch.int16),
        "target_token_ids": targets,
        "thresholds": asdict(config),
        "vocab_size": int(vocabulary),
        "vocab_entropy": entropy.to(torch.float16),
        "vocab_entropy_normalized": entropy_normalized.to(torch.float16),
        "top1_probability": top1_probability.to(torch.float16),
        "top2_probability": top2_probability.to(torch.float16),
        "top1_top2_margin": margin.to(torch.float16),
        "target_probability": target_probability.to(torch.float16),
        "target_rank": target_rank,
        "top1_token_id": top1_token_id,
        "jsd_to_final": jsd_to_final.to(torch.float16),
        "kl_final_to_layer": kl_final_to_layer.to(torch.float16),
        "adjacent_jsd": adjacent_jsd.to(torch.float16),
        "top1_flip_count": token_flip_count,
        "hidden_cosine_to_final": hidden_cosine_to_final.to(torch.float16),
        "hidden_delta_norm": hidden_delta_norm.to(torch.float16),
        "hidden_delta_cosine": hidden_delta_cosine.to(torch.float16),
        "hidden_channel_energy_entropy": hidden_entropy.to(torch.float16),
        "hidden_layer_mean": hidden.mean(dim=0).to(torch.float16),
        "commitment_depths": depths,
        "commitment_summaries": depth_summaries,
        "coe_mean": coe_metrics(hidden.numpy(), token_pool="mean"),
        "coe_last": coe_metrics(hidden.numpy(), token_pool="last"),
    }
    if head_energy_entropy is not None:
        head_values = torch.as_tensor(head_energy_entropy).detach().float().cpu()
        if head_values.ndim != 2 or head_values.shape[1] != tokens:
            raise ValueError("head_energy_entropy must be [full_layers, tokens].")
        head_ids = [int(x) for x in (head_layer_ids or [])]
        if len(head_ids) != head_values.shape[0]:
            raise ValueError("One head_layer_id is required per head entropy layer.")
        output["head_layer_ids"] = torch.tensor(head_ids, dtype=torch.int16)
        output["head_output_energy_entropy"] = head_values.to(torch.float16)
    return output

