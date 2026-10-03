"""ICR Score extraction and the ICR hallucination probe.

This module is an integration-oriented reproduction of ICR Probe from Zhang
et al. (ACL 2025).  It follows the paper's equations and the reference
implementation at https://github.com/XavierZhang2002/ICR_Probe (revision
40ec490e762cadbac6bcefdc24a8f0d5974e8448), with numerical and data-loading
fixes needed by this project.

The saved/probed score is an *error probability*: 0 means faithful and 1
means hallucinated.  This matches the uncertainty convention used throughout
this repository and Algorithm 2 in the paper.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


ICR_FEATURE_NAME = "icr_layer_mean"
ICR_MATRIX_NAME = "icr_scores"
ICR_SOURCE_REVISION = "40ec490e762cadbac6bcefdc24a8f0d5974e8448"


def _move_tensors(container: Any, device: torch.device) -> Any:
    """Recursively move tensors while preserving generation-output nesting."""
    if torch.is_tensor(container):
        return container.to(device)
    if isinstance(container, list):
        return [_move_tensors(value, device) for value in container]
    if isinstance(container, tuple):
        return tuple(_move_tensors(value, device) for value in container)
    if isinstance(container, dict):
        return {key: _move_tensors(value, device) for key, value in container.items()}
    return container


def _as_tensor_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)) or len(value) == 0:
        raise ValueError(f"`{name}` must be a non-empty tuple/list.")
    return value


def _unwrap_step_layers(step: Any, name: str) -> Sequence[torch.Tensor]:
    """Undo the extra singleton tuple produced by some generation patches."""
    while (
        isinstance(step, (tuple, list))
        and len(step) == 1
        and isinstance(step[0], (tuple, list))
    ):
        step = step[0]
    if not isinstance(step, (tuple, list)) or not step:
        raise ValueError(f"Each `{name}` step must contain layer tensors.")
    if not all(torch.is_tensor(item) for item in step):
        raise ValueError(f"Each `{name}` layer must be a tensor.")
    return step


def _stable_standardize(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Standardize without the one-element NaN in torch's unbiased std."""
    values = values.float()
    scale = values.std(unbiased=False).clamp_min(epsilon)
    return (values - values.mean()) / scale


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Reference-compatible JSD after z-score and softmax normalization."""
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape or p.numel() == 0:
        raise ValueError("JSD inputs must be non-empty vectors with equal shape.")
    p_prob, q_prob = _direction_probabilities(p, q)
    mixture = 0.5 * (p_prob + q_prob)
    p_kl = torch.sum(p_prob * (torch.log(p_prob) - torch.log(mixture)))
    q_kl = torch.sum(q_prob * (torch.log(q_prob) - torch.log(mixture)))
    return 0.5 * (p_kl + q_kl)


def _direction_probabilities(
    projection: torch.Tensor, attention: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the paper's Proj and Attn directions on a shared key set."""
    if (
        projection.ndim != 1
        or attention.ndim != 1
        or projection.shape != attention.shape
        or projection.numel() == 0
    ):
        raise ValueError("ICR directions must be non-empty vectors with equal shape.")
    return (
        F.softmax(_stable_standardize(projection), dim=0),
        F.softmax(_stable_standardize(attention), dim=0),
    )


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Entropy in [0, 1], with the one-key direction defined as zero."""
    if probabilities.numel() <= 1:
        return probabilities.new_zeros(())
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12))
    )
    return entropy / math.log(int(probabilities.numel()))


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compatibility helper exposed by the upstream implementation."""
    p = torch.as_tensor(p).float()
    q = torch.as_tensor(q).float()
    if p.shape != q.shape or p.numel() == 0:
        raise ValueError("KL inputs must be non-empty tensors with equal shape.")
    return float(torch.sum(p * (torch.log(p) - torch.log(q))).item())


@dataclass(frozen=True)
class ICRScoreConfig:
    """Configuration for Equation 7 and its published ablations."""

    top_k: Optional[int] = 20
    top_p: Optional[float] = None
    pooling: str = "mean"
    attention_scope: str = "all"
    attention_uniform: bool = False
    hidden_uniform: bool = False
    use_induction_head: bool = False
    skew_threshold: float = 0.0
    entropy_threshold: float = 1e5
    save_direction_vectors: bool = False

    def validate(self) -> None:
        if self.top_k is not None and int(self.top_k) < 1:
            raise ValueError("ICR top_k must be positive or None.")
        if self.top_p is not None and not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("ICR top_p must be in (0, 1].")
        if self.pooling not in {"mean", "max", "min"}:
            raise ValueError("ICR pooling must be mean, max, or min.")
        if self.attention_scope not in {"all", "prompt", "response"}:
            raise ValueError("ICR attention_scope must be all, prompt, or response.")


class ICRScore:
    """Compute token-by-layer ICR scores from Hugging Face internals.

    Two cache layouts are supported:

    * ``dense``: one teacher-forced forward pass. ``hidden_states`` is L+1
      tensors of shape ``[batch, sequence, hidden]`` and ``attentions`` is L
      tensors of shape ``[batch, heads, sequence, sequence]``.
    * ``generate``: the nested per-step cache returned by ``generate``. This
      preserves compatibility with the public ICR_Probe quick-start API.

    ``core_positions`` uses half-open indices. ``response_start`` is required
    for dense caches; generation caches infer it from the prefill length.
    """

    def __init__(
        self,
        hidden_states: Sequence[Any],
        attentions: Sequence[Any],
        skew_threshold: float = 0.0,
        entropy_threshold: float = 1e5,
        core_positions: Optional[Mapping[str, int]] = None,
        icr_device: Optional[str] = None,
        *,
        cache_format: str = "auto",
        layer_ids: Optional[Sequence[int]] = None,
        attention_query_start: Optional[int] = None,
    ):
        hidden_states = _as_tensor_sequence(hidden_states, "hidden_states")
        attentions = _as_tensor_sequence(attentions, "attentions")
        first_hidden = hidden_states[0]
        inferred = "dense" if torch.is_tensor(first_hidden) else "generate"
        if cache_format == "auto":
            cache_format = inferred
        if cache_format not in {"dense", "generate"}:
            raise ValueError("cache_format must be auto, dense, or generate.")
        if cache_format != inferred:
            raise ValueError(
                f"Requested {cache_format} cache but inputs look like {inferred}."
            )

        first_tensor = (
            first_hidden
            if torch.is_tensor(first_hidden)
            else _unwrap_step_layers(first_hidden, "hidden_states")[0]
        )
        self.original_device = first_tensor.device
        self.icr_device = torch.device(icr_device) if icr_device else self.original_device
        # An explicit target also gathers model-parallel layer outputs whose
        # first tensor already happens to be on that target device.
        if icr_device is not None:
            hidden_states = _move_tensors(hidden_states, self.icr_device)
            attentions = _move_tensors(attentions, self.icr_device)

        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cache_format = cache_format
        self.skew_threshold = float(skew_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.core_positions = dict(core_positions or {})
        self._supplied_layer_ids = None if layer_ids is None else tuple(map(int, layer_ids))
        self._attention_query_start = attention_query_start
        self._prepare_layout()

    def _prepare_layout(self) -> None:
        if self.cache_format == "dense":
            if not all(torch.is_tensor(item) and item.ndim == 3 for item in self.hidden_states):
                raise ValueError("Dense hidden states must have shape [batch, sequence, hidden].")
            if not all(torch.is_tensor(item) and item.ndim == 4 for item in self.attentions):
                raise ValueError(
                    "Dense attentions must have shape [batch, heads, query, key]."
                )
            self.num_layers = len(self.attentions)
            if self._supplied_layer_ids is None:
                self.model_layer_ids = tuple(range(self.num_layers))
            else:
                if len(self._supplied_layer_ids) != self.num_layers:
                    raise ValueError("layer_ids must contain one physical layer per attention tensor.")
                if len(set(self._supplied_layer_ids)) != self.num_layers:
                    raise ValueError("layer_ids must be unique.")
                self.model_layer_ids = self._supplied_layer_ids
            if min(self.model_layer_ids) < 0 or max(self.model_layer_ids) + 1 >= len(self.hidden_states):
                raise ValueError(
                    "Dense hidden states do not cover every physical layer in layer_ids."
                )
            sequence_length = int(self.hidden_states[0].shape[1])
            self.response_start = int(self.core_positions.get("response_start", 0))
            self.response_end = int(self.core_positions.get("response_end", sequence_length))
            self.prompt_start = int(self.core_positions.get("user_prompt_start", 0))
            self.prompt_end = int(
                self.core_positions.get("user_prompt_end", self.response_start)
            )
            if not 0 <= self.response_start < self.response_end <= sequence_length:
                raise ValueError("Invalid dense response_start/response_end positions.")
            self.num_tokens = self.response_end - self.response_start
            query_lengths = {int(item.shape[-2]) for item in self.attentions}
            if len(query_lengths) != 1:
                raise ValueError("Dense attention query lengths are inconsistent.")
            query_length = next(iter(query_lengths))
            if self._attention_query_start is None:
                if query_length == sequence_length:
                    self.attention_query_start = 0
                elif query_length == self.num_tokens:
                    self.attention_query_start = self.response_start
                else:
                    raise ValueError(
                        "Dense attention queries must cover the full sequence or exactly "
                        "the configured response span; otherwise set attention_query_start."
                    )
            else:
                self.attention_query_start = int(self._attention_query_start)
            if not (
                self.attention_query_start <= self.response_start
                and self.response_end
                <= self.attention_query_start + query_length
            ):
                raise ValueError("Dense attention rows do not cover the response span.")
            return

        hidden_steps = [
            _unwrap_step_layers(step, "hidden_states") for step in self.hidden_states
        ]
        attention_steps = [
            _unwrap_step_layers(step, "attentions") for step in self.attentions
        ]
        self.hidden_states = hidden_steps
        self.attentions = attention_steps
        self.num_layers = len(attention_steps[0])
        if self._supplied_layer_ids is not None:
            raise ValueError("layer_ids is currently supported only for dense caches.")
        self.model_layer_ids = tuple(range(self.num_layers))
        if len(hidden_steps[0]) != self.num_layers + 1:
            raise ValueError("Generation ICR needs L+1 hidden states per step.")
        if any(len(step) != self.num_layers for step in attention_steps):
            raise ValueError("Generation attention layer counts are inconsistent.")
        if any(len(step) != self.num_layers + 1 for step in hidden_steps):
            raise ValueError("Generation hidden-state layer counts are inconsistent.")
        self.response_start = int(hidden_steps[0][0].shape[1])
        supplied_start = self.core_positions.get("response_start")
        if supplied_start is not None and int(supplied_start) != self.response_start:
            logging.debug(
                "Generation response_start=%d is inferred from prefill; ignoring supplied %d.",
                self.response_start,
                int(supplied_start),
            )
        self.prompt_start = int(self.core_positions.get("user_prompt_start", 0))
        self.prompt_end = int(
            self.core_positions.get("user_prompt_end", self.response_start)
        )
        # Step 0 predicts the first emitted token. Step 1 contains that token's
        # state and attention, so the last emitted token has no cached state.
        self.num_tokens = min(len(hidden_steps), len(attention_steps)) - 1
        self.response_end = self.response_start + max(self.num_tokens, 0)
        if self.num_tokens < 1:
            raise ValueError(
                "Generation cache contains no response-token state. Use a dense "
                "teacher-forced pass for one-token answers."
            )

    def _attention_heads(self, layer: int, token: int) -> torch.Tensor:
        if self.cache_format == "dense":
            query = self.response_start + token
            matrix = self.attentions[layer]
            if matrix.shape[0] != 1:
                raise ValueError("ICR extraction currently requires batch size 1.")
            key_end = min(query + 1, int(matrix.shape[-1]))
            query_row = query - self.attention_query_start
            return matrix[0, :, query_row, :key_end].float()

        step_matrix = self.attentions[token + 1][layer]
        if step_matrix.ndim != 4 or step_matrix.shape[0] != 1:
            raise ValueError("Generation attention must be [1, heads, query, key].")
        return step_matrix[0, :, -1, :].float()

    def _hidden_values(
        self, layer: int, token: int, key_end: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache_format == "dense":
            query = self.response_start + token
            physical_layer = self.model_layer_ids[layer]
            previous = self.hidden_states[physical_layer]
            current = self.hidden_states[physical_layer + 1]
            if previous.shape[0] != 1 or current.shape[0] != 1:
                raise ValueError("ICR extraction currently requires batch size 1.")
            return (
                current[0, query].float(),
                previous[0, query].float(),
                previous[0, :key_end].float(),
            )

        current_step = self.hidden_states[token + 1]
        current = current_step[layer + 1][0, -1].float()
        previous = current_step[layer][0, -1].float()
        context_parts = [self.hidden_states[0][layer][0].float()]
        context_parts.extend(
            self.hidden_states[step][layer][0, -1:].float()
            for step in range(1, token + 2)
        )
        context = torch.cat(context_parts, dim=0)
        return current, previous, context[:key_end]

    def _key_bounds(self, key_end: int) -> Tuple[int, int]:
        if self._active_config.attention_scope == "all":
            return 0, key_end
        if self._active_config.attention_scope == "prompt":
            return max(0, self.prompt_start), min(key_end, self.prompt_end)
        return min(key_end, self.response_start), key_end

    @staticmethod
    def _skewness_entropy(rows: torch.Tensor) -> Tuple[float, float]:
        """Mean positional skewness/entropy for one head's valid rows."""
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        row_sums = rows.sum(dim=-1, keepdim=True)
        valid = row_sums.squeeze(-1) > 0
        if not bool(valid.any()):
            return float("-inf"), float("inf")
        probabilities = rows[valid] / row_sums[valid].clamp_min(1e-12)
        indices = torch.arange(
            1,
            probabilities.shape[-1] + 1,
            device=probabilities.device,
            dtype=probabilities.dtype,
        ).unsqueeze(0)
        means = torch.sum(probabilities * indices, dim=-1, keepdim=True)
        centered = indices - means
        variance = torch.sum(probabilities * centered.square(), dim=-1)
        third = torch.sum(probabilities * centered.pow(3), dim=-1)
        skewness = third / variance.clamp_min(1e-12).pow(1.5)
        entropy = -torch.sum(
            probabilities * torch.log2(probabilities.clamp_min(1e-12)), dim=-1
        )
        return float(skewness.mean().item()), float(entropy.mean().item())

    def _selected_heads(self, layer: int) -> Optional[torch.LongTensor]:
        if not self._active_config.use_induction_head:
            return None
        # Selection is performed over response rows. Padding to a common width
        # is used only for this optional upstream ablation, never for ICR top-k.
        token_rows = [self._attention_heads(layer, token) for token in range(self.num_tokens)]
        num_heads = int(token_rows[0].shape[0])
        max_keys = max(int(row.shape[-1]) for row in token_rows)
        padded = [F.pad(row, (0, max_keys - row.shape[-1])) for row in token_rows]
        stacked = torch.stack(padded, dim=1)  # [heads, tokens, keys]
        skewness = []
        selected = []
        for head in range(num_heads):
            skew, entropy = self._skewness_entropy(stacked[head])
            skewness.append(skew)
            selected.append(
                skew >= self.skew_threshold and entropy <= self.entropy_threshold
            )
        minimum = max(num_heads // 8, 1)
        if sum(selected) < minimum:
            order = np.argsort(np.asarray(skewness, dtype=float))[::-1][:minimum]
            selected = [False] * num_heads
            for index in order:
                selected[int(index)] = True
        return torch.tensor(
            [index for index, keep in enumerate(selected) if keep],
            device=stacked.device,
            dtype=torch.long,
        )

    def _pool_heads(
        self, attention: torch.Tensor, selected: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        if selected is not None:
            attention = attention.index_select(0, selected)
        if attention.shape[0] == 0:
            raise ValueError("No attention heads were selected for ICR.")
        if self._active_config.pooling == "mean":
            return attention.mean(dim=0)
        if self._active_config.pooling == "max":
            return attention.max(dim=0).values
        return attention.min(dim=0).values

    def _top_count(self, available: int) -> int:
        if available < 1:
            raise ValueError(
                f"No keys are available in ICR attention scope "
                f"`{self._active_config.attention_scope}`."
            )
        if self._active_config.top_p is not None:
            return max(1, min(available, int(float(self._active_config.top_p) * available)))
        if self._active_config.top_k is None:
            return available
        return max(1, min(available, int(self._active_config.top_k)))

    def compute_icr(
        self,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = None,
        pooling: str = "mean",
        attention_uniform: bool = False,
        hidden_uniform: bool = False,
        use_induction_head: bool = False,
        *,
        attention_scope: str = "all",
    ) -> Tuple[List[List[float]], float]:
        """Return ``[layer][response_token]`` ICR scores and selected-key ratio."""
        config = ICRScoreConfig(
            top_k=top_k,
            top_p=top_p,
            pooling=pooling,
            attention_scope=attention_scope,
            attention_uniform=attention_uniform,
            hidden_uniform=hidden_uniform,
            use_induction_head=use_induction_head,
            skew_threshold=self.skew_threshold,
            entropy_threshold=self.entropy_threshold,
        )
        config.validate()
        bundle = self._compute_direction_bundle(config, save_vectors=False)
        return (
            bundle[ICR_MATRIX_NAME].float().tolist(),
            float(bundle["icr_selected_key_ratio"]),
        )

    def _compute_direction_bundle(
        self, config: ICRScoreConfig, *, save_vectors: bool
    ) -> Dict[str, Any]:
        """Compute ICR plus both paper directions across answer tokens/layers."""
        self._active_config = config
        shape = (self.num_layers, self.num_tokens)
        matrices = {
            ICR_MATRIX_NAME: torch.empty(shape, dtype=torch.float32),
            "icr_hs_only_scores": torch.empty(shape, dtype=torch.float32),
            "icr_attention_entropy": torch.empty(shape, dtype=torch.float32),
            "icr_projection_entropy": torch.empty(shape, dtype=torch.float32),
            "icr_direction_cosine": torch.empty(shape, dtype=torch.float32),
            "icr_attention_topk_mass": torch.empty(shape, dtype=torch.float32),
        }
        selected_ratios: List[float] = []
        direction_rows: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []

        with torch.no_grad():
            for layer in range(self.num_layers):
                selected_heads = self._selected_heads(layer)
                layer_rows = []
                for token in range(self.num_tokens):
                    heads = self._attention_heads(layer, token)
                    attention = self._pool_heads(heads, selected_heads)
                    key_end = int(attention.shape[-1])
                    key_start, scoped_end = self._key_bounds(key_end)
                    if scoped_end <= key_start:
                        raise ValueError(
                            "The selected ICR attention scope is empty for response token "
                            f"{token}."
                        )
                    scoped_attention = attention[key_start:scoped_end]
                    count = self._top_count(int(scoped_attention.numel()))
                    top_attention, local_indices = torch.topk(scoped_attention, k=count)
                    key_indices = local_indices + key_start
                    attention_topk_mass = top_attention.sum() / (
                        scoped_attention.sum().clamp_min(1e-12)
                    )

                    current, previous, context = self._hidden_values(layer, token, key_end)
                    # ``device_map=auto`` may place consecutive model layers on
                    # different GPUs. Move only the values used by this layer.
                    current = current.to(attention.device)
                    previous = previous.to(attention.device)
                    context = context.to(attention.device)
                    selected_context = context.index_select(0, key_indices)
                    update = current - previous
                    projections = torch.sum(update * selected_context, dim=-1) / (
                        torch.linalg.vector_norm(selected_context, dim=-1) + 1e-8
                    )
                    if config.attention_uniform:
                        top_attention = torch.ones_like(top_attention) / count
                    if config.hidden_uniform:
                        projections = torch.ones_like(projections) / count

                    projection_prob, attention_prob = _direction_probabilities(
                        projections, top_attention
                    )
                    values = {
                        ICR_MATRIX_NAME: js_divergence(projections, top_attention),
                        "icr_hs_only_scores": js_divergence(
                            projections, torch.ones_like(projections)
                        ),
                        "icr_attention_entropy": _normalized_entropy(attention_prob),
                        "icr_projection_entropy": _normalized_entropy(projection_prob),
                        "icr_direction_cosine": F.cosine_similarity(
                            projection_prob.unsqueeze(0), attention_prob.unsqueeze(0)
                        ).squeeze(0),
                        "icr_attention_topk_mass": attention_topk_mass,
                    }
                    for name, value in values.items():
                        matrices[name][layer, token] = value.detach().float().cpu()
                    selected_ratios.append(count / int(scoped_attention.numel()))
                    if save_vectors:
                        layer_rows.append(
                            (
                                key_indices.detach().long().cpu(),
                                attention_prob.detach().float().cpu(),
                                projection_prob.detach().float().cpu(),
                            )
                        )
                if save_vectors:
                    direction_rows.append(layer_rows)

        for name, matrix in matrices.items():
            if not bool(torch.isfinite(matrix).all()):
                raise FloatingPointError(f"{name} computation produced non-finite values.")

        result: Dict[str, Any] = {
            **matrices,
            "icr_selected_key_ratio": float(np.mean(selected_ratios)),
        }
        if save_vectors:
            width = max(row[0].numel() for layer in direction_rows for row in layer)
            indices = torch.full((*shape, width), -1, dtype=torch.long)
            mask = torch.zeros((*shape, width), dtype=torch.bool)
            attention_direction = torch.zeros((*shape, width), dtype=torch.float32)
            projection_direction = torch.zeros((*shape, width), dtype=torch.float32)
            for layer, rows in enumerate(direction_rows):
                for token, (row_indices, attn, proj) in enumerate(rows):
                    count = row_indices.numel()
                    indices[layer, token, :count] = row_indices
                    mask[layer, token, :count] = True
                    attention_direction[layer, token, :count] = attn
                    projection_direction[layer, token, :count] = proj
            result.update(
                {
                    "icr_direction_key_indices": indices,
                    "icr_direction_mask": mask,
                    "icr_attention_direction": attention_direction.to(torch.float16),
                    "icr_projection_direction": projection_direction.to(torch.float16),
                }
            )
        return result

    def compute_features(self, config: Optional[ICRScoreConfig] = None) -> Dict[str, Any]:
        """Compute and package the matrix plus the paper's 1 x L probe input."""
        config = config or ICRScoreConfig(
            skew_threshold=self.skew_threshold,
            entropy_threshold=self.entropy_threshold,
        )
        config.validate()
        bundle = self._compute_direction_bundle(
            config, save_vectors=bool(config.save_direction_vectors)
        )
        matrix = bundle[ICR_MATRIX_NAME]
        result = {
            **bundle,
            ICR_MATRIX_NAME: matrix.to(torch.float16),
            ICR_FEATURE_NAME: matrix.mean(dim=1),
            "icr_token_mean": matrix.mean(dim=0),
            "icr_num_layers": int(matrix.shape[0]),
            "icr_num_tokens": int(matrix.shape[1]),
            "icr_model_layer_ids": torch.tensor(self.model_layer_ids, dtype=torch.long),
            "icr_config": asdict(config),
            "icr_source_revision": ICR_SOURCE_REVISION,
        }
        for name in (
            "icr_hs_only_scores",
            "icr_attention_entropy",
            "icr_projection_entropy",
            "icr_direction_cosine",
            "icr_attention_topk_mass",
        ):
            result[f"{name}_layer_mean"] = result[name].mean(dim=1)
        # These complementary summaries expose the paper's qualitative
        # interpretation; the unmodified JSD remains the primary ICR metric.
        normalized_icr = (matrix / math.log(2.0)).clamp(0.0, 1.0)
        result["icr_attention_alignment"] = 1.0 - normalized_icr
        result["icr_residual_deviation"] = normalized_icr
        result["icr_attention_alignment_layer_mean"] = result[
            "icr_attention_alignment"
        ].mean(dim=1)
        result["icr_residual_deviation_layer_mean"] = result[
            "icr_residual_deviation"
        ].mean(dim=1)
        return result


class ICRProbe(nn.Module):
    """Published ``L -> 128 -> 64 -> 32 -> 1`` ICR Probe MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("ICRProbe input_dim must be positive.")
        if tuple(int(dim) for dim in hidden_dims) != (128, 64, 32):
            logging.warning(
                "Using non-paper ICR Probe hidden dimensions: %s", tuple(hidden_dims)
            )
        dimensions = [int(input_dim), *(int(dim) for dim in hidden_dims)]
        blocks: List[nn.Module] = []
        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            blocks.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(dimensions[-1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight, a=0.01, nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("ICRProbe input must have shape [batch, layers].")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("ICRProbe input contains NaN or infinity.")
        return torch.sigmoid(self.output(self.hidden(values)))


@dataclass(frozen=True)
class ICRProbeTrainingConfig:
    """Training hyperparameters from Appendix B.4."""

    hidden_dims: Tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.3
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    lr_factor: float = 0.5
    lr_patience: int = 5
    random_seed: int = 10
    device: str = "auto"

    def validate(self) -> None:
        if self.batch_size < 2:
            raise ValueError("ICR Probe batch_size must be at least 2 for BatchNorm.")
        if self.num_epochs < 1:
            raise ValueError("ICR Probe num_epochs must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid ICR Probe optimizer parameters.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("ICR Probe validation_fraction must be in (0, 1).")
        if not 0.0 < self.lr_factor < 1.0 or self.lr_patience < 0:
            raise ValueError("Invalid ICR Probe scheduler parameters.")


def icr_feature_matrix(
    probe_features: Iterable[Mapping[str, Any]],
    feature_name: str = ICR_FEATURE_NAME,
) -> np.ndarray:
    """Convert saved layer-wise ICR vectors into a dense sample matrix."""
    rows = []
    for sample_index, features in enumerate(probe_features):
        value = features.get(feature_name)
        if value is None:
            raise ValueError(
                f"Probe feature `{feature_name}` is missing from sample {sample_index}. "
                "Generate data with --collect_icr_probe."
            )
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if row.size == 0 or not np.isfinite(row).all():
            raise ValueError(f"Invalid ICR feature vector at sample {sample_index}.")
        rows.append(row)
    if not rows:
        raise ValueError("No ICR probe features were supplied.")
    dimensions = {row.shape[0] for row in rows}
    if len(dimensions) != 1:
        raise ValueError(f"ICR layer dimensions are inconsistent: {sorted(dimensions)}")
    return np.stack(rows, axis=0)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested ICR Probe device `{device}`, but CUDA is unavailable.")
    return resolved


def _split_training_indices(
    labels: np.ndarray,
    validation_fraction: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    class_counts = np.bincount(labels, minlength=2)
    if labels.shape[0] < 6 or int(class_counts.min()) < 2:
        raise ValueError(
            "ICR Probe requires at least six training samples and two samples "
            "from each class for a train-only validation split."
        )
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=labels,
    )
    if train_indices.shape[0] < 2:
        raise ValueError("ICR Probe training split is too small for BatchNorm.")
    return np.asarray(train_indices), np.asarray(validation_indices)


def _training_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    random_seed: int,
) -> DataLoader:
    effective_batch = min(int(batch_size), len(features))
    drop_last = len(features) % effective_batch == 1
    generator = torch.Generator().manual_seed(int(random_seed))
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=effective_batch,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )


def fit_icr_probe_and_score(
    train_features: Sequence[Mapping[str, Any]],
    train_is_false: Sequence[float],
    eval_features: Sequence[Mapping[str, Any]],
    *,
    feature_name: str = ICR_FEATURE_NAME,
    config: Optional[ICRProbeTrainingConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any], ICRProbe]:
    """Fit the published MLP on train-only labels and return error probabilities."""
    config = config or ICRProbeTrainingConfig()
    config.validate()
    x_train = icr_feature_matrix(train_features, feature_name)
    x_eval = icr_feature_matrix(eval_features, feature_name)
    if x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("Train/eval ICR vectors have different layer counts.")
    y_train = np.asarray(train_is_false, dtype=np.float32).reshape(-1)
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("ICR training feature/label counts do not match.")
    if not np.isin(y_train, (0.0, 1.0)).all() or np.unique(y_train).size != 2:
        raise ValueError("ICR Probe requires binary labels containing both classes.")

    train_indices, validation_indices = _split_training_indices(
        y_train.astype(np.int64), config.validation_fraction, config.random_seed
    )
    torch.manual_seed(int(config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.random_seed))
    device = _resolve_device(config.device)
    model = ICRProbe(
        input_dim=x_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.lr_factor),
        patience=int(config.lr_patience),
    )
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train).unsqueeze(1)
    loader = _training_loader(
        x_tensor[train_indices],
        y_tensor[train_indices],
        config.batch_size,
        config.random_seed,
    )
    x_validation = x_tensor[validation_indices].to(device)
    y_validation = y_tensor[validation_indices].to(device)

    best_loss = math.inf
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    for epoch in range(int(config.num_epochs)):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        if batches == 0:
            raise RuntimeError("ICR Probe training loader produced no batches.")

        model.eval()
        with torch.no_grad():
            validation_predictions = model(x_validation)
            validation_loss = float(
                criterion(validation_predictions, y_validation).item()
            )
        scheduler.step(validation_loss)
        train_loss = total_loss / batches
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("ICR Probe failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = (
            model(torch.from_numpy(x_eval).to(device))
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        validation_scores = (
            model(x_validation).squeeze(1).detach().cpu().numpy()
        )
    if not np.isfinite(scores).all():
        raise FloatingPointError("ICR Probe produced non-finite probabilities.")

    validation_labels = y_train[validation_indices]
    validation_auroc = (
        float(roc_auc_score(validation_labels, validation_scores))
        if np.unique(validation_labels).size == 2
        else float("nan")
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metadata: Dict[str, Any] = {
        "feature_name": feature_name,
        "input_dim": int(x_train.shape[1]),
        "architecture": [int(x_train.shape[1]), *config.hidden_dims, 1],
        "parameter_count": parameter_count,
        "label_semantics": "0=faithful, 1=hallucinated/error",
        "score_semantics": "hallucination/error probability",
        "train_samples": int(x_train.shape[0]),
        "fit_samples": int(train_indices.shape[0]),
        "holdout_samples": int(validation_indices.shape[0]),
        "eval_samples": int(x_eval.shape[0]),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "holdout_auroc": validation_auroc,
        "training_config": asdict(config),
        "training_history": history,
        "source_revision": ICR_SOURCE_REVISION,
    }
    logging.info(
        "ICR Probe trained on %d samples with %d layers; holdout AUROC %.4f.",
        x_train.shape[0],
        x_train.shape[1],
        validation_auroc,
    )
    return scores.astype(np.float64), metadata, model.cpu()


def save_icr_probe_checkpoint(
    path: str,
    model: ICRProbe,
    metadata: Mapping[str, Any],
) -> str:
    """Save a self-describing, CPU-portable ICR Probe checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": int(metadata["input_dim"]),
        "hidden_dims": tuple(metadata["architecture"][1:-1]),
        "dropout": float(metadata["training_config"]["dropout"]),
        "metadata": dict(metadata),
        "format_version": 1,
    }
    torch.save(payload, destination)
    return str(destination)


def load_icr_probe_checkpoint(
    path: str,
    *,
    device: str = "cpu",
) -> Tuple[ICRProbe, Dict[str, Any]]:
    """Load a checkpoint created by :func:`save_icr_probe_checkpoint`."""
    resolved = _resolve_device(device)
    payload = torch.load(path, map_location=resolved, weights_only=False)
    model = ICRProbe(
        input_dim=int(payload["input_dim"]),
        hidden_dims=tuple(payload.get("hidden_dims", (128, 64, 32))),
        dropout=float(payload.get("dropout", 0.3)),
    ).to(resolved)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("metadata", {}))
