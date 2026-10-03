"""Training-free entropy features from cached multimodal decoder internals.

The functions in this module intentionally separate Shannon entropy of an
attention probability map from *energy entropy*.  The latter normalises
non-negative squared activations or transition magnitudes and describes how
widely representation energy is distributed; it is not predictive entropy.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1e-12


def head_output_energy_entropy(
    attention_head_outputs: Any, *, eps: float = EPS
) -> np.ndarray:
    """Compute normalized energy entropy across attention-head outputs.

    ``attention_head_outputs`` must end in ``[heads, head_dim]`` and must be
    captured before concatenation/output projection.  For every leading
    index, this implements exactly

    ``E_h = ||z_h||_2^2``, ``p_h = (E_h + eps) / sum_j(E_j + eps)``, and
    ``H = -sum_h p_h log(p_h) / log(num_heads)``.

    A one-head input is assigned zero because normalisation by ``log(1)`` is
    undefined and there is no head-wise dispersion to measure.
    """
    values = np.asarray(attention_head_outputs, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(
            "attention_head_outputs must end in [heads, head_dim]."
        )
    if not np.isfinite(values).all():
        raise ValueError("attention_head_outputs must be finite.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive.")
    num_heads = values.shape[-2]
    if num_heads < 1 or values.shape[-1] < 1:
        raise ValueError("heads and head_dim must both be non-empty.")
    if num_heads == 1:
        return np.zeros(values.shape[:-2], dtype=np.float64)

    energy = np.sum(np.square(values), axis=-1)
    smoothed_energy = energy + float(eps)
    probabilities = smoothed_energy / np.sum(
        smoothed_energy, axis=-1, keepdims=True
    )
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    # Roundoff can otherwise produce values a few ulps outside [0, 1].
    return np.clip(entropy / np.log(num_heads), 0.0, 1.0)


def head_output_energy_entropy_torch(
    attention_head_outputs: Any, *, eps: float = EPS
) -> Any:
    """Torch-native head-output energy entropy for online hooks.

    Activations are promoted to float32 before squaring so ``eps=1e-12`` does
    not underflow when the model itself runs in float16.
    """
    import torch

    values = torch.as_tensor(attention_head_outputs).float()
    if values.ndim < 2:
        raise ValueError(
            "attention_head_outputs must end in [heads, head_dim]."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("attention_head_outputs must be finite.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive.")
    num_heads = values.shape[-2]
    if num_heads < 1 or values.shape[-1] < 1:
        raise ValueError("heads and head_dim must both be non-empty.")
    if num_heads == 1:
        return torch.zeros(
            values.shape[:-2], dtype=values.dtype, device=values.device
        )

    energy = torch.sum(torch.square(values), dim=-1) + float(eps)
    probabilities = energy / torch.sum(energy, dim=-1, keepdim=True)
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    return torch.clamp(entropy / np.log(num_heads), min=0.0, max=1.0)


def normalized_entropy(
    weights: Any, *, axis: int = -1, eps: float = EPS
) -> np.ndarray:
    """Return Shannon entropy divided by ``log(size(axis))``.

    Inputs are treated as non-negative weights and normalised along ``axis``.
    An all-zero row has entropy zero because it contains no distributed mass.
    """
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("weights must have at least one dimension.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    size = values.shape[axis]
    if size < 2:
        return np.zeros(np.sum(values, axis=axis).shape, dtype=np.float64)
    total = np.sum(values, axis=axis, keepdims=True)
    probabilities = np.divide(
        values,
        np.maximum(total, eps),
        out=np.zeros_like(values),
        where=total > eps,
    )
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, eps, None)), axis=axis
    )
    return entropy / np.log(size)


def binary_entropy(probabilities: Any, *, eps: float = EPS) -> np.ndarray:
    """Normalised Bernoulli entropy, with values in ``[0, 1]``."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite.")
    values = np.clip(values, 0.0, 1.0)
    complement = 1.0 - values
    entropy = -(
        values * np.log(np.clip(values, eps, None))
        + complement * np.log(np.clip(complement, eps, None))
    )
    return entropy / np.log(2.0)


def trajectory_features(values: Any, *, prefix: str) -> dict[str, float]:
    """Summarise a scalar layer trajectory without learned parameters."""
    path = np.asarray(values, dtype=np.float64)
    if path.ndim != 1 or path.size < 2 or not np.isfinite(path).all():
        raise ValueError("A finite one-dimensional trajectory is required.")
    transitions = np.abs(np.diff(path))
    quarter = max(path.size // 4, 1)
    return {
        f"{prefix}_mean": float(np.mean(path)),
        f"{prefix}_variance": float(np.var(path)),
        f"{prefix}_total_variation": float(np.mean(transitions)),
        f"{prefix}_transition_entropy": float(normalized_entropy(transitions)),
        f"{prefix}_endpoint_abs_delta": float(abs(path[-1] - path[0])),
        f"{prefix}_late_minus_early": float(
            np.mean(path[-quarter:]) - np.mean(path[:quarter])
        ),
    }


def attention_entropy_paths(
    attention: Any,
    *,
    num_visual_tokens: int,
    visual_suffix_tokens: int = 6,
) -> dict[str, np.ndarray]:
    """Compute per-layer entropy paths from ``[layer, head, prompt]`` attention."""
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1:
        raise ValueError("attention must have shape [layers, heads, prompt tokens].")
    if not np.isfinite(values).all():
        raise ValueError("attention must be finite.")
    values = np.clip(values, 0.0, None)
    visual_end = values.shape[-1] - int(visual_suffix_tokens)
    visual_start = visual_end - int(num_visual_tokens)
    if visual_start < 0 or visual_end > values.shape[-1] or visual_start >= visual_end:
        raise ValueError("Inferred visual-token span is invalid.")
    visual = values[..., visual_start:visual_end]

    head_spatial = normalized_entropy(visual, axis=-1)
    visual_probabilities = visual / np.maximum(
        np.sum(visual, axis=-1, keepdims=True), EPS
    )
    mixture = np.mean(visual_probabilities, axis=1)
    spatial_head_mean = np.mean(head_spatial, axis=1)
    spatial_mixture = normalized_entropy(mixture, axis=-1)

    prompt_mass = np.sum(values, axis=-1)
    visual_mass = np.sum(visual, axis=-1)
    visual_share = visual_mass / np.maximum(prompt_mass, EPS)
    return {
        "head_spatial_entropy": head_spatial,
        "spatial_head_mean_entropy": spatial_head_mean,
        "spatial_mixture_entropy": spatial_mixture,
        "spatial_js_disagreement": np.maximum(
            spatial_mixture - spatial_head_mean, 0.0
        ),
        "modality_binary_entropy": np.mean(binary_entropy(visual_share), axis=1),
        "visual_mass_head_entropy": normalized_entropy(visual_mass, axis=1),
        "visual_attention_mass": visual_mass,
    }


def energy_entropy_paths(
    hidden_states: Any,
    attention_head_outputs: Any | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute layer/channel/head energy entropy and CoE-like transition entropy."""
    hidden = np.asarray(hidden_states, dtype=np.float64)
    if hidden.ndim != 2 or min(hidden.shape) < 2 or not np.isfinite(hidden).all():
        raise ValueError("hidden_states must be finite [layers, hidden_dim].")

    hidden_energy = np.square(hidden)
    differences = np.diff(hidden, axis=0)
    transition_energy = np.sum(np.square(differences), axis=-1)
    left, right = hidden[:-1], hidden[1:]
    cosine = np.sum(left * right, axis=-1) / np.maximum(
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), EPS
    )
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    output: dict[str, np.ndarray | float] = {
        "hidden_channel_energy_entropy": normalized_entropy(hidden_energy, axis=-1),
        "hidden_layer_energy_entropy": float(
            normalized_entropy(np.sum(hidden_energy, axis=-1))
        ),
        "hidden_transition_energy_entropy": float(
            normalized_entropy(transition_energy)
        ),
        "hidden_transition_angle_entropy": float(normalized_entropy(angles)),
        "hidden_transition_energy_cv": float(
            np.std(transition_energy) / max(np.mean(transition_energy), EPS)
        ),
    }
    if attention_head_outputs is not None:
        head_outputs = np.asarray(attention_head_outputs, dtype=np.float64)
        if head_outputs.ndim != 3 or head_outputs.shape[:2] != (
            hidden.shape[0],
            head_outputs.shape[1],
        ):
            raise ValueError(
                "attention_head_outputs must have shape [layers, heads, head_dim]."
            )
        head_entropy = head_output_energy_entropy(head_outputs)
        output["attention_output_head_energy_entropy"] = head_entropy
    return output


def hybrid_decoder_energy_entropy(
    hidden_states: Any,
    full_attention_head_outputs: Any,
    *,
    full_attention_layer_ids: Sequence[int],
) -> dict[str, np.ndarray | float]:
    """Energy entropy for a hybrid decoder such as Qwen3.5.

    ``hidden_states`` contains every physical decoder layer, whereas ordinary
    multi-head outputs exist only at ``full_attention_layer_ids``.  Both inputs
    must refer to the same query position and forward pass.
    """
    hidden = np.asarray(hidden_states, dtype=np.float64)
    heads = np.asarray(full_attention_head_outputs, dtype=np.float64)
    layer_ids = np.asarray(list(full_attention_layer_ids), dtype=np.int64)
    if hidden.ndim != 2 or min(hidden.shape) < 2 or not np.isfinite(hidden).all():
        raise ValueError("hidden_states must be finite [physical_layers, hidden_dim].")
    if heads.ndim != 3 or min(heads.shape) < 1 or not np.isfinite(heads).all():
        raise ValueError(
            "full_attention_head_outputs must be finite [full_layers, heads, head_dim]."
        )
    if layer_ids.shape != (heads.shape[0],):
        raise ValueError("One physical layer id is required per full-attention layer.")
    if len(set(map(int, layer_ids))) != len(layer_ids):
        raise ValueError("full_attention_layer_ids must be unique.")
    if np.any(layer_ids < 0) or np.any(layer_ids >= hidden.shape[0]):
        raise ValueError("A full-attention layer id is outside the hidden-state path.")

    hidden_entropy = normalized_entropy(np.square(hidden), axis=-1)
    head_entropy = head_output_energy_entropy(heads)
    differences = np.diff(hidden, axis=0)
    transition_energy = np.sum(np.square(differences), axis=-1)
    transition_cv = float(
        np.std(transition_energy) / max(np.mean(transition_energy), EPS)
    )
    return {
        "hidden_channel_energy_entropy": hidden_entropy,
        "head_output_energy_entropy": head_entropy,
        "same_layer_dual_energy_entropy": (
            hidden_entropy[layer_ids] + head_entropy
        )
        / 2.0,
        "hidden_transition_energy": transition_energy,
        "hidden_transition_energy_cv": transition_cv,
        "hidden_transition_energy_cv_compressed": transition_cv
        / (1.0 + transition_cv),
    }


def pool_layer_bands(
    paths: Mapping[str, Any], bands: Mapping[str, Sequence[int]]
) -> dict[str, float]:
    """Average one-dimensional layer paths over named layer sets."""
    output: dict[str, float] = {}
    for metric, raw_path in paths.items():
        path = np.asarray(raw_path, dtype=np.float64)
        if path.ndim != 1:
            continue
        for band_name, indices in bands.items():
            chosen = np.asarray(list(indices), dtype=np.int64)
            if chosen.size and np.all((0 <= chosen) & (chosen < path.size)):
                output[f"{metric}_{band_name}"] = float(np.mean(path[chosen]))
    return output
