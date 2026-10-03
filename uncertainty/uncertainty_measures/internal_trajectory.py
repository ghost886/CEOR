"""Internal trajectory metrics adapted from SIVR and Chain-of-Embedding.

The public APIs in this module deliberately operate on compact arrays with
shape ``[tokens, decoder_layers, hidden_dim]``.  They do not retain a model or
generation cache, which lets experiment drivers compute the metrics and free
the large hidden-state tensors immediately.

References
----------
* Srey et al., *Learning Uncertainty from Sequential Internal Dispersion in
  Large Language Models* (SIVR, ACL 2026).
* Wang et al., *Latent Space Chain-of-Embedding Enables Output-free LLM
  Self-Evaluation* (CoE, ICLR 2025).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


EPS = 1e-8


def _trajectory(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(
            "Hidden trajectory must have shape [tokens, layers, hidden_dim] "
            f"or [layers, hidden_dim], got {array.shape}."
        )
    if min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Hidden trajectory must be non-empty and finite.")
    return array


def extract_generation_hidden_trajectory(
    hidden_states: Sequence[Any],
    *,
    generated_token_count: int,
    num_decoder_layers: int,
    batch_index: int = 0,
) -> np.ndarray:
    """Convert a Hugging Face generation cache to ``[token, layer, dim]``.

    The state at the final query position of generation step ``t`` is the
    representation used to predict generated token ``t``.  HF commonly
    returns an embedding state plus one state per decoder block; selecting the
    last ``num_decoder_layers`` entries maps array index ``l`` to decoder layer
    ``l`` and matches the Head-probe layer convention used in this repository.
    """
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ValueError("Generation output contains no hidden-state steps.")
    steps = min(int(generated_token_count), len(hidden_states))
    if steps < 1:
        raise ValueError("At least one generated-token hidden state is required.")
    rows: list[np.ndarray] = []
    for step_index in range(steps):
        step = hidden_states[step_index]
        if not isinstance(step, (tuple, list)):
            raise ValueError("Each generation hidden-state step must contain layers.")
        if len(step) < int(num_decoder_layers):
            raise ValueError(
                f"Hidden step has {len(step)} states, fewer than the requested "
                f"{num_decoder_layers} decoder layers."
            )
        decoder_states = step[-int(num_decoder_layers) :]
        layer_rows = []
        for state in decoder_states:
            tensor = torch.as_tensor(state)
            if tensor.ndim != 3:
                raise ValueError(
                    "Generation hidden state must have shape [batch, sequence, dim]."
                )
            layer_rows.append(
                tensor[int(batch_index), -1].detach().float().cpu().numpy()
            )
        rows.append(np.stack(layer_rows, axis=0))
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def extract_generation_logits(
    scores: Sequence[Any], *, generated_token_count: int, batch_index: int = 0
) -> np.ndarray:
    """Return generation logits with shape ``[token, vocabulary]``."""
    if not isinstance(scores, (tuple, list)) or not scores:
        raise ValueError("Generation output contains no token scores.")
    steps = min(int(generated_token_count), len(scores))
    rows = []
    for score in scores[:steps]:
        tensor = torch.as_tensor(score)
        if tensor.ndim != 2:
            raise ValueError("Generation score must have shape [batch, vocabulary].")
        rows.append(tensor[int(batch_index)].detach().float().cpu().numpy())
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _select_layers(
    trajectory: np.ndarray, layer_indices: Sequence[int] | None
) -> tuple[np.ndarray, list[int]]:
    layers = trajectory.shape[1]
    indices = (
        list(range(layers))
        if layer_indices is None
        else list(dict.fromkeys(int(index) for index in layer_indices))
    )
    if len(indices) < 2:
        raise ValueError("Trajectory metrics require at least two distinct layers.")
    invalid = [index for index in indices if not 0 <= index < layers]
    if invalid:
        raise ValueError(f"Layer indices outside [0, {layers - 1}]: {invalid}.")
    return trajectory[:, indices, :], indices


def coe_metrics(
    hidden_trajectory: Any,
    *,
    layer_indices: Sequence[int] | None = None,
    token_pool: str = "mean",
    eps: float = EPS,
) -> dict[str, Any]:
    """Compute all official Chain-of-Embedding trajectory features.

    ``magnitude_path`` and ``angle_path`` reproduce the normalized adjacent
    layer quantities.  Their means/variances plus CoE-R and CoE-C are retained.
    Raw (unnormalized) paths are also emitted so that local key-layer
    contribution is not distorted by different subset denominators.
    """
    trajectory, indices = _select_layers(_trajectory(hidden_trajectory), layer_indices)
    if token_pool == "mean":
        states = trajectory.mean(axis=0)
    elif token_pool == "last":
        states = trajectory[-1]
    elif token_pool == "first":
        states = trajectory[0]
    else:
        raise ValueError("token_pool must be one of: mean, first, last.")

    differences = states[1:] - states[:-1]
    raw_magnitude = np.linalg.norm(differences, axis=-1)
    endpoint_magnitude = float(np.linalg.norm(states[-1] - states[0]))
    magnitude = raw_magnitude / max(endpoint_magnitude, float(eps))

    left, right = states[:-1], states[1:]
    edge_cosine = np.sum(left * right, axis=-1) / np.maximum(
        np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), eps
    )
    raw_angle = np.arccos(np.clip(edge_cosine, -1.0, 1.0))
    endpoint_cosine = float(
        np.dot(states[0], states[-1])
        / max(np.linalg.norm(states[0]) * np.linalg.norm(states[-1]), eps)
    )
    endpoint_angle = float(np.arccos(np.clip(endpoint_cosine, -1.0, 1.0)))
    angle = raw_angle / max(endpoint_angle, float(eps))

    magnitude_mean = float(np.mean(magnitude))
    angle_mean = float(np.mean(angle))
    x_mean = float(np.mean(magnitude * np.cos(angle)))
    y_mean = float(np.mean(magnitude * np.sin(angle)))
    paper_x_mean = float(np.mean(magnitude * np.cos(raw_angle)))
    paper_y_mean = float(np.mean(magnitude * np.sin(raw_angle)))
    return {
        "layer_indices": indices,
        "token_pool": token_pool,
        "raw_magnitude_path": raw_magnitude.astype(float).tolist(),
        "raw_angle_path": raw_angle.astype(float).tolist(),
        "magnitude_path": magnitude.astype(float).tolist(),
        "angle_path": angle.astype(float).tolist(),
        "endpoint_magnitude": endpoint_magnitude,
        "endpoint_angle": endpoint_angle,
        "magnitude_mean": magnitude_mean,
        "magnitude_variance": float(np.var(magnitude)),
        "angle_mean": angle_mean,
        "angle_variance": float(np.var(angle)),
        "coe_r": magnitude_mean - angle_mean,
        # The released repository applies cos/sin to the endpoint-normalised
        # angle, whereas Eq. (7) of the paper applies them to the raw angle.
        "coe_c": float(np.sqrt(x_mean * x_mean + y_mean * y_mean)),
        "coe_c_repository": float(np.sqrt(x_mean * x_mean + y_mean * y_mean)),
        "coe_c_paper": float(
            np.sqrt(paper_x_mean * paper_x_mean + paper_y_mean * paper_y_mean)
        ),
    }


def sivr_internal_features(
    hidden_trajectory: Any,
    *,
    layer_indices: Sequence[int] | None = None,
    covariance_regularization: float = 1e-3,
) -> dict[str, Any]:
    """Compute SIVR circular variance and covariance log-determinant per token."""
    trajectory, indices = _select_layers(_trajectory(hidden_trajectory), layer_indices)
    norms = np.linalg.norm(trajectory, axis=-1, keepdims=True)
    directions = trajectory / np.maximum(norms, EPS)
    mean_resultant_length = np.linalg.norm(directions.mean(axis=1), axis=-1)
    circular_variance = 1.0 - np.clip(mean_resultant_length, 0.0, 1.0)

    centred = trajectory - trajectory.mean(axis=-1, keepdims=True)
    covariance = centred @ np.swapaxes(centred, 1, 2)
    covariance /= max(trajectory.shape[-1] - 1, 1)
    identity = np.eye(trajectory.shape[1], dtype=np.float64)[None, :, :]
    eigenvalues = np.linalg.eigvalsh(
        covariance + float(covariance_regularization) * identity
    )
    covariance_logdet_mean = np.log(np.clip(eigenvalues, EPS, None)).mean(axis=-1)
    covariance_logdet = np.log(np.clip(eigenvalues, EPS, None)).sum(axis=-1)
    return {
        "layer_indices": indices,
        "circular_variance_per_token": circular_variance.astype(float).tolist(),
        "covariance_logdet_mean_per_token": (
            covariance_logdet_mean.astype(float).tolist()
        ),
        "covariance_logdet_per_token": covariance_logdet.astype(float).tolist(),
        "circular_variance_mean": float(np.mean(circular_variance)),
        "circular_variance_std": float(np.std(circular_variance)),
        "circular_variance_max": float(np.max(circular_variance)),
        "covariance_logdet_mean": float(np.mean(covariance_logdet_mean)),
        "covariance_logdet_std": float(np.std(covariance_logdet_mean)),
        "covariance_logdet_max": float(np.max(covariance_logdet_mean)),
        # Eq. (1) is a sum.  The official repository divides it by the number
        # of layer-space eigenvalues; retain both definitions explicitly.
        "covariance_logdet_paper_mean": float(np.mean(covariance_logdet)),
        "covariance_logdet_paper_std": float(np.std(covariance_logdet)),
        "covariance_logdet_paper_max": float(np.max(covariance_logdet)),
    }


def output_token_features(
    logits: Any, *, temperature: float = 0.7, eps: float = EPS
) -> dict[str, Any]:
    """Compute output-space baselines recorded by both reference repositories."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("Logits must have shape [tokens, vocabulary].")
    if float(temperature) <= 0:
        raise ValueError("Output-score temperature must be positive.")
    # HF generation processors legitimately set forbidden tokens to -inf.
    # Such entries carry zero probability and should not invalidate a row.
    # NaN, +inf, or an entirely masked row still indicate a genuine failure.
    if (
        np.isnan(values).any()
        or np.isposinf(values).any()
        or not np.isfinite(values).any(axis=-1).all()
    ):
        raise ValueError(
            "Each logits row must contain a finite value and no NaN/+inf; "
            "-inf token masks are supported."
        )

    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    max_probability = probabilities.max(axis=-1)
    entropy = -(probabilities * np.log(np.clip(probabilities, eps, None))).sum(axis=-1)
    negative_log_max_probability = -np.log(np.clip(max_probability, eps, None))

    scaled_values = values / float(temperature)
    scaled_max = scaled_values.max(axis=-1, keepdims=True)
    scaled = scaled_values - scaled_max
    scaled_probabilities = np.exp(scaled)
    scaled_probabilities /= scaled_probabilities.sum(axis=-1, keepdims=True)
    temperature_max_probability = scaled_probabilities.max(axis=-1)
    energy = -float(temperature) * (
        np.log(np.exp(scaled).sum(axis=-1)) + scaled_max.squeeze(-1)
    )
    return {
        "max_probability_per_token": max_probability.astype(float).tolist(),
        "entropy_per_token": entropy.astype(float).tolist(),
        "negative_log_max_probability_per_token": (
            negative_log_max_probability.astype(float).tolist()
        ),
        "temperature_max_probability_per_token": (
            temperature_max_probability.astype(float).tolist()
        ),
        "energy_per_token": energy.astype(float).tolist(),
        "max_probability_mean": float(np.mean(max_probability)),
        "entropy_mean": float(np.mean(entropy)),
        "negative_log_max_probability_mean": float(
            np.mean(negative_log_max_probability)
        ),
        "temperature_max_probability_mean": float(
            np.mean(temperature_max_probability)
        ),
        "energy_mean": float(np.mean(energy)),
    }


def layer_edge_concentration(
    full_coe: Mapping[str, Any], *, key_layers: Sequence[int]
) -> dict[str, Any]:
    """Measure whether large CoE transitions terminate at the nominated layers."""
    layers = [int(value) for value in full_coe["layer_indices"]]
    raw_magnitude = np.asarray(full_coe["raw_magnitude_path"], dtype=np.float64)
    raw_angle = np.asarray(full_coe["raw_angle_path"], dtype=np.float64)
    if len(layers) != raw_magnitude.size + 1:
        raise ValueError("CoE layer/path lengths are inconsistent.")
    edge_targets = layers[1:]
    key_set = {int(value) for value in key_layers}
    selected = np.asarray([layer in key_set for layer in edge_targets], dtype=bool)
    if not selected.any():
        raise ValueError("No key layer corresponds to a full-trajectory incoming edge.")

    def summarize(values: np.ndarray) -> dict[str, float]:
        key = values[selected]
        rest = values[~selected]
        return {
            "key_mean": float(np.mean(key)),
            "rest_mean": float(np.mean(rest)) if rest.size else float("nan"),
            "key_over_rest": float(
                np.mean(key) / max(float(np.mean(rest)), EPS)
            ) if rest.size else float("nan"),
            "key_share": float(key.sum() / max(float(values.sum()), EPS)),
        }

    return {
        "edge_target_layers": edge_targets,
        "key_edge_mask": selected.astype(int).tolist(),
        "magnitude": summarize(raw_magnitude),
        "angle": summarize(raw_angle),
    }


def cross_draw_layer_dispersion(pooled_layer_states: Any) -> dict[str, Any]:
    """Compute exact layer-wise dispersion across stochastic generations.

    Parameters
    ----------
    pooled_layer_states:
        Array ``[draws, layers, hidden_dim]``.  A caller can pool generated
        tokens within each draw and discard all token-level hidden states
        before invoking this function.
    """
    values = np.asarray(pooled_layer_states, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("Pooled states must have shape [draws, layers, hidden_dim].")
    draws = values.shape[0]
    centre = values.mean(axis=0, keepdims=True)
    numerator = np.square(values - centre).sum(axis=(0, 2))
    denominator = np.square(values).sum(axis=(0, 2))
    relative_l2_variance = numerator / np.maximum(denominator, EPS)

    unit = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), EPS)
    resultant = np.linalg.norm(unit.mean(axis=0), axis=-1)
    circular_variance = 1.0 - np.clip(resultant, 0.0, 1.0)
    if draws > 1:
        sum_unit = unit.sum(axis=0)
        mean_pairwise_cosine = (
            np.square(sum_unit).sum(axis=-1) - draws
        ) / (draws * (draws - 1))
        mean_pairwise_cosine_distance = 1.0 - np.clip(
            mean_pairwise_cosine, -1.0, 1.0
        )
    else:
        mean_pairwise_cosine_distance = np.zeros(values.shape[1], dtype=np.float64)
    layer_norms = np.linalg.norm(values, axis=-1)
    norm_cv = layer_norms.std(axis=0) / np.maximum(layer_norms.mean(axis=0), EPS)
    return {
        "num_draws": int(draws),
        "relative_l2_variance_by_layer": relative_l2_variance.astype(float).tolist(),
        "circular_variance_by_layer": circular_variance.astype(float).tolist(),
        "mean_pairwise_cosine_distance_by_layer": (
            mean_pairwise_cosine_distance.astype(float).tolist()
        ),
        "norm_cv_by_layer": norm_cv.astype(float).tolist(),
    }


def trajectory_metric_bundle(
    hidden_trajectory: Any,
    logits: Any,
    *,
    layer_views: Mapping[str, Sequence[int]],
    key_layers: Sequence[int],
    output_temperature: float = 0.7,
) -> dict[str, Any]:
    """Compute paper metrics for the full path and matched layer subsets."""
    trajectory = _trajectory(hidden_trajectory)
    views: dict[str, Any] = {}
    for name, indices in layer_views.items():
        mean_coe = coe_metrics(
            trajectory, layer_indices=indices, token_pool="mean"
        )
        views[str(name)] = {
            "coe_mean_token": mean_coe,
            "coe_last_token": coe_metrics(
                trajectory, layer_indices=indices, token_pool="last"
            ),
            "sivr": sivr_internal_features(trajectory, layer_indices=indices),
        }
    if "full" not in views:
        raise ValueError("layer_views must include a 'full' trajectory.")
    return {
        "num_tokens": int(trajectory.shape[0]),
        "num_layers": int(trajectory.shape[1]),
        "hidden_dim": int(trajectory.shape[2]),
        "output": output_token_features(logits, temperature=output_temperature),
        "views": views,
        "key_edge_concentration": layer_edge_concentration(
            views["full"]["coe_mean_token"], key_layers=key_layers
        ),
    }
