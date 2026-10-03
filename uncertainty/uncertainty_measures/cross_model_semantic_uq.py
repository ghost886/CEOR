"""Cross-model semantic uncertainty from Hamidieh et al. (ICLR 2026).

The method operates on black-box response text.  For a target/reference model,
AU is one minus mean within-model semantic similarity; TU is one minus mean
similarity between target responses and responses from an auxiliary ensemble;
EU is TU - AU.  Pairwise means include diagonal target-target pairs, matching
the empirical equations in Section 3.3 of the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


CROSS_MODEL_SEMANTIC_UQ_VERSION = "cross_model_semantic_uq_iclr2026_v1"
DEFAULT_SENTENCE_ENCODER = "sentence-transformers/sentence-t5-xl"


@dataclass(frozen=True)
class CrossModelSemanticUQConfig:
    encoder_name: str = DEFAULT_SENTENCE_ENCODER
    device: str = "cpu"
    batch_size: int = 32

    def validate(self) -> None:
        if not str(self.encoder_name).strip():
            raise ValueError("encoder_name must be non-empty.")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive.")


class SentenceT5SimilarityEncoder:
    """Lazy sentence-transformers wrapper used by generation and offline runs."""

    def __init__(self, config: Optional[CrossModelSemanticUQConfig] = None) -> None:
        self.config = config or CrossModelSemanticUQConfig()
        self.config.validate()
        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Cross-model semantic UQ requires `sentence-transformers`."
                ) from exc
            self._model = SentenceTransformer(
                self.config.encoder_name,
                device=self.config.device,
            )
        return self._model

    def encode(self, responses: Sequence[str]) -> np.ndarray:
        texts = [str(response) for response in responses]
        if not texts:
            raise ValueError("At least one response is required for encoding.")
        embeddings = self._get_model().encode(
            texts,
            batch_size=int(self.config.batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float64)

    def describe(self) -> Dict[str, Any]:
        return asdict(self.config)

    def close(self) -> None:
        self._model = None


def _normalise_embeddings(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} embeddings must have shape [samples, dimensions].")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} embeddings must be finite.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool((norms <= 0.0).any()):
        raise ValueError(f"{name} embeddings must have non-zero norm.")
    return array / norms


def _normalise_model_weights(
    model_names: Sequence[str],
    weights: Optional[Mapping[str, float]],
) -> np.ndarray:
    if not model_names:
        raise ValueError("At least one auxiliary model is required.")
    if weights is None:
        return np.full(len(model_names), 1.0 / len(model_names), dtype=np.float64)
    unknown = sorted(set(weights) - set(model_names))
    if unknown:
        raise ValueError(f"Weights contain unknown auxiliary models: {unknown}.")
    values = np.asarray([float(weights.get(name, 0.0)) for name in model_names])
    if not np.isfinite(values).all() or bool((values < 0.0).any()):
        raise ValueError("Auxiliary weights must be finite and non-negative.")
    if float(values.sum()) <= 0.0:
        raise ValueError("At least one auxiliary weight must be positive.")
    return values / float(values.sum())


def compute_cross_model_semantic_uq_from_embeddings(
    *,
    target_embeddings: Any,
    auxiliary_embeddings: Mapping[str, Any],
    auxiliary_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the paper's empirical AU, EU and TU from response embeddings."""
    target = _normalise_embeddings(target_embeddings, name="target")
    model_names = list(auxiliary_embeddings)
    weights = _normalise_model_weights(model_names, auxiliary_weights)

    self_similarity = float(np.mean(target @ target.T))
    cross_similarities: Dict[str, float] = {}
    auxiliary_sample_counts: Dict[str, int] = {}
    for model_name in model_names:
        auxiliary = _normalise_embeddings(
            auxiliary_embeddings[model_name], name=f"auxiliary {model_name!r}"
        )
        if auxiliary.shape[1] != target.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch for auxiliary model {model_name!r}."
            )
        cross_similarities[model_name] = float(np.mean(target @ auxiliary.T))
        auxiliary_sample_counts[model_name] = int(auxiliary.shape[0])

    weighted_cross_similarity = float(sum(
        weight * cross_similarities[model_name]
        for model_name, weight in zip(model_names, weights)
    ))
    aleatoric = 1.0 - self_similarity
    total = 1.0 - weighted_cross_similarity
    epistemic = total - aleatoric
    decomposition_residual = total - (aleatoric + epistemic)
    if not all(math.isfinite(value) for value in (aleatoric, epistemic, total)):
        raise ValueError("Cross-model semantic uncertainty is non-finite.")

    return {
        "version": CROSS_MODEL_SEMANTIC_UQ_VERSION,
        "method": "complementing_self_consistency_with_cross_model_disagreement",
        "uncertainty": {
            "u_aleatoric": float(aleatoric),
            "u_epistemic": float(epistemic),
            "u_total": float(total),
            "decomposition_residual": float(decomposition_residual),
        },
        "similarity": {
            "target_self_similarity": self_similarity,
            "weighted_cross_model_similarity": weighted_cross_similarity,
            "cross_model_similarity_by_model": cross_similarities,
            "metric": "cosine",
        },
        "sampling": {
            "num_target_responses": int(target.shape[0]),
            "num_auxiliary_responses_by_model": auxiliary_sample_counts,
            "auxiliary_weights": {
                name: float(weight) for name, weight in zip(model_names, weights)
            },
        },
    }


def _response_text(record: Any) -> Optional[str]:
    if isinstance(record, Mapping):
        if bool(record.get("excluded_from_uncertainty", False)):
            return None
        if "infrastructure_error" in record or "content_filter_error" in record:
            return None
        value = record.get("response", record.get("text", record.get("content")))
    else:
        value = record
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def response_texts(records: Sequence[Any]) -> List[str]:
    """Extract successful non-empty response text from saved sampling records."""
    return [text for record in records if (text := _response_text(record)) is not None]


def compute_cross_model_semantic_uq(
    *,
    target_responses: Sequence[Any],
    auxiliary_responses: Mapping[str, Sequence[Any]],
    encoder: SentenceT5SimilarityEncoder,
    auxiliary_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Encode raw responses and compute the paper's black-box uncertainty scores."""
    target_texts = response_texts(target_responses)
    if not target_texts:
        raise ValueError("No usable target responses are available.")
    auxiliary_texts = {
        model_name: response_texts(records)
        for model_name, records in auxiliary_responses.items()
    }
    empty_models = [name for name, texts in auxiliary_texts.items() if not texts]
    if empty_models:
        raise ValueError(f"Auxiliary models have no usable responses: {empty_models}.")

    ordered_texts = list(target_texts)
    spans: Dict[str, slice] = {}
    for model_name, texts in auxiliary_texts.items():
        start = len(ordered_texts)
        ordered_texts.extend(texts)
        spans[model_name] = slice(start, len(ordered_texts))
    embeddings = encoder.encode(ordered_texts)
    result = compute_cross_model_semantic_uq_from_embeddings(
        target_embeddings=embeddings[:len(target_texts)],
        auxiliary_embeddings={
            model_name: embeddings[span] for model_name, span in spans.items()
        },
        auxiliary_weights=auxiliary_weights,
    )
    result["encoder"] = encoder.describe()
    result["response_representation"] = "raw_generated_text"
    return result


__all__ = [
    "CROSS_MODEL_SEMANTIC_UQ_VERSION",
    "CrossModelSemanticUQConfig",
    "DEFAULT_SENTENCE_ENCODER",
    "SentenceT5SimilarityEncoder",
    "compute_cross_model_semantic_uq",
    "compute_cross_model_semantic_uq_from_embeddings",
    "response_texts",
]
