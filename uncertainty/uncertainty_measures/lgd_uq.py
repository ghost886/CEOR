"""Latent Grounding Distribution (LGD) uncertainty teacher.

This module implements the distribution-recovery part of
``docs/LGD_UQ_实验方案与理论推导_v2.md``.  It is deliberately model agnostic:
callers provide one hard ground-truth box, repeated predictions from each
auxiliary/reference model, and repeated predictions from the target model.

The target predictions are never used to construct answer modes.  They are
only projected onto the mode space fixed by ``GT + auxiliary predictions``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from uncertainty.uncertainty_measures.grounding_uncertainty import (
    find_first_box_in_text,
)


Box = Tuple[float, float, float, float]
LGD_UQ_VERSION = "lgd_uq_v1"
OTHER_MODE_NAME = "other_invalid"


@dataclass(frozen=True)
class LGDUQConfig:
    """Numerical configuration for sample-level LGD recovery."""

    iou_threshold: float = 0.5
    smoothing: float = 0.01
    qwen_coordinate_scale: float = 999.0

    def validate(self) -> None:
        if not 0.0 < float(self.iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1].")
        if not float(self.smoothing) > 0.0:
            raise ValueError("smoothing must be positive.")
        if not float(self.qwen_coordinate_scale) > 1.0:
            raise ValueError("qwen_coordinate_scale must be greater than 1.")


def _valid_normalized_box(values: Sequence[float]) -> Optional[Box]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    x1, y1, x2, y2 = (float(value) for value in array)
    if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _normalise_prediction_values(
    values: Sequence[float],
    *,
    image_size: Optional[Tuple[int, int]],
    qwen_coordinate_scale: float,
) -> Optional[Box]:
    """Normalise common VLM box coordinate conventions conservatively.

    Predictions with reversed corners, zero area, non-finite values, or values
    outside the inferred coordinate system remain invalid.  A coordinate-wise
    fallback repairs mixed large-scale and normalised boundary values only when
    the ordinary whole-box interpretation is invalid.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 4 or not np.isfinite(array).all():
        return None
    if float(array.min()) < 0.0:
        return None
    maximum = float(array.max())
    if maximum <= 1.0:
        normalised = array
    elif maximum <= float(qwen_coordinate_scale) + 1.0:
        normalised = array / float(qwen_coordinate_scale)
    elif image_size is not None:
        width, height = image_size
        if int(width) <= 0 or int(height) <= 0:
            return None
        normalised = array.copy()
        normalised[[0, 2]] /= float(width)
        normalised[[1, 3]] /= float(height)
    else:
        return None
    # A coordinate exactly equal to 1000 is common in 0--999/1000 VLM output.
    normalised = np.clip(normalised, 0.0, 1.0)
    valid = _valid_normalized_box(normalised)
    if valid is not None:
        return valid

    # Some VLMs mix a large-coordinate convention with already-normalised
    # right/bottom boundaries, e.g. ``[629, 0, 1.0, 667]``.  Scaling every
    # coordinate turns x2=1.0 into ~0.001 and reverses the box.  Only use the
    # coordinate-wise fallback when the ordinary interpretation is invalid;
    # this preserves legitimate small (for example one-pixel) coordinates in
    # boxes that were already valid under the original convention.
    has_large_coordinate = bool((array > 1.0).any())
    has_normalised_positive_boundary = bool(
        ((array > 0.0) & (array <= 1.0)).any()
    )
    if not (has_large_coordinate and has_normalised_positive_boundary):
        return None

    mixed = array.copy()
    if maximum <= float(qwen_coordinate_scale) + 1.0:
        mixed[array > 1.0] /= float(qwen_coordinate_scale)
    elif image_size is not None:
        width, height = image_size
        x_large = array[[0, 2]] > 1.0
        y_large = array[[1, 3]] > 1.0
        mixed[[0, 2]] = np.where(
            x_large, array[[0, 2]] / float(width), array[[0, 2]]
        )
        mixed[[1, 3]] = np.where(
            y_large, array[[1, 3]] / float(height), array[[1, 3]]
        )
    else:
        return None
    return _valid_normalized_box(np.clip(mixed, 0.0, 1.0))


def parse_grounding_prediction(
    prediction: Any,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    qwen_coordinate_scale: float = 999.0,
) -> Optional[Box]:
    """Parse a box, decoded VLM response, or reference-sample dictionary."""
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        for key in ("box", "bbox", "pred_box"):
            if key in prediction and prediction[key] is not None:
                return parse_grounding_prediction(
                    prediction[key],
                    image_size=image_size,
                    qwen_coordinate_scale=qwen_coordinate_scale,
                )
        for key in ("response", "text", "content"):
            if key in prediction:
                return parse_grounding_prediction(
                    prediction[key],
                    image_size=image_size,
                    qwen_coordinate_scale=qwen_coordinate_scale,
                )
        return None
    if isinstance(prediction, str):
        info = find_first_box_in_text(prediction)
        if info.coord_values is None:
            return None
        values: Sequence[float] = info.coord_values
    else:
        try:
            values = np.asarray(prediction, dtype=float).reshape(-1).tolist()
        except (TypeError, ValueError):
            return None
    return _normalise_prediction_values(
        values,
        image_size=image_size,
        qwen_coordinate_scale=qwen_coordinate_scale,
    )


def normalise_ground_truth_box(
    ground_truth: Any,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    box_format: str = "xyxy",
) -> Box:
    """Normalise the dataset's trusted GT box to xyxy in [0, 1]."""
    if isinstance(ground_truth, str):
        parsed = parse_grounding_prediction(ground_truth, image_size=image_size)
        if parsed is None:
            raise ValueError("Could not parse the hard ground-truth box.")
        return parsed
    values = np.asarray(ground_truth, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("The hard ground-truth box must contain four finite values.")
    if str(box_format).lower() == "xywh":
        values = np.asarray(
            [values[0], values[1], values[0] + values[2], values[1] + values[3]],
            dtype=float,
        )
    elif str(box_format).lower() != "xyxy":
        raise ValueError(f"Unsupported GT box format: {box_format!r}.")
    if float(values.max()) > 1.0:
        if image_size is None:
            raise ValueError("Pixel-space GT requires image_size=(width, height).")
        width, height = image_size
        values[[0, 2]] /= float(width)
        values[[1, 3]] /= float(height)
    # Dataset annotations are trusted, but canonicalise corner ordering.
    x1, y1, x2, y2 = (float(value) for value in values)
    canonical = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    valid = _valid_normalized_box(canonical)
    if valid is None:
        raise ValueError("The hard ground-truth box is invalid after normalisation.")
    return valid


def box_iou(box_a: Box, box_b: Box) -> float:
    """IoU for two normalised xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _connected_components(boxes: Sequence[Box], threshold: float) -> List[List[int]]:
    """Deterministic IoU graph clustering used for the method prototype."""
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            if box_iou(boxes[left], boxes[right]) >= threshold:
                union(left, right)
    groups: Dict[int, List[int]] = {}
    for index in range(len(boxes)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda group: min(group))


def _medoid(boxes: Sequence[Box], indices: Sequence[int]) -> Box:
    best_index = min(indices)
    best_score = -1.0
    for candidate in indices:
        score = sum(box_iou(boxes[candidate], boxes[other]) for other in indices)
        if score > best_score:
            best_index, best_score = candidate, score
    return boxes[best_index]


def _smoothed_distribution(counts: np.ndarray, smoothing: float) -> np.ndarray:
    total = float(counts.sum())
    return (counts + smoothing) / (total + counts.size * smoothing)


def _entropy(probabilities: np.ndarray) -> float:
    return float(-np.sum(probabilities * np.log(probabilities)))


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sum(left * (np.log(left) - np.log(right))))


def _normalise_weights(
    model_names: Sequence[str],
    weights: Optional[Mapping[str, float]],
) -> np.ndarray:
    if not model_names:
        raise ValueError("At least one auxiliary/reference model is required.")
    if weights is None:
        return np.full(len(model_names), 1.0 / len(model_names), dtype=float)
    unknown = sorted(set(weights) - set(model_names))
    if unknown:
        raise ValueError(f"Weights were provided for unknown models: {unknown}.")
    raw = np.asarray([float(weights.get(name, 0.0)) for name in model_names])
    if not np.isfinite(raw).all() or bool((raw < 0.0).any()) or float(raw.sum()) <= 0.0:
        raise ValueError("Auxiliary weights must be finite, non-negative, and sum above zero.")
    return raw / float(raw.sum())


def recover_lgd_uncertainty(
    *,
    ground_truth_box: Any,
    auxiliary_predictions: Mapping[str, Sequence[Any]],
    target_predictions: Optional[Sequence[Any]] = None,
    auxiliary_weights: Optional[Mapping[str, float]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    ground_truth_box_format: str = "xyxy",
    config: Optional[LGDUQConfig] = None,
) -> Dict[str, Any]:
    """Recover the surrogate ideal distribution and decompose uncertainty.

    The returned dictionary contains only Python scalars/lists/dicts and can be
    embedded directly into ``*_generations.pkl``.
    """
    cfg = config or LGDUQConfig()
    cfg.validate()
    model_names = list(auxiliary_predictions)
    weights = _normalise_weights(model_names, auxiliary_weights)
    gt_box = normalise_ground_truth_box(
        ground_truth_box,
        image_size=image_size,
        box_format=ground_truth_box_format,
    )

    boxes: List[Box] = [gt_box]
    owners: List[Tuple[str, int]] = [("__ground_truth__", 0)]
    parsed_by_model: Dict[str, List[Optional[Box]]] = {}
    for model_name in model_names:
        samples = list(auxiliary_predictions[model_name])
        if not samples:
            raise ValueError(f"Auxiliary model {model_name!r} has no samples.")
        parsed_samples = [
            parse_grounding_prediction(
                sample,
                image_size=image_size,
                qwen_coordinate_scale=cfg.qwen_coordinate_scale,
            )
            for sample in samples
        ]
        parsed_by_model[model_name] = parsed_samples
        for sample_index, parsed in enumerate(parsed_samples):
            if parsed is not None:
                boxes.append(parsed)
                owners.append((model_name, sample_index))

    components = _connected_components(boxes, cfg.iou_threshold)
    membership: Dict[int, int] = {}
    mode_records: List[Dict[str, Any]] = []
    for mode_index, component in enumerate(components):
        for box_index in component:
            membership[box_index] = mode_index
        prototype = _medoid(boxes, component)
        member_owners = [owners[index] for index in component]
        mode_records.append({
            "mode_id": f"mode_{mode_index}",
            "prototype_box": list(prototype),
            "member_count": len(component),
            "contains_ground_truth": any(owner == "__ground_truth__" for owner, _ in member_owners),
            "auxiliary_member_count": sum(owner != "__ground_truth__" for owner, _ in member_owners),
        })
    other_index = len(mode_records)
    mode_records.append({
        "mode_id": OTHER_MODE_NAME,
        "prototype_box": None,
        "member_count": sum(
            parsed is None
            for parsed_samples in parsed_by_model.values()
            for parsed in parsed_samples
        ),
        "contains_ground_truth": False,
        "auxiliary_member_count": sum(
            parsed is None
            for parsed_samples in parsed_by_model.values()
            for parsed in parsed_samples
        ),
    })
    mode_count = len(mode_records)

    owner_to_box_index = {owner: index for index, owner in enumerate(owners)}
    distributions = []
    auxiliary_records: Dict[str, Any] = {}
    for model_name in model_names:
        parsed_samples = parsed_by_model[model_name]
        counts = np.zeros(mode_count, dtype=float)
        assignments: List[str] = []
        for sample_index, parsed in enumerate(parsed_samples):
            if parsed is None:
                assigned = other_index
            else:
                assigned = membership[owner_to_box_index[(model_name, sample_index)]]
            counts[assigned] += 1.0
            assignments.append(mode_records[assigned]["mode_id"])
        distribution = _smoothed_distribution(counts, cfg.smoothing)
        distributions.append(distribution)
        auxiliary_records[model_name] = {
            "weight": float(weights[model_names.index(model_name)]),
            "num_samples": len(parsed_samples),
            "num_valid": sum(parsed is not None for parsed in parsed_samples),
            "counts": counts.astype(int).tolist(),
            "distribution": distribution.tolist(),
            "assignments": assignments,
            "parsed_boxes": [list(parsed) if parsed is not None else None for parsed in parsed_samples],
        }

    auxiliary_matrix = np.stack(distributions, axis=0)
    consensus = np.sum(weights[:, None] * auxiliary_matrix, axis=0)
    u_a = float(sum(
        weight * _entropy(distribution)
        for weight, distribution in zip(weights, auxiliary_matrix)
    ))
    u_d = float(sum(
        weight * _kl(distribution, consensus)
        for weight, distribution in zip(weights, auxiliary_matrix)
    ))
    consensus_entropy = _entropy(consensus)

    target_record: Optional[Dict[str, Any]] = None
    u_e: Optional[float] = None
    if target_predictions is not None:
        target_samples = list(target_predictions)
        if not target_samples:
            raise ValueError("target_predictions was provided but is empty.")
        counts = np.zeros(mode_count, dtype=float)
        assignments = []
        parsed_target = []
        # Target outputs can match any member of a fixed auxiliary mode.  They
        # never create a component or alter a prototype (target-leakage guard).
        real_components = components
        for sample in target_samples:
            parsed = parse_grounding_prediction(
                sample,
                image_size=image_size,
                qwen_coordinate_scale=cfg.qwen_coordinate_scale,
            )
            parsed_target.append(parsed)
            assigned = other_index
            best_iou = -1.0
            if parsed is not None:
                for mode_index, component in enumerate(real_components):
                    overlap = max(box_iou(parsed, boxes[index]) for index in component)
                    if overlap >= cfg.iou_threshold and overlap > best_iou:
                        assigned, best_iou = mode_index, overlap
            counts[assigned] += 1.0
            assignments.append(mode_records[assigned]["mode_id"])
        target_distribution = _smoothed_distribution(counts, cfg.smoothing)
        u_e = _kl(consensus, target_distribution)
        target_record = {
            "num_samples": len(target_samples),
            "num_valid": sum(parsed is not None for parsed in parsed_target),
            "counts": counts.astype(int).tolist(),
            "distribution": target_distribution.tolist(),
            "assignments": assignments,
            "parsed_boxes": [list(parsed) if parsed is not None else None for parsed in parsed_target],
        }

    ideal_modes = []
    for mode, probability in zip(mode_records, consensus):
        ideal_modes.append({**mode, "probability": float(probability)})
    denominator = math.log(mode_count) if mode_count > 1 else 1.0
    result: Dict[str, Any] = {
        "version": LGD_UQ_VERSION,
        "task": "visual_grounding",
        "config": asdict(cfg),
        "ideal_answer_distribution": {
            "interpretation": "surrogate_latent_grounding_answer_distribution",
            "construction": "GT-anchored IoU modes; weighted auxiliary-model posterior mean",
            "target_leakage_guard": True,
            "mode_ids": [mode["mode_id"] for mode in mode_records],
            "probabilities": consensus.tolist(),
            "modes": ideal_modes,
        },
        "auxiliary_models": auxiliary_records,
        "target_model": target_record,
        "uncertainty": {
            "u_a": u_a,
            "u_d": u_d,
            "u_e": u_e,
            "u_total": None if u_e is None else u_a + u_d + u_e,
            "consensus_entropy": consensus_entropy,
            "u_a_normalized": u_a / denominator,
            "u_d_normalized": u_d / denominator,
            "decomposition_residual": consensus_entropy - (u_a + u_d),
            "units": "nats",
        },
        "diagnostics": {
            "num_auxiliary_models": len(model_names),
            "num_real_modes": len(mode_records) - 1,
            "num_modes_including_other": mode_count,
            "ground_truth_box": list(gt_box),
            "all_auxiliary_samples_valid": all(
                record["num_valid"] == record["num_samples"]
                for record in auxiliary_records.values()
            ),
        },
    }
    return result


def recover_lgd_iou_ablation(
    *,
    ground_truth_box: Any,
    auxiliary_predictions: Mapping[str, Sequence[Any]],
    iou_thresholds: Sequence[float],
    target_predictions: Optional[Sequence[Any]] = None,
    auxiliary_weights: Optional[Mapping[str, float]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    ground_truth_box_format: str = "xyxy",
    config: Optional[LGDUQConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Recompute LGD-UQ over multiple IoU clustering thresholds.

    The prediction samples are reused exactly; this function performs no model
    calls. Keys use the compact decimal form (for example ``"0.3"``).
    """
    base_config = config or LGDUQConfig()
    thresholds = sorted({float(threshold) for threshold in iou_thresholds})
    if not thresholds:
        raise ValueError("At least one IoU ablation threshold is required.")

    results: Dict[str, Dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_config = replace(base_config, iou_threshold=threshold)
        threshold_config.validate()
        results[f"{threshold:g}"] = recover_lgd_uncertainty(
            ground_truth_box=ground_truth_box,
            auxiliary_predictions=auxiliary_predictions,
            target_predictions=target_predictions,
            auxiliary_weights=auxiliary_weights,
            image_size=image_size,
            ground_truth_box_format=ground_truth_box_format,
            config=threshold_config,
        )
    return results


__all__ = [
    "Box",
    "LGDUQConfig",
    "LGD_UQ_VERSION",
    "OTHER_MODE_NAME",
    "box_iou",
    "normalise_ground_truth_box",
    "parse_grounding_prediction",
    "recover_lgd_iou_ablation",
    "recover_lgd_uncertainty",
]
