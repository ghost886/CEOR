"""Manuscript Eqs. (13)--(16), extracted from the prediction-box experiments.

Masking follows experiments/run_qwen_prediction_box_deletion.py. Relative-L2
uses its head_state_change definition; post-Key1 aggregation follows
scripts/analyze_qwen3vl_all9_key1_layer_ablation.py and the manuscript.
No label, ground-truth box, auxiliary model, or replacement answer is needed.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageStat

from uncertainty.uncertainty_measures import grounding_uncertainty

MASK_METHODS = ("image_mean", "black", "white", "gaussian_blur")


def parse_prediction_box(answer):
    info = grounding_uncertainty.find_first_box_in_text(str(answer))
    box = grounding_uncertainty._coerce_pred_box_to_normalized(info)
    if box is None:
        return None
    try:
        return normalize_box(box)
    except ValueError:
        return None


def normalize_box(box):
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        raise ValueError("Box must have four finite normalized xyxy coordinates.")
    x1, y1, x2, y2 = np.clip(values, 0.0, 1.0)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Predicted box is degenerate.")
    return [float(x1), float(y1), float(x2), float(y2)]


def pixel_box(box, size):
    box = normalize_box(box)
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Image must be nonempty.")
    x1 = max(0, min(width - 1, math.floor(box[0] * width)))
    y1 = max(0, min(height - 1, math.floor(box[1] * height)))
    x2 = max(x1 + 1, min(width, math.ceil(box[2] * width)))
    y2 = max(y1 + 1, min(height, math.ceil(box[3] * height)))
    return [x1, y1, x2, y2]


def mask_image(image: Image.Image, box, *, method="image_mean", blur_fraction=0.15):
    image = image.convert("RGB")
    pixels = pixel_box(box, image.size)
    output = image.copy()
    details = {"pixel_box": pixels, "method": method}
    if method in {"black", "white", "image_mean"}:
        fill = {"black": (0, 0, 0), "white": (255, 255, 255)}.get(method)
        if fill is None:
            fill = tuple(int(round(value)) for value in ImageStat.Stat(image).mean[:3])
        ImageDraw.Draw(output).rectangle(
            [pixels[0], pixels[1], pixels[2] - 1, pixels[3] - 1], fill=fill
        )
        details["fill_rgb"] = list(fill)
    elif method == "gaussian_blur":
        crop = image.crop(tuple(pixels))
        radius = max(2.0, float(blur_fraction) * min(crop.size))
        output.paste(crop.filter(ImageFilter.GaussianBlur(radius=radius)), tuple(pixels))
        details["blur_radius_pixels"] = radius
    else:
        raise ValueError(f"Unsupported mask method: {method}")
    return output, details


def layer_ids_for_shape(shape, model_layer_ids=None):
    if len(shape) != 3 or any(int(x) < 1 for x in shape):
        raise ValueError("Head states must have shape [layers, heads, head_dim].")
    ids = list(range(shape[0])) if model_layer_ids is None else list(model_layer_ids)
    if len(ids) != shape[0] or any(int(x) != x or x < 0 for x in ids):
        raise ValueError("Physical layer IDs must align with the captured layer axis.")
    ids = [int(x) for x in ids]
    if ids != sorted(set(ids)):
        raise ValueError("Physical layer IDs must be unique and ascending.")
    return ids


def head_changes(clean, masked):
    clean = np.asarray(clean, dtype=np.float64)
    masked = np.asarray(masked, dtype=np.float64)
    layer_ids_for_shape(clean.shape)
    if clean.shape != masked.shape:
        raise ValueError("Clean and masked head states have different shapes.")
    if not np.isfinite(clean).all() or not np.isfinite(masked).all():
        raise ValueError("Head states must be finite.")
    clean_norm = np.linalg.norm(clean, axis=-1)
    relative_l2 = np.linalg.norm(masked - clean, axis=-1) / np.maximum(clean_norm, 1e-8)
    cosine = 1.0 - np.clip(
        np.sum(masked * clean, axis=-1)
        / np.maximum(np.linalg.norm(masked, axis=-1) * clean_norm, 1e-8), -1.0, 1.0
    )
    return relative_l2, cosine


def occlusion_response(clean, masked, *, key_layer, model_layer_ids=None):
    """Negative mean relative change over ALL heads STRICTLY after Key1.

    key_layer is a zero-based physical layer ID, not a captured-array index.
    Terminal Key1 yields an unavailable response, never an all-layer fallback.
    """
    relative, cosine = head_changes(clean, masked)
    ids = layer_ids_for_shape(np.shape(clean), model_layer_ids)
    if key_layer not in ids:
        raise ValueError(f"Key1 L{key_layer} is absent from captured layers {ids}.")
    after = [i for i, layer in enumerate(ids) if layer > key_layer]
    score = -float(relative[after].mean()) if after else None
    return {
        "ceor": score,
        "response_available": bool(after),
        "missing_reason": None if after else "no_layers_after_key1",
        "key_layer": int(key_layer),
        "response_layers": [ids[i] for i in after],
        "negative_cosine": -float(cosine[after].mean()) if after else None,
        "relative_l2_per_head": relative.tolist(),
        "cosine_per_head": cosine.tolist(),
    }
