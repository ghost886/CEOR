"""Grounding-specific uncertainty metrics for visual localization outputs.

This module is intentionally independent from the generation patch.  It takes
decoded text, generated-token logits, and attention tensors, then computes
coordinate confidence and box-attention consistency when a bbox can be parsed
from the model output.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


TensorLikeBox = Union[torch.Tensor, Sequence[float]]


@dataclass
class BoxInfo:
    pred_box: Optional[torch.Tensor]
    box_token_indices: torch.Tensor
    coord_token_groups: List[torch.Tensor]
    coord_values: Optional[Tuple[float, float, float, float]] = None
    box_char_span: Optional[Tuple[int, int]] = None
    coord_char_spans: Optional[List[Tuple[int, int]]] = None


@dataclass
class TokenRoleSets:
    vision_indices: torch.Tensor
    text_indices: torch.Tensor
    referring_phrase_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    output_box_token_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    coord_token_groups: List[torch.Tensor] = field(default_factory=list)
    distractor_boxes: Optional[List[torch.Tensor]] = None
    visual_active_layers: Optional[Tuple[int, ...]] = None
    localization_heads: Optional[List[Tuple[int, int]]] = None
    head_weights: Optional[Dict[Tuple[int, int], float]] = None


@dataclass
class CoordUncertainty:
    u_coord_nll: float
    u_coord_entropy: float
    u_coord_margin: float
    per_coord_nll: List[float]
    per_coord_entropy: List[float]
    per_coord_margin: List[float]


@dataclass
class AttentionProfile:
    raw_profile: torch.Tensor
    norm_profile: torch.Tensor
    visual_reliance: float
    used_heads: int


@dataclass
class GroundingUncertainty:
    u_ground: float
    u_box_attn: float
    mass_in_pred_box: float
    spatial_entropy: float
    visual_reliance: float
    box_attn_iou: float
    center_distance: float
    attn_box: Optional[torch.Tensor]
    attn_box_area: float = 0.0
    top_patch_mass: float = 0.0
    attention_valid: float = 1.0


@dataclass
class StepGroundingMetrics:
    u_step_inside: float
    step_mass_in_pred_box: float
    u_step_box_attn: float
    step_box_attn_iou: float
    u_step_distractor: float
    u_step_temporal: float
    step_visual_reliance: float
    step_attn_box: Optional[torch.Tensor]
    step_attn_box_area: float = 0.0
    step_attention_valid: float = 1.0
    selected_step_count: int = 0
    selected_layer_count: int = 0
    selected_head_count: int = 0


@dataclass
class ConsistencyUncertainty:
    u_cons: float
    mean_pairwise_iou: float
    center_std: float
    area_std: float
    num_boxes: int


@dataclass
class UncertaintyBreakdown:
    u_coord: float
    u_ground: float
    u_box_attn: float
    u_cons: float
    u_distractor: float
    u_total: float
    reject: bool
    w_coord: float
    w_ground: float
    w_box_attn: float
    w_cons: float
    w_distractor: float


_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)"
_BOX_PATTERNS = [
    re.compile(
        rf"<box>\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*</box>",
        re.IGNORECASE,
    ),
    # Qwen-VL grounding format: (x1,y1),(x2,y2), usually in 0-999 coordinates.
    re.compile(
        rf"\(?\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*\)?"
    ),
    re.compile(
        rf"\[\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*\]"
    ),
    re.compile(
        rf"\(\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*[,;\s]\s*({_NUM})\s*\)"
    ),
]


def normalize_xyxy_box(
    box: TensorLikeBox,
    image_size: Optional[Tuple[int, int]] = None,
    clamp: bool = True,
) -> torch.Tensor:
    b = torch.as_tensor(box, dtype=torch.float32).clone()
    if b.numel() != 4:
        raise ValueError(f"box must have 4 values, got {b.numel()}")
    if image_size is not None and float(b.max()) > 1.5:
        w, h = image_size
        b[0] = b[0] / float(w)
        b[2] = b[2] / float(w)
        b[1] = b[1] / float(h)
        b[3] = b[3] / float(h)
    x1, y1, x2, y2 = b.tolist()
    out = torch.tensor([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=torch.float32)
    return out.clamp(0.0, 1.0) if clamp else out


def xywh_to_xyxy(box: TensorLikeBox) -> torch.Tensor:
    b = torch.as_tensor(box, dtype=torch.float32)
    x, y, w, h = b.tolist()
    return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)


def find_first_box_in_text(
    text: str,
    *,
    box_format: str = "xyxy",
    image_size: Optional[Tuple[int, int]] = None,
) -> BoxInfo:
    for pattern in _BOX_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        values = tuple(float(match.group(i)) for i in range(1, 5))
        box = torch.tensor(values, dtype=torch.float32)
        if box_format == "xywh":
            box = xywh_to_xyxy(box)
        elif box_format != "xyxy":
            raise ValueError(f"Unsupported box_format: {box_format}")
        return BoxInfo(
            pred_box=normalize_xyxy_box(box, image_size=image_size),
            box_token_indices=torch.empty(0, dtype=torch.long),
            coord_token_groups=[],
            coord_values=values,
            box_char_span=match.span(0),
            coord_char_spans=[match.span(i) for i in range(1, 5)],
        )
    return BoxInfo(
        pred_box=None,
        box_token_indices=torch.empty(0, dtype=torch.long),
        coord_token_groups=[],
    )


def _token_indices_overlapping_char_span(
    offsets: Sequence[Tuple[int, int]],
    char_span: Tuple[int, int],
) -> torch.Tensor:
    start, end = char_span
    indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start or tok_start >= end or tok_start == tok_end:
            continue
        indices.append(idx)
    return torch.tensor(indices, dtype=torch.long)


def extract_output_box_info(
    output_text: str,
    tokenizer=None,
    *,
    box_format: str = "xyxy",
    image_size: Optional[Tuple[int, int]] = None,
) -> BoxInfo:
    info = find_first_box_in_text(output_text, box_format=box_format, image_size=image_size)
    if tokenizer is None or info.pred_box is None:
        return info
    try:
        enc = tokenizer(output_text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:
        return info
    offsets = enc["offset_mapping"]
    if info.box_char_span is not None:
        info.box_token_indices = _token_indices_overlapping_char_span(offsets, info.box_char_span)
    if info.coord_char_spans is not None:
        info.coord_token_groups = [
            _token_indices_overlapping_char_span(offsets, span)
            for span in info.coord_char_spans
        ]
    return info


def shift_box_info_token_indices(info: BoxInfo, offset: int) -> BoxInfo:
    return BoxInfo(
        pred_box=info.pred_box,
        box_token_indices=info.box_token_indices + offset,
        coord_token_groups=[group + offset for group in info.coord_token_groups],
        coord_values=info.coord_values,
        box_char_span=info.box_char_span,
        coord_char_spans=info.coord_char_spans,
    )


def box_area_xyxy(box: TensorLikeBox) -> torch.Tensor:
    b = torch.as_tensor(box, dtype=torch.float32)
    return (b[2] - b[0]).clamp_min(0.0) * (b[3] - b[1]).clamp_min(0.0)


def box_iou_xyxy(box1: TensorLikeBox, box2: TensorLikeBox) -> float:
    b1 = torch.as_tensor(box1, dtype=torch.float32)
    b2 = torch.as_tensor(box2, dtype=torch.float32)
    inter = box_area_xyxy([
        torch.maximum(b1[0], b2[0]),
        torch.maximum(b1[1], b2[1]),
        torch.minimum(b1[2], b2[2]),
        torch.minimum(b1[3], b2[3]),
    ])
    union = box_area_xyxy(b1) + box_area_xyxy(b2) - inter
    if float(union) <= 1e-12:
        return 0.0
    return float((inter / union).clamp(0.0, 1.0))


def box_center_xyxy(box: TensorLikeBox) -> torch.Tensor:
    b = torch.as_tensor(box, dtype=torch.float32)
    return torch.tensor([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=torch.float32)


def box_center_distance(box1: TensorLikeBox, box2: TensorLikeBox) -> float:
    return float(torch.norm(box_center_xyxy(box1) - box_center_xyxy(box2), p=2) / math.sqrt(2.0))


def _coerce_pred_box_to_normalized(
    info: BoxInfo,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    qwen_coordinate_scale: float = 999.0,
) -> Optional[torch.Tensor]:
    if info.pred_box is None:
        return None
    if info.coord_values is None:
        return normalize_xyxy_box(info.pred_box)

    raw = torch.as_tensor(info.coord_values, dtype=torch.float32)
    if raw.numel() != 4:
        return None
    if float(raw.max()) <= 1.5:
        return normalize_xyxy_box(raw)

    # Qwen-VL's official grounding evaluation parses ``(x1,y1),(x2,y2)``
    # in a 0-999 coordinate system before comparing with the ground-truth box.
    if float(raw.max()) <= qwen_coordinate_scale + 1:
        return normalize_xyxy_box(raw / float(qwen_coordinate_scale))

    if image_size is not None:
        return normalize_xyxy_box(raw, image_size=image_size)
    return normalize_xyxy_box(raw)


def _raw_pred_box_to_normalized(
    info: BoxInfo,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    qwen_coordinate_scale: float = 999.0,
) -> Optional[torch.Tensor]:
    """Normalize coordinate units while preserving raw order and range."""
    if info.coord_values is None:
        return None
    raw = torch.as_tensor(info.coord_values, dtype=torch.float32).clone()
    if raw.numel() != 4 or not bool(torch.isfinite(raw).all()):
        return None
    if float(raw.abs().max()) <= 1.5:
        return raw
    if float(raw.abs().max()) <= qwen_coordinate_scale + 1:
        return raw / float(qwen_coordinate_scale)
    if image_size is not None:
        width, height = image_size
        raw[0] /= float(width)
        raw[2] /= float(width)
        raw[1] /= float(height)
        raw[3] /= float(height)
        return raw
    return raw


def evaluate_grounding_prediction(
    predicted_answer: str,
    example: dict,
    *,
    iou_thresholds: Sequence[float] = (0.25, 0.5, 0.75),
    accuracy_iou_threshold: float = 0.5,
    qwen_coordinate_scale: float = 999.0,
) -> Dict[str, float]:
    """Evaluate one grounding prediction.

    This mirrors Qwen-VL's RefCOCO evaluation in spirit: parse a predicted box,
    compute IoU against the target box, and report grounding accuracy /
    Precision@1 at IoU 0.5.
    The parser accepts both normalized ``[x1,y1,x2,y2]`` and Qwen-style
    ``(x1,y1),(x2,y2)`` / 0-999 coordinates.
    """
    width = example.get("image_width", None)
    height = example.get("image_height", None)
    image_size = None
    if width is not None and height is not None and int(width) > 0 and int(height) > 0:
        image_size = (int(width), int(height))

    pred_info = find_first_box_in_text(predicted_answer)
    raw_pred_box = _raw_pred_box_to_normalized(
        pred_info,
        image_size=image_size,
        qwen_coordinate_scale=qwen_coordinate_scale,
    )
    pred_box = _coerce_pred_box_to_normalized(
        pred_info,
        image_size=image_size,
        qwen_coordinate_scale=qwen_coordinate_scale,
    )

    if "bbox" in example:
        gt_box = normalize_xyxy_box(example["bbox"], image_size=image_size)
    elif "answer" in example:
        gt_info = find_first_box_in_text(str(example["answer"]))
        gt_box = gt_info.pred_box
    else:
        gt_box = None

    parsed = pred_box is not None
    has_gt = gt_box is not None
    if not parsed or not has_gt:
        out = {
            "parse_success": float(parsed),
            "has_ground_truth": float(has_gt),
            "iou": 0.0,
            "accuracy": 0.0,
            f"accuracy_iou_{accuracy_iou_threshold:g}": 0.0,
            "precision_at_1": 0.0,
            "center_distance": 1.0,
        }
    else:
        iou = box_iou_xyxy(pred_box, gt_box)
        is_correct = 1.0 if iou >= float(accuracy_iou_threshold) else 0.0
        out = {
            "parse_success": 1.0,
            "has_ground_truth": 1.0,
            "iou": float(iou),
            "accuracy": is_correct,
            f"accuracy_iou_{accuracy_iou_threshold:g}": is_correct,
            "precision_at_1": is_correct,
            "center_distance": box_center_distance(pred_box, gt_box),
        }

    for threshold in iou_thresholds:
        out[f"precision_at_iou_{threshold:g}"] = 1.0 if out["iou"] >= float(threshold) else 0.0

    if parsed:
        for key, value in zip(("pred_x1", "pred_y1", "pred_x2", "pred_y2"), pred_box.tolist()):
            out[key] = float(value)
    if raw_pred_box is not None:
        for key, value in zip(
            ("pred_raw_x1", "pred_raw_y1", "pred_raw_x2", "pred_raw_y2"),
            raw_pred_box.tolist(),
        ):
            out[key] = float(value)
        raw_x1, raw_y1, raw_x2, raw_y2 = raw_pred_box.tolist()
        out["raw_geometry_valid"] = float(
            raw_x2 > raw_x1
            and raw_y2 > raw_y1
            and raw_x1 >= 0.0
            and raw_y1 >= 0.0
            and raw_x2 <= 1.0
            and raw_y2 <= 1.0
        )
    if has_gt:
        for key, value in zip(("gt_x1", "gt_y1", "gt_x2", "gt_y2"), gt_box.tolist()):
            out[key] = float(value)
    return out


def patches_inside_box(
    box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    mode: str = "center",
) -> torch.Tensor:
    box = normalize_xyxy_box(box)
    x1, y1, x2, y2 = box.tolist()
    gh, gw = grid_size
    mask = torch.zeros(gh * gw, dtype=torch.bool)
    for row in range(gh):
        for col in range(gw):
            idx = row * gw + col
            if mode == "center":
                px = (col + 0.5) / gw
                py = (row + 0.5) / gh
                inside = (x1 <= px <= x2) and (y1 <= py <= y2)
            elif mode == "overlap":
                patch_x1, patch_y1 = col / gw, row / gh
                patch_x2, patch_y2 = (col + 1) / gw, (row + 1) / gh
                inside = min(x2, patch_x2) > max(x1, patch_x1) and min(y2, patch_y2) > max(y1, patch_y1)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            mask[idx] = inside
    return mask


def box_from_attention_map(
    attn_profile: torch.Tensor,
    grid_size: Tuple[int, int],
    *,
    mass_ratio: Optional[float] = None,
    top_fraction: float = 0.15,
    largest_component: bool = True,
) -> Optional[torch.Tensor]:
    if attn_profile.numel() == 0:
        return None
    gh, gw = grid_size
    if attn_profile.numel() != gh * gw:
        raise ValueError(f"attn_profile length {attn_profile.numel()} != grid_h * grid_w {gh * gw}")
    p = attn_profile.float()
    p = p / p.sum().clamp_min(1e-12)
    order = torch.argsort(p, descending=True)
    if mass_ratio is not None:
        cumsum = torch.cumsum(p[order], dim=0)
        selected_count = int((cumsum < float(mass_ratio)).sum().item()) + 1
    else:
        selected_count = int(math.ceil(float(top_fraction) * p.numel()))
    selected_count = max(1, min(selected_count, int(p.numel())))
    selected = order[:selected_count]

    if largest_component and selected.numel() > 1:
        candidate = torch.zeros(gh * gw, dtype=torch.bool, device=p.device)
        candidate[selected] = True
        visited = torch.zeros_like(candidate)
        best_mass = -1.0
        best_comp = selected
        for start in selected.tolist():
            if bool(visited[start]):
                continue
            stack = [int(start)]
            visited[start] = True
            comp = []
            while stack:
                idx = stack.pop()
                comp.append(idx)
                row, col = divmod(idx, gw)
                for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if nr < 0 or nr >= gh or nc < 0 or nc >= gw:
                        continue
                    nidx = nr * gw + nc
                    if bool(candidate[nidx]) and not bool(visited[nidx]):
                        visited[nidx] = True
                        stack.append(nidx)
            comp_idx = torch.tensor(comp, dtype=torch.long, device=p.device)
            comp_mass = float(p[comp_idx].sum())
            if comp_mass > best_mass:
                best_mass = comp_mass
                best_comp = comp_idx
        selected = best_comp

    rows = selected // gw
    cols = selected % gw
    return torch.tensor(
        [
            int(cols.min().item()) / gw,
            int(rows.min().item()) / gh,
            (int(cols.max().item()) + 1) / gw,
            (int(rows.max().item()) + 1) / gh,
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)


def _step_logits_to_vector(logits_i: torch.Tensor) -> torch.Tensor:
    if logits_i.dim() == 1:
        return logits_i.float()
    if logits_i.dim() == 2:
        return logits_i[0].float()
    if logits_i.dim() == 3:
        return logits_i[0, -1, :].float()
    raise ValueError(f"Unsupported logits shape: {tuple(logits_i.shape)}")


def compute_u_coord(
    step_logits: Sequence[torch.Tensor],
    token_ids: Sequence[int],
    *,
    coord_token_groups: Optional[List[torch.Tensor]] = None,
    box_token_indices: Optional[torch.Tensor] = None,
) -> CoordUncertainty:
    if len(step_logits) == 0 or len(token_ids) == 0:
        return CoordUncertainty(0.0, 0.0, 0.0, [], [], [])
    if coord_token_groups:
        groups = coord_token_groups
    elif box_token_indices is not None and box_token_indices.numel() > 0:
        groups = [box_token_indices]
    else:
        groups = [torch.arange(len(token_ids), dtype=torch.long)]

    per_coord_nll: List[float] = []
    per_coord_entropy: List[float] = []
    per_coord_margin: List[float] = []
    for group in groups:
        nll_vals = []
        ent_vals = []
        margin_vals = []
        for pos in group.tolist():
            if pos < 0 or pos >= len(step_logits) or pos >= len(token_ids):
                continue
            logits = _step_logits_to_vector(step_logits[pos])
            tid = int(token_ids[pos])
            if tid < 0 or tid >= logits.numel():
                continue
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            top2 = torch.topk(probs, k=min(2, probs.numel())).values
            margin = top2[0] - top2[1] if top2.numel() == 2 else top2[0]
            nll_vals.append(float(-log_probs[tid]))
            ent_vals.append(float((-(probs * log_probs).sum() / math.log(max(logits.numel(), 2))).clamp(0.0, 1.0)))
            margin_vals.append(float((1.0 - margin).clamp(0.0, 1.0)))
        if nll_vals:
            per_coord_nll.append(float(sum(nll_vals) / len(nll_vals)))
            per_coord_entropy.append(float(sum(ent_vals) / len(ent_vals)))
            per_coord_margin.append(float(sum(margin_vals) / len(margin_vals)))

    if not per_coord_nll:
        return CoordUncertainty(0.0, 0.0, 0.0, [], [], [])
    return CoordUncertainty(
        u_coord_nll=float(sum(per_coord_nll) / len(per_coord_nll)),
        u_coord_entropy=float(sum(per_coord_entropy) / len(per_coord_entropy)),
        u_coord_margin=float(sum(per_coord_margin) / len(per_coord_margin)),
        per_coord_nll=per_coord_nll,
        per_coord_entropy=per_coord_entropy,
        per_coord_margin=per_coord_margin,
    )


def _attention_layer_to_heads(attn_l: torch.Tensor, *, batch_idx: int = 0) -> torch.Tensor:
    if attn_l.dim() == 4:
        return attn_l[batch_idx].float()
    if attn_l.dim() == 3:
        return attn_l.float()
    raise ValueError(f"Unsupported attention shape: {tuple(attn_l.shape)}")


def _valid_indices(indices: torch.Tensor, upper: int) -> torch.Tensor:
    indices = indices.long()
    return indices[(indices >= 0) & (indices < upper)]


def aggregate_attention_profile(
    attentions: Sequence[torch.Tensor],
    *,
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    localization_heads: Optional[List[Tuple[int, int]]] = None,
    head_weights: Optional[Dict[Tuple[int, int], float]] = None,
    batch_idx: int = 0,
) -> AttentionProfile:
    if len(attentions) == 0:
        raise ValueError("attentions is empty")
    device = attentions[0].device
    if query_indices.numel() == 0 or key_indices.numel() == 0:
        z = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
        return AttentionProfile(z, z, 0.0, 0)
    if layers is None:
        layers = list(range(len(attentions)))
    layer_to_heads: Dict[int, List[int]] = {}
    if localization_heads is not None:
        for layer_idx, head_idx in localization_heads:
            layer_to_heads.setdefault(layer_idx, []).append(head_idx)

    acc = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
    weight_sum = 0.0
    used_heads = 0
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(attentions):
            continue
        ah = _attention_layer_to_heads(attentions[layer_idx], batch_idx=batch_idx)
        num_heads, q_len, k_len = ah.shape
        q_idx = _valid_indices(query_indices.to(device), q_len)
        k_idx = _valid_indices(key_indices.to(device), k_len)
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        heads = list(range(num_heads)) if localization_heads is None else layer_to_heads.get(layer_idx, [])
        valid_key_mask = (key_indices.to(device) >= 0) & (key_indices.to(device) < k_len)
        for head_idx in heads:
            if head_idx < 0 or head_idx >= num_heads:
                continue
            weight = float(head_weights.get((layer_idx, head_idx), 1.0)) if head_weights else 1.0
            profile = ah[head_idx].index_select(0, q_idx).index_select(1, k_idx).mean(dim=0)
            full_profile = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
            full_profile[valid_key_mask] = profile
            acc += weight * full_profile
            weight_sum += weight
            used_heads += 1
    if weight_sum <= 0:
        z = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
        return AttentionProfile(z, z, 0.0, 0)
    raw = acc / weight_sum
    norm = raw / raw.sum().clamp_min(1e-12)
    return AttentionProfile(raw, norm, float(raw.sum().clamp(0.0, 1.0)), used_heads)


def _normalize_generation_step_indices(
    step_indices: torch.Tensor,
    step_count: int,
) -> torch.Tensor:
    if step_count <= 0 or step_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    idx = torch.unique(step_indices.detach().cpu().long())
    return idx[(idx >= 0) & (idx < step_count)]


def _default_late_layers(n_layers: int, late_fraction: float = 0.25) -> Tuple[int, ...]:
    if n_layers <= 0:
        return ()
    start = max(0, int(math.floor(n_layers * (1.0 - late_fraction))))
    return tuple(range(start, n_layers))


def aggregate_step_attention_profile(
    step_attentions: Sequence[Sequence[torch.Tensor]],
    *,
    step_indices: torch.Tensor,
    key_indices: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    localization_heads: Optional[List[Tuple[int, int]]] = None,
    head_weights: Optional[Dict[Tuple[int, int], float]] = None,
    batch_idx: int = 0,
) -> AttentionProfile:
    """Aggregate decode-step attention from selected generated-token steps to vision keys.

    Each generation step is expected to be a tuple/list over layers.  With KV
    cache enabled, decode steps usually have query length 1; for the first saved
    prefill step, the last query row is used as the best available proxy.
    """
    valid_steps = _normalize_generation_step_indices(step_indices, len(step_attentions))
    device = key_indices.device
    if len(step_attentions) > 0 and len(step_attentions[0]) > 0:
        device = step_attentions[0][0].device
    if valid_steps.numel() == 0 or key_indices.numel() == 0:
        z = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
        return AttentionProfile(z, z, 0.0, 0)

    first_step = step_attentions[int(valid_steps[0].item())]
    if layers is None:
        layers = _default_late_layers(len(first_step))
    layer_to_heads: Dict[int, List[int]] = {}
    if localization_heads is not None:
        for layer_idx, head_idx in localization_heads:
            layer_to_heads.setdefault(int(layer_idx), []).append(int(head_idx))

    acc = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
    weight_sum = 0.0
    used_heads = 0
    key_indices_device = key_indices.to(device)
    for step_idx in valid_steps.tolist():
        layer_attentions = step_attentions[int(step_idx)]
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(layer_attentions):
                continue
            ah = _attention_layer_to_heads(layer_attentions[layer_idx], batch_idx=batch_idx)
            num_heads, q_len, k_len = ah.shape
            if q_len <= 0 or k_len <= 0:
                continue
            q_idx = torch.tensor([q_len - 1], dtype=torch.long, device=ah.device)
            k_idx = _valid_indices(key_indices_device, k_len)
            if k_idx.numel() == 0:
                continue
            heads = list(range(num_heads)) if localization_heads is None else layer_to_heads.get(int(layer_idx), [])
            valid_key_mask = (key_indices_device >= 0) & (key_indices_device < k_len)
            for head_idx in heads:
                if head_idx < 0 or head_idx >= num_heads:
                    continue
                weight = float(head_weights.get((int(layer_idx), int(head_idx)), 1.0)) if head_weights else 1.0
                profile = ah[head_idx].index_select(0, q_idx).index_select(1, k_idx).squeeze(0)
                full_profile = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
                full_profile[valid_key_mask] = profile
                acc += weight * full_profile
                weight_sum += weight
                used_heads += 1
    if weight_sum <= 0:
        z = torch.zeros(key_indices.numel(), dtype=torch.float32, device=device)
        return AttentionProfile(z, z, 0.0, 0)
    raw = acc / weight_sum
    norm = raw / raw.sum().clamp_min(1e-12)
    return AttentionProfile(raw, norm, float(raw.sum().clamp(0.0, 1.0)), used_heads)


def _generalized_jensen_shannon(profiles: Sequence[torch.Tensor]) -> float:
    """Normalized generalized JS divergence for visual attention profiles."""
    valid = []
    for profile in profiles:
        values = profile.detach().float().cpu().reshape(-1).clamp_min(0.0)
        if values.numel() == 0 or float(values.sum()) <= 1e-12:
            continue
        valid.append(values / values.sum())
    if len(valid) < 2:
        return 0.0
    stacked = torch.stack(valid, dim=0)
    mean = stacked.mean(dim=0)

    def entropy(values: torch.Tensor) -> torch.Tensor:
        return -(values * values.clamp_min(1e-12).log()).sum(dim=-1)

    divergence = entropy(mean) - entropy(stacked).mean()
    maximum = math.log(max(2, min(len(valid), int(mean.numel()))))
    return float((divergence / max(maximum, 1e-12)).clamp(0.0, 1.0))


def compute_step_head_js(
    step_attentions: Sequence[Sequence[torch.Tensor]],
    token_roles: TokenRoleSets,
    *,
    step_token_indices: Optional[torch.Tensor] = None,
    layers: Optional[Sequence[int]] = None,
    localization_heads: Optional[List[Tuple[int, int]]] = None,
    batch_idx: int = 0,
) -> Dict[str, float]:
    """Measure decode-step visual-profile disagreement across localization heads.

    This is a diagnostic proxy, not a causal use-of-attention claim.  Passing a
    fixed ``localization_heads`` set is preferred for confirmatory experiments;
    when omitted, all heads in the selected layers are included.
    """
    if not step_attentions or token_roles.vision_indices.numel() == 0:
        return {
            "head_js": float("nan"),
            "valid_heads": 0.0,
        }
    if step_token_indices is None:
        parts = [
            group.detach().cpu().long()
            for group in token_roles.coord_token_groups
            if group.numel() > 0
        ]
        step_token_indices = (
            torch.unique(torch.cat(parts))
            if parts else token_roles.output_box_token_indices.detach().cpu().long()
        )
    valid_steps = _normalize_generation_step_indices(step_token_indices, len(step_attentions))
    if valid_steps.numel() == 0:
        return {
            "head_js": float("nan"),
            "valid_heads": 0.0,
        }
    if layers is None:
        layers = _default_late_layers(len(step_attentions[int(valid_steps[0].item())]))
    layers = tuple(int(layer) for layer in layers)
    requested_heads: Dict[int, List[int]] = {}
    if localization_heads is not None:
        for layer, head in localization_heads:
            requested_heads.setdefault(int(layer), []).append(int(head))

    head_profiles: List[torch.Tensor] = []
    for layer in layers:
        first_layer = step_attentions[int(valid_steps[0].item())]
        if layer < 0 or layer >= len(first_layer):
            continue
        available = _attention_layer_to_heads(first_layer[layer], batch_idx=batch_idx).shape[0]
        heads = requested_heads.get(layer, []) if localization_heads is not None else list(range(available))
        for head in heads:
            profile = aggregate_step_attention_profile(
                step_attentions,
                step_indices=valid_steps,
                key_indices=token_roles.vision_indices,
                layers=(layer,),
                localization_heads=[(layer, int(head))],
                head_weights=token_roles.head_weights,
                batch_idx=batch_idx,
            )
            if profile.used_heads > 0 and float(profile.norm_profile.sum()) > 0:
                head_profiles.append(profile.norm_profile.detach().float().cpu())

    return {
        "head_js": _generalized_jensen_shannon(head_profiles),
        "valid_heads": float(len(head_profiles)),
    }


def select_localization_heads_by_concentration(
    attentions: Sequence[torch.Tensor],
    *,
    query_indices: torch.Tensor,
    key_indices: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    top_k_per_layer: int = 4,
    batch_idx: int = 0,
) -> List[Tuple[int, int]]:
    if len(attentions) == 0 or query_indices.numel() == 0 or key_indices.numel() == 0:
        return []
    if layers is None:
        layers = list(range(len(attentions)))
    selected: List[Tuple[int, int]] = []
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(attentions):
            continue
        ah = _attention_layer_to_heads(attentions[layer_idx], batch_idx=batch_idx)
        num_heads, q_len, k_len = ah.shape
        q_idx = _valid_indices(query_indices.to(ah.device), q_len)
        k_idx = _valid_indices(key_indices.to(ah.device), k_len)
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            continue
        scored: List[Tuple[float, int]] = []
        for head_idx in range(num_heads):
            raw = ah[head_idx].index_select(0, q_idx).index_select(1, k_idx).mean(dim=0).float()
            reliance = float(raw.sum().clamp(0.0, 1.0))
            if reliance <= 0:
                continue
            p = raw / raw.sum().clamp_min(1e-12)
            concentration = 1.0 - normalized_entropy(p)
            scored.append((reliance * concentration, head_idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected.extend((layer_idx, head_idx) for _, head_idx in scored[:max(1, top_k_per_layer)])
    return selected


def _attention_query_len(attentions: Sequence[torch.Tensor]) -> int:
    if not attentions:
        return 0
    first = attentions[0]
    if first.dim() >= 2:
        return int(first.shape[-2])
    return 0


def _select_grounding_query_indices(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    query_priority: Tuple[str, ...],
) -> torch.Tensor:
    """Select query tokens that are addressable in the available attention tensors.

    Qwen generation currently exposes reliable prefill attentions.  Generated bbox
    token positions are still stored in ``output_box_token_indices`` as full-sequence
    positions, but they are not valid query rows for a prefill-only attention map.
    This helper validates indices before choosing a role and falls back to prompt
    referring/text tokens when generated-token queries are unavailable.
    """
    q_len = _attention_query_len(attentions)
    for name in query_priority:
        if name == "output_box":
            indices = token_roles.output_box_token_indices
        elif name == "referring_phrase":
            indices = token_roles.referring_phrase_indices
        elif name == "text":
            indices = token_roles.text_indices
        else:
            continue
        valid = _valid_indices(indices, q_len)
        if valid.numel() > 0:
            return valid
    device = attentions[0].device if attentions else token_roles.vision_indices.device
    return torch.empty(0, dtype=torch.long, device=device)


def normalized_entropy(p: torch.Tensor) -> float:
    p = p.float()
    p = p / p.sum().clamp_min(1e-12)
    if p.numel() <= 1:
        return 0.0
    return float((-(p * (p + 1e-12).log()).sum() / math.log(p.numel())).clamp(0.0, 1.0))


def compute_grounding_uncertainty(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    pred_box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    query_priority: Tuple[str, ...] = ("output_box", "referring_phrase", "text"),
    box_mass_ratio_for_attn_box: Optional[float] = None,
    box_top_fraction_for_attn_box: float = 0.15,
    max_valid_attn_box_area: float = 0.95,
    ground_weights: Optional[Dict[str, float]] = None,
    batch_idx: int = 0,
) -> GroundingUncertainty:
    weights = ground_weights or {"inside": 0.45, "entropy": 0.25, "reliance": 0.30}
    query_indices = _select_grounding_query_indices(attentions, token_roles, query_priority)
    if query_indices.numel() == 0 or token_roles.vision_indices.numel() == 0:
        return GroundingUncertainty(1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, None)

    layers = token_roles.visual_active_layers or tuple(range(len(attentions)))
    profile = aggregate_attention_profile(
        attentions,
        query_indices=query_indices,
        key_indices=token_roles.vision_indices,
        layers=layers,
        localization_heads=token_roles.localization_heads,
        head_weights=token_roles.head_weights,
        batch_idx=batch_idx,
    )
    pred_box_t = normalize_xyxy_box(pred_box)
    inside_mask = patches_inside_box(pred_box_t, grid_size).to(profile.norm_profile.device)
    if inside_mask.numel() != profile.norm_profile.numel():
        raise ValueError(
            f"inside_mask length {inside_mask.numel()} != vision profile length {profile.norm_profile.numel()}"
        )
    mass_in_box = float(profile.norm_profile[inside_mask].sum().clamp(0.0, 1.0))
    ent = normalized_entropy(profile.norm_profile)
    top_patch_mass = float(profile.norm_profile.max().clamp(0.0, 1.0))
    attn_box = box_from_attention_map(
        profile.norm_profile.detach().cpu(),
        grid_size,
        mass_ratio=box_mass_ratio_for_attn_box,
        top_fraction=box_top_fraction_for_attn_box,
        largest_component=True,
    )
    if attn_box is None:
        box_attn_iou = 0.0
        center_dist = 1.0
        attn_box_area = 0.0
        attention_valid = 0.0
    else:
        attn_box_area = float(box_area_xyxy(attn_box))
        attention_valid = 1.0 if attn_box_area < max_valid_attn_box_area else 0.0
        box_attn_iou = box_iou_xyxy(pred_box_t, attn_box) if attention_valid else 0.0
        center_dist = box_center_distance(pred_box_t, attn_box) if attention_valid else 1.0
    u_ground = (
        weights.get("inside", 0.45) * (1.0 - mass_in_box)
        + weights.get("entropy", 0.25) * ent
        + weights.get("reliance", 0.30) * (1.0 - profile.visual_reliance)
    )
    return GroundingUncertainty(
        u_ground=float(max(0.0, min(1.0, u_ground))),
        # Reversed direction for evaluation: higher means stronger box-attention alignment.
        u_box_attn=float(max(0.0, min(1.0, box_attn_iou))),
        mass_in_pred_box=mass_in_box,
        spatial_entropy=ent,
        visual_reliance=profile.visual_reliance,
        box_attn_iou=box_attn_iou,
        center_distance=center_dist,
        attn_box=attn_box,
        attn_box_area=attn_box_area,
        top_patch_mass=top_patch_mass,
        attention_valid=attention_valid,
    )


def propose_distractor_boxes_from_attention(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    pred_box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    query_priority: Tuple[str, ...] = ("output_box", "referring_phrase", "text"),
    max_boxes: int = 3,
    top_mass_ratio: float = 0.60,
    top_patch_fraction: float = 0.10,
    min_box_mass: float = 0.03,
    max_pred_iou: float = 0.50,
    batch_idx: int = 0,
) -> List[torch.Tensor]:
    """Mine high-attention non-predicted regions as distractor boxes.

    Without annotated distractors, we treat concentrated visual evidence outside
    the predicted box as hard negative regions.  The procedure is deliberately
    conservative: remove patches overlapping the predicted box, keep the top
    outside-attention patches, form connected components, and return boxes whose
    attention mass is large enough and whose IoU with the prediction is small.
    """
    if not attentions or token_roles.vision_indices.numel() == 0:
        return []
    gh, gw = grid_size
    if gh <= 0 or gw <= 0:
        return []
    query_indices = _select_grounding_query_indices(attentions, token_roles, query_priority)
    if query_indices.numel() == 0:
        return []

    profile = aggregate_attention_profile(
        attentions,
        query_indices=query_indices,
        key_indices=token_roles.vision_indices,
        layers=token_roles.visual_active_layers or tuple(range(len(attentions))),
        localization_heads=token_roles.localization_heads,
        head_weights=token_roles.head_weights,
        batch_idx=batch_idx,
    )
    if profile.norm_profile.numel() != gh * gw or float(profile.norm_profile.sum()) <= 0:
        return []

    pred_box_t = normalize_xyxy_box(pred_box)
    pred_mask = patches_inside_box(pred_box_t, grid_size, mode="overlap").to(profile.norm_profile.device)
    outside = ~pred_mask
    outside_mass = profile.norm_profile[outside].sum()
    if float(outside_mass) < min_box_mass:
        return []

    outside_profile = profile.norm_profile.clone()
    outside_profile[~outside] = 0
    order = torch.argsort(outside_profile, descending=True)
    order = order[outside_profile[order] > 0]
    if order.numel() == 0:
        return []
    mass_keep = int((torch.cumsum(outside_profile[order], dim=0) < float(top_mass_ratio) * outside_mass).sum().item()) + 1
    frac_keep = int(math.ceil(float(top_patch_fraction) * int(order.numel())))
    keep_count = max(1, min(mass_keep, frac_keep, int(order.numel())))
    candidate = torch.zeros(gh * gw, dtype=torch.bool, device=profile.norm_profile.device)
    candidate[order[:keep_count]] = True

    visited = torch.zeros_like(candidate)
    components: List[Tuple[float, torch.Tensor]] = []
    for start in torch.nonzero(candidate, as_tuple=False).flatten().tolist():
        if bool(visited[start]):
            continue
        stack = [int(start)]
        visited[start] = True
        comp = []
        while stack:
            idx = stack.pop()
            comp.append(idx)
            row, col = divmod(idx, gw)
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nr < 0 or nr >= gh or nc < 0 or nc >= gw:
                    continue
                nidx = nr * gw + nc
                if bool(candidate[nidx]) and not bool(visited[nidx]):
                    visited[nidx] = True
                    stack.append(nidx)
        comp_idx = torch.tensor(comp, dtype=torch.long, device=profile.norm_profile.device)
        mass = float(profile.norm_profile[comp_idx].sum().clamp(0.0, 1.0))
        if mass < min_box_mass:
            continue
        rows = comp_idx // gw
        cols = comp_idx % gw
        box = torch.tensor(
            [
                int(cols.min().item()) / gw,
                int(rows.min().item()) / gh,
                (int(cols.max().item()) + 1) / gw,
                (int(rows.max().item()) + 1) / gh,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        if box_iou_xyxy(pred_box_t, box) <= max_pred_iou:
            components.append((mass, box.cpu()))

    components.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in components[:max_boxes]]


def compute_u_distractor(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    pred_box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    batch_idx: int = 0,
) -> float:
    if not token_roles.distractor_boxes:
        return 0.0
    query_indices = _select_grounding_query_indices(
        attentions,
        token_roles,
        ("output_box", "referring_phrase", "text"),
    )
    if query_indices.numel() == 0 or token_roles.vision_indices.numel() == 0:
        return 0.0
    profile = aggregate_attention_profile(
        attentions,
        query_indices=query_indices,
        key_indices=token_roles.vision_indices,
        layers=token_roles.visual_active_layers or tuple(range(len(attentions))),
        localization_heads=token_roles.localization_heads,
        head_weights=token_roles.head_weights,
        batch_idx=batch_idx,
    )
    pred_box_t = normalize_xyxy_box(pred_box)
    pred_mask = patches_inside_box(pred_box_t, grid_size).to(profile.norm_profile.device)
    pred_mass = (
        float(profile.norm_profile[pred_mask].sum().clamp(0.0, 1.0))
        if pred_mask.numel() == profile.norm_profile.numel()
        else 0.0
    )
    max_mass = 0.0
    for dbox in token_roles.distractor_boxes:
        dbox_t = normalize_xyxy_box(dbox)
        if box_iou_xyxy(pred_box_t, dbox_t) > 0.5:
            continue
        mask = patches_inside_box(dbox_t, grid_size).to(profile.norm_profile.device)
        if mask.numel() == profile.norm_profile.numel():
            max_mass = max(max_mass, float(profile.norm_profile[mask].sum().clamp(0.0, 1.0)))
    denom = pred_mass + max_mass
    if denom <= 1e-12:
        return 0.0
    return float(max_mass / denom)


def _propose_distractor_boxes_from_profile(
    profile: torch.Tensor,
    pred_box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    max_boxes: int = 3,
    top_mass_ratio: float = 0.60,
    top_patch_fraction: float = 0.10,
    min_box_mass: float = 0.03,
    max_pred_iou: float = 0.50,
) -> List[torch.Tensor]:
    gh, gw = grid_size
    if profile.numel() != gh * gw or float(profile.sum()) <= 0:
        return []
    pred_box_t = normalize_xyxy_box(pred_box)
    pred_mask = patches_inside_box(pred_box_t, grid_size, mode="overlap").to(profile.device)
    outside = ~pred_mask
    outside_mass = profile[outside].sum()
    if float(outside_mass) < min_box_mass:
        return []

    outside_profile = profile.clone()
    outside_profile[~outside] = 0
    order = torch.argsort(outside_profile, descending=True)
    order = order[outside_profile[order] > 0]
    if order.numel() == 0:
        return []
    mass_keep = int((torch.cumsum(outside_profile[order], dim=0) < float(top_mass_ratio) * outside_mass).sum().item()) + 1
    frac_keep = int(math.ceil(float(top_patch_fraction) * int(order.numel())))
    keep_count = max(1, min(mass_keep, frac_keep, int(order.numel())))
    candidate = torch.zeros(gh * gw, dtype=torch.bool, device=profile.device)
    candidate[order[:keep_count]] = True

    visited = torch.zeros_like(candidate)
    components: List[Tuple[float, torch.Tensor]] = []
    for start in torch.nonzero(candidate, as_tuple=False).flatten().tolist():
        if bool(visited[start]):
            continue
        stack = [int(start)]
        visited[start] = True
        comp = []
        while stack:
            idx = stack.pop()
            comp.append(idx)
            row, col = divmod(idx, gw)
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nr < 0 or nr >= gh or nc < 0 or nc >= gw:
                    continue
                nidx = nr * gw + nc
                if bool(candidate[nidx]) and not bool(visited[nidx]):
                    visited[nidx] = True
                    stack.append(nidx)
        comp_idx = torch.tensor(comp, dtype=torch.long, device=profile.device)
        mass = float(profile[comp_idx].sum().clamp(0.0, 1.0))
        if mass < min_box_mass:
            continue
        rows = comp_idx // gw
        cols = comp_idx % gw
        box = torch.tensor(
            [
                int(cols.min().item()) / gw,
                int(rows.min().item()) / gh,
                (int(cols.max().item()) + 1) / gw,
                (int(rows.max().item()) + 1) / gh,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        if box_iou_xyxy(pred_box_t, box) <= max_pred_iou:
            components.append((mass, box.cpu()))

    components.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in components[:max_boxes]]


def _mean_pairwise_profile_distance(profiles: Sequence[torch.Tensor]) -> float:
    valid = [p.float() / p.float().sum().clamp_min(1e-12) for p in profiles if p.numel() > 0 and float(p.sum()) > 0]
    if len(valid) < 2:
        return 0.0
    distances: List[float] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sim = F.cosine_similarity(valid[i], valid[j], dim=0, eps=1e-12)
            distances.append(float((1.0 - sim).clamp(0.0, 1.0)))
    return float(sum(distances) / len(distances)) if distances else 0.0


def compute_step_grounding_metrics(
    step_attentions: Sequence[Sequence[torch.Tensor]],
    token_roles: TokenRoleSets,
    pred_box: TensorLikeBox,
    grid_size: Tuple[int, int],
    *,
    step_token_indices: Optional[torch.Tensor] = None,
    coord_token_groups: Optional[List[torch.Tensor]] = None,
    layers: Optional[Sequence[int]] = None,
    localization_heads: Optional[List[Tuple[int, int]]] = None,
    box_top_fraction_for_attn_box: float = 0.15,
    max_valid_attn_box_area: float = 0.95,
    batch_idx: int = 0,
) -> StepGroundingMetrics:
    if not step_attentions or token_roles.vision_indices.numel() == 0:
        return StepGroundingMetrics(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, None, selected_step_count=0)
    if step_token_indices is None:
        if coord_token_groups:
            parts = [g.detach().cpu().long() for g in coord_token_groups if g.numel() > 0]
            step_token_indices = torch.unique(torch.cat(parts)) if parts else torch.empty(0, dtype=torch.long)
        else:
            step_token_indices = token_roles.output_box_token_indices.detach().cpu().long()
    valid_steps = _normalize_generation_step_indices(step_token_indices, len(step_attentions))
    if valid_steps.numel() == 0:
        return StepGroundingMetrics(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, None, selected_step_count=0)

    if layers is None:
        layers = _default_late_layers(len(step_attentions[int(valid_steps[0].item())]))
    profile = aggregate_step_attention_profile(
        step_attentions,
        step_indices=valid_steps,
        key_indices=token_roles.vision_indices,
        layers=layers,
        localization_heads=localization_heads,
        head_weights=token_roles.head_weights,
        batch_idx=batch_idx,
    )
    pred_box_t = normalize_xyxy_box(pred_box)
    inside_mask = patches_inside_box(pred_box_t, grid_size).to(profile.norm_profile.device)
    if inside_mask.numel() == profile.norm_profile.numel():
        mass_in_box = float(profile.norm_profile[inside_mask].sum().clamp(0.0, 1.0))
    else:
        mass_in_box = 0.0

    attn_box = box_from_attention_map(
        profile.norm_profile.detach().cpu(),
        grid_size,
        top_fraction=box_top_fraction_for_attn_box,
        largest_component=True,
    )
    if attn_box is None:
        box_attn_iou = 0.0
        attn_box_area = 0.0
        attention_valid = 0.0
    else:
        attn_box_area = float(box_area_xyxy(attn_box))
        attention_valid = 1.0 if attn_box_area < max_valid_attn_box_area else 0.0
        box_attn_iou = box_iou_xyxy(pred_box_t, attn_box) if attention_valid else 0.0

    distractor_boxes = token_roles.distractor_boxes or _propose_distractor_boxes_from_profile(
        profile.norm_profile,
        pred_box_t,
        grid_size,
    )
    max_distractor_mass = 0.0
    for dbox in distractor_boxes:
        dbox_t = normalize_xyxy_box(dbox)
        if box_iou_xyxy(pred_box_t, dbox_t) > 0.5:
            continue
        mask = patches_inside_box(dbox_t, grid_size).to(profile.norm_profile.device)
        if mask.numel() == profile.norm_profile.numel():
            max_distractor_mass = max(
                max_distractor_mass,
                float(profile.norm_profile[mask].sum().clamp(0.0, 1.0)),
            )
    denom = mass_in_box + max_distractor_mass
    u_step_distractor = float(max_distractor_mass / denom) if denom > 1e-12 else 0.0

    coord_profiles: List[torch.Tensor] = []
    for group in coord_token_groups or []:
        group_steps = _normalize_generation_step_indices(group, len(step_attentions))
        if group_steps.numel() == 0:
            continue
        group_profile = aggregate_step_attention_profile(
            step_attentions,
            step_indices=group_steps,
            key_indices=token_roles.vision_indices,
            layers=layers,
            localization_heads=localization_heads,
            head_weights=token_roles.head_weights,
            batch_idx=batch_idx,
        )
        coord_profiles.append(group_profile.norm_profile.detach())
    u_step_temporal = _mean_pairwise_profile_distance(coord_profiles)

    return StepGroundingMetrics(
        u_step_inside=float(max(0.0, min(1.0, 1.0 - mass_in_box))),
        step_mass_in_pred_box=mass_in_box,
        u_step_box_attn=float(max(0.0, min(1.0, 1.0 - box_attn_iou))),
        step_box_attn_iou=float(box_attn_iou),
        u_step_distractor=float(max(0.0, min(1.0, u_step_distractor))),
        u_step_temporal=float(max(0.0, min(1.0, u_step_temporal))),
        step_visual_reliance=profile.visual_reliance,
        step_attn_box=attn_box,
        step_attn_box_area=attn_box_area,
        step_attention_valid=attention_valid,
        selected_step_count=int(valid_steps.numel()),
        selected_layer_count=len(tuple(layers)),
        selected_head_count=profile.used_heads,
    )


def _select_query_debug(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    query_priority: Tuple[str, ...] = ("output_box", "referring_phrase", "text"),
) -> Tuple[str, torch.Tensor]:
    q_len = _attention_query_len(attentions)
    for name in query_priority:
        if name == "output_box":
            indices = token_roles.output_box_token_indices
        elif name == "referring_phrase":
            indices = token_roles.referring_phrase_indices
        elif name == "text":
            indices = token_roles.text_indices
        else:
            continue
        valid = _valid_indices(indices, q_len)
        if valid.numel() > 0:
            return name, valid
    device = attentions[0].device if attentions else token_roles.vision_indices.device
    return "none", torch.empty(0, dtype=torch.long, device=device)


def _profile_to_grid(profile: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    gh, gw = grid_size
    if profile.numel() != gh * gw:
        return torch.empty(0, dtype=torch.float32)
    return profile.detach().float().cpu().reshape(gh, gw)


def _safe_token_label(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return str(token_id)


def _save_heatmap_png(matrix: torch.Tensor, path: str, *, title: str = "") -> None:
    if matrix.numel() == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for attention debug plots: {exc}") from exc
    arr = matrix.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="viridis")
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_profile_csv(matrix: torch.Tensor, path: str) -> None:
    arr = matrix.detach().float().cpu()
    with open(path, "w", encoding="utf-8") as f:
        for row in arr.tolist():
            f.write(",".join(f"{float(v):.8g}" for v in row) + "\n")


def _draw_box(ax, box: TensorLikeBox, label: str, color: str) -> None:
    import matplotlib.patches as patches
    b = normalize_xyxy_box(box).tolist()
    rect = patches.Rectangle(
        (b[0], b[1]),
        max(b[2] - b[0], 1e-6),
        max(b[3] - b[1], 1e-6),
        linewidth=2.0,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        b[0],
        max(0.0, b[1] - 0.015),
        label,
        color=color,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1},
    )


def _save_overlay_png(
    image,
    profile_grid: torch.Tensor,
    path: str,
    *,
    pred_box: Optional[TensorLikeBox] = None,
    attn_box: Optional[TensorLikeBox] = None,
    distractor_boxes: Optional[Sequence[TensorLikeBox]] = None,
    title: str = "",
) -> None:
    if profile_grid.numel() == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for attention debug plots: {exc}") from exc
    arr = profile_grid.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    if image is not None:
        ax.imshow(image, extent=(0, 1, 1, 0))
        ax.imshow(arr, cmap="magma", alpha=0.45, extent=(0, 1, 1, 0), interpolation="nearest")
    else:
        ax.imshow(arr, cmap="magma", extent=(0, 1, 1, 0), interpolation="nearest")
    if pred_box is not None:
        _draw_box(ax, pred_box, "pred", "lime")
    if attn_box is not None:
        _draw_box(ax, attn_box, "attn", "cyan")
    for idx, dbox in enumerate(distractor_boxes or []):
        _draw_box(ax, dbox, f"d{idx}", "red")
    if title:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grounding_attention_debug(
    *,
    output_dir: str,
    sample_id: str,
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
    grid_size: Tuple[int, int],
    input_ids: Optional[torch.Tensor] = None,
    tokenizer=None,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    image=None,
    pred_box: Optional[TensorLikeBox] = None,
    grounding_unc: Optional[GroundingUncertainty] = None,
    u_distractor: Optional[float] = None,
    batch_idx: int = 0,
) -> Dict[str, str]:
    """Write detailed diagnostics for attention-based grounding metrics.

    The function is intentionally side-effect-only for debugging; it does not
    change any uncertainty values.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))[:120] or "sample"
    prefix = os.path.join(output_dir, safe_id)
    artifacts: Dict[str, str] = {}

    query_name, query_indices = _select_query_debug(attentions, token_roles)
    layers = token_roles.visual_active_layers or tuple(range(len(attentions)))
    vision_profile = aggregate_attention_profile(
        attentions,
        query_indices=query_indices,
        key_indices=token_roles.vision_indices,
        layers=layers,
        localization_heads=token_roles.localization_heads,
        head_weights=token_roles.head_weights,
        batch_idx=batch_idx,
    )
    vision_grid = _profile_to_grid(vision_profile.norm_profile, grid_size)

    per_layer = []
    for li in range(len(attentions)):
        ah = _attention_layer_to_heads(attentions[li], batch_idx=batch_idx)
        layer_row = {
            "layer": li,
            "heads": int(ah.shape[0]),
            "q_len": int(ah.shape[1]),
            "k_len": int(ah.shape[2]),
            "active": li in set(layers),
        }
        if query_indices.numel() > 0 and token_roles.vision_indices.numel() > 0:
            q_idx = _valid_indices(query_indices.to(ah.device), ah.shape[-2])
            v_idx = _valid_indices(token_roles.vision_indices.to(ah.device), ah.shape[-1])
            if q_idx.numel() > 0 and v_idx.numel() > 0:
                per_head = ah.index_select(1, q_idx).index_select(2, v_idx).mean(dim=(1, 2))
                top_k = min(5, int(per_head.numel()))
                vals, heads = torch.topk(per_head.float(), top_k)
                layer_row["top_t2v_heads"] = [
                    (int(h.item()), float(v.item())) for h, v in zip(heads, vals)
                ]
        per_layer.append(layer_row)

    summary_path = f"{prefix}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"sample_id: {sample_id}\n")
        if question is not None:
            f.write(f"question: {question}\n")
        if answer is not None:
            f.write(f"answer: {answer}\n")
        f.write(f"attention_layers: {len(attentions)}\n")
        f.write(f"grid_size: {grid_size}\n")
        f.write(f"active_layers: {tuple(layers)}\n")
        f.write(f"query_role_used: {query_name}\n")
        f.write(f"query_indices_count: {int(query_indices.numel())}\n")
        f.write(f"vision_indices_count: {int(token_roles.vision_indices.numel())}\n")
        f.write(f"text_indices_count: {int(token_roles.text_indices.numel())}\n")
        f.write(f"output_box_token_indices_count: {int(token_roles.output_box_token_indices.numel())}\n")
        f.write(f"grounding_u_distractor: {u_distractor}\n")
        f.write(f"used_heads_in_vision_profile: {vision_profile.used_heads}\n")
        f.write(f"visual_reliance_raw_sum: {vision_profile.visual_reliance}\n")
        if pred_box is not None:
            f.write(f"pred_box: {[float(x) for x in normalize_xyxy_box(pred_box).tolist()]}\n")
        if grounding_unc is not None:
            f.write(f"mass_in_pred_box: {grounding_unc.mass_in_pred_box}\n")
            f.write(f"box_attn_iou: {grounding_unc.box_attn_iou}\n")
            f.write(f"attn_box: {grounding_unc.attn_box.tolist() if grounding_unc.attn_box is not None else None}\n")
        if token_roles.distractor_boxes:
            f.write("distractor_boxes:\n")
            for didx, dbox in enumerate(token_roles.distractor_boxes):
                dbox_t = normalize_xyxy_box(dbox)
                mass = None
                if vision_profile.norm_profile.numel() == grid_size[0] * grid_size[1]:
                    mask = patches_inside_box(dbox_t, grid_size).to(vision_profile.norm_profile.device)
                    mass = float(vision_profile.norm_profile[mask].sum().clamp(0.0, 1.0))
                f.write(f"  {didx}. box={dbox_t.tolist()} mass={mass}\n")
        f.write("per_layer_head_summary:\n")
        for row in per_layer:
            f.write(f"  {row}\n")
    artifacts["summary"] = summary_path

    if vision_grid.numel() > 0:
        csv_path = f"{prefix}_text_to_image_attention.csv"
        png_path = f"{prefix}_text_to_image_attention.png"
        overlay_path = f"{prefix}_image_attention_overlay.png"
        _save_profile_csv(vision_grid, csv_path)
        _save_heatmap_png(vision_grid, png_path, title=f"{safe_id} text/query -> image")
        _save_overlay_png(
            image,
            vision_grid,
            overlay_path,
            pred_box=pred_box,
            attn_box=grounding_unc.attn_box if grounding_unc is not None else None,
            distractor_boxes=token_roles.distractor_boxes,
            title=f"{safe_id} attention / boxes",
        )
        artifacts["text_to_image_csv"] = csv_path
        artifacts["text_to_image_png"] = png_path
        artifacts["overlay_png"] = overlay_path

    if query_indices.numel() > 0 and token_roles.text_indices.numel() > 0:
        text_profile = aggregate_attention_profile(
            attentions,
            query_indices=query_indices,
            key_indices=token_roles.text_indices,
            layers=layers,
            localization_heads=token_roles.localization_heads,
            head_weights=token_roles.head_weights,
            batch_idx=batch_idx,
        )
        txt_path = f"{prefix}_question_token_attention.csv"
        row_ids = input_ids[0].detach().cpu() if input_ids is not None and input_ids.numel() > 0 else None
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("rank,seq_pos,token_id,token,raw_attention,normalized_attention\n")
            order = torch.argsort(text_profile.raw_profile.float(), descending=True)
            for rank, local_idx in enumerate(order.tolist(), start=1):
                seq_pos = int(token_roles.text_indices[local_idx].item())
                token_id = int(row_ids[seq_pos].item()) if row_ids is not None and seq_pos < row_ids.numel() else -1
                f.write(
                    f"{rank},{seq_pos},{token_id},"
                    f"{_safe_token_label(tokenizer, token_id)!r},"
                    f"{float(text_profile.raw_profile[local_idx]):.8g},"
                    f"{float(text_profile.norm_profile[local_idx]):.8g}\n"
                )
        artifacts["question_token_csv"] = txt_path

    return artifacts


def compute_box_consistency(boxes: Sequence[TensorLikeBox]) -> ConsistencyUncertainty:
    valid_boxes = []
    for box in boxes:
        try:
            norm_box = normalize_xyxy_box(box)
            if float(box_area_xyxy(norm_box)) > 1e-8:
                valid_boxes.append(norm_box)
        except Exception:
            continue
    n = len(valid_boxes)
    if n <= 1:
        return ConsistencyUncertainty(0.0, 1.0, 0.0, 0.0, n)
    ious = [box_iou_xyxy(valid_boxes[i], valid_boxes[j]) for i in range(n) for j in range(i + 1, n)]
    centers = torch.stack([box_center_xyxy(box) for box in valid_boxes], dim=0)
    areas = torch.stack([box_area_xyxy(box) for box in valid_boxes], dim=0)
    mean_iou = float(sum(ious) / max(len(ious), 1))
    return ConsistencyUncertainty(
        u_cons=float(max(0.0, min(1.0, 1.0 - mean_iou))),
        mean_pairwise_iou=mean_iou,
        center_std=float(centers.std(dim=0).mean().clamp(0.0, 1.0)),
        area_std=float(areas.std().clamp(0.0, 1.0)),
        num_boxes=n,
    )


@dataclass
class QuantileNormalizer:
    q_low: float = 0.05
    q_high: float = 0.95
    stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def fit(self, values: Dict[str, Sequence[float]]) -> "QuantileNormalizer":
        self.stats = {}
        for key, arr in values.items():
            t = torch.as_tensor(arr, dtype=torch.float32)
            if t.numel() == 0:
                continue
            lo = float(torch.quantile(t, self.q_low))
            hi = float(torch.quantile(t, self.q_high))
            self.stats[key] = (lo, hi if hi > lo else lo + 1e-6)
        return self

    def transform_value(self, key: str, value: float) -> float:
        if key not in self.stats:
            return float(value)
        lo, hi = self.stats[key]
        return float(max(0.0, min(1.0, (float(value) - lo) / (hi - lo))))


def compute_total_grounding_uncertainty(
    *,
    u_coord: float,
    u_ground: float,
    u_box_attn: float,
    u_cons: float = 0.0,
    u_distractor: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
) -> UncertaintyBreakdown:
    w = {
        "coord": 0.20,
        "ground": 0.25,
        "box_attn": 0.25,
        "cons": 0.20,
        "distractor": 0.05,
    }
    if weights is not None:
        w.update(weights)
    weight_sum = sum(w.values())
    if weight_sum <= 0:
        raise ValueError("weights sum must be positive")
    w = {key: value / weight_sum for key, value in w.items()}
    u_total = (
        w["coord"] * u_coord
        + w["ground"] * u_ground
        + w["box_attn"] * u_box_attn
        + w["cons"] * u_cons
        + w["distractor"] * u_distractor
    )
    u_total = float(max(0.0, min(1.0, u_total)))
    return UncertaintyBreakdown(
        u_coord=float(u_coord),
        u_ground=float(u_ground),
        u_box_attn=float(u_box_attn),
        u_cons=float(u_cons),
        u_distractor=float(u_distractor),
        u_total=u_total,
        reject=bool(u_total >= threshold) if threshold is not None else False,
        w_coord=w["coord"],
        w_ground=w["ground"],
        w_box_attn=w["box_attn"],
        w_cons=w["cons"],
        w_distractor=w["distractor"],
    )


def compute_grounding_metrics_pipeline(
    *,
    output_text: str,
    tokenizer,
    attentions: Sequence[torch.Tensor],
    step_logits: Sequence[torch.Tensor],
    generated_token_ids: Sequence[int],
    token_roles: TokenRoleSets,
    grid_size: Tuple[int, int],
    image_size: Optional[Tuple[int, int]] = None,
    box_format: str = "xyxy",
    generation_token_offset_in_full_sequence: Optional[int] = None,
    consistency_boxes: Optional[Sequence[TensorLikeBox]] = None,
    coord_nll_normalizer: Optional[QuantileNormalizer] = None,
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
) -> Tuple[
    Optional[BoxInfo],
    Optional[CoordUncertainty],
    Optional[GroundingUncertainty],
    Optional[ConsistencyUncertainty],
    Optional[UncertaintyBreakdown],
]:
    box_info = extract_output_box_info(output_text, tokenizer=tokenizer, box_format=box_format, image_size=image_size)
    if box_info.pred_box is None:
        return box_info, None, None, None, None
    coerced_pred_box = _coerce_pred_box_to_normalized(box_info, image_size=image_size)
    if coerced_pred_box is not None:
        box_info.pred_box = coerced_pred_box
    coord_unc = compute_u_coord(
        step_logits,
        generated_token_ids,
        coord_token_groups=box_info.coord_token_groups,
        box_token_indices=box_info.box_token_indices,
    )
    u_coord = (
        coord_nll_normalizer.transform_value("u_coord_nll", coord_unc.u_coord_nll)
        if coord_nll_normalizer is not None
        else coord_unc.u_coord_entropy
    )
    attn_token_roles = token_roles
    if generation_token_offset_in_full_sequence is not None:
        abs_box_info = shift_box_info_token_indices(box_info, generation_token_offset_in_full_sequence)
        attn_token_roles = TokenRoleSets(
            vision_indices=token_roles.vision_indices,
            text_indices=token_roles.text_indices,
            referring_phrase_indices=token_roles.referring_phrase_indices,
            output_box_token_indices=abs_box_info.box_token_indices,
            coord_token_groups=abs_box_info.coord_token_groups,
            distractor_boxes=token_roles.distractor_boxes,
            visual_active_layers=token_roles.visual_active_layers,
            localization_heads=token_roles.localization_heads,
            head_weights=token_roles.head_weights,
        )
    if not attn_token_roles.distractor_boxes:
        distractor_boxes = propose_distractor_boxes_from_attention(
            attentions,
            attn_token_roles,
            box_info.pred_box,
            grid_size,
        )
        if distractor_boxes:
            attn_token_roles = TokenRoleSets(
                vision_indices=attn_token_roles.vision_indices,
                text_indices=attn_token_roles.text_indices,
                referring_phrase_indices=attn_token_roles.referring_phrase_indices,
                output_box_token_indices=attn_token_roles.output_box_token_indices,
                coord_token_groups=attn_token_roles.coord_token_groups,
                distractor_boxes=distractor_boxes,
                visual_active_layers=attn_token_roles.visual_active_layers,
                localization_heads=attn_token_roles.localization_heads,
                head_weights=attn_token_roles.head_weights,
            )
    grounding_unc = compute_grounding_uncertainty(attentions, attn_token_roles, box_info.pred_box, grid_size)
    u_distractor = compute_u_distractor(attentions, attn_token_roles, box_info.pred_box, grid_size)
    consistency_unc = (
        compute_box_consistency(consistency_boxes)
        if consistency_boxes is not None
        else ConsistencyUncertainty(0.0, 1.0, 0.0, 0.0, 1)
    )
    breakdown = compute_total_grounding_uncertainty(
        u_coord=u_coord,
        u_ground=grounding_unc.u_ground,
        u_box_attn=grounding_unc.u_box_attn,
        u_cons=consistency_unc.u_cons,
        u_distractor=u_distractor,
        weights=weights,
        threshold=threshold,
    )
    return box_info, coord_unc, grounding_unc, consistency_unc, breakdown
