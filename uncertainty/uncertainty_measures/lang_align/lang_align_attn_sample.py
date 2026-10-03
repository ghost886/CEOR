"""
Attention-based language–vision alignment sampling for multimodal uncertainty.

During ``generate``, this module:
1. Collects self-attention on the prefill step.
2. Resolves text and vision token regions for attention summaries.
3. Optionally runs a no-image chain (``prepare_inputs_for_generation_cd``) for visual
   evidence, same interface as ``vcd_sample`` but with a separate output type and patch.
4. Computes per-step and sequence-level uncertainty combining LM probability,
   visual-evidence gap, and text-to-vision attention concentration.

Usage::

    from uncertainty.uncertainty_measures.lang_align.lang_align_attn_sample import evolve_lang_align_attn_sampling
    evolve_lang_align_attn_sampling()
    out = model.generate(
        **inputs,
        lang_align_decode=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.utils import ModelOutput

try:
    from transformers import Cache
except ImportError:  # Older transformers versions do not export Cache.
    Cache = object  # type: ignore[misc,assignment]

try:
    from transformers.generation.utils import GenerateNonBeamOutput, logger
except ImportError:
    from transformers.generation.utils import logger
    GenerateNonBeamOutput = torch.LongTensor  # type: ignore[misc,assignment]

# from uncertainty.uncertainty_measures.lang_align.scoring import pred_probs_from_step_logits

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

_ORIGINAL_SAMPLE = None


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class TokenRoleSets:
    """Index sets (sequence positions) for prefill text and vision regions."""

    vision_indices: torch.LongTensor
    text_indices: torch.LongTensor
    visual_active_layers: Tuple[int, ...] = ()


@dataclass
class StepUncertaintyMetrics:
    """Per generated token t."""

    token_id: int
    u_lang: float
    u_align: float
    u_attn: float
    u_total: float


@dataclass
class UncertaintyBreakdown:
    """
    Sequence-level uncertainty decomposition.

    U_total = w_lang * U_lang + w_align * U_align + w_attn * U_attn
    """

    u_lang: float
    u_align: float
    u_attn: float
    u_total: float
    w_lang: float = 1.0 / 3.0
    w_align: float = 1.0 / 3.0
    w_attn: float = 1.0 / 3.0
    reject: bool = False


# @dataclass
# class LangAlignGenerateOutput(ModelOutput):
#     """
#     Returned when ``lang_align_decode=True`` and ``return_dict_in_generate=True``.
#     """
#
#     sequences: Optional[torch.LongTensor] = None
#     sequences_no_image: Optional[torch.LongTensor] = None
#     generated_tokens: Optional[torch.LongTensor] = None
#     scores: Optional[Tuple[torch.FloatTensor, ...]] = None
#     logits: Optional[Tuple[torch.FloatTensor, ...]] = None
#     logits_no_image: Optional[Tuple[torch.FloatTensor, ...]] = None
#     token_roles: Optional[TokenRoleSets] = None
#     per_step_uncertainty: Optional[List[StepUncertaintyMetrics]] = field(default_factory=list)
#     sequence_uncertainty: Optional[float] = 0.0

@dataclass
class LangAlignGenerateDecoderOnlyOutput(ModelOutput):
    """
    Returned when ``lang_align_decode=True`` and ``return_dict_in_generate=True``.

    Includes generation outputs plus token roles and uncertainty decomposition
    (``uncertainty.u_total = w_lang·U_lang + w_align·U_align + w_attn·U_attn``).
    """

    sequences: torch.LongTensor = None
    sequences_no_image: torch.LongTensor = None
    # generated_tokens_image: torch.LongTensor = None
    # generated_tokens_no_image: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    scores_no_image: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits_no_image: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    generated_tokens: Optional[torch.LongTensor] = None
    token_roles: Optional[TokenRoleSets] = None
    uncertainty: Optional[UncertaintyBreakdown] = None
    per_step_uncertainty: Optional[List[StepUncertaintyMetrics]] = None
    sequence_uncertainty: Optional[float] = None

@dataclass
class LangAlignGenerateEncoderDecoderOutput(ModelOutput):
    """
    Returned when ``lang_align_decode=True`` and ``return_dict_in_generate=True``.

    Includes generation outputs plus token roles and uncertainty decomposition
    (``uncertainty.u_total = w_lang·U_lang + w_align·U_align + w_attn·U_attn``).
    """

    sequences: torch.LongTensor = None
    sequences_no_image: torch.LongTensor = None
    # generated_tokens_image: torch.LongTensor = None
    # generated_tokens_no_image: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    scores_no_image: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    logits_no_image: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[tuple[tuple[torch.FloatTensor]]] = None
    encoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Cache] = None
    generated_tokens: Optional[torch.LongTensor] = None
    token_roles: Optional[TokenRoleSets] = None
    uncertainty: Optional[UncertaintyBreakdown] = None
    per_step_uncertainty: Optional[List[StepUncertaintyMetrics]] = None
    sequence_uncertainty: Optional[float] = None
# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def select_visual_active_layers(
    attentions: Sequence[torch.Tensor],
    vision_indices: torch.LongTensor,
    *,
    active_ratio: float = 0.35,
) -> Tuple[int, ...]:
    """Select layers with high average intra-visual attention reception."""
    if vision_indices.numel() < 2:
        return tuple(range(len(attentions)))

    layer_means: List[float] = []
    layer_maxes: List[float] = []
    for attn in attentions:
        mat = attn[0].float().mean(dim=0)
        block = mat.index_select(0, vision_indices).index_select(1, vision_indices)
        eye = torch.eye(block.shape[0], dtype=torch.bool, device=block.device)
        if block.shape[0] > 1:
            block = block.masked_fill(eye, 0.0)
            denom = max(block.shape[0] - 1, 1)
            scores = block.sum(dim=0) / denom
        else:
            scores = block.mean(dim=0)
        layer_means.append(float(scores.mean()))
        layer_maxes.append(float(scores.max()))
    if not layer_means or max(layer_maxes) <= 0:
        return ()
    thresh = active_ratio * max(layer_maxes)
    selected = tuple(i for i, m in enumerate(layer_means) if m >= thresh)
    if selected:
        return selected
    return (int(torch.tensor(layer_means).argmax().item()),)


def _num_vision_tokens_from_grid(
    image_grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int = 2,
) -> Optional[int]:
    if image_grid_thw is None or not torch.is_tensor(image_grid_thw) or image_grid_thw.numel() == 0:
        return None
    grid = image_grid_thw[0]
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    h_m = max(h // spatial_merge_size, 1)
    w_m = max(w // spatial_merge_size, 1)
    return t * h_m * w_m


def resolve_token_regions(
    input_ids: torch.LongTensor,
    seq_len: int,
    *,
    image_token_id: Optional[int] = None,
    vision_start: Optional[int] = None,
    vision_end: Optional[int] = None,
    vision_token_indices: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Return (vision_indices, text_indices) in ``[0, seq_len)``.

    Priority: ``vision_token_indices`` > ``vision_start``/``vision_end`` >
    ``image_token_id`` + ``image_grid_thw`` > ``image_token_id`` contiguous block.
    """
    device = input_ids.device
    if vision_token_indices is not None and vision_token_indices.numel() > 0:
        v_idx = vision_token_indices.to(device=device, dtype=torch.long)
    elif vision_start is not None and vision_end is not None:
        v_idx = torch.arange(int(vision_start), int(vision_end), device=device, dtype=torch.long)
    elif image_token_id is not None:
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        placeholders = torch.nonzero(row == image_token_id, as_tuple=False).squeeze(-1)
        n_vis = _num_vision_tokens_from_grid(image_grid_thw, spatial_merge_size)
        if placeholders.numel() >= 1 and n_vis is not None:
            start = int(placeholders[0].item())
            end = min(start + n_vis, seq_len)
            v_idx = torch.arange(start, end, device=device, dtype=torch.long)
        elif placeholders.numel() >= 2:
            v_idx = placeholders
        elif placeholders.numel() == 1:
            pos = int(placeholders.item())
            v_idx = torch.arange(pos, seq_len, device=device, dtype=torch.long)
        else:
            v_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        v_idx = torch.empty(0, dtype=torch.long, device=device)

    all_idx = torch.arange(seq_len, device=device, dtype=torch.long)
    if v_idx.numel() > 0:
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[v_idx] = False
        t_idx = all_idx[mask]
    else:
        t_idx = all_idx
    return v_idx, t_idx


def classify_tokens_from_attention(
    attentions: Sequence[torch.Tensor],
    input_ids: torch.LongTensor,
    *,
    image_token_id: Optional[int] = None,
    vision_start: Optional[int] = None,
    vision_end: Optional[int] = None,
    vision_token_indices: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
) -> TokenRoleSets:
    """Resolve prefill text/vision regions and attention summary layers."""
    seq_len = attentions[0].shape[-1]
    vision_idx, text_idx = resolve_token_regions(
        input_ids,
        seq_len,
        image_token_id=image_token_id,
        vision_start=vision_start,
        vision_end=vision_end,
        vision_token_indices=vision_token_indices,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    active_layers = select_visual_active_layers(attentions, vision_idx)

    return TokenRoleSets(
        vision_indices=vision_idx,
        text_indices=text_idx,
        visual_active_layers=active_layers,
    )


# ---------------------------------------------------------------------------
# Uncertainty metrics  (U_total = w1·U_lang + w2·U_align + w3·U_attn)
# ---------------------------------------------------------------------------


def _text_to_vision_profile(
    attentions: Sequence[torch.Tensor],
    text_indices: torch.LongTensor,
    vision_indices: torch.LongTensor,
    layer_indices: Sequence[int],
) -> torch.Tensor:
    """
    A_vis(i): mean attention from all text queries to vision key i.

    Averaged over selected layers (heads mean inside each layer).
    Returns shape (|V|,) aligned with ``vision_indices`` order.
    """
    if text_indices.numel() == 0 or vision_indices.numel() == 0 or not layer_indices:
        dev = attentions[0].device if attentions else "cpu"
        return torch.zeros(vision_indices.numel(), device=dev)

    profile = torch.zeros(vision_indices.numel(), device=attentions[0].device, dtype=torch.float32)
    for li in layer_indices:
        attn = attentions[li][0].float().mean(dim=0)  # (seq, seq)
        block = attn[text_indices][:, vision_indices]  # (|T|, |V|)
        profile += block.mean(dim=0)
    profile /= len(layer_indices)
    return profile


def compute_u_lang(
    step_logits: Sequence[torch.Tensor],
    token_ids: Sequence[int],
    eps: float = 1e-12,
) -> Tuple[float, List[float]]:
    """
    U_lang = -(1/T) Σ_t log p(y_t | x_img, x_text, y_<t)

    Returns (sequence mean NLL, per-step NLL list).
    """
    if not token_ids:
        return 0.0, []
    per_step: List[float] = []
    for i, tid in enumerate(token_ids):
        if i >= len(step_logits):
            break
        logits_i = step_logits[i]
        if logits_i.dim() == 3:
            logits_i = logits_i[:, -1, :]
        prob = nn.functional.softmax(logits_i.float(), dim=-1)[0, int(tid)].clamp_min(eps)
        per_step.append(float(-torch.log(prob)))
    if not per_step:
        return 0.0, []
    return float(sum(per_step) / len(per_step)), per_step


def compute_u_align(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
) -> float:
    """
    U_align = 1 - concentration of text-to-vision attention.

    The concentration is the mass captured by the top quartile of vision tokens
    under the average text-to-vision attention profile.
    """
    vision = token_roles.vision_indices
    text = token_roles.text_indices
    if vision.numel() == 0 or text.numel() == 0:
        return 1.0

    layers = token_roles.visual_active_layers or tuple(range(len(attentions)))
    a_vis = _text_to_vision_profile(attentions, text, vision, layers)
    denom = a_vis.sum().clamp_min(1e-12)

    k = max(1, int(vision.numel() * 0.25))
    numer = torch.topk(a_vis, k).values.sum()
    concentration = (numer / denom).clamp(0.0, 1.0)
    return float(1.0 - concentration)


def compute_u_attn(
    attentions: Sequence[torch.Tensor],
    token_roles: TokenRoleSets,
) -> float:
    """Entropy of the average text-to-vision attention profile."""
    vision = token_roles.vision_indices
    text = token_roles.text_indices
    if vision.numel() == 0 or text.numel() == 0:
        return 0.0

    layers = token_roles.visual_active_layers or tuple(range(len(attentions)))
    profile = _text_to_vision_profile(attentions, text, vision, layers)
    prob = profile / profile.sum().clamp_min(1e-12)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
    max_entropy = torch.log(torch.tensor(float(max(int(prob.numel()), 1)), device=prob.device))
    if float(max_entropy) <= 0.0:
        return 0.0
    return float((entropy / max_entropy).clamp(0.0, 1.0))


def compute_total_uncertainty(
    u_lang: float,
    u_align: float,
    u_attn: float,
    *,
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
) -> UncertaintyBreakdown:
    """Weighted fusion and optional reject flag."""
    w = {"u_lang": 1.0 / 3.0, "u_align": 1.0 / 3.0, "u_attn": 1.0 / 3.0}
    if weights:
        w["u_lang"] = weights.get("u_lang", weights.get("w_lang", w["u_lang"]))
        w["u_align"] = weights.get("u_align", weights.get("w_align", w["u_align"]))
        w["u_attn"] = weights.get("u_attn", weights.get("w_attn", w["u_attn"]))

    u_total = w["u_lang"] * u_lang + w["u_align"] * u_align + w["u_attn"] * u_attn
    reject = u_total >= threshold if threshold is not None else False
    return UncertaintyBreakdown(
        u_lang=u_lang,
        u_align=u_align,
        u_attn=u_attn,
        u_total=u_total,
        w_lang=w["u_lang"],
        w_align=w["u_align"],
        w_attn=w["u_attn"],
        reject=reject,
    )


def compute_uncertainty_metrics(
    attentions: Optional[Sequence[torch.Tensor]],
    token_roles: Optional[TokenRoleSets],
    step_logits: Optional[Sequence[torch.Tensor]],
    token_ids: Sequence[int],
    *,
    weights: Optional[Dict[str, float]] = None,
    threshold: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[UncertaintyBreakdown, List[StepUncertaintyMetrics]]:
    """
    Full inference-time pipeline: U_lang + U_align + U_attn → U_total.

    U_align / U_attn use prefill attentions and resolved token regions;
    U_lang uses per-step generation logits.
    """
    u_lang, per_step_nll = compute_u_lang(step_logits or (), token_ids, eps=eps)
    u_align = compute_u_align(attentions, token_roles) if attentions and token_roles else 1.0
    u_attn = compute_u_attn(attentions, token_roles) if attentions and token_roles else 0.0

    breakdown = compute_total_uncertainty(
        u_lang, u_align, u_attn, weights=weights, threshold=threshold
    )

    w = breakdown
    per_step: List[StepUncertaintyMetrics] = []
    for i, tid in enumerate(token_ids):
        if i >= len(per_step_nll):
            break
        u_l = per_step_nll[i]
        u_t = w.w_lang * u_l + w.w_align * u_align + w.w_attn * u_attn
        per_step.append(
            StepUncertaintyMetrics(
                token_id=int(tid),
                u_lang=u_l,
                u_align=u_align,
                u_attn=u_attn,
                u_total=u_t,
            )
        )
    return breakdown, per_step


def aggregate_sequence_uncertainty(
    per_step: Sequence[StepUncertaintyMetrics],
    *,
    skip_token_ids: Optional[torch.Tensor] = None,
    generated_ids: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> float:
    """Aggregate per-step U_total; default mean over non-special tokens."""
    if not per_step:
        return 0.0
    values: List[float] = []
    for i, step in enumerate(per_step):
        if skip_token_ids is not None and generated_ids is not None:
            tid = int(generated_ids[i].item()) if generated_ids.dim() == 1 else int(generated_ids[0, i])
            if torch.isin(torch.tensor(tid), skip_token_ids):
                continue
        values.append(step.u_total)
    if not values:
        values = [s.u_total for s in per_step]
    if reduction == "max":
        return float(max(values))
    return float(sum(values) / len(values))


# ---------------------------------------------------------------------------
# Sampling loop
# ---------------------------------------------------------------------------


def _resolve_blank_pixels(model_kwargs: dict) -> Optional[torch.Tensor]:
    blank = model_kwargs.get("pixel_values_blank")
    if blank is not None:
        return blank
    pv = model_kwargs.get("pixel_values")
    if pv is None:
        return None
    return pv * 0


def _sample_one_chain(
    logits_processor: LogitsProcessorList,
    input_ids: torch.LongTensor,
    next_token_logits: torch.FloatTensor,
    do_sample: bool,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    next_token_scores = logits_processor(input_ids, next_token_logits)
    if do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)
    return next_token_scores, next_tokens


def _lang_align_attn_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[LangAlignGenerateDecoderOnlyOutput, GenerateNonBeamOutput, torch.LongTensor]:
    """
    Image-chain generation with prefill attention analysis and optional no-image chain
    for visual-evidence uncertainty. Distinct from ``vcd_sample._dual_chain_sample``.
    """
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
    do_sample = generation_config.do_sample

    use_no_image = model_kwargs.get(
        "lang_align_use_no_image",
        getattr(generation_config, "lang_align_use_no_image", True),
    )
    pixel_values_blank = _resolve_blank_pixels(model_kwargs)

    image_token_id = model_kwargs.get("image_token_id")
    if image_token_id is None and hasattr(self, "config"):
        image_token_id = getattr(self.config, "image_token_id", None)

    vision_start = model_kwargs.get("vision_start")
    vision_end = model_kwargs.get("vision_end")
    vision_token_indices = model_kwargs.get("vision_token_indices")
    image_grid_thw = model_kwargs.get("image_grid_thw")
    spatial_merge_size = int(model_kwargs.get("spatial_merge_size", 2))
    uncertainty_weights = model_kwargs.get(
        "uncertainty_weights",
        getattr(generation_config, "lang_align_uncertainty_weights", None),
    )
    uncertainty_threshold = model_kwargs.get("uncertainty_threshold")
    blur_epsilon = float(model_kwargs.get("blur_epsilon", getattr(generation_config, "lang_align_blur_epsilon", 1e-12)))
    collect_step_attentions = bool(
        model_kwargs.get(
            "collect_step_attentions",
            getattr(generation_config, "lang_align_collect_step_attentions", False),
        )
    )
    store_attentions = bool(output_attentions or collect_step_attentions)

    batch_size, cur_len = input_ids.shape[:2]
    prompt_len = cur_len
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and store_attentions) else None
    cross_attentions = () if (return_dict_in_generate and store_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
    logits_img = () if (return_dict_in_generate and output_logits) else None
    logits_blank = () if (return_dict_in_generate and output_logits and use_no_image) else None

    generated: List[torch.Tensor] = []
    token_roles: Optional[TokenRoleSets] = None
    prefill_attentions: Optional[Tuple[torch.Tensor, ...]] = None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if store_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    model_kwargs_img = self._get_initial_cache_position(cur_len, input_ids.device, dict(model_kwargs))
    model_kwargs_img["past_key_values"] = None
    model_kwargs_img["output_hidden_states"] =True
    is_prefill = True

    model_kwargs_blank = None
    is_prefill_blank = True
    if use_no_image and pixel_values_blank is not None:
        model_kwargs_blank = self._get_initial_cache_position(
            cur_len, input_ids.device, dict(model_kwargs)
        )
        model_kwargs_blank["past_key_values"] = None
        model_kwargs_blank["pixel_values_blank"] = pixel_values_blank

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        if self.config._attn_implementation == "flash_attention_2":
            if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                logger.warning_once(
                    "When using Flash Attention 2 and a static cache, you cannot use the option "
                    "`CompileConfig(fullgraph=True)` as FA2 introduces graph breaks. "
                    "We overrode the option with `fullgraph=False`."
                )
                generation_config.compile_config.fullgraph = False
        model_forward = self.get_compiled_call(generation_config.compile_config)

    input_ids_blank = input_ids.clone() if model_kwargs_blank is not None else None

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        need_attn = is_prefill or collect_step_attentions
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs_img)

        if is_prefill:
            outputs = self(
                **model_inputs,
                return_dict=True,
                # output_attentions=need_attn,
            )
            is_prefill = False
            if outputs.attentions is not None:
                prefill_attentions = outputs.attentions
                token_roles = classify_tokens_from_attention(
                    prefill_attentions,
                    input_ids,
                    image_token_id=image_token_id,
                    vision_start=vision_start,
                    vision_end=vision_end,
                    vision_token_indices=vision_token_indices,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=spatial_merge_size,
                )
        else:
            outputs = model_forward(**model_inputs, return_dict=True)#,output_attentions=need_attn)

        model_kwargs_img = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs_img,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )
        next_scores, next_tokens = _sample_one_chain(
            logits_processor, input_ids, next_logits, do_sample
        )

        next_logits_blank = None
        if model_kwargs_blank is not None:
            if not hasattr(self, "prepare_inputs_for_generation_cd"):
                raise AttributeError(
                    "lang_align_decode with no-image chain requires "
                    "model.prepare_inputs_for_generation_cd."
                )
            model_inputs_blank = self.prepare_inputs_for_generation_cd(# 专门用于空白图的推进
                input_ids, **model_kwargs_blank
            )
            if is_prefill_blank:
                outputs_blank = self(
                    **model_inputs_blank,
                    return_dict=True,
                    # output_attentions=False,
                )
                is_prefill_blank = False
            else:
                outputs_blank = model_forward(**model_inputs_blank, return_dict=True)

            next_logits_blank = outputs_blank.logits[:, -1, :].to(
                copy=True, dtype=torch.float32, device=input_ids.device
            )
            model_kwargs_blank = self._update_model_kwargs_for_generation(
                outputs_blank,
                model_kwargs_blank,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            del outputs_blank

        if return_dict_in_generate:
            if output_scores:
                scores += (next_scores,)
            if output_logits:
                logits_img += (next_logits,)
                if next_logits_blank is not None:
                    logits_blank += (next_logits_blank,)
            if store_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        generated.append(next_tokens)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if input_ids_blank is not None:
            _, next_blank = _sample_one_chain(
                logits_processor, input_ids_blank, next_logits_blank, do_sample
            )
            if has_eos_stopping_criteria:
                next_blank = next_blank * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            input_ids_blank = torch.cat([input_ids_blank, next_blank[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        del outputs

    if streamer is not None:
        streamer.end()

    gen_tokens = torch.stack(generated, dim=1) if generated else input_ids[:, prompt_len:]
    token_ids = [int(t) for t in gen_tokens[0].tolist()] if gen_tokens.numel() > 0 else []

    uncertainty_breakdown: Optional[UncertaintyBreakdown] = None
    per_step_uncertainty: List[StepUncertaintyMetrics] = []
    seq_u = 0.0
    if return_dict_in_generate and token_ids:
        uncertainty_breakdown, per_step_uncertainty = compute_uncertainty_metrics(
            prefill_attentions,
            token_roles,
            logits_img,
            token_ids,
            weights=uncertainty_weights,
            threshold=uncertainty_threshold,
            eps=blur_epsilon,
        )
        seq_u = uncertainty_breakdown.u_total

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return LangAlignGenerateEncoderDecoderOutput(
                sequences=input_ids,
                sequences_no_image=input_ids_blank,
                generated_tokens=gen_tokens,
                scores=scores,
                logits=logits_img,
                logits_no_image=logits_blank,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                token_roles=token_roles,
                uncertainty=uncertainty_breakdown,
                per_step_uncertainty=per_step_uncertainty,
                sequence_uncertainty=seq_u,
                past_key_values =  model_kwargs.get("past_key_values"),
            )
        else:
            return LangAlignGenerateDecoderOnlyOutput(
                sequences=input_ids,
                sequences_no_image=input_ids_blank,
                generated_tokens=gen_tokens,
                scores=scores,
                logits=logits_img,
                logits_no_image=logits_blank,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                token_roles=token_roles,
                uncertainty=uncertainty_breakdown,
                per_step_uncertainty=per_step_uncertainty,
                sequence_uncertainty=seq_u,
                past_key_values =  model_kwargs.get("past_key_values"),
            )
    return input_ids


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    # ``GenerationMixin._sample`` is patched globally, not only on Qwen-VL.
    # Defaulting this flag to True hijacked every subsequently loaded HF model,
    # including FactualSceneGraph's encoder-decoder Flan-T5.  Only the
    # LangAlign runner is allowed to opt in.
    lang_align = model_kwargs.pop(
        "lang_align_decode",
        getattr(generation_config, "lang_align_decode", False),
    )
    if lang_align:
        if model_kwargs.get("pixel_values") is None:
            warnings.warn(
                "lang_align_decode without pixel_values: visual attention analysis may be limited.",
                UserWarning,
            )
        # 用于返回正常的结果
        generation_config.output_scores = True
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.output_hidden_states = True
        generation_config.output_attentions = True
        return _lang_align_attn_sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    if _ORIGINAL_SAMPLE is None:
        raise RuntimeError("Call evolve_lang_align_attn_sampling() before generate().")
    return _ORIGINAL_SAMPLE(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )


def evolve_lang_align_attn_sampling() -> None:
    """Patch ``GenerationMixin._sample`` with language-alignment attention sampling."""
    global _ORIGINAL_SAMPLE
    from transformers.generation.utils import GenerationMixin
    if _ORIGINAL_SAMPLE is None:
        if hasattr(GenerationMixin, "_sample"):
            _ORIGINAL_SAMPLE = GenerationMixin._sample
        elif hasattr(GenerationMixin, "sample"):
            _ORIGINAL_SAMPLE = GenerationMixin.sample
        else:
            raise RuntimeError(
                "Could not find GenerationMixin._sample or GenerationMixin.sample to patch."
            )

    if hasattr(GenerationMixin, "_sample"):
        GenerationMixin._sample = _sample
        transformers.generation.utils.GenerationMixin._sample = _sample
    if hasattr(GenerationMixin, "sample"):
        GenerationMixin.sample = _sample
        transformers.generation.utils.GenerationMixin.sample = _sample
