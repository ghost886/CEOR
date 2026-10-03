"""Internal feature extraction shared by VIB-Probe and LRP.

The two papers use different Transformer internals:

* VIB-Probe consumes the output of every attention head *before* the
  output projection mixes heads.
* Latent Representation Probing (LRP) consumes output-token hidden states
  and attention from output queries to input keys.

This module keeps those definitions explicit and contains no supervised
training code.  In particular, attention weights are never substituted for
the pre-projection head outputs required by VIB-Probe.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


VIB_FEATURE_NAME = "vib_attention_head_outputs"
LRP_HIDDEN_FEATURE_NAME = "lrp_output_hidden_states"
LRP_ATTENTION_FEATURE_NAME = "lrp_attention_patterns"
LRP_VISUAL_ATTENTION_FEATURE_NAME = "lrp_visual_attention"
PAPER_PROBE_FEATURE_VERSION = 1


def _natural_key(value: str) -> Tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
    )


def _unwrap_step_layers(step: Any, name: str) -> Tuple[torch.Tensor, ...]:
    """Normalize the nesting returned by vanilla and patched ``generate``."""
    while (
        isinstance(step, (tuple, list))
        and len(step) == 1
        and isinstance(step[0], (tuple, list))
    ):
        step = step[0]
    if not isinstance(step, (tuple, list)) or not step:
        raise ValueError(f"Each {name} generation step must contain layer tensors.")
    if not all(torch.is_tensor(value) for value in step):
        raise ValueError(f"Each {name} layer value must be a tensor.")
    return tuple(step)


@dataclass(frozen=True)
class AttentionProjectionSpec:
    """One language-model attention output projection."""

    name: str
    module: torch.nn.Module
    num_heads: int


def find_language_attention_output_projections(
    model: torch.nn.Module,
) -> Tuple[AttentionProjectionSpec, ...]:
    """Find decoder ``self_attn.o_proj`` modules in layer order.

    Qwen model wrappers changed their nesting several times across
    Transformers releases.  Selecting by the semantic module suffix is more
    stable than depending on one concrete ``model.language_model`` path.  A
    vision tower normally uses ``attn.proj`` rather than ``self_attn.o_proj``;
    names containing an explicit vision component are nevertheless excluded.
    """
    modules = dict(model.named_modules())
    root_config = getattr(model, "config", None)
    text_config = getattr(root_config, "text_config", None)
    default_num_heads = getattr(text_config, "num_attention_heads", None)
    if default_num_heads is None:
        default_num_heads = getattr(root_config, "num_attention_heads", None)
    candidates: List[AttentionProjectionSpec] = []
    for name, module in modules.items():
        if not name.endswith("self_attn.o_proj"):
            continue
        lower = name.lower()
        if any(part in lower for part in ("visual.blocks", "vision_model", "vision_tower")):
            continue
        parent_name = name[: -len(".o_proj")]
        parent = modules.get(parent_name)
        num_heads = getattr(parent, "num_heads", None)
        if num_heads is None:
            num_heads = getattr(parent, "num_attention_heads", None)
        if num_heads is None:
            config = getattr(parent, "config", None)
            num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            head_dim = getattr(parent, "head_dim", None)
            input_width = getattr(module, "in_features", None)
            if (
                isinstance(head_dim, int)
                and head_dim > 0
                and isinstance(input_width, int)
                and input_width % head_dim == 0
            ):
                num_heads = input_width // head_dim
        if num_heads is None:
            num_heads = default_num_heads
        if num_heads is None:
            raise ValueError(
                f"Cannot determine the number of heads for attention module `{parent_name}`."
            )
        candidates.append(
            AttentionProjectionSpec(name=name, module=module, num_heads=int(num_heads))
        )
    candidates.sort(key=lambda item: _natural_key(item.name))
    if not candidates:
        raise ValueError(
            "No language self-attention output projections were found. "
            "VIB-Probe currently requires modules named `self_attn.o_proj`."
        )
    return tuple(candidates)


class AttentionHeadOutputCapture(AbstractContextManager):
    """Capture the final query's pre-``o_proj`` head outputs during generation.

    Each language layer is called once per decoding step.  The hook overwrites
    its previous value and stores only the latest final-query row, so memory is
    ``O(layers * hidden_size)``.  Captures are detached by default; live VIB
    attribution uses :class:`AttentionHeadOutputIntervention` below.
    """

    def __init__(self, model: torch.nn.Module, *, detach: bool = True):
        self.specs = find_language_attention_output_projections(model)
        self.detach = bool(detach)
        self.records: List[Optional[torch.Tensor]] = [None for _ in self.specs]
        self._handles: List[Any] = []

    def _hook(self, layer_index: int):
        def capture(_module, args):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Attention o_proj received no tensor input.")
            value = args[0]
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError(
                    "VIB feature collection currently requires a batch-one "
                    "[batch, query, hidden] attention output."
                )
            value = value[:, -1, :]
            if self.detach:
                value = value.detach()
            self.records[layer_index] = value
            return None

        return capture

    def __enter__(self):
        self._handles = [
            spec.module.register_forward_pre_hook(self._hook(layer_index))
            for layer_index, spec in enumerate(self.specs)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def final_tensor(self) -> torch.Tensor:
        """Return the last decoding step as ``[layers, heads, head_dim]``."""
        missing = [
            index for index, value in enumerate(self.records) if value is None
        ]
        if missing:
            raise RuntimeError(f"Missing VIB attention captures for layers {missing}.")
        layers = []
        for spec, value in zip(self.specs, self.records):
            if value is None:  # guarded above; keeps static type checkers happy
                raise RuntimeError("Missing VIB attention capture.")
            if value.shape[-1] % spec.num_heads != 0:
                raise ValueError(
                    f"o_proj input width {value.shape[-1]} is not divisible by "
                    f"{spec.num_heads} heads in `{spec.name}`."
                )
            layers.append(
                value.reshape(1, spec.num_heads, -1)
                .squeeze(0)
                .detach()
                .float()
                .cpu()
            )
        shapes = {tuple(value.shape) for value in layers}
        if len(shapes) != 1:
            raise ValueError(f"VIB head-output layer shapes differ: {sorted(shapes)}")
        return torch.stack(layers, dim=0)


class AttentionHeadOutputTrajectoryCapture(AbstractContextManager):
    """Capture every generated query's pre-``o_proj`` head output.

    Unlike :class:`AttentionHeadOutputCapture`, this collector keeps one row
    per autoregressive generation step.  The resulting token axis aligns with
    ``generate(...).sequences[:, prompt_length:]``: the final prompt query
    produces generated token zero, and each cached one-token query produces
    the next generated token.  This is intended for coordinate-semantic
    pooling in training-free head-dispersion experiments.
    """

    def __init__(self, model: torch.nn.Module, *, detach: bool = True):
        self.specs = find_language_attention_output_projections(model)
        self.detach = bool(detach)
        self.records: List[List[torch.Tensor]] = [[] for _ in self.specs]
        self._handles: List[Any] = []

    def _hook(self, layer_index: int):
        def capture(_module, args):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Attention o_proj received no tensor input.")
            value = args[0]
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError(
                    "Head-trajectory collection requires batch-one "
                    "[batch, query, hidden] attention output."
                )
            value = value[:, -1, :]
            if self.detach:
                value = value.detach()
            self.records[layer_index].append(value)
            return None

        return capture

    def __enter__(self):
        self._handles = [
            spec.module.register_forward_pre_hook(self._hook(layer_index))
            for layer_index, spec in enumerate(self.specs)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def trajectory_tensor(self) -> torch.Tensor:
        """Return ``[generated_steps, layers, heads, head_dim]`` on CPU."""
        missing = [index for index, values in enumerate(self.records) if not values]
        if missing:
            raise RuntimeError(
                f"Missing head-trajectory captures for layers {missing}."
            )
        step_counts = {len(values) for values in self.records}
        if len(step_counts) != 1:
            raise RuntimeError(
                "Decoder layers produced different trajectory lengths: "
                f"{sorted(step_counts)}."
            )
        layers = []
        for spec, values in zip(self.specs, self.records):
            layer = torch.cat(values, dim=0)
            if layer.shape[-1] % spec.num_heads != 0:
                raise ValueError(
                    f"o_proj input width {layer.shape[-1]} is not divisible by "
                    f"{spec.num_heads} heads in `{spec.name}`."
                )
            layers.append(
                layer.reshape(layer.shape[0], spec.num_heads, -1)
                .detach()
                .float()
                .cpu()
            )
        shapes = {tuple(value.shape) for value in layers}
        if len(shapes) != 1:
            raise ValueError(
                f"Head-trajectory layer shapes differ: {sorted(shapes)}"
            )
        return torch.stack(layers, dim=1)


class AttentionHeadOutputIntervention(AbstractContextManager):
    """Capture differentiable head outputs and optionally scale selected heads."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        scales: Optional[torch.Tensor] = None,
    ):
        self.specs = find_language_attention_output_projections(model)
        if scales is not None:
            scales = torch.as_tensor(scales).detach().float().cpu()
            expected = (len(self.specs), self.specs[0].num_heads)
            if tuple(scales.shape) != expected:
                raise ValueError(
                    f"Head scales must have shape {expected}, got {tuple(scales.shape)}."
                )
        self.scales = scales
        self.records: List[Optional[torch.Tensor]] = [None for _ in self.specs]
        self._handles: List[Any] = []

    def _hook(self, layer_index: int):
        def capture_and_scale(_module, args):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Attention o_proj received no tensor input.")
            value = args[0]
            spec = self.specs[layer_index]
            if value.ndim != 3 or value.shape[-1] % spec.num_heads != 0:
                raise ValueError("Unexpected pre-o_proj attention output shape.")
            view = value.reshape(
                value.shape[0], value.shape[1], spec.num_heads, -1
            )
            self.records[layer_index] = view[:, -1, :, :]
            if self.scales is None:
                return None
            layer_scales = self.scales[layer_index].to(
                device=value.device, dtype=value.dtype
            )
            # Only the current decoding query is edited, as in the paper's
            # single-step intervention.  Earlier prefix states remain intact.
            prefix = view[:, :-1, :, :]
            current = view[:, -1:, :, :] * layer_scales.view(1, 1, -1, 1)
            edited = torch.cat((prefix, current), dim=1).reshape_as(value)
            return (edited, *args[1:])

        return capture_and_scale

    def __enter__(self):
        self._handles = [
            spec.module.register_forward_pre_hook(self._hook(layer_index))
            for layer_index, spec in enumerate(self.specs)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        missing = [index for index, value in enumerate(self.records) if value is None]
        if missing:
            raise RuntimeError(f"Missing differentiable head outputs for layers {missing}.")
        values = [value for value in self.records if value is not None]
        shapes = {tuple(value.shape[1:]) for value in values}
        if len(shapes) != 1:
            raise ValueError(f"VIB live head shapes differ: {sorted(shapes)}")
        if device is None:
            device = values[0].device
        return torch.stack([value.to(device=device, dtype=torch.float32) for value in values], dim=1)


class AttentionHeadAdditiveIntervention(AbstractContextManager):
    """Apply paper-style ITI shifts to current pre-``o_proj`` head outputs.

    ``directions`` has shape ``[layers, heads, head_dim]`` and ``scales`` has
    shape ``[layers, heads]``. Only selected heads in ``mask`` are shifted by
    ``alpha * scales * directions``. On the initial prompt forward and every
    cached decoding forward, only the final/current query is edited, matching
    the original honest_llama intervention.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        directions: torch.Tensor,
        scales: torch.Tensor,
        mask: torch.Tensor,
        alpha: float,
    ):
        self.specs = find_language_attention_output_projections(model)
        self.directions = torch.as_tensor(directions).detach().float().cpu()
        self.scales = torch.as_tensor(scales).detach().float().cpu()
        self.mask = torch.as_tensor(mask).detach().bool().cpu()
        self.alpha = float(alpha)
        expected_prefix = (len(self.specs), self.specs[0].num_heads)
        if tuple(self.directions.shape[:2]) != expected_prefix:
            raise ValueError(
                f"ITI directions must start with {expected_prefix}, got "
                f"{tuple(self.directions.shape)}."
            )
        if tuple(self.scales.shape) != expected_prefix or tuple(self.mask.shape) != expected_prefix:
            raise ValueError("ITI scales and mask must have shape [layers, heads].")
        if self.directions.ndim != 3:
            raise ValueError("ITI directions must be [layers, heads, head_dim].")
        self._handles: List[Any] = []

    def _hook(self, layer_index: int):
        def add_direction(_module, args):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Attention o_proj received no tensor input.")
            value = args[0]
            spec = self.specs[layer_index]
            if value.ndim != 3 or value.shape[-1] % spec.num_heads != 0:
                raise ValueError("Unexpected pre-o_proj attention output shape.")
            view = value.reshape(value.shape[0], value.shape[1], spec.num_heads, -1)
            if view.shape[-1] != self.directions.shape[-1]:
                raise ValueError(
                    f"ITI head_dim mismatch: model={view.shape[-1]}, "
                    f"artifact={self.directions.shape[-1]}."
                )
            direction = self.directions[layer_index].to(value.device, value.dtype)
            scale = self.scales[layer_index].to(value.device, value.dtype)
            selected = self.mask[layer_index].to(value.device, value.dtype)
            shift = self.alpha * direction * scale[:, None] * selected[:, None]
            # Avoid an in-place write into a view needed by autograd/model code.
            prefix = view[:, :-1]
            current = view[:, -1:] + shift.view(1, 1, spec.num_heads, -1)
            edited = torch.cat((prefix, current), dim=1).reshape_as(value)
            return (edited, *args[1:])

        return add_direction

    def __enter__(self):
        self._handles = [
            spec.module.register_forward_pre_hook(self._hook(layer_index))
            for layer_index, spec in enumerate(self.specs)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False


def _generation_step_count(
    hidden_states: Sequence[Any], generated_token_count: int
) -> int:
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ValueError("Generation output contains no hidden-state steps.")
    return min(len(hidden_states), int(generated_token_count))


def extract_lrp_generation_features(
    hidden_states: Sequence[Any],
    attentions: Sequence[Any],
    *,
    prompt_length: int,
    generated_token_count: int,
    vision_indices: Optional[torch.LongTensor] = None,
) -> Dict[str, Any]:
    """Extract all three representations defined by the LRP paper.

    ``hidden_states`` and ``attentions`` are the nested per-step caches from
    Hugging Face generation.  Each step's final query is the state used to emit
    the matching output token.  Attention keys are restricted to the original
    input sequence, matching Eq. (10)'s ``n`` input-token positions.
    """
    if not isinstance(attentions, (tuple, list)) or not attentions:
        raise ValueError(
            "LRP attention extraction requires generation with output_attentions=True."
        )
    steps = _generation_step_count(hidden_states, generated_token_count)
    steps = min(steps, len(attentions))
    if steps < 1:
        raise ValueError("LRP requires at least one generated token.")

    first_attention_layers = _unwrap_step_layers(attentions[0], "attention")
    num_layers = len(first_attention_layers)
    hidden_sums: List[Optional[torch.Tensor]] = [None] * num_layers
    attention_sums: List[Optional[torch.Tensor]] = [None] * num_layers

    for step_index in range(steps):
        hidden_layers = _unwrap_step_layers(hidden_states[step_index], "hidden-state")
        attention_layers = _unwrap_step_layers(attentions[step_index], "attention")
        if len(attention_layers) != num_layers:
            raise ValueError("LRP attention layer count changed across generation steps.")
        # HF returns embeddings plus one state per block.  Patched outputs may
        # contain only block outputs; selecting the last L works for both.
        if len(hidden_layers) < num_layers:
            raise ValueError("LRP hidden-state cache has fewer entries than attention layers.")
        hidden_layers = hidden_layers[-num_layers:]
        for layer_index, (hidden, attention) in enumerate(
            zip(hidden_layers, attention_layers)
        ):
            hidden_row = hidden[0, -1, :].detach().float().cpu()
            attention_row = attention[0, :, -1, :prompt_length].detach().float().cpu()
            if attention_row.shape[-1] != prompt_length:
                raise ValueError("LRP attention cache is shorter than the prompt.")
            hidden_sums[layer_index] = (
                hidden_row
                if hidden_sums[layer_index] is None
                else hidden_sums[layer_index] + hidden_row
            )
            attention_sums[layer_index] = (
                attention_row
                if attention_sums[layer_index] is None
                else attention_sums[layer_index] + attention_row
            )

    hidden = torch.stack(
        [value / steps for value in hidden_sums if value is not None], dim=0
    )
    attention = torch.stack(
        [value / steps for value in attention_sums if value is not None], dim=0
    )
    if hidden.shape[0] != num_layers or attention.shape[0] != num_layers:
        raise RuntimeError("LRP failed to collect every Transformer layer.")

    if vision_indices is None:
        vision_indices = torch.empty(0, dtype=torch.long)
    vision_indices = torch.as_tensor(vision_indices, dtype=torch.long).cpu()
    vision_indices = vision_indices[
        (vision_indices >= 0) & (vision_indices < prompt_length)
    ]
    if vision_indices.numel() > 0:
        visual_attention = attention.index_select(-1, vision_indices).mean(dim=-1)
        visual_available = True
    else:
        visual_attention = torch.zeros(
            attention.shape[:2], dtype=attention.dtype
        )
        visual_available = False

    return {
        LRP_HIDDEN_FEATURE_NAME: hidden.to(torch.float16),
        LRP_ATTENTION_FEATURE_NAME: attention.to(torch.float16),
        LRP_VISUAL_ATTENTION_FEATURE_NAME: visual_attention.to(torch.float32),
        "lrp_num_layers": int(num_layers),
        "lrp_num_heads": int(attention.shape[1]),
        "lrp_hidden_dim": int(hidden.shape[1]),
        "lrp_num_input_tokens": int(prompt_length),
        "lrp_num_output_tokens": int(steps),
        "lrp_num_visual_tokens": int(vision_indices.numel()),
        "lrp_visual_attention_available": bool(visual_available),
        "paper_probe_feature_version": PAPER_PROBE_FEATURE_VERSION,
    }


def extract_lrp_dense_features(
    hidden_states: Sequence[torch.Tensor],
    attentions: Sequence[torch.Tensor],
    *,
    response_start: int,
    response_end: int,
    prompt_length: int,
    vision_indices: Optional[torch.LongTensor] = None,
) -> Dict[str, Any]:
    """Extract Eqs. (9)-(10) from a complete teacher-forced sequence.

    Unlike a generation cache, this layout contains a hidden state and an
    attention query for the final emitted answer token.  It is therefore the
    canonical LRP extraction used by the Qwen integration.
    """
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise ValueError("Dense LRP extraction requires hidden states.")
    if not isinstance(attentions, (tuple, list)) or not attentions:
        raise ValueError("Dense LRP extraction requires attentions.")
    if not 0 <= int(response_start) < int(response_end):
        raise ValueError("LRP response span must contain at least one token.")
    if int(response_end) > int(hidden_states[-1].shape[1]):
        raise ValueError("LRP response span exceeds the dense sequence length.")
    num_layers = len(attentions)
    if len(hidden_states) < num_layers:
        raise ValueError("Dense hidden-state cache has fewer entries than layers.")
    block_hidden = hidden_states[-num_layers:]
    hidden = torch.stack(
        [
            value[0, response_start:response_end, :]
            .detach()
            .float()
            .cpu()
            .mean(dim=0)
            for value in block_hidden
        ],
        dim=0,
    )
    attention = torch.stack(
        [
            value[0, :, response_start:response_end, :prompt_length]
            .detach()
            .float()
            .cpu()
            .mean(dim=1)
            for value in attentions
        ],
        dim=0,
    )
    if attention.shape[-1] != int(prompt_length):
        raise ValueError("Dense LRP attention cache is shorter than the prompt.")

    if vision_indices is None:
        vision_indices = torch.empty(0, dtype=torch.long)
    vision_indices = torch.as_tensor(vision_indices, dtype=torch.long).cpu()
    vision_indices = vision_indices[
        (vision_indices >= 0) & (vision_indices < int(prompt_length))
    ]
    if vision_indices.numel() > 0:
        visual_attention = attention.index_select(-1, vision_indices).mean(dim=-1)
        visual_available = True
    else:
        visual_attention = torch.zeros(attention.shape[:2], dtype=attention.dtype)
        visual_available = False
    return {
        LRP_HIDDEN_FEATURE_NAME: hidden.to(torch.float16),
        LRP_ATTENTION_FEATURE_NAME: attention.to(torch.float16),
        LRP_VISUAL_ATTENTION_FEATURE_NAME: visual_attention.to(torch.float32),
        "lrp_num_layers": int(num_layers),
        "lrp_num_heads": int(attention.shape[1]),
        "lrp_hidden_dim": int(hidden.shape[1]),
        "lrp_num_input_tokens": int(prompt_length),
        "lrp_num_output_tokens": int(response_end - response_start),
        "lrp_num_visual_tokens": int(vision_indices.numel()),
        "lrp_visual_attention_available": bool(visual_available),
        "lrp_cache_format": "dense_teacher_forced",
        "paper_probe_feature_version": PAPER_PROBE_FEATURE_VERSION,
    }


def attach_vib_generation_feature(
    features: Mapping[str, Any],
    capture: AttentionHeadOutputCapture,
) -> Dict[str, Any]:
    """Return a copy of ``features`` with the paper-faithful VIB tensor."""
    result = dict(features)
    value = capture.final_tensor().detach().float().cpu()
    result[VIB_FEATURE_NAME] = value.to(torch.float16)
    result.update(
        {
            "vib_num_layers": int(value.shape[0]),
            "vib_num_heads": int(value.shape[1]),
            "vib_head_dim": int(value.shape[2]),
            "paper_probe_feature_version": PAPER_PROBE_FEATURE_VERSION,
        }
    )
    return result
