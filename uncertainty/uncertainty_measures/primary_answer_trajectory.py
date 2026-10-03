"""Capture causal primary-answer trajectories during the shared probe forward.

ICR keeps its paper's answer-input query convention. CoE/logit-lens features
use the preceding query that predicts each answer token. No answer is sampled
or re-tokenized here, and full vocabulary distributions are not persisted.
"""
from contextlib import AbstractContextManager
import re

import torch

from uncertainty.uncertainty_measures.decision_commitment import (
    compute_decision_commitment_features,
)
from uncertainty.uncertainty_measures.internal_entropy import (
    head_output_energy_entropy_torch,
)
from uncertainty.uncertainty_measures.internal_trajectory import sivr_internal_features
from uncertainty.uncertainty_measures.paper_probe_features import (
    find_language_attention_output_projections,
)


class PrimaryAnswerInternalCapture(AbstractContextManager):
    """Qwen3-VL post-block states and prompt-final pre-o_proj head outputs."""

    def __init__(self, model, *, prompt_length: int, response_end: int):
        self.start = int(prompt_length) - 1
        self.end = int(response_end) - 1
        if not 0 <= self.start < self.end:
            raise ValueError("Primary-answer predictive queries must be non-empty.")
        self.specs = find_language_attention_output_projections(model)
        modules = dict(model.named_modules())
        self.blocks = []
        self.layer_ids = []
        stack_names = set()
        for spec in self.specs:
            block_name = spec.name.removesuffix(".self_attn.o_proj")
            match = re.fullmatch(r"(.+)\.layers\.(\d+)", block_name)
            if match is None or block_name not in modules:
                raise ValueError(f"Cannot resolve Qwen decoder block: {spec.name}")
            stack_names.add(match.group(1))
            self.layer_ids.append(int(match.group(2)))
            self.blocks.append(modules[block_name])
        config = getattr(model.config, "text_config", model.config)
        expected = int(getattr(config, "num_hidden_layers", len(self.blocks)))
        if len(stack_names) != 1 or self.layer_ids != list(range(expected)):
            raise ValueError("Primary trajectories require every Qwen3-VL decoder layer.")
        self.final_norm = modules.get(f"{next(iter(stack_names))}.norm")
        self.lm_head = model.get_output_embeddings()
        if self.final_norm is None or self.lm_head is None:
            raise ValueError("Cannot resolve language final norm and lm_head.")
        self.hidden = [None] * len(self.blocks)
        self.head_entropy = [None] * len(self.blocks)
        self.answer_free_heads = [None] * len(self.blocks)
        self.handles = []

    def _span(self, value):
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < self.end:
            raise ValueError("Expected batch-one states covering all predictive queries.")
        return value[0, self.start:self.end].detach().to(
            device="cpu", dtype=torch.float32, copy=True,
        )

    def _hidden_hook(self, index):
        def capture(_module, _args, output):
            if self.hidden[index] is not None:
                raise RuntimeError("A decoder block was called twice in one probe forward.")
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.hidden[index] = self._span(value)
        return capture

    def _head_hook(self, index):
        def capture(_module, args):
            if self.head_entropy[index] is not None:
                raise RuntimeError("An attention projection was called twice.")
            selected = self._span(args[0])
            heads = self.specs[index].num_heads
            if selected.shape[-1] % heads:
                raise ValueError("Attention output width is not divisible by head count.")
            selected = selected.reshape(selected.shape[0], heads, -1)
            # The causal prompt-final row cannot attend to the supplied answer.
            self.answer_free_heads[index] = selected[0].to(torch.float16)
            self.head_entropy[index] = head_output_energy_entropy_torch(selected)
        return capture

    def __enter__(self):
        try:
            for index, (block, spec) in enumerate(zip(self.blocks, self.specs)):
                self.handles.append(block.register_forward_hook(self._hidden_hook(index)))
                self.handles.append(spec.module.register_forward_pre_hook(self._head_hook(index)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def features(self, answer_token_ids):
        if any(value is None for value in (*self.hidden, *self.head_entropy, *self.answer_free_heads)):
            raise RuntimeError("Incomplete primary-answer internal captures.")
        hidden = torch.stack(self.hidden, dim=1)
        trajectory = compute_decision_commitment_features(
            hidden,
            answer_token_ids,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            layer_ids=self.layer_ids,
            head_energy_entropy=torch.stack(self.head_entropy),
            head_layer_ids=self.layer_ids,
        )
        # Save the sequence classifier's sufficient features before discarding
        # token-by-layer hidden states; no additional model forward is needed.
        trajectory["sivr"] = sivr_internal_features(hidden.numpy())
        trajectory["sivr"]["output_entropy_per_token"] = (
            trajectory["vocab_entropy"][-1].float().tolist()
        )
        trajectory.update({
            "answer_token_ids": torch.as_tensor(answer_token_ids).cpu().tolist(),
            "answer_token_count": self.end - self.start,
            "predictive_query_start": self.start,
            "predictive_query_end": self.end,
            "hidden_capture": "post_decoder_block_predictive_answer_queries_v1",
            "collection": "teacher_forced_base_model_on_exact_generated_tokens",
            "logit_lens_temperature": 1.0,
        })
        return {
            "primary_answer_trajectory": trajectory,
            "answer_free_head_outputs": torch.stack(self.answer_free_heads),
            "answer_free_model_layer_ids": torch.tensor(self.layer_ids, dtype=torch.long),
            "answer_free_query_index": self.start,
            "answer_free_feature_position": "prompt_final_query_pre_o_proj",
        }
