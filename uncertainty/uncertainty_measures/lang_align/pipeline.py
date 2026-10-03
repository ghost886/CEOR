from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch

from .blur import zero_pixel_baseline
from .config import LangAlignConfig
from .lang_align_attn_sample import (
    LangAlignGenerateDecoderOnlyOutput,
    aggregate_sequence_uncertainty,
    evolve_lang_align_attn_sampling,
)
from .scoring import resolve_skip_token_ids


# from uncertainty.uncertainty_measures.lang_align.blur import zero_pixel_baseline
# from uncertainty.uncertainty_measures.lang_align.config import LangAlignConfig
# from uncertainty.uncertainty_measures.lang_align.lang_align_attn_sample import (
#     LangAlignGenerateDecoderOnlyOutput,
#     aggregate_sequence_uncertainty,
#     evolve_lang_align_attn_sampling,
# )
# from uncertainty.uncertainty_measures.lang_align.scoring import resolve_skip_token_ids


class LangAlignUncertaintyRunner:
    """Run lang-align attention sampling and return uncertainty metrics per sample."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        config: Optional[LangAlignConfig] = None,
        blur_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *,
        lang_align_patched: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.config = config or LangAlignConfig()
        self.blur_fn = blur_fn or zero_pixel_baseline
        self._lang_align_patched = lang_align_patched

    def _ensure_lang_align_patch(self) -> None:
        if self._lang_align_patched:
            return
        evolve_lang_align_attn_sampling()
        self._lang_align_patched = True

    def build_messages(self, image: Any, question: str) -> List[dict]:
        content = [{"type": "text", "text": question}]
        if image is not None:
            content.append({"type": "image", "image": image})
        return [{"role": "user", "content": content}]

    def _build_gen_kwargs(self, cfg: LangAlignConfig) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_scores": True,
            "output_attentions": True,
            "output_hidden_states": True,
        }
        if cfg.generation_temperature and cfg.generation_temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = cfg.generation_temperature
        else:
            gen_kwargs["do_sample"] = False
        return gen_kwargs

    # 不要乱用这个函数
    # def _build_gen_kwargs(self, cfg: LangAlignConfig) -> Dict[str, Any]:
    #     gen_kwargs: Dict[str, Any] = {
    #         "max_new_tokens": cfg.max_new_tokens,
    #         "lang_align_decode": True,
    #         "lang_align_use_no_image": cfg.lang_align_use_no_image,
    #         "return_dict_in_generate": True,
    #         "output_logits": True,
    #         "output_scores": True,
    #         "blur_epsilon": cfg.blur_epsilon,
    #     }
    #     if cfg.uncertainty_weights is not None:
    #         gen_kwargs["uncertainty_weights"] = cfg.uncertainty_weights
    #     if cfg.generation_temperature and cfg.generation_temperature > 0:
    #         gen_kwargs["do_sample"] = True
    #         gen_kwargs["temperature"] = cfg.generation_temperature
    #     else:
    #         gen_kwargs["do_sample"] = False
    #     return gen_kwargs

    def _apply_uncertainty_reduction(
        self,
        lang_out,
        cfg: LangAlignConfig,
    ) -> LangAlignGenerateDecoderOnlyOutput:
        if cfg.uncertainty_reduction != "max" or not lang_out.per_step_uncertainty:
            return lang_out

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        skip = resolve_skip_token_ids(
            tokenizer, cfg.skip_token_ids, cfg.extra_skip_token_ids
        )
        gen_ids = lang_out.generated_tokens
        if gen_ids is None and lang_out.sequences is not None:
            gen_ids = lang_out.sequences[:, -len(lang_out.per_step_uncertainty) :]
        if gen_ids is None:
            return lang_out

        lang_out.sequence_uncertainty = aggregate_sequence_uncertainty(
            lang_out.per_step_uncertainty,
            skip_token_ids=skip,
            generated_ids=gen_ids[0] if gen_ids.dim() == 2 else gen_ids,
            reduction="max",
        )
        return lang_out

    @torch.inference_mode()
    def run_one(
        self,
        question: str,
        image: Any,
        *,
        temperature: Optional[float] = None,
    ) -> LangAlignGenerateDecoderOnlyOutput:
        cfg = self.config
        if temperature is not None:
            cfg = LangAlignConfig(
                **{**cfg.__dict__, "generation_temperature": temperature}
            )

        self._ensure_lang_align_patch()

        messages = self.build_messages(image, question)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = inputs.to(device)

        gen_kwargs = self._build_gen_kwargs(cfg)
        # if cfg.lang_align_use_no_image and "pixel_values" in inputs:
        #     gen_kwargs["pixel_values_blank"] = self.blur_fn(inputs["pixel_values"])

        self.model.generation_config.return_dict_in_generate = True
        self.model.generation_config.output_logits = True
        # GenerationMixin is patched process-wide, but activation belongs to
        # this Qwen model's generation config.  Passing this as an arbitrary
        # generate() kwarg would fail transformers' model-kwarg validation.
        self.model.generation_config.lang_align_decode = True
        self.model.generation_config.lang_align_collect_step_attentions = cfg.collect_step_attentions
        self.model.generation_config.lang_align_use_no_image = cfg.lang_align_use_no_image
        self.model.generation_config.lang_align_blur_epsilon = cfg.blur_epsilon
        self.model.generation_config.lang_align_uncertainty_weights = cfg.uncertainty_weights

        lang_out = self.model.generate(**inputs, **gen_kwargs)
        # if not isinstance(lang_out, LangAlignGenerateOutput):
        #     lang_out = LangAlignGenerateOutput(**lang_out)

        lang_out = self._apply_uncertainty_reduction(lang_out, cfg)
        return inputs , lang_out
