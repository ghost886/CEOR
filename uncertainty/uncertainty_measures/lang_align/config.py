from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LangAlignConfig:
    """Hyperparameters for attention-based language–vision alignment uncertainty."""

    max_new_tokens: int = 128
    generation_temperature: float = 0.0
    """0.0 means greedy decoding (do_sample=False)."""

    blur_epsilon: float = 1e-12
    lang_align_use_no_image: bool = True
    """Run empty-image chain for visual-evidence scoring."""

    uncertainty_reduction: str = "mean"
    """Sequence-level aggregation: ``mean`` or ``max``."""

    uncertainty_weights: Optional[Dict[str, float]] = None
    """Override default per-signal weights in ``compute_step_uncertainty``."""

    extra_skip_token_ids: Optional[List[int]] = None
    skip_token_ids: Optional[List[int]] = None

    display_output: bool = True
    """Print question / output / uncertainty summary per sample."""

    attention_debug: bool = False
    """Write detailed attention summaries and visualizations during generation."""

    attention_debug_dir: str = "attention_debug"
    """Directory for attention debug artifacts."""

    attention_debug_max_samples: int = 20
    """Maximum number of samples for which debug artifacts are written."""

    collect_step_attentions: bool = True
    """Collect decode-step attentions for bbox-token grounding metrics."""


# vcd sample 的 config
@dataclass
class KeywordExtractConfig:
    """Hyperparameters aligned with LVLM_Interpretation `find_keywords` logic."""

    log_prob_gap: float = 1.0
    """Select token if log p(clear) - log p(blur) > this value (same spirit as reference)."""

    min_prob_clear: float = 0.0
    """Lower bound on p(token | clear image); reference used >= 0.0."""

    blur_epsilon: float = 1e-12
    """Stabilizes log when blur probabilities are near zero."""

    max_new_tokens: int = 128
    generation_temperature: float = 0.0
    """0.0 means greedy decoding (do_sample=False)."""

    use_dual_sample: bool = False
    """If True, use dual-chain generate + find_keywords_dual_sample (single pass)."""

    extra_skip_token_ids: Optional[List[int]] = None
    """Merged with tokenizer `all_special_ids` when filtering keywords."""

    skip_token_ids: Optional[List[int]] = None
    """If set, used exclusively instead of tokenizer specials + extra_skip_token_ids."""
