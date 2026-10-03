# Third-party notices

## Inference-Time Intervention / honest_llama

The additive pre-`o_proj` intervention hook retained in
`uncertainty/uncertainty_measures/paper_probe_features.py` originates from the
source project's Qwen visual-grounding adaptation of:

> Kenneth Li et al. “Inference-Time Intervention: Eliciting Truthful Answers
> from a Language Model.” NeurIPS 2023.

- Source: <https://github.com/likenneth/honest_llama>
- Reviewed revision: `2c6b2179be7b5aa8f0a171688cf9e01b812ca327`
- Retained changes: Qwen3-VL module discovery and token-aligned pre-`o_proj`
  capture/intervention support. The upstream `qwen_grounding_iti.py` experiment
  driver is not part of this extraction.
- License: MIT; see
  [`third_party/honest_llama_LICENSE`](third_party/honest_llama_LICENSE).

## ICR Probe

The implementation in
`uncertainty/uncertainty_measures/icr_probe.py` and the Qwen primary-answer
trajectory support in `primary_answer_trajectory.py` are an
adapted and extended reproduction of:

> Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe Zhang, and Xiaojun Wan.
> “ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination
> Detection in LLMs.” ACL 2025.

- Source: <https://github.com/XavierZhang2002/ICR_Probe>
- Integrated revision: `40ec490e762cadbac6bcefdc24a8f0d5974e8448`
- Retained changes: dense teacher-forced cache support, explicit Attn/Proj
  trajectory outputs, numerical/device corrections, data handling and
  training, checkpointing, project CLI integration, tests, and
  error-probability semantics. The source project's separate Qwen3.5 and
  InternVL collectors are not included.
- License: Apache License 2.0; see
  [`third_party/ICR_Probe_LICENSE`](third_party/ICR_Probe_LICENSE).
