# CEOR: Method and Implementation

This document maps Sections 3.2–3.5 of `main_twocolumn.pdf` to the implementation. **CEOR** is the method name and the final uncertainty score. LGD is the legacy name for OSD in the code. Key1 is selected from training data rather than replaced with constants from historical experiment tables.

## OSD: Equations (1)–(9)

Implementation: `uncertainty/uncertainty_measures/lgd_uq.py::recover_lgd_uncertainty`, also exposed as `ceor.osd.recover_osd`.

1. Convert ground-truth (GT) and auxiliary boxes to normalized xyxy coordinates. Connect valid boxes when their pairwise IoU is at least 0.5, construct modes from the connected components, record an IoU medoid as each representative box, and append an `other_invalid` mode.
2. GT helps define the modes but does not count as an auxiliary prediction or receive a dedicated pseudocount. Target boxes are projected onto existing components without creating or changing modes. A target box belongs to a component if its IoU with any member reaches the threshold.
3. Each model uses its own number of actual returned answers as its sample count. Unparseable answers still count toward `other_invalid`. Requests that fail or are rejected by the service without returning an actual model answer are excluded through `excluded_from_uncertainty`. These two cases must remain distinct.
4. Apply additive smoothing with ε=0.01: `q_m(r)=(n_mr+ε)/(K_m+Rε)`. The target distribution `p(r)` uses the same support. Model weights are uniform by default, with `pi=sum(w_m*q_m)`.
5. Compute `U_A=sum(w_m*H(q_m))`, `U_D=sum(w_m*KL(q_m||pi))`, `U_E=KL(pi||p)`, and `U_total=U_A+U_D+U_E=-sum(pi*log(p))`. All logarithms are natural, and uncertainty is measured in nats.

`prepare-labels` selects auxiliary models on the training split using accuracy under matched repeated sampling, parse rate, and coverage. The validation split uses the frozen selection. Standalone `ceor osd` does not perform this selection automatically: supply a preselected auxiliary set or perform selection on the training split afterward.

## Probes and Key1: Equations (10)–(11)

Original implementation: `experiments/analyze_qwen_uncertainty_heads.py`. Pipeline wrapper and layer selection: `ceor/probe.py`.

- Features are the per-head outputs immediately before `o_proj` at the final query position of the initial prompt prefill. Their shape is `[L,H,D]` per sample and `[N,L,H,D]` for the full collection. They are neither answer-token hidden states nor attention weights.
- Fit an independent multi-output Ridge probe for each head, using targets `log1p([U_A,U_D,U_E,U_total])` and α=10. Fit a StandardScaler within each training fold and obtain out-of-fold (OOF) predictions through five-fold GroupKFold grouped by image ID. With fewer than five image groups, the original implementation reduces the fold count to the number of groups; at least two groups are required. A full five-fold experiment requires at least five groups.
- Apply the inverse transform `max(expm1(prediction),0)` and compute the head's training OOF Spearman correlation.
- For each layer, take the arithmetic mean of the `U_total` OOF Spearman correlations over all heads. Select Key1 by descending mean, breaking ties by ascending physical layer ID. Individual head rankings, Top-K head density, and validation AUC do not replace this criterion.
- If constant predictions or labels make a head's Spearman correlation undefined, its layer is ineligible for Key1. Undefined heads are not dropped from the all-head average. Raise an error if no layer is eligible.
- `model_layer_ids` distinguishes feature-array positions from physical model layers; both are zero-based. Qwen3-VL uses contiguous layers, while the generic functions also accept sparse physical-layer mappings such as `[3,7,11,...]`.

## CEOR: Predicted-Region Response, Equations (13)–(16)

The implementation is extracted from `pixel_box`, `mask_image`, and `head_state_change` in the source project's `experiments/run_qwen_prediction_box_deletion.py`, together with the layer-mean selection in `scripts/analyze_qwen3vl_all9_key1_layer_ablation.py`. The standalone implementation is in `ceor/occlusion.py`, and the inference entry point is in `ceor/inference.py`.

Parse predicted boxes using the original grounding evaluation rules for normalized coordinates or Qwen's 0–999 coordinates. Apply floor to the upper-left pixel boundary and ceil to the lower-right boundary. Subtract one from the lower-right coordinates when drawing a PIL rectangle to preserve the half-open region. The primary fill color is the rounded per-channel mean of the **entire original image**. Pixels outside the box remain unchanged.

Clean and masked passes use the same prompt, processor, model, and query position. Compute each head's relative L2 response in float64:

```text
d_lh = norm(z_masked_lh - z_clean_lh) / max(norm(z_clean_lh), 1e-8)
CEOR = -mean(d_lh over all heads with physical layer > Key1)
```

Normalize each head before averaging; concatenating all vectors and normalizing once would change the score. Key1 itself is excluded. Scoring requires no Top-K head selection, trained probe predictions, placebo controls, or GT. Black, white, Gaussian blur, and negative cosine are retained as ablation options; the primary definition uses image-mean masking and relative L2.

Unparseable boxes, degenerate boxes, and a Key1 at the final layer yield an unavailable response with a recorded reason. They are not assigned zero or given an all-layer fallback. Report response availability and compare metrics on the same valid samples. If downstream fusion requires missing-value imputation, fit and freeze the imputation rule on the training/development split only. This extraction provides the original baselines and the primary score. Historical result-assembly scripts for paper-specific fusion, confidence intervals, and plots are outside the default pipeline.

Before reusing cached features, validate the sample ID, frozen answer, prompt, model identifier, processor image limits, image SHA256, and feature dimensions. A digest of the source generation file is recorded in the collection identity to prevent silently reusing an output directory for a different set of generations.

## Scope of Use

Auxiliary models and GT are used only for offline supervision and evaluation; the `score` interface requires neither. Obtain a fixed answer before scoring a new prediction, either through `generate_answers.py` or by writing an externally generated answer and its exact prompt to JSONL.

Offline collection performs an additional clean prefill. In deployment, retaining the same clean head states during normal generation leaves only one masked prefill. This implementation likewise uses one pass when reusing the collection cache and explicitly performs two when the cache is unavailable. CEOR is a diagnostic measure of internal response, not a calibrated error probability.
