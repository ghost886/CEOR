# CEOR

Code extracted for the paper **Revealing Hidden Epistemic Uncertainty in Visual Grounding via Probe-Guided Occlusion**. The implementation comes from `semantic_uncertainty_vlm_v3/semantic_uncertainty` and follows the method described in `main_twocolumn.pdf`.

**Method name:** **CEOR** denotes the method and the final score defined in Section 3.4.3, Equation (16). The score is returned in the `ceor` field. The legacy names `LGD / lgd_uq` correspond to **OSD** in the paper and are retained for compatibility with the original generation entry point.

The extraction adds source code, configuration templates, tests, and documentation. Original experiment results, W&B runs, images, feature caches, and model weights are not included. An existing local copy of `main_twocolumn.pdf` is retained in the workspace and excluded by `.gitignore`.

## Entry Points

| Pipeline stage | Recommended command | Core implementation |
| --- | --- | --- |
| OSD: target and auxiliary sampling, answer modes, and four supervision targets | `python generate_answers.py --collect_lgd_uq ...` | [Auxiliary model adapters](uncertainty/models/lgd_reference_models.py), [OSD formulas](uncertainty/uncertainty_measures/lgd_uq.py) |
| Generate auxiliary outputs separately or compute OSD from existing boxes | `python -m ceor osd ...` | [ceor/osd.py](ceor/osd.py) |
| Select auxiliary models on the training split and freeze the selection | `python -m ceor prepare-labels ...` | [Label preparation](experiments/prepare_lgd_quality_filtered_labels.py) |
| Collect head features from each layer before generation | `python -m ceor collect-heads ...` | [Feature collection](experiments/collect_qwen_uncertainty_heads.py) |
| Fit Ridge probes with image-level grouping and select Key1 | `python -m ceor fit-probes ...` | [Original probes](experiments/analyze_qwen_uncertainty_heads.py), [layer selection](ceor/probe.py) |
| Freeze the answer, mask the predicted box, and compute CEOR | `python -m ceor score ...` | [Occlusion and response](ceor/occlusion.py), [inference](ceor/inference.py) |
| Original answer generation and conventional uncertainty methods | `generate_answers.py`, `compute_uncertainty_measures.py` | Original entry points and their full in-project dependencies |
| Cached trajectory metrics, including CoE and SIVR | `python -m scripts.compute_primary_cached_metrics ...` | [Cached metrics entry point](scripts/compute_primary_cached_metrics.py) |

The complete image → generation → collection → occlusion pipeline currently targets **Qwen3-VL**. OSD, Ridge probes, layer selection, and the response formulas are model-independent and support explicit physical layer IDs. Historical Qwen3.5 / InternVL queue and cache-assembly scripts from the source project are not released as supported target-model entry points; those model names cannot be passed directly to the Qwen3-VL wrapper. See the [extraction scope](docs/EXTRACTION.md).

## Installation

Use a separate Python 3.10 or later environment and run the following from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Install for the OpenAI-compatible or Zhipu SDK; unnecessary for stdlib HTTP adapters
python -m pip install -e '.[api]'

# Optional: local W&B logging and the cross-model sentence-T5 similarity baseline
python -m pip install -e '.[logging,semantic]'
```

Core dependencies are listed in [pyproject.toml](pyproject.toml). `transformers` is pinned to `5.16.1`, the version used in the validation environment; [requirements-tested.txt](requirements-tested.txt) records the key dependency versions used for validation. Full VLM experiments also require model weights, datasets, and a working CUDA/PyTorch environment.

The programs do not override proxy settings or `HF_HOME` and do not depend on paths from the source project. Set the following variables to your local paths before running:

```bash
export VISUAL_GROUNDING_ROOT=/path/to/visual_grounding
export HF_HOME=/path/to/huggingface-cache
export WANDB_MODE=offline
export SCRATCH_DIR=experiments

MODEL=/path/to/Qwen3-VL-4B-Instruct
DATASET=refcoco
```

You can also set `MODEL` to `Qwen/Qwen3-VL-4B-Instruct` and let Transformers download the model. `.env.example` is a template only: export its variables in your shell, as the programs do not load `.env` automatically.

## Run CEOR from Scratch

The following example uses RefCOCO. The sample counts of 1000/500 are adjustable examples; **they do not imply that all nine datasets in the paper use the same sample counts**. Formal comparisons must fix sample IDs, image splits, target answers, and the auxiliary model selection.

### 1. Configure Auxiliary Models and Generate Answers / OSD

Copy [configs/auxiliary_models.example.json](configs/auxiliary_models.example.json) to `configs/auxiliary_models.local.json` and replace the model IDs, service URLs, or local model paths with actual values. The `.example` URLs are placeholders. API keys are supplied only through the environment variables named by `api_key_env`. The target model must not also appear in the auxiliary model set.

```bash
python generate_answers.py \
  --model_name "$MODEL" --dataset "$DATASET" \
  --train_num_samples 1000 --validation_num_samples 500 \
  --num_generations 10 --temperature 1.0 --low_temperature 0 \
  --model_max_new_tokens 128 \
  --get_training_set_generations \
  --no-get_training_set_generations_most_likely_only \
  --collect_lgd_uq \
  --lgd_reference_config configs/auxiliary_models.local.json \
  --lgd_num_reference_samples 10 --lgd_num_target_samples 10 \
  --lgd_reference_temperature 0.8 \
  --lgd_iou_threshold 0.5 --lgd_smoothing 0.01 \
  --collect_probe_features \
  --no-collect_primary_trajectories \
  --no-collect_cross_model_semantic_uq \
  --generation_checkpoint_dir "experiments/cache/$DATASET"
```

This command disables the optional trajectory and sentence-T5 baselines to run the OSD → probe → occlusion pipeline first; instructions for enabling those baselines appear below. Auxiliary API calls send images and questions and consume the corresponding account quota. Sampling uses per-model checkpoint caches, so rerunning the same configuration resumes from completed samples.

`generate_answers.py` uses W&B offline mode by default. The log reports a new run directory containing `files/train_generations.pkl`, `files/validation_generations.pkl`, auxiliary-result sidecars, and the original uncertainty caches. If W&B is unavailable, the local fallback writes directly to `experiments/$USER/uncertainty/files/`. Use a different `SCRATCH_DIR` for each independent experiment.

Set `RUN_DIR` to the newly generated run directory or its `files/` subdirectory:

```bash
RUN_DIR=/path/to/new/run
LABELS="experiments/outputs/$DATASET/labels"
FEATURES="experiments/outputs/$DATASET/heads"
PROBES="experiments/outputs/$DATASET/probes"

python -m ceor prepare-labels \
  --run-dir "$RUN_DIR" --output-dir "$LABELS"
```

Selection uses the training split only. The defaults require a parse rate of at least 0.90, sample coverage of at least 0.95, and at least three auxiliary models. Models whose accuracy trails the target by no more than 0.05 are preferred; if necessary, the set is supplemented only with models whose accuracy gap is at most 0.15. Validation uses the frozen selection. If these conditions cannot be met, the program raises an error; inspect model quality or explicitly document a revised selection protocol.

### 2. Collect Head Features Before Generation, Train Probes, and Select the Key Layer

```bash
python -m ceor collect-heads \
  --run-dir "$RUN_DIR" --model "$MODEL" \
  --splits train validation --views answer_free \
  --label-sidecar "$LABELS/label_sidecar.json" \
  --output-dir "$FEATURES"

python -m ceor fit-probes \
  --features-dir "$FEATURES" --output-dir "$PROBES" \
  --num-folds 5 --ridge-alpha 10 --jobs 4
```

Features are captured **at the final query position of the initial prompt prefill, before the attention `o_proj`**, with no answer tokens present. Each head receives a four-output Ridge probe trained on `log1p(U_A, U_D, U_E, U_total)`. Standardization parameters are fitted within each training fold only.

`fit-probes` saves per-head out-of-fold (OOF) rankings and fitted parameters, then writes `key_layer.json`. Key1 is selected by the mean `U_total` OOF Spearman correlation over **all heads** in each layer, with ties resolved in favor of the smaller physical layer ID. Validation performance is not used for selection. Layer IDs are zero-based.

Image overlap between training and validation raises an error by default. For datasets such as Ref-Adv, add `--exclude-validation-group-overlap` to exclude overlapping training images explicitly. The exclusions are recorded in the probe metadata.

### 3. Freeze the Original Answer and Compute the Predicted-Region Occlusion Response

```bash
python -m ceor score \
  --run-dir "$RUN_DIR" --split validation \
  --model "$MODEL" --key-layer "$PROBES/key_layer.json" \
  --features-dir "$FEATURES" \
  --output "experiments/outputs/$DATASET/ceor.jsonl"
```

By default, the predicted box is filled with the **rounded RGB channel means of the entire original image**. The masked pass performs prompt prefill without generating a new answer. The score is:

```text
d[layer, head] = ||z_masked - z_clean||₂ / max(||z_clean||₂, 1e-8)
CEOR = -mean(d[layer, head] for every head in layers strictly after Key1)
```

A higher score (closer to zero) indicates a weaker response and higher risk. CEOR is a ranking score rather than an error probability. Scoring uses no GT boxes or additional auxiliary model calls. By default, the output records the original answer, predicted box, score, and response availability. Unparseable or degenerate boxes, or a Key1 at the final collected layer, produce `null` with a recorded reason instead of zero or an all-layer fallback.

When matching clean features can be reused, each valid sample requires only **one additional prefill**. Omitting `--features-dir` requires recomputing both clean and masked features, for two prefill passes. The program validates the model, image digest, prompt, original answer, and feature dimensions. Keep `QWEN_VL_MIN_PIXELS` / `QWEN_VL_MAX_PIXELS` consistent across generation, probe collection, and scoring.

Use `--mask-method black|white|gaussian_blur` for fill-method ablations. `--save-head-responses` additionally saves each head's relative L2 and cosine change. Image-mean masking remains the primary method used in the paper.

## Conventional Methods and Original Entry Points

The original filenames are **`generate_answers.py`** and **`compute_uncertainty_measures.py`**. These spellings are preserved; aliases such as `genenrate_answer` are not provided.

```bash
python compute_uncertainty_measures.py \
  --use_local_wandb --local_eval_run_dir "$RUN_DIR" \
  --local_output_dir "experiments/outputs/$DATASET/baselines" \
  --compute_predictive_entropy --entailment_model deberta \
  --compute_probe_sep_cv
```

Local computation requires a `--local_output_dir` different from the input directory. The DeBERTa semantic entropy baseline loads/downloads an NLI model on its first run; after caching the weights, you can set `HF_HUB_OFFLINE=1`. Add `--no-compute_predictive_entropy` when computing only probes on existing features.

| Method | Generation requirements | Computation |
| --- | --- | --- |
| Regular / semantic / cluster-assignment entropy | Multiple `responses` and token log likelihoods | `--compute_predictive_entropy` |
| P(True) | `--compute_p_true` during generation; optionally configure `--p_true_num_fewshot` | `p_false_fixed = 1-exp(log_p_true)` in the generation cache |
| P(IK) | Embeddings of training and validation answers | `--compute_p_ik` |
| SEP / SEP CV / PCA CV | `--collect_probe_features` | `--compute_lightweight_probe` / `--compute_probe_sep_cv` / `--compute_probe_sep_pca_cv` |
| LRP | `--collect_lrp_probe` | `--compute_lrp_probe` |
| ICR / VIB / raw VIB | `--collect_icr_probe` / `--collect_vib_probe` | Respective flags: `--compute_icr_probe` / `--compute_vib_probe` / `--compute_raw_vib_probe` |
| CoE / SIVR / internal entropy trajectories | `--collect_primary_trajectories` | `python -m scripts.compute_primary_cached_metrics` |
| Cross-model semantic similarity baseline | OSD sampling with `--collect_cross_model_semantic_uq` enabled | Computed during generation; also requires `.[semantic]` and sentence-T5 weights |

For example, use a run with cached trajectories and previously generated P(True) values:

```bash
python -m scripts.compute_primary_cached_metrics \
  --run-dir "$RUN_DIR" \
  --base-results "experiments/outputs/$DATASET/baselines/uncertainty_measures.pkl" \
  --output-dir "experiments/outputs/$DATASET/trajectory_baselines" \
  --sivr-epochs 100
```

All comparisons must align by sample ID to the same frozen answer. Greedy, draw0, and historical majority answers use different prediction protocols and cannot be mixed directly.

## Standalone Inputs and Development Validation

`ceor osd` and `ceor score` can run without an original W&B directory. See [data and file formats](docs/DATA_AND_FORMATS.md) for the JSONL schemas. Method formulas, edge cases, and source-code provenance are described in the [method and implementation guide](docs/METHOD.md) and [extraction inventory](docs/EXTRACTION.md).

```bash
python -m ceor --help
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q -p no:cacheprovider --basetemp=/tmp/ceor-tests
```

The tests use temporary synthetic inputs, mocked auxiliary interfaces, and a small, randomly initialized Qwen3-VL model. They require no GPU, API keys, or pretrained weights. The [validation notes](docs/VALIDATION.md) describe the scope and limitations of the recorded validation.

The original license is retained in [LICENSE](LICENSE). Third-party notices and bundled licenses are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/`.
