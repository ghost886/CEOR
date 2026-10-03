"""Utility functions."""
import os
import logging
import argparse
import pickle
import string
import warnings

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

try:
    import wandb
except ImportError:  # pragma: no cover - local-only fallback
    from uncertainty.utils import wandb_stub as wandb

try:
    from evaluate import load
except ImportError:  # pragma: no cover - optional unless metrics are recomputed
    def load(*args, **kwargs):
        raise ImportError(
            "`evaluate` is required for this metric. Install evaluate or run "
            "without --recompute_accuracy."
        )

from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.models.qwen_vl_models import QwenVLModel
from uncertainty.uncertainty_measures import grounding_uncertainty as grounding_u
from uncertainty.uncertainty_measures.icr_probe import ICRScoreConfig
from uncertainty.uncertainty_measures.lang_align.config import LangAlignConfig
from uncertainty.utils import openai as oai

import re
import string

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n'}

GENERATION_DATASET_CHOICES = [
    'trivia_qa', 'squad', 'bioasq', 'nq', 'svamp', 'vqa', 'flickr30k',
    'flickr30k_entities', 'refcoco', 'refcoco_plus', 'refcocog',
    'refcoco_grounding', 'grefcoco', 'grefcoco_single', 'pr_bench', 'ref_l4',
    'ref_adv', 'talk2car', 'visual7w', 'qwen3_vl', 'gqa', 'clearVQA', 'mmmu',
    'scienceQA',
    'textQA',
]


def get_parser(stages=['generate', 'compute']):
    entity = os.getenv('WANDB_SEM_UNC_ENTITY', None)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument('--entity', type=str, default=entity)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        '--icr_top_k', type=int, default=20,
        help="Top attention keys used by ICR (paper default: 20).")
    parser.add_argument(
        '--icr_top_p', type=float, default=None,
        help="Optional top-key fraction; when set, overrides --icr_top_k.")
    parser.add_argument(
        '--icr_attention_pooling', choices=('mean', 'max', 'min'), default='mean',
        help="How ICR pools attention heads (paper default: mean).")
    parser.add_argument(
        '--icr_attention_scope', choices=('all', 'prompt', 'response'), default='all',
        help="Keys visible to ICR; all follows the paper's context formulation.")
    parser.add_argument(
        '--icr_use_induction_head', default=False,
        action=argparse.BooleanOptionalAction,
        help="Use the optional induction-head selector from the reference code.")
    parser.add_argument('--icr_skew_threshold', type=float, default=0.0)
    parser.add_argument('--icr_entropy_threshold', type=float, default=1e5)
    parser.add_argument(
        '--icr_attention_uniform', default=False,
        action=argparse.BooleanOptionalAction,
        help="ICR attention-uniform ablation (HS-only in the paper).")
    parser.add_argument(
        '--icr_hidden_uniform', default=False,
        action=argparse.BooleanOptionalAction,
        help="ICR hidden-projection-uniform ablation.")
    parser.add_argument(
        '--icr_device', default=None,
        help="Optional device for ICR extraction, for example cpu or cuda:1.")
    parser.add_argument(
        '--icr_save_direction_vectors', default=False,
        action=argparse.BooleanOptionalAction,
        help="Save the top-k Attn/Proj vectors and layer trajectory diagnostics.")
    parser.add_argument(
        "--metric", type=str, default="squad",
        choices=[
            'squad', 'llm', 'llm_gpt-3.5', 'llm_gpt-4',
            'vqa', 'vqa_soft', 'grounding',
        ],
        help="Metric to assign accuracy to generations.")
    parser.add_argument(
        "--compute_accuracy_at_all_temps",
        action=argparse.BooleanOptionalAction, default=True,
        help="Compute accuracy at all temperatures or only t<<1.")
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, default="Llama-2-7b-chat", help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens", type=int, default=500,
            help="Max number of tokens generated.",
        )
        parser.add_argument(
            "--dataset", type=str, default="trivia_qa",
            choices=GENERATION_DATASET_CHOICES,
            help="Dataset to use")
        parser.add_argument(
            "--pr_bench_train_fraction", type=float, default=0.0,
            help=(
                "Optional deterministic fraction of PR-Bench positive records "
                "to expose as a derived train split. The default 0 preserves "
                "the official test-only validation protocol."))
        parser.add_argument(
            "--ood_train_dataset", type=str, default=None,
            choices=GENERATION_DATASET_CHOICES,
            help="Dataset to use to assemble few-shot prompt, p_true prompt, and train p_ik.")
        parser.add_argument(
            "--num_samples", type=int, default=400,
            help=(
                "Fallback number of samples to use for each dataset split. "
                "Overridden by --train_num_samples or "
                "--validation_num_samples when those are provided."))
        parser.add_argument(
            "--train_num_samples", type=int, default=None,
            help=(
                "Number of training samples to generate. Defaults to "
                "--num_samples when omitted."))
        parser.add_argument(
            "--validation_num_samples", type=int, default=None,
            help=(
                "Number of validation samples to generate. Defaults to "
                "--num_samples when omitted."))
        parser.add_argument(
            "--num_few_shot", type=int, default=0,#5,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_hint", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--num_generations", type=int, default=10,
            help="Number of generations to use")
        parser.add_argument(
            "--temperature", type=float, default=1.0,
            help="Temperature")
        parser.add_argument(
            "--low_temperature", type=float, default=0.1,
            help=(
                "Temperature for the primary answer used by probes and task "
                "accuracy. Set to 0 for greedy decoding (LRP paper setup)."))
        parser.add_argument(
            "--collect_lgd_uq", default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Collect Latent Grounding Distribution teacher labels U_A/U_D/U_E "
                "from a heterogeneous reference-VLM ensemble."))
        parser.add_argument(
            "--lgd_reference_config", default=None,
            help=(
                "JSON file defining huggingface and/or OpenAI-compatible "
                "reference VLMs. Required by --collect_lgd_uq."))
        parser.add_argument(
            "--lgd_num_reference_samples", type=int, default=8,
            help="Stochastic grounding draws per auxiliary/reference model.")
        parser.add_argument(
            "--lgd_num_target_samples", type=int, default=8,
            help="High-temperature target-model draws used only for U_E.")
        parser.add_argument(
            "--lgd_reference_temperature", type=float, default=0.8,
            help="Sampling temperature shared by configured reference VLMs.")
        parser.add_argument(
            "--lgd_iou_threshold", type=float, default=0.5,
            help="IoU edge threshold for sample-specific grounding modes.")
        parser.add_argument(
            "--lgd_iou_ablation_thresholds", nargs="+", type=float,
            default=[0.3, 0.7, 0.9],
            help=(
                "Additional IoU clustering thresholds recomputed from the same "
                "LGD samples. The primary threshold is included automatically."))
        parser.add_argument(
            "--lgd_smoothing", type=float, default=0.01,
            help="Laplace smoothing epsilon for auxiliary and target mode counts.")
        parser.add_argument(
            "--lgd_qwen_coordinate_scale", type=float, default=999.0,
            help="Coordinate scale accepted for Qwen-style non-normalized boxes.")
        parser.add_argument(
            "--lgd_splits", nargs="+", choices=("train", "validation"),
            default=["train", "validation"],
            help="Dataset splits on which the expensive LGD teacher is collected.")
        parser.add_argument(
            "--lgd_parallel_workers", type=int, default=0,
            help=(
                "Number of auxiliary-model workers used during LGD prefetch. "
                "Zero uses one worker per configured reference model."))
        parser.add_argument(
            "--lgd_requests_per_model", type=int, default=2,
            help=(
                "Concurrent sample requests for each remote auxiliary model. "
                "Local Hugging Face references always use one. Reduce this to "
                "1 when a provider enforces a strict rate limit."))
        parser.add_argument(
            "--generation_checkpoint_dir",
            default="experiments/cache/generation_checkpoints",
            help=(
                "Parent directory for crash-safe target and per-reference-model "
                "sample shards. A configuration fingerprint is appended. Passing "
                "an existing run directory containing manifest.json resumes that "
                "exact run."))
        parser.add_argument(
            "--resume_generation", default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Reuse completed target samples and auxiliary draws from the "
                "matching generation checkpoint directory."))
        parser.add_argument(
            "--collect_cross_model_semantic_uq", default=None,
            action=argparse.BooleanOptionalAction,
            help=(
                "Compute Hamidieh et al. ICLR 2026 response-similarity AU/EU/TU "
                "from the target and auxiliary samples collected by LGD-UQ. "
                "Defaults to enabled whenever --collect_lgd_uq is active; use "
                "--no-collect_cross_model_semantic_uq to opt out."))
        parser.add_argument(
            "--cross_model_semantic_encoder",
            default="sentence-transformers/sentence-t5-xl",
            help="Sentence embedding checkpoint used for semantic similarity.")
        parser.add_argument(
            "--cross_model_semantic_device", default="cpu",
            help="Device for the semantic encoder, for example cpu or cuda:1.")
        parser.add_argument(
            "--cross_model_semantic_batch_size", type=int, default=32,
            help="Sentence-encoder batch size for cross-model semantic UQ.")
        parser.add_argument(
            "--use_mc_options", type=bool, default=True,
            help="Include MC options question?")
        parser.add_argument(
            "--get_training_set_generations", default=True,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--use_context", default=False,
            action=argparse.BooleanOptionalAction,
            help="Get generations for training set?")
        parser.add_argument(
            "--get_training_set_generations_most_likely_only", default=True,
            action=argparse.BooleanOptionalAction,
            help=(
                "Only get embedding of most likely answer for training set. "
                "This is all that's needed for p_true."))
        parser.add_argument('--compute_p_true', default=False,#True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_always", default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--enable_brief", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument(
            "--brief_prompt", default='default', type=str)
        parser.add_argument(
            "--prompt_type", default='default', type=str)
        parser.add_argument(
            "--answerable_only", default=False,
            action=argparse.BooleanOptionalAction,
            help='Exclude unanswerable questions.')
        parser.add_argument(
            "--compute_lang_align", default=False,
            action=argparse.BooleanOptionalAction,
            help="Enable language-vision attention diagnostics during generation.")
        parser.add_argument(
            "--lang_align_max_new_tokens", type=int, default=None,
            help="Max new tokens for lang-align decode (defaults to model_max_new_tokens).")
        parser.add_argument(
            "--no_lang_align_no_image_chain", default=False,
            action=argparse.BooleanOptionalAction,
            help="Disable empty-image chain in lang-align uncertainty.")
        parser.add_argument(
            "--lang_align_uncertainty_reduction", type=str,
            choices=("mean", "max"), default="mean",
            help="Sequence-level lang-align uncertainty aggregation.")
        parser.add_argument(
            "--lang_align_blur_epsilon", type=float, default=1e-12,
            help="Stabilizes log-probability terms in lang-align scoring.")
        parser.add_argument(
            "--attention_debug", default=False, action=argparse.BooleanOptionalAction,
            help="Log and save attention diagnostics for lang-align/grounding metrics.")
        parser.add_argument(
            "--attention_debug_dir", type=str, default="attention_debug",
            help="Directory for attention diagnostic PNG/CSV/TXT artifacts.")
        parser.add_argument(
            "--attention_debug_max_samples", type=int, default=20,
            help="Maximum number of samples to dump attention diagnostics for.")
        parser.add_argument(
            "--collect_primary_trajectories", default=None,
            action=argparse.BooleanOptionalAction,
            help=(
                "Collect primary-answer ICR Attn/Proj directions, answer-free "
                "heads, CoE, layer entropy and hidden convergence. Enabled "
                "automatically for Qwen; adds a shared teacher-forced forward. "
                "Use --no-collect_primary_trajectories to disable."))
        parser.add_argument(
            "--collect_probe_features", default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Save lightweight hidden-state features for post-hoc SEP "
                "probe analysis during answer generation."))
        parser.add_argument(
            "--collect_icr_probe", default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Collect the token-by-layer ICR matrix and its layer-wise mean "
                "from a teacher-forced pass over each low-temperature answer."))
        parser.add_argument(
            "--collect_vib_probe", default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Capture every decoder attention head's pre-o_proj output at "
                "the final answer decoding step for VIB-Probe."))
        parser.add_argument(
            "--collect_lrp_probe", default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Collect all-layer output hidden states, input attention "
                "patterns, and visual-token attention for LRP abstention."))
        parser.add_argument(
            "--vib_mitigation_checkpoint", default=None,
            help=(
                "Optional vib_probe.pt used for per-token gradient head "
                "suppression during low-temperature inference."))
        parser.add_argument(
            "--vib_mitigation_threshold_logit", type=float, default=None,
            help="Override the checkpoint's mean-training-logit trigger threshold.")
        parser.add_argument(
            "--vib_mitigation_top_fraction", type=float, default=0.05,
            help="Fraction of heads suppressed after VIB attribution (paper: 0.05).")
        parser.add_argument(
            "--vib_mitigation_strength", type=float, default=0.001,
            help="Single-step VIB suppression strength lambda (paper: 0.001).")
        parser.add_argument(
            "--vib_mitigation_minimum_scale", type=float, default=None,
            help=(
                "Optional safety clamp for head scales; omitted reproduces "
                "the paper's unclamped Eq. (14)."))

    if 'compute' in stages:
        parser.add_argument('--recompute_accuracy',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--eval_wandb_runid', type=str,
                            help='wandb run id of the dataset to evaluate on')
        parser.add_argument('--train_wandb_runid', type=str, default=None,
                            help='wandb run id of the dataset from which training embeddings and p_true samples will be taken')
        parser.add_argument(
            '--use_local_wandb', default=False,
            action=argparse.BooleanOptionalAction,
            help="Read generation artifacts from a local wandb run directory instead of wandb.Api.")
        parser.add_argument(
            '--local_wandb_dir', type=str, default=None,
            help="Local wandb root containing run-* directories.")
        parser.add_argument(
            '--local_eval_run_dir', type=str, default=None,
            help="Local eval run directory, e.g. .../wandb/run-...-id.")
        parser.add_argument(
            '--local_train_run_dir', type=str, default=None,
            help="Optional local train run directory for OOD p_ik/probe training.")
        parser.add_argument(
            '--local_output_dir', type=str, default=None,
            help="Directory to write local compute outputs; required and must differ from the source files directory.")
        parser.add_argument('--num_eval_samples', type=int, default=int(1e19))
        parser.add_argument('--compute_predictive_entropy',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_ik', default=False,#True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument(
            '--compute_lightweight_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help="Train SEP on saved generation probe_features.")
        parser.add_argument(
            '--compute_icr_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help="Train and evaluate the ACL 2025 ICR Probe MLP.")
        parser.add_argument('--icr_probe_epochs', type=int, default=50)
        parser.add_argument('--icr_probe_batch_size', type=int, default=32)
        parser.add_argument('--icr_probe_learning_rate', type=float, default=5e-4)
        parser.add_argument('--icr_probe_weight_decay', type=float, default=1e-5)
        parser.add_argument(
            '--icr_probe_validation_fraction', type=float, default=0.2,
            help="Train-only holdout fraction used for scheduler/checkpoint selection.")
        parser.add_argument(
            '--icr_probe_device', default='auto',
            help="ICR MLP training device: auto, cpu, cuda, or cuda:N.")
        parser.add_argument(
            '--compute_vib_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train the 2026 VIB-Probe detector from pre-o_proj attention "
                "head outputs collected with --collect_vib_probe."))
        parser.add_argument(
            '--compute_raw_vib_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train an encoder-free affine probe directly on flattened "
                "vib_attention_head_outputs."))
        parser.add_argument('--raw_vib_probe_epochs', type=int, default=100)
        parser.add_argument('--raw_vib_probe_batch_size', type=int, default=64)
        parser.add_argument(
            '--raw_vib_probe_learning_rate', type=float, default=1e-3)
        parser.add_argument(
            '--raw_vib_probe_weight_decay', type=float, default=1e-4)
        parser.add_argument(
            '--raw_vib_probe_validation_fraction', type=float, default=0.2)
        parser.add_argument('--raw_vib_probe_patience', type=int, default=12)
        parser.add_argument('--raw_vib_probe_min_delta', type=float, default=1e-4)
        parser.add_argument('--raw_vib_probe_device', default='auto')
        parser.add_argument(
            '--compute_raw_vib_shallow_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train a one-hidden-layer MLP directly on flattened "
                "vib_attention_head_outputs, without a VIB bottleneck."))
        parser.add_argument('--raw_vib_shallow_epochs', type=int, default=50)
        parser.add_argument('--raw_vib_shallow_batch_size', type=int, default=64)
        parser.add_argument(
            '--raw_vib_shallow_learning_rate', type=float, default=1e-4)
        parser.add_argument(
            '--raw_vib_shallow_weight_decay', type=float, default=1e-4)
        parser.add_argument(
            '--raw_vib_shallow_validation_fraction', type=float, default=0.2)
        parser.add_argument('--raw_vib_shallow_patience', type=int, default=8)
        parser.add_argument('--raw_vib_shallow_min_delta', type=float, default=1e-4)
        parser.add_argument('--raw_vib_shallow_hidden_dim', type=int, default=128)
        parser.add_argument('--raw_vib_shallow_dropout', type=float, default=0.1)
        parser.add_argument('--raw_vib_shallow_device', default='auto')
        parser.add_argument(
            '--compute_vib_size_ablation', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train two controlled lower-parameter VIB input-adapter "
                "ablations from the same cached head-output features."))
        parser.add_argument(
            '--compute_vib_input_ablation', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Compare the original all-layer/all-head VIB input against "
                "the probe's last_token_h under the same VIB training setup."))
        parser.add_argument(
            '--compute_vib_mixture_ablation', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Compare last-token standard VIB, Gaussian-mixture prior VIB, "
                "and mixture-responsibility risk readout VIB."))
        parser.add_argument(
            '--vib_mixture_components', type=int, default=5,
            help="Number of unsupervised Gaussian risk prototypes.")
        parser.add_argument(
            '--vib_mixture_mean_init_scale', type=float, default=0.05,
            help="Standard deviation used to break mixture-mean symmetry.")
        parser.add_argument(
            '--vib_size_ablation_variants', nargs='+',
            choices=('head_projection_16', 'factorized_8x8x32'),
            default=['head_projection_16', 'factorized_8x8x32'],
            help=(
                "Reduced VIB adapters: shared per-head 128->16 projection, "
                "or learned feature/head/layer factorization."))
        parser.add_argument('--vib_probe_epochs', type=int, default=50)
        parser.add_argument('--vib_probe_batch_size', type=int, default=32)
        parser.add_argument('--vib_probe_learning_rate', type=float, default=2e-5)
        parser.add_argument('--vib_probe_weight_decay', type=float, default=1e-4)
        parser.add_argument('--vib_probe_beta', type=float, default=3e-4)
        parser.add_argument('--vib_probe_beta_warmup_fraction', type=float, default=0.2)
        parser.add_argument('--vib_probe_validation_fraction', type=float, default=0.2)
        parser.add_argument('--vib_probe_patience', type=int, default=8)
        parser.add_argument('--vib_probe_min_delta', type=float, default=1e-4)
        parser.add_argument('--vib_probe_device', default='auto')
        parser.add_argument(
            '--vib_probe_correctness_threshold', type=float, default=0.5,
            help=(
                "Binarize task correctness for the paper's hallucination label; "
                "accuracy below this value is y=1 (error)."))
        parser.add_argument(
            '--compute_lrp_probe', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train Reading Between the Lines latent-representation "
                "abstention probes from --collect_lrp_probe features."))
        parser.add_argument(
            '--lrp_variants', nargs='+',
            choices=(
                'concat_hidden', 'visual_attention', 'ensemble_hidden',
                'concat_attention', 'ensemble_attention',
            ),
            default=[
                'concat_hidden', 'visual_attention', 'ensemble_hidden',
                'concat_attention', 'ensemble_attention',
            ],
            help="LRP designs to train; the paper reports all five variants.")
        parser.add_argument('--lrp_epochs', type=int, default=30)
        parser.add_argument('--lrp_batch_size', type=int, default=32)
        parser.add_argument('--lrp_learning_rate', type=float, default=1e-3)
        parser.add_argument('--lrp_weight_decay', type=float, default=1e-4)
        parser.add_argument('--lrp_validation_fraction', type=float, default=0.2)
        parser.add_argument('--lrp_patience', type=int, default=6)
        parser.add_argument('--lrp_min_delta', type=float, default=1e-4)
        parser.add_argument(
            '--lrp_mlp_hidden_dims', nargs=3, type=int, default=[256, 128, 32])
        parser.add_argument('--lrp_transformer_model_dim', type=int, default=64)
        parser.add_argument('--lrp_transformer_layers', type=int, default=4)
        parser.add_argument('--lrp_transformer_heads', type=int, default=4)
        parser.add_argument('--lrp_dropout', type=float, default=0.1)
        parser.add_argument('--lrp_top_k_layers', type=int, default=5)
        parser.add_argument('--lrp_device', default='auto')
        parser.add_argument(
            '--probe_feature_name', type=str, default='argus_h_mean',
            help="Probe feature vector to use, e.g. argus_h_mean or numeric_token_h_mean.")
        parser.add_argument(
            '--compute_probe_sep_cv', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Tune the original single-view SEP logistic C by train-only "
                "CV and save it as probe_sep_cv."))
        parser.add_argument(
            '--compute_probe_sep_pca_cv', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Tune PCA dimension and logistic C for the original single "
                "SEP view and save it as probe_sep_pca_cv."))
        parser.add_argument(
            '--compute_probe_sep_v2', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Train the multi-view IoU-aware PCA/CV probe with out-of-fold "
                "stacking and save it as probe_sep_v2."))
        parser.add_argument(
            '--compute_probe_sep_v2_4view', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Run the SEP-v2 view ablation without late_h_mean and "
                "numeric_token_h_mean, saved as probe_sep_v2_4view."))
        parser.add_argument(
            '--compute_probe_sep_v2_2view', default=False,
            action=argparse.BooleanOptionalAction,
            help=(
                "Run SEP-v2 using only last_token_h and delta_last_first_h, "
                "saved as probe_sep_v2_2view."))
        parser.add_argument(
            '--probe_v2_feature_names', nargs='+',
            default=[
                'argus_h_mean',
                'late_h_mean',
                'last_token_h',
                'delta_last_first_h',
                'token_h_std',
                'numeric_token_h_mean',
            ],
            help="Saved hidden-state views used by probe_sep_v2.")
        parser.add_argument(
            '--probe_v2_pca_dims', nargs='+', type=int,
            default=[32, 64, 128],
            help="PCA dimensions selected by train-only CV for each v2 view.")
        parser.add_argument(
            '--probe_v2_c_values', nargs='+', type=float,
            default=[0.001, 0.01, 0.1, 1.0, 10.0],
            help="Logistic-regression C grid for v2 base and stacking heads.")
        parser.add_argument(
            '--probe_v2_ridge_alphas', nargs='+', type=float,
            default=[0.1, 1.0, 10.0, 100.0],
            help="Ridge alpha grid for the continuous 1-IoU v2 heads.")
        parser.add_argument(
            '--probe_v2_cv_folds', type=int, default=5,
            help="Number of train-only stratified folds used by probe_sep_v2.")
        parser.add_argument('--compute_p_ik_answerable', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_context_entails_response', default=False,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--analyze_run', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--assign_new_wandb_id', default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--restore_entity_eval', type=str, default=entity)
        parser.add_argument('--restore_entity_train', type=str, default=entity)
        parser.add_argument('--condition_on_question',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--strict_entailment',
                            default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument("--entailment_model", default='deberta', type=str)
        parser.add_argument(
            "--entailment_cache_id", default=None, type=str,
            help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
        parser.add_argument('--entailment_cache_only', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_p_true_in_compute_stage',
                            default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--reuse_entailment_model',
                            default=False, action=argparse.BooleanOptionalAction,
                            help='Use entailment model as p_true model.')
    return parser


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def construct_fewshot_prompt_from_indices_multimodal(
        dataset, example_indices, brief, brief_always, make_prompt, image_key: str = "image"):
    """Few-shot prompt constructor for multimodal (image+text) datasets.

    This mirrors `construct_fewshot_prompt_from_indices` but is explicitly
    designed for datasets that also contain an image field, e.g. TextVQA / VQA:

        example = {
            "question": <str>,
            "context": <str or None>,
            "image": <PIL.Image | np.ndarray | torch.Tensor | str path>,
            "answers": {"text": [<str>, ...]},
            "id": <str>,
        }

    Note:
        The image itself is **not** serialized into the textual few-shot prompt.
        It is passed separately to the model (e.g. Qwen3‑VL) at generation time.
        This function only prepares the textual part (question, optional context,
        reference answer) in the same format as the single‑modal version.
    """

    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:
        example = dataset[example_index]
        # Text fields.
        context = example.get("context", "")
        question = example["question"]
        if "answers" in example:
            if example["answers"] is None:#ScienceQ,vqav2的测试集均为Nnone
                answer = ""
            else:
                answer = example["answers"][0]
        elif "answer" in  example:
            answer = example["answer"]
        else:
            answer = ""

        # Image is intentionally unused in the textual few-shot prompt; it will
        # be provided to the multimodal model as a separate input tensor/object.
        _ = example.get(image_key, None)
        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices

def split_vlm_dataset(dataset): # 暂时设置为整个数据集
    """Get indices of answerable and unanswerable questions."""
    def clen(ex):

        return len(ex["answers"])

    # answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    # unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    answerable_indices = [i for i in range(len(dataset))]
    unanswerable_indices = []
    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        predicted_answer, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)


def get_gpt_metric(metric_name):

    model_name = '_'.join(metric_name.split('_')[1:])

    class EntailmentGPT():
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}
    return reference


def init_model(args):
    mn = args.model_name
    if 'llama' in mn.lower() or 'falcon' in mn or 'mistral' in mn.lower():
        model = HuggingfaceModel(
            mn, stop_sequences='default',
            max_new_tokens=args.model_max_new_tokens)
    elif 'qwen' in mn.lower():
        icr_score_config = None
        if getattr(args, 'collect_icr_probe', False):
            icr_score_config = ICRScoreConfig(
                top_k=args.icr_top_k,
                top_p=args.icr_top_p,
                pooling=args.icr_attention_pooling,
                attention_scope=args.icr_attention_scope,
                attention_uniform=args.icr_attention_uniform,
                hidden_uniform=args.icr_hidden_uniform,
                use_induction_head=args.icr_use_induction_head,
                skew_threshold=args.icr_skew_threshold,
                entropy_threshold=args.icr_entropy_threshold,
                save_direction_vectors=args.icr_save_direction_vectors,
            )
            icr_score_config.validate()
        model = QwenVLModel(
            mn,
            max_new_tokens=args.model_max_new_tokens,
            stop_sequences='default',
            icr_score_config=icr_score_config,
            icr_device=getattr(args, 'icr_device', None),
            collect_vib_probe=getattr(args, 'collect_vib_probe', False),
            collect_lrp_probe=getattr(args, 'collect_lrp_probe', False),
            collect_primary_trajectories=bool(getattr(args, 'collect_primary_trajectories', False)),
        )
        if getattr(args, 'vib_mitigation_checkpoint', None):
            model.setup_vib_mitigation(
                args.vib_mitigation_checkpoint,
                threshold_logit=args.vib_mitigation_threshold_logit,
                top_fraction=args.vib_mitigation_top_fraction,
                strength=args.vib_mitigation_strength,
                minimum_scale=args.vib_mitigation_minimum_scale,
            )
        if getattr(args, 'compute_lang_align', False):
            lang_align_max_new = (
                args.lang_align_max_new_tokens or args.model_max_new_tokens
            )
            model.setup_lang_align(LangAlignConfig(
                max_new_tokens=lang_align_max_new,
                lang_align_use_no_image=not args.no_lang_align_no_image_chain,
                uncertainty_reduction=args.lang_align_uncertainty_reduction,
                blur_epsilon=args.lang_align_blur_epsilon,
                display_output=False,
                attention_debug=args.attention_debug,
                attention_debug_dir=args.attention_debug_dir,
                attention_debug_max_samples=args.attention_debug_max_samples,
            ))
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


def get_make_prompt(args):
    if args.prompt_type == 'grounding':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if context:
                prompt += f"Context: {context}\n"
            prompt += (
                "Locate the object described below in the image. "
                "Return only one normalized bounding box in xyxy format, exactly like "
                "[x1, y1, x2, y2]. All four numbers must be in [0, 1].\n"
            )
            prompt += f"Description: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += "Answer:"
            return prompt
    elif args.prompt_type == 'default':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    else:
        raise ValueError

    return make_prompt

def get_metric(metric):
    if metric == 'squad':
        # squad_metric = load("squad_v2")
        squad_metric = load(os.getenv("SQUAD_METRIC_PATH", "squad_v2"))
        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    elif metric == 'llm_gpt-3.5':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4':
        metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric

def vqa_match(pred_sentence: str, answer_phrase: str) -> int:
    """
    短语在句子里就返回1，否则0
    不区分大小写，忽略前后空格
    """
    s = pred_sentence.strip().lower()
    a = answer_phrase.strip().lower()
    return 1 if a in s else 0


def count_matched_answers(pred_sentence: str, answers: list) -> int:
    """
    统计预测句子能匹配到的答案列表中的数量
    :param pred_sentence: 模型预测的句子（字符串）
    :param answers: 标准答案列表（元素为字符串）
    :return: 匹配成功的答案数量（整数）
    """
    # 空列表直接返回0
    if not answers:
        return 0

    # 遍历答案列表，累加匹配成功的数量
    matched_count = 0
    for ans in answers:
        matched_count += vqa_match(pred_sentence, ans)

    return matched_count

# 这里的答案还是需要按照语义匹配的方式进行
def get_metric_vlm(metric):
    """Get accuracy metric for VLM / multimodal setups (e.g. VQA).

    For non-VLM metrics, this simply falls back to `get_metric` so that
    existing behaviours (SQuAD / LLM metrics) remain unchanged.
    """

    if metric == 'grounding':
        def grounding_metric(predicted_answer, example, *args, **kwargs):
            return grounding_u.evaluate_grounding_prediction(
                predicted_answer,
                example,
            )["accuracy"]

        return grounding_metric

    # VQA-style exact-match over a set of reference answers
    if metric in {'vqa', 'vqa_soft'}:
        preserve_soft_score = metric == 'vqa_soft'
        def vqa_metric(predicted_answer, example, *args, **kwargs):
            """
            VQA 官方标准指标计算
            :param predicted_answer: 模型预测的答案 (str)
            :param example: 数据集样本，包含 'answers' 字段（10个标注答案）
            :return: 0~1 之间的准确率分数
            """
            # ---------------------- 1. 提取标注答案 ----------------------
            # example['answers'] 是列表，每个元素是字典，包含 'answer' 字段
            # 格式示例：[{'answer': 'yes'}, {'answer': 'yes'}, ...]
            reference_answers = []
            if "answers" in example:
                if isinstance(example["answers"],list):
                    for ans_ in example['answers']:
                        if isinstance(ans_,dict):#对vqa类型数据，忽略了answer等类型嗯
                            ref_ans = ans_["answer"].strip()
                        elif isinstance(ans_,str):#textQA,clearQA,GQA
                            ref_ans = ans_.strip()
                        reference_answers.append(ref_ans)
            elif "answer" in example:#针对mmmu,scienceQA
                ref_ans = str(example["answer"]).strip()
                reference_answers = [ref_ans]*10


            # ---------------------- 2. 答案标准化（VQA官方标准） ----------------------
            def normalize_answer(s):
                """文本标准化：小写、去标点、去冠词、去多余空格"""
                # 小写
                s = s.lower()
                # 去除标点符号
                s = s.translate(str.maketrans('', '', string.punctuation))
                # 移除冠词 a/an/the
                s = re.sub(r'\b(a|an|the)\b', ' ', s)
                # 移除多余空格
                s = re.sub(r'\s+', ' ', s).strip()
                return s

            # 标准化预测答案
            pred_norm = normalize_answer(predicted_answer)
            # 标准化所有标注答案
            ref_norm_list = [normalize_answer(ans) for ans in reference_answers]

            # ---------------------- 3. 计算匹配数量 ----------------------
            match_count = count_matched_answers(pred_norm,ref_norm_list)

            # ---------------------- 4. VQA官方计分规则 ----------------------
            # 得分 = min(匹配数 / 3, 1) → 至少匹配3个即为满分1.0
            score = min(match_count / 3.0, 1)
            if not preserve_soft_score and score < 1:
                score = 0

            return score

        return vqa_metric

    return get_metric(metric)


def save(object, file):
    with open(f'{wandb.run.dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{file}')
