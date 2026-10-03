"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
from dataclasses import replace
import glob
import logging
import os, gc
import time
# os.environ["WANDB_MODE"] = "offline"
import pickle
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
try:
    import wandb
except ImportError:  # pragma: no cover - local-only fallback
    from uncertainty.utils import wandb_stub as wandb

from analyze_results import analyze_run, filter_reportable_uncertainty_measures
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.lightweight_probe import fit_sep_and_score
from uncertainty.uncertainty_measures.icr_probe import (
    ICRProbeTrainingConfig,
    fit_icr_probe_and_score,
    save_icr_probe_checkpoint,
)
from uncertainty.uncertainty_measures.vib_probe import (
    VIB_FEATURE_NAME,
    VIB_LAST_TOKEN_FEATURE_NAME,
    VIB_SIZE_ABLATION_PRESETS,
    VIBProbeTrainingConfig,
    fit_vib_probe_and_score,
    save_vib_probe_checkpoint,
)
from uncertainty.uncertainty_measures.raw_vib_probe import (
    RawVIBLinearProbeConfig,
    fit_raw_vib_linear_probe_and_score,
    fit_raw_vib_shallow_probe_and_score,
    save_raw_vib_linear_probe_checkpoint,
)
from uncertainty.uncertainty_measures.lrp_probe import (
    LRPTrainingConfig,
    fit_lrp_probes_and_score,
    save_lrp_probe_checkpoint,
)
from uncertainty.uncertainty_measures.probe_cv_baselines import (
    fit_sep_cv_and_score,
    fit_sep_pca_cv_and_score,
)
from uncertainty.uncertainty_measures.sep_v2_probe import fit_sep_v2_and_score
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils
from uncertainty.utils.generation_checkpoints import (
    validate_lgd_reference_sidecar,
)


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'
DISABLED_GROUNDING_MEASURES = {
    'grounding_u_coord',
    'grounding_u_cons',
    'grounding_u_coord_nll',
    'grounding_u_coord_entropy',
}


def _metric_name_for_generation_artifacts(configured_metric, generations):
    """Infer grounding accuracy from saved fields when compute has no dataset arg."""
    first_generation = next(iter(generations.values()), {})
    first_metadata = first_generation.get('sample_metadata', {})
    first_answer = first_generation.get('most_likely_answer', {})
    if (
        isinstance(first_metadata, dict) and 'bbox' in first_metadata
    ) or first_answer.get('grounding_eval') is not None:
        return 'grounding'
    return configured_metric


def _format_elapsed_time(seconds):
    """Format a wall-clock duration for concise training logs."""
    seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:d}h {minutes:02d}m {whole_seconds:02d}s'
    if minutes:
        return f'{minutes:d}m {whole_seconds:02d}s'
    return f'{seconds:.2f}s'


def _synchronize_cuda_for_timing():
    """Synchronize queued CUDA work so wall-clock timings are accurate."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (RuntimeError, AssertionError):
        # CPU-only/partially configured CUDA environments can report an
        # available runtime before device initialization succeeds. Timing must
        # not change the behavior of the underlying fitting method.
        logging.debug('CUDA synchronization unavailable for training timer.')


def _timed_training(
    training_times,
    method_name,
    function,
    *args,
    _skip_exceptions=(),
    **kwargs,
):
    """Run one fitting routine and record/log its wall-clock duration."""
    if method_name in training_times:
        raise ValueError(f'Duplicate training timer name: {method_name}')
    logging.info('TRAINING START | method=%s', method_name)
    _synchronize_cuda_for_timing()
    start = time.perf_counter()
    try:
        output = function(*args, **kwargs)
    except Exception as exc:
        _synchronize_cuda_for_timing()
        elapsed = time.perf_counter() - start
        skipped = isinstance(exc, tuple(_skip_exceptions))
        training_times[method_name] = {
            'status': 'skipped' if skipped else 'failed',
            'seconds': float(elapsed),
            'formatted': _format_elapsed_time(elapsed),
        }
        if skipped:
            logging.warning(
                'TRAINING SKIPPED | method=%s | elapsed=%s (%.3f seconds) '
                '| reason=%s',
                method_name,
                training_times[method_name]['formatted'],
                elapsed,
                exc,
            )
        else:
            logging.exception(
                'TRAINING FAILED | method=%s | elapsed=%s (%.3f seconds)',
                method_name,
                training_times[method_name]['formatted'],
                elapsed,
            )
        raise
    _synchronize_cuda_for_timing()
    elapsed = time.perf_counter() - start
    training_times[method_name] = {
        'status': 'completed',
        'seconds': float(elapsed),
        'formatted': _format_elapsed_time(elapsed),
    }
    logging.info(
        'TRAINING COMPLETE | method=%s | elapsed=%s (%.3f seconds)',
        method_name,
        training_times[method_name]['formatted'],
        elapsed,
    )
    return output


def _attach_training_time(metadata, training_times, method_name):
    """Copy one timing record into a method's metadata dictionary."""
    if metadata is None or method_name not in training_times:
        return
    record = training_times[method_name]
    metadata['training_time_status'] = record['status']
    metadata['training_time_seconds'] = record['seconds']
    metadata['training_time_formatted'] = record['formatted']
    metadata['training_time_scope'] = (
        'fit_and_eval_score_excludes_checkpoint_serialization'
    )


def _vib_probability_metrics(error_targets, scores):
    """Return compact label-only reporting metrics for one VIB ablation."""
    targets = np.asarray(error_targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.shape != probabilities.shape:
        raise ValueError("VIB ablation eval targets and scores must align.")
    if not np.isfinite(probabilities).all():
        raise ValueError("VIB ablation scores must be finite.")
    metrics = {
        'brier': float(np.mean((probabilities - targets) ** 2)),
        'mean_error_probability': float(np.mean(probabilities)),
    }
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    metrics['nll'] = float(np.mean(
        -targets * np.log(clipped) - (1 - targets) * np.log(1.0 - clipped)
    ))
    expected_calibration_error = 0.0
    bin_indices = np.minimum((clipped * 10).astype(np.int64), 9)
    for bin_index in range(10):
        mask = bin_indices == bin_index
        if np.any(mask):
            expected_calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(clipped[mask])) - float(np.mean(targets[mask]))
            )
    metrics['ece_10_bin'] = float(expected_calibration_error)
    if np.unique(targets).size == 2:
        metrics['auroc'] = float(roc_auc_score(targets, probabilities))
        metrics['auprc'] = float(average_precision_score(targets, probabilities))
    else:
        metrics['auroc'] = float('nan')
        metrics['auprc'] = float('nan')
    return metrics


def _vib_input_ablation_size_fields(input_shape, parameter_count, baseline=None):
    """Compute input/parameter reductions against the multi-head row."""
    input_elements = int(np.prod(input_shape))
    parameter_count = int(parameter_count)
    if input_elements < 1 or parameter_count < 1:
        raise ValueError("VIB input size and parameter count must be positive.")
    baseline = baseline or {
        'input_elements': input_elements,
        'parameters': parameter_count,
    }
    baseline_input_elements = int(baseline['input_elements'])
    baseline_parameters = int(baseline['parameters'])
    if baseline_input_elements < 1 or baseline_parameters < 1:
        raise ValueError("VIB multi-head baseline sizes must be positive.")
    return {
        'input_elements': input_elements,
        'parameters': parameter_count,
        'input_reduction_percent': 100.0 * (
            1.0 - input_elements / baseline_input_elements
        ),
        'parameter_reduction_percent': 100.0 * (
            1.0 - parameter_count / baseline_parameters
        ),
    }


def _log_training_time_summary(training_times):
    """Print a compact, descending wall-clock training-time table."""
    if not training_times:
        logging.info('TRAINING TIME SUMMARY | no trainable methods were requested.')
        return
    completed_seconds = sum(
        row['seconds'] for row in training_times.values()
        if row['status'] == 'completed'
    )
    logging.info(
        'TRAINING TIME SUMMARY | methods=%d | completed_total=%s '
        '(%.3f seconds)',
        len(training_times),
        _format_elapsed_time(completed_seconds),
        completed_seconds,
    )
    logging.info('%-42s %-10s %14s', 'method', 'status', 'elapsed')
    for method_name, row in sorted(
        training_times.items(),
        key=lambda item: item[1]['seconds'],
        reverse=True,
    ):
        logging.info(
            '%-42s %-10s %14s',
            method_name,
            row['status'],
            row['formatted'],
        )


def _as_files_dir(run_dir: str) -> str:
    """Return the wandb files directory for a local run path."""
    if os.path.basename(os.path.normpath(run_dir)) == 'files':
        return run_dir
    # Locally augmented generation artifacts live directly in their output
    # directory, without W&B's extra ``files`` level.
    if any(os.path.isfile(os.path.join(run_dir, name)) for name in
           (EXP_DETAILS, 'validation_generations.pkl', 'train_generations.pkl')):
        return run_dir
    return os.path.join(run_dir, 'files')


def _resolve_local_run_dir(local_wandb_dir: str, runid: str) -> str:
    """Find a local wandb run directory by short run id."""
    if not local_wandb_dir:
        raise ValueError(
            "Local wandb mode requires --local_wandb_dir or an explicit "
            "--local_eval_run_dir/--local_train_run_dir.")
    candidates = sorted(glob.glob(os.path.join(local_wandb_dir, f'*-{runid}')))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(local_wandb_dir, f'*{runid}*')))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find local wandb run for id `{runid}` under `{local_wandb_dir}`.")
    if len(candidates) > 1:
        logging.warning(
            "Multiple local runs matched id `%s`; using `%s`.",
            runid,
            candidates[-1],
        )
    return candidates[-1]


def _pickle_load(path: str):
    _install_pil_pickle_compat()
    with open(path, 'rb') as infile:
        return pickle.load(infile)


def _pickle_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as outfile:
        pickle.dump(obj, outfile)


def _install_pil_pickle_compat() -> None:
    """Allow old run pickles with newer PIL Image state tuples to load."""
    try:
        from PIL import Image
    except ImportError:
        return

    if getattr(Image.Image, "_semunc_pickle_compat", False):
        return

    original_setstate = Image.Image.__setstate__

    def compat_setstate(self, state):
        if isinstance(state, (tuple, list)) and len(state) > 5:
            state = state[:5]
        return original_setstate(self, state)

    Image.Image.__setstate__ = compat_setstate
    Image.Image._semunc_pickle_compat = True


def _load_and_align_lgd_reference_sidecar(path, generations, *, split):
    """Load portable auxiliary samples and verify generations.pkl correspondence."""
    payload = _pickle_load(path)
    sample_ids = list(generations)
    summary = validate_lgd_reference_sidecar(
        payload,
        expected_split=split,
        expected_sample_ids=sample_ids,
    )
    verified_embedded = 0
    attached_from_sidecar = 0
    samples_without_lgd = 0
    identity_fields = (
        "draw",
        "seed",
        "response",
        "model",
        "backend",
        "provider",
        "excluded_from_uncertainty",
        "error_kind",
    )
    for sample_id in sample_ids:
        sidecar_records = payload["samples"][sample_id]
        lgd_result = generations[sample_id].get("lgd_uq")
        if not isinstance(lgd_result, dict):
            samples_without_lgd += 1
            continue
        sampling = lgd_result.setdefault("sampling", {})
        auxiliary_cache = sampling.get("auxiliary_cache", {})
        portable_sidecar = auxiliary_cache.get("portable_sidecar")
        if portable_sidecar not in {None, os.path.basename(path)}:
            raise ValueError(
                f"LGD sidecar pointer mismatch for sample {sample_id!r}: "
                f"{portable_sidecar!r} != {os.path.basename(path)!r}."
            )
        sample_key = auxiliary_cache.get("sample_key")
        if sample_key not in {None, sample_id}:
            raise ValueError(
                f"LGD sidecar sample key mismatch for {sample_id!r}: "
                f"{sample_key!r}."
            )
        embedded_records = sampling.get("auxiliary")
        if not isinstance(embedded_records, dict):
            sampling["auxiliary"] = sidecar_records
            attached_from_sidecar += 1
            continue
        if set(embedded_records) != set(sidecar_records):
            raise ValueError(
                f"Embedded LGD model keys disagree with {path!r} for sample "
                f"{sample_id!r}."
            )
        for model_name, expected_records in sidecar_records.items():
            actual_records = embedded_records[model_name]
            if len(actual_records) != len(expected_records):
                raise ValueError(
                    f"Embedded LGD draw count disagrees with {path!r} for "
                    f"{sample_id!r}/{model_name!r}."
                )
            for actual, expected in zip(actual_records, expected_records):
                if any(
                    actual.get(field) != expected.get(field)
                    for field in identity_fields
                ):
                    raise ValueError(
                        f"Embedded LGD record disagrees with {path!r} for "
                        f"{sample_id!r}/{model_name!r}/draw "
                        f"{expected.get('draw')!r}."
                    )
        verified_embedded += 1
    return payload, {
        **summary,
        "filename": os.path.basename(path),
        "sample_order_matches_generations": True,
        "num_embedded_samples_verified": verified_embedded,
        "num_samples_attached_from_sidecar": attached_from_sidecar,
        "num_samples_without_lgd": samples_without_lgd,
    }


def main(args):

    training_times = {}

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.getenv('USER', 'ceor')
    scratch_dir = os.getenv('SCRATCH_DIR', 'experiments')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    local_mode = bool(args.use_local_wandb or args.local_eval_run_dir or args.local_wandb_dir)
    local_eval_run_dir = None
    local_eval_files_dir = None
    local_output_dir = None

    if local_mode:
        local_eval_run_dir = (
            args.local_eval_run_dir
            or _resolve_local_run_dir(args.local_wandb_dir, args.eval_wandb_runid)
        )
        local_eval_files_dir = _as_files_dir(local_eval_run_dir)
        if not args.local_output_dir:
            raise ValueError("Local computation requires --local_output_dir; use a new output directory.")
        local_output_dir = args.local_output_dir
        if os.path.realpath(local_output_dir) == os.path.realpath(local_eval_files_dir):
            raise ValueError("--local_output_dir must differ from the source run files directory.")
        os.makedirs(local_output_dir, exist_ok=True)
        if args.compute_vib_size_ablation:
            vib_ablation_log_path = os.path.join(
                local_output_dir, 'vib_size_ablation.log'
            )
            file_handler = logging.FileHandler(
                vib_ablation_log_path, mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logging.info(
                'Writing VIB size-ablation log to: %s',
                vib_ablation_log_path,
            )
        if args.compute_vib_input_ablation:
            vib_input_log_path = os.path.join(
                local_output_dir, 'vib_input_ablation.log'
            )
            file_handler = logging.FileHandler(
                vib_input_log_path, mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logging.info(
                'Writing VIB input-ablation log to: %s',
                vib_input_log_path,
            )
        if args.compute_vib_mixture_ablation:
            vib_mixture_log_path = os.path.join(
                local_output_dir, 'vib_mixture_ablation.log'
            )
            file_handler = logging.FileHandler(
                vib_mixture_log_path, mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logging.info(
                'Writing VIB mixture-ablation log to: %s',
                vib_mixture_log_path,
            )
        if args.compute_raw_vib_probe:
            raw_vib_log_path = os.path.join(
                local_output_dir, 'raw_vib_linear_probe.log'
            )
            file_handler = logging.FileHandler(
                raw_vib_log_path, mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logging.info(
                'Writing Raw VIB linear-probe log to: %s', raw_vib_log_path
            )
        if args.compute_raw_vib_shallow_probe:
            raw_vib_shallow_log_path = os.path.join(
                local_output_dir, 'raw_vib_shallow_probe.log'
            )
            file_handler = logging.FileHandler(
                raw_vib_shallow_log_path, mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)-8s %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
            logging.info(
                'Writing Raw VIB shallow-probe log to: %s',
                raw_vib_shallow_log_path,
            )
        logging.info('Using local wandb eval run directory: %s', local_eval_run_dir)
        logging.info('Writing local compute outputs to: %s', local_output_dir)
        if wandb.run is None:
            wandb.init(mode="disabled")

        def restore(filename):
            path = os.path.join(local_eval_files_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing local artifact `{path}`.")

            class Restored:
                name = path

            return Restored

        def save_local(obj, filename):
            path = os.path.join(local_output_dir, filename)
            _pickle_save(obj, path)
            logging.info('Saved `%s`.', path)
    elif args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    def restore_optional(filename):
        """Restore a new sidecar while remaining compatible with older runs."""
        try:
            restored = restore(filename)
            if not os.path.exists(restored.name):
                raise FileNotFoundError(restored.name)
            return restored
        except Exception as exc:  # W&B uses provider-specific missing-file errors.
            logging.warning(
                "Optional artifact %r is unavailable; falling back to records "
                "embedded in generations.pkl: %s",
                filename,
                exc,
            )
            return None

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        filename = 'train_generations.pkl'
        if local_mode:
            local_train_run_dir = (
                args.local_train_run_dir
                or _resolve_local_run_dir(args.local_wandb_dir, args.train_wandb_runid)
            )
            train_path = os.path.join(_as_files_dir(local_train_run_dir), filename)
            train_generations = _pickle_load(train_path)
            wandb.config.update(
                {"ood_training_set": os.path.basename(local_train_run_dir)},
                allow_val_change=True)
        else:
            api = wandb.Api()
            old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
            old_run_train.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)
            train_generations = _pickle_load(f'{wandb.run.dir}/{filename}')
            wandb.config.update(
                {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if (
            args.compute_p_ik
            or args.compute_p_ik_answerable
            or args.compute_lightweight_probe
            or args.compute_icr_probe
            or args.compute_vib_probe
            or args.compute_vib_size_ablation
            or args.compute_vib_input_ablation
            or args.compute_vib_mixture_ablation
            or args.compute_raw_vib_probe
            or args.compute_raw_vib_shallow_probe
            or args.compute_lrp_probe
            or args.compute_probe_sep_cv
            or args.compute_probe_sep_pca_cv
            or args.compute_probe_sep_v2
            or args.compute_probe_sep_v2_4view
            or args.compute_probe_sep_v2_2view
        ):
            train_generations_pickle = restore('train_generations.pkl')
            train_generations = _pickle_load(train_generations_pickle.name)

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')

    if args.compute_p_true_in_compute_stage:
        # This is usually not called.
        from uncertainty.data.data_utils import load_ds

        old_exp = restore(EXP_DETAILS)
        old_exp = _pickle_load(old_exp.name)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt_vlm(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric_vlm(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')

    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    result_dict = _pickle_load(result_dict_pickle.name)
    # These grounding diagnostics are no longer recorded. Remove stale values
    # as well when recomputing an older generation run.
    for disabled_measure in DISABLED_GROUNDING_MEASURES:
        result_dict.get('uncertainty_measures', {}).pop(disabled_measure, None)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    validation_generations = _pickle_load(validation_generations_pickle.name)
    validation_lgd_sidecar = restore_optional(
        'validation_lgd_references.pkl'
    )
    if validation_lgd_sidecar is not None:
        _, sidecar_summary = _load_and_align_lgd_reference_sidecar(
            validation_lgd_sidecar.name,
            validation_generations,
            split='validation',
        )
        result_dict.setdefault('lgd_reference_sidecars', {})[
            'validation'
        ] = sidecar_summary
        logging.info(
            "Loaded aligned validation LGD reference sidecar: %s",
            sidecar_summary,
        )
    if (
        args.train_wandb_runid == args.eval_wandb_runid
        and 'train_generations' in locals()
    ):
        train_lgd_sidecar = restore_optional('train_lgd_references.pkl')
        if train_lgd_sidecar is not None:
            _, train_sidecar_summary = _load_and_align_lgd_reference_sidecar(
                train_lgd_sidecar.name,
                train_generations,
                split='train',
            )
            result_dict.setdefault('lgd_reference_sidecars', {})[
                'train'
            ] = train_sidecar_summary
            logging.info(
                "Loaded aligned train LGD reference sidecar: %s",
                train_sidecar_summary,
            )

    if args.recompute_accuracy:
        metric_name = _metric_name_for_generation_artifacts(
            args.metric, validation_generations
        )
        if metric_name == 'grounding' and args.metric != 'grounding':
            logging.info(
                'Detected grounding metadata in generation artifacts; '
                'using the grounding metric for --recompute_accuracy.'
            )
        metric = utils.get_metric_vlm(metric_name)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    validation_probe_features = []
    validation_generation_records, validation_sample_ids = [], []
    grounding_eval_records = []
    validation_grounding_ious = []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return True
        # return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']
        validation_generation_records.append(example)
        validation_sample_ids.append(tid)

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                metric_example = dict(example.get('sample_metadata', {}))
                metric_example.update(example)
                acc = metric(
                    most_likely_answer['response'], metric_example, None
                )
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        if 'probe_features' in most_likely_answer:
            validation_probe_features.append(most_likely_answer['probe_features'])
        grounding_eval = most_likely_answer.get('grounding_eval')
        if grounding_eval is not None:
            grounding_eval_records.append(grounding_eval)
        grounding_iou = (
            grounding_eval.get('iou')
            if isinstance(grounding_eval, dict) else None
        )
        try:
            grounding_iou = float(grounding_iou)
        except (TypeError, ValueError):
            grounding_iou = float('nan')
        validation_grounding_ious.append(grounding_iou)

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            # logging.info('Context:')
            # logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(str(example['answer']))####
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false
    result_dict['validation_sample_ids'] = validation_sample_ids

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))
    if grounding_eval_records:
        result_dict['grounding_eval'] = grounding_eval_records
    # Preserve sample alignment for correlation analysis.  The compact
    # grounding_eval list omits missing records and therefore cannot safely be
    # zipped with every uncertainty vector.
    result_dict['validation_grounding_ious'] = validation_grounding_ious

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for embedding classification.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if (
        args.compute_lightweight_probe
        or args.compute_icr_probe
        or args.compute_vib_probe
        or args.compute_vib_size_ablation
        or args.compute_vib_input_ablation
        or args.compute_vib_mixture_ablation
        or args.compute_raw_vib_probe
        or args.compute_raw_vib_shallow_probe
        or args.compute_lrp_probe
        or args.compute_probe_sep_cv
        or args.compute_probe_sep_pca_cv
        or args.compute_probe_sep_v2
        or args.compute_probe_sep_v2_4view
        or args.compute_probe_sep_v2_2view
    ):
        train_is_true, train_probe_features, train_generation_records = [], [], []
        train_grounding_errors = []
        for tid in train_generations:
            train_generation = train_generations[tid]
            most_likely_answer = train_generation['most_likely_answer']
            probe_features = most_likely_answer.get('probe_features')
            if probe_features is None:
                raise ValueError(
                    "Training generations do not contain probe_features. "
                    "Run generate_answers.py with --collect_probe_features.")
            train_probe_features.append(probe_features)
            train_generation_records.append(train_generation)
            train_is_true.append(most_likely_answer['accuracy'])
            grounding_eval = most_likely_answer.get('grounding_eval')
            grounding_iou = (
                grounding_eval.get('iou')
                if isinstance(grounding_eval, dict) else None
            )
            try:
                grounding_iou = float(grounding_iou)
            except (TypeError, ValueError):
                grounding_iou = float('nan')
            train_grounding_errors.append(
                1.0 - float(np.clip(grounding_iou, 0.0, 1.0))
                if np.isfinite(grounding_iou) else float('nan')
            )
        if len(validation_probe_features) != len(validation_is_false):
            raise ValueError(
                "Validation generations do not all contain probe_features. "
                "Run generate_answers.py with --collect_probe_features.")
        train_is_false_probe = [0.0 if is_t else 1.0 for is_t in train_is_true]

        if args.compute_icr_probe:
            icr_training_config = ICRProbeTrainingConfig(
                batch_size=int(args.icr_probe_batch_size),
                num_epochs=int(args.icr_probe_epochs),
                learning_rate=float(args.icr_probe_learning_rate),
                weight_decay=float(args.icr_probe_weight_decay),
                validation_fraction=float(args.icr_probe_validation_fraction),
                random_seed=int(args.random_seed),
                device=str(args.icr_probe_device),
            )
            icr_scores, icr_metadata, icr_model = _timed_training(
                training_times,
                'icr_probe',
                fit_icr_probe_and_score,
                train_probe_features,
                train_is_false_probe,
                validation_probe_features,
                config=icr_training_config,
            )
            _attach_training_time(icr_metadata, training_times, 'icr_probe')
            result_dict['uncertainty_measures']['icr_probe'] = icr_scores
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            checkpoint_path = save_icr_probe_checkpoint(
                os.path.join(checkpoint_dir, 'icr_probe.pt'),
                icr_model,
                icr_metadata,
            )
            icr_metadata['checkpoint_file'] = os.path.basename(checkpoint_path)
            result_dict['icr_probe_metadata'] = icr_metadata
            if not local_mode:
                wandb.save(checkpoint_path)

        if (
            args.compute_vib_probe
            or args.compute_vib_size_ablation
            or args.compute_vib_input_ablation
            or args.compute_vib_mixture_ablation
            or args.compute_raw_vib_probe
            or args.compute_raw_vib_shallow_probe
        ):
            correctness_threshold = float(args.vib_probe_correctness_threshold)
            if not 0.0 <= correctness_threshold <= 1.0:
                raise ValueError(
                    "--vib_probe_correctness_threshold must be in [0, 1]."
                )
            # VIB-Probe Eq. (10) uses binary hallucination labels.  Preserve
            # LRP's soft labels separately below.
            vib_error_targets = [
                float(float(value) < correctness_threshold)
                for value in train_is_true
            ]
            vib_training_config = VIBProbeTrainingConfig(
                epochs=int(args.vib_probe_epochs),
                batch_size=int(args.vib_probe_batch_size),
                learning_rate=float(args.vib_probe_learning_rate),
                weight_decay=float(args.vib_probe_weight_decay),
                beta=float(args.vib_probe_beta),
                beta_warmup_fraction=float(args.vib_probe_beta_warmup_fraction),
                validation_fraction=float(args.vib_probe_validation_fraction),
                patience=int(args.vib_probe_patience),
                min_delta=float(args.vib_probe_min_delta),
                random_seed=int(args.random_seed),
                device=str(args.vib_probe_device),
            )

        if args.compute_vib_probe:
            vib_scores, vib_metadata, vib_model = _timed_training(
                training_times,
                'vib_probe',
                fit_vib_probe_and_score,
                train_probe_features,
                vib_error_targets,
                validation_probe_features,
                config=vib_training_config,
            )
            _attach_training_time(vib_metadata, training_times, 'vib_probe')
            vib_metadata["correctness_threshold"] = correctness_threshold
            vib_metadata["checkpoint_file"] = "vib_probe.pt"
            result_dict['uncertainty_measures']['vib_probe'] = vib_scores
            result_dict['vib_probe_metadata'] = vib_metadata
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            checkpoint_path = save_vib_probe_checkpoint(
                os.path.join(checkpoint_dir, 'vib_probe.pt'),
                vib_model,
                vib_metadata,
            )
            if not local_mode:
                wandb.save(checkpoint_path)

        if args.compute_raw_vib_probe:
            raw_vib_config = RawVIBLinearProbeConfig(
                epochs=int(args.raw_vib_probe_epochs),
                batch_size=int(args.raw_vib_probe_batch_size),
                learning_rate=float(args.raw_vib_probe_learning_rate),
                weight_decay=float(args.raw_vib_probe_weight_decay),
                validation_fraction=float(
                    args.raw_vib_probe_validation_fraction
                ),
                patience=int(args.raw_vib_probe_patience),
                min_delta=float(args.raw_vib_probe_min_delta),
                random_seed=int(args.random_seed),
                device=str(args.raw_vib_probe_device),
            )
            raw_vib_scores, raw_vib_metadata, raw_vib_model = _timed_training(
                training_times,
                'raw_vib_linear_probe',
                fit_raw_vib_linear_probe_and_score,
                train_probe_features,
                vib_error_targets,
                validation_probe_features,
                config=raw_vib_config,
            )
            _attach_training_time(
                raw_vib_metadata, training_times, 'raw_vib_linear_probe'
            )
            raw_vib_eval_targets = [
                int(float(value) < correctness_threshold)
                for value in validation_is_true
            ]
            raw_vib_metrics = _vib_probability_metrics(
                raw_vib_eval_targets, raw_vib_scores
            )
            full_vib_parameters = 152052737
            raw_vib_metadata.update({
                'correctness_threshold': correctness_threshold,
                'eval_reporting_metrics': raw_vib_metrics,
                'eval_label_usage': (
                    'reporting only; eval labels are not used for fitting, '
                    'early stopping, or checkpoint selection'
                ),
                'full_vib_reference_parameters': full_vib_parameters,
                'parameter_reduction_vs_full_vib_fraction': float(
                    1.0
                    - raw_vib_metadata['parameter_count']
                    / full_vib_parameters
                ),
                'checkpoint_file': 'raw_vib_linear_probe.pt',
            })
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            checkpoint_path = save_raw_vib_linear_probe_checkpoint(
                os.path.join(checkpoint_dir, 'raw_vib_linear_probe.pt'),
                raw_vib_model,
                raw_vib_metadata,
            )
            checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))
            raw_vib_metadata['checkpoint_size_bytes'] = checkpoint_size_bytes
            raw_vib_metadata['checkpoint_size_mib'] = float(
                checkpoint_size_bytes / (1024.0 ** 2)
            )
            result_dict['uncertainty_measures'][
                'raw_vib_linear_probe'
            ] = raw_vib_scores
            result_dict['raw_vib_probe_metadata'] = raw_vib_metadata
            layer_norm = np.asarray(
                raw_vib_metadata['layer_weight_l2'], dtype=np.float64
            )
            top_layers = np.argsort(layer_norm)[::-1][:5].astype(int).tolist()
            logging.info(
                'RAW VIB LINEAR EVAL | eval_auroc=%.4f | eval_auprc=%.4f '
                '| Brier=%.4f | NLL=%.4f | ECE=%.4f | parameters=%d '
                '| reduction_vs_full_vib=%.2f%% | checkpoint=%.2f MiB '
                '| top_layers=%s',
                raw_vib_metrics['auroc'],
                raw_vib_metrics['auprc'],
                raw_vib_metrics['brier'],
                raw_vib_metrics['nll'],
                raw_vib_metrics['ece_10_bin'],
                raw_vib_metadata['parameter_count'],
                100.0 * raw_vib_metadata[
                    'parameter_reduction_vs_full_vib_fraction'
                ],
                raw_vib_metadata['checkpoint_size_mib'],
                top_layers,
            )
            if not local_mode:
                wandb.save(checkpoint_path)
            del raw_vib_model
            gc.collect()
            torch.cuda.empty_cache()

        if args.compute_raw_vib_shallow_probe:
            raw_shallow_config = RawVIBLinearProbeConfig(
                epochs=int(args.raw_vib_shallow_epochs),
                batch_size=int(args.raw_vib_shallow_batch_size),
                learning_rate=float(args.raw_vib_shallow_learning_rate),
                weight_decay=float(args.raw_vib_shallow_weight_decay),
                validation_fraction=float(
                    args.raw_vib_shallow_validation_fraction
                ),
                patience=int(args.raw_vib_shallow_patience),
                min_delta=float(args.raw_vib_shallow_min_delta),
                random_seed=int(args.random_seed),
                device=str(args.raw_vib_shallow_device),
                hidden_dim=int(args.raw_vib_shallow_hidden_dim),
                dropout=float(args.raw_vib_shallow_dropout),
            )
            shallow_scores, shallow_metadata, shallow_model = _timed_training(
                training_times,
                'raw_vib_shallow_probe',
                fit_raw_vib_shallow_probe_and_score,
                train_probe_features,
                vib_error_targets,
                validation_probe_features,
                config=raw_shallow_config,
            )
            _attach_training_time(
                shallow_metadata, training_times, 'raw_vib_shallow_probe'
            )
            shallow_eval_targets = [
                int(float(value) < correctness_threshold)
                for value in validation_is_true
            ]
            shallow_metrics = _vib_probability_metrics(
                shallow_eval_targets, shallow_scores
            )
            full_vib_parameters = 152052737
            shallow_metadata.update({
                'correctness_threshold': correctness_threshold,
                'eval_reporting_metrics': shallow_metrics,
                'eval_label_usage': (
                    'reporting only; eval labels are not used for fitting, '
                    'early stopping, or checkpoint selection'
                ),
                'full_vib_reference_parameters': full_vib_parameters,
                'parameter_reduction_vs_full_vib_fraction': float(
                    1.0
                    - shallow_metadata['parameter_count']
                    / full_vib_parameters
                ),
                'checkpoint_file': 'raw_vib_shallow_probe.pt',
            })
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            checkpoint_path = save_raw_vib_linear_probe_checkpoint(
                os.path.join(checkpoint_dir, 'raw_vib_shallow_probe.pt'),
                shallow_model,
                shallow_metadata,
            )
            checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))
            shallow_metadata['checkpoint_size_bytes'] = checkpoint_size_bytes
            shallow_metadata['checkpoint_size_mib'] = float(
                checkpoint_size_bytes / (1024.0 ** 2)
            )
            result_dict['uncertainty_measures'][
                'raw_vib_shallow_probe'
            ] = shallow_scores
            result_dict['raw_vib_shallow_probe_metadata'] = shallow_metadata
            layer_norm = np.asarray(
                shallow_metadata['layer_weight_l2'], dtype=np.float64
            )
            top_layers = np.argsort(layer_norm)[::-1][:5].astype(int).tolist()
            logging.info(
                'RAW VIB SHALLOW EVAL | hidden_dim=%d | dropout=%.3f '
                '| eval_auroc=%.4f | eval_auprc=%.4f | Brier=%.4f '
                '| NLL=%.4f | ECE=%.4f | parameters=%d '
                '| reduction_vs_full_vib=%.2f%% | checkpoint=%.2f MiB '
                '| top_layers=%s',
                shallow_metadata['hidden_dim'],
                shallow_metadata['dropout'],
                shallow_metrics['auroc'],
                shallow_metrics['auprc'],
                shallow_metrics['brier'],
                shallow_metrics['nll'],
                shallow_metrics['ece_10_bin'],
                shallow_metadata['parameter_count'],
                100.0 * shallow_metadata[
                    'parameter_reduction_vs_full_vib_fraction'
                ],
                shallow_metadata['checkpoint_size_mib'],
                top_layers,
            )
            if not local_mode:
                wandb.save(checkpoint_path)
            del shallow_model
            gc.collect()
            torch.cuda.empty_cache()

        if args.compute_vib_size_ablation:
            eval_vib_error_targets = [
                int(float(value) < correctness_threshold)
                for value in validation_is_true
            ]
            selected_variants = list(args.vib_size_ablation_variants)
            unknown_variants = sorted(
                set(selected_variants) - set(VIB_SIZE_ABLATION_PRESETS)
            )
            if unknown_variants:
                raise ValueError(
                    f"Unknown VIB size-ablation variants {unknown_variants}."
                )
            ablation_metadata = {
                'experiment': 'vib_probe_size_ablation',
                'controlled_variables': (
                    'same cached VIB features, binary targets, split, seed, '
                    'optimizer, VIB loss, encoder, bottleneck, and early stopping; '
                    'only the input adapter changes'
                ),
                'correctness_threshold': correctness_threshold,
                'variants': {},
            }
            ablation_rows = []
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            logging.info(
                'VIB SIZE ABLATION START | variants=%s',
                ','.join(selected_variants),
            )
            for variant_name in selected_variants:
                variant_config = replace(
                    vib_training_config,
                    **VIB_SIZE_ABLATION_PRESETS[variant_name],
                )
                timer_name = f'vib_probe_size_ablation/{variant_name}'
                variant_scores, variant_metadata, variant_model = _timed_training(
                    training_times,
                    timer_name,
                    fit_vib_probe_and_score,
                    train_probe_features,
                    vib_error_targets,
                    validation_probe_features,
                    config=variant_config,
                )
                _attach_training_time(
                    variant_metadata, training_times, timer_name
                )
                eval_metrics = _vib_probability_metrics(
                    eval_vib_error_targets, variant_scores
                )
                metric_name = (
                    f'vib_probe__size_ablation__{variant_name}'
                )
                checkpoint_file = f'vib_probe_{variant_name}.pt'
                variant_metadata.update({
                    'ablation_variant': variant_name,
                    'correctness_threshold': correctness_threshold,
                    'checkpoint_file': checkpoint_file,
                    'eval_reporting_metrics': eval_metrics,
                    'eval_label_usage': (
                        'reporting only; eval labels are not used for fitting, '
                        'early stopping, or checkpoint selection'
                    ),
                })
                checkpoint_path = save_vib_probe_checkpoint(
                    os.path.join(checkpoint_dir, checkpoint_file),
                    variant_model,
                    variant_metadata,
                )
                checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))
                variant_metadata['checkpoint_size_bytes'] = checkpoint_size_bytes
                variant_metadata['checkpoint_size_mib'] = float(
                    checkpoint_size_bytes / (1024.0 ** 2)
                )
                result_dict['uncertainty_measures'][metric_name] = variant_scores
                ablation_metadata['variants'][variant_name] = variant_metadata
                row = {
                    'variant': variant_name,
                    'parameters': variant_metadata['parameter_count'],
                    'reduction_percent': 100.0 * variant_metadata[
                        'parameter_reduction_fraction'
                    ],
                    'checkpoint_mib': variant_metadata['checkpoint_size_mib'],
                    'elapsed_seconds': variant_metadata['training_time_seconds'],
                    'best_epoch': variant_metadata['best_epoch'],
                    'holdout_auroc': variant_metadata['holdout_auroc'],
                    'eval_auroc': eval_metrics['auroc'],
                    'eval_auprc': eval_metrics['auprc'],
                    'eval_brier': eval_metrics['brier'],
                }
                ablation_rows.append(row)
                logging.info(
                    'VIB SIZE ABLATION RESULT | variant=%s | parameters=%d '
                    '| reduction=%.2f%% | checkpoint=%.2f MiB | elapsed=%.3fs '
                    '| best_epoch=%d | holdout_auroc=%.6f | eval_auroc=%.6f '
                    '| eval_auprc=%.6f | eval_brier=%.6f',
                    row['variant'], row['parameters'], row['reduction_percent'],
                    row['checkpoint_mib'], row['elapsed_seconds'],
                    row['best_epoch'], row['holdout_auroc'], row['eval_auroc'],
                    row['eval_auprc'], row['eval_brier'],
                )
                if not local_mode:
                    wandb.save(checkpoint_path)
                del variant_model
                gc.collect()
                torch.cuda.empty_cache()
            ablation_metadata['comparison'] = ablation_rows
            result_dict['vib_probe_size_ablation_metadata'] = ablation_metadata
            logging.info(
                '%-28s %12s %10s %10s %9s %9s %9s',
                'VIB size variant', 'parameters', 'reduction', 'time',
                'hold_AUC', 'eval_AUC', 'Brier',
            )
            for row in ablation_rows:
                logging.info(
                    '%-28s %12d %9.2f%% %9.2fs %9.4f %9.4f %9.4f',
                    row['variant'], row['parameters'], row['reduction_percent'],
                    row['elapsed_seconds'], row['holdout_auroc'],
                    row['eval_auroc'], row['eval_brier'],
                )
            logging.info('VIB SIZE ABLATION COMPLETE')

        if args.compute_vib_input_ablation:
            eval_vib_error_targets = [
                int(float(value) < correctness_threshold)
                for value in validation_is_true
            ]
            # Only the VIB input source changes.  The last-token vector is
            # exposed as [1, 1, hidden_dim], preserving the downstream
            # encoder, bottleneck, objective, split, and optimization code.
            input_variants = {
                'multi_head': {
                    'feature_name': VIB_FEATURE_NAME,
                    'feature_layout': 'layer_head',
                    'description': (
                        'all cached pre-o_proj attention-head outputs from '
                        'every decoder layer'
                    ),
                },
                'last_token_hidden': {
                    'feature_name': VIB_LAST_TOKEN_FEATURE_NAME,
                    'feature_layout': 'vector',
                    'description': (
                        'the same last_token_h vector used by the conventional '
                        'hidden-state probe'
                    ),
                },
            }
            input_ablation_metadata = {
                'experiment': 'vib_probe_input_source_ablation',
                'hypothesis': (
                    'test whether the final-token hidden state retains the '
                    'risk signal supplied by all layer/head outputs'
                ),
                'controlled_variables': (
                    'same cached samples, binary targets, train/holdout split, '
                    'seed, optimizer, VIB BCE+KL objective, beta warmup, '
                    'downstream encoder widths, latent bottleneck, classifier, '
                    'and early stopping; only the input feature source changes'
                ),
                'necessary_architecture_difference': (
                    'the first encoder projection follows the input dimension: '
                    'L*H*d_head for multi_head versus hidden_dim for '
                    'last_token_hidden'
                ),
                'correctness_threshold': correctness_threshold,
                'variants': {},
            }
            input_rows = []
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            controlled_config = replace(
                vib_training_config,
                input_adapter='flatten',
            )
            logging.info(
                'VIB INPUT ABLATION START | variants=%s',
                ','.join(input_variants),
            )
            for variant_name, variant_spec in input_variants.items():
                timer_name = f'vib_probe_input_ablation/{variant_name}'
                variant_scores, variant_metadata, variant_model = _timed_training(
                    training_times,
                    timer_name,
                    fit_vib_probe_and_score,
                    train_probe_features,
                    vib_error_targets,
                    validation_probe_features,
                    config=controlled_config,
                    feature_name=variant_spec['feature_name'],
                    feature_layout=variant_spec['feature_layout'],
                )
                _attach_training_time(
                    variant_metadata, training_times, timer_name
                )
                eval_metrics = _vib_probability_metrics(
                    eval_vib_error_targets, variant_scores
                )
                metric_name = f'vib_probe__input_ablation__{variant_name}'
                checkpoint_file = f'vib_probe_input_{variant_name}.pt'
                variant_metadata.update({
                    'ablation_variant': variant_name,
                    'input_description': variant_spec['description'],
                    'correctness_threshold': correctness_threshold,
                    'checkpoint_file': checkpoint_file,
                    'eval_reporting_metrics': eval_metrics,
                    'eval_label_usage': (
                        'reporting only; eval labels are not used for fitting, '
                        'early stopping, or checkpoint selection'
                    ),
                })
                size_fields = _vib_input_ablation_size_fields(
                    variant_metadata['input_shape'],
                    variant_metadata['parameter_count'],
                    baseline=input_rows[0] if input_rows else None,
                )
                variant_metadata.update({
                    'input_reduction_vs_multi_head_fraction': (
                        size_fields['input_reduction_percent'] / 100.0
                    ),
                    'parameter_reduction_vs_multi_head_fraction': (
                        size_fields['parameter_reduction_percent'] / 100.0
                    ),
                })
                checkpoint_path = save_vib_probe_checkpoint(
                    os.path.join(checkpoint_dir, checkpoint_file),
                    variant_model,
                    variant_metadata,
                )
                checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))
                variant_metadata['checkpoint_size_bytes'] = checkpoint_size_bytes
                variant_metadata['checkpoint_size_mib'] = float(
                    checkpoint_size_bytes / (1024.0 ** 2)
                )
                result_dict['uncertainty_measures'][metric_name] = variant_scores
                input_ablation_metadata['variants'][variant_name] = variant_metadata
                input_rows.append({
                    'variant': variant_name,
                    'feature_name': variant_metadata['feature_name'],
                    'input_shape': variant_metadata['input_shape'],
                    **size_fields,
                    'checkpoint_mib': variant_metadata['checkpoint_size_mib'],
                    'elapsed_seconds': variant_metadata['training_time_seconds'],
                    'best_epoch': variant_metadata['best_epoch'],
                    'holdout_auroc': variant_metadata['holdout_auroc'],
                    'eval_auroc': eval_metrics['auroc'],
                    'eval_auprc': eval_metrics['auprc'],
                    'eval_brier': eval_metrics['brier'],
                    'eval_nll': eval_metrics['nll'],
                    'eval_ece': eval_metrics['ece_10_bin'],
                })
                if not local_mode:
                    wandb.save(checkpoint_path)
                del variant_model
                gc.collect()
                torch.cuda.empty_cache()

            input_ablation_metadata['comparison'] = input_rows
            result_dict['vib_probe_input_ablation_metadata'] = (
                input_ablation_metadata
            )
            logging.info(
                '%-20s %16s %12s %10s %9s %9s %9s %9s %9s',
                'VIB input', 'shape', 'parameters', 'time', 'hold_AUC',
                'eval_AUC', 'AUPRC', 'Brier', 'NLL',
            )
            for row in input_rows:
                logging.info(
                    '%-20s %16s %12d %9.2fs %9.4f %9.4f %9.4f %9.4f %9.4f '
                    '| input_reduction=%.2f%% | parameter_reduction=%.2f%% '
                    '| ECE=%.4f',
                    row['variant'], 'x'.join(map(str, row['input_shape'])),
                    row['parameters'], row['elapsed_seconds'],
                    row['holdout_auroc'], row['eval_auroc'], row['eval_auprc'],
                    row['eval_brier'], row['eval_nll'],
                    row['input_reduction_percent'],
                    row['parameter_reduction_percent'], row['eval_ece'],
                )
            logging.info('VIB INPUT ABLATION COMPLETE')

        if args.compute_vib_mixture_ablation:
            mixture_components = int(args.vib_mixture_components)
            mixture_init_scale = float(args.vib_mixture_mean_init_scale)
            if mixture_components < 2:
                raise ValueError("--vib_mixture_components must be at least 2.")
            if mixture_init_scale <= 0.0:
                raise ValueError(
                    "--vib_mixture_mean_init_scale must be positive to break "
                    "component symmetry."
                )
            eval_vib_error_targets = np.asarray([
                int(float(value) < correctness_threshold)
                for value in validation_is_true
            ], dtype=np.int64)
            mixture_variants = {
                'standard': {
                    'prior_type': 'standard_normal',
                    'mixture_readout': False,
                    'description': 'single N(0,I) prior and latent-only classifier',
                },
                f'mixture_prior_k{mixture_components}': {
                    'prior_type': 'gaussian_mixture',
                    'mixture_readout': False,
                    'description': (
                        f'{mixture_components} learned diagonal-Gaussian '
                        'prototypes; latent-only classifier'
                    ),
                },
                f'mixture_readout_k{mixture_components}': {
                    'prior_type': 'gaussian_mixture',
                    'mixture_readout': True,
                    'description': (
                        f'{mixture_components} learned diagonal-Gaussian '
                        'prototypes; classifier consumes [z; r_1...r_K]'
                    ),
                },
            }
            mixture_ablation_metadata = {
                'experiment': 'last_token_mixture_vib_ablation',
                'feature_name': VIB_LAST_TOKEN_FEATURE_NAME,
                'feature_layout': 'vector',
                'mixture_components': mixture_components,
                'hypothesis': (
                    'last-token grounding risk is multi-modal rather than one '
                    'single standard-Gaussian latent direction'
                ),
                'controlled_variables': (
                    'same cached last_token_h, binary targets, train/holdout '
                    'split, seed, encoder, latent dimension, BCE objective, KL '
                    'weight and warmup, optimizer, and early stopping'
                ),
                'variant_difference': (
                    'standard changes to a learnable Gaussian-mixture prior; '
                    'the final variant additionally exposes prototype '
                    'responsibilities to the risk classifier'
                ),
                'prototype_semantics': (
                    'unsupervised latent failure modes; component indices are '
                    'not assigned entity/attribute/relation labels'
                ),
                'correctness_threshold': correctness_threshold,
                'variants': {},
            }
            mixture_rows = []
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            logging.info(
                'VIB MIXTURE ABLATION START | components=%d | variants=%s',
                mixture_components,
                ','.join(mixture_variants),
            )
            for variant_name, variant_spec in mixture_variants.items():
                variant_config = replace(
                    vib_training_config,
                    input_adapter='flatten',
                    prior_type=variant_spec['prior_type'],
                    mixture_components=mixture_components,
                    mixture_readout=variant_spec['mixture_readout'],
                    mixture_mean_init_scale=mixture_init_scale,
                )
                timer_name = f'vib_probe_mixture_ablation/{variant_name}'
                variant_scores, variant_metadata, variant_model = _timed_training(
                    training_times,
                    timer_name,
                    fit_vib_probe_and_score,
                    train_probe_features,
                    vib_error_targets,
                    validation_probe_features,
                    config=variant_config,
                    feature_name=VIB_LAST_TOKEN_FEATURE_NAME,
                    feature_layout='vector',
                )
                _attach_training_time(
                    variant_metadata, training_times, timer_name
                )
                eval_metrics = _vib_probability_metrics(
                    eval_vib_error_targets, variant_scores
                )
                diagnostics = variant_metadata.get('mixture_diagnostics')
                if diagnostics is not None:
                    eval_responsibilities = np.asarray(
                        diagnostics['eval_prototype_responsibilities'],
                        dtype=np.float64,
                    )
                    eval_assignments = np.argmax(eval_responsibilities, axis=1)
                    diagnostics['eval']['hard_component_error_rate'] = [
                        float(np.mean(
                            eval_vib_error_targets[
                                eval_assignments == component_index
                            ]
                        ))
                        if np.any(eval_assignments == component_index)
                        else float('nan')
                        for component_index in range(mixture_components)
                    ]
                    diagnostics['eval_label_usage'] = (
                        'component error rates are post-hoc reporting only'
                    )
                metric_name = f'vib_probe__mixture_ablation__{variant_name}'
                checkpoint_file = f'vib_probe_mixture_{variant_name}.pt'
                variant_metadata.update({
                    'ablation_variant': variant_name,
                    'variant_description': variant_spec['description'],
                    'correctness_threshold': correctness_threshold,
                    'checkpoint_file': checkpoint_file,
                    'eval_reporting_metrics': eval_metrics,
                    'eval_label_usage': (
                        'reporting only; eval labels are not used for fitting, '
                        'early stopping, checkpoint selection, or prototype '
                        'formation'
                    ),
                })
                checkpoint_path = save_vib_probe_checkpoint(
                    os.path.join(checkpoint_dir, checkpoint_file),
                    variant_model,
                    variant_metadata,
                )
                checkpoint_size_bytes = int(os.path.getsize(checkpoint_path))
                variant_metadata['checkpoint_size_bytes'] = checkpoint_size_bytes
                variant_metadata['checkpoint_size_mib'] = float(
                    checkpoint_size_bytes / (1024.0 ** 2)
                )
                result_dict['uncertainty_measures'][metric_name] = variant_scores
                mixture_ablation_metadata['variants'][variant_name] = (
                    variant_metadata
                )
                mixture_rows.append({
                    'variant': variant_name,
                    'prior_type': variant_metadata['prior_type'],
                    'mixture_readout': variant_metadata['mixture_readout'],
                    'parameters': variant_metadata['parameter_count'],
                    'checkpoint_mib': variant_metadata['checkpoint_size_mib'],
                    'elapsed_seconds': variant_metadata['training_time_seconds'],
                    'best_epoch': variant_metadata['best_epoch'],
                    'holdout_auroc': variant_metadata['holdout_auroc'],
                    'eval_auroc': eval_metrics['auroc'],
                    'eval_auprc': eval_metrics['auprc'],
                    'eval_brier': eval_metrics['brier'],
                    'eval_nll': eval_metrics['nll'],
                    'eval_ece': eval_metrics['ece_10_bin'],
                    'effective_components': (
                        diagnostics['eval']['effective_component_count']
                        if diagnostics is not None else float('nan')
                    ),
                    'active_components': (
                        diagnostics['eval']['hard_active_component_count']
                        if diagnostics is not None else 0
                    ),
                    'mean_max_responsibility': (
                        diagnostics['eval']['mean_max_responsibility']
                        if diagnostics is not None else float('nan')
                    ),
                })
                if diagnostics is not None:
                    logging.info(
                        'VIB MIXTURE COMPONENTS | variant=%s | prior_weights=%s '
                        '| eval_usage=%s | eval_hard_counts=%s '
                        '| eval_error_rates=%s',
                        variant_name,
                        np.round(diagnostics['prior_weights'], 4).tolist(),
                        np.round(
                            diagnostics['eval']['component_usage'], 4
                        ).tolist(),
                        diagnostics['eval']['hard_assignment_counts'],
                        np.round(
                            diagnostics['eval']['hard_component_error_rate'], 4
                        ).tolist(),
                    )
                if not local_mode:
                    wandb.save(checkpoint_path)
                del variant_model
                gc.collect()
                torch.cuda.empty_cache()

            mixture_ablation_metadata['comparison'] = mixture_rows
            result_dict['vib_probe_mixture_ablation_metadata'] = (
                mixture_ablation_metadata
            )
            logging.info(
                '%-24s %11s %8s %9s %9s %9s %9s %9s %9s %7s',
                'VIB mixture variant', 'parameters', 'time', 'hold_AUC',
                'eval_AUC', 'AUPRC', 'Brier', 'NLL', 'ECE', 'eff_K',
            )
            for row in mixture_rows:
                logging.info(
                    '%-24s %11d %7.2fs %9.4f %9.4f %9.4f %9.4f %9.4f '
                    '%9.4f %7.3f | hard_active_K=%d | mean_max_r=%.4f',
                    row['variant'], row['parameters'], row['elapsed_seconds'],
                    row['holdout_auroc'], row['eval_auroc'], row['eval_auprc'],
                    row['eval_brier'], row['eval_nll'], row['eval_ece'],
                    row['effective_components'], row['active_components'],
                    row['mean_max_responsibility'],
                )
            logging.info('VIB MIXTURE ABLATION COMPLETE')

        if args.compute_lrp_probe:
            lrp_training_config = LRPTrainingConfig(
                variants=tuple(args.lrp_variants),
                epochs=int(args.lrp_epochs),
                batch_size=int(args.lrp_batch_size),
                learning_rate=float(args.lrp_learning_rate),
                weight_decay=float(args.lrp_weight_decay),
                validation_fraction=float(args.lrp_validation_fraction),
                patience=int(args.lrp_patience),
                min_delta=float(args.lrp_min_delta),
                mlp_hidden_dims=tuple(int(value) for value in args.lrp_mlp_hidden_dims),
                transformer_model_dim=int(args.lrp_transformer_model_dim),
                transformer_layers=int(args.lrp_transformer_layers),
                transformer_heads=int(args.lrp_transformer_heads),
                dropout=float(args.lrp_dropout),
                top_k_layers=int(args.lrp_top_k_layers),
                random_seed=int(args.random_seed),
                device=str(args.lrp_device),
            )
            lrp_scores, lrp_metadata, lrp_checkpoint = _timed_training(
                training_times,
                'lrp_probe',
                fit_lrp_probes_and_score,
                train_probe_features,
                train_is_true,
                validation_probe_features,
                eval_correctness=validation_is_true,
                config=lrp_training_config,
            )
            _attach_training_time(lrp_metadata, training_times, 'lrp_probe')
            result_dict['uncertainty_measures'].update(lrp_scores)
            lrp_metadata["checkpoint_file"] = "lrp_probe.pt"
            lrp_checkpoint["metadata"] = lrp_metadata
            result_dict['lrp_probe_metadata'] = lrp_metadata
            checkpoint_dir = local_output_dir if local_mode else wandb.run.dir
            checkpoint_path = save_lrp_probe_checkpoint(
                os.path.join(checkpoint_dir, 'lrp_probe.pt'), lrp_checkpoint
            )
            if not local_mode:
                wandb.save(checkpoint_path)

        if args.compute_lightweight_probe:
            sep_scores, sep_model = _timed_training(
                training_times,
                'probe_sep',
                fit_sep_and_score,
                train_probe_features,
                train_is_false_probe,
                validation_probe_features,
                feature_name=args.probe_feature_name,
            )
            result_dict['uncertainty_measures']['probe_sep'] = sep_scores
            sep_coef = sep_model.named_steps['logisticregression'].coef_[0]
            top_coef_idx = np.argsort(np.abs(sep_coef))[::-1][:20]
            result_dict['probe_metadata'] = {
                'feature_name': args.probe_feature_name,
                'sep_top_abs_coef_indices': top_coef_idx.tolist(),
                'sep_top_abs_coef_values': sep_coef[top_coef_idx].tolist(),
                'train_probe_samples': len(train_probe_features),
                'validation_probe_samples': len(validation_probe_features),
            }
            _attach_training_time(
                result_dict['probe_metadata'], training_times, 'probe_sep'
            )

        if args.compute_probe_sep_cv:
            sep_cv_scores, sep_cv_metadata = _timed_training(
                training_times,
                'probe_sep_cv',
                fit_sep_cv_and_score,
                train_probe_features,
                train_is_false_probe,
                validation_probe_features,
                feature_name=args.probe_feature_name,
                c_values=args.probe_v2_c_values,
                cv_folds=args.probe_v2_cv_folds,
                random_seed=args.random_seed,
            )
            _attach_training_time(
                sep_cv_metadata, training_times, 'probe_sep_cv'
            )
            result_dict['uncertainty_measures']['probe_sep_cv'] = sep_cv_scores
            result_dict['probe_sep_cv_metadata'] = sep_cv_metadata

        if args.compute_probe_sep_pca_cv:
            sep_pca_cv_scores, sep_pca_cv_metadata = _timed_training(
                training_times,
                'probe_sep_pca_cv',
                fit_sep_pca_cv_and_score,
                train_probe_features,
                train_is_false_probe,
                validation_probe_features,
                feature_name=args.probe_feature_name,
                pca_dims=args.probe_v2_pca_dims,
                c_values=args.probe_v2_c_values,
                cv_folds=args.probe_v2_cv_folds,
                random_seed=args.random_seed,
            )
            _attach_training_time(
                sep_pca_cv_metadata, training_times, 'probe_sep_pca_cv'
            )
            result_dict['uncertainty_measures'][
                'probe_sep_pca_cv'
            ] = sep_pca_cv_scores
            result_dict['probe_sep_pca_cv_metadata'] = sep_pca_cv_metadata

        if args.compute_probe_sep_v2 :
            sep_v2_scores, sep_v2_metadata = _timed_training(
                training_times,
                'probe_sep_v2',
                fit_sep_v2_and_score,
                train_probe_features,
                train_is_false_probe,
                train_generation_records,
                validation_probe_features,
                feature_names=args.probe_v2_feature_names,
                pca_dims=args.probe_v2_pca_dims,
                c_values=args.probe_v2_c_values,
                ridge_alphas=args.probe_v2_ridge_alphas,
                cv_folds=args.probe_v2_cv_folds,
                random_seed=args.random_seed,
            )
            _attach_training_time(
                sep_v2_metadata, training_times, 'probe_sep_v2'
            )
            result_dict['uncertainty_measures']['probe_sep_v2'] = sep_v2_scores
            result_dict['probe_sep_v2_metadata'] = sep_v2_metadata

        # if args.compute_probe_sep_v2_4view:
        #     sep_v2_4view_scores, sep_v2_4view_metadata = fit_sep_v2_and_score(
        #         train_probe_features,
        #         train_is_false_probe,
        #         train_generation_records,
        #         validation_probe_features,
        #         feature_names=V2_FOUR_VIEW_FEATURE_NAMES,
        #         pca_dims=args.probe_v2_pca_dims,
        #         c_values=args.probe_v2_c_values,
        #         ridge_alphas=args.probe_v2_ridge_alphas,
        #         cv_folds=args.probe_v2_cv_folds,
        #         random_seed=args.random_seed,
        #     )
        #     result_dict['uncertainty_measures'][
        #         'probe_sep_v2_4view'
        #     ] = sep_v2_4view_scores
        #     result_dict['probe_sep_v2_4view_metadata'] = sep_v2_4view_metadata

        # if args.compute_probe_sep_v2_2view:
        #     sep_v2_2view_scores, sep_v2_2view_metadata = fit_sep_v2_and_score(
        #         train_probe_features,
        #         train_is_false_probe,
        #         train_generation_records,
        #         validation_probe_features,
        #         feature_names=V2_TWO_VIEW_FEATURE_NAMES,
        #         pca_dims=args.probe_v2_pca_dims,
        #         c_values=args.probe_v2_c_values,
        #         ridge_alphas=args.probe_v2_ridge_alphas,
        #         cv_folds=args.probe_v2_cv_folds,
        #         random_seed=args.random_seed,
        #     )
        #     result_dict['uncertainty_measures'][
        #         'probe_sep_v2_2view'
        #     ] = sep_v2_2view_scores
        #     result_dict['probe_sep_v2_2view_metadata'] = sep_v2_2view_metadata

    if args.compute_p_ik:
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = _timed_training(
            training_times,
            'p_ik_error',
            get_p_ik,
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = _timed_training(
            training_times,
            'p_ik_answerable',
            get_p_ik,
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    _log_training_time_summary(training_times)
    completed_training_seconds = sum(
        row['seconds'] for row in training_times.values()
        if row['status'] == 'completed'
    )
    result_dict['training_time_summary'] = {
        'clock': 'wall_clock_perf_counter',
        'scope': 'fit_and_eval_score_excludes_checkpoint_serialization',
        'completed_total_seconds': float(completed_training_seconds),
        'completed_total_formatted': _format_elapsed_time(
            completed_training_seconds
        ),
        'methods': training_times,
    }

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    before_filter = set(result_dict.get('uncertainty_measures', {}))
    result_dict['uncertainty_measures'] = filter_reportable_uncertainty_measures(
        result_dict.get('uncertainty_measures', {})
    )
    skipped = sorted(before_filter - set(result_dict['uncertainty_measures']))
    if skipped:
        logging.info(
            'Removed %d intermediate uncertainty measures before saving: %s',
            len(skipped),
            skipped,
        )

    if local_mode:
        save_local(result_dict, 'uncertainty_measures.pkl')
    else:
        utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        if local_mode:
            analyze_run(
                args.eval_wandb_runid,
                local_run_dir=local_output_dir,
            )
        else:
            analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    args.assign_new_wandb_id = True
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(50 * '#X')
    logging.info('STARTING `compute_uncertainty_measures`!')
    logging.info("Args: %s", args)

    main(args)
