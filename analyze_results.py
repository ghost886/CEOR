"""Compute overall performance metrics from predicted uncertainties."""
import argparse
import csv
import functools
import logging
import os
import pickle

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
try:
    import wandb
except ImportError:  # pragma: no cover - local-only fallback
    from uncertainty.utils import wandb_stub as wandb

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    area_under_thresholded_accuracy,
    accuracy_at_quantile,
    binary_nll,
    bootstrap,
    brier_score,
    compatible_bootstrap,
    expected_calibration_error,
    auprc,
    auroc,
)


utils.setup_logger()

result_dict = {}

UNC_MEAS = 'uncertainty_measures.pkl'


# These scores were empirically indistinguishable from chance (or degenerate)
# in the 4k-sample grounding evaluation.  Keep their underlying structured
# diagnostics available to structured scene-graph methods, but do not publish them as standalone
# uncertainty measures.
DEPRECATED_UNCERTAINTY_MEASURES = frozenset({
    'lang_align_u_align',
    'lang_align_u_attn',
})

# Retired language/vision alignment and attention uncertainty scores.  Keep
# grounding evaluation fields (IoU, parse success, etc.) and unrelated scores.
RETIRED_UNCERTAINTY_PREFIXES = ('lang_align_u_', 'grounding_u_')


def is_retired_uncertainty_measure(measure_name: str) -> bool:
    return measure_name.startswith(RETIRED_UNCERTAINTY_PREFIXES)


def is_error_probability_measure(measure_name: str, values) -> bool:
    """Return whether a score is an error probability suitable for calibration."""
    if not (
        measure_name.startswith('probe_sep')
        or measure_name.startswith('probe_v3_')
        or measure_name == 'icr_probe'
        or measure_name.startswith('vib_probe')
        or measure_name == 'raw_vib_linear_probe'
        or measure_name == 'raw_vib_shallow_probe'
        or measure_name == 'sivr_sequence_error_probability'
        or measure_name.startswith('lrp_')
    ):
        return False
    array = np.asarray(values, dtype=float)
    return bool(
        array.size
        and np.isfinite(array).all()
        and float(array.min()) >= 0.0
        and float(array.max()) <= 1.0
    )


def is_reportable_uncertainty_measure(measure_name: str) -> bool:
    """Return True for headline uncertainty scores, not grounding diagnostics."""
    if is_retired_uncertainty_measure(measure_name):
        return False
    if measure_name in DEPRECATED_UNCERTAINTY_MEASURES:
        return False
    return not measure_name.startswith('grounding_')


def filter_reportable_uncertainty_measures(measures: dict) -> dict:
    """Drop intermediate and deprecated scores from uncertainty evaluation."""
    return {
        name: values
        for name, values in measures.items()
        if is_reportable_uncertainty_measure(name)
    }


def compute_uncertainty_iou_correlations(measures: dict, iou_values) -> dict:
    """Correlate every aligned scalar uncertainty with IoU and grounding error."""
    ious = np.asarray(iou_values, dtype=np.float64).reshape(-1)
    output = {
        "num_iou_samples": int(ious.size),
        "target_iou": "ground-truth box IoU; larger is better",
        "target_one_minus_iou": "1 - IoU; larger is worse",
        "expected_direction": (
            "A useful uncertainty normally has negative correlation with IoU "
            "and positive correlation with 1-IoU."
        ),
        "measures": {},
    }

    def statistic_and_pvalue(result):
        if hasattr(result, "statistic") and hasattr(result, "pvalue"):
            statistic, pvalue = result.statistic, result.pvalue
        else:
            statistic, pvalue = result
        return float(statistic), float(pvalue)

    for measure_name, raw_values in measures.items():
        row = {"num_measure_values": None, "status": "computed"}
        try:
            values = np.asarray(raw_values, dtype=np.float64)
        except (TypeError, ValueError):
            row["status"] = "skipped_non_numeric"
            output["measures"][measure_name] = row
            continue
        row["num_measure_values"] = int(values.size)
        if values.ndim != 1:
            row["status"] = "skipped_non_scalar_per_sample"
            output["measures"][measure_name] = row
            continue
        if values.size != ious.size:
            row["status"] = "skipped_length_mismatch"
            output["measures"][measure_name] = row
            continue

        finite = np.isfinite(values) & np.isfinite(ious)
        values_current = values[finite]
        ious_current = ious[finite]
        row["num_finite_pairs"] = int(finite.sum())
        if values_current.size < 3:
            row["status"] = "skipped_fewer_than_3_finite_pairs"
            output["measures"][measure_name] = row
            continue
        if np.unique(values_current).size < 2:
            row["status"] = "skipped_constant_uncertainty"
            output["measures"][measure_name] = row
            continue
        if np.unique(ious_current).size < 2:
            row["status"] = "skipped_constant_iou"
            output["measures"][measure_name] = row
            continue

        grounding_error = 1.0 - ious_current
        for target_name, target in (
            ("iou", ious_current),
            ("one_minus_iou", grounding_error),
        ):
            pearson_r, pearson_p = statistic_and_pvalue(
                pearsonr(values_current, target)
            )
            spearman_rho, spearman_p = statistic_and_pvalue(
                spearmanr(values_current, target)
            )
            kendall_tau, kendall_p = statistic_and_pvalue(
                kendalltau(values_current, target)
            )
            row.update({
                f"pearson_r_with_{target_name}": pearson_r,
                f"pearson_pvalue_with_{target_name}": pearson_p,
                f"spearman_rho_with_{target_name}": spearman_rho,
                f"spearman_pvalue_with_{target_name}": spearman_p,
                f"kendall_tau_with_{target_name}": kendall_tau,
                f"kendall_pvalue_with_{target_name}": kendall_p,
            })
        output["measures"][measure_name] = row
    return output


def _write_uncertainty_iou_correlations_csv(payload: dict, path: str) -> None:
    """Write a flat, spreadsheet-friendly copy of correlation results."""
    rows = [
        {"uncertainty_measure": name, **values}
        for name, values in payload.get("measures", {}).items()
    ]
    fieldnames = ["uncertainty_measure"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(wandb_runid, assign_new_wandb_id, experiment_lot, entity):
    """Initialize wandb session."""
    user = os.getenv('USER', 'ceor')
    slurm_jobid = os.getenv('SLURM_JOB_ID')
    scratch_dir = os.getenv('SCRATCH_DIR', 'experiments')
    kwargs = dict(
        entity=entity,
        project='semantic_uncertainty',
        dir=f'{scratch_dir}/{user}/uncertainty',
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {experiment_lot}',
    )
    if not assign_new_wandb_id:
        # Restore wandb session.
        wandb.init(
            id=wandb_runid,
            resume=True,
            **kwargs)
        wandb.restore(UNC_MEAS)
    else:
        api = wandb.Api()
        wandb.init(**kwargs)

        old_run = api.run(f'{entity}/semantic_uncertainty/{wandb_runid}')
        old_run.file(UNC_MEAS).download(
            replace=True, exist_ok=False, root=wandb.run.dir)


def analyze_run(
        wandb_runid, assign_new_wandb_id=False, answer_fractions_mode='default',
        experiment_lot=None, entity=None, local_run_dir=None):
    """Analyze the uncertainty measures for a given wandb run id."""
    logging.info('Analyzing wandb_runid `%s`.', wandb_runid)

    # Set up evaluation metrics.
    if answer_fractions_mode == 'default':
        answer_fractions = [0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
    elif answer_fractions_mode == 'finegrained':
        answer_fractions = [round(i, 3) for i in np.linspace(0, 1, 20+1)]
    else:
        raise ValueError

    rng = np.random.default_rng(41)
    eval_metrics = dict(zip(
        ['AUROC', 'AUPRC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, auprc, area_under_thresholded_accuracy, np.mean],
            [
                compatible_bootstrap,
                compatible_bootstrap,
                compatible_bootstrap,
                bootstrap,
            ]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    if local_run_dir is not None:
        run_files_dir = local_run_dir
        if (
            os.path.basename(run_files_dir) != 'files'
            and not os.path.exists(os.path.join(run_files_dir, UNC_MEAS))
        ):
            run_files_dir = os.path.join(run_files_dir, 'files')
        logging.info('Reading local analysis inputs from `%s`.', run_files_dir)
    elif wandb.run is None:
        init_wandb(
            wandb_runid, assign_new_wandb_id=assign_new_wandb_id,
            experiment_lot=experiment_lot, entity=entity)

    elif wandb.run.id != wandb_runid:
        raise ValueError

    # Load the results dictionary from a pickle file.
    result_path = (
        os.path.join(run_files_dir, UNC_MEAS)
        if local_run_dir is not None
        else f'{wandb.run.dir}/{UNC_MEAS}'
    )
    with open(result_path, 'rb') as file:
        results_old = pickle.load(file)

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute simple accuracy metrics for model predictions.
    all_accuracies = dict()
    all_accuracies['accuracy'] = 1 - np.array(results_old['validation_is_false'])

    for name, target in all_accuracies.items():
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = np.mean(target)
        result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)

    grounding_ious = None
    if 'grounding_eval' in results_old and results_old['grounding_eval']:
        grounding_records = results_old['grounding_eval']

        def grounding_array(key, default=0.0):
            return np.array([float(record.get(key, default)) for record in grounding_records])

        grounding_metrics = {
            'grounding_parse_rate': grounding_array('parse_success'),
            'grounding_mean_iou': grounding_array('iou'),
            # Qwen-VL evaluate_grounding.py reports this as Precision @ 1:
            # one prediction per sample is correct when IoU >= 0.5.
            'grounding_precision_at_1': grounding_array('precision_at_1'),
            'grounding_precision_at_iou_0.25': grounding_array('precision_at_iou_0.25'),
            'grounding_precision_at_iou_0.5': grounding_array('precision_at_iou_0.5'),
            'grounding_precision_at_iou_0.75': grounding_array('precision_at_iou_0.75'),
            'grounding_mean_center_distance': grounding_array('center_distance'),
        }

        for name, target in grounding_metrics.items():
            result_dict['performance'][name] = {}
            result_dict['performance'][name]['mean'] = np.mean(target)
            result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)
        grounding_ious = grounding_metrics['grounding_mean_iou']

    # compute_uncertainty_measures stores this aligned vector (including NaN for
    # a missing parse). Prefer it over the compact grounding_eval list so every
    # uncertainty remains paired with the correct validation sample.
    aligned_grounding_ious = results_old.get('validation_grounding_ious')
    if aligned_grounding_ious is not None:
        grounding_ious = np.asarray(aligned_grounding_ious, dtype=np.float64)

    rum = dict(results_old.get('uncertainty_measures', {}))
    excluded_measures = {
        "grounding_u_total",
        "grounding_u_ground",
        "grounding_u_coord",
        "grounding_u_cons",
        "grounding_u_box_attn",
        "grounding_u_distractor",
        "grounding_u_coord_nll",
        "grounding_u_coord_entropy",
        "grounding_u_coord_margin",
        "grounding_u_step_temporal",
        "grounding_u_head_js",
        # 继续添加不想分析的指标
    }

    if 'p_false' in rum and 'p_false_fixed' not in rum:
        # Restore log probs true: y = 1 - x --> x = 1 - y.
        # Convert to probs --> np.exp(1 - y).
        # Convert to p_false --> 1 - np.exp(1 - y).
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    correlation_payload = None
    if grounding_ious is not None:
        correlation_measures = filter_reportable_uncertainty_measures(rum)
        correlation_payload = compute_uncertainty_iou_correlations(
            correlation_measures,
            grounding_ious,
        )
        result_dict['grounding_uncertainty_iou_correlation'] = correlation_payload
        computed_count = sum(
            row.get('status') == 'computed'
            for row in correlation_payload['measures'].values()
        )
        logging.info(
            'Computed IoU correlations for %d/%d reportable uncertainty measures.',
            computed_count,
            len(correlation_payload['measures']),
        )

    # Retain the historical headline-metric exclusions, but only after the
    # comprehensive IoU-correlation table above has seen every reportable score.
    rum = {
        name: values for name, values in rum.items() if name not in excluded_measures
    }
    skipped_measures = sorted(set(rum) - set(filter_reportable_uncertainty_measures(rum)))
    rum = filter_reportable_uncertainty_measures(rum)
    if skipped_measures:
        logging.info(
            'Skipping %d intermediate uncertainty measures: %s',
            len(skipped_measures),
            skipped_measures,
        )

    # Next: Uncertainty Measures.
    # Iterate through the dictionary and compute additional metrics for each measure.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)

        # Validation accuracy.
        task_error_target = results_old['validation_is_false']
        validation_is_falses = [
            task_error_target,
            results_old['validation_unanswerable']
        ]

        logging_names = ['', '_UNANSWERABLE']

        # Iterate over predictions of 'falseness' or 'answerability'.
        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):
            validation_is_false = np.array(validation_is_false)
            if logging_name == '_UNANSWERABLE' and np.unique(validation_is_false).size < 2:
                logging.info(
                    'Skipping unanswerable analysis for `%s`: target has one class.',
                    measure_name,
                )
                continue
            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_accuracy = 1 - validation_is_false
            # LRP follows the paper's soft correctness supervision.  Ranking
            # metrics still require a binary event, while selective accuracy
            # and proper probability scores can retain the soft target.
            validation_error_event = (
                validation_is_false >= 0.5
            ).astype(float)
            measure_values_current = np.asarray(measure_values, dtype=float)
            if len(measure_values) > len(validation_is_false):
                # This can happen, but only for p_false.
                if 'p_false' not in measure_name:
                    raise ValueError
                logging.warning(
                    'More measure values for %s than in validation_is_false. Len(measure values): %d, Len(validation_is_false): %d',
                    measure_name, len(measure_values), len(validation_is_false))
                measure_values_current = measure_values_current[:len(validation_is_false)]

            fargs = {
                'AUROC': [validation_error_event, measure_values_current],
                'AUPRC': [validation_error_event, measure_values_current],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values_current],
                'mean_uncertainty': [measure_values_current]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [
                    validation_accuracy,
                    measure_values_current,
                ]

            measure_eval_metrics = dict(eval_metrics)
            # These scores model P(error), not P(unanswerable). Calibration against
            # the unanswerable target would therefore have the wrong semantics.
            if (
                not logging_name
                and is_error_probability_measure(
                    measure_name,
                    measure_values_current,
                )
            ):
                measure_eval_metrics.update({
                    'Brier_score': [brier_score, compatible_bootstrap],
                    'NLL': [binary_nll, compatible_bootstrap],
                    'ECE': [expected_calibration_error, compatible_bootstrap],
                })
                fargs.update({
                    'Brier_score': [validation_is_false, measure_values_current],
                    'NLL': [validation_is_false, measure_values_current],
                    'ECE': [validation_is_false, measure_values_current],
                })

            for fname, (function, bs_function) in measure_eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = metric_i
                logging.info("%s for measure name `%s`: %f", fname, name, metric_i)
                result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                    function, rng)(*fargs[fname])

    if correlation_payload is not None:
        correlation_csv_path = os.path.join(
            os.path.dirname(result_path),
            'uncertainty_iou_correlations.csv',
        )
        _write_uncertainty_iou_correlations_csv(
            correlation_payload, correlation_csv_path
        )
        logging.info(
            'Saved uncertainty/IoU correlations to `%s`.',
            correlation_csv_path,
        )
        if local_run_dir is None:
            wandb.save(correlation_csv_path)

    if local_run_dir is not None:
        out_path = os.path.join(run_files_dir, 'analysis_results.pkl')
        with open(out_path, 'wb') as file:
            pickle.dump(result_dict, file)
        logging.info('Saved local analysis results to `%s`.', out_path)
    else:
        wandb.log(result_dict)
    logging.info(
        'Analysis for wandb_runid `%s` finished: %d performance targets and '
        '%d uncertainty measures evaluated.',
        wandb_runid,
        len(result_dict['performance']),
        len(result_dict['uncertainty']),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_runids', nargs='+', type=str,
                        help='Wandb run ids of the datasets to evaluate on.')
    parser.add_argument('--assign_new_wandb_id', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--answer_fractions_mode', type=str, default='default')
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    parser.add_argument(
        "--entity", type=str, help="Wandb entity.")

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    wandb_runids = args.wandb_runids
    for wid in wandb_runids:
        logging.info('Evaluating wandb_runid `%s`.', wid)
        analyze_run(
            wid, args.assign_new_wandb_id, args.answer_fractions_mode,
            experiment_lot=args.experiment_lot, entity=args.entity)
