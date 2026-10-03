"""Functions for performance evaluation, mainly used in analyze_results.py."""
import numpy as np
import scipy
from sklearn import metrics


# pylint: disable=missing-function-docstring


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def auprc(y_true, y_score):
    """Area under the precision-recall curve (average precision)."""
    return float(metrics.average_precision_score(y_true, y_score))


def brier_score(y_true, y_probability):
    """Mean squared error of binary error probabilities."""
    y_true = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    return float(np.mean((y_probability - y_true) ** 2))


def binary_nll(y_true, y_probability, eps=1e-7):
    """Binary negative log likelihood; lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(y_probability, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(
        y_true * np.log(probability)
        + (1.0 - y_true) * np.log(1.0 - probability)
    ))


def expected_calibration_error(y_true, y_probability, n_bins=10):
    """Equal-width binary ECE over predicted error probabilities."""
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive.")
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(y_probability, dtype=float)
    if y_true.shape != probability.shape:
        raise ValueError("Targets and probabilities must have matching shapes.")
    if y_true.size == 0:
        return float("nan")
    # Probability 1.0 belongs to the final bin.
    bin_indices = np.minimum(
        (np.clip(probability, 0.0, 1.0) * int(n_bins)).astype(int),
        int(n_bins) - 1,
    )
    ece = 0.0
    for bin_idx in range(int(n_bins)):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue
        confidence = float(np.mean(probability[mask]))
        frequency = float(np.mean(y_true[mask]))
        ece += float(mask.mean()) * abs(confidence - frequency)
    return float(ece)


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    return np.mean(accuracies[select])


def area_under_thresholded_accuracy(accuracies, uncertainties):
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array([accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles])
    dx = quantiles[1] - quantiles[0]
    area = (select_accuracies * dx).sum()
    return area


# Need wrappers because scipy expects 1D data.
def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func
