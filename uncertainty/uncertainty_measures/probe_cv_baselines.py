"""Cross-validated single-view baselines for fair SEP comparisons."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from uncertainty.uncertainty_measures.lightweight_probe import feature_matrix


FoldCache = List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def _cv_splits(
    target: np.ndarray,
    requested_folds: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(target.astype(np.int64), minlength=2)
    if np.count_nonzero(counts) < 2:
        raise ValueError("SEP CV baselines need both correct and incorrect samples.")
    n_splits = min(int(requested_folds), int(counts[counts > 0].min()))
    if n_splits < 2:
        raise ValueError("SEP CV baselines need at least two samples in each class.")
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_seed),
    )
    return list(splitter.split(np.zeros(len(target)), target))


def _classifier(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )


def _select_c(
    fold_cache: FoldCache,
    target: np.ndarray,
    c_values: Sequence[float],
) -> Tuple[float, float, List[Dict[str, float]]]:
    results = []
    best = None
    for c_value in c_values:
        oof = np.zeros(len(target), dtype=np.float32)
        for fit_idx, holdout_idx, z_fit, z_holdout in fold_cache:
            model = _classifier(c_value)
            model.fit(z_fit, target[fit_idx])
            oof[holdout_idx] = model.predict_proba(z_holdout)[:, 1]
        auc = float(roc_auc_score(target, oof))
        results.append({"c": float(c_value), "oof_auroc": auc})
        candidate = (auc, -float(c_value), float(c_value))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[0], results


def _raw_fold_cache(
    values: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> FoldCache:
    cache = []
    for fit_idx, holdout_idx in splits:
        scaler = StandardScaler()
        z_fit = scaler.fit_transform(values[fit_idx]).astype(np.float32)
        z_holdout = scaler.transform(values[holdout_idx]).astype(np.float32)
        cache.append((fit_idx, holdout_idx, z_fit, z_holdout))
    return cache


def _pca_fold_caches(
    values: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    pca_dims: Sequence[int],
    random_seed: int,
) -> Dict[int, FoldCache]:
    max_dim = max(pca_dims)
    caches: Dict[int, FoldCache] = {int(dim): [] for dim in pca_dims}
    for fold_idx, (fit_idx, holdout_idx) in enumerate(splits):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(values[fit_idx])
        x_holdout = scaler.transform(values[holdout_idx])
        pca = PCA(
            n_components=max_dim,
            svd_solver="randomized",
            iterated_power=2,
            random_state=int(random_seed) + fold_idx,
        )
        z_fit_max = pca.fit_transform(x_fit).astype(np.float32)
        z_holdout_max = pca.transform(x_holdout).astype(np.float32)
        for dim in caches:
            caches[dim].append((
                fit_idx,
                holdout_idx,
                z_fit_max[:, :dim],
                z_holdout_max[:, :dim],
            ))
    return caches


def fit_sep_cv_and_score(
    train_features: List[Dict],
    train_is_false: List[float],
    eval_features: List[Dict],
    *,
    feature_name: str = "argus_h_mean",
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 5,
    random_seed: int = 10,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Tune only logistic C by fold-local scaling, then refit on full train."""
    x_train = feature_matrix(train_features, feature_name)
    x_eval = feature_matrix(eval_features, feature_name)
    target = np.asarray(train_is_false, dtype=np.int64)
    if len(x_train) != len(target):
        raise ValueError("Training features and labels have inconsistent lengths.")
    c_values = sorted({float(value) for value in c_values if float(value) > 0.0})
    if not c_values:
        raise ValueError("SEP CV needs at least one positive C value.")

    splits = _cv_splits(target, cv_folds, random_seed)
    selected_c, oof_auc, cv_results = _select_c(
        _raw_fold_cache(x_train, splits), target, c_values)

    scaler = StandardScaler()
    z_train = scaler.fit_transform(x_train)
    z_eval = scaler.transform(x_eval)
    model = _classifier(selected_c)
    model.fit(z_train, target)
    scores = model.predict_proba(z_eval)[:, 1]
    metadata = {
        "version": "probe_sep_cv_v1",
        "feature_name": feature_name,
        "train_samples": int(len(x_train)),
        "eval_samples": int(len(x_eval)),
        "input_dim": int(x_train.shape[1]),
        "cv_folds": int(len(splits)),
        "c_values": c_values,
        "selected_c": float(selected_c),
        "oof_auroc": float(oof_auc),
        "cv_results": cv_results,
        "preprocessing_scope": "fit_within_each_cv_fold",
    }
    logging.info(
        "probe_sep_cv selected C=%g with OOF AUROC=%.6f.",
        selected_c,
        oof_auc,
    )
    return np.asarray(scores, dtype=np.float32), metadata


def fit_sep_pca_cv_and_score(
    train_features: List[Dict],
    train_is_false: List[float],
    eval_features: List[Dict],
    *,
    feature_name: str = "argus_h_mean",
    pca_dims: Sequence[int] = (32, 64, 128),
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    cv_folds: int = 5,
    random_seed: int = 10,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Tune PCA dimension and logistic C with fold-local preprocessing."""
    x_train = feature_matrix(train_features, feature_name)
    x_eval = feature_matrix(eval_features, feature_name)
    target = np.asarray(train_is_false, dtype=np.int64)
    if len(x_train) != len(target):
        raise ValueError("Training features and labels have inconsistent lengths.")
    c_values = sorted({float(value) for value in c_values if float(value) > 0.0})
    requested_dims = sorted({int(dim) for dim in pca_dims if int(dim) > 0})
    if not c_values or not requested_dims:
        raise ValueError("SEP PCA CV needs positive PCA dimensions and C values.")

    splits = _cv_splits(target, cv_folds, random_seed)
    max_components = min(
        x_train.shape[1],
        min(len(fit_idx) for fit_idx, _ in splits),
    )
    valid_dims = [dim for dim in requested_dims if dim <= max_components]
    if not valid_dims:
        valid_dims = [max_components]
    caches = _pca_fold_caches(x_train, splits, valid_dims, random_seed)

    dimension_results = []
    best = None
    for dim in valid_dims:
        selected_c, auc, c_results = _select_c(caches[dim], target, c_values)
        row = {
            "pca_dim": int(dim),
            "selected_c": float(selected_c),
            "oof_auroc": float(auc),
            "c_results": c_results,
        }
        dimension_results.append(row)
        candidate = (auc, -int(dim), -float(selected_c), row)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    selected = best[3]
    selected_dim = int(selected["pca_dim"])
    selected_c = float(selected["selected_c"])
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_eval_s = scaler.transform(x_eval)
    pca = PCA(
        n_components=selected_dim,
        svd_solver="randomized",
        iterated_power=2,
        random_state=int(random_seed),
    )
    z_train = pca.fit_transform(x_train_s)
    z_eval = pca.transform(x_eval_s)
    model = _classifier(selected_c)
    model.fit(z_train, target)
    scores = model.predict_proba(z_eval)[:, 1]
    metadata = {
        "version": "probe_sep_pca_cv_v1",
        "feature_name": feature_name,
        "train_samples": int(len(x_train)),
        "eval_samples": int(len(x_eval)),
        "input_dim": int(x_train.shape[1]),
        "cv_folds": int(len(splits)),
        "pca_dims_requested": requested_dims,
        "pca_dims_evaluated": valid_dims,
        "c_values": c_values,
        "selected_pca_dim": selected_dim,
        "selected_c": selected_c,
        "oof_auroc": float(selected["oof_auroc"]),
        "explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()),
        "dimension_results": dimension_results,
        "preprocessing_scope": "fit_within_each_cv_fold",
    }
    logging.info(
        "probe_sep_pca_cv selected PCA=%d and C=%g with OOF AUROC=%.6f.",
        selected_dim,
        selected_c,
        float(selected["oof_auroc"]),
    )
    return np.asarray(scores, dtype=np.float32), metadata
