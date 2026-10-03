"""Multi-view, IoU-aware stacked probe for grounding uncertainty."""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


DEFAULT_FEATURE_NAMES = (
    "argus_h_mean",
    "late_h_mean",
    "last_token_h",
    "delta_last_first_h",
    "token_h_std",
    "numeric_token_h_mean",
)

# Stable preset for the lightweight two-view SEP-v2 ablation.
V2_TWO_VIEW_FEATURE_NAMES = (
    "last_token_h",
    "delta_last_first_h",
)


def _as_vector(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _feature_matrices(
    train_features: Sequence[Mapping],
    eval_features: Sequence[Mapping],
    feature_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build same-width matrices, representing missing feature views with NaN."""
    exemplar = None
    for features in train_features:
        value = features.get(feature_name)
        if value is not None:
            exemplar = _as_vector(value)
            break
    if exemplar is None:
        raise ValueError(f"No training sample contains probe feature `{feature_name}`.")

    width = int(exemplar.size)

    def build(rows: Sequence[Mapping]) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.full((len(rows), width), np.nan, dtype=np.float32)
        missing = np.ones(len(rows), dtype=np.float32)
        for row_idx, features in enumerate(rows):
            value = features.get(feature_name)
            if value is None:
                continue
            vector = _as_vector(value)
            if vector.size != width:
                raise ValueError(
                    f"Probe feature `{feature_name}` has inconsistent widths: "
                    f"expected {width}, got {vector.size} at row {row_idx}.")
            matrix[row_idx] = vector
            missing[row_idx] = 0.0
        return matrix, missing

    x_train, train_missing = build(train_features)
    x_eval, eval_missing = build(eval_features)
    return x_train, x_eval, train_missing, eval_missing


def _impute_fit(values: np.ndarray) -> np.ndarray:
    means = np.nanmean(values, axis=0)
    return np.where(np.isfinite(means), means, 0.0).astype(np.float32)


def _impute_transform(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    if not np.isnan(values).any():
        return values
    return np.where(np.isnan(values), means[None, :], values).astype(np.float32)


def _fit_projection(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    n_components: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    means = _impute_fit(x_train)
    train_imputed = _impute_transform(x_train, means)
    eval_imputed = _impute_transform(x_eval, means)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    eval_scaled = scaler.transform(eval_imputed)
    pca = PCA(
        n_components=int(n_components),
        svd_solver="randomized",
        iterated_power=2,
        random_state=int(random_seed),
    )
    z_train = pca.fit_transform(train_scaled).astype(np.float32)
    z_eval = pca.transform(eval_scaled).astype(np.float32)
    metadata = {
        "explained_variance_ratios": pca.explained_variance_ratio_.tolist(),
        "input_dim": int(x_train.shape[1]),
        "fitted_pca_dim": int(n_components),
    }
    return z_train, z_eval, metadata


def _projection_caches(
    projected: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    component_values: Sequence[int],
) -> Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Reuse one train-fitted unsupervised PCA for supervised CV and OOF heads."""
    caches = {int(value): [] for value in component_values}
    for fit_idx, holdout_idx in splits:
        for n_components in caches:
            caches[n_components].append((
                fit_idx,
                holdout_idx,
                projected[fit_idx, :n_components],
                projected[holdout_idx, :n_components],
            ))
    return caches


def _binary_head_cv(
    fold_cache: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    target: np.ndarray,
    c_values: Sequence[float],
) -> Tuple[float, np.ndarray, float]:
    """Select C and return its out-of-fold probabilities and AUROC."""
    best = None
    for c_value in c_values:
        oof = np.zeros(len(target), dtype=np.float32)
        valid = np.zeros(len(target), dtype=bool)
        for fit_idx, holdout_idx, z_fit, z_holdout in fold_cache:
            y_fit = target[fit_idx]
            if np.unique(y_fit).size < 2:
                continue
            classifier = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
            )
            classifier.fit(z_fit, y_fit)
            oof[holdout_idx] = classifier.predict_proba(z_holdout)[:, 1]
            valid[holdout_idx] = True
        if not valid.all() or np.unique(target[valid]).size < 2:
            continue
        auc = float(roc_auc_score(target[valid], oof[valid]))
        candidate = (auc, -float(c_value), float(c_value), oof)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Could not fit a binary auxiliary head with two classes.")
    return best[2], best[3], best[0]


def _fit_binary_full(
    z_train: np.ndarray,
    target: np.ndarray,
    z_eval: np.ndarray,
    c_value: float,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
    )
    classifier.fit(z_train, target)
    return classifier.predict_proba(z_eval)[:, 1], classifier.coef_[0]


def _regression_head_cv(
    fold_cache: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    target: np.ndarray,
    alpha_values: Sequence[float],
) -> Tuple[float, np.ndarray, float]:
    best = None
    for alpha in alpha_values:
        oof = np.zeros(len(target), dtype=np.float32)
        valid = np.zeros(len(target), dtype=bool)
        for fit_idx, holdout_idx, z_fit, z_holdout in fold_cache:
            fit_valid = np.isfinite(target[fit_idx])
            if fit_valid.sum() < 2:
                continue
            regressor = Ridge(alpha=float(alpha))
            regressor.fit(z_fit[fit_valid], target[fit_idx][fit_valid])
            oof[holdout_idx] = regressor.predict(z_holdout)
            valid[holdout_idx] = True
        valid &= np.isfinite(target)
        if valid.sum() < 2:
            continue
        mse = float(mean_squared_error(target[valid], oof[valid]))
        candidate = (-mse, -float(alpha), float(alpha), oof)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError("Could not fit the continuous IoU-risk auxiliary head.")
    return best[2], np.clip(best[3], 0.0, 1.0), -best[0]


def _fit_regression_full(
    z_train: np.ndarray,
    target: np.ndarray,
    z_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    valid = np.isfinite(target)
    regressor = Ridge(alpha=float(alpha))
    regressor.fit(z_train[valid], target[valid])
    return np.clip(regressor.predict(z_eval), 0.0, 1.0)


def _training_ious(train_generation_records: Sequence[Mapping]) -> np.ndarray:
    ious = []
    missing = []
    for row_idx, generation in enumerate(train_generation_records):
        answer = generation.get("most_likely_answer", {})
        grounding_eval = answer.get("grounding_eval")
        if grounding_eval is None:
            grounding_eval = generation.get("grounding_eval")
        value = None if grounding_eval is None else grounding_eval.get("iou")
        if value is None or not np.isfinite(float(value)):
            missing.append(row_idx)
            ious.append(np.nan)
        else:
            ious.append(float(np.clip(float(value), 0.0, 1.0)))
    if missing:
        raise ValueError(
            "probe_sep_v2 requires a finite grounding IoU for every training "
            f"sample; missing at rows {missing[:10]} (total={len(missing)}).")
    return np.asarray(ious, dtype=np.float32)


def _cv_splits(
    target: np.ndarray,
    requested_folds: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(target.astype(np.int64), minlength=2)
    positive_counts = counts[counts > 0]
    if len(positive_counts) < 2:
        raise ValueError("probe_sep_v2 needs both correct and incorrect training samples.")
    n_splits = min(int(requested_folds), int(positive_counts.min()))
    if n_splits < 2:
        raise ValueError("probe_sep_v2 needs at least two samples in each class.")
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_seed),
    )
    dummy = np.zeros(len(target), dtype=np.float32)
    return list(splitter.split(dummy, target))


def fit_sep_v2_and_score(
    train_features: List[Dict],
    train_is_false: List[float],
    train_generation_records: List[Mapping],
    eval_features: List[Dict],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    pca_dims: Sequence[int] = (32, 64, 128),
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    cv_folds: int = 5,
    random_seed: int = 10,
    include_oof_scores: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit the multi-view IoU-aware OOF stack and score evaluation samples."""
    if len(train_features) != len(train_is_false):
        raise ValueError("Training features and labels have inconsistent lengths.")
    if len(train_generation_records) != len(train_is_false):
        raise ValueError("Training generation records and labels have inconsistent lengths.")
    if not feature_names:
        raise ValueError("probe_sep_v2 needs at least one feature view.")

    y_error = np.asarray(train_is_false, dtype=np.int64)
    ious = _training_ious(train_generation_records)
    targets = {
        "error": y_error,
        "iou_lt_0p25": (ious < 0.25).astype(np.int64),
        "iou_lt_0p5": (ious < 0.5).astype(np.int64),
        "iou_lt_0p75": (ious < 0.75).astype(np.int64),
    }
    iou_risk = 1.0 - ious
    splits = _cv_splits(y_error, cv_folds, random_seed)
    pca_dims = sorted({int(dim) for dim in pca_dims if int(dim) > 0})
    c_values = sorted({float(value) for value in c_values if float(value) > 0.0})
    ridge_alphas = sorted({float(value) for value in ridge_alphas if float(value) > 0.0})
    if not pca_dims or not c_values or not ridge_alphas:
        raise ValueError("PCA dimensions, C values, and Ridge alphas must be positive.")

    stack_train_columns = []
    stack_eval_columns = []
    stack_names = []
    view_metadata: Dict[str, object] = {}

    for view_idx, feature_name in enumerate(feature_names):
        try:
            x_train, x_eval, train_missing, eval_missing = _feature_matrices(
                train_features, eval_features, feature_name)
        except ValueError as exc:
            logging.warning("Skipping probe_sep_v2 view `%s`: %s", feature_name, exc)
            view_metadata[feature_name] = {"status": "skipped", "reason": str(exc)}
            continue

        max_components = min(
            x_train.shape[1],
            min(len(fit_idx) for fit_idx, _ in splits),
        )
        valid_dims = [dim for dim in pca_dims if dim <= max_components]
        if not valid_dims:
            valid_dims = [max_components]

        max_dim = max(valid_dims)
        z_train_max, z_eval_max, projection_metadata = _fit_projection(
            x_train,
            x_eval,
            max_dim,
            random_seed + view_idx * 1000,
        )
        dim_caches = _projection_caches(
            z_train_max,
            splits,
            valid_dims,
        )
        dimension_results = []
        for dim in valid_dims:
            selected_c, _, auc = _binary_head_cv(
                dim_caches[dim], y_error, c_values)
            dimension_results.append({
                "pca_dim": int(dim),
                "selected_c": float(selected_c),
                "error_oof_auroc": float(auc),
            })

        best_dimension = max(
            dimension_results,
            key=lambda item: (item["error_oof_auroc"], -item["pca_dim"]),
        )
        selected_dim = int(best_dimension["pca_dim"])
        selected_cache = dim_caches[selected_dim]
        z_train = z_train_max[:, :selected_dim]
        z_eval = z_eval_max[:, :selected_dim]
        explained_variance_ratios = projection_metadata.pop(
            "explained_variance_ratios")
        projection_metadata.update({
            "pca_dim": selected_dim,
            "explained_variance_ratio_sum": float(
                sum(explained_variance_ratios[:selected_dim])),
            "pca_fit_scope": "all_train_features_without_labels",
        })

        head_metadata = {}
        for target_name, target in targets.items():
            if np.unique(target).size < 2:
                head_metadata[target_name] = {
                    "status": "skipped",
                    "reason": "target has one class",
                }
                continue
            selected_c, oof, auc = _binary_head_cv(
                selected_cache, target, c_values)
            eval_scores, coefficients = _fit_binary_full(
                z_train, target, z_eval, selected_c)
            stack_train_columns.append(oof)
            stack_eval_columns.append(eval_scores)
            stack_names.append(f"{feature_name}__{target_name}")
            head_metadata[target_name] = {
                "status": "completed",
                "selected_c": float(selected_c),
                "oof_auroc": float(auc),
                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            }

        selected_alpha, risk_oof, risk_mse = _regression_head_cv(
            selected_cache, iou_risk, ridge_alphas)
        risk_eval = _fit_regression_full(
            z_train, iou_risk, z_eval, selected_alpha)
        stack_train_columns.append(risk_oof)
        stack_eval_columns.append(risk_eval)
        stack_names.append(f"{feature_name}__iou_risk")
        head_metadata["iou_risk"] = {
            "status": "completed",
            "selected_alpha": float(selected_alpha),
            "oof_mse": float(risk_mse),
        }

        if train_missing.any() or eval_missing.any():
            stack_train_columns.append(train_missing)
            stack_eval_columns.append(eval_missing)
            stack_names.append(f"{feature_name}__missing")

        view_metadata[feature_name] = {
            "status": "completed",
            "train_missing": int(train_missing.sum()),
            "eval_missing": int(eval_missing.sum()),
            "dimension_cv": dimension_results,
            **projection_metadata,
            "heads": head_metadata,
        }
        logging.info(
            "probe_sep_v2 view `%s`: selected PCA=%d; primary OOF AUROC=%.6f.",
            feature_name,
            selected_dim,
            float(head_metadata["error"]["oof_auroc"]),
        )

    if not stack_train_columns:
        raise ValueError("No probe_sep_v2 feature view could be fitted.")

    stack_train = np.column_stack(stack_train_columns).astype(np.float32)
    stack_eval = np.column_stack(stack_eval_columns).astype(np.float32)
    # Select meta regularization using only out-of-fold base predictions.
    meta_cache = []
    for fit_idx, holdout_idx in splits:
        fold_scaler = StandardScaler()
        z_fit = fold_scaler.fit_transform(stack_train[fit_idx])
        z_holdout = fold_scaler.transform(stack_train[holdout_idx])
        meta_cache.append((fit_idx, holdout_idx, z_fit, z_holdout))
    meta_c, meta_oof, meta_auc = _binary_head_cv(meta_cache, y_error, c_values)
    meta_scaler = StandardScaler()
    stack_train_s = meta_scaler.fit_transform(stack_train)
    stack_eval_s = meta_scaler.transform(stack_eval)
    meta_eval, meta_coef = _fit_binary_full(
        stack_train_s, y_error, stack_eval_s, meta_c)
    top_indices = np.argsort(np.abs(meta_coef))[::-1][:20]

    metadata = {
        "version": "probe_sep_v2_multiview_iou_oof_v1",
        "train_samples": int(len(train_features)),
        "eval_samples": int(len(eval_features)),
        "cv_folds": int(len(splits)),
        "feature_names_requested": list(feature_names),
        "pca_dims": pca_dims,
        "c_values": c_values,
        "ridge_alphas": ridge_alphas,
        "supervision": [
            "binary_error",
            "continuous_1_minus_iou",
            "iou_lt_0.25",
            "iou_lt_0.5",
            "iou_lt_0.75",
        ],
        "views": view_metadata,
        "stack_feature_names": stack_names,
        "stack_dim": int(stack_train.shape[1]),
        "meta_selected_c": float(meta_c),
        "meta_oof_auroc": float(meta_auc),
        "meta_top_abs_coef_features": [stack_names[idx] for idx in top_indices],
        "meta_top_abs_coef_values": meta_coef[top_indices].tolist(),
        "meta_oof_mean": float(meta_oof.mean()),
    }
    if include_oof_scores:
        # Used by the evidence-fusion ablation suite. Keeping this opt-in avoids
        # enlarging the standard SEP-v2 metadata payload.
        metadata["meta_oof_scores"] = meta_oof.tolist()
    logging.info(
        "probe_sep_v2 fitted with %d stacked signals; meta OOF AUROC=%.6f.",
        stack_train.shape[1],
        meta_auc,
    )
    return np.asarray(meta_eval, dtype=np.float32), metadata
